"""Build results bundles from a fitted :class:`~arachne.forward_model.pipeline.ForwardModel`.

This is the only module of :mod:`arachne.results` that imports JAX: it evaluates the forward
model on a thinned set of posterior draws.  Every product (convolved and intrinsic cubes,
parameter maps, scalars, nuisances, components) is computed from the **same** draws, so they
are mutually consistent; the reader, plots and viewer never need the emulator again.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from arachne.inference.posterior_predictive import _chunked_map, is_multiresolution

from .io import write_content
from .metadata import pivot_um as lookup_pivot_um
from .provenance import collect_provenance
from .regions import Region, detection_region
from .schema import BundleContent, ComponentContent, TruthContent

__all__ = ["content_from_forward_model", "write_bundle", "thin_indices"]

_SHAPE_COLUMNS = ["mu_y", "mu_x", "size_y", "size_x", "rho", "sersic_n"]


def thin_indices(n_total: int, n_draws: int) -> np.ndarray:
    """Evenly spaced indices ``unique(round(linspace(0, N-1, S)))``.

    Args:
        n_total: Number of available samples N.
        n_draws: Requested number of draws S (``<= 0`` gives none).

    Returns:
        Sorted unique int64 indices, at most ``min(S, N)`` of them.
    """
    if n_total <= 0 or n_draws <= 0:
        return np.zeros(0, dtype=np.int64)
    s = min(int(n_draws), int(n_total))
    return np.unique(np.round(np.linspace(0, n_total - 1, s)).astype(np.int64))


def _log10_sum_pow10(m: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    mmax = np.max(m, axis=axis, keepdims=True)
    return np.squeeze(np.log10(np.sum(10.0 ** (m - mmax), axis=axis, keepdims=True)) + mmax, axis)


class _Adapter:
    """How to turn a spatial theta into parameter maps, scalars and components."""

    kind = "generic"
    map_names: list[str] = []
    scalar_names: list[str] = []
    mass_param = ""

    def maps(self, ts: jnp.ndarray) -> jnp.ndarray | None:  # (Nm, H, W)
        return None

    def scalars(self, ts: jnp.ndarray) -> jnp.ndarray | None:  # (Ns,) before derived ones
        return None


class _FieldAdapter(_Adapter):
    """CorrelatedFieldMap: maps = field_maps; scalars = globals, exp(hyper), couplings."""

    kind = "field"

    def __init__(self, sm: Any):
        self.sm = sm
        self.map_names = list(sm.field_names)
        self.mass_param = sm.mass_param
        hyper = [n.replace("log_", "", 1) for n in getattr(sm, "hyper_names", [])]
        self._raw_scalar_names = (
            list(sm.global_names) + hyper + list(getattr(sm, "coupling_names", []))
        )
        self.scalar_names = [*self._raw_scalar_names, "total_log_mass"]

    def maps(self, ts):
        return self.sm.field_maps(ts)

    def scalars(self, ts):
        parts = [jnp.atleast_1d(self.sm.global_values(ts))]
        if getattr(self.sm, "n_hyper", 0):
            parts.append(jnp.exp(self.sm.hyper_theta(ts)))
        if getattr(self.sm, "n_coupling", 0):
            parts.append(self.sm.coupling_theta(ts))
        return jnp.concatenate([p.astype(parts[0].dtype) for p in parts])


class _AdditiveAdapter(_Adapter):
    """AdditiveComponentModel: no maps; components; scalar total_log_mass."""

    kind = "additive"

    def __init__(self, sm: Any, emulator: Any):
        self.sm = sm
        self.emulator = emulator
        self.scalar_names = ["total_log_mass"]
        names = list(sm.emulator_param_names)
        self.mass_col = names.index(sm.mass_param)
        self.param_names = _SHAPE_COLUMNS + names
        self.profile_names = [p.name for p in sm.profile_objects]

    def component_rows(self, ts):
        """(seds (K, B), params (K, Pc)) for one spatial theta."""
        mu, sigma, rho, sps = self.sm.component_params(ts)
        shapes = self.sm.component_shapes(ts)
        n = jnp.stack(
            [jnp.reshape(s["n"], ()) if "n" in s else jnp.asarray(jnp.nan) for s in shapes]
        ).astype(mu.dtype)
        params = jnp.concatenate(
            [mu, sigma, rho[:, None], n[:, None], sps.astype(mu.dtype)], axis=1
        )
        seds = self.emulator.predict(sps)
        return seds, params


class _DecodeAdapter(_Adapter):
    """Fallback: ``decode`` rows (H*W, N_emulator) reshaped into maps."""

    kind = "decode"

    def __init__(self, sm: Any, names: list[str], shape: tuple[int, int]):
        self.sm = sm
        self.shape = shape
        self.map_names = list(names)
        self.mass_param = getattr(sm, "mass_param", "") or ""
        if self.mass_param not in self.map_names:
            self.mass_param = ""

    def maps(self, ts):
        h, w = self.shape
        rows = self.sm.decode(ts, (h, w))
        return jnp.reshape(rows.T, (rows.shape[1], h, w))


def _adapter(fm: Any) -> _Adapter:
    sm = fm.spatial_model
    shape = tuple(fm.observation.image_shape)
    if hasattr(sm, "field_maps"):
        return _FieldAdapter(sm)
    if hasattr(sm, "component_params"):
        return _AdditiveAdapter(sm, fm.emulator)
    names = list(getattr(sm, "emulator_param_names", None) or fm.emulator.param_names)
    adapter = _DecodeAdapter(sm, names, shape)
    try:
        probe = jax.eval_shape(adapter.maps, jnp.zeros(sm.n_params))
        if probe.shape != (len(names), *shape):
            raise ValueError(f"decode gives {probe.shape}")
    except Exception as err:  # noqa: BLE001 - best effort
        warnings.warn(f"{type(sm).__name__}: no parameter maps in the bundle ({err})", stacklevel=3)
        return _Adapter()
    warnings.warn(
        f"{type(sm).__name__}: parameter maps taken from decode() (best effort)", stacklevel=3
    )
    return adapter


def _per_draw_fn(fm: Any, adapter: _Adapter, store_intrinsic: bool):
    """Build theta -> dict of every per-draw product (one compiled program)."""
    h, w = fm.observation.image_shape
    nuis = fm.nuisance if fm.nuisance is not None and fm.nuisance.n_params > 0 else None

    def fn(theta):
        ts, tn = fm.split_theta(theta)
        out = {"convolved": fm._model_image(theta)}
        if store_intrinsic:
            out["intrinsic"] = fm.spatial_model.model_image(ts, fm.emulator, (h, w))
        maps = adapter.maps(ts)
        if maps is not None:
            out["maps"] = maps
        scal = adapter.scalars(ts)
        if scal is not None:
            out["scalars"] = scal
        if isinstance(adapter, _AdditiveAdapter):
            out["seds"], out["params"] = adapter.component_rows(ts)
        if nuis is not None:
            out.update({f"nuisance_{k}": v for k, v in nuis.split(tn).items()})
        return out

    return fn


def _component_cube_fn(fm: Any, adapter: _AdditiveAdapter):
    """Map theta to (profiles (K, H, W), convolved components (K, B, H, W), no sky)."""
    nuis = fm.nuisance if fm.nuisance is not None and fm.nuisance.n_params > 0 else None

    def fn(theta):
        ts, tn = fm.split_theta(theta)
        prof = adapter.sm.profiles(ts)
        seds = adapter.sm.component_seds(ts, fm.emulator)
        comps = seds[:, :, None, None] * prof[:, None]
        shifts = None
        if nuis is not None and nuis.fit_shifts:
            shifts = nuis.split(tn)["shifts"]
        conv = jax.vmap(lambda c: fm.convolver(c, shifts=shifts))(comps)
        return prof, conv

    return fn


def _np_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(np.asarray, tree)


def _evaluate(fn, thetas: jnp.ndarray, chunk: int) -> dict[str, np.ndarray]:
    if thetas.shape[0] == 0:
        return {}
    return _np_tree(_chunked_map(fn, thetas, chunk))


def content_from_forward_model(
    forward_model: Any,
    *,
    theta_map: ArrayLike | None,
    samples: ArrayLike | None = None,
    posterior_source: str = "map",
    n_draws: int = 200,
    chain_ids: ArrayLike | None = None,
    n_chains: int | None = None,
    store_intrinsic: bool = True,
    store_theta: bool = True,
    component_draws: int = 50,
    order_components: bool = True,
    truth: TruthContent | None = None,
    prior_std_maps: ArrayLike | None = None,
    regions: Mapping[str, Region | np.ndarray] | None = None,
    default_regions: bool = True,
    metrics: Mapping | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    pivot_um: Sequence[float] | None = None,
    labels: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    title: str = "",
    command_line: str | None = None,
    extra: Mapping[str, ArrayLike] | None = None,
    psf: ArrayLike | None = None,
    chunk: int = 16,
) -> BundleContent:
    """Evaluate a fitted forward model on thinned draws and collect a bundle's content.

    Args:
        forward_model: A single-grid ``ForwardModel``.
        theta_map: Best point (full theta).  ``None`` with ``samples`` uses the draw with the
            highest ``log_posterior`` among the thinned draws (``best_point="best_draw"``).
        samples: Posterior samples (N, P), chains concatenated, or None (MAP only).
        posterior_source: ``"nuts"``, ``"laplace"``, ``"nss"``, ``"mclmc"`` or ``"map"``
            (forced to ``"map"`` when there are no samples).
        n_draws: Number of evenly thinned draws S to store.
        chain_ids: (N,) chain of every sample; otherwise derived from ``n_chains`` (equal
            consecutive blocks) or 0.
        n_chains: Number of chains (for ``chain_id`` and the posterior attrs).
        store_intrinsic: Store the unconvolved model cubes.
        store_theta: Store the thinned theta draws.
        component_draws: Draws (subset of S) with full component image cubes (additive).
        order_components: Sort additive components compact-first in every draw.
        truth: Ground truth for mocks, in the bundle's map order.
        prior_std_maps: (Nm, H, W) prior std of the maps (enables ``information``).
        regions: Named regions (Region or (H, W) masks); added to the defaults.
        default_regions: Add ``"all"`` (pixels valid in any band) and ``"segment"``
            (:func:`detection_region`, when something is detected).
        metrics: JSON-serialisable run metrics.
        diagnostics: Sampler diagnostics: scalars and per-parameter arrays (``rhat``, ``ess``),
            optionally ``log_evidence_laplace``.
        pivot_um: (B,) pivots in micron (default :func:`arachne.results.metadata.pivot_um`).
        labels: LaTeX labels per parameter name.
        units: Units per parameter name.
        title: Bundle title.
        command_line: Overrides the recorded command line.
        extra: Free-form arrays for ``/extra``.
        psf: (B, h, w) PSF kernels to store.
        chunk: Draws per vmapped evaluation.

    Returns:
        The validated :class:`BundleContent`.

    Raises:
        NotImplementedError: For multi-resolution forward models (``grid="per_band"``).
        ValueError: Without ``theta_map`` and ``samples``.
    """
    fm = forward_model
    if is_multiresolution(fm):
        raise NotImplementedError("grid='per_band' reserved in schema 1.0")
    sm = fm.spatial_model
    obs = fm.observation
    flux = np.asarray(obs.flux, dtype=np.float32)
    variance = np.asarray(obs.variance, dtype=np.float32)
    mask = np.asarray(obs.mask, dtype=np.float32)
    nb, h, w = flux.shape
    band_names = [str(b) for b in obs.band_names]
    mef = np.broadcast_to(
        np.asarray(getattr(fm.likelihood, "model_error_frac", 0.0), dtype=np.float64), (nb,)
    ).copy()
    adapter = _adapter(fm)
    provenance = collect_provenance()
    if command_line is not None:
        provenance["command_line"] = command_line

    # ---- thinning
    if theta_map is None and samples is None:
        raise ValueError("need theta_map or samples")
    samples_np = None if samples is None else np.asarray(samples)
    n_total = 0 if samples_np is None else int(samples_np.shape[0])
    idx = thin_indices(n_total, n_draws)
    s = int(idx.size)
    if s == 0:
        posterior_source = "map"
    dtype = jnp.asarray(samples_np if samples_np is not None else theta_map).dtype
    thetas = jnp.asarray(samples_np[idx], dtype) if s else jnp.zeros((0, fm.n_params), dtype)
    if isinstance(adapter, _AdditiveAdapter) and order_components:
        if s:
            thetas = _chunked_map(sm.order_components_by_size, thetas, chunk)
        if theta_map is not None:
            theta_map = sm.order_components_by_size(jnp.asarray(theta_map, dtype))
    if chain_ids is not None:
        chain_id = np.asarray(chain_ids)[idx]
    elif n_chains and n_total:
        chain_id = idx // max(1, n_total // int(n_chains))
    else:
        chain_id = np.zeros(s, dtype=np.int64)
    if theta_map is None:
        if s == 0:
            raise ValueError("samples is empty and theta_map is None")
        logp = np.asarray(_chunked_map(fm.log_posterior, thetas, chunk))
        theta_map = thetas[int(np.nanargmax(logp))]
        provenance["best_point"] = "best_draw"
    theta_map = jnp.asarray(theta_map, dtype)

    # ---- evaluation (best point and draws through the same program)
    fn = _per_draw_fn(fm, adapter, store_intrinsic)
    best = _np_tree(jax.jit(fn)(theta_map))
    draws = _evaluate(fn, thetas, chunk)

    def d(key: str) -> np.ndarray | None:
        return draws.get(key) if s else None

    nuisance = None
    if fm.nuisance is not None and fm.nuisance.n_params > 0:
        nuisance = {}
        for key, flag in (
            ("sky", "fit_sky"),
            ("shifts", "fit_shifts"),
            ("log_noise_scale", "fit_noise_scale"),
        ):
            if getattr(fm.nuisance, flag):
                nuisance[f"{key}_map"] = best[f"nuisance_{key}"]
                if s:
                    nuisance[key] = draws[f"nuisance_{key}"]

    # ---- parameters
    map_names = list(adapter.map_names)
    maps_map = best.get("maps")
    maps_draws = d("maps")
    scalar_names = list(adapter.scalar_names)
    scalars_map = scalars_draws = None
    if isinstance(adapter, _FieldAdapter):
        mi = map_names.index(adapter.mass_param)
        scalars_map = np.append(best["scalars"], _log10_sum_pow10(maps_map[mi], (-2, -1)))
        if s:
            tot = _log10_sum_pow10(maps_draws[:, mi], (-2, -1))
            scalars_draws = np.concatenate([draws["scalars"], tot[:, None]], axis=1)

    components = None
    if isinstance(adapter, _AdditiveAdapter):
        mc = adapter.mass_col
        scalars_map = np.atleast_1d(_log10_sum_pow10(best["params"][:, 6 + mc], (0,)))
        if s:
            scalars_draws = _log10_sum_pow10(draws["params"][:, :, 6 + mc], (1,))[:, None]
        cube_fn = _component_cube_fn(fm, adapter)
        prof_map, conv_map = _np_tree(jax.jit(cube_fn)(theta_map))
        cidx = thin_indices(s, component_draws)
        prof_draws = conv_draws = None
        if cidx.size:
            prof_draws, conv_draws = _np_tree(_chunked_map(cube_fn, thetas[cidx], chunk))
        components = ComponentContent(
            profile_names=adapter.profile_names,
            param_names=adapter.param_names,
            factorised=True,
            sed_intrinsic_map=best["seds"],
            sed_intrinsic_draws=d("seds"),
            profiles_map=prof_map,
            profiles_draws=prof_draws,
            intrinsic_map=None,
            intrinsic_draws=None,
            convolved_map=conv_map,
            convolved_draws=conv_draws,
            draw_index=cidx if cidx.size else None,
            params_map=best["params"],
            params_draws=d("params"),
            ordering="size" if order_components else "none",
        )

    # ---- regions
    all_regions: dict[str, Region] = {}
    if default_regions:
        all_regions["all"] = Region.from_mask((mask > 0).any(axis=0), name="all")
        seg = detection_region(flux, variance, mask)
        if seg.any():
            all_regions["segment"] = Region.from_mask(seg, name="segment", kind="mask")
    for name, reg in (regions or {}).items():
        all_regions[name] = (
            reg if isinstance(reg, Region) else Region.from_mask(np.asarray(reg), name=name)
        )

    diag = dict(diagnostics or {})
    diag.setdefault("n_samples_total", n_total)
    if n_chains is not None:
        diag.setdefault("n_chains", int(n_chains))
    fixed = {k: float(v) for k, v in (getattr(sm, "fixed_params", None) or {}).items()}

    content = BundleContent(
        band_names=band_names,
        pivot_um=(
            np.asarray(pivot_um, dtype=float)
            if pivot_um is not None
            else lookup_pivot_um(band_names)
        ),
        pixel_scale=float(obs.pixel_scale) if obs.pixel_scale is not None else float("nan"),
        flux=flux,
        variance=variance,
        mask=mask,
        model_error_frac=mef,
        convolved_map=best["convolved"],
        convolved_draws=d("convolved"),
        intrinsic_map=best.get("intrinsic"),
        intrinsic_draws=d("intrinsic"),
        map_names=map_names,
        param_maps_map=maps_map,
        param_maps_draws=maps_draws,
        prior_std_maps=None if prior_std_maps is None else np.asarray(prior_std_maps),
        scalar_names=scalar_names if scalars_map is not None else [],
        scalars_map=scalars_map,
        scalars_draws=scalars_draws,
        fixed_params=fixed,
        mass_param=adapter.mass_param,
        labels=dict(labels or {}),
        units=dict(units or {}),
        nuisance=nuisance,
        components=components,
        truth=truth,
        theta_map=np.asarray(theta_map),
        theta_draws=np.asarray(thetas) if (store_theta and s) else None,
        draw_index=idx if s else None,
        chain_id=chain_id if s else None,
        diagnostics=diag,
        regions=all_regions,
        metrics=None if metrics is None else dict(metrics),
        extra={k: np.asarray(v) for k, v in (extra or {}).items()},
        posterior_source=posterior_source,
        spatial_model=type(sm).__name__,
        forward_model=type(fm).__name__,
        redshift=float(fixed.get("redshift", float("nan"))),
        title=title,
        degraded=False,
        provenance=provenance,
        psf=None if psf is None else np.asarray(psf),
    )
    content.validate()
    return content


def write_bundle(
    path: str | Path,
    forward_model: Any,
    *,
    mantissa_bits: int | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> Path:
    """Evaluate ``forward_model`` and write a results bundle.

    Equivalent to ``write_content(path, content_from_forward_model(forward_model, **kwargs))``.

    Args:
        path: Output ``.h5`` path.
        forward_model: Fitted single-grid ``ForwardModel``.
        mantissa_bits: Optional lossy rounding of the draws (see :func:`write_content`).
        overwrite: Replace an existing file.
        **kwargs: Passed to :func:`content_from_forward_model` (``theta_map`` is required).

    Returns:
        The output path.
    """
    content = content_from_forward_model(forward_model, **kwargs)
    return write_content(path, content, mantissa_bits=mantissa_bits, overwrite=overwrite)
