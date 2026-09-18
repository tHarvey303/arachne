"""Blind (truth-free) initialisation and MAP finding for additive-component models.

The posterior of an :class:`~arachne.spatial.additive.AdditiveComponentModel`
is strongly multimodal in the label / size / mass directions, so gradient
samplers started from ``theta = 0`` frequently land in the wrong basin.  This
module provides a cheap, deterministic pipeline that reaches the right basin
using only the data:

1. :func:`image_moments` — centroid and rms size from the S/N-stacked image.
2. :func:`blind_initial_theta` — every component at the moment centroid, with
   sizes on a geometric ladder around the moment size and *neutral* SPS
   values (no information about the true galaxy is used).
3. :func:`solve_component_masses` — the K stellar masses enter the model
   linearly (flux ∝ 10**log_mass) so, given every other parameter, they are
   the solution of a K×K weighted least-squares system.  This removes the
   dominant degeneracy before any gradient step.  It is *nuisance aware*: the
   fitted sky is subtracted from the data, the fitted shifts are applied to the
   templates, and the fitted noise scale enters the weights.
4. :func:`find_map` — a few rounds of Adam on ``-log_posterior`` with the
   linear mass solve repeated between rounds, then a low-learning-rate polish.
5. :func:`multistart_map` — :func:`find_map` from several SPS *archetypes*
   (e.g. dusty vs dust-free); return the best.

Steps 3-5 take the **full** theta (spatial parameters followed by the nuisance
block); :func:`blind_initial_full_theta` produces one directly from a forward
model.  Everything here supports both
:class:`~arachne.forward_model.pipeline.ForwardModel` and the optional
:class:`~arachne.forward_model.multires.MultiResolutionForwardModel`, whose
bands live on different pixel grids (see ``ref_band`` in
:func:`blind_initial_theta`).

Everything is float32 and runs in a few tens of seconds on CPU for the
32-parameter bulge+disk demo problem.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax

from arachne.data.multires import MultiResolutionObservation
from arachne.forward_model.multires import MultiResolutionForwardModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.spatial.additive import AdditiveComponentModel
from arachne.spatial.profiles import SersicProfile
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

#: Either flavour of forward model; both expose the same ``log_*`` interface.
AnyForwardModel = Union[ForwardModel, MultiResolutionForwardModel]

_BOUND_MARGIN = 0.01  # fraction of (hi - lo) kept clear of each bound before logit
_MASS_REF = 9.0  # templates in solve_component_masses are per 10**_MASS_REF Msun

# Neutral physical values for commonly used emulator parameter names.  Anything
# not listed here (and not matched by ``_neutral_default``) starts at the
# midpoint of its bounds.
_NEUTRAL_DEFAULTS: dict[str, float] = {
    "Av": 0.5,
    "log10metallicity": -2.5,
    "log_mass": 9.5,
    "fesc_lya": 0.2,
    "dust_bump_amplitude": 1.0,
    "slope": 0.0,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _neutral_default(name: str, bounds: tuple[float, float], mass_param: str) -> float:
    """Neutral physical value for parameter ``name`` (see :func:`blind_initial_theta`)."""
    if name.startswith("logsfr_ratio"):
        return 0.0
    if name == mass_param:
        return _NEUTRAL_DEFAULTS["log_mass"]
    if name in _NEUTRAL_DEFAULTS:
        return _NEUTRAL_DEFAULTS[name]
    lo, hi = bounds
    return 0.5 * (float(lo) + float(hi))


def _logit_in_bounds(value: float, lo: float, hi: float, margin: float = _BOUND_MARGIN) -> float:
    """Map a physical value to its unconstrained raw, clipped strictly inside (lo, hi)."""
    lo, hi = float(lo), float(hi)
    width = hi - lo
    v = min(max(float(value), lo + margin * width), hi - margin * width)
    u = (v - lo) / width
    return math.log(u) - math.log1p(-u)


def _require_additive(spatial_model: Any, where: str) -> AdditiveComponentModel:
    if not isinstance(spatial_model, AdditiveComponentModel):
        raise TypeError(
            f"{where} requires an AdditiveComponentModel spatial model, "
            f"got {type(spatial_model).__name__}"
        )
    return spatial_model


def _emulator_mass_column(model: AdditiveComponentModel) -> int:
    """Column of ``mass_param`` in the full emulator input (``component_params`` rows)."""
    return model.emulator_param_names.index(model.mass_param)


# ---------------------------------------------------------------------------
# 1. Image moments
# ---------------------------------------------------------------------------


def image_moments(
    flux: Any,
    variance: Any,
    mask: Any | None = None,
) -> tuple[float, float, float]:
    """Centroid and rms size of a source from the S/N-stacked multi-band image.

    The stacked image is ``S(y, x) = Σ_b mask_b flux_b / sqrt(variance_b)``,
    minus its median (sky/background level), clipped at zero.  The centroid
    is the ``S``-weighted mean pixel position and the size is
    ``sqrt(0.5 * (var_y + var_x))`` of the ``S``-weighted second moments.

    No model or truth information is used; this is the seed for
    :func:`blind_initial_theta`.

    Args:
        flux: Array of shape (N_bands, H, W).
        variance: Array of shape (N_bands, H, W); non-positive entries are ignored.
        mask: Optional array of shape (N_bands, H, W); 0/False marks bad pixels.

    Returns:
        ``(cy, cx, sigma_px)`` as Python floats, in pixel (row, col) units.  If
        the stacked image has no positive pixels, the image centre and
        ``max(H, W) / 8`` are returned.
    """
    flux = np.asarray(flux, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if flux.ndim != 3 or flux.shape != variance.shape:
        raise ValueError(
            f"flux and variance must both be (N_bands, H, W); got {flux.shape}, {variance.shape}"
        )
    valid = variance > 0
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_b = np.where(valid, flux / np.sqrt(np.where(valid, variance, 1.0)), 0.0)
    snr = snr_b.sum(axis=0)
    snr = np.clip(snr - np.median(snr), 0.0, None)

    _, H, W = flux.shape
    wsum = snr.sum()
    if not np.isfinite(wsum) or wsum <= 0:
        return (H - 1) / 2.0, (W - 1) / 2.0, max(H, W) / 8.0

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cy = float((snr * yy).sum() / wsum)
    cx = float((snr * xx).sum() / wsum)
    var_y = float((snr * (yy - cy) ** 2).sum() / wsum)
    var_x = float((snr * (xx - cx) ** 2).sum() / wsum)
    sigma = math.sqrt(max(0.5 * (var_y + var_x), 1e-6))
    return cy, cx, sigma


# ---------------------------------------------------------------------------
# 2. Blind initial theta
# ---------------------------------------------------------------------------


def _shape_init_values(
    profile: Any, mu_y: float, mu_x: float, log_size: float, log_n: float
) -> list[float]:
    """Initial raw shape values for one component, driven by ``shape_param_names``.

    ``mu_y`` / ``mu_x`` go to the centre entries, every ``log_sigma_*`` entry to
    ``log_size``, ``atanh_rho`` to 0 (round), ``log_n`` to ``log_n``, and any
    unrecognised shape parameter to 0.

    Args:
        profile: The component's :class:`~arachne.spatial.profiles.Profile`.
        mu_y: Centre row coordinate in the model's units.
        mu_x: Centre column coordinate in the model's units.
        log_size: Log of the initial size in the model's units.
        log_n: Log of the initial Sersic index (ignored by other profiles).

    Returns:
        List of ``profile.n_shape`` floats, in block order.
    """
    values: list[float] = []
    for name in profile.shape_param_names:
        if name == "mu_y":
            values.append(float(mu_y))
        elif name == "mu_x":
            values.append(float(mu_x))
        elif name.startswith("log_sigma"):
            values.append(float(log_size))
        elif name == "log_n":
            values.append(float(log_n))
        else:  # atanh_rho and anything a future profile adds
            values.append(0.0)
    return values


def _default_sersic_indices(model: AdditiveComponentModel, size_scales: Sequence[float]) -> dict:
    """Default Sersic index per component: bulge n=4 for the most compact, n=1 for the rest."""
    sersic_ks = [k for k, p in enumerate(model.profile_objects) if isinstance(p, SersicProfile)]
    if not sersic_ks:
        return {}
    if model.n_components == 1:
        # Nothing to play the bulge against: start at the log_n prior mean.
        k = sersic_ks[0]
        return {k: float(math.exp(model.profile_objects[k].log_n_mu))}
    compact = min(sersic_ks, key=lambda k: size_scales[k])
    return {k: (4.0 if k == compact else 1.0) for k in sersic_ks}


def reference_band_index(obs: MultiResolutionObservation, ref_band: str | int | None = None) -> int:
    """Index of the band a multi-resolution initialiser should take moments on.

    With ``ref_band=None`` the band with the **highest total S/N** is chosen:
    ``Σ max(flux / sqrt(variance), 0)`` over the band's valid pixels.  That is
    the band whose centroid and second moments are best measured; ties (an
    exact draw is essentially impossible on real data) go to the earlier band.
    The finest pixel scale is deliberately *not* the default — a sharp but
    shallow band gives a noisier centroid than a well-detected coarse one.

    Args:
        obs: MultiResolutionObservation.
        ref_band: Band name, band index, or ``None`` for the automatic choice.

    Returns:
        Index of the reference band.

    Raises:
        ValueError: If ``ref_band`` names an absent band or is out of range.
    """
    if ref_band is None:
        scores = []
        for band in obs.bands:
            flux = np.asarray(band.flux, dtype=np.float64)
            variance = np.asarray(band.variance, dtype=np.float64)
            mask = np.asarray(band.mask, dtype=np.float64)
            good = np.isfinite(variance) & (variance > 0) & (mask > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.where(good, flux / np.sqrt(np.where(good, variance, 1.0)), 0.0)
            scores.append(float(np.sum(np.clip(snr, 0.0, None))))
        return int(np.argmax(scores))
    if isinstance(ref_band, str):
        names = obs.band_names
        if ref_band not in names:
            raise ValueError(f"ref_band {ref_band!r} is not one of {names}")
        return names.index(ref_band)
    index = int(ref_band)
    if not 0 <= index < obs.n_bands:
        raise ValueError(f"ref_band index {index} is out of range for {obs.n_bands} bands")
    return index


def _multires_moment_seed(
    model: AdditiveComponentModel,
    obs: MultiResolutionObservation,
    ref_band: str | int | None,
) -> tuple[float, float, float, str]:
    """Moment centroid and size of a multi-resolution observation, in sky arcsec.

    Moments are taken on the reference band's own pixel grid
    (:func:`reference_band_index`) and mapped to the sky frame with that band's
    affine matrix, ``(dy, dx) = affine @ (cy - ref_row, cx - ref_col)``; the
    size is ``sigma_px * pixel_scale_b``.

    Args:
        model: The additive component model (must be in arcsec mode).
        obs: MultiResolutionObservation.
        ref_band: Band name / index / None.

    Returns:
        Tuple ``(mu_y, mu_x, sigma_arcsec, description)``.

    Raises:
        ValueError: If the model is in pixel-index mode.
    """
    if model.pixel_scale is None:
        raise ValueError(
            "blind_initial_theta on a MultiResolutionObservation needs a spatial model in "
            "arcsec mode: build AdditiveComponentModel(..., pixel_scale=<arcsec/px>, "
            "normalisation='analytic')."
        )
    index = reference_band_index(obs, ref_band)
    band = obs.bands[index]
    cy, cx, sigma_px = image_moments(
        np.asarray(band.flux)[None],
        np.asarray(band.variance)[None],
        np.asarray(band.mask)[None],
    )
    affine = np.asarray(band.affine, dtype=np.float64)
    dr = cy - band.ref_pixel[0]
    dc = cx - band.ref_pixel[1]
    mu_y = float(affine[0, 0] * dr + affine[0, 1] * dc)
    mu_x = float(affine[1, 0] * dr + affine[1, 1] * dc)
    sigma_unit = float(sigma_px * band.pixel_scale)
    desc = (
        f"reference band {band.band_name} pixel centroid ({cy:.2f}, {cx:.2f}), "
        f"sigma {sigma_px:.2f} px -> sky ({mu_y:.4f}, {mu_x:.4f}) arcsec, "
        f"sigma {sigma_unit:.4f} arcsec"
    )
    return mu_y, mu_x, sigma_unit, desc


def blind_initial_theta(
    spatial_model: AdditiveComponentModel,
    obs: Any,
    size_scales: Sequence[float] | None = None,
    neutral_values: dict[str, float] | None = None,
    shared_values: dict[str, float] | None = None,
    sersic_n: dict[int, float] | None = None,
    ref_band: str | int | None = None,
) -> jnp.ndarray:
    """Truth-free starting theta for an :class:`AdditiveComponentModel`.

    Every component is placed at the :func:`image_moments` centroid with
    ``rho = 0`` and an isotropic size ``size_scales[k] * sigma_moment``.  By
    default the K sizes form a geometric ladder from ``0.4`` to ``1.5`` times
    the moment size (``K = 2`` gives ``[0.4, 1.5]``, compact first; ``K = 1``
    gives ``[1.0]``), so the components are already ordered by size.  The
    moment centroid and size are in pixels and are converted to the model's
    coordinate units (they are used unchanged in pixel mode; in ``pixel_scale``
    mode the centroid becomes ``(c - (N - 1) / 2) * pixel_scale`` arcsec and the
    size is multiplied by ``pixel_scale``).

    Shape parameters are filled per profile:

    - Gaussian: centre, ``log_sigma`` on both axes, ``atanh_rho = 0``;
    - Sérsic: the same plus ``log_n``, taken from ``sersic_n[k]`` if given,
      otherwise ``n = 4`` for the Sérsic component with the smallest
      ``size_scales`` entry and ``n = 1`` for the other Sérsic components
      (a bulge + disk start); for a single-component model, the profile's
      ``log_n`` prior mean;
    - point source: the centre only.

    SPS raws are the logit of *neutral* physical values:

    - ``neutral_values[name]`` if given (applies to every component);
    - otherwise a built-in default for known names: ``logsfr_ratio_*`` → 0,
      ``Av`` → 0.5, ``log10metallicity`` → -2.5, the mass parameter → 9.5,
      ``fesc_lya`` → 0.2, ``dust_bump_amplitude`` → 1.0, ``slope`` → 0;
    - otherwise the midpoint of the parameter's bounds.

    Shared parameters use ``shared_values`` with the same fall-through.  All
    values are clipped 1% inside their bounds before the logit so no raw is
    infinite.  Masses are only placeholders here — follow with
    :func:`solve_component_masses`.

    Multi-resolution observations
    -----------------------------
    ``obs`` may also be a
    :class:`~arachne.data.multires.MultiResolutionObservation`, in which case
    the model must be in arcsec mode.  Moments are then taken on a single
    *reference* band (``ref_band``; by default the band with the highest total
    S/N — see :func:`reference_band_index`), because the bands have different
    pixel grids and cannot simply be stacked.  The pixel centroid is mapped to
    the **sky frame** (``+y`` North, ``+x`` East) with that band's affine
    matrix, ``(dy, dx) = affine @ (cy - ref_row, cx - ref_col)``, and the size
    is ``sigma_px * pixel_scale_b`` — matching the coordinate convention of
    :class:`~arachne.forward_model.multires.MultiResolutionForwardModel`.  The
    single-grid path is untouched and bit-identical.

    Args:
        spatial_model: The additive component model.
        obs: Either a single-grid cube — anything with ``.flux``, ``.variance``
            and ``.mask`` arrays of shape (N_bands, H, W), e.g. an
            :class:`~arachne.data.observation.ObservationCube` — or a
            :class:`~arachne.data.multires.MultiResolutionObservation` (numpy
            or JAX).  See the multi-resolution note above.
        size_scales: K multipliers of the moment size, one per component.
            Entries for point-source components are ignored.
        neutral_values: Physical overrides for per-component SPS parameters.
        shared_values: Physical overrides for shared SPS parameters.
        sersic_n: ``{component index: Sersic index}`` overriding the default rule.
        ref_band: Multi-resolution only: the band to take moments on, by name
            or index.  ``None`` (default) picks the highest-total-S/N band, see
            :func:`reference_band_index`.  Ignored for a single-grid cube.

    Returns:
        float32 theta of shape ``(spatial_model.n_params,)``.

    Raises:
        TypeError: If ``spatial_model`` is not an AdditiveComponentModel.
        ValueError: If ``size_scales`` has the wrong length, an override names
            an unknown parameter, ``sersic_n`` keys a non-Sersic component,
            ``ref_band`` is unknown, or a multi-resolution observation is
            passed with a pixel-index-mode model.
    """
    model = _require_additive(spatial_model, "blind_initial_theta")
    K = model.n_components
    neutral_values = dict(neutral_values or {})
    shared_values = dict(shared_values or {})

    unknown = set(neutral_values) - set(model.sps_param_names)
    if unknown:
        raise ValueError(f"neutral_values not in sps_param_names: {sorted(unknown)}")
    unknown = set(shared_values) - set(model.shared_param_names)
    if unknown:
        raise ValueError(f"shared_values not in shared_param_names: {sorted(unknown)}")

    if size_scales is None:
        size_scales = [1.0] if K == 1 else list(np.geomspace(0.4, 1.5, K))
    size_scales = [float(s) for s in size_scales]
    if len(size_scales) != K:
        raise ValueError(f"size_scales must have length K={K}, got {len(size_scales)}")

    n_values = _default_sersic_indices(model, size_scales)
    for k, value in dict(sersic_n or {}).items():
        if not isinstance(k, (int, np.integer)) or not 0 <= int(k) < K:
            raise ValueError(f"sersic_n key {k!r} is not a component index in [0, {K})")
        if not isinstance(model.profile_objects[int(k)], SersicProfile):
            raise ValueError(f"sersic_n[{k}] given but component {k} is not a Sersic profile")
        n_values[int(k)] = float(value)

    if isinstance(obs, MultiResolutionObservation):
        mu_y, mu_x, sigma_unit, seed_desc = _multires_moment_seed(model, obs, ref_band)
    else:
        cy, cx, sigma_px = image_moments(obs.flux, obs.variance, getattr(obs, "mask", None))
        H, W = model.image_shape
        if model.pixel_scale is None:
            mu_y, mu_x, sigma_unit = cy, cx, sigma_px
        else:
            ps = model.pixel_scale
            mu_y = (cy - (H - 1) / 2.0) * ps
            mu_x = (cx - (W - 1) / 2.0) * ps
            sigma_unit = sigma_px * ps
        seed_desc = f"moment centroid ({cy:.2f}, {cx:.2f}), sigma {sigma_px:.2f} px"
    logger.info(
        f"blind_initial_theta: {seed_desc}, "
        f"K={K} size scales {np.round(size_scales, 3).tolist()}, "
        f"profiles {[p.name for p in model.profile_objects]}"
    )

    sps_raw = []
    for name in model.sps_param_names:
        lo, hi = model.param_bounds[name]
        value = neutral_values.get(name, _neutral_default(name, (lo, hi), model.mass_param))
        sps_raw.append(_logit_in_bounds(value, lo, hi))

    theta = np.zeros(model.n_params, dtype=np.float32)
    for k, profile in enumerate(model.profile_objects):
        sl = model.component_slices[k]
        n_shape = model.n_shape_params[k]
        log_size = math.log(max(size_scales[k] * sigma_unit, 1e-3))
        log_n = math.log(n_values.get(k, 1.0))
        theta[sl.start : sl.start + n_shape] = _shape_init_values(
            profile, mu_y, mu_x, log_size, log_n
        )
        theta[sl.start + n_shape : sl.stop] = sps_raw

    for j, name in enumerate(model.shared_param_names):
        lo, hi = model.param_bounds[name]
        value = shared_values.get(name, _neutral_default(name, (lo, hi), model.mass_param))
        theta[model.shared_slice.start + j] = _logit_in_bounds(value, lo, hi)

    return jnp.asarray(theta, dtype=jnp.float32)


def blind_initial_full_theta(
    forward_model: AnyForwardModel,
    obs: Any | None = None,
    **kwargs: Any,
) -> jnp.ndarray:
    """Blind starting point for the **full** theta, nuisance block included.

    Shorthand for
    ``forward_model.initial_theta_from_spatial(blind_initial_theta(
    forward_model.spatial_model, obs, **kwargs))`` — the vector every sampler
    and :func:`find_map` expect when a
    :class:`~arachne.forward_model.nuisance.NuisanceModel` is attached.  The
    nuisance parameters start at their prior means (zeros: no sky, no shift,
    nominal noise).

    Works for both :class:`~arachne.forward_model.pipeline.ForwardModel` and
    :class:`~arachne.forward_model.multires.MultiResolutionForwardModel`; with
    the latter, ``obs`` defaults to the model's own
    :class:`~arachne.data.multires.MultiResolutionObservation` and moments are
    taken on the reference band (see ``ref_band`` in
    :func:`blind_initial_theta`).

    Args:
        forward_model: Assembled forward model.
        obs: Observation to take moments on.  ``None`` (default) uses
            ``forward_model.observation``.
        **kwargs: Forwarded to :func:`blind_initial_theta` (``size_scales``,
            ``neutral_values``, ``shared_values``, ``sersic_n``, ``ref_band``).

    Returns:
        float32 theta of shape ``(forward_model.n_params,)``.
    """
    if obs is None:
        obs = forward_model.observation
    theta_spatial = blind_initial_theta(forward_model.spatial_model, obs, **kwargs)
    return jnp.asarray(forward_model.initial_theta_from_spatial(theta_spatial), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# 3. Linear mass solve
# ---------------------------------------------------------------------------


def _split_for_mass_solve(
    forward_model: AnyForwardModel, theta: jnp.ndarray
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray | None]]:
    """Split theta and decode the nuisance blocks, tolerating a spatial-only theta.

    Args:
        forward_model: ForwardModel or MultiResolutionForwardModel.
        theta: Either a full theta (spatial + nuisance) or a spatial-only one.

    Returns:
        Tuple ``(theta_spatial, nuisance_blocks)``; the blocks are all ``None``
        when there is no nuisance model or ``theta`` carries only the spatial
        parameters.
    """
    none_blocks: dict[str, jnp.ndarray | None] = {
        "shifts": None,
        "sky": None,
        "log_noise_scale": None,
    }
    n_spatial = forward_model.spatial_model.n_params
    if theta.shape[0] <= n_spatial:
        return theta, none_blocks
    decode = getattr(forward_model, "_nuisance_blocks", None)
    if decode is None:
        return theta[:n_spatial], none_blocks
    return theta[:n_spatial], decode(theta[n_spatial:])


def _mass_solve_blocks(
    forward_model: AnyForwardModel,
    theta_spatial: jnp.ndarray,
    nuis: dict[str, jnp.ndarray | None],
):
    """Yield ``(templates, data, weights)`` triples for the linear mass solve.

    Each triple is flattened over pixels: ``templates`` is (K, N), ``data`` and
    ``weights`` are (N,).  A single-grid model yields one triple covering every
    band; a multi-resolution model yields one per band (ragged shapes, which is
    why they are summed into the normal equations separately).

    The per-band sky is subtracted from the data, the registration shifts are
    applied when the templates are convolved, and the weights use the same
    effective variance the likelihood does, bar the model-error floor (which
    depends on the model being solved for).

    Args:
        forward_model: ForwardModel or MultiResolutionForwardModel.
        theta_spatial: Spatial parameter vector.
        nuis: Decoded nuisance blocks (``shifts``, ``sky``, ``log_noise_scale``).

    Yields:
        Tuples ``(templates, data, weights)`` of JAX arrays.
    """
    model = forward_model.spatial_model
    shifts, sky, log_noise = nuis["shifts"], nuis["sky"], nuis["log_noise_scale"]
    K = model.n_components

    if hasattr(forward_model, "model_images"):  # multi-resolution
        comp_per_band = forward_model.component_images_per_band(theta_spatial)
        for b, band in enumerate(forward_model.observation.bands):
            convolver = forward_model.convolvers[b]
            shift_b = None if shifts is None else shifts[b : b + 1]
            templates = jax.vmap(lambda ci, c=convolver, s=shift_b: c(ci[None], shifts=s)[0])(
                comp_per_band[b]
            )  # (K, H_b, W_b)
            data = jnp.asarray(band.flux)
            if sky is not None:
                data = data - sky[b]
            variance = jnp.asarray(band.variance)
            if log_noise is not None:
                variance = variance * jnp.exp(2.0 * log_noise[b])
            weights = jnp.asarray(band.mask) / (variance + 1e-30)
            yield templates.reshape(K, -1), data.reshape(-1), weights.reshape(-1)
        return

    comp_images = model.component_images(theta_spatial, forward_model.emulator)
    templates = jax.vmap(lambda ci: forward_model.convolver(ci, shifts=shifts))(comp_images)
    obs = forward_model.observation
    data = jnp.asarray(obs.flux)
    if sky is not None:
        data = data - sky[:, None, None]
    variance = jnp.asarray(obs.variance)
    if log_noise is not None:
        variance = variance * jnp.exp(2.0 * log_noise)[:, None, None]
    weights = jnp.asarray(obs.mask) / (variance + 1e-30)
    yield templates.reshape(K, -1), data.reshape(-1), weights.reshape(-1)


def solve_component_masses(forward_model: AnyForwardModel, theta: jnp.ndarray) -> jnp.ndarray:
    """Replace each component's mass with its weighted linear least-squares value.

    Holding every other parameter at its current value, the PSF-convolved
    model image is ``Σ_k a_k T_k`` where ``T_k`` is component ``k``'s
    convolved image per ``1e9 Msun`` and ``a_k = 10**(logM_k - 9)``.  This
    assumes the emulator flux is exactly linear in ``10**log_mass`` — true
    for :class:`~arachne.emulator.parrot_emulator_v2.ParrotEmulatorV2`
    with mass normalisation (and for any emulator that scales its output by
    the mass) — and a good approximation otherwise, in which case
    :func:`find_map` corrects the residual nonlinearity by gradient descent.

    The templates are obtained from the spatial model's per-component images at
    the *current* masses (any mix of Gaussian, Sérsic and point-source
    profiles), convolved through the model's own PSF, then divided by
    ``10**(logM_k - 9)``; the K×K normal equations ``(T W Tᵀ) a = T W y`` are
    solved (with a negligible relative ridge for numerical safety), ``a`` is
    clipped to ``>= 1e-3`` (crude non-negativity), and ``logM = 9 + log10(a)``
    is clipped 1% inside the mass bounds before being written back as a raw.

    Nuisance parameters
    -------------------
    ``theta`` may be a **full** vector (spatial + nuisance) or a spatial-only
    one; when a nuisance block is present its parameters are *used*, not
    ignored:

    - the per-band **sky** is subtracted from the data before the normal
      equations are formed (otherwise a pedestal is absorbed into the masses);
    - the per-band **registration shifts** are applied when the templates are
      convolved, so template and data are aligned;
    - the per-band **log noise scale** enters the weights
      ``W = mask / (variance · e^{2s})``, the same effective variance the
      likelihood uses.

    The model-error floor ``(f · model)²`` is deliberately *not* included: it
    depends on the very amplitudes being solved for.  Only the mass raws are
    modified; the nuisance block is returned untouched.

    Multi-resolution
    ----------------
    A :class:`~arachne.forward_model.multires.MultiResolutionForwardModel` is
    handled by building the templates band by band through that band's own
    convolver and accumulating every band's contribution into the same K×K
    normal equations — exactly the sum the joint likelihood is.

    Pure JAX; safe under ``jax.jit``.

    Args:
        forward_model: ForwardModel or MultiResolutionForwardModel whose
            spatial model is an AdditiveComponentModel.
        theta: Flat theta of shape (n_params,) or (spatial n_params,).

    Returns:
        theta with the K mass raws replaced, same shape as the input, float32.

    Raises:
        TypeError: If the spatial model is not an AdditiveComponentModel.
    """
    model = _require_additive(forward_model.spatial_model, "solve_component_masses")
    theta = jnp.asarray(theta, dtype=jnp.float32)
    K = model.n_components
    mass_col = _emulator_mass_column(model)
    lo, hi = (float(v) for v in model.param_bounds[model.mass_param])

    theta_spatial, nuis = _split_for_mass_solve(forward_model, theta)
    _, _, _, sps_phys = model.component_params(theta_spatial)
    log_mass = sps_phys[:, mass_col]  # (K,)
    per_mass = (10.0 ** (log_mass - _MASS_REF))[:, None]  # (K, 1)

    normal = jnp.zeros((K, K), dtype=jnp.float32)
    rhs = jnp.zeros((K,), dtype=jnp.float32)
    for templates, data, weights in _mass_solve_blocks(forward_model, theta_spatial, nuis):
        sqrt_w = jnp.sqrt(weights)
        t_w = (templates / per_mass) * sqrt_w[None, :]  # (K, N_data)
        y_w = data * sqrt_w  # (N_data,)
        normal = normal + t_w @ t_w.T
        rhs = rhs + t_w @ y_w

    ridge = 1e-7 * jnp.trace(normal) / K + 1e-30
    amp = jnp.linalg.solve(normal + ridge * jnp.eye(K, dtype=normal.dtype), rhs)
    amp = jnp.clip(amp, 1e-3, None)

    margin = _BOUND_MARGIN * (hi - lo)
    new_log_mass = jnp.clip(_MASS_REF + jnp.log10(amp), lo + margin, hi - margin)
    u = (new_log_mass - lo) / (hi - lo)
    new_raw = jnp.log(u) - jnp.log1p(-u)

    new_raw = new_raw.astype(jnp.float32)
    for k in range(K):
        sps_k = model.sps_raw(theta, k).at[model.mass_index].set(new_raw[k])
        theta = model.set_sps_raw(theta, k, sps_k)
    return theta


# ---------------------------------------------------------------------------
# 4. MAP finding
# ---------------------------------------------------------------------------


@dataclass
class MAPResult:
    """Output of :func:`find_map` / :func:`multistart_map`.

    Attributes:
        theta: MAP estimate (unconstrained), shape (n_params,), float32.
        neg_log_posterior: ``-log_posterior(theta)`` at the returned theta.
        history: Per-step loss (``-log_posterior``) over the whole run, shape (n_steps,).
        n_steps: Total number of Adam steps taken.
    """

    theta: jnp.ndarray
    neg_log_posterior: float
    history: np.ndarray
    n_steps: int


def _make_adam_runner(loss_fn, clip_norm: float):
    """Return ``run(theta, lr, n_steps) -> (theta, losses)``; jitted, static in ``n_steps``."""
    value_and_grad = jax.value_and_grad(loss_fn)

    def _run(theta: jnp.ndarray, lr: jnp.ndarray, n_steps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        opt = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(lr))
        opt_state = opt.init(theta)

        def step(carry, _):
            t, s = carry
            loss, grad = value_and_grad(t)
            updates, s = opt.update(grad, s, t)
            return (optax.apply_updates(t, updates), s), loss

        (theta, _), losses = jax.lax.scan(step, (theta, opt_state), None, length=n_steps)
        return theta, losses

    # ``length`` must be static; wrap so n_steps is hashed into the compile cache.
    return jax.jit(_run, static_argnums=2)


def find_map(
    forward_model: AnyForwardModel,
    theta0: jnp.ndarray,
    n_rounds: int = 4,
    steps_per_round: int = 300,
    lr: float = 0.05,
    final_steps: int = 400,
    final_lr: float = 0.01,
    resolve_masses: bool = True,
    clip_norm: float = 50.0,
    order_by_size: bool = True,
) -> MAPResult:
    """Find a MAP estimate with Adam, re-solving component masses between rounds.

    Minimises ``-forward_model.log_posterior`` with
    ``optax.chain(clip_by_global_norm(clip_norm), adam(lr))`` for ``n_rounds``
    rounds of ``steps_per_round`` steps (fresh optimiser state each round),
    then ``final_steps`` at ``final_lr``.  If the spatial model is an
    :class:`AdditiveComponentModel` and ``resolve_masses`` is true, the K
    masses are replaced by their linear least-squares solution
    (:func:`solve_component_masses`) before the first round and after every
    round except the final polish, which collapses the mass/size/colour
    degeneracy far faster than gradient steps do.  Finally, for additive
    models with ``order_by_size``, components are permuted compact-first
    (``order_components_by_size``) to fix the label symmetry.

    Each round is a single jitted ``lax.scan``; compilation happens once per
    distinct step count (so at most twice).

    ``theta0`` is a **full** theta: spatial parameters followed by the nuisance
    block, exactly what the sampler sees.  When the forward model carries a
    :class:`~arachne.forward_model.nuisance.NuisanceModel`, callers therefore
    pass ``fm.initial_theta_from_spatial(blind_initial_theta(...))`` — or just
    :func:`blind_initial_full_theta`, which does both.  A spatial-only theta is
    still accepted when the model has no nuisance block.  Both
    :class:`~arachne.forward_model.pipeline.ForwardModel` and
    :class:`~arachne.forward_model.multires.MultiResolutionForwardModel` are
    supported.

    Args:
        forward_model: Assembled ForwardModel or MultiResolutionForwardModel.
        theta0: Starting full theta, shape (n_params,) — e.g.
            :func:`blind_initial_full_theta`.
        n_rounds: Number of Adam rounds at ``lr``.
        steps_per_round: Adam steps per round.
        lr: Adam learning rate for the rounds.
        final_steps: Steps in the final polish (0 to skip).
        final_lr: Learning rate of the final polish.
        resolve_masses: Re-solve masses linearly between rounds (additive models only).
        clip_norm: Global gradient-norm clip.
        order_by_size: Order components compact-first at the end (additive models only).

    Returns:
        :class:`MAPResult`.
    """
    theta = jnp.asarray(theta0, dtype=jnp.float32)
    is_additive = isinstance(forward_model.spatial_model, AdditiveComponentModel)
    do_masses = bool(resolve_masses and is_additive)

    def loss_fn(t: jnp.ndarray) -> jnp.ndarray:
        return -forward_model.log_posterior(t)

    loss_jit = jax.jit(loss_fn)
    run_adam = _make_adam_runner(loss_fn, float(clip_norm))
    solve_masses = jax.jit(lambda t: solve_component_masses(forward_model, t))

    history: list[np.ndarray] = []
    if do_masses:
        theta = solve_masses(theta)
    logger.info(
        f"find_map: start -log_post = {float(loss_jit(theta)):.2f} "
        f"({n_rounds} x {steps_per_round} steps @ lr={lr}, then {final_steps} @ {final_lr})"
    )
    for rnd in range(int(n_rounds)):
        if steps_per_round > 0:
            theta, losses = run_adam(theta, jnp.float32(lr), int(steps_per_round))
            history.append(np.asarray(losses, dtype=np.float32))
        if do_masses:
            theta = solve_masses(theta)
        logger.info(
            f"find_map: round {rnd + 1}/{n_rounds} -log_post = {float(loss_jit(theta)):.2f}"
        )

    if final_steps > 0:
        theta, losses = run_adam(theta, jnp.float32(final_lr), int(final_steps))
        history.append(np.asarray(losses, dtype=np.float32))

    if order_by_size and is_additive:
        # ``order_components_by_size`` rebuilds theta from the spatial blocks
        # alone, so the nuisance tail has to be re-attached by hand.
        n_spatial = forward_model.spatial_model.n_params
        ordered = forward_model.spatial_model.order_components_by_size(theta[:n_spatial])
        theta = jnp.concatenate([ordered, theta[n_spatial:]])

    theta = jnp.asarray(theta, dtype=jnp.float32)
    final_loss = float(loss_jit(theta))
    hist = np.concatenate(history) if history else np.zeros(0, dtype=np.float32)
    logger.info(f"find_map: final -log_post = {final_loss:.2f} after {hist.size} steps")
    return MAPResult(
        theta=theta, neg_log_posterior=final_loss, history=hist, n_steps=int(hist.size)
    )


# ---------------------------------------------------------------------------
# 5. Multi-start
# ---------------------------------------------------------------------------


def _apply_archetype(
    model: AdditiveComponentModel, theta: jnp.ndarray, archetype: dict[str, float]
) -> jnp.ndarray:
    """Set per-component (all K) and shared SPS raws from physical ``archetype`` values."""
    shared_raw = theta[model.shared_slice]
    for name, value in archetype.items():
        lo, hi = model.param_bounds.get(name, (None, None))
        if name in model.sps_param_names:
            raw = _logit_in_bounds(value, lo, hi)
            col = model.sps_param_names.index(name)
            for k in range(model.n_components):
                theta = model.set_sps_raw(theta, k, model.sps_raw(theta, k).at[col].set(raw))
        elif name in model.shared_param_names:
            raw = _logit_in_bounds(value, lo, hi)
            shared_raw = shared_raw.at[model.shared_param_names.index(name)].set(raw)
        else:
            raise ValueError(f"archetype parameter {name!r} is not a free SPS parameter")
    return theta.at[model.shared_slice].set(shared_raw)


def multistart_map(
    forward_model: AnyForwardModel,
    theta0: jnp.ndarray,
    archetypes: Sequence[dict[str, float]],
    **find_map_kwargs: Any,
) -> MAPResult:
    """Run :func:`find_map` from ``theta0`` and from each SPS archetype; return the best.

    An archetype is a ``{param_name: physical value}`` dict applied to *all*
    components' per-component SPS raws (shared names are allowed too), e.g.
    ``[{"Av": 0.3}, {"Av": 2.0}]`` to probe the dust/age degeneracy from both
    sides.  ``theta0`` itself is always one of the starts; an archetype that
    leaves ``theta0`` unchanged (e.g. ``{}``) is not run twice.

    Args:
        forward_model: ForwardModel or MultiResolutionForwardModel with an
            AdditiveComponentModel spatial model.
        theta0: Base starting full theta, shape (n_params,).
        archetypes: Physical overrides, one dict per additional start.
        **find_map_kwargs: Forwarded to :func:`find_map`.

    Returns:
        The :class:`MAPResult` with the lowest ``neg_log_posterior``.

    Raises:
        TypeError: If the spatial model is not an AdditiveComponentModel.
        ValueError: If an archetype names a parameter that is not free.
    """
    model = _require_additive(forward_model.spatial_model, "multistart_map")
    theta0 = jnp.asarray(theta0, dtype=jnp.float32)
    starts: list[tuple[str, jnp.ndarray]] = [("theta0", theta0)]
    for i, arch in enumerate(archetypes):
        t = _apply_archetype(model, theta0, dict(arch))
        if np.array_equal(np.asarray(t), np.asarray(theta0)):
            continue
        starts.append((f"archetype {i} {dict(arch)}", t))

    best: MAPResult | None = None
    for label, t in starts:
        result = find_map(forward_model, t, **find_map_kwargs)
        logger.info(f"multistart_map: {label}: -log_post = {result.neg_log_posterior:.2f}")
        if (
            best is None
            or not math.isfinite(best.neg_log_posterior)
            or result.neg_log_posterior < best.neg_log_posterior
        ):
            best = result
    assert best is not None
    return best
