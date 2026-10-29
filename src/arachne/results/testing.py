"""Synthetic results bundles for tests and examples (numpy + scipy only).

Three flavours, fully specified in ``docs/design/results_bundle_gui_spec.md`` section 3:

- ``"field"``: two parameter maps (``log_mass``, ``Av``) with per-pixel draws, intrinsic and
  convolved cubes, prior std maps, a ``total_log_mass`` scalar.
- ``"additive"``: two factorised Gaussian components (sigma 1.5 and 4 px), no parameter maps.
- ``"degraded"``: the field flavour with S=0, MAP cubes plus parameter summaries only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .metadata import DEFAULT_LABELS
from .regions import Region
from .schema import BundleContent, ComponentContent, TruthContent

__all__ = ["toy_sed", "make_synthetic_content", "make_synthetic_bundle", "BAND_NAMES", "PIVOTS"]

BAND_NAMES = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F444W"]
PIVOTS = np.array([1.154, 1.988, 4.402])
PSF_SIGMAS = (0.8, 1.0, 1.5)
PIXEL_SCALE = 0.03
REDSHIFT = 2.0
COMPONENT_PARAM_NAMES = ["mu_y", "mu_x", "size_y", "size_x", "rho", "sersic_n", "log_mass", "Av"]


def toy_sed(log_mass: Any, av: Any, pivot_um: Any, redshift: float = 2.0) -> np.ndarray:
    """Toy mass-linear, dust-attenuated SED in nJy.

    ``10**(log_mass - 8) * 10 * (lam/2)**0.5 * 10**(-0.4 * av * (lam/(1+z)/0.55)**-1)``.

    Args:
        log_mass: log10 mass, any shape.
        av: V-band attenuation, broadcastable to ``log_mass``.
        pivot_um: (B,) pivot wavelengths in micron.
        redshift: Redshift.

    Returns:
        Array of shape ``(..., B)``.
    """
    lm = np.asarray(log_mass, dtype=float)[..., None]
    a = np.asarray(av, dtype=float)[..., None]
    lam = np.asarray(pivot_um, dtype=float)
    rest = lam / (1.0 + redshift) / 0.55
    return 10.0 ** (lm - 8.0) * 10.0 * (lam / 2.0) ** 0.5 * 10.0 ** (-0.4 * a * rest**-1)


def _blur(cube: np.ndarray) -> np.ndarray:
    """Per-band Gaussian "PSF" on an array (..., B, H, W) with B = 3."""
    from scipy.ndimage import gaussian_filter

    cube = np.asarray(cube, dtype=float)
    out = np.empty_like(cube)
    lead = cube.ndim - 3
    for b, sig in enumerate(PSF_SIGMAS):
        sl = (Ellipsis, b, slice(None), slice(None))
        out[sl] = gaussian_filter(cube[sl], sigma=[0.0] * lead + [sig, sig], mode="constant")
    return out


def _radius(shape: tuple[int, int]) -> tuple[np.ndarray, float, float]:
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    return np.hypot(yy - cy, xx - cx), cy, cx


def _observe(clean: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    sigma = 0.03 * clean.max(axis=(1, 2))
    flux = clean + rng.normal(0.0, 1.0, clean.shape) * sigma[:, None, None]
    variance = np.broadcast_to((sigma**2)[:, None, None], clean.shape).copy()
    mask = np.ones(clean.shape)
    mask[0, 0, 0] = 0.0
    return flux, variance, mask


def _chi2_red(flux, variance, mask, model) -> float:
    good = mask > 0
    return float(np.mean(((flux - model) ** 2 / variance)[good]))


def _regions(shape: tuple[int, int]) -> dict[str, Region]:
    r, cy, cx = _radius(shape)
    return {
        "all": Region.from_mask(np.ones(shape, dtype=bool), name="all"),
        "core": Region("core", r < 3.0, "circle", {"cy": cy, "cx": cx, "r": 3.0}),
    }


def _log_sum(log_mass: np.ndarray) -> np.ndarray:
    """log10 of the sum of 10**m over the last two axes."""
    m = np.asarray(log_mass, dtype=float)
    mmax = m.max(axis=(-2, -1), keepdims=True)
    return np.log10((10.0 ** (m - mmax)).sum(axis=(-2, -1))) + mmax[..., 0, 0]


def _field(shape, n_draws, with_truth, seed) -> BundleContent:
    r, _, _ = _radius(shape)
    truth_maps = np.stack([8.0 - 0.15 * r, 0.2 + 0.6 * np.exp(-(r**2) / 18.0)])  # (2, H, W)
    truth_int = np.moveaxis(toy_sed(truth_maps[0], truth_maps[1], PIVOTS, REDSHIFT), -1, 0)
    truth_conv = _blur(truth_int)
    rng = np.random.default_rng(seed)
    flux, variance, mask = _observe(truth_conv, rng)

    map_maps = truth_maps + 0.02
    int_map = np.moveaxis(toy_sed(map_maps[0], map_maps[1], PIVOTS, REDSHIFT), -1, 0)
    conv_map = _blur(int_map)

    s = int(n_draws)
    draws = int_draws = conv_draws = scalars_draws = None
    if s > 0:
        draws = truth_maps[None] + rng.normal(0.0, 0.05, (s, *truth_maps.shape))
        draws[:, 1] = np.clip(draws[:, 1], 0.0, None)
        int_draws = np.moveaxis(toy_sed(draws[:, 0], draws[:, 1], PIVOTS, REDSHIFT), -1, 1)
        conv_draws = _blur(int_draws)
        scalars_draws = _log_sum(draws[:, 0])[:, None]
    p50 = np.median(conv_draws, axis=0) if s > 0 else conv_map

    truth = None
    if with_truth:
        truth = TruthContent(
            convolved=truth_conv,
            intrinsic=truth_int,
            param_maps=truth_maps,
            scalars=np.array([_log_sum(truth_maps[0])]),
            theta=truth_maps.ravel(),
        )
    return BundleContent(
        band_names=list(BAND_NAMES),
        pivot_um=PIVOTS.copy(),
        pixel_scale=PIXEL_SCALE,
        flux=flux,
        variance=variance,
        mask=mask,
        model_error_frac=np.array([0.05] * 3),
        convolved_map=conv_map,
        convolved_draws=conv_draws,
        intrinsic_map=int_map,
        intrinsic_draws=int_draws,
        map_names=["log_mass", "Av"],
        param_maps_map=map_maps,
        param_maps_draws=draws,
        prior_std_maps=np.full(truth_maps.shape, 0.5),
        scalar_names=["total_log_mass"],
        scalars_map=np.array([_log_sum(map_maps[0])]),
        scalars_draws=scalars_draws,
        fixed_params={"redshift": REDSHIFT},
        mass_param="log_mass",
        labels={n: DEFAULT_LABELS[n] for n in ("log_mass", "Av", "total_log_mass")},
        units={"log_mass": "dex", "Av": "mag", "total_log_mass": "dex"},
        truth=truth,
        theta_map=map_maps.ravel(),
        theta_draws=None if draws is None else draws.reshape(s, -1),
        draw_index=None if s == 0 else np.arange(s),
        chain_id=None if s == 0 else np.zeros(s, dtype=int),
        regions=_regions(shape),
        metrics={"chi2_red": _chi2_red(flux, variance, mask, p50)},
        posterior_source="nuts",
        spatial_model="CorrelatedFieldMap",
        redshift=REDSHIFT,
        title="synthetic field",
    )


def _additive(shape, n_draws, with_truth, seed) -> BundleContent:
    r, cy, cx = _radius(shape)
    sizes = np.array([1.5, 4.0])
    prof = np.exp(-(r[None] ** 2) / (2.0 * sizes[:, None, None] ** 2))
    prof /= prof.sum(axis=(1, 2), keepdims=True)  # (K, H, W)
    lm_true = np.array([9.5, 10.0])
    av_true = np.array([0.1, 0.8])
    sed_true = toy_sed(lm_true, av_true, PIVOTS, REDSHIFT)  # (K, B)
    comp_int_true = sed_true[:, :, None, None] * prof[:, None]  # (K, B, H, W)
    comp_conv_true = _blur(comp_int_true)
    truth_int = comp_int_true.sum(axis=0)
    truth_conv = comp_conv_true.sum(axis=0)
    rng = np.random.default_rng(seed)
    flux, variance, mask = _observe(truth_conv, rng)

    def params(lm, av):
        lm = np.asarray(lm, dtype=float)
        n = lm.shape[:-1]
        cols = [
            np.full(lm.shape, cy),
            np.full(lm.shape, cx),
            np.broadcast_to(sizes, lm.shape),
            np.broadcast_to(sizes, lm.shape),
            np.zeros(lm.shape),
            np.full(lm.shape, np.nan),
            lm,
            np.asarray(av, dtype=float),
        ]
        out = np.stack(cols, axis=-1)
        return out.reshape(*n, 2, len(COMPONENT_PARAM_NAMES))

    s = int(n_draws)
    sed_draws = comp_int_draws = comp_conv_draws = params_draws = None
    conv_draws = int_draws = prof_draws = draw_index = theta_draws = scalars_draws = None
    if s > 0:
        lm_d = lm_true[None] + rng.normal(0.0, 0.02, (s, 2))
        av_d = np.clip(av_true[None] + rng.normal(0.0, 0.03, (s, 2)), 0.0, None)
        sed_draws = toy_sed(lm_d, av_d, PIVOTS, REDSHIFT)  # (S, K, B)
        comp_int_draws = sed_draws[:, :, :, None, None] * prof[None, :, None]
        comp_conv_draws = _blur(comp_int_draws)
        conv_draws = comp_conv_draws.sum(axis=1)
        int_draws = comp_int_draws.sum(axis=1)
        prof_draws = np.repeat(prof[None], s, axis=0)
        draw_index = np.arange(s)
        params_draws = params(lm_d, av_d)
        theta_draws = np.concatenate([lm_d, av_d], axis=1)
        scalars_draws = np.log10((10.0**lm_d).sum(axis=1))[:, None]
    conv_map = truth_conv.copy()
    p50 = np.median(conv_draws, axis=0) if s > 0 else conv_map

    components = ComponentContent(
        profile_names=["gaussian", "gaussian"],
        param_names=list(COMPONENT_PARAM_NAMES),
        factorised=True,
        sed_intrinsic_map=sed_true,
        sed_intrinsic_draws=sed_draws,
        profiles_map=prof,
        profiles_draws=prof_draws,
        intrinsic_map=None,
        intrinsic_draws=None,
        convolved_map=comp_conv_true,
        convolved_draws=comp_conv_draws,
        draw_index=draw_index,
        params_map=params(lm_true, av_true),
        params_draws=params_draws,
        ordering="size",
    )
    truth = None
    if with_truth:
        truth = TruthContent(
            convolved=truth_conv,
            intrinsic=truth_int,
            scalars=np.array([np.log10((10.0**lm_true).sum())]),
            component_seds=sed_true,
            theta=np.concatenate([lm_true, av_true]),
        )
    return BundleContent(
        band_names=list(BAND_NAMES),
        pivot_um=PIVOTS.copy(),
        pixel_scale=PIXEL_SCALE,
        flux=flux,
        variance=variance,
        mask=mask,
        model_error_frac=np.array([0.05] * 3),
        convolved_map=conv_map,
        convolved_draws=conv_draws,
        intrinsic_map=truth_int.copy(),
        intrinsic_draws=int_draws,
        scalar_names=["total_log_mass"],
        scalars_map=np.array([np.log10((10.0**lm_true).sum())]),
        scalars_draws=scalars_draws,
        fixed_params={"redshift": REDSHIFT},
        labels={"total_log_mass": DEFAULT_LABELS["total_log_mass"]},
        units={"total_log_mass": "dex"},
        components=components,
        truth=truth,
        theta_map=np.concatenate([lm_true, av_true]),
        theta_draws=theta_draws,
        draw_index=draw_index,
        chain_id=None if s == 0 else np.zeros(s, dtype=int),
        regions=_regions(shape),
        metrics={"chi2_red": _chi2_red(flux, variance, mask, p50)},
        posterior_source="nuts",
        spatial_model="AdditiveComponentModel",
        redshift=REDSHIFT,
        title="synthetic additive",
    )


def _degraded(shape, n_draws, with_truth, seed) -> BundleContent:
    c = _field(shape, n_draws if n_draws > 0 else 20, with_truth, seed)
    draws = np.asarray(c.param_maps_draws)
    std = draws.std(axis=0)
    c.param_summaries = {
        "p50": np.median(draws, axis=0),
        "std": std,
        "information": 1.0 - (std / 0.5) ** 2,
    }
    for name in (
        "convolved_draws",
        "intrinsic_map",
        "intrinsic_draws",
        "param_maps_draws",
        "prior_std_maps",
        "scalars_draws",
        "theta_draws",
        "draw_index",
        "chain_id",
    ):
        setattr(c, name, None)
    c.degraded = True
    c.posterior_source = "map"
    c.title = "synthetic degraded"
    c.metrics = {"chi2_red": _chi2_red(c.flux, c.variance, c.mask, c.convolved_map)}
    return c


_FLAVOURS = {"field": _field, "additive": _additive, "degraded": _degraded}


def make_synthetic_content(
    flavour: str = "field",
    *,
    shape: tuple[int, int] = (16, 16),
    n_draws: int = 20,
    with_truth: bool = True,
    seed: int = 0,
) -> BundleContent:
    """Build a small synthetic :class:`BundleContent`.

    Args:
        flavour: ``"field"``, ``"additive"`` or ``"degraded"``.
        shape: Image shape (H, W).
        n_draws: Number of draws S (the degraded flavour always stores none; it uses
            ``n_draws`` draws, or 20 if 0, to compute its parameter summaries).
        with_truth: Populate :class:`TruthContent`.
        seed: Seed of ``np.random.default_rng`` (noise first, then draws).

    Returns:
        The content (validated).
    """
    if flavour not in _FLAVOURS:
        raise ValueError(f"flavour must be one of {sorted(_FLAVOURS)}, got {flavour!r}")
    content = _FLAVOURS[flavour](tuple(int(n) for n in shape), int(n_draws), with_truth, seed)
    content.validate()
    return content


def make_synthetic_bundle(
    path: str | Path,
    flavour: str = "field",
    *,
    mantissa_bits: int | None = None,
    overwrite: bool = True,
    **kw: Any,
) -> Path:
    """Write a synthetic bundle to ``path`` (see :func:`make_synthetic_content` for ``kw``)."""
    from .io import write_content

    content = make_synthetic_content(flavour, **kw)
    return write_content(path, content, mantissa_bits=mantissa_bits, overwrite=overwrite)
