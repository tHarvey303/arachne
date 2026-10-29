"""Pixel regions and aperture SEDs over results bundles (numpy only).

A :class:`Region` is a boolean mask on the model grid plus optional geometry (in pixel
coordinates, ``[y, x]`` with pixel centres at integers) so it can be re-rasterised later.
:func:`region_sed` sums the observed flux, the model draws and the truth over the same pixels and
returns a :class:`RegionSED`.  Every function takes a ``ResultsBundle`` (duck typed: only its
public accessors are used).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Region",
    "RegionSED",
    "region_sed",
    "total_sed",
    "pixel_sed",
    "component_seds",
    "component_region_seds",
    "region_param_summary",
    "detection_region",
]


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = int(shape[0]), int(shape[1])
    return np.mgrid[0:h, 0:w].astype(float)


@dataclass(frozen=True, eq=False)
class Region:
    """A named set of pixels on the model grid.

    Attributes:
        name: Region name (used as the SED label by default).
        mask: Boolean array of shape (H, W).
        kind: ``"mask"``, ``"pixel"``, ``"box"``, ``"circle"`` or ``"polygon"``.
        geometry: Shape parameters in pixel coordinates: pixel ``{"y", "x"}``; box
            ``{"y0", "y1", "x0", "x1"}`` (centres inclusive); circle ``{"cy", "cx", "r"}``
            (centre distance ``<= r``); polygon ``{"vertices": [[y, x], ...]}`` (centres
            inside).
    """

    name: str
    mask: np.ndarray
    kind: str = "mask"
    geometry: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce the mask to a 2-D boolean array."""
        mask = np.asarray(self.mask).astype(bool)
        if mask.ndim != 2:
            raise ValueError(f"Region mask must be 2-D (H, W), got shape {mask.shape}")
        object.__setattr__(self, "mask", mask)

    @property
    def n_pix(self) -> int:
        """Number of pixels in the region."""
        return int(self.mask.sum())

    @property
    def shape(self) -> tuple[int, int]:
        """Shape (H, W) of the grid the mask lives on."""
        return self.mask.shape

    @classmethod
    def from_mask(
        cls,
        mask: np.ndarray,
        name: str = "mask",
        kind: str = "mask",
        geometry: dict | None = None,
    ) -> Region:
        """Wrap an existing (H, W) mask."""
        return cls(name, np.asarray(mask), kind, dict(geometry or {}))

    @classmethod
    def pixel(cls, shape: tuple[int, int], y: int, x: int, name: str | None = None) -> Region:
        """A single pixel ``[y, x]``."""
        mask = np.zeros(shape, dtype=bool)
        mask[int(y), int(x)] = True
        return cls(name or f"pixel ({int(y)},{int(x)})", mask, "pixel", {"y": int(y), "x": int(x)})

    @classmethod
    def box(
        cls,
        shape: tuple[int, int],
        y0: float,
        y1: float,
        x0: float,
        x1: float,
        name: str | None = None,
    ) -> Region:
        """Pixels whose centres lie in ``[y0, y1] x [x0, x1]`` (inclusive)."""
        yy, xx = _grid(shape)
        ylo, yhi = sorted((float(y0), float(y1)))
        xlo, xhi = sorted((float(x0), float(x1)))
        mask = (yy >= ylo) & (yy <= yhi) & (xx >= xlo) & (xx <= xhi)
        geom = {"y0": ylo, "y1": yhi, "x0": xlo, "x1": xhi}
        return cls(name or f"box ({ylo:g}:{yhi:g},{xlo:g}:{xhi:g})", mask, "box", geom)

    @classmethod
    def circle(
        cls, shape: tuple[int, int], cy: float, cx: float, r: float, name: str | None = None
    ) -> Region:
        """Pixels whose centres are within distance ``r`` of ``(cy, cx)``."""
        yy, xx = _grid(shape)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= float(r) ** 2
        geom = {"cy": float(cy), "cx": float(cx), "r": float(r)}
        return cls(name or f"circle ({cy:g},{cx:g}) r={r:g}", mask, "circle", geom)

    @classmethod
    def polygon(
        cls, shape: tuple[int, int], vertices: Sequence[Sequence[float]], name: str | None = None
    ) -> Region:
        """Pixels whose centres lie inside the polygon with ``[[y, x], ...]`` vertices."""
        from matplotlib.path import Path as MplPath

        verts = np.asarray(vertices, dtype=float).reshape(-1, 2)
        yy, xx = _grid(shape)
        pts = np.column_stack([yy.ravel(), xx.ravel()])
        mask = MplPath(verts).contains_points(pts).reshape(shape)
        geom = {"vertices": verts.tolist()}
        return cls(name or f"polygon ({len(verts)} vertices)", mask, "polygon", geom)

    @classmethod
    def from_geometry(
        cls, shape: tuple[int, int], kind: str, geometry: dict, name: str | None = None
    ) -> Region:
        """Re-rasterise a region from its geometry on a (possibly different) grid."""
        g = geometry
        if kind == "pixel":
            return cls.pixel(shape, g["y"], g["x"], name)
        if kind == "box":
            return cls.box(shape, g["y0"], g["y1"], g["x0"], g["x1"], name)
        if kind == "circle":
            return cls.circle(shape, g["cy"], g["cx"], g["r"], name)
        if kind == "polygon":
            return cls.polygon(shape, g["vertices"], name)
        raise ValueError(f"Cannot rasterise a region of kind {kind!r} from geometry")


@dataclass(frozen=True, eq=False)
class RegionSED:
    """Observed and model SED summed over one region.

    Attributes:
        label: Display label.
        band_names: Band names (B,).
        pivot_um: Pivot wavelengths in micron (B,).
        obs_flux: (B,) observed flux summed over region ∩ valid pixels, or None.
        obs_err: (B,) ``sqrt(sum variance) * error_scale * exp(s)``, or None.
        n_pix: (B,) number of valid pixels per band.
        model_draws: (S, B) PSF-convolved model draws summed over the same pixels, or None.
        model_map: (B,) best-point convolved model, or None.
        intrinsic_draws: (S, B) unconvolved model summed over the region (mask ignored), or
            None.
        intrinsic_map: (B,) best-point unconvolved model, or None.
        truth_convolved: (B,) noise-free truth over the valid pixels, or None.
        truth_intrinsic: (B,) unconvolved truth over the region, or None.
    """

    label: str
    band_names: list[str]
    pivot_um: np.ndarray
    obs_flux: np.ndarray | None
    obs_err: np.ndarray | None
    n_pix: np.ndarray
    model_draws: np.ndarray | None
    model_map: np.ndarray | None
    intrinsic_draws: np.ndarray | None
    intrinsic_map: np.ndarray | None
    truth_convolved: np.ndarray | None = None
    truth_intrinsic: np.ndarray | None = None

    def model_percentiles(
        self, q: Sequence[float] = (16, 50, 84), frame: str = "convolved"
    ) -> np.ndarray:
        """Percentiles of the model over draws.

        Args:
            q: Percentiles in [0, 100].
            frame: ``"convolved"`` or ``"intrinsic"``.

        Returns:
            Array of shape (len(q), B).  Without draws the best point is repeated.

        Raises:
            KeyError: If the frame has neither draws nor a best point.
        """
        if frame == "convolved":
            draws, best = self.model_draws, self.model_map
        elif frame == "intrinsic":
            draws, best = self.intrinsic_draws, self.intrinsic_map
        else:
            raise ValueError(f"frame must be 'convolved' or 'intrinsic', got {frame!r}")
        if draws is not None and len(draws) > 0:
            return np.percentile(draws, list(q), axis=0)
        if best is None:
            raise KeyError(f"No {frame} model in this RegionSED")
        return np.repeat(np.asarray(best, dtype=float)[None, :], len(q), axis=0)

    def chi(self, include_model_spread: bool = True) -> np.ndarray:
        """Normalised residual ``(obs - p50) / sqrt(obs_err^2 + [(p84 - p16)/2]^2)``.

        Args:
            include_model_spread: Add the model 16-84 half-width in quadrature.

        Returns:
            Array of shape (B,).
        """
        if self.obs_flux is None or self.obs_err is None:
            raise ValueError("RegionSED has no observed flux")
        p16, p50, p84 = self.model_percentiles((16, 50, 84))
        var = np.asarray(self.obs_err, dtype=float) ** 2
        if include_model_spread:
            var = var + (0.5 * (p84 - p16)) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            return (self.obs_flux - p50) / np.sqrt(var)

    def frac_residual(self) -> np.ndarray:
        """Fractional residual ``(obs - p50) / p50`` of shape (B,)."""
        if self.obs_flux is None:
            raise ValueError("RegionSED has no observed flux")
        p50 = self.model_percentiles((50,))[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (self.obs_flux - p50) / p50


def _as_region(bundle: Any, region: Region | np.ndarray, name: str = "mask") -> Region:
    if isinstance(region, Region):
        reg = region
    else:
        reg = Region.from_mask(np.asarray(region), name=name)
    if reg.mask.shape != tuple(bundle.image_shape):
        raise ValueError(
            f"Region shape {reg.mask.shape} does not match image shape {bundle.image_shape}"
        )
    return reg


def _sum_hw(cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sum ``cube[..., H, W]`` with weights broadcast over the trailing axes."""
    return (np.asarray(cube, dtype=float) * weights).sum(axis=(-2, -1))


def region_sed(
    bundle: Any,
    region: Region | np.ndarray,
    *,
    label: str = "",
    error_scale: float | np.ndarray = 1.0,
    model_floor: bool = False,
) -> RegionSED:
    """Observed and model SED summed over a region.

    For band ``b`` the pixels used are ``region ∩ (mask_b > 0)``, for the data, the convolved
    model and the convolved truth alike.  The intrinsic model and truth are summed over the
    whole region (they are not aperture-comparable).

    Args:
        bundle: A ``ResultsBundle``.
        region: :class:`Region` or (H, W) boolean mask.
        label: Display label (defaults to the region name).
        error_scale: Scalar or (B,) factor applied to the aperture error.
        model_floor: Add ``(f_b * sum model_b)^2`` to the error variance (a coherent model
            floor within the aperture; ``f`` = ``model_error_frac``).

    Returns:
        A :class:`RegionSED`.
    """
    reg = _as_region(bundle, region)
    valid = reg.mask[None, :, :] & (np.asarray(bundle.mask) > 0)  # (B, H, W)
    w = valid.astype(float)
    n_pix = valid.sum(axis=(1, 2))
    empty = n_pix == 0

    def _nan_empty(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        return np.where(empty, np.nan, arr)

    obs_flux = _nan_empty(_sum_hw(bundle.flux, w))
    err = np.sqrt(_sum_hw(bundle.variance, w)) * np.asarray(error_scale, dtype=float)
    err = err * bundle.noise_scale()

    model_map = _nan_empty(_sum_hw(bundle.cube("model", "map"), w))
    model_draws = None
    if bundle.has_draws:
        model_draws = np.einsum("sbhw,bhw->sb", bundle.model_draws("convolved"), w)
        model_draws = np.where(empty[None, :], np.nan, model_draws)

    rmask = reg.mask.astype(float)
    intrinsic_map = intrinsic_draws = None
    if bundle.has_intrinsic:
        intrinsic_map = _sum_hw(bundle.cube("intrinsic", "map"), rmask)
        if bundle.has_draws:
            try:
                intrinsic_draws = _sum_hw(bundle.model_draws("intrinsic"), rmask)
            except KeyError:
                intrinsic_draws = None

    truth_conv = truth_int = None
    truth = bundle.truth
    if truth is not None:
        if truth.convolved is not None:
            truth_conv = _nan_empty(_sum_hw(truth.convolved, w))
        if truth.intrinsic is not None:
            truth_int = _sum_hw(truth.intrinsic, rmask)

    if model_floor:
        best = np.median(model_draws, axis=0) if model_draws is not None else model_map
        err = np.sqrt(err**2 + (np.asarray(bundle.model_error_frac) * best) ** 2)
    err = _nan_empty(err)

    return RegionSED(
        label=label or reg.name,
        band_names=list(bundle.band_names),
        pivot_um=np.asarray(bundle.pivot_um, dtype=float),
        obs_flux=obs_flux,
        obs_err=err,
        n_pix=n_pix,
        model_draws=model_draws,
        model_map=model_map,
        intrinsic_draws=intrinsic_draws,
        intrinsic_map=intrinsic_map,
        truth_convolved=truth_conv,
        truth_intrinsic=truth_int,
    )


def _default_region(bundle: Any) -> Region:
    regions = bundle.regions
    for name in ("segment", "all"):
        if name in regions:
            return regions[name]
    return Region.from_mask(np.ones(bundle.image_shape, dtype=bool), name="all")


def total_sed(bundle: Any, **kw: Any) -> RegionSED:
    """SED over ``regions["segment"]`` if present, else ``"all"``, else every pixel.

    Args:
        bundle: A ``ResultsBundle``.
        **kw: Passed to :func:`region_sed`.
    """
    reg = _default_region(bundle)
    kw.setdefault("label", f"total ({reg.name})")
    return region_sed(bundle, reg, **kw)


def pixel_sed(bundle: Any, y: int, x: int, **kw: Any) -> RegionSED:
    """SED of the single pixel ``[y, x]`` (see :func:`region_sed` for ``kw``)."""
    return region_sed(bundle, Region.pixel(bundle.image_shape, y, x), **kw)


def component_seds(bundle: Any) -> list[RegionSED]:
    """Intrinsic (unconvolved, whole-plane) SED of every component plus their total.

    Args:
        bundle: A ``ResultsBundle``.

    Returns:
        ``K + 1`` RegionSEDs with ``obs_*`` and ``model_*`` None and the SED in
        ``intrinsic_draws`` (S, B) / ``intrinsic_map`` (B,); the last one is the total.
        Empty list when the bundle has no components.
    """
    comps = bundle.components
    if comps is None:
        return []
    seds = comps.sed_intrinsic()  # (S|1, K, B)
    sed_map = comps.sed_intrinsic_map()
    has_draws = comps.has_draws
    truth = bundle.truth
    truth_seds = truth.component_seds if truth is not None else None
    n_bands = len(bundle.band_names)
    full = np.full(n_bands, int(np.prod(bundle.image_shape)))
    out = []

    def _make(label, draws, best, truth_sed):
        return RegionSED(
            label=label,
            band_names=list(bundle.band_names),
            pivot_um=np.asarray(bundle.pivot_um, dtype=float),
            obs_flux=None,
            obs_err=None,
            n_pix=full,
            model_draws=None,
            model_map=None,
            intrinsic_draws=draws if has_draws else None,
            intrinsic_map=best,
            truth_convolved=None,
            truth_intrinsic=truth_sed,
        )

    for k in range(comps.n_components):
        t = None if truth_seds is None else np.asarray(truth_seds[k], dtype=float)
        out.append(_make(f"comp {k} ({comps.profile_names[k]})", seds[:, k], sed_map[k], t))
    t_tot = None if truth_seds is None else np.asarray(truth_seds, dtype=float).sum(axis=0)
    out.append(_make("total", seds.sum(axis=1), sed_map.sum(axis=0), t_tot))
    return out


def component_region_seds(bundle: Any, region: Region | np.ndarray) -> list[RegionSED]:
    """Per-component SEDs over a region, from the component image cubes.

    The convolved component cubes (PSF and shifts, no sky) are summed over
    ``region ∩ (mask_b > 0)`` into ``model_*``; the intrinsic component images are summed
    over the region into ``intrinsic_*``.  ``obs_*`` are None.

    Args:
        bundle: A ``ResultsBundle``.
        region: :class:`Region` or (H, W) mask.

    Returns:
        One RegionSED per component; empty when the bundle has no components.
    """
    comps = bundle.components
    if comps is None:
        return []
    reg = _as_region(bundle, region)
    valid = (reg.mask[None, :, :] & (np.asarray(bundle.mask) > 0)).astype(float)
    rmask = reg.mask.astype(float)
    n_pix = valid.sum(axis=(1, 2)).astype(int)
    out = []
    for k in range(comps.n_components):
        model_map = model_draws = int_map = int_draws = None
        if comps.has_frame("convolved"):
            model_map = _sum_hw(comps.image(k, "convolved", "map"), valid)
            if comps.has_frame_draws("convolved"):
                model_draws = _sum_hw(comps.image_draws(k, "convolved"), valid)
        if comps.has_frame("intrinsic"):
            int_map = _sum_hw(comps.image(k, "intrinsic", "map"), rmask)
            if comps.has_frame_draws("intrinsic"):
                int_draws = _sum_hw(comps.image_draws(k, "intrinsic"), rmask)
        out.append(
            RegionSED(
                label=f"comp {k} ({comps.profile_names[k]}) in {reg.name}",
                band_names=list(bundle.band_names),
                pivot_um=np.asarray(bundle.pivot_um, dtype=float),
                obs_flux=None,
                obs_err=None,
                n_pix=n_pix,
                model_draws=model_draws,
                model_map=model_map,
                intrinsic_draws=int_draws,
                intrinsic_map=int_map,
            )
        )
    return out


def region_param_summary(bundle: Any, region: Region | np.ndarray) -> dict[str, np.ndarray]:
    """Per-draw region aggregates of the parameter maps.

    The mass parameter (``bundle.mass_param``, log10) becomes ``log10 sum 10**m`` over the
    region; every other map becomes its mass-weighted mean (plain mean when there is no mass
    parameter).  Uses the map draws when present, otherwise the best-point maps.

    Args:
        bundle: A ``ResultsBundle``.
        region: :class:`Region` or (H, W) mask.

    Returns:
        ``{name: (S,) or (1,)}`` for every map name, plus ``"n_pix"`` (0-d int array).
    """
    reg = _as_region(bundle, region)
    out: dict[str, np.ndarray] = {"n_pix": np.asarray(reg.n_pix)}
    names = list(bundle.map_names)
    if not names or reg.n_pix == 0:
        return out
    if bundle.has_param_draws:
        maps = np.stack([bundle.param_draws(n) for n in names], axis=1)  # (S, Nm, H, W)
    else:
        maps = np.stack([bundle.param_map(n, "map") for n in names], axis=0)[None]
    vals = np.asarray(maps, dtype=float)[:, :, reg.mask]  # (S, Nm, N)
    mass = bundle.mass_param
    if mass and mass in names:
        m = vals[:, names.index(mass)]
        mmax = m.max(axis=1, keepdims=True)
        wts = 10.0 ** (m - mmax)
        wsum = wts.sum(axis=1)
        log_total = np.log10(wsum) + mmax[:, 0]
    else:
        wts = np.ones(vals[:, 0].shape)
        wsum = wts.sum(axis=1)
        log_total = None
    for i, name in enumerate(names):
        if name == mass and log_total is not None:
            out[name] = log_total
        else:
            out[name] = (vals[:, i] * wts).sum(axis=1) / wsum
    return out


def detection_region(
    flux: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    nsigma: float = 3.0,
    smooth_px: float = 1.0,
) -> np.ndarray:
    """Largest connected detection in an inverse-variance-weighted S/N image.

    The detection image is ``sum_b w_b f_b / sqrt(sum_b w_b)`` with ``w = 1/variance`` on
    valid pixels (a unit-variance S/N map), smoothed with a Gaussian of ``smooth_px`` and
    renormalised so its noise stays unit, then thresholded at ``nsigma``.

    Args:
        flux: (B, H, W) flux.
        variance: (B, H, W) variance.
        mask: (B, H, W) weights (> 0 = valid).
        nsigma: Detection threshold in units of the (smoothed) noise.
        smooth_px: Gaussian smoothing sigma in pixels (0 = none).

    Returns:
        Boolean (H, W) mask; all False when nothing is detected.
    """
    from scipy import ndimage

    flux = np.asarray(flux, dtype=float)
    variance = np.asarray(variance, dtype=float)
    valid = (np.asarray(mask) > 0) & np.isfinite(flux) & np.isfinite(variance) & (variance > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        wts = np.where(valid, 1.0 / np.where(valid, variance, 1.0), 0.0)
    num = (wts * np.where(valid, flux, 0.0)).sum(axis=0)
    den = np.sqrt(wts.sum(axis=0))
    snr = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    if smooth_px > 0:
        snr = ndimage.gaussian_filter(snr, smooth_px, mode="constant")
        delta = np.zeros((int(8 * smooth_px) * 2 + 1,) * 2)
        delta[delta.shape[0] // 2, delta.shape[1] // 2] = 1.0
        kern = ndimage.gaussian_filter(delta, smooth_px, mode="constant")
        snr = snr / np.sqrt((kern**2).sum())
    det = snr > nsigma
    labels, n = ndimage.label(det)
    if n == 0:
        return np.zeros(snr.shape, dtype=bool)
    sizes = ndimage.sum(det, labels, index=np.arange(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)
