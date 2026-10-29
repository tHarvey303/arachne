"""Band images: data / model / residual / chi mosaics and RGB composites."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure, SubFigure

from ..results.display import percentile_limits, stretch
from ..results.reader import ResultsBundle
from .style import (
    as_bundle,
    band_colors,
    image_kw,
    new_figure,
    root_figure,
    stretch_norm,
    style_context,
)

__all__ = ["plot_band_mosaic", "plot_rgb", "default_rgb_bands"]

_KIND_TITLES = {
    "data": "data",
    "model": "model",
    "intrinsic": "intrinsic model",
    "residual": "data − model",
    "chi": r"$\chi$ = residual / $\sigma$",
    "truth_model": "truth (noise-free)",
    "truth_intrinsic": "truth (intrinsic)",
    "truth_residual": "data − truth",
    "sigma": r"$\sigma$",
    "variance": "variance",
    "mask": "mask",
}
_FLUX_KINDS = ("data", "model", "truth_model")
_SYM_KINDS = ("residual", "truth_residual")
CHI_LIM = 4.0


def _available(bundle: ResultsBundle, kind: str, stat: str) -> bool:
    try:
        bundle.band_image(0, kind, stat)
    except KeyError:
        return False
    return True


def plot_band_mosaic(
    bundle: ResultsBundle | str | Path,
    bands: Sequence[int] | None = None,
    kinds: Sequence[str] = ("data", "model", "residual", "chi"),
    stat: str = "p50",
    stretch: str = "asinh",
    model_floor: bool = True,
    *,
    fig: Figure | SubFigure | None = None,
) -> Figure:
    """Mosaic of band images: one row per band, one column per image kind.

    Data, model and truth share per-band limits and stretch (colourbar in nJy); residuals use
    a symmetric linear scale set by the residuals of that band; chi uses a fixed [-4, 4]
    RdBu_r.  Kinds not in the bundle (e.g. intrinsic in a degraded bundle) are dropped.

    Args:
        bundle: Bundle or path.
        bands: Band indices (default: all, in stored order).
        kinds: Image kinds as in :meth:`ResultsBundle.band_image`.
        stat: Model statistic (``p50`` falls back to ``map`` without draws).
        stretch: Stretch for flux-like images (``linear``, ``asinh``, ``log``, ``sqrt``).
        model_floor: Use the likelihood's model-error floor in chi.
        fig: Figure or SubFigure to draw into.
    """
    bundle = as_bundle(bundle)
    bands = list(range(bundle.n_bands)) if bands is None else [int(b) for b in bands]
    kinds = [k for k in kinds if _available(bundle, k, stat)]
    bundle.band_image(0, "model", stat)
    fell_back = bundle.last_fallback
    cols = band_colors(bundle.pivot_um)
    with style_context():
        nr, nc = len(bands), len(kinds)
        fig = new_figure(fig, (2.3 * nc + 0.4, 2.05 * nr + 0.6))
        axes = fig.subplots(nr, nc, squeeze=False)
        for i, b in enumerate(bands):
            data = bundle.band_image(b, "data")
            lo, hi = percentile_limits(data, (0.5, 99.7))
            for j, kind in enumerate(kinds):
                ax = axes[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.minorticks_off()
                img = bundle.band_image(b, kind, stat, model_floor=model_floor)
                if kind in _FLUX_KINDS or kind == "intrinsic":
                    if kind == "intrinsic":
                        ilo, ihi = percentile_limits(img, (0.5, 99.7))
                        norm = stretch_norm(stretch, ilo, ihi)
                    else:
                        norm = stretch_norm(stretch, lo, hi)
                    kw = image_kw("magma", norm)
                elif kind in _SYM_KINDS:
                    rlo, rhi = percentile_limits(img, (0.5, 99.5), symmetric=True)
                    kw = image_kw("RdBu_r", Normalize(rlo, rhi))
                elif kind == "chi":
                    kw = image_kw("RdBu_r", Normalize(-CHI_LIM, CHI_LIM))
                else:
                    kw = image_kw("viridis", Normalize(*percentile_limits(img)))
                im = ax.imshow(img, **kw)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.92)
                cb.ax.tick_params(labelsize=6.5)
                cb.ax.minorticks_off()
                if i == 0:
                    title = _KIND_TITLES.get(kind, kind)
                    if kind in ("model", "residual", "chi") and fell_back:
                        title += " (MAP)"
                    elif kind in ("model", "residual", "chi") and stat != "p50":
                        title += f" ({stat})"
                    ax.set_title(title)
                if j == 0:
                    lam = bundle.pivot_um[b]
                    lab = bundle.short_names[b]
                    if np.isfinite(lam):
                        lab += f"\n{lam:.2f} µm"
                    ax.set_ylabel(lab, color=cols[b], fontsize=9.5, fontweight="bold")
        units = "flux in nJy; " if any(k in _FLUX_KINDS for k in kinds) else ""
        fig.suptitle(
            f"{bundle.title + ': ' if bundle.title else ''}band images "
            f"({units}{stretch} stretch, grey = masked)"
        )
    return root_figure(fig)


def default_rgb_bands(bundle: ResultsBundle) -> tuple[int, int, int]:
    """(red, green, blue) band indices: reddest, middle and bluest by pivot."""
    lam = np.asarray(bundle.pivot_um, dtype=float)
    key = np.where(np.isfinite(lam), lam, np.arange(lam.size))
    order = np.argsort(key)
    n = order.size
    return int(order[-1]), int(order[n // 2]), int(order[0])


def _rgb_from(planes: list[np.ndarray], hi: float, asinh_a: float = 0.1) -> np.ndarray:
    out = np.stack([stretch(p, "asinh", 0.0, hi, asinh_a=asinh_a) for p in planes], axis=-1)
    return np.nan_to_num(out, nan=0.0)


def plot_rgb(
    bundle: ResultsBundle | str | Path,
    bands: tuple[int, int, int] | None = None,
    kinds: Sequence[str] = ("data", "model", "residual"),
    *,
    stat: str = "p50",
    fig: Figure | SubFigure | None = None,
    pct: float = 99.5,
    asinh_a: float = 0.1,
) -> Figure:
    """RGB composites of data, model and residuals on one shared scale.

    Data and model (and truth) use one asinh scale set by the data, so colours are directly
    comparable.  The residual composite is centred on mid-grey: grey means a perfect fit and
    colour tints show band-dependent misfits, scaled to the 99.5th percentile of |residual|.

    Args:
        bundle: Bundle or path.
        bands: (red, green, blue) band indices (default: reddest, middle, bluest).
        kinds: Any of ``data``, ``model``, ``residual``, ``truth_model``, ``intrinsic``.
        stat: Model statistic.
        fig: Figure or SubFigure to draw into.
        pct: Upper percentile of the shared flux scale.
        asinh_a: Asinh softening.
    """
    bundle = as_bundle(bundle)
    rgb = default_rgb_bands(bundle) if bands is None else tuple(int(b) for b in bands)
    kinds = [k for k in kinds if _available(bundle, k, stat)]
    data = [bundle.band_image(b, "data") for b in rgb]
    allv = np.concatenate([d[np.isfinite(d)] for d in data])
    hi = float(np.percentile(allv, pct)) if allv.size else 1.0
    hi = hi if hi > 0 else 1.0
    with style_context():
        fig = new_figure(fig, (2.9 * len(kinds) + 0.2, 3.3))
        axes = np.atleast_1d(fig.subplots(1, len(kinds), squeeze=False)[0])
        for ax, kind in zip(axes, kinds):
            planes = [bundle.band_image(b, kind, stat) for b in rgb]
            title = _KIND_TITLES.get(kind, kind)
            if kind in _SYM_KINDS:
                vals = np.concatenate([np.abs(p[np.isfinite(p)]) for p in planes])
                rmax = float(np.percentile(vals, 99.5)) if vals.size else 1.0
                rmax = rmax if rmax > 0 else 1.0
                img = np.stack([0.5 + 0.5 * np.clip(p / rmax, -1, 1) for p in planes], axis=-1)
                img = np.nan_to_num(img, nan=0.0)
                title += f"\n(grey = 0, full scale ±{rmax:.3g} nJy)"
            else:
                if kind == "model" and bundle.last_fallback:
                    title += " (MAP)"
                img = _rgb_from(planes, hi, asinh_a)
            ax.imshow(img, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.minorticks_off()
            ax.set_title(title)
        names = [bundle.short_names[b] for b in rgb]
        fig.suptitle(
            f"{bundle.title + ': ' if bundle.title else ''}RGB = "
            f"{names[0]} / {names[1]} / {names[2]}  (shared asinh scale)"
        )
    return root_figure(fig)
