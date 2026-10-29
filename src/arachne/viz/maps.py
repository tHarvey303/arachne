"""Parameter maps and region overlays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import Patch

from ..results.display import percentile_limits
from ..results.reader import ResultsBundle
from ..results.regions import Region
from .sed import resolve_regions
from .style import (
    as_bundle,
    draw_mask_outline,
    image_kw,
    new_figure,
    note_figure,
    param_label,
    region_color,
    root_figure,
    stat_label,
    stretch_norm,
    style_context,
)

__all__ = ["plot_param_maps", "plot_regions", "param_map_style"]

DEFAULT_STATS = ("truth", "p50", "std", "z", "information")
_VALUE_STATS = ("map", "p16", "p50", "p84", "mean", "truth")


def param_map_style(stat: str, img: np.ndarray, value_lim: tuple[float, float] | None) -> dict:
    """``imshow`` keywords for a parameter-map statistic (colour map and limits)."""
    from matplotlib.colors import Normalize

    if stat in _VALUE_STATS:
        lo, hi = value_lim if value_lim is not None else percentile_limits(img, (1, 99))
        return image_kw("viridis", Normalize(lo, hi))
    if stat in ("std", "prior_std"):
        lo, hi = percentile_limits(img, (1, 99))
        return image_kw("magma", Normalize(0.0, hi))
    if stat == "z":
        return image_kw("RdBu_r", Normalize(-3.0, 3.0))
    if stat == "diff":
        lo, hi = percentile_limits(img, (1, 99), symmetric=True)
        return image_kw("RdBu_r", Normalize(lo, hi))
    if stat == "information":
        return image_kw("cividis", Normalize(0.0, 1.0))
    lo, hi = percentile_limits(img)
    return image_kw("viridis", Normalize(lo, hi))


def _clean_ax(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()


def plot_param_maps(
    bundle: ResultsBundle | str | Path,
    names: Sequence[str] | None = None,
    stats: Sequence[str] | None = None,
    mask_region: str | None = "segment",
    *,
    fig: Figure | SubFigure | None = None,
) -> Figure:
    """Grid of parameter maps: one row per parameter, one column per statistic.

    Value-like statistics (truth, MAP, percentiles) of one parameter share colour limits.
    ``std`` uses magma from zero, ``z`` = (median - truth)/std a fixed [-3, 3] RdBu_r,
    ``information`` cividis on [0, 1].

    Args:
        bundle: Bundle or path.
        names: Parameter names (default: all :attr:`~ResultsBundle.map_names`).
        stats: Statistics (default: those of ``truth, p50, std, z, information`` that the
            bundle provides; ``map`` replaces ``p50`` when there are no summaries).
        mask_region: Grey out pixels outside this bundle region (ignored when absent).
        fig: Figure or SubFigure to draw into.

    Returns:
        The Figure; a note figure when the bundle has no parameter maps.
    """
    bundle = as_bundle(bundle)
    names = list(names) if names is not None else list(bundle.map_names)
    if not names:
        return note_figure("No parameter maps in this bundle.", fig=fig)
    avail = bundle.available_param_stats()
    if stats is None:
        stats = [s for s in DEFAULT_STATS if s in avail]
        if "p50" not in stats:
            stats.insert(1 if "truth" in stats else 0, "map")
    stats = list(stats)
    region_mask = None
    if mask_region and mask_region in bundle.regions:
        region_mask = bundle.regions[mask_region].mask

    with style_context():
        nr, nc = len(names), len(stats)
        fig = new_figure(fig, (2.35 * nc + 0.6, 2.15 * nr + 0.5))
        axes = fig.subplots(nr, nc, squeeze=False)
        for i, name in enumerate(names):
            imgs: dict[str, np.ndarray] = {}
            for s in stats:
                try:
                    img = np.asarray(bundle.param_map(name, s), dtype=float)
                except KeyError:
                    continue
                if bundle.last_fallback:  # a summary silently replaced by MAP: not shown
                    continue
                imgs[s] = img
                if region_mask is not None:
                    imgs[s] = np.where(region_mask, imgs[s], np.nan)
            vals = [imgs[s] for s in imgs if s in _VALUE_STATS]
            vlim = percentile_limits(np.stack(vals), (1, 99)) if vals else None
            for j, s in enumerate(stats):
                ax = axes[i, j]
                _clean_ax(ax)
                if i == 0:
                    ax.set_title(stat_label(s))
                if j == 0:
                    ax.set_ylabel(param_label(bundle, name), fontsize=10)
                if s not in imgs:
                    ax.text(
                        0.5,
                        0.5,
                        "not available",
                        ha="center",
                        va="center",
                        color="0.5",
                        transform=ax.transAxes,
                    )
                    ax.set_frame_on(False)
                    continue
                im = ax.imshow(imgs[s], **param_map_style(s, imgs[s], vlim))
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.92)
                cb.ax.tick_params(labelsize=7)
                cb.ax.minorticks_off()
        title = bundle.title or "parameter maps"
        src = bundle.posterior_source
        extra = f", masked outside '{mask_region}'" if region_mask is not None else ""
        fig.suptitle(f"{title}: parameter maps (posterior: {src}{extra})")
    return root_figure(fig)


def plot_regions(
    bundle: ResultsBundle | str | Path,
    regions: Mapping[str, Region | np.ndarray] | Sequence[str] | None = None,
    band: int = -1,
    *,
    ax: Any = None,
    fig: Figure | SubFigure | None = None,
    stretch: str = "asinh",
) -> Figure:
    """Region outlines over one band of the data.

    Args:
        bundle: Bundle or path.
        regions: As in :func:`~arachne.viz.sed.resolve_regions` (default: bundle regions).
        band: Band index of the background image (default: reddest = last).
        ax: Existing axes to draw into.
        fig: Figure or SubFigure to draw into (ignored when ``ax`` is given).
        stretch: Display stretch of the background.
    """
    bundle = as_bundle(bundle)
    regs = resolve_regions(bundle, regions)
    with style_context():
        if ax is None:
            fig = new_figure(fig, (5.2, 4.6))
            ax = fig.add_subplot()
        else:
            fig = ax.figure
        img = bundle.band_image(band, "data")
        lo, hi = percentile_limits(img, (1.0, 99.7))
        im = ax.imshow(img, **image_kw("Greys", stretch_norm(stretch, lo, hi)))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("flux [nJy]")
        cb.ax.minorticks_off()
        handles = []
        for i, (name, reg) in enumerate(regs.items()):
            c = region_color(i)
            draw_mask_outline(ax, reg.mask, c, lw=1.6)
            handles.append(
                Patch(facecolor="none", edgecolor=c, lw=1.6, label=f"{name} ({reg.n_pix} px)")
            )
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.08),
            ncol=min(3, len(handles)),
            frameon=False,
        )
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        name = bundle.short_names[band] if bundle.short_names else str(band)
        ax.set_title(f"{bundle.title + ': ' if bundle.title else ''}regions on {name}")
    return root_figure(fig)
