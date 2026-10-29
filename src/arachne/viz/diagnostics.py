"""Goodness-of-fit diagnostics: per-band chi histograms and reduced chi^2 per band."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure, SubFigure

from ..results.reader import ResultsBundle
from .style import (
    OBS_COLOR,
    add_band_axis,
    as_bundle,
    band_colors,
    new_figure,
    root_figure,
    set_micron_axis,
    style_context,
)

__all__ = ["plot_chi_histograms", "plot_chi2_per_band"]

CHI_RANGE = 6.0


def _floor_text(model_floor: bool) -> str:
    return "with model floor" if model_floor else "data variance only"


def plot_chi_histograms(
    bundle: ResultsBundle | str | Path,
    *,
    model_floor: bool = True,
    stat: str = "p50",
    fig: Figure | SubFigure | None = None,
) -> Figure:
    """Histogram of per-pixel chi in every band against a unit normal.

    Values beyond ±6 are collected in the edge bins; the title gives the rms and the fraction
    of pixels with |chi| > 3.

    Args:
        bundle: Bundle or path.
        model_floor: Include the model-error floor in chi (as in the likelihood).
        stat: Model statistic (``p50`` falls back to ``map`` without draws).
        fig: Figure or SubFigure to draw into.
    """
    bundle = as_bundle(bundle)
    nb = bundle.n_bands
    ncols = min(nb, 4)
    nrows = int(np.ceil(nb / ncols))
    cols = band_colors(bundle.pivot_um)
    edges = np.linspace(-CHI_RANGE, CHI_RANGE, 49)
    xs = np.linspace(-CHI_RANGE, CHI_RANGE, 300)
    gauss = np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi)
    order = np.argsort(np.where(np.isfinite(bundle.pivot_um), bundle.pivot_um, np.inf))
    with style_context():
        fig = new_figure(fig, (2.6 * ncols + 0.3, 2.2 * nrows + 0.6))
        axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True)
        for ax in axes.ravel()[nb:]:
            ax.set_axis_off()
        for i, b in enumerate(order):
            ax = axes.ravel()[i]
            chi = bundle.band_image(int(b), "chi", stat, model_floor=model_floor)
            vals = chi[np.isfinite(chi)]
            if vals.size == 0:
                ax.text(0.5, 0.5, "no valid pixels", ha="center", transform=ax.transAxes)
                continue
            clipped = np.clip(vals, -CHI_RANGE + 1e-9, CHI_RANGE - 1e-9)
            ax.hist(
                clipped,
                bins=edges,
                density=True,
                color=cols[b],
                alpha=0.75,
                histtype="stepfilled",
                lw=0,
            )
            ax.plot(xs, gauss, color=OBS_COLOR, lw=1.0, ls="--", label="N(0, 1)")
            rms = float(np.sqrt(np.mean(vals**2)))
            out3 = 100.0 * float(np.mean(np.abs(vals) > 3))
            ax.set_title(
                f"{bundle.short_names[b]}: rms {rms:.2f}, |χ|>3: {out3:.1f}%", color=cols[b]
            )
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1.0)
            ax.set_xlim(-CHI_RANGE, CHI_RANGE)
            if i % ncols == 0:
                ax.set_ylabel("density")
            if i >= nb - ncols:
                ax.set_xlabel(r"$\chi$")
                ax.tick_params(labelbottom=True)
        axes.ravel()[0].legend(loc="upper left")
        fig.suptitle(
            f"{bundle.title + ': ' if bundle.title else ''}per-pixel χ "
            f"({_floor_text(model_floor)}{', MAP model' if not bundle.has_draws else ''})"
        )
    return root_figure(fig)


def plot_chi2_per_band(
    bundle: ResultsBundle | str | Path,
    *,
    model_floor: bool = True,
    ax: Any = None,
    fig: Figure | SubFigure | None = None,
) -> Figure:
    """Reduced chi^2 per band against pivot wavelength.

    Filled points use the requested chi definition; open points show the other one (with or
    without the model-error floor) for comparison.

    Args:
        bundle: Bundle or path.
        model_floor: Main series includes the model-error floor.
        ax: Existing axes to draw into.
        fig: Figure or SubFigure to draw into (ignored when ``ax`` is given).
    """
    bundle = as_bundle(bundle)
    lam = np.asarray(bundle.pivot_um, dtype=float)
    x = np.where(np.isfinite(lam), lam, np.nan)
    cols = band_colors(lam)
    main = bundle.chi2_red_per_band(model_floor=model_floor)
    other = bundle.chi2_red_per_band(model_floor=not model_floor)
    with style_context():
        if ax is None:
            fig = new_figure(fig, (5.6, 3.6))
            ax = fig.add_subplot()
        else:
            fig = ax.figure
        ax.axhline(1.0, color="0.4", lw=0.8, ls="--", zorder=1)
        order = np.argsort(x)
        ax.plot(
            x[order],
            other[order],
            color="0.6",
            lw=0.8,
            marker="o",
            mfc="white",
            mec="0.45",
            ms=5,
            zorder=2,
            label=_floor_text(not model_floor),
        )
        ax.plot(x[order], main[order], color="0.3", lw=0.9, zorder=3)
        ax.scatter(
            x,
            main,
            c=cols,
            s=40,
            edgecolors=OBS_COLOR,
            lw=0.8,
            zorder=4,
            label=_floor_text(model_floor),
        )
        vals = np.concatenate([main, other])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size and (vals.max() / max(vals.min(), 1e-12) > 30):
            ax.set_yscale("log")
        else:
            top = max(2.0, 1.15 * float(vals.max())) if vals.size else 2.0
            ax.set_ylim(0.0, top)
        ax.set_ylabel(r"$\chi^2_{\rm red}$ (per pixel, median model)")
        set_micron_axis(ax, lam)
        add_band_axis(ax, lam, bundle.band_names)
        tot = bundle.metrics.get("chi2_red") if isinstance(bundle.metrics, dict) else None
        title = r"reduced $\chi^2$ per band"
        if isinstance(tot, (int, float)) and np.isfinite(tot):
            title += rf" (fit: $\chi^2_{{\rm red}}$ = {tot:.3f})"
        ax.set_title(title, pad=24)
        ax.legend(loc="best")
    return root_figure(fig)
