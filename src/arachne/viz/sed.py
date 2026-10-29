"""SED figures: observed aperture photometry against the posterior model, with residual panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D

from ..results.reader import ResultsBundle
from ..results.regions import (
    Region,
    RegionSED,
    component_region_seds,
    component_seds,
    region_sed,
    total_sed,
)
from .style import (
    INTRINSIC_COLOR,
    MAP_COLOR,
    MODEL_COLOR,
    OBS_COLOR,
    TRUTH_COLOR,
    add_band_axis,
    as_bundle,
    band_colors,
    new_figure,
    note_figure,
    region_color,
    root_figure,
    set_log_flux_axis,
    set_micron_axis,
    style_context,
)

__all__ = [
    "plot_sed",
    "plot_total_sed",
    "plot_region_seds",
    "plot_component_seds",
    "resolve_regions",
]

RESIDUAL_KINDS = ("sigma", "frac")
INTRINSIC_LABEL = "intrinsic (unconvolved, not aperture-comparable)"


# ---------------------------------------------------------------------------- helpers


def resolve_regions(
    bundle: ResultsBundle,
    regions: Mapping[str, Region | np.ndarray] | Sequence[str] | None = None,
) -> dict[str, Region]:
    """Normalise a region specification to ``{name: Region}``.

    Args:
        bundle: The bundle (source of named regions and the image shape).
        regions: ``None`` (every bundle region, or all pixels when there are none), a sequence
            of bundle region names (unknown names raise KeyError), or a mapping of name to
            :class:`Region` / (H, W) boolean mask.

    Returns:
        Ordered dict of regions.
    """
    if regions is None:
        out = dict(bundle.regions)
        if not out:
            out = {"all": Region.from_mask(np.ones(bundle.image_shape, bool), name="all")}
        return out
    if isinstance(regions, Mapping):
        return {
            name: r if isinstance(r, Region) else Region.from_mask(np.asarray(r), name=name)
            for name, r in regions.items()
        }
    out = {}
    for name in regions:
        if name not in bundle.regions:
            raise KeyError(f"Region {name!r} not in bundle (have {sorted(bundle.regions)})")
        out[name] = bundle.regions[name]
    return out


def _has(arr: np.ndarray | None) -> bool:
    return arr is not None and len(arr) > 0


def _model_band(
    ax: Any,
    lam: np.ndarray,
    sed: RegionSED,
    frame: str,
    color: Any,
    label: str,
    *,
    ls: str = "-",
    marker: str = "o",
    alpha: float = 0.25,
) -> list[np.ndarray]:
    """Draw p50 line (+ 16-84 band with draws) for one frame; return the plotted arrays."""
    draws = sed.model_draws if frame == "convolved" else sed.intrinsic_draws
    p16, p50, p84 = sed.model_percentiles((16, 50, 84), frame=frame)
    order = np.argsort(lam)
    x = lam[order]
    if _has(draws):
        ax.fill_between(x, p16[order], p84[order], color=color, alpha=alpha, lw=0, zorder=1)
        lab = f"{label}: median, 16–84%"
    else:
        lab = f"{label} (MAP)"
    ax.plot(x, p50[order], ls=ls, marker=marker, ms=4, color=color, label=lab, zorder=3)
    return [p16, p50, p84] if _has(draws) else [p50]


def _set_log_limits(ax: Any, arrays: Sequence[np.ndarray]) -> bool:
    vals = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays if a is not None])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return False
    lo, hi = np.log10(vals.min()), np.log10(vals.max())
    pad = max(0.12, 0.08 * (hi - lo))
    ax.set_yscale("log")
    ax.set_ylim(10 ** (lo - pad), 10 ** (hi + pad))
    set_log_flux_axis(ax)
    return True


def _obs_points(ax: Any, lam: np.ndarray, sed: RegionSED, logy: bool, cols: Sequence) -> None:
    """Observed fluxes with 1-sigma errors; band-coloured faces, black edges."""
    f = np.asarray(sed.obs_flux, dtype=float)
    e = np.asarray(sed.obs_err, dtype=float)
    ok = np.isfinite(f)
    det = ok & ((f > 0) | (not logy))
    ax.errorbar(
        lam[det],
        f[det],
        yerr=e[det],
        fmt="none",
        ecolor=OBS_COLOR,
        elinewidth=1.1,
        capsize=2.5,
        zorder=5,
    )
    ax.scatter(
        lam[det],
        f[det],
        s=34,
        c=[cols[i] for i in np.flatnonzero(det)],
        edgecolors=OBS_COLOR,
        linewidths=0.9,
        zorder=6,
        label="observed (±1σ)",
    )
    if logy:
        # non-positive fluxes cannot be drawn on a log axis: show a 2-sigma upper limit
        lim = ok & (f <= 0) & np.isfinite(e) & (f + 2 * e > 0)
        if lim.any():
            ax.scatter(
                lam[lim],
                (f + 2 * e)[lim],
                marker="v",
                s=36,
                facecolors="none",
                edgecolors=OBS_COLOR,
                zorder=6,
                label="observed 2σ upper limit",
            )


def _residual_axes_style(ax: Any, ylabel: str) -> None:
    ax.axhline(0.0, color="0.25", lw=0.8, zorder=1)
    ax.set_ylabel(ylabel)


# ---------------------------------------------------------------------------- plot_sed


def plot_sed(
    sed: RegionSED,
    *,
    fig: Figure | SubFigure | None = None,
    show_intrinsic: bool = False,
    show_truth: bool = True,
    show_map: bool = True,
    residuals: Sequence[str] = RESIDUAL_KINDS,
    logy: bool = True,
    color: Any = None,
    label_bands: bool = True,
    title: str | None = None,
) -> Figure:
    """Observed SED of one region against the posterior model, with residual panels.

    Top panel: observed fluxes with 1-sigma errors (band-coloured points), the PSF-convolved
    model summed over the same pixels (median line, 16-84% band when draws exist), MAP model
    as open diamonds, optionally the intrinsic (unconvolved) model as a dashed line, and the
    noise-free truth as crosses.  Band names label a twin top axis.

    Residual panels (only when the SED has observed fluxes):

    - ``"sigma"``: exactly :meth:`RegionSED.chi` (observed minus model median over the
      observed error and the model 16-84 half-width in quadrature), with ±1 and ±3 guides.
    - ``"frac"``: ``100 (obs - median) / median`` with the observed error bars and the model
      16-84% band around zero.

    SEDs without a convolved model (e.g. from :func:`~arachne.results.regions.component_seds`)
    are drawn from their intrinsic model instead.

    Args:
        sed: The region SED.
        fig: Figure or SubFigure to draw into (a new one when None).
        show_intrinsic: Overlay the intrinsic (unconvolved) model when available.
        show_truth: Draw the truth when available.
        show_map: Draw the MAP model when draws exist (without draws the line is the MAP).
        residuals: Residual panels, any of ``"sigma"``, ``"frac"`` (in that order).
        logy: Log flux axis (falls back to linear when nothing is positive).
        color: Model colour (default blue).
        label_bands: Add the band-name top axis.
        title: Panel title (default ``sed.label`` and the pixel count).

    Returns:
        The (root) Figure.
    """
    bad = [r for r in residuals if r not in RESIDUAL_KINDS]
    if bad:
        raise ValueError(f"residuals must be among {RESIDUAL_KINDS}, got {bad}")
    has_obs = sed.obs_flux is not None and sed.obs_err is not None
    has_conv = sed.model_map is not None or _has(sed.model_draws)
    has_int = sed.intrinsic_map is not None or _has(sed.intrinsic_draws)
    if not (has_conv or has_int or has_obs):
        raise ValueError("RegionSED has neither observations nor a model")
    residuals = [r for r in RESIDUAL_KINDS if r in residuals] if has_obs and has_conv else []
    mcol = color or MODEL_COLOR
    lam = np.asarray(sed.pivot_um, dtype=float)
    cols = band_colors(lam)

    with style_context():
        n_res = len(residuals)
        fig = new_figure(fig, (6.0, 4.0 + 1.3 * n_res))
        gs = fig.add_gridspec(1 + n_res, 1, height_ratios=[3.2] + [1.0] * n_res)
        ax = fig.add_subplot(gs[0])
        res_axes = [fig.add_subplot(gs[i + 1], sharex=ax) for i in range(n_res)]
        plotted: list[np.ndarray] = []

        if has_conv:
            plotted += _model_band(ax, lam, sed, "convolved", mcol, "model")
            if show_map and _has(sed.model_draws) and sed.model_map is not None:
                ax.scatter(
                    lam,
                    sed.model_map,
                    marker="D",
                    s=28,
                    facecolors="none",
                    edgecolors=MAP_COLOR,
                    linewidths=1.0,
                    zorder=4,
                    label="model (MAP)",
                )
                plotted.append(sed.model_map)
        if has_int and (show_intrinsic or not has_conv):
            int_col = INTRINSIC_COLOR if has_conv else mcol
            int_label = INTRINSIC_LABEL if has_conv else "intrinsic model"
            plotted += _model_band(
                ax, lam, sed, "intrinsic", int_col, int_label, ls="--", marker="s", alpha=0.15
            )
        if show_truth:
            if sed.truth_convolved is not None and has_conv:
                ax.scatter(
                    lam,
                    sed.truth_convolved,
                    marker="x",
                    s=40,
                    color=TRUTH_COLOR,
                    lw=1.4,
                    zorder=7,
                    label="truth (convolved, noise-free)",
                )
                plotted.append(sed.truth_convolved)
            if sed.truth_intrinsic is not None and (show_intrinsic or not has_conv):
                ax.scatter(
                    lam,
                    sed.truth_intrinsic,
                    marker="+",
                    s=55,
                    color=TRUTH_COLOR,
                    lw=1.4,
                    zorder=7,
                    label="truth (intrinsic)",
                )
                plotted.append(sed.truth_intrinsic)
        if has_obs:
            f, e = np.asarray(sed.obs_flux, float), np.asarray(sed.obs_err, float)
            plotted += [f + e, np.where(f - e > 0, f - e, np.nan)]
            _obs_points(ax, lam, sed, logy, cols)

        if not (logy and _set_log_limits(ax, plotted)):
            ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_ylabel("flux density [nJy]")
        set_micron_axis(ax, lam, label=n_res == 0)
        ax.legend(loc="best", ncol=1, handlelength=1.8)
        if title is None:
            n = int(np.max(sed.n_pix)) if np.size(sed.n_pix) else 0
            title = f"{sed.label}  ({n} px)" if n else sed.label
        if label_bands:
            add_band_axis(ax, lam, sed.band_names)
            ax.set_title(title, pad=24 if len(lam) else 6)
        else:
            ax.set_title(title)

        if n_res:
            p16, p50, p84 = sed.model_percentiles((16, 50, 84))
            has_draws = _has(sed.model_draws)
        for kind, rax in zip(residuals, res_axes):
            if kind == "sigma":
                chi = sed.chi()
                rax.axhspan(-1, 1, color="0.88", lw=0, zorder=0)
                for v in (-3, 3):
                    rax.axhline(v, color="0.45", lw=0.7, ls=":", zorder=1)
                _residual_axes_style(rax, r"$\chi$ [$\sigma$]")
                ok = np.isfinite(chi)
                rax.scatter(
                    lam[ok],
                    chi[ok],
                    c=[cols[i] for i in np.flatnonzero(ok)],
                    s=30,
                    edgecolors=OBS_COLOR,
                    linewidths=0.8,
                    zorder=5,
                )
                m = np.nanmax(np.abs(chi)) if ok.any() else 0.0
                lim = max(3.8, 1.2 * m)
                rax.set_ylim(-lim, lim)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = 100.0 * sed.frac_residual()
                    ferr = 100.0 * np.asarray(sed.obs_err, float) / np.abs(p50)
                    lo_b, hi_b = 100.0 * (p16 / p50 - 1.0), 100.0 * (p84 / p50 - 1.0)
                order = np.argsort(lam)
                if has_draws:
                    rax.fill_between(
                        lam[order],
                        lo_b[order],
                        hi_b[order],
                        color=mcol,
                        alpha=0.25,
                        lw=0,
                        zorder=1,
                    )
                _residual_axes_style(rax, "Δ/model [%]")
                ok = np.isfinite(frac)
                rax.errorbar(
                    lam[ok],
                    frac[ok],
                    yerr=np.where(np.isfinite(ferr), ferr, 0)[ok],
                    fmt="none",
                    ecolor=OBS_COLOR,
                    elinewidth=1.0,
                    capsize=2,
                    zorder=4,
                )
                rax.scatter(
                    lam[ok],
                    frac[ok],
                    c=[cols[i] for i in np.flatnonzero(ok)],
                    s=30,
                    edgecolors=OBS_COLOR,
                    linewidths=0.8,
                    zorder=5,
                )
                span = np.abs(np.concatenate([frac + ferr, frac - ferr, lo_b, hi_b]))
                span = span[np.isfinite(span)]
                lim = max(5.0, 1.15 * float(span.max())) if span.size else 5.0
                rax.set_ylim(-lim, lim)
        for a in (ax, *res_axes[:-1]):
            a.tick_params(labelbottom=False)
        if res_axes:
            set_micron_axis(res_axes[-1], lam)
        fig.align_ylabels([ax, *res_axes])
    return root_figure(fig)


# ---------------------------------------------------------------------------- bundle-level


def plot_total_sed(
    bundle: ResultsBundle | str | Path,
    *,
    error_scale: float | np.ndarray = 1.0,
    model_floor: bool = False,
    show_intrinsic: bool | None = None,
    **kw: Any,
) -> Figure:
    """SED of the total region (``segment``, else ``all``, else every pixel).

    Args:
        bundle: Bundle or path.
        error_scale: Passed to :func:`~arachne.results.regions.region_sed`.
        model_floor: Passed to :func:`~arachne.results.regions.region_sed`.
        show_intrinsic: Overlay the intrinsic model (default: when the bundle has one).
        **kw: Passed to :func:`plot_sed`.
    """
    bundle = as_bundle(bundle)
    sed = total_sed(bundle, error_scale=error_scale, model_floor=model_floor)
    if show_intrinsic is None:
        show_intrinsic = bundle.has_intrinsic
    if "title" not in kw and bundle.title:
        n = int(np.max(sed.n_pix))
        kw["title"] = f"{bundle.title}: {sed.label}, {n} px, {_source_text(bundle)}"
    return plot_sed(sed, show_intrinsic=show_intrinsic, **kw)


def _source_text(bundle: ResultsBundle) -> str:
    if bundle.has_draws:
        return f"{bundle.posterior_source}, S={bundle.n_draws}"
    return f"{bundle.posterior_source}, MAP only"


def plot_region_seds(
    bundle: ResultsBundle | str | Path,
    regions: Mapping[str, Region | np.ndarray] | Sequence[str] | None = None,
    layout: str = "grid",
    *,
    error_scale: float | np.ndarray = 1.0,
    model_floor: bool = False,
    fig: Figure | SubFigure | None = None,
    **kw: Any,
) -> Figure:
    """SEDs of several regions.

    Args:
        bundle: Bundle or path.
        regions: Mapping of name to Region/mask, a list of bundle region names, or None for
            every region stored in the bundle.
        layout: ``"grid"`` (one :func:`plot_sed` panel per region) or ``"overlay"`` (all
            regions on one axis in region colours, with a sigma-residual panel).
        error_scale: Passed to :func:`~arachne.results.regions.region_sed`.
        model_floor: Passed to :func:`~arachne.results.regions.region_sed`.
        fig: Figure or SubFigure to draw into.
        **kw: Passed to :func:`plot_sed` (grid layout only).
    """
    bundle = as_bundle(bundle)
    regs = resolve_regions(bundle, regions)
    seds = [
        region_sed(bundle, r, label=name, error_scale=error_scale, model_floor=model_floor)
        for name, r in regs.items()
    ]
    if layout == "grid":
        n = len(seds)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))
        n_res = len(kw.get("residuals", RESIDUAL_KINDS))
        with style_context():
            fig = new_figure(fig, (5.2 * ncols, (3.8 + 1.2 * n_res) * nrows))
            subs = np.atleast_1d(fig.subfigures(nrows, ncols, squeeze=False)).ravel()
            for i, (sub, sed) in enumerate(zip(subs, seds)):
                kw_i = dict(kw)
                kw_i.setdefault("color", region_color(i))
                plot_sed(sed, fig=sub, **kw_i)
            if bundle.title:
                fig.suptitle(f"{bundle.title}: region SEDs ({_source_text(bundle)})")
        return root_figure(fig)
    if layout != "overlay":
        raise ValueError(f"layout must be 'grid' or 'overlay', got {layout!r}")
    return _overlay_region_seds(bundle, seds, fig=fig)


def _overlay_region_seds(
    bundle: ResultsBundle, seds: list[RegionSED], *, fig: Figure | SubFigure | None
) -> Figure:
    lam = np.asarray(bundle.pivot_um, dtype=float)
    with style_context():
        fig = new_figure(fig, (6.4, 5.4))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.1])
        ax = fig.add_subplot(gs[0])
        rax = fig.add_subplot(gs[1], sharex=ax)
        plotted = []
        n = len(seds)
        # small multiplicative dithers so points of different regions do not overlap
        dith = 10 ** (0.012 * (np.arange(n) - (n - 1) / 2))
        for i, sed in enumerate(seds):
            c = region_color(i)
            plotted += _model_band(ax, lam, sed, "convolved", c, sed.label, alpha=0.18, marker="")
            f, e = np.asarray(sed.obs_flux, float), np.asarray(sed.obs_err, float)
            ok = np.isfinite(f) & (f > 0)
            ax.errorbar(
                lam[ok] * dith[i],
                f[ok],
                yerr=e[ok],
                fmt="o",
                ms=4.5,
                mfc="white",
                mec=c,
                ecolor=c,
                capsize=2,
                zorder=5,
            )
            plotted += [f + e]
            chi = sed.chi()
            rax.scatter(lam * dith[i], chi, color=c, s=22, edgecolors="k", lw=0.5, zorder=4)
        _set_log_limits(ax, plotted)
        ax.set_ylabel("flux density [nJy]")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], ls="none", marker="o", mfc="white", mec="0.2"))
        labels.append("observed (±1σ)")
        ax.legend(handles, labels, loc="best")
        set_micron_axis(ax, lam, label=False)
        add_band_axis(ax, lam, bundle.band_names)
        ax.set_title(f"{bundle.title or 'region SEDs'} ({_source_text(bundle)})", pad=24)
        rax.axhspan(-1, 1, color="0.88", lw=0, zorder=0)
        for v in (-3, 3):
            rax.axhline(v, color="0.45", lw=0.7, ls=":")
        _residual_axes_style(rax, r"$\chi$ [$\sigma$]")
        allchi = np.concatenate([s.chi() for s in seds])
        m = np.nanmax(np.abs(allchi)) if np.isfinite(allchi).any() else 0.0
        rax.set_ylim(-max(3.8, 1.2 * m), max(3.8, 1.2 * m))
        ax.tick_params(labelbottom=False)
        rax.set_xlabel(r"observed wavelength [$\mu$m]")
        fig.align_ylabels([ax, rax])
    return root_figure(fig)


def plot_component_seds(
    bundle: ResultsBundle | str | Path,
    *,
    include_total: bool = True,
    region: Region | np.ndarray | str | None = None,
    fig: Figure | SubFigure | None = None,
) -> Figure:
    """Per-component SEDs of an additive model.

    Left: the intrinsic (unconvolved, whole-image) SED of every component with its 16-84%
    band, the total in black, and the true component SEDs as crosses for mocks.  Right (when
    convolved component cubes are stored): the observed-frame SED of each component summed over
    ``region`` (default: the total region) against the observed photometry; the sum of the
    components plus sky is the full model.

    Args:
        bundle: Bundle or path.
        include_total: Draw the summed SED.
        region: Region (or bundle region name) for the observed-frame panel.
        fig: Figure or SubFigure to draw into.

    Returns:
        The Figure; a note figure when the bundle has no components.
    """
    bundle = as_bundle(bundle)
    seds = component_seds(bundle)
    if not seds:
        return note_figure("No components in this bundle (not an additive model).", fig=fig)
    comps = bundle.components
    lam = np.asarray(bundle.pivot_um, dtype=float)
    show_obs = comps.has_frame("convolved")
    with style_context():
        fig = new_figure(fig, (11.0 if show_obs else 6.0, 4.4))
        axes = fig.subplots(1, 2 if show_obs else 1, squeeze=False)[0]
        ax = axes[0]
        plotted = []
        comp_seds, tot = seds[:-1], seds[-1]
        for k, sed in enumerate(comp_seds):
            c = region_color(k)
            plotted += _model_band(ax, lam, sed, "intrinsic", c, sed.label, marker="o")
            if sed.truth_intrinsic is not None:
                ax.scatter(lam, sed.truth_intrinsic, marker="x", s=40, color=c, lw=1.4, zorder=6)
                plotted.append(sed.truth_intrinsic)
        if include_total:
            plotted += _model_band(ax, lam, tot, "intrinsic", "black", "total", alpha=0.12)
            if tot.truth_intrinsic is not None:
                ax.scatter(
                    lam, tot.truth_intrinsic, marker="x", s=40, color="black", lw=1.4, zorder=6
                )
                plotted.append(tot.truth_intrinsic)
        handles, labels = ax.get_legend_handles_labels()
        if any(s.truth_intrinsic is not None for s in seds):
            handles.append(Line2D([], [], ls="none", marker="x", color="0.3"))
            labels.append("truth (marker colour = component)")
        _set_log_limits(ax, plotted)
        ax.legend(handles, labels, loc="best")
        ax.set_ylabel("flux density [nJy]")
        set_micron_axis(ax, lam)
        add_band_axis(ax, lam, bundle.band_names)
        ax.set_title("intrinsic (unconvolved) component SEDs, whole image", pad=24)

        if show_obs:
            ax2 = axes[1]
            if isinstance(region, str):
                region = bundle.regions[region]
            tot_obs = (
                total_sed(bundle)
                if region is None
                else region_sed(bundle, region, label=getattr(region, "name", "region"))
            )
            reg = region if region is not None else _total_region(bundle)
            plotted = []
            for k, sed in enumerate(component_region_seds(bundle, reg)):
                c = region_color(k)
                plotted += _model_band(
                    ax2, lam, sed, "convolved", c, sed.label.split(" in ")[0], marker="o"
                )
            plotted += _model_band(
                ax2, lam, tot_obs, "convolved", "black", "full model (+sky)", alpha=0.12
            )
            f, e = np.asarray(tot_obs.obs_flux, float), np.asarray(tot_obs.obs_err, float)
            plotted += [f + e]
            _obs_points(ax2, lam, tot_obs, True, band_colors(lam))
            _set_log_limits(ax2, plotted)
            ax2.legend(loc="best")
            ax2.set_ylabel("flux density [nJy]")
            set_micron_axis(ax2, lam)
            add_band_axis(ax2, lam, bundle.band_names)
            ax2.set_title(f"observed frame (PSF-convolved), {tot_obs.label}", pad=24)
        if bundle.title:
            fig.suptitle(f"{bundle.title}: components ({_source_text(bundle)})")
    return root_figure(fig)


def _total_region(bundle: ResultsBundle) -> Region:
    for name in ("segment", "all"):
        if name in bundle.regions:
            return bundle.regions[name]
    return Region.from_mask(np.ones(bundle.image_shape, bool), name="all")
