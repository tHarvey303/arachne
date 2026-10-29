"""One-page summary of a bundle and a helper that writes every standard figure to disk."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from ..results.reader import ResultsBundle
from .diagnostics import plot_chi2_per_band, plot_chi_histograms
from .images import default_rgb_bands, plot_band_mosaic, plot_rgb
from .maps import param_map_style, plot_param_maps, plot_regions
from .sed import plot_component_seds, plot_region_seds, plot_total_sed, resolve_regions
from .style import as_bundle, param_label, stat_label, style_context

__all__ = ["plot_summary_page", "save_all", "summary_text"]


def summary_text(bundle: ResultsBundle) -> str:
    """One-line description: title, posterior source, draws, chi^2, degraded flag."""
    parts = [bundle.title or (bundle.path.name if bundle.path else "bundle")]
    parts.append(f"posterior: {bundle.posterior_source}")
    parts.append(f"S = {bundle.n_draws}" if bundle.has_draws else "MAP only")
    chi2 = bundle.metrics.get("chi2_red") if isinstance(bundle.metrics, dict) else None
    if isinstance(chi2, (int, float)) and np.isfinite(chi2):
        parts.append(rf"$\chi^2_{{\rm red}}$ = {chi2:.3f}")
    parts.append(f"{bundle.n_bands} bands, {bundle.image_shape[0]}×{bundle.image_shape[1]} px")
    if np.isfinite(bundle.redshift):
        parts.append(f"z = {bundle.redshift:g}")
    if bundle.degraded:
        parts.append("DEGRADED (no model draws)")
    return "  |  ".join(parts)


def _param_row(fig, bundle: ResultsBundle) -> None:
    names = list(bundle.map_names)[:5]
    stat = "p50" if "p50" in bundle.available_param_stats() else "map"
    region = bundle.regions.get("segment")
    axes = np.atleast_1d(fig.subplots(1, len(names), squeeze=False)[0])
    for ax, name in zip(axes, names):
        img = np.asarray(bundle.param_map(name, stat), dtype=float)
        if region is not None:
            img = np.where(region.mask, img, np.nan)
        im = ax.imshow(img, **param_map_style(stat, img, None))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
        ax.set_title(f"{param_label(bundle, name)} ({stat_label(stat)})")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        cb.ax.minorticks_off()


def _region_name(bundle: ResultsBundle) -> str:
    for name in ("segment", "all"):
        if name in bundle.regions:
            return f"region '{name}'"
    return "all pixels"


def plot_summary_page(bundle: ResultsBundle | str | Path) -> Figure:
    """A one-page overview of a fit.

    Rows: RGB composites of data, model and residual; the total SED with sigma and fractional
    residuals next to the reduced chi^2 per band; the median parameter maps (or, for additive
    models, the component SEDs).

    Args:
        bundle: Bundle or path.
    """
    bundle = as_bundle(bundle)
    has_bottom = bool(bundle.map_names) or bundle.has_components
    with style_context():
        fig = Figure(figsize=(14.0, 15.0 if has_bottom else 11.0), layout="constrained")
        ratios = [1.0, 1.75] + ([1.2] if has_bottom else [])
        rows = fig.subfigures(len(ratios), 1, height_ratios=ratios)
        plot_rgb(bundle, fig=rows[0])
        names = [bundle.short_names[b] for b in default_rgb_bands(bundle)]
        rows[0].suptitle(f"RGB = {' / '.join(names)} (shared asinh scale)")
        mid = rows[1].subfigures(1, 2, width_ratios=[1.6, 1.0])
        plot_total_sed(bundle, fig=mid[0], title="")
        mid[0].suptitle(f"total SED, {_region_name(bundle)}")
        plot_chi2_per_band(bundle, fig=mid[1])
        if bundle.map_names:
            _param_row(rows[2], bundle)
        elif bundle.has_components:
            plot_component_seds(bundle, fig=rows[2])
            rows[2].suptitle("components")
        fig.suptitle(summary_text(bundle), fontsize=11, fontweight="bold")
    return fig


_ALL = (
    "summary",
    "total_sed",
    "region_seds",
    "component_seds",
    "param_maps",
    "regions",
    "band_mosaic",
    "rgb",
    "chi_histograms",
    "chi2_per_band",
)


def save_all(
    bundle: ResultsBundle | str | Path,
    outdir: str | Path,
    fmt: str = "png",
    dpi: int = 150,
    *,
    regions: Sequence[str] | None = None,
) -> list[Path]:
    """Write every applicable standard figure to ``outdir`` as ``<name>.<fmt>``.

    Figures: ``summary``, ``total_sed``, ``region_seds``, ``component_seds`` (additive models
    only), ``param_maps`` (when the bundle has parameter maps), ``regions``, ``band_mosaic``,
    ``rgb``, ``chi_histograms`` and ``chi2_per_band``.  A figure that fails is reported as a
    warning and skipped, so one bad panel never loses the rest.

    Args:
        bundle: Bundle or path.
        outdir: Output directory (created).
        fmt: File format (``png``, ``pdf``, ``svg``).
        dpi: Raster resolution.
        regions: Bundle region names for ``region_seds`` and ``regions`` (default: all
            stored regions); unknown names are skipped with a warning.

    Returns:
        Paths written, in the order above.
    """
    bundle = as_bundle(bundle)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reg_names = None
    if regions is not None:
        reg_names = [r for r in regions if r in bundle.regions]
        missing = [r for r in regions if r not in bundle.regions]
        if missing:
            warnings.warn(
                f"Regions {missing} not in bundle (have {sorted(bundle.regions)}); skipped",
                stacklevel=2,
            )
        if not reg_names:
            reg_names = None
    regs = resolve_regions(bundle, reg_names)

    makers = {
        "summary": lambda: plot_summary_page(bundle),
        "total_sed": lambda: plot_total_sed(bundle),
        "region_seds": lambda: plot_region_seds(bundle, regs),
        "component_seds": lambda: plot_component_seds(bundle),
        "param_maps": lambda: plot_param_maps(bundle),
        "regions": lambda: plot_regions(bundle, regs),
        "band_mosaic": lambda: plot_band_mosaic(bundle),
        "rgb": lambda: plot_rgb(bundle),
        "chi_histograms": lambda: plot_chi_histograms(bundle),
        "chi2_per_band": lambda: plot_chi2_per_band(bundle),
    }
    skip = set()
    if not bundle.has_components:
        skip.add("component_seds")
    if not bundle.map_names:
        skip.add("param_maps")
    written = []
    for name in _ALL:
        if name in skip:
            continue
        try:
            fig = makers[name]()
        except Exception as err:  # noqa: BLE001 - one failing figure must not stop the rest
            warnings.warn(f"Figure {name!r} failed: {err!r}", stacklevel=2)
            continue
        path = outdir / f"{name}.{fmt}"
        with style_context():
            fig.savefig(path, dpi=dpi, format=fmt)
        written.append(path)
    return written
