"""Static matplotlib figures for ``arachne.results`` bundles (no JAX).

Every function takes a :class:`~arachne.results.ResultsBundle` (or a path to one) and returns a
:class:`matplotlib.figure.Figure` built without pyplot, so nothing is shown or kept alive
unless you ask: call ``fig.savefig(...)`` in scripts; Jupyter displays returned figures.
Missing products (draws, truth, intrinsic model, components) are left out of the figures.
"""

from __future__ import annotations

from . import style
from .diagnostics import plot_chi2_per_band, plot_chi_histograms
from .images import plot_band_mosaic, plot_rgb
from .maps import plot_param_maps, plot_regions
from .sed import plot_component_seds, plot_region_seds, plot_sed, plot_total_sed
from .style import band_colors, region_color
from .summary import plot_summary_page, save_all

__all__ = [
    "plot_sed",
    "plot_total_sed",
    "plot_region_seds",
    "plot_component_seds",
    "plot_param_maps",
    "plot_regions",
    "plot_band_mosaic",
    "plot_rgb",
    "plot_chi_histograms",
    "plot_chi2_per_band",
    "plot_summary_page",
    "save_all",
    "band_colors",
    "region_color",
    "style",
]
