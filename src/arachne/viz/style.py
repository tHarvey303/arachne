"""Shared plotting style: band and region colours, rc defaults, labels and small helpers.

Every figure in :mod:`arachne.viz` is a plain :class:`matplotlib.figure.Figure` (no pyplot state),
drawn inside :func:`style_context` so the package never changes the user's global rcParams.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import FuncNorm, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure, SubFigure
from matplotlib.ticker import FixedLocator, NullLocator

from ..results.metadata import short_band_name
from ..results.reader import ResultsBundle, load_bundle

__all__ = [
    "BAND_CMAP",
    "REGION_COLORS",
    "OBS_COLOR",
    "MODEL_COLOR",
    "MAP_COLOR",
    "INTRINSIC_COLOR",
    "TRUTH_COLOR",
    "RC_PARAMS",
    "style_context",
    "band_colors",
    "region_color",
    "as_bundle",
    "new_figure",
    "root_figure",
    "param_label",
    "stat_label",
    "band_labels",
    "stretch_norm",
    "set_micron_axis",
    "add_band_axis",
    "set_log_flux_axis",
    "PlainLogFormatter",
    "draw_mask_outline",
    "note_figure",
    "image_kw",
]

#: Blue -> red ramp used for band colours (log pivot wavelength); readable on white throughout.
BAND_CMAP = LinearSegmentedColormap.from_list(
    "arachne_bands", ["#3b4cc0", "#2a8fbd", "#2a9d6f", "#c99a1e", "#e0602b", "#b40426"]
)
#: Okabe-Ito order (colour-blind safe), assigned to regions in a fixed order.
REGION_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)
OBS_COLOR = "black"
MODEL_COLOR = "#0072B2"
MAP_COLOR = "#0b3d66"
INTRINSIC_COLOR = "#D55E00"
TRUTH_COLOR = "#009E73"

RC_PARAMS: dict[str, Any] = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
    "figure.dpi": 100,
    "mathtext.fontset": "dejavusans",
}


@contextlib.contextmanager
def style_context(**overrides: Any) -> Iterator[None]:
    """Context manager applying :data:`RC_PARAMS` (plus ``overrides``) temporarily."""
    with mpl.rc_context({**RC_PARAMS, **overrides}):
        yield


def band_colors(pivot_um: Sequence[float] | np.ndarray) -> list[tuple[float, ...]]:
    """One colour per band, blue to red by log pivot wavelength.

    The ramp is stretched over the bands given, so the bluest band is always blue and the
    reddest red.  Bands with an unknown (NaN) pivot are grey.

    Args:
        pivot_um: (B,) pivot wavelengths in micron.

    Returns:
        List of B RGBA tuples.
    """
    lam = np.asarray(pivot_um, dtype=float)
    out: list[tuple[float, ...]] = [(0.5, 0.5, 0.5, 1.0)] * lam.size
    good = np.isfinite(lam) & (lam > 0)
    if not good.any():
        return out
    loglam = np.log10(lam[good])
    lo, hi = loglam.min(), loglam.max()
    t = (loglam - lo) / (hi - lo) if hi > lo else np.full(loglam.shape, 0.5)
    cols = BAND_CMAP(t)
    for i, c in zip(np.flatnonzero(good), cols):
        out[i] = tuple(c)
    return out


def region_color(i: int) -> str:
    """Colour of the ``i``-th region (fixed Okabe-Ito order, wraps after 8)."""
    return REGION_COLORS[i % len(REGION_COLORS)]


def as_bundle(bundle: ResultsBundle | str | Path) -> ResultsBundle:
    """Return ``bundle`` itself, or open it with :func:`load_bundle` when given a path."""
    if isinstance(bundle, (str, Path)):
        return load_bundle(bundle)
    return bundle


def new_figure(fig: Figure | SubFigure | None, figsize: tuple[float, float]) -> Figure | SubFigure:
    """Return ``fig`` or a new constrained-layout :class:`Figure` of ``figsize`` inches."""
    if fig is not None:
        return fig
    return Figure(figsize=figsize, layout="constrained")


def root_figure(fig: Figure | SubFigure) -> Figure:
    """The top-level :class:`Figure` owning ``fig`` (itself for a Figure)."""
    if isinstance(fig, SubFigure):
        return fig.get_figure(root=True)
    return fig


def param_label(bundle: ResultsBundle, name: str, *, units: bool = True) -> str:
    """Display label of a parameter, with its unit in brackets when known."""
    label = bundle.labels.get(name, name)
    unit = bundle.units.get(name, "") if units else ""
    return f"{label} [{unit}]" if unit else label


_STAT_LABELS = {
    "map": "MAP",
    "p16": "16th pct",
    "p50": "median",
    "p84": "84th pct",
    "mean": "mean",
    "std": r"std $\sigma$",
    "truth": "truth",
    "diff": "median $-$ truth",
    "z": r"(median $-$ truth) / $\sigma$",
    "information": r"information $1-(\sigma/\sigma_{\rm prior})^2$",
    "prior_std": r"prior $\sigma$",
}


def stat_label(stat: str) -> str:
    """Human-readable name of a parameter or model statistic."""
    return _STAT_LABELS.get(stat, stat)


def band_labels(band_names: Sequence[str]) -> list[str]:
    """Short filter names (``"JWST/NIRCam.F115W"`` -> ``"F115W"``)."""
    return [short_band_name(b) for b in band_names]


def stretch_norm(kind: str, lo: float, hi: float, *, asinh_a: float = 0.1) -> Normalize | FuncNorm:
    """A matplotlib norm matching :func:`arachne.results.display.stretch`.

    Unlike pre-stretched arrays, the norm keeps the colourbar in physical units.

    Args:
        kind: ``linear``, ``asinh``, ``log`` or ``sqrt``.
        lo: Lower limit.
        hi: Upper limit.
        asinh_a: Softening of the asinh stretch.
    """
    span = hi - lo if hi > lo else 1.0
    if kind == "linear":
        return Normalize(lo, hi, clip=True)
    if kind == "asinh":
        fwd = lambda x: np.arcsinh((x - lo) / (span * asinh_a))  # noqa: E731
        inv = lambda y: lo + span * asinh_a * np.sinh(y)  # noqa: E731
    elif kind == "log":
        fwd = lambda x: np.log10(1.0 + 1000.0 * np.clip((x - lo) / span, 0, None))  # noqa: E731
        inv = lambda y: lo + span * (10.0**y - 1.0) / 1000.0  # noqa: E731
    elif kind == "sqrt":
        fwd = lambda x: np.sqrt(np.clip((x - lo) / span, 0, None))  # noqa: E731
        inv = lambda y: lo + span * y**2  # noqa: E731
    else:
        raise ValueError(f"stretch must be linear, asinh, log or sqrt, got {kind!r}")
    return FuncNorm((fwd, inv), vmin=lo, vmax=hi, clip=True)


def image_kw(cmap: str | Any, norm: Normalize) -> dict[str, Any]:
    """Keyword arguments for ``imshow`` with NaN shown in light grey."""
    cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    cm = cm.with_extremes(bad="0.85")
    return {"cmap": cm, "norm": norm, "origin": "lower", "interpolation": "nearest"}


_NICE_MICRON = (0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30)
_NICE_MICRON_FINE = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10)


def set_micron_axis(ax: Any, pivots: np.ndarray, *, label: bool = True) -> None:
    """Log-scaled wavelength x axis in micron with plain-number ticks and padded limits."""
    lam = np.asarray(pivots, dtype=float)
    lam = lam[np.isfinite(lam) & (lam > 0)]
    ax.set_xscale("log")
    if lam.size:
        lo, hi = np.log10(lam.min()), np.log10(lam.max())
        pad = max(0.06, 0.07 * (hi - lo))
        ax.set_xlim(10 ** (lo - pad), 10 ** (hi + pad))
        xmin, xmax = 10 ** (lo - pad), 10 ** (hi + pad)
        nice = _NICE_MICRON_FINE if xmax / xmin < 6 else _NICE_MICRON
        ticks = [t for t in nice if xmin <= t <= xmax]
        if len(ticks) < 2:
            ticks = sorted({float(f"{v:.2g}") for v in (lam.min(), lam.max())})
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if label:
        ax.set_xlabel(r"observed wavelength [$\mu$m]")


class PlainLogFormatter(mpl.ticker.LogFormatterSciNotation):
    """Log-axis formatter that labels the same ticks as matplotlib but as plain numbers."""

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Format ``x`` as ``600`` rather than ``6 x 10^2`` (blank where matplotlib is)."""
        s = super().__call__(x, pos)
        return f"{x:.6g}" if s else ""


def set_log_flux_axis(ax: Any) -> None:
    """Plain-number tick labels on a log y axis."""
    ax.yaxis.set_major_formatter(PlainLogFormatter())
    ax.yaxis.set_minor_formatter(
        PlainLogFormatter(labelOnlyBase=False, minor_thresholds=(1.2, 0.6))
    )


_BAND_FS = 7.0


def _thin_band_labels(ax: Any, lam: np.ndarray, names: list[str]) -> tuple[list[bool], bool]:
    """Decide which band labels to show and whether to rotate them.

    Labels are rotated when horizontal ones would collide; if even vertical labels collide,
    labels are dropped greedily (wide ``W`` bands first) so that no two shown labels overlap.
    The ticks themselves are always drawn.  Sizes are estimated before layout, with margin.
    """
    n = len(lam)
    if n < 2:
        return [True] * n, False
    x0, x1 = (np.log10(v) for v in ax.get_xlim())
    root = ax.get_figure(root=True)
    width_in = 0.85 * ax.get_window_extent().width / root.dpi  # constrained layout shrinks
    pos = (np.log10(lam) - x0) / (x1 - x0) * max(width_in, 1e-3)
    char_in = _BAND_FS * 0.62 / 72
    widest = max(len(s) for s in names) * char_in
    gaps = np.diff(np.sort(pos))
    if gaps.min() >= 1.1 * widest:
        return [True] * n, False
    need = _BAND_FS * 1.25 / 72  # height of a rotated label
    if gaps.min() >= need:
        return [True] * n, True
    keep = [False] * n
    priority = sorted(range(n), key=lambda i: (not names[i].upper().endswith("W"), lam[i]))
    for i in priority:
        if all(abs(pos[i] - pos[j]) >= need for j in range(n) if keep[j]):
            keep[i] = True
    return keep, True


def add_band_axis(
    ax: Any, pivots: np.ndarray, band_names: Sequence[str], *, colors: Sequence | None = None
) -> Any:
    """Twin top axis with one tick per band, labelled with its short name.

    Labels are rotated vertically when they would otherwise collide, and thinned (ticks kept)
    when even vertical labels would; their colours follow :func:`band_colors`.

    Returns:
        The secondary axis.
    """
    lam = np.asarray(pivots, dtype=float)
    names = band_labels(band_names)
    good = np.isfinite(lam) & (lam > 0)
    lam_g = lam[good]
    names_g = [n for n, g in zip(names, good) if g]
    shown = _thin_band_labels(ax, lam_g, names_g)
    sec = ax.secondary_xaxis("top")
    sec.xaxis.set_major_locator(FixedLocator(lam_g))
    sec.xaxis.set_major_formatter(
        mpl.ticker.FixedFormatter([n if k else "" for n, k in zip(names_g, shown[0])])
    )
    sec.xaxis.set_minor_locator(NullLocator())
    sec.tick_params(axis="x", direction="in", length=3, pad=2, labelsize=_BAND_FS)
    if shown[1]:
        sec.tick_params(axis="x", labelrotation=90)
    cols = list(colors) if colors is not None else band_colors(lam)
    cols_g = [c for c, g in zip(cols, good) if g]
    # ticks are created once for the fixed locator and reused, so per-label colours persist
    for tick, c in zip(sec.xaxis.get_major_ticks(len(lam_g)), cols_g):
        tick.label1.set_color(c)
        tick.label2.set_color(c)
    return sec


def draw_mask_outline(ax: Any, mask: np.ndarray, color: Any, lw: float = 1.4, **kw: Any) -> Any:
    """Outline a boolean pixel mask exactly along pixel edges (pixel centres at integers)."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    pad = np.pad(m, 1)
    segs = []
    # horizontal edges between rows y-1 and y (in padded coords)
    dy = pad[1:, 1:-1] != pad[:-1, 1:-1]  # (h+1, w)
    for y, x in zip(*np.nonzero(dy)):
        segs.append([(x - 0.5, y - 0.5), (x + 0.5, y - 0.5)])
    dx = pad[1:-1, 1:] != pad[1:-1, :-1]  # (h, w+1)
    for y, x in zip(*np.nonzero(dx)):
        segs.append([(x - 0.5, y - 0.5), (x - 0.5, y + 0.5)])
    kw.setdefault("clip_on", False)
    lc = LineCollection(segs, colors=[color], linewidths=lw, **kw)
    ax.add_collection(lc)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    return lc


def note_figure(text: str, *, fig: Figure | SubFigure | None = None, title: str = "") -> Figure:
    """A small figure carrying only a message (used when there is nothing to plot)."""
    with style_context():
        fig = new_figure(fig, (5.0, 2.0))
        ax = fig.add_subplot()
        ax.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, color="0.3")
        if title:
            ax.set_title(title)
    return root_figure(fig)
