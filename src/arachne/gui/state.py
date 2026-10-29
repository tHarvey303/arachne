"""Viewer state and the pure functions behind every viewer panel (numpy only).

Nothing here imports bokeh: :class:`ViewerState` records what the user has chosen and the
functions turn ``(bundle, state)`` into plain arrays and dicts that the bokeh layer pushes into
``ColumnDataSource``s.  All bundle access goes through the public ``ResultsBundle`` accessors
(``band_image``, ``param_map``, ...) and :mod:`arachne.results.regions`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..results.display import percentile_limits, stretch
from ..results.regions import (
    Region,
    RegionSED,
    component_region_seds,
    region_param_summary,
    region_sed,
    total_sed,
)

__all__ = [
    "ViewerState",
    "SYMMETRIC_KINDS",
    "SYMMETRIC_PARAM_STATS",
    "SED_OPTIONS",
    "SELECTION_MODES",
    "COLORMAPS",
    "default_state",
    "band_order",
    "image_kinds",
    "model_stats",
    "param_choices",
    "param_stats",
    "segment_region",
    "image_for",
    "is_symmetric",
    "display_for",
    "selection_region",
    "region_from_indices",
    "sed_for",
    "total_for",
    "component_seds_for",
    "sed_columns",
    "residual_columns",
    "hist_for",
    "plain_label",
    "tex_label",
    "header_text",
]

SYMMETRIC_KINDS = ("residual", "chi", "truth_residual")
SYMMETRIC_PARAM_STATS = ("diff", "z")
SED_OPTIONS = ("intrinsic", "truth", "map", "log_y", "total", "components")
SELECTION_MODES = ("pixel", "circle", "box", "lasso", "polygon")
COLORMAPS = ("Viridis256", "Magma256", "Inferno256", "Cividis256", "Greys256", "Turbo256")
_ALL_KINDS = ("data", "model", "intrinsic", "residual", "chi", "truth_model", "truth_residual")
_COMP_PREFIX = "__comp"


@dataclass
class ViewerState:
    """Everything the user has chosen in the viewer.

    Attributes:
        band: Band index (into ``bundle.band_names``).
        kinds: Image kinds of panels 1 and 2 (see ``ResultsBundle.band_image``).
        model_stat: Model statistic for model-derived image kinds.
        stretch: Display stretch (``linear``, ``asinh``, ``log``, ``sqrt``).
        pct: Lower and upper display percentiles.
        cmap: Colormap name for non-diverging images (a bokeh palette name).
        model_floor: Include the model-error floor in the chi images.
        param: Parameter of panel 3: a map name, ``"__comp<k>"`` (component k fraction of
            the current band) or None.
        param_stat: Statistic of the parameter map.
        mask_outside_segment: Blank image pixels outside the segment region.
        component: Selected component index, or None for the total.
        mode: Selection mode, one of :data:`SELECTION_MODES`.
        radius: Circle radius in pixels (``mode == "circle"``).
        selection: Current selection, or None (then the default total region is used).
        overlays: Kept selections drawn as coloured overlays (up to 5).
        sed_options: Enabled SED options, a subset of :data:`SED_OPTIONS`.
        image_bundle: Index of the bundle shown in the image panels (compare mode).
    """

    band: int = 0
    kinds: list[str] = field(default_factory=lambda: ["data", "chi"])
    model_stat: str = "p50"
    stretch: str = "asinh"
    pct: tuple[float, float] = (0.5, 99.5)
    cmap: str = "Viridis256"
    model_floor: bool = True
    param: str | None = None
    param_stat: str = "p50"
    mask_outside_segment: bool = False
    component: int | None = None
    mode: str = "pixel"
    radius: float = 2.0
    selection: Region | None = None
    overlays: list[Region] = field(default_factory=list)
    sed_options: set[str] = field(default_factory=lambda: {"map", "total", "truth"})
    image_bundle: int = 0


# ---------------------------------------------------------------------------- menus


def band_order(bundle: Any) -> list[int]:
    """Band indices sorted by pivot wavelength (unknown pivots last)."""
    piv = np.asarray(bundle.pivot_um, dtype=float)
    key = np.where(np.isfinite(piv), piv, np.inf)
    return [int(i) for i in np.argsort(key, kind="stable")]


def image_kinds(bundle: Any) -> list[str]:
    """Image kinds this bundle can show in panels 1 and 2."""
    out = []
    truth = bundle.truth
    for kind in _ALL_KINDS:
        if kind == "intrinsic" and not bundle.has_intrinsic:
            continue
        if kind in ("truth_model", "truth_residual") and (truth is None or truth.convolved is None):
            continue
        out.append(kind)
    return out


def model_stats(bundle: Any) -> list[str]:
    """Model statistics offered for model-derived images (no silent fallbacks)."""
    if bundle.has_draws:
        return ["p50", "map", "mean", "p16", "p84", "std"]
    return ["map"]


def param_choices(bundle: Any) -> list[tuple[str, str]]:
    """Parameter-panel choices as ``(key, label)``: map names, then component fractions."""
    out = [(n, n) for n in bundle.map_names]
    comps = bundle.components
    if comps is not None and comps.has_frame("convolved"):
        for k in range(comps.n_components):
            out.append((f"{_COMP_PREFIX}{k}", f"comp {k} ({comps.profile_names[k]}) fraction"))
    return out


def _comp_index(key: str | None) -> int | None:
    if key is not None and key.startswith(_COMP_PREFIX):
        return int(key[len(_COMP_PREFIX) :])
    return None


def param_stats(bundle: Any, key: str | None) -> list[str]:
    """Statistics offered for a parameter-panel choice."""
    if key is None:
        return []
    if _comp_index(key) is not None:
        return model_stats(bundle)[:2] if bundle.has_draws else ["map"]
    return list(bundle.available_param_stats())


def default_state(bundle: Any) -> ViewerState:
    """A sensible initial state for ``bundle``."""
    choices = param_choices(bundle)
    param = choices[0][0] if choices else None
    pstats = param_stats(bundle, param)
    pstat = "p50" if "p50" in pstats else (pstats[0] if pstats else "map")
    opts = {"map", "total"}
    if bundle.truth is not None:
        opts.add("truth")
    if bundle.components is not None:
        opts.add("components")
    return ViewerState(
        band=band_order(bundle)[0] if bundle.n_bands else 0,
        model_stat=model_stats(bundle)[0],
        param=param,
        param_stat=pstat,
        sed_options=opts,
    )


# ---------------------------------------------------------------------------- images


def segment_region(bundle: Any) -> Region | None:
    """The region used by "mask outside segment": ``segment``, else ``galaxy``, else None."""
    for name in ("segment", "galaxy"):
        if name in bundle.regions:
            return bundle.regions[name]
    return None


def _image_stat(kind: str, stat: str) -> str:
    # residual / chi of the model *std* is meaningless: use the median model instead
    if kind in ("residual", "chi") and stat == "std":
        return "p50"
    return stat


def image_for(bundle: Any, state: ViewerState, panel: int = 0) -> np.ndarray:
    """Raw (unstretched) image of a panel.

    Panels 0 and 1 show ``state.kinds[panel]`` of ``state.band`` (the selected component's
    image for ``model``/``intrinsic`` when a component is selected); panel 2 shows the
    parameter map ``state.param`` with ``state.param_stat``.

    Args:
        bundle: A ``ResultsBundle``.
        state: Viewer state.
        panel: 0, 1 (band images) or 2 (parameter map).

    Returns:
        Float array (H, W); NaN outside the segment when ``state.mask_outside_segment``.

    Raises:
        KeyError: If the product is not in the bundle.
    """
    b = state.band
    if panel in (0, 1):
        kind = state.kinds[panel]
        stat = _image_stat(kind, state.model_stat)
        comps = bundle.components
        k = state.component
        if comps is not None and k is not None and kind in ("model", "intrinsic"):
            frame = "convolved" if kind == "model" else "intrinsic"
            img = np.asarray(comps.image(k, frame, stat)[b], dtype=float)
        else:
            img = np.asarray(bundle.band_image(b, kind, stat, model_floor=state.model_floor))
    elif panel == 2:
        if state.param is None:
            raise KeyError("No parameter selected")
        k = _comp_index(state.param)
        if k is not None:
            stat = state.param_stat if state.param_stat in ("map", "p50") else "map"
            part = np.asarray(bundle.components.image(k, "convolved", stat)[b], dtype=float)
            total = np.asarray(bundle.band_image(b, "model", stat), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                img = np.where(total != 0, part / total, np.nan)
        else:
            img = np.asarray(bundle.param_map(state.param, state.param_stat), dtype=float)
    else:
        raise ValueError(f"panel must be 0, 1 or 2, got {panel}")
    img = np.asarray(img, dtype=float)
    if state.mask_outside_segment:
        seg = segment_region(bundle)
        if seg is not None:
            img = np.where(seg.mask, img, np.nan)
    return img


def is_symmetric(state: ViewerState, panel: int) -> bool:
    """Whether a panel uses symmetric limits and a diverging colormap."""
    if panel in (0, 1):
        return state.kinds[panel] in SYMMETRIC_KINDS
    return state.param_stat in SYMMETRIC_PARAM_STATS


def _fmt(v: float) -> str:
    if v == 0 or not np.isfinite(v):
        return f"{v:g}"
    if abs(v) >= 1e4 or abs(v) < 1e-2:
        return f"{v:.2e}"
    return f"{v:.3g}"


def display_for(img: np.ndarray, state: ViewerState, symmetric: bool = False) -> dict[str, Any]:
    """Stretch an image for display.

    Symmetric images (residuals, chi, diff, z) are shown linearly between symmetric
    percentile limits; others use ``state.stretch`` between ``state.pct`` limits.

    Args:
        img: Raw image (H, W).
        state: Viewer state (``stretch``, ``pct``).
        symmetric: Use symmetric limits and a linear stretch.

    Returns:
        ``{"img": (H, W) in [0, 1] (NaN preserved), "lo", "hi", "ticks": positions in
        [0, 1], "tick_labels": raw values as strings}``.
    """
    lo, hi = percentile_limits(img, state.pct, symmetric=symmetric)
    kind = "linear" if symmetric else state.stretch
    out = stretch(img, kind, lo, hi)
    raw_ticks = np.linspace(lo, hi, 5)
    pos = stretch(raw_ticks, kind, lo, hi)
    return {
        "img": out,
        "lo": lo,
        "hi": hi,
        "ticks": [float(p) for p in pos],
        "tick_labels": [_fmt(float(v)) for v in raw_ticks],
    }


# ---------------------------------------------------------------------------- selections


def region_from_indices(
    shape: tuple[int, int], flat_indices: Sequence[int], name: str | None = None
) -> Region:
    """A mask region from flat pixel indices (``y * W + x``)."""
    mask = np.zeros(int(shape[0]) * int(shape[1]), dtype=bool)
    idx = np.asarray(list(flat_indices), dtype=int)
    if idx.size:
        mask[idx] = True
    mask = mask.reshape(shape)
    return Region.from_mask(mask, name=name or f"selection ({int(mask.sum())} px)")


def selection_region(bundle: Any, state: ViewerState) -> Region:
    """The current selection, else ``segment``, else ``all``, else every pixel."""
    if state.selection is not None:
        return state.selection
    for name in ("segment", "all"):
        if name in bundle.regions:
            return bundle.regions[name]
    return Region.from_mask(np.ones(bundle.image_shape, dtype=bool), name="all")


def sed_for(bundle: Any, state: ViewerState) -> list[RegionSED]:
    """SEDs of the current selection followed by every kept overlay.

    The first entry is the main SED (the selection, or the default total region when nothing
    is selected); it is never empty.
    """
    main = region_sed(bundle, selection_region(bundle, state))
    return [main] + [region_sed(bundle, reg) for reg in state.overlays]


def total_for(bundle: Any) -> RegionSED:
    """The reference total SED (``total_sed``)."""
    return total_sed(bundle)


def component_seds_for(bundle: Any, state: ViewerState) -> list[RegionSED]:
    """Per-component SEDs over the current selection (only component k when selected)."""
    if bundle.components is None:
        return []
    seds = component_region_seds(bundle, selection_region(bundle, state))
    if state.component is not None:
        return [seds[state.component]]
    return seds


def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def sed_columns(sed: RegionSED) -> dict[str, np.ndarray | list]:
    """Columns of a SED for plotting, sorted by pivot wavelength.

    Returns:
        ``x`` (micron), ``band``, ``obs``, ``obs_lo``, ``obs_hi``, ``p16``, ``p50``, ``p84``
        (convolved model; the MAP repeated without draws), ``map``, ``int_p16``,
        ``int_p50``, ``int_p84``, ``int_map``, ``truth_conv``, ``truth_int``.  Missing
        quantities are NaN.
    """
    n = len(sed.band_names)
    order = np.argsort(np.where(np.isfinite(sed.pivot_um), sed.pivot_um, np.inf), kind="stable")
    obs = _nan(n) if sed.obs_flux is None else np.asarray(sed.obs_flux, dtype=float)
    err = _nan(n) if sed.obs_err is None else np.asarray(sed.obs_err, dtype=float)
    try:
        p16, p50, p84 = sed.model_percentiles((16, 50, 84), "convolved")
    except KeyError:
        p16 = p50 = p84 = _nan(n)
    try:
        i16, i50, i84 = sed.model_percentiles((16, 50, 84), "intrinsic")
    except KeyError:
        i16 = i50 = i84 = _nan(n)

    def _opt(a):
        return _nan(n) if a is None else np.asarray(a, dtype=float)

    cols = {
        "x": np.asarray(sed.pivot_um, dtype=float),
        "band": np.asarray([str(b).rsplit(".", 1)[-1] for b in sed.band_names], dtype=object),
        "obs": obs,
        "obs_lo": obs - err,
        "obs_hi": obs + err,
        "p16": p16,
        "p50": p50,
        "p84": p84,
        "map": _opt(sed.model_map),
        "int_p16": i16,
        "int_p50": i50,
        "int_p84": i84,
        "int_map": _opt(sed.intrinsic_map),
        "truth_conv": _opt(sed.truth_convolved),
        "truth_int": _opt(sed.truth_intrinsic),
    }
    return {k: (list(v[order]) if k == "band" else np.asarray(v)[order]) for k, v in cols.items()}


def residual_columns(sed: RegionSED) -> dict[str, np.ndarray | list]:
    """Residual columns of a SED, sorted by pivot wavelength.

    Returns:
        ``x``, ``band``, ``chi`` (``RegionSED.chi()``), ``frac``
        (``RegionSED.frac_residual()``) and its 1-sigma range ``frac_lo``/``frac_hi``.
    """
    n = len(sed.band_names)
    order = np.argsort(np.where(np.isfinite(sed.pivot_um), sed.pivot_um, np.inf), kind="stable")
    if sed.obs_flux is None or sed.obs_err is None:
        chi = frac = ferr = _nan(n)
    else:
        chi = sed.chi()
        frac = sed.frac_residual()
        p50 = sed.model_percentiles((50,))[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            ferr = np.asarray(sed.obs_err, dtype=float) / np.abs(p50)
    cols = {
        "x": np.asarray(sed.pivot_um, dtype=float),
        "band": np.asarray([str(b).rsplit(".", 1)[-1] for b in sed.band_names], dtype=object),
        "chi": chi,
        "frac": frac,
        "frac_lo": frac - ferr,
        "frac_hi": frac + ferr,
    }
    return {k: (list(v[order]) if k == "band" else np.asarray(v)[order]) for k, v in cols.items()}


# ---------------------------------------------------------------------------- histograms


class _MapsProxy:
    """Duck-typed bundle exposing fixed maps to ``region_param_summary``."""

    has_param_draws = False

    def __init__(self, bundle: Any, maps: np.ndarray):
        self.map_names = list(bundle.map_names)
        self.mass_param = bundle.mass_param
        self.image_shape = tuple(bundle.image_shape)
        self._maps = np.asarray(maps, dtype=float)

    def param_map(self, name: str, stat: str = "map") -> np.ndarray:
        return self._maps[self.map_names.index(name)]


def _hist_entry(
    label: str,
    values: np.ndarray,
    bins: int | None,
    truth: float | None = None,
    best: float | None = None,
    unit: str = "",
) -> dict[str, Any]:
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    entry: dict[str, Any] = {
        "label": label,
        "unit": unit,
        "values": vals,
        "truth": truth,
        "map": best,
        "left": np.zeros(0),
        "right": np.zeros(0),
        "top": np.zeros(0),
        "summary": "",
    }
    if vals.size > 1:
        if bins is None:
            bins = int(np.clip(round(1.5 * np.sqrt(vals.size)), 5, 40))
        lo, hi = float(vals.min()), float(vals.max())
        if not hi > lo:
            pad = max(abs(lo) * 1e-3, 1e-6)
            lo, hi = lo - pad, hi + pad
        counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
        entry["left"], entry["right"], entry["top"] = edges[:-1], edges[1:], counts.astype(float)
        p16, p50, p84 = np.percentile(vals, [16, 50, 84])
        entry["summary"] = f"{p50:.3g} +{p84 - p50:.2g} / -{p50 - p16:.2g}"
    elif vals.size == 1:
        entry["map"] = float(vals[0]) if best is None else best
        entry["summary"] = f"MAP {vals[0]:.3g}"
    return entry


def hist_for(bundle: Any, state: ViewerState, bins: int | None = None) -> dict[str, dict[str, Any]]:
    """Posterior histograms for the current selection.

    One entry per parameter map (region aggregates from ``region_param_summary``: the mass
    parameter is ``log10 sum 10**m``, the others mass-weighted means), one per scalar
    (global), and for additive bundles with a selected component one per finite, varying
    component parameter.  Degraded bundles return ``{}`` (no per-pixel posteriors).

    Returns:
        ``{key: {"label", "unit", "values", "left", "right", "top", "truth", "map",
        "summary"}}``; ``truth``/``map`` are floats or None (``map`` is set when there are no
        draws, so the panel shows a MAP span instead of a histogram).
    """
    if bundle.degraded:
        return {}
    out: dict[str, dict[str, Any]] = {}
    reg = selection_region(bundle, state)
    units = getattr(bundle, "units", {}) or {}
    names = list(bundle.map_names)
    if names and reg.n_pix > 0:
        summ = region_param_summary(bundle, reg)
        truth_s = None
        truth = bundle.truth
        if truth is not None and truth.param_maps is not None:
            truth_s = region_param_summary(_MapsProxy(bundle, truth.param_maps), reg)
        for n in names:
            t = None if truth_s is None else float(np.asarray(truth_s[n]).ravel()[0])
            out[n] = _hist_entry(n, summ[n], bins, truth=t, unit=units.get(n, ""))
    sdraws = bundle.scalar_draws()
    tsc = None if bundle.truth is None else bundle.truth.scalars
    for i, n in enumerate(bundle.scalar_names):
        t = None
        if tsc is not None and i < len(np.atleast_1d(tsc)):
            t = float(np.atleast_1d(tsc)[i])
        out[f"scalar:{n}"] = _hist_entry(
            f"{n} (global)", sdraws[n], bins, truth=t, unit=units.get(n, "")
        )
    comps = bundle.components
    k = state.component
    if comps is not None and k is not None:
        params = comps.params()  # (S, K, Pc)
        for j, pname in enumerate(comps.param_names):
            col = np.asarray(params[:, k, j], dtype=float)
            if not np.isfinite(col).any():
                continue
            if col.size > 1 and np.nanstd(col) == 0:
                continue
            out[f"comp{k}:{pname}"] = _hist_entry(f"comp {k} {pname}", col, bins)
    return out


# ---------------------------------------------------------------------------- labels


def plain_label(name: str, units: dict[str, str] | None = None) -> str:
    """``name [unit]`` without TeX (for widgets)."""
    u = (units or {}).get(name, "")
    return f"{name} [{u}]" if u else name


def tex_label(label: str) -> str:
    """Convert a matplotlib-style ``$...$`` label to bokeh's ``$$...$$`` math delimiters."""
    s = str(label)
    if s.startswith("$$") or "$" not in s:
        return s
    if s.startswith("$") and s.endswith("$") and s.count("$") == 2:
        return f"$${s[1:-1]}$$"
    return s.replace("$", "")


def header_text(bundle: Any) -> str:
    """One-line summary: title, posterior source, S, chi^2_red and path."""
    chi2 = (bundle.metrics or {}).get("chi2_red")
    parts = [
        f"<b>{bundle.title or 'arachne results'}</b>",
        f"posterior: {bundle.posterior_source}",
        f"S = {bundle.n_draws}",
    ]
    if isinstance(chi2, (int, float)):
        parts.append(f"&chi;&sup2;<sub>red</sub> = {chi2:.3g}")
    parts.append(f"{bundle.spatial_model}" if bundle.spatial_model else "")
    parts.append(f"<code>{bundle.path}</code>" if bundle.path else "<i>in memory</i>")
    return " &nbsp;|&nbsp; ".join(p for p in parts if p)
