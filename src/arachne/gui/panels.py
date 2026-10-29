"""Bokeh builders for the viewer: image, SED, residual and histogram panels and the controls.

Every builder returns plain bokeh models plus the ``ColumnDataSource``s they draw from; the
viewer (:mod:`arachne.gui.app`) owns the callbacks and fills the sources from
:mod:`arachne.gui.state`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from bokeh.layouts import column
from bokeh.models import (
    BoxSelectTool,
    Button,
    CheckboxGroup,
    ColorBar,
    ColumnDataSource,
    CustomJSTickFormatter,
    Div,
    FixedTicker,
    HoverTool,
    LassoSelectTool,
    LinearColorMapper,
    PolySelectTool,
    RadioButtonGroup,
    Range1d,
    RangeSlider,
    Select,
    Slider,
    Span,
    Toggle,
    Whisker,
)
from bokeh.palettes import all_palettes
from bokeh.plotting import figure

from .state import COLORMAPS, SED_OPTIONS, SELECTION_MODES

__all__ = [
    "OVERLAY_COLORS",
    "COMPONENT_COLORS",
    "palette",
    "ImagePanel",
    "image_panel",
    "SEDPanel",
    "sed_sources",
    "sed_panel",
    "residual_panel",
    "HistPanel",
    "hist_panel",
    "controls",
    "section",
]

OVERLAY_COLORS = ("#e6550d", "#31a354", "#756bb1", "#636363", "#d6616b")
COMPONENT_COLORS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02")
MODEL_COLOR = "#1f77b4"
TRUTH_COLOR = "#d62728"
COMPARE_COLOR = "#9467bd"
TOTAL_COLOR = "#999999"
INTRINSIC_COLOR = "#ff7f0e"
SED_OPTION_LABELS = {
    "intrinsic": "intrinsic",
    "truth": "truth",
    "map": "MAP markers",
    "log_y": "log y",
    "total": "total (grey)",
    "components": "components",
}

MODE_LABELS = {"polygon": "poly"}

_TICK_JS = """
let best = -1, bestd = 1e-6;
for (let i = 0; i < pos.length; i++) {
    const d = Math.abs(pos[i] - tick);
    if (d < bestd) { best = i; bestd = d; }
}
return best >= 0 ? labels[best] : "";
"""


def palette(name: str) -> tuple[str, ...]:
    """A 256-colour bokeh palette by name (``"Viridis256"``) or the diverging ``"RdBu"``."""
    if name == "RdBu":
        return tuple(all_palettes["RdBu"][11])  # low = blue, high = red
    base = name[:-3] if name.endswith("256") else name
    pals = all_palettes.get(base)
    if pals is None:
        raise KeyError(f"Unknown palette {name!r}")
    return pals[max(pals)]


def _blank(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, np.nan)


# ---------------------------------------------------------------------------- images


@dataclass
class ImagePanel:
    """One image figure and its models."""

    fig: Any
    source: ColumnDataSource
    mapper: LinearColorMapper
    colorbar: ColorBar
    formatter: CustomJSTickFormatter
    ticker: FixedTicker
    tools: dict[str, Any] = field(default_factory=dict)


def image_panel(
    shape: tuple[int, int],
    x_range: Range1d,
    y_range: Range1d,
    pixel_source: ColumnDataSource,
    selection_source: ColumnDataSource,
    overlay_sources: list[ColumnDataSource],
    *,
    title: str = "",
    width: int = 330,
    height: int = 300,
) -> ImagePanel:
    """An image figure with a colour bar, hover readout and pixel selection tools.

    The image is drawn with pixel ``[y, x]`` centred on ``(x, y)``.  ``pixel_source`` holds
    one invisible point per pixel centre (shared between panels so selections are linked);
    the box, lasso and polygon tools act on it.  ``selection_source`` and
    ``overlay_sources`` hold pixel centres drawn as translucent squares.

    Args:
        shape: Image shape (H, W).
        x_range: Shared x range.
        y_range: Shared y range.
        pixel_source: ``x``, ``y`` of every pixel centre (row-major, ``y * W + x``).
        selection_source: ``x``, ``y`` of the selected pixels.
        overlay_sources: One source per kept overlay.
        title: Figure title.
        width: Plot width in px.
        height: Plot height in px.

    Returns:
        An :class:`ImagePanel`.
    """
    h, w = shape
    fig = figure(
        title=title,
        x_range=x_range,
        y_range=y_range,
        width=width,
        height=height,
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="above",
        output_backend="webgl",
    )
    fig.grid.visible = False
    fig.xaxis.axis_label = "x [px]"
    fig.yaxis.axis_label = "y [px]"
    source = ColumnDataSource({"img": [_blank(shape)], "raw": [_blank(shape)]})
    mapper = LinearColorMapper(
        palette=palette("Viridis256"), low=0.0, high=1.0, nan_color="rgba(0,0,0,0)"
    )
    img = fig.image(
        image="img", x=-0.5, y=-0.5, dw=w, dh=h, color_mapper=mapper, source=source, level="image"
    )
    ticker = FixedTicker(ticks=[0.0, 1.0])
    formatter = CustomJSTickFormatter(args={"pos": [0.0, 1.0], "labels": ["", ""]}, code=_TICK_JS)
    colorbar = ColorBar(
        color_mapper=mapper, ticker=ticker, formatter=formatter, width=10, label_standoff=6
    )
    fig.add_layout(colorbar, "right")
    fig.add_tools(
        HoverTool(
            renderers=[img],
            tooltips=[("x", "$x{0}"), ("y", "$y{0}"), ("value", "@raw{0.000g}")],
        )
    )
    pts = fig.scatter(
        "x",
        "y",
        source=pixel_source,
        size=4,
        alpha=0.0,
        selection_alpha=0.0,
        nonselection_alpha=0.0,
    )
    box = BoxSelectTool(renderers=[pts])
    lasso = LassoSelectTool(renderers=[pts])
    poly = PolySelectTool(renderers=[pts])
    fig.add_tools(box, lasso, poly)
    fig.rect(
        "x",
        "y",
        1,
        1,
        source=selection_source,
        fill_color="white",
        fill_alpha=0.15,
        line_color="white",
        line_alpha=0.6,
        line_width=0.5,
    )
    for src, col in zip(overlay_sources, OVERLAY_COLORS):
        fig.rect(
            "x",
            "y",
            1,
            1,
            source=src,
            fill_color=col,
            fill_alpha=0.2,
            line_color=col,
            line_alpha=0.6,
            line_width=0.5,
        )
    pan = next(t for t in fig.tools if type(t).__name__ == "PanTool")
    return ImagePanel(
        fig,
        source,
        mapper,
        colorbar,
        formatter,
        ticker,
        {"box": box, "lasso": lasso, "polygon": poly, "pan": pan},
    )


# ---------------------------------------------------------------------------- SED


@dataclass
class SEDPanel:
    """An SED figure and its renderers grouped by the option that toggles them."""

    fig: Any
    groups: dict[str, list[Any]]


def _empty_sed_cols() -> dict[str, list]:
    keys = (
        "x",
        "band",
        "obs",
        "obs_lo",
        "obs_hi",
        "p16",
        "p50",
        "p84",
        "map",
        "int_p16",
        "int_p50",
        "int_p84",
        "int_map",
        "truth_conv",
        "truth_int",
    )
    return {k: [] for k in keys}


def sed_sources(
    n_components: int, n_overlays: int
) -> dict[str, ColumnDataSource | list[ColumnDataSource]]:
    """Empty sources for the SED panels: main, total, compare, components, overlays."""
    return {
        "main": ColumnDataSource(_empty_sed_cols()),
        "total": ColumnDataSource(_empty_sed_cols()),
        "compare": ColumnDataSource(_empty_sed_cols()),
        "components": [ColumnDataSource(_empty_sed_cols()) for _ in range(n_components)],
        "overlays": [ColumnDataSource(_empty_sed_cols()) for _ in range(n_overlays)],
    }


def sed_panel(
    sources: dict[str, Any],
    x_range: Range1d | None = None,
    *,
    y_axis_type: str = "linear",
    component_labels: list[str] | None = None,
    include: set[str] | None = None,
    width: int = 600,
    height: int = 340,
) -> SEDPanel:
    """Observed-versus-model SED figure (log-x micron).

    Observed points with ``Whisker`` error bars; model p50 line with a p16-p84 band; MAP open
    markers; optional intrinsic model, truth, total (grey), per-component and compare-bundle
    curves, and overlay SEDs.

    Args:
        sources: From :func:`sed_sources`.
        x_range: Optional shared x range.
        y_axis_type: ``"linear"`` or ``"log"``.
        component_labels: Legend labels of the component curves.
        include: Optional groups to draw (``band``, ``intrinsic``, ``truth``,
            ``components``, ``compare``); default all.  Groups left out get no renderers,
            so they never appear in the legend.
        width: Plot width in px.
        height: Plot height in px.

    Returns:
        An :class:`SEDPanel`; ``groups`` keys: ``obs``, ``band``, ``model``, ``map``,
        ``intrinsic``, ``intrinsic_band``, ``truth``, ``total``, ``components``,
        ``compare``, ``overlays``.
    """
    optional = {"band", "intrinsic", "truth", "components", "compare"}
    inc = optional if include is None else optional & set(include)
    kw = {"x_range": x_range} if x_range is not None else {}
    fig = figure(
        x_axis_type="log",
        y_axis_type=y_axis_type,
        width=width,
        height=height,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above",
        title="SED",
        **kw,
    )
    fig.xaxis.axis_label = "wavelength [µm]"
    fig.yaxis.axis_label = "flux [nJy]"
    main, total, cmp_src = sources["main"], sources["total"], sources["compare"]
    groups = ("obs", "band", "model", "map", "intrinsic", "intrinsic_band", "truth", "total")
    g: dict[str, list[Any]] = {k: [] for k in (*groups, "components", "compare", "overlays")}

    g["total"] += [
        fig.line(
            "x",
            "p50",
            source=total,
            color=TOTAL_COLOR,
            line_width=2,
            alpha=0.7,
            legend_label="total model",
        ),
        fig.scatter(
            "x", "obs", source=total, color=TOTAL_COLOR, size=6, alpha=0.7, legend_label="total obs"
        ),
    ]
    if "band" in inc:
        g["band"].append(
            fig.varea(
                "x",
                "p16",
                "p84",
                source=main,
                fill_color=MODEL_COLOR,
                fill_alpha=0.25,
                legend_label="model 16-84",
            )
        )
    g["model"].append(
        fig.line("x", "p50", source=main, color=MODEL_COLOR, line_width=2, legend_label="model")
    )
    g["map"].append(
        fig.scatter(
            "x",
            "map",
            source=main,
            marker="circle",
            size=9,
            fill_color=None,
            line_color=MODEL_COLOR,
            line_width=1.5,
            legend_label="MAP",
        )
    )
    if "intrinsic" in inc:
        if "band" in inc:
            g["intrinsic_band"].append(
                fig.varea(
                    "x",
                    "int_p16",
                    "int_p84",
                    source=main,
                    fill_color=INTRINSIC_COLOR,
                    fill_alpha=0.15,
                )
            )
        g["intrinsic"].append(
            fig.line(
                "x",
                "int_p50",
                source=main,
                color=INTRINSIC_COLOR,
                line_dash="dashed",
                line_width=2,
                legend_label="intrinsic",
            )
        )
    if "truth" in inc:
        g["truth"] += [
            fig.scatter(
                "x",
                "truth_conv",
                source=main,
                marker="x",
                size=10,
                color=TRUTH_COLOR,
                line_width=2,
                legend_label="truth",
            ),
            fig.line(
                "x",
                "truth_int",
                source=main,
                color=TRUTH_COLOR,
                line_dash="dotted",
                legend_label="truth intrinsic",
            ),
        ]
    if "components" in inc:
        labels = component_labels or [f"comp {k}" for k in range(len(sources["components"]))]
        for k, src in enumerate(sources["components"]):
            col = COMPONENT_COLORS[k % len(COMPONENT_COLORS)]
            g["components"].append(
                fig.line(
                    "x",
                    "p50",
                    source=src,
                    color=col,
                    line_width=1.5,
                    line_dash="dashdot",
                    legend_label=labels[k],
                )
            )
    if "compare" in inc:
        g["compare"] += [
            fig.line(
                "x",
                "p50",
                source=cmp_src,
                color=COMPARE_COLOR,
                line_width=2,
                legend_label="bundle B",
            ),
            fig.varea("x", "p16", "p84", source=cmp_src, fill_color=COMPARE_COLOR, fill_alpha=0.15),
        ]
    for src, col in zip(sources["overlays"], OVERLAY_COLORS):
        g["overlays"].append(fig.line("x", "p50", source=src, color=col, line_width=1.5))
        g["overlays"].append(fig.scatter("x", "obs", source=src, color=col, size=6))
    whisker = Whisker(
        base="x",
        upper="obs_hi",
        lower="obs_lo",
        source=main,
        line_color="black",
        upper_head=None,
        lower_head=None,
    )
    fig.add_layout(whisker)
    obs = fig.scatter("x", "obs", source=main, color="black", size=7, legend_label="observed")
    g["obs"] += [whisker, obs]
    fig.add_tools(
        HoverTool(
            renderers=[obs],
            tooltips=[("band", "@band"), ("obs", "@obs{0.000g}"), ("model", "@p50{0.000g}")],
        )
    )
    legend = fig.legend[0]
    legend.click_policy = "hide"
    legend.label_text_font_size = "8pt"
    legend.spacing = 0
    legend.padding = 4
    legend.glyph_width = 16
    fig.add_layout(legend, "right")
    return SEDPanel(fig, g)


def residual_panel(
    source: ColumnDataSource,
    x_range: Range1d | None = None,
    *,
    width: int = 420,
    height: int = 170,
) -> tuple[Any, dict[str, Any]]:
    """Residual figures, stacked: normalised (σ, with ±1/±3 spans) and fractional.

    Args:
        source: Columns from :func:`arachne.gui.state.residual_columns`.
        x_range: Shared x range (the SED figure's).
        width: Plot width in px.
        height: Height of each residual figure in px.

    Returns:
        ``(column, {"chi": fig, "frac": fig})``.
    """
    kw = {"x_range": x_range} if x_range is not None else {}
    figs = {}
    for key, ylab in (("chi", "(obs − model) / σ"), ("frac", "(obs − model) / model")):
        extra = {"y_range": Range1d(-3.5, 3.5)} if key == "chi" else {}
        fig = figure(
            x_axis_type="log",
            width=width,
            height=height,
            tools="pan,wheel_zoom,reset",
            toolbar_location=None,
            **kw,
            **extra,
        )
        fig.xaxis.axis_label = "wavelength [µm]"
        fig.yaxis.axis_label = ylab
        fig.add_layout(Span(location=0, dimension="width", line_color="black", line_width=1))
        if key == "chi":
            for v, dash in ((1, "dashed"), (-1, "dashed"), (3, "dotted"), (-3, "dotted")):
                fig.add_layout(
                    Span(location=v, dimension="width", line_color="grey", line_dash=dash)
                )
            r = fig.scatter("x", "chi", source=source, size=7, color="black")
        else:
            fig.add_layout(
                Whisker(
                    base="x",
                    upper="frac_hi",
                    lower="frac_lo",
                    source=source,
                    upper_head=None,
                    lower_head=None,
                )
            )
            r = fig.scatter("x", "frac", source=source, size=7, color="black")
        fig.add_tools(HoverTool(renderers=[r], tooltips=[("band", "@band"), (key, f"@{key}")]))
        figs[key] = fig
    figs["chi"].title.text = "residuals (σ)"
    figs["frac"].title.text = "fractional residuals"
    return column(figs["chi"], figs["frac"]), figs


# ---------------------------------------------------------------------------- histograms


@dataclass
class HistPanel:
    """A histogram figure with truth and MAP spans."""

    fig: Any
    source: ColumnDataSource
    truth: Span
    best: Span


def hist_panel(width: int = 230, height: int = 190) -> HistPanel:
    """An empty posterior-histogram figure (``quad`` + truth and MAP ``Span``s)."""
    fig = figure(width=width, height=height, tools="pan,wheel_zoom,reset", toolbar_location=None)
    fig.yaxis.visible = False
    fig.title.text_font_size = "9pt"
    fig.xaxis.ticker.desired_num_ticks = 4
    source = ColumnDataSource({"left": [], "right": [], "top": []})
    fig.quad(
        left="left",
        right="right",
        top="top",
        bottom=0,
        source=source,
        fill_color=MODEL_COLOR,
        fill_alpha=0.5,
        line_color=MODEL_COLOR,
    )
    truth = Span(location=0, dimension="height", line_color=TRUTH_COLOR, line_width=2)
    best = Span(location=0, dimension="height", line_color="black", line_dash="dashed")
    truth.visible = best.visible = False
    fig.add_layout(truth)
    fig.add_layout(best)
    return HistPanel(fig, source, truth, best)


# ---------------------------------------------------------------------------- controls


def controls(
    *,
    band_options: list[tuple[str, str]],
    band_value: str,
    kinds: list[str],
    kind_active: tuple[int, int],
    model_stats: list[str],
    model_stat: str,
    stretch: str,
    pct: tuple[float, float],
    cmap: str,
    model_floor: bool,
    param_options: list[tuple[str, str]],
    param_value: str | None,
    param_stats: list[str],
    param_stat: str,
    has_segment: bool,
    component_options: list[tuple[str, str]],
    region_names: list[str],
    radius: float,
    max_radius: float,
    sed_active: list[int],
    compare: bool = False,
) -> dict[str, Any]:
    """Build every control widget (no callbacks attached).

    Returns:
        Widgets keyed ``band, kind0, kind1, model_stat, stretch, pct, cmap, floor, param,
        param_stat, mask_segment, component, mode, radius, region, clear, keep, export,
        sed_options, image_bundle``.
    """
    from ..results.display import STRETCHES

    w: dict[str, Any] = {}
    w["band"] = Select(title="Band", options=band_options, value=band_value, width=250)
    w["kind0"] = Select(title="Panel 1", options=kinds, value=kinds[kind_active[0]], width=250)
    w["kind1"] = Select(title="Panel 2", options=kinds, value=kinds[kind_active[1]], width=250)
    w["model_stat"] = Select(title="Model stat", options=model_stats, value=model_stat, width=250)
    w["stretch"] = Select(title="Stretch", options=list(STRETCHES), value=stretch, width=120)
    w["cmap"] = Select(title="Colormap", options=list(COLORMAPS), value=cmap, width=125)
    w["pct"] = RangeSlider(
        title="Percentile range", start=0, end=100, step=0.5, value=tuple(pct), width=250
    )
    w["floor"] = Toggle(label="χ includes model floor", active=model_floor, width=250)
    w["param"] = Select(
        title="Parameter map",
        options=param_options or [("", "(none)")],
        value=param_value or "",
        width=250,
        disabled=not param_options,
    )
    w["param_stat"] = Select(
        title="Parameter stat",
        options=param_stats or [""],
        value=param_stat if param_stats else "",
        width=250,
        disabled=not param_stats,
    )
    w["mask_segment"] = Toggle(
        label="Mask outside segment", active=False, width=250, disabled=not has_segment
    )
    w["component"] = Select(
        title="Component",
        options=component_options or [("total", "total")],
        value="total",
        width=250,
        visible=bool(component_options),
    )
    w["mode"] = RadioButtonGroup(
        labels=[MODE_LABELS.get(m, m) for m in SELECTION_MODES], active=0, width=250
    )
    w["radius"] = Slider(
        title="Circle radius [px]", start=0.5, end=max_radius, step=0.5, value=radius, width=250
    )
    w["region"] = Select(
        title="Named region", options=["(none)", *region_names], value="(none)", width=250
    )
    w["clear"] = Button(label="Clear selection", width=120)
    w["keep"] = Button(label="Keep as overlay", width=125)
    w["export"] = Button(label="Export regions (JSON)", width=250)
    w["sed_options"] = CheckboxGroup(
        labels=[SED_OPTION_LABELS[o] for o in SED_OPTIONS], active=sed_active, width=250
    )
    w["image_bundle"] = Select(
        title="Images from",
        options=["bundle A", "bundle B"],
        value="bundle A",
        width=250,
        visible=compare,
    )
    return w


def section(text: str) -> Div:
    """A small bold section heading for the control column."""
    return Div(text=f"<b>{text}</b>", width=250, styles={"margin-top": "6px"})
