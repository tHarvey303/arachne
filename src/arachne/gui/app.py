"""The interactive results viewer: :class:`ArachneViewer` and :func:`view`.

A plain-bokeh layout with Python callbacks.  Every widget callback calls one of the public
controller methods (``set_band``, ``select_pixel``, ...), which update :attr:`ArachneViewer.state`
and push arrays computed by :mod:`arachne.gui.state` into the ``ColumnDataSource``s.  Tests drive
the same methods without a server.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from bokeh import events
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Range1d

from ..results.reader import ResultsBundle, load_bundle
from ..results.regions import Region, RegionSED, region_sed
from . import panels as P
from . import state as S

__all__ = ["ArachneViewer", "view", "MAX_OVERLAYS"]

MAX_OVERLAYS = 5
_SED_OPTIONS = list(S.SED_OPTIONS)


def _as_bundle(obj: ResultsBundle | str | Path) -> ResultsBundle:
    if isinstance(obj, ResultsBundle):
        return obj
    return load_bundle(obj)


def _pixel_cols(mask: np.ndarray) -> dict[str, np.ndarray]:
    yy, xx = np.nonzero(mask)
    return {"x": xx.astype(float), "y": yy.astype(float)}


class ArachneViewer:
    """Interactive viewer of an ``arachne.results`` bundle.

    Args:
        bundle: A :class:`~arachne.results.ResultsBundle` or a path to one.
        compare: Optional second bundle (same image shape) whose model SED is overlaid for
            the same region; the image panels can be switched to it.

    Attributes:
        bundle: The main bundle.
        compare: The comparison bundle or None.
        state: The :class:`~arachne.gui.state.ViewerState`.
        layout: The bokeh layout (built in ``__init__``).
        widgets: Control widgets by name.
        images: The three :class:`~arachne.gui.panels.ImagePanel`s.
        sources: Data sources by name (``sed``, ``total``, ``residual``, ...).
    """

    def __init__(
        self,
        bundle: ResultsBundle | str | Path,
        compare: ResultsBundle | str | Path | None = None,
    ):
        """Load the bundle(s) and build the layout."""
        self.bundle = _as_bundle(bundle)
        self.compare = None if compare is None else _as_bundle(compare)
        if self.compare is not None and tuple(self.compare.image_shape) != tuple(
            self.bundle.image_shape
        ):
            raise ValueError(
                f"compare bundle shape {self.compare.image_shape} differs from "
                f"{self.bundle.image_shape}"
            )
        self.state = S.default_state(self.bundle)
        self._busy = False
        self._suppress_selection = False
        self._seds: list[RegionSED] = []
        self._component_seds: list[RegionSED] = []
        self._compare_sed: RegionSED | None = None
        self._hists: dict[str, dict] = {}
        self._notes: list[str] = []
        self._total = S.total_for(self.bundle)
        self.last_update_ms = 0.0
        self._build()
        self._refresh_images()
        self._refresh_selection()

    # ------------------------------------------------------------------ properties

    @property
    def bundles(self) -> list[ResultsBundle]:
        """The main bundle, then the comparison bundle if any."""
        return [self.bundle] + ([self.compare] if self.compare is not None else [])

    @property
    def image_bundle(self) -> ResultsBundle:
        """The bundle shown in the image panels."""
        return self.bundles[min(self.state.image_bundle, len(self.bundles) - 1)]

    @property
    def show_uncertainty(self) -> bool:
        """Whether the bundle has draws (uncertainty bands and histograms)."""
        return bool(self.bundle.has_draws) and not self.bundle.degraded

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        b = self.bundle
        st = self.state
        h, w = b.image_shape
        self.x_range = Range1d(-0.5, w - 0.5)
        self.y_range = Range1d(-0.5, h - 0.5)
        yy, xx = np.mgrid[0:h, 0:w]
        self.sources: dict[str, Any] = {
            "pixels": ColumnDataSource(
                {"x": xx.ravel().astype(float), "y": yy.ravel().astype(float)}
            ),
            "selection": ColumnDataSource({"x": [], "y": []}),
            "overlay_pixels": [ColumnDataSource({"x": [], "y": []}) for _ in range(MAX_OVERLAYS)],
            "residual": ColumnDataSource(
                {k: [] for k in ("x", "band", "chi", "frac", "frac_lo", "frac_hi")}
            ),
        }
        n_comp = b.components.n_components if b.components is not None else 0
        sed_src = P.sed_sources(n_comp, MAX_OVERLAYS)
        self.sources.update(
            sed=sed_src["main"],
            total=sed_src["total"],
            compare=sed_src["compare"],
            components=sed_src["components"],
            overlays=sed_src["overlays"],
        )
        self.images = [
            P.image_panel(
                b.image_shape,
                self.x_range,
                self.y_range,
                self.sources["pixels"],
                self.sources["selection"],
                self.sources["overlay_pixels"],
            )
            for _ in range(3)
        ]
        for panel in self.images:
            panel.fig.on_event(events.Tap, self._on_tap)
        self.sources["pixels"].selected.on_change("indices", self._on_indices)

        comp_labels = []
        if b.components is not None:
            comp_labels = [f"comp {k} ({b.components.profile_names[k]})" for k in range(n_comp)]
        include = {"components"} if b.components is not None else set()
        if self.show_uncertainty:
            include.add("band")
        if b.has_intrinsic:
            include.add("intrinsic")
        if b.truth is not None:
            include.add("truth")
        if self.compare is not None:
            include.add("compare")
        self.sed_lin = P.sed_panel(
            sed_src, y_axis_type="linear", component_labels=comp_labels, include=include
        )
        self.sed_log = P.sed_panel(
            sed_src,
            x_range=self.sed_lin.fig.x_range,
            y_axis_type="log",
            component_labels=comp_labels,
            include=include,
        )
        self.residual_panel, self.residual_figs = P.residual_panel(
            self.sources["residual"], x_range=self.sed_lin.fig.x_range
        )

        n_hist = len(b.map_names) + len(b.scalar_names)
        if b.components is not None:
            n_hist += len(b.components.param_names)
        self.hist_panels = [P.hist_panel() for _ in range(n_hist)]

        kinds = S.image_kinds(b)
        self._kinds = kinds
        k0 = kinds.index("data")
        k1 = kinds.index("chi") if "chi" in kinds else min(1, len(kinds) - 1)
        st.kinds = [kinds[k0], kinds[k1]]
        order = S.band_order(b)
        comp_opts = []
        if b.components is not None:
            comp_opts = [("total", "total")] + [(str(k), lab) for k, lab in enumerate(comp_labels)]
        self._param_options = S.param_choices(b)
        self.widgets = P.controls(
            band_options=[(str(i), b.short_names[i]) for i in order],
            band_value=str(st.band),
            kinds=kinds,
            kind_active=(k0, k1),
            model_stats=S.model_stats(b),
            model_stat=st.model_stat,
            stretch=st.stretch,
            pct=st.pct,
            cmap=st.cmap,
            model_floor=st.model_floor,
            param_options=self._param_options,
            param_value=st.param,
            param_stats=S.param_stats(b, st.param),
            param_stat=st.param_stat,
            has_segment=S.segment_region(b) is not None,
            component_options=comp_opts,
            region_names=list(b.regions),
            radius=st.radius,
            max_radius=max(2.0, max(h, w) / 2.0),
            sed_active=[i for i, o in enumerate(_SED_OPTIONS) if o in st.sed_options],
            compare=self.compare is not None,
        )
        self.widgets["export"].visible = b.path is not None
        self._wire()

        self.header = Div(text=S.header_text(b), sizing_mode="stretch_width")
        self.banner = Div(text=self._banner_text(), visible=bool(self._banner_text()))
        self.status = Div(text="", width=250, styles={"font-size": "85%", "color": "#555"})
        self.hist_note = Div(text="", visible=False)
        W = self.widgets
        ctrl = column(
            P.section("Image"),
            W["band"],
            W["kind0"],
            W["kind1"],
            W["model_stat"],
            row(W["stretch"], W["cmap"]),
            W["pct"],
            W["floor"],
            W["mask_segment"],
            W["image_bundle"],
            P.section("Parameters"),
            W["param"],
            W["param_stat"],
            W["component"],
            P.section("Selection"),
            W["mode"],
            W["radius"],
            W["region"],
            row(W["clear"], W["keep"]),
            W["export"],
            P.section("SED"),
            W["sed_options"],
            self.status,
            width=265,
        )
        self.images[2].fig.visible = st.param is not None
        images = row(*[p.fig for p in self.images])
        seds = row(column(self.sed_lin.fig, self.sed_log.fig), self.residual_panel)
        self.hist_row = row(*[hp.fig for hp in self.hist_panels])
        centre = column(images, seds, self.hist_note, self.hist_row)
        self.layout = column(self.header, self.banner, row(ctrl, centre))
        self._apply_mode()

    def _banner_text(self) -> str:
        b = self.bundle
        style = "background:#fff3cd;border:1px solid #e0b000;padding:4px 8px;border-radius:4px;"
        if b.degraded:
            return (
                f"<div style='{style}'><b>Degraded bundle</b> (posterior source "
                f"'{b.posterior_source}'): SEDs and residuals use the MAP model only; no model "
                "uncertainty bands and no per-pixel posteriors. Parameter maps show the stored "
                "summaries.</div>"
            )
        if not b.has_draws:
            return (
                f"<div style='{style}'><b>No posterior draws</b>: MAP-only SEDs; histograms "
                "show the MAP value.</div>"
            )
        return ""

    def _wire(self) -> None:
        W = self.widgets
        W["band"].on_change("value", lambda a, o, n: self.set_band(int(n)))
        W["kind0"].on_change("value", lambda a, o, n: self.set_image_kind(0, n))
        W["kind1"].on_change("value", lambda a, o, n: self.set_image_kind(1, n))
        W["model_stat"].on_change("value", lambda a, o, n: self.set_model_stat(n))
        W["stretch"].on_change("value", lambda a, o, n: self.set_display(stretch=n))
        W["cmap"].on_change("value", lambda a, o, n: self.set_display(cmap=n))
        W["pct"].on_change("value_throttled", lambda a, o, n: self.set_display(pct=tuple(n)))
        W["floor"].on_change("active", lambda a, o, n: self.set_model_floor(bool(n)))
        W["mask_segment"].on_change(
            "active", lambda a, o, n: self.set_mask_outside_segment(bool(n))
        )
        W["param"].on_change("value", lambda a, o, n: self.set_param(n) if n else None)
        W["param_stat"].on_change(
            "value", lambda a, o, n: self.set_param(self.state.param, n) if n else None
        )
        W["component"].on_change(
            "value", lambda a, o, n: self.set_component(None if n == "total" else int(n))
        )
        W["mode"].on_change("active", lambda a, o, n: self.set_mode(S.SELECTION_MODES[n]))
        W["radius"].on_change("value", lambda a, o, n: self.set_radius(float(n)))
        W["region"].on_change(
            "value", lambda a, o, n: self.select_region(n) if n != "(none)" else None
        )
        W["clear"].on_click(lambda: self.clear_selection())
        W["keep"].on_click(lambda: self.keep_selection())
        W["export"].on_click(lambda: self.export_regions())
        W["sed_options"].on_change(
            "active", lambda a, o, n: self.set_sed_options({_SED_OPTIONS[i] for i in n})
        )
        W["image_bundle"].on_change(
            "value", lambda a, o, n: self.set_image_bundle(0 if n == "bundle A" else 1)
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _set(widget: Any, attr: str, value: Any) -> None:
        if getattr(widget, attr) != value:
            setattr(widget, attr, value)

    def _timed(self, fn) -> None:
        t0 = time.perf_counter()
        fn()
        self.last_update_ms = 1e3 * (time.perf_counter() - t0)
        self._update_status()

    def _update_status(self) -> None:
        reg = S.selection_region(self.bundle, self.state)
        sel = "none (default region)" if self.state.selection is None else reg.name
        notes = "".join(f"<br><i>{n}</i>" for n in self._notes)
        self.status.text = (
            f"selection: {sel}, {reg.n_pix} px<br>update: {self.last_update_ms:.0f} ms{notes}"
        )

    # ------------------------------------------------------------------ image refresh

    def _image_title(self, panel: int) -> str:
        st = self.state
        band = self.bundle.short_names[st.band]
        tag = " [B]" if st.image_bundle == 1 else ""
        if panel in (0, 1):
            kind = st.kinds[panel]
            if st.component is not None and kind in ("model", "intrinsic"):
                kind = f"comp {st.component} {kind}"
            derived = st.kinds[panel] in ("model", "intrinsic", "residual", "chi")
            stat = f" ({st.model_stat})" if derived else ""
            return f"{band} {kind}{stat}{tag}"
        if st.param is None:
            return ""
        label = dict(self._param_options).get(st.param, st.param)
        extra = f" ({band})" if st.param.startswith("__comp") else ""
        return f"{label}{extra} [{st.param_stat}]{tag}"

    def _band_in(self, bundle: ResultsBundle) -> int | None:
        name = self.bundle.band_names[self.state.band]
        return bundle.band_names.index(name) if name in bundle.band_names else None

    def _refresh_image(self, panel: int) -> None:
        st = self.state
        p = self.images[panel]
        bb = self.image_bundle
        if panel == 2 and st.param is None:
            p.fig.visible = False
            return
        if panel == 2:
            p.fig.visible = True
        note = ""
        try:
            band = self._band_in(bb)
            if band is None:
                raise KeyError("band not in bundle B")
            local = S.ViewerState(**{**st.__dict__, "band": band})
            raw = S.image_for(bb, local, panel)
            if getattr(bb, "last_fallback", False):
                note = " [MAP fallback]"
        except (KeyError, ValueError, IndexError) as err:
            raw = np.full(self.bundle.image_shape, np.nan)
            note = f" [n/a: {str(err).strip(chr(39))}]"
        sym = S.is_symmetric(st, panel)
        disp = S.display_for(raw, st, symmetric=sym)
        p.source.data = {"img": [disp["img"]], "raw": [raw]}
        p.mapper.palette = P.palette("RdBu" if sym else st.cmap)
        p.ticker.ticks = disp["ticks"]
        p.formatter.args = {"pos": disp["ticks"], "labels": disp["tick_labels"]}
        p.fig.title.text = self._image_title(panel) + note

    def _refresh_images(self, panels: Sequence[int] = (0, 1, 2)) -> None:
        for i in panels:
            self._refresh_image(i)

    # ------------------------------------------------------------------ selection refresh

    def _refresh_selection(self) -> None:
        """Recompute everything that depends on the selected region."""
        st = self.state
        self.sources["selection"].data = (
            _pixel_cols(st.selection.mask) if st.selection is not None else {"x": [], "y": []}
        )
        for i, src in enumerate(self.sources["overlay_pixels"]):
            reg = st.overlays[i] if i < len(st.overlays) else None
            src.data = _pixel_cols(reg.mask) if reg is not None else {"x": [], "y": []}
        self._refresh_sed()
        self._refresh_hists()
        self._update_status()

    def _refresh_sed(self) -> None:
        b = self.bundle
        st = self.state
        self._seds = S.sed_for(b, st)
        main = self._seds[0]
        self.sources["sed"].data = S.sed_columns(main)
        res = S.residual_columns(main)
        self.sources["residual"].data = res
        chi = np.asarray(res["chi"], dtype=float)
        lim = max(3.5, 1.15 * float(np.nanmax(np.abs(chi)))) if np.isfinite(chi).any() else 3.5
        yr = self.residual_figs["chi"].y_range
        yr.start, yr.end = -lim, lim
        self.sources["total"].data = S.sed_columns(self._total)
        for i, src in enumerate(self.sources["overlays"]):
            src.data = (
                S.sed_columns(self._seds[i + 1]) if i + 1 < len(self._seds) else P._empty_sed_cols()
            )
        self._component_seds = S.component_seds_for(b, st)
        comp_cols = {}
        if b.components is not None:
            sel = range(b.components.n_components) if st.component is None else [st.component]
            comp_cols = dict(zip(sel, self._component_seds))
        for k, src in enumerate(self.sources["components"]):
            src.data = S.sed_columns(comp_cols[k]) if k in comp_cols else P._empty_sed_cols()
        self._compare_sed = None
        if self.compare is not None:
            reg = S.selection_region(b, st)
            self._compare_sed = region_sed(self.compare, reg.mask, label=f"B: {reg.name}")
            self.sources["compare"].data = S.sed_columns(self._compare_sed)
        for fig in (self.sed_lin.fig, self.sed_log.fig):
            fig.title.text = f"SED: {main.label}"
        self._apply_sed_visibility()

    def _apply_sed_visibility(self) -> None:
        st = self.state
        opts = st.sed_options
        b = self.bundle
        has_truth = b.truth is not None
        vis = {
            "obs": True,
            "model": True,
            "band": self.show_uncertainty,
            "map": "map" in opts,
            "intrinsic": "intrinsic" in opts and b.has_intrinsic,
            "intrinsic_band": "intrinsic" in opts and b.has_intrinsic and self.show_uncertainty,
            "truth": "truth" in opts and has_truth,
            "total": "total" in opts,
            "components": "components" in opts and b.components is not None,
            "compare": self.compare is not None,
            "overlays": True,
        }
        for panel in (self.sed_lin, self.sed_log):
            for key, rends in panel.groups.items():
                for r in rends:
                    self._set(r, "visible", vis[key])
        log_y = "log_y" in opts
        self._set(self.sed_lin.fig, "visible", not log_y)
        self._set(self.sed_log.fig, "visible", log_y)

    def _refresh_hists(self) -> None:
        self._hists = S.hist_for(self.bundle, self.state)
        keys = list(self._hists)
        show = bool(keys) and not self.bundle.degraded
        self.hist_row.visible = show
        if self.bundle.degraded:
            self.hist_note.text = (
                "<i>Posterior histograms unavailable: degraded (MAP-only) bundle.</i>"
            )
            self.hist_note.visible = True
        else:
            self.hist_note.visible = False
        units = self.bundle.units or {}
        for i, hp in enumerate(self.hist_panels):
            if i >= len(keys):
                hp.fig.visible = False
                continue
            e = self._hists[keys[i]]
            hp.fig.visible = True
            hp.source.data = {"left": e["left"], "right": e["right"], "top": e["top"]}
            name = keys[i].split(":", 1)[-1]
            unit = e.get("unit") or units.get(name, "")
            hp.fig.title.text = f"{e['label']}: {e['summary']}"
            hp.fig.xaxis.axis_label = f"{name} [{unit}]" if unit else name
            hp.truth.visible = e["truth"] is not None and np.isfinite(e["truth"])
            if hp.truth.visible:
                hp.truth.location = float(e["truth"])
            hp.best.visible = e["map"] is not None and np.isfinite(e["map"])
            if hp.best.visible:
                hp.best.location = float(e["map"])

    # ------------------------------------------------------------------ controller API

    def set_band(self, b: int) -> None:
        """Show band ``b`` in the band panels (and component fractions)."""
        if not 0 <= int(b) < self.bundle.n_bands:
            raise IndexError(f"band {b} out of range")
        self.state.band = int(b)
        self._set(self.widgets["band"], "value", str(int(b)))
        self._timed(lambda: self._refresh_images())

    def set_image_kind(self, panel: int, kind: str, stat: str | None = None) -> None:
        """Set the kind (and optionally the model stat) of band panel 0 or 1."""
        if panel not in (0, 1):
            raise ValueError("panel must be 0 or 1")
        if kind not in self._kinds:
            raise KeyError(f"image kind {kind!r} not available (have {self._kinds})")
        self.state.kinds[panel] = kind
        self._set(self.widgets[f"kind{panel}"], "value", kind)
        if stat is not None:
            self.set_model_stat(stat)
        else:
            self._timed(lambda: self._refresh_image(panel))

    def set_model_stat(self, stat: str) -> None:
        """Model statistic for model-derived images."""
        stats = S.model_stats(self.bundle)
        if stat not in stats:
            raise KeyError(f"model stat {stat!r} not available (have {stats})")
        self.state.model_stat = stat
        self._set(self.widgets["model_stat"], "value", stat)
        self._timed(lambda: self._refresh_images())

    def set_param(self, name: str, stat: str | None = None) -> None:
        """Show parameter ``name`` (a map name or ``"__comp<k>"``) with ``stat`` in panel 3."""
        keys = [k for k, _ in self._param_options]
        if name not in keys:
            raise KeyError(f"parameter {name!r} not available (have {keys})")
        stats = S.param_stats(self.bundle, name)
        if stat is None:
            stat = self.state.param_stat if self.state.param_stat in stats else stats[0]
            if "p50" in stats and self.state.param != name and self.state.param_stat not in stats:
                stat = "p50"
        if stat not in stats:
            raise KeyError(f"parameter stat {stat!r} not available (have {stats})")
        self.state.param = name
        self.state.param_stat = stat
        W = self.widgets
        self._set(W["param_stat"], "options", stats)
        self._set(W["param"], "value", name)
        self._set(W["param_stat"], "value", stat)
        self._timed(lambda: self._refresh_image(2))

    def set_display(
        self,
        *,
        stretch: str | None = None,
        pct: tuple[float, float] | None = None,
        cmap: str | None = None,
    ) -> None:
        """Change the display stretch, percentile range or colormap."""
        st = self.state
        W = self.widgets
        if stretch is not None:
            st.stretch = stretch
            self._set(W["stretch"], "value", stretch)
        if pct is not None:
            st.pct = (float(pct[0]), float(pct[1]))
            self._set(W["pct"], "value", st.pct)
        if cmap is not None:
            P.palette(cmap)
            st.cmap = cmap
            self._set(W["cmap"], "value", cmap)
        self._timed(lambda: self._refresh_images())

    def set_model_floor(self, on: bool) -> None:
        """Include the model-error floor in the chi images."""
        self.state.model_floor = bool(on)
        self._set(self.widgets["floor"], "active", bool(on))
        self._timed(lambda: self._refresh_images((0, 1)))

    def set_mask_outside_segment(self, on: bool) -> None:
        """Blank the images outside the segment region."""
        self.state.mask_outside_segment = bool(on)
        self._set(self.widgets["mask_segment"], "active", bool(on))
        self._timed(lambda: self._refresh_images())

    def set_component(self, k: int | None) -> None:
        """Select component ``k`` (None = total); additive bundles only."""
        comps = self.bundle.components
        if k is not None:
            if comps is None:
                raise ValueError("This bundle has no components")
            if not 0 <= int(k) < comps.n_components:
                raise IndexError(f"component {k} out of range")
            k = int(k)
        self.state.component = k
        self._set(self.widgets["component"], "value", "total" if k is None else str(k))

        def _do():
            self._refresh_images()
            self._refresh_selection()

        self._timed(_do)

    def set_mode(self, mode: str) -> None:
        """Selection mode: ``pixel``, ``circle``, ``box``, ``lasso`` or ``polygon``."""
        if mode not in S.SELECTION_MODES:
            raise ValueError(f"mode must be one of {S.SELECTION_MODES}")
        self.state.mode = mode
        self._set(self.widgets["mode"], "active", S.SELECTION_MODES.index(mode))
        self._apply_mode()

    def _apply_mode(self) -> None:
        mode = self.state.mode
        for p in self.images:
            tb = p.fig.toolbar
            if mode in ("box", "lasso"):
                tb.active_drag = p.tools[mode]
                tb.active_tap = None
            elif mode == "polygon":
                tb.active_drag = p.tools["pan"]
                tb.active_tap = p.tools["polygon"]
            else:
                tb.active_drag = p.tools["pan"]
                tb.active_tap = None

    def set_radius(self, r: float) -> None:
        """Circle radius in pixels used by tap in circle mode."""
        self.state.radius = float(r)
        self._set(self.widgets["radius"], "value", float(r))

    def _set_selection(self, region: Region | None) -> None:
        self.state.selection = region
        self._suppress_selection = True
        try:
            if self.sources["pixels"].selected.indices:
                self.sources["pixels"].selected.indices = []
        finally:
            self._suppress_selection = False
        if region is None or region.name not in self.bundle.regions:
            self._set(self.widgets["region"], "value", "(none)")
        self._timed(self._refresh_selection)

    def select_pixel(self, y: int, x: int) -> None:
        """Select the single pixel ``[y, x]``."""
        h, w = self.bundle.image_shape
        y, x = int(y), int(x)
        if not (0 <= y < h and 0 <= x < w):
            return
        self._set_selection(Region.pixel((h, w), y, x))

    def select_indices(self, flat_indices: Sequence[int]) -> None:
        """Select pixels by flat index ``y * W + x`` (box, lasso and polygon selections)."""
        idx = list(flat_indices)
        if not idx:
            self._set_selection(None)
            return
        reg = S.region_from_indices(self.bundle.image_shape, idx)
        self.state.selection = reg
        self._set(self.widgets["region"], "value", "(none)")
        self._timed(self._refresh_selection)

    def select_circle(self, cy: float, cx: float, r: float) -> None:
        """Select the pixels whose centres are within ``r`` of ``(cy, cx)``."""
        self._set_selection(Region.circle(self.bundle.image_shape, cy, cx, r))

    def select_region(self, name: str) -> None:
        """Select one of the bundle's named regions."""
        if name not in self.bundle.regions:
            raise KeyError(f"region {name!r} not in bundle (have {list(self.bundle.regions)})")
        self._set(self.widgets["region"], "value", name)
        self._set_selection(self.bundle.regions[name])

    def clear_selection(self) -> None:
        """Clear the selection (the SED falls back to the default total region)."""
        self._set_selection(None)

    def keep_selection(self) -> None:
        """Keep the current selection as a coloured overlay (at most :data:`MAX_OVERLAYS`)."""
        sel = self.state.selection
        if sel is None:
            return
        ov = self.state.overlays
        ov.append(sel)
        del ov[:-MAX_OVERLAYS]
        self._timed(self._refresh_selection)

    def set_sed_options(self, options: set[str] | Sequence[str]) -> None:
        """Enable SED options (subset of ``intrinsic, truth, map, log_y, total, components``)."""
        opts = set(options)
        bad = opts - set(_SED_OPTIONS)
        if bad:
            raise ValueError(f"unknown SED options {sorted(bad)}")
        self.state.sed_options = opts
        self._set(
            self.widgets["sed_options"],
            "active",
            [i for i, o in enumerate(_SED_OPTIONS) if o in opts],
        )
        self._apply_sed_visibility()

    def set_image_bundle(self, i: int) -> None:
        """Show bundle A (0) or the comparison bundle B (1) in the image panels."""
        if i not in (0, 1) or (i == 1 and self.compare is None):
            raise ValueError("image bundle must be 0, or 1 with a comparison bundle")
        self.state.image_bundle = i
        self._set(self.widgets["image_bundle"], "value", "bundle A" if i == 0 else "bundle B")
        self._timed(lambda: self._refresh_images())

    def current_seds(self) -> list[RegionSED]:
        """SEDs shown now: the main selection first, then the kept overlays."""
        return list(self._seds)

    def current_component_seds(self) -> list[RegionSED]:
        """Per-component SEDs over the current selection (empty without components)."""
        return list(self._component_seds)

    def current_hists(self) -> dict[str, dict]:
        """Histogram entries shown now (see :func:`arachne.gui.state.hist_for`)."""
        return dict(self._hists)

    def export_regions(self, path: str | Path | None = None) -> Path:
        """Write the selection and overlays to ``<bundle>.regions.json`` (bundle untouched).

        Args:
            path: Output path (default: next to the bundle).

        Returns:
            The path written.
        """
        if path is None:
            if self.bundle.path is None:
                raise ValueError("in-memory bundle: pass an explicit path")
            path = Path(str(self.bundle.path) + ".regions.json")
        regs = list(self.state.overlays)
        if self.state.selection is not None:
            regs.append(self.state.selection)
        out = []
        for reg in regs:
            yy, xx = np.nonzero(reg.mask)
            out.append(
                {
                    "name": reg.name,
                    "kind": reg.kind,
                    "geometry": reg.geometry,
                    "pixels_yx": np.column_stack([yy, xx]).tolist(),
                }
            )
        path = Path(path)
        path.write_text(json.dumps({"shape": list(self.bundle.image_shape), "regions": out}))
        self._notes = [f"exported {len(out)} regions to {path.name}"]
        self._update_status()
        return path

    # ------------------------------------------------------------------ events

    def _on_tap(self, event: events.Tap) -> None:
        if event.x is None or event.y is None:
            return
        y, x = float(event.y), float(event.x)
        if self.state.mode == "pixel":
            self.select_pixel(int(round(y)), int(round(x)))
        elif self.state.mode == "circle":
            self.select_circle(y, x, self.state.radius)

    def _on_indices(self, attr: str, old: list[int], new: list[int]) -> None:
        if self._suppress_selection or not new:
            return
        self.select_indices(new)

    # ------------------------------------------------------------------ serving

    def modify_doc(self, doc: Any) -> None:
        """Add the layout to a bokeh ``Document`` (server or notebook)."""
        doc.add_root(self.layout)
        doc.title = f"arachne: {self.bundle.title or 'results'}"

    def _session_doc(self, doc: Any) -> None:
        # bokeh models belong to one document: every further session gets its own viewer
        viewer = self if self.layout.document is None else ArachneViewer(self.bundle, self.compare)
        viewer.modify_doc(doc)

    def application(self) -> Application:
        """A bokeh ``Application`` serving one viewer per session."""
        return Application(FunctionHandler(self._session_doc))

    def show(self) -> Any:
        """Display in Jupyter: a ``panel.pane.Bokeh`` when panel is installed, else bokeh show.

        Returns:
            The Panel pane (display it as the last expression of a cell), or None.
        """
        try:
            import panel as pn
        except ImportError:
            pn = None
        if pn is not None:
            pn.extension()
            return pn.pane.Bokeh(self.layout)
        from bokeh.io import output_notebook, show

        output_notebook()
        show(self._session_doc)
        return None

    def serve(
        self,
        port: int = 5006,
        address: str = "127.0.0.1",
        open_browser: bool = True,
        allow_websocket_origin: Sequence[str] | None = None,
    ) -> None:
        """Run a bokeh server for this viewer (blocks until interrupted).

        Args:
            port: Port.
            address: Interface to bind (``127.0.0.1`` = local only; use SSH forwarding).
            open_browser: Open a browser tab.
            allow_websocket_origin: Extra allowed origins (``HOST:PORT``), e.g. when the
                forwarded local port differs from ``port``.
        """
        from bokeh.server.server import Server

        origins = [f"localhost:{port}", f"127.0.0.1:{port}"]
        if address not in ("127.0.0.1", "localhost", "0.0.0.0", ""):
            origins.append(f"{address}:{port}")
        origins += list(allow_websocket_origin or [])
        server = Server(
            {"/": self.application()},
            port=port,
            address=address,
            allow_websocket_origin=origins,
        )
        server.start()
        print(f"arachne viewer running at http://{address or 'localhost'}:{server.port}/")
        if open_browser:
            server.io_loop.add_callback(server.show, "/")
        try:
            server.io_loop.start()
        except KeyboardInterrupt:
            print("stopping")
        finally:
            server.stop()


def view(path: ResultsBundle | str | Path, **kw: Any) -> ArachneViewer:
    """Open a bundle in an :class:`ArachneViewer` (``kw`` go to the constructor)."""
    return ArachneViewer(path, **kw)
