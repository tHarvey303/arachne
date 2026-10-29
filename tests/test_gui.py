"""Smoke tests of the bokeh viewer (``arachne.gui``); no server is ever started."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

pytest.importorskip("bokeh")

from bokeh.document import Document  # noqa: E402

from arachne.gui import ArachneViewer, cli, view  # noqa: E402
from arachne.gui import state as S  # noqa: E402
from arachne.results import (  # noqa: E402
    Region,
    ResultsBundle,
    make_synthetic_bundle,
    make_synthetic_content,
    region_param_summary,
    region_sed,
)

FLAVOURS = ("field", "additive", "degraded")


def _viewer(flavour: str, **kw) -> ArachneViewer:
    return ArachneViewer(ResultsBundle.from_content(make_synthetic_content(flavour, **kw)))


def _check_sed_source(v: ArachneViewer, region) -> None:
    ref = region_sed(v.bundle, region)
    data = v.sources["sed"].data
    np.testing.assert_allclose(data["obs"], ref.obs_flux)
    np.testing.assert_allclose(data["p50"], ref.model_percentiles((50,))[0])
    np.testing.assert_allclose(data["map"], ref.model_map)
    np.testing.assert_allclose(v.sources["residual"].data["chi"], ref.chi())
    np.testing.assert_allclose(v.current_seds()[0].obs_flux, ref.obs_flux)


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_viewer_builds_and_serialises(flavour):
    """The layout builds and serialises into a bokeh Document."""
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    v = _viewer(flavour)
    html = file_html(v.layout, resources=CDN, title="t")
    assert "bokeh" in html and len(html) > 1000
    v = _viewer(flavour)  # file_html leaves models attached to its own document
    doc = Document()
    v.modify_doc(doc)
    assert v.layout.document is doc
    assert "synthetic" in v.header.text


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_controller_images(flavour):
    """Image sources hold ``band_image``/``param_map`` arrays after each controller call."""
    v = _viewer(flavour)
    b = v.bundle
    stat = S.model_stats(b)[0]
    for band in (2, 0):
        v.set_band(band)
        assert v.widgets["band"].value == str(band)
        for kind in S.image_kinds(b):
            v.set_image_kind(0, kind)
            np.testing.assert_array_equal(
                v.images[0].source.data["raw"][0], b.band_image(band, kind, stat)
            )
            assert v.widgets["kind0"].value == kind
    v.set_image_kind(1, "chi")
    v.set_model_floor(False)
    np.testing.assert_array_equal(
        v.images[1].source.data["raw"][0], b.band_image(0, "chi", stat, model_floor=False)
    )
    disp = v.images[1].source.data["img"][0]
    assert np.nanmin(disp) >= 0 and np.nanmax(disp) <= 1
    v.set_image_kind(0, "data")
    v.set_display(stretch="linear", pct=(1.0, 99.0), cmap="Magma256")
    assert v.images[0].mapper.palette[0] != v.images[1].mapper.palette[0]  # chi is diverging
    if b.map_names:
        for pstat in b.available_param_stats():
            v.set_param("Av", pstat)
            np.testing.assert_array_equal(
                v.images[2].source.data["raw"][0], b.param_map("Av", pstat)
            )
        assert v.images[2].fig.visible
    with pytest.raises(KeyError):
        v.set_image_kind(0, "nonsense")


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_controller_selections(flavour):
    """Every selection path updates the SED, residual and histogram sources."""
    v = _viewer(flavour)
    b = v.bundle
    shape = b.image_shape
    _check_sed_source(v, b.regions["all"])  # default region

    v.select_pixel(5, 9)
    _check_sed_source(v, Region.pixel(shape, 5, 9))
    assert len(v.sources["selection"].data["x"]) == 1

    v.select_circle(8.0, 7.0, 3.0)
    _check_sed_source(v, Region.circle(shape, 8.0, 7.0, 3.0))

    idx = [17, 18, 33, 34, 35]
    v.select_indices(idx)
    reg = S.region_from_indices(shape, idx)
    _check_sed_source(v, reg)
    assert v.state.selection.n_pix == 5

    v.select_region("core")
    _check_sed_source(v, b.regions["core"])
    assert v.widgets["region"].value == "core"
    if b.map_names and not b.degraded:
        ref = region_param_summary(b, b.regions["core"])
        hp = v.hist_panels[0]
        assert hp.fig.visible and v.hist_row.visible
        assert sum(hp.source.data["top"]) == len(ref["log_mass"])
        assert hp.truth.visible

    v.keep_selection()
    v.select_pixel(2, 2)
    assert len(v.current_seds()) == 2
    np.testing.assert_allclose(
        v.sources["overlays"][0].data["obs"], region_sed(b, b.regions["core"]).obs_flux
    )
    assert len(v.sources["overlay_pixels"][0].data["x"]) == b.regions["core"].n_pix

    v.clear_selection()
    assert v.state.selection is None
    _check_sed_source(v, b.regions["all"])


def test_browser_callbacks_route_to_controller():
    """Tap events and selection-index changes go through the controller methods."""
    from bokeh import events

    v = _viewer("field")
    v.set_mode("pixel")
    v._on_tap(events.Tap(v.images[0].fig, x=3.2, y=6.7))
    assert v.state.selection.geometry == {"y": 7, "x": 3}
    v.set_mode("circle")
    v.set_radius(2.0)
    v._on_tap(events.Tap(v.images[0].fig, x=8.0, y=8.0))
    assert v.state.selection.kind == "circle"
    v.set_mode("box")
    v._on_tap(events.Tap(v.images[0].fig, x=1.0, y=1.0))
    assert v.state.selection.kind == "circle"  # taps ignored in box mode
    assert v.images[0].fig.toolbar.active_drag is v.images[0].tools["box"]
    v.sources["pixels"].selected.indices = [0, 1, 2]
    assert v.state.selection.n_pix == 3
    v.select_pixel(4, 4)  # programmatic selection clears the browser selection silently
    assert v.sources["pixels"].selected.indices == []
    assert v.state.selection.n_pix == 1
    v.widgets["band"].value = "1"
    assert v.state.band == 1
    v.widgets["sed_options"].active = [3]  # log y only
    assert v.state.sed_options == {"log_y"}
    assert v.sed_log.fig.visible and not v.sed_lin.fig.visible


def test_degraded_hides_uncertainty():
    """Degraded: banner, no uncertainty bands, no histograms, MAP-only model."""
    v = _viewer("degraded")
    assert v.banner.visible and "Degraded" in v.banner.text
    assert not v.show_uncertainty
    assert all(not r.visible for r in v.sed_lin.groups["band"])
    assert all(not r.visible for r in v.sed_lin.groups["intrinsic"])
    v.select_region("core")
    assert not v.hist_row.visible and v.hist_note.visible
    data = v.sources["sed"].data
    np.testing.assert_allclose(data["p50"], data["map"])
    assert v.widgets["model_stat"].options == ["map"]
    full = _viewer("field")
    assert not full.banner.visible
    assert all(r.visible for r in full.sed_lin.groups["band"])


def test_components_only_for_additive():
    """The component selector exists only for additive bundles and drives SEDs/images."""
    assert not _viewer("field").widgets["component"].visible
    v = _viewer("additive")
    assert v.widgets["component"].visible
    assert len(v.sources["components"]) == 2
    v.select_region("core")
    assert len(v.current_component_seds()) == 2
    assert all(len(s.data["x"]) == 3 for s in v.sources["components"])
    v.set_component(1)
    assert v.widgets["component"].value == "1"
    assert len(v.sources["components"][0].data["x"]) == 0
    v.set_image_kind(0, "model")
    np.testing.assert_allclose(
        v.images[0].source.data["raw"][0], v.bundle.components.image(1, "convolved", "p50")[0]
    )
    assert any(k.startswith("comp1:") for k in v.current_hists())
    v.set_param("__comp0", "map")
    assert "fraction" in v.images[2].fig.title.text
    v.set_component(None)
    assert v.state.component is None
    with pytest.raises(ValueError):
        _viewer("field").set_component(0)


def test_compare_and_files(tmp_path):
    """Compare mode overlays bundle B; ``view`` loads paths; regions export as JSON."""
    a = make_synthetic_bundle(tmp_path / "a.h5", "field")
    bpath = make_synthetic_bundle(tmp_path / "b.h5", "degraded")
    v = view(a, compare=bpath)
    assert v.widgets["image_bundle"].visible
    v.select_region("core")
    ref = region_sed(v.compare, v.bundle.regions["core"])
    np.testing.assert_allclose(v.sources["compare"].data["p50"], ref.model_map)
    v.set_image_bundle(1)
    assert "[B]" in v.images[0].fig.title.text
    v.keep_selection()
    out = v.export_regions()
    assert out == tmp_path / "a.h5.regions.json"
    doc = json.loads(out.read_text())
    assert [r["name"] for r in doc["regions"]] == ["core", "core"]
    with pytest.raises(ValueError):
        ArachneViewer(
            ResultsBundle.from_content(make_synthetic_content("field", shape=(12, 12))),
            compare=a,
        )


def test_no_draws_field():
    """S=0 without the degraded flag: MAP-only, histograms show MAP spans."""
    v = _viewer("field", n_draws=0)
    assert v.banner.visible and "No posterior draws" in v.banner.text
    v.select_pixel(8, 8)
    hp = v.hist_panels[0]
    assert hp.fig.visible and hp.best.visible and len(hp.source.data["top"]) == 0


def test_cli_help(capsys):
    """``arachne-gui --help`` works."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--no-browser" in out and "--allow-websocket-origin" in out and "ssh -L" in out
    assert "ssh -L 5099:localhost:5099" in cli.remote_recipe(5099, "node")


def test_panel_pane():
    """The layout renders as a Panel pane (Jupyter path)."""
    pn = pytest.importorskip("panel")
    v = _viewer("field")
    pn.pane.Bokeh(v.layout).get_root()


def test_missing_bokeh_message(monkeypatch):
    """Without bokeh, the viewer import raises a friendly ImportError."""
    import arachne.gui as gui

    monkeypatch.setitem(sys.modules, "bokeh", None)
    with pytest.raises(ImportError, match=r"arachne\[gui\]"):
        gui.require_bokeh()
    with pytest.raises(ImportError, match=r"arachne\[gui\]"):
        gui.__getattr__("ArachneViewer")
