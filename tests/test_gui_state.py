"""Tests for the numpy-only viewer state functions (``arachne.gui.state``)."""

from __future__ import annotations

import numpy as np
import pytest

from arachne.gui import state as S
from arachne.results import (
    Region,
    ResultsBundle,
    make_synthetic_content,
    region_param_summary,
    region_sed,
)

FLAVOURS = ("field", "additive", "degraded")


@pytest.fixture(scope="module")
def bundles():
    """One in-memory bundle per synthetic flavour."""
    return {f: ResultsBundle.from_content(make_synthetic_content(f)) for f in FLAVOURS}


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_default_state_and_menus(bundles, flavour):
    """Defaults only offer products the bundle has."""
    b = bundles[flavour]
    st = S.default_state(b)
    assert S.band_order(b) == [0, 1, 2]
    kinds = S.image_kinds(b)
    assert {"data", "model", "residual", "chi"} <= set(kinds)
    assert ("intrinsic" in kinds) == b.has_intrinsic
    if flavour == "degraded":
        assert S.model_stats(b) == ["map"]
        assert st.model_stat == "map"
    else:
        assert st.model_stat == "p50"
    keys = [k for k, _ in S.param_choices(b)]
    if flavour == "additive":
        assert keys == ["__comp0", "__comp1"]
        assert "components" in st.sed_options
    else:
        assert keys == ["log_mass", "Av"]
        assert st.param == "log_mass"
        assert st.param_stat == "p50"
        assert set(S.param_stats(b, "log_mass")) == set(b.available_param_stats())


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_image_for_matches_band_image(bundles, flavour):
    """Band panels are exactly ``band_image``; the parameter panel is ``param_map``."""
    b = bundles[flavour]
    st = S.default_state(b)
    for band in range(b.n_bands):
        st.band = band
        for kind in S.image_kinds(b):
            st.kinds[0] = kind
            for stat in S.model_stats(b):
                st.model_stat = stat
                got = S.image_for(b, st, 0)
                want = b.band_image(band, kind, S._image_stat(kind, stat))
                np.testing.assert_array_equal(got, want)
    if b.map_names:
        st.param, st.param_stat = "Av", "map"
        np.testing.assert_array_equal(S.image_for(b, st, 2), b.param_map("Av", "map"))


def test_chi_model_floor(bundles):
    """The floor toggle switches between the two chi flavours."""
    b = bundles["field"]
    st = S.default_state(b)
    st.kinds[1] = "chi"
    st.model_floor = False
    np.testing.assert_array_equal(S.image_for(b, st, 1), b.band_image(0, "chi", model_floor=False))
    st.model_floor = True
    np.testing.assert_array_equal(S.image_for(b, st, 1), b.band_image(0, "chi", model_floor=True))


def test_component_images_and_fractions(bundles):
    """Selected component images come from the components view; fractions sum to one."""
    b = bundles["additive"]
    st = S.default_state(b)
    st.component = 1
    st.kinds[0] = "model"
    np.testing.assert_allclose(S.image_for(b, st, 0), b.components.image(1, "convolved", "p50")[0])
    st.param_stat = "map"
    fracs = []
    for k in range(2):
        st.param = f"__comp{k}"
        fracs.append(S.image_for(b, st, 2))
    total = np.nansum(fracs, axis=0)
    np.testing.assert_allclose(total[np.isfinite(fracs[0])], 1.0, rtol=1e-6)


def test_mask_outside_segment():
    """Pixels outside the segment region are blanked."""
    c = make_synthetic_content("field")
    seg = np.zeros((16, 16), dtype=bool)
    seg[4:12, 4:12] = True
    c.regions["segment"] = Region.from_mask(seg, name="segment")
    b = ResultsBundle.from_content(c)
    st = S.default_state(b)
    st.mask_outside_segment = True
    for panel in (0, 2):
        img = S.image_for(b, st, panel)
        assert np.isnan(img[~seg]).all()
        assert np.isfinite(img[seg]).all()
    assert S.selection_region(b, st).name == "segment"


def test_display_for():
    """Stretched images live in [0, 1]; symmetric ones are centred on 0.5."""
    st = S.ViewerState()
    img = np.linspace(-3, 5, 64).reshape(8, 8)
    img[0, 0] = np.nan
    d = S.display_for(img, st)
    good = np.isfinite(d["img"])
    assert np.isnan(d["img"][0, 0]) and good.sum() == 63
    assert d["img"][good].min() >= 0 and d["img"][good].max() <= 1
    assert len(d["ticks"]) == len(d["tick_labels"]) == 5
    sym = S.display_for(img, st, symmetric=True)
    assert sym["lo"] == -sym["hi"]
    assert sym["ticks"][2] == pytest.approx(0.5)
    st.pct = (0.0, 100.0)
    st.stretch = "linear"
    lin = S.display_for(np.arange(4.0).reshape(2, 2), st)
    np.testing.assert_allclose(lin["img"].ravel(), [0, 1 / 3, 2 / 3, 1])


def test_region_from_indices():
    """Flat indices map to ``y * W + x``."""
    reg = S.region_from_indices((4, 5), [0, 7, 19])
    assert reg.n_pix == 3
    assert reg.mask[0, 0] and reg.mask[1, 2] and reg.mask[3, 4]
    assert S.region_from_indices((4, 5), []).n_pix == 0


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_sed_for_and_columns(bundles, flavour):
    """The main SED equals ``region_sed`` and the columns are its values."""
    b = bundles[flavour]
    st = S.default_state(b)
    main = S.sed_for(b, st)[0]
    assert main.label == "all"  # default region without a selection
    st.selection = Region.circle(b.image_shape, 7.5, 7.5, 2.5)
    st.overlays = [b.regions["core"]]
    seds = S.sed_for(b, st)
    assert len(seds) == 2
    ref = region_sed(b, st.selection)
    cols = S.sed_columns(seds[0])
    np.testing.assert_allclose(cols["obs"], ref.obs_flux)
    np.testing.assert_allclose(cols["obs_hi"] - cols["obs"], ref.obs_err)
    p16, p50, p84 = ref.model_percentiles()
    np.testing.assert_allclose(cols["p50"], p50)
    np.testing.assert_allclose(cols["p16"], p16)
    np.testing.assert_allclose(cols["map"], ref.model_map)
    assert cols["band"] == ["F115W", "F200W", "F444W"]
    if flavour == "degraded":
        np.testing.assert_allclose(cols["p16"], cols["p84"])  # MAP repeated, no band
        assert np.isnan(cols["int_p50"]).all()
    else:
        assert (cols["p84"] >= cols["p16"]).all()
        np.testing.assert_allclose(cols["truth_conv"], ref.truth_convolved)
    res = S.residual_columns(seds[0])
    np.testing.assert_allclose(res["chi"], ref.chi())
    np.testing.assert_allclose(res["frac"], ref.frac_residual())
    assert S.total_for(b).label.startswith("total")


def test_component_seds_for(bundles):
    """All components by default, only component k when selected; none without components."""
    b = bundles["additive"]
    st = S.default_state(b)
    assert len(S.component_seds_for(b, st)) == 2
    st.component = 0
    (only,) = S.component_seds_for(b, st)
    assert only.label.startswith("comp 0")
    assert S.component_seds_for(bundles["field"], S.default_state(bundles["field"])) == []


def test_hist_for_field(bundles):
    """Region histograms use ``region_param_summary``; truth is aggregated the same way."""
    b = bundles["field"]
    st = S.default_state(b)
    st.selection = b.regions["core"]
    h = S.hist_for(b, st)
    assert list(h) == ["log_mass", "Av", "scalar:total_log_mass"]
    ref = region_param_summary(b, b.regions["core"])
    np.testing.assert_allclose(np.sort(h["log_mass"]["values"]), np.sort(ref["log_mass"]))
    assert h["log_mass"]["top"].sum() == b.n_draws
    assert h["log_mass"]["truth"] is not None and h["Av"]["truth"] is not None
    assert h["scalar:total_log_mass"]["truth"] == pytest.approx(float(b.truth.scalars[0]))
    st.selection = Region.pixel(b.image_shape, 3, 4)
    hp = S.hist_for(b, st)
    np.testing.assert_allclose(hp["Av"]["values"], b.pixel_param_draws(3, 4)["Av"])
    assert hp["Av"]["truth"] == pytest.approx(b.truth.param_maps[1, 3, 4])


def test_hist_for_degraded_and_additive(bundles):
    """Degraded bundles have no histograms; additive ones show the selected component."""
    deg = bundles["degraded"]
    assert S.hist_for(deg, S.default_state(deg)) == {}
    b = bundles["additive"]
    st = S.default_state(b)
    assert list(S.hist_for(b, st)) == ["scalar:total_log_mass"]
    st.component = 1
    h = S.hist_for(b, st)
    assert {"comp1:log_mass", "comp1:Av"} <= set(h)
    assert "comp1:sersic_n" not in h  # all NaN
    assert "comp1:mu_y" not in h  # constant


def test_hist_without_draws():
    """S=0 (not degraded): histograms collapse to a MAP value."""
    b = ResultsBundle.from_content(make_synthetic_content("field", n_draws=0))
    h = S.hist_for(b, S.default_state(b))
    assert h["Av"]["map"] is not None
    assert h["Av"]["top"].size == 0
    assert h["Av"]["summary"].startswith("MAP")


def test_labels():
    """TeX conversion and the header line."""
    assert S.tex_label(r"$\log M_*$") == r"$$\log M_*$$"
    assert S.tex_label("slope") == "slope"
    assert S.plain_label("Av", {"Av": "mag"}) == "Av [mag]"
    b = ResultsBundle.from_content(make_synthetic_content("degraded"))
    text = S.header_text(b)
    assert "synthetic degraded" in text and "posterior: map" in text
