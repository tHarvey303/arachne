"""Tests for ``arachne.results.regions`` and ``arachne.results.display``."""

from __future__ import annotations

import numpy as np
import pytest

from arachne.results import (
    Region,
    RegionSED,
    ResultsBundle,
    component_region_seds,
    component_seds,
    detection_region,
    make_synthetic_content,
    percentile_limits,
    pixel_sed,
    region_param_summary,
    region_sed,
    rgb_composite,
    stretch,
    total_sed,
)


@pytest.fixture(scope="module")
def field():
    """In-memory field-flavour bundle."""
    return ResultsBundle.from_content(make_synthetic_content("field"))


@pytest.fixture(scope="module")
def additive():
    """In-memory additive-flavour bundle."""
    return ResultsBundle.from_content(make_synthetic_content("additive"))


@pytest.fixture(scope="module")
def degraded():
    """In-memory degraded-flavour bundle."""
    return ResultsBundle.from_content(make_synthetic_content("degraded"))


def test_full_mask_equals_manual_sums(field):
    """A full-frame region reproduces manual masked sums."""
    sed = region_sed(field, np.ones((16, 16), dtype=bool), label="all")
    valid = field.mask > 0
    np.testing.assert_allclose(sed.obs_flux, (field.flux * valid).sum(axis=(1, 2)), rtol=1e-6)
    np.testing.assert_array_equal(sed.n_pix, [255, 256, 256])
    draws = (field.model_draws() * valid).sum(axis=(2, 3))
    np.testing.assert_allclose(sed.model_draws, draws, rtol=1e-6)
    np.testing.assert_allclose(
        sed.model_map, (field.cube("model", "map") * valid).sum(axis=(1, 2)), rtol=1e-6
    )
    # intrinsic ignores the band mask
    np.testing.assert_allclose(
        sed.intrinsic_draws, field.model_draws("intrinsic").sum(axis=(2, 3)), rtol=1e-6
    )
    np.testing.assert_allclose(
        sed.truth_convolved, (field.truth.convolved * valid).sum(axis=(1, 2)), rtol=1e-6
    )
    assert sed.label == "all"


def test_masked_pixel_excluded_per_band(field):
    """A pixel masked in one band is excluded in that band only."""
    sed = pixel_sed(field, 0, 0)
    assert np.isnan(sed.obs_flux[0]) and np.isnan(sed.obs_err[0]) and np.isnan(sed.model_map[0])
    np.testing.assert_allclose(sed.obs_flux[1:], field.flux[1:, 0, 0], rtol=1e-6)
    np.testing.assert_array_equal(sed.n_pix, [0, 1, 1])
    assert np.isfinite(sed.intrinsic_map).all()


def test_variance_propagation(field):
    """Aperture errors add variances, error_scale, noise scale and floor."""
    reg = field.regions["core"]
    sed = region_sed(field, reg)
    expected = np.sqrt((field.variance * reg.mask).sum(axis=(1, 2)))
    np.testing.assert_allclose(sed.obs_err, expected, rtol=1e-6)
    sed2 = region_sed(field, reg, error_scale=np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(sed2.obs_err, expected * [1, 2, 3], rtol=1e-6)
    floor = region_sed(field, reg, model_floor=True)
    p50 = np.median(sed.model_draws, axis=0)
    np.testing.assert_allclose(floor.obs_err, np.sqrt(expected**2 + (0.05 * p50) ** 2), rtol=1e-6)

    content = make_synthetic_content("field")
    content.nuisance = {"log_noise_scale": np.log(np.full((20, 3), 2.0))}
    scaled = region_sed(ResultsBundle.from_content(content), reg)
    np.testing.assert_allclose(scaled.obs_err, 2 * expected, rtol=1e-5)


def test_region_membership():
    """Pixel/box/circle/polygon rasterise by pixel-centre membership."""
    shape = (10, 12)
    px = Region.pixel(shape, 3, 4)
    assert px.n_pix == 1 and px.mask[3, 4] and px.geometry == {"y": 3, "x": 4}
    box = Region.box(shape, 2, 4, 5, 8)
    assert box.n_pix == 3 * 4 and box.mask[2, 5] and box.mask[4, 8] and not box.mask[5, 8]
    circ = Region.circle(shape, 5, 5, 1.0)
    assert circ.n_pix == 5 and circ.mask[4, 5] and not circ.mask[4, 4]
    poly = Region.polygon(shape, [[0.5, 0.5], [0.5, 3.5], [3.5, 3.5], [3.5, 0.5]])
    expected = np.zeros(shape, dtype=bool)
    expected[1:4, 1:4] = True
    np.testing.assert_array_equal(poly.mask, expected)
    tri = Region.polygon(shape, [[-0.5, -0.5], [-0.5, 9.5], [9.5, -0.5]])
    assert tri.mask[0, 0] and tri.mask[8, 0] and not tri.mask[8, 8]
    for reg in (px, box, circ, poly):
        again = Region.from_geometry(shape, reg.kind, reg.geometry)
        np.testing.assert_array_equal(again.mask, reg.mask)
    with pytest.raises(ValueError):
        Region.from_mask(np.ones(5))


def test_region_shape_mismatch(field):
    """A mask on the wrong grid is rejected."""
    with pytest.raises(ValueError, match="shape"):
        region_sed(field, np.ones((4, 4), dtype=bool))


def test_region_sed_percentiles_chi_frac():
    """RegionSED percentiles, chi and fractional residuals."""
    draws = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    sed = RegionSED(
        label="t",
        band_names=["a", "b"],
        pivot_um=np.array([1.0, 2.0]),
        obs_flux=np.array([4.0, 30.0]),
        obs_err=np.array([1.0, 3.0]),
        n_pix=np.array([1, 1]),
        model_draws=draws,
        model_map=np.array([2.5, 25.0]),
        intrinsic_draws=None,
        intrinsic_map=np.array([7.0, 8.0]),
    )
    q = sed.model_percentiles()
    np.testing.assert_allclose(q, np.percentile(draws, [16, 50, 84], axis=0))
    half = 0.5 * (q[2] - q[0])
    np.testing.assert_allclose(sed.chi(), ([4.0, 30.0] - q[1]) / np.sqrt([1.0, 9.0] + half**2))
    np.testing.assert_allclose(sed.chi(False), ([4.0, 30.0] - q[1]) / np.array([1.0, 3.0]))
    np.testing.assert_allclose(sed.frac_residual(), ([4.0, 30.0] - q[1]) / q[1])
    np.testing.assert_allclose(sed.model_percentiles(frame="intrinsic"), [[7, 8]] * 3)


def test_degraded_region_sed_uses_map(degraded):
    """Without draws the MAP model stands in for the percentiles."""
    sed = total_sed(degraded)
    assert sed.model_draws is None and sed.intrinsic_map is None
    np.testing.assert_allclose(sed.model_percentiles(), np.repeat(sed.model_map[None], 3, 0))
    chi = sed.chi()
    np.testing.assert_allclose(chi, (sed.obs_flux - sed.model_map) / sed.obs_err)


def test_total_sed_prefers_segment(field):
    """total_sed uses 'segment' before 'all'."""
    assert total_sed(field).label == "total (all)"
    content = make_synthetic_content("field")
    seg = np.zeros((16, 16), dtype=bool)
    seg[4:12, 4:12] = True
    content.regions["segment"] = Region.from_mask(seg, name="segment")
    sed = total_sed(ResultsBundle.from_content(content))
    assert sed.label == "total (segment)" and sed.n_pix[1] == 64


def test_region_param_summary(field):
    """Region mass is a log-sum, other maps are mass-weighted means."""
    reg = field.regions["core"]
    out = region_param_summary(field, reg)
    lm = field.param_draws("log_mass")[:, reg.mask].astype(float)
    av = field.param_draws("Av")[:, reg.mask].astype(float)
    np.testing.assert_allclose(out["log_mass"], np.log10((10.0**lm).sum(axis=1)), rtol=1e-10)
    wav = ((10.0**lm) * av).sum(axis=1) / (10.0**lm).sum(axis=1)
    np.testing.assert_allclose(out["Av"], wav, rtol=1e-10)
    assert int(out["n_pix"]) == reg.n_pix
    one = region_param_summary(field, Region.pixel((16, 16), 5, 6))
    np.testing.assert_allclose(one["Av"], field.param_draws("Av")[:, 5, 6], rtol=1e-6)


def test_region_param_summary_degraded_and_additive(degraded, additive):
    """Summaries fall back to MAP maps; no maps gives n_pix only."""
    out = region_param_summary(degraded, degraded.regions["all"])
    assert out["log_mass"].shape == (1,)
    lm = degraded.param_map("log_mass", "map").astype(float)
    np.testing.assert_allclose(out["log_mass"], np.log10((10.0**lm).sum()), rtol=1e-10)
    assert set(region_param_summary(additive, additive.regions["all"])) == {"n_pix"}


def test_component_seds(additive, field):
    """Component SEDs sum to the total and carry the truth."""
    seds = component_seds(additive)
    assert len(seds) == 3 and seds[-1].label == "total"
    assert seds[0].obs_flux is None and seds[0].intrinsic_draws.shape == (20, 3)
    np.testing.assert_allclose(
        seds[0].intrinsic_draws + seds[1].intrinsic_draws, seds[2].intrinsic_draws
    )
    np.testing.assert_allclose(seds[0].truth_intrinsic, additive.truth.component_seds[0])
    assert component_seds(field) == []


def test_component_region_seds_sum_to_model(additive):
    """Component region SEDs add up to the model SED (no sky)."""
    reg = additive.regions["core"]
    parts = component_region_seds(additive, reg)
    total = region_sed(additive, reg)
    np.testing.assert_allclose(sum(p.model_map for p in parts), total.model_map, rtol=1e-5)
    np.testing.assert_allclose(sum(p.model_draws for p in parts), total.model_draws, rtol=1e-5)
    np.testing.assert_allclose(sum(p.intrinsic_map for p in parts), total.intrinsic_map, rtol=1e-5)


def test_detection_region(field):
    """The detection finds the synthetic galaxy and nothing in pure noise."""
    det = detection_region(field.flux, field.variance, field.mask)
    assert det.dtype == bool and det[8, 8] and det.sum() > 20
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(3, 32, 32))
    assert not detection_region(noise, np.ones_like(noise), np.ones_like(noise), nsigma=6).any()


def test_display_helpers():
    """Percentile limits, stretches and the RGB composite."""
    img = np.arange(100.0).reshape(10, 10)
    img[0, 0] = np.nan
    lo, hi = percentile_limits(img, (0, 100))
    assert (lo, hi) == (1.0, 99.0)
    assert percentile_limits(img - 50, (0, 100), symmetric=True) == (-49.0, 49.0)
    assert percentile_limits(np.full((3, 3), np.nan)) == (0.0, 1.0)
    for kind in ("linear", "asinh", "log", "sqrt"):
        s = stretch(img, kind, 0.0, 99.0)
        assert np.isnan(s[0, 0]) and np.nanmin(s) >= 0 and np.nanmax(s) == pytest.approx(1.0)
    np.testing.assert_allclose(stretch(img, "linear", 0, 99)[0, 1], 1 / 99)
    with pytest.raises(ValueError):
        stretch(img, "bogus")
    rgb = rgb_composite(img, img * 0.5, img * 0.1)
    assert rgb.shape == (10, 10, 3) and np.isfinite(rgb).all()
    assert rgb.min() >= 0 and rgb.max() <= 1 and (rgb[..., 0] >= rgb[..., 2]).all()
