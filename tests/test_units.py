"""Tests for flux-unit parsing and conversion to nanoJansky."""

import numpy as np
import pytest

from arachne.data.units import (
    AB_ZEROPOINT_NJY,
    ARCSEC2_IN_SR,
    FLUX_DENSITY,
    SURFACE_BRIGHTNESS,
    UNKNOWN,
    bunit_scale_to_nJy,
    flux_to_nJy,
    parse_bunit,
    variance_to_nJy2,
    weight_to_variance,
    zeropoint_scale_to_nJy,
)


class TestParseBunit:
    """Tests for parse_bunit."""

    @pytest.mark.parametrize(
        "bunit, expected",
        [
            ("nJy", 1.0),
            ("nanoJansky", 1.0),
            ("nanojansky", 1.0),
            ("10.0*nanoJansky", 10.0),
            ("1e-8*Jy", 1e-8 * 1e9),
            ("uJy", 1e3),
            ("microJansky", 1e3),
            ("µJy", 1e3),
            ("mJy", 1e6),
            ("millijansky", 1e6),
            ("Jy", 1e9),
            ("jansky", 1e9),
            ("3 Jy", 3e9),
        ],
    )
    def test_flux_density_units(self, bunit, expected):
        """Recognised per-pixel flux units give the right nJy scale."""
        scale, kind = parse_bunit(bunit)
        assert kind == FLUX_DENSITY
        assert scale == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("bunit", ["MJy/sr", "MJy / sr", "MJy sr**-1", "MJy/steradian"])
    def test_surface_brightness_units(self, bunit):
        """MJy/sr spellings parse as a surface brightness in nJy/sr."""
        scale, kind = parse_bunit(bunit)
        assert kind == SURFACE_BRIGHTNESS
        # 1 MJy = 1e6 Jy = 1e15 nJy.
        assert scale == pytest.approx(1e15, rel=1e-12)

    @pytest.mark.parametrize(
        "bunit", ["electron/s", "ELECTRONS/S", "DN/s", "counts", "", None, "banana"]
    )
    def test_unknown_units(self, bunit):
        """Unrecognised units report kind='unknown' and no scale."""
        scale, kind = parse_bunit(bunit)
        assert kind == UNKNOWN
        assert scale is None

    def test_quoted_bunit(self):
        """A BUNIT string with stray quotes still parses."""
        scale, kind = parse_bunit("'10.0*nanoJansky'")
        assert kind == FLUX_DENSITY
        assert scale == pytest.approx(10.0)

    def test_bunit_scale_needs_pixel_area_for_surface_brightness(self):
        """bunit_scale_to_nJy returns None for MJy/sr without a pixel area."""
        assert bunit_scale_to_nJy("MJy/sr") is None
        assert bunit_scale_to_nJy("MJy/sr", pixel_area_arcsec2=0.03**2) is not None


class TestZeropoint:
    """Tests for the AB zeropoint convention."""

    def test_njy_zeropoint_is_unity(self):
        """A zeropoint of 31.4 means pixel values are already nJy."""
        assert zeropoint_scale_to_nJy(AB_ZEROPOINT_NJY) == pytest.approx(1.0, rel=1e-6)

    def test_ujy_zeropoint(self):
        """A zeropoint 7.5 mag brighter than 31.4 corresponds to microJansky."""
        assert zeropoint_scale_to_nJy(AB_ZEROPOINT_NJY - 7.5) == pytest.approx(1e3, rel=1e-6)

    def test_zeropoint_takes_precedence_over_bunit(self):
        """An explicit AB zeropoint overrides BUNIT."""
        out = flux_to_nJy(np.ones(3), bunit="Jy", zeropoint_ab=AB_ZEROPOINT_NJY)
        np.testing.assert_allclose(out, 1.0, rtol=1e-6)


class TestFluxToNJy:
    """Tests for flux_to_nJy and variance_to_nJy2."""

    def test_bunit_path(self):
        """The BUNIT path scales the data by the parsed factor."""
        data = np.arange(4.0)
        np.testing.assert_allclose(flux_to_nJy(data, bunit="uJy"), data * 1e3)

    def test_surface_brightness_path(self):
        """MJy/sr converts using the pixel solid angle."""
        pixel_scale = 0.03
        out = flux_to_nJy(np.ones(1), bunit="MJy/sr", pixel_area_arcsec2=pixel_scale**2)
        expected = 1e15 * pixel_scale**2 * ARCSEC2_IN_SR
        np.testing.assert_allclose(out, expected, rtol=1e-10)
        # Sanity: 1 MJy/sr in a 0.03" pixel is about 21 nJy.
        assert 21.0 < float(out[0]) < 21.3

    def test_surface_brightness_without_pixel_area_raises(self):
        """MJy/sr without a pixel area is an error, not a silent guess."""
        with pytest.raises(ValueError, match="pixel_area_arcsec2"):
            flux_to_nJy(np.ones(1), bunit="MJy/sr")

    def test_unknown_unit_raises(self):
        """An unknown unit with no zeropoint raises ValueError."""
        with pytest.raises(ValueError, match="not a recognised flux unit"):
            flux_to_nJy(np.ones(1), bunit="electron/s")

    def test_no_information_raises(self):
        """No BUNIT and no zeropoint raises ValueError."""
        with pytest.raises(ValueError):
            flux_to_nJy(np.ones(1))

    def test_float32_input_stays_float32(self):
        """A float32 image is not silently promoted to float64."""
        data = np.ones((4, 4), dtype=np.float32)
        assert flux_to_nJy(data, bunit="nJy").dtype == np.float32

    def test_variance_scales_as_the_square(self):
        """variance_to_nJy2 applies the square of the flux scale."""
        var = np.full(3, 2.0)
        np.testing.assert_allclose(variance_to_nJy2(var, bunit="uJy"), var * 1e6)

    def test_flux_and_variance_are_consistent(self):
        """sqrt(variance) transforms like flux."""
        flux = np.array([3.0])
        var = np.array([9.0])
        f = flux_to_nJy(flux, bunit="10.0*nanoJansky")
        v = variance_to_nJy2(var, bunit="10.0*nanoJansky")
        np.testing.assert_allclose(np.sqrt(v), f, rtol=1e-10)


class TestDJAConvention:
    """Round-trip checks against the DJA header convention."""

    def test_bunit_matches_photfnu_zeropoint(self):
        """BUNIT='10.0*nanoJansky' agrees with the PHOTFNU zeropoint of 28.9.

        DJA cutouts carry PHOTFNU = 1e-8 Jy per pixel value, i.e. 10 nJy, whose
        AB zeropoint is -2.5*log10(1e-8) + 8.9 = 28.9.  BUNIT and that
        zeropoint must agree to well under 1%.
        """
        data = np.array([1.0, 2.5, -0.3])
        from_bunit = flux_to_nJy(data, bunit="10.0*nanoJansky")
        from_zp = flux_to_nJy(data, zeropoint_ab=28.9)
        np.testing.assert_allclose(from_bunit, from_zp, rtol=1e-3)

    def test_dja_zp_card_disagrees_with_bunit(self):
        """The DJA ``ZP`` card is the *original* mosaic zeropoint, not the data's.

        Regression guard for a real trap: DJA thumbnails ship
        BUNIT='10.0*nanoJansky' alongside ZP=28.0025 (F200W), which is
        -2.5*log10(OPHOTFNU) + 8.9 for the pre-rescaling mosaic.  The two differ
        by a factor of ~2.3, so loaders must prefer BUNIT.
        """
        ratio = zeropoint_scale_to_nJy(28.0025) / 10.0
        assert ratio == pytest.approx(2.2856, rel=1e-3)
        assert abs(ratio - 1.0) > 0.01

    def test_photfnu_zeropoint_identity(self):
        """-2.5*log10(PHOTFNU) + 8.9 reproduces the nJy-per-DN factor."""
        photfnu_jy = 1e-8
        zp = -2.5 * np.log10(photfnu_jy) + 8.9
        assert zeropoint_scale_to_nJy(zp) == pytest.approx(photfnu_jy * 1e9, rel=1e-6)


class TestWeightToVariance:
    """Tests for weight_to_variance."""

    def test_positive_weights_invert(self):
        """Positive weights become 1/weight variances with mask 1."""
        wht = np.array([[4.0, 0.25]])
        var, mask = weight_to_variance(wht)
        np.testing.assert_allclose(var, [[0.25, 4.0]])
        np.testing.assert_allclose(mask, [[1.0, 1.0]])

    def test_zero_weight_is_infinite_variance_and_masked(self):
        """Zero weight gives infinite variance and mask 0."""
        var, mask = weight_to_variance(np.array([0.0, 1.0]))
        assert np.isinf(var[0])
        assert mask[0] == 0.0
        assert mask[1] == 1.0

    def test_negative_and_nan_weights_are_masked(self):
        """Negative and non-finite weights are treated as invalid."""
        var, mask = weight_to_variance(np.array([-1.0, np.nan, np.inf, 2.0]))
        np.testing.assert_allclose(mask, [0.0, 0.0, 0.0, 1.0])
        assert np.isinf(var[0]) and np.isinf(var[1])

    def test_floor(self):
        """Weights at or below the floor are masked out."""
        var, mask = weight_to_variance(np.array([0.5, 5.0]), floor=1.0)
        np.testing.assert_allclose(mask, [0.0, 1.0])
        assert np.isinf(var[0])

    def test_dtypes(self):
        """Outputs are float32."""
        var, mask = weight_to_variance(np.ones((3, 3)))
        assert var.dtype == np.float32
        assert mask.dtype == np.float32
