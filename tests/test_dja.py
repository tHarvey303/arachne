"""Tests for the DAWN JWST Archive (DJA) cutout client.

The tests that touch the network are marked ``@pytest.mark.network`` *and*
guarded by a 5 s reachability probe, so they skip cleanly on a machine without
internet access rather than failing.  The pure name-mapping tests run always.

Set ``ARACHNE_TEST_CACHE`` to a persistent directory to reuse downloads between
runs (see the ``arachne_cache_dir`` fixture in ``conftest.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from arachne.data.dja import (
    DJA_BASE_URL,
    band_names_from_dja_filters,
    dja_filter_names,
    fetch_dja_cutout,
    load_dja_cutout,
    server_reachable,
)
from arachne.data.multires import MultiResolutionObservation

#: Coordinates verified to have deep NIRCam coverage (GOODS-S / JADES origin).
PROBE_RA, PROBE_DEC = 53.1625, -27.7914
#: Full cutout width in arcsec requested from the thumbnail server.
PROBE_SIZE = 6.0
PROBE_FILTERS = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F444W"]


@pytest.fixture(scope="session")
def dja_server():
    """Skip the calling test unless the DJA cutout server answers within 5 s.

    Probed lazily and once per session, so an offline machine pays a single
    5 s timeout and every network test skips rather than failing.

    Returns:
        The DJA base URL.
    """
    if not server_reachable(DJA_BASE_URL, timeout=5.0):
        pytest.skip(f"{DJA_BASE_URL} unreachable (offline?)")
    return DJA_BASE_URL


class TestFilterNameMapping:
    """Band-name to DJA-filter translation, which needs no network."""

    def test_band_name_to_dja_filter(self):
        """Canonical NIRCam band names gain the ``-clear`` pupil suffix."""
        assert dja_filter_names("JWST/NIRCam.F200W") == "f200w-clear"
        assert dja_filter_names("JWST/NIRCam.F115W") == "f115w-clear"

    def test_iterable_input_gives_list(self):
        """An iterable of band names maps to a list of filter strings."""
        got = dja_filter_names(PROBE_FILTERS)
        assert got == ["f115w-clear", "f200w-clear", "f444w-clear"]

    def test_already_dja_filter_is_idempotent(self):
        """A string that is already a DJA filter is passed through lower-cased."""
        assert dja_filter_names("F200W-CLEAR") == "f200w-clear"
        assert dja_filter_names("f444w-clear") == "f444w-clear"

    def test_miri_keeps_no_pupil(self):
        """MIRI filters have no pupil wheel, so no ``-clear`` is appended."""
        assert dja_filter_names("JWST/MIRI.F770W") == "f770w"

    def test_round_trip(self):
        """Filter strings map back to canonical band names."""
        filters = dja_filter_names(PROBE_FILTERS)
        assert band_names_from_dja_filters(filters) == PROBE_FILTERS
        assert band_names_from_dja_filters("f200w-clear") == "JWST/NIRCam.F200W"


class TestServerProbe:
    """The reachability probe used to gate the network tests."""

    def test_unreachable_host_is_false(self):
        """A host that cannot resolve reports False rather than raising."""
        assert server_reachable("https://not-a-real-host.invalid", timeout=2.0) is False


@pytest.mark.network
class TestFetchDJACutout:
    """Downloading and parsing a real DJA thumbnail cutout."""

    @pytest.fixture(scope="class")
    def cutout_path(self, dja_server, arachne_cache_dir):
        """Download (or reuse) a three-filter cutout at the probe position."""
        return fetch_dja_cutout(
            PROBE_RA,
            PROBE_DEC,
            PROBE_SIZE,
            PROBE_FILTERS,
            output="fits_weight",
            cache_dir=arachne_cache_dir,
        )

    def test_file_is_fits_with_sci_and_wht(self, cutout_path):
        """The product is a multi-extension FITS with SCI/WHT pairs per filter."""
        assert cutout_path.exists() and cutout_path.stat().st_size > 0
        with fits.open(cutout_path) as hdul:
            extnames = [hdu.header.get("EXTNAME") for hdu in hdul if hdu.data is not None]
            extvers = [str(hdu.header.get("EXTVER")) for hdu in hdul if hdu.data is not None]
        assert set(extnames) == {"F115W-CLEAR", "F200W-CLEAR", "F444W-CLEAR"}
        assert set(extvers) == {"SCI", "WHT"}
        assert len(extnames) == 6

    def test_bunit_is_the_dja_convention(self, cutout_path):
        """Every extension carries ``BUNIT = '10.0*nanoJansky'`` (PHOTFNU 1e-8)."""
        with fits.open(cutout_path) as hdul:
            for hdu in hdul:
                if hdu.data is None:
                    continue
                assert hdu.header["BUNIT"] == "10.0*nanoJansky"
                assert hdu.header["PHOTFNU"] == pytest.approx(1e-8, rel=1e-6)

    def test_caching_is_a_no_op(self, cutout_path, arachne_cache_dir):
        """A repeated call with the same query returns the cached file."""
        mtime = cutout_path.stat().st_mtime
        again = fetch_dja_cutout(
            PROBE_RA,
            PROBE_DEC,
            PROBE_SIZE,
            PROBE_FILTERS,
            output="fits_weight",
            cache_dir=arachne_cache_dir,
        )
        assert again == cutout_path
        assert again.stat().st_mtime == mtime

    def test_size_is_a_half_width(self, cutout_path):
        """``size_arcsec`` is the full width: 6 arcsec at 0.05"/px gives 120 px."""
        with fits.open(cutout_path) as hdul:
            shapes = {hdu.data.shape for hdu in hdul if hdu.data is not None}
        assert shapes == {(120, 120)}


@pytest.mark.network
class TestLoadDJACutout:
    """The DJA cutout loaded into a MultiResolutionObservation."""

    @pytest.fixture(scope="class")
    def obs(self, dja_server, arachne_cache_dir):
        """Load the probe cutout as a MultiResolutionObservation."""
        path = fetch_dja_cutout(
            PROBE_RA,
            PROBE_DEC,
            PROBE_SIZE,
            PROBE_FILTERS,
            output="fits_weight",
            cache_dir=arachne_cache_dir,
        )
        return load_dja_cutout(path, PROBE_RA, PROBE_DEC)

    def test_band_names_and_count(self, obs):
        """EXTNAMEs become canonical band names, in file order."""
        assert isinstance(obs, MultiResolutionObservation)
        assert obs.n_bands == 3
        assert obs.band_names == PROBE_FILTERS

    def test_shapes_and_pixel_scales(self, obs):
        """The thumbnail server resamples every filter onto one 0.05"/px grid."""
        assert obs.shapes == [(120, 120)] * 3
        for scale in obs.pixel_scales:
            assert scale == pytest.approx(0.05, abs=1e-3)

    def test_fluxes_converted_to_nJy(self, obs, arachne_cache_dir):
        """Pixels are scaled by 10 from the delivered ``10.0*nanoJansky`` values."""
        path = fetch_dja_cutout(
            PROBE_RA,
            PROBE_DEC,
            PROBE_SIZE,
            PROBE_FILTERS,
            output="fits_weight",
            cache_dir=arachne_cache_dir,
        )
        with fits.open(path) as hdul:
            raw = np.asarray(
                next(
                    hdu.data
                    for hdu in hdul
                    if hdu.data is not None
                    and hdu.header.get("EXTNAME") == "F200W-CLEAR"
                    and str(hdu.header.get("EXTVER")) == "SCI"
                ),
                dtype=np.float64,
            )
        flux = np.asarray(obs["JWST/NIRCam.F200W"].flux, dtype=np.float64)
        np.testing.assert_allclose(flux, raw * 10.0, rtol=1e-5, atol=1e-7)
        assert np.isfinite(flux).all()
        # Not the misleading ZP card: that would imply ~23 nJy per pixel value.
        assert obs.band_names == PROBE_FILTERS

    def test_variance_from_weights(self, obs):
        """WHT extensions become finite positive variances with a valid mask."""
        for band in obs:
            var = np.asarray(band.variance)
            mask = np.asarray(band.mask)
            assert mask.sum() > 0.5 * mask.size
            assert np.all(var[mask > 0] > 0)
            assert np.all(np.isfinite(var[mask > 0]))

    def test_sky_coords_span_the_requested_size(self, obs):
        """``sky_coords`` covers roughly +-size/2 arcsec about the reference."""
        half = PROBE_SIZE / 2.0
        for band in obs:
            yy, xx = band.sky_coords()
            assert yy.shape == (band.shape[0] * band.shape[1],)
            for arr in (yy, xx):
                assert arr.min() == pytest.approx(-half, abs=0.2)
                assert arr.max() == pytest.approx(half, abs=0.2)

    def test_reference_pixel_is_near_the_centre(self, obs):
        """The requested RA/Dec lands within a pixel of the cutout centre."""
        for band in obs:
            h, w = band.shape
            assert band.ref_pixel[0] == pytest.approx((h - 1) / 2.0, abs=1.5)
            assert band.ref_pixel[1] == pytest.approx((w - 1) / 2.0, abs=1.5)

    def test_east_is_positive_dx_north_is_positive_dy(self, obs):
        """North up / East left: ``affine[1, 1]`` (d dx / d col) is negative."""
        for band in obs:
            assert band.affine[0, 0] > 0.0  # increasing row -> North
            assert band.affine[1, 1] < 0.0  # increasing col -> West

    def test_collapses_to_observation_cube(self, obs):
        """All three thumbnails share a grid, so the single-grid bridge works."""
        cube = obs.to_observation_cube()
        assert cube.flux.shape == (3, 120, 120)
        assert cube.band_names == PROBE_FILTERS
        assert cube.pixel_scale == pytest.approx(0.05, abs=1e-3)
        assert cube.flux_unit == "nJy"

    def test_to_jax(self, obs):
        """``to_jax`` converts every band's arrays without changing shapes."""
        jobs = obs.to_jax()
        for band, jband in zip(obs, jobs):
            assert jband.flux.shape == band.shape
            assert jband.flux.dtype.name == "float32"
