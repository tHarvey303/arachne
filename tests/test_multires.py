"""Tests for the multi-resolution observation containers and PSF resampling."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from conftest import make_tan_header, write_image_fits

from arachne.data.multires import (
    BandImage,
    MultiResolutionObservation,
    canonical_band_name,
    pixel_scale_from_affine,
    tangent_plane_affine,
)
from arachne.data.psf import PSFModel

REF_RA, REF_DEC = 53.1625, -27.7914


def _gaussian_kernel(size: int, sigma_px: float) -> np.ndarray:
    """Build a normalised 2-D Gaussian kernel.

    Args:
        size: Odd kernel size in pixels.
        sigma_px: Standard deviation in pixels.

    Returns:
        (size, size) kernel summing to 1.
    """
    half = size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    k = np.exp(-(x**2 + y**2) / (2.0 * sigma_px**2))
    return k / k.sum()


def _kernel_sigma(kernel: np.ndarray) -> float:
    """Measure the second-moment sigma of a centred kernel, in pixels.

    Args:
        kernel: 2-D normalised kernel.

    Returns:
        Geometric-mean sigma in pixels.
    """
    h, w = kernel.shape
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    total = kernel.sum()
    var_y = (kernel * (y - cy) ** 2).sum() / total
    var_x = (kernel * (x - cx) ** 2).sum() / total
    return float(np.sqrt(np.sqrt(var_y * var_x)))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestTangentPlaneAffine:
    """Tests for the pixel -> tangent-plane affine map."""

    def test_unrotated_orientation(self):
        """North-up/East-left images give dy = +row, dx = -col."""
        header = make_tan_header((64, 64), 0.05, REF_RA, REF_DEC)
        affine, ref_pixel = tangent_plane_affine(WCS(header), REF_RA, REF_DEC)
        assert affine[0, 0] == pytest.approx(0.05, rel=1e-4)  # d dy / d row
        assert affine[1, 1] == pytest.approx(-0.05, rel=1e-4)  # d dx / d col
        assert abs(affine[0, 1]) < 1e-6
        assert abs(affine[1, 0]) < 1e-6
        np.testing.assert_allclose(ref_pixel, (31.5, 31.5), atol=1e-6)

    def test_pixel_scale_from_affine(self):
        """The geometric-mean pixel scale is recovered under rotation."""
        header = make_tan_header((64, 64), 0.031, REF_RA, REF_DEC, rotation_deg=37.0)
        affine, _ = tangent_plane_affine(WCS(header), REF_RA, REF_DEC)
        assert pixel_scale_from_affine(affine) == pytest.approx(0.031, rel=1e-4)

    def test_rotation_enters_the_off_diagonals(self):
        """A rotated image has non-zero off-diagonal affine terms."""
        header = make_tan_header((64, 64), 0.05, REF_RA, REF_DEC, rotation_deg=30.0)
        affine, _ = tangent_plane_affine(WCS(header), REF_RA, REF_DEC)
        assert abs(affine[0, 1]) > 0.01
        assert abs(affine[1, 0]) > 0.01

    def test_reference_pixel_is_fractional(self):
        """A reference position offset from CRVAL lands on a fractional pixel."""
        header = make_tan_header((64, 64), 0.1, REF_RA, REF_DEC)
        # Move 0.25 pixel north of CRVAL.
        ref = SkyCoord(REF_RA * u.deg, REF_DEC * u.deg).directional_offset_by(
            0 * u.deg, 0.025 * u.arcsec
        )
        _, ref_pixel = tangent_plane_affine(WCS(header), ref.ra.deg, ref.dec.deg)
        assert ref_pixel[0] == pytest.approx(31.75, abs=1e-3)


class TestSkyCoords:
    """sky_coords must reproduce astropy's own spherical offsets."""

    @pytest.mark.parametrize("rotation", [0.0, 30.0, -73.5])
    @pytest.mark.parametrize("pixel_scale", [0.02, 0.06])
    def test_matches_astropy_over_a_10_arcsec_cutout(self, tmp_path, rotation, pixel_scale):
        """sky_coords agrees with pixel_to_world + spherical_offsets_to to <1e-3 arcsec."""
        npix = int(np.ceil(14.0 / pixel_scale))
        header = make_tan_header(
            (npix, npix),
            pixel_scale,
            REF_RA + 0.0004,
            REF_DEC - 0.0003,
            rotation_deg=rotation,
        )
        path = write_image_fits(tmp_path / "sci.fits", np.zeros((npix, npix)), header)

        obs = MultiResolutionObservation.from_fits(
            flux_paths={"B": path},
            ref_ra=REF_RA,
            ref_dec=REF_DEC,
            size_arcsec=10.0,
        )
        band = obs["B"]
        yy, xx = band.sky_coords()

        h, w = band.shape
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = band.wcs.pixel_to_world(cols.ravel(), rows.ravel())
        ref = SkyCoord(REF_RA * u.deg, REF_DEC * u.deg)
        dlon, dlat = ref.spherical_offsets_to(coords)

        np.testing.assert_allclose(yy, dlat.to_value(u.arcsec), atol=1e-3)
        np.testing.assert_allclose(xx, dlon.to_value(u.arcsec), atol=1e-3)

    def test_ranges_span_the_cutout(self):
        """Offsets span roughly +/- half the cutout size."""
        band = _simple_band(size_px=101, pixel_scale=0.1)
        yy, xx = band.sky_coords()
        assert yy.min() == pytest.approx(-5.0, abs=0.15)
        assert yy.max() == pytest.approx(5.0, abs=0.15)
        assert xx.min() == pytest.approx(-5.0, abs=0.15)
        assert xx.max() == pytest.approx(5.0, abs=0.15)

    def test_east_is_positive_dx(self):
        """A pixel to the East of the reference has positive dx."""
        band = _simple_band(size_px=21, pixel_scale=0.1)
        yy, xx = band.sky_coords()
        h, w = band.shape
        east_pixel = np.ravel_multi_index((h // 2, 0), (h, w))  # column 0 = East
        assert xx[east_pixel] > 0
        assert abs(yy[east_pixel]) < 1e-6


def _simple_band(size_px: int = 21, pixel_scale: float = 0.1) -> BandImage:
    """Build a BandImage directly from a synthetic header.

    Args:
        size_px: Image size in pixels.
        pixel_scale: Pixel scale in arcsec/pixel.

    Returns:
        A BandImage centred on the reference position.
    """
    header = make_tan_header((size_px, size_px), pixel_scale, REF_RA, REF_DEC)
    wcs = WCS(header)
    affine, ref_pixel = tangent_plane_affine(wcs, REF_RA, REF_DEC)
    data = np.zeros((size_px, size_px), dtype=np.float32)
    return BandImage(
        band_name="TEST",
        flux=data,
        variance=np.ones_like(data),
        mask=np.ones_like(data),
        pixel_scale=pixel_scale_from_affine(affine),
        affine=affine,
        ref_pixel=ref_pixel,
        wcs=wcs,
    )


# ---------------------------------------------------------------------------
# BandImage
# ---------------------------------------------------------------------------


class TestBandImage:
    """Validation and conversion on BandImage."""

    def test_shape_mismatch_raises(self):
        """Mismatched variance shape raises ValueError."""
        with pytest.raises(ValueError, match="variance"):
            BandImage(
                band_name="B",
                flux=np.zeros((4, 4)),
                variance=np.zeros((2, 2)),
                mask=np.zeros((4, 4)),
                pixel_scale=0.03,
                affine=np.eye(2),
                ref_pixel=(1.5, 1.5),
            )

    def test_bad_affine_raises(self):
        """A non-2x2 affine raises ValueError."""
        with pytest.raises(ValueError, match="affine"):
            BandImage(
                band_name="B",
                flux=np.zeros((4, 4)),
                variance=np.zeros((4, 4)),
                mask=np.zeros((4, 4)),
                pixel_scale=0.03,
                affine=np.eye(3),
                ref_pixel=(1.5, 1.5),
            )

    def test_to_jax(self):
        """to_jax converts arrays and preserves geometry."""
        import jax.numpy as jnp

        band = _simple_band()
        band = BandImage(**{**band.__dict__, "psf": _gaussian_kernel(9, 1.5)})
        jband = band.to_jax()
        assert isinstance(jband.flux, jnp.ndarray)
        assert isinstance(jband.psf, jnp.ndarray)
        assert jband.pixel_scale == band.pixel_scale
        np.testing.assert_allclose(jband.affine, band.affine)


# ---------------------------------------------------------------------------
# from_fits: the genuinely multi-resolution path
# ---------------------------------------------------------------------------


class TestFromFitsMultiResolution:
    """Two hand-written bands at different pixel scales and rotations."""

    def test_bands_keep_their_own_grids(self, multires_fits_pair):
        """Each band keeps its native pixel scale and shape."""
        spec = multires_fits_pair
        obs = MultiResolutionObservation.from_fits(
            flux_paths=spec["flux_paths"],
            ref_ra=spec["ref_ra"],
            ref_dec=spec["ref_dec"],
            size_arcsec=1.0,
            weight_paths=spec["weight_paths"],
        )
        assert obs.n_bands == 2
        assert obs.band_names == ["BAND_A", "BAND_B"]
        assert obs["BAND_A"].pixel_scale == pytest.approx(0.02, rel=1e-3)
        assert obs["BAND_B"].pixel_scale == pytest.approx(0.04, rel=1e-3)
        assert obs["BAND_A"].shape == (50, 50)
        assert obs["BAND_B"].shape == (25, 25)

    def test_units_converted_per_band(self, multires_fits_pair):
        """Bands in different units both end up in nJy."""
        spec = multires_fits_pair
        obs = MultiResolutionObservation.from_fits(
            flux_paths=spec["flux_paths"],
            ref_ra=spec["ref_ra"],
            ref_dec=spec["ref_dec"],
            size_arcsec=1.0,
            weight_paths=spec["weight_paths"],
        )
        raw_a = fits.getdata(spec["flux_paths"]["BAND_A"])
        raw_b = fits.getdata(spec["flux_paths"]["BAND_B"])
        # BAND_A is 10*nJy, BAND_B is uJy = 1e3 nJy.
        assert np.nanmax(np.abs(obs["BAND_A"].flux)) == pytest.approx(
            10.0 * np.nanmax(np.abs(raw_a[7:57, 7:57])), rel=1e-4
        )
        assert np.nanmax(np.abs(obs["BAND_B"].flux)) == pytest.approx(
            1e3 * np.nanmax(np.abs(raw_b[3:28, 3:28])), rel=1e-4
        )

    def test_weights_become_variance_in_njy2(self, multires_fits_pair):
        """Inverse-variance weights are inverted and scaled by scale^2."""
        spec = multires_fits_pair
        obs = MultiResolutionObservation.from_fits(
            flux_paths=spec["flux_paths"],
            ref_ra=spec["ref_ra"],
            ref_dec=spec["ref_dec"],
            size_arcsec=1.0,
            weight_paths=spec["weight_paths"],
        )
        # weight 4 -> variance 0.25 (pixel units) -> * 10^2 for BAND_A.
        np.testing.assert_allclose(np.asarray(obs["BAND_A"].variance), 25.0, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(obs["BAND_B"].variance), 0.25 * 1e6, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(obs["BAND_A"].mask), 1.0)

    def test_sky_coords_agree_between_bands(self, multires_fits_pair):
        """Both bands' pixel grids describe the same patch of sky."""
        spec = multires_fits_pair
        obs = MultiResolutionObservation.from_fits(
            flux_paths=spec["flux_paths"],
            ref_ra=spec["ref_ra"],
            ref_dec=spec["ref_dec"],
            size_arcsec=1.0,
            weight_paths=spec["weight_paths"],
        )
        for band in obs:
            yy, xx = band.sky_coords()
            # A 1 arcsec cutout: the corner radius is 0.5 * sqrt(2), whatever the
            # rotation, and the two bands must cover the same patch of sky.
            radius = np.hypot(yy, xx).max()
            assert radius == pytest.approx(0.5 * np.sqrt(2.0), rel=0.05)

    def test_psf_resampled_to_each_band_grid(self, multires_fits_pair):
        """A single PSF kernel is resampled onto each band's own pixel scale."""
        spec = multires_fits_pair
        kernel = _gaussian_kernel(41, 4.0)  # sigma = 4 px at 0.01 arcsec/px
        obs = MultiResolutionObservation.from_fits(
            flux_paths=spec["flux_paths"],
            ref_ra=spec["ref_ra"],
            ref_dec=spec["ref_dec"],
            size_arcsec=1.0,
            psfs={"BAND_A": (kernel, 0.01), "BAND_B": (kernel, 0.01)},
        )
        # 0.04 arcsec/px sigma should be half the 0.02 arcsec/px sigma.
        sigma_a = _kernel_sigma(np.asarray(obs["BAND_A"].psf))
        sigma_b = _kernel_sigma(np.asarray(obs["BAND_B"].psf))
        assert sigma_a == pytest.approx(2.0, rel=0.02)
        assert sigma_b == pytest.approx(1.0, rel=0.05)
        assert np.asarray(obs["BAND_A"].psf).sum() == pytest.approx(1.0, rel=1e-5)

    def test_explicit_zeropoint_overrides_bunit(self, multires_fits_pair):
        """A per-band zeropoint takes precedence over the header BUNIT."""
        spec = multires_fits_pair
        obs = MultiResolutionObservation.from_fits(
            flux_paths={"BAND_A": spec["flux_paths"]["BAND_A"]},
            ref_ra=spec["ref_ra"],
            ref_dec=spec["ref_dec"],
            size_arcsec=1.0,
            zeropoints={"BAND_A": 31.4},
        )
        raw = fits.getdata(spec["flux_paths"]["BAND_A"])[7:57, 7:57]
        np.testing.assert_allclose(np.asarray(obs["BAND_A"].flux), raw, rtol=1e-4)

    def test_missing_wcs_raises(self, tmp_path):
        """A band with no celestial WCS is rejected."""
        path = write_image_fits(tmp_path / "nowcs.fits", np.zeros((16, 16)))
        with pytest.raises(ValueError, match="celestial WCS"):
            MultiResolutionObservation.from_fits(
                flux_paths={"B": path}, ref_ra=REF_RA, ref_dec=REF_DEC, size_arcsec=1.0
            )

    def test_mjy_per_sr_band(self, tmp_path):
        """A band in MJy/sr converts using its own pixel area."""
        scale = 0.03
        header = make_tan_header((64, 64), scale, REF_RA, REF_DEC, bunit="MJy/sr")
        path = write_image_fits(tmp_path / "sb.fits", np.ones((64, 64)), header)
        obs = MultiResolutionObservation.from_fits(
            flux_paths={"B": path}, ref_ra=REF_RA, ref_dec=REF_DEC, size_arcsec=1.0
        )
        arcsec2_sr = (np.pi / (180 * 3600)) ** 2
        expected = 1e15 * scale**2 * arcsec2_sr
        np.testing.assert_allclose(np.asarray(obs["B"].flux), expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# Accessors and bridging
# ---------------------------------------------------------------------------


class TestAccessors:
    """Container protocol on MultiResolutionObservation."""

    def test_getitem_by_index_and_name(self, multires_fits_pair):
        """__getitem__ accepts an index or a band name."""
        spec = multires_fits_pair
        obs = MultiResolutionObservation.from_fits(
            flux_paths=spec["flux_paths"],
            ref_ra=spec["ref_ra"],
            ref_dec=spec["ref_dec"],
            size_arcsec=1.0,
        )
        assert obs[0] is obs["BAND_A"]
        assert obs.n_bands == len(obs) == 2
        assert obs.pixel_scales[1] == pytest.approx(0.04, rel=1e-3)
        with pytest.raises(KeyError):
            obs["NOPE"]

    def test_to_jax(self, multires_fits_pair):
        """to_jax converts every band."""
        import jax.numpy as jnp

        spec = multires_fits_pair
        obs = MultiResolutionObservation.from_fits(
            flux_paths=spec["flux_paths"],
            ref_ra=spec["ref_ra"],
            ref_dec=spec["ref_dec"],
            size_arcsec=1.0,
        ).to_jax()
        assert all(isinstance(b.flux, jnp.ndarray) for b in obs)

    def test_duplicate_band_names_raise(self):
        """Two bands with the same name are rejected."""
        band = _simple_band()
        with pytest.raises(ValueError, match="Duplicate"):
            MultiResolutionObservation(bands=[band, band], ref_ra=REF_RA, ref_dec=REF_DEC)

    def test_no_bands_raises(self):
        """An empty band list is rejected."""
        with pytest.raises(ValueError, match="at least one band"):
            MultiResolutionObservation(bands=[], ref_ra=REF_RA, ref_dec=REF_DEC)


class TestToObservationCube:
    """Bridging back to the single-grid ObservationCube."""

    def test_common_grid_succeeds(self, tmp_path):
        """Bands on one grid collapse to an ObservationCube."""
        header = make_tan_header((64, 64), 0.03, REF_RA, REF_DEC, bunit="nJy")
        paths = {}
        for i, band in enumerate(["B1", "B2"]):
            paths[band] = write_image_fits(
                tmp_path / f"{band}.fits", np.full((64, 64), float(i + 1)), header
            )
        obs = MultiResolutionObservation.from_fits(
            flux_paths=paths, ref_ra=REF_RA, ref_dec=REF_DEC, size_arcsec=1.2
        )
        cube = obs.to_observation_cube()
        assert cube.flux.shape == (2, 40, 40)
        assert cube.band_names == ["B1", "B2"]
        assert cube.flux_unit == "nJy"
        assert cube.pixel_scale == pytest.approx(0.03, rel=1e-3)
        np.testing.assert_allclose(np.asarray(cube.flux[1]), 2.0, rtol=1e-5)

    def test_different_grids_raise(self, multires_fits_pair):
        """Bands on different grids cannot be collapsed."""
        spec = multires_fits_pair
        obs = MultiResolutionObservation.from_fits(
            flux_paths=spec["flux_paths"],
            ref_ra=spec["ref_ra"],
            ref_dec=spec["ref_dec"],
            size_arcsec=1.0,
        )
        with pytest.raises(ValueError, match="not on a common grid"):
            obs.to_observation_cube()

    def test_same_shape_different_rotation_raises(self, tmp_path):
        """Equal shapes but different orientations are still not a common grid."""
        paths = {}
        for band, rot in [("B1", 0.0), ("B2", 15.0)]:
            header = make_tan_header((64, 64), 0.03, REF_RA, REF_DEC, rotation_deg=rot, bunit="nJy")
            paths[band] = write_image_fits(tmp_path / f"{band}.fits", np.zeros((64, 64)), header)
        obs = MultiResolutionObservation.from_fits(
            flux_paths=paths, ref_ra=REF_RA, ref_dec=REF_DEC, size_arcsec=1.2
        )
        with pytest.raises(ValueError, match="affine matrices differ"):
            obs.to_observation_cube()


# ---------------------------------------------------------------------------
# DJA multi-extension parsing (synthetic file, no network)
# ---------------------------------------------------------------------------


def _write_synthetic_dja(path, filters, pixel_scales, npix=64):
    """Write a DJA-style multi-extension FITS with SCI/WHT pairs.

    Args:
        path: Output path.
        filters: List of EXTNAME filter strings.
        pixel_scales: Pixel scale per filter, in arcsec/pixel.
        npix: Image size per extension.

    Returns:
        The path written.
    """
    hdus = []
    for i, (filt, scale) in enumerate(zip(filters, pixel_scales)):
        header = make_tan_header((npix, npix), scale, REF_RA, REF_DEC, bunit="10.0*nanoJansky")
        header["FILTER"] = filt
        header["ZP"] = 28.0025  # the misleading DJA card
        header["PHOTFNU"] = 1e-8
        planes = (
            ("SCI", np.full((npix, npix), float(i + 1))),
            ("WHT", np.full((npix, npix), 4.0)),
        )
        for kind, data in planes:
            hdr = header.copy()
            hdr["EXTNAME"] = filt
            hdr["EXTVER"] = kind
            cls = fits.PrimaryHDU if not hdus else fits.ImageHDU
            hdus.append(cls(data.astype(np.float32), header=hdr))
    fits.HDUList(hdus).writeto(str(path), overwrite=True)
    return path


class TestFromDJAFits:
    """Parsing of the DJA multi-extension layout."""

    def test_band_names_and_units(self, tmp_path):
        """EXTNAMEs map to canonical band names and BUNIT wins over ZP."""
        path = _write_synthetic_dja(
            tmp_path / "dja.fits", ["F115W-CLEAR", "F444W-CLEAR"], [0.02, 0.04]
        )
        obs = MultiResolutionObservation.from_dja_fits(path, REF_RA, REF_DEC)
        assert obs.band_names == ["JWST/NIRCam.F115W", "JWST/NIRCam.F444W"]
        # SCI planes hold 1 and 2; BUNIT is 10 nJy per pixel value.
        np.testing.assert_allclose(np.asarray(obs[0].flux), 10.0, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(obs[1].flux), 20.0, rtol=1e-5)

    def test_per_band_pixel_scales(self, tmp_path):
        """Extensions at different pixel scales stay at those scales."""
        path = _write_synthetic_dja(
            tmp_path / "dja.fits", ["F115W-CLEAR", "F444W-CLEAR"], [0.02, 0.04]
        )
        obs = MultiResolutionObservation.from_dja_fits(path, REF_RA, REF_DEC)
        assert obs.pixel_scales[0] == pytest.approx(0.02, rel=1e-3)
        assert obs.pixel_scales[1] == pytest.approx(0.04, rel=1e-3)

    def test_weights_to_variance(self, tmp_path):
        """WHT extensions become variance in nJy^2 with a validity mask."""
        path = _write_synthetic_dja(tmp_path / "dja.fits", ["F200W-CLEAR"], [0.05])
        obs = MultiResolutionObservation.from_dja_fits(path, REF_RA, REF_DEC)
        np.testing.assert_allclose(np.asarray(obs[0].variance), 0.25 * 100.0, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(obs[0].mask), 1.0)

    def test_band_name_map_override(self, tmp_path):
        """An explicit EXTNAME -> band mapping is honoured."""
        path = _write_synthetic_dja(tmp_path / "dja.fits", ["F200W-CLEAR"], [0.05])
        obs = MultiResolutionObservation.from_dja_fits(
            path, REF_RA, REF_DEC, band_name_map={"F200W-CLEAR": "custom"}
        )
        assert obs.band_names == ["custom"]

    def test_size_arcsec_trims(self, tmp_path):
        """size_arcsec trims the delivered extensions."""
        path = _write_synthetic_dja(tmp_path / "dja.fits", ["F200W-CLEAR"], [0.05], npix=120)
        obs = MultiResolutionObservation.from_dja_fits(path, REF_RA, REF_DEC, size_arcsec=2.0)
        assert obs[0].shape == (40, 40)


class TestCanonicalBandName:
    """Filter-name canonicalisation."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("F200W-CLEAR", "JWST/NIRCam.F200W"),
            ("f115w-clear", "JWST/NIRCam.F115W"),
            ("F444W", "JWST/NIRCam.F444W"),
            ("F335M-CLEAR", "JWST/NIRCam.F335M"),
            ("F770W", "JWST/MIRI.F770W"),
            ("JWST/NIRCam.F090W", "JWST/NIRCam.F090W"),
        ],
    )
    def test_mapping(self, raw, expected):
        """Known filter spellings map to canonical band names."""
        assert canonical_band_name(raw) == expected


# ---------------------------------------------------------------------------
# PSF resampling
# ---------------------------------------------------------------------------


class TestPSFResample:
    """Flux-conserving PSF resampling between pixel scales."""

    def test_coarsening_halves_the_pixel_sigma(self):
        """Resampling to a 2x coarser grid halves the sigma in pixels."""
        kernel = _gaussian_kernel(41, 4.0)
        coarse = PSFModel.resample(kernel, 0.02, 0.04)
        assert coarse.sum() == pytest.approx(1.0, rel=1e-6)
        assert _kernel_sigma(coarse) == pytest.approx(2.0, rel=0.02)
        assert coarse.shape[0] % 2 == 1

    def test_refining_doubles_the_pixel_sigma(self):
        """Resampling to a 2x finer grid doubles the sigma in pixels."""
        kernel = _gaussian_kernel(21, 2.0)
        fine = PSFModel.resample(kernel, 0.04, 0.02)
        assert fine.sum() == pytest.approx(1.0, rel=1e-6)
        assert _kernel_sigma(fine) == pytest.approx(4.0, rel=0.02)

    def test_round_trip_to_one_percent(self):
        """Coarsening and refining back recovers the original profile to 1%."""
        kernel = _gaussian_kernel(41, 4.0)
        back = PSFModel.resample(PSFModel.resample(kernel, 0.02, 0.04), 0.04, 0.02, out_size=41)
        assert back.sum() == pytest.approx(1.0, rel=1e-6)
        assert _kernel_sigma(back) == pytest.approx(_kernel_sigma(kernel), rel=0.01)
        assert np.abs(back - kernel).sum() < 0.01

    def test_identity_scale_is_a_no_op(self):
        """Resampling to the same scale returns the same kernel."""
        kernel = _gaussian_kernel(21, 2.0)
        same = PSFModel.resample(kernel, 0.03, 0.03)
        np.testing.assert_allclose(same, kernel, atol=1e-6)

    def test_out_size_must_be_odd(self):
        """An even out_size is rejected so the kernel stays centred."""
        with pytest.raises(ValueError, match="odd"):
            PSFModel.resample(_gaussian_kernel(21, 2.0), 0.03, 0.06, out_size=20)

    def test_non_2d_kernel_raises(self):
        """A 3-D kernel is rejected."""
        with pytest.raises(ValueError, match="2-D"):
            PSFModel.resample(np.ones((2, 5, 5)), 0.03, 0.06)

    def test_resample_to_per_band(self, gaussian_psf):
        """resample_to produces one kernel per band at the requested scales."""
        out = gaussian_psf.resample_to({b: 0.06 for b in gaussian_psf.band_names}, 0.03)
        assert set(out) == set(gaussian_psf.band_names)
        assert all(k.sum() == pytest.approx(1.0, rel=1e-6) for k in out.values())
