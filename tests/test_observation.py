"""Tests for ObservationCube."""

import numpy as np
import pytest

from arachne.data.observation import ObservationCube


def test_observation_shapes(tiny_observation):
    """ObservationCube has consistent array shapes."""
    obs = tiny_observation
    assert obs.flux.shape == (3, 16, 16)
    assert obs.variance.shape == (3, 16, 16)
    assert obs.mask.shape == (3, 16, 16)
    assert len(obs.band_names) == 3


def test_observation_n_bands(tiny_observation):
    """n_bands property returns correct count."""
    assert tiny_observation.n_bands == 3


def test_observation_image_shape(tiny_observation):
    """image_shape property returns (H, W)."""
    assert tiny_observation.image_shape == (16, 16)


def test_to_jax_returns_jax_arrays(tiny_observation_numpy):
    """to_jax() converts numpy arrays to JAX arrays."""
    import jax.numpy as jnp

    obs_jax = tiny_observation_numpy.to_jax()
    assert isinstance(obs_jax.flux, jnp.ndarray)
    assert isinstance(obs_jax.variance, jnp.ndarray)
    assert isinstance(obs_jax.mask, jnp.ndarray)


def test_to_jax_preserves_values(tiny_observation_numpy):
    """to_jax() does not change array values."""
    import numpy as np

    obs = tiny_observation_numpy
    obs_jax = obs.to_jax()
    np.testing.assert_allclose(np.asarray(obs_jax.flux), obs.flux, rtol=1e-6)


def test_band_names_preserved(tiny_observation):
    """Band names list is preserved."""
    assert tiny_observation.band_names[0] == "JWST/NIRCam.F115W"


def test_shape_mismatch_raises():
    """Mismatched flux/variance shapes raise ValueError."""
    flux = np.ones((3, 16, 16), dtype=np.float32)
    variance = np.ones((3, 8, 8), dtype=np.float32)
    mask = np.ones((3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="variance"):
        ObservationCube(
            flux=flux,
            variance=variance,
            mask=mask,
            band_names=["a", "b", "c"],
            pixel_scale=0.031,
        )


def test_band_name_count_mismatch_raises():
    """Mismatched band_names count raises ValueError."""
    flux = np.ones((3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="band names"):
        ObservationCube(
            flux=flux,
            variance=flux.copy(),
            mask=flux.copy(),
            band_names=["a", "b"],  # Only 2, should be 3
            pixel_scale=0.031,
        )


def test_mask_default_all_valid():
    """When mask is all ones, all pixels are valid."""
    flux = np.ones((2, 4, 4), dtype=np.float32)
    obs = ObservationCube(
        flux=flux,
        variance=flux.copy(),
        mask=np.ones_like(flux),
        band_names=["F115W", "F200W"],
        pixel_scale=0.031,
    )
    import jax.numpy as jnp

    obs_jax = obs.to_jax()
    assert jnp.all(obs_jax.mask == 1.0)


# ---------------------------------------------------------------------------
# from_fits tests
# ---------------------------------------------------------------------------


def _write_fits(path, data, header=None):
    """Write a 2-D numpy array to a FITS file."""
    from astropy.io import fits as afits

    hdu = afits.PrimaryHDU(data, header=header)
    afits.HDUList([hdu]).writeto(str(path), overwrite=True)


def _make_tan_wcs(ra=150.0, dec=2.0, crpix_x=17, crpix_y=17, pixel_scale_deg=0.031 / 3600):
    """Return a minimal astropy WCS with a TAN projection."""
    from astropy.wcs import WCS as AstroWCS

    w = AstroWCS(naxis=2)
    w.wcs.crpix = [crpix_x, crpix_y]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


class TestFromFits:
    """Tests for ObservationCube.from_fits."""

    def test_basic_load(self, tmp_path):
        """from_fits loads flux, variance, and mask from FITS files."""
        H, W = 16, 16
        flux_paths, var_paths, mask_paths = [], [], []
        for i in range(3):
            fp = tmp_path / f"flux_{i}.fits"
            vp = tmp_path / f"var_{i}.fits"
            mp = tmp_path / f"mask_{i}.fits"
            _write_fits(fp, np.full((H, W), float(i + 1), dtype=np.float32))
            _write_fits(vp, np.ones((H, W), dtype=np.float32))
            _write_fits(mp, np.ones((H, W), dtype=np.float32))
            flux_paths.append(fp)
            var_paths.append(vp)
            mask_paths.append(mp)

        obs = ObservationCube.from_fits(
            flux_paths=flux_paths,
            variance_paths=var_paths,
            band_names=["F115W", "F200W", "F277W"],
            mask_paths=mask_paths,
        )
        assert obs.flux.shape == (3, H, W)
        assert obs.variance.shape == (3, H, W)
        assert obs.mask.shape == (3, H, W)

    def test_no_mask_defaults_to_all_valid(self, tmp_path):
        """When mask_paths is None every pixel is marked valid."""
        H, W = 8, 8
        fp = tmp_path / "flux.fits"
        vp = tmp_path / "var.fits"
        _write_fits(fp, np.ones((H, W), dtype=np.float32))
        _write_fits(vp, np.ones((H, W), dtype=np.float32))
        obs = ObservationCube.from_fits(
            flux_paths=[fp],
            variance_paths=[vp],
            band_names=["F115W"],
        )
        assert np.all(np.asarray(obs.mask) == 1.0)

    def test_pixel_cutout(self, tmp_path):
        """Pixel-space cutout returns the correct region."""
        H, W = 32, 32
        data = np.arange(H * W, dtype=np.float32).reshape(H, W)
        fp = tmp_path / "flux.fits"
        vp = tmp_path / "var.fits"
        _write_fits(fp, data)
        _write_fits(vp, np.ones((H, W), dtype=np.float32))

        obs = ObservationCube.from_fits(
            flux_paths=[fp],
            variance_paths=[vp],
            band_names=["F115W"],
            cutout_center=(16, 16),  # (cy, cx) pixel coords (no WCS)
            cutout_size=8,
        )
        assert obs.flux.shape == (1, 8, 8)
        # The cutout should contain pixels from rows 12:20, cols 12:20
        expected = data[12:20, 12:20]
        np.testing.assert_array_equal(np.asarray(obs.flux[0]), expected)

    # --- Regression test for fix 1: multi-band WCS cutout alignment ---

    def test_wcs_cutout_all_bands_same_shape(self, tmp_path):
        """All bands have the same spatial shape after a WCS cutout.

        Regression test for the bug where bands i > 0 entered the pixel-space
        cutout branch and interpreted (RA, Dec) as (cy, cx) pixel coordinates,
        producing a misaligned or wrongly shaped cutout for every band beyond
        the first.
        """
        H, W = 32, 32
        cutout_size = 16
        wcs = _make_tan_wcs()
        header = wcs.to_header()

        flux_paths, var_paths = [], []
        for i in range(3):
            fp = tmp_path / f"flux_{i}.fits"
            vp = tmp_path / f"var_{i}.fits"
            _write_fits(fp, np.full((H, W), float(i + 1), dtype=np.float32), header=header)
            _write_fits(vp, np.ones((H, W), dtype=np.float32))
            flux_paths.append(fp)
            var_paths.append(vp)

        obs = ObservationCube.from_fits(
            flux_paths=flux_paths,
            variance_paths=var_paths,
            band_names=["F115W", "F200W", "F277W"],
            cutout_center=(150.0, 2.0),  # (RA, Dec) = crval → image centre
            cutout_size=cutout_size,
        )
        assert obs.flux.shape == (3, cutout_size, cutout_size), (
            f"Expected (3, {cutout_size}, {cutout_size}), got {obs.flux.shape}. "
            "Likely cause: bands 1+ had their cutout applied with wrong coordinates."
        )

    def test_wcs_cutout_data_values_consistent_across_bands(self, tmp_path):
        """All bands contain data from the same spatial region after a WCS cutout.

        Each band is filled with a distinct constant (1, 2, 3).  After a centred
        WCS cutout every pixel in band i should equal i + 1.
        """
        H, W = 32, 32
        wcs = _make_tan_wcs()
        header = wcs.to_header()

        flux_paths, var_paths = [], []
        for i in range(3):
            fp = tmp_path / f"flux_{i}.fits"
            vp = tmp_path / f"var_{i}.fits"
            _write_fits(fp, np.full((H, W), float(i + 1), dtype=np.float32), header=header)
            _write_fits(vp, np.ones((H, W), dtype=np.float32))
            flux_paths.append(fp)
            var_paths.append(vp)

        obs = ObservationCube.from_fits(
            flux_paths=flux_paths,
            variance_paths=var_paths,
            band_names=["F115W", "F200W", "F277W"],
            cutout_center=(150.0, 2.0),
            cutout_size=16,
        )
        for i in range(3):
            np.testing.assert_allclose(
                np.asarray(obs.flux[i]),
                float(i + 1),
                atol=1e-5,
                err_msg=f"Band {i} has wrong values after WCS cutout.",
            )


# ---------------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------------


class TestFluxUnits:
    """from_fits converts every band to nJy."""

    def _write_band(self, tmp_path, name, data, bunit=None, wcs=None, extra=None):
        """Write a flux FITS with an optional BUNIT and WCS.

        Args:
            tmp_path: Temp directory.
            name: File stem.
            data: 2-D array.
            bunit: Optional BUNIT string.
            wcs: Optional astropy WCS.
            extra: Optional dict of extra header cards.

        Returns:
            Path to the written file.
        """
        from astropy.io import fits as afits

        header = afits.Header() if wcs is None else wcs.to_header()
        if bunit is not None:
            header["BUNIT"] = bunit
        for key, value in (extra or {}).items():
            header[key] = value
        path = tmp_path / f"{name}.fits"
        _write_fits(path, np.asarray(data, dtype=np.float32), header=header)
        return path

    def test_default_flux_unit_field(self, tiny_observation_numpy):
        """A directly constructed cube reports nJy."""
        assert tiny_observation_numpy.flux_unit == "nJy"
        assert tiny_observation_numpy.to_jax().flux_unit == "nJy"

    def test_no_bunit_leaves_data_untouched(self, tmp_path):
        """Headers with no unit information are assumed to be nJy already."""
        fp = self._write_band(tmp_path, "flux", np.full((8, 8), 3.0))
        vp = self._write_band(tmp_path, "var", np.full((8, 8), 4.0))
        obs = ObservationCube.from_fits([fp], [vp], ["F115W"])
        np.testing.assert_allclose(np.asarray(obs.flux), 3.0)
        np.testing.assert_allclose(np.asarray(obs.variance), 4.0)
        assert obs.flux_unit == "nJy"

    def test_microjansky_fits(self, tmp_path):
        """A synthetic image in uJy is scaled by 1e3 and its variance by 1e6."""
        fp = self._write_band(tmp_path, "flux", np.full((8, 8), 2.0), bunit="uJy")
        vp = self._write_band(tmp_path, "var", np.full((8, 8), 5.0))
        obs = ObservationCube.from_fits([fp], [vp], ["F115W"])
        np.testing.assert_allclose(np.asarray(obs.flux), 2.0e3, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(obs.variance), 5.0e6, rtol=1e-6)

    def test_mjy_per_sr_fits(self, tmp_path):
        """A synthetic image in MJy/sr is converted using the WCS pixel area."""
        pixel_scale = 0.03
        wcs = _make_tan_wcs(pixel_scale_deg=pixel_scale / 3600)
        fp = self._write_band(tmp_path, "flux", np.ones((8, 8)), bunit="MJy/sr", wcs=wcs)
        vp = self._write_band(tmp_path, "var", np.ones((8, 8)))
        obs = ObservationCube.from_fits([fp], [vp], ["F115W"], pixel_scale=pixel_scale)
        arcsec2_sr = (np.pi / (180 * 3600)) ** 2
        expected = 1e15 * pixel_scale**2 * arcsec2_sr
        np.testing.assert_allclose(np.asarray(obs.flux), expected, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(obs.variance), expected**2, rtol=1e-4)

    def test_mjy_per_sr_without_wcs_uses_pixel_scale_argument(self, tmp_path):
        """With no WCS the pixel_scale argument supplies the pixel area."""
        fp = self._write_band(tmp_path, "flux", np.ones((8, 8)), bunit="MJy/sr")
        vp = self._write_band(tmp_path, "var", np.ones((8, 8)))
        obs = ObservationCube.from_fits([fp], [vp], ["F115W"], pixel_scale=0.06)
        arcsec2_sr = (np.pi / (180 * 3600)) ** 2
        np.testing.assert_allclose(np.asarray(obs.flux), 1e15 * 0.06**2 * arcsec2_sr, rtol=1e-6)

    def test_dja_bunit(self, tmp_path):
        """The DJA '10.0*nanoJansky' convention scales by 10."""
        fp = self._write_band(
            tmp_path, "flux", np.ones((8, 8)), bunit="10.0*nanoJansky", extra={"ZP": 28.0025}
        )
        vp = self._write_band(tmp_path, "var", np.ones((8, 8)))
        obs = ObservationCube.from_fits([fp], [vp], ["F115W"])
        # BUNIT must win over the (inconsistent) DJA ZP card.
        np.testing.assert_allclose(np.asarray(obs.flux), 10.0, rtol=1e-6)

    def test_zp_fallback_when_no_bunit(self, tmp_path):
        """With no BUNIT the header ZP is used as an AB zeropoint."""
        fp = self._write_band(tmp_path, "flux", np.ones((8, 8)), extra={"ZP": 31.4})
        vp = self._write_band(tmp_path, "var", np.ones((8, 8)))
        obs = ObservationCube.from_fits([fp], [vp], ["F115W"])
        np.testing.assert_allclose(np.asarray(obs.flux), 1.0, rtol=1e-4)

    def test_explicit_flux_unit_overrides_header(self, tmp_path):
        """An explicit flux_unit string is used instead of BUNIT."""
        fp = self._write_band(tmp_path, "flux", np.ones((8, 8)), bunit="nJy")
        vp = self._write_band(tmp_path, "var", np.ones((8, 8)))
        obs = ObservationCube.from_fits([fp], [vp], ["F115W"], flux_unit="uJy")
        np.testing.assert_allclose(np.asarray(obs.flux), 1e3, rtol=1e-6)

    def test_bad_explicit_flux_unit_raises(self, tmp_path):
        """An unparseable explicit flux_unit is an error."""
        fp = self._write_band(tmp_path, "flux", np.ones((4, 4)))
        vp = self._write_band(tmp_path, "var", np.ones((4, 4)))
        with pytest.raises(ValueError, match="not a recognised flux unit"):
            ObservationCube.from_fits([fp], [vp], ["F115W"], flux_unit="electron/s")

    def test_zeropoints_dict_overrides_bunit(self, tmp_path):
        """A per-band zeropoint takes precedence over BUNIT."""
        fp = self._write_band(tmp_path, "flux", np.ones((4, 4)), bunit="Jy")
        vp = self._write_band(tmp_path, "var", np.ones((4, 4)))
        obs = ObservationCube.from_fits([fp], [vp], ["F115W"], zeropoints={"F115W": 31.4 - 2.5})
        np.testing.assert_allclose(np.asarray(obs.flux), 10.0, rtol=1e-4)

    def test_zeropoints_list(self, tmp_path):
        """Zeropoints may also be given as a list in band order."""
        paths = []
        for i in range(2):
            paths.append(
                (
                    self._write_band(tmp_path, f"f{i}", np.ones((4, 4))),
                    self._write_band(tmp_path, f"v{i}", np.ones((4, 4))),
                )
            )
        obs = ObservationCube.from_fits(
            [p[0] for p in paths],
            [p[1] for p in paths],
            ["A", "B"],
            zeropoints=[31.4, 31.4 - 5.0],
        )
        np.testing.assert_allclose(np.asarray(obs.flux[0]), 1.0, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(obs.flux[1]), 100.0, rtol=1e-4)
