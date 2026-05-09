"""Tests for ObservationCube."""

import numpy as np
import pytest

from arachne.data.observation import ObservationCube

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


def _make_tan_wcs(ra=150.0, dec=2.0, crpix_x=17, crpix_y=17,
                  pixel_scale_deg=0.031 / 3600):
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
            cutout_center=(16, 16),   # (cy, cx) pixel coords (no WCS)
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
            cutout_center=(150.0, 2.0),   # (RA, Dec) = crval → image centre
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
