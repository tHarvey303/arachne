"""Tests for PSFModel and PSFConvolver.

The critical test is test_psf_centering: a PSF-convolved point source must
have its centroid at the same location as the input. A missing ifftshift in
PSFConvolver or PSFModel.pad_to_image_size will cause this test to fail.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.psf.convolution import PSFConvolver


def test_psf_model_shape(gaussian_psf):
    """PSFModel has shape (N_bands, H_psf, W_psf)."""
    assert gaussian_psf.kernels.ndim == 3
    assert gaussian_psf.kernels.shape[0] == 3


def test_psf_model_n_bands(gaussian_psf):
    """n_bands property is correct."""
    assert gaussian_psf.n_bands == 3


def test_pad_to_image_size_shape(gaussian_psf):
    """pad_to_image_size() returns (N_bands, H, W) kernels."""
    padded = gaussian_psf.pad_to_image_size(16, 16)
    assert padded.kernels.shape == (3, 16, 16)


def test_pad_to_image_size_too_small_raises(gaussian_psf):
    """pad_to_image_size() raises when image is smaller than PSF."""
    # PSF is 9x9; image 4x4 is too small
    with pytest.raises(ValueError, match="larger than image"):
        gaussian_psf.pad_to_image_size(4, 4)


def test_psf_centering(delta_psf):
    """PSF-convolved point source has centroid at the input position.

    This is the critical test: if ifftshift is missing, the convolved
    point source will be spatially shifted from the input centroid.
    """
    H, W = 16, 16
    convolver = PSFConvolver(delta_psf, image_shape=(H, W))

    # Create a point source at pixel (6, 7)
    py, px = 6, 7
    image = np.zeros((3, H, W), dtype=np.float32)
    image[:, py, px] = 1.0
    image_jax = jnp.array(image)

    convolved = convolver(image_jax)

    # Find centroid of first band
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    band = convolved[0]
    total = jnp.sum(band)
    centroid_y = jnp.sum(yy * band) / total
    centroid_x = jnp.sum(xx * band) / total

    assert abs(float(centroid_y) - py) < 0.5, (
        f"Centroid y={float(centroid_y):.2f} shifted from input py={py}. "
        "Check ifftshift in PSFConvolver."
    )
    assert abs(float(centroid_x) - px) < 0.5, (
        f"Centroid x={float(centroid_x):.2f} shifted from input px={px}. "
        "Check ifftshift in PSFConvolver."
    )


def test_psf_flux_conservation(gaussian_psf):
    """PSF convolution conserves total flux (sum of image)."""
    H, W = 16, 16
    convolver = PSFConvolver(gaussian_psf, image_shape=(H, W))

    rng = np.random.default_rng(42)
    image = jnp.array(rng.uniform(0, 1, (3, H, W)).astype(np.float32))
    convolved = convolver(image)

    for b in range(3):
        orig_flux = float(jnp.sum(image[b]))
        conv_flux = float(jnp.sum(convolved[b]))
        assert abs(conv_flux - orig_flux) / (orig_flux + 1e-10) < 0.01, (
            f"Band {b}: flux not conserved. Original={orig_flux:.4f}, convolved={conv_flux:.4f}"
        )


def test_psf_convolver_differentiable(gaussian_psf):
    """jax.grad can differentiate through PSFConvolver."""
    H, W = 16, 16
    convolver = PSFConvolver(gaussian_psf, image_shape=(H, W))

    def loss(image):
        return jnp.sum(convolver(image))

    image = jnp.ones((3, H, W))
    grad = jax.grad(loss)(image)
    assert grad.shape == image.shape
    assert jnp.all(jnp.isfinite(grad))


def test_psf_convolver_output_shape(gaussian_psf):
    """PSFConvolver output has the same shape as input."""
    H, W = 16, 16
    convolver = PSFConvolver(gaussian_psf, image_shape=(H, W))
    image = jnp.ones((3, H, W))
    out = convolver(image)
    assert out.shape == (3, H, W)


def test_delta_psf_is_identity(delta_psf):
    """A delta-function PSF should leave a smooth image nearly unchanged."""
    H, W = 16, 16
    convolver = PSFConvolver(delta_psf, image_shape=(H, W))

    rng = np.random.default_rng(7)
    image = jnp.array(rng.uniform(1, 2, (3, H, W)).astype(np.float32))
    convolved = convolver(image)

    # With a delta PSF, convolved ~ original (small FFT rounding errors)
    np.testing.assert_allclose(np.asarray(convolved), np.asarray(image), atol=1e-4)


def test_psf_centering_all_bands(delta_psf):
    """PSF-convolved point source centroid is correct for every band.

    Regression test: verifies that ifftshift is applied consistently for all
    bands, not only band 0.  A bug in per-band padding would shift the
    centroid for bands 1+.
    """
    H, W = 16, 16
    convolver = PSFConvolver(delta_psf, image_shape=(H, W))

    py, px = 5, 8
    image = np.zeros((3, H, W), dtype=np.float32)
    image[:, py, px] = 1.0
    convolved = convolver(jnp.array(image))

    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")

    for b in range(3):
        band = convolved[b]
        total = jnp.sum(band)
        cy = float(jnp.sum(yy * band) / total)
        cx = float(jnp.sum(xx * band) / total)
        assert abs(cy - py) < 0.5, f"Band {b}: centroid_y={cy:.2f} shifted from py={py}"
        assert abs(cx - px) < 0.5, f"Band {b}: centroid_x={cx:.2f} shifted from px={px}"


# ---------------------------------------------------------------------------
# PSFModel.from_fits tests
# ---------------------------------------------------------------------------


def _write_psf_fits(path, kernel):
    """Write a 2-D numpy array as a FITS file."""
    from astropy.io import fits as afits
    hdu = afits.PrimaryHDU(kernel)
    afits.HDUList([hdu]).writeto(str(path), overwrite=True)


class TestPSFModelFromFits:
    """Tests for PSFModel.from_fits."""

    def test_basic_load_shape(self, tmp_path):
        """from_fits returns correct shape and band ordering."""
        from arachne.data.psf import PSFModel
        bands = ["F115W", "F200W", "F277W"]
        psf_paths = {}
        for band in bands:
            k = np.ones((9, 9), dtype=np.float32)
            k /= k.sum()
            fp = tmp_path / f"psf_{band}.fits"
            _write_psf_fits(fp, k)
            psf_paths[band] = fp

        psf = PSFModel.from_fits(psf_paths)
        assert psf.kernels.shape == (3, 9, 9)
        assert psf.band_names == bands

    def test_values_preserved(self, tmp_path):
        """Kernel values loaded from FITS match what was written."""
        from arachne.data.psf import PSFModel
        rng = np.random.default_rng(42)
        k = rng.uniform(0, 1, (7, 7)).astype(np.float32)
        k /= k.sum()
        fp = tmp_path / "psf.fits"
        _write_psf_fits(fp, k)

        psf = PSFModel.from_fits({"band_A": fp})
        np.testing.assert_allclose(psf.kernels[0], k, rtol=1e-5)

    def test_3d_psf_takes_first_plane(self, tmp_path):
        """A (1, H, W) PSF FITS file is reduced to (H, W) by taking the first plane."""
        from astropy.io import fits as afits
        from arachne.data.psf import PSFModel
        k3d = np.ones((1, 9, 9), dtype=np.float32) / 81.0
        fp = tmp_path / "psf_3d.fits"
        afits.HDUList([afits.PrimaryHDU(k3d)]).writeto(str(fp), overwrite=True)

        psf = PSFModel.from_fits({"band_A": fp})
        assert psf.kernels.shape == (1, 9, 9)

    def test_pads_to_common_size(self, tmp_path):
        """PSFs of different sizes are padded to the largest common size."""
        from arachne.data.psf import PSFModel
        small = np.ones((5, 5), dtype=np.float32)
        small /= small.sum()
        large = np.ones((9, 9), dtype=np.float32)
        large /= large.sum()

        fp_s = tmp_path / "psf_small.fits"
        fp_l = tmp_path / "psf_large.fits"
        _write_psf_fits(fp_s, small)
        _write_psf_fits(fp_l, large)

        psf = PSFModel.from_fits({"band_A": fp_s, "band_B": fp_l})
        assert psf.kernels.shape == (2, 9, 9)

    def test_n_bands_property(self, tmp_path):
        """n_bands property equals number of loaded PSFs."""
        from arachne.data.psf import PSFModel
        paths = {}
        for i in range(4):
            k = np.eye(5, dtype=np.float32)
            k /= k.sum()
            fp = tmp_path / f"psf_{i}.fits"
            _write_psf_fits(fp, k)
            paths[f"band_{i}"] = fp

        psf = PSFModel.from_fits(paths)
        assert psf.n_bands == 4
