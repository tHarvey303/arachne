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
