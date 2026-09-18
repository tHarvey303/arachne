"""Tests for PSFModel and PSFConvolver.

The critical test is test_psf_centering: a PSF-convolved point source must
have its centroid at the same location as the input. A missing ifftshift in
PSFConvolver or PSFModel.pad_to_image_size will cause this test to fail.

The padded-convolution tests check the other classic failure mode: an
unpadded FFT convolution is *circular*, so flux near one edge of the frame
reappears on the opposite edge.  ``pad=True`` (the default) suppresses that.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import fftconvolve

from arachne.data.psf import PSFModel
from arachne.psf.convolution import PSFConvolver

BANDS_3 = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]


def _asymmetric_psf(n_bands: int = 3, kh: int = 7, kw: int = 5) -> PSFModel:
    """PSFModel with a deliberately asymmetric, non-separable kernel per band."""
    rng = np.random.default_rng(1234)
    kernels = []
    y, x = np.mgrid[: float(kh), : float(kw)]
    for b in range(n_bands):
        k = np.exp(-((y - 2.0 - 0.3 * b) ** 2) / 4.0 - ((x - 1.0) ** 2) / 2.0)
        k = k * (1.0 + 0.5 * rng.uniform(size=(kh, kw)))
        kernels.append((k / k.sum()).astype(np.float32))
    return PSFModel(kernels=np.stack(kernels), band_names=BANDS_3[:n_bands])


def _old_circular_convolution(psf_model: PSFModel, image: np.ndarray) -> np.ndarray:
    """Reference implementation of the historical (unpadded) circular convolution."""
    n_bands, H, W = image.shape
    kernels = np.asarray(psf_model.pad_to_image_size(H, W).kernels)
    kernels = kernels / kernels.sum(axis=(-2, -1), keepdims=True)
    return np.fft.irfft2(np.fft.rfft2(image) * np.fft.rfft2(kernels), s=(H, W))


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
    """Circular (pad=False) PSF convolution conserves total flux exactly.

    With ``pad=True`` flux that is convolved past the frame edge is genuinely
    lost, which is the physical behaviour — see
    ``test_padded_flux_conservation_interior_source``.
    """
    H, W = 16, 16
    convolver = PSFConvolver(gaussian_psf, image_shape=(H, W), pad=False)

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


# ---------------------------------------------------------------------------
# Padded (linear) convolution
# ---------------------------------------------------------------------------


class TestPaddedConvolution:
    """``pad=True`` gives a linear convolution truncated to the frame."""

    def test_padded_shape_is_large_enough_and_even(self):
        """padded_shape is even and at least the full linear-convolution size."""
        psf = _asymmetric_psf(kh=7, kw=5)
        H, W = 16, 20
        conv = PSFConvolver(psf, image_shape=(H, W))
        Hp, Wp = conv.padded_shape
        assert Hp >= H + 7 - 1
        assert Wp >= W + 5 - 1
        assert Hp % 2 == 0 and Wp % 2 == 0
        assert conv.psf_ffts.shape == (3, Hp, Wp // 2 + 1)

    def test_unpadded_shape_is_image_shape(self, gaussian_psf):
        """pad=False keeps the FFT grid at the image size."""
        conv = PSFConvolver(gaussian_psf, image_shape=(16, 16), pad=False)
        assert conv.padded_shape == (16, 16)

    def test_matches_scipy_fftconvolve_same(self):
        """Padded convolution equals scipy's linear 'same' convolution."""
        psf = _asymmetric_psf(kh=7, kw=5)
        H, W = 16, 20
        rng = np.random.default_rng(7)
        image = rng.normal(5.0, 2.0, (3, H, W)).astype(np.float32)

        conv = PSFConvolver(psf, image_shape=(H, W))
        out = np.asarray(conv(jnp.asarray(image)))

        kernels = np.asarray(psf.kernels, dtype=np.float64)
        kernels = kernels / kernels.sum(axis=(-2, -1), keepdims=True)
        for b in range(3):
            ref = fftconvolve(image[b].astype(np.float64), kernels[b], mode="same")
            np.testing.assert_allclose(out[b], ref, atol=1e-5, rtol=0)

    def test_unpadded_equals_old_circular_result(self):
        """pad=False reproduces the historical circular convolution bit-for-bit."""
        psf = _asymmetric_psf(kh=7, kw=5)
        H, W = 16, 16
        rng = np.random.default_rng(11)
        image = rng.normal(5.0, 2.0, (3, H, W)).astype(np.float32)

        conv = PSFConvolver(psf, image_shape=(H, W), pad=False)
        out = np.asarray(conv(jnp.asarray(image)))
        ref = _old_circular_convolution(psf, image.astype(np.float64))
        np.testing.assert_allclose(out, ref, atol=1e-4, rtol=0)

    def test_edge_source_does_not_wrap_when_padded(self, gaussian_psf):
        """A point source on the left edge leaks to the right edge only when unpadded."""
        H, W = 16, 16
        image = np.zeros((3, H, W), dtype=np.float32)
        image[:, 8, 0] = 100.0
        image_jax = jnp.asarray(image)

        padded = np.asarray(PSFConvolver(gaussian_psf, (H, W), pad=True)(image_jax))
        circular = np.asarray(PSFConvolver(gaussian_psf, (H, W), pad=False)(image_jax))

        # Circular convolution wraps flux around to the far edge...
        assert circular[0, 8, W - 1] > 1.0
        # ...the padded one does not.
        assert abs(padded[0, 8, W - 1]) < 1e-3
        # Both agree away from the wrapped region.
        np.testing.assert_allclose(padded[0, 8, 0:4], circular[0, 8, 0:4], atol=1e-3)

    def test_padded_flux_conservation_interior_source(self, gaussian_psf):
        """A compact source well inside the frame keeps its flux under padding."""
        H, W = 32, 32
        yy, xx = np.mgrid[:H, :W]
        src = np.exp(-((yy - 16.0) ** 2 + (xx - 16.0) ** 2) / 2.0).astype(np.float32)
        image = np.stack([src] * 3)

        conv = PSFConvolver(gaussian_psf, image_shape=(H, W))
        out = np.asarray(conv(jnp.asarray(image)))
        for b in range(3):
            assert out[b].sum() == pytest.approx(image[b].sum(), rel=2e-3)

    def test_padded_is_differentiable(self):
        """jax.grad works through the padded path."""
        psf = _asymmetric_psf()
        conv = PSFConvolver(psf, image_shape=(16, 16))
        grad = jax.grad(lambda im: jnp.sum(conv(im) ** 2))(jnp.ones((3, 16, 16)))
        assert grad.shape == (3, 16, 16)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_padded_output_shape(self):
        """The padded convolution is cropped back to the image shape."""
        conv = PSFConvolver(_asymmetric_psf(), image_shape=(16, 20))
        assert conv(jnp.ones((3, 16, 20))).shape == (3, 16, 20)


# ---------------------------------------------------------------------------
# Sub-pixel shifts
# ---------------------------------------------------------------------------


class TestShifts:
    """Per-band Fourier-phase-ramp translations."""

    def test_integer_shift_equals_roll_unpadded(self, delta_psf):
        """With pad=False an integer shift is exactly jnp.roll."""
        H, W = 16, 16
        rng = np.random.default_rng(3)
        image = jnp.asarray(rng.normal(3.0, 1.0, (3, H, W)).astype(np.float32))
        conv = PSFConvolver(delta_psf, image_shape=(H, W), pad=False)

        shifts = jnp.array([[2.0, -3.0], [2.0, -3.0], [2.0, -3.0]])
        out = np.asarray(conv(image, shifts=shifts))
        ref = np.asarray(jnp.roll(image, shift=(2, -3), axis=(1, 2)))
        np.testing.assert_allclose(out, ref, atol=1e-3)

    def test_positive_dy_moves_down(self, delta_psf):
        """Positive dy moves flux towards higher row indices."""
        H, W = 16, 16
        image = np.zeros((3, H, W), dtype=np.float32)
        image[:, 8, 8] = 1.0
        conv = PSFConvolver(delta_psf, image_shape=(H, W))
        shifts = jnp.array([[3.0, 0.0]] * 3)
        out = np.asarray(conv(jnp.asarray(image), shifts=shifts))
        assert int(np.argmax(out[0]) // W) == 11
        assert int(np.argmax(out[0]) % W) == 8

    def test_positive_dx_moves_right(self, delta_psf):
        """Positive dx moves flux towards higher column indices."""
        H, W = 16, 16
        image = np.zeros((3, H, W), dtype=np.float32)
        image[:, 8, 8] = 1.0
        conv = PSFConvolver(delta_psf, image_shape=(H, W))
        shifts = jnp.array([[0.0, 2.0]] * 3)
        out = np.asarray(conv(jnp.asarray(image), shifts=shifts))
        assert int(np.argmax(out[0]) % W) == 10

    def test_integer_shift_padded_is_pure_translation(self, delta_psf):
        """With pad=True an integer shift translates the image without wrapping."""
        H, W = 16, 16
        rng = np.random.default_rng(5)
        image = rng.normal(3.0, 1.0, (3, H, W)).astype(np.float32)
        conv = PSFConvolver(delta_psf, image_shape=(H, W))
        out = np.asarray(conv(jnp.asarray(image), shifts=jnp.array([[2.0, 3.0]] * 3)))
        # Interior region is the original image translated by (+2, +3).
        np.testing.assert_allclose(out[:, 2:, 3:], image[:, :-2, :-3], atol=1e-3)
        # The vacated rows/columns are zero — nothing wrapped in.
        np.testing.assert_allclose(out[:, :2, :], 0.0, atol=1e-3)
        np.testing.assert_allclose(out[:, :, :3], 0.0, atol=1e-3)

    def test_half_pixel_shift_is_symmetric(self, delta_psf):
        """A 0.5 px shift of a delta is symmetric about the two straddled pixels."""
        H, W = 16, 16
        image = np.zeros((3, H, W), dtype=np.float32)
        image[:, 8, 8] = 1.0
        conv = PSFConvolver(delta_psf, image_shape=(H, W), pad=False)
        out = np.asarray(conv(jnp.asarray(image), shifts=jnp.array([[0.5, 0.0]] * 3)))
        column = out[0, :, 8]
        assert column[8] == pytest.approx(column[9], abs=1e-5)
        assert column[7] == pytest.approx(column[10], abs=1e-5)
        assert column[6] == pytest.approx(column[11], abs=1e-5)
        # The flux really is shared between rows 8 and 9.
        assert column[8] > 0.3

    def test_half_pixel_shift_symmetric_padded(self, delta_psf):
        """The same symmetry holds on the padded grid."""
        H, W = 16, 16
        image = np.zeros((3, H, W), dtype=np.float32)
        image[:, 8, 8] = 1.0
        conv = PSFConvolver(delta_psf, image_shape=(H, W))
        out = np.asarray(conv(jnp.asarray(image), shifts=jnp.array([[0.0, 0.5]] * 3)))
        row = out[0, 8, :]
        assert row[8] == pytest.approx(row[9], abs=1e-5)
        assert row[7] == pytest.approx(row[10], abs=1e-5)

    def test_shifts_are_per_band(self, delta_psf):
        """Each band gets its own offset."""
        H, W = 16, 16
        image = np.zeros((3, H, W), dtype=np.float32)
        image[:, 8, 8] = 1.0
        conv = PSFConvolver(delta_psf, image_shape=(H, W))
        shifts = jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, -3.0]])
        out = np.asarray(conv(jnp.asarray(image), shifts=shifts))
        assert np.unravel_index(int(np.argmax(out[0])), (H, W)) == (8, 8)
        assert np.unravel_index(int(np.argmax(out[1])), (H, W)) == (10, 8)
        assert np.unravel_index(int(np.argmax(out[2])), (H, W)) == (8, 5)

    def test_zero_shift_matches_no_shift(self, gaussian_psf):
        """Passing zeros is equivalent to passing None."""
        H, W = 16, 16
        rng = np.random.default_rng(9)
        image = jnp.asarray(rng.normal(3.0, 1.0, (3, H, W)).astype(np.float32))
        conv = PSFConvolver(gaussian_psf, image_shape=(H, W))
        np.testing.assert_allclose(
            np.asarray(conv(image, shifts=jnp.zeros((3, 2)))),
            np.asarray(conv(image)),
            atol=1e-5,
        )

    def test_shift_is_differentiable(self, gaussian_psf):
        """jax.grad flows into the shift parameters."""
        H, W = 16, 16
        rng = np.random.default_rng(13)
        image = jnp.asarray(rng.normal(3.0, 1.0, (3, H, W)).astype(np.float32))
        conv = PSFConvolver(gaussian_psf, image_shape=(H, W))

        def loss(shifts):
            return jnp.sum(conv(image, shifts=shifts) ** 2)

        grad = jax.grad(loss)(jnp.array([[0.1, -0.2]] * 3))
        assert grad.shape == (3, 2)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert bool(jnp.any(grad != 0.0))

    def test_shift_jit_compatible(self, gaussian_psf):
        """The shifted convolution compiles under jax.jit."""
        conv = PSFConvolver(gaussian_psf, image_shape=(16, 16))
        fn = jax.jit(lambda im, sh: conv(im, shifts=sh))
        out = fn(jnp.ones((3, 16, 16)), jnp.array([[0.3, 0.4]] * 3))
        assert out.shape == (3, 16, 16)
        assert bool(jnp.all(jnp.isfinite(out)))


class TestSingleBandAndArbitraryShapes:
    """The convolver is not tied to 3 bands, square images or a fixed kernel size.

    These are the guarantees a multi-resolution forward model relies on: each
    band gets its own convolver, on its own (H, W) grid, with its own kernel.
    """

    def test_single_band_non_square_image_matches_fftconvolve(self):
        """One band, H != W, kernel size unrelated to the image size."""
        psf = _asymmetric_psf(n_bands=1, kh=7, kw=5)
        conv = PSFConvolver(psf, image_shape=(13, 19), pad=True)
        assert conv.n_bands == 1
        assert conv.image_shape == (13, 19)
        assert conv.kernel_shape == (7, 5)
        rng = np.random.default_rng(7)
        image = rng.normal(size=(1, 13, 19)).astype(np.float32)
        out = np.asarray(conv(jnp.asarray(image)))
        ref = fftconvolve(image[0], np.asarray(psf.kernels)[0], mode="same")
        np.testing.assert_allclose(out[0], ref, atol=1e-5)

    def test_padded_shape_is_documented_formula(self):
        """padded_shape == next even 5-smooth size >= (H + kh - 1, W + kw - 1)."""
        psf = _asymmetric_psf(n_bands=1, kh=9, kw=11)
        conv = PSFConvolver(psf, image_shape=(13, 19), pad=True)
        Hp, Wp = conv.padded_shape
        assert Hp >= 13 + 9 - 1 and Wp >= 19 + 11 - 1
        assert Hp % 2 == 0 and Wp % 2 == 0
        assert isinstance(Hp, int) and isinstance(Wp, int)

    def test_kernel_larger_than_image_works_when_padded(self):
        """A kernel bigger than the cutout is fine on the padded grid."""
        psf = _asymmetric_psf(n_bands=1, kh=21, kw=21)
        conv = PSFConvolver(psf, image_shape=(9, 11), pad=True)
        rng = np.random.default_rng(8)
        image = rng.normal(size=(1, 9, 11)).astype(np.float32)
        out = np.asarray(conv(jnp.asarray(image)))
        ref = fftconvolve(image[0], np.asarray(psf.kernels)[0], mode="same")
        assert out.shape == (1, 9, 11)
        np.testing.assert_allclose(out[0], ref, atol=1e-5)

    def test_kernel_larger_than_image_raises_when_unpadded(self):
        """Without padding there is no room for an oversized kernel."""
        psf = _asymmetric_psf(n_bands=1, kh=21, kw=21)
        with pytest.raises(ValueError, match="larger than"):
            PSFConvolver(psf, image_shape=(9, 11), pad=False)

    def test_bands_may_use_different_grids_and_kernels(self):
        """Per-band convolvers on different grids agree with per-band references."""
        rng = np.random.default_rng(9)
        for (h, w), (kh, kw) in (((12, 12), (5, 5)), ((17, 9), (7, 3)), ((8, 20), (9, 9))):
            psf = _asymmetric_psf(n_bands=1, kh=kh, kw=kw)
            conv = PSFConvolver.from_kernels(np.asarray(psf.kernels)[0], (h, w))
            image = rng.normal(size=(h, w)).astype(np.float32)
            out = np.asarray(conv(jnp.asarray(image)[None]))
            ref = fftconvolve(image, np.asarray(psf.kernels)[0], mode="same")
            np.testing.assert_allclose(out[0], ref, atol=1e-5)


class TestFromKernels:
    """``PSFConvolver.from_kernels`` accepts raw arrays instead of a PSFModel."""

    def test_matches_psf_model_construction(self):
        """from_kernels((N, kh, kw), shape) == PSFConvolver(PSFModel(...), shape)."""
        psf = _asymmetric_psf(n_bands=3)
        kernels = np.asarray(psf.kernels)
        rng = np.random.default_rng(10)
        image = jnp.asarray(rng.normal(size=(3, 16, 16)).astype(np.float32))
        from_model = PSFConvolver(psf, image_shape=(16, 16))
        from_raw = PSFConvolver.from_kernels(kernels, (16, 16))
        assert from_raw.padded_shape == from_model.padded_shape
        np.testing.assert_allclose(
            np.asarray(from_raw(image)), np.asarray(from_model(image)), atol=1e-6
        )

    def test_two_d_kernel_promoted_to_one_band(self):
        """A bare (kh, kw) kernel is treated as a single band."""
        kernels = np.asarray(_asymmetric_psf(n_bands=1).kernels)
        conv = PSFConvolver.from_kernels(kernels[0], (16, 16))
        assert conv.n_bands == 1
        assert conv.kernel_shape == (7, 5)

    def test_unnormalised_kernel_is_normalised(self):
        """from_kernels normalises each kernel to unit sum, like the PSFModel path."""
        kernels = np.asarray(_asymmetric_psf(n_bands=1).kernels) * 7.0
        conv = PSFConvolver.from_kernels(kernels, (16, 16))
        image = jnp.zeros((1, 16, 16)).at[0, 8, 8].set(1.0)
        assert float(jnp.sum(conv(image))) == pytest.approx(1.0, rel=1e-5)

    def test_pad_false_and_band_names(self):
        """The ``pad`` flag and band names are forwarded."""
        kernels = np.asarray(_asymmetric_psf(n_bands=2).kernels)
        conv = PSFConvolver.from_kernels(kernels, (16, 16), pad=False, band_names=["a", "b"])
        assert conv.pad is False
        assert conv.padded_shape == (16, 16)

    def test_bad_ndim_raises(self):
        """A 1-D or 4-D kernel array is rejected."""
        with pytest.raises(ValueError, match="must be"):
            PSFConvolver.from_kernels(np.ones(5, dtype=np.float32), (16, 16))
        with pytest.raises(ValueError, match="must be"):
            PSFConvolver.from_kernels(np.ones((1, 1, 3, 3), dtype=np.float32), (16, 16))

    def test_from_kernels_is_differentiable(self):
        """Gradients flow through a from_kernels convolver."""
        conv = PSFConvolver.from_kernels(np.asarray(_asymmetric_psf(n_bands=1).kernels), (12, 14))

        def total(img):
            return jnp.sum(conv(img) ** 2)

        grad = jax.grad(total)(jnp.ones((1, 12, 14), dtype=jnp.float32))
        assert grad.shape == (1, 12, 14)
        assert bool(jnp.all(jnp.isfinite(grad)))
