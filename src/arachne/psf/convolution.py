"""FFT-based differentiable PSF convolution."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from arachne.data.psf import PSFModel
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


class PSFConvolver:
    """Differentiable PSF convolution via FFT with pre-computed PSF spectra.

    The PSF FFTs are computed once at construction and frozen.  During
    inference, each call requires only two FFTs (forward + inverse) per band,
    making convolution cheap relative to emulator evaluation.

    The ifftshift convention
    ------------------------
    ifftshift is applied to the padded PSF *before* computing its FFT.  This
    moves the PSF peak from the image centre (where it sits naturally after
    padding) to pixel (0, 0) — the convention required so that
    ``irfft2(rfft2(image) * rfft2(psf))`` yields a convolution without any
    spatial shift artefact.

    **Missing ifftshift is the most common PSF convolution bug**: the sampler
    will silently compensate by shifting the spatial model, producing
    incorrectly inferred component positions.  The test ``test_psf_centering``
    guards against this.

    Attributes:
        psf_ffts: Pre-computed PSF FFTs of shape (N_bands, H, W//2+1), complex.
        image_shape: Spatial image dimensions (H, W).
        n_bands: Number of photometric bands.
    """

    def __init__(self, psf_model: PSFModel, image_shape: tuple[int, int]) -> None:
        """Pre-compute PSF FFTs for the given image shape.

        Args:
            psf_model: PSFModel containing per-band kernel arrays.
            image_shape: (H, W) spatial dimensions of the science image.
        """
        H, W = image_shape
        self.image_shape = image_shape
        self.n_bands = psf_model.n_bands

        # Pad kernels to image size and apply ifftshift
        padded_psf = psf_model.pad_to_image_size(H, W)
        kernels = np.asarray(padded_psf.kernels)  # (N_bands, H, W), ifftshifted

        # Normalise each kernel to sum to 1 (flux conservation)
        kernel_sums = kernels.sum(axis=(-2, -1), keepdims=True)
        kernel_sums = np.where(kernel_sums == 0, 1.0, kernel_sums)
        kernels = kernels / kernel_sums

        # Pre-compute rfft2 — the PSF FFT peak is at (0,0) after ifftshift
        psf_ffts_np = np.fft.rfft2(kernels)  # (N_bands, H, W//2+1) complex
        self.psf_ffts = jnp.array(psf_ffts_np)

        logger.info(
            f"PSFConvolver initialised: {self.n_bands} bands, image shape {image_shape}, "
            f"PSF FFT shape {self.psf_ffts.shape}"
        )

    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        """Convolve a multi-band image with the per-band PSFs.

        This function is fully differentiable with ``jax.grad``.

        Args:
            image: Multi-band image array of shape (N_bands, H, W).

        Returns:
            PSF-convolved image of shape (N_bands, H, W).
        """
        H, W = self.image_shape
        image_fft = jnp.fft.rfft2(image)  # (N_bands, H, W//2+1) complex
        convolved_fft = image_fft * self.psf_ffts
        return jnp.fft.irfft2(convolved_fft, s=(H, W))  # (N_bands, H, W)
