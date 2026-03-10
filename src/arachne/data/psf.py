"""PSF model container for per-band PSF kernels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astropy.io import fits

from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


@dataclass
class PSFModel:
    """Per-band PSF kernel container.

    Holds PSF kernels for all photometric bands. Kernels can be loaded from
    FITS files and padded to the science image size for FFT-based convolution.

    Attributes:
        kernels: PSF kernel array of shape (N_bands, H_psf, W_psf).
        band_names: List of band identifiers matching the observation bands.
    """

    kernels: np.ndarray | jnp.ndarray
    band_names: list[str]

    def __post_init__(self) -> None:
        """Validate consistency between kernels and band_names."""
        if self.kernels.ndim != 3:
            raise ValueError(f"kernels must be 3D (N_bands, H, W), got shape {self.kernels.shape}")
        if self.kernels.shape[0] != len(self.band_names):
            raise ValueError(
                f"kernels has {self.kernels.shape[0]} planes but {len(self.band_names)} band names."
            )

    @classmethod
    def from_fits(cls, psf_paths: dict[str, str | Path]) -> "PSFModel":
        """Load PSF kernels from FITS files.

        Args:
            psf_paths: Mapping from band name to path of the PSF FITS file.
                E.g. ``{"JWST/NIRCam.F115W": "psf_f115w.fits"}``.

        Returns:
            PSFModel with float32 numpy kernel arrays.

        Raises:
            FileNotFoundError: If any FITS path does not exist.
        """
        band_names = list(psf_paths.keys())
        kernels = []
        for band, path in psf_paths.items():
            with fits.open(path) as hdul:
                kernel = hdul[0].data.astype(np.float32)
                if kernel.ndim == 3:
                    # Some PSF files store (1, H, W); take first plane
                    kernel = kernel[0]
                elif kernel.ndim != 2:
                    raise ValueError(
                        f"PSF FITS for band {band} has unexpected shape {kernel.shape}."
                    )
            kernels.append(kernel)
            logger.debug(f"Loaded PSF for {band}: shape {kernel.shape}")

        # Pad to a common size (largest PSF across bands)
        max_h = max(k.shape[0] for k in kernels)
        max_w = max(k.shape[1] for k in kernels)
        padded = []
        for k in kernels:
            pad_h = max_h - k.shape[0]
            pad_w = max_w - k.shape[1]
            padded.append(
                np.pad(
                    k,
                    ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                )
            )

        kernels_array = np.stack(padded, axis=0)
        logger.info(
            f"Loaded PSFModel: {len(band_names)} bands, kernel shape {kernels_array.shape[1:]}"
        )
        return cls(kernels=kernels_array, band_names=band_names)

    def pad_to_image_size(self, H: int, W: int) -> "PSFModel":
        """Zero-pad and ifftshift kernels to (N_bands, H, W) for FFT convolution.

        The PSF kernels are padded to the full image size so that rfft2 can be
        applied directly. ifftshift is applied to place the PSF peak at pixel (0,0),
        which is the convention required for correct FFT-based convolution.

        Args:
            H: Target image height in pixels.
            W: Target image width in pixels.

        Returns:
            New PSFModel with kernels of shape (N_bands, H, W).

        Raises:
            ValueError: If the PSF kernel is larger than the target image size.
        """
        n_bands, h_psf, w_psf = self.kernels.shape
        if h_psf > H or w_psf > W:
            raise ValueError(
                f"PSF kernel ({h_psf}×{w_psf}) is larger than image ({H}×{W}). "
                "Either crop the PSF or use a larger image cutout."
            )

        # Place the PSF centre at (H//2, W//2) so that ifftshift moves it
        # exactly to (0, 0) — the convention for correct FFT convolution.
        # Symmetric padding (pad_h//2 each side) is wrong when H-h_psf is odd:
        # it places the centre at (h_psf//2 + pad_h//2) ≠ H//2.
        center_h, center_w = h_psf // 2, w_psf // 2
        pad_top = H // 2 - center_h
        pad_left = W // 2 - center_w
        pad_bottom = H - h_psf - pad_top
        pad_right = W - w_psf - pad_left
        padded = np.pad(
            np.asarray(self.kernels),
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        )
        # ifftshift: move PSF peak from (H//2, W//2) to (0,0) for FFT convention
        padded = np.fft.ifftshift(padded, axes=(-2, -1))
        return PSFModel(kernels=padded.astype(np.float32), band_names=self.band_names)

    @property
    def n_bands(self) -> int:
        """Number of PSF bands."""
        return self.kernels.shape[0]
