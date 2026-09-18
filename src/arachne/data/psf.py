"""PSF model container for per-band PSF kernels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

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

    @staticmethod
    def resample(
        kernel: np.ndarray,
        from_scale: float,
        to_scale: float,
        out_size: int | None = None,
    ) -> np.ndarray:
        """Resample a PSF kernel from one pixel scale to another, conserving flux.

        The kernel is treated as a band-limited surface sampled on a grid of
        ``from_scale`` arcsec pixels centred on ``((h - 1) / 2, (w - 1) / 2)``.
        It is re-sampled onto a centred, odd-sized grid of ``to_scale`` arcsec
        pixels with third-order spline interpolation
        (:func:`scipy.ndimage.map_coordinates`, equivalent to
        :func:`scipy.ndimage.zoom` with ``order=3`` but with explicit control of
        the centre), multiplied by the pixel-area ratio and then renormalised to
        sum exactly 1.

        This is interpolation, not exact area-weighted rebinning: the latter is
        only exact when ``from_scale / to_scale`` is rational *and* the kernel is
        a piecewise-constant surface, which a PSF is not.  Because the output is
        renormalised, total flux is conserved by construction; what the cubic
        interpolation costs is a little sub-pixel fidelity when the PSF is
        critically sampled.  Round-tripping a Gaussian through a 2x coarser grid
        and back reproduces the original profile to better than 1%.

        Args:
            kernel: 2-D PSF kernel.
            from_scale: Pixel scale of ``kernel`` in arcsec/pixel.
            to_scale: Desired pixel scale in arcsec/pixel.
            out_size: Optional odd output size in pixels.  Defaults to the
                smallest odd size that covers the same angular extent.

        Returns:
            2-D kernel on the new grid, normalised to sum 1.

        Raises:
            ValueError: If the kernel is not 2-D, a scale is non-positive, or
                ``out_size`` is not a positive odd integer.
        """
        arr = np.asarray(kernel, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"PSF kernel must be 2-D, got shape {arr.shape}.")
        if from_scale <= 0 or to_scale <= 0:
            raise ValueError(
                f"Pixel scales must be positive, got from_scale={from_scale}, to_scale={to_scale}."
            )

        ratio = float(to_scale) / float(from_scale)  # input pixels per output pixel
        h_in, w_in = arr.shape

        if out_size is None:
            h_out = int(np.ceil(h_in / ratio))
            w_out = int(np.ceil(w_in / ratio))
            h_out += 1 - (h_out % 2)
            w_out += 1 - (w_out % 2)
        else:
            out_size = int(out_size)
            if out_size <= 0 or out_size % 2 == 0:
                raise ValueError(f"out_size must be a positive odd integer, got {out_size}.")
            h_out = w_out = out_size

        cy_in, cx_in = (h_in - 1) / 2.0, (w_in - 1) / 2.0
        cy_out, cx_out = (h_out - 1) / 2.0, (w_out - 1) / 2.0

        rows = (np.arange(h_out) - cy_out) * ratio + cy_in
        cols = (np.arange(w_out) - cx_out) * ratio + cx_in
        rr, cc = np.meshgrid(rows, cols, indexing="ij")

        resampled = map_coordinates(
            arr, np.stack([rr.ravel(), cc.ravel()]), order=3, mode="constant", cval=0.0
        ).reshape(h_out, w_out)

        # Surface-brightness -> per-pixel flux: each output pixel covers
        # ratio**2 input pixels.
        resampled *= ratio**2
        total = resampled.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Resampled PSF kernel has non-positive total flux.")
        resampled /= total
        return resampled.astype(np.float32)

    def resample_to(self, pixel_scales, from_scale: float, out_size: int | None = None):
        """Resample every band kernel to a (possibly per-band) pixel scale.

        Args:
            pixel_scales: Either a single target pixel scale in arcsec/pixel or
                a mapping from band name to target pixel scale.
            from_scale: Pixel scale of the stored kernels in arcsec/pixel.
            out_size: Optional odd output size, applied to every band.

        Returns:
            Dict mapping band name to the resampled kernel.  A dict rather than a
            PSFModel because different bands generally end up on different grids.
        """
        if isinstance(pixel_scales, dict):
            targets = {b: float(pixel_scales[b]) for b in self.band_names}
        else:
            targets = {b: float(pixel_scales) for b in self.band_names}
        return {
            band: PSFModel.resample(
                np.asarray(self.kernels)[i], from_scale, targets[band], out_size=out_size
            )
            for i, band in enumerate(self.band_names)
        }

    @property
    def n_bands(self) -> int:
        """Number of PSF bands."""
        return self.kernels.shape[0]
