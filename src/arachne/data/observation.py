"""Multi-band FITS observation container."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


@dataclass
class ObservationCube:
    """Multi-band image data container for a single galaxy target.

    Holds flux, variance, and mask arrays for all photometric bands, loaded
    from FITS files. All arrays are stored as JAX arrays after calling
    `to_jax()`.

    Attributes:
        flux: Flux array of shape (N_bands, H, W) in nJy.
        variance: Variance array of shape (N_bands, H, W) in nJy^2.
        mask: Boolean mask of shape (N_bands, H, W). True = valid pixel.
        band_names: List of band identifiers, e.g. ["JWST/NIRCam.F115W", ...].
        pixel_scale: Pixel scale in arcsec/pixel.
        wcs: Astropy WCS object from the first band's FITS header.
    """

    flux: np.ndarray | jnp.ndarray
    variance: np.ndarray | jnp.ndarray
    mask: np.ndarray | jnp.ndarray
    band_names: list[str]
    pixel_scale: float
    wcs: Optional[WCS] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Validate array shapes are consistent."""
        if self.flux.shape != self.variance.shape:
            raise ValueError(
                f"flux shape {self.flux.shape} != variance shape {self.variance.shape}"
            )
        if self.flux.shape != self.mask.shape:
            raise ValueError(f"flux shape {self.flux.shape} != mask shape {self.mask.shape}")
        n_bands = self.flux.shape[0]
        if len(self.band_names) != n_bands:
            raise ValueError(
                f"Got {len(self.band_names)} band names but {n_bands} bands in flux array."
            )

    @classmethod
    def from_fits(
        cls,
        flux_paths: list[str | Path],
        variance_paths: list[str | Path],
        band_names: list[str],
        mask_paths: Optional[list[str | Path]] = None,
        cutout_center: Optional[tuple[float, float]] = None,
        cutout_size: Optional[int | tuple[int, int]] = None,
        pixel_scale: float = 0.031,
    ) -> "ObservationCube":
        """Load an ObservationCube from FITS files.

        Args:
            flux_paths: List of paths to flux FITS files, one per band.
            variance_paths: List of paths to variance FITS files, one per band.
            band_names: List of band name strings in the same order as flux_paths.
            mask_paths: Optional list of paths to mask FITS files. If None, all
                pixels are assumed valid (mask = True everywhere).
            cutout_center: Optional (RA, Dec) in degrees for a spatial cutout.
            cutout_size: Optional cutout size in pixels. An integer gives a square
                cutout; a tuple (H, W) gives a rectangular one.
            pixel_scale: Pixel scale in arcsec/pixel. Defaults to 0.031 (JWST NIRCam).

        Returns:
            Populated ObservationCube with float32 numpy arrays.

        Raises:
            ValueError: If the number of paths does not match band_names length.
        """
        if len(flux_paths) != len(band_names):
            raise ValueError("flux_paths and band_names must have the same length.")
        if len(variance_paths) != len(band_names):
            raise ValueError("variance_paths and band_names must have the same length.")
        if mask_paths is not None and len(mask_paths) != len(band_names):
            raise ValueError("mask_paths and band_names must have the same length.")

        flux_list = []
        var_list = []
        mask_list = []
        wcs_ref = None

        for i, (fp, vp) in enumerate(zip(flux_paths, variance_paths)):
            with fits.open(fp) as hdul:
                flux_data = hdul[0].data.astype(np.float32)
                header = hdul[0].header
                if i == 0:
                    try:
                        wcs_ref = WCS(header)
                    except Exception:
                        wcs_ref = None

            with fits.open(vp) as hdul:
                var_data = hdul[0].data.astype(np.float32)

            if mask_paths is not None:
                with fits.open(mask_paths[i]) as hdul:
                    mask_data = hdul[0].data.astype(bool)
            else:
                mask_data = np.ones(flux_data.shape, dtype=bool)

            if cutout_center is not None and cutout_size is not None:
                if wcs_ref is not None and i == 0:
                    import astropy.units as u
                    from astropy.coordinates import SkyCoord

                    coord = SkyCoord(ra=cutout_center[0] * u.deg, dec=cutout_center[1] * u.deg)
                    cutout = Cutout2D(flux_data, coord, cutout_size, wcs=wcs_ref)
                    wcs_ref = cutout.wcs
                    flux_data = cutout.data
                    # Re-cut variance and mask to same region
                    var_cutout = Cutout2D(var_data, coord, cutout_size)
                    var_data = var_cutout.data
                    mask_cutout = Cutout2D(mask_data.astype(np.float32), coord, cutout_size)
                    mask_data = mask_cutout.data.astype(bool)
                else:
                    # pixel-space cutout
                    cy, cx = cutout_center
                    if isinstance(cutout_size, int):
                        hs = cutout_size // 2
                        flux_data = flux_data[cy - hs : cy + hs, cx - hs : cx + hs]
                        var_data = var_data[cy - hs : cy + hs, cx - hs : cx + hs]
                        mask_data = mask_data[cy - hs : cy + hs, cx - hs : cx + hs]

            flux_list.append(flux_data)
            var_list.append(var_data)
            mask_list.append(mask_data)

        flux = np.stack(flux_list, axis=0)
        variance = np.stack(var_list, axis=0)
        mask = np.stack(mask_list, axis=0)

        logger.info(
            f"Loaded ObservationCube: {len(band_names)} bands, image shape {flux.shape[1:]}"
        )
        return cls(
            flux=flux,
            variance=variance,
            mask=mask,
            band_names=band_names,
            pixel_scale=pixel_scale,
            wcs=wcs_ref,
        )

    def to_jax(self) -> "ObservationCube":
        """Convert all arrays to JAX arrays (float32).

        Should be called once before inference. After this call, all array
        operations will be performed on the JAX device.

        Returns:
            New ObservationCube with JAX arrays.
        """
        return ObservationCube(
            flux=jnp.array(self.flux, dtype=jnp.float32),
            variance=jnp.array(self.variance, dtype=jnp.float32),
            mask=jnp.array(self.mask, dtype=jnp.float32),
            band_names=self.band_names,
            pixel_scale=self.pixel_scale,
            wcs=self.wcs,
        )

    @property
    def n_bands(self) -> int:
        """Number of photometric bands."""
        return self.flux.shape[0]

    @property
    def image_shape(self) -> tuple[int, int]:
        """Spatial dimensions (H, W) of the image."""
        return self.flux.shape[1], self.flux.shape[2]
