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

from arachne.data import units as flux_units
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
        flux_unit: Informational label for the units the arrays were loaded
            from.  The stored arrays are always nJy (and nJy^2 for variance)
            after construction; this records what they were converted *from*.
    """

    flux: np.ndarray | jnp.ndarray
    variance: np.ndarray | jnp.ndarray
    mask: np.ndarray | jnp.ndarray
    band_names: list[str]
    pixel_scale: float
    wcs: Optional[WCS] = field(default=None, compare=False)
    flux_unit: str = "nJy"

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
        flux_unit: str = "auto",
        zeropoints: Optional[dict[str, float] | list[float]] = None,
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
            flux_unit: Unit handling. ``"auto"`` reads each band's ``BUNIT``
                keyword (falling back to an AB ``ZP``/``MAGZERO`` card, and
                finally to "already nJy") and converts flux to nJy and variance
                to nJy^2. Any other string, e.g. ``"uJy"`` or ``"MJy/sr"``, is
                used as the BUNIT for every band. ``"nJy"`` is a no-op.
            zeropoints: Optional AB zeropoints overriding the header, either a
                dict keyed by band name or a list in band order. A zeropoint
                takes precedence over BUNIT for that band.

        Returns:
            Populated ObservationCube with float32 numpy arrays in nJy.

        Raises:
            ValueError: If the number of paths does not match band_names length,
                or if an explicit ``flux_unit`` cannot be parsed.
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
        _wcs_cutout_slices: Optional[tuple] = None  # pixel slices from band-0 WCS cutout

        scales: list[float] = []
        for i, (fp, vp) in enumerate(zip(flux_paths, variance_paths)):
            with fits.open(fp) as hdul:
                flux_data = hdul[0].data.astype(np.float32)
                header = hdul[0].header.copy()
                if i == 0:
                    try:
                        candidate = WCS(header)
                        wcs_ref = candidate if candidate.has_celestial else None
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
                if i == 0 and wcs_ref is not None:
                    # Band 0 with WCS: compute cutout position and save pixel slices
                    import astropy.units as u
                    from astropy.coordinates import SkyCoord

                    coord = SkyCoord(ra=cutout_center[0] * u.deg, dec=cutout_center[1] * u.deg)
                    cutout = Cutout2D(flux_data, coord, cutout_size, wcs=wcs_ref)
                    _wcs_cutout_slices = cutout.slices_original
                    wcs_ref = cutout.wcs
                    flux_data = cutout.data
                    var_data = var_data[_wcs_cutout_slices]
                    mask_data = mask_data[_wcs_cutout_slices]
                elif _wcs_cutout_slices is not None:
                    # Bands 1+ with WCS: reuse the pixel slices determined from band 0
                    flux_data = flux_data[_wcs_cutout_slices]
                    var_data = var_data[_wcs_cutout_slices]
                    mask_data = mask_data[_wcs_cutout_slices]
                else:
                    # Pure pixel-space cutout (no WCS available)
                    cy, cx = cutout_center
                    if isinstance(cutout_size, int):
                        hs = cutout_size // 2
                        flux_data = flux_data[cy - hs : cy + hs, cx - hs : cx + hs]
                        var_data = var_data[cy - hs : cy + hs, cx - hs : cx + hs]
                        mask_data = mask_data[cy - hs : cy + hs, cx - hs : cx + hs]

            scale = cls._unit_scale(
                band_names[i], header, flux_unit, zeropoints, i, pixel_scale, wcs_ref
            )
            scales.append(scale)
            if scale != 1.0:
                flux_data = (flux_data * scale).astype(np.float32)
                var_data = (var_data * scale**2).astype(np.float32)

            flux_list.append(flux_data)
            var_list.append(var_data)
            mask_list.append(mask_data)

        flux = np.stack(flux_list, axis=0)
        variance = np.stack(var_list, axis=0)
        mask = np.stack(mask_list, axis=0)

        logger.info(
            f"Loaded ObservationCube: {len(band_names)} bands, image shape {flux.shape[1:]}, "
            f"unit scales to nJy {[float(f'{s:.6g}') for s in scales]}"
        )
        return cls(
            flux=flux,
            variance=variance,
            mask=mask,
            band_names=band_names,
            pixel_scale=pixel_scale,
            wcs=wcs_ref,
            flux_unit="nJy",
        )

    @staticmethod
    def _unit_scale(
        band_name: str,
        header,
        flux_unit: str,
        zeropoints: Optional[dict[str, float] | list[float]],
        index: int,
        pixel_scale: float,
        wcs_ref: Optional[WCS],
    ) -> float:
        """Resolve the nJy-per-pixel-value factor for one band of ``from_fits``.

        Args:
            band_name: Band name, used to look up ``zeropoints`` and for messages.
            header: The band's FITS header.
            flux_unit: ``"auto"`` or an explicit BUNIT string.
            zeropoints: Optional dict or list of AB zeropoints.
            index: Band index, used when ``zeropoints`` is a list.
            pixel_scale: Fallback pixel scale in arcsec/pixel.
            wcs_ref: WCS of the first band, used to refine the pixel area.

        Returns:
            Conversion factor in nJy per pixel value.

        Raises:
            ValueError: If an explicit ``flux_unit`` cannot be parsed.
        """
        zp: Optional[float] = None
        if isinstance(zeropoints, dict):
            zp = zeropoints.get(band_name)
        elif zeropoints is not None:
            zp = zeropoints[index]
        if zp is not None:
            return flux_units.zeropoint_scale_to_nJy(float(zp))

        area = float(pixel_scale) ** 2
        if wcs_ref is not None:
            try:
                area = float(abs(np.linalg.det(wcs_ref.pixel_scale_matrix))) * 3600.0**2
            except Exception:  # pragma: no cover - degenerate WCS
                pass

        bunit = header.get("BUNIT") if flux_unit == "auto" else flux_unit
        scale = flux_units.bunit_scale_to_nJy(bunit, pixel_area_arcsec2=area)
        if scale is not None:
            return scale

        if flux_unit != "auto":
            raise ValueError(f"{band_name}: flux_unit={flux_unit!r} is not a recognised flux unit.")

        for key in ("ZP", "MAGZERO", "MAGZPT", "ZPAB"):
            if key in header:
                logger.warning(
                    f"{band_name}: no usable BUNIT; falling back to the {key} AB zeropoint "
                    f"({header[key]})."
                )
                return flux_units.zeropoint_scale_to_nJy(float(header[key]))

        logger.debug(f"{band_name}: no BUNIT or zeropoint found; assuming data are already nJy.")
        return 1.0

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
            flux_unit=self.flux_unit,
        )

    @property
    def n_bands(self) -> int:
        """Number of photometric bands."""
        return self.flux.shape[0]

    @property
    def image_shape(self) -> tuple[int, int]:
        """Spatial dimensions (H, W) of the image."""
        return self.flux.shape[1], self.flux.shape[2]
