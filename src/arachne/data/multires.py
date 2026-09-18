"""Multi-resolution multi-band observation containers.

:class:`~arachne.data.observation.ObservationCube` assumes every band shares a
single pixel grid.  Real JWST data do not: NIRCam short-wavelength mosaics are
natively 0.02-0.03 arcsec/pixel while long-wavelength mosaics are 0.04-0.06,
and drizzling everything onto one grid either throws away SW resolution or
interpolates LW data (correlating its noise).

:class:`MultiResolutionObservation` keeps every band on its own grid and
records, for each band, the affine map from that band's pixel indices to
tangent-plane offsets in arcsec from one common reference sky position.  A
forward model can then evaluate analytic component profiles directly on each
band's own sampling, convolve with that band's PSF at that band's pixel scale,
and compare with that band's data -- no resampling of the data at all.

Sign convention
---------------
``(dy, dx)`` are offsets on the tangent plane about the reference position,
``dy`` positive towards North (increasing Dec) and ``dx`` positive towards East
(increasing RA).  ``affine`` maps ``(row - ref_row, col - ref_col)`` to
``(dy, dx)``; note that for the usual "North up, East left" orientation
``affine[1, 1]`` (d dx / d col) is *negative*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from arachne.data import units as flux_units
from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

PathLike = Union[str, Path]
#: A FITS source: a path, or a ``(path, extension)`` pair.
FitsSource = Union[PathLike, tuple[PathLike, Union[int, str]]]

# MIRI imaging filters, to disambiguate JWST instrument from a bare filter name.
_MIRI_FILTERS = {
    "F560W",
    "F770W",
    "F1000W",
    "F1065C",
    "F1130W",
    "F1140C",
    "F1280W",
    "F1550C",
    "F1500W",
    "F1800W",
    "F2100W",
    "F2300C",
    "F2550W",
}


def canonical_band_name(filter_name: str) -> str:
    """Map a DJA/FITS filter string to arachne's canonical band name.

    ``'F200W-CLEAR'`` and ``'f200w-clear'`` both become
    ``'JWST/NIRCam.F200W'``; MIRI filters become ``'JWST/MIRI.F770W'``.  A name
    that is already canonical (contains ``'/'``) is returned unchanged, and an
    unrecognised filter is returned uppercased with any pupil suffix stripped.

    Args:
        filter_name: Filter string such as ``'F444W-CLEAR'``.

    Returns:
        Canonical band name.
    """
    name = str(filter_name).strip()
    if "/" in name:
        return name
    base = name.upper().split("-")[0].split("_")[0]
    if base in _MIRI_FILTERS:
        return f"JWST/MIRI.{base}"
    if re.fullmatch(r"F\d{3}[WMN]P?", base) or re.fullmatch(r"F\d{3}[WMN]", base):
        return f"JWST/NIRCam.{base}"
    return base


def tangent_plane_affine(
    wcs: WCS,
    ref_ra: float,
    ref_dec: float,
    step: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Build the local pixel-to-tangent-plane affine map for a WCS.

    The Jacobian is evaluated numerically from ``wcs.pixel_to_world`` at the
    reference pixel plus and minus ``step`` pixels along each axis, with the
    offsets measured by :meth:`astropy.coordinates.SkyCoord.spherical_offsets_to`
    so that the cos(Dec) factor and the projection are both handled exactly.

    Args:
        wcs: Celestial WCS of the image.
        ref_ra: Reference right ascension in degrees.
        ref_dec: Reference declination in degrees.
        step: Finite-difference step in pixels.

    Returns:
        Tuple ``(affine, ref_pixel)``.  ``affine`` is the (2, 2) matrix mapping
        ``(row - ref_row, col - ref_col)`` to ``(dy, dx)`` in arcsec, and
        ``ref_pixel`` is the fractional ``(row, col)`` of the reference position.
    """
    ref = SkyCoord(ra=ref_ra * u.deg, dec=ref_dec * u.deg)
    x0, y0 = wcs.world_to_pixel(ref)
    x0, y0 = float(x0), float(y0)

    def _offsets(x: float, y: float) -> np.ndarray:
        coord = wcs.pixel_to_world(x, y)
        dlon, dlat = ref.spherical_offsets_to(coord)
        return np.array([dlat.to_value(u.arcsec), dlon.to_value(u.arcsec)])

    d_drow = (_offsets(x0, y0 + step) - _offsets(x0, y0 - step)) / (2.0 * step)
    d_dcol = (_offsets(x0 + step, y0) - _offsets(x0 - step, y0)) / (2.0 * step)
    affine = np.column_stack([d_drow, d_dcol]).astype(np.float64)
    return affine, (y0, x0)


def pixel_scale_from_affine(affine: np.ndarray) -> float:
    """Return the geometric-mean pixel scale implied by an affine matrix.

    Args:
        affine: (2, 2) pixel-to-arcsec matrix.

    Returns:
        Pixel scale in arcsec/pixel, ``sqrt(|det affine|)``.
    """
    return float(np.sqrt(np.abs(np.linalg.det(np.asarray(affine, dtype=np.float64)))))


@dataclass
class BandImage:
    """One band's image on its own pixel grid.

    Attributes:
        band_name: Canonical band name, e.g. ``'JWST/NIRCam.F200W'``.
        flux: (H_b, W_b) image in nJy.
        variance: (H_b, W_b) variance in nJy^2.  ``inf`` marks unusable pixels.
        mask: (H_b, W_b) validity mask, 1.0 valid and 0.0 invalid.
        pixel_scale: Pixel scale in arcsec/pixel (geometric mean of the CD matrix).
        affine: (2, 2) matrix mapping ``(row - ref_row, col - ref_col)`` to
            ``(dy, dx)`` arcsec on the tangent plane about the reference sky
            position, with ``dy`` towards North and ``dx`` towards East.
        ref_pixel: Fractional ``(row, col)`` of the common reference RA/Dec in
            this band's pixel grid.
        psf: PSF kernel sampled on *this* band's pixel grid, normalised to sum
            1, or None.
        wcs: The band's astropy WCS, if available.
    """

    band_name: str
    flux: np.ndarray | jnp.ndarray
    variance: np.ndarray | jnp.ndarray
    mask: np.ndarray | jnp.ndarray
    pixel_scale: float
    affine: np.ndarray
    ref_pixel: tuple[float, float]
    psf: Optional[np.ndarray | jnp.ndarray] = None
    wcs: Optional[WCS] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Validate array shapes and the affine matrix."""
        if self.flux.ndim != 2:
            raise ValueError(f"{self.band_name}: flux must be 2-D, got shape {self.flux.shape}.")
        if self.variance.shape != self.flux.shape:
            raise ValueError(
                f"{self.band_name}: flux shape {self.flux.shape} != variance shape "
                f"{self.variance.shape}"
            )
        if self.mask.shape != self.flux.shape:
            raise ValueError(
                f"{self.band_name}: flux shape {self.flux.shape} != mask shape {self.mask.shape}"
            )
        affine = np.asarray(self.affine, dtype=np.float64)
        if affine.shape != (2, 2):
            raise ValueError(f"{self.band_name}: affine must be (2, 2), got {affine.shape}.")
        self.affine = affine
        self.ref_pixel = (float(self.ref_pixel[0]), float(self.ref_pixel[1]))

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial shape (H_b, W_b) of this band."""
        return int(self.flux.shape[0]), int(self.flux.shape[1])

    def sky_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Tangent-plane offsets of every pixel centre from the reference position.

        Returns:
            Tuple ``(yy, xx)`` of flattened (H_b * W_b,) arrays in arcsec, with
            ``yy`` towards North and ``xx`` towards East, in C (row-major) order.
        """
        h, w = self.shape
        rows = np.arange(h, dtype=np.float64) - self.ref_pixel[0]
        cols = np.arange(w, dtype=np.float64) - self.ref_pixel[1]
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        a = self.affine
        yy = a[0, 0] * rr + a[0, 1] * cc
        xx = a[1, 0] * rr + a[1, 1] * cc
        return yy.ravel(), xx.ravel()

    def to_jax(self) -> "BandImage":
        """Return a copy with flux, variance, mask and PSF as JAX float32 arrays.

        Returns:
            New BandImage backed by JAX arrays.
        """
        return BandImage(
            band_name=self.band_name,
            flux=jnp.asarray(self.flux, dtype=jnp.float32),
            variance=jnp.asarray(self.variance, dtype=jnp.float32),
            mask=jnp.asarray(self.mask, dtype=jnp.float32),
            pixel_scale=self.pixel_scale,
            affine=self.affine,
            ref_pixel=self.ref_pixel,
            psf=None if self.psf is None else jnp.asarray(self.psf, dtype=jnp.float32),
            wcs=self.wcs,
        )


@dataclass
class MultiResolutionObservation:
    """A target observed in several bands, each on its own pixel grid.

    Attributes:
        bands: List of :class:`BandImage`, one per band.
        ref_ra: Reference right ascension in degrees, shared by all bands.
        ref_dec: Reference declination in degrees, shared by all bands.
    """

    bands: list[BandImage]
    ref_ra: float
    ref_dec: float

    def __post_init__(self) -> None:
        """Validate that at least one band is present and names are unique."""
        if not self.bands:
            raise ValueError("MultiResolutionObservation requires at least one band.")
        names = [b.band_name for b in self.bands]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate band names: {names}")

    @property
    def band_names(self) -> list[str]:
        """Band names in stored order."""
        return [b.band_name for b in self.bands]

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return len(self.bands)

    @property
    def pixel_scales(self) -> list[float]:
        """Per-band pixel scales in arcsec/pixel."""
        return [b.pixel_scale for b in self.bands]

    @property
    def shapes(self) -> list[tuple[int, int]]:
        """Per-band image shapes."""
        return [b.shape for b in self.bands]

    def __len__(self) -> int:
        """Number of bands."""
        return len(self.bands)

    def __iter__(self):
        """Iterate over the BandImages."""
        return iter(self.bands)

    def __getitem__(self, key: int | str) -> BandImage:
        """Get a band by index or by band name.

        Args:
            key: Integer index or band name.

        Returns:
            The matching BandImage.

        Raises:
            KeyError: If a band name is not present.
        """
        if isinstance(key, str):
            for band in self.bands:
                if band.band_name == key:
                    return band
            raise KeyError(f"No band named {key!r}; have {self.band_names}.")
        return self.bands[key]

    def to_jax(self) -> "MultiResolutionObservation":
        """Return a copy with every band's arrays as JAX float32 arrays.

        Returns:
            New MultiResolutionObservation backed by JAX arrays.
        """
        return MultiResolutionObservation(
            bands=[b.to_jax() for b in self.bands],
            ref_ra=self.ref_ra,
            ref_dec=self.ref_dec,
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_fits(
        cls,
        flux_paths: dict[str, FitsSource],
        ref_ra: float,
        ref_dec: float,
        size_arcsec: float,
        variance_paths: Optional[dict[str, FitsSource]] = None,
        weight_paths: Optional[dict[str, FitsSource]] = None,
        flux_unit: str = "auto",
        psfs: Optional[dict[str, tuple[np.ndarray, float]]] = None,
        zeropoints: Optional[dict[str, float]] = None,
    ) -> "MultiResolutionObservation":
        """Build a multi-resolution observation from arbitrary per-band FITS files.

        Every band is cut out independently about the same sky position with
        :class:`astropy.nddata.Cutout2D`, so bands at different native pixel
        scales simply end up with different cutout shapes -- which is the point.

        Args:
            flux_paths: Mapping band name -> path, or ``(path, extension)``.
            ref_ra: Reference right ascension in degrees.
            ref_dec: Reference declination in degrees.
            size_arcsec: Full width (and height) of the cutout in arcsec.
            variance_paths: Optional mapping band name -> variance FITS source,
                in (flux unit)^2.
            weight_paths: Optional mapping band name -> inverse-variance weight
                FITS source.  Used only for bands absent from ``variance_paths``.
            flux_unit: ``"auto"`` reads BUNIT (and, failing that, an AB ``ZP``
                keyword) from each band's header.  Any other string is used as
                the BUNIT for every band.
            psfs: Optional mapping band name -> ``(kernel, psf_pixel_scale)``.
                Each kernel is resampled to that band's pixel scale with
                :meth:`arachne.data.psf.PSFModel.resample`.
            zeropoints: Optional mapping band name -> AB zeropoint, overriding
                the header for those bands.

        Returns:
            Populated MultiResolutionObservation with float32 numpy arrays.

        Raises:
            ValueError: If a band has no celestial WCS or its units cannot be
                resolved.
        """
        bands: list[BandImage] = []
        for band_name, source in flux_paths.items():
            data, header = _read_fits(source)
            wcs = _celestial_wcs(header)
            if wcs is None:
                raise ValueError(f"{band_name}: FITS header has no celestial WCS.")

            cutout = _cutout(data, wcs, ref_ra, ref_dec, size_arcsec)
            flux_cut, wcs_cut, slices = cutout.data, cutout.wcs, cutout.slices_original

            affine, ref_pixel = tangent_plane_affine(wcs_cut, ref_ra, ref_dec)
            pixel_scale = pixel_scale_from_affine(affine)

            bunit = header.get("BUNIT") if flux_unit == "auto" else flux_unit
            zp = None if zeropoints is None else zeropoints.get(band_name)
            scale = _resolve_scale(band_name, bunit, header, zp, pixel_scale, flux_unit)

            variance, mask = _variance_and_mask(
                band_name,
                variance_paths,
                weight_paths,
                slices,
                flux_cut.shape,
                scale,
            )

            psf = None
            if psfs is not None and band_name in psfs:
                kernel, psf_scale = psfs[band_name]
                psf = PSFModel.resample(kernel, float(psf_scale), pixel_scale)

            bands.append(
                BandImage(
                    band_name=band_name,
                    flux=(flux_cut * scale).astype(np.float32),
                    variance=variance,
                    mask=mask,
                    pixel_scale=pixel_scale,
                    affine=affine,
                    ref_pixel=ref_pixel,
                    psf=psf,
                    wcs=wcs_cut,
                )
            )
            logger.info(
                f"{band_name}: {flux_cut.shape} at {pixel_scale:.4f} arcsec/px "
                f"(unit scale {scale:.4g} nJy per pixel value)"
            )

        return cls(bands=bands, ref_ra=float(ref_ra), ref_dec=float(ref_dec))

    @classmethod
    def from_dja_fits(
        cls,
        path: PathLike,
        ref_ra: float,
        ref_dec: float,
        psfs: Optional[dict[str, tuple[np.ndarray, float]]] = None,
        band_name_map: Optional[dict[str, str]] = None,
        size_arcsec: Optional[float] = None,
    ) -> "MultiResolutionObservation":
        """Parse a DJA cutout-server multi-extension FITS into BandImages.

        The DJA ``thumb?...&output=fits_weight`` product is a multi-extension
        FITS with one pair of extensions per filter: ``EXTNAME`` is the filter
        (e.g. ``'F200W-CLEAR'``) and ``EXTVER`` is the string ``'SCI'`` or
        ``'WHT'`` (the latter an inverse-variance map).  Pixels are in the unit
        given by ``BUNIT`` (``'10.0*nanoJansky'`` in practice).

        The ``ZP`` card in these files is deliberately ignored: it reproduces
        ``-2.5*log10(OPHOTFNU) + 8.9``, the zeropoint of the *original* mosaic
        before the server rescaled the pixels, and disagrees with BUNIT by a
        factor of a few.  See :mod:`arachne.data.units`.

        Args:
            path: Path to the downloaded multi-extension FITS.
            ref_ra: Reference right ascension in degrees.
            ref_dec: Reference declination in degrees.
            psfs: Optional mapping band name -> ``(kernel, psf_pixel_scale)``.
            band_name_map: Optional explicit mapping from EXTNAME to band name.
                Defaults to :func:`canonical_band_name`.
            size_arcsec: Optional full cutout width in arcsec.  By default the
                extensions are used at their delivered size.

        Returns:
            Populated MultiResolutionObservation.

        Raises:
            ValueError: If the file contains no usable SCI extensions.
        """
        bands: list[BandImage] = []
        with fits.open(path) as hdul:
            sci, wht = _group_dja_extensions(hdul)
            if not sci:
                raise ValueError(f"No SCI extensions found in DJA file {path}.")

            for extname, hdu in sci.items():
                band_name = (
                    band_name_map[extname]
                    if band_name_map is not None and extname in band_name_map
                    else canonical_band_name(extname)
                )
                header = hdu.header
                data = np.asarray(hdu.data, dtype=np.float64)
                wcs = _celestial_wcs(header)
                if wcs is None:
                    raise ValueError(f"{extname}: DJA extension has no celestial WCS.")

                wht_data = None
                if extname in wht and wht[extname].data is not None:
                    wht_data = np.asarray(wht[extname].data, dtype=np.float64)

                if size_arcsec is not None:
                    cut = _cutout(data, wcs, ref_ra, ref_dec, size_arcsec)
                    data, wcs_cut = cut.data, cut.wcs
                    if wht_data is not None:
                        wht_data = wht_data[cut.slices_original]
                else:
                    wcs_cut = wcs

                affine, ref_pixel = tangent_plane_affine(wcs_cut, ref_ra, ref_dec)
                pixel_scale = pixel_scale_from_affine(affine)
                scale = flux_units.flux_scale_to_nJy(
                    bunit=header.get("BUNIT"), pixel_area_arcsec2=pixel_scale**2
                )

                if wht_data is not None:
                    # WHT is inverse variance in (pixel value)^-2.
                    variance, mask = flux_units.weight_to_variance(wht_data)
                    variance = (variance.astype(np.float64) * scale**2).astype(np.float32)
                else:
                    variance = np.ones(data.shape, dtype=np.float32)
                    mask = np.ones(data.shape, dtype=np.float32)

                psf = None
                if psfs is not None and band_name in psfs:
                    kernel, psf_scale = psfs[band_name]
                    psf = PSFModel.resample(kernel, float(psf_scale), pixel_scale)

                bands.append(
                    BandImage(
                        band_name=band_name,
                        flux=(data * scale).astype(np.float32),
                        variance=variance,
                        mask=mask,
                        pixel_scale=pixel_scale,
                        affine=affine,
                        ref_pixel=ref_pixel,
                        psf=psf,
                        wcs=wcs_cut,
                    )
                )

        logger.info(
            f"Loaded DJA cutout {Path(path).name}: bands "
            f"{[b.band_name for b in bands]} at scales "
            f"{[round(b.pixel_scale, 4) for b in bands]} arcsec/px"
        )
        return cls(bands=bands, ref_ra=float(ref_ra), ref_dec=float(ref_dec))

    # ------------------------------------------------------------------
    # Bridge to the single-grid path
    # ------------------------------------------------------------------

    def to_observation_cube(self, atol: float = 1e-6) -> ObservationCube:
        """Collapse to a single-grid :class:`ObservationCube`.

        Only valid when every band shares one pixel grid: the same shape, the
        same affine matrix and the same reference pixel.

        Args:
            atol: Absolute tolerance in arcsec/pixel on the affine elements and
                in pixels on the reference pixel.

        Returns:
            ObservationCube stacking every band.

        Raises:
            ValueError: If the bands are not on a common grid.
        """
        first = self.bands[0]
        for band in self.bands[1:]:
            if band.shape != first.shape:
                raise ValueError(
                    "Bands are not on a common grid: "
                    f"{first.band_name} has shape {first.shape} but {band.band_name} has "
                    f"{band.shape}.  Use the multi-resolution forward model instead."
                )
            if not np.allclose(band.affine, first.affine, atol=atol, rtol=0.0):
                raise ValueError(
                    "Bands are not on a common grid: affine matrices differ between "
                    f"{first.band_name} and {band.band_name}."
                )
            if not np.allclose(band.ref_pixel, first.ref_pixel, atol=max(atol, 1e-3), rtol=0.0):
                raise ValueError(
                    "Bands are not on a common grid: reference pixel differs between "
                    f"{first.band_name} ({first.ref_pixel}) and {band.band_name} "
                    f"({band.ref_pixel})."
                )

        return ObservationCube(
            flux=np.stack([np.asarray(b.flux) for b in self.bands], axis=0).astype(np.float32),
            variance=np.stack([np.asarray(b.variance) for b in self.bands], axis=0).astype(
                np.float32
            ),
            mask=np.stack([np.asarray(b.mask) for b in self.bands], axis=0).astype(np.float32),
            band_names=self.band_names,
            pixel_scale=first.pixel_scale,
            wcs=first.wcs,
            flux_unit="nJy",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_fits(source: FitsSource) -> tuple[np.ndarray, fits.Header]:
    """Read data and header from a path or ``(path, extension)`` pair.

    Args:
        source: Path, or ``(path, extension)``.

    Returns:
        Tuple ``(data, header)`` with the data as float64.

    Raises:
        ValueError: If the chosen extension has no image data.
    """
    if isinstance(source, tuple):
        path, ext = source
    else:
        path, ext = source, None
    with fits.open(path) as hdul:
        hdu = hdul[0] if ext is None else hdul[ext]
        if ext is None and hdu.data is None:
            hdu = hdul[1]
        if hdu.data is None:
            raise ValueError(f"{path}: extension {ext!r} contains no image data.")
        return np.asarray(hdu.data, dtype=np.float64), hdu.header.copy()


def _celestial_wcs(header: fits.Header) -> Optional[WCS]:
    """Return the celestial WCS of a header, or None.

    Args:
        header: FITS header.

    Returns:
        WCS with celestial axes, or None if absent or unparseable.
    """
    try:
        wcs = WCS(header)
    except Exception:  # pragma: no cover - malformed headers
        return None
    return wcs if wcs.has_celestial else None


def _cutout(
    data: np.ndarray, wcs: WCS, ref_ra: float, ref_dec: float, size_arcsec: float
) -> Cutout2D:
    """Cut a square region of a given angular size about a sky position.

    Args:
        data: Full image.
        wcs: Celestial WCS of ``data``.
        ref_ra: Right ascension in degrees.
        ref_dec: Declination in degrees.
        size_arcsec: Full width and height of the cutout in arcsec.

    Returns:
        The Cutout2D.
    """
    coord = SkyCoord(ra=ref_ra * u.deg, dec=ref_dec * u.deg)
    size = u.Quantity((size_arcsec, size_arcsec), u.arcsec)
    return Cutout2D(data, coord, size, wcs=wcs, mode="partial", fill_value=0.0)


def _resolve_scale(
    band_name: str,
    bunit: Optional[str],
    header: fits.Header,
    zeropoint: Optional[float],
    pixel_scale: float,
    flux_unit: str,
) -> float:
    """Resolve the nJy-per-pixel-value factor for one band.

    Precedence: an explicit per-band zeropoint, then BUNIT, then the header's
    ``ZP``/``MAGZERO`` card, then 1.0 (assume the data are already nJy).

    Args:
        band_name: Band name, for messages.
        bunit: BUNIT string to use (header value when ``flux_unit == "auto"``).
        header: The band's FITS header.
        zeropoint: Explicit AB zeropoint, or None.
        pixel_scale: Pixel scale in arcsec/pixel, for surface-brightness units.
        flux_unit: The caller's ``flux_unit`` argument.

    Returns:
        Conversion factor in nJy per pixel value.

    Raises:
        ValueError: If an explicit ``flux_unit`` string cannot be parsed.
    """
    if zeropoint is not None:
        return flux_units.zeropoint_scale_to_nJy(zeropoint)

    scale = flux_units.bunit_scale_to_nJy(bunit, pixel_area_arcsec2=pixel_scale**2)
    if scale is not None:
        return scale

    if flux_unit != "auto":
        raise ValueError(f"{band_name}: flux_unit={flux_unit!r} is not a recognised flux unit.")

    for key in ("ZP", "MAGZERO", "MAGZPT", "ZPAB"):
        if key in header:
            logger.warning(
                f"{band_name}: no usable BUNIT; falling back to the {key} AB zeropoint "
                f"({header[key]}).  Check this is the zeropoint of the delivered pixels."
            )
            return flux_units.zeropoint_scale_to_nJy(float(header[key]))

    logger.warning(
        f"{band_name}: no BUNIT or zeropoint keyword found; assuming the data are already in nJy."
    )
    return 1.0


def _variance_and_mask(
    band_name: str,
    variance_paths: Optional[dict[str, FitsSource]],
    weight_paths: Optional[dict[str, FitsSource]],
    slices: tuple,
    shape: tuple[int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the variance (nJy^2) and validity mask for one band.

    Args:
        band_name: Band name.
        variance_paths: Optional mapping band -> variance FITS source.
        weight_paths: Optional mapping band -> inverse-variance FITS source.
        slices: Pixel slices of the cutout in the full image.
        shape: Cutout shape.
        scale: nJy per pixel value for the flux image.

    Returns:
        Tuple ``(variance, mask)`` as float32 arrays.
    """
    if variance_paths is not None and band_name in variance_paths:
        var, _ = _read_fits(variance_paths[band_name])
        var = var[slices]
        variance = (var * scale**2).astype(np.float32)
        mask = (np.isfinite(variance) & (variance > 0)).astype(np.float32)
        variance = np.where(mask > 0, variance, np.inf).astype(np.float32)
        return variance, mask

    if weight_paths is not None and band_name in weight_paths:
        wht, _ = _read_fits(weight_paths[band_name])
        variance, mask = flux_units.weight_to_variance(wht[slices])
        variance = (variance.astype(np.float64) * scale**2).astype(np.float32)
        return variance, mask

    return np.ones(shape, dtype=np.float32), np.ones(shape, dtype=np.float32)


def _group_dja_extensions(hdul: fits.HDUList) -> tuple[dict, dict]:
    """Group a DJA multi-extension FITS into SCI and WHT extensions by filter.

    Args:
        hdul: Open HDUList.

    Returns:
        Tuple ``(sci, wht)`` of dicts keyed by EXTNAME (the filter string).
    """
    sci: dict[str, fits.hdu.base.ExtensionHDU] = {}
    wht: dict[str, fits.hdu.base.ExtensionHDU] = {}
    for hdu in hdul:
        if hdu.data is None:
            continue
        extname = hdu.header.get("EXTNAME")
        if extname is None:
            continue
        kind = str(hdu.header.get("EXTVER", "SCI")).strip().upper()
        if kind.startswith("WHT"):
            wht[str(extname)] = hdu
        else:
            sci[str(extname)] = hdu
    return sci, wht
