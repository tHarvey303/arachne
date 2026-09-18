"""Flux-unit parsing and conversion to nanoJansky.

``arachne`` stores every image in nanoJansky (nJy) and every variance map in
nJy^2.  Real JWST/HST products arrive in a zoo of conventions: the DAWN JWST
Archive (DJA) writes ``BUNIT = '10.0*nanoJansky'``, the JWST calibration
pipeline writes ``MJy/sr``, many older mosaics carry only an AB zeropoint
keyword, and detector-level products are in ``electron/s``.  This module
centralises the parsing and the arithmetic.

Conventions
-----------
* An AB magnitude zeropoint ``ZP`` means "a pixel value of 1 has AB magnitude
  ZP", so ``f_nJy = data * 10 ** (-0.4 * (ZP - 31.4))`` because a 1 nJy source
  has AB magnitude 31.4.
* ``parse_bunit`` returns ``(scale, kind)``.  For ``kind == "flux_density"``
  the scale converts a pixel value straight to nJy.  For
  ``kind == "surface_brightness"`` the scale converts a pixel value to nJy per
  steradian, so it must still be multiplied by the pixel solid angle.  For
  ``kind == "unknown"`` the scale is ``None``.

A note on the DJA ``ZP`` keyword
--------------------------------
DJA thumbnail cutouts carry *both* ``BUNIT = '10.0*nanoJansky'``
(equivalently ``PHOTFNU = 1e-8`` Jy/pixel-value, i.e. AB zeropoint 28.9) and a
``ZP`` card that is **not** consistent with it: ``ZP`` reproduces
``-2.5*log10(OPHOTFNU) + 8.9``, the zeropoint of the *original* mosaic before
the cutout server rescaled the pixels.  BUNIT/PHOTFNU is the authoritative
description of the delivered pixels, so ``from_fits``-style loaders in arachne
prefer BUNIT and only fall back to ``ZP`` when no usable BUNIT is present.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

#: AB magnitude of a 1 nJy source: ``-2.5 * log10(1e-9 / 3631)``.
AB_ZEROPOINT_NJY = 31.4

#: Solid angle of one square arcsecond, in steradian.
ARCSEC2_IN_SR = (np.pi / (180.0 * 3600.0)) ** 2

#: ``kind`` returned by :func:`parse_bunit` for per-pixel flux densities.
FLUX_DENSITY = "flux_density"
#: ``kind`` returned by :func:`parse_bunit` for per-solid-angle surface brightness.
SURFACE_BRIGHTNESS = "surface_brightness"
#: ``kind`` returned by :func:`parse_bunit` when the unit is not a flux unit.
UNKNOWN = "unknown"

# Jansky per named unit.  Keys are already case-normalised except for the
# mJy/MJy pair, which is resolved before this table is consulted.
_JY_PER_UNIT = {
    "jy": 1.0,
    "jansky": 1.0,
    "janskys": 1.0,
    "njy": 1e-9,
    "nanojy": 1e-9,
    "nanojansky": 1e-9,
    "nanojanskys": 1e-9,
    "ujy": 1e-6,
    "mujy": 1e-6,
    "microjy": 1e-6,
    "microjansky": 1e-6,
    "microjanskys": 1e-6,
    "millijansky": 1e-3,
    "millijanskys": 1e-3,
    "megajansky": 1e6,
    "megajanskys": 1e6,
}

_PREFIX_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*[*x×]?\s*")
_PER_SR_RE = re.compile(
    r"(?:/|\s+|\.)\s*(?:sr|steradian|steradians)\s*(?:\*\*|\^)?\s*-?1?\s*$", re.I
)
_SR_POWER_RE = re.compile(r"\s*(?:sr|steradian|steradians)\s*(?:\*\*|\^)\s*-\s*1\s*$", re.I)


def _strip_per_steradian(unit: str) -> tuple[str, bool]:
    """Split a trailing ``/sr`` (or ``sr-1``) off a unit string.

    Args:
        unit: Unit string, e.g. ``"MJy/sr"`` or ``"MJy sr**-1"``.

    Returns:
        Tuple ``(remaining_unit, per_steradian)``.
    """
    for pattern in (_SR_POWER_RE, _PER_SR_RE):
        match = pattern.search(unit)
        if match is not None:
            return unit[: match.start()].strip(), True
    return unit.strip(), False


def _jy_per_unit(unit: str) -> Optional[float]:
    """Return Jansky per one of ``unit``, or None if the unit is unrecognised.

    ``mJy`` (millijansky) and ``MJy`` (megajansky) are distinguished by case;
    everything else is matched case-insensitively.

    Args:
        unit: Bare unit string with any numeric prefix and ``/sr`` removed.

    Returns:
        Jansky per unit, or None.
    """
    unit = unit.strip().strip("'\"").strip()
    if not unit:
        return None
    # Case-sensitive milli/mega disambiguation before normalising case.
    if unit in ("MJy", "MJY"):
        return 1e6
    if unit == "mJy":
        return 1e-3
    normalised = unit.replace("µ", "u").replace("μ", "u").lower()
    normalised = normalised.replace(" ", "").replace("-", "").replace("_", "")
    if normalised == "mjy":
        # Ambiguous spelling that survived the case test (e.g. "MJY " -> handled
        # above, "mjy" here).  Treat lowercase-only spellings as millijansky,
        # which is the FITS standard meaning of "mJy".
        return 1e-3
    return _JY_PER_UNIT.get(normalised)


def parse_bunit(bunit: Optional[str]) -> tuple[Optional[float], str]:
    """Parse a FITS ``BUNIT`` string into a conversion factor to nanoJansky.

    Recognised forms include ``nJy``, ``nanoJansky``, ``10.0*nanoJansky``
    (any numeric prefix times any known unit), ``uJy``/``microJansky``,
    ``mJy``, ``Jy``, ``MJy/sr`` and ``MJy sr**-1``.  Anything else (for example
    ``electron/s`` or ``DN/s``) is reported as unknown.

    Args:
        bunit: The BUNIT string, or None.

    Returns:
        Tuple ``(scale, kind)``:

        * ``kind == "flux_density"``: ``scale`` multiplies pixel values to give
          nJy.
        * ``kind == "surface_brightness"``: ``scale`` multiplies pixel values to
          give nJy per steradian; multiply by the pixel solid angle in sr to get
          nJy per pixel.
        * ``kind == "unknown"``: ``scale`` is None.
    """
    if bunit is None:
        return None, UNKNOWN
    text = str(bunit).strip().strip("'\"").strip()
    if not text:
        return None, UNKNOWN

    prefix = 1.0
    match = _PREFIX_RE.match(text)
    if match is not None:
        remainder = text[match.end() :]
        # Only treat the leading number as a multiplicative prefix if something
        # unit-like follows it; a bare number is not a flux unit.
        if remainder.strip():
            prefix = float(match.group(1))
            text = remainder

    bare, per_sr = _strip_per_steradian(text)
    jy = _jy_per_unit(bare)
    if jy is None:
        logger.debug(f"Unrecognised BUNIT {bunit!r}; treating as unknown units.")
        return None, UNKNOWN

    # nJy (per steradian, if per_sr) per pixel value.
    scale = prefix * jy * 1e9
    return scale, SURFACE_BRIGHTNESS if per_sr else FLUX_DENSITY


def zeropoint_scale_to_nJy(zeropoint_ab: float) -> float:
    """Return the nJy corresponding to a pixel value of 1 for an AB zeropoint.

    Args:
        zeropoint_ab: AB magnitude of a source with a pixel value of 1.

    Returns:
        Conversion factor in nJy per pixel value.
    """
    return float(10.0 ** (-0.4 * (float(zeropoint_ab) - AB_ZEROPOINT_NJY)))


def bunit_scale_to_nJy(
    bunit: Optional[str],
    pixel_area_arcsec2: Optional[float] = None,
) -> Optional[float]:
    """Return the nJy-per-pixel-value factor implied by a BUNIT string.

    Args:
        bunit: BUNIT string, or None.
        pixel_area_arcsec2: Pixel solid angle in arcsec^2.  Required for
            surface-brightness units such as ``MJy/sr``.

    Returns:
        Conversion factor in nJy per pixel value, or None if the unit is
        unknown or a surface brightness was given without a pixel area.
    """
    scale, kind = parse_bunit(bunit)
    if kind == FLUX_DENSITY:
        return scale
    if kind == SURFACE_BRIGHTNESS:
        if pixel_area_arcsec2 is None:
            return None
        return float(scale) * float(pixel_area_arcsec2) * ARCSEC2_IN_SR
    return None


def flux_scale_to_nJy(
    bunit: Optional[str] = None,
    zeropoint_ab: Optional[float] = None,
    pixel_area_arcsec2: Optional[float] = None,
) -> float:
    """Resolve the scalar that converts pixel values to nJy.

    Precedence: an explicit AB zeropoint wins; otherwise the BUNIT string is
    used (with the pixel area when it describes a surface brightness).

    Args:
        bunit: BUNIT string, or None.
        zeropoint_ab: AB magnitude zeropoint, or None.
        pixel_area_arcsec2: Pixel solid angle in arcsec^2, needed for MJy/sr.

    Returns:
        Conversion factor in nJy per pixel value.

    Raises:
        ValueError: If neither input determines a conversion.
    """
    if zeropoint_ab is not None:
        return zeropoint_scale_to_nJy(zeropoint_ab)

    scale, kind = parse_bunit(bunit)
    if kind == FLUX_DENSITY:
        return float(scale)
    if kind == SURFACE_BRIGHTNESS:
        if pixel_area_arcsec2 is None:
            raise ValueError(
                f"BUNIT {bunit!r} is a surface brightness; pixel_area_arcsec2 is required "
                "to convert it to nJy per pixel."
            )
        return float(scale) * float(pixel_area_arcsec2) * ARCSEC2_IN_SR
    raise ValueError(
        f"Cannot convert to nJy: BUNIT={bunit!r} is not a recognised flux unit and no "
        "AB zeropoint was supplied."
    )


def _apply_scale(data, scale: float):
    """Multiply an array by a Python float, preserving a float32 input dtype.

    Args:
        data: Array-like input.
        scale: Multiplicative factor.

    Returns:
        Scaled numpy array.
    """
    arr = np.asarray(data)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr * float(scale)


def flux_to_nJy(
    data,
    bunit: Optional[str] = None,
    zeropoint_ab: Optional[float] = None,
    pixel_area_arcsec2: Optional[float] = None,
):
    """Convert an image to nanoJansky per pixel.

    Args:
        data: Image array in the units described by ``bunit``/``zeropoint_ab``.
        bunit: FITS BUNIT string, e.g. ``'10.0*nanoJansky'`` or ``'MJy/sr'``.
        zeropoint_ab: AB magnitude zeropoint.  Takes precedence over ``bunit``.
        pixel_area_arcsec2: Pixel solid angle in arcsec^2, required for
            surface-brightness units.

    Returns:
        Array of the same shape in nJy.

    Raises:
        ValueError: If the units cannot be resolved.
    """
    scale = flux_scale_to_nJy(
        bunit=bunit, zeropoint_ab=zeropoint_ab, pixel_area_arcsec2=pixel_area_arcsec2
    )
    return _apply_scale(data, scale)


def variance_to_nJy2(
    data,
    bunit: Optional[str] = None,
    zeropoint_ab: Optional[float] = None,
    pixel_area_arcsec2: Optional[float] = None,
):
    """Convert a variance map to nJy^2 per pixel.

    The variance scales as the square of the flux conversion factor.  ``bunit``
    should be the unit of the *flux* (variance headers frequently repeat the
    science BUNIT rather than its square).

    Args:
        data: Variance array in (flux unit)^2.
        bunit: FITS BUNIT string of the corresponding flux image.
        zeropoint_ab: AB magnitude zeropoint.  Takes precedence over ``bunit``.
        pixel_area_arcsec2: Pixel solid angle in arcsec^2.

    Returns:
        Array of the same shape in nJy^2.

    Raises:
        ValueError: If the units cannot be resolved.
    """
    scale = flux_scale_to_nJy(
        bunit=bunit, zeropoint_ab=zeropoint_ab, pixel_area_arcsec2=pixel_area_arcsec2
    )
    return _apply_scale(data, scale**2)


def weight_to_variance(wht, floor: float = 0.0):
    """Convert an inverse-variance weight map to variance plus a validity mask.

    Args:
        wht: Inverse-variance weight array.  Non-positive (or non-finite)
            weights mark unusable pixels.
        floor: Weights at or below this value are treated as invalid.  The
            default of 0 treats only zero/negative weights as invalid.

    Returns:
        Tuple ``(variance, mask)`` where ``variance`` is ``1 / wht`` with
        ``inf`` at invalid pixels and ``mask`` is 1.0 for valid pixels and 0.0
        for invalid ones, both as float32 arrays.
    """
    w = np.asarray(wht, dtype=np.float64)
    valid = np.isfinite(w) & (w > float(floor))
    variance = np.full(w.shape, np.inf, dtype=np.float64)
    np.divide(1.0, w, out=variance, where=valid)
    variance[~valid] = np.inf
    mask = valid.astype(np.float32)
    return variance.astype(np.float32), mask
