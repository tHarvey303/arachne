"""Image scaling helpers shared by the static plots and the viewer (numpy only)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["percentile_limits", "stretch", "rgb_composite", "STRETCHES"]

STRETCHES = ("linear", "asinh", "log", "sqrt")


def percentile_limits(
    img: np.ndarray,
    pct: Sequence[float] = (0.5, 99.5),
    *,
    symmetric: bool = False,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Display limits from percentiles of the finite pixels.

    Args:
        img: Image (any shape).
        pct: Lower and upper percentiles.
        symmetric: Return ``(-m, m)`` with ``m`` the ``pct[1]`` percentile of ``|img|``
            (for residual and chi images).
        mask: Optional weights/boolean of the same shape; only pixels with ``mask > 0`` count.

    Returns:
        ``(lo, hi)`` with ``hi > lo``; ``(0, 1)`` when no pixel is finite.
    """
    arr = np.asarray(img, dtype=float)
    good = np.isfinite(arr)
    if mask is not None:
        good &= np.asarray(mask) > 0
    vals = arr[good]
    if vals.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        m = float(np.percentile(np.abs(vals), pct[1]))
        if not m > 0:
            m = 1.0
        return -m, m
    lo, hi = (float(v) for v in np.percentile(vals, [pct[0], pct[1]]))
    if not hi > lo:
        span = abs(lo) if lo != 0 else 1.0
        hi = lo + span
    return lo, hi


def stretch(
    img: np.ndarray,
    kind: str = "asinh",
    lo: float | None = None,
    hi: float | None = None,
    *,
    asinh_a: float = 0.1,
) -> np.ndarray:
    """Map an image to [0, 1] with a display stretch.

    The image is first normalised to ``x = clip((img - lo) / (hi - lo), 0, 1)``; then
    ``linear`` x, ``asinh`` ``asinh(x/a)/asinh(1/a)``, ``log`` ``log10(1 + 1000 x)/log10(1001)``,
    ``sqrt`` ``sqrt(x)``.

    Args:
        img: Image.
        kind: ``linear``, ``asinh``, ``log`` or ``sqrt``.
        lo: Lower limit (default: 0.5th percentile).
        hi: Upper limit (default: 99.5th percentile).
        asinh_a: Softening of the asinh stretch (smaller = stronger).

    Returns:
        Float array of the same shape in [0, 1]; NaN where ``img`` is NaN.
    """
    arr = np.asarray(img, dtype=float)
    if lo is None or hi is None:
        dlo, dhi = percentile_limits(arr)
        lo = dlo if lo is None else lo
        hi = dhi if hi is None else hi
    span = hi - lo if hi != lo else 1.0
    with np.errstate(invalid="ignore"):
        x = np.clip((arr - lo) / span, 0.0, 1.0)
    if kind == "linear":
        out = x
    elif kind == "asinh":
        out = np.arcsinh(x / asinh_a) / np.arcsinh(1.0 / asinh_a)
    elif kind == "log":
        out = np.log10(1.0 + 1000.0 * x) / np.log10(1001.0)
    elif kind == "sqrt":
        out = np.sqrt(x)
    else:
        raise ValueError(f"stretch kind must be one of {STRETCHES}, got {kind!r}")
    return np.where(np.isnan(arr), np.nan, out)


def rgb_composite(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    *,
    pct: float = 99.5,
    asinh_a: float = 0.1,
) -> np.ndarray:
    """Asinh RGB composite with one shared scale for the three channels.

    The scale runs from 0 to the ``pct`` percentile of all finite values of the three images,
    so colours reflect flux ratios.  NaN pixels become 0 (black).

    Args:
        r: Red channel image (H, W) (longest wavelength).
        g: Green channel image (H, W).
        b: Blue channel image (H, W) (shortest wavelength).
        pct: Upper percentile of the shared scale.
        asinh_a: Asinh softening.

    Returns:
        Array (H, W, 3) in [0, 1].
    """
    chans = [np.asarray(c, dtype=float) for c in (r, g, b)]
    allv = np.concatenate([c[np.isfinite(c)] for c in chans])
    hi = float(np.percentile(allv, pct)) if allv.size else 1.0
    if not hi > 0:
        hi = float(np.max(np.abs(allv))) if allv.size and np.max(np.abs(allv)) > 0 else 1.0
    out = np.stack([stretch(c, "asinh", 0.0, hi, asinh_a=asinh_a) for c in chans], axis=-1)
    return np.nan_to_num(out, nan=0.0)
