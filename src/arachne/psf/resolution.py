"""PSF resolution diagnostics (modulation transfer function)."""

from __future__ import annotations

import numpy as np

__all__ = ["mtf_profile", "psf_k_max"]


def mtf_profile(kernel, n_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged modulation transfer function of a PSF kernel.

    Args:
        kernel: 2D PSF kernel (any normalisation; the centre position is irrelevant).
        n_bins: Number of radial wavenumber bins between 0 and pi rad/pixel.

    Returns:
        ``(k_centres, mtf)``: bin centres in rad/pixel and the mean ``|FFT|``
        normalised to 1 at ``k = 0``.  Empty bins are interpolated.
    """
    kern = np.asarray(kernel, np.float64)
    if kern.ndim != 2:
        raise ValueError(f"kernel must be 2D, got shape {kern.shape}")
    amp = np.abs(np.fft.fft2(kern))
    amp = amp / amp[0, 0]
    ky = 2 * np.pi * np.fft.fftfreq(kern.shape[0])[:, None]
    kx = 2 * np.pi * np.fft.fftfreq(kern.shape[1])[None, :]
    kr = np.hypot(ky, kx).ravel()
    edges = np.linspace(0.0, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(kr, edges) - 1, 0, n_bins)  # bin n_bins = beyond pi (corners)
    sums = np.bincount(idx, weights=amp.ravel(), minlength=n_bins + 1)[:n_bins]
    cnt = np.bincount(idx, minlength=n_bins + 1)[:n_bins]
    centres = 0.5 * (edges[:-1] + edges[1:])
    good = cnt > 0
    mtf = np.interp(centres, centres[good], sums[good] / cnt[good])
    return centres, mtf


def psf_k_max(psf_model, mtf_threshold: float = 0.1, band: str | int | None = None) -> float:
    """Wavenumber where the sharpest band's MTF first drops below a threshold.

    Args:
        psf_model: ``PSFModel`` (``kernels``, ``band_names``).
        mtf_threshold: MTF level defining the cutoff.
        band: Band name or index to use instead of the sharpest band.

    Returns:
        Wavenumber in rad/pixel (linearly interpolated crossing); ``pi`` (Nyquist)
        if the MTF never falls below the threshold.
    """
    kernels = np.asarray(psf_model.kernels)
    if band is None:
        sel = range(len(kernels))
    elif isinstance(band, str):
        sel = [list(psf_model.band_names).index(band)]
    else:
        sel = [int(band)]
    best = 0.0
    for i in sel:
        k, m = mtf_profile(kernels[i])
        below = np.nonzero(m < mtf_threshold)[0]
        if len(below) == 0:
            kc = np.pi
        else:
            j = below[0]
            if j == 0:
                kc = float(k[0])
            else:
                frac = (m[j - 1] - mtf_threshold) / max(m[j - 1] - m[j], 1e-30)
                kc = float(k[j - 1] + frac * (k[j] - k[j - 1]))
        best = max(best, kc)
    return float(best)
