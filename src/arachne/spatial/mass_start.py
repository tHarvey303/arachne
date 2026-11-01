"""Colour-aware blind start for the stellar-mass map.

A single neutral SPS template converts one band into mass with the wrong
mass-to-light ratio wherever the stellar population differs (young clumps,
dusty regions).  :func:`template_mass_map` picks, per pixel, the best of a small
grid of templates from the multi-band colours and corrects the baseline map.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from arachne.emulator.base import SPSEmulator
from arachne.spatial.correlated_field import mass_map_from_image

__all__ = ["MassStart", "default_sfh_dust_templates", "template_mass_map"]


@dataclass
class MassStart:
    """Result of :func:`template_mass_map`.

    Attributes:
        log_mass: ``(H, W)`` float32 start map, ``baseline + correction``.
        correction: ``(H, W)`` log10 correction applied to the baseline.
        template_index: ``(H, W)`` int index of the chosen template per pixel.
        chi2_best: ``(H, W)`` chi-square of the chosen template.
        templates: The template override dicts, index-aligned with ``template_index``.
    """

    log_mass: np.ndarray
    correction: np.ndarray
    template_index: np.ndarray
    chi2_best: np.ndarray
    templates: list[dict] = field(default_factory=list)


def default_sfh_dust_templates(
    av_values: Sequence[float] = (0.2, 0.6, 1.2),
    ratio0_values: Sequence[float] = (-1.0, 0.0, 1.0),
    ratio_name: str = "logsfr_ratio_0",
) -> list[dict]:
    """Grid of dust and recent-SFH-ratio template overrides.

    Args:
        av_values: ``Av`` values.
        ratio0_values: Values of the most recent SFH ratio.
        ratio_name: Name of the SFH ratio parameter.

    Returns:
        List of ``{"Av": a, ratio_name: r}`` dicts, ``Av`` varying slowest.
    """
    return [{"Av": float(a), ratio_name: float(r)} for a in av_values for r in ratio0_values]


def _gauss_smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """FFT Gaussian smoothing (periodic)."""
    if sigma <= 0:
        return img
    ky = np.fft.fftfreq(img.shape[0])[:, None]
    kx = np.fft.fftfreq(img.shape[1])[None, :]
    kern = np.exp(-2.0 * (np.pi * sigma) ** 2 * (ky**2 + kx**2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * kern))


def template_mass_map(
    obs,
    emulator: SPSEmulator,
    reference_params: Mapping[str, float],
    templates: Sequence[Mapping[str, float]],
    *,
    mass_param: str = "log_mass",
    band: str | int | None = None,
    smooth_sigma_px: float = 1.0,
    correction_smooth_px: float = 2.0,
    snr_min: float = 5.0,
    floor_frac: float = 1e-3,
) -> MassStart:
    """Mass-map start with a per-pixel SPS template chosen from the colours.

    For each template ``t`` the flux per solar mass ``f_tb`` is predicted and the
    weighted least-squares mass ``M_t = sum_b w_b d_b f_tb / sum_b w_b f_tb^2``
    (``w_b = mask / variance``) is solved per pixel.  The template with minimal
    chi-square is chosen where the summed band S/N exceeds ``snr_min`` (else
    template 0).  The result is the :func:`mass_map_from_image` baseline plus the
    smoothed ``log10(M_best / M_0)``, zero where the S/N criterion fails.  Here
    ``M_0`` is the template-0 mass from the baseline band alone (what the baseline
    itself assumes), so the correction removes exactly the baseline's template bias.

    Args:
        obs: ``ObservationCube`` (flux/variance/mask ``(N_bands, H, W)``).
        emulator: Mass-linear SPS emulator whose bands match ``obs``.
        reference_params: Values for every emulator input except the mass.
        templates: Parameter overrides on top of ``reference_params``; index 0
            is the fallback template.
        mass_param: Name of the mass input.
        band: Baseline band (name or index); default the reddest.
        smooth_sigma_px: Smoothing of the baseline image (pixels).
        correction_smooth_px: Gaussian smoothing of the log correction (pixels).
        snr_min: Minimum summed S/N for a pixel to use the template choice.
        floor_frac: Floor fraction for the baseline and the mass clip.

    Returns:
        A :class:`MassStart`.
    """
    templates = [dict(t) for t in templates]
    if not templates:
        raise ValueError("templates must contain at least one entry.")
    names = list(emulator.param_names)
    rows = []
    for t in templates:
        p = {**reference_params, **t}
        rows.append([0.0 if n == mass_param else float(p[n]) for n in names])
    f = np.asarray(emulator.predict(jnp.asarray(np.array(rows, np.float32))), np.float64)  # (T, B)

    d = np.asarray(obs.flux, np.float64)
    var = np.asarray(obs.variance, np.float64)
    mask = np.asarray(obs.mask, np.float64)
    ok = (mask > 0) & (var > 0)
    w = np.where(ok, 1.0 / np.where(ok, var, 1.0), 0.0)  # (B, H, W)

    n_t = f.shape[0]
    mass = np.empty((n_t,) + d.shape[1:])
    chi2 = np.empty_like(mass)
    wd = w * d
    for t in range(n_t):
        ft = f[t][:, None, None]
        den = np.sum(w * ft**2, axis=0)
        m = np.sum(wd * ft, axis=0) / np.where(den > 0, den, 1.0)
        mass[t] = m
        chi2[t] = np.sum(w * (d - m[None] * ft) ** 2, axis=0)

    n_valid = np.maximum(ok.sum(axis=0), 1)
    snr = np.sum(np.sqrt(w) * d, axis=0) / np.sqrt(n_valid)
    use = snr > snr_min

    best = np.where(use, np.argmin(chi2, axis=0), 0)
    chi2_best = np.take_along_axis(chi2, best[None], axis=0)[0]
    m_best = np.take_along_axis(mass, best[None], axis=0)[0]

    # Reference mass: template 0 in the baseline band alone, so the correction is
    # consistent with the baseline map it is added to.
    b = (
        len(obs.band_names) - 1
        if band is None
        else (list(obs.band_names).index(band) if isinstance(band, str) else int(band))
    )
    m_ref = d[b] / f[0, b]
    floor = max(floor_frac * float(mass[0].max()), np.finfo(np.float64).tiny)
    corr = np.log10(np.maximum(m_best, floor)) - np.log10(np.maximum(m_ref, floor))
    corr = _gauss_smooth(corr, correction_smooth_px)
    corr = np.where(use, corr, 0.0)

    base = mass_map_from_image(
        obs,
        emulator,
        reference_params,
        band=band,
        mass_param=mass_param,
        smooth_sigma_px=smooth_sigma_px,
        floor_frac=floor_frac,
    )
    return MassStart(
        log_mass=(base + corr).astype(np.float32),
        correction=corr.astype(np.float32),
        template_index=best.astype(np.int32),
        chi2_best=chi2_best,
        templates=templates,
    )
