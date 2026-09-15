"""Gaussian (chi-squared) log-likelihood for multi-band photometry."""

from __future__ import annotations

import jax.numpy as jnp

from arachne.data.observation import ObservationCube


class GaussianLikelihood:
    """Weighted chi-squared log-likelihood for multi-band imaging data.

    Assumes independent Gaussian noise in each pixel/band.  With the default
    ``model_error_frac = 0`` the log-likelihood is the plain weighted
    chi-squared::

        log p(data | model) = -0.5 * Σ mask * (flux - model)² / variance

    Invalid pixels (mask = 0) are excluded from the sum.  The variance is
    assumed to include all noise contributions (Poisson + read noise + sky).

    Model-error floor
    -----------------
    With ``model_error_frac = f > 0`` a fractional *model* uncertainty is
    added in quadrature to the measurement variance::

        var_eff = variance + (f * model)²
        log p   = -0.5 * Σ mask * [ (flux - model)² / var_eff + log(var_eff) ]

    Because ``var_eff`` depends on the model, the ``log(var_eff)``
    normalisation term is required — without it the likelihood could be
    increased without bound by inflating the model where the data are noisy.
    (When ``f = 0`` the normalisation is a constant and is dropped, so the
    default behaviour is exactly the plain chi-squared above.)

    Why a floor is needed: a neural SPS emulator is accurate to a few per
    cent at best, and a resolved fit sums tens of thousands of pixels, many at
    S/N of tens.  Pure photon noise then implies a posterior far narrower than
    the emulator's own systematic error, so the fit is over-confident and any
    emulator bias is treated as a real signal.  Adding a per-pixel fractional
    floor keeps the posterior width honest; the unresolved catalogue pipeline
    (``scripts/fit_catalogue.py``) applies a fractional error floor to each
    band's flux for the same reason.

    Attributes:
        obs: ObservationCube containing flux, variance, and mask arrays.
            All arrays must be JAX float32 arrays (call ``obs.to_jax()`` first).
        model_error_frac: Fractional model-error floor ``f`` (0 disables it).
    """

    def __init__(self, obs: ObservationCube, model_error_frac: float = 0.0) -> None:
        """Initialise the likelihood with an observation cube.

        Args:
            obs: ObservationCube. Call ``obs.to_jax()`` before passing to ensure
                all arrays are JAX float32 for GPU-accelerated inference.
            model_error_frac: Fractional model uncertainty added in quadrature
                to the measurement variance (see class docstring).  ``0``
                (default) reproduces the plain weighted chi-squared exactly.

        Raises:
            ValueError: If ``model_error_frac`` is negative.
        """
        if model_error_frac < 0.0:
            raise ValueError(f"model_error_frac must be >= 0, got {model_error_frac}")
        self.obs = obs
        self.model_error_frac = float(model_error_frac)

    def __call__(self, model_image: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-likelihood.

        Args:
            model_image: Predicted (PSF-convolved) image of shape
                (N_bands, H, W) in nJy.

        Returns:
            Scalar log-likelihood value.
        """
        residuals = self.obs.flux - model_image  # (N_bands, H, W)
        if self.model_error_frac > 0.0:
            var_eff = self.obs.variance + (self.model_error_frac * model_image) ** 2 + 1e-30
            terms = residuals**2 / var_eff + jnp.log(var_eff)
            return -0.5 * jnp.sum(self.obs.mask * terms)
        chi2 = self.obs.mask * residuals**2 / (self.obs.variance + 1e-30)
        return -0.5 * jnp.sum(chi2)
