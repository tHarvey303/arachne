"""Gaussian (chi-squared) log-likelihood for multi-band photometry."""

from __future__ import annotations

import jax.numpy as jnp

from arachne.data.observation import ObservationCube


class GaussianLikelihood:
    """Weighted chi-squared log-likelihood for multi-band imaging data.

    Assumes independent Gaussian noise in each pixel/band:

        log p(data | model) = -0.5 * Σ mask * (flux - model)² / variance

    Invalid pixels (mask = 0) are excluded from the sum.  The variance is
    assumed to include all noise contributions (Poisson + read noise + sky).

    Attributes:
        obs: ObservationCube containing flux, variance, and mask arrays.
            All arrays must be JAX float32 arrays (call ``obs.to_jax()`` first).
    """

    def __init__(self, obs: ObservationCube) -> None:
        """Initialise the likelihood with an observation cube.

        Args:
            obs: ObservationCube. Call ``obs.to_jax()`` before passing to ensure
                all arrays are JAX float32 for GPU-accelerated inference.
        """
        self.obs = obs

    def __call__(self, model_image: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-likelihood.

        Args:
            model_image: Predicted (PSF-convolved) image of shape
                (N_bands, H, W) in nJy.

        Returns:
            Scalar log-likelihood value.
        """
        residuals = self.obs.flux - model_image  # (N_bands, H, W)
        chi2 = self.obs.mask * residuals**2 / (self.obs.variance + 1e-30)
        return -0.5 * jnp.sum(chi2)
