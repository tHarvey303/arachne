"""Physical parameter priors for SPS parameter vectors."""

from __future__ import annotations

import jax.numpy as jnp


class IndependentUniformPrior:
    """Independent uniform prior over SPS parameter bounds.

    Returns 0 within the bounds and -inf outside.  Note that for the
    FreeFormPixelMap and GaussianMixtureSpatialModel the bounds are already
    enforced by the sigmoid/atanh parameterisation, so this prior adds no
    gradient signal — it is useful primarily as a sanity check or for
    likelihood-based inference with explicitly bounded parameters.

    Attributes:
        lows: Lower bounds array of shape (N_params,).
        highs: Upper bounds array of shape (N_params,).
    """

    def __init__(self, lows: jnp.ndarray, highs: jnp.ndarray) -> None:
        """Initialise the uniform prior.

        Args:
            lows: Lower bounds for each parameter, shape (N_params,).
            highs: Upper bounds for each parameter, shape (N_params,).
        """
        self.lows = jnp.asarray(lows, dtype=jnp.float32)
        self.highs = jnp.asarray(highs, dtype=jnp.float32)

    def log_prob(self, params: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-prior.

        Args:
            params: Parameter array of shape (..., N_params).

        Returns:
            Scalar log-prior value (0 if in bounds, -inf if out of bounds).
        """
        in_bounds = jnp.all((params >= self.lows) & (params <= self.highs))
        return jnp.where(in_bounds, 0.0, -jnp.inf)


class LogNormalPrior:
    """Independent log-Normal prior for positive-definite SPS parameters.

    Suitable for parameters such as stellar mass or SFR where a log-Normal
    distribution is physically motivated.

    Log-prior:
        log p(x) = -0.5 * ((log(x) - mu) / sigma)^2

    Attributes:
        mu: Log-space mean array of shape (N_params,).
        sigma: Log-space standard deviation array of shape (N_params,).
    """

    def __init__(self, mu: jnp.ndarray, sigma: jnp.ndarray) -> None:
        """Initialise the log-Normal prior.

        Args:
            mu: Log-space mean for each parameter, shape (N_params,).
            sigma: Log-space standard deviation for each parameter, shape (N_params,).
        """
        self.mu = jnp.asarray(mu, dtype=jnp.float32)
        self.sigma = jnp.asarray(sigma, dtype=jnp.float32)

    def log_prob(self, log_params: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-prior assuming log_params = log10(params).

        Args:
            log_params: Log-space parameter array of shape (..., N_params).

        Returns:
            Scalar log-prior value.
        """
        return -0.5 * jnp.sum(((log_params - self.mu) / self.sigma) ** 2)
