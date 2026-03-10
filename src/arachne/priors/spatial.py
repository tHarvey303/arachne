"""Spatial smoothness priors for galaxy parameter maps."""

from __future__ import annotations

import jax.numpy as jnp


class GradientPenaltyPrior:
    """L2 gradient penalty (Gaussian Markov Random Field) prior.

    Penalises the squared finite-difference gradient of the parameter map,
    equivalent to a nearest-neighbour Gaussian MRF.  This is the v1 default
    smoothness prior:
    - Smooth everywhere (unlike TV which allows discontinuities)
    - O(H*W*C) cost (unlike full GP which is O((H*W)^3))
    - Trivially differentiable with jax.grad

    Log-prior:
        log p ∝ -λ * Σ_{i,j,k} [(∂_y params_k)²_{ij} + (∂_x params_k)²_{ij}]

    Attributes:
        strength: Regularisation coefficient λ (≥ 0). Higher = smoother maps.
    """

    def __init__(self, strength: float = 1.0) -> None:
        """Initialise the gradient penalty prior.

        Args:
            strength: Regularisation coefficient λ. Set to 0 for no smoothness
                penalty (uniform prior over all spatial configurations).
        """
        self.strength = strength

    def log_prob(self, param_map: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-prior probability of the parameter map.

        Args:
            param_map: Physical parameter map of shape (H, W, C) where C is
                the number of SPS parameters.

        Returns:
            Scalar log-prior value (≤ 0).
        """
        dy = param_map[1:, :, :] - param_map[:-1, :, :]  # (H-1, W, C)
        dx = param_map[:, 1:, :] - param_map[:, :-1, :]  # (H, W-1, C)
        return -self.strength * (jnp.sum(dy**2) + jnp.sum(dx**2))


class TotalVariationPrior:
    """Total variation (L1 gradient) prior.

    Promotes piecewise-constant parameter maps with sharp edges rather than
    smooth gradients.  Useful when the galaxy has distinct structural components
    (bulge + disc) with abrupt transitions.

    The L1 norm is approximated as sqrt(x² + ε²) for numerical stability and
    differentiability.

    Log-prior:
        log p ∝ -λ * Σ_{i,j,k} sqrt((∂_y params_k)²_{ij} + (∂_x params_k)²_{ij} + ε)

    Attributes:
        strength: Regularisation coefficient λ (≥ 0).
        epsilon: Small value for numerical stability in the L1 approximation.
    """

    def __init__(self, strength: float = 1.0, epsilon: float = 1e-8) -> None:
        """Initialise the total variation prior.

        Args:
            strength: Regularisation coefficient λ.
            epsilon: Numerical stability constant for the smooth L1 approximation.
        """
        self.strength = strength
        self.epsilon = epsilon

    def log_prob(self, param_map: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-prior probability of the parameter map.

        Args:
            param_map: Physical parameter map of shape (H, W, C).

        Returns:
            Scalar log-prior value (≤ 0).
        """
        dy = param_map[1:, :, :] - param_map[:-1, :, :]  # (H-1, W, C)
        dx = param_map[:, 1:, :] - param_map[:, :-1, :]  # (H, W-1, C)
        # Pad dy and dx to same shape for element-wise sum
        dy_pad = jnp.pad(dy, ((0, 1), (0, 0), (0, 0)))
        dx_pad = jnp.pad(dx, ((0, 0), (0, 1), (0, 0)))
        tv = jnp.sqrt(dy_pad**2 + dx_pad**2 + self.epsilon)
        return -self.strength * jnp.sum(tv)
