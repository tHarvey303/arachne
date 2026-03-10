"""Free-form per-pixel spatial model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from arachne.priors.spatial import GradientPenaltyPrior
from arachne.spatial.base import SpatialModel


class FreeFormPixelMap(SpatialModel):
    """Per-pixel SPS parameter map with L2 smoothness prior.

    Each pixel in the image has an independent SPS parameter vector.
    For a 75×75 image with 12 SPS params this gives 67,500 free parameters —
    GPU + NUTS gradient sampling is essential at this scale.

    Parameterisation
    ----------------
    The free parameter vector ``theta`` is unconstrained (∈ ℝ^n).
    Physical parameters are recovered via a sigmoid transform:

        param_k = lo_k + (hi_k - lo_k) * sigmoid(theta_k)

    This enforces physical bounds without hard constraints.

    Prior
    -----
    L2 gradient penalty applied in *decoded* (physical) parameter space:

        log p ∝ -λ * Σ_{i,j,k} (∇ params_k)²_{ij}

    where ∇ is the 2D finite-difference gradient across the image.
    The ``smoothness_strength`` can be a scalar (same λ for all parameters)
    or a dict mapping parameter names to per-parameter λ values.

    Attributes:
        image_shape: Spatial dimensions (H, W) of the galaxy image.
        sps_param_names: Ordered list of SPS parameter names.
        param_bounds: Dict mapping param name to (lo, hi) physical bounds.
        smoothness_strength: Scalar or dict of per-parameter λ values.
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        sps_param_names: list[str],
        param_bounds: dict[str, tuple[float, float]],
        smoothness_strength: float | dict[str, float] = 1.0,
    ) -> None:
        """Initialise the FreeFormPixelMap.

        Args:
            image_shape: (H, W) spatial dimensions of the galaxy image.
            sps_param_names: Ordered list of SPS parameter names.
            param_bounds: Dict of param_name → (lo, hi) physical bounds.
            smoothness_strength: Scalar λ or dict of per-parameter λ values.
                Higher values enforce stronger spatial smoothness.
        """
        self.image_shape = image_shape
        self.sps_param_names = sps_param_names
        self.param_bounds = param_bounds
        self.smoothness_strength = smoothness_strength

        H, W = image_shape
        N = len(sps_param_names)
        self._n_params = H * W * N

        # Pre-compute bounds arrays for vectorised sigmoid transform
        lows = jnp.array([param_bounds[p][0] for p in sps_param_names], dtype=jnp.float32)
        highs = jnp.array([param_bounds[p][1] for p in sps_param_names], dtype=jnp.float32)
        self._lows = lows  # (N,)
        self._highs = highs  # (N,)

        # Pre-compute smoothness strengths
        if isinstance(smoothness_strength, dict):
            lambdas = jnp.array(
                [smoothness_strength.get(p, 1.0) for p in sps_param_names],
                dtype=jnp.float32,
            )
        else:
            lambdas = jnp.full((N,), smoothness_strength, dtype=jnp.float32)
        self._lambdas = lambdas  # (N,)

    @property
    def n_params(self) -> int:
        """Total number of free parameters (H * W * N_sps_params)."""
        return self._n_params

    def decode(self, theta: jnp.ndarray, image_shape: tuple) -> jnp.ndarray:
        """Map unconstrained theta to per-pixel physical SPS parameters.

        Applies sigmoid transform to enforce physical bounds:
        ``param = lo + (hi - lo) * sigmoid(theta_raw)``

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).
            image_shape: Spatial dimensions (H, W) — must match self.image_shape.

        Returns:
            Per-pixel SPS parameter array of shape (H*W, N_sps_params).
        """
        H, W = image_shape
        N = len(self.sps_param_names)
        # Reshape to (H*W, N)
        theta_2d = theta.reshape(H * W, N)
        # Sigmoid transform to physical space
        params = self._lows + (self._highs - self._lows) * jax.nn.sigmoid(theta_2d)
        return params  # (H*W, N)

    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the L2 gradient penalty prior on the decoded parameter map.

        The prior is applied in physical (decoded) parameter space to penalise
        spatial discontinuities:

            log p ∝ -Σ_k λ_k * (Σ_{i,j} (dy_k)^2 + (dx_k)^2)

        where dy, dx are finite-difference gradients in y and x directions.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).

        Returns:
            Scalar log-prior value (≤ 0).
        """
        H, W = self.image_shape
        N = len(self.sps_param_names)
        # Decode to physical space
        params = self.decode(theta, self.image_shape)  # (H*W, N)
        param_map = params.reshape(H, W, N)  # (H, W, N)

        # Finite-difference gradients
        dy = param_map[1:, :, :] - param_map[:-1, :, :]  # (H-1, W, N)
        dx = param_map[:, 1:, :] - param_map[:, :-1, :]  # (H, W-1, N)

        # Weighted sum over parameters
        penalty = jnp.sum(self._lambdas * (jnp.sum(dy**2, axis=(0, 1)) + jnp.sum(dx**2, axis=(0, 1))))
        return -penalty
