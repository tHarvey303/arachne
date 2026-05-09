"""Abstract base class for spatial models."""

from abc import ABC, abstractmethod

import jax.numpy as jnp


class SpatialModel(ABC):
    """Abstract base class for spatial parameterisations of galaxy images.

    A spatial model maps an unconstrained flat vector ``theta`` to per-pixel
    physical SPS parameter arrays.  Inference runs over ``theta``; the model
    is responsible for enforcing physical parameter bounds via its ``decode``
    method and for supplying a differentiable log-prior.

    All subclasses must implement:
    - ``decode(theta, image_shape)`` — theta → pixel params (H*W, N_sps_params)
    - ``log_prior(theta)`` — scalar log-prior, jax.grad-differentiable

    Subclasses may optionally override ``log_prior_from_decoded`` to avoid
    re-decoding theta when decoded params are already available (e.g. in
    ``ForwardModel.log_posterior``).  The default delegates to ``log_prior``.
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Total number of free parameters in the unconstrained vector theta."""
        ...

    @abstractmethod
    def decode(self, theta: jnp.ndarray, image_shape: tuple) -> jnp.ndarray:
        """Map unconstrained theta to per-pixel physical SPS parameters.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).
            image_shape: Spatial dimensions (H, W) of the target image.

        Returns:
            Per-pixel SPS parameter array of shape (H*W, N_sps_params).
            Values are in physical units (bounded by param_bounds).
        """
        ...

    @abstractmethod
    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-prior probability of theta.

        Must be differentiable with ``jax.grad``.  No side effects allowed.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).

        Returns:
            Scalar log-prior value.
        """
        ...

    def log_prior_from_decoded(
        self,
        theta: jnp.ndarray,
        decoded_params: jnp.ndarray,
        image_shape: tuple,
    ) -> jnp.ndarray:
        """Compute the log-prior when decoded params are already available.

        Called by ``ForwardModel.log_posterior`` to avoid re-decoding theta.
        The default implementation ignores ``decoded_params`` and calls
        ``log_prior(theta)``.  Override in subclasses where the prior can be
        computed directly from decoded params (e.g. ``FreeFormPixelMap``).

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).
            decoded_params: Pre-decoded pixel params of shape (H*W, N_sps_params).
            image_shape: Spatial dimensions (H, W).

        Returns:
            Scalar log-prior value.
        """
        return self.log_prior(theta)
