"""Abstract base class for spatial models."""

from abc import ABC, abstractmethod

import jax.numpy as jnp


class SpatialModel(ABC):
    """Abstract base class for spatial parameterisations of galaxy images.

    A spatial model maps an unconstrained flat vector ``theta`` to per-pixel
    physical SPS parameter arrays.  Inference runs over ``theta``; the model
    is responsible for enforcing physical parameter bounds via its ``decode``
    method and for supplying a differentiable log-prior.

    All subclasses must implement exactly two methods:
    - ``decode(theta, image_shape)`` — theta → pixel params (H*W, N_sps_params)
    - ``log_prior(theta)`` — scalar log-prior, jax.grad-differentiable

    These are the only methods called by ``ForwardModel``.  New spatial models
    are drop-in replacements as long as they satisfy this interface.
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
