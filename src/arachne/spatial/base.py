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

    Subclasses may also override ``model_image`` when the unconvolved model
    image is not naturally a per-pixel emulator evaluation (e.g.
    ``AdditiveComponentModel`` evaluates the emulator once per component and
    multiplies by surface-brightness profiles).  The default implementation
    is ``decode -> emulator.predict -> reshape``.

    Models with a proper (normalisable) prior may implement ``sample_prior``
    for use by nested sampling.
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

    def model_image(
        self,
        theta: jnp.ndarray,
        emulator,
        image_shape: tuple,
    ) -> jnp.ndarray:
        """Compute the unconvolved model image (N_bands, H, W).

        Default implementation: decode theta to per-pixel physical SPS
        parameters, evaluate the emulator on every pixel, and reshape the
        (H*W, N_bands) flux table to an image cube.  Subclasses may override
        this to use a cheaper or physically different construction.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).
            emulator: ``SPSEmulator`` whose ``predict`` maps (N, N_sps) -> (N, N_bands).
            image_shape: Spatial dimensions (H, W).

        Returns:
            Unconvolved model image of shape (N_bands, H, W) in nJy.
        """
        H, W = image_shape
        pixel_params = self.decode(theta, (H, W))  # (H*W, N_sps)
        pixel_fluxes = emulator.predict(pixel_params)  # (H*W, N_bands)
        n_bands = pixel_fluxes.shape[1]
        return pixel_fluxes.T.reshape(n_bands, H, W)

    def sample_prior(self, key, n: int) -> jnp.ndarray:
        """Draw ``n`` samples of theta from the prior.

        Used by nested sampling; implement for models with a proper prior.
        The default raises ``NotImplementedError``.

        Args:
            key: ``jax.random`` PRNG key.
            n: Number of samples to draw.

        Returns:
            Array of shape (n, n_params) of unconstrained theta vectors.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a proper prior to sample from."
        )
