"""Forward model pipeline composing all arachne components."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.likelihood.gaussian import GaussianLikelihood
from arachne.psf.convolution import PSFConvolver
from arachne.spatial.base import SpatialModel
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


class ForwardModel:
    """Forward model pipeline for spatially-resolved galaxy SED fitting.

    Composes all arachne components into a single ``log_posterior(theta)``
    function that is:

    - **Pure**: no side effects, no mutable state — safe for ``jax.jit``
    - **Differentiable**: ``jax.grad(log_posterior)`` is defined everywhere
    - **JIT-compilable**: the full pipeline can be traced and compiled to GPU

    Data flow
    ---------
    ::

        theta (n_params,)
            │ SpatialModel.decode()
        pixel_params (H*W, N_sps_params)
            │ SPSEmulator.predict()
        pixel_fluxes (H*W, N_bands)
            │ reshape + mask
        model_image (N_bands, H, W)
            │ PSFConvolver
        convolved_image (N_bands, H, W)
            │ GaussianLikelihood + SpatialModel.log_prior()
        log_posterior (scalar)

    Attributes:
        observation: ObservationCube with JAX float32 arrays.
        spatial_model: SpatialModel (FreeFormPixelMap or GaussianMixtureSpatialModel).
        emulator: SPSEmulator (JAXFlowEmulator) — frozen Equinox pytree.
        convolver: PSFConvolver — pre-computed PSF FFTs.
        likelihood: GaussianLikelihood — weighted chi-squared.
    """

    def __init__(
        self,
        observation: ObservationCube,
        spatial_model: SpatialModel,
        emulator: SPSEmulator,
        convolver: PSFConvolver,
        likelihood: GaussianLikelihood,
    ) -> None:
        """Initialise the ForwardModel.

        Args:
            observation: ObservationCube (call ``to_jax()`` before passing).
            spatial_model: Spatial parameterisation.
            emulator: SPS emulator for photometry prediction.
            convolver: PSF convolver with pre-computed PSF FFTs.
            likelihood: Log-likelihood function.
        """
        self.observation = observation
        self.spatial_model = spatial_model
        self.emulator = emulator
        self.convolver = convolver
        self.likelihood = likelihood

    @classmethod
    def build(
        cls,
        obs: ObservationCube,
        psf_model: PSFModel,
        spatial_model: SpatialModel,
        emulator: SPSEmulator,
    ) -> "ForwardModel":
        """Convenience constructor that assembles all components.

        Calls ``obs.to_jax()``, constructs a ``PSFConvolver``, and wires up
        the ``GaussianLikelihood`` — so the caller only needs the raw data
        objects.

        Args:
            obs: ObservationCube (numpy arrays are fine; ``to_jax()`` is called here).
            psf_model: PSFModel with per-band PSF kernels.
            spatial_model: Spatial parameterisation.
            emulator: SPS emulator.

        Returns:
            Fully assembled ForwardModel ready for inference.
        """
        obs_jax = obs.to_jax()
        H, W = obs_jax.image_shape
        convolver = PSFConvolver(psf_model, image_shape=(H, W))
        likelihood = GaussianLikelihood(obs_jax)
        logger.info(
            f"ForwardModel built: image {H}×{W}, {obs_jax.n_bands} bands, "
            f"{spatial_model.n_params} free parameters."
        )
        return cls(
            observation=obs_jax,
            spatial_model=spatial_model,
            emulator=emulator,
            convolver=convolver,
            likelihood=likelihood,
        )

    def _model_image(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the PSF-convolved predicted image from unconstrained theta.

        This is a pure JAX function with no side effects.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).

        Returns:
            PSF-convolved predicted image of shape (N_bands, H, W) in nJy.
        """
        H, W = self.observation.image_shape
        N_bands = self.observation.n_bands

        # 1. Spatial model: theta -> pixel SPS params
        pixel_params = self.spatial_model.decode(theta, (H, W))  # (H*W, N_sps)

        # 2. Emulator: pixel SPS params -> pixel fluxes
        pixel_fluxes = self.emulator.predict(pixel_params)  # (H*W, N_bands)

        # 3. Reshape to image
        model_image = pixel_fluxes.T.reshape(N_bands, H, W)  # (N_bands, H, W)

        # 4. PSF convolution
        convolved = self.convolver(model_image)  # (N_bands, H, W)
        return convolved

    def log_posterior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-posterior probability of theta given the data.

        This is a pure JAX function — no logging, no file I/O, no mutable
        state.  It is safe to pass to ``jax.jit`` and ``jax.grad``.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).

        Returns:
            Scalar log-posterior value:
            ``log p(theta | data) = log_likelihood + log_prior``
        """
        log_like = self.likelihood(self._model_image(theta))
        log_prior = self.spatial_model.log_prior(theta)
        return log_like + log_prior
