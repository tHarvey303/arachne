"""Forward model pipeline composing all arachne components."""

from __future__ import annotations

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
            │ SpatialModel.model_image(theta, emulator, (H, W))
            │   default: decode() -> pixel_params (H*W, N_sps)
            │            -> SPSEmulator.predict() -> (H*W, N_bands) -> reshape
            │   AdditiveComponentModel: K component SEDs x K profiles
        model_image (N_bands, H, W)
            │ PSFConvolver
        convolved_image (N_bands, H, W)
            │ GaussianLikelihood                      -> log_likelihood
            │ SpatialModel.log_prior() (or
            │   log_prior_from_decoded() if overridden) -> log_prior
        log_posterior = log_likelihood + log_prior (scalar)

    Attributes:
        observation: ObservationCube with JAX float32 arrays.
        spatial_model: SpatialModel (FreeFormPixelMap, GaussianMixtureSpatialModel
            or AdditiveComponentModel).
        emulator: SPSEmulator (e.g. ParrotEmulatorV2) — frozen Equinox pytree.
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
        model_error_frac: float = 0.0,
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
            model_error_frac: Fractional model-error floor added in quadrature
                to the pixel variance by ``GaussianLikelihood`` (0 disables it).
                Use a few per cent when the data S/N exceeds the emulator's
                accuracy, otherwise the posterior is narrower than the
                emulator's own systematics.

        Returns:
            Fully assembled ForwardModel ready for inference.
        """
        obs_jax = obs.to_jax()
        H, W = obs_jax.image_shape
        convolver = PSFConvolver(psf_model, image_shape=(H, W))
        likelihood = GaussianLikelihood(obs_jax, model_error_frac=model_error_frac)
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

    @property
    def _prior_needs_decoded(self) -> bool:
        """True when the spatial model overrides ``log_prior_from_decoded``.

        For such models (e.g. ``FreeFormPixelMap``) the prior is computed from
        the decoded pixel map, so ``log_posterior`` decodes once and shares the
        result.  For all other models ``log_prior(theta)`` is called directly
        and no decode is performed — important for ``AdditiveComponentModel``,
        whose ``decode`` is a summary map the likelihood never uses.
        """
        cls = type(self.spatial_model)
        return cls.log_prior_from_decoded is not SpatialModel.log_prior_from_decoded

    def _convolved_image(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Unconvolved model image via the spatial model, then PSF convolution."""
        H, W = self.observation.image_shape
        model_image = self.spatial_model.model_image(theta, self.emulator, (H, W))
        return self.convolver(model_image)  # (N_bands, H, W)

    def _model_image(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the PSF-convolved predicted image from unconstrained theta.

        This is a pure JAX function with no side effects.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).

        Returns:
            PSF-convolved predicted image of shape (N_bands, H, W) in nJy.
        """
        return self._convolved_image(theta)

    def log_likelihood(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-likelihood of theta given the data.

        Pure JAX function: convolved model image -> ``GaussianLikelihood``.
        Exposed separately from ``log_posterior`` so that nested sampling can
        use the likelihood and prior independently.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).

        Returns:
            Scalar log-likelihood value.
        """
        return self.likelihood(self._convolved_image(theta))

    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the spatial model's log-prior of theta.

        Uses ``log_prior_from_decoded`` (with a fresh decode) when the spatial
        model overrides it, otherwise ``log_prior(theta)`` directly.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).

        Returns:
            Scalar log-prior value.
        """
        if self._prior_needs_decoded:
            H, W = self.observation.image_shape
            decoded = self.spatial_model.decode(theta, (H, W))
            return self.spatial_model.log_prior_from_decoded(theta, decoded, (H, W))
        return self.spatial_model.log_prior(theta)

    def log_posterior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-posterior probability of theta given the data.

        This is a pure JAX function — no logging, no file I/O, no mutable
        state.  It is safe to pass to ``jax.jit`` and ``jax.grad``.

        ``log_posterior(theta) == log_likelihood(theta) + log_prior(theta)``
        always holds; the two are only fused here to share a decode when the
        spatial model's prior needs the decoded pixel map.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).

        Returns:
            Scalar log-posterior value:
            ``log p(theta | data) = log_likelihood + log_prior``
        """
        H, W = self.observation.image_shape

        if self._prior_needs_decoded:
            # Decode once; reuse for both the emulator and the prior.
            pixel_params = self.spatial_model.decode(theta, (H, W))  # (H*W, N_sps)
            if type(self.spatial_model).model_image is SpatialModel.model_image:
                # Default per-pixel construction: share the decode with the prior.
                pixel_fluxes = self.emulator.predict(pixel_params)  # (H*W, N_bands)
                n_bands = pixel_fluxes.shape[1]
                model_image = pixel_fluxes.T.reshape(n_bands, H, W)
            else:
                model_image = self.spatial_model.model_image(theta, self.emulator, (H, W))
            log_like = self.likelihood(self.convolver(model_image))
            log_prior = self.spatial_model.log_prior_from_decoded(theta, pixel_params, (H, W))
            return log_like + log_prior

        log_like = self.likelihood(self._convolved_image(theta))
        return log_like + self.spatial_model.log_prior(theta)
