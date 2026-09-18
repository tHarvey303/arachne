"""Forward model pipeline composing all arachne components."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.nuisance import NuisanceModel
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
            │ split_theta -> (theta_spatial, theta_nuisance)
            │ SpatialModel.model_image(theta_spatial, emulator, (H, W))
            │   default: decode() -> pixel_params (H*W, N_sps)
            │            -> SPSEmulator.predict() -> (H*W, N_bands) -> reshape
            │   AdditiveComponentModel: K component SEDs x K profiles
        model_image (N_bands, H, W)
            │ PSFConvolver(..., shifts=nuisance shifts)
            │ + per-band sky
        convolved_image (N_bands, H, W)
            │ GaussianLikelihood(..., log_noise_scale) -> log_likelihood
            │ SpatialModel.log_prior() (or
            │   log_prior_from_decoded() if overridden)
            │ + NuisanceModel.log_prior()               -> log_prior
        log_posterior = log_likelihood + log_prior (scalar)

    Parameter vector layout
    -----------------------
    ``theta = concat(theta_spatial, theta_nuisance)``.  With ``nuisance=None``
    (the default) ``theta`` is exactly the spatial model's vector and every
    method behaves as it did before nuisance support was added.

    Attributes:
        observation: ObservationCube with JAX float32 arrays.
        spatial_model: SpatialModel (FreeFormPixelMap, GaussianMixtureSpatialModel
            or AdditiveComponentModel).
        emulator: SPSEmulator (e.g. ParrotEmulatorV2) — frozen Equinox pytree.
        convolver: PSFConvolver — pre-computed PSF FFTs.
        likelihood: GaussianLikelihood — weighted chi-squared.
        nuisance: Optional NuisanceModel appended to theta (sky, shifts,
            noise rescaling), or None.
    """

    def __init__(
        self,
        observation: ObservationCube,
        spatial_model: SpatialModel,
        emulator: SPSEmulator,
        convolver: PSFConvolver,
        likelihood: GaussianLikelihood,
        nuisance: NuisanceModel | None = None,
    ) -> None:
        """Initialise the ForwardModel.

        Args:
            observation: ObservationCube (call ``to_jax()`` before passing).
            spatial_model: Spatial parameterisation.
            emulator: SPS emulator for photometry prediction.
            convolver: PSF convolver with pre-computed PSF FFTs.
            likelihood: Log-likelihood function.
            nuisance: Optional instrumental nuisance model whose parameters are
                appended to the spatial theta.  ``None`` (default) disables it.
        """
        self.observation = observation
        self.spatial_model = spatial_model
        self.emulator = emulator
        self.convolver = convolver
        self.likelihood = likelihood
        self.nuisance = nuisance

    @classmethod
    def build(
        cls,
        obs: ObservationCube,
        psf_model: PSFModel,
        spatial_model: SpatialModel,
        emulator: SPSEmulator,
        model_error_frac: float | jnp.ndarray = 0.0,
        nuisance: NuisanceModel | None = None,
        pad_psf: bool = True,
    ) -> "ForwardModel":
        """Convenience constructor that assembles all components.

        Calls ``obs.to_jax()``, constructs a ``PSFConvolver``, and wires up
        the ``GaussianLikelihood`` — so the caller only needs the raw data
        objects.

        .. note::
           ``pad_psf`` defaults to **True**, which changes the numerics
           relative to earlier versions of arachne: the PSF convolution is now
           a linear (zero-padded) convolution rather than a circular one, so
           model pixels within roughly half a PSF width of the frame edge no
           longer receive flux wrapped around from the opposite edge.  This is
           the physically correct behaviour; pass ``pad_psf=False`` to
           reproduce the old, wrapping convolution bit-for-bit.

        Args:
            obs: ObservationCube (numpy arrays are fine; ``to_jax()`` is called here).
            psf_model: PSFModel with per-band PSF kernels.
            spatial_model: Spatial parameterisation.
            emulator: SPS emulator.
            model_error_frac: Fractional model-error floor added in quadrature
                to the pixel variance by ``GaussianLikelihood`` (0 disables it);
                a scalar or a per-band array of shape (N_bands,).
                Use a few per cent when the data S/N exceeds the emulator's
                accuracy, otherwise the posterior is narrower than the
                emulator's own systematics.
            nuisance: Optional ``NuisanceModel`` whose parameters are appended
                to the spatial theta.
            pad_psf: Use the zero-padded (linear) PSF convolution.  See the
                note above.

        Returns:
            Fully assembled ForwardModel ready for inference.
        """
        obs_jax = obs.to_jax()
        H, W = obs_jax.image_shape
        convolver = PSFConvolver(psf_model, image_shape=(H, W), pad=pad_psf)
        likelihood = GaussianLikelihood(obs_jax, model_error_frac=model_error_frac)
        n_nuisance = nuisance.n_params if nuisance is not None else 0
        logger.info(
            f"ForwardModel built: image {H}×{W}, {obs_jax.n_bands} bands, "
            f"{spatial_model.n_params} spatial + {n_nuisance} nuisance free parameters "
            f"(pad_psf={pad_psf})."
        )
        return cls(
            observation=obs_jax,
            spatial_model=spatial_model,
            emulator=emulator,
            convolver=convolver,
            likelihood=likelihood,
            nuisance=nuisance,
        )

    # ------------------------------------------------------------------
    # Parameter-vector plumbing
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Total free parameters: spatial model plus nuisance block."""
        n = self.spatial_model.n_params
        if self.nuisance is not None:
            n += self.nuisance.n_params
        return n

    def split_theta(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Split a full theta into its spatial and nuisance sub-vectors.

        Args:
            theta: Full parameter vector of shape (n_params,).

        Returns:
            Tuple ``(theta_spatial, theta_nuisance)``.  The nuisance part has
            length 0 when no ``NuisanceModel`` is attached.
        """
        theta = jnp.asarray(theta)
        n_spatial = self.spatial_model.n_params
        return theta[:n_spatial], theta[n_spatial:]

    def initial_theta_from_spatial(self, theta_spatial: jnp.ndarray) -> jnp.ndarray:
        """Extend a spatial-only theta to a full theta with nuisance defaults.

        Useful when an initialiser (``blind_initial_theta``, ``find_map``)
        produces only the spatial block.

        Args:
            theta_spatial: Spatial parameter vector of shape
                (spatial_model.n_params,).

        Returns:
            Full theta of shape (n_params,) with the nuisance block at its
            prior mean (zeros).
        """
        theta_spatial = jnp.asarray(theta_spatial)
        if self.nuisance is None:
            return theta_spatial
        return jnp.concatenate([theta_spatial, self.nuisance.initial_theta()])

    def sample_prior(self, key, n: int) -> jnp.ndarray:
        """Draw ``n`` full theta vectors from the joint prior.

        Args:
            key: ``jax.random`` PRNG key.
            n: Number of samples.

        Returns:
            Array of shape (n, n_params).

        Raises:
            NotImplementedError: Propagated from the spatial model when it does
                not define a proper, samplable prior.
        """
        if self.nuisance is None:
            return self.spatial_model.sample_prior(key, n)
        key_spatial, key_nuisance = jax.random.split(key)
        theta_spatial = self.spatial_model.sample_prior(key_spatial, n)
        theta_nuisance = self.nuisance.sample_prior(key_nuisance, n)
        return jnp.concatenate([theta_spatial, theta_nuisance], axis=1)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

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

    def _nuisance_blocks(self, theta_nuisance: jnp.ndarray) -> dict[str, jnp.ndarray | None]:
        """Decode the nuisance sub-vector into optional shifts/sky/noise terms.

        Blocks that are switched off (or absent entirely) come back as ``None``
        so the downstream calls keep their zero-cost fast paths.

        Args:
            theta_nuisance: Nuisance parameter vector of shape (n_nuisance,).

        Returns:
            Dict with keys ``"shifts"``, ``"sky"`` and ``"log_noise_scale"``,
            each either a JAX array or ``None``.
        """
        if self.nuisance is None or self.nuisance.n_params == 0:
            return {"shifts": None, "sky": None, "log_noise_scale": None}
        blocks = self.nuisance.split(theta_nuisance)
        return {
            "shifts": blocks["shifts"] if self.nuisance.fit_shifts else None,
            "sky": blocks["sky"] if self.nuisance.fit_sky else None,
            "log_noise_scale": (
                blocks["log_noise_scale"] if self.nuisance.fit_noise_scale else None
            ),
        }

    def _apply_instrument(
        self, model_image: jnp.ndarray, nuis: dict[str, jnp.ndarray | None]
    ) -> jnp.ndarray:
        """PSF-convolve (with optional shifts) and add the per-band sky.

        Args:
            model_image: Unconvolved model image of shape (N_bands, H, W).
            nuis: Output of :meth:`_nuisance_blocks`.

        Returns:
            Observed-frame model image of shape (N_bands, H, W).
        """
        convolved = self.convolver(model_image, shifts=nuis["shifts"])
        if nuis["sky"] is not None:
            convolved = convolved + nuis["sky"][:, None, None]
        return convolved

    def _convolved_image(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Spatial model image, PSF convolution, shifts and sky, from full theta."""
        H, W = self.observation.image_shape
        theta_spatial, theta_nuisance = self.split_theta(theta)
        model_image = self.spatial_model.model_image(theta_spatial, self.emulator, (H, W))
        return self._apply_instrument(model_image, self._nuisance_blocks(theta_nuisance))

    def _model_image(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the PSF-convolved predicted image from unconstrained theta.

        This is a pure JAX function with no side effects.  It includes the
        nuisance sub-pixel shifts and per-band sky levels when a
        ``NuisanceModel`` is attached.

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
        H, W = self.observation.image_shape
        theta_spatial, theta_nuisance = self.split_theta(theta)
        nuis = self._nuisance_blocks(theta_nuisance)
        model_image = self.spatial_model.model_image(theta_spatial, self.emulator, (H, W))
        return self.likelihood(
            self._apply_instrument(model_image, nuis),
            log_noise_scale=nuis["log_noise_scale"],
        )

    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the joint log-prior of theta (spatial + nuisance).

        Uses ``log_prior_from_decoded`` (with a fresh decode) when the spatial
        model overrides it, otherwise ``log_prior(theta)`` directly.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).

        Returns:
            Scalar log-prior value.
        """
        theta_spatial, theta_nuisance = self.split_theta(theta)
        if self._prior_needs_decoded:
            H, W = self.observation.image_shape
            decoded = self.spatial_model.decode(theta_spatial, (H, W))
            log_prior = self.spatial_model.log_prior_from_decoded(theta_spatial, decoded, (H, W))
        else:
            log_prior = self.spatial_model.log_prior(theta_spatial)
        if self.nuisance is not None:
            log_prior = log_prior + self.nuisance.log_prior(theta_nuisance)
        return log_prior

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
        theta_spatial, theta_nuisance = self.split_theta(theta)
        nuis = self._nuisance_blocks(theta_nuisance)

        if self._prior_needs_decoded:
            # Decode once; reuse for both the emulator and the prior.
            pixel_params = self.spatial_model.decode(theta_spatial, (H, W))  # (H*W, N_sps)
            if type(self.spatial_model).model_image is SpatialModel.model_image:
                # Default per-pixel construction: share the decode with the prior.
                pixel_fluxes = self.emulator.predict(pixel_params)  # (H*W, N_bands)
                n_bands = pixel_fluxes.shape[1]
                model_image = pixel_fluxes.T.reshape(n_bands, H, W)
            else:
                model_image = self.spatial_model.model_image(theta_spatial, self.emulator, (H, W))
            log_like = self.likelihood(
                self._apply_instrument(model_image, nuis),
                log_noise_scale=nuis["log_noise_scale"],
            )
            log_prior = self.spatial_model.log_prior_from_decoded(
                theta_spatial, pixel_params, (H, W)
            )
        else:
            model_image = self.spatial_model.model_image(theta_spatial, self.emulator, (H, W))
            log_like = self.likelihood(
                self._apply_instrument(model_image, nuis),
                log_noise_scale=nuis["log_noise_scale"],
            )
            log_prior = self.spatial_model.log_prior(theta_spatial)

        if self.nuisance is not None:
            log_prior = log_prior + self.nuisance.log_prior(theta_nuisance)
        return log_like + log_prior
