"""Gaussian (chi-squared) log-likelihood for multi-band photometry."""

from __future__ import annotations

import copy

import jax.numpy as jnp
import numpy as np

from arachne.data.observation import ObservationCube


class GaussianLikelihood:
    """Weighted chi-squared log-likelihood for multi-band imaging data.

    Assumes independent Gaussian noise in each pixel/band.  With the default
    ``model_error_frac = 0`` and no noise rescaling the log-likelihood is the
    plain weighted chi-squared::

        log p(data | model) = -0.5 * Σ mask * (flux - model)² / variance

    Invalid pixels (mask = 0) are excluded from the sum.  The variance is
    assumed to include all noise contributions (Poisson + read noise + sky).

    Model-error floor
    -----------------
    With ``model_error_frac = f > 0`` a fractional *model* uncertainty is
    added in quadrature to the measurement variance::

        var_eff = variance + (f * model)²
        log p   = -0.5 * Σ mask * [ (flux - model)² / var_eff + log(var_eff) ]

    ``f`` may be a scalar or a per-band array of shape (N_bands,) — useful
    because emulator accuracy and calibration systematics are wavelength
    dependent (e.g. a larger floor in the bluest and reddest filters).

    Because ``var_eff`` depends on the model, the ``log(var_eff)``
    normalisation term is required — without it the likelihood could be
    increased without bound by inflating the model where the data are noisy.
    (When ``f = 0`` and no noise rescaling is used the normalisation is a
    constant and is dropped, so the default behaviour is exactly the plain
    chi-squared above.)

    Noise rescaling
    ---------------
    ``__call__`` optionally accepts a per-band ``log_noise_scale`` ``s``
    (see :class:`~arachne.forward_model.nuisance.NuisanceModel`), which
    multiplies the *standard deviation* of each band by ``exp(s)``::

        var_eff = variance * exp(2 s) + (f * model)²

    Quoted variance maps are routinely wrong by tens of per cent — drizzled
    mosaics have correlated pixel noise, and sky variance is often
    underestimated — so letting the data set the noise level protects against
    an over-confident posterior.  Since ``var_eff`` then depends on a fitted
    parameter, the ``log(var_eff)`` term is again mandatory and is always
    included when ``log_noise_scale`` is supplied.

    Per-band composition
    --------------------
    The log-likelihood is a plain sum over bands and pixels, so a multi-band
    evaluation is *exactly* the sum of independent single-band evaluations::

        sum_b GaussianLikelihood.from_arrays(flux[b], var[b], mask[b],
                                             frac[b])(model[b])
            == GaussianLikelihood(cube, frac)(model)

    A multi-resolution model, whose bands do not share a pixel grid, therefore
    builds one likelihood per band (``from_arrays`` with that band's own 2-D
    flux/variance/mask and its own scalar ``model_error_frac``) and adds the
    results; pass that band's ``log_noise_scale`` as a length-1 array.  No
    shared normalisation is missing from such a sum.

    Why a floor is needed: a neural SPS emulator is accurate to a few per
    cent at best, and a resolved fit sums tens of thousands of pixels, many at
    S/N of tens.  Pure photon noise then implies a posterior far narrower than
    the emulator's own systematic error, so the fit is over-confident and any
    emulator bias is treated as a real signal.  Adding a per-pixel fractional
    floor keeps the posterior width honest; the unresolved catalogue pipeline
    (``scripts/fit_catalogue.py``) applies a fractional error floor to each
    band's flux for the same reason.

    Attributes:
        obs: ObservationCube containing flux, variance, and mask arrays.
            All arrays must be JAX float32 arrays (call ``obs.to_jax()`` first).
        model_error_frac: Fractional model-error floor ``f`` — a Python float
            when a scalar was supplied, otherwise a JAX array of shape
            (N_bands,).  ``0`` disables the floor.
    """

    def __init__(
        self,
        obs: ObservationCube,
        model_error_frac: float | np.ndarray | jnp.ndarray = 0.0,
    ) -> None:
        """Initialise the likelihood with an observation cube.

        Args:
            obs: ObservationCube. Call ``obs.to_jax()`` before passing to ensure
                all arrays are JAX float32 for GPU-accelerated inference.
            model_error_frac: Fractional model uncertainty added in quadrature
                to the measurement variance (see class docstring).  Either a
                scalar applied to all bands or an array of shape (N_bands,).
                ``0`` (default) reproduces the plain weighted chi-squared
                exactly.

        Raises:
            ValueError: If ``model_error_frac`` is negative, has more than one
                dimension, or its length does not match the number of bands.
        """
        frac_np = np.asarray(model_error_frac, dtype=np.float64)
        if frac_np.ndim > 1:
            raise ValueError(
                f"model_error_frac must be a scalar or 1-D (N_bands,), got shape {frac_np.shape}"
            )
        if np.any(frac_np < 0.0):
            raise ValueError(f"model_error_frac must be >= 0, got {model_error_frac}")

        n_bands = int(obs.flux.shape[0])
        if frac_np.ndim == 1 and frac_np.shape[0] != n_bands:
            raise ValueError(
                f"model_error_frac has {frac_np.shape[0]} entries but the cube has {n_bands} bands."
            )

        self.obs = obs
        self._frac_is_scalar = frac_np.ndim == 0
        # ``_frac`` broadcasts against (N_bands, H, W).  A scalar stays a
        # Python float so the frac=0 fast path is bit-for-bit unchanged.
        if self._frac_is_scalar:
            self.model_error_frac = float(frac_np)
            self._frac = self.model_error_frac
        else:
            self.model_error_frac = jnp.asarray(frac_np, dtype=jnp.float32)
            self._frac = self.model_error_frac[:, None, None]
        self._has_frac = bool(np.any(frac_np > 0.0))

    @classmethod
    def from_arrays(
        cls,
        flux: np.ndarray | jnp.ndarray,
        variance: np.ndarray | jnp.ndarray,
        mask: np.ndarray | jnp.ndarray | None = None,
        model_error_frac: float | np.ndarray | jnp.ndarray = 0.0,
        band_names: list[str] | None = None,
    ) -> "GaussianLikelihood":
        """Build a likelihood from bare arrays instead of an ``ObservationCube``.

        Intended for per-band use, where one band lives on its own pixel grid
        with its own variance and mask (see *Per-band composition* in the class
        docstring).  2-D inputs are promoted to a single-band (1, H, W) cube, so
        the returned object behaves exactly like any other
        ``GaussianLikelihood`` with ``N_bands = 1``.

        Args:
            flux: Observed flux, (H, W) or (N_bands, H, W), in nJy.
            variance: Variance with the same shape as ``flux``.  ``inf`` for
                unusable pixels is safe as long as ``mask`` is 0 there.
            mask: Validity weights with the same shape as ``flux``; ``None``
                (default) marks every pixel valid.
            model_error_frac: As in :meth:`__init__`.
            band_names: Optional labels; defaults to ``band_0 ...``.

        Returns:
            ``GaussianLikelihood`` over the supplied arrays.

        Raises:
            ValueError: If ``flux`` is neither 2-D nor 3-D.
        """
        flux_arr = jnp.asarray(flux, dtype=jnp.float32)
        var_arr = jnp.asarray(variance, dtype=jnp.float32)
        if flux_arr.ndim == 2:
            flux_arr = flux_arr[None, :, :]
            var_arr = var_arr[None, :, :]
        elif flux_arr.ndim != 3:
            raise ValueError(f"flux must be (H, W) or (N_bands, H, W), got {flux_arr.shape}")
        if mask is None:
            mask_arr = jnp.ones_like(flux_arr)
        else:
            mask_arr = jnp.asarray(mask, dtype=jnp.float32).reshape(flux_arr.shape)
        if band_names is None:
            band_names = [f"band_{i}" for i in range(flux_arr.shape[0])]
        obs = ObservationCube(
            flux=flux_arr,
            variance=var_arr,
            mask=mask_arr,
            band_names=band_names,
            pixel_scale=1.0,
        )
        return cls(obs, model_error_frac=model_error_frac)

    def with_observation(self, obs) -> "GaussianLikelihood":
        """Return a shallow copy evaluated against a different observation.

        ``obs`` must have the same band count and image shape (its arrays may
        be tracers); the model-error configuration is kept.

        Args:
            obs: ObservationCube (JAX arrays) with ``flux``, ``variance``, ``mask``.

        Returns:
            A copy of this likelihood bound to ``obs``.
        """
        if tuple(obs.flux.shape) != tuple(self.obs.flux.shape):
            raise ValueError(
                f"obs.flux must have shape {tuple(self.obs.flux.shape)}, "
                f"got {tuple(obs.flux.shape)}."
            )
        new = copy.copy(self)
        new.obs = obs
        return new

    def __call__(
        self,
        model_image: jnp.ndarray,
        log_noise_scale: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute the log-likelihood.

        Args:
            model_image: Predicted (PSF-convolved) image of shape
                (N_bands, H, W) in nJy.
            log_noise_scale: Optional per-band log noise scaling ``s`` of shape
                (N_bands,); the variance is multiplied by ``exp(2 s)``.  When
                supplied the ``log(var_eff)`` normalisation is always included,
                because ``var_eff`` then depends on a fitted parameter.

        Returns:
            Scalar log-likelihood value.
        """
        residuals = self.obs.flux - model_image  # (N_bands, H, W)

        if not self._has_frac and log_noise_scale is None:
            chi2 = self.obs.mask * residuals**2 / (self.obs.variance + 1e-30)
            return -0.5 * jnp.sum(chi2)

        if log_noise_scale is None:
            var_eff = self.obs.variance + (self._frac * model_image) ** 2 + 1e-30
        else:
            scale = jnp.exp(2.0 * jnp.asarray(log_noise_scale))[:, None, None]
            var_eff = self.obs.variance * scale + (self._frac * model_image) ** 2 + 1e-30

        terms = residuals**2 / var_eff + jnp.log(var_eff)
        # ``where`` rather than a bare product: masked-out pixels routinely
        # carry variance = inf (that is how MultiResolutionObservation flags
        # them), whose log is inf, and 0 * inf is NaN.  Weights between 0 and 1
        # still act multiplicatively.
        weighted = jnp.where(self.obs.mask > 0.0, self.obs.mask * terms, 0.0)
        return -0.5 * jnp.sum(weighted)
