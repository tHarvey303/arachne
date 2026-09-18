"""Optional WCS-level multi-resolution forward model.

:class:`~arachne.forward_model.pipeline.ForwardModel` assumes every band shares
one pixel grid.  Real JWST imaging does not: NIRCam short-wavelength mosaics
are natively 0.02-0.03 arcsec/pixel and long-wavelength ones 0.04-0.06, so a
single-grid fit must first drizzle everything onto a common grid — throwing
away SW resolution, or interpolating (and correlating the noise of) the LW
data.

:class:`MultiResolutionForwardModel` avoids that entirely: the *data* are never
resampled.  Each band keeps its own pixel grid and its own PSF, and the same
physical components are rendered directly on that grid through the band's WCS,
using :func:`~arachne.spatial.profiles.render_on_grid` on the tangent-plane
coordinates supplied by
:meth:`~arachne.data.multires.BandImage.sky_coords`.  Because
:class:`~arachne.likelihood.gaussian.GaussianLikelihood` is a plain sum over
pixels and bands, the joint log-likelihood is *exactly* the sum of the per-band
log-likelihoods — nothing is approximated by the split.

Multi-resolution fitting is **optional and more expensive** than the
single-grid path: the bands have ragged shapes, so they cannot be vmapped and
each one costs its own FFT convolution.  The emulator, which dominates the cost
of a resolved fit, is still evaluated exactly ``K`` times per likelihood call
(once per component, shared across all bands).  Use
:class:`~arachne.forward_model.pipeline.ForwardModel` whenever the bands really
do share a grid (e.g. a DJA ``thumb`` cutout, where the server has already
resampled every filter onto one 0.05 arcsec/pixel grid).

Coordinate frame
----------------
**This model works in the sky frame, not the pixel frame.**  Component centres,
sizes and position angles are expressed in arcsec offsets from the observation's
reference position ``(ref_ra, ref_dec)``, with ``mu_y`` positive towards
**North** and ``mu_x`` positive towards **East** — the convention of
:meth:`~arachne.data.multires.BandImage.sky_coords`.  A single-grid
:class:`~arachne.forward_model.pipeline.ForwardModel` in arcsec mode instead
uses ``+y = +row`` and ``+x = +column``, and for the usual "North up, East
left" orientation ``+column`` points **West**.  The two frames are therefore
mirror images of one another in ``x``: a component at ``mu_x = +0.1`` here lies
0.1 arcsec East of the reference position, whereas in the single-grid model it
lies 0.1 arcsec towards higher column index.  Sizes are unaffected but the sign
of ``rho`` (the position-angle direction) flips with the handedness.  Use
:meth:`sky_to_pixel` / :meth:`pixel_to_sky` to move between the sky frame and
any band's pixel indices, e.g. to overlay fitted component centres on an image.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from arachne.data.multires import MultiResolutionObservation
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.nuisance import NuisanceModel
from arachne.likelihood.gaussian import GaussianLikelihood
from arachne.psf.convolution import PSFConvolver
from arachne.spatial.additive import AdditiveComponentModel
from arachne.spatial.profiles import render_on_grid
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

__all__ = ["MultiResolutionForwardModel"]


class MultiResolutionForwardModel:
    """Forward model fitting every band on its own native pixel grid.

    The parameter vector, the priors and every ``log_*`` method have exactly
    the same meaning and layout as in
    :class:`~arachne.forward_model.pipeline.ForwardModel`, so
    :class:`~arachne.inference.nss_sampler.NSSSampler`,
    :class:`~arachne.inference.nuts_sampler.NUTSSampler`,
    :func:`~arachne.inference.initialisation.find_map` and the
    posterior-predictive helpers all work on it unchanged::

        theta = concat(
            theta_spatial,
            theta_nuisance,
        )

    Per likelihood call the model

    1. evaluates the emulator once per component —
       ``spatial_model.component_seds(theta_spatial, emulator)`` — giving the
       ``(K, N_emulator_bands)`` SED matrix, shared by every band;
    2. for each band ``b``, renders the ``K`` profiles analytically on that
       band's pixel centres (``render_on_grid`` on the tangent-plane
       coordinates, with that band's pixel area and oversampling) and forms
       ``I_b = Σ_k F_k[band_index_b] · P_kb``;
    3. convolves ``I_b`` with that band's PSF on that band's grid (optionally
       with the nuisance sub-pixel shift), adds the band's sky, and evaluates
       that band's :class:`~arachne.likelihood.gaussian.GaussianLikelihood`;
    4. sums the per-band log-likelihoods.

    The loop over bands is a *Python* loop over static band indices, so the
    whole thing traces into one jit-able graph despite the ragged shapes (which
    is also why ``vmap`` over bands is impossible, and expected to be).

    Requirements on the spatial model
    ---------------------------------
    ``spatial_model`` must be an
    :class:`~arachne.spatial.additive.AdditiveComponentModel` built with

    - ``pixel_scale`` set (arcsec mode — the components must carry physical
      angular sizes, not pixel sizes), and
    - ``normalisation="analytic"`` (a frame-normalised profile is not a surface
      brightness and is meaningless on another band's grid).

    Its ``image_shape`` and ``pixel_scale`` no longer describe any particular
    band: they only set the scale of the default centre and log-size priors
    (``frame size = max(H, W) * pixel_scale``).  Pick them to describe the
    *field* being fitted — typically the finest band's shape and scale.

    Attributes:
        observation: JAX :class:`~arachne.data.multires.MultiResolutionObservation`.
        spatial_model: The AdditiveComponentModel, in arcsec sky coordinates.
        emulator: SPS emulator — frozen Equinox pytree.
        nuisance: Optional :class:`~arachne.forward_model.nuisance.NuisanceModel`
            whose parameters are appended to the spatial theta, or None.
        convolvers: List of per-band :class:`~arachne.psf.convolution.PSFConvolver`.
        likelihoods: List of per-band
            :class:`~arachne.likelihood.gaussian.GaussianLikelihood`.
        band_indices: Column of each observed band in the emulator's band list.
        band_names: Observed band names, in observation order.
        n_bands: Number of bands.
        shapes: Per-band ``(H_b, W_b)``.
        pixel_areas: Per-band pixel area in arcsec^2, ``|det(affine_b)|``.
        oversamples: Per-band sub-samples per pixel side used when rendering.
    """

    def __init__(
        self,
        observation: MultiResolutionObservation,
        spatial_model: AdditiveComponentModel,
        emulator: SPSEmulator,
        convolvers: Sequence[PSFConvolver],
        likelihoods: Sequence[GaussianLikelihood],
        band_indices: Sequence[int],
        nuisance: NuisanceModel | None = None,
        oversample: int | Sequence[int] | None = None,
    ) -> None:
        """Assemble the model from already-built per-band pieces.

        Most callers should use :meth:`build`, which constructs the convolvers
        and likelihoods from the observation itself.

        Args:
            observation: MultiResolutionObservation (call ``to_jax()`` first).
            spatial_model: AdditiveComponentModel in arcsec mode with analytic
                normalisation.
            emulator: SPS emulator.
            convolvers: One PSFConvolver per band, on that band's grid.
            likelihoods: One GaussianLikelihood per band, over that band's
                own 2-D flux/variance/mask.
            band_indices: Column of each observed band in ``emulator.band_names``.
            nuisance: Optional nuisance model (``n_bands`` must match).
            oversample: Sub-samples per pixel side: ``None`` uses the spatial
                model's own ``oversample`` for every band, an int applies to
                every band, or a sequence gives one value per band.

        Raises:
            TypeError: If ``spatial_model`` is not an AdditiveComponentModel.
            ValueError: If the spatial model is not in arcsec / analytic mode,
                or any per-band sequence has the wrong length, or an
                ``oversample`` entry is < 1.
        """
        _validate_spatial_model(spatial_model)
        n_bands = observation.n_bands
        for name, seq in (
            ("convolvers", convolvers),
            ("likelihoods", likelihoods),
            ("band_indices", band_indices),
        ):
            if len(seq) != n_bands:
                raise ValueError(
                    f"{name} has {len(seq)} entries but the observation has {n_bands} bands."
                )
        if nuisance is not None and nuisance.n_bands != n_bands:
            raise ValueError(
                f"nuisance.n_bands={nuisance.n_bands} but the observation has {n_bands} bands."
            )

        self.observation = observation
        self.spatial_model = spatial_model
        self.emulator = emulator
        self.convolvers = list(convolvers)
        self.likelihoods = list(likelihoods)
        self.band_indices = [int(i) for i in band_indices]
        self.nuisance = nuisance
        self.oversamples = _resolve_oversample(oversample, spatial_model.oversample, n_bands)

        # Static per-band geometry: tangent-plane coordinates of the pixel
        # centres (arcsec, +North / +East) and the pixel area in arcsec^2.
        self._coords: list[tuple[jnp.ndarray, jnp.ndarray]] = []
        self.pixel_areas: list[float] = []
        for band in observation.bands:
            h, w = band.shape
            yy, xx = band.sky_coords()
            self._coords.append(
                (
                    jnp.asarray(yy.reshape(h, w), dtype=jnp.float32),
                    jnp.asarray(xx.reshape(h, w), dtype=jnp.float32),
                )
            )
            self.pixel_areas.append(float(np.abs(np.linalg.det(np.asarray(band.affine)))))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        observation: MultiResolutionObservation,
        spatial_model: AdditiveComponentModel,
        emulator: SPSEmulator,
        model_error_frac: float | Sequence[float] = 0.0,
        nuisance: NuisanceModel | None = None,
        pad_psf: bool = True,
        oversample: int | Sequence[int] | None = None,
    ) -> "MultiResolutionForwardModel":
        """Assemble the model from a multi-resolution observation.

        Calls ``observation.to_jax()``, then builds one
        :class:`~arachne.psf.convolution.PSFConvolver` and one
        :class:`~arachne.likelihood.gaussian.GaussianLikelihood` per band on
        that band's own grid.

        Args:
            observation: MultiResolutionObservation with a PSF on every band
                (numpy arrays are fine; ``to_jax()`` is called here).
            spatial_model: AdditiveComponentModel with ``pixel_scale`` set and
                ``normalisation="analytic"``.
            emulator: SPS emulator whose ``band_names`` cover every observed band.
            model_error_frac: Fractional model-error floor, a scalar for every
                band or one value per band.  Always use a few per cent on real
                resolved data (see :class:`GaussianLikelihood`).
            nuisance: Optional ``NuisanceModel`` over the same bands, in the
                observation's band order.
            pad_psf: Use the zero-padded (linear) PSF convolution on every band.
            oversample: Sub-samples per pixel side, ``None`` (the spatial
                model's own value), one int for every band, or one per band.
                Coarse bands deserve more than fine ones for cuspy profiles.

        Returns:
            Assembled MultiResolutionForwardModel.

        Raises:
            TypeError: If ``spatial_model`` is not an AdditiveComponentModel.
            ValueError: If the spatial model is not in arcsec / analytic mode,
                a band has no PSF, the emulator is missing an observed band, or
                ``model_error_frac`` has the wrong length.
        """
        _validate_spatial_model(spatial_model)
        obs = observation.to_jax()
        n_bands = obs.n_bands

        missing_psf = [b.band_name for b in obs.bands if b.psf is None]
        if missing_psf:
            raise ValueError(
                "MultiResolutionForwardModel needs a PSF kernel on each band's own pixel "
                f"grid, but these bands have psf=None: {missing_psf}.  Pass "
                "psfs={band: (kernel, kernel_pixel_scale)} when loading the observation."
            )

        emulator_bands = list(emulator.band_names)
        lookup = {name: i for i, name in enumerate(emulator_bands)}
        missing_bands = [n for n in obs.band_names if n not in lookup]
        if missing_bands:
            raise ValueError(
                f"The emulator does not predict these observed bands: {missing_bands}.  "
                f"Emulator bands: {emulator_bands}"
            )
        band_indices = [lookup[n] for n in obs.band_names]

        fracs = np.asarray(model_error_frac, dtype=np.float64)
        if fracs.ndim == 0:
            fracs = np.full(n_bands, float(fracs))
        elif fracs.ndim != 1 or fracs.shape[0] != n_bands:
            raise ValueError(
                f"model_error_frac must be a scalar or have {n_bands} entries, "
                f"got shape {fracs.shape}"
            )

        convolvers: list[PSFConvolver] = []
        likelihoods: list[GaussianLikelihood] = []
        for b, band in enumerate(obs.bands):
            convolvers.append(
                PSFConvolver.from_kernels(
                    band.psf,
                    image_shape=band.shape,
                    pad=pad_psf,
                    band_names=[band.band_name],
                )
            )
            likelihoods.append(
                GaussianLikelihood.from_arrays(
                    band.flux,
                    band.variance,
                    band.mask,
                    model_error_frac=float(fracs[b]),
                    band_names=[band.band_name],
                )
            )

        n_nuisance = nuisance.n_params if nuisance is not None else 0
        logger.info(
            f"MultiResolutionForwardModel built: {n_bands} bands "
            f"{list(zip(obs.band_names, obs.shapes, strict=True))}, "
            f"pixel scales {[round(s, 4) for s in obs.pixel_scales]} arcsec/px, "
            f"{spatial_model.n_params} spatial + {n_nuisance} nuisance free parameters "
            f"(pad_psf={pad_psf})."
        )
        return cls(
            observation=obs,
            spatial_model=spatial_model,
            emulator=emulator,
            convolvers=convolvers,
            likelihoods=likelihoods,
            band_indices=band_indices,
            nuisance=nuisance,
            oversample=oversample,
        )

    # ------------------------------------------------------------------
    # Static description
    # ------------------------------------------------------------------

    @property
    def band_names(self) -> list[str]:
        """Observed band names, in observation order."""
        return self.observation.band_names

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return self.observation.n_bands

    @property
    def shapes(self) -> list[tuple[int, int]]:
        """Per-band image shapes ``(H_b, W_b)``."""
        return self.observation.shapes

    @property
    def n_data(self) -> int:
        """Total number of unmasked pixels summed over all bands."""
        return int(sum(int(jnp.sum(band.mask > 0)) for band in self.observation.bands))

    def __repr__(self) -> str:
        """Compact description of the bands and parameter count."""
        return (
            f"MultiResolutionForwardModel(n_bands={self.n_bands}, "
            f"shapes={self.shapes}, n_params={self.n_params})"
        )

    # ------------------------------------------------------------------
    # Parameter-vector plumbing (identical to ForwardModel)
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
            Tuple ``(theta_spatial, theta_nuisance)``; the nuisance part has
            length 0 when no ``NuisanceModel`` is attached.
        """
        theta = jnp.asarray(theta)
        n_spatial = self.spatial_model.n_params
        return theta[:n_spatial], theta[n_spatial:]

    def initial_theta_from_spatial(self, theta_spatial: jnp.ndarray) -> jnp.ndarray:
        """Extend a spatial-only theta with the nuisance block at its prior mean.

        Args:
            theta_spatial: Spatial parameter vector of shape
                (spatial_model.n_params,).

        Returns:
            Full theta of shape (n_params,).
        """
        theta_spatial = jnp.asarray(theta_spatial)
        if self.nuisance is None:
            return theta_spatial
        return jnp.concatenate([theta_spatial, self.nuisance.initial_theta()])

    def sample_prior(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw ``n`` full theta vectors from the joint prior.

        Args:
            key: ``jax.random`` PRNG key.
            n: Number of samples.

        Returns:
            Array of shape (n, n_params).
        """
        if self.nuisance is None:
            return self.spatial_model.sample_prior(key, n)
        key_spatial, key_nuisance = jax.random.split(key)
        theta_spatial = self.spatial_model.sample_prior(key_spatial, n)
        theta_nuisance = self.nuisance.sample_prior(key_nuisance, n)
        return jnp.concatenate([theta_spatial, theta_nuisance], axis=1)

    def _nuisance_blocks(self, theta_nuisance: jnp.ndarray) -> dict[str, jnp.ndarray | None]:
        """Decode the nuisance sub-vector into optional shifts/sky/noise terms.

        Args:
            theta_nuisance: Nuisance parameter vector of shape (n_nuisance,).

        Returns:
            Dict with keys ``"shifts"``, ``"sky"`` and ``"log_noise_scale"``,
            each a JAX array or ``None`` when the block is switched off.
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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def band_profiles(self, theta_spatial: jnp.ndarray, band_index: int) -> jnp.ndarray:
        """Unit-flux profiles of every component on one band's grid.

        Each entry is the fraction of that component's *total* flux falling in
        that pixel (analytic normalisation), so the map sums to ``<= 1`` — the
        deficit is flux outside the cutout, as in reality.

        Args:
            theta_spatial: Spatial parameter vector.
            band_index: Index of the band in the observation.

        Returns:
            Array of shape (K, H_b, W_b).
        """
        model = self.spatial_model
        yy, xx = self._coords[band_index]
        area = self.pixel_areas[band_index]
        over = self.oversamples[band_index]
        return jnp.stack(
            [
                render_on_grid(profile, model.shape_raw(theta_spatial, k), yy, xx, area, over)
                for k, profile in enumerate(model.profile_objects)
            ]
        )

    def _band_image(
        self, theta_spatial: jnp.ndarray, seds: jnp.ndarray, band_index: int
    ) -> jnp.ndarray:
        """Unconvolved model image of one band, shape (1, H_b, W_b)."""
        profiles = self.band_profiles(theta_spatial, band_index)  # (K, H_b, W_b)
        sed = seds[:, self.band_indices[band_index]]  # (K,)
        return jnp.einsum("k,khw->hw", sed, profiles)[None, :, :]

    def _apply_instrument(
        self, image: jnp.ndarray, band_index: int, nuis: dict[str, jnp.ndarray | None]
    ) -> jnp.ndarray:
        """PSF-convolve one band (with its shift) and add its sky, shape (1, H_b, W_b)."""
        shifts = None
        if nuis["shifts"] is not None:
            shifts = nuis["shifts"][band_index : band_index + 1]
        convolved = self.convolvers[band_index](image, shifts=shifts)
        if nuis["sky"] is not None:
            convolved = convolved + nuis["sky"][band_index]
        return convolved

    def _observed_images(self, theta: jnp.ndarray) -> list[jnp.ndarray]:
        """Per-band convolved model images including sky, each (1, H_b, W_b)."""
        theta_spatial, theta_nuisance = self.split_theta(theta)
        nuis = self._nuisance_blocks(theta_nuisance)
        seds = self.spatial_model.component_seds(theta_spatial, self.emulator)
        return [
            self._apply_instrument(self._band_image(theta_spatial, seds, b), b, nuis)
            for b in range(self.n_bands)
        ]

    def model_images(self, theta: jnp.ndarray) -> list[jnp.ndarray]:
        """PSF-convolved model image of every band, on that band's own grid.

        This is what the likelihood compares against: convolved with the band's
        PSF, shifted by the nuisance registration offset and offset by the
        band's sky, all in nJy.

        Args:
            theta: Full parameter vector of shape (n_params,).

        Returns:
            List of ``n_bands`` arrays of shape (H_b, W_b).
        """
        return [img[0] for img in self._observed_images(theta)]

    def component_images_per_band(self, theta: jnp.ndarray) -> list[jnp.ndarray]:
        """Unconvolved per-component images, per band, in nJy.

        Summing over the component axis of entry ``b`` gives that band's
        *unconvolved* model image (no PSF, no shift, no sky).

        Args:
            theta: Full parameter vector of shape (n_params,).

        Returns:
            List of ``n_bands`` arrays of shape (K, H_b, W_b).
        """
        theta_spatial, _ = self.split_theta(theta)
        seds = self.spatial_model.component_seds(theta_spatial, self.emulator)
        out = []
        for b in range(self.n_bands):
            profiles = self.band_profiles(theta_spatial, b)  # (K, H_b, W_b)
            out.append(seds[:, self.band_indices[b]][:, None, None] * profiles)
        return out

    # ------------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------------

    def log_likelihood(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Joint log-likelihood: the sum of the per-band log-likelihoods.

        Pure JAX, jit-able and differentiable.  The sum is exact — a
        multi-band Gaussian log-likelihood *is* the sum over bands, so nothing
        is lost by splitting the bands onto their own grids.

        Args:
            theta: Full parameter vector of shape (n_params,).

        Returns:
            Scalar log-likelihood.
        """
        theta_spatial, theta_nuisance = self.split_theta(theta)
        nuis = self._nuisance_blocks(theta_nuisance)
        seds = self.spatial_model.component_seds(theta_spatial, self.emulator)
        total = jnp.zeros((), dtype=jnp.float32)
        for b in range(self.n_bands):
            model_b = self._apply_instrument(self._band_image(theta_spatial, seds, b), b, nuis)
            scale = None
            if nuis["log_noise_scale"] is not None:
                scale = nuis["log_noise_scale"][b : b + 1]
            total = total + self.likelihoods[b](model_b, log_noise_scale=scale)
        return total

    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Joint log-prior of theta (spatial + nuisance).

        Args:
            theta: Full parameter vector of shape (n_params,).

        Returns:
            Scalar log-prior.
        """
        theta_spatial, theta_nuisance = self.split_theta(theta)
        log_prior = self.spatial_model.log_prior(theta_spatial)
        if self.nuisance is not None:
            log_prior = log_prior + self.nuisance.log_prior(theta_nuisance)
        return log_prior

    def log_posterior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Log-posterior ``log_likelihood(theta) + log_prior(theta)``.

        Pure: no I/O, no mutable state, no Python branching on traced values,
        so it is safe under ``jax.jit`` and ``jax.grad``.

        Args:
            theta: Full parameter vector of shape (n_params,).

        Returns:
            Scalar log-posterior.
        """
        return self.log_likelihood(theta) + self.log_prior(theta)

    # ------------------------------------------------------------------
    # Residual diagnostics
    # ------------------------------------------------------------------

    def chi_maps(self, theta: jnp.ndarray) -> list[jnp.ndarray]:
        """Per-band photon-noise chi maps ``(flux - model) / sqrt(variance)``.

        Masked pixels are exactly zero.  The model-error floor and any fitted
        noise rescaling are deliberately *not* applied, so the maps are
        comparable across nuisance configurations.

        Args:
            theta: Full parameter vector of shape (n_params,).

        Returns:
            List of ``n_bands`` arrays of shape (H_b, W_b).
        """
        models = self.model_images(theta)
        out = []
        for band, model in zip(self.observation.bands, models, strict=True):
            sigma = jnp.sqrt(jnp.asarray(band.variance) + 1e-30)
            out.append(jnp.where(band.mask > 0, (jnp.asarray(band.flux) - model) / sigma, 0.0))
        return out

    def _chi2_value(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Total masked photon-noise chi-squared as a traceable scalar."""
        models = self.model_images(theta)
        total = jnp.zeros((), dtype=jnp.float32)
        for band, model in zip(self.observation.bands, models, strict=True):
            resid = jnp.asarray(band.flux) - model
            term = jnp.asarray(band.mask) * resid**2 / (jnp.asarray(band.variance) + 1e-30)
            total = total + jnp.sum(jnp.where(band.mask > 0, term, 0.0))
        return total

    def chi2(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, int]:
        """Total masked chi-squared and the number of valid data points.

        Args:
            theta: Full parameter vector of shape (n_params,).

        Returns:
            Tuple ``(chi2_total, n_data)`` summed over every band, with
            ``n_data`` a Python int (static).
        """
        return self._chi2_value(theta), self.n_data

    # ------------------------------------------------------------------
    # Frame conversions
    # ------------------------------------------------------------------

    def sky_to_pixel(
        self,
        band_index: int | str,
        dy: float | np.ndarray,
        dx: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert sky-frame offsets to one band's fractional pixel indices.

        Inverts ``(dy, dx) = affine_b @ (row - ref_row, col - ref_col)``, so a
        fitted component centre can be drawn on that band's image.

        Args:
            band_index: Index or name of the band.
            dy: Offset(s) towards North in arcsec from the reference position.
            dx: Offset(s) towards East in arcsec, same shape as ``dy``.

        Returns:
            Tuple ``(row, col)`` of fractional pixel indices, numpy arrays with
            the shape of ``dy``.
        """
        band = self.observation[band_index]
        inv = np.linalg.inv(np.asarray(band.affine, dtype=np.float64))
        dy_a = np.asarray(dy, dtype=np.float64)
        dx_a = np.asarray(dx, dtype=np.float64)
        row = band.ref_pixel[0] + inv[0, 0] * dy_a + inv[0, 1] * dx_a
        col = band.ref_pixel[1] + inv[1, 0] * dy_a + inv[1, 1] * dx_a
        return row, col

    def pixel_to_sky(
        self,
        band_index: int | str,
        row: float | np.ndarray,
        col: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert one band's pixel indices to sky-frame offsets.

        Args:
            band_index: Index or name of the band.
            row: Fractional row index/indices.
            col: Fractional column index/indices, same shape as ``row``.

        Returns:
            Tuple ``(dy, dx)`` in arcsec from the reference position, ``dy``
            towards North and ``dx`` towards East, numpy arrays with the shape
            of ``row``.
        """
        band = self.observation[band_index]
        affine = np.asarray(band.affine, dtype=np.float64)
        dr = np.asarray(row, dtype=np.float64) - band.ref_pixel[0]
        dc = np.asarray(col, dtype=np.float64) - band.ref_pixel[1]
        return affine[0, 0] * dr + affine[0, 1] * dc, affine[1, 0] * dr + affine[1, 1] * dc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_spatial_model(spatial_model: AdditiveComponentModel) -> None:
    """Check that the spatial model can be rendered on foreign grids.

    Args:
        spatial_model: Candidate spatial model.

    Raises:
        TypeError: If it is not an AdditiveComponentModel.
        ValueError: If it is in pixel-index mode or frame-normalised.
    """
    if not isinstance(spatial_model, AdditiveComponentModel):
        raise TypeError(
            "MultiResolutionForwardModel requires an AdditiveComponentModel spatial model "
            f"(it renders analytic profiles on each band's grid), got "
            f"{type(spatial_model).__name__}."
        )
    if spatial_model.pixel_scale is None:
        raise ValueError(
            "MultiResolutionForwardModel requires a spatial model in arcsec mode: build "
            "AdditiveComponentModel(..., pixel_scale=<arcsec/px>).  With pixel_scale=None "
            "the component sizes are in pixels of one particular grid, which is meaningless "
            "when the bands have different pixel scales."
        )
    if spatial_model.normalisation != "analytic":
        raise ValueError(
            "MultiResolutionForwardModel requires normalisation='analytic': a "
            f"'{spatial_model.normalisation}'-normalised profile is divided by its sum over "
            "one particular grid, so it is not a surface brightness and cannot be rendered "
            "on another band's pixels."
        )


def _resolve_oversample(
    oversample: int | Sequence[int] | None, default: int, n_bands: int
) -> list[int]:
    """Expand the ``oversample`` argument to one integer per band.

    Args:
        oversample: ``None`` (use ``default``), one int, or one per band.
        default: The spatial model's own ``oversample``.
        n_bands: Number of bands.

    Returns:
        List of ``n_bands`` integers.

    Raises:
        ValueError: If a sequence has the wrong length or any entry is < 1.
    """
    if oversample is None:
        values = [int(default)] * n_bands
    elif isinstance(oversample, (int, np.integer)):
        values = [int(oversample)] * n_bands
    else:
        values = [int(o) for o in oversample]
        if len(values) != n_bands:
            raise ValueError(
                f"oversample must be a scalar or have {n_bands} entries, got {len(values)}"
            )
    bad = [v for v in values if v < 1]
    if bad:
        raise ValueError(f"oversample entries must be >= 1, got {values}")
    return values
