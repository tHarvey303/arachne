"""Instrumental nuisance parameters: sky offsets, astrometric shifts, noise rescaling."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class NuisanceModel:
    """Per-band instrumental nuisance parameters appended to the spatial theta.

    Real imaging is never a clean realisation of the astrophysical model: the
    background subtraction leaves a small residual pedestal, the astrometric
    solution between filters is good only to a fraction of a pixel, and the
    quoted variance maps are often wrong by tens of per cent (correlated noise
    from drizzling, unmodelled sky variance).  Each of these biases the
    inferred stellar populations if it is held fixed at the wrong value, so
    they are better marginalised over.

    Three optional blocks are supported, in this fixed order within the
    nuisance sub-vector:

    1. ``sky`` — one constant added to every pixel of a band, in nJy/pixel.
       Prior: ``Normal(0, sky_prior_sigma)``.
    2. ``shifts`` — ``(dy, dx)`` sub-pixel registration offset per band, in
       pixels, applied by :class:`~arachne.psf.convolution.PSFConvolver` as a
       Fourier phase ramp.  Stored row-major, i.e. ``[dy_0, dx_0, dy_1, ...]``.
       Prior: ``Normal(0, shift_prior_sigma)`` on each component.  A common
       shift of *all* bands is exactly degenerate with moving every spatial
       component, so by default pass ``shift_reference_band`` to pin one
       band's shift to zero (its row is still present in ``split()["shifts"]``
       but is not a free parameter); the other bands' shifts are then
       registrations relative to that band.
    3. ``log_noise_scale`` — natural log of a per-band multiplicative factor on
       the noise *standard deviation*, so the variance is scaled by
       ``exp(2 * s)``.  Prior: ``Normal(0, noise_scale_prior_sigma)``.

    All configuration is stored as Python scalars/bools, so the model is
    static under ``jax.jit``: only the parameter vector is traced.  When every
    block is disabled ``n_params == 0`` and all methods still work, returning
    zero-length arrays and a zero log-prior.

    Attributes:
        n_bands: Number of photometric bands.
        fit_sky: Whether a per-band sky pedestal is fitted.
        fit_shifts: Whether per-band (dy, dx) offsets are fitted.
        fit_noise_scale: Whether per-band log noise scalings are fitted.
        sky_prior_sigma: Prior sigma on each sky level (nJy per pixel).
        shift_prior_sigma: Prior sigma on each shift component (pixels).
        noise_scale_prior_sigma: Prior sigma on each log noise scale.
        shift_reference_band: Index of the band whose shift is pinned to zero,
            or ``None`` to fit all bands (degenerate with the model centres).
    """

    def __init__(
        self,
        n_bands: int,
        fit_sky: bool = True,
        fit_shifts: bool = False,
        fit_noise_scale: bool = False,
        sky_prior_sigma: float = 1.0,
        shift_prior_sigma: float = 0.5,
        noise_scale_prior_sigma: float = 0.3,
        shift_reference_band: int | None = None,
    ) -> None:
        """Initialise the nuisance parameterisation.

        Args:
            n_bands: Number of photometric bands.
            fit_sky: Fit a constant sky level per band.
            fit_shifts: Fit a (dy, dx) sub-pixel offset per band.
            fit_noise_scale: Fit a log noise-scale factor per band.
            sky_prior_sigma: Sigma of the zero-mean Gaussian prior on each
                band's sky level, in nJy per pixel.
            shift_prior_sigma: Sigma of the zero-mean Gaussian prior on each
                shift component, in pixels.
            shift_reference_band: Band index whose (dy, dx) is fixed at zero when
                shifts are fitted, breaking the shift/centre degeneracy.  ``None``
                (default) fits every band.
            noise_scale_prior_sigma: Sigma of the zero-mean Gaussian prior on
                each band's log noise scale.

        Raises:
            ValueError: If ``n_bands`` is not positive or any prior sigma of an
                enabled block is not strictly positive.
        """
        if n_bands <= 0:
            raise ValueError(f"n_bands must be positive, got {n_bands}")
        self.n_bands = int(n_bands)
        self.fit_sky = bool(fit_sky)
        self.fit_shifts = bool(fit_shifts)
        self.fit_noise_scale = bool(fit_noise_scale)
        self.sky_prior_sigma = float(sky_prior_sigma)
        self.shift_prior_sigma = float(shift_prior_sigma)
        self.noise_scale_prior_sigma = float(noise_scale_prior_sigma)
        if shift_reference_band is not None:
            ref = int(shift_reference_band)
            if not 0 <= ref < self.n_bands:
                raise ValueError(
                    f"shift_reference_band={ref} out of range for n_bands={self.n_bands}."
                )
            shift_reference_band = ref
        self.shift_reference_band = shift_reference_band

        for enabled, name, sigma in (
            (self.fit_sky, "sky_prior_sigma", self.sky_prior_sigma),
            (self.fit_shifts, "shift_prior_sigma", self.shift_prior_sigma),
            (self.fit_noise_scale, "noise_scale_prior_sigma", self.noise_scale_prior_sigma),
        ):
            if enabled and not sigma > 0.0:
                raise ValueError(f"{name} must be > 0 when the block is fitted, got {sigma}")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @property
    def n_sky(self) -> int:
        """Number of sky parameters (0 or ``n_bands``)."""
        return self.n_bands if self.fit_sky else 0

    @property
    def n_shift(self) -> int:
        """Number of shift parameters: ``2 * n_bands``, minus 2 with a reference band."""
        if not self.fit_shifts:
            return 0
        return 2 * (self.n_bands - (self.shift_reference_band is not None))

    @property
    def shift_bands(self) -> list[int]:
        """Indices of the bands whose shifts are free parameters (vector order)."""
        if not self.fit_shifts:
            return []
        return [b for b in range(self.n_bands) if b != self.shift_reference_band]

    @property
    def n_noise(self) -> int:
        """Number of log noise-scale parameters (0 or ``n_bands``)."""
        return self.n_bands if self.fit_noise_scale else 0

    @property
    def n_params(self) -> int:
        """Total number of nuisance parameters."""
        return self.n_sky + self.n_shift + self.n_noise

    def param_names(self, band_names: list[str] | None = None) -> list[str]:
        """Human-readable names for each nuisance parameter, in vector order.

        Args:
            band_names: Optional band identifiers used in the names.  Defaults
                to ``band_0 ... band_{n_bands-1}``.

        Returns:
            List of ``n_params`` names such as ``"sky[JWST/NIRCam.F200W]"``,
            ``"dy[...]"``, ``"dx[...]"`` and ``"log_noise_scale[...]"``.

        Raises:
            ValueError: If ``band_names`` has the wrong length.
        """
        if band_names is None:
            band_names = [f"band_{i}" for i in range(self.n_bands)]
        elif len(band_names) != self.n_bands:
            raise ValueError(
                f"band_names has {len(band_names)} entries but n_bands={self.n_bands}."
            )
        names: list[str] = []
        if self.fit_sky:
            names += [f"sky[{b}]" for b in band_names]
        if self.fit_shifts:
            for b in self.shift_bands:
                names += [f"dy[{band_names[b]}]", f"dx[{band_names[b]}]"]
        if self.fit_noise_scale:
            names += [f"log_noise_scale[{b}]" for b in band_names]
        return names

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def split(self, theta_n: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Split the nuisance sub-vector into its named blocks.

        Blocks that are not fitted are returned as zeros of the right shape,
        so downstream code never has to branch on the configuration.

        Args:
            theta_n: Nuisance parameter vector of shape (n_params,).

        Returns:
            Dict with keys ``"sky"`` (N_bands,), ``"shifts"`` (N_bands, 2) as
            ``(dy, dx)``, and ``"log_noise_scale"`` (N_bands,).
        """
        theta_n = jnp.asarray(theta_n)
        nb = self.n_bands
        i = 0
        if self.fit_sky:
            sky = theta_n[i : i + nb]
            i += nb
        else:
            sky = jnp.zeros(nb, dtype=theta_n.dtype)
        if self.fit_shifts:
            free = theta_n[i : i + self.n_shift].reshape(-1, 2)
            i += self.n_shift
            if self.shift_reference_band is None:
                shifts = free
            else:
                shifts = jnp.zeros((nb, 2), dtype=theta_n.dtype)
                shifts = shifts.at[jnp.asarray(self.shift_bands)].set(free)
        else:
            shifts = jnp.zeros((nb, 2), dtype=theta_n.dtype)
        if self.fit_noise_scale:
            log_noise_scale = theta_n[i : i + nb]
        else:
            log_noise_scale = jnp.zeros(nb, dtype=theta_n.dtype)
        return {"sky": sky, "shifts": shifts, "log_noise_scale": log_noise_scale}

    def log_prior(self, theta_n: jnp.ndarray) -> jnp.ndarray:
        """Normalised Gaussian log-prior of the nuisance vector.

        Every fitted parameter contributes a full zero-mean Gaussian log
        density (including the ``-log(sigma) - 0.5 log(2*pi)`` normalisation),
        so the value is directly comparable between configurations and usable
        as an evidence term.

        Args:
            theta_n: Nuisance parameter vector of shape (n_params,).

        Returns:
            Scalar log-prior; exactly ``0.0`` when ``n_params == 0``.
        """
        theta_n = jnp.asarray(theta_n)
        total = jnp.zeros((), dtype=jnp.float32)
        if self.n_params == 0:
            return total
        blocks = self.split(theta_n)
        if self.fit_sky:
            total = total + self._gaussian_logpdf(blocks["sky"], self.sky_prior_sigma)
        if self.fit_shifts:
            free_shifts = blocks["shifts"][jnp.asarray(self.shift_bands)]
            total = total + self._gaussian_logpdf(free_shifts, self.shift_prior_sigma)
        if self.fit_noise_scale:
            total = total + self._gaussian_logpdf(
                blocks["log_noise_scale"], self.noise_scale_prior_sigma
            )
        return total

    @staticmethod
    def _gaussian_logpdf(x: jnp.ndarray, sigma: float) -> jnp.ndarray:
        """Summed zero-mean Gaussian log-density over all entries of ``x``.

        Args:
            x: Array of parameter values.
            sigma: Standard deviation of the prior.

        Returns:
            Scalar sum of ``log N(x_i | 0, sigma^2)``.
        """
        n = x.size
        log_norm = n * (math.log(sigma) + _LOG_SQRT_2PI)
        return -0.5 * jnp.sum((x / sigma) ** 2) - log_norm

    def initial_theta(self) -> jnp.ndarray:
        """Default starting point: every nuisance parameter at its prior mean.

        Returns:
            Zero array of shape (n_params,).
        """
        return jnp.zeros(self.n_params, dtype=jnp.float32)

    def sample_prior(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw ``n`` nuisance vectors from the Gaussian prior.

        Args:
            key: ``jax.random`` PRNG key.
            n: Number of samples.

        Returns:
            Array of shape (n, n_params); shape ``(n, 0)`` when nothing is fitted.
        """
        if self.n_params == 0:
            return jnp.zeros((n, 0), dtype=jnp.float32)
        sigmas = jnp.concatenate(
            [
                jnp.full(self.n_sky, self.sky_prior_sigma, dtype=jnp.float32),
                jnp.full(self.n_shift, self.shift_prior_sigma, dtype=jnp.float32),
                jnp.full(self.n_noise, self.noise_scale_prior_sigma, dtype=jnp.float32),
            ]
        )
        return sigmas[None, :] * jax.random.normal(key, (n, self.n_params), dtype=jnp.float32)

    def __repr__(self) -> str:
        """Compact description of the enabled blocks."""
        return (
            f"NuisanceModel(n_bands={self.n_bands}, fit_sky={self.fit_sky}, "
            f"fit_shifts={self.fit_shifts}, shift_reference_band={self.shift_reference_band}, "
            f"fit_noise_scale={self.fit_noise_scale}, "
            f"n_params={self.n_params})"
        )
