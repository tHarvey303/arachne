"""Surface-brightness profiles for additive multi-component spatial models.

Each :class:`Profile` describes the *shape* of one additive component: how its
unit total flux is spread over the sky.  A profile owns

- a fixed number of shape parameters (``n_shape``) stored as a contiguous slice
  of the model's ``theta`` vector (``shape_raw``),
- the map from those raw values to physical quantities (:meth:`Profile.parse`),
- a normalised surface brightness ``I(y, x)`` with
  ``∫∫ I dy dx = 1`` over the whole plane (:meth:`Profile.surface_brightness`),
- a proper (normalised) log-prior and a matching sampler.

Coordinates ``(yy, xx)`` are passed in explicitly, so the same component can be
rendered on any grid — the image's own pixel grid, a finer sub-grid, or another
band's pixel grid expressed in arcsec offsets from a common reference point.
Nothing here knows about pixels except through the ``pixel_area`` argument of
:func:`render_on_grid`.

Implementations provide :meth:`Profile.log_surface_brightness`; the linear
version is its exponential.  Working in the log domain keeps components that
have wandered far outside the frame finite and lets the caller renormalise with
``logsumexp`` instead of dividing by a sum that has underflowed to zero.

Available profiles are collected in :data:`PROFILES` and resolved by
:func:`get_profile`.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)
_RHO_MAX = 0.99
# Floor on the squared elliptical radius of cuspy profiles.  exp(-b_n r^(1/n))
# is finite at r = 0 but its derivative is not, so a tiny floor (r >= 1e-3
# effective radii) keeps ``jax.grad`` finite at the exact centre.  The enclosed
# flux inside that radius is < 1e-4 even for n = 10, so the analytic
# normalisation is unaffected.
_R2_FLOOR = 1e-6


def _normal_logpdf(x: jnp.ndarray, loc: jnp.ndarray | float, sd: float) -> jnp.ndarray:
    """Normalised Gaussian log-density, elementwise."""
    sd = jnp.asarray(sd, dtype=jnp.float32)
    return -0.5 * ((x - loc) / sd) ** 2 - jnp.log(sd) - 0.5 * _LOG_2PI


def _soft_clip(x: jnp.ndarray, lo: float, hi: float, beta: float) -> jnp.ndarray:
    """Smooth clamp of ``x`` into ``(lo, hi)``; identity to ~1e-4 well inside."""
    x = lo + jax.nn.softplus(beta * (x - lo)) / beta
    return hi - jax.nn.softplus(beta * (hi - x)) / beta


def sersic_b(n: jnp.ndarray | float) -> jnp.ndarray:
    """Sersic ``b_n`` from the Ciotti & Bertin (1999) asymptotic expansion.

    ``b_n`` is defined by ``gamma(2n, b_n) = Gamma(2n) / 2``, i.e. it makes
    ``sigma`` the half-light (effective) radius.  The expansion is accurate to
    better than 1e-4 for ``n >= 0.3``.

    Args:
        n: Sersic index, scalar or array.

    Returns:
        ``b_n``, same shape as ``n``.
    """
    inv = 1.0 / jnp.asarray(n, dtype=jnp.float32)
    return (
        2.0 * jnp.asarray(n, dtype=jnp.float32)
        - 1.0 / 3.0
        + (4.0 / 405.0) * inv
        + (46.0 / 25515.0) * inv**2
        + (131.0 / 1148175.0) * inv**3
        - (2194697.0 / 30690717750.0) * inv**4
    )


class Profile:
    """Base class for normalised surface-brightness profiles.

    Subclasses define ``name``, ``n_shape``, ``shape_param_names`` and
    implement :meth:`parse`, :meth:`log_surface_brightness`, :meth:`log_prior`,
    :meth:`sample_prior` and :meth:`size_statistic`.

    All methods are pure JAX and safe under ``jit`` / ``grad`` / ``vmap``;
    ``shape_raw`` is always a 1-D float32 array of length ``n_shape``.

    Attributes:
        name: Registry key of the profile (e.g. ``"sersic"``).
        n_shape: Number of shape parameters in a component block.
        shape_param_names: Names of those parameters, in block order.
    """

    name: str = "base"
    n_shape: int = 0
    shape_param_names: tuple[str, ...] = ()

    def __repr__(self) -> str:
        """Short representation naming the profile."""
        return f"{type(self).__name__}(name={self.name!r}, n_shape={self.n_shape})"

    # ------------------------------------------------------------------ maps
    def parse(self, shape_raw: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Map a raw shape block to physical quantities.

        Args:
            shape_raw: Raw shape parameters, shape (n_shape,).

        Returns:
            Dict with at least ``mu`` (2,), ``sigma`` (2,) and ``rho`` (scalar);
            Sersic profiles add ``n``.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def log_surface_brightness(
        self, shape_raw: jnp.ndarray, yy: jnp.ndarray, xx: jnp.ndarray
    ) -> jnp.ndarray:
        """Log of the analytically normalised surface brightness at ``(yy, xx)``.

        Args:
            shape_raw: Raw shape parameters, shape (n_shape,).
            yy: Row coordinates, any shape, in the model's coordinate units.
            xx: Column coordinates, same shape as ``yy``.

        Returns:
            ``log I(y, x)``, same shape as ``yy``, with ``∫∫ I dy dx = 1``.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def surface_brightness(
        self, shape_raw: jnp.ndarray, yy: jnp.ndarray, xx: jnp.ndarray
    ) -> jnp.ndarray:
        """Analytically normalised surface brightness at ``(yy, xx)``.

        Normalised so that ``∫∫ I dy dx = 1`` over the whole plane (flux is
        *per unit coordinate area*, so multiply by the pixel area to get the
        flux fraction in a pixel — see :func:`render_on_grid`).

        Args:
            shape_raw: Raw shape parameters, shape (n_shape,).
            yy: Row coordinates, any shape, in the model's coordinate units.
            xx: Column coordinates, same shape as ``yy``.

        Returns:
            ``I(y, x)``, same shape as ``yy``.
        """
        return jnp.exp(self.log_surface_brightness(shape_raw, yy, xx))

    # ----------------------------------------------------------------- prior
    def log_prior(
        self,
        shape_raw: jnp.ndarray,
        centre: jnp.ndarray,
        centre_sigma: float,
        log_size_prior: tuple[float, float],
        rho_sigma: float,
    ) -> jnp.ndarray:
        """Normalised log-prior density of the shape block.

        Args:
            shape_raw: Raw shape parameters, shape (n_shape,).
            centre: Prior centre ``(y, x)`` in the model's coordinate units.
            centre_sigma: Sigma of the Gaussian prior on each centre coordinate.
            log_size_prior: ``(mean, sd)`` of the Gaussian prior on each log size.
            rho_sigma: Sigma of the Gaussian prior on ``atanh_rho``.

        Returns:
            Scalar log-density.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def sample_prior(
        self,
        key: jax.Array,
        centre: jnp.ndarray,
        centre_sigma: float,
        log_size_prior: tuple[float, float],
        rho_sigma: float,
        batch_shape: tuple[int, ...] = (),
    ) -> jnp.ndarray:
        """Draw shape blocks from the density of :meth:`log_prior`.

        Args:
            key: ``jax.random`` PRNG key.
            centre: Prior centre ``(y, x)``.
            centre_sigma: Sigma of the centre prior.
            log_size_prior: ``(mean, sd)`` of the log-size prior.
            rho_sigma: Sigma of the ``atanh_rho`` prior.
            batch_shape: Leading sample shape; ``()`` returns a single
                ``(n_shape,)`` block.

        Returns:
            Array of shape ``(*batch_shape, n_shape)``.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    # ------------------------------------------------------------- summaries
    def size_statistic(self, shape_raw: jnp.ndarray) -> jnp.ndarray:
        """Scalar log size used to order components compact-first.

        Args:
            shape_raw: Raw shape parameters, shape (n_shape,).

        Returns:
            Scalar ``log`` of the geometric-mean size.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError

    def centre(self, shape_raw: jnp.ndarray) -> jnp.ndarray:
        """Component centre ``(y, x)`` in the model's coordinate units.

        Args:
            shape_raw: Raw shape parameters, shape (n_shape,).

        Returns:
            Array of shape (2,).
        """
        return shape_raw[0:2]


class GaussianProfile(Profile):
    """Bivariate Gaussian, ``shape = (mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho)``.

    ``sigma = exp(log_sigma)`` per axis, ``rho = tanh(atanh_rho)`` clipped to
    +-0.99, and the analytic normalisation is
    ``1 / (2 pi sigma_y sigma_x sqrt(1 - rho^2))``.  This is exactly the
    parametrisation and prior used by ``AdditiveComponentModel`` before
    profiles were pluggable.
    """

    name = "gaussian"
    n_shape = 5
    shape_param_names = ("mu_y", "mu_x", "log_sigma_y", "log_sigma_x", "atanh_rho")

    def parse(self, shape_raw: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Map the raw block to ``mu`` (2,), ``sigma`` (2,) and ``rho``.

        Args:
            shape_raw: Raw shape parameters, shape (5,).

        Returns:
            Dict with keys ``mu``, ``sigma``, ``rho``.
        """
        return {
            "mu": shape_raw[0:2],
            "sigma": jnp.exp(shape_raw[2:4]),
            "rho": jnp.clip(jnp.tanh(shape_raw[4]), -_RHO_MAX, _RHO_MAX),
        }

    def log_surface_brightness(
        self, shape_raw: jnp.ndarray, yy: jnp.ndarray, xx: jnp.ndarray
    ) -> jnp.ndarray:
        """Log of the unit-flux bivariate Gaussian at ``(yy, xx)``.

        Args:
            shape_raw: Raw shape parameters, shape (5,).
            yy: Row coordinates.
            xx: Column coordinates, same shape as ``yy``.

        Returns:
            ``log I(y, x)``, same shape as ``yy``.
        """
        p = self.parse(shape_raw)
        z, one_m_r2 = _elliptical_quadratic(p["mu"], p["sigma"], p["rho"], yy, xx)
        log_norm = (
            _LOG_2PI + jnp.log(p["sigma"][0]) + jnp.log(p["sigma"][1]) + 0.5 * jnp.log(one_m_r2)
        )
        return -0.5 * z / one_m_r2 - log_norm

    def log_prior(
        self,
        shape_raw: jnp.ndarray,
        centre: jnp.ndarray,
        centre_sigma: float,
        log_size_prior: tuple[float, float],
        rho_sigma: float,
    ) -> jnp.ndarray:
        """Gaussian priors on ``mu``, ``log_sigma`` and ``atanh_rho``.

        Args:
            shape_raw: Raw shape parameters, shape (5,).
            centre: Prior centre ``(y, x)``.
            centre_sigma: Sigma of the centre prior.
            log_size_prior: ``(mean, sd)`` of the log-size prior.
            rho_sigma: Sigma of the ``atanh_rho`` prior.

        Returns:
            Scalar log-density.
        """
        ls_mu, ls_sd = log_size_prior
        return (
            jnp.sum(_normal_logpdf(shape_raw[0:2], centre, centre_sigma))
            + jnp.sum(_normal_logpdf(shape_raw[2:4], ls_mu, ls_sd))
            + _normal_logpdf(shape_raw[4], 0.0, rho_sigma)
        )

    def sample_prior(
        self,
        key: jax.Array,
        centre: jnp.ndarray,
        centre_sigma: float,
        log_size_prior: tuple[float, float],
        rho_sigma: float,
        batch_shape: tuple[int, ...] = (),
    ) -> jnp.ndarray:
        """Draw ``(mu, log_sigma, atanh_rho)`` from their Gaussian priors.

        Args:
            key: PRNG key.
            centre: Prior centre ``(y, x)``.
            centre_sigma: Sigma of the centre prior.
            log_size_prior: ``(mean, sd)`` of the log-size prior.
            rho_sigma: Sigma of the ``atanh_rho`` prior.
            batch_shape: Leading sample shape.

        Returns:
            Array of shape ``(*batch_shape, 5)``.
        """
        ls_mu, ls_sd = log_size_prior
        k_mu, k_ls, k_rho = jax.random.split(key, 3)
        f32 = jnp.float32
        mu = centre + centre_sigma * jax.random.normal(k_mu, (*batch_shape, 2), dtype=f32)
        log_sigma = ls_mu + ls_sd * jax.random.normal(k_ls, (*batch_shape, 2), dtype=f32)
        atanh_rho = rho_sigma * jax.random.normal(k_rho, (*batch_shape, 1), dtype=f32)
        return jnp.concatenate([mu, log_sigma, atanh_rho], axis=-1)

    def size_statistic(self, shape_raw: jnp.ndarray) -> jnp.ndarray:
        """Log geometric-mean sigma, ``0.5 * (log_sigma_y + log_sigma_x)``.

        Args:
            shape_raw: Raw shape parameters, shape (5,).

        Returns:
            Scalar.
        """
        return 0.5 * (shape_raw[2] + shape_raw[3])


class SersicProfile(Profile):
    """Sersic profile, ``shape = (mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho, log_n)``.

    The elliptical radius uses exactly the Gaussian covariance
    parametrisation, ``r^2 = [dy dx] Sigma^-1 [dy dx]^T`` with
    ``Sigma = [[sy^2, rho sy sx], [rho sy sx, sx^2]]``, so ``sigma`` is the
    effective (half-light) radius along each axis and ``n = 0.5`` reproduces a
    Gaussian of sigma ``sigma / sqrt(2 b_n)``.

    The surface brightness is ``I ∝ exp(-b_n (r^(1/n) - 1))`` with ``b_n`` from
    :func:`sersic_b`, normalised by the analytic total flux
    ``2 pi n e^{b_n} b_n^{-2n} Gamma(2n) sigma_y sigma_x sqrt(1 - rho^2)``.

    ``n = exp(log_n)`` is smoothly clamped into ``n_bounds`` (default
    ``[0.3, 10]``) by a softplus clamp of stiffness ``softness`` in ``log n``,
    rather than a hard ``clip``, so ``n`` keeps a useful gradient up to and at
    the wall (half the unclamped value there).  With the
    default ``softness = 40`` the map is the identity to float32 precision more
    than ~0.1 in ``log n`` inside the bounds, and the realised ``n`` right at a
    boundary is offset by ``log(2) / softness`` (n_min -> 0.305); use
    :meth:`parse` to read back the value actually used.

    Cuspy profiles (``n >= 2``) are badly under-resolved by a single sample per
    pixel; pass ``oversample >= 3`` to :func:`render_on_grid`.
    """

    name = "sersic"
    n_shape = 6
    shape_param_names = ("mu_y", "mu_x", "log_sigma_y", "log_sigma_x", "atanh_rho", "log_n")

    def __init__(
        self,
        log_n_mu: float = math.log(2.0),
        log_n_sd: float = 0.7,
        n_bounds: tuple[float, float] = (0.3, 10.0),
        softness: float = 40.0,
    ) -> None:
        """Initialise the Sersic profile.

        Args:
            log_n_mu: Mean of the Gaussian prior on ``log_n`` (default ``log 2``).
            log_n_sd: Sd of the Gaussian prior on ``log_n`` (default 0.7, which
                covers ``n ~ 0.5`` to ``n ~ 8`` at 2 sigma).
            n_bounds: ``(n_min, n_max)`` for the smooth clamp on ``n``.
            softness: Stiffness of the softplus clamp, in ``log n`` units.
        """
        self.log_n_mu = float(log_n_mu)
        self.log_n_sd = float(log_n_sd)
        self.n_bounds = (float(n_bounds[0]), float(n_bounds[1]))
        self.softness = float(softness)

    def __repr__(self) -> str:
        """Representation including the ``log_n`` prior and bounds."""
        return (
            f"SersicProfile(log_n_mu={self.log_n_mu:.4f}, log_n_sd={self.log_n_sd}, "
            f"n_bounds={self.n_bounds})"
        )

    def parse(self, shape_raw: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Map the raw block to ``mu``, ``sigma``, ``rho``, ``n`` and ``b_n``.

        Args:
            shape_raw: Raw shape parameters, shape (6,).

        Returns:
            Dict with keys ``mu``, ``sigma``, ``rho``, ``n``, ``b_n``.
        """
        lo, hi = self.n_bounds
        log_n = _soft_clip(shape_raw[5], math.log(lo), math.log(hi), self.softness)
        n = jnp.exp(log_n)
        return {
            "mu": shape_raw[0:2],
            "sigma": jnp.exp(shape_raw[2:4]),
            "rho": jnp.clip(jnp.tanh(shape_raw[4]), -_RHO_MAX, _RHO_MAX),
            "n": n,
            "b_n": sersic_b(n),
        }

    def log_surface_brightness(
        self, shape_raw: jnp.ndarray, yy: jnp.ndarray, xx: jnp.ndarray
    ) -> jnp.ndarray:
        """Log of the unit-flux Sersic profile at ``(yy, xx)``.

        Args:
            shape_raw: Raw shape parameters, shape (6,).
            yy: Row coordinates.
            xx: Column coordinates, same shape as ``yy``.

        Returns:
            ``log I(y, x)``, same shape as ``yy``.
        """
        p = self.parse(shape_raw)
        n, b = p["n"], p["b_n"]
        z, one_m_r2 = _elliptical_quadratic(p["mu"], p["sigma"], p["rho"], yy, xx)
        r2 = jnp.maximum(z / one_m_r2, _R2_FLOOR)
        r_pow = jnp.exp(0.5 * jnp.log(r2) / n)  # r ** (1 / n)
        log_norm = (
            _LOG_2PI
            + jnp.log(n)
            + b
            - 2.0 * n * jnp.log(b)
            + jax.scipy.special.gammaln(2.0 * n)
            + jnp.log(p["sigma"][0])
            + jnp.log(p["sigma"][1])
            + 0.5 * jnp.log(one_m_r2)
        )
        return -b * (r_pow - 1.0) - log_norm

    def log_prior(
        self,
        shape_raw: jnp.ndarray,
        centre: jnp.ndarray,
        centre_sigma: float,
        log_size_prior: tuple[float, float],
        rho_sigma: float,
    ) -> jnp.ndarray:
        """Gaussian priors on ``mu``, ``log_sigma``, ``atanh_rho`` and ``log_n``.

        Args:
            shape_raw: Raw shape parameters, shape (6,).
            centre: Prior centre ``(y, x)``.
            centre_sigma: Sigma of the centre prior.
            log_size_prior: ``(mean, sd)`` of the log-size prior.
            rho_sigma: Sigma of the ``atanh_rho`` prior.

        Returns:
            Scalar log-density.
        """
        ls_mu, ls_sd = log_size_prior
        return (
            jnp.sum(_normal_logpdf(shape_raw[0:2], centre, centre_sigma))
            + jnp.sum(_normal_logpdf(shape_raw[2:4], ls_mu, ls_sd))
            + _normal_logpdf(shape_raw[4], 0.0, rho_sigma)
            + _normal_logpdf(shape_raw[5], self.log_n_mu, self.log_n_sd)
        )

    def sample_prior(
        self,
        key: jax.Array,
        centre: jnp.ndarray,
        centre_sigma: float,
        log_size_prior: tuple[float, float],
        rho_sigma: float,
        batch_shape: tuple[int, ...] = (),
    ) -> jnp.ndarray:
        """Draw ``(mu, log_sigma, atanh_rho, log_n)`` from their Gaussian priors.

        Args:
            key: PRNG key.
            centre: Prior centre ``(y, x)``.
            centre_sigma: Sigma of the centre prior.
            log_size_prior: ``(mean, sd)`` of the log-size prior.
            rho_sigma: Sigma of the ``atanh_rho`` prior.
            batch_shape: Leading sample shape.

        Returns:
            Array of shape ``(*batch_shape, 6)``.
        """
        k_gauss, k_n = jax.random.split(key, 2)
        base = GaussianProfile().sample_prior(
            k_gauss, centre, centre_sigma, log_size_prior, rho_sigma, batch_shape
        )
        log_n = self.log_n_mu + self.log_n_sd * jax.random.normal(
            k_n, (*batch_shape, 1), dtype=jnp.float32
        )
        return jnp.concatenate([base, log_n], axis=-1)

    def size_statistic(self, shape_raw: jnp.ndarray) -> jnp.ndarray:
        """Log geometric-mean effective radius, ``0.5 * (log_sigma_y + log_sigma_x)``.

        Args:
            shape_raw: Raw shape parameters, shape (6,).

        Returns:
            Scalar.
        """
        return 0.5 * (shape_raw[2] + shape_raw[3])


class PointSourceProfile(Profile):
    """Unresolved point source (AGN / nuclear source), ``shape = (mu_y, mu_x)``.

    Because :meth:`surface_brightness` is evaluated on arbitrary coordinates
    rather than deposited into pixels, the point source is represented as a
    circular Gaussian of fixed width ``point_sigma``: unit flux spread over a
    kernel much narrower than the PSF, so that after PSF convolution the
    rendered component *is* the PSF centred on ``mu``, and a single sample per
    pixel already interpolates the flux onto the neighbouring pixels roughly as
    a bilinear deposit would.

    ``point_sigma`` is in the model's coordinate units (pixels, or arcsec when
    the model has a ``pixel_scale``) and must be much smaller than the PSF FWHM
    — the default of 0.5 pixel-equivalents is a good choice for well-sampled
    imaging.  Setting it much below ~0.4 pixels aliases the source onto a
    single pixel and makes the centre poorly constrained by gradients.  The
    default is a *pixel* width, so
    :class:`~arachne.spatial.additive.AdditiveComponentModel` builds a
    ``"point"`` component with ``point_sigma = 0.5 * pixel_scale`` in arcsec
    mode; construct the profile yourself to pin an absolute width.

    The profile has no size or ellipticity parameters, so the log-size and
    ``rho`` priors do not apply; only the centre prior contributes.
    """

    name = "point"
    n_shape = 2
    shape_param_names = ("mu_y", "mu_x")

    def __init__(self, point_sigma: float = 0.5) -> None:
        """Initialise the point-source profile.

        Args:
            point_sigma: Fixed Gaussian width in the model's coordinate units.

        Raises:
            ValueError: If ``point_sigma`` is not positive.
        """
        if float(point_sigma) <= 0.0:
            raise ValueError(f"point_sigma must be > 0, got {point_sigma}")
        self.point_sigma = float(point_sigma)

    def __repr__(self) -> str:
        """Representation including the fixed width."""
        return f"PointSourceProfile(point_sigma={self.point_sigma})"

    def parse(self, shape_raw: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Map the raw block to ``mu``, plus the fixed ``sigma`` and ``rho = 0``.

        Args:
            shape_raw: Raw shape parameters, shape (2,).

        Returns:
            Dict with keys ``mu``, ``sigma``, ``rho``.
        """
        s = jnp.asarray(self.point_sigma, dtype=jnp.float32)
        return {
            "mu": shape_raw[0:2],
            "sigma": jnp.stack([s, s]),
            "rho": jnp.zeros((), dtype=jnp.float32),
        }

    def log_surface_brightness(
        self, shape_raw: jnp.ndarray, yy: jnp.ndarray, xx: jnp.ndarray
    ) -> jnp.ndarray:
        """Log of the narrow unit-flux Gaussian kernel at ``(yy, xx)``.

        Args:
            shape_raw: Raw shape parameters, shape (2,).
            yy: Row coordinates.
            xx: Column coordinates, same shape as ``yy``.

        Returns:
            ``log I(y, x)``, same shape as ``yy``.
        """
        s = self.point_sigma
        dy = (yy - shape_raw[0]) / s
        dx = (xx - shape_raw[1]) / s
        return -0.5 * (dy**2 + dx**2) - _LOG_2PI - 2.0 * math.log(s)

    def log_prior(
        self,
        shape_raw: jnp.ndarray,
        centre: jnp.ndarray,
        centre_sigma: float,
        log_size_prior: tuple[float, float],
        rho_sigma: float,
    ) -> jnp.ndarray:
        """Gaussian prior on ``mu`` only (no size or ellipticity parameters).

        Args:
            shape_raw: Raw shape parameters, shape (2,).
            centre: Prior centre ``(y, x)``.
            centre_sigma: Sigma of the centre prior.
            log_size_prior: Ignored.
            rho_sigma: Ignored.

        Returns:
            Scalar log-density.
        """
        del log_size_prior, rho_sigma
        return jnp.sum(_normal_logpdf(shape_raw[0:2], centre, centre_sigma))

    def sample_prior(
        self,
        key: jax.Array,
        centre: jnp.ndarray,
        centre_sigma: float,
        log_size_prior: tuple[float, float],
        rho_sigma: float,
        batch_shape: tuple[int, ...] = (),
    ) -> jnp.ndarray:
        """Draw ``mu`` from its Gaussian prior.

        Args:
            key: PRNG key.
            centre: Prior centre ``(y, x)``.
            centre_sigma: Sigma of the centre prior.
            log_size_prior: Ignored.
            rho_sigma: Ignored.
            batch_shape: Leading sample shape.

        Returns:
            Array of shape ``(*batch_shape, 2)``.
        """
        del log_size_prior, rho_sigma
        return centre + centre_sigma * jax.random.normal(key, (*batch_shape, 2), dtype=jnp.float32)

    def size_statistic(self, shape_raw: jnp.ndarray) -> jnp.ndarray:
        """``log(point_sigma)``: constant, and smaller than any resolved component.

        Args:
            shape_raw: Raw shape parameters, shape (2,); unused.

        Returns:
            Scalar.
        """
        del shape_raw
        return jnp.asarray(math.log(self.point_sigma), dtype=jnp.float32)


def _elliptical_quadratic(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    rho: jnp.ndarray,
    yy: jnp.ndarray,
    xx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(z, 1 - rho^2)`` where ``r^2 = z / (1 - rho^2)`` is Mahalanobis."""
    dy = (yy - mu[0]) / sigma[0]
    dx = (xx - mu[1]) / sigma[1]
    z = dy**2 - 2.0 * rho * dy * dx + dx**2
    return z, 1.0 - rho**2


PROFILES: dict[str, type[Profile]] = {
    "gaussian": GaussianProfile,
    "sersic": SersicProfile,
    "point": PointSourceProfile,
}


def get_profile(name_or_instance: str | Profile) -> Profile:
    """Resolve a profile name or instance to a :class:`Profile` instance.

    Args:
        name_or_instance: A key of :data:`PROFILES` (``"gaussian"``,
            ``"sersic"``, ``"point"``) or an already-built :class:`Profile`,
            which is returned unchanged (so callers can tune e.g.
            ``SersicProfile(log_n_sd=0.3)``).

    Returns:
        A :class:`Profile` instance.

    Raises:
        ValueError: If the name is unknown.
        TypeError: If the argument is neither a string nor a Profile.
    """
    if isinstance(name_or_instance, Profile):
        return name_or_instance
    if isinstance(name_or_instance, str):
        try:
            return PROFILES[name_or_instance]()
        except KeyError:
            raise ValueError(
                f"Unknown profile {name_or_instance!r}; available: {sorted(PROFILES)}"
            ) from None
    raise TypeError(f"Expected a profile name or Profile instance, got {type(name_or_instance)}")


def log_render_on_grid(
    profile: Profile,
    shape_raw: jnp.ndarray,
    yy: jnp.ndarray,
    xx: jnp.ndarray,
    pixel_area: float | jnp.ndarray,
    oversample: int = 1,
) -> jnp.ndarray:
    """Log of the flux per pixel of ``profile`` on the grid ``(yy, xx)``.

    Same as ``log(render_on_grid(...))`` but computed entirely in the log
    domain, so a component far outside the frame stays finite instead of
    underflowing to zero.

    Args:
        profile: The profile to render.
        shape_raw: Raw shape parameters, shape (profile.n_shape,).
        yy: Row coordinates of the pixel centres, any shape.
        xx: Column coordinates of the pixel centres, same shape as ``yy``.
        pixel_area: Area of one pixel in the coordinate units squared.
        oversample: Number of sub-samples per pixel side (>= 1).

    Returns:
        ``log`` flux per pixel, same shape as ``yy``.

    Raises:
        ValueError: If ``oversample < 1``.
    """
    oversample = int(oversample)
    if oversample < 1:
        raise ValueError(f"oversample must be >= 1, got {oversample}")
    log_area = jnp.log(jnp.asarray(pixel_area, dtype=jnp.float32))
    if oversample == 1:
        return profile.log_surface_brightness(shape_raw, yy, xx) + log_area
    # Square pixels of side sqrt(pixel_area); sub-sample centres are offset by
    # ((i + 0.5) / oversample - 0.5) * side from the pixel centre.
    frac = jnp.asarray((np.arange(oversample) + 0.5) / oversample - 0.5, dtype=jnp.float32)
    offsets = frac * jnp.sqrt(jnp.asarray(pixel_area, dtype=jnp.float32))
    sub = jnp.stack(
        [
            profile.log_surface_brightness(shape_raw, yy + offsets[i], xx + offsets[j])
            for i in range(oversample)
            for j in range(oversample)
        ]
    )
    log_mean = jax.scipy.special.logsumexp(sub, axis=0) - math.log(oversample**2)
    return log_mean + log_area


def render_on_grid(
    profile: Profile,
    shape_raw: jnp.ndarray,
    yy: jnp.ndarray,
    xx: jnp.ndarray,
    pixel_area: float | jnp.ndarray,
    oversample: int = 1,
) -> jnp.ndarray:
    """Flux per pixel of ``profile`` on the grid ``(yy, xx)``.

    Evaluates :meth:`Profile.surface_brightness` at ``oversample x oversample``
    sub-positions inside each pixel, averages them, and multiplies by
    ``pixel_area``.  With ``oversample = 1`` this is exactly
    ``surface_brightness(...) * pixel_area``.  The pixels are assumed square
    with side ``sqrt(pixel_area)``.

    Summed over a grid that covers the whole plane the result tends to 1;
    flux that falls outside the grid is simply lost, which is the physically
    correct behaviour for a source near the edge of a cutout.

    Args:
        profile: The profile to render.
        shape_raw: Raw shape parameters, shape (profile.n_shape,).
        yy: Row coordinates of the pixel centres, any shape.
        xx: Column coordinates of the pixel centres, same shape as ``yy``.
        pixel_area: Area of one pixel in the coordinate units squared.
        oversample: Number of sub-samples per pixel side (>= 1).

    Returns:
        Flux per pixel, same shape as ``yy``.
    """
    return jnp.exp(log_render_on_grid(profile, shape_raw, yy, xx, pixel_area, oversample))
