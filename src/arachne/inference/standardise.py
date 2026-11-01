r"""Standardising (prior-whitening) transform for correlated-field forward models.

A :class:`~arachne.spatial.correlated_field.CorrelatedFieldMap` prior is
almost a standard normal already: the field coefficients are ``N(0, I)``, the
level offsets, hyperparameters and free couplings are Gaussian.  Only the
global SPS parameters are not: they live as ``raw = logit((x - lo) / (hi -
lo))`` with a density that is uniform (or Normal / Student-t) in the physical
value ``x``.  :class:`StandardisingTransform` is an elementwise, invertible
map ``theta = T(xi)`` such that ``xi ~ N(0, I)`` implies ``theta ~ prior``:

==================  ======================================================
block               ``theta_i = T_i(xi_i)``
==================  ======================================================
field coefficients  ``xi``
levels              ``level_loc + level_sigma * xi``
hyperparameters     ``hyper_loc + hyper_scale * xi``
free couplings      ``coupling_loc + coupling_sigma * xi``
uniform global      ``raw = logit(Phi(xi))`` (``x`` uniform within bounds)
Normal global       ``x = loc + scale * F^-1(F(a) + Phi(xi) (F(b) - F(a)))``
Student-t(2) global same, with the Student-t(2) CDF / quantile
nuisance block      ``sigma * xi`` (sky, shifts, log noise scale)
==================  ======================================================

For Normal / Student-t globals ``a`` and ``b`` are the bounds in units of the
prior scale, so the physical value follows the prior **truncated to the
bounds**.  The library's ``log_prior`` is the untruncated density restricted
to the bounds (no renormalisation), which differs from the truncated one by
the constant ``log(F(b) - F(a))``; the identity::

    fm.log_prior(T(xi)) + log|dT/dxi| = log N(xi | 0, I) + const

therefore holds *exactly* (to rounding), with no clipping kink.  The CDF
differences are evaluated from whichever tail is smaller, so neither tail
cancels in float32 (the same trick as
:func:`~arachne.spatial.correlated_field.marginal_from_gaussian`).  Gaussian
values are clipped to ``|xi| <= 8`` before the non-linear global maps, which
only matters for draws that never occur under ``N(0, 1)``.  In float32 a
truncated global whose value lies within ~1e-6 of the bounds width of a bound
(``|xi| >~ 5`` for a prior whose bound sits ~3 sigma from its centre) is
rounded onto the bound; the identity then holds to ~0.1 nat rather than to
rounding.  Run in float64 if such tails matter.

Sampling or variational inference in ``xi`` therefore sees a prior that is
exactly ``N(0, I)`` -- what NIFTy's MGVI / geoVI assume (see
:mod:`arachne.inference.mgvi`) and a well-scaled starting point for NUTS /
MCLMC (:class:`StandardisedLogDensity`).

The transform reads the theta layout through
:meth:`~arachne.spatial.correlated_field.CorrelatedFieldMap.split_theta`
(:class:`~arachne.spatial.correlated_field.ThetaBlocks`) and refuses blocks it
does not know; :meth:`StandardisingTransform.prior_mismatch` checks the
identity above numerically, so a change of the spatial model's prior is
caught rather than silently mis-standardised.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["StandardisingTransform", "StandardisedLogDensity", "standard_normal_logpdf"]

_LOG_2PI = math.log(2.0 * math.pi)
#: Gaussian-space clip for the non-linear global maps (as the copula marginals use).
_XI_MAX = 8.0
#: Physical values of non-uniform globals are kept this fraction of the bounds width inside.
_EDGE_FRAC = 1e-6

# Global kinds
_UNIFORM, _NORMAL, _STUDENTT = "uniform", "normal", "studentt"

#: ThetaBlocks fields this module knows how to standardise.
_AFFINE_BLOCKS = ("levels", "hyper", "coupling", "z")
_KNOWN_BLOCKS = (*_AFFINE_BLOCKS, "global_raw")


def _dtype() -> Any:
    """JAX's float dtype (float64 iff x64 is enabled); follows the forward model's dtype."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def standard_normal_logpdf(xi: jnp.ndarray) -> jnp.ndarray:
    """``log N(xi | 0, I)`` summed over the last axis.

    Args:
        xi: Array of shape ``(d,)`` or ``(..., d)``.

    Returns:
        Array of shape ``()`` or ``(...)``.
    """
    xi = jnp.asarray(xi)
    return -0.5 * jnp.sum(xi**2, axis=-1) - 0.5 * xi.shape[-1] * _LOG_2PI


def _std_cdf_tails(t: float, dist: str) -> tuple[float, float]:
    """``(F(t), 1 - F(t))`` of the standard Normal or Student-t(2), in float64."""
    if math.isinf(t):
        return (0.0, 1.0) if t < 0 else (1.0, 0.0)
    if dist == _NORMAL:
        return 0.5 * math.erfc(-t / math.sqrt(2.0)), 0.5 * math.erfc(t / math.sqrt(2.0))
    r = math.sqrt(2.0 + t * t)
    tail = 1.0 / (r * (r + abs(t)))  # P(T > |t|) without cancellation
    return (tail, 1.0 - tail) if t < 0 else (1.0 - tail, tail)


def _std_logpdf(t: jnp.ndarray, dist: str) -> jnp.ndarray:
    """Log density of the standard Normal / Student-t(2) (constants included)."""
    if dist == _NORMAL:
        return -0.5 * t**2 - 0.5 * _LOG_2PI
    return -1.5 * jnp.log(2.0 + t**2)  # Student-t(2): (2 + t^2)^(-3/2)


class _Global:
    """Static description of one global parameter's standardising map."""

    def __init__(self, index: int, name: str, lo: float, hi: float, spec: dict | None) -> None:
        self.index = int(index)  # position in the full theta
        self.name = name
        self.lo, self.hi = float(lo), float(hi)
        if spec is None:
            self.kind = _UNIFORM
            return
        dist = spec["dist"]
        if dist == _STUDENTT and float(spec.get("df", 0.0)) != 2.0:
            raise NotImplementedError(
                f"global {name!r}: only Student-t with df = 2 has a closed-form quantile; "
                f"got df = {spec.get('df')}"
            )
        if dist not in (_NORMAL, _STUDENTT):
            raise NotImplementedError(f"global {name!r}: prior {dist!r} is not supported")
        self.kind = dist
        self.loc = float(spec.get("loc", 0.0))
        self.scale = float(spec["scale"])
        a = (self.lo - self.loc) / self.scale
        b = (self.hi - self.loc) / self.scale
        self.cdf_lo, _ = _std_cdf_tails(a, dist)  # F(a): prior mass below the bounds
        _, self.sf_hi = _std_cdf_tails(b, dist)  # 1 - F(b): prior mass above the bounds
        self.mass = 1.0 - self.cdf_lo - self.sf_hi
        if not self.mass > 0.0:
            raise ValueError(f"global {name!r}: prior has no mass inside the bounds")
        self.log_mass = math.log(self.mass)


class StandardisingTransform:
    """Elementwise map ``theta = T(xi)`` that turns ``xi ~ N(0, I)`` into the prior.

    See the module docstring for the per-block maps.  Works on any
    :class:`~arachne.forward_model.pipeline.ForwardModel` (or anything with
    ``spatial_model`` and optional ``nuisance`` attributes) whose spatial model
    is a :class:`~arachne.spatial.correlated_field.CorrelatedFieldMap`.

    Attributes:
        n_params: Dimension ``d`` of ``theta`` and ``xi``.
        spatial_model: The correlated-field model.
        nuisance: The nuisance model, or ``None``.
    """

    def __init__(self, fm: Any) -> None:
        """Build the transform from a forward model's prior.

        Args:
            fm: Forward model exposing ``spatial_model`` (a
                ``CorrelatedFieldMap``) and optionally ``nuisance``.

        Raises:
            TypeError: If the spatial model is not a ``CorrelatedFieldMap``.
            NotImplementedError: For theta blocks, global priors or a nuisance
                prior this transform cannot standardise.
        """
        from arachne.spatial.correlated_field import CorrelatedFieldMap

        sm = fm.spatial_model
        if not isinstance(sm, CorrelatedFieldMap):
            raise TypeError(
                f"StandardisingTransform needs a CorrelatedFieldMap, got {type(sm).__name__}"
            )
        self.spatial_model = sm
        self.nuisance = getattr(fm, "nuisance", None)
        n_spatial = int(sm.n_params)
        n_nuis = 0 if self.nuisance is None else int(self.nuisance.n_params)
        self.n_params = n_spatial + n_nuis

        loc = np.zeros(self.n_params)
        scale = np.ones(self.n_params)
        idx = sm.split_theta(np.arange(n_spatial))
        unknown = [
            name
            for name in idx._fields
            if name not in _KNOWN_BLOCKS and np.asarray(getattr(idx, name)).size
        ]
        if unknown:
            raise NotImplementedError(
                f"StandardisingTransform does not know the theta block(s) {unknown} of "
                "CorrelatedFieldMap; extend arachne.inference.standardise"
            )

        def vec(attr, n, default):
            v = getattr(sm, attr, None)
            if v is None:
                return np.full(n, default, np.float64)
            v = np.asarray(v, np.float64).reshape(-1)
            return np.broadcast_to(v, (n,)).copy() if n else np.zeros(0)

        n_lev = np.asarray(idx.levels).size
        loc[np.asarray(idx.levels)] = vec("_level_loc", n_lev, 0.0)
        scale[np.asarray(idx.levels)] = vec("_level_sigma", n_lev, 1.0)
        n_h = np.asarray(idx.hyper).size
        loc[np.asarray(idx.hyper)] = vec("_hyper_loc", n_h, 0.0)
        scale[np.asarray(idx.hyper)] = vec("_hyper_scale", n_h, 1.0)
        n_c = np.asarray(idx.coupling).size
        loc[np.asarray(idx.coupling)] = vec("_coupling_loc", n_c, 0.0)
        scale[np.asarray(idx.coupling)] = vec("_coupling_sigma", n_c, 1.0)

        specs = dict(getattr(sm, "_global_prior_specs", {}) or {})
        self.globals: list[_Global] = []
        for i, name in zip(np.asarray(idx.global_raw), sm.global_names, strict=True):
            lo, hi = (float(v) for v in sm.param_bounds[name])
            self.globals.append(_Global(int(i), name, lo, hi, specs.get(name)))
        self._global_idx = np.asarray([g.index for g in self.globals], dtype=int)

        if self.nuisance is not None and n_nuis:
            from arachne.forward_model.nuisance import NuisanceModel

            nm = self.nuisance
            if type(nm).log_prior is not NuisanceModel.log_prior:
                raise NotImplementedError(
                    f"{type(nm).__name__} overrides log_prior; only the zero-mean Gaussian "
                    "NuisanceModel prior can be standardised"
                )
            sig = np.concatenate(
                [
                    np.full(nm.n_sky, nm.sky_prior_sigma),
                    np.full(nm.n_shift, nm.shift_prior_sigma),
                    np.full(nm.n_noise, nm.noise_scale_prior_sigma),
                ]
            )
            if sig.size != n_nuis:
                raise NotImplementedError(
                    "NuisanceModel layout is not [sky, shifts, log_noise_scale]; cannot "
                    "standardise its prior"
                )
            scale[n_spatial:] = sig

        # Globals are handled separately: identity placeholders in the affine part.
        loc[self._global_idx] = 0.0
        scale[self._global_idx] = 1.0
        if not np.all(scale > 0.0):
            raise ValueError("all prior scales must be positive")
        self._loc = loc
        self._scale = scale
        self._log_scale_sum = float(np.sum(np.log(scale)))

    # ------------------------------------------------------------------ helpers
    def _consts(self, dtype):
        return jnp.asarray(self._loc, dtype), jnp.asarray(self._scale, dtype)

    @staticmethod
    def _global_forward(g: _Global, xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``(raw, log|d raw / d xi|)`` for one global, elementwise over ``xi``."""
        w = jnp.clip(xi, -_XI_MAX, _XI_MAX)
        log_phi = -0.5 * w**2 - 0.5 * _LOG_2PI
        if g.kind == _UNIFORM:
            lp, lq = jax.scipy.special.log_ndtr(w), jax.scipy.special.log_ndtr(-w)
            return lp - lq, log_phi - lp - lq
        p = jax.scipy.special.ndtr(w)
        q = jax.scipy.special.ndtr(-w)
        lower = g.cdf_lo + g.mass * p  # F(t)
        upper = g.sf_hi + g.mass * q  # 1 - F(t)
        if g.kind == _NORMAL:
            t = jnp.where(
                lower < upper,
                jax.scipy.special.ndtri(jnp.minimum(lower, 0.5)),
                -jax.scipy.special.ndtri(jnp.minimum(upper, 0.5)),
            )
        else:
            t = (lower - upper) / jnp.sqrt(2.0 * lower * upper)
        width = g.hi - g.lo
        eps = _EDGE_FRAC * width
        x = jnp.clip(g.loc + g.scale * t, g.lo + eps, g.hi - eps)
        frac = (x - g.lo) / width
        raw = jnp.log(frac) - jnp.log1p(-frac)
        # d raw/dx = width / ((x - lo)(hi - x)); dx/dxi = scale * mass * phi(xi) / f(t)
        log_jac = (
            math.log(width)
            - jnp.log(x - g.lo)
            - jnp.log(g.hi - x)
            + math.log(g.scale)
            + g.log_mass
            + log_phi
            - _std_logpdf(t, g.kind)
        )
        return raw, log_jac

    @staticmethod
    def _global_inverse(g: _Global, raw: jnp.ndarray) -> jnp.ndarray:
        """``xi`` for one global from its ``raw`` value (inverse of :meth:`_global_forward`)."""
        ndtri = jax.scipy.special.ndtri
        if g.kind == _UNIFORM:
            s_lo, s_hi = jax.nn.sigmoid(raw), jax.nn.sigmoid(-raw)
            return jnp.where(raw < 0, ndtri(jnp.minimum(s_lo, 0.5)), -ndtri(jnp.minimum(s_hi, 0.5)))
        x = g.lo + (g.hi - g.lo) * jax.nn.sigmoid(raw)
        t = (x - g.loc) / g.scale
        if g.kind == _NORMAL:
            lower, upper = jax.scipy.special.ndtr(t), jax.scipy.special.ndtr(-t)
        else:
            r = jnp.sqrt(2.0 + t**2)
            tail = 1.0 / (r * (r + jnp.abs(t)))
            lower = jnp.where(t < 0, tail, 1.0 - tail)
            upper = jnp.where(t < 0, 1.0 - tail, tail)
        a = (lower - g.cdf_lo) / g.mass  # Phi(xi)
        b = (upper - g.sf_hi) / g.mass  # Phi(-xi)
        tiny = jnp.finfo(jnp.result_type(raw)).tiny
        a = jnp.clip(a, tiny, 0.5)
        b = jnp.clip(b, tiny, 0.5)
        return jnp.where(a < b, ndtri(a), -ndtri(b))

    # ------------------------------------------------------------------ public
    def to_theta(self, xi: jnp.ndarray) -> jnp.ndarray:
        """Map standard-normal latents to theta.

        Args:
            xi: Array of shape ``(d,)`` or ``(..., d)``.

        Returns:
            ``theta`` of the same shape, in the forward model's dtype.
        """
        dtype = _dtype()
        xi = jnp.asarray(xi, dtype)
        loc, scale = self._consts(dtype)
        theta = loc + scale * xi
        for g in self.globals:
            raw, _ = self._global_forward(g, xi[..., g.index])
            theta = theta.at[..., g.index].set(raw)
        return theta

    def from_theta(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Inverse of :meth:`to_theta`.

        Args:
            theta: Array of shape ``(d,)`` or ``(..., d)``.

        Returns:
            ``xi`` of the same shape, in the forward model's dtype.
        """
        dtype = _dtype()
        theta = jnp.asarray(theta, dtype)
        loc, scale = self._consts(dtype)
        xi = (theta - loc) / scale
        for g in self.globals:
            xi = xi.at[..., g.index].set(self._global_inverse(g, theta[..., g.index]))
        return xi

    def log_det_jacobian(self, xi: jnp.ndarray) -> jnp.ndarray:
        """``log|det dT/dxi|`` (the map is elementwise, so the Jacobian is diagonal).

        Args:
            xi: Array of shape ``(d,)`` or ``(..., d)``.

        Returns:
            Array of shape ``()`` or ``(...)``.
        """
        dtype = _dtype()
        xi = jnp.asarray(xi, dtype)
        total = jnp.full(xi.shape[:-1], self._log_scale_sum, dtype)
        for g in self.globals:
            _, lj = self._global_forward(g, xi[..., g.index])
            total = total + lj
        return total

    def prior_mismatch(self, fm: Any, key: jax.Array, n: int = 8, spread: float = 1.0) -> float:
        """Numerical check that ``T`` standardises ``fm``'s prior.

        Evaluates ``fm.log_prior(T(xi)) + log_det_jacobian(xi) - log N(xi | 0, I)``
        at ``n`` draws ``xi ~ N(0, spread^2 I)``; it must be constant.

        Args:
            fm: The forward model the transform was built from.
            key: PRNG key.
            n: Number of test points.
            spread: Scale of the test draws.

        Returns:
            Peak-to-peak range of the difference across the draws (nats).
        """
        xi = spread * jax.random.normal(key, (n, self.n_params), _dtype())
        diff = jax.vmap(
            lambda x: fm.log_prior(self.to_theta(x))
            + self.log_det_jacobian(x)
            - standard_normal_logpdf(x)
        )(xi)
        diff = np.asarray(diff, np.float64)
        return float(diff.max() - diff.min())

    def __repr__(self) -> str:
        """Short description."""
        kinds = {}
        for g in self.globals:
            kinds[g.kind] = kinds.get(g.kind, 0) + 1
        return f"StandardisingTransform(n_params={self.n_params}, globals={kinds})"


class StandardisedLogDensity:
    """``log_posterior`` in standardised coordinates ``theta = T(xi)``.

    Same duck-type interface as
    :class:`~arachne.inference.laplace.WhitenedLogDensity`, so
    :class:`~arachne.inference.nuts_sampler.NUTSSampler` and the MCLMC sampler
    can sample ``xi`` directly.  Unlike the Laplace whitening the Jacobian is
    *included* (it is not constant here)::

        log_posterior(xi) = fm.log_posterior(T(xi)) + log|dT/dxi|

    so the prior part is ``log N(xi | 0, I)`` up to a constant.  Pair it with a
    diagonal metric from :func:`arachne.inference.mgvi.mgvi_scale`
    (``inverse_mass_matrix = scale**2``).

    Attributes:
        forward_model: The wrapped model.
        transform: The :class:`StandardisingTransform`.
        n_params: ``d``.
        spatial_model: Passed through from the wrapped model.
        mean: ``(d,)`` zeros in the model dtype, the centre of the ``xi`` coordinates (the
            prior mean), present for ``WhitenedLogDensity`` compatibility (e.g.
            ``run_whitened_nuts(whitened=...)`` reads its dtype).  It is *not* a posterior
            centre: to start chains at a point, map it with :meth:`from_theta` (or pass
            theta-space ``initial_positions`` to ``run_whitened_nuts``).
    """

    def __init__(self, fm: Any, transform: StandardisingTransform | None = None) -> None:
        """Build the standardised density.

        Args:
            fm: Forward model with a ``CorrelatedFieldMap`` spatial model.
            transform: Pre-built transform (default: built from ``fm``).
        """
        self.forward_model = fm
        self.transform = transform if transform is not None else StandardisingTransform(fm)
        self.n_params = self.transform.n_params
        self.spatial_model = getattr(fm, "spatial_model", None)
        self.mean = jnp.zeros(self.n_params, _dtype())

    def to_theta(self, xi: jnp.ndarray) -> jnp.ndarray:
        """Map ``xi`` (``(d,)`` or ``(..., d)``) to ``theta``."""
        return self.transform.to_theta(xi)

    def from_theta(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Map ``theta`` (``(d,)`` or ``(..., d)``) to ``xi``."""
        return self.transform.from_theta(theta)

    def log_posterior(self, xi: jnp.ndarray) -> jnp.ndarray:
        """Log posterior density of ``xi`` (Jacobian included)."""
        xi = jnp.asarray(xi, _dtype())
        return self.forward_model.log_posterior(
            self.transform.to_theta(xi)
        ) + self.transform.log_det_jacobian(xi)

    #: Alias, for callers that think in terms of a log density rather than a posterior.
    log_prob = log_posterior

    def __repr__(self) -> str:
        """One-line representation naming the wrapped model and the dimension."""
        inner = type(self.forward_model).__name__
        return f"StandardisedLogDensity(n_params={self.n_params}, model={inner})"
