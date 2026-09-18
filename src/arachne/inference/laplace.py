r"""Laplace approximation and whitened (preconditioned) sampling.

Resolved SED-fitting posteriors are *extremely* badly conditioned in the raw
``theta`` coordinates that :class:`~arachne.forward_model.pipeline.ForwardModel`
exposes.  On the two-component Sersic demo (62 parameters, 10 NIRCam bands,
160x160 pixels) the curvature eigenvalues at the mode span ``1e-1`` to
``4e7``: a per-band sky pedestal is measured to ~1e-3 nJy by 2.6e4 pixels
while ``logsfr_ratio_4`` is barely constrained at all.  The stiff directions
are also strongly *correlated* (component centre against size against mass
against the per-band registration offsets), so even an exact **diagonal**
metric leaves a condition number of ~2e7.  A diagonal-metric NUTS then adapts
a step size of ~1e-7, every trajectory saturates the tree-doubling cap, and
four chains come back with ``R-hat ~ 4000`` and ``ESS = 4``.

The cure is to sample the **whitened** coordinates

.. math::  \theta = \mu + L z, \qquad L L^{T} = \Sigma_\mathrm{Laplace}

with :math:`\Sigma_\mathrm{Laplace}` the inverse Hessian of
``-log_posterior`` at the mode :math:`\mu` (the Laplace approximation) and
:math:`L` its Cholesky factor.  If the posterior were Gaussian this makes the
target exactly ``N(0, I)``, so the identity metric is right, the step size is
of order one and trajectories are short.  Real posteriors are not Gaussian,
but the whitening still removes the eight orders of magnitude of scale
mismatch, and a **dense** metric adapted *inside* the whitened space mops up
the residual O(10-100) correlations of whatever ridge the Laplace
approximation missed (see :func:`run_whitened_nuts`).

Two details matter enough to be the default here:

* **Negative curvature is floored by magnitude, not by variance.**  Adam and
  even a modified-Newton polish leave a handful of directions with slightly
  negative curvature (a nonquadratic mass/dust ridge; the mode is not a
  perfect quadratic minimum).  Giving those directions "the widest variance
  the model allows" makes the Laplace covariance 10-100x *too wide* there,
  which in whitened coordinates means a direction of width 5e-4 instead of 1
  -- and the step size collapses to that width.  Measured on the demo: with a
  positive clip at ``1 / max_variance`` the whitened posterior still had a
  condition number of 4.4e4 (marginal sd 5.4e-4 to 23); replacing the clip by
  ``|lambda|`` (``use_abs_eigenvalues=True``, the default) brought that to
  350, because ``1/sqrt(|lambda|)`` is the scale on which the log posterior
  actually varies along such a direction, whether the curvature is positive
  or negative.
* **Float64.**  The Hessian of a 2.6e5-pixel likelihood is meaningless in
  float32 (the rounding error of the sum is a few tenths of a nat), so the
  eigen-decomposition is always done in ``numpy`` float64 even when JAX runs
  in float32.  The library never calls ``jax.config.update("jax_enable_x64")``
  itself: the caller decides, and everything here follows the dtype of the
  forward model's own arrays.

Typical use is a single call to :func:`run_whitened_nuts` with the MAP, a
PRNG key and the usual chain settings; it builds the Hessian, the whitening
and the sampler, and returns a
:class:`~arachne.inference.nuts_sampler.NUTSResult` whose samples and
diagnostics are back in ``theta`` coordinates.  When the whitening is wanted
for something else (a Fisher forecast, a proposal covariance, an initial mass
matrix), go through the three pieces instead: ``hess =
hessian_neg_log_post(fm, theta_map)``, ``cov, chol =
laplace_covariance(hess)``, ``white = WhitenedLogDensity(fm, theta_map,
chol)``, and move between the two coordinate systems with
``white.to_theta(z)`` and ``white.from_theta(theta)``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from arachne.inference.diagnostics import ess, split_rhat
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

__all__ = [
    "make_hessian_fn",
    "hessian_neg_log_post",
    "laplace_covariance",
    "WhitenedLogDensity",
    "laplace_whitening",
    "posterior_whitening",
    "run_whitened_nuts",
]

#: Largest variance (in raw ``theta`` units) a Laplace direction may be given.
#: The sigmoid-unconstrained SPS parameters have a prior width of order 1-3,
#: so 9 is "as wide as the prior allows".
DEFAULT_MAX_VARIANCE = 9.0


def _model_dtype(fm: Any) -> Any:
    """Float dtype the forward model runs in (float64 iff JAX x64 is enabled)."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def make_hessian_fn(fm: Any) -> Callable[[Any], np.ndarray]:
    """Build a reusable ``theta -> Hessian of -log_posterior`` callable.

    ``jax.hessian`` re-traces its argument on every call and tracing a
    resolved forward model costs tens of seconds, so build this **once** per
    forward model and call it at as many points as needed.

    Args:
        fm: Object exposing ``log_posterior(theta) -> scalar`` (a
            :class:`~arachne.forward_model.pipeline.ForwardModel`,
            a ``MultiResolutionForwardModel`` or a
            :class:`WhitenedLogDensity`).

    Returns:
        Callable mapping a parameter vector to the symmetrised Hessian of
        ``-log_posterior`` as a ``numpy`` float64 array of shape ``(d, d)``.
    """
    jitted = jax.jit(jax.hessian(lambda t: -fm.log_posterior(t)))

    def hessian_at(theta) -> np.ndarray:
        h = np.asarray(jitted(jnp.asarray(theta, _model_dtype(fm))), dtype=np.float64)
        return 0.5 * (h + h.T)

    return hessian_at


def hessian_neg_log_post(fm: Any, theta) -> np.ndarray:
    """Symmetrised Hessian of ``-log_posterior`` at ``theta``.

    Convenience wrapper around :func:`make_hessian_fn` for a single
    evaluation; use :func:`make_hessian_fn` when several points are needed.

    Args:
        fm: Object exposing ``log_posterior(theta) -> scalar``.
        theta: Parameter vector of shape ``(d,)``, usually the MAP.

    Returns:
        ``numpy`` float64 array of shape ``(d, d)``.
    """
    return make_hessian_fn(fm)(theta)


def laplace_covariance(
    hess: np.ndarray,
    *,
    max_variance: float = DEFAULT_MAX_VARIANCE,
    rel_floor: float = 1e-8,
    use_abs_eigenvalues: bool = True,
    jitter: float = 1e-10,
    return_info: bool = False,
):
    """Eigen-modified Laplace covariance ``H^-1`` and its Cholesky factor.

    The Hessian is diagonalised, its eigenvalues are modified so that the
    inverse exists and is positive definite, and the covariance is rebuilt as
    ``V diag(1 / lambda') V^T``.  Two modifications are applied:

    * ``use_abs_eigenvalues`` (default ``True``) replaces ``lambda`` by
      ``|lambda|``.  A direction of *negative* curvature at an imperfect mode
      is not a wide direction -- ``1 / sqrt(|lambda|)`` is still the scale on
      which the log posterior changes by ~1 nat along it, and using it keeps
      the whitened posterior near unit width.  Clipping such a direction to
      the largest allowed variance instead (``use_abs_eigenvalues=False``, the
      historical behaviour) inflates it by 1-2 orders of magnitude and
      collapses the sampler's step size; see the module docstring for the
      measured numbers.
    * Eigenvalues smaller in magnitude than
      ``max(|lambda|_max * rel_floor, 1 / max_variance)`` are raised to that
      floor, which caps any marginal variance at ``max_variance`` (of the
      order of the prior width) and keeps the Cholesky factor finite for
      genuinely unconstrained directions.

    Args:
        hess: Symmetric ``(d, d)`` Hessian of ``-log_posterior``.
        max_variance: Largest variance any direction may be given, in raw
            ``theta`` units.
        rel_floor: Relative eigenvalue floor, as a fraction of the largest
            ``|lambda|``; guards against numerically zero directions in a
            Hessian whose dynamic range is ~1e8.
        use_abs_eigenvalues: Use ``|lambda|`` instead of clipping negative
            eigenvalues (strongly recommended; see above).
        jitter: Relative jitter added to the diagonal before the Cholesky
            factorisation, as a fraction of ``trace(cov) / d``.
        return_info: Also return a dict of diagnostics.

    Returns:
        ``(cov, chol)``, or ``(cov, chol, info)`` when ``return_info`` is set.
        ``cov`` is the ``(d, d)`` covariance, ``chol`` its lower-triangular
        Cholesky factor (both ``numpy`` float64).  ``info`` holds
        ``eigenvalues`` (the raw ones), ``n_negative``, ``n_floored``,
        ``marginal_sd`` and ``condition_number`` (of the modified spectrum).

    Raises:
        ValueError: If ``hess`` is not a square 2-D array.
    """
    hess = np.asarray(hess, dtype=np.float64)
    if hess.ndim != 2 or hess.shape[0] != hess.shape[1]:
        raise ValueError(f"hess must be a square 2-D array; got shape {hess.shape}")
    d = hess.shape[0]
    evals, evecs = np.linalg.eigh(0.5 * (hess + hess.T))

    floor = max(float(np.abs(evals).max()) * rel_floor, 1.0 / max_variance)
    modified = np.maximum(np.abs(evals), floor) if use_abs_eigenvalues else np.clip(
        evals, floor, None
    )  # fmt: skip
    n_floored = int((modified > (np.abs(evals) if use_abs_eigenvalues else evals)).sum())

    cov = (evecs / modified) @ evecs.T
    cov = 0.5 * (cov + cov.T)
    chol = np.linalg.cholesky(cov + jitter * np.eye(d) * np.trace(cov) / d)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    n_negative = int((evals < 0).sum())
    logger.info(
        f"Laplace covariance: curvature eigenvalues [{evals.min():.3g}, {evals.max():.3g}], "
        f"{n_negative} negative, {n_floored} floored at {floor:.3g}; marginal sd "
        f"[{sd.min():.3g}, {sd.max():.3g}] in raw units"
    )
    if not return_info:
        return cov, chol
    info = {
        "eigenvalues": evals,
        "modified_eigenvalues": modified,
        "n_negative": n_negative,
        "n_floored": n_floored,
        "floor": float(floor),
        "marginal_sd": sd,
        "condition_number": float(modified.max() / modified.min()),
    }
    return cov, chol, info


class WhitenedLogDensity:
    """``log_posterior`` in whitened coordinates ``theta = mean + L z``.

    Quacks like a :class:`~arachne.forward_model.pipeline.ForwardModel` as far
    as the samplers are concerned: it exposes ``log_posterior``,
    ``n_params`` and ``spatial_model``, so
    :class:`~arachne.inference.nuts_sampler.NUTSSampler` (and the MCLMC
    sampler) can be pointed at it unchanged.  ``log_prob`` is an alias of
    ``log_posterior``.

    The Jacobian term ``log|L|`` is a constant and is dropped, so
    ``log_prob(z)`` differs from the true whitened log density by that
    constant.  Nothing in MCMC or in a Laplace/Hessian calculation cares;
    evidence estimators would, which is why nested sampling is *not* run
    through this class (it needs the prior, not a local reparameterisation).

    Attributes:
        forward_model: The wrapped model.
        mean: ``(d,)`` centre of the whitening (usually the MAP).
        chol: ``(d, d)`` lower-triangular whitening factor.
        n_params: ``d``.
        spatial_model: Passed through from the wrapped model.
    """

    def __init__(self, fm: Any, mean, chol) -> None:
        """Build the whitened density.

        Args:
            fm: Object exposing ``log_posterior(theta) -> scalar``.
            mean: ``(d,)`` centre, typically the MAP.
            chol: ``(d, d)`` lower-triangular Cholesky factor of the Laplace
                covariance, e.g. from :func:`laplace_covariance`.

        Raises:
            ValueError: If the shapes of ``mean`` and ``chol`` disagree.
        """
        dtype = _model_dtype(fm)
        self.forward_model = fm
        self.mean = jnp.asarray(np.asarray(mean), dtype)
        self.chol = jnp.asarray(np.asarray(chol), dtype)
        if self.mean.ndim != 1 or self.chol.shape != (self.mean.size, self.mean.size):
            raise ValueError(
                f"mean must be (d,) and chol (d, d); got {self.mean.shape} and {self.chol.shape}"
            )
        self.n_params = int(self.mean.size)
        self.spatial_model = getattr(fm, "spatial_model", None)

    def to_theta(self, z: jnp.ndarray) -> jnp.ndarray:
        """Map whitened ``z`` (``(d,)`` or ``(..., d)``) back to ``theta``."""
        z = jnp.asarray(z)
        if z.ndim == 1:
            return self.mean + self.chol @ z
        return self.mean + z @ self.chol.T

    def from_theta(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Map ``theta`` (``(d,)`` or ``(..., d)``) to whitened coordinates."""
        theta = jnp.asarray(theta)
        delta = theta - self.mean
        if delta.ndim == 1:
            return jax.scipy.linalg.solve_triangular(self.chol, delta, lower=True)
        solved = jax.scipy.linalg.solve_triangular(
            self.chol, delta.reshape(-1, self.n_params).T, lower=True
        )
        return solved.T.reshape(delta.shape)

    def log_posterior(self, z: jnp.ndarray) -> jnp.ndarray:
        """Log posterior density at whitened ``z`` (up to the constant ``log|L|``)."""
        return self.forward_model.log_posterior(self.mean + self.chol @ jnp.asarray(z))

    #: Alias, for callers that think in terms of a log density rather than a posterior.
    log_prob = log_posterior

    def __repr__(self) -> str:
        """One-line representation naming the wrapped model and the dimension."""
        inner = type(self.forward_model).__name__
        return f"WhitenedLogDensity(n_params={self.n_params}, model={inner})"


def laplace_whitening(
    fm: Any,
    theta_map,
    *,
    hess: np.ndarray | None = None,
    max_variance: float = DEFAULT_MAX_VARIANCE,
    use_abs_eigenvalues: bool = True,
) -> tuple[WhitenedLogDensity, dict]:
    """Hessian -> Laplace covariance -> :class:`WhitenedLogDensity`, in one call.

    Args:
        fm: Object exposing ``log_posterior(theta) -> scalar``.
        theta_map: ``(d,)`` mode to centre the whitening on.
        hess: Pre-computed Hessian of ``-log_posterior`` at ``theta_map``
            (saves a ~1 minute trace + evaluation when one is already in hand).
        max_variance: See :func:`laplace_covariance`.
        use_abs_eigenvalues: See :func:`laplace_covariance`.

    Returns:
        ``(whitened, info)`` with ``info`` as returned by
        :func:`laplace_covariance` plus the key ``"cov"``.
    """
    if hess is None:
        hess = hessian_neg_log_post(fm, theta_map)
    cov, chol, info = laplace_covariance(
        hess,
        max_variance=max_variance,
        use_abs_eigenvalues=use_abs_eigenvalues,
        return_info=True,
    )
    info["cov"] = cov
    info["chol"] = chol
    return WhitenedLogDensity(fm, theta_map, chol), info


def posterior_whitening(
    fm: Any,
    chains,
    *,
    prior_cov: np.ndarray | None = None,
    shrinkage: float | None = None,
    mean=None,
    jitter: float = 1e-10,
) -> tuple[WhitenedLogDensity, dict]:
    """Re-centre and re-scale the whitening on an *exploratory chain*.

    A Laplace approximation is a local quadratic fit at the mode.  When the
    posterior has a long curved ridge -- two components exchanging stellar
    mass against dust, say -- the curvature at the mode says nothing about how
    far the ridge extends, and a whitened chain still has to travel tens of
    units along it: exactly the "step size 0.03, every trajectory saturates
    the tree cap, R-hat 1.3" regime measured on the 62-parameter demo with a
    dense metric adapted from a 500-draw warmup.  The fix is to take a short
    (even badly converged) whitened run, estimate the covariance the chains
    actually explored, and whiten *with that* -- a second stage then starts
    from a metric that already knows the ridge.

    The empirical covariance is shrunk toward ``prior_cov`` (normally the
    Laplace covariance) as ``(1 - a) * Sigma_emp + a * prior_cov``.  This is
    not optional cosmetics: with ``d`` parameters and a few hundred correlated
    draws, ``Sigma_emp`` is rank-deficient along many directions and would
    whiten them to zero width.  The default ``a = d / (d + n_draws)`` is the
    usual Stan-style compromise.

    Args:
        fm: Object exposing ``log_posterior(theta)``.
        chains: Exploratory draws, ``(n_chains, n_samples, d)`` or
            ``(n_samples, d)``, in **theta** coordinates.
        prior_cov: Covariance to shrink toward (default: a diagonal matrix of
            the empirical variances, which only decorrelates).
        shrinkage: Weight ``a`` in ``[0, 1]`` (default ``d / (d + n_draws)``).
        mean: Centre for the new coordinates (default: the pooled mean of
            ``chains``, which is a better centre than the mode once the
            posterior is skewed).
        jitter: Relative Cholesky jitter (see :func:`laplace_covariance`).

    Returns:
        ``(whitened, info)``; ``info`` holds ``cov``, ``chol``, ``shrinkage``,
        ``n_draws``, ``mean`` and ``condition_number``.

    Raises:
        ValueError: If ``chains`` is not 2- or 3-dimensional.
    """
    arr = np.asarray(chains, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(
            f"chains must be (n_chains, n_samples, d) or (n_samples, d); got {arr.shape}"
        )
    flat = arr.reshape(-1, arr.shape[-1])
    n_draws, d = flat.shape
    centre = np.asarray(mean, dtype=np.float64) if mean is not None else flat.mean(axis=0)
    dev = flat - centre
    emp = dev.T @ dev / max(n_draws - 1, 1)
    base = np.diag(np.clip(np.diag(emp), 1e-30, None)) if prior_cov is None else np.asarray(
        prior_cov, dtype=np.float64
    )  # fmt: skip
    a = float(d / (d + n_draws)) if shrinkage is None else float(shrinkage)
    cov = (1.0 - a) * emp + a * base
    cov = 0.5 * (cov + cov.T)
    chol = np.linalg.cholesky(cov + jitter * np.eye(d) * np.trace(cov) / d)
    evals = np.linalg.eigvalsh(cov)
    logger.info(
        f"Posterior whitening from {n_draws} draws (shrinkage {a:.3f} toward the "
        f"{'Laplace' if prior_cov is not None else 'diagonal'} covariance): variance range "
        f"[{evals.min():.3g}, {evals.max():.3g}], condition number "
        f"{evals.max() / max(evals.min(), 1e-300):.3g}"
    )
    info = {
        "cov": cov,
        "chol": chol,
        "shrinkage": a,
        "n_draws": int(n_draws),
        "mean": centre,
        "condition_number": float(evals.max() / max(evals.min(), 1e-300)),
    }
    return WhitenedLogDensity(fm, centre, chol), info


def run_whitened_nuts(
    fm: Any,
    theta_map,
    rng_key,
    *,
    n_warmup: int = 1000,
    n_samples: int = 500,
    n_chains: int = 4,
    chain_jitter: float = 0.3,
    max_num_doublings: int = 10,
    target_accept_rate: float = 0.8,
    dense_mass_matrix: bool = True,
    whitened: WhitenedLogDensity | None = None,
    hess: np.ndarray | None = None,
    max_variance: float = DEFAULT_MAX_VARIANCE,
):
    """Run NUTS in whitened coordinates and return the result in ``theta`` space.

    The chains are sampled in ``z`` (see the module docstring), then mapped
    back through ``theta = mean + L z``; split-R-hat and bulk ESS are
    **recomputed in theta coordinates**, because those are the parameters the
    user cares about and an affine map does not leave R-hat invariant once
    chains are finite.  Everything else in the returned
    :class:`~arachne.inference.nuts_sampler.NUTSResult` (divergences, tree
    depth, step size, acceptance) refers to the whitened run, which is where
    the sampler actually lived; ``diagnostics["whitened"]`` records that.

    ``dense_mass_matrix=True`` (the default) adapts a full covariance metric
    *inside* the whitened space.  That is the combination that converged the
    62-parameter demo: the Laplace whitening removes the 1e8 condition number,
    and the dense adaptation then absorbs the residual O(100) correlations of
    the mass-exchange / mass-dust ridge that the Laplace approximation does
    not capture.  A dense metric costs ``d^2`` memory and one ``d x d``
    triangular solve per leapfrog step, which is nothing next to a
    2.6e5-pixel likelihood, but it needs a warmup long enough to estimate
    ``d(d+1)/2`` numbers -- use ``n_warmup >= 1000`` for ``d ~ 60``.

    Args:
        fm: Object exposing ``log_posterior(theta)``; the model to sample.
        theta_map: ``(d,)`` MAP estimate; the whitening centre and the start
            of every chain (chain 0 exactly, the others jittered).
        rng_key: JAX PRNG key.
        n_warmup: Window-adaptation steps per chain.
        n_samples: Draws per chain after warmup.
        n_chains: Number of chains (>= 4 for meaningful R-hat).
        chain_jitter: Start jitter in units of the whitened posterior sd
            (i.e. of a Laplace sigma).
        max_num_doublings: NUTS tree-doubling cap.
        target_accept_rate: Dual-averaging target.
        dense_mass_matrix: Adapt a dense (rather than diagonal) metric in the
            whitened space.
        whitened: Pre-built :class:`WhitenedLogDensity` (skips the Hessian).
        hess: Pre-computed Hessian at ``theta_map`` (skips its evaluation).
        max_variance: See :func:`laplace_covariance`.

    Returns:
        ``(result, whitened)``: a ``NUTSResult`` whose ``samples`` and
        ``chains`` are in ``theta`` coordinates, and the
        :class:`WhitenedLogDensity` that was sampled.
    """
    from arachne.inference.nuts_sampler import NUTSSampler

    if whitened is None:
        whitened, _info = laplace_whitening(fm, theta_map, hess=hess, max_variance=max_variance)
    sampler = NUTSSampler(
        whitened,
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept_rate=target_accept_rate,
        max_num_doublings=max_num_doublings,
        dense_mass_matrix=dense_mass_matrix,
    )
    z_init = jnp.zeros(whitened.n_params, dtype=whitened.mean.dtype)
    result = sampler.run(z_init, rng_key, n_chains=n_chains, chain_jitter=chain_jitter)

    chains = result.chains
    diagnostics = dict(result.diagnostics)
    diagnostics["whitened"] = True
    if chains is not None:
        chains = jax.vmap(whitened.to_theta)(chains)
        diagnostics["rhat"] = split_rhat(np.asarray(chains))
        diagnostics["ess"] = ess(np.asarray(chains))
    result = dataclasses.replace(
        result,
        samples=whitened.to_theta(result.samples),
        chains=chains,
        diagnostics=diagnostics,
    )
    if whitened.spatial_model is not None:
        result = dataclasses.replace(result, spatial_model=whitened.spatial_model)
    return result, whitened
