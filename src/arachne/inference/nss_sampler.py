"""Nested Slice Sampling (``blackjax.nss``) for arachne forward models.

Nested sampling gives the Bayesian evidence ``Z = ∫ L(θ) π(θ) dθ`` as well as
posterior samples, which is what makes it possible to compare
:class:`~arachne.spatial.additive.AdditiveComponentModel` fits with
different numbers of components ``K``.  It also does not need a starting
point: the live points are drawn from the prior, so the label / size /
mass multimodality that troubles gradient samplers is handled by
construction (each mode is populated from the start, at the cost of many
more likelihood evaluations).

The run happens in the *unconstrained* theta space:

- ``logprior_fn = forward_model.log_prior`` — a proper density in theta
  (sigmoid Jacobian → uniform in physical SPS space, Gaussians on the shape
  parameters), so ``logZ`` is meaningful and comparable across ``K``;
- ``loglikelihood_fn = forward_model.log_likelihood``;
- initial live points from ``spatial_model.sample_prior``.

The driver follows the compiled ``init`` / ``step`` pattern of
``scripts/fit_catalogue.py``: a Python ``while`` loop over jitted steps,
terminating when the live-point evidence contribution
``logZ_live - logZ`` drops below ``termination`` (or after ``max_steps``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.nuts_sampler import NUTSResult
from arachne.spatial.base import SpatialModel
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

_N_LOGZ_DRAWS = 100  # Monte-Carlo shrinkage draws used for logZ error / ESS


@dataclass
class NSSResult(NUTSResult):
    """Nested-sampling result: equal-weight posterior samples plus the evidence.

    ``samples`` holds **equal-weight resampled** theta vectors of shape
    ``(n_samples_out, n_params)`` (via ``blackjax.ns.utils.sample``), so
    ``get_parameter_map`` and every other :class:`NUTSResult` method work
    unchanged.  ``infos`` is the finalised ``blackjax`` ``NSInfo`` (all dead
    particles followed by the final live ones) or ``None`` after loading.

    Attributes:
        logZ: Log-evidence point estimate (mean over 100 shrinkage draws).
        logZ_err: Monte-Carlo standard deviation of ``logZ`` over those draws.
        ess: Kish effective sample size of the importance weights.
        n_steps: Number of outer NS iterations (each replaces ``num_delete`` points).
        n_dead: Total number of dead particles accumulated.
        log_weights: MC-mean log importance weight of every particle (dead then
            live), shape ``(n_dead + num_live,)``, or ``None``.
    """

    logZ: float = float("nan")
    logZ_err: float = float("nan")
    ess: float = float("nan")
    n_steps: int = 0
    n_dead: int = 0
    log_weights: np.ndarray | None = None

    def to_hdf5(self, path: str | Path) -> None:
        """Save samples (as :meth:`NUTSResult.to_hdf5`) plus the evidence summary.

        Adds root attributes ``sampler="nss"``, ``logZ``, ``logZ_err``, ``ess``,
        ``n_steps``, ``n_dead`` and, when available, the dataset
        ``diagnostics/log_weights``.

        Args:
            path: Output HDF5 file path.
        """
        super().to_hdf5(path)
        with h5py.File(Path(path), "a") as f:
            f.attrs["sampler"] = "nss"
            f.attrs["logZ"] = float(self.logZ)
            f.attrs["logZ_err"] = float(self.logZ_err)
            f.attrs["ess"] = float(self.ess)
            f.attrs["n_steps"] = int(self.n_steps)
            f.attrs["n_dead"] = int(self.n_dead)
            if self.log_weights is not None:
                diag = f.require_group("diagnostics")
                diag.create_dataset(
                    "log_weights", data=np.asarray(self.log_weights), compression="gzip"
                )

    @classmethod
    def from_hdf5(cls, path: str | Path, spatial_model: SpatialModel) -> "NSSResult":
        """Load an :class:`NSSResult` written by :meth:`to_hdf5`.

        Args:
            path: Path to the HDF5 file.
            spatial_model: Spatial model instance (needed for ``get_parameter_map``).

        Returns:
            NSSResult with samples, evidence attributes and log weights restored.
        """
        with h5py.File(path, "r") as f:
            samples = jnp.array(f["samples/theta"][()])
            attrs = f.attrs
            log_weights = None
            if "diagnostics" in f and "log_weights" in f["diagnostics"]:
                log_weights = np.asarray(f["diagnostics/log_weights"][()])
            return cls(
                samples=samples,
                infos=None,
                spatial_model=spatial_model,
                logZ=float(attrs.get("logZ", np.nan)),
                logZ_err=float(attrs.get("logZ_err", np.nan)),
                ess=float(attrs.get("ess", np.nan)),
                n_steps=int(attrs.get("n_steps", 0)),
                n_dead=int(attrs.get("n_dead", 0)),
                log_weights=log_weights,
            )


def _logz_from_weights(logw: jnp.ndarray) -> tuple[float, float]:
    """LogZ point estimate and MC std from an (N_particles, n_draws) log-weight matrix."""
    lw = jnp.nan_to_num(logw, nan=-jnp.inf, posinf=-jnp.inf)
    logz_draws = jax.scipy.special.logsumexp(lw, axis=0)  # (n_draws,)
    return float(jnp.mean(logz_draws)), float(jnp.std(logz_draws))


def _ess_from_weights(logw: jnp.ndarray) -> float:
    """Kish ESS ``exp(2 logsumexp(w) - logsumexp(2w))`` of the MC-mean log weights."""
    lw_mean = jnp.nan_to_num(logw, nan=-jnp.inf, posinf=-jnp.inf).mean(axis=-1)
    lw_mean = lw_mean - jnp.max(lw_mean)
    ls = jax.scipy.special.logsumexp(lw_mean)
    ls2 = jax.scipy.special.logsumexp(2.0 * lw_mean)
    return float(jnp.exp(2.0 * ls - ls2))


class NSSSampler:
    """Nested Slice Sampling driver around ``blackjax.nss``.

    Attributes:
        forward_model: ForwardModel providing ``log_prior`` / ``log_likelihood``
            and a spatial model with ``sample_prior``.
        num_live: Number of live points.
        num_inner_steps: Slice-sampling steps used to generate each replacement
            live point.  Default ``3 * n_params`` (blackjax recommends at least
            ``max(5, 2 * dim)``; fewer steps bias ``logZ`` upward).
        num_delete: Live points replaced per outer step (vectorised on GPU).
            Default ``max(1, num_live // 10)``.
        termination: Stop when ``logZ_live - logZ < termination``.
        n_samples_out: Number of equal-weight posterior samples to return.
        max_steps: Hard cap on outer steps (guards against a non-converging run).
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        num_live: int = 500,
        num_inner_steps: int | None = None,
        num_delete: int | None = None,
        termination: float = 1e-3,
        n_samples_out: int = 1000,
        max_steps: int = 100_000,
    ) -> None:
        """Initialise the sampler.

        Args:
            forward_model: Assembled ForwardModel.
            num_live: Number of live points (evidence error ~ 1/sqrt(num_live)).
            num_inner_steps: Slice steps per new live point; default ``3 * n_params``.
            num_delete: Points replaced per step; default ``max(1, num_live // 10)``.
            termination: Threshold on the remaining live-point evidence fraction
                ``logZ_live - logZ``.
            n_samples_out: Number of equal-weight posterior samples to produce.
            max_steps: Maximum outer NS iterations.
        """
        self.forward_model = forward_model
        n_params = int(forward_model.spatial_model.n_params)
        self.num_live = int(num_live)
        self.num_inner_steps = int(num_inner_steps) if num_inner_steps is not None else 3 * n_params
        self.num_delete = int(num_delete) if num_delete is not None else max(1, self.num_live // 10)
        self.termination = float(termination)
        self.n_samples_out = int(n_samples_out)
        self.max_steps = int(max_steps)
        if self.num_delete < 1 or self.num_delete >= self.num_live:
            raise ValueError(
                f"num_delete must be in [1, num_live); got {self.num_delete} with "
                f"num_live={self.num_live}"
            )

    def _initial_live_points(self, key: jax.Array, initial_theta) -> jnp.ndarray:
        n_params = int(self.forward_model.spatial_model.n_params)
        if initial_theta is None:
            try:
                live = self.forward_model.spatial_model.sample_prior(key, self.num_live)
            except NotImplementedError as e:
                raise NotImplementedError(
                    "NSSSampler needs prior-distributed initial live points but "
                    f"{type(self.forward_model.spatial_model).__name__} does not implement "
                    "sample_prior(key, n). Implement it, or pass initial_theta of shape "
                    "(num_live, n_params) drawn from the prior."
                ) from e
        else:
            live = jnp.asarray(initial_theta)
        live = jnp.asarray(live, dtype=jnp.float32)
        if live.ndim != 2 or live.shape[1] != n_params:
            raise ValueError(
                f"initial live points must have shape (num_live, {n_params}); got {live.shape}"
            )
        if live.shape[0] != self.num_live:
            logger.warning(
                f"initial_theta has {live.shape[0]} rows; overriding num_live={self.num_live}"
            )
            self.num_live = int(live.shape[0])
            if self.num_delete >= self.num_live:
                self.num_delete = max(1, self.num_live // 10)
        return live

    def run(self, rng_key: jax.Array, initial_theta: jnp.ndarray | None = None) -> NSSResult:
        """Run nested sampling to termination and return posterior samples + evidence.

        Args:
            rng_key: ``jax.random`` PRNG key.
            initial_theta: Optional initial live points of shape
                ``(num_live, n_params)``.  These **must** be prior draws for the
                evidence to be valid; by default they are taken from
                ``spatial_model.sample_prior``.

        Returns:
            :class:`NSSResult` with equal-weight ``samples`` of shape
            ``(n_samples_out, n_params)`` and evidence diagnostics.

        Raises:
            ImportError: If ``blackjax`` is not installed.
            NotImplementedError: If the spatial model cannot sample its prior and
                no ``initial_theta`` is given.
        """
        try:
            import blackjax
            from blackjax.ns.utils import finalise, log_weights, sample
        except ImportError as e:
            raise ImportError(
                "blackjax>=1.6.2 is required for nested sampling. "
                "Install with: pip install blackjax"
            ) from e

        fm = self.forward_model
        rng_key, init_key = jax.random.split(rng_key)
        live = self._initial_live_points(init_key, initial_theta)
        n_params = live.shape[1]

        logger.info(
            f"Starting NSS: num_live={self.num_live}, num_delete={self.num_delete}, "
            f"num_inner_steps={self.num_inner_steps}, termination={self.termination:g}, "
            f"n_params={n_params}"
        )

        algo = blackjax.nss(
            logprior_fn=fm.log_prior,
            loglikelihood_fn=fm.log_likelihood,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner_steps,
        )
        init_fn = jax.jit(algo.init)
        step_fn = jax.jit(algo.step)

        t0 = time.perf_counter()
        state = init_fn(live)
        jax.block_until_ready(state)
        logger.info(f"NSS init compiled + evaluated in {time.perf_counter() - t0:.1f}s")

        dead = []
        n_steps = 0
        t0 = time.perf_counter()
        while True:
            remaining = float(state.integrator.logZ_live) - float(state.integrator.logZ)
            if not (remaining > self.termination):  # also stops on NaN
                break
            if n_steps >= self.max_steps:
                logger.warning(
                    f"NSS hit max_steps={self.max_steps} with logZ_live - logZ = {remaining:.3g}"
                )
                break
            rng_key, step_key = jax.random.split(rng_key)
            state, dead_info = step_fn(step_key, state)
            dead.append(dead_info)
            n_steps += 1
            if n_steps % 50 == 0:
                logger.info(
                    f"NSS step {n_steps}: logZ={float(state.integrator.logZ):.3f}, "
                    f"logZ_live-logZ={remaining:.3g}, "
                    f"min logL={float(jnp.min(state.particles.loglikelihood)):.2f}, "
                    f"{time.perf_counter() - t0:.0f}s"
                )
        jax.block_until_ready(state)
        elapsed = time.perf_counter() - t0

        final_info = finalise(state, dead, update_info=False)
        rng_key, w_key, s_key = jax.random.split(rng_key, 3)
        logw = log_weights(w_key, final_info, shape=_N_LOGZ_DRAWS)  # (N_total, 100)
        logz, logz_err = _logz_from_weights(logw)
        ess = _ess_from_weights(logw)

        resampled = sample(s_key, final_info, shape=self.n_samples_out)
        samples = jnp.asarray(resampled.position, dtype=jnp.float32)  # (S, n_params)
        n_dead = n_steps * self.num_delete

        logger.info(
            f"NSS complete: {n_steps} steps, {n_dead} dead points, {elapsed:.1f}s; "
            f"logZ = {logz:.3f} +/- {logz_err:.3f}, ESS = {ess:.0f}, "
            f"samples {tuple(samples.shape)}"
        )

        return NSSResult(
            samples=samples,
            infos=final_info,
            spatial_model=fm.spatial_model,
            logZ=logz,
            logZ_err=logz_err,
            ess=ess,
            n_steps=n_steps,
            n_dead=n_dead,
            log_weights=np.asarray(jnp.nan_to_num(logw, nan=-np.inf).mean(axis=-1)),
        )
