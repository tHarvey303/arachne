r"""Metric Gaussian VI (MGVI) and geoVI for correlated-field fits, via NIFTy.re.

The Laplace approximation used so far expands the posterior at a single
Newton mode, so its widths depend on the curvature at one point and on where
the optimiser stopped (``docs/design/correlated_field_inference.md`` §2).
MGVI (Knollmüller & Enßlin 2019) and geoVI (Frank, Leike & Enßlin 2021)
instead fit a Gaussian (MGVI) or a Gaussian in the coordinates of the
posterior's Fisher-metric geometry (geoVI) whose covariance is the inverse
*Fisher metric averaged over its own samples*, alternating:

1. draw ``n_samples`` antithetic (mirrored) pairs of residuals around the
   current latent mean ``xi_bar`` from ``N(0, M(xi_bar)^-1)``,
   ``M = I + J^T N^-1 J`` (conjugate gradients, Hessian-free); geoVI then
   curves each linear sample non-linearly (a few Newton-CG steps);
2. minimise the sample-averaged energy ``KL(xi_bar) ~ mean_i H(xi_bar + r_i)``
   over ``xi_bar`` with Newton-CG, using the sample-averaged metric as the
   Hessian.

Everything is matrix-free (Jacobian-vector products only), so it scales to
d ~ 1e4-1e5 where the dense Laplace Hessian does not.

Both algorithms need two things this module provides:

* **A standard-normal prior.**  The sampling is done in the latents ``xi`` of
  :class:`~arachne.inference.standardise.StandardisingTransform`
  (``theta = T(xi)``), whose prior is exactly ``N(0, I)`` up to a constant.
* **A Gaussian likelihood with model-independent noise.**  ``jft.Gaussian``
  takes a fixed inverse covariance; it is ``mask / variance`` over the valid
  pixels (``mask > 0``) and the forward map is ``xi -> fm._model_image(T(xi))``
  on those pixels.  That equals arachne's
  :class:`~arachne.likelihood.gaussian.GaussianLikelihood` only when
  ``model_error_frac == 0`` and no ``log_noise_scale`` is fitted: with a
  fractional model-error floor or a fitted noise scale the variance depends
  on the parameters, the Fisher metric is not ``J^T N^-1 J`` and the
  ``log(var_eff)`` term would be missing.  Those configurations are refused
  (``NotImplementedError``); a ``jft.VariableCovarianceGaussian`` route could
  handle them later.

Non-Gaussian global priors (Normal / Student-t(2) on a bounded global) are
handled exactly by the transform (truncated quantile maps), so the KL that
NIFTy minimises is the true posterior KL up to a constant.

MGVI's samples are mirrored pairs ``xi_mean +- r_i``, so their mean in ``xi``
is exactly ``xi_mean``; geoVI curves each member of a pair separately, so its
samples are only approximately symmetric.  ``theta_mean = T(xi_mean)`` is *not*
the mean of ``theta_samples`` for the non-linearly mapped globals; average the
samples (or the derived maps) when a posterior mean is wanted.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import operator
import time
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from arachne.inference.standardise import StandardisingTransform

__all__ = ["MGVIResult", "run_mgvi", "mgvi_scale", "default_mgvi_kwargs"]


def _dtype() -> Any:
    """Float dtype the forward model runs in (float64 iff JAX x64 is enabled)."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


@dataclasses.dataclass
class MGVIResult:
    """Output of :func:`run_mgvi`.

    Attributes:
        theta_mean: ``(d,)`` ``T(xi_mean)``, the latent mean mapped to theta.
        theta_samples: ``(2 n_samples, d)`` final VI samples in theta
            (antithetic pairs about ``xi_mean`` mapped through ``T``; exactly
            mirrored for MGVI, curved separately for geoVI).
        xi_mean: ``(d,)`` latent mean.
        xi_samples: ``(2 n_samples, d)`` final VI samples in ``xi``.
        n_iterations: Number of VI iterations run.
        history: One dict per iteration with ``iteration``, ``kl`` (the
            sample-averaged energy NIFTy minimised, up to a constant),
            ``kl_newton_iterations``, ``neg_log_post_mean`` (``-log p(T(xi_mean)
            | d)`` from the forward model) and ``time_s`` (since the start).
        runtime_s: Wall time of the whole call.
        transform: The :class:`StandardisingTransform` used.
        mode: ``"geovi"`` or ``"mgvi"``.
        theta_extra_samples: ``(n_final_samples, d)`` extra linear (MGVI)
            samples at the final mean, or ``None``.
        xi_extra_samples: The same in ``xi``, or ``None``.
        jft_samples: The final ``nifty8.re.Samples`` (to restart NIFTy).
    """

    theta_mean: np.ndarray
    theta_samples: np.ndarray
    xi_mean: np.ndarray
    xi_samples: np.ndarray
    n_iterations: int
    history: list[dict]
    runtime_s: float
    transform: StandardisingTransform
    mode: str = "geovi"
    theta_extra_samples: np.ndarray | None = None
    xi_extra_samples: np.ndarray | None = None
    jft_samples: Any = None


def _merge(base: dict, override: dict | None) -> dict:
    """Recursive dict merge (``override`` wins; nested dicts merged)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def default_mgvi_kwargs(n_params: int) -> dict[str, dict]:
    """Default NIFTy.re solver settings for a problem of dimension ``n_params``.

    Follows the NIFTy demos' scaling (``delta = 1e-4``; CG stops when the
    quadratic's energy changes by less than ``delta * d / 10``) with iteration
    caps sized for d ~ 1e3-1e4: CG for the sample draws <= 300 iterations;
    geoVI's non-linear update <= 5 Newton-CG steps; the KL minimisation <= 20
    Newton-CG steps per VI iteration with <= 300 CG iterations each.  From a
    cold start (the prior mean, thousands of nats from the posterior) the first
    one or two VI iterations use all 20 steps; a cap of 10 left the toy test
    problem ~20 nats short of the mode after 4 iterations, 20-30 within ~1.

    Args:
        n_params: Latent dimension ``d``.

    Returns:
        ``{"draw_linear_kwargs": ..., "nonlinearly_update_kwargs": ...,
        "kl_kwargs": ...}`` in the format of ``nifty8.re.optimize_kl``.
    """
    delta = 1e-4
    absdelta = delta * n_params / 10.0
    return {
        "draw_linear_kwargs": {
            "cg_name": None,
            "cg_kwargs": {"absdelta": absdelta, "maxiter": 300},
        },
        "nonlinearly_update_kwargs": {
            "minimize_kwargs": {
                "name": None,
                "xtol": delta,
                "maxiter": 5,
                "cg_kwargs": {"name": None, "maxiter": 50},
            }
        },
        "kl_kwargs": {
            "minimize_kwargs": {
                "name": None,
                "xtol": delta,
                "absdelta": absdelta,
                "maxiter": 20,
                "cg_kwargs": {"name": None, "maxiter": 300},
            }
        },
    }


def _check_forward_model(fm: Any) -> None:
    """Refuse forward models whose Gaussian noise depends on the parameters."""
    if not (hasattr(fm, "_model_image") and hasattr(fm, "observation")):
        raise TypeError(
            "run_mgvi needs a single-grid ForwardModel (with _model_image and observation); "
            f"got {type(fm).__name__}"
        )
    lik = fm.likelihood
    if bool(getattr(lik, "_has_frac", False)) or np.any(
        np.asarray(getattr(lik, "model_error_frac", 0.0)) > 0
    ):
        raise NotImplementedError(
            "run_mgvi: model_error_frac > 0 makes the noise variance depend on the model "
            "(var_eff = variance + (f * model)^2); NIFTy's Gaussian needs a fixed noise "
            "covariance.  Build the forward model with model_error_frac=0 (or inflate the "
            "variance map by hand)."
        )
    nuis = getattr(fm, "nuisance", None)
    if nuis is not None and getattr(nuis, "fit_noise_scale", False):
        raise NotImplementedError(
            "run_mgvi: a fitted log_noise_scale makes the noise covariance parameter "
            "dependent; use NuisanceModel(fit_noise_scale=False)."
        )


def _build_likelihood(fm: Any, transform: StandardisingTransform, dtype):
    """``jft.Gaussian(data, N^-1).amend(xi -> model on valid pixels)``."""
    import nifty8.re as jft

    obs = fm.observation
    flux = np.asarray(obs.flux, np.float64).reshape(-1)
    var = np.asarray(obs.variance, np.float64).reshape(-1)
    mask = np.asarray(obs.mask, np.float64).reshape(-1)
    valid = np.flatnonzero(mask > 0)
    if valid.size == 0:
        raise ValueError("run_mgvi: no valid pixels (mask > 0)")
    inv_var = mask[valid] / (var[valid] + 1e-30)  # as GaussianLikelihood: mask r^2 / var
    data = jnp.asarray(flux[valid], dtype)
    inv_var_j = jnp.asarray(inv_var, dtype)
    inv_std_j = jnp.asarray(np.sqrt(inv_var), dtype)
    idx = jnp.asarray(valid)

    def forward(xi):
        return fm._model_image(transform.to_theta(xi)).reshape(-1)[idx]

    gauss = jft.Gaussian(
        data,
        noise_cov_inv=jax.tree_util.Partial(operator.mul, inv_var_j),
        noise_std_inv=jax.tree_util.Partial(operator.mul, inv_std_j),
    )
    domain = jft.ShapeWithDtype((transform.n_params,), dtype)
    return gauss.amend(forward, domain=domain), int(valid.size)


def run_mgvi(
    fm: Any,
    theta0,
    key: jax.Array,
    *,
    n_iterations: int = 6,
    n_samples: int = 4,
    mode: str = "geovi",
    n_final_samples: int | None = None,
    draw_linear_kwargs: dict | None = None,
    nonlinearly_update_kwargs: dict | None = None,
    kl_kwargs: dict | None = None,
    check_prior: bool = True,
    verbose: bool = True,
    print_fn: Callable = print,
) -> MGVIResult:
    """Fit a correlated-field forward model with geoVI or MGVI (``nifty8.re.optimize_kl``).

    Args:
        fm: :class:`~arachne.forward_model.pipeline.ForwardModel` with a
            :class:`~arachne.spatial.correlated_field.CorrelatedFieldMap`,
            ``model_error_frac == 0`` and no fitted noise scale (see the module
            docstring).  Sky and shift nuisances are fine.
        theta0: ``(d,)`` starting theta, e.g. ``model.initial_theta()`` or a
            MAP; mapped to ``xi`` with ``T^-1``.
        key: PRNG key.
        n_iterations: VI iterations (sample draw + KL minimisation each).
        n_samples: Antithetic sample *pairs* per iteration (NIFTy's
            ``n_samples``); the KL is averaged over ``2 n_samples`` samples.
        mode: ``"geovi"`` (``sample_mode="nonlinear_resample"``: linear samples
            curved by a non-linear update) or ``"mgvi"``
            (``"linear_resample"``).  Both redraw the samples every iteration.
        n_final_samples: If given, draw this many extra linear (MGVI) samples
            at the final mean (``ceil(n / 2)`` mirrored pairs, truncated to
            ``n``) for summarising maps.
        draw_linear_kwargs: Overrides merged into the defaults of
            :func:`default_mgvi_kwargs` (nested dicts are merged).
        nonlinearly_update_kwargs: As above, for geoVI's non-linear update.
        kl_kwargs: As above, for the KL minimisation.
        check_prior: Check numerically that the transform standardises the
            model's prior (cheap; guards against unsupported prior changes).
        verbose: Print one line per iteration.
        print_fn: Printer for those lines.

    Returns:
        :class:`MGVIResult`.

    Raises:
        ValueError: On an unknown ``mode`` or a failed prior check.
        NotImplementedError: For model-dependent noise (``model_error_frac >
            0`` or a fitted noise scale) or priors the transform cannot
            standardise.
    """
    import nifty8.re as jft

    if mode not in ("geovi", "mgvi"):
        raise ValueError(f"mode must be 'geovi' or 'mgvi', got {mode!r}")
    _check_forward_model(fm)
    t_start = time.time()
    dtype = _dtype()
    transform = StandardisingTransform(fm)
    d = transform.n_params
    key, k_check, k_vi, k_final = jax.random.split(key, 4)
    if check_prior:
        mismatch = transform.prior_mismatch(fm, k_check, n=4)
        tol = 1e-6 * max(d, 1) + 1e-6 if dtype == jnp.float64 else 1e-5 * d + 1e-2
        if not mismatch < tol:
            raise ValueError(
                f"StandardisingTransform does not standardise this model's prior "
                f"(log_prior(T(xi)) + log|J| - log N(xi) varies by {mismatch:.3g} nats > {tol:.3g})"
            )

    defaults = default_mgvi_kwargs(d)
    dl_kw = _merge(defaults["draw_linear_kwargs"], draw_linear_kwargs)
    nl_kw = _merge(defaults["nonlinearly_update_kwargs"], nonlinearly_update_kwargs)
    kl_kw = _merge(defaults["kl_kwargs"], kl_kwargs)

    lh, n_data = _build_likelihood(fm, transform, dtype)
    xi0 = transform.from_theta(jnp.asarray(np.asarray(theta0), dtype))
    if xi0.shape != (d,):
        raise ValueError(f"theta0 must have shape ({d},), got {tuple(np.shape(theta0))}")

    neg_log_post = jax.jit(lambda xi: -fm.log_posterior(transform.to_theta(xi)))
    history: list[dict] = []
    if verbose:
        print_fn(
            f"[{mode}] d={d}, {n_data} valid pixels, {n_iterations} iterations x "
            f"{2 * n_samples} samples, -log p(start)={float(neg_log_post(xi0)):.3f}"
        )

    def callback(samples, state):
        ms = state.minimization_state
        entry = {
            "iteration": int(state.nit),
            "kl": float(ms.fun),
            "kl_newton_iterations": int(ms.nit),
            "neg_log_post_mean": float(neg_log_post(samples.pos)),
            "time_s": time.time() - t_start,
        }
        history.append(entry)
        if verbose:
            print_fn(
                f"[{mode}] iter {entry['iteration']}/{n_iterations}: KL~{entry['kl']:.3f}, "
                f"-log p(mean)={entry['neg_log_post_mean']:.3f}, "
                f"{entry['kl_newton_iterations']} Newton steps, {entry['time_s']:.1f} s"
            )

    # NIFTy logs per-iteration residual statistics (INFO) and "Iteration Limit
    # Reached" (ERROR, from inside jitted solvers) on its own handler; silence
    # it unless verbose.
    jft_logger = logging.getLogger("nifty8.re.logger")
    old_level = jft_logger.level
    if not verbose:
        jft_logger.setLevel(logging.CRITICAL)
    try:
        samples, _state = jft.optimize_kl(
            lh,
            xi0,
            key=k_vi,
            n_total_iterations=int(n_iterations),
            n_samples=int(n_samples),
            draw_linear_kwargs=dl_kw,
            nonlinearly_update_kwargs=nl_kw,
            kl_kwargs=kl_kw,
            sample_mode="nonlinear_resample" if mode == "geovi" else "linear_resample",
            callback=callback,
        )
    finally:
        jft_logger.setLevel(old_level)

    xi_mean = samples.pos
    xi_samples = samples.samples if len(samples) else xi_mean[None]
    to_theta_batch = jax.jit(transform.to_theta)
    result = MGVIResult(
        theta_mean=np.asarray(transform.to_theta(xi_mean)),
        theta_samples=np.asarray(to_theta_batch(xi_samples)),
        xi_mean=np.asarray(xi_mean),
        xi_samples=np.asarray(xi_samples),
        n_iterations=int(n_iterations),
        history=history,
        runtime_s=0.0,
        transform=transform,
        mode=mode,
        jft_samples=samples,
    )

    if n_final_samples:
        n_extra = int(n_final_samples)
        n_pairs = math.ceil(n_extra / 2)
        draw = jax.jit(jax.vmap(partial(jft.draw_linear_residual, lh, **dl_kw), in_axes=(None, 0)))
        keys = jax.random.split(k_final, n_pairs)
        chunk = 8
        res = [draw(xi_mean, keys[i : i + chunk])[0] for i in range(0, n_pairs, chunk)]
        res = jnp.concatenate(res, axis=0)
        res = jnp.stack([res, -res], axis=1).reshape(-1, d)[:n_extra]
        xi_extra = xi_mean[None] + res
        result.xi_extra_samples = np.asarray(xi_extra)
        result.theta_extra_samples = np.asarray(to_theta_batch(xi_extra))

    result.runtime_s = time.time() - t_start
    if verbose:
        print_fn(f"[{mode}] done in {result.runtime_s:.1f} s")
    return result


def mgvi_scale(
    result: MGVIResult, *, include_extra: bool = True, floor: float = 1e-6
) -> np.ndarray:
    """Per-coordinate posterior standard deviation in ``xi`` from the VI samples.

    The samples come in antithetic pairs about ``xi_mean`` (exactly mirrored
    for MGVI), so the spread is measured about that centre rather than the
    sample mean.  Use ``scale**2`` as a diagonal inverse mass
    matrix for NUTS / MCLMC on a
    :class:`~arachne.inference.standardise.StandardisedLogDensity`.  With only
    a few sample pairs the per-coordinate estimate is noisy (it has
    ``n_pairs`` degrees of freedom); ask for ``n_final_samples`` to sharpen it.

    Args:
        result: Output of :func:`run_mgvi`.
        include_extra: Also use ``xi_extra_samples`` when present.
        floor: Lower bound on the returned scales.

    Returns:
        ``(d,)`` float64 array.
    """
    xs = [np.asarray(result.xi_samples, np.float64)]
    if include_extra and result.xi_extra_samples is not None:
        xs.append(np.asarray(result.xi_extra_samples, np.float64))
    resid = np.concatenate(xs, axis=0) - np.asarray(result.xi_mean, np.float64)[None]
    return np.maximum(np.sqrt(np.mean(resid**2, axis=0)), floor)
