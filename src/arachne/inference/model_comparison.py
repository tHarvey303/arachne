"""Bayesian model comparison over the number of additive components ``K``.

Nested sampling returns the evidence ``logZ``, so the natural way to ask "does
this galaxy need a bulge as well as a disk?" is to run
:class:`~arachne.inference.nss_sampler.NSSSampler` once per candidate ``K`` and
compare the evidences.  :func:`compare_n_components` does exactly that and
returns one :class:`ModelComparisonRow` per ``K``;
:func:`bayes_factor_table` renders the log Bayes factors relative to the best
model with Kass & Raftery verdicts.

**The caller owns prior comparability.**  ``make_forward_model(K)`` must build
models that differ *only* in the number of components: identical observation,
PSF, emulator, parameter bounds, shape priors and nuisance setup.  The evidence
integrates the prior, so a wider bound or an extra nuisance parameter in one
model silently changes ``logZ`` and the Bayes factor then measures the prior
difference rather than the data's preference.

``make_forward_model(K)`` may return either a
:class:`~arachne.forward_model.pipeline.ForwardModel` or a
:class:`~arachne.forward_model.multires.MultiResolutionForwardModel`: the
evidence comes from ``log_prior`` / ``log_likelihood``, which both expose, and
the chi-squared columns go through the dispatching helpers in
:mod:`arachne.inference.posterior_predictive`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from arachne.forward_model.multires import MultiResolutionForwardModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.nss_sampler import NSSResult, NSSSampler
from arachne.inference.posterior_predictive import (
    chi2_reduced,
    chi2_reduced_samples,
    n_model_params,
)
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

#: Either flavour of forward model; ``compare_n_components`` accepts both.
AnyForwardModel = Union[ForwardModel, MultiResolutionForwardModel]

__all__ = [
    "ModelComparisonRow",
    "compare_n_components",
    "bayes_factor_table",
]


@dataclass
class ModelComparisonRow:
    """One model in an evidence comparison.

    Attributes:
        n_components: Number of additive components ``K``.
        logZ: Log-evidence from nested sampling.
        logZ_err: Monte-Carlo uncertainty on ``logZ``.
        ess: Kish effective sample size of the nested-sampling weights.
        n_params: Free parameters of this model (spatial + nuisance).
        chi2_red_map: Reduced chi-squared at the best-fit theta (the
            maximum-likelihood posterior sample, or the point returned by
            ``map_theta_fn``).
        chi2_red_median: Median over posterior samples of the per-sample
            reduced chi-squared — a typical fit quality rather than the best one.
        runtime_s: Wall-clock seconds spent in ``NSSSampler.run``.
        result: The full :class:`NSSResult` (samples, weights, diagnostics).
    """

    n_components: int
    logZ: float
    logZ_err: float
    ess: float
    n_params: int
    chi2_red_map: float
    chi2_red_median: float
    runtime_s: float
    result: NSSResult


def _chi2_stats(
    forward_model: AnyForwardModel,
    result: NSSResult,
    map_theta_fn: Callable[[AnyForwardModel], jnp.ndarray] | None,
    n_chi2_samples: int,
) -> tuple[float, float]:
    """Best-fit and median reduced chi-squared for one fitted model.

    Args:
        forward_model: The fitted ForwardModel or MultiResolutionForwardModel.
        result: Nested-sampling result with equal-weight ``samples``.
        map_theta_fn: Optional callable returning a best-fit theta; when
            ``None`` the maximum-likelihood posterior sample is used.
        n_chi2_samples: Number of posterior samples used for the median.

    Returns:
        Tuple ``(chi2_red_map, chi2_red_median)``.
    """
    samples = jnp.asarray(result.samples)
    if map_theta_fn is not None:
        theta_map = jnp.asarray(map_theta_fn(forward_model))
    else:
        loglike = jax.jit(jax.vmap(forward_model.log_likelihood))(samples)
        theta_map = samples[int(jnp.argmax(loglike))]
    chi2_map = chi2_reduced(forward_model, theta_map)

    chi2s = chi2_reduced_samples(forward_model, samples, n_max=int(n_chi2_samples))
    return chi2_map, float(np.median(chi2s))


def compare_n_components(
    make_forward_model: Callable[[int], AnyForwardModel],
    ks: Sequence[int],
    rng_key: jnp.ndarray,
    sampler_kwargs: dict | None = None,
    map_theta_fn: Callable[[AnyForwardModel], jnp.ndarray] | None = None,
    n_chi2_samples: int = 64,
) -> list[ModelComparisonRow]:
    """Run nested sampling for each ``K`` and collect the evidence comparison.

    Each ``K`` gets its own PRNG key split from ``rng_key``, so the comparison
    is reproducible and the models are independent.

    **Prior comparability is the caller's responsibility**: ``make_forward_model``
    must vary nothing but the component count (see the module docstring).

    Args:
        make_forward_model: ``K -> ForwardModel`` factory; a
            ``MultiResolutionForwardModel`` factory works identically.
        ks: Component counts to compare, e.g. ``(1, 2, 3)``.
        rng_key: ``jax.random`` PRNG key.
        sampler_kwargs: Extra keyword arguments for :class:`NSSSampler`
            (``num_live``, ``num_inner_steps``, ``termination``, ``max_steps``,
            ...).  The same settings are used for every ``K``.
        map_theta_fn: Optional ``forward model -> theta`` callable used for
            ``chi2_red_map``; defaults to the maximum-likelihood posterior
            sample.
        n_chi2_samples: Posterior samples used for the median reduced
            chi-squared.

    Returns:
        List of :class:`ModelComparisonRow`, one per entry of ``ks``, in the
        given order.

    Raises:
        ValueError: If ``ks`` is empty.
    """
    ks = list(ks)
    if not ks:
        raise ValueError("ks must contain at least one component count")
    sampler_kwargs = dict(sampler_kwargs or {})

    keys = jax.random.split(jnp.asarray(rng_key), len(ks))
    rows: list[ModelComparisonRow] = []
    for k, key in zip(ks, keys):
        logger.info(f"Model comparison: fitting K={k}")
        fm = make_forward_model(int(k))
        sampler = NSSSampler(fm, **sampler_kwargs)
        t0 = time.perf_counter()
        result = sampler.run(key)
        runtime = time.perf_counter() - t0
        chi2_map, chi2_med = _chi2_stats(fm, result, map_theta_fn, n_chi2_samples)
        rows.append(
            ModelComparisonRow(
                n_components=int(k),
                logZ=float(result.logZ),
                logZ_err=float(result.logZ_err),
                ess=float(result.ess),
                n_params=n_model_params(fm),
                chi2_red_map=chi2_map,
                chi2_red_median=chi2_med,
                runtime_s=float(runtime),
                result=result,
            )
        )
        logger.info(
            f"K={k}: logZ = {result.logZ:.2f} +/- {result.logZ_err:.2f}, "
            f"chi2_red(MAP) = {chi2_map:.2f}, {runtime:.1f}s"
        )
    return rows


def _verdict(ln_k: float) -> str:
    """Kass & Raftery (1995) verdict for a log Bayes factor against the best model.

    Their scale is stated in ``2 ln K``; the thresholds below are the same
    boundaries expressed in ``ln K`` (1, 3 and 5).

    Args:
        ln_k: ``logZ_best - logZ_model`` (>= 0).

    Returns:
        A short verdict string.
    """
    a = abs(ln_k)
    if not np.isfinite(a):
        return "undefined"
    if a < 1.0:
        return "inconclusive"
    if a < 3.0:
        return "positive"
    if a < 5.0:
        return "strong"
    return "very strong"


def bayes_factor_table(rows: Sequence[ModelComparisonRow]) -> str:
    """Format log Bayes factors relative to the highest-evidence model.

    ``ln K = logZ_best - logZ_model`` is reported for every row, with its
    uncertainty added in quadrature from the two ``logZ_err`` values and a
    Kass & Raftery verdict for how decisively the best model is preferred.
    A ``ln K`` smaller than its own uncertainty is reported as
    ``inconclusive (MC noise)`` — nested sampling's own error bar then swamps
    the model preference and the run needs more live points.

    Args:
        rows: Rows from :func:`compare_n_components`.

    Returns:
        A multi-line table (no trailing newline).

    Raises:
        ValueError: If ``rows`` is empty.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("rows must contain at least one model")

    logzs = np.array([r.logZ for r in rows], dtype=float)
    finite = np.isfinite(logzs)
    best_i = int(np.argmax(np.where(finite, logzs, -np.inf))) if finite.any() else 0
    best = rows[best_i]

    header = (
        f"{'K':>3}  {'n_par':>5}  {'logZ':>12}  {'+/-':>7}  {'lnK vs best':>12}  "
        f"{'+/-':>7}  {'ESS':>7}  {'chi2r MAP':>9}  {'chi2r med':>9}  {'time/s':>8}  verdict"
    )
    lines = [
        f"Bayes factors relative to K={best.n_components} "
        f"(logZ = {best.logZ:.2f} +/- {best.logZ_err:.2f})",
        header,
        "-" * len(header),
    ]
    for i, r in enumerate(rows):
        ln_k = best.logZ - r.logZ
        err = float(np.sqrt(best.logZ_err**2 + r.logZ_err**2))
        if i == best_i:
            verdict = "best"
        elif np.isfinite(ln_k) and np.isfinite(err) and ln_k <= err:
            verdict = "inconclusive (MC noise)"
        else:
            verdict = _verdict(ln_k)
        lines.append(
            f"{r.n_components:>3}  {r.n_params:>5}  {r.logZ:>12.3f}  {r.logZ_err:>7.3f}  "
            f"{ln_k:>12.3f}  {err:>7.3f}  {r.ess:>7.1f}  {r.chi2_red_map:>9.3f}  "
            f"{r.chi2_red_median:>9.3f}  {r.runtime_s:>8.1f}  {verdict}"
        )
    lines.append(
        "Verdicts follow Kass & Raftery (1995); they are only meaningful if every "
        "model used identical priors apart from K."
    )
    return "\n".join(lines)
