"""Per-parameter prior specifications for physical-space SPS parameters.

A prior *spec* is a small JSON-serialisable dict such as
``{"dist": "studentt", "df": 2, "loc": 0, "scale": 0.3}``.  This module owns
the default specs for the ParrotEmulator parameter set, spec validation,
merging of user overrides over the defaults, and the construction of
log-prior functions and samplers from a set of specs.

Supported ``dist`` values::

    uniform                                  (flat over the domain)
    loguniform                               (~1/x; requires bound_lo > 0)
    normal       loc, scale                  (truncated to the domain)
    studentt     df, loc, scale              (truncated to the domain)
    halfnormal   scale                       (intended for bound_lo ~ 0)
    exponential  scale  (= mean)             (intended for bound_lo ~ 0)
    lognormal    loc, scale                  (requires bound_lo > 0)

Normalisation contract
----------------------
Every sampler in this project confines each parameter to its emulator
training domain ``[lo, hi]`` (sigmoid transform for NUTS, ``-inf`` outside
the box for NSS).  The spec therefore only describes the *shape* of the
density over that domain: priors whose support extends beyond the bounds
are implicitly truncated and the truncation normalisation is **dropped**.
Furthermore ``halfnormal``, ``exponential`` and ``lognormal`` omit their
untruncated normalising constants as well, while ``normal`` and
``studentt`` include theirs. These constants affect absolute log-density and
ELBO values. Catalogue NSS measures evidence relative to the normalised
bounded prior by drawing its initial live points from it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "DEFAULT_PRIORS",
    "SUPPORTED_DISTS",
    "build_component_log_prior",
    "build_log_prior",
    "build_prior_sampler",
    "default_prior",
    "log_prior_1d",
    "prior_config_template",
    "resolve_prior_specs",
    "sigmoid_log_jacobian",
    "validate_prior_spec",
]

_LOG2PI: float = math.log(2.0 * math.pi)

SUPPORTED_DISTS: frozenset[str] = frozenset(
    {
        "uniform",
        "loguniform",
        "normal",
        "studentt",
        "halfnormal",
        "exponential",
        "lognormal",
    }
)

# Default priors for the v6-library parameter set (other known parameters: see
# arachne.emulator.spec.KNOWN_PARAMS, used by default_prior).  The SPS grid used to
# train the emulator drew logsfr_ratio_* from Student-t(df=2, scale=0.3); a
# flat prior on them is both wrong and pushes the emulator out of domain.
DEFAULT_PRIORS: dict[str, dict] = {
    "redshift": {"dist": "uniform"},
    "log_mass": {"dist": "uniform"},
    "slope": {"dist": "uniform"},
    "fesc_lya": {"dist": "uniform"},
    "dust_bump_amplitude": {"dist": "uniform"},
    "log10metallicity": {"dist": "uniform"},
    "Av": {"dist": "uniform"},
    "logsfr_ratio_0": {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
    "logsfr_ratio_1": {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
    "logsfr_ratio_2": {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
    "logsfr_ratio_3": {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
    "logsfr_ratio_4": {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
}

_UNIFORM: dict = {"dist": "uniform"}


def default_prior(name: str, defaults: dict[str, dict] | None = None) -> dict:
    """Default prior spec for parameter ``name`` (a copy).

    Looked up in ``defaults`` (e.g. ``emulator.spec.priors``), then
    :data:`DEFAULT_PRIORS`, then the registry of known library parameters
    (:func:`arachne.emulator.spec.param_info`); uniform if none has it.
    """
    if defaults and name in defaults:
        return dict(defaults[name])
    if name in DEFAULT_PRIORS:
        return dict(DEFAULT_PRIORS[name])
    from arachne.emulator.spec import param_info

    return dict(param_info(name).prior or _UNIFORM)


# ---------------------------------------------------------------------------
# Validation / resolution
# ---------------------------------------------------------------------------


def validate_prior_spec(name: str, spec: dict, bounds: tuple[float, float]) -> None:
    """Raise ValueError if ``spec`` is not a valid prior for a parameter on ``bounds``.

    Args:
        name: Parameter name, used only in error messages.
        spec: Prior spec dict.  A missing ``"dist"`` key means uniform.
        bounds: ``(lo, hi)`` domain of the parameter.

    Raises:
        ValueError: On an unknown ``dist``, non-finite distribution parameters,
            a non-positive ``scale`` or ``df``, or a ``loguniform`` / ``lognormal``
            prior on a domain with ``lo <= 0``.
    """
    dist = spec.get("dist", "uniform").lower()
    lo, hi = bounds
    if dist not in SUPPORTED_DISTS:
        raise ValueError(
            f"{name}: unknown prior dist {dist!r}. Use one of {sorted(SUPPORTED_DISTS)}."
        )
    if dist in ("normal", "studentt", "halfnormal", "exponential", "lognormal"):
        scale = float(spec.get("scale", 0.0))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{name}: {dist} requires scale > 0 and finite.")
    if dist == "studentt":
        df = float(spec.get("df", 0.0))
        if not math.isfinite(df) or df <= 0.0:
            raise ValueError(f"{name}: studentt requires df > 0 and finite.")
    if dist in ("normal", "studentt", "lognormal"):
        if not math.isfinite(float(spec.get("loc", 0.0))):
            raise ValueError(f"{name}: {dist} requires finite loc.")
    if dist in ("loguniform", "lognormal") and lo <= 0.0:
        raise ValueError(f"{name}: {dist} requires the lower bound > 0 (bound_lo={lo}).")
    if dist in ("halfnormal", "exponential") and lo < 0.0:
        print(
            f"  ! {name}: {dist} prior on a domain that includes negatives "
            f"(bound_lo={lo}); shape may be unintended."
        )


def resolve_prior_specs(
    param_names: Sequence[str],
    user_priors: dict[str, dict] | None,
    bounds: dict[str, tuple[float, float]],
    defaults: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Merge user prior overrides over the default priors for ``param_names``.

    Defaults come from :func:`default_prior` (``defaults``, then
    :data:`DEFAULT_PRIORS`, then the known-parameter registry, else uniform).
    Every resolved spec is validated against ``bounds[name]``.

    Args:
        param_names: Parameters to resolve, in order.
        user_priors: ``{name: spec}`` overrides (e.g. ``config["priors"]``), or None.
        bounds: ``{name: (lo, hi)}`` for every name in ``param_names``.
        defaults: ``{name: spec}`` default priors that take precedence over the
            built-in ones, e.g. ``emulator.spec.priors``.

    Returns:
        ``{name: spec}`` with one (copied) spec per parameter, in ``param_names`` order.

    Raises:
        ValueError: If ``user_priors`` names a parameter not in ``param_names``,
            or any resolved spec fails :func:`validate_prior_spec`.
    """
    names = list(param_names)
    priors: dict[str, dict] = {p: default_prior(p, defaults) for p in names}
    user = user_priors or {}
    for k, v in user.items():
        if k not in names:
            raise ValueError(f"priors: unknown parameter {k!r}. Valid: {names}")
        priors[k] = dict(v)
    for p in names:
        validate_prior_spec(p, priors[p], bounds[p])
    return priors


def prior_config_template(
    param_names: Sequence[str],
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict]:
    """Return the ``"priors"`` block of a config template: defaults for ``param_names``.

    Args:
        param_names: Parameters to include, in order.
        bounds: Optional ``{name: (lo, hi)}``; if given, each default is validated.

    Returns:
        ``{name: spec}`` with the default spec (uniform if none is registered).
    """
    template = {p: default_prior(p) for p in param_names}
    if bounds is not None:
        for p, spec in template.items():
            validate_prior_spec(p, spec, bounds[p])
    return template


# ---------------------------------------------------------------------------
# Log-densities
# ---------------------------------------------------------------------------


def _make_log_prior_1d(spec: dict, lo: float, hi: float) -> Callable:
    """Return ``fn(x) -> scalar`` for one spec, with Python-side constants precomputed.

    See the module docstring for the (deliberately partial) normalisation.
    ``lo`` / ``hi`` are accepted for interface symmetry but unused: truncation
    to the domain is enforced by the sampler, not by the density.
    """
    dist = spec.get("dist", "uniform").lower()
    if dist == "uniform":
        return lambda x: jnp.zeros((), dtype=x.dtype)
    if dist == "loguniform":
        return lambda x: -jnp.log(x)
    if dist == "normal":
        loc = float(spec.get("loc", 0.0))
        scale = float(spec["scale"])
        return lambda x: -0.5 * _LOG2PI - math.log(scale) - 0.5 * ((x - loc) / scale) ** 2
    if dist == "studentt":
        df = float(spec["df"])
        loc = float(spec.get("loc", 0.0))
        scale = float(spec["scale"])
        c = (
            math.lgamma(0.5 * (df + 1.0))
            - math.lgamma(0.5 * df)
            - 0.5 * math.log(df * math.pi)
            - math.log(scale)
        )
        return lambda x: c - 0.5 * (df + 1.0) * jnp.log1p(((x - loc) / scale) ** 2 / df)
    if dist == "halfnormal":
        scale = float(spec["scale"])
        return lambda x: -0.5 * (x / scale) ** 2
    if dist == "exponential":
        scale = float(spec["scale"])
        return lambda x: -x / scale
    if dist == "lognormal":
        loc = float(spec.get("loc", 0.0))
        scale = float(spec["scale"])
        return lambda x: -jnp.log(x) - 0.5 * ((jnp.log(x) - loc) / scale) ** 2
    raise ValueError(f"unhandled dist {dist!r}")


def log_prior_1d(spec: dict, x: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """Scalar log-density of one prior spec at physical value ``x``.

    Normalisation follows the module docstring: ``normal`` and ``studentt``
    carry their untruncated normalising constants; ``uniform`` is 0;
    ``loguniform``, ``halfnormal``, ``exponential`` and ``lognormal`` omit
    their constants; no truncation correction is ever applied.

    Args:
        spec: Prior spec dict.
        x: Physical value (scalar JAX array).
        lo: Lower bound of the domain (unused; see :func:`_make_log_prior_1d`).
        hi: Upper bound of the domain (unused).

    Returns:
        Scalar log-density (same dtype as ``x``).
    """
    return _make_log_prior_1d(spec, lo, hi)(x)


def build_log_prior(
    param_names: Sequence[str],
    specs: dict[str, dict],
    bounds: dict[str, tuple[float, float]],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build ``log_prior(x_phys) -> scalar`` summing per-parameter log-densities.

    Args:
        param_names: Parameter order of the physical vector.
        specs: ``{name: spec}`` covering every name in ``param_names``.
        bounds: ``{name: (lo, hi)}`` covering every name in ``param_names``.

    Returns:
        A jit/grad-safe function of a ``(N,)`` physical parameter vector.
    """
    fns = [_make_log_prior_1d(specs[p], bounds[p][0], bounds[p][1]) for p in param_names]

    def log_prior(x: jnp.ndarray) -> jnp.ndarray:
        total = jnp.zeros((), dtype=x.dtype)
        for i, fn in enumerate(fns):
            total = total + fn(x[i])
        return total

    return log_prior


def _prior_quantile(name: str, spec: dict, lo: float, hi: float) -> Callable:
    """Return the inverse CDF for the corresponding log-prior shape on [lo, hi]."""
    dist = spec.get("dist", "uniform").lower()
    if dist == "uniform":
        return lambda u: lo + (hi - lo) * u
    if dist == "loguniform":
        log_lo, log_hi = math.log(lo), math.log(hi)
        return lambda u: np.exp(log_lo + (log_hi - log_lo) * u)

    scale = float(spec["scale"])
    if dist == "exponential":
        # The configured log-prior is -x/scale throughout the fit bounds.
        mass = -math.expm1(-(hi - lo) / scale)
        return lambda u: lo - scale * np.log1p(-mass * u)

    from scipy import stats

    # The halfnormal log-prior uses the zero-mean Gaussian shape over the
    # full fit bounds, even when they include negative values.
    loc = 0.0 if dist == "halfnormal" else float(spec.get("loc", 0.0))
    if dist == "studentt":
        t = stats.t(float(spec["df"]), loc=loc, scale=scale)
        # Use survival probabilities in the upper tail to avoid subtracting from 1.
        cdf, quantile = (t.sf, t.isf) if lo >= loc else (t.cdf, t.ppf)
        p_lo, p_hi = cdf(lo), cdf(hi)
        if p_lo == p_hi:
            raise ValueError(f"{name}: Student-t prior mass is numerically zero in the fit bounds.")
        return lambda u: quantile(p_lo * (1.0 - u) + p_hi * u)

    a, b = (math.log(lo), math.log(hi)) if dist == "lognormal" else (lo, hi)
    normal = stats.truncnorm((a - loc) / scale, (b - loc) / scale, loc=loc, scale=scale)
    if dist == "lognormal":
        return lambda u: np.exp(normal.ppf(u))
    return normal.ppf


def build_prior_sampler(
    param_names: Sequence[str],
    specs: dict[str, dict],
    bounds: dict[str, tuple[float, float]],
) -> Callable[[jax.Array, int], jax.Array]:
    """Build ``sample_prior(key, n)`` for independent bounded physical parameters.

    Returns float32 draws of shape ``(n, len(param_names))`` in the supplied
    parameter order. This is for host-side initialisation. For mixed priors,
    the inverse CDFs run on the CPU in float64, outside the JIT-compiled
    likelihood and sampler transitions.
    """
    names = tuple(param_names)
    limits = np.array([bounds[name] for name in names], dtype=np.float64).reshape(-1, 2)
    lows, highs = limits.T
    quantiles = []
    for name, (lo, hi) in zip(names, limits, strict=True):
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"{name}: prior sampling requires finite, increasing bounds.")
        validate_prior_spec(name, specs[name], (lo, hi))
        quantiles.append(_prior_quantile(name, specs[name], float(lo), float(hi)))
    if all(specs[name].get("dist", "uniform").lower() == "uniform" for name in names):
        lo_jax, hi_jax = jnp.asarray(lows, jnp.float32), jnp.asarray(highs, jnp.float32)

        def sample_uniform(key, n):
            return jax.random.uniform(key, (n, len(names)), jnp.float32, lo_jax, hi_jax)

        return sample_uniform

    def sample(key, n):
        u = np.asarray(jax.random.uniform(key, (n, len(names)), jnp.float32), dtype=np.float64)
        draws = np.column_stack([quantile(u[:, i]) for i, quantile in enumerate(quantiles)])
        # Map the zero quantile to the exact lower bound even if a CDF rounded
        # that endpoint to zero in its tail. Clipping also handles round-off.
        draws = np.clip(np.where(u == 0.0, lows, draws), lows, highs)
        if not np.isfinite(draws).all():
            raise ValueError("Could not draw finite live points from the configured prior.")
        return jnp.asarray(draws, dtype=jnp.float32)

    return sample


def sigmoid_log_jacobian(raw: jnp.ndarray, lows: jnp.ndarray, highs: jnp.ndarray) -> jnp.ndarray:
    """Log |dx/draw| summed over parameters for ``x = lo + (hi - lo) * sigmoid(raw)``.

    Adding this to a log-posterior evaluated in unconstrained space makes a
    flat prior there correspond to a uniform density over ``[lo, hi]`` in
    physical space.

    Args:
        raw: Unconstrained parameter vector.
        lows: Lower bounds, broadcastable to ``raw``.
        highs: Upper bounds, broadcastable to ``raw``.

    Returns:
        Scalar ``sum(log(hi - lo) + log_sigmoid(raw) + log_sigmoid(-raw))``.
    """
    return jnp.sum(jnp.log(highs - lows) + jax.nn.log_sigmoid(raw) + jax.nn.log_sigmoid(-raw))


def build_component_log_prior(
    emulator_param_names: Sequence[str],
    specs: dict[str, dict],
    bounds: dict[str, tuple[float, float]],
    shared_param_names: Sequence[str] = (),
    fixed_param_names: Sequence[str] = (),
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a log-prior over a ``(K, N_emulator)`` matrix of component parameters.

    For multi-component spatial models each of the ``K`` components carries a
    full emulator parameter vector (columns in ``emulator_param_names`` order).
    Shared parameters take the same value in every row and are counted once
    (from row 0); fixed parameters are constants and contribute nothing; all
    remaining parameters are per-component and are summed over rows.

    Args:
        emulator_param_names: Column order of the parameter matrix.
        specs: ``{name: spec}`` covering every non-fixed parameter.
        bounds: ``{name: (lo, hi)}`` covering every non-fixed parameter.
        shared_param_names: Names shared across components (counted once).
        fixed_param_names: Names held constant (no prior contribution).

    Returns:
        A jit/grad/vmap-safe function ``fn(X (K, N)) -> scalar``.

    Raises:
        ValueError: If a shared or fixed name is not an emulator parameter, or
            appears in both lists.
    """
    names = list(emulator_param_names)
    shared = list(shared_param_names)
    fixed = list(fixed_param_names)
    for label, group in (("shared", shared), ("fixed", fixed)):
        unknown = [p for p in group if p not in names]
        if unknown:
            raise ValueError(f"{label}_param_names not in emulator params: {unknown}")
    overlap = sorted(set(shared) & set(fixed))
    if overlap:
        raise ValueError(f"parameters both shared and fixed: {overlap}")

    per_comp = [
        (i, _make_log_prior_1d(specs[p], bounds[p][0], bounds[p][1]))
        for i, p in enumerate(names)
        if p not in shared and p not in fixed
    ]
    shared_fns = [
        (i, _make_log_prior_1d(specs[p], bounds[p][0], bounds[p][1]))
        for i, p in enumerate(names)
        if p in shared
    ]

    def _row_log_prior(row: jnp.ndarray) -> jnp.ndarray:
        total = jnp.zeros((), dtype=row.dtype)
        for i, fn in per_comp:
            total = total + fn(row[i])
        return total

    def log_prior(X: jnp.ndarray) -> jnp.ndarray:
        total = jnp.sum(jax.vmap(_row_log_prior)(X))
        for i, fn in shared_fns:
            total = total + fn(X[0, i])
        return total

    return log_prior
