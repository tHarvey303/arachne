"""Blind (truth-free) initialisation and MAP finding for additive-component models.

The posterior of an :class:`~arachne.spatial.additive.AdditiveComponentModel`
is strongly multimodal in the label / size / mass directions, so gradient
samplers started from ``theta = 0`` frequently land in the wrong basin.  This
module provides a cheap, deterministic pipeline that reaches the right basin
using only the data:

1. :func:`image_moments` — centroid and rms size from the S/N-stacked image.
2. :func:`blind_initial_theta` — every component at the moment centroid, with
   sizes on a geometric ladder around the moment size and *neutral* SPS
   values (no information about the true galaxy is used).
3. :func:`solve_component_masses` — the K stellar masses enter the model
   linearly (flux ∝ 10**log_mass) so, given every other parameter, they are
   the solution of a K×K weighted least-squares system.  This removes the
   dominant degeneracy before any gradient step.
4. :func:`find_map` — a few rounds of Adam on ``-log_posterior`` with the
   linear mass solve repeated between rounds, then a low-learning-rate polish.
5. :func:`multistart_map` — :func:`find_map` from several SPS *archetypes*
   (e.g. dusty vs dust-free); return the best.

Everything is float32 and runs in a few tens of seconds on CPU for the
32-parameter bulge+disk demo problem.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from arachne.forward_model.pipeline import ForwardModel
from arachne.spatial.additive import AdditiveComponentModel
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

_BOUND_MARGIN = 0.01  # fraction of (hi - lo) kept clear of each bound before logit
_MASS_REF = 9.0  # templates in solve_component_masses are per 10**_MASS_REF Msun

# Neutral physical values for commonly used emulator parameter names.  Anything
# not listed here (and not matched by ``_neutral_default``) starts at the
# midpoint of its bounds.
_NEUTRAL_DEFAULTS: dict[str, float] = {
    "Av": 0.5,
    "log10metallicity": -2.5,
    "log_mass": 9.5,
    "fesc_lya": 0.2,
    "dust_bump_amplitude": 1.0,
    "slope": 0.0,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _neutral_default(name: str, bounds: tuple[float, float], mass_param: str) -> float:
    """Neutral physical value for parameter ``name`` (see :func:`blind_initial_theta`)."""
    if name.startswith("logsfr_ratio"):
        return 0.0
    if name == mass_param:
        return _NEUTRAL_DEFAULTS["log_mass"]
    if name in _NEUTRAL_DEFAULTS:
        return _NEUTRAL_DEFAULTS[name]
    lo, hi = bounds
    return 0.5 * (float(lo) + float(hi))


def _logit_in_bounds(value: float, lo: float, hi: float, margin: float = _BOUND_MARGIN) -> float:
    """Map a physical value to its unconstrained raw, clipped strictly inside (lo, hi)."""
    lo, hi = float(lo), float(hi)
    width = hi - lo
    v = min(max(float(value), lo + margin * width), hi - margin * width)
    u = (v - lo) / width
    return math.log(u) - math.log1p(-u)


def _require_additive(spatial_model: Any, where: str) -> AdditiveComponentModel:
    if not isinstance(spatial_model, AdditiveComponentModel):
        raise TypeError(
            f"{where} requires an AdditiveComponentModel spatial model, "
            f"got {type(spatial_model).__name__}"
        )
    return spatial_model


def _emulator_mass_column(model: AdditiveComponentModel) -> int:
    """Column of ``mass_param`` in the full emulator input (``component_params`` rows)."""
    return model.emulator_param_names.index(model.mass_param)


# ---------------------------------------------------------------------------
# 1. Image moments
# ---------------------------------------------------------------------------


def image_moments(
    flux: Any,
    variance: Any,
    mask: Any | None = None,
) -> tuple[float, float, float]:
    """Centroid and rms size of a source from the S/N-stacked multi-band image.

    The stacked image is ``S(y, x) = Σ_b mask_b flux_b / sqrt(variance_b)``,
    minus its median (sky/background level), clipped at zero.  The centroid
    is the ``S``-weighted mean pixel position and the size is
    ``sqrt(0.5 * (var_y + var_x))`` of the ``S``-weighted second moments.

    No model or truth information is used; this is the seed for
    :func:`blind_initial_theta`.

    Args:
        flux: Array of shape (N_bands, H, W).
        variance: Array of shape (N_bands, H, W); non-positive entries are ignored.
        mask: Optional array of shape (N_bands, H, W); 0/False marks bad pixels.

    Returns:
        ``(cy, cx, sigma_px)`` as Python floats, in pixel (row, col) units.  If
        the stacked image has no positive pixels, the image centre and
        ``max(H, W) / 8`` are returned.
    """
    flux = np.asarray(flux, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if flux.ndim != 3 or flux.shape != variance.shape:
        raise ValueError(
            f"flux and variance must both be (N_bands, H, W); got {flux.shape}, {variance.shape}"
        )
    valid = variance > 0
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_b = np.where(valid, flux / np.sqrt(np.where(valid, variance, 1.0)), 0.0)
    snr = snr_b.sum(axis=0)
    snr = np.clip(snr - np.median(snr), 0.0, None)

    _, H, W = flux.shape
    wsum = snr.sum()
    if not np.isfinite(wsum) or wsum <= 0:
        return (H - 1) / 2.0, (W - 1) / 2.0, max(H, W) / 8.0

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cy = float((snr * yy).sum() / wsum)
    cx = float((snr * xx).sum() / wsum)
    var_y = float((snr * (yy - cy) ** 2).sum() / wsum)
    var_x = float((snr * (xx - cx) ** 2).sum() / wsum)
    sigma = math.sqrt(max(0.5 * (var_y + var_x), 1e-6))
    return cy, cx, sigma


# ---------------------------------------------------------------------------
# 2. Blind initial theta
# ---------------------------------------------------------------------------


def blind_initial_theta(
    spatial_model: AdditiveComponentModel,
    obs: Any,
    size_scales: Sequence[float] | None = None,
    neutral_values: dict[str, float] | None = None,
    shared_values: dict[str, float] | None = None,
) -> jnp.ndarray:
    """Truth-free starting theta for an :class:`AdditiveComponentModel`.

    Every component is placed at the :func:`image_moments` centroid with
    ``rho = 0`` and an isotropic size ``size_scales[k] * sigma_moment``.  By
    default the K sizes form a geometric ladder from ``0.4`` to ``1.5`` times
    the moment size (``K = 2`` gives ``[0.4, 1.5]``, compact first; ``K = 1``
    gives ``[1.0]``), so the components are already ordered by size.

    SPS raws are the logit of *neutral* physical values:

    - ``neutral_values[name]`` if given (applies to every component);
    - otherwise a built-in default for known names: ``logsfr_ratio_*`` → 0,
      ``Av`` → 0.5, ``log10metallicity`` → -2.5, the mass parameter → 9.5,
      ``fesc_lya`` → 0.2, ``dust_bump_amplitude`` → 1.0, ``slope`` → 0;
    - otherwise the midpoint of the parameter's bounds.

    Shared parameters use ``shared_values`` with the same fall-through.  All
    values are clipped 1% inside their bounds before the logit so no raw is
    infinite.  Masses are only placeholders here — follow with
    :func:`solve_component_masses`.

    Args:
        spatial_model: The additive component model.
        obs: Anything with ``.flux``, ``.variance`` and ``.mask`` arrays of shape
            (N_bands, H, W), e.g. an :class:`~arachne.data.observation.ObservationCube`.
        size_scales: K multipliers of the moment size, one per component.
        neutral_values: Physical overrides for per-component SPS parameters.
        shared_values: Physical overrides for shared SPS parameters.

    Returns:
        float32 theta of shape ``(spatial_model.n_params,)``.

    Raises:
        TypeError: If ``spatial_model`` is not an AdditiveComponentModel.
        ValueError: If ``size_scales`` has the wrong length or an override names
            an unknown parameter.
    """
    model = _require_additive(spatial_model, "blind_initial_theta")
    K = model.n_components
    neutral_values = dict(neutral_values or {})
    shared_values = dict(shared_values or {})

    unknown = set(neutral_values) - set(model.sps_param_names)
    if unknown:
        raise ValueError(f"neutral_values not in sps_param_names: {sorted(unknown)}")
    unknown = set(shared_values) - set(model.shared_param_names)
    if unknown:
        raise ValueError(f"shared_values not in shared_param_names: {sorted(unknown)}")

    if size_scales is None:
        size_scales = [1.0] if K == 1 else list(np.geomspace(0.4, 1.5, K))
    size_scales = [float(s) for s in size_scales]
    if len(size_scales) != K:
        raise ValueError(f"size_scales must have length K={K}, got {len(size_scales)}")

    cy, cx, sigma = image_moments(obs.flux, obs.variance, getattr(obs, "mask", None))
    logger.info(
        f"blind_initial_theta: moment centroid ({cy:.2f}, {cx:.2f}), sigma {sigma:.2f} px, "
        f"K={K} size scales {np.round(size_scales, 3).tolist()}"
    )

    sps_raw = []
    for name in model.sps_param_names:
        lo, hi = model.param_bounds[name]
        value = neutral_values.get(name, _neutral_default(name, (lo, hi), model.mass_param))
        sps_raw.append(_logit_in_bounds(value, lo, hi))

    blocks = np.zeros((K, model.n_params_per_component), dtype=np.float32)
    for k, scale in enumerate(size_scales):
        log_size = math.log(max(scale * sigma, 1e-3))
        blocks[k, :5] = [cy, cx, log_size, log_size, 0.0]
        blocks[k, 5:] = sps_raw

    shared_raw = np.zeros(model.n_shared, dtype=np.float32)
    for j, name in enumerate(model.shared_param_names):
        lo, hi = model.param_bounds[name]
        value = shared_values.get(name, _neutral_default(name, (lo, hi), model.mass_param))
        shared_raw[j] = _logit_in_bounds(value, lo, hi)

    return model.join_theta(jnp.asarray(blocks), jnp.asarray(shared_raw)).astype(jnp.float32)


# ---------------------------------------------------------------------------
# 3. Linear mass solve
# ---------------------------------------------------------------------------


def solve_component_masses(forward_model: ForwardModel, theta: jnp.ndarray) -> jnp.ndarray:
    """Replace each component's mass with its weighted linear least-squares value.

    Holding every other parameter at its current value, the PSF-convolved
    model image is ``Σ_k a_k T_k`` where ``T_k`` is component ``k``'s
    convolved image per ``1e9 Msun`` and ``a_k = 10**(logM_k - 9)``.  This
    assumes the emulator flux is exactly linear in ``10**log_mass`` — true
    for :class:`~arachne.emulator.parrot_emulator_v2.ParrotEmulatorV2`
    with mass normalisation (and for any emulator that scales its output by
    the mass) — and a good approximation otherwise, in which case
    :func:`find_map` corrects the residual nonlinearity by gradient descent.

    The templates are obtained from ``spatial_model.component_seds`` and
    ``spatial_model.profiles`` at the *current* masses, then divided by
    ``10**(logM_k - 9)``; the K×K normal equations ``(T W Tᵀ) a = T W y``
    with ``W = mask / variance`` are solved (with a negligible relative ridge
    for numerical safety), ``a`` is clipped to ``>= 1e-3`` (crude
    non-negativity), and ``logM = 9 + log10(a)`` is clipped 1% inside the
    mass bounds before being written back as a raw.

    Pure JAX; safe under ``jax.jit``.

    Args:
        forward_model: ForwardModel whose spatial model is an AdditiveComponentModel.
        theta: Flat theta of shape (n_params,).

    Returns:
        theta with the K mass raws replaced, shape (n_params,), float32.

    Raises:
        TypeError: If the spatial model is not an AdditiveComponentModel.
    """
    model = _require_additive(forward_model.spatial_model, "solve_component_masses")
    theta = jnp.asarray(theta, dtype=jnp.float32)
    obs = forward_model.observation
    K = model.n_components
    mass_col = _emulator_mass_column(model)
    lo, hi = (float(v) for v in model.param_bounds[model.mass_param])

    _, _, _, sps_phys = model.component_params(theta)
    seds = model.component_seds(theta, forward_model.emulator)  # (K, N_bands)
    profiles = model.profiles(theta)  # (K, H, W)
    log_mass = sps_phys[:, mass_col]  # (K,)

    comp_images = jnp.einsum("kb,khw->kbhw", seds, profiles)  # (K, N_bands, H, W)
    templates = jax.vmap(forward_model.convolver)(comp_images)
    templates = templates / (10.0 ** (log_mass - _MASS_REF))[:, None, None, None]

    weights = obs.mask / (obs.variance + 1e-30)
    sqrt_w = jnp.sqrt(weights)
    t_w = (templates * sqrt_w[None]).reshape(K, -1)  # (K, N_data)
    y_w = (obs.flux * sqrt_w).reshape(-1)  # (N_data,)
    normal = t_w @ t_w.T  # (K, K)
    rhs = t_w @ y_w  # (K,)
    ridge = 1e-7 * jnp.trace(normal) / K + 1e-30
    amp = jnp.linalg.solve(normal + ridge * jnp.eye(K, dtype=normal.dtype), rhs)
    amp = jnp.clip(amp, 1e-3, None)

    margin = _BOUND_MARGIN * (hi - lo)
    new_log_mass = jnp.clip(_MASS_REF + jnp.log10(amp), lo + margin, hi - margin)
    u = (new_log_mass - lo) / (hi - lo)
    new_raw = jnp.log(u) - jnp.log1p(-u)

    blocks, shared_raw = model.split_theta(theta)
    blocks = blocks.at[:, 5 + model.mass_index].set(new_raw.astype(jnp.float32))
    return model.join_theta(blocks, shared_raw)


# ---------------------------------------------------------------------------
# 4. MAP finding
# ---------------------------------------------------------------------------


@dataclass
class MAPResult:
    """Output of :func:`find_map` / :func:`multistart_map`.

    Attributes:
        theta: MAP estimate (unconstrained), shape (n_params,), float32.
        neg_log_posterior: ``-log_posterior(theta)`` at the returned theta.
        history: Per-step loss (``-log_posterior``) over the whole run, shape (n_steps,).
        n_steps: Total number of Adam steps taken.
    """

    theta: jnp.ndarray
    neg_log_posterior: float
    history: np.ndarray
    n_steps: int


def _make_adam_runner(loss_fn, clip_norm: float):
    """Return ``run(theta, lr, n_steps) -> (theta, losses)``; jitted, static in ``n_steps``."""
    value_and_grad = jax.value_and_grad(loss_fn)

    def _run(theta: jnp.ndarray, lr: jnp.ndarray, n_steps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        opt = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(lr))
        opt_state = opt.init(theta)

        def step(carry, _):
            t, s = carry
            loss, grad = value_and_grad(t)
            updates, s = opt.update(grad, s, t)
            return (optax.apply_updates(t, updates), s), loss

        (theta, _), losses = jax.lax.scan(step, (theta, opt_state), None, length=n_steps)
        return theta, losses

    # ``length`` must be static; wrap so n_steps is hashed into the compile cache.
    return jax.jit(_run, static_argnums=2)


def find_map(
    forward_model: ForwardModel,
    theta0: jnp.ndarray,
    n_rounds: int = 4,
    steps_per_round: int = 300,
    lr: float = 0.05,
    final_steps: int = 400,
    final_lr: float = 0.01,
    resolve_masses: bool = True,
    clip_norm: float = 50.0,
    order_by_size: bool = True,
) -> MAPResult:
    """Find a MAP estimate with Adam, re-solving component masses between rounds.

    Minimises ``-forward_model.log_posterior`` with
    ``optax.chain(clip_by_global_norm(clip_norm), adam(lr))`` for ``n_rounds``
    rounds of ``steps_per_round`` steps (fresh optimiser state each round),
    then ``final_steps`` at ``final_lr``.  If the spatial model is an
    :class:`AdditiveComponentModel` and ``resolve_masses`` is true, the K
    masses are replaced by their linear least-squares solution
    (:func:`solve_component_masses`) before the first round and after every
    round except the final polish, which collapses the mass/size/colour
    degeneracy far faster than gradient steps do.  Finally, for additive
    models with ``order_by_size``, components are permuted compact-first
    (``order_components_by_size``) to fix the label symmetry.

    Each round is a single jitted ``lax.scan``; compilation happens once per
    distinct step count (so at most twice).

    Args:
        forward_model: Assembled ForwardModel.
        theta0: Starting theta, shape (n_params,) — e.g. :func:`blind_initial_theta`.
        n_rounds: Number of Adam rounds at ``lr``.
        steps_per_round: Adam steps per round.
        lr: Adam learning rate for the rounds.
        final_steps: Steps in the final polish (0 to skip).
        final_lr: Learning rate of the final polish.
        resolve_masses: Re-solve masses linearly between rounds (additive models only).
        clip_norm: Global gradient-norm clip.
        order_by_size: Order components compact-first at the end (additive models only).

    Returns:
        :class:`MAPResult`.
    """
    theta = jnp.asarray(theta0, dtype=jnp.float32)
    is_additive = isinstance(forward_model.spatial_model, AdditiveComponentModel)
    do_masses = bool(resolve_masses and is_additive)

    def loss_fn(t: jnp.ndarray) -> jnp.ndarray:
        return -forward_model.log_posterior(t)

    loss_jit = jax.jit(loss_fn)
    run_adam = _make_adam_runner(loss_fn, float(clip_norm))
    solve_masses = jax.jit(lambda t: solve_component_masses(forward_model, t))

    history: list[np.ndarray] = []
    if do_masses:
        theta = solve_masses(theta)
    logger.info(
        f"find_map: start -log_post = {float(loss_jit(theta)):.2f} "
        f"({n_rounds} x {steps_per_round} steps @ lr={lr}, then {final_steps} @ {final_lr})"
    )
    for rnd in range(int(n_rounds)):
        if steps_per_round > 0:
            theta, losses = run_adam(theta, jnp.float32(lr), int(steps_per_round))
            history.append(np.asarray(losses, dtype=np.float32))
        if do_masses:
            theta = solve_masses(theta)
        logger.info(
            f"find_map: round {rnd + 1}/{n_rounds} -log_post = {float(loss_jit(theta)):.2f}"
        )

    if final_steps > 0:
        theta, losses = run_adam(theta, jnp.float32(final_lr), int(final_steps))
        history.append(np.asarray(losses, dtype=np.float32))

    if order_by_size and is_additive:
        theta = forward_model.spatial_model.order_components_by_size(theta)

    theta = jnp.asarray(theta, dtype=jnp.float32)
    final_loss = float(loss_jit(theta))
    hist = np.concatenate(history) if history else np.zeros(0, dtype=np.float32)
    logger.info(f"find_map: final -log_post = {final_loss:.2f} after {hist.size} steps")
    return MAPResult(
        theta=theta, neg_log_posterior=final_loss, history=hist, n_steps=int(hist.size)
    )


# ---------------------------------------------------------------------------
# 5. Multi-start
# ---------------------------------------------------------------------------


def _apply_archetype(
    model: AdditiveComponentModel, theta: jnp.ndarray, archetype: dict[str, float]
) -> jnp.ndarray:
    """Set per-component (all K) and shared SPS raws from physical ``archetype`` values."""
    blocks, shared_raw = model.split_theta(theta)
    for name, value in archetype.items():
        lo, hi = model.param_bounds.get(name, (None, None))
        if name in model.sps_param_names:
            raw = _logit_in_bounds(value, lo, hi)
            blocks = blocks.at[:, 5 + model.sps_param_names.index(name)].set(raw)
        elif name in model.shared_param_names:
            raw = _logit_in_bounds(value, lo, hi)
            shared_raw = shared_raw.at[model.shared_param_names.index(name)].set(raw)
        else:
            raise ValueError(f"archetype parameter {name!r} is not a free SPS parameter")
    return model.join_theta(blocks, shared_raw)


def multistart_map(
    forward_model: ForwardModel,
    theta0: jnp.ndarray,
    archetypes: Sequence[dict[str, float]],
    **find_map_kwargs: Any,
) -> MAPResult:
    """Run :func:`find_map` from ``theta0`` and from each SPS archetype; return the best.

    An archetype is a ``{param_name: physical value}`` dict applied to *all*
    components' per-component SPS raws (shared names are allowed too), e.g.
    ``[{"Av": 0.3}, {"Av": 2.0}]`` to probe the dust/age degeneracy from both
    sides.  ``theta0`` itself is always one of the starts; an archetype that
    leaves ``theta0`` unchanged (e.g. ``{}``) is not run twice.

    Args:
        forward_model: ForwardModel with an AdditiveComponentModel spatial model.
        theta0: Base starting theta, shape (n_params,).
        archetypes: Physical overrides, one dict per additional start.
        **find_map_kwargs: Forwarded to :func:`find_map`.

    Returns:
        The :class:`MAPResult` with the lowest ``neg_log_posterior``.

    Raises:
        TypeError: If the spatial model is not an AdditiveComponentModel.
        ValueError: If an archetype names a parameter that is not free.
    """
    model = _require_additive(forward_model.spatial_model, "multistart_map")
    theta0 = jnp.asarray(theta0, dtype=jnp.float32)
    starts: list[tuple[str, jnp.ndarray]] = [("theta0", theta0)]
    for i, arch in enumerate(archetypes):
        t = _apply_archetype(model, theta0, dict(arch))
        if np.array_equal(np.asarray(t), np.asarray(theta0)):
            continue
        starts.append((f"archetype {i} {dict(arch)}", t))

    best: MAPResult | None = None
    for label, t in starts:
        result = find_map(forward_model, t, **find_map_kwargs)
        logger.info(f"multistart_map: {label}: -log_post = {result.neg_log_posterior:.2f}")
        if (
            best is None
            or not math.isfinite(best.neg_log_posterior)
            or result.neg_log_posterior < best.neg_log_posterior
        ):
            best = result
    assert best is not None
    return best
