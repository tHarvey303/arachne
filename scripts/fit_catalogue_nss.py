#!/usr/bin/env python3
r"""Catalogue SED fitting via Nested Slice Sampling (NSS) + comparison with NUTS.

NSS uses blackjax.nss (Hit-and-Run Slice Sampling inner kernel).  It is
gradient-free, naturally handles multimodality, and provides a calibrated
evidence estimate logZ.  The trade-off vs Pathfinder+NUTS is that NSS runs
sequentially per galaxy (no GPU batch axis) and requires more likelihood
evaluations.

Key design choices
------------------
* Physical space: NSS operates directly on x ∈ [lo, hi].  Out-of-bound
  proposals return loglikelihood = -inf, forcing the HRSS sampler to shrink
  back inside the domain without touching the emulator.
* Single compiled kernel: nss_init and nss_step are @jax.jit with obs_flux
  and flux_err as explicit traced arguments.  This compiles once for the
  (n_bands, n_params) shape and reuses the XLA binary across all galaxies.
* num_delete: default 50 — vmaps 50 replacement chains per NS step, so the
  Python while-loop runs ~10x fewer iterations at the same total dead-particle
  count.

Usage
-----
    # NSS only
    python scripts/fit_catalogue_nss.py catalogue.fits config.json \
        --n-galaxies 50 --num-live 500 --num-inner-steps 24 --num-delete 50

    # NSS + NUTS comparison (NUTS uses dual-averaging, no Pathfinder)
    python scripts/fit_catalogue_nss.py catalogue.fits config.json \
        --n-galaxies 20 --compare-nuts \
        --n-samples 500 --n-warmup 300 --n-chains 2

Output HDF5 layout
------------------
    /galaxy_id          (N,)            identifier from catalogue
    /nss_samples        (N, S, P)       posterior samples, physical space
    /nss_logZ           (N,)            log evidence point estimate
    /nss_logZ_err       (N,)            MC std of logZ over 100 shrinkage draws
    /nss_ess            (N,)            NS effective sample size
    /nss_n_steps        (N,)            number of NS while-loop iterations
    /nss_n_dead         (N,)            total dead particles accumulated
    /nss_time           (N,)            wall seconds (sampling only, excl. compile)
    /nss_rhat           (N, P)          split-R-hat on NS posterior samples
    [if --compare-nuts:]
    /nuts_samples       (N, C, S, P)    NUTS samples, physical space
    /nuts_accept        (N, C)          mean acceptance rate per chain
    /nuts_rhat          (N, P)          split-R-hat (physical space)
    /nuts_time          (N,)            wall seconds per galaxy

    attrs: param_names, band_names, param_bounds_lo, param_bounds_hi,
           prior_spec, emulator_path, run_timestamp, n_galaxies,
           num_live, num_inner_steps, num_delete, termination,
           n_samples_nss, sampler=nss
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

try:
    import blackjax
    from blackjax.ns.utils import finalise, log_weights
    from blackjax.ns.utils import sample as ns_resample
except ImportError as e:
    raise ImportError("pip install blackjax") from e

# ---------------------------------------------------------------------------
# Re-use constants and helpers from fit_catalogue.py
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from fit_catalogue import (  # noqa: E402
    SPS_PARAM_NAMES,
    PARAM_BOUNDS,
    LOG2PI,
    MISSING_SIGMA,
    OBS_MASK_THRESH,
    DEFAULT_EMULATOR,
    load_config,
    load_catalogue,
    load_emulator_and_band_indices,
    make_log_prior_fn,
    make_log_posterior_fn,
    split_rhat,
    check_gpu_linalg,
)

DEFAULT_NSS_OUTPUT = Path("outputs/catalogue_fit/results_nss.hdf5")
_P = len(SPS_PARAM_NAMES)

# ---------------------------------------------------------------------------
# NSS likelihood (physical space)
# ---------------------------------------------------------------------------


def make_log_likelihood_physical(emulator, band_idx: np.ndarray, min_frac_err: float):
    """Return log_likelihood(x_phys, obs_flux, flux_err) in physical parameter space.

    Returns -inf for positions outside the emulator training bounds so that
    the HRSS sampler never queries the emulator out-of-domain.
    """
    lows = jnp.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    highs = jnp.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    _bidx = jnp.array(band_idx, dtype=jnp.int32)

    def log_likelihood(x_phys, obs_flux, flux_err):
        in_bounds = jnp.all((x_phys >= lows) & (x_phys <= highs))
        pred = emulator.predict(x_phys[None, :])[0][_bidx]
        mask = (flux_err < OBS_MASK_THRESH).astype(x_phys.dtype)
        eff_err = (
            jnp.maximum(flux_err, min_frac_err * jnp.abs(obs_flux))
            if min_frac_err > 0
            else flux_err
        )
        chi2 = jnp.sum(mask * (obs_flux - pred) ** 2 / eff_err ** 2)
        log_norm = jnp.sum(mask * (-0.5 * LOG2PI - jnp.log(eff_err)))
        return jnp.where(in_bounds, log_norm - 0.5 * chi2, -jnp.inf)

    return log_likelihood


# ---------------------------------------------------------------------------
# JIT-compiled NSS init + step (obs/err as explicit traced arguments)
# ---------------------------------------------------------------------------


def build_nss_fns(
    emulator,
    band_idx: np.ndarray,
    log_prior_fn,
    log_like_fn,
    num_inner_steps: int,
    num_delete: int,
):
    """Return (nss_init_jit, nss_step_jit) compiled once for (n_bands, n_params) shapes.

    Both functions accept (obs_flux, flux_err) as explicit JAX arguments so
    the XLA binary is reused across all galaxies with the same filter set.
    """

    @jax.jit
    def nss_init(initial_samples, obs_flux, flux_err):
        def ll(x): return log_like_fn(x, obs_flux, flux_err)
        algo = blackjax.nss(
            logprior_fn=log_prior_fn,
            loglikelihood_fn=ll,
            num_delete=num_delete,
            num_inner_steps=num_inner_steps,
        )
        return algo.init(initial_samples)

    @jax.jit
    def nss_step(rng_key, state, obs_flux, flux_err):
        def ll(x): return log_like_fn(x, obs_flux, flux_err)
        algo = blackjax.nss(
            logprior_fn=log_prior_fn,
            loglikelihood_fn=ll,
            num_delete=num_delete,
            num_inner_steps=num_inner_steps,
        )
        return algo.step(rng_key, state)

    return nss_init, nss_step


# ---------------------------------------------------------------------------
# NSS evidence + ESS helpers
# ---------------------------------------------------------------------------


def _logz_from_weights(logw: jnp.ndarray):
    """logZ point estimate and MC std from log-weight matrix (N, 100)."""
    lw = jnp.nan_to_num(logw, nan=jnp.nan_to_num(logw).min())
    # logsumexp over particles for each of the 100 shrinkage draws
    logz_draws = jax.scipy.special.logsumexp(lw, axis=0)  # (100,)
    return float(logz_draws.mean()), float(logz_draws.std())


def _ess_from_weights(logw: jnp.ndarray):
    """ESS from log-weight matrix (N, 100), averaged over MC draws."""
    lw_mean = jnp.nan_to_num(logw).mean(axis=-1)  # (N,)
    lw_mean = lw_mean - lw_mean.max()
    ls = jax.scipy.special.logsumexp(lw_mean)
    ls2 = jax.scipy.special.logsumexp(2.0 * lw_mean)
    return float(jnp.exp(2.0 * ls - ls2))


# ---------------------------------------------------------------------------
# Per-galaxy NSS runner
# ---------------------------------------------------------------------------


def run_nss_galaxy(
    rng_key,
    obs_flux_i: np.ndarray,
    flux_err_i: np.ndarray,
    nss_init_fn,
    nss_step_fn,
    num_live: int,
    termination: float,
    n_samples_out: int,
    verbose: bool = False,
):
    """Run NSS on one galaxy.

    Returns
    -------
    samples_phys : ndarray (n_samples_out, P)
        Posterior samples in physical parameter space.
    logz : float
        Log evidence point estimate.
    logz_err : float
        MC uncertainty on logZ.
    ess : float
        Effective sample size.
    n_steps : int
        Number of NS while-loop iterations.
    n_dead : int
        Total dead particles accumulated.
    elapsed : float
        Wall seconds (sampling loop only, excluding compilation).
    """
    lows_np = np.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=np.float32)
    highs_np = np.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=np.float32)

    obs_jax = jnp.asarray(obs_flux_i, dtype=jnp.float32)
    err_jax = jnp.asarray(flux_err_i, dtype=jnp.float32)

    rng_key, init_key = jax.random.split(rng_key)
    initial_samples = jax.random.uniform(
        init_key,
        (num_live, _P),
        minval=jnp.asarray(lows_np),
        maxval=jnp.asarray(highs_np),
    )

    state = nss_init_fn(initial_samples, obs_jax, err_jax)
    jax.block_until_ready(state)

    t0 = time.perf_counter()
    dead: list = []
    n_steps = 0

    while float(state.integrator.logZ_live) - float(state.integrator.logZ) > termination:
        rng_key, subkey = jax.random.split(rng_key)
        state, dead_info = nss_step_fn(subkey, state, obs_jax, err_jax)
        dead.append(dead_info)
        n_steps += 1
        if verbose and n_steps % 50 == 0:
            gap = float(state.integrator.logZ_live) - float(state.integrator.logZ)
            print(f"    step {n_steps:5d}  logZ_gap={gap:.3f}")

    jax.block_until_ready(state)
    elapsed = time.perf_counter() - t0

    # --- post-process ---
    final_state = finalise(state, dead)
    rng_key, w_key, s_key = jax.random.split(rng_key, 3)
    logw = log_weights(w_key, final_state)           # (N_total, 100)
    logz, logz_err = _logz_from_weights(logw)
    ess = _ess_from_weights(logw)

    resampled = ns_resample(s_key, final_state, shape=n_samples_out)
    samples_phys = np.array(resampled.position, dtype=np.float32)   # (S, P)

    n_dead = len(dead) * dead[0].particles.loglikelihood.shape[0] if dead else 0

    return samples_phys, logz, logz_err, ess, n_steps, n_dead, elapsed


# ---------------------------------------------------------------------------
# Per-galaxy NUTS runner (dual-averaging, no Pathfinder; used for comparison)
# ---------------------------------------------------------------------------


def build_nuts_fns(
    emulator,
    band_idx: np.ndarray,
    log_prior_fn,
    min_frac_err: float,
    n_samples: int,
    n_warmup: int,
    n_chains: int,
    eps0: float,
    target_accept: float,
):
    """Return JIT-compiled per-galaxy NUTS runner (takes obs/err as explicit args).

    Starts every chain from a small random perturbation of the unconstrained
    domain midpoint (theta=0 → physical midpoint).  No Pathfinder warm-start.
    Uses an identity inverse mass matrix and dual-averaging step-size adaptation.
    """
    log_post_fn = make_log_posterior_fn(emulator, band_idx, log_prior_fn, min_frac_err)
    LOG_EPS0 = math.log(eps0)
    MU = math.log(10.0 * eps0)

    # dual-averaging state: (m, log_step, log_step_bar, h_bar)
    def _da_init():
        return (
            jnp.array(0.0),
            jnp.array(LOG_EPS0),
            jnp.array(LOG_EPS0),
            jnp.array(0.0),
        )

    def _da_update(da, accept):
        m, log_step, log_step_bar, h_bar = da
        m += 1.0
        accept = jnp.clip(jnp.nan_to_num(accept, nan=0.0), 0.0, 1.0)
        w = 1.0 / (m + 10.0)
        h_bar = (1.0 - w) * h_bar + w * (target_accept - accept)
        log_step = MU - jnp.sqrt(m) / 0.05 * h_bar
        log_step = jnp.clip(log_step, -12.0, 2.0)
        eta = m ** (-0.75)
        log_step_bar = eta * log_step + (1.0 - eta) * log_step_bar
        return (m, log_step, log_step_bar, h_bar)

    eye = jnp.eye(_P, dtype=jnp.float32)

    @jax.jit
    def nuts_run_galaxy(rng_key, chain_inits, obs_flux, flux_err):
        """Run n_chains independent NUTS chains for one galaxy.

        chain_inits : (C, P) unconstrained starting points
        Returns (samples_C_S_P, accept_C)
        """
        def lp(theta):
            return log_post_fn(theta, obs_flux, flux_err)

        def _one_chain(theta_init, chain_key):
            warm_key, samp_key = jax.random.split(chain_key)
            state = blackjax.nuts(lp, step_size=eps0, inverse_mass_matrix=eye).init(theta_init)

            def warm_step(carry, k):
                st, da = carry
                kern = blackjax.nuts(lp, step_size=jnp.exp(da[1]), inverse_mass_matrix=eye)
                st, info = kern.step(k, st)
                da = _da_update(da, info.acceptance_rate)
                return (st, da), None

            (state, da), _ = jax.lax.scan(
                warm_step, (state, _da_init()), jax.random.split(warm_key, n_warmup)
            )
            eps_final = jnp.where(jnp.isfinite(jnp.exp(da[2])), jnp.exp(da[2]), eps0)
            kern = blackjax.nuts(lp, step_size=eps_final, inverse_mass_matrix=eye)

            def samp_step(st, k):
                st, info = kern.step(k, st)
                return st, (st.position, info.acceptance_rate)

            _, (samples, ar) = jax.lax.scan(
                samp_step, state, jax.random.split(samp_key, n_samples)
            )
            return samples, jnp.mean(ar), eps_final

        samples_CSP, accept_C, eps_C = jax.vmap(_one_chain)(
            chain_inits, jax.random.split(rng_key, n_chains)
        )
        return samples_CSP, accept_C   # (C, S, P), (C,)

    return nuts_run_galaxy


def run_nuts_galaxy(
    rng_key,
    obs_flux_i: np.ndarray,
    flux_err_i: np.ndarray,
    nuts_fn,
    n_chains: int,
    chain_jitter: float,
):
    """Wrap nuts_fn for a single galaxy, returning physical-space samples."""
    obs_jax = jnp.asarray(obs_flux_i, dtype=jnp.float32)
    err_jax = jnp.asarray(flux_err_i, dtype=jnp.float32)

    rng_key, k_init = jax.random.split(rng_key)
    # Start chains at domain midpoint (theta=0) + small jitter
    chain_inits = (
        jax.random.normal(k_init, (_P,)) * 0.0
        + jax.random.normal(k_init, (n_chains, _P)) * chain_jitter
    )

    t0 = time.perf_counter()
    samples_CSP, accept_C = nuts_fn(rng_key, chain_inits, obs_jax, err_jax)
    jax.block_until_ready(samples_CSP)
    elapsed = time.perf_counter() - t0

    # Convert unconstrained → physical
    lows = np.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=np.float32)
    highs = np.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=np.float32)
    s = np.array(samples_CSP, dtype=np.float32)   # (C, S, P)
    phys = lows + (highs - lows) / (1.0 + np.exp(-s))

    return phys, np.array(accept_C, dtype=np.float32), elapsed


# ---------------------------------------------------------------------------
# HDF5 creation
# ---------------------------------------------------------------------------


def create_output_file(
    output_path: Path,
    n_galaxies: int,
    n_samples_out: int,
    n_chains: int,
    n_samples_nuts: int,
    band_names: list[str],
    emulator_path: Path,
    prior_specs: dict,
    galaxy_ids: list,
    compare_nuts: bool,
    run_args: dict,
) -> h5py.File:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(output_path, "w")

    try:
        f.create_dataset("galaxy_id", data=np.array(galaxy_ids, dtype=np.int64))
    except (ValueError, TypeError):
        dt = h5py.string_dtype()
        f.create_dataset("galaxy_id", data=np.array([str(g) for g in galaxy_ids], dtype=object), dtype=dt)

    N, S, P = n_galaxies, n_samples_out, _P
    ckw = dict(compression="gzip", compression_opts=4)

    f.create_dataset("nss_samples", shape=(N, S, P), dtype=np.float32,
                     chunks=(1, S, P), **ckw)
    f.create_dataset("nss_logZ", shape=(N,), dtype=np.float32)
    f.create_dataset("nss_logZ_err", shape=(N,), dtype=np.float32)
    f.create_dataset("nss_ess", shape=(N,), dtype=np.float32)
    f.create_dataset("nss_n_steps", shape=(N,), dtype=np.int32)
    f.create_dataset("nss_n_dead", shape=(N,), dtype=np.int32)
    f.create_dataset("nss_time", shape=(N,), dtype=np.float32)
    f.create_dataset("nss_rhat", shape=(N, P), dtype=np.float32)

    if compare_nuts:
        C = n_chains
        f.create_dataset("nuts_samples", shape=(N, C, n_samples_nuts, P), dtype=np.float32,
                         chunks=(1, C, n_samples_nuts, P), **ckw)
        f.create_dataset("nuts_accept", shape=(N, C), dtype=np.float32)
        f.create_dataset("nuts_rhat", shape=(N, P), dtype=np.float32)
        f.create_dataset("nuts_time", shape=(N,), dtype=np.float32)

    f.attrs["param_names"] = [s.encode() for s in SPS_PARAM_NAMES]
    f.attrs["band_names"] = [s.encode() for s in band_names]
    f.attrs["param_bounds_lo"] = np.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES])
    f.attrs["param_bounds_hi"] = np.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES])
    f.attrs["prior_spec"] = json.dumps(prior_specs)
    f.attrs["emulator_path"] = str(emulator_path)
    f.attrs["run_timestamp"] = datetime.now(timezone.utc).isoformat()
    f.attrs["n_galaxies"] = n_galaxies
    f.attrs["sampler"] = "nss"
    for k, v in run_args.items():
        f.attrs[k] = v
    return f


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_catalogue_nss(
    obs_flux: np.ndarray,
    flux_err: np.ndarray,
    galaxy_ids: list,
    emulator,
    band_idx: np.ndarray,
    prior_specs: dict,
    output_path: Path,
    emulator_path: Path,
    num_live: int,
    num_inner_steps: int,
    num_delete: int,
    termination: float,
    n_samples_out: int,
    min_frac_err: float,
    seed: int,
    compare_nuts: bool,
    n_samples_nuts: int,
    n_warmup_nuts: int,
    n_chains_nuts: int,
    chain_jitter: float,
) -> None:
    N, n_bands = obs_flux.shape
    band_names = [emulator.band_names[int(i)] for i in band_idx]
    lows_np = np.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=np.float32)
    highs_np = np.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=np.float32)

    print(f"\nNSS catalogue fit: {N} galaxies", flush=True)
    print(f"  num_live={num_live}  num_inner_steps={num_inner_steps}  "
          f"num_delete={num_delete}  termination={termination}", flush=True)
    print(f"  n_samples_out={n_samples_out}  min_frac_err={min_frac_err:.1%}", flush=True)
    if compare_nuts:
        eps0 = 0.5 / (_P ** 0.25)
        print(f"  NUTS comparison: n_chains={n_chains_nuts}  n_warmup={n_warmup_nuts}  "
              f"n_samples={n_samples_nuts}  eps0={eps0:.4g}", flush=True)

    log_prior_fn = make_log_prior_fn(prior_specs)
    log_like_fn = make_log_likelihood_physical(emulator, band_idx, min_frac_err)

    nss_init_fn, nss_step_fn = build_nss_fns(
        emulator, band_idx, log_prior_fn, log_like_fn,
        num_inner_steps, num_delete,
    )

    nuts_fn = None
    if compare_nuts:
        eps0 = 0.5 / (_P ** 0.25)
        nuts_fn = build_nuts_fns(
            emulator, band_idx, log_prior_fn, min_frac_err,
            n_samples_nuts, n_warmup_nuts, n_chains_nuts,
            eps0, target_accept=0.8,
        )

    run_args = dict(
        num_live=num_live, num_inner_steps=num_inner_steps,
        num_delete=num_delete, termination=float(termination),
        n_samples_nss=n_samples_out, min_frac_err=float(min_frac_err),
    )
    if compare_nuts:
        run_args.update(n_chains_nuts=n_chains_nuts, n_warmup_nuts=n_warmup_nuts,
                        n_samples_nuts=n_samples_nuts)

    h5 = create_output_file(
        output_path, N, n_samples_out, n_chains_nuts, n_samples_nuts,
        band_names, emulator_path, prior_specs, galaxy_ids, compare_nuts, run_args,
    )

    # Warm-up JIT on first galaxy (don't time it)
    print("\nWarm-up JIT compilation (first galaxy) ...", flush=True)
    rng = jax.random.PRNGKey(seed)
    rng, k0 = jax.random.split(rng)
    obs0 = jnp.asarray(obs_flux[0:1], dtype=jnp.float32)[0]
    err0 = jnp.asarray(flux_err[0:1], dtype=jnp.float32)[0]
    live0 = jax.random.uniform(k0, (num_live, _P),
                                minval=jnp.asarray(lows_np),
                                maxval=jnp.asarray(highs_np))
    _st = nss_init_fn(live0, obs0, err0)
    rng, _k = jax.random.split(rng)
    _st, _ = nss_step_fn(_k, _st, obs0, err0)
    jax.block_until_ready(_st)
    if nuts_fn is not None:
        rng, k_n = jax.random.split(rng)
        _ci = jax.random.normal(k_n, (n_chains_nuts, _P)) * 0.1
        _s, _a = nuts_fn(k_n, _ci, obs0, err0)
        jax.block_until_ready(_s)
    print("  JIT compile done.", flush=True)

    rng = jax.random.PRNGKey(seed)   # reset for reproducibility
    t_global = time.perf_counter()

    nss_times: list[float] = []
    nuts_times: list[float] = []

    for i in range(N):
        rng, gal_key = jax.random.split(rng)

        # ---- NSS ----
        rng, nss_key = jax.random.split(rng)
        samp_phys, logz, logz_err, ess, n_steps, n_dead, nss_t = run_nss_galaxy(
            nss_key,
            obs_flux[i], flux_err[i],
            nss_init_fn, nss_step_fn,
            num_live, termination, n_samples_out,
        )
        nss_times.append(nss_t)

        # R-hat from NS posterior samples (treat as single split chain).
        # samp_phys is already in physical space — no sigmoid transform needed.
        rhat_nss = split_rhat(
            samp_phys[np.newaxis, np.newaxis, :, :],
            lows=None, highs=None,
        )[0]  # (P,)

        h5["nss_samples"][i] = samp_phys
        h5["nss_logZ"][i] = logz
        h5["nss_logZ_err"][i] = logz_err
        h5["nss_ess"][i] = ess
        h5["nss_n_steps"][i] = n_steps
        h5["nss_n_dead"][i] = n_dead
        h5["nss_time"][i] = nss_t
        h5["nss_rhat"][i] = rhat_nss

        diag = (f"logZ={logz:.2f}±{logz_err:.2f}  ESS={ess:.0f}  "
                f"n_dead={n_dead}  rhat_max={np.nanmax(rhat_nss):.3f}  "
                f"t={nss_t:.1f}s")

        # ---- NUTS (optional) ----
        if nuts_fn is not None:
            rng, nuts_key = jax.random.split(rng)
            phys_CSP, accept_C, nuts_t = run_nuts_galaxy(
                nuts_key, obs_flux[i], flux_err[i],
                nuts_fn, n_chains_nuts, chain_jitter,
            )
            nuts_times.append(nuts_t)

            # R-hat in physical space: (1, C, S, P).
            # phys_CSP is already sigmoid-transformed — no second transform.
            rhat_nuts = split_rhat(
                phys_CSP[np.newaxis, :, :, :],
                lows=None, highs=None,
            )[0]  # (P,)

            h5["nuts_samples"][i] = phys_CSP
            h5["nuts_accept"][i] = accept_C
            h5["nuts_rhat"][i] = rhat_nuts
            h5["nuts_time"][i] = nuts_t

            diag += (f"  |  NUTS: accept={np.mean(accept_C):.2f}  "
                     f"rhat_max={np.nanmax(rhat_nuts):.3f}  t={nuts_t:.1f}s")

        elapsed_total = time.perf_counter() - t_global
        eta = elapsed_total / (i + 1) * (N - i - 1)
        print(f"  [{i + 1:>{len(str(N))}}/{N}]  {diag}  ETA={eta:.0f}s", flush=True)

    # ---- summary ----
    total = time.perf_counter() - t_global
    print(f"\n{'=' * 70}")
    print(f"Done: {N} galaxies in {total:.1f}s")

    nss_arr = np.array(nss_times)
    print(f"\nNSS timing (excl. first-galaxy JIT):")
    print(f"  median {np.median(nss_arr):.2f}s  mean {np.mean(nss_arr):.2f}s  "
          f"total {nss_arr.sum():.1f}s")
    print(f"  logZ:  mean={np.nanmean(h5['nss_logZ'][:]):.2f}  "
          f"std={np.nanstd(h5['nss_logZ'][:]):.2f}")
    print(f"  ESS:   median={np.nanmedian(h5['nss_ess'][:]):.0f}")
    rh = np.array(h5["nss_rhat"][:])
    worst = np.nanmax(rh, axis=1)
    print(f"  Rhat<1.05: {np.mean(worst < 1.05):.1%}  "
          f"(>1.1: {np.mean(worst > 1.1):.1%})")

    if compare_nuts and nuts_times:
        nuts_arr = np.array(nuts_times)
        ratio = np.median(nss_arr) / np.median(nuts_arr)
        print(f"\nNUTS timing:")
        print(f"  median {np.median(nuts_arr):.2f}s  mean {np.mean(nuts_arr):.2f}s")
        rh_n = np.array(h5["nuts_rhat"][:])
        worst_n = np.nanmax(rh_n, axis=1)
        print(f"  Rhat<1.05: {np.mean(worst_n < 1.05):.1%}  "
              f"(>1.1: {np.mean(worst_n > 1.1):.1%})")
        print(f"\nSpeed ratio (NSS/NUTS per galaxy): {ratio:.1f}x  "
              f"{'(NSS slower)' if ratio > 1 else '(NSS faster)'}")

        # Per-parameter comparison
        nss_s = np.array(h5["nss_samples"][:])       # (N, S, P)
        nuts_s = np.array(h5["nuts_samples"][:])      # (N, C, S, P)
        nuts_flat = nuts_s.reshape(N, -1, _P)          # (N, C*S, P)
        print(f"\nPosterior mean comparison (NSS vs NUTS, averaged over {N} galaxies):")
        print(f"  {'param':<22}  {'NSS mean':>9}  {'NUTS mean':>9}  "
              f"{'NSS std':>9}  {'NUTS std':>9}  {'Δmean/σ':>8}")
        for pi, pname in enumerate(SPS_PARAM_NAMES):
            nss_m = nss_s[:, :, pi].mean(axis=1)    # (N,)
            nuts_m = nuts_flat[:, :, pi].mean(axis=1)
            nss_sig = nss_s[:, :, pi].std(axis=1)
            nuts_sig = nuts_flat[:, :, pi].std(axis=1)
            delta = np.abs(nss_m - nuts_m) / np.maximum(nss_sig, 1e-10)
            print(f"  {pname:<22}  {nss_m.mean():>9.3f}  {nuts_m.mean():>9.3f}  "
                  f"{nss_sig.mean():>9.3f}  {nuts_sig.mean():>9.3f}  "
                  f"{delta.mean():>8.3f}")

    print(f"\n  Output: {output_path}")
    h5.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit galaxy catalogue with Nested Slice Sampling (NSS) and optionally compare to NUTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("catalogue", help="Input catalogue (FITS/CSV/HDF5)")
    parser.add_argument("config", help="Band config JSON (same format as fit_catalogue.py)")
    parser.add_argument("--emulator", default=str(DEFAULT_EMULATOR))
    parser.add_argument("--output", default=str(DEFAULT_NSS_OUTPUT))
    parser.add_argument("--n-galaxies", type=int, default=None,
                        help="Subset of galaxies to fit (default: all).")
    parser.add_argument("--num-live", type=int, default=500,
                        help="Number of NS live particles (>= ~50*P for 12-param problem).")
    parser.add_argument("--num-inner-steps", type=int, default=24,
                        help="HRSS inner steps per replacement (recommend 2*P=24).")
    parser.add_argument("--num-delete", type=int, default=50,
                        help="Dead particles per NS iteration (vmapped in parallel; larger "
                             "= fewer Python iterations, better GPU utilisation).")
    parser.add_argument("--termination", type=float, default=-3.0,
                        help="Stop when logZ_live - logZ < termination (log scale).")
    parser.add_argument("--n-samples-out", type=int, default=500,
                        help="NS posterior samples to resample and store.")
    parser.add_argument("--seed", type=int, default=0)

    # NUTS comparison
    parser.add_argument("--compare-nuts", action="store_true",
                        help="Also run per-galaxy NUTS (dual-averaging, no Pathfinder) "
                             "for performance comparison.")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="NUTS samples per chain (if --compare-nuts).")
    parser.add_argument("--n-warmup", type=int, default=300,
                        help="NUTS dual-averaging warmup steps (if --compare-nuts).")
    parser.add_argument("--n-chains", type=int, default=2,
                        help="NUTS chains per galaxy (if --compare-nuts).")
    parser.add_argument("--chain-jitter", type=float, default=0.5,
                        help="Std of NUTS chain init perturbations (unconstrained space).")

    args = parser.parse_args()

    emulator_path = Path(args.emulator)
    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator not found: {emulator_path}")

    check_gpu_linalg()

    config = load_config(Path(args.config))
    obs_flux, flux_err, galaxy_ids = load_catalogue(Path(args.catalogue), config)
    emulator, band_idx = load_emulator_and_band_indices(
        emulator_path, list(config["bands"].keys())
    )

    if args.n_galaxies is not None:
        n = min(args.n_galaxies, len(galaxy_ids))
        obs_flux = obs_flux[:n]
        flux_err = flux_err[:n]
        galaxy_ids = galaxy_ids[:n]
        print(f"  Subset: using first {n} galaxies.")

    run_catalogue_nss(
        obs_flux=obs_flux,
        flux_err=flux_err,
        galaxy_ids=galaxy_ids,
        emulator=emulator,
        band_idx=band_idx,
        prior_specs=config["resolved_priors"],
        output_path=Path(args.output),
        emulator_path=emulator_path,
        num_live=args.num_live,
        num_inner_steps=args.num_inner_steps,
        num_delete=args.num_delete,
        termination=args.termination,
        n_samples_out=args.n_samples_out,
        min_frac_err=float(config.get("min_frac_err", 0.15)),
        seed=args.seed,
        compare_nuts=args.compare_nuts,
        n_samples_nuts=args.n_samples,
        n_warmup_nuts=args.n_warmup,
        n_chains_nuts=args.n_chains,
        chain_jitter=args.chain_jitter,
    )


if __name__ == "__main__":
    main()
