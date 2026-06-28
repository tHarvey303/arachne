#!/usr/bin/env python3
r"""Single-galaxy SED fitting -- a debuggable reference implementation.

Fits ONE galaxy at a time with verbose diagnostics, so problems are easy to
see and isolate before scaling up to the batched catalogue fitter.  Same
forward model and likelihood as fit_catalogue.py (sigmoid-reparameterised,
masked Gaussian likelihood, configurable priors), plus:

  * A redshift-grid initialiser.  Amplitude-matching log_mass alone fails
    because the domain-midpoint shape sits at z~7, where the emulator zeroes
    the bands a real low-z galaxy actually has flux in -- so the mass solve is
    garbage.  Here we grid over redshift, amplitude-solve log_mass at each z
    (from a *sensible* reference shape, not the wide-bound midpoint), and take
    the best chi-squared.  This localises the z-mass degeneracy that dominates
    initialisation.

  * Three selectable samplers, to compare:
      pathfinder_nuts : Pathfinder metric + fixed-step NUTS   (the batched method)
      window_nuts     : Stan-style window_adaptation + NUTS    (robust standard)
      mclmc           : Microcanonical Langevin MC + tuning    (GPU-friendly)

Usage
-----
    # real galaxy by row index
    python fit_one_galaxy.py catalogue.fits bands.json --index 0
    python fit_one_galaxy.py catalogue.fits bands.json --id 12345 --sampler window_nuts
    python fit_one_galaxy.py catalogue.fits bands.json --index 0 --sampler mclmc

    # synthetic sanity check (no catalogue needed; recovers known truth)
    python fit_one_galaxy.py --mock --sampler window_nuts

Output (under outputs/sed_one/)
    samples_<id>.hdf5  : posterior samples (n_chains, n_samples, P), unconstrained
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import math
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

try:
    import blackjax
    import blackjax.vi.pathfinder as pf_mod
    from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install blackjax") from e

OUTDIR = Path("outputs/sed_one")
DEFAULT_EMULATOR = Path("scripts/outputs/emulators/parrot_emulator.eqx")

MISSING_SIGMA = 1e10
OBS_MASK_THRESH = MISSING_SIGMA * 0.5
LOG2PI = math.log(2.0 * math.pi)

SPS_PARAM_NAMES = [
    "redshift", "log_mass", "slope", "fesc_lya", "dust_bump_amplitude",
    "log10metallicity", "Av", "logsfr_ratio_0", "logsfr_ratio_1",
    "logsfr_ratio_2", "logsfr_ratio_3", "logsfr_ratio_4",
]
PARAM_BOUNDS = {
    "redshift": (0.01, 14.0), "log_mass": (4.0, 12.0), "slope": (-0.3, 1.1),
    "fesc_lya": (0.0, 1.0), "dust_bump_amplitude": (0.0, 5.0),
    "log10metallicity": (-4.0, -1.39), "Av": (0.001, 5.0),
    "logsfr_ratio_0": (-10.0, 10.0), "logsfr_ratio_1": (-10.0, 10.0),
    "logsfr_ratio_2": (-10.0, 10.0), "logsfr_ratio_3": (-10.0, 10.0),
    "logsfr_ratio_4": (-10.0, 10.0),
}
DEFAULT_PRIORS = {p: {"dist": "uniform"} for p in SPS_PARAM_NAMES}
for i in range(5):
    DEFAULT_PRIORS[f"logsfr_ratio_{i}"] = {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3}

# Sensible reference physical params for the *initialiser* shape (NOT midpoints
# of the wide bounds, which are unphysical: Av=2.5, Z=-2.7, ...).  Only the SED
# shape matters here; z is gridded and log_mass is amplitude-solved.
REF_INIT_PHYS = {
    "redshift": 1.0, "log_mass": 8.0, "slope": 0.0, "fesc_lya": 0.5,
    "dust_bump_amplitude": 0.0, "log10metallicity": -2.0, "Av": 0.3,
    "logsfr_ratio_0": 0.0, "logsfr_ratio_1": 0.0, "logsfr_ratio_2": 0.0,
    "logsfr_ratio_3": 0.0, "logsfr_ratio_4": 0.0,
}

REDSHIFT_IDX = SPS_PARAM_NAMES.index("redshift")
LOGMASS_IDX = SPS_PARAM_NAMES.index("log_mass")
AV_IDX = SPS_PARAM_NAMES.index("Av")

FLUX_UNIT_TO_NJY = {"nJy": 1.0, "uJy": 1e3, "ujy": 1e3, "µJy": 1e3, "mJy": 1e6, "Jy": 1e9}
_KNOWN_DISTS = {"uniform", "loguniform", "normal", "studentt", "halfnormal", "exponential", "lognormal"}


# ---------------------------------------------------------------------------
# Priors  (same contract as fit_catalogue.py)
# ---------------------------------------------------------------------------


def _build_prior_logprob(spec, lo, hi):
    dist = spec.get("dist", "uniform").lower()
    if dist == "uniform":
        return lambda x: jnp.zeros((), x.dtype)
    if dist == "loguniform":
        return lambda x: -jnp.log(x)
    if dist == "normal":
        loc, sc = float(spec.get("loc", 0.0)), float(spec["scale"])
        return lambda x: -0.5 * LOG2PI - math.log(sc) - 0.5 * ((x - loc) / sc) ** 2
    if dist == "studentt":
        df, loc, sc = float(spec["df"]), float(spec.get("loc", 0.0)), float(spec["scale"])
        c = math.lgamma(0.5 * (df + 1)) - math.lgamma(0.5 * df) - 0.5 * math.log(df * math.pi) - math.log(sc)
        return lambda x: c - 0.5 * (df + 1) * jnp.log1p(((x - loc) / sc) ** 2 / df)
    if dist == "halfnormal":
        sc = float(spec["scale"])
        return lambda x: -0.5 * (x / sc) ** 2
    if dist == "exponential":
        sc = float(spec["scale"])
        return lambda x: -x / sc
    if dist == "lognormal":
        loc, sc = float(spec.get("loc", 0.0)), float(spec["scale"])
        return lambda x: -jnp.log(x) - 0.5 * ((jnp.log(x) - loc) / sc) ** 2
    raise ValueError(f"unknown dist {dist!r}")


def make_log_prior_fn(prior_specs):
    fns = [_build_prior_logprob(prior_specs[p], *PARAM_BOUNDS[p]) for p in SPS_PARAM_NAMES]

    def log_prior(x):
        total = jnp.zeros((), x.dtype)
        for i, fn in enumerate(fns):
            total = total + fn(x[i])
        return total

    return log_prior


def resolve_priors(config):
    priors = {p: dict(DEFAULT_PRIORS[p]) for p in SPS_PARAM_NAMES}
    for k, v in (config.get("priors", {}) or {}).items():
        if k not in SPS_PARAM_NAMES:
            raise ValueError(f"priors: unknown parameter {k!r}")
        priors[k] = dict(v)
    for p in SPS_PARAM_NAMES:
        if priors[p].get("dist", "uniform").lower() not in _KNOWN_DISTS:
            raise ValueError(f"{p}: unknown dist {priors[p]['dist']!r}")
    return priors


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_LOWS = jnp.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
_HIGHS = jnp.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=jnp.float32)


def to_phys(theta):
    return _LOWS + (_HIGHS - _LOWS) * jax.nn.sigmoid(theta)


def to_theta(x):
    u = jnp.clip((jnp.asarray(x) - _LOWS) / (_HIGHS - _LOWS), 1e-4, 1.0 - 1e-4)
    return jnp.log(u / (1.0 - u))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_config(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    if "bands" not in cfg:
        raise ValueError("Config must contain 'bands'.")
    cfg.setdefault("flux_unit", "nJy")
    cfg.setdefault("id_col", None)
    cfg["resolved_priors"] = resolve_priors(cfg)
    return cfg


def load_one_galaxy(catalogue_path, config, index=None, gal_id=None):
    """Return (obs_flux, flux_err, gid, band_names) for a single galaxy (nJy)."""
    from astropy.table import Table

    t = Table.read(str(catalogue_path))
    if gal_id is not None:
        id_col = config.get("id_col")
        if not id_col or id_col not in t.colnames:
            raise ValueError(f"--id given but id_col {id_col!r} not in catalogue.")
        rows = np.where(np.asarray(t[id_col]).astype(str) == str(gal_id))[0]
        if len(rows) == 0:
            raise ValueError(f"id {gal_id!r} not found.")
        index = int(rows[0])
    index = index or 0
    row = t[index]

    unit = config["flux_unit"]
    factor = FLUX_UNIT_TO_NJY.get(unit, 1.0)
    use_abmag = unit == "ABmag"
    band_names = list(config["bands"].keys())
    obs = np.zeros(len(band_names), np.float32)
    err = np.full(len(band_names), MISSING_SIGMA, np.float32)
    for b, band in enumerate(band_names):
        f = float(row[config["bands"][band]["flux"]])
        e = float(row[config["bands"][band]["err"]])
        if use_abmag:
            f, e = (10.0 ** ((8.9 - f) / 2.5) * 1e9), None
        else:
            f, e = f * factor, e * factor
        if np.isfinite(f) and (e is not None) and np.isfinite(e) and e > 0:
            obs[b], err[b] = f, e  # negative fluxes kept on purpose

    id_col = config.get("id_col")
    gid = str(row[id_col]) if id_col and id_col in t.colnames else str(index)
    return obs, err, gid, band_names


def make_mock(emulator, band_idx, seed=0):
    """Generate a noisy mock SED from a fixed, realistic truth (for sanity checks)."""
    rng = np.random.default_rng(seed)
    true = np.array([REF_INIT_PHYS[p] for p in SPS_PARAM_NAMES], np.float32)
    true[LOGMASS_IDX] = 10.0
    true[REDSHIFT_IDX] = 1.3
    flux = np.asarray(emulator.predict(jnp.asarray(true)[None, :]))[0][np.asarray(band_idx)]
    err = 0.05 * np.abs(flux) + 1e-3
    obs = (flux + rng.normal(0, err)).astype(np.float32)
    return obs.astype(np.float32), err.astype(np.float32), true


# ---------------------------------------------------------------------------
# Emulator
# ---------------------------------------------------------------------------


def load_emulator(emulator_path, band_names):
    from arachne.emulator.parrot_emulator import ParrotEmulator

    emu = ParrotEmulator.load(emulator_path)
    if emu.param_names != SPS_PARAM_NAMES:
        raise ValueError(f"Emulator params {emu.param_names} != {SPS_PARAM_NAMES}")
    missing = [b for b in band_names if b not in emu.band_names]
    if missing:
        raise ValueError(f"Bands not in emulator: {missing}")
    band_idx = np.array([emu.band_names.index(b) for b in band_names], np.int32)
    return emu, band_idx


# ---------------------------------------------------------------------------
# Log-posterior
# ---------------------------------------------------------------------------


def make_log_posterior(emulator, band_idx, obs, err, log_prior_fn, min_frac_err=0.0):
    bidx = jnp.asarray(band_idx, jnp.int32)
    obs_j, err_j = jnp.asarray(obs), jnp.asarray(err)
    mask = (err_j < OBS_MASK_THRESH).astype(jnp.float32)
    # Apply minimum fractional error floor to prevent pure photon-noise errors
    # from creating an unnavigably sharp posterior. Accounts for emulator
    # approximation error, calibration uncertainty, and model incompleteness.
    if min_frac_err > 0.0:
        err_j = jnp.maximum(err_j, min_frac_err * jnp.abs(obs_j))
    inv_var = 1.0 / (err_j ** 2)
    log_range = jnp.log(_HIGHS - _LOWS)

    def log_posterior(theta):
        x = to_phys(theta)
        pred = emulator.predict(x[None, :])[0][bidx]
        resid = obs_j - pred
        chi2 = jnp.sum(mask * resid * resid * inv_var)
        log_norm = jnp.sum(mask * (-0.5 * LOG2PI - jnp.log(err_j)))
        log_jac = jnp.sum(log_range + jax.nn.log_sigmoid(theta) + jax.nn.log_sigmoid(-theta))
        return log_norm - 0.5 * chi2 + log_jac + log_prior_fn(x)

    return log_posterior


def reduced_chi2(theta, obs, err, emulator, band_idx, min_frac_err=0.0):
    x = to_phys(jnp.asarray(theta))
    pred = np.asarray(emulator.predict(x[None, :]))[0][np.asarray(band_idx)]
    mask = np.asarray(err) < OBS_MASK_THRESH
    eff_err = np.maximum(err, min_frac_err * np.abs(obs)) if min_frac_err > 0 else np.asarray(err)
    chi2 = float(np.sum(mask * ((np.asarray(obs) - pred) / eff_err) ** 2))
    dof = max(int(mask.sum()) - len(SPS_PARAM_NAMES), 1)
    return chi2 / dof, pred


# ---------------------------------------------------------------------------
# Redshift-grid initialiser
# ---------------------------------------------------------------------------


def robust_init(obs, err, emulator, band_idx, n_z=60, min_frac_err=0.0):
    """Grid over (redshift, Av), amplitude-solve log_mass at each, take best chi2.

    Gridding over Av as well as z prevents the initialiser from failing on dusty
    galaxies, where fixing Av=0.3 causes the model to over-predict the blue bands
    by factors of 10-30, producing a meaningless chi2 minimum.

    Returns (theta0, info_dict).
    """
    z_lo, z_hi = PARAM_BOUNDS["redshift"]
    lm_lo, lm_hi = PARAM_BOUNDS["log_mass"]
    z_grid = np.geomspace(max(z_lo, 0.05), z_hi, n_z)
    av_grid = np.array([0.0, 0.3, 0.7, 1.5, 3.0])

    ref = np.array([REF_INIT_PHYS[p] for p in SPS_PARAM_NAMES], np.float64)
    ref = np.clip(ref, np.asarray(_LOWS), np.asarray(_HIGHS))

    # Build full (n_z * n_av, P) candidate grid
    zz, aa = np.meshgrid(z_grid, av_grid, indexing="ij")   # (n_z, n_av)
    zz, aa = zz.ravel(), aa.ravel()
    n_cands = len(zz)
    cands = np.tile(ref, (n_cands, 1))
    cands[:, REDSHIFT_IDX] = zz
    cands[:, AV_IDX] = aa
    cands[:, LOGMASS_IDX] = REF_INIT_PHYS["log_mass"]

    preds = np.asarray(emulator.predict(jnp.asarray(cands, jnp.float32)))[:, np.asarray(band_idx)]
    mask = np.asarray(err) < OBS_MASK_THRESH
    eff_err = np.maximum(err, min_frac_err * np.abs(obs)) if min_frac_err > 0 else np.asarray(err)
    w = mask / eff_err ** 2
    num = np.sum(w * np.asarray(obs) * preds, axis=1)
    den = np.sum(w * preds ** 2, axis=1)
    s = np.where(den > 0, num / den, 1.0)                       # amplitude per (z, Av)
    lm = np.clip(REF_INIT_PHYS["log_mass"] + np.log10(np.clip(s, 1e-30, 1e30)), lm_lo, lm_hi)
    chi2 = np.sum(w * (np.asarray(obs)[None, :] - s[:, None] * preds) ** 2, axis=1)
    dof = max(int(mask.sum()) - len(SPS_PARAM_NAMES), 1)

    j = int(np.argmin(chi2))
    x0 = ref.copy()
    x0[REDSHIFT_IDX] = zz[j]
    x0[AV_IDX] = aa[j]
    x0[LOGMASS_IDX] = lm[j]
    theta0 = to_theta(jnp.asarray(x0, jnp.float32))
    info = {"best_z": float(zz[j]), "best_Av": float(aa[j]), "best_logmass": float(lm[j]),
            "init_redchi2": float(chi2[j] / dof)}
    return theta0, info


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------


def psd_metric(inv, diag_only, cond_cap=1e6):
    inv = 0.5 * (inv + inv.T)
    w, V = jnp.linalg.eigh(inv)
    floor = jnp.where(w[-1] > 0, w[-1] / cond_cap, 1.0)
    w = jnp.maximum(w, floor)
    out = (V * w) @ V.T
    out = jnp.where(jnp.isfinite(out).all(), out, jnp.eye(inv.shape[0]))
    if diag_only:
        out = jnp.diag(jnp.diag(out))
    return out


# ---------------------------------------------------------------------------
# Samplers (each returns samples (n_chains, n_samples, P) and an info dict)
# ---------------------------------------------------------------------------


def _chain_inits(theta0, key, n_chains, jitter):
    out = [theta0]
    for c in range(1, n_chains):
        out.append(theta0 + jitter * jax.random.normal(jax.random.fold_in(key, c), theta0.shape))
    return out


def sample_pathfinder_nuts(logpost, theta0, key, n_warmup, n_samples, n_chains,
                           target_accept, diag_metric):
    """Pathfinder for MAP init, then window_adaptation for step size + mass matrix.

    Using Pathfinder's fixed step size heuristic causes high divergence rates when
    the mass matrix is a poor estimate of the posterior curvature (common on real
    data). Running window_adaptation from the Pathfinder MAP combines the benefit of
    a good start position (Pathfinder) with properly tuned geometry (Stan-style
    dual averaging + mass matrix estimation).
    """
    pf_key, sample_key = jax.random.split(key)
    state, _ = pf_mod.approximate(pf_key, logpost, theta0, num_samples=256)
    pf_map = state.position
    print(f"  Pathfinder MAP logpost = {float(logpost(pf_map)):+.2f}  "
          f"(start logpost = {float(logpost(theta0)):+.2f})")

    samples, acc, div, steps = [], [], [], []
    for c, init in enumerate(_chain_inits(pf_map, sample_key, n_chains, 0.3)):
        wkey, skey = jax.random.split(jax.random.fold_in(sample_key, 100 + c))
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logpost,
            target_acceptance_rate=target_accept,
            is_mass_matrix_diagonal=diag_metric,
        )
        (state, params), _ = warmup.run(wkey, init, num_steps=n_warmup)
        nuts = blackjax.nuts(logpost, **params)

        @jax.jit
        def run(st, k):
            def step(s, kk):
                s, info = nuts.step(kk, s)
                return s, (s.position, info.acceptance_rate, info.is_divergent)
            _, out = jax.lax.scan(step, st, jax.random.split(k, n_samples))
            return out

        pos, ar, dv = run(state, skey)
        samples.append(np.asarray(pos)); acc.append(float(np.mean(ar)))
        div.append(float(np.mean(dv))); steps.append(float(params["step_size"]))
    return np.stack(samples), {"accept": acc, "divergence": div,
                               "step_size": steps, "metric": "diag" if diag_metric else "dense"}


def sample_window_nuts(logpost, theta0, key, n_warmup, n_samples, n_chains, target_accept, diag_metric):
    samples, acc, div, steps = [], [], [], []
    for c, init in enumerate(_chain_inits(theta0, key, n_chains, 0.3)):
        wkey, skey = jax.random.split(jax.random.fold_in(key, 100 + c))
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logpost,
            target_acceptance_rate=target_accept,
            is_mass_matrix_diagonal=diag_metric,
        )
        (state, params), _ = warmup.run(wkey, init, num_steps=n_warmup)
        nuts = blackjax.nuts(logpost, **params)

        @jax.jit
        def run(st, k):
            def step(s, kk):
                s, info = nuts.step(kk, s)
                return s, (s.position, info.acceptance_rate, info.is_divergent)
            _, out = jax.lax.scan(step, st, jax.random.split(k, n_samples))
            return out

        pos, ar, dv = run(state, skey)
        samples.append(np.asarray(pos)); acc.append(float(np.mean(ar)))
        div.append(float(np.mean(dv))); steps.append(float(params["step_size"]))
    return np.stack(samples), {"accept": acc, "divergence": div,
                               "step_size": steps, "metric": "diag" if diag_metric else "dense"}


def sample_mclmc(logpost, theta0, key, n_warmup, n_samples, n_chains):
    """MCLMC with built-in L/step-size tuning.  diagonal preconditioning only."""
    from blackjax.mcmc.integrators import isokinetic_mclachlan

    samples, tuned = [], []
    for c, init in enumerate(_chain_inits(theta0, key, n_chains, 0.3)):
        ik, tk, rk = jax.random.split(jax.random.fold_in(key, 200 + c), 3)
        init_state = blackjax.mcmc.mclmc.init(position=init, logdensity_fn=logpost, rng_key=ik)

        def kernel(inverse_mass_matrix):
            return blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=logpost,
                integrator=isokinetic_mclachlan,
                inverse_mass_matrix=inverse_mass_matrix,
            )

        tuned_state, params = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, num_steps=n_warmup, state=init_state,
            rng_key=tk, diagonal_preconditioning=True,
        )
        # Guard (#761): tuning can wander to a junk point. Keep tuned L/step,
        # but restart from our good init if the tuned state is much worse.
        if float(logpost(tuned_state.position)) < float(logpost(init_state.position)) - 5.0:
            tuned_state = init_state

        alg = blackjax.mclmc(logpost, L=params.L, step_size=params.step_size)

        @jax.jit
        def run(st, k):
            def step(s, kk):
                s, _info = alg.step(kk, s)
                return s, s.position
            _, pos = jax.lax.scan(step, st, jax.random.split(k, n_samples))
            return pos

        samples.append(np.asarray(run(tuned_state, rk)))
        tuned.append((float(params.L), float(params.step_size)))
    return np.stack(samples), {"L_step": tuned, "metric": "diag (mclmc precond)"}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def split_rhat(samples):  # (C, S, P)
    C, S, P = samples.shape
    n = S // 2
    if n < 2:
        return np.full(P, np.nan)
    x = samples[:, : 2 * n, :].reshape(C, 2, n, P).reshape(2 * C, n, P)
    m = 2 * C
    cm, cv = x.mean(1), x.var(1, ddof=1)
    gm = cm.mean(0)
    B = n * ((cm - gm) ** 2).sum(0) / (m - 1)
    W = cv.mean(0)
    return np.sqrt(((n - 1) / n * W + B / n) / np.where(W > 0, W, np.nan))


def diagnostics(samples):
    """Return (rhat (P,), ess (P,)) using blackjax.diagnostics if available."""
    try:
        from blackjax.diagnostics import potential_scale_reduction
        rhat = np.asarray(potential_scale_reduction(samples, chain_axis=0, sample_axis=1))
    except Exception:
        rhat = split_rhat(samples)
    try:
        from blackjax.diagnostics import effective_sample_size
        ess = np.asarray(effective_sample_size(samples, chain_axis=0, sample_axis=1))
    except Exception:
        ess = np.full(samples.shape[-1], np.nan)
    return rhat, ess


def print_sed(obs, err, pred, band_names, min_frac_err=0.0):
    """Print SED table. (o-p)/e uses effective errors (with frac floor applied)."""
    eff_err = np.maximum(err, min_frac_err * np.abs(obs)) if min_frac_err > 0 else err
    print(f"  {'band':<26} {'obs':>11} {'eff_err':>10} {'pred':>11} {'(o-p)/e':>8}")
    for b, name in enumerate(band_names):
        if err[b] < OBS_MASK_THRESH:
            print(f"  {name:<26} {obs[b]:11.4g} {eff_err[b]:10.3g} {pred[b]:11.4g} "
                  f"{(obs[b] - pred[b]) / eff_err[b]:8.2f}")
        else:
            print(f"  {name:<26} {'(masked)':>11}")


def print_recovery(phys, true_phys):
    print(f"\n  {'parameter':<22}{'truth':>10}{'median':>10}{'-1sig':>9}{'+1sig':>9}{'bias/sig':>9}")
    print("  " + "-" * 68)
    for i, name in enumerate(SPS_PARAM_NAMES):
        med = np.median(phys[:, i])
        lo, hi = np.percentile(phys[:, i], [16, 84])
        sig = (hi - lo) / 2 or np.nan
        truth = "" if true_phys is None else f"{true_phys[i]:10.4f}"
        bias = "" if true_phys is None else f"{(med - true_phys[i]) / sig:9.2f}"
        print(f"  {name:<22}{truth}{med:10.4f}{lo - med:9.4f}{hi - med:9.4f}{bias}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Fit a single galaxy SED (debuggable).")
    ap.add_argument("catalogue", nargs="?", help="catalogue (FITS/CSV/HDF5)")
    ap.add_argument("config", nargs="?", help="band config JSON")
    ap.add_argument("--emulator", default=str(DEFAULT_EMULATOR))
    ap.add_argument("--index", type=int, default=None, help="row index of galaxy")
    ap.add_argument("--id", default=None, help="galaxy id (needs id_col in config)")
    ap.add_argument("--mock", action="store_true", help="fit a synthetic galaxy (known truth)")
    ap.add_argument("--sampler", choices=["pathfinder_nuts", "window_nuts", "mclmc"],
                    default="pathfinder_nuts")
    ap.add_argument("--n-warmup", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--step-size", type=float, default=0.0, help="fixed NUTS step (pathfinder_nuts); 0=auto")
    ap.add_argument("--dense-metric", action="store_true")
    ap.add_argument("--no-mass-init", action="store_true", help="start at domain midpoint instead")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emu_path = Path(args.emulator)
    if not emu_path.exists():
        raise FileNotFoundError(emu_path)

    # ---- data ----
    if args.mock:
        cfg = {"flux_unit": "nJy", "id_col": None,
               "bands": {b: {} for b in [
                   "JWST/NIRCam.F090W", "JWST/NIRCam.F115W", "JWST/NIRCam.F150W",
                   "JWST/NIRCam.F200W", "JWST/NIRCam.F277W", "JWST/NIRCam.F356W",
                   "JWST/NIRCam.F444W"]},
               "resolved_priors": resolve_priors({})}
        band_names = list(cfg["bands"].keys())
        emulator, band_idx = load_emulator(emu_path, band_names)
        obs, err, true_phys = make_mock(emulator, band_idx, args.seed)
        gid = "mock"
    else:
        if not args.catalogue or not args.config:
            ap.error("catalogue and config required unless --mock")
        cfg = load_config(Path(args.config))
        obs, err, gid, band_names = load_one_galaxy(
            Path(args.catalogue), cfg, index=args.index, gal_id=args.id)
        emulator, band_idx = load_emulator(emu_path, band_names)
        true_phys = None

    P = len(SPS_PARAM_NAMES)
    print(f"\n=== Galaxy {gid} | {int((np.asarray(err) < OBS_MASK_THRESH).sum())}/{len(band_names)} "
          f"bands | sampler={args.sampler} ===")

    min_frac_err = float(cfg.get("min_frac_err", 0.05))
    print(f"Min fractional error floor: {min_frac_err:.1%}")
    log_post = make_log_posterior(emulator, band_idx, obs, err,
                                  make_log_prior_fn(cfg["resolved_priors"]),
                                  min_frac_err=min_frac_err)

    # ---- initialisation ----
    if args.no_mass_init:
        theta0 = jnp.zeros(P, jnp.float32)
        print("Init: domain midpoint (--no-mass-init)")
    else:
        theta0, iinfo = robust_init(obs, err, emulator, band_idx, min_frac_err=min_frac_err)
        print(f"Init (z,Av-grid): z={iinfo['best_z']:.3f}  Av={iinfo['best_Av']:.2f}  "
              f"log_mass={iinfo['best_logmass']:.2f}  reduced_chi2={iinfo['init_redchi2']:.2f}")
    rc0, pred0 = reduced_chi2(theta0, obs, err, emulator, band_idx, min_frac_err=min_frac_err)
    print(f"Reduced chi2 at init: {rc0:.2f}")
    print_sed(obs, err, pred0, band_names, min_frac_err=min_frac_err)

    # ---- sample ----
    key = jax.random.PRNGKey(args.seed)
    print(f"\nSampling: {args.sampler}  chains={args.n_chains}  "
          f"warmup={args.n_warmup}  samples={args.n_samples}")
    if args.sampler == "pathfinder_nuts":
        samples, info = sample_pathfinder_nuts(
            log_post, theta0, key, args.n_warmup, args.n_samples, args.n_chains,
            args.target_accept, not args.dense_metric)
    elif args.sampler == "window_nuts":
        samples, info = sample_window_nuts(
            log_post, theta0, key, args.n_warmup, args.n_samples, args.n_chains,
            args.target_accept, not args.dense_metric)
    else:
        samples, info = sample_mclmc(
            log_post, theta0, key, args.n_warmup, args.n_samples, args.n_chains)

    # ---- diagnostics ----
    rhat, ess = diagnostics(samples)
    phys = np.asarray(jax.vmap(to_phys)(jnp.asarray(samples.reshape(-1, P))))
    med_theta = np.median(samples.reshape(-1, P), axis=0)
    rc_post, pred_post = reduced_chi2(med_theta, obs, err, emulator, band_idx, min_frac_err=min_frac_err)

    print(f"\n--- diagnostics ({args.sampler}) ---")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  reduced chi2 @ posterior median: {rc_post:.2f}")
    print(f"  max R-hat: {np.nanmax(rhat):.3f}   "
          f"min ESS: {np.nanmin(ess):.0f}   ({samples.shape[0]}x{samples.shape[1]} draws)")
    worst = int(np.nanargmax(rhat))
    print(f"  worst-mixing param: {SPS_PARAM_NAMES[worst]} (R-hat={rhat[worst]:.3f}, ESS={ess[worst]:.0f})")
    print("\n  SED at posterior median:")
    print_sed(obs, err, pred_post, band_names, min_frac_err=min_frac_err)
    print_recovery(phys, true_phys)

    # ---- save ----
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"samples_{gid}.hdf5"
    with h5py.File(out, "w") as f:
        f.create_dataset("theta_samples", data=samples.astype(np.float32), compression="gzip")
        f.create_dataset("rhat", data=np.asarray(rhat, np.float32))
        f.create_dataset("ess", data=np.asarray(ess, np.float32))
        f.create_dataset("obs_flux", data=np.asarray(obs, np.float32))
        f.create_dataset("flux_err", data=np.asarray(err, np.float32))
        f.attrs["param_names"] = [s.encode() for s in SPS_PARAM_NAMES]
        f.attrs["band_names"] = [s.encode() for s in band_names]
        f.attrs["param_bounds_lo"] = np.asarray(_LOWS)
        f.attrs["param_bounds_hi"] = np.asarray(_HIGHS)
        f.attrs["sampler"] = args.sampler
        f.attrs["reduced_chi2_median"] = rc_post
        if true_phys is not None:
            f.attrs["true_params"] = np.asarray(true_phys, np.float32)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
