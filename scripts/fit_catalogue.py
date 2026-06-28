#!/usr/bin/env python3
r"""Production SED fitting for large galaxy catalogues.

Two samplers are available:

  --sampler nuts (default)
      GPU-batched multi-path Pathfinder + multi-chain NUTS.
      Fits thousands of galaxies per minute on a single GPU.
      Outputs unconstrained samples; use sigmoid to recover physical params.

  --sampler nss
      Per-galaxy Nested Slice Sampling (blackjax.nss).
      Gradient-free; handles multimodal posteriors natively; provides logZ.
      Runs sequentially (one galaxy at a time); ~27x slower per galaxy than
      the batched NUTS pipeline, but much better R-hat on multimodal SEDs.
      Outputs physical-space samples directly.

Key features (both samplers)
-----------------------------
* Full L-BFGS inverse-Hessian from Pathfinder, not just the diagonal alpha.
* Multi-path Pathfinder per galaxy (best-ELBO path selected).
* Multi-chain NUTS with per-galaxy dual-averaging step-size adaptation.
* Negative observed fluxes are kept (valid Gaussian measurements).
* Configurable per-parameter priors (uniform, loguniform, normal,
  studentt, halfnormal, exponential, lognormal).
* Reduced chi2 at the Pathfinder MAP (fit-quality flag).

Transform / prior contract (NUTS)
----------------------------------
The NUTS sampling domain for every parameter is fixed to the emulator's training
bounds (PARAM_BOUNDS) via a sigmoid transform.  This guarantees emulator
inputs are always in-domain.  The chosen prior is a density shape applied
over that domain; priors with support beyond the bounds are implicitly
truncated (truncation normalisation is dropped -- within-galaxy inference
is exact; absolute log-density / ELBO values are unnormalised constants).

NSS operates directly in physical space x ∈ [lo, hi].  Out-of-bound
proposals return loglikelihood=-inf so the HRSS sampler shrinks back
inside the domain without querying the emulator out-of-domain.

Band config JSON
----------------
    {
        "id_col":     "ID",
        "flux_unit":  "nJy",
        "bands": {
            "JWST/NIRCam.F090W": {"flux": "f_F090W", "err": "e_F090W"},
            ...
        },
        "priors": {                         # optional; merged over defaults
            "Av":             {"dist": "exponential", "scale": 0.5},
            "logsfr_ratio_0": {"dist": "studentt", "df": 2, "loc": 0.0, "scale": 0.3}
        }
    }

    flux_unit: "nJy" | "uJy" | "mJy" | "Jy" | "ABmag"

    Supported prior "dist" values:
        uniform                                  (flat over the domain)
        loguniform                               (~1/x; requires bound_lo > 0)
        normal       loc, scale                  (truncated to the domain)
        studentt     df, loc, scale              (truncated to the domain)
        halfnormal   scale                       (intended for bound_lo ~ 0)
        exponential  scale  (= mean)             (intended for bound_lo ~ 0)
        lognormal    loc, scale                  (requires bound_lo > 0)

Usage
-----
    # NUTS (default): batched Pathfinder + NUTS
    python scripts/fit_catalogue.py catalogue.fits bands.json
    python scripts/fit_catalogue.py catalogue.fits bands.json \\
        --n-samples 500 --n-warmup 300 --n-chains 2 --n-paths 4 \\
        --batch-size 2000

    # NSS: nested slice sampling, per-galaxy sequential
    python scripts/fit_catalogue.py catalogue.fits bands.json --sampler nss \\
        --num-live 500 --num-inner-steps 24 --num-delete 50 \\
        --termination -3.0 --n-galaxies 50

    # Pathfinder MAP only (NUTS sampler, skip sampling)
    python scripts/fit_catalogue.py catalogue.fits bands.json --pathfinder-only

    # Print a config template:
    python scripts/fit_catalogue.py --print-config-template \\
        --emulator scripts/outputs/emulators/parrot_emulator.eqx

NUTS output HDF5 layout
-----------------------
    /galaxy_id          (N,)            identifier from catalogue (int or str)
    /theta_map          (N, P)          best-path Pathfinder mean, unconstrained
    /inv_mass           (N, P, P)       dense Pathfinder inverse mass matrix
    /elbo               (N,)            best-path ELBO (within-galaxy comparable)
    /pathfinder_ok      (N,)            bool: best path returned a finite result
    /theta_samples      (N, C, S, P)    NUTS samples, unconstrained (omitted if
                                        --pathfinder-only)
    /accept_rate        (N, C)          mean NUTS acceptance per chain
    /step_size          (N, C)          adapted NUTS step size per chain
    /divergent_frac     (N, C)          fraction of divergent transitions
    /rhat               (N, P)          split-R-hat per parameter (physical space)
    /reduced_chi2_map   (N,)            reduced chi2 at the Pathfinder MAP
    attrs: param_names, band_names, param_bounds_lo, param_bounds_hi,
           prior_spec (json), emulator_path, run_timestamp, n_galaxies,
           n_samples, n_warmup, n_chains, n_paths, batch_size, target_accept

    To recover physical parameters from /theta_samples:
        lo  = f["param_bounds_lo"][:]
        hi  = f["param_bounds_hi"][:]
        sps = lo + (hi - lo) / (1 + exp(-theta_samples))   # sigmoid

NSS output HDF5 layout
-----------------------
    /galaxy_id          (N,)
    /nss_samples        (N, S, P)   posterior samples in physical space
    /nss_logZ           (N,)        log evidence point estimate
    /nss_logZ_err       (N,)        MC std of logZ over 100 shrinkage draws
    /nss_ess            (N,)        effective sample size
    /nss_n_steps        (N,)        NS while-loop iterations
    /nss_n_dead         (N,)        total dead particles accumulated
    /nss_time           (N,)        wall seconds (excl. XLA compilation)
    /nss_rhat           (N, P)      split-R-hat on resampled posterior
    attrs: param_names, band_names, param_bounds_lo, param_bounds_hi,
           prior_spec, emulator_path, run_timestamp, n_galaxies, sampler,
           num_live, num_inner_steps, num_delete, termination, n_samples_nss
"""

from __future__ import annotations

import os

# cuSolver (used by Pathfinder + dense NUTS metric) allocates outside XLA's
# preallocated pool.  Allocate on demand so cuSolver has headroom.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import json
import math
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

try:
    import blackjax
    import blackjax.vi.pathfinder as pf_mod
    from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
except ImportError as e:
    raise ImportError("pip install blackjax") from e

# NSS utilities (only used when --sampler nss)
try:
    from blackjax.ns.utils import finalise as _ns_finalise
    from blackjax.ns.utils import log_weights as _ns_log_weights
    from blackjax.ns.utils import sample as _ns_resample
    _NSS_AVAILABLE = True
except ImportError:
    _NSS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EMULATOR  = Path("scripts/outputs/emulators/parrot_emulator.eqx")
DEFAULT_OUTPUT    = Path("outputs/catalogue_fit/results.hdf5")
DEFAULT_XLA_CACHE = Path.home() / ".cache" / "arachne_xla"


# ---------------------------------------------------------------------------
# XLA persistent compilation cache
# ---------------------------------------------------------------------------

def setup_xla_cache(cache_dir: Path) -> None:
    """Enable JAX's persistent XLA compilation cache.

    Compiled GPU binaries are stored on disk keyed by a hash of the HLO
    program, XLA flags, JAX version, and GPU architecture.  On a cache hit
    the ~10-20 minute NSS first-compilation is reduced to a few seconds.

    The cache is invalidated automatically when any of those inputs change
    (different JAX version, different GPU, different num_inner_steps /
    num_delete / band count, etc.).  Different parameter combinations get
    their own cache entries and do not evict each other.

    Args:
        cache_dir: Directory for the cache.  Created if it does not exist.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    # Cache every compilation regardless of duration or output size.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    print(f"XLA cache: {cache_dir}")

MISSING_SIGMA: float = 1e10
OBS_MASK_THRESH: float = MISSING_SIGMA * 0.5

LOG2PI: float = math.log(2.0 * math.pi)

SPS_PARAM_NAMES: list[str] = [
    "redshift",
    "log_mass",
    "slope",
    "fesc_lya",
    "dust_bump_amplitude",
    "log10metallicity",
    "Av",
    "logsfr_ratio_0",
    "logsfr_ratio_1",
    "logsfr_ratio_2",
    "logsfr_ratio_3",
    "logsfr_ratio_4",
]
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "redshift":            (0.01, 14.0),
    "log_mass":            (4.0,  12.0),
    "slope":               (-0.3,  1.1),
    "fesc_lya":            (0.0,   1.0),
    "dust_bump_amplitude": (0.0,   5.0),
    "log10metallicity":    (-4.0, -1.39),
    "Av":                  (0.001, 5.0),
    "logsfr_ratio_0":      (-10.0, 10.0),
    "logsfr_ratio_1":      (-10.0, 10.0),
    "logsfr_ratio_2":      (-10.0, 10.0),
    "logsfr_ratio_3":      (-10.0, 10.0),
    "logsfr_ratio_4":      (-10.0, 10.0),
}

REDSHIFT_IDX: int = SPS_PARAM_NAMES.index("redshift")
LOGMASS_IDX:  int = SPS_PARAM_NAMES.index("log_mass")
AV_IDX:       int = SPS_PARAM_NAMES.index("Av")

DEFAULT_PRIORS: dict[str, dict] = {
    "redshift":            {"dist": "uniform"},
    "log_mass":            {"dist": "uniform"},
    "slope":               {"dist": "uniform"},
    "fesc_lya":            {"dist": "uniform"},
    "dust_bump_amplitude": {"dist": "uniform"},
    "log10metallicity":    {"dist": "uniform"},
    "Av":                  {"dist": "uniform"},
    "logsfr_ratio_0":      {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
    "logsfr_ratio_1":      {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
    "logsfr_ratio_2":      {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
    "logsfr_ratio_3":      {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
    "logsfr_ratio_4":      {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
}

FLUX_UNIT_TO_NJY: dict[str, float] = {
    "nJy": 1.0,
    "uJy": 1e3, "ujy": 1e3, "µJy": 1e3,
    "mJy": 1e6,
    "Jy":  1e9,
}

_KNOWN_DISTS = {
    "uniform", "loguniform", "normal", "studentt",
    "halfnormal", "exponential", "lognormal",
}


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def validate_prior_spec(name: str, spec: dict, bounds: tuple[float, float]) -> None:
    dist = spec.get("dist", "uniform").lower()
    lo, hi = bounds
    if dist not in _KNOWN_DISTS:
        raise ValueError(f"{name}: unknown prior dist {dist!r}. Use one of {sorted(_KNOWN_DISTS)}.")
    if dist in ("normal", "studentt", "halfnormal", "exponential", "lognormal"):
        if float(spec.get("scale", 0.0)) <= 0.0:
            raise ValueError(f"{name}: {dist} requires scale > 0.")
    if dist == "studentt" and float(spec.get("df", 0.0)) <= 0.0:
        raise ValueError(f"{name}: studentt requires df > 0.")
    if dist in ("loguniform", "lognormal") and lo <= 0.0:
        raise ValueError(f"{name}: {dist} requires the lower bound > 0 (bound_lo={lo}).")
    if dist in ("halfnormal", "exponential") and lo < 0.0:
        print(f"  ! {name}: {dist} prior on a domain that includes negatives "
              f"(bound_lo={lo}); shape may be unintended.")


def resolve_priors(config: dict) -> dict[str, dict]:
    priors: dict[str, dict] = {
        p: dict(DEFAULT_PRIORS.get(p, {"dist": "uniform"})) for p in SPS_PARAM_NAMES
    }
    user = config.get("priors", {}) or {}
    for k, v in user.items():
        if k not in SPS_PARAM_NAMES:
            raise ValueError(f"priors: unknown parameter {k!r}. Valid: {SPS_PARAM_NAMES}")
        priors[k] = dict(v)
    for p in SPS_PARAM_NAMES:
        validate_prior_spec(p, priors[p], PARAM_BOUNDS[p])
    return priors


def _build_prior_logprob(spec: dict, lo: float, hi: float):
    dist = spec.get("dist", "uniform").lower()
    if dist == "uniform":
        return lambda x: jnp.zeros((), dtype=x.dtype)
    if dist == "loguniform":
        return lambda x: -jnp.log(x)
    if dist == "normal":
        loc = float(spec.get("loc", 0.0))
        scale = float(spec["scale"])
        return lambda x: -0.5 * LOG2PI - math.log(scale) - 0.5 * ((x - loc) / scale) ** 2
    if dist == "studentt":
        df = float(spec["df"])
        loc = float(spec.get("loc", 0.0))
        scale = float(spec["scale"])
        c = (math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df)
             - 0.5 * math.log(df * math.pi) - math.log(scale))
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


def make_log_prior_fn(prior_specs: dict[str, dict]):
    """Return log_prior(x_phys) summing per-parameter prior densities."""
    fns = [
        _build_prior_logprob(prior_specs[p], PARAM_BOUNDS[p][0], PARAM_BOUNDS[p][1])
        for p in SPS_PARAM_NAMES
    ]

    def log_prior(x: jnp.ndarray) -> jnp.ndarray:
        total = jnp.zeros((), dtype=x.dtype)
        for i, fn in enumerate(fns):
            total = total + fn(x[i])
        return total

    return log_prior


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    if "bands" not in cfg:
        raise ValueError("Config must contain a 'bands' key.")
    cfg.setdefault("id_col", None)
    cfg.setdefault("flux_unit", "nJy")
    unit = cfg["flux_unit"]
    if unit not in FLUX_UNIT_TO_NJY and unit != "ABmag":
        raise ValueError(f"Unknown flux_unit {unit!r}. Use one of: {list(FLUX_UNIT_TO_NJY)} or 'ABmag'.")
    cfg.setdefault("min_frac_err", 0.05)
    cfg["resolved_priors"] = resolve_priors(cfg)
    non_uniform = {p: s for p, s in cfg["resolved_priors"].items() if s.get("dist") != "uniform"}
    print(f"Config: {len(cfg['bands'])} bands, flux_unit={unit!r}  "
          f"min_frac_err={cfg['min_frac_err']:.1%}")
    if non_uniform:
        print("  Non-uniform priors: "
              + ", ".join(f"{p}={s['dist']}" for p, s in non_uniform.items()))
    return cfg


def print_config_template(emulator_path: Path) -> None:
    from arachne.emulator.parrot_emulator import ParrotEmulator
    emu = ParrotEmulator.load(emulator_path)
    bands = {}
    for b in emu.band_names:
        safe = b.replace("/", "_").replace(".", "_")
        bands[b] = {"flux": f"{safe}_FLUX", "err": f"{safe}_FLUXERR"}
    template = {
        "id_col": "ID",
        "flux_unit": "nJy",
        "bands": bands,
        "priors": {p: DEFAULT_PRIORS[p] for p in SPS_PARAM_NAMES},
    }
    print(json.dumps(template, indent=2))


# ---------------------------------------------------------------------------
# Catalogue loading
# ---------------------------------------------------------------------------

def load_catalogue(catalogue_path: Path, config: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """Load fluxes and errors (in nJy).  Keeps negative fluxes; masks only non-finite or err<=0."""
    from astropy.table import Table

    print(f"\nLoading catalogue: {catalogue_path}")
    t = Table.read(str(catalogue_path))
    N = len(t)
    print(f"  {N} rows, {len(t.colnames)} columns")

    unit = config["flux_unit"]
    factor = FLUX_UNIT_TO_NJY.get(unit, 1.0)
    use_abmag = unit == "ABmag"

    band_names = list(config["bands"].keys())
    n_bands = len(band_names)
    obs_flux = np.zeros((N, n_bands), dtype=np.float32)
    flux_err = np.full((N, n_bands), MISSING_SIGMA, dtype=np.float32)

    for b, band in enumerate(band_names):
        flux_col = config["bands"][band]["flux"]
        err_col  = config["bands"][band]["err"]
        f_raw = np.asarray(t[flux_col], dtype=np.float64)
        e_raw = np.asarray(t[err_col],  dtype=np.float64)
        if use_abmag:
            f_nJy = 10.0 ** ((8.9 - f_raw) / 2.5) * 1e9
            e_nJy = f_nJy * np.abs(e_raw) * np.log(10.0) / 2.5
        else:
            f_nJy = f_raw * factor
            e_nJy = e_raw * factor
        valid = np.isfinite(f_nJy) & np.isfinite(e_nJy) & (e_nJy > 0)
        obs_flux[valid, b] = f_nJy[valid].astype(np.float32)
        flux_err[valid, b] = e_nJy[valid].astype(np.float32)
        n_bad = int((~valid).sum())
        if n_bad:
            print(f"  {band}: {N - n_bad}/{N} usable ({n_bad} masked)")

    id_col = config.get("id_col")
    if id_col and id_col in t.colnames:
        galaxy_ids = list(t[id_col])
    else:
        galaxy_ids = list(range(N))
    return obs_flux, flux_err, galaxy_ids


# ---------------------------------------------------------------------------
# Emulator + band resolution
# ---------------------------------------------------------------------------

def load_emulator_and_band_indices(emulator_path: Path, band_names: list[str]) -> tuple:
    from arachne.emulator.parrot_emulator import ParrotEmulator
    emu = ParrotEmulator.load(emulator_path)
    print(f"\nLoaded ParrotEmulator: {len(emu.param_names)} params, {len(emu.band_names)} bands")
    if emu.param_names != SPS_PARAM_NAMES:
        raise ValueError(f"Emulator params {emu.param_names} != expected {SPS_PARAM_NAMES}")
    all_bands = emu.band_names
    missing = [b for b in band_names if b not in all_bands]
    if missing:
        raise ValueError(f"Bands not found in emulator: {missing}")
    band_idx = np.array([all_bands.index(b) for b in band_names], dtype=np.int32)
    return emu, band_idx


# ---------------------------------------------------------------------------
# Log-posterior (NUTS — unconstrained space)
# ---------------------------------------------------------------------------

def make_log_posterior_fn(emulator, band_idx: np.ndarray, log_prior_fn,
                          min_frac_err: float = 0.0):
    """Return log_posterior(theta, obs_flux, flux_err).

    theta is in unconstrained space; physical = lo + (hi-lo)*sigmoid(theta).
    Jacobian of the sigmoid transform is included so the sampler sees a flat
    prior density in the physical domain.
    """
    lows  = jnp.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    highs = jnp.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    log_range = jnp.log(highs - lows)
    _bidx = jnp.array(band_idx, dtype=jnp.int32)

    def log_posterior(theta, obs_flux, flux_err):
        x = lows + (highs - lows) * jax.nn.sigmoid(theta)
        pred = emulator.predict(x[None, :])[0][_bidx]
        mask = (flux_err < OBS_MASK_THRESH).astype(theta.dtype)
        eff_err = (jnp.maximum(flux_err, min_frac_err * jnp.abs(obs_flux))
                   if min_frac_err > 0 else flux_err)
        chi2 = jnp.sum(mask * (obs_flux - pred) ** 2 / eff_err ** 2)
        log_norm = jnp.sum(mask * (-0.5 * LOG2PI - jnp.log(eff_err)))
        log_jac = jnp.sum(
            log_range + jax.nn.log_sigmoid(theta) + jax.nn.log_sigmoid(-theta)
        )
        return log_norm - 0.5 * chi2 + log_jac + log_prior_fn(x)

    return log_posterior


# ---------------------------------------------------------------------------
# Log-likelihood (NSS — physical space)
# ---------------------------------------------------------------------------

def make_log_likelihood_physical(emulator, band_idx: np.ndarray, min_frac_err: float):
    """Return log_likelihood(x_phys, obs_flux, flux_err).

    Returns -inf for positions outside PARAM_BOUNDS so HRSS always shrinks
    back inside the domain without querying the emulator out-of-domain.
    """
    lows  = jnp.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    highs = jnp.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    _bidx = jnp.array(band_idx, dtype=jnp.int32)

    def log_likelihood(x_phys, obs_flux, flux_err):
        in_bounds = jnp.all((x_phys >= lows) & (x_phys <= highs))
        pred = emulator.predict(x_phys[None, :])[0][_bidx]
        mask = (flux_err < OBS_MASK_THRESH).astype(x_phys.dtype)
        eff_err = (jnp.maximum(flux_err, min_frac_err * jnp.abs(obs_flux))
                   if min_frac_err > 0 else flux_err)
        chi2 = jnp.sum(mask * (obs_flux - pred) ** 2 / eff_err ** 2)
        log_norm = jnp.sum(mask * (-0.5 * LOG2PI - jnp.log(eff_err)))
        return jnp.where(in_bounds, log_norm - 0.5 * chi2, -jnp.inf)

    return log_likelihood


# ---------------------------------------------------------------------------
# NUTS — compiled batch functions
# ---------------------------------------------------------------------------

def build_batch_fns(
    log_posterior_fn,
    n_params: int,
    n_samples: int,
    n_warmup: int,
    n_pf_samples: int,
    eps0: float,
    target_accept: float,
    vmap_paths: bool = False,
) -> tuple:
    """JIT-compile vmapped multi-path Pathfinder and multi-chain NUTS."""
    eye = jnp.eye(n_params, dtype=jnp.float32)

    def _pf_one(obs_flux, flux_err, theta_init, rng_key):
        def lp(theta): return log_posterior_fn(theta, obs_flux, flux_err)
        state, _ = pf_mod.approximate(rng_key, lp, theta_init, num_samples=n_pf_samples)
        inv = lbfgs_inverse_hessian_formula_1(state.alpha, state.beta, state.gamma)
        inv = 0.5 * (inv + inv.T) + 1e-8 * eye
        ok = (jnp.isfinite(state.position).all() & jnp.isfinite(inv).all()
              & jnp.isfinite(state.elbo))
        elbo = jnp.where(ok, state.elbo, -jnp.inf)
        return state.position, inv, elbo, ok

    def _pf_galaxy(obs_flux, flux_err, theta_inits, rng_keys):
        if vmap_paths:
            pos, inv, elbo, ok = jax.vmap(_pf_one, in_axes=(None, None, 0, 0))(
                obs_flux, flux_err, theta_inits, rng_keys)
        else:
            pos, inv, elbo, ok = jax.lax.map(
                lambda tk: _pf_one(obs_flux, flux_err, tk[0], tk[1]),
                (theta_inits, rng_keys),
            )
        j = jnp.argmax(elbo)
        return pos[j], inv[j], elbo[j], ok[j]

    batch_pf = jax.jit(jax.vmap(_pf_galaxy, in_axes=(0, 0, 0, 0)))

    log_eps0 = math.log(eps0)
    mu = math.log(10.0 * eps0)

    def da_init():
        return (jnp.array(0.0), jnp.array(log_eps0), jnp.array(log_eps0), jnp.array(0.0))

    def da_update(da, accept):
        m, log_step, log_step_bar, h_bar = da
        m = m + 1.0
        accept = jnp.clip(jnp.nan_to_num(accept, nan=0.0), 0.0, 1.0)
        w = 1.0 / (m + 10.0)
        h_bar = (1.0 - w) * h_bar + w * (target_accept - accept)
        log_step = mu - jnp.sqrt(m) / 0.05 * h_bar
        log_step = jnp.clip(log_step, -12.0, 2.0)
        eta = m ** (-0.75)
        log_step_bar = eta * log_step + (1.0 - eta) * log_step_bar
        return (m, log_step, log_step_bar, h_bar)

    def _nuts_one(obs_flux, flux_err, theta_init, inv_mass, rng_key):
        def lp(theta): return log_posterior_fn(theta, obs_flux, flux_err)
        warm_key, sample_key = jax.random.split(rng_key)
        state = blackjax.nuts(lp, step_size=eps0, inverse_mass_matrix=inv_mass).init(theta_init)

        if n_warmup > 0:
            def warm_step(carry, k):
                st, da = carry
                eps = jnp.exp(da[1])
                kern = blackjax.nuts(lp, step_size=eps, inverse_mass_matrix=inv_mass)
                st, info = kern.step(k, st)
                da = da_update(da, info.acceptance_rate)
                return (st, da), None
            (state, da), _ = jax.lax.scan(
                warm_step, (state, da_init()), jax.random.split(warm_key, n_warmup))
            eps_final = jnp.exp(da[2])
        else:
            eps_final = jnp.array(eps0)

        eps_final = jnp.where(jnp.isfinite(eps_final) & (eps_final > 0), eps_final, eps0)
        kern = blackjax.nuts(lp, step_size=eps_final, inverse_mass_matrix=inv_mass)

        def sample_step(st, k):
            st, info = kern.step(k, st)
            return st, (st.position, info.acceptance_rate, info.is_divergent)

        _, (samples, ar, div) = jax.lax.scan(
            sample_step, state, jax.random.split(sample_key, n_samples))
        return samples, jnp.mean(ar), eps_final, jnp.mean(div.astype(jnp.float32))

    nuts_chains = jax.vmap(_nuts_one, in_axes=(None, None, 0, None, 0))
    batch_nuts = jax.jit(jax.vmap(nuts_chains, in_axes=(0, 0, 0, 0, 0)))

    return batch_pf, batch_nuts


# ---------------------------------------------------------------------------
# NSS — compiled init + step functions
# ---------------------------------------------------------------------------

def build_nss_fns(
    log_prior_fn,
    log_like_fn,
    num_inner_steps: int,
    num_delete: int,
):
    """Return (nss_init_jit, nss_step_jit) compiled once for the (n_bands, n_params) shapes.

    Both take (obs_flux, flux_err) as explicit JAX arguments so the XLA binary
    is reused across all galaxies (single compilation, not one per galaxy).

    Note: first call triggers XLA compilation (~10-20 min for the full nested
    vmap→scan→while_loop→emulator graph). Subsequent calls reuse the binary.
    """
    @jax.jit
    def nss_init(initial_samples, obs_flux, flux_err):
        def ll(x): return log_like_fn(x, obs_flux, flux_err)
        algo = blackjax.nss(
            logprior_fn=log_prior_fn, loglikelihood_fn=ll,
            num_delete=num_delete, num_inner_steps=num_inner_steps,
        )
        return algo.init(initial_samples)

    @jax.jit
    def nss_step(rng_key, state, obs_flux, flux_err):
        def ll(x): return log_like_fn(x, obs_flux, flux_err)
        algo = blackjax.nss(
            logprior_fn=log_prior_fn, loglikelihood_fn=ll,
            num_delete=num_delete, num_inner_steps=num_inner_steps,
        )
        return algo.step(rng_key, state)

    return nss_init, nss_step


def _logz_from_weights(logw: jnp.ndarray) -> tuple[float, float]:
    """logZ point estimate and MC std from (N_particles, 100) log-weight matrix."""
    lw = jnp.nan_to_num(logw, nan=jnp.nan_to_num(logw).min())
    logz_draws = jax.scipy.special.logsumexp(lw, axis=0)  # (100,)
    return float(logz_draws.mean()), float(logz_draws.std())


def _ess_from_weights(logw: jnp.ndarray) -> float:
    """ESS = exp(2*logsumexp(w) - logsumexp(2w)) averaged over 100 MC draws."""
    lw_mean = jnp.nan_to_num(logw).mean(axis=-1)
    lw_mean = lw_mean - lw_mean.max()
    ls  = jax.scipy.special.logsumexp(lw_mean)
    ls2 = jax.scipy.special.logsumexp(2.0 * lw_mean)
    return float(jnp.exp(2.0 * ls - ls2))


def run_nss_galaxy(
    rng_key,
    obs_flux_i: np.ndarray,
    flux_err_i: np.ndarray,
    nss_init_fn,
    nss_step_fn,
    num_live: int,
    termination: float,
    n_samples_out: int,
) -> tuple:
    """Run NSS on one galaxy.  Returns (samples_phys, logz, logz_err, ess, n_steps, n_dead, elapsed)."""
    lows_np  = np.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=np.float32)
    highs_np = np.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=np.float32)
    P = len(SPS_PARAM_NAMES)

    obs_jax = jnp.asarray(obs_flux_i, dtype=jnp.float32)
    err_jax = jnp.asarray(flux_err_i, dtype=jnp.float32)

    rng_key, init_key = jax.random.split(rng_key)
    initial_samples = jax.random.uniform(
        init_key, (num_live, P),
        minval=jnp.asarray(lows_np), maxval=jnp.asarray(highs_np),
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

    jax.block_until_ready(state)
    elapsed = time.perf_counter() - t0

    final_state = _ns_finalise(state, dead)
    rng_key, w_key, s_key = jax.random.split(rng_key, 3)
    logw = _ns_log_weights(w_key, final_state)          # (N_total, 100)
    logz, logz_err = _logz_from_weights(logw)
    ess = _ess_from_weights(logw)

    resampled = _ns_resample(s_key, final_state, shape=n_samples_out)
    samples_phys = np.array(resampled.position, dtype=np.float32)   # (S, P)

    n_dead = (len(dead) * dead[0].particles.loglikelihood.shape[0]) if dead else 0
    return samples_phys, logz, logz_err, ess, n_steps, n_dead, elapsed


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def split_rhat(
    samples: np.ndarray,
    lows: np.ndarray | None = None,
    highs: np.ndarray | None = None,
) -> np.ndarray:
    """Split-R-hat per parameter.

    Args:
        samples: (B, C, S, P).  If lows/highs provided, applies sigmoid to
                 convert from unconstrained to physical space before computing.
                 Pass lows=None for samples already in physical space.
    Returns:
        (B, P) split-R-hat.
    """
    B, C, S, P = samples.shape
    n = S // 2
    if n < 2:
        return np.full((B, P), np.nan, dtype=np.float32)
    x = samples[:, :, : 2 * n, :].reshape(B, C, 2, n, P).reshape(B, 2 * C, n, P)
    if lows is not None and highs is not None:
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
        x = lows + (highs - lows) * sig
    m = 2 * C
    chain_mean  = x.mean(axis=2)
    chain_var   = x.var(axis=2, ddof=1)
    grand_mean  = chain_mean.mean(axis=1, keepdims=True)
    b_var = n * ((chain_mean - grand_mean) ** 2).sum(axis=1) / (m - 1)
    w = chain_var.mean(axis=1)
    var_plus = (n - 1) / n * w + b_var / n
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.sqrt(var_plus / np.where(w > 0, w, np.nan))
    return rhat.astype(np.float32)


def regularize_metric(
    inv: np.ndarray, cond_cap: float = 1e6, diag_only: bool = False
) -> tuple[np.ndarray, int, int]:
    """Project per-galaxy inverse-mass matrices to PSD with bounded conditioning."""
    B, P, _ = inv.shape
    inv = 0.5 * (inv + np.transpose(inv, (0, 2, 1)))
    w, V = np.linalg.eigh(inv)
    wmax = w[:, -1]
    bad = ~np.isfinite(wmax) | (wmax <= 0)
    floor = np.where(wmax > 0, wmax / cond_cap, 1.0)[:, None]
    floored_any = (w < floor).any(axis=1)
    w_reg = np.maximum(np.nan_to_num(w, nan=0.0), floor)
    out = (V * w_reg[:, None, :]) @ np.transpose(V, (0, 2, 1))
    out[bad] = np.eye(P, dtype=out.dtype)
    if diag_only:
        d = np.einsum("bii->bi", out).copy()
        out = np.zeros_like(out)
        out[:, np.arange(P), np.arange(P)] = d
    return out.astype(np.float32), int(bad.sum()), int(floored_any.sum())


def logmass_init_theta(
    obs: jnp.ndarray,
    err: jnp.ndarray,
    emulator,
    band_idx: np.ndarray,
    lows: jnp.ndarray,
    highs: jnp.ndarray,
    ref: float = 8.0,
) -> jnp.ndarray:
    """Per-galaxy unconstrained theta for log_mass from amplitude matching."""
    lo, hi = float(lows[LOGMASS_IDX]), float(highs[LOGMASS_IDX])
    x_ref = (0.5 * (lows + highs)).at[LOGMASS_IDX].set(ref)
    pred_ref = emulator.predict(x_ref[None, :])[0][jnp.asarray(band_idx, dtype=jnp.int32)]
    w = (err < OBS_MASK_THRESH).astype(obs.dtype) / (err ** 2)
    num = jnp.sum(w * obs * pred_ref, axis=1)
    den = jnp.sum(w * pred_ref ** 2, axis=1)
    s = jnp.where(den > 0, num / den, 1.0)
    lm = jnp.clip(ref + jnp.log10(jnp.clip(s, 1e-30, 1e30)), lo + 1e-3, hi - 1e-3)
    u = jnp.clip((lm - lo) / (hi - lo), 1e-4, 1.0 - 1e-4)
    return jnp.log(u / (1.0 - u))


def reduced_chi2_at(
    theta: np.ndarray,
    obs: np.ndarray,
    err: np.ndarray,
    emulator,
    band_idx: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    min_frac_err: float = 0.0,
) -> np.ndarray:
    """Reduced chi2 at given unconstrained theta, per galaxy."""
    lows_j  = jnp.asarray(lows)
    highs_j = jnp.asarray(highs)
    x    = lows_j + (highs_j - lows_j) * jax.nn.sigmoid(jnp.asarray(theta))
    pred = np.asarray(emulator.predict(x))[:, np.asarray(band_idx)]
    mask = np.asarray(err) < OBS_MASK_THRESH
    obs_np, err_np = np.asarray(obs), np.asarray(err)
    eff_err = np.maximum(err_np, min_frac_err * np.abs(obs_np)) if min_frac_err > 0 else err_np
    resid = (obs_np - pred) / eff_err
    chi2  = np.sum(mask * resid * resid, axis=1)
    dof   = np.maximum(mask.sum(axis=1) - len(SPS_PARAM_NAMES), 1)
    return (chi2 / dof).astype(np.float32)


def check_gradients(log_post_fn, obs_flux: np.ndarray, flux_err: np.ndarray,
                    n_check: int = 8, seed: int = 0) -> None:
    """Robust directional autodiff vs central-FD gradient check (near each galaxy's mode)."""
    P = len(SPS_PARAM_NAMES)
    val_fn  = jax.jit(lambda th, o, e: log_post_fn(th, o, e))
    grad_fn = jax.jit(jax.grad(lambda th, o, e: log_post_fn(th, o, e)))

    @jax.jit
    def ascend(th0, o, e):
        def body(t, _):
            g = jax.grad(lambda x: log_post_fn(x, o, e))(t)
            return t + 0.03 * g / (jnp.linalg.norm(g) + 1e-8), None
        t, _ = jax.lax.scan(body, th0, None, length=150)
        return t

    key = jax.random.PRNGKey(seed)
    print("\nGradient self-test (directional autodiff vs central FD, near each mode):")
    worst = 0.0
    n = min(n_check, obs_flux.shape[0])
    for i in range(n):
        o = jnp.asarray(obs_flux[i])
        e = jnp.asarray(flux_err[i])
        key, kth, kv, ko = jax.random.split(key, 4)
        th_map = ascend(jax.random.normal(kth, (P,)) * 0.5, o, e)
        offset = np.asarray(jax.random.normal(ko, (P,)))
        offset /= np.linalg.norm(offset)
        th = th_map + 0.5 * jnp.asarray(offset, dtype=jnp.float32)

        g  = np.asarray(grad_fn(th, o, e))
        v  = np.asarray(jax.random.normal(kv, (P,)))
        v /= np.linalg.norm(v)
        dd_ad = float(g @ v)
        f0 = float(val_fn(th, o, e))

        best_rel, best_fd, best_noise = float("inf"), float("nan"), float("nan")
        for eps in (3e-2, 1e-2, 3e-3):
            fp = float(val_fn(jnp.asarray(th + eps * v), o, e))
            fm = float(val_fn(jnp.asarray(th - eps * v), o, e))
            dd_fd  = (fp - fm) / (2 * eps)
            noise  = (abs(f0) + abs(fp) + abs(fm)) * 1.2e-7 / (2 * eps)
            scale  = max(abs(dd_ad), abs(dd_fd), 1e-12)
            rel    = max(0.0, abs(dd_ad - dd_fd) - 5 * noise) / scale
            if rel < best_rel:
                best_rel, best_fd, best_noise = rel, dd_fd, noise

        finite  = bool(np.isfinite(g).all() and np.isfinite(best_fd))
        suspect = (not finite) or best_rel > 0.1
        worst   = max(worst, best_rel if finite else float("inf"))
        flag = "  <-- SUSPECT" if suspect else ""
        print(f"  galaxy {i}: rel err = {best_rel:.2e}  "
              f"(ad={dd_ad:.3g} fd={best_fd:.3g} roundoff~{best_noise:.1g})  "
              f"finite={finite}{flag}")
    print(f"  worst rel err over {n} galaxies: {worst:.2e}")
    if not np.isfinite(worst) or worst > 0.1:
        print("  ! Gradients look genuinely inconsistent -- investigate the emulator.")
    else:
        print("  Emulator gradients are consistent.")


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _largest_divisor_leq(n: int, cap: int) -> int:
    cap = max(1, min(int(cap), n))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def _chunk_rows(batch_size: int, per_row_bytes: int, n_total: int,
                target_bytes: int = 4 << 20) -> int:
    rows = max(1, target_bytes // max(1, per_row_bytes))
    rows = _largest_divisor_leq(batch_size, rows)
    return max(1, min(rows, n_total))


def _write_common_attrs(f: h5py.File, n_galaxies: int, prior_specs: dict,
                        emulator_path: Path, band_names: list[str]) -> None:
    f.attrs["param_names"]      = [s.encode() for s in SPS_PARAM_NAMES]
    f.attrs["band_names"]       = [s.encode() for s in band_names]
    f.attrs["param_bounds_lo"]  = np.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES])
    f.attrs["param_bounds_hi"]  = np.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES])
    f.attrs["prior_spec"]       = json.dumps(prior_specs)
    f.attrs["emulator_path"]    = str(emulator_path)
    f.attrs["run_timestamp"]    = datetime.now(timezone.utc).isoformat()
    f.attrs["n_galaxies"]       = n_galaxies


def create_output_file_nuts(
    output_path: Path,
    n_galaxies: int,
    n_samples: int,
    n_chains: int,
    n_params: int,
    band_names: list[str],
    emulator_path: Path,
    batch_size: int,
    n_warmup: int,
    n_paths: int,
    target_accept: float,
    prior_specs: dict,
    galaxy_ids: list,
    pathfinder_only: bool,
) -> h5py.File:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(output_path, "w")
    try:
        f.create_dataset("galaxy_id", data=np.array(galaxy_ids, dtype=np.int64))
    except (ValueError, TypeError):
        dt = h5py.string_dtype()
        f.create_dataset("galaxy_id",
                         data=np.array([str(g) for g in galaxy_ids], dtype=object), dtype=dt)

    N, P, S, C = n_galaxies, n_params, n_samples, n_chains
    ckw = dict(compression="gzip", compression_opts=4)
    bs  = min(batch_size, N)

    r_map = _chunk_rows(bs, P * 4, N)
    f.create_dataset("theta_map",        shape=(N, P),    dtype=np.float32, chunks=(r_map, P), **ckw)
    r_inv = _chunk_rows(bs, P * P * 4, N)
    f.create_dataset("inv_mass",         shape=(N, P, P), dtype=np.float32, chunks=(r_inv, P, P), **ckw)
    f.create_dataset("elbo",             shape=(N,),      dtype=np.float32)
    f.create_dataset("pathfinder_ok",    shape=(N,),      dtype=bool)
    f.create_dataset("accept_rate",      shape=(N, C),    dtype=np.float32)
    f.create_dataset("step_size",        shape=(N, C),    dtype=np.float32)
    f.create_dataset("divergent_frac",   shape=(N, C),    dtype=np.float32)
    f.create_dataset("rhat",             shape=(N, P),    dtype=np.float32)
    f.create_dataset("reduced_chi2_map", shape=(N,),      dtype=np.float32)

    if not pathfinder_only:
        r_s = _chunk_rows(bs, C * S * P * 4, N)
        f.create_dataset("theta_samples", shape=(N, C, S, P), dtype=np.float32,
                         chunks=(r_s, C, S, P), **ckw)

    _write_common_attrs(f, N, prior_specs, emulator_path, band_names)
    f.attrs["sampler"]       = "nuts"
    f.attrs["n_samples"]     = n_samples
    f.attrs["n_warmup"]      = n_warmup
    f.attrs["n_chains"]      = n_chains
    f.attrs["n_paths"]       = n_paths
    f.attrs["batch_size"]    = batch_size
    f.attrs["target_accept"] = target_accept
    return f


def create_output_file_nss(
    output_path: Path,
    n_galaxies: int,
    n_samples_out: int,
    band_names: list[str],
    emulator_path: Path,
    prior_specs: dict,
    galaxy_ids: list,
    num_live: int,
    num_inner_steps: int,
    num_delete: int,
    termination: float,
) -> h5py.File:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(output_path, "w")
    try:
        f.create_dataset("galaxy_id", data=np.array(galaxy_ids, dtype=np.int64))
    except (ValueError, TypeError):
        dt = h5py.string_dtype()
        f.create_dataset("galaxy_id",
                         data=np.array([str(g) for g in galaxy_ids], dtype=object), dtype=dt)

    N, S, P = n_galaxies, n_samples_out, len(SPS_PARAM_NAMES)
    ckw = dict(compression="gzip", compression_opts=4)
    f.create_dataset("nss_samples",  shape=(N, S, P), dtype=np.float32, chunks=(1, S, P), **ckw)
    f.create_dataset("nss_logZ",     shape=(N,), dtype=np.float32)
    f.create_dataset("nss_logZ_err", shape=(N,), dtype=np.float32)
    f.create_dataset("nss_ess",      shape=(N,), dtype=np.float32)
    f.create_dataset("nss_n_steps",  shape=(N,), dtype=np.int32)
    f.create_dataset("nss_n_dead",   shape=(N,), dtype=np.int32)
    f.create_dataset("nss_time",     shape=(N,), dtype=np.float32)
    f.create_dataset("nss_rhat",     shape=(N, P), dtype=np.float32)

    _write_common_attrs(f, N, prior_specs, emulator_path, band_names)
    f.attrs["sampler"]          = "nss"
    f.attrs["num_live"]         = num_live
    f.attrs["num_inner_steps"]  = num_inner_steps
    f.attrs["num_delete"]       = num_delete
    f.attrs["termination"]      = termination
    f.attrs["n_samples_nss"]    = n_samples_out
    return f


# ---------------------------------------------------------------------------
# Main fitting loop — NUTS
# ---------------------------------------------------------------------------

def run_catalogue(
    obs_flux: np.ndarray,
    flux_err: np.ndarray,
    galaxy_ids: list,
    emulator,
    band_idx: np.ndarray,
    prior_specs: dict,
    output_path: Path,
    emulator_path: Path,
    batch_size: int,
    n_samples: int,
    n_warmup: int,
    n_chains: int,
    n_paths: int,
    n_pf_samples: int,
    eps0: float,
    target_accept: float,
    init_jitter: float,
    chain_jitter: float,
    seed: int,
    pathfinder_only: bool,
    vmap_pf_paths: bool = False,
    diag_metric: bool = True,
    min_frac_err: float = 0.05,
) -> None:
    N, n_bands = obs_flux.shape
    P = len(SPS_PARAM_NAMES)
    band_names = [emulator.band_names[int(i)] for i in band_idx]
    lows     = jnp.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    highs    = jnp.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    lows_np  = np.asarray(lows)
    highs_np = np.asarray(highs)

    mode_str     = "Pathfinder only" if pathfinder_only else "Pathfinder + NUTS"
    pf_path_mode = "parallel" if vmap_pf_paths else "sequential"
    print(f"\nFitting {N:,} galaxies  |  batch={batch_size}  paths={n_paths}  "
          f"chains={n_chains}  warmup={n_warmup}  samples={n_samples}")
    print(f"  {mode_str}  |  eps0={eps0:.4g}  target_accept={target_accept}  "
          f"pf_paths={pf_path_mode}  pf_samples={n_pf_samples}  "
          f"metric={'diagonal' if diag_metric else 'dense'}")

    log_prior_fn  = make_log_prior_fn(prior_specs)
    log_post_fn   = make_log_posterior_fn(emulator, band_idx, log_prior_fn, min_frac_err)
    batch_pf, batch_nuts = build_batch_fns(
        log_post_fn, P, n_samples, n_warmup, n_pf_samples, eps0, target_accept,
        vmap_paths=vmap_pf_paths,
    )

    h5 = create_output_file_nuts(
        output_path, N, n_samples, n_chains, P, band_names, emulator_path,
        batch_size, n_warmup, n_paths, target_accept, prior_specs, galaxy_ids, pathfinder_only,
    )

    wq: queue.Queue = queue.Queue(maxsize=2)

    def _writer():
        while True:
            item = wq.get()
            if item is None:
                break
            s, e, tmap, inv, elbo, pf_ok, samp, acc, step, divf, rhat, rchi2 = item
            h5["theta_map"][s:e]          = tmap
            h5["inv_mass"][s:e]           = inv
            h5["elbo"][s:e]               = elbo
            h5["pathfinder_ok"][s:e]      = pf_ok
            h5["accept_rate"][s:e]        = acc
            h5["step_size"][s:e]          = step
            h5["divergent_frac"][s:e]     = divf
            h5["rhat"][s:e]               = rhat
            h5["reduced_chi2_map"][s:e]   = rchi2
            if samp is not None:
                h5["theta_samples"][s:e]  = samp
            wq.task_done()

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()

    rng = jax.random.PRNGKey(seed)
    n_batches   = (N + batch_size - 1) // batch_size
    dummy_obs   = np.zeros((batch_size, n_bands), dtype=np.float32)
    dummy_err   = np.full((batch_size, n_bands), MISSING_SIGMA, dtype=np.float32)
    metric_id_fallback = 0
    metric_floored     = 0
    t0 = time.perf_counter()

    _lm_lo, _lm_hi = float(PARAM_BOUNDS["log_mass"][0]), float(PARAM_BOUNDS["log_mass"][1])
    _z_lo,  _z_hi  = float(PARAM_BOUNDS["redshift"][0]), float(PARAM_BOUNDS["redshift"][1])
    _av_lo, _av_hi = float(PARAM_BOUNDS["Av"][0]),       float(PARAM_BOUNDS["Av"][1])
    _SEEDS_Z_AV    = [(0.3, 0.3), (1.0, 1.0), (3.0, 0.5), (7.0, 0.1)]
    _bidx_j        = jnp.asarray(band_idx, dtype=jnp.int32)

    for bi in range(n_batches):
        start    = bi * batch_size
        end_true = min(start + batch_size, N)
        true     = end_true - start
        pad      = batch_size - true

        if pad > 0:
            obs_b = np.concatenate([obs_flux[start:end_true], dummy_obs[:pad]], axis=0)
            err_b = np.concatenate([flux_err[start:end_true], dummy_err[:pad]], axis=0)
        else:
            obs_b = obs_flux[start:end_true]
            err_b = flux_err[start:end_true]

        obs_jax = jnp.asarray(obs_b)
        err_jax = jnp.asarray(err_b)

        rng, k_pinit, k_pf, k_cinit, k_nuts = jax.random.split(rng, 5)

        # ---- multi-path Pathfinder: seeds at (z, Av) covering main populations ----
        path_inits = jax.random.normal(k_pinit, (batch_size, n_paths, P)) * init_jitter
        path_inits = path_inits.at[:, 0, :].set(0.0)

        _eff_err = jnp.maximum(err_jax, min_frac_err * jnp.abs(obs_jax))
        _mask_w  = (err_jax < OBS_MASK_THRESH).astype(jnp.float32) / (_eff_err ** 2)
        for _pi, (_z_s, _av_s) in enumerate(_SEEDS_Z_AV[:n_paths]):
            _x_ref = (0.5 * (lows + highs)).at[REDSHIFT_IDX].set(float(_z_s)) \
                                            .at[AV_IDX].set(float(_av_s)) \
                                            .at[LOGMASS_IDX].set(8.0)
            _pred  = emulator.predict(_x_ref[None, :])[0][_bidx_j]
            _num   = jnp.sum(_mask_w * obs_jax * _pred, axis=1)
            _den   = jnp.sum(_mask_w * _pred ** 2, axis=1)
            _s     = jnp.where(_den > 0, _num / _den, 1.0)
            _lm    = jnp.clip(8.0 + jnp.log10(jnp.clip(_s, 1e-30, 1e30)), _lm_lo, _lm_hi)
            _lm_u  = jnp.clip((_lm - _lm_lo) / (_lm_hi - _lm_lo), 1e-4, 1 - 1e-4)
            _z_u   = float(np.clip((_z_s - _z_lo) / (_z_hi - _z_lo), 1e-4, 1 - 1e-4))
            _av_u  = float(np.clip((_av_s - _av_lo) / (_av_hi - _av_lo), 1e-4, 1 - 1e-4))
            path_inits = path_inits.at[:, _pi, REDSHIFT_IDX].set(float(np.log(_z_u / (1 - _z_u))))
            path_inits = path_inits.at[:, _pi, AV_IDX].set(float(np.log(_av_u / (1 - _av_u))))
            path_inits = path_inits.at[:, _pi, LOGMASS_IDX].set(jnp.log(_lm_u / (1 - _lm_u)))

        pf_keys = jax.random.split(k_pf, batch_size * n_paths).reshape(batch_size, n_paths, -1)
        pos, inv, elbo, ok = batch_pf(obs_jax, err_jax, path_inits, pf_keys)
        pos    = np.array(pos)
        inv    = np.array(inv)
        elbo   = np.asarray(elbo)
        pf_ok  = (np.asarray(ok) & np.isfinite(pos).all(-1)
                  & np.isfinite(inv).reshape(batch_size, -1).all(-1))
        pos[~pf_ok] = 0.0
        inv[~pf_ok] = np.eye(P, dtype=np.float32)

        inv, n_id_fallback, n_floored = regularize_metric(inv, diag_only=diag_metric)
        metric_id_fallback += n_id_fallback
        metric_floored     += n_floored

        red_chi2 = reduced_chi2_at(pos, obs_b, err_b, emulator, band_idx,
                                   lows_np, highs_np, min_frac_err=min_frac_err)

        # ---- multi-chain NUTS ----
        if pathfinder_only:
            samples_out = None
            accept = np.zeros((batch_size, n_chains), dtype=np.float32)
            step   = np.full((batch_size, n_chains), eps0, dtype=np.float32)
            divf   = np.zeros((batch_size, n_chains), dtype=np.float32)
            rhat   = np.full((batch_size, P), np.nan, dtype=np.float32)
        else:
            # Geometry-aware chain jitter: scale by posterior std from Pathfinder metric
            _inv_diag = jnp.diagonal(jnp.asarray(inv), axis1=-2, axis2=-1)
            _pf_std   = jnp.sqrt(jnp.clip(_inv_diag, 1e-6, None))
            cjit      = (jax.random.normal(k_cinit, (batch_size, n_chains, P))
                         * chain_jitter * _pf_std[:, None, :])
            cjit      = cjit.at[:, 0, :].set(0.0)
            chain_inits = jnp.asarray(pos)[:, None, :] + cjit
            nuts_keys   = jax.random.split(k_nuts, batch_size * n_chains).reshape(
                batch_size, n_chains, -1)
            samples_j, accept_j, step_j, divf_j = batch_nuts(
                obs_jax, err_jax, chain_inits, jnp.asarray(inv), nuts_keys)
            samples_out = np.asarray(samples_j[:true])
            accept      = np.asarray(accept_j)
            step        = np.asarray(step_j)
            divf        = np.asarray(divf_j)
            rhat        = np.full((batch_size, P), np.nan, dtype=np.float32)
            rhat[:true] = split_rhat(samples_out, lows_np, highs_np)

        wq.put((start, end_true,
                pos[:true], inv[:true], elbo[:true], pf_ok[:true],
                samples_out, accept[:true], step[:true], divf[:true], rhat[:true],
                red_chi2[:true]))

        elapsed = time.perf_counter() - t0
        rate    = end_true / elapsed if elapsed > 0 else 0.0
        eta     = (N - end_true) / rate if rate > 0 else float("inf")
        if pathfinder_only:
            diag = (f"pf_ok={np.mean(pf_ok[:true]):.1%}  "
                    f"redchi2_med={np.nanmedian(red_chi2[:true]):.1f}")
        else:
            diag = (f"accept={np.nanmean(accept[:true]):.2f}  "
                    f"div={np.nanmean(divf[:true]):.1%}  "
                    f"rhat<1.05={np.mean(np.nanmax(rhat[:true], axis=1) < 1.05):.1%}  "
                    f"redchi2_med={np.nanmedian(red_chi2[:true]):.1f}  "
                    f"pf_ok={np.mean(pf_ok[:true]):.1%}")
        print(f"  [{end_true:>{len(str(N))}}/{N}]  batch {bi + 1}/{n_batches}  "
              f"{rate:.0f} gal/s  ETA {eta:.0f}s  {diag}", flush=True)

    wq.put(None)
    writer.join()

    total       = time.perf_counter() - t0
    pf_ok_all   = np.asarray(h5["pathfinder_ok"][:])
    rchi2_all   = np.asarray(h5["reduced_chi2_map"][:])
    finite_rchi2 = rchi2_all[np.isfinite(rchi2_all)]
    print(f"\n{'=' * 60}")
    print(f"Finished {N:,} galaxies in {total:.1f}s  ({N / total:.0f} gal/s)")
    print(f"  Pathfinder OK:    {pf_ok_all.sum():,}/{N}  ({pf_ok_all.mean():.1%})")
    if finite_rchi2.size:
        frac_bad = float(np.mean(finite_rchi2 > 10.0))
        print(f"  Reduced chi2 @ MAP: median={np.median(finite_rchi2):.2f}  frac>10={frac_bad:.1%}")
        if frac_bad > 0.1:
            print("  !  Many galaxies have a poor MAP fit -- cut on reduced_chi2_map downstream.")
    print(f"  Metric regularised (eigen-floored): {metric_floored:,}  |  "
          f"identity fallback: {metric_id_fallback:,}")
    if metric_id_fallback > 0.1 * N or metric_floored > 0.5 * N:
        print("  !  Pathfinder metrics were frequently indefinite/ill-conditioned.")
    if not pathfinder_only:
        acc  = np.asarray(h5["accept_rate"][:])
        div  = np.asarray(h5["divergent_frac"][:])
        rh   = np.asarray(h5["rhat"][:])
        worst_rhat = np.nanmax(rh, axis=1)
        print(f"  Mean accept:      {np.nanmean(acc):.3f}")
        print(f"  Galaxies w/ any divergence:  {np.mean(np.nanmax(div, axis=1) > 0):.1%}")
        print(f"  Galaxies w/ max-rhat < 1.05: {np.mean(worst_rhat < 1.05):.1%}  "
              f"(>1.1: {np.mean(worst_rhat > 1.1):.1%})")
        n_div = int((np.nanmax(div, axis=1) > 0.01).sum())
        if n_div:
            print(f"  !  {n_div:,} galaxies with >1% divergences -- "
                  f"raise --target-accept or revisit priors.")
    print(f"  Output:           {output_path}")
    h5.close()


# ---------------------------------------------------------------------------
# Main fitting loop — NSS
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
) -> None:
    if not _NSS_AVAILABLE:
        raise RuntimeError("blackjax.ns not available — cannot run NSS.")

    N = len(galaxy_ids)
    P = len(SPS_PARAM_NAMES)
    band_names = [emulator.band_names[int(i)] for i in band_idx]
    lows_np  = np.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=np.float32)
    highs_np = np.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=np.float32)

    print(f"\nNSS catalogue fit: {N} galaxies", flush=True)
    print(f"  num_live={num_live}  num_inner_steps={num_inner_steps}  "
          f"num_delete={num_delete}  termination={termination}", flush=True)
    print(f"  n_samples_out={n_samples_out}  min_frac_err={min_frac_err:.1%}", flush=True)
    print(f"  Note: XLA compilation on first galaxy takes ~10-20 min; "
          f"subsequent galaxies reuse the compiled binary.", flush=True)

    log_prior_fn = make_log_prior_fn(prior_specs)
    log_like_fn  = make_log_likelihood_physical(emulator, band_idx, min_frac_err)
    nss_init_fn, nss_step_fn = build_nss_fns(
        log_prior_fn, log_like_fn, num_inner_steps, num_delete)

    h5 = create_output_file_nss(
        output_path, N, n_samples_out, band_names, emulator_path, prior_specs,
        galaxy_ids, num_live, num_inner_steps, num_delete, termination,
    )

    # Trigger XLA compilation (or load from cache) on galaxy 0 before the timed loop.
    print("\nWarm-up: compiling or loading XLA binary for NSS ...", flush=True)
    rng    = jax.random.PRNGKey(seed)
    rng, k0 = jax.random.split(rng)
    obs0   = jnp.asarray(obs_flux[0], dtype=jnp.float32)
    err0   = jnp.asarray(flux_err[0], dtype=jnp.float32)
    live0  = jax.random.uniform(k0, (num_live, P),
                                 minval=jnp.asarray(lows_np), maxval=jnp.asarray(highs_np))
    t_warmup = time.perf_counter()
    _st    = nss_init_fn(live0, obs0, err0)
    rng, _k = jax.random.split(rng)
    _st, _ = nss_step_fn(_k, _st, obs0, err0)
    jax.block_until_ready(_st)
    t_warmup = time.perf_counter() - t_warmup

    # Heuristic: a cache hit loads in seconds; a fresh compile takes minutes.
    cache_hit = t_warmup < 60.0
    try:
        from jax._src.compilation_cache import is_cache_used
        cache_hit = is_cache_used()
    except Exception:
        pass
    source = "loaded from XLA cache" if cache_hit else f"freshly compiled"
    print(f"  Done in {t_warmup:.1f}s ({source}).", flush=True)

    rng    = jax.random.PRNGKey(seed)   # reset for reproducibility
    t_global = time.perf_counter()

    for i in range(N):
        rng, gal_key = jax.random.split(rng)
        samp_phys, logz, logz_err, ess, n_steps, n_dead, nss_t = run_nss_galaxy(
            gal_key, obs_flux[i], flux_err[i],
            nss_init_fn, nss_step_fn,
            num_live, termination, n_samples_out,
        )

        # R-hat on physical-space NS samples (treat resampled set as single split chain)
        rhat_nss = split_rhat(
            samp_phys[np.newaxis, np.newaxis, :, :], lows=None, highs=None)[0]

        h5["nss_samples"][i]  = samp_phys
        h5["nss_logZ"][i]     = logz
        h5["nss_logZ_err"][i] = logz_err
        h5["nss_ess"][i]      = ess
        h5["nss_n_steps"][i]  = n_steps
        h5["nss_n_dead"][i]   = n_dead
        h5["nss_time"][i]     = nss_t
        h5["nss_rhat"][i]     = rhat_nss

        elapsed_total = time.perf_counter() - t_global
        eta = elapsed_total / (i + 1) * (N - i - 1)
        print(f"  [{i + 1:>{len(str(N))}}/{N}]  "
              f"logZ={logz:.2f}±{logz_err:.2f}  ESS={ess:.0f}  "
              f"n_dead={n_dead}  rhat_max={np.nanmax(rhat_nss):.3f}  "
              f"t={nss_t:.1f}s  ETA={eta:.0f}s", flush=True)

    total   = time.perf_counter() - t_global
    t_arr   = np.array(h5["nss_time"][:])
    logz_arr = np.array(h5["nss_logZ"][:])
    ess_arr  = np.array(h5["nss_ess"][:])
    rh       = np.array(h5["nss_rhat"][:])
    worst    = np.nanmax(rh, axis=1)

    print(f"\n{'=' * 60}")
    print(f"Done: {N} galaxies in {total:.1f}s")
    print(f"  time/gal: median={np.nanmedian(t_arr):.1f}s  total={np.nansum(t_arr):.1f}s")
    print(f"  logZ:     mean={np.nanmean(logz_arr):.2f}  std={np.nanstd(logz_arr):.2f}")
    print(f"  ESS:      median={np.nanmedian(ess_arr):.0f}")
    print(f"  R-hat<1.05: {np.mean(worst < 1.05):.1%}  (>1.1: {np.mean(worst > 1.1):.1%})")
    print(f"  Output: {output_path}")
    h5.close()


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def check_gpu_linalg() -> None:
    """Probe the GPU linear-algebra path (Pathfinder + NUTS dense metric)."""
    backend = jax.default_backend()
    try:
        a       = jnp.eye(8, dtype=jnp.float32) + 1e-3
        batched = jnp.broadcast_to(a, (16, 8, 8))
        jax.vmap(jnp.linalg.cholesky)(batched).block_until_ready()
        jax.vmap(lambda m: jnp.linalg.qr(m)[0])(batched).block_until_ready()
    except Exception as e:
        raise RuntimeError(
            "GPU linear-algebra self-test failed -- environment issue, not your data.\n"
            f"  backend={backend}  error={type(e).__name__}: {e}\n"
            "Fix: set XLA_PYTHON_CLIENT_MEM_FRACTION=0.6, or verify jaxlib/CUDA versions."
        ) from e
    print(f"GPU linalg self-test OK (backend={backend}).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a galaxy catalogue with GPU-batched Pathfinder+NUTS or Nested Slice Sampling (NSS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("catalogue", nargs="?", help="Input catalogue (FITS/CSV/HDF5)")
    parser.add_argument("config",    nargs="?", help="Band config JSON")
    parser.add_argument("--emulator", default=str(DEFAULT_EMULATOR))
    parser.add_argument("--output",   default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sampler",  choices=["nuts", "nss"], default="nuts",
                        help="Inference method.  'nuts': GPU-batched Pathfinder+NUTS (fast, "
                             "batched).  'nss': Nested Slice Sampling (sequential, gradient-free, "
                             "provides logZ, better for multimodal posteriors).")
    parser.add_argument("--n-galaxies", type=int, default=None,
                        help="Fit only N galaxies (useful for testing or splitting across workers).")
    parser.add_argument("--row-start",  type=int, default=0,
                        help="First catalogue row to process (0-based). Use with --n-galaxies "
                             "to divide a catalogue across parallel workers.")

    # ---- NUTS arguments ----
    nuts_grp = parser.add_argument_group("NUTS options (--sampler nuts)")
    nuts_grp.add_argument("--batch-size",   type=int,   default=2000,
                          help="Galaxies per GPU call. Decrease on OOM.")
    nuts_grp.add_argument("--n-paths",      type=int,   default=4,
                          help="Pathfinder paths per galaxy (best ELBO selected).")
    nuts_grp.add_argument("--n-chains",     type=int,   default=2,
                          help="NUTS chains per galaxy.")
    nuts_grp.add_argument("--n-warmup",     type=int,   default=300,
                          help="NUTS dual-averaging warmup steps (discarded).")
    nuts_grp.add_argument("--n-samples",    type=int,   default=500,
                          help="NUTS samples per chain.")
    nuts_grp.add_argument("--n-pf-samples", type=int,   default=50,
                          help="Pathfinder ELBO Monte-Carlo samples per path point.")
    nuts_grp.add_argument("--vmap-pf-paths", action="store_true",
                          help="Run Pathfinder paths in parallel (faster, n_paths x peak memory).")
    nuts_grp.add_argument("--dense-metric",  action="store_true",
                          help="Use full PSD-regularised dense Pathfinder metric (captures "
                               "correlations). Default is the more stable diagonal metric.")
    nuts_grp.add_argument("--target-accept", type=float, default=0.8)
    nuts_grp.add_argument("--step-size",     type=float, default=0.0,
                          help="Initial NUTS step size. 0 = auto (0.5/d^0.25).")
    nuts_grp.add_argument("--init-jitter",   type=float, default=2.0,
                          help="Std of Pathfinder path inits (unconstrained space).")
    nuts_grp.add_argument("--chain-jitter",  type=float, default=0.5,
                          help="Std of NUTS chain inits around the Pathfinder mean.")
    nuts_grp.add_argument("--pathfinder-only", action="store_true",
                          help="Skip NUTS; store MAP + inv_mass only.")
    nuts_grp.add_argument("--check-gradients", action="store_true",
                          help="Run autodiff vs FD gradient self-test and exit.")

    # ---- NSS arguments ----
    nss_grp = parser.add_argument_group("NSS options (--sampler nss)")
    nss_grp.add_argument("--num-live",        type=int,   default=500,
                         help="Number of NS live particles (>= ~50*P for 12-param problem).")
    nss_grp.add_argument("--num-inner-steps", type=int,   default=24,
                         help="HRSS steps per replacement (recommend 2*P=24).")
    nss_grp.add_argument("--num-delete",      type=int,   default=50,
                         help="Dead particles per NS iteration (vmapped in parallel).")
    nss_grp.add_argument("--termination",     type=float, default=-3.0,
                         help="Stop when logZ_live - logZ < termination.")
    nss_grp.add_argument("--n-samples-out",   type=int,   default=500,
                         help="NS posterior samples to importance-resample and store.")

    # ---- common ----
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--xla-cache-dir",
        default=str(DEFAULT_XLA_CACHE),
        help="Directory for the persistent XLA compilation cache.  On a cache hit the "
             "~10-20 min NSS first-compilation is reduced to seconds.  The cache is "
             "automatically invalidated when JAX version, GPU, or compiled function shapes "
             "change.  Set to '' to disable.",
    )
    parser.add_argument("--print-config-template", action="store_true",
                        help="Print a JSON config template (with default priors) and exit.")
    args = parser.parse_args()

    emulator_path = Path(args.emulator)

    if args.print_config_template:
        print_config_template(emulator_path)
        return

    if not args.catalogue or not args.config:
        parser.error("catalogue and config are required unless --print-config-template is used.")
    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator not found: {emulator_path}")

    # Set up XLA cache before the first JIT call (check_gpu_linalg).
    if args.xla_cache_dir:
        setup_xla_cache(Path(args.xla_cache_dir))

    check_gpu_linalg()

    config = load_config(Path(args.config))
    obs_flux, flux_err, galaxy_ids = load_catalogue(Path(args.catalogue), config)
    emulator, band_idx = load_emulator_and_band_indices(
        emulator_path, list(config["bands"].keys()))

    if args.row_start > 0:
        start = min(args.row_start, len(galaxy_ids))
        obs_flux   = obs_flux[start:]
        flux_err   = flux_err[start:]
        galaxy_ids = galaxy_ids[start:]

    if args.n_galaxies is not None:
        n = min(args.n_galaxies, len(galaxy_ids))
        obs_flux   = obs_flux[:n]
        flux_err   = flux_err[:n]
        galaxy_ids = galaxy_ids[:n]

    if args.row_start > 0 or args.n_galaxies is not None:
        print(f"Subset: rows {args.row_start}–{args.row_start + len(galaxy_ids) - 1} "
              f"({len(galaxy_ids)} galaxies).")

    min_frac_err = float(config.get("min_frac_err", 0.05))

    if args.sampler == "nss":
        if not _NSS_AVAILABLE:
            raise RuntimeError(
                "blackjax.ns not found.  The installed blackjax does not include the "
                "nested-sampling utilities.  Install the fork:\n"
                "  pip install git+https://github.com/<fork>/blackjax"
            )
        run_catalogue_nss(
            obs_flux, flux_err, galaxy_ids, emulator, band_idx,
            config["resolved_priors"],
            output_path=Path(args.output),
            emulator_path=emulator_path,
            num_live=args.num_live,
            num_inner_steps=args.num_inner_steps,
            num_delete=args.num_delete,
            termination=args.termination,
            n_samples_out=args.n_samples_out,
            min_frac_err=min_frac_err,
            seed=args.seed,
        )
        return

    # NUTS path
    if args.n_chains < 1 or args.n_paths < 1:
        parser.error("--n-chains and --n-paths must be >= 1.")

    log_prior_fn = make_log_prior_fn(config["resolved_priors"])
    eps0 = args.step_size if args.step_size > 0 else 0.5 / (len(SPS_PARAM_NAMES) ** 0.25)

    if args.check_gradients:
        log_post_fn = make_log_posterior_fn(
            emulator, band_idx, log_prior_fn, min_frac_err)
        check_gradients(log_post_fn, obs_flux, flux_err, seed=args.seed)
        return

    run_catalogue(
        obs_flux, flux_err, galaxy_ids, emulator, band_idx,
        config["resolved_priors"],
        output_path=Path(args.output),
        emulator_path=emulator_path,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        n_chains=args.n_chains,
        n_paths=args.n_paths,
        n_pf_samples=args.n_pf_samples,
        eps0=eps0,
        target_accept=args.target_accept,
        init_jitter=args.init_jitter,
        chain_jitter=args.chain_jitter,
        seed=args.seed,
        pathfinder_only=args.pathfinder_only,
        vmap_pf_paths=args.vmap_pf_paths,
        diag_metric=not args.dense_metric,
        min_frac_err=min_frac_err,
    )


if __name__ == "__main__":
    main()
