#!/usr/bin/env python3
r"""Production SED fitting for large galaxy catalogues (revised).

Fits every galaxy in a catalogue using GPU-batched Pathfinder + NUTS.

Key differences from the previous version
------------------------------------------
1. Mass matrix is the *full* L-BFGS inverse-Hessian from Pathfinder
   (lbfgs_inverse_hessian_formula_1(alpha, beta, gamma)), passed to NUTS as
   a dense matrix.  The old code used `state.alpha` alone, which is only the
   diagonal init term and drops all parameter correlations.
2. Multi-path Pathfinder per galaxy (over-dispersed inits); the best-ELBO
   path is selected.  Mitigates single-path mode collapse.
3. Multi-chain NUTS with per-galaxy dual-averaging step-size adaptation
   (mass matrix held fixed from Pathfinder).  Replaces the single global
   step size, so acceptance is well-controlled per galaxy.
4. Warmup samples are discarded; sampling step size is frozen post-warmup.
5. Divergences and split-R-hat are recorded per galaxy.
6. Negative observed fluxes are *kept* (valid Gaussian measurements).
   Only finite flux with a finite, positive error is required.  The old
   `flux > 0` test discarded downward noise fluctuations -> Eddington bias.
7. Configurable per-parameter priors (uniform, loguniform, normal,
   studentt, halfnormal, exponential, lognormal).  The default for the
   continuity-SFH logsfr_ratios is Student-t(df=2, scale=0.3).

Transform / prior contract
--------------------------
The sampling domain for every parameter is fixed to the emulator's training
bounds (PARAM_BOUNDS) via a sigmoid transform.  This guarantees emulator
inputs are always in-domain.  The chosen prior is a *density shape* applied
over that domain; priors with support beyond the bounds are implicitly
truncated to them (truncation normalisation is constant in theta and
dropped, so within-galaxy inference is exact; absolute log-density / ELBO
values are unnormalised constants apart).

Band config JSON
----------------
    {
        "id_col":     "ID",
        "flux_unit":  "uJy",
        "bands": {
            "JWST/NIRCam.F090W": {"flux": "f_F090W", "err": "e_F090W"},
            ...
        },
        "priors": {                         # optional; merged over defaults
            "Av":             {"dist": "exponential", "scale": 0.5},
            "redshift":       {"dist": "uniform"},
            "logsfr_ratio_0": {"dist": "studentt", "df": 2, "loc": 0.0, "scale": 0.3}
        }
    }

    flux_unit: "nJy" | "uJy" | "mJy" | "Jy" | "ABmag"

    Supported prior "dist" values (params are read per-key):
        uniform                                  (flat over the domain)
        loguniform                               (~1/x; requires bound_lo > 0)
        normal       loc, scale                  (truncated to the domain)
        studentt     df, loc, scale              (truncated to the domain)
        halfnormal   scale                       (intended for bound_lo ~ 0)
        exponential  scale  (= mean)             (intended for bound_lo ~ 0)
        lognormal    loc, scale                  (requires bound_lo > 0)

Usage
-----
    python scripts/fit_catalogue.py catalogue.fits bands.json
    python scripts/fit_catalogue.py catalogue.fits bands.json \\
        --emulator scripts/outputs/emulators/parrot_emulator.eqx \\
        --output outputs/catalogue_fit/results.hdf5 \\
        --n-samples 500 --n-warmup 300 --n-chains 2 --n-paths 4 \\
        --batch-size 2000

    # Fast MAP-only run (multi-path Pathfinder, no NUTS):
    python scripts/fit_catalogue.py catalogue.fits bands.json --pathfinder-only

    # Print a config template (incl. default priors) for the loaded emulator:
    python scripts/fit_catalogue.py --print-config-template \\
        --emulator scripts/outputs/emulators/parrot_emulator.eqx

Output HDF5 layout
------------------
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
    /rhat               (N, P)          split-R-hat per parameter
    /reduced_chi2_map    (N,)           reduced chi-squared at the Pathfinder MAP
                                        (fit-quality flag; cut on this downstream)
    attrs: param_names, band_names, param_bounds_lo, param_bounds_hi,
           prior_spec (json), emulator_path, run_timestamp, n_galaxies,
           n_samples, n_warmup, n_chains, n_paths, batch_size, target_accept

    To recover physical parameters from /theta_samples:
        lo  = f["param_bounds_lo"][:]
        hi  = f["param_bounds_hi"][:]
        sps = lo + (hi - lo) / (1 + exp(-theta_samples))   # sigmoid
"""

from __future__ import annotations

import os

# cuSolver -- used by Pathfinder's ELBO/covariance step and by the NUTS dense
# metric -- allocates its handle and workspace OUTSIDE XLA's preallocated memory
# pool.  With JAX's default preallocation (~75% of VRAM) there may be no room
# left for cuSolver, which then fails with "INTERNAL: cuSolver internal error".
# Allocating on demand leaves cuSolver the headroom it needs.  These must be set
# before `import jax`; override them in the environment (e.g. your SLURM script)
# if you prefer a fixed memory fraction (XLA_PYTHON_CLIENT_MEM_FRACTION).
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
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install blackjax") from e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EMULATOR = Path("scripts/outputs/emulators/parrot_emulator.eqx")
DEFAULT_OUTPUT = Path("outputs/catalogue_fit/results.hdf5")

# Sigma assigned to missing/masked bands.  Bands with err >= OBS_MASK_THRESH
# are excluded from the likelihood entirely (contribute exactly 0).
MISSING_SIGMA: float = 1e10  # nJy
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
    "redshift": (0.01, 14.0),
    "log_mass": (4.0, 12.0),
    "slope": (-0.3, 1.1),
    "fesc_lya": (0.0, 1.0),
    "dust_bump_amplitude": (0.0, 5.0),
    "log10metallicity": (-4.0, -1.39),
    "Av": (0.001, 5.0),
    "logsfr_ratio_0": (-10.0, 10.0),
    "logsfr_ratio_1": (-10.0, 10.0),
    "logsfr_ratio_2": (-10.0, 10.0),
    "logsfr_ratio_3": (-10.0, 10.0),
    "logsfr_ratio_4": (-10.0, 10.0),
}

# Index of the amplitude (mass-normalisation) parameter — used for data-driven
# initialisation, since flux scales as 10**log_mass.
REDSHIFT_IDX: int = SPS_PARAM_NAMES.index("redshift")
LOGMASS_IDX: int = SPS_PARAM_NAMES.index("log_mass")
AV_IDX: int = SPS_PARAM_NAMES.index("Av")

# Sensible defaults.  The continuity-SFH logsfr_ratios get the standard
# Student-t(df=2, scale=0.3) prior; everything else is uniform-over-domain.
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

FLUX_UNIT_TO_NJY: dict[str, float] = {
    "nJy": 1.0,
    "uJy": 1e3,
    "ujy": 1e3,
    "µJy": 1e3,
    "mJy": 1e6,
    "Jy": 1e9,
}

_KNOWN_DISTS = {
    "uniform",
    "loguniform",
    "normal",
    "studentt",
    "halfnormal",
    "exponential",
    "lognormal",
}

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


def validate_prior_spec(name: str, spec: dict, bounds: tuple[float, float]) -> None:
    """Validate a single parameter's prior spec; raise on error."""
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
    """Merge user priors over defaults, validate, return one spec per param."""
    priors: dict[str, dict] = {p: dict(DEFAULT_PRIORS.get(p, {"dist": "uniform"})) for p in SPS_PARAM_NAMES}
    user = config.get("priors", {}) or {}
    for k, v in user.items():
        if k not in SPS_PARAM_NAMES:
            raise ValueError(f"priors: unknown parameter {k!r}. Valid: {SPS_PARAM_NAMES}")
        priors[k] = dict(v)
    for p in SPS_PARAM_NAMES:
        validate_prior_spec(p, priors[p], PARAM_BOUNDS[p])
    return priors


def _build_prior_logprob(spec: dict, lo: float, hi: float):
    """Return a closure x_scalar -> log prior density (constants may be dropped)."""
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
        c = (
            math.lgamma(0.5 * (df + 1.0))
            - math.lgamma(0.5 * df)
            - 0.5 * math.log(df * math.pi)
            - math.log(scale)
        )
        return lambda x: c - 0.5 * (df + 1.0) * jnp.log1p(((x - loc) / scale) ** 2 / df)

    if dist == "halfnormal":
        scale = float(spec["scale"])
        return lambda x: -0.5 * (x / scale) ** 2  # const log(2)-log(scale)-... dropped

    if dist == "exponential":
        scale = float(spec["scale"])  # mean
        return lambda x: -x / scale

    if dist == "lognormal":
        loc = float(spec.get("loc", 0.0))
        scale = float(spec["scale"])
        return lambda x: -jnp.log(x) - 0.5 * ((jnp.log(x) - loc) / scale) ** 2

    raise ValueError(f"unhandled dist {dist!r}")  # pragma: no cover


def make_log_prior_fn(prior_specs: dict[str, dict]):
    """Return log_prior(x_vec) summing per-parameter prior densities."""
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
    """Load, validate, and resolve priors for the band configuration JSON."""
    with open(config_path) as f:
        cfg = json.load(f)

    if "bands" not in cfg:
        raise ValueError("Config must contain a 'bands' key.")

    cfg.setdefault("id_col", None)
    cfg.setdefault("flux_unit", "nJy")

    unit = cfg["flux_unit"]
    if unit not in FLUX_UNIT_TO_NJY and unit != "ABmag":
        raise ValueError(
            f"Unknown flux_unit {unit!r}. Use one of: {list(FLUX_UNIT_TO_NJY)} or 'ABmag'."
        )

    cfg.setdefault("min_frac_err", 0.05)
    cfg["resolved_priors"] = resolve_priors(cfg)
    non_uniform = {p: s for p, s in cfg["resolved_priors"].items() if s.get("dist") != "uniform"}
    print(f"Config: {len(cfg['bands'])} bands, flux_unit={unit!r}  "
          f"min_frac_err={cfg['min_frac_err']:.1%}")
    if non_uniform:
        print(f"  Non-uniform priors: "
              + ", ".join(f"{p}={s['dist']}" for p, s in non_uniform.items()))
    return cfg


def print_config_template(emulator_path: Path) -> None:
    """Print a JSON config template (with default priors) for the loaded emulator."""
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
    """Load galaxy catalogue and extract flux / flux-error arrays (in nJy).

    A band is masked (obs=0, err=MISSING_SIGMA) only when its flux or error is
    non-finite, or its error is non-positive.  Negative fluxes are KEPT: they
    are valid Gaussian measurements, and discarding them biases faint sources.
    """
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
        err_col = config["bands"][band]["err"]

        f_raw = np.asarray(t[flux_col], dtype=np.float64)
        e_raw = np.asarray(t[err_col], dtype=np.float64)

        if use_abmag:
            # ABmag cannot represent negative flux; only finite mags are valid.
            f_nJy = 10.0 ** ((8.9 - f_raw) / 2.5) * 1e9
            e_nJy = f_nJy * np.abs(e_raw) * np.log(10.0) / 2.5
        else:
            f_nJy = f_raw * factor
            e_nJy = e_raw * factor

        # Keep negative fluxes; require only finite values and a positive error.
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
    """Load a ParrotEmulator and resolve config band names to emulator indices."""
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
# Log-posterior
# ---------------------------------------------------------------------------


def make_log_posterior_fn(emulator, band_idx: np.ndarray, log_prior_fn,
                          min_frac_err: float = 0.0):
    """Return log_posterior(theta, obs_flux, flux_err) -> scalar.

    theta is unconstrained; physical params = lo + (hi-lo)*sigmoid(theta),
    which keeps every emulator input inside the training domain.  Masked bands
    (err >= OBS_MASK_THRESH) contribute exactly 0 to the Gaussian likelihood.

    min_frac_err: minimum error as a fraction of |obs_flux|.  Accounts for
    emulator approximation error, calibration uncertainty, and physical model
    incompleteness — all irreducible at sub-percent level.  Without this, pure
    photon-noise errors on bright galaxies (S/N > 1000) create a posterior so
    sharp that no MCMC sampler can make accepted proposals.
    """
    lows = jnp.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    highs = jnp.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    log_range = jnp.log(highs - lows)
    _bidx = jnp.array(band_idx, dtype=jnp.int32)

    def log_posterior(theta, obs_flux, flux_err):
        x = lows + (highs - lows) * jax.nn.sigmoid(theta)
        pred = emulator.predict(x[None, :])[0][_bidx]

        mask = (flux_err < OBS_MASK_THRESH).astype(theta.dtype)
        eff_err = jnp.maximum(flux_err, min_frac_err * jnp.abs(obs_flux)) if min_frac_err > 0 else flux_err
        inv_var = 1.0 / (eff_err ** 2)
        resid = obs_flux - pred
        chi2 = jnp.sum(mask * resid * resid * inv_var)
        log_norm = jnp.sum(mask * (-0.5 * LOG2PI - jnp.log(eff_err)))
        log_like = log_norm - 0.5 * chi2

        log_jac = jnp.sum(
            log_range + jax.nn.log_sigmoid(theta) + jax.nn.log_sigmoid(-theta)
        )
        return log_like + log_jac + log_prior_fn(x)

    return log_posterior


# ---------------------------------------------------------------------------
# Compiled batch functions
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
    """JIT-compile vmapped multi-path Pathfinder and multi-chain NUTS.

    Returns:
        batch_pf:   (obs_B, err_B, theta_init_BKP, keys_BK)
                    -> (pos_BP, inv_mass_BPP, elbo_B, ok_B)   [best path / galaxy]
        batch_nuts: (obs_B, err_B, theta_init_BCP, inv_mass_BPP, keys_BC)
                    -> (samples_BCSP, accept_BC, step_BC, divfrac_BC)
    """
    eye = jnp.eye(n_params, dtype=jnp.float32)

    # ---- Pathfinder (one path) ----
    def _pf_one(obs_flux, flux_err, theta_init, rng_key):
        def lp(theta):
            return log_posterior_fn(theta, obs_flux, flux_err)

        state, _ = pf_mod.approximate(rng_key, lp, theta_init, num_samples=n_pf_samples)
        inv = lbfgs_inverse_hessian_formula_1(state.alpha, state.beta, state.gamma)
        inv = 0.5 * (inv + inv.T) + 1e-8 * eye
        ok = jnp.isfinite(state.position).all() & jnp.isfinite(inv).all() & jnp.isfinite(state.elbo)
        elbo = jnp.where(ok, state.elbo, -jnp.inf)
        return state.position, inv, elbo, ok

    # ---- Pathfinder (best of K paths for one galaxy) ----
    # Paths can run in parallel (vmap, fast, K x peak memory) or sequentially
    # (lax.map, K x slower, memory of a single path).  Pathfinder's ELBO step
    # evaluates the emulator n_pf_samples times per path, so the path axis is a
    # large memory multiplier; sequential is the safe default on tight GPUs.
    def _pf_galaxy(obs_flux, flux_err, theta_inits, rng_keys):
        if vmap_paths:
            pos, inv, elbo, ok = jax.vmap(_pf_one, in_axes=(None, None, 0, 0))(
                obs_flux, flux_err, theta_inits, rng_keys
            )
        else:
            pos, inv, elbo, ok = jax.lax.map(
                lambda tk: _pf_one(obs_flux, flux_err, tk[0], tk[1]),
                (theta_inits, rng_keys),
            )
        j = jnp.argmax(elbo)
        return pos[j], inv[j], elbo[j], ok[j]

    batch_pf = jax.jit(jax.vmap(_pf_galaxy, in_axes=(0, 0, 0, 0)))

    # ---- Dual-averaging step-size adaptation (Hoffman & Gelman 2014) ----
    log_eps0 = math.log(eps0)
    mu = math.log(10.0 * eps0)

    def da_init():
        # (m, log_step, log_step_bar, h_bar)
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

    # ---- NUTS (one chain) ----
    def _nuts_one(obs_flux, flux_err, theta_init, inv_mass, rng_key):
        def lp(theta):
            return log_posterior_fn(theta, obs_flux, flux_err)

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
                warm_step, (state, da_init()), jax.random.split(warm_key, n_warmup)
            )
            eps_final = jnp.exp(da[2])
        else:
            eps_final = jnp.array(eps0)

        eps_final = jnp.where(jnp.isfinite(eps_final) & (eps_final > 0), eps_final, eps0)

        kern = blackjax.nuts(lp, step_size=eps_final, inverse_mass_matrix=inv_mass)

        def sample_step(st, k):
            st, info = kern.step(k, st)
            return st, (st.position, info.acceptance_rate, info.is_divergent)

        _, (samples, ar, div) = jax.lax.scan(
            sample_step, state, jax.random.split(sample_key, n_samples)
        )
        return samples, jnp.mean(ar), eps_final, jnp.mean(div.astype(jnp.float32))

    # over chains (theta_init, key vary; obs/err/inv shared), then over galaxies
    nuts_chains = jax.vmap(_nuts_one, in_axes=(None, None, 0, None, 0))
    batch_nuts = jax.jit(jax.vmap(nuts_chains, in_axes=(0, 0, 0, 0, 0)))

    return batch_pf, batch_nuts


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def split_rhat(
    samples: np.ndarray,
    lows: np.ndarray | None = None,
    highs: np.ndarray | None = None,
) -> np.ndarray:
    """Split-R-hat per parameter, computed in physical space.

    Args:
        samples: (B, C, S, P) NUTS samples in unconstrained (logit) space.
        lows:    (P,) physical lower bounds. If provided with highs, samples are
                 transformed to physical space before computing R-hat.  R-hat in
                 unconstrained space is inflated near sigmoid boundaries (e.g. a
                 low-z galaxy has tiny Δz but large Δθ), making the diagnostic
                 misleading.
        highs:   (P,) physical upper bounds.

    Returns:
        (B, P) split-R-hat.  Valid for C>=1 (single chain -> split into halves).
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
    chain_mean = x.mean(axis=2)                       # (B, m, P)
    chain_var = x.var(axis=2, ddof=1)                 # (B, m, P)
    grand_mean = chain_mean.mean(axis=1, keepdims=True)
    b_var = n * ((chain_mean - grand_mean) ** 2).sum(axis=1) / (m - 1)  # (B, P)
    w = chain_var.mean(axis=1)                        # (B, P)
    var_plus = (n - 1) / n * w + b_var / n
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.sqrt(var_plus / np.where(w > 0, w, np.nan))
    return rhat.astype(np.float32)


def regularize_metric(
    inv: np.ndarray, cond_cap: float = 1e6, diag_only: bool = False
) -> tuple[np.ndarray, int, int]:
    """Project per-galaxy inverse-mass matrices to PSD with bounded conditioning.

    The L-BFGS inverse-Hessian from Pathfinder is NOT guaranteed positive
    definite; an indefinite inverse_mass_matrix gives NUTS non-physical
    kinetic energy (low acceptance + divergences regardless of step size).
    Here we symmetrise, eigen-floor at lambda_max / cond_cap, and fall back to
    identity for matrices with no positive eigenvalue.

    Args:
        inv:      (B, P, P) candidate inverse mass matrices (host array).
        cond_cap: maximum allowed condition number after flooring.
        diag_only: if True, keep only the (PSD) diagonal.

    Returns:
        (regularized (B, P, P) float32, n_identity_fallback, n_floored)
    """
    B, P, _ = inv.shape
    inv = 0.5 * (inv + np.transpose(inv, (0, 2, 1)))
    w, V = np.linalg.eigh(inv)                       # ascending eigenvalues, (B, P)
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
        idx = np.arange(P)
        out[:, idx, idx] = d
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
    """Per-galaxy unconstrained theta for log_mass, from amplitude matching.

    Flux scales as 10**log_mass at fixed SED shape, so the optimal mass given a
    reference shape is a weighted least-squares amplitude.  Starting Pathfinder
    here (instead of the domain midpoint) drops each galaxy into the correct
    basin rather than 4-5 dex below it on a near-flat misfit plateau.

    Args:
        obs, err:  (B, n_bands) fluxes and errors in nJy.
        emulator:  Loaded ParrotEmulator.
        band_idx:  Band indices into the emulator output.
        lows/highs: (P,) physical bounds.
        ref:       Reference log_mass at which the shape is evaluated.

    Returns:
        (B,) unconstrained theta for the log_mass dimension.
    """
    lo, hi = float(lows[LOGMASS_IDX]), float(highs[LOGMASS_IDX])
    x_ref = (0.5 * (lows + highs)).at[LOGMASS_IDX].set(ref)        # midpoint shape @ ref mass
    pred_ref = emulator.predict(x_ref[None, :])[0][jnp.asarray(band_idx, dtype=jnp.int32)]
    w = (err < OBS_MASK_THRESH).astype(obs.dtype) / (err ** 2)     # (B, n_bands)
    num = jnp.sum(w * obs * pred_ref, axis=1)
    den = jnp.sum(w * pred_ref ** 2, axis=1)
    s = jnp.where(den > 0, num / den, 1.0)                         # amplitude scale per galaxy
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
    """Reduced chi-squared at given (unconstrained) theta, per galaxy.

    A fit-quality flag with teeth: pf_ok only checks finiteness, so a 3e8
    chi-squared "MAP" passes it.  This is computed on the host (no autodiff)
    over the observed bands only.  Uses the same effective errors as the
    likelihood (with min_frac_err floor applied) so the reported chi2 is
    consistent with what the sampler sees.

    Args:
        theta:        (B, P) unconstrained positions (e.g. Pathfinder MAPs).
        obs, err:     (B, n_bands) fluxes and errors.
        emulator:     Loaded ParrotEmulator.
        band_idx:     Band indices into the emulator output.
        lows/highs:   (P,) physical bounds.
        min_frac_err: fractional error floor (same value used in log_posterior).

    Returns:
        (B,) reduced chi-squared (chi2 / dof, dof = n_observed_bands - P, floored at 1).
    """
    lows_j = jnp.asarray(lows)
    highs_j = jnp.asarray(highs)
    x = lows_j + (highs_j - lows_j) * jax.nn.sigmoid(jnp.asarray(theta))    # (B, P)
    pred = np.asarray(emulator.predict(x))[:, np.asarray(band_idx)]         # (B, n_bands)
    mask = np.asarray(err) < OBS_MASK_THRESH
    obs_np, err_np = np.asarray(obs), np.asarray(err)
    eff_err = np.maximum(err_np, min_frac_err * np.abs(obs_np)) if min_frac_err > 0 else err_np
    resid = (obs_np - pred) / eff_err
    chi2 = np.sum(mask * resid * resid, axis=1)
    dof = np.maximum(mask.sum(axis=1) - len(SPS_PARAM_NAMES), 1)
    return (chi2 / dof).astype(np.float32)


def check_gradients(log_post_fn, obs_flux: np.ndarray, flux_err: np.ndarray,
                    n_check: int = 8, seed: int = 0) -> None:
    """Robust autodiff-vs-finite-difference gradient check.

    Naive per-component central differences in float32 are unreliable when
    |log_posterior| is large (the full-range midpoint is a wild misfit to a
    real galaxy, giving |log p| ~ 1e5-1e6, so f(x+e)-f(x-e) is dominated by
    float32 round-off).  This version instead:
      * descends to near each galaxy's mode, then offsets by a fixed amount,
        so |log p| is moderate and the gradient is non-zero (real signal);
      * uses a *directional* derivative (one well-conditioned number);
      * subtracts a float32 cancellation-noise floor before judging error,
        so round-off is not mistaken for a wrong gradient.
    """
    P = len(SPS_PARAM_NAMES)
    val_fn = jax.jit(lambda th, o, e: log_post_fn(th, o, e))
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
        th = th_map + 0.5 * jnp.asarray(offset, dtype=jnp.float32)  # off-mode: real gradient

        g = np.asarray(grad_fn(th, o, e))
        v = np.asarray(jax.random.normal(kv, (P,)))
        v /= np.linalg.norm(v)
        dd_ad = float(g @ v)
        f0 = float(val_fn(th, o, e))

        best_rel, best_fd, best_noise = float("inf"), float("nan"), float("nan")
        for eps in (3e-2, 1e-2, 3e-3):
            fp = float(val_fn(jnp.asarray(th + eps * v), o, e))
            fm = float(val_fn(jnp.asarray(th - eps * v), o, e))
            dd_fd = (fp - fm) / (2 * eps)
            noise = (abs(f0) + abs(fp) + abs(fm)) * 1.2e-7 / (2 * eps)  # float32 round-off 1-sigma
            scale = max(abs(dd_ad), abs(dd_fd), 1e-12)
            rel = max(0.0, abs(dd_ad - dd_fd) - 5 * noise) / scale  # net of round-off
            if rel < best_rel:
                best_rel, best_fd, best_noise = rel, dd_fd, noise

        finite = bool(np.isfinite(g).all() and np.isfinite(best_fd))
        suspect = (not finite) or best_rel > 0.1
        worst = max(worst, best_rel if finite else float("inf"))
        flag = "  <-- SUSPECT" if suspect else ""
        print(f"  galaxy {i}: rel err = {best_rel:.2e}  "
              f"(ad={dd_ad:.3g} fd={best_fd:.3g} roundoff~{best_noise:.1g})  "
              f"finite={finite}{flag}")
    print(f"  worst rel err over {n} galaxies: {worst:.2e}")
    if not np.isfinite(worst) or worst > 0.1:
        print("  ! Gradients look genuinely inconsistent -- investigate the emulator.")
    else:
        print("  Emulator gradients are consistent. NUTS failures are sampler-side, not the emulator.")


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------


def _largest_divisor_leq(n: int, cap: int) -> int:
    cap = max(1, min(int(cap), n))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def _chunk_rows(batch_size: int, per_row_bytes: int, n_total: int, target_bytes: int = 4 << 20) -> int:
    rows = max(1, target_bytes // max(1, per_row_bytes))
    rows = _largest_divisor_leq(batch_size, rows)
    return max(1, min(rows, n_total))


def create_output_file(
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
    """Pre-allocate the output HDF5 file with batch-aligned gzip chunks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(output_path, "w")

    try:
        f.create_dataset("galaxy_id", data=np.array(galaxy_ids, dtype=np.int64))
    except (ValueError, TypeError):
        dt = h5py.string_dtype()
        ids = np.array([str(g) for g in galaxy_ids], dtype=object)
        f.create_dataset("galaxy_id", data=ids, dtype=dt)

    N, P, S, C = n_galaxies, n_params, n_samples, n_chains
    ckw = dict(compression="gzip", compression_opts=4)
    bs = min(batch_size, N)

    r_map = _chunk_rows(bs, P * 4, N)
    f.create_dataset("theta_map", shape=(N, P), dtype=np.float32, chunks=(r_map, P), **ckw)
    r_inv = _chunk_rows(bs, P * P * 4, N)
    f.create_dataset("inv_mass", shape=(N, P, P), dtype=np.float32, chunks=(r_inv, P, P), **ckw)
    f.create_dataset("elbo", shape=(N,), dtype=np.float32)
    f.create_dataset("pathfinder_ok", shape=(N,), dtype=bool)
    f.create_dataset("accept_rate", shape=(N, C), dtype=np.float32)
    f.create_dataset("step_size", shape=(N, C), dtype=np.float32)
    f.create_dataset("divergent_frac", shape=(N, C), dtype=np.float32)
    f.create_dataset("rhat", shape=(N, P), dtype=np.float32)
    f.create_dataset("reduced_chi2_map", shape=(N,), dtype=np.float32)

    if not pathfinder_only:
        r_s = _chunk_rows(bs, C * S * P * 4, N)
        f.create_dataset(
            "theta_samples", shape=(N, C, S, P), dtype=np.float32, chunks=(r_s, C, S, P), **ckw
        )

    f.attrs["param_names"] = [s.encode() for s in SPS_PARAM_NAMES]
    f.attrs["band_names"] = [s.encode() for s in band_names]
    f.attrs["param_bounds_lo"] = np.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES])
    f.attrs["param_bounds_hi"] = np.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES])
    f.attrs["prior_spec"] = json.dumps(prior_specs)
    f.attrs["emulator_path"] = str(emulator_path)
    f.attrs["run_timestamp"] = datetime.now(timezone.utc).isoformat()
    f.attrs["n_galaxies"] = n_galaxies
    f.attrs["n_samples"] = n_samples
    f.attrs["n_warmup"] = n_warmup
    f.attrs["n_chains"] = n_chains
    f.attrs["n_paths"] = n_paths
    f.attrs["batch_size"] = batch_size
    f.attrs["target_accept"] = target_accept
    return f


# ---------------------------------------------------------------------------
# Main fitting loop
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
    """Fit all galaxies and stream results to HDF5."""
    N, n_bands = obs_flux.shape
    P = len(SPS_PARAM_NAMES)
    band_names = [emulator.band_names[int(i)] for i in band_idx]
    lows = jnp.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    highs = jnp.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    lows_np = np.asarray(lows)
    highs_np = np.asarray(highs)

    mode_str = "multi-path Pathfinder only" if pathfinder_only else "Pathfinder + NUTS"
    print(
        f"\nFitting {N:,} galaxies  |  batch={batch_size}  paths={n_paths}  "
        f"chains={n_chains}  warmup={n_warmup}  samples={n_samples}"
    )
    pf_path_mode = "parallel" if vmap_pf_paths else "sequential"
    print(f"  {mode_str}  |  eps0={eps0:.4g}  target_accept={target_accept}  "
          f"pf_paths={pf_path_mode}  pf_samples={n_pf_samples}  "
          f"metric={'diagonal' if diag_metric else 'dense'}")

    log_post_fn = make_log_posterior_fn(emulator, band_idx, make_log_prior_fn(prior_specs),
                                        min_frac_err=min_frac_err)
    batch_pf, batch_nuts = build_batch_fns(
        log_post_fn, P, n_samples, n_warmup, n_pf_samples, eps0, target_accept,
        vmap_paths=vmap_pf_paths,
    )

    h5 = create_output_file(
        output_path, N, n_samples, n_chains, P, band_names, emulator_path,
        batch_size, n_warmup, n_paths, target_accept, prior_specs, galaxy_ids, pathfinder_only,
    )

    # Background HDF5 writer.  Only this thread touches the file during the run;
    # the main thread joins it before reading back, so access stays single-threaded.
    wq: queue.Queue = queue.Queue(maxsize=2)

    def _writer():
        while True:
            item = wq.get()
            if item is None:
                break
            s, e, tmap, inv, elbo, pf_ok, samp, acc, step, divf, rhat, rchi2 = item
            h5["theta_map"][s:e] = tmap
            h5["inv_mass"][s:e] = inv
            h5["elbo"][s:e] = elbo
            h5["pathfinder_ok"][s:e] = pf_ok
            h5["accept_rate"][s:e] = acc
            h5["step_size"][s:e] = step
            h5["divergent_frac"][s:e] = divf
            h5["rhat"][s:e] = rhat
            h5["reduced_chi2_map"][s:e] = rchi2
            if samp is not None:
                h5["theta_samples"][s:e] = samp
            wq.task_done()

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()

    rng = jax.random.PRNGKey(seed)
    n_batches = (N + batch_size - 1) // batch_size
    dummy_obs = np.zeros((batch_size, n_bands), dtype=np.float32)
    dummy_err = np.full((batch_size, n_bands), MISSING_SIGMA, dtype=np.float32)
    metric_id_fallback = 0
    metric_floored = 0
    t0 = time.perf_counter()

    for bi in range(n_batches):
        start = bi * batch_size
        end_true = min(start + batch_size, N)
        true = end_true - start
        pad = batch_size - true

        if pad > 0:
            obs_b = np.concatenate([obs_flux[start:end_true], dummy_obs[:pad]], axis=0)
            err_b = np.concatenate([flux_err[start:end_true], dummy_err[:pad]], axis=0)
        else:
            obs_b = obs_flux[start:end_true]
            err_b = flux_err[start:end_true]

        obs_jax = jnp.asarray(obs_b)
        err_jax = jnp.asarray(err_b)

        rng, k_pinit, k_pf, k_cinit, k_nuts = jax.random.split(rng, 5)

        # ---- multi-path Pathfinder ----
        path_inits = jax.random.normal(k_pinit, (batch_size, n_paths, P)) * init_jitter
        path_inits = path_inits.at[:, 0, :].set(0.0)  # path 0 at the domain midpoint

        # Seed each Pathfinder path at a specific (redshift, Av) covering the main
        # galaxy populations, with log_mass amplitude-solved at each seed's predicted SED.
        # This replaces the old approach of starting all paths at z~7 (domain midpoint)
        # where the emulator predicts near-zero optical flux, giving garbage log_mass inits.
        _SEEDS_Z_AV = [(0.3, 0.3), (1.0, 1.0), (3.0, 0.5), (7.0, 0.1)]
        _lm_lo, _lm_hi = float(PARAM_BOUNDS["log_mass"][0]), float(PARAM_BOUNDS["log_mass"][1])
        _z_lo, _z_hi = float(PARAM_BOUNDS["redshift"][0]), float(PARAM_BOUNDS["redshift"][1])
        _av_lo, _av_hi = float(PARAM_BOUNDS["Av"][0]), float(PARAM_BOUNDS["Av"][1])
        _bidx_j = jnp.asarray(band_idx, dtype=jnp.int32)
        _eff_err = jnp.maximum(err_jax, min_frac_err * jnp.abs(obs_jax))
        _mask_w = (err_jax < OBS_MASK_THRESH).astype(jnp.float32) / (_eff_err ** 2)
        for _pi, (_z_s, _av_s) in enumerate(_SEEDS_Z_AV[:n_paths]):
            _x_ref = (0.5 * (lows + highs)).at[REDSHIFT_IDX].set(float(_z_s)).at[AV_IDX].set(float(_av_s)).at[LOGMASS_IDX].set(8.0)
            _pred = emulator.predict(_x_ref[None, :])[0][_bidx_j]           # (n_bands,)
            _num = jnp.sum(_mask_w * obs_jax * _pred, axis=1)               # (B,)
            _den = jnp.sum(_mask_w * _pred ** 2, axis=1)                    # (B,)
            _s = jnp.where(_den > 0, _num / _den, 1.0)
            _lm = jnp.clip(8.0 + jnp.log10(jnp.clip(_s, 1e-30, 1e30)), _lm_lo, _lm_hi)
            _lm_u = jnp.clip((_lm - _lm_lo) / (_lm_hi - _lm_lo), 1e-4, 1 - 1e-4)
            _z_u = float(np.clip((_z_s - _z_lo) / (_z_hi - _z_lo), 1e-4, 1 - 1e-4))
            _av_u = float(np.clip((_av_s - _av_lo) / (_av_hi - _av_lo), 1e-4, 1 - 1e-4))
            path_inits = path_inits.at[:, _pi, REDSHIFT_IDX].set(float(np.log(_z_u / (1 - _z_u))))
            path_inits = path_inits.at[:, _pi, AV_IDX].set(float(np.log(_av_u / (1 - _av_u))))
            path_inits = path_inits.at[:, _pi, LOGMASS_IDX].set(jnp.log(_lm_u / (1 - _lm_u)))
        pf_keys = jax.random.split(k_pf, batch_size * n_paths).reshape(batch_size, n_paths, -1)

        pos, inv, elbo, ok = batch_pf(obs_jax, err_jax, path_inits, pf_keys)
        pos = np.array(pos)   # writable copy (np.asarray gives a read-only view of JAX arrays)
        inv = np.array(inv)
        elbo = np.asarray(elbo)
        pf_ok = np.asarray(ok) & np.isfinite(pos).all(-1) & np.isfinite(inv).reshape(batch_size, -1).all(-1)
        pos[~pf_ok] = 0.0
        inv[~pf_ok] = np.eye(P, dtype=np.float32)

        # Project the dense Pathfinder metric to PSD with bounded conditioning.
        # An indefinite inverse_mass_matrix is the usual cause of low acceptance
        # + divergences that step-size adaptation cannot fix.
        inv, n_id_fallback, n_floored = regularize_metric(inv, diag_only=diag_metric)
        metric_id_fallback += n_id_fallback
        metric_floored += n_floored

        # Fit quality at the Pathfinder MAP (pf_ok only checks finiteness).
        red_chi2 = reduced_chi2_at(pos, obs_b, err_b, emulator, band_idx, lows_np, highs_np,
                                   min_frac_err=min_frac_err)

        # ---- multi-chain NUTS ----
        if pathfinder_only:
            samples_out = None
            accept = np.zeros((batch_size, n_chains), dtype=np.float32)
            step = np.full((batch_size, n_chains), eps0, dtype=np.float32)
            divf = np.zeros((batch_size, n_chains), dtype=np.float32)
            rhat = np.full((batch_size, P), np.nan, dtype=np.float32)
        else:
            # Geometry-aware chain jitter: scale perturbations by the Pathfinder
            # posterior std per dimension (sqrt of inv-mass-matrix diagonal) so
            # chains stay within the posterior region. Isotropic jitter ignores the
            # posterior shape and can send chains into log-prob cliffs where the
            # dual-averaging step size collapses to ~0, freezing the chain entirely.
            _inv_diag = jnp.diagonal(jnp.asarray(inv), axis1=-2, axis2=-1)  # (B, P)
            _pf_std = jnp.sqrt(jnp.clip(_inv_diag, 1e-6, None))            # (B, P)
            cjit = jax.random.normal(k_cinit, (batch_size, n_chains, P)) * chain_jitter * _pf_std[:, None, :]
            cjit = cjit.at[:, 0, :].set(0.0)  # chain 0 starts exactly at the Pathfinder mean
            chain_inits = jnp.asarray(pos)[:, None, :] + cjit
            nuts_keys = jax.random.split(k_nuts, batch_size * n_chains).reshape(
                batch_size, n_chains, -1
            )
            samples_j, accept_j, step_j, divf_j = batch_nuts(
                obs_jax, err_jax, chain_inits, jnp.asarray(inv), nuts_keys
            )
            samples_out = np.asarray(samples_j[:true])  # (true, C, S, P)
            accept = np.asarray(accept_j)
            step = np.asarray(step_j)
            divf = np.asarray(divf_j)
            rhat = np.full((batch_size, P), np.nan, dtype=np.float32)
            rhat[:true] = split_rhat(samples_out, lows_np, highs_np)

        wq.put(
            (
                start, end_true,
                pos[:true], inv[:true], elbo[:true], pf_ok[:true],
                samples_out, accept[:true], step[:true], divf[:true], rhat[:true],
                red_chi2[:true],
            )
        )

        # ---- progress ----
        elapsed = time.perf_counter() - t0
        rate = end_true / elapsed if elapsed > 0 else 0.0
        eta = (N - end_true) / rate if rate > 0 else float("inf")
        if pathfinder_only:
            diag = (f"pf_ok={np.mean(pf_ok[:true]):.1%}  "
                    f"redchi2_med={np.nanmedian(red_chi2[:true]):.1f}")
        else:
            diag = (
                f"accept={np.nanmean(accept[:true]):.2f}  "
                f"div={np.nanmean(divf[:true]):.1%}  "
                f"rhat<1.05={np.mean(np.nanmax(rhat[:true], axis=1) < 1.05):.1%}  "
                f"redchi2_med={np.nanmedian(red_chi2[:true]):.1f}  "
                f"pf_ok={np.mean(pf_ok[:true]):.1%}"
            )
        print(
            f"  [{end_true:>{len(str(N))}}/{N}]  batch {bi + 1}/{n_batches}  "
            f"{rate:.0f} gal/s  ETA {eta:.0f}s  {diag}"
        )

    wq.put(None)
    writer.join()

    # ---- summary ----
    total = time.perf_counter() - t0
    pf_ok_all = np.asarray(h5["pathfinder_ok"][:])
    print(f"\n{'=' * 60}")
    print(f"Finished {N:,} galaxies in {total:.1f}s  ({N / total:.0f} gal/s)")
    print(f"  Pathfinder OK:    {pf_ok_all.sum():,}/{N}  ({pf_ok_all.mean():.1%})")
    rchi2_all = np.asarray(h5["reduced_chi2_map"][:])
    finite_rchi2 = rchi2_all[np.isfinite(rchi2_all)]
    if finite_rchi2.size:
        frac_bad = float(np.mean(finite_rchi2 > 10.0))
        print(f"  Reduced chi2 @ MAP: median={np.median(finite_rchi2):.2f}  "
              f"frac>10={frac_bad:.1%}")
        if frac_bad > 0.1:
            print("  !  Many galaxies have a poor MAP fit (reduced chi2 > 10) -- the model "
                  "cannot reproduce these SEDs, or initialisation is still off. Cut on "
                  "reduced_chi2_map downstream; these posteriors are not trustworthy.")
    print(f"  Metric regularised (eigen-floored): {metric_floored:,}  |  "
          f"identity fallback: {metric_id_fallback:,}")
    if metric_id_fallback > 0.1 * N or metric_floored > 0.5 * N:
        print("  !  Pathfinder metrics were frequently indefinite/ill-conditioned "
              "(expected with --dense-metric; harmless with the default diagonal metric).")
    if not pathfinder_only:
        acc = np.asarray(h5["accept_rate"][:])
        div = np.asarray(h5["divergent_frac"][:])
        rh = np.asarray(h5["rhat"][:])
        worst_rhat = np.nanmax(rh, axis=1)
        print(f"  Mean accept:      {np.nanmean(acc):.3f}")
        print(f"  Galaxies w/ any divergence:  {np.mean(np.nanmax(div, axis=1) > 0):.1%}")
        print(f"  Galaxies w/ max-rhat < 1.05: {np.mean(worst_rhat < 1.05):.1%}  "
              f"(>1.1: {np.mean(worst_rhat > 1.1):.1%} -- likely multimodal / unconverged)")
        n_div = int((np.nanmax(div, axis=1) > 0.01).sum())
        if n_div:
            print(f"  !  {n_div:,} galaxies with >1% divergences "
                  f"-- raise --target-accept (e.g. 0.95) or revisit priors.")
    print(f"  Output:           {output_path}")
    h5.close()


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------


def check_gpu_linalg() -> None:
    """Probe the GPU linear-algebra path Pathfinder/NUTS depend on.

    Pathfinder's ELBO step and the NUTS dense metric call batched
    Cholesky / QR via cuSolver.  If cuSolver is broken in this environment
    (driver/jaxlib mismatch, or no workspace headroom), it raises
    'INTERNAL: cuSolver internal error'.  Running a trivial solve here
    separates an environment problem (this probe fails) from a data problem
    (this probe passes but a real batch fails).
    """
    backend = jax.default_backend()
    try:
        a = jnp.eye(8, dtype=jnp.float32) + 1e-3
        batched = jnp.broadcast_to(a, (16, 8, 8))
        jax.vmap(jnp.linalg.cholesky)(batched).block_until_ready()
        jax.vmap(lambda m: jnp.linalg.qr(m)[0])(batched).block_until_ready()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "GPU linear-algebra self-test failed -- this is an environment "
            "issue, not your data.\n"
            f"  backend = {backend}\n"
            f"  error   = {type(e).__name__}: {e}\n"
            "Likely causes / fixes:\n"
            "  1. cuSolver had no memory for its workspace. This script already "
            "sets XLA_PYTHON_CLIENT_PREALLOCATE=false; also try lowering "
            "--batch-size / --n-paths, or set XLA_PYTHON_CLIENT_MEM_FRACTION=0.6.\n"
            "  2. jaxlib / CUDA / cuSolver version mismatch. Verify the jax[cuda] "
            "build matches the cluster's CUDA module (some jaxlib releases break "
            "even a trivial solve on GPU).\n"
            "  3. As a last resort, force CPU linear algebra by running with "
            "JAX_PLATFORMS=cpu (slow) to confirm the rest of the pipeline is sound."
        ) from e
    print(f"GPU linalg self-test OK (backend={backend}).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and dispatch to the catalogue fitting pipeline."""
    parser = argparse.ArgumentParser(
        description="Fit a galaxy catalogue with GPU-batched multi-path Pathfinder + multi-chain NUTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("catalogue", nargs="?", help="Input catalogue (FITS/CSV/HDF5)")
    parser.add_argument("config", nargs="?", help="Band config JSON")
    parser.add_argument("--emulator", default=str(DEFAULT_EMULATOR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Galaxies per GPU call. Decrease on OOM.")
    parser.add_argument("--n-paths", type=int, default=4,
                        help="Pathfinder paths per galaxy (best ELBO selected).")
    parser.add_argument("--n-chains", type=int, default=2,
                        help="NUTS chains per galaxy (>=2 enables cross-chain R-hat).")
    parser.add_argument("--n-warmup", type=int, default=300,
                        help="NUTS dual-averaging warmup steps (discarded).")
    parser.add_argument("--n-samples", type=int, default=500, help="NUTS samples per chain.")
    parser.add_argument("--n-pf-samples", type=int, default=50,
                        help="Pathfinder ELBO Monte-Carlo samples per path point (NOT L-BFGS "
                        "history). Each is an emulator eval: peak Pathfinder memory scales as "
                        "batch_size x n_paths(if --vmap-pf-paths) x n_pf_samples. Lower this first on OOM.")
    parser.add_argument("--vmap-pf-paths", action="store_true",
                        help="Run Pathfinder paths in parallel (faster, n_paths x peak memory). "
                        "Default runs them sequentially (memory of a single path).")
    parser.add_argument("--dense-metric", action="store_true",
                        help="Use the full PSD-regularized dense Pathfinder metric (captures "
                        "parameter correlations). Default is the diagonal metric, which is the "
                        "proven-stable choice; the raw dense L-BFGS inverse-Hessian is often "
                        "indefinite and gave near-zero acceptance.")
    parser.add_argument("--check-gradients", action="store_true",
                        help="Run an autodiff-vs-finite-difference gradient self-test on a few "
                        "galaxies and exit. Use this to rule out a non-smooth emulator.")
    parser.add_argument("--target-accept", type=float, default=0.8,
                        help="Dual-averaging target acceptance.")
    parser.add_argument("--step-size", type=float, default=0.0,
                        help="Initial NUTS step size for dual averaging. 0 = auto (0.5/d^0.25).")
    parser.add_argument("--init-jitter", type=float, default=2.0,
                        help="Std of Pathfinder path inits in unconstrained space.")
    parser.add_argument("--chain-jitter", type=float, default=0.5,
                        help="Std of NUTS chain inits around the Pathfinder mean.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pathfinder-only", action="store_true",
                        help="Skip NUTS; store MAP + dense inv_mass only.")
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
    if args.n_chains < 1 or args.n_paths < 1:
        parser.error("--n-chains and --n-paths must be >= 1.")

    eps0 = args.step_size if args.step_size > 0 else 0.5 / (len(SPS_PARAM_NAMES) ** 0.25)

    check_gpu_linalg()

    config = load_config(Path(args.config))
    obs_flux, flux_err, ids = load_catalogue(Path(args.catalogue), config)
    emulator, band_idx = load_emulator_and_band_indices(emulator_path, list(config["bands"].keys()))

    if args.check_gradients:
        log_post_fn = make_log_posterior_fn(
            emulator, band_idx, make_log_prior_fn(config["resolved_priors"])
        )
        check_gradients(log_post_fn, obs_flux, flux_err, seed=args.seed)
        return

    run_catalogue(
        obs_flux, flux_err, ids, emulator, band_idx, config["resolved_priors"],
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
        min_frac_err=float(config.get("min_frac_err", 0.05)),
    )


if __name__ == "__main__":
    main()