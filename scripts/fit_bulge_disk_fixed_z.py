#!/usr/bin/env python3
"""Fit COSMOS2025 bulge/disk photometry at fixed spectroscopic redshift using NSS.

Reads the bulge+disk flux decomposition catalogue, applies the same quality
filtering as the noise-model training script, then runs Nested Slice Sampling
over the 11 free SPS parameters (redshift fixed to `zfinal` per galaxy).

Outputs one HDF5 file per component with the same layout as fit_catalogue.py
(nss_samples shape (N, S, 12) with the redshift column filled from the
catalogue, so corner plots and downstream scripts work without modification).

Usage
-----
    python scripts/fit_bulge_disk_fixed_z.py
    python scripts/fit_bulge_disk_fixed_z.py --component disk --n-galaxies 100
    python scripts/fit_bulge_disk_fixed_z.py --component both --row-start 1000 --n-galaxies 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from astropy.table import Table

# ---------------------------------------------------------------------------
# Arachne imports
# ---------------------------------------------------------------------------

_ARACHNE = Path(__file__).parent.parent
sys.path.insert(0, str(_ARACHNE / "scripts"))

import os
os.chdir(str(_ARACHNE))

from fit_catalogue import (
    DEFAULT_EMULATOR,
    DEFAULT_XLA_CACHE,
    LOG2PI,
    OBS_MASK_THRESH,
    PARAM_BOUNDS,
    SPS_PARAM_NAMES,
    load_emulator_and_band_indices,
    setup_xla_cache,
    split_rhat,
    _logz_from_weights,
    _ess_from_weights,
)

try:
    import blackjax
    from blackjax.ns.utils import finalise as _ns_finalise
    from blackjax.ns.utils import log_weights as _ns_log_weights
    from blackjax.ns.utils import sample as _ns_resample
    _NSS_AVAILABLE = True
except Exception:
    _NSS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOGUE      = Path("/cosma7/data/dp276/dc-harv3/work/catalogs/fluxes_bulge_disk_C25.csv")
DEFAULT_OUTPUT = Path("/cosma7/data/dp276/dc-harv3/work/outputs/nss_bulge_disk")

BANDS_COSMOS   = ["f115w", "f150w", "f277w", "f444w"]
BAND_NAMES_EMU = [
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F444W",
]

# Redshift is the first SPS parameter; free params are the remaining 11.
_Z_IDX        = SPS_PARAM_NAMES.index("redshift")          # == 0
FREE_PARAM_NAMES = [p for p in SPS_PARAM_NAMES if p != "redshift"]
_FREE_LOWS    = np.array([PARAM_BOUNDS[p][0] for p in FREE_PARAM_NAMES], dtype=np.float32)
_FREE_HIGHS   = np.array([PARAM_BOUNDS[p][1] for p in FREE_PARAM_NAMES], dtype=np.float32)

BULGE_BT_BAND  = "f444w"
BULGE_BT_MIN   = 0.05


# ---------------------------------------------------------------------------
# Catalogue loading
# ---------------------------------------------------------------------------

def _replace_zero_errors(flux: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Replace per-band zero errors with the median error of <3-sigma detections.

    Mirrors the noise-model training preprocessing so that zero-uncertainty
    entries don't blow up chi2.  Returns a copy of err with zeros replaced.
    """
    err = err.copy()
    n_bands = flux.shape[1]
    for b in range(n_bands):
        zero_mask  = err[:, b] == 0
        lo3sig     = (err[:, b] > 0) & (flux[:, b] / np.where(err[:, b] > 0, err[:, b], 1) < 3)
        med        = float(np.median(err[lo3sig, b])) if lo3sig.any() else float(np.median(err[err[:, b] > 0, b]))
        err[zero_mask, b] = med
    return err


def load_component(component: str, row_start: int = 0, n_galaxies: int | None = None):
    """Load, filter, and return flux/err/redshift/id arrays for one component.

    Args:
        component:  'bulge' or 'disk'.
        row_start:  First row to use (after quality cuts), 0-based.
        n_galaxies: Max rows to return; None = all.

    Returns:
        obs_flux   (N, 4) float32  µJy
        flux_err   (N, 4) float32  µJy  (zero errors replaced)
        z_fixed    (N,)   float32
        galaxy_ids (N,)   int64
    """
    t = Table.read(CATALOGUE)

    # --- quality filter: finite flux and error (>=0) in all 4 bands ---
    ok = np.ones(len(t), dtype=bool)
    for band in BANDS_COSMOS:
        flux_col = f"flux_{component}_{band}_bd"
        err_col  = f"flux_{component}_{band}_bd_err"
        f = np.array(t[flux_col], dtype=float)
        e = np.array(t[err_col],  dtype=float)
        ok &= np.isfinite(f) & np.isfinite(e) & (e >= 0)

    # --- redshift must be finite and positive ---
    z = np.array(t["zfinal"], dtype=float)
    ok &= np.isfinite(z) & (z > 0)

    # --- bulge-only: require BT_f444w >= BULGE_BT_MIN ---
    if component == "bulge":
        bt = np.array(t[f"BT_{BULGE_BT_BAND}"], dtype=float)
        ok &= np.isfinite(bt) & (bt >= BULGE_BT_MIN)

    t = t[ok]
    print(f"  After quality cuts: {len(t)} {component} sources")

    # --- row slicing ---
    if row_start > 0:
        t = t[row_start:]
    if n_galaxies is not None:
        t = t[:n_galaxies]
    print(f"  Using rows {row_start}–{row_start + len(t) - 1} ({len(t)} galaxies)")

    # --- build arrays ---
    flux = np.array(
        [[t[f"flux_{component}_{b}_bd"][i] for b in BANDS_COSMOS] for i in range(len(t))],
        dtype=np.float32,
    )
    err = np.array(
        [[t[f"flux_{component}_{b}_bd_err"][i] for b in BANDS_COSMOS] for i in range(len(t))],
        dtype=np.float32,
    )

    # replace zero errors
    err = _replace_zero_errors(flux.astype(float), err.astype(float)).astype(np.float32)

    z_arr  = np.array(t["zfinal"], dtype=np.float32)
    id_arr = np.array(t["Id"],     dtype=np.int64)

    return flux, err, z_arr, id_arr


# ---------------------------------------------------------------------------
# Log-likelihood at fixed redshift (11-dim free parameters)
# ---------------------------------------------------------------------------

def make_log_likelihood_fixed_z(emulator, band_idx: np.ndarray, min_frac_err: float):
    """Return a log-likelihood function for 11-dim x with z passed separately.

    The returned function signature is:
        log_likelihood(x_free, obs_flux, flux_err, z_fixed) -> scalar

    where x_free is the 11 free SPS parameters (all except redshift), and
    z_fixed is the scalar redshift for this galaxy.  All four arguments are
    JAX-traced so the compiled binary is reused across all galaxies.
    """
    lows  = jnp.array(_FREE_LOWS)
    highs = jnp.array(_FREE_HIGHS)
    bidx  = jnp.array(band_idx, dtype=jnp.int32)

    def log_likelihood(x_free, obs_flux, flux_err, z_fixed):
        in_bounds = jnp.all((x_free >= lows) & (x_free <= highs))

        # Reconstruct 12-dim vector: [z, free_params...]
        x_full = jnp.concatenate([jnp.array([z_fixed]), x_free])

        pred = emulator.predict(x_full[None, :])[0][bidx]

        mask    = (flux_err < OBS_MASK_THRESH).astype(x_free.dtype)
        eff_err = (jnp.maximum(flux_err, min_frac_err * jnp.abs(obs_flux))
                   if min_frac_err > 0 else flux_err)
        chi2     = jnp.sum(mask * (obs_flux - pred) ** 2 / eff_err ** 2)
        log_norm = jnp.sum(mask * (-0.5 * LOG2PI - jnp.log(eff_err)))
        return jnp.where(in_bounds, log_norm - 0.5 * chi2, -jnp.inf)

    return log_likelihood


def build_nss_fns_fixed_z(log_prior_fn, log_like_fn, num_inner_steps: int, num_delete: int):
    """Build JIT-compiled NSS init/step functions for 11 free params + fixed z.

    z_fixed is passed as an explicit traced argument so the XLA binary is
    compiled once and reused for all galaxies regardless of their redshift.
    """
    @jax.jit
    def nss_init(initial_samples, obs_flux, flux_err, z_fixed):
        def ll(x): return log_like_fn(x, obs_flux, flux_err, z_fixed)
        algo = blackjax.nss(
            logprior_fn=log_prior_fn, loglikelihood_fn=ll,
            num_delete=num_delete, num_inner_steps=num_inner_steps,
        )
        return algo.init(initial_samples)

    @jax.jit
    def nss_step(rng_key, state, obs_flux, flux_err, z_fixed):
        def ll(x): return log_like_fn(x, obs_flux, flux_err, z_fixed)
        algo = blackjax.nss(
            logprior_fn=log_prior_fn, loglikelihood_fn=ll,
            num_delete=num_delete, num_inner_steps=num_inner_steps,
        )
        return algo.step(rng_key, state)

    return nss_init, nss_step


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def create_output_file(path: Path, n_galaxies: int, n_samples: int, component: str) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    N, S, P = n_galaxies, n_samples, len(SPS_PARAM_NAMES)   # store 12-dim for compatibility
    f.attrs["n_galaxies"]  = n_galaxies
    f.attrs["component"]   = component
    f.attrs["param_names"] = np.array([s.encode() for s in SPS_PARAM_NAMES])
    f.attrs["free_params"] = np.array([s.encode() for s in FREE_PARAM_NAMES])
    f.attrs["note"]        = b"redshift column is fixed to catalogue zfinal, not sampled"
    f.create_dataset("galaxy_id",    (N,),       dtype="i8",  data=np.zeros(N, dtype=np.int64))
    f.create_dataset("z_fixed",      (N,),       dtype="f4",  data=np.zeros(N, dtype=np.float32))
    f.create_dataset("nss_samples",  (N, S, P),  dtype="f4",  fillvalue=np.nan)
    f.create_dataset("nss_logZ",     (N,),       dtype="f4",  fillvalue=np.nan)
    f.create_dataset("nss_logZ_err", (N,),       dtype="f4",  fillvalue=np.nan)
    f.create_dataset("nss_ess",      (N,),       dtype="f4",  fillvalue=np.nan)
    f.create_dataset("nss_n_dead",   (N,),       dtype="i4",  fillvalue=0)
    f.create_dataset("nss_rhat",     (N, P),     dtype="f4",  fillvalue=np.nan)
    f.create_dataset("nss_time",     (N,),       dtype="f4",  fillvalue=0.0)
    return f


# ---------------------------------------------------------------------------
# Main fitting loop
# ---------------------------------------------------------------------------

def run_component(
    component: str,
    emulator_path: Path,
    out_path: Path,
    row_start: int,
    n_galaxies: int | None,
    num_live: int,
    num_inner_steps: int,
    num_delete: int,
    termination: float,
    n_samples_out: int,
    min_frac_err: float,
    seed: int,
) -> None:
    print(f"\n{'='*60}")
    print(f"Component: {component.upper()}")

    obs_flux, flux_err, z_fixed_arr, galaxy_ids = load_component(component, row_start, n_galaxies)
    N = len(galaxy_ids)
    P_free = len(FREE_PARAM_NAMES)

    emulator, band_idx = load_emulator_and_band_indices(emulator_path, BAND_NAMES_EMU)
    print(f"  Emulator: {emulator_path.name}  bands: {BAND_NAMES_EMU}")
    print(f"  Free params: {P_free} (redshift fixed)")
    print(f"  num_live={num_live}  num_inner_steps={num_inner_steps}  num_delete={num_delete}")
    print(f"  termination={termination}  n_samples_out={n_samples_out}")

    # --- build NSS functions ---
    log_like_fn   = make_log_likelihood_fixed_z(emulator, band_idx, min_frac_err)
    log_prior_fn  = lambda x: jnp.float32(0.0)   # uniform; bounds enforced in likelihood

    nss_init_fn, nss_step_fn = build_nss_fns_fixed_z(
        log_prior_fn, log_like_fn, num_inner_steps, num_delete)

    lows_np  = _FREE_LOWS
    highs_np = _FREE_HIGHS

    # --- warm-up: compile on galaxy 0 ---
    print("\nWarm-up: compiling or loading XLA binary ...", flush=True)
    rng = jax.random.PRNGKey(seed)
    rng, k0 = jax.random.split(rng)
    obs0   = jnp.asarray(obs_flux[0])
    err0   = jnp.asarray(flux_err[0])
    z0     = jnp.float32(z_fixed_arr[0])
    live0  = jax.random.uniform(k0, (num_live, P_free),
                                minval=jnp.asarray(lows_np), maxval=jnp.asarray(highs_np))
    t_wu = time.perf_counter()
    _st  = nss_init_fn(live0, obs0, err0, z0)
    rng, _k = jax.random.split(rng)
    _st, _  = nss_step_fn(_k, _st, obs0, err0, z0)
    jax.block_until_ready(_st)
    t_wu = time.perf_counter() - t_wu
    src = "loaded from XLA cache" if t_wu < 60 else "freshly compiled"
    print(f"  Done in {t_wu:.1f}s ({src}).", flush=True)

    # --- output file ---
    with create_output_file(out_path, N, n_samples_out, component) as hf:
        hf["galaxy_id"][:] = galaxy_ids
        hf["z_fixed"][:]   = z_fixed_arr

        times = []
        for i in range(N):
            obs_i = jnp.asarray(obs_flux[i])
            err_i = jnp.asarray(flux_err[i])
            z_i   = jnp.float32(z_fixed_arr[i])

            # initialise live points
            rng, k_live = jax.random.split(rng)
            live_pts = jax.random.uniform(
                k_live, (num_live, P_free),
                minval=jnp.asarray(lows_np), maxval=jnp.asarray(highs_np),
            )
            state = nss_init_fn(live_pts, obs_i, err_i, z_i)
            jax.block_until_ready(state)

            # nested sampling loop (do-while: at least one step so dead is never empty)
            t0   = time.perf_counter()
            dead = []
            rng, k_step = jax.random.split(rng)
            state, dead_info = nss_step_fn(k_step, state, obs_i, err_i, z_i)
            dead.append(dead_info)
            while (float(state.integrator.logZ_live)
                   - float(state.integrator.logZ)) > termination:
                rng, k_step = jax.random.split(rng)
                state, dead_info = nss_step_fn(k_step, state, obs_i, err_i, z_i)
                dead.append(dead_info)

            jax.block_until_ready(state)
            t_gal = time.perf_counter() - t0

            # --- evidence, ESS, posterior samples ---
            final_state       = _ns_finalise(state, dead)
            rng, w_key, s_key = jax.random.split(rng, 3)
            logw              = _ns_log_weights(w_key, final_state)
            logz, logz_err    = _logz_from_weights(logw)
            ess               = _ess_from_weights(logw)

            resampled    = _ns_resample(s_key, final_state, shape=n_samples_out)
            samples_free = np.array(resampled.position, dtype=np.float32)  # (S, P_free)

            # Expand to 12-dim: prepend fixed-z column so downstream scripts work
            z_col      = np.full((n_samples_out, 1), z_fixed_arr[i], dtype=np.float32)
            samples_12 = np.concatenate([z_col, samples_free], axis=1)

            # R-hat on 11 free params; split_rhat expects (B, C, S, P) → reshape
            rhat_vals = split_rhat(samples_free[None, None, :, :], lows=None, highs=None)[0]
            rhat_12   = np.full(len(SPS_PARAM_NAMES), np.nan, dtype=np.float32)
            rhat_12[1:] = rhat_vals

            n_dead   = len(dead) * dead[0].particles.loglikelihood.shape[0] if dead else 0
            rhat_max = float(np.nanmax(rhat_vals))
            times.append(t_gal)
            eta = (N - i - 1) * float(np.median(times))

            print(f"  [{i+1:4d}/{N}]  z={z_fixed_arr[i]:.3f}  "
                  f"logZ={logz:.2f}±{logz_err:.2f}  "
                  f"ESS={ess:.0f}  n_dead={n_dead}  "
                  f"rhat_max={rhat_max:.3f}  "
                  f"t={t_gal:.1f}s  ETA={eta:.0f}s", flush=True)

            hf["nss_samples"][i]  = samples_12
            hf["nss_logZ"][i]     = logz
            hf["nss_logZ_err"][i] = logz_err
            hf["nss_ess"][i]      = ess
            hf["nss_n_dead"][i]   = n_dead
            hf["nss_rhat"][i]     = rhat_12
            hf["nss_time"][i]     = t_gal
            hf.flush()

        print(f"\n{'='*60}")
        print(f"Done: {N} {component} galaxies in {sum(times):.0f}s")
        print(f"  time/gal: median={np.median(times):.1f}s")
        print(f"  Output: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if not _NSS_AVAILABLE:
        raise RuntimeError("blackjax.nss not available — check blackjax installation.")

    parser = argparse.ArgumentParser(
        description="Fit COSMOS2025 bulge/disk photometry at fixed redshift using NSS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--component", choices=["bulge", "disk", "both"], default="both",
                        help="Which component(s) to fit.")
    parser.add_argument("--catalogue", default=str(CATALOGUE),
                        help="Path to fluxes_bulge_disk_C25.csv.")
    parser.add_argument("--emulator",  default=str(DEFAULT_EMULATOR),
                        help="Path to parrot_emulator.eqx.")
    parser.add_argument("--out-dir",   default=str(DEFAULT_OUTPUT),
                        help="Output directory (one HDF5 per component).")
    parser.add_argument("--row-start", type=int, default=0,
                        help="First row (after quality cuts) to process.")
    parser.add_argument("--n-galaxies", type=int, default=None,
                        help="Max galaxies to fit per component.")
    parser.add_argument("--num-live",         type=int,   default=500)
    parser.add_argument("--num-inner-steps",  type=int,   default=24)
    parser.add_argument("--num-delete",       type=int,   default=50)
    parser.add_argument("--termination",      type=float, default=-3.0)
    parser.add_argument("--n-samples-out",    type=int,   default=500)
    parser.add_argument("--min-frac-err",     type=float, default=0.15,
                        help="Minimum fractional flux error floor (0 = disabled).")
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--xla-cache-dir",    default=str(DEFAULT_XLA_CACHE),
                        help="XLA compilation cache directory. '' to disable.")
    args = parser.parse_args()

    if args.xla_cache_dir:
        setup_xla_cache(Path(args.xla_cache_dir))

    emulator_path = Path(args.emulator)
    out_dir       = Path(args.out_dir)
    components    = ["bulge", "disk"] if args.component == "both" else [args.component]

    for comp in components:
        out_path = out_dir / f"results_{comp}.hdf5"
        run_component(
            component        = comp,
            emulator_path    = emulator_path,
            out_path         = out_path,
            row_start        = args.row_start,
            n_galaxies       = args.n_galaxies,
            num_live         = args.num_live,
            num_inner_steps  = args.num_inner_steps,
            num_delete       = args.num_delete,
            termination      = args.termination,
            n_samples_out    = args.n_samples_out,
            min_frac_err     = args.min_frac_err,
            seed             = args.seed,
        )


if __name__ == "__main__":
    main()
