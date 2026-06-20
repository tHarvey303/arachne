#!/usr/bin/env python3
r"""Production SED fitting for large galaxy catalogues.

Fits every galaxy in a catalogue using GPU-batched Pathfinder + NUTS:

  - Pathfinder (vmapped over B galaxies simultaneously) gives a MAP position
    and diagonal inverse mass matrix per galaxy — no sequential warmup.
  - NUTS (vmapped, using Pathfinder's mass matrix directly) draws posterior
    samples with no window_adaptation overhead.
  - A background thread writes each completed batch to HDF5 while the GPU
    is already computing the next batch.

Band-to-column mapping is provided via a JSON config file.

Band config JSON
----------------
    {
        "id_col":     "ID",
        "flux_unit":  "uJy",
        "bands": {
            "JWST/NIRCam.F090W": {"flux": "f_F090W", "err": "e_F090W"},
            "JWST/NIRCam.F115W": {"flux": "f_F115W", "err": "e_F115W"},
            ...
        }
    }

    flux_unit: "nJy" | "uJy" | "mJy" | "Jy" | "ABmag"
    id_col:    catalogue column to use as galaxy identifier (optional)

    Bands not present in the config are silently ignored.
    Galaxies with NaN / non-positive flux or error in a band have that
    band masked (set to sigma=1e10 nJy, contributing nothing to the chi2).

Usage
-----
    python scripts/fit_catalogue.py catalogue.fits bands.json
    python scripts/fit_catalogue.py catalogue.fits bands.json \\
        --emulator scripts/outputs/emulators/parrot_emulator.eqx \\
        --output outputs/catalogue_fit/results.hdf5 \\
        --n-samples 500 --batch-size 2000

    # Fast MAP-only run (Pathfinder, no NUTS):
    python scripts/fit_catalogue.py catalogue.fits bands.json --pathfinder-only

    # Print a config template for the loaded emulator:
    python scripts/fit_catalogue.py --print-config-template \\
        --emulator scripts/outputs/emulators/parrot_emulator.eqx

Output HDF5 layout
------------------
    /galaxy_id          (N,)          identifier from catalogue (int or str)
    /theta_map          (N, P)        Pathfinder MAP, unconstrained space
    /inv_mass           (N, P)        Pathfinder diagonal inv-Hessian
    /theta_samples      (N, S, P)     NUTS posterior samples, unconstrained
    /accept_rate        (N,)          mean NUTS acceptance rate per galaxy
    /pathfinder_ok      (N,)          bool: Pathfinder returned a finite MAP
    attrs: param_names, band_names, param_bounds_lo, param_bounds_hi,
           emulator_path, run_timestamp, n_samples, batch_size

    To recover physical parameters from /theta_samples:
        lo  = f["param_bounds_lo"][:]
        hi  = f["param_bounds_hi"][:]
        sps = lo + (hi - lo) / (1 + exp(-theta_samples))   # sigmoid
"""

from __future__ import annotations

import argparse
import json
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
except ImportError as e:
    raise ImportError("pip install blackjax") from e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EMULATOR = Path("scripts/outputs/emulators/parrot_emulator.eqx")
DEFAULT_OUTPUT = Path("outputs/catalogue_fit/results.hdf5")

# Sigma assigned to missing/masked bands — contributes ≈0 to chi-squared
MISSING_SIGMA: float = 1e10  # nJy

# Number of training-distribution standard deviations used to set parameter bounds
# when no explicit bounds are provided.  4σ covers >99.99% of a Gaussian training set.
DEFAULT_N_SIGMA: float = 4.0

FLUX_UNIT_TO_NJY: dict[str, float] = {
    "nJy": 1.0,
    "uJy": 1e3,
    "ujy": 1e3,
    "µJy": 1e3,
    "mJy": 1e6,
    "Jy": 1e9,
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    """Load and validate the band configuration JSON.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Validated config dict with defaults filled in.
    """
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

    print(f"Config: {len(cfg['bands'])} bands, flux_unit={unit!r}")
    return cfg


def derive_param_bounds(
    emulator,
    config: dict,
    n_sigma: float = DEFAULT_N_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive parameter bounds from the emulator's training distribution.

    Uses in_mean ± n_sigma * in_std from the emulator's input normalisation
    statistics, which reflect the actual training data range.  Any parameter
    can be overridden per-parameter via ``config["param_bounds"]``.

    Config override format (all keys optional)::

        {
            "param_bounds": {
                "redshift": [
                    0.0,
                    15.0,
                ],
                "log_mass": [
                    6.0,
                    13.0,
                ],
            }
        }

    Args:
        emulator:  Loaded ParrotEmulator (must have in_mean, in_std attributes).
        config:    Validated config dict; may contain optional "param_bounds" key.
        n_sigma:   Half-width in training-distribution standard deviations.

    Returns:
        lows:   (P,) float32 lower bounds in physical parameter space.
        highs:  (P,) float32 upper bounds in physical parameter space.
    """
    in_mean = np.array(emulator.in_mean)
    in_std = np.array(emulator.in_std)

    lows = (in_mean - n_sigma * in_std).astype(np.float32)
    highs = (in_mean + n_sigma * in_std).astype(np.float32)

    overrides = config.get("param_bounds", {})
    param_names = emulator.param_names
    for i, name in enumerate(param_names):
        if name in overrides:
            lo, hi = overrides[name]
            lows[i] = float(lo)
            highs[i] = float(hi)

    print("\nParameter bounds (in_mean ± 4σ from training distribution):")
    for i, name in enumerate(param_names):
        src = " [config override]" if name in overrides else ""
        print(f"  {name:25s}: [{lows[i]:.4g}, {highs[i]:.4g}]{src}")

    return lows, highs


def print_config_template(emulator_path: Path) -> None:
    """Print a JSON config template with all bands from the loaded emulator."""
    from arachne.emulator.parrot_emulator import ParrotEmulator

    emu = ParrotEmulator.load(emulator_path)
    bands = {}
    for b in emu.band_names:
        safe = b.replace("/", "_").replace(".", "_")
        bands[b] = {"flux": f"{safe}_FLUX", "err": f"{safe}_FLUXERR"}

    # Show derived bounds as a starting point for param_bounds overrides
    in_mean = np.array(emu.in_mean)
    in_std = np.array(emu.in_std)
    param_bounds = {
        name: [
            round(float(in_mean[i] - DEFAULT_N_SIGMA * in_std[i]), 4),
            round(float(in_mean[i] + DEFAULT_N_SIGMA * in_std[i]), 4),
        ]
        for i, name in enumerate(emu.param_names)
    }

    template = {
        "id_col": "ID",
        "flux_unit": "nJy",
        "param_bounds": param_bounds,
        "bands": bands,
    }
    print(json.dumps(template, indent=2))


# ---------------------------------------------------------------------------
# Catalogue loading
# ---------------------------------------------------------------------------


def load_catalogue(catalogue_path: Path, config: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """Load galaxy catalogue and extract flux / flux-error arrays.

    Supports FITS, CSV, ECSV, ASCII, and HDF5/FITS_rec via astropy.

    Missing/bad data per band (NaN, non-positive error, non-positive flux)
    is replaced with (obs_flux=0, flux_err=MISSING_SIGMA), which contributes
    zero to the chi-squared likelihood.

    Args:
        catalogue_path: Path to the catalogue.
        config: Validated config dict from load_config.

    Returns:
        obs_flux:   (N, B) float32 in nJy.
        flux_err:   (N, B) float32 in nJy.
        galaxy_ids: list of length N.
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
            f_nJy = 10.0 ** ((8.9 - f_raw) / 2.5) * 1e9
            e_nJy = f_nJy * np.abs(e_raw) * np.log(10.0) / 2.5
        else:
            f_nJy = f_raw * factor
            e_nJy = e_raw * factor

        valid = np.isfinite(f_nJy) & np.isfinite(e_nJy) & (e_nJy > 0) & (f_nJy > 0)
        obs_flux[valid, b] = f_nJy[valid].astype(np.float32)
        flux_err[valid, b] = e_nJy[valid].astype(np.float32)
        n_bad = int((~valid).sum())
        if n_bad:
            print(f"  {band}: {N - n_bad}/{N} detected ({n_bad} masked)")

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
    """Load a self-contained ParrotEmulator and resolve band indices.

    Args:
        emulator_path: Path to the .eqx checkpoint.
        band_names:    Ordered list of band names to use (from config).

    Returns:
        emulator:  Loaded ParrotEmulator.
        band_idx:  numpy int32 array of indices into the emulator output.
    """
    from arachne.emulator.parrot_emulator import ParrotEmulator

    emu = ParrotEmulator.load(emulator_path)
    print(f"\nLoaded ParrotEmulator: {len(emu.param_names)} params, {len(emu.band_names)} bands")
    print(f"  Params: {emu.param_names}")

    all_bands = emu.band_names
    missing = [b for b in band_names if b not in all_bands]
    if missing:
        raise ValueError(f"Bands not found in emulator: {missing}")

    band_idx = np.array([all_bands.index(b) for b in band_names], dtype=np.int32)
    return emu, band_idx


# ---------------------------------------------------------------------------
# Log-posterior (explicit galaxy data args — required for vmap)
# ---------------------------------------------------------------------------


def make_log_posterior_fn(
    emulator,
    band_idx: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
):
    """Return a pure-JAX log_posterior(theta, obs_flux, flux_err) -> scalar.

    Unlike fit_sed.py, obs_flux and flux_err are *explicit arguments* rather
    than closed-over values.  This is required for jax.vmap to batch over
    galaxies: each vmapped call receives a different (obs_flux, flux_err)
    slice while theta is the per-galaxy unconstrained parameter vector.

    Prior: uniform in physical space, enforced by the log-Jacobian term.
    Likelihood: normalised Gaussian chi-squared over the observed bands.
    Missing bands (flux_err == MISSING_SIGMA) contribute ≈0 to the chi-squared.

    Args:
        emulator:  Loaded ParrotEmulator.
        band_idx:  int32 array of band indices into emulator output.
        lows:      (P,) lower bounds in physical parameter space.
        highs:     (P,) upper bounds in physical parameter space.
    """
    _lows = jnp.array(lows, dtype=jnp.float32)
    _highs = jnp.array(highs, dtype=jnp.float32)
    _bidx = jnp.array(band_idx, dtype=jnp.int32)
    n_bands = len(band_idx)

    def log_posterior(
        theta: jnp.ndarray,
        obs_flux: jnp.ndarray,
        flux_err: jnp.ndarray,
    ) -> jnp.ndarray:
        sps = _lows + (_highs - _lows) * jax.nn.sigmoid(theta)
        pred = emulator.predict(sps[None, :])[0][_bidx]

        log_norm = -0.5 * n_bands * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(flux_err))
        log_like = log_norm - 0.5 * jnp.sum(((obs_flux - pred) / flux_err) ** 2)
        log_jac = jnp.sum(
            jnp.log(_highs - _lows) + jax.nn.log_sigmoid(theta) + jax.nn.log_sigmoid(-theta)
        )
        return log_like + log_jac

    return log_posterior


# ---------------------------------------------------------------------------
# Compiled batch functions
# ---------------------------------------------------------------------------


def build_batch_fns(
    log_posterior_fn,
    n_params: int,
    n_samples: int,
    n_pf_samples: int,
    step_size: float,
) -> tuple:
    """JIT-compile vmapped Pathfinder and NUTS callables.

    Both returned functions accept a batch of B galaxies as leading arrays
    and run them simultaneously on the GPU in a single kernel launch.

    JIT compilation happens on the first call (typically 20–60 s for a
    batch of 2000 galaxies).  All subsequent batches of the same size are
    instant (cache hit).  The last batch is padded to the same size so that
    no recompilation is triggered.

    Args:
        log_posterior_fn: (theta, obs_flux, flux_err) -> scalar.
        n_params:         Number of SPS parameters.
        n_samples:        NUTS samples per galaxy.
        n_pf_samples:     L-BFGS samples for Pathfinder.
        step_size:        NUTS step size in preconditioned space.

    Returns:
        batch_pathfinder: (obs_B, err_B, keys_B) -> (maps_B, inv_masses_B)
        batch_nuts:       (obs_B, err_B, maps_B, inv_masses_B, keys_B)
                          -> (samples_B, accept_rates_B)
    """
    theta0 = jnp.zeros(n_params, dtype=jnp.float32)

    # --- Pathfinder (one galaxy) ---
    def _pf_one(obs_flux, flux_err, rng_key):
        def lp(theta):
            return log_posterior_fn(theta, obs_flux, flux_err)

        state, _ = pf_mod.approximate(rng_key, lp, theta0, num_samples=n_pf_samples)
        return state.position, state.alpha

    batch_pathfinder = jax.jit(jax.vmap(_pf_one))

    # --- NUTS chain (one galaxy) ---
    def _nuts_one(obs_flux, flux_err, theta_map, inv_mass, rng_key):
        def lp(theta):
            return log_posterior_fn(theta, obs_flux, flux_err)

        kernel = blackjax.nuts(lp, step_size=step_size, inverse_mass_matrix=inv_mass)
        state = kernel.init(theta_map)

        def one_step(carry, rng_k):
            s, info = kernel.step(rng_k, carry)
            return s, (s.position, info.acceptance_rate)

        _, (samples, ar) = jax.lax.scan(one_step, state, jax.random.split(rng_key, n_samples))
        return samples, jnp.mean(ar)

    batch_nuts = jax.jit(jax.vmap(_nuts_one))

    return batch_pathfinder, batch_nuts


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------


def create_output_file(
    output_path: Path,
    n_galaxies: int,
    n_samples: int,
    n_params: int,
    band_names: list[str],
    param_names: list[str],
    lows: np.ndarray,
    highs: np.ndarray,
    emulator_path: Path,
    batch_size: int,
    galaxy_ids: list,
) -> h5py.File:
    """Pre-allocate the output HDF5 file.

    Pre-allocation (rather than appending) lets background writes jump
    directly to the correct offset without any file-size bookkeeping.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(output_path, "w")

    # Galaxy IDs (integer or string)
    try:
        f.create_dataset("galaxy_id", data=np.array(galaxy_ids, dtype=np.int64))
    except (ValueError, TypeError):
        dt = h5py.string_dtype()
        ids = np.array([str(g) for g in galaxy_ids], dtype=object)
        f.create_dataset("galaxy_id", data=ids, dtype=dt)

    ckw = dict(compression="gzip", compression_opts=4)
    N, P, S = n_galaxies, n_params, n_samples
    f.create_dataset("theta_map", shape=(N, P), dtype=np.float32, **ckw)
    f.create_dataset("inv_mass", shape=(N, P), dtype=np.float32, **ckw)
    f.create_dataset("theta_samples", shape=(N, S, P), dtype=np.float32, **ckw)
    f.create_dataset("accept_rate", shape=(N,), dtype=np.float32)
    f.create_dataset("pathfinder_ok", shape=(N,), dtype=bool)

    f.attrs["param_names"] = [s.encode() for s in param_names]
    f.attrs["band_names"] = [s.encode() for s in band_names]
    f.attrs["param_bounds_lo"] = lows
    f.attrs["param_bounds_hi"] = highs
    f.attrs["emulator_path"] = str(emulator_path)
    f.attrs["run_timestamp"] = datetime.now(timezone.utc).isoformat()
    f.attrs["n_galaxies"] = n_galaxies
    f.attrs["n_samples"] = n_samples
    f.attrs["batch_size"] = batch_size
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
    lows: np.ndarray,
    highs: np.ndarray,
    output_path: Path,
    emulator_path: Path,
    batch_size: int,
    n_samples: int,
    n_pf_samples: int,
    step_size: float,
    seed: int,
    pathfinder_only: bool,
) -> None:
    """Fit all galaxies and write results to HDF5.

    Batches of ``batch_size`` galaxies are processed together on the GPU.
    The last batch is padded to ``batch_size`` with uninformative dummy
    galaxies (flux_err = MISSING_SIGMA) to avoid JAX recompilation, then
    the padding rows are discarded before writing.

    A background thread writes each completed batch to disk while the GPU
    computes the next one.

    Args:
        obs_flux:        (N, n_bands) observed fluxes in nJy.
        flux_err:        (N, n_bands) flux uncertainties in nJy.
        galaxy_ids:      Length-N list of galaxy identifiers.
        emulator:        Loaded ParrotEmulator.
        band_idx:        int32 array of band indices into emulator output.
        lows:            (P,) lower bounds in physical parameter space.
        highs:           (P,) upper bounds in physical parameter space.
        output_path:     Output HDF5 path.
        emulator_path:   Used for metadata only.
        batch_size:      Galaxies per GPU call.
        n_samples:       NUTS posterior samples per galaxy.
        n_pf_samples:    Pathfinder L-BFGS samples.
        step_size:       NUTS step size.
        seed:            JAX PRNG seed.
        pathfinder_only: If True, skip NUTS and store only MAP + inv_mass.
    """
    N, n_bands = obs_flux.shape
    param_names = emulator.param_names
    n_params = len(param_names)
    band_names = [emulator.band_names[int(i)] for i in band_idx]

    mode_str = "Pathfinder-only (MAP + Gaussian approx)" if pathfinder_only else "Pathfinder + NUTS"
    print(
        f"\nFitting {N:,} galaxies  |  batch={batch_size}  "
        f"n_samples={n_samples}  step_size={step_size:.4g}"
    )
    print(f"  {mode_str}")

    log_post_fn = make_log_posterior_fn(emulator, band_idx, lows, highs)
    batch_pf, batch_nuts = build_batch_fns(
        log_post_fn, n_params, n_samples, n_pf_samples, step_size
    )

    h5 = create_output_file(
        output_path,
        N,
        n_samples,
        n_params,
        band_names,
        param_names,
        lows,
        highs,
        emulator_path,
        batch_size,
        galaxy_ids,
    )

    # Background HDF5 writer — overlaps CPU I/O with GPU compute.
    # Only one thread writes at a time; h5py's default (sec2) driver is safe.
    wq: queue.Queue = queue.Queue(maxsize=3)

    def _writer():
        while True:
            item = wq.get()
            if item is None:
                break
            start, end, maps, inv_masses, samples, accept_rates, pf_ok = item
            h5["theta_map"][start:end] = maps
            h5["inv_mass"][start:end] = inv_masses
            h5["accept_rate"][start:end] = accept_rates
            h5["pathfinder_ok"][start:end] = pf_ok
            if samples is not None:
                h5["theta_samples"][start:end] = samples
            h5.flush()
            wq.task_done()

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()

    rng = jax.random.PRNGKey(seed)
    n_batches = (N + batch_size - 1) // batch_size
    dummy_obs = np.zeros((batch_size, n_bands), dtype=np.float32)
    dummy_err = np.full((batch_size, n_bands), MISSING_SIGMA, dtype=np.float32)
    t0 = time.perf_counter()

    for bi in range(n_batches):
        start = bi * batch_size
        end_true = min(start + batch_size, N)
        pad = batch_size - (end_true - start)

        # Build padded batch (padding with dummy uninformative galaxies)
        if pad > 0:
            obs_b = np.concatenate([obs_flux[start:end_true], dummy_obs[:pad]], axis=0)
            err_b = np.concatenate([flux_err[start:end_true], dummy_err[:pad]], axis=0)
        else:
            obs_b = obs_flux[start:end_true]
            err_b = flux_err[start:end_true]

        rng, pf_key, nuts_key = jax.random.split(rng, 3)
        pf_keys = jax.random.split(pf_key, batch_size)
        nuts_keys = jax.random.split(nuts_key, batch_size)

        obs_jax = jnp.array(obs_b)
        err_jax = jnp.array(err_b)

        # --- Pathfinder ---
        maps, inv_masses = batch_pf(obs_jax, err_jax, pf_keys)

        # Detect and repair Pathfinder failures (non-finite MAP)
        pf_ok = np.isfinite(np.array(maps)).all(axis=-1)  # (B,)
        ok_mask = jnp.array(pf_ok)[:, None]
        maps = jnp.where(ok_mask, maps, jnp.zeros_like(maps))
        inv_masses = jnp.where(ok_mask, inv_masses, jnp.ones_like(inv_masses))

        # --- NUTS ---
        if pathfinder_only:
            samples_out = None
            accept_rates = np.zeros(batch_size, dtype=np.float32)
        else:
            samples_jax, accept_rates = batch_nuts(obs_jax, err_jax, maps, inv_masses, nuts_keys)
            samples_out = np.array(samples_jax[: end_true - start])
            accept_rates = np.array(accept_rates)

        # Trim padding before queuing the write
        sl = slice(0, end_true - start)
        wq.put(
            (
                start,
                end_true,
                np.array(maps[sl]),
                np.array(inv_masses[sl]),
                samples_out,
                accept_rates[sl],
                pf_ok[sl],
            )
        )

        # Progress
        elapsed = time.perf_counter() - t0
        done = end_true
        rate = done / elapsed
        eta = (N - done) / rate if rate > 0 else float("inf")
        mean_ar = float(np.mean(accept_rates[sl])) if not pathfinder_only else float("nan")
        pf_frac = float(np.mean(pf_ok[sl]))
        ar_str = f"  accept={mean_ar:.2f}" if not pathfinder_only else ""
        print(
            f"  [{done:>{len(str(N))}}/{N}]  "
            f"batch {bi + 1}/{n_batches}  "
            f"{rate:.0f} gal/s  ETA {eta:.0f}s"
            f"{ar_str}  pf_ok={pf_frac:.1%}"
        )

    # Drain writer thread before closing the file
    wq.put(None)
    writer.join()

    # Summary diagnostics
    total_time = time.perf_counter() - t0
    pf_ok_all = np.array(h5["pathfinder_ok"][:])
    accept_all = np.array(h5["accept_rate"][:])

    print(f"\n{'=' * 60}")
    print(f"Finished {N:,} galaxies in {total_time:.1f}s  ({N / total_time:.0f} gal/s)")
    print(f"  Pathfinder OK:    {pf_ok_all.sum():,}/{N}  ({pf_ok_all.mean():.1%})")
    if not pathfinder_only:
        print(f"  Mean accept rate: {accept_all.mean():.3f}")
        n_low = int((accept_all < 0.65).sum())
        n_high = int((accept_all > 0.95).sum())
        if n_low:
            print(f"  ⚠  {n_low:,} galaxies accept < 0.65 — try --step-size {step_size * 0.5:.4g}")
        if n_high:
            print(f"  ⚠  {n_high:,} galaxies accept > 0.95 — try --step-size {step_size * 2.0:.4g}")
    print(f"  Output:           {output_path}")

    h5.close()


# ---------------------------------------------------------------------------
# Single-galaxy debug
# ---------------------------------------------------------------------------


def debug_galaxy(
    galaxy_idx: int,
    obs_flux: np.ndarray,
    flux_err: np.ndarray,
    emulator,
    band_idx: np.ndarray,
    band_names: list[str],
    lows: np.ndarray,
    highs: np.ndarray,
    seed: int = 0,
    n_scan: int = 50,
) -> None:
    """Run a detailed diagnostic on one galaxy and print a report.

    Checks, in order:
      1. Observed flux vs midpoint prediction — confirms the emulator is
         producing non-trivial output for this object.
      2. Log-posterior value and per-parameter gradient at theta=0
         (midpoint of prior).  Near-zero gradients reveal which parameters
         have no constraining power from the data.
      3. 1-D likelihood scans for each parameter — varies one parameter at
         a time across its prior range while holding others at the midpoint.
         A flat scan means that parameter is genuinely unconstrained.
      4. Pathfinder run (single galaxy, no vmapping) — shows the MAP
         position and posterior width.
      5. 20 NUTS steps from the MAP — measures the actual acceptance rate
         with the current step size.

    Args:
        galaxy_idx: Row index in the catalogue.
        obs_flux:   (N, B) observed flux array.
        flux_err:   (N, B) flux error array.
        emulator:   Loaded ParrotEmulator.
        band_idx:   int32 array of band indices into emulator output.
        band_names: List of band name strings (same order as band_idx).
        lows:       (P,) lower bounds in physical parameter space.
        highs:      (P,) upper bounds in physical parameter space.
        seed:       PRNG seed.
        n_scan:     Number of points per 1-D parameter scan.
    """
    param_names = emulator.param_names
    n_params = len(param_names)
    obs = jnp.array(obs_flux[galaxy_idx])
    err = jnp.array(flux_err[galaxy_idx])
    _lows = jnp.array(lows)
    _highs = jnp.array(highs)
    _bidx = jnp.array(band_idx, dtype=jnp.int32)

    print(f"\n{'=' * 65}")
    print(f"DEBUG: galaxy index {galaxy_idx}")
    print(f"{'=' * 65}")

    # ------------------------------------------------------------------
    # 1. Observed vs predicted flux at prior midpoint (theta=0)
    # ------------------------------------------------------------------
    theta0 = jnp.zeros(n_params)
    sps_mid = _lows + (_highs - _lows) * 0.5
    pred_mid = emulator.predict(sps_mid[None, :])[0][_bidx]

    print("\n[1] Observed flux vs midpoint prediction (uJy)")
    print(f"  {'Band':<30}  {'Obs':>10}  {'Err':>10}  {'Pred':>10}  {'SNR':>6}  {'chi':>7}")
    for b, name in enumerate(band_names):
        o = float(obs[b])
        e = float(err[b])
        p = float(pred_mid[b])
        masked = e >= MISSING_SIGMA / 2
        snr = o / e if not masked else float("nan")
        chi = (o - p) / e if not masked else float("nan")
        flag = " [masked]" if masked else ""
        print(
            f"  {name:<30}  {o / 1e3:>10.4f}  {e / 1e3:>10.4f}  "
            f"{p / 1e3:>10.4f}  {snr:>6.2f}  {chi:>7.3f}{flag}"
        )

    # ------------------------------------------------------------------
    # 2. Log-posterior value and gradient at theta=0
    # ------------------------------------------------------------------
    log_post_fn = make_log_posterior_fn(emulator, band_idx, lows, highs)

    def lp(theta):
        return log_post_fn(theta, obs, err)

    lp_val = float(lp(theta0))
    grad = np.array(jax.grad(lp)(theta0))

    print(f"\n[2] Log-posterior at theta=0 (prior midpoint): {lp_val:.4f}")
    print(f"  {'Parameter':<25}  {'lo':>9}  {'mid':>9}  {'hi':>9}  {'grad@0':>10}  {'|grad|':>8}")
    for i, name in enumerate(param_names):
        lo_v = float(lows[i])
        hi_v = float(highs[i])
        mid_v = (lo_v + hi_v) / 2
        g = float(grad[i])
        print(
            f"  {name:<25}  {lo_v:>9.3g}  {mid_v:>9.3g}  {hi_v:>9.3g}  {g:>10.4f}  {abs(g):>8.4f}"
        )
    print(f"\n  Gradient L2 norm:  {np.linalg.norm(grad):.6f}")
    print(f"  Gradient L∞ norm:  {np.abs(grad).max():.6f}")
    if np.abs(grad).max() < 1e-3:
        print("  *** WARNING: gradient is essentially zero — the likelihood")
        print("      has no information to guide Pathfinder or NUTS.")
        print("      Likely cause: emulator flux is insensitive to parameters")
        print("      in this prior range, OR the data SNR is too low.")

    # ------------------------------------------------------------------
    # 3. 1-D likelihood scans (vary one parameter, hold others at midpoint)
    # ------------------------------------------------------------------
    print(f"\n[3] 1-D log-likelihood scans (n_scan={n_scan} per parameter)")
    print(f"  log_like_mid = {lp_val:.3f}")
    print(f"  {'Parameter':<25}  {'Δlp_max':>9}  {'best_theta':>11}  {'best_sps':>11}  shape")

    theta_vals = np.linspace(-5.0, 5.0, n_scan)  # covers most of sigmoid range

    for i, name in enumerate(param_names):
        scan_theta = np.zeros((n_scan, n_params), dtype=np.float32)
        scan_theta[:, i] = theta_vals
        lp_vals = np.array(jax.vmap(lambda t: lp(t))(jnp.array(scan_theta)))
        valid = np.isfinite(lp_vals)
        if not valid.any():
            print(f"  {name:<25}  {'ALL NaN':>9}")
            continue
        lp_rel = lp_vals - lp_val
        best_i = int(np.argmax(lp_vals[valid]))
        best_theta = float(theta_vals[valid][best_i])
        best_sps = float(lows[i] + (highs[i] - lows[i]) / (1 + np.exp(-best_theta)))
        dlp_max = float(lp_rel[valid].max())

        # Characterise shape: flat, peaked, or monotone
        lp_sorted = lp_vals[valid]
        if dlp_max < 0.1:
            shape = "flat"
        elif np.argmax(lp_sorted) in (0, len(lp_sorted) - 1):
            shape = "monotone"
        else:
            shape = "peaked"

        print(f"  {name:<25}  {dlp_max:>9.3f}  {best_theta:>11.4f}  {best_sps:>11.4g}  {shape}")

    # ------------------------------------------------------------------
    # 4. Pathfinder
    # ------------------------------------------------------------------
    print("\n[4] Running Pathfinder on this galaxy...")
    rng = jax.random.PRNGKey(seed)
    rng, pf_key = jax.random.split(rng)
    state, pf_info = pf_mod.approximate(pf_key, lp, theta0, num_samples=200)
    theta_map = np.array(state.position)
    inv_mass = np.array(state.alpha)
    pf_ok = np.isfinite(theta_map).all()

    print(f"  Pathfinder converged: {pf_ok}")
    sps_map = lows + (highs - lows) / (1 + np.exp(-theta_map))
    pred_map = np.array(emulator.predict(jnp.array(sps_map[None, :]))[0][_bidx])

    print(f"  {'Parameter':<25}  {'theta_MAP':>10}  {'sps_MAP':>11}  {'inv_mass':>10}")
    for i, name in enumerate(param_names):
        print(f"  {name:<25}  {theta_map[i]:>10.4f}  {sps_map[i]:>11.4g}  {inv_mass[i]:>10.4f}")

    lp_map = float(lp(jnp.array(theta_map)))
    print(f"\n  log_posterior at MAP: {lp_map:.4f}  (vs midpoint: {lp_val:.4f})")
    print(f"\n  {'Band':<30}  {'Obs':>10}  {'Pred@MAP':>10}  {'chi':>7}")
    for b, name in enumerate(band_names):
        o = float(obs[b])
        e = float(err[b])
        p = float(pred_map[b])
        masked = e >= MISSING_SIGMA / 2
        chi = (o - p) / e if not masked else float("nan")
        flag = " [masked]" if masked else ""
        print(f"  {name:<30}  {o / 1e3:>10.4f}  {p / 1e3:>10.4f}  {chi:>7.3f}{flag}")

    # ------------------------------------------------------------------
    # 5. Short NUTS run from MAP
    # ------------------------------------------------------------------
    print("\n[5] 20 NUTS steps from MAP position...")
    n_params_f = len(param_names)
    step_size = 0.5 / (n_params_f**0.25)
    init_pos = jnp.array(theta_map) if pf_ok else theta0
    init_mass = jnp.array(inv_mass) if pf_ok else jnp.ones(n_params_f)

    nuts_kernel = blackjax.nuts(lp, step_size=step_size, inverse_mass_matrix=init_mass)
    state_nuts = nuts_kernel.init(init_pos)

    rng, nuts_key = jax.random.split(rng)

    def one_step(carry, rng_k):
        s, info = nuts_kernel.step(rng_k, carry)
        return s, (s.position, info.acceptance_rate, info.num_integration_steps)

    _, (samples, ar, n_steps) = jax.lax.scan(one_step, state_nuts, jax.random.split(nuts_key, 20))
    ar_np = np.array(ar)
    n_steps_np = np.array(n_steps)
    print(f"  Step size:           {step_size:.4f}")
    print(f"  Mean accept rate:    {ar_np.mean():.4f}")
    print(f"  Accept rates:        {np.round(ar_np, 3)}")
    print(f"  Mean leapfrog steps: {n_steps_np.mean():.1f}")
    print(
        f"  log_post range:      [{float(jax.vmap(lp)(samples).min()):.2f}, "
        f"{float(jax.vmap(lp)(samples).max()):.2f}]"
    )
    print(f"\n{'=' * 65}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and dispatch to the catalogue fitting pipeline."""
    parser = argparse.ArgumentParser(
        description="Fit a large galaxy catalogue with GPU-batched Pathfinder + NUTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("catalogue", nargs="?", help="Input catalogue (FITS/CSV/HDF5)")
    parser.add_argument("config", nargs="?", help="Band config JSON")
    parser.add_argument(
        "--emulator",
        default=str(DEFAULT_EMULATOR),
        help="Path to ParrotEmulator checkpoint (.eqx)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Galaxies per GPU call.  Increase for more GPU utilisation, "
        "decrease if you hit OOM errors.",
    )
    parser.add_argument("--n-samples", type=int, default=500, help="NUTS samples per galaxy")
    parser.add_argument("--n-pf-samples", type=int, default=200, help="Pathfinder L-BFGS samples")
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.0,
        help="NUTS step size. 0 = auto (0.5 / d^0.25). "
        "Tune based on accept rate: target 0.65-0.90.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pathfinder-only",
        action="store_true",
        help="Run only Pathfinder (MAP + Gaussian approx), skip NUTS. "
        "Much faster; /theta_samples dataset will be all zeros.",
    )
    parser.add_argument(
        "--print-config-template",
        action="store_true",
        help="Print a JSON config template for --emulator and exit.",
    )
    parser.add_argument(
        "--debug-galaxy",
        type=int,
        default=None,
        metavar="IDX",
        help="Run a detailed single-galaxy diagnostic on row IDX and exit. "
        "Prints gradient, 1-D likelihood scans, Pathfinder result, and "
        "short NUTS acceptance rate to diagnose fitting failures.",
    )
    args = parser.parse_args()

    emulator_path = Path(args.emulator)

    if args.print_config_template:
        print_config_template(emulator_path)
        return

    if not args.catalogue or not args.config:
        parser.error("catalogue and config are required unless --print-config-template is used.")

    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator not found: {emulator_path}")

    config = load_config(Path(args.config))
    obs_flux, flux_err, ids = load_catalogue(Path(args.catalogue), config)
    emulator, band_idx = load_emulator_and_band_indices(emulator_path, list(config["bands"].keys()))

    lows, highs = derive_param_bounds(emulator, config)
    n_params = len(emulator.param_names)
    step_size = args.step_size if args.step_size > 0 else 0.5 / (n_params**0.25)

    if args.debug_galaxy is not None:
        idx = args.debug_galaxy
        n = len(obs_flux)
        if not (0 <= idx < n):
            raise ValueError(f"--debug-galaxy {idx} out of range [0, {n - 1}]")
        band_names = list(config["bands"].keys())
        debug_galaxy(
            galaxy_idx=idx,
            obs_flux=obs_flux,
            flux_err=flux_err,
            emulator=emulator,
            band_idx=band_idx,
            band_names=band_names,
            lows=lows,
            highs=highs,
            seed=args.seed,
        )
        return

    run_catalogue(
        obs_flux,
        flux_err,
        ids,
        emulator,
        band_idx,
        lows,
        highs,
        output_path=Path(args.output),
        emulator_path=emulator_path,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        n_pf_samples=args.n_pf_samples,
        step_size=step_size,
        seed=args.seed,
        pathfinder_only=args.pathfinder_only,
    )


if __name__ == "__main__":
    main()
