#!/usr/bin/env python3
"""Fit a single integrated galaxy SED using the arachne emulator + NUTS + Pathfinder.

No spatial model, image, or PSF machinery — input is a 1-D flux vector and
per-band flux uncertainties.

Steps
-----
1. Load ParrotEmulator (self-contained checkpoint — param/band names stored inside).
2. Resolve JADES band indices from the emulator's full band list.
3. Generate a mock SED (emulator forward pass at true params + Gaussian noise).
4. Build log_posterior(theta): sigmoid reparameterisation + normalised Gaussian
   log-likelihood + log-Jacobian for a uniform prior in physical parameter space.
5. Warm-start with Pathfinder (L-BFGS MAP + diagonal mass-matrix estimate).
6. Run NUTS with window adaptation.
7. Report parameter recovery vs. truth.

Likelihood
----------
The log-posterior is::

    log p(theta | d) = log L(theta) + log J(theta)

    log L = -N/2 * log(2π) - sum(log σ_b) - 0.5 * sum(((d_b - f_b(θ)) / σ_b)²)

    log J = sum_i [ log(hi_i - lo_i) + log σ(θ_i) + log σ(-θ_i) ]

where f_b(θ) is the emulator flux in band b, σ_b is the flux uncertainty,
and log J is the log-Jacobian that makes the prior uniform in physical space
(as opposed to the U-shaped Beta(0,0) prior implied by omitting it).

Usage
-----
    python scripts/fit_sed.py
    python scripts/fit_sed.py --emulator /path/to/parrot.eqx
    python scripts/fit_sed.py --n-warmup 500 --n-samples 2000

Outputs (under outputs/sed_fitting/)
-------------------------------------
- mock_sed.npz    — observed fluxes, flux errors, and true params
- samples.hdf5    — posterior samples, shape (n_samples, N_params)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

OUTDIR = Path("outputs/sed_fitting")

# Default emulator path
DEFAULT_EMULATOR = Path("scripts/outputs/emulators/parrot_emulator.eqx")

# JADES-like NIRCam filter set: wide bands (excl. F070W) + F335M + F410M
JADES_BANDS = [
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F335M",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F444W",
]

# SPS parameter names and physical bounds — must match training library
SPS_PARAM_NAMES = [
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
    "redshift": (0.011, 1.410),
    "log_mass": (11.850, 12.500),
    "slope": (-0.300, -0.160),
    "fesc_lya": (0.900, 1.000),
    "dust_bump_amplitude": (4.500, 5.000),
    "log10metallicity": (-2.173, -1.912),
    "Av": (4.511, 5.012),
    "logsfr_ratio_0": (23.325, 29.212),
    "logsfr_ratio_1": (17.896, 23.865),
    "logsfr_ratio_2": (-29.456, -23.521),
    "logsfr_ratio_3": (24.008, 29.993),
    "logsfr_ratio_4": (17.941, 23.883),
}

# True parameters for the mock SED — midpoints of each training range
TRUE_PARAMS: dict[str, float] = {name: (lo + hi) / 2.0 for name, (lo, hi) in PARAM_BOUNDS.items()}

NOISE_FRACTION = 0.05  # 5% per-band flux uncertainty → S/N ~ 20


# ---------------------------------------------------------------------------
# Step 1 & 2: Load emulator and resolve band indices
# ---------------------------------------------------------------------------


def load_emulator_and_band_indices(
    emulator_path: Path,
) -> tuple:
    """Load a self-contained ParrotEmulator and compute JADES band indices.

    The ParrotEmulator checkpoint stores its own param/band name lists, so no
    external metadata is needed at load time.  After loading we verify that the
    checkpoint's parameter names match ``SPS_PARAM_NAMES`` and resolve the
    integer indices of ``JADES_BANDS`` within the emulator's full output vector.

    Args:
        emulator_path: Path to a ``*.eqx`` ParrotEmulator checkpoint.

    Returns:
        emulator:    Loaded ParrotEmulator instance.
        band_idx:    JAX integer array of shape (N_jades,) — indices into the
                     emulator's (N_pixels, N_all_bands) output selecting the
                     JADES bands in the order defined by ``JADES_BANDS``.

    Raises:
        ValueError: If the checkpoint's param names differ from ``SPS_PARAM_NAMES``,
                    or if any JADES band is absent from the emulator's band list.
    """
    from arachne.emulator.parrot_emulator import ParrotEmulator

    emulator = ParrotEmulator.load(emulator_path)
    print(f"Loaded ParrotEmulator from {emulator_path}")
    print(f"  {len(emulator.param_names)} params, {len(emulator.band_names)} bands")

    if emulator.param_names != SPS_PARAM_NAMES:
        raise ValueError(
            f"Emulator param names {emulator.param_names} don't match expected {SPS_PARAM_NAMES}"
        )

    all_bands = emulator.band_names
    missing = [b for b in JADES_BANDS if b not in all_bands]
    if missing:
        raise ValueError(f"JADES bands not found in emulator: {missing}")

    band_idx = jnp.array([all_bands.index(b) for b in JADES_BANDS], dtype=jnp.int32)
    print(f"  Using {len(JADES_BANDS)} JADES bands (indices {band_idx.tolist()})")
    return emulator, band_idx


# ---------------------------------------------------------------------------
# Step 3: Generate mock SED
# ---------------------------------------------------------------------------


def make_mock_sed(
    emulator,
    band_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, np.ndarray]:
    """Generate a noisy mock SED from ``TRUE_PARAMS``.

    Calls the emulator at the true physical parameters, extracts the JADES
    bands, and adds Gaussian noise scaled to ``NOISE_FRACTION`` of the true flux.

    Args:
        emulator:  Loaded ParrotEmulator instance.
        band_idx:  Integer indices selecting JADES bands from the full output.

    Returns:
        obs_flux:  Noisy observed fluxes, shape (N_jades,) in nJy.
        flux_err:  Per-band 1-sigma uncertainties, shape (N_jades,) in nJy.
        true_phys: True physical SPS params, shape (N_params,).
    """
    true_phys = np.array([TRUE_PARAMS[p] for p in SPS_PARAM_NAMES], dtype=np.float32)

    # predict expects (N_pixels, N_params); extract JADES subset
    all_flux = np.array(emulator.predict(true_phys[None, :]))[0]  # (N_all_bands,)
    true_flux = all_flux[np.array(band_idx)]  # (N_jades,)

    rng = np.random.default_rng(42)
    flux_err = NOISE_FRACTION * np.abs(true_flux)
    obs_flux = (true_flux + rng.normal(0.0, flux_err)).astype(np.float32)

    print(f"\nMock SED ({len(JADES_BANDS)} JADES bands, {NOISE_FRACTION * 100:.0f}% noise):")
    for b, band in enumerate(JADES_BANDS):
        print(
            f"  {band}: {obs_flux[b]:8.2f} ± {flux_err[b]:.2f} nJy"
            f"  (S/N = {obs_flux[b] / flux_err[b]:.1f})"
        )

    return jnp.array(obs_flux), jnp.array(flux_err.astype(np.float32)), true_phys


# ---------------------------------------------------------------------------
# Step 4: Log-posterior
# ---------------------------------------------------------------------------


def make_log_posterior(
    emulator,
    band_idx: jnp.ndarray,
    obs_flux: jnp.ndarray,
    flux_err: jnp.ndarray,
):
    """Build a pure-JAX log_posterior(theta) closure.

    The posterior is::

        log p(theta | d) = log L(theta) + log J(theta)

    **Likelihood** — normalised Gaussian over the JADES bands::

        log L = -N/2 * log(2π) - sum(log σ_b)
                - 0.5 * sum(((d_b - f_b(θ)) / σ_b)²)

    The normalisation constants are theta-independent but are included so
    the log-posterior equals a properly scaled log-probability.

    **Log-Jacobian** — corrects for the sigmoid reparameterisation so that the
    prior is uniform in physical parameter space rather than the U-shaped
    Beta(0,0) implied by a flat prior on the unconstrained vector::

        log J = sum_i [ log(hi_i - lo_i)
                       + log σ(θ_i) + log σ(-θ_i) ]

    where σ is the sigmoid function.

    Args:
        emulator:  Loaded ParrotEmulator instance.
        band_idx:  Integer indices selecting JADES bands from the full output.
        obs_flux:  Observed fluxes, shape (N_jades,) in nJy.
        flux_err:  Per-band 1-sigma uncertainties, shape (N_jades,) in nJy.

    Returns:
        log_posterior: Callable ``theta (N_params,) -> scalar``.
    """
    lows = jnp.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=jnp.float32)
    highs = jnp.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=jnp.float32)

    # Likelihood normalisation constant (theta-independent)
    n_bands = obs_flux.shape[0]
    log_norm = -0.5 * n_bands * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(flux_err))

    def log_posterior(theta: jnp.ndarray) -> jnp.ndarray:
        # Unconstrained → physical via sigmoid
        sps_params = lows + (highs - lows) * jax.nn.sigmoid(theta)  # (N_params,)

        # Emulator: full band output → extract JADES subset
        pred_flux = emulator.predict(sps_params[None, :])[0][band_idx]  # (N_jades,)

        # Normalised Gaussian log-likelihood
        log_like = log_norm - 0.5 * jnp.sum(((obs_flux - pred_flux) / flux_err) ** 2)

        # Log-Jacobian: enforces uniform prior in physical space
        log_jacobian = jnp.sum(
            jnp.log(highs - lows) + jax.nn.log_sigmoid(theta) + jax.nn.log_sigmoid(-theta)
        )

        return log_like + log_jacobian

    return log_posterior


# ---------------------------------------------------------------------------
# Step 5: Pathfinder warm-start
# ---------------------------------------------------------------------------


def warm_start(
    log_posterior_fn,
    theta_init: jnp.ndarray,
    rng_key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run Pathfinder to find an approximate MAP and diagonal mass-matrix.

    Args:
        log_posterior_fn: Pure-JAX log-posterior callable.
        theta_init: Starting point for L-BFGS, shape (N_params,).
        rng_key: JAX PRNG key.

    Returns:
        theta_map:       MAP-like position, shape (N_params,).
        inv_mass_matrix: Diagonal inverse mass-matrix estimate, shape (N_params,).
    """
    from arachne.inference.mclmc_sampler import run_pathfinder

    print(f"\nRunning Pathfinder (d={theta_init.shape[0]})...")
    theta_map, inv_mass = run_pathfinder(log_posterior_fn, theta_init, rng_key)
    print(f"  log_posterior at MAP: {float(log_posterior_fn(theta_map)):.2f}")
    return theta_map, inv_mass


# ---------------------------------------------------------------------------
# Step 6: NUTS sampling
# ---------------------------------------------------------------------------


def run_nuts(
    log_posterior_fn,
    theta_init: jnp.ndarray,
    rng_key: jnp.ndarray,
    n_warmup: int = 300,
    n_samples: int = 1000,
) -> np.ndarray:
    """Run BlackJAX NUTS with window adaptation.

    Mirrors NUTSSampler.run but accepts a raw log_posterior callable rather
    than a ForwardModel, keeping this script free of spatial/image machinery.

    Args:
        log_posterior_fn: Pure-JAX log-posterior callable.
        theta_init: Starting position, shape (N_params,).
        rng_key: JAX PRNG key.
        n_warmup: Number of dual-averaging warmup steps.
        n_samples: Number of posterior samples to draw after warmup.

    Returns:
        Posterior samples, shape (n_samples, N_params).
    """
    import blackjax

    logpost = jax.jit(log_posterior_fn)
    print(f"\nRunning NUTS ({n_warmup} warmup + {n_samples} samples)...")

    warmup = blackjax.window_adaptation(blackjax.nuts, logpost, target_acceptance_rate=0.8)
    rng_key, warmup_key = jax.random.split(rng_key)
    (state, params), _ = warmup.run(warmup_key, theta_init, n_warmup)
    print(f"  Warmup done. Step size: {float(params['step_size']):.4g}")

    nuts_kernel = blackjax.nuts(logpost, **params)

    def one_step(carry, rng_k):
        state, info = nuts_kernel.step(rng_k, carry)
        return state, (state.position, info)

    rng_key, sample_key = jax.random.split(rng_key)
    _, (samples, infos) = jax.lax.scan(one_step, state, jax.random.split(sample_key, n_samples))

    print(f"  Mean acceptance rate: {float(jnp.mean(infos.acceptance_rate)):.3f}")
    return np.array(samples)


# ---------------------------------------------------------------------------
# Step 7: Report recovery
# ---------------------------------------------------------------------------


def report_recovery(samples: np.ndarray, true_phys: np.ndarray) -> None:
    """Decode posterior samples and print a recovery table.

    Args:
        samples:   Unconstrained posterior samples, shape (n_samples, N_params).
        true_phys: True physical parameter values, shape (N_params,).
    """
    lows = np.array([PARAM_BOUNDS[p][0] for p in SPS_PARAM_NAMES], dtype=np.float64)
    highs = np.array([PARAM_BOUNDS[p][1] for p in SPS_PARAM_NAMES], dtype=np.float64)
    # Sigmoid in float64 to avoid overflow on large |theta| values
    phys_samples = lows + (highs - lows) / (1.0 + np.exp(-samples.astype(np.float64)))

    print("\nParameter recovery:")
    print(f"  {'Parameter':<22} {'Truth':>9} {'Median':>9} {'−1σ':>8} {'+1σ':>8} {'Bias/σ':>7}")
    print("  " + "─" * 65)
    all_ok = True
    for i, name in enumerate(SPS_PARAM_NAMES):
        truth = float(true_phys[i])
        med = float(np.median(phys_samples[:, i]))
        lo16 = float(np.percentile(phys_samples[:, i], 16))
        hi84 = float(np.percentile(phys_samples[:, i], 84))
        sigma = (hi84 - lo16) / 2.0
        bias_sigma = (med - truth) / sigma if sigma > 0 else float("nan")
        ok = np.isfinite(bias_sigma) and abs(bias_sigma) < 3
        flag = "  ✓" if ok else "  ✗ (>3σ)"
        all_ok = all_ok and ok
        print(
            f"  {name:<22} {truth:>9.4f} {med:>9.4f} "
            f"{lo16 - med:>8.4f} {hi84 - med:>8.4f} {bias_sigma:>7.2f}{flag}"
        )

    print()
    if all_ok:
        print("  All parameters recovered within 3σ. ✓")
    else:
        print("  WARNING: one or more parameters show >3σ bias.")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------


def save_results(
    samples: np.ndarray,
    obs_flux: np.ndarray,
    flux_err: np.ndarray,
    true_phys: np.ndarray,
) -> None:
    """Save mock SED and posterior samples to disk.

    Args:
        samples:   Unconstrained posterior samples, shape (n_samples, N_params).
        obs_flux:  Observed fluxes, shape (N_jades,).
        flux_err:  Flux uncertainties, shape (N_jades,).
        true_phys: True physical SPS params, shape (N_params,).
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUTDIR / "mock_sed.npz",
        obs_flux=obs_flux,
        flux_err=flux_err,
        true_params=true_phys,
        filter_codes=np.array(JADES_BANDS, dtype="S"),
        param_names=np.array(SPS_PARAM_NAMES, dtype="S"),
    )

    with h5py.File(OUTDIR / "samples.hdf5", "w") as f:
        f.create_dataset("theta_unconstrained", data=samples, compression="gzip")
        f.attrs["param_names"] = [s.encode() for s in SPS_PARAM_NAMES]
        f.attrs["filter_codes"] = [s.encode() for s in JADES_BANDS]
        f.attrs["n_samples"] = samples.shape[0]

    print(f"Saved: {OUTDIR / 'mock_sed.npz'}")
    print(f"Saved: {OUTDIR / 'samples.hdf5'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the full SED fitting pipeline."""
    parser = argparse.ArgumentParser(
        description="Fit a single integrated galaxy SED with NUTS + Pathfinder."
    )
    parser.add_argument(
        "--emulator",
        default=str(DEFAULT_EMULATOR),
        help=f"Path to ParrotEmulator checkpoint (.eqx). Default: {DEFAULT_EMULATOR}",
    )
    parser.add_argument("--n-warmup", type=int, default=300, help="NUTS warmup steps.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Posterior samples.")
    parser.add_argument("--seed", type=int, default=0, help="JAX PRNG seed.")
    args = parser.parse_args()

    emulator_path = Path(args.emulator)
    if not emulator_path.exists():
        raise FileNotFoundError(f"Emulator not found: {emulator_path}")

    print("=== Single-galaxy integrated SED fitting ===")

    # 1 & 2. Load emulator + resolve JADES band indices
    emulator, band_idx = load_emulator_and_band_indices(emulator_path)

    # 3. Generate mock SED
    obs_flux, flux_err, true_phys = make_mock_sed(emulator, band_idx)

    # 4. Build log-posterior (Gaussian likelihood + Jacobian)
    log_posterior_fn = make_log_posterior(emulator, band_idx, obs_flux, flux_err)

    # Sanity check: finite value and non-zero gradient at theta=0
    theta0 = jnp.zeros(len(SPS_PARAM_NAMES), dtype=jnp.float32)
    lp0 = float(log_posterior_fn(theta0))
    grad_norm = float(jnp.linalg.norm(jax.grad(log_posterior_fn)(theta0)))
    print(f"\nSanity check at theta=0: log_posterior={lp0:.2f}, grad_norm={grad_norm:.4f}")
    assert np.isfinite(lp0), "log_posterior not finite at theta=0"

    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, pf_key = jax.random.split(rng_key)

    # 5. Pathfinder warm-start
    theta_map, _ = warm_start(log_posterior_fn, theta0, pf_key)

    # 6. NUTS
    samples = run_nuts(log_posterior_fn, theta_map, rng_key, args.n_warmup, args.n_samples)

    # 7. Save + report
    save_results(samples, np.array(obs_flux), np.array(flux_err), true_phys)
    report_recovery(samples, true_phys)

    print(f"\nDone. Outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
