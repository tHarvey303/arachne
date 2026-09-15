#!/usr/bin/env python3
"""Smoke test: official PyPI blackjax==1.6.2 against arachne's NSS pipeline.

Mocks one galaxy directly (no catalogue needed) and drives it through the
exact same build_nss_fns / run_nss_galaxy path used by fit_catalogue_nss.py,
to confirm the swap from the handley-lab fork to the official blackjax
release still produces sane results.
"""

import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import blackjax

print("blackjax version:", blackjax.__version__)
print("jax devices:", jax.devices())

from fit_catalogue import (
    DEFAULT_PRIORS,
    PARAM_BOUNDS,
    SPS_PARAM_NAMES,
    load_emulator_and_band_indices,
    make_log_prior_fn,
)
from fit_catalogue_nss import (
    build_nss_fns,
    make_log_likelihood_physical,
    run_nss_galaxy,
)

EMULATOR = Path(__file__).parent.parent / "outputs/emulators/parrot_emulator_v2.eqx"

band_names = [
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F444W",
]

emu, band_idx = load_emulator_and_band_indices(EMULATOR, band_names)

log_prior_fn = make_log_prior_fn(DEFAULT_PRIORS)
log_like_fn = make_log_likelihood_physical(emu, band_idx, min_frac_err=0.1)

# Mock observation at a fiducial parameter point (middle of bounds).
rng = jax.random.PRNGKey(0)
true_params = np.array(
    [0.5 * (PARAM_BOUNDS[p][0] + PARAM_BOUNDS[p][1]) for p in SPS_PARAM_NAMES],
    dtype=np.float32,
)
true_flux = np.array(emu.predict(true_params[None, :])[0][band_idx])
flux_err = np.maximum(0.05 * np.abs(true_flux), 1e-3).astype(np.float32)
rng, nkey = jax.random.split(rng)
obs_flux = (true_flux + flux_err * np.array(jax.random.normal(nkey, true_flux.shape))).astype(
    np.float32
)

print(f"n_bands={len(band_idx)}  n_params={len(SPS_PARAM_NAMES)}")

nss_init_fn, nss_step_fn = build_nss_fns(
    emu,
    band_idx,
    log_prior_fn,
    log_like_fn,
    num_inner_steps=24,
    num_delete=50,
)

t0 = time.perf_counter()
samples, logz, logz_err, ess, n_steps, n_dead, elapsed = run_nss_galaxy(
    rng,
    obs_flux,
    flux_err,
    nss_init_fn,
    nss_step_fn,
    num_live=500,
    termination=-3.0,
    n_samples_out=500,
    verbose=True,
)
compile_and_run = time.perf_counter() - t0

print("\n=== NSS smoke test result ===")
print(f"logZ = {logz:.3f} +/- {logz_err:.3f}")
print(f"ESS = {ess:.1f}")
print(f"n_steps = {n_steps}  n_dead = {n_dead}")
print(f"sampling time = {elapsed:.1f}s  (incl. compile: {compile_and_run:.1f}s)")
print(f"samples shape = {samples.shape}  finite frac = {np.isfinite(samples).mean():.4f}")

post_mean = samples.mean(axis=0)
print("\nparam        true      post_mean")
for i, p in enumerate(SPS_PARAM_NAMES):
    print(f"{p:12s} {true_params[i]:8.3f}  {post_mean[i]:8.3f}")

assert np.isfinite(logz), "logZ is not finite"
assert np.isfinite(samples).all(), "non-finite posterior samples"
assert ess > 1.0, "ESS collapsed"
print("\nSMOKE TEST PASSED")
