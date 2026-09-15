#!/usr/bin/env python3
"""Diagnostics for the frozen-NUTS-chain problem in demo_resolved_sed_fitting.py.

Step 0 from the Opus consult: evaluate log_posterior/grad at the known
injected truth, run NUTS starting AT the truth, and instrument the warmup
trace (step_size / acceptance / divergence per step) to see exactly where
and how the adaptation collapses.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_ARACHNE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ARACHNE / "examples"))
sys.path.insert(0, str(_ARACHNE / "scripts"))

import blackjax  # noqa: E402
import demo_resolved_sed_fitting as d  # noqa: E402

from arachne.data.observation import ObservationCube  # noqa: E402
from arachne.forward_model.pipeline import ForwardModel  # noqa: E402

print("=== Building pipeline (real noisy observation) ===")
emulator, psf_model = d.load_emulator_and_psf()
spatial_model = d.build_spatial_model()
theta_true = d.build_truth_theta()

placeholder_fwd = d._placeholder_forward_model(emulator, psf_model, spatial_model)
true_image = np.array(placeholder_fwd._model_image(jnp.asarray(theta_true)))
flux_noisy, variance = d.add_noise(true_image, seed=0)
obs = ObservationCube(
    flux=flux_noisy,
    variance=variance,
    mask=np.ones_like(flux_noisy),
    band_names=d.BAND_NAMES,
    pixel_scale=d.PIXEL_SCALE,
)
fwd = ForwardModel.build(
    obs=obs, psf_model=psf_model, spatial_model=spatial_model, emulator=emulator
)

print("\n=== Step 0a: log_posterior and grad at the INJECTED TRUTH ===")
theta_true_jax = jnp.asarray(theta_true, dtype=jnp.float32)
lp_truth = fwd.log_posterior(theta_true_jax)
grad_truth = jax.grad(fwd.log_posterior)(theta_true_jax)
grad_norm_truth = float(jnp.linalg.norm(grad_truth))
print(
    f"  log_posterior(truth) = {float(lp_truth):.2f}  "
    f"(i.e. -log_posterior = {-float(lp_truth):.2f})"
)
print(f"  |grad| at truth = {grad_norm_truth:.4f}")
n_data = flux_noisy.size
print(
    f"  N_data (pixels x bands) = {n_data}; a good fit should give "
    f"-log_posterior ~ 0.5*N_data + const ~ {0.5 * n_data:.0f} + const"
)

print("\n=== Step 0b/0c: NUTS window_adaptation starting AT the truth, instrumented ===")
logpost = jax.jit(fwd.log_posterior)
warmup = blackjax.window_adaptation(
    blackjax.nuts,
    logpost,
    target_acceptance_rate=0.8,
)
rng_key = jax.random.PRNGKey(0)
rng_key, warmup_key = jax.random.split(rng_key)
(state, params), warmup_info = warmup.run(warmup_key, theta_true_jax, 50)

print(f"  Final adapted step_size: {params.get('step_size'):.6e}")
imm = params.get("inverse_mass_matrix")
print(
    f"  Final inverse_mass_matrix: shape={np.asarray(imm).shape}  "
    f"min={float(jnp.min(imm)):.6e}  max={float(jnp.max(imm)):.6e}"
)

print(
    "\n  warmup_info fields:",
    warmup_info._fields if hasattr(warmup_info, "_fields") else type(warmup_info),
)

# Try to extract a per-step trace of whatever's available.
for field in getattr(warmup_info, "_fields", []):
    val = getattr(warmup_info, field)
    try:
        arr = np.asarray(val)
        print(f"  {field}: shape={arr.shape} dtype={arr.dtype}")
    except Exception:
        print(f"  {field}: {type(val)} (not directly array-able)")

# info.info is often the inner MCMC step info (per-step acceptance/divergence)
inner = getattr(warmup_info, "info", None)
if inner is not None and hasattr(inner, "_fields"):
    print("\n  inner info fields:", inner._fields)
    for field in inner._fields:
        val = getattr(inner, field)
        try:
            arr = np.asarray(val)
            print(f"    {field}: shape={arr.shape}")
            if arr.ndim == 1 and arr.shape[0] <= 60:
                print(f"      values: {arr}")
            elif arr.ndim == 1:
                print(f"      first 10: {arr[:10]}  last 10: {arr[-10:]}")
        except Exception as e:
            print(f"    {field}: not array-able ({e})")

print("\n=== Sampling from this (truth-started) warmup state ===")
nuts_kernel = blackjax.nuts(logpost, max_num_doublings=5, **params)


def one_step(carry, rng_key):
    """One NUTS step for ``jax.lax.scan``."""
    state, info = nuts_kernel.step(rng_key, carry)
    return state, (state.position, info)


rng_key, sample_key = jax.random.split(rng_key)
sample_keys = jax.random.split(sample_key, 100)
final_state, (samples, infos) = jax.lax.scan(one_step, state, sample_keys)
samples_np = np.array(samples)
print(f"  Mean acceptance rate: {float(jnp.mean(infos.acceptance_rate)):.4f}")
print(f"  Sample spread (std) per-dim, first 10 dims: {samples_np[:, :10].std(axis=0)}")
print(
    f"  Total movement from truth: mean |theta_sample - theta_true| = "
    f"{np.abs(samples_np - theta_true[None, :]).mean():.6f}"
)
print(
    f"  Fraction of samples identical to first sample: "
    f"{np.mean(np.all(np.isclose(samples_np, samples_np[0]), axis=1)):.3f}"
)

print("\nDONE")
