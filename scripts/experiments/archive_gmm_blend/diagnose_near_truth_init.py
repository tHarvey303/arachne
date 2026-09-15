#!/usr/bin/env python3
"""Archived: near-truth-init NUTS check for the old parameter-blending GMM demo.

Confirm: a small random jitter around the injected truth (matching
fit_mock_jwst.py's own established precedent) gives a healthy, moving NUTS
chain and good recovery -- as opposed to blind neutral-start optimization,
which repeatedly fails to find the true basin in this 32-dim landscape.
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

rng = np.random.default_rng(1)
theta_jitter = jnp.asarray(
    theta_true + rng.normal(0, 0.1, size=theta_true.shape), dtype=jnp.float32
)
lp_jitter = float(fwd.log_posterior(theta_jitter))
print(f"-log_posterior(truth) = {-float(fwd.log_posterior(jnp.asarray(theta_true))):.2f}")
print(f"-log_posterior(truth + N(0,0.1) jitter) = {-lp_jitter:.2f}")

print("\n=== NUTS from jittered truth (50 warmup + 100 samples) ===")
logpost = jax.jit(fwd.log_posterior)
warmup = blackjax.window_adaptation(blackjax.nuts, logpost, target_acceptance_rate=0.8)
rng_key = jax.random.PRNGKey(0)
rng_key, warmup_key = jax.random.split(rng_key)
(state, params), warmup_info = warmup.run(warmup_key, theta_jitter, 50)
print(f"  Final adapted step_size: {params.get('step_size'):.6e}")

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
print(
    f"  Fraction of samples identical to first: "
    f"{np.mean(np.all(np.isclose(samples_np, samples_np[0]), axis=1)):.3f}"
)
print(f"  Sample spread (std), first 10 dims: {samples_np[:, :10].std(axis=0)}")

n_components = spatial_model.n_components
decoded_samples = np.array(jax.vmap(lambda t: d.decode_component_sps(t, n_components))(samples))
truth_sps = np.array(d.decode_component_sps(jnp.asarray(theta_true), n_components))
print("\n=== Recovery ===")
for k in range(n_components):
    print(f"\n  component {k}:")
    for j, pname in enumerate(d.FREE_SPS_PARAM_NAMES):
        med = np.median(decoded_samples[:, k, j])
        lo16 = np.percentile(decoded_samples[:, k, j], 16)
        hi84 = np.percentile(decoded_samples[:, k, j], 84)
        truth_val = float(truth_sps[k, j])
        within = lo16 <= truth_val <= hi84
        mark = "OK" if within else "MISS"
        print(
            f"    [{mark}] {pname:20s}: truth={truth_val:7.3f}  "
            f"recovered={med:7.3f} [{lo16:7.3f},{hi84:7.3f}]"
        )

print("\nDONE")
