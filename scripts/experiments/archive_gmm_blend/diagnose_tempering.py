#!/usr/bin/env python3
"""Likelihood-tempering ladder to escape the dust-mass degenerate trap.

Confirmed: NUTS works fine started at the injected truth; a single Adam+
L-BFGS warm start (even multi-started over 5 archetypes) gets stuck in a
local optimum ~145,000-254,000 vs the truth's -log_posterior=128,231.
Standard fix for local optima in a peaked likelihood: anneal the data's
weight (beta) from ~0 (flat, trivially optimizable) up to 1 (full posterior)
in stages, carrying the optimizer's position forward between stages.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

_ARACHNE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ARACHNE / "examples"))
sys.path.insert(0, str(_ARACHNE / "scripts"))

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

H, W = fwd.observation.image_shape
N_bands = fwd.observation.n_bands


def tempered_log_posterior(theta, beta):
    """Log posterior with the likelihood tempered by ``beta``."""
    pixel_params = fwd.spatial_model.decode(theta, (H, W))
    pixel_fluxes = fwd.emulator.predict(pixel_params)
    convolved = fwd.convolver(pixel_fluxes.T.reshape(N_bands, H, W))
    log_like = fwd.likelihood(convolved)
    log_prior = fwd.spatial_model.log_prior_from_decoded(theta, pixel_params, (H, W))
    return beta * log_like + log_prior


def adam_stage(logpost_fn, theta_init, n_steps=200, lr=0.1, clip_norm=100.0):
    """One Adam stage on ``-logpost_fn``; return the final theta and loss."""
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(lr))
    opt_state = optimizer.init(theta_init)

    def loss_fn(t):
        return -logpost_fn(t)

    @jax.jit
    def step(theta, opt_state):
        loss, grad = jax.value_and_grad(loss_fn)(theta)
        updates, opt_state = optimizer.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss

    theta = theta_init
    for _ in range(n_steps):
        theta, opt_state, loss = step(theta, opt_state)
    return theta, float(loss)


BETAS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

theta = d.neutral_initial_theta(spatial_model.n_params, spatial_model.n_components)
print(f"Starting tempering ladder from neutral centered init, betas={BETAS}")
for beta in BETAS:
    theta, loss_tempered = adam_stage(lambda t: tempered_log_posterior(t, beta), theta, n_steps=200)
    true_loss = -float(fwd.log_posterior(theta))
    print(
        f"  beta={beta:8.4f}  tempered_loss={loss_tempered:14.2f}  "
        f"true(-log_posterior)={true_loss:14.2f}"
    )

print("\n=== Final L-BFGS refinement at beta=1 ===")
import jaxopt


def loss_fn(t):
    """Negative log posterior at beta=1."""
    return -fwd.log_posterior(t)


lbfgs = jaxopt.LBFGS(fun=loss_fn, maxiter=300, tol=1e-6)
theta_refined, state = lbfgs.run(init_params=theta)
print(
    f"  -log_posterior: {-float(fwd.log_posterior(theta)):.2f} -> {float(state.value):.2f}  "
    f"(|grad|={float(jnp.linalg.norm(state.grad)):.2f})"
)
print(
    "  Truth reference: -log_posterior(truth) = "
    f"{-float(fwd.log_posterior(jnp.asarray(theta_true))):.2f}"
)

decoded = np.array(d.decode_component_sps(theta_refined, spatial_model.n_components))
truth_sps = np.array(d.decode_component_sps(jnp.asarray(theta_true), spatial_model.n_components))
for k in range(spatial_model.n_components):
    print(f"\n  component {k}:")
    for j, pname in enumerate(d.FREE_SPS_PARAM_NAMES):
        print(f"    {pname:20s}: truth={truth_sps[k, j]:7.3f}  tempered={decoded[k, j]:7.3f}")

np.save(_ARACHNE / "scripts/experiments/best_tempered_theta.npy", np.array(theta_refined))
print("\nSaved to scripts/experiments/best_tempered_theta.npy")
print("DONE")
