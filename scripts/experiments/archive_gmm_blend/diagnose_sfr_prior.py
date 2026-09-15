#!/usr/bin/env python3
"""Archived: Student-t SFR-ratio prior test for the old parameter-blending GMM demo.

Test whether adding the catalog pipeline's Student-t SFR-ratio prior
(missing from GaussianMixtureSpatialModel's flat prior) fixes the trap.

fit_catalogue.py's DEFAULT_PRIORS uses studentt(df=2, loc=0, scale=0.3) on
each logsfr_ratio_i specifically because these are weakly constrained by
photometry alone and prone to running to their bounds under a flat prior --
exactly the pathology observed here (logsfr_ratio -> +10, dust_bump -> 5.0
in every previous MAP-finding attempt). GaussianMixtureSpatialModel's own
log_prior is flat/uniform on all SPS params. This adds the SAME Student-t
penalty as an extra term on top of fwd.log_posterior for MAP-finding.
"""

import math
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

n_sps = len(d.FREE_SPS_PARAM_NAMES)
per_comp = 5 + n_sps
n_components = spatial_model.n_components

SFR_IDX = [d.FREE_SPS_PARAM_NAMES.index(f"logsfr_ratio_{i}") for i in range(5)]
DF, LOC, SCALE = 2.0, 0.0, 0.3
_C = (
    math.lgamma(0.5 * (DF + 1.0))
    - math.lgamma(0.5 * DF)
    - 0.5 * math.log(DF * math.pi)
    - math.log(SCALE)
)


def studentt_logpdf(x):
    """Student-t log-density matching the catalogue SFR-ratio prior."""
    return _C - 0.5 * (DF + 1.0) * jnp.log1p(((x - LOC) / SCALE) ** 2 / DF)


def sfr_prior_penalty(theta):
    """Student-t SFR-ratio prior penalty summed over components.

    Sum of studentt(df=2,loc=0,scale=0.3) log-density on each component's
    physical logsfr_ratio_i, matching fit_catalogue.py's DEFAULT_PRIORS.
    """
    total = 0.0
    for k in range(n_components):
        for j in SFR_IDX:
            raw = theta[k * per_comp + 5 + j]
            lo, hi = d.FREE_PARAM_BOUNDS[d.FREE_SPS_PARAM_NAMES[j]]
            phys = lo + (hi - lo) * jax.nn.sigmoid(raw)
            total = total + studentt_logpdf(phys)
    return total


def log_posterior_with_sfr_prior(theta):
    """Forward-model log posterior plus the Student-t SFR-ratio prior."""
    return fwd.log_posterior(theta) + sfr_prior_penalty(theta)


def adam_run(logpost_fn, theta_init, n_steps=500, lr=0.1, clip_norm=100.0):
    """Run Adam on ``-logpost_fn`` from ``theta_init``; return the final theta and loss."""
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
    for i in range(n_steps):
        theta, opt_state, loss = step(theta, opt_state)
        if i == 0 or (i + 1) % 100 == 0:
            print(f"    step {i + 1}/{n_steps}: -logpost(+sfr_prior)={float(loss):.2f}")
    return theta


theta0 = d.neutral_initial_theta(spatial_model.n_params, n_components)
print("\n=== Adam with SFR-ratio Student-t prior, neutral centered start ===")
theta_adam = adam_run(log_posterior_with_sfr_prior, theta0, n_steps=500)

print("\n=== L-BFGS refinement (still with sfr prior) ===")
import jaxopt


def loss_fn(t):
    """Negative log posterior including the SFR-ratio prior."""
    return -log_posterior_with_sfr_prior(t)


lbfgs = jaxopt.LBFGS(fun=loss_fn, maxiter=300, tol=1e-6)
theta_refined, state = lbfgs.run(init_params=theta_adam)
print(
    f"  -logpost(+sfr_prior): {float(loss_fn(theta_adam)):.2f} -> {float(state.value):.2f}  "
    f"(|grad|={float(jnp.linalg.norm(state.grad)):.2f})"
)

true_loss = -float(fwd.log_posterior(theta_refined))
print(f"\n  Pure -log_posterior (no sfr prior) at this theta: {true_loss:.2f}")
print(
    "  Truth reference: -log_posterior(truth) = "
    f"{-float(fwd.log_posterior(jnp.asarray(theta_true))):.2f}"
)
true_grad_norm = float(jnp.linalg.norm(jax.grad(fwd.log_posterior)(theta_refined)))
print(f"  |grad of pure log_posterior| at this theta: {true_grad_norm:.2f}")

decoded = np.array(d.decode_component_sps(theta_refined, n_components))
truth_sps = np.array(d.decode_component_sps(jnp.asarray(theta_true), n_components))
for k in range(n_components):
    print(f"\n  component {k}:")
    for j, pname in enumerate(d.FREE_SPS_PARAM_NAMES):
        print(f"    {pname:20s}: truth={truth_sps[k, j]:7.3f}  found={decoded[k, j]:7.3f}")

np.save(_ARACHNE / "scripts/experiments/best_sfrprior_theta.npy", np.array(theta_refined))
print("\nSaved to scripts/experiments/best_sfrprior_theta.npy")
print("DONE")
