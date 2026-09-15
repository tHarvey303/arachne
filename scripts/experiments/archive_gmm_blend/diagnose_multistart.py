#!/usr/bin/env python3
"""Multi-start Adam+L-BFGS sweep to escape the dust-mass degenerate trap.

Confirmed via diagnose_nuts_freeze.py: NUTS works perfectly when started at
the injected truth (healthy warmup, real posterior movement). The single
neutral-start Adam+L-BFGS warm start instead lands in a genuine dust-mass
degenerate local optimum. This script tries a handful of physically
archetypal (Av, log_mass) starting guesses per component -- NOT the true
values, just generic "low-dust/high-mass" vs "high-dust/lower-mass"
archetypes, standard multi-start practice -- and keeps whichever converges
to the lowest loss.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

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
n_params = spatial_model.n_params
n_components = spatial_model.n_components

# Archetypes: (Av, log_mass), roughly "clean/massive" vs "dusty/lighter" --
# generic multi-start seeds, not the true values.
ARCHETYPES = {
    "clean_massive": (0.3, 10.5),
    "dusty_light": (2.0, 9.0),
    "moderate": (1.0, 9.8),
}
AV_IDX = d.FREE_SPS_PARAM_NAMES.index("Av")
MASS_IDX = d.FREE_SPS_PARAM_NAMES.index("log_mass")


def make_theta(comp_archetypes: list[str]) -> jnp.ndarray:
    """Build a neutral theta with the given (Av, log_mass) archetype per component."""
    theta = d.neutral_initial_theta(n_params, n_components)
    for k, name in enumerate(comp_archetypes):
        av, mass = ARCHETYPES[name]
        av_raw = d.to_raw(av, *d.FREE_PARAM_BOUNDS["Av"])
        mass_raw = d.to_raw(mass, *d.FREE_PARAM_BOUNDS["log_mass"])
        theta = theta.at[k * per_comp + 5 + AV_IDX].set(av_raw)
        theta = theta.at[k * per_comp + 5 + MASS_IDX].set(mass_raw)
    return theta


CANDIDATES = [
    ["clean_massive", "dusty_light"],
    ["dusty_light", "clean_massive"],
    ["moderate", "moderate"],
    ["clean_massive", "clean_massive"],
    ["dusty_light", "dusty_light"],
]

results = []
for i, archetypes in enumerate(CANDIDATES):
    theta0 = make_theta(archetypes)
    print(f"\n--- Candidate {i}: {archetypes} ---")
    lp0 = float(fwd.log_posterior(theta0))
    print(f"  initial -log_posterior = {-lp0:.2f}")
    theta_adam = d.warm_start_map(fwd.log_posterior, theta0, n_steps=300, lr=0.1, clip_norm=100.0)
    lp_final = float(fwd.log_posterior(theta_adam))
    print(f"  Candidate {i} final -log_posterior = {-lp_final:.2f}")
    results.append((archetypes, theta_adam, -lp_final))

results.sort(key=lambda r: r[2])
print("\n=== Summary (sorted, best first) ===")
for archetypes, _, loss in results:
    print(f"  {archetypes}: -log_posterior = {loss:.2f}")

best_archetypes, best_theta, best_loss = results[0]
print(f"\nBest candidate: {best_archetypes}  -log_posterior={best_loss:.2f}")
print("(Truth reference: -log_posterior(truth) = 128231.23)")

# Decode best candidate's SPS params for inspection
decoded = np.array(d.decode_component_sps(best_theta, n_components))
truth_sps = np.array(d.decode_component_sps(jnp.asarray(theta_true), n_components))
for k in range(n_components):
    print(f"\n  component {k}:")
    for j, pname in enumerate(d.FREE_SPS_PARAM_NAMES):
        print(f"    {pname:20s}: truth={truth_sps[k, j]:7.3f}  best_candidate={decoded[k, j]:7.3f}")

np.save(_ARACHNE / "scripts/experiments/best_multistart_theta.npy", np.array(best_theta))
print("\nSaved best theta to scripts/experiments/best_multistart_theta.npy")
print("DONE")
