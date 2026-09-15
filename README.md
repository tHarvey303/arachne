# arachne

**JAX-native SED fitting and image-level forward modelling of galaxy populations.**

arachne trains fast neural emulators of stellar-population-synthesis (SPS) photometry and
uses them for GPU-accelerated Bayesian inference — from fitting a single integrated SED, to
fitting a catalogue of thousands of galaxies in parallel, to full spatially-resolved
image-level forward modelling with PSF convolution.

## Overview

Two use cases share the same emulator + inference core:

1. **Catalogue / single-galaxy SED fitting** (`scripts/fit_*.py`) — fit integrated photometry
   for one galaxy or an entire catalogue. GPU-batched Pathfinder+NUTS for speed, or
   gradient-free Nested Slice Sampling (NSS) for multimodal posteriors and evidence (`logZ`).
2. **Image-level forward modelling** — describe a galaxy as a sum of `K` additive light
   components (`AdditiveComponentModel`: each component is a Gaussian surface-brightness
   profile carrying the emulator SED of its *own* SPS parameters and total stellar mass), or
   as a free-form per-pixel parameter map; forward-model the full multi-band image including
   FFT-based PSF convolution; initialise *blind* from the data; and sample the posterior with
   NSS (evidence + multimodality), NUTS, or MCLMC.

The entire pipeline is pure JAX/Equinox — end-to-end differentiable, JIT-compilable to GPU,
and has no PyTorch dependency at inference time.

## Emulators

Both emulators are trained directly from the HDF5 model library produced by
[synference](https://github.com/tHarvey303/synference).

### ParrotEmulatorV2 (recommended)

The current production emulator (`arachne.ParrotEmulatorV2` / `arachne.load_emulator`), built
on three validated improvements over the original Parrot design (see the module docstring in
[`src/arachne/emulator/parrot_emulator_v2.py`](src/arachne/emulator/parrot_emulator_v2.py) for
the full rationale and ablations):

1. **Analytic mass factorisation** — photometry is exactly linear in stellar mass, so the
   network predicts flux per 10⁹ M☉ and `log_mass` is reapplied analytically at inference.
2. **Fourier redshift features** — sin/cos encodings let the network represent the sharp
   Lyman-break cutoff without extreme depth (the single largest accuracy win, ~2×).
3. **arsinh-compressed SFH-ratio inputs** — preserves the informative core of the
   Student-t-distributed `logsfr_ratio_*` inputs that plain z-scoring squashes.

```bash
# Train from a synference HDF5 library:
python scripts/train_parrot_emulator_v2.py \
    --library galaxy_library.hdf5 \
    --output outputs/emulators/parrot_emulator_v2.eqx

# Validate against held-out data:
python scripts/validate_parrot_emulator.py \
    --emulator outputs/emulators/parrot_emulator_v2.eqx \
    --library galaxy_library.hdf5 \
    --output-dir outputs/validation/
```

`load_emulator(path)` (`arachne.load_emulator`) loads either a V1 or V2 checkpoint
transparently — use it in downstream code instead of `ParrotEmulatorV2.load` directly.

The architecture-search harness that produced the V2 recipe (Optuna HPO, ablation sweeps,
scaling studies) lives in `scripts/experiments/` — see [Experiments](#experiments--training-harness) below.

### ParrotEmulator (legacy) and SPSMLPEmulator

`ParrotEmulator` (v1) is the original Parrot-style MLP
([Mathews et al. 2023](https://iopscience.iop.org/article/10.3847/1538-4357/ace720)) with
GELU activations and an arsinh-magnitude output transform; superseded by V2 above but kept for
compatibility. `SPSMLPEmulator` uses the [Speculator](https://arxiv.org/abs/1911.11778)
architecture (Alsing et al. 2020), trained in log10-flux space.

All emulators produce `.eqx` checkpoints — native Equinox pytrees, differentiable and
JIT-compilable with no PyTorch dependency.

## Installation

```bash
git clone https://github.com/tHarvey303/arachne
cd arachne
pip install -e ".[dev,test]"

# JAX with GPU support — install the CUDA plugin at the SAME version as jax/jaxlib,
# or JAX will silently fall back to CPU instead of erroring (see Gotchas below):
pip install "jax[cuda12]==<jaxlib version>"

# synference (provides the HDF5 model libraries used to train the emulator):
pip install -e /path/to/synference
```

Nested Slice Sampling requires `blackjax>=1.6.2` (nested sampling shipped upstream in
blackjax 1.6; no fork is needed any more — see [Samplers](#samplers)).

### Shared / HPC environments

If you're working in a venv shared across multiple projects (e.g. the department HPC venv),
**do not upgrade or remove packages without checking the blast radius first** — `pip`'s
resolver will happily bump transitive dependencies (numpy, jax, jaxlib, ...) that other
projects in the same venv pin to older versions. See [Gotchas](#gotchas) below for a concrete
example of this breaking the environment.

## Catalogue / Single-Galaxy SED Fitting

Three scripts share the same forward model and likelihood machinery
(`scripts/fit_catalogue.py` is the shared driver — the others import from it); the
per-parameter prior specifications they use live in the library, in
[`arachne.priors.specs`](src/arachne/priors/specs.py) (see [Priors](#priors) below):

| Script | Use for |
|---|---|
| `fit_sed.py` | One mock galaxy, minimal reference example (no catalogue needed). |
| `fit_one_galaxy.py` | One real or mock galaxy with verbose diagnostics — debug fits here before scaling up. |
| `fit_catalogue.py` | Production catalogue fitting, thousands of galaxies. |

`fit_catalogue.py` supports two inference backends via `--sampler`:

```bash
# GPU-batched Pathfinder + multi-chain NUTS (default; fast, thousands of galaxies/min)
python scripts/fit_catalogue.py catalogue.fits bands_config.json \
    --emulator outputs/emulators/parrot_emulator_v2.eqx \
    --sampler nuts --n-chains 2 --n-samples 500

# Nested Slice Sampling (blackjax.nss) — gradient-free, handles multimodal posteriors
# natively, and reports the log-evidence logZ. Runs sequentially per galaxy (~10-30x
# slower than the batched NUTS path), so use it for QA / hard cases, not full catalogues.
python scripts/fit_catalogue.py catalogue.fits bands_config.json \
    --emulator outputs/emulators/parrot_emulator_v2.eqx \
    --sampler nss --num-live 500 --num-inner-steps 24 --num-delete 50
```

Print a band-config JSON template (bands + default priors) for a given emulator:

```bash
python scripts/fit_catalogue.py --print-config-template --emulator outputs/emulators/parrot_emulator_v2.eqx
```

See `scripts/configs/*.json` for worked examples (band lists, flux units, per-parameter priors).

`diagnose_parrot_emulator.py` and `validate_parrot_emulator.py` round out the toolkit for
inspecting emulator accuracy before trusting it in a fit.

## Experiments / Training Harness

`scripts/experiments/` holds the emulator architecture-search harness used to derive the V2
recipe: `hpo_lab.py` (Optuna HPO), `emulator_lab.py` / `train_final_v2.py` (ablation training),
`bench_inference.py` (inference-speed benchmarking), `make_summary_figs.py`, and shell drivers
(`run_ablations.sh`, `run_scaling.sh`, ...). Results are written to `scripts/outputs/` (trial
JSON/npz pairs, an Optuna sqlite DB) — this directory is gitignored; only the harness code
itself is tracked. `scripts/experiments/archive_gmm_blend/` holds retired diagnostics from the
old parameter-blending demo (kept for the record; they do not run against the current code).

## Image-Level Forward Modelling

### Quick Start

A blind bulge + disk fit with the additive component model. Light is additive, so each
component gets its own full SED (including its own *total* stellar mass) and a unit-sum
Gaussian surface-brightness profile; the likelihood costs `K` emulator calls per evaluation
regardless of image size.

```python
import jax
from arachne import (
    ObservationCube, PSFModel, ForwardModel,
    AdditiveComponentModel, NSSSampler, NUTSSampler,
    blind_initial_theta, multistart_map, load_emulator,
)
from arachne.priors import build_component_log_prior, resolve_prior_specs

# Observations, PSFs and emulator
obs = ObservationCube.from_fits(
    flux_paths=["f115w.fits", "f200w.fits", "f277w.fits"],
    variance_paths=["f115w_var.fits", "f200w_var.fits", "f277w_var.fits"],
    band_names=["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"],
)
psf = PSFModel.from_fits({
    "JWST/NIRCam.F115W": "psf_f115w.fits",
    "JWST/NIRCam.F200W": "psf_f200w.fits",
    "JWST/NIRCam.F277W": "psf_f277w.fits",
})
emulator = load_emulator("outputs/emulators/parrot_emulator_v2.eqx")

# Physical bounds = the emulator's training domain (e.g. scripts/fit_catalogue.py PARAM_BOUNDS)
names = list(emulator.param_names)
bounds = {"redshift": (0.0, 12.0), "log_mass": (6.0, 12.0), "Av": (0.0, 4.0),
          "logsfr_ratio_0": (-5.0, 5.0)}  # ... one entry per emulator parameter

# Physical-space prior on the (K, N_emulator) component matrix.  DEFAULT_PRIORS gives
# Student-t(df=2, scale=0.3) on logsfr_ratio_* and uniform elsewhere; a fixed parameter
# needs no prior and a shared one is counted once.
free = [p for p in names if p != "redshift"]
specs = resolve_prior_specs(free, None, bounds)
sps_log_prior = build_component_log_prior(names, specs, bounds, fixed_param_names=["redshift"])

H, W = obs.image_shape
spatial_model = AdditiveComponentModel(
    n_components=2,                       # bulge + disk
    emulator_param_names=names,
    param_bounds=bounds,
    image_shape=(H, W),
    fixed_params={"redshift": 2.0},       # or shared_param_names=["redshift"] to fit one z
    mass_param="log_mass",                # per-component TOTAL stellar mass
    sps_log_prior=sps_log_prior,
)

# model_error_frac adds a fractional model-error floor in quadrature (a few per cent
# of emulator systematics dominate photon noise once ~1e5 pixels are summed).
forward_model = ForwardModel.build(
    obs=obs, psf_model=psf, spatial_model=spatial_model, emulator=emulator,
    model_error_frac=0.05,
)
obs_jax = forward_model.observation

# Blind initialisation: image moments -> neutral SPS values -> Adam with linear mass re-solves,
# from a few generic SPS archetypes; the best -log_posterior wins.  No truth, no zeros.
theta0 = blind_initial_theta(spatial_model, obs_jax)
map_result = multistart_map(forward_model, theta0, archetypes=[{}, {"Av": 0.3}, {"Av": 2.0}])

# Nested slice sampling: live points drawn from the prior (no starting point needed);
# returns equal-weight samples plus the evidence, so fits with different K can be compared.
result = NSSSampler(forward_model, num_live=500).run(jax.random.PRNGKey(0))
print(result.logZ, result.logZ_err, result.ess)

# ...or NUTS from the blind MAP:
# result = NUTSSampler(forward_model, n_warmup=500, n_samples=1000).run(
#     map_result.theta, jax.random.PRNGKey(0))

# Components are exchangeable: order compact-first before summarising.
samples = jax.vmap(spatial_model.order_components_by_size)(result.samples)
mu, sigma, rho, sps_phys = jax.vmap(spatial_model.component_params)(samples)  # physical per component
param_maps = result.get_parameter_map(image_shape=(H, W))  # summary maps for plotting only
result.to_hdf5("posterior.h5")
```

`theta` is a flat float32 vector of length `K * (5 + N_free) + N_shared`, laid out as `K`
component blocks `[mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho, sps_raw...]` followed by
the shared raws; every SPS value is mapped to its bounds by a sigmoid. Use
`spatial_model.split_theta` / `join_theta` / `component_params` rather than slicing by hand.

### Data Flow

```
synference HDF5 library
       │
       ▼ train_parrot_emulator_v2.py
emulator.eqx  ←  mass-factorised MLP, Fourier-z, arsinh SFH inputs
       │
       │  (inference time)
       ▼
theta (n_params,)  ← BlackJAX NSS / NUTS / MCLMC  (started from blind_initial_theta → multistart_map)
       │
       ▼ SpatialModel.model_image(theta, emulator, (H, W))
       │    AdditiveComponentModel:  K component SEDs (K, N_bands)  ←  ParrotEmulatorV2.predict
       │                              × K unit-sum profiles (K, H, W)  →  Σ_k F_k ⊗ P_k
       │    FreeFormPixelMap / GMM:   decode → pixel_params (H*W, N_sps) → predict per pixel
model_image (N_bands, H, W)
       │
       ▼ PSFConvolver (FFT, pre-computed)
convolved_image (N_bands, H, W)
       │
       ▼ GaussianLikelihood (optional fractional model-error floor)  +  SpatialModel.log_prior()
log_likelihood(theta), log_prior(theta), log_posterior(theta)  ← scalar, differentiable
       │
       ├─▶ NSS: log_prior + log_likelihood separately, live points from sample_prior → samples + logZ
       └─▶ NUTS / MCLMC: jax.grad(log_posterior) → leapfrog
```

### Spatial Models

| Model | Parameters | Best for |
|---|---|---|
| `AdditiveComponentModel` (**recommended**) | `K × (5 + N_free) + N_shared` | Bulge/disk and other multi-component decompositions. Light is additive: each component has its own SED at its own total mass times a unit-sum Gaussian profile; `K` emulator calls per likelihood. Proper prior with `sample_prior` (works with NSS); supports `fixed_params` and `shared_param_names`. |
| `GaussianMixtureSpatialModel` | `K × (5 + N_sps)` | **Caveat:** blends SPS *parameters* per pixel with softmax weights that sum to 1, so it has no surface-brightness profile — every pixel carries a full galaxy mass and a lone "disk" fills the whole frame at constant flux. Only for smooth parameter *gradients* across an already-resolved source; not for bulge/disk work. Kept for compatibility. |
| `FreeFormPixelMap` | `H × W × N_sps` | Maximum flexibility; per-pixel emulator calls, requires GPU; L2 smoothness prior. |

All models implement `decode` / `log_prior`; `model_image` (the hook `ForwardModel` calls) has
a per-pixel default and is overridden by `AdditiveComponentModel`. Note that
`AdditiveComponentModel.decode` returns a *summary* map (mass-weighted component parameters,
with log10 stellar-mass surface density in the mass column) for plotting — the likelihood
never uses it.

### Initialisation

`arachne.inference.initialisation` provides a deterministic, truth-free start for additive
models: `image_moments` (centroid + rms size from the S/N-stacked image) →
`blind_initial_theta` (all components at the centroid on a geometric size ladder, neutral SPS
values) → `find_map` / `multistart_map` (Adam on `-log_posterior`). The reason this works is
that the `K` component masses are **linear amplitudes** — flux ∝ `10**log_mass` — so given
every other parameter they are the exact solution of a `K × K` weighted least-squares system
(`solve_component_masses`). `find_map` re-solves them between Adam rounds, which collapses the
dominant mass/size/colour degeneracy far faster than gradient steps, and `multistart_map` runs
from a few generic SPS archetypes (e.g. low vs high dust) to avoid the wrong basin of the
dust/age degeneracy. Components are ordered compact-first at the end
(`order_components_by_size`) to fix the label symmetry.

### Priors

Per-parameter physical-space priors are dict specs (`{"dist": "studentt", "df": 2, "loc": 0,
"scale": 0.3}`) handled by [`arachne.priors.specs`](src/arachne/priors/specs.py):
`DEFAULT_PRIORS` (Student-t on `logsfr_ratio_*` — the emulator training grid was drawn from
it — uniform elsewhere), `resolve_prior_specs` (merge user overrides, validate against bounds),
`build_log_prior` (catalogue fits, `(N,)` vector), `build_component_log_prior` (image fits,
`(K, N_emulator)` matrix; shared parameters counted once, fixed ones ignored),
`prior_config_template` and `sigmoid_log_jacobian`. Supported `dist` values: `uniform`,
`loguniform`, `normal`, `studentt`, `halfnormal`, `exponential`, `lognormal`. The spatial
models add their own sigmoid Jacobian (uniform in *physical* space) plus Gaussians on the
shape parameters, so `AdditiveComponentModel.log_prior` is a proper density and `logZ` values
are comparable across `K`.

### Resolved demo

[`examples/demo_resolved_sed_fitting.py`](examples/demo_resolved_sed_fitting.py) is the
end-to-end reference for the image-level pipeline. It renders a two-component mock (compact,
old, quenched bulge + extended, young, dusty disk at a common fixed redshift) as a 10-band
JWST/NIRCam image with the real trained `ParrotEmulatorV2` checkpoint and real JADES empirical
PSFs, adds noise, and fits it back **blind** — the injected truth is used only to generate the
data and to grade the answer, never to start the fit (`blind_initial_theta` →
`multistart_map` → `NSSSampler` from the prior, or `NUTSSampler` from the blind MAP). It needs
a GPU node (160×160 px, 133×133 px PSF kernels) and, as written, the checkpoint under
`scripts/outputs/emulators/` and the JADES PSF directory on the cluster.

```bash
ssh gn004                                                     # any GPU node from CLAUDE.md
python examples/demo_resolved_sed_fitting.py --quick          # smoke run (num_live 100, fewer Adam steps)
python examples/demo_resolved_sed_fitting.py                  # NSS, num_live=500 (default)
python examples/demo_resolved_sed_fitting.py --sampler nuts --n-warmup 500 --n-samples 1000
# other flags: --model-err-frac 0.05  --noise-scale 1.0  --seed 0  --outdir DIR
#              --num-live N  --num-inner-steps N  --termination 1e-3
```

Outputs go to `--outdir` (default `outputs/demo/resolved_sed_fitting/`): `truth.json`,
`map.json` (blind MAP with `-log_post` vs the truth's), `posterior.hdf5` (size-ordered
samples, plus `logZ` for NSS) and figures — truth/model/residual mosaic per band, summary
SPS-parameter maps, bulge- and disk-pixel SEDs, 1-D posteriors per component, sampler
diagnostics and component radial profiles. Run its module docstring for the assumptions and
known limitations.

## Samplers

| Sampler | Best for |
|---|---|
| `NSSSampler` (`blackjax.nss`) | Image models with a proper prior (`AdditiveComponentModel`): gradient-free, live points drawn from `sample_prior` (no starting point), handles label/size/mass multimodality by construction, returns equal-weight samples plus `logZ`, `logZ_err`, ESS — the tool for comparing different `K`. Many more likelihood evaluations than NUTS; use a GPU. |
| `NUTSSampler` / `fit_catalogue.py --sampler nuts` | GPU-batched catalogue fitting; low-d image models (additive, tens of params) started from the blind MAP |
| `MCLMCSampler` | `FreeFormPixelMap` (high-d, ~45k–67k params); O(1) gradient evals per effective sample vs NUTS's O(d^{1/4}) |
| `fit_catalogue.py --sampler nss` (`blackjax.nss`) | Catalogue fits: multimodal posteriors, per-galaxy log-evidence (`logZ`); sequential, not GPU-batched |

`ForwardModel` exposes `log_likelihood`, `log_prior` and `log_posterior` separately; NSS uses
the first two, the gradient samplers the third. `run_pathfinder` provides a fast L-BFGS
warm-start (MAP position + diagonal inverse-mass-matrix estimate) that can be passed to
either gradient-based sampler to skip expensive warmup.

## Gotchas

- **`pip install`ing anything that touches `jax`/`jaxlib` can silently break GPU support.**
  JAX requires the CUDA plugin (`jax-cuda12-plugin`, `jax-cuda12-pjrt`) to match `jaxlib`'s
  version *exactly*; if a resolver bumps `jaxlib` without bumping the plugin, JAX does not
  error — it silently falls back to CPU. After any such install, verify on an actual GPU node
  (`jax.devices()` on a login node proves nothing — there's no GPU there to detect):
  ```bash
  ssh gn004   # or any node from the GPU table in CLAUDE.md
  python -c "import jax; print(jax.devices())"   # must show CudaDevice, not CpuDevice
  ```
- **A `blackjax`/`jax` upgrade can also drag `numpy` along**, breaking any package in the same
  venv with a tighter numpy pin (astropy, numba, and others have all broken this way). Check
  `pip check` and actually `import` the packages you care about — `pip check`'s static version
  comparison misses runtime breaks that don't declare a strict upper bound (astropy did not,
  but still broke on `numpy.in1d` removal).
- **Never upgrade a shared/multi-project venv without checking who else depends on it.** Pin
  versions explicitly in `pyproject.toml` rather than leaving loose lower bounds once you've
  found a known-good combination.
- **Data S/N above the emulator's accuracy makes a resolved fit over-confident.** A resolved
  fit sums tens of thousands of pixels; with photon noise alone the posterior is far narrower
  than the emulator's few-per-cent systematics. Use `GaussianLikelihood(obs,
  model_error_frac=...)` to add a fractional floor in quadrature (with the `log var` term).

## License

GPLv3 — see [LICENSE.md](LICENSE.md).
