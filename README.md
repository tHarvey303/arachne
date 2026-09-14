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
2. **Image-level forward modelling** — parameterise how SPS parameters vary spatially across
   a galaxy (Gaussian Mixture Model or free-form pixel map), forward-model the full multi-band
   image including FFT-based PSF convolution, and sample the posterior with NUTS or MCLMC.

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

Three scripts share the same forward model, likelihood, and prior machinery
(`scripts/fit_catalogue.py` is the shared library — the others import from it):

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
itself is tracked.

## Image-Level Forward Modelling

### Quick Start

```python
from arachne import ParrotEmulatorV2, load_emulator

# Load the trained emulator
emulator = load_emulator("outputs/emulators/parrot_emulator_v2.eqx")
```

```python
import jax
import jax.numpy as jnp
from arachne import (
    ObservationCube, PSFModel, load_emulator,
    GaussianMixtureSpatialModel, ForwardModel, NUTSSampler,
)

# Load observations and PSFs
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

# Set up spatial model and run inference
H, W = obs.image_shape
spatial_model = GaussianMixtureSpatialModel(
    n_components=3,
    sps_param_names=emulator.param_names,
    param_bounds={
        "log_stellar_mass": (6.0, 12.0),
        "log_age": (7.0, 10.1),
        "log_metallicity": (-2.0, 0.5),
        "tau_v": (0.0, 4.0),
    },
    image_shape=(H, W),
)

forward_model = ForwardModel.build(
    obs=obs, psf_model=psf, spatial_model=spatial_model, emulator=emulator
)

sampler = NUTSSampler(forward_model=forward_model, n_warmup=500, n_samples=1000)
# For high-dimensional FreeFormPixelMap use MCLMCSampler instead (O(1) grad evals/sample):
# from arachne import MCLMCSampler, run_pathfinder
# pos, imm = run_pathfinder(forward_model.log_posterior, theta_init, key)
# sampler = MCLMCSampler(forward_model, n_warmup=1000, n_samples=500)
# result = sampler.run(pos, key, inverse_mass_matrix=imm)
result = sampler.run(jnp.zeros(spatial_model.n_params), jax.random.PRNGKey(0))

# Posterior parameter maps: {param_name: (n_percentiles, H, W)}
param_maps = result.get_parameter_map(image_shape=(H, W))
result.to_hdf5("posterior.h5")
```

### Data Flow

```
synference HDF5 library
       │
       ▼ train_parrot_emulator_v2.py
emulator.eqx  ←  mass-factorised MLP, Fourier-z, arsinh SFH inputs
       │
       │  (inference time)
       ▼
theta (n_params,)  ← BlackJAX NUTS / MCLMC / NSS
       │
       ▼ SpatialModel.decode()
pixel_params (H*W, N_sps_params)
       │
       ▼ ParrotEmulatorV2.predict()   ← pure JAX, no PyTorch
pixel_fluxes (H*W, N_bands)
       │
       ▼ reshape → model_image (N_bands, H, W)
       │
       ▼ PSFConvolver (FFT, pre-computed)
convolved_image (N_bands, H, W)
       │
       ▼ GaussianLikelihood + SpatialModel.log_prior()
log_posterior(theta)  ← scalar, differentiable
       │
       ▼ jax.grad → grad_theta → NUTS leapfrog
```

### Spatial Models

| Model | Parameters | Best for |
|---|---|---|
| `GaussianMixtureSpatialModel` | `K × (5 + N_sps)` | Structured galaxies; fast inference |
| `FreeFormPixelMap` | `H × W × N_sps` | Maximum flexibility; requires GPU |

## Samplers

| Sampler | Best for |
|---|---|
| `NUTSSampler` / `fit_catalogue.py --sampler nuts` | GPU-batched fitting; GMM spatial models (low-d, ~tens of params) |
| `MCLMCSampler` | `FreeFormPixelMap` (high-d, ~45k–67k params); O(1) gradient evals per effective sample vs NUTS's O(d^{1/4}) |
| `fit_catalogue.py --sampler nss` (`blackjax.nss`) | Gradient-free, multimodal posteriors, per-galaxy log-evidence (`logZ`); sequential, not GPU-batched |

`run_pathfinder` provides a fast L-BFGS warm-start (MAP position + diagonal inverse-mass-matrix
estimate) that can be passed to either gradient-based sampler to skip expensive warmup.

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

## License

GPLv3 — see [LICENSE.md](LICENSE.md).
