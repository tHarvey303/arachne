# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**arachne** is a Python package for forward modelling of galaxy populations at the image level,
enabling spatially-resolved galaxy SED fitting. It is a standalone package that uses `synference`
as a dependency for the underlying SPS emulators (normalising flows trained on Synthesizer mock
photometry).

The core scientific workflow:
1. **Observation loading** — load multi-band FITS images, PSF models per band, noise/variance maps
2. **Spatial model setup** — describe how stellar population parameters vary across the galaxy
   (free-form per-pixel map or Gaussian Mixture Model)
3. **Emulator loading** — export a trained synference normalising flow to JAX/Equinox (frozen weights)
4. **Forward modelling** — spatial params → SPS emulator → PSF convolution → predicted image
5. **Inference** — NUTS or MCLMC via BlackJAX samples the full posterior over all spatial parameters

The entire forward model pipeline (steps 3–5) is implemented in pure JAX, making it
end-to-end differentiable and JIT-compilable to GPU. BlackJAX NUTS calls `jax.grad` of
`ForwardModel.log_posterior` directly.

## Development Setup

```bash
pip install -e ".[dev,test]"
pre-commit install
# synference must be installed separately:
pip install -e /path/to/synference
# JAX with GPU support (match your CUDA version):
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Commands

```bash
pytest                              # Full test suite (runs on CPU, no GPU needed)
pytest tests/test_forward_model.py  # Single file
pytest -m "not gpu"                 # Skip GPU-requiring tests in CI
ruff check --fix src/ scripts/ tests/  # Lint + auto-fix (include scripts/)
ruff format src/ scripts/ tests/    # Format
cd docs && make html                # Build docs

# Training / validation scripts:
python scripts/train_parrot_emulator.py --library <lib.hdf5> --output outputs/emulators/parrot.eqx
python scripts/validate_parrot_emulator.py --emulator outputs/emulators/parrot.eqx --library <lib.hdf5>
```

## Architecture

All source lives in `src/arachne/`. The package exports the key public API from `__init__.py`.

| Module | Responsibility |
|---|---|
| `data/observation.py` | `ObservationCube` — multi-band FITS image container |
| `data/psf.py` | `PSFModel` — per-band PSF kernels, padded for FFT |
| `emulator/parrot_emulator.py` | `ParrotEmulator` — **recommended emulator**; Mathews et al. 2023 GELU (tanh approx) MLP with arsinh-magnitude output and equal per-filter RMSE loss; 3-step NAdam schedule (1e-3→1e-4→1e-5, batch=1000, patience=20/step); trained from synference HDF5 via `scripts/train_parrot_emulator.py` |
| `emulator/jax_mlp_emulator.py` | `SPSMLPEmulator` — Alsing et al. 2020 Speculator MLP in log10-flux space; fully JAX-native |
| `emulator/jax_emulator.py` | `JAXFlowEmulator` — **legacy**; exports synference normalising flow weights to JAX/Equinox; use only if a lampe checkpoint is the only available artefact |
| `spatial/pixel_map.py` | `FreeFormPixelMap` — per-pixel SPS parameters with L2 smoothness prior |
| `spatial/gmm.py` | `GaussianMixtureSpatialModel` — K-component GMM with per-component SPS parameters |
| `psf/convolution.py` | `PSFConvolver` — FFT-based differentiable PSF convolution, pre-computed PSF FFTs |
| `likelihood/gaussian.py` | `GaussianLikelihood` — weighted chi-squared log-likelihood |
| `priors/spatial.py` | `GradientPenaltyPrior`, `TotalVariationPrior` — spatial smoothness priors |
| `forward_model/pipeline.py` | `ForwardModel` — composes all components into a single `log_posterior(theta)` |
| `inference/nuts_sampler.py` | `NUTSSampler` — BlackJAX NUTS with window adaptation, `jax.lax.scan` sampling loop |
| `inference/mclmc_sampler.py` | `MCLMCSampler` — BlackJAX MCLMC; preferred for high-d `FreeFormPixelMap` (O(1) grad evals/sample). `run_pathfinder` provides L-BFGS warm-start |

## Key Design Principles

### 1. Differentiability is sacred
Every function in the forward model pipeline (`decode`, `predict`, `convolve`, `log_likelihood`,
`log_prior`) must be differentiable with `jax.grad`. Never use numpy/scipy inside the hot path.
Never use Python control flow that branches on JAX-traced values.

### 2. PyTorch ↔ JAX bridge: weight export only
DLPack and torch2jax cannot propagate JAX gradients across the PyTorch boundary, which would
break NUTS. The correct approach is one-time weight export: load the synference checkpoint,
convert all tensors to numpy, reconstruct an Equinox pytree. At inference time, no PyTorch
code runs. See `JAXFlowEmulator.from_synference_checkpoint()`.

### 3. `log_posterior` must be a pure function
No logging, no file I/O, no mutable state inside `ForwardModel.log_posterior()`. It must be
side-effect free for `jax.jit` to work correctly.

### 4. GPU memory discipline
Use `jnp.float32` throughout (not float64) unless convergence issues require it.
Set `max_num_doublings=5` in `blackjax.nuts` to cap trajectory length.
For pixel map mode on 75×75 images: theta has ~67,500 params; profile memory before large runs.

### 5. Spatial model contract
All `SpatialModel` subclasses implement exactly two methods called by `ForwardModel`:
- `decode(theta, image_shape) -> (H*W, N_sps_params)` — unconstrained → physical params
- `log_prior(theta) -> scalar` — differentiable log-prior

New spatial models are drop-in replacements if they satisfy this interface.

### 6. Emulator: use ParrotEmulator for new work
The recommended emulator is `ParrotEmulator` (Mathews et al. 2023) — GELU (tanh approx) MLP
with arsinh-magnitude output. Training uses RMSE loss with output std=1 (equal per-filter
weighting) and a 3-step NAdam schedule (1e-3→1e-4→1e-5, batch=1000, patience=20/step), exactly
matching the paper. The arsinh transform handles near-zero fluxes (Lyman-break dropouts, ~50%
of blue-band samples at high-z) without divergence. Train via:

```bash
python scripts/train_parrot_emulator.py \
    --library galaxy_library.h5 \
    --output outputs/emulators/parrot.eqx \
    --epochs 1000
```

`SPSMLPEmulator` (Alsing et al. 2020 Speculator, log10-flux space) remains available and
is tested. `JAXFlowEmulator` is legacy-only (lampe checkpoint input); do not use for new work.
Emulator checkpoints (`.eqx`) and validation outputs go in `outputs/` (gitignored).

### 7. Sampler: use MCLMCSampler for FreeFormPixelMap
`NUTSSampler` is adequate for `GaussianMixtureSpatialModel` (low-d). For `FreeFormPixelMap`
(d~45k–67k), use `MCLMCSampler` — persistent momentum, no tree doubling, O(1) gradient evals
per effective sample vs NUTS's O(d^{1/4}). Pair with `run_pathfinder` for a fast L-BFGS
warm-start that skips expensive warmup.

## Conventions

- **Linter/formatter**: `ruff` (line-length 100, Google docstrings). `__init__.py` excluded.
  Run `ruff check --fix` before committing.
- **Notebooks**: committed clean (outputs stripped) via `nb-clean` pre-commit hook.
- **Input format**: FITS for observations and PSFs.
- **Output format**: HDF5 for posterior samples and parameter maps.
- **JAX arrays everywhere** inside the forward model. Convert to numpy only at I/O boundaries.
- **Units**: log10 stellar mass (Msun), log10 age (yr), log10 metallicity (Z/Zsun), tau_v (mag),
  fluxes in nJy. Document units in every docstring. Follow synference parameter conventions.
- Tests run on CPU with 16×16 synthetic arrays. GPU-requiring tests marked `@pytest.mark.gpu`.
