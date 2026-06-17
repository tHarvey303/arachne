# arachne

**Image-level forward modelling of galaxy populations.**

arachne enables spatially-resolved galaxy SED fitting by forward-modelling the full
multi-band image, including PSF convolution, and using GPU-accelerated gradient-based
inference (NUTS via BlackJAX) to explore the high-dimensional posterior.

## Overview

Traditional spatially-resolved SED fitting approaches either ignore PSF effects (pixel-by-pixel)
or are computationally prohibitive (full spectral cube fitting). arachne takes a forward
modelling approach:

1. **Spatial model** — parameterise how stellar population (SPS) parameters vary across the galaxy
   (free-form pixel map or Gaussian Mixture Model)
2. **SPS emulator** — a fast native JAX/Equinox MLP trained directly on a synference model library,
   predicting per-pixel photometry from SPS parameters
3. **PSF convolution** — FFT-based differentiable convolution produces the predicted image
4. **Inference** — BlackJAX NUTS samples the full posterior using automatic differentiation

The entire pipeline is implemented in pure JAX, making it end-to-end differentiable and
JIT-compilable to GPU.

## Emulators

arachne provides two native JAX/Equinox emulators, both trained directly from the HDF5
model library produced by synference with no PyTorch dependency at inference time.

### ParrotEmulator (recommended)

A Parrot-style MLP ([Mathews et al. 2023](https://arxiv.org/abs/2302.05560)) with GELU
activations (tanh approximation) and an arsinh-magnitude output transform.  The arsinh
transform handles near-zero fluxes (e.g. Lyman-break dropouts) gracefully, which log-space
cannot.  Training exactly follows the paper: RMSE loss, output normalisation with std=1 (equal
per-filter weighting), and a 3-step independent NAdam schedule (lr 1e-3→1e-4→1e-5, batch 1000,
patience 20 per step) where each step uses a fresh random 5 % validation split and the best
weights carry into the next step.

```bash
# Train from a synference HDF5 library (recommended):
python scripts/train_parrot_emulator.py \
    --library galaxy_library.h5 \
    --output outputs/emulators/parrot.eqx \
    --epochs 1000

# Validate against held-out data:
python scripts/validate_parrot_emulator.py \
    --emulator outputs/emulators/parrot.eqx \
    --library galaxy_library.h5 \
    --output-dir outputs/validation/
```

### SPSMLPEmulator

Uses the [Speculator](https://arxiv.org/abs/1911.11778) architecture (Alsing et al. 2020)
— fully-connected layers with learnable self-gating activations, trained in log10-flux space.

Both emulators produce `.eqx` checkpoints — native Equinox pytrees that are differentiable,
JIT-compilable, and loadable with no PyTorch dependency.

## Installation

```bash
git clone https://github.com/tHarvey303/arachne
cd arachne
pip install -e ".[dev,test]"

# JAX with GPU support (adjust cuda version to match your system):
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# synference (provides the HDF5 model libraries used to train the emulator):
pip install -e /path/to/synference
```

## Quick Start

### Step 1 — Train the emulator

Train once from a synference HDF5 model library:

```python
from arachne import ParrotEmulator

emulator = ParrotEmulator.from_synference_library(
    library_path="galaxy_library.h5",
    param_names=["log_stellar_mass", "log_age", "log_metallicity", "tau_v"],
    band_names=["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"],
)
emulator.save("emulator.eqx")
```

Or via the training script (equivalent, with more CLI options):

```bash
python scripts/train_parrot_emulator.py \
    --library galaxy_library.h5 \
    --output emulator.eqx \
    --params log_stellar_mass log_age log_metallicity tau_v \
    --bands JWST/NIRCam.F115W JWST/NIRCam.F200W JWST/NIRCam.F277W
```

### Step 2 — Fit a galaxy

```python
import jax
import jax.numpy as jnp
from arachne import (
    ObservationCube, PSFModel, ParrotEmulator,
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

# Load the trained emulator
emulator = ParrotEmulator.load(
    "emulator.eqx",
    param_names=["log_stellar_mass", "log_age", "log_metallicity", "tau_v"],
    band_names=["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"],
)

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

## Data Flow

```
synference HDF5 library
       │
       ▼ ParrotEmulator.from_synference_library()
emulator.eqx  ←  GELU MLP, trained in arsinh-magnitude space
       │
       │  (inference time)
       ▼
theta (n_params,)  ← BlackJAX NUTS
       │
       ▼ SpatialModel.decode()
pixel_params (H*W, N_sps_params)
       │
       ▼ ParrotEmulator.predict()   ← pure JAX, no PyTorch
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

## Spatial Models

| Model | Parameters | Best for |
|---|---|---|
| `GaussianMixtureSpatialModel` | `K × (5 + N_sps)` | Structured galaxies; fast inference |
| `FreeFormPixelMap` | `H × W × N_sps` | Maximum flexibility; requires GPU |

## Samplers

| Sampler | Best for |
|---|---|
| `NUTSSampler` | GMM spatial models (low-d, ~tens of params) |
| `MCLMCSampler` | FreeFormPixelMap (high-d, ~45k–67k params); O(1) gradient evals per effective sample vs NUTS's O(d^{1/4}) |

`run_pathfinder` provides a fast L-BFGS warm-start (MAP position + diagonal inverse-mass-matrix
estimate) that can be passed to either sampler to skip expensive warmup.

## License

GPLv3 — see [LICENSE.md](LICENSE.md).
