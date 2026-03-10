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

## Emulator

arachne's emulator (`SPSMLPEmulator`) uses the [Speculator](https://arxiv.org/abs/1911.11778)
architecture (Alsing et al. 2020) — a stack of fully-connected layers with learnable self-gating
activations well-suited to the smooth mappings produced by SPS codes. It is trained directly
from the HDF5 model library produced by synference, with no PyTorch dependency at inference time.

```bash
# Train once from a synference library:
python examples/train_emulator.py \
    --library galaxy_library.h5 \
    --output emulator.eqx \
    --param-names log_stellar_mass log_age log_metallicity tau_v \
    --band-names JWST/NIRCam.F115W JWST/NIRCam.F200W JWST/NIRCam.F277W
```

The resulting `.eqx` checkpoint is a native Equinox pytree — differentiable, JIT-compilable,
and loadable with no PyTorch dependency.

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
from arachne import SPSMLPEmulator

emulator = SPSMLPEmulator.from_synference_library(
    library_path="galaxy_library.h5",
    param_names=["log_stellar_mass", "log_age", "log_metallicity", "tau_v"],
    band_names=["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"],
    hidden_sizes=[256, 256, 256],
    n_epochs=300,
)
emulator.save("emulator.eqx")
```

### Step 2 — Fit a galaxy

```python
import jax
import jax.numpy as jnp
from arachne import (
    ObservationCube, PSFModel, SPSMLPEmulator,
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
emulator = SPSMLPEmulator.load(
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
result = sampler.run(jnp.zeros(spatial_model.n_params), jax.random.PRNGKey(0))

# Posterior parameter maps: {param_name: (n_percentiles, H, W)}
param_maps = result.get_parameter_map(image_shape=(H, W))
result.to_hdf5("posterior.h5")
```

## Data Flow

```
synference HDF5 library
       │
       ▼ SPSMLPEmulator.from_synference_library()
emulator.eqx  ←  Alsing-layer MLP, trained in log10 flux space
       │
       │  (inference time)
       ▼
theta (n_params,)  ← BlackJAX NUTS
       │
       ▼ SpatialModel.decode()
pixel_params (H*W, N_sps_params)
       │
       ▼ SPSMLPEmulator.predict()   ← pure JAX, no PyTorch
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

## License

GPLv3 — see [LICENSE.md](LICENSE.md).
