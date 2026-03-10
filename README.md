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
2. **SPS emulator** — a fast neural network (normalising flow) predicts per-pixel photometry from
   SPS parameters (imported from `synference`, converted to JAX/Equinox)
3. **PSF convolution** — FFT-based differentiable convolution produces the predicted image
4. **Inference** — BlackJAX NUTS samples the full posterior using automatic differentiation

The entire pipeline is implemented in pure JAX, making it end-to-end differentiable and
JIT-compilable to GPU.

## Installation

```bash
pip install -e ".[dev,test]"
# JAX with GPU support:
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# synference (SPS emulator source):
pip install -e /path/to/synference
```

## Quick Start

```python
from arachne import ObservationCube, PSFModel, JAXFlowEmulator
from arachne import GaussianMixtureSpatialModel, ForwardModel, NUTSSampler
import jax
import jax.numpy as jnp

# Load observations
obs = ObservationCube.from_fits(
    flux_paths=["band1.fits", "band2.fits"],
    variance_paths=["band1_var.fits", "band2_var.fits"],
    band_names=["JWST/NIRCam.F115W", "JWST/NIRCam.F200W"],
)

# Load PSF models
psf = PSFModel.from_fits({"JWST/NIRCam.F115W": "psf_f115w.fits", "JWST/NIRCam.F200W": "psf_f200w.fits"})

# Load emulator (forward direction: params -> photometry)
emulator = JAXFlowEmulator.from_synference_checkpoint(
    "path/to/forward_checkpoint.pkl",
    param_names=["log_stellar_mass", "log_age", "log_metallicity", "tau_v"],
    band_names=["JWST/NIRCam.F115W", "JWST/NIRCam.F200W"],
)

# Set up spatial model
H, W = obs.flux.shape[1:]
spatial_model = GaussianMixtureSpatialModel(
    n_components=3,
    sps_param_names=emulator.param_names,
    param_bounds={"log_stellar_mass": (6.0, 12.0), "log_age": (7.0, 10.1),
                  "log_metallicity": (-2.0, 0.5), "tau_v": (0.0, 4.0)},
    image_shape=(H, W),
)

# Build forward model
forward_model = ForwardModel.build(obs=obs, psf_model=psf, spatial_model=spatial_model, emulator=emulator)

# Run NUTS inference
sampler = NUTSSampler(forward_model=forward_model, n_warmup=500, n_samples=1000)
rng_key = jax.random.PRNGKey(0)
theta_init = jnp.zeros(spatial_model.n_params)
result = sampler.run(theta_init, rng_key)

# Get posterior parameter maps
param_maps = result.get_parameter_map(image_shape=(H, W))
result.to_hdf5("posterior.h5")
```

## Data Flow

```
theta (n_params,)  ← BlackJAX NUTS
       │
       ▼ SpatialModel.decode()
pixel_params (H*W, N_sps_params)
       │
       ▼ JAXFlowEmulator.predict()
pixel_fluxes (H*W, N_bands)
       │
       ▼ reshape → model_image (N_bands, H, W)
       │
       ▼ PSFConvolver (FFT)
convolved_image (N_bands, H, W)
       │
       ▼ GaussianLikelihood + SpatialModel.log_prior()
log_posterior(theta)  ← scalar, differentiable
       │
       ▼ jax.grad → grad_theta → NUTS leapfrog
```

## License

GPLv3 — see [LICENSE.md](LICENSE.md).
