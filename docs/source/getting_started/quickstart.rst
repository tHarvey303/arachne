Quickstart
==========

This guide walks through fitting a galaxy image with a 2-component GMM spatial model.

Load observations
-----------------

.. code-block:: python

   from arachne import ObservationCube, PSFModel

   obs = ObservationCube.from_fits(
       flux_paths=["f115w.fits", "f200w.fits", "f277w.fits"],
       variance_paths=["f115w_var.fits", "f200w_var.fits", "f277w_var.fits"],
       band_names=["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"],
       pixel_scale=0.031,
   )

   psf = PSFModel.from_fits({
       "JWST/NIRCam.F115W": "psf_f115w.fits",
       "JWST/NIRCam.F200W": "psf_f200w.fits",
       "JWST/NIRCam.F277W": "psf_f277w.fits",
   })

Load the emulator
-----------------

The emulator must be trained in the **forward direction** (params → photometry):

.. code-block:: python

   from arachne import JAXFlowEmulator

   emulator = JAXFlowEmulator.from_synference_checkpoint(
       "path/to/forward_checkpoint.pkl",
       param_names=["log_stellar_mass", "log_age", "log_metallicity", "tau_v"],
       band_names=["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"],
       direction="forward",
   )

Set up spatial model and run inference
---------------------------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from arachne import GaussianMixtureSpatialModel, ForwardModel, NUTSSampler

   H, W = obs.image_shape
   spatial_model = GaussianMixtureSpatialModel(
       n_components=2,
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

   # Save and inspect results
   result.to_hdf5("posterior.h5")
   param_maps = result.get_parameter_map(image_shape=(H, W))
   # param_maps["log_stellar_mass"] has shape (3, H, W) for [16th, 50th, 84th] percentiles
