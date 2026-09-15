Quickstart
==========

This guide walks through a blind bulge + disk fit of a galaxy image with the
2-component :class:`~arachne.AdditiveComponentModel`.  The README's
"Image-Level Forward Modelling" section has the same workflow with more commentary.

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

Use a trained :class:`~arachne.ParrotEmulatorV2` checkpoint (see the README for
training from a synference library):

.. code-block:: python

   from arachne import load_emulator

   emulator = load_emulator("outputs/emulators/parrot_emulator_v2.eqx")
   names = list(emulator.param_names)
   bounds = {...}  # {name: (lo, hi)} = the emulator training domain, for every parameter

Set up the spatial model and forward model
------------------------------------------

Light is additive: each component carries the emulator SED of its *own* SPS parameters
(including its own total stellar mass) times a unit-sum Gaussian profile.  Redshift is
fixed here (use ``shared_param_names=["redshift"]`` to fit one common value instead).

.. code-block:: python

   from arachne import (
       AdditiveComponentModel, ForwardModel,
   )
   from arachne.priors import build_component_log_prior, resolve_prior_specs

   free = [p for p in names if p != "redshift"]
   specs = resolve_prior_specs(free, None, bounds)           # DEFAULT_PRIORS + overrides
   sps_log_prior = build_component_log_prior(
       names, specs, bounds, fixed_param_names=["redshift"]
   )

   H, W = obs.image_shape
   spatial_model = AdditiveComponentModel(
       n_components=2,
       emulator_param_names=names,
       param_bounds=bounds,
       image_shape=(H, W),
       fixed_params={"redshift": 2.0},
       mass_param="log_mass",
       sps_log_prior=sps_log_prior,
   )

   forward_model = ForwardModel.build(
       obs=obs, psf_model=psf, spatial_model=spatial_model, emulator=emulator,
       model_error_frac=0.05,  # fractional model-error floor for emulator systematics
   )
   obs_jax = forward_model.observation

Blind initialisation and sampling
---------------------------------

.. code-block:: python

   import jax
   from arachne import NSSSampler, NUTSSampler, blind_initial_theta, multistart_map

   theta0 = blind_initial_theta(spatial_model, obs_jax)
   map_result = multistart_map(forward_model, theta0, archetypes=[{}, {"Av": 0.3}, {"Av": 2.0}])

   # Nested slice sampling from the prior: samples + log-evidence
   result = NSSSampler(forward_model, num_live=500).run(jax.random.PRNGKey(0))
   print(result.logZ, result.logZ_err)

   # ...or NUTS started from the blind MAP
   # result = NUTSSampler(forward_model, n_warmup=500, n_samples=1000).run(
   #     map_result.theta, jax.random.PRNGKey(0))

   samples = jax.vmap(spatial_model.order_components_by_size)(result.samples)
   mu, sigma, rho, sps_phys = jax.vmap(spatial_model.component_params)(samples)
   result.to_hdf5("posterior.h5")
   param_maps = result.get_parameter_map(image_shape=(H, W))  # summary maps, plotting only
