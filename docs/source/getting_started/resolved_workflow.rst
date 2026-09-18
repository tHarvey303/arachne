The resolved workflow
=====================

:doc:`quickstart` fits a two-component galaxy on a single pixel grid with no instrumental
nuisances.  This page adds the three pieces a real fit usually needs — light profiles,
instrumental nuisances and multi-resolution data — and then the checks that decide whether
the answer can be believed.  The README section "Image-Level Forward Modelling" carries the
same material with more prose.

Choosing light profiles
-----------------------

Each component's shape is a :class:`~arachne.Profile`, chosen per component:

.. list-table::
   :header-rows: 1
   :widths: 12 34 54

   * - ``profiles``
     - Shape block
     - Notes
   * - ``"gaussian"``
     - ``mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho`` (5)
     - The historical parametrisation.  ``sigma = exp(log_sigma)``,
       ``rho = tanh(atanh_rho)`` clipped to ±0.99.
   * - ``"sersic"``
     - the above ``+ log_n`` (6)
     - ``sigma`` is the effective radius per axis; ``n = exp(log_n)`` is softly clamped
       into ``n_bounds`` (default ``(0.3, 10)``); prior ``Normal(log 2, 0.7)`` on
       ``log_n``.  Use ``oversample >= 3``.
   * - ``"point"``
     - ``mu_y, mu_x`` (2)
     - A fixed circular Gaussian of width ``point_sigma`` (0.5 px), so after PSF
       convolution the component *is* the PSF at ``mu``.

.. code-block:: python

   from arachne import AdditiveComponentModel, SersicProfile

   spatial_model = AdditiveComponentModel(
       n_components=3,
       emulator_param_names=names,
       param_bounds=bounds,
       image_shape=(H, W),
       fixed_params={"redshift": 2.0},
       mass_param="log_mass",
       sps_log_prior=sps_log_prior,
       profiles=[SersicProfile(log_n_sd=0.3), "gaussian", "point"],
       pixel_scale=0.031,          # mu / sigma now in arcsec, not pixels
       normalisation="analytic",   # unit flux over the whole plane
       oversample=3,               # sub-samples per pixel side, for the cuspy Sersic
   )

Because profiles have different block lengths,
:meth:`~arachne.AdditiveComponentModel.split_theta` returns a *list* of blocks for a
mixed-profile model (a rectangular ``(K, ·)`` array when every component shares a profile),
and :meth:`~arachne.AdditiveComponentModel.order_components_by_size` only permutes
components that share a profile.

``normalisation="analytic"`` is required whenever the model is rendered anywhere but its own
grid (:meth:`~arachne.AdditiveComponentModel.model_image_on`, multi-resolution fitting);
``"frame"`` divides by the sum over the model's own grid and is meaningless elsewhere.

Instrumental nuisances
----------------------

:class:`~arachne.NuisanceModel` appends up to three per-band blocks to ``theta`` — a sky
pedestal (nJy/pixel), a sub-pixel ``(dy, dx)`` registration offset applied as an exact
Fourier phase ramp, and a log multiplier on the noise standard deviation — each with a
normalised zero-mean Gaussian prior:

.. code-block:: python

   from arachne import ForwardModel, NuisanceModel, blind_initial_full_theta, multistart_map

   nuisance = NuisanceModel(
       n_bands=len(band_names),
       fit_sky=True,
       fit_shifts=True,
       fit_noise_scale=True,
       shift_reference_band=0,      # REQUIRED with fit_shifts: see below
   )
   forward_model = ForwardModel.build(
       obs=obs, psf_model=psf, spatial_model=spatial_model, emulator=emulator,
       model_error_frac=0.05, nuisance=nuisance,
   )

   theta0 = blind_initial_full_theta(forward_model)
   map_result = multistart_map(forward_model, theta0, archetypes=[{}, {"Av": 2.0}])
   print(nuisance.param_names(band_names))

.. warning::
   A shift common to *every* band is exactly degenerate with moving every spatial
   component.  Always pin one band with ``shift_reference_band=i``; the free parameter
   count is then ``2 * (n_bands - 1)``.

``theta = concat(theta_spatial, theta_nuisance)``.  Use
:meth:`~arachne.ForwardModel.split_theta`,
:meth:`~arachne.ForwardModel.initial_theta_from_spatial` and
:meth:`~arachne.ForwardModel.sample_prior` rather than slicing; ``nuisance.split(theta_n)``
always returns all three blocks, zero-filled when a block is disabled.

Multi-resolution fitting
------------------------

:class:`~arachne.MultiResolutionForwardModel` fits every band on its own native pixel grid,
so the data are never resampled.  The joint log-likelihood is *exactly* the sum of the
per-band ones.

.. code-block:: python

   from arachne import MultiResolutionForwardModel, blind_initial_full_theta
   from arachne.data.dja import fetch_dja_native_cutout

   mro = fetch_dja_native_cutout(
       ra, dec, size_arcsec=6.0, filters=["f200w", "f444w"],
       psfs={"JWST/NIRCam.F200W": (kernel_f200w, 0.03),
             "JWST/NIRCam.F444W": (kernel_f444w, 0.03)},
   )
   fm = MultiResolutionForwardModel.build(
       mro, spatial_model, emulator, model_error_frac=0.05, oversample=3,
   )
   theta0 = blind_initial_full_theta(fm)
   row, col = fm.sky_to_pixel("JWST/NIRCam.F444W", dy, dx)

``n_params``, ``split_theta``, ``initial_theta_from_spatial``, ``sample_prior``,
``log_prior``, ``log_likelihood`` and ``log_posterior`` mean exactly what they do on
:class:`~arachne.ForwardModel`, so :class:`~arachne.NSSSampler`,
:class:`~arachne.NUTSSampler`, :func:`~arachne.find_map` and the posterior-predictive
helpers work unchanged.

.. note::
   The spatial model must be in arcsec mode **and** ``normalisation="analytic"``, and every
   band needs a PSF kernel sampled on its own grid.  This model works in the **sky frame**:
   ``mu_y`` is positive towards North and ``mu_x`` towards East, mirrored in ``x`` relative
   to the single-grid arcsec frame (where ``+x`` is ``+column``, usually West), and the sign
   of ``rho`` flips with the handedness.  Expect about 1.3× the cost of a single-grid fit on
   nine NIRCam bands.

When every band really does share one grid — a DJA ``thumb`` cutout, for instance — call
``load_dja_cutout(...).to_observation_cube()`` and use the plain
:class:`~arachne.ForwardModel`.

Choosing a sampler
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Situation
     - Sampler
   * - ≲ 40 parameters, evidence wanted
     - :class:`~arachne.NSSSampler` — starts from the prior, gives ``logZ``, handles
       multimodality by construction
   * - More than ~40 parameters
     - ``arachne.inference.laplace.run_whitened_nuts`` — NUTS from the polished MAP in
       whitened coordinates, with a dense metric adapted inside them
   * - :class:`~arachne.FreeFormPixelMap`
     - :class:`~arachne.MCLMCSampler`, optionally warm-started by
       :func:`~arachne.run_pathfinder`
   * - Many equal-shape cutouts
     - :func:`~arachne.batched_nuts` on a :class:`~arachne.BatchedForwardModel`

Nested sampling from the prior becomes impractical above ~40 parameters (a 62-parameter
problem took over 3 s per outer step and was still far below the MAP after 50 steps).  NUTS
with a diagonal mass matrix also fails on these posteriors: the curvature eigenvalues at the
mode span many orders of magnitude and the stiff directions are correlated, so warmup
collapses the step size and every trajectory saturates the tree-depth cap.  Sample
``theta = MAP + L z`` with ``L`` the Cholesky factor of the Laplace covariance at a properly
polished MAP — :func:`~arachne.inference.laplace.run_whitened_nuts` builds the Hessian, the
whitening and the sampler in one call and returns a ``NUTSResult`` back in ``theta``
coordinates — and still check R-hat and ESS before believing the result.

Checking the fit
----------------

.. code-block:: python

   import jax
   from arachne import residual_summary
   from arachne.inference.diagnostics import summarise_chains
   from arachne.inference.model_comparison import compare_n_components, bayes_factor_table

   samples = jax.vmap(spatial_model.order_components_by_size)(result.samples)

   print(result.summary())                    # step size, R-hat, ESS, divergences
   print(summarise_chains(result.chains, param_names)["table"])

   summary = residual_summary(forward_model, samples)
   print(summary["chi2_red"], summary["chi2_red_per_band"], summary["frac_chi_gt_3"])

   rows = compare_n_components(make_forward_model, ks=[1, 2, 3],
                               rng_key=jax.random.PRNGKey(0))
   print(bayes_factor_table(rows))

Run at least four chains and require ``rhat < 1.01`` and ``ess > 100`` per chain before
trusting posterior quantiles.  For a multi-resolution model the image-space products come
back as per-band lists rather than stacked arrays
(:func:`~arachne.inference.posterior_predictive.is_multiresolution` tells you which).

.. warning::
   ``make_forward_model(K)`` must differ *only* in the number of components — same
   observation, PSF, emulator, bounds, shape priors and nuisance setup.  The evidence
   integrates the prior, so a wider bound or an extra nuisance parameter in one model makes
   the Bayes factor measure that difference instead of the data's preference.

Fitting many galaxies at once
-----------------------------

When a sample of cutouts shares the same bands and the same ``(H, W)``, every fit is the
same XLA program with different data:

.. code-block:: python

   from arachne import (BatchedForwardModel, batched_blind_initial_theta,
                        batched_multistart_map, batched_nuts)

   bfm = BatchedForwardModel.build(
       observations=cubes, psf_models=psf, spatial_model=spatial_model,
       emulator=emulator, model_error_frac=0.05,
       fixed_params_per_galaxy={"redshift": z_spec}, galaxy_ids=ids,
   )
   maps = batched_multistart_map(bfm, batched_blind_initial_theta(bfm),
                                 archetypes=[{}, {"Av": 2.0}])
   result = batched_nuts(bfm, maps.theta, jax.random.PRNGKey(0), n_chains=4)
   print(result.summary())

Per-galaxy redshifts go through ``fixed_params_per_galaxy`` (which must name a parameter
already in ``fixed_params``), not through a tight prior.  Nested sampling cannot be batched,
so :func:`~arachne.fit_batch_nss` loops over galaxies and each one pays its own XLA compile.
