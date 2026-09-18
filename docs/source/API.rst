API Reference
=============

Data
----

.. autoclass:: arachne.ObservationCube
   :members:
   :undoc-members:

.. autoclass:: arachne.PSFModel
   :members:
   :undoc-members:

Multi-resolution observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every band on its own pixel grid, tied together by a tangent-plane affine about one
reference sky position.  ``(dy, dx)`` offsets are in arcsec with ``dy`` towards North
and ``dx`` towards East.

.. autoclass:: arachne.BandImage
   :members:
   :undoc-members:

.. autoclass:: arachne.MultiResolutionObservation
   :members:
   :undoc-members:

.. automodule:: arachne.data.multires
   :members: canonical_band_name, tangent_plane_affine, pixel_scale_from_affine

Flux units
^^^^^^^^^^

Everything inside arachne is nanoJansky (variances nJy\ :sup:`2`).

.. automodule:: arachne.data.units
   :members: parse_bunit, flux_scale_to_nJy, bunit_scale_to_nJy, zeropoint_scale_to_nJy,
             flux_to_nJy, variance_to_nJy2, weight_to_variance

Archive clients
^^^^^^^^^^^^^^^

.. automodule:: arachne.data.dja
   :members: fetch_dja_cutout, load_dja_cutout, fetch_dja_native_cutout,
             query_dja_assoc_mosaic, fetch_dja_mosaic_file, dja_filter_names,
             band_names_from_dja_filters, server_reachable

.. automodule:: arachne.data.jades
   :members: download_jades_dr4_specz, load_jades_dr4_specz, select_targets

Emulator
--------

.. autoclass:: arachne.SPSEmulator
   :members:
   :undoc-members:

.. autoclass:: arachne.ParrotEmulatorV2
   :members:
   :undoc-members:

.. autofunction:: arachne.load_emulator

.. autoclass:: arachne.ParrotEmulator
   :members:
   :undoc-members:

.. autoclass:: arachne.SPSMLPEmulator
   :members:
   :undoc-members:

.. autoclass:: arachne.JAXFlowEmulator
   :members:
   :undoc-members:

   .. deprecated:: 0.1.0
      Prefer :class:`~arachne.ParrotEmulatorV2` (via :func:`~arachne.load_emulator`) for all new work.

Spatial Models
--------------

.. autoclass:: arachne.SpatialModel
   :members:
   :undoc-members:

.. autoclass:: arachne.AdditiveComponentModel
   :members:
   :undoc-members:

.. autoclass:: arachne.FreeFormPixelMap
   :members:
   :undoc-members:

.. autoclass:: arachne.GaussianMixtureSpatialModel
   :members:
   :undoc-members:

   .. warning::
      Blends SPS *parameters* per pixel, not light; use
      :class:`~arachne.AdditiveComponentModel` for bulge/disk decompositions.

Light profiles
^^^^^^^^^^^^^^

The shape of one additive component: a normalised surface brightness evaluated on
arbitrary coordinates, with its own prior and sampler.

.. autoclass:: arachne.Profile
   :members:
   :undoc-members:

.. autoclass:: arachne.GaussianProfile
   :members:
   :undoc-members:

.. autoclass:: arachne.SersicProfile
   :members:
   :undoc-members:

.. autoclass:: arachne.PointSourceProfile
   :members:
   :undoc-members:

.. autofunction:: arachne.get_profile

.. autofunction:: arachne.render_on_grid

.. autofunction:: arachne.spatial.profiles.log_render_on_grid

.. autofunction:: arachne.sersic_b

.. autodata:: arachne.spatial.profiles.PROFILES

PSF Convolution
---------------

.. autoclass:: arachne.PSFConvolver
   :members:
   :undoc-members:

Priors
------

Per-parameter prior specifications (``{"dist": "studentt", "df": 2, ...}``):

.. automodule:: arachne.priors.specs
   :members: DEFAULT_PRIORS, SUPPORTED_DISTS, resolve_prior_specs, prior_config_template,
             build_log_prior, build_component_log_prior, sigmoid_log_jacobian, log_prior_1d,
             validate_prior_spec

.. autoclass:: arachne.GradientPenaltyPrior
   :members:
   :undoc-members:

.. autoclass:: arachne.TotalVariationPrior
   :members:
   :undoc-members:

.. autoclass:: arachne.IndependentUniformPrior
   :members:
   :undoc-members:

.. autoclass:: arachne.LogNormalPrior
   :members:
   :undoc-members:

Likelihood
----------

.. autoclass:: arachne.GaussianLikelihood
   :members:
   :undoc-members:

Forward Model
-------------

.. autoclass:: arachne.ForwardModel
   :members:
   :undoc-members:

.. autoclass:: arachne.NuisanceModel
   :members:
   :undoc-members:

.. autoclass:: arachne.MultiResolutionForwardModel
   :members:
   :undoc-members:

Inference
---------

Initialisation
^^^^^^^^^^^^^^

.. autofunction:: arachne.image_moments

.. autofunction:: arachne.blind_initial_theta

.. autofunction:: arachne.blind_initial_full_theta

.. autofunction:: arachne.reference_band_index

.. autofunction:: arachne.solve_component_masses

.. autofunction:: arachne.find_map

.. autofunction:: arachne.multistart_map

.. autoclass:: arachne.MAPResult
   :members:
   :undoc-members:

Samplers
^^^^^^^^

.. autoclass:: arachne.NSSSampler
   :members:
   :undoc-members:

.. autoclass:: arachne.NSSResult
   :members:
   :undoc-members:

.. autoclass:: arachne.NUTSSampler
   :members:
   :undoc-members:

.. autoclass:: arachne.NUTSResult
   :members:
   :undoc-members:

.. autoclass:: arachne.MCLMCSampler
   :members:
   :undoc-members:

.. autofunction:: arachne.run_pathfinder

Laplace approximation and whitened sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Resolved posteriors are badly conditioned in raw ``theta``; these helpers build the Laplace
covariance at the MAP and sample the whitened coordinates ``theta = mean + L z``.

.. automodule:: arachne.inference.laplace
   :members: make_hessian_fn, hessian_neg_log_post, laplace_covariance, WhitenedLogDensity,
             laplace_whitening, run_whitened_nuts

Batched fitting
^^^^^^^^^^^^^^^

One vmapped XLA program over many equal-shape cutouts.

.. autoclass:: arachne.BatchedForwardModel
   :members:
   :undoc-members:

.. autofunction:: arachne.batched_blind_initial_theta

.. autofunction:: arachne.batched_find_map

.. autofunction:: arachne.batched_multistart_map

.. autofunction:: arachne.batched_nuts

.. autofunction:: arachne.fit_batch_nss

.. autoclass:: arachne.BatchedMAPResult
   :members:
   :undoc-members:

.. autoclass:: arachne.BatchedNUTSResult
   :members:
   :undoc-members:

Convergence diagnostics
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: arachne.inference.diagnostics
   :members: split_rhat, ess, chain_movement, summarise_chains

Model comparison
^^^^^^^^^^^^^^^^

.. autofunction:: arachne.compare_n_components

.. autofunction:: arachne.bayes_factor_table

.. autoclass:: arachne.ModelComparisonRow
   :members:
   :undoc-members:

Posterior predictive checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: arachne.inference.posterior_predictive
   :members: model_image_samples, component_image_samples, residual_summary,
             predictive_bands, chi2_reduced, chi2_reduced_samples, is_multiresolution,
             n_model_params
