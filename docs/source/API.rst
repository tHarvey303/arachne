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

   .. deprecated::
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

Inference
---------

Initialisation
^^^^^^^^^^^^^^

.. autofunction:: arachne.image_moments

.. autofunction:: arachne.blind_initial_theta

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
