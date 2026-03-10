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

.. autoclass:: arachne.SPSMLPEmulator
   :members:
   :undoc-members:

.. autoclass:: arachne.JAXFlowEmulator
   :members:
   :undoc-members:

   .. deprecated::
      Prefer :class:`~arachne.SPSMLPEmulator` for all new work.

Spatial Models
--------------

.. autoclass:: arachne.SpatialModel
   :members:
   :undoc-members:

.. autoclass:: arachne.FreeFormPixelMap
   :members:
   :undoc-members:

.. autoclass:: arachne.GaussianMixtureSpatialModel
   :members:
   :undoc-members:

PSF Convolution
---------------

.. autoclass:: arachne.PSFConvolver
   :members:
   :undoc-members:

Priors
------

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

.. autoclass:: arachne.NUTSSampler
   :members:
   :undoc-members:

.. autoclass:: arachne.NUTSResult
   :members:
   :undoc-members:
