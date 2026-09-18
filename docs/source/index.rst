arachne
=======

**Image-level forward modelling of galaxy populations.**

arachne enables spatially-resolved galaxy SED fitting by forward-modelling the full
multi-band image as a sum of additive light components — Gaussian, Sérsic or point-source
profiles, each carrying its own emulated SED — including PSF convolution and optional
instrumental nuisances (sky, sub-pixel registration, noise rescaling), and sampling the
posterior with GPU-accelerated BlackJAX samplers: nested slice sampling (evidence +
multimodality), NUTS, or MCLMC.

Bands may share one pixel grid (:class:`~arachne.ForwardModel`) or each keep its own WCS grid
with no resampling of the data (:class:`~arachne.MultiResolutionForwardModel`), and a whole
sample of equal-shape cutouts can be fitted in one vmapped program
(:class:`~arachne.BatchedForwardModel`).  Data-layer clients fetch cutouts from the DAWN JWST
Archive and spectroscopic targets from JADES DR4.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started/index
   API

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
