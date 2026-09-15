arachne
=======

**Image-level forward modelling of galaxy populations.**

arachne enables spatially-resolved galaxy SED fitting by forward-modelling the full
multi-band image as a sum of additive light components (each with its own emulated SED),
including PSF convolution, and sampling the posterior with GPU-accelerated BlackJAX
samplers: nested slice sampling (evidence + multimodality), NUTS, or MCLMC.

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
