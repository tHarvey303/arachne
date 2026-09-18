Installation
============

Requirements
------------

- Python ≥ 3.10
- JAX (with optional CUDA support for GPU acceleration)
- ``blackjax >= 1.6.2`` (nested slice sampling ships upstream from 1.6; no fork is needed)
- astropy (FITS, WCS, cutouts), h5py, equinox, optax
- synference, to build the HDF5 model libraries the emulators are trained from

Basic install
-------------

.. code-block:: bash

   git clone https://github.com/tHarvey303/arachne
   cd arachne
   pip install -e ".[dev,test]"

JAX with GPU support
--------------------

.. code-block:: bash

   pip install "jax[cuda12]==<jaxlib version>"

.. warning::
   The CUDA plugin (``jax-cuda12-plugin``, ``jax-cuda12-pjrt``) must match ``jaxlib``
   *exactly*.  If a resolver bumps ``jaxlib`` without bumping the plugin, JAX does not raise —
   it silently falls back to CPU.  Verify on a machine that actually has a GPU::

      python -c "import jax; print(jax.devices())"   # must show CudaDevice

.. warning::
   In a venv shared between projects, do not upgrade or remove packages without checking what
   else depends on them: a ``jax``/``blackjax`` bump can drag ``numpy`` along and break
   astropy, numba and friends at runtime even when ``pip check`` is happy.

synference
----------

arachne depends on synference for the SPS model libraries used to train the emulators:

.. code-block:: bash

   pip install -e /path/to/synference

Running tests
-------------

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m pytest tests -q -p no:cacheprovider   # CPU-only, ~9 minutes
   pytest -m "not gpu"                                              # skip GPU-marked tests
   pytest                                                           # full suite (needs a GPU)

The CPU suite uses 3-band 16×16 synthetic images and tiny dummy emulators, so it needs no GPU,
no network and no checkpoint.

Building the documentation
--------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs && make html      # output in docs/build/html
