Installation
============

Requirements
------------

- Python ≥ 3.10
- JAX (with optional CUDA support for GPU acceleration)
- synference (SPS emulator source)

Basic install
-------------

.. code-block:: bash

   git clone https://github.com/arachne-project/arachne
   cd arachne
   pip install -e ".[dev,test]"

JAX with GPU support
--------------------

.. code-block:: bash

   pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Adjust ``cuda12_pip`` to match your installed CUDA version.

synference
----------

arachne depends on synference for SPS emulator checkpoints:

.. code-block:: bash

   pip install -e /path/to/synference

Running tests
-------------

.. code-block:: bash

   pytest -m "not gpu"   # CPU-only (no GPU or checkpoint required)
   pytest                # Full suite (requires GPU for marked tests)
