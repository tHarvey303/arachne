"""``python -m arachne.gui bundle.h5``: run the viewer (CPU-only JAX, no GPU preallocation)."""

import os

# arachne/__init__ imports JAX eagerly: keep the viewer off the GPU before anything imports it.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from arachne.gui.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
