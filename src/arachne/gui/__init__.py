"""Interactive viewer of ``arachne.results`` bundles (bokeh; Panel optional in Jupyter).

``ArachneViewer`` and ``view`` need bokeh (``pip install 'arachne[gui]'``).  They are imported
lazily so that the numpy-only :mod:`arachne.gui.state` works without bokeh.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["ArachneViewer", "view", "require_bokeh"]

_INSTALL_HINT = (
    "The arachne viewer needs bokeh: install it with  pip install 'arachne[gui]'  "
    "(or 'arachne[gui-jupyter]' for JupyterLab support via Panel)"
)


def require_bokeh() -> None:
    """Raise a friendly ImportError when bokeh is missing."""
    try:
        import bokeh  # noqa: F401
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err


def __getattr__(name: str) -> Any:
    """Import the bokeh-backed viewer on first access."""
    if name in ("ArachneViewer", "view"):
        require_bokeh()
        app = importlib.import_module(f"{__name__}.app")
        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
