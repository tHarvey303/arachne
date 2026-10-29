"""``arachne-gui``: serve the interactive viewer for a results bundle.

Sets ``JAX_PLATFORMS=cpu`` and ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` before importing arachne
(whose ``__init__`` imports JAX), so the viewer never takes GPU memory on a compute node.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import socket
from collections.abc import Sequence

__all__ = ["main", "build_parser", "remote_recipe"]


def build_parser() -> argparse.ArgumentParser:
    """The ``arachne-gui`` argument parser."""
    p = argparse.ArgumentParser(
        prog="arachne-gui",
        description="Interactive viewer for arachne results bundles (bokeh server).",
        epilog=(
            "Remote use: run with --no-browser on the node, then on your laptop "
            "`ssh -L 5006:localhost:5006 <node>` and open http://localhost:5006/."
        ),
    )
    p.add_argument("bundle", help="results bundle (.h5)")
    p.add_argument("--compare", default=None, help="second bundle to compare (same image shape)")
    p.add_argument("--port", type=int, default=5006, help="server port (default 5006)")
    p.add_argument("--address", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.add_argument(
        "--allow-websocket-origin",
        action="append",
        default=[],
        metavar="HOST:PORT",
        help="extra allowed websocket origin (repeatable), e.g. localhost:8888 when the "
        "forwarded local port differs",
    )
    return p


def remote_recipe(port: int, host: str | None = None) -> str:
    """How to reach a viewer running on a remote node through an SSH tunnel."""
    host = host or socket.gethostname()
    return (
        "Remote use (viewer running on a node):\n"
        f"  on your laptop:  ssh -L {port}:localhost:{port} {host}\n"
        f"  then open:       http://localhost:{port}/\n"
        "  (through a jump host add  -J <gateway>; if the local port differs, restart with\n"
        "   --allow-websocket-origin localhost:<local port>)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``arachne-gui``.

    Args:
        argv: Arguments (default ``sys.argv[1:]``).

    Returns:
        Exit status.
    """
    args = build_parser().parse_args(argv)
    from . import require_bokeh

    require_bokeh()
    from .app import ArachneViewer

    viewer = ArachneViewer(args.bundle, compare=args.compare)
    print(f"loaded {viewer.bundle!r}")
    print(remote_recipe(args.port))
    viewer.serve(
        port=args.port,
        address=args.address,
        open_browser=not args.no_browser,
        allow_websocket_origin=args.allow_websocket_origin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
