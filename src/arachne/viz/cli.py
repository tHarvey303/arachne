"""``arachne-plots``: write the standard figures of a results bundle.

Usage::

    arachne-plots bundle.h5 --outdir figs [--format png|pdf|svg] [--dpi 150]
                  [--regions core,segment]
    python -m arachne.viz bundle.h5 --outdir figs
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    p = argparse.ArgumentParser(
        prog="arachne-plots",
        description="Write the standard static figures (SEDs, maps, images, diagnostics, "
        "summary page) of an arachne results bundle.",
    )
    p.add_argument("bundle", help="path to a results bundle (.h5)")
    p.add_argument("--outdir", default="figs", help="output directory (default: figs)")
    p.add_argument(
        "--format", dest="fmt", default="png", choices=("png", "pdf", "svg"), help="file format"
    )
    p.add_argument("--dpi", type=int, default=150, help="raster resolution (default: 150)")
    p.add_argument(
        "--regions",
        default=None,
        help="comma-separated bundle region names for the region SEDs (default: all)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")

    from ..results.reader import load_bundle
    from .summary import save_all

    regions = [r.strip() for r in args.regions.split(",") if r.strip()] if args.regions else None
    bundle = load_bundle(args.bundle)
    paths = save_all(bundle, args.outdir, fmt=args.fmt, dpi=args.dpi, regions=regions)
    for path in paths:
        print(path)
    print(f"wrote {len(paths)} figures to {args.outdir}")
    return 0
