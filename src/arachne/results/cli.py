"""``arachne-bundle``: inspect and validate results bundles (``python -m arachne.results``).

Usage::

    arachne-bundle info results.h5 [--json]
    arachne-bundle validate results.h5 [more.h5 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .reader import load_bundle
from .schema import validate


def bundle_info(path: str | Path) -> dict[str, Any]:
    """Summary of a bundle as a JSON-serialisable dict."""
    b = load_bundle(path)
    info: dict[str, Any] = {
        "path": str(path),
        "size_mib": round(Path(path).stat().st_size / 2**20, 2),
        "title": b.title,
        "schema_version": b.schema_version,
        "posterior_source": b.posterior_source,
        "n_draws": b.n_draws,
        "degraded": b.degraded,
        "spatial_model": b.spatial_model,
        "image_shape": list(b.image_shape),
        "pixel_scale_arcsec": b.pixel_scale,
        "redshift": b.redshift,
        "bands": b.short_names,
        "map_names": b.map_names,
        "param_stats": b.available_param_stats(),
        "scalar_names": b.scalar_names,
        "fixed_params": b.fixed_params,
        "regions": {name: int(r.n_pix) for name, r in b.regions.items()},
        "has_intrinsic": b.has_intrinsic,
        "has_truth": b.has_truth,
        "components": None,
        "chi2_red_per_band": [round(float(x), 3) for x in b.chi2_red_per_band()],
        "provenance": {k: v for k, v in b.provenance.items() if not isinstance(v, np.ndarray)},
    }
    if b.components is not None:
        c = b.components
        info["components"] = {"n": c.n_components, "profiles": c.profile_names}
    return info


def _print_info(info: dict[str, Any]) -> None:
    width = max(len(k) for k in info)
    for key, val in info.items():
        if key == "provenance":
            continue
        print(f"{key:<{width}}  {val}")
    print("provenance:")
    for key, val in info["provenance"].items():
        print(f"  {key}: {val}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Arguments (default ``sys.argv[1:]``).

    Returns:
        Exit code: 0 on success, 1 if any bundle is invalid.
    """
    ap = argparse.ArgumentParser(prog="arachne-bundle", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p_info = sub.add_parser("info", help="summarise a bundle")
    p_info.add_argument("path")
    p_info.add_argument("--json", action="store_true", help="print JSON")
    p_val = sub.add_parser("validate", help="check bundle structure and readability")
    p_val.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)

    if args.command == "info":
        info = bundle_info(args.path)
        if args.json:
            print(json.dumps(info, indent=2, default=str))
        else:
            _print_info(info)
        return 0

    status = 0
    for path in args.paths:
        problems = validate(path)
        if not problems:
            try:
                load_bundle(path)
            except Exception as err:  # noqa: BLE001 - report any read failure
                problems = [f"reader failed: {err}"]
        if problems:
            status = 1
            print(f"INVALID {path}")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"OK {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
