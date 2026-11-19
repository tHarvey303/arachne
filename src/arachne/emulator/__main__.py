"""Inspect and annotate emulator checkpoints.

Usage::

    python -m arachne.emulator describe CKPT
    python -m arachne.emulator annotate CKPT --library LIB.hdf5 [-o OUT]
    python -m arachne.emulator annotate CKPT --bounds bounds.json [-o OUT]
    python -m arachne.emulator annotate CKPT --band-errors perband_fractional_error.json
    python -m arachne.emulator annotate CKPT --simulator-library LIB.hdf5 [--simulator-preset v6]
    python -m arachne.emulator check-simulator CKPT [--grid-dir DIR]

``annotate`` records, in a ``ParrotEmulatorV2`` checkpoint that lacks them, the
training domain of every input (``--library`` or ``--bounds``), the per-band
emulator error table (``--band-errors``) and the recipe of the SPS model the
library was generated with (``--simulator-library``; see
:mod:`arachne.postprocess`).  The weights are not changed.  ``check-simulator``
rebuilds that model with Synthesizer and compares it with the reference library
rows stored in the recipe; it exits 1 unless they agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _library_metadata(library: Path, param_names: list[str]) -> dict:
    """Min/max and units of ``param_names`` over the finite rows of a synference library."""
    import h5py
    import numpy as np

    with h5py.File(library, "r") as f:
        lib_p = [str(x) for x in f.attrs["ParameterNames"]]
        lib_u = [str(x) for x in f.attrs.get("ParameterUnits", [])]
        rows = f["Grid/Parameters"][()]
    missing = [p for p in param_names if p not in lib_p]
    if missing:
        raise SystemExit(f"{library.name} lacks the emulator's inputs {missing}")
    ok = np.all(np.isfinite(rows), axis=0)
    bounds = {
        p: [float(rows[lib_p.index(p)][ok].min()), float(rows[lib_p.index(p)][ok].max())]
        for p in param_names
    }
    units = {p: lib_u[lib_p.index(p)] for p in param_names} if len(lib_u) == len(lib_p) else {}
    return {"param_bounds": bounds, "param_units": units, "library": library.name}


def _check_simulator(args) -> int:
    from arachne.postprocess.recipe import recipe_from_checkpoint
    from arachne.postprocess.simulator import check_recipe

    recipe = recipe_from_checkpoint(args.checkpoint)
    if recipe is None:
        raise SystemExit(f"{args.checkpoint} carries no simulator recipe (annotate it first)")
    report = check_recipe(recipe, grid_dir=args.grid_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        w = report["in_window"]
        print(
            f"{report['status'].upper()}: {report['n_rows']} library rows, Synthesizer "
            f"{report['synthesizer_version']}, grid {report['grid']}"
        )
        lo, hi = report["window_um"]
        print(
            f"  rest {lo}-{hi} um: median |dF/F| {w['median']:.2e}, 90th percentile "
            f"{w['p90']:.2e}, max {w['max']:.2e} ({w['n']} fluxes)"
        )
        for key, label in (("blueward", f"< {lo} um"), ("redward", f"> {hi} um")):
            if report[key].get("n"):
                side = report[key]
                print(f"  rest {label}: median {side['median']:.2e}, max {side['max']:.2e}")
        for name, diff in report["derived"].items():
            print(f"  {name}: {diff}")
    return 0 if report["status"] == "exact" else 1


def main(argv=None) -> int:
    """Run the ``describe`` / ``annotate`` command line."""
    parser = argparse.ArgumentParser(prog="python -m arachne.emulator", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    describe = sub.add_parser("describe", help="print a checkpoint's inputs, domain and bands")
    describe.add_argument("checkpoint", type=Path)
    describe.add_argument("--bands", action="store_true", help="also list the band names")
    annotate = sub.add_parser("annotate", help="record the training domain in a V2 checkpoint")
    annotate.add_argument("checkpoint", type=Path)
    source = annotate.add_mutually_exclusive_group()
    source.add_argument("--library", type=Path, help="synference HDF5 library it was trained on")
    source.add_argument("--bounds", type=Path, help='JSON file {"name": [lo, hi], ...}')
    annotate.add_argument(
        "--band-errors",
        type=Path,
        help="per-band emulator error table (JSON from validate_parrot_emulator.py)",
    )
    annotate.add_argument(
        "--simulator-library",
        type=Path,
        help="synference HDF5 library whose simulator recipe (and reference rows) to embed",
    )
    annotate.add_argument(
        "--simulator-preset", help="recipe preset (default: guessed from the library file name)"
    )
    annotate.add_argument("-o", "--output", type=Path, help="default: overwrite the checkpoint")
    check = sub.add_parser("check-simulator", help="check the embedded simulator recipe")
    check.add_argument("checkpoint", type=Path)
    check.add_argument("--grid-dir", type=Path, help="directory of the SPS grid")
    check.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)

    if args.command == "check-simulator":
        return _check_simulator(args)

    from arachne.emulator.io import load_emulator

    emulator = load_emulator(args.checkpoint)
    if args.command == "describe":
        print(f"{args.checkpoint}: {type(emulator).__name__}")
        print(emulator.spec.describe())
        if args.bands:
            print("bands: " + ", ".join(emulator.band_names))
        return 0

    if not hasattr(emulator, "with_metadata"):
        raise SystemExit(f"{type(emulator).__name__} checkpoints cannot be annotated")
    if not (args.library or args.bounds or args.band_errors or args.simulator_library):
        parser.error("annotate needs --library, --bounds, --band-errors or --simulator-library")
    metadata = {}
    if args.library:
        metadata = _library_metadata(args.library, list(emulator.param_names))
    elif args.bounds:
        metadata = {"param_bounds": json.loads(args.bounds.read_text())}
    if args.band_errors:
        metadata["band_errors"] = json.loads(args.band_errors.read_text())
    if args.simulator_library:
        from arachne.postprocess.recipe import recipe_from_library

        metadata["simulator"] = recipe_from_library(
            args.simulator_library,
            preset=args.simulator_preset,
            band_names=list(emulator.band_names),
        )
    annotated = emulator.with_metadata(**metadata)
    annotated.save(args.output or args.checkpoint)
    print(annotated.spec.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
