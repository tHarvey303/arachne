"""Inspect and annotate emulator checkpoints.

Usage::

    python -m arachne.emulator describe CKPT
    python -m arachne.emulator annotate CKPT --library LIB.hdf5 [-o OUT]
    python -m arachne.emulator annotate CKPT --bounds bounds.json [-o OUT]
    python -m arachne.emulator annotate CKPT --band-errors perband_fractional_error.json

``annotate`` records, in a ``ParrotEmulatorV2`` checkpoint that lacks them, the
training domain of every input (``--library`` or ``--bounds``) and the per-band
emulator error table (``--band-errors``).  The weights are not changed.
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
    annotate.add_argument("-o", "--output", type=Path, help="default: overwrite the checkpoint")
    args = parser.parse_args(argv)

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
    if not (args.library or args.bounds or args.band_errors):
        parser.error("annotate needs --library, --bounds or --band-errors")
    metadata = {}
    if args.library:
        metadata = _library_metadata(args.library, list(emulator.param_names))
    elif args.bounds:
        metadata = {"param_bounds": json.loads(args.bounds.read_text())}
    if args.band_errors:
        metadata["band_errors"] = json.loads(args.band_errors.read_text())
    annotated = emulator.with_metadata(**metadata)
    annotated.save(args.output or args.checkpoint)
    print(annotated.spec.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
