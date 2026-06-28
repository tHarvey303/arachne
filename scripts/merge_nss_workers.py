#!/usr/bin/env python3
"""Merge per-worker NSS HDF5 outputs into a single file.

Each worker produces results for a contiguous slice of the catalogue.
This script concatenates them in order and writes one combined HDF5.

Usage
-----
    python scripts/merge_nss_workers.py worker_0/results.hdf5 worker_1/results.hdf5 -o results.hdf5
    python scripts/merge_nss_workers.py worker_*/results.hdf5 -o merged/results.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    """CLI entry point: merge per-worker NSS HDF5 outputs into a single file."""
    parser = argparse.ArgumentParser(
        description="Merge per-worker NSS HDF5 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Worker HDF5 files, in order.")
    parser.add_argument("-o", "--output", required=True, help="Output HDF5 path.")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all files and collect arrays.
    parts: list[dict] = []
    ref_attrs: dict = {}

    for i, path in enumerate(inputs):
        with h5py.File(path, "r") as f:
            attrs = dict(f.attrs)
            data  = {k: f[k][:] for k in f.keys()}
        parts.append(data)
        if i == 0:
            ref_attrs = attrs
        else:
            # sanity: same param_names across workers
            if not np.array_equal(attrs.get("param_names"), ref_attrs.get("param_names")):
                raise ValueError(f"param_names mismatch between worker 0 and worker {i}")

    # Concatenate along axis 0 (galaxy axis).
    all_keys = set()
    for p in parts:
        all_keys |= set(p.keys())

    merged: dict[str, np.ndarray] = {}
    for key in sorted(all_keys):
        arrays = [p[key] for p in parts if key in p]
        merged[key] = np.concatenate(arrays, axis=0)

    total = len(merged["galaxy_id"])
    print(f"Merging {len(inputs)} workers → {total} galaxies total")

    # Write output.
    with h5py.File(out_path, "w") as f:
        # Copy attrs from first file, update n_galaxies.
        for k, v in ref_attrs.items():
            f.attrs[k] = v
        f.attrs["n_galaxies"] = total

        for key, arr in merged.items():
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

    print(f"Written: {out_path}")

    # Print summary stats.
    with h5py.File(out_path, "r") as f:
        rhat = f["nss_rhat"][:]
        ess  = f["nss_ess"][:]
        t    = f["nss_time"][:]
        rhat_max = np.nanmax(rhat, axis=1)
        print(f"  ESS:         median={np.median(ess):.0f}")
        print(f"  R-hat<1.05:  {100*(rhat_max<1.05).mean():.1f}%")
        print(f"  R-hat>1.1:   {100*(rhat_max>1.10).mean():.1f}%")
        print(f"  Time/galaxy: median={np.median(t):.1f}s  total={t.sum()/3600:.2f}h")


if __name__ == "__main__":
    main()
