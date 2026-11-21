"""Measure spatial correlation from MaNGA DR17 pyPipe3D continuum maps."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

import numpy as np
from astropy.io import fits

RELEASE = "MaNGA DR17 pyPipe3D v3_1_1/3.1.1"
ROOT = "https://data.sdss.org/sas/dr17/manga/spectro/pipe3d/v3_1_1/3.1.1"
CATALOGUE = "SDSS17Pipe3D_v3_1_1.fits"


def download(url, path):
    """Cache an upstream file after a complete download."""
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        with urlopen(url, timeout=120) as response, partial.open("wb") as output:
            copyfileobj(response, output)
        partial.replace(path)
    return path


def select_sample(catalogue, per_group, seed):
    """Split a size- and quality-selected sample within two activity groups."""
    table = fits.getdata(catalogue, "GAL_PROPERTIES")
    eligible = (
        (table["QCFLAG"] == 0)
        & (table["nsa_redshift"] >= 0.012)
        & (table["nsa_redshift"] <= 0.055)
        & (table["Re_arc"] >= 6)
        & (table["Re_arc"] <= 17)
        & (table["Re_kpc"] > 0)
        & np.char.startswith(table["ifudsgn"].astype(str), "127")
        & (table["ellip"] >= 0)
        & (table["ellip"] < 0.75)
        & (table["log_Mass"] > 9)
        & (table["log_Mass"] < 11.7)
    )
    rng = np.random.default_rng(seed)
    _, unique = np.unique(table["mangaid"][eligible], return_index=True)
    eligible = table[eligible][unique]
    active = eligible["log_SFR_Ha"] - eligible["log_Mass"] > -11
    records = []
    for group, mask in (("active", active), ("low_activity", ~active)):
        subset = eligible[mask]
        for i, row in enumerate(subset[rng.permutation(len(subset))[:per_group]]):
            plate, ifu = row["plateifu"].strip().split("-")
            name = f"manga-{plate}-{ifu}.Pipe3D.cube.fits.gz"
            records.append(
                {
                    "id": row["mangaid"].strip(),
                    "plateifu": row["plateifu"].strip(),
                    "file": name,
                    "url": f"{ROOT}/{plate}/{name}",
                    "split": "test" if i % 4 == 0 else "train",
                    "group": group,
                    "z": float(row["nsa_redshift"]),
                    "re_arcsec": float(row["Re_arc"]),
                    "re_kpc": float(row["Re_kpc"]),
                    "ra": float(row["objra"]),
                    "dec": float(row["objdec"]),
                    "pa_deg": float(row["PA"]),
                    "axis_ratio": float(1 - row["ellip"]),
                    "log_mass": float(row["log_Mass"]),
                }
            )
    return records


def main():
    """Download the sample and dispatch the requested measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "measure"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=24)
    parser.add_argument("--seed", type=int, default=261121)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    catalogue = args.cache / CATALOGUE
    if args.command == "fetch":
        download(f"{ROOT}/{CATALOGUE}", catalogue)
        records = select_sample(catalogue, args.per_group, args.seed)
        (args.outdir / "sample.json").write_text(json.dumps(records, indent=2) + "\n")
        with ThreadPoolExecutor(max_workers=5) as pool:
            paths = pool.map(lambda row: download(row["url"], args.cache / row["file"]), records)
            for path in paths:
                print(path.name, flush=True)
        return
    from pipe3d_field_measurements import measure_sample

    records = json.loads((args.outdir / "sample.json").read_text())
    measure_sample(records, args.cache, args.outdir, catalogue, args.seed)


if __name__ == "__main__":
    main()
