"""Fit Matérn covariance to GalSyn DR1 TNG50 maps with matched broadband images."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
from astropy.io import fits
from calibrate_pipe3d_spatial_priors import download
from field_covariance import NU, BinGeometry, fit_profiles
from pipe3d_field_measurements import LENGTHS_RE

SOURCE = "GalSyn DR1 TNG50 massive progenitors, angle1/dust0-skirt-like"
FOLDER = "1MqlGAEC4VbNkhNsY-cL1EEqrJvkc7Crq"
HISTORY_ID = "17FwW1ec1U_g6w6Epj9XrNz1JQpFgY2iw"
FIELD_UNITS = {"Av": "mag", "log10metallicity": "dex", "logssfr_100myr": "dex"}
AMPLITUDE_BOUNDS = {"Av": (0.001, 3), "log10metallicity": (0.001, 2), "logssfr_100myr": (0.005, 3)}
NUISANCE_FLOORS = {"Av": 0.01, "log10metallicity": 0.015, "logssfr_100myr": 0.05}
SNAPSHOTS = (50, 33, 25)
SSFR_FLOOR = 1e-11


def download_drive(file_id, path):
    """Download a published GalSyn archive and verify that Drive returned the file."""
    if not path.exists():
        urls = (
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
        )
        for url in urls:
            try:
                download(url, path)
            except HTTPError as error:
                if error.code not in (403, 404):
                    raise
                continue
            with path.open("rb") as stream:
                magic = stream.read(6)
            if magic.startswith((b"\x1f\x8b", b"SIMPLE")):
                break
            path.unlink()
        else:
            raise FileNotFoundError(f"GalSyn file {file_id} could not be downloaded")
    return path


def select_sample(index, history, per_group, seed):
    """Select disjoint progenitor lineages within activity strata at three snapshots."""
    pattern = (
        r'<a href="https://drive.google.com/file/d/([^/]+)/[^"]+"[^>]*>.*?'
        r"flip-entry-title[^>]*>(galsyn_tng50_shalo_(\d+)_(\d+)\.fits\.gz)"
    )
    items = re.findall(pattern, index.read_text(), flags=re.S)
    available = {(int(snap), int(sub)): (key, file) for key, file, snap, sub in items}
    rng = np.random.default_rng(seed)
    used = set()
    sample = []
    with fits.open(history) as hdus:
        epochs = hdus["SNAPSHOT_NUMBER"].data
        tree = hdus["SUBHALO_IDS"].data
        masses = hdus["LOG_MSTAR_2RE"].data
        sfr = hdus["LOG_SFR_2RE"].data
        for snap in SNAPSHOTS:
            at = np.flatnonzero(epochs["snaps"] == snap).item()
            candidates = []
            for lineage in tree.names:
                sub = int(tree[lineage][at])
                logm = float(masses[lineage][at])
                logssfr = max(-16.0, float(sfr[lineage][at] - logm))
                if (snap, sub) in available and 9.5 <= logm <= 11.7:
                    key, file = available[(snap, sub)]
                    candidates.append((lineage, sub, key, file, logm, logssfr))
            subs, counts = np.unique([row[1] for row in candidates], return_counts=True)
            unambiguous = set(subs[counts == 1])
            candidates = [row for row in candidates if row[1] in unambiguous]
            median = np.nanmedian([row[-1] for row in candidates])
            for group, active in (("active", True), ("low_activity", False)):
                pool = [
                    row for row in candidates if (row[-1] > median) == active and row[0] not in used
                ]
                if len(pool) < per_group:
                    raise ValueError(f"Only {len(pool)} eligible galaxies at {snap} in {group}")
                for i in rng.choice(len(pool), per_group, replace=False):
                    lineage, sub, key, file, logm, logssfr = pool[i]
                    used.add(lineage)
                    sample.append(
                        {
                            "id": f"{snap}-{sub}",
                            "lineage": lineage,
                            "snapshot": snap,
                            "group": group,
                            "z": float(epochs["z"][at]),
                            "file": file,
                            "key": key,
                            "log_mass": logm,
                            "log_ssfr_catalogue": logssfr,
                        }
                    )
                for i, record in enumerate(sample[-per_group:]):
                    record["split"] = "test" if i % 4 == 0 else "train"
    return sample


def read_truth(path, ssfr_floor=SSFR_FLOOR):
    """Read intrinsic maps and obtain a projected half-stellar-mass radius."""
    with fits.open(path, memmap=False) as h:
        header = h[0].header
        mass = np.maximum(np.asarray(h["STARS_MASS"].data, float), 0)
        metal = np.asarray(h["STARS_MW_ZSOL"].data, float)
        av = np.asarray(h["DUST_MEAN_AV"].data, float)
        sfr100 = np.asarray(h["SFR_100MYR"].data, float)
        scale = float(header["PIXSIZE"])
        kpc = float(header["PIX_KPC"])
        z = float(header["REDSHIFT"])
        # Dimming in the observed wavelength nearest rest-frame Johnson V.
        filters = {
            0.70: "F070W",
            0.90: "F090W",
            1.15: "F115W",
            1.50: "F150W",
            2.00: "F200W",
            2.77: "F277W",
        }
        band = filters[
            min(filters, key=lambda wavelength: abs(np.log(wavelength / (0.55 * (1 + z)))))
        ]
        flux_d = np.asarray(h[f"DUST_JWST_NIRCAM_{band}"].data, float)
        flux_0 = np.asarray(h[f"NODUST_JWST_NIRCAM_{band}"].data, float)
        x, y = float(header["CRPIX1"]) - 1, float(header["CRPIX2"]) - 1
    yy, xx = np.indices(mass.shape)
    r = np.hypot(xx - x, yy - y)
    use = r * kpc < 20
    order = np.argsort(r[use])
    cumulative = np.cumsum(mass[use][order])
    re_px = float(np.interp(cumulative[-1] / 2, cumulative, r[use][order]))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_dex = np.log10(metal * 0.02)
        ssfr = np.log10(np.maximum(sfr100 / mass, ssfr_floor))
        veff = -2.5 * np.log10(flux_d / flux_0)
    return {
        "maps": {"Av": veff, "log10metallicity": z_dex, "logssfr_100myr": ssfr},
        "mass": mass,
        "sfr100": sfr100,
        "radius": r,
        "re_px": re_px,
        "pixel_kpc": kpc,
        "pixel_arcsec": scale,
        "center": (x, y),
        "mean_particle_av": av,
        "v_flux_pair": (flux_d, flux_0),
        "z": z,
    }


def measure_galaxy(galaxy, rng, ssfr_floor=SSFR_FLOOR):
    """Measure intrinsic correlations above particle and pixel sampling scales."""
    r, re = galaxy["radius"], galaxy["re_px"]
    mass = galaxy["mass"]
    # Marry the relative mask with a 20-particle mass threshold so the young SFR
    # censoring and rare-particle metallicities in the outskirts do not dominate.
    good = (r <= 1.5 * re) & (mass >= max(1.6e6, 1e-3 * mass.max()))
    idx = np.flatnonzero(good)
    if idx.size < 25:
        return {}
    idx = np.sort(rng.choice(idx, min(160, idx.size), replace=False))
    segmentation = np.zeros(mass.shape, dtype=np.int32)
    labels = np.arange(1, len(idx) + 1)
    segmentation.flat[idx] = labels
    geom = BinGeometry(segmentation, np.ones_like(mass), labels, r / re, 0, max_points=1)
    maps = {
        name: (values.flat[idx], np.full(len(idx), NUISANCE_FLOORS[name]))
        for name, values in galaxy["maps"].items()
    }
    fits_by_field = fit_profiles(geom, maps, LENGTHS_RE * re, AMPLITUDE_BOUNDS)
    for name, result in fits_by_field.items():
        result["fraction_sfr_floor"] = (
            float(np.mean(maps[name][0] <= np.log10(ssfr_floor)))
            if name == "logssfr_100myr"
            else None
        )
    return fits_by_field


def main():
    """Download or measure a fixed-split GalSyn sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "measure"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=261122)
    parser.add_argument("--ssfr-floor", type=float, default=SSFR_FLOOR)
    args = parser.parse_args()
    if not np.isfinite(args.ssfr_floor) or args.ssfr_floor <= 0:
        parser.error("--ssfr-floor must be finite and positive")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = args.outdir / "sample.json"
    if args.command == "fetch":
        index = download(
            f"https://drive.google.com/embeddedfolderview?id={FOLDER}", args.cache / "index.html"
        )
        history = download_drive(HISTORY_ID, args.cache / "history.fits")
        rows = select_sample(index, history, args.per_group, args.seed)
        manifest.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")

        def fetch(row):
            try:
                download_drive(row["key"], args.cache / row["file"])
                return row["id"], "ok"
            except FileNotFoundError:
                return row["id"], "unavailable"

        with ThreadPoolExecutor(max_workers=5) as pool:
            for result in pool.map(fetch, rows):
                print(*result, flush=True)
        return
    rows = json.loads(manifest.read_text())
    result = {
        "version": 1,
        "source": SOURCE,
        "nu": NU,
        "seed": args.seed,
        "field_units": FIELD_UNITS,
        "transferable_fields": ["Av", "log10metallicity"],
        "measurement_error_floor": NUISANCE_FLOORS,
        "ssfr_floor_per_year": args.ssfr_floor,
        "ell_over_re_grid": LENGTHS_RE.tolist(),
        "galaxies": [],
    }
    for row in rows:
        path = args.cache / row["file"]
        if not path.exists():
            print(row["id"], "unavailable", flush=True)
            continue
        galaxy = read_truth(path, args.ssfr_floor)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, *map(ord, row["id"])]))
        fields = measure_galaxy(galaxy, rng, args.ssfr_floor)
        result["galaxies"].append(
            {
                **row,
                "fields": fields,
                "re_kpc": galaxy["re_px"] * galaxy["pixel_kpc"],
                "beam_fwhm_over_re": 0.0,
                "pixel_over_re": 1 / galaxy["re_px"],
            }
        )
        print(row["id"], row["split"], len(fields), flush=True)
        (args.outdir / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
