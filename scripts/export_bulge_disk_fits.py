#!/usr/bin/env python3
"""Export NSS bulge+disk posteriors to a FITS catalogue.

For each component (bulge, disk) writes one FITS file with:
  - galaxy_id, redshift (fixed)
  - median and 1-sigma (16th/84th percentile) for each of the 11 free parameters
  - derived: log_M_total (bulge+disk, in matched file)
  - NSS diagnostics: logZ, logZ_err, ESS, n_dead, rhat_max

Also writes a third FITS file merging matched bulge+disk pairs with
bulge-fraction and total mass.

Usage
-----
    python scripts/export_bulge_disk_fits.py
    python scripts/export_bulge_disk_fits.py --work-dir /path/to/outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column

FREE_PARAMS = [
    "log_mass", "slope", "fesc_lya", "dust_bump_amplitude",
    "log10metallicity", "Av",
    "logsfr_ratio_0", "logsfr_ratio_1", "logsfr_ratio_2",
    "logsfr_ratio_3", "logsfr_ratio_4",
]
ALL_PARAMS = ["redshift"] + FREE_PARAMS   # samples[:,0] = fixed z


def load_merged(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {
            "galaxy_id":    f["galaxy_id"][:],
            "z_fixed":      f["z_fixed"][:],
            "samples":      f["nss_samples"][:],    # (N, S, 12)
            "logZ":         f["nss_logZ"][:],
            "logZ_err":     f["nss_logZ_err"][:],
            "ess":          f["nss_ess"][:],
            "n_dead":       f["nss_n_dead"][:],
            "rhat":         f["nss_rhat"][:],       # (N, 12)
            "time":         f["nss_time"][:],
        }


def make_component_table(data: dict, component: str) -> Table:
    gid    = data["galaxy_id"]
    z      = data["z_fixed"]
    samps  = data["samples"]          # (N, S, 12)
    N      = len(gid)

    # Percentiles along sample axis
    plo, pmed, phi = np.percentile(samps, [16, 50, 84], axis=1)  # each (N, 12)

    t = Table()
    t["galaxy_id"]  = Column(gid,  dtype="i8",   description="COSMOS2025 catalogue Id")
    t["redshift"]   = Column(z.astype("f4"),
                             description="Fixed spectroscopic redshift (zfinal)")
    t["component"]  = Column(np.full(N, component, dtype="U5"),
                             description="Component: bulge or disk")

    for i, p in enumerate(FREE_PARAMS):
        pi = ALL_PARAMS.index(p)          # column index in samples
        lo  = plo [:, pi].astype("f4")
        med = pmed[:, pi].astype("f4")
        hi  = phi [:, pi].astype("f4")
        err_lo = (med - lo).astype("f4")
        err_hi = (hi - med).astype("f4")

        t[p]               = Column(med,    description=f"{p} posterior median")
        t[f"{p}_err_lo"]   = Column(err_lo, description=f"{p} -1σ (median−16th pctile)")
        t[f"{p}_err_hi"]   = Column(err_hi, description=f"{p} +1σ (84th pctile−median)")

    # NSS diagnostics
    rhat_max = np.nanmax(data["rhat"], axis=1).astype("f4")
    t["nss_logZ"]    = Column(data["logZ"].astype("f4"),   description="NSS log evidence")
    t["nss_logZ_err"]= Column(data["logZ_err"].astype("f4"),
                               description="NSS log evidence MC uncertainty")
    t["nss_ess"]     = Column(data["ess"].astype("f4"),    description="Effective sample size")
    t["nss_n_dead"]  = Column(data["n_dead"].astype("i4"), description="Total dead particles")
    t["nss_rhat_max"]= Column(rhat_max,                    description="Worst split-R-hat (free params)")
    t["nss_time_s"]  = Column(data["time"].astype("f4"),   description="Wall time per galaxy (s)")

    return t


def make_combined_table(B: dict, D: dict) -> Table:
    """Matched table for galaxies with both bulge and disk fits."""
    bid = set(B["galaxy_id"].tolist())
    did = set(D["galaxy_id"].tolist())
    shared = np.array(sorted(bid & did), dtype="i8")

    b_map = {int(g): i for i, g in enumerate(B["galaxy_id"])}
    d_map = {int(g): i for i, g in enumerate(D["galaxy_id"])}

    ib = np.array([b_map[int(g)] for g in shared])
    id_ = np.array([d_map[int(g)] for g in shared])

    Bsamps = B["samples"]
    Dsamps = D["samples"]
    i_mass = ALL_PARAMS.index("log_mass")
    i_av   = ALL_PARAMS.index("Av")
    i_met  = ALL_PARAMS.index("log10metallicity")

    # Compute total mass from per-sample sum (propagates uncertainty properly)
    Mb_samps = Bsamps[ib, :, i_mass]   # (N, S)
    Md_samps = Dsamps[id_, :, i_mass]
    # Total stellar mass (sum in linear space)
    log_Mtot_samps = np.log10(10**Mb_samps + 10**Md_samps)  # (N, S)
    # Bulge mass fraction per sample
    f_bulge_samps  = 10**Mb_samps / (10**Mb_samps + 10**Md_samps)

    def pct(arr):
        lo, med, hi = np.percentile(arr, [16, 50, 84], axis=1)
        return med.astype("f4"), (med - lo).astype("f4"), (hi - med).astype("f4")

    Mtot_med, Mtot_lo, Mtot_hi = pct(log_Mtot_samps)
    fb_med,   fb_lo,   fb_hi   = pct(f_bulge_samps)

    Bmed = np.percentile(Bsamps[ib],  50, axis=1)
    Dmed = np.percentile(Dsamps[id_], 50, axis=1)

    t = Table()
    t["galaxy_id"]           = Column(shared, dtype="i8")
    t["redshift"]            = Column(B["z_fixed"][ib].astype("f4"))

    t["log_M_bulge"]         = Column(np.percentile(Bsamps[ib,  :, i_mass], 50, axis=1).astype("f4"),
                                       description="Bulge log M*/Msun median")
    t["log_M_bulge_err_lo"]  = Column((np.percentile(Bsamps[ib,  :, i_mass], 50, axis=1)
                                       - np.percentile(Bsamps[ib,  :, i_mass], 16, axis=1)).astype("f4"))
    t["log_M_bulge_err_hi"]  = Column((np.percentile(Bsamps[ib,  :, i_mass], 84, axis=1)
                                       - np.percentile(Bsamps[ib,  :, i_mass], 50, axis=1)).astype("f4"))

    t["log_M_disk"]          = Column(np.percentile(Dsamps[id_, :, i_mass], 50, axis=1).astype("f4"),
                                       description="Disk log M*/Msun median")
    t["log_M_disk_err_lo"]   = Column((np.percentile(Dsamps[id_, :, i_mass], 50, axis=1)
                                       - np.percentile(Dsamps[id_, :, i_mass], 16, axis=1)).astype("f4"))
    t["log_M_disk_err_hi"]   = Column((np.percentile(Dsamps[id_, :, i_mass], 84, axis=1)
                                       - np.percentile(Dsamps[id_, :, i_mass], 50, axis=1)).astype("f4"))

    t["log_M_total"]         = Column(Mtot_med, description="Total log M*/Msun (bulge+disk, median)")
    t["log_M_total_err_lo"]  = Column(Mtot_lo)
    t["log_M_total_err_hi"]  = Column(Mtot_hi)

    t["f_bulge"]             = Column(fb_med, description="Stellar bulge fraction M_b/(M_b+M_d)")
    t["f_bulge_err_lo"]      = Column(fb_lo)
    t["f_bulge_err_hi"]      = Column(fb_hi)

    # Av and metallicity for both components
    for label, samps_match, i_p, pname in [
        ("Av_bulge", Bsamps[ib],  i_av,  "Av"),
        ("Av_disk",  Dsamps[id_], i_av,  "Av"),
        ("met_bulge",Bsamps[ib],  i_met, "log10metallicity"),
        ("met_disk", Dsamps[id_], i_met, "log10metallicity"),
    ]:
        lo, med, hi = np.percentile(samps_match[:, :, i_p], [16, 50, 84], axis=1)
        t[label]            = Column(med.astype("f4"))
        t[f"{label}_err_lo"]= Column((med - lo).astype("f4"))
        t[f"{label}_err_hi"]= Column((hi - med).astype("f4"))

    return t


def write_fits(table: Table, path: Path, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(str(path), format="fits", overwrite=True)
    print(f"  Written: {path}  ({len(table)} rows, {len(table.colnames)} cols)")
    print(f"    {description}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export NSS posteriors to FITS catalogues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--work-dir",
        default="/cosma7/data/dp276/dc-harv3/work/outputs/nss_bulge_disk",
    )
    parser.add_argument(
        "--out-dir",
        default="/cosma7/data/dp276/dc-harv3/work/catalogs",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    out_dir  = Path(args.out_dir)

    print("Loading merged HDF5 files …")
    B = load_merged(work_dir / "results_bulge_merged.hdf5")
    D = load_merged(work_dir / "results_disk_merged.hdf5")
    print(f"  Bulge: {len(B['galaxy_id'])} galaxies")
    print(f"  Disk:  {len(D['galaxy_id'])} galaxies")

    print("\nBuilding FITS tables …")
    Btab = make_component_table(B, "bulge")
    Dtab = make_component_table(D, "disk")
    Ctab = make_combined_table(B, D)

    # Summary stats
    for name, tab in [("Bulge", Btab), ("Disk", Dtab)]:
        print(f"\n{name} summary:")
        print(f"  log_mass: {tab['log_mass'].mean():.2f} ± {tab['log_mass'].std():.2f}  "
              f"range [{tab['log_mass'].min():.1f}, {tab['log_mass'].max():.1f}]")
        print(f"  Av:       {tab['Av'].mean():.2f} ± {tab['Av'].std():.2f}")
        print(f"  median ESS: {np.median(tab['nss_ess']):.0f}")
        print(f"  n_dead>50: {(tab['nss_n_dead']>50).mean()*100:.1f}%")

    print(f"\nCombined (matched) summary:")
    print(f"  {len(Ctab)} galaxies with both components")
    print(f"  log_M_total: {Ctab['log_M_total'].mean():.2f} ± {Ctab['log_M_total'].std():.2f}")
    print(f"  f_bulge:     {np.median(Ctab['f_bulge']):.3f} (median)")

    print("\nWriting FITS …")
    write_fits(Btab, out_dir / "COSMOS2025_NSS_bulge.fits",
               "Per-galaxy bulge posteriors: median + 1σ for 11 SPS params + NSS diagnostics")
    write_fits(Dtab, out_dir / "COSMOS2025_NSS_disk.fits",
               "Per-galaxy disk posteriors: median + 1σ for 11 SPS params + NSS diagnostics")
    write_fits(Ctab, out_dir / "COSMOS2025_NSS_combined.fits",
               "Matched bulge+disk: total mass, bulge fraction, Av and metallicity for both")

    print("\nDone.")


if __name__ == "__main__":
    main()
