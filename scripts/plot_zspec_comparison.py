#!/usr/bin/env python3
"""Plot spec-z vs NSS photo-z comparison from an HDF5 results file.

Matches galaxy IDs to a FITS catalogue and reads specz + z_Spec_flag.
Produces a two-panel figure: z_spec vs z_NSS scatter, and Δz/(1+z) histogram.

Usage
-----
    python scripts/plot_zspec_comparison.py results.hdf5 catalogue.fits
    python scripts/plot_zspec_comparison.py results.hdf5 catalogue.fits --out figure.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from astropy.table import Table


def parse_flags(flag_col) -> np.ndarray:
    """Return string array from a possibly-masked bytes flag column."""
    out = []
    for f in flag_col:
        if ma.is_masked(f) or (hasattr(f, "mask") and np.all(f.mask)):
            out.append("secure")
        else:
            v = f.data[0] if hasattr(f, "data") else f
            s = v.decode() if isinstance(v, bytes) else str(v).strip()
            out.append(s if s else "secure")
    return np.array(out)


def nmad(dz: np.ndarray) -> float:
    """Normalised median absolute deviation of a redshift error array."""
    return 1.4826 * float(np.median(np.abs(dz - np.median(dz))))


def main() -> None:
    """CLI entry point: plot spec-z vs NSS photo-z comparison."""
    parser = argparse.ArgumentParser(
        description="spec-z vs NSS photo-z comparison plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hdf5",      help="NSS results HDF5 file.")
    parser.add_argument("catalogue", help="FITS catalogue with specz and z_Spec_flag.")
    parser.add_argument("--out",     default=None,
                        help="Output PNG path. Defaults to <hdf5_dir>/zspec_vs_znss.png.")
    parser.add_argument("--dpi",     type=int, default=150)
    parser.add_argument("--zmax",    type=float, default=None,
                        help="Upper limit for z axes. Auto if not set.")
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)
    out_path  = Path(args.out) if args.out else hdf5_path.parent / "zspec_vs_znss.png"

    # ---- load NSS results ----
    with h5py.File(hdf5_path, "r") as f:
        param_names = [b.decode() if isinstance(b, bytes) else b
                       for b in f.attrs["param_names"]]
        gids   = f["galaxy_id"][:]
        samps  = f["nss_samples"][:]   # (N, S, P)

    z_idx   = param_names.index("redshift")
    z_samps = samps[:, :, z_idx]
    z_lo    = np.percentile(z_samps, 16, axis=1)
    z_med   = np.percentile(z_samps, 50, axis=1)
    z_hi    = np.percentile(z_samps, 84, axis=1)
    z_elo   = z_med - z_lo
    z_ehi   = z_hi  - z_med

    # ---- load catalogue ----
    cat      = Table.read(args.catalogue)
    cat_id   = np.array(cat["UNIQUE_ID"])
    cat_z    = np.array(cat["specz"], dtype=float)
    cat_flag = parse_flags(cat["z_Spec_flag"])

    id_to_row = {int(uid): i for i, uid in enumerate(cat_id)}
    missing   = [gid for gid in gids if int(gid) not in id_to_row]
    if missing:
        print(f"Warning: {len(missing)} galaxy IDs not found in catalogue.")

    specz = np.array([cat_z[id_to_row[int(g)]] for g in gids if int(g) in id_to_row])
    flags = np.array([cat_flag[id_to_row[int(g)]] for g in gids if int(g) in id_to_row])
    keep  = np.array([int(g) in id_to_row for g in gids])
    z_med = z_med[keep]
    z_elo = z_elo[keep]
    z_ehi = z_ehi[keep]

    dz = (z_med - specz) / (1 + specz)

    # ---- per-flag stats ----
    flag_order  = ["secure", "A", "B", "C"]
    flag_labels = {"secure": "Secure (no flag)", "A": "Flag A", "B": "Flag B", "C": "Flag C"}
    flag_colors = {"secure": "#2060a0", "A": "#20a060", "B": "#e08000", "C": "#cc2020"}
    flag_marker = {"secure": "o", "A": "D", "B": "s", "C": "^"}

    print(f"\n{'Flag':8s} {'N':>5s}  {'NMAD':>7s}  {'|Δz|>0.15':>10s}")
    print("-" * 38)
    for flag in flag_order:
        m = flags == flag
        if not m.sum():
            continue
        n_out = (np.abs(dz[m]) > 0.15).sum()
        print(f"{flag:8s} {m.sum():>5d}  {nmad(dz[m]):>7.4f}  "
              f"{n_out:>4d}/{m.sum():<4d} ({100*n_out/m.sum():.1f}%)")

    m_sec = flags == "secure"
    nmad_sec = nmad(dz[m_sec])
    out_sec  = 100 * (np.abs(dz[m_sec]) > 0.15).mean()

    # ---- plot ----
    zmax = args.zmax or max(z_med.max(), specz.max()) * 1.08

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # left: scatter
    ax = axes[0]
    ax.plot([0, zmax], [0, zmax], "k-", lw=0.8, zorder=0)
    ax.fill_between([0, zmax], [0, zmax * (1 - 0.15)], [0, zmax * (1 + 0.15)],
                    color="grey", alpha=0.08, zorder=0, label=r"$|\Delta z|/(1+z)$ < 0.15")

    for flag in flag_order:
        m = flags == flag
        if not m.sum():
            continue
        n_out = (np.abs(dz[m]) > 0.15).sum()
        label = f"{flag_labels[flag]} ({m.sum()},  NMAD={nmad(dz[m]):.3f})"
        ax.errorbar(
            specz[m], z_med[m],
            yerr=[z_elo[m], z_ehi[m]],
            fmt=flag_marker[flag], ms=3, color=flag_colors[flag],
            ecolor=flag_colors[flag], elinewidth=0.5, capsize=0,
            lw=0, alpha=0.7, label=label, zorder=3,
        )

    ax.set_xlabel(r"$z_\mathrm{spec}$", fontsize=12)
    ax.set_ylabel(r"$z_\mathrm{NSS}$ (median ± 68% CI)", fontsize=12)
    ax.set_xlim(0, zmax)
    ax.set_ylim(0, zmax)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, markerscale=1.5)
    ax.set_title("Spectroscopic vs NSS photometric redshift", fontsize=11)
    ax.text(
        0.04, 0.96,
        f"Secure  NMAD = {nmad_sec:.4f}\nOutlier frac = {out_sec:.1f}%",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85),
    )

    # right: residual histogram
    ax2 = axes[1]
    ax2.axvline(0, color="k", lw=0.8)
    ax2.axvspan(-0.15, 0.15, color="grey", alpha=0.08)
    dz_range = max(np.abs(dz).max() * 1.05, 0.3)
    bins = np.linspace(-dz_range, dz_range, 60)
    for flag in flag_order:
        m = flags == flag
        if not m.sum():
            continue
        ax2.hist(dz[m], bins=bins, color=flag_colors[flag],
                 edgecolor="none", alpha=0.6,
                 label=f"{flag_labels[flag]} ({m.sum()})")
    ax2.set_xlabel(r"$\Delta z / (1+z_\mathrm{spec})$", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Redshift residuals", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.text(
        0.04, 0.96, f"NMAD (secure) = {nmad_sec:.4f}",
        transform=ax2.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
