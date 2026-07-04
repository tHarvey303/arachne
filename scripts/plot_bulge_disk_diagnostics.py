#!/usr/bin/env python3
"""Diagnostic plots for NSS bulge+disk fixed-z fits.

Loads completed per-worker HDF5 files, concatenates them, and produces
three figures: run QC, physical parameter distributions, and science plots.

Usage
-----
    python scripts/plot_bulge_disk_diagnostics.py \\
        --work-dir /cosma7/data/dp276/dc-harv3/work/outputs/nss_bulge_disk \\
        --workers 1 2 3 4 5 6 \\
        --catalogue /cosma7/data/dp276/dc-harv3/work/catalogs/fluxes_bulge_disk_C25.csv \\
        --out-dir /cosma7/data/dp276/dc-harv3/work/outputs/nss_bulge_disk/diagnostics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from astropy.table import Table

PARAM_NAMES = [
    "redshift", "log_mass", "slope", "fesc_lya", "dust_bump_amplitude",
    "log10metallicity", "Av",
    "logsfr_ratio_0", "logsfr_ratio_1", "logsfr_ratio_2",
    "logsfr_ratio_3", "logsfr_ratio_4",
]
PARAM_LABELS = {
    "redshift":            r"$z$ (fixed)",
    "log_mass":            r"$\log M_\star / M_\odot$",
    "slope":               r"$\delta$ (attenuation slope)",
    "fesc_lya":            r"$f_\mathrm{esc,Ly\alpha}$",
    "dust_bump_amplitude": r"$B_{2175}$ (dust bump)",
    "log10metallicity":    r"$\log Z / Z_\odot$",
    "Av":                  r"$A_V$ (mag)",
    "logsfr_ratio_0":      r"$\log \mathrm{SFR}_0/\mathrm{SFR}_1$",
    "logsfr_ratio_1":      r"$\log \mathrm{SFR}_1/\mathrm{SFR}_2$",
    "logsfr_ratio_2":      r"$\log \mathrm{SFR}_2/\mathrm{SFR}_3$",
    "logsfr_ratio_3":      r"$\log \mathrm{SFR}_3/\mathrm{SFR}_4$",
    "logsfr_ratio_4":      r"$\log \mathrm{SFR}_4/\mathrm{SFR}_5$",
}

C_BULGE = "#d62728"
C_DISK  = "#1f77b4"
ALPHA   = 0.7


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_workers(work_dir: Path, workers: list[int], component: str) -> dict:
    """Load and concatenate HDF5 files from multiple workers."""
    parts = []
    for w in workers:
        p = work_dir / f"w{w}" / f"results_{component}.hdf5"
        if not p.exists():
            print(f"  WARNING: {p} not found — skipping worker {w}", flush=True)
            continue
        with h5py.File(p, "r") as f:
            parts.append({
                "galaxy_id":   f["galaxy_id"][:],
                "z_fixed":     f["z_fixed"][:],
                "nss_samples": f["nss_samples"][:],    # (N, S, 12)
                "nss_logZ":    f["nss_logZ"][:],
                "nss_logZ_err":f["nss_logZ_err"][:],
                "nss_ess":     f["nss_ess"][:],
                "nss_n_dead":  f["nss_n_dead"][:],
                "nss_rhat":    f["nss_rhat"][:],       # (N, 12)
                "nss_time":    f["nss_time"][:],
            })
        print(f"  loaded w{w}/{component}: {parts[-1]['galaxy_id'].shape[0]} galaxies")

    if not parts:
        raise FileNotFoundError(f"No {component} files found.")

    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def percentiles(samples: np.ndarray, q=(16, 50, 84)):
    """Return percentiles along the sample axis (N, S, P) → (3, N, P)."""
    return np.percentile(samples, q, axis=1)


# ---------------------------------------------------------------------------
# Figure 1: Run QC
# ---------------------------------------------------------------------------

def fig_qc(B: dict, D: dict, out_path: Path, dpi: int = 150) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("NSS Run QC — Bulge & Disk", fontsize=13, y=1.01)

    # logZ — clip extreme outliers (< 1st percentile) for display
    for comp, data, c in [("Bulge", B, C_BULGE), ("Disk", D, C_DISK)]:
        logz = data["nss_logZ"]
        lo = np.percentile(logz, 1)
        hi = np.percentile(logz, 99)
        n_bad = (logz < lo).sum()
        good = logz[logz >= lo]
        ax = axes[0, 0]
        ax.hist(good, bins=80, color=c, alpha=ALPHA, label=f"{comp} (N={len(logz)}, {n_bad} clipped)")
    ax.set_xlabel("log Z (NSS evidence)")
    ax.set_ylabel("Count")
    ax.set_title("Evidence distribution (1–99th pctile)")
    ax.legend(fontsize=8)
    ax.axvline(0, color="k", lw=0.8, ls="--")

    # logZ_err
    ax = axes[0, 1]
    for comp, data, c in [("Bulge", B, C_BULGE), ("Disk", D, C_DISK)]:
        ax.hist(np.clip(data["nss_logZ_err"], 0, np.percentile(data["nss_logZ_err"], 99)),
                bins=60, color=c, alpha=ALPHA, label=comp)
    ax.set_xlabel(r"$\sigma(\log Z)$")
    ax.set_title("Evidence uncertainty")
    ax.legend(fontsize=8)

    # ESS
    ax = axes[0, 2]
    for comp, data, c in [("Bulge", B, C_BULGE), ("Disk", D, C_DISK)]:
        ax.hist(data["nss_ess"], bins=60, color=c, alpha=ALPHA, label=comp)
    ax.set_xlabel("Effective sample size")
    ax.set_title("ESS")
    ax.axvline(500, color="grey", lw=0.8, ls="--", label="ESS=500")
    ax.legend(fontsize=8)

    # R-hat max
    ax = axes[1, 0]
    for comp, data, c in [("Bulge", B, C_BULGE), ("Disk", D, C_DISK)]:
        rmax = np.nanmax(data["nss_rhat"], axis=1)
        ax.hist(rmax, bins=60, color=c, alpha=ALPHA, label=comp)
    ax.set_xlabel(r"$\hat{R}_\mathrm{max}$ (worst parameter)")
    ax.set_title("Split R-hat (1.0 = perfect)")
    ax.axvline(1.05, color="k", lw=0.8, ls="--", label="1.05")
    ax.legend(fontsize=8)

    # n_dead
    ax = axes[1, 1]
    for comp, data, c in [("Bulge", B, C_BULGE), ("Disk", D, C_DISK)]:
        nd = data["nss_n_dead"]
        frac_min = (nd == nd.min()).mean() * 100
        ax.hist(nd, bins=80, color=c, alpha=ALPHA,
                label=f"{comp} (1-step: {frac_min:.1f}%)")
    ax.set_xlabel("Dead particles")
    ax.set_title("NSS iterations (n_dead)")
    ax.legend(fontsize=8)

    # Time per galaxy
    ax = axes[1, 2]
    for comp, data, c in [("Bulge", B, C_BULGE), ("Disk", D, C_DISK)]:
        t = data["nss_time"]
        ax.hist(t[t < np.percentile(t, 99)], bins=60, color=c, alpha=ALPHA,
                label=f"{comp} (med={np.median(t):.1f}s)")
    ax.set_xlabel("Time per galaxy (s)")
    ax.set_title("Fitting time")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Physical parameter distributions
# ---------------------------------------------------------------------------

def fig_params(B: dict, D: dict, out_path: Path, dpi: int = 150) -> None:
    # Use only well-converged galaxies (n_dead > minimum = 50)
    bm = B["nss_n_dead"] > 50
    dm = D["nss_n_dead"] > 50
    print(f"  Well-converged: bulge {bm.sum()}/{len(bm)}  disk {dm.sum()}/{len(dm)}")

    # Posterior medians for each galaxy
    Bmed = np.median(B["nss_samples"][bm], axis=1)  # (N_b, 12)
    Dmed = np.median(D["nss_samples"][dm], axis=1)  # (N_d, 12)
    # Posterior widths (84-16 percentile)
    Bq = np.percentile(B["nss_samples"][bm], [16, 84], axis=1)  # (2, N, 12)
    Dq = np.percentile(D["nss_samples"][dm], [16, 84], axis=1)
    Bwidth = Bq[1] - Bq[0]  # (N, 12)
    Dwidth = Dq[1] - Dq[0]

    # Parameters to show (skip redshift = fixed, skip SFR ratios for space)
    show = ["log_mass", "Av", "log10metallicity", "slope",
            "dust_bump_amplitude", "fesc_lya"]
    show_idx = [PARAM_NAMES.index(p) for p in show]

    fig, axes = plt.subplots(3, len(show), figsize=(18, 11))
    fig.suptitle("Posterior medians and constraint widths", fontsize=13)

    for col, (p, pidx) in enumerate(zip(show, show_idx)):
        label = PARAM_LABELS[p]

        # Row 0: distribution of posterior medians
        ax = axes[0, col]
        vb, vd = Bmed[:, pidx], Dmed[:, pidx]
        lo = min(np.percentile(vb, 1), np.percentile(vd, 1))
        hi = max(np.percentile(vb, 99), np.percentile(vd, 99))
        bins = np.linspace(lo, hi, 50)
        ax.hist(vb, bins=bins, color=C_BULGE, alpha=ALPHA, label="Bulge", density=True)
        ax.hist(vd, bins=bins, color=C_DISK,  alpha=ALPHA, label="Disk",  density=True)
        ax.set_xlabel(label, fontsize=8)
        if col == 0:
            ax.set_ylabel("Density", fontsize=8)
            ax.set_title("Posterior median", fontsize=9)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=7)

        # Row 1: posterior width (84-16 CI)
        ax = axes[1, col]
        wb, wd = Bwidth[:, pidx], Dwidth[:, pidx]
        lo = 0
        hi = max(np.percentile(wb, 99), np.percentile(wd, 99))
        bins = np.linspace(lo, hi, 50)
        ax.hist(wb, bins=bins, color=C_BULGE, alpha=ALPHA, density=True)
        ax.hist(wd, bins=bins, color=C_DISK,  alpha=ALPHA, density=True)
        ax.set_xlabel(label, fontsize=8)
        if col == 0:
            ax.set_ylabel("Density", fontsize=8)
            ax.set_title("84–16% CI width", fontsize=9)
        ax.tick_params(labelsize=7)

        # Row 2: posterior width vs median (scatter, both components)
        ax = axes[2, col]
        ax.hexbin(vb, wb, gridsize=40, cmap="Reds",  mincnt=1,
                  extent=[lo, hi, 0, np.percentile(wb, 99)], alpha=0.85)
        ax.hexbin(vd, wd, gridsize=40, cmap="Blues", mincnt=1,
                  extent=[lo, hi, 0, np.percentile(wd, 99)], alpha=0.7)
        ax.set_xlabel(label, fontsize=8)
        if col == 0:
            ax.set_ylabel("CI width", fontsize=8)
            ax.set_title("Constraint vs value", fontsize=9)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Science — bulge vs disk comparisons
# ---------------------------------------------------------------------------

def fig_science(B: dict, D: dict, cat_path: Path, out_path: Path, dpi: int = 150) -> None:
    cat = Table.read(cat_path)
    cat_id   = np.array(cat["Id"],      dtype=np.int64)
    cat_bt   = np.array(cat["BT_f444w"], dtype=float)
    cat_z    = np.array(cat["zfinal"],  dtype=float)

    # --- match bulge to disk by galaxy_id ---
    # Only use well-converged galaxies
    bm = B["nss_n_dead"] > 50
    dm = D["nss_n_dead"] > 50

    bid_all = B["galaxy_id"][bm]
    did_all = D["galaxy_id"][dm]
    shared  = np.intersect1d(bid_all, did_all)
    print(f"  Shared bulge+disk IDs: {len(shared)}")

    b_idx = {gid: i for i, gid in enumerate(bid_all)}
    d_idx = {gid: i for i, gid in enumerate(did_all)}
    c_idx = {gid: i for i, gid in enumerate(cat_id)}

    ib = np.array([b_idx[g] for g in shared])
    id_ = np.array([d_idx[g] for g in shared])
    ic  = np.array([c_idx[g] for g in shared if g in c_idx])
    shared_cat = np.array([g for g in shared if g in c_idx])

    Bsamp = B["nss_samples"][bm]
    Dsamp = D["nss_samples"][dm]
    i_mass = PARAM_NAMES.index("log_mass")
    i_av   = PARAM_NAMES.index("Av")
    i_met  = PARAM_NAMES.index("log10metallicity")

    Bmed = np.median(Bsamp, axis=1)   # (N_b, 12)
    Dmed = np.median(Dsamp, axis=1)

    bM  = Bmed[ib, i_mass]   # log_mass bulge (matched)
    dM  = Dmed[id_, i_mass]  # log_mass disk  (matched)
    bAv = Bmed[ib, i_av]
    dAv = Dmed[id_, i_av]
    bZ  = Bmed[ib, i_met]
    dZ  = Dmed[id_, i_met]
    zz  = B["z_fixed"][bm][ib]

    # Bulge mass fraction from NSS
    Mb  = 10 ** bM
    Md  = 10 ** dM
    nss_f_bulge = Mb / (Mb + Md)

    # Catalogue BT_f444w for the matched sample
    ic2 = np.array([c_idx[g] for g in shared_cat])
    bt_matched = cat_bt[ic2]
    shared_cat_set = set(shared_cat.tolist())
    shared_bt = np.array([g in shared_cat_set for g in shared])
    bt444 = np.array([cat_bt[c_idx[g]] if g in c_idx else np.nan for g in shared])

    # All bulge medians for mass distribution
    all_bM  = Bmed[:, i_mass]
    all_dM  = Dmed[:, i_mass]
    all_bZ  = B["z_fixed"][bm]
    all_dZ  = D["z_fixed"][dm]

    # --- total stellar mass per galaxy (bulge+disk) ---
    total_log_mass = np.log10(Mb + Md)

    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Bulge+Disk Science Diagnostics", fontsize=13)

    # (0,0) Bulge vs disk log_mass scatter coloured by z
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(dM, bM, c=zz, s=2, cmap="plasma", vmin=0, vmax=3, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="$z$", shrink=0.85)
    lims = [min(dM.min(), bM.min())-0.1, max(dM.max(), bM.max())+0.1]
    ax.plot(lims, lims, "k-", lw=0.8, zorder=0)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r"$\log M_\mathrm{disk} / M_\odot$")
    ax.set_ylabel(r"$\log M_\mathrm{bulge} / M_\odot$")
    ax.set_title(f"Bulge vs disk mass (N={len(bM)})")
    ax.set_aspect("equal")

    # (0,1) NSS bulge fraction vs catalogue BT_f444w
    ax = fig.add_subplot(gs[0, 1])
    good_bt = np.isfinite(bt444) & (bt444 > 0)
    ax.hexbin(bt444[good_bt], nss_f_bulge[good_bt],
              gridsize=50, cmap="viridis", mincnt=1, extent=[0,1,0,1])
    ax.plot([0, 1], [0, 1], "w-", lw=1.0)
    ax.set_xlabel("BT$_{F444W}$ (catalogue morphology)")
    ax.set_ylabel(r"$M_\mathrm{bulge}/(M_\mathrm{bulge}+M_\mathrm{disk})$ (NSS)")
    ax.set_title("Stellar vs flux bulge fraction")

    # (0,2) Total log_mass vs redshift (all, bulge+disk)
    ax = fig.add_subplot(gs[0, 2])
    ax.hexbin(zz, total_log_mass, gridsize=50, cmap="Greens", mincnt=1)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$\log(M_\mathrm{bulge}+M_\mathrm{disk}) / M_\odot$")
    ax.set_title("Total stellar mass vs redshift")

    # (1,0) log_mass distribution, all galaxies
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(6, 13, 60)
    ax.hist(all_bM, bins=bins, color=C_BULGE, alpha=ALPHA, label=f"Bulge (N={len(all_bM)})", density=True)
    ax.hist(all_dM, bins=bins, color=C_DISK,  alpha=ALPHA, label=f"Disk (N={len(all_dM)})", density=True)
    ax.set_xlabel(r"$\log M_\star / M_\odot$ (median posterior)")
    ax.set_ylabel("Density")
    ax.set_title("Stellar mass distributions")
    ax.legend(fontsize=8)

    # (1,1) Av(bulge) vs Av(disk)
    ax = fig.add_subplot(gs[1, 1])
    ax.hexbin(dAv, bAv, gridsize=50, cmap="Oranges", mincnt=1, extent=[0, 4, 0, 4])
    ax.plot([0, 4], [0, 4], "k-", lw=0.8, zorder=0)
    ax.set_xlabel(r"$A_V$ disk (mag)")
    ax.set_ylabel(r"$A_V$ bulge (mag)")
    ax.set_title(f"Dust attenuation (N={len(bAv)})")

    # (1,2) Metallicity bulge vs disk
    ax = fig.add_subplot(gs[1, 2])
    ax.hexbin(dZ, bZ, gridsize=50, cmap="Purples", mincnt=1)
    ax.plot([bZ.min(), bZ.max()], [bZ.min(), bZ.max()], "k-", lw=0.8)
    ax.set_xlabel(r"$\log Z/Z_\odot$ disk")
    ax.set_ylabel(r"$\log Z/Z_\odot$ bulge")
    ax.set_title("Metallicity")

    # (2,0) Bulge mass fraction distribution
    ax = fig.add_subplot(gs[2, 0])
    bins2 = np.linspace(0, 1, 50)
    ax.hist(nss_f_bulge, bins=bins2, color="grey", alpha=0.8)
    ax.set_xlabel(r"$M_\mathrm{bulge}/(M_\mathrm{bulge}+M_\mathrm{disk})$")
    ax.set_ylabel("Count")
    ax.set_title("NSS stellar bulge fraction")
    med_f = np.median(nss_f_bulge)
    ax.axvline(med_f, color="r", lw=1.2, label=f"median={med_f:.2f}")
    ax.legend(fontsize=8)

    # (2,1) Bulge log_mass vs redshift vs disk
    ax = fig.add_subplot(gs[2, 1])
    ax.hexbin(all_bZ, all_bM, gridsize=40, cmap="Reds",  mincnt=1, alpha=0.85)
    ax.hexbin(all_dZ, all_dM, gridsize=40, cmap="Blues", mincnt=1, alpha=0.7)
    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_BULGE, alpha=0.8, label="Bulge"),
                       Patch(color=C_DISK,  alpha=0.8, label="Disk")], fontsize=8)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$\log M_\star / M_\odot$")
    ax.set_title("Mass vs redshift by component")

    # (2,2) Av vs redshift
    ax = fig.add_subplot(gs[2, 2])
    all_bAv = Bmed[:, i_av]
    all_dAv = Dmed[:, i_av]
    ax.hexbin(all_bZ, all_bAv, gridsize=40, cmap="Reds",  mincnt=1, alpha=0.85)
    ax.hexbin(all_dZ, all_dAv, gridsize=40, cmap="Blues", mincnt=1, alpha=0.7)
    ax.legend(handles=[Patch(color=C_BULGE, alpha=0.8, label="Bulge"),
                       Patch(color=C_DISK,  alpha=0.8, label="Disk")], fontsize=8)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$A_V$ (mag)")
    ax.set_title("Dust attenuation vs redshift")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Per-parameter posterior widths (violin)
# ---------------------------------------------------------------------------

def fig_violin(B: dict, D: dict, out_path: Path, dpi: int = 150) -> None:
    bm = B["nss_n_dead"] > 50
    dm = D["nss_n_dead"] > 50

    Bwidth = (np.percentile(B["nss_samples"][bm], 84, axis=1)
              - np.percentile(B["nss_samples"][bm], 16, axis=1))  # (N, 12)
    Dwidth = (np.percentile(D["nss_samples"][dm], 84, axis=1)
              - np.percentile(D["nss_samples"][dm], 16, axis=1))

    # Show all free params (skip redshift = column 0, fixed)
    free = list(range(1, 12))
    labels = [PARAM_LABELS[PARAM_NAMES[i]] for i in free]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=False)
    fig.suptitle("Posterior constraint widths (84–16% CI)", fontsize=13)

    for ax, data, width, comp, c in [
        (axes[0], B, Bwidth, "Bulge", C_BULGE),
        (axes[1], D, Dwidth, "Disk",  C_DISK),
    ]:
        vdata = [width[:, i] for i in free]
        # Clip each to 1–99th pctile for violin stability
        vdata_clipped = [v[v < np.percentile(v, 99)] for v in vdata]
        parts = ax.violinplot(vdata_clipped, positions=range(len(free)),
                              showmedians=True, showextrema=False, widths=0.7)
        for pc in parts["bodies"]:
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks(range(len(free)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("84–16% CI width")
        ax.set_title(f"{comp} (N={len(data['nss_logZ'][bm if comp=='Bulge' else dm])})")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic plots for NSS bulge+disk fits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--work-dir", default="/cosma7/data/dp276/dc-harv3/work/outputs/nss_bulge_disk",
        help="Directory containing per-worker subdirs (w1/, w2/, ...).",
    )
    parser.add_argument(
        "--workers", type=int, nargs="+", default=list(range(1, 7)),
        help="Worker indices to load.",
    )
    parser.add_argument(
        "--catalogue",
        default="/cosma7/data/dp276/dc-harv3/work/catalogs/fluxes_bulge_disk_C25.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/cosma7/data/dp276/dc-harv3/work/outputs/nss_bulge_disk/diagnostics",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading workers {args.workers} …")
    B = load_workers(work_dir, args.workers, "bulge")
    D = load_workers(work_dir, args.workers, "disk")
    print(f"Total: {len(B['galaxy_id'])} bulge, {len(D['galaxy_id'])} disk galaxies")

    print("\nFigure 1: Run QC …")
    fig_qc(B, D, out_dir / "fig1_qc.png", dpi=args.dpi)

    print("Figure 2: Parameter distributions …")
    fig_params(B, D, out_dir / "fig2_params.png", dpi=args.dpi)

    print("Figure 3: Science plots …")
    fig_science(B, D, Path(args.catalogue), out_dir / "fig3_science.png", dpi=args.dpi)

    print("Figure 4: Violin constraint widths …")
    fig_violin(B, D, out_dir / "fig4_violin.png", dpi=args.dpi)

    print(f"\nAll done. Figures in {out_dir}")


if __name__ == "__main__":
    main()
