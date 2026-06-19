#!/usr/bin/env python3
r"""Detailed diagnostic analysis of a trained ParrotEmulator.

Unlike ``validate_parrot_emulator.py`` (overall per-band statistics), this
script locates *where* in the SPS parameter space the emulator performs worst
and quantifies the gap relative to the Parrot / Speculator <1% flux-error
benchmark.

Outputs (all saved to ``--output-dir``)
----------------------------------------
fig1_error_summary.png
    Per-band abs % flux error: median, 84th-, 95th-percentile bars with a
    1 % reference line.  Also shows band-averaged arsinh-mag residuals.
fig2_param_profiles.png
    1D per-parameter error profiles: for each SPS input, median abs % flux
    error binned along the parameter range, with 16th–84th shading and a
    sample-count histogram.  Directly shows which parts of parameter space
    are worst.
fig3_2d_heatmaps.png
    2D error heatmaps: the two highest-variance parameters, plus the most
    important parameter paired against every other.  Reveals interaction
    effects (e.g. high-z × high-dust).
fig4_flux_level.png
    Abs % flux error as a function of log10(true flux) per band.  Shows
    whether errors blow up in the dropout / faint-source regime.
fig5_outliers.png
    Parameter distributions for the worst 5 % of samples (by band-averaged
    abs % flux error) versus the full validation set.  Shows where
    catastrophic failures cluster.
fig6_band_param_correlation.png
    Spearman rank-correlation matrix (abs arsinh-mag residual per band ×
    each input parameter).  Identifies which parameter drives errors in which
    photometric band.
diagnostics.h5
    HDF5 with all intermediate arrays for further analysis in a notebook.

Usage
-----
::

    python scripts/diagnose_parrot_emulator.py \
        --emulator  outputs/emulators/parrot_emulator_optuna.eqx \
        --library   /path/to/grid.hdf5 \
        --output-dir outputs/diagnostics/

Run ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Detailed ParrotEmulator diagnostic: error as a function of SPS parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--emulator", required=True, help="Path to trained .eqx checkpoint.")
    p.add_argument("--library", required=True, help="Path to synference HDF5 library.")
    p.add_argument(
        "--output-dir",
        default="outputs/diagnostics",
        help="Directory for diagnostic figures and HDF5.",
    )
    p.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="SPS parameter names.  Defaults to those embedded in the checkpoint.",
    )
    p.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help="Band names.  Defaults to those embedded in the checkpoint.",
    )
    p.add_argument(
        "--n-val",
        type=int,
        default=50_000,
        help="Number of samples to use for diagnostics.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for subsetting.")
    p.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of quantile bins for 1D parameter profiles and 2D heatmaps.",
    )
    p.add_argument(
        "--outlier-percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for 'outlier' samples in Fig 5.",
    )
    p.add_argument(
        "--min-flux",
        type=float,
        default=1.0,
        help="Minimum flux (nJy) used as the denominator when computing %% flux error, "
        "preventing division by near-zero for dropout sources.",
    )
    p.add_argument(
        "--detect-threshold",
        type=float,
        default=0.0,
        help="Only include pixels with true_flux > this value (nJy) in %% flux error "
        "calculations.  Set > 0 to exclude dropout / non-detections (e.g. 5.0).",
    )
    p.add_argument(
        "--n-heatmap-pairs",
        type=int,
        default=6,
        help="Number of 2D heatmap parameter pairs to plot in Fig 3.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_and_predict(args, emulator):
    """Load library subset, run predictions, return error arrays."""
    import h5py
    import jax.numpy as jnp

    from arachne.emulator.parrot_emulator import (
        _flux_to_asinh_mag_np,
        _read_band_names,
        _read_param_names,
        _select_indices,
    )

    param_names = args.params if args.params is not None else emulator.param_names
    band_names = args.bands if args.bands is not None else emulator.band_names

    with h5py.File(args.library, "r") as f:
        raw_params = f["Grid/Parameters"][()]
        raw_phot = f["Grid/Photometry"][()]
        lib_param_names = _read_param_names(f)
        lib_band_names = _read_band_names(f)

    param_indices = _select_indices(param_names, lib_param_names, "parameter")
    band_indices = _select_indices(band_names, lib_band_names, "band")

    params_all = raw_params[param_indices, :].T.astype(np.float32)
    phot_all = raw_phot[band_indices, :].T.astype(np.float32)

    valid = np.all(np.isfinite(params_all), axis=1)
    params_all = params_all[valid]
    phot_all = phot_all[valid]

    rng = np.random.default_rng(args.seed)
    n = min(args.n_val, len(params_all))
    idx = rng.choice(len(params_all), size=n, replace=False)
    params_val = params_all[idx]
    phot_val = phot_all[idx]

    # Apply the same flux floor used during training so validation is apples-to-apples.
    flux_floor = emulator._flux_floor
    if flux_floor > 0:
        phot_val = np.where(phot_val < flux_floor, 0.0, phot_val)

    print(
        f"Validation set: {n} samples, {len(param_names)} params, {len(band_names)} bands"
        f" (flux_floor={flux_floor:.2e} nJy applied to reference)"
    )

    pred_flux = np.asarray(emulator.predict(jnp.array(params_val)))  # (N, B)

    # arsinh-mag residuals — use emulator's own mu0 to match training encoding
    mu0 = emulator._asinh_mu0
    true_mag = _flux_to_asinh_mag_np(phot_val, mu0=mu0)
    pred_mag = _flux_to_asinh_mag_np(pred_flux, mu0=mu0)
    resid_mag = pred_mag - true_mag  # positive = over-predicted

    # % flux error (mask near-zero denominators)
    denom = np.where(np.abs(phot_val) >= args.min_flux, np.abs(phot_val), args.min_flux)
    frac_err = (pred_flux - phot_val) / denom  # (N, B)
    abs_pct = np.abs(frac_err) * 100.0  # (N, B) in %

    # Optionally mask non-detected sources (true_flux below threshold)
    if args.detect_threshold > 0:
        detected = phot_val > args.detect_threshold  # (N, B)
        abs_pct_detected = np.where(detected, abs_pct, np.nan)
    else:
        abs_pct_detected = abs_pct

    return {
        "params_val": params_val,
        "phot_val": phot_val,
        "pred_flux": pred_flux,
        "true_mag": true_mag,
        "pred_mag": pred_mag,
        "resid_mag": resid_mag,
        "abs_resid_mag": np.abs(resid_mag),
        "abs_pct": abs_pct,
        "abs_pct_detected": abs_pct_detected,
        "param_names": param_names,
        "band_names": band_names,
        "n": n,
        "rng": rng,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _print_summary(data, args):
    """Print headline statistics to console."""
    abs_pct = data["abs_pct_detected"]
    resid_mag = data["resid_mag"]
    param_names = data["param_names"]
    band_names = data["band_names"]

    med_pct = float(np.nanmedian(abs_pct))
    p84_pct = float(np.nanpercentile(abs_pct, 84))
    p95_pct = float(np.nanpercentile(abs_pct, 95))

    print(f"\n{'='*68}")
    print(f"  Abs %% flux error (all bands / all samples above {args.detect_threshold} nJy)")
    print(f"  Median: {med_pct:.2f}%   84th-pct: {p84_pct:.2f}%   95th-pct: {p95_pct:.2f}%")
    print(f"  Target (Parrot paper): < 1%")
    print(f"{'='*68}\n")

    print(f"{'Band':45s}  {'med%err':>8s}  {'p84%err':>8s}  {'bias(mag)':>10s}  {'σ(mag)':>8s}")
    print("-" * 86)
    for i, bname in enumerate(band_names):
        col_pct = abs_pct[:, i]
        col_res = resid_mag[:, i]
        med = np.nanmedian(col_pct)
        p84 = np.nanpercentile(col_pct, 84)
        bias = np.nanmean(col_res)
        sig = np.nanstd(col_res)
        flag = "  <<" if med > 1.0 else ""
        print(f"{bname:45s}  {med:8.2f}  {p84:8.2f}  {bias:+10.4f}  {sig:8.4f}{flag}")

    print()

    # Per-parameter: which quantile range has highest error?

    params_val = data["params_val"]
    # band-averaged abs % error per sample
    sample_err = np.nanmean(abs_pct, axis=1)
    print("Per-parameter worst region (by band-averaged abs %% flux error):")
    print(f"{'Parameter':30s}  {'Worst range':35s}  {'Median err in worst bin':>24s}")
    print("-" * 96)
    for pi, pname in enumerate(param_names):
        col = params_val[:, pi]
        finite = np.isfinite(col) & np.isfinite(sample_err)
        if not finite.any():
            continue
        _, bin_edges = np.histogram(col[finite], bins=10)
        bin_idx = np.digitize(col[finite], bin_edges[:-1]) - 1
        bin_errs = [sample_err[finite][bin_idx == b] for b in range(10)]
        bin_meds = [np.nanmedian(e) if len(e) > 0 else np.nan for e in bin_errs]
        worst_bin = int(np.nanargmax(bin_meds))
        lo, hi = bin_edges[worst_bin], bin_edges[worst_bin + 1]
        print(
            f"{pname:30s}  [{lo:+.3f}, {hi:+.3f}]"
            f"{'':8s}{np.nanmax(bin_meds):>24.2f}%"
        )


# ---------------------------------------------------------------------------
# Figure 1: Per-band error summary
# ---------------------------------------------------------------------------


def _fig1_error_summary(data, output_dir):
    """Bar chart of per-band % flux error statistics with 1% reference line."""
    abs_pct = data["abs_pct_detected"]
    resid_mag = data["resid_mag"]
    band_names = data["band_names"]
    n_bands = len(band_names)
    short_names = [b.split("/")[-1] for b in band_names]

    med_pct = np.nanmedian(abs_pct, axis=0)
    p84_pct = np.nanpercentile(abs_pct, 84, axis=0)
    p95_pct = np.nanpercentile(abs_pct, 95, axis=0)
    bias_mag = np.nanmean(resid_mag, axis=0)
    sig_mag = np.nanstd(resid_mag, axis=0)

    x = np.arange(n_bands)
    fig, axes = plt.subplots(2, 2, figsize=(max(14, n_bands * 0.55), 9))

    # Row 0: % flux error
    ax0, ax1 = axes[0]
    ax0.bar(x, med_pct, color="steelblue", alpha=0.8, label="Median")
    ax0.bar(x, p84_pct - med_pct, bottom=med_pct, color="orange", alpha=0.6, label="↑ 84th pct")
    ax0.axhline(1.0, color="red", ls="--", lw=1.2, label="1% target")
    ax0.set_ylabel("Abs % flux error")
    ax0.set_title("Median + 84th-pct abs % flux error per band")
    ax0.legend(fontsize=8)

    ax1.bar(x, p95_pct, color="tomato", alpha=0.8)
    ax1.axhline(1.0, color="red", ls="--", lw=1.2, label="1% target")
    ax1.set_ylabel("Abs % flux error (95th pct)")
    ax1.set_title("95th-percentile abs % flux error per band")
    ax1.legend(fontsize=8)

    # Row 1: arsinh-mag residuals
    ax2, ax3 = axes[1]
    ax2.bar(x, bias_mag, color=np.where(bias_mag >= 0, "steelblue", "tomato"), alpha=0.8)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_ylabel("Bias (arsinh-mag)")
    ax2.set_title("Per-band bias (pred − true, arsinh-mag)")

    ax3.bar(x, sig_mag, color="darkorange", alpha=0.8)
    ax3.set_ylabel("Scatter σ (arsinh-mag)")
    ax3.set_title("Per-band scatter")

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=55, ha="right", fontsize=7)

    fig.suptitle("Emulator error summary — per band", fontsize=13, y=1.01)
    fig.tight_layout()
    path = output_dir / "fig1_error_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 1] Error summary → {path}")


# ---------------------------------------------------------------------------
# Figure 2: Per-parameter 1D error profiles
# ---------------------------------------------------------------------------


def _fig2_param_profiles(data, args, output_dir):
    """1D per-parameter error profiles with sample-count histogram."""

    params_val = data["params_val"]
    abs_pct = data["abs_pct_detected"]
    param_names = data["param_names"]
    band_names = data["band_names"]
    n_params = len(param_names)
    n_bins = args.n_bins

    # Band-averaged abs % error per sample
    sample_err = np.nanmean(abs_pct, axis=1)

    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for pi, pname in enumerate(param_names):
        ax = axes_flat[pi]
        col = params_val[:, pi]

        finite = np.isfinite(col) & np.isfinite(sample_err)
        c_f = col[finite]
        e_f = sample_err[finite]

        # Use quantile-based bins so each bin has roughly equal count
        bin_edges = np.quantile(c_f, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # collapse duplicates for discrete params
        actual_bins = len(bin_edges) - 1
        if actual_bins < 2:
            ax.text(0.5, 0.5, "insufficient range", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(pname, fontsize=9)
            continue

        bin_idx = np.digitize(c_f, bin_edges[:-1]) - 1
        bin_idx = np.clip(bin_idx, 0, actual_bins - 1)

        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        med_e = np.full(actual_bins, np.nan)
        p16_e = np.full(actual_bins, np.nan)
        p84_e = np.full(actual_bins, np.nan)
        counts = np.zeros(actual_bins, dtype=int)

        for b in range(actual_bins):
            mask = bin_idx == b
            vals = e_f[mask]
            if len(vals) > 0:
                counts[b] = len(vals)
                med_e[b] = np.nanmedian(vals)
                p16_e[b] = np.nanpercentile(vals, 16)
                p84_e[b] = np.nanpercentile(vals, 84)

        finite_bins = np.isfinite(med_e)

        # Also compute per-band profiles to show the spread across bands
        ax_twin = ax.twinx()
        ax_twin.bar(
            centers, counts, width=np.diff(bin_edges), color="lightgray",
            alpha=0.5, label="N samples", zorder=0,
        )
        ax_twin.set_ylabel("N samples", fontsize=7, color="gray")
        ax_twin.tick_params(axis="y", labelcolor="gray", labelsize=6)

        ax.fill_between(
            centers[finite_bins],
            p16_e[finite_bins],
            p84_e[finite_bins],
            alpha=0.25,
            color="steelblue",
            label="16–84th pct",
            zorder=2,
        )
        ax.plot(
            centers[finite_bins], med_e[finite_bins],
            color="steelblue", lw=2, marker="o", ms=4, label="Median", zorder=3,
        )
        ax.axhline(1.0, color="red", ls="--", lw=1, label="1% target", zorder=4)
        ax.set_xlabel(pname, fontsize=8)
        ax.set_ylabel("Abs % flux error", fontsize=8)
        ax.set_title(pname, fontsize=9)
        ax.tick_params(labelsize=7)
        if pi == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "1D per-parameter error profiles (band-averaged abs % flux error)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    path = output_dir / "fig2_param_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 2] 1D parameter profiles → {path}")

    # Return per-parameter importance (variance of per-bin medians) for Fig 3
    importances = {}
    for pi, pname in enumerate(param_names):
        col = params_val[:, pi]
        finite = np.isfinite(col) & np.isfinite(sample_err)
        c_f, e_f = col[finite], sample_err[finite]
        bin_edges = np.quantile(c_f, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1
        if actual_bins < 2:
            importances[pname] = 0.0
            continue
        bin_idx = np.clip(np.digitize(c_f, bin_edges[:-1]) - 1, 0, actual_bins - 1)
        meds = [np.nanmedian(e_f[bin_idx == b]) for b in range(actual_bins)]
        importances[pname] = float(np.nanvar(meds))
    return importances


# ---------------------------------------------------------------------------
# Figure 3: 2D error heatmaps
# ---------------------------------------------------------------------------


def _fig3_2d_heatmaps(data, args, importances, output_dir):
    """2D error heatmaps: most-important parameter vs each other parameter."""
    params_val = data["params_val"]
    abs_pct = data["abs_pct_detected"]
    param_names = data["param_names"]
    n_bins = max(8, args.n_bins // 2)  # coarser grid for 2D

    sample_err = np.nanmean(abs_pct, axis=1)

    # Rank parameters by importance
    ranked = sorted(importances, key=importances.get, reverse=True)
    pivot = ranked[0]
    pivot_idx = param_names.index(pivot)

    # Pairs: pivot vs all others, plus top-2 vs top-3 if budget allows
    other_params = [p for p in ranked if p != pivot]
    pairs = [(pivot, p) for p in other_params]

    # Add the second vs third most important pair
    if len(ranked) >= 3:
        pairs.append((ranked[1], ranked[2]))

    pairs = pairs[: args.n_heatmap_pairs]
    n_pairs = len(pairs)

    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes_flat = np.array(axes).flatten() if n_pairs > 1 else [axes]

    for k, (pname_x, pname_y) in enumerate(pairs):
        ax = axes_flat[k]
        px_idx = param_names.index(pname_x)
        py_idx = param_names.index(pname_y)
        px = params_val[:, px_idx]
        py = params_val[:, py_idx]
        finite = np.isfinite(px) & np.isfinite(py) & np.isfinite(sample_err)

        px_f, py_f, e_f = px[finite], py[finite], sample_err[finite]

        x_edges = np.quantile(px_f, np.linspace(0, 1, n_bins + 1))
        y_edges = np.quantile(py_f, np.linspace(0, 1, n_bins + 1))
        x_edges = np.unique(x_edges)
        y_edges = np.unique(y_edges)
        nx, ny = len(x_edges) - 1, len(y_edges) - 1

        grid = np.full((ny, nx), np.nan)
        xi = np.clip(np.digitize(px_f, x_edges[:-1]) - 1, 0, nx - 1)
        yi = np.clip(np.digitize(py_f, y_edges[:-1]) - 1, 0, ny - 1)
        for i in range(nx):
            for j in range(ny):
                mask = (xi == i) & (yi == j)
                if mask.sum() >= 5:
                    grid[j, i] = np.nanmedian(e_f[mask])

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap="hot_r",
            vmin=0,
        )
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("Median abs % flux error", fontsize=7)
        cb.ax.tick_params(labelsize=7)
        ax.set_xlabel(pname_x, fontsize=9)
        ax.set_ylabel(pname_y, fontsize=9)
        short_x = pname_x.replace("sfh_quantile_", "sfhq").replace("log10metallicity", "logZ")
        short_y = pname_y.replace("sfh_quantile_", "sfhq").replace("log10metallicity", "logZ")
        ax.set_title(f"{short_x}  ×  {short_y}", fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(n_pairs, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"2D error heatmaps — most important: '{pivot}'\n"
        "(band-averaged median abs % flux error per 2D bin)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    path = output_dir / "fig3_2d_heatmaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 3] 2D heatmaps → {path}")


# ---------------------------------------------------------------------------
# Figure 4: Error vs flux level (dropout diagnosis)
# ---------------------------------------------------------------------------


def _fig4_flux_level(data, output_dir):
    """Abs % flux error vs log10(true flux) per band."""

    phot_val = data["phot_val"]
    abs_pct = data["abs_pct"]  # use unmasked here to show dropout behaviour
    band_names = data["band_names"]
    n_bands = len(band_names)

    # Select a representative subset of bands for clarity
    highlight = [
        b for b in [
            "JWST/NIRCam.F200W", "JWST/NIRCam.F277W", "JWST/NIRCam.F444W",
            "HST/ACS_WFC.F814W", "CTIO/DECam.r", "JWST/MIRI.F560W",
        ]
        if b in band_names
    ]
    if not highlight:
        highlight = band_names[:: max(1, n_bands // 6)][:6]

    n_cols = len(highlight)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for ax, bname in zip(axes, highlight):
        bi = band_names.index(bname)
        flux_col = phot_val[:, bi]
        err_col = abs_pct[:, bi]

        positive = flux_col > 0
        log_flux = np.log10(np.where(positive, flux_col, np.nan))

        finite = np.isfinite(log_flux) & np.isfinite(err_col)
        lf_f, e_f = log_flux[finite], err_col[finite]

        # Bin by log flux
        n_bins = 30
        bin_edges = np.linspace(np.nanmin(lf_f), np.nanmax(lf_f), n_bins + 1)
        bin_idx = np.clip(np.digitize(lf_f, bin_edges[:-1]) - 1, 0, n_bins - 1)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        med_e = np.full(n_bins, np.nan)
        p84_e = np.full(n_bins, np.nan)

        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() >= 5:
                med_e[b] = np.nanmedian(e_f[mask])
                p84_e[b] = np.nanpercentile(e_f[mask], 84)

        finite_b = np.isfinite(med_e)
        ax.fill_between(
            centers[finite_b], med_e[finite_b], p84_e[finite_b],
            alpha=0.25, color="steelblue",
        )
        ax.plot(centers[finite_b], med_e[finite_b], color="steelblue", lw=1.8, label="Median")
        ax.plot(centers[finite_b], p84_e[finite_b], color="steelblue", lw=1, ls="--", label="84th pct")
        ax.axhline(1.0, color="red", ls="--", lw=1, label="1% target")

        # Shade dropout regime (flux < 5 nJy, log < ~0.7)
        dropout_log = np.log10(5.0)
        ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] < dropout_log else dropout_log - 2,
                   dropout_log, color="orange", alpha=0.10, label="faint (<5 nJy)")

        ax.set_xlabel("log₁₀(true flux / nJy)", fontsize=9)
        ax.set_ylabel("Abs % flux error", fontsize=9)
        ax.set_title(bname.split("/")[-1], fontsize=9)
        ax.tick_params(labelsize=7)
        if ax is axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle("Abs % flux error vs log₁₀(true flux) — dropout regime on the left", fontsize=11)
    fig.tight_layout()
    path = output_dir / "fig4_flux_level.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 4] Flux-level error → {path}")


# ---------------------------------------------------------------------------
# Figure 5: Outlier parameter distributions
# ---------------------------------------------------------------------------


def _fig5_outliers(data, args, output_dir):
    """Parameter distributions for worst-performing samples vs. full set."""


    params_val = data["params_val"]
    abs_pct = data["abs_pct_detected"]
    param_names = data["param_names"]
    n_params = len(param_names)

    sample_err = np.nanmean(abs_pct, axis=1)
    threshold = np.nanpercentile(sample_err, args.outlier_percentile)
    outlier_mask = sample_err >= threshold
    n_out = outlier_mask.sum()

    print(
        f"\nOutlier analysis: {n_out} samples ({100*n_out/len(sample_err):.1f}%) "
        f"with band-avg abs %% error ≥ {threshold:.2f}%"
    )

    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes_flat = np.array(axes).flatten()

    for pi, pname in enumerate(param_names):
        ax = axes_flat[pi]
        col = params_val[:, pi]

        all_vals = col[np.isfinite(col)]
        out_vals = col[outlier_mask & np.isfinite(col)]
        if len(all_vals) == 0 or len(out_vals) == 0:
            ax.set_visible(False)
            continue

        lo, hi = all_vals.min(), all_vals.max()
        bins = np.linspace(lo, hi, 30)
        ax.hist(all_vals, bins=bins, density=True, histtype="step",
                color="steelblue", lw=1.5, label=f"All (N={len(all_vals)})")
        ax.hist(out_vals, bins=bins, density=True, histtype="stepfilled",
                color="tomato", alpha=0.45, label=f"Worst {100-args.outlier_percentile:.0f}% (N={len(out_vals)})")

        # KS statistic
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(all_vals, out_vals)
        ax.set_title(f"{pname}\nKS={ks_stat:.3f}, p={ks_p:.2e}", fontsize=8)
        ax.set_xlabel(pname, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if pi == 0:
            ax.legend(fontsize=7)

    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Outlier analysis: parameter distributions\n"
        f"worst {100-args.outlier_percentile:.0f}% (band-avg abs % flux error "
        f"≥ {threshold:.2f}%) vs full validation set",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    path = output_dir / "fig5_outliers.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 5] Outlier analysis → {path}")


# ---------------------------------------------------------------------------
# Figure 6: Band × parameter Spearman correlation heatmap
# ---------------------------------------------------------------------------


def _fig6_correlation(data, output_dir):
    """Spearman correlation: |arsinh-mag residual| per band × each parameter."""
    from scipy.stats import spearmanr

    params_val = data["params_val"]
    abs_resid = data["abs_resid_mag"]  # (N, B)
    param_names = data["param_names"]
    band_names = data["band_names"]
    n_params = len(param_names)
    n_bands = len(band_names)
    short_bands = [b.split("/")[-1] for b in band_names]

    corr_matrix = np.zeros((n_params, n_bands))
    for pi in range(n_params):
        for bi in range(n_bands):
            col_p = params_val[:, pi]
            col_e = abs_resid[:, bi]
            finite = np.isfinite(col_p) & np.isfinite(col_e)
            if finite.sum() > 10:
                r, _ = spearmanr(col_p[finite], col_e[finite])
                corr_matrix[pi, bi] = r if np.isfinite(r) else 0.0

    fig, ax = plt.subplots(figsize=(max(10, n_bands * 0.45), max(4, n_params * 0.6)))
    im = ax.imshow(corr_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Spearman r", fontsize=9)

    ax.set_xticks(np.arange(n_bands))
    ax.set_xticklabels(short_bands, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n_params))
    ax.set_yticklabels(param_names, fontsize=8)
    ax.set_title(
        "Spearman rank correlation: |residual| per band × input parameter\n"
        "Red = parameter↑ → larger error; Blue = parameter↑ → smaller error",
        fontsize=10,
    )

    # Annotate cells with strongest correlations
    for pi in range(n_params):
        for bi in range(n_bands):
            r = corr_matrix[pi, bi]
            if abs(r) > 0.25:
                ax.text(bi, pi, f"{r:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(r) > 0.5 else "black")

    fig.tight_layout()
    path = output_dir / "fig6_band_param_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 6] Band×parameter correlation → {path}")

    # Print top correlations
    print("\nTop parameter–band correlations (|r| > 0.25):")
    pairs = []
    for pi in range(n_params):
        for bi in range(n_bands):
            r = corr_matrix[pi, bi]
            if abs(r) > 0.25:
                pairs.append((abs(r), r, param_names[pi], band_names[bi]))
    pairs.sort(reverse=True)
    for _, r, pname, bname in pairs[:20]:
        direction = "↑error" if r > 0 else "↓error"
        print(f"  {pname:30s}  ×  {bname:40s}  r={r:+.3f} ({direction})")


# ---------------------------------------------------------------------------
# Save HDF5
# ---------------------------------------------------------------------------


def _save_hdf5(data, importances, output_dir):
    """Save all diagnostic arrays to HDF5 for notebook follow-up."""
    import h5py

    path = output_dir / "diagnostics.h5"
    with h5py.File(path, "w") as f:
        for key in (
            "params_val", "phot_val", "pred_flux",
            "resid_mag", "abs_resid_mag", "abs_pct", "abs_pct_detected",
        ):
            f.create_dataset(key, data=data[key].astype(np.float32), compression="gzip")
        f.attrs["param_names"] = np.array(data["param_names"], dtype="S")
        f.attrs["band_names"] = np.array(data["band_names"], dtype="S")
        imp_grp = f.create_group("param_importance")
        for pname, val in importances.items():
            imp_grp.attrs[pname] = float(val)
    print(f"\nDiagnostic HDF5 → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    """Run the full diagnostic suite."""
    args = parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    from arachne.emulator.parrot_emulator import ParrotEmulator

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading emulator: {args.emulator}")
    emulator = ParrotEmulator.load(args.emulator)
    print(
        f"  {len(emulator.param_names)} params: {emulator.param_names}\n"
        f"  {len(emulator.band_names)} bands"
    )

    data = _load_and_predict(args, emulator)
    _print_summary(data, args)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping all plots.")
        return 1

    _fig1_error_summary(data, output_dir)
    importances = _fig2_param_profiles(data, args, output_dir)

    ranked = sorted(importances, key=importances.get, reverse=True)
    print(f"\nParameter importance ranking (variance of per-bin median error):")
    for rank, pname in enumerate(ranked, 1):
        print(f"  {rank}. {pname:30s}  var={importances[pname]:.4f}")

    _fig3_2d_heatmaps(data, args, importances, output_dir)
    _fig4_flux_level(data, output_dir)

    try:
        from scipy.stats import ks_2samp  # noqa: F401 (just check availability)
        _fig5_outliers(data, args, output_dir)
        _fig6_correlation(data, output_dir)
    except ImportError:
        print("scipy not installed — skipping outlier KS test and correlation figure.")

    _save_hdf5(data, importances, output_dir)

    print(f"\nAll diagnostics written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
