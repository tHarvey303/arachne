#!/usr/bin/env python3
r"""Validate a trained ParrotEmulator against a synference HDF5 library.

Produces:
- Console table: per-band bias (Δμ), scatter (σ), and 95th-percentile
  absolute error in arsinh-magnitude units.
- PNG figure: per-band violin plots of residuals (saved to ``--output-dir``).
- PNG figure: predicted vs true scatter plot for a random subset.
- HDF5 residuals file for downstream analysis (saved to ``--output-dir``).

Usage
-----
::

    python scripts/validate_parrot_emulator.py \
        --emulator  parrot_emulator.eqx \
        --library   /path/to/grid.hdf5 \
        --output-dir validation/

Run ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path



def parse_args(argv=None):
    """Parse command-line arguments for validation."""
    p = argparse.ArgumentParser(
        description="Validate a ParrotEmulator on held-out synference library data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--emulator", required=True, help="Path to trained .eqx checkpoint.")
    p.add_argument("--library", required=True, help="Path to synference HDF5 library.")
    p.add_argument(
        "--output-dir",
        default="outputs/validation",
        help="Directory for plots and residual HDF5.",
    )
    p.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="SPS parameter names to validate against. "
        "Defaults to the names embedded in the checkpoint.",
    )
    p.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help="Band names to validate against. "
        "Defaults to the names embedded in the checkpoint.",
    )
    p.add_argument(
        "--n-val",
        type=int,
        default=50_000,
        help="Number of samples to use for validation (random subset of library).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for subsetting.")
    p.add_argument(
        "--scatter-n",
        type=int,
        default=5_000,
        help="Number of points for per-band scatter plots.",
    )
    return p.parse_args(argv)


def main(argv=None):  # noqa: C901
    """Run ParrotEmulator validation and produce diagnostic plots."""
    args = parse_args(argv)

    import h5py
    import jax.numpy as jnp
    import numpy as np

    from arachne.emulator.parrot_emulator import (
        ParrotEmulator,
        _flux_to_asinh_mag_np,
        _read_band_names,
        _read_param_names,
        _select_indices,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load emulator (self-contained — no architecture args required)
    # ------------------------------------------------------------------
    emulator = ParrotEmulator.load(args.emulator)
    print(f"Loaded emulator: {len(emulator.param_names)} params, {len(emulator.band_names)} bands")

    # Allow CLI overrides; default to what the checkpoint already knows.
    params = args.params if args.params is not None else emulator.param_names
    bands = args.bands if args.bands is not None else emulator.band_names

    # ------------------------------------------------------------------
    # Load library subset
    # ------------------------------------------------------------------
    with h5py.File(args.library, "r") as f:
        raw_params = f["Grid/Parameters"][()]
        raw_phot = f["Grid/Photometry"][()]
        lib_param_names = _read_param_names(f)
        lib_band_names = _read_band_names(f)

    param_indices = _select_indices(params, lib_param_names, "parameter")
    band_indices = _select_indices(bands, lib_band_names, "band")

    params_all = raw_params[param_indices, :].T.astype(np.float32)
    phot_all = raw_phot[band_indices, :].T.astype(np.float32)

    # Keep finite-parameter rows only
    valid = np.all(np.isfinite(params_all), axis=1)
    params_all = params_all[valid]
    phot_all = phot_all[valid]

    rng = np.random.default_rng(args.seed)
    n = min(args.n_val, len(params_all))
    idx = rng.choice(len(params_all), size=n, replace=False)
    params_val = params_all[idx]
    phot_val = phot_all[idx]

    print(f"Validation set: {n} samples")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    params_jax = jnp.array(params_val)
    pred_flux = np.asarray(emulator.predict(params_jax))  # (N, B)

    # ------------------------------------------------------------------
    # Compute residuals in arsinh magnitude space
    # ------------------------------------------------------------------
    true_mag = _flux_to_asinh_mag_np(phot_val)   # (N, B)
    pred_mag = _flux_to_asinh_mag_np(pred_flux)   # (N, B)
    residuals = pred_mag - true_mag               # (N, B); positive = over-predicted

    # ------------------------------------------------------------------
    # Per-band statistics
    # ------------------------------------------------------------------
    bias = np.nanmean(residuals, axis=0)          # (B,)
    scatter = np.nanstd(residuals, axis=0)        # (B,)
    p95 = np.nanpercentile(np.abs(residuals), 95, axis=0)  # (B,)

    print("\n{:40s}  {:>8s}  {:>8s}  {:>10s}".format("Band", "bias", "sigma", "95th |err|"))
    print("-" * 72)
    for i, bname in enumerate(bands):
        print(f"{bname:40s}  {bias[i]:+8.4f}  {scatter[i]:8.4f}  {p95[i]:10.4f}")

    print(f"\nMedian |bias|  : {np.median(np.abs(bias)):.4f} asinh-mag")
    print(f"Median scatter : {np.median(scatter):.4f} asinh-mag")
    print(f"Median 95th p95: {np.median(p95):.4f} asinh-mag")

    # ------------------------------------------------------------------
    # Save residuals HDF5
    # ------------------------------------------------------------------
    resid_path = output_dir / "residuals.h5"
    with h5py.File(resid_path, "w") as fout:
        fout.create_dataset("residuals_asinh_mag", data=residuals.astype(np.float32))
        fout.create_dataset("true_asinh_mag", data=true_mag.astype(np.float32))
        fout.create_dataset("pred_asinh_mag", data=pred_mag.astype(np.float32))
        fout.create_dataset("true_flux_nJy", data=phot_val.astype(np.float32))
        fout.create_dataset("pred_flux_nJy", data=pred_flux.astype(np.float32))
        fout.create_dataset("params", data=params_val.astype(np.float32))
        fout.attrs["band_names"] = np.array(bands, dtype="S")
        fout.attrs["param_names"] = np.array(params, dtype="S")
        fout.attrs["bias"] = bias.astype(np.float32)
        fout.attrs["scatter"] = scatter.astype(np.float32)
        fout.attrs["p95_abs_err"] = p95.astype(np.float32)
    print(f"\nResiduals saved to {resid_path}")

    # ------------------------------------------------------------------
    # Plots (optional: skip gracefully if matplotlib not installed)
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return 0

    n_bands = len(bands)

    # ---- Figure 1: per-band bias and scatter summary bar chart ----
    fig, axes = plt.subplots(2, 1, figsize=(max(12, n_bands * 0.4), 8), sharex=True)
    x = np.arange(n_bands)
    axes[0].bar(x, bias, color="steelblue", alpha=0.8)
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("Bias (arsinh-mag)")
    axes[0].set_title("Per-band emulator bias")
    axes[1].bar(x, scatter, color="darkorange", alpha=0.8)
    axes[1].set_ylabel("Scatter σ (arsinh-mag)")
    axes[1].set_title("Per-band emulator scatter")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [b.split("/")[-1] for b in bands], rotation=45, ha="right", fontsize=7
        )
    fig.tight_layout()
    bias_path = output_dir / "bias_scatter.png"
    fig.savefig(bias_path, dpi=150)
    plt.close(fig)
    print(f"Bias/scatter figure saved to {bias_path}")

    # ---- Figure 2: predicted vs true scatter for a subset of bands ----
    highlight_bands = [
        b for b in [
            "JWST/NIRCam.F200W", "JWST/NIRCam.F277W", "JWST/NIRCam.F444W",
            "HST/ACS_WFC.F814W", "CTIO/DECam.r",
        ]
        if b in bands
    ][:5]
    if not highlight_bands:
        highlight_bands = bands[:min(5, n_bands)]

    n_scatter = min(args.scatter_n, n)
    scatter_idx = rng.choice(n, size=n_scatter, replace=False)

    n_cols = len(highlight_bands)
    fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes2 = [axes2]

    for ax, bname in zip(axes2, highlight_bands):
        bi = bands.index(bname)
        t = true_mag[scatter_idx, bi]
        pr = pred_mag[scatter_idx, bi]
        finite = np.isfinite(t) & np.isfinite(pr)
        ax.scatter(t[finite], pr[finite], s=1, alpha=0.3, rasterized=True)
        lim = (min(t[finite].min(), pr[finite].min()), max(t[finite].max(), pr[finite].max()))
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("True (arsinh-mag)")
        ax.set_ylabel("Predicted (arsinh-mag)")
        ax.set_title(bname.split("/")[-1])
    fig2.suptitle("Predicted vs True (arsinh-mag)", y=1.02)
    fig2.tight_layout()
    scatter_path = output_dir / "pred_vs_true.png"
    fig2.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Scatter plot saved to {scatter_path}")

    # ---- Figure 3: residual histograms per band ----
    n_rows = (n_bands + 7) // 8
    fig3, axes3 = plt.subplots(n_rows, 8, figsize=(24, 3 * n_rows))
    axes3_flat = axes3.flatten() if hasattr(axes3, "flatten") else [axes3]
    for i, bname in enumerate(bands):
        ax = axes3_flat[i]
        r = residuals[:, i]
        r = r[np.isfinite(r)]
        ax.hist(r, bins=80, density=True, histtype="step", color="steelblue")
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.set_title(bname.split("/")[-1], fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(n_bands, len(axes3_flat)):
        axes3_flat[j].set_visible(False)
    fig3.suptitle("Residual distributions (pred − true, arsinh-mag)")
    fig3.tight_layout()
    hist_path = output_dir / "residual_histograms.png"
    fig3.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close(fig3)
    print(f"Residual histograms saved to {hist_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
