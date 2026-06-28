#!/usr/bin/env python3
"""Generate corner plots from NSS posterior samples stored in an HDF5 file.

Each galaxy gets one PNG: a corner plot of all 12 SPS parameters in their
physical units, with the median marked and parameter ranges labelled.

Usage
-----
    python scripts/plot_nss_corners.py results.hdf5 [--out-dir corners/]
    python scripts/plot_nss_corners.py results.hdf5 --galaxy-ids 0 5 12
    python scripts/plot_nss_corners.py results.hdf5 --n-galaxies 10
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import corner
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Nicer LaTeX labels and physical units for each SPS parameter.
_LABELS = {
    "redshift":            r"$z$",
    "log_mass":            r"$\log M_\star / M_\odot$",
    "slope":               r"$\delta$ (slope)",
    "fesc_lya":            r"$f_\mathrm{esc,Ly\alpha}$",
    "dust_bump_amplitude": r"$B_{2175}$",
    "log10metallicity":    r"$\log Z / Z_\odot$",
    "Av":                  r"$A_V$ (mag)",
    "logsfr_ratio_0":      r"$\log \mathrm{SFR}_0 / \mathrm{SFR}_1$",
    "logsfr_ratio_1":      r"$\log \mathrm{SFR}_1 / \mathrm{SFR}_2$",
    "logsfr_ratio_2":      r"$\log \mathrm{SFR}_2 / \mathrm{SFR}_3$",
    "logsfr_ratio_3":      r"$\log \mathrm{SFR}_3 / \mathrm{SFR}_4$",
    "logsfr_ratio_4":      r"$\log \mathrm{SFR}_4 / \mathrm{SFR}_5$",
}


def plot_corner(
    samples: np.ndarray,
    param_names: list[str],
    galaxy_id,
    logz: float,
    logz_err: float,
    ess: float,
    rhat_max: float,
    out_path: Path,
    dpi: int = 120,
) -> None:
    """Render and save a corner plot for one galaxy.

    Args:
        samples:     (S, P) array of posterior samples in physical space.
        param_names: list of P parameter name strings.
        galaxy_id:   catalogue identifier (int or str) for the title.
        logz:        log evidence point estimate.
        logz_err:    MC uncertainty on logZ.
        ess:         effective sample size.
        rhat_max:    worst split-R-hat across parameters.
        out_path:    output PNG path.
        dpi:         figure resolution.
    """
    labels = [_LABELS.get(p, p) for p in param_names]
    P = samples.shape[1]

    # Drop samples with any NaN/Inf (shouldn't happen but be safe).
    ok = np.isfinite(samples).all(axis=1)
    samples = samples[ok]
    if len(samples) < 10:
        print(f"  galaxy {galaxy_id}: only {len(samples)} finite samples — skipping.")
        return

    # Quantiles for the 1D marginals.
    quantiles = [0.16, 0.50, 0.84]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = corner.corner(
            samples,
            labels=labels,
            quantiles=quantiles,
            show_titles=True,
            title_fmt=".3g",
            title_kwargs={"fontsize": 7},
            label_kwargs={"fontsize": 8},
            smooth=1.0,
            smooth1d=1.0,
            plot_datapoints=False,
            fill_contours=True,
            levels=(0.68, 0.95),
            contourf_kwargs={"alpha": 0.6},
            color="#2060a0",
        )

    # Suptitle with per-galaxy diagnostics.
    rhat_str = f"{rhat_max:.3f}" if np.isfinite(rhat_max) else "nan"
    fig.suptitle(
        f"Galaxy {galaxy_id}   "
        f"logZ = {logz:.2f} ± {logz_err:.2f}   "
        f"ESS = {ess:.0f}   "
        r"$\hat{R}$" + f"_max = {rhat_str}",
        fontsize=9, y=1.002,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corner plots from NSS HDF5 posterior samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("hdf5",       help="NSS results HDF5 file.")
    parser.add_argument("--out-dir",  default=None,
                        help="Output directory.  Defaults to <hdf5_stem>/corners/ next to the file.")
    parser.add_argument("--n-galaxies", type=int, default=None,
                        help="Plot only the first N galaxies.")
    parser.add_argument("--galaxy-ids", type=int, nargs="+", default=None,
                        help="Row indices (0-based) to plot.  Overrides --n-galaxies.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--suffix", default="",
                        help="Optional suffix appended to each PNG filename.")
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    out_dir = Path(args.out_dir) if args.out_dir else hdf5_path.parent / "corners"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        if "nss_samples" not in f:
            raise ValueError(
                f"{hdf5_path} does not contain 'nss_samples'. "
                "Run with --sampler nss first."
            )

        param_names = [
            (b.decode() if isinstance(b, bytes) else b) for b in f.attrs["param_names"]
        ]
        galaxy_ids_all = f["galaxy_id"][:]
        N              = len(galaxy_ids_all)
        logz_all       = f["nss_logZ"][:]
        logz_err_all   = f["nss_logZ_err"][:]
        ess_all        = f["nss_ess"][:]
        rhat_all       = f["nss_rhat"][:]    # (N, P)

        if args.galaxy_ids is not None:
            indices = [i for i in args.galaxy_ids if 0 <= i < N]
        elif args.n_galaxies is not None:
            indices = list(range(min(args.n_galaxies, N)))
        else:
            indices = list(range(N))

        print(f"Plotting {len(indices)} corner plots → {out_dir}")

        for rank, i in enumerate(indices):
            gid     = galaxy_ids_all[i]
            samples = f["nss_samples"][i]           # (S, P)
            logz    = float(logz_all[i])
            logz_err = float(logz_err_all[i])
            ess     = float(ess_all[i])
            rhat_max = float(np.nanmax(rhat_all[i]))

            suffix = f"_{args.suffix}" if args.suffix else ""
            out_path = out_dir / f"galaxy_{gid}{suffix}.png"

            plot_corner(
                samples, param_names, gid,
                logz, logz_err, ess, rhat_max,
                out_path, dpi=args.dpi,
            )
            print(f"  [{rank + 1}/{len(indices)}]  galaxy {gid}  "
                  f"logZ={logz:.2f}  ESS={ess:.0f}  rhat_max={rhat_max:.3f}  → {out_path.name}")

    print(f"\nDone. {len(indices)} plots saved to {out_dir}")


if __name__ == "__main__":
    main()
