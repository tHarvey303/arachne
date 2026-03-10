"""Train an SPSMLPEmulator from a synference HDF5 model library.

This script replaces the fragile JAXFlowEmulator weight-export path.
The resulting .eqx checkpoint is a native JAX/Equinox model that can be
loaded directly into arachne without any PyTorch dependency.

Usage
-----
    python train_emulator.py \\
        --library /path/to/galaxy_library.h5 \\
        --output emulator.eqx \\
        --param-names log_stellar_mass log_age log_metallicity tau_v \\
        --band-names JWST/NIRCam.F115W JWST/NIRCam.F200W JWST/NIRCam.F277W \\
        --hidden-sizes 256 256 256 \\
        --n-epochs 300

Then use the emulator in arachne:

    emulator = SPSMLPEmulator.load(
        "emulator.eqx",
        param_names=[...],
        band_names=[...],
        hidden_sizes=[256, 256, 256],
    )
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from arachne.emulator.jax_mlp_emulator import SPSMLPEmulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an SPSMLPEmulator from a synference HDF5 library."
    )
    parser.add_argument("--library", type=Path, required=True, help="synference HDF5 library path")
    parser.add_argument("--output", type=Path, default=Path("emulator.eqx"))
    parser.add_argument(
        "--param-names",
        nargs="+",
        default=["log_stellar_mass", "log_age", "log_metallicity", "tau_v"],
        help="SPS parameter names (must match library ParameterNames attribute)",
    )
    parser.add_argument(
        "--band-names",
        nargs="+",
        required=True,
        help="Photometric band names (must match library FilterCodes attribute)",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=[256, 256, 256],
        help="Hidden layer widths, e.g. --hidden-sizes 256 256 256",
    )
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--plot", action="store_true", help="Plot training diagnostics")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training SPSMLPEmulator")
    print(f"  Library:     {args.library}")
    print(f"  Params:      {args.param_names}")
    print(f"  Bands:       {args.band_names}")
    print(f"  Architecture: {args.hidden_sizes}")
    print(f"  Epochs:      {args.n_epochs}, batch {args.batch_size}, lr {args.lr}")

    emulator = SPSMLPEmulator.from_synference_library(
        library_path=args.library,
        param_names=args.param_names,
        band_names=args.band_names,
        hidden_sizes=args.hidden_sizes,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    emulator.save(str(args.output))
    print(f"Emulator saved to {args.output}")

    if args.plot:
        _make_validation_plot(emulator, args)


def _make_validation_plot(emulator: SPSMLPEmulator, args) -> None:
    """Quick scatter plot: predicted vs true flux for a random validation subset."""
    import h5py
    from arachne.emulator.jax_mlp_emulator import _decode_str_attr, _select_indices

    with h5py.File(args.library, "r") as f:
        raw_params = f["Grid/Parameters"][()]
        raw_phot = f["Grid/Photometry"][()]
        lib_param_names = _decode_str_attr(f.attrs.get("ParameterNames", []))
        lib_band_names = _decode_str_attr(f.attrs.get("FilterCodes", []))

    pidx = _select_indices(args.param_names, lib_param_names, "parameter")
    bidx = _select_indices(args.band_names, lib_band_names, "band")
    params = raw_params[pidx, :].T.astype(np.float32)
    phot_true = raw_phot[bidx, :].T.astype(np.float32)

    rng = np.random.default_rng(99)
    n_plot = min(2000, len(params))
    idx = rng.choice(len(params), n_plot, replace=False)
    phot_pred = np.asarray(emulator.predict(jnp.array(params[idx])))
    phot_true_sub = phot_true[idx]

    n_bands = len(args.band_names)
    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4))
    if n_bands == 1:
        axes = [axes]
    for i, (ax, bname) in enumerate(zip(axes, args.band_names)):
        ax.scatter(
            np.log10(phot_true_sub[:, i] + 1e-30),
            np.log10(phot_pred[:, i] + 1e-30),
            s=2, alpha=0.3,
        )
        lo = min(np.log10(phot_true_sub[:, i]).min(), np.log10(phot_pred[:, i]).min())
        hi = max(np.log10(phot_true_sub[:, i]).max(), np.log10(phot_pred[:, i]).max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="1:1")
        ax.set_xlabel("log10(true flux / nJy)")
        ax.set_ylabel("log10(pred flux / nJy)")
        ax.set_title(bname.split(".")[-1])
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = args.output.with_suffix(".png")
    plt.savefig(plot_path, dpi=150)
    print(f"Validation plot saved to {plot_path}")


if __name__ == "__main__":
    main()
