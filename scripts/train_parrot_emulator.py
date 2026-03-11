#!/usr/bin/env python3
r"""Train a ParrotEmulator on a synference HDF5 library.

Usage
-----
::

    python scripts/train_parrot_emulator.py \
        --library /path/to/grid.hdf5 \
        --output  parrot_emulator.eqx \
        --bands   JWST/NIRCam.F115W JWST/NIRCam.F150W JWST/NIRCam.F200W \
        --epochs  1000 \
        --batch   4096

Run ``python scripts/train_parrot_emulator.py --help`` for all options.

Default parameter set
---------------------
All 8 parameters from the BPASS Chab DenseBasis v4 library are used by
default::

    redshift  log10metallicity  Av  log_sfr
    sfh_quantile_25  sfh_quantile_50  sfh_quantile_75  tau_v

Default band set
----------------
A representative 24-band set spanning optical–MIR (DECam + HST + JWST
NIRCam wide-bands + MIRI) is used if ``--bands`` is not supplied.  Adjust
to match your science case.
"""

from __future__ import annotations

import argparse
import sys

# Default band selection: broad wavelength coverage from the BPASS v4 library
_DEFAULT_BANDS = [
    "CTIO/DECam.g",
    "CTIO/DECam.r",
    "CTIO/DECam.i",
    "CTIO/DECam.z",
    "CTIO/DECam.Y",
    "HST/ACS_WFC.F435W",
    "HST/ACS_WFC.F606W",
    "HST/ACS_WFC.F814W",
    "HST/ACS_WFC.F850LP",
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F444W",
    "JWST/MIRI.F560W",
    "JWST/MIRI.F770W",
    "JWST/MIRI.F1000W",
    "Spitzer/IRAC.I1",
    "Spitzer/IRAC.I2",
    "Euclid/VIS.vis",
    "Euclid/NISP.H",
]

_DEFAULT_PARAMS = [
    "redshift",
    "log10metallicity",
    "Av",
    "log_sfr",
    "sfh_quantile_25",
    "sfh_quantile_50",
    "sfh_quantile_75",
    "tau_v",
]


def parse_args(argv=None):
    """Parse command-line arguments for training."""
    p = argparse.ArgumentParser(
        description="Train a Parrot-style MLP emulator on a synference HDF5 library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--library", required=True, help="Path to synference HDF5 library file.")
    p.add_argument(
        "--output",
        default="outputs/emulators/parrot_emulator.eqx",
        help="Output .eqx checkpoint path.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path for best-val-loss safety checkpoint during training. "
        "Defaults to outputs/emulators/<stem>.best.eqx",
    )
    p.add_argument(
        "--params",
        nargs="+",
        default=_DEFAULT_PARAMS,
        help="SPS parameter names to use as inputs.",
    )
    p.add_argument(
        "--bands",
        nargs="+",
        default=_DEFAULT_BANDS,
        help="Photometric band names to predict.",
    )
    p.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[512, 512, 512, 512, 512],
        help="Hidden layer widths (5 layers = 6-layer network as in Parrot).",
    )
    p.add_argument("--epochs", type=int, default=1000, help="Maximum training epochs.")
    p.add_argument("--batch", type=int, default=4096, help="Mini-batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial Adam/NADAM learning rate.")
    p.add_argument(
        "--lr-decay-epochs",
        nargs=2,
        type=int,
        default=[300, 700],
        metavar=("E1", "E2"),
        help="Epochs at which LR is decayed (3-phase schedule).",
    )
    p.add_argument(
        "--lr-decay-factors",
        nargs=2,
        type=float,
        default=[0.1, 0.1],
        metavar=("F1", "F2"),
        help="Multiplicative LR factors at each decay epoch.",
    )
    p.add_argument(
        "--val-fraction", type=float, default=0.1, help="Fraction of data held out for validation."
    )
    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience (epochs without val improvement).",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--log-interval", type=int, default=10, help="Log every N epochs.")
    return p.parse_args(argv)


def main(argv=None):
    """Train a ParrotEmulator and save the checkpoint."""
    args = parse_args(argv)

    import os

    from arachne.emulator.parrot_emulator import ParrotEmulator

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    checkpoint = args.checkpoint
    if checkpoint is None:
        stem = args.output[: -len(".eqx")] if args.output.endswith(".eqx") else args.output
        checkpoint = stem + ".best.eqx"

    print(f"Library  : {args.library}")
    print(f"Output   : {args.output}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Params   : {args.params}")
    print(f"Bands    : {len(args.bands)} bands")
    print(f"Hidden   : {args.hidden}")
    print(f"Epochs   : {args.epochs}  Batch: {args.batch}  LR: {args.lr}")

    emulator = ParrotEmulator.from_synference_library(
        library_path=args.library,
        param_names=args.params,
        band_names=args.bands,
        hidden_sizes=args.hidden,
        n_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        lr_decay_steps=tuple(args.lr_decay_epochs),
        lr_decay_factors=tuple(args.lr_decay_factors),
        val_fraction=args.val_fraction,
        early_stopping_patience=args.patience,
        seed=args.seed,
        log_interval=args.log_interval,
        checkpoint_path=checkpoint,
    )

    emulator.save(args.output)
    print(f"Saved trained emulator to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
