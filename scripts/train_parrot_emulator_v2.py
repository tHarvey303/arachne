#!/usr/bin/env python3
r"""Train a ParrotEmulatorV2 on a synference HDF5 library.

V2 improvements over the original ParrotEmulator (see
``arachne/emulator/parrot_emulator_v2.py``): analytic mass factorisation,
Fourier redshift features, arsinh-compressed SFH-ratio inputs, and a
warmup+cosine AdamW training recipe that runs entirely on-GPU.

Usage
-----
::

    python scripts/train_parrot_emulator_v2.py \
        --library /path/to/library.hdf5 \
        --output  outputs/emulators/parrot_emulator_v2.eqx

Defaults reproduce the configuration selected by the Optuna search in
``scripts/experiments/hpo_lab.py``.
"""

from __future__ import annotations

import argparse
import sys

from train_parrot_emulator import _DEFAULT_BANDS


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Train a ParrotEmulatorV2 (mass-factorised, Fourier-z).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--library", required=True)
    p.add_argument("--output", default="outputs/emulators/parrot_emulator_v2.eqx")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="best-val safety checkpoint path (default: <output>.best.eqx)",
    )
    p.add_argument("--params", nargs="+", default="all")
    p.add_argument("--bands", nargs="+", default=_DEFAULT_BANDS)
    p.add_argument("--arch", choices=["mlp", "resmlp"], default="mlp")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=5, help="hidden layers (mlp)")
    p.add_argument("--blocks", type=int, default=4, help="residual blocks (resmlp)")
    p.add_argument("--fourier-k", type=int, default=16)
    p.add_argument("--no-mass-norm", action="store_true")
    p.add_argument("--keep-mass-input", action="store_true")
    p.add_argument("--no-sfr-arsinh", action="store_true")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-fraction", type=float, default=0.025)
    p.add_argument("--flux-floor", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=25)
    return p.parse_args(argv)


def main(argv=None):
    """Train a ParrotEmulatorV2 from a synference library and save the checkpoint."""
    args = parse_args(argv)
    import os

    from arachne.emulator.parrot_emulator_v2 import ParrotEmulatorV2

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    checkpoint = args.checkpoint
    if checkpoint is None:
        stem = args.output[: -len(".eqx")] if args.output.endswith(".eqx") else args.output
        checkpoint = stem + ".best.eqx"

    print(f"Library : {args.library}")
    print(f"Output  : {args.output}")
    print(
        f"Arch    : {args.arch} width={args.width} "
        f"{'depth=' + str(args.depth) if args.arch == 'mlp' else 'blocks=' + str(args.blocks)} "
        f"fourier_k={args.fourier_k}"
    )
    print(f"Training: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")

    emulator = ParrotEmulatorV2.from_synference_library(
        library_path=args.library,
        param_names=args.params,
        band_names=args.bands,
        arch=args.arch,
        width=args.width,
        depth=args.depth,
        blocks=args.blocks,
        mass_norm=not args.no_mass_norm,
        keep_mass_input=args.keep_mass_input,
        sfr_arsinh=not args.no_sfr_arsinh,
        fourier_k=args.fourier_k,
        flux_floor=args.flux_floor,
        n_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        log_interval=args.log_interval,
        checkpoint_path=checkpoint,
    )
    emulator.save(args.output)
    print(f"Saved trained V2 emulator to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
