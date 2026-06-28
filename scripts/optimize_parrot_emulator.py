#!/usr/bin/env python3
r"""Optimise ParrotEmulator hyperparameters with Optuna.

Searches over network architecture and optimiser settings to find the
ParrotEmulator configuration that minimises emulation error (combined
bias + scatter, in arsinh-magnitude units) on a subset of a synference
HDF5 library, then retrains the winning configuration for longer and
saves it.

Usage
-----
::

    python scripts/optimize_parrot_emulator.py \
        --library          /path/to/grid.hdf5 \
        --output           outputs/emulators/parrot_emulator_optuna.eqx \
        --n-trials         50 \
        --epochs-per-trial 300 \
        --final-epochs     1500

Run ``python scripts/optimize_parrot_emulator.py --help`` for all options.

What is optimised
-----------------
Each Optuna trial samples:

- ``n_layers``         number of hidden layers (depth)
- ``width``            units per hidden layer (uniform width)
- ``learning_rate``    initial NADAM/Adam learning rate (log scale)
- ``batch_size``       mini-batch size
- ``lr_decay_factor``  multiplicative LR factor applied at each decay step

trains a short emulator with those settings, and scores per-band bias and
scatter in arsinh-magnitude space.  The objective minimised by default is
the median over bands of ``sqrt(bias^2 + scatter^2)``.

Objective metrics (``--metric``)
--------------------------------
- ``combined``       median_b sqrt(bias_b^2 + scatter_b^2)   [default]
- ``combined_mean``  mean_b   sqrt(bias_b^2 + scatter_b^2)
- ``combined_max``   max_b    sqrt(bias_b^2 + scatter_b^2)
- ``scatter``        median_b scatter_b
- ``bias``           median_b |bias_b|

The default parameter and band sets mirror those in
``train_parrot_emulator.py`` (BPASS Chab DenseBasis v4, 8 params, 24 bands).

Note on the evaluation split
----------------------------
For speed and to stay within the public ``from_synference_library`` API,
each trial trains on the library (using its own internal early-stopping
split) and is scored on a *fixed* random subset of the same library.  This
is appropriate for *ranking* configurations against one another.  For
unbiased final numbers, run ``validate_parrot_emulator.py`` on a genuinely
held-out library after this search completes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

_METRICS = ("combined", "combined_mean", "combined_max", "scatter", "bias")


def parse_args(argv=None):
    """Parse command-line arguments for hyperparameter optimisation."""
    p = argparse.ArgumentParser(
        description="Optimise ParrotEmulator hyperparameters with Optuna "
        "(minimise bias/scatter on a synference HDF5 library).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- I/O --------------------------------------------------------------
    p.add_argument("--library", required=True, help="Path to synference HDF5 library file.")
    p.add_argument(
        "--output",
        default="outputs/emulators/parrot_emulator_optuna.eqx",
        help="Output .eqx checkpoint for the best (retrained) emulator.",
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

    # ---- Objective --------------------------------------------------------
    p.add_argument(
        "--metric",
        choices=_METRICS,
        default="combined",
        help="Scalar objective to minimise (see module docstring).",
    )

    # ---- Optuna study -----------------------------------------------------
    p.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials.")
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional wall-clock budget for the whole search, in seconds.",
    )
    p.add_argument("--study-name", default="parrot_emulator", help="Optuna study name.")
    p.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL (e.g. sqlite:///optuna.db) to persist "
        "and resume the study. If set, an existing study is loaded.",
    )
    p.add_argument("--sampler-seed", type=int, default=0, help="Seed for the TPE sampler.")

    # ---- Search-space bounds ---------------------------------------------
    p.add_argument(
        "--layers-range",
        nargs=2,
        type=int,
        default=[3, 6],
        metavar=("MIN", "MAX"),
        help="Inclusive range for the number of hidden layers.",
    )
    p.add_argument(
        "--width-choices",
        nargs="+",
        type=int,
        default=[256, 512, 768, 1024],
        help="Candidate hidden-layer widths (units per layer).",
    )
    p.add_argument(
        "--batch-choices",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096, 8192],
        help="Candidate mini-batch sizes.",
    )
    p.add_argument(
        "--lr-range",
        nargs=2,
        type=float,
        default=[1e-4, 5e-3],
        metavar=("MIN", "MAX"),
        help="Log-uniform range for the initial learning rate.",
    )
    p.add_argument(
        "--lr-decay-range",
        nargs=2,
        type=float,
        default=[0.05, 0.5],
        metavar=("MIN", "MAX"),
        help="Log-uniform range for the per-step LR decay factor "
        "(applied at both decay epochs).",
    )

    # ---- Training budget --------------------------------------------------
    p.add_argument(
        "--epochs-per-trial",
        type=int,
        default=300,
        help="Max training epochs per Optuna trial (kept short for speed).",
    )
    p.add_argument(
        "--final-epochs",
        type=int,
        default=1500,
        help="Max training epochs when retraining the best configuration.",
    )
    p.add_argument(
        "--trial-patience",
        type=int,
        default=15,
        help="Early-stopping patience during trials.",
    )
    p.add_argument(
        "--final-patience",
        type=int,
        default=30,
        help="Early-stopping patience for the final retrain.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction held out internally by the trainer for early stopping.",
    )
    p.add_argument("--train-seed", type=int, default=0, help="Random seed passed to the trainer.")
    p.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Trainer log interval (epochs); larger = quieter during search.",
    )

    # ---- Evaluation subset ------------------------------------------------
    p.add_argument(
        "--n-eval",
        type=int,
        default=50_000,
        help="Number of library samples used to score each trial.",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Seed for the (fixed) evaluation subset, shared across trials.",
    )
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# Data / evaluation helpers
# ----------------------------------------------------------------------------
def _load_eval_subset(library, params, bands, n_eval, seed):
    """Load a fixed random subset of the library for scoring trials.

    Returns ``(params_val, phot_val)`` as float32 arrays of shape
    ``(N, n_params)`` and ``(N, n_bands)`` respectively.
    """
    import h5py
    import numpy as np

    from arachne.emulator.parrot_emulator import (
        _read_band_names,
        _read_param_names,
        _select_indices,
    )

    with h5py.File(library, "r") as f:
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

    rng = np.random.default_rng(seed)
    n = min(n_eval, len(params_all))
    idx = rng.choice(len(params_all), size=n, replace=False)
    return params_all[idx], phot_all[idx]


def _evaluate(emulator, params_val, phot_val):
    """Compute per-band bias, scatter and 95th-percentile |err| (arsinh-mag)."""
    import jax.numpy as jnp
    import numpy as np

    from arachne.emulator.parrot_emulator import _flux_to_asinh_mag_np

    pred_flux = np.asarray(emulator.predict(jnp.array(params_val)))  # (N, B)
    true_mag = _flux_to_asinh_mag_np(phot_val)                       # (N, B)
    pred_mag = _flux_to_asinh_mag_np(pred_flux)                      # (N, B)
    residuals = pred_mag - true_mag                                  # positive = over-predicted

    bias = np.nanmean(residuals, axis=0)                    # (B,)
    scatter = np.nanstd(residuals, axis=0)                  # (B,)
    p95 = np.nanpercentile(np.abs(residuals), 95, axis=0)   # (B,)
    return bias, scatter, p95


def _objective_value(bias, scatter, metric):
    """Reduce per-band (bias, scatter) arrays to a single scalar to minimise."""
    import numpy as np

    rms = np.sqrt(np.square(bias) + np.square(scatter))
    if metric == "combined":
        return float(np.nanmedian(rms))
    if metric == "combined_mean":
        return float(np.nanmean(rms))
    if metric == "combined_max":
        return float(np.nanmax(rms))
    if metric == "scatter":
        return float(np.nanmedian(scatter))
    if metric == "bias":
        return float(np.nanmedian(np.abs(bias)))
    raise ValueError(f"Unknown metric: {metric!r}")


# ----------------------------------------------------------------------------
# Training helper
# ----------------------------------------------------------------------------
def _train_emulator(
    args,
    hidden_sizes,
    learning_rate,
    batch_size,
    lr_decay_factor,
    n_epochs,
    patience,
    checkpoint_path,
):
    """Train a single ParrotEmulator with the given hyperparameters.

    The two LR-decay epochs are placed at ~30% and ~70% of ``n_epochs`` to
    mirror the 3-phase schedule used by ``train_parrot_emulator.py``.
    """
    from arachne.emulator.parrot_emulator import ParrotEmulator

    e1 = max(1, int(round(0.3 * n_epochs)))
    e2 = max(e1 + 1, int(round(0.7 * n_epochs)))

    return ParrotEmulator.from_synference_library(
        library_path=args.library,
        param_names=args.params,
        band_names=args.bands,
        hidden_sizes=hidden_sizes,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_decay_steps=(e1, e2),
        lr_decay_factors=(lr_decay_factor, lr_decay_factor),
        val_fraction=args.val_fraction,
        early_stopping_patience=patience,
        seed=args.train_seed,
        log_interval=args.log_interval,
        checkpoint_path=checkpoint_path,
    )


def _make_objective(args, params_val, phot_val, tmpdir):
    """Build the Optuna objective closure."""
    import os

    import numpy as np
    import optuna

    def objective(trial):
        # --- sample hyperparameters ---
        n_layers = trial.suggest_int("n_layers", args.layers_range[0], args.layers_range[1])
        width = trial.suggest_categorical("width", args.width_choices)
        learning_rate = trial.suggest_float(
            "learning_rate", args.lr_range[0], args.lr_range[1], log=True
        )
        batch_size = trial.suggest_categorical("batch_size", args.batch_choices)
        lr_decay_factor = trial.suggest_float(
            "lr_decay_factor", args.lr_decay_range[0], args.lr_decay_range[1], log=True
        )
        hidden_sizes = [width] * n_layers

        ckpt = str(Path(tmpdir) / f"trial_{trial.number}.best.eqx")
        print(
            f"\n[trial {trial.number}] hidden={hidden_sizes} lr={learning_rate:.2e} "
            f"batch={batch_size} decay={lr_decay_factor:.3f}"
        )

        try:
            emulator = _train_emulator(
                args,
                hidden_sizes=hidden_sizes,
                learning_rate=learning_rate,
                batch_size=batch_size,
                lr_decay_factor=lr_decay_factor,
                n_epochs=args.epochs_per_trial,
                patience=args.trial_patience,
                checkpoint_path=ckpt,
            )
        except Exception as exc:  # OOM, NaN losses, bad config, ...
            print(f"[trial {trial.number}] training failed -> pruned: {exc}")
            raise optuna.TrialPruned() from exc
        finally:
            # Don't let per-trial checkpoints accumulate on disk.
            for suffix in ("", ".best.eqx"):
                stale = ckpt + suffix if suffix and not ckpt.endswith(suffix) else ckpt
                try:
                    if os.path.exists(stale):
                        os.remove(stale)
                except OSError:
                    pass

        bias, scatter, p95 = _evaluate(emulator, params_val, phot_val)
        value = _objective_value(bias, scatter, args.metric)

        # Record human-readable diagnostics for later inspection.
        trial.set_user_attr("median_abs_bias", float(np.nanmedian(np.abs(bias))))
        trial.set_user_attr("median_scatter", float(np.nanmedian(scatter)))
        trial.set_user_attr("median_p95", float(np.nanmedian(p95)))
        trial.set_user_attr("hidden_sizes", hidden_sizes)

        print(
            f"[trial {trial.number}] {args.metric}={value:.5f}  "
            f"|bias|={np.nanmedian(np.abs(bias)):.4f}  "
            f"scatter={np.nanmedian(scatter):.4f}  p95={np.nanmedian(p95):.4f}"
        )
        return value

    return objective


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):  # noqa: C901
    """Run the Optuna search, retrain the best configuration, and save it."""
    args = parse_args(argv)

    import json
    import os
    import tempfile

    import numpy as np

    try:
        import optuna
    except ImportError:
        print("Optuna is required: pip install optuna", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    out_dir = Path(os.path.abspath(args.output)).parent

    print(f"Library  : {args.library}")
    print(f"Output   : {args.output}")
    print(f"Params   : {args.params}")
    print(f"Bands    : {len(args.bands)} bands")
    print(f"Metric   : {args.metric}")
    print(f"Trials   : {args.n_trials}  (epochs/trial: {args.epochs_per_trial})")

    # ------------------------------------------------------------------
    # Fixed evaluation subset (loaded once, shared by every trial)
    # ------------------------------------------------------------------
    print("\nLoading evaluation subset ...")
    params_val, phot_val = _load_eval_subset(
        args.library, args.params, args.bands, args.n_eval, args.eval_seed
    )
    print(f"  {len(params_val)} samples x {len(args.bands)} bands")

    # ------------------------------------------------------------------
    # Optuna study
    # ------------------------------------------------------------------
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        objective = _make_objective(args, params_val, phot_val, tmpdir)
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            gc_after_trial=True,
        )

    n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
    if n_complete == 0:
        print("No trials completed successfully — nothing to retrain.", file=sys.stderr)
        return 1

    print("\n==== Optuna search complete ====")
    print(f"Completed trials : {n_complete}/{len(study.trials)}")
    print(f"Best {args.metric:14s}: {study.best_value:.5f} asinh-mag")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:16s}: {v}")

    # ------------------------------------------------------------------
    # Persist study trials + best hyperparameters
    # ------------------------------------------------------------------
    try:
        df = study.trials_dataframe()
        trials_csv = out_dir / "optuna_trials.csv"
        df.to_csv(trials_csv, index=False)
        print(f"Trial table -> {trials_csv}")
    except Exception as exc:  # pandas optional / IO issues
        print(f"(could not write trials CSV: {exc})")

    best = study.best_params
    hidden_sizes = [best["width"]] * best["n_layers"]
    best_json = out_dir / "best_hyperparameters.json"
    with open(best_json, "w") as fh:
        json.dump(
            {
                "metric": args.metric,
                "best_value": study.best_value,
                "hidden_sizes": hidden_sizes,
                "learning_rate": best["learning_rate"],
                "batch_size": best["batch_size"],
                "lr_decay_factor": best["lr_decay_factor"],
                "final_epochs": args.final_epochs,
                "params": args.params,
                "bands": args.bands,
            },
            fh,
            indent=2,
        )
    print(f"Best hyperparameters -> {best_json}")

    # ------------------------------------------------------------------
    # Retrain the winning configuration for longer and save it
    # ------------------------------------------------------------------
    print(f"\nRetraining best configuration for up to {args.final_epochs} epochs ...")
    stem = args.output[: -len(".eqx")] if args.output.endswith(".eqx") else args.output
    final_ckpt = stem + ".best.eqx"

    emulator = _train_emulator(
        args,
        hidden_sizes=hidden_sizes,
        learning_rate=best["learning_rate"],
        batch_size=best["batch_size"],
        lr_decay_factor=best["lr_decay_factor"],
        n_epochs=args.final_epochs,
        patience=args.final_patience,
        checkpoint_path=final_ckpt,
    )
    emulator.save(args.output)
    print(f"Saved optimised emulator to {args.output}")

    # ------------------------------------------------------------------
    # Final report on the evaluation subset
    # ------------------------------------------------------------------
    bias, scatter, p95 = _evaluate(emulator, params_val, phot_val)
    print("\nFinal emulator (scored on eval subset):")
    print(f"  median |bias|   : {np.nanmedian(np.abs(bias)):.4f} asinh-mag")
    print(f"  median scatter  : {np.nanmedian(scatter):.4f} asinh-mag")
    print(f"  median 95th |err|: {np.nanmedian(p95):.4f} asinh-mag")
    print(
        "\nFor unbiased diagnostics, run validate_parrot_emulator.py on a "
        "held-out library:\n"
        f"  python scripts/validate_parrot_emulator.py --emulator {args.output} "
        f"--library <held_out.hdf5> --hidden {' '.join(map(str, hidden_sizes))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
