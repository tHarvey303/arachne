#!/usr/bin/env python3
"""Optuna hyperparameter search on top of emulator_lab.

Each trial trains a variant at a reduced budget (default 400 epochs) and is
scored on the fixed held-out test split by median per-band scatter in
asinh-mag space. Uses sqlite storage so the study can be resumed.

Preprocessing (mass-norm, sfr-arsinh) is fixed ON — those are strict wins
from the ablation series. Architecture family, size, Fourier-K and optimiser
settings are searched.
"""
from __future__ import annotations

import argparse
import json
import sys

import emulator_lab as lab

OUT = lab.OUT_DIR


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--study", default="parrot_v2_hpo")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import optuna

    storage = f"sqlite:///{OUT}/optuna_hpo.db"

    def objective(trial):
        arch = trial.suggest_categorical("arch", ["mlp", "resmlp"])
        width = trial.suggest_categorical("width", [384, 512, 768, 1024])
        if arch == "mlp":
            depth = trial.suggest_int("depth", 4, 8)
            size_args = ["--depth", str(depth)]
        else:
            blocks = trial.suggest_int("blocks", 3, 6)
            size_args = ["--blocks", str(blocks)]
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        batch = trial.suggest_categorical("batch", [4096, 8192])
        wd = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4])
        fourier = trial.suggest_categorical("fourier_k", [8, 16, 32, 64])
        keep_mass = trial.suggest_categorical("keep_mass_input", [False, True])

        name = f"hpo_t{trial.number:03d}"
        argv = ["--name", name, "--arch", arch, "--width", str(width),
                *size_args, "--lr", f"{lr:.6g}", "--batch", str(batch),
                "--epochs", str(args.epochs),
                "--mass-norm", "--sfr-arsinh"]
        if fourier:
            argv += ["--fourier-k", str(fourier)]
        if wd:
            argv += ["--weight-decay", str(wd)]
        if keep_mass:
            argv += ["--keep-mass-input"]
        print(f"\n[trial {trial.number}] {' '.join(argv)}", flush=True)
        try:
            lab.main(argv)
        except Exception as e:
            print(f"[trial {trial.number}] failed: {e}", flush=True)
            raise optuna.TrialPruned() from e
        with open(OUT / f"{name}.json") as fh:
            res = json.load(fh)
        for k in ["med_scatter", "det_scatter", "trans_rms", "bright_flux_err",
                  "rmse", "leak_p95_nJy", "n_params"]:
            trial.set_user_attr(k, res[k])
        # Fitting-relevant objective: scatter in the detectable regime plus a
        # 0.2-weighted penalty on the dropout-transition RMS.
        return res["det_scatter"] + 0.2 * res["trans_rms"]

    study = optuna.create_study(
        study_name=args.study, storage=storage, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=1, multivariate=True),
        load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print("\n==== HPO complete ====")
    print("best value:", study.best_value)
    print("best params:", study.best_params)
    df = study.trials_dataframe()
    df.to_csv(OUT / "hpo_trials.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
