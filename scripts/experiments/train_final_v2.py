#!/usr/bin/env python3
"""Train the production ParrotEmulatorV2 with the winning configuration.

Excludes the experiment framework's held-out test rows (SPLIT_SEED=123,
first 50k of the permutation over finite-parameter rows) so the saved
checkpoint can be validated cleanly on them afterwards.
"""
import numpy as np

import emulator_lab as lab
from arachne.emulator.parrot_emulator_v2 import ParrotEmulatorV2

OUT = "/cosma/apps/dp276/dc-harv3/arachne/scripts/outputs/emulators/parrot_emulator_v2.eqx"

param_names, band_names, params, flux, tr, va, te = lab.load_data(lab.LIBRARY)
print(f"excluding {len(te)} test rows; training pool = {len(params) - len(te)}")

emu = ParrotEmulatorV2.from_synference_library(
    lab.LIBRARY,
    band_names=band_names,
    param_names=param_names,
    arch="resmlp",
    width=512,
    blocks=4,
    fourier_k=64,
    mass_norm=True,
    sfr_arsinh=True,
    n_epochs=2400,
    batch_size=2048,
    learning_rate=1e-3,
    val_fraction=0.026,
    seed=0,
    log_interval=50,
    checkpoint_path=OUT.replace(".eqx", ".best.eqx"),
    exclude_rows=te,
)
emu.save(OUT)
print(f"saved {OUT}")
