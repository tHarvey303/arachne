#!/usr/bin/env python3
"""Convert the best lab-trained weights into a ParrotEmulatorV2 checkpoint.

Convert the best lab-trained weights (F_res512_f64) into a production
ParrotEmulatorV2 checkpoint, then verify the conversion reproduces the lab
test metrics exactly.

The lab's ResMLP/Block classes have the same pytree structure (field names and
leaf order) as ParrotEmulatorV2's _ResMLPNet/_ResBlock, so the serialized
leaves can be loaded directly into a V2 net.
"""

import io
import json

import emulator_lab as lab
import equinox as eqx
import jax
import numpy as np

from arachne.emulator.parrot_emulator_v2 import (
    ParrotEmulatorV2,
    _mu0_from_floor,
    load_emulator,
)

SRC = lab.OUT_DIR / "F_res512_f64_weights.npz"
DST = "/cosma/apps/dp276/dc-harv3/arachne/scripts/outputs/emulators/parrot_emulator_v2.eqx"

raw = np.load(SRC, allow_pickle=False)
run_cfg = json.loads(str(raw["config"]))
assert run_cfg["arch"] == "resmlp" and run_cfg["mass_norm"] and run_cfg["sfr_arsinh"]
in_mean, in_std, out_mean = raw["in_mean"], raw["in_std"], raw["out_mean"]

# Rebuild the exact preprocessing constants the lab run used.
param_names, band_names, params_raw, flux_raw, tr, va, te = lab.load_data(lab.LIBRARY)
i_mass = param_names.index("log_mass")
i_z = param_names.index("redshift")
m9 = 10.0 ** (params_raw[:, i_mass] - 9.0)
unit_floor = lab.FLUX_FLOOR / m9.max()
mu0_train = _mu0_from_floor(unit_floor)

keep_idx = [i for i in range(len(param_names)) if i != i_mass]
sfr_cols = [keep_idx.index(i) for i, n in enumerate(param_names) if n.startswith("logsfr_ratio")]

cfg = dict(
    arch="resmlp",
    width=run_cfg["width"],
    depth=run_cfg["depth"],
    blocks=run_cfg["blocks"],
    mass_norm=True,
    keep_mass_input=False,
    sfr_arsinh=True,
    fourier_k=run_cfg["fourier_k"],
    flux_floor=lab.FLUX_FLOOR,
    mu0_train=float(mu0_train),
    i_mass=i_mass,
    i_z=i_z,
    keep_idx=keep_idx,
    sfr_cols=sfr_cols,
)

emu = ParrotEmulatorV2(
    param_names, band_names, cfg, in_mean, in_std, out_mean, key=jax.random.PRNGKey(0)
)
# The lab serialized the bare net; its leaf structure matches emu.net.
net = eqx.tree_deserialise_leaves(io.BytesIO(bytes(raw["weights_bytes"])), emu.net)
emu = eqx.tree_at(lambda m: m.net, emu, net)
emu.save(DST)

# ---- verification: must reproduce F_res512_f64.json on the test split ------
import jax.numpy as jnp

emu2 = load_emulator(DST)
pred = []
for i in range(0, len(te), 65536):
    pred.append(
        np.asarray(emu2.predict(jnp.array(params_raw[te[i : i + 65536]], dtype=jnp.float32)))
    )
pred_flux = np.concatenate(pred).astype(np.float64)

flux_true = np.where(flux_raw[te] < lab.FLUX_FLOOR, 0.0, flux_raw[te])
mag_true = lab.flux_to_mag(flux_true)
r = lab.flux_to_mag(pred_flux) - mag_true
det = flux_true > 1.0
det_scatter = float(np.median([r[det[:, b], b].std() for b in range(r.shape[1])]))

ref = json.load(open(lab.OUT_DIR / "F_res512_f64.json"))
print(f"converted det_scatter = {det_scatter:.6f}  (lab run: {ref['det_scatter']:.6f})")
assert abs(det_scatter - ref["det_scatter"]) < 5e-4, "conversion mismatch!"
print("conversion verified — production checkpoint:", DST)
