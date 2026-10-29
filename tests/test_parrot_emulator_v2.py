"""Tests for ParrotEmulatorV2 persistence."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from arachne.emulator.parrot_emulator_v2 import ParrotEmulatorV2, load_emulator

CFG = {
    "arch": "mlp",
    "width": 8,
    "depth": 2,
    "blocks": 1,
    "flux_floor": 1e-4,
    "fourier_k": 2,
    "i_mass": 1,
    "i_z": 0,
    "keep_idx": [0, 2],
    "keep_mass_input": False,
    "mass_norm": True,
    "mu0_train": 19.5,
    "sfr_arsinh": False,
    "sfr_cols": [],
}


def _tiny() -> ParrotEmulatorV2:
    """Untrained 3-input, 2-band V2 emulator (features: 2 kept inputs + 4 Fourier terms)."""
    return ParrotEmulatorV2(
        ["redshift", "log_mass", "Av"],
        ["b1", "b2"],
        CFG,
        in_mean=np.zeros(6, np.float32),
        in_std=np.ones(6, np.float32),
        out_mean=np.full(2, 15.0, np.float32),
        key=jax.random.PRNGKey(1),
    )


PARAMS = np.array([[2.0, 9.0, 0.5], [1.0, 8.0, 1.5]], np.float32)


def test_save_load_roundtrip(tmp_path):
    """A saved checkpoint reloads to identical predictions."""
    emu = _tiny()
    path = tmp_path / "tiny.eqx"
    emu.save(path)
    loaded = load_emulator(path)
    np.testing.assert_array_equal(
        np.asarray(loaded.predict(jnp.asarray(PARAMS))),
        np.asarray(emu.predict(jnp.asarray(PARAMS))),
    )


def test_load_under_x64(tmp_path):
    """Float32 checkpoints load with x64 enabled and predict in float64."""
    emu = _tiny()
    path = tmp_path / "tiny.eqx"
    emu.save(path)
    expect = np.asarray(emu.predict(jnp.asarray(PARAMS)), np.float64)
    with jax.enable_x64(True):
        loaded = load_emulator(path)
        out = loaded.predict(jnp.asarray(PARAMS, jnp.float64))
        assert out.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(out), expect, rtol=1e-5)
    assert np.all(expect > 0)
