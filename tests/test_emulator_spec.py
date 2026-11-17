"""Tests for emulator metadata (EmulatorSpec), EmulatorView and the generic loader."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.emulator import EmulatorSpec, EmulatorView, load_emulator, param_info
from arachne.emulator.fixed_param_wrapper import FixedParamEmulator
from arachne.emulator.parrot_emulator_v2 import ParrotEmulatorV2
from arachne.emulator.validation import (
    band_error_table,
    select_band_errors,
    select_z_bin,
    table_z_bins,
)
from arachne.priors.specs import resolve_prior_specs

NAMES = ["redshift", "log_mass", "dust_slope", "fesc", "Av", "logsfr_ratio_0"]
CFG = {
    "arch": "mlp",
    "width": 8,
    "depth": 2,
    "blocks": 1,
    "flux_floor": 1e-4,
    "fourier_k": 2,
    "z_scale": 20.0,
    "i_mass": 1,
    "i_z": 0,
    "keep_idx": [0, 2, 3, 4, 5],
    "keep_mass_input": False,
    "mass_norm": True,
    "mu0_train": 19.5,
    "sfr_arsinh": True,
    "sfr_cols": [4],
}
BOUNDS = {
    "redshift": (0.01, 20.0),
    "log_mass": (6.0, 12.5),
    "dust_slope": (-0.3, 1.1),
    "fesc": (0.0, 1.0),
    "Av": (0.001, 5.0),
    "logsfr_ratio_0": (-5.0, 5.0),
}


def _tiny() -> ParrotEmulatorV2:
    """Untrained 6-input, 3-band V2 emulator with v7-style names."""
    return ParrotEmulatorV2(
        NAMES,
        ["b1", "b2", "b3"],
        CFG,
        in_mean=np.zeros(9, np.float32),
        in_std=np.ones(9, np.float32),
        out_mean=np.full(3, 15.0, np.float32),
        key=jax.random.PRNGKey(3),
    )


FULL = np.array([[2.0, 9.0, 0.2, 0.1, 0.5, 0.3], [5.0, 10.0, -0.1, 0.1, 1.5, -1.0]], np.float32)


def test_param_info_aliases_and_ratios():
    """Aliases resolve to the canonical entry; any SFH ratio index is known."""
    assert param_info("slope").name == "dust_slope"
    assert param_info("logsfr_ratio_7").role == "sfh_ratio"
    assert param_info("logsfr_ratio_7").prior["dist"] == "studentt"
    assert param_info("something_new").bounds is None


def test_spec_fallback_and_roles():
    """Without recorded bounds the spec falls back to the registry and says so."""
    spec = _tiny().spec
    assert spec.mass_param == "log_mass" and spec.redshift_param == "redshift"
    assert spec.sfh_ratio_params == ("logsfr_ratio_0",)
    assert spec.meta["bounds_source"] == "fallback"
    assert spec.param_bounds["redshift"] == (0.01, 14.0)
    assert spec.defaults["fesc"] == pytest.approx(0.1)


def test_with_metadata_roundtrip(tmp_path):
    """Recorded bounds survive save/load and do not change predictions."""
    emu = _tiny()
    annotated = emu.with_metadata(param_bounds=BOUNDS, library="lib_v7.hdf5")
    path = tmp_path / "tiny.eqx"
    annotated.save(path)
    loaded = load_emulator(path)
    assert loaded.spec.meta == {"bounds_source": "checkpoint", "library": "lib_v7.hdf5"}
    assert loaded.spec.param_bounds == BOUNDS
    assert loaded.param_bounds["logsfr_ratio_0"] == (-5.0, 5.0)
    np.testing.assert_array_equal(
        np.asarray(loaded.predict(jnp.asarray(FULL))), np.asarray(emu.predict(jnp.asarray(FULL)))
    )
    with pytest.raises(ValueError, match="not emulator parameters"):
        emu.with_metadata(param_bounds={"nope": (0, 1)})


def test_z_scale_is_used():
    """The Fourier period comes from the checkpoint, not a module constant."""
    a = _tiny()
    b = ParrotEmulatorV2(
        NAMES,
        ["b1", "b2", "b3"],
        {**CFG, "z_scale": 14.0},
        in_mean=np.zeros(9, np.float32),
        in_std=np.ones(9, np.float32),
        out_mean=np.full(3, 15.0, np.float32),
        key=jax.random.PRNGKey(3),
    )
    x = jnp.asarray(FULL)
    assert not np.allclose(np.asarray(a._features(x)), np.asarray(b._features(x)))


def test_view_matches_inner_with_fixed_renamed_reordered():
    """A view with an alias, a fixed input, a new order and a band subset."""
    emu = _tiny().with_metadata(param_bounds=BOUNDS)
    view = EmulatorView(
        emu,
        params=["log_mass", "redshift", "slope", "Av", "logsfr_ratio_0"],
        bands=["b3", "b1"],
    )
    assert view.param_names == ["log_mass", "redshift", "slope", "Av", "logsfr_ratio_0"]
    assert view.fixed_params == {"fesc": pytest.approx(0.1)}
    x = FULL[:, [1, 0, 2, 4, 5]]
    np.testing.assert_allclose(
        np.asarray(view.predict(jnp.asarray(x))),
        np.asarray(emu.predict(jnp.asarray(FULL)))[:, [2, 0]],
        rtol=1e-6,
    )
    spec = view.spec
    assert spec.param_names == tuple(view.param_names)
    assert spec.band_names == ("b3", "b1")
    assert spec.param_bounds["slope"] == (-0.3, 1.1)
    assert spec.mass_param == "log_mass"
    assert spec.meta["fixed"] == {"fesc": pytest.approx(0.1)}
    assert spec.meta["renamed"] == {"dust_slope": "slope"}
    grad = jax.grad(lambda t: view.predict(t[None])[0, 0])(jnp.asarray(x[0]))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_view_errors():
    """Unknown names, double use and out-of-domain constants are rejected."""
    emu = _tiny().with_metadata(param_bounds=BOUNDS)
    with pytest.raises(ValueError, match="not an input"):
        EmulatorView(emu, params=["redshift", "log_age"])
    with pytest.raises(ValueError, match="both exposed and fixed"):
        EmulatorView(emu, params=["redshift", "fesc"], fixed={"fesc": 0.2})
    with pytest.raises(ValueError, match="outside the emulator domain"):
        EmulatorView(emu, fixed={"fesc": 1.5})
    with pytest.raises(ValueError, match="neither exposed nor fixed"):
        EmulatorView(emu, params=["redshift", "log_mass"], fill_defaults=False)
    with pytest.raises(ValueError, match="bands not found"):
        EmulatorView(emu, bands=["nope"])


def test_load_emulator_with_view(tmp_path):
    """``load_emulator`` keyword arguments return a view; none returns the raw class."""
    path = tmp_path / "tiny.eqx"
    _tiny().save(path)
    assert isinstance(load_emulator(path), ParrotEmulatorV2)
    view = load_emulator(path, fixed={"redshift": 2.0}, bands=["b2"])
    assert view.param_names == NAMES[1:]
    assert view.spec.redshift_param is None
    assert view.predict(jnp.asarray(FULL[:, 1:])).shape == (2, 1)


def test_clip_bounds():
    """Driver bounds are intersected with the emulator domain."""
    spec = _tiny().with_metadata(param_bounds=BOUNDS).spec
    out = spec.clip_bounds({"redshift": (0.01, 14.0), "logsfr_ratio_0": (-10.0, 10.0)})
    assert out["redshift"] == (0.01, 14.0)
    assert out["logsfr_ratio_0"] == (-5.0, 5.0)
    assert out["log_mass"] == (6.0, 12.5)
    with pytest.raises(ValueError, match="do not overlap"):
        spec.clip_bounds({"redshift": (25.0, 30.0)})


def test_fixed_param_wrapper_spec():
    """The legacy single-parameter wrapper forwards the inner spec."""
    emu = _tiny().with_metadata(param_bounds=BOUNDS)
    spec = FixedParamEmulator(emu, "redshift", 2.0).spec
    assert "redshift" not in spec.param_names
    assert spec.meta["fixed"] == {"redshift": 2.0}
    assert spec.param_bounds["log_mass"] == (6.0, 12.5)


def test_base_spec_and_prior_defaults(mock_emulator):
    """Any SPSEmulator has a spec; priors resolve from it for unknown parameter sets."""
    spec = mock_emulator.spec
    assert isinstance(spec, EmulatorSpec)
    assert spec.param_names == tuple(mock_emulator.param_names)
    names = ["fesc", "logsfr_ratio_6"]
    priors = resolve_prior_specs(names, None, {"fesc": (0, 1), "logsfr_ratio_6": (-5, 5)})
    assert priors["fesc"]["dist"] == "uniform"
    assert priors["logsfr_ratio_6"]["dist"] == "studentt"
    over = resolve_prior_specs(
        names,
        None,
        {"fesc": (0, 1), "logsfr_ratio_6": (-5, 5)},
        defaults={"logsfr_ratio_6": {"dist": "uniform"}},
    )
    assert over["logsfr_ratio_6"]["dist"] == "uniform"


def _table(n=4000, seed=0):
    """Band-error table for three bands with known scatter: 1 %, 2 %, and 5 % above z = 3."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, 6.0, n)
    true = np.full((n, 3), 100.0)
    true[: n // 10, 2] = 0.5  # undetected rows in b3 are ignored
    scatter = np.stack([np.full(n, 0.01), np.full(n, 0.02), np.where(z > 3.0, 0.05, 0.01)], axis=1)
    pred = true * (1.0 + scatter * rng.standard_normal((n, 3)))
    return band_error_table(pred, true, z, ["b1", "b2", "b3"], source="test")


def test_band_error_table_values_and_bins():
    """The table recovers the injected scatter per band and per redshift bin."""
    table = _table()
    assert table["z_bins"]["all"] is None and table["z_bins"]["z2-3"] == [2.0, 3.0]
    assert "z8-10" not in table["z_bins"]  # bins stop at the largest redshift
    all_ = {b: table["bands"][b]["all"] for b in ("b1", "b2", "b3")}
    assert all_["b1"]["mad"] == pytest.approx(0.01, rel=0.1)
    assert all_["b2"]["mad"] == pytest.approx(0.02, rel=0.1)
    assert all_["b3"]["n"] == 3600
    assert table["bands"]["b3"]["z4-6"]["mad"] == pytest.approx(0.05, rel=0.15)
    assert table["bands"]["b3"]["z1-1.5"]["mad"] == pytest.approx(0.01, rel=0.2)
    assert abs(all_["b1"]["bias"]) < 0.002


def test_select_bin_and_errors():
    """Bin choice: narrowest populated bin containing z, else the widest."""
    table = _table()
    assert select_z_bin(table, 2.1) == "z1.8-2.5"
    assert select_z_bin(table, 5.0) == "z4-6"
    assert select_z_bin(table, None) == "all"
    assert select_z_bin(table, 30.0) == "all"
    assert select_z_bin(table, 5.0, min_rows=10**6) == "all"
    values, name = select_band_errors(table, ["b3", "b1"], redshift=5.0)
    assert name == "z4-6" and values.shape == (2,)
    assert values[0] > 3 * values[1]
    assert select_band_errors(table, ["b1"], redshift=5.0, z_bin="all")[1] == "all"
    with pytest.raises(ValueError, match="no entry"):
        select_band_errors(table, ["nope"])
    legacy = {"bands": {"b1": {"z1.8-2.5": {"mad": 0.004, "n": 500}, "all": {"mad": 0.01}}}}
    assert table_z_bins(legacy) == {"z1.8-2.5": (1.8, 2.5), "all": None}
    assert select_band_errors(legacy, ["b1"], redshift=2.0)[0][0] == 0.004
    assert select_band_errors(legacy, ["b1"], redshift=4.0)[0][0] == 0.01


def test_band_errors_in_checkpoint(tmp_path):
    """The table survives save/load, follows a view's band order, and is validated."""
    emu = _tiny()
    assert emu.spec.band_errors is None
    with pytest.raises(ValueError, match="no band-error table"):
        emu.spec.band_error(2.0)
    table = _table()
    path = tmp_path / "tiny.eqx"
    emu.with_metadata(band_errors=table).save(path)
    loaded = load_emulator(path)
    np.testing.assert_array_equal(
        np.asarray(loaded.predict(jnp.asarray(FULL))), np.asarray(emu.predict(jnp.asarray(FULL)))
    )
    errs = loaded.spec.band_error(5.0)
    assert errs.shape == (3,) and errs[2] == pytest.approx(0.05, rel=0.15)
    view = load_emulator(path, fixed={"redshift": 5.0}, bands=["b3", "b1"])
    v, name = view.spec.band_error(5.0, return_bin=True)
    assert name == "z4-6"
    np.testing.assert_allclose(v, errs[[2, 0]])
    assert "band errors: 3 bands" in loaded.spec.describe()
    with pytest.raises(ValueError, match="not an emulator band"):
        emu.with_metadata(band_errors={"bands": {"zzz": {"all": {"mad": 0.1}}}})
    with pytest.raises(ValueError, match="bands"):
        emu.with_metadata(band_errors={"nope": 1})
