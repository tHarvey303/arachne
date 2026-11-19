"""Simulator recipe, checkpoint storage and the bundle post-processing container.

The Synthesizer-backed tests run only when ``ARACHNE_TEST_SIMULATOR_CKPT`` names a
checkpoint with a recipe and its SPS grid is findable (``ARACHNE_SPS_GRID_DIR``).
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import pytest

from arachne.emulator.parrot_emulator_v2 import ParrotEmulatorV2
from arachne.postprocess import recipe as rc
from arachne.postprocess.bundle import PostprocessResult, load_postprocess

V6_NAMES = [
    "redshift",
    "log_mass",
    "slope",
    "fesc_lya",
    "dust_bump_amplitude",
    "log10metallicity",
    "Av",
    *[f"logsfr_ratio_{i}" for i in range(5)],
]
BANDS = ["JWST/NIRCam.F150W", "JWST/NIRCam.F444W"]


def _fake_library(path, n=400, seed=0):
    """A synference-style library file with the v6 header fields the recipe reader uses."""
    from astropy.cosmology import Planck18

    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.attrs["ParameterNames"] = V6_NAMES
        f.attrs["FilterCodes"] = [*BANDS, "JWST/NIRCam.F277W"]
        f.attrs["SupplementaryParameterNames"] = ["sfr_100", "mass_weighted_age"]
        params = rng.uniform(0.1, 1.0, (len(V6_NAMES), n))
        params[0, 3] = np.nan  # a bad row, to be skipped
        f["Grid/Parameters"] = params
        f["Grid/Photometry"] = rng.uniform(1, 100, (3, n))
        f["Grid/SupplementaryParameters"] = rng.uniform(1, 10, (2, n))
        m = f.create_group("Model")
        m.attrs["grid_name"] = "some-grid"
        m.attrs["cosmology"] = Planck18.to_format("yaml")
        m.attrs["sfh_class"] = "Continuity"
        m.attrs["metallicity_distribution_class"] = "DeltaConstant"
        m.attrs["emission_model_key"] = "total"
        m.create_group("EmissionModel").attrs["name"] = "PacmanEmissionNoEscapedWithDust"
    return path


def _tiny():
    """Untrained 12-input, 2-band V2 emulator with the v6 library's names."""
    import jax

    cfg = {
        "arch": "mlp",
        "width": 8,
        "depth": 2,
        "blocks": 1,
        "flux_floor": 1e-4,
        "fourier_k": 2,
        "z_scale": 14.0,
        "i_mass": 1,
        "i_z": 0,
        "keep_idx": [0, *range(2, 12)],
        "keep_mass_input": False,
        "mass_norm": True,
        "mu0_train": 19.5,
        "sfr_arsinh": True,
        "sfr_cols": [6, 7, 8, 9, 10],
    }
    return ParrotEmulatorV2(
        V6_NAMES,
        BANDS,
        cfg,
        in_mean=np.zeros(15, np.float32),
        in_std=np.ones(15, np.float32),
        out_mean=np.full(2, 15.0, np.float32),
        key=jax.random.PRNGKey(0),
    )


def test_agebins_match_library_rule():
    """Age bins reproduce the v6 library's bins at the redshift of a fitted galaxy."""
    from astropy.cosmology import Planck18

    bins = rc.continuity_agebins(1.8973807, Planck18, rc.PRESETS["v6"]["agebins"])
    assert bins.shape == (6, 2)
    np.testing.assert_allclose(bins[:, 1], [3, 10, 65, 426, 2779, 3270], atol=1.0)
    np.testing.assert_allclose(bins[1:, 0], bins[:-1, 1])
    # the last bin is the oldest 15 % of the available time
    assert (bins[-1, 1] - bins[-1, 0]) / bins[-1, 1] == pytest.approx(0.15)
    with pytest.raises(ValueError):
        rc.continuity_agebins(19.99, Planck18, rc.PRESETS["v6"]["agebins"])


def test_continuity_sfh_mass_and_ratios():
    """Bin masses sum to the formed mass and the ratios follow the Prospector convention."""
    bins = np.array([[0, 3], [3, 10], [10, 100.0]])
    sfr, mass = rc.continuity_sfh(9.0, [0.5, -1.0], bins)
    assert mass.sum() == pytest.approx(1e9)
    assert np.log10(sfr[0] / sfr[1]) == pytest.approx(0.5)
    assert np.log10(sfr[1] / sfr[2]) == pytest.approx(-1.0)
    np.testing.assert_allclose(sfr * (bins[:, 1] - bins[:, 0]) * 1e6, mass)


def test_recipe_from_library_and_checks(tmp_path):
    """A recipe built from a library header validates, hashes its physics and rejects bad input."""
    lib = _fake_library(tmp_path / "library_fake_v6_x.hdf5")
    assert rc.guess_preset(lib) == "v6"
    recipe = rc.recipe_from_library(lib, band_names=BANDS, n_reference=16)
    rc.check_recipe_structure(recipe)
    ref = recipe["reference"]
    assert np.shape(ref["params"]) == (16, 12)
    assert np.shape(ref["photometry_nJy"]) == (16, 2)
    assert np.all(np.isfinite(ref["params"]))
    assert set(ref["derived"]) == {"sfr_100", "age_mass_weighted"}
    assert recipe["igm"]["class"] == "Asada25" and recipe["sfh"]["agebins"]["z_form"] == 20.0
    json.dumps(recipe)
    # the hash covers the physics, not the reference rows
    other = {**recipe, "reference": {}}
    assert rc.recipe_hash(other) == rc.recipe_hash(recipe)
    changed = json.loads(json.dumps(recipe))
    changed["sfh"]["agebins"]["z_form"] = 30.0
    assert rc.recipe_hash(changed) != rc.recipe_hash(recipe)
    bad = {k: v for k, v in recipe.items() if k != "igm"}
    with pytest.raises(ValueError, match="lacks"):
        rc.check_recipe_structure(bad)
    bad = json.loads(json.dumps(recipe))
    bad["stellar_params"]["tau_v"] = {"product": ["A_V"]}
    with pytest.raises(ValueError, match="unknown parameters"):
        rc.check_recipe_structure(bad)
    with pytest.raises(ValueError, match="preset"):
        rc.recipe_from_library(_fake_library(tmp_path / "library_other.hdf5"))


def test_params_and_stellar_attributes(tmp_path):
    """Fit parameters map to library inputs through aliases; emitter attributes follow."""
    recipe = rc.recipe_from_library(_fake_library(tmp_path / "lib_v6.hdf5"), n_reference=4)
    fit = dict.fromkeys(V6_NAMES, 0.3)
    fit.pop("slope")
    fit.update(dust_slope=-0.2, Av=1.086, total_log_mass=11.0)  # view name + an extra entry
    p = rc.resolve_params(recipe, fit)
    assert list(p) == V6_NAMES and p["slope"] == -0.2
    attrs = rc.stellar_attributes(recipe, p)
    assert attrs["tau_v"] == pytest.approx(1.0) and attrs["slope"] == -0.2
    fit.pop("Av")
    with pytest.raises(KeyError, match="Av"):
        rc.resolve_params(recipe, fit)


def test_recipe_in_checkpoint(tmp_path):
    """The recipe survives a checkpoint save / load and is readable without JAX."""
    recipe = rc.recipe_from_library(_fake_library(tmp_path / "lib_v6.hdf5"), band_names=BANDS)
    emu = _tiny()
    assert emu.spec.simulator is None and "simulator: none" in emu.spec.describe()
    path = tmp_path / "emu.eqx"
    emu.with_metadata(simulator=recipe).save(path)
    loaded = ParrotEmulatorV2.load(path)
    assert loaded.spec.simulator == recipe
    assert "16 reference rows" not in loaded.spec.describe()
    assert "48 reference rows" in loaded.spec.describe()
    assert rc.recipe_from_checkpoint(path) == recipe  # no JAX needed for this read
    short = {**recipe, "param_names": V6_NAMES[:-1]}
    with pytest.raises(ValueError):
        emu.with_metadata(simulator=short)


def test_postprocess_result_roundtrip(tmp_path):
    """A PostprocessResult written into a bundle file reads back unchanged."""
    rng = np.random.default_rng(1)
    mask = np.zeros((6, 5), bool)
    mask[1:4, 1:4] = True
    maps = {"log_mass_surviving": rng.normal(8, 0.1, (3, 6, 5)).astype(np.float32)}
    maps["log_mass_surviving"][:, ~mask] = np.nan
    res = PostprocessResult(
        region="segment",
        regions={"segment": mask},
        draw_index=np.array([0, 5, 9]),
        band_names=BANDS,
        wavelength_um=np.linspace(0.5, 5, 40),
        sed={"segment": rng.uniform(1, 2, (3, 40))},
        photometry={"segment": rng.uniform(1, 2, (3, 2))},
        maps=maps,
        totals={"segment": {"log_mass_surviving": np.array([9.0, 9.1, 9.2])}},
        sfh={"segment": rng.uniform(1, 2, (3, 6))},
        sfh_agebins_myr=np.arange(12.0).reshape(6, 2),
        emulator_check={"total": np.array([1.0, 0.99])},
        attrs={"recipe_hash": "abc", "from_draws": True, "check_status": "exact"},
    )
    path = tmp_path / "results.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "arachne.results"
    assert load_postprocess(path) is None
    res.write(path)
    res.write(path)  # replaces the group
    back = load_postprocess(path)
    assert back.region == "segment" and back.band_names == BANDS and back.n_draws == 3
    np.testing.assert_allclose(back.sed["segment"], res.sed["segment"], rtol=1e-6)
    np.testing.assert_array_equal(back.regions["segment"], mask)
    assert back.total("log_mass_surviving")[1] == pytest.approx(9.1)
    assert np.isnan(back.stat(back.maps["log_mass_surviving"])[0, 0])
    assert back.attrs["check_status"] == "exact"
    with h5py.File(path) as f:  # percentiles are stored next to the draws
        assert f["postprocess/sed/segment/p50"].shape == (40,)


@pytest.mark.skipif(
    not os.getenv("ARACHNE_TEST_SIMULATOR_CKPT"),
    reason="set ARACHNE_TEST_SIMULATOR_CKPT (and ARACHNE_SPS_GRID_DIR) to test against Synthesizer",
)
def test_simulator_reproduces_library():
    """The rebuilt simulator matches the reference library rows and returns derived values."""
    pytest.importorskip("synthesizer")
    from arachne.postprocess.simulator import DERIVED_NAMES, Simulator, check_recipe

    recipe = rc.recipe_from_checkpoint(os.environ["ARACHNE_TEST_SIMULATOR_CKPT"])
    bands = recipe["reference"]["band_names"]
    sim = Simulator(recipe, band_names=bands)
    report = check_recipe(recipe, simulator=sim)
    assert report["status"] == "exact", report
    params = dict(zip(recipe["param_names"], recipe["reference"]["params"][0]))
    out = sim.evaluate(params)
    assert out["spectrum"].shape == out["wavelength_um"].shape
    derived = dict(zip(DERIVED_NAMES, out["derived"]))
    assert derived["log_mass_surviving"] < derived["log_mass_formed"]
    # The analytic SFH and Synthesizer's SFR over 0-10 Myr agree to the discretisation of the
    # SFH on the grid's age bins (a few per cent).
    bins, sfr, mass = sim.sfh(params)
    assert mass[:2].sum() / 1e7 == pytest.approx(derived["sfr_10"], rel=0.1)
