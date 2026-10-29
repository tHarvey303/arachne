"""Tests for ``scripts/convert_correlated_field_run.py`` on a fake old-demo run directory."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from arachne.results import load_bundle, validate

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_correlated_field_run.py"


@pytest.fixture(scope="module")
def convert():
    """The converter script imported as a module."""
    spec = importlib.util.spec_from_file_location("convert_run", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_run(root: Path, *, nuts: bool, bands: bool) -> Path:
    rng = np.random.default_rng(0)
    nb, n = 8, 12
    run = root / ("run_nuts" if nuts else "run_laplace")
    run.mkdir()
    maps = rng.normal(size=(4, n, n))
    maps[0] += 7.0
    model = np.abs(rng.normal(size=(nb, n, n))) + 1.0
    sigma = np.full(nb, 0.1)
    data = model + rng.normal(size=model.shape) * 0.1
    np.savez(
        run / "maps.npz",
        truth=maps + 0.01,
        start=maps * 0,
        map=maps.astype(np.float32),
        model_image=model,
        data=data.astype(np.float32),
        sigma=sigma,
        theta_map=np.arange(30, dtype=np.float32),
        fields=np.array(["log_mass", "Av", "log10metallicity", "logsfr_ratio_0"]),
    )
    summ = dict(median=maps, std=np.full_like(maps, 0.1), information=np.full_like(maps, 0.9))
    np.savez(run / "laplace.npz", theta=np.arange(30.0), **summ)
    if nuts:
        np.savez(run / "nuts.npz", **{k: v * 2 for k, v in summ.items()})
    metrics = {"truth": "prior", "chi2_red_per_band": [1.0] * nb, "laplace": {"x": 1}}
    if bands:
        filters = ("090W", "115W", "150W", "200W", "277W", "356W", "410M", "444W")
        metrics["bands"] = [f"JWST/NIRCam.F{w}" for w in filters]
    (run / "metrics.json").write_text(json.dumps(metrics))
    return run


def test_convert_nuts_run(tmp_path, convert):
    """NUTS summaries are primary, Laplace goes to /extra, truth and regions are set."""
    run = _fake_run(tmp_path, nuts=True, bands=False)
    path = convert.convert_run(run)
    assert path == run / "results_degraded.h5" and validate(path) == []
    b = load_bundle(path)
    assert b.degraded and b.posterior_source == "map" and b.n_draws == 0
    assert b.short_names[0] == "F090W" and b.title == "prior_wide8"
    assert b.map_names == ["log_mass", "Av", "log10metallicity", "logsfr_ratio_0"]
    maps = np.load(run / "maps.npz")
    np.testing.assert_allclose(b.param_map("Av", "p50"), 2 * maps["map"][1], rtol=1e-6)
    np.testing.assert_allclose(b.extra("laplace_p50")[1], maps["map"][1], rtol=1e-6)
    np.testing.assert_allclose(b.param_map("Av", "truth"), maps["truth"][1], rtol=1e-6)
    np.testing.assert_allclose(b.band_image(2, "sigma"), 0.1, rtol=1e-6)
    np.testing.assert_allclose(b.band_image(2, "model", "map"), maps["model_image"][2], rtol=1e-6)
    assert {"all", "galaxy"} <= set(b.regions)
    assert b.fixed_params["fesc_lya"] == 0.2 and b.redshift == 2.0
    assert not b.has_intrinsic and b.metrics["laplace"] == {"x": 1}
    assert "start_maps" in b.extra_names and "laplace_theta" in b.extra_names


def test_convert_laplace_run_with_toy_emulator(tmp_path, convert):
    """Laplace-only run with an explicit band list and the toy intrinsic model."""
    run = _fake_run(tmp_path, nuts=False, bands=True)
    out = tmp_path / "x.h5"
    assert convert.main([str(run), "--out", str(out), "--emulator", "toy"]) == 0
    b = load_bundle(out)
    assert b.has_intrinsic and b.truth.intrinsic is not None
    assert b.available_param_stats() == ["map", "p50", "std", "truth", "diff", "z", "information"]
    assert np.isfinite(b.cube("intrinsic", "map")).all()
    assert not [n for n in b.extra_names if n.startswith("nuts")]


def test_convert_rejects_new_runs(tmp_path, convert):
    """Runs with a different field list are refused."""
    run = _fake_run(tmp_path, nuts=False, bands=False)
    z = dict(np.load(run / "maps.npz"))
    z["fields"] = np.array(["log_mass", "Av", "log10metallicity", "logsfr_ratio_3"])
    np.savez(run / "maps.npz", **z)
    with pytest.raises(ValueError, match="old 4-field"):
        convert.convert_run(run)
