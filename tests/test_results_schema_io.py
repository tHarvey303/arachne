"""Round-trip, versioning and validation tests for ``arachne.results`` bundles."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from arachne.results import (
    SCHEMA_VERSION,
    Region,
    ResultsBundle,
    load_bundle,
    make_synthetic_bundle,
    make_synthetic_content,
    pivot_um,
    short_band_name,
    validate,
    write_content,
)
from arachne.results.io import round_mantissa
from arachne.results.reader import IMAGE_KINDS

FLAVOURS = ("field", "additive", "degraded")


def _assert_bundles_equal(a: ResultsBundle, b: ResultsBundle) -> None:
    """Every public accessor gives the same answer for two bundles."""
    assert a.band_names == b.band_names
    assert a.short_names == b.short_names
    np.testing.assert_array_equal(a.pivot_um, b.pivot_um)
    assert a.pixel_scale == b.pixel_scale
    assert a.image_shape == b.image_shape
    assert (a.n_draws, a.degraded, a.posterior_source) == (
        b.n_draws,
        b.degraded,
        b.posterior_source,
    )
    assert (a.has_draws, a.has_intrinsic, a.has_truth, a.has_components) == (
        b.has_draws,
        b.has_intrinsic,
        b.has_truth,
        b.has_components,
    )
    for name in ("flux", "variance", "mask", "model_error_frac"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.map_names == b.map_names and a.scalar_names == b.scalar_names
    assert a.labels == b.labels and a.units == b.units
    assert a.metrics == b.metrics
    assert a.fixed_params == b.fixed_params and a.mass_param == b.mass_param
    assert set(a.regions) == set(b.regions)
    for name, reg in a.regions.items():
        np.testing.assert_array_equal(reg.mask, b.regions[name].mask)
        assert reg.kind == b.regions[name].kind and reg.geometry == b.regions[name].geometry
    for b_idx in range(a.n_bands):
        for kind in IMAGE_KINDS:
            for stat in ("map", "p50", "std"):
                try:
                    ia = a.band_image(b_idx, kind, stat)
                except KeyError:
                    with pytest.raises(KeyError):
                        b.band_image(b_idx, kind, stat)
                    continue
                ib = b.band_image(b_idx, kind, stat)
                np.testing.assert_array_equal(ia, ib)
    assert a.available_param_stats() == b.available_param_stats()
    for name in a.map_names:
        for stat in a.available_param_stats():
            np.testing.assert_array_equal(a.param_map(name, stat), b.param_map(name, stat))
    np.testing.assert_array_equal(a.chi2_red_per_band(), b.chi2_red_per_band())
    if a.has_draws:
        np.testing.assert_array_equal(a.model_draws(), b.model_draws())
        np.testing.assert_array_equal(a.band_draws(1), b.band_draws(1))
    for k, v in a.scalar_draws().items():
        np.testing.assert_array_equal(v, b.scalar_draws()[k])
    pa, pb = a.pixel_param_draws(3, 4), b.pixel_param_draws(3, 4)
    assert pa.keys() == pb.keys()
    for k in pa:
        np.testing.assert_array_equal(pa[k], pb[k])
    if a.truth is not None:
        for f in ("convolved", "intrinsic", "param_maps", "scalars", "component_seds"):
            va, vb = getattr(a.truth, f), getattr(b.truth, f)
            assert (va is None) == (vb is None)
            if va is not None:
                np.testing.assert_array_equal(va, vb)
    if a.components is not None:
        ca, cb = a.components, b.components
        assert ca.n_components == cb.n_components and ca.param_names == cb.param_names
        np.testing.assert_array_equal(ca.sed_intrinsic(), cb.sed_intrinsic())
        np.testing.assert_array_equal(ca.params(), cb.params())
        for k in range(ca.n_components):
            for frame in ("intrinsic", "convolved"):
                np.testing.assert_allclose(ca.image(k, frame), cb.image(k, frame))


@pytest.mark.parametrize("flavour", FLAVOURS)
def test_round_trip_matches_in_memory(tmp_path, flavour):
    """File round trip gives the same answers as from_content."""
    content = make_synthetic_content(flavour)
    mem = ResultsBundle.from_content(content)
    path = write_content(tmp_path / "b.h5", content)
    assert validate(path) == []
    with load_bundle(path) as disk:
        _assert_bundles_equal(mem, disk)
        np.testing.assert_array_equal(disk.flux, content.flux.astype(np.float32))
        np.testing.assert_array_equal(
            disk.band_image(0, "model", "map"), content.convolved_map[0].astype(np.float32)
        )
        assert disk.title == f"synthetic {flavour}"
        assert disk.provenance["numpy_version"] == np.__version__


def test_field_flavour_contents():
    """The field flavour matches the spec."""
    b = ResultsBundle.from_content(make_synthetic_content("field", n_draws=7))
    assert b.n_draws == 7 and b.has_draws and b.has_intrinsic and b.has_truth
    assert b.map_names == ["log_mass", "Av"] and b.scalar_names == ["total_log_mass"]
    assert b.mask[0, 0, 0] == 0 and b.mask.sum() == b.mask.size - 1
    assert set(b.regions) == {"all", "core"}
    assert b.regions["core"].kind == "circle"
    assert b.model_draws().shape == (7, 3, 16, 16)
    assert b.param_draws("Av").shape == (7, 16, 16) and (b.param_draws("Av") >= 0).all()
    np.testing.assert_allclose(
        b.param_map("log_mass", "map") - b.param_map("log_mass", "truth"), 0.02, atol=1e-6
    )
    np.testing.assert_allclose(b.param_map("Av", "prior_std"), 0.5)
    info = 1 - (b.param_map("Av", "std") / 0.5) ** 2
    np.testing.assert_allclose(b.param_map("Av", "information"), info, rtol=1e-5)
    z = (b.param_map("Av", "p50") - b.param_map("Av", "truth")) / b.param_map("Av", "std")
    np.testing.assert_allclose(b.param_map("Av", "z"), z, rtol=1e-6)
    assert b.metrics["chi2_red"] > 0


def test_additive_flavour_contents():
    """The additive flavour matches the spec."""
    b = ResultsBundle.from_content(make_synthetic_content("additive"))
    comps = b.components
    assert comps is not None and comps.n_components == 2 and comps.factorised
    assert comps.param_names[:6] == ["mu_y", "mu_x", "size_y", "size_x", "rho", "sersic_n"]
    assert b.map_names == [] and b.available_param_stats() == []
    assert comps.image_draws(0, "convolved").shape == (20, 3, 16, 16)
    total = sum(comps.image(k, "convolved", "map") for k in range(2))
    np.testing.assert_allclose(total, b.cube("model", "map"), rtol=1e-5, atol=1e-6)
    intr = sum(comps.image(k, "intrinsic", "map") for k in range(2))
    np.testing.assert_allclose(intr, b.cube("intrinsic", "map"), rtol=1e-5, atol=1e-7)
    assert comps.params().shape == (20, 2, 8)
    np.testing.assert_allclose(comps.params()[0, :, 2], [1.5, 4.0])


def test_degraded_fallbacks():
    """Degraded bundles fall back to MAP and report it."""
    b = ResultsBundle.from_content(make_synthetic_content("degraded"))
    assert b.degraded and not b.has_draws and not b.has_intrinsic
    assert b.posterior_source == "map"
    img = b.band_image(0, "model", "p50")
    assert b.last_fallback
    np.testing.assert_array_equal(img, b.band_image(0, "model", "map"))
    assert not b.last_fallback
    with pytest.raises(KeyError):
        b.band_image(0, "model", "std")
    with pytest.raises(KeyError):
        b.band_draws(0)
    with pytest.raises(KeyError):
        b.band_image(0, "intrinsic", "map")
    assert b.available_param_stats() == ["map", "p50", "std", "truth", "diff", "z", "information"]
    b.param_map("Av", "p50")
    assert not b.last_fallback
    b.param_map("Av", "mean")
    assert b.last_fallback
    with pytest.raises(KeyError):
        b.param_map("Av", "prior_std")
    assert b.pixel_param_draws(2, 2)["Av"].shape == (1,)
    assert b.scalar_draws()["total_log_mass"].shape == (1,)


def test_band_image_semantics():
    """Residual, chi (with/without floor), masking and noise scale."""
    content = make_synthetic_content("field")
    content.nuisance = {"log_noise_scale_map": np.log(np.array([1.0, 2.0, 1.5]))}
    b = ResultsBundle.from_content(content)
    flux, var = b.flux[1].astype(float), b.variance[1].astype(float)
    model = b.band_image(1, "model", "p50")
    res = b.band_image(1, "residual", "p50")
    np.testing.assert_allclose(res, flux - model)
    var_eff = var * 4.0 + (0.05 * model) ** 2
    np.testing.assert_allclose(b.band_image(1, "chi"), (flux - model) / np.sqrt(var_eff))
    np.testing.assert_allclose(
        b.band_image(1, "chi", model_floor=False), (flux - model) / np.sqrt(var)
    )
    assert np.isnan(b.band_image(0, "data")[0, 0]) and np.isnan(b.band_image(0, "chi")[0, 0])
    assert np.isfinite(b.band_image(1, "data")[0, 0])
    assert np.isfinite(b.band_image(0, "model")[0, 0])
    assert b.band_image(0, "mask")[0, 0] == 0
    np.testing.assert_allclose(b.band_image(2, "sigma") ** 2, b.band_image(2, "variance"))
    assert b.cube("chi").shape == (3, 16, 16)
    with pytest.raises(ValueError):
        b.band_image(0, "nonsense")


def test_optional_pieces_absent(tmp_path):
    """Missing optional groups read as 'not available'."""
    c = make_synthetic_content("field", with_truth=False)
    c.intrinsic_map = c.intrinsic_draws = None
    c.prior_std_maps = None
    c.regions = {}
    c.metrics = None
    c.theta_map = c.theta_draws = None
    c.scalar_names, c.scalars_map, c.scalars_draws = [], None, None
    path = write_content(tmp_path / "min.h5", c)
    assert validate(path) == []
    b = load_bundle(path)
    assert not b.has_truth and not b.has_intrinsic and not b.has_components
    assert b.regions == {} and b.metrics == {} and b.scalar_draws() == {}
    assert b.theta_map is None and b.theta_draws() is None
    assert "information" not in b.available_param_stats()
    assert "truth" not in b.available_param_stats()
    with pytest.raises(KeyError):
        b.band_image(0, "truth_model")
    with pytest.raises(KeyError):
        b.param_map("Av", "z")
    with pytest.raises(KeyError):
        b.model_draws("intrinsic")


def test_draws_load_lazily(tmp_path):
    """Draws are read on first use and memoised."""
    path = make_synthetic_bundle(tmp_path / "f.h5", "field")
    b = load_bundle(path)
    assert "model/convolved/draws" not in b._cache
    assert "model/convolved/draws" not in b._store._data
    d = b.model_draws()
    assert b.model_draws() is d  # memoised
    b.close()
    assert "model/convolved/draws" not in b._cache


def test_version_checks(tmp_path):
    """Major mismatch raises, newer minor warns, reserved grid raises."""
    path = make_synthetic_bundle(tmp_path / "v.h5", "field")
    with h5py.File(path, "a") as f:
        f.attrs["schema_version"] = "1.7"
    with pytest.warns(UserWarning, match="newer"):
        load_bundle(path)
    with h5py.File(path, "a") as f:
        f.attrs["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="incompatible"):
        load_bundle(path)
    assert any("schema_version" in p for p in validate(path))
    with h5py.File(path, "a") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["schema"] = "something.else"
    with pytest.raises(ValueError, match="Not an"):
        load_bundle(path)
    with h5py.File(path, "a") as f:
        f.attrs["schema"] = "arachne.results"
        f.attrs["grid"] = "per_band"
    with pytest.raises(NotImplementedError):
        load_bundle(path)


def test_unknown_groups_ignored(tmp_path):
    """Unknown groups do not break the reader."""
    path = make_synthetic_bundle(tmp_path / "u.h5", "additive")
    with h5py.File(path, "a") as f:
        f.create_dataset("future/thing", data=np.arange(3))
    b = load_bundle(path)
    assert b.has_components


def test_mantissa_rounding(tmp_path):
    """12-bit rounding of draws keeps relative error below 3e-4."""
    content = make_synthetic_content("field")
    path = write_content(tmp_path / "m.h5", content, mantissa_bits=12)
    b = load_bundle(path)
    exact = content.convolved_draws.astype(np.float32)
    rel = np.abs(b.model_draws() - exact) / np.abs(exact)
    assert rel.max() < 3e-4 and rel.max() > 0
    # summaries and maps are not rounded
    mem = ResultsBundle.from_content(content)
    np.testing.assert_array_equal(
        b.band_image(0, "model", "p50"), mem.band_image(0, "model", "p50")
    )
    np.testing.assert_array_equal(
        b.band_image(0, "model", "map"), mem.band_image(0, "model", "map")
    )
    x = np.array([1.0, -3.3, np.nan, np.inf, 0.0], dtype=np.float32)
    y = round_mantissa(x, 12)
    assert np.isnan(y[2]) and np.isinf(y[3]) and y[4] == 0
    with pytest.raises(ValueError):
        round_mantissa(x, 0)


def test_large_metrics_json(tmp_path):
    """Metrics JSON above the 64 KB attribute limit round-trips."""
    content = make_synthetic_content("degraded")
    content.metrics = {"big": list(range(30000)), "nested": {"a": [1.5, 2.5]}}
    path = write_content(tmp_path / "big.h5", content)
    with h5py.File(path, "r") as f:
        assert len(f["metrics/json"].asstr()[()]) > 64 * 1024
    b = load_bundle(path)
    assert b.metrics["big"][-1] == 29999 and b.metrics["nested"]["a"] == [1.5, 2.5]


def test_overwrite_and_compression(tmp_path):
    """Overwrite protection, compression and chunking."""
    path = make_synthetic_bundle(tmp_path / "o.h5", "field")
    with pytest.raises(FileExistsError):
        write_content(path, make_synthetic_content("field"))
    write_content(path, make_synthetic_content("field"), overwrite=True)
    with h5py.File(path, "r") as f:
        ds = f["model/convolved/draws"]
        assert ds.compression == "gzip" and ds.shuffle and ds.chunks == (1, 3, 16, 16)
        assert f["regions/core"].dtype == np.uint8
        assert json.loads(f["regions/core"].attrs["geometry"])["r"] == 3.0
        assert list(f["observation"].attrs["band_names"]) == [
            "JWST/NIRCam.F115W",
            "JWST/NIRCam.F200W",
            "JWST/NIRCam.F444W",
        ]


def test_validate_reports_problems(tmp_path):
    """validate() reports missing and mis-shaped datasets."""
    path = make_synthetic_bundle(tmp_path / "bad.h5", "field")
    with h5py.File(path, "a") as f:
        del f["model/convolved/map"]
        del f["params/maps/std"]
        f.create_dataset("params/maps/std", data=np.zeros((2, 5, 5), dtype=np.float32))
        f.attrs["n_draws"] = 3
    problems = validate(path)
    assert any("model/convolved/map" in p for p in problems)
    assert any("params/maps/std" in p for p in problems)
    assert any("model/convolved/draws" in p for p in problems)


def test_content_validate_raises():
    """BundleContent.validate catches shape errors."""
    c = make_synthetic_content("field")
    c.variance = c.variance[:, :5]
    with pytest.raises(ValueError, match="variance"):
        c.validate()
    c = make_synthetic_content("field")
    c.map_names = ["log_mass"]
    with pytest.raises(ValueError, match="param_maps_map"):
        ResultsBundle.from_content(c)
    c = make_synthetic_content("field")
    c.regions = {"bad": Region.from_mask(np.ones((4, 4)))}
    with pytest.raises(ValueError, match="region"):
        c.validate()


def test_metadata_helpers():
    """Band-name and pivot lookups."""
    assert short_band_name("JWST/NIRCam.F115W") == "F115W"
    assert short_band_name("F444W") == "F444W"
    piv = pivot_um(["JWST/NIRCam.F115W", "F444W", "HST/ACS_WFC.F814W", "f160w", "nope"])
    np.testing.assert_allclose(piv[:4], [1.154, 4.402, 0.8045, 1.5369])
    assert np.isnan(piv[4])


def test_lazy_writer_attribute():
    """write_bundle resolves lazily; unknown names raise AttributeError."""
    import arachne.results as results

    assert callable(results.write_bundle)
    with pytest.raises(AttributeError):
        results.no_such_name  # noqa: B018


def test_cli_info_and_validate(tmp_path, capsys):
    """``arachne-bundle info`` / ``validate`` (python -m arachne.results)."""
    from arachne.results.cli import main

    good = make_synthetic_bundle(tmp_path / "good.h5", "additive")
    assert main(["info", str(good)]) == 0
    assert "AdditiveComponentModel" in capsys.readouterr().out
    assert main(["info", str(good), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["components"]["n"] == 2 and info["n_draws"] == 20
    bad = make_synthetic_bundle(tmp_path / "bad.h5", "field")
    with h5py.File(bad, "a") as f:
        del f["model/convolved/map"]
    assert main(["validate", str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert f"OK {good}" in out and f"INVALID {bad}" in out
