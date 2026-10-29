"""Tests for ``arachne.results.writer`` (forward model -> bundle) on tiny CPU models."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import arachne.results as results
from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.nuisance import NuisanceModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.results import load_bundle, total_sed, validate
from arachne.results.writer import content_from_forward_model, thin_indices, write_bundle
from arachne.spatial.additive import AdditiveComponentModel
from arachne.spatial.correlated_field import CorrelatedFieldMap

BANDS = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F444W"]
LAM = jnp.array([1.15, 2.0, 4.44])
H = W = 16


class ToySED(SPSEmulator, eqx.Module):
    """(N, 4) [redshift, log_mass, Av, logZ] -> (N, 3) nJy, mass-linear."""

    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)

    @property
    def param_names(self):
        """Input names."""
        return self._param_names

    @property
    def band_names(self):
        """Band names."""
        return self._band_names

    def predict(self, params):
        """Power law with metallicity tilt and dust."""
        z, logm, av, logz = (params[:, i : i + 1] for i in range(4))
        tau = av * (LAM[None, :] / (0.55 * (1.0 + z))) ** -1.0
        return 10.0 ** (logm - 6.0) * LAM[None, :] ** (0.5 + 0.5 * logz) * 10.0 ** (-0.4 * tau)


class LinearMassEmulator(SPSEmulator, eqx.Module):
    """flux_b = 10**(logM - 9) * (b + 1) * (1 + 0.1 * tau_v)."""

    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)

    @property
    def param_names(self):
        """Input names."""
        return self._param_names

    @property
    def band_names(self):
        """Band names."""
        return self._band_names

    def predict(self, params):
        """Mass-linear flux with a mild tau_v colour term."""
        m9 = 10.0 ** (params[:, 0:1] - 9.0)
        colour = 1.0 + 0.1 * params[:, 2:3]
        bands = jnp.arange(1, len(self._band_names) + 1, dtype=jnp.float32)[None, :]
        return m9 * colour * bands


def _psf():
    y, x = np.mgrid[-4:5, -4:5]
    ks = [np.exp(-(x**2 + y**2) / (2 * s**2)) for s in (1.0, 1.2, 1.5)]
    return PSFModel(np.stack([k / k.sum() for k in ks]).astype(np.float32), band_names=BANDS)


def _observation(clean: np.ndarray, seed: int = 0) -> ObservationCube:
    sigma = 0.05 * clean.max(axis=(1, 2))
    rng = np.random.default_rng(seed)
    flux = clean + rng.normal(size=clean.shape) * sigma[:, None, None]
    var = np.broadcast_to(sigma[:, None, None] ** 2, clean.shape)
    mask = np.ones(clean.shape, np.float32)
    mask[0, 0, 0] = 0.0
    return ObservationCube(flux.astype(np.float32), var.astype(np.float32), mask, BANDS, 0.03, None)


def _blank() -> ObservationCube:
    z = np.zeros((3, H, W), np.float32)
    return ObservationCube(z, z + 1, z + 1, BANDS, 0.03, None)


@pytest.fixture(scope="module")
def field_setup():
    """CorrelatedFieldMap with a fitted hyperparameter and coupling; truth + samples."""
    names = ["redshift", "log_mass", "Av", "log10metallicity"]
    yy, xx = np.mgrid[0:H, 0:W] - (H - 1) / 2.0
    model = CorrelatedFieldMap(
        image_shape=(H, W),
        emulator_param_names=names,
        param_bounds={"Av": (0.0, 3.0), "log10metallicity": (-2.0, 0.3)},
        field_params=["log_mass", "Av"],
        fixed_params={"redshift": 2.0},
        corr_lengths={"log_mass": 2.0, "Av": 3.0},
        amplitudes={"log_mass": 0.3, "Av": 0.3},
        mean_maps={"log_mass": (8.0 - 0.1 * np.hypot(yy, xx)).astype(np.float32)},
        hyper_priors={"Av": {"log_amplitude": {"loc": np.log(0.3), "scale": 0.3}}},
        free_couplings={"Av": {"log_mass": 0.5}},
    )
    emulator = ToySED(_param_names=names, _band_names=BANDS)
    theta_true = model.sample_prior(jax.random.PRNGKey(0), 1)[0]
    clean = np.asarray(
        ForwardModel.build(_blank(), _psf(), model, emulator)._model_image(theta_true)
    )
    fm = ForwardModel.build(_observation(clean), _psf(), model, emulator, model_error_frac=0.05)
    samples = theta_true[None] + 0.05 * jax.random.normal(
        jax.random.PRNGKey(1), (40, model.n_params)
    )
    return fm, model, theta_true, samples, clean


def _additive_fm(nuisance: NuisanceModel | None):
    names = ["log_stellar_mass", "log_age", "tau_v"]
    bounds = {"log_stellar_mass": (6.0, 12.0), "log_age": (7.0, 10.1), "tau_v": (0.0, 4.0)}
    model = AdditiveComponentModel(
        n_components=2,
        emulator_param_names=names,
        param_bounds=bounds,
        image_shape=(H, W),
        mass_param="log_stellar_mass",
        profiles=["gaussian", "sersic"],
    )
    emulator = LinearMassEmulator(_param_names=names, _band_names=BANDS)
    theta = model.sample_prior(jax.random.PRNGKey(3), 1)[0]
    clean = np.asarray(ForwardModel.build(_blank(), _psf(), model, emulator)._model_image(theta))
    fm = ForwardModel.build(_observation(clean, 2), _psf(), model, emulator, nuisance=nuisance)
    samples = fm.sample_prior(jax.random.PRNGKey(4), 24)
    return fm, model, samples


def test_thin_indices():
    """Even thinning with unique rounded indices."""
    np.testing.assert_array_equal(thin_indices(10, 4), [0, 3, 6, 9])
    np.testing.assert_array_equal(thin_indices(3, 10), [0, 1, 2])
    assert thin_indices(0, 5).size == 0 and thin_indices(5, 0).size == 0
    assert thin_indices(1000, 200).size == 200


def test_field_bundle_invariants(tmp_path, field_setup):
    """Maps, scalars, draws and truth from a CorrelatedFieldMap fit."""
    fm, model, theta_true, samples, clean = field_setup
    truth = results.TruthContent(
        convolved=clean, param_maps=np.asarray(model.field_maps(theta_true))
    )
    path = write_bundle(
        tmp_path / "field.h5",
        fm,
        theta_map=theta_true,
        samples=samples,
        posterior_source="laplace",
        n_draws=10,
        n_chains=2,
        truth=truth,
        diagnostics={"log_evidence_laplace": -12.5, "rhat": np.ones(model.n_params)},
        title="toy field",
    )
    assert validate(path) == []
    b = load_bundle(path)
    assert (
        b.n_draws == 10
        and b.posterior_source == "laplace"
        and b.spatial_model == ("CorrelatedFieldMap")
    )
    np.testing.assert_array_equal(b.draw_index, thin_indices(40, 10))
    np.testing.assert_array_equal(b.chain_id, thin_indices(40, 10) // 20)
    assert b.map_names == ["log_mass", "Av"] and b.mass_param == "log_mass"
    assert b.scalar_names == [
        "log10metallicity",
        "amplitude[Av]",
        "c[Av<-log_mass]",
        "total_log_mass",
    ]
    assert b.posterior_attrs["log_evidence_laplace"] == -12.5
    assert b.posterior_attrs["n_chains"] == 2 and b.posterior_attrs["n_samples_total"] == 40
    assert b.diagnostics["rhat"].shape == (model.n_params,)
    np.testing.assert_allclose(b.model_error_frac, 0.05)
    assert b.redshift == 2.0 and b.fixed_params == {"redshift": 2.0}
    assert {"all", "segment"} <= set(b.regions)
    assert b.provenance["command_line"]

    thetas = b.theta_draws()
    for i in (0, 9):
        np.testing.assert_allclose(
            b.model_draws()[i], np.asarray(fm._model_image(jnp.asarray(thetas[i]))), rtol=1e-5
        )
        np.testing.assert_allclose(
            b.param_draws("Av")[i], np.asarray(model.field_maps(thetas[i]))[1], rtol=1e-5
        )
    sc = b.scalar_draws()
    np.testing.assert_allclose(
        sc["amplitude[Av]"], np.exp([model.hyper_theta(t)[0] for t in thetas]), rtol=1e-5
    )
    np.testing.assert_allclose(
        sc["log10metallicity"], [float(model.global_values(t)[0]) for t in thetas], rtol=1e-5
    )
    lm = b.param_draws("log_mass").astype(float)
    np.testing.assert_allclose(
        sc["total_log_mass"], np.log10((10.0**lm).sum(axis=(1, 2))), rtol=1e-6
    )
    np.testing.assert_allclose(
        b.band_image(1, "model", "map"),
        np.asarray(fm._model_image(theta_true))[1],
        rtol=1e-5,
    )
    assert b.has_intrinsic and b.has_truth
    assert np.isfinite(b.param_map("Av", "diff")).all()
    sed = total_sed(b)
    assert sed.model_draws.shape == (10, 3) and np.isfinite(sed.chi()).all()


def test_map_only_and_best_draw(field_setup):
    """No samples gives an S=0 'map' bundle; no theta_map picks the best draw."""
    fm, _, theta_true, samples, _ = field_setup
    c = content_from_forward_model(
        fm, theta_map=theta_true, posterior_source="nuts", store_intrinsic=False
    )
    assert c.n_draws == 0 and c.posterior_source == "map" and c.intrinsic_map is None
    assert c.scalars_draws is None and c.param_maps_draws is None
    b = results.ResultsBundle.from_content(c)
    assert not b.has_draws and b.band_image(0, "model").shape == (H, W)

    c2 = content_from_forward_model(fm, theta_map=None, samples=samples, n_draws=5)
    logp = [float(fm.log_posterior(samples[i])) for i in thin_indices(40, 5)]
    best = samples[thin_indices(40, 5)[int(np.argmax(logp))]]
    np.testing.assert_allclose(c2.theta_map, np.asarray(best))
    assert c2.provenance["best_point"] == "best_draw"
    with pytest.raises(ValueError):
        content_from_forward_model(fm, theta_map=None)


@pytest.mark.parametrize(
    "nuisance",
    [None, "sky", "sky+shifts"],
)
def test_additive_component_invariants(tmp_path, nuisance):
    """sum_k components + sky == model; sum_k sed x profile == intrinsic; size ordering."""
    nm = None
    if nuisance is not None:
        nm = NuisanceModel(n_bands=3, fit_sky=True, fit_shifts=nuisance == "sky+shifts")
    fm, model, samples = _additive_fm(nm)
    c = content_from_forward_model(
        fm, theta_map=samples[0], samples=samples, n_draws=12, component_draws=5
    )
    path = results.write_content(tmp_path / "add.h5", c)
    assert validate(path) == []
    b = load_bundle(path)
    comps = b.components
    assert comps.n_components == 2 and comps.profile_names == ["gaussian", "sersic"]
    assert comps.param_names[:6] == ["mu_y", "mu_x", "size_y", "size_x", "rho", "sersic_n"]
    assert comps.param_names[6:] == ["log_stellar_mass", "log_age", "tau_v"]
    assert b.map_names == [] and b.scalar_names == ["total_log_mass"]
    params = comps.params()
    assert params.shape == (12, 2, 9)
    assert np.isnan(params[:, 0, 5]).all() and np.isfinite(params[:, 1, 5]).all()

    sky_map = b.nuisance.get("sky_map", np.zeros(3))
    total = sum(comps.image(k, "convolved", "map") for k in range(2))
    scale = np.abs(b.cube("model", "map")).max()
    np.testing.assert_allclose(
        total + sky_map[:, None, None], b.cube("model", "map"), atol=1e-5 * scale
    )
    intr = sum(comps.image(k, "intrinsic", "map") for k in range(2))
    np.testing.assert_allclose(intr, b.cube("intrinsic", "map"), atol=1e-5 * scale)

    cidx = comps.draw_index
    np.testing.assert_array_equal(cidx, thin_indices(12, 5))
    conv_draws = sum(comps.image_draws(k, "convolved") for k in range(2))
    sky = b.nuisance["sky"][cidx] if "sky" in b.nuisance else np.zeros((len(cidx), 3))
    np.testing.assert_allclose(
        conv_draws + sky[:, :, None, None], b.model_draws()[cidx], atol=1e-5 * scale
    )
    int_draws = sum(comps.image_draws(k, "intrinsic") for k in range(2))
    np.testing.assert_allclose(int_draws, b.model_draws("intrinsic")[cidx], atol=1e-5 * scale)
    thetas = b.theta_draws()
    np.testing.assert_allclose(
        b.model_draws()[3], np.asarray(fm._model_image(jnp.asarray(thetas[3]))), atol=1e-5 * scale
    )
    assert comps.sed_intrinsic().shape == (12, 2, 3)
    mass = params[:, :, 6]
    np.testing.assert_allclose(
        b.scalar_draws()["total_log_mass"], np.log10((10.0**mass).sum(axis=1)), rtol=1e-6
    )
    if nuisance == "sky+shifts":
        assert b.nuisance["shifts"].shape == (12, 3, 2)


def test_multires_raises():
    """A multi-resolution forward model is reserved (grid='per_band')."""

    class FakeMultires:
        model_images = None

    with pytest.raises(NotImplementedError, match="per_band"):
        content_from_forward_model(FakeMultires(), theta_map=np.zeros(3))


def test_lazy_exports():
    """write_bundle resolves lazily from arachne.results and arachne."""
    import arachne

    assert results.write_bundle is write_bundle
    assert arachne.write_bundle is write_bundle
    assert arachne.load_bundle is results.load_bundle
