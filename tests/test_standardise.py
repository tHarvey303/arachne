"""Tests for arachne.inference.standardise (prior-standardising transform)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.nuisance import NuisanceModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.standardise import (
    StandardisedLogDensity,
    StandardisingTransform,
    standard_normal_logpdf,
)
from arachne.spatial.correlated_field import CorrelatedFieldMap

NAMES = ["redshift", "log_mass", "log10metallicity", "Av", "fesc", "logsfr_ratio_0"]
BOUNDS = {
    "log10metallicity": (-2.0, 0.3),
    "Av": (0.0, 3.0),
    "fesc": (0.0, 1.0),
    "logsfr_ratio_0": (-10.0, 10.0),
}
BANDS = ["F115W", "F200W", "F277W", "F444W"]
LAM = jnp.array([1.15, 2.0, 2.77, 4.44])


class _ToyEmulator(SPSEmulator, eqx.Module):
    """Mass-linear toy: metallicity tilts the colour, Av reddens."""

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
        """(N, 6) -> (N, 4) nJy."""
        z, logm, logz, av = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
        beta = 0.5 + 0.8 * (logz + 0.5)
        tau = av * (LAM[None, :] / (0.55 * (1.0 + z))) ** -1.0
        return 10.0 ** (logm - 6.0) * LAM[None, :] ** beta * 10.0 ** (-0.4 * tau)


def _build(nuisance: NuisanceModel | None = None, H: int = 10, W: int = 10) -> ForwardModel:
    """Model with uniform / Normal / Student-t globals, a fitted hyperparameter and a coupling."""
    model = CorrelatedFieldMap(
        image_shape=(H, W),
        emulator_param_names=NAMES,
        param_bounds=BOUNDS,
        field_params=["log_mass", "Av"],
        fixed_params={"redshift": 2.0},
        corr_lengths=3.0,
        amplitudes={"log_mass": 0.3, "Av": 0.2},
        mean_maps={"log_mass": 7.0, "Av": 1.0},
        hyper_priors={"Av": {"log_amplitude": {"loc": np.log(0.2), "scale": 0.5}}},
        free_couplings={"Av": {"log_mass": 0.5}},
        global_priors={
            "log10metallicity": {"dist": "normal", "loc": -0.85, "scale": 0.4},
            "logsfr_ratio_0": {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
        },
    )
    y, x = np.mgrid[-3:4, -3:4]
    k = np.exp(-(x**2 + y**2) / 2.0)
    psf = PSFModel(kernels=np.stack([k / k.sum()] * 4).astype(np.float32), band_names=BANDS)
    obs = ObservationCube(
        flux=np.zeros((4, H, W), np.float32),
        variance=np.ones((4, H, W), np.float32),
        mask=np.ones((4, H, W), np.float32),
        band_names=BANDS,
        pixel_scale=0.03,
    )
    emu = _ToyEmulator(_param_names=NAMES, _band_names=BANDS)
    return ForwardModel.build(
        obs=obs, psf_model=psf, spatial_model=model, emulator=emu, nuisance=nuisance
    )


NUIS = dict(fit_sky=True, fit_shifts=True, shift_reference_band=0)


@pytest.fixture(params=[False, True], ids=["float32", "float64"])
def x64(request):
    """Run the test body with x64 off and on."""
    with jax.enable_x64(request.param):
        yield request.param


def test_layout_covers_every_global_kind():
    """The test model really has uniform, Normal and Student-t globals."""
    fm = _build()
    tr = StandardisingTransform(fm)
    kinds = sorted(g.kind for g in tr.globals)
    assert kinds == ["normal", "studentt", "uniform"]
    assert fm.spatial_model.n_hyper == 1 and fm.spatial_model.n_coupling == 1
    assert tr.n_params == fm.n_params


@pytest.mark.parametrize("nuisance", [False, True], ids=["plain", "nuisance"])
def test_round_trip(x64, nuisance):
    """from_theta(to_theta(xi)) == xi, for single vectors and batches."""
    fm = _build(NuisanceModel(4, **NUIS) if nuisance else None)
    tr = StandardisingTransform(fm)
    xi = 1.5 * jax.random.normal(jax.random.PRNGKey(0), (16, tr.n_params))
    theta = tr.to_theta(xi)
    assert theta.dtype == (jnp.float64 if x64 else jnp.float32)
    back = tr.from_theta(theta)
    # Float32 loses a few ulps through the CDF/quantile chain of the globals.
    atol = 1e-6 if x64 else 2e-4
    np.testing.assert_allclose(np.asarray(back), np.asarray(xi), atol=atol, rtol=0)
    np.testing.assert_allclose(
        np.asarray(tr.from_theta(tr.to_theta(xi[3]))), np.asarray(xi[3]), atol=atol, rtol=0
    )


@pytest.mark.parametrize("nuisance", [False, True], ids=["plain", "nuisance"])
def test_standardises_the_prior(x64, nuisance):
    """log_prior(T(xi)) + log|J| - log N(xi) is the same constant for every xi."""
    fm = _build(NuisanceModel(4, **NUIS) if nuisance else None)
    tr = StandardisingTransform(fm)
    xi = jax.random.normal(jax.random.PRNGKey(1), (32, tr.n_params))
    # Include tail values of the globals.  In float32 a truncated global whose physical
    # value lies within ~1e-6 of the bounds width of a bound loses precision (|xi| >~ 5
    # for the metallicity prior here), so the float32 tail points stay at |xi| <~ 4.
    xi = xi.at[:4].multiply(2.5 if x64 else 1.6)
    diff = jax.vmap(
        lambda x: fm.log_prior(tr.to_theta(x)) + tr.log_det_jacobian(x) - standard_normal_logpdf(x)
    )(xi)
    diff = np.asarray(diff, np.float64)
    assert np.all(np.isfinite(diff))
    # float32: the prior sums ~300 terms of O(1), so its rounding is ~1e-4 nats.
    tol = 1e-5 if x64 else 2e-3
    assert np.ptp(diff) < tol
    assert tr.prior_mismatch(fm, jax.random.PRNGKey(2)) < tol


def test_constant_is_the_truncated_mass():
    """The constant is the log prior mass inside the bounds (untruncated densities)."""
    with jax.enable_x64(True):
        fm = _build()
        tr = StandardisingTransform(fm)
        xi = jnp.zeros(tr.n_params)
        c = float(fm.log_prior(tr.to_theta(xi)) + tr.log_det_jacobian(xi))
        c -= float(standard_normal_logpdf(xi))
        expect = sum(g.log_mass for g in tr.globals if g.kind != "uniform")
        assert c == pytest.approx(expect, abs=1e-9)


def test_pushforward_matches_prior_samples():
    """Globals of T(xi) follow the same marginals as sample_prior."""
    fm = _build()
    sm = fm.spatial_model
    tr = StandardisingTransform(fm)
    key1, key2 = jax.random.split(jax.random.PRNGKey(4))
    a = np.asarray(
        jax.vmap(sm.global_values)(tr.to_theta(jax.random.normal(key1, (4000, tr.n_params))))
    )
    b = np.asarray(jax.vmap(sm.global_values)(sm.sample_prior(key2, 4000)))
    for j in range(sm.n_global):
        qa = np.percentile(a[:, j], [10, 25, 50, 75, 90])
        qb = np.percentile(b[:, j], [10, 25, 50, 75, 90])
        width = BOUNDS[sm.global_names[j]][1] - BOUNDS[sm.global_names[j]][0]
        np.testing.assert_allclose(qa, qb, atol=0.03 * width)


def test_standardised_log_density(x64):
    """StandardisedLogDensity = log_posterior(T(xi)) + log|J|, jit- and grad-safe."""
    fm = _build(NuisanceModel(4, fit_sky=True))
    dens = StandardisedLogDensity(fm)
    assert dens.n_params == fm.n_params and dens.spatial_model is fm.spatial_model
    xi = 0.3 * jax.random.normal(jax.random.PRNGKey(5), (dens.n_params,))
    lp = jax.jit(dens.log_posterior)(xi)
    expect = fm.log_posterior(dens.to_theta(xi)) + dens.transform.log_det_jacobian(xi)
    assert float(lp) == pytest.approx(float(expect), rel=1e-5)
    g = jax.grad(dens.log_prob)(xi)
    assert g.shape == xi.shape and bool(jnp.all(jnp.isfinite(g)))
    np.testing.assert_allclose(
        np.asarray(dens.from_theta(dens.to_theta(xi))), np.asarray(xi), atol=2e-4
    )


def test_standardised_density_runs_whitened_nuts():
    """``mean`` is (d,) zeros in the model dtype; run_whitened_nuts accepts the density."""
    from arachne.inference.laplace import run_whitened_nuts

    fm = _build(H=6, W=6)
    dens = StandardisedLogDensity(fm)
    assert dens.mean.shape == (dens.n_params,) and dens.mean.dtype == jnp.zeros(()).dtype
    assert not np.any(np.asarray(dens.mean))
    xi = 0.1 * jax.random.normal(jax.random.PRNGKey(7), (2, dens.n_params))
    starts = dens.to_theta(xi)
    result, w = run_whitened_nuts(
        fm,
        starts[0],
        jax.random.PRNGKey(8),
        n_warmup=20,
        n_samples=10,
        n_chains=2,
        dense_mass_matrix=False,
        whitened=dens,
        initial_positions=starts,
    )
    assert w is dens
    assert result.samples.shape == (20, fm.n_params)
    assert bool(jnp.all(jnp.isfinite(result.samples)))


def test_rejects_unsupported_models():
    """Non-CorrelatedFieldMap spatial models and non-Gaussian nuisance priors are refused."""

    class _Fake:
        spatial_model = object()

    with pytest.raises(TypeError, match="CorrelatedFieldMap"):
        StandardisingTransform(_Fake())

    class _OddNuisance(NuisanceModel):
        def log_prior(self, theta_n):
            return -jnp.sum(jnp.abs(theta_n))

    fm = _build()
    fm.nuisance = _OddNuisance(4, fit_sky=True)
    with pytest.raises(NotImplementedError, match="log_prior"):
        StandardisingTransform(fm)
