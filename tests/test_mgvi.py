"""Tests for arachne.inference.mgvi (geoVI / MGVI through NIFTy.re)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("nifty8.re")

from arachne.data.observation import ObservationCube  # noqa: E402
from arachne.data.psf import PSFModel  # noqa: E402
from arachne.emulator.base import SPSEmulator  # noqa: E402
from arachne.forward_model.nuisance import NuisanceModel  # noqa: E402
from arachne.forward_model.pipeline import ForwardModel  # noqa: E402
from arachne.inference.laplace import laplace_covariance, make_hessian_fn  # noqa: E402
from arachne.inference.mgvi import MGVIResult, mgvi_scale, run_mgvi  # noqa: E402
from arachne.inference.newton import trust_region_newton  # noqa: E402
from arachne.inference.standardise import StandardisedLogDensity  # noqa: E402
from arachne.spatial.correlated_field import CorrelatedFieldMap  # noqa: E402

NAMES = ["redshift", "log_mass", "log10metallicity", "Av", "fesc"]
BOUNDS = {"log10metallicity": (-2.0, 0.3), "Av": (0.0, 3.0), "fesc": (0.0, 1.0)}
BANDS = ["F115W", "F200W", "F277W", "F444W"]
LAM = jnp.array([1.15, 2.0, 2.77, 4.44])
H = W = 10


class _ToyEmulator(SPSEmulator, eqx.Module):
    """Mass-linear toy: metallicity tilts the colour, Av reddens, fesc ignored."""

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
        """(N, 5) -> (N, 4) nJy."""
        z, logm, logz, av = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
        beta = 0.5 + 0.8 * (logz + 0.5)
        tau = av * (LAM[None, :] / (0.55 * (1.0 + z))) ** -1.0
        return 10.0 ** (logm - 6.0) * LAM[None, :] ** beta * 10.0 ** (-0.4 * tau)


def _model() -> CorrelatedFieldMap:
    """Mass + one bounded field (Av) and one uniform global (fesc).

    The toy emulator ignores ``fesc``, so its posterior is its prior: a
    uniform global whose standardised latent is exactly N(0, 1).  Metallicity
    is fixed: freeing it makes this small problem multimodal (metallicity-Av
    trading), and the tests below are about the near-Gaussian case.
    """
    return CorrelatedFieldMap(
        image_shape=(H, W),
        emulator_param_names=NAMES,
        param_bounds=BOUNDS,
        field_params=["log_mass", "Av"],
        fixed_params={"redshift": 2.0, "log10metallicity": -0.85},
        corr_lengths=3.0,
        amplitudes={"log_mass": 0.5, "Av": 0.3},
        variance_fraction=0.9,
        mean_maps={"log_mass": 8.0, "Av": 1.0},
    )


def _build(flux=None, variance=None, **kw) -> ForwardModel:
    y, x = np.mgrid[-3:4, -3:4]
    k = np.exp(-(x**2 + y**2) / 2.0)
    psf = PSFModel(kernels=np.stack([k / k.sum()] * 4).astype(np.float32), band_names=BANDS)
    shape = (4, H, W)
    obs = ObservationCube(
        flux=np.zeros(shape, np.float32) if flux is None else flux,
        variance=np.ones(shape, np.float32) if variance is None else variance,
        mask=np.ones(shape, np.float32),
        band_names=BANDS,
        pixel_scale=0.03,
    )
    emu = _ToyEmulator(_param_names=NAMES, _band_names=BANDS)
    return ForwardModel.build(obs=obs, psf_model=psf, spatial_model=_model(), emulator=emu, **kw)


def _mock_data():
    """Data from a prior draw with 5%-of-peak white noise."""
    fm0 = _build()
    theta_true = fm0.spatial_model.sample_prior(jax.random.PRNGKey(0), 1)[0]
    clean = np.asarray(fm0._model_image(theta_true))
    sigma = 0.05 * float(np.median(clean))
    flux = clean + np.random.default_rng(0).normal(0.0, sigma, clean.shape)
    return flux.astype(np.float32), np.full(clean.shape, sigma**2, np.float32)


@pytest.fixture(scope="module")
def problem():
    """Forward model, a geoVI run (float32) and the Newton mode + Laplace sd (float64)."""
    flux, var = _mock_data()
    fm = _build(flux, var)
    theta0 = fm.spatial_model.initial_theta()
    res = run_mgvi(
        fm,
        theta0,
        jax.random.PRNGKey(1),
        n_iterations=4,
        n_samples=2,
        mode="geovi",
        n_final_samples=6,
        verbose=False,
    )
    with jax.enable_x64(True):
        # Newton from the prior mean and from the VI mean; keep the better mode.
        best = None
        for start in (theta0, res.theta_mean):
            nr = trust_region_newton(
                lambda t: -fm.log_posterior(t),
                jnp.asarray(np.asarray(start), jnp.float64),
                verbose=False,
            )
            nlp = float(-fm.log_posterior(nr.theta))
            if best is None or nlp < best[0]:
                best = (nlp, nr)
        nlp_mode, newton = best
        mode = np.asarray(newton.theta)
        nlp_mean = float(-fm.log_posterior(jnp.asarray(res.theta_mean, jnp.float64)))
        hess = make_hessian_fn(fm)(mode)
    cov, _ = laplace_covariance(hess)
    return {
        "fm": fm,
        "res": res,
        "converged": bool(newton.converged),
        "nlp_mode": nlp_mode,
        "nlp_mean": nlp_mean,
        "laplace_sd": np.sqrt(np.diag(cov)),
    }


def test_runs_and_shapes(problem):
    """Finite results with the documented shapes and a per-iteration history."""
    res, fm = problem["res"], problem["fm"]
    d = fm.n_params
    assert isinstance(res, MGVIResult)
    assert res.theta_mean.shape == (d,) and res.xi_mean.shape == (d,)
    assert res.theta_samples.shape == (4, d) and res.xi_samples.shape == (4, d)
    assert res.theta_extra_samples.shape == (6, d) and res.xi_extra_samples.shape == (6, d)
    for a in (res.theta_mean, res.theta_samples, res.xi_samples, res.theta_extra_samples):
        assert np.all(np.isfinite(a))
    assert res.theta_mean.dtype == np.float32
    # The extra (linear) samples are mirrored pairs about the latent mean.
    np.testing.assert_allclose(
        res.xi_extra_samples[0] + res.xi_extra_samples[1], 2 * res.xi_mean, atol=1e-5
    )
    assert len(res.history) == res.n_iterations == 4
    assert [h["iteration"] for h in res.history] == [1, 2, 3, 4]
    assert res.history[-1]["neg_log_post_mean"] < res.history[0]["neg_log_post_mean"]
    assert res.runtime_s > 0
    scale = mgvi_scale(res)
    assert scale.shape == (d,) and np.all(scale > 0) and np.all(np.isfinite(scale))


def test_mean_close_to_newton_mode(problem):
    """For this near-Gaussian problem the VI mean sits within a few nats of the mode."""
    assert problem["converged"]
    gap = problem["nlp_mean"] - problem["nlp_mode"]
    assert -0.5 < gap < 5.0, gap


def test_widths_track_laplace(problem):
    """Sample sd follows the Laplace sd on the well-constrained coordinates."""
    res = problem["res"]
    lap = problem["laplace_sd"]
    samples = np.concatenate([res.theta_samples, res.theta_extra_samples], axis=0)
    sd = np.sqrt(np.mean((samples - samples.mean(0)) ** 2, axis=0))
    prior_scale = res.transform._scale
    well = lap < 0.5 * prior_scale
    assert well.sum() > 20
    ratio = sd[well] / lap[well]
    assert 0.5 < np.median(ratio) < 2.0, np.median(ratio)
    r = np.corrcoef(np.log(sd[well]), np.log(lap[well]))[0, 1]
    assert r > 0.5, r


def test_mgvi_mode_under_x64():
    """MGVI (linear samples) runs in float64 and returns float64 results."""
    flux, var = _mock_data()
    with jax.enable_x64(True):
        fm = _build(flux, var, nuisance=NuisanceModel(4, fit_sky=True))
        theta0 = fm.initial_theta_from_spatial(fm.spatial_model.initial_theta())
        res = run_mgvi(
            fm,
            theta0,
            jax.random.PRNGKey(2),
            n_iterations=3,
            n_samples=1,
            mode="mgvi",
            verbose=False,
        )
        assert res.theta_mean.dtype == np.float64 and res.theta_samples.shape == (2, fm.n_params)
        assert np.all(np.isfinite(res.theta_samples))
        # Linear (MGVI) samples are mirrored pairs about the latent mean.
        np.testing.assert_allclose(res.xi_samples.mean(0), res.xi_mean, atol=1e-10)
        nlp0 = float(-fm.log_posterior(jnp.asarray(theta0, jnp.float64)))
        assert res.history[-1]["neg_log_post_mean"] < nlp0
        # The standardised density accepts the VI latents.
        dens = StandardisedLogDensity(fm, res.transform)
        assert np.isfinite(float(dens.log_posterior(res.xi_mean)))


def test_refuses_model_dependent_noise():
    """A model-error floor or a fitted noise scale make the Gaussian model dependent."""
    flux, var = _mock_data()
    fm = _build(flux, var, model_error_frac=0.05)
    with pytest.raises(NotImplementedError, match="model_error_frac"):
        run_mgvi(fm, fm.spatial_model.initial_theta(), jax.random.PRNGKey(0), verbose=False)
    fm = _build(flux, var, nuisance=NuisanceModel(4, fit_sky=False, fit_noise_scale=True))
    theta0 = fm.initial_theta_from_spatial(fm.spatial_model.initial_theta())
    with pytest.raises(NotImplementedError, match="noise"):
        run_mgvi(fm, theta0, jax.random.PRNGKey(0), verbose=False)
    with pytest.raises(ValueError, match="mode"):
        run_mgvi(_build(flux, var), theta0[:-4], jax.random.PRNGKey(0), mode="laplace")
