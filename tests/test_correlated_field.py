"""Tests for CorrelatedFieldMap (Gaussian-random-field parameter maps)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.initialisation import find_map
from arachne.spatial.correlated_field import (
    CorrelatedFieldMap,
    mass_map_from_image,
    matern_spectrum,
    wiener_mass_map,
)

NAMES = ["redshift", "log_mass", "log10metallicity", "Av", "fesc"]
BOUNDS = {"log10metallicity": (-2.0, 0.3), "Av": (0.0, 3.0), "fesc": (0.0, 1.0)}
BANDS = ["F115W", "F200W", "F277W", "F444W"]
LAM = jnp.array([1.15, 2.0, 2.77, 4.44])


class ToySEDEmulator(SPSEmulator, eqx.Module):
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


@pytest.fixture
def emulator():
    """Toy SED emulator instance."""
    return ToySEDEmulator(_param_names=NAMES, _band_names=BANDS)


def make_model(H=24, W=24, **kw):
    """Small three-field CorrelatedFieldMap with optional overrides."""
    defaults = dict(
        image_shape=(H, W),
        emulator_param_names=NAMES,
        param_bounds=BOUNDS,
        field_params=["log_mass", "log10metallicity", "Av"],
        fixed_params={"redshift": 2.0},
        corr_lengths={"log_mass": 2.0, "log10metallicity": 6.0, "Av": 3.0},
        amplitudes={"log_mass": 0.3, "log10metallicity": 0.15, "Av": 0.3},
    )
    defaults.update(kw)
    return CorrelatedFieldMap(**defaults)


class TestLayout:
    """Parameter roles, theta layout and decode."""

    def test_roles_and_counts(self):
        """Roles and counts."""
        m = make_model()
        assert m.field_names == ["log_mass", "log10metallicity", "Av"]
        assert m.global_names == ["fesc"]
        assert m.n_params == 3 + 1 + sum(2 * f.n_modes for f in m.fields)
        assert m.n_params < 3 * 24 * 24  # truncated basis is much smaller than per-pixel

    def test_longer_correlation_needs_fewer_modes(self):
        """Longer correlation needs fewer modes."""
        m = make_model(corr_lengths={"log_mass": 1.5, "log10metallicity": 8.0, "Av": 3.0})
        n = {f.name: f.n_modes for f in m.fields}
        assert n["log10metallicity"] < n["Av"] < n["log_mass"]

    def test_bad_configs_raise(self):
        """Bad configs raise."""
        with pytest.raises(ValueError, match="mass_param"):
            make_model(field_params=["Av"])
        with pytest.raises(ValueError, match="both mapped and fixed"):
            make_model(fixed_params={"redshift": 2.0, "Av": 0.1})
        with pytest.raises(ValueError, match="no bounds"):
            make_model(param_bounds={"Av": (0.0, 3.0), "fesc": (0, 1)})

    def test_decode_column_order_and_values(self, emulator):
        """Decode column order and values."""
        m = make_model(mean_maps={"log_mass": 7.0})
        theta = m.initial_theta(global_values={"fesc": 0.3})
        rows = np.asarray(m.decode(theta))
        assert rows.shape == (24 * 24, len(NAMES))
        np.testing.assert_allclose(rows[:, 0], 2.0)  # fixed redshift
        np.testing.assert_allclose(rows[:, 1], 7.0, atol=1e-5)  # mass mean map
        np.testing.assert_allclose(rows[:, 4], 0.3, atol=1e-4)  # global fesc
        np.testing.assert_allclose(rows[:, 3], 1.5, atol=1e-3)  # Av mid-range default mean

    def test_split_join_roundtrip(self):
        """Split join roundtrip."""
        m = make_model()
        theta = m.sample_prior(jax.random.PRNGKey(0), 1)[0]
        np.testing.assert_allclose(m.join_theta(*m.split_theta(theta)), theta)


class TestPrior:
    """Field statistics and the normalised prior."""

    def test_unit_field_variance_matches_kept_fraction(self):
        """Unit field variance matches kept fraction."""
        m = make_model(H=32, W=32)
        thetas = m.sample_prior(jax.random.PRNGKey(1), 200)
        for k, f in enumerate(m.fields):
            g = jax.vmap(lambda t, k=k, f=f: m.unit_field(m.split_theta(t)[2][f.slice], k))(thetas)
            var = float(jnp.mean(g**2))
            assert var == pytest.approx(f.variance_kept, rel=0.15), f.name

    def test_correlation_length_orders_smoothness(self):
        """Correlation length orders smoothness."""
        m = make_model(H=32, W=32, corr_lengths=2.0, variance_fraction=0.999)
        m_long = make_model(H=32, W=32, corr_lengths=8.0, variance_fraction=0.999)

        def lag1_corr(model):
            """Lag1 corr."""
            th = model.sample_prior(jax.random.PRNGKey(2), 100)
            g = jax.vmap(
                lambda t: model.unit_field(model.split_theta(t)[2][model.fields[0].slice], 0)
            )(th)
            g = np.asarray(g)
            return np.mean(g[:, :, 1:] * g[:, :, :-1]) / np.mean(g**2)

        assert lag1_corr(m_long) > lag1_corr(m) > 0.3

    def test_all_modes_kept_gives_unit_variance(self):
        """All modes kept gives unit variance."""
        m = make_model(H=16, W=16, variance_fraction=1.0, corr_lengths=1.0, pad=4)
        th = m.sample_prior(jax.random.PRNGKey(3), 400)
        g = jax.vmap(lambda t: m.unit_field(m.split_theta(t)[2][m.fields[0].slice], 0))(th)
        assert m.fields[0].variance_kept > 0.97
        assert float(jnp.mean(g**2)) == pytest.approx(1.0, rel=0.1)

    def test_log_prior_is_normalised_gaussian(self):
        """Log prior is normalised gaussian."""
        m = make_model()
        theta = m.sample_prior(jax.random.PRNGKey(4), 1)[0]
        levels, raw, z = m.split_theta(theta)
        sig = np.array([f.level_sigma for f in m.fields])
        expect = (
            -0.5 * np.sum(np.asarray(z) ** 2)
            - 0.5 * z.size * np.log(2 * np.pi)
            + np.sum(-0.5 * (np.asarray(levels) / sig) ** 2 - np.log(sig) - 0.5 * np.log(2 * np.pi))
            + np.sum(np.log(jax.nn.sigmoid(raw)) + np.log(jax.nn.sigmoid(-raw)))
        )
        assert float(m.log_prior(theta)) == pytest.approx(float(expect), rel=1e-5)

    def test_sample_prior_moments(self):
        """Sample prior moments."""
        m = make_model()
        th = np.asarray(m.sample_prior(jax.random.PRNGKey(5), 4000))
        levels, _, z = th[:, : m.n_fields], th[:, m.n_fields : m.n_fields + 1], th[:, 4:]
        np.testing.assert_allclose(levels.std(0), [f.level_sigma for f in m.fields], rtol=0.06)
        assert abs(z.std() - 1.0) < 0.02

    def test_bounds_respected_for_extreme_draws(self):
        """Bounds respected for extreme draws."""
        m = make_model()
        theta = 8.0 * m.sample_prior(jax.random.PRNGKey(6), 1)[0]
        maps = np.asarray(m.field_maps(theta))
        for k, f in enumerate(m.fields[1:], start=1):
            assert maps[k].min() >= f.bounds[0] - 1e-4 and maps[k].max() <= f.bounds[1] + 1e-4

    def test_coupling_propagates_mass_deviation(self):
        """Coupling propagates mass deviation."""
        m = make_model(couplings={"Av": 0.5}, mean_maps={"log_mass": 7.0})
        t0 = m.initial_theta()
        t1 = t0.at[0].set(0.2)  # raise the mass level by 0.2 dex
        av0 = np.asarray(m.field_maps(t0))[2]
        av1 = np.asarray(m.field_maps(t1))[2]
        np.testing.assert_allclose(av1 - av0, 0.1, atol=2e-3)  # 0.5 * 0.2, interior of bounds

    def test_float64_end_to_end(self):
        """Under x64 the fields are computed in float64 (no complex64 downcast)."""
        with jax.enable_x64(True):
            m = make_model()
            theta = m.sample_prior(jax.random.PRNGKey(7), 1)[0]
            assert theta.dtype == jnp.float64
            maps = m.field_maps(theta)
            assert maps.dtype == jnp.float64
            g = m.unit_field(m.split_theta(theta)[2][m.fields[0].slice], 0)
            g32 = m.unit_field(m.split_theta(theta)[2][m.fields[0].slice].astype(jnp.float32), 0)
            assert g.dtype == jnp.float64
            assert float(jnp.max(jnp.abs(g - g32))) > 0  # genuinely higher precision

    def test_project_to_field_reproduces_in_basis_target(self):
        """A map drawn from the field basis is reproduced; white noise is mostly rejected."""
        m = make_model(mean_maps={"log_mass": 7.0})
        draw = m.sample_prior(jax.random.PRNGKey(11), 1)[0].at[: m.n_fields].set(0.0)
        target = np.asarray(m.field_maps(draw))[0]
        theta = m.project_to_field(m.initial_theta(), "log_mass", target, ridge=1e-4)
        got = np.asarray(m.field_maps(theta))[0]
        np.testing.assert_allclose(got, target, atol=0.02)
        noise = 7.0 + 0.3 * np.random.default_rng(0).normal(size=target.shape)
        theta_n = m.project_to_field(m.initial_theta(), "log_mass", noise, ridge=1.0)
        resid_rms = np.std(np.asarray(m.field_maps(theta_n))[0] - 7.0)
        assert resid_rms < 0.5 * 0.3  # pixel-scale noise is not in the smooth basis

    def test_matern_spectrum_slope(self):
        """Matern spectrum slope."""
        k = np.array([10.0, 100.0])
        p = matern_spectrum(k, corr_length=5.0, nu=0.5)
        assert np.log(p[1] / p[0]) / np.log(10.0) == pytest.approx(-3.0, abs=0.01)


def _mock(emulator, model, key=0, noise=0.03, H=24, W=24):
    """Disk mean + prior-drawn fields, Gaussian PSF, white noise."""
    theta_true = model.sample_prior(jax.random.PRNGKey(key), 1)[0]
    theta_true = theta_true.at[: model.n_fields].set(0.0)  # truth around the mean maps
    psf = PSFModel(kernels=_gaussian_kernels([1.0, 1.2, 1.4, 1.8]), band_names=BANDS)
    blank = ObservationCube(
        flux=np.zeros((4, H, W), np.float32),
        variance=np.ones((4, H, W), np.float32),
        mask=np.ones((4, H, W), np.float32),
        band_names=BANDS,
        pixel_scale=0.03,
    )
    fm = ForwardModel.build(obs=blank, psf_model=psf, spatial_model=model, emulator=emulator)
    clean = np.asarray(fm._model_image(theta_true))
    sigma = noise * clean.max()
    flux = clean + np.random.default_rng(key).normal(0, sigma, clean.shape)
    obs = ObservationCube(
        flux=flux.astype(np.float32),
        variance=np.full(clean.shape, sigma**2, np.float32),
        mask=np.ones(clean.shape, np.float32),
        band_names=BANDS,
        pixel_scale=0.03,
    )
    return theta_true, obs, psf


def _gaussian_kernels(sigmas):
    """Unit-sum Gaussian PSF kernels (13x13), one per sigma."""
    y, x = np.mgrid[-6:7, -6:7]
    ks = [np.exp(-(x**2 + y**2) / (2 * s**2)) for s in sigmas]
    return np.stack([k / k.sum() for k in ks]).astype(np.float32)


def _disk(H, W, peak=8.0, rd=5.0):
    """Exponential-disk log10 mass-per-pixel map centred on the frame."""
    yy, xx = np.mgrid[0:H, 0:W] - (H - 1) / 2.0
    return (peak - np.hypot(yy, xx) / rd / np.log(10)).astype(np.float32)


class TestForwardModel:
    """Integration with ForwardModel, the mass start and blind MAP recovery."""

    def test_log_posterior_finite_gradient(self, emulator):
        """Log posterior finite gradient."""
        model = make_model(mean_maps={"log_mass": _disk(24, 24)})
        _, obs, psf = _mock(emulator, model)
        fm = ForwardModel.build(obs=obs, psf_model=psf, spatial_model=model, emulator=emulator)
        theta = model.initial_theta()
        g = jax.grad(fm.log_posterior)(theta)
        assert g.shape == (model.n_params,) and bool(jnp.all(jnp.isfinite(g)))
        lp = fm.log_posterior(theta)
        assert float(lp) == pytest.approx(float(fm.log_likelihood(theta) + fm.log_prior(theta)))

    def test_mass_map_from_image_recovers_mass_for_point_psf(self, emulator):
        """Mass map from image recovers mass for point psf."""
        model = make_model(mean_maps={"log_mass": _disk(24, 24)})
        theta = model.initial_theta()
        fm_img = np.asarray(model.model_image(theta, emulator, (24, 24)))
        obs = ObservationCube(
            flux=fm_img,
            variance=np.ones_like(fm_img),
            mask=np.ones_like(fm_img),
            band_names=BANDS,
            pixel_scale=0.03,
        )
        ref = {"redshift": 2.0, "log10metallicity": -0.85, "Av": 1.5, "fesc": 0.5}
        logm = mass_map_from_image(obs, emulator, ref, smooth_sigma_px=0.0)
        np.testing.assert_allclose(logm, _disk(24, 24), atol=1e-3)

    def test_wiener_mass_start_beats_smoothed_image(self, emulator):
        """With the right reference SED, deconvolution recovers a sharp mass map better."""
        H = W = 32
        model = make_model(H=H, W=W, mean_maps={"log_mass": _disk(H, W, rd=4.0)})
        n0 = model.n_fields + model.n_global
        z = 2.0 * model.sample_prior(jax.random.PRNGKey(9), 1)[0][n0:]
        z = z.at[model.fields[0].slice.stop :].set(0.0)  # clumpy mass; Z and Av flat at means
        theta = model.initial_theta().at[n0:].set(z)
        truth = np.asarray(model.field_maps(theta))[0]
        psf = PSFModel(kernels=_gaussian_kernels([1.0, 1.4, 1.8, 2.4]), band_names=BANDS)
        blank = ObservationCube(
            flux=np.zeros((4, H, W), np.float32),
            variance=np.ones((4, H, W), np.float32),
            mask=np.ones((4, H, W), np.float32),
            band_names=BANDS,
            pixel_scale=0.03,
        )
        blur = np.asarray(ForwardModel.build(blank, psf, model, emulator)._model_image(theta))
        sig = 0.01 * blur.max()
        obs = ObservationCube(
            flux=(blur + np.random.default_rng(3).normal(0, sig, blur.shape)).astype(np.float32),
            variance=np.full(blur.shape, sig**2, np.float32),
            mask=np.ones(blur.shape, np.float32),
            band_names=BANDS,
            pixel_scale=0.03,
        )
        maps = np.asarray(model.field_maps(theta))
        ref = {
            "redshift": 2.0,
            "log10metallicity": float(maps[1].mean()),
            "Av": float(maps[2].mean()),
            "fesc": 0.5,
        }
        smooth = mass_map_from_image(obs, emulator, ref, smooth_sigma_px=1.0)
        wiener = wiener_mass_map(obs, psf, emulator, ref)
        inner = slice(4, -4)
        err_s = np.median(np.abs(smooth - truth)[inner, inner])
        err_w = np.median(np.abs(wiener - truth)[inner, inner])
        assert err_w < 0.7 * err_s, (err_w, err_s)

    def test_blind_map_recovers_mass_map(self, emulator):
        """Truth drawn from the prior; fit starts from the image-based mass mean only."""
        H = W = 24
        truth_model = make_model(mean_maps={"log_mass": _disk(H, W)})
        theta_true, obs, psf = _mock(emulator, truth_model)
        ref = {"redshift": 2.0, "log10metallicity": -0.85, "Av": 1.5, "fesc": 0.5}
        mass_mean = mass_map_from_image(obs, emulator, ref, smooth_sigma_px=1.0)
        fit_model = make_model(mean_maps={"log_mass": mass_mean})
        fm = ForwardModel.build(obs=obs, psf_model=psf, spatial_model=fit_model, emulator=emulator)
        res = find_map(
            fm, fit_model.initial_theta(), n_rounds=3, steps_per_round=300, lr=0.05, final_steps=300
        )
        true_maps = np.asarray(truth_model.field_maps(theta_true))
        fit_maps = np.asarray(fit_model.field_maps(res.theta))
        assert res.neg_log_posterior < float(-fm.log_posterior(fit_model.initial_theta()))
        r_mass = np.corrcoef(true_maps[0].ravel(), fit_maps[0].ravel())[0, 1]
        assert r_mass > 0.9
        assert np.median(np.abs(fit_maps[0] - true_maps[0])) < 0.15
