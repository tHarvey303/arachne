"""Tests for arachne.inference.initialisation (blind init, mass solve, MAP)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.data.observation import ObservationCube
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.initialisation import (
    MAPResult,
    blind_initial_full_theta,
    blind_initial_theta,
    find_map,
    image_moments,
    multistart_map,
    solve_component_masses,
)
from arachne.spatial.additive import AdditiveComponentModel

# Mirror tests/conftest.py constants (tests/ is not a package).
N_BANDS = 3
H = 16
W = 16
BAND_NAMES = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]
SPS_PARAM_NAMES = ["log_stellar_mass", "log_age", "tau_v"]
PARAM_BOUNDS = {
    "log_stellar_mass": (6.0, 12.0),
    "log_age": (7.0, 10.1),
    "tau_v": (0.0, 4.0),
}
MASS = "log_stellar_mass"
FULL_NAMES = ["log_stellar_mass", "log_age", "tau_v", "redshift", "log_zmet"]
FULL_BOUNDS = {**PARAM_BOUNDS, "redshift": (0.0, 10.0), "log_zmet": (-2.0, 0.5)}


class ColourLinearEmulator(SPSEmulator, eqx.Module):
    """flux_b = 10**(logM - 9) * (b + 1) * exp(-0.4 * tau_v * (2 - b)).

    Exactly linear in ``10**logM``; ``tau_v`` reddens (suppresses the bluer
    bands) so it is identifiable from colour; ``log_age`` has no effect.
    """

    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)

    @property
    def param_names(self) -> list[str]:
        """SPS parameter names."""
        return self._param_names

    @property
    def band_names(self) -> list[str]:
        """Band names."""
        return self._band_names

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Mass-linear flux with a band-dependent attenuation term."""
        m9 = 10.0 ** (params[:, 0:1] - 9.0)
        b = jnp.arange(len(self._band_names), dtype=jnp.float32)[None, :]
        atten = jnp.exp(-0.4 * params[:, 2:3] * (2.0 - b))
        return m9 * (b + 1.0) * atten


def _raw(value, lo, hi):
    u = (value - lo) / (hi - lo)
    return float(np.log(u) - np.log1p(-u))


def _sps_raw(log_m, log_age, tau_v):
    return [
        _raw(log_m, *PARAM_BOUNDS[MASS]),
        _raw(log_age, *PARAM_BOUNDS["log_age"]),
        _raw(tau_v, *PARAM_BOUNDS["tau_v"]),
    ]


def _block(mu_y, mu_x, sy, sx, rho, sps_raw):
    return [mu_y, mu_x, np.log(sy), np.log(sx), np.arctanh(rho), *sps_raw]


TRUE_LOG_M = (9.4, 9.9)  # compact, extended


@pytest.fixture
def emulator() -> ColourLinearEmulator:
    """Mass-linear, colour-carrying dummy emulator."""
    return ColourLinearEmulator(_param_names=SPS_PARAM_NAMES, _band_names=BAND_NAMES)


@pytest.fixture
def model_k2() -> AdditiveComponentModel:
    """K=2 additive model on the 16x16 grid."""
    return AdditiveComponentModel(
        n_components=2,
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(H, W),
        mass_param=MASS,
    )


@pytest.fixture
def theta_true(model_k2) -> jnp.ndarray:
    """Compact + extended truth, compact first."""
    b0 = _block(7.0, 7.5, 1.2, 1.2, 0.0, _sps_raw(TRUE_LOG_M[0], 8.5, 0.5))
    b1 = _block(8.0, 8.0, 4.0, 3.0, 0.2, _sps_raw(TRUE_LOG_M[1], 9.5, 2.5))
    return jnp.array(b0 + b1, dtype=jnp.float32)


def _mock_forward_model(model, emulator, psf, theta_true, seed=1, pixel_scale=0.031):
    """ForwardModel on a high-S/N (peak S/N ~ 100) mock generated from ``theta_true``."""
    from arachne.psf.convolution import PSFConvolver

    conv = PSFConvolver(psf, image_shape=(H, W))
    truth = np.asarray(conv(model.model_image(theta_true, emulator, (H, W))))
    sigma = truth.max(axis=(1, 2)) / 100.0
    rng = np.random.default_rng(seed)
    flux = (truth + rng.normal(size=truth.shape) * sigma[:, None, None]).astype(np.float32)
    variance = np.broadcast_to(sigma[:, None, None] ** 2, truth.shape).astype(np.float32)
    obs = ObservationCube(
        flux=flux,
        variance=variance,
        mask=np.ones_like(flux),
        band_names=BAND_NAMES,
        pixel_scale=pixel_scale,
        wcs=None,
    )
    return ForwardModel.build(obs, psf, model, emulator)


@pytest.fixture
def mock_fm(model_k2, emulator, gaussian_psf, theta_true):
    """ForwardModel on a high-S/N mock generated from ``theta_true``."""
    return _mock_forward_model(model_k2, emulator, gaussian_psf, theta_true)


def _log_masses(model, theta):
    _, _, _, sps = model.component_params(theta)
    return np.asarray(sps[:, 0])


# ---------------------------------------------------------------------------
# image_moments
# ---------------------------------------------------------------------------


class TestImageMoments:
    """Centroid / size from the S/N-stacked image."""

    def test_recovers_gaussian_blob(self):
        """Centroid to < 0.5 px and sigma to ~20% for a synthetic Gaussian blob."""
        cy, cx, sig = 6.3, 9.1, 2.0
        yy, xx = np.mgrid[0:H, 0:W]
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig**2))
        flux = np.stack([blob * (b + 1) for b in range(N_BANDS)]).astype(np.float32)
        variance = np.full_like(flux, 1e-4)
        ey, ex, esig = image_moments(flux, variance, np.ones_like(flux))
        assert abs(ey - cy) < 0.5 and abs(ex - cx) < 0.5
        assert abs(esig - sig) / sig < 0.2

    def test_mask_and_bad_variance_ignored(self):
        """Masked / zero-variance pixels do not move the centroid."""
        yy, xx = np.mgrid[0:H, 0:W]
        blob = np.exp(-((yy - 8) ** 2 + (xx - 8) ** 2) / 8.0)
        flux = np.stack([blob] * N_BANDS).astype(np.float32)
        flux[:, 0, 0] = 1e6  # hot pixel
        variance = np.ones_like(flux)
        mask = np.ones_like(flux)
        mask[:, 0, 0] = 0
        cy, cx, _ = image_moments(flux, variance, mask)
        assert abs(cy - 8) < 0.2 and abs(cx - 8) < 0.2
        variance[:, 0, 0] = 0.0  # zero variance is ignored even without the mask
        cy, cx, _ = image_moments(flux, variance)
        assert abs(cy - 8) < 0.2 and abs(cx - 8) < 0.2

    def test_empty_image_falls_back(self):
        """A flat image returns the image centre and a default size."""
        flux = np.zeros((N_BANDS, H, W), dtype=np.float32)
        cy, cx, sig = image_moments(flux, np.ones_like(flux))
        assert (cy, cx) == ((H - 1) / 2, (W - 1) / 2)
        assert sig > 0

    def test_shape_error(self):
        """Mismatched shapes raise."""
        with pytest.raises(ValueError):
            image_moments(np.zeros((3, 4, 4)), np.zeros((3, 4, 5)))


# ---------------------------------------------------------------------------
# blind_initial_theta
# ---------------------------------------------------------------------------


class TestBlindInitialTheta:
    """Truth-free starting point."""

    def test_shape_finite_bounds_sizes(self, mock_fm, model_k2):
        """n_params, finite, decoded SPS inside bounds, sizes ascending."""
        theta = blind_initial_theta(model_k2, mock_fm.observation)
        assert theta.shape == (model_k2.n_params,)
        assert theta.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(theta))
        mu, sigma, rho, sps = model_k2.component_params(theta)
        for j, name in enumerate(SPS_PARAM_NAMES):
            lo, hi = PARAM_BOUNDS[name]
            assert np.all(np.asarray(sps[:, j]) > lo) and np.all(np.asarray(sps[:, j]) < hi)
        size = np.asarray(sigma).sum(axis=1)
        assert np.all(np.diff(size) > 0)
        np.testing.assert_allclose(rho, 0.0)
        # both components sit at the moment centroid, near the true light centre
        assert np.allclose(mu[0], mu[1])
        assert abs(float(mu[0, 0]) - 7.8) < 1.0 and abs(float(mu[0, 1]) - 7.9) < 1.0
        # neutral defaults: mass -> 9.5, others midpoint
        np.testing.assert_allclose(sps[:, 0], 9.5, atol=1e-4)
        np.testing.assert_allclose(sps[:, 2], 2.0, atol=1e-4)

    def test_overrides_and_shared(self, mock_fm):
        """neutral_values / shared_values are honoured and clipped inside bounds."""
        model = AdditiveComponentModel(
            n_components=3,
            emulator_param_names=FULL_NAMES,
            param_bounds=FULL_BOUNDS,
            image_shape=(H, W),
            shared_param_names=["redshift"],
            fixed_params={"log_zmet": -0.3},
            mass_param=MASS,
        )
        theta = blind_initial_theta(
            model,
            mock_fm.observation,
            size_scales=[0.5, 1.0, 2.0],
            neutral_values={"tau_v": 99.0},  # clipped to 1% inside the upper bound
            shared_values={"redshift": 2.0},
        )
        assert theta.shape == (model.n_params,)
        assert jnp.all(jnp.isfinite(theta))
        _, sigma, _, sps = model.component_params(theta)
        np.testing.assert_allclose(sps[:, FULL_NAMES.index("redshift")], 2.0, atol=1e-4)
        np.testing.assert_allclose(sps[:, FULL_NAMES.index("tau_v")], 4.0 - 0.04, atol=1e-4)
        np.testing.assert_allclose(sps[:, FULL_NAMES.index("log_zmet")], -0.3, atol=1e-6)
        np.testing.assert_allclose(sigma[:, 0] / sigma[0, 0], [1.0, 2.0, 4.0], rtol=1e-4)

    def test_errors(self, mock_fm, gmm_model, model_k2):
        """Non-additive model -> TypeError; bad names / lengths -> ValueError."""
        with pytest.raises(TypeError):
            blind_initial_theta(gmm_model, mock_fm.observation)
        with pytest.raises(ValueError):
            blind_initial_theta(model_k2, mock_fm.observation, size_scales=[1.0])
        with pytest.raises(ValueError):
            blind_initial_theta(model_k2, mock_fm.observation, neutral_values={"nope": 1.0})


# ---------------------------------------------------------------------------
# solve_component_masses
# ---------------------------------------------------------------------------


class TestSolveComponentMasses:
    """Linear least-squares mass solve."""

    def test_recovers_masses_from_wrong_start(self, mock_fm, model_k2, theta_true):
        """True shapes and colours, wrong masses -> masses within 0.05 dex."""
        blocks, shared = model_k2.split_theta(theta_true)
        lo, hi = PARAM_BOUNDS[MASS]
        wrong = blocks.at[:, 5].set(jnp.array([_raw(8.0, lo, hi), _raw(10.8, lo, hi)]))
        theta_wrong = model_k2.join_theta(wrong, shared)

        theta_solved = solve_component_masses(mock_fm, theta_wrong)
        assert theta_solved.shape == theta_true.shape
        assert theta_solved.dtype == jnp.float32
        np.testing.assert_allclose(_log_masses(model_k2, theta_solved), TRUE_LOG_M, atol=0.05)
        # everything but the mass raws is untouched
        keep = np.ones(model_k2.n_params, dtype=bool)
        keep[[5, 5 + model_k2.n_params_per_component]] = False
        np.testing.assert_array_equal(np.asarray(theta_solved)[keep], np.asarray(theta_wrong)[keep])
        # the posterior improved
        assert float(mock_fm.log_posterior(theta_solved)) > float(
            mock_fm.log_posterior(theta_wrong)
        )

    def test_jit_and_idempotent(self, mock_fm, theta_true):
        """jit-safe and a fixed point when starting from the solution."""
        f = jax.jit(lambda t: solve_component_masses(mock_fm, t))
        once = f(theta_true)
        twice = f(once)
        np.testing.assert_allclose(np.asarray(twice), np.asarray(once), atol=1e-4)

    def test_type_error(self, tiny_observation, gaussian_psf, mock_emulator, gmm_model):
        """Non-additive spatial model raises TypeError."""
        fm = ForwardModel.build(tiny_observation, gaussian_psf, gmm_model, mock_emulator)
        with pytest.raises(TypeError):
            solve_component_masses(fm, jnp.zeros(gmm_model.n_params))


# ---------------------------------------------------------------------------
# find_map / multistart_map
# ---------------------------------------------------------------------------


FAST = dict(n_rounds=2, steps_per_round=150, final_steps=150)


class TestFindMAP:
    """Adam MAP finder."""

    def test_blind_map_reaches_truth_basin(self, mock_fm, model_k2, theta_true):
        """From blind_initial_theta: -log_post <= truth + tol and masses within 0.1 dex."""
        theta0 = blind_initial_theta(model_k2, mock_fm.observation)
        result = find_map(mock_fm, theta0, **FAST)
        assert isinstance(result, MAPResult)
        assert result.theta.shape == theta0.shape
        assert result.theta.dtype == jnp.float32
        assert result.n_steps == 2 * 150 + 150
        assert result.history.shape == (result.n_steps,)
        assert np.all(np.isfinite(result.history))
        nlp_truth = -float(mock_fm.log_posterior(theta_true))
        assert result.neg_log_posterior <= nlp_truth + 5.0
        np.testing.assert_allclose(_log_masses(model_k2, result.theta), TRUE_LOG_M, atol=0.1)
        # components are returned compact-first
        _, sigma, _, _ = model_k2.component_params(result.theta)
        assert float(sigma[0].sum()) < float(sigma[1].sum())

    def test_options(self, mock_fm, theta_true):
        """resolve_masses/order_by_size off and zero polish still run."""
        result = find_map(
            mock_fm,
            theta_true,
            n_rounds=1,
            steps_per_round=20,
            final_steps=0,
            resolve_masses=False,
            order_by_size=False,
        )
        assert result.n_steps == 20
        assert np.isfinite(result.neg_log_posterior)

    def test_non_additive_model(self, tiny_observation, gaussian_psf, mock_emulator, gmm_model):
        """Works (mass solve skipped) for a non-additive spatial model."""
        fm = ForwardModel.build(tiny_observation, gaussian_psf, gmm_model, mock_emulator)
        result = find_map(
            fm, jnp.zeros(gmm_model.n_params), n_rounds=1, steps_per_round=20, final_steps=10
        )
        assert result.theta.shape == (gmm_model.n_params,)
        assert result.n_steps == 30
        assert result.history[-1] <= result.history[0]


class TestMultistartMAP:
    """Best-of-several MAP."""

    def test_returns_best(self, mock_fm, model_k2):
        """Result is no worse than the plain find_map from theta0 and has valid fields."""
        theta0 = blind_initial_theta(model_k2, mock_fm.observation)
        kwargs = dict(n_rounds=1, steps_per_round=60, final_steps=40)
        single = find_map(mock_fm, theta0, **kwargs)
        best = multistart_map(mock_fm, theta0, [{"tau_v": 0.2}, {"tau_v": 3.5}, {}], **kwargs)
        assert isinstance(best, MAPResult)
        assert best.neg_log_posterior <= single.neg_log_posterior + 1e-3
        assert np.isclose(
            best.neg_log_posterior, -float(mock_fm.log_posterior(best.theta)), rtol=1e-5
        )

    def test_errors(
        self, mock_fm, model_k2, tiny_observation, gaussian_psf, mock_emulator, gmm_model
    ):
        """Unknown archetype parameter -> ValueError; non-additive -> TypeError."""
        theta0 = blind_initial_theta(model_k2, mock_fm.observation)
        with pytest.raises(ValueError):
            multistart_map(mock_fm, theta0, [{"nope": 1.0}], n_rounds=0, final_steps=0)
        fm = ForwardModel.build(tiny_observation, gaussian_psf, gmm_model, mock_emulator)
        with pytest.raises(TypeError):
            multistart_map(fm, jnp.zeros(gmm_model.n_params), [])


# ---------------------------------------------------------------------------
# Mixed profiles (Sersic / point source) and arcsec coordinates
# ---------------------------------------------------------------------------


MIXED_LOG_M = (9.4, 9.9)  # Sersic bulge, Gaussian disk


@pytest.fixture
def bulge_disk_model() -> AdditiveComponentModel:
    """Sersic bulge + Gaussian disk, analytic normalisation, 3x oversampled."""
    return AdditiveComponentModel(
        n_components=2,
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(H, W),
        mass_param=MASS,
        profiles=["sersic", "gaussian"],
        normalisation="analytic",
        oversample=3,
    )


@pytest.fixture
def bulge_disk_truth(bulge_disk_model) -> jnp.ndarray:
    """Compact n=4 bulge + extended Gaussian disk."""
    bulge = [
        7.5,
        7.5,
        np.log(1.5),
        np.log(1.5),
        0.0,
        np.log(4.0),
        *_sps_raw(MIXED_LOG_M[0], 8.5, 0.5),
    ]
    disk = _block(8.0, 8.0, 4.0, 3.0, 0.2, _sps_raw(MIXED_LOG_M[1], 9.5, 2.5))
    return jnp.array(bulge + disk, dtype=jnp.float32)


@pytest.fixture
def bulge_disk_fm(bulge_disk_model, emulator, gaussian_psf, bulge_disk_truth):
    """ForwardModel on a mock rendered from ``bulge_disk_truth``."""
    return _mock_forward_model(bulge_disk_model, emulator, gaussian_psf, bulge_disk_truth)


class TestMixedProfileInitialisation:
    """blind init / mass solve / MAP across variable-length component blocks."""

    def test_blind_theta_fills_sersic_and_point_blocks(self, mock_fm):
        """Each profile gets the shape entries it owns; n=4 for the compact Sersic."""
        model = AdditiveComponentModel(
            n_components=3,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(H, W),
            mass_param=MASS,
            profiles=["sersic", "sersic", "point"],
        )
        theta = blind_initial_theta(model, mock_fm.observation, size_scales=[0.4, 1.5, 1.0])
        assert theta.shape == (model.n_params,)
        assert jnp.all(jnp.isfinite(theta))
        shapes = model.component_shapes(theta)
        # compact Sersic -> bulge n=4, extended Sersic -> disk n=1
        np.testing.assert_allclose(float(shapes[0]["n"]), 4.0, rtol=1e-3)
        np.testing.assert_allclose(float(shapes[1]["n"]), 1.0, rtol=1e-3)
        # the point source carries only a centre, at the same moment centroid
        assert model.n_shape_params == (6, 6, 2)
        np.testing.assert_allclose(shapes[2]["mu"], shapes[0]["mu"], atol=1e-6)
        np.testing.assert_allclose(shapes[2]["sigma"], [0.5, 0.5])
        # an explicit override wins, and a non-Sersic index is rejected
        theta = blind_initial_theta(model, mock_fm.observation, sersic_n={1: 2.5})
        np.testing.assert_allclose(float(model.component_shapes(theta)[1]["n"]), 2.5, rtol=1e-3)
        with pytest.raises(ValueError):
            blind_initial_theta(model, mock_fm.observation, sersic_n={2: 3.0})

    def test_blind_theta_in_arcsec_mode(self, mock_fm):
        """Moment centroid and size are converted from pixels to arcsec."""
        ps = 0.031
        model = AdditiveComponentModel(
            n_components=2,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(H, W),
            mass_param=MASS,
            profiles=["gaussian", "point"],
            pixel_scale=ps,
            normalisation="analytic",
        )
        cy, cx, sigma_px = image_moments(
            mock_fm.observation.flux, mock_fm.observation.variance, mock_fm.observation.mask
        )
        theta = blind_initial_theta(model, mock_fm.observation, size_scales=[0.8, 1.0])
        mu, sigma, _, _ = model.component_params(theta)
        np.testing.assert_allclose(
            np.asarray(mu[0]),
            [(cy - (H - 1) / 2) * ps, (cx - (W - 1) / 2) * ps],
            rtol=1e-4,
        )
        np.testing.assert_allclose(np.asarray(mu[1]), np.asarray(mu[0]), atol=1e-7)
        np.testing.assert_allclose(np.asarray(sigma[0]), 0.8 * sigma_px * ps, rtol=1e-4)
        # a "point" component named in arcsec mode keeps its half-pixel width
        np.testing.assert_allclose(np.asarray(sigma[1]), 0.5 * ps, rtol=1e-6)
        assert jnp.isfinite(model.log_prior(theta))

    def test_mass_solve_and_blind_map_recover_bulge_disk(
        self, bulge_disk_fm, bulge_disk_model, bulge_disk_truth
    ):
        """Sersic bulge + Gaussian disk: blind MAP recovers both masses to 0.1 dex."""
        model = bulge_disk_model
        theta0 = blind_initial_theta(model, bulge_disk_fm.observation)
        assert theta0.shape == (model.n_params,)
        # the linear solve already works across the unequal-length blocks
        solved = solve_component_masses(bulge_disk_fm, theta0)
        assert solved.shape == theta0.shape
        np.testing.assert_allclose(_log_masses(model, solved), MIXED_LOG_M, atol=0.3)

        result = find_map(bulge_disk_fm, theta0, **FAST)
        np.testing.assert_allclose(_log_masses(model, result.theta), MIXED_LOG_M, atol=0.1)
        nlp_truth = -float(bulge_disk_fm.log_posterior(bulge_disk_truth))
        assert result.neg_log_posterior <= nlp_truth + 5.0
        shapes = model.component_shapes(result.theta)
        assert 2.0 < float(shapes[0]["n"]) < 8.0  # the bulge stays cuspy
        assert float(shapes[0]["sigma"].sum()) < float(shapes[1]["sigma"].sum())


def test_package_exports():
    """Public names are exported from the top-level package."""
    import arachne

    for name in [
        "blind_initial_theta",
        "blind_initial_full_theta",
        "reference_band_index",
        "find_map",
        "multistart_map",
        "solve_component_masses",
        "image_moments",
        "MAPResult",
    ]:
        assert hasattr(arachne, name) and name in arachne.__all__


# ---------------------------------------------------------------------------
# Nuisance-aware initialisation (full theta: spatial + nuisance)
# ---------------------------------------------------------------------------


@pytest.fixture
def sky_fm(model_k2, emulator, gaussian_psf, theta_true):
    """``(fm, sky_value)``: the K=2 mock with a constant pedestal in band 1.

    A ``NuisanceModel`` with a per-band sky is attached, so ``theta`` is the
    full spatial+nuisance vector and the true sky of band 1 is ``sky_value``
    (15 sigma per pixel — large enough that ignoring it visibly biases the
    linear mass solve).
    """
    from arachne.forward_model.nuisance import NuisanceModel
    from arachne.psf.convolution import PSFConvolver

    conv = PSFConvolver(gaussian_psf, image_shape=(H, W))
    truth = np.asarray(conv(model_k2.model_image(theta_true, emulator, (H, W))))
    sigma = truth.max(axis=(1, 2)) / 100.0
    sky_value = float(truth.max(axis=(1, 2))[1] * 0.2)

    rng = np.random.default_rng(3)
    flux = truth + rng.normal(size=truth.shape) * sigma[:, None, None]
    flux[1] += sky_value
    variance = np.broadcast_to(sigma[:, None, None] ** 2, truth.shape).astype(np.float32)
    obs = ObservationCube(
        flux=flux.astype(np.float32),
        variance=variance,
        mask=np.ones_like(variance),
        band_names=BAND_NAMES,
        pixel_scale=0.031,
        wcs=None,
    )
    nuisance = NuisanceModel(N_BANDS, fit_sky=True, sky_prior_sigma=1.0)
    fm = ForwardModel.build(obs, gaussian_psf, model_k2, emulator, nuisance=nuisance)
    return fm, sky_value


class TestNuisanceAwareInitialisation:
    """solve_component_masses / find_map on a full (spatial + nuisance) theta."""

    def test_mass_solve_uses_the_fitted_sky(self, sky_fm, model_k2, theta_true):
        """A pedestal left out of theta biases the mass; put in theta it does not."""
        fm, sky_value = sky_fm
        lo, hi = PARAM_BOUNDS[MASS]
        blocks, shared = model_k2.split_theta(theta_true)
        wrong = blocks.at[:, 5].set(jnp.array([_raw(8.0, lo, hi), _raw(10.8, lo, hi)]))
        theta_spatial = model_k2.join_theta(wrong, shared)

        no_sky = jnp.concatenate([theta_spatial, jnp.zeros(N_BANDS, dtype=jnp.float32)])
        with_sky = jnp.concatenate(
            [theta_spatial, jnp.array([0.0, sky_value, 0.0], dtype=jnp.float32)]
        )

        biased = _log_masses(model_k2, solve_component_masses(fm, no_sky))
        unbiased = _log_masses(model_k2, solve_component_masses(fm, with_sky))
        # the unmodelled pedestal is soaked up by the extended component
        assert np.max(np.abs(biased - np.array(TRUE_LOG_M))) > 0.05
        np.testing.assert_allclose(unbiased, TRUE_LOG_M, atol=0.02)
        # only the mass raws change; the nuisance block is returned untouched
        solved = solve_component_masses(fm, with_sky)
        assert solved.shape == with_sky.shape
        np.testing.assert_array_equal(
            np.asarray(solved)[model_k2.n_params :], np.asarray(with_sky)[model_k2.n_params :]
        )

    def test_spatial_only_theta_still_accepted(self, sky_fm, theta_true):
        """A spatial-only theta is treated as "no nuisance", as before."""
        fm, _ = sky_fm
        solved = solve_component_masses(fm, theta_true)
        assert solved.shape == theta_true.shape

    def test_blind_initial_full_theta(self, sky_fm, model_k2):
        """The convenience wrapper appends the nuisance block at its prior mean."""
        fm, _ = sky_fm
        spatial = blind_initial_theta(model_k2, fm.observation)
        full = blind_initial_full_theta(fm)
        assert full.shape == (fm.n_params,)
        assert full.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(full[: model_k2.n_params]), np.asarray(spatial))
        np.testing.assert_array_equal(np.asarray(full[model_k2.n_params :]), 0.0)
        # an explicit observation and blind_initial_theta kwargs are forwarded
        other = blind_initial_full_theta(fm, fm.observation, size_scales=[0.5, 2.0])
        assert other.shape == (fm.n_params,)

    def test_find_map_keeps_the_nuisance_block(self, sky_fm):
        """find_map on a full theta returns a full theta and fits the sky."""
        fm, sky_value = sky_fm
        theta0 = blind_initial_full_theta(fm)
        result = find_map(fm, theta0, n_rounds=2, steps_per_round=200, final_steps=200)
        assert result.theta.shape == (fm.n_params,)
        n_spatial = fm.spatial_model.n_params
        fitted_sky = np.asarray(result.theta[n_spatial:])
        assert abs(float(fitted_sky[1]) - sky_value) < 0.3 * sky_value
        # ordering the components did not truncate the vector
        _, sigma, _, _ = fm.spatial_model.component_params(result.theta)
        assert float(sigma[0].sum()) < float(sigma[1].sum())
