"""Tests for AdditiveComponentModel and its ForwardModel integration."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.emulator.base import SPSEmulator
from arachne.forward_model.pipeline import ForwardModel
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

# ---------------------------------------------------------------------------
# Helper emulators
# ---------------------------------------------------------------------------


class LinearMassEmulator(SPSEmulator, eqx.Module):
    """flux_b = 10**(logM - 9) * (b + 1) * (1 + 0.1 * tau_v): exactly linear in 10**logM."""

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
        """Mass-linear flux with a mild tau_v colour term."""
        m9 = 10.0 ** (params[:, 0:1] - 9.0)
        colour = 1.0 + 0.1 * params[:, 2:3]
        bands = jnp.arange(1, len(self._band_names) + 1, dtype=jnp.float32)[None, :]
        return m9 * colour * bands


class RecordingEmulator:
    """Plain (non-jit) wrapper that records the shape of every predict() input."""

    def __init__(self, inner: SPSEmulator) -> None:
        """Wrap ``inner`` and start an empty call log."""
        self.inner = inner
        self.calls: list[tuple[int, ...]] = []

    @property
    def param_names(self) -> list[str]:
        """Delegate to the wrapped emulator."""
        return self.inner.param_names

    @property
    def band_names(self) -> list[str]:
        """Delegate to the wrapped emulator."""
        return self.inner.band_names

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Record the input shape, then delegate."""
        self.calls.append(tuple(params.shape))
        return self.inner.predict(params)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_emulator() -> LinearMassEmulator:
    """Mass-linear emulator with the 3 conftest SPS params and 3 bands."""
    return LinearMassEmulator(_param_names=SPS_PARAM_NAMES, _band_names=BAND_NAMES)


@pytest.fixture
def additive_k2() -> AdditiveComponentModel:
    """K=2 additive model, 3 free SPS params, 16x16 image."""
    return AdditiveComponentModel(
        n_components=2,
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(H, W),
        mass_param=MASS,
    )


@pytest.fixture
def additive_k1() -> AdditiveComponentModel:
    """K=1 additive model, 3 free SPS params, 16x16 image."""
    return AdditiveComponentModel(
        n_components=1,
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(H, W),
        mass_param=MASS,
    )


@pytest.fixture
def additive_mixed() -> AdditiveComponentModel:
    """K=2 model with one shared (redshift) and one fixed (log_zmet) parameter."""
    return AdditiveComponentModel(
        n_components=2,
        emulator_param_names=FULL_NAMES,
        param_bounds=FULL_BOUNDS,
        image_shape=(H, W),
        shared_param_names=["redshift"],
        fixed_params={"log_zmet": -0.3},
        mass_param=MASS,
    )


def _block(mu_y, mu_x, log_sy, log_sx, atanh_rho, sps_raw):
    return jnp.array([mu_y, mu_x, log_sy, log_sx, atanh_rho, *sps_raw], dtype=jnp.float32)


def _theta_k2(model, sps0=(0.0, 0.0, 0.0), sps1=(0.0, 0.0, 0.0)):
    """Two well-separated components on the 16x16 grid."""
    b0 = _block(5.0, 5.0, jnp.log(1.5), jnp.log(1.5), 0.0, sps0)
    b1 = _block(10.0, 10.0, jnp.log(3.0), jnp.log(2.0), 0.3, sps1)
    return model.join_theta(jnp.stack([b0, b1]), jnp.zeros(model.n_shared))


# ---------------------------------------------------------------------------
# Construction / layout
# ---------------------------------------------------------------------------


class TestConstruction:
    """Parameter-role validation and theta layout."""

    def test_n_params_plain(self, additive_k2):
        """K * (5 + N_free) with no shared params."""
        assert additive_k2.n_params == 2 * (5 + 3)
        assert additive_k2.n_params_per_component == 8
        assert additive_k2.sps_param_names == SPS_PARAM_NAMES
        assert additive_k2.shared_param_names == []
        assert additive_k2.mass_index == 0

    def test_n_params_shared_and_fixed(self, additive_mixed):
        """K * (5 + N_free) + N_shared with a shared and a fixed param."""
        assert additive_mixed.sps_param_names == ["log_stellar_mass", "log_age", "tau_v"]
        assert additive_mixed.shared_param_names == ["redshift"]
        assert additive_mixed.fixed_params == {"log_zmet": -0.3}
        assert additive_mixed.n_params == 2 * (5 + 3) + 1

    def test_split_join_roundtrip(self, additive_mixed):
        """join_theta(split_theta(theta)) == theta with correct shapes."""
        theta = jnp.arange(additive_mixed.n_params, dtype=jnp.float32)
        blocks, shared = additive_mixed.split_theta(theta)
        assert blocks.shape == (2, 8)
        assert shared.shape == (1,)
        assert float(shared[0]) == additive_mixed.n_params - 1
        np.testing.assert_array_equal(additive_mixed.join_theta(blocks, shared), theta)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(fixed_params={"nope": 1.0}),
            dict(shared_param_names=["nope"]),
            dict(shared_param_names=["tau_v"], fixed_params={"tau_v": 1.0}),
            dict(shared_param_names=[MASS]),
            dict(fixed_params={MASS: 9.0}),
            dict(mass_param="not_a_param"),
            dict(param_bounds={"log_age": (7.0, 10.0), "tau_v": (0.0, 4.0)}),
        ],
    )
    def test_value_errors(self, kwargs):
        """Unknown names, overlaps, non-per-component mass, missing bounds raise."""
        base = dict(
            n_components=1,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(H, W),
            mass_param=MASS,
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            AdditiveComponentModel(**base)


# ---------------------------------------------------------------------------
# Profiles / physical params
# ---------------------------------------------------------------------------


class TestProfiles:
    """Surface-brightness profiles."""

    def test_profiles_sum_to_one(self, additive_k2):
        """Each component profile sums to 1 over the image."""
        p = additive_k2.profiles(_theta_k2(additive_k2))
        assert p.shape == (2, H, W)
        np.testing.assert_allclose(p.sum(axis=(1, 2)), [1.0, 1.0], atol=1e-5)

    def test_profile_peak_at_mu(self, additive_k2):
        """The maximum of each profile lies at the (integer) centre mu."""
        p = np.asarray(additive_k2.profiles(_theta_k2(additive_k2)))
        assert np.unravel_index(p[0].argmax(), p[0].shape) == (5, 5)
        assert np.unravel_index(p[1].argmax(), p[1].shape) == (10, 10)

    def test_off_image_component_finite(self, additive_k1):
        """A component far outside the frame still gives a finite, normalised profile."""
        theta = _block(500.0, -300.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0))
        p = additive_k1.profiles(theta)
        assert jnp.all(jnp.isfinite(p))
        np.testing.assert_allclose(float(p.sum()), 1.0, atol=1e-5)

    def test_component_params_columns(self, additive_mixed):
        """Fixed and shared values land in the right emulator columns."""
        theta = jnp.zeros(additive_mixed.n_params)
        # shared redshift raw = 2.0 -> 10 * sigmoid(2)
        theta = theta.at[-1].set(2.0)
        # component 1 log_age raw = -1
        blocks, shared = additive_mixed.split_theta(theta)
        blocks = blocks.at[1, 5 + 1].set(-1.0)
        theta = additive_mixed.join_theta(blocks, shared)

        mu, sigma, rho, sps = additive_mixed.component_params(theta)
        assert mu.shape == (2, 2) and sigma.shape == (2, 2) and rho.shape == (2,)
        assert sps.shape == (2, len(FULL_NAMES))
        # fixed column
        np.testing.assert_allclose(sps[:, FULL_NAMES.index("log_zmet")], -0.3, rtol=1e-6)
        # shared column, identical in every row
        z = 10.0 * jax.nn.sigmoid(2.0)
        np.testing.assert_allclose(sps[:, FULL_NAMES.index("redshift")], z, rtol=1e-6)
        # per-component column
        lo, hi = PARAM_BOUNDS["log_age"]
        np.testing.assert_allclose(sps[0, 1], lo + (hi - lo) * 0.5, rtol=1e-6)
        np.testing.assert_allclose(sps[1, 1], lo + (hi - lo) * jax.nn.sigmoid(-1.0), rtol=1e-6)
        # raw=0 -> midpoint for mass and tau_v
        np.testing.assert_allclose(sps[:, 0], 9.0, rtol=1e-6)
        np.testing.assert_allclose(sps[:, 2], 2.0, rtol=1e-6)
        np.testing.assert_allclose(sigma, 1.0)
        np.testing.assert_allclose(rho, 0.0)

    def test_rho_clipped(self, additive_k1):
        """Correlation rho is clipped to +-0.99."""
        theta = _block(8.0, 8.0, 0.0, 0.0, 50.0, (0.0, 0.0, 0.0))
        _, _, rho, _ = additive_k1.component_params(theta)
        assert float(rho[0]) <= 0.99 + 1e-6


# ---------------------------------------------------------------------------
# Model image
# ---------------------------------------------------------------------------


class TestModelImage:
    """Flux additivity and linearity."""

    def test_shape(self, additive_k2, linear_emulator):
        """model_image returns (N_bands, H, W)."""
        img = additive_k2.model_image(_theta_k2(additive_k2), linear_emulator, (H, W))
        assert img.shape == (N_BANDS, H, W)
        assert jnp.all(jnp.isfinite(img))

    def test_additivity(self, additive_k2, additive_k1, linear_emulator):
        """image(K=2) == image(component 0 alone) + image(component 1 alone)."""
        theta = _theta_k2(additive_k2, sps0=(0.5, 0.0, 1.0), sps1=(-0.5, 1.0, 0.0))
        blocks, _ = additive_k2.split_theta(theta)
        img2 = additive_k2.model_image(theta, linear_emulator, (H, W))
        img_a = additive_k1.model_image(blocks[0], linear_emulator, (H, W))
        img_b = additive_k1.model_image(blocks[1], linear_emulator, (H, W))
        np.testing.assert_allclose(img2, img_a + img_b, rtol=1e-5, atol=1e-6)

    def test_linearity_in_mass(self, additive_k1, linear_emulator):
        """Doubling 10**logM doubles the image with a mass-linear emulator."""
        lo, hi = PARAM_BOUNDS[MASS]

        def raw_for(log_m):
            u = (log_m - lo) / (hi - lo)
            return float(jnp.log(u) - jnp.log1p(-u))

        t1 = _block(8.0, 8.0, jnp.log(2.0), jnp.log(2.0), 0.0, (raw_for(9.0), 0.0, 0.0))
        t2 = _block(
            8.0, 8.0, jnp.log(2.0), jnp.log(2.0), 0.0, (raw_for(9.0 + np.log10(2)), 0.0, 0.0)
        )
        img1 = additive_k1.model_image(t1, linear_emulator, (H, W))
        img2 = additive_k1.model_image(t2, linear_emulator, (H, W))
        np.testing.assert_allclose(img2, 2.0 * img1, rtol=1e-4)

    def test_total_flux_equals_component_sed(self, additive_k1, linear_emulator):
        """Summing the image over pixels recovers the component SED exactly."""
        theta = _block(8.0, 8.0, jnp.log(2.0), jnp.log(2.0), 0.0, (0.3, -0.2, 0.7))
        img = additive_k1.model_image(theta, linear_emulator, (H, W))
        sed = additive_k1.component_seds(theta, linear_emulator)
        assert sed.shape == (1, N_BANDS)
        np.testing.assert_allclose(img.sum(axis=(1, 2)), sed[0], rtol=1e-5)

    def test_emulator_called_with_k_rows(self, additive_k2, linear_emulator):
        """model_image evaluates the emulator exactly once, with K rows."""
        rec = RecordingEmulator(linear_emulator)
        additive_k2.model_image(_theta_k2(additive_k2), rec, (H, W))
        assert rec.calls == [(2, len(SPS_PARAM_NAMES))]

    def test_jit_vmap_grad(self, additive_k2, linear_emulator):
        """model_image is jit/vmap/grad safe."""
        f = jax.jit(lambda t: additive_k2.model_image(t, linear_emulator, (H, W)))
        theta = _theta_k2(additive_k2)
        img = f(theta)
        imgs = jax.vmap(f)(jnp.stack([theta, theta + 0.1]))
        assert imgs.shape == (2, N_BANDS, H, W)
        g = jax.grad(lambda t: f(t).sum())(theta)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.all(jnp.isfinite(img))


# ---------------------------------------------------------------------------
# decode (summary map)
# ---------------------------------------------------------------------------


class TestDecode:
    """Mass-weighted summary map."""

    def test_shape(self, additive_mixed):
        """Summary map has shape (H*W, N_free) — shared and fixed params excluded."""
        d = additive_mixed.decode(jnp.zeros(additive_mixed.n_params), (H, W))
        assert d.shape == (H * W, 3)
        assert jnp.all(jnp.isfinite(d))

    def test_mass_column_is_surface_density_k1(self, additive_k1):
        """For K=1 the mass column is log10(P * 10**logM) and others are constant."""
        theta = _block(8.0, 8.0, jnp.log(4.0), jnp.log(4.0), 0.0, (0.4, -0.7, 1.2))
        d = additive_k1.decode(theta, (H, W))
        _, _, _, sps = additive_k1.component_params(theta)
        p = additive_k1.profiles(theta).reshape(-1)
        expected = jnp.log10(p * 10.0 ** sps[0, 0])
        np.testing.assert_allclose(d[:, 0], expected, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(d[:, 1], sps[0, 1], rtol=1e-5)
        np.testing.assert_allclose(d[:, 2], sps[0, 2], rtol=1e-5)

    def test_mass_weighted_mean_k2(self, additive_k2):
        """Non-mass columns are the mass-weighted mean of component values."""
        theta = _theta_k2(additive_k2, sps0=(1.0, -2.0, 0.0), sps1=(-1.0, 2.0, 0.0))
        d = additive_k2.decode(theta, (H, W))
        _, _, _, sps = additive_k2.component_params(theta)
        lo, hi = PARAM_BOUNDS["log_age"]
        assert jnp.all(d[:, 1] >= min(sps[:, 1]) - 1e-4)
        assert jnp.all(d[:, 1] <= max(sps[:, 1]) + 1e-4)
        # near component 0's centre, its log_age dominates
        idx0 = 5 * W + 5
        assert abs(float(d[idx0, 1]) - float(sps[0, 1])) < 0.05 * (hi - lo)


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------


class TestPrior:
    """log_prior, its Jacobian, and sample_prior."""

    def test_finite_and_differentiable(self, additive_mixed):
        """log_prior is a finite scalar with a finite gradient."""
        theta = jnp.zeros(additive_mixed.n_params)
        lp = additive_mixed.log_prior(theta)
        assert lp.shape == ()
        assert jnp.isfinite(lp)
        g = jax.grad(additive_mixed.log_prior)(theta)
        assert g.shape == theta.shape
        assert jnp.all(jnp.isfinite(g))

    def test_jacobian_difference(self, additive_k1):
        """log_prior(raw=0) - log_prior(raw=3) equals the analytic sigmoid Jacobian gap."""
        shape = (8.0, 8.0, 0.0, 0.0, 0.0)
        t0 = _block(*shape, (0.0, 0.0, 0.0))
        t3 = _block(*shape, (3.0, 3.0, 3.0))
        diff = float(additive_k1.log_prior(t0) - additive_k1.log_prior(t3))
        ls = jax.nn.log_sigmoid
        per_param = float(2 * ls(0.0) - (ls(3.0) + ls(-3.0)))
        np.testing.assert_allclose(diff, 3 * per_param, rtol=1e-5)

    def test_jacobian_bounds_constant(self, additive_k1):
        """Absolute value at raw=0: normalised uniform-in-physical prior (no log(hi-lo) term)."""
        theta = _block(
            *additive_k1.centre.tolist(), *additive_k1.log_size_prior[:1] * 2, 0.0, (0.0, 0.0, 0.0)
        )
        lp = float(additive_k1.log_prior(theta))
        jac = 3 * 2 * float(jax.nn.log_sigmoid(0.0))
        sd_c = additive_k1.centre_prior_sigma
        sd_s = additive_k1.log_size_prior[1]
        sd_r = additive_k1.rho_prior_sigma
        gauss = -2 * (np.log(sd_c) + 0.5 * np.log(2 * np.pi))
        gauss += -2 * (np.log(sd_s) + 0.5 * np.log(2 * np.pi))
        gauss += -(np.log(sd_r) + 0.5 * np.log(2 * np.pi))
        np.testing.assert_allclose(lp, jac + gauss, rtol=1e-5)

    def test_sps_log_prior_hook(self):
        """A physical-space sps_log_prior callable is added to log_prior."""
        model = AdditiveComponentModel(1, SPS_PARAM_NAMES, PARAM_BOUNDS, (H, W), mass_param=MASS)
        model_hook = AdditiveComponentModel(
            1,
            SPS_PARAM_NAMES,
            PARAM_BOUNDS,
            (H, W),
            mass_param=MASS,
            sps_log_prior=lambda sps: -0.5 * jnp.sum(((sps[:, 0] - 9.0) / 0.5) ** 2),
        )
        theta = _block(8.0, 8.0, 0.0, 0.0, 0.0, (1.0, 0.0, 0.0))
        _, _, _, sps = model.component_params(theta)
        expected = -0.5 * ((float(sps[0, 0]) - 9.0) / 0.5) ** 2
        np.testing.assert_allclose(
            float(model_hook.log_prior(theta) - model.log_prior(theta)), expected, rtol=1e-5
        )

    def test_sample_prior_shape_and_bounds(self, additive_mixed):
        """sample_prior gives (n, n_params) and physical SPS values inside bounds."""
        key = jax.random.PRNGKey(0)
        samples = additive_mixed.sample_prior(key, 64)
        assert samples.shape == (64, additive_mixed.n_params)
        assert samples.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(samples))
        _, sigma, rho, sps = jax.vmap(additive_mixed.component_params)(samples)
        for j, name in enumerate(FULL_NAMES):
            if name in additive_mixed.fixed_params:
                continue
            lo, hi = FULL_BOUNDS[name]
            assert jnp.all(sps[:, :, j] >= lo) and jnp.all(sps[:, :, j] <= hi)
        assert jnp.all(sigma > 0)
        assert jnp.all(jnp.abs(rho) <= 0.99)
        # decoded summary maps in bounds too
        d = jax.vmap(lambda t: additive_mixed.decode(t, (H, W)))(samples[:8])
        for c, name in enumerate(additive_mixed.sps_param_names):
            if c == additive_mixed.mass_index:
                continue
            lo, hi = FULL_BOUNDS[name]
            assert jnp.all(d[:, :, c] >= lo - 1e-4) and jnp.all(d[:, :, c] <= hi + 1e-4)
        # log_prior finite at every prior sample
        lps = jax.vmap(additive_mixed.log_prior)(samples)
        assert jnp.all(jnp.isfinite(lps))

    def test_sample_prior_consistent_with_log_prior(self, additive_k1):
        """Sample statistics match the Gaussian hyper-parameters in log_prior."""
        samples = additive_k1.sample_prior(jax.random.PRNGKey(1), 20000)
        blocks = samples  # K=1, no shared params: theta is a single block
        np.testing.assert_allclose(blocks[:, 0:2].mean(0), additive_k1.centre, atol=0.15)
        np.testing.assert_allclose(blocks[:, 0:2].std(0), additive_k1.centre_prior_sigma, rtol=0.05)
        np.testing.assert_allclose(blocks[:, 2:4].mean(0), additive_k1.log_size_prior[0], atol=0.03)
        np.testing.assert_allclose(blocks[:, 4].std(), additive_k1.rho_prior_sigma, rtol=0.05)
        # uniform in physical space: mean of sigmoid(raw) ~ 0.5, std ~ 1/sqrt(12)
        u = jax.nn.sigmoid(blocks[:, 5:])
        np.testing.assert_allclose(u.mean(0), 0.5, atol=0.02)
        np.testing.assert_allclose(u.std(0), 1 / np.sqrt(12), rtol=0.05)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class TestOrdering:
    """Label-symmetry breaking."""

    def test_order_components_by_size(self, additive_mixed):
        """The compact component is moved to block 0; shared params untouched."""
        big = _block(5.0, 5.0, jnp.log(4.0), jnp.log(3.0), 0.0, (1.0, 2.0, 3.0))
        small = _block(9.0, 9.0, jnp.log(1.0), jnp.log(1.5), 0.2, (-1.0, -2.0, -3.0))
        theta = additive_mixed.join_theta(jnp.stack([big, small]), jnp.array([0.7]))
        ordered = additive_mixed.order_components_by_size(theta)
        blocks, shared = additive_mixed.split_theta(ordered)
        np.testing.assert_array_equal(blocks[0], small)
        np.testing.assert_array_equal(blocks[1], big)
        np.testing.assert_array_equal(shared, jnp.array([0.7]))
        # already ordered -> unchanged; jit-safe
        np.testing.assert_array_equal(
            jax.jit(additive_mixed.order_components_by_size)(ordered), ordered
        )


# ---------------------------------------------------------------------------
# ForwardModel integration
# ---------------------------------------------------------------------------


class TestForwardModelIntegration:
    """AdditiveComponentModel inside ForwardModel."""

    @pytest.fixture
    def fm(self, tiny_observation, gaussian_psf, linear_emulator, additive_k2):
        """ForwardModel.build with the K=2 additive model."""
        return ForwardModel.build(
            obs=tiny_observation,
            psf_model=gaussian_psf,
            spatial_model=additive_k2,
            emulator=linear_emulator,
        )

    def test_model_image_shape(self, fm, additive_k2):
        """_model_image returns (N_bands, H, W)."""
        img = fm._model_image(_theta_k2(additive_k2))
        assert img.shape == (N_BANDS, H, W)
        assert jnp.all(jnp.isfinite(img))

    def test_log_posterior_finite_and_grad(self, fm, additive_k2):
        """log_posterior and its gradient are finite; jit compiles."""
        theta = _theta_k2(additive_k2)
        lp = fm.log_posterior(theta)
        assert lp.shape == () and jnp.isfinite(lp)
        g = jax.grad(fm.log_posterior)(theta)
        assert g.shape == theta.shape
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(g != 0.0)
        assert jnp.isfinite(jax.jit(fm.log_posterior)(theta))

    def test_posterior_is_likelihood_plus_prior(self, fm, additive_k2):
        """log_posterior == log_likelihood + log_prior."""
        theta = _theta_k2(additive_k2, sps0=(0.3, 0.1, -0.2))
        lp = float(fm.log_posterior(theta))
        ll = float(fm.log_likelihood(theta))
        lpr = float(fm.log_prior(theta))
        np.testing.assert_allclose(lp, ll + lpr, rtol=1e-6, atol=1e-3)
        np.testing.assert_allclose(lpr, float(additive_k2.log_prior(theta)), rtol=1e-6)

    def test_log_posterior_does_not_decode(self, tiny_observation, gaussian_psf, linear_emulator):
        """The additive path calls the emulator once with K rows and never decode()."""

        class SpyModel(AdditiveComponentModel):
            decode_calls = 0

            def decode(self, theta, image_shape):
                SpyModel.decode_calls += 1
                return super().decode(theta, image_shape)

        model = SpyModel(2, SPS_PARAM_NAMES, PARAM_BOUNDS, (H, W), mass_param=MASS)
        rec = RecordingEmulator(linear_emulator)
        fm = ForwardModel.build(tiny_observation, gaussian_psf, model, rec)
        fm.log_posterior(_theta_k2(model))
        assert SpyModel.decode_calls == 0
        assert rec.calls == [(2, len(SPS_PARAM_NAMES))]

    def test_pixel_map_still_uses_decoded_prior(
        self, tiny_observation, gaussian_psf, mock_emulator, pixel_map_model
    ):
        """FreeFormPixelMap keeps the log_prior_from_decoded path and the identity holds."""
        fm = ForwardModel.build(tiny_observation, gaussian_psf, pixel_map_model, mock_emulator)
        assert fm._prior_needs_decoded
        theta = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (pixel_map_model.n_params,))
        lp = float(fm.log_posterior(theta))
        np.testing.assert_allclose(
            lp, float(fm.log_likelihood(theta)) + float(fm.log_prior(theta)), rtol=1e-6, atol=1e-2
        )

    def test_package_export(self):
        """AdditiveComponentModel is exported at the package top level."""
        import arachne

        assert arachne.AdditiveComponentModel is AdditiveComponentModel
        assert "AdditiveComponentModel" in arachne.__all__


# ---------------------------------------------------------------------------
# Pluggable profiles
# ---------------------------------------------------------------------------


class TestProfileOptions:
    """Sersic / point-source components, coordinate units and normalisation."""

    @pytest.fixture
    def mixed_profiles(self) -> AdditiveComponentModel:
        """K=2 model with a Sersic component and a point source."""
        return AdditiveComponentModel(
            n_components=2,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(H, W),
            mass_param=MASS,
            profiles=["sersic", "point"],
            normalisation="analytic",
        )

    @pytest.fixture
    def arcsec_model(self) -> AdditiveComponentModel:
        """K=1 analytic-normalisation model in arcsec coordinates."""
        return AdditiveComponentModel(
            n_components=1,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(H, W),
            mass_param=MASS,
            pixel_scale=0.05,
            normalisation="analytic",
        )

    @staticmethod
    def _mixed_theta(model):
        sersic = [8.0, 8.0, jnp.log(2.0), jnp.log(3.0), 0.2, jnp.log(4.0)]
        point = [6.0, 9.0]
        return jnp.array([*sersic, 0.0, 0.0, 0.0, *point, 0.0, 0.0, 0.0], dtype=jnp.float32)

    def test_defaults_are_gaussian(self, additive_k2):
        """Without ``profiles`` every component is a Gaussian on the pixel grid."""
        assert [p.name for p in additive_k2.profile_objects] == ["gaussian", "gaussian"]
        assert additive_k2.n_shape_params == (5, 5)
        assert additive_k2.pixel_scale is None
        assert additive_k2.pixel_area == 1.0
        assert additive_k2.normalisation == "frame"
        assert additive_k2.oversample == 1
        yy, xx = additive_k2.coords
        assert yy.shape == (H * W,) and xx.shape == (H * W,)
        np.testing.assert_allclose(additive_k2.centre, [(H - 1) / 2, (W - 1) / 2])

    def test_mixed_layout(self, mixed_profiles):
        """Variable-length blocks: slices, n_params and the split/join contract."""
        m = mixed_profiles
        assert m.n_shape_params == (6, 2)
        assert m.n_params == (6 + 3) + (2 + 3)
        assert m.component_slices == [slice(0, 9), slice(9, 14)]
        assert m.shared_slice == slice(14, 14)
        with pytest.raises(ValueError):
            _ = m.n_params_per_component
        theta = jnp.arange(m.n_params, dtype=jnp.float32)
        blocks, shared = m.split_theta(theta)
        assert isinstance(blocks, list)
        assert [b.shape for b in blocks] == [(9,), (5,)]
        np.testing.assert_array_equal(m.join_theta(blocks, shared), theta)
        np.testing.assert_array_equal(m.shape_raw(theta, 0), theta[0:6])
        np.testing.assert_array_equal(m.sps_raw(theta, 1), theta[11:14])
        updated = m.set_sps_raw(theta, 1, jnp.array([-1.0, -2.0, -3.0]))
        np.testing.assert_array_equal(updated[11:14], [-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(updated[:11], theta[:11])

    def test_bad_profile_spec(self):
        """Wrong-length or unknown profile specifications raise."""
        base = dict(
            n_components=2,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(H, W),
            mass_param=MASS,
        )
        for kwargs in [
            dict(profiles=["gaussian"]),
            dict(profiles="moffat"),
            dict(normalisation="none"),
            dict(oversample=0),
            dict(pixel_scale=-1.0),
        ]:
            with pytest.raises(ValueError):
                AdditiveComponentModel(**base, **kwargs)

    def test_mixed_component_params_and_shapes(self, mixed_profiles):
        """component_params keeps its (mu, sigma, rho, sps) contract for any mix."""
        m = mixed_profiles
        theta = self._mixed_theta(m)
        mu, sigma, rho, sps = m.component_params(theta)
        assert mu.shape == (2, 2) and sigma.shape == (2, 2) and rho.shape == (2,)
        assert sps.shape == (2, len(SPS_PARAM_NAMES))
        np.testing.assert_allclose(mu[1], [6.0, 9.0])
        np.testing.assert_allclose(sigma[0], [2.0, 3.0], rtol=1e-5)
        np.testing.assert_allclose(sigma[1], [0.5, 0.5])  # point_sigma
        np.testing.assert_allclose(float(rho[1]), 0.0)
        shapes = m.component_shapes(theta)
        np.testing.assert_allclose(float(shapes[0]["n"]), 4.0, rtol=1e-4)
        assert "n" not in shapes[1]

    def test_mixed_prior_and_sampling(self, mixed_profiles, linear_emulator):
        """log_prior / sample_prior / model_image work across mixed block lengths."""
        m = mixed_profiles
        theta = self._mixed_theta(m)
        assert jnp.isfinite(m.log_prior(theta))
        g = jax.grad(m.log_prior)(theta)
        assert g.shape == theta.shape and jnp.all(jnp.isfinite(g))
        samples = m.sample_prior(jax.random.PRNGKey(0), 32)
        assert samples.shape == (32, m.n_params)
        assert jnp.all(jnp.isfinite(jax.vmap(m.log_prior)(samples)))
        img = jax.jit(lambda t: m.model_image(t, linear_emulator, (H, W)))(theta)
        assert img.shape == (N_BANDS, H, W) and jnp.all(jnp.isfinite(img))

    def test_analytic_normalisation_loses_flux_off_frame(self, additive_k1):
        """Analytic profiles sum to <= 1; frame profiles sum to exactly 1."""
        analytic = AdditiveComponentModel(
            1,
            SPS_PARAM_NAMES,
            PARAM_BOUNDS,
            (H, W),
            mass_param=MASS,
            normalisation="analytic",
        )
        near_edge = _block(1.0, 1.0, jnp.log(3.0), jnp.log(3.0), 0.0, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(float(additive_k1.profiles(near_edge).sum()), 1.0, atol=1e-5)
        assert 0.2 < float(analytic.profiles(near_edge).sum()) < 0.75
        centred = _block(7.5, 7.5, jnp.log(1.0), jnp.log(1.0), 0.0, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(float(analytic.profiles(centred).sum()), 1.0, rtol=2e-3)

    def test_model_image_on_finer_grid_conserves_flux(self, arcsec_model, linear_emulator):
        """A 2x finer arcsec grid over the same area gives the same total flux."""
        m = arcsec_model
        theta = _block(0.0, 0.0, jnp.log(0.1), jnp.log(0.08), 0.2, (0.0, 0.0, 0.0))
        coarse = m.model_image(theta, linear_emulator, (H, W))
        fine_c = (np.arange(2 * H) - (2 * H - 1) / 2.0) * (m.pixel_scale / 2.0)
        yy, xx = np.meshgrid(fine_c, fine_c, indexing="ij")
        fine = m.model_image_on(
            theta,
            linear_emulator,
            jnp.asarray(yy, dtype=jnp.float32),
            jnp.asarray(xx, dtype=jnp.float32),
            (m.pixel_scale / 2.0) ** 2,
        )
        assert fine.shape == (N_BANDS, 2 * H, 2 * W)
        np.testing.assert_allclose(
            np.asarray(fine.sum(axis=(1, 2))), np.asarray(coarse.sum(axis=(1, 2))), rtol=0.01
        )
        sed = m.component_seds(theta, linear_emulator)[0]
        np.testing.assert_allclose(np.asarray(fine.sum(axis=(1, 2))), np.asarray(sed), rtol=0.01)

    def test_model_image_on_accepts_any_coordinate_shape(self, arcsec_model, linear_emulator):
        """Arbitrary-shaped (yy, xx) in arcsec: output is (N_bands, *yy.shape)."""
        m = arcsec_model
        theta = _block(0.0, 0.0, jnp.log(0.1), jnp.log(0.08), 0.2, (0.0, 0.0, 0.0))
        # a scattered, unstructured list of sky positions (a multi-resolution
        # band's pixel centres need not be a regular grid in this frame)
        rng = np.random.default_rng(0)
        yy = jnp.asarray(rng.uniform(-0.2, 0.2, size=(7,)), dtype=jnp.float32)
        xx = jnp.asarray(rng.uniform(-0.2, 0.2, size=(7,)), dtype=jnp.float32)
        flat = m.model_image_on(theta, linear_emulator, yy, xx, 0.01)
        assert flat.shape == (N_BANDS, 7)
        # the same points reshaped: values follow the requested layout exactly
        block = m.model_image_on(
            theta, linear_emulator, yy[:6].reshape(2, 3), xx[:6].reshape(2, 3), 0.01
        )
        assert block.shape == (N_BANDS, 2, 3)
        np.testing.assert_allclose(
            np.asarray(block).reshape(N_BANDS, 6), np.asarray(flat)[:, :6], rtol=1e-6
        )
        # linear in the target pixel area, and jit/grad safe
        double = m.model_image_on(theta, linear_emulator, yy, xx, 0.02)
        np.testing.assert_allclose(np.asarray(double), 2.0 * np.asarray(flat), rtol=1e-5)
        g = jax.grad(lambda t: jnp.sum(m.model_image_on(t, linear_emulator, yy, xx, 0.01)))(theta)
        assert jnp.all(jnp.isfinite(g))

    def test_frame_normalisation_rejects_foreign_grid(self, additive_k1, linear_emulator):
        """Frame normalisation is only defined on the model's own grid."""
        theta = _block(8.0, 8.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0))
        yy, xx = additive_k1.coords
        # the model's own arrays are accepted
        assert additive_k1.model_image_on(theta, linear_emulator, yy, xx, 1.0).shape == (
            N_BANDS,
            H * W,
        )
        with pytest.raises(ValueError, match="own grid"):
            additive_k1.model_image_on(theta, linear_emulator, yy * 1.0, xx * 1.0, 1.0)

    def test_component_images_sum_to_model_image(self, additive_k2, linear_emulator):
        """component_images summed over K equals model_image."""
        theta = _theta_k2(additive_k2, sps0=(0.4, -0.2, 0.9), sps1=(-0.3, 0.8, 0.1))
        comps = additive_k2.component_images(theta, linear_emulator)
        assert comps.shape == (2, N_BANDS, H, W)
        np.testing.assert_allclose(
            comps.sum(axis=0),
            additive_k2.model_image(theta, linear_emulator, (H, W)),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_oversampling_changes_a_cuspy_render(self, linear_emulator):
        """A Sersic n=4 component is rendered differently with oversample > 1."""
        kwargs = dict(
            n_components=1,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(H, W),
            mass_param=MASS,
            profiles="sersic",
            normalisation="analytic",
        )
        # mu sits exactly on a pixel centre, so a single sample per pixel lands
        # on the cusp and grossly overestimates the central pixel.
        theta = jnp.array(
            [8.0, 8.0, jnp.log(2.0), jnp.log(2.0), 0.0, jnp.log(4.0), 0.0, 0.0, 0.0],
            dtype=jnp.float32,
        )
        plain = AdditiveComponentModel(**kwargs).profiles(theta)
        fine = AdditiveComponentModel(**kwargs, oversample=5).profiles(theta)
        assert float(plain.max()) > 3.0 * float(fine.max())
        assert float(fine.sum()) < float(plain.sum())
        # The resolved outskirts are unaffected.  "Resolved" has to be measured
        # in effective radii, not in brightness: an n=4 cusp with r_e = 2 px is
        # still curving steeply across the pixels at r = 1-2 px, where a single
        # central sample is off by ~7%.  Beyond 2 r_e the profile is smooth on a
        # pixel scale and the two renders agree (oversample=5 is itself
        # converged to 0.1% there, checked against oversample=11).
        yy, xx = np.mgrid[0:H, 0:W]
        outer = (yy - 8.0) ** 2 + (xx - 8.0) ** 2 > (2 * 2.0) ** 2
        np.testing.assert_allclose(
            np.asarray(plain)[0][outer], np.asarray(fine)[0][outer], rtol=0.03
        )

    def test_ordering_within_profile_groups(self):
        """Components are sorted by size among those sharing a profile; others stay put."""
        model = AdditiveComponentModel(
            3,
            SPS_PARAM_NAMES,
            PARAM_BOUNDS,
            (H, W),
            mass_param=MASS,
            profiles=["sersic", "point", "sersic"],
        )
        big = [4.0, 4.0, jnp.log(5.0), jnp.log(4.0), 0.0, jnp.log(4.0), 1.0, 2.0, 3.0]
        point = [8.0, 8.0, 0.5, 0.5, 0.5]
        small = [9.0, 9.0, jnp.log(1.0), jnp.log(1.2), 0.1, jnp.log(1.0), -1.0, -2.0, -3.0]
        theta = jnp.array([*big, *point, *small], dtype=jnp.float32)
        ordered = jax.jit(model.order_components_by_size)(theta)
        np.testing.assert_allclose(ordered[model.component_slices[0]], jnp.array(small))
        np.testing.assert_allclose(ordered[model.component_slices[1]], jnp.array(point))
        np.testing.assert_allclose(ordered[model.component_slices[2]], jnp.array(big))

    def test_profiles_exported(self):
        """Profile classes are exported at the package top level."""
        import arachne

        for name in ["Profile", "GaussianProfile", "SersicProfile", "PointSourceProfile"]:
            assert hasattr(arachne, name) and name in arachne.__all__


def test_order_components_by_size_passes_nuisance_tail_through():
    """A full forward-model theta (spatial + nuisance) keeps its nuisance tail."""
    import jax.numpy as jnp
    import numpy as np

    from arachne.spatial.additive import AdditiveComponentModel

    names = ["log_mass", "Av"]
    bounds = {"log_mass": (6.0, 12.0), "Av": (0.0, 4.0)}
    model = AdditiveComponentModel(2, names, bounds, (16, 16), mass_param="log_mass")
    theta_s = jnp.asarray(np.random.default_rng(0).normal(size=model.n_params), dtype=jnp.float32)
    # make component 0 the larger one so a swap happens
    sl1 = model.component_slices[1]
    theta_s = theta_s.at[2:4].set(2.0).at[sl1.start + 2 : sl1.start + 4].set(-1.0)
    tail = jnp.asarray([7.0, -3.0, 0.5], dtype=jnp.float32)
    full = jnp.concatenate([theta_s, tail])
    out = model.order_components_by_size(full)
    assert out.shape == full.shape
    np.testing.assert_array_equal(np.asarray(out[model.n_params :]), np.asarray(tail))
    np.testing.assert_allclose(
        np.asarray(out[: model.n_params]), np.asarray(model.order_components_by_size(theta_s))
    )
    with pytest.raises(ValueError):
        model.order_components_by_size(theta_s[:-1])
