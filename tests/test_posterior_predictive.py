"""Tests for posterior-predictive images and residual diagnostics.

Uses the tiny linear-emulator + AdditiveComponentModel mock of
``tests/test_nss_sampler.py`` so everything runs on CPU in a couple of seconds.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.data.observation import ObservationCube
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.posterior_predictive import (
    _chunked_map,
    chi2_reduced,
    component_image_samples,
    is_multiresolution,
    model_image_samples,
    predictive_bands,
    residual_summary,
)
from arachne.psf.convolution import PSFConvolver
from arachne.spatial.additive import AdditiveComponentModel

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


class LinearMassEmulator(SPSEmulator, eqx.Module):
    """flux_b = 10**(logM - 9) * (b + 1) * (1 + 0.1 * tau_v)."""

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


def _emulator() -> LinearMassEmulator:
    return LinearMassEmulator(_param_names=SPS_PARAM_NAMES, _band_names=BAND_NAMES)


def _model(k: int) -> AdditiveComponentModel:
    return AdditiveComponentModel(
        n_components=k,
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(H, W),
        mass_param=MASS,
    )


def _component_theta(y: float, x: float, log_mass: float) -> list[float]:
    """One component's 8 raw parameters (position, size, rho, then SPS)."""
    lo, hi = PARAM_BOUNDS[MASS]
    u = (log_mass - lo) / (hi - lo)
    return [y, x, float(np.log(2.0)), float(np.log(2.5)), 0.1, float(np.log(u / (1 - u))), 0.0, 0.0]


_MOCK: dict = {}


@pytest.fixture
def mock(gaussian_psf):
    """(forward_model, theta_true, noiseless_image, sigma) for a K=2 truth.

    Cached at module level so every test shares one ForwardModel (and hence
    one set of XLA-compiled kernels).
    """
    if "value" in _MOCK:
        return _MOCK["value"]
    model = _model(2)
    emulator = _emulator()
    theta_true = jnp.asarray(
        _component_theta(5.5, 6.0, 9.5) + _component_theta(10.0, 10.5, 9.2),
        dtype=jnp.float32,
    )
    conv = PSFConvolver(gaussian_psf, image_shape=(H, W))
    truth = np.asarray(conv(model.model_image(theta_true, emulator, (H, W))))
    sigma = truth.max(axis=(1, 2)) / 20.0
    rng = np.random.default_rng(7)
    flux = (truth + rng.normal(size=truth.shape) * sigma[:, None, None]).astype(np.float32)
    variance = np.broadcast_to(sigma[:, None, None] ** 2, truth.shape).astype(np.float32)
    obs = ObservationCube(flux, variance, np.ones_like(flux), BAND_NAMES, 0.031, None)
    fm = ForwardModel.build(obs, gaussian_psf, model, emulator)
    _MOCK["value"] = (fm, theta_true, jnp.asarray(truth), sigma)
    return _MOCK["value"]


@pytest.fixture
def truth_samples(mock):
    """Twenty copies of the true theta — the posterior of a perfect fit."""
    _, theta_true, _, _ = mock
    return jnp.tile(theta_true[None, :], (20, 1))


class TestModelImageSamples:
    """model_image_samples shapes, thinning and correctness."""

    def test_shape(self, mock, truth_samples):
        """Returns (n, N_bands, H, W)."""
        fm = mock[0]
        images = model_image_samples(fm, truth_samples, n_max=20)
        assert images.shape == (20, N_BANDS, H, W)
        assert jnp.all(jnp.isfinite(images))

    def test_thinning_respects_n_max(self, mock, truth_samples):
        """More samples than n_max are thinned down."""
        fm = mock[0]
        assert model_image_samples(fm, truth_samples, n_max=7).shape[0] == 7

    def test_random_thinning(self, mock, truth_samples):
        """A PRNG key selects a random subset of the right size."""
        fm = mock[0]
        images = model_image_samples(fm, truth_samples, n_max=5, rng_key=jax.random.PRNGKey(0))
        assert images.shape == (5, N_BANDS, H, W)

    def test_matches_forward_model_directly(self, mock, truth_samples):
        """Chunked vmap agrees with a single _model_image call.

        Tolerance note: the vmapped batch of ``n`` thetas and the single-theta
        call are *not* bit-identical in float32.  XLA picks a different FFT /
        reduction schedule for a batched convolution than for an unbatched one,
        so the summation order inside the PSF convolution differs.  The
        observed disagreement is ~1e-8 in absolute terms against a peak flux of
        ~0.3, i.e. ~1e-7 of the image scale (float32 eps is 1.2e-7), but it
        shows up as a ~5e-4 *relative* error in the faint wings where the model
        is 1e-8.  We therefore test with a relative tolerance plus an absolute
        floor tied to the image peak, rather than forcing the two code paths to
        be identical (which would mean giving up batched evaluation, the whole
        point of the chunked vmap).
        """
        fm, theta_true, _, _ = mock
        images = model_image_samples(fm, truth_samples, n_max=3)
        direct = np.asarray(fm._model_image(theta_true))
        atol = 1e-6 * float(np.max(np.abs(direct)))
        np.testing.assert_allclose(np.asarray(images[0]), direct, rtol=1e-3, atol=atol)

    def test_chunking_is_transparent(self, mock, truth_samples):
        """A chunk size that does not divide n gives the same answer.

        Same float32 caveat as ``test_matches_forward_model_directly``: chunk
        sizes 16 and 7 compile to different batched convolutions, so agreement
        is to ~1e-7 of the image peak, not bit-exact.
        """
        fm = mock[0]
        a = np.asarray(model_image_samples(fm, truth_samples, n_max=20, chunk=16))
        b = np.asarray(model_image_samples(fm, truth_samples, n_max=20, chunk=7))
        atol = 1e-6 * float(np.max(np.abs(a)))
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=atol)


class TestComponentImageSamples:
    """component_image_samples shapes and additivity."""

    def test_shape(self, mock, truth_samples):
        """Returns (n, K, N_bands, H, W)."""
        fm = mock[0]
        comps = component_image_samples(fm, truth_samples, n_max=6)
        assert comps.shape == (6, 2, N_BANDS, H, W)
        assert jnp.all(jnp.isfinite(comps))

    def test_components_sum_to_unconvolved_image(self, mock, truth_samples):
        """Summing the components reproduces the unconvolved model image."""
        fm, theta_true, _, _ = mock
        comps = component_image_samples(fm, truth_samples, n_max=2)
        total = jnp.sum(comps[0], axis=0)
        theta_spatial = fm.split_theta(theta_true)[0]
        expected = fm.spatial_model.model_image(theta_spatial, fm.emulator, (H, W))
        np.testing.assert_allclose(np.asarray(total), np.asarray(expected), rtol=1e-4)

    def test_components_are_unconvolved(self, mock, truth_samples):
        """The per-component images are sharper than the convolved model."""
        fm = mock[0]
        comps = np.asarray(component_image_samples(fm, truth_samples, n_max=1))[0]
        convolved = np.asarray(model_image_samples(fm, truth_samples, n_max=1))[0]
        assert comps.sum(axis=0).max() > convolved.max()

    def test_unsupported_model_raises(
        self, tiny_observation, gaussian_psf, mock_emulator, gmm_model
    ):
        """A model without a component decomposition is rejected clearly."""
        fm = ForwardModel.build(tiny_observation, gaussian_psf, gmm_model, mock_emulator)
        with pytest.raises(AttributeError, match="component"):
            component_image_samples(fm, jnp.zeros((2, gmm_model.n_params)), n_max=2)


class TestResidualSummary:
    """residual_summary values on a truth-model fit of noisy data."""

    def test_keys_and_shapes(self, mock, truth_samples):
        """Every documented key is present with the right shape."""
        fm = mock[0]
        s = residual_summary(fm, truth_samples, n_max=10)
        assert s["median_model"].shape == (N_BANDS, H, W)
        assert s["chi"].shape == (N_BANDS, H, W)
        assert s["chi2_red_per_band"].shape == (N_BANDS,)
        assert s["n_data"] == N_BANDS * H * W
        assert s["n_params"] == fm.n_params
        assert s["dof"] == s["n_data"] - s["n_params"]
        assert s["band_names"] == BAND_NAMES
        assert s["n_samples_used"] == 10

    def test_median_model_matches_truth(self, mock, truth_samples):
        """With identical samples the median image is the truth image."""
        fm, _, truth_image, _ = mock
        s = residual_summary(fm, truth_samples, n_max=5)
        np.testing.assert_allclose(
            np.asarray(s["median_model"]), np.asarray(truth_image), rtol=1e-4, atol=1e-4
        )

    def test_chi2_red_near_one_for_truth(self, mock, truth_samples):
        """The true model on photon-noise data gives reduced chi2 ~ 1."""
        fm = mock[0]
        s = residual_summary(fm, truth_samples, n_max=5)
        assert 0.75 < s["chi2_red"] < 1.3
        assert np.all(s["chi2_red_per_band"] > 0.6)
        assert np.all(s["chi2_red_per_band"] < 1.5)

    def test_chi_map_is_standard_normal(self, mock, truth_samples):
        """The chi map has ~unit scatter and few |chi| > 3 outliers."""
        fm = mock[0]
        s = residual_summary(fm, truth_samples, n_max=5)
        chi = np.asarray(s["chi"])
        assert abs(float(chi.std()) - 1.0) < 0.2
        assert s["frac_chi_gt_3"] < 0.02

    def test_bad_model_inflates_chi2(self, mock, truth_samples):
        """Halving every mass makes the fit visibly worse."""
        fm, theta_true, _, _ = mock
        good = residual_summary(fm, truth_samples, n_max=3)["chi2_red"]
        bad_theta = theta_true.at[5].add(-1.0).at[13].add(-1.0)
        bad = residual_summary(fm, jnp.tile(bad_theta[None, :], (3, 1)), n_max=3)["chi2_red"]
        assert bad > 5.0 * good


class TestChi2Reduced:
    """chi2_reduced on a single theta."""

    def test_truth_gives_about_one(self, mock):
        """The generating theta has reduced chi2 ~ 1."""
        fm, theta_true, _, _ = mock
        assert 0.75 < chi2_reduced(fm, theta_true) < 1.3

    def test_consistent_with_residual_summary(self, mock, truth_samples):
        """Single-theta chi2 matches the median-image chi2 for constant samples."""
        fm, theta_true, _, _ = mock
        s = residual_summary(fm, truth_samples, n_max=3)
        assert chi2_reduced(fm, theta_true) == pytest.approx(s["chi2_red"], rel=1e-3)


class TestPredictiveBands:
    """predictive_bands percentile envelopes."""

    def test_shapes_and_order(self):
        """p16 <= p50 <= p84, with the sample axis removed."""
        images = jnp.asarray(np.random.default_rng(0).normal(size=(50, N_BANDS, H, W)))
        p16, p50, p84 = predictive_bands(images)
        assert p16.shape == p50.shape == p84.shape == (N_BANDS, H, W)
        assert jnp.all(p16 <= p50) and jnp.all(p50 <= p84)

    def test_works_on_component_cubes(self, mock, truth_samples):
        """The (n, K, N_bands, H, W) layout is handled too."""
        fm = mock[0]
        comps = component_image_samples(fm, truth_samples, n_max=4)
        p16, p50, p84 = predictive_bands(comps)
        assert p50.shape == (2, N_BANDS, H, W)

    def test_needs_sample_axis(self):
        """A 1-D input is rejected."""
        with pytest.raises(ValueError):
            predictive_bands(jnp.zeros(5))


def test_package_exports():
    """Posterior-predictive helpers are re-exported at package level."""
    import arachne

    assert arachne.model_image_samples is model_image_samples
    assert arachne.component_image_samples is component_image_samples
    assert arachne.residual_summary is residual_summary
    for name in ("model_image_samples", "component_image_samples", "residual_summary"):
        assert name in arachne.__all__


class TestMultiResolutionDispatch:
    """The helpers branch on ``model_images``; a single-grid model is unchanged."""

    def test_single_grid_is_not_multiresolution(self, mock):
        """A plain ForwardModel takes the stacked-array path."""
        fm = mock[0]
        assert is_multiresolution(fm) is False
        assert hasattr(fm, "_model_image") and not hasattr(fm, "model_images")

    def test_chunked_map_stacks_ragged_pytrees(self):
        """``_chunked_map`` concatenates every leaf of a list-valued result.

        This is what lets ``model_image_samples`` return one stack per band for
        a multi-resolution model, whose bands have different shapes.
        """
        xs = jnp.arange(10, dtype=jnp.float32)[:, None]

        def fn(row):
            return [row * jnp.ones((2, 3)), row * jnp.ones((4,))]

        out = _chunked_map(fn, xs, chunk=4)
        assert isinstance(out, list) and len(out) == 2
        assert out[0].shape == (10, 2, 3) and out[1].shape == (10, 4)
        np.testing.assert_allclose(np.asarray(out[0][:, 0, 0]), np.arange(10))
