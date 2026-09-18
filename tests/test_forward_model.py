"""Tests for ForwardModel pipeline.

The critical tests are:
- test_grad_log_posterior: jax.grad(log_posterior) is finite and non-zero
- test_jit_compiles: jax.jit(log_posterior) runs without error
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.data.observation import ObservationCube
from arachne.forward_model.nuisance import NuisanceModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.likelihood.gaussian import GaussianLikelihood
from arachne.psf.convolution import PSFConvolver


@pytest.fixture
def forward_model_gmm(tiny_observation, gaussian_psf, mock_emulator, gmm_model):
    """ForwardModel with GMM spatial model."""
    return ForwardModel.build(
        obs=tiny_observation,
        psf_model=gaussian_psf,
        spatial_model=gmm_model,
        emulator=mock_emulator,
    )


@pytest.fixture
def forward_model_pixel_map(tiny_observation, gaussian_psf, mock_emulator, pixel_map_model):
    """ForwardModel with FreeFormPixelMap spatial model."""
    return ForwardModel.build(
        obs=tiny_observation,
        psf_model=gaussian_psf,
        spatial_model=pixel_map_model,
        emulator=mock_emulator,
    )


class TestForwardModelGMM:
    """Tests using the GMM spatial model."""

    def test_model_image_shape(self, forward_model_gmm, gmm_model):
        """_model_image() returns (N_bands, H, W)."""
        theta = jnp.zeros(gmm_model.n_params)
        img = forward_model_gmm._model_image(theta)
        assert img.shape == (3, 16, 16)

    def test_log_posterior_scalar(self, forward_model_gmm, gmm_model):
        """log_posterior() returns a finite scalar."""
        theta = jnp.zeros(gmm_model.n_params)
        lp = forward_model_gmm.log_posterior(theta)
        assert lp.shape == ()
        assert jnp.isfinite(lp)

    def test_grad_log_posterior(self, forward_model_gmm, gmm_model):
        """jax.grad(log_posterior) is finite and non-zero."""
        theta = jnp.zeros(gmm_model.n_params)
        grad = jax.grad(forward_model_gmm.log_posterior)(theta)
        assert grad.shape == theta.shape
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)

    def test_jit_compiles(self, forward_model_gmm, gmm_model):
        """jax.jit(log_posterior) compiles and runs."""
        logpost_jit = jax.jit(forward_model_gmm.log_posterior)
        theta = jnp.zeros(gmm_model.n_params)
        lp = logpost_jit(theta)
        assert jnp.isfinite(lp)

    def test_jit_grad_compiles(self, forward_model_gmm, gmm_model):
        """jax.jit(jax.grad(log_posterior)) compiles and runs."""
        grad_jit = jax.jit(jax.grad(forward_model_gmm.log_posterior))
        theta = jnp.zeros(gmm_model.n_params)
        grad = grad_jit(theta)
        assert grad.shape == theta.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_model_image_finite(self, forward_model_gmm, gmm_model):
        """_model_image() returns finite values."""
        theta = jnp.zeros(gmm_model.n_params)
        img = forward_model_gmm._model_image(theta)
        assert jnp.all(jnp.isfinite(img))


class TestForwardModelPixelMap:
    """Tests using the FreeFormPixelMap spatial model."""

    def test_log_posterior_scalar(self, forward_model_pixel_map, pixel_map_model):
        """log_posterior() returns a finite scalar."""
        theta = jnp.zeros(pixel_map_model.n_params)
        lp = forward_model_pixel_map.log_posterior(theta)
        assert lp.shape == ()
        assert jnp.isfinite(lp)

    def test_grad_log_posterior(self, forward_model_pixel_map, pixel_map_model):
        """jax.grad(log_posterior) is finite and non-zero."""
        theta = jnp.zeros(pixel_map_model.n_params)
        grad = jax.grad(forward_model_pixel_map.log_posterior)(theta)
        assert grad.shape == theta.shape
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)

    def test_jit_compiles(self, forward_model_pixel_map, pixel_map_model):
        """jax.jit(log_posterior) compiles and runs."""
        logpost_jit = jax.jit(forward_model_pixel_map.log_posterior)
        theta = jnp.zeros(pixel_map_model.n_params)
        lp = logpost_jit(theta)
        assert jnp.isfinite(lp)


class TestForwardModelBuild:
    """Tests for ForwardModel.build() convenience constructor."""

    def test_build_calls_to_jax(
        self, tiny_observation_numpy, gaussian_psf, mock_emulator, gmm_model
    ):
        """ForwardModel.build() converts numpy obs to JAX arrays."""
        fm = ForwardModel.build(
            obs=tiny_observation_numpy,
            psf_model=gaussian_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
        )
        assert isinstance(fm.observation.flux, jnp.ndarray)

    def test_build_creates_convolver(
        self, tiny_observation, gaussian_psf, mock_emulator, gmm_model
    ):
        """ForwardModel.build() creates a PSFConvolver."""
        fm = ForwardModel.build(
            obs=tiny_observation,
            psf_model=gaussian_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
        )
        assert isinstance(fm.convolver, PSFConvolver)

    def test_build_creates_likelihood(
        self, tiny_observation, gaussian_psf, mock_emulator, gmm_model
    ):
        """ForwardModel.build() creates a GaussianLikelihood."""
        fm = ForwardModel.build(
            obs=tiny_observation,
            psf_model=gaussian_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
        )
        assert isinstance(fm.likelihood, GaussianLikelihood)


# ---------------------------------------------------------------------------
# Numerical correctness tests
# ---------------------------------------------------------------------------


def _flat_obs(flux_val: float, bands, H: int = 16, W: int = 16) -> ObservationCube:
    """ObservationCube filled with a constant flux value (JAX arrays)."""
    flux = jnp.full((len(bands), H, W), flux_val, dtype=jnp.float32)
    ones = jnp.ones((len(bands), H, W), dtype=jnp.float32)
    return ObservationCube(
        flux=flux,
        variance=ones,
        mask=ones,
        band_names=bands,
        pixel_scale=0.031,
    )


BANDS = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]


class TestForwardModelNumericalCorrectness:
    """Numerical correctness tests for ForwardModel.log_posterior.

    Uses a delta PSF (identity convolution) and the mock emulator so that
    the predicted flux can be computed analytically:

        theta = 0  →  sigmoid(0) = 0.5
        log_stellar_mass = 6.0 + (12.0 − 6.0) × 0.5 = 9.0
        mock_emulator: flux = |9.0| + 1.0 = 10.0 nJy  (all bands)
    """

    def test_perfect_model_log_posterior_zero(self, delta_psf, mock_emulator, pixel_map_model):
        """log_posterior = 0 when model exactly matches observation.

        obs.flux = 10.0, model = 10.0 (mock emulator at theta=0).
        chi2 = 0 → log_like = 0. Uniform theta → log_prior = 0.
        """
        obs = _flat_obs(10.0, BANDS)
        fm = ForwardModel.build(
            obs=obs,
            psf_model=delta_psf,
            spatial_model=pixel_map_model,
            emulator=mock_emulator,
        )
        theta = jnp.zeros(pixel_map_model.n_params)
        lp = float(fm.log_posterior(theta))
        assert lp == pytest.approx(0.0, abs=1e-3)

    def test_offset_model_log_posterior_known_value(
        self, delta_psf, mock_emulator, pixel_map_model
    ):
        """log_posterior = −384 when model is uniformly 1 nJy below observation.

        obs.flux = 11.0, model = 10.0, variance = 1.0.
        log_like = −0.5 × N_bands × H × W = −0.5 × 3 × 16 × 16 = −384.
        log_prior = 0 (flat theta → zero gradient penalty).
        """
        obs = _flat_obs(11.0, BANDS)
        fm = ForwardModel.build(
            obs=obs,
            psf_model=delta_psf,
            spatial_model=pixel_map_model,
            emulator=mock_emulator,
        )
        theta = jnp.zeros(pixel_map_model.n_params)
        lp = float(fm.log_posterior(theta))
        assert lp == pytest.approx(-384.0, rel=1e-4)


# ---------------------------------------------------------------------------
# PSF padding default and nuisance plumbing
# ---------------------------------------------------------------------------


class TestPadPSFDefault:
    """``build(pad_psf=...)`` controls the PSF convolver's padding."""

    def test_pad_psf_defaults_to_true(
        self, tiny_observation, gaussian_psf, mock_emulator, gmm_model
    ):
        """The linear (zero-padded) convolution is the default."""
        fm = ForwardModel.build(
            obs=tiny_observation,
            psf_model=gaussian_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
        )
        assert fm.convolver.pad is True
        assert fm.convolver.padded_shape != fm.convolver.image_shape

    def test_pad_psf_false_uses_circular(
        self, tiny_observation, gaussian_psf, mock_emulator, gmm_model
    ):
        """pad_psf=False restores the circular convolution on the image grid."""
        fm = ForwardModel.build(
            obs=tiny_observation,
            psf_model=gaussian_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
            pad_psf=False,
        )
        assert fm.convolver.pad is False
        assert fm.convolver.padded_shape == (16, 16)

    def test_padding_changes_edge_pixels(
        self, tiny_observation, gaussian_psf, mock_emulator, gmm_model
    ):
        """Padding changes the model near the frame edge but not far inside it.

        This documents the numerical change for existing users: a flat model
        loses flux to the frame edge under the linear convolution instead of
        receiving it back by wrap-around.
        """
        theta = jnp.zeros(gmm_model.n_params)
        kwargs = dict(
            obs=tiny_observation,
            psf_model=gaussian_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
        )
        padded = ForwardModel.build(**kwargs, pad_psf=True)._model_image(theta)
        circular = ForwardModel.build(**kwargs, pad_psf=False)._model_image(theta)
        assert float(jnp.max(jnp.abs(padded[:, 0, :] - circular[:, 0, :]))) > 1e-3
        assert float(jnp.max(jnp.abs(padded[:, 8, 8] - circular[:, 8, 8]))) < 1e-3

    def test_delta_psf_unaffected_by_padding(
        self, tiny_observation, delta_psf, mock_emulator, gmm_model
    ):
        """With a delta PSF the padded and circular results are identical."""
        theta = jnp.zeros(gmm_model.n_params)
        kwargs = dict(
            obs=tiny_observation,
            psf_model=delta_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
        )
        padded = ForwardModel.build(**kwargs, pad_psf=True)._model_image(theta)
        circular = ForwardModel.build(**kwargs, pad_psf=False)._model_image(theta)
        assert float(jnp.max(jnp.abs(padded - circular))) < 1e-4


class TestForwardModelParameterVector:
    """``n_params`` / ``split_theta`` with and without a NuisanceModel."""

    def test_n_params_without_nuisance(self, forward_model_gmm, gmm_model):
        """Without nuisance, n_params is the spatial model's count."""
        assert forward_model_gmm.nuisance is None
        assert forward_model_gmm.n_params == gmm_model.n_params

    def test_split_theta_without_nuisance(self, forward_model_gmm, gmm_model):
        """The nuisance slice is empty when no NuisanceModel is attached."""
        theta = jnp.arange(gmm_model.n_params, dtype=jnp.float32)
        theta_s, theta_n = forward_model_gmm.split_theta(theta)
        assert theta_n.shape == (0,)
        assert jnp.allclose(theta_s, theta)

    def test_n_params_with_nuisance(self, tiny_observation, gaussian_psf, mock_emulator, gmm_model):
        """n_params adds the nuisance block."""
        nuisance = NuisanceModel(3, fit_sky=True, fit_noise_scale=True)
        fm = ForwardModel.build(
            obs=tiny_observation,
            psf_model=gaussian_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
            nuisance=nuisance,
        )
        assert fm.n_params == gmm_model.n_params + 6
        theta = jnp.arange(fm.n_params, dtype=jnp.float32)
        theta_s, theta_n = fm.split_theta(theta)
        assert theta_s.shape == (gmm_model.n_params,)
        assert theta_n.shape == (6,)

    def test_log_posterior_equals_sum_with_nuisance(
        self, tiny_observation, gaussian_psf, mock_emulator, pixel_map_model
    ):
        """log_posterior == log_likelihood + log_prior with nuisance attached.

        Uses FreeFormPixelMap so the decode-sharing fast path is exercised.
        """
        nuisance = NuisanceModel(3, fit_sky=True, fit_shifts=True, fit_noise_scale=True)
        fm = ForwardModel.build(
            obs=tiny_observation,
            psf_model=gaussian_psf,
            spatial_model=pixel_map_model,
            emulator=mock_emulator,
            nuisance=nuisance,
        )
        rng = np.random.default_rng(4)
        theta = jnp.asarray(rng.normal(0.0, 0.1, fm.n_params).astype(np.float32))
        lp = float(fm.log_posterior(theta))
        expected = float(fm.log_likelihood(theta)) + float(fm.log_prior(theta))
        assert lp == pytest.approx(expected, rel=1e-5)

    def test_log_posterior_equals_sum_without_nuisance(self, forward_model_gmm, gmm_model):
        """The identity also holds in the nuisance-free case."""
        rng = np.random.default_rng(5)
        theta = jnp.asarray(rng.normal(0.0, 0.1, gmm_model.n_params).astype(np.float32))
        lp = float(forward_model_gmm.log_posterior(theta))
        expected = float(forward_model_gmm.log_likelihood(theta)) + float(
            forward_model_gmm.log_prior(theta)
        )
        assert lp == pytest.approx(expected, rel=1e-5)

    def test_grad_with_nuisance_is_finite(
        self, tiny_observation, gaussian_psf, mock_emulator, pixel_map_model
    ):
        """jax.grad through the fused fast path reaches the nuisance block."""
        nuisance = NuisanceModel(3, fit_sky=True, fit_shifts=True)
        fm = ForwardModel.build(
            obs=tiny_observation,
            psf_model=gaussian_psf,
            spatial_model=pixel_map_model,
            emulator=mock_emulator,
            nuisance=nuisance,
        )
        theta = jnp.zeros(fm.n_params).at[-9:].set(0.05)
        grad = jax.grad(fm.log_posterior)(theta)
        assert grad.shape == theta.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert bool(jnp.any(grad[-9:] != 0.0))
