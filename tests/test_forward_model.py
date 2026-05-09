"""Tests for ForwardModel pipeline.

The critical tests are:
- test_grad_log_posterior: jax.grad(log_posterior) is finite and non-zero
- test_jit_compiles: jax.jit(log_posterior) runs without error
"""

import jax
import jax.numpy as jnp
import pytest

from arachne.data.observation import ObservationCube
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

    def test_perfect_model_log_posterior_zero(
        self, delta_psf, mock_emulator, pixel_map_model
    ):
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
