"""Tests for ForwardModel pipeline.

The critical tests are:
- test_grad_log_posterior: jax.grad(log_posterior) is finite and non-zero
- test_jit_compiles: jax.jit(log_posterior) runs without error
"""

import jax
import jax.numpy as jnp
import pytest

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
