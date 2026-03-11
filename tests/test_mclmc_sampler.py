"""Tests for MCLMCSampler and run_pathfinder.

Uses a very short warmup (100 steps) and few samples (10) with the GMM
model so CI finishes quickly on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.mclmc_sampler import MCLMCSampler, run_pathfinder
from arachne.inference.nuts_sampler import NUTSResult

blackjax = pytest.importorskip("blackjax", reason="blackjax not installed")


@pytest.fixture
def forward_model(tiny_observation, gaussian_psf, mock_emulator, gmm_model):
    """Small ForwardModel for inference tests."""
    return ForwardModel.build(
        obs=tiny_observation,
        psf_model=gaussian_psf,
        spatial_model=gmm_model,
        emulator=mock_emulator,
    )


@pytest.fixture
def sampler(forward_model):
    """MCLMCSampler configured for a very short run."""
    return MCLMCSampler(
        forward_model=forward_model,
        n_warmup=100,
        n_samples=10,
        diagonal_preconditioning=True,
    )


@pytest.fixture
def mclmc_result(sampler, gmm_model):
    """Run MCLMCSampler and return NUTSResult."""
    rng_key = jax.random.PRNGKey(42)
    theta_init = jnp.zeros(gmm_model.n_params)
    return sampler.run(theta_init, rng_key)


# ---------------------------------------------------------------------------
# MCLMCSampler tests
# ---------------------------------------------------------------------------


class TestMCLMCSampler:
    """Tests for MCLMCSampler.run()."""

    def test_samples_shape(self, mclmc_result, gmm_model):
        """Samples array has shape (n_samples, n_params)."""
        assert mclmc_result.samples.shape == (10, gmm_model.n_params)

    def test_samples_finite(self, mclmc_result):
        """All sample values are finite."""
        assert jnp.all(jnp.isfinite(mclmc_result.samples))

    def test_returns_nuts_result(self, mclmc_result):
        """MCLMCSampler returns a NUTSResult for interface compatibility."""
        assert isinstance(mclmc_result, NUTSResult)

    def test_n_samples_property(self, mclmc_result):
        """n_samples property is correct."""
        assert mclmc_result.n_samples == 10

    def test_spatial_model_stored(self, mclmc_result, gmm_model):
        """NUTSResult stores the spatial model reference."""
        assert mclmc_result.spatial_model is gmm_model

    def test_with_inverse_mass_matrix(self, forward_model, gmm_model):
        """MCLMCSampler accepts an external inverse_mass_matrix."""
        sampler = MCLMCSampler(forward_model=forward_model, n_warmup=100, n_samples=5)
        rng_key = jax.random.PRNGKey(7)
        theta_init = jnp.zeros(gmm_model.n_params)
        imm = jnp.ones(gmm_model.n_params)  # identity (isotropic)
        result = sampler.run(theta_init, rng_key, inverse_mass_matrix=imm)
        assert result.samples.shape == (5, gmm_model.n_params)
        assert jnp.all(jnp.isfinite(result.samples))

    def test_hdf5_roundtrip(self, mclmc_result, gmm_model, tmp_path):
        """NUTSResult from MCLMC saves and loads from HDF5."""
        path = tmp_path / "mclmc_result.h5"
        mclmc_result.to_hdf5(str(path))
        assert path.exists()
        loaded = NUTSResult.from_hdf5(str(path), gmm_model)
        np.testing.assert_allclose(
            np.asarray(loaded.samples),
            np.asarray(mclmc_result.samples),
            rtol=1e-5,
        )

    def test_get_parameter_map_shape(self, mclmc_result, gmm_model):
        """get_parameter_map() returns correct shapes from MCLMC samples."""
        param_maps = mclmc_result.get_parameter_map(image_shape=(16, 16), percentiles=[16, 50, 84])
        assert len(param_maps) == len(gmm_model.sps_param_names)
        for name, arr in param_maps.items():
            assert arr.shape == (3, 16, 16), f"Wrong shape for {name}: {arr.shape}"


# ---------------------------------------------------------------------------
# run_pathfinder tests
# ---------------------------------------------------------------------------


class TestRunPathfinder:
    """Tests for run_pathfinder()."""

    def test_returns_position_and_imm(self, forward_model, gmm_model):
        """run_pathfinder returns (position, inverse_mass_matrix) with correct shapes."""
        rng_key = jax.random.PRNGKey(0)
        theta_init = jnp.zeros(gmm_model.n_params)
        position, imm = run_pathfinder(
            forward_model.log_posterior, theta_init, rng_key, num_samples=50
        )
        assert position.shape == (gmm_model.n_params,)
        assert imm.shape == (gmm_model.n_params,)

    def test_position_finite(self, forward_model, gmm_model):
        """Pathfinder position is finite."""
        rng_key = jax.random.PRNGKey(1)
        theta_init = jnp.zeros(gmm_model.n_params)
        position, _ = run_pathfinder(
            forward_model.log_posterior, theta_init, rng_key, num_samples=50
        )
        assert jnp.all(jnp.isfinite(position))

    def test_imm_positive(self, forward_model, gmm_model):
        """Pathfinder inverse-mass-matrix diagonal is positive."""
        rng_key = jax.random.PRNGKey(2)
        theta_init = jnp.zeros(gmm_model.n_params)
        _, imm = run_pathfinder(forward_model.log_posterior, theta_init, rng_key, num_samples=50)
        assert jnp.all(imm > 0)

    def test_pathfinder_then_mclmc(self, forward_model, gmm_model):
        """Pathfinder output can be passed directly to MCLMCSampler."""
        k1, k2 = jax.random.split(jax.random.PRNGKey(99))
        theta_init = jnp.zeros(gmm_model.n_params)

        position, imm = run_pathfinder(forward_model.log_posterior, theta_init, k1, num_samples=50)
        sampler = MCLMCSampler(forward_model=forward_model, n_warmup=100, n_samples=5)
        result = sampler.run(position, k2, inverse_mass_matrix=imm)

        assert result.samples.shape == (5, gmm_model.n_params)
        assert jnp.all(jnp.isfinite(result.samples))
