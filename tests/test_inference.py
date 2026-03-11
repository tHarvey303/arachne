"""Tests for NUTS sampler and NUTSResult.

Uses a very short run (5 warmup + 10 samples) with the GMM model so CI
finishes quickly on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.nuts_sampler import NUTSResult, NUTSSampler

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
    """NUTSSampler configured for a very short run."""
    return NUTSSampler(
        forward_model=forward_model,
        n_warmup=5,
        n_samples=10,
        target_accept_rate=0.8,
        max_num_doublings=3,
    )


@pytest.fixture
def nuts_result(sampler, gmm_model):
    """Run sampler and return NUTSResult."""
    rng_key = jax.random.PRNGKey(42)
    theta_init = jnp.zeros(gmm_model.n_params)
    return sampler.run(theta_init, rng_key)


class TestNUTSSampler:
    """Tests for NUTSSampler.run()."""

    def test_samples_shape(self, nuts_result, gmm_model):
        """Samples array has shape (n_samples, n_params)."""
        assert nuts_result.samples.shape == (10, gmm_model.n_params)

    def test_samples_finite(self, nuts_result):
        """All sample values are finite."""
        assert jnp.all(jnp.isfinite(nuts_result.samples))

    def test_n_samples_property(self, nuts_result):
        """n_samples property is correct."""
        assert nuts_result.n_samples == 10

    def test_acceptance_rate_in_range(self, nuts_result):
        """Mean acceptance rate is in [0, 1]."""
        rate = nuts_result.acceptance_rate
        if rate is not None:
            assert 0.0 <= rate <= 1.0

    def test_spatial_model_stored(self, nuts_result, gmm_model):
        """NUTSResult stores the spatial model reference."""
        assert nuts_result.spatial_model is gmm_model


class TestNUTSResult:
    """Tests for NUTSResult serialisation and analysis."""

    def test_hdf5_roundtrip(self, nuts_result, gmm_model, tmp_path):
        """NUTSResult saves and loads from HDF5 correctly."""
        path = tmp_path / "result.h5"
        nuts_result.to_hdf5(str(path))
        assert path.exists()

        loaded = NUTSResult.from_hdf5(str(path), gmm_model)
        np.testing.assert_allclose(
            np.asarray(loaded.samples),
            np.asarray(nuts_result.samples),
            rtol=1e-5,
        )

    def test_hdf5_creates_parent_dirs(self, nuts_result, tmp_path):
        """to_hdf5() creates parent directories if they do not exist."""
        path = tmp_path / "subdir" / "result.h5"
        nuts_result.to_hdf5(str(path))
        assert path.exists()

    def test_get_parameter_map_shapes(self, nuts_result, gmm_model):
        """get_parameter_map() returns (n_percentiles, H, W) per parameter."""
        param_maps = nuts_result.get_parameter_map(image_shape=(16, 16), percentiles=[16, 50, 84])
        assert len(param_maps) == len(gmm_model.sps_param_names)
        for name, arr in param_maps.items():
            assert arr.shape == (3, 16, 16), f"Wrong shape for {name}: {arr.shape}"

    def test_get_parameter_map_finite(self, nuts_result, gmm_model):
        """get_parameter_map() returns finite values."""
        param_maps = nuts_result.get_parameter_map(image_shape=(16, 16))
        for name, arr in param_maps.items():
            assert jnp.all(jnp.isfinite(arr)), f"Non-finite values in {name}"

    def test_get_parameter_map_within_bounds(self, nuts_result, gmm_model):
        """Median parameter map values lie within physical bounds."""
        param_maps = nuts_result.get_parameter_map(image_shape=(16, 16), percentiles=[50])
        bounds = gmm_model.param_bounds
        for i, name in enumerate(gmm_model.sps_param_names):
            lo, hi = bounds[name]
            median_map = param_maps[name][0]  # percentile index 0 = 50th
            assert jnp.all(median_map >= lo - 1e-2), f"{name} below lower bound"
            assert jnp.all(median_map <= hi + 1e-2), f"{name} above upper bound"
