"""Tests for NUTS sampler and NUTSResult (single- and multi-chain).

Uses a very short run (5 warmup + 10 samples) with the GMM model so CI
finishes quickly on CPU.  The multi-chain tests exercise
``run(..., n_chains, chain_jitter, inverse_mass_matrix)`` and the
``chains`` / ``diagnostics`` / ``summary()`` additions to ``NUTSResult``;
``_jitter_inits`` is tested directly because the post-warmup states no longer
reveal the starting points.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.diagnostics import split_rhat
from arachne.inference.nuts_sampler import NUTSResult, NUTSSampler, _jitter_inits

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

    def test_hdf5_param_names_metadata_roundtrip(self, nuts_result, tmp_path):
        """param_names metadata is saved and decoded correctly from HDF5.

        Regression test for fix 2: the original code passed the list directly
        to np.bytes_(), which encoded the list repr rather than each name
        individually.  The fix encodes each string separately.
        """
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "meta.h5"
        nuts_result.to_hdf5(str(path))

        with h5py.File(str(path), "r") as f:
            raw = f["metadata"].attrs["param_names"]
        decoded = [s.decode("utf-8") if isinstance(s, bytes) else s for s in raw]
        assert decoded == nuts_result.spatial_model.sps_param_names

    def test_get_parameter_map_within_bounds(self, nuts_result, gmm_model):
        """Median parameter map values lie within physical bounds."""
        param_maps = nuts_result.get_parameter_map(image_shape=(16, 16), percentiles=[50])
        bounds = gmm_model.param_bounds
        for i, name in enumerate(gmm_model.sps_param_names):
            lo, hi = bounds[name]
            median_map = param_maps[name][0]  # percentile index 0 = 50th
            assert jnp.all(median_map >= lo - 1e-2), f"{name} below lower bound"
            assert jnp.all(median_map <= hi + 1e-2), f"{name} above upper bound"


# ---------------------------------------------------------------------------
# Multi-chain NUTS
# ---------------------------------------------------------------------------

N_CHAINS = 3
DIAGNOSTIC_KEYS = {
    "rhat",
    "ess",
    "n_divergent",
    "mean_tree_depth",
    "step_size",
    "acceptance_rate",
    "fraction_dims_moved",
    "warmup_per_chain",
    "max_num_doublings",
    "dense_mass_matrix",
}


@pytest.fixture(scope="module")
def _multi_cache():
    """Module-level cache so the 3-chain run is compiled and executed once."""
    return {}


@pytest.fixture
def multi_result(sampler, gmm_model, _multi_cache):
    """A 3-chain run with geometry-aware jitter, shared across tests."""
    if "result" not in _multi_cache:
        _multi_cache["result"] = sampler.run(
            jnp.zeros(gmm_model.n_params),
            jax.random.PRNGKey(3),
            n_chains=N_CHAINS,
            chain_jitter=0.05,
        )
    return _multi_cache["result"]


class TestJitterInits:
    """_jitter_inits: replication, geometry awareness and shape handling."""

    def test_chain_zero_is_unjittered(self):
        """Chain 0 starts exactly at theta_init so it matches a 1-chain run."""
        theta = jnp.arange(4, dtype=jnp.float32)
        inits = _jitter_inits(theta, 4, 0.5, None, jax.random.PRNGKey(0))
        assert inits.shape == (4, 4)
        np.testing.assert_allclose(np.asarray(inits[0]), np.asarray(theta))
        assert not np.allclose(np.asarray(inits[1]), np.asarray(theta))

    def test_zero_jitter_replicates(self):
        """chain_jitter=0 gives identical starting points."""
        theta = jnp.arange(3, dtype=jnp.float32)
        inits = _jitter_inits(theta, 5, 0.0, None, jax.random.PRNGKey(0))
        assert inits.shape == (5, 3)
        for row in np.asarray(inits):
            np.testing.assert_allclose(row, np.asarray(theta))

    def test_single_chain_never_jittered(self):
        """n_chains=1 returns theta_init unchanged whatever the jitter."""
        theta = jnp.ones(3)
        inits = _jitter_inits(theta, 1, 10.0, None, jax.random.PRNGKey(0))
        np.testing.assert_allclose(np.asarray(inits), np.asarray(theta)[None, :])

    def test_jitter_scales_with_mass_matrix(self):
        """The spread per dimension follows sqrt(diag(inverse_mass_matrix))."""
        d = 3
        imm = jnp.array([1e-4, 1.0, 1e4])
        inits = np.asarray(_jitter_inits(jnp.zeros(d), 4000, 1.0, imm, jax.random.PRNGKey(1)))
        spread = inits[1:].std(axis=0)
        expected = np.sqrt(np.asarray(imm))
        np.testing.assert_allclose(spread / expected, 1.0, rtol=0.1)

    def test_dense_mass_matrix_uses_diagonal(self):
        """A dense inverse mass matrix is reduced to its diagonal."""
        imm = jnp.diag(jnp.array([1e-4, 1.0, 1e4]))
        inits = np.asarray(_jitter_inits(jnp.zeros(3), 4000, 1.0, imm, jax.random.PRNGKey(2)))
        spread = inits[1:].std(axis=0)
        np.testing.assert_allclose(spread / np.array([1e-2, 1.0, 1e2]), 1.0, rtol=0.1)

    def test_isotropic_without_mass_matrix(self):
        """Without a metric the jitter is isotropic at scale chain_jitter."""
        inits = np.asarray(_jitter_inits(jnp.zeros(3), 4000, 0.25, None, jax.random.PRNGKey(3)))
        np.testing.assert_allclose(inits[1:].std(axis=0), 0.25, rtol=0.1)

    def test_explicit_per_chain_inits_passed_through(self):
        """A (n_chains, d) theta_init is used verbatim (no jitter added)."""
        given = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        inits = _jitter_inits(given, 2, 1.0, None, jax.random.PRNGKey(0))
        np.testing.assert_allclose(np.asarray(inits), np.asarray(given))

    def test_wrong_row_count_raises(self):
        """A (n, d) theta_init with n != n_chains is rejected."""
        with pytest.raises(ValueError, match="rows"):
            _jitter_inits(jnp.zeros((2, 3)), 4, 0.0, None, jax.random.PRNGKey(0))

    def test_bad_ndim_raises(self):
        """A 3-D theta_init is rejected."""
        with pytest.raises(ValueError, match="1-D or 2-D"):
            _jitter_inits(jnp.zeros((2, 3, 4)), 2, 0.0, None, jax.random.PRNGKey(0))


class TestMultiChainRun:
    """NUTSSampler.run(..., n_chains>1)."""

    def test_shapes(self, multi_result, gmm_model):
        """The chains array is (n_chains, n_samples, d) and samples is its flat view."""
        d = gmm_model.n_params
        assert multi_result.chains.shape == (N_CHAINS, 10, d)
        assert multi_result.samples.shape == (N_CHAINS * 10, d)
        assert multi_result.n_chains == N_CHAINS
        assert multi_result.n_samples == N_CHAINS * 10

    def test_samples_are_chains_flattened(self, multi_result):
        """The samples array is exactly chains reshaped, chain-major."""
        np.testing.assert_array_equal(
            np.asarray(multi_result.samples),
            np.asarray(multi_result.chains).reshape(multi_result.samples.shape),
        )

    def test_samples_finite(self, multi_result):
        """No chain produced NaNs."""
        assert jnp.all(jnp.isfinite(multi_result.samples))

    def test_chains_are_distinct(self, multi_result):
        """Independent keys give genuinely different chains."""
        c = np.asarray(multi_result.chains)
        assert not np.allclose(c[0], c[1])

    def test_diagnostics_keys(self, multi_result):
        """All documented diagnostics keys are present."""
        assert set(multi_result.diagnostics) == DIAGNOSTIC_KEYS

    def test_diagnostics_values(self, multi_result, gmm_model):
        """Per-parameter diagnostics are shaped (d,) and scalars are sane."""
        d = multi_result.diagnostics
        assert d["rhat"].shape == (gmm_model.n_params,)
        assert d["ess"].shape == (gmm_model.n_params,)
        assert d["step_size"].shape == (N_CHAINS,)
        assert np.all(d["step_size"] > 0.0)
        assert d["n_divergent"] >= 0
        assert d["mean_tree_depth"] >= 1.0
        assert 0.0 <= d["acceptance_rate"] <= 1.0
        assert 0.0 <= d["fraction_dims_moved"] <= 1.0
        assert d["max_num_doublings"] == 3

    def test_warmup_is_per_chain(self, multi_result):
        """The vmapped window_adaptation path is the one that ran."""
        assert multi_result.diagnostics["warmup_per_chain"] is True
        # Per-chain adaptation means per-chain step sizes.
        assert len(set(np.asarray(multi_result.diagnostics["step_size"]).tolist())) > 1

    def test_rhat_matches_diagnostics_module(self, multi_result):
        """diagnostics["rhat"] is exactly split_rhat(chains)."""
        np.testing.assert_allclose(
            multi_result.diagnostics["rhat"], split_rhat(np.asarray(multi_result.chains))
        )

    def test_explicit_per_chain_theta_init(self, sampler, gmm_model):
        """A (n_chains, d) theta_init is accepted end to end."""
        inits = jax.random.normal(jax.random.PRNGKey(9), (2, gmm_model.n_params)) * 0.01
        result = sampler.run(inits, jax.random.PRNGKey(4), n_chains=2)
        assert result.chains.shape == (2, 10, gmm_model.n_params)

    def test_inverse_mass_matrix_seeds_warmup(self, sampler, gmm_model):
        """A supplied diagonal inverse mass matrix is accepted by warmup."""
        imm = jnp.full((gmm_model.n_params,), 0.25)
        result = sampler.run(
            jnp.zeros(gmm_model.n_params),
            jax.random.PRNGKey(5),
            n_chains=2,
            chain_jitter=0.1,
            inverse_mass_matrix=imm,
        )
        assert result.chains.shape == (2, 10, gmm_model.n_params)
        assert jnp.all(jnp.isfinite(result.samples))

    def test_bad_n_chains_raises(self, sampler, gmm_model):
        """n_chains < 1 is a programming error."""
        with pytest.raises(ValueError, match="n_chains"):
            sampler.run(jnp.zeros(gmm_model.n_params), jax.random.PRNGKey(0), n_chains=0)

    def test_single_chain_keeps_flat_info_layout(self, nuts_result):
        """A 1-chain run still exposes (n_samples,) info leaves and chains (1, n, d)."""
        assert nuts_result.chains.shape == (1, 10, nuts_result.samples.shape[1])
        assert nuts_result.infos.acceptance_rate.shape == (10,)
        assert set(nuts_result.diagnostics) == DIAGNOSTIC_KEYS


class TestDenseMassMatrix:
    """``NUTSSampler(dense_mass_matrix=True)`` adapts a full metric."""

    def test_dense_warmup_runs(self, forward_model, gmm_model):
        """A dense-metric run completes and records the flag in diagnostics."""
        sampler = NUTSSampler(
            forward_model=forward_model,
            n_warmup=20,
            n_samples=10,
            max_num_doublings=3,
            dense_mass_matrix=True,
        )
        result = sampler.run(
            jnp.zeros(gmm_model.n_params), jax.random.PRNGKey(7), n_chains=2, chain_jitter=0.05
        )
        assert result.diagnostics["dense_mass_matrix"] is True
        assert result.chains.shape == (2, 10, gmm_model.n_params)
        assert np.all(np.isfinite(np.asarray(result.samples)))

    def test_default_is_diagonal(self, sampler):
        """The default keeps the historical diagonal metric."""
        assert sampler.dense_mass_matrix is False


class TestSummary:
    """NUTSResult.summary()."""

    def test_mentions_key_numbers(self, multi_result):
        """The paragraph reports chains, R-hat, ESS, divergences and step size."""
        text = multi_result.summary()
        assert isinstance(text, str)
        for token in (
            f"{N_CHAINS} chain(s)",
            "split-R-hat",
            "bulk ESS",
            "divergent",
            "trajectory length",
            "step size",
            "dimensions moved",
        ):
            assert token in text, token

    def test_flags_unconverged_chains(self, multi_result):
        """A deliberately bad R-hat triggers the WARNING suffix."""
        d = dict(multi_result.diagnostics)
        d["rhat"] = np.array([1.9, 1.0, 1.0])
        bad = NUTSResult(
            samples=multi_result.samples,
            infos=None,
            spatial_model=multi_result.spatial_model,
            chains=multi_result.chains,
            diagnostics=d,
        )
        text = bad.summary()
        assert "WARNING" in text and "NOT converged" in text

    def test_flags_tree_cap_saturation(self, multi_result):
        """A mean trajectory length at 2**max_num_doublings - 1 is flagged."""
        d = dict(multi_result.diagnostics)
        d["mean_tree_depth"] = 2**3 - 1.0
        d["rhat"] = np.array([1.0, 1.0, 1.0])
        bad = NUTSResult(
            samples=multi_result.samples,
            infos=None,
            spatial_model=multi_result.spatial_model,
            chains=multi_result.chains,
            diagnostics=d,
        )
        assert "tree cap" in bad.summary(max_num_doublings=3)

    def test_flags_divergences(self, multi_result):
        """Divergent transitions are reported in the warning."""
        d = dict(multi_result.diagnostics)
        d["n_divergent"] = 7
        d["rhat"] = np.array([1.0, 1.0, 1.0])
        bad = NUTSResult(
            samples=multi_result.samples,
            infos=None,
            spatial_model=multi_result.spatial_model,
            chains=multi_result.chains,
            diagnostics=d,
        )
        assert "7 divergent transitions" in bad.summary()

    def test_empty_diagnostics_does_not_crash(self, multi_result):
        """summary() is safe on a result loaded without diagnostics."""
        bare = NUTSResult(
            samples=multi_result.samples,
            infos=None,
            spatial_model=multi_result.spatial_model,
        )
        assert "nan" in bare.summary().lower()


class TestMultiChainHDF5:
    """chains + diagnostics survive the HDF5 round trip."""

    def test_roundtrip(self, multi_result, gmm_model, tmp_path):
        """The chains array and the scalar/vector diagnostics come back unchanged."""
        path = tmp_path / "multi.h5"
        multi_result.to_hdf5(path)
        loaded = NUTSResult.from_hdf5(path, gmm_model)
        assert loaded.n_chains == N_CHAINS
        np.testing.assert_allclose(
            np.asarray(loaded.chains), np.asarray(multi_result.chains), rtol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(loaded.samples), np.asarray(multi_result.samples), rtol=1e-6
        )
        assert set(loaded.diagnostics) == DIAGNOSTIC_KEYS
        for key in ("rhat", "ess", "step_size", "mean_tree_depth", "n_divergent"):
            np.testing.assert_allclose(
                np.asarray(loaded.diagnostics[key], dtype=float),
                np.asarray(multi_result.diagnostics[key], dtype=float),
                rtol=1e-6,
            )

    def test_loaded_result_summarises(self, multi_result, gmm_model, tmp_path):
        """A reloaded result can still print its own summary."""
        path = tmp_path / "multi2.h5"
        multi_result.to_hdf5(path)
        loaded = NUTSResult.from_hdf5(path, gmm_model)
        assert "split-R-hat" in loaded.summary()


def test_package_exports():
    """Diagnostics helpers are re-exported at package level."""
    import arachne
    from arachne.inference.diagnostics import ess, summarise_chains

    assert arachne.split_rhat is split_rhat
    assert arachne.ess is ess
    assert arachne.summarise_chains is summarise_chains
    for name in ("split_rhat", "ess", "summarise_chains"):
        assert name in arachne.__all__
