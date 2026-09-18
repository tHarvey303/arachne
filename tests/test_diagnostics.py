"""Tests for the MCMC convergence diagnostics (split-R-hat, bulk ESS, summaries).

Pure numpy, no JAX/blackjax needed, so this module runs in well under a second.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne.inference.diagnostics import (
    chain_movement,
    ess,
    split_rhat,
    summarise_chains,
)


def _ar1(rho: float, n_chains: int, n_samples: int, d: int, seed: int = 0) -> np.ndarray:
    """Stationary AR(1) chains with unit marginal variance."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n_chains, n_samples, d)) * np.sqrt(1.0 - rho**2)
    x = np.empty_like(noise)
    x[:, 0] = rng.normal(size=(n_chains, d))
    for t in range(1, n_samples):
        x[:, t] = rho * x[:, t - 1] + noise[:, t]
    return x


class TestSplitRhat:
    """split_rhat on cases with a known answer."""

    def test_identical_chains_give_one(self):
        """Four copies of the same chain have no between-chain variance."""
        rng = np.random.default_rng(0)
        one = rng.normal(size=(1, 2000, 3))
        chains = np.repeat(one, 4, axis=0)
        r = split_rhat(chains)
        assert r.shape == (3,)
        np.testing.assert_allclose(r, 1.0, atol=0.02)

    def test_shifted_means_flagged(self):
        """Chains centred on different means give R-hat well above 1.1."""
        rng = np.random.default_rng(1)
        chains = rng.normal(size=(4, 500, 3)) + np.arange(4)[:, None, None]
        assert np.all(split_rhat(chains) > 1.1)

    def test_well_mixed_chains_near_one(self):
        """Independent draws from the same distribution give R-hat ~ 1."""
        rng = np.random.default_rng(2)
        chains = rng.normal(size=(4, 2000, 4))
        assert np.all(split_rhat(chains) < 1.01)

    def test_within_chain_drift_detected(self):
        """A linear drift within each chain is caught by the split."""
        rng = np.random.default_rng(3)
        drift = np.linspace(0.0, 5.0, 1000)[None, :, None]
        chains = rng.normal(size=(4, 1000, 2)) + drift
        assert np.all(split_rhat(chains) > 1.1)

    def test_single_chain_and_2d_input(self):
        """A 2-D input is treated as one chain and still split in half."""
        rng = np.random.default_rng(4)
        x = rng.normal(size=(1000, 3))
        np.testing.assert_allclose(split_rhat(x), split_rhat(x[None]), rtol=1e-12)

    def test_too_few_samples_is_nan(self):
        """Fewer than four draws cannot be split; result is NaN."""
        assert np.all(np.isnan(split_rhat(np.zeros((2, 3, 5)))))

    def test_constant_parameter_is_nan(self):
        """A frozen dimension has zero within-chain variance -> NaN, not inf."""
        rng = np.random.default_rng(5)
        chains = rng.normal(size=(4, 200, 2))
        chains = np.concatenate([chains, np.zeros((4, 200, 1))], axis=2)
        r = split_rhat(chains)
        assert np.isnan(r[2]) and np.all(np.isfinite(r[:2]))

    def test_bad_shape_raises(self):
        """A 1-D input is rejected."""
        with pytest.raises(ValueError):
            split_rhat(np.zeros(10))


class TestEss:
    """ESS against analytically known autocorrelation times."""

    def test_iid_draws_recover_n(self):
        """i.i.d. normal draws have ESS ~ n_total (within 30%)."""
        rng = np.random.default_rng(10)
        n_chains, n_samples = 4, 1000
        chains = rng.normal(size=(n_chains, n_samples, 5))
        n_total = n_chains * n_samples
        e = ess(chains)
        assert e.shape == (5,)
        assert np.all(np.abs(e - n_total) < 0.3 * n_total)

    def test_ar1_matches_theory(self):
        """AR(1) with rho=0.9 has ESS ~ n (1-rho)/(1+rho) within a factor 1.5."""
        rho = 0.9
        n_chains, n_samples = 4, 4000
        chains = _ar1(rho, n_chains, n_samples, 2, seed=11)
        expected = n_chains * n_samples * (1 - rho) / (1 + rho)
        e = ess(chains)
        assert np.all(e < 1.5 * expected)
        assert np.all(e > expected / 1.5)

    def test_ar1_negative_correlation_gives_superefficiency(self):
        """A rho of -0.5 gives ESS above n_total, as theory requires."""
        chains = _ar1(-0.5, 4, 4000, 1, seed=12)
        assert ess(chains)[0] > 4 * 4000

    def test_more_correlation_lowers_ess(self):
        """ESS decreases monotonically with the autocorrelation."""
        vals = [ess(_ar1(r, 4, 2000, 1, seed=13))[0] for r in (0.0, 0.5, 0.9)]
        assert vals[0] > vals[1] > vals[2]

    def test_without_rank_normalisation_agrees_for_normals(self):
        """Rank normalisation is a no-op (to ~5%) for Gaussian marginals."""
        chains = _ar1(0.7, 4, 2000, 2, seed=14)
        a = ess(chains, rank_normalise=True)
        b = ess(chains, rank_normalise=False)
        np.testing.assert_allclose(a, b, rtol=0.05)

    def test_constant_parameter_is_nan(self):
        """A frozen dimension has no ESS."""
        chains = np.zeros((4, 200, 1))
        assert np.isnan(ess(chains)[0])

    def test_too_few_draws_is_nan(self):
        """Fewer than 8 draws per chain is not enough to estimate ESS."""
        assert np.all(np.isnan(ess(np.zeros((4, 6, 2)))))


class TestChainMovement:
    """chain_movement counts dimensions that actually moved."""

    def test_all_moved(self):
        """Random draws move in every dimension."""
        rng = np.random.default_rng(20)
        assert chain_movement(rng.normal(size=(2, 50, 7))) == 1.0

    def test_none_moved(self):
        """A completely stuck sampler scores 0."""
        assert chain_movement(np.ones((3, 40, 5))) == 0.0

    def test_partial(self):
        """Half-frozen chains give 0.5."""
        rng = np.random.default_rng(21)
        chains = np.concatenate(
            [rng.normal(size=(2, 30, 2)), np.zeros((2, 30, 2))],
            axis=2,
        )
        assert chain_movement(chains) == 0.5

    def test_frozen_in_one_chain_only_still_counts(self):
        """A dimension that moves in any chain counts as moved."""
        chains = np.zeros((2, 30, 1))
        chains[1, 5, 0] = 1.0
        assert chain_movement(chains) == 1.0


class TestSummariseChains:
    """summarise_chains output structure and values."""

    def test_keys_and_shapes(self):
        """All documented keys are present with the right shapes."""
        rng = np.random.default_rng(30)
        chains = rng.normal(size=(4, 500, 3))
        s = summarise_chains(chains)
        for key in (
            "rhat",
            "ess",
            "mean",
            "sd",
            "quantiles",
            "quantile_levels",
            "fraction_dims_moved",
            "max_rhat",
            "min_ess",
        ):
            assert key in s
        assert s["rhat"].shape == (3,)
        assert s["ess"].shape == (3,)
        assert s["mean"].shape == (3,)
        assert s["sd"].shape == (3,)
        assert s["quantiles"].shape == (3, 3)
        assert s["n_chains"] == 4 and s["n_samples"] == 500 and s["n_params"] == 3

    def test_moments_are_right(self):
        """Mean/sd/quantiles match the pooled draws."""
        rng = np.random.default_rng(31)
        chains = rng.normal(loc=2.0, scale=3.0, size=(4, 4000, 2))
        s = summarise_chains(chains)
        np.testing.assert_allclose(s["mean"], 2.0, atol=0.1)
        np.testing.assert_allclose(s["sd"], 3.0, atol=0.1)
        assert np.all(s["quantiles"][0] < s["quantiles"][1])
        assert np.all(s["quantiles"][1] < s["quantiles"][2])

    def test_param_names_give_table(self):
        """Supplying names adds a printable table containing every name."""
        rng = np.random.default_rng(32)
        s = summarise_chains(rng.normal(size=(2, 100, 2)), param_names=["alpha", "beta"])
        assert s["names"] == ["alpha", "beta"]
        assert "alpha" in s["table"] and "beta" in s["table"]
        assert "rhat" in s["table"] and "ess" in s["table"]

    def test_wrong_param_names_length_raises(self):
        """A mismatched name list is an error."""
        with pytest.raises(ValueError):
            summarise_chains(np.zeros((2, 100, 3)), param_names=["a"])

    def test_max_rhat_min_ess_consistent(self):
        """The scalar summaries match the per-parameter arrays."""
        rng = np.random.default_rng(33)
        s = summarise_chains(rng.normal(size=(4, 400, 3)))
        assert s["max_rhat"] == pytest.approx(np.nanmax(s["rhat"]))
        assert s["min_ess"] == pytest.approx(np.nanmin(s["ess"]))


def test_package_exports():
    """Diagnostics are re-exported at package level."""
    import arachne

    assert arachne.split_rhat is split_rhat and "split_rhat" in arachne.__all__
    assert arachne.ess is ess and "ess" in arachne.__all__
    assert arachne.summarise_chains is summarise_chains and "summarise_chains" in arachne.__all__
