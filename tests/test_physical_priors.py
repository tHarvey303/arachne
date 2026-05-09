"""Tests for IndependentUniformPrior and LogNormalPrior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.priors.physical import IndependentUniformPrior, LogNormalPrior


class TestIndependentUniformPrior:
    @pytest.fixture
    def prior(self):
        return IndependentUniformPrior(
            lows=jnp.array([0.0, -1.0, 5.0]),
            highs=jnp.array([1.0, 1.0, 10.0]),
        )

    def test_zero_inside_bounds(self, prior):
        """log_prob = 0 for parameters strictly inside bounds."""
        assert float(prior.log_prob(jnp.array([0.5, 0.0, 7.5]))) == pytest.approx(0.0)

    def test_zero_at_boundary(self, prior):
        """log_prob = 0 exactly at the boundary (boundary is included)."""
        assert float(prior.log_prob(jnp.array([0.0, -1.0, 10.0]))) == pytest.approx(0.0)

    def test_negative_below_lower_bound(self, prior):
        """log_prob < 0 when a parameter is below its lower bound."""
        assert float(prior.log_prob(jnp.array([-1.0, 0.0, 7.5]))) < 0.0

    def test_negative_above_upper_bound(self, prior):
        """log_prob < 0 when a parameter is above its upper bound."""
        assert float(prior.log_prob(jnp.array([0.5, 2.0, 7.5]))) < 0.0

    def test_penalty_grows_with_violation_magnitude(self, prior):
        """Larger violation produces a more negative log_prob."""
        small_violation = jnp.array([-0.1, 0.0, 7.5])
        large_violation = jnp.array([-2.0, 0.0, 7.5])
        assert float(prior.log_prob(large_violation)) < float(prior.log_prob(small_violation))

    # --- Regression test for fix 6 (old code returned -inf, killing HMC gradients) ---

    def test_gradient_finite_inside_bounds(self, prior):
        """jax.grad is finite for in-bounds parameters."""
        grad = jax.grad(prior.log_prob)(jnp.array([0.5, 0.0, 7.5]))
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_finite_outside_bounds(self, prior):
        """jax.grad is finite for out-of-bounds parameters (no -inf / NaN).

        Regression test: the original implementation returned ``-jnp.inf``
        outside bounds, which makes ``jax.grad`` return NaN everywhere and
        prevents HMC-family samplers from recovering feasibility.
        """
        grad = jax.grad(prior.log_prob)(jnp.array([-5.0, 5.0, 100.0]))
        assert jnp.all(jnp.isfinite(grad)), (
            "Gradient must be finite outside bounds so HMC can step back in-bounds. "
            "Check that the implementation uses a quadratic barrier, not -inf."
        )

    def test_gradient_points_inward_below_lower_bound(self, prior):
        """Gradient pushes a violated parameter back toward its lower bound."""
        # params[0] = -1.0 < lower bound of 0.0 → gradient should be positive (push up)
        grad = jax.grad(prior.log_prob)(jnp.array([-1.0, 0.0, 7.5]))
        assert float(grad[0]) > 0.0

    def test_gradient_points_inward_above_upper_bound(self, prior):
        """Gradient pushes a violated parameter back toward its upper bound."""
        # params[1] = 2.0 > upper bound of 1.0 → gradient should be negative (push down)
        grad = jax.grad(prior.log_prob)(jnp.array([0.5, 2.0, 7.5]))
        assert float(grad[1]) < 0.0

    def test_jit_compatible(self, prior):
        """Prior survives jax.jit compilation."""
        result = jax.jit(prior.log_prob)(jnp.array([0.5, 0.0, 7.5]))
        assert float(result) == pytest.approx(0.0)


class TestLogNormalPrior:
    @pytest.fixture
    def prior(self):
        return LogNormalPrior(
            mu=jnp.array([9.0, 8.5]),
            sigma=jnp.array([1.0, 0.5]),
        )

    def test_zero_at_mean(self, prior):
        """log_prob = 0 when log_params equal the means (all z-scores are 0)."""
        assert float(prior.log_prob(jnp.array([9.0, 8.5]))) == pytest.approx(0.0)

    def test_known_value_one_sigma_off(self, prior):
        """Verify a non-trivial known value: 1σ deviation on one parameter.

        z = (10.0 − 9.0) / 1.0 = 1  → log_prob = −0.5 × 1² = −0.5
        """
        assert float(prior.log_prob(jnp.array([10.0, 8.5]))) == pytest.approx(-0.5, rel=1e-5)

    def test_maximum_is_at_mean(self, prior):
        """log_prob is maximised at the mean (any deviation reduces it)."""
        at_mean = float(prior.log_prob(jnp.array([9.0, 8.5])))
        off_mean = float(prior.log_prob(jnp.array([10.0, 9.0])))
        assert at_mean > off_mean

    def test_symmetric_around_mean(self, prior):
        """Equal-magnitude deviations above and below the mean give equal log_prob."""
        above = float(prior.log_prob(jnp.array([10.0, 8.5])))
        below = float(prior.log_prob(jnp.array([8.0, 8.5])))
        assert above == pytest.approx(below, rel=1e-5)

    def test_tighter_sigma_penalises_more(self, prior):
        """A tighter sigma gives a more negative log_prob for the same deviation."""
        tight = LogNormalPrior(mu=jnp.array([9.0]), sigma=jnp.array([0.1]))
        loose = LogNormalPrior(mu=jnp.array([9.0]), sigma=jnp.array([2.0]))
        params = jnp.array([9.5])
        assert float(tight.log_prob(params)) < float(loose.log_prob(params))

    def test_differentiable(self, prior):
        """jax.grad passes through LogNormalPrior."""
        grad = jax.grad(prior.log_prob)(jnp.array([9.5, 9.0]))
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)

    def test_gradient_direction(self, prior):
        """Gradient pushes log_params toward the mean."""
        # log_params[0] = 10.0 > mean 9.0 → gradient should be negative (pull down)
        grad = jax.grad(prior.log_prob)(jnp.array([10.0, 8.5]))
        assert float(grad[0]) < 0.0

    def test_jit_compatible(self, prior):
        """Prior survives jax.jit compilation."""
        result = jax.jit(prior.log_prob)(jnp.array([9.0, 8.5]))
        assert float(result) == pytest.approx(0.0)
