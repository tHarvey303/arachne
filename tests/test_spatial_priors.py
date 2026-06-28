"""Tests for GradientPenaltyPrior and TotalVariationPrior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.priors.spatial import GradientPenaltyPrior, TotalVariationPrior


def _uniform(H=8, W=8, C=2, val=1.0):
    return jnp.full((H, W, C), val, dtype=jnp.float32)


def _rough(seed=0, H=8, W=8, C=2):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.standard_normal((H, W, C)).astype(np.float32))


class TestGradientPenaltyPrior:
    """Tests for GradientPenaltyPrior."""

    def test_uniform_map_zero_penalty(self):
        """Spatially constant map has zero gradient → log_prob = 0."""
        prior = GradientPenaltyPrior(strength=1.0)
        assert float(prior.log_prob(_uniform())) == pytest.approx(0.0)

    def test_rough_map_negative(self):
        """Spatially varying map has log_prob < 0."""
        prior = GradientPenaltyPrior(strength=1.0)
        assert float(prior.log_prob(_rough())) < 0.0

    def test_always_non_positive(self):
        """log_prob ≤ 0 for any map (penalty can only subtract)."""
        prior = GradientPenaltyPrior(strength=1.0)
        for seed in range(5):
            assert float(prior.log_prob(_rough(seed))) <= 0.0

    def test_strength_scales_penalty_linearly(self):
        """log_prob scales exactly with strength (L2 penalty is linear in λ)."""
        m = _rough()
        lp1 = float(GradientPenaltyPrior(strength=1.0).log_prob(m))
        lp3 = float(GradientPenaltyPrior(strength=3.0).log_prob(m))
        assert lp3 == pytest.approx(3.0 * lp1, rel=1e-5)

    def test_known_value(self):
        """Verify against a hand-computed example.

        2×2, 1-channel map with values [[0,1],[0,1]]:
        - dy (vertical diff): [[0],[0]]  → sum dy² = 0
        - dx (horizontal diff): [[1],[1]] → sum dx² = 2
        penalty = 0 + 2 = 2  →  log_prob = −2
        """
        prior = GradientPenaltyPrior(strength=1.0)
        m = jnp.array([[[0.0], [1.0]], [[0.0], [1.0]]])
        assert float(prior.log_prob(m)) == pytest.approx(-2.0, rel=1e-5)

    def test_differentiable(self):
        """jax.grad passes through GradientPenaltyPrior."""
        prior = GradientPenaltyPrior(strength=1.0)
        grad = jax.grad(prior.log_prob)(_rough())
        assert jnp.all(jnp.isfinite(grad))

    def test_zero_strength_constant_grad(self):
        """With strength = 0 the prior is constant; its gradient is all zeros."""
        prior = GradientPenaltyPrior(strength=0.0)
        grad = jax.grad(prior.log_prob)(_rough())
        assert jnp.allclose(grad, 0.0)

    def test_smoother_map_higher_log_prob(self):
        """A smoother map has a less negative log_prob than a rough one."""
        prior = GradientPenaltyPrior(strength=1.0)
        smooth = _uniform() + jnp.linspace(0, 0.01, 8)[:, None, None]  # tiny gradient
        rough = _rough()
        assert float(prior.log_prob(smooth)) > float(prior.log_prob(rough))


class TestTotalVariationPrior:
    """Tests for TotalVariationPrior."""

    def test_rough_map_negative(self):
        """Spatially varying map has log_prob < 0."""
        prior = TotalVariationPrior(strength=1.0)
        assert float(prior.log_prob(_rough())) < 0.0

    def test_always_non_positive(self):
        """log_prob ≤ 0 for any map."""
        prior = TotalVariationPrior(strength=1.0)
        for seed in range(5):
            assert float(prior.log_prob(_rough(seed))) <= 0.0

    def test_strength_scales_penalty(self):
        """Higher strength gives a more negative log_prob for the same map."""
        m = _rough()
        lp1 = float(TotalVariationPrior(strength=1.0).log_prob(m))
        lp5 = float(TotalVariationPrior(strength=5.0).log_prob(m))
        assert lp5 < lp1

    def test_differentiable_on_rough_map(self):
        """jax.grad passes through TotalVariationPrior on a rough map."""
        prior = TotalVariationPrior(strength=1.0)
        grad = jax.grad(prior.log_prob)(_rough())
        assert jnp.all(jnp.isfinite(grad))

    def test_differentiable_on_uniform_map(self):
        """jax.grad is finite even at a uniform map (gradient = 0 everywhere).

        The smooth-L1 approximation (sqrt(x²+ε)) avoids the kink at zero that
        would otherwise make the gradient undefined for a flat map.
        """
        prior = TotalVariationPrior(strength=1.0)
        grad = jax.grad(prior.log_prob)(_uniform())
        assert jnp.all(jnp.isfinite(grad))

    def test_epsilon_controls_smoothness(self):
        """Larger epsilon raises the log_prob floor on a uniform map."""
        m = _uniform()
        lp_small_eps = float(TotalVariationPrior(strength=1.0, epsilon=1e-12).log_prob(m))
        lp_large_eps = float(TotalVariationPrior(strength=1.0, epsilon=1.0).log_prob(m))
        # With a uniform map all dy=dx=0; TV = sum sqrt(0+ε) = N*sqrt(ε)
        # Larger ε → larger penalty → more negative log_prob
        assert lp_large_eps < lp_small_eps

    def test_jit_compatible(self):
        """TotalVariationPrior survives jax.jit."""
        prior = TotalVariationPrior(strength=1.0)
        result = jax.jit(prior.log_prob)(_rough())
        assert jnp.isfinite(result)
