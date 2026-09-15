"""Tests for GaussianLikelihood."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.data.observation import ObservationCube
from arachne.likelihood.gaussian import GaussianLikelihood


def _obs(flux, variance=None, mask=None):
    """Build a JAX ObservationCube from array arguments."""
    flux = jnp.asarray(flux, dtype=jnp.float32)
    if variance is None:
        variance = jnp.ones_like(flux)
    if mask is None:
        mask = jnp.ones_like(flux)
    n_bands = flux.shape[0]
    return ObservationCube(
        flux=flux,
        variance=jnp.asarray(variance, dtype=jnp.float32),
        mask=jnp.asarray(mask, dtype=jnp.float32),
        band_names=[f"band_{i}" for i in range(n_bands)],
        pixel_scale=0.031,
    )


class TestGaussianLikelihood:
    """Tests for GaussianLikelihood."""

    def test_returns_scalar(self):
        """log_likelihood returns a shape-() scalar."""
        obs = _obs(jnp.ones((2, 8, 8)))
        result = GaussianLikelihood(obs)(jnp.ones((2, 8, 8)))
        assert result.shape == ()

    def test_perfect_model_gives_zero(self):
        """When model exactly matches data the log-likelihood is 0."""
        flux = jnp.full((1, 4, 4), 5.0)
        obs = _obs(flux)
        assert float(GaussianLikelihood(obs)(flux)) == pytest.approx(0.0)

    def test_known_value(self):
        """Verify against a manually computed chi-squared.

        1-band 2×2 image: flux [[1,2],[3,4]], model all-ones, variance 1.
        residuals = [[0,1],[2,3]]  → chi2 = [[0,1],[4,9]]  → sum = 14
        log_like = −0.5 × 14 = −7.
        """
        flux = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
        model = jnp.ones((1, 2, 2))
        obs = _obs(flux)
        assert float(GaussianLikelihood(obs)(model)) == pytest.approx(-7.0, rel=1e-5)

    def test_all_masked_out_gives_zero(self):
        """When every pixel is masked the likelihood is 0 regardless of residuals."""
        flux = jnp.ones((1, 4, 4))
        model = jnp.full((1, 4, 4), 1000.0)  # wildly wrong
        mask = jnp.zeros((1, 4, 4))
        obs = _obs(flux, mask=mask)
        assert float(GaussianLikelihood(obs)(model)) == pytest.approx(0.0)

    def test_partial_mask_excludes_flagged_pixels(self):
        """Masked pixels are excluded; unmasked pixels contribute normally.

        2×2 image, mask out pixel (0,1).  Unmasked residuals: 0, 4, 9 → sum=13.
        """
        flux = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
        model = jnp.ones((1, 2, 2))
        mask = jnp.array([[[1.0, 0.0], [1.0, 1.0]]])
        obs = _obs(flux, mask=mask)
        assert float(GaussianLikelihood(obs)(model)) == pytest.approx(-0.5 * 13.0, rel=1e-5)

    def test_higher_variance_reduces_chi_squared(self):
        """Larger variance produces a less negative log-likelihood for the same residual."""
        flux = jnp.array([[[2.0]]])
        model = jnp.zeros((1, 1, 1))
        obs_tight = _obs(flux, variance=jnp.full((1, 1, 1), 0.1))
        obs_loose = _obs(flux, variance=jnp.full((1, 1, 1), 10.0))
        ll_tight = float(GaussianLikelihood(obs_tight)(model))
        ll_loose = float(GaussianLikelihood(obs_loose)(model))
        assert ll_tight < ll_loose

    def test_multi_band_sums_over_bands(self):
        """Log-likelihood sums chi-squared contributions across all bands.

        Band 0: model = flux (residual 0).
        Band 1: model = flux − 1 everywhere (residual 1 per pixel).
        H×W = 4×4 → chi2 from band 1 = 16 → log_like = −8.
        """
        H, W = 4, 4
        flux = jnp.stack([jnp.ones((H, W)), jnp.full((H, W), 2.0)])
        model = jnp.ones((2, H, W))  # band 1 residual = 1
        obs = _obs(flux)
        expected = -0.5 * H * W  # band 0 contributes 0, band 1 contributes H*W
        assert float(GaussianLikelihood(obs)(model)) == pytest.approx(expected, rel=1e-5)

    def test_differentiable(self):
        """jax.grad passes through GaussianLikelihood."""
        flux = jnp.full((2, 4, 4), 3.0)
        obs = _obs(flux)
        grad = jax.grad(GaussianLikelihood(obs))(jnp.zeros((2, 4, 4)))
        assert grad.shape == (2, 4, 4)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)

    def test_gradient_zero_at_perfect_model(self):
        """Gradient of log-likelihood is zero when model = data (minimum of chi-squared)."""
        flux = jnp.full((1, 4, 4), 5.0)
        obs = _obs(flux)
        grad = jax.grad(GaussianLikelihood(obs))(flux)
        assert jnp.allclose(grad, 0.0, atol=1e-6)

    def test_gradient_direction(self):
        """Gradient points from model toward data (negative residual → positive gradient)."""
        flux = jnp.array([[[3.0]]])  # data
        model = jnp.array([[[1.0]]])  # model < data → residual negative
        obs = _obs(flux)
        grad = jax.grad(GaussianLikelihood(obs))(model)
        # d/d(model) of  −0.5*(data−model)²/var = (data−model)/var > 0
        assert float(grad[0, 0, 0]) > 0.0


class TestModelErrorFloor:
    """Tests for the ``model_error_frac`` fractional model-error floor."""

    @staticmethod
    def _numpy_reference(flux, variance, mask, model, frac):
        flux, variance, mask, model = (
            np.asarray(a, dtype=np.float64) for a in (flux, variance, mask, model)
        )
        var_eff = variance + (frac * model) ** 2
        return -0.5 * np.sum(mask * ((flux - model) ** 2 / var_eff + np.log(var_eff)))

    def test_default_frac_is_zero_and_identical_to_plain_chi2(self):
        """frac=0 (the default) reproduces the exact plain chi-squared expression."""
        rng = np.random.default_rng(0)
        flux = rng.normal(10.0, 2.0, size=(3, 5, 5)).astype(np.float32)
        variance = rng.uniform(0.5, 2.0, size=flux.shape).astype(np.float32)
        model = rng.normal(10.0, 2.0, size=flux.shape).astype(np.float32)
        obs = _obs(flux, variance=variance)
        ll_default = float(GaussianLikelihood(obs)(jnp.asarray(model)))
        ll_zero = float(GaussianLikelihood(obs, model_error_frac=0.0)(jnp.asarray(model)))
        expected = -0.5 * np.sum((flux - model) ** 2 / variance)
        assert GaussianLikelihood(obs).model_error_frac == 0.0
        assert ll_default == ll_zero
        assert ll_default == pytest.approx(expected, rel=1e-5)

    def test_negative_frac_raises(self):
        """A negative floor is rejected."""
        with pytest.raises(ValueError):
            GaussianLikelihood(_obs(jnp.ones((1, 2, 2))), model_error_frac=-0.1)

    def test_frac_matches_numpy_reference(self):
        """frac>0 equals the float64 numpy reference including the log-determinant."""
        rng = np.random.default_rng(1)
        flux = rng.normal(50.0, 5.0, size=(2, 6, 6)).astype(np.float32)
        variance = rng.uniform(1.0, 4.0, size=flux.shape).astype(np.float32)
        mask = (rng.uniform(size=flux.shape) > 0.2).astype(np.float32)
        model = rng.normal(50.0, 5.0, size=flux.shape).astype(np.float32)
        frac = 0.05
        obs = _obs(flux, variance=variance, mask=mask)
        ll = float(GaussianLikelihood(obs, model_error_frac=frac)(jnp.asarray(model)))
        ref = self._numpy_reference(flux, variance, mask, model, frac)
        assert np.isfinite(ll)
        assert ll == pytest.approx(ref, rel=1e-5)

    def test_frac_changes_value(self):
        """The floor changes the log-likelihood (it is not silently ignored)."""
        flux = jnp.full((1, 3, 3), 100.0)
        model = jnp.full((1, 3, 3), 90.0)
        obs = _obs(flux, variance=jnp.full((1, 3, 3), 4.0))
        ll0 = float(GaussianLikelihood(obs)(model))
        ll1 = float(GaussianLikelihood(obs, model_error_frac=0.1)(model))
        assert ll0 != ll1

    def test_log_det_term_present(self):
        """For a perfect model, the floor gives -0.5*sum(log var_eff), not zero.

        Without the normalisation term a perfect model would score 0 regardless
        of the floor; the log-determinant is required because var_eff depends on
        the model.
        """
        flux = jnp.full((1, 2, 2), 20.0)
        variance = jnp.full((1, 2, 2), 1.0)
        obs = _obs(flux, variance=variance)
        frac = 0.1
        ll = float(GaussianLikelihood(obs, model_error_frac=frac)(flux))
        expected = -0.5 * 4 * np.log(1.0 + (frac * 20.0) ** 2)
        assert ll == pytest.approx(expected, rel=1e-5)

    def test_frac_differentiable(self):
        """jax.grad passes through the floor branch and matches a finite difference."""
        rng = np.random.default_rng(2)
        flux = rng.normal(30.0, 3.0, size=(2, 4, 4)).astype(np.float32)
        variance = rng.uniform(1.0, 2.0, size=flux.shape).astype(np.float32)
        model = rng.normal(30.0, 3.0, size=flux.shape).astype(np.float32)
        obs = _obs(flux, variance=variance)
        like = GaussianLikelihood(obs, model_error_frac=0.05)
        grad = jax.grad(like)(jnp.asarray(model))
        assert grad.shape == model.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        # Finite-difference check on one element (float64 reference).
        eps = 1e-3
        mp = model.copy()
        mm = model.copy()
        mp[0, 1, 2] += eps
        mm[0, 1, 2] -= eps
        fd = (
            self._numpy_reference(flux, variance, np.ones_like(flux), mp, 0.05)
            - self._numpy_reference(flux, variance, np.ones_like(flux), mm, 0.05)
        ) / (2 * eps)
        assert float(grad[0, 1, 2]) == pytest.approx(fd, rel=2e-2, abs=1e-3)

    def test_frac_jit_compatible(self):
        """The floor branch is a Python-level constant, so jax.jit works."""
        flux = jnp.full((1, 3, 3), 10.0)
        obs = _obs(flux)
        like = GaussianLikelihood(obs, model_error_frac=0.05)
        ll_jit = float(jax.jit(like)(flux * 0.9))
        ll_eager = float(like(flux * 0.9))
        assert ll_jit == pytest.approx(ll_eager, rel=1e-6)
