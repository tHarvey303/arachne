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


class TestPerBandModelErrorFrac:
    """``model_error_frac`` may be a per-band array of shape (N_bands,)."""

    @staticmethod
    def _numpy_reference(flux, variance, mask, model, frac):
        flux, variance, mask, model = (
            np.asarray(a, dtype=np.float64) for a in (flux, variance, mask, model)
        )
        frac = np.asarray(frac, dtype=np.float64).reshape(-1, 1, 1)
        var_eff = variance + (frac * model) ** 2
        return -0.5 * np.sum(mask * ((flux - model) ** 2 / var_eff + np.log(var_eff)))

    def test_per_band_array_matches_numpy_reference(self):
        """A (N_bands,) floor is broadcast over pixels and matches numpy."""
        rng = np.random.default_rng(21)
        flux = rng.normal(50.0, 5.0, size=(3, 5, 5)).astype(np.float32)
        variance = rng.uniform(1.0, 4.0, size=flux.shape).astype(np.float32)
        model = rng.normal(50.0, 5.0, size=flux.shape).astype(np.float32)
        frac = np.array([0.02, 0.05, 0.10], dtype=np.float32)
        obs = _obs(flux, variance=variance)
        ll = float(GaussianLikelihood(obs, model_error_frac=frac)(jnp.asarray(model)))
        ref = self._numpy_reference(flux, variance, np.ones_like(flux), model, frac)
        assert ll == pytest.approx(ref, rel=1e-5)

    def test_uniform_array_equals_scalar(self):
        """A constant array floor gives the same answer as the scalar version."""
        rng = np.random.default_rng(22)
        flux = rng.normal(20.0, 2.0, size=(2, 4, 4)).astype(np.float32)
        model = rng.normal(20.0, 2.0, size=flux.shape).astype(np.float32)
        obs = _obs(flux)
        ll_scalar = float(GaussianLikelihood(obs, model_error_frac=0.07)(jnp.asarray(model)))
        ll_array = float(
            GaussianLikelihood(obs, model_error_frac=np.full(2, 0.07, np.float32))(
                jnp.asarray(model)
            )
        )
        assert ll_array == pytest.approx(ll_scalar, rel=1e-5)

    def test_wrong_length_raises(self):
        """An array whose length does not match the number of bands is rejected."""
        obs = _obs(jnp.ones((3, 2, 2)))
        with pytest.raises(ValueError, match="bands"):
            GaussianLikelihood(obs, model_error_frac=np.array([0.1, 0.2]))

    def test_negative_entry_raises(self):
        """Any negative entry in the array is rejected."""
        obs = _obs(jnp.ones((2, 2, 2)))
        with pytest.raises(ValueError, match=">= 0"):
            GaussianLikelihood(obs, model_error_frac=np.array([0.1, -0.2]))

    def test_too_many_dims_raises(self):
        """A 2-D floor is rejected."""
        obs = _obs(jnp.ones((2, 2, 2)))
        with pytest.raises(ValueError, match="1-D"):
            GaussianLikelihood(obs, model_error_frac=np.zeros((2, 2)))

    def test_zero_array_is_plain_chi2(self):
        """An all-zero array floor keeps the plain chi-squared fast path."""
        flux = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
        model = jnp.ones((1, 2, 2))
        obs = _obs(flux)
        like = GaussianLikelihood(obs, model_error_frac=np.zeros(1, np.float32))
        assert float(like(model)) == pytest.approx(-7.0, rel=1e-5)


class TestNoiseScale:
    """Per-band ``log_noise_scale`` rescaling of the variance."""

    @staticmethod
    def _numpy_reference(flux, variance, mask, model, frac, s):
        flux, variance, mask, model = (
            np.asarray(a, dtype=np.float64) for a in (flux, variance, mask, model)
        )
        frac = np.asarray(frac, dtype=np.float64).reshape(-1, 1, 1)
        s = np.asarray(s, dtype=np.float64).reshape(-1, 1, 1)
        var_eff = variance * np.exp(2.0 * s) + (frac * model) ** 2
        return -0.5 * np.sum(mask * ((flux - model) ** 2 / var_eff + np.log(var_eff)))

    def test_matches_numpy_reference_including_log_det(self):
        """The noise-scale branch matches a float64 numpy reference exactly."""
        rng = np.random.default_rng(31)
        flux = rng.normal(40.0, 4.0, size=(3, 6, 6)).astype(np.float32)
        variance = rng.uniform(1.0, 3.0, size=flux.shape).astype(np.float32)
        mask = (rng.uniform(size=flux.shape) > 0.2).astype(np.float32)
        model = rng.normal(40.0, 4.0, size=flux.shape).astype(np.float32)
        s = np.array([-0.3, 0.0, 0.4], dtype=np.float32)
        obs = _obs(flux, variance=variance, mask=mask)
        ll = float(GaussianLikelihood(obs)(jnp.asarray(model), log_noise_scale=jnp.asarray(s)))
        ref = self._numpy_reference(flux, variance, mask, model, np.zeros(3), s)
        assert ll == pytest.approx(ref, rel=1e-5)

    def test_combined_with_per_band_frac(self):
        """Noise scale and a per-band model-error floor combine as documented."""
        rng = np.random.default_rng(32)
        flux = rng.normal(40.0, 4.0, size=(2, 5, 5)).astype(np.float32)
        variance = rng.uniform(1.0, 3.0, size=flux.shape).astype(np.float32)
        model = rng.normal(40.0, 4.0, size=flux.shape).astype(np.float32)
        frac = np.array([0.03, 0.08], dtype=np.float32)
        s = np.array([0.2, -0.5], dtype=np.float32)
        obs = _obs(flux, variance=variance)
        ll = float(
            GaussianLikelihood(obs, model_error_frac=frac)(
                jnp.asarray(model), log_noise_scale=jnp.asarray(s)
            )
        )
        ref = self._numpy_reference(flux, variance, np.ones_like(flux), model, frac, s)
        assert ll == pytest.approx(ref, rel=1e-5)

    def test_zero_scale_equals_plain_chi2_plus_log_det(self):
        """Setting s = 0 keeps var_eff = variance, but the log-det term stays."""
        flux = jnp.full((1, 2, 2), 20.0)
        variance = jnp.full((1, 2, 2), 4.0)
        obs = _obs(flux, variance=variance)
        ll = float(GaussianLikelihood(obs)(flux, log_noise_scale=jnp.zeros(1)))
        assert ll == pytest.approx(-0.5 * 4 * np.log(4.0), rel=1e-5)

    def test_log_det_penalises_inflated_noise(self):
        """For a perfect model, inflating the noise only costs log-determinant."""
        flux = jnp.full((1, 3, 3), 10.0)
        obs = _obs(flux, variance=jnp.ones((1, 3, 3)))
        like = GaussianLikelihood(obs)
        ll0 = float(like(flux, log_noise_scale=jnp.zeros(1)))
        ll1 = float(like(flux, log_noise_scale=jnp.full(1, 0.5)))
        assert ll1 < ll0
        assert ll1 - ll0 == pytest.approx(-9.0 * 0.5, rel=1e-5)

    def test_none_keeps_exact_plain_chi2(self):
        """log_noise_scale=None with frac=0 is the historical plain chi-squared."""
        flux = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
        model = jnp.ones((1, 2, 2))
        obs = _obs(flux)
        assert float(GaussianLikelihood(obs)(model, log_noise_scale=None)) == pytest.approx(
            -7.0, rel=1e-5
        )

    def test_differentiable_in_noise_scale(self):
        """jax.grad flows into log_noise_scale and matches a finite difference."""
        rng = np.random.default_rng(33)
        flux = rng.normal(30.0, 3.0, size=(2, 4, 4)).astype(np.float32)
        variance = rng.uniform(1.0, 2.0, size=flux.shape).astype(np.float32)
        model = rng.normal(30.0, 3.0, size=flux.shape).astype(np.float32)
        obs = _obs(flux, variance=variance)
        like = GaussianLikelihood(obs)

        def fn(s):
            return like(jnp.asarray(model), log_noise_scale=s)

        s0 = np.array([0.1, -0.2], dtype=np.float32)
        grad = jax.grad(fn)(jnp.asarray(s0))
        assert grad.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(grad)))
        eps = 1e-3
        sp, sm = s0.copy(), s0.copy()
        sp[0] += eps
        sm[0] -= eps
        ones = np.ones_like(flux)
        fd = (
            self._numpy_reference(flux, variance, ones, model, np.zeros(2), sp)
            - self._numpy_reference(flux, variance, ones, model, np.zeros(2), sm)
        ) / (2 * eps)
        assert float(grad[0]) == pytest.approx(fd, rel=2e-2, abs=1e-2)

    def test_jit_compatible(self):
        """The noise-scale branch compiles under jax.jit."""
        flux = jnp.full((2, 3, 3), 10.0)
        obs = _obs(flux)
        like = GaussianLikelihood(obs)
        fn = jax.jit(lambda m, s: like(m, log_noise_scale=s))
        assert float(fn(flux * 0.9, jnp.array([0.1, 0.2]))) == pytest.approx(
            float(like(flux * 0.9, log_noise_scale=jnp.array([0.1, 0.2]))), rel=1e-6
        )


class TestPerBandComposition:
    """A per-band likelihood composes additively — what a multi-resolution fit needs."""

    def test_from_arrays_two_d_single_band(self):
        """A bare (H, W) flux/variance/mask is promoted to one band."""
        rng = np.random.default_rng(11)
        flux = rng.normal(10.0, 1.0, (6, 7)).astype(np.float32)
        var = rng.uniform(0.5, 2.0, (6, 7)).astype(np.float32)
        mask = np.ones((6, 7), dtype=np.float32)
        mask[0, 0] = 0.0
        like = GaussianLikelihood.from_arrays(flux, var, mask)
        assert like.obs.flux.shape == (1, 6, 7)
        model = jnp.asarray(flux * 0.9)[None]
        expected = -0.5 * np.sum(mask * (flux - flux * 0.9) ** 2 / var)
        assert float(like(model)) == pytest.approx(float(expected), rel=1e-4)

    def test_from_arrays_mask_defaults_to_all_valid(self):
        """mask=None marks every pixel valid."""
        like = GaussianLikelihood.from_arrays(np.zeros((4, 4), np.float32), np.ones((4, 4)))
        assert float(jnp.sum(like.obs.mask)) == pytest.approx(16.0)

    def test_from_arrays_bad_ndim_raises(self):
        """A 1-D flux array is rejected."""
        with pytest.raises(ValueError, match="must be"):
            GaussianLikelihood.from_arrays(np.zeros(5, np.float32), np.ones(5, np.float32))

    @pytest.mark.parametrize("frac", [0.0, 0.05])
    @pytest.mark.parametrize("use_scale", [False, True])
    def test_sum_of_per_band_equals_multi_band(self, frac, use_scale):
        """sum_b L_b(model_b) == L(model) exactly, with and without frac/noise scale."""
        rng = np.random.default_rng(12)
        n_bands, H, W = 3, 5, 6
        flux = rng.normal(10.0, 1.0, (n_bands, H, W)).astype(np.float32)
        var = rng.uniform(0.5, 2.0, (n_bands, H, W)).astype(np.float32)
        mask = (rng.uniform(size=(n_bands, H, W)) > 0.2).astype(np.float32)
        model = jnp.asarray(flux + rng.normal(0.0, 0.5, flux.shape).astype(np.float32))
        scale = jnp.asarray(np.array([0.1, -0.2, 0.3], dtype=np.float32)) if use_scale else None

        joint = GaussianLikelihood(_obs(flux, var, mask), model_error_frac=frac)
        total_joint = float(joint(model, log_noise_scale=scale))

        total_bands = 0.0
        for b in range(n_bands):
            like_b = GaussianLikelihood.from_arrays(flux[b], var[b], mask[b], model_error_frac=frac)
            s_b = None if scale is None else scale[b : b + 1]
            total_bands += float(like_b(model[b][None], log_noise_scale=s_b))
        assert total_bands == pytest.approx(total_joint, rel=1e-5)

    def test_per_band_frac_array_splits_per_band(self):
        """A per-band frac array is equivalent to per-band scalar fracs."""
        rng = np.random.default_rng(13)
        flux = rng.normal(10.0, 1.0, (2, 4, 4)).astype(np.float32)
        var = np.ones((2, 4, 4), dtype=np.float32)
        mask = np.ones((2, 4, 4), dtype=np.float32)
        model = jnp.asarray(flux * 0.95)
        fracs = np.array([0.02, 0.1], dtype=np.float32)
        joint = float(GaussianLikelihood(_obs(flux, var, mask), model_error_frac=fracs)(model))
        parts = sum(
            float(
                GaussianLikelihood.from_arrays(
                    flux[b], var[b], mask[b], model_error_frac=float(fracs[b])
                )(model[b][None])
            )
            for b in range(2)
        )
        assert parts == pytest.approx(joint, rel=1e-5)

    def test_infinite_variance_on_masked_pixels_is_not_nan(self):
        """Masked pixels may carry variance = inf (as MultiResolutionObservation sets).

        The log(var_eff) term would otherwise multiply 0 by inf and give NaN.
        """
        flux = np.ones((1, 4, 4), dtype=np.float32)
        var = np.ones((1, 4, 4), dtype=np.float32)
        mask = np.ones((1, 4, 4), dtype=np.float32)
        mask[0, 0, 0] = 0.0
        var[0, 0, 0] = np.inf
        model = jnp.full((1, 4, 4), 0.8, dtype=jnp.float32)

        for kwargs in ({"model_error_frac": 0.05}, {}):
            like = GaussianLikelihood.from_arrays(flux, var, mask, **kwargs)
            scale = jnp.zeros(1, dtype=jnp.float32)
            value = float(like(model, log_noise_scale=scale))
            assert np.isfinite(value)
        # And the masked pixel contributes nothing: a 3x4-valid reference matches.
        like = GaussianLikelihood.from_arrays(flux, var, mask, model_error_frac=0.05)
        var_eff = 1.0 + (0.05 * 0.8) ** 2
        expected = -0.5 * 15 * ((1.0 - 0.8) ** 2 / var_eff + np.log(var_eff))
        assert float(like(model)) == pytest.approx(expected, rel=1e-4)
