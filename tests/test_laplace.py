"""Tests for the Laplace approximation / whitened sampling helpers.

Everything runs on CPU with the tiny 3-band 16x16 fixtures, plus a couple of
analytic Gaussian targets where the right answer is known exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.laplace import (
    WhitenedLogDensity,
    hessian_neg_log_post,
    laplace_covariance,
    laplace_whitening,
    make_hessian_fn,
    posterior_whitening,
    run_whitened_nuts,
)


class _Quadratic:
    """Analytic ``log_posterior`` of a correlated Gaussian, with the ForwardModel API."""

    def __init__(self, cov: np.ndarray, mean: np.ndarray | None = None) -> None:
        self.cov = np.asarray(cov, dtype=np.float64)
        self.precision = jnp.asarray(np.linalg.inv(self.cov), jnp.float32)
        self.mean = jnp.asarray(
            np.zeros(self.cov.shape[0]) if mean is None else np.asarray(mean), jnp.float32
        )
        self.n_params = self.cov.shape[0]
        self.spatial_model = None

    def log_posterior(self, theta: jnp.ndarray) -> jnp.ndarray:
        d = jnp.asarray(theta) - self.mean
        return -0.5 * d @ self.precision @ d


@pytest.fixture
def forward_model(tiny_observation, gaussian_psf, mock_emulator, gmm_model):
    """Small ForwardModel (3 bands, 16x16, K=2 GMM) for the integration tests."""
    return ForwardModel.build(
        obs=tiny_observation,
        psf_model=gaussian_psf,
        spatial_model=gmm_model,
        emulator=mock_emulator,
    )


# ---------------------------------------------------------------------------
# Hessian
# ---------------------------------------------------------------------------


class TestHessian:
    """``make_hessian_fn`` / ``hessian_neg_log_post``."""

    def test_matches_analytic_precision(self):
        """For a Gaussian target the Hessian of -log p is the precision matrix."""
        cov = np.array([[4.0, 1.5], [1.5, 1.0]])
        model = _Quadratic(cov)
        hess = hessian_neg_log_post(model, np.zeros(2))
        np.testing.assert_allclose(hess, np.linalg.inv(cov), rtol=1e-4, atol=1e-5)

    def test_symmetric_and_shaped(self, forward_model, gmm_model):
        """A real forward model gives a symmetric (d, d) float64 Hessian."""
        hess_at = make_hessian_fn(forward_model)
        hess = hess_at(jnp.zeros(gmm_model.n_params))
        assert hess.shape == (gmm_model.n_params, gmm_model.n_params)
        assert hess.dtype == np.float64
        np.testing.assert_allclose(hess, hess.T, rtol=0, atol=0)

    def test_reusable_at_several_points(self, forward_model, gmm_model):
        """The built closure can be called at more than one point."""
        hess_at = make_hessian_fn(forward_model)
        h0 = hess_at(jnp.zeros(gmm_model.n_params))
        h1 = hess_at(0.1 * jnp.ones(gmm_model.n_params))
        assert np.isfinite(h0).all() and np.isfinite(h1).all()


# ---------------------------------------------------------------------------
# Laplace covariance
# ---------------------------------------------------------------------------


class TestLaplaceCovariance:
    """``laplace_covariance``."""

    def test_inverts_a_well_conditioned_hessian(self):
        """With no clipping the covariance is exactly the inverse Hessian."""
        rng = np.random.default_rng(0)
        a = rng.normal(size=(5, 5))
        hess = a @ a.T + 5.0 * np.eye(5)
        cov, chol = laplace_covariance(hess, max_variance=1e6)
        np.testing.assert_allclose(cov, np.linalg.inv(hess), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(chol @ chol.T, cov, rtol=1e-6, atol=1e-9)

    def test_cholesky_is_lower_triangular(self):
        """``chol`` is lower triangular."""
        _cov, chol = laplace_covariance(np.diag([1.0, 2.0, 3.0]))
        assert np.allclose(np.triu(chol, 1), 0.0)

    def test_negative_curvature_uses_magnitude(self):
        """A negative eigenvalue becomes 1/|lambda|, not the maximum variance."""
        hess = np.diag([100.0, -25.0])
        cov, _chol = laplace_covariance(hess, max_variance=9.0, use_abs_eigenvalues=True)
        np.testing.assert_allclose(np.diag(cov), [1 / 100.0, 1 / 25.0], rtol=1e-8)

    def test_negative_curvature_clipped_when_disabled(self):
        """The historical behaviour clips a negative direction to ``max_variance``."""
        hess = np.diag([100.0, -25.0])
        cov, _chol = laplace_covariance(hess, max_variance=9.0, use_abs_eigenvalues=False)
        np.testing.assert_allclose(np.diag(cov), [1 / 100.0, 9.0], rtol=1e-8)

    def test_variance_capped_by_max_variance(self):
        """A nearly flat direction is capped at ``max_variance``."""
        hess = np.diag([1e3, 1e-9])
        cov, _chol = laplace_covariance(hess, max_variance=4.0)
        assert np.diag(cov)[1] == pytest.approx(4.0, rel=1e-6)

    def test_info_dict(self):
        """``return_info`` reports the raw spectrum and what was modified."""
        hess = np.diag([1e3, -1.0, 1e-12])
        _cov, _chol, info = laplace_covariance(hess, max_variance=9.0, return_info=True)
        assert info["n_negative"] == 1
        assert info["n_floored"] >= 1
        assert info["marginal_sd"].shape == (3,)
        assert info["condition_number"] > 1.0
        np.testing.assert_allclose(np.sort(info["eigenvalues"]), np.sort(np.diag(hess)))

    def test_rejects_non_square(self):
        """A non-square Hessian raises."""
        with pytest.raises(ValueError, match="square"):
            laplace_covariance(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# WhitenedLogDensity
# ---------------------------------------------------------------------------


class TestWhitenedLogDensity:
    """``WhitenedLogDensity`` and ``laplace_whitening``."""

    def test_round_trip(self):
        """``from_theta(to_theta(z)) == z`` for single vectors and batches."""
        cov = np.array([[4.0, 1.5], [1.5, 1.0]])
        model = _Quadratic(cov, mean=[0.3, -0.2])
        white, _info = laplace_whitening(model, np.array([0.3, -0.2]))
        z = jnp.array([0.7, -1.3])
        np.testing.assert_allclose(np.asarray(white.from_theta(white.to_theta(z))), z, atol=1e-5)
        zb = jnp.array([[0.7, -1.3], [0.0, 2.0], [-1.0, 0.5]])
        np.testing.assert_allclose(np.asarray(white.from_theta(white.to_theta(zb))), zb, atol=1e-5)

    def test_whitened_target_is_standard_normal(self):
        """Whitening a Gaussian with its own covariance gives curvature = I."""
        cov = np.array([[4.0, 1.5], [1.5, 1.0]])
        model = _Quadratic(cov)
        white, _info = laplace_whitening(model, np.zeros(2))
        hess_z = hessian_neg_log_post(white, np.zeros(2))
        np.testing.assert_allclose(hess_z, np.eye(2), rtol=1e-3, atol=1e-4)

    def test_log_prob_alias_and_offset(self):
        """``log_prob`` is ``log_posterior`` and equals the model at ``to_theta(z)``."""
        cov = np.array([[4.0, 1.5], [1.5, 1.0]])
        model = _Quadratic(cov)
        white = WhitenedLogDensity(model, np.zeros(2), np.linalg.cholesky(cov))
        z = jnp.array([0.4, -0.9])
        assert float(white.log_prob(z)) == pytest.approx(float(white.log_posterior(z)))
        assert float(white.log_posterior(z)) == pytest.approx(
            float(model.log_posterior(white.to_theta(z))), rel=1e-5
        )

    def test_is_jittable(self):
        """``log_posterior`` is a pure function of z and can be jitted / differentiated."""
        cov = np.array([[4.0, 1.5], [1.5, 1.0]])
        white = WhitenedLogDensity(_Quadratic(cov), np.zeros(2), np.linalg.cholesky(cov))
        g = jax.jit(jax.grad(white.log_posterior))(jnp.array([1.0, 0.0]))
        np.testing.assert_allclose(np.asarray(g), [-1.0, 0.0], atol=1e-4)

    def test_forward_model_attributes(self, forward_model, gmm_model):
        """The wrapper exposes ``n_params`` and the wrapped ``spatial_model``."""
        hess = np.eye(gmm_model.n_params)
        white = WhitenedLogDensity(forward_model, np.zeros(gmm_model.n_params), hess)
        assert white.n_params == gmm_model.n_params
        assert white.spatial_model is gmm_model
        assert "WhitenedLogDensity" in repr(white)

    def test_shape_validation(self):
        """Mismatched ``mean`` / ``chol`` shapes raise."""
        with pytest.raises(ValueError, match="mean must be"):
            WhitenedLogDensity(_Quadratic(np.eye(2)), np.zeros(2), np.eye(3))


# ---------------------------------------------------------------------------
# posterior_whitening
# ---------------------------------------------------------------------------


class TestPosteriorWhitening:
    """``posterior_whitening``: re-centre / re-scale on exploratory draws."""

    def _chains(self, cov, mean, n=400, seed=0):
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(mean, cov, size=(2, n))
        return draws

    def test_recovers_the_explored_covariance(self):
        """With little shrinkage the whitened draws have unit covariance."""
        cov = np.array([[4.0, 1.8], [1.8, 1.0]])
        mean = np.array([1.0, -2.0])
        chains = self._chains(cov, mean, n=2000)
        model = _Quadratic(cov, mean=mean)
        white, info = posterior_whitening(model, chains, shrinkage=0.0)
        np.testing.assert_allclose(info["mean"], mean, atol=0.1)
        np.testing.assert_allclose(info["cov"], cov, rtol=0.15, atol=0.15)
        z = np.asarray(white.from_theta(jnp.asarray(chains.reshape(-1, 2))))
        np.testing.assert_allclose(np.cov(z.T), np.eye(2), atol=0.1)

    def test_shrinkage_default_and_mixing(self):
        """The default shrinkage is d / (d + n_draws) and mixes in ``prior_cov``."""
        cov = np.eye(3)
        chains = self._chains(cov, np.zeros(3), n=50)
        prior = 9.0 * np.eye(3)
        white, info = posterior_whitening(_Quadratic(cov), chains, prior_cov=prior)
        assert info["n_draws"] == 100
        assert info["shrinkage"] == pytest.approx(3 / 103.0)
        emp = np.cov(chains.reshape(-1, 3).T, bias=False)
        expected = (1 - info["shrinkage"]) * emp + info["shrinkage"] * prior
        np.testing.assert_allclose(info["cov"], expected, rtol=1e-6, atol=1e-8)
        assert white.n_params == 3

    def test_explicit_mean(self):
        """An explicit centre is used instead of the pooled mean."""
        cov = np.eye(2)
        chains = self._chains(cov, np.array([5.0, 5.0]), n=100)
        _white, info = posterior_whitening(_Quadratic(cov), chains, mean=np.zeros(2))
        np.testing.assert_allclose(info["mean"], np.zeros(2))

    def test_accepts_flat_draws(self):
        """A 2-D (n_samples, d) array is accepted."""
        cov = np.eye(2)
        flat = self._chains(cov, np.zeros(2), n=100).reshape(-1, 2)
        _white, info = posterior_whitening(_Quadratic(cov), flat)
        assert info["n_draws"] == 200

    def test_rejects_1d(self):
        """A 1-D input raises."""
        with pytest.raises(ValueError, match="chains must be"):
            posterior_whitening(_Quadratic(np.eye(2)), np.zeros(2))

    def test_feeds_run_whitened_nuts(self):
        """The refined whitening can be passed straight to ``run_whitened_nuts``."""
        pytest.importorskip("blackjax")
        cov = np.array([[4.0, 1.8], [1.8, 1.0]])
        model = _Quadratic(cov)
        stage1, _ = run_whitened_nuts(
            model, np.zeros(2), jax.random.PRNGKey(0), n_warmup=150, n_samples=200, n_chains=2
        )
        white, _info = posterior_whitening(model, stage1.chains, prior_cov=cov)
        stage2, _ = run_whitened_nuts(
            model,
            np.zeros(2),
            jax.random.PRNGKey(1),
            n_warmup=150,
            n_samples=200,
            n_chains=2,
            whitened=white,
        )
        samples = np.asarray(stage2.samples)
        np.testing.assert_allclose(samples.mean(axis=0), np.zeros(2), atol=0.3)
        np.testing.assert_allclose(np.cov(samples.T), cov, rtol=0.5, atol=0.5)


# ---------------------------------------------------------------------------
# run_whitened_nuts
# ---------------------------------------------------------------------------


class TestRunWhitenedNUTS:
    """End-to-end whitened NUTS on tiny problems."""

    def test_returns_theta_coordinates(self):
        """Samples come back in theta space and recover a correlated Gaussian."""
        pytest.importorskip("blackjax")
        cov = np.array([[4.0, 1.9], [1.9, 1.0]])
        model = _Quadratic(cov, mean=[1.0, -1.0])
        result, white = run_whitened_nuts(
            model,
            np.array([1.0, -1.0]),
            jax.random.PRNGKey(0),
            n_warmup=200,
            n_samples=400,
            n_chains=2,
            max_num_doublings=6,
        )
        assert result.chains.shape == (2, 400, 2)
        assert result.diagnostics["whitened"] is True
        assert result.diagnostics["dense_mass_matrix"] is True
        samples = np.asarray(result.samples)
        np.testing.assert_allclose(samples.mean(axis=0), [1.0, -1.0], atol=0.25)
        np.testing.assert_allclose(np.cov(samples.T), cov, rtol=0.4, atol=0.4)
        # The sampler lived in z: those draws should be ~N(0, I).
        z = np.asarray(white.from_theta(result.samples))
        np.testing.assert_allclose(np.cov(z.T), np.eye(2), atol=0.35)

    def test_diagonal_metric_option(self):
        """``dense_mass_matrix=False`` still runs and is recorded."""
        pytest.importorskip("blackjax")
        model = _Quadratic(np.diag([1.0, 2.0]))
        result, _white = run_whitened_nuts(
            model,
            np.zeros(2),
            jax.random.PRNGKey(1),
            n_warmup=100,
            n_samples=100,
            n_chains=2,
            dense_mass_matrix=False,
            max_num_doublings=5,
        )
        assert result.diagnostics["dense_mass_matrix"] is False
        assert np.isfinite(np.asarray(result.samples)).all()

    def test_forward_model_integration(self, forward_model, gmm_model):
        """A real (tiny) ForwardModel runs end to end and keeps its spatial model."""
        pytest.importorskip("blackjax")
        theta0 = jnp.zeros(gmm_model.n_params)
        hess = make_hessian_fn(forward_model)(theta0)
        result, _white = run_whitened_nuts(
            forward_model,
            theta0,
            jax.random.PRNGKey(2),
            n_warmup=20,
            n_samples=20,
            n_chains=2,
            hess=hess,
            max_num_doublings=4,
        )
        assert result.samples.shape == (40, gmm_model.n_params)
        assert result.spatial_model is gmm_model
        assert np.isfinite(np.asarray(result.samples)).all()
        assert result.diagnostics["rhat"].shape == (gmm_model.n_params,)
