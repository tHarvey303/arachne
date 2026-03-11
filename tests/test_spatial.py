"""Tests for spatial models (GMM and FreeFormPixelMap)."""

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# FreeFormPixelMap tests
# ---------------------------------------------------------------------------


class TestFreeFormPixelMap:
    """Tests for FreeFormPixelMap."""

    def test_n_params(self, pixel_map_model):
        """n_params == H * W * N_sps_params."""
        assert pixel_map_model.n_params == 16 * 16 * 3

    def test_decode_shape(self, pixel_map_model):
        """decode() returns (H*W, N_sps) shaped array."""
        theta = jnp.zeros(pixel_map_model.n_params)
        out = pixel_map_model.decode(theta, (16, 16))
        assert out.shape == (16 * 16, 3)

    def test_decode_bounds_enforced(self, pixel_map_model):
        """decode() enforces physical parameter bounds via sigmoid."""
        theta_high = jnp.full(pixel_map_model.n_params, 10.0)
        out_high = pixel_map_model.decode(theta_high, (16, 16))
        theta_low = jnp.full(pixel_map_model.n_params, -10.0)
        out_low = pixel_map_model.decode(theta_low, (16, 16))

        bounds = pixel_map_model.param_bounds
        for i, name in enumerate(pixel_map_model.sps_param_names):
            lo, hi = bounds[name]
            assert jnp.all(out_high[:, i] < hi + 1e-3)
            assert jnp.all(out_low[:, i] > lo - 1e-3)
            assert jnp.all(out_high[:, i] > lo)
            assert jnp.all(out_low[:, i] < hi)

    def test_log_prior_scalar(self, pixel_map_model):
        """log_prior() returns a scalar."""
        theta = jnp.zeros(pixel_map_model.n_params)
        lp = pixel_map_model.log_prior(theta)
        assert lp.shape == ()

    def test_log_prior_nonpositive(self, pixel_map_model):
        """log_prior() is <= 0 (L2 penalty)."""
        theta = jnp.ones(pixel_map_model.n_params)
        lp = pixel_map_model.log_prior(theta)
        assert float(lp) <= 0.0

    def test_log_prior_differentiable(self, pixel_map_model):
        """jax.grad can differentiate through log_prior."""
        theta = jnp.zeros(pixel_map_model.n_params)
        grad = jax.grad(pixel_map_model.log_prior)(theta)
        assert grad.shape == theta.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_decode_differentiable(self, pixel_map_model):
        """jax.grad can differentiate through decode()."""

        def loss(theta):
            return jnp.sum(pixel_map_model.decode(theta, (16, 16)))

        theta = jnp.zeros(pixel_map_model.n_params)
        grad = jax.grad(loss)(theta)
        assert grad.shape == theta.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_flat_theta_uniform_map(self, pixel_map_model):
        """A constant theta produces a spatially uniform parameter map."""
        theta = jnp.zeros(pixel_map_model.n_params)
        out = pixel_map_model.decode(theta, (16, 16))
        for i in range(3):
            assert float(jnp.std(out[:, i])) < 1e-5

    def test_smoothness_penalty_increases_with_roughness(self, pixel_map_model):
        """A rough theta has a more negative log_prior than a flat theta."""
        theta_flat = jnp.zeros(pixel_map_model.n_params)
        rng = np.random.default_rng(0)
        theta_rough = jnp.array(rng.standard_normal(pixel_map_model.n_params).astype(np.float32))
        lp_flat = float(pixel_map_model.log_prior(theta_flat))
        lp_rough = float(pixel_map_model.log_prior(theta_rough))
        assert lp_flat >= lp_rough


# ---------------------------------------------------------------------------
# GaussianMixtureSpatialModel tests
# ---------------------------------------------------------------------------


class TestGMM:
    """Tests for GaussianMixtureSpatialModel."""

    def test_n_params(self, gmm_model):
        """n_params == K * (5 + N_sps_params)."""
        K = gmm_model.n_components
        N = len(gmm_model.sps_param_names)
        assert gmm_model.n_params == K * (5 + N)

    def test_decode_shape(self, gmm_model):
        """decode() returns (H*W, N_sps) shaped array."""
        theta = jnp.zeros(gmm_model.n_params)
        out = gmm_model.decode(theta, (16, 16))
        assert out.shape == (16 * 16, 3)

    def test_decoded_values_within_bounds(self, gmm_model):
        """Decoded pixel params lie within physical bounds."""
        theta = jnp.zeros(gmm_model.n_params)
        out = gmm_model.decode(theta, (16, 16))
        bounds = gmm_model.param_bounds
        for i, name in enumerate(gmm_model.sps_param_names):
            lo, hi = bounds[name]
            assert jnp.all(out[:, i] >= lo - 1e-3)
            assert jnp.all(out[:, i] <= hi + 1e-3)

    def test_decode_differentiable(self, gmm_model):
        """jax.grad can differentiate through decode()."""

        def loss(theta):
            return jnp.sum(gmm_model.decode(theta, (16, 16)))

        theta = jnp.zeros(gmm_model.n_params)
        grad = jax.grad(loss)(theta)
        assert grad.shape == theta.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_log_prior_scalar(self, gmm_model):
        """log_prior() returns a scalar."""
        theta = jnp.zeros(gmm_model.n_params)
        lp = gmm_model.log_prior(theta)
        assert lp.shape == ()

    def test_log_prior_differentiable(self, gmm_model):
        """jax.grad can differentiate through log_prior."""
        theta = jnp.zeros(gmm_model.n_params)
        grad = jax.grad(gmm_model.log_prior)(theta)
        assert grad.shape == theta.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_pixel_coords_shape(self, gmm_model):
        """pixel_coords has shape (H*W, 2)."""
        assert gmm_model.pixel_coords.shape == (16 * 16, 2)

    def test_separated_components_produce_spatial_variation(self, gmm_model):
        """Components at opposite corners produce a spatially varying parameter map."""
        N = len(gmm_model.sps_param_names)
        theta_list = list(np.zeros(gmm_model.n_params, dtype=np.float32))
        # Component 0: top-left (mu_y=0, mu_x=0), sps[0] at lower bound (theta=-5 -> sigmoid~0)
        theta_list[0] = 0.0
        theta_list[1] = 0.0
        for j in range(N):
            theta_list[5 + j] = -5.0  # push SPS toward lower bound
        # Component 1: bottom-right (mu_y=15, mu_x=15), sps[0] at upper bound
        # (theta=+5 -> sigmoid~1)
        start = 5 + N
        theta_list[start] = 15.0
        theta_list[start + 1] = 15.0
        for j in range(N):
            theta_list[start + 5 + j] = 5.0  # push SPS toward upper bound
        theta = jnp.array(theta_list)
        out = gmm_model.decode(theta, (16, 16))
        assert float(jnp.std(out[:, 0])) > 0
