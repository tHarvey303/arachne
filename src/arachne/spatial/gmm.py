"""Gaussian Mixture Model spatial parameterisation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from arachne.spatial.base import SpatialModel


class GaussianMixtureSpatialModel(SpatialModel):
    """K-component Gaussian Mixture Model spatial model.

    Each Gaussian component occupies a region of the image and carries its own
    SPS parameter vector.  Per-pixel SPS parameters are the mixture-weighted
    average:

        params(i,j) = Σ_k w_k(i,j) * sps_k

    where the weight of component k at pixel (i,j) is the normalised bivariate
    Gaussian evaluated at the pixel's image coordinates.

    Free parameters per component
    ------------------------------
    (mu_x, mu_y, log_sigma_x, log_sigma_y, atanh_rho, sps_params...)

    Total free parameters: ``n_components * (5 + N_sps_params)``

    Parameterisation
    ----------------
    - ``mu_x, mu_y``: component centre in pixel coordinates (unconstrained).
    - ``log_sigma_x, log_sigma_y``: log of spatial standard deviations.
    - ``atanh_rho``: inverse-tanh of correlation coefficient ρ ∈ (-1, 1).
    - ``sps_params``: per-component SPS parameters mapped through sigmoid to
      enforce physical bounds.

    Prior
    -----
    - Component means: soft Gaussian penalty to keep components inside the image.
    - log_sigma: Normal(0, 1) — discourages extreme component widths.
    - SPS params: independent uniform within bounds (flat in sigmoid space).

    Attributes:
        n_components: Number of Gaussian components K.
        sps_param_names: Ordered list of SPS parameter names.
        param_bounds: Dict of param_name → (lo, hi) physical bounds.
        pixel_coords: Static array of pixel (y, x) coordinates, shape (H*W, 2).
        image_shape: Spatial dimensions (H, W) stored for initialising coords.
    """

    def __init__(
        self,
        n_components: int,
        sps_param_names: list[str],
        param_bounds: dict[str, tuple[float, float]],
        image_shape: tuple[int, int],
    ) -> None:
        """Initialise the GaussianMixtureSpatialModel.

        Args:
            n_components: Number of Gaussian components K.
            sps_param_names: Ordered list of SPS parameter names.
            param_bounds: Dict of param_name → (lo, hi) physical bounds.
            image_shape: (H, W) spatial dimensions of the galaxy image.
        """
        self.n_components = n_components
        self.sps_param_names = sps_param_names
        self.param_bounds = param_bounds
        self.image_shape = image_shape

        H, W = image_shape
        # Pre-compute pixel coordinate grid (y, x) — static
        ys = jnp.arange(H, dtype=jnp.float32)
        xs = jnp.arange(W, dtype=jnp.float32)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
        self.pixel_coords = jnp.stack([yy.ravel(), xx.ravel()], axis=-1)  # (H*W, 2)

        N = len(sps_param_names)
        self._n_sps = N
        self._n_params_per_component = 5 + N
        self._n_params = n_components * (5 + N)

        lows = jnp.array([param_bounds[p][0] for p in sps_param_names], dtype=jnp.float32)
        highs = jnp.array([param_bounds[p][1] for p in sps_param_names], dtype=jnp.float32)
        self._lows = lows
        self._highs = highs

        H_f = float(H)
        W_f = float(W)
        self._H = H_f
        self._W = W_f

    @property
    def n_params(self) -> int:
        """Total number of free parameters: K * (5 + N_sps_params)."""
        return self._n_params

    def _parse_component(self, theta_k: jnp.ndarray) -> tuple:
        """Parse the 5 + N_sps raw parameters for one component.

        Args:
            theta_k: Raw parameter vector for component k, shape (5 + N_sps,).

        Returns:
            Tuple of (mu, sigma, rho, sps_params) where:
            - mu: shape (2,) centre in pixel coords (y, x)
            - sigma: shape (2,) standard deviations (sigma_y, sigma_x)
            - rho: scalar correlation coefficient ∈ (-1, 1)
            - sps_params: shape (N_sps,) physical SPS parameters
        """
        mu = theta_k[:2]  # mu_y, mu_x (unconstrained)
        log_sigma = theta_k[2:4]
        atanh_rho = theta_k[4]
        sps_raw = theta_k[5:]

        sigma = jnp.exp(log_sigma)  # positive
        rho = jnp.tanh(atanh_rho)  # ∈ (-1, 1)
        sps_params = self._lows + (self._highs - self._lows) * jax.nn.sigmoid(sps_raw)
        return mu, sigma, rho, sps_params

    def _gaussian_log_weights(
        self, mu: jnp.ndarray, sigma: jnp.ndarray, rho: jnp.ndarray
    ) -> jnp.ndarray:
        """Evaluate log of unnormalised bivariate Gaussian at all pixel coordinates.

        Returning log-weights rather than weights avoids float32 underflow when
        components are far from pixels.  Normalisation is done via
        ``jax.nn.softmax`` in ``decode()``, which is numerically stable.

        Args:
            mu: Component centre (y, x), shape (2,).
            sigma: Standard deviations (sigma_y, sigma_x), shape (2,).
            rho: Correlation coefficient, scalar.

        Returns:
            Log unnormalised Gaussian weights at each pixel, shape (H*W,).
        """
        dy = self.pixel_coords[:, 0] - mu[0]  # (H*W,)
        dx = self.pixel_coords[:, 1] - mu[1]  # (H*W,)
        sy, sx = sigma[0], sigma[1]
        z = (dy / sy) ** 2 - 2 * rho * (dy / sy) * (dx / sx) + (dx / sx) ** 2
        denom = 2 * (1 - rho**2)
        return -z / denom  # log of unnormalised Gaussian, shape (H*W,)

    def decode(self, theta: jnp.ndarray, image_shape: tuple) -> jnp.ndarray:
        """Map unconstrained theta to per-pixel physical SPS parameters.

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).
            image_shape: Spatial dimensions (H, W) — must match self.image_shape.

        Returns:
            Per-pixel SPS parameter array of shape (H*W, N_sps_params).
        """
        K = self.n_components
        N = self._n_sps
        # Parse per-component parameters
        theta_components = theta.reshape(K, 5 + N)

        def component_log_weights_and_sps(
            theta_k: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            mu, sigma, rho, sps = self._parse_component(theta_k)
            log_weights = self._gaussian_log_weights(mu, sigma, rho)  # (H*W,)
            return log_weights, sps

        # vmap over components
        all_log_weights, all_sps = jax.vmap(component_log_weights_and_sps)(theta_components)
        # all_log_weights: (K, H*W), all_sps: (K, N_sps)

        # Softmax across components at each pixel — numerically stable, always sums to 1
        norm_weights = jax.nn.softmax(all_log_weights, axis=0)  # (K, H*W)

        # Mixture-weighted SPS params
        # pixel_params = einsum('kp,kc->pc', norm_weights, all_sps)
        pixel_params = jnp.einsum("kp,kc->pc", norm_weights, all_sps)  # (H*W, N_sps)
        return pixel_params

    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the log-prior for the GMM parameters.

        Prior structure:
        - Component centres: soft Gaussian penalty to keep within image extent.
        - log_sigma: Standard Normal — encourages moderate component widths.
        - atanh_rho: Standard Normal — mild regularisation on correlation.
        - SPS params: uniform in sigmoid space (flat, no penalty).

        Args:
            theta: Unconstrained parameter vector of shape (n_params,).

        Returns:
            Scalar log-prior value.
        """
        K = self.n_components
        N = self._n_sps
        theta_components = theta.reshape(K, 5 + N)

        def component_log_prior(theta_k: jnp.ndarray) -> jnp.ndarray:
            mu = theta_k[:2]  # mu_y, mu_x
            log_sigma = theta_k[2:4]
            atanh_rho = theta_k[4]

            # Soft Gaussian prior on centres — keep within image
            mu_prior = -0.5 * ((mu[0] / self._H) ** 2 + (mu[1] / self._W) ** 2)
            # Normal(0, 1) on log_sigma — moderate component widths
            log_sigma_prior = -0.5 * jnp.sum(log_sigma**2)
            # Normal(0, 1) on atanh_rho — mild regularisation
            rho_prior = -0.5 * atanh_rho**2
            return mu_prior + log_sigma_prior + rho_prior

        return jnp.sum(jax.vmap(component_log_prior)(theta_components))
