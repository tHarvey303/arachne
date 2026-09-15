"""Additive-flux multi-component spatial model."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from arachne.emulator.base import SPSEmulator
from arachne.spatial.base import SpatialModel

_RHO_MAX = 0.99
_LOG_2PI = math.log(2.0 * math.pi)
_LN10 = math.log(10.0)


class AdditiveComponentModel(SpatialModel):
    """K additive components, each a Gaussian light profile with its own SED.

    Light is additive; SPS parameters are not.  This model therefore describes
    the galaxy as a sum of ``K`` components.  Component ``k`` has a bivariate
    Gaussian surface-brightness profile ``P_k(y, x)`` normalised to sum to 1
    over the image, and a full SED ``F_k`` (N_bands,) predicted by the
    emulator from that component's *own* SPS parameters, including its own
    total stellar mass.  The unconvolved model image is::

        I_b(y, x) = Σ_k F_kb · P_k(y, x)

    which costs exactly ``K`` emulator evaluations per likelihood call, no
    matter how many pixels the image has.  This is the physically correct
    replacement for :class:`~arachne.spatial.gmm.GaussianMixtureSpatialModel`
    for bulge/disk-style decompositions.

    Parameter roles
    ---------------
    Every name in ``emulator_param_names`` (the emulator's full ordered input
    list) must play exactly one of three roles:

    - **fixed** (``fixed_params``): held constant at the given value.
    - **shared** (``shared_param_names``): one free value common to all
      components (e.g. redshift).
    - **per-component** (everything else): one free value per component.
      ``mass_param`` must be per-component.

    Per-component free names, in emulator order, are exposed as
    ``sps_param_names`` (this is what ``decode`` returns columns for).

    theta layout
    ------------
    ``theta`` is a flat float32 vector of length
    ``n_params = K * (5 + N_free) + N_shared``::

        theta = concat([block_0, block_1, ..., block_{K-1}, shared_raw])
        block_k = [mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho, sps_raw_k (N_free,)]

    Use :meth:`split_theta` / :meth:`join_theta` rather than slicing by hand.

    Physical mapping
    ----------------
    - ``mu = (mu_y, mu_x)``: component centre in pixel (row, col) coordinates,
      unconstrained.
    - ``sigma = exp(log_sigma)`` in pixels, per axis.
    - ``rho = tanh(atanh_rho)`` clipped to ±0.99.
    - every SPS parameter (per-component and shared):
      ``phys = lo + (hi - lo) * sigmoid(raw)``.

    Prior
    -----
    ``log_prior`` is the sum of:

    (a) ``log_sigmoid(raw) + log_sigmoid(-raw)`` for every per-component and
        shared SPS entry: the normalised log-density of a prior *uniform in
        physical space* pulled back to ``raw`` (the sigmoid Jacobian);
    (b) ``sps_log_prior(sps_phys)`` if supplied, where ``sps_phys`` is the
        (K, N_emulator) physical matrix from :meth:`component_params`.  Shared
        parameters appear identically in every row — the callable is
        responsible for counting them once;
    (c) ``Normal(image centre, centre_prior_sigma)`` on each centre coordinate
        (default sigma ``max(H, W) / 4``);
    (d) ``Normal(mu, sd) = log_size_prior`` on each ``log_sigma``
        (default ``(log(max(H, W) / 16), 1.0)``);
    (e) ``Normal(0, rho_prior_sigma)`` on ``atanh_rho``.

    Terms (c)–(e) plus the Jacobian make the prior a proper density in theta;
    :meth:`sample_prior` draws from exactly that density (without the optional
    ``sps_log_prior`` term).

    ``decode`` is a summary map for plotting / ``NUTSResult.get_parameter_map``
    compatibility.  It is **not** what the likelihood uses.

    Attributes:
        n_components: Number of components K.
        image_shape: (H, W).
        emulator_param_names: Full ordered emulator input list.
        sps_param_names: Per-component free SPS names (emulator order).
        shared_param_names: Shared free SPS names (emulator order).
        fixed_params: Dict of fixed name -> value.
        param_bounds: Dict of name -> (lo, hi) for every free parameter.
        mass_param: Name of the per-component log10 stellar-mass parameter.
        mass_index: Column of ``mass_param`` within ``sps_param_names``.
    """

    def __init__(
        self,
        n_components: int,
        emulator_param_names: list[str],
        param_bounds: dict[str, tuple[float, float]],
        image_shape: tuple[int, int],
        shared_param_names: Sequence[str] = (),
        fixed_params: dict[str, float] | None = None,
        mass_param: str = "log_mass",
        sps_log_prior: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        centre_prior_sigma: float | None = None,
        log_size_prior: tuple[float, float] | None = None,
        rho_prior_sigma: float = 1.0,
    ) -> None:
        """Initialise the AdditiveComponentModel.

        Args:
            n_components: Number of additive components K (>= 1).
            emulator_param_names: The emulator's full ordered parameter list.
            param_bounds: name -> (lo, hi) physical bounds; required for every
                free (per-component or shared) parameter.
            image_shape: (H, W) of the galaxy image.
            shared_param_names: Names with one free value common to all components.
            fixed_params: name -> constant value, excluded from theta.
            mass_param: Name of the per-component log10 stellar mass parameter.
            sps_log_prior: Optional callable ``(K, N_emulator) -> scalar`` adding
                a physical-space prior on the component SPS matrix.
            centre_prior_sigma: Sigma (pixels) of the Gaussian centre prior.
                Default ``max(H, W) / 4``.
            log_size_prior: (mean, sd) of the Gaussian prior on each log_sigma.
                Default ``(log(max(H, W) / 16), 1.0)``.
            rho_prior_sigma: Sigma of the Gaussian prior on atanh_rho.

        Raises:
            ValueError: On unknown names, overlapping roles, a non-per-component
                mass parameter, missing bounds, or ``n_components < 1``.
        """
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        emulator_param_names = list(emulator_param_names)
        if len(set(emulator_param_names)) != len(emulator_param_names):
            raise ValueError("emulator_param_names contains duplicates")
        fixed_params = dict(fixed_params or {})
        shared_list = list(shared_param_names)
        name_set = set(emulator_param_names)

        unknown_fixed = set(fixed_params) - name_set
        if unknown_fixed:
            raise ValueError(f"fixed_params not in emulator_param_names: {sorted(unknown_fixed)}")
        unknown_shared = set(shared_list) - name_set
        if unknown_shared:
            raise ValueError(
                f"shared_param_names not in emulator_param_names: {sorted(unknown_shared)}"
            )
        overlap = set(fixed_params) & set(shared_list)
        if overlap:
            raise ValueError(f"Parameters both fixed and shared: {sorted(overlap)}")
        if mass_param not in name_set:
            raise ValueError(f"mass_param {mass_param!r} not in emulator_param_names")
        if mass_param in fixed_params or mass_param in shared_list:
            raise ValueError(f"mass_param {mass_param!r} must be per-component (not fixed/shared)")

        shared_set = set(shared_list)
        # Roles, all in emulator order
        self._free_names = [
            p for p in emulator_param_names if p not in fixed_params and p not in shared_set
        ]
        self._shared_names = [p for p in emulator_param_names if p in shared_set]
        self._fixed_names = [p for p in emulator_param_names if p in fixed_params]

        missing = [p for p in self._free_names + self._shared_names if p not in param_bounds]
        if missing:
            raise ValueError(f"param_bounds missing for free parameters: {missing}")

        self.n_components = int(n_components)
        self.emulator_param_names = emulator_param_names
        self.param_bounds = param_bounds
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.fixed_params = fixed_params
        self.mass_param = mass_param
        self.sps_log_prior = sps_log_prior

        self._n_free = len(self._free_names)
        self._n_shared = len(self._shared_names)
        self._n_fixed = len(self._fixed_names)
        self._n_emulator = len(emulator_param_names)
        self._block = 5 + self._n_free
        self._n_params = self.n_components * self._block + self._n_shared
        self.mass_index = self._free_names.index(mass_param)

        # Permutation: emulator column j -> column in concat([free, shared, fixed])
        concat_order = self._free_names + self._shared_names + self._fixed_names
        self._perm = jnp.array(
            [concat_order.index(p) for p in emulator_param_names], dtype=jnp.int32
        )

        f32 = jnp.float32
        self._free_lows = jnp.array([param_bounds[p][0] for p in self._free_names], dtype=f32)
        self._free_highs = jnp.array([param_bounds[p][1] for p in self._free_names], dtype=f32)
        self._shared_lows = jnp.array([param_bounds[p][0] for p in self._shared_names], dtype=f32)
        self._shared_highs = jnp.array([param_bounds[p][1] for p in self._shared_names], dtype=f32)
        self._fixed_vals = jnp.array([fixed_params[p] for p in self._fixed_names], dtype=f32)

        # Pixel coordinate grid (row, col), static
        H, W = self.image_shape
        yy, xx = jnp.meshgrid(jnp.arange(H, dtype=f32), jnp.arange(W, dtype=f32), indexing="ij")
        self._yy = yy.ravel()  # (H*W,)
        self._xx = xx.ravel()  # (H*W,)

        # Shape-prior hyper-parameters
        size = float(max(H, W))
        self.centre = jnp.array([(H - 1) / 2.0, (W - 1) / 2.0], dtype=f32)
        self.centre_prior_sigma = (
            float(centre_prior_sigma) if centre_prior_sigma is not None else size / 4.0
        )
        if log_size_prior is None:
            log_size_prior = (math.log(size / 16.0), 1.0)
        self.log_size_prior = (float(log_size_prior[0]), float(log_size_prior[1]))
        self.rho_prior_sigma = float(rho_prior_sigma)

    # ------------------------------------------------------------------ props
    @property
    def n_params(self) -> int:
        """Total free parameters: ``K * (5 + N_free) + N_shared``."""
        return self._n_params

    @property
    def n_params_per_component(self) -> int:
        """Free parameters per component block: ``5 + N_free``."""
        return self._block

    @property
    def sps_param_names(self) -> list[str]:
        """Per-component free SPS parameter names, in emulator order."""
        return list(self._free_names)

    @property
    def shared_param_names(self) -> list[str]:
        """Shared free SPS parameter names, in emulator order."""
        return list(self._shared_names)

    @property
    def n_free(self) -> int:
        """Number of per-component free SPS parameters."""
        return self._n_free

    @property
    def n_shared(self) -> int:
        """Number of shared free SPS parameters."""
        return self._n_shared

    # ---------------------------------------------------------------- layout
    def split_theta(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Split theta into component blocks and shared raw values.

        Args:
            theta: Flat vector of shape (n_params,).

        Returns:
            ``(blocks, shared_raw)`` with shapes ``(K, 5 + N_free)`` and ``(N_shared,)``.
        """
        n_block = self.n_components * self._block
        blocks = theta[:n_block].reshape(self.n_components, self._block)
        shared_raw = theta[n_block:]
        return blocks, shared_raw

    def join_theta(self, blocks: jnp.ndarray, shared_raw: jnp.ndarray) -> jnp.ndarray:
        """Inverse of :meth:`split_theta`.

        Args:
            blocks: Component blocks of shape (K, 5 + N_free).
            shared_raw: Shared raw values of shape (N_shared,).

        Returns:
            Flat theta of shape (n_params,).
        """
        return jnp.concatenate([blocks.reshape(-1), jnp.reshape(shared_raw, (-1,))])

    # -------------------------------------------------------------- physical
    def component_params(
        self, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Map theta to physical component parameters.

        Args:
            theta: Flat vector of shape (n_params,).

        Returns:
            ``(mu, sigma, rho, sps_phys)``:

            - ``mu`` (K, 2): centres in pixel (row, col) coordinates.
            - ``sigma`` (K, 2): standard deviations (sigma_y, sigma_x) in pixels.
            - ``rho`` (K,): correlation coefficients, clipped to ±0.99.
            - ``sps_phys`` (K, N_emulator): FULL physical emulator input rows in
              ``emulator_param_names`` order, with fixed and shared values filled in.
        """
        blocks, shared_raw = self.split_theta(theta)
        K = self.n_components
        mu = blocks[:, 0:2]
        sigma = jnp.exp(blocks[:, 2:4])
        rho = jnp.clip(jnp.tanh(blocks[:, 4]), -_RHO_MAX, _RHO_MAX)
        free_phys = self._free_lows + (self._free_highs - self._free_lows) * jax.nn.sigmoid(
            blocks[:, 5:]
        )  # (K, N_free)
        shared_phys = self._shared_lows + (self._shared_highs - self._shared_lows) * jax.nn.sigmoid(
            shared_raw
        )  # (N_shared,)
        concat = jnp.concatenate(
            [
                free_phys,
                jnp.broadcast_to(shared_phys, (K, self._n_shared)),
                jnp.broadcast_to(self._fixed_vals, (K, self._n_fixed)),
            ],
            axis=1,
        )  # (K, N_emulator) in [free, shared, fixed] order
        sps_phys = concat[:, self._perm]  # emulator order
        return mu, sigma, rho, sps_phys

    def _log_profiles_unnorm(
        self, mu: jnp.ndarray, sigma: jnp.ndarray, rho: jnp.ndarray
    ) -> jnp.ndarray:
        """Log of unnormalised bivariate Gaussians at every pixel, shape (K, H*W)."""
        dy = (self._yy[None, :] - mu[:, 0:1]) / sigma[:, 0:1]  # (K, H*W)
        dx = (self._xx[None, :] - mu[:, 1:2]) / sigma[:, 1:2]
        r = rho[:, None]
        z = dy**2 - 2.0 * r * dy * dx + dx**2
        return -z / (2.0 * (1.0 - r**2))

    def profiles(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Normalised surface-brightness profiles, shape (K, H, W).

        Each component's profile sums to 1 over the image.  Normalisation is
        done in the log domain (``log_p - logsumexp(log_p)``), which is
        shift-invariant, so a component far outside the frame still yields a
        finite, normalised profile instead of 0/0.

        Args:
            theta: Flat vector of shape (n_params,).

        Returns:
            Array of shape (K, H, W), each slice summing to 1.
        """
        mu, sigma, rho, _ = self.component_params(theta)
        return self._profiles_from_shape(mu, sigma, rho)

    def _log_profiles(self, mu: jnp.ndarray, sigma: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
        """Log of the normalised profiles, shape (K, H*W); each row logsumexps to 0."""
        log_p = self._log_profiles_unnorm(mu, sigma, rho)  # (K, H*W)
        return log_p - jax.scipy.special.logsumexp(log_p, axis=1, keepdims=True)

    def _profiles_from_shape(
        self, mu: jnp.ndarray, sigma: jnp.ndarray, rho: jnp.ndarray
    ) -> jnp.ndarray:
        H, W = self.image_shape
        return jnp.exp(self._log_profiles(mu, sigma, rho)).reshape(self.n_components, H, W)

    def component_seds(self, theta: jnp.ndarray, emulator: SPSEmulator) -> jnp.ndarray:
        """Per-component SEDs at each component's own total mass, shape (K, N_bands).

        Exactly K emulator evaluations; exact for any emulator (no linearity
        in mass assumed).

        Args:
            theta: Flat vector of shape (n_params,).
            emulator: SPS emulator with ``predict((K, N_emulator)) -> (K, N_bands)``.

        Returns:
            Array of shape (K, N_bands) in nJy.
        """
        _, _, _, sps_phys = self.component_params(theta)
        return emulator.predict(sps_phys)

    def model_image(
        self, theta: jnp.ndarray, emulator: SPSEmulator, image_shape: tuple
    ) -> jnp.ndarray:
        """Unconvolved model image ``Σ_k F_k ⊗ P_k``, shape (N_bands, H, W).

        Args:
            theta: Flat vector of shape (n_params,).
            emulator: SPS emulator.
            image_shape: (H, W); must equal ``self.image_shape``.

        Returns:
            Array of shape (N_bands, H, W) in nJy.
        """
        mu, sigma, rho, sps_phys = self.component_params(theta)
        profiles = self._profiles_from_shape(mu, sigma, rho)  # (K, H, W)
        seds = emulator.predict(sps_phys)  # (K, N_bands)
        return jnp.einsum("kb,khw->bhw", seds, profiles)

    # ----------------------------------------------------------- summary map
    def decode(self, theta: jnp.ndarray, image_shape: tuple) -> jnp.ndarray:
        """Per-pixel summary map of the per-component SPS parameters, (H*W, N_free).

        **This is a summary for plotting and ``NUTSResult.get_parameter_map``;
        the likelihood never uses it** (see :meth:`model_image`).

        Column ``c`` (for ``c != mass_index``) is the stellar-mass-weighted mean
        of the components' per-component parameter ``c``::

            w_k(y, x) = P_k(y, x) 10**logM_k / Σ_j P_j(y, x) 10**logM_j
            map_c(y, x) = Σ_k w_k(y, x) sps_kc

        The mass column is replaced by log10 of the stellar-mass surface density
        per pixel, ``log10(Σ_k P_k(y, x) 10**logM_k)`` in Msun per pixel.

        Args:
            theta: Flat vector of shape (n_params,).
            image_shape: (H, W); must equal ``self.image_shape``.

        Returns:
            Array of shape (H*W, N_free), columns in ``sps_param_names`` order.
        """
        mu, sigma, rho, _ = self.component_params(theta)
        blocks, _ = self.split_theta(theta)
        free_phys = self._free_lows + (self._free_highs - self._free_lows) * jax.nn.sigmoid(
            blocks[:, 5:]
        )  # (K, N_free)
        log_profiles = self._log_profiles(mu, sigma, rho)  # (K, H*W)
        # log of stellar-mass surface density of each component at each pixel
        log_density = log_profiles + (free_phys[:, self.mass_index] * _LN10)[:, None]
        # Mass weights: softmax over components is exact even where every
        # component has underflowed, so the summary map never leaves bounds.
        weights = jax.nn.softmax(log_density, axis=0)  # (K, H*W)
        summary = jnp.einsum("kp,kc->pc", weights, free_phys)  # (H*W, N_free)
        log10_total = jax.scipy.special.logsumexp(log_density, axis=0) / _LN10  # (H*W,)
        return summary.at[:, self.mass_index].set(log10_total)

    # ------------------------------------------------------------------ prior
    def _jacobian_term(self, blocks: jnp.ndarray, shared_raw: jnp.ndarray) -> jnp.ndarray:
        """Sigmoid log-Jacobian summed over per-component and shared SPS raws."""
        free_raw = blocks[:, 5:]
        # log_sigmoid(r) + log_sigmoid(-r) is the exact log-density of
        # Uniform(lo, hi) pulled back through phys = lo + (hi-lo)*sigmoid(r):
        # the 1/(hi-lo) of the uniform cancels the (hi-lo) of dphys/dr.  No
        # log(hi-lo) term, so the prior is properly normalised and logZ values
        # from nested sampling are comparable across models with different K.
        jac_free = jnp.sum(jax.nn.log_sigmoid(free_raw) + jax.nn.log_sigmoid(-free_raw))
        jac_shared = jnp.sum(jax.nn.log_sigmoid(shared_raw) + jax.nn.log_sigmoid(-shared_raw))
        return jac_free + jac_shared

    def _shape_log_prior(self, blocks: jnp.ndarray) -> jnp.ndarray:
        """Normalised Gaussian log-densities on mu, log_sigma and atanh_rho."""
        mu = blocks[:, 0:2]
        log_sigma = blocks[:, 2:4]
        atanh_rho = blocks[:, 4]
        ls_mu, ls_sd = self.log_size_prior

        def normal_logpdf(x: jnp.ndarray, loc: float | jnp.ndarray, sd: float) -> jnp.ndarray:
            return -0.5 * ((x - loc) / sd) ** 2 - math.log(sd) - 0.5 * _LOG_2PI

        lp_mu = jnp.sum(normal_logpdf(mu, self.centre[None, :], self.centre_prior_sigma))
        lp_size = jnp.sum(normal_logpdf(log_sigma, ls_mu, ls_sd))
        lp_rho = jnp.sum(normal_logpdf(atanh_rho, 0.0, self.rho_prior_sigma))
        return lp_mu + lp_size + lp_rho

    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Scalar log-prior of theta (see class docstring for the terms).

        Args:
            theta: Flat vector of shape (n_params,).

        Returns:
            Scalar log-prior, ``jax.grad``-differentiable.
        """
        blocks, shared_raw = self.split_theta(theta)
        lp = self._jacobian_term(blocks, shared_raw) + self._shape_log_prior(blocks)
        if self.sps_log_prior is not None:
            _, _, _, sps_phys = self.component_params(theta)
            lp = lp + self.sps_log_prior(sps_phys)
        return lp

    def sample_prior(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw ``n`` theta vectors from the prior defined by :meth:`log_prior`.

        SPS values are uniform within bounds (mapped to raw via logit); centres,
        log-sizes and atanh_rho are drawn from their Gaussians.  The optional
        ``sps_log_prior`` term is *not* sampled from.

        Args:
            key: ``jax.random`` PRNG key.
            n: Number of samples.

        Returns:
            Array of shape (n, n_params).
        """
        K, B = self.n_components, self._block
        k_mu, k_ls, k_rho, k_free, k_shared = jax.random.split(key, 5)
        ls_mu, ls_sd = self.log_size_prior
        f32 = jnp.float32

        mu = self.centre[None, None, :] + self.centre_prior_sigma * jax.random.normal(
            k_mu, (n, K, 2), dtype=f32
        )
        log_sigma = ls_mu + ls_sd * jax.random.normal(k_ls, (n, K, 2), dtype=f32)
        atanh_rho = self.rho_prior_sigma * jax.random.normal(k_rho, (n, K, 1), dtype=f32)
        # Uniform in physical space == logistic in raw space
        free_raw = jax.random.logistic(k_free, (n, K, self._n_free), dtype=f32)
        shared_raw = jax.random.logistic(k_shared, (n, self._n_shared), dtype=f32)

        blocks = jnp.concatenate([mu, log_sigma, atanh_rho, free_raw], axis=-1)  # (n, K, B)
        return jnp.concatenate([blocks.reshape(n, K * B), shared_raw], axis=1)

    # -------------------------------------------------------------- utilities
    def order_components_by_size(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Permute component blocks so ``log_sigma_y + log_sigma_x`` is ascending.

        Compact components come first.  Pure JAX; used to break label
        symmetry when reporting.

        Args:
            theta: Flat vector of shape (n_params,).

        Returns:
            Reordered theta of shape (n_params,).
        """
        blocks, shared_raw = self.split_theta(theta)
        size = blocks[:, 2] + blocks[:, 3]
        order = jnp.argsort(size)
        return self.join_theta(jnp.take(blocks, order, axis=0), shared_raw)
