"""BlackJAX NUTS sampler with window adaptation for arachne forward models.

Supports multi-chain sampling: ``NUTSSampler.run(..., n_chains=4)`` runs
``n_chains`` independent warmups and chains under ``jax.vmap`` and attaches
convergence diagnostics (split-R-hat, bulk ESS, divergences, tree-depth
saturation) to the returned :class:`NUTSResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.diagnostics import chain_movement, ess, split_rhat
from arachne.spatial.base import SpatialModel
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


@dataclass
class NUTSResult:
    """Container for NUTS posterior samples and diagnostics.

    Attributes:
        samples: Posterior sample array of shape (n_samples_total, n_params),
            with every chain concatenated along axis 0.  Each row is an
            unconstrained theta vector.  For a single chain this is exactly
            the chain, so downstream code that predates multi-chain support
            keeps working.
        infos: BlackJAX NUTS info namedtuple containing per-draw diagnostics
            (acceptance_rate, num_integration_steps, is_divergent, ...).
            Leaves have shape (n_samples,) for a single chain and
            (n_chains, n_samples) for several.
        spatial_model: Reference to the spatial model used for inference
            (needed to decode theta into physical parameters).
        chains: Per-chain samples of shape (n_chains, n_samples, n_params),
            or ``None`` for samplers that do not expose a chain structure.
        diagnostics: Dict of convergence diagnostics; see
            :meth:`NUTSSampler.run` for the keys.
    """

    samples: jnp.ndarray
    infos: Any
    spatial_model: SpatialModel
    chains: Optional[jnp.ndarray] = None
    diagnostics: dict = field(default_factory=dict)

    def to_hdf5(self, path: str | Path) -> None:
        """Serialise the NUTS result to an HDF5 file.

        Saves the raw sample array, the per-chain array (when present) and key
        scalar diagnostics.  The spatial model metadata (param names, bounds)
        is saved as HDF5 attributes.

        Args:
            path: Output HDF5 file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            samples_group = f.create_group("samples")
            samples_group.create_dataset("theta", data=np.asarray(self.samples), compression="gzip")
            if self.chains is not None:
                samples_group.create_dataset(
                    "chains", data=np.asarray(self.chains), compression="gzip"
                )
            # Save key diagnostics if available
            infos_group = f.create_group("diagnostics")
            if hasattr(self.infos, "acceptance_rate"):
                infos_group.create_dataset(
                    "acceptance_rate",
                    data=np.asarray(self.infos.acceptance_rate),
                )
            if hasattr(self.infos, "num_integration_steps"):
                infos_group.create_dataset(
                    "num_integration_steps",
                    data=np.asarray(self.infos.num_integration_steps),
                )
            for key, value in self.diagnostics.items():
                arr = np.asarray(value)
                if arr.dtype.kind in "fiub":
                    infos_group.attrs[f"summary_{key}"] = arr if arr.ndim else arr.item()
            # Save spatial model metadata
            meta = f.create_group("metadata")
            if hasattr(self.spatial_model, "sps_param_names"):
                meta.attrs["param_names"] = np.array(
                    [s.encode("utf-8") for s in self.spatial_model.sps_param_names]
                )
            meta.attrs["n_params"] = self.spatial_model.n_params
            meta.attrs["n_samples"] = self.samples.shape[0]

        logger.info(f"NUTSResult saved to {path}")

    @classmethod
    def from_hdf5(cls, path: str | Path, spatial_model: SpatialModel) -> "NUTSResult":
        """Load a NUTSResult from an HDF5 file.

        Args:
            path: Path to the HDF5 file written by ``to_hdf5``.
            spatial_model: Spatial model instance (needed for ``get_parameter_map``).

        Returns:
            NUTSResult with samples, per-chain array and diagnostics restored.
        """
        with h5py.File(path, "r") as f:
            samples = jnp.array(f["samples/theta"][()])
            chains = None
            if "samples/chains" in f:
                chains = jnp.array(f["samples/chains"][()])
            diagnostics: dict = {}
            if "diagnostics" in f:
                for key, value in f["diagnostics"].attrs.items():
                    if key.startswith("summary_"):
                        diagnostics[key[len("summary_") :]] = np.asarray(value)
        return cls(
            samples=samples,
            infos=None,
            spatial_model=spatial_model,
            chains=chains,
            diagnostics=diagnostics,
        )

    def get_parameter_map(
        self,
        image_shape: tuple[int, int],
        percentiles: list[int] | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Compute posterior parameter maps at the requested percentiles.

        Decodes each sample's theta vector into a physical parameter map and
        computes percentiles over the sample dimension.

        Args:
            image_shape: (H, W) spatial dimensions of the output map.
            percentiles: List of percentiles to compute. Defaults to [16, 50, 84].

        Returns:
            Dict mapping parameter name → array of shape (n_percentiles, H, W).
        """
        if percentiles is None:
            percentiles = [16, 50, 84]

        H, W = image_shape
        n_samples = self.samples.shape[0]
        spatial_model = self.spatial_model
        param_names = getattr(spatial_model, "sps_param_names", [])

        if not param_names:
            logger.warning("spatial_model has no sps_param_names; returning empty parameter map.")
            return {}

        logger.info(f"Decoding {n_samples} samples into parameter maps ({H}×{W})...")

        # Decode all samples: (n_samples, H*W, N_sps)
        def decode_single(theta: jnp.ndarray) -> jnp.ndarray:
            return spatial_model.decode(theta, image_shape)

        all_decoded = jax.vmap(decode_single)(self.samples)  # (n_samples, H*W, N_sps)
        all_maps = all_decoded.reshape(n_samples, H, W, len(param_names))

        # Compute percentiles
        result = {}
        for i, name in enumerate(param_names):
            param_samples = all_maps[:, :, :, i]  # (n_samples, H, W)
            pct_maps = jnp.percentile(
                param_samples, jnp.array(percentiles, dtype=jnp.float32), axis=0
            )  # (n_percentiles, H, W)
            result[name] = pct_maps

        return result

    @property
    def n_samples(self) -> int:
        """Number of posterior samples (all chains concatenated)."""
        return self.samples.shape[0]

    @property
    def n_chains(self) -> int:
        """Number of chains (1 when no per-chain array is stored)."""
        return 1 if self.chains is None else int(self.chains.shape[0])

    @property
    def acceptance_rate(self) -> Optional[float]:
        """Mean acceptance rate during sampling (if available)."""
        if hasattr(self.infos, "acceptance_rate") and self.infos is not None:
            return float(jnp.mean(self.infos.acceptance_rate))
        return None

    def summary(self, max_num_doublings: int | None = None) -> str:
        """One-paragraph human-readable convergence summary.

        Flags the worst split-R-hat, the smallest bulk ESS, the divergence
        count, and NUTS trees that ran into the doubling cap (a sign the step
        size is too small or the posterior has a long, badly conditioned
        direction).  A warning is logged when ``max_rhat > 1.05`` or when the
        mean trajectory length saturates the tree cap.

        Args:
            max_num_doublings: Tree-doubling cap used for the run.  Defaults to
                the value stored in ``diagnostics``.

        Returns:
            A single paragraph of text.
        """
        d = self.diagnostics
        cap = max_num_doublings if max_num_doublings is not None else d.get("max_num_doublings")
        rhat = np.asarray(d.get("rhat", np.array([np.nan])), dtype=float)
        e = np.asarray(d.get("ess", np.array([np.nan])), dtype=float)
        max_rhat = float(np.nanmax(rhat)) if np.any(np.isfinite(rhat)) else float("nan")
        min_ess = float(np.nanmin(e)) if np.any(np.isfinite(e)) else float("nan")
        worst = int(np.nanargmax(rhat)) if np.any(np.isfinite(rhat)) else -1
        n_div = int(np.asarray(d.get("n_divergent", 0)).item())
        depth = float(np.asarray(d.get("mean_tree_depth", np.nan)).item())
        step = np.asarray(d.get("step_size", np.nan), dtype=float)
        acc = float(np.asarray(d.get("acceptance_rate", np.nan)).item())
        moved = float(np.asarray(d.get("fraction_dims_moved", np.nan)).item())

        saturated = cap is not None and np.isfinite(depth) and depth >= 2 ** int(cap) - 1
        parts = [
            f"{self.n_chains} chain(s) x {self.samples.shape[0] // max(self.n_chains, 1)} draws, "
            f"{self.samples.shape[1]} parameters.",
            f"max split-R-hat = {max_rhat:.3f}" + (f" (param {worst})" if worst >= 0 else ""),
            f"min bulk ESS = {min_ess:.0f}",
            f"{n_div} divergent transition(s)",
            f"mean trajectory length = {depth:.1f} leapfrog steps"
            + (f" (cap {2 ** int(cap) - 1})" if cap is not None else ""),
            f"mean acceptance = {acc:.3f}",
            f"step size = {np.array2string(step, precision=4)}",
            f"{moved:.0%} of dimensions moved",
        ]
        text = parts[0] + " " + "; ".join(parts[1:]) + "."
        flags = []
        if np.isfinite(max_rhat) and max_rhat > 1.05:
            flags.append(f"max R-hat = {max_rhat:.3f} > 1.05 — chains have NOT converged")
        if saturated:
            flags.append(
                f"mean trajectory length {depth:.1f} saturates the "
                f"max_num_doublings={cap} tree cap — raise it or improve the metric"
            )
        if n_div > 0:
            flags.append(f"{n_div} divergent transitions — reparameterise or raise target_accept")
        for flag in flags:
            logger.warning(flag)
        if flags:
            text += "  WARNING: " + "; ".join(flags) + "."
        return text


def _jitter_inits(
    theta_init: jnp.ndarray,
    n_chains: int,
    chain_jitter: float,
    inverse_mass_matrix: jnp.ndarray | None,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Replicate ``theta_init`` across chains and add geometry-aware jitter.

    Chain 0 is left at ``theta_init`` exactly (so a single-chain run and the
    first chain of a multi-chain run start identically); chains 1..n-1 are
    offset by ``chain_jitter * sqrt(diag(inverse_mass_matrix)) * N(0, 1)``,
    falling back to an isotropic ``chain_jitter * N(0, 1)`` when no mass
    matrix is available.  Scaling by the metric matters because the posterior
    width differs by orders of magnitude between, say, a log-mass and a
    position offset — an isotropic jitter either does nothing or throws
    chains off the typical set.  Ported from ``scripts/fit_catalogue.py``.

    Args:
        theta_init: Shape (n_params,) or (n_chains, n_params).
        n_chains: Number of chains.
        chain_jitter: Jitter scale in units of the posterior standard deviation.
        inverse_mass_matrix: Diagonal (n_params,) or dense (n_params, n_params)
            inverse mass matrix, or ``None``.
        key: PRNG key.

    Returns:
        Array of shape (n_chains, n_params).

    Raises:
        ValueError: If ``theta_init`` is 2-D with the wrong number of rows.
    """
    theta_init = jnp.atleast_1d(jnp.asarray(theta_init))
    if theta_init.ndim == 2:
        if theta_init.shape[0] != n_chains:
            raise ValueError(
                f"theta_init has {theta_init.shape[0]} rows but n_chains={n_chains}; "
                "pass a (n_params,) vector or a (n_chains, n_params) array"
            )
        return theta_init
    if theta_init.ndim != 1:
        raise ValueError(f"theta_init must be 1-D or 2-D; got shape {theta_init.shape}")

    inits = jnp.tile(theta_init[None, :], (n_chains, 1))
    if n_chains == 1 or chain_jitter <= 0.0:
        return inits

    d = theta_init.shape[0]
    if inverse_mass_matrix is None:
        scale = jnp.ones(d)
    else:
        imm = jnp.asarray(inverse_mass_matrix)
        diag = jnp.diagonal(imm) if imm.ndim == 2 else imm
        scale = jnp.sqrt(jnp.clip(diag, 1e-12, None))
    noise = jax.random.normal(key, (n_chains, d)) * chain_jitter * scale[None, :]
    noise = noise.at[0, :].set(0.0)
    return inits + noise


class NUTSSampler:
    """BlackJAX NUTS sampler with Stan-style dual-averaging warmup.

    Uses ``blackjax.window_adaptation`` for warmup (step size + mass matrix
    adaptation) followed by a ``jax.lax.scan`` sampling loop for full GPU
    efficiency.  With ``n_chains > 1`` both stages are wrapped in
    ``jax.vmap``, so every chain adapts its own step size and mass matrix and
    all chains are compiled into one XLA program.

    The sampling loop uses ``jax.lax.scan`` rather than a Python for loop
    so that the entire chain is compiled into a single XLA computation.
    This avoids Python overhead between steps and allows the GPU to run at
    full utilisation.

    **Important**: create a new ``NUTSSampler`` for each new model or image
    shape.  The XLA-compiled graph is tied to the shape of ``theta_init``
    and the forward model's structure — a different shape will trigger a
    recompile.

    Attributes:
        forward_model: ForwardModel whose ``log_posterior`` is sampled.
        n_warmup: Number of warmup steps for mass matrix adaptation.
        n_samples: Number of posterior samples to draw *per chain*.
        target_accept_rate: Target acceptance rate for dual averaging.
        max_num_doublings: Maximum number of NUTS tree doublings (caps memory use).
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        n_warmup: int = 500,
        n_samples: int = 1000,
        target_accept_rate: float = 0.8,
        max_num_doublings: int = 5,
        dense_mass_matrix: bool = False,
    ) -> None:
        """Initialise the NUTS sampler.

        Args:
            forward_model: Assembled ForwardModel, or any object exposing
                ``log_posterior(theta)`` and ``spatial_model`` (e.g.
                :class:`arachne.inference.laplace.WhitenedLogDensity`).
            n_warmup: Number of warmup (adaptation) steps. 500 is typically
                sufficient for well-conditioned models; increase to 1000+ for
                large pixel maps.
            n_samples: Number of posterior samples to collect after warmup,
                per chain.
            target_accept_rate: Dual-averaging target acceptance rate.
                0.8 is the default for NUTS (Stan convention).
            max_num_doublings: Maximum NUTS tree depth. Caps GPU memory use.
                Set to 5 for large pixel maps to avoid OOM errors.
            dense_mass_matrix: Adapt a **full** ``(d, d)`` inverse mass matrix
                instead of a diagonal one.  A diagonal metric only rescales
                the coordinate axes, so it cannot help a posterior whose stiff
                directions are correlated (the usual case for resolved fits:
                component centre against size against mass against the
                per-band registration offsets).  The dense metric costs
                ``d^2`` memory and one triangular solve per leapfrog step --
                negligible next to a 1e5-pixel likelihood -- but estimating
                ``d(d+1)/2`` numbers needs a longer warmup (>= 1000 steps for
                ``d ~ 60``).  It is most effective on an already whitened
                target; see :mod:`arachne.inference.laplace`.
        """
        self.forward_model = forward_model
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.target_accept_rate = target_accept_rate
        self.max_num_doublings = max_num_doublings
        self.dense_mass_matrix = dense_mass_matrix

    def _warmup_fn(self, logpost: Callable, inverse_mass_matrix: jnp.ndarray | None) -> Callable:
        """Build a ``(key, position) -> (state, step_size, inv_mass)`` warmup closure."""
        import blackjax

        kwargs: dict[str, Any] = {}
        if inverse_mass_matrix is not None:
            kwargs["initial_inverse_mass_matrix"] = jnp.asarray(inverse_mass_matrix)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logpost,
            is_mass_matrix_diagonal=not self.dense_mass_matrix,
            target_acceptance_rate=self.target_accept_rate,
            max_num_doublings=self.max_num_doublings,
            **kwargs,
        )

        def run_one(key: jnp.ndarray, position: jnp.ndarray):
            (state, params), _info = warmup.run(key, position, self.n_warmup)
            return state, params["step_size"], params["inverse_mass_matrix"]

        return run_one

    def run(
        self,
        theta_init: jnp.ndarray,
        rng_key: jnp.ndarray,
        n_chains: int = 1,
        chain_jitter: float = 0.0,
        inverse_mass_matrix: jnp.ndarray | None = None,
    ) -> NUTSResult:
        """Run NUTS warmup and sampling, optionally with several chains.

        Warmup is run **per chain**: ``blackjax.window_adaptation.run`` is
        vmapped over ``(key, position)`` so each chain gets its own step size
        and diagonal mass matrix.  This is exact here because
        ``window_adaptation.run`` is a ``lax.scan`` over a schedule built from
        the Python integer ``num_steps``, so nothing inside it depends on a
        traced value.  Should a future blackjax break that (the call raises),
        the sampler falls back to adapting **chain 0 only** and reusing its
        step size and mass matrix for every chain; a warning is logged and
        ``diagnostics["warmup_per_chain"]`` records which path was taken.

        Args:
            theta_init: Initial parameter vector of shape (n_params,), which is
                replicated and jittered across chains, or an explicit
                (n_chains, n_params) array of per-chain starting points (no
                jitter is added in that case).  A good starting point is the
                MAP estimate or a zero vector (the centre of the sigmoid bounds).
            rng_key: JAX random key for reproducible sampling.
            n_chains: Number of independent chains.  Four or more are needed
                for split-R-hat to mean anything.
            chain_jitter: Scale of the per-chain starting-point jitter, in
                units of the posterior standard deviation implied by
                ``inverse_mass_matrix`` (isotropic if none is given).  Chain 0
                always starts exactly at ``theta_init``.
            inverse_mass_matrix: Optional initial inverse mass matrix
                (diagonal ``(n_params,)`` or dense), e.g. from Pathfinder.  It
                seeds the window adaptation and sets the jitter geometry.

        Returns:
            NUTSResult whose ``samples`` are all chains concatenated
            ``(n_chains * n_samples, n_params)``, ``chains`` is
            ``(n_chains, n_samples, n_params)``, and ``diagnostics`` holds
            ``rhat``, ``ess``, ``n_divergent``, ``mean_tree_depth``,
            ``step_size``, ``acceptance_rate``, ``fraction_dims_moved``,
            ``warmup_per_chain`` and ``max_num_doublings``.

        Raises:
            ImportError: If ``blackjax`` is not installed.
            ValueError: If ``n_chains < 1`` or ``theta_init`` has a bad shape.
        """
        try:
            import blackjax
        except ImportError as e:
            raise ImportError(
                "blackjax is required for NUTS sampling. Install it with: pip install blackjax"
            ) from e

        if n_chains < 1:
            raise ValueError(f"n_chains must be >= 1; got {n_chains}")

        # JIT-compile the log-posterior once
        logpost = jax.jit(self.forward_model.log_posterior)

        rng_key, jitter_key = jax.random.split(rng_key)
        inits = _jitter_inits(
            theta_init, n_chains, chain_jitter, inverse_mass_matrix, jitter_key
        )  # (n_chains, d)
        n_params = int(inits.shape[1])
        logger.info(
            f"Starting NUTS: {n_chains} chain(s), {self.n_warmup} warmup + "
            f"{self.n_samples} samples each, {n_params} parameters"
        )

        # 1. Warmup: window adaptation (step size + diagonal mass matrix) per chain
        run_one_warmup = self._warmup_fn(logpost, inverse_mass_matrix)
        rng_key, warmup_key = jax.random.split(rng_key)
        warmup_keys = jax.random.split(warmup_key, n_chains)
        warmup_per_chain = True
        try:
            states, step_sizes, inv_masses = jax.vmap(run_one_warmup)(warmup_keys, inits)
        except Exception as exc:  # pragma: no cover - blackjax-version dependent
            warmup_per_chain = False
            logger.warning(
                f"vmapped window_adaptation failed ({type(exc).__name__}: {exc}); "
                "adapting chain 0 only and sharing its step size / mass matrix."
            )
            state0, step0, inv0 = run_one_warmup(warmup_keys[0], inits[0])
            step_sizes = jnp.full((n_chains,), step0)
            inv_masses = jnp.tile(jnp.asarray(inv0)[None, ...], (n_chains,) + (1,) * inv0.ndim)
            init_kernel = blackjax.nuts(
                logpost,
                step_size=step0,
                inverse_mass_matrix=inv0,
                max_num_doublings=self.max_num_doublings,
            )
            states = jax.vmap(init_kernel.init)(inits)
            del state0
        logger.info(
            f"Warmup complete ({'per-chain' if warmup_per_chain else 'shared'}). "
            f"Step sizes: {np.array2string(np.asarray(step_sizes), precision=4)}"
        )

        # 2/3. Build the NUTS kernel with each chain's adapted parameters and
        #      run the lax.scan sampling loop, vmapped over chains.
        def sample_one(key: jnp.ndarray, state: Any, step_size: jnp.ndarray, inv_mass: jnp.ndarray):
            kernel = blackjax.nuts(
                logpost,
                step_size=step_size,
                inverse_mass_matrix=inv_mass,
                max_num_doublings=self.max_num_doublings,
            )

            def one_step(carry: Any, k: jnp.ndarray) -> tuple[Any, tuple[jnp.ndarray, Any]]:
                new_state, info = kernel.step(k, carry)
                return new_state, (new_state.position, info)

            keys = jax.random.split(key, self.n_samples)
            _, out = jax.lax.scan(one_step, state, keys)
            return out

        rng_key, sample_key = jax.random.split(rng_key)
        sample_keys = jax.random.split(sample_key, n_chains)
        chains, infos = jax.vmap(sample_one)(sample_keys, states, step_sizes, inv_masses)
        chains = jnp.asarray(chains)  # (n_chains, n_samples, d)
        samples = chains.reshape(-1, n_params)

        diagnostics = self._diagnostics(chains, infos, step_sizes, warmup_per_chain)
        if n_chains == 1:
            # Preserve the pre-multi-chain info layout for single-chain callers.
            infos = jax.tree_util.tree_map(lambda x: x[0], infos)

        result = NUTSResult(
            samples=samples,
            infos=infos,
            spatial_model=self.forward_model.spatial_model,
            chains=chains,
            diagnostics=diagnostics,
        )
        logger.info(result.summary(max_num_doublings=self.max_num_doublings))
        return result

    def _diagnostics(
        self,
        chains: jnp.ndarray,
        infos: Any,
        step_sizes: jnp.ndarray,
        warmup_per_chain: bool,
    ) -> dict:
        """Assemble the diagnostics dict from the per-chain samples and infos.

        ``mean_tree_depth`` is the mean of ``num_integration_steps``, i.e. the
        mean number of leapfrog steps per draw.  A NUTS tree of depth ``j``
        takes ``2**j - 1`` steps, so a mean at or above
        ``2**max_num_doublings - 1`` means essentially every trajectory hit the
        doubling cap.

        Args:
            chains: (n_chains, n_samples, n_params) samples.
            infos: Vmapped NUTS info pytree with a leading chain axis.
            step_sizes: (n_chains,) adapted step sizes.
            warmup_per_chain: Whether warmup was run separately per chain.

        Returns:
            Diagnostics dict (see :meth:`run`).
        """
        chains_np = np.asarray(chains)
        n_div = 0
        if hasattr(infos, "is_divergent"):
            n_div = int(np.sum(np.asarray(infos.is_divergent)))
        mean_depth = float("nan")
        if hasattr(infos, "num_integration_steps"):
            mean_depth = float(np.mean(np.asarray(infos.num_integration_steps)))
        accept = float("nan")
        if hasattr(infos, "acceptance_rate"):
            accept = float(np.mean(np.asarray(infos.acceptance_rate)))
        return {
            "rhat": split_rhat(chains_np),
            "ess": ess(chains_np),
            "n_divergent": n_div,
            "mean_tree_depth": mean_depth,
            "step_size": np.asarray(step_sizes, dtype=np.float64),
            "acceptance_rate": accept,
            "fraction_dims_moved": chain_movement(chains_np),
            "warmup_per_chain": bool(warmup_per_chain),
            "max_num_doublings": int(self.max_num_doublings),
            "dense_mass_matrix": bool(self.dense_mass_matrix),
        }
