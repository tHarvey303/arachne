"""BlackJAX NUTS sampler with window adaptation for arachne forward models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from arachne.forward_model.pipeline import ForwardModel
from arachne.spatial.base import SpatialModel
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


@dataclass
class NUTSResult:
    """Container for NUTS posterior samples and diagnostics.

    Attributes:
        samples: Posterior sample array of shape (n_samples, n_params).
            Each row is an unconstrained theta vector.
        infos: BlackJAX NUTS info namedtuple containing diagnostics
            (acceptance_rate, num_integration_steps, etc.).
        spatial_model: Reference to the spatial model used for inference
            (needed to decode theta into physical parameters).
    """

    samples: jnp.ndarray
    infos: Any
    spatial_model: SpatialModel

    def to_hdf5(self, path: str | Path) -> None:
        """Serialise the NUTS result to an HDF5 file.

        Saves the raw sample array and key scalar diagnostics. The spatial
        model metadata (param names, bounds) is saved as HDF5 attributes.

        Args:
            path: Output HDF5 file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            samples_group = f.create_group("samples")
            samples_group.create_dataset(
                "theta", data=np.asarray(self.samples), compression="gzip"
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
            # Save spatial model metadata
            meta = f.create_group("metadata")
            if hasattr(self.spatial_model, "sps_param_names"):
                meta.attrs["param_names"] = np.bytes_(self.spatial_model.sps_param_names)
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
            NUTSResult with samples loaded from disk.
        """
        with h5py.File(path, "r") as f:
            samples = jnp.array(f["samples/theta"][()])
        return cls(samples=samples, infos=None, spatial_model=spatial_model)

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
            logger.warning(
                "spatial_model has no sps_param_names; returning empty parameter map."
            )
            return {}

        logger.info(
            f"Decoding {n_samples} samples into parameter maps ({H}×{W})..."
        )

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
        """Number of posterior samples."""
        return self.samples.shape[0]

    @property
    def acceptance_rate(self) -> Optional[float]:
        """Mean acceptance rate during sampling (if available)."""
        if hasattr(self.infos, "acceptance_rate") and self.infos is not None:
            return float(jnp.mean(self.infos.acceptance_rate))
        return None


class NUTSSampler:
    """BlackJAX NUTS sampler with Stan-style dual-averaging warmup.

    Uses ``blackjax.window_adaptation`` for warmup (step size + mass matrix
    adaptation) followed by a ``jax.lax.scan`` sampling loop for full GPU
    efficiency.

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
        n_samples: Number of posterior samples to draw.
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
    ) -> None:
        """Initialise the NUTS sampler.

        Args:
            forward_model: Assembled ForwardModel.
            n_warmup: Number of warmup (adaptation) steps. 500 is typically
                sufficient for well-conditioned models; increase to 1000+ for
                large pixel maps.
            n_samples: Number of posterior samples to collect after warmup.
            target_accept_rate: Dual-averaging target acceptance rate.
                0.8 is the default for NUTS (Stan convention).
            max_num_doublings: Maximum NUTS tree depth. Caps GPU memory use.
                Set to 5 for large pixel maps to avoid OOM errors.
        """
        self.forward_model = forward_model
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.target_accept_rate = target_accept_rate
        self.max_num_doublings = max_num_doublings

    def run(
        self,
        theta_init: jnp.ndarray,
        rng_key: jnp.ndarray,
    ) -> NUTSResult:
        """Run NUTS warmup and sampling.

        Args:
            theta_init: Initial parameter vector of shape (n_params,).
                A good starting point is the MAP estimate or a zero vector
                (which maps to the centre of the sigmoid bounds).
            rng_key: JAX random key for reproducible sampling.

        Returns:
            NUTSResult containing posterior samples and diagnostics.
        """
        try:
            import blackjax
        except ImportError as e:
            raise ImportError(
                "blackjax is required for NUTS sampling. "
                "Install it with: pip install blackjax"
            ) from e

        # JIT-compile the log-posterior once
        logpost = jax.jit(self.forward_model.log_posterior)
        logger.info(
            f"Starting NUTS: {self.n_warmup} warmup + {self.n_samples} samples, "
            f"theta_init shape {theta_init.shape}"
        )

        # 1. Warmup: window adaptation (step size + diagonal mass matrix)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logpost,
            target_acceptance_rate=self.target_accept_rate,
        )
        rng_key, warmup_key = jax.random.split(rng_key)
        (state, params), warmup_info = warmup.run(warmup_key, theta_init, self.n_warmup)
        logger.info(
            f"Warmup complete. Step size: {params.get('step_size', 'N/A'):.4g}"
        )

        # 2. Build NUTS kernel with adapted parameters
        nuts_kernel = blackjax.nuts(
            logpost,
            max_num_doublings=self.max_num_doublings,
            **params,
        )

        # 3. Sampling: jax.lax.scan loop for full GPU efficiency
        def one_step(
            carry: Any, rng_key: jnp.ndarray
        ) -> tuple[Any, tuple[jnp.ndarray, Any]]:
            state, info = nuts_kernel.step(rng_key, carry)
            return state, (state.position, info)

        rng_key, sample_key = jax.random.split(rng_key)
        sample_keys = jax.random.split(sample_key, self.n_samples)
        final_state, (samples, infos) = jax.lax.scan(one_step, state, sample_keys)

        accept_rate = float(jnp.mean(infos.acceptance_rate))
        logger.info(
            f"Sampling complete. Mean acceptance rate: {accept_rate:.3f}. "
            f"Samples shape: {samples.shape}"
        )

        return NUTSResult(
            samples=samples,
            infos=infos,
            spatial_model=self.forward_model.spatial_model,
        )
