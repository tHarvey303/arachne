"""JAX/Equinox reimplementation of synference normalising flow emulators.

.. deprecated::
    ``JAXFlowEmulator`` is the **legacy** emulator path.  The preferred
    approach is :class:`~arachne.emulator.jax_mlp_emulator.SPSMLPEmulator`,
    which trains a native JAX/Equinox MLP (Alsing et al. 2020 Speculator
    architecture) directly from a synference HDF5 model library.  This avoids
    the fragile PyTorch weight-export and gives a fully JAX-native pipeline.

    Use ``JAXFlowEmulator`` only if you have a pre-trained lampe flow checkpoint
    and cannot regenerate the library.

Design note — emulator direction
---------------------------------
synference trains ``p(params | photometry)`` (NPE posterior, params are the
*output* of the flow). arachne needs the **forward** direction:
``photometry = f(params)``.

``JAXFlowEmulator.from_synference_checkpoint`` expects a checkpoint where the
flow maps *params → photometry*.  If you load a checkpoint trained in the
standard NPE direction you will obtain wrong predictions — the ``direction``
attribute documents which was used.

Weight export strategy
-----------------------
DLPack / torch2jax cannot propagate JAX gradients through the PyTorch boundary,
which would break BlackJAX NUTS.  The only viable strategy is **one-time weight
export**: load the ``lampe`` checkpoint, convert every tensor to numpy, and
reconstruct an equivalent Equinox pytree.  At inference time no PyTorch code
runs — the entire forward pass is pure JAX.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from arachne.emulator.base import SPSEmulator
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


# ---------------------------------------------------------------------------
# Low-level Equinox modules mirroring lampe MAF / NSF building blocks
# ---------------------------------------------------------------------------


class MaskedLinear(eqx.Module):
    """Linear layer with a binary connection mask (used in MAF autoregressive layers).

    The mask is stored as a static (non-trainable) field so that JAX's
    tracing machinery does not treat it as a gradient leaf.

    Attributes:
        weight: Weight matrix of shape (out_features, in_features).
        bias: Bias vector of shape (out_features,).
        mask: Static binary mask of shape (out_features, in_features).
    """

    weight: jnp.ndarray
    bias: jnp.ndarray
    mask: jnp.ndarray = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply masked linear transformation.

        Args:
            x: Input array of shape (..., in_features).

        Returns:
            Output array of shape (..., out_features).
        """
        return x @ (self.weight * self.mask).T + self.bias


class _FCNN(eqx.Module):
    """Fully-connected neural network used as the conditioner in MAF.

    Attributes:
        layers: List of linear layers (no mask needed for the conditioner).
    """

    layers: list

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the conditioner network.

        Args:
            x: Input array.

        Returns:
            Output array.
        """
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


class MAFTransform(eqx.Module):
    """Single masked autoregressive transform.

    Implements an affine autoregressive transform:
    ``z_i = (x_i - mu_i) / exp(log_scale_i)``
    where (mu, log_scale) are computed by the masked conditioner network.

    Attributes:
        conditioner: _FCNN that outputs (mu, log_scale) given masked x.
        n_features: Number of input/output features.
    """

    conditioner: _FCNN
    n_features: int = eqx.field(static=True)

    def forward(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Transform x -> z and compute log |det J|.

        Args:
            x: Input array of shape (n_features,).

        Returns:
            Tuple of (z, log_det_jac) each of shape (n_features,).
        """
        params = self.conditioner(x)  # (2 * n_features,)
        mu, log_scale = jnp.split(params, 2, axis=-1)
        log_scale = jnp.tanh(log_scale)  # stabilise
        z = (x - mu) * jnp.exp(-log_scale)
        return z, -jnp.sum(log_scale)

    def inverse(self, z: jnp.ndarray) -> jnp.ndarray:
        """Invert the transform: z -> x (sequential, autoregressive).

        Args:
            z: Latent array of shape (n_features,).

        Returns:
            Reconstructed x of shape (n_features,).
        """

        def step(x_partial: jnp.ndarray, i: int) -> tuple[jnp.ndarray, None]:
            params = self.conditioner(x_partial)
            mu, log_scale = jnp.split(params, 2, axis=-1)
            log_scale = jnp.tanh(log_scale)
            xi = z[i] * jnp.exp(log_scale[i]) + mu[i]
            x_partial = x_partial.at[i].set(xi)
            return x_partial, None

        x = jnp.zeros_like(z)
        for i in range(self.n_features):
            x, _ = step(x, i)
        return x


class NSFTransform(eqx.Module):
    """Single neural spline flow (rational-quadratic) transform.

    Uses a rational-quadratic spline conditioned on an autoregressive network.

    Attributes:
        conditioner: _FCNN that outputs spline parameters (W, H, D) per feature.
        n_features: Number of input/output features.
        n_bins: Number of spline bins.
        bound: Spline input/output domain [-bound, bound].
    """

    conditioner: _FCNN
    n_features: int = eqx.field(static=True)
    n_bins: int = eqx.field(static=True)
    bound: float = eqx.field(static=True)

    def _spline_params(self, raw: jnp.ndarray, i: int) -> tuple:
        """Extract and normalise spline parameters for feature i.

        Args:
            raw: Raw conditioner output (n_features * (3*n_bins - 1),).
            i: Feature index.

        Returns:
            Tuple of (widths, heights, derivatives) arrays.
        """
        n = 3 * self.n_bins - 1
        raw_i = raw[i * n : (i + 1) * n]
        widths = jax.nn.softmax(raw_i[: self.n_bins]) * 2 * self.bound
        heights = jax.nn.softmax(raw_i[self.n_bins : 2 * self.n_bins]) * 2 * self.bound
        derivatives = jax.nn.softplus(raw_i[2 * self.n_bins :]) + 1e-5
        # Append boundary derivatives (= 1 at each end)
        derivatives = jnp.concatenate([jnp.array([1.0]), derivatives, jnp.array([1.0])])
        return widths, heights, derivatives

    @staticmethod
    def _rqs(
        x: jnp.ndarray,
        widths: jnp.ndarray,
        heights: jnp.ndarray,
        derivatives: jnp.ndarray,
        bound: float,
        inverse: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply rational-quadratic spline to a scalar.

        Args:
            x: Input scalar.
            widths: Bin widths array of shape (n_bins,).
            heights: Bin heights array of shape (n_bins,).
            derivatives: Knot derivative array of shape (n_bins+1,).
            bound: Spline domain bound.
            inverse: If True, compute inverse transform.

        Returns:
            Tuple of (transformed scalar, log |derivative|).
        """
        # Cumulative bin edges
        cum_widths = jnp.concatenate([jnp.array([-bound]), jnp.cumsum(widths) - bound])
        cum_heights = jnp.concatenate([jnp.array([-bound]), jnp.cumsum(heights) - bound])

        # Identity outside bounds
        below = x < -bound
        above = x > bound

        # Find bin
        bin_idx = jnp.sum(cum_widths <= x) - 1
        bin_idx = jnp.clip(bin_idx, 0, len(widths) - 1)

        x0 = cum_widths[bin_idx]
        x1 = cum_widths[bin_idx + 1]
        y0 = cum_heights[bin_idx]
        y1 = cum_heights[bin_idx + 1]
        s = (y1 - y0) / (x1 - x0)
        d0 = derivatives[bin_idx]
        d1 = derivatives[bin_idx + 1]
        w = (x1 - x0)
        h = (y1 - y0)

        if not inverse:
            xi = (x - x0) / w
            numerator = h * (s * xi**2 + d0 * xi * (1 - xi))
            denominator = s + ((d1 + d0 - 2 * s) * xi * (1 - xi))
            y = y0 + numerator / denominator
            dy_dx = (
                s**2
                * (d1 * xi**2 + 2 * s * xi * (1 - xi) + d0 * (1 - xi) ** 2)
                / denominator**2
            )
            log_det = jnp.log(jnp.abs(dy_dx))
        else:
            # Inverse: quadratic formula
            a = h * (s - d0) + (x - y0) * (d1 + d0 - 2 * s)
            b = h * d0 - (x - y0) * (d1 + d0 - 2 * s)
            c = -s * (x - y0)
            discriminant = b**2 - 4 * a * c
            xi = (2 * c) / (-b - jnp.sqrt(jnp.maximum(discriminant, 0.0)))
            y = xi * w + x0
            dy_dx = (
                s**2
                * (d1 * xi**2 + 2 * s * xi * (1 - xi) + d0 * (1 - xi) ** 2)
                / (s + (d1 + d0 - 2 * s) * xi * (1 - xi)) ** 2
            )
            log_det = -jnp.log(jnp.abs(dy_dx))

        # Identity map outside bounds
        y = jnp.where(below | above, x, y)
        log_det = jnp.where(below | above, 0.0, log_det)
        return y, log_det

    def forward(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Transform x -> z and compute log |det J|.

        Args:
            x: Input array of shape (n_features,).

        Returns:
            Tuple of (z, log_det_jac) where log_det_jac is scalar.
        """
        raw = self.conditioner(x)
        z = jnp.zeros_like(x)
        log_det_total = 0.0
        for i in range(self.n_features):
            widths, heights, derivatives = self._spline_params(raw, i)
            zi, ldi = self._rqs(x[i], widths, heights, derivatives, self.bound)
            z = z.at[i].set(zi)
            log_det_total = log_det_total + ldi
        return z, log_det_total

    def inverse(self, z: jnp.ndarray) -> jnp.ndarray:
        """Invert: z -> x (sequential).

        Args:
            z: Latent array of shape (n_features,).

        Returns:
            Reconstructed x of shape (n_features,).
        """
        raw = self.conditioner(z)
        x = jnp.zeros_like(z)
        for i in range(self.n_features):
            widths, heights, derivatives = self._spline_params(raw, i)
            xi, _ = self._rqs(z[i], widths, heights, derivatives, self.bound, inverse=True)
            x = x.at[i].set(xi)
        return x


# ---------------------------------------------------------------------------
# JAXFlowEmulator
# ---------------------------------------------------------------------------


class JAXFlowEmulator(SPSEmulator, eqx.Module):
    """SPS emulator backed by a normalising flow exported from synference.

    The emulator wraps a normalising flow (MAF or NSF) whose weights have been
    exported from a synference checkpoint and reconstructed as a frozen
    Equinox pytree.  No PyTorch code runs at inference time.

    **Direction**: this class expects a *forward-direction* model, i.e. a flow
    trained with ``params`` as the conditioning context and ``photometry`` as
    the target (the inverse of the default synference NPE direction).  If you
    load a checkpoint trained in the NPE direction, predictions will be wrong.
    Use the ``direction`` attribute to record which was loaded.

    Attributes:
        transforms: List of MAFTransform or NSFTransform layers (frozen).
        param_scaler_mean: StandardScaler mean for SPS params, shape (N_params,).
        param_scaler_std: StandardScaler std for SPS params, shape (N_params,).
        phot_scaler_mean: StandardScaler mean for photometry, shape (N_bands,).
        phot_scaler_std: StandardScaler std for photometry, shape (N_bands,).
        _param_names: SPS parameter name list.
        _band_names: Photometric band name list.
        direction: String recording the emulator direction ('forward' or 'npe').
    """

    transforms: list
    param_scaler_mean: jnp.ndarray
    param_scaler_std: jnp.ndarray
    phot_scaler_mean: jnp.ndarray
    phot_scaler_std: jnp.ndarray
    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)
    direction: str = eqx.field(static=True)

    @property
    def param_names(self) -> list[str]:
        """Names of SPS input parameters."""
        return self._param_names

    @property
    def band_names(self) -> list[str]:
        """Names of photometric output bands."""
        return self._band_names

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Predict photometry from SPS parameters.

        Scales the input parameters with the stored StandardScaler, passes them
        through the normalising flow in the inverse direction (latent → data),
        and unscales the output to physical flux units (nJy).

        Args:
            params: SPS parameter array of shape (N_pixels, N_params).
                Units: log10 stellar mass (Msun), log10 age (yr),
                log10 metallicity (Z/Zsun), tau_v (mag).

        Returns:
            Predicted photometry of shape (N_pixels, N_bands) in nJy.
        """
        # Normalise inputs
        scaled_params = (params - self.param_scaler_mean) / self.param_scaler_std

        def _predict_single(p: jnp.ndarray) -> jnp.ndarray:
            z = p
            for transform in self.transforms:
                z = transform.inverse(z)
            return z

        scaled_phot = jax.vmap(_predict_single)(scaled_params)
        # Unscale outputs
        return scaled_phot * self.phot_scaler_std + self.phot_scaler_mean

    @classmethod
    def from_synference_checkpoint(
        cls,
        checkpoint_path: str | Path,
        param_names: list[str],
        band_names: list[str],
        direction: str = "forward",
        n_bins: int = 8,
        spline_bound: float = 5.0,
    ) -> "JAXFlowEmulator":
        """Load a synference flow checkpoint and export weights to JAX.

        This method loads the PyTorch checkpoint, walks the state_dict, converts
        every tensor to numpy, and reconstructs an equivalent Equinox pytree.
        After construction, no PyTorch code runs during inference.

        Args:
            checkpoint_path: Path to the synference checkpoint file (joblib/pickle).
            param_names: Ordered list of SPS parameter names in the checkpoint.
            band_names: Ordered list of photometric band names in the checkpoint.
            direction: Direction the flow was trained. Use ``'forward'`` for a
                model trained to predict photometry from params, or ``'npe'``
                for the standard synference NPE posterior direction.
            n_bins: Number of rational-quadratic spline bins for NSF flows.
            spline_bound: Domain bound for rational-quadratic spline.

        Returns:
            JAXFlowEmulator with frozen Equinox weights.

        Raises:
            ImportError: If PyTorch or joblib is not installed.
            ValueError: If the checkpoint format is not recognised.
        """
        try:
            import joblib
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch and joblib are required to load synference checkpoints. "
                "Install them with: pip install torch joblib"
            ) from e

        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading synference checkpoint: {checkpoint_path}")

        param_dict = joblib.load(checkpoint_path)

        # Extract the posterior object — synference saves it under 'posteriors'
        posterior = param_dict.get("posteriors")
        if posterior is None:
            raise ValueError(
                "Checkpoint does not contain 'posteriors' key. "
                f"Available keys: {list(param_dict.keys())}"
            )

        # Handle ensemble posteriors (list) vs single posterior
        if isinstance(posterior, list):
            posterior = posterior[0]
            logger.info("Ensemble posterior detected; using first member.")

        # Extract scalers
        param_scaler = param_dict.get("param_scaler") or param_dict.get("x_scaler")
        phot_scaler = param_dict.get("phot_scaler") or param_dict.get("x_scaler")

        if param_scaler is not None and hasattr(param_scaler, "mean_"):
            param_mean = jnp.array(param_scaler.mean_, dtype=jnp.float32)
            param_std = jnp.array(param_scaler.scale_, dtype=jnp.float32)
        else:
            n_p = len(param_names)
            param_mean = jnp.zeros(n_p, dtype=jnp.float32)
            param_std = jnp.ones(n_p, dtype=jnp.float32)
            logger.warning("No param scaler found in checkpoint; using identity scaling.")

        if phot_scaler is not None and hasattr(phot_scaler, "mean_"):
            phot_mean = jnp.array(phot_scaler.mean_, dtype=jnp.float32)
            phot_std = jnp.array(phot_scaler.scale_, dtype=jnp.float32)
        else:
            n_b = len(band_names)
            phot_mean = jnp.zeros(n_b, dtype=jnp.float32)
            phot_std = jnp.ones(n_b, dtype=jnp.float32)
            logger.warning("No phot scaler found in checkpoint; using identity scaling.")

        # Extract the lampe flow module from the posterior
        flow = cls._extract_flow(posterior)
        transforms = cls._build_transforms(flow, len(param_names), n_bins, spline_bound)

        logger.info(f"Exported {len(transforms)} flow transform(s) to JAX/Equinox.")
        return cls(
            transforms=transforms,
            param_scaler_mean=param_mean,
            param_scaler_std=param_std,
            phot_scaler_mean=phot_mean,
            phot_scaler_std=phot_std,
            _param_names=param_names,
            _band_names=band_names,
            direction=direction,
        )

    @staticmethod
    def _extract_flow(posterior: Any) -> Any:
        """Extract the underlying lampe flow from a synference posterior.

        Args:
            posterior: A synference/ltu-ili posterior object.

        Returns:
            The lampe flow module.

        Raises:
            ValueError: If the flow cannot be found.
        """
        # ltu-ili wraps lampe flows — try common attribute paths
        for attr in ("flow", "_flow", "net", "_net", "estimator", "_estimator"):
            flow = getattr(posterior, attr, None)
            if flow is not None:
                return flow
        # Some posteriors are themselves flows
        if hasattr(posterior, "state_dict"):
            return posterior
        raise ValueError(
            "Cannot extract flow from posterior object. "
            f"Type: {type(posterior)}. Attributes: {dir(posterior)}"
        )

    @staticmethod
    def _tensor_to_jax(tensor: Any) -> jnp.ndarray:
        """Convert a PyTorch tensor to a JAX float32 array.

        Args:
            tensor: PyTorch tensor.

        Returns:
            JAX float32 array.
        """
        return jnp.array(tensor.detach().cpu().numpy(), dtype=jnp.float32)

    @classmethod
    def _build_transforms(
        cls,
        flow: Any,
        n_features: int,
        n_bins: int,
        spline_bound: float,
    ) -> list:
        """Build a list of Equinox transform modules from a lampe flow state_dict.

        Args:
            flow: Lampe flow PyTorch module with a state_dict.
            n_features: Number of features (dimensionality).
            n_bins: Number of spline bins (NSF only).
            spline_bound: Spline domain bound (NSF only).

        Returns:
            List of MAFTransform or NSFTransform Equinox modules.
        """
        state_dict = flow.state_dict()
        transform_keys = sorted(
            {k.split(".")[0] for k in state_dict if k.startswith("transform")}
        )

        if not transform_keys:
            # Flat structure: try to detect from key names
            logger.warning(
                "No 'transform.*' keys in state_dict. "
                "Attempting to build a single-transform flow from raw weights."
            )
            transforms = [cls._build_single_transform(state_dict, "", n_features, n_bins, spline_bound)]
            return transforms

        transforms = []
        for tkey in transform_keys:
            sub = {
                k[len(tkey) + 1 :]: v
                for k, v in state_dict.items()
                if k.startswith(tkey + ".")
            }
            t = cls._build_single_transform(sub, tkey, n_features, n_bins, spline_bound)
            transforms.append(t)
        return transforms

    @classmethod
    def _build_single_transform(
        cls,
        sub_dict: dict,
        prefix: str,
        n_features: int,
        n_bins: int,
        spline_bound: float,
    ) -> MAFTransform | NSFTransform:
        """Build a single MAFTransform or NSFTransform from a state_dict sub-dict.

        Args:
            sub_dict: Subset of the flow's state_dict for one transform.
            prefix: Key prefix used for logging only.
            n_features: Feature dimensionality.
            n_bins: Number of spline bins.
            spline_bound: Spline domain bound.

        Returns:
            Equinox MAFTransform or NSFTransform.
        """
        # Detect transform type from key names
        is_nsf = any("spline" in k or "bin" in k or "derivative" in k for k in sub_dict)

        # Build conditioner layers from 'conditioner.layers.*' or 'layers.*'
        conditioner_layers = cls._extract_linear_layers(sub_dict)

        conditioner = _FCNN(layers=conditioner_layers)

        if is_nsf:
            return NSFTransform(
                conditioner=conditioner,
                n_features=n_features,
                n_bins=n_bins,
                bound=spline_bound,
            )
        else:
            return MAFTransform(
                conditioner=conditioner,
                n_features=n_features,
            )

    @classmethod
    def _extract_linear_layers(cls, sub_dict: dict) -> list:
        """Extract linear layers from a state_dict sub-dictionary.

        Supports weight/bias keys following the pattern:
        ``layers.0.weight``, ``conditioner.layers.0.weight``, etc.

        Args:
            sub_dict: State dict sub-dictionary.

        Returns:
            List of Equinox linear layer modules (simple weight + bias).
        """
        import re

        # Find all layer indices
        pattern = re.compile(r"(?:conditioner\.)?(?:layers|net)\.(\d+)\.weight")
        indices = sorted({int(m.group(1)) for k in sub_dict for m in [pattern.match(k)] if m})

        if not indices:
            raise ValueError(
                f"Cannot find any linear layer weights in sub_dict. Keys: {list(sub_dict.keys())}"
            )

        layers = []
        for idx in indices:
            for prefix in ("conditioner.layers", "layers", "conditioner.net", "net"):
                wkey = f"{prefix}.{idx}.weight"
                bkey = f"{prefix}.{idx}.bias"
                if wkey in sub_dict:
                    weight = cls._tensor_to_jax(sub_dict[wkey])
                    bias = cls._tensor_to_jax(sub_dict[bkey])
                    # Build a simple linear layer (eqx.nn.Linear-compatible)
                    layer = _SimpleLinear(weight=weight, bias=bias)
                    layers.append(layer)
                    break

        return layers

    @classmethod
    def from_weights(
        cls,
        transforms: list,
        param_scaler_mean: np.ndarray,
        param_scaler_std: np.ndarray,
        phot_scaler_mean: np.ndarray,
        phot_scaler_std: np.ndarray,
        param_names: list[str],
        band_names: list[str],
        direction: str = "forward",
    ) -> "JAXFlowEmulator":
        """Construct a JAXFlowEmulator directly from pre-built Equinox transforms.

        Useful for testing: pass in ``MockTransform`` objects without needing
        a real synference checkpoint.

        Args:
            transforms: List of Equinox transform objects (MAFTransform, NSFTransform,
                or any object with a callable ``inverse`` method).
            param_scaler_mean: Mean array for input parameter scaling, shape (N_params,).
            param_scaler_std: Std array for input parameter scaling, shape (N_params,).
            phot_scaler_mean: Mean array for output photometry unscaling, shape (N_bands,).
            phot_scaler_std: Std array for output photometry unscaling, shape (N_bands,).
            param_names: SPS parameter name list.
            band_names: Photometric band name list.
            direction: Emulator direction string.

        Returns:
            JAXFlowEmulator instance.
        """
        return cls(
            transforms=transforms,
            param_scaler_mean=jnp.array(param_scaler_mean, dtype=jnp.float32),
            param_scaler_std=jnp.array(param_scaler_std, dtype=jnp.float32),
            phot_scaler_mean=jnp.array(phot_scaler_mean, dtype=jnp.float32),
            phot_scaler_std=jnp.array(phot_scaler_std, dtype=jnp.float32),
            _param_names=param_names,
            _band_names=band_names,
            direction=direction,
        )


class _SimpleLinear(eqx.Module):
    """Simple linear layer without autoregressive masking.

    Used for conditioner networks inside MAF/NSF transforms.

    Attributes:
        weight: Weight matrix of shape (out_features, in_features).
        bias: Bias vector of shape (out_features,).
    """

    weight: jnp.ndarray
    bias: jnp.ndarray

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Linear forward pass.

        Args:
            x: Input array of shape (..., in_features).

        Returns:
            Output array of shape (..., out_features).
        """
        return x @ self.weight.T + self.bias
