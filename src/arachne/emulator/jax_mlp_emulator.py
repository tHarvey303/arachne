"""JAX-native MLP emulator for stellar population synthesis (SPS) forward modelling.

This module implements a fast, fully differentiable emulator that maps SPS
parameters directly to photometry, trained from a synference model library.

Architecture
------------
Based on the *Speculator* architecture (Alsing et al. 2020,
`arXiv:1911.11778 <https://arxiv.org/abs/1911.11778>`_).  Each hidden layer
uses the Alsing activation — a learnable self-gating transform that is
empirically well-suited to the smooth, monotonic mappings produced by SPS
codes:

    z = W x + b
    y = (β + σ(α ⊙ z) ⊙ (1 − β)) ⊙ z

where W, b, α, β are all learnable per-unit parameters.

Training
--------
The emulator is trained directly from a synference HDF5 model library
(``Grid/Parameters``, ``Grid/Photometry``).  Training is done in **log₁₀
flux space** (normalised to zero mean / unit variance) with MSE loss, using
Optax Adam with cosine LR decay.

This approach replaces the fragile ``JAXFlowEmulator.from_synference_checkpoint``
weight-export path and gives a fully native JAX pipeline with no PyTorch
dependency at inference time.

Example:
-------
.. code-block:: python

    emulator = SPSMLPEmulator.from_synference_library(
        library_path="galaxy_library.h5",
        param_names=["log_stellar_mass", "log_age", "log_metallicity", "tau_v"],
        band_names=["JWST/NIRCam.F115W", "JWST/NIRCam.F200W"],
    )
    emulator.save("emulator.eqx")

    # Later:
    emulator = SPSMLPEmulator.load("emulator.eqx", param_names=..., band_names=...)
    fluxes = emulator.predict(params)  # (N_pixels, N_bands) in nJy
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from arachne.emulator.base import SPSEmulator
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


# ---------------------------------------------------------------------------
# Alsing layer
# ---------------------------------------------------------------------------


class AlsingLayer(eqx.Module):
    """Alsing et al. (2020) Speculator activation layer.

    Implements a learnable self-gating transform:

        z = W x + b
        y = (β + σ(α ⊙ z) ⊙ (1 − β)) ⊙ z

    This activation has been shown to work particularly well for emulating
    stellar population synthesis codes because it can represent the smooth,
    quasi-monotonic mappings produced by SPS models while maintaining
    differentiability everywhere.

    Attributes:
        weight: Linear weight matrix of shape (out_features, in_features).
        bias: Bias vector of shape (out_features,).
        alpha: Gate width parameter of shape (out_features,) — controls the
            width of the sigmoid transition.
        beta: Gate offset parameter of shape (out_features,) — controls the
            minimum activation pass-through.
    """

    weight: jnp.ndarray
    bias: jnp.ndarray
    alpha: jnp.ndarray
    beta: jnp.ndarray

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.random.KeyArray,
    ) -> None:
        """Initialise the Alsing layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            key: JAX random key for weight initialisation.
        """
        k1, k2, k3, k4 = jax.random.split(key, 4)
        # He initialisation for weight (good default for deep networks)
        scale = jnp.sqrt(2.0 / in_features)
        self.weight = jax.random.normal(k1, (out_features, in_features)) * scale
        self.bias = jnp.zeros(out_features)
        # Initialise alpha near 1 and beta near 0 — close to ReLU at start
        self.alpha = jax.random.normal(k3, (out_features,)) * 0.1 + 1.0
        self.beta = jax.random.normal(k4, (out_features,)) * 0.1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the Alsing activation.

        Args:
            x: Input array of shape (..., in_features).

        Returns:
            Output array of shape (..., out_features).
        """
        z = x @ self.weight.T + self.bias
        gate = self.beta + jax.nn.sigmoid(self.alpha * z) * (1.0 - self.beta)
        return gate * z


# ---------------------------------------------------------------------------
# MLP emulator
# ---------------------------------------------------------------------------


class SPSMLPEmulator(SPSEmulator, eqx.Module):
    """Fast JAX-native MLP emulator mapping SPS parameters to photometry.

    All parameters (weights, normalisations) are stored as an Equinox pytree,
    making the emulator fully compatible with ``jax.jit``, ``jax.grad``, and
    ``jax.vmap``.

    The emulator operates internally in normalised log₁₀ flux space:

    1. Normalise input params: ``p_norm = (p − μ_p) / σ_p``
    2. Forward pass through Alsing MLP
    3. Unnormalise output: ``log10_flux = out * σ_phot + μ_phot``
    4. Return flux in nJy: ``flux = 10 ** log10_flux``

    Attributes:
        hidden_layers: List of AlsingLayer hidden layers.
        output_layer: Final linear layer (no activation).
        in_mean: Input normalisation mean, shape (N_params,). Frozen.
        in_std: Input normalisation std, shape (N_params,). Frozen.
        out_mean: Output (log10 flux) normalisation mean, shape (N_bands,). Frozen.
        out_std: Output (log10 flux) normalisation std, shape (N_bands,). Frozen.
        _param_names: SPS parameter name list. Static.
        _band_names: Photometric band name list. Static.
    """

    hidden_layers: tuple
    output_layer: eqx.nn.Linear
    in_mean: jnp.ndarray
    in_std: jnp.ndarray
    out_mean: jnp.ndarray
    out_std: jnp.ndarray
    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)

    def __init__(
        self,
        param_names: list[str],
        band_names: list[str],
        hidden_sizes: list[int],
        in_mean: np.ndarray,
        in_std: np.ndarray,
        out_mean: np.ndarray,
        out_std: np.ndarray,
        key: jax.random.KeyArray,
    ) -> None:
        """Initialise the SPSMLPEmulator with random weights.

        Args:
            param_names: Ordered list of SPS parameter names.
            band_names: Ordered list of photometric band names.
            hidden_sizes: List of hidden layer widths, e.g. [256, 256, 256].
            in_mean: Input normalisation mean, shape (N_params,).
            in_std: Input normalisation std, shape (N_params,).
            out_mean: Output (log10 flux) normalisation mean, shape (N_bands,).
            out_std: Output (log10 flux) normalisation std, shape (N_bands,).
            key: JAX random key for weight initialisation.
        """
        self._param_names = param_names
        self._band_names = band_names

        n_in = len(param_names)
        n_out = len(band_names)

        self.in_mean = jnp.array(in_mean, dtype=jnp.float32)
        self.in_std = jnp.array(in_std, dtype=jnp.float32)
        self.out_mean = jnp.array(out_mean, dtype=jnp.float32)
        self.out_std = jnp.array(out_std, dtype=jnp.float32)

        sizes = [n_in] + hidden_sizes
        keys = jax.random.split(key, len(sizes))
        self.hidden_layers = tuple(
            AlsingLayer(in_sz, out_sz, keys[i])
            for i, (in_sz, out_sz) in enumerate(zip(sizes[:-1], sizes[1:]))
        )

        out_key = jax.random.split(key, 1)[0]
        self.output_layer = eqx.nn.Linear(sizes[-1], n_out, key=out_key)

    @property
    def param_names(self) -> list[str]:
        """Names of SPS input parameters."""
        return self._param_names

    @property
    def band_names(self) -> list[str]:
        """Names of photometric output bands."""
        return self._band_names

    def _forward_normalised(self, p_norm: jnp.ndarray) -> jnp.ndarray:
        """Forward pass on a single normalised input vector.

        Args:
            p_norm: Normalised SPS parameter vector of shape (N_params,).

        Returns:
            Normalised log10 flux of shape (N_bands,).
        """
        x = p_norm
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Predict photometry from SPS parameters.

        Args:
            params: SPS parameter array of shape (N_pixels, N_params).
                Units: log10 stellar mass (Msun), log10 age (yr),
                log10 metallicity (Z/Zsun), tau_v (mag).

        Returns:
            Predicted photometry of shape (N_pixels, N_bands) in nJy.
        """
        p_norm = (params - self.in_mean) / (self.in_std + 1e-8)
        out_norm = jax.vmap(self._forward_normalised)(p_norm)  # (N_pixels, N_bands)
        log10_flux = out_norm * self.out_std + self.out_mean
        return 10.0**log10_flux  # nJy

    def save(self, path: str | Path) -> None:
        """Save the emulator weights to an Equinox checkpoint file.

        Args:
            path: Output file path (conventionally ``*.eqx``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(str(path), self)
        logger.info(f"Emulator saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        param_names: list[str],
        band_names: list[str],
        hidden_sizes: list[int] | None = None,
    ) -> "SPSMLPEmulator":
        """Load a saved emulator from an Equinox checkpoint file.

        The architecture (hidden_sizes) must match what was used during training.
        A dummy model with the same architecture is constructed first, then its
        leaves are replaced by the saved values.

        Args:
            path: Path to the saved ``.eqx`` checkpoint file.
            param_names: SPS parameter names (must match training order).
            band_names: Band names (must match training order).
            hidden_sizes: Hidden layer widths used during training.
                Defaults to [256, 256, 256].

        Returns:
            SPSMLPEmulator loaded from disk.
        """
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 256]
        n_p = len(param_names)
        n_b = len(band_names)
        dummy = cls(
            param_names=param_names,
            band_names=band_names,
            hidden_sizes=hidden_sizes,
            in_mean=np.zeros(n_p),
            in_std=np.ones(n_p),
            out_mean=np.zeros(n_b),
            out_std=np.ones(n_b),
            key=jax.random.PRNGKey(0),
        )
        loaded = eqx.tree_deserialise_leaves(str(path), dummy)
        logger.info(f"Emulator loaded from {path}")
        return loaded

    @classmethod
    def from_synference_library(
        cls,
        library_path: str | Path,
        param_names: list[str],
        band_names: list[str],
        hidden_sizes: list[int] | None = None,
        n_epochs: int = 300,
        batch_size: int = 512,
        learning_rate: float = 3e-4,
        val_fraction: float = 0.1,
        seed: int = 0,
        log_interval: int = 10,
    ) -> "SPSMLPEmulator":
        """Train an emulator from a synference HDF5 model library.

        Loads ``Grid/Parameters`` (SPS params) and ``Grid/Photometry`` (fluxes
        in nJy) from the HDF5 file, selects the requested param/band columns,
        and trains the MLP with MSE loss in normalised log₁₀ flux space.

        Args:
            library_path: Path to the synference HDF5 library file.
            param_names: Ordered list of SPS parameter names to use as inputs.
                Must be a subset of the library's ``ParameterNames`` attribute.
            band_names: Ordered list of photometric band names to predict.
                Must be a subset of the library's ``FilterCodes`` attribute.
            hidden_sizes: Hidden layer widths. Defaults to [256, 256, 256].
            n_epochs: Number of training epochs. Defaults to 300.
            batch_size: Minibatch size. Defaults to 512.
            learning_rate: Peak Adam learning rate. Defaults to 3e-4.
            val_fraction: Fraction of library held out for validation. Defaults
                to 0.1.
            seed: Random seed for reproducibility.
            log_interval: Log training loss every this many epochs.

        Returns:
            Trained SPSMLPEmulator.

        Raises:
            ImportError: If h5py or optax is not installed.
            KeyError: If a requested param_name or band_name is not in the library.
        """
        try:
            import h5py
            import optax
        except ImportError as e:
            raise ImportError(
                "h5py and optax are required for training. "
                "Install with: pip install h5py optax"
            ) from e

        if hidden_sizes is None:
            hidden_sizes = [256, 256, 256]

        library_path = Path(library_path)
        logger.info(f"Loading synference library: {library_path}")

        with h5py.File(library_path, "r") as f:
            # Load raw arrays — shape (n_params, n_models) and (n_filters, n_models)
            raw_params = f["Grid/Parameters"][()]  # (N_p_all, N_models)
            raw_phot = f["Grid/Photometry"][()]    # (N_b_all, N_models)

            # Read metadata attributes
            lib_param_names = _decode_str_attr(f.attrs.get("ParameterNames", []))
            lib_band_names = _decode_str_attr(f.attrs.get("FilterCodes", []))

            # Fallback: FilterCodes may be a dataset
            if not lib_band_names and "Grid/FilterCodes" in f:
                lib_band_names = _decode_str_attr(f["Grid/FilterCodes"][()])

        logger.info(
            f"Library: {raw_params.shape[1]} models, "
            f"{len(lib_param_names)} params, {len(lib_band_names)} bands"
        )

        # Select requested parameter columns
        param_indices = _select_indices(param_names, lib_param_names, "parameter")
        band_indices = _select_indices(band_names, lib_band_names, "band")

        params = raw_params[param_indices, :].T.astype(np.float32)  # (N_models, N_p)
        phot = raw_phot[band_indices, :].T.astype(np.float32)        # (N_models, N_b)

        # Filter out non-positive fluxes (unphysical; can't take log)
        valid = np.all(phot > 0, axis=1) & np.all(np.isfinite(params), axis=1)
        params = params[valid]
        phot = phot[valid]
        logger.info(f"After filtering: {params.shape[0]} valid models")

        # Work in log10 flux space
        log_phot = np.log10(phot)

        # Train / validation split
        rng = np.random.default_rng(seed)
        n_total = params.shape[0]
        n_val = max(1, int(n_total * val_fraction))
        idx = rng.permutation(n_total)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        p_train, lp_train = params[train_idx], log_phot[train_idx]
        p_val, lp_val = params[val_idx], log_phot[val_idx]

        # Compute normalisation statistics from training set
        in_mean = p_train.mean(axis=0)
        in_std = p_train.std(axis=0) + 1e-8
        out_mean = lp_train.mean(axis=0)
        out_std = lp_train.std(axis=0) + 1e-8

        # Normalise
        p_train_n = (p_train - in_mean) / in_std
        lp_train_n = (lp_train - out_mean) / out_std
        p_val_n = (p_val - in_mean) / in_std
        lp_val_n = (lp_val - out_mean) / out_std

        # Build model
        key = jax.random.PRNGKey(seed)
        model = cls(
            param_names=param_names,
            band_names=band_names,
            hidden_sizes=hidden_sizes,
            in_mean=in_mean,
            in_std=in_std,
            out_mean=out_mean,
            out_std=out_std,
            key=key,
        )

        # Training with Optax Adam + cosine LR decay
        n_steps_per_epoch = max(1, len(p_train_n) // batch_size)
        total_steps = n_epochs * n_steps_per_epoch
        lr_schedule = optax.cosine_decay_schedule(learning_rate, total_steps)
        optimiser = optax.adam(lr_schedule)
        opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def step(
            model: "SPSMLPEmulator",
            opt_state: optax.OptState,
            p_batch: jnp.ndarray,
            lp_batch: jnp.ndarray,
        ) -> tuple["SPSMLPEmulator", optax.OptState, jnp.ndarray]:
            def loss_fn(m: "SPSMLPEmulator") -> jnp.ndarray:
                pred = jax.vmap(m._forward_normalised)(p_batch)
                return jnp.mean((pred - lp_batch) ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, opt_state_new = optimiser.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model_new = eqx.apply_updates(model, updates)
            return model_new, opt_state_new, loss

        @eqx.filter_jit
        def val_loss(model: "SPSMLPEmulator", p_v: jnp.ndarray, lp_v: jnp.ndarray) -> jnp.ndarray:
            pred = jax.vmap(model._forward_normalised)(p_v)
            return jnp.mean((pred - lp_v) ** 2)

        p_train_jax = jnp.array(p_train_n)
        lp_train_jax = jnp.array(lp_train_n)
        p_val_jax = jnp.array(p_val_n)
        lp_val_jax = jnp.array(lp_val_n)

        n_train = len(p_train_jax)
        logger.info(
            f"Training: {n_train} train / {len(p_val_jax)} val, "
            f"{n_epochs} epochs, batch {batch_size}, lr {learning_rate}"
        )

        key_epoch = jax.random.PRNGKey(seed + 1)
        for epoch in range(1, n_epochs + 1):
            # Shuffle training data
            key_epoch, subkey = jax.random.split(key_epoch)
            perm = jax.random.permutation(subkey, n_train)
            p_shuf = p_train_jax[perm]
            lp_shuf = lp_train_jax[perm]

            train_loss_acc = 0.0
            for i in range(0, n_train, batch_size):
                pb = p_shuf[i : i + batch_size]
                lpb = lp_shuf[i : i + batch_size]
                model, opt_state, loss = step(model, opt_state, pb, lpb)
                train_loss_acc += float(loss)

            if epoch % log_interval == 0 or epoch == 1:
                v_loss = float(val_loss(model, p_val_jax, lp_val_jax))
                t_loss = train_loss_acc / max(1, n_train // batch_size)
                logger.info(f"Epoch {epoch:4d}/{n_epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        final_val = float(val_loss(model, p_val_jax, lp_val_jax))
        logger.info(f"Training complete. Final val MSE (normalised log space): {final_val:.4f}")
        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_str_attr(attr) -> list[str]:
    """Decode an HDF5 attribute that may be bytes or strings.

    Args:
        attr: HDF5 attribute value (bytes array, string array, or similar).

    Returns:
        List of Python strings.
    """
    if attr is None:
        return []
    result = []
    for item in attr:
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        elif isinstance(item, np.bytes_):
            result.append(item.tobytes().decode("utf-8"))
        else:
            result.append(str(item))
    return result


def _select_indices(requested: list[str], available: list[str], kind: str) -> list[int]:
    """Return indices of requested names in the available list.

    Args:
        requested: Names the caller wants.
        available: Names present in the library.
        kind: Human-readable label for error messages (e.g. 'parameter').

    Returns:
        List of integer indices into ``available``.

    Raises:
        KeyError: If any requested name is not in available.
    """
    indices = []
    for name in requested:
        if name not in available:
            raise KeyError(
                f"Requested {kind} '{name}' not found in library. "
                f"Available: {available}"
            )
        indices.append(available.index(name))
    return indices
