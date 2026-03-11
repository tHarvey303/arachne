"""Parrot-style JAX emulator for SPS forward modelling.

This module implements the *Parrot* architecture (Mathews et al. 2023,
`ApJ 954 <https://doi.org/10.3847/1538-4357/ace720>`_): a fully-connected
MLP that maps SPS parameters directly to multi-band photometry using an
arsinh magnitude output transform.

Architecture
------------
- Fully-connected GELU MLP; single network for all output bands.
- Input: z-scored SPS parameters.
- Output: arsinh magnitudes (z-scored to zero mean / unit variance), then
  converted to flux (nJy) via the inverse arsinh transform.

The arsinh magnitude transform (Lupton et al. 1999; Parrot §3.1) maps the
full dynamic range of galaxy photometry — including undetected sources and
high-redshift Lyman-break dropouts that produce formally zero flux — to a
smooth, bounded space without the divergence of log magnitudes at zero flux::

    a = 2.5 * log10(
        e
    )  # ≈ 1.086
    mu0 = 35  # softening magnitude (reference zeropoint)
    mu = (
        -a
        * arcsinh(
            f
            * exp(
                mu0 / a
            )
            / 2
        )
        + mu0
    )  # flux -> asinh mag
    f = (
        2
        * exp(-mu0 / a)
        * sinh(
            (mu0 - mu)
            / a
        )
    )  # asinh mag -> flux

Key differences from SPSMLPEmulator (Speculator/Alsing)
---------------------------------------------------------
- GELU activations vs Alsing self-gating layers.
- arsinh magnitude output space vs log10 flux.  Handles near-zero fluxes
  (Lyman-break dropouts) without distorting the loss landscape.
- Metadata reader that understands the synference v4 HDF5 format
  (``Model.attrs['varying_param_names']`` + ``Model.attrs['stellar_params']``
  + ``Model.attrs['filters']``), in addition to the older root-level attrs
  used by ``SPSMLPEmulator.from_synference_library``.

Example:
-------
.. code-block:: python

    emulator = ParrotEmulator.from_synference_library(
        library_path="grid_BPASS_...hdf5",
        param_names=[
            "redshift",
            "log10metallicity",
            "Av",
            "log_sfr",
            "sfh_quantile_25",
            "sfh_quantile_50",
            "sfh_quantile_75",
            "tau_v",
        ],
        band_names=[
            "JWST/NIRCam.F200W",
            "JWST/NIRCam.F277W",
            ...,
        ],
    )
    emulator.save(
        "parrot_emulator.eqx"
    )

    # Later:
    emulator = ParrotEmulator.load(
        "parrot_emulator.eqx",
        param_names=[
            ...
        ],
        band_names=[
            ...
        ],
    )
    fluxes = emulator.predict(
        params
    )  # (N_pixels, N_bands) in nJy
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
# arsinh magnitude transform (Parrot / Lupton et al. 1999)
# ---------------------------------------------------------------------------

_ASINH_A: float = 2.5 * np.log10(np.e)  # ≈ 1.0857
_ASINH_MU0: float = 35.0  # softening magnitude (AB-like zeropoint in nJy)


def flux_to_asinh_mag(flux: jnp.ndarray) -> jnp.ndarray:
    """Convert flux (nJy) to arsinh magnitude.

    Smooth over the full dynamic range including zero and negative flux,
    asymptoting to standard AB magnitudes for bright sources.

    Args:
        flux: Flux in nJy, any shape.  Need not be positive.

    Returns:
        arsinh magnitude, same shape as flux.  Bright sources (large flux)
        give small (negative) values; faint/non-detected sources give
        values near ``mu0`` = 35.
    """
    a = _ASINH_A
    mu0 = _ASINH_MU0
    b = jnp.exp(mu0 / a) / 2.0  # softening parameter
    return -a * jnp.arcsinh(flux * b) + mu0


def asinh_mag_to_flux(mag: jnp.ndarray) -> jnp.ndarray:
    """Convert arsinh magnitude back to flux (nJy).

    Inverse of :func:`flux_to_asinh_mag`.

    Args:
        mag: arsinh magnitude, any shape.

    Returns:
        Flux in nJy, same shape as mag.
    """
    a = _ASINH_A
    mu0 = _ASINH_MU0
    return 2.0 * jnp.exp(-mu0 / a) * jnp.sinh((mu0 - mag) / a)


# ---------------------------------------------------------------------------
# ParrotEmulator
# ---------------------------------------------------------------------------


class ParrotEmulator(SPSEmulator):
    """Parrot-style MLP emulator: GELU layers + arsinh magnitude output.

    All parameters are stored as an Equinox pytree, making the model
    fully compatible with ``jax.jit``, ``jax.grad``, and ``jax.vmap``.

    Internal pipeline for a single input vector p (shape N_params)::

        p_norm = (
            (p - in_mean)
            / in_std
        )  # z-score inputs
        mu = MLP(
            p_norm
        )  # GELU MLP → N_bands
        mu_phys = (
            mu * out_std
            + out_mean
        )  # un-normalise asinh mags
        flux = asinh_mag_to_flux(
            mu_phys
        )  # nJy

    Attributes:
        layers: Equinox Linear layers (includes the final output layer).
        in_mean: Input normalisation mean, shape (N_params,). Static (frozen).
        in_std: Input normalisation std, shape (N_params,). Static (frozen).
        out_mean: Output (arsinh mag) normalisation mean, shape (N_bands,). Frozen.
        out_std: Output (arsinh mag) normalisation std, shape (N_bands,). Frozen.
        _param_names: SPS parameter name list. Static (not differentiated).
        _band_names: Photometric band name list. Static (not differentiated).
    """

    layers: tuple
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
        """Initialise ParrotEmulator with random weights.

        Args:
            param_names: Ordered list of SPS parameter names.
            band_names: Ordered list of photometric band names.
            hidden_sizes: Hidden layer widths, e.g. ``[512, 512, 512, 512, 512]``.
            in_mean: Input normalisation mean, shape (N_params,).
            in_std: Input normalisation std, shape (N_params,).
            out_mean: Output (arsinh mag) normalisation mean, shape (N_bands,).
            out_std: Output (arsinh mag) normalisation std, shape (N_bands,).
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

        layer_sizes = [n_in] + hidden_sizes + [n_out]
        keys = jax.random.split(key, len(layer_sizes) - 1)
        self.layers = tuple(
            eqx.nn.Linear(in_sz, out_sz, key=k)
            for (in_sz, out_sz, k) in zip(layer_sizes[:-1], layer_sizes[1:], keys)
        )

    @property
    def param_names(self) -> list[str]:
        """Names of SPS input parameters."""
        return self._param_names

    @property
    def band_names(self) -> list[str]:
        """Names of photometric output bands."""
        return self._band_names

    def _forward_normalised(self, p_norm: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for a single normalised input vector.

        All hidden layers apply GELU; the final output layer is linear.

        Args:
            p_norm: Normalised SPS parameter vector of shape (N_params,).

        Returns:
            Normalised arsinh magnitude vector of shape (N_bands,).
        """
        x = p_norm
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Predict photometry from SPS parameters.

        Args:
            params: SPS parameter array of shape (N_pixels, N_params).
                Parameter order and units must match the library used for training.
                Typical synference v4 parameters: redshift, log10metallicity,
                Av (log10), log_sfr, sfh_quantile_{25,50,75}, tau_v.

        Returns:
            Predicted photometry of shape (N_pixels, N_bands) in nJy.
            Near-zero fluxes (Lyman-break dropouts) are correctly recovered
            as near-zero positive values via the arsinh inverse transform.
        """
        p_norm = (params - self.in_mean) / (self.in_std + 1e-8)
        out_norm = jax.vmap(self._forward_normalised)(p_norm)  # (N_pixels, N_bands)
        asinh_mag = out_norm * self.out_std + self.out_mean
        return asinh_mag_to_flux(asinh_mag)  # nJy

    def save(self, path: str | Path) -> None:
        """Save the emulator weights to an Equinox checkpoint file.

        Args:
            path: Output file path (conventionally ``*.eqx``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(str(path), self)
        logger.info(f"ParrotEmulator saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        param_names: list[str],
        band_names: list[str],
        hidden_sizes: list[int] | None = None,
    ) -> "ParrotEmulator":
        """Load a saved ParrotEmulator from an Equinox checkpoint file.

        The architecture (``hidden_sizes``) must match what was used during
        training.  A dummy model with the same architecture is constructed
        first, then its leaves are replaced by the saved values.

        Args:
            path: Path to the ``.eqx`` checkpoint file.
            param_names: SPS parameter names (must match training order).
            band_names: Band names (must match training order).
            hidden_sizes: Hidden layer widths used during training.
                Defaults to ``[512, 512, 512, 512, 512]``.

        Returns:
            ParrotEmulator loaded from disk.
        """
        if hidden_sizes is None:
            hidden_sizes = [512, 512, 512, 512, 512]
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
        logger.info(f"ParrotEmulator loaded from {path}")
        return loaded

    @classmethod
    def from_synference_library(
        cls,
        library_path: str | Path,
        param_names: list[str],
        band_names: list[str],
        hidden_sizes: list[int] | None = None,
        n_epochs: int = 1000,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        lr_decay_steps: tuple[int, int] = (300, 700),
        lr_decay_factors: tuple[float, float] = (0.1, 0.1),
        val_fraction: float = 0.1,
        early_stopping_patience: int = 20,
        seed: int = 0,
        log_interval: int = 10,
        checkpoint_path: str | Path | None = None,
    ) -> "ParrotEmulator":
        """Train a ParrotEmulator from a synference HDF5 model library.

        Loads ``Grid/Parameters`` and ``Grid/Photometry`` from the HDF5
        library, selects the requested param/band columns, converts photometry
        to arsinh magnitudes (Parrot transform), and trains the MLP with MSE
        loss in normalised arsinh-magnitude space.

        Parameter names and filter codes are read from the library metadata.
        Both the old root-attr format (``ParameterNames``, ``FilterCodes``) and
        the v4 ``Model.attrs`` format (``varying_param_names``,
        ``stellar_params``, ``filters``) are supported.

        Args:
            library_path: Path to the synference HDF5 library file.
            param_names: Ordered list of SPS parameter names to use as inputs.
                E.g. ``["redshift", "log10metallicity", "Av", "log_sfr",
                "sfh_quantile_25", "sfh_quantile_50", "sfh_quantile_75",
                "tau_v"]``.
            band_names: Ordered list of photometric band names to predict.
                E.g. ``["JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]``.
            hidden_sizes: Hidden layer widths.  Defaults to
                ``[512, 512, 512, 512, 512]`` (6-layer network as in Parrot).
            n_epochs: Maximum training epochs.  Defaults to 1000.
            batch_size: Mini-batch size.  Defaults to 4096.
            learning_rate: Initial Adam learning rate.  Defaults to 1e-3.
            lr_decay_steps: Epochs at which LR is multiplied by the
                corresponding ``lr_decay_factors`` (3-phase decay as in
                Parrot).  Defaults to ``(300, 700)``.
            lr_decay_factors: Multiplicative LR factors at each decay step.
                Defaults to ``(0.1, 0.1)`` (1e-3 → 1e-4 → 1e-5).
            val_fraction: Fraction held out for validation.  Defaults to 0.1.
            early_stopping_patience: Stop if val loss does not improve for
                this many epochs.  Set to ``n_epochs`` to disable.
                Defaults to 20.
            seed: Random seed for reproducibility.
            log_interval: Log training/val loss every this many epochs.
            checkpoint_path: If given, save the best-validation-loss model
                here during training (safety checkpoint).

        Returns:
            Trained ParrotEmulator (best validation-loss weights).

        Raises:
            ImportError: If h5py or optax is not installed.
            KeyError: If a requested param_name or band_name is not found.
        """
        try:
            import h5py
            import optax
        except ImportError as e:
            raise ImportError(
                "h5py and optax are required for training. Install with: pip install h5py optax"
            ) from e

        if hidden_sizes is None:
            hidden_sizes = [512, 512, 512, 512, 512]

        library_path = Path(library_path)
        logger.info(f"Loading synference library: {library_path}")

        with h5py.File(library_path, "r") as f:
            raw_params = f["Grid/Parameters"][()]  # (N_p_all, N_models)
            raw_phot = f["Grid/Photometry"][()]  # (N_b_all, N_models)

            # --- Resolve parameter names from library metadata ---
            lib_param_names = _read_param_names(f)
            lib_band_names = _read_band_names(f)

        logger.info(
            f"Library: {raw_params.shape[1]} models, "
            f"{len(lib_param_names)} params, {len(lib_band_names)} bands"
        )

        param_indices = _select_indices(param_names, lib_param_names, "parameter")
        band_indices = _select_indices(band_names, lib_band_names, "band")

        params_np = raw_params[param_indices, :].T.astype(np.float32)  # (N, P)
        phot_np = raw_phot[band_indices, :].T.astype(np.float32)  # (N, B)

        # Remove rows with non-finite parameters (photometry can be ~0; arsinh handles that)
        valid = np.all(np.isfinite(params_np), axis=1)
        params_np = params_np[valid]
        phot_np = phot_np[valid]
        logger.info(f"After finite-param filter: {params_np.shape[0]} models")

        # Convert photometry to arsinh magnitudes (numpy-side, one-time)
        asinh_phot = _flux_to_asinh_mag_np(phot_np)

        # Train / validation split
        rng = np.random.default_rng(seed)
        n_total = params_np.shape[0]
        n_val = max(1, int(n_total * val_fraction))
        idx = rng.permutation(n_total)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        p_train, am_train = params_np[train_idx], asinh_phot[train_idx]
        p_val, am_val = params_np[val_idx], asinh_phot[val_idx]

        # Normalisation statistics from training set
        in_mean = p_train.mean(axis=0)
        in_std = p_train.std(axis=0) + 1e-8
        out_mean = am_train.mean(axis=0)
        out_std = am_train.std(axis=0) + 1e-8

        p_train_n = (p_train - in_mean) / in_std
        am_train_n = (am_train - out_mean) / out_std
        p_val_n = (p_val - in_mean) / in_std
        am_val_n = (am_val - out_mean) / out_std

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

        # 3-phase LR schedule: step functions at specified epoch boundaries
        def get_lr(epoch: int) -> float:
            lr = learning_rate
            for step_epoch, factor in zip(lr_decay_steps, lr_decay_factors):
                if epoch >= step_epoch:
                    lr *= factor
            return lr

        p_train_jax = jnp.array(p_train_n)
        am_train_jax = jnp.array(am_train_n)
        p_val_jax = jnp.array(p_val_n)
        am_val_jax = jnp.array(am_val_n)

        n_train = len(p_train_jax)
        n_steps_per_epoch = max(1, n_train // batch_size)
        logger.info(
            f"Training: {n_train} train / {len(p_val_jax)} val, "
            f"max {n_epochs} epochs, batch {batch_size}, "
            f"initial lr {learning_rate}, decay at epochs {lr_decay_steps}"
        )

        # Initialise optimiser with the initial LR; we rebuild opt_state at each
        # decay step to apply the new LR cleanly.
        current_lr = learning_rate
        optimiser = optax.nadam(current_lr)
        opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def step(
            model: "ParrotEmulator",
            opt_state,
            optimiser,
            p_batch: jnp.ndarray,
            am_batch: jnp.ndarray,
        ):
            def loss_fn(m):
                pred = jax.vmap(m._forward_normalised)(p_batch)
                return jnp.mean((pred - am_batch) ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, opt_state_new = optimiser.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            return eqx.apply_updates(model, updates), opt_state_new, loss

        @eqx.filter_jit
        def compute_val_loss(model: "ParrotEmulator", p_v, am_v):
            pred = jax.vmap(model._forward_normalised)(p_v)
            return jnp.mean((pred - am_v) ** 2)

        best_val = float("inf")
        best_model = model
        epochs_no_improve = 0
        key_epoch = jax.random.PRNGKey(seed + 1)

        for epoch in range(1, n_epochs + 1):
            # Update LR at decay boundaries (rebuild optimiser + re-init state)
            new_lr = get_lr(epoch)
            if new_lr != current_lr:
                logger.info(f"  LR decay at epoch {epoch}: {current_lr:.2e} → {new_lr:.2e}")
                current_lr = new_lr
                optimiser = optax.nadam(current_lr)
                opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

            # Shuffle
            key_epoch, subkey = jax.random.split(key_epoch)
            perm = jax.random.permutation(subkey, n_train)
            p_shuf = p_train_jax[perm]
            am_shuf = am_train_jax[perm]

            epoch_loss = 0.0
            for i in range(0, n_train, batch_size):
                pb = p_shuf[i : i + batch_size]
                amb = am_shuf[i : i + batch_size]
                model, opt_state, loss = step(model, opt_state, optimiser, pb, amb)
                epoch_loss += float(loss)

            v_loss = float(compute_val_loss(model, p_val_jax, am_val_jax))

            if v_loss < best_val:
                best_val = v_loss
                best_model = model
                epochs_no_improve = 0
                if checkpoint_path is not None:
                    best_model.save(checkpoint_path)
            else:
                epochs_no_improve += 1

            if epoch % log_interval == 0 or epoch == 1:
                t_loss = epoch_loss / max(1, n_steps_per_epoch)
                logger.info(
                    f"Epoch {epoch:4d}/{n_epochs}  "
                    f"train={t_loss:.5f}  val={v_loss:.5f}  "
                    f"best_val={best_val:.5f}"
                )

            if epochs_no_improve >= early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no val improvement for {early_stopping_patience} epochs)"
                )
                break

        logger.info(f"Training complete. Best val MSE (normalised asinh space): {best_val:.5f}")
        return best_model


# ---------------------------------------------------------------------------
# Numpy-side arsinh transform (used during data preprocessing only)
# ---------------------------------------------------------------------------


def _flux_to_asinh_mag_np(flux: np.ndarray) -> np.ndarray:
    """Convert flux array (nJy) to arsinh magnitudes using numpy.

    Used during training data preprocessing only (not on the JAX hot path).

    Args:
        flux: Flux array in nJy, any shape.

    Returns:
        arsinh magnitude array, same shape.
    """
    a = _ASINH_A
    mu0 = _ASINH_MU0
    b = np.exp(mu0 / a) / 2.0
    return -a * np.arcsinh(flux * b) + mu0


# ---------------------------------------------------------------------------
# Library metadata readers
# ---------------------------------------------------------------------------


def _read_param_names(f) -> list[str]:
    """Read SPS parameter names from a synference HDF5 file.

    Supports two formats:

    * **v4 format**: ``Model.attrs['varying_param_names']`` +
      ``Model.attrs['stellar_params']`` (order: varying first, then stellar).
    * **legacy format**: root ``attrs['ParameterNames']``.

    Args:
        f: Open h5py File handle.

    Returns:
        Ordered list of parameter name strings.
    """
    # v4 format
    if "Model" in f and "varying_param_names" in f["Model"].attrs:
        varying = _decode_str_list(f["Model"].attrs["varying_param_names"])
        stellar = _decode_str_list(f["Model"].attrs.get("stellar_params", []))
        return varying + stellar

    # Legacy root attrs
    if "ParameterNames" in f.attrs:
        return _decode_str_list(f.attrs["ParameterNames"])

    return []


def _read_band_names(f) -> list[str]:
    """Read photometric band names from a synference HDF5 file.

    Supports two formats:

    * **v4 format**: ``Model.attrs['filters']``.
    * **legacy format**: root ``attrs['FilterCodes']`` or
      dataset ``Grid/FilterCodes``.

    Args:
        f: Open h5py File handle.

    Returns:
        Ordered list of band name strings (same order as ``Grid/Photometry``
        rows).
    """
    # v4 format
    if "Model" in f and "filters" in f["Model"].attrs:
        return _decode_str_list(f["Model"].attrs["filters"])

    # Legacy root attrs
    if "FilterCodes" in f.attrs:
        return _decode_str_list(f.attrs["FilterCodes"])

    # Legacy dataset

    if "Grid/FilterCodes" in f:
        return _decode_str_list(f["Grid/FilterCodes"][()])

    return []


def _decode_str_list(attr) -> list[str]:
    """Decode an HDF5 attribute or dataset that may contain bytes or strings.

    Args:
        attr: HDF5 attribute value (bytes array, string array, or similar).

    Returns:
        List of Python strings.
    """
    result = []
    for item in attr:
        if isinstance(item, (bytes, np.bytes_)):
            decoded = (
                item.decode("utf-8") if isinstance(item, bytes) else item.tobytes().decode("utf-8")
            )
            result.append(decoded)
        else:
            result.append(str(item))
    return result


def _select_indices(requested: list[str], available: list[str], kind: str) -> list[int]:
    """Return indices of requested names in the available list.

    Args:
        requested: Names the caller wants.
        available: Names present in the library.
        kind: Label for error messages (e.g. ``'parameter'``).

    Returns:
        List of integer indices into ``available``.

    Raises:
        KeyError: If any requested name is not in available.
    """
    indices = []
    for name in requested:
        if name not in available:
            raise KeyError(
                f"Requested {kind} '{name}' not found in library. Available: {available}"
            )
        indices.append(available.index(name))
    return indices
