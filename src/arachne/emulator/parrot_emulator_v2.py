"""ParrotEmulatorV2: improved SPS photometry emulator.

Improvements over :class:`~arachne.emulator.parrot_emulator.ParrotEmulator`
(each validated on a held-out split of the v6 BPASS library — see
``scripts/experiments/``):

1. **Analytic mass factorisation** — photometry is exactly linear in stellar
   mass, so the network predicts flux per 10^9 Msun and ``log_mass`` is
   reapplied analytically at inference.  Removes ~80 % of the target variance.
2. **Fourier redshift features** — sin/cos encodings of redshift let the
   network represent the sharp Lyman-break flux cutoff without needing
   extreme depth.  This is the single largest accuracy win (~2x).
3. **arsinh compression of logsfr_ratio inputs** — the Student-t-distributed
   SFH ratios span ±30 but saturate photometrically beyond |x| ~ 3; arsinh
   preserves the informative core that plain z-scoring squashes.
4. **Modern training recipe** — AdamW-style warmup + cosine decay, MSE loss,
   whole-dataset-on-GPU scan training (minutes per run on an H100).

The public API matches ParrotEmulator: ``predict(params) -> nJy`` where
``params`` has the full library parameter vector (including ``log_mass``),
plus self-contained ``save``/``load``.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from arachne.emulator.base import SPSEmulator
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

_ASINH_A: float = 2.5 * np.log10(np.e)  # ≈ 1.0857
DEFAULT_FLUX_FLOOR: float = 1e-4  # nJy (per 1e9 Msun in unit-flux space)
_Z_SCALE: float = 14.0  # redshift normalisation for Fourier features


def _mu0_from_floor(floor: float) -> float:
    return _ASINH_A * float(np.log(2.0 / floor))


def _flux_to_mag(flux, mu0):
    b = jnp.exp(mu0 / _ASINH_A) / 2.0
    return -_ASINH_A * jnp.arcsinh(flux * b) + mu0


def _mag_to_flux(mag, mu0):
    return 2.0 * jnp.exp(-mu0 / _ASINH_A) * jnp.sinh((mu0 - mag) / _ASINH_A)


def _flux_to_mag_np(flux, mu0):
    b = np.exp(mu0 / _ASINH_A) / 2.0
    return -_ASINH_A * np.arcsinh(flux * b) + mu0


# ---------------------------------------------------------------------------
# network bodies
# ---------------------------------------------------------------------------


class _MLPNet(eqx.Module):
    layers: tuple

    def __init__(self, n_in, n_out, width, depth, key):
        sizes = [n_in] + [width] * depth + [n_out]
        ks = jax.random.split(key, len(sizes) - 1)
        self.layers = tuple(
            eqx.nn.Linear(a, b, key=k) for a, b, k in zip(sizes[:-1], sizes[1:], ks)
        )

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x), approximate=True)
        return self.layers[-1](x)


class _ResBlock(eqx.Module):
    ln: eqx.nn.LayerNorm
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, width, key):
        k1, k2 = jax.random.split(key)
        self.ln = eqx.nn.LayerNorm(width)
        self.l1 = eqx.nn.Linear(width, width, key=k1)
        self.l2 = eqx.nn.Linear(width, width, key=k2)

    def __call__(self, x):
        h = self.ln(x)
        h = jax.nn.gelu(self.l1(h), approximate=True)
        return x + self.l2(h)


class _ResMLPNet(eqx.Module):
    inproj: eqx.nn.Linear
    blocks: tuple
    ln_f: eqx.nn.LayerNorm
    out: eqx.nn.Linear

    def __init__(self, n_in, n_out, width, n_blocks, key):
        ks = jax.random.split(key, n_blocks + 2)
        self.inproj = eqx.nn.Linear(n_in, width, key=ks[0])
        self.blocks = tuple(_ResBlock(width, k) for k in ks[1:-1])
        self.ln_f = eqx.nn.LayerNorm(width)
        self.out = eqx.nn.Linear(width, n_out, key=ks[-1])

    def __call__(self, x):
        h = jax.nn.gelu(self.inproj(x), approximate=True)
        for b in self.blocks:
            h = b(h)
        return self.out(self.ln_f(h))


def _build_net(cfg: dict, n_in: int, n_out: int, key):
    if cfg["arch"] == "mlp":
        return _MLPNet(n_in, n_out, cfg["width"], cfg["depth"], key)
    return _ResMLPNet(n_in, n_out, cfg["width"], cfg["blocks"], key)


# ---------------------------------------------------------------------------
# emulator
# ---------------------------------------------------------------------------


class ParrotEmulatorV2(SPSEmulator):
    """Mass-factorised, Fourier-featured SPS photometry emulator.

    ``predict`` takes the raw library parameter vector (same order as
    ``param_names``, including ``log_mass``) and returns nJy fluxes, so the
    class is a drop-in replacement for ParrotEmulator in the fitting scripts.
    """

    net: _MLPNet | _ResMLPNet
    in_mean: jnp.ndarray
    in_std: jnp.ndarray
    out_mean: jnp.ndarray
    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)
    _cfg: str = eqx.field(static=True)  # JSON string (hashable for jit)

    def __init__(self, param_names, band_names, cfg, in_mean, in_std, out_mean, key):
        """Build an untrained emulator with the given normalisation statistics."""
        self._param_names = list(param_names)
        self._band_names = list(band_names)
        cfg = dict(cfg)
        cfg["n_in"] = int(np.asarray(in_mean).shape[0])
        cfg["n_out"] = len(band_names)
        self._cfg = json.dumps(cfg, sort_keys=True)
        self.in_mean = jnp.asarray(in_mean, dtype=jnp.float32)
        self.in_std = jnp.asarray(in_std, dtype=jnp.float32)
        self.out_mean = jnp.asarray(out_mean, dtype=jnp.float32)
        self.net = _build_net(cfg, cfg["n_in"], cfg["n_out"], key)

    # -- config helpers ------------------------------------------------------
    @property
    def cfg(self) -> dict:
        """Architecture/feature configuration as a dict (parsed from the stored JSON)."""
        return json.loads(self._cfg)

    @property
    def param_names(self) -> list[str]:
        """Ordered emulator input parameter names."""
        return self._param_names

    @property
    def band_names(self) -> list[str]:
        """Ordered output band names."""
        return self._band_names

    # Compatibility with validate/diagnose scripts written for V1: metrics are
    # computed in absolute-flux asinh space with the standard floor.
    @property
    def _asinh_mu0(self) -> float:
        return _mu0_from_floor(self.cfg["flux_floor"])

    @property
    def _flux_floor(self) -> float:
        return self.cfg["flux_floor"]

    # -- feature transform ----------------------------------------------------
    def _features(self, params: jnp.ndarray) -> jnp.ndarray:
        """Raw params (N, P) -> network features (N, F). JIT-compatible."""
        cfg = self.cfg
        keep = tuple(cfg["keep_idx"])
        x = params[:, keep]
        if cfg["sfr_arsinh"] and cfg["sfr_cols"]:
            cols = jnp.array(cfg["sfr_cols"])
            x = x.at[:, cols].set(jnp.arcsinh(x[:, cols]))
        feats = [x]
        k, kmax = 1, cfg["fourier_k"]
        if kmax > 0:
            zn = params[:, cfg["i_z"] : cfg["i_z"] + 1] / _Z_SCALE
            while k <= kmax:
                feats.append(jnp.sin(2 * jnp.pi * k * zn))
                feats.append(jnp.cos(2 * jnp.pi * k * zn))
                k *= 2
        return jnp.concatenate(feats, axis=1)

    # -- inference -------------------------------------------------------------
    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Predict photometry (nJy) from raw SPS parameters (N, P)."""
        cfg = self.cfg
        params = jnp.atleast_2d(params)
        x = (self._features(params) - self.in_mean) / self.in_std
        mag = jax.vmap(self.net)(x) + self.out_mean
        flux = _mag_to_flux(mag, cfg["mu0_train"])
        if cfg["mass_norm"]:
            m9 = 10.0 ** (params[:, cfg["i_mass"]] - 9.0)
            flux = flux * m9[:, None]
        return flux

    # -- persistence -----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialise config, names and weights to a self-contained ``.eqx`` (npz) file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        eqx.tree_serialise_leaves(buf, self)
        with open(path, "wb") as f:
            np.savez(
                f,
                v2_config=np.array(self._cfg),
                param_names=np.array(self._param_names),
                band_names=np.array(self._band_names),
                weights_bytes=np.frombuffer(buf.getvalue(), dtype=np.uint8),
            )
        logger.info(f"ParrotEmulatorV2 saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ParrotEmulatorV2":
        """Load a checkpoint written by :meth:`save`."""
        with np.load(str(path), allow_pickle=False) as raw:
            if "v2_config" not in raw:
                raise ValueError(
                    f"{path} is not a ParrotEmulatorV2 checkpoint; use ParrotEmulator.load instead."
                )
            cfg = json.loads(str(raw["v2_config"]))
            param_names = raw["param_names"].tolist()
            band_names = raw["band_names"].tolist()
            weights = bytes(raw["weights_bytes"])
        dummy = cls(
            param_names,
            band_names,
            cfg,
            in_mean=np.zeros(cfg["n_in"], np.float32),
            in_std=np.ones(cfg["n_in"], np.float32),
            out_mean=np.zeros(cfg["n_out"], np.float32),
            key=jax.random.PRNGKey(0),
        )
        loaded = eqx.tree_deserialise_leaves(io.BytesIO(weights), dummy)
        logger.info(f"ParrotEmulatorV2 loaded from {path}")
        return loaded

    # -- training ---------------------------------------------------------------
    @classmethod
    def from_synference_library(  # noqa: C901
        cls,
        library_path: str | Path,
        band_names: list[str],
        param_names: list[str] | str = "all",
        arch: str = "mlp",
        width: int = 512,
        depth: int = 5,
        blocks: int = 4,
        mass_norm: bool = True,
        keep_mass_input: bool = False,
        sfr_arsinh: bool = True,
        fourier_k: int = 16,
        flux_floor: float = DEFAULT_FLUX_FLOOR,
        n_epochs: int = 1500,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        min_lr_frac: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_epochs: int = 3,
        val_fraction: float = 0.025,
        seed: int = 0,
        log_interval: int = 25,
        checkpoint_path: str | Path | None = None,
        exclude_rows: np.ndarray | None = None,
    ) -> "ParrotEmulatorV2":
        """Train from a synference HDF5 library with the modern recipe.

        The full training set lives on the GPU and each epoch is a single
        ``lax.scan`` over shuffled mini-batches; the best-validation-loss
        weights are returned.  ``exclude_rows`` (library row indices) lets a
        caller hold out an external test set.
        """
        import h5py
        import optax

        jax.config.update("jax_default_matmul_precision", "highest")

        with h5py.File(library_path, "r") as f:
            lib_p = [str(x) for x in f.attrs["ParameterNames"]]
            lib_b = [str(x) for x in f.attrs["FilterCodes"]]
            P = f["Grid/Parameters"][()]
            F = f["Grid/Photometry"][()]
        if param_names == "all":
            param_names = lib_p
        pi = [lib_p.index(n) for n in param_names]
        bi = [lib_b.index(n) for n in band_names]
        params_raw = P[pi].T.astype(np.float64)
        flux_raw = F[bi].T.astype(np.float64)
        ok = np.all(np.isfinite(params_raw), axis=1)
        params_raw, flux_raw = params_raw[ok], flux_raw[ok]
        if exclude_rows is not None:
            mask = np.ones(len(params_raw), bool)
            mask[exclude_rows] = False
            params_raw, flux_raw = params_raw[mask], flux_raw[mask]
        logger.info(
            f"Library: {len(params_raw)} usable rows, "
            f"{len(param_names)} params, {len(band_names)} bands"
        )

        i_mass = param_names.index("log_mass") if "log_mass" in param_names else -1
        i_z = param_names.index("redshift")
        if mass_norm and i_mass < 0:
            raise ValueError("mass_norm=True requires a 'log_mass' parameter")

        # targets
        if mass_norm:
            m9 = 10.0 ** (params_raw[:, i_mass] - 9.0)
            unit_floor = flux_floor / m9.max()
            mu0_train = _mu0_from_floor(unit_floor)
            f_t = flux_raw / m9[:, None]
            f_t = np.where(f_t < unit_floor, 0.0, f_t)
        else:
            mu0_train = _mu0_from_floor(flux_floor)
            f_t = np.where(flux_raw < flux_floor, 0.0, flux_raw)
        y_all = _flux_to_mag_np(f_t, mu0_train)

        keep_idx = [
            i
            for i in range(len(param_names))
            if not (mass_norm and not keep_mass_input and i == i_mass)
        ]
        sfr_cols = (
            [
                keep_idx.index(i)
                for i, n in enumerate(param_names)
                if n.startswith("logsfr_ratio") and i in keep_idx
            ]
            if sfr_arsinh
            else []
        )

        cfg = dict(
            arch=arch,
            width=width,
            depth=depth,
            blocks=blocks,
            mass_norm=mass_norm,
            keep_mass_input=keep_mass_input,
            sfr_arsinh=sfr_arsinh,
            fourier_k=fourier_k,
            flux_floor=flux_floor,
            mu0_train=mu0_train,
            i_mass=i_mass,
            i_z=i_z,
            keep_idx=keep_idx,
            sfr_cols=sfr_cols,
        )

        # features via a temporary instance (needs normalisation set first)
        proto = cls(
            param_names,
            band_names,
            cfg,
            in_mean=np.zeros(len(keep_idx), np.float32),
            in_std=np.ones(len(keep_idx), np.float32),
            out_mean=np.zeros(len(band_names), np.float32),
            key=jax.random.PRNGKey(seed),
        )
        # proto n_in was wrong pre-fourier; recompute features in numpy instead
        x_all = np.asarray(
            jax.device_get(proto._features(jnp.asarray(params_raw, dtype=jnp.float64)))
        )

        rng = np.random.default_rng(seed)
        n = len(x_all)
        n_val = max(1, int(n * val_fraction))
        perm = rng.permutation(n)
        va, tr = perm[:n_val], perm[n_val:]

        in_mean = x_all[tr].mean(axis=0)
        in_std = x_all[tr].std(axis=0) + 1e-8
        out_mean = y_all[tr].mean(axis=0)

        model = cls(
            param_names, band_names, cfg, in_mean, in_std, out_mean, key=jax.random.PRNGKey(seed)
        )

        xt = jnp.array((x_all[tr] - in_mean) / in_std, dtype=jnp.float32)
        yt = jnp.array(y_all[tr] - out_mean, dtype=jnp.float32)
        xv = jnp.array((x_all[va] - in_mean) / in_std, dtype=jnp.float32)
        yv = jnp.array(y_all[va] - out_mean, dtype=jnp.float32)

        n_in, n_out = xt.shape[1], yt.shape[1]
        n_train = xt.shape[0]
        steps_per_epoch = n_train // batch_size
        sched = optax.warmup_cosine_decay_schedule(
            0.0,
            learning_rate,
            warmup_epochs * steps_per_epoch,
            n_epochs * steps_per_epoch,
            learning_rate * min_lr_frac,
        )
        optim = (
            optax.adamw(sched, weight_decay=weight_decay) if weight_decay > 0 else optax.adam(sched)
        )

        net_params, net_static = eqx.partition(model.net, eqx.is_array)
        opt_state = optim.init(net_params)

        @jax.jit
        def train_epoch(net_params, opt_state, perm):
            xs = xt[perm].reshape(steps_per_epoch, batch_size, n_in)
            ys = yt[perm].reshape(steps_per_epoch, batch_size, n_out)

            def step(carry, batch):
                p, s = carry
                xb, yb = batch

                def loss_fn(p):
                    net = eqx.combine(p, net_static)
                    return jnp.mean((jax.vmap(net)(xb) - yb) ** 2)

                loss, g = jax.value_and_grad(loss_fn)(p)
                upd, s = optim.update(g, s, p)
                return (optax.apply_updates(p, upd), s), loss

            (net_params, opt_state), losses = jax.lax.scan(step, (net_params, opt_state), (xs, ys))
            return net_params, opt_state, jnp.mean(losses)

        @jax.jit
        def val_loss(p):
            net = eqx.combine(p, net_static)
            return jnp.mean((jax.vmap(net)(xv) - yv) ** 2)

        logger.info(
            f"Training {arch} width={width} "
            f"{'depth=' + str(depth) if arch == 'mlp' else 'blocks=' + str(blocks)} "
            f"n_in={n_in} n_out={n_out}: {n_epochs} epochs x {steps_per_epoch} steps, "
            f"batch={batch_size}, lr={learning_rate:.1e}, wd={weight_decay:.1e}"
        )

        best_val, best_np = np.inf, net_params
        key_ep = jax.random.PRNGKey(seed + 1)
        t0 = time.time()
        for ep in range(1, n_epochs + 1):
            key_ep, sk = jax.random.split(key_ep)
            perm_ep = jax.random.permutation(sk, n_train)[: steps_per_epoch * batch_size]
            net_params, opt_state, tl = train_epoch(net_params, opt_state, perm_ep)
            vl = float(val_loss(net_params))
            if vl < best_val:
                best_val, best_np = vl, net_params
                if checkpoint_path is not None and ep % 50 == 0:
                    obj = eqx.tree_at(lambda m: m.net, model, eqx.combine(best_np, net_static))
                    obj.save(checkpoint_path)
            if ep % log_interval == 0 or ep == 1:
                logger.info(
                    f"  ep {ep:5d}/{n_epochs} train={float(tl):.6f} "
                    f"val={vl:.6f} best={best_val:.6f} "
                    f"({time.time() - t0:.0f}s)"
                )
        logger.info(f"Training complete in {time.time() - t0:.0f}s; best val MSE {best_val:.6f}")
        return eqx.tree_at(lambda m: m.net, model, eqx.combine(best_np, net_static))


def load_emulator(path: str | Path):
    """Load a Parrot checkpoint of either generation (V2 or V1)."""
    with np.load(str(path), allow_pickle=False) as raw:
        is_v2 = "v2_config" in raw
    if is_v2:
        return ParrotEmulatorV2.load(path)
    from arachne.emulator.parrot_emulator import ParrotEmulator

    return ParrotEmulator.load(path)
