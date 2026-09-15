#!/usr/bin/env python3
"""Experiment framework for ParrotEmulator architecture/preprocessing ablations.

Trains emulator variants on the v6 BPASS library with a FIXED held-out test
split (never seen in training) and reports metrics in the same absolute
asinh-magnitude space as validate_parrot_emulator.py, so results are directly
comparable across variants and to the production checkpoint.

Variants
--------
--mass-norm      Factor out log_mass analytically: the network predicts
                 flux per 10^9 Msun (photometry is exactly linear in mass);
                 log_mass is removed from the inputs.
--sfr-arsinh     arsinh-compress the heavy-tailed logsfr_ratio inputs.
--fourier-k K    Append Fourier features of redshift (frequencies 1,2,4..K
                 cycles over the z range) to sharpen the Lyman-break response.
--arch           mlp (plain GELU stack) or resmlp (pre-LN residual blocks).
--ema            Evaluate an exponential moving average of the weights.

Metrics (on the held-out test rows, always in absolute-flux asinh space
with mu0 = a*ln(2/1e-4), floor 1e-4 nJy applied to the true fluxes):
  med_scatter   median over bands of per-band residual std   [headline]
  med_bias      median over bands of per-band |mean residual|
  med_p95       median over bands of per-band 95th pct |residual|
  rmse          overall RMSE
  trans_rms     RMS over pixels with true flux in [1e-4, 1e-2] nJy
                (Lyman-break dropout transition — hardest regime)
  bright_flux_err  median over bands of median |dF|/F for F_true > 10 nJy
                (compare to the Parrot <1% benchmark)

Example:
-------
python emulator_lab.py --name massnorm_fourier --mass-norm --sfr-arsinh \
    --fourier-k 16 --arch resmlp --width 512 --blocks 4 --epochs 150
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LIBRARY = (
    "/cosma/apps/dp276/dc-harv3/synference/libraries/"
    "library_BPASS_Chab_Continuity_SFH_0.01_z_14_logN_6.0_Calzetti_IGM_Asada25_v6_multinode.hdf5"
)
REF_CHECKPOINT = "/cosma/apps/dp276/dc-harv3/arachne/scripts/outputs/emulators/parrot_emulator.eqx"
OUT_DIR = Path("/cosma/apps/dp276/dc-harv3/arachne/scripts/outputs/experiments")

SPLIT_SEED = 123
N_TEST = 50_000
N_VAL = 25_000

FLUX_FLOOR = 1e-4  # nJy (or nJy per 1e9 Msun in mass-normalised space)


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--library", default=LIBRARY)
    p.add_argument("--arch", choices=["mlp", "resmlp"], default="mlp")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=5, help="hidden layers (mlp)")
    p.add_argument("--blocks", type=int, default=4, help="residual blocks (resmlp)")
    p.add_argument("--mass-norm", action="store_true")
    p.add_argument(
        "--keep-mass-input",
        action="store_true",
        help="with --mass-norm: still feed log_mass as an input feature",
    )
    p.add_argument("--sfr-arsinh", action="store_true")
    p.add_argument(
        "--fourier-k",
        type=int,
        default=0,
        help="max Fourier frequency (powers of 2 up to K); 0 = off",
    )
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-frac", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay, 0 = off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=0, help="subsample train set; 0 = all")
    p.add_argument("--save-weights", action="store_true")
    p.add_argument(
        "--eval-checkpoint",
        default=None,
        help="skip training; evaluate an existing ParrotEmulator .eqx on the test split",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

_DATA_CACHE = {}


def load_data(library):
    """Load parameters, fluxes and metadata from a synference HDF5 library."""
    if library in _DATA_CACHE:
        return _DATA_CACHE[library]
    import h5py
    import numpy as np

    ck = np.load(REF_CHECKPOINT, allow_pickle=False)
    param_names = ck["param_names"].tolist()
    band_names = ck["band_names"].tolist()

    with h5py.File(library, "r") as f:
        lib_p = [str(x) for x in f.attrs["ParameterNames"]]
        lib_b = [str(x) for x in f.attrs["FilterCodes"]]
        P = f["Grid/Parameters"][()]
        F = f["Grid/Photometry"][()]
    pi = [lib_p.index(n) for n in param_names]
    bi = [lib_b.index(n) for n in band_names]
    params = P[pi].T.astype(np.float64)  # (N, 12)
    flux = F[bi].T.astype(np.float64)  # (N, 54) nJy
    ok = np.all(np.isfinite(params), axis=1)
    params, flux = params[ok], flux[ok]

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(params))
    test = perm[:N_TEST]
    val = perm[N_TEST : N_TEST + N_VAL]
    train = perm[N_TEST + N_VAL :]
    out = (param_names, band_names, params, flux, train, val, test)
    _DATA_CACHE[library] = out
    return out


ASINH_A = 1.0857362047581294  # 2.5 log10(e)
MU0 = ASINH_A * float(__import__("numpy").log(2.0 / FLUX_FLOOR))  # 10.7526


def flux_to_mag(f, mu0=MU0):
    """Arsinh-magnitude transform of flux with softening set by ``mu0``."""
    import numpy as np

    b = np.exp(mu0 / ASINH_A) / 2.0
    return -ASINH_A * np.arcsinh(f * b) + mu0


def mag_to_flux(m, mu0=MU0):
    """Inverse of :func:`flux_to_mag`."""
    import numpy as np

    return 2.0 * np.exp(-mu0 / ASINH_A) * np.sinh((mu0 - m) / ASINH_A)


# ---------------------------------------------------------------------------
# feature transform
# ---------------------------------------------------------------------------


class FeatureSpec:
    """Maps raw library params (N,12) -> network input features (numpy)."""

    def __init__(self, param_names, mass_norm, sfr_arsinh, fourier_k, keep_mass_input=False):
        """Configure the feature transform for the given ablation flags."""
        self.param_names = param_names
        self.mass_norm = mass_norm
        self.sfr_arsinh = sfr_arsinh
        self.fourier_k = fourier_k
        self.i_mass = param_names.index("log_mass")
        self.i_z = param_names.index("redshift")
        self.sfr_idx = [i for i, n in enumerate(param_names) if n.startswith("logsfr_ratio")]
        self.keep = [
            i
            for i in range(len(param_names))
            if not (mass_norm and not keep_mass_input and i == self.i_mass)
        ]

    def __call__(self, raw):
        """Map raw library parameters to network input features."""
        import numpy as np

        x = raw[:, self.keep].copy()
        if self.sfr_arsinh:
            cols = [self.keep.index(i) for i in self.sfr_idx]
            x[:, cols] = np.arcsinh(x[:, cols])
        feats = [x]
        if self.fourier_k > 0:
            zn = raw[:, self.i_z : self.i_z + 1] / 14.0
            k = 1
            while k <= self.fourier_k:
                feats.append(np.sin(2 * np.pi * k * zn))
                feats.append(np.cos(2 * np.pi * k * zn))
                k *= 2
        return np.concatenate(feats, axis=1)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def build_model(arch, n_in, n_out, width, depth, blocks, key):
    """Construct the network body for ``arch`` with the given width and depth."""
    import equinox as eqx
    import jax

    if arch == "mlp":

        class MLP(eqx.Module):
            layers: tuple

            def __init__(self, key):
                sizes = [n_in] + [width] * depth + [n_out]
                ks = jax.random.split(key, len(sizes) - 1)
                self.layers = tuple(
                    eqx.nn.Linear(a, b, key=k) for a, b, k in zip(sizes[:-1], sizes[1:], ks)
                )

            def __call__(self, x):
                for layer in self.layers[:-1]:
                    x = jax.nn.gelu(layer(x), approximate=True)
                return self.layers[-1](x)

        return MLP(key)

    class Block(eqx.Module):
        ln: object
        l1: object
        l2: object

        def __init__(self, key):
            import equinox as eqx

            k1, k2 = jax.random.split(key)
            self.ln = eqx.nn.LayerNorm(width)
            self.l1 = eqx.nn.Linear(width, width, key=k1)
            self.l2 = eqx.nn.Linear(width, width, key=k2)

        def __call__(self, x):
            h = self.ln(x)
            h = jax.nn.gelu(self.l1(h), approximate=True)
            return x + self.l2(h)

    class ResMLP(eqx.Module):
        inproj: object
        blocks: tuple
        ln_f: object
        out: object

        def __init__(self, key):
            import equinox as eqx

            ks = jax.random.split(key, blocks + 2)
            self.inproj = eqx.nn.Linear(n_in, width, key=ks[0])
            self.blocks = tuple(Block(k) for k in ks[1:-1])
            self.ln_f = eqx.nn.LayerNorm(width)
            self.out = eqx.nn.Linear(width, n_out, key=ks[-1])

        def __call__(self, x):
            h = jax.nn.gelu(self.inproj(x), approximate=True)
            for b in self.blocks:
                h = b(h)
            return self.out(self.ln_f(h))

    return ResMLP(key)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    """Train and evaluate one emulator configuration from the command line."""
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import numpy as np

    param_names, band_names, params_raw, flux_raw, tr, va, te = load_data(args.library)
    print(f"data: {len(params_raw)} rows, train={len(tr)} val={len(va)} test={len(te)}")

    # ---- reference truth on test split (absolute space, floored) ----------
    flux_test_true = np.where(flux_raw[te] < FLUX_FLOOR, 0.0, flux_raw[te])
    mag_test_true = flux_to_mag(flux_test_true)

    if args.eval_checkpoint:
        import jax.numpy as jnp

        from arachne.emulator.parrot_emulator_v2 import load_emulator

        emu = load_emulator(args.eval_checkpoint)
        pred = []
        for i in range(0, len(te), 65536):
            pred.append(
                np.asarray(emu.predict(jnp.array(params_raw[te[i : i + 65536]], dtype=jnp.float32)))
            )
        pred_flux = np.concatenate(pred).astype(np.float64)
        report(
            args,
            band_names,
            params_raw[te],
            flux_test_true,
            mag_test_true,
            pred_flux,
            n_params=None,
            train_s=0.0,
        )
        return 0

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_default_matmul_precision", "highest")

    # ---- targets -----------------------------------------------------------
    if args.mass_norm:
        m9 = 10.0 ** (params_raw[:, param_names.index("log_mass")] - 9.0)
        # Unit-flux floor must be scaled by the max mass so that near-floor
        # prediction noise, multiplied back by up to m9_max, stays below the
        # absolute FLUX_FLOOR (otherwise high-mass dropouts leak flux).
        unit_floor = FLUX_FLOOR / m9.max()
        mu0_train = ASINH_A * float(np.log(2.0 / unit_floor))
        f_unit = flux_raw / m9[:, None]
        f_unit = np.where(f_unit < unit_floor, 0.0, f_unit)
        y_all = flux_to_mag(f_unit, mu0_train)
    else:
        mu0_train = MU0
        f_ab = np.where(flux_raw < FLUX_FLOOR, 0.0, flux_raw)
        y_all = flux_to_mag(f_ab, mu0_train)

    fs = FeatureSpec(
        param_names,
        args.mass_norm,
        args.sfr_arsinh,
        args.fourier_k,
        keep_mass_input=args.keep_mass_input,
    )
    x_all = fs(params_raw)

    tr_use = tr if args.n_train == 0 else tr[: args.n_train]

    in_mean = x_all[tr_use].mean(axis=0)
    in_std = x_all[tr_use].std(axis=0) + 1e-8
    out_mean = y_all[tr_use].mean(axis=0)

    xt = jnp.array((x_all[tr_use] - in_mean) / in_std, dtype=jnp.float32)
    yt = jnp.array(y_all[tr_use] - out_mean, dtype=jnp.float32)
    xv = jnp.array((x_all[va] - in_mean) / in_std, dtype=jnp.float32)
    yv = jnp.array(y_all[va] - out_mean, dtype=jnp.float32)

    n_in, n_out = xt.shape[1], yt.shape[1]
    key = jax.random.PRNGKey(args.seed)
    key, mk = jax.random.split(key)
    model = build_model(args.arch, n_in, n_out, args.width, args.depth, args.blocks, mk)
    n_params = sum(
        int(np.prod(leaf.shape))
        for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    print(f"model: {args.arch} in={n_in} out={n_out} params={n_params:,}")

    n_train = xt.shape[0]
    steps_per_epoch = n_train // args.batch
    total_steps = steps_per_epoch * args.epochs
    sched = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, args.warmup_epochs * steps_per_epoch, total_steps, args.lr * args.min_lr_frac
    )
    optim = (
        optax.adamw(sched, weight_decay=args.weight_decay)
        if args.weight_decay > 0
        else optax.adam(sched)
    )

    fparams, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(fparams)
    ema_params = fparams

    @jax.jit
    def train_epoch(fparams, opt_state, ema_params, perm):
        xs = xt[perm].reshape(steps_per_epoch, args.batch, n_in)
        ys = yt[perm].reshape(steps_per_epoch, args.batch, n_out)

        def step(carry, batch):
            fp, os_, ema = carry
            xb, yb = batch

            def loss_fn(fp):
                m = eqx.combine(fp, static)
                pred = jax.vmap(m)(xb)
                return jnp.mean((pred - yb) ** 2)

            loss, g = jax.value_and_grad(loss_fn)(fp)
            upd, os_ = optim.update(g, os_, fp)
            fp = optax.apply_updates(fp, upd)
            if args.ema > 0:
                ema = jax.tree.map(lambda e, p: args.ema * e + (1 - args.ema) * p, ema, fp)
            return (fp, os_, ema), loss

        (fparams, opt_state, ema_params), losses = jax.lax.scan(
            step, (fparams, opt_state, ema_params), (xs, ys)
        )
        return fparams, opt_state, ema_params, jnp.mean(losses)

    @jax.jit
    def val_loss(fp):
        m = eqx.combine(fp, static)
        pred = jax.vmap(m)(xv)
        return jnp.mean((pred - yv) ** 2)

    best_val, best_params = np.inf, fparams
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, n_train)[: steps_per_epoch * args.batch]
        fparams, opt_state, ema_params, tl = train_epoch(fparams, opt_state, ema_params, perm)
        eval_p = ema_params if args.ema > 0 else fparams
        vl = float(val_loss(eval_p))
        if vl < best_val:
            best_val, best_params = vl, eval_p
        if ep % 10 == 0 or ep == 1:
            print(
                f"  ep {ep:4d} train={float(tl):.6f} val={vl:.6f} "
                f"best={best_val:.6f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    train_s = time.time() - t0
    print(f"training done in {train_s:.0f}s, best val MSE {best_val:.6f}")

    # ---- predict test in absolute flux space -------------------------------
    model = eqx.combine(best_params, static)
    xte = jnp.array((x_all[te] - in_mean) / in_std, dtype=jnp.float32)

    @jax.jit
    def predict(xb):
        return jax.vmap(model)(xb)

    preds = []
    for i in range(0, len(te), 65536):
        preds.append(np.asarray(predict(xte[i : i + 65536])))
    y_pred = np.concatenate(preds).astype(np.float64) + out_mean

    pred_flux_space = mag_to_flux(y_pred, mu0_train)  # unit flux if mass_norm else absolute
    if args.mass_norm:
        m9_te = 10.0 ** (params_raw[te, param_names.index("log_mass")] - 9.0)
        pred_flux = pred_flux_space * m9_te[:, None]
    else:
        pred_flux = pred_flux_space

    if args.save_weights:
        import io as _io

        buf = _io.BytesIO()
        eqx.tree_serialise_leaves(buf, model)
        np.savez(
            OUT_DIR / f"{args.name}_weights.npz",
            weights_bytes=np.frombuffer(buf.getvalue(), dtype=np.uint8),
            in_mean=in_mean,
            in_std=in_std,
            out_mean=out_mean,
            config=json.dumps(vars(args)),
        )

    report(
        args,
        band_names,
        params_raw[te],
        flux_test_true,
        mag_test_true,
        pred_flux,
        n_params,
        train_s,
    )
    return 0


def report(args, band_names, params_te, flux_true, mag_true, pred_flux, n_params, train_s):
    """Print and save accuracy metrics for a trained emulator."""
    import numpy as np

    np.savez_compressed(OUT_DIR / f"{args.name}_preds.npz", pred_flux=pred_flux.astype(np.float32))

    mag_pred = flux_to_mag(pred_flux)
    r = mag_pred - mag_true

    bias = r.mean(axis=0)
    scatter = r.std(axis=0)
    p95 = np.percentile(np.abs(r), 95, axis=0)

    trans = (flux_true > 0) & (flux_true < 1e-2)
    bright = flux_true > 10.0
    bright_err = []
    det_scatter = []
    for b in range(r.shape[1]):
        m = bright[:, b]
        if m.sum() > 100:
            bright_err.append(np.median(np.abs(pred_flux[m, b] / flux_true[m, b] - 1)))
        d = flux_true[:, b] > 1.0
        if d.sum() > 100:
            det_scatter.append(r[d, b].std())
    # Dropout leakage: predicted absolute flux where the truth is a hard zero.
    zero = flux_true == 0
    leak_p95 = float(np.percentile(pred_flux[zero], 95)) if zero.any() else 0.0
    leak_max = float(pred_flux[zero].max()) if zero.any() else 0.0
    res = {
        "name": args.name,
        "config": {k: v for k, v in vars(args).items() if k != "library"},
        "n_params": n_params,
        "train_seconds": round(train_s, 1),
        "med_scatter": float(np.median(scatter)),
        "med_abs_bias": float(np.median(np.abs(bias))),
        "med_p95": float(np.median(p95)),
        "rmse": float(np.sqrt(np.mean(r**2))),
        "trans_rms": float(np.sqrt(np.mean(r[trans] ** 2))),
        "bright_flux_err": float(np.median(bright_err)),
        "det_scatter": float(np.median(det_scatter)),
        "leak_p95_nJy": leak_p95,
        "leak_max_nJy": leak_max,
        "worst_band_scatter": float(scatter.max()),
        "per_band": {
            b: {"bias": float(bias[i]), "scatter": float(scatter[i]), "p95": float(p95[i])}
            for i, b in enumerate(band_names)
        },
    }
    out = OUT_DIR / f"{args.name}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n==== {args.name} ====")
    for k in [
        "med_scatter",
        "med_abs_bias",
        "med_p95",
        "rmse",
        "trans_rms",
        "bright_flux_err",
        "det_scatter",
        "leak_p95_nJy",
        "leak_max_nJy",
        "worst_band_scatter",
        "n_params",
        "train_seconds",
    ]:
        print(f"  {k:18s}: {res[k]}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    import sys

    sys.exit(main())
