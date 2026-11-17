#!/usr/bin/env python3
"""Library-agnostic emulator lab for the v3 (10M-row, 17-parameter) libraries.

Generalises ``emulator_lab.py`` (which is tied to the v6 library: 12 parameters,
z <= 14, 1M rows) so the same recipe can be trained and scored on any synference
library:

* the input parameters are all of the library's ``ParameterNames``;
* the output bands are those of a reference checkpoint (the 54 v2 bands);
* the Fourier redshift scale defaults to the library's maximum redshift;
* the optimisation budget is set in steps (``--steps``), so data-scaling runs
  at different ``--n-train`` share one budget;
* the trained net is written straight into a ``ParrotEmulatorV2`` checkpoint
  (``--save-checkpoint``), and that checkpoint is reloaded and re-scored to
  verify it.

Metrics are those of ``emulator_lab.py`` (absolute-flux asinh magnitudes, floor
1e-4 nJy) on a fixed held-out split, plus the same metrics restricted to the v2
domain (z <= 14) and per-band fractional flux errors in the format of
``validation_v2/perband_fractional_error.json``.

``--eval-checkpoint`` scores an existing ParrotEmulatorV2 checkpoint on a
library with the same split and metrics (used to put v2 on the same footing).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

SPLIT_SEED = 123
FLUX_FLOOR = 1e-4  # nJy (or nJy per 1e9 Msun in mass-normalised space)
ASINH_A = 1.0857362047581294  # 2.5 log10(e)
MU0 = ASINH_A * float(np.log(2.0 / FLUX_FLOOR))


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--library", required=True)
    p.add_argument("--bands-from", required=True, help="checkpoint whose band_names are used")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-test", type=int, default=100_000)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--n-train", type=int, default=0, help="subsample train set; 0 = all")
    p.add_argument("--arch", choices=["mlp", "resmlp"], default="resmlp")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=5, help="hidden layers (mlp)")
    p.add_argument("--blocks", type=int, default=4, help="residual blocks (resmlp)")
    p.add_argument("--no-mass-norm", action="store_true")
    p.add_argument("--no-sfr-arsinh", action="store_true")
    p.add_argument("--fourier-k", type=int, default=64)
    p.add_argument("--z-scale", type=float, default=0.0, help="0 = library max redshift")
    p.add_argument("--steps", type=int, default=1_100_000, help="total optimiser steps")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-frac", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument(
        "--bright-weight",
        type=float,
        default=0.0,
        help="extra loss weight for bands within --bright-window mag of each row's brightest band",
    )
    p.add_argument("--bright-window", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=0.0, help="stop training after this; 0 = off")
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--eval-checkpoint", default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_data(library, bands_from, n_test, n_val):
    """Load all parameters and the reference bands; return a fixed split."""
    import h5py

    with np.load(bands_from, allow_pickle=False) as ck:
        band_names = ck["band_names"].tolist()

    t0 = time.time()
    with h5py.File(library, "r") as f:
        param_names = [str(x) for x in f.attrs["ParameterNames"]]
        lib_b = [str(x) for x in f.attrs["FilterCodes"]]
        bi = [lib_b.index(n) for n in band_names]
        P = f["Grid/Parameters"]
        F = f["Grid/Photometry"]
        # Row-wise reads match the (1, chunk) layout of synference libraries.
        params = np.stack([P[i] for i in range(P.shape[0])], axis=1)
        flux = np.stack([F[i] for i in range(F.shape[0]) if i in set(bi)], axis=1)
        order = np.argsort(np.argsort(bi))  # rows were read in library order
        flux = flux[:, order]
    print(f"read {library} in {time.time() - t0:.0f}s: params {params.shape} flux {flux.shape}")

    ok = np.all(np.isfinite(params), axis=1) & np.all(np.isfinite(flux), axis=1)
    params, flux = params[ok], flux[ok]

    perm = np.random.default_rng(SPLIT_SEED).permutation(len(params))
    test = perm[:n_test]
    val = perm[n_test : n_test + n_val]
    train = perm[n_test + n_val :]
    return param_names, band_names, params, flux, train, val, test


def flux_to_mag(f, mu0=MU0):
    """Arsinh-magnitude transform of flux with softening set by ``mu0``."""
    b = np.exp(mu0 / ASINH_A) / 2.0
    return -ASINH_A * np.arcsinh(f * b) + mu0


def mag_to_flux(m, mu0=MU0):
    """Inverse of :func:`flux_to_mag`."""
    return 2.0 * np.exp(-mu0 / ASINH_A) * np.sinh((mu0 - m) / ASINH_A)


def predict_checkpoint(path, params, chunk=65536):
    """Predict nJy fluxes with a saved ParrotEmulatorV2 checkpoint."""
    import jax.numpy as jnp

    from arachne.emulator.parrot_emulator_v2 import load_emulator

    emu = load_emulator(path)
    out = [
        np.asarray(emu.predict(jnp.asarray(params[i : i + chunk], dtype=jnp.float32)))
        for i in range(0, len(params), chunk)
    ]
    return emu, np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):  # noqa: C901
    """Train and evaluate one emulator configuration from the command line."""
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    param_names, band_names, params_raw, flux_raw, tr, va, te = load_data(
        args.library, args.bands_from, args.n_test, args.n_val
    )
    print(f"data: {len(params_raw)} rows, train={len(tr)} val={len(va)} test={len(te)}")
    print(f"params: {param_names}")
    i_z = param_names.index("redshift")
    i_mass = param_names.index("log_mass")
    flux_test_true = np.where(flux_raw[te] < FLUX_FLOOR, 0.0, flux_raw[te])

    if args.eval_checkpoint:
        emu, pred_flux = predict_checkpoint(args.eval_checkpoint, params_raw[te])
        assert emu.param_names == param_names, (emu.param_names, param_names)
        assert emu.band_names == band_names
        report(args, out_dir, band_names, params_raw[te, i_z], flux_test_true, pred_flux, None, 0)
        return 0

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    from arachne.emulator.parrot_emulator_v2 import ParrotEmulatorV2, _build_net

    jax.config.update("jax_default_matmul_precision", "highest")

    mass_norm = not args.no_mass_norm
    sfr_arsinh = not args.no_sfr_arsinh
    z_scale = args.z_scale or float(math.ceil(params_raw[:, i_z].max()))
    tr_use = tr if args.n_train == 0 else tr[: args.n_train]

    # ---- targets (identical to emulator_lab.py) ---------------------------
    if mass_norm:
        m9 = 10.0 ** (params_raw[:, i_mass] - 9.0)
        unit_floor = FLUX_FLOOR / m9.max()
        mu0_train = ASINH_A * float(np.log(2.0 / unit_floor))
        f_unit = flux_raw / m9[:, None]
        y_all = flux_to_mag(np.where(f_unit < unit_floor, 0.0, f_unit), mu0_train)
        del f_unit
    else:
        mu0_train = MU0
        y_all = flux_to_mag(np.where(flux_raw < FLUX_FLOOR, 0.0, flux_raw), mu0_train)

    keep_idx = [i for i in range(len(param_names)) if not (mass_norm and i == i_mass)]
    sfr_cols = (
        [keep_idx.index(i) for i, n in enumerate(param_names) if n.startswith("logsfr_ratio")]
        if sfr_arsinh
        else []
    )
    cfg = dict(
        arch=args.arch,
        width=args.width,
        depth=args.depth,
        blocks=args.blocks,
        mass_norm=mass_norm,
        keep_mass_input=False,
        sfr_arsinh=sfr_arsinh,
        fourier_k=args.fourier_k,
        z_scale=z_scale,
        flux_floor=FLUX_FLOOR,
        mu0_train=float(mu0_train),
        i_mass=i_mass,
        i_z=i_z,
        keep_idx=keep_idx,
        sfr_cols=sfr_cols,
        # training domain and provenance, read back through ``emulator.spec``
        param_bounds={
            n: [float(params_raw[:, i].min()), float(params_raw[:, i].max())]
            for i, n in enumerate(param_names)
        },
        library=Path(args.library).name,
    )

    # Features through the production class, so training and inference share code.
    n_feat = len(keep_idx) + 2 * (int(math.log2(args.fourier_k)) + 1 if args.fourier_k else 0)
    proto = ParrotEmulatorV2(
        param_names,
        band_names,
        cfg,
        np.zeros(n_feat, np.float32),
        np.ones(n_feat, np.float32),
        np.zeros(len(band_names), np.float32),
        key=jax.random.PRNGKey(0),
    )
    feat = jax.jit(lambda p: proto._features(p))
    x_all = np.concatenate(
        [
            np.asarray(feat(jnp.asarray(params_raw[i : i + 1_000_000], dtype=jnp.float32)))
            for i in range(0, len(params_raw), 1_000_000)
        ]
    )
    assert x_all.shape[1] == n_feat

    in_mean = x_all[tr_use].mean(axis=0)
    in_std = x_all[tr_use].std(axis=0) + 1e-8
    out_mean = y_all[tr_use].mean(axis=0)

    xt = jnp.asarray((x_all[tr_use] - in_mean) / in_std, dtype=jnp.float32)
    yt = jnp.asarray(y_all[tr_use] - out_mean, dtype=jnp.float32)
    xv = jnp.asarray((x_all[va] - in_mean) / in_std, dtype=jnp.float32)
    yv = jnp.asarray(y_all[va] - out_mean, dtype=jnp.float32)
    del y_all

    n_in, n_out = xt.shape[1], yt.shape[1]
    key = jax.random.PRNGKey(args.seed)
    key, mk = jax.random.split(key)
    model = _build_net(cfg, n_in, n_out, mk)
    n_params = sum(
        int(np.prod(leaf.shape))
        for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )

    n_train = xt.shape[0]
    steps_per_epoch = n_train // args.batch
    # Epochs are capped at ~200k steps so the best-val checkpoint is tracked often enough.
    chunk = min(steps_per_epoch, 2000)
    n_chunks = math.ceil(args.steps / chunk)
    total_steps = n_chunks * chunk
    print(
        f"model: {args.arch} {args.width}x{args.blocks if args.arch == 'resmlp' else args.depth} "
        f"in={n_in} out={n_out} params={n_params:,}; z_scale={z_scale}; "
        f"n_train={n_train:,} steps={total_steps:,} ({total_steps / steps_per_epoch:.1f} epochs)",
        flush=True,
    )
    sched = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, args.warmup_steps, total_steps, args.lr * args.min_lr_frac
    )
    optim = (
        optax.adamw(sched, weight_decay=args.weight_decay)
        if args.weight_decay > 0
        else optax.adam(sched)
    )
    fparams, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(fparams)

    out_mean_j = jnp.asarray(out_mean, dtype=jnp.float32)

    def sq_err(pred, yb):
        """(Weighted) mean squared error in asinh magnitudes."""
        e2 = (pred - yb) ** 2
        if args.bright_weight <= 0:
            return jnp.mean(e2)
        y = yb + out_mean_j
        w = 1.0 + args.bright_weight * (y - y.min(axis=1, keepdims=True) < args.bright_window)
        return jnp.sum(w * e2) / jnp.sum(w)

    @jax.jit
    def train_chunk(fparams, opt_state, xt, yt, idx):
        xs = xt[idx].reshape(chunk, args.batch, n_in)
        ys = yt[idx].reshape(chunk, args.batch, n_out)

        def step(carry, batch):
            fp, os_ = carry
            xb, yb = batch

            def loss_fn(fp):
                pred = jax.vmap(eqx.combine(fp, static))(xb)
                return sq_err(pred, yb)

            loss, g = jax.value_and_grad(loss_fn)(fp)
            upd, os_ = optim.update(g, os_, fp)
            return (optax.apply_updates(fp, upd), os_), loss

        (fparams, opt_state), losses = jax.lax.scan(step, (fparams, opt_state), (xs, ys))
        return fparams, opt_state, jnp.mean(losses)

    @jax.jit
    def val_loss(fp, xv, yv):
        m = eqx.combine(fp, static)
        return jnp.mean(
            jnp.stack(
                [
                    sq_err(jax.vmap(m)(xv[i : i + 25000]), yv[i : i + 25000])
                    for i in range(0, xv.shape[0], 25000)
                ]
            )
        )

    # Sample without replacement through successive permutations of the train set.
    best_val, best_params = np.inf, fparams
    history = []
    t0 = time.time()
    perm, pos = None, steps_per_epoch * args.batch
    for c in range(1, n_chunks + 1):
        if pos + chunk * args.batch > steps_per_epoch * args.batch:
            key, sk = jax.random.split(key)
            perm, pos = jax.random.permutation(sk, n_train), 0
        idx = perm[pos : pos + chunk * args.batch]
        pos += chunk * args.batch
        fparams, opt_state, tl = train_chunk(fparams, opt_state, xt, yt, idx)
        vl = float(val_loss(fparams, xv, yv))
        if vl < best_val:
            best_val, best_params = vl, fparams
        history.append((c * chunk, float(tl), vl))
        if c % 10 == 0 or c == 1 or c == n_chunks:
            print(
                f"  step {c * chunk:8d} train={float(tl):.6f} val={vl:.6f} "
                f"best={best_val:.6f} lr={float(sched(c * chunk)):.2e} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        if args.max_hours and time.time() - t0 > 3600 * args.max_hours:
            print(f"  stopping at --max-hours {args.max_hours} after {c * chunk} steps", flush=True)
            break
    train_s = time.time() - t0
    print(f"training done in {train_s:.0f}s, best val MSE {best_val:.6f}", flush=True)

    # ---- production checkpoint + test predictions ---------------------------
    emu = ParrotEmulatorV2(
        param_names, band_names, cfg, in_mean, in_std, out_mean, key=jax.random.PRNGKey(0)
    )
    emu = eqx.tree_at(lambda m: m.net, emu, eqx.combine(best_params, static))
    ck_path = out_dir / f"{args.name}.eqx"
    emu.save(ck_path)
    _, pred_flux = predict_checkpoint(ck_path, params_raw[te])
    if not args.save_checkpoint:
        ck_path.unlink()
    np.save(out_dir / f"{args.name}_history.npy", np.array(history))
    table = report(
        args,
        out_dir,
        band_names,
        params_raw[te, i_z],
        flux_test_true,
        pred_flux,
        n_params,
        train_s,
        extra={"best_val_mse": best_val, "n_train": int(n_train), "steps": len(history) * chunk},
    )
    if args.save_checkpoint:
        # the held-out error table travels with the checkpoint (emulator.spec.band_error)
        emu.with_metadata(band_errors=table).save(ck_path)
    return 0


def _metrics(r, flux_true, pred_flux):
    """Scalar metrics of emulator_lab.py for one subset of test rows."""
    scatter = r.std(axis=0)
    trans = (flux_true > 0) & (flux_true < 1e-2)
    bright_err, det_scatter = [], []
    for b in range(r.shape[1]):
        m = flux_true[:, b] > 10.0
        if m.sum() > 100:
            bright_err.append(np.median(np.abs(pred_flux[m, b] / flux_true[m, b] - 1)))
        d = flux_true[:, b] > 1.0
        if d.sum() > 100:
            det_scatter.append(r[d, b].std())
    zero = flux_true == 0
    return {
        "n": int(len(r)),
        "med_scatter": float(np.median(scatter)),
        "med_abs_bias": float(np.median(np.abs(r.mean(axis=0)))),
        "med_p95": float(np.median(np.percentile(np.abs(r), 95, axis=0))),
        "rmse": float(np.sqrt(np.mean(r**2))),
        "trans_rms": float(np.sqrt(np.mean(r[trans] ** 2))),
        "bright_flux_err": float(np.median(bright_err)),
        "det_scatter": float(np.median(det_scatter)),
        "leak_p95_nJy": float(np.percentile(pred_flux[zero], 95)) if zero.any() else 0.0,
        "worst_band_scatter": float(scatter.max()),
    }


def report(args, out_dir, band_names, z_te, flux_true, pred_flux, n_params, train_s, extra=None):
    """Print and save accuracy metrics for a trained emulator."""
    r = flux_to_mag(pred_flux) - flux_to_mag(flux_true)
    res = {
        "name": args.name,
        "config": vars(args),
        "n_params": n_params,
        "train_seconds": round(train_s, 1),
        **(extra or {}),
        **_metrics(r, flux_true, pred_flux),
    }
    v2dom = z_te <= 14.0
    res["v2_domain"] = _metrics(r[v2dom], flux_true[v2dom], pred_flux[v2dom])
    res["per_band"] = {
        b: {"bias": float(r[:, i].mean()), "scatter": float(r[:, i].std())}
        for i, b in enumerate(band_names)
    }

    # Fractional flux errors per band in redshift bins, for detected rows (true flux
    # > 1 nJy): the table a checkpoint carries (arachne.emulator.validation).
    from arachne.emulator.validation import band_error_table

    table = band_error_table(
        pred_flux,
        flux_true,
        z_te,
        band_names,
        source=f"emulator_lab_v3.py {args.name} on {Path(args.library).name}, "
        f"{len(z_te)} held-out rows (split seed {SPLIT_SEED})",
    )
    frac = table["bands"]
    with open(out_dir / f"{args.name}_perband_fractional_error.json", "w") as fh:
        json.dump(table, fh, indent=1)
    nircam = [b for b in band_names if "NIRCam" in b]
    res["nircam_z2_mad_median"] = float(np.median([frac[b]["z1.8-2.5"]["mad"] for b in nircam]))
    res["nircam_z2_mad_max"] = float(np.max([frac[b]["z1.8-2.5"]["mad"] for b in nircam]))

    with open(out_dir / f"{args.name}.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n==== {args.name} ====")
    for k, v in res.items():
        if k not in ("config", "per_band"):
            print(f"  {k:22s}: {v}")
    print(f"saved -> {out_dir / (args.name + '.json')}")
    return table


if __name__ == "__main__":
    import sys

    sys.exit(main())
