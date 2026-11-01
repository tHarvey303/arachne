"""Correlated-field fit of a real galaxy from DJA thumbnails (0.05"/px, single grid).

The resolved-map counterpart of ``fit_jades_dja.py``: ``CorrelatedFieldMap`` maps of stellar
mass, A_V, metallicity and the SFH (``--sfh basis2`` by default) fitted blind to the DJA
``gds-grizli-v7.2`` thumbnails of one galaxy, with the JADES DR4 spec-z held fixed.  The
pipeline (start, Adam, trust-region Newton, Laplace or geoVI, whitened NUTS) is imported from
``demo_correlated_field.py``; this script replaces the mock with data.

Data.  ``fetch_dja_cutout`` on a ``--frame-arcsec`` (default 6") field in every requested band
(bands the server does not return are dropped), background per band from the median outside
the detection segments, the central ``--size-px`` box kept, pixels of other segments masked.
Variance is the DJA inverse-variance map, which carries no source Poisson term, so the
model-error floor is mandatory: per band ``f_b = sqrt(f_model^2 + f_emu,b^2)``, with
``--model-err-frac`` the physics/model term and ``f_emu,b`` the emulator's own fractional
error per band from ``--emulator-err-json`` (``perband_fractional_error.json`` from
``scripts/validate_parrot_emulator.py`` residuals; ~0.4 % at z ~ 2).

PSFs.  ``--psf-dir`` must hold ``{FILTER}_psf_norm.fits`` kernels **on the thumbnail grid**
(0.05"/px).  The DJA SW mosaics are broader than the JADES DR3 mosaics (EE(0.2") 0.67 vs
0.82 in F115W), so kernels built from stars in the same thumbnails
(``real_targets/33151/build_dja_psf.py``) are used, not resampled JADES PSFs.

Limitations.  No sky / shift nuisances (the background is subtracted, not fitted), so the demo's
map helpers apply unchanged; one pixel grid (no multi-resolution).

Example:
    python examples/fit_correlated_field_dja.py --target goods-s-mediumjwst_33151 \\
        --emulator emu.eqx --psf-dir psfs/DJA-gds-v7.2-0.05 --float64 --posterior laplace --nuts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_correlated_field as demo  # noqa: E402
import real_data_utils as rdu  # noqa: E402

from arachne import ForwardModel, ObservationCube, PSFModel, load_emulator  # noqa: E402
from arachne.data.dja import fetch_dja_cutout, load_dja_cutout  # noqa: E402
from arachne.psf.resolution import psf_k_max  # noqa: E402
from arachne.results.io import write_content  # noqa: E402
from arachne.results.regions import detection_region  # noqa: E402
from arachne.results.writer import content_from_forward_model  # noqa: E402
from arachne.spatial.correlated_field import mass_map_from_image  # noqa: E402
from arachne.spatial.mass_start import default_sfh_dust_templates, template_mass_map  # noqa: E402

ALL_BANDS = sorted(demo.NIRCAM_PIVOT_UM, key=demo.NIRCAM_PIVOT_UM.get)


def configure_demo(filters, pixel_scale, redshift):
    """Point the demo's module globals at this data set (bands, grid, redshift)."""
    filters = sorted(filters, key=demo.NIRCAM_PIVOT_UM.get)
    demo.BANDS = [f"JWST/NIRCam.{f}" for f in filters]
    demo.PIVOT_UM = np.array([demo.NIRCAM_PIVOT_UM[f] for f in filters])
    demo.FWHM_ARCSEC = np.maximum(0.030, 0.032 * demo.PIVOT_UM)
    demo.PIXEL_SCALE = float(pixel_scale)
    demo.REDSHIFT = float(redshift)
    demo.FIXED["redshift"] = float(redshift)
    demo.FIXED_TRUTH["redshift"] = float(redshift)
    return filters


def load_data(ra, dec, filters, frame_arcsec, size_px, cache_dir, min_exptime=0.0):
    """DJA thumbnails -> background-subtracted, neighbour-masked ``ObservationCube``.

    Returns the cube, the per-band background (nJy), the detection segment of the wide frame
    trimmed to the cutout, the wide frame cube (for context plots) and the band info.
    """
    path = fetch_dja_cutout(
        ra,
        dec,
        frame_arcsec,
        [f.lower() + "-clear" for f in filters],
        output="fits_weight",
        cache_dir=cache_dir,
    )
    mro = load_dja_cutout(path, ref_ra=ra, ref_dec=dec)
    bands = [b for b in mro.bands if b.band_name.split(".")[-1] in filters]
    info = {}
    keep = []
    for b in bands:
        f = b.band_name.split(".")[-1]
        hdr_exp = float(getattr(b, "exptime", 0.0) or 0.0)
        valid = (
            (np.asarray(b.mask) > 0)
            & np.isfinite(np.asarray(b.flux))
            & (np.asarray(b.variance) > 0)
        )
        info[f] = {
            "pixel_scale": float(b.pixel_scale),
            "valid_fraction": float(valid.mean()),
            "median_sigma_nJy": float(np.sqrt(np.median(np.asarray(b.variance)[valid])))
            if valid.any()
            else None,
        }
        if valid.mean() > 0.95 and hdr_exp >= min_exptime:
            keep.append(b)
        else:
            info[f]["dropped"] = True
    if not keep:
        raise RuntimeError("no usable band")
    scales = {float(b.pixel_scale) for b in keep}
    if len(scales) != 1:
        raise RuntimeError(f"thumbnails are not on one grid: {scales}")
    pix = scales.pop()
    flux = np.stack([np.asarray(b.flux, np.float64) for b in keep])
    var = np.stack([np.asarray(b.variance, np.float64) for b in keep])
    mask = np.stack(
        [
            (np.asarray(b.mask) > 0)
            & np.isfinite(np.asarray(b.flux))
            & (np.asarray(b.variance) > 0)
            for b in keep
        ]
    )
    names = [b.band_name for b in keep]
    flux = np.where(mask, flux, 0.0)
    var = np.where(mask, var, 1.0)
    # detection segments on the inverse-variance-weighted stack of the wide frame
    seg_all = detection_region(flux, var, mask.astype(np.float32), nsigma=3.0, smooth_px=1.0)
    from scipy import ndimage

    sm = ndimage.gaussian_filter(
        (flux / var * mask).sum(0) / np.sqrt((mask / var).sum(0) + 1e-30), 1.0
    )
    thr = 3.0 * 1.4826 * np.median(np.abs(sm - np.median(sm)))
    lab, _ = ndimage.label(sm > thr)
    H, W = sm.shape
    centre_label = lab[H // 2, W // 2]
    if centre_label == 0:
        yy, xx = np.mgrid[:H, :W]
        d = np.where(lab > 0, np.hypot(yy - H / 2, xx - W / 2), np.inf)
        centre_label = lab[np.unravel_index(np.argmin(d), d.shape)]
    target = lab == centre_label
    others = (lab > 0) & ~target
    sky_px = (lab == 0) & mask.all(0)
    bkg = np.array([np.median(flux[i][sky_px]) for i in range(len(keep))])
    flux = flux - bkg[:, None, None]
    # trim to the central box
    lo = (H - size_px) // 2
    sl = slice(lo, lo + size_px)
    mask_cut = mask[:, sl, sl] & ~others[None, sl, sl]
    obs = ObservationCube(
        flux=flux[:, sl, sl].astype(np.float32),
        variance=var[:, sl, sl].astype(np.float32),
        mask=mask_cut.astype(np.float32),
        band_names=names,
        pixel_scale=pix,
    )
    wide = {"flux": flux, "mask": mask, "target": target, "others": others, "sky": sky_px}
    for i, b in enumerate(keep):
        f = b.band_name.split(".")[-1]
        info[f]["background_nJy"] = float(bkg[i])
        info[f]["peak_snr"] = float(np.max(flux[i, sl, sl] / np.sqrt(var[i, sl, sl])))
        info[f]["target_flux_nJy"] = float(flux[i][target].sum())
    return obs, bkg, target[sl, sl], wide, info, seg_all


def plot_data_model(obs, model_img, var_eff, path, show):
    """data / model / chi for a few bands."""
    names = [b.split(".")[-1] for b in obs.band_names]
    rows = [names.index(f) for f in show if f in names]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.4 * len(rows)), squeeze=False)
    for r, b in enumerate(rows):
        d = np.asarray(obs.flux[b])
        m = model_img[b]
        chi = (d - m) / np.sqrt(var_eff[b]) * np.asarray(obs.mask[b])
        hi = np.nanpercentile(d, 99.5)
        for ax, img, title in zip(axes[r], (d, m, chi), ("data", "model", "chi")):
            if title == "chi":
                im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4)
            else:
                im = ax.imshow(
                    np.arcsinh(10 * np.clip(img, 0, None) / hi), origin="lower", cmap="gray_r"
                )
            ax.set_title(f"{names[b]} {title}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_param_maps(names, map_maps, med, std, mask, path):
    """Rows: parameters; columns: MAP, posterior median, posterior std (segment only)."""
    n = len(names)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.6 * n), squeeze=False)
    for i, name in enumerate(names):
        cols = [
            (map_maps[i], "MAP"),
            (med[i] if med is not None else None, "posterior median"),
            (std[i] if std is not None else None, "posterior std"),
        ]
        for ax, (img, title) in zip(axes[i], cols):
            if img is None:
                ax.axis("off")
                continue
            shown = np.where(mask, img, np.nan)
            lo, hi = np.nanpercentile(shown, [2, 98])
            im = ax.imshow(
                shown,
                origin="lower",
                vmin=lo,
                vmax=hi,
                cmap="viridis" if title != "posterior std" else "magma",
            )
            ax.set_title(f"{demo.LABELS.get(name, name)} {title}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    """Run the fit."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--target", default="goods-s-mediumjwst_33151", help="JADES DR4 id (or a vetted id)"
    )
    ap.add_argument("--ra", type=float, default=None)
    ap.add_argument("--dec", type=float, default=None)
    ap.add_argument("--z", type=float, default=None)
    ap.add_argument("--emulator", required=True)
    ap.add_argument("--psf-dir", required=True)
    ap.add_argument(
        "--bands",
        default="all",
        help="comma-separated filters, or 'all' (every band the server returns)",
    )
    ap.add_argument("--size-px", type=int, default=64)
    ap.add_argument("--frame-arcsec", type=float, default=6.0)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--model-err-frac",
        type=float,
        default=0.05,
        help="model-inadequacy floor (fraction of the model)",
    )
    ap.add_argument(
        "--emulator-err-json",
        default=None,
        help="per-band emulator error JSON (validate_parrot_emulator residuals)",
    )
    ap.add_argument("--emulator-err-key", default="z1.8-2.5", help="redshift bin in the JSON")
    ap.add_argument("--sfh", choices=demo.SFH_SETUPS, default="basis2")
    ap.add_argument("--sfh-prior", choices=sorted(demo.SFH_PRIOR_SCALES), default="continuity")
    ap.add_argument("--free-globals", action="store_true")
    ap.add_argument("--nu", type=float, default=1.0)
    ap.add_argument(
        "--corr-scale",
        type=float,
        default=0.6,
        help="x the demo's correlation lengths (set in 0.03\" px; 0.6 keeps the angular "
        'lengths at 0.05"/px)',
    )
    ap.add_argument("--hyper", action="store_true")
    ap.add_argument("--couple", action="store_true")
    ap.add_argument("--k-max-mtf", type=float, default=0.0)
    ap.add_argument("--mass-start", choices=["image", "template"], default="image")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--block-newton", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--n-starts", type=int, default=1)
    ap.add_argument("--posterior", choices=["map", "laplace", "geovi", "mgvi"], default="map")
    ap.add_argument("--n-laplace", type=int, default=400)
    ap.add_argument("--hess-chunk", type=int, default=32)
    ap.add_argument("--max-newton", type=int, default=300)
    ap.add_argument("--precond", choices=["dense", "diag", "fourier", "none"], default="dense")
    ap.add_argument("--vi-iterations", type=int, default=6)
    ap.add_argument("--vi-samples", type=int, default=4)
    ap.add_argument("--nuts", action="store_true")
    ap.add_argument("--nuts-warmup", type=int, default=1000)
    ap.add_argument("--nuts-samples", type=int, default=500)
    ap.add_argument("--nuts-chains", type=int, default=4)
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="outputs/real_data_cf")
    ap.add_argument("--tag", default="")
    ap.add_argument("--bundle-draws", type=int, default=200)
    ap.add_argument("--bundle-plots", action="store_true")
    ap.add_argument("--no-bundle", action="store_true")
    args = ap.parse_args()
    if args.float64:
        jax.config.update("jax_enable_x64", True)
    if args.quick:
        args.rounds, args.steps = 2, 300
    vi = args.posterior in ("geovi", "mgvi")
    laplace = args.posterior == "laplace"
    if vi and args.model_err_frac > 0:
        print("NOTE: geoVI/MGVI need model_error_frac = 0 (run_mgvi refuses a floor); forcing 0")
        args.model_err_frac = 0.0

    target = rdu.resolve_target(args.target, args.ra, args.dec, args.z)
    ra, dec, z = target.ra, target.dec, target.z
    tid = target.id if hasattr(target, "id") else args.target
    print(f"target {tid}: ra {ra:.6f} dec {dec:.6f} z {z:.4f}")
    outdir = Path(args.outdir) / f"{tid}{'_' + args.tag if args.tag else ''}"
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    want = ALL_BANDS if args.bands == "all" else [f.upper() for f in args.bands.split(",")]
    psf_dir = Path(args.psf_dir)
    want = [f for f in want if (psf_dir / f"{f}_psf_norm.fits").exists()]
    obs, bkg, segment, wide, band_info, _ = load_data(
        ra, dec, want, args.frame_arcsec, args.size_px, args.cache_dir
    )
    filters = configure_demo([b.split(".")[-1] for b in obs.band_names], obs.pixel_scale, z)
    # ObservationCube band order must match demo.BANDS (sorted by pivot)
    order = [list(obs.band_names).index(b) for b in demo.BANDS]
    obs = ObservationCube(
        flux=np.asarray(obs.flux)[order],
        variance=np.asarray(obs.variance)[order],
        mask=np.asarray(obs.mask)[order],
        band_names=list(demo.BANDS),
        pixel_scale=obs.pixel_scale,
    )
    n = args.size_px
    print(f'{len(filters)} bands on a {n}x{n} grid at {obs.pixel_scale:.3f}"/px: {filters}')
    for f in filters:
        bi = band_info[f]
        print(
            f"  {f}: bkg {bi['background_nJy']:+.3f} nJy/px, "
            f"sigma {bi['median_sigma_nJy']:.3f} nJy/px, "
            f"peak S/N {bi['peak_snr']:.0f}, flux {bi['target_flux_nJy'] / 1e3:.2f} uJy"
        )

    # model-error floor per band
    emu_err = np.zeros(len(filters))
    if args.emulator_err_json:
        tab = json.load(open(args.emulator_err_json))["bands"]
        emu_err = np.array([tab[f"JWST/NIRCam.{f}"][args.emulator_err_key]["mad"] for f in filters])
    frac = np.sqrt(args.model_err_frac**2 + emu_err**2)
    print("model_error_frac per band:", np.array2string(frac, precision=4))

    demo.set_sfh(args.sfh, args.free_globals)
    demo.SFH_PRIOR = args.sfh_prior
    emulator = demo.BandSubsetEmulator(load_emulator(args.emulator), demo.BANDS)
    assert list(emulator.param_names) == demo.SPS_PARAM_NAMES
    psf = PSFModel.from_fits({b: psf_dir / f"{b.split('.')[-1]}_psf_norm.fits" for b in demo.BANDS})
    basis_info = None
    if any(k in demo.LATENT_SFH for k in demo.FIELDS):
        demo.SFH_BASIS, s_vals = demo.sfh_basis(emulator, n=len(demo.LATENT_SFH))
        basis_info = {"B": demo.SFH_BASIS.tolist(), "singular_values": s_vals.tolist()}
        print(f"SFH basis singular values {np.array2string(s_vals, precision=4)}")

    ref = {**demo.NEUTRAL_GLOBALS, **demo.NEUTRAL, **demo.FIXED}
    if args.mass_start == "template":
        ms = template_mass_map(
            obs, emulator, ref, [{}, *default_sfh_dust_templates()], smooth_sigma_px=1.0
        )
        mass_mean = ms.log_mass
    else:
        mass_mean = mass_map_from_image(
            obs, emulator, ref, band=len(demo.BANDS) - 1, smooth_sigma_px=1.0
        )
    k_max = psf_k_max(psf, args.k_max_mtf) if args.k_max_mtf > 0 else None
    model = demo.build_model(
        n,
        mass_mean,
        corr_scale=args.corr_scale,
        nu=args.nu,
        hyper=args.hyper,
        couple=args.couple,
        k_max=k_max,
    )
    print(model.summary())
    fm = ForwardModel.build(obs, psf, model, emulator, model_error_frac=frac.astype(np.float32))
    theta0 = model.initial_theta(global_values=demo.NEUTRAL_GLOBALS)
    pstd_model = demo.prior_map_std(model, 400, jax.random.PRNGKey(args.seed + 200))
    names = list(model.param_map_names)

    starts = [theta0] + [
        demo.perturbed_start(model, theta0, jax.random.PRNGKey(args.seed + 1000 + i))
        for i in range(1, args.n_starts)
    ]
    runs = []
    for i, ts in enumerate(starts):
        if len(starts) > 1:
            print(f"===== start {i + 1}/{len(starts)}", flush=True)
        t0 = time.time()
        r, block_log = demo.map_stage(fm, ts, args.rounds, args.steps, block=args.block_newton)
        rt = time.time() - t0
        p = None
        if laplace:
            p = demo.laplace_posterior(
                fm,
                model,
                r.theta,
                args.n_laplace,
                args.hess_chunk,
                jax.random.PRNGKey(args.seed + 100),
                max_newton=args.max_newton,
                precond=args.precond,
            )
            summ = {
                "neg_log_post": float(p["newton_history"][-1]["neg_log_post"]),
                "newton_converged": p["newton_converged"],
                "newton_reason": p["newton_reason"],
                "log_evidence_laplace": p["log_evidence_laplace"],
                "n_negative": p["n_negative"],
            }
            print(
                f"  start {i + 1}: -log_post {summ['neg_log_post']:.2f}, "
                f"log Z {summ['log_evidence_laplace']:.2f}, "
                f"converged {summ['newton_converged']}, {summ['n_negative']} neg. eig.",
                flush=True,
            )
        elif vi:
            p = demo.vi_posterior(
                fm,
                model,
                r.theta,
                jax.random.PRNGKey(args.seed + 100),
                args.posterior,
                args.vi_iterations,
                args.vi_samples,
                args.n_laplace,
            )
            summ = {"neg_log_post": float(p["neg_log_post_mean"])}
        else:
            summ = {"neg_log_post": float(r.neg_log_posterior)}
        summ["map_neg_log_post"] = float(r.neg_log_posterior)
        if p is not None:
            mt = np.log10(np.sum(10.0 ** p["maps"][:, 0] * segment[None], axis=(1, 2)))
            summ["total_log_mass_segment_16_50_84"] = [
                float(x) for x in np.percentile(mt, [16, 50, 84])
            ]
            summ["global_medians"] = {
                k: v["p16_50_84"][1] for k, v in demo.scalar_summary(model, p["thetas"]).items()
            }
        runs.append({"res": r, "runtime": rt, "post": p, "summary": summ, "block": block_log})
    i_best = demo.select_best(runs)
    res, post = runs[i_best]["res"], runs[i_best]["post"]
    theta_map = post["theta"] if post is not None else res.theta
    fit_maps = np.asarray(model.param_maps(theta_map))
    model_img = np.asarray(fm._model_image(theta_map))
    var_eff = np.asarray(obs.variance) + (frac[:, None, None] * model_img) ** 2
    chi = (np.asarray(obs.flux) - model_img) / np.sqrt(var_eff)
    m = np.asarray(obs.mask) > 0
    seg3 = m & segment[None]

    def chi2(sel):
        return [float(np.mean(chi[b][sel[b]] ** 2)) for b in range(len(filters))]

    metrics = {
        "target": tid,
        "ra": ra,
        "dec": dec,
        "z": z,
        "bands": filters,
        "band_info": band_info,
        "pixel_scale": float(obs.pixel_scale),
        "size_px": n,
        "frame_arcsec": args.frame_arcsec,
        "model_err_frac": args.model_err_frac,
        "emulator_err_frac": emu_err.tolist(),
        "model_error_frac_per_band": frac.tolist(),
        "sfh": args.sfh,
        "sfh_prior": args.sfh_prior,
        "free_globals": args.free_globals,
        "nu": args.nu,
        "corr_scale": args.corr_scale,
        "hyper": args.hyper,
        "couple": args.couple,
        "k_max": k_max,
        "mass_start": args.mass_start,
        "posterior": args.posterior,
        "n_starts": args.n_starts,
        "best_start": i_best,
        "seed": args.seed,
        "psf_dir": str(psf_dir),
        "emulator": args.emulator,
        "fields": demo.FIELDS,
        "param_maps": names,
        "fixed_fit": {k: float(v) for k, v in demo.FIXED.items()},
        "global_names": list(model.global_names),
        "n_params": model.n_params,
        "modes_per_field": {f.name: f.n_modes for f in model.fields},
        "segment_pixels": int(segment.sum()),
        "masked_pixel_fraction": float(1 - m.mean()),
        "neg_log_post_start": float(-fm.log_posterior(theta0)),
        "neg_log_post_map": float(res.neg_log_posterior),
        "neg_log_post_mode": float(-fm.log_posterior(theta_map)),
        "chi2_red_all": float(np.mean(chi[m] ** 2)),
        "chi2_red_per_band": chi2(m),
        "chi2_red_segment_per_band": chi2(seg3),
        "chi2_red_photon_only_segment": float(
            np.mean(
                ((np.asarray(obs.flux) - model_img) / np.sqrt(np.asarray(obs.variance)))[seg3] ** 2
            )
        ),
        "map_runtime_s": runs[i_best]["runtime"],
        "adam_steps": res.n_steps,
        "map_scalars": demo.scalar_summary(model, theta_map[None, :]),
        "total_log_mass_map_segment": float(np.log10(np.sum(10.0 ** fit_maps[0][segment]))),
        "total_log_mass_map_all": float(np.log10(np.sum(10.0 ** fit_maps[0]))),
        "modes": [x["summary"] for x in runs],
        "block_newton_log": runs[i_best]["block"],
    }
    if basis_info:
        metrics["sfh_basis"] = basis_info
    med = std = None
    nuts = None
    if post is not None:
        stage = "laplace" if laplace else "vi"
        drop = ("theta", "maps", "chol", "thetas", "result", "scale")
        pm = metrics[stage] = {k: v for k, v in post.items() if k not in drop}
        pm["n_draws"] = int(post["thetas"].shape[0])
        pm["scalars"] = demo.scalar_summary(model, post["thetas"])
        med, std = np.median(post["maps"], axis=0), np.std(post["maps"], axis=0)
        pm["median_post_std_segment"] = {
            k: float(np.median(std[i][segment])) for i, k in enumerate(names)
        }
        mt = np.log10(np.sum(10.0 ** post["maps"][:, 0] * segment[None], axis=(1, 2)))
        pm["total_log_mass_segment_16_50_84"] = [float(x) for x in np.percentile(mt, [16, 50, 84])]
        if args.nuts:
            key = jax.random.PRNGKey(args.seed + 300)
            if vi:
                from arachne.inference.standardise import StandardisedLogDensity

                density = StandardisedLogDensity(fm, post["result"].transform)
                z_starts = demo.vi_chain_starts(
                    density,
                    [post["theta"]],
                    post["scale"],
                    jax.random.PRNGKey(args.seed + 301),
                    args.nuts_chains,
                )
                theta_starts = np.asarray(density.to_theta(jnp.asarray(z_starts)))
                nuts = demo.nuts_posterior(
                    fm,
                    model,
                    post["theta"],
                    None,
                    key,
                    args.nuts_warmup,
                    args.nuts_samples,
                    args.nuts_chains,
                    init_thetas=theta_starts,
                    whitened=density,
                )
            else:
                nuts = demo.nuts_posterior(
                    fm,
                    model,
                    post["theta"],
                    post["chol"],
                    key,
                    args.nuts_warmup,
                    args.nuts_samples,
                    args.nuts_chains,
                )
            mc = metrics["nuts"] = {
                k: v for k, v in nuts.items() if k not in ("maps", "theta_samples")
            }
            mc["scalars"] = demo.scalar_summary(model, nuts["theta_samples"])
            med, std = np.median(nuts["maps"], axis=0), np.std(nuts["maps"], axis=0)
            mc["median_post_std_segment"] = {
                k: float(np.median(std[i][segment])) for i, k in enumerate(names)
            }
            mt = np.log10(np.sum(10.0 ** nuts["maps"][:, 0] * segment[None], axis=(1, 2)))
            mc["total_log_mass_segment_16_50_84"] = [
                float(x) for x in np.percentile(mt, [16, 50, 84])
            ]
            metrics["posterior_used"] = "nuts"
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    np.savez(
        outdir / "maps.npz",
        map=fit_maps,
        median=med if med is not None else np.zeros(0),
        std=std if std is not None else np.zeros(0),
        names=np.array(names),
        segment=segment,
        model_image=model_img,
        data=np.asarray(obs.flux),
        var_eff=var_eff,
        mask=np.asarray(obs.mask),
        theta_map=np.asarray(theta_map),
        background_nJy=bkg,
    )
    show = [f for f in ("F115W", "F150W", "F200W", "F277W", "F444W") if f in filters][:4]
    plot_data_model(obs, model_img, var_eff, outdir / "figures" / "images.png", show)
    plot_param_maps(names, fit_maps, med, std, segment, outdir / "figures" / "maps.png")

    if not args.no_bundle:
        try:
            source = "nuts" if nuts is not None else (args.posterior if post is not None else "map")
            samples = (
                nuts["theta_samples"]
                if nuts is not None
                else (post["thetas"] if post is not None else None)
            )
            diag = {}
            if nuts is not None:
                diag = {
                    k: metrics["nuts"][k]
                    for k in ("max_rhat", "frac_rhat_gt_1.01", "min_ess", "median_ess")
                }
            if laplace and post is not None:
                diag["log_evidence_laplace"] = post["log_evidence_laplace"]
            content = content_from_forward_model(
                fm,
                theta_map=theta_map,
                samples=samples,
                posterior_source=source,
                n_draws=args.bundle_draws,
                n_chains=nuts["n_chains"] if nuts is not None else None,
                truth=None,
                prior_std_maps=pstd_model,
                regions={"segment": segment},
                metrics=metrics,
                diagnostics=diag,
                pivot_um=demo.PIVOT_UM,
                labels=demo.LABELS,
                units={
                    "log_mass": "dex",
                    "Av": "mag",
                    "log10metallicity": "dex",
                    "total_log_mass": "dex",
                },
                title=f"{tid} correlated-field {source}",
            )
            path = write_content(outdir / "results.h5", content, overwrite=True)
            print(f"bundle {path} ({path.stat().st_size / 2**20:.1f} MiB)")
            if args.bundle_plots:
                demo.bundle_plots(path, outdir / "figures" / "bundle")
        except Exception:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print("WARNING: writing the results bundle failed (see traceback above)")

    print(
        f"MAP in {runs[i_best]['runtime']:.0f} s ({res.n_steps} steps); "
        f"-log_post {metrics['neg_log_post_mode']:.2f}; "
        f"chi2_red {metrics['chi2_red_all']:.3f} (segment, per band: "
        f"{np.array2string(np.array(metrics['chi2_red_segment_per_band']), precision=2)}); "
        f"photon-only chi2_red in the segment {metrics['chi2_red_photon_only_segment']:.1f}"
    )
    print(f"  total log M* in the segment (MAP) {metrics['total_log_mass_map_segment']:.3f}")
    for k, v in metrics["map_scalars"].items():
        print(f"  {k:22s} {v['p16_50_84'][1]:+.3f}")
    for stage in ("laplace", "vi", "nuts"):
        if stage in metrics:
            q = metrics[stage]
            m16, m50, m84 = q["total_log_mass_segment_16_50_84"]
            print(
                f"{stage}: total log M* (segment) {m50:.3f} +{m84 - m50:.3f} -{m50 - m16:.3f}; "
                "median post std "
                + ", ".join(f"{k} {v:.3f}" for k, v in q["median_post_std_segment"].items())
            )
            if stage == "nuts":
                print(f"  {q['summary']}")
    print(f"outputs in {outdir}")


if __name__ == "__main__":
    main()
