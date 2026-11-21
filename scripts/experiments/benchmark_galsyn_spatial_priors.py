"""Benchmark spatial priors on matched GalSyn imaging, independently rendered from Arachne."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Cosmology
from astropy.io import fits
from benchmark_spatial_calibration import demo
from calibrate_galsyn_spatial_priors import download_drive, read_truth
from scipy.signal import fftconvolve

from arachne import ForwardModel, ObservationCube, PSFModel, load_emulator, mass_map_from_image
from arachne.data.units import flux_to_nJy
from arachne.inference.initialisation import find_map
from arachne.inference.newton import trust_region_newton
from arachne.postprocess.recipe import continuity_agebins
from arachne.priors.spatial_calibration import load_spatial_calibration
from arachne.psf.resolution import psf_k_max

FIT_BANDS = ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F410M", "F444W"]
TEST_BANDS = ["F182M", "F335M"]


def formed_ssfr100(ratios, agebins):
    """Log sSFR averaged over 100 Myr, per unit total formed mass."""
    sfr = 10.0 ** -np.cumsum(np.concatenate((np.zeros_like(ratios[:1]), ratios)), axis=0)
    widths = np.diff(agebins, axis=1).ravel()
    recent = (np.minimum(agebins[:, 1], 100) - agebins[:, 0]).clip(min=0)
    return np.log10(np.tensordot(recent, sfr, axes=1) / np.tensordot(widths, sfr, axes=1)) - 8


def orient_sfh_basis(basis, agebins):
    """Rotate the identifiable subspace so its first axis tracks recent sSFR."""
    steps = np.eye(basis.shape[1]) * 1e-4
    gradient = (formed_ssfr100(steps, agebins) - formed_ssfr100(-steps, agebins)) / 2e-4
    first = basis @ gradient
    sensitivity = np.linalg.norm(first)
    if sensitivity == 0:
        return basis, sensitivity
    first /= sensitivity
    rotation = np.array([first, [-first[1], first[0]]])
    return rotation @ basis, sensitivity


def mock_observation(path, galaxy, psf_dir, n, snr, rng):
    """Convolve released GalSyn images and add noise; aggregate target maps in flux or mass."""
    factor = max(1, int(np.ceil(3.4 * galaxy["re_px"] / n)))
    width = n * factor
    x, y = galaxy["center"]
    left, top = int(round(x - (width - 1) / 2)), int(round(y - (width - 1) / 2))
    if (
        min(top, left) < 0
        or top + width > galaxy["mass"].shape[0]
        or left + width > (galaxy["mass"].shape[1])
    ):
        raise ValueError(f"Benchmark stamp exceeds {path.name}")
    source_scale = galaxy["pixel_arcsec"]
    scale = factor * source_scale

    def block_sum(values):
        cutout = values[top : top + width, left : left + width]
        return cutout.reshape(n, factor, n, factor).sum(axis=(1, 3))

    def regrid_mean(values, weights):
        good = np.isfinite(values)
        return block_sum(np.where(good, values, 0) * weights) / np.maximum(
            block_sum(np.where(good, weights, 0)), 1e-30
        )

    mass = block_sum(galaxy["mass"])
    truth = {
        "Av": regrid_mean(galaxy["maps"]["Av"], galaxy["mass"]),
        "log10metallicity": regrid_mean(galaxy["maps"]["log10metallicity"], galaxy["mass"]),
        "logssfr_100myr": np.log10(
            np.maximum(block_sum(galaxy["sfr100"]) / np.maximum(mass, 1), 1e-11)
        ),
    }
    score = (mass >= 1.6e6) & (block_sum(galaxy["radius"]) / factor**2 <= 1.25 * galaxy["re_px"])
    if score.sum() < 10:
        raise ValueError(f"Insufficient resolved score area in {path.name}")

    observations, kernels = [], []
    with fits.open(path, memmap=False) as hdus:
        for band in FIT_BANDS + TEST_BANDS:
            hdu = hdus[f"DUST_JWST_NIRCAM_{band}"]
            source = flux_to_nJy(
                np.asarray(hdu.data),
                hdu.header.get("BUNIT", "MJy/sr"),
                pixel_area_arcsec2=source_scale**2,
            )
            kernel = np.asarray(fits.getdata(psf_dir / f"{band}_psf_norm.fits"), float)
            size = min(2 * (n // 2) - 1, 25)
            # Apply the same effective PSF on the fitting grid in the truth and the fit.
            effective = PSFModel.resample(kernel, from_scale=0.03, to_scale=scale, out_size=size)
            profile = fftconvolve(block_sum(source), effective, mode="same")
            observations.append(profile)
            kernels.append(effective)
    clean = np.stack(observations)
    sigma = np.percentile(clean[:, score], 95, axis=1) / snr
    noisy = clean + sigma[:, None, None] * rng.normal(size=clean.shape)
    nfit = len(FIT_BANDS)
    bands = [f"JWST/NIRCam.{band}" for band in FIT_BANDS]
    obs = ObservationCube(
        noisy[:nfit],
        np.broadcast_to(sigma[:nfit, None, None] ** 2, noisy[:nfit].shape),
        np.ones_like(noisy[:nfit], dtype=bool),
        bands,
        scale,
    )
    return (
        obs,
        PSFModel(np.stack(kernels[:nfit]), bands),
        kernels[nfit:],
        noisy[nfit:],
        sigma[nfit:],
        truth,
        score,
    )


def benchmark(row, path, fit_emulator, test_emulator, calibration, args):
    """Compare fixed priors at identical optimizer budgets and independent flux models."""
    galaxy = read_truth(path)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, *map(ord, row["id"])]))
    obs, psf, test_psfs, withheld, sigma, truth, score = mock_observation(
        path, galaxy, args.psf_dir, args.size, args.snr, rng
    )
    recipe = fit_emulator.spec.simulator
    cosmo = Cosmology.from_format(recipe["cosmology"], format="yaml")
    z = galaxy["z"]
    agebins = continuity_agebins(z, cosmo, recipe["sfh"]["agebins"])
    demo.REDSHIFT = z
    demo.PIXEL_SCALE = obs.pixel_scale
    demo.PARAM_BOUNDS.update(fit_emulator.spec.clip_bounds(demo.PARAM_BOUNDS))
    demo.set_sfh("basis2")
    basis, _ = demo.sfh_basis(fit_emulator)
    oriented, sensitivity = orient_sfh_basis(basis, agebins)
    fixed = {
        "redshift": z,
        "slope": 0.0,
        "fesc_lya": demo.NEUTRAL_GLOBALS["fesc_lya"],
        "dust_bump_amplitude": demo.NEUTRAL_GLOBALS["dust_bump_amplitude"],
    }
    neutral = {**demo.NEUTRAL_GLOBALS, **demo.NEUTRAL, **fixed}
    mass_mean = mass_map_from_image(obs, fit_emulator, neutral)
    # Estimate the radius from blurred fitted photometry, independent of the truth maps.
    intensity = np.where(obs.flux[-1] >= 2 * np.sqrt(obs.variance[-1]), obs.flux[-1], 0)
    yy, xx = np.indices(intensity.shape)
    x, y = (intensity * xx).sum() / intensity.sum(), (intensity * yy).sum() / intensity.sum()
    radii = np.hypot(xx - x, yy - y).ravel()
    order = np.argsort(radii)
    curve = np.cumsum(intensity.ravel()[order])
    proxy_re = max(0.5, float(np.interp(curve[-1] / 2, curve, radii[order])))
    priors = load_spatial_calibration(
        args.calibration, redshift=z, re_pixels=proxy_re, allow_extrapolation=True
    )
    k_max = psf_k_max(psf, args.mtf) if args.mtf > 0 else None
    ssfr_prior = calibration["fields"]["logssfr_100myr"]
    original_lengths, original_amplitudes = demo.CORR_LENGTHS.copy(), demo.AMPLITUDES.copy()
    outcomes = {}
    for scenario in ("baseline", "calibrated"):
        demo.SFH_BASIS = basis
        if scenario == "calibrated":
            demo.CORR_LENGTHS.update(priors["corr_lengths"])
            demo.AMPLITUDES.update(priors["amplitudes"])
            if sensitivity >= 0.1 and not args.skip_sfh:
                demo.SFH_BASIS = oriented
                demo.CORR_LENGTHS["sfh_a"] = ssfr_prior["length"]["re"]["median"] * proxy_re
                demo.AMPLITUDES["sfh_a"] = ssfr_prior["amplitude"]["median"] / sensitivity
        model = demo.build_model(args.size, mass_mean, fixed=fixed, nu=1.5, k_max=k_max)
        error_floor = np.hypot(0.05, fit_emulator.spec.band_error(z))
        forward = ForwardModel.build(obs, psf, model, fit_emulator, model_error_frac=error_floor)
        start = time.monotonic()
        adam = find_map(
            forward, jnp.zeros(forward.n_params), n_rounds=3, steps_per_round=200, final_steps=250
        )
        polish = trust_region_newton(
            lambda theta: -forward.log_posterior(theta),
            adam.theta.astype(jnp.float64),
            max_iter=args.newton_steps,
            max_cg=130,
            verbose=False,
        )
        maps = demo.fit_param_maps(model, polish.theta)
        params = {**maps, **fixed}
        n = args.size
        array = np.stack(
            [np.broadcast_to(params[p], (n, n)).ravel() for p in test_emulator.param_names], axis=1
        )
        flux = np.asarray(test_emulator.predict(jnp.asarray(array))).T.reshape(-1, n, n)
        test = np.stack([fftconvolve(f, p, mode="same") for f, p in zip(flux, test_psfs)])
        predicted_ssfr = formed_ssfr100(np.stack([maps[p] for p in demo.RATIOS]), agebins)
        maps["logssfr_100myr"] = predicted_ssfr
        delta = {k: maps[k][score] - truth[k][score] for k in truth}
        outcomes[scenario] = {
            "map_rmse": {k: float(np.sqrt(np.mean(d**2))) for k, d in delta.items()},
            "centered_map_rmse": {k: float(np.std(d)) for k, d in delta.items()},
            "withheld_rmse_in_noise_units": dict(
                zip(
                    TEST_BANDS,
                    (
                        np.sqrt(np.mean((test[:, score] - withheld[:, score]) ** 2, axis=1)) / sigma
                    ).tolist(),
                )
            ),
            "fit_log_likelihood": float(forward.log_likelihood(polish.theta)),
            "converged": polish.converged,
            "status": polish.reason,
            "n_params": forward.n_params,
            "seconds": time.monotonic() - start,
            "n_hvp": polish.n_hvp,
        }
    demo.CORR_LENGTHS.update(original_lengths)
    demo.AMPLITUDES.update(original_amplitudes)
    return {
        **row,
        "score_pixels": int(score.sum()),
        "re_proxy_arcsec": proxy_re * obs.pixel_scale,
        "re_stellar_arcsec": galaxy["re_px"] * galaxy["pixel_arcsec"],
        "ssfr_basis_sensitivity": float(sensitivity),
        "pixel_arcsec": obs.pixel_scale,
        "k_max_rad_per_px": k_max,
        **outcomes,
    }


def main():
    """Run the high-redshift validation against independent mock imaging."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("measurements", "calibration", "cache", "emulator", "psf-dir", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--newton-steps", type=int, default=85)
    p.add_argument("--snr", type=float, default=40)
    p.add_argument("--mtf", type=float, default=0.15)
    p.add_argument("--per-snapshot", type=int, default=2)
    p.add_argument("--seed", type=int, default=261122)
    p.add_argument("--skip-sfh", action="store_true")
    args = p.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    calibration = json.loads(args.calibration.read_text())
    data = json.loads(args.measurements.read_text())
    eligible = [r for r in data["galaxies"] if r["split"] == "test" and r["fields"]]
    fit = load_emulator(
        args.emulator, params=demo.SPS_PARAM_NAMES, bands=[f"JWST/NIRCam.{b}" for b in FIT_BANDS]
    )
    held = load_emulator(
        args.emulator, params=demo.SPS_PARAM_NAMES, bands=[f"JWST/NIRCam.{b}" for b in TEST_BANDS]
    )
    result = {"devices": list(map(str, jax.devices())), "snr": args.snr, "galaxies": []}
    for snap in sorted({r["snapshot"] for r in eligible}, reverse=True):
        rows = [r for r in eligible if r["snapshot"] == snap]
        rows = [
            r
            for pair in zip(
                [r for r in rows if r["group"] == "active"],
                [r for r in rows if r["group"] == "low_activity"],
            )
            for r in pair
        ][: args.per_snapshot]
        for row in rows:
            print(row["id"], row["z"], row["group"], flush=True)
            path = download_drive(row["key"], args.cache / row["file"])
            minimal = {k: v for k, v in row.items() if k != "fields"}
            result["galaxies"].append(benchmark(minimal, path, fit, held, calibration, args))
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
