"""Compare fixed spatial priors on controlled photometry from held-out MaNGA galaxies."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Cosmology
from calibrate_pipe3d_spatial_priors import CATALOGUE, ROOT, download
from pipe3d_field_measurements import read_maps, sfh_mass, template_metadata
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from arachne import ForwardModel, ObservationCube, PSFModel, load_emulator, mass_map_from_image
from arachne.inference.initialisation import find_map
from arachne.inference.newton import trust_region_newton
from arachne.postprocess.recipe import continuity_agebins
from arachne.priors.spatial_calibration import load_spatial_calibration

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import demo_correlated_field as demo  # noqa: E402

FIT_BANDS = [
    "HST/ACS_WFC.F435W",
    "HST/ACS_WFC.F606W",
    "HST/ACS_WFC.F814W",
    "HST/WFC3_IR.F105W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F356W",
]
TEST_BANDS = ["HST/ACS_WFC.F475W", "JWST/NIRCam.F182M"]


def prepare_maps(galaxy, spec, agebins, n):
    """Regrid the IFU measurements and construct emulator-input truth maps."""
    valid = (galaxy["segmentation"] > 0) & np.isfinite(galaxy["log_mass"])
    valid &= (galaxy["log_mass"] > 0) & (galaxy["stellar_templates"].sum(axis=0) > 0)
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    near = tuple(nearest)
    # The scoring mask excludes extrapolated pixels. A short taper propagates light
    # through the IFU edge during convolution without inserting a sharp step.
    mass = 10.0 ** (galaxy["log_mass"][near] + 0.20) * np.exp(-distance / 3)
    x, y = galaxy["pixel_center"]
    left, top = round(x) - n, round(y) - n
    if min(top, left) < 0 or top + 2 * n > valid.shape[0] or left + 2 * n > valid.shape[1]:
        raise ValueError("Benchmark stamp exceeds the source map")

    def block_sum(image):
        cutout = image[..., top : top + 2 * n, left : left + 2 * n]
        return cutout.reshape((*cutout.shape[:-2], n, 2, n, 2)).sum(axis=(-1, -3))

    total = block_sum(mass)
    templates = galaxy["stellar_templates"][:, near[0], near[1]]
    template_frac = templates / templates.sum(axis=0)
    edges = np.concatenate((agebins[:, 0], agebins[-1:, 1]))
    ages = np.minimum(galaxy["ages_myr"], edges[-1])
    by_age = sfh_mass(template_frac, ages, edges)
    widths = np.diff(edges)
    sfr = block_sum(mass * by_age) / widths[:, None, None]
    sfr += 1e-5 * total / widths.sum()
    ratio = np.log10(sfr[:-1] / sfr[1:])
    truth = {"log_mass": np.log10(total)}
    for name in ("Av", "log10metallicity"):
        truth[name] = block_sum(mass * galaxy["maps"][name][0][near]) / total
    truth.update(zip(spec.sfh_ratio_params, ratio, strict=True))
    for name, image in truth.items():
        if name != "log_mass":
            truth[name] = np.clip(image, *spec.param_bounds[name])
    mask = block_sum(valid) == 4
    score = mask & (block_sum(galaxy["radius"]) <= 4 * 1.25)
    return truth, mask, score


def observe(emulator, truth, fixed, mask, scale, snr, rng):
    """Render controlled ten-band photometry with a Gaussian PSF and white noise."""
    n = mask.shape[0]
    coordinates = np.arange(-5, 6)
    radius2 = coordinates[:, None] ** 2 + coordinates[None, :] ** 2
    fwhm = np.array([1.5, 1.4, 1.4, 1.5, 1.5, 1.6, 1.8, 2.0, 1.5, 1.6])
    sigma = fwhm / (2.35482 * scale)
    kernels = np.exp(-0.5 * radius2 / sigma[:, None, None] ** 2)
    kernels /= kernels.sum(axis=(1, 2), keepdims=True)

    def render(maps):
        inputs = {**fixed, **maps}
        params = np.stack(
            [np.broadcast_to(inputs[p], (n, n)).ravel() for p in emulator.param_names], axis=1
        )
        predicted = np.asarray(emulator.predict(jnp.asarray(params))).T.reshape(-1, n, n)
        return np.stack(
            [fftconvolve(image, psf, mode="same") for image, psf in zip(predicted, kernels)]
        )

    clean = render(truth)
    var = (np.percentile(clean[:, mask], 95, axis=1) / snr) ** 2
    noisy = clean + np.sqrt(var[:, None, None]) * rng.normal(size=clean.shape)
    nfit = len(FIT_BANDS)
    obs = ObservationCube(
        flux=noisy[:nfit],
        variance=np.broadcast_to(var[:nfit, None, None], noisy[:nfit].shape),
        mask=np.broadcast_to(mask, noisy[:nfit].shape),
        band_names=FIT_BANDS,
        pixel_scale=scale,
    )
    return obs, PSFModel(kernels[:nfit], FIT_BANDS), noisy[nfit:], var[nfit:], render


def sfh_contrast(maps, ratio_names, agebins):
    """Compute the 10–100 / 100–1000 Myr contrast implied by model-input ratios."""
    ratios = np.stack([maps[name] for name in ratio_names])
    first = np.zeros_like(ratios[:1])
    sfr = 10 ** -np.cumsum(np.concatenate((first, ratios)), axis=0)
    means = []
    for low, high in ((10, 100), (100, 1000)):
        overlap = (np.minimum(agebins[:, 1], high) - np.maximum(agebins[:, 0], low)).clip(0)
        means.append(np.tensordot(overlap / (high - low), sfr, axes=1))
    return np.log10(means[0] / means[1])


def run_target(row, galaxy, full, emulator, args):
    """Use the same truth, data, start recipe and optimizer for both fits."""
    recipe = full.spec.simulator
    cosmo = Cosmology.from_format(recipe["cosmology"], format="yaml")
    agebins = continuity_agebins(row["z"], cosmo, recipe["sfh"]["agebins"])
    truth, mask, score = prepare_maps(galaxy, emulator.spec, agebins, args.size)
    scale = 2 * galaxy["pixel_arcsec"]
    fixed = {
        "redshift": row["z"],
        "slope": 0.0,
        "fesc_lya": demo.NEUTRAL_GLOBALS["fesc_lya"],
        "dust_bump_amplitude": demo.NEUTRAL_GLOBALS["dust_bump_amplitude"],
    }
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, *map(ord, row["id"])]))
    obs, psf, withheld, withheld_var, render = observe(
        full, truth, fixed, mask, scale, args.snr, rng
    )
    demo.REDSHIFT = row["z"]
    demo.PIXEL_SCALE = scale
    demo.PARAM_BOUNDS.update(emulator.spec.clip_bounds(demo.PARAM_BOUNDS))
    demo.set_sfh("basis2")
    demo.SFH_BASIS, _ = demo.sfh_basis(emulator)
    # A single baseline mass map based only on fitted photometry.
    neutral = {**demo.NEUTRAL_GLOBALS, **demo.NEUTRAL, **fixed}
    mass_mean = mass_map_from_image(obs, emulator, neutral, smooth_sigma_px=1.0)
    priors = load_spatial_calibration(
        args.calibration,
        redshift=row["z"],
        re_pixels=row["re_arcsec"] / scale,
        allow_extrapolation=True,
    )
    original_lengths, original_amplitudes = demo.CORR_LENGTHS.copy(), demo.AMPLITUDES.copy()
    ratio_names = emulator.spec.sfh_ratio_params
    truth_sfh = sfh_contrast(truth, ratio_names, agebins)
    result = {}
    for scenario in ("baseline", "calibrated"):
        if scenario == "calibrated":
            demo.CORR_LENGTHS.update(priors["corr_lengths"])
            demo.AMPLITUDES.update(priors["amplitudes"])
        model = demo.build_model(args.size, mass_mean, fixed=fixed, nu=1.5)
        fm = ForwardModel.build(obs, psf, model, emulator, model_error_frac=0.035)
        start = time.monotonic()
        adam = find_map(
            fm, jnp.zeros(fm.n_params), n_rounds=3, steps_per_round=200, final_steps=250
        )
        polish = trust_region_newton(
            lambda t: -fm.log_posterior(t),
            jnp.asarray(adam.theta, dtype=jnp.float64),
            max_iter=args.newton_steps,
            max_cg=170,
            verbose=False,
        )
        maps = demo.fit_param_maps(model, polish.theta)
        pred = render(maps)[len(FIT_BANDS) :]
        error = np.sqrt(np.mean((pred[:, score] - withheld[:, score]) ** 2, axis=1))
        metrics = {
            name: float(np.sqrt(np.mean((maps[name][score] - truth[name][score]) ** 2)))
            for name in ("Av", "log10metallicity", "log_mass")
        }
        metrics["sfh_contrast"] = float(
            np.sqrt(
                np.mean((sfh_contrast(maps, ratio_names, agebins)[score] - truth_sfh[score]) ** 2)
            )
        )
        result[scenario] = {
            "map_rmse": metrics,
            "withheld_rmse_in_noise_units": dict(
                zip(TEST_BANDS, (error / withheld_var**0.5).tolist())
            ),
            "fit_log_likelihood": float(fm.log_likelihood(polish.theta)),
            "n_params": fm.n_params,
            "seconds": time.monotonic() - start,
            "converged": polish.converged,
            "status": polish.reason,
            "n_hvp": polish.n_hvp,
        }
    demo.CORR_LENGTHS.update(original_lengths)
    demo.AMPLITUDES.update(original_amplitudes)
    return {"scoring_pixels": int(score.sum()), **result}


def main():
    """Run a fixed-split comparison on the requested accelerator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--emulator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--snr", type=float, default=40)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--newton-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=261121)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    sample = json.loads(args.sample.read_text())
    chosen = [
        row
        for group in ("active", "low_activity")
        for row in [r for r in sample if r["group"] == group and r["split"] == "test"][
            : args.per_group
        ]
    ]
    jax.config.update("jax_enable_x64", True)
    full = load_emulator(args.emulator, params=demo.SPS_PARAM_NAMES, bands=FIT_BANDS + TEST_BANDS)
    emulator = load_emulator(args.emulator, params=demo.SPS_PARAM_NAMES, bands=FIT_BANDS)
    catalogue = download(f"{ROOT}/{CATALOGUE}", args.cache / CATALOGUE)
    template = template_metadata(catalogue)
    result = {"devices": list(map(str, jax.devices())), "snr": args.snr, "galaxies": []}
    for row in chosen:
        print(f"{row['id']} {row['plateifu']} {row['group']}", flush=True)
        path = download(row["url"], args.cache / row["file"])
        galaxy = read_maps(path, row, template)
        result["galaxies"].append({**row, **run_target(row, galaxy, full, emulator, args)})
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
