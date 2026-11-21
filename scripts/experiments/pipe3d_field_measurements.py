"""Translate the MaNGA pyPipe3D continuum products into covariance measurements."""

from __future__ import annotations

import json
import re
from warnings import catch_warnings, simplefilter

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from field_covariance import NU, BinGeometry, fit_profiles

FIELD_UNITS = {
    "Av": "mag",
    "log10metallicity": "dex",
    "sfh_10_100_over_100_1000_myr": "dex",
}
FLOORS = dict(zip(FIELD_UNITS, (0.025, 0.025, 0.20), strict=True))
AMPLITUDES = dict(zip(FIELD_UNITS, ((0.005, 2.5), (0.005, 1.5), (0.015, 3.0)), strict=True))
LENGTHS_RE = np.geomspace(0.025, 2.5, 39)


def template_metadata(catalogue):
    """Read the ordered SSP ages, metallicity mass fractions and light per mass."""
    header = fits.getheader(catalogue, "MASTAR_CB19_SSP")
    names = [header[f"NAME{i}"] for i in range(header["NAXIS2"])]
    age_met = [re.fullmatch(r"spec_ssp_([0-9.]+)_z(\d+)\.spec", name) for name in names]
    age = np.array([float(match[1]) for match in age_met]) * 1000
    metallicity = np.array([float("0." + match[2]) for match in age_met])
    norm = np.array([header[f"NORM{i}"] for i in range(len(names))])
    return age, metallicity, norm


def sfh_mass(weights, ages_myr, edges_myr):
    """Sum SSP masses by age bin, with the oldest endpoint included."""
    indices = np.searchsorted(edges_myr, ages_myr, side="right") - 1
    indices[ages_myr == edges_myr[-1]] = len(edges_myr) - 2
    return np.stack([weights[indices == i].sum(axis=0) for i in range(len(edges_myr) - 1)])


def read_maps(path, target, template):
    """Load measurements, map masks and geometry from the released products."""
    ages, metallicity, norm = template
    with fits.open(path, memmap=False) as hdus:
        og = hdus["ORG_HDR"].header
        with catch_warnings():
            simplefilter("ignore", FITSFixedWarning)
            wcs = WCS(og).celestial
        pixel_arcsec = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600)
        x0, y0 = wcs.world_to_pixel_values(target["ra"], target["dec"])
        av_plane = hdus["SSP"].header["FILE_11"]
        if "Av_ssp" not in av_plane or "mass weighted metallicity" not in (
            hdus["SSP"].header["DESC_9"].lower()
        ):
            raise ValueError(f"Unexpected SSP plane layout in {path}")
        ssp = np.asarray(hdus["SSP"].data, dtype=float)
        sfh = hdus["SFH"]
        for i, (age, metal) in enumerate(zip(ages, metallicity, strict=True)):
            desc = sfh.header[f"DESC_{i}"]
            entry = re.search(r"age-met ([0-9.]+)-([0-9.]+) SSP", desc)
            if entry is None or not np.allclose(
                (float(entry[1]), float(entry[2])), (age / 1000, metal), rtol=0, atol=0.000051
            ):
                raise ValueError(f"SSP order differs from catalogue at {path}, plane {i}")
        light = np.asarray(sfh.data[: len(norm)], dtype=float).clip(min=0)
        stellar_templates = light / norm[:, None, None]
        masked = (hdus["GAIA_MASK"].data != 0) | (hdus["SELECT_REG"].data <= 0)
        fwhm_px = float(og["RFWHM"]) / pixel_arcsec
    yy, xx = np.indices(masked.shape)
    east, north = x0 - xx, yy - y0
    sin_pa, cos_pa = np.sin(np.deg2rad(target["pa_deg"])), np.cos(np.deg2rad(target["pa_deg"]))
    major = sin_pa * east + cos_pa * north
    minor = cos_pa * east - sin_pa * north
    re_px = target["re_arcsec"] / pixel_arcsec
    radius = np.hypot(major, minor / target["axis_ratio"]) / re_px
    segmentation = np.where(masked | ~np.isfinite(ssp[1]), 0, ssp[1]).astype(np.int32)
    rest_mass = stellar_templates.sum(axis=0)
    recent, intermediate = sfh_mass(stellar_templates, ages, [10, 100, 1000])
    regularizer = 1e-4 * rest_mass
    with np.errstate(divide="ignore", invalid="ignore"):
        contrast = np.log10((recent + regularizer) / (intermediate + regularizer)) + 1
    # The release's continuum metallicity is log10(Z/Zsun) with Zsun=0.02.
    maps = {
        "Av": (ssp[11], ssp[12]),
        "log10metallicity": (ssp[9] + np.log10(0.02), ssp[10]),
        "sfh_10_100_over_100_1000_myr": (contrast, np.zeros_like(contrast)),
    }
    crop = (radius <= 1.5) & (segmentation > 0) & (ssp[0] > 0)
    return {
        "segmentation": segmentation,
        "maps": maps,
        "crop": crop,
        "radius": radius,
        "continuum": ssp[0],
        "fwhm_px": fwhm_px,
        "pixel_arcsec": pixel_arcsec,
        "re_px": re_px,
        "stellar_templates": stellar_templates,
        "ages_myr": ages,
        "log_mass": ssp[19],
        "pixel_center": (x0, y0),
    }


def extract_bins(galaxy, rng, max_bins=160):
    """Sample continuum bins within 1.5 Re and retain one measurement per bin."""
    seg, crop = galaxy["segmentation"], galaxy["crop"]
    labels = np.unique(seg[crop])
    if len(labels) < 25:
        return None
    if len(labels) > max_bins:
        labels = np.sort(rng.choice(labels, max_bins, replace=False))
    geom = BinGeometry(
        seg, galaxy["continuum"], labels, galaxy["radius"], galaxy["fwhm_px"] / 2.35482
    )
    yx = np.array([np.argwhere((seg == label) & crop)[0] for label in labels]).T
    index = tuple(yx)
    fields = {}
    for name, (values, errors) in galaxy["maps"].items():
        y = values[index]
        sigma = np.hypot(errors[index], FLOORS[name])
        if name == "Av":
            # Keep the physical boundary in the aperture. The Gaussian fit is an approximation
            # when the upstream optimiser puts many distinct bins exactly on the same floor.
            sigma = np.where((y >= 0) & (errors[index] >= 0), sigma, np.nan)
        fields[name] = (y, sigma)
    return geom, fields


def measure_sample(records, cache, outdir, catalogue, seed):
    """Measure per-galaxy profile likelihoods with stable splits and bin sampling."""
    template = template_metadata(catalogue)
    result = {
        "version": 1,
        "nu": NU,
        "ell_over_re_grid": LENGTHS_RE.tolist(),
        "measurement_error_floor": FLOORS,
        "seed": seed,
        "galaxies": [],
    }
    for row in records:
        galaxy = read_maps(cache / row["file"], row, template)
        rng = np.random.default_rng(np.random.SeedSequence([seed, *map(ord, row["id"])]))
        sample = extract_bins(galaxy, rng)
        beam_re = galaxy["fwhm_px"] / galaxy["re_px"]
        fields = {}
        if sample is not None:
            geom, maps = sample
            fields = fit_profiles(geom, maps, LENGTHS_RE * galaxy["re_px"], AMPLITUDES)
            for name, measurement in fields.items():
                y, _ = maps[name]
                measurement["fraction_zero_av"] = float(np.mean(y == 0)) if name == "Av" else None
        result["galaxies"].append(
            {
                **row,
                "beam_fwhm_over_re": beam_re,
                "fields": fields,
            }
        )
        print(f"{row['id']} {row['group']} {row['split']} {len(fields)} fields", flush=True)
        (outdir / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
