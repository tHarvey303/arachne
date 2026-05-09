#!/usr/bin/env python3
"""Generate a mock JWST NIRCam dataset and training library using Synthesizer + BPASS.

Steps
-----
1. **Training library** — sample 3000 SSP models from the BPASS grid (varying
   log10_mass, log10_age, log10_metallicity, tau_v), compute NIRCam F115W–F356W
   photometry with PacmanEmission, and save to HDF5 in the synference layout
   expected by ``SPSMLPEmulator.from_synference_library``.

2. **Mock FITS images** — build a 2-component galaxy (compact bulge + extended
   disk) with the arachne forward model itself (GMM K=2), using a lightly
   pre-trained emulator, add Gaussian noise and a Gaussian PSF, and write the
   result to per-band FITS files so ``ObservationCube.from_fits`` can load them.

3. **Truth record** — write ``true_params.json`` with the unconstrained GMM
   parameters used to generate the mock so the fit can be compared.

Outputs (all under ``outputs/mock_data/``)
------------------------------------------
- ``training_library.h5``         — synference-format photometry library
- ``{band}_sci.fits``             — per-band flux images (nJy)
- ``{band}_var.fits``             — per-band variance images (nJy²)
- ``{band}_psf.fits``             — per-band PSF kernels
- ``true_params.json``            — truth GMM parameter dict
- ``true_image.npy``              — noiseless model image (N_bands, H, W)

Usage
-----
    python scripts/generate_mock_jwst.py

Run ``scripts/fit_mock_jwst.py`` afterwards to train the emulator and fit.

Grid
----
Requires the BPASS-2.2.1 + Cloudy SPS grid at::

    /Users/user/Documents/PhD/synthesizer/grids/
    bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from scipy.signal import windows
from unyt import Msun, yr

warnings.filterwarnings("ignore")  # suppress unyt/synthesizer informational warnings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_DIR = Path("/Users/user/Documents/PhD/synthesizer/grids")
GRID_NAME = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps"
OUTDIR = Path("outputs/mock_data")
OUTDIR.mkdir(parents=True, exist_ok=True)

REDSHIFT = 1.5
PIXEL_SCALE = 0.031  # arcsec/pixel  (JWST NIRCam short-wavelength channel)
NPIX = 32
FILTER_CODES = [
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F162M",
    "JWST/NIRCam.F182M",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F210M",
    "JWST/NIRCam.F250M",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F300M",
    "JWST/NIRCam.F335M",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F430M",
    "JWST/NIRCam.F444W",
]
N_BANDS = len(FILTER_CODES)
N_LIB = 3000  # training library size
NOISE_SIGMA_NJY = 5.0  # per-pixel noise std in nJy
PSF_FWHM_PIX = 2.0  # Gaussian PSF FWHM in pixels (uniform across bands for simplicity)
PSF_SIZE = 15  # kernel array size

# Zsun used by BPASS
ZSUN = 0.02

# SPS parameter names and physical bounds (must match fit_mock_jwst.py)
SPS_PARAM_NAMES = ["log_stellar_mass", "log_age", "log_metallicity", "tau_v"]
PARAM_BOUNDS = {
    "log_stellar_mass": (5.0, 11.0),
    "log_age": (7.0, 10.3),
    "log_metallicity": (-3.0, 0.5),
    "tau_v": (0.0, 3.0),
}

# ---------------------------------------------------------------------------
# Helper: one SED → NIRCam photometry
# ---------------------------------------------------------------------------


def _compute_photometry(
    log10_mass: float,
    log10_age: float,
    metallicity: float,
    tau_v: float,
    grid,
    fc,
) -> np.ndarray:
    """Return NIRCam photometry (nJy) for one SSP model via PacmanEmission."""
    from synthesizer.emission_models import PacmanEmission
    from synthesizer.emission_models.attenuation import PowerLaw
    from synthesizer.parametric import SFH, Stars, ZDist
    from synthesizer.parametric.galaxy import Galaxy

    stars = Stars(
        grid.log10ages,
        grid.metallicities,
        sf_hist=SFH.Constant(max_age=10**log10_age * yr),
        metal_dist=ZDist.DeltaConstant(metallicity=metallicity),
        initial_mass=10**log10_mass * Msun,
    )
    gal = Galaxy(stars, redshift=REDSHIFT)
    model = PacmanEmission(grid, tau_v=tau_v, dust_curve=PowerLaw(slope=-1), fesc=0.0)
    sed = gal.stars.get_spectra(model)
    sed.get_fnu(cosmo, REDSHIFT)
    return np.array(
        [float(f.apply_filter(sed._fnu, nu=sed.obsnu)) for f in fc],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Step 1: Training library
# ---------------------------------------------------------------------------


def make_training_library(grid, fc) -> None:
    """Generate and save a 3000-model SSP photometry library."""
    print(f"\n--- Step 1: Training library ({N_LIB} models) ---")
    rng = np.random.default_rng(42)

    # Sample parameters
    log10_masses = rng.uniform(PARAM_BOUNDS["log_stellar_mass"][0],
                               PARAM_BOUNDS["log_stellar_mass"][1], N_LIB).astype(np.float32)
    # Age: uniformly sample indices from the BPASS grid
    age_indices = rng.integers(0, len(grid.log10ages), N_LIB)
    log10_ages = grid.log10ages[age_indices].astype(np.float32)
    # Metallicity: uniformly sample from BPASS grid values
    Z_indices = rng.integers(0, len(grid.metallicities), N_LIB)
    metallicities = np.array(grid.metallicities)[Z_indices].astype(np.float32)
    log10_mets = np.log10(metallicities / ZSUN).astype(np.float32)
    tau_vs = rng.uniform(PARAM_BOUNDS["tau_v"][0],
                         PARAM_BOUNDS["tau_v"][1], N_LIB).astype(np.float32)

    photometry = np.zeros((N_LIB, N_BANDS), dtype=np.float32)
    failed = 0
    for i in range(N_LIB):
        if i % 300 == 0:
            print(f"  model {i}/{N_LIB} ...")
        try:
            phot = _compute_photometry(
                log10_masses[i], log10_ages[i], metallicities[i], tau_vs[i], grid, fc
            )
            if np.all(phot > 0) and np.all(np.isfinite(phot)):
                photometry[i] = phot
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"  {failed} models failed / had non-positive flux (will be filtered in training).")

    # Parameters array: shape (N_params, N_models) as expected by SPSMLPEmulator
    params = np.stack([log10_masses, log10_ages, log10_mets, tau_vs], axis=0)  # (4, N)

    outpath = OUTDIR / "training_library.h5"
    with h5py.File(outpath, "w") as f:
        f.create_dataset("Grid/Parameters", data=params)
        f.create_dataset("Grid/Photometry", data=photometry.T)  # (N_bands, N_models)
        f.attrs["ParameterNames"] = [s.encode() for s in SPS_PARAM_NAMES]
        f.attrs["FilterCodes"] = [s.encode() for s in FILTER_CODES]
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Step 2: Mock FITS images (generated by the arachne forward model)
# ---------------------------------------------------------------------------


def make_psf_kernel() -> np.ndarray:
    """Gaussian PSF kernel, shape (PSF_SIZE, PSF_SIZE), sum=1."""
    sigma = PSF_FWHM_PIX / 2.355
    row = windows.gaussian(PSF_SIZE, sigma)
    kernel = np.outer(row, row).astype(np.float32)
    return kernel / kernel.sum()


def make_mock_images() -> dict:
    """Generate mock images and save FITS files.  Returns truth parameter dict."""
    from scipy.ndimage import convolve

    print("\n--- Step 2: Mock FITS images ---")

    # ---- truth parameters for K=2 GMM ----
    # Layout per component: [mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho,
    #                         sps_raw_mass, sps_raw_age, sps_raw_met, sps_raw_tauv]
    # sps_raw = logit((sps_phys - lo) / (hi - lo))

    def logit(p: float) -> float:
        return float(np.log(p / (1.0 - p)))

    def to_raw(phys: float, lo: float, hi: float) -> float:
        return logit((phys - lo) / (hi - lo))

    lo_m, hi_m = PARAM_BOUNDS["log_stellar_mass"]
    lo_a, hi_a = PARAM_BOUNDS["log_age"]
    lo_z, hi_z = PARAM_BOUNDS["log_metallicity"]
    lo_t, hi_t = PARAM_BOUNDS["tau_v"]

    center = float(NPIX // 2)

    # Bulge: compact round, high surface density, old, solar metallicity, low dust
    bulge_log_mass = 8.5   # high per-pixel surface brightness
    bulge_log_age = 9.0    # 1 Gyr
    bulge_log_met = 0.0    # solar
    bulge_tau_v = 0.3

    theta_bulge = [
        center, center,                          # mu_y, mu_x
        np.log(3.5), np.log(3.5),               # log_sigma_y, log_sigma_x
        np.arctanh(0.0),                         # atanh_rho
        to_raw(bulge_log_mass, lo_m, hi_m),
        to_raw(bulge_log_age, lo_a, hi_a),
        to_raw(bulge_log_met, lo_z, hi_z),
        to_raw(bulge_tau_v, lo_t, hi_t),
    ]

    # Disk: extended elliptical, lower surface density, young, sub-solar, dusty
    disk_log_mass = 7.5   # lower per-pixel surface brightness
    disk_log_age = 8.5    # ~300 Myr
    disk_log_met = -0.5   # sub-solar
    disk_tau_v = 1.2

    theta_disk = [
        center, center,                          # mu_y, mu_x (co-centred)
        np.log(10.0), np.log(7.0),              # log_sigma_y, log_sigma_x (elongated)
        np.arctanh(0.3),                         # atanh_rho
        to_raw(disk_log_mass, lo_m, hi_m),
        to_raw(disk_log_age, lo_a, hi_a),
        to_raw(disk_log_met, lo_z, hi_z),
        to_raw(disk_tau_v, lo_t, hi_t),
    ]

    # theta = [bulge_params, disk_params] (K=2, bulge first)
    theta_true = np.array(theta_bulge + theta_disk, dtype=np.float32)

    truth = {
        "theta_true": theta_true.tolist(),
        "bulge": {
            "log_stellar_mass": bulge_log_mass,
            "log_age": bulge_log_age,
            "log_metallicity": bulge_log_met,
            "tau_v": bulge_tau_v,
            "sigma_y_pix": 3.5,
            "sigma_x_pix": 3.5,
        },
        "disk": {
            "log_stellar_mass": disk_log_mass,
            "log_age": disk_log_age,
            "log_metallicity": disk_log_met,
            "tau_v": disk_tau_v,
            "sigma_y_pix": 10.0,
            "sigma_x_pix": 7.0,
        },
        "pixel_scale_arcsec": PIXEL_SCALE,
        "npix": NPIX,
        "redshift": REDSHIFT,
        "filter_codes": FILTER_CODES,
        "noise_sigma_nJy": NOISE_SIGMA_NJY,
    }

    # We need the emulator to generate the mock.  Load it if already trained,
    # otherwise fall back to a simple mass-scaling proxy so the mock FITS files
    # can still be inspected without running the full training first.
    emulator_path = OUTDIR / "emulator.eqx"
    if emulator_path.exists():
        print("  Loading pre-trained emulator from", emulator_path)
        from arachne.emulator.jax_mlp_emulator import SPSMLPEmulator
        emulator = SPSMLPEmulator.load(
            emulator_path,
            param_names=SPS_PARAM_NAMES,
            band_names=FILTER_CODES,
        )
    else:
        print("  Emulator not yet trained — building placeholder mock from analytic SED proxy.")
        print("  Run fit_mock_jwst.py (which trains the emulator first) then re-run this script")
        print("  to regenerate the mock with the real emulator.  Writing placeholder FITS now.")
        emulator = None

    psf_kernel = make_psf_kernel()

    if emulator is not None:
        import jax.numpy as jnp
        from arachne.data.observation import ObservationCube
        from arachne.data.psf import PSFModel
        from arachne.forward_model.pipeline import ForwardModel
        from arachne.spatial.gmm import GaussianMixtureSpatialModel

        # Build a dummy observation (will be replaced by noisy mock below)
        dummy_flux = np.ones((N_BANDS, NPIX, NPIX), dtype=np.float32)
        dummy_var = np.ones((N_BANDS, NPIX, NPIX), dtype=np.float32)
        dummy_mask = np.ones((N_BANDS, NPIX, NPIX), dtype=np.float32)
        obs = ObservationCube(
            flux=dummy_flux, variance=dummy_var, mask=dummy_mask,
            band_names=FILTER_CODES, pixel_scale=PIXEL_SCALE,
        ).to_jax()

        psf_kernels = np.stack([psf_kernel] * N_BANDS, axis=0)
        psf_model = PSFModel(kernels=psf_kernels, band_names=FILTER_CODES)

        spatial_model = GaussianMixtureSpatialModel(
            n_components=2,
            sps_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(NPIX, NPIX),
        )

        fwd = ForwardModel.build(obs=obs, psf_model=psf_model,
                                 spatial_model=spatial_model, emulator=emulator)

        theta_jax = jnp.array(theta_true)
        true_image = np.array(fwd._model_image(theta_jax))  # (N_bands, H, W)
        print(f"  True image: shape {true_image.shape}, "
              f"flux range [{true_image.min():.1f}, {true_image.max():.1f}] nJy")
    else:
        # Analytic proxy: Gaussian blobs with rough SED colours
        from scipy.ndimage import convolve as scipy_convolve
        true_image = _analytic_mock(psf_kernel)

    np.save(OUTDIR / "true_image.npy", true_image)

    rng = np.random.default_rng(0)
    for b, band in enumerate(FILTER_CODES):
        safe = band.replace("/", "_").replace(".", "_")
        flux_2d = true_image[b]

        # Apply PSF
        flux_psf = _convolve_psf(flux_2d, psf_kernel)

        # Add noise
        noise = rng.normal(0.0, NOISE_SIGMA_NJY, size=flux_psf.shape).astype(np.float32)
        flux_noisy = flux_psf + noise
        variance = np.full_like(flux_psf, NOISE_SIGMA_NJY ** 2)

        _write_fits(flux_noisy, OUTDIR / f"{safe}_sci.fits")
        _write_fits(variance, OUTDIR / f"{safe}_var.fits")
        _write_fits(psf_kernel, OUTDIR / f"{safe}_psf.fits")
        print(f"  {band}: peak flux {flux_psf.max():.1f} nJy, noise {NOISE_SIGMA_NJY} nJy/pix")

    truth_path = OUTDIR / "true_params.json"
    with open(truth_path, "w") as f:
        json.dump(truth, f, indent=2)
    print(f"  Saved truth: {truth_path}")
    return truth


def _convolve_psf(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve
    return convolve(image.astype(np.float64), psf.astype(np.float64), mode="reflect").astype(np.float32)


def _analytic_mock(psf_kernel: np.ndarray) -> np.ndarray:
    """Fallback: two Gaussian blobs with rough NIRCam colours in nJy."""
    H = W = NPIX
    yy, xx = np.mgrid[:H, :W].astype(float)
    cy, cx = H / 2, W / 2

    # Bulge blob
    bulge = np.exp(-((yy - cy)**2 / (2 * 3.5**2) + (xx - cx)**2 / (2 * 3.5**2)))
    # Disk blob (elongated)
    disk = np.exp(-((yy - cy)**2 / (2 * 10.0**2) + (xx - cx)**2 / (2 * 7.0**2)))

    # Rough NIRCam colours (nJy for 10^8.5 bulge mass, 10^7.5 disk mass)
    bulge_colours = np.array([80.0, 100.0, 120.0, 150.0, 160.0], dtype=np.float32)
    disk_colours = np.array([30.0, 35.0, 40.0, 38.0, 36.0], dtype=np.float32)

    images = []
    for b in range(N_BANDS):
        img = bulge_colours[b] * bulge + disk_colours[b] * disk
        images.append(img.astype(np.float32))
    return np.stack(images, axis=0)


def _write_fits(array: np.ndarray, path: Path) -> None:
    hdu = fits.PrimaryHDU(data=array.astype(np.float32))
    hdu.writeto(str(path), overwrite=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from synthesizer.grid import Grid
    from synthesizer.instruments import FilterCollection

    print("Loading BPASS grid ...")
    grid = Grid(GRID_NAME, grid_dir=str(GRID_DIR))
    print(f"  Ages: {len(grid.log10ages)} bins  "
          f"[{grid.log10ages.min():.1f}, {grid.log10ages.max():.1f}] log10(yr)")
    print(f"  Metallicities: {len(grid.metallicities)} bins  "
          f"[{float(grid.metallicities.min()):.1e}, {float(grid.metallicities.max()):.1e}]")

    # Use native filter wavelength grids (do NOT pass new_lam=grid.lam —
    # the BPASS grid spans X-ray to radio and confuses filter interpolation)
    fc = FilterCollection(FILTER_CODES)

    make_training_library(grid, fc)
    make_mock_images()

    print("\nDone. Now run:  python scripts/fit_mock_jwst.py")


if __name__ == "__main__":
    main()
