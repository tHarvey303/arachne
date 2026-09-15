#!/usr/bin/env python3
"""Generate a mock JWST NIRCam dataset and training library using Synthesizer + BPASS.

Steps
-----
1. **Training library** -- sample ``N_LIB`` SSP models from the BPASS grid
   (varying log10 mass, log10 age, log10 metallicity, tau_v), compute NIRCam
   photometry with PacmanEmission, and save to HDF5 in the synference layout
   expected by ``SPSMLPEmulator.from_synference_library``.

2. **Mock FITS images** -- build a 2-component galaxy (compact bulge + extended
   disk) with the arachne forward model itself, ``AdditiveComponentModel``
   (K=2): each component is a unit-sum Gaussian light profile times the
   emulator SED of its own SPS parameters, including its own *total* stellar
   mass.  The image is PSF-convolved once (by the same ``PSFConvolver`` the
   fit uses), Gaussian noise is added, and the result is written to per-band
   FITS files so ``ObservationCube.from_fits`` can load them.  This step needs
   the trained toy emulator (``emulator.eqx``); until it exists a placeholder
   analytic mock is written so the files can be inspected.

3. **Truth record** -- write ``true_params.json`` with the unconstrained theta
   (in the model's own layout) used to generate the mock.

Outputs (all under ``outputs/mock_data/`` next to the package root)
--------------------------------------------------------------------
- ``training_library.h5``         -- synference-format photometry library
- ``{band}_sci.fits``             -- per-band flux images (nJy)
- ``{band}_var.fits``             -- per-band variance images (nJy^2)
- ``{band}_psf.fits``             -- per-band PSF kernels
- ``true_params.json``            -- truth theta + physical parameters
- ``true_image.npy``              -- noiseless PSF-convolved image (N_bands, H, W)

Usage
-----
    python scripts/generate_mock_jwst.py            # library + placeholder mock
    python scripts/fit_mock_jwst.py                 # trains emulator, regenerates mock, fits

Grid
----
Requires the BPASS-2.2.1 + Cloudy SPS grid
``bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps.hdf5``; the directory
is taken from ``$SYNTHESIZER_GRID_DIR`` if set, else the cosma work area.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from scipy.signal import windows

warnings.filterwarnings("ignore")  # suppress unyt/synthesizer informational warnings

# ---------------------------------------------------------------------------
# Configuration (shared with fit_mock_jwst.py, which imports these names)
# ---------------------------------------------------------------------------

GRID_DIR = Path(os.environ.get("SYNTHESIZER_GRID_DIR", "/cosma7/data/dp276/dc-harv3/work/grids"))
GRID_NAME = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps"
_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = _ROOT / "outputs/mock_data"

REDSHIFT = 1.5
PIXEL_SCALE = 0.031  # arcsec/pixel  (JWST NIRCam short-wavelength channel)
NPIX = 32
# Bands in the training library (superset) ...
LIBRARY_FILTER_CODES = [
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
# ... and the bands the mock image / emulator / fit actually use.
FILTER_CODES = [
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F356W",
]
N_BANDS = len(FILTER_CODES)
N_LIB = 3000  # training library size
SNR_CENTRE = 20.0  # per-band noise: bulge-centre pixel S/N in the noiseless mock
PSF_FWHM_PIX = 2.0  # Gaussian PSF FWHM in pixels (uniform across bands for simplicity)
PSF_SIZE = 15  # kernel array size

# Zsun used by BPASS
ZSUN = 0.02

# SPS parameter names and physical bounds of the toy emulator.
SPS_PARAM_NAMES = ["log_stellar_mass", "log_age", "log_metallicity", "tau_v"]
PARAM_BOUNDS = {
    "log_stellar_mass": (5.0, 11.0),
    "log_age": (7.0, 10.3),
    "log_metallicity": (-3.0, 0.5),
    "tau_v": (0.0, 3.0),
}
MASS_PARAM = "log_stellar_mass"
EMULATOR_HIDDEN = [256, 256, 256]

# Truth: per-component TOTAL stellar masses (the model's mass parameter is the
# component total, not a per-pixel value) and shapes in pixels.
CENTER = float(NPIX // 2)
BULGE_PHYS = {"log_stellar_mass": 9.5, "log_age": 9.0, "log_metallicity": 0.0, "tau_v": 0.3}
BULGE_SHAPE = dict(sigma_y=2.5, sigma_x=2.5, rho=0.0)
DISK_PHYS = {"log_stellar_mass": 9.8, "log_age": 8.5, "log_metallicity": -0.5, "tau_v": 1.2}
DISK_SHAPE = dict(sigma_y=7.0, sigma_x=4.5, rho=0.3)


def safe_band(band: str) -> str:
    """``JWST/NIRCam.F115W`` -> ``JWST_NIRCam_F115W`` for file names."""
    return band.replace("/", "_").replace(".", "_")


def build_spatial_model():
    """The K=2 AdditiveComponentModel shared by the mock generator and the fit."""
    from arachne.spatial.additive import AdditiveComponentModel

    return AdditiveComponentModel(
        n_components=2,
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(NPIX, NPIX),
        mass_param=MASS_PARAM,
    )


def build_truth_theta(spatial_model) -> np.ndarray:
    """Truth theta in the model's own layout (see AdditiveComponentModel docstring)."""
    import jax.numpy as jnp

    def to_raw(phys: float, lo: float, hi: float) -> float:
        p = (phys - lo) / (hi - lo)
        return float(np.log(p / (1.0 - p)))

    blocks = []
    for phys, shape in ((BULGE_PHYS, BULGE_SHAPE), (DISK_PHYS, DISK_SHAPE)):
        raw = [to_raw(phys[p], *PARAM_BOUNDS[p]) for p in spatial_model.sps_param_names]
        blocks.append(
            [
                CENTER,
                CENTER,
                np.log(shape["sigma_y"]),
                np.log(shape["sigma_x"]),
                np.arctanh(shape["rho"]),
                *raw,
            ]
        )
    theta = spatial_model.join_theta(
        jnp.array(blocks, dtype=jnp.float32), jnp.zeros(spatial_model.n_shared)
    )
    return np.asarray(theta, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper: one SED -> NIRCam photometry
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
    from astropy.cosmology import Planck18 as cosmo
    from synthesizer.emission_models import PacmanEmission
    from synthesizer.emission_models.attenuation import PowerLaw
    from synthesizer.parametric import SFH, Stars, ZDist
    from synthesizer.parametric.galaxy import Galaxy
    from unyt import Msun, yr

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
    """Generate and save an ``N_LIB``-model SSP photometry library."""
    print(f"\n--- Step 1: Training library ({N_LIB} models) ---")
    rng = np.random.default_rng(42)

    lo_m, hi_m = PARAM_BOUNDS["log_stellar_mass"]
    lo_t, hi_t = PARAM_BOUNDS["tau_v"]
    log10_masses = rng.uniform(lo_m, hi_m, N_LIB).astype(np.float32)
    # Age / metallicity: uniformly sample indices from the BPASS grid
    age_indices = rng.integers(0, len(grid.log10ages), N_LIB)
    log10_ages = np.asarray(grid.log10ages)[age_indices].astype(np.float32)
    Z_indices = rng.integers(0, len(grid.metallicities), N_LIB)
    metallicities = np.asarray(grid.metallicities)[Z_indices].astype(np.float32)
    log10_mets = np.log10(metallicities / ZSUN).astype(np.float32)
    tau_vs = rng.uniform(lo_t, hi_t, N_LIB).astype(np.float32)

    photometry = np.zeros((N_LIB, len(LIBRARY_FILTER_CODES)), dtype=np.float32)
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

    OUTDIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTDIR / "training_library.h5"
    with h5py.File(outpath, "w") as f:
        f.create_dataset("Grid/Parameters", data=params)
        f.create_dataset("Grid/Photometry", data=photometry.T)  # (N_bands, N_models)
        f.attrs["ParameterNames"] = [s.encode() for s in SPS_PARAM_NAMES]
        f.attrs["FilterCodes"] = [s.encode() for s in LIBRARY_FILTER_CODES]
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


def load_toy_emulator():
    """Load the trained toy emulator, or None if it has not been trained yet."""
    emulator_path = OUTDIR / "emulator.eqx"
    if not emulator_path.exists():
        return None
    from arachne.emulator.jax_mlp_emulator import SPSMLPEmulator

    return SPSMLPEmulator.load(
        emulator_path,
        param_names=SPS_PARAM_NAMES,
        band_names=FILTER_CODES,
        hidden_sizes=EMULATOR_HIDDEN,
    )


def make_mock_images() -> dict:
    """Generate mock images and save FITS files.  Returns truth parameter dict."""
    print("\n--- Step 2: Mock FITS images ---")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    spatial_model = build_spatial_model()
    theta_true = build_truth_theta(spatial_model)
    psf_kernel = make_psf_kernel()

    emulator = load_toy_emulator()
    if emulator is not None:
        print(f"  Using trained emulator {OUTDIR / 'emulator.eqx'}")
        import jax.numpy as jnp

        from arachne.data.psf import PSFModel
        from arachne.psf.convolution import PSFConvolver

        psf_model = PSFModel(kernels=np.stack([psf_kernel] * N_BANDS), band_names=FILTER_CODES)
        convolver = PSFConvolver(psf_model, image_shape=(NPIX, NPIX))
        # Unconvolved additive image, convolved ONCE by the same operator the fit uses.
        unconvolved = spatial_model.model_image(jnp.asarray(theta_true), emulator, (NPIX, NPIX))
        true_image = np.asarray(convolver(unconvolved), dtype=np.float32)
        print(
            f"  True image: shape {true_image.shape}, "
            f"flux range [{true_image.min():.2f}, {true_image.max():.1f}] nJy"
        )
    else:
        print("  Emulator not yet trained -- writing a placeholder analytic mock.")
        print("  Run fit_mock_jwst.py (which trains the emulator, then regenerates the mock).")
        true_image = _analytic_mock(psf_kernel)

    np.save(OUTDIR / "true_image.npy", true_image)

    # Per-band noise: bulge-centre pixel S/N = SNR_CENTRE.
    c = NPIX // 2
    sigma = (true_image[:, c, c] / SNR_CENTRE).astype(np.float32)
    rng = np.random.default_rng(0)
    for b, band in enumerate(FILTER_CODES):
        flux_noisy = true_image[b] + rng.normal(0.0, sigma[b], size=true_image[b].shape)
        variance = np.full_like(true_image[b], sigma[b] ** 2)
        _write_fits(flux_noisy.astype(np.float32), OUTDIR / f"{safe_band(band)}_sci.fits")
        _write_fits(variance, OUTDIR / f"{safe_band(band)}_var.fits")
        _write_fits(psf_kernel, OUTDIR / f"{safe_band(band)}_psf.fits")
        print(f"  {band}: peak flux {true_image[b].max():.1f} nJy, noise {sigma[b]:.2f} nJy/pix")

    truth = {
        "theta_true": theta_true.tolist(),
        "theta_layout": "per component [mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho, "
        "sps_raw...]; sps in sps_param_names order; bulge block first",
        "sps_param_names": spatial_model.sps_param_names,
        "log_stellar_mass_is_component_total": True,
        "bulge": {
            **BULGE_PHYS,
            **{f"{k}_pix" if k != "rho" else k: v for k, v in BULGE_SHAPE.items()},
        },
        "disk": {
            **DISK_PHYS,
            **{f"{k}_pix" if k != "rho" else k: v for k, v in DISK_SHAPE.items()},
        },
        "pixel_scale_arcsec": PIXEL_SCALE,
        "npix": NPIX,
        "redshift": REDSHIFT,
        "filter_codes": FILTER_CODES,
        "noise_sigma_nJy": sigma.tolist(),
        "snr_centre": SNR_CENTRE,
        "generated_with_trained_emulator": emulator is not None,
    }
    truth_path = OUTDIR / "true_params.json"
    with open(truth_path, "w") as f:
        json.dump(truth, f, indent=2)
    print(f"  Saved truth: {truth_path}")
    return truth


def _analytic_mock(psf_kernel: np.ndarray) -> np.ndarray:
    """Fallback: two Gaussian blobs with rough NIRCam colours in nJy (PSF-convolved)."""
    from scipy.ndimage import convolve

    H = W = NPIX
    yy, xx = np.mgrid[:H, :W].astype(float)
    cy, cx = CENTER, CENTER
    bulge = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * BULGE_SHAPE["sigma_y"] ** 2))
    disk = np.exp(
        -(
            (yy - cy) ** 2 / (2 * DISK_SHAPE["sigma_y"] ** 2)
            + (xx - cx) ** 2 / (2 * DISK_SHAPE["sigma_x"] ** 2)
        )
    )
    bulge_colours = np.array([80.0, 100.0, 120.0, 150.0, 160.0], dtype=np.float32)
    disk_colours = np.array([30.0, 35.0, 40.0, 38.0, 36.0], dtype=np.float32)
    images = []
    for b in range(N_BANDS):
        img = bulge_colours[b] * bulge + disk_colours[b] * disk
        img = convolve(img, psf_kernel.astype(np.float64), mode="reflect")
        images.append(img.astype(np.float32))
    return np.stack(images, axis=0)


def _write_fits(array: np.ndarray, path: Path) -> None:
    hdu = fits.PrimaryHDU(data=array.astype(np.float32))
    hdu.writeto(str(path), overwrite=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: generate the training library and the mock image cubes."""
    try:
        from synthesizer.grid import Grid
        from synthesizer.instruments import FilterCollection
    except ImportError as e:
        raise SystemExit(
            "generate_mock_jwst.py needs the `synthesizer` package to build the training "
            f"library (import failed: {e}). Install it or run in an environment that has it."
        ) from e

    grid_path = GRID_DIR / f"{GRID_NAME}.hdf5"
    if not grid_path.exists():
        raise SystemExit(
            f"BPASS grid not found: {grid_path}\n"
            "Set $SYNTHESIZER_GRID_DIR to the directory holding "
            f"{GRID_NAME}.hdf5 (see the synthesizer grid downloader)."
        )

    print(f"Loading BPASS grid from {grid_path} ...")
    grid = Grid(GRID_NAME, grid_dir=str(GRID_DIR))
    print(
        f"  Ages: {len(grid.log10ages)} bins  "
        f"[{float(np.min(grid.log10ages)):.1f}, {float(np.max(grid.log10ages)):.1f}] log10(yr)"
    )
    print(
        f"  Metallicities: {len(grid.metallicities)} bins  "
        f"[{float(np.min(grid.metallicities)):.1e}, {float(np.max(grid.metallicities)):.1e}]"
    )

    # Use native filter wavelength grids (do NOT pass new_lam=grid.lam --
    # the BPASS grid spans X-ray to radio and confuses filter interpolation)
    fc = FilterCollection(LIBRARY_FILTER_CODES)

    make_training_library(grid, fc)
    make_mock_images()

    print("\nDone. Now run:  python scripts/fit_mock_jwst.py")


if __name__ == "__main__":
    main()
