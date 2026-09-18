#!/usr/bin/env python3
r"""Blind resolved SED fit of a real JWST/NIRCam galaxy: JADES spec-z + DJA imaging.

This is arachne's first end-to-end fit to *real* data.  Nothing is simulated:
the images are 9-band JWST/NIRCam cutouts served by the DAWN JWST Archive
(DJA) thumbnail service, the redshift is a JADES DR4 spectroscopic redshift
held fixed, the PSFs are the empirical JADES DR3 GOODS-S kernels, and the only
external check is the JADES DR3 catalogue's total photometry, which never
enters the fit.

The model
--------
``AdditiveComponentModel`` in *arcsec* mode with *analytic* normalisation::

    I_b(y, x) = Σ_k F_kb(θ_k) · P_k(y, x)        (K emulator calls per likelihood)

``F_kb`` is the ``ParrotEmulatorV2`` SED of component ``k`` at that component's
own total stellar mass; ``P_k`` is a Sérsic (or point-source) surface
brightness profile of unit total flux **over the whole plane**, so the fitted
``log_mass`` is a total mass and not "the mass inside the cutout".  The model
image is PSF-convolved per band (zero-padded, linear) and a per-band sky
pedestal from ``NuisanceModel`` is added.  Redshift is fixed at the JADES
spec-z; every other emulator parameter is free per component, except the ones
made shared by ``--share-dust``.

Masking: what actually limits the fit
-------------------------------------
On the first real target (``goods-s-mediumhst_12281``) the residuals were not
limited by the PSF, the pixel scale or the emulator but by *what is in the
frame*.  A MAP-only scan (15 configurations, ``--map-only``; see the Phase 2b
report) gave ``chi2_red`` with a 5% floor of

====================================================  =========
configuration                                         chi2_red
====================================================  =========
6" frame, neighbour segments masked only                   7.27
8" frame, neighbour segments masked only                   6.70
8" + compact clumps on the galaxy peeled                   2.40
8" + clumps + the target's segment beyond 1.5" masked      1.62
8" + clumps + the target's segment beyond 1.0" masked      1.50
====================================================  =========

while resampling the PSF by exact area rebinning instead of a cubic spline,
broadening it by the 0.03"->0.05" pixel-integration difference (0.0115"), or
changing its assumed pixel scale by +/-3% each moved ``chi2_red`` by less than
1%.  The defaults below follow: an 8" frame, the clump peel on, and the
target's own segment masked beyond 1.5".  Every masking decision is a
*modelling* decision and is recorded in ``summary.json``
(``mask_info``, ``masked_pixel_fraction``: 19.6% for this galaxy).

Why a model-error floor is mandatory here
-----------------------------------------
DJA weight maps carry no source Poisson term, so the quoted per-pixel S/N of
these galaxies reaches several thousand.  A pure photon-noise likelihood would
then demand that a two-component Sersic model reproduce a real, clumpy,
spiral-armed galaxy to one part in a thousand (the fits below sit at
``chi2_red`` ~ 2000 on photon noise alone).  ``--model-err-frac`` adds
``(frac * model)`` in quadrature to the pixel sigma, which is the honest
statement of how well a smooth Sersic model and a neural SPS emulator can
possibly do.  The default is **10%**, near the emulator's own 5-15% accuracy:
on the recommended configuration the same MAP gives ``chi2_red`` = 1.62 (5%),
1.13 (10%) and 1.00 (15%), so 10% is the value at which the model is "almost
but not quite" adequate and 5% would over-weight the brightest pixels.
``--poisson-floor EPS`` additionally inflates the variance by ``EPS * |flux|``
as a crude stand-in for the missing source shot noise (off by default; ``EPS``
is in nJy, i.e. the inverse of an effective gain, and is a guard, not a
calibration).

Sizes and Sersic indices are priors, not bounds
-----------------------------------------------
The blind fit measures a half-light radius from the curve of growth of the
masked multi-band S/N stack (0.247" for 12281, against 0.246" in the JADES
catalogue) and centres the log-size prior there with sd ``--size-prior-sd``
(0.5 in natural log).  ``--max-re-arcsec`` narrows that sd further so the cap
sits 3 sigma above the measurement.  Both are *proper priors*, which
``AdditiveComponentModel.sample_prior`` draws from, so nested sampling's
``logZ`` stays meaningful; a hard wall or a penalty term would not.  Sersic
indices are clamped to ``--n-bounds`` (0.5 to 8 by default, tighter than the
library's 0.3-10).  Together these stop the failure seen in the first K=2 run,
where one "component" became a 14"-wide n=8 pedestal holding 30% of the light.

Blind
-----
The fit never sees the catalogue photometry.  The starting point comes from
image moments (``blind_initial_theta``), masses from a linear least-squares
solve, then multistart Adam over four SPS archetypes, and the posterior from
nested slice sampling whose live points are drawn from the prior.

Blind, continued
----------------
Nothing in the initialisation, the optimisation or the sampling sees the
catalogue: image moments give the centre, the curve of growth gives the size,
a linear least-squares solve gives the masses, multistart Adam over four SPS
archetypes gives the MAP (then an annealed Adam + damped-Newton polish), and
nested slice sampling draws its live points from the prior.  The JADES DR3
total photometry is compared with the answer, never fitted to.

Usage
-----
::

    python examples/fit_jades_dja.py --list-targets
    python examples/fit_jades_dja.py --target goods-s-mediumhst_12281 --quick
    python examples/fit_jades_dja.py --target goods-s-mediumhst_12281 --map-only
    python examples/fit_jades_dja.py --target goods-s-mediumhst_12281 --k 1 2
    python examples/fit_jades_dja.py --target goods-s-mediumhst_12281 --sampler nuts
    python examples/fit_jades_dja.py --target goods-s-mediumhst_12281 --multires
    python examples/fit_jades_dja.py --all-targets --num-live 150
    python examples/fit_jades_dja.py --ra 53.14104 --dec -27.76680 --z 1.8989

``--map-only`` stops after the blind MAP (a minute on a GPU) and writes
``map_summary.json`` plus a chi mosaic: that is the loop to use when choosing
masks, PSFs or profiles.  ``--multires`` fits every band on its **native** DJA
mosaic grid (NIRCam SW at 0.02"/px, LW at 0.04"/px) through
``MultiResolutionForwardModel`` instead of on the server-resampled 0.05"
thumbnail; the sub-mosaics must already be in
``<--mosaic-cache>/assoc_mosaic`` because the GPU nodes have no outbound
network (run once on the login node to populate it).

Outputs land in ``outputs/real_data/<target_id>/K<k>/``:

- ``summary.json``            fit configuration, evidence, chi2, every physical
                              parameter (median/16/84), sizes in arcsec and kpc,
                              nuisance sky levels, runtimes
- ``photometry_comparison.csv``  per band: catalogue total vs model total
- ``posterior.hdf5``          size-ordered equal-weight samples (+ logZ)
- ``map.json``                the blind MAP
- ``figures/*.png``           band mosaic, component images, component SEDs,
                              1-D posteriors, sampler diagnostics

and ``outputs/real_data/<target_id>/model_comparison.txt`` when more than one
``K`` is run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_ARACHNE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ARACHNE / "scripts"))
sys.path.insert(0, str(_ARACHNE / "examples"))

import real_data_utils as rdu  # noqa: E402
from fit_catalogue import PARAM_BOUNDS, SPS_PARAM_NAMES  # noqa: E402

from arachne.data.observation import ObservationCube  # noqa: E402
from arachne.emulator.parrot_emulator_v2 import load_emulator  # noqa: E402
from arachne.forward_model.multires import MultiResolutionForwardModel  # noqa: E402
from arachne.forward_model.nuisance import NuisanceModel  # noqa: E402
from arachne.forward_model.pipeline import ForwardModel  # noqa: E402
from arachne.inference.diagnostics import ess as chain_ess  # noqa: E402
from arachne.inference.diagnostics import split_rhat  # noqa: E402
from arachne.inference.initialisation import (  # noqa: E402
    MAPResult,
    blind_initial_full_theta,
    blind_initial_theta,
    find_map,
    multistart_map,
)
from arachne.inference.model_comparison import ModelComparisonRow, bayes_factor_table  # noqa: E402
from arachne.inference.nss_sampler import NSSResult, NSSSampler  # noqa: E402
from arachne.inference.nuts_sampler import NUTSSampler  # noqa: E402
from arachne.inference.posterior_predictive import (  # noqa: E402
    chi2_reduced,
    chi2_reduced_samples,
    component_image_samples,
    is_multiresolution,
    residual_summary,
)
from arachne.priors.specs import build_component_log_prior, resolve_prior_specs  # noqa: E402
from arachne.spatial.additive import AdditiveComponentModel  # noqa: E402
from arachne.spatial.profiles import SersicProfile  # noqa: E402

# Generic SPS archetypes for the multistart MAP.  They probe the dust/age
# degeneracy from both sides and a metal-poor solution; none is a "truth".
ARCHETYPES = [{}, {"Av": 0.3}, {"Av": 1.5}, {"log10metallicity": -1.0}]

# --share-dust: the three parameters that nine broad/medium bands cannot
# constrain separately for two overlapping components.  The Phase 2 brief names
# them "dust_slope, fesc_lya, log_fagn"; ParrotEmulatorV2 calls its attenuation
# slope "slope" and has no AGN fraction, so the third shared parameter is the
# 2175 A bump amplitude instead.  Sharing them means "one dust law and one
# Lyman-alpha escape fraction for the galaxy", which is what a bulge+disk
# decomposition of nine-band imaging can actually support.
SHARED_DUST_PARAMS = ["slope", "fesc_lya", "dust_bump_amplitude"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", type=str, default=None, help="JADES DR4 id, e.g. goods-s-...")
    p.add_argument("--ra", type=float, default=None, help="RA in degrees (with --dec --z)")
    p.add_argument("--dec", type=float, default=None, help="Dec in degrees (with --ra --z)")
    p.add_argument("--z", type=float, default=None, help="spectroscopic redshift override")
    p.add_argument(
        "--all-targets",
        action="store_true",
        help="loop over the four primary targets (57578, 47870, 12281, 44832)",
    )
    p.add_argument("--outdir", type=Path, default=_ARACHNE / "outputs/real_data")
    p.add_argument("--cache-dir", type=Path, default=rdu.DJA_CACHE)

    p.add_argument("--size-arcsec", type=float, default=8.0, help="full cutout width")
    p.add_argument("--bands", nargs="+", default=rdu.DEFAULT_BANDS)
    p.add_argument("--psf-scale", type=float, default=0.03, help="arcsec/px of the JADES PSFs")
    p.add_argument(
        "--psf-resample",
        choices=["spline", "area"],
        default="spline",
        help="PSF resampler: cubic spline (surface) or exact area rebin (pixel-integrated)",
    )
    p.add_argument(
        "--psf-broaden",
        type=float,
        default=0.0,
        help="extra Gaussian sigma (arcsec) convolved into the PSF; 0.0115 matches the "
        "0.03->0.05 pixel-integration difference",
    )
    p.add_argument("--oversample", type=int, default=3, help="profile sub-samples per pixel side")

    p.add_argument("--k", type=int, nargs="+", default=[1, 2], help="component counts to compare")
    p.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        metavar="NAME",
        help="explicit profile per component (sersic|gaussian|point); overrides --k/--point",
    )
    p.add_argument("--point", action="store_true", help="append a point-source component")
    p.add_argument(
        "--n-bounds",
        type=float,
        nargs=2,
        default=(0.5, 8.0),
        metavar=("N_MIN", "N_MAX"),
        help="Sersic index bounds (smooth clamp)",
    )
    p.add_argument(
        "--size-prior-sd",
        type=float,
        default=0.5,
        help="sd of the Gaussian prior on log(size), natural log",
    )
    p.add_argument(
        "--max-re-arcsec",
        type=float,
        default=None,
        help="soft cap on a component's effective radius; narrows the log-size prior "
        "so the cap sits 3 sigma above the measured half-light radius",
    )
    p.add_argument("--share-dust", action="store_true", help=f"share {SHARED_DUST_PARAMS}")
    p.add_argument("--fix", nargs="*", default=[], metavar="NAME=VALUE", help="extra fixed params")

    p.add_argument(
        "--mask-neighbours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="segment the cutout and mask every source but the central one",
    )
    p.add_argument("--mask-nsigma", type=float, default=1.5, help="detection threshold (sigma)")
    p.add_argument("--mask-npixels", type=int, default=10, help="minimum connected pixels")
    p.add_argument("--mask-dilate", type=int, default=3, help="dilate neighbour segments (px)")
    p.add_argument(
        "--mask-clump-nsigma",
        type=float,
        default=4.0,
        help="high-pass threshold for compact clumps/companions sitting ON the target (0 = off)",
    )
    p.add_argument("--mask-clump-scale", type=float, default=0.5, help="high-pass scale (arcsec)")
    p.add_argument(
        "--mask-plume",
        type=float,
        default=1.5,
        metavar="R_ARCSEC",
        help="mask the target's own segment beyond R arcsec (tidal plumes); 0 = off",
    )
    p.add_argument(
        "--mask-radius",
        type=float,
        default=None,
        metavar="R_ARCSEC",
        help="hard aperture: mask everything beyond R arcsec from the target",
    )

    p.add_argument(
        "--model-err-frac",
        type=float,
        default=0.10,
        help="fractional model-error floor added in quadrature to the pixel sigma",
    )
    p.add_argument("--poisson-floor", type=float, default=0.0, help="var += EPS*|flux| (nJy)")
    p.add_argument("--fit-shifts", action="store_true", help="fit per-band (dy, dx)")
    p.add_argument("--fit-noise-scale", action="store_true", help="fit per-band noise rescaling")
    p.add_argument("--sky-prior-sigma", type=float, default=1.0, help="nJy/px")

    p.add_argument("--sampler", choices=["nss", "nuts"], default="nss")
    p.add_argument("--num-live", type=int, default=300)
    p.add_argument("--num-inner-steps", type=int, default=None, help="absolute NSS slice steps")
    p.add_argument(
        "--inner-steps-factor",
        type=float,
        default=3.0,
        help="NSS slice steps as this multiple of n_params (blackjax wants >= 2)",
    )
    p.add_argument("--termination", type=float, default=1e-3)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-warmup", type=int, default=500, help="NUTS warmup")
    p.add_argument("--n-chains", type=int, default=4, help="NUTS chains")
    p.add_argument(
        "--chain-jitter",
        type=float,
        default=0.3,
        help="NUTS per-chain start jitter, in units of a whitened (Laplace) sigma",
    )
    p.add_argument(
        "--diagonal-metric",
        action="store_true",
        help="adapt a diagonal rather than a dense mass matrix inside the whitened space",
    )
    p.add_argument("--map-only", action="store_true", help="stop after the blind MAP (fast checks)")
    p.add_argument(
        "--polish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="anneal Adam without the mass re-solve, then damped Newton steps, after multistart",
    )
    p.add_argument("--newton-steps", type=int, default=3, help="damped Newton steps after Adam")
    p.add_argument(
        "--whiten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="NUTS in whitened coordinates theta = MAP + L z (L from the Laplace covariance)",
    )
    p.add_argument("--max-doublings", type=int, default=10, help="NUTS max tree doublings")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing NSS checkpoint")
    p.add_argument("--quick", action="store_true", help="smoke run: few live points and steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--float64", action="store_true", help="enable jax_enable_x64 (A100/H100 only)")
    p.add_argument("--list-targets", action="store_true", help="print the vetted targets and exit")
    p.add_argument(
        "--multires",
        action="store_true",
        help="fit every band on its own native DJA mosaic grid (SW 0.02, LW 0.04 arcsec/px)",
    )
    p.add_argument(
        "--multires-bands",
        nargs="+",
        default=None,
        help="band subset for --multires (native mosaics are ~70 MB per band to fetch)",
    )
    p.add_argument(
        "--mosaic-cache",
        type=Path,
        default=rdu.DATA_ROOT,
        help="cache root for the native association mosaics (they land in <root>/assoc_mosaic)",
    )
    return p.parse_args(argv)


def parse_fixed(pairs: list[str]) -> dict[str, float]:
    fixed: dict[str, float] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--fix expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        if name not in SPS_PARAM_NAMES:
            raise ValueError(f"--fix {name!r} is not an emulator parameter")
        fixed[name] = float(value)
    return fixed


# ---------------------------------------------------------------------------
# Dataset: the single place where the observation is constructed
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Dataset:
    """Everything the forward model needs about one galaxy's imaging.

    ``obs``/``psf_model`` describe the single-grid DJA thumbnail, which is
    always built: it is cheap, and it is where the neighbour mask, the image
    moments and the half-light size guess come from.  ``mro`` is additionally
    populated in ``--multires`` mode with the native-resolution bands, and then
    it, not ``obs``, is what the forward model sees.
    """

    target: rdu.Target
    obs: ObservationCube
    bands: list[str]
    psf_model: object
    pixel_scale: float
    cutout_path: Path
    catalogue: dict
    mask_fraction: float = 0.0
    mask_products: tuple | None = None
    mask_info: dict = dataclasses.field(default_factory=dict)
    r_half_arcsec: float = 0.2
    mro: object | None = None


def build_dataset(target: rdu.Target, args) -> Dataset:
    """Fetch the imaging, the PSFs and the catalogue photometry for one galaxy.

    This is the only place the observation is constructed: ``--multires``
    attaches a :class:`MultiResolutionObservation` here and everything
    downstream dispatches on ``Dataset.mro``.
    """
    print(f"\n=== {target.id}  RA={target.ra:.7f} Dec={target.dec:+.7f} z={target.z:.4f} "
          f"({target.source}) ===")  # fmt: skip
    obs, bands, path = rdu.load_dja_observation(
        target, args.size_arcsec, list(args.bands), args.cache_dir
    )
    print(f"  cutout {path.name}: {obs.flux.shape} at {obs.pixel_scale:.5f}\"/px, bands "
          f"{[rdu.short_band(b) for b in bands]}")  # fmt: skip

    flux = np.asarray(obs.flux, dtype=np.float32)
    variance = np.asarray(obs.variance, dtype=np.float32)
    mask = np.asarray(obs.mask, dtype=np.float32)
    if args.poisson_floor > 0:
        variance = variance + np.float32(args.poisson_floor) * np.abs(flux)
        print(f"  Poisson-like floor: var += {args.poisson_floor} * |flux| (crude guard)")
    bad = ~np.isfinite(flux) | ~np.isfinite(variance) | (variance <= 0)
    if bad.any():
        print(f"  masking {int(bad.sum())} non-finite / zero-variance pixels")
        mask = np.where(bad, 0.0, mask)
        flux = np.where(bad, 0.0, flux)
        variance = np.where(bad, 1.0, variance)
    pixel_scale = float(obs.pixel_scale)
    mask_fraction = 0.0
    mask_products = None
    mask_info: dict = {}
    bad_map = np.zeros(flux.shape[1:], dtype=bool)
    if args.mask_neighbours:
        new_mask, segmentation, mask_info = rdu.neighbour_mask(
            flux,
            variance,
            mask,
            nsigma=args.mask_nsigma,
            npixels=args.mask_npixels,
            dilate=args.mask_dilate,
            pixel_scale=pixel_scale,
            clump_nsigma=args.mask_clump_nsigma,
            clump_scale_arcsec=args.mask_clump_scale,
            plume_radius_arcsec=(
                args.mask_plume if args.mask_plume and args.mask_plume > 0 else None
            ),
            aperture_radius_arcsec=args.mask_radius,
        )
        bad_map = (mask[0] > 0) & (new_mask[0] == 0)
        mask = new_mask
        mask_fraction = float(mask_info["frac_masked"])
        print(f"  neighbour mask: {mask_info['n_segments']} segments at >{args.mask_nsigma} sigma; "
              f"masked {100 * mask_fraction:.1f}% of pixels "
              f"(neighbours {100 * mask_info.get('frac_neighbours', 0):.1f}%, "
              f"clumps {100 * mask_info.get('frac_clumps', 0):.1f}%, "
              f"plume {100 * mask_info.get('frac_plume', 0):.1f}%, "
              f"aperture {100 * mask_info.get('frac_aperture', 0):.1f}%)")  # fmt: skip
        mask_products = (
            rdu.detection_stack(flux, variance, np.ones_like(flux)),
            segmentation,
            bad_map,
        )
    obs = ObservationCube(
        flux=flux,
        variance=variance,
        mask=mask,
        band_names=list(bands),
        pixel_scale=pixel_scale,
        wcs=obs.wcs,
    )
    for i, band in enumerate(bands):
        sigma = float(np.sqrt(np.median(variance[i])))
        print(f"    {rdu.short_band(band):6s} peak {flux[i].max():8.1f} sum {flux[i].sum():9.1f}"
              f" sigma_med {sigma:6.3f} nJy/px  peak S/N {flux[i].max() / sigma:7.0f}")  # fmt: skip

    r_half = rdu.half_light_radius_arcsec(flux, variance, mask, pixel_scale)
    print(f'  half-light radius of the masked S/N stack: {r_half:.3f}"')

    psf_model = rdu.build_psf_model(
        bands,
        to_scale=pixel_scale,
        psf_scale=args.psf_scale,
        method=args.psf_resample,
        broaden_arcsec=args.psf_broaden,
    )
    print(f"  PSFs: {rdu.PSF_DIR.name} resampled {args.psf_scale}\" -> "
          f"{pixel_scale:.5f}\"/px by {args.psf_resample}"
          f"{f', broadened by {args.psf_broaden}\"' if args.psf_broaden else ''}, "
          f"kernel {psf_model.kernels.shape[1:]}")  # fmt: skip

    mro = None
    if args.multires:
        mro = build_multires_observation(target, args, bands, bad_map, obs)

    catalogue = rdu.catalogue_photometry(target, bands)
    if catalogue:
        print(f"  JADES DR3 match: UNIQUE_ID {catalogue['dr3_unique_id']} at "
              f"{catalogue['sep_arcsec']:.3f}\"")  # fmt: skip
    return Dataset(
        target,
        obs,
        bands,
        psf_model,
        pixel_scale,
        path,
        catalogue,
        mask_fraction,
        mask_products,
        mask_info,
        r_half,
        mro,
    )


def build_multires_observation(target: rdu.Target, args, bands, bad_map, thumb_obs):
    """Native-resolution bands from the DJA association mosaics, masked like the thumb.

    Each band is cut from its own deepest association sub-mosaic (NIRCam SW at
    0.02"/px, LW at 0.04"/px) and keeps that grid; nothing is resampled.  The
    PSF for each band is the JADES kernel resampled to *that band's* scale.
    The neighbour mask is transferred from the thumbnail grid in sky
    coordinates (:func:`real_data_utils.mirror_mask_to_band`), so every band is
    masked on the same physical region.
    """
    from arachne.data.multires import MultiResolutionObservation, tangent_plane_affine

    want = list(args.multires_bands) if args.multires_bands else list(bands)
    want = [b if "." in b else f"JWST/NIRCam.{b}" for b in want]
    missing = [b for b in want if b not in bands]
    if missing:
        raise ValueError(f"--multires-bands {missing} are not among the fitted bands {bands}")
    native = rdu.native_psf_kernels(want)
    psf_arg = {b: (native[b], args.psf_scale) for b in want}
    t0 = time.perf_counter()
    # The GPU nodes have no outbound network, so use the already-downloaded
    # sub-mosaics when they cover the target and only call the DJA API (which
    # needs the login node) when something is missing.
    flux_paths, weight_paths = rdu.cached_mosaic_paths(
        args.mosaic_cache, want, target.ra, target.dec
    )
    if len(flux_paths) == len(want):
        print(f"  multires: cached native mosaics for {[rdu.short_band(b) for b in want]}")
        for band, path in flux_paths.items():
            print(f"    {rdu.short_band(band):6s} {path.name}")
        mro = MultiResolutionObservation.from_fits(
            flux_paths=flux_paths,
            ref_ra=target.ra,
            ref_dec=target.dec,
            size_arcsec=args.size_arcsec,
            weight_paths=weight_paths or None,
            psfs=psf_arg,
        )
    else:
        from arachne.data.dja import fetch_dja_native_cutout

        print(f"  multires: fetching native mosaics for {[rdu.short_band(b) for b in want]} "
              f"(cache {args.mosaic_cache}; needs the login node's network)")  # fmt: skip
        mro = fetch_dja_native_cutout(
            target.ra,
            target.dec,
            args.size_arcsec,
            want,
            cache_dir=args.mosaic_cache,
            psfs=psf_arg,
            with_weights=True,
        )
    print(f"  multires: {mro.n_bands} bands in {time.perf_counter() - t0:.0f}s")

    # The thumbnail's own BandImage-like description, for the mask transfer.
    affine, ref_pixel = tangent_plane_affine(thumb_obs.wcs, target.ra, target.dec)
    thumb_band = dataclasses.make_dataclass("ThumbGrid", ["affine", "ref_pixel", "shape"])(
        affine, ref_pixel, tuple(np.asarray(thumb_obs.flux).shape[1:])
    )
    bad_maps, psfs = {}, {}
    for band in mro.bands:
        transferred = rdu.mirror_mask_to_band(bad_map, thumb_band, band)
        finite = np.isfinite(np.asarray(band.flux)) & np.isfinite(np.asarray(band.variance))
        bad_maps[band.band_name] = transferred | ~finite
        psfs[band.band_name] = rdu.resample_psf(
            native[band.band_name],
            args.psf_scale,
            float(band.pixel_scale),
            args.psf_resample,
            args.psf_broaden,
        )
        h, w = band.shape
        print(f"    {rdu.short_band(band.band_name):6s} {h:4d}x{w:<4d} at "
              f"{band.pixel_scale:.4f}\"/px, PSF {psfs[band.band_name].shape}, "
              f"{100 * bad_maps[band.band_name].mean():.1f}% masked")  # fmt: skip
    return rdu.apply_mask_to_mro(mro, bad_maps, psfs)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def component_profiles(k_sersic: int, args) -> list:
    """Profile objects for one component count: ``--profiles``, else K Sérsics (+ point)."""
    names = (
        [p.lower() for p in args.profiles]
        if args.profiles
        else ["sersic"] * int(k_sersic) + (["point"] if args.point else [])
    )
    n_min, n_max = float(args.n_bounds[0]), float(args.n_bounds[1])
    out = []
    for name in names:
        if name == "sersic":
            # n_bounds tightened from the library default (0.3, 10): n > 8 on a
            # 6-8" frame is a pedestal, not a galaxy, and n < 0.5 is unphysical.
            out.append(SersicProfile(n_bounds=(n_min, n_max)))
        else:
            out.append(name)
    return out


def build_spatial_model(k_sersic: int, data: Dataset, args, fixed: dict) -> AdditiveComponentModel:
    """K Sérsic components (plus an optional point source) at the fixed spec-z."""
    profiles = component_profiles(k_sersic, args)
    shared = list(SHARED_DUST_PARAMS) if args.share_dust else []
    names = list(SPS_PARAM_NAMES)
    free = [p for p in names if p not in fixed]
    specs = resolve_prior_specs(free, None, PARAM_BOUNDS)
    sps_log_prior = build_component_log_prior(
        names,
        specs,
        PARAM_BOUNDS,
        shared_param_names=shared,
        fixed_param_names=list(fixed),
    )
    h, w = data.obs.flux.shape[1:]
    size_prior = rdu.log_size_prior_from_guess(
        data.r_half_arcsec, sd=args.size_prior_sd, max_re_arcsec=args.max_re_arcsec
    )
    return AdditiveComponentModel(
        n_components=len(profiles),
        emulator_param_names=names,
        param_bounds=PARAM_BOUNDS,
        image_shape=(h, w),
        shared_param_names=shared,
        fixed_params=fixed,
        mass_param="log_mass",
        sps_log_prior=sps_log_prior,
        log_size_prior=size_prior,
        profiles=profiles,
        pixel_scale=data.pixel_scale,
        normalisation="analytic",
        oversample=int(args.oversample),
    )


def build_forward_model(k_sersic: int, data: Dataset, args, emulator, fixed: dict):
    """Assemble the forward model for one component count (single-grid or multi-resolution)."""
    model = build_spatial_model(k_sersic, data, args, fixed)
    if data.mro is not None:
        nuisance = NuisanceModel(
            n_bands=data.mro.n_bands,
            fit_sky=True,
            fit_shifts=args.fit_shifts,
            fit_noise_scale=args.fit_noise_scale,
            sky_prior_sigma=args.sky_prior_sigma,
            # One band's shift is degenerate with the component centres.
            **({"shift_reference_band": 0} if args.fit_shifts else {}),
        )
        return MultiResolutionForwardModel.build(
            observation=data.mro,
            spatial_model=model,
            emulator=emulator,
            model_error_frac=args.model_err_frac,
            nuisance=nuisance,
            pad_psf=True,
            oversample=int(args.oversample),
        )
    nuisance = NuisanceModel(
        n_bands=len(data.bands),
        fit_sky=True,
        fit_shifts=args.fit_shifts,
        fit_noise_scale=args.fit_noise_scale,
        sky_prior_sigma=args.sky_prior_sigma,
        **({"shift_reference_band": 0} if args.fit_shifts else {}),
    )
    return ForwardModel.build(
        obs=data.obs,
        psf_model=data.psf_model,
        spatial_model=model,
        emulator=emulator,
        model_error_frac=args.model_err_frac,
        nuisance=nuisance,
        pad_psf=True,
    )


def order_samples(fm: ForwardModel, samples: jnp.ndarray) -> jnp.ndarray:
    """Break label symmetry on the spatial block, keeping the nuisance block attached.

    ``AdditiveComponentModel.order_components_by_size`` returns a vector of
    length ``spatial_model.n_params``, so it silently drops the nuisance block
    when handed a full theta (see the report notes); slice first, then rejoin.
    """
    n_spatial = fm.spatial_model.n_params

    def one(theta):
        ordered = fm.spatial_model.order_components_by_size(theta[:n_spatial])
        return jnp.concatenate([ordered, theta[n_spatial:]])

    return jax.vmap(one)(jnp.atleast_2d(samples))


# ---------------------------------------------------------------------------
# Blind initialisation + MAP
# ---------------------------------------------------------------------------


POLISH_STAGES = [(0.01, 1000), (0.003, 3000), (0.001, 2500), (0.0003, 2500)]
POLISH_STAGES_QUICK = [(0.003, 400), (0.001, 400)]


def polish_map(fm, res: MAPResult, args) -> MAPResult:
    """Annealed Adam without the mass re-solve, then damped Newton steps.

    ``multistart_map`` stops tens of nats short of the mode on these
    posteriors: the Sersic ``log_n`` and the log-size directions are far
    stiffer than the SPS raws.  Two fixes, both taken from the mock demo:

    * anneal Adam (0.01 -> 3e-4), discarding any stage that makes things worse;
    * turn the *linear mass re-solve* off.  It finds the right basin from a
      blind start, but it maximises the floor-free likelihood and ignores the
      prior, so near the mode it moves the masses the wrong way.

    Then a few modified-Newton steps (eigen-floored Hessian + Armijo
    backtracking) take the point from "nearly stationary" to stationary, which
    also makes the Laplace covariance used by whitened NUTS meaningful.
    """
    stages = POLISH_STAGES_QUICK if args.quick else POLISH_STAGES
    print(f"  annealed polish (no mass re-solve), stages {stages} ...")
    for lr, steps in stages:
        nxt = find_map(
            fm,
            res.theta,
            n_rounds=0,
            final_steps=steps,
            final_lr=lr,
            resolve_masses=False,
            order_by_size=False,
        )
        kept = nxt.neg_log_posterior < res.neg_log_posterior
        print(f"    lr={lr:<7g} {steps:5d} steps -> -log_post = {nxt.neg_log_posterior:.1f}"
              f"{'' if kept else '  (worse; discarded)'}")  # fmt: skip
        if kept:
            res = nxt
    if args.newton_steps > 0:
        res = newton_polish(fm, res, args.newton_steps)
    return res


def newton_polish(fm, res: MAPResult, n_iter: int) -> MAPResult:
    """Modified-Newton steps with an Armijo line search (see ``polish_map``)."""
    loss = jax.jit(lambda t: -fm.log_posterior(t))
    grad = jax.jit(jax.grad(lambda t: -fm.log_posterior(t)))
    hess_at = rdu.hessian_fn(fm)
    theta = jnp.asarray(res.theta)
    best = float(res.neg_log_posterior)
    for it in range(int(n_iter)):
        t0 = time.perf_counter()
        g = np.asarray(grad(theta), np.float64)
        hess = hess_at(theta)
        evals, evecs = np.linalg.eigh(hess)
        floor = max(float(np.abs(evals).max()) * 1e-8, 1.0 / rdu.MAX_LAPLACE_VAR)
        step = -(evecs @ ((evecs.T @ g) / np.maximum(np.abs(evals), floor)))
        slope, scale = float(g @ step), 1.0
        for _ in range(25):
            trial = jnp.asarray(theta + scale * step, theta.dtype)
            value = float(loss(trial))
            if value <= best + 1e-4 * scale * slope:
                break
            scale *= 0.5
        else:
            print("    line search failed; keeping the Adam point")
            break
        print(f"    newton {it + 1}/{n_iter}: -log_post = {value:.2f} (step x{scale:g}, "
              f"{int((evals < 0).sum())} negative-curvature directions) "
              f"[{time.perf_counter() - t0:.0f}s]")  # fmt: skip
        theta, best = trial, value
    return dataclasses.replace(res, theta=theta, neg_log_posterior=best)


def blind_map(fm, args) -> MAPResult:
    """Image moments -> neutral SPS -> linear mass solve -> multistart Adam -> polish."""
    t0 = time.perf_counter()
    model = fm.spatial_model
    if is_multiresolution(fm):
        theta0 = blind_initial_full_theta(fm, ref_band=reference_band(fm))
    else:
        theta0 = fm.initial_theta_from_spatial(blind_initial_theta(model, fm.observation))
    shapes = model.component_shapes(theta0[: model.n_params])
    print(f"  blind start: -log_post = {-float(fm.log_posterior(theta0)):.1f}, centres "
          f"{[np.asarray(s['mu']).round(3).tolist() for s in shapes]}\", sizes "
          f"{[np.asarray(s['sigma']).round(3).tolist() for s in shapes]}\"")  # fmt: skip
    kwargs = dict(n_rounds=2, steps_per_round=150, final_steps=200) if args.quick else {}
    # order_by_size=False: the library helper would truncate the nuisance block.
    res = multistart_map(fm, theta0, ARCHETYPES, order_by_size=False, **kwargs)
    print(f"  multistart best: -log_post = {res.neg_log_posterior:.1f} "
          f"({res.n_steps} Adam steps per start, {time.perf_counter() - t0:.0f}s)")  # fmt: skip
    if args.polish:
        res = polish_map(fm, res, args)
    theta = order_samples(fm, res.theta[None])[0]
    res = dataclasses.replace(res, theta=theta)
    print(f"  blind MAP:   -log_post = {res.neg_log_posterior:.1f} "
          f"[{time.perf_counter() - t0:.0f}s total]")  # fmt: skip
    return res


def reference_band(fm) -> int:
    """Index of F200W among the fitted bands (else the middle band)."""
    names = list(fm.observation.band_names)
    for i, name in enumerate(names):
        if name.endswith("F200W"):
            return i
    return len(names) // 2


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def run_nss(fm: ForwardModel, args, outdir: Path, key) -> NSSResult:
    num_live = 60 if args.quick else args.num_live
    n_out = 200 if args.quick else args.n_samples
    inner = args.num_inner_steps or (
        max(5, fm.n_params) if args.quick else max(5, int(args.inner_steps_factor * fm.n_params))
    )
    print(f"\n--- NSS: num_live={num_live}, num_inner_steps={inner}, "
          f"termination={args.termination:g}, n_samples_out={n_out} ---")  # fmt: skip
    sampler = NSSSampler(
        fm,
        num_live=num_live,
        num_inner_steps=inner,
        termination=args.termination,
        n_samples_out=n_out,
    )
    t0 = time.perf_counter()
    result = sampler.run(
        key,
        checkpoint_path=outdir / f"nss_ckpt_L{num_live}_I{inner}",
        checkpoint_every=25,
        resume=not args.no_resume,
    )
    print(f"  logZ = {result.logZ:.2f} +/- {result.logZ_err:.2f}  ESS = {result.ess:.0f}  "
          f"steps = {result.n_steps}  dead = {result.n_dead}  "
          f"[{time.perf_counter() - t0:.0f}s]")  # fmt: skip
    return result


def run_nuts(fm, theta_map, args, key):
    """NUTS from the blind MAP, by default in Laplace-whitened coordinates.

    An identity/diagonal metric is hopeless on these posteriors: the curvature
    eigenvalues at the mode span ~1e-1 to ~4e7 and the stiff directions are
    correlated, so warmup drives the step size to 1e-7 and every trajectory
    saturates the tree cap.  With ``theta = MAP + L z`` (``L`` the Cholesky
    factor of the eigen-clipped Laplace covariance) the target is roughly
    ``N(0, I)``.  R-hat and ESS are re-derived in *theta* afterwards, which is
    what the report is about.

    ``arachne.inference.laplace`` now provides this; the local copy in
    ``real_data_utils`` is only the fallback for an older checkout.
    """
    n_warmup = 100 if args.quick else args.n_warmup
    n_samples = 200 if args.quick else args.n_samples
    n_chains = 2 if args.quick else args.n_chains
    print(f"\n--- NUTS from the blind MAP: {n_chains} chains, {n_warmup} warmup + "
          f"{n_samples} samples, whiten={args.whiten} ---")  # fmt: skip
    t0 = time.perf_counter()
    laplace = rdu.import_laplace() if args.whiten else None
    if laplace is not None:
        # The library module (added in Phase 2b) also takes |eigenvalue| rather
        # than clipping negative-curvature directions, and adapts a dense metric
        # inside the whitened space; both matter on these posteriors.
        print("  whitening with arachne.inference.laplace.run_whitened_nuts "
              "(|lambda| eigenvalues, dense metric in z)")  # fmt: skip
        result, _ = laplace.run_whitened_nuts(
            fm,
            theta_map,
            key,
            n_warmup=n_warmup,
            n_samples=n_samples,
            n_chains=n_chains,
            chain_jitter=args.chain_jitter,
            max_num_doublings=args.max_doublings,
            dense_mass_matrix=not args.diagonal_metric,
        )
        print(result.summary(max_num_doublings=args.max_doublings))
        print(f"  [{time.perf_counter() - t0:.0f}s]")
        return result
    target, theta_init, whitened, laplace_info = fm, theta_map, None, {}
    if args.whiten:
        _, chol, laplace_info = rdu.laplace_covariance(rdu.hessian_fn(fm)(theta_map))
        print(f"  Laplace at the MAP: curvature [{laplace_info['curvature_min']:.3g}, "
              f"{laplace_info['curvature_max']:.3g}], {laplace_info['n_floored']} floored; "
              f"marginal sd [{laplace_info['marginal_sd_min']:.3g}, "
              f"{laplace_info['marginal_sd_max']:.3g}] raw units")  # fmt: skip
        whitened = rdu.WhitenedModel(fm, theta_map, chol)
        target = whitened
        theta_init = jnp.zeros(whitened.n_params, dtype=jnp.asarray(theta_map).dtype)
    sampler = NUTSSampler(
        target, n_warmup=n_warmup, n_samples=n_samples, max_num_doublings=args.max_doublings
    )
    result = sampler.run(theta_init, key, n_chains=n_chains, chain_jitter=args.chain_jitter)
    if whitened is not None:
        chains = result.chains
        diag = dict(result.diagnostics)
        if chains is not None:
            chains = jax.vmap(whitened.to_theta)(chains)
            diag["rhat"] = np.asarray(split_rhat(chains))
            diag["ess"] = np.asarray(chain_ess(chains))
        diag["laplace"] = laplace_info
        result = dataclasses.replace(
            result,
            samples=whitened.to_theta(result.samples),
            chains=chains,
            diagnostics=diag,
        )
    print(result.summary(max_num_doublings=args.max_doublings))
    print(f"  [{time.perf_counter() - t0:.0f}s]")
    return result


# ---------------------------------------------------------------------------
# Decoding the posterior into physical quantities
# ---------------------------------------------------------------------------


def decode_samples(fm: ForwardModel, samples: jnp.ndarray, emulator) -> dict:
    """Physical per-component parameters, SEDs and in-frame flux fractions."""
    model = fm.spatial_model
    n_spatial = model.n_params
    free_cols = jnp.array([model.emulator_param_names.index(p) for p in model.sps_param_names])
    shared_cols = (
        jnp.array([model.emulator_param_names.index(p) for p in model.shared_param_names])
        if model.n_shared
        else None
    )

    def one(theta):
        spatial = theta[:n_spatial]
        mu, sigma, rho, sps_full = model.component_params(spatial)
        shapes = model.component_shapes(spatial)
        n_sersic = jnp.stack(
            [jnp.asarray(s["n"], dtype=jnp.float32) if "n" in s else jnp.float32(np.nan)
             for s in shapes]
        )  # fmt: skip
        seds = model.component_seds(spatial, emulator)  # (K, N_bands)
        frac = jnp.sum(model.profiles(spatial), axis=(1, 2))  # in-frame flux fraction
        out = {
            "mu": mu,
            "sigma": sigma,
            "rho": rho,
            "sps": sps_full[:, free_cols],
            "n_sersic": n_sersic,
            "seds": seds,
            "in_frame_frac": frac,
        }
        if shared_cols is not None:
            out["shared"] = sps_full[0, shared_cols]
        if fm.nuisance is not None and fm.nuisance.n_params:
            out["sky"] = fm.nuisance.split(theta[n_spatial:])["sky"]
        return out

    chunk = 64
    samples = jnp.atleast_2d(samples)
    pieces = [
        jax.jit(jax.vmap(one))(samples[i : i + chunk]) for i in range(0, samples.shape[0], chunk)
    ]
    return {k: np.asarray(jnp.concatenate([p[k] for p in pieces])) for k in pieces[0]}


def quantiles(values: np.ndarray) -> dict:
    q16, q50, q84 = np.percentile(np.asarray(values, dtype=float), [16, 50, 84])
    return {"median": float(q50), "p16": float(q16), "p84": float(q84)}


def kpc_per_arcsec(z: float) -> float:
    from astropy.cosmology import Planck18

    return float(1.0 / Planck18.arcsec_per_kpc_proper(z).value)


def observed_bands(fm) -> tuple[list, list, list]:
    """``(flux, variance, mask)`` as per-band numpy arrays, for either model class.

    The multi-resolution model's bands have different shapes, so everything
    downstream of the fit works on *lists* of 2-D arrays rather than on one
    (N_bands, H, W) cube; on the single-grid path the list is just the cube's
    first axis and nothing else changes.
    """
    obs = fm.observation
    if is_multiresolution(fm):
        return (
            [np.asarray(b.flux) for b in obs.bands],
            [np.asarray(b.variance) for b in obs.bands],
            [np.asarray(b.mask) for b in obs.bands],
        )
    return (
        list(np.asarray(obs.flux)),
        list(np.asarray(obs.variance)),
        list(np.asarray(obs.mask)),
    )


def model_image_list(fm, theta) -> list:
    """Per-band model images (PSF-convolved, sky added) for either model class."""
    images = (
        fm.model_images(jnp.asarray(theta))
        if is_multiresolution(fm)
        else fm._model_image(jnp.asarray(theta))
    )
    return [np.asarray(im) for im in images]


def effective_chi(fm, model_images: list, frac: float) -> list:
    """(data - model) / sqrt(variance + (frac*model)^2) per band, zero where masked."""
    flux, variance, mask = observed_bands(fm)
    out = []
    for f, v, m, model in zip(flux, variance, mask, model_images):
        sigma = np.sqrt(np.asarray(v, dtype=np.float64) + (frac * model) ** 2)
        out.append(np.where(m > 0, (f - model) / sigma, 0.0))
    return out


def chi_stats(chi: list, mask: list, n_params: int) -> dict:
    """Total / per-band reduced chi-squared and the |chi| > 3 fraction."""
    valid = [np.asarray(m) > 0 for m in mask]
    n_valid = np.array([int(v.sum()) for v in valid])
    chi2_band = np.array([float((c**2).sum()) for c in chi])
    dof = max(int(n_valid.sum()) - int(n_params), 1)
    gt3 = sum(int((np.abs(c) > 3).sum()) for c in chi)
    return {
        "chi2_red": float(chi2_band.sum() / dof),
        "chi2_red_per_band": chi2_band / np.maximum(n_valid, 1),
        "frac_chi_gt_3": float(gt3 / max(int(n_valid.sum()), 1)),
        "n_valid": int(n_valid.sum()),
        "dof": dof,
    }


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


def nss_equal_weight_samples(result, n_out: int, rng_key) -> jnp.ndarray:
    """Re-derive equal-weight samples from the Monte-Carlo-mean nested-sampling weights.

    ``NSSSampler`` returns ``blackjax.ns.utils.sample(...)``, which resamples
    using a *single* shrinkage realisation of the prior volumes.  For a
    likelihood as sharp as a 1.3e5-pixel image fit that realisation is wildly
    degenerate: 1000 draws collapse onto ~16 distinct particles (one repeated
    ~940 times) even though the Kish ESS of the 100-draw-averaged weights is
    ~250, so every posterior interval comes out as a delta function.  The
    averaged weights are already stored in ``NSSResult.log_weights``; a
    low-variance systematic resample from those recovers ~ESS distinct
    particles.  ``logZ`` and ``logZ_err`` are unaffected — they are computed
    from the averaged weights.

    Falls back to ``result.samples`` when the particle positions are not
    available (e.g. a result loaded from HDF5).
    """
    particles = getattr(getattr(result, "infos", None), "particles", None)
    position = getattr(particles, "position", None)
    if position is None or result.log_weights is None:
        return jnp.asarray(result.samples)
    log_w = np.asarray(result.log_weights, dtype=np.float64).ravel()
    pos = np.asarray(position, dtype=np.float32).reshape(log_w.size, -1)
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    offset = float(jax.random.uniform(rng_key))
    idx = np.searchsorted(cdf, (np.arange(n_out) + offset) / n_out)
    out = pos[np.clip(idx, 0, log_w.size - 1)]
    print(f"  resampled {n_out} equal-weight draws from {log_w.size} particles: "
          f"{len(np.unique(out, axis=0))} distinct (blackjax's own resample gave "
          f"{len(np.unique(np.asarray(result.samples), axis=0))})")  # fmt: skip
    return jnp.asarray(out)


def map_report(fm, data: Dataset, args, emulator, map_res: MAPResult, outdir: Path) -> dict:
    """Cheap MAP-only diagnostics: chi2 with and without the floor, and total photometry.

    Used by ``--map-only`` for configuration scans (PSF scale, oversampling,
    profile choices) where a full posterior would cost hours.
    """
    model_images = model_image_list(fm, map_res.theta)
    flux, variance, mask = observed_bands(fm)
    chi_eff = effective_chi(fm, model_images, args.model_err_frac)
    chi_phot = [
        np.where(m > 0, (f - mi) / np.sqrt(np.asarray(v, dtype=np.float64)), 0.0)
        for f, v, m, mi in zip(flux, variance, mask, model_images)
    ]
    stats_eff = chi_stats(chi_eff, mask, fm.n_params)
    stats_phot = chi_stats(chi_phot, mask, fm.n_params)
    floors = {}
    for trial in (0.05, 0.10, 0.15):
        floors[f"{trial:.2f}"] = chi_stats(
            effective_chi(fm, model_images, trial), mask, fm.n_params
        )["chi2_red"]
    theta_spatial = map_res.theta[: fm.spatial_model.n_params]
    seds = np.asarray(fm.spatial_model.component_seds(theta_spatial, emulator))
    total = seds.sum(axis=0)
    shapes = fm.spatial_model.component_shapes(theta_spatial)
    cat = data.catalogue.get("flux", {})
    ratios = [
        float(total[b] / cat[band][0])
        for b, band in enumerate(data.bands)
        if band in cat and cat[band][0] > 0
    ]
    mass_col = fm.spatial_model.sps_param_names.index("log_mass")
    _, _, _, sps = fm.spatial_model.component_params(theta_spatial)
    names = fm.spatial_model.emulator_param_names
    free_cols = [names.index(p) for p in fm.spatial_model.sps_param_names]
    log_mass = np.asarray(sps)[:, free_cols][:, mass_col]
    sky = {}
    if fm.nuisance is not None and fm.nuisance.n_params:
        blocks = fm.nuisance.split(map_res.theta[fm.spatial_model.n_params :])
        sky = {b: float(v) for b, v in zip(data.bands, np.asarray(blocks["sky"]))}
    out = {
        "target": data.target.id,
        "n_params": int(fm.n_params),
        "map_neg_log_posterior": float(map_res.neg_log_posterior),
        "chi2_red_photon": stats_phot["chi2_red"],
        "chi2_red_with_model_floor": stats_eff["chi2_red"],
        "chi2_red_vs_floor": floors,
        "chi2_red_with_floor_per_band": {
            b: float(v) for b, v in zip(data.bands, stats_eff["chi2_red_per_band"])
        },
        "frac_chi_gt_3": stats_eff["frac_chi_gt_3"],
        "masked_pixel_fraction": float(data.mask_fraction),
        "mask_info": {k: float(v) for k, v in data.mask_info.items()},
        "r_half_guess_arcsec": float(data.r_half_arcsec),
        "components": [
            {
                "profile": p.name,
                "log_mass": float(log_mass[k]),
                "r_eff_arcsec": float(np.sqrt(np.prod(np.asarray(s["sigma"])))),
                "sersic_n": float(s["n"]) if "n" in s else None,
                "mu_arcsec": np.asarray(s["mu"]).round(4).tolist(),
            }
            for k, (p, s) in enumerate(zip(fm.spatial_model.profile_objects, shapes))
        ],
        "log_mass_total": float(np.log10(np.sum(10.0**log_mass))),
        "sky_nJy_per_pixel": sky,
        "model_total_nJy": {b: float(v) for b, v in zip(data.bands, total)},
        "photometry_ratio_median": float(np.median(ratios)) if ratios else None,
        "photometry_ratio_range": [float(min(ratios)), float(max(ratios))] if ratios else None,
        "config": {
            "psf_scale": args.psf_scale,
            "psf_resample": args.psf_resample,
            "psf_broaden": args.psf_broaden,
            "size_arcsec": args.size_arcsec,
            "oversample": args.oversample,
            "profiles": [p.name for p in fm.spatial_model.profile_objects],
            "n_bounds": list(args.n_bounds),
            "max_re_arcsec": args.max_re_arcsec,
            "size_prior": list(fm.spatial_model.log_size_prior),
            "model_err_frac": args.model_err_frac,
            "poisson_floor": args.poisson_floor,
            "mask_neighbours": bool(args.mask_neighbours),
            "mask_plume": args.mask_plume,
            "mask_radius": args.mask_radius,
            "mask_clump_nsigma": args.mask_clump_nsigma,
            "multires": bool(args.multires),
        },
    }
    with open(outdir / "map_summary.json", "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"  MAP-only: chi2_red(floor) = {out['chi2_red_with_model_floor']:.3f}  "
          f"photon = {out['chi2_red_photon']:.1f}  "
          f"frac|chi|>3 = {out['frac_chi_gt_3']:.4f}  "
          f"phot ratio median = {out['photometry_ratio_median']}")  # fmt: skip
    print("   chi2_red vs floor: " + "  ".join(f"{k} -> {v:.3f}" for k, v in floors.items()))
    per_band = out["chi2_red_with_floor_per_band"]
    print("   per band: " + "  ".join(f"{rdu.short_band(b)} {v:.2f}" for b, v in per_band.items()))
    for k, comp in enumerate(out["components"]):
        print(f"   component {k} ({comp['profile']}): log M {comp['log_mass']:.2f}, "
              f"r_e {comp['r_eff_arcsec']:.3f}\", n {comp['sersic_n']}")  # fmt: skip
    if sky:
        print(f"   sky {min(sky.values()):+.4f} .. {max(sky.values()):+.4f} nJy/px")
    if args.mask_neighbours and data.mask_products is not None:
        chi_png = outdir / "map_chi_mosaic.png"
        plt = rdu.setup_mpl()
        safe_figure(
            rdu.plot_band_mosaic,
            plt,
            chi_png,
            data.bands,
            flux,
            model_images,
            chi_eff,
            f"{data.target.id} MAP-only  chi2_red(floor)={stats_eff['chi2_red']:.2f}",
        )
        print(f"   saved {chi_png}")
    return out


def safe_figure(fn, *fn_args, **fn_kwargs):
    """Run a plotting call, reporting rather than raising: a long GPU run must not die here."""
    import traceback

    try:
        return fn(*fn_args, **fn_kwargs)
    except Exception:  # noqa: BLE001 - figures are never worth losing a multi-hour fit
        print(f"  WARNING: figure {getattr(fn, '__name__', fn)} failed:")
        traceback.print_exc()
        return None


def make_products(
    fm: ForwardModel,
    data: Dataset,
    args,
    emulator,
    result,
    map_res: MAPResult,
    k_sersic: int,
    outdir: Path,
    runtimes: dict,
) -> dict:
    """Write summary.json, the photometry CSV, the posterior and all figures."""
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    model = fm.spatial_model
    bands = data.bands
    raw_samples = jnp.asarray(result.samples)
    if isinstance(result, NSSResult):
        raw_samples = nss_equal_weight_samples(
            result, int(raw_samples.shape[0]), jax.random.PRNGKey(args.seed + 7)
        )
    samples = order_samples(fm, raw_samples)
    result_ordered = dataclasses.replace(result, samples=samples)
    result_ordered.to_hdf5(outdir / "posterior.hdf5")

    print("\n--- Posterior predictive ---")
    resid = residual_summary(fm, samples, n_max=100)
    median_model = [np.asarray(im) for im in resid["median_model"]]
    obs_flux, _, obs_mask = observed_bands(fm)
    chi_eff = effective_chi(fm, median_model, args.model_err_frac)
    stats_eff = chi_stats(chi_eff, obs_mask, fm.n_params)
    chi2_eff_band = stats_eff["chi2_red_per_band"]
    chi2_eff = stats_eff["chi2_red"]
    print(f"  chi2_red (photon noise only) = {resid['chi2_red']:.2f}   "
          f"with the {args.model_err_frac:.0%} floor = {chi2_eff:.2f}   "
          f"frac |chi|>3 = {stats_eff['frac_chi_gt_3']:.4f}")  # fmt: skip
    for b, band in enumerate(bands):
        print(f"    {rdu.short_band(band):6s} chi2_red photon {resid['chi2_red_per_band'][b]:9.2f}"
              f"   with floor {chi2_eff_band[b]:7.3f}")  # fmt: skip

    dec = decode_samples(fm, samples, emulator)
    kpc = kpc_per_arcsec(data.target.z)
    K = model.n_components

    # Total stellar mass: components add in linear mass.
    mass_col = model.sps_param_names.index("log_mass")
    log_mass_tot = np.log10(np.sum(10.0 ** dec["sps"][:, :, mass_col], axis=1))

    components = []
    for k in range(K):
        entry = {
            "profile": model.profile_objects[k].name,
            "mu_y_arcsec": quantiles(dec["mu"][:, k, 0]),
            "mu_x_arcsec": quantiles(dec["mu"][:, k, 1]),
            "sigma_y_arcsec": quantiles(dec["sigma"][:, k, 0]),
            "sigma_x_arcsec": quantiles(dec["sigma"][:, k, 1]),
            "rho": quantiles(dec["rho"][:, k]),
            "r_eff_arcsec": quantiles(np.sqrt(dec["sigma"][:, k, 0] * dec["sigma"][:, k, 1])),
            "r_eff_kpc": quantiles(np.sqrt(dec["sigma"][:, k, 0] * dec["sigma"][:, k, 1]) * kpc),
            "axis_ratio": quantiles(
                np.minimum(dec["sigma"][:, k, 0], dec["sigma"][:, k, 1])
                / np.maximum(dec["sigma"][:, k, 0], dec["sigma"][:, k, 1])
            ),
            "in_frame_flux_fraction": quantiles(dec["in_frame_frac"][:, k]),
        }
        if np.isfinite(dec["n_sersic"][:, k]).all():
            entry["sersic_n"] = quantiles(dec["n_sersic"][:, k])
        for j, name in enumerate(model.sps_param_names):
            entry[name] = quantiles(dec["sps"][:, k, j])
        components.append(entry)

    shared = {}
    for j, name in enumerate(model.shared_param_names):
        shared[name] = quantiles(dec["shared"][:, j])

    sky = {}
    if "sky" in dec:
        for b, band in enumerate(bands):
            sky[band] = quantiles(dec["sky"][:, b])

    # Photometry comparison: the model total is the whole-plane SED sum.
    total_sed = dec["seds"].sum(axis=1)  # (n, N_bands)
    in_frame = (dec["seds"] * dec["in_frame_frac"][:, :, None]).sum(axis=1)
    data_sum = np.array([float(np.asarray(f).sum()) for f in obs_flux])
    cat = data.catalogue
    rows, ratios = [], []
    for b, band in enumerate(bands):
        cf, ce = cat.get("flux", {}).get(band, (np.nan, np.nan))
        p16, p50, p84 = np.percentile(total_sed[:, b], [16, 50, 84])
        f50 = float(np.median(in_frame[:, b]))
        rows.append(
            {
                "band": rdu.short_band(band),
                "pivot_um": rdu.PIVOT_UM[rdu.short_band(band)],
                "catalogue_nJy": round(float(cf), 2),
                "catalogue_err_nJy": round(float(ce), 2),
                "model_total_p16": round(float(p16), 2),
                "model_total_p50": round(float(p50), 2),
                "model_total_p84": round(float(p84), 2),
                "model_in_frame_p50": round(f50, 2),
                "data_frame_sum": round(float(data_sum[b]), 2),
                "ratio_total_over_cat": round(float(p50 / cf), 4) if np.isfinite(cf) else "",
                "ratio_in_frame_over_cat": round(float(f50 / cf), 4) if np.isfinite(cf) else "",
            }
        )
        if np.isfinite(cf) and cf > 0:
            ratios.append(float(p50 / cf))
    rdu.write_photometry_csv(outdir / "photometry_comparison.csv", rows)
    print("\n--- Model vs JADES DR3 total photometry ---")
    for row in rows:
        print(f"    {row['band']:6s} cat {row['catalogue_nJy']:9.1f}  model_total "
              f"{row['model_total_p50']:9.1f}  in_frame {row['model_in_frame_p50']:9.1f}  "
              f"ratio {row['ratio_total_over_cat']}")  # fmt: skip
    if ratios:
        print(f"  ratio(model total / catalogue): median {np.median(ratios):.3f}, "
              f"range [{min(ratios):.3f}, {max(ratios):.3f}]")  # fmt: skip

    summary = {
        "target": dataclasses.asdict(data.target),
        "cutout": str(data.cutout_path),
        "bands": bands,
        "image_shape": [list(np.asarray(f).shape) for f in obs_flux],
        "pixel_scale_arcsec": data.pixel_scale,
        "kpc_per_arcsec": kpc,
        "config": {
            "k_sersic": int(k_sersic),
            "n_components": int(K),
            "profiles": [p.name for p in model.profile_objects],
            "point_source": bool(args.point),
            "shared_params": list(model.shared_param_names),
            "fixed_params": {k: float(v) for k, v in model.fixed_params.items()},
            "model_err_frac": args.model_err_frac,
            "poisson_floor": args.poisson_floor,
            "oversample": args.oversample,
            "psf_scale_arcsec": args.psf_scale,
            "size_arcsec": args.size_arcsec,
            "mask_neighbours": bool(args.mask_neighbours),
            "mask_nsigma": args.mask_nsigma,
            "mask_dilate": args.mask_dilate,
            "mask_clump_nsigma": args.mask_clump_nsigma,
            "mask_plume": args.mask_plume,
            "mask_radius": args.mask_radius,
            "psf_resample": args.psf_resample,
            "psf_broaden": args.psf_broaden,
            "n_bounds": list(args.n_bounds),
            "max_re_arcsec": args.max_re_arcsec,
            "log_size_prior": list(model.log_size_prior),
            "r_half_guess_arcsec": float(data.r_half_arcsec),
            "multires": bool(args.multires),
            "sampler": args.sampler,
            "inner_steps_factor": args.inner_steps_factor,
            "num_live": args.num_live,
            "quick": bool(args.quick),
            "seed": args.seed,
        },
        "n_params": int(fm.n_params),
        "n_params_spatial": int(model.n_params),
        "n_params_nuisance": int(fm.nuisance.n_params) if fm.nuisance else 0,
        "map_neg_log_posterior": float(map_res.neg_log_posterior),
        "chi2_red_photon": float(resid["chi2_red"]),
        "chi2_red_with_model_floor": chi2_eff,
        "chi2_red_photon_per_band": {
            b: float(v) for b, v in zip(bands, resid["chi2_red_per_band"])
        },
        "chi2_red_with_floor_per_band": {b: float(v) for b, v in zip(bands, chi2_eff_band)},
        "frac_chi_gt_3": float(stats_eff["frac_chi_gt_3"]),
        "frac_chi_gt_3_photon": float(resid["frac_chi_gt_3"]),
        "masked_pixel_fraction": float(data.mask_fraction),
        "mask_info": {k: float(v) for k, v in data.mask_info.items()},
        "n_data": int(resid["n_data"]),
        "dof": int(resid["dof"]),
        "log_mass_total": quantiles(log_mass_tot),
        "components": components,
        "shared": shared,
        "sky_nJy_per_pixel": sky,
        "photometry": rows,
        "photometry_ratio_median": float(np.median(ratios)) if ratios else None,
        "runtimes_s": runtimes,
    }
    if isinstance(result, NSSResult):
        summary["logZ"] = float(result.logZ)
        summary["logZ_err"] = float(result.logZ_err)
        summary["ess"] = float(result.ess)
        summary["n_nss_steps"] = int(result.n_steps)
    else:
        diag = result.diagnostics or {}
        summary["ess"] = float(np.min(np.asarray(diag.get("ess", [np.nan]))))
        summary["rhat_max"] = float(np.max(np.asarray(diag.get("rhat", [np.nan]))))
        summary["n_divergent"] = int(diag.get("n_divergent", 0))
    with open(outdir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(outdir / "map.json", "w") as handle:
        json.dump(
            {
                "theta_map": np.asarray(map_res.theta).tolist(),
                "neg_log_posterior": float(map_res.neg_log_posterior),
                "n_steps_per_start": int(map_res.n_steps),
                "archetypes": ARCHETYPES,
                "chi2_red_map": chi2_reduced(fm, map_res.theta),
            },
            handle,
            indent=2,
        )

    print("\n--- Figures ---")
    plt = rdu.setup_mpl()
    paths = [
        safe_figure(
            rdu.plot_band_mosaic,
            plt,
            figdir / "band_mosaic.png",
            bands,
            obs_flux,
            median_model,
            chi_eff,
            f"{data.target.id}  z={data.target.z:.4f}  K={K}  chi2_red(floor)={chi2_eff:.2f}",
        )
    ]
    if K >= 2:
        comp = component_image_samples(fm, samples, n_max=40)
        if is_multiresolution(fm):
            # Per band: (n, K, H_b, W_b) -> list of (K, H_b, W_b) medians, then
            # index [k][b] like the single-grid (K, N_bands, H, W) array.
            per_band = [np.median(np.asarray(c), axis=0) for c in comp]
            comp_median = [[per_band[b][k] for b in range(len(bands))] for k in range(K)]
            centres_px = [
                [
                    tuple(
                        float(v)
                        for v in fm.sky_to_pixel(
                            b,
                            float(np.median(dec["mu"][:, k, 0])),
                            float(np.median(dec["mu"][:, k, 1])),
                        )
                    )
                    for b in range(len(bands))
                ]
                for k in range(K)
            ]
        else:
            comp_median = np.median(np.asarray(comp), axis=0)
            h, w = median_model[0].shape
            centres_px = [
                [
                    (
                        float(np.median(dec["mu"][:, k, 0])) / data.pixel_scale + (h - 1) / 2,
                        float(np.median(dec["mu"][:, k, 1])) / data.pixel_scale + (w - 1) / 2,
                    )
                ]
                * len(bands)
                for k in range(K)
            ]
        show = [b for b in ("JWST/NIRCam.F200W", "JWST/NIRCam.F444W") if b in bands]
        paths.append(
            safe_figure(
                rdu.plot_component_images,
                plt,
                figdir / "component_images.png",
                bands,
                comp_median,
                centres_px,
                show,
            )
        )
    paths.append(
        safe_figure(
            rdu.plot_component_seds,
            plt,
            figdir / "component_seds.png",
            bands,
            dec["seds"],
            total_sed,
            cat,
            data_sum,
        )
    )
    panels = []
    for k in range(K):
        panels.append((f"c{k} log_mass", dec["sps"][:, k, mass_col]))
        panels.append((f"c{k} r_e ['']", np.sqrt(dec["sigma"][:, k, 0] * dec["sigma"][:, k, 1])))
        if np.isfinite(dec["n_sersic"][:, k]).all():
            panels.append((f"c{k} sersic n", dec["n_sersic"][:, k]))
        for j, name in enumerate(model.sps_param_names):
            if name == "log_mass":
                continue
            panels.append((f"c{k} {name}", dec["sps"][:, k, j]))
    for j, name in enumerate(model.shared_param_names):
        panels.append((f"shared {name}", dec["shared"][:, j]))
    panels.append(("log M_total", log_mass_tot))
    paths.append(
        safe_figure(
            rdu.plot_posterior_1d,
            plt,
            figdir / "posterior_summary_1d.png",
            panels,
            f"{data.target.id}: posterior (median and 16-84%), K={K}",
        )
    )
    if isinstance(result, NSSResult):
        paths.append(
            safe_figure(rdu.plot_nss_diagnostics, plt, figdir / "sampler_diagnostics.png", result)
        )
    else:
        names = [f"theta[{i}]" for i in range(fm.n_params)]
        paths.append(
            safe_figure(
                rdu.plot_nuts_diagnostics, plt, figdir / "sampler_diagnostics.png", result, names
            )
        )
    for path in paths:
        if path is not None:
            print(f"  saved {path}")

    print("\n--- Headline numbers ---")
    mtot = summary["log_mass_total"]
    print(f"  log M_star (total)   {mtot['median']:.2f} "
          f"[{mtot['p16']:.2f}, {mtot['p84']:.2f}]")  # fmt: skip
    for k, entry in enumerate(components):
        line = (f"  component {k} ({entry['profile']}): log M "
                f"{entry['log_mass']['median']:.2f}, r_e "
                f"{entry['r_eff_arcsec']['median']:.3f}\" = "
                f"{entry['r_eff_kpc']['median']:.2f} kpc, "
                f"Av {entry['Av']['median']:.2f}")  # fmt: skip
        if "sersic_n" in entry:
            line += f", n {entry['sersic_n']['median']:.2f}"
        print(line)
    if sky:
        levels = [v["median"] for v in sky.values()]
        print(f"  sky pedestals: {min(levels):+.4f} to {max(levels):+.4f} nJy/px "
              f"(peak data {max(float(np.max(f)) for f in obs_flux):.0f} nJy/px)")  # fmt: skip
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def fit_target(target: rdu.Target, args, emulator) -> None:
    fixed = {"redshift": float(target.z)}
    fixed.update(parse_fixed(args.fix))
    data = build_dataset(target, args)
    base = args.outdir / target.id
    base.mkdir(parents=True, exist_ok=True)
    if data.mask_products is not None:
        snr, segmentation, bad = data.mask_products
        rdu.save_mask_products(
            base / "neighbour_mask.png",
            base / "neighbour_mask.fits",
            snr,
            segmentation,
            bad,
            target.id,
        )
        print(f"  saved {base / 'neighbour_mask.png'}")
    rows: list[ModelComparisonRow] = []

    for k_sersic in args.k:
        outdir = base / (f"K{k_sersic}" + ("_quick" if args.quick else ""))
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}\n{target.id}: K = {k_sersic}"
              f"{' + point source' if args.point else ''}\n{'=' * 70}")  # fmt: skip
        t_start = time.perf_counter()
        fm = build_forward_model(k_sersic, data, args, emulator, fixed)
        print(f"  {fm.spatial_model.n_params} spatial + "
              f"{fm.nuisance.n_params} nuisance = {fm.n_params} free parameters")  # fmt: skip

        t0 = time.perf_counter()
        map_res = blind_map(fm, args)
        t_map = time.perf_counter() - t0
        if args.map_only:
            map_report(fm, data, args, emulator, map_res, outdir)
            print(f"  K={k_sersic} MAP-only finished in {t_map:.0f}s -> {outdir}")
            continue

        key = jax.random.PRNGKey(args.seed + 1000 * k_sersic)
        t0 = time.perf_counter()
        if args.sampler == "nss":
            result = run_nss(fm, args, outdir, key)
        else:
            result = run_nuts(fm, map_res.theta, args, key)
        t_sample = time.perf_counter() - t0

        summary = make_products(
            fm,
            data,
            args,
            emulator,
            result,
            map_res,
            k_sersic,
            outdir,
            {
                "map": round(t_map, 1),
                "sampling": round(t_sample, 1),
                "total": round(time.perf_counter() - t_start, 1),
            },
        )
        if isinstance(result, NSSResult):
            rows.append(
                ModelComparisonRow(
                    n_components=int(fm.spatial_model.n_components),
                    logZ=float(result.logZ),
                    logZ_err=float(result.logZ_err),
                    ess=float(result.ess),
                    n_params=int(fm.n_params),
                    chi2_red_map=chi2_reduced(fm, map_res.theta),
                    chi2_red_median=float(
                        np.median(chi2_reduced_samples(fm, jnp.asarray(result.samples), n_max=64))
                    ),
                    runtime_s=t_sample,
                    result=result,
                )
            )
        print(f"  K={k_sersic} finished in {summary['runtimes_s']['total']:.0f}s -> {outdir}")

    if len(rows) > 1:
        table = bayes_factor_table(rows)
        print(f"\n{table}")
        (base / "model_comparison.txt").write_text(table + "\n")
        best = max(rows, key=lambda r: r.logZ)
        payload = [
            {
                "n_components": r.n_components,
                "logZ": r.logZ,
                "logZ_err": r.logZ_err,
                "ess": r.ess,
                "n_params": r.n_params,
                "chi2_red_map": r.chi2_red_map,
                "chi2_red_median": r.chi2_red_median,
                "runtime_s": r.runtime_s,
                "ln_bayes_factor_vs_best": best.logZ - r.logZ,
            }
            for r in rows
        ]
        (base / "model_comparison.json").write_text(json.dumps(payload, indent=2))


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.list_targets:
        print(f"{'id':32s} {'RA':>11s} {'Dec':>11s} {'z_spec':>7s}")
        for tid, (ra, dec, z) in rdu.VETTED_TARGETS.items():
            flag = "*" if tid in rdu.DEFAULT_TARGET_ORDER else " "
            print(f"{flag}{tid:31s} {ra:11.6f} {dec:+11.6f} {z:7.4f}")
        print("* = part of --all-targets")
        return
    if args.float64:
        jax.config.update("jax_enable_x64", True)
        print("  float64 enabled (use only on A100/H100-class cards)")
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    if jax.default_backend() == "cpu":
        print("  WARNING: running on CPU; this script is meant for a GPU node.")

    if args.all_targets:
        targets = [rdu.resolve_target(tid, None, None, None) for tid in rdu.DEFAULT_TARGET_ORDER]
    else:
        targets = [rdu.resolve_target(args.target, args.ra, args.dec, args.z)]

    if args.multires and args.multires_bands:
        # One band list everywhere: the emulator, the thumbnail (which supplies
        # the mask and the size guess), the catalogue comparison and the figures
        # must all describe the same bands as the multi-resolution observation.
        args.bands = [b if "." in b else f"JWST/NIRCam.{b}" for b in args.multires_bands]
        print(f"  --multires-bands restricts the whole fit to "
              f"{[rdu.short_band(b) for b in args.bands]}")  # fmt: skip

    print(f"Loading emulator: {rdu.CHECKPOINT}")
    inner = load_emulator(rdu.CHECKPOINT)
    bands = [b for b in args.bands if b in inner.band_names]
    dropped = [b for b in args.bands if b not in inner.band_names]
    have_psf = [b for b in bands if (rdu.PSF_DIR / f"{rdu.short_band(b)}_psf_norm.fits").exists()]
    dropped += [b for b in bands if b not in have_psf]
    if dropped:
        print(f"  dropping bands missing from the emulator or the PSF directory: {dropped}")
    args.bands = have_psf
    emulator = rdu.BandSubsetEmulator(inner, args.bands)
    assert list(emulator.param_names) == SPS_PARAM_NAMES
    print(f"  {len(emulator.param_names)} parameters, {len(emulator.band_names)} bands used")

    t0 = time.perf_counter()
    for target in targets:
        fit_target(target, args, emulator)
    print(f"\nAll done in {time.perf_counter() - t0:.0f}s. Outputs under {args.outdir}")


if __name__ == "__main__":
    main()
