"""Blind recovery of SPS parameter maps with CorrelatedFieldMap (simplest single-grid case).

A mock galaxy is rendered on one 0.03"/px grid in eight NIRCam bands, convolved
with per-band PSFs and given white Gaussian noise, then fitted blind.  There
are no registration shifts, no sky pedestals and no multi-resolution grids.

Truth (``--truth``):
  prior   maps drawn from the CorrelatedFieldMap prior itself around an
          exponential disk: the fit model is exactly right (self-consistency).
  spiral  a face-on two-armed spiral with clumps, dust lanes and young stars
          on the arms, and a radial metallicity gradient: the prior is *not*
          the generating process (model-misspecification test).

Mapped parameters: log_mass (per pixel), Av, log10metallicity and the SFH (``--sfh``):
  basis2       (default) two latent maps ``sfh_a``, ``sfh_b``; all five logsfr_ratio_i are
               derived per pixel as ``sum_k B[k, i] sfh_k``, with ``B`` the top two right-singular
               vectors of the fit emulator's Jacobian d ln F / d ratio at a neutral reference
               (``sfh_basis``): the photometry constrains ~2 SFH directions per pixel, so mapping
               more leaves a null space the Laplace step fills confidently.
  map03        ratios 0 and 3 mapped, 1/2/4 free globals.
  fixed_truth  map03 with ratios 1/2/4 held at their true values.
  map_all      every ratio mapped.
  merge0       map03 with ratio_0 pinned at 0.
redshift = 2 is fixed; fesc_lya and dust_bump_amplitude are fixed at neutral values (they
change the fluxes by < 0.3 % per unit at z = 2 in these bands; ``--free-globals`` frees them);
the dust slope is a free global.  The truth (spiral or prior draw) always maps mass, A_V, Z and
ratios 0 and 3 (every ratio for map_all), with the other globals at TRUTH_GLOBALS, so the mocks
are identical across SFH setups; fit and truth are compared by emulator-parameter name.

Start (no truth used): log-mass mean map from the smoothed reddest band at
neutral SPS values (``mass_map_from_image``).  ``--mass-start template`` adds a colour-based
per-pixel template correction (``template_mass_map``: dust / recent-SFH grid; the neutral
template biases young clumps).  With ``--mass-start wiener`` the
mass-field coefficients start from a ridge projection
(``project_to_field``) of a multi-band Wiener deconvolution
(``wiener_mass_map``) -- detail goes into the coefficients, which the prior
penalises, never into the mean map, which it does not.  (On the toy spiral the
Wiener start gave no measurable gain, so ``image`` is the default.)  Other means at neutral
constants; zero levels.  Then Adam MAP (``find_map``; with ``--block-newton``,
block-coordinate descent: Adam on the field coefficients alternating with exact Newton on the
galaxy-level block of levels and globals, ``block_newton``), and a posterior
(``--posterior``): ``laplace`` (alias
``--laplace``: trust-region Newton to the mode plus a Laplace posterior: Hessian, Gaussian map
samples, coverage of the truth, information maps and the Laplace evidence) or ``geovi`` /
``mgvi`` (``arachne.inference.mgvi``, from the MAP), optionally followed by MCMC (``--nuts``).
``--n-starts`` keeps the start with the lowest ``-log p`` among
converged modes with a positive-definite Hessian (never the Laplace evidence, which favours
unconverged modes).  ``--k-max-mtf T`` truncates every field's Fourier basis at the
wavenumber where the sharpest PSF's MTF drops below ``T`` (``psf_k_max``).

Emulator and PSFs:
  --emulator CKPT   a ParrotEmulatorV2/V1 checkpoint (``load_emulator``);
                    default is a toy mass-linear SED, good for CPU tests only.
  --psf-dir DIR     directory with ``{FILTER}_psf_norm.fits`` (JADES empirical
                    PSFs, 0.03"/px, used as is); default Gaussian PSFs with
                    approximate NIRCam FWHMs.

Outputs in ``--outdir``: ``metrics.json``, ``maps.npz``, ``figures/`` and a results bundle
``results.h5`` (``arachne.results``: data, MAP and thinned posterior draws of the model cubes,
parameter maps and scalars, truth and regions; open it with ``arachne.load_bundle``, check it
with ``python -m arachne.results info results.h5``).  ``--no-bundle`` skips it.

Example:
    python examples/demo_correlated_field.py --truth prior --quick
    python examples/demo_correlated_field.py --truth spiral --emulator emu.eqx --psf-dir psfs/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from arachne import (  # noqa: E402
    CorrelatedFieldMap,
    ForwardModel,
    MAPResult,
    ObservationCube,
    PSFModel,
    SPSEmulator,
    default_sfh_dust_templates,
    find_map,
    mass_map_from_image,
    psf_k_max,
    template_mass_map,
    wiener_mass_map,
)
from arachne.results import Region, TruthContent, write_content  # noqa: E402
from arachne.results.writer import content_from_forward_model  # noqa: E402

PIXEL_SCALE = 0.03
REDSHIFT = 2.0
# NIRCam pivot wavelengths (micron) for every band with both an emulator output and a JADES
# DR3 PSF.  Gaussian-PSF FWHMs (toy runs only) follow ~0.032"/micron, floored at 0.030".
NIRCAM_PIVOT_UM = {
    "F070W": 0.704, "F090W": 0.902, "F115W": 1.154, "F150W": 1.501, "F162M": 1.627,
    "F182M": 1.845, "F200W": 1.988, "F210M": 2.096, "F250M": 2.503, "F277W": 2.776,
    "F300M": 2.996, "F335M": 3.365, "F356W": 3.564, "F410M": 4.092, "F430M": 4.280,
    "F444W": 4.402, "F460M": 4.630, "F480M": 4.817,
}  # fmt: skip
BAND_SETS = {
    # the original 8: seven wide bands + F410M
    "wide8": ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F410M", "F444W"],
    # wide8 plus every other medium band with a PSF (F140M/F360M have none): 17 bands
    "medium": ["F090W", "F115W", "F150W", "F162M", "F182M", "F200W", "F210M", "F250M", "F277W",
               "F300M", "F335M", "F356W", "F410M", "F430M", "F444W", "F460M", "F480M"],
}  # fmt: skip


def set_bands(name):
    """Select the band set (module globals used by the emulators, PSFs and plots)."""
    global BANDS, PIVOT_UM, FWHM_ARCSEC
    filters = sorted(BAND_SETS[name], key=NIRCAM_PIVOT_UM.get)
    BANDS = [f"JWST/NIRCam.{f}" for f in filters]
    PIVOT_UM = np.array([NIRCAM_PIVOT_UM[f] for f in filters])
    FWHM_ARCSEC = np.maximum(0.030, 0.032 * PIVOT_UM)


set_bands("wide8")
SPS_PARAM_NAMES = [
    "redshift", "log_mass", "slope", "fesc_lya", "dust_bump_amplitude", "log10metallicity",
    "Av", "logsfr_ratio_0", "logsfr_ratio_1", "logsfr_ratio_2", "logsfr_ratio_3",
    "logsfr_ratio_4",
]  # fmt: skip
PARAM_BOUNDS = {
    "slope": (-0.3, 1.1), "fesc_lya": (0.0, 1.0), "dust_bump_amplitude": (0.0, 5.0),
    "log10metallicity": (-4.0, -1.39), "Av": (0.001, 5.0),
    **{f"logsfr_ratio_{i}": (-10.0, 10.0) for i in range(5)},
}  # fmt: skip
# Neutral values: the prior centres used for the blind start (never the truth).
NEUTRAL = {"Av": 0.5, "log10metallicity": -2.5, **{f"logsfr_ratio_{i}": 0.0 for i in range(5)}}
NEUTRAL_GLOBALS = {
    "slope": 0.0, "fesc_lya": 0.2, "dust_bump_amplitude": 1.0,
    **{f"logsfr_ratio_{i}": 0.0 for i in range(5)},
}  # fmt: skip
# "sfh" setup: true values of the non-mapped parameters, deliberately away from NEUTRAL_GLOBALS
# so the fit has to find them.
TRUTH_GLOBALS = {
    "slope": 0.25, "fesc_lya": 0.5, "dust_bump_amplitude": 2.0,
    "logsfr_ratio_1": 0.3, "logsfr_ratio_2": -0.3, "logsfr_ratio_4": 0.2,
}  # fmt: skip
# "--honest" mocks: truth means offset from NEUTRAL (the fit's prior centres and start), and
# for the prior-draw truth, hyperparameters different from the fit's.
HONEST_OFFSETS = {"Av": 0.4, "log10metallicity": 0.4, "logsfr_ratio_0": -0.4, "logsfr_ratio_3": 0.5}
HONEST_CORR_SCALE = 1.6
HONEST_AMP_SCALE = 1.4
# SFH-ratio priors (Student-t, 2 dof).  "continuity" is the library default
# (arachne.priors.specs.DEFAULT_PRIORS, scale 0.3); "bursty" is bursty continuity (scale 1.0,
# Tacchella et al. 2022).  Free global ratios get the density directly; mapped ratios get it as
# their per-pixel marginal through a Gaussian copula (CorrelatedFieldMap field_marginals).
# "legacy": continuity on the globals, Gaussian maps (the runs before 2026-10-30).
SFH_PRIOR_SCALES = {"bursty": 1.0, "continuity": 0.3, "legacy": 0.3}
# The default is continuity: under bursty the heavy tail lets a whole map's level run to the
# bound (spiral ratio_0 -> -9, chi2_red 1.028 vs 0.996; runs/2026-10-30_optimizer).  The
# production emulator was trained on bursty continuity (arsinh-input std 1.10), so both priors
# stay inside its training domain.
SFH_PRIOR = "continuity"
# Prior hyperparameters: break scale in px, rms in parameter units.
CORR_LENGTHS = {
    "log_mass": 3.0, "Av": 5.0, "log10metallicity": 12.0, "logsfr_ratio_0": 4.0,
    "logsfr_ratio_1": 4.0, "logsfr_ratio_2": 5.0, "logsfr_ratio_3": 6.0, "logsfr_ratio_4": 8.0,
}  # fmt: skip
AMPLITUDES = {
    "log_mass": 0.25, "Av": 0.3, "log10metallicity": 0.15,
    **{f"logsfr_ratio_{i}": 0.5 for i in range(5)},
}  # fmt: skip
LEVEL_SIGMAS = {
    "log_mass": 1.0, "Av": 1.0, "log10metallicity": 0.5,
    **{f"logsfr_ratio_{i}": 1.0 for i in range(5)},
}  # fmt: skip
LABELS = {
    "log_mass": r"$\log M_*$ / px",
    "Av": r"$A_V$",
    "log10metallicity": r"$\log Z$",
    "logsfr_ratio_0": r"logsfr_ratio_0 (0-3 / 3-10 Myr)",
    "logsfr_ratio_3": r"logsfr_ratio_3 (64-411 Myr / 0.4-2.6 Gyr)",
    "logsfr_ratio_1": r"logsfr_ratio_1 (3-10 / 10-64 Myr)",
    "logsfr_ratio_2": r"logsfr_ratio_2 (10-64 / 64-411 Myr)",
    "logsfr_ratio_4": r"logsfr_ratio_4 (0.4-2.6 / 2.6-3.1 Gyr)",
    "sfh_a": r"SFH basis a (latent)",
    "sfh_b": r"SFH basis b (latent)",
}


SHORT = {"log_mass": "M", "Av": "Av", "log10metallicity": "Z", "sfh_a": "a", "sfh_b": "b",
         **{f"logsfr_ratio_{i}": f"r{i}" for i in range(5)}}  # fmt: skip
RATIOS = [f"logsfr_ratio_{i}" for i in range(5)]
# basis2: latent SFH maps (bounds, prior hyperparameters; Gaussian fields, mean 0).
LATENT_SFH = {"sfh_a": (-10.0, 10.0), "sfh_b": (-10.0, 10.0)}
CORR_LENGTHS.update({"sfh_a": 4.0, "sfh_b": 6.0})
AMPLITUDES.update({"sfh_a": 0.5, "sfh_b": 0.5})
LEVEL_SIGMAS.update({"sfh_a": 1.0, "sfh_b": 1.0})
# Globals with < 0.3 % flux change per unit at z = 2 in these bands: fixed unless
# --free-globals (fit at NEUTRAL_GLOBALS, truth at TRUTH_GLOBALS).
WEAK_GLOBALS = ("fesc_lya", "dust_bump_amplitude")


# FIELDS: the fit's fields (set_sfh).  TRUTH_FIELDS: the truth's maps (mass, A_V, Z, the
# 0-3/3-10 Myr ratio and the 64-411 Myr/0.4-2.6 Gyr ratio; every ratio for map_all), the same
# for every fit setup so the mocks are comparable.  The truth uses TRUTH_GLOBALS; the fit
# starts from NEUTRAL_GLOBALS.  COMPARE_FIELDS (set in main from the fit model's
# param_map_names): the emulator-parameter maps compared with the truth, by name.
FIELDS = ["log_mass", "Av", "log10metallicity", "sfh_a", "sfh_b"]
TRUTH_FIELDS = ["log_mass", "Av", "log10metallicity", "logsfr_ratio_0", "logsfr_ratio_3"]
COMPARE_FIELDS = ["log_mass", "Av", "log10metallicity", *RATIOS]
FIXED = {"redshift": REDSHIFT, **{k: NEUTRAL_GLOBALS[k] for k in WEAK_GLOBALS}}
FIXED_TRUTH = {"redshift": REDSHIFT, **TRUTH_GLOBALS}
SFH_SETUPS = ("basis2", "map03", "fixed_truth", "map_all", "merge0")
PINNED = ()  # fit fields held at their mean (tiny prior amplitude); see set_sfh
SFH_BASIS = None  # (2, 5) basis2 weights B[k, i] (sfh_basis on the fit emulator; set in main)


def set_sfh(name, free_globals=False):
    """SFH setup (module globals FIELDS, TRUTH_FIELDS, FIXED, FIXED_TRUTH, PINNED).

    ``basis2``: latent maps ``sfh_a``/``sfh_b`` with every ratio derived through SFH_BASIS
    (set in main by :func:`sfh_basis`).  ``map03``: ratios 0 and 3 mapped, 1/2/4 free
    globals.  ``fixed_truth``: as map03 but the
    fit holds ratios 1/2/4 at their true values (diagnostic for the bin-trading degeneracy).
    ``map_all``: every ratio mapped; the truths of 1/2/4 are fields around TRUTH_GLOBALS
    (constant maps for the spiral).  ``merge0``: as map03, but the fit pins ratio_0 at 0
    (equal SFR in 0-3 and 3-10 Myr, i.e. one 0-10 Myr bin) while the truth keeps its map:
    a deliberately coarser SFH than the truth, testing whether removing that bin's freedom
    stops the bin trading.

    WEAK_GLOBALS are fixed in the fit at NEUTRAL_GLOBALS unless ``free_globals``; the truth
    always holds every non-mapped parameter at TRUTH_GLOBALS.
    """
    global FIELDS, TRUTH_FIELDS, FIXED, FIXED_TRUTH, PINNED
    PINNED = ("logsfr_ratio_0",) if name == "merge0" else ()
    ratios = [0, 1, 2, 3, 4] if name == "map_all" else [0, 3]
    base = ["log_mass", "Av", "log10metallicity"]
    TRUTH_FIELDS = base + [f"logsfr_ratio_{i}" for i in ratios]
    FIELDS = base + list(LATENT_SFH) if name == "basis2" else list(TRUTH_FIELDS)
    FIXED = {"redshift": REDSHIFT}
    if not free_globals:
        FIXED.update({k: NEUTRAL_GLOBALS[k] for k in WEAK_GLOBALS})
    if name == "fixed_truth":
        FIXED.update({k: v for k, v in TRUTH_GLOBALS.items() if k.startswith("logsfr_ratio")})
    FIXED_TRUTH = {"redshift": REDSHIFT,
                   **{k: v for k, v in TRUTH_GLOBALS.items() if k not in TRUTH_FIELDS}}  # fmt: skip


def sfh_basis(emulator, n=2):
    """Top-``n`` right-singular vectors of d ln F_b / d logsfr_ratio_i at a neutral point.

    The Jacobian is taken at redshift REDSHIFT, log_mass 7, NEUTRAL + NEUTRAL_GLOBALS with
    every ratio 0 (never the truth), on ``emulator`` (the fit's), and decomposed in numpy
    float64.  Each basis vector's largest-|component| entry is made positive.  Directions the
    photometry does not constrain (``s_k < 1e-6 s_0``, e.g. the toy emulator, which only
    uses ratio_0) are replaced by unit vectors orthogonal to the earlier ones, so the basis is
    always orthonormal.

    Returns:
        ``(B, s)``: ``B`` (n, 5) with ``ratio_i = sum_k B[k, i] sfh_k``, and the top-``n``
        singular values ``s`` (per unit ratio, in ln flux).
    """
    names = list(emulator.param_names)
    ref = {**NEUTRAL_GLOBALS, **NEUTRAL, "redshift": REDSHIFT, "log_mass": 7.0}
    row = jnp.asarray([float(ref[p]) for p in names])
    idx = jnp.asarray([names.index(r) for r in RATIOS])

    def log_flux(r):
        return jnp.log(emulator.predict(row.at[idx].set(r)[None, :])[0])

    jac = np.asarray(jax.jacobian(log_flux)(jnp.zeros(len(RATIOS), row.dtype)), np.float64)
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    s = np.concatenate([s, np.zeros(max(0, n - s.size))])[:n]
    basis = []
    for k in range(n):
        v = vt[k] if k < vt.shape[0] and s[k] >= 1e-6 * s[0] else None
        if v is None:
            # Any unit vector orthogonal to the earlier ones: Gram-Schmidt on the axes, those
            # least represented in the earlier vectors first.
            weight = np.abs(np.array(basis)).sum(0) if basis else np.zeros(len(RATIOS))
            for e in np.eye(len(RATIOS))[np.argsort(weight, kind="stable")]:
                w = e - sum((np.dot(e, b) * b for b in basis), np.zeros(len(RATIOS)))
                if np.linalg.norm(w) > 1e-3:
                    v = w
                    break
        v = v / np.linalg.norm(v)
        basis.append(v * np.sign(v[np.argmax(np.abs(v))]))
    return np.stack(basis), s


# ---------------------------------------------------------------------------
# Emulators and PSFs
# ---------------------------------------------------------------------------


class ToySEDEmulator(SPSEmulator, eqx.Module):
    """Mass-linear toy SED over the 12 production inputs (uses z, mass, Z, Av, ratio_0)."""

    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)

    @property
    def param_names(self):
        """Input names (the production 12)."""
        return self._param_names

    @property
    def band_names(self):
        """NIRCam band names."""
        return self._band_names

    def predict(self, params):
        """(N, 12) -> (N, 8) nJy: power-law continuum, Balmer break, young boost, dust."""
        lam = jnp.asarray(PIVOT_UM)[None, :]
        z = params[:, 0:1]
        logm = params[:, 1:2]
        logz = params[:, 5:6]
        av = params[:, 6:7]
        r0 = params[:, 7:8]
        rest = lam / (1.0 + z)
        beta = 0.4 + 0.5 * (logz + 2.5)
        young = jax.nn.sigmoid(r0)  # more recent SF -> weaker break, bluer UV
        brk = jnp.where(rest < 0.40, 0.35 + 0.65 * young, 1.0)
        dust = 10.0 ** (-0.4 * av * (rest / 0.55) ** -1.0)
        return 2.0e-6 * 10.0**logm * lam**beta * brk * dust


class BandSubsetEmulator(SPSEmulator, eqx.Module):
    """Restrict a production emulator's output to ``BANDS``, in that order."""

    inner: SPSEmulator
    _band_names: list[str] = eqx.field(static=True)
    _idx: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, inner, bands):
        """Wrap ``inner`` and select ``bands``."""
        self.inner = inner
        self._band_names = list(bands)
        self._idx = tuple(inner.band_names.index(b) for b in bands)

    @property
    def param_names(self):
        """Inner emulator inputs."""
        return self.inner.param_names

    @property
    def band_names(self):
        """Selected bands."""
        return self._band_names

    def predict(self, params):
        """Inner prediction, selected bands."""
        return self.inner.predict(params)[:, jnp.array(self._idx)]


def load_emulator_and_psf(args) -> tuple[SPSEmulator, PSFModel]:
    """Production emulator / JADES PSFs if given, else the toy SED and Gaussian PSFs."""
    if args.emulator:
        from arachne import load_emulator

        emulator = BandSubsetEmulator(load_emulator(args.emulator), BANDS)
        assert list(emulator.param_names) == SPS_PARAM_NAMES, emulator.param_names
    else:
        emulator = ToySEDEmulator(_param_names=SPS_PARAM_NAMES, _band_names=BANDS)
    if args.psf_dir:
        paths = {b: Path(args.psf_dir) / f"{b.split('.')[-1]}_psf_norm.fits" for b in BANDS}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing PSFs: {missing}")
        psf = PSFModel.from_fits(paths)  # JADES empirical PSFs are already 0.03"/px
    else:
        y, x = np.mgrid[-15:16, -15:16]
        ks = []
        for fwhm in FWHM_ARCSEC / PIXEL_SCALE:
            s = fwhm / 2.355
            k = np.exp(-(x**2 + y**2) / (2 * s**2))
            ks.append(k / k.sum())
        psf = PSFModel(kernels=np.stack(ks).astype(np.float32), band_names=BANDS)
    return emulator, psf


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------


def disk_mass_map(n, log_mtot=10.0, rd_px=7.0):
    """Exponential disk, log10 mass per pixel, total ``10**log_mtot``."""
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2.0
    r = np.hypot(yy, xx)
    sigma0 = 10.0**log_mtot / (2 * np.pi * rd_px**2)
    return np.log10(sigma0 * np.exp(-r / rd_px)).astype(np.float32)


def spiral_truth(n, rng):
    """Face-on spiral: maps (len(TRUTH_FIELDS), n, n) in TRUTH_FIELDS order, arm weight (n, n).

    Mapped ratios other than 0 and 3 are constant at their TRUTH_GLOBALS values.
    """
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2.0
    r, phi = np.hypot(yy, xx), np.arctan2(yy, xx)
    pitch = np.deg2rad(18.0)
    arm = np.clip(np.cos(2 * (phi - np.log(r + 2.0) / np.tan(pitch))), 0, None) ** 2
    win = np.clip((r - 3) / 5, 0, 1) * np.exp(-r / (0.5 * n))
    rd = 0.11 * n
    sig = np.exp(-r / rd) * (1 + 1.2 * win * arm) + 0.8 * np.exp(-((r / (0.05 * n)) ** 2))
    clumps = np.zeros_like(r)
    for _ in range(8):
        r0 = rng.uniform(0.15, 0.4) * n
        ph = np.log(r0 + 2) / np.tan(pitch) + rng.choice([0, np.pi])
        cy, cx = r0 * np.sin(ph), r0 * np.cos(ph)
        clumps += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5**2))
    sig = sig + 0.15 * clumps * np.exp(-r / (0.4 * n))
    sigma0 = 10.0**10.0 / (2 * np.pi * rd**2)
    logm = np.log10(sigma0 * sig / (1 + 1.2 * 0.25 + 0.0))
    av = 0.3 + 0.9 * win * arm + 0.4 * np.exp(-r / rd)
    logz = -2.2 - 0.6 * (r / (0.5 * n))
    r0 = -0.8 + 2.0 * win * arm + 1.5 * np.clip(clumps, 0, 1)
    # Age structure: old bulge and inter-arm (less SF 64-411 Myr than 0.4-2.6 Gyr ago),
    # younger arms and clumps.
    r3 = -0.8 + 1.3 * win * arm + 1.0 * np.clip(clumps, 0, 1) - 0.5 * np.exp(-r / (0.06 * n))
    maps = {"log_mass": logm, "Av": av, "log10metallicity": logz, "logsfr_ratio_0": r0,
            "logsfr_ratio_3": r3}  # fmt: skip
    stack = [maps[k] if k in maps else np.full_like(r, TRUTH_GLOBALS[k]) for k in TRUTH_FIELDS]
    return np.stack(stack).astype(np.float32), win * arm


def build_model(n, mass_mean, mean_values=None, corr_scale=1.0, nu=1.5, fixed=None,
                hyper=False, couple=False, amp_scale=1.0, sfh_prior=None,
                pinned=None, fields=None, k_max=None):  # fmt: skip
    """CorrelatedFieldMap with the demo's hyperparameters.

    ``fields`` defaults to FIELDS (the fit); pass TRUTH_FIELDS for the truth model.  Latent
    SFH fields (``sfh_a``/``sfh_b``) make every ratio a derived parameter through SFH_BASIS.
    ``corr_scale`` multiplies every correlation length; ``nu`` is the Matérn slope of the
    non-mass fields (mass keeps 1.5); ``amp_scale`` multiplies every amplitude.  ``fixed``
    defaults to FIXED (the fit); pass FIXED_TRUTH for the truth model.  ``hyper`` fits
    amplitude and correlation length of every non-mass field (log-normal priors around the
    defaults); ``couple`` fits A_V's
    response to the mass and SFH deviations (N(0, 0.5^2) each).  ``sfh_prior``
    (default SFH_PRIOR) sets the SFH-ratio priors; the truth model passes ``"legacy"`` so
    the mocks do not depend on the fit's prior.  ``pinned`` (default PINNED; the truth
    passes ``()``) fields get amplitude and level sigma 1e-3 around a zero mean: held at 0.
    ``k_max`` (rad/px) truncates every field's basis: the mass field keeps every mode below
    it (more detail for mass), the others also keep the usual 0.99 variance cut.
    """
    fields = FIELDS if fields is None else list(fields)
    pinned = PINNED if pinned is None else pinned
    fixed = FIXED if fixed is None else fixed
    sfh_prior = SFH_PRIOR if sfh_prior is None else sfh_prior
    spec = {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": SFH_PRIOR_SCALES[sfh_prior]}
    ratios = [p for p in SPS_PARAM_NAMES if p.startswith("logsfr_ratio")]
    marginals = None
    if sfh_prior != "legacy":
        marginals = {p: spec for p in ratios if p in fields and p not in pinned}
    latent = {k: v for k, v in LATENT_SFH.items() if k in fields}
    derived = None
    if latent:
        if SFH_BASIS is None or len(SFH_BASIS) != len(latent):
            raise ValueError("latent SFH fields need SFH_BASIS (sfh_basis) for each of them")
        derived = {r: {"offset": 0.0,
                       "weights": {k: float(SFH_BASIS[j, i]) for j, k in enumerate(latent)}}
                   for i, r in enumerate(RATIOS)}  # fmt: skip
    means = {"log_mass": mass_mean}
    means.update({k: v for k, v in (mean_values or NEUTRAL).items() if k in fields})
    means.update({k: 0.0 for k in latent})
    means.update({k: 0.0 for k in pinned})
    free_globals = [p for p in SPS_PARAM_NAMES
                    if p not in fields and p not in fixed and p not in (derived or {})]  # fmt: skip
    ells = {k: corr_scale * v for k, v in CORR_LENGTHS.items() if k in fields}
    kmax_kw = {"variance_fraction": 0.99}
    if k_max is not None:
        kmax_kw = {
            "k_max": {k: float(k_max) for k in fields},
            "variance_fraction": {k: 1.0 if k == "log_mass" else 0.99 for k in fields},
        }
    hyper_priors = None
    if hyper:
        hyper_priors = {
            k: {
                "log_amplitude": {"loc": float(np.log(AMPLITUDES[k])), "scale": 0.5},
                "log_corr_length": {"loc": float(np.log(ells[k])), "scale": 0.25},
            }
            for k in fields
            if k != "log_mass"
        }
    free_couplings = None
    if couple:
        sources = ["log_mass"] + [k for k in fields if k.startswith(("logsfr_ratio", "sfh_"))]
        # A_V must come after its sources: reorder fields so it is last.
        free_couplings = {"Av": {s: 0.5 for s in sources}}
    field_order = [k for k in fields if k != "Av"] + ["Av"] if couple else fields
    return CorrelatedFieldMap(
        image_shape=(n, n),
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        field_params=field_order,
        mass_param="log_mass",
        fixed_params=fixed,
        corr_lengths=ells,
        nus={k: (1.5 if k == "log_mass" else nu) for k in fields},
        amplitudes={
            k: 1e-3 if k in pinned else amp_scale * v for k, v in AMPLITUDES.items() if k in fields
        },
        level_sigmas={
            k: 1e-3 if k in pinned else v for k, v in LEVEL_SIGMAS.items() if k in fields
        },
        mean_maps=means,
        field_marginals=marginals,
        pixel_scale=PIXEL_SCALE,
        hyper_priors=hyper_priors,
        free_couplings=free_couplings,
        global_priors={p: spec for p in ratios if p in free_globals},
        latent_fields=latent or None,
        derived_params=derived,
        **kmax_kw,
    )


def fit_param_maps(model, thetas):
    """Emulator-parameter maps of ``thetas`` ((P,) or (S, P)) by name: ``{name: (…, H, W)}``.

    Mapped emulator parameters and derived ones (``model.param_maps``); latent fields are
    not emulator inputs and are left out.
    """
    thetas = jnp.asarray(thetas)
    if thetas.ndim == 1:
        maps = np.asarray(model.param_maps(thetas))
    else:
        maps = np.asarray(jax.lax.map(model.param_maps, thetas, batch_size=32))
    return {name: maps[..., i, :, :] for i, name in enumerate(model.param_map_names)}


def compare_names(model):
    """The fit's emulator-parameter maps compared with the truth, in a fixed display order."""
    order = ["log_mass", "Av", "log10metallicity", *RATIOS]
    return [p for p in order if p in model.param_map_names]


def stack_compare(named):
    """Stack a ``{name: (…, H, W)}`` dict into (…, len(COMPARE_FIELDS), H, W)."""
    return np.stack([np.asarray(named[k]) for k in COMPARE_FIELDS], axis=-3)


def in_fields_order(model, param_maps_model_order):
    """Reorder (…, N, H, W) ``param_maps`` (``model.param_map_names``) to COMPARE_FIELDS."""
    idx = [model.param_map_names.index(f) for f in COMPARE_FIELDS]
    return np.take(np.asarray(param_maps_model_order), idx, axis=-3)


def latent_maps(model, theta):
    """``{name: (H, W)}`` of the latent (non-emulator) fields at ``theta``."""
    if not model.latent_names:
        return {}
    fmaps = np.asarray(model.field_maps(jnp.asarray(theta)))
    return {k: fmaps[model.field_names.index(k)] for k in model.latent_names}


def scalar_summary(model, thetas, truth_globals=None):
    """Posterior 16/50/84 of globals, fitted hyperparameters and couplings (+ truth)."""
    thetas = jnp.asarray(thetas)
    out = {}
    if model.n_global:
        g = np.asarray(jax.vmap(model.global_values)(thetas))
        for i, name in enumerate(model.global_names):
            q = np.percentile(g[:, i], [16, 50, 84])
            out[name] = {"p16_50_84": [float(x) for x in q],
                         "truth": (truth_globals or {}).get(name)}  # fmt: skip
    if model.n_hyper:
        h = np.exp(np.asarray(jax.vmap(model.hyper_theta)(thetas)))
        for i, name in enumerate(model.hyper_names):
            q = np.percentile(h[:, i], [16, 50, 84])
            out[name.replace("log_", "")] = {"p16_50_84": [float(x) for x in q]}
    if model.n_coupling:
        c = np.asarray(jax.vmap(model.coupling_theta)(thetas))
        for i, name in enumerate(model.coupling_names):
            q = np.percentile(c[:, i], [16, 50, 84])
            out[name] = {"p16_50_84": [float(x) for x in q]}
    return out


class FixedMapsModel(CorrelatedFieldMap):
    """Renders given physical maps (truth images for the spiral case); no free parameters."""

    def __init__(self, maps, template: CorrelatedFieldMap):
        """Copy ``template``'s layout and store ``maps`` (N_fields, H, W)."""
        self.__dict__.update(template.__dict__)
        self._truth = jnp.asarray(maps)

    def field_maps(self, theta):
        """The stored maps, ignoring theta."""
        return self._truth


def make_observation(emulator, psf, model_for_truth, theta_truth, snr, rng):
    """Render, convolve and add white noise with per-band sigma = peak / snr."""
    n = model_for_truth.image_shape[0]
    blank = ObservationCube(
        flux=np.zeros((len(BANDS), n, n), np.float32),
        variance=np.ones((len(BANDS), n, n), np.float32),
        mask=np.ones((len(BANDS), n, n), np.float32),
        band_names=BANDS,
        pixel_scale=PIXEL_SCALE,
    )
    fm = ForwardModel.build(blank, psf, model_for_truth, emulator)
    clean = np.asarray(fm._model_image(theta_truth), np.float64)
    sigma = clean.reshape(len(BANDS), -1).max(1) / snr
    noisy = clean + rng.normal(0.0, 1.0, clean.shape) * sigma[:, None, None]
    obs = ObservationCube(
        flux=noisy.astype(np.float32),
        variance=np.broadcast_to(sigma[:, None, None] ** 2, clean.shape).astype(np.float32),
        mask=np.ones(clean.shape, np.float32),
        band_names=BANDS,
        pixel_scale=PIXEL_SCALE,
    )
    return obs, clean, sigma


# ---------------------------------------------------------------------------
# Figures and metrics
# ---------------------------------------------------------------------------


def map_metrics(truth, fit, weight):
    """Per-field Pearson r, light-weighted bias and median |error| over the galaxy.

    ``truth`` and ``fit`` are (len(COMPARE_FIELDS), H, W) stacks; r is NaN when either map
    is constant (e.g. a truth ratio held at its TRUTH_GLOBALS value).
    """
    out = {}
    sel = weight > 0.02 * weight.max()
    for k, name in enumerate(COMPARE_FIELDS):
        t, f = truth[k][sel], fit[k][sel]
        varies = np.ptp(f) > 0 and np.ptp(t) > 0
        out[name] = {
            "pearson_r": float(np.corrcoef(t, f)[0, 1]) if varies else float("nan"),
            "median_abs_err": float(np.median(np.abs(f - t))),
            "bias": float(np.average(f - t, weights=weight[sel])),
            "truth_rms": float(np.std(t)),
        }
    return out


def _limits(truth, fit, sel):
    """Colour limits from the truth's 1-99 percentiles, widened if the truth is constant."""
    lo, hi = np.percentile(truth[sel], [1, 99])
    if hi - lo < 1e-6:
        flo, fhi = np.percentile(fit[sel], [1, 99])
        lo, hi = min(lo, flo) - 0.1, max(hi, fhi) + 0.1
    return lo, hi


def plot_maps(truth, start, fit, weight, path, latent=None):
    """Rows: COMPARE_FIELDS.  Columns: truth, start (mean map), MAP, MAP - truth.

    ``latent`` ``{name: (start, MAP)}`` adds a row per latent field (no truth).
    """
    sel = weight > 0.02 * weight.max()
    latent = latent or {}
    nrow = len(COMPARE_FIELDS) + len(latent)
    fig, axes = plt.subplots(nrow, 4, figsize=(9.5, 2.3 * nrow), squeeze=False)
    for k, name in enumerate(COMPARE_FIELDS):
        lo, hi = _limits(truth[k], fit[k], sel)
        for j, (img, title) in enumerate(
            [(truth[k], "truth"), (start[k], "start (mean map)"), (fit[k], "MAP")]
        ):
            ax = axes[k, j]
            im = ax.imshow(np.where(sel, img, np.nan), origin="lower", vmin=lo, vmax=hi,
                           cmap="magma" if k == 0 else "viridis")  # fmt: skip
            ax.set_title(f"{LABELS[name]}: {title}", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)
        d = np.where(sel, fit[k] - truth[k], np.nan)
        lim = np.nanpercentile(np.abs(d), 98) or 1e-3
        im = axes[k, 3].imshow(d, origin="lower", cmap="RdBu_r",
                               norm=TwoSlopeNorm(0, -lim, lim))  # fmt: skip
        axes[k, 3].set_title("MAP - truth", fontsize=8)
        plt.colorbar(im, ax=axes[k, 3], fraction=0.046)
    for r, (name, (st, ft)) in enumerate(latent.items(), start=len(COMPARE_FIELDS)):
        lo, hi = np.percentile(np.concatenate([st[sel], ft[sel]]), [1, 99])
        hi = max(hi, lo + 1e-3)
        axes[r, 0].axis("off")
        axes[r, 3].axis("off")
        for j, (img, title) in ((1, (st, "start (mean map)")), (2, (ft, "MAP"))):
            im = axes[r, j].imshow(np.where(sel, img, np.nan), origin="lower", vmin=lo, vmax=hi,
                                   cmap="PuOr")  # fmt: skip
            axes[r, j].set_title(f"{LABELS.get(name, name)}: {title}", fontsize=8)
            plt.colorbar(im, ax=axes[r, j], fraction=0.046)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_images(obs, model_img, sigma, path, show=("F090W", "F200W", "F444W")):
    """Rows: bands.  Columns: data, PSF-convolved MAP model, residual / sigma."""
    bands = [BANDS.index(f"JWST/NIRCam.{f}") for f in show]
    fig, axes = plt.subplots(len(bands), 3, figsize=(7.5, 2.4 * len(bands)))
    for r, b in enumerate(bands):
        data = np.asarray(obs.flux[b])
        vmax = np.percentile(data, 99.5)
        for c, (img, title) in enumerate([(data, "data"), (model_img[b], "MAP model")]):
            im = axes[r, c].imshow(img, origin="lower", cmap="gray_r", vmin=-0.05 * vmax,
                                   vmax=vmax)  # fmt: skip
            axes[r, c].set_title(f"{BANDS[b].split('.')[-1]} {title}", fontsize=8)
            plt.colorbar(im, ax=axes[r, c], fraction=0.046)
        chi = (data - model_img[b]) / sigma[b]
        im = axes[r, 2].imshow(chi, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4)
        axes[r, 2].set_title(f"residual / sigma (rms {np.std(chi):.2f})", fontsize=8)
        plt.colorbar(im, ax=axes[r, 2], fraction=0.046)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Laplace posterior
# ---------------------------------------------------------------------------


def polish_mode(fm, theta, chunk, precond="dense", max_iter=100, tol=1e-2, key=None, M0=None):
    """Trust-region Newton (``arachne.inference.newton``) from an Adam MAP to a local mode.

    ``precond``: ``dense`` builds ``M = V |Lambda| V^T`` from the full Hessian at the start
    (the Laplace step's cost, once) and rebuilds it every 20 accepted steps or when CG hits
    its cap; ``diag`` uses a Hutchinson diagonal; ``fourier`` the diagonal Fisher of the
    field basis (``arachne.inference.field_precond.fisher_preconditioner``; experimental,
    measured slower than the identity on the toy); ``none`` the identity.  ``M0`` is a
    ready-made initial preconditioner (skips the first build).
    """
    from arachne.inference.laplace import make_hessian_fn
    from arachne.inference.newton import (
        dense_preconditioner,
        diagonal_preconditioner,
        trust_region_newton,
    )

    def neg_lp(t):
        return -fm.log_posterior(t)

    refresh = None
    if precond == "dense":
        hess_fn = make_hessian_fn(fm, chunk_size=chunk)

        def refresh(t):
            t0 = time.time()
            M = dense_preconditioner(hess_fn(t))
            print(f"  preconditioner: dense Hessian + eigh in {time.time() - t0:.0f} s", flush=True)
            return M
    elif precond == "diag":
        key = jax.random.PRNGKey(0) if key is None else key

        def refresh(t):
            return diagonal_preconditioner(neg_lp, t, key)
    elif precond == "fourier":
        from arachne.inference.field_precond import fisher_preconditioner

        def refresh(t):
            return fisher_preconditioner(fm, t)

    M = M0 if M0 is not None else (refresh(theta) if refresh else None)
    res = trust_region_newton(neg_lp, theta, preconditioner=M, refresh_preconditioner=refresh,
                              refresh_every=20, max_iter=max_iter, tol=tol,
                              print_fn=lambda m: print(m, flush=True))  # fmt: skip
    print(res.summary(), flush=True)
    return res


class CoefficientAdam:
    """Adam on the field coefficients only, with the galaxy-level block frozen.

    One jitted runner, built once and reused for every round (the frozen block is an
    argument, not a closure constant, so new block values do not recompile).  Same
    optimiser as ``find_map``: ``clip_by_global_norm(50)`` then Adam, fresh state per call.
    """

    def __init__(self, fm, idx, clip_norm=50.0):
        """Split theta into the block ``idx`` (frozen) and the rest (optimised)."""
        import optax

        d = fm.n_params
        self.free = jnp.asarray(np.setdiff1d(np.arange(d), np.asarray(idx)))

        def loss(tf, base):
            return -fm.log_posterior(base.at[self.free].set(tf))

        vg = jax.value_and_grad(loss)

        def run(tf, base, lr, n_steps):
            opt = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(lr))

            def step(carry, _):
                t, s = carry
                val, g = vg(t, base)
                upd, s = opt.update(g, s, t)
                return (optax.apply_updates(t, upd), s), val

            (tf, _), losses = jax.lax.scan(step, (tf, opt.init(tf)), None, length=n_steps)
            return tf, losses

        self._run = jax.jit(run, static_argnums=3)
        self._loss = jax.jit(loss)

    def __call__(self, theta, lr, n_steps):
        """``n_steps`` Adam steps from ``theta``; returns ``(theta, losses)``.

        Like ``find_map``'s rounds, a round that ends above its start is discarded.
        """
        tf0 = theta[self.free]
        tf, losses = self._run(tf0, theta, jnp.asarray(lr, theta.dtype), int(n_steps))
        if not float(self._loss(tf, theta)) < float(self._loss(tf0, theta)):
            return theta, np.asarray(losses)
        return theta.at[self.free].set(tf), np.asarray(losses)


def map_stage(fm, theta0, rounds, steps, block=True):
    """Adam MAP, optionally as block-coordinate descent with the galaxy-level block.

    Without ``block``: one ``find_map`` call, ``rounds`` x ``steps`` Adam steps at lr 0.05
    then ``steps`` at 0.01.  With ``block``: the same Adam steps, but on the field
    coefficients only, each round followed by an exact Newton solve (``block_newton``, up to
    10 iterations, coefficients fixed) on the levels, globals, hyperparameters and couplings
    (``galaxy_block_indices``), whose curvature (1e6-1e8) Adam cannot follow; Adam can then
    not undo the block solve.  The final polish is one lr-0.01 coefficient round and a last
    block solve.

    Returns:
        ``(MAPResult, log)``: ``log`` has one record per round.
    """
    if not block:
        r = find_map(fm, theta0, n_rounds=rounds, steps_per_round=steps, lr=0.05,
                     final_steps=steps, final_lr=0.01)  # fmt: skip
        return r, []
    from arachne.inference.newton import block_newton, galaxy_block_indices

    idx = galaxy_block_indices(fm)
    neg_lp = jax.jit(lambda t: -fm.log_posterior(t))
    adam = CoefficientAdam(fm, idx)
    theta, hist, log = jnp.asarray(theta0), [], []
    schedule = [(f"round {r + 1}", 0.05) for r in range(rounds)] + [("final", 0.01)]
    for stage, lr in schedule:
        t0 = time.time()
        theta, losses = adam(theta, lr, steps)
        hist.append(losses)
        after_adam = float(neg_lp(theta))
        t1 = time.time()
        b = block_newton(neg_lp, theta, idx, max_iter=10)
        theta = b.theta
        rec = {"stage": stage, "neg_log_post_adam": after_adam,
               "neg_log_post_newton": float(b.neg_log_post), "newton_iter": b.n_iter,
               "newton_reason": b.reason, "adam_s": t1 - t0,
               "newton_s": time.time() - t1}  # fmt: skip
        log.append(rec)
        print(f"  {stage:8s}: -log_post after Adam {after_adam:.2f} ({rec['adam_s']:.1f} s), "
              f"after block Newton {rec['neg_log_post_newton']:.2f} ({b.reason}, {b.n_iter} it, "
              f"{rec['newton_s']:.1f} s)", flush=True)  # fmt: skip
    h = np.concatenate(hist)
    nlp = float(neg_lp(theta))
    return MAPResult(theta=theta, neg_log_posterior=nlp, history=h, n_steps=int(h.size)), log


def laplace_posterior(fm, model, theta_map, n_samples, chunk, key, max_newton=300,
                      precond="dense", max_escapes=3):  # fmt: skip
    """Trust-region Newton from the MAP to the mode, one full Hessian there, Gaussian samples.

    After each Newton run the Hessian is checked for negative eigenvalues; if a step along
    those directions lowers ``-log p`` the polish restarts from there (up to
    ``max_escapes`` times), preconditioned by that Hessian.

    Run with ``--float64``: the Hessian of a 1e4-pixel likelihood is round-off
    dominated in float32.

    Returns:
        Dict with the polished theta, ``param_maps`` samples (S, N, H, W), the
        Laplace evidence and diagnostics.
    """
    from arachne.inference.laplace import laplace_covariance, make_hessian_fn
    from arachne.inference.newton import Preconditioner, negative_curvature_step

    dtype = jnp.zeros(()).dtype
    neg_lp = jax.jit(lambda t: -fm.log_posterior(t))
    hess_fn = make_hessian_fn(fm, chunk_size=chunk)
    theta, M0, history, escapes = jnp.asarray(theta_map, dtype), None, [], []
    t_newton = t_hess = 0.0
    for attempt in range(max_escapes + 1):
        t0 = time.time()
        newton = polish_mode(fm, theta, chunk, precond=precond, max_iter=max_newton, M0=M0)
        theta = newton.theta
        history += newton.history
        t_newton += time.time() - t0
        t0 = time.time()
        hess = hess_fn(theta)
        t_hess += time.time() - t0
        # Saddle check: CG stopping on an interior step never sees weak negative curvature.
        evals, evecs = jnp.linalg.eigh(0.5 * (jnp.asarray(hess) + jnp.asarray(hess).T))
        if attempt == max_escapes or float(evals[0]) >= 0.0:
            break
        cand, gain, n_neg = negative_curvature_step(neg_lp, theta, evals, evecs)
        print(f"  saddle check: {n_neg} negative eigenvalues (min {float(evals[0]):.3g}); "
              f"best step along them gains {gain:.3g} nats", flush=True)  # fmt: skip
        escapes.append({"n_negative": n_neg, "eig_min": float(evals[0]), "gain": gain})
        if gain < 1e-2:
            break
        lam = jnp.abs(evals)
        M0 = Preconditioner("dense", jnp.maximum(lam, jnp.maximum(1e-8 * lam.max(), 1e-6)), evecs)
        theta = cand
    del evals, evecs
    cov, chol, info = laplace_covariance(hess, return_info=True)
    # Laplace evidence: log Z ~ log p(d, theta*) + d/2 log 2pi - 1/2 log det H.  The
    # Gaussian likelihood omits its data-only normalisation, so compare log Z only
    # between models fitted to the same data (e.g. different ell / nu).
    lam = np.asarray(info["modified_eigenvalues"])
    log_z = float(-neg_lp(theta)) + 0.5 * lam.size * np.log(2 * np.pi) - 0.5 * np.sum(np.log(lam))
    eps = jax.random.normal(key, (n_samples, model.n_params), dtype)
    thetas = theta[None, :] + eps @ jnp.asarray(chol, dtype).T
    maps = np.asarray(jax.lax.map(model.param_maps, thetas, batch_size=32))
    return {
        "theta": theta,
        "maps": maps,
        "thetas": np.asarray(thetas),
        "chol": chol,
        "newton_s": t_newton,
        "hessian_s": t_hess,
        "newton_history": history,
        "newton_converged": newton.converged,
        "newton_reason": newton.reason,
        "newton_hvps": newton.n_hvp,
        "saddle_escapes": escapes,
        "log_evidence_laplace": log_z,
        "eig_min": float(info["eigenvalues"].min()),
        "eig_max": float(info["eigenvalues"].max()),
        "n_negative": int(info["n_negative"]),
        "n_floored": int(info["n_floored"]),
    }


def nuts_posterior(fm, model, theta, chol, key, n_warmup, n_samples, n_chains, max_maps=1000,
                   init_thetas=None, whitened=None):  # fmt: skip
    """Whitened NUTS (theta = mode + L z, L from the Laplace Hessian), diagonal metric in z.

    A dense metric cannot be adapted at d ~ 4e3, so the Laplace whitening does the
    conditioning and the warmup only adapts a diagonal.  ``init_thetas`` ``(n_chains, d)``
    starts each chain at its own point (modes from different optimiser starts).  ``whitened``
    replaces the Laplace whitening, e.g. a VI ``StandardisedLogDensity`` (``chol`` unused;
    its diagonal metric is then adapted by the warmup alone, since ``run_whitened_nuts`` has
    no hook for an initial inverse mass matrix).  Returns map samples
    (thinned to ``max_maps``) and convergence diagnostics in theta coordinates.
    """
    from arachne.inference.laplace import WhitenedLogDensity, run_whitened_nuts

    dtype = jnp.zeros(()).dtype
    if whitened is None:
        whitened = WhitenedLogDensity(fm, jnp.asarray(theta, dtype), jnp.asarray(chol, dtype))
    t0 = time.time()
    result, _ = run_whitened_nuts(
        fm, theta, key, whitened=whitened, n_warmup=n_warmup, n_samples=n_samples,
        n_chains=n_chains, dense_mass_matrix=False, initial_positions=init_thetas,
    )  # fmt: skip
    runtime = time.time() - t0
    samples = np.asarray(result.samples)
    step = max(1, samples.shape[0] // max_maps)
    maps = np.asarray(jax.lax.map(model.param_maps, jnp.asarray(samples[::step]), batch_size=32))
    d = result.diagnostics
    rhat = np.asarray(d.get("rhat", np.nan), float)
    ess_ = np.asarray(d.get("ess", np.nan), float)
    print(result.summary())
    return {
        "maps": maps,
        "theta_samples": samples[::step],
        "n_chains": n_chains,
        "runtime_s": runtime,
        "max_rhat": float(np.nanmax(rhat)),
        "frac_rhat_gt_1.01": float(np.mean(rhat > 1.01)),
        "min_ess": float(np.nanmin(ess_)),
        "median_ess": float(np.nanmedian(ess_)),
        "summary": result.summary(),
    }


def vi_posterior(fm, model, theta_map, key, mode, n_iterations, n_samples, n_final):
    """geoVI / MGVI (``arachne.inference.mgvi.run_mgvi``) from the MAP.

    The posterior draws are the ``n_final`` extra linear samples at the final mean (the VI
    samples themselves if there are none).  Returns a dict shaped like
    :func:`laplace_posterior`'s (``theta`` = ``T(xi_mean)``, ``maps``, ``thetas``) plus the
    VI result, the per-coordinate ``xi`` scale (``mgvi_scale``) and the history.
    """
    from arachne.inference.mgvi import mgvi_scale, run_mgvi

    res = run_mgvi(fm, theta_map, key, n_iterations=n_iterations, n_samples=n_samples,
                   mode=mode, n_final_samples=n_final or None)  # fmt: skip
    thetas = res.theta_extra_samples if res.theta_extra_samples is not None else res.theta_samples
    thetas = jnp.asarray(thetas)
    theta_mean = jnp.asarray(res.theta_mean)
    return {
        "theta": theta_mean,
        "maps": np.asarray(jax.lax.map(model.param_maps, thetas, batch_size=32)),
        "thetas": np.asarray(thetas),
        "result": res,
        "scale": mgvi_scale(res),
        "mode": mode,
        "runtime_s": res.runtime_s,
        "history": res.history,
        "neg_log_post_mean": float(-fm.log_posterior(theta_mean)),
    }


def vi_chain_starts(density, centres, scale, key, n_chains, jitter=0.3):
    """``(n_chains, d)`` starts in ``xi``: the centres (theta) cycled, jittered by jitter*scale."""
    xi = [np.asarray(density.from_theta(jnp.asarray(c))) for c in centres]
    eps = np.asarray(jax.random.normal(key, (n_chains, density.n_params)))
    return np.stack([xi[c % len(xi)] + jitter * scale * eps[c] for c in range(n_chains)])


def mclmc_posterior(fm, model, theta, chol, key, n_warmup, n_samples, n_chains, max_maps=1000,
                    init_thetas=None, n_burn=500, diag=False, whitened=None, z_starts=None,
                    z_tune=None):  # fmt: skip
    """Whitened MCLMC: as ``nuts_posterior`` but blackjax MCLMC.

    One tuning run from the mode (no diagonal re-preconditioning unless ``diag``, since the
    target is already whitened) is shared by every chain.  Tuning each chain separately gave
    L 15-157 and step 0.6-34 across four chains and R-hat 2.4 (runs/2026-11-01_mclmc, commit
    f16613e); with d = 5035 a step of ~30 is ~0.5 sigma per coordinate, so the small-step
    chains barely moved.  Chains start at z ~ N(0, 0.3^2) around the mode, or at
    ``init_thetas`` when given, and discard ``n_burn`` steps; R-hat and ESS are computed in
    theta.  ``whitened`` replaces the Laplace whitening (e.g. a VI
    ``StandardisedLogDensity``); then ``z_starts`` gives the per-chain starts and
    ``z_tune`` the tuning start in its coordinates.
    """
    from arachne.inference.diagnostics import ess, split_rhat
    from arachne.inference.laplace import WhitenedLogDensity
    from arachne.inference.mclmc_sampler import MCLMCSampler

    dtype = jnp.zeros(()).dtype
    if whitened is None:
        whitened = WhitenedLogDensity(fm, jnp.asarray(theta, dtype), jnp.asarray(chol, dtype))
    sampler = MCLMCSampler(
        whitened, n_warmup=n_warmup, n_samples=n_samples, diagonal_preconditioning=diag
    )
    t0 = time.time()
    k_tune, key = jax.random.split(key)
    z_tune = jnp.zeros(whitened.n_params, dtype) if z_tune is None else z_tune
    _, params = sampler.tune(jnp.asarray(z_tune, dtype), k_tune)
    tuned = {"L": float(params.L), "step_size": float(params.step_size)}
    print(f"MCLMC tuning: {time.time() - t0:.0f} s, L {tuned['L']:.3g}, "
          f"step {tuned['step_size']:.3g}")  # fmt: skip
    chains = []
    for c, k in enumerate(jax.random.split(key, n_chains)):
        k_init, k_run = jax.random.split(k)
        z0 = 0.3 * jax.random.normal(k_init, (whitened.n_params,), dtype)
        if z_starts is not None:
            z0 = jnp.asarray(z_starts[c], dtype)
        elif init_thetas is not None:
            z0 = whitened.from_theta(jnp.asarray(init_thetas[c], dtype))
        res = sampler.run(z0, k_run, params=params, n_burn=n_burn)
        chains.append(np.asarray(jax.vmap(whitened.to_theta)(res.samples)))
        print(f"MCLMC chain {c + 1}/{n_chains}: {time.time() - t0:.0f} s")
    runtime = time.time() - t0
    chains = np.stack(chains)
    rhat, ess_ = split_rhat(chains), ess(chains)
    samples = chains.reshape(-1, chains.shape[-1])
    step = max(1, samples.shape[0] // max_maps)
    maps = np.asarray(jax.lax.map(model.param_maps, jnp.asarray(samples[::step]), batch_size=32))
    return {
        "maps": maps,
        "theta_samples": samples[::step],
        "n_chains": n_chains,
        "runtime_s": runtime,
        "max_rhat": float(np.nanmax(rhat)),
        "frac_rhat_gt_1.01": float(np.mean(rhat > 1.01)),
        "min_ess": float(np.nanmin(ess_)),
        "median_ess": float(np.nanmedian(ess_)),
        "tuned": tuned,
    }


def prior_map_std(model, n, key):
    """Per-pixel prior std of each ``param_maps`` map's *local* structure (levels at zero).

    The level prior is deliberately broad (e.g. 1 dex in mass), so including it
    would make every pixel look ~100% data-dominated.  Holding the levels fixed
    measures how much the data constrain the spatial fluctuations themselves.
    Returns (N, H, W) in ``model.param_map_names`` order (the bundle's map order).
    """
    thetas = model.sample_prior(key, n).at[:, : model.n_fields].set(0.0)
    maps = np.asarray(jax.lax.map(model.param_maps, thetas, batch_size=32))
    return maps.std(0)


def posterior_metrics(truth, samples, prior_std, weight):
    """Coverage of the truth by 68/95% intervals, z-scores and information per field.

    Pixels with zero posterior (prior) std, e.g. a derived ratio with zero basis weight,
    get NaN z-scores (information).
    """
    sel = weight > 0.02 * weight.max()
    lo68, med, hi68 = np.percentile(samples, [16, 50, 84], axis=0)
    lo95, hi95 = np.percentile(samples, [2.5, 97.5], axis=0)
    std = samples.std(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        info = np.where(prior_std > 0, 1.0 - (std / prior_std) ** 2, np.nan)
    out = {}
    for k, name in enumerate(COMPARE_FIELDS):
        t = truth[k][sel]
        sd = std[k][sel]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sd > 0, (med[k][sel] - t) / sd, np.nan)
        out[name] = {
            "coverage68": float(np.mean((t >= lo68[k][sel]) & (t <= hi68[k][sel]))),
            "coverage95": float(np.mean((t >= lo95[k][sel]) & (t <= hi95[k][sel]))),
            "z_rms": float(np.sqrt(np.nanmean(z**2))) if np.any(sd > 0) else float("nan"),
            "median_post_std": float(np.median(std[k][sel])),
            "median_information": (
                float(np.nanmedian(info[k][sel])) if np.isfinite(info[k][sel]).any() else np.nan
            ),
        }
    return out, med, std, info


def plot_laplace(truth, med, std, info, weight, path):
    """Rows: COMPARE_FIELDS.  Columns: truth, median, posterior std, z-score, information."""
    sel = weight > 0.02 * weight.max()
    nrow = len(COMPARE_FIELDS)
    fig, axes = plt.subplots(nrow, 5, figsize=(12, 2.3 * nrow), squeeze=False)
    for k, name in enumerate(COMPARE_FIELDS):
        lo, hi = _limits(truth[k], med[k], sel)
        cmap = "magma" if k == 0 else "viridis"
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(std[k] > 0, (med[k] - truth[k]) / std[k], np.nan)
        panels = [
            (truth[k], "truth", dict(cmap=cmap, vmin=lo, vmax=hi)),
            (med[k], "posterior median", dict(cmap=cmap, vmin=lo, vmax=hi)),
            (std[k], "posterior std", dict(cmap="cividis")),
            (z, "(median - truth) / std", dict(cmap="RdBu_r", vmin=-3, vmax=3)),
            (info[k], "information 1 - var/var_prior", dict(cmap="Greys", vmin=0, vmax=1)),
        ]
        for j, (img, title, kw) in enumerate(panels):
            ax = axes[k, j]
            im = ax.imshow(np.where(sel, img, np.nan), origin="lower", **kw)
            ax.set_title(f"{LABELS[name]}: {title}", fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results bundle
# ---------------------------------------------------------------------------


def truth_scalars(model, scalar_names, truth_maps, prior_truth, corr_scale=1.0, amp_scale=1.0):
    """Truth values of the bundle scalars, by name (NaN where the truth has none).

    Globals come from TRUTH_GLOBALS.  For the prior truth the hyperparameters are the truth
    model's fixed values and the couplings are 0; the spiral has neither.
    """
    out = []
    for name in scalar_names:
        if name == "total_log_mass":
            out.append(
                float(np.log10(np.sum(10.0 ** truth_maps[COMPARE_FIELDS.index("log_mass")])))
            )
        elif name in TRUTH_GLOBALS:
            out.append(float(TRUTH_GLOBALS[name]))
        elif prior_truth and name.startswith("amplitude["):
            out.append(amp_scale * float(AMPLITUDES[name[len("amplitude[") : -1]]))
        elif prior_truth and name.startswith("corr_length["):
            out.append(corr_scale * float(CORR_LENGTHS[name[len("corr_length[") : -1]]))
        elif prior_truth and name.startswith("c["):
            out.append(0.0)
        else:
            out.append(float("nan"))
    return np.asarray(out)


def demo_regions(truth_maps, arm_weight=None):
    """``galaxy`` (the metrics selection), plus ``arms``/``interarm`` for the spiral.

    ``truth_maps``: the COMPARE_FIELDS stack (mass first).
    """
    weight = 10.0 ** truth_maps[0]
    galaxy = weight > 0.02 * weight.max()
    regions = {"galaxy": Region.from_mask(galaxy, name="galaxy")}
    if arm_weight is not None:
        arms = galaxy & (arm_weight > 0.3)
        regions["arms"] = Region.from_mask(arms, name="arms")
        # disk outside the arms (arm_weight is zero in the bulge, so exclude r < 3 px there)
        n = truth_maps.shape[-1]
        yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2.0
        regions["interarm"] = Region.from_mask(
            galaxy & (arm_weight < 0.05) & (np.hypot(yy, xx) > 3.0), name="interarm"
        )
    return regions


def write_demo_bundle(path, fm, model, *, theta_map, samples, source, n_chains, args,
                      truth, truth_maps, regions, prior_std, metrics, diagnostics):  # fmt: skip
    """``content_from_forward_model`` + truth scalars + ``write_content``; returns the path."""
    t0 = time.time()
    content = content_from_forward_model(
        fm,
        theta_map=theta_map,
        samples=samples,
        posterior_source=source,
        n_draws=args.bundle_draws,
        n_chains=n_chains,
        store_intrinsic=not args.bundle_no_intrinsic,
        truth=truth,
        prior_std_maps=prior_std,
        regions=regions,
        metrics=metrics,
        diagnostics=diagnostics,
        pivot_um=PIVOT_UM,
        labels=LABELS,
        units={
            "log_mass": "dex",
            "Av": "mag",
            "log10metallicity": "dex",
            "total_log_mass": "dex",
        },  # fmt: skip
        title=path.parent.name if path.name == "results.h5" else f"{path.parent.name} {source}",
    )
    content.truth.scalars = truth_scalars(
        model,
        content.scalar_names,
        truth_maps,
        prior_truth=args.truth == "prior",
        **truth_hyper_scales(args),
    )
    write_content(path, content, mantissa_bits=args.bundle_mantissa_bits, overwrite=True)
    size = path.stat().st_size / 2**20
    print(f"bundle {path} ({content.n_draws} draws, {size:.1f} MiB, {time.time() - t0:.1f} s)")
    return path


def truth_hyper_scales(args):
    """Truth-model correlation-length and amplitude scales (mismatched for ``--honest``)."""
    if args.honest:
        return {"corr_scale": HONEST_CORR_SCALE, "amp_scale": HONEST_AMP_SCALE}
    return {"corr_scale": 1.0, "amp_scale": 1.0}


def bundle_plots(path, outdir):
    """``arachne.viz.save_all`` of the bundle into ``outdir`` (skipped if viz is missing)."""
    try:
        from arachne.viz import save_all
    except ImportError as err:
        print(f"--bundle-plots: arachne.viz unavailable ({err}); skipped")
        return
    from arachne.results import load_bundle

    files = save_all(load_bundle(path), outdir)
    print(f"bundle plots: {len(files)} files in {outdir}")


def perturbed_start(model, theta0, key):
    """``theta0`` with the non-mass levels and the globals redrawn from the prior.

    The mass level and every field coefficient are kept, so each start sees the same
    data-driven mass map but a different guess for the SPS parameters: the places where
    SED fitting is multimodal (age-dust-metallicity, SFH bin trading).
    """
    draw = model.sample_prior(key, 1)[0]
    nf, ng = model.n_fields, model.n_global
    mass = np.arange(nf) == model.field_names.index(model.mass_param)
    levels = jnp.where(jnp.asarray(mass), theta0[:nf], draw[:nf])
    return theta0.at[:nf].set(levels).at[nf : nf + ng].set(draw[nf : nf + ng])


def mode_summary(model, post, truth_maps, pstd, weight):
    """Compact per-start record: optimiser status, evidence, coverage, total mass, globals.

    For a VI posterior ``neg_log_post`` is ``-log p(T(xi_mean))`` and the Newton / Laplace
    entries are None.
    """
    maps = in_fields_order(model, post["maps"])
    pm = posterior_metrics(truth_maps, maps, pstd, weight)[0]
    mass_tot = np.log10(np.sum(10.0 ** maps[:, 0], axis=(1, 2)))
    sc = scalar_summary(model, post["thetas"], TRUTH_GLOBALS)
    laplace = "newton_history" in post
    return {
        "neg_log_post": float(
            post["newton_history"][-1]["neg_log_post"] if laplace else post["neg_log_post_mean"]
        ),  # fmt: skip
        "newton_converged": post.get("newton_converged"),
        "newton_reason": post.get("newton_reason"),
        "log_evidence_laplace": post.get("log_evidence_laplace"),
        "n_negative": post.get("n_negative"),
        "total_log_mass_16_50_84": [float(x) for x in np.percentile(mass_tot, [16, 50, 84])],
        "coverage68": {k: v["coverage68"] for k, v in pm.items()},
        "global_medians": {
            k: v["p16_50_84"][1]
            for k, v in sc.items()
            if not k.startswith(("amplitude", "corr_length", "c["))
        },  # fmt: skip
    }


def select_best(runs):
    """Index of the best optimiser start.

    Lowest ``neg_log_post`` among starts whose Newton polish converged with no negative
    Hessian eigenvalue; if none did (or there was no Laplace step, e.g. a VI posterior, where
    ``neg_log_post`` is ``-log p`` at the VI mean), lowest ``neg_log_post`` overall.  The
    Laplace evidence is never used: it rewards the broad, flat curvature of unconverged
    modes.
    """
    nlp = [x["summary"]["neg_log_post"] for x in runs]
    good = [
        i
        for i, x in enumerate(runs)
        if x["summary"].get("newton_converged") and x["summary"].get("n_negative") == 0
    ]
    pool = good or range(len(runs))
    return min(pool, key=lambda i: nlp[i])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the demo."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--truth", choices=["prior", "spiral"], default="prior")
    ap.add_argument("--size", type=int, default=64, help='cutout size in pixels (0.03"/px)')
    ap.add_argument("--snr", type=float, default=30.0, help="per-pixel S/N at each band's peak")
    ap.add_argument("--emulator", default=None, help="production emulator checkpoint")
    ap.add_argument("--psf-dir", default=None, help="directory of {FILTER}_psf_norm.fits")
    ap.add_argument(
        "--bands",
        choices=sorted(BAND_SETS),
        default="wide8",
        help="wide8: 7 wide + F410M; medium: + all other medium bands (17)",
    )
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=500, help="Adam steps per round")
    ap.add_argument("--quick", action="store_true", help="2 rounds x 300 steps")
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="outputs/demo/correlated_field")
    ap.add_argument(
        "--mass-start",
        choices=["wiener", "image", "template"],
        default="image",
        help="start: smoothed-F444W mean (image); plus a per-pixel colour-template correction "
        "of the mean (template, template_mass_map); or plus field coefficients projected "
        "from a multi-band Wiener deconvolution (wiener)",
    )
    ap.add_argument(
        "--corr-scale",
        type=float,
        default=1.0,
        help="multiply every fit correlation length (truth unchanged)",
    )
    ap.add_argument("--nu", type=float, default=1.5,
                    help="Matern nu of the fit model's non-mass fields")  # fmt: skip
    ap.add_argument("--hyper", action="store_true",
                    help="fit amplitude and correlation length of the non-mass fields")  # fmt: skip
    ap.add_argument("--couple", action="store_true",
                    help="fit A_V couplings to the mass and SFH-ratio deviations")  # fmt: skip
    ap.add_argument("--honest", action="store_true",
                    help="offset the truth means from the fit's prior centres (HONEST_OFFSETS) "
                    "and draw the prior truth with different hyperparameters")  # fmt: skip
    ap.add_argument("--sfh", choices=SFH_SETUPS, default="basis2",
                    help="SFH ratios mapped / derived / free / fixed (see set_sfh)")  # fmt: skip
    ap.add_argument("--free-globals", action="store_true",
                    help="fit fesc_lya and dust_bump_amplitude as free globals (default: fixed "
                    "at NEUTRAL_GLOBALS; they barely change the fluxes at z = 2)")  # fmt: skip
    ap.add_argument("--k-max-mtf", type=float, default=0.0,
                    help="if > 0: truncate every field at the wavenumber where the sharpest "
                    "PSF's MTF drops below this (psf_k_max); mass keeps every mode below "
                    "it")  # fmt: skip
    ap.add_argument("--tag", default="", help="suffix for the output directory name")
    ap.add_argument("--posterior", choices=["map", "laplace", "geovi", "mgvi"], default="map",
                    help="posterior after the MAP: none (map), Laplace at the polished mode, or "
                    "geoVI / MGVI (arachne.inference.mgvi, from the MAP)")  # fmt: skip
    ap.add_argument("--laplace", action="store_true", help="alias for --posterior laplace")
    ap.add_argument("--n-laplace", type=int, default=400,
                    help="posterior draws (Laplace samples, or VI final samples)")  # fmt: skip
    ap.add_argument("--vi-iterations", type=int, default=6, help="geoVI/MGVI iterations")
    ap.add_argument("--vi-samples", type=int, default=4,
                    help="geoVI/MGVI antithetic sample pairs per iteration")  # fmt: skip
    ap.add_argument("--block-newton", action=argparse.BooleanOptionalAction, default=False,
                    help="block-coordinate MAP: Adam on the field coefficients alternating with "
                    "exact Newton on the galaxy-level block (levels, globals, hyper, "
                    "couplings).  Off by default: on the toy spiral it gained 19 nats for "
                    "basis2 but lost 2 for map03 at equal Adam steps")  # fmt: skip
    ap.add_argument("--nuts", action="store_true",
                    help="whitened MCMC (--sampler) after the Laplace step")  # fmt: skip
    ap.add_argument("--sampler", choices=["nuts", "mclmc"], default="nuts")
    ap.add_argument(
        "--mclmc-warmup",
        type=int,
        default=5000,
        help="MCLMC tuning steps (one tuning shared by all chains)",
    )
    ap.add_argument(
        "--mclmc-burn",
        type=int,
        default=500,
        help="MCLMC steps discarded per chain after the shared tuning",
    )
    ap.add_argument(
        "--mclmc-diag",
        action="store_true",
        help="also adapt a diagonal mass matrix on top of the whitening",
    )
    ap.add_argument("--mclmc-samples", type=int, default=2000, help="MCLMC steps per chain")
    ap.add_argument("--nuts-warmup", type=int, default=1000)
    ap.add_argument("--nuts-samples", type=int, default=500)
    ap.add_argument("--nuts-chains", type=int, default=4)
    ap.add_argument("--hess-chunk", type=int, default=32, help="HVP rows per Hessian batch")
    ap.add_argument("--precond", choices=["dense", "diag", "fourier", "none"], default="dense",
                    help="trust-region Newton preconditioner (see polish_mode); fourier is "
                    "experimental, measured slower than identity on the toy")  # fmt: skip
    ap.add_argument("--max-newton", type=int, default=300, help="trust-region Newton iterations")
    ap.add_argument("--n-starts", type=int, default=1,
                    help="optimiser starts: the neutral start plus prior redraws of the non-mass "
                    "levels and globals; the lowest -log p among converged modes with no "
                    "negative Hessian eigenvalue is kept (lowest overall if none converged; "
                    "never the Laplace evidence), and MCMC chains start at the modes within "
                    "10 nats of it")  # fmt: skip
    ap.add_argument("--sfh-prior", choices=sorted(SFH_PRIOR_SCALES), default="continuity",
                    help="SFH-ratio prior: bursty / continuity Student-t (maps via a copula), "
                    "or legacy (continuity globals, Gaussian maps)")  # fmt: skip
    ap.add_argument("--no-bundle", action="store_true", help="do not write results.h5")
    ap.add_argument("--bundle-draws", type=int, default=200, help="posterior draws in the bundle")
    ap.add_argument("--bundle-mantissa-bits", type=int, default=None,
                    help="round bundle draws to N mantissa bits (12: ~1e-4)")  # fmt: skip
    ap.add_argument("--bundle-no-intrinsic", action="store_true",
                    help="omit the unconvolved model cubes from the bundle")  # fmt: skip
    ap.add_argument("--bundle-all", action="store_true",
                    help="also write results_laplace.h5 when NUTS ran")  # fmt: skip
    ap.add_argument("--bundle-plots", action="store_true",
                    help="arachne.viz.save_all into figures/bundle/")  # fmt: skip
    args = ap.parse_args()
    if args.laplace:
        if args.posterior not in ("map", "laplace"):
            ap.error("--laplace conflicts with --posterior " + args.posterior)
        args.posterior = "laplace"
    args.laplace = args.posterior == "laplace"
    vi = args.posterior in ("geovi", "mgvi")
    set_bands(args.bands)
    set_sfh(args.sfh, free_globals=args.free_globals)
    global SFH_PRIOR, SFH_BASIS, COMPARE_FIELDS
    SFH_PRIOR = args.sfh_prior
    if args.float64:
        jax.config.update("jax_enable_x64", True)
    if args.quick:
        args.rounds, args.steps = 2, 300
    outdir = Path(args.outdir) / f"{args.truth}_{args.bands}{'_' + args.tag if args.tag else ''}"
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n = args.size

    emulator, psf = load_emulator_and_psf(args)
    print(f"emulator: {type(emulator).__name__}; PSFs: {'files' if args.psf_dir else 'Gaussian'}")
    basis_info = None
    if any(k in LATENT_SFH for k in FIELDS):
        SFH_BASIS, s_vals = sfh_basis(emulator, n=len(LATENT_SFH))
        basis_info = {
            "B": SFH_BASIS.tolist(),
            "singular_values": s_vals.tolist(),
            "sigma_per_pixel_at_peak": [float(1.0 / (args.snr * max(v, 1e-30))) for v in s_vals],
        }
        widths = np.array(basis_info["sigma_per_pixel_at_peak"])
        print(f"SFH basis (d ln F / d ratio at the neutral reference, fit emulator): singular "
              f"values {np.array2string(s_vals, precision=4)}; per-pixel 1-sigma at peak S/N "
              f"{args.snr:g}: {np.array2string(widths, precision=3)}")  # fmt: skip
        for k, name in enumerate(LATENT_SFH):
            print(f"  {name} = {np.array2string(SFH_BASIS[k], precision=3, sign='+')}")

    # ----- truth (TRUTH_FIELDS maps, the same for every --sfh setup)
    disk = disk_mass_map(n, log_mtot=10.0, rd_px=0.11 * n)
    offsets = HONEST_OFFSETS if args.honest else {}
    truth_means = {k: TRUTH_GLOBALS.get(k, v) + offsets.get(k, 0.0) for k, v in NEUTRAL.items()}
    truth_model = build_model(n, disk, mean_values=truth_means, fixed=FIXED_TRUTH,
                              sfh_prior="legacy", pinned=(), fields=TRUTH_FIELDS,
                              **truth_hyper_scales(args))  # fmt: skip
    arm_weight = None
    if args.truth == "prior":
        theta_truth = truth_model.sample_prior(jax.random.PRNGKey(args.seed), 1)[0]
        theta_truth = theta_truth.at[: truth_model.n_fields].set(0.0)  # around the means
        truth_field_maps = np.asarray(truth_model.field_maps(theta_truth))
        render_model = truth_model
    else:
        truth_field_maps, arm_weight = spiral_truth(n, rng)
        for k, dv in offsets.items():
            truth_field_maps[TRUTH_FIELDS.index(k)] += dv
        render_model = FixedMapsModel(truth_field_maps, truth_model)
        theta_truth = truth_model.initial_theta()
    obs, clean, sigma = make_observation(emulator, psf, render_model, theta_truth, args.snr, rng)
    # Truth by emulator-parameter name: its maps plus constant maps of its fixed values.
    truth_named = dict(zip(truth_model.field_names, truth_field_maps, strict=True))
    for k, v in FIXED_TRUTH.items():
        if k != "redshift" and k not in truth_named:
            truth_named[k] = np.full((n, n), v, np.float32)

    # ----- blind start: mass mean from the reddest band at neutral SPS values
    ref = {**NEUTRAL_GLOBALS, **NEUTRAL, **FIXED}
    # The mean map must stay smooth: structure in it is not penalised by the prior.
    template_info = None
    if args.mass_start == "template":
        ms = template_mass_map(obs, emulator, ref, [{}, *default_sfh_dust_templates()],
                               smooth_sigma_px=1.0)  # fmt: skip
        mass_mean = ms.log_mass
        counts = np.bincount(ms.template_index.ravel(), minlength=len(ms.templates))
        template_info = {
            "templates": ms.templates,
            "pixel_fraction": (counts / counts.sum()).tolist(),
            "correction_range": [float(ms.correction.min()), float(ms.correction.max())],
        }
        print(f"template mass start: correction [{ms.correction.min():+.3f}, "
              f"{ms.correction.max():+.3f}] dex; pixels per template "
              f"{counts.tolist()}")  # fmt: skip
    else:
        mass_mean = mass_map_from_image(obs, emulator, ref, band=len(BANDS) - 1,
                                        smooth_sigma_px=1.0)  # fmt: skip
    k_max = psf_k_max(psf, args.k_max_mtf) if args.k_max_mtf > 0 else None
    if k_max is not None:
        print(f"--k-max-mtf {args.k_max_mtf:g}: k_max = {k_max:.3f} rad/px")
    model = build_model(n, mass_mean, corr_scale=args.corr_scale, nu=args.nu,
                        hyper=args.hyper, couple=args.couple, k_max=k_max)  # fmt: skip
    print(model.summary())
    COMPARE_FIELDS = compare_names(model)
    truth_maps = stack_compare(truth_named)  # (N_compare, H, W)
    fm = ForwardModel.build(obs, psf, model, emulator)
    theta0 = model.initial_theta(global_values=NEUTRAL_GLOBALS)
    if args.mass_start == "wiener":
        # Sharp detail enters the *starting coefficients*, through the field prior.
        theta0 = model.project_to_field(
            theta0, "log_mass", wiener_mass_map(obs, psf, emulator, ref)
        )
    start_maps = stack_compare(fit_param_maps(model, theta0))

    weight = 10.0 ** truth_named["log_mass"]
    if args.laplace and not args.float64:
        print("WARNING: --laplace without --float64; the Hessian will be float32-noisy")
    pstd_model = prior_map_std(model, 400, jax.random.PRNGKey(args.seed + 200))
    pstd = in_fields_order(model, pstd_model)
    starts = [theta0] + [perturbed_start(model, theta0, jax.random.PRNGKey(args.seed + 1000 + i))
                         for i in range(1, args.n_starts)]  # fmt: skip
    runs = []
    for i, ts in enumerate(starts):
        if len(starts) > 1:
            print(f"===== start {i + 1}/{len(starts)}", flush=True)
        t0 = time.time()
        r, block_log = map_stage(fm, ts, args.rounds, args.steps, block=args.block_newton)
        rt = time.time() - t0
        p = None
        if args.laplace:
            p = laplace_posterior(fm, model, r.theta, args.n_laplace, args.hess_chunk,
                                  jax.random.PRNGKey(args.seed + 100),
                                  max_newton=args.max_newton, precond=args.precond)  # fmt: skip
            summ = mode_summary(model, p, truth_maps, pstd, weight)
            print(f"  start {i + 1}: -log_post {summ['neg_log_post']:.2f}, log Z "
                  f"{summ['log_evidence_laplace']:.2f}, converged {summ['newton_converged']}, "
                  f"{summ['n_negative']} neg. eig.", flush=True)  # fmt: skip
        elif vi:
            p = vi_posterior(fm, model, r.theta, jax.random.PRNGKey(args.seed + 100),
                             args.posterior, args.vi_iterations, args.vi_samples,
                             args.n_laplace)  # fmt: skip
            summ = mode_summary(model, p, truth_maps, pstd, weight)
            print(f"  start {i + 1}: -log_post(VI mean) {summ['neg_log_post']:.2f}, "
                  f"{p['runtime_s']:.0f} s", flush=True)  # fmt: skip
        else:
            summ = {"neg_log_post": float(r.neg_log_posterior)}
        summ["map_neg_log_post"] = float(r.neg_log_posterior)
        runs.append({"res": r, "runtime": rt, "post": p, "summary": summ, "block": block_log})
        if i > 0 and p is not None:
            # Keep the heavy arrays only for the best mode so far.
            best_so_far = runs[select_best(runs)]
            for x in runs:
                if x is not best_so_far and x["post"] is not None:
                    x["post"] = {"theta": x["post"]["theta"]}
    i_best = select_best(runs)
    res, runtime = runs[i_best]["res"], runs[i_best]["runtime"]
    best_post = runs[i_best]["post"]
    fit_maps = stack_compare(fit_param_maps(model, res.theta))
    start_latent, fit_latent = latent_maps(model, theta0), latent_maps(model, res.theta)
    model_img = np.asarray(fm._model_image(res.theta))
    chi = (np.asarray(obs.flux) - model_img) / sigma[:, None, None]

    metrics = {
        "truth": args.truth,
        "emulator": args.emulator or "toy",
        "mass_start": args.mass_start,
        "bands": BANDS,
        "corr_scale": args.corr_scale,
        "nu": args.nu,
        "hyper": args.hyper,
        "couple": args.couple,
        "honest": args.honest,
        "sfh": args.sfh,
        "sfh_prior": args.sfh_prior,
        "free_globals": args.free_globals,
        "precond": args.precond,
        "n_starts": args.n_starts,
        "best_start": i_best,
        "best_start_rule": "lowest neg_log_post among converged modes with 0 negative "
        "eigenvalues, else lowest neg_log_post (VI: -log p at the VI mean)",
        "posterior": args.posterior,
        "block_newton": args.block_newton,
        "block_newton_log": runs[i_best]["block"],
        "seed": args.seed,
        "truth_offsets": offsets,
        "truth_hyper_scales": truth_hyper_scales(args),
        "fields": FIELDS,
        "truth_fields": TRUTH_FIELDS,
        "compare_fields": COMPARE_FIELDS,
        "param_maps": list(model.param_map_names),
        "derived": list(model.derived_names),
        "fixed_fit": {k: float(v) for k, v in FIXED.items()},
        "truth_globals": {k: v for k, v in FIXED_TRUTH.items() if k != "redshift"},
        "global_names": list(model.global_names),
        "map_scalars": scalar_summary(model, res.theta[None, :]),
        "psf": args.psf_dir or "gaussian",
        "size_px": n,
        "snr_peak": args.snr,
        "n_params": model.n_params,
        "k_max_mtf": args.k_max_mtf,
        "k_max": k_max,
        "modes_per_field": {f.name: f.n_modes for f in model.fields},
        "neg_log_post_start": float(-fm.log_posterior(theta0)),
        "neg_log_post_map": float(res.neg_log_posterior),
        "chi2_red_map": float(np.mean(chi**2)),
        "chi2_red_per_band": [float(np.mean(c**2)) for c in chi],
        "map_runtime_s": runtime,
        "adam_steps": res.n_steps,
        "start_vs_truth": map_metrics(truth_maps, start_maps, weight),
        "map_vs_truth": map_metrics(truth_maps, fit_maps, weight),
        "total_log_mass_truth": float(np.log10(np.sum(10.0 ** truth_maps[0]))),
        "total_log_mass_map": float(np.log10(np.sum(10.0 ** fit_maps[0]))),
    }
    if basis_info is not None:
        metrics["sfh_basis"] = basis_info
    if template_info is not None:
        metrics["template_mass_start"] = template_info
    if args.truth == "prior":
        metrics["neg_log_post_truth_under_truth_model"] = float(
            -ForwardModel.build(obs, psf, truth_model, emulator).log_posterior(theta_truth)
        )
    post = nuts = None
    init_thetas = None
    if best_post is not None:
        post = best_post
        stage = "laplace" if args.laplace else "vi"
        nlp_best = runs[i_best]["summary"]["neg_log_post"]
        if len(runs) > 1:
            if args.laplace:
                from jax.scipy.linalg import solve_triangular

                chol = jnp.asarray(post["chol"])

                def dist(t):
                    return solve_triangular(chol, t - post["theta"], lower=True)
            else:
                tr, xi_best = post["result"].transform, post["result"].xi_mean

                def dist(t):
                    return (np.asarray(tr.from_theta(jnp.asarray(t))) - xi_best) / post["scale"]

            for x in runs:
                dz = dist(x["post"]["theta"])
                x["summary"]["whitened_distance_to_best"] = float(np.linalg.norm(dz))
            metrics["modes"] = [x["summary"] for x in runs]
        near = [runs[i_best]["post"]["theta"]] + [
            x["post"]["theta"] for j, x in enumerate(runs)
            if j != i_best and x["summary"]["neg_log_post"] < nlp_best + 10.0
        ]  # fmt: skip
        if len(near) > 1:
            init_thetas = np.stack([np.asarray(near[c % len(near)])
                                    for c in range(args.nuts_chains)])  # fmt: skip
        post["maps"] = in_fields_order(model, post["maps"])
        pm, med, std, info = posterior_metrics(truth_maps, post["maps"], pstd, weight)
        drop = ("theta", "maps", "chol", "thetas", "result", "scale")
        pmet = metrics[stage] = {k: v for k, v in post.items() if k not in drop}
        pmet["n_draws"] = int(post["thetas"].shape[0])
        pmet["scalars"] = scalar_summary(model, post["thetas"], TRUTH_GLOBALS)
        pmet["per_field"] = pm
        mass_tot = np.log10(np.sum(10.0 ** post["maps"][:, 0], axis=(1, 2)))
        pmet["total_log_mass_16_50_84"] = [float(x) for x in np.percentile(mass_tot, [16, 50, 84])]
        name = "laplace" if args.laplace else args.posterior
        plot_laplace(truth_maps, med, std, info, weight, outdir / "figures" / f"{name}.png")
        np.savez(outdir / f"{name}.npz", median=med, std=std, information=info,
                 theta=np.asarray(post["theta"]), fields=np.array(COMPARE_FIELDS))  # fmt: skip
        if args.nuts:
            key = jax.random.PRNGKey(args.seed + 300)
            density = z_starts = None
            if vi:
                from arachne.inference.standardise import StandardisedLogDensity

                density = StandardisedLogDensity(fm, post["result"].transform)
                z_starts = vi_chain_starts(density, near, post["scale"],
                                           jax.random.PRNGKey(args.seed + 301),
                                           args.nuts_chains)  # fmt: skip
            if args.sampler == "mclmc":
                nuts = mclmc_posterior(fm, model, post["theta"], post.get("chol"), key,
                                       args.mclmc_warmup, args.mclmc_samples, args.nuts_chains,
                                       init_thetas=init_thetas, n_burn=args.mclmc_burn,
                                       diag=args.mclmc_diag, whitened=density,
                                       z_starts=z_starts,
                                       z_tune=None if density is None
                                       else post["result"].xi_mean)  # fmt: skip
            elif vi:
                theta_starts = np.asarray(density.to_theta(jnp.asarray(z_starts)))
                nuts = nuts_posterior(fm, model, post["theta"], None, key, args.nuts_warmup,
                                      args.nuts_samples, args.nuts_chains,
                                      init_thetas=theta_starts, whitened=density)  # fmt: skip
            else:
                nuts = nuts_posterior(fm, model, post["theta"], post["chol"], key,
                                      args.nuts_warmup, args.nuts_samples, args.nuts_chains,
                                      init_thetas=init_thetas)  # fmt: skip
            nuts["maps"] = in_fields_order(model, nuts["maps"])
            npm, nmed, nstd, ninfo = posterior_metrics(truth_maps, nuts["maps"], pstd, weight)
            mc = metrics[args.sampler] = {
                k: v for k, v in nuts.items() if k not in ("maps", "theta_samples")
            }
            mc["after"] = args.posterior
            mc["scalars"] = scalar_summary(model, nuts["theta_samples"], TRUTH_GLOBALS)
            mc["per_field"] = npm
            nmass = np.log10(np.sum(10.0 ** nuts["maps"][:, 0], axis=(1, 2)))
            mc["total_log_mass_16_50_84"] = [float(x) for x in np.percentile(nmass, [16, 50, 84])]
            fig_path = outdir / "figures" / f"{args.sampler}.png"
            plot_laplace(truth_maps, nmed, nstd, ninfo, weight, fig_path)
            np.savez(outdir / f"{args.sampler}.npz", median=nmed, std=nstd, information=ninfo,
                     fields=np.array(COMPARE_FIELDS))  # fmt: skip
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    latent_kw = {f"latent_{k}_{w}": v for w, d in (("start", start_latent), ("map", fit_latent))
                 for k, v in d.items()}  # fmt: skip
    np.savez(outdir / "maps.npz", truth=truth_maps, start=start_maps, map=fit_maps,
             model_image=model_img, data=np.asarray(obs.flux), sigma=sigma,
             theta_map=np.asarray(res.theta), fields=np.array(COMPARE_FIELDS),
             **latent_kw)  # fmt: skip
    plot_maps(truth_maps, start_maps, fit_maps, weight, outdir / "figures" / "maps.png",
              latent={k: (start_latent[k], fit_latent[k]) for k in fit_latent})  # fmt: skip
    plot_images(obs, model_img, sigma, outdir / "figures" / "images.png")

    if not args.no_bundle:
        try:
            truth = TruthContent(
                convolved=clean,
                intrinsic=np.asarray(render_model.model_image(theta_truth, emulator, (n, n))),
                # aligned by name with the bundle's maps (model.param_map_names)
                param_maps=np.stack([truth_named[k] for k in model.param_map_names]),
            )
            common = dict(fm=fm, model=model, args=args, truth=truth, truth_maps=truth_maps,
                          regions=demo_regions(truth_maps, arm_weight), prior_std=pstd_model,
                          metrics=metrics)  # fmt: skip
            post_diag = {}
            if post is not None and args.laplace:
                post_diag = {"log_evidence_laplace": post["log_evidence_laplace"]}
            elif post is not None:
                post_diag = {"neg_log_post_vi_mean": post["neg_log_post_mean"]}
            laplace_kw = dict(
                theta_map=post["theta"] if post else None,
                samples=post["thetas"] if post else None,
                source=args.posterior,
                n_chains=None,
                diagnostics=post_diag,
            )
            if nuts is not None:
                nm = metrics[args.sampler]
                diag = {k: nm[k] for k in ("max_rhat", "frac_rhat_gt_1.01", "min_ess",
                                            "median_ess")}  # fmt: skip
                diag.update(post_diag)
                path = write_demo_bundle(outdir / "results.h5", theta_map=post["theta"],
                                         samples=nuts["theta_samples"], source=args.sampler,
                                         n_chains=nuts["n_chains"], diagnostics=diag,
                                         **common)  # fmt: skip
                if args.bundle_all:
                    write_demo_bundle(outdir / f"results_{args.posterior}.h5", **laplace_kw,
                                      **common)  # fmt: skip
            elif post is not None:
                path = write_demo_bundle(outdir / "results.h5", **laplace_kw, **common)
            else:
                path = write_demo_bundle(outdir / "results.h5", theta_map=res.theta,
                                         samples=None, source="map", n_chains=None,
                                         diagnostics={}, **common)  # fmt: skip
            if args.bundle_plots:
                bundle_plots(path, outdir / "figures" / "bundle")
        except Exception:  # noqa: BLE001 - never lose a long run's other outputs
            import traceback

            traceback.print_exc()
            print("WARNING: writing the results bundle failed (see traceback above)")

    print(f"MAP in {runtime:.0f} s ({res.n_steps} steps); chi2_red = {metrics['chi2_red_map']:.3f}")
    for name in COMPARE_FIELDS:
        s, m = metrics["start_vs_truth"][name], metrics["map_vs_truth"][name]
        print(f"  {name:18s} r: {s['pearson_r']:+.2f} -> {m['pearson_r']:+.2f}   "
              f"|err|: {s['median_abs_err']:.3f} -> {m['median_abs_err']:.3f}   "
              f"bias {m['bias']:+.3f}   truth rms {m['truth_rms']:.3f}")  # fmt: skip
    print(f"  total log M*: truth {metrics['total_log_mass_truth']:.3f}, "
          f"MAP {metrics['total_log_mass_map']:.3f}")  # fmt: skip
    for b in metrics["block_newton_log"]:
        print(f"  {b['stage']:8s}: -log_post after Adam {b['neg_log_post_adam']:.2f}, after "
              f"block Newton {b['neg_log_post_newton']:.2f} ({b['newton_reason']})")  # fmt: skip
    if post is not None and vi:
        vm = metrics["vi"]
        print(f"{args.posterior}: {vm['runtime_s']:.0f} s, {len(vm['history'])} iterations; "
              f"-log_post at the VI mean {vm['neg_log_post_mean']:.2f} (MAP "
              f"{metrics['neg_log_post_map']:.2f}); {vm['n_draws']} draws")  # fmt: skip
        for h in vm["history"]:
            print(f"  iter {h['iteration']}: KL~{h['kl']:.2f}  -log_post(mean) "
                  f"{h['neg_log_post_mean']:.2f}  {h['time_s']:.0f} s")  # fmt: skip
        for name in COMPARE_FIELDS:
            q = vm["per_field"][name]
            print(f"  {name:18s} cov68 {q['coverage68']:.2f}  cov95 {q['coverage95']:.2f}  "
                  f"z_rms {q['z_rms']:.2f}  post std {q['median_post_std']:.3f}  "
                  f"info {q['median_information']:.2f}")  # fmt: skip
        m16, m50, m84 = vm["total_log_mass_16_50_84"]
        print(f"  total log M* {m50:.3f} +{m84 - m50:.3f} -{m50 - m16:.3f} "
              f"(truth {metrics['total_log_mass_truth']:.3f})")  # fmt: skip
    if post is not None and args.laplace:
        lp = metrics["laplace"]
        print(f"Laplace: TR-Newton {lp['newton_s']:.0f} s ({lp['newton_reason']}, "
              f"{lp['newton_hvps']} HVPs), Hessian {lp['hessian_s']:.0f} s; "
              f"eigenvalues [{lp['eig_min']:.3g}, {lp['eig_max']:.3g}], "
              f"{lp['n_negative']} neg.")  # fmt: skip
        h0, h1 = lp["newton_history"][0], lp["newton_history"][-1]
        print(f"  TR-Newton {len(lp['newton_history']) - 1} steps: -log_post "
              f"{h0['neg_log_post']:.2f} -> {h1['neg_log_post']:.2f}, |grad| "
              f"{h0['grad_norm']:.3g} -> {h1['grad_norm']:.3g}")  # fmt: skip
        for name in COMPARE_FIELDS:
            q = lp["per_field"][name]
            print(f"  {name:18s} cov68 {q['coverage68']:.2f}  cov95 {q['coverage95']:.2f}  "
                  f"z_rms {q['z_rms']:.2f}  post std {q['median_post_std']:.3f}  "
                  f"info {q['median_information']:.2f}")  # fmt: skip
        print(f"  Laplace log evidence {lp['log_evidence_laplace']:.2f}")
        m16, m50, m84 = lp["total_log_mass_16_50_84"]
        print(f"  total log M* {m50:.3f} +{m84 - m50:.3f} -{m50 - m16:.3f} "
              f"(truth {metrics['total_log_mass_truth']:.3f})")  # fmt: skip
    if args.sampler in metrics:
        nm = metrics[args.sampler]
        print(f"{args.sampler.upper()}: {nm['runtime_s']:.0f} s; max R-hat {nm['max_rhat']:.3f} "
              f"({100 * nm['frac_rhat_gt_1.01']:.1f}% > 1.01); ESS min {nm['min_ess']:.0f}, "
              f"median {nm['median_ess']:.0f}")  # fmt: skip
        for name in COMPARE_FIELDS:
            q = nm["per_field"][name]
            print(f"  {name:18s} cov68 {q['coverage68']:.2f}  cov95 {q['coverage95']:.2f}  "
                  f"z_rms {q['z_rms']:.2f}  post std {q['median_post_std']:.3f}")  # fmt: skip
        m16, m50, m84 = nm["total_log_mass_16_50_84"]
        print(f"  total log M* {m50:.3f} +{m84 - m50:.3f} -{m50 - m16:.3f}")
    if "modes" in metrics:
        rule = (
            "lowest -log_post at the VI mean"
            if vi
            else "lowest -log_post among "
            "converged modes without negative eigenvalues; log Z is not used"
        )
        print(f"modes ({len(runs)} starts; best = start {i_best + 1}, {rule}):")
        for i, m in enumerate(metrics["modes"]):
            c = " ".join(f"{SHORT.get(k, k)}={v:.2f}" for k, v in m["coverage68"].items())
            lz = m["log_evidence_laplace"]
            print(f"  {i + 1}: -log_post {m['neg_log_post']:.1f}  conv {m['newton_converged']}  "
                  f"neg {m['n_negative']}  log Z {'-' if lz is None else f'{lz:.1f}'}"
                  f"  |dz| {m['whitened_distance_to_best']:.1f}  M* "
                  f"{m['total_log_mass_16_50_84'][1]:.3f}  cov68 {c}")  # fmt: skip
    for stage in ("laplace", "vi", args.sampler):
        sc = metrics.get(stage, {}).get("scalars")
        if sc:
            print(f"{stage} scalars (16/50/84, truth):")
            for name, v in sc.items():
                q = v["p16_50_84"]
                tr = v.get("truth")
                print(f"  {name:28s} {q[1]:+.3f} [{q[0]:+.3f}, {q[2]:+.3f}]"
                      + (f"  truth {tr:+.3f}" if tr is not None else ""))  # fmt: skip
    print(f"outputs in {outdir}")


if __name__ == "__main__":
    main()
