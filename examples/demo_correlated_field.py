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

Mapped parameters: log_mass (per pixel), Av, log10metallicity, logsfr_ratio_0.
Everything else is fixed at neutral values (redshift = 2).

Start (no truth used): log-mass mean map from the smoothed reddest band at
neutral SPS values (``mass_map_from_image``).  With ``--mass-start wiener`` the
mass-field coefficients start from a ridge projection
(``project_to_field``) of a multi-band Wiener deconvolution
(``wiener_mass_map``) -- detail goes into the coefficients, which the prior
penalises, never into the mean map, which it does not.  (On the toy spiral the
Wiener start gave no measurable gain, so ``image`` is the default.)  Other means at neutral
constants; zero levels.  Then Adam MAP (``find_map``) and,
with ``--laplace``, damped Newton to the mode plus a Laplace posterior
(Hessian, Gaussian map samples, coverage of the truth, information maps and
the Laplace evidence).

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
    ObservationCube,
    PSFModel,
    SPSEmulator,
    find_map,
    mass_map_from_image,
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
SFH_PRIOR = "bursty"
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
}


SHORT = {"log_mass": "M", "Av": "Av", "log10metallicity": "Z",
         **{f"logsfr_ratio_{i}": f"r{i}" for i in range(5)}}  # fmt: skip


# Mapped: mass, A_V, Z, the 0-3/3-10 Myr ratio and the 64-411 Myr/0.4-2.6 Gyr ratio (age
# structure).  Free globals with library priors: dust slope, bump, fesc_lya, ratios 1/2/4.
# The truth uses TRUTH_GLOBALS; the fit starts from NEUTRAL_GLOBALS.
FIELDS = ["log_mass", "Av", "log10metallicity", "logsfr_ratio_0", "logsfr_ratio_3"]
FIXED = {"redshift": REDSHIFT}
FIXED_TRUTH = {"redshift": REDSHIFT, **TRUTH_GLOBALS}
SFH_SETUPS = ("map03", "fixed_truth", "map_all")


def set_sfh(name):
    """SFH setup (module globals FIELDS, FIXED, FIXED_TRUTH).

    ``map03``: ratios 0 and 3 mapped, 1/2/4 free globals.  ``fixed_truth``: as map03 but the
    fit holds ratios 1/2/4 at their true values (diagnostic for the bin-trading degeneracy).
    ``map_all``: every ratio mapped; the truths of 1/2/4 are fields around TRUTH_GLOBALS
    (constant maps for the spiral).
    """
    global FIELDS, FIXED, FIXED_TRUTH
    ratios = [0, 1, 2, 3, 4] if name == "map_all" else [0, 3]
    FIELDS = ["log_mass", "Av", "log10metallicity"] + [f"logsfr_ratio_{i}" for i in ratios]
    FIXED = {"redshift": REDSHIFT}
    if name == "fixed_truth":
        FIXED.update({k: v for k, v in TRUTH_GLOBALS.items() if k.startswith("logsfr_ratio")})
    FIXED_TRUTH = {"redshift": REDSHIFT,
                   **{k: v for k, v in TRUTH_GLOBALS.items() if k not in FIELDS}}  # fmt: skip


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
    """Face-on spiral: maps (len(FIELDS), n, n) in FIELDS order and the arm weight (n, n).

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
    stack = [maps[k] if k in maps else np.full_like(r, TRUTH_GLOBALS[k]) for k in FIELDS]
    return np.stack(stack).astype(np.float32), win * arm


def build_model(n, mass_mean, mean_values=None, corr_scale=1.0, nu=1.5, fixed=None,
                hyper=False, couple=False, amp_scale=1.0, sfh_prior=None):  # fmt: skip
    """CorrelatedFieldMap with the demo's hyperparameters.

    ``corr_scale`` multiplies every correlation length; ``nu`` is the Matérn slope of the
    non-mass fields (mass keeps 1.5); ``amp_scale`` multiplies every amplitude.  ``fixed``
    defaults to FIXED (the fit); pass FIXED_TRUTH for the truth model.  ``hyper`` fits
    amplitude and correlation length of every non-mass field (log-normal priors around the
    defaults); ``couple`` fits A_V's
    response to the mass and SFH-ratio deviations (N(0, 0.5^2) each).  ``sfh_prior``
    (default SFH_PRIOR) sets the SFH-ratio priors; the truth model passes ``"legacy"`` so
    the mocks do not depend on the fit's prior.
    """
    fixed = FIXED if fixed is None else fixed
    sfh_prior = SFH_PRIOR if sfh_prior is None else sfh_prior
    spec = {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": SFH_PRIOR_SCALES[sfh_prior]}
    ratios = [p for p in SPS_PARAM_NAMES if p.startswith("logsfr_ratio")]
    marginals = None
    if sfh_prior != "legacy":
        marginals = {p: spec for p in ratios if p in FIELDS}
    means = {"log_mass": mass_mean}
    means.update({k: v for k, v in (mean_values or NEUTRAL).items() if k in FIELDS})
    free_globals = [p for p in SPS_PARAM_NAMES if p not in FIELDS and p not in fixed]
    ells = {k: corr_scale * v for k, v in CORR_LENGTHS.items() if k in FIELDS}
    hyper_priors = None
    if hyper:
        hyper_priors = {
            k: {
                "log_amplitude": {"loc": float(np.log(AMPLITUDES[k])), "scale": 0.5},
                "log_corr_length": {"loc": float(np.log(ells[k])), "scale": 0.25},
            }
            for k in FIELDS
            if k != "log_mass"
        }
    free_couplings = None
    if couple:
        sources = ["log_mass"] + [k for k in FIELDS if k.startswith("logsfr_ratio")]
        # A_V must come after its sources: reorder fields so it is last.
        free_couplings = {"Av": {s: 0.5 for s in sources}}
    field_order = [k for k in FIELDS if k != "Av"] + ["Av"] if couple else FIELDS
    return CorrelatedFieldMap(
        image_shape=(n, n),
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        field_params=field_order,
        mass_param="log_mass",
        fixed_params=fixed,
        corr_lengths=ells,
        nus={k: (1.5 if k == "log_mass" else nu) for k in FIELDS},
        amplitudes={k: amp_scale * v for k, v in AMPLITUDES.items() if k in FIELDS},
        level_sigmas={k: v for k, v in LEVEL_SIGMAS.items() if k in FIELDS},
        mean_maps=means,
        field_marginals=marginals,
        variance_fraction=0.99,
        pixel_scale=PIXEL_SCALE,
        hyper_priors=hyper_priors,
        free_couplings=free_couplings,
        global_priors={p: spec for p in ratios if p in free_globals},
    )


def in_fields_order(model, maps_model_order):
    """Reorder (…, N_fields, H, W) maps from ``model.field_names`` order to FIELDS order."""
    idx = [model.field_names.index(f) for f in FIELDS]
    return np.take(np.asarray(maps_model_order), idx, axis=-3)


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
    """Per-field Pearson r, light-weighted bias and median |error| over the galaxy."""
    out = {}
    sel = weight > 0.02 * weight.max()
    for k, name in enumerate(FIELDS):
        t, f = truth[k][sel], fit[k][sel]
        out[name] = {
            "pearson_r": float(np.corrcoef(t, f)[0, 1]) if np.std(f) > 0 else float("nan"),
            "median_abs_err": float(np.median(np.abs(f - t))),
            "bias": float(np.average(f - t, weights=weight[sel])),
            "truth_rms": float(np.std(t)),
        }
    return out


def plot_maps(truth, start, fit, weight, path):
    """Rows: fields.  Columns: truth, start (mean map), MAP, MAP - truth."""
    sel = weight > 0.02 * weight.max()
    fig, axes = plt.subplots(len(FIELDS), 4, figsize=(9.5, 2.3 * len(FIELDS)))
    for k, name in enumerate(FIELDS):
        lo, hi = np.percentile(truth[k][sel], [1, 99])
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
    its cap; ``diag`` uses a Hutchinson diagonal; ``none`` the identity.  ``M0`` is a
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

    M = M0 if M0 is not None else (refresh(theta) if refresh else None)
    res = trust_region_newton(neg_lp, theta, preconditioner=M, refresh_preconditioner=refresh,
                              refresh_every=20, max_iter=max_iter, tol=tol,
                              print_fn=lambda m: print(m, flush=True))  # fmt: skip
    print(res.summary(), flush=True)
    return res


def laplace_posterior(fm, model, theta_map, n_samples, chunk, key, max_newton=100,
                      precond="dense", max_escapes=3):  # fmt: skip
    """Trust-region Newton from the MAP to the mode, one full Hessian there, Gaussian samples.

    After each Newton run the Hessian is checked for negative eigenvalues; if a step along
    those directions lowers ``-log p`` the polish restarts from there (up to
    ``max_escapes`` times), preconditioned by that Hessian.

    Run with ``--float64``: the Hessian of a 1e4-pixel likelihood is round-off
    dominated in float32.

    Returns:
        Dict with the polished theta, map samples (S, N_fields, H, W), the
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
    maps = np.asarray(jax.lax.map(model.field_maps, thetas, batch_size=32))
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
                   init_thetas=None):  # fmt: skip
    """Whitened NUTS (theta = mode + L z, L from the Laplace Hessian), diagonal metric in z.

    A dense metric cannot be adapted at d ~ 4e3, so the Laplace whitening does the
    conditioning and the warmup only adapts a diagonal.  ``init_thetas`` ``(n_chains, d)``
    starts each chain at its own point (modes from different optimiser starts).  Returns map samples
    (thinned to ``max_maps``) and convergence diagnostics in theta coordinates.
    """
    from arachne.inference.laplace import WhitenedLogDensity, run_whitened_nuts

    dtype = jnp.zeros(()).dtype
    whitened = WhitenedLogDensity(fm, jnp.asarray(theta, dtype), jnp.asarray(chol, dtype))
    t0 = time.time()
    result, _ = run_whitened_nuts(
        fm, theta, key, whitened=whitened, n_warmup=n_warmup, n_samples=n_samples,
        n_chains=n_chains, dense_mass_matrix=False, initial_positions=init_thetas,
    )  # fmt: skip
    runtime = time.time() - t0
    samples = np.asarray(result.samples)
    step = max(1, samples.shape[0] // max_maps)
    maps = np.asarray(jax.lax.map(model.field_maps, jnp.asarray(samples[::step]), batch_size=32))
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


def mclmc_posterior(fm, model, theta, chol, key, n_warmup, n_samples, n_chains, max_maps=1000,
                    init_thetas=None):  # fmt: skip
    """Whitened MCLMC: as ``nuts_posterior`` but blackjax MCLMC, one tuned run per chain.

    Chains start at z ~ N(0, 0.3^2) around the mode, or at ``init_thetas`` when given;
    R-hat and ESS are computed in theta.
    """
    from arachne.inference.diagnostics import ess, split_rhat
    from arachne.inference.laplace import WhitenedLogDensity
    from arachne.inference.mclmc_sampler import MCLMCSampler

    dtype = jnp.zeros(()).dtype
    whitened = WhitenedLogDensity(fm, jnp.asarray(theta, dtype), jnp.asarray(chol, dtype))
    sampler = MCLMCSampler(whitened, n_warmup=n_warmup, n_samples=n_samples)
    t0 = time.time()
    chains = []
    for c, k in enumerate(jax.random.split(key, n_chains)):
        k_init, k_run = jax.random.split(k)
        z0 = 0.3 * jax.random.normal(k_init, (whitened.n_params,), dtype)
        if init_thetas is not None:
            z0 = whitened.from_theta(jnp.asarray(init_thetas[c], dtype))
        res = sampler.run(z0, k_run)
        chains.append(np.asarray(jax.vmap(whitened.to_theta)(res.samples)))
        print(f"MCLMC chain {c + 1}/{n_chains}: {time.time() - t0:.0f} s")
    runtime = time.time() - t0
    chains = np.stack(chains)
    rhat, ess_ = split_rhat(chains), ess(chains)
    samples = chains.reshape(-1, chains.shape[-1])
    step = max(1, samples.shape[0] // max_maps)
    maps = np.asarray(jax.lax.map(model.field_maps, jnp.asarray(samples[::step]), batch_size=32))
    return {
        "maps": maps,
        "theta_samples": samples[::step],
        "n_chains": n_chains,
        "runtime_s": runtime,
        "max_rhat": float(np.nanmax(rhat)),
        "frac_rhat_gt_1.01": float(np.mean(rhat > 1.01)),
        "min_ess": float(np.nanmin(ess_)),
        "median_ess": float(np.nanmedian(ess_)),
    }


def prior_map_std(model, n, key):
    """Per-pixel prior std of each map's *local* structure (levels held at zero).

    The level prior is deliberately broad (e.g. 1 dex in mass), so including it
    would make every pixel look ~100% data-dominated.  Holding the levels fixed
    measures how much the data constrain the spatial fluctuations themselves.
    """
    thetas = model.sample_prior(key, n).at[:, : model.n_fields].set(0.0)
    maps = np.asarray(jax.lax.map(model.field_maps, thetas, batch_size=32))
    return maps.std(0)


def posterior_metrics(truth, samples, prior_std, weight):
    """Coverage of the truth by 68/95% intervals, z-scores and information per field."""
    sel = weight > 0.02 * weight.max()
    lo68, med, hi68 = np.percentile(samples, [16, 50, 84], axis=0)
    lo95, hi95 = np.percentile(samples, [2.5, 97.5], axis=0)
    std = samples.std(0)
    info = 1.0 - (std / prior_std) ** 2
    out = {}
    for k, name in enumerate(FIELDS):
        t = truth[k][sel]
        z = (med[k][sel] - t) / std[k][sel]
        out[name] = {
            "coverage68": float(np.mean((t >= lo68[k][sel]) & (t <= hi68[k][sel]))),
            "coverage95": float(np.mean((t >= lo95[k][sel]) & (t <= hi95[k][sel]))),
            "z_rms": float(np.sqrt(np.mean(z**2))),
            "median_post_std": float(np.median(std[k][sel])),
            "median_information": float(np.median(info[k][sel])),
        }
    return out, med, std, info


def plot_laplace(truth, med, std, info, weight, path):
    """Rows: fields.  Columns: truth, posterior median, posterior std, z-score, information."""
    sel = weight > 0.02 * weight.max()
    fig, axes = plt.subplots(len(FIELDS), 5, figsize=(12, 2.3 * len(FIELDS)))
    for k, name in enumerate(FIELDS):
        lo, hi = np.percentile(truth[k][sel], [1, 99])
        cmap = "magma" if k == 0 else "viridis"
        z = (med[k] - truth[k]) / std[k]
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
            out.append(float(np.log10(np.sum(10.0 ** truth_maps[FIELDS.index("log_mass")]))))
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
    """``galaxy`` (the metrics selection), plus ``arms``/``interarm`` for the spiral."""
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
    """Compact per-start record: optimiser status, evidence, coverage, total mass, globals."""
    maps = in_fields_order(model, post["maps"])
    pm = posterior_metrics(truth_maps, maps, pstd, weight)[0]
    mass_tot = np.log10(np.sum(10.0 ** maps[:, 0], axis=(1, 2)))
    sc = scalar_summary(model, post["thetas"], TRUTH_GLOBALS)
    return {
        "neg_log_post": float(post["newton_history"][-1]["neg_log_post"]),
        "newton_converged": post["newton_converged"],
        "newton_reason": post["newton_reason"],
        "log_evidence_laplace": post["log_evidence_laplace"],
        "n_negative": post["n_negative"],
        "total_log_mass_16_50_84": [float(x) for x in np.percentile(mass_tot, [16, 50, 84])],
        "coverage68": {k: v["coverage68"] for k, v in pm.items()},
        "global_medians": {
            k: v["p16_50_84"][1]
            for k, v in sc.items()
            if not k.startswith(("amplitude", "corr_length", "c["))
        },  # fmt: skip
    }


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
        choices=["wiener", "image"],
        default="image",
        help="start: smoothed-F444W mean (image) plus field coefficients projected "
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
    ap.add_argument("--sfh", choices=SFH_SETUPS, default="map03",
                    help="which SFH ratios are mapped / free / fixed (see set_sfh)")  # fmt: skip
    ap.add_argument("--tag", default="", help="suffix for the output directory name")
    ap.add_argument("--laplace", action="store_true", help="Laplace posterior at the MAP")
    ap.add_argument("--n-laplace", type=int, default=400, help="Laplace samples")
    ap.add_argument("--nuts", action="store_true",
                    help="whitened MCMC (--sampler) after the Laplace step")  # fmt: skip
    ap.add_argument("--sampler", choices=["nuts", "mclmc"], default="nuts")
    ap.add_argument("--mclmc-warmup", type=int, default=2000, help="MCLMC tuning steps")
    ap.add_argument("--mclmc-samples", type=int, default=2000, help="MCLMC steps per chain")
    ap.add_argument("--nuts-warmup", type=int, default=1000)
    ap.add_argument("--nuts-samples", type=int, default=500)
    ap.add_argument("--nuts-chains", type=int, default=4)
    ap.add_argument("--hess-chunk", type=int, default=32, help="HVP rows per Hessian batch")
    ap.add_argument("--precond", choices=["dense", "diag", "none"], default="dense",
                    help="trust-region Newton preconditioner (see polish_mode)")  # fmt: skip
    ap.add_argument("--max-newton", type=int, default=100, help="trust-region Newton iterations")
    ap.add_argument("--n-starts", type=int, default=1,
                    help="optimiser starts: the neutral start plus prior redraws of the non-mass "
                    "levels and globals; with --laplace the best Laplace evidence is kept and "
                    "MCMC chains start at the modes within 10 nats of it")  # fmt: skip
    ap.add_argument("--sfh-prior", choices=sorted(SFH_PRIOR_SCALES), default="bursty",
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
    set_bands(args.bands)
    set_sfh(args.sfh)
    global SFH_PRIOR
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

    # ----- truth
    disk = disk_mass_map(n, log_mtot=10.0, rd_px=0.11 * n)
    offsets = HONEST_OFFSETS if args.honest else {}
    truth_means = {k: TRUTH_GLOBALS.get(k, v) + offsets.get(k, 0.0) for k, v in NEUTRAL.items()}
    truth_model = build_model(n, disk, mean_values=truth_means, fixed=FIXED_TRUTH,
                              sfh_prior="legacy", **truth_hyper_scales(args))  # fmt: skip
    arm_weight = None
    if args.truth == "prior":
        theta_truth = truth_model.sample_prior(jax.random.PRNGKey(args.seed), 1)[0]
        theta_truth = theta_truth.at[: truth_model.n_fields].set(0.0)  # around the means
        truth_maps = np.asarray(truth_model.field_maps(theta_truth))
        render_model = truth_model
    else:
        truth_maps, arm_weight = spiral_truth(n, rng)
        for k, dv in offsets.items():
            truth_maps[FIELDS.index(k)] += dv
        render_model = FixedMapsModel(truth_maps, truth_model)
        theta_truth = truth_model.initial_theta()
    obs, clean, sigma = make_observation(emulator, psf, render_model, theta_truth, args.snr, rng)

    # ----- blind start: mass mean from the reddest band at neutral SPS values
    ref = {**NEUTRAL_GLOBALS, **NEUTRAL, **FIXED}
    # The mean map must stay smooth: structure in it is not penalised by the prior.
    mass_mean = mass_map_from_image(obs, emulator, ref, band=len(BANDS) - 1, smooth_sigma_px=1.0)
    model = build_model(n, mass_mean, corr_scale=args.corr_scale, nu=args.nu,
                        hyper=args.hyper, couple=args.couple)  # fmt: skip
    print(model.summary())
    fm = ForwardModel.build(obs, psf, model, emulator)
    theta0 = model.initial_theta(global_values=NEUTRAL_GLOBALS)
    if args.mass_start == "wiener":
        # Sharp detail enters the *starting coefficients*, through the field prior.
        theta0 = model.project_to_field(
            theta0, "log_mass", wiener_mass_map(obs, psf, emulator, ref)
        )
    start_maps = in_fields_order(model, model.field_maps(theta0))

    weight = 10.0 ** truth_maps[0]
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
        r = find_map(fm, ts, n_rounds=args.rounds, steps_per_round=args.steps, lr=0.05,
                     final_steps=args.steps, final_lr=0.01)  # fmt: skip
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
        else:
            summ = {"neg_log_post": float(r.neg_log_posterior)}
        runs.append({"res": r, "runtime": rt, "post": p, "summary": summ})
        if i > 0 and p is not None:
            # Keep the heavy arrays only for the best mode so far.
            best_so_far = max(runs, key=lambda x: x["summary"]["log_evidence_laplace"])
            for x in runs:
                if x is not best_so_far and x["post"] is not None:
                    x["post"] = {"theta": x["post"]["theta"]}
    key_fn = ((lambda x: x["summary"]["log_evidence_laplace"]) if args.laplace
              else (lambda x: -x["summary"]["neg_log_post"]))  # fmt: skip
    i_best = max(range(len(runs)), key=lambda i: key_fn(runs[i]))
    res, runtime = runs[i_best]["res"], runs[i_best]["runtime"]
    best_post = runs[i_best]["post"]
    fit_maps = in_fields_order(model, model.field_maps(res.theta))
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
        "precond": args.precond,
        "n_starts": args.n_starts,
        "best_start": i_best,
        "seed": args.seed,
        "truth_offsets": offsets,
        "truth_hyper_scales": truth_hyper_scales(args),
        "fields": FIELDS,
        "truth_globals": {k: v for k, v in FIXED_TRUTH.items() if k not in FIXED},
        "map_scalars": scalar_summary(model, res.theta[None, :]),
        "psf": args.psf_dir or "gaussian",
        "size_px": n,
        "snr_peak": args.snr,
        "n_params": model.n_params,
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
    if args.truth == "prior":
        metrics["neg_log_post_truth_under_truth_model"] = float(
            -ForwardModel.build(obs, psf, truth_model, emulator).log_posterior(theta_truth)
        )
    post = nuts = None
    init_thetas = None
    if args.laplace:
        post = best_post
        if len(runs) > 1:
            from jax.scipy.linalg import solve_triangular

            chol = jnp.asarray(post["chol"])
            lz_best = runs[i_best]["summary"]["log_evidence_laplace"]
            for x in runs:
                dz = solve_triangular(chol, x["post"]["theta"] - post["theta"], lower=True)
                x["summary"]["whitened_distance_to_best"] = float(jnp.linalg.norm(dz))
            metrics["modes"] = [x["summary"] for x in runs]
            near = [x["post"]["theta"] for x in runs
                    if x["summary"]["log_evidence_laplace"] > lz_best - 10.0]  # fmt: skip
            if len(near) > 1:
                init_thetas = np.stack([np.asarray(near[c % len(near)])
                                        for c in range(args.nuts_chains)])  # fmt: skip
        post["maps"] = in_fields_order(model, post["maps"])
        pm, med, std, info = posterior_metrics(truth_maps, post["maps"], pstd, weight)
        metrics["laplace"] = {
            k: v for k, v in post.items() if k not in ("theta", "maps", "chol", "thetas")
        }
        metrics["laplace"]["scalars"] = scalar_summary(model, post["thetas"], TRUTH_GLOBALS)
        metrics["laplace"]["per_field"] = pm
        mass_tot = np.log10(np.sum(10.0 ** post["maps"][:, 0], axis=(1, 2)))
        metrics["laplace"]["total_log_mass_16_50_84"] = [
            float(x) for x in np.percentile(mass_tot, [16, 50, 84])
        ]
        plot_laplace(truth_maps, med, std, info, weight, outdir / "figures" / "laplace.png")
        np.savez(outdir / "laplace.npz", median=med, std=std, information=info,
                 theta=np.asarray(post["theta"]))  # fmt: skip
        if args.nuts:
            if args.sampler == "mclmc":
                nuts = mclmc_posterior(fm, model, post["theta"], post["chol"],
                                       jax.random.PRNGKey(args.seed + 300), args.mclmc_warmup,
                                       args.mclmc_samples, args.nuts_chains,
                                       init_thetas=init_thetas)  # fmt: skip
            else:
                nuts = nuts_posterior(fm, model, post["theta"], post["chol"],
                                      jax.random.PRNGKey(args.seed + 300), args.nuts_warmup,
                                      args.nuts_samples, args.nuts_chains,
                                      init_thetas=init_thetas)  # fmt: skip
            nuts["maps"] = in_fields_order(model, nuts["maps"])
            npm, nmed, nstd, ninfo = posterior_metrics(truth_maps, nuts["maps"], pstd, weight)
            mc = metrics[args.sampler] = {
                k: v for k, v in nuts.items() if k not in ("maps", "theta_samples")
            }
            mc["scalars"] = scalar_summary(model, nuts["theta_samples"], TRUTH_GLOBALS)
            mc["per_field"] = npm
            nmass = np.log10(np.sum(10.0 ** nuts["maps"][:, 0], axis=(1, 2)))
            mc["total_log_mass_16_50_84"] = [float(x) for x in np.percentile(nmass, [16, 50, 84])]
            fig_path = outdir / "figures" / f"{args.sampler}.png"
            plot_laplace(truth_maps, nmed, nstd, ninfo, weight, fig_path)
            np.savez(outdir / f"{args.sampler}.npz", median=nmed, std=nstd, information=ninfo)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(outdir / "maps.npz", truth=truth_maps, start=start_maps, map=fit_maps,
             model_image=model_img, data=np.asarray(obs.flux), sigma=sigma,
             theta_map=np.asarray(res.theta), fields=np.array(FIELDS))  # fmt: skip
    plot_maps(truth_maps, start_maps, fit_maps, weight, outdir / "figures" / "maps.png")
    plot_images(obs, model_img, sigma, outdir / "figures" / "images.png")

    if not args.no_bundle:
        try:
            to_model = [FIELDS.index(f) for f in model.field_names]
            truth = TruthContent(
                convolved=clean,
                intrinsic=np.asarray(render_model.model_image(theta_truth, emulator, (n, n))),
                param_maps=np.asarray(truth_maps)[to_model],
            )
            common = dict(fm=fm, model=model, args=args, truth=truth, truth_maps=truth_maps,
                          regions=demo_regions(truth_maps, arm_weight), prior_std=pstd_model,
                          metrics=metrics)  # fmt: skip
            laplace_kw = dict(
                theta_map=post["theta"] if post else None,
                samples=post["thetas"] if post else None,
                source="laplace",
                n_chains=None,
                diagnostics={
                    "log_evidence_laplace": post["log_evidence_laplace"] if post else None
                },
            )
            if nuts is not None:
                nm = metrics[args.sampler]
                diag = {k: nm[k] for k in ("max_rhat", "frac_rhat_gt_1.01", "min_ess",
                                            "median_ess")}  # fmt: skip
                diag["log_evidence_laplace"] = post["log_evidence_laplace"]
                path = write_demo_bundle(outdir / "results.h5", theta_map=post["theta"],
                                         samples=nuts["theta_samples"], source=args.sampler,
                                         n_chains=nuts["n_chains"], diagnostics=diag,
                                         **common)  # fmt: skip
                if args.bundle_all:
                    write_demo_bundle(outdir / "results_laplace.h5", **laplace_kw, **common)
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
    for name in FIELDS:
        s, m = metrics["start_vs_truth"][name], metrics["map_vs_truth"][name]
        print(f"  {name:18s} r: {s['pearson_r']:+.2f} -> {m['pearson_r']:+.2f}   "
              f"|err|: {s['median_abs_err']:.3f} -> {m['median_abs_err']:.3f}   "
              f"bias {m['bias']:+.3f}   truth rms {m['truth_rms']:.3f}")  # fmt: skip
    print(f"  total log M*: truth {metrics['total_log_mass_truth']:.3f}, "
          f"MAP {metrics['total_log_mass_map']:.3f}")  # fmt: skip
    if post is not None:
        lp = metrics["laplace"]
        print(f"Laplace: TR-Newton {lp['newton_s']:.0f} s ({lp['newton_reason']}, "
              f"{lp['newton_hvps']} HVPs), Hessian {lp['hessian_s']:.0f} s; "
              f"eigenvalues [{lp['eig_min']:.3g}, {lp['eig_max']:.3g}], "
              f"{lp['n_negative']} neg.")  # fmt: skip
        h0, h1 = lp["newton_history"][0], lp["newton_history"][-1]
        print(f"  TR-Newton {len(lp['newton_history']) - 1} steps: -log_post "
              f"{h0['neg_log_post']:.2f} -> {h1['neg_log_post']:.2f}, |grad| "
              f"{h0['grad_norm']:.3g} -> {h1['grad_norm']:.3g}")  # fmt: skip
        for name in FIELDS:
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
        for name in FIELDS:
            q = nm["per_field"][name]
            print(f"  {name:18s} cov68 {q['coverage68']:.2f}  cov95 {q['coverage95']:.2f}  "
                  f"z_rms {q['z_rms']:.2f}  post std {q['median_post_std']:.3f}")  # fmt: skip
        m16, m50, m84 = nm["total_log_mass_16_50_84"]
        print(f"  total log M* {m50:.3f} +{m84 - m50:.3f} -{m50 - m16:.3f}")
    if "modes" in metrics:
        print(f"modes ({len(runs)} starts; best = start {i_best + 1}):")
        for i, m in enumerate(metrics["modes"]):
            c = " ".join(f"{SHORT.get(k, k)}={v:.2f}" for k, v in m["coverage68"].items())
            print(f"  {i + 1}: log Z {m['log_evidence_laplace']:.1f}  -log_post "
                  f"{m['neg_log_post']:.1f}  conv {m['newton_converged']}  neg {m['n_negative']}"
                  f"  |dz| {m['whitened_distance_to_best']:.1f}  M* "
                  f"{m['total_log_mass_16_50_84'][1]:.3f}  cov68 {c}")  # fmt: skip
    for stage in ("laplace", args.sampler):
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
