#!/usr/bin/env python3
r"""Resolved SED fitting demo: Sersic bulge + disk, nuisances, ParrotEmulatorV2, real JWST PSFs.

A two-component galaxy (compact, old, quenched bulge + extended, young, dusty
star-forming disk at a common redshift) is rendered as a 10-band JWST/NIRCam
mock with the real trained ``ParrotEmulatorV2`` checkpoint and real JADES
empirical PSFs, contaminated with per-band sky pedestals, per-band sub-pixel
astrometric shifts and Gaussian noise, and then fitted back **blind** -- the
injected truth is used only to generate the data and to grade the answer,
never to start the fit.

The model
---------
``AdditiveComponentModel`` (``src/arachne/spatial/additive.py``) in **arcsec
mode** (``pixel_scale=0.03``) with **analytic** normalisation and
``oversample=3``: light is additive, so each component ``k`` has a
unit-total-flux surface-brightness profile ``P_k(y, x)`` times the emulator
SED ``F_k`` of its *own* SPS parameters, including its own **total** stellar
mass::

    I_b(y, x) = Sum_k F_kb(theta_k) . P_k(y, x)      (K emulator calls per likelihood)

Both components are **Sersic** profiles (``--profiles sersic``, the default):
each carries 6 shape parameters ``[mu_y, mu_x, log_sigma_y, log_sigma_x,
atanh_rho, log_n]`` where ``mu`` is an arcsec offset from the frame centre,
``sigma`` is the effective radius along each axis in arcsec, and ``n`` is the
Sersic index.  ``--profiles gaussian`` swaps in the historical
``GaussianProfile`` family (5 shape parameters, ``sigma`` = Gaussian sigma);
everything else (arcsec coordinates, analytic normalisation, oversampling)
is unchanged, so the two options differ only in the light profile.

With analytic normalisation the profile integrates to 1 over the *whole
plane*, so flux that falls outside the 4.8" cutout is genuinely lost (as in
reality) and the same components could be rendered on another band's grid
with ``model_image_on``.

Instrumental nuisances
----------------------
``NuisanceModel(n_bands, fit_sky=True, fit_shifts=True,
shift_reference_band=F200W)`` appends ``N_bands`` sky pedestals (nJy/pixel)
and ``2 (N_bands - 1)`` sub-pixel registration offsets ``(dy, dx)`` in pixels
to ``theta``.  The mock **injects** both (sky ~ N(0, 0.3 nJy/px), shifts
~ N(0, 0.3 px)) and the fit **marginalises** over them with the library's own
Gaussian priors (1.0 nJy/px, 0.5 px), which are deliberately wider than the
injected scatter.  The PSF convolution is the zero-padded *linear* one
(``pad_psf=True``) and the shifts are applied as an exact Fourier phase ramp
inside ``PSFConvolver``.  ``--no-nuisance`` switches both the injection and
the fitting off.

Fitting a shift in *every* band would be exactly degenerate: shifting all
bands by ``+d`` pixels and moving *both* component centres by ``-d`` pixels
leaves the data unchanged, and that direction is held only by the (soft)
shift and centre priors.  A diagonal-metric sampler crawls along it -- with
all 10 bands free, 4 chains x 200 draws returned ``max split-R-hat = 1813``
and ``ESS = 4``.  So F200W is the **astrometric reference**: its ``(dy, dx)``
is pinned to zero (``shift_reference_band``), the injected shifts are made
relative to it, and the fitted offsets are *relative registrations*, which is
what a real pipeline measures.  ``--all-band-shifts`` restores the degenerate
parameterisation if you want to watch it fail.

theta layout: ``[block_0, block_1, sky (10), dy_b, dx_b (9 bands)]`` with
``block_k = [shape_k (6), sps_raw_k (11)]`` -> 34 spatial + 28 nuisance
= **62 free parameters** for the default Sersic + nuisance configuration.
Redshift is *fixed* at the true value (a bulge and disk of one galaxy share
one redshift; fitting it is not the point here).

The prior is the model's own: uniform in physical space for every SPS
parameter (via the sigmoid Jacobian), the catalogue pipeline's default
Student-t(df=2, scale=0.3) on the ``logsfr_ratio_*`` (the emulator training
grid was drawn from it), Gaussians on the shape parameters (centre, log size,
``atanh_rho``, and ``log_n ~ N(log 2, 0.7)`` for Sersic components) and the
nuisance Gaussians above.  The ``model_error_frac`` fractional floor in
``GaussianLikelihood`` (default 5%) is added in quadrature to the photon
noise: the emulator is only accurate to a few per cent and a resolved fit sums
~2.5e5 pixels, so without the floor the posterior would be far narrower than
the emulator's own systematics.

The blind pipeline
------------------
1. ``blind_initial_theta``: centroid + rms size from the S/N-stacked image;
   two components on a size ladder (0.4x, 1.5x the moment size); Sersic
   ``n = 4`` for the compact one and ``n = 1`` for the extended one; generic
   neutral SPS values.  ``ForwardModel.initial_theta_from_spatial`` then
   appends the nuisance block at its prior mean (all zeros).
2. ``multistart_map``: ``find_map`` (Adam with the K masses re-solved by
   weighted linear least squares between rounds) from the neutral start and
   from a few generic archetypes (low/high dust, rising SFH); the best
   ``-log_posterior`` wins.  That finds the right *basin* but stops tens of
   nats short of the mode, so ``polish_map`` then anneals Adam
   (0.01 -> 0.0003) from it with the mass re-solve switched off -- the linear
   solve maximises the floor-free likelihood and actually *raises*
   ``-log_post`` once the fit is close.  ``-log_post(MAP)`` is compared with
   ``-log_post(truth)`` *after* the fact, as a check that the true basin was
   found.
3. Sampling: ``--sampler auto`` (the default) runs **NUTS** with 4 chains from
   the MAP whenever the nuisances are fitted, and **NSS** for the
   34-parameter ``--no-nuisance`` model.  Nested sampling starts from the prior
   and has to compress the whole prior volume, which is affordable at 34
   parameters but not at 62: a measured ``num_live=100`` NSS run on the full
   nuisance model needed >3 s per outer step with the likelihood still
   ~1.5e6 nats below the MAP after 50 steps.  ``--sampler nss`` forces nested
   sampling anyway (it yields the evidence ``logZ`` and checkpoints to
   ``<outdir>/nss_ckpt``, so an interrupted run resumes bit-exactly).
   NUTS runs through ``arachne.inference.laplace.run_whitened_nuts``, which
   samples **whitened coordinates** ``theta = MAP + L z`` (``L`` = Cholesky
   factor of the Laplace covariance built from the Newton Hessian) and adapts
   a **dense** metric inside that whitened space.  Three separate things were
   needed to make this posterior sample at all, and the demo flags let each be
   turned off to watch it fail:

   * **Float64** (``--float32`` reproduces the failure): in float32 the
     rounding error of a 2.6e5-pixel log-likelihood is a few tenths of a nat,
     which is larger than the energy differences NUTS is trying to measure.
   * **Whitening** (``--no-whiten``): in raw coordinates the curvature
     eigenvalues at the mode span ``1e-1`` to ``4e7``, and because the stiff
     directions are strongly correlated even the exact *diagonal* metric
     leaves a condition number of ~2e7; warmup then collapses to a step size
     of ~4e-8 and 4 chains return ``R-hat ~ 4000`` with ``ESS = 4``.
   * **A dense metric in the whitened space** (``--diagonal-metric``), plus
     the ``|lambda|`` treatment of negative curvature inside
     ``laplace_covariance``.  Whitening alone is not enough: the Laplace
     approximation is built at a point that still has a handful of slightly
     *negative* curvature directions (the mass-exchange / mass-dust ridge is
     not quadratic), and the old "give those directions the widest variance
     allowed" rule made them 10-100x too wide, which is exactly as bad as
     being too narrow -- the whitened posterior still had marginal widths from
     5e-4 to 23, a condition number of 4.4e4, a step size of 2e-3 and
     saturated trajectories (max R-hat 4.5-7, min ESS 5, 8 divergences on
     4 x 150 draws).  Flooring by ``|lambda|`` instead takes the whitened
     condition number to ~350, and the dense metric adapted during warmup
     absorbs what is left.

   Even all three together are not quite enough, because the ridge is
   *curved*: no fixed metric describes a banana.  So the NUTS path is run in
   **two stages** (``--stage1 N``, ``--stage1 0`` disables it): stage 1 is a
   short exploratory run in the Laplace whitening, and its draws are then fed
   to ``arachne.inference.laplace.posterior_whitening``, which re-centres the
   coordinates on the posterior mean and rescales them with the covariance
   the chains actually explored (shrunk toward the Laplace covariance, weight
   ``d / (d + n_draws)``).  Stage 2 samples in those coordinates.  This is the
   metric the sampler could never adapt on its own: a dense metric estimated
   inside a 500-draw warmup window is fitted to ~1 effective sample per
   direction, whereas stage 1 provides ``n_chains x N`` draws.

   ``chain_jitter`` (default 0.3) is in units of a whitened posterior sigma.
   Split-R-hat and bulk ESS are recomputed in the *physical* theta
   coordinates after the run (``arachne.inference.diagnostics``) and printed
   with the divergences, tree depth and per-chain step sizes.
4. Components are exchangeable; labels are broken by ordering by size
   (compact first) in the MAP and in every posterior sample.
5. Posterior products come from ``arachne.inference.posterior_predictive``
   (``model_image_samples``, ``component_image_samples``, ``residual_summary``,
   ``predictive_bands``), not from hand-rolled equivalents.

Assumptions
-----------
* Pixel scale 0.03"/px (assumed; the JADES PSF FITS files carry no WCS).
* Noise: Gaussian, per band, with sigma set so the azimuthally averaged truth
  surface brightness at r = 0.45" (the disk's effective radius) has S/N = 3
  per pixel in every band (``--noise-scale`` rescales it).  This is not a
  survey depth; it is chosen to make the disk well measured and is robust to
  the cuspiness of an n = 4 core (unlike a peak-pixel definition).
* Truth: bulge 10^10.3 Msun, quenched (declining SFH, Av = 0.15), n = 4,
  r_e = 0.12", b/a = 0.8, PA = 0 deg; disk 10^10.6 Msun, star-forming and
  dusty (Av = 1.8), n = 1, r_e = 0.45", b/a = 0.6, PA = 30 deg, offset by
  (+0.02", -0.03") from the bulge; both at z = 2.  ``r_e`` is the *geometric
  mean* effective radius sqrt(a b), which is what the recovery table grades.
  The two components have comparable total fluxes (the bulge dominates the
  surface brightness inside ~0.2" and the disk outside it), which is what
  makes the decomposition identifiable.

Known limitations
-----------------
* The mock is generated by the same emulator and the same renderer that fit
  it, so emulator and pixelisation systematics are absent from the data; the
  5% floor is therefore a conservative choice here and the posterior is honest
  about photon noise only.  On real data emulator error dominates at this S/N.
* Photometry-only constraints on ``fesc_lya``, ``dust_bump_amplitude`` and
  the higher-order ``logsfr_ratio_*`` are weak; wide (prior-like) intervals
  there are the correct answer, not a failure.
* Two co-centred Sersic profiles is the mock's own family; real galaxies are
  not.  ``--compare-k`` shows what the evidence says about K = 1 vs K = 2.

Usage
-----
    ssh mad06                     # needs a GPU node (see CLAUDE.md)
    python examples/demo_resolved_sed_fitting.py --quick        # smoke run
    python examples/demo_resolved_sed_fitting.py                # NUTS, 4 chains
    python examples/demo_resolved_sed_fitting.py --no-nuisance  # NSS + logZ, 34 params
    python examples/demo_resolved_sed_fitting.py --profiles gaussian --no-nuisance
    python examples/demo_resolved_sed_fitting.py --compare-k --quick

CLI (all optional)
------------------
``--outdir``, ``--seed``, ``--quick``, ``--profiles {sersic,gaussian}``,
``--oversample``, ``--no-nuisance``, ``--all-band-shifts``, ``--noise-scale``,
``--model-err-frac``, ``--sampler {auto,nss,nuts}``, ``--num-live``,
``--num-inner-steps``, ``--termination``, ``--no-resume``, ``--n-chains``,
``--n-warmup``, ``--n-samples``, ``--max-doublings``, ``--chain-jitter``,
``--n-newton``, ``--stage1``, ``--diagonal-metric``, ``--no-whiten``,
``--float32``, ``--compare-k``, ``--compare-ks``.

Expected runtimes (A100, mad06, GPU shared with another job)
------------------------------------------------------------
* blind MAP (multistart over 4 archetypes + polish):      ~5 min
* ``--quick`` (NUTS, 4 chains x (150 + 200)):             ~20 min
* default (NUTS, 4 chains x (500 + 1000)):                ~1-3 h
* ``--no-nuisance --sampler nss --quick``:                ~30 min

NSS cost model: each NS step replaces ``num_live // 10`` points, so the
*number* of steps is nearly independent of ``num_live``, while the per-step
cost is proportional to ``num_delete x num_inner_steps`` (both vectorised on a
GPU, so the ``num_delete`` scaling is sub-linear) and the evidence error scales
as ``1/sqrt(num_live)``.  ``num_inner_steps`` therefore sets the wall clock
almost linearly; the default here is ``1.5 * n_params`` rather than blackjax's
recommended ``3 * n_params`` (which doubles the cost; fewer inner steps bias
``logZ`` upward).  NUTS cost is ``n_chains`` (vmapped) x draws x the tree
length, which is capped by ``--max-doublings``.  A second job sharing the GPU
slows any of this several-fold.

Outputs (under ``--outdir``, default ``outputs/demo/resolved_sed_fitting/``)
----------------------------------------------------------------------------
- ``truth.json``       injected truth (raw theta + physical parameters +
                       Sersic/ellipse geometry + injected sky and shifts)
- ``map.json``         blind MAP (raw theta + physical parameters, -log_post,
                       recovered sky and shifts)
- ``posterior.hdf5``   size-ordered posterior samples (+ logZ for NSS)
- ``sampler_summary.txt``  the sampler's own convergence summary
- ``bayes_factors.txt``    evidence table (only with ``--compare-k``)
- ``figures/truth_data_model_mosaic.png``
- ``figures/chi_map_mosaic.png``
- ``figures/sps_param_maps.png``
- ``figures/sed_bulge_disk.png``
- ``figures/posterior_summary_1d.png``
- ``figures/sampler_diagnostics.png``
- ``figures/component_profiles.png``
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

_ARACHNE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ARACHNE / "scripts"))

from fit_catalogue import PARAM_BOUNDS, SPS_PARAM_NAMES  # noqa: E402

from arachne.data.observation import ObservationCube  # noqa: E402
from arachne.data.psf import PSFModel  # noqa: E402
from arachne.emulator.base import SPSEmulator  # noqa: E402
from arachne.emulator.parrot_emulator_v2 import load_emulator  # noqa: E402
from arachne.forward_model.nuisance import NuisanceModel  # noqa: E402
from arachne.forward_model.pipeline import ForwardModel  # noqa: E402
from arachne.inference.initialisation import (  # noqa: E402
    MAPResult,
    blind_initial_theta,
    find_map,
    multistart_map,
)
from arachne.inference.laplace import (  # noqa: E402
    laplace_whitening,
    make_hessian_fn,
    posterior_whitening,
    run_whitened_nuts,
)
from arachne.inference.model_comparison import (  # noqa: E402
    bayes_factor_table,
    compare_n_components,
)
from arachne.inference.nss_sampler import NSSResult, NSSSampler  # noqa: E402
from arachne.inference.nuts_sampler import NUTSSampler  # noqa: E402
from arachne.inference.posterior_predictive import (  # noqa: E402
    component_image_samples,
    model_image_samples,
    predictive_bands,
    residual_summary,
)
from arachne.likelihood.gaussian import GaussianLikelihood  # noqa: E402
from arachne.priors.specs import (  # noqa: E402
    DEFAULT_PRIORS,
    build_component_log_prior,
    resolve_prior_specs,
)
from arachne.psf.convolution import PSFConvolver  # noqa: E402
from arachne.spatial.additive import AdditiveComponentModel  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT = _ARACHNE / "scripts/outputs/emulators/parrot_emulator_v2.eqx"
PSF_DIR = Path("/cosma/apps/dp276/dc-harv3/synference/priv/JADES-DR3-GS")
DEFAULT_OUTDIR = _ARACHNE / "outputs/demo/resolved_sed_fitting"

PIXEL_SCALE = 0.03  # arcsec/px, assumed -- see module docstring
NPIX = 160  # hard lower bound: real PSF kernel is 133x133 px
TRUE_REDSHIFT = 2.0
SNR_RADIUS = 0.45  # arcsec; radius at which the per-pixel S/N is set (the disk r_e)
SNR_AT_RADIUS = 3.0  # per-pixel S/N of the azimuthal mean at SNR_RADIUS
SKY_TRUE_SIGMA = 0.3  # nJy/px, sigma of the injected per-band sky pedestals
SHIFT_TRUE_SIGMA = 0.3  # px, sigma of the injected per-band (dy, dx) offsets

# NIRCam wide+medium bands present in both the emulator checkpoint and PSF_DIR.
BAND_NAMES = [
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F335M",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F444W",
    "JWST/NIRCam.F480M",
]
REF_BAND = "JWST/NIRCam.F200W"  # band used for the radial-profile figure
PIVOT_UM = {
    "F090W": 0.90, "F115W": 1.15, "F150W": 1.50, "F200W": 2.00, "F277W": 2.77,
    "F335M": 3.35, "F356W": 3.56, "F410M": 4.10, "F444W": 4.44, "F480M": 4.80,
}  # fmt: skip

PARAM_LABELS = {
    "log_mass": r"$\log M_\star / M_\odot$",
    "slope": r"$\delta$ (attenuation slope)",
    "fesc_lya": r"$f_\mathrm{esc,Ly\alpha}$",
    "dust_bump_amplitude": r"$B_{2175}$ (dust bump)",
    "log10metallicity": r"$\log Z$",
    "Av": r"$A_V$ (mag)",
    "logsfr_ratio_0": r"$\log \mathrm{SFR}_0/\mathrm{SFR}_1$",
    "logsfr_ratio_1": r"$\log \mathrm{SFR}_1/\mathrm{SFR}_2$",
    "logsfr_ratio_2": r"$\log \mathrm{SFR}_2/\mathrm{SFR}_3$",
    "logsfr_ratio_3": r"$\log \mathrm{SFR}_3/\mathrm{SFR}_4$",
    "logsfr_ratio_4": r"$\log \mathrm{SFR}_4/\mathrm{SFR}_5$",
}
MAP_FIGURE_PARAMS = ["log_mass", "Av", "log10metallicity", "logsfr_ratio_0"]
COMPONENT_NAMES = ["bulge (compact, comp 0)", "disk (extended, comp 1)"]
SHORT_NAMES = ["bulge", "disk"]

# Derived (physically meaningful) shape quantities graded in the report.
SHAPE_KEYS = ["mu_y", "mu_x", "r_e", "axis_ratio", "PA_deg", "sersic_n"]
SHAPE_LABELS = {
    "mu_y": 'mu_y ["]',
    "mu_x": 'mu_x ["]',
    "r_e": 'r_eff ["]',
    "axis_ratio": "axis ratio b/a",
    "PA_deg": "PA [deg]",
    "sersic_n": "Sersic n",
}

# Truth.  Masses are the TOTAL stellar mass of each component (the model's
# mass parameter is the component total, not a per-pixel surface density).
BULGE_PHYS = {
    "log_mass": 10.3,
    "slope": 0.5,
    "fesc_lya": 0.05,
    "dust_bump_amplitude": 0.2,
    "log10metallicity": -1.7,
    "Av": 0.15,
    "logsfr_ratio_0": -1.0,
    "logsfr_ratio_1": 0.5,
    "logsfr_ratio_2": 0.5,
    "logsfr_ratio_3": 0.0,
    "logsfr_ratio_4": 0.0,
}
# Geometry in arcsec / degrees.  r_e is the geometric-mean effective radius
# sqrt(a*b); PA is measured from the +x (column) axis towards +y (row).
BULGE_SHAPE = dict(mu_y=0.0, mu_x=0.0, r_e=0.12, axis_ratio=0.8, PA_deg=0.0, sersic_n=4.0)

DISK_PHYS = {
    "log_mass": 10.6,
    "slope": 0.0,
    "fesc_lya": 0.15,
    "dust_bump_amplitude": 1.5,
    "log10metallicity": -2.6,
    "Av": 1.8,
    "logsfr_ratio_0": -0.5,
    "logsfr_ratio_1": -0.3,
    "logsfr_ratio_2": 0.0,
    "logsfr_ratio_3": 0.2,
    "logsfr_ratio_4": 0.0,
}
DISK_SHAPE = dict(mu_y=0.02, mu_x=-0.03, r_e=0.45, axis_ratio=0.6, PA_deg=30.0, sersic_n=1.0)

# Generic SPS archetypes for multistart_map.  These are NOT the truth values:
# they probe the dust/age degeneracy from both sides and a rising SFH.
ARCHETYPES = [{}, {"Av": 0.3}, {"Av": 2.0}, {"logsfr_ratio_0": 1.0}]

# Polish schedule after multistart_map: (learning rate, Adam steps) per stage,
# annealed, with the linear mass re-solve switched OFF (see ``polish_map``).
POLISH_STAGES_QUICK = [(0.01, 600), (0.003, 1600), (0.001, 1200), (0.0003, 1200)]
POLISH_STAGES_FULL = [(0.01, 1000), (0.003, 3000), (0.001, 2500), (0.0003, 2500)]
POLISH_STALL_NATS = 0.2  # stop annealing when two stages in a row buy less than this
MAX_LAPLACE_VAR = 9.0  # cap on a Laplace metric variance (raw units; ~the prior width)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def to_raw(phys: float, lo: float, hi: float) -> float:
    p = (phys - lo) / (hi - lo)
    return float(np.log(p / (1.0 - p)))


def cov_from_ellipse(r_e: float, axis_ratio: float, pa_deg: float) -> tuple[float, float, float]:
    """(r_e, b/a, PA) -> (sigma_y, sigma_x, rho) of the model's covariance form."""
    a = r_e / math.sqrt(axis_ratio)
    b = r_e * math.sqrt(axis_ratio)
    th = math.radians(pa_deg)
    s, c = math.sin(th), math.cos(th)
    syy = a**2 * s**2 + b**2 * c**2
    sxx = a**2 * c**2 + b**2 * s**2
    syx = (a**2 - b**2) * s * c
    sy, sx = math.sqrt(syy), math.sqrt(sxx)
    return sy, sx, syx / (sy * sx)


def ellipse_from_cov(sigma_y, sigma_x, rho):
    """(sigma_y, sigma_x, rho) -> (r_e, b/a, PA_deg); JAX-friendly, elementwise."""
    syy = sigma_y**2
    sxx = sigma_x**2
    syx = rho * sigma_y * sigma_x
    diff = jnp.sqrt((syy - sxx) ** 2 + 4.0 * syx**2)
    lam1 = 0.5 * (syy + sxx + diff)
    lam2 = jnp.maximum(0.5 * (syy + sxx - diff), 1e-20)
    a = jnp.sqrt(lam1)
    b = jnp.sqrt(lam2)
    pa = jnp.rad2deg(0.5 * jnp.arctan2(2.0 * syx, sxx - syy))
    return jnp.sqrt(a * b), b / a, pa


def wrap180(x: np.ndarray) -> np.ndarray:
    """Wrap a position-angle difference into (-90, 90] degrees."""
    return (np.asarray(x) + 90.0) % 180.0 - 90.0


class BandSubsetEmulator(SPSEmulator):
    """Restrict an emulator's output to a subset of its bands (same inputs)."""

    inner: SPSEmulator
    _band_names: tuple[str, ...] = eqx.field(static=True)
    _band_idx: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, inner: SPSEmulator, band_names: list[str]) -> None:
        missing = [b for b in band_names if b not in inner.band_names]
        if missing:
            raise ValueError(f"bands not in emulator: {missing}")
        self.inner = inner
        self._band_names = tuple(band_names)
        self._band_idx = tuple(inner.band_names.index(b) for b in band_names)

    @property
    def param_names(self) -> list[str]:
        return list(self.inner.param_names)

    @property
    def band_names(self) -> list[str]:
        return list(self._band_names)

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        return self.inner.predict(params)[:, jnp.array(self._band_idx)]


def _psf_path(band_name: str) -> Path:
    return PSF_DIR / f"{band_name.split('.')[-1]}_psf_norm.fits"


def load_emulator_and_psf() -> tuple[SPSEmulator, PSFModel]:
    print(f"Loading emulator checkpoint: {CHECKPOINT}")
    emulator = BandSubsetEmulator(load_emulator(CHECKPOINT), BAND_NAMES)
    assert list(emulator.param_names) == SPS_PARAM_NAMES
    print(f"  {len(emulator.param_names)} emulator params, {len(emulator.band_names)} bands")
    missing = [b for b in BAND_NAMES if not _psf_path(b).exists()]
    if missing:
        raise FileNotFoundError(f"Missing PSF files for {missing} under {PSF_DIR}")
    psf_model = PSFModel.from_fits({b: _psf_path(b) for b in BAND_NAMES})
    print(f"  Loaded {psf_model.n_bands} PSF kernels from {PSF_DIR}")
    return emulator, psf_model


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_spatial_model(
    emulator: SPSEmulator, n_components: int, profile: str, oversample: int
) -> AdditiveComponentModel:
    names = list(emulator.param_names)
    free = [p for p in names if p != "redshift"]
    specs = resolve_prior_specs(free, None, PARAM_BOUNDS)  # DEFAULT_PRIORS resolved
    assert all(specs[p] == DEFAULT_PRIORS.get(p, {"dist": "uniform"}) for p in free)
    sps_log_prior = build_component_log_prior(
        names, specs, PARAM_BOUNDS, shared_param_names=[], fixed_param_names=["redshift"]
    )
    return AdditiveComponentModel(
        n_components=n_components,
        emulator_param_names=names,
        param_bounds=PARAM_BOUNDS,
        image_shape=(NPIX, NPIX),
        fixed_params={"redshift": TRUE_REDSHIFT},
        mass_param="log_mass",
        sps_log_prior=sps_log_prior,
        profiles=[profile] * n_components,
        pixel_scale=PIXEL_SCALE,
        normalisation="analytic",
        oversample=oversample,
    )


def build_nuisance(use_nuisance: bool, all_band_shifts: bool = False) -> NuisanceModel | None:
    """Per-band sky + registration nuisances, with F200W as the astrometric reference.

    Fitting a ``(dy, dx)`` in *every* band is exactly degenerate with moving
    every component (see the module docstring), and a diagonal-metric sampler
    crawls along that direction: a 4-chain NUTS run with all 10 bands free
    reached ``max split-R-hat = 1813`` and ``ESS = 4``.
    ``shift_reference_band`` pins the reference band's shift to zero, so the
    other bands' offsets are *relative registrations* -- the physically
    meaningful quantity -- and the degeneracy is gone.  ``--all-band-shifts``
    restores the degenerate parameterisation for demonstration.
    """
    if not use_nuisance:
        return None
    return NuisanceModel(
        len(BAND_NAMES),
        fit_sky=True,
        fit_shifts=True,
        shift_reference_band=None if all_band_shifts else BAND_NAMES.index(REF_BAND),
    )


def build_forward_model(
    obs: ObservationCube,
    model: AdditiveComponentModel,
    emulator: SPSEmulator,
    convolver: PSFConvolver,
    model_err_frac: float,
    nuisance: NuisanceModel | None,
) -> ForwardModel:
    obs_jax = obs.to_jax()
    likelihood = GaussianLikelihood(obs_jax, model_error_frac=model_err_frac)
    return ForwardModel(
        observation=obs_jax,
        spatial_model=model,
        emulator=emulator,
        convolver=convolver,
        likelihood=likelihood,
        nuisance=nuisance,
    )


def promote_to_float64(fwd: ForwardModel) -> ForwardModel:
    """Re-cast an assembled ForwardModel to float64 and switch JAX to x64.

    **This is what makes gradient-based sampling possible at all here.**  The
    likelihood sums 10 x 160 x 160 = 2.6e5 pixels; in float32 the individual
    terms (chi-squared and ``log var_eff``) accumulate to ~1e6 in magnitude, so
    the *rounding error* of the total is a few tenths of a nat -- and it
    changes unpredictably whenever the model image changes.  A Hamiltonian
    sampler cannot see past that noise floor: the energy difference of a
    proposal is dominated by rounding, the acceptance probability sticks near
    the dual-averaging target whatever the step size, and warmup drives the
    step size to ~1e-7 while every trajectory saturates the tree cap.  That is
    exactly the observed failure (``R-hat ~ 4000``, ``ESS = 4``,
    0 divergences, 100% of trajectories at the cap).  A fine scan of
    ``-log_post`` across the mode is pure noise of +/-0.5 nats in float32 and
    perfectly smooth in float64, and on an A100/H100 float64 costs nothing
    measurable here (~5 ms per evaluation either way: this model is
    memory-bound, not FLOP-bound).

    The emulator checkpoint must be deserialised *before* x64 is enabled
    (``eqx.tree_deserialise_leaves`` refuses a float64 template for float32
    weights on disk), which is why this is a post-hoc promotion of an already
    assembled model rather than a flag at the top of the file.
    ``--float32`` skips it and reproduces the failure.
    """
    jax.config.update("jax_enable_x64", True)

    def up(x):
        if not eqx.is_inexact_array(x):
            return x
        return x.astype(jnp.complex128 if jnp.iscomplexobj(x) else jnp.float64)

    return jax.tree_util.tree_map(up, fwd)


# ---------------------------------------------------------------------------
# Truth and mock observation
# ---------------------------------------------------------------------------


def build_truth_spatial_theta(model: AdditiveComponentModel) -> np.ndarray:
    """Injected spatial theta (bulge block, disk block).  Never seen by the fit."""
    blocks = []
    for phys, shape in ((BULGE_PHYS, BULGE_SHAPE), (DISK_PHYS, DISK_SHAPE)):
        raw = [to_raw(phys[p], *model.param_bounds[p]) for p in model.sps_param_names]
        sy, sx, rho = cov_from_ellipse(shape["r_e"], shape["axis_ratio"], shape["PA_deg"])
        block = [
            shape["mu_y"],
            shape["mu_x"],
            np.log(sy),
            np.log(sx),
            np.arctanh(rho),
        ]
        if model.profile_objects[0].n_shape == 6:  # Sersic
            block.append(np.log(shape["sersic_n"]))
        blocks.append([*block, *raw])
    theta = model.join_theta(jnp.array(blocks, dtype=jnp.float32), jnp.zeros(model.n_shared))
    return np.asarray(theta, dtype=np.float32)


def draw_nuisance_truth(nuisance: NuisanceModel | None, seed: int) -> np.ndarray:
    """Injected sky pedestals and per-band (dy, dx) offsets, in NuisanceModel order.

    With a ``shift_reference_band`` the injected shifts are made *relative* to
    that band (its own offset is subtracted from every band), which is what
    the fitted model can represent and what a real registration measures.
    """
    if nuisance is None:
        return np.zeros(0, dtype=np.float32)
    rng = np.random.default_rng(seed + 7)
    nb = nuisance.n_bands
    sky = rng.normal(0.0, SKY_TRUE_SIGMA, size=nb)
    shifts = rng.normal(0.0, SHIFT_TRUE_SIGMA, size=(nb, 2))
    ref = nuisance.shift_reference_band
    if ref is not None:
        shifts = shifts - shifts[ref]
    free = shifts[np.asarray(nuisance.shift_bands)]
    return np.concatenate([sky, free.reshape(-1)]).astype(np.float32)


def synthesise_truth_images(
    model: AdditiveComponentModel,
    emulator: SPSEmulator,
    convolver: PSFConvolver,
    theta_spatial: np.ndarray,
    nuisance: NuisanceModel | None,
    theta_nuisance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (astrophysical convolved+shifted image, same + injected sky)."""
    unconvolved = model.model_image(jnp.asarray(theta_spatial), emulator, (NPIX, NPIX))
    if nuisance is None:
        astro = convolver(unconvolved)
        sky = jnp.zeros(unconvolved.shape[0])
    else:
        blocks = nuisance.split(jnp.asarray(theta_nuisance))
        astro = convolver(unconvolved, shifts=blocks["shifts"])
        sky = blocks["sky"]
    astro_np = np.asarray(astro, dtype=np.float32)
    return astro_np, (astro_np + np.asarray(sky)[:, None, None]).astype(np.float32)


def _radius_px(centre_px: tuple[float, float]) -> np.ndarray:
    yy, xx = np.mgrid[:NPIX, :NPIX]
    return np.hypot(yy - centre_px[0], xx - centre_px[1])


def noise_sigma(true_image: np.ndarray, centre_px, noise_scale: float) -> np.ndarray:
    """sigma_b from the azimuthal mean surface brightness at ``SNR_RADIUS``."""
    r = _radius_px(centre_px)
    r0 = SNR_RADIUS / PIXEL_SCALE
    ring = (r > r0 - 1.0) & (r < r0 + 1.0)
    sb = true_image[:, ring].mean(axis=1)
    return (np.abs(sb) / SNR_AT_RADIUS * noise_scale).astype(np.float32)


def add_noise(
    true_image: np.ndarray, sigma: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=true_image.shape).astype(np.float32) * sigma[:, None, None]
    variance = np.broadcast_to(sigma[:, None, None] ** 2, true_image.shape)
    return (true_image + noise).astype(np.float32), variance.astype(np.float32)


def print_noise_summary(true_image: np.ndarray, sigma: np.ndarray, centre_px, sky_true) -> None:
    print("\n--- Mock noise summary ---")
    print(
        f'  sigma set so the azimuthal mean at r = {SNR_RADIUS}" has '
        f"S/N = {SNR_AT_RADIUS:.0f} per pixel in every band"
    )
    r = _radius_px(centre_px)
    radii = (BULGE_SHAPE["r_e"], SNR_RADIUS, 2.0 * SNR_RADIUS)
    rings = [(r > rr / PIXEL_SCALE - 1) & (r < rr / PIXEL_SCALE + 1) for rr in radii]
    header = f"  {'band':22s} {'sigma[nJy]':>10s} {'sky[nJy]':>9s} {'peak S/N':>9s}"
    header += "".join(f"{f'S/N r={rr}as':>12s}" for rr in radii)
    header += f" {'S/N total':>10s} {'N(S/N>3)':>9s}"
    print(header)
    npix = true_image[0].size
    for b, band in enumerate(BAND_NAMES):
        img = true_image[b]
        sn_tot = img.sum() / (sigma[b] * np.sqrt(npix))
        n3 = int((img / sigma[b] > 3).sum())
        sky_b = float(sky_true[b]) if len(sky_true) else 0.0
        line = f"  {band:22s} {sigma[b]:10.3f} {sky_b:9.3f} {img.max() / sigma[b]:9.0f}"
        line += "".join(f"{img[ring].mean() / sigma[b]:12.2f}" for ring in rings)
        print(line + f" {sn_tot:10.0f} {n3:9d}")


# ---------------------------------------------------------------------------
# Blind fit
# ---------------------------------------------------------------------------


def order_full_theta(fwd: ForwardModel, theta: jnp.ndarray) -> jnp.ndarray:
    """Order components compact-first, preserving the nuisance block.

    ``AdditiveComponentModel.order_components_by_size`` returns a *spatial*
    theta, so applying it to a full theta would silently drop the nuisance
    parameters (see the library-bug note in the report).
    """
    theta_spatial, theta_nuisance = fwd.split_theta(theta)
    ordered = fwd.spatial_model.order_components_by_size(theta_spatial)
    return jnp.concatenate([ordered, theta_nuisance])


def polish_map(fwd: ForwardModel, res: MAPResult, quick: bool) -> MAPResult:
    """Anneal Adam from ``multistart_map``'s best point, without the mass re-solve.

    ``multistart_map`` leaves the fit tens of nats short of the mode: the
    Sersic ``log_n`` and the two log-size directions are far stiffer than the
    SPS raws, and the nuisance block only moves by gradient descent.
    Two things fix that, both measured on the quick mock:

    * **Annealing.** Restarting Adam at 0.01 -> 0.003 -> 0.001 -> 0.0003 keeps
      going long after a single run at one learning rate has stalled; the
      stiff directions need the small steps and the soft ones need the long
      runs.  Every stage restarts from the best point so far, and a stage that
      makes things worse (the largest learning rate sometimes kicks the fit out
      of a narrow valley) is simply discarded.
    * **``resolve_masses=False``.** The linear mass solve is what finds the
      right basin from the blind start, but it maximises the *likelihood
      without the model-error floor* and ignores the prior, so near the mode it
      moves the masses slightly the wrong way: re-solving the masses at the
      polished point *raised* ``-log_post`` from 3069.3 to 3078.7 in a probe.

    Measured effect on the quick run: ``-log_post`` 3069.3 -> ~3035 (the
    MAP-minus-truth gap falls from +36 to a couple of nats) for ~40 s of extra
    GPU time, against ~20 min for the sampler.
    """
    stages = POLISH_STAGES_QUICK if quick else POLISH_STAGES_FULL
    print(f"  annealed polish (no mass re-solve), stages {stages} ...")
    last_stalled = False
    for lr, steps in stages:
        # n_rounds=0 -> a single Adam run of ``steps`` steps at ``lr``.
        nxt = find_map(
            fwd,
            res.theta,
            n_rounds=0,
            final_steps=steps,
            final_lr=lr,
            resolve_masses=False,
            order_by_size=False,
        )
        gain = res.neg_log_posterior - nxt.neg_log_posterior
        kept = gain > 0.0
        print(
            f"    lr={lr:<7g} {steps:5d} steps -> -log_post = {nxt.neg_log_posterior:.1f}"
            f"{'' if kept else '  (worse; discarded)'}"
        )
        if kept:
            res = nxt
        stalled = gain < POLISH_STALL_NATS
        if stalled and last_stalled:
            print("    two stages in a row bought nothing; stopping the anneal")
            break
        last_stalled = stalled
    print(f"  after polish:    -log_post = {res.neg_log_posterior:.1f}")
    return res


def model_dtype():
    """The float dtype the forward model currently runs in."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def newton_polish(fwd: ForwardModel, res: MAPResult, n_iter: int, hessian_at) -> MAPResult:
    """Modified-Newton steps from the annealed Adam point to a *genuine* mode.

    Adam gets close but never stationary: the posterior's curvature spans
    ``1e-1`` to ``4e7`` in raw units, so the shallow directions still move by
    ~0.03 after 10^4 Adam steps.  The remaining negative curvature matters
    beyond a nat or two of ``-log_post``, because the sampler below is
    preconditioned with this Hessian and a saddle point has no Laplace
    approximation at all.

    With only ~60 parameters the exact Hessian costs ~60 Hessian-vector
    products (seconds on a GPU), so a Newton step is affordable.  The Hessian
    is eigen-modified (``|lambda|`` floored at a small positive value) to make
    the step a descent direction even at a saddle, and a backtracking line
    search on ``-log_post`` guarantees a decrease.  Two or three iterations
    are enough, and the same Hessian then preconditions the sampler.
    """
    if n_iter <= 0:
        return res
    dt = model_dtype()
    theta = jnp.asarray(res.theta, dt)
    best = float(res.neg_log_posterior)
    loss = jax.jit(lambda t: -fwd.log_posterior(t))
    grad = jax.jit(jax.grad(lambda t: -fwd.log_posterior(t)))
    print(f"  modified-Newton polish ({n_iter} iterations) ...")
    for it in range(n_iter):
        t0 = time.perf_counter()
        g = np.asarray(grad(theta), np.float64)
        hess = hessian_at(theta)
        evals, evecs = np.linalg.eigh(hess)
        floor = max(float(np.abs(evals).max()) * 1e-8, 1.0 / MAX_LAPLACE_VAR)
        modified = np.maximum(np.abs(evals), floor)
        step = -(evecs @ ((evecs.T @ g) / modified))
        slope = float(g @ step)
        scale = 1.0
        for _ in range(25):  # backtracking line search (Armijo)
            trial = jnp.asarray(theta + scale * step, dt)
            value = float(loss(trial))
            if value <= best + 1e-4 * scale * slope:
                break
            scale *= 0.5
        else:
            print("    line search failed; keeping the Adam point")
            break
        print(
            f"    newton {it + 1}/{n_iter}: -log_post = {value:.2f} "
            f"(step x{scale:g}, {int((evals < 0).sum())} negative curvature directions) "
            f"[{time.perf_counter() - t0:.0f}s]"
        )
        theta, best = trial, value
    return dataclasses.replace(res, theta=theta, neg_log_posterior=best)


def run_blind_map(fwd: ForwardModel, quick: bool, n_newton: int) -> tuple[MAPResult, Callable]:
    """Blind MAP: data -> image moments -> neutral start -> multistart Adam.

    This function receives ONLY the forward model (which holds the noisy
    observation).  The injected truth is not an argument and is not used
    anywhere in the initialisation or optimisation.
    """
    print("\n--- Blind initialisation + MAP ---")
    t0 = time.perf_counter()
    model = fwd.spatial_model
    theta0_spatial = blind_initial_theta(model, fwd.observation)
    theta0 = fwd.initial_theta_from_spatial(theta0_spatial)
    shapes = model.component_shapes(theta0_spatial)
    mu0 = [np.asarray(s["mu"]).round(3).tolist() for s in shapes]
    sig0 = [np.asarray(s["sigma"]).round(3).tolist() for s in shapes]
    n0 = [float(s.get("n", np.nan)) for s in shapes]
    print(f'  moments start: centres {mu0} ", sizes {sig0} ", n = {np.round(n0, 2).tolist()}')
    print(f"  -log_post(blind start, masses unsolved) = {-float(fwd.log_posterior(theta0)):.1f}")
    kwargs = dict(n_rounds=2, steps_per_round=150, final_steps=200) if quick else {}
    # order_by_size=False: keep the ordering in one place (order_full_theta),
    # which is explicit about the nuisance tail.
    print(f"  multistart_map over {len(ARCHETYPES)} archetypes {ARCHETYPES} ...")
    res = multistart_map(fwd, theta0, ARCHETYPES, order_by_size=False, **kwargs)
    print(
        f"  multistart best: -log_post = {res.neg_log_posterior:.1f} "
        f"[{time.perf_counter() - t0:.0f}s]"
    )

    polished = polish_map(fwd, res, quick)
    if polished.neg_log_posterior < res.neg_log_posterior:
        res = polished
    # One jitted Hessian closure for the whole run: the modified-Newton polish
    # and the Laplace whitening of the sampler share it, which saves a ~40 s
    # re-trace of ``jax.hessian``.
    hessian_at = make_hessian_fn(fwd)
    res = newton_polish(fwd, res, n_newton, hessian_at)
    res = dataclasses.replace(res, theta=order_full_theta(fwd, res.theta))
    print(
        f"  blind MAP: -log_post = {res.neg_log_posterior:.1f} "
        f"[{time.perf_counter() - t0:.0f}s total]"
    )
    return res, hessian_at


def report_bounds(
    model: AdditiveComponentModel, fwd: ForwardModel, theta_map: jnp.ndarray, tol: float = 0.01
) -> list[str]:
    """Print the SPS parameters sitting within ``tol`` of a prior bound at the MAP.

    Every SPS parameter is sampled through a sigmoid onto its emulator
    training range, so a MAP pinned against a bound has a *one-sided*
    posterior that no Laplace approximation can describe: the curvature there
    is whatever the sigmoid Jacobian says, the whitening is wrong by an
    arbitrary factor, and NUTS answers with divergences.  This check is
    cheap, so it is always run.  (On the default mock nothing comes closer
    than 5% of a bound -- ``slope`` for the disk -- which is why the
    divergences seen in the 2026-09-17 runs were *not* a boundary effect but
    a metric one; see the module docstring.)

    Args:
        model: The spatial model (for the bounds and parameter names).
        fwd: Forward model (unused beyond symmetry with the other reporters).
        theta_map: Full MAP theta.
        tol: Fractional distance from a bound counted as "pinned".

    Returns:
        List of ``"component:parameter"`` labels that are pinned.
    """
    del fwd
    dec = decode_components(model, jnp.asarray(theta_map)[None])
    pinned = []
    worst = []
    for k in range(model.n_components):
        name = SHORT_NAMES[k] if k < len(SHORT_NAMES) else f"comp{k}"
        for j, pname in enumerate(model.sps_param_names):
            lo, hi = model.param_bounds[pname]
            frac = (float(dec["sps"][0, k, j]) - lo) / (hi - lo)
            edge = min(frac, 1.0 - frac)
            worst.append((edge, f"{name}:{pname}", frac))
            if edge < tol:
                pinned.append(f"{name}:{pname}")
    worst.sort()
    print("\n  MAP distance from the SPS prior bounds (fraction of the range):")
    print(
        "    closest: "
        + ", ".join(f"{label} {frac:.3f}" for _edge, label, frac in worst[:4])
        + (f"   PINNED (<{tol:.0%}): {pinned}" if pinned else f"   none within {tol:.0%}")
    )
    return pinned


def nss_summary(result: NSSResult) -> str:
    return (
        f"NSS: logZ = {result.logZ:.2f} +/- {result.logZ_err:.2f}, ESS = {result.ess:.0f} "
        f"of {result.samples.shape[0]} equal-weight samples, {result.n_steps} outer steps, "
        f"{result.n_dead} dead points, {result.samples.shape[1]} parameters."
    )


def run_sampler(fwd: ForwardModel, theta_map: jnp.ndarray, args, outdir: Path, hessian_at):
    key = jax.random.PRNGKey(args.seed + 1)
    t0 = time.perf_counter()
    if args.sampler == "nss":
        n_inner = args.num_inner_steps
        print(
            f"\n--- Nested slice sampling: num_live={args.num_live}, "
            f"num_inner_steps={n_inner}, termination={args.termination:g}, "
            f"n_samples_out={args.n_samples} ---"
        )
        print("  (live points are drawn from the prior; the MAP is not used to start NSS)")
        sampler = NSSSampler(
            fwd,
            num_live=args.num_live,
            num_inner_steps=n_inner,
            termination=args.termination,
            n_samples_out=args.n_samples,
        )
        result = sampler.run(
            key,
            checkpoint_path=outdir / "nss_ckpt",
            checkpoint_every=25,
            resume=not args.no_resume,
        )
        summary = nss_summary(result)
    else:
        metric = "dense" if args.dense_metric else "diagonal"
        print(
            f"\n--- NUTS from the blind MAP: {args.n_chains} chains x "
            f"({args.n_warmup} warmup + {args.n_samples} samples), "
            f"chain_jitter={args.chain_jitter}, max_num_doublings={args.max_doublings}, "
            f"{metric} metric ---"
        )
        if args.no_whiten:
            print("  sampling RAW theta with an identity start metric (documented failure mode)")
            sampler = NUTSSampler(
                fwd,
                n_warmup=args.n_warmup,
                n_samples=args.n_samples,
                max_num_doublings=args.max_doublings,
                dense_mass_matrix=args.dense_metric,
            )
            result = sampler.run(
                theta_map,
                key,
                n_chains=args.n_chains,
                chain_jitter=args.chain_jitter,
            )
        else:
            print(
                "  sampling whitened coordinates z (theta = MAP + L z, "
                f"{metric} metric adapted in z)"
            )
            hess = hessian_at(theta_map)
            whitened, lap_info = laplace_whitening(
                fwd, theta_map, hess=hess, max_variance=MAX_LAPLACE_VAR
            )
            if args.stage1 > 0:
                key, key1 = jax.random.split(key)
                print(
                    f"  stage 1 (exploratory): {args.n_chains} chains x "
                    f"({args.stage1} warmup + {args.stage1} draws) in the LAPLACE whitening"
                )
                stage1, _ = run_whitened_nuts(
                    fwd,
                    theta_map,
                    key1,
                    n_warmup=args.stage1,
                    n_samples=args.stage1,
                    n_chains=args.n_chains,
                    chain_jitter=args.chain_jitter,
                    max_num_doublings=args.max_doublings,
                    dense_mass_matrix=args.dense_metric,
                    whitened=whitened,
                )
                print("  stage 1: " + stage1.summary(max_num_doublings=args.max_doublings))
                whitened, post_info = posterior_whitening(
                    fwd, stage1.chains, prior_cov=lap_info["cov"]
                )
                shift = np.abs(
                    (post_info["mean"] - np.asarray(theta_map)) / lap_info["marginal_sd"]
                ).max()
                print(
                    f"  re-whitened on {post_info['n_draws']} stage-1 draws "
                    f"(shrinkage {post_info['shrinkage']:.3f} toward the Laplace covariance); "
                    f"centre moved by up to {shift:.2f} Laplace sigma; new condition number "
                    f"{post_info['condition_number']:.3g}"
                )
                print(f"  stage 2: {args.n_warmup} warmup + {args.n_samples} draws per chain")
            result, _whitened = run_whitened_nuts(
                fwd,
                theta_map,
                key,
                n_warmup=args.n_warmup,
                n_samples=args.n_samples,
                n_chains=args.n_chains,
                chain_jitter=args.chain_jitter,
                max_num_doublings=args.max_doublings,
                dense_mass_matrix=args.dense_metric,
                whitened=whitened,
            )
        summary = result.summary(max_num_doublings=args.max_doublings)
    print(f"  {summary}")
    print(f"  sampler wall time: {time.perf_counter() - t0:.0f}s")
    (outdir / "sampler_summary.txt").write_text(summary + "\n")
    return result, summary


# ---------------------------------------------------------------------------
# Decoding and recovery report
# ---------------------------------------------------------------------------


def decode_components(model: AdditiveComponentModel, thetas: jnp.ndarray) -> dict:
    """thetas (n, >=n_spatial) -> physical per-component arrays.

    ``sps`` (n, K, N_free) holds the free SPS parameters in
    ``model.sps_param_names`` order; ``shape`` maps each entry of
    ``SHAPE_KEYS`` to an (n, K) array (``sersic_n`` is NaN for non-Sersic
    profiles).
    """
    free_cols = jnp.array([model.emulator_param_names.index(p) for p in model.sps_param_names])
    nan = jnp.asarray(np.nan, model_dtype())

    def one(theta):
        shapes = model.component_shapes(theta)
        _, _, _, sps_full = model.component_params(theta)
        mu = jnp.stack([s["mu"] for s in shapes])
        sigma = jnp.stack([s["sigma"] for s in shapes])
        rho = jnp.stack([jnp.reshape(s["rho"], ()) for s in shapes])
        n_ser = jnp.stack([jnp.reshape(s.get("n", nan), ()) for s in shapes])
        r_e, q, pa = ellipse_from_cov(sigma[:, 0], sigma[:, 1], rho)
        shape = jnp.stack([mu[:, 0], mu[:, 1], r_e, q, pa, n_ser], axis=0)  # (6, K)
        return sps_full[:, free_cols], shape

    sps, shape = jax.vmap(one)(jnp.atleast_2d(thetas))
    shape_np = np.asarray(shape)
    return {
        "sps": np.asarray(sps),
        "shape": {k: shape_np[:, i, :] for i, k in enumerate(SHAPE_KEYS)},
    }


def decode_nuisance(fwd: ForwardModel, thetas: jnp.ndarray) -> dict:
    """thetas (n, n_params) -> {"sky": (n, N), "dy": (n, N), "dx": (n, N)}."""
    nb = len(BAND_NAMES)
    if fwd.nuisance is None:
        z = np.zeros((jnp.atleast_2d(thetas).shape[0], nb))
        return {"sky": z, "dy": z.copy(), "dx": z.copy()}

    def one(theta):
        blocks = fwd.nuisance.split(fwd.split_theta(theta)[1])
        return blocks["sky"], blocks["shifts"]

    sky, shifts = jax.vmap(one)(jnp.atleast_2d(thetas))
    shifts = np.asarray(shifts)
    return {"sky": np.asarray(sky), "dy": shifts[:, :, 0], "dx": shifts[:, :, 1]}


class IntervalTally:
    """Accumulates truth-in-interval counts while printing one row per parameter."""

    def __init__(self) -> None:
        self.n = 0
        self.n68 = 0
        self.n95 = 0

    def row(self, label: str, samples_1d: np.ndarray, truth: float, map_value: float) -> None:
        q2, q16, q50, q84, q97 = np.percentile(samples_1d, [2.5, 16, 50, 84, 97.5])
        in68 = bool(q16 <= truth <= q84)
        in95 = bool(q2 <= truth <= q97)
        half = max(0.5 * (q84 - q16), 1e-12)
        dev = (q50 - truth) / half
        self.n += 1
        self.n68 += in68
        self.n95 += in95
        print(
            f"    {label:22s} truth={truth:9.3f}  median={q50:9.3f}  "
            f"[{q16:9.3f},{q84:9.3f}]  in68={'yes' if in68 else ' no':>3s}  "
            f"in95={'yes' if in95 else ' no':>3s}  (med-truth)/halfwidth={dev:+6.2f}  "
            f"MAP={map_value:9.3f}"
        )

    def line(self, what: str) -> str:
        exp68, exp95 = 0.68 * self.n, 0.95 * self.n
        return (
            f"  {what:26s} inside 68%: {self.n68:3d}/{self.n:<3d} (expect ~{exp68:4.1f})   "
            f"inside 95%: {self.n95:3d}/{self.n:<3d} (expect ~{exp95:4.1f})"
        )


def report_recovery(
    model: AdditiveComponentModel,
    fwd: ForwardModel,
    samples: jnp.ndarray,
    theta_true: np.ndarray,
    theta_map: jnp.ndarray,
) -> dict:
    print("\n=== Posterior recovery (truth used for grading only) ===")
    post = decode_components(model, samples)
    tru = decode_components(model, jnp.asarray(theta_true)[None])
    mapd = decode_components(model, jnp.asarray(theta_map)[None])
    names = model.sps_param_names
    sps_tally, shape_tally = IntervalTally(), IntervalTally()

    for k, cname in enumerate(COMPONENT_NAMES[: model.n_components]):
        print(f"\n  {cname}:")
        for j, pname in enumerate(names):
            sps_tally.row(pname, post["sps"][:, k, j], tru["sps"][0, k, j], mapd["sps"][0, k, j])
        for key in SHAPE_KEYS:
            truth = float(tru["shape"][key][0, k])
            if not np.isfinite(truth):
                continue  # Sersic n with Gaussian profiles
            draws = post["shape"][key][:, k]
            map_value = float(mapd["shape"][key][0, k])
            if key == "PA_deg":  # circular: unwrap onto the branch nearest the truth
                draws = truth + wrap180(draws - truth)
                map_value = truth + float(wrap180(map_value - truth))
            shape_tally.row(SHAPE_LABELS[key], draws, truth, map_value)

    mi = model.mass_index
    tot = np.log10(np.sum(10.0 ** post["sps"][:, :, mi], axis=1))
    tot_true = float(np.log10(np.sum(10.0 ** tru["sps"][0, :, mi])))
    tot_map = float(np.log10(np.sum(10.0 ** mapd["sps"][0, :, mi])))
    print("\n  derived:")
    total_tally = IntervalTally()
    total_tally.row("log10 M_total", tot, tot_true, tot_map)

    nuis = None
    nuis_tally = IntervalTally()
    if fwd.nuisance is not None:
        nuis = decode_nuisance(fwd, samples)
        nuis_true = decode_nuisance(fwd, jnp.asarray(theta_true)[None])
        nuis_map = decode_nuisance(fwd, jnp.asarray(theta_map)[None])
        ref_band = fwd.nuisance.shift_reference_band
        if ref_band is not None:
            print(
                f"\n  instrumental nuisances (per band; shifts are relative to "
                f"{BAND_NAMES[ref_band].split('.')[-1]}, whose (dy, dx) is pinned to zero "
                "and not graded):"
            )
        else:
            print("\n  instrumental nuisances (per band):")
        for key, unit in (("sky", "nJy/px"), ("dy", "px"), ("dx", "px")):
            for b, band in enumerate(BAND_NAMES):
                if key in ("dy", "dx") and b == ref_band:
                    continue
                nuis_tally.row(
                    f"{key}[{band.split('.')[-1]}] {unit}",
                    nuis[key][:, b],
                    float(nuis_true[key][0, b]),
                    float(nuis_map[key][0, b]),
                )

    print("\n  Summary (truth inside the posterior interval):")
    print(sps_tally.line("SPS parameters"))
    print(shape_tally.line("shape parameters"))
    if fwd.nuisance is not None:
        print(nuis_tally.line("sky + shift nuisances"))
    print(
        "  Note: with 10-band photometry alone, fesc_lya, dust_bump_amplitude and the "
        "higher-order logsfr_ratio_* are weakly constrained, so wide intervals there are "
        "correct behaviour.  The per-band shifts share one exact degeneracy with the "
        "component centres (see the module docstring)."
    )
    return {"post": post, "truth": tru, "map": mapd, "nuisance": nuis}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_truth_data_model(plt, figdir, true_image, obs_flux, median_model):
    nb = len(BAND_NAMES)
    rows = (
        (true_image, "Truth (noiseless)"),
        (obs_flux, "Observed (noisy)"),
        (median_model, "Posterior median model"),
    )
    fig, axes = plt.subplots(3, nb, figsize=(2.2 * nb, 6.8))
    for b, band in enumerate(BAND_NAMES):
        scale = max(true_image[b].max() / 200.0, 1e-3)
        vmax = np.arcsinh(true_image[b].max() / scale)
        for r, (cube, lab) in enumerate(rows):
            ax = axes[r, b]
            ax.imshow(np.arcsinh(cube[b] / scale), origin="lower", cmap="viridis", vmin=0,
                      vmax=vmax)  # fmt: skip
            ax.set_xticks([])
            ax.set_yticks([])
            if b == 0:
                ax.set_ylabel(lab, fontsize=9)
        axes[0, b].set_title(band.split(".")[-1], fontsize=9)
    fig.suptitle("Truth vs observed vs posterior-median model, per band (arcsinh)", fontsize=12)
    fig.tight_layout()
    path = figdir / "truth_data_model_mosaic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_chi_maps(plt, figdir, res):
    chi = np.asarray(res["chi"])
    nb = chi.shape[0]
    ncol = (nb + 1) // 2
    fig, axes = plt.subplots(2, ncol + 1, figsize=(2.1 * (ncol + 1), 4.8))
    for i in range(2 * (ncol + 1)):
        axes.flat[i].axis("off")
    for b, band in enumerate(BAND_NAMES):
        ax = axes.flat[b + (b >= ncol)]  # leave the last column for the histogram
        ax.axis("on")
        im = ax.imshow(chi[b], origin="lower", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{band.split('.')[-1]}  $\\chi^2/N$ = {res['chi2_red_per_band'][b]:.2f}", fontsize=8
        )
    cax = axes[0, ncol]
    cax.axis("on")
    valid = chi[np.abs(chi) > 0]
    cax.hist(valid.ravel(), bins=80, range=(-5, 5), density=True, color="tab:gray")
    grid = np.linspace(-5, 5, 200)
    cax.plot(grid, np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi), "k-", lw=1, label="N(0,1)")
    cax.set_yticks([])
    cax.set_xlabel(r"$\chi$", fontsize=8)
    cax.legend(fontsize=6)
    cax.set_title("all bands", fontsize=8)
    fig.colorbar(im, ax=axes[1, ncol], fraction=0.5, pad=0.02, label=r"$\chi$")
    fig.suptitle(
        r"Residual $\chi$ = (obs - median model)/$\sigma$   "
        f"[total $\\chi^2_\\nu$ = {res['chi2_red']:.3f}, "
        f"{100 * res['frac_chi_gt_3']:.2f}% of pixels with $|\\chi| > 3$]",
        fontsize=11,
    )
    fig.tight_layout()
    path = figdir / "chi_map_mosaic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_sps_param_maps(plt, figdir, model, result, theta_true):
    names = model.sps_param_names
    truth_map = np.asarray(model.decode(jnp.asarray(theta_true), (NPIX, NPIX)))
    truth_map = truth_map.reshape(NPIX, NPIX, len(names))
    pct_maps = result.get_parameter_map((NPIX, NPIX), percentiles=[16, 50, 84])
    log_sigma_true = truth_map[:, :, model.mass_index]
    shown = log_sigma_true > log_sigma_true.max() - 3.5  # hide meaningless outskirts

    n_p = len(MAP_FIGURE_PARAMS)
    fig, axes = plt.subplots(n_p, 3, figsize=(9.5, 2.8 * n_p))
    for i, pname in enumerate(MAP_FIGURE_PARAMS):
        j = names.index(pname)
        truth_2d = np.where(shown, truth_map[:, :, j], np.nan)
        lo16, med, hi84 = (np.where(shown, np.asarray(m), np.nan) for m in pct_maps[pname])
        vmin, vmax = np.nanmin(truth_2d), np.nanmax(truth_2d)
        if pname == "log_mass":
            ylabel = r"$\log_{10}\,\Sigma_\star$ [$M_\odot$/px]"
        else:
            ylabel = "mass-weighted " + PARAM_LABELS[pname]
        for c, (img, title, cmap, kw) in enumerate(
            (
                (truth_2d, "Truth", "magma", dict(vmin=vmin, vmax=vmax)),
                (med, "Recovered median", "magma", dict(vmin=vmin, vmax=vmax)),
                ((hi84 - lo16) / 2, r"Recovered $\sigma$ (68% CI / 2)", "cividis", {}),
            )
        ):
            ax = axes[i, c]
            im = ax.imshow(img, origin="lower", cmap=cmap, **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if i == 0:
                ax.set_title(title, fontsize=10)
        axes[i, 0].set_ylabel(ylabel, fontsize=9)
    fig.suptitle(
        "Summary maps from decode(): log stellar-mass surface density and\n"
        "mass-weighted component parameters (shown where $\\Sigma_\\star$ > peak - 3.5 dex)",
        fontsize=11,
    )
    fig.tight_layout()
    path = figdir / "sps_param_maps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_sed_bulge_disk(plt, figdir, true_image, obs_flux, obs_variance, pp_images, pixels):
    wl = np.array([PIVOT_UM[b.split(".")[-1]] for b in BAND_NAMES])
    p16, p50, p84 = (np.asarray(a) for a in predictive_bands(pp_images))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (px, title) in zip(axes, pixels):
        y, x = px
        ax.fill_between(
            wl, p16[:, y, x], p84[:, y, x], color="tab:red", alpha=0.25, label="Posterior 16-84%"
        )
        ax.plot(wl, true_image[:, y, x], "k-", label="Truth (noiseless)", zorder=2)
        ax.errorbar(
            wl,
            obs_flux[:, y, x],
            yerr=np.sqrt(obs_variance[:, y, x]),
            fmt="o",
            color="tab:gray",
            label="Noisy observed",
            alpha=0.8,
        )
        ax.plot(wl, p50[:, y, x], "o-", color="tab:red", ms=3, label="Posterior median")
        ax.set_xlabel(r"Pivot wavelength [$\mu$m]")
        ax.set_ylabel("Flux [nJy / px]")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Per-pixel SEDs: truth vs observed vs posterior predictive", fontsize=12)
    fig.tight_layout()
    path = figdir / "sed_bulge_disk.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_posterior_summary_1d(plt, figdir, model, dec):
    names = model.sps_param_names
    post, tru, mapd = dec["post"]["sps"], dec["truth"]["sps"], dec["map"]["sps"]
    k_max = model.n_components
    fig, axes = plt.subplots(k_max, len(names), figsize=(2.0 * len(names), 2.4 * k_max),
                             squeeze=False)  # fmt: skip
    for k in range(k_max):
        for j, pname in enumerate(names):
            ax = axes[k, j]
            ax.hist(post[:, k, j], bins=30, color="tab:blue", alpha=0.7)
            ax.axvline(tru[0, k, j], color="k", linestyle="--", linewidth=1, label="truth")
            ax.axvline(mapd[0, k, j], color="tab:red", linestyle=":", linewidth=1, label="MAP")
            ax.set_yticks([])
            ax.tick_params(labelsize=6)
            if k == k_max - 1:
                ax.set_xlabel(pname, fontsize=6, rotation=45, ha="right")
            if j == 0:
                ax.set_ylabel(SHORT_NAMES[k] if k < len(SHORT_NAMES) else f"comp {k}", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(
        "Posterior SPS parameters per component (dashed = truth, dotted = blind MAP)", fontsize=11
    )
    fig.tight_layout()
    path = figdir / "posterior_summary_1d.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _trace_dims(model: AdditiveComponentModel) -> list[tuple[int, str]]:
    """A few informative theta indices (log r_e and mass of each component)."""
    dims = []
    for k in range(min(model.n_components, 2)):
        sl = model.component_slices[k]
        name = SHORT_NAMES[k] if k < len(SHORT_NAMES) else f"comp {k}"
        dims.append((sl.start + 2, f"{name} log_sigma_y"))
        dims.append((sl.start + model.n_shape_params[k] + model.mass_index, f"{name} log_mass raw"))
    return dims


def plot_sampler_diagnostics(plt, figdir, result, sampler_name, model, summary):
    path = figdir / "sampler_diagnostics.png"
    if sampler_name == "nss":
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
        if result.infos is not None:
            logl = np.asarray(result.infos.particles.loglikelihood)
            it = np.arange(logl.size)
            axes[0].plot(it, logl, linewidth=0.6)
            axes[0].set_ylabel("log L of dead point")
            lo = np.percentile(logl, 5)
            axes[0].set_ylim(lo, logl.max() + 0.05 * (logl.max() - lo))
            if result.log_weights is not None:
                lw = np.asarray(result.log_weights)
                axes[1].plot(it[: lw.size], np.exp(lw - lw.max()), linewidth=0.6,
                             color="tab:orange")  # fmt: skip
                axes[1].set_ylabel("posterior weight (rel.)")
        axes[1].set_xlabel("dead-point index (nested-sampling iteration)")
        fig.suptitle(
            f"NSS: logZ = {result.logZ:.2f} +/- {result.logZ_err:.2f}, ESS = {result.ess:.0f}, "
            f"{result.n_steps} steps",
            fontsize=11,
        )
    else:
        chains = np.asarray(result.chains) if result.chains is not None else None
        if chains is None:
            chains = np.asarray(result.samples)[None]
        dims = [(d, lab) for d, lab in _trace_dims(model) if d < chains.shape[2]]
        fig, axes = plt.subplots(2, max(len(dims), 3), figsize=(3.6 * max(len(dims), 3), 6.4))
        for ax, (d, lab) in zip(axes[0], dims):
            for c in range(chains.shape[0]):
                ax.plot(chains[c, :, d], linewidth=0.5, alpha=0.8, label=f"chain {c}")
            ax.set_title(f"theta[{d}] = {lab}", fontsize=9)
            ax.set_xlabel("draw", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0, 0].legend(fontsize=6)
        diag = result.diagnostics
        rhat = np.asarray(diag.get("rhat", []), dtype=float)
        ess_v = np.asarray(diag.get("ess", []), dtype=float)
        axes[1, 0].hist(rhat[np.isfinite(rhat)], bins=25, color="tab:blue")
        axes[1, 0].axvline(1.01, color="k", ls="--", lw=1)
        axes[1, 0].set_xlabel("split R-hat (per parameter)", fontsize=8)
        axes[1, 1].hist(ess_v[np.isfinite(ess_v)], bins=25, color="tab:green")
        axes[1, 1].axvline(100 * chains.shape[0], color="k", ls="--", lw=1)
        axes[1, 1].set_xlabel("bulk ESS (per parameter)", fontsize=8)
        steps = getattr(result.infos, "num_integration_steps", None)
        ax = axes[1, 2]
        if steps is not None:
            ax.hist(np.asarray(steps).ravel(), bins=30, color="tab:orange")
            ax.set_xlabel("leapfrog steps per draw", fontsize=8)
            ax.set_yscale("log")
        for ax in axes[1, 3:]:
            ax.axis("off")
        for ax in axes.flat:
            ax.tick_params(labelsize=7)
        fig.suptitle("NUTS diagnostics\n" + summary, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _radial_bins(centre_px, edges_px: np.ndarray):
    """Bin index, in-range mask and per-bin pixel counts for a radial profile."""
    r = _radius_px(centre_px).ravel()
    nbins = edges_px.size - 1
    idx = np.digitize(r, edges_px) - 1
    ok = (idx >= 0) & (idx < nbins)
    counts = np.bincount(idx[ok], minlength=nbins)
    return idx, ok, counts


def _radial_profiles(cube: np.ndarray, centre_px, edges_px: np.ndarray) -> np.ndarray:
    """Azimuthal means of ``cube`` (..., H, W) in the given radial bins."""
    cube = np.asarray(cube)
    idx, ok, counts = _radial_bins(centre_px, edges_px)
    nbins = edges_px.size - 1
    flat = cube.reshape(-1, idx.size)
    out = np.stack([np.bincount(idx[ok], weights=f[ok], minlength=nbins) for f in flat])
    return (out / np.maximum(counts, 1)).reshape(*cube.shape[:-2], nbins)


def plot_component_profiles(
    plt, figdir, fwd, model, emulator, theta_true, obs_flux, sigma, pp_images, comp_images, centre
):
    b = BAND_NAMES.index(REF_BAND)
    edges = np.linspace(0.0, NPIX / 2.0, 41)
    r_arcsec = 0.5 * (edges[1:] + edges[:-1]) * PIXEL_SCALE

    total_prof = _radial_profiles(pp_images[:, b], centre, edges)  # (n, nbins)
    lo, med, hi = np.percentile(total_prof, [16, 50, 84], axis=0)
    comp_prof = _radial_profiles(comp_images[:, :, b], centre, edges)  # (n, K, nbins)

    th = jnp.asarray(theta_true, dtype=model_dtype())
    theta_true_spatial = fwd.split_theta(th)[0]
    truth_total = _radial_profiles(np.asarray(fwd._model_image(th)[b]), centre, edges)
    truth_comp = np.asarray(model.component_images(theta_true_spatial, emulator))[:, b]
    truth_comp_prof = _radial_profiles(truth_comp, centre, edges)
    data_prof = _radial_profiles(obs_flux[b], centre, edges)
    counts = np.maximum(_radial_bins(centre, edges)[2], 1)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        r_arcsec, data_prof, yerr=sigma[b] / np.sqrt(counts), fmt="o", ms=3, color="tab:gray",
        label="data", zorder=1,
    )  # fmt: skip
    ax.fill_between(r_arcsec, lo, hi, color="k", alpha=0.25, label="posterior 16-84% (total)")
    ax.plot(r_arcsec, med, "k-", lw=1.5, label="posterior median (total, convolved)")
    ax.plot(r_arcsec, truth_total, "k--", lw=1.2, label="truth (total, convolved)")
    for k, col in zip(range(model.n_components), ["tab:orange", "tab:blue", "tab:green"]):
        name = SHORT_NAMES[k] if k < len(SHORT_NAMES) else f"comp {k}"
        clo, cmed, chi_ = np.percentile(comp_prof[:, k], [16, 50, 84], axis=0)
        ax.fill_between(r_arcsec, clo, chi_, color=col, alpha=0.25)
        ax.plot(r_arcsec, cmed, "-", color=col, lw=1.2, label=f"{name} (posterior, unconvolved)")
        ax.plot(r_arcsec, truth_comp_prof[k], "--", color=col, lw=1.0, label=f"{name} (truth)")
    ax.set_yscale("log")
    ax.set_ylim(sigma[b] / 100.0, 30.0 * float(np.max(truth_total)))
    ax.axhline(sigma[b], color="tab:gray", linestyle=":", linewidth=1)
    ax.text(r_arcsec[-1], sigma[b] * 1.15, r"1$\sigma$ per pixel", ha="right", fontsize=8)
    ax.set_xlabel('radius from the bulge centre [arcsec]  (1 px = 0.03")')
    ax.set_ylabel(f"{REF_BAND.split('.')[-1]} surface brightness [nJy / px]")
    ax.set_title("Component radial profiles with the posterior-predictive 16-84% band")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = figdir / "component_profiles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def run_model_comparison(args, obs, emulator, psf_model, convolver, outdir: Path) -> str:
    ks = [int(k) for k in args.compare_ks.split(",")]
    print(f"\n--- Model comparison over K = {ks} (same priors, same sampler settings) ---")

    def make_forward_model(k: int) -> ForwardModel:
        model_k = build_spatial_model(emulator, k, args.profiles, args.oversample)
        return build_forward_model(
            obs,
            model_k,
            emulator,
            convolver,
            args.model_err_frac,
            build_nuisance(args.nuisance, args.all_band_shifts),
        )

    sampler_kwargs = dict(
        num_live=args.num_live,
        num_inner_steps=args.num_inner_steps,
        termination=args.termination,
        n_samples_out=args.n_samples,
    )
    rows = compare_n_components(
        make_forward_model,
        ks,
        jax.random.PRNGKey(args.seed + 2),
        sampler_kwargs=sampler_kwargs,
        n_chi2_samples=64,
    )
    table = bayes_factor_table(rows)
    print(table)
    (outdir / "bayes_factors.txt").write_text(table + "\n")
    print(f"  Saved {outdir / 'bayes_factors.txt'}")
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _phys_record(model: AdditiveComponentModel, fwd: ForwardModel, theta) -> dict:
    d = decode_components(model, jnp.asarray(theta)[None])
    n = decode_nuisance(fwd, jnp.asarray(theta)[None])
    out: dict = {}
    for k in range(model.n_components):
        name = SHORT_NAMES[k] if k < len(SHORT_NAMES) else f"comp{k}"
        out[name] = {p: float(d["sps"][0, k, j]) for j, p in enumerate(model.sps_param_names)}
        out[name + "_shape"] = {
            key: float(d["shape"][key][0, k])
            for key in SHAPE_KEYS
            if np.isfinite(d["shape"][key][0, k])
        }
    if fwd.nuisance is not None:
        out["nuisance"] = {
            "sky_nJy_per_px": n["sky"][0].tolist(),
            "dy_px": n["dy"][0].tolist(),
            "dx_px": n["dx"][0].tolist(),
            "band_names": BAND_NAMES,
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick", action="store_true", help="small smoke run (num_live 100, fewer steps)"
    )
    parser.add_argument(
        "--profiles",
        choices=["sersic", "gaussian"],
        default="sersic",
        help="light profile family for every component (default: sersic)",
    )
    parser.add_argument(
        "--oversample", type=int, default=3, help="sub-samples per pixel side when rendering"
    )
    parser.add_argument(
        "--no-nuisance",
        dest="nuisance",
        action="store_false",
        help="do not inject and do not fit per-band sky levels and shifts",
    )
    parser.add_argument("--noise-scale", type=float, default=1.0, help="multiply the noise sigma")
    parser.add_argument(
        "--model-err-frac", type=float, default=0.05, help="fractional model-error floor"
    )
    parser.add_argument(
        "--sampler",
        choices=["auto", "nss", "nuts"],
        default="auto",
        help="auto = NUTS when the nuisances are fitted (64 parameters), NSS otherwise",
    )
    parser.add_argument("--num-live", type=int, default=None, help="NSS live points (default 500)")
    parser.add_argument(
        "--num-inner-steps", type=int, default=None, help="NSS slice steps (default 3*n_params)"
    )
    parser.add_argument(
        "--termination", type=float, default=1e-3, help="NSS stop when logZ_live-logZ < this"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="ignore an existing NSS checkpoint"
    )
    parser.add_argument("--n-chains", type=int, default=4, help="NUTS chains")
    parser.add_argument("--n-warmup", type=int, default=None, help="NUTS warmup (default 500)")
    parser.add_argument("--max-doublings", type=int, default=10, help="NUTS max tree doublings")
    parser.add_argument(
        "--chain-jitter",
        type=float,
        default=0.3,
        help="NUTS per-chain start jitter, in posterior sigmas of the Laplace metric",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="keep the forward model in float32 (documented sampler failure mode)",
    )
    parser.add_argument(
        "--stage1",
        type=int,
        default=None,
        help="exploratory stage-1 draws before re-whitening on them (0 disables the two-stage run)",
    )
    parser.add_argument(
        "--diagonal-metric",
        dest="dense_metric",
        action="store_false",
        help="adapt a diagonal (not dense) mass matrix in the whitened space",
    )
    parser.add_argument(
        "--no-whiten",
        action="store_true",
        help="sample raw theta with an identity metric instead of whitening (fails here)",
    )
    parser.add_argument(
        "--n-newton",
        type=int,
        default=3,
        help="modified-Newton iterations after the Adam anneal (0 disables)",
    )
    parser.add_argument(
        "--all-band-shifts",
        action="store_true",
        help="fit a shift in every band (exactly degenerate with the component centres)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="NUTS samples per chain / NSS equal-weight samples out (default 1000)",
    )
    parser.add_argument(
        "--compare-k", action="store_true", help="also run compare_n_components over --compare-ks"
    )
    parser.add_argument("--compare-ks", type=str, default="1,2", help="component counts to compare")
    return parser.parse_args()


def resolve_run_settings(args, n_params: int) -> None:
    """Pick the sampler and fill in the sizes left at their defaults.

    ``--sampler auto`` chooses **NUTS** as soon as the instrumental nuisances
    are fitted.  Nested sampling starts from the prior, and the prior volume it
    has to compress grows exponentially with the dimension: with the 30
    nuisance parameters added (64 in total) a from-the-prior NSS run on this
    image needs many thousands of outer steps, while NUTS starts from the MAP
    and only has to explore the posterior.  NSS remains the default for the
    34-parameter ``--no-nuisance`` model, where the evidence is affordable and
    worth having, and ``--sampler nss`` always forces it.

    NSS cost model: the per-step cost is proportional to
    ``num_delete x num_inner_steps`` (both vectorised on the GPU, so the
    scaling with ``num_delete`` is sub-linear), and the *number* of steps is
    essentially independent of ``num_live`` (each step replaces
    ``num_live / 10`` points).  ``num_inner_steps`` therefore sets the wall
    clock almost linearly; blackjax recommends ``>= max(5, 2 * dim)`` and
    fewer steps bias ``logZ`` upward, so the default here is ``1.5 * n_params``
    (a compromise that was measured at roughly half the cost of ``3 *
    n_params``), and ``--quick`` halves it again for a smoke test.
    """
    if args.sampler == "auto":
        args.sampler = "nuts" if args.nuisance else "nss"
    if args.sampler == "nss" and not args.float32:
        # Library gap (2026-09-18): NSSSampler.run hard-casts its live points
        # to float32 (`inference/nss_sampler.py:360`, and the output samples at
        # :522), while a float64 forward model makes blackjax's slice kernel
        # return a float64 position -- `jax.lax.while_loop` then rejects the
        # carry as a dtype mismatch.  Nested slice sampling is derivative-free,
        # so it never needed float64 in the first place (the float64 promotion
        # exists for the NUTS gradients); run the whole NSS path in float32
        # until the cast follows `jax.config.jax_enable_x64`.
        print(
            "  NOTE: forcing --float32 for the NSS path (nss_sampler.py:360 casts the live "
            "points to float32; a float64 model then trips a lax.while_loop dtype check). "
            "Slice sampling needs no gradients, so this costs nothing."
        )
        args.float32 = True
    if args.num_live is None:
        args.num_live = 100 if args.quick else 500
    if args.n_samples is None:
        if args.sampler == "nuts":
            args.n_samples = 300 if args.quick else 1000
        else:
            args.n_samples = 300 if args.quick else 1000
    if args.stage1 is None:
        args.stage1 = 300 if args.quick else 500
    if args.n_warmup is None:
        # A DENSE metric has to estimate d(d+1)/2 numbers, so the last (and
        # only useful) adaptation window must hold many more than d draws:
        # 500 is the minimum that worked at d = 62, 1000 is comfortable.
        args.n_warmup = 500 if args.quick else 1000
    if args.num_inner_steps is None:
        # Never go below ~1 x n_params: with 0.5 * n_params the slice sampler
        # stops decorrelating the live points, the live set collapses onto one
        # likelihood value and the run *stops early* with a silently wrong
        # evidence.  Measured on 2026-09-18: the K=2 no-nuisance model
        # (34 parameters, num_inner_steps=17) ended after 172 steps with
        # `logZ_live - logZ` NEGATIVE and logZ = -8.9e4 against the converged
        # -7.1e3, while K=1 (17 parameters, 17 inner steps = 1 x d) converged
        # normally in 723 steps.  blackjax recommends >= 2 * n_params.
        args.num_inner_steps = max(10, n_params if args.quick else (3 * n_params) // 2)


def main():
    args = parse_args()
    outdir: Path = args.outdir
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    print(f"JAX backend: {jax.default_backend()}   devices: {jax.devices()}")
    if jax.default_backend() == "cpu":
        print("  WARNING: running on CPU; this demo is meant for a GPU node (see docstring).")
    t_start = time.perf_counter()

    emulator, psf_model = load_emulator_and_psf()
    model = build_spatial_model(emulator, 2, args.profiles, args.oversample)
    nuisance = build_nuisance(args.nuisance, args.all_band_shifts)
    convolver = PSFConvolver(psf_model, image_shape=(NPIX, NPIX), pad=True)
    n_params = model.n_params + (nuisance.n_params if nuisance else 0)
    resolve_run_settings(args, n_params)
    print(
        f"Spatial model: AdditiveComponentModel K={model.n_components}, "
        f'profiles={[p.name for p in model.profile_objects]}, pixel_scale={model.pixel_scale}"/px, '
        f"normalisation={model.normalisation}, oversample={model.oversample}"
    )
    print(
        f"  {model.n_params} spatial + {nuisance.n_params if nuisance else 0} nuisance "
        f"= {n_params} free parameters; fixed {model.fixed_params}"
    )
    print(f"  nuisance: {nuisance}")
    print(f"  PSF convolution: padded linear, FFT grid {convolver.padded_shape}")

    # ---- truth + mock (the ONLY place theta_true enters the data path) ----
    theta_true_spatial = build_truth_spatial_theta(model)
    theta_true_nuisance = draw_nuisance_truth(nuisance, args.seed)
    theta_true = np.concatenate([theta_true_spatial, theta_true_nuisance]).astype(np.float32)
    true_astro, true_image = synthesise_truth_images(
        model, emulator, convolver, theta_true_spatial, nuisance, theta_true_nuisance
    )
    assert np.all(np.isfinite(true_image)) and true_astro.max() > 0
    centre_px = (
        BULGE_SHAPE["mu_y"] / PIXEL_SCALE + (NPIX - 1) / 2.0,
        BULGE_SHAPE["mu_x"] / PIXEL_SCALE + (NPIX - 1) / 2.0,
    )
    sigma = noise_sigma(true_astro, centre_px, args.noise_scale)
    flux_noisy, variance = add_noise(true_image, sigma, args.seed)
    sky_true = theta_true_nuisance[: len(BAND_NAMES)] if nuisance else np.zeros(0)
    print_noise_summary(true_astro, sigma, centre_px, sky_true)

    frac_in_frame = float(np.sum(np.asarray(model.profiles(jnp.asarray(theta_true_spatial)))[0]))
    print(
        f"  analytic normalisation: {100 * frac_in_frame:.2f}% of the bulge's total flux is "
        "rendered inside the frame (the rest is lost to the cutout and the unresolved core)"
    )

    truth_record = {
        "theta_true": theta_true.tolist(),
        "theta_layout": (
            f"per component [mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho"
            f"{', log_n' if args.profiles == 'sersic' else ''}, sps_raw...] in arcsec units; "
            "sps in sps_param_names order; then the nuisance block [sky (N), dy0, dx0, ...]"
        ),
        "sps_param_names": model.sps_param_names,
        "profiles": [p.name for p in model.profile_objects],
        "pixel_scale_arcsec": PIXEL_SCALE,
        "normalisation": model.normalisation,
        "oversample": model.oversample,
        "bulge": BULGE_PHYS,
        "bulge_shape": BULGE_SHAPE,
        "disk": DISK_PHYS,
        "disk_shape": DISK_SHAPE,
        "shape_units": "mu and r_e in arcsec (r_e = sqrt(a*b)), PA in deg from +x towards +y",
        "log_mass_is_component_total": True,
        "redshift": TRUE_REDSHIFT,
        "npix": NPIX,
        "band_names": BAND_NAMES,
        "noise_sigma_nJy": sigma.tolist(),
        "noise_definition": (
            f'sigma_b = <I_b(r = {SNR_RADIUS}")> / {SNR_AT_RADIUS} x {args.noise_scale}'
        ),
        "model_err_frac": args.model_err_frac,
        "seed": args.seed,
        "n_params": n_params,
    }
    if nuisance is not None:
        blocks_true = nuisance.split(jnp.asarray(theta_true_nuisance))
        truth_record["nuisance"] = {
            "sky_nJy_per_px": np.asarray(blocks_true["sky"]).tolist(),
            "dy_px": np.asarray(blocks_true["shifts"])[:, 0].tolist(),
            "dx_px": np.asarray(blocks_true["shifts"])[:, 1].tolist(),
            "shift_reference_band": (
                None
                if nuisance.shift_reference_band is None
                else BAND_NAMES[nuisance.shift_reference_band]
            ),
            "sky_true_sigma": SKY_TRUE_SIGMA,
            "shift_true_sigma": SHIFT_TRUE_SIGMA,
            "prior_sigmas": {
                "sky": nuisance.sky_prior_sigma,
                "shift": nuisance.shift_prior_sigma,
            },
        }
    with open(outdir / "truth.json", "w") as f:
        json.dump(truth_record, f, indent=2)

    obs = ObservationCube(
        flux=flux_noisy,
        variance=variance,
        mask=np.ones_like(flux_noisy),
        band_names=BAND_NAMES,
        pixel_scale=PIXEL_SCALE,
    )
    fwd = build_forward_model(obs, model, emulator, convolver, args.model_err_frac, nuisance)
    print(f"Likelihood: Gaussian with model_error_frac = {args.model_err_frac}")
    if not args.float32:
        fwd = promote_to_float64(fwd)
        # Everything downstream (fit, decoding, figures) uses the promoted copies.
        model, emulator, convolver = fwd.spatial_model, fwd.emulator, fwd.convolver
        theta_true = theta_true.astype(np.float64)
        theta_true_spatial = theta_true_spatial.astype(np.float64)
        print("  precision: float64 (JAX x64 enabled) -- see promote_to_float64")
    else:
        print("  precision: float32 -- expect the sampler to fail (see promote_to_float64)")

    # ---- blind fit: no truth information from here until the report ----
    map_res, hessian_at = run_blind_map(fwd, args.quick, args.n_newton)
    theta_map = map_res.theta
    report_bounds(model, fwd, theta_map)
    # Truth-based grading of the MAP (reporting only; the fit is already done).
    nlp_true = -float(fwd.log_posterior(jnp.asarray(theta_true)))
    print(
        f"  check: -log_post(blind MAP) = {map_res.neg_log_posterior:.1f} vs "
        f"-log_post(truth) = {nlp_true:.1f}  "
        f"(MAP - truth = {map_res.neg_log_posterior - nlp_true:+.1f})"
    )
    with open(outdir / "map.json", "w") as f:
        json.dump(
            {
                "theta_map": np.asarray(theta_map).tolist(),
                "neg_log_posterior": map_res.neg_log_posterior,
                "neg_log_posterior_truth": nlp_true,
                "n_steps_per_start": map_res.n_steps,
                "archetypes": ARCHETYPES,
                **_phys_record(model, fwd, theta_map),
            },
            f,
            indent=2,
        )

    result, summary = run_sampler(fwd, theta_map, args, outdir, hessian_at)

    # ---- break label symmetry: compact component first in every sample ----
    samples_ordered = jax.vmap(lambda t: order_full_theta(fwd, t))(result.samples)
    result_ordered = dataclasses.replace(result, samples=samples_ordered)
    if getattr(result, "chains", None) is not None:
        result_ordered = dataclasses.replace(
            result_ordered,
            chains=jax.vmap(jax.vmap(lambda t: order_full_theta(fwd, t)))(result.chains),
        )
    result_ordered.to_hdf5(outdir / "posterior.hdf5")
    print(f"  Saved posterior: {outdir / 'posterior.hdf5'}")

    # ---- library posterior-predictive products ----
    print("\n--- Posterior-predictive products ---")
    t_pp = time.perf_counter()
    res = residual_summary(fwd, samples_ordered, n_max=100)
    pp_images = np.asarray(model_image_samples(fwd, samples_ordered, n_max=100))
    comp_images = np.asarray(component_image_samples(fwd, samples_ordered, n_max=50))
    median_model = np.asarray(res["median_model"])
    print(
        f"  residual_summary: chi2_red = {res['chi2_red']:.4f} "
        f"(chi2 = {res['chi2']:.1f}, dof = {res['dof']}), "
        f"{100 * res['frac_chi_gt_3']:.3f}% of pixels with |chi| > 3, "
        f"{res['n_samples_used']} samples  [{time.perf_counter() - t_pp:.0f}s]"
    )
    print(
        "  per-band chi2/N: "
        + ", ".join(
            f"{b.split('.')[-1]} {v:.3f}" for b, v in zip(BAND_NAMES, res["chi2_red_per_band"])
        )
    )

    # ---- report ----
    dec = report_recovery(model, fwd, samples_ordered, theta_true, theta_map)
    if isinstance(result, NSSResult):
        print(f"\n  {nss_summary(result)}")

    # ---- figures ----
    print("\n--- Figures ---")
    plt = _setup_mpl()
    plot_truth_data_model(plt, figdir, true_image, flux_noisy, median_model)
    plot_chi_maps(plt, figdir, res)
    thin = max(1, samples_ordered.shape[0] // 200)
    plot_sps_param_maps(
        plt,
        figdir,
        model,
        dataclasses.replace(result_ordered, samples=samples_ordered[::thin]),
        theta_true_spatial,
    )
    disk_px = (
        int(round(centre_px[0])),
        int(round(centre_px[1] + DISK_SHAPE["r_e"] / PIXEL_SCALE)),
    )
    plot_sed_bulge_disk(
        plt,
        figdir,
        true_image,
        flux_noisy,
        variance,
        pp_images,
        [
            ((int(round(centre_px[0])), int(round(centre_px[1]))), "Bulge centre pixel"),
            (disk_px, f'Disk pixel (r = {DISK_SHAPE["r_e"]}")'),
        ],
    )
    plot_posterior_summary_1d(plt, figdir, model, dec)
    plot_sampler_diagnostics(plt, figdir, result, args.sampler, model, summary)
    plot_component_profiles(
        plt,
        figdir,
        fwd,
        model,
        emulator,
        theta_true,
        flux_noisy,
        sigma,
        pp_images,
        comp_images,
        centre_px,
    )

    if args.compare_k:
        run_model_comparison(args, obs, emulator, psf_model, convolver, outdir)

    print(f"\nAll done in {time.perf_counter() - t_start:.0f}s. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
