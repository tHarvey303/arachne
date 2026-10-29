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

Outputs in ``--outdir``: ``metrics.json``, ``maps.npz`` and ``figures/``.

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
FIELDS = ["log_mass", "Av", "log10metallicity", "logsfr_ratio_0"]
NEUTRAL = {"Av": 0.5, "log10metallicity": -2.5, "logsfr_ratio_0": 0.0}
FIXED = {
    "redshift": REDSHIFT, "slope": 0.0, "fesc_lya": 0.2, "dust_bump_amplitude": 1.0,
    **{f"logsfr_ratio_{i}": 0.0 for i in range(1, 5)},
}  # fmt: skip
# Prior hyperparameters (fixed for now): break scale in px, rms in parameter units.
CORR_LENGTHS = {"log_mass": 3.0, "Av": 5.0, "log10metallicity": 12.0, "logsfr_ratio_0": 4.0}
AMPLITUDES = {"log_mass": 0.25, "Av": 0.3, "log10metallicity": 0.15, "logsfr_ratio_0": 0.5}
LEVEL_SIGMAS = {"log_mass": 1.0, "Av": 1.0, "log10metallicity": 0.5, "logsfr_ratio_0": 1.0}
LABELS = {
    "log_mass": r"$\log M_*$ / px",
    "Av": r"$A_V$",
    "log10metallicity": r"$\log Z$",
    "logsfr_ratio_0": r"logsfr_ratio_0",
}


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
    """Face-on spiral: maps (4, n, n) in FIELDS order."""
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
    return np.stack([logm, av, logz, r0]).astype(np.float32)


def build_model(n, mass_mean, mean_values=None, corr_scale=1.0, nu=1.5):
    """CorrelatedFieldMap with the demo's hyperparameters (ell scaled by ``corr_scale``)."""
    means = {"log_mass": mass_mean}
    means.update(mean_values or NEUTRAL)
    return CorrelatedFieldMap(
        image_shape=(n, n),
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        field_params=FIELDS,
        mass_param="log_mass",
        fixed_params=FIXED,
        corr_lengths={k: corr_scale * v for k, v in CORR_LENGTHS.items()},
        nus=nu,
        amplitudes=AMPLITUDES,
        level_sigmas=LEVEL_SIGMAS,
        mean_maps=means,
        variance_fraction=0.99,
        pixel_scale=PIXEL_SCALE,
    )


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


def newton_cg(fm, theta, max_iter=30, cg_iter=100, tol=0.05, mu0=1.0):
    """Matrix-free damped (Levenberg-Marquardt) Newton on -log_posterior.

    Each step solves ``(H + mu I) p = -g`` by conjugate gradients using
    Hessian-vector products only (``cg_iter`` HVPs instead of the ``d`` a full
    Hessian needs), then accepts it if ``-log_post`` drops (``mu /= 3``) or
    rejects it (``mu *= 10``).  Stops when a step gains less than ``tol`` nat.
    """
    neg_lp = jax.jit(lambda t: -fm.log_posterior(t))
    grad_fn = jax.grad(lambda t: -fm.log_posterior(t))
    grad = jax.jit(grad_fn)

    @jax.jit
    def solve(t, g, mu):
        def hvp(v):
            return jax.jvp(grad_fn, (t,), (v,))[1] + mu * v

        p, _ = jax.scipy.sparse.linalg.cg(hvp, -g, maxiter=cg_iter)
        return p

    mu = mu0
    f = float(neg_lp(theta))
    history = []
    for it in range(max_iter):
        g = grad(theta)
        p = solve(theta, g, jnp.asarray(mu, theta.dtype))
        cand = theta + p
        f_new = float(neg_lp(cand))
        gnorm = float(jnp.linalg.norm(g))
        history.append({"iter": it, "neg_log_post": f, "grad_norm": gnorm, "mu": mu})
        print(f"  Newton-CG {it}: -log_post {f:.3f} -> {f_new:.3f}  |grad| {gnorm:.3g}  "
              f"mu {mu:.2g}", flush=True)  # fmt: skip
        if np.isfinite(f_new) and f_new < f:
            gain = f - f_new
            theta, f, mu = cand, f_new, max(mu / 3.0, 1e-6)
            if gain < tol:
                break
        else:
            mu *= 10.0
            if mu > 1e8:
                break
    history.append({"iter": len(history), "neg_log_post": f,
                    "grad_norm": float(jnp.linalg.norm(grad(theta))), "mu": mu})  # fmt: skip
    return theta, history


def laplace_posterior(fm, model, theta_map, n_samples, chunk, key, max_newton=30):
    """Newton-CG from the MAP to the mode, one full Hessian there, Gaussian map samples.

    Run with ``--float64``: the Hessian of a 1e4-pixel likelihood is round-off
    dominated in float32.

    Returns:
        Dict with the polished theta, map samples (S, N_fields, H, W), the
        Laplace evidence and diagnostics.
    """
    from arachne.inference.laplace import laplace_covariance, make_hessian_fn

    dtype = jnp.zeros(()).dtype
    neg_lp = jax.jit(lambda t: -fm.log_posterior(t))
    t0 = time.time()
    theta, history = newton_cg(fm, jnp.asarray(theta_map, dtype), max_iter=max_newton)
    t_newton = time.time() - t0
    t0 = time.time()
    hess = make_hessian_fn(fm, chunk_size=chunk)(theta)
    t_hess = time.time() - t0
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
        "chol": chol,
        "newton_s": t_newton,
        "hessian_s": t_hess,
        "newton_history": history,
        "log_evidence_laplace": log_z,
        "eig_min": float(info["eigenvalues"].min()),
        "eig_max": float(info["eigenvalues"].max()),
        "n_negative": int(info["n_negative"]),
        "n_floored": int(info["n_floored"]),
    }


def nuts_posterior(fm, model, theta, chol, key, n_warmup, n_samples, n_chains, max_maps=1000):
    """Whitened NUTS (theta = mode + L z, L from the Laplace Hessian), diagonal metric in z.

    A dense metric cannot be adapted at d ~ 4e3, so the Laplace whitening does the
    conditioning and the warmup only adapts a diagonal.  Returns map samples
    (thinned to ``max_maps``) and convergence diagnostics in theta coordinates.
    """
    from arachne.inference.laplace import WhitenedLogDensity, run_whitened_nuts

    dtype = jnp.zeros(()).dtype
    whitened = WhitenedLogDensity(fm, jnp.asarray(theta, dtype), jnp.asarray(chol, dtype))
    t0 = time.time()
    result, _ = run_whitened_nuts(
        fm, theta, key, whitened=whitened, n_warmup=n_warmup, n_samples=n_samples,
        n_chains=n_chains, dense_mass_matrix=False,
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
        "runtime_s": runtime,
        "max_rhat": float(np.nanmax(rhat)),
        "frac_rhat_gt_1.01": float(np.mean(rhat > 1.01)),
        "min_ess": float(np.nanmin(ess_)),
        "median_ess": float(np.nanmedian(ess_)),
        "summary": result.summary(),
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
    ap.add_argument("--nu", type=float, default=1.5, help="Matern nu of the fit model")
    ap.add_argument("--laplace", action="store_true", help="Laplace posterior at the MAP")
    ap.add_argument("--n-laplace", type=int, default=400, help="Laplace samples")
    ap.add_argument("--nuts", action="store_true", help="whitened NUTS after the Laplace step")
    ap.add_argument("--nuts-warmup", type=int, default=1000)
    ap.add_argument("--nuts-samples", type=int, default=500)
    ap.add_argument("--nuts-chains", type=int, default=4)
    ap.add_argument("--hess-chunk", type=int, default=32, help="HVP rows per Hessian batch")
    args = ap.parse_args()
    set_bands(args.bands)
    if args.float64:
        jax.config.update("jax_enable_x64", True)
    if args.quick:
        args.rounds, args.steps = 2, 300
    outdir = Path(args.outdir) / f"{args.truth}_{args.bands}"
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n = args.size

    emulator, psf = load_emulator_and_psf(args)
    print(f"emulator: {type(emulator).__name__}; PSFs: {'files' if args.psf_dir else 'Gaussian'}")

    # ----- truth
    disk = disk_mass_map(n, log_mtot=10.0, rd_px=0.11 * n)
    truth_model = build_model(n, disk)
    if args.truth == "prior":
        theta_truth = truth_model.sample_prior(jax.random.PRNGKey(args.seed), 1)[0]
        theta_truth = theta_truth.at[: truth_model.n_fields].set(0.0)  # around the means
        truth_maps = np.asarray(truth_model.field_maps(theta_truth))
        render_model = truth_model
    else:
        truth_maps = spiral_truth(n, rng)
        render_model = FixedMapsModel(truth_maps, truth_model)
        theta_truth = truth_model.initial_theta()
    obs, clean, sigma = make_observation(emulator, psf, render_model, theta_truth, args.snr, rng)

    # ----- blind start: mass mean from the reddest band at neutral SPS values
    ref = {**FIXED, **NEUTRAL}
    # The mean map must stay smooth: structure in it is not penalised by the prior.
    mass_mean = mass_map_from_image(obs, emulator, ref, band=len(BANDS) - 1, smooth_sigma_px=1.0)
    model = build_model(n, mass_mean, corr_scale=args.corr_scale, nu=args.nu)
    print(model.summary())
    fm = ForwardModel.build(obs, psf, model, emulator)
    theta0 = model.initial_theta()
    if args.mass_start == "wiener":
        # Sharp detail enters the *starting coefficients*, through the field prior.
        theta0 = model.project_to_field(
            theta0, "log_mass", wiener_mass_map(obs, psf, emulator, ref)
        )
    start_maps = np.asarray(model.field_maps(theta0))

    t0 = time.time()
    res = find_map(fm, theta0, n_rounds=args.rounds, steps_per_round=args.steps, lr=0.05,
                   final_steps=args.steps, final_lr=0.01)  # fmt: skip
    runtime = time.time() - t0
    fit_maps = np.asarray(model.field_maps(res.theta))
    model_img = np.asarray(fm._model_image(res.theta))
    chi = (np.asarray(obs.flux) - model_img) / sigma[:, None, None]

    weight = 10.0 ** truth_maps[0]
    metrics = {
        "truth": args.truth,
        "emulator": args.emulator or "toy",
        "mass_start": args.mass_start,
        "bands": BANDS,
        "corr_scale": args.corr_scale,
        "nu": args.nu,
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
    post = None
    if args.laplace:
        if not args.float64:
            print("WARNING: --laplace without --float64; the Hessian will be float32-noisy")
        post = laplace_posterior(fm, model, res.theta, args.n_laplace, args.hess_chunk,
                                 jax.random.PRNGKey(args.seed + 100))  # fmt: skip
        pstd = prior_map_std(model, 400, jax.random.PRNGKey(args.seed + 200))
        pm, med, std, info = posterior_metrics(truth_maps, post["maps"], pstd, weight)
        metrics["laplace"] = {k: v for k, v in post.items() if k not in ("theta", "maps", "chol")}
        metrics["laplace"]["per_field"] = pm
        mass_tot = np.log10(np.sum(10.0 ** post["maps"][:, 0], axis=(1, 2)))
        metrics["laplace"]["total_log_mass_16_50_84"] = [
            float(x) for x in np.percentile(mass_tot, [16, 50, 84])
        ]
        plot_laplace(truth_maps, med, std, info, weight, outdir / "figures" / "laplace.png")
        np.savez(outdir / "laplace.npz", median=med, std=std, information=info,
                 theta=np.asarray(post["theta"]))  # fmt: skip
        if args.nuts:
            nuts = nuts_posterior(fm, model, post["theta"], post["chol"],
                                  jax.random.PRNGKey(args.seed + 300), args.nuts_warmup,
                                  args.nuts_samples, args.nuts_chains)  # fmt: skip
            npm, nmed, nstd, ninfo = posterior_metrics(truth_maps, nuts["maps"], pstd, weight)
            metrics["nuts"] = {k: v for k, v in nuts.items() if k != "maps"}
            metrics["nuts"]["per_field"] = npm
            nmass = np.log10(np.sum(10.0 ** nuts["maps"][:, 0], axis=(1, 2)))
            metrics["nuts"]["total_log_mass_16_50_84"] = [
                float(x) for x in np.percentile(nmass, [16, 50, 84])
            ]
            plot_laplace(truth_maps, nmed, nstd, ninfo, weight, outdir / "figures" / "nuts.png")
            np.savez(outdir / "nuts.npz", median=nmed, std=nstd, information=ninfo)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(outdir / "maps.npz", truth=truth_maps, start=start_maps, map=fit_maps,
             model_image=model_img, data=np.asarray(obs.flux), sigma=sigma,
             theta_map=np.asarray(res.theta), fields=np.array(FIELDS))  # fmt: skip
    plot_maps(truth_maps, start_maps, fit_maps, weight, outdir / "figures" / "maps.png")
    plot_images(obs, model_img, sigma, outdir / "figures" / "images.png")

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
        print(f"Laplace: Newton-CG {lp['newton_s']:.0f} s, Hessian {lp['hessian_s']:.0f} s; "
              f"eigenvalues [{lp['eig_min']:.3g}, {lp['eig_max']:.3g}], "
              f"{lp['n_negative']} neg.")  # fmt: skip
        h0, h1 = lp["newton_history"][0], lp["newton_history"][-1]
        print(f"  Newton-CG {len(lp['newton_history']) - 1} steps: -log_post "
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
    if "nuts" in metrics:
        nm = metrics["nuts"]
        print(f"NUTS: {nm['runtime_s']:.0f} s; max R-hat {nm['max_rhat']:.3f} "
              f"({100 * nm['frac_rhat_gt_1.01']:.1f}% > 1.01); ESS min {nm['min_ess']:.0f}, "
              f"median {nm['median_ess']:.0f}")  # fmt: skip
        for name in FIELDS:
            q = nm["per_field"][name]
            print(f"  {name:18s} cov68 {q['coverage68']:.2f}  cov95 {q['coverage95']:.2f}  "
                  f"z_rms {q['z_rms']:.2f}  post std {q['median_post_std']:.3f}")  # fmt: skip
        m16, m50, m84 = nm["total_log_mass_16_50_84"]
        print(f"  total log M* {m50:.3f} +{m84 - m50:.3f} -{m50 - m16:.3f}")
    print(f"outputs in {outdir}")


if __name__ == "__main__":
    main()
