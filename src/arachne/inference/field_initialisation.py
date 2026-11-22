"""Data-driven starts for correlated-field fits with a fixed posterior target."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from arachne.data.psf import PSFModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.initialisation import image_moments
from arachne.likelihood.gaussian import GaussianLikelihood
from arachne.spatial.correlated_field import CorrelatedFieldMap
from arachne.spatial.profiles import SersicProfile, log_render_on_grid


def smooth_mass_reference(shape: tuple[int, int], total_log_mass: float) -> np.ndarray:
    """A circular exponential reference centred on the cutout, with Re = min(shape) / 8."""
    yy, xx = np.indices(shape, dtype=float)
    radius = np.hypot(yy - (shape[0] - 1) / 2, xx - (shape[1] - 1) / 2)
    mass = np.exp(-1.67835 * radius / (min(shape) / 8))
    return (total_log_mass + np.log10(mass / mass.sum())).astype(np.float32)


def project_mass_start(
    model: CorrelatedFieldMap, theta: jax.Array, log_mass: np.ndarray | jax.Array
) -> jax.Array:
    """Seed a log-mass level and Fourier coefficients, preserving the target total mass."""
    k = model.field_names.index(model.mass_param)
    target = jnp.asarray(log_mass, dtype=theta.dtype)
    current = model.field_maps(theta)[k]
    theta = theta.at[k].add(jnp.mean(target - current))
    theta = model.project_to_field(theta, model.mass_param, target, ridge=0.1)
    fitted = model.field_maps(theta)[k]
    ln10 = math.log(10)
    delta = jax.scipy.special.logsumexp(ln10 * target) - jax.scipy.special.logsumexp(ln10 * fitted)
    return theta.at[k].add(delta / ln10)


def _minimise(loss, start, bounds, maxiter):
    """Keep the incoming seed if a line search produces a worse or non-finite point."""
    value_grad = jax.jit(jax.value_and_grad(loss))

    def evaluate(x):
        value, grad = value_grad(jnp.asarray(x, dtype=start.dtype))
        return float(value), np.asarray(grad, dtype=float)

    start_loss, _ = evaluate(start)
    result = minimize(
        evaluate,
        np.asarray(start, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-9 if start.dtype == jnp.float64 else 1e-6},
    )
    if np.isfinite(result.fun) and result.fun < start_loss:
        return jnp.asarray(result.x, dtype=start.dtype), int(result.nit)
    return start, 0


def _aperture_likelihood(obs, regions, model_error_frac, min_snr=None, eligible=None):
    """Sum valid observed and predicted flux in the same apertures for every band."""
    windows = np.asarray(obs.mask, dtype=bool)[:, None] & np.asarray(regions)[None]
    flux = np.sum(np.where(windows, np.asarray(obs.flux)[:, None], 0.0), axis=(-2, -1))
    var = np.sum(np.where(windows, np.asarray(obs.variance)[:, None], 0.0), axis=(-2, -1))
    use = (var > 0).all(axis=1)
    if min_snr is not None:
        use &= (flux >= min_snr * np.sqrt(var)).all(axis=1)
    if eligible is not None:
        use &= eligible
    likelihood = GaussianLikelihood.from_arrays(
        flux[:, None],
        np.where(var > 0, var, 1.0)[:, None],
        np.broadcast_to(use[:, None, None], (len(use), 1, var.shape[1])),
        model_error_frac,
    )
    return likelihood, jnp.asarray(windows), use


def _aperture_flux(image, windows):
    return jnp.sum(image[:, None] * windows, axis=(-2, -1))[:, None]


def _regional_start(fm, theta, obs, pixel_loss, aperture, yy, xx, shape, psf, fields):
    """Use two resolved apertures to seed one smooth radial contrast per SPS field."""
    if not fields:
        return theta, [], 0, 0.0
    sy, sx = np.exp(shape[2:4])
    rho = float(np.tanh(shape[4]))
    y, x = (np.asarray(yy) - shape[0]) / sy, (np.asarray(xx) - shape[1]) / sx
    r2 = (y**2 + x**2 - 2 * rho * y * x) / (1 - rho**2)
    minor = np.sqrt(np.linalg.eigvalsh([[sy**2, rho * sy * sx], [rho * sy * sx, sx**2]])[0])

    kernels = np.maximum(np.asarray(psf.kernels), 0.0)
    kh, kw = kernels.shape[-2:]
    ky, kx = np.indices((kh, kw))
    radii = np.hypot(ky - kh // 2, kx - kw // 2).ravel()
    order = np.argsort(radii)
    encircled = np.cumsum(kernels.reshape(len(kernels), -1)[:, order], axis=1)
    psf_r50 = radii[order][np.argmax(encircled >= 0.5 * encircled[:, -1:], axis=1)]
    resolved = minor >= 2 * np.maximum(psf_r50, 1.0)
    regions = np.stack([aperture & (r2 <= 1), aperture & (r2 > 1)])
    likelihood, windows, use = _aperture_likelihood(
        obs, regions, pixel_loss.model_error_frac, min_snr=10.0, eligible=resolved
    )
    if use.sum() < 3:
        return theta, [], 0, 0.0

    model = fm.spatial_model
    maps = model.field_maps(theta)
    contrast = jnp.exp(-0.5 * jnp.asarray(r2))
    contrast -= contrast.mean()
    mass_level = jnp.zeros_like(theta).at[model.field_names.index(model.mass_param)].set(1)
    directions = [mass_level]
    for field in [model.mass_param, *fields]:
        current = maps[model.field_names.index(field)]
        directions.append(
            model.project_to_field(theta, field, current + contrast)
            - model.project_to_field(theta, field, current)
        )
    directions = jnp.stack(directions)

    def predicted(theta):
        return fm.convolver(model.model_image(theta, fm.emulator, model.image_shape))

    def loss(weights):
        candidate = theta + weights @ directions
        return -likelihood(_aperture_flux(predicted(candidate), windows)) - model.log_prior(
            candidate
        )

    n = len(directions)
    weights, steps = _minimise(loss, jnp.zeros(n, theta.dtype), [(-2, 2)] * n, 30)
    delta = weights @ directions

    @jax.jit
    def score(candidate):
        return pixel_loss(predicted(candidate)) + model.log_prior(candidate)

    # Keep aperture adjustments only to the extent they help the native-pixel target.
    scale = 0.0
    if steps:
        fractions = (0.0, 0.25, 0.5, 1.0)
        scores = np.asarray([float(score(theta + f * delta)) for f in fractions])
        scale = fractions[int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))]
    bands = [band for band, good in zip(obs.band_names, use, strict=True) if good]
    return theta + scale * delta, bands, steps, scale


def coarse_field_start(
    fm: ForwardModel,
    theta: jax.Array,
    psf: PSFModel,
    aperture: np.ndarray,
    sps_fields: Sequence[str],
) -> tuple[jax.Array, dict]:
    """Fit a PSF-forward Sérsic mass seed and seed SPS fields from coarse colours.

    First fit the aperture SED at fixed shape, then refine the intrinsic shape and SED
    against the native pixels. Two resolved, high-S/N apertures can seed radial SPS
    contrasts. The returned vector can initialise MAP or sampling on the original model.

    Args:
        fm: Common-grid correlated-field forward model without instrumental nuisances.
        theta: Initial theta with no Fourier fluctuations or nonzero field couplings.
        psf: PSF kernels used to build ``fm``, in observation band order.
        aperture: Boolean target footprint on the image grid, used for aperture colours.
        sps_fields: Fields whose levels and radial contrasts may be initialised.

    Returns:
        Initial theta and a dictionary of coarse-fit diagnostics.
    """
    model = fm.spatial_model
    if not isinstance(model, CorrelatedFieldMap) or fm.nuisance is not None:
        raise ValueError("coarse_field_start needs a CorrelatedFieldMap without nuisances")
    if tuple(psf.band_names) != tuple(fm.observation.band_names):
        raise ValueError("PSF bands must match the observation in order")
    mass_k = model.field_names.index(model.mass_param)
    if model.mass_param in sps_fields:
        raise ValueError("sps_fields should exclude the mass field")
    mass_column = model.emulator_param_names.index(model.mass_param)
    obs = fm.observation
    h, w = obs.image_shape
    aperture = np.asarray(aperture, dtype=bool)
    if aperture.shape != (h, w):
        raise ValueError("aperture must match the observation image shape")
    yy, xx = jnp.indices((h, w), dtype=theta.dtype)
    if not aperture.any():
        aperture = np.asarray(jnp.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) <= min(h, w) / 4)

    valid = (
        (obs.mask > 0) & jnp.isfinite(obs.flux) & jnp.isfinite(obs.variance) & (obs.variance > 0)
    )
    obs = replace(
        obs,
        flux=jnp.where(valid, obs.flux, 0.0),
        variance=jnp.where(valid, obs.variance, 1.0),
        mask=valid.astype(theta.dtype),
    )
    pixel_loss = fm.likelihood.with_observation(obs)
    frac = fm.likelihood.model_error_frac
    ap_loss, windows, usable = _aperture_likelihood(obs, aperture[None], frac)
    if not usable.any():
        return theta, {"status": "no_usable_aperture", "regional_bands": []}
    if usable.sum() < 3:
        sps_fields = ()
    field_indices = jnp.asarray([model.field_names.index(f) for f in sps_fields], dtype=int)

    cy, cx, sigma = image_moments(obs.flux, obs.variance, valid & aperture)
    size = np.clip(sigma, 0.6, max(h, w) / 2)
    profile = SersicProfile(n_bounds=(0.7, 4.0))
    shape0 = jnp.asarray([cy, cx, np.log(size), np.log(size), 0.0, 0.0], dtype=theta.dtype)
    mass_ref = float(model.field_maps(theta)[mass_k, h // 2, w // 2])

    def log_profile(shape):
        logp = log_render_on_grid(profile, shape, yy, xx, 1.0, oversample=3)
        return logp - jax.scipy.special.logsumexp(logp)

    def seed_theta(values):
        return theta.at[field_indices].set(values[1:])

    def image(values, shape):
        row = model.decode(seed_theta(values))[h // 2 * w + w // 2]
        # Mass linearity allows a single SPS prediction for the whole profile.
        row = row.at[mass_column].set(mass_ref)
        sed = fm.emulator.predict(row[None])[0] * 10.0 ** (values[0] - mass_ref)
        return fm.convolver(sed[:, None, None] * jnp.exp(log_profile(shape))[None])

    ap_flux = ap_loss.obs.flux
    ap_var = ap_loss.obs.variance
    ap_ref = _aperture_flux(image(jnp.r_[mass_ref, theta[field_indices]], shape0), windows)
    norm = jnp.sum(ap_ref**2 / ap_var)
    scale = float(jnp.sum(ap_flux * ap_ref / ap_var) / jnp.maximum(norm, 1e-30))
    if not np.isfinite(scale) or scale <= 0.0:
        return theta, {"status": "no_positive_aperture_flux", "regional_bands": []}
    logm = mass_ref + np.log10(scale)
    values0 = jnp.r_[jnp.asarray(logm, theta.dtype), theta[field_indices]]
    level_bounds = [(logm - 3, logm + 3), *[(float(x) - 3, float(x) + 3) for x in values0[1:]]]

    def aperture_loss(values):
        return -ap_loss(_aperture_flux(image(values, shape0), windows)) - model.log_prior(
            seed_theta(values)
        )

    values, aperture_steps = _minimise(aperture_loss, values0, level_bounds, 40)

    def shape_loss(x):
        return -pixel_loss(image(x[6:], x[:6])) - model.log_prior(seed_theta(x[6:]))

    size_bound = (math.log(0.4), math.log(max(h, w)))
    shape_bounds = [
        (0, h - 1),
        (0, w - 1),
        size_bound,
        size_bound,
        (-1.2, 1.2),
        (math.log(0.7), math.log(4.0)),
    ]
    fit, profile_steps = _minimise(
        shape_loss, jnp.r_[shape0, values], shape_bounds + level_bounds, 80
    )
    shape, values = fit[:6], fit[6:]
    profile_map = jnp.logaddexp(log_profile(shape), math.log(1e-4 / (h * w)))
    theta = project_mass_start(model, seed_theta(values), values[0] + profile_map / math.log(10))
    theta, bands, regional_steps, regional_scale = _regional_start(
        fm, theta, obs, pixel_loss, aperture, yy, xx, np.asarray(shape), psf, sps_fields
    )
    return theta, {
        "status": "fitted",
        "aperture_steps": aperture_steps,
        "profile_steps": profile_steps,
        "profile_log_mass": float(values[0]),
        "profile_shape_px": np.asarray(shape).tolist(),
        "sps_fields_seeded": list(sps_fields),
        "regional_bands": bands,
        "regional_steps": regional_steps,
        "regional_scale": regional_scale,
    }
