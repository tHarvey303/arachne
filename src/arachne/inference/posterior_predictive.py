"""Posterior-predictive images, per-component images and residual diagnostics.

Given posterior samples from :class:`~arachne.inference.nuts_sampler.NUTSSampler`
or :class:`~arachne.inference.nss_sampler.NSSSampler` (always **full** theta
vectors of length ``forward_model.n_params``), these helpers push the samples
back through the forward model so the fit can be checked in data space:

- :func:`model_image_samples` — PSF-convolved model images (with sky and
  sub-pixel shifts when a nuisance model is attached), one per sample;
- :func:`component_image_samples` — the *unconvolved* per-component images of
  an :class:`~arachne.spatial.additive.AdditiveComponentModel`, so a
  bulge/disk split can be inspected component by component;
- :func:`residual_summary` — the median model image, its chi map, and the
  reduced chi-squared (overall and per band);
- :func:`predictive_bands` — 16th/50th/84th percentile envelopes.

Everything is evaluated with ``jax.vmap`` over chunks of ``chunk`` samples
(default 16) so memory stays bounded even for 100+ samples of a large cube.

Multi-resolution models
-----------------------
Every function here also accepts a
:class:`~arachne.forward_model.multires.MultiResolutionForwardModel`, detected
by the presence of its ``model_images`` method.  Because the bands then have
different shapes, anything that was a stacked ``(..., N_bands, H, W)`` array
comes back as a **list of per-band arrays** (``model_image_samples`` ->
``[(n, H_b, W_b), ...]``, ``component_image_samples`` ->
``[(n, K, H_b, W_b), ...]``, ``residual_summary``'s ``median_model`` and
``chi`` -> lists of ``(H_b, W_b)``).  Every scalar key (``chi2``, ``n_data``,
``dof``, ``chi2_red``, ``chi2_red_per_band``, ``frac_chi_gt_3``) keeps exactly
the same meaning, summed over bands.  Single-grid outputs are unchanged.
"""

from __future__ import annotations

from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np

from arachne.forward_model.multires import MultiResolutionForwardModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

#: Either flavour of forward model.
AnyForwardModel = Union[ForwardModel, MultiResolutionForwardModel]

__all__ = [
    "model_image_samples",
    "component_image_samples",
    "residual_summary",
    "predictive_bands",
    "chi2_reduced",
    "chi2_reduced_samples",
]

_DEFAULT_CHUNK = 16


def is_multiresolution(forward_model: AnyForwardModel) -> bool:
    """Whether ``forward_model`` fits each band on its own pixel grid.

    Duck-typed on the presence of ``model_images`` (the ragged, per-band
    counterpart of ``ForwardModel._model_image``), so any future
    multi-resolution pipeline is picked up automatically.

    Args:
        forward_model: Assembled forward model.

    Returns:
        True for a MultiResolutionForwardModel, False for a single-grid one.
    """
    return hasattr(forward_model, "model_images")


def n_model_params(forward_model: AnyForwardModel) -> int:
    """Number of free parameters of the model (spatial + nuisance).

    Args:
        forward_model: Assembled ForwardModel.

    Returns:
        ``forward_model.n_params`` when available, else the spatial model's.
    """
    return int(getattr(forward_model, "n_params", forward_model.spatial_model.n_params))


def _thin(samples: jnp.ndarray, n_max: int, rng_key: jnp.ndarray | None) -> jnp.ndarray:
    """Reduce ``samples`` to at most ``n_max`` rows.

    Args:
        samples: (n_samples, n_params) posterior draws.
        n_max: Maximum number of draws to keep.
        rng_key: If given, draw ``n_max`` rows without replacement; otherwise
            thin deterministically with an evenly spaced stride (which keeps
            the draws spread over the whole chain).

    Returns:
        Array of shape (min(n_samples, n_max), n_params).
    """
    samples = jnp.atleast_2d(jnp.asarray(samples))
    n = samples.shape[0]
    if n_max is None or n <= n_max:
        return samples
    if rng_key is None:
        idx = jnp.linspace(0, n - 1, n_max).round().astype(jnp.int32)
    else:
        idx = jax.random.choice(rng_key, n, shape=(n_max,), replace=False)
    return samples[idx]


def _chunked_map(fn: Callable, xs: jnp.ndarray, chunk: int = _DEFAULT_CHUNK) -> jnp.ndarray:
    """Apply ``jax.vmap(fn)`` to ``xs`` in fixed-size chunks.

    The final chunk is padded up to ``chunk`` rows and sliced afterwards so
    that only one XLA program is compiled regardless of ``len(xs)``.

    Args:
        fn: Function mapping one row of ``xs`` to an array.
        xs: Array whose leading axis is mapped over.
        chunk: Rows evaluated per vmapped call.

    Returns:
        Stacked results with leading axis ``xs.shape[0]``.  ``fn`` may return a
        pytree (e.g. the ragged per-band list of a multi-resolution model), in
        which case every leaf is stacked.
    """
    n = xs.shape[0]
    if n == 0:
        raise ValueError("no samples to evaluate")
    chunk = max(1, min(int(chunk), n))
    batched = jax.jit(jax.vmap(fn))
    outs = []
    for start in range(0, n, chunk):
        block = xs[start : start + chunk]
        take = block.shape[0]
        if take < chunk:  # pad so the compiled shape never changes
            block = jnp.concatenate([block, jnp.tile(block[-1:], (chunk - take, 1))], axis=0)
        outs.append(jax.tree_util.tree_map(lambda a, t=take: a[:t], batched(block)))
    if len(outs) == 1:
        return outs[0]
    return jax.tree_util.tree_map(lambda *parts: jnp.concatenate(parts, axis=0), *outs)


def model_image_samples(
    forward_model: ForwardModel,
    samples: jnp.ndarray,
    n_max: int = 100,
    rng_key: jnp.ndarray | None = None,
    chunk: int = _DEFAULT_CHUNK,
) -> jnp.ndarray:
    """Posterior-predictive model images, one per (thinned) sample.

    Uses ``ForwardModel._model_image`` (or ``model_images`` for a
    multi-resolution model), so the returned images are exactly what the
    likelihood compares against: PSF-convolved, with the nuisance sub-pixel
    shifts and per-band sky applied when a nuisance model is attached.

    Args:
        forward_model: Assembled ForwardModel or MultiResolutionForwardModel.
        samples: Full theta draws of shape (n_samples, n_params).
        n_max: Maximum number of draws to evaluate.
        rng_key: Optional PRNG key; random thinning instead of a fixed stride.
        chunk: Samples evaluated per vmapped call.

    Returns:
        Array of shape (n, N_bands, H, W) in nJy, with
        ``n = min(n_samples, n_max)`` — or, for a multi-resolution model, a
        list of ``n_bands`` arrays of shape (n, H_b, W_b).
    """
    theta = _thin(samples, n_max, rng_key)
    if is_multiresolution(forward_model):
        return _chunked_map(forward_model.model_images, theta, chunk)
    return _chunked_map(forward_model._model_image, theta, chunk)


def _component_images_fn(forward_model: ForwardModel) -> Callable:
    """Return ``theta_spatial -> (K, N_bands, H, W)`` unconvolved component images.

    Prefers ``AdditiveComponentModel.component_images``; falls back to the
    outer product of ``component_seds`` and ``profiles`` (which is the same
    quantity by construction: the model image is their sum over components).
    A multi-resolution model returns its own ``component_images_per_band``
    instead, i.e. a list of ``(K, H_b, W_b)`` arrays.

    Args:
        forward_model: Assembled forward model.

    Returns:
        A pure JAX callable.

    Raises:
        AttributeError: If the spatial model has no per-component decomposition.
    """
    if is_multiresolution(forward_model):
        return forward_model.component_images_per_band
    model = forward_model.spatial_model
    emulator = forward_model.emulator
    if hasattr(model, "component_images"):
        return lambda theta_spatial: model.component_images(theta_spatial, emulator)
    if hasattr(model, "component_seds") and hasattr(model, "profiles"):

        def fallback(theta_spatial: jnp.ndarray) -> jnp.ndarray:
            seds = model.component_seds(theta_spatial, emulator)  # (K, N_bands)
            profiles = model.profiles(theta_spatial)  # (K, H, W)
            return jnp.einsum("kb,khw->kbhw", seds, profiles)

        return fallback
    raise AttributeError(
        f"{type(model).__name__} has no component_images / component_seds+profiles; "
        "component_image_samples only applies to additive component models."
    )


def component_image_samples(
    forward_model: ForwardModel,
    samples: jnp.ndarray,
    n_max: int = 50,
    rng_key: jnp.ndarray | None = None,
    chunk: int = _DEFAULT_CHUNK,
) -> jnp.ndarray:
    """Unconvolved per-component model images, one set per (thinned) sample.

    The nuisance block is stripped with ``ForwardModel.split_theta`` (the
    components are a property of the spatial model alone), and no PSF is
    applied — summing over the component axis reproduces the *unconvolved*
    model image.

    Args:
        forward_model: Assembled ForwardModel with an additive spatial model.
        samples: Full theta draws of shape (n_samples, n_params).
        n_max: Maximum number of draws to evaluate (component cubes are K times
            larger than a model image, hence the smaller default).
        rng_key: Optional PRNG key; random thinning instead of a fixed stride.
        chunk: Samples evaluated per vmapped call.

    Returns:
        Array of shape (n, K, N_bands, H, W) in nJy — or, for a
        multi-resolution model, a list of ``n_bands`` arrays of shape
        (n, K, H_b, W_b).

    Raises:
        AttributeError: If the spatial model has no per-component decomposition.
    """
    theta = _thin(samples, n_max, rng_key)
    comp_fn = _component_images_fn(forward_model)
    split = getattr(forward_model, "split_theta", None)

    def one(theta_full: jnp.ndarray) -> jnp.ndarray:
        theta_spatial = split(theta_full)[0] if split is not None else theta_full
        return comp_fn(theta_spatial)

    return _chunked_map(one, theta, chunk)


def chi2_reduced(forward_model: ForwardModel, theta: jnp.ndarray) -> float:
    """Reduced chi-squared of a single theta against the observation.

    Photon-noise chi-squared: ``sum(mask * (flux - model)^2 / variance)``
    divided by ``N_data - n_params``, where ``N_data`` is the number of
    unmasked pixel-band entries.  The model-error floor and any fitted noise
    rescaling are deliberately *not* applied, so the number is comparable
    across models with different nuisance setups.

    Args:
        forward_model: Assembled ForwardModel or MultiResolutionForwardModel.
        theta: Full theta vector of shape (n_params,).

    Returns:
        Reduced chi-squared (``NaN`` if there are no degrees of freedom).
    """
    if is_multiresolution(forward_model):
        chi2_jax, n_data = forward_model.chi2(jnp.asarray(theta))
        chi2 = float(chi2_jax)
    else:
        model = forward_model._model_image(jnp.asarray(theta))
        chi2, n_data = _chi2_and_ndata(forward_model, model)
    dof = n_data - n_model_params(forward_model)
    return float(chi2 / dof) if dof > 0 else float("nan")


def chi2_reduced_samples(
    forward_model: ForwardModel,
    samples: jnp.ndarray,
    n_max: int = 64,
    rng_key: jnp.ndarray | None = None,
    chunk: int = _DEFAULT_CHUNK,
) -> np.ndarray:
    """Per-sample reduced chi-squared, vectorised over the (thinned) draws.

    Same definition as :func:`chi2_reduced` but evaluated with the chunked
    ``vmap`` of :func:`model_image_samples`, so a posterior's fit-quality
    distribution costs one compiled program instead of ``n`` Python calls.

    Args:
        forward_model: Assembled ForwardModel or MultiResolutionForwardModel.
        samples: Full theta draws of shape (n_samples, n_params).
        n_max: Maximum number of draws to evaluate.
        rng_key: Optional PRNG key; random thinning instead of a fixed stride.
        chunk: Samples evaluated per vmapped call.

    Returns:
        ``numpy`` array of shape ``(min(n_samples, n_max),)``.  All entries are
        ``NaN`` when the model has no degrees of freedom.
    """
    if is_multiresolution(forward_model):
        theta = _thin(samples, n_max, rng_key)
        chi2 = _chunked_map(forward_model._chi2_value, theta, chunk)
        dof = forward_model.n_data - n_model_params(forward_model)
        if dof <= 0:
            return np.full((int(chi2.shape[0]),), np.nan)
        return np.asarray(chi2 / dof)

    images = model_image_samples(forward_model, samples, n_max=n_max, rng_key=rng_key, chunk=chunk)
    obs = forward_model.observation
    mask = jnp.asarray(obs.mask)
    inv_var = mask / (jnp.asarray(obs.variance) + 1e-30)
    resid = jnp.asarray(obs.flux)[None, ...] - images
    chi2 = jnp.sum(inv_var[None, ...] * resid**2, axis=(1, 2, 3))
    dof = int(jnp.sum(mask > 0)) - n_model_params(forward_model)
    if dof <= 0:
        return np.full((int(images.shape[0]),), np.nan)
    return np.asarray(chi2 / dof)


def _chi2_and_ndata(forward_model: ForwardModel, model_image: jnp.ndarray) -> tuple[float, int]:
    """Total masked photon-noise chi-squared and the number of valid data points."""
    obs = forward_model.observation
    mask = jnp.asarray(obs.mask)
    resid = jnp.asarray(obs.flux) - model_image
    chi2 = jnp.sum(mask * resid**2 / (jnp.asarray(obs.variance) + 1e-30))
    n_data = int(jnp.sum(mask > 0))
    return float(chi2), n_data


def residual_summary(
    forward_model: ForwardModel,
    samples: jnp.ndarray,
    n_max: int = 100,
    rng_key: jnp.ndarray | None = None,
    chunk: int = _DEFAULT_CHUNK,
) -> dict:
    """Median posterior-predictive image and its residual diagnostics.

    Args:
        forward_model: Assembled ForwardModel or MultiResolutionForwardModel.
        samples: Full theta draws of shape (n_samples, n_params).
        n_max: Maximum number of draws used for the median image.
        rng_key: Optional PRNG key for random thinning.
        chunk: Samples evaluated per vmapped call.

    Returns:
        Dict with

        - ``median_model`` (N_bands, H, W): pixel-wise median model image
          (a list of ``n_bands`` (H_b, W_b) arrays for a multi-resolution model);
        - ``chi`` (N_bands, H, W): ``(flux - median_model) / sqrt(variance)``,
          zero where the mask is zero (likewise a per-band list);
        - ``chi2``: total masked chi-squared of the median model;
        - ``n_data``, ``n_params``, ``dof``;
        - ``chi2_red``: ``chi2 / dof`` (photon noise, ``dof = n_data - n_params``);
        - ``chi2_red_per_band`` (N_bands,): per-band chi-squared divided by the
          band's valid pixel count (the parameters are shared across bands, so
          no per-band parameter subtraction is made);
        - ``frac_chi_gt_3``: fraction of valid pixels with ``|chi| > 3``;
        - ``band_names``, ``n_samples_used``.
    """
    images = model_image_samples(forward_model, samples, n_max=n_max, rng_key=rng_key, chunk=chunk)
    if is_multiresolution(forward_model):
        return _multires_residual_summary(forward_model, images)
    median_model = jnp.median(images, axis=0)  # (N_bands, H, W)

    obs = forward_model.observation
    mask = jnp.asarray(obs.mask)
    sigma = jnp.sqrt(jnp.asarray(obs.variance) + 1e-30)
    chi = jnp.where(mask > 0, (jnp.asarray(obs.flux) - median_model) / sigma, 0.0)

    chi2, n_data = _chi2_and_ndata(forward_model, median_model)
    n_params = n_model_params(forward_model)
    dof = n_data - n_params
    valid = mask > 0
    n_valid_band = jnp.sum(valid, axis=(1, 2))
    chi2_band = jnp.sum(chi**2, axis=(1, 2))
    frac_gt3 = float(jnp.sum(jnp.abs(chi) > 3.0) / jnp.maximum(jnp.sum(valid), 1))

    return {
        "median_model": median_model,
        "chi": chi,
        "chi2": chi2,
        "n_data": n_data,
        "n_params": n_params,
        "dof": int(dof),
        "chi2_red": float(chi2 / dof) if dof > 0 else float("nan"),
        "chi2_red_per_band": np.asarray(chi2_band / jnp.maximum(n_valid_band, 1)),
        "frac_chi_gt_3": frac_gt3,
        "band_names": list(getattr(obs, "band_names", [])),
        "n_samples_used": int(images.shape[0]),
    }


def _multires_residual_summary(forward_model: MultiResolutionForwardModel, images: list) -> dict:
    """Residual summary for a multi-resolution model, with per-band lists.

    Args:
        forward_model: The multi-resolution forward model.
        images: Per-band posterior-predictive images, ``n_bands`` arrays of
            shape (n, H_b, W_b), as returned by :func:`model_image_samples`.

    Returns:
        The same dict as :func:`residual_summary`, with ``median_model`` and
        ``chi`` as lists of ``(H_b, W_b)`` arrays and every scalar summed over
        bands.
    """
    median_model = [jnp.median(im, axis=0) for im in images]
    chi: list[jnp.ndarray] = []
    chi2_band = []
    n_valid_band = []
    chi2 = 0.0
    n_data = 0
    n_gt3 = 0
    n_valid = 0
    for band, model in zip(forward_model.observation.bands, median_model, strict=True):
        mask = jnp.asarray(band.mask)
        sigma = jnp.sqrt(jnp.asarray(band.variance) + 1e-30)
        chi_b = jnp.where(mask > 0, (jnp.asarray(band.flux) - model) / sigma, 0.0)
        chi.append(chi_b)
        valid_b = int(jnp.sum(mask > 0))
        chi2_b = float(jnp.sum(mask * chi_b**2))
        chi2_band.append(chi2_b)
        n_valid_band.append(valid_b)
        chi2 += chi2_b
        n_data += valid_b
        n_gt3 += int(jnp.sum((jnp.abs(chi_b) > 3.0) & (mask > 0)))
        n_valid += valid_b

    n_params = n_model_params(forward_model)
    dof = n_data - n_params
    per_band = np.asarray(chi2_band) / np.maximum(np.asarray(n_valid_band), 1)
    return {
        "median_model": median_model,
        "chi": chi,
        "chi2": float(chi2),
        "n_data": int(n_data),
        "n_params": n_params,
        "dof": int(dof),
        "chi2_red": float(chi2 / dof) if dof > 0 else float("nan"),
        "chi2_red_per_band": per_band,
        "frac_chi_gt_3": float(n_gt3 / max(n_valid, 1)),
        "band_names": list(forward_model.band_names),
        "n_samples_used": int(images[0].shape[0]),
    }


def predictive_bands(images: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """16th / 50th / 84th percentile envelopes over the sample axis.

    Args:
        images: Array whose leading axis indexes posterior samples, e.g. the
            (n, N_bands, H, W) output of :func:`model_image_samples` or the
            (n, K, N_bands, H, W) output of :func:`component_image_samples`.

    Returns:
        Tuple ``(p16, p50, p84)``, each with the leading sample axis removed.

    Raises:
        ValueError: If ``images`` has no sample axis.
    """
    arr = jnp.asarray(images)
    if arr.ndim < 2:
        raise ValueError(f"images must have a leading sample axis; got shape {arr.shape}")
    pct = jnp.percentile(arr, jnp.array([16.0, 50.0, 84.0]), axis=0)
    return pct[0], pct[1], pct[2]
