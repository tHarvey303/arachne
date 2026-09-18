"""GPU-batched resolved fitting of many equal-shape galaxy cutouts.

Catalogue fitting in arachne is vmapped over galaxies; resolved fitting has so
far been one galaxy at a time.  When a sample of hundreds of cutouts shares the
same bands and the same ``(H, W)`` — the usual outcome of a fixed-size cutout
service such as DJA — every one of those fits is the *same* XLA program with
different data.  This module turns the sample into one batched program:

- :class:`BatchedForwardModel` — ``N`` single-grid
  :class:`~arachne.forward_model.pipeline.ForwardModel` objects with identical
  structure, evaluated under ``jax.vmap``;
- :func:`batched_blind_initial_theta` — moments + neutral SPS values per galaxy
  (numpy, a Python loop) followed by one **batched** linear mass solve;
- :func:`batched_find_map` / :func:`batched_multistart_map` — vmapped Adam with
  the mass solve repeated between rounds, all galaxies and all archetypes in one
  compiled program;
- :func:`batched_nuts` — vmapped window adaptation + ``lax.scan`` sampling over
  ``(galaxy, chain)``, with per-galaxy convergence diagnostics;
- :func:`fit_batch_nss` — a thin **sequential** helper, because nested slice
  sampling terminates on a per-problem Python ``while`` loop and cannot be
  batched.

How the batching works
----------------------
The galaxy-0 forward model is built with the ordinary
:meth:`ForwardModel.build` and kept as a *template* that carries all the static
structure (image shape, band names, padded FFT grid, spatial model, emulator,
nuisance model, model-error floor).  Everything that varies between galaxies —
``observation.flux/variance/mask``, ``convolver.psf_ffts`` and the fixed SPS
parameter values — is stacked into arrays with a leading galaxy axis and stored
in a single pytree (``BatchedForwardModel.data``).

:meth:`BatchedForwardModel.forward_model_from_pack` then rebuilds a genuine
``ForwardModel`` from one slice of that pytree, substituting the per-galaxy
arrays into shallow copies of the template's components.  Under ``jax.vmap``
those slices are tracers, so the rebuild happens once at trace time and the
batched log-posterior runs *literally the library's*
``ForwardModel.log_posterior`` — there is no second implementation of the
image → convolve → likelihood chain to drift out of step, and the batched
result equals the per-galaxy one to float32 round-off.

Per-galaxy fixed parameters
---------------------------
``AdditiveComponentModel.fixed_params`` is a static dict, so a spec-z that
differs per galaxy cannot simply be passed there.  Two workarounds were
considered and rejected: making the redshift a *shared free* parameter pinned by
a very tight Gaussian prior changes the parameter count and leaves a (small) free
direction that the sampler must still explore, and overwriting the shared raw in
``theta`` after every step is not a pure function of ``theta`` and would break
``jax.grad`` and every sampler.  Instead ``fixed_params_per_galaxy`` stacks the
model's ``fixed_params`` vector into a ``(N, n_fixed)`` array which becomes a
batch axis of the data pytree: the parameter count is identical for every galaxy,
the value is pinned exactly (a delta function, not a narrow Gaussian), and
``log_posterior`` stays a pure function of ``theta``.  The parameter must already
be declared in the spatial model's ``fixed_params`` (any placeholder value); see
:meth:`BatchedForwardModel.build`.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.nuisance import NuisanceModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.diagnostics import chain_movement, ess, split_rhat
from arachne.inference.initialisation import (
    _apply_archetype,
    blind_initial_theta,
    solve_component_masses,
)
from arachne.inference.nss_sampler import NSSResult, NSSSampler
from arachne.inference.nuts_sampler import NUTSResult, _jitter_inits
from arachne.psf.convolution import PSFConvolver
from arachne.spatial.additive import AdditiveComponentModel
from arachne.spatial.base import SpatialModel
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

__all__ = [
    "BatchedForwardModel",
    "BatchedMAPResult",
    "BatchedNUTSResult",
    "batched_blind_initial_theta",
    "batched_find_map",
    "batched_multistart_map",
    "batched_nuts",
    "fit_batch_nss",
]


# ---------------------------------------------------------------------------
# Batched forward model
# ---------------------------------------------------------------------------


def _stack_observations(
    observations: Sequence[ObservationCube] | Mapping[str, Any],
) -> tuple[dict[str, jnp.ndarray], ObservationCube]:
    """Stack a sample of equal-shape cutouts into ``(N, N_bands, H, W)`` arrays.

    Args:
        observations: Either a sequence of
            :class:`~arachne.data.observation.ObservationCube` objects that all
            share the same bands and image shape, or a mapping with the keys
            ``"flux"``, ``"variance"`` and optionally ``"mask"`` holding
            ``(N, N_bands, H, W)`` arrays plus optionally ``"band_names"`` and
            ``"pixel_scale"``.

    Returns:
        Tuple ``(arrays, template_cube)`` where ``arrays`` has the keys
        ``"flux"``, ``"variance"`` and ``"mask"`` with a leading galaxy axis and
        ``template_cube`` is galaxy 0 as a plain ``ObservationCube``.

    Raises:
        ValueError: If the sample is empty, the cutouts have inconsistent shapes
            or band names, or a mapping is missing ``flux``/``variance``.
    """
    if isinstance(observations, Mapping):
        missing = {"flux", "variance"} - set(observations)
        if missing:
            raise ValueError(f"stacked observations mapping is missing {sorted(missing)}")
        flux = jnp.asarray(observations["flux"], dtype=jnp.float32)
        variance = jnp.asarray(observations["variance"], dtype=jnp.float32)
        if flux.ndim != 4:
            raise ValueError(f"stacked flux must be (N, N_bands, H, W); got {flux.shape}")
        mask = observations.get("mask")
        mask = (
            jnp.ones_like(flux) if mask is None else jnp.asarray(mask, dtype=jnp.float32)
        ).reshape(flux.shape)
        n_bands = int(flux.shape[1])
        band_names = list(observations.get("band_names") or [f"band_{i}" for i in range(n_bands)])
        pixel_scale = float(observations.get("pixel_scale", 1.0))
        arrays = {"flux": flux, "variance": variance, "mask": mask}
        template = ObservationCube(
            flux=flux[0],
            variance=variance[0],
            mask=mask[0],
            band_names=band_names,
            pixel_scale=pixel_scale,
        )
        return arrays, template

    cubes = list(observations)
    if not cubes:
        raise ValueError("observations is empty; BatchedForwardModel needs at least one galaxy.")
    ref = cubes[0]
    for i, cube in enumerate(cubes[1:], start=1):
        if cube.flux.shape != ref.flux.shape:
            raise ValueError(
                f"galaxy {i} has image shape {cube.flux.shape}, expected {ref.flux.shape}; "
                "BatchedForwardModel requires cutouts of identical shape."
            )
        if list(cube.band_names) != list(ref.band_names):
            raise ValueError(
                f"galaxy {i} has bands {list(cube.band_names)}, expected {list(ref.band_names)}."
            )
    arrays = {
        "flux": jnp.stack([jnp.asarray(c.flux, dtype=jnp.float32) for c in cubes]),
        "variance": jnp.stack([jnp.asarray(c.variance, dtype=jnp.float32) for c in cubes]),
        "mask": jnp.stack([jnp.asarray(c.mask, dtype=jnp.float32) for c in cubes]),
    }
    return arrays, ref.to_jax()


class BatchedForwardModel:
    """``N`` structurally identical single-grid forward models under ``jax.vmap``.

    Every galaxy shares the bands, the image shape, the spatial model, the
    emulator, the nuisance model and the model-error floor; only the data, the
    PSFs and (optionally) the fixed SPS parameter values differ.  Those live in
    :attr:`data`, a pytree whose leaves carry a leading galaxy axis (except a
    shared PSF, which is stored once and broadcast).

    See the module docstring for how the batched evaluation reuses
    :class:`~arachne.forward_model.pipeline.ForwardModel` verbatim.

    Attributes:
        template: Galaxy 0's ``ForwardModel``; the source of all static structure.
        data: Dict of batched arrays (``flux``, ``variance``, ``mask``,
            ``psf_ffts`` and optionally ``fixed_vals``).
        in_axes: Matching dict of vmap axes (``0`` for batched leaves, ``None``
            for shared ones).
        n_galaxies: Number of galaxies ``N``.
        galaxy_ids: Labels used in reports and HDF5 group names.
    """

    def __init__(
        self,
        template: ForwardModel,
        data: dict[str, jnp.ndarray],
        in_axes: dict[str, int | None],
        n_galaxies: int,
        galaxy_ids: Sequence[str] | None = None,
    ) -> None:
        """Assemble a batched forward model from a template and stacked arrays.

        Most callers should use :meth:`build` instead.

        Args:
            template: Galaxy 0's forward model.
            data: Batched data pytree.
            in_axes: vmap axis per key of ``data``.
            n_galaxies: Number of galaxies.
            galaxy_ids: Optional labels; defaults to ``galaxy_0000 ...``.
        """
        self.template = template
        self.data = data
        self.in_axes = in_axes
        self.n_galaxies = int(n_galaxies)
        self.galaxy_ids = (
            [f"galaxy_{i:04d}" for i in range(self.n_galaxies)]
            if galaxy_ids is None
            else [str(g) for g in galaxy_ids]
        )
        if len(self.galaxy_ids) != self.n_galaxies:
            raise ValueError(
                f"galaxy_ids has {len(self.galaxy_ids)} entries but there are "
                f"{self.n_galaxies} galaxies."
            )
        self._cache: dict[Any, Any] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        observations: Sequence[ObservationCube] | Mapping[str, Any],
        psf_models: PSFModel | Sequence[PSFModel],
        spatial_model: SpatialModel,
        emulator: SPSEmulator,
        model_error_frac: float | np.ndarray | jnp.ndarray = 0.0,
        nuisance: NuisanceModel | None = None,
        pad_psf: bool = True,
        fixed_params_per_galaxy: Mapping[str, Any] | None = None,
        galaxy_ids: Sequence[str] | None = None,
    ) -> "BatchedForwardModel":
        """Assemble a batched forward model from a sample of equal-shape cutouts.

        Args:
            observations: Sequence of ``ObservationCube`` (same bands, same
                ``(H, W)``) or a mapping of stacked ``(N, N_bands, H, W)``
                arrays; see :func:`_stack_observations`.
            psf_models: One ``PSFModel`` shared by every galaxy, or a sequence of
                ``N`` per-galaxy models.  Per-galaxy kernels must produce the
                same padded FFT grid (equal kernel sizes is the easy way).
            spatial_model: Spatial parameterisation, shared by every galaxy.
            emulator: SPS emulator, shared by every galaxy.
            model_error_frac: Fractional model-error floor, scalar or
                ``(N_bands,)``; shared by every galaxy.
            nuisance: Optional ``NuisanceModel`` appended to every galaxy's theta.
            pad_psf: Use the zero-padded (linear) PSF convolution.
            fixed_params_per_galaxy: ``{param_name: (N,) array}`` overriding a
                *fixed* SPS parameter per galaxy — typically ``{"redshift":
                z_spec}``.  The name must already appear in
                ``spatial_model.fixed_params`` (the value there is only a
                placeholder).  See the module docstring for why this, rather
                than a tight prior on a shared free parameter, is the mechanism.
            galaxy_ids: Optional labels for reports and HDF5 groups.

        Returns:
            Configured :class:`BatchedForwardModel`.

        Raises:
            ValueError: On inconsistent cutout shapes, a wrong number of PSF
                models, mismatched padded FFT grids, or a
                ``fixed_params_per_galaxy`` entry that is not a fixed parameter
                of the spatial model or has the wrong length.
            TypeError: If ``fixed_params_per_galaxy`` is used with a spatial
                model that is not an :class:`AdditiveComponentModel`.
        """
        arrays, template_cube = _stack_observations(observations)
        n_galaxies = int(arrays["flux"].shape[0])
        H, W = template_cube.image_shape

        data: dict[str, jnp.ndarray] = dict(arrays)
        in_axes: dict[str, int | None] = {"flux": 0, "variance": 0, "mask": 0}

        # --- PSFs -----------------------------------------------------
        if isinstance(psf_models, PSFModel):
            template_psf = psf_models
            convolver = PSFConvolver(template_psf, image_shape=(H, W), pad=pad_psf)
            data["psf_ffts"] = convolver.psf_ffts
            in_axes["psf_ffts"] = None
        else:
            psf_list = list(psf_models)
            if len(psf_list) != n_galaxies:
                raise ValueError(
                    f"psf_models has {len(psf_list)} entries but there are {n_galaxies} galaxies."
                )
            convolvers = [PSFConvolver(p, image_shape=(H, W), pad=pad_psf) for p in psf_list]
            convolver = convolvers[0]
            for i, conv in enumerate(convolvers[1:], start=1):
                if conv.padded_shape != convolver.padded_shape:
                    raise ValueError(
                        f"galaxy {i} PSF gives padded FFT grid {conv.padded_shape}, expected "
                        f"{convolver.padded_shape}; per-galaxy kernels must be the same size."
                    )
            data["psf_ffts"] = jnp.stack([c.psf_ffts for c in convolvers])
            in_axes["psf_ffts"] = 0
            template_psf = psf_list[0]

        # --- Template forward model ----------------------------------
        template = ForwardModel.build(
            obs=template_cube,
            psf_model=template_psf,
            spatial_model=spatial_model,
            emulator=emulator,
            model_error_frac=model_error_frac,
            nuisance=nuisance,
            pad_psf=pad_psf,
        )

        # --- Per-galaxy fixed parameters ------------------------------
        if fixed_params_per_galaxy:
            if not isinstance(spatial_model, AdditiveComponentModel):
                raise TypeError(
                    "fixed_params_per_galaxy requires an AdditiveComponentModel spatial model, "
                    f"got {type(spatial_model).__name__}"
                )
            fixed_names = spatial_model.fixed_param_names
            unknown = set(fixed_params_per_galaxy) - set(fixed_names)
            if unknown:
                raise ValueError(
                    f"fixed_params_per_galaxy names {sorted(unknown)} are not fixed parameters of "
                    f"the spatial model (fixed: {fixed_names}).  Declare them in "
                    "AdditiveComponentModel(fixed_params=...) with a placeholder value first."
                )
            base = jnp.asarray(spatial_model.fixed_values, dtype=jnp.float32)
            fixed_vals = jnp.tile(base[None, :], (n_galaxies, 1))
            for name, values in fixed_params_per_galaxy.items():
                col = fixed_names.index(name)
                arr = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
                if arr.shape[0] != n_galaxies:
                    raise ValueError(
                        f"fixed_params_per_galaxy[{name!r}] has {arr.shape[0]} entries but there "
                        f"are {n_galaxies} galaxies."
                    )
                fixed_vals = fixed_vals.at[:, col].set(arr)
            data["fixed_vals"] = fixed_vals
            in_axes["fixed_vals"] = 0

        logger.info(
            f"BatchedForwardModel built: {n_galaxies} galaxies, image {H}×{W}, "
            f"{template_cube.n_bands} bands, {template.n_params} parameters each, "
            f"PSF {'per galaxy' if in_axes['psf_ffts'] == 0 else 'shared'}, "
            f"{len(fixed_params_per_galaxy or {})} per-galaxy fixed parameter(s)."
        )
        return cls(template, data, in_axes, n_galaxies, galaxy_ids=galaxy_ids)

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Free parameters per galaxy (spatial + nuisance)."""
        return self.template.n_params

    @property
    def n_spatial_params(self) -> int:
        """Free parameters of the spatial model alone."""
        return self.template.spatial_model.n_params

    @property
    def spatial_model(self) -> SpatialModel:
        """The shared spatial model."""
        return self.template.spatial_model

    @property
    def emulator(self) -> SPSEmulator:
        """The shared SPS emulator."""
        return self.template.emulator

    @property
    def nuisance(self) -> NuisanceModel | None:
        """The shared nuisance model, or ``None``."""
        return self.template.nuisance

    @property
    def band_names(self) -> list[str]:
        """Photometric band names."""
        return list(self.template.observation.band_names)

    @property
    def n_bands(self) -> int:
        """Number of photometric bands."""
        return int(self.template.observation.n_bands)

    @property
    def image_shape(self) -> tuple[int, int]:
        """Image shape ``(H, W)``, identical for every galaxy."""
        return self.template.observation.image_shape

    def __repr__(self) -> str:
        """One-line description."""
        H, W = self.image_shape
        return (
            f"BatchedForwardModel(n_galaxies={self.n_galaxies}, n_bands={self.n_bands}, "
            f"image_shape=({H}, {W}), n_params={self.n_params})"
        )

    # ------------------------------------------------------------------
    # Rebuilding a ForwardModel from one slice of the data pytree
    # ------------------------------------------------------------------

    def forward_model_from_pack(self, pack: Mapping[str, jnp.ndarray]) -> ForwardModel:
        """Rebuild a ``ForwardModel`` from one galaxy's arrays (possibly tracers).

        Shallow copies of the template's observation, convolver, likelihood and
        spatial model take the supplied arrays; everything else (image shape,
        padded FFT grid, model-error floor, emulator, nuisance model) is shared
        with the template.  Called once per ``jax.vmap`` trace, so the batched
        program runs the library's own ``ForwardModel`` code.

        Args:
            pack: Mapping with ``"flux"``, ``"variance"``, ``"mask"``,
                ``"psf_ffts"`` and optionally ``"fixed_vals"``.

        Returns:
            A ``ForwardModel`` for that galaxy.
        """
        template = self.template
        obs = dataclasses.replace(
            template.observation,
            flux=pack["flux"],
            variance=pack["variance"],
            mask=pack["mask"],
        )
        convolver = template.convolver.with_psf_ffts(pack["psf_ffts"])
        likelihood = template.likelihood.with_observation(obs)
        spatial_model = template.spatial_model
        if "fixed_vals" in pack:
            spatial_model = spatial_model.with_fixed_values(pack["fixed_vals"])
        return ForwardModel(
            observation=obs,
            spatial_model=spatial_model,
            emulator=template.emulator,
            convolver=convolver,
            likelihood=likelihood,
            nuisance=template.nuisance,
        )

    def pack(self, i: int) -> dict[str, jnp.ndarray]:
        """Concrete data slice for galaxy ``i``.

        Args:
            i: Galaxy index.

        Returns:
            Mapping suitable for :meth:`forward_model_from_pack`.
        """
        index = int(i)
        if not 0 <= index < self.n_galaxies:
            raise IndexError(f"galaxy index {i} out of range for {self.n_galaxies} galaxies")
        return {
            key: (value[index] if self.in_axes[key] == 0 else value)
            for key, value in self.data.items()
        }

    def forward_model(self, i: int) -> ForwardModel:
        """Ordinary single-galaxy ``ForwardModel`` view of galaxy ``i``.

        Use it for anything the batched path does not cover — nested sampling,
        posterior-predictive checks, plotting.

        Args:
            i: Galaxy index.

        Returns:
            ``ForwardModel`` sharing the template's structure with galaxy ``i``'s
            data, PSF and fixed parameters.
        """
        return self.forward_model_from_pack(self.pack(i))

    def observation(self, i: int) -> ObservationCube:
        """Galaxy ``i``'s observation cube.

        Args:
            i: Galaxy index.

        Returns:
            ``ObservationCube`` with JAX arrays.
        """
        return self.forward_model(i).observation

    # ------------------------------------------------------------------
    # Batched evaluation
    # ------------------------------------------------------------------

    def _batched(self, name: str):
        """Return (and cache) a jitted vmapped ``ForwardModel`` method by name."""
        key = ("method", name)
        if key not in self._cache:

            def call(pack, theta):
                return getattr(self.forward_model_from_pack(pack), name)(theta)

            self._cache[key] = jax.jit(jax.vmap(call, in_axes=(self.in_axes, 0)))
        return self._cache[key]

    def log_posterior(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """Log-posterior of every galaxy.

        Args:
            thetas: Parameter matrix of shape ``(N, n_params)``.

        Returns:
            Array of shape ``(N,)``.
        """
        return self._batched("log_posterior")(self.data, jnp.asarray(thetas, dtype=jnp.float32))

    def log_likelihood(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """Log-likelihood of every galaxy.

        Args:
            thetas: Parameter matrix of shape ``(N, n_params)``.

        Returns:
            Array of shape ``(N,)``.
        """
        return self._batched("log_likelihood")(self.data, jnp.asarray(thetas, dtype=jnp.float32))

    def log_prior(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """Log-prior of every galaxy.

        Args:
            thetas: Parameter matrix of shape ``(N, n_params)``.

        Returns:
            Array of shape ``(N,)``.
        """
        return self._batched("log_prior")(self.data, jnp.asarray(thetas, dtype=jnp.float32))

    def log_posterior_single(self, i: int, theta: jnp.ndarray) -> jnp.ndarray:
        """Log-posterior of one galaxy (convenience, not batched).

        Args:
            i: Galaxy index.
            theta: Parameter vector of shape ``(n_params,)``.

        Returns:
            Scalar log-posterior.
        """
        return self.forward_model(i).log_posterior(jnp.asarray(theta, dtype=jnp.float32))

    def model_images(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """PSF-convolved model images for every galaxy.

        Args:
            thetas: Parameter matrix of shape ``(N, n_params)``.

        Returns:
            Array of shape ``(N, N_bands, H, W)`` in nJy.
        """
        return self._batched("_model_image")(self.data, jnp.asarray(thetas, dtype=jnp.float32))

    # ------------------------------------------------------------------
    # Parameter-vector plumbing
    # ------------------------------------------------------------------

    def split_theta(self, thetas: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Split batched thetas into spatial and nuisance blocks.

        Args:
            thetas: Array of shape ``(N, n_params)``.

        Returns:
            Tuple ``(thetas_spatial, thetas_nuisance)`` of shapes
            ``(N, n_spatial)`` and ``(N, n_nuisance)``.
        """
        thetas = jnp.atleast_2d(jnp.asarray(thetas))
        n_spatial = self.n_spatial_params
        return thetas[:, :n_spatial], thetas[:, n_spatial:]

    def initial_theta_from_spatial(self, thetas_spatial: jnp.ndarray) -> jnp.ndarray:
        """Extend spatial-only thetas with the nuisance block at its prior mean.

        Args:
            thetas_spatial: Array of shape ``(N, n_spatial)``, or a single
                ``(n_spatial,)`` vector which is broadcast to all ``N`` galaxies.

        Returns:
            Array of shape ``(N, n_params)``, float32.
        """
        thetas = jnp.asarray(thetas_spatial, dtype=jnp.float32)
        if thetas.ndim == 1:
            thetas = jnp.tile(thetas[None, :], (self.n_galaxies, 1))
        if self.nuisance is None:
            return thetas
        tail = jnp.tile(self.nuisance.initial_theta()[None, :], (thetas.shape[0], 1))
        return jnp.concatenate([thetas, tail], axis=1).astype(jnp.float32)

    def sample_prior(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw ``n`` prior samples for each galaxy.

        The prior is the same distribution for every galaxy (fixed parameters do
        not enter it), but each galaxy gets its own independent draws so the
        result can seed a batched sampler directly.

        Args:
            key: PRNG key.
            n: Number of draws per galaxy.

        Returns:
            Array of shape ``(N, n, n_params)``.
        """
        keys = jax.random.split(key, self.n_galaxies)
        return jax.vmap(lambda k: self.template.sample_prior(k, int(n)))(keys)

    def order_components_by_size(self, thetas: jnp.ndarray) -> jnp.ndarray:
        """Permute components compact-first for every galaxy (label symmetry).

        Args:
            thetas: Array of shape ``(N, n_params)``; any nuisance tail is passed
                through untouched.

        Returns:
            Array of the same shape.
        """
        if "order" not in self._cache:
            self._cache["order"] = jax.jit(
                jax.vmap(self.template.spatial_model.order_components_by_size)
            )
        return self._cache["order"](jnp.asarray(thetas, dtype=jnp.float32))


# ---------------------------------------------------------------------------
# Batched blind initialisation
# ---------------------------------------------------------------------------


def batched_blind_initial_theta(
    bfm: BatchedForwardModel,
    solve_masses: bool = True,
    **blind_kwargs: Any,
) -> jnp.ndarray:
    """Truth-free starting thetas for every galaxy in the batch.

    :func:`~arachne.inference.initialisation.blind_initial_theta` is numpy code
    driven by image moments, so it runs in a Python loop over galaxies (it costs
    microseconds per cutout).  The expensive part — the weighted linear mass
    solve — is then done **once, batched**, as a single compiled program.

    Args:
        bfm: Batched forward model.
        solve_masses: Run the batched
            :func:`~arachne.inference.initialisation.solve_component_masses`
            after the moment seed (additive models only).
        **blind_kwargs: Forwarded to ``blind_initial_theta`` (``size_scales``,
            ``neutral_values``, ``shared_values``, ``sersic_n``).

    Returns:
        Full thetas of shape ``(N, n_params)``, float32.
    """
    spatial_model = bfm.template.spatial_model
    rows = []
    for i in range(bfm.n_galaxies):
        obs = bfm.observation(i)
        rows.append(blind_initial_theta(spatial_model, obs, **blind_kwargs))
    thetas = bfm.initial_theta_from_spatial(jnp.stack(rows))
    if solve_masses and isinstance(spatial_model, AdditiveComponentModel):
        thetas = _mass_solver(bfm)(bfm.data, thetas)
    return jnp.asarray(thetas, dtype=jnp.float32)


def _mass_solver(bfm: BatchedForwardModel):
    """Return (and cache) the jitted, vmapped linear mass solve."""
    if "mass_solve" not in bfm._cache:

        def solve(pack, theta):
            return solve_component_masses(bfm.forward_model_from_pack(pack), theta)

        bfm._cache["mass_solve"] = jax.jit(jax.vmap(solve, in_axes=(bfm.in_axes, 0)))
    return bfm._cache["mass_solve"]


# ---------------------------------------------------------------------------
# Batched MAP
# ---------------------------------------------------------------------------


@dataclass
class BatchedMAPResult:
    """Output of :func:`batched_find_map` / :func:`batched_multistart_map`.

    Attributes:
        theta: MAP estimates of shape ``(N, n_params)``, float32.
        log_post: ``log_posterior(theta)`` per galaxy, shape ``(N,)``.
        archetype_index: Index of the winning start per galaxy, shape ``(N,)``;
            ``0`` is ``theta0`` itself and ``k >= 1`` is ``archetypes[k - 1]``.
        history: Per-galaxy Adam loss (``-log_posterior``) trace of the winning
            start, shape ``(N, n_steps)``.
        n_steps: Total number of Adam steps per start.
        start_labels: Human-readable label of each start, in index order.
    """

    theta: jnp.ndarray
    log_post: jnp.ndarray
    archetype_index: np.ndarray
    history: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    n_steps: int = 0
    start_labels: list[str] = field(default_factory=list)

    @property
    def neg_log_posterior(self) -> np.ndarray:
        """``-log_post`` per galaxy, matching ``MAPResult.neg_log_posterior``."""
        return -np.asarray(self.log_post, dtype=np.float64)

    def summary(self, galaxy_ids: Sequence[str] | None = None) -> str:
        """One line per galaxy: winning start and final log-posterior.

        Args:
            galaxy_ids: Optional labels; defaults to the row index.

        Returns:
            Multi-line table string.
        """
        n = int(np.asarray(self.log_post).shape[0])
        ids = list(galaxy_ids) if galaxy_ids is not None else [str(i) for i in range(n)]
        lines = [f"{'galaxy':<24}{'log_post':>14}  start"]
        for i in range(n):
            k = int(self.archetype_index[i])
            label = self.start_labels[k] if k < len(self.start_labels) else f"start {k}"
            lines.append(f"{ids[i]:<24}{float(self.log_post[i]):>14.2f}  {label}")
        return "\n".join(lines)


def _adam_runner(bfm: BatchedForwardModel, clip_norm: float):
    """Return (and cache) ``run(data, thetas, lr, n_steps) -> (thetas, losses)``.

    The Adam state and the global-norm clip are *per galaxy* (the optimiser is
    built inside the vmapped function), so the batched run is exactly ``N``
    independent copies of the optimiser used by
    :func:`~arachne.inference.initialisation.find_map`.

    Args:
        bfm: Batched forward model.
        clip_norm: Global gradient-norm clip, per galaxy.

    Returns:
        Jitted callable; ``n_steps`` is static.
    """
    key = ("adam", float(clip_norm))
    if key not in bfm._cache:

        def loss(pack, theta):
            return -bfm.forward_model_from_pack(pack).log_posterior(theta)

        value_and_grad = jax.value_and_grad(loss, argnums=1)

        def run_single(pack, theta, lr, n_steps):
            opt = optax.chain(optax.clip_by_global_norm(float(clip_norm)), optax.adam(lr))
            opt_state = opt.init(theta)

            def step(carry, _):
                t, s = carry
                value, grad = value_and_grad(pack, t)
                updates, s = opt.update(grad, s, t)
                return (optax.apply_updates(t, updates), s), value

            (theta, _), losses = jax.lax.scan(step, (theta, opt_state), None, length=n_steps)
            return theta, losses

        batched = jax.vmap(run_single, in_axes=(bfm.in_axes, 0, None, None))
        bfm._cache[key] = jax.jit(batched, static_argnums=3)
    return bfm._cache[key]


def batched_find_map(
    bfm: BatchedForwardModel,
    thetas0: jnp.ndarray,
    n_rounds: int = 4,
    steps_per_round: int = 300,
    lr: float = 0.05,
    final_steps: int = 400,
    final_lr: float = 0.01,
    resolve_masses: bool = True,
    clip_norm: float = 50.0,
    order_by_size: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, np.ndarray]:
    """Vmapped Adam MAP for every galaxy, in one compiled program per step count.

    The schedule mirrors :func:`~arachne.inference.initialisation.find_map`
    exactly — ``n_rounds`` rounds of ``steps_per_round`` Adam steps at ``lr``
    with a fresh optimiser state each round and the linear mass solve applied
    before the first round and after every round, then ``final_steps`` at
    ``final_lr`` — but every step updates all ``N`` galaxies at once.

    Args:
        bfm: Batched forward model.
        thetas0: Starting full thetas of shape ``(N, n_params)``.
        n_rounds: Number of Adam rounds at ``lr``.
        steps_per_round: Adam steps per round.
        lr: Adam learning rate for the rounds.
        final_steps: Steps in the final polish (0 to skip).
        final_lr: Learning rate of the final polish.
        resolve_masses: Re-solve component masses linearly between rounds.
        clip_norm: Per-galaxy global gradient-norm clip.
        order_by_size: Order components compact-first at the end.

    Returns:
        Tuple ``(thetas (N, d), log_post (N,), history (N, n_steps))``.
    """
    thetas = jnp.asarray(thetas0, dtype=jnp.float32)
    if thetas.ndim != 2 or thetas.shape != (bfm.n_galaxies, bfm.n_params):
        raise ValueError(
            f"thetas0 must have shape ({bfm.n_galaxies}, {bfm.n_params}); got {thetas.shape}"
        )
    is_additive = isinstance(bfm.template.spatial_model, AdditiveComponentModel)
    do_masses = bool(resolve_masses and is_additive)
    run_adam = _adam_runner(bfm, clip_norm)
    solve = _mass_solver(bfm) if do_masses else None

    history: list[np.ndarray] = []
    if do_masses:
        thetas = solve(bfm.data, thetas)
    for rnd in range(int(n_rounds)):
        if steps_per_round > 0:
            thetas, losses = run_adam(bfm.data, thetas, jnp.float32(lr), int(steps_per_round))
            history.append(np.asarray(losses, dtype=np.float32))
        if do_masses:
            thetas = solve(bfm.data, thetas)
        logger.info(
            f"batched_find_map: round {rnd + 1}/{n_rounds} mean -log_post = "
            f"{float(-jnp.mean(bfm.log_posterior(thetas))):.2f}"
        )
    if final_steps > 0:
        thetas, losses = run_adam(bfm.data, thetas, jnp.float32(final_lr), int(final_steps))
        history.append(np.asarray(losses, dtype=np.float32))

    if order_by_size and is_additive:
        thetas = bfm.order_components_by_size(thetas)

    thetas = jnp.asarray(thetas, dtype=jnp.float32)
    log_post = bfm.log_posterior(thetas)
    hist = (
        np.concatenate(history, axis=1)
        if history
        else np.zeros((bfm.n_galaxies, 0), dtype=np.float32)
    )
    return thetas, log_post, hist


def batched_multistart_map(
    bfm: BatchedForwardModel,
    theta0: jnp.ndarray,
    archetypes: Sequence[Mapping[str, float]] = (),
    **find_map_kwargs: Any,
) -> BatchedMAPResult:
    """Batched :func:`batched_find_map` from ``theta0`` and each SPS archetype.

    An archetype is a ``{param_name: physical value}`` dict applied to *all*
    components of *all* galaxies (shared names are allowed), e.g.
    ``[{"Av": 0.3}, {"Av": 2.0}]`` to probe the dust/age degeneracy from both
    sides.  ``theta0`` itself is always start 0.  Each start is one compiled
    program covering the whole sample, and the compiled graph is reused across
    starts, so ``S`` archetypes cost ``S`` batched runs and a single compile.

    The winner is chosen **per galaxy**: galaxy ``i`` keeps the start with the
    highest ``log_posterior``.

    Args:
        bfm: Batched forward model.
        theta0: Base starting thetas of shape ``(N, n_params)``.
        archetypes: Physical overrides, one dict per additional start.
        **find_map_kwargs: Forwarded to :func:`batched_find_map`.

    Returns:
        :class:`BatchedMAPResult`.

    Raises:
        TypeError: If the spatial model is not an ``AdditiveComponentModel``.
        ValueError: If an archetype names a parameter that is not free.
    """
    model = bfm.template.spatial_model
    if not isinstance(model, AdditiveComponentModel):
        raise TypeError(
            "batched_multistart_map requires an AdditiveComponentModel spatial model, "
            f"got {type(model).__name__}"
        )
    theta0 = jnp.asarray(theta0, dtype=jnp.float32)

    starts: list[tuple[str, jnp.ndarray]] = [("theta0", theta0)]
    for i, arch in enumerate(archetypes):
        arch = dict(arch)
        if not arch:
            continue
        thetas = jax.vmap(lambda t, a=arch: _apply_archetype(model, t, a))(theta0)
        if np.array_equal(np.asarray(thetas), np.asarray(theta0)):
            continue
        starts.append((f"archetype {i} {arch}", thetas))

    best_theta: jnp.ndarray | None = None
    best_log_post: jnp.ndarray | None = None
    best_index: np.ndarray | None = None
    best_history: np.ndarray | None = None
    for s, (label, start) in enumerate(starts):
        thetas, log_post, hist = batched_find_map(bfm, start, **find_map_kwargs)
        log_post = jnp.nan_to_num(log_post, nan=-jnp.inf)
        if best_theta is None:
            best_theta, best_log_post = thetas, log_post
            best_index = np.zeros(bfm.n_galaxies, dtype=np.int32)
            best_history = hist
        else:
            better = np.asarray(log_post > best_log_post)
            best_theta = jnp.where(better[:, None], thetas, best_theta)
            best_log_post = jnp.where(better, log_post, best_log_post)
            best_index = np.where(better, s, best_index).astype(np.int32)
            if hist.shape == best_history.shape:
                best_history = np.where(better[:, None], hist, best_history)
        logger.info(
            f"batched_multistart_map: {label}: mean log_post = {float(jnp.mean(log_post)):.2f}"
        )

    assert best_theta is not None and best_log_post is not None and best_index is not None
    return BatchedMAPResult(
        theta=best_theta,
        log_post=best_log_post,
        archetype_index=best_index,
        history=best_history if best_history is not None else np.zeros((0, 0), dtype=np.float32),
        n_steps=int(best_history.shape[1]) if best_history is not None else 0,
        start_labels=[label for label, _ in starts],
    )


# ---------------------------------------------------------------------------
# Batched NUTS
# ---------------------------------------------------------------------------


@dataclass
class BatchedNUTSResult:
    """Multi-galaxy, multi-chain NUTS output with per-galaxy diagnostics.

    Attributes:
        chains: Samples of shape ``(N, n_chains, n_samples, n_params)``.
        diagnostics: Dict with ``rhat`` ``(N, d)``, ``ess`` ``(N, d)``,
            ``n_divergent`` ``(N,)``, ``mean_tree_depth`` ``(N,)``,
            ``acceptance_rate`` ``(N,)``, ``fraction_dims_moved`` ``(N,)``,
            ``step_size`` ``(N, n_chains)``, plus the scalars
            ``warmup_per_chain`` and ``max_num_doublings``.
        spatial_model: The shared spatial model (needed to decode theta).
        galaxy_ids: Per-galaxy labels.
        infos: BlackJAX info pytree with leading axes ``(N, n_chains, n_samples)``,
            or ``None``.
    """

    chains: jnp.ndarray
    diagnostics: dict
    spatial_model: SpatialModel
    galaxy_ids: list[str] = field(default_factory=list)
    infos: Any = None

    @property
    def n_galaxies(self) -> int:
        """Number of galaxies."""
        return int(self.chains.shape[0])

    @property
    def n_chains(self) -> int:
        """Number of chains per galaxy."""
        return int(self.chains.shape[1])

    @property
    def n_samples(self) -> int:
        """Number of draws per chain."""
        return int(self.chains.shape[2])

    @property
    def samples(self) -> jnp.ndarray:
        """Per-galaxy samples with the chains concatenated, ``(N, C*S, d)``."""
        n, c, s, d = self.chains.shape
        return self.chains.reshape(n, c * s, d)

    def result(self, i: int) -> NUTSResult:
        """Single-galaxy :class:`NUTSResult` view, for existing per-galaxy tooling.

        Args:
            i: Galaxy index.

        Returns:
            ``NUTSResult`` holding galaxy ``i``'s chains and diagnostics.
        """
        index = int(i)
        diag = {
            key: (value[index] if key in _PER_GALAXY_DIAGNOSTICS else value)
            for key, value in self.diagnostics.items()
        }
        chains = self.chains[index]
        return NUTSResult(
            samples=chains.reshape(-1, chains.shape[-1]),
            infos=None
            if self.infos is None
            else jax.tree_util.tree_map(lambda x: x[index], self.infos),
            spatial_model=self.spatial_model,
            chains=chains,
            diagnostics=diag,
        )

    def summary(self, header: bool = True) -> str:
        """One line per galaxy: worst R-hat, smallest ESS, divergences, saturation.

        Args:
            header: Prepend the column header.  ``False`` gives exactly
                ``n_galaxies`` lines, for callers that append the rows to a
                larger table.

        Returns:
            Table string with ``n_galaxies`` galaxy rows (plus the header).
        """
        d = self.diagnostics
        cap = d.get("max_num_doublings")
        rhat = np.asarray(d["rhat"], dtype=float)
        e = np.asarray(d["ess"], dtype=float)
        n_div = np.asarray(d["n_divergent"], dtype=int)
        depth = np.asarray(d["mean_tree_depth"], dtype=float)
        accept = np.asarray(d["acceptance_rate"], dtype=float)
        ids = self.galaxy_ids or [str(i) for i in range(self.n_galaxies)]
        lines = (
            [f"{'galaxy':<24}{'maxRhat':>9}{'minESS':>9}{'div':>7}{'depth':>9}{'accept':>9}  flags"]
            if header
            else []
        )
        for i in range(self.n_galaxies):
            row_rhat = float(np.nanmax(rhat[i])) if np.any(np.isfinite(rhat[i])) else float("nan")
            row_ess = float(np.nanmin(e[i])) if np.any(np.isfinite(e[i])) else float("nan")
            flags = []
            if np.isfinite(row_rhat) and row_rhat > 1.05:
                flags.append("NOT-CONVERGED")
            if cap is not None and np.isfinite(depth[i]) and depth[i] >= 2 ** int(cap) - 1:
                flags.append("TREE-SATURATED")
            if n_div[i] > 0:
                flags.append("DIVERGENCES")
            lines.append(
                f"{ids[i]:<24}{row_rhat:>9.3f}{row_ess:>9.0f}{int(n_div[i]):>7d}"
                f"{depth[i]:>9.1f}{accept[i]:>9.3f}  {','.join(flags) if flags else 'ok'}"
            )
        return "\n".join(lines)

    def to_hdf5(self, path: str | Path) -> None:
        """Write one HDF5 group per galaxy with its chains and diagnostics.

        Layout: ``/<galaxy_id>/chains`` ``(C, S, d)``, ``/<galaxy_id>/theta``
        ``(C*S, d)``, ``/<galaxy_id>/rhat``, ``/<galaxy_id>/ess``, plus scalar
        diagnostics as attributes of that group.  Root attributes record the
        sample-wide settings.

        Args:
            path: Output HDF5 file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.diagnostics
        with h5py.File(path, "w") as f:
            f.attrs["sampler"] = "batched_nuts"
            f.attrs["n_galaxies"] = self.n_galaxies
            f.attrs["n_chains"] = self.n_chains
            f.attrs["n_samples"] = self.n_samples
            f.attrs["warmup_per_chain"] = bool(d.get("warmup_per_chain", True))
            f.attrs["max_num_doublings"] = int(d.get("max_num_doublings", 0))
            if hasattr(self.spatial_model, "sps_param_names"):
                f.attrs["param_names"] = np.array(
                    [s.encode("utf-8") for s in self.spatial_model.sps_param_names]
                )
            ids = self.galaxy_ids or [f"galaxy_{i:04d}" for i in range(self.n_galaxies)]
            chains = np.asarray(self.chains)
            for i, gid in enumerate(ids):
                group = f.create_group(str(gid))
                group.create_dataset("chains", data=chains[i], compression="gzip")
                group.create_dataset(
                    "theta", data=chains[i].reshape(-1, chains.shape[-1]), compression="gzip"
                )
                group.create_dataset("rhat", data=np.asarray(d["rhat"])[i])
                group.create_dataset("ess", data=np.asarray(d["ess"])[i])
                group.create_dataset("step_size", data=np.asarray(d["step_size"])[i])
                for key in (
                    "n_divergent",
                    "mean_tree_depth",
                    "acceptance_rate",
                    "fraction_dims_moved",
                ):
                    group.attrs[key] = np.asarray(d[key])[i].item()
        logger.info(f"BatchedNUTSResult saved to {path} ({self.n_galaxies} galaxy groups)")


#: Diagnostics whose leading axis is the galaxy axis.
_PER_GALAXY_DIAGNOSTICS = frozenset(
    {
        "rhat",
        "ess",
        "n_divergent",
        "mean_tree_depth",
        "acceptance_rate",
        "fraction_dims_moved",
        "step_size",
    }
)


def _split_grid(key: jax.Array, n_rows: int, n_cols: int) -> jax.Array:
    """Split ``key`` into an ``(n_rows, n_cols)`` grid of keys.

    Works with both raw ``uint32`` keys (trailing axis of 2) and the newer typed
    key arrays (no trailing axis).

    Args:
        key: PRNG key.
        n_rows: Outer count (galaxies).
        n_cols: Inner count (chains).

    Returns:
        Key array whose leading two axes are ``(n_rows, n_cols)``.
    """
    keys = jax.random.split(key, int(n_rows) * int(n_cols))
    return jnp.reshape(keys, (int(n_rows), int(n_cols), *keys.shape[1:]))


def batched_nuts(
    bfm: BatchedForwardModel,
    theta_init: jnp.ndarray,
    rng_key: jax.Array,
    n_warmup: int = 500,
    n_samples: int = 1000,
    n_chains: int = 2,
    chain_jitter: float = 0.1,
    max_num_doublings: int = 8,
    target_accept_rate: float = 0.8,
    inverse_mass_matrix: jnp.ndarray | None = None,
) -> BatchedNUTSResult:
    """Multi-chain NUTS for every galaxy in one compiled program.

    Exactly the algorithm of
    :class:`~arachne.inference.nuts_sampler.NUTSSampler` — ``blackjax``
    window adaptation per chain followed by a ``lax.scan`` sampling loop — with
    the galaxy axis added as an outer ``jax.vmap``.  Each ``(galaxy, chain)``
    pair therefore adapts its own step size and diagonal mass matrix, and the
    whole sample is a single XLA computation.

    If the vmapped window adaptation raises (a blackjax-version hazard the
    single-galaxy sampler guards against too), the run falls back to adapting
    **chain 0** of each galaxy and sharing its step size and mass matrix across
    that galaxy's chains; ``diagnostics["warmup_per_chain"]`` records which path
    was taken.

    Args:
        bfm: Batched forward model.
        theta_init: Starting points, ``(N, n_params)`` (jittered across chains)
            or an explicit ``(N, n_chains, n_params)`` array (used as given).
        rng_key: PRNG key.
        n_warmup: Warmup steps per chain.
        n_samples: Draws per chain after warmup.
        n_chains: Chains per galaxy.  Four or more make split-R-hat meaningful.
        chain_jitter: Per-chain starting-point jitter in units of the posterior
            standard deviation implied by ``inverse_mass_matrix``.  Chain 0
            always starts exactly at ``theta_init``.
        max_num_doublings: NUTS tree-doubling cap.
        target_accept_rate: Dual-averaging target acceptance rate.
        inverse_mass_matrix: Optional shared initial inverse mass matrix
            (diagonal ``(d,)`` or dense ``(d, d)``) seeding the adaptation and
            the jitter geometry.

    Returns:
        :class:`BatchedNUTSResult`.

    Raises:
        ImportError: If ``blackjax`` is not installed.
        ValueError: If ``n_chains < 1`` or ``theta_init`` has a bad shape.
    """
    try:
        import blackjax
    except ImportError as e:  # pragma: no cover - blackjax is a hard dependency here
        raise ImportError(
            "blackjax is required for NUTS sampling. Install it with: pip install blackjax"
        ) from e

    if n_chains < 1:
        raise ValueError(f"n_chains must be >= 1; got {n_chains}")
    theta_init = jnp.asarray(theta_init, dtype=jnp.float32)
    if theta_init.shape[0] != bfm.n_galaxies or theta_init.shape[-1] != bfm.n_params:
        raise ValueError(
            f"theta_init must have shape ({bfm.n_galaxies}, [{n_chains},] {bfm.n_params}); "
            f"got {theta_init.shape}"
        )

    rng_key, jitter_key = jax.random.split(rng_key)
    jitter_keys = jax.random.split(jitter_key, bfm.n_galaxies)
    inits = jax.vmap(lambda t, k: _jitter_inits(t, n_chains, chain_jitter, inverse_mass_matrix, k))(
        theta_init, jitter_keys
    )  # (N, C, d)

    warmup_kwargs: dict[str, Any] = {}
    if inverse_mass_matrix is not None:
        warmup_kwargs["initial_inverse_mass_matrix"] = jnp.asarray(inverse_mass_matrix)

    def _logpost_fn(pack):
        fm = bfm.forward_model_from_pack(pack)
        return fm.log_posterior

    def _sample_chains(logpost, keys, states, step_sizes, inv_masses):
        """Vmapped lax.scan sampling loop over one galaxy's chains."""

        def sample_one(key, state, step_size, inv_mass):
            kernel = blackjax.nuts(
                logpost,
                step_size=step_size,
                inverse_mass_matrix=inv_mass,
                max_num_doublings=max_num_doublings,
            )

            def one_step(carry, k):
                new_state, info = kernel.step(k, carry)
                return new_state, (new_state.position, info)

            _, out = jax.lax.scan(one_step, state, jax.random.split(key, n_samples))
            return out

        return jax.vmap(sample_one)(keys, states, step_sizes, inv_masses)

    def run_galaxy(pack, galaxy_inits, warmup_keys, sample_keys):
        logpost = _logpost_fn(pack)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logpost,
            target_acceptance_rate=target_accept_rate,
            max_num_doublings=max_num_doublings,
            **warmup_kwargs,
        )

        def warm_one(key, position):
            (state, params), _info = warmup.run(key, position, n_warmup)
            return state, params["step_size"], params["inverse_mass_matrix"]

        states, step_sizes, inv_masses = jax.vmap(warm_one)(warmup_keys, galaxy_inits)
        chains, infos = _sample_chains(logpost, sample_keys, states, step_sizes, inv_masses)
        return chains, infos, step_sizes

    def run_galaxy_shared_warmup(pack, galaxy_inits, warmup_keys, sample_keys):
        logpost = _logpost_fn(pack)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logpost,
            target_acceptance_rate=target_accept_rate,
            max_num_doublings=max_num_doublings,
            **warmup_kwargs,
        )
        (state0, params), _info = warmup.run(warmup_keys[0], galaxy_inits[0], n_warmup)
        step0 = params["step_size"]
        inv0 = params["inverse_mass_matrix"]
        step_sizes = jnp.full((n_chains,), step0)
        inv_masses = jnp.tile(inv0[None, ...], (n_chains,) + (1,) * inv0.ndim)
        init_kernel = blackjax.nuts(
            logpost,
            step_size=step0,
            inverse_mass_matrix=inv0,
            max_num_doublings=max_num_doublings,
        )
        states = jax.vmap(init_kernel.init)(galaxy_inits)
        del state0
        chains, infos = _sample_chains(logpost, sample_keys, states, step_sizes, inv_masses)
        return chains, infos, step_sizes

    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    warmup_keys = _split_grid(warmup_key, bfm.n_galaxies, n_chains)
    sample_keys = _split_grid(sample_key, bfm.n_galaxies, n_chains)

    logger.info(
        f"Starting batched NUTS: {bfm.n_galaxies} galaxies x {n_chains} chain(s), "
        f"{n_warmup} warmup + {n_samples} samples each, {bfm.n_params} parameters"
    )
    warmup_per_chain = True
    try:
        chains, infos, step_sizes = jax.jit(jax.vmap(run_galaxy, in_axes=(bfm.in_axes, 0, 0, 0)))(
            bfm.data, inits, warmup_keys, sample_keys
        )
    except Exception as exc:  # pragma: no cover - blackjax-version dependent
        warmup_per_chain = False
        logger.warning(
            f"vmapped per-chain window_adaptation failed ({type(exc).__name__}: {exc}); "
            "adapting chain 0 of each galaxy and sharing its step size / mass matrix."
        )
        chains, infos, step_sizes = jax.jit(
            jax.vmap(run_galaxy_shared_warmup, in_axes=(bfm.in_axes, 0, 0, 0))
        )(bfm.data, inits, warmup_keys, sample_keys)

    chains = jnp.asarray(chains)  # (N, C, S, d)
    diagnostics = _batched_diagnostics(
        chains, infos, step_sizes, warmup_per_chain, max_num_doublings
    )
    result = BatchedNUTSResult(
        chains=chains,
        diagnostics=diagnostics,
        spatial_model=bfm.template.spatial_model,
        galaxy_ids=list(bfm.galaxy_ids),
        infos=infos,
    )
    logger.info("batched NUTS complete:\n" + result.summary())
    return result


def _batched_diagnostics(
    chains: jnp.ndarray,
    infos: Any,
    step_sizes: jnp.ndarray,
    warmup_per_chain: bool,
    max_num_doublings: int,
) -> dict:
    """Per-galaxy convergence diagnostics from batched chains and infos.

    Args:
        chains: ``(N, C, S, d)`` samples.
        infos: BlackJAX info pytree with leading axes ``(N, C, S)``.
        step_sizes: ``(N, C)`` adapted step sizes.
        warmup_per_chain: Whether warmup ran separately per chain.
        max_num_doublings: Tree-doubling cap used.

    Returns:
        Diagnostics dict; see :class:`BatchedNUTSResult`.
    """
    chains_np = np.asarray(chains)
    n_galaxies = chains_np.shape[0]
    rhat = np.stack([split_rhat(chains_np[i]) for i in range(n_galaxies)])
    ess_arr = np.stack([ess(chains_np[i]) for i in range(n_galaxies)])
    moved = np.array([chain_movement(chains_np[i]) for i in range(n_galaxies)], dtype=np.float64)

    def _per_galaxy(attr: str, reducer) -> np.ndarray:
        if not hasattr(infos, attr):
            return np.full(n_galaxies, np.nan)
        arr = np.asarray(getattr(infos, attr))
        return reducer(arr.reshape(n_galaxies, -1), axis=1)

    return {
        "rhat": rhat,
        "ess": ess_arr,
        "n_divergent": _per_galaxy("is_divergent", np.sum).astype(np.int64),
        "mean_tree_depth": _per_galaxy("num_integration_steps", np.mean),
        "acceptance_rate": _per_galaxy("acceptance_rate", np.mean),
        "fraction_dims_moved": moved,
        "step_size": np.asarray(step_sizes, dtype=np.float64),
        "warmup_per_chain": bool(warmup_per_chain),
        "max_num_doublings": int(max_num_doublings),
    }


# ---------------------------------------------------------------------------
# Sequential nested sampling
# ---------------------------------------------------------------------------


def fit_batch_nss(
    bfm: BatchedForwardModel,
    rng_key: jax.Array,
    initial_thetas: jnp.ndarray | None = None,
    checkpoint_dir: str | Path | None = None,
    **nss_kwargs: Any,
) -> list[NSSResult]:
    """Run nested slice sampling on each galaxy in turn.

    NSS cannot join the batched program: its termination criterion drives a
    Python ``while`` loop whose length differs per galaxy, so the galaxies would
    have to run to the *longest* galaxy's step count with masked updates.  This
    helper therefore loops, building a single-galaxy
    :meth:`BatchedForwardModel.forward_model` view for each galaxy and handing it
    to :class:`~arachne.inference.nss_sampler.NSSSampler`.

    .. note::
       ``blackjax.nss`` closes over the galaxy's ``log_prior`` / ``log_likelihood``
       and ``NSSSampler`` jits the resulting step function, so each galaxy pays
       its own XLA compile.  The *shapes* are identical across galaxies, so the
       cost is compile time only — expect a few minutes per galaxy for the first
       call on GPU and roughly the same for the rest.  Batch the MAP stage (which
       does share one program) and reserve NSS for the galaxies that need an
       evidence.

    Args:
        bfm: Batched forward model.
        rng_key: PRNG key; split once per galaxy.
        initial_thetas: Optional ``(N, num_live, n_params)`` prior draws, e.g.
            from :meth:`BatchedForwardModel.sample_prior`.  ``None`` lets each
            sampler draw its own live points from the prior.
        checkpoint_dir: Optional directory; each galaxy checkpoints to
            ``<dir>/<galaxy_id>.npz``.
        **nss_kwargs: Forwarded to the ``NSSSampler`` constructor (``num_live``,
            ``num_inner_steps``, ``num_delete``, ``termination``,
            ``n_samples_out``, ``max_steps``).

    Returns:
        One :class:`~arachne.inference.nss_sampler.NSSResult` per galaxy, in
        galaxy order.
    """
    keys = jax.random.split(rng_key, bfm.n_galaxies)
    results: list[NSSResult] = []
    for i in range(bfm.n_galaxies):
        fm = bfm.forward_model(i)
        sampler = NSSSampler(fm, **nss_kwargs)
        initial = None if initial_thetas is None else jnp.asarray(initial_thetas)[i]
        path = None if checkpoint_dir is None else Path(checkpoint_dir) / f"{bfm.galaxy_ids[i]}.npz"
        start = time.perf_counter()
        result = sampler.run(keys[i], initial_theta=initial, checkpoint_path=path)
        logger.info(
            f"fit_batch_nss: {bfm.galaxy_ids[i]} logZ = {result.logZ:.3f} +/- "
            f"{result.logZ_err:.3f} in {time.perf_counter() - start:.1f} s"
        )
        results.append(result)
    return results
