"""Additive-flux multi-component spatial model."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from arachne.emulator.base import SPSEmulator
from arachne.spatial.base import SpatialModel
from arachne.spatial.profiles import (
    PointSourceProfile,
    Profile,
    get_profile,
    log_render_on_grid,
)

_LN10 = math.log(10.0)
_NORMALISATIONS = ("frame", "analytic")


class AdditiveComponentModel(SpatialModel):
    """K additive components, each a surface-brightness profile with its own SED.

    Light is additive; SPS parameters are not.  This model therefore describes
    the galaxy as a sum of ``K`` components.  Component ``k`` has a
    surface-brightness profile ``P_k(y, x)`` carrying unit total flux (see
    ``normalisation`` below), and a full SED ``F_k`` (N_bands,) predicted by
    the emulator from that component's *own* SPS parameters, including its own
    total stellar mass.  The unconvolved model image is::

        I_b(y, x) = Σ_k F_kb · P_k(y, x)

    which costs exactly ``K`` emulator evaluations per likelihood call, no
    matter how many pixels the image has.  This is the physically correct
    replacement for :class:`~arachne.spatial.gmm.GaussianMixtureSpatialModel`
    for bulge/disk-style decompositions.

    Profiles
    --------
    The shape of each component is delegated to a
    :class:`~arachne.spatial.profiles.Profile`: ``"gaussian"`` (the default and
    the historical behaviour), ``"sersic"`` (bulges ``n ≈ 4``, disks
    ``n ≈ 1``) or ``"point"`` (an AGN / nuclear source that renders as the PSF
    alone).  ``profiles`` takes one name for all components or one entry per
    component, and entries may be pre-built ``Profile`` instances if their
    settings need tuning.  Profiles may have different numbers of shape
    parameters, so component blocks are not all the same length in general.

    Coordinates
    -----------
    With ``pixel_scale=None`` (default) coordinates are pixel indices and
    ``mu`` / ``sigma`` are in pixels.  With ``pixel_scale`` set, coordinates
    are arcsec offsets from the frame centre ``((H - 1) / 2, (W - 1) / 2)``,
    and ``mu`` / ``sigma`` are in arcsec.  Arcsec mode is what makes the same
    component renderable on another band's pixel grid (via that band's WCS)
    without resampling — see :meth:`model_image_on`.  Profile defaults that
    carry a length follow the coordinate units: a component asked for by the
    *name* ``"point"`` gets ``point_sigma = 0.5 * pixel_scale`` in arcsec mode,
    the same half-pixel kernel as in pixel mode (pass an explicit
    ``PointSourceProfile(point_sigma=...)`` to override).

    Normalisation
    -------------
    - ``"frame"`` (default, historical): each profile is divided by its sum
      over the model's own grid, so it sums to exactly 1 there.  Cheap and
      shift-invariant (a component far outside the frame still normalises),
      but not physical, and meaningless on any other grid.
    - ``"analytic"``: the profile is normalised so that its integral over the
      *whole plane* is 1; the summed flux fraction over the frame is ≤ 1 and
      flux that falls outside the cutout is lost, as in reality.  Required for
      multi-resolution rendering.

    ``oversample`` evaluates the profile at ``oversample × oversample``
    sub-positions per pixel and averages: leave it at 1 for Gaussians, raise it
    to 3–5 for cuspy Sérsic components.

    Parameter roles
    ---------------
    Every name in ``emulator_param_names`` (the emulator's full ordered input
    list) must play exactly one of three roles:

    - **fixed** (``fixed_params``): held constant at the given value.
    - **shared** (``shared_param_names``): one free value common to all
      components (e.g. redshift).
    - **per-component** (everything else): one free value per component.
      ``mass_param`` must be per-component.

    Per-component free names, in emulator order, are exposed as
    ``sps_param_names`` (this is what ``decode`` returns columns for).

    theta layout
    ------------
    ``theta`` is a flat float32 vector::

        theta = concat([block_0, block_1, ..., block_{K-1}, shared_raw])
        block_k = [shape_k (profile_k.n_shape,), sps_raw_k (N_free,)]

    so ``n_params = Σ_k (n_shape_k + N_free) + N_shared``.  For the common case
    of a single profile for every component this is
    ``K * (n_shape + N_free) + N_shared`` and, with the default Gaussian
    profile, exactly the historical ``K * (5 + N_free) + N_shared`` with
    ``shape_k = [mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho]``.

    Use :meth:`split_theta` / :meth:`join_theta`, :attr:`component_slices` or
    the :meth:`shape_raw` / :meth:`sps_raw` / :meth:`set_sps_raw` helpers
    rather than slicing by hand.

    Physical mapping
    ----------------
    - shape parameters: see the component's ``Profile`` (Gaussian and Sérsic
      share ``mu``, ``sigma = exp(log_sigma)``, ``rho = tanh(atanh_rho)``
      clipped to ±0.99; Sérsic adds ``n = exp(log_n)``; point sources have only
      ``mu``).
    - every SPS parameter (per-component and shared):
      ``phys = lo + (hi - lo) * sigmoid(raw)``.

    Prior
    -----
    ``log_prior`` is the sum of:

    (a) ``log_sigmoid(raw) + log_sigmoid(-raw)`` for every per-component and
        shared SPS entry: the normalised log-density of a prior *uniform in
        physical space* pulled back to ``raw`` (the sigmoid Jacobian);
    (b) ``sps_log_prior(sps_phys)`` if supplied, where ``sps_phys`` is the
        (K, N_emulator) physical matrix from :meth:`component_params`.  Shared
        parameters appear identically in every row — the callable is
        responsible for counting them once;
    (c) each component's ``Profile.log_prior``, i.e.
        ``Normal(centre, centre_prior_sigma)`` on each centre coordinate
        (default sigma ``frame size / 4``),
        ``Normal(mu, sd) = log_size_prior`` on each ``log_sigma``
        (default ``(log(frame size / 16), 1.0)``),
        ``Normal(0, rho_prior_sigma)`` on ``atanh_rho``, and for Sérsic
        components ``Normal(log 2, 0.7)`` on ``log_n``.  "Frame size" is
        ``max(H, W)`` in pixel mode and ``max(H, W) * pixel_scale`` in arcsec
        mode.

    Terms (a) and (c) make the prior a proper density in theta;
    :meth:`sample_prior` draws from exactly that density (without the optional
    ``sps_log_prior`` term).

    ``decode`` is a summary map for plotting / ``NUTSResult.get_parameter_map``
    compatibility.  It is **not** what the likelihood uses.

    Attributes:
        n_components: Number of components K.
        image_shape: (H, W).
        emulator_param_names: Full ordered emulator input list.
        sps_param_names: Per-component free SPS names (emulator order).
        shared_param_names: Shared free SPS names (emulator order).
        fixed_params: Dict of fixed name -> value.
        param_bounds: Dict of name -> (lo, hi) for every free parameter.
        mass_param: Name of the per-component log10 stellar-mass parameter.
        mass_index: Column of ``mass_param`` within ``sps_param_names``.
        profile_objects: Tuple of K ``Profile`` instances.
        pixel_scale: Arcsec per pixel, or None for pixel coordinates.
        normalisation: ``"frame"`` or ``"analytic"``.
        oversample: Sub-samples per pixel side used when rendering.
        centre: Prior centre (2,) in the model's coordinate units.
    """

    def __init__(
        self,
        n_components: int,
        emulator_param_names: list[str],
        param_bounds: dict[str, tuple[float, float]],
        image_shape: tuple[int, int],
        shared_param_names: Sequence[str] = (),
        fixed_params: dict[str, float] | None = None,
        mass_param: str = "log_mass",
        sps_log_prior: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        centre_prior_sigma: float | None = None,
        log_size_prior: tuple[float, float] | None = None,
        rho_prior_sigma: float = 1.0,
        *,
        profiles: str | Profile | Sequence[str | Profile] = "gaussian",
        pixel_scale: float | None = None,
        normalisation: str = "frame",
        oversample: int = 1,
    ) -> None:
        """Initialise the AdditiveComponentModel.

        The keyword-only arguments all default to the historical behaviour:
        ``K`` Gaussian components on a pixel-index grid, frame-normalised,
        one sample per pixel.

        Args:
            n_components: Number of additive components K (>= 1).
            emulator_param_names: The emulator's full ordered parameter list.
            param_bounds: name -> (lo, hi) physical bounds; required for every
                free (per-component or shared) parameter.
            image_shape: (H, W) of the galaxy image.
            shared_param_names: Names with one free value common to all components.
            fixed_params: name -> constant value, excluded from theta.
            mass_param: Name of the per-component log10 stellar mass parameter.
            sps_log_prior: Optional callable ``(K, N_emulator) -> scalar`` adding
                a physical-space prior on the component SPS matrix.
            centre_prior_sigma: Sigma of the Gaussian centre prior, in the model's
                coordinate units.  Default ``frame size / 4``.
            log_size_prior: (mean, sd) of the Gaussian prior on each log_sigma.
                Default ``(log(frame size / 16), 1.0)``.
            rho_prior_sigma: Sigma of the Gaussian prior on atanh_rho.
            profiles: One profile name / :class:`Profile` for every component,
                or a sequence of K of them.
            pixel_scale: Arcsec per pixel.  ``None`` keeps coordinates in pixel
                indices; a float switches to arcsec offsets from the frame centre.
            normalisation: ``"frame"`` (profile sums to 1 over the model's grid)
                or ``"analytic"`` (unit total flux over the whole plane).
            oversample: Integer sub-samples per pixel side (>= 1).

        Raises:
            ValueError: On unknown names, overlapping roles, a non-per-component
                mass parameter, missing bounds, ``n_components < 1``, a bad
                profile specification, a non-positive ``pixel_scale``, an
                unknown ``normalisation`` or ``oversample < 1``.
        """
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        emulator_param_names = list(emulator_param_names)
        if len(set(emulator_param_names)) != len(emulator_param_names):
            raise ValueError("emulator_param_names contains duplicates")
        fixed_params = dict(fixed_params or {})
        shared_list = list(shared_param_names)
        name_set = set(emulator_param_names)

        unknown_fixed = set(fixed_params) - name_set
        if unknown_fixed:
            raise ValueError(f"fixed_params not in emulator_param_names: {sorted(unknown_fixed)}")
        unknown_shared = set(shared_list) - name_set
        if unknown_shared:
            raise ValueError(
                f"shared_param_names not in emulator_param_names: {sorted(unknown_shared)}"
            )
        overlap = set(fixed_params) & set(shared_list)
        if overlap:
            raise ValueError(f"Parameters both fixed and shared: {sorted(overlap)}")
        if mass_param not in name_set:
            raise ValueError(f"mass_param {mass_param!r} not in emulator_param_names")
        if mass_param in fixed_params or mass_param in shared_list:
            raise ValueError(f"mass_param {mass_param!r} must be per-component (not fixed/shared)")

        shared_set = set(shared_list)
        # Roles, all in emulator order
        self._free_names = [
            p for p in emulator_param_names if p not in fixed_params and p not in shared_set
        ]
        self._shared_names = [p for p in emulator_param_names if p in shared_set]
        self._fixed_names = [p for p in emulator_param_names if p in fixed_params]

        missing = [p for p in self._free_names + self._shared_names if p not in param_bounds]
        if missing:
            raise ValueError(f"param_bounds missing for free parameters: {missing}")

        self.n_components = int(n_components)
        self.emulator_param_names = emulator_param_names
        self.param_bounds = param_bounds
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.fixed_params = fixed_params
        self.mass_param = mass_param
        self.sps_log_prior = sps_log_prior

        self._n_free = len(self._free_names)
        self._n_shared = len(self._shared_names)
        self._n_fixed = len(self._fixed_names)
        self._n_emulator = len(emulator_param_names)
        self.mass_index = self._free_names.index(mass_param)

        # --------------------------------------------------- coordinate units
        # Validated before the profiles are built because a profile named by
        # string may need the pixel scale to size itself (see
        # :func:`_resolve_profiles`).
        if pixel_scale is not None:
            pixel_scale = float(pixel_scale)
            if pixel_scale <= 0.0:
                raise ValueError(f"pixel_scale must be > 0, got {pixel_scale}")
        self.pixel_scale = pixel_scale

        # ------------------------------------------------------------ profiles
        self.profile_objects = _resolve_profiles(profiles, self.n_components, pixel_scale)
        self._n_shape_params = tuple(p.n_shape for p in self.profile_objects)

        # Variable-length component blocks: [shape_k, sps_raw_k]
        self._shape_slices: list[slice] = []
        self._sps_slices: list[slice] = []
        self.component_slices: list[slice] = []
        start = 0
        for n_shape in self._n_shape_params:
            self._shape_slices.append(slice(start, start + n_shape))
            self._sps_slices.append(slice(start + n_shape, start + n_shape + self._n_free))
            self.component_slices.append(slice(start, start + n_shape + self._n_free))
            start += n_shape + self._n_free
        self._n_block_total = start
        self.shared_slice = slice(start, start + self._n_shared)
        self._n_params = start + self._n_shared
        # split_theta can return a rectangular (K, block) array only when every
        # component block has the same length, i.e. the same number of shape
        # parameters (in particular whenever all components share a profile).
        self._rectangular = len(set(self._n_shape_params)) == 1
        self._block = (self._n_shape_params[0] + self._n_free) if self._rectangular else None

        # Permutation: emulator column j -> column in concat([free, shared, fixed])
        concat_order = self._free_names + self._shared_names + self._fixed_names
        self._perm = jnp.array(
            [concat_order.index(p) for p in emulator_param_names], dtype=jnp.int32
        )

        f32 = jnp.float32
        self._free_lows = jnp.array([param_bounds[p][0] for p in self._free_names], dtype=f32)
        self._free_highs = jnp.array([param_bounds[p][1] for p in self._free_names], dtype=f32)
        self._shared_lows = jnp.array([param_bounds[p][0] for p in self._shared_names], dtype=f32)
        self._shared_highs = jnp.array([param_bounds[p][1] for p in self._shared_names], dtype=f32)
        self._fixed_vals = jnp.array([fixed_params[p] for p in self._fixed_names], dtype=f32)

        # --------------------------------------------------------- coordinates
        if normalisation not in _NORMALISATIONS:
            raise ValueError(
                f"normalisation must be one of {_NORMALISATIONS}, got {normalisation!r}"
            )
        self.normalisation = str(normalisation)
        oversample = int(oversample)
        if oversample < 1:
            raise ValueError(f"oversample must be >= 1, got {oversample}")
        self.oversample = oversample

        H, W = self.image_shape
        rows = jnp.arange(H, dtype=f32)
        cols = jnp.arange(W, dtype=f32)
        if pixel_scale is None:
            self.pixel_area = 1.0
            frame_size = float(max(H, W))
            self.centre = jnp.array([(H - 1) / 2.0, (W - 1) / 2.0], dtype=f32)
        else:
            self.pixel_area = pixel_scale**2
            frame_size = float(max(H, W)) * pixel_scale
            self.centre = jnp.zeros(2, dtype=f32)
            rows = (rows - (H - 1) / 2.0) * pixel_scale
            cols = (cols - (W - 1) / 2.0) * pixel_scale
        yy, xx = jnp.meshgrid(rows, cols, indexing="ij")
        self._yy = yy.ravel()  # (H*W,)
        self._xx = xx.ravel()  # (H*W,)

        # Shape-prior hyper-parameters
        self.centre_prior_sigma = (
            float(centre_prior_sigma) if centre_prior_sigma is not None else frame_size / 4.0
        )
        if log_size_prior is None:
            log_size_prior = (math.log(frame_size / 16.0), 1.0)
        self.log_size_prior = (float(log_size_prior[0]), float(log_size_prior[1]))
        self.rho_prior_sigma = float(rho_prior_sigma)

    # ------------------------------------------------------------------ props
    @property
    def fixed_param_names(self) -> list[str]:
        """Names of the fixed emulator parameters, in the order of ``fixed_values``."""
        return list(self._fixed_names)

    @property
    def fixed_values(self) -> jnp.ndarray:
        """Physical values of the fixed parameters, shape (n_fixed,), float32."""
        return self._fixed_vals

    def with_fixed_values(self, values: jnp.ndarray) -> "AdditiveComponentModel":
        """Return a shallow copy with different values for the fixed parameters.

        Used to pin per-galaxy quantities (e.g. each galaxy's spectroscopic
        redshift) inside a batched, vmapped program: ``values`` may be a tracer.
        The set and order of fixed parameters is unchanged (``fixed_param_names``).

        Args:
            values: New physical values in ``fixed_param_names`` order, shape (n_fixed,).

        Returns:
            A copy of this model sharing all static configuration.

        Raises:
            ValueError: If ``values`` does not have shape (n_fixed,).
        """
        values = jnp.asarray(values, dtype=jnp.float32)
        if values.shape != (self._n_fixed,):
            raise ValueError(
                f"values must have shape ({self._n_fixed},) for fixed parameters "
                f"{self._fixed_names}, got {values.shape}."
            )
        model = copy.copy(self)
        model._fixed_vals = values
        return model

    @property
    def n_params(self) -> int:
        """Total free parameters: ``Σ_k (n_shape_k + N_free) + N_shared``."""
        return self._n_params

    @property
    def n_params_per_component(self) -> int:
        """Free parameters per component block: ``n_shape + N_free``.

        Raises:
            ValueError: If the components do not all have the same number of
                shape parameters (use :attr:`component_slices` instead).
        """
        if self._block is None:
            raise ValueError(
                "n_params_per_component is undefined for mixed-profile models; "
                "use component_slices / n_shape_params."
            )
        return self._block

    @property
    def n_shape_params(self) -> tuple[int, ...]:
        """Number of shape parameters of each component, length K."""
        return self._n_shape_params

    @property
    def sps_param_names(self) -> list[str]:
        """Per-component free SPS parameter names, in emulator order."""
        return list(self._free_names)

    @property
    def shared_param_names(self) -> list[str]:
        """Shared free SPS parameter names, in emulator order."""
        return list(self._shared_names)

    @property
    def n_free(self) -> int:
        """Number of per-component free SPS parameters."""
        return self._n_free

    @property
    def n_shared(self) -> int:
        """Number of shared free SPS parameters."""
        return self._n_shared

    @property
    def coords(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The model's own pixel-centre coordinates ``(yy, xx)``, each (H*W,).

        In pixel indices, or arcsec offsets from the frame centre when
        ``pixel_scale`` is set.
        """
        return self._yy, self._xx

    # ---------------------------------------------------------------- layout
    def split_theta(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Split theta into component blocks and shared raw values.

        Args:
            theta: Flat vector of shape (n_params,).

        Returns:
            ``(blocks, shared_raw)``.  ``blocks`` is a rectangular
            ``(K, n_shape + N_free)`` array when every component has the same
            number of shape parameters (in particular when they share a
            profile, the common case), and a **list of K 1-D blocks** of
            differing lengths otherwise.  ``shared_raw`` has shape (N_shared,).
        """
        shared_raw = theta[self.shared_slice]
        if self._rectangular:
            blocks = theta[: self._n_block_total].reshape(self.n_components, self._block)
        else:
            blocks = [theta[sl] for sl in self.component_slices]
        return blocks, shared_raw

    def join_theta(self, blocks: jnp.ndarray, shared_raw: jnp.ndarray) -> jnp.ndarray:
        """Inverse of :meth:`split_theta`.

        Args:
            blocks: Component blocks: a (K, n_shape + N_free) array, or a
                sequence of K 1-D blocks for mixed-profile models.
            shared_raw: Shared raw values of shape (N_shared,).

        Returns:
            Flat theta of shape (n_params,).
        """
        if isinstance(blocks, (list, tuple)):
            flat = jnp.concatenate([jnp.reshape(b, (-1,)) for b in blocks])
        else:
            flat = jnp.reshape(blocks, (-1,))
        return jnp.concatenate([flat, jnp.reshape(shared_raw, (-1,))])

    def shape_raw(self, theta: jnp.ndarray, k: int) -> jnp.ndarray:
        """Raw shape parameters of component ``k``, shape ``(n_shape_k,)``.

        Args:
            theta: Flat vector of shape (n_params,).
            k: Component index.

        Returns:
            1-D slice of theta.
        """
        return theta[self._shape_slices[k]]

    def sps_raw(self, theta: jnp.ndarray, k: int) -> jnp.ndarray:
        """Raw per-component SPS values of component ``k``, shape ``(N_free,)``.

        Args:
            theta: Flat vector of shape (n_params,).
            k: Component index.

        Returns:
            1-D slice of theta, in ``sps_param_names`` order.
        """
        return theta[self._sps_slices[k]]

    def set_sps_raw(self, theta: jnp.ndarray, k: int, values: jnp.ndarray) -> jnp.ndarray:
        """Return theta with component ``k``'s SPS raws replaced by ``values``.

        Args:
            theta: Flat vector of shape (n_params,).
            k: Component index.
            values: Replacement raws of shape (N_free,).

        Returns:
            Updated theta of shape (n_params,).
        """
        return theta.at[self._sps_slices[k]].set(values)

    # -------------------------------------------------------------- physical
    def component_shapes(self, theta: jnp.ndarray) -> list[dict[str, jnp.ndarray]]:
        """Physical shape parameters of every component.

        Args:
            theta: Flat vector of shape (n_params,).

        Returns:
            List of K dicts as returned by each component's
            ``Profile.parse`` — always ``mu`` (2,), ``sigma`` (2,) and ``rho``,
            plus ``n`` and ``b_n`` for Sérsic components.
        """
        return [
            profile.parse(theta[sl])
            for profile, sl in zip(self.profile_objects, self._shape_slices, strict=True)
        ]

    def _free_phys(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Physical per-component free SPS values, shape (K, N_free)."""
        raw = jnp.stack([theta[sl] for sl in self._sps_slices])  # (K, N_free)
        return self._free_lows + (self._free_highs - self._free_lows) * jax.nn.sigmoid(raw)

    def component_params(
        self, theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Map theta to physical component parameters.

        Args:
            theta: Flat vector of shape (n_params,).

        Returns:
            ``(mu, sigma, rho, sps_phys)``:

            - ``mu`` (K, 2): centres in the model's coordinate units.
            - ``sigma`` (K, 2): sizes (sigma_y, sigma_x).  Gaussian standard
              deviations, Sérsic effective radii, or the fixed ``point_sigma``
              of a point source.
            - ``rho`` (K,): correlation coefficients, clipped to ±0.99 (0 for
              point sources).
            - ``sps_phys`` (K, N_emulator): FULL physical emulator input rows in
              ``emulator_param_names`` order, with fixed and shared values filled in.

            Use :meth:`component_shapes` for the full per-profile dicts, which
            also carry the Sérsic index.
        """
        shapes = self.component_shapes(theta)
        mu = jnp.stack([s["mu"] for s in shapes])
        sigma = jnp.stack([s["sigma"] for s in shapes])
        rho = jnp.stack([jnp.reshape(s["rho"], ()) for s in shapes])
        free_phys = self._free_phys(theta)  # (K, N_free)
        shared_raw = theta[self.shared_slice]
        shared_phys = self._shared_lows + (self._shared_highs - self._shared_lows) * jax.nn.sigmoid(
            shared_raw
        )  # (N_shared,)
        K = self.n_components
        concat = jnp.concatenate(
            [
                free_phys,
                jnp.broadcast_to(shared_phys, (K, self._n_shared)),
                jnp.broadcast_to(self._fixed_vals, (K, self._n_fixed)),
            ],
            axis=1,
        )  # (K, N_emulator) in [free, shared, fixed] order
        sps_phys = concat[:, self._perm]  # emulator order
        return mu, sigma, rho, sps_phys

    # --------------------------------------------------------------- render
    def _log_render(
        self,
        theta: jnp.ndarray,
        yy: jnp.ndarray,
        xx: jnp.ndarray,
        pixel_area: float | jnp.ndarray,
        oversample: int | None,
    ) -> jnp.ndarray:
        """Log flux per pixel of every component on ``(yy, xx)``, shape (K, N)."""
        os = self.oversample if oversample is None else int(oversample)
        return jnp.stack(
            [
                log_render_on_grid(profile, theta[sl], yy, xx, pixel_area, os)
                for profile, sl in zip(self.profile_objects, self._shape_slices, strict=True)
            ]
        )

    def _log_profiles(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Log flux-fraction maps on the model's own grid, shape (K, H*W).

        With ``"frame"`` normalisation each row logsumexps to 0; with
        ``"analytic"`` each row logsumexps to ``<= 0``.
        """
        logs = self._log_render(theta, self._yy, self._xx, self.pixel_area, None)
        if self.normalisation == "frame":
            # Shift-invariant renormalisation: a component far outside the frame
            # still yields a finite, normalised profile instead of 0/0.
            logs = logs - jax.scipy.special.logsumexp(logs, axis=1, keepdims=True)
        return logs

    def profiles(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Per-pixel flux-fraction maps of every component, shape (K, H, W).

        With ``normalisation="frame"`` each component's map sums to exactly 1
        over the image; with ``"analytic"`` it sums to ``<= 1``, the deficit
        being the flux that falls outside the cutout.

        Args:
            theta: Flat vector of shape (n_params,).

        Returns:
            Array of shape (K, H, W).
        """
        H, W = self.image_shape
        return jnp.exp(self._log_profiles(theta)).reshape(self.n_components, H, W)

    def component_seds(self, theta: jnp.ndarray, emulator: SPSEmulator) -> jnp.ndarray:
        """Per-component SEDs at each component's own total mass, shape (K, N_bands).

        Exactly K emulator evaluations; exact for any emulator (no linearity
        in mass assumed).

        Args:
            theta: Flat vector of shape (n_params,).
            emulator: SPS emulator with ``predict((K, N_emulator)) -> (K, N_bands)``.

        Returns:
            Array of shape (K, N_bands) in nJy.
        """
        _, _, _, sps_phys = self.component_params(theta)
        return emulator.predict(sps_phys)

    def model_image_on(
        self,
        theta: jnp.ndarray,
        emulator: SPSEmulator,
        yy: jnp.ndarray,
        xx: jnp.ndarray,
        pixel_area: float | jnp.ndarray,
        oversample: int | None = None,
    ) -> jnp.ndarray:
        """Unconvolved model image on an arbitrary coordinate grid.

        This is the multi-resolution hook: pass another band's pixel-centre
        coordinates (in the model's units — arcsec offsets from the same
        reference point, obtained from that band's WCS) and its pixel area, and
        the same physical components are rendered natively on that grid with no
        resampling.

        Args:
            theta: Flat vector of shape (n_params,).
            emulator: SPS emulator.
            yy: Row coordinates of the target pixel centres, any shape.
            xx: Column coordinates, same shape as ``yy``.
            pixel_area: Area of one target pixel in the model's coordinate units
                squared.
            oversample: Override the model's ``oversample`` for this render.

        Returns:
            Array of shape ``(N_bands, *yy.shape)`` in nJy.

        Raises:
            ValueError: If ``normalisation="frame"`` and the grid is not the
                model's own (frame normalisation is defined only there; build
                the model with ``normalisation="analytic"`` for foreign grids).
        """
        own_grid = yy is self._yy and xx is self._xx
        if self.normalisation == "frame" and not own_grid:
            raise ValueError(
                "normalisation='frame' only makes sense on the model's own grid "
                "(a frame-normalised profile is not a physical surface brightness). "
                "Use normalisation='analytic' to render on other grids."
            )
        out_shape = jnp.shape(yy)
        logs = self._log_render(
            theta, jnp.ravel(yy), jnp.ravel(xx), pixel_area, oversample
        )  # (K, N)
        if self.normalisation == "frame":
            logs = logs - jax.scipy.special.logsumexp(logs, axis=1, keepdims=True)
        profiles = jnp.exp(logs)
        _, _, _, sps_phys = self.component_params(theta)
        seds = emulator.predict(sps_phys)  # (K, N_bands)
        image = jnp.einsum("kb,kn->bn", seds, profiles)
        return image.reshape(image.shape[0], *out_shape)

    def model_image(
        self, theta: jnp.ndarray, emulator: SPSEmulator, image_shape: tuple
    ) -> jnp.ndarray:
        """Unconvolved model image ``Σ_k F_k ⊗ P_k``, shape (N_bands, H, W).

        Args:
            theta: Flat vector of shape (n_params,).
            emulator: SPS emulator.
            image_shape: (H, W); must equal ``self.image_shape``.

        Returns:
            Array of shape (N_bands, H, W) in nJy.
        """
        H, W = self.image_shape
        flat = self.model_image_on(theta, emulator, self._yy, self._xx, self.pixel_area)
        return flat.reshape(flat.shape[0], H, W)

    def component_images(self, theta: jnp.ndarray, emulator: SPSEmulator) -> jnp.ndarray:
        """Unconvolved per-component images, shape (K, N_bands, H, W).

        Summing over the component axis reproduces :meth:`model_image`.

        Args:
            theta: Flat vector of shape (n_params,).
            emulator: SPS emulator.

        Returns:
            Array of shape (K, N_bands, H, W) in nJy.
        """
        profiles = self.profiles(theta)  # (K, H, W)
        seds = self.component_seds(theta, emulator)  # (K, N_bands)
        return jnp.einsum("kb,khw->kbhw", seds, profiles)

    # ----------------------------------------------------------- summary map
    def decode(self, theta: jnp.ndarray, image_shape: tuple) -> jnp.ndarray:
        """Per-pixel summary map of the per-component SPS parameters, (H*W, N_free).

        **This is a summary for plotting and ``NUTSResult.get_parameter_map``;
        the likelihood never uses it** (see :meth:`model_image`).

        Column ``c`` (for ``c != mass_index``) is the stellar-mass-weighted mean
        of the components' per-component parameter ``c``::

            w_k(y, x) = P_k(y, x) 10**logM_k / Σ_j P_j(y, x) 10**logM_j
            map_c(y, x) = Σ_k w_k(y, x) sps_kc

        The mass column is replaced by log10 of the stellar-mass surface density
        per pixel, ``log10(Σ_k P_k(y, x) 10**logM_k)`` in Msun per pixel.

        Args:
            theta: Flat vector of shape (n_params,).
            image_shape: (H, W); must equal ``self.image_shape``.

        Returns:
            Array of shape (H*W, N_free), columns in ``sps_param_names`` order.
        """
        free_phys = self._free_phys(theta)  # (K, N_free)
        log_profiles = self._log_profiles(theta)  # (K, H*W)
        # log of stellar-mass surface density of each component at each pixel
        log_density = log_profiles + (free_phys[:, self.mass_index] * _LN10)[:, None]
        # Mass weights: softmax over components is exact even where every
        # component has underflowed, so the summary map never leaves bounds.
        weights = jax.nn.softmax(log_density, axis=0)  # (K, H*W)
        summary = jnp.einsum("kp,kc->pc", weights, free_phys)  # (H*W, N_free)
        log10_total = jax.scipy.special.logsumexp(log_density, axis=0) / _LN10  # (H*W,)
        return summary.at[:, self.mass_index].set(log10_total)

    # ------------------------------------------------------------------ prior
    def _jacobian_term(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Sigmoid log-Jacobian summed over per-component and shared SPS raws."""
        free_raw = jnp.concatenate([theta[sl] for sl in self._sps_slices])
        shared_raw = theta[self.shared_slice]
        # log_sigmoid(r) + log_sigmoid(-r) is the exact log-density of
        # Uniform(lo, hi) pulled back through phys = lo + (hi-lo)*sigmoid(r):
        # the 1/(hi-lo) of the uniform cancels the (hi-lo) of dphys/dr.  No
        # log(hi-lo) term, so the prior is properly normalised and logZ values
        # from nested sampling are comparable across models with different K.
        jac_free = jnp.sum(jax.nn.log_sigmoid(free_raw) + jax.nn.log_sigmoid(-free_raw))
        jac_shared = jnp.sum(jax.nn.log_sigmoid(shared_raw) + jax.nn.log_sigmoid(-shared_raw))
        return jac_free + jac_shared

    def _shape_log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Sum of the per-component profile shape log-priors."""
        terms = [
            profile.log_prior(
                theta[sl],
                self.centre,
                self.centre_prior_sigma,
                self.log_size_prior,
                self.rho_prior_sigma,
            )
            for profile, sl in zip(self.profile_objects, self._shape_slices, strict=True)
        ]
        return sum(terms[1:], terms[0])

    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Scalar log-prior of theta (see class docstring for the terms).

        Args:
            theta: Flat vector of shape (n_params,).

        Returns:
            Scalar log-prior, ``jax.grad``-differentiable.
        """
        lp = self._jacobian_term(theta) + self._shape_log_prior(theta)
        if self.sps_log_prior is not None:
            _, _, _, sps_phys = self.component_params(theta)
            lp = lp + self.sps_log_prior(sps_phys)
        return lp

    def sample_prior(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw ``n`` theta vectors from the prior defined by :meth:`log_prior`.

        SPS values are uniform within bounds (mapped to raw via logit); shape
        parameters come from each component's ``Profile.sample_prior``.  The
        optional ``sps_log_prior`` term is *not* sampled from.

        Args:
            key: ``jax.random`` PRNG key.
            n: Number of samples.

        Returns:
            Array of shape (n, n_params).
        """
        K = self.n_components
        keys = jax.random.split(key, K + 2)
        f32 = jnp.float32
        # Uniform in physical space == logistic in raw space
        free_raw = jax.random.logistic(keys[0], (n, K, self._n_free), dtype=f32)
        shared_raw = jax.random.logistic(keys[1], (n, self._n_shared), dtype=f32)

        columns: list[jnp.ndarray] = []
        for k, profile in enumerate(self.profile_objects):
            columns.append(
                profile.sample_prior(
                    keys[2 + k],
                    self.centre,
                    self.centre_prior_sigma,
                    self.log_size_prior,
                    self.rho_prior_sigma,
                    batch_shape=(n,),
                )
            )
            columns.append(free_raw[:, k, :])
        columns.append(shared_raw)
        return jnp.concatenate(columns, axis=1).astype(f32)

    # -------------------------------------------------------------- utilities
    def order_components_by_size(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Permute component blocks so each profile's components are compact-first.

        Components are sorted on ``Profile.size_statistic`` (the log
        geometric-mean size).  Blocks of different profiles have different
        lengths and different meanings, so **only components that share a
        profile are permuted among themselves**: each profile's set of block
        positions is reordered internally and the positions themselves are
        untouched.  For the usual single-profile model this is a full sort.

        Pure JAX; used to break label symmetry when reporting.

        Args:
            theta: Flat spatial vector of shape (n_params,), or a full forward-model
                vector of shape (n_params + n_nuisance,); any trailing nuisance block
                is passed through untouched.

        Returns:
            Reordered theta with the same shape as the input.

        Raises:
            ValueError: If ``theta`` is shorter than ``n_params``.
        """
        if theta.shape[0] < self.n_params:
            raise ValueError(
                f"theta has length {theta.shape[0]} but the model has {self.n_params} parameters."
            )
        tail = theta[self.n_params :]
        theta = theta[: self.n_params]
        blocks = [theta[sl] for sl in self.component_slices]
        groups: dict[str, list[int]] = {}
        for k, profile in enumerate(self.profile_objects):
            groups.setdefault(f"{type(profile).__name__}:{profile.n_shape}", []).append(k)
        new_blocks = list(blocks)
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            stacked = jnp.stack([blocks[k] for k in idxs])  # (m, block)
            sizes = jnp.stack(
                [self.profile_objects[k].size_statistic(theta[self._shape_slices[k]]) for k in idxs]
            )
            permuted = jnp.take(stacked, jnp.argsort(sizes), axis=0)
            for j, k in enumerate(idxs):
                new_blocks[k] = permuted[j]
        ordered = self.join_theta(new_blocks, theta[self.shared_slice])
        return jnp.concatenate([ordered, tail]) if tail.shape[0] else ordered


def _resolve_profiles(
    profiles: str | Profile | Sequence[str | Profile],
    n_components: int,
    pixel_scale: float | None = None,
) -> tuple[Profile, ...]:
    """Expand the ``profiles`` constructor argument into K Profile instances.

    A profile given by *name* is built with defaults expressed in the model's
    coordinate units: in arcsec mode the ``"point"`` default width becomes
    ``0.5 * pixel_scale`` arcsec, i.e. the same half-pixel kernel as in pixel
    mode.  A pre-built :class:`Profile` instance is used exactly as given, so
    an explicit ``PointSourceProfile(point_sigma=...)`` always wins.
    """

    def _one(spec: str | Profile) -> Profile:
        if spec == "point" and pixel_scale is not None:
            return PointSourceProfile(point_sigma=0.5 * pixel_scale)
        return get_profile(spec)

    if isinstance(profiles, (str, Profile)):
        return (_one(profiles),) * n_components
    seq = list(profiles)
    if len(seq) != n_components:
        raise ValueError(
            f"profiles must be a single profile or a sequence of length "
            f"n_components={n_components}, got {len(seq)}"
        )
    return tuple(_one(p) for p in seq)
