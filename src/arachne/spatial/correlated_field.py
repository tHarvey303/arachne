"""Correlated-field spatial model: SPS parameter maps with Gaussian-random-field priors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from arachne.emulator.base import SPSEmulator
from arachne.spatial.base import SpatialModel

_LOG_2PI = math.log(2.0 * math.pi)


def _default_float():
    """JAX's default float dtype (float64 iff ``jax_enable_x64``)."""
    return jnp.zeros(()).dtype


def matern_spectrum(k: np.ndarray, corr_length: float, nu: float) -> np.ndarray:
    """Un-normalised isotropic 2-D Matérn power spectrum ``[1 + (k ell)^2]^-(nu + 1)``.

    Args:
        k: Angular wavenumber in rad/pixel (any shape).
        corr_length: Break scale ``ell`` in pixels.  Above it the spectrum is a
            power law of slope ``-2 (nu + 1)``; below it the spectrum is flat.
        nu: Smoothness.  ``nu = 1.5`` is the classic Matérn-3/2 (slope -5);
            small ``nu`` gives a shallower spectrum with more small-scale power.

    Returns:
        Array of the same shape as ``k``.
    """
    return (1.0 + (np.asarray(k) * corr_length) ** 2) ** (-(nu + 1.0))


def _softclip(u: jnp.ndarray, lo: float, hi: float, sharpness: float) -> jnp.ndarray:
    """Smooth clamp into ``[lo, hi]`` that is ~identity well inside the interval."""
    b = sharpness / (hi - lo)
    return lo + jax.nn.softplus(b * (u - lo)) / b - jax.nn.softplus(b * (u - hi)) / b


class _FieldSpec:
    """Static description of one mapped parameter (bounds, spectrum, Fourier modes)."""

    def __init__(
        self,
        name: str,
        bounds: tuple[float, float] | None,
        amplitude: float,
        corr_length: float,
        nu: float,
        level_sigma: float,
        coupling: float,
    ) -> None:
        self.name = name
        self.bounds = bounds
        self.amplitude = float(amplitude)
        self.corr_length = float(corr_length)
        self.nu = float(nu)
        self.level_sigma = float(level_sigma)
        self.coupling = float(coupling)
        # Filled in by CorrelatedFieldMap._build_modes
        self.iy: np.ndarray | None = None
        self.ix: np.ndarray | None = None
        self.amp: np.ndarray | None = None
        self.n_modes = 0
        self.variance_kept = 1.0


class CorrelatedFieldMap(SpatialModel):
    """Per-pixel SPS parameter maps, each a Gaussian random field around a mean map.

    Every emulator parameter plays one of three roles:

    - **field**: a map ``p_k(x)`` with a Gaussian-random-field (GP) prior.  The
      mass parameter must be a field (``log_mass`` per *pixel*, unbounded:
      emulators with an analytic mass factorisation are exact at any mass).
    - **global**: one free scalar shared by every pixel, uniform within its
      bounds (sigmoid-mapped, with the Jacobian in the prior).
    - **fixed**: a constant, e.g. the spectroscopic redshift.

    Field model
    -----------
    For field ``k`` with mean map ``m_k(x)``, free level offset ``a_k`` and white
    coefficients ``z_k``::

        u_k(x) = m_k(x) + a_k + c_k * [mass(x) - m_mass(x) - a_mass] + s_k * g_k(x)
        g_k    = IFFT[ sqrt(P_k) * Z(z_k) ]      (unit-variance stationary field)
        p_k(x) = softclip(u_k(x), bounds_k)      (identity for the mass field)

    ``P_k`` is a Matérn spectrum with break ``corr_length`` (pixels) and slope
    set by ``nu``.  ``c_k`` (``couplings``) propagates local mass fluctuations
    into the other maps, e.g. a resolved Σ*–A_V relation; the *smooth* part of
    any relation belongs in ``mean_maps``.  The field lives on a zero-padded
    grid so that it does not wrap around the cutout.

    Only Fourier modes that carry most of the prior variance are kept
    (``variance_fraction``, or an explicit ``k_max``), so a map costs a few
    hundred parameters rather than ``H * W``.  The DC mode is excluded (the free
    level ``a_k`` replaces it) and so are the self-conjugate Nyquist modes.

    Theta layout::

        theta = [
            levels(
                N_fields
            ),
            global_raw(
                N_global
            ),
            z_field_0,
            z_field_1,
            ...,
        ]

    The prior is proper and normalised: ``z ~ N(0, I)``, ``a_k ~ N(0,
    level_sigma_k^2)``, globals uniform in physical space.  ``sample_prior`` draws
    from exactly that density.
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        emulator_param_names: Sequence[str],
        param_bounds: Mapping[str, tuple[float, float]],
        field_params: Sequence[str],
        mass_param: str = "log_mass",
        fixed_params: Mapping[str, float] | None = None,
        corr_lengths: Mapping[str, float] | float = 4.0,
        amplitudes: Mapping[str, float] | float = 0.3,
        nus: Mapping[str, float] | float = 1.5,
        level_sigmas: Mapping[str, float] | None = None,
        couplings: Mapping[str, float] | None = None,
        mean_maps: Mapping[str, np.ndarray | float] | None = None,
        variance_fraction: float = 0.99,
        k_max: float | None = None,
        pad: int | None = None,
        pixel_scale: float | None = None,
        softclip_sharpness: float = 20.0,
    ) -> None:
        """Initialise the model.

        Args:
            image_shape: ``(H, W)`` of the observation.
            emulator_param_names: The emulator's full input vector, in order.
            param_bounds: ``(lo, hi)`` per non-mass parameter (the emulator's
                training domain).  Mapped parameters are soft-clipped into it;
                globals are uniform within it.
            field_params: Parameters that get maps; must include ``mass_param``.
            mass_param: Name of the per-pixel log10 stellar mass input.
            fixed_params: Constant values (e.g. ``{"redshift": 2.0}``).
            corr_lengths: Spectrum break scale in pixels, per field or one for all.
            amplitudes: Field rms ``s_k`` in parameter units (dex for log
                quantities, mag for ``Av``), per field or one for all.
            nus: Matérn smoothness per field or one for all.
            level_sigmas: Prior sigma of the free level offset per field.
                Default 1.0 for the mass field and a quarter of the bounds width
                otherwise.
            couplings: ``c_k`` per non-mass field: response of ``p_k`` to the
                local mass-map deviation (parameter units per dex).
            mean_maps: Mean map per field, an ``(H, W)`` array or a scalar.
                Default: the centre of the bounds (non-mass) or 0 (mass); set
                the mass mean from the data with :func:`mass_map_from_image`.
            variance_fraction: Keep the highest-power modes that together hold
                this fraction of each field's prior variance.
            k_max: Alternatively keep every mode with ``|k| <= k_max`` (rad/px).
            pad: Zero-padding in pixels on each side.  Default
                ``min(max(H, W), ceil(2 * max corr_length))``, at least 4.
            pixel_scale: Arcsec per pixel (stored for bookkeeping only).
            softclip_sharpness: Sharpness of the soft clamp, in units of
                ``1 / bounds width``; larger is closer to a hard clip.

        Raises:
            ValueError: On an unknown or doubly assigned parameter, a missing
                bound, or a mass parameter that is not a field.
        """
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.emulator_param_names = list(emulator_param_names)
        self.param_bounds = dict(param_bounds)
        self.mass_param = mass_param
        self.fixed_params = dict(fixed_params or {})
        self.pixel_scale = pixel_scale
        self.softclip_sharpness = float(softclip_sharpness)

        names = self.emulator_param_names
        field_params = list(field_params)
        if mass_param not in field_params:
            raise ValueError(f"mass_param {mass_param!r} must be one of field_params")
        for p in list(field_params) + list(self.fixed_params):
            if p not in names:
                raise ValueError(f"{p!r} is not an emulator parameter ({names})")
        overlap = set(field_params) & set(self.fixed_params)
        if overlap:
            raise ValueError(f"parameters both mapped and fixed: {sorted(overlap)}")
        # Mass field first, so couplings can refer to it.
        self.field_names = [mass_param] + [p for p in field_params if p != mass_param]
        self.global_names = [
            p for p in names if p not in self.field_names and p not in self.fixed_params
        ]
        for p in self.field_names[1:] + self.global_names:
            if p not in self.param_bounds:
                raise ValueError(f"no bounds given for {p!r}")

        def per_field(value, name, default=None):
            if isinstance(value, Mapping):
                return value.get(name, default)
            return value

        level_sigmas = level_sigmas or {}
        couplings = couplings or {}
        self.fields: list[_FieldSpec] = []
        for p in self.field_names:
            is_mass = p == mass_param
            bounds = None if is_mass else tuple(map(float, self.param_bounds[p]))
            default_ls = 1.0 if is_mass else 0.25 * (bounds[1] - bounds[0])
            self.fields.append(
                _FieldSpec(
                    name=p,
                    bounds=bounds,
                    amplitude=per_field(amplitudes, p, 0.3),
                    corr_length=per_field(corr_lengths, p, 4.0),
                    nu=per_field(nus, p, 1.5),
                    level_sigma=level_sigmas.get(p, default_ls),
                    coupling=0.0 if is_mass else couplings.get(p, 0.0),
                )
            )

        H, W = self.image_shape
        max_ell = max(f.corr_length for f in self.fields)
        if pad is None:
            pad = min(max(H, W), max(4, int(math.ceil(2.0 * max_ell))))
        self.pad = int(pad)
        self.grid_shape = (H + 2 * self.pad, W + 2 * self.pad)
        self.variance_fraction = float(variance_fraction)
        self.k_max = k_max
        self._build_modes()

        # Mean maps (H, W), float32
        mean_maps = dict(mean_maps or {})
        means = []
        for f in self.fields:
            if f.name in mean_maps:
                m = np.broadcast_to(np.asarray(mean_maps[f.name], np.float32), (H, W))
            elif f.bounds is None:
                m = np.zeros((H, W), np.float32)
            else:
                m = np.full((H, W), 0.5 * (f.bounds[0] + f.bounds[1]), np.float32)
            means.append(np.array(m, np.float32))
        self._mean_maps = jnp.asarray(np.stack(means))  # (N_fields, H, W)

        self._global_lo = jnp.asarray([self.param_bounds[p][0] for p in self.global_names])
        self._global_hi = jnp.asarray([self.param_bounds[p][1] for p in self.global_names])
        self._level_sigma = jnp.asarray([f.level_sigma for f in self.fields])
        self._n_params = len(self.fields) + len(self.global_names) + self.n_modes_total

    # ------------------------------------------------------------------ modes
    def _build_modes(self) -> None:
        """Select the retained half-plane Fourier modes and their amplitudes per field."""
        Hp, Wp = self.grid_shape
        n_pix = Hp * Wp
        fy = np.fft.fftfreq(Hp)
        fx = np.fft.rfftfreq(Wp)
        iy_all, ix_all = np.meshgrid(np.arange(Hp), np.arange(len(fx)), indexing="ij")
        k_all = 2.0 * np.pi * np.hypot(fy[iy_all], fx[ix_all])
        # Non-self-conjugate half plane: kx in (0, Nyquist) for any ky, plus kx = 0 with
        # 0 < ky < Nyquist.  Each mode stands for itself and its conjugate partner.
        nyq_x = (Wp % 2 == 0) * (Wp // 2)
        interior = (ix_all > 0) & ((ix_all != nyq_x) | (nyq_x == 0))
        axis = (ix_all == 0) & (iy_all > 0) & (iy_all < (Hp + 1) // 2)
        cand = interior | axis
        iy_c, ix_c, k_c = iy_all[cand], ix_all[cand], k_all[cand]

        # Full-grid (non-DC) wavenumbers, for normalising each spectrum to unit variance.
        k_full = 2.0 * np.pi * np.hypot(*np.meshgrid(fy, np.fft.fftfreq(Wp), indexing="ij"))
        nondc = k_full > 0

        start = 0
        for f in self.fields:
            p_full = matern_spectrum(k_full, f.corr_length, f.nu)
            norm = p_full[nondc].mean()  # mean over N-1 modes ≈ unit pixel variance
            p_c = matern_spectrum(k_c, f.corr_length, f.nu) / norm
            # Each half-plane mode holds 2 p / N of the variance (itself + conjugate).
            share = 2.0 * p_c / n_pix
            if self.k_max is not None:
                keep = k_c <= self.k_max
            else:
                order = np.argsort(-p_c, kind="stable")
                cum = np.cumsum(share[order]) / share.sum()
                n_keep = int(np.searchsorted(cum, self.variance_fraction) + 1)
                keep = np.zeros_like(p_c, bool)
                keep[order[:n_keep]] = True
            f.iy = iy_c[keep].astype(np.int32)
            f.ix = ix_c[keep].astype(np.int32)
            # Complex white noise Z has Re/Im variance N/2 for a unit-variance real field.
            f.amp = np.sqrt(p_c[keep] * n_pix / 2.0)  # float64; cast at use
            f.n_modes = int(keep.sum())
            f.variance_kept = float(share[keep].sum() / share.sum())
            f.slice = slice(start, start + 2 * f.n_modes)
            start += 2 * f.n_modes
        self.n_modes_total = start

    # ------------------------------------------------------------------ layout
    @property
    def n_params(self) -> int:
        """Total free parameters: levels + globals + 2 per retained mode per field."""
        return self._n_params

    @property
    def n_fields(self) -> int:
        """Number of mapped parameters."""
        return len(self.fields)

    @property
    def n_global(self) -> int:
        """Number of free global scalars."""
        return len(self.global_names)

    def split_theta(self, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Split theta into ``(levels (N_fields,), global_raw (N_global,), z (n_modes_total,))``."""
        nf, ng = self.n_fields, self.n_global
        return theta[:nf], theta[nf : nf + ng], theta[nf + ng :]

    def join_theta(self, levels, global_raw, z) -> jnp.ndarray:
        """Inverse of :meth:`split_theta`."""
        return jnp.concatenate(
            [jnp.atleast_1d(levels), jnp.atleast_1d(global_raw), jnp.atleast_1d(z)]
        ).astype(_default_float())

    def initial_theta(self, global_values: Mapping[str, float] | None = None) -> jnp.ndarray:
        """Theta at the prior mean: zero levels and coefficients, globals mid-range or given."""
        raw = []
        for i, p in enumerate(self.global_names):
            lo, hi = float(self._global_lo[i]), float(self._global_hi[i])
            v = (global_values or {}).get(p, 0.5 * (lo + hi))
            frac = min(max((v - lo) / (hi - lo), 1e-4), 1 - 1e-4)
            raw.append(math.log(frac / (1.0 - frac)))
        return self.join_theta(
            jnp.zeros(self.n_fields), jnp.asarray(raw), jnp.zeros(self.n_modes_total)
        )

    # ------------------------------------------------------------------ fields
    def unit_field(self, z_k: jnp.ndarray, k: int) -> jnp.ndarray:
        """Unit-variance random field ``g_k`` (H, W) from field ``k``'s coefficients."""
        f = self.fields[k]
        Hp, Wp = self.grid_shape
        re, im = z_k[: f.n_modes], z_k[f.n_modes :]
        cdtype = jnp.complex128 if z_k.dtype == jnp.float64 else jnp.complex64
        c = (re + 1j * im).astype(cdtype) * jnp.asarray(f.amp, z_k.dtype)
        spec = jnp.zeros((Hp, Wp // 2 + 1), cdtype)
        spec = spec.at[f.iy, f.ix].set(c, unique_indices=True)  # each mode once
        # kx = 0 modes need their conjugate partner placed explicitly.
        on_axis = f.ix == 0
        if np.any(on_axis):
            iy_ax = np.asarray(f.iy[on_axis])
            spec = spec.at[(-iy_ax) % Hp, 0].set(
                jnp.conj(c[np.flatnonzero(on_axis)]), unique_indices=True
            )
        g = jnp.fft.irfft2(spec, s=(Hp, Wp))
        P = self.pad
        H, W = self.image_shape
        return g[P : P + H, P : P + W]

    def project_to_field(
        self,
        theta: jnp.ndarray,
        field: str,
        target_map: np.ndarray,
        ridge: float = 1.0,
        maxiter: int = 500,
    ) -> jnp.ndarray:
        """Return ``theta`` with one field's coefficients set to best reproduce ``target_map``.

        Solves the ridge regression ``min_z |s g(z) - r|^2 / s^2 + ridge |z|^2`` by
        conjugate gradients, where ``r`` is the target minus everything else in that
        field's pre-clip value (mean map, level, coupling).  With ``ridge = 1`` this
        is the posterior mean of ``z`` given ``r`` observed with unit noise in units
        of ``s``, i.e. it respects the field prior.  Use it to start a fit from a
        sharp guess (a deconvolved image, an IFU map) **without** putting that
        detail into the mean map, where the prior would not penalise it.

        For bounded fields ``target_map`` is in physical units and is treated as
        the pre-clip value (the soft clamp is ~identity inside the bounds).

        Args:
            theta: Current theta (levels, globals and other fields are kept).
            field: Field name.
            target_map: ``(H, W)`` target.
            ridge: Prior weight on the coefficients (1 = the model prior).
            maxiter: CG iterations.

        Returns:
            Updated theta.
        """
        k = self.field_names.index(field)
        f = self.fields[k]
        n0 = self.n_fields + self.n_global
        z_slice = slice(n0 + f.slice.start, n0 + f.slice.stop)
        theta0 = theta.at[z_slice].set(0.0)
        current = self.field_maps(theta0)[k]
        if f.bounds is not None:
            # Pre-clip value without the field term (softclip ~ identity inside bounds).
            current = jnp.clip(current, f.bounds[0], f.bounds[1])
        r = (jnp.asarray(target_map, current.dtype) - current) / f.amplitude
        z0 = jnp.zeros(2 * f.n_modes, current.dtype)
        g = lambda z: self.unit_field(z, k)  # noqa: E731
        gt = jax.linear_transpose(g, z0)

        def normal_op(z):
            return gt(g(z))[0] + ridge * z

        rhs = gt(r)[0]
        z, _ = jax.scipy.sparse.linalg.cg(normal_op, rhs, x0=z0, maxiter=maxiter)
        return theta.at[z_slice].set(z)

    def field_maps(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Physical parameter maps ``(N_fields, H, W)`` in ``field_names`` order."""
        levels, _, z = self.split_theta(theta)
        mass_dev = None
        maps = []
        for k, f in enumerate(self.fields):
            g = self.unit_field(z[f.slice], k)
            dev = levels[k] + f.amplitude * g
            if k == 0:
                mass_dev = dev
                u = self._mean_maps[0] + dev
                maps.append(u)
                continue
            u = self._mean_maps[k] + dev + f.coupling * mass_dev
            maps.append(_softclip(u, f.bounds[0], f.bounds[1], self.softclip_sharpness))
        return jnp.stack(maps)

    def global_values(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Physical values of the global scalars, shape (N_global,)."""
        _, raw, _ = self.split_theta(theta)
        return self._global_lo + (self._global_hi - self._global_lo) * jax.nn.sigmoid(raw)

    def decode(self, theta: jnp.ndarray, image_shape: tuple | None = None) -> jnp.ndarray:
        """Per-pixel emulator input rows ``(H*W, N_emulator)`` in emulator order."""
        H, W = self.image_shape
        maps = self.field_maps(theta).reshape(self.n_fields, H * W)
        gvals = self.global_values(theta)
        cols = []
        for p in self.emulator_param_names:
            if p in self.field_names:
                cols.append(maps[self.field_names.index(p)])
            elif p in self.fixed_params:
                cols.append(jnp.full((H * W,), self.fixed_params[p], maps.dtype))
            else:
                cols.append(jnp.broadcast_to(gvals[self.global_names.index(p)], (H * W,)))
        return jnp.stack(cols, axis=1)

    def model_image(self, theta: jnp.ndarray, emulator: SPSEmulator, image_shape: tuple):
        """Unconvolved image ``(N_bands, H, W)``: one emulator call per pixel."""
        H, W = self.image_shape
        flux = emulator.predict(self.decode(theta))  # (H*W, N_bands)
        return flux.T.reshape(flux.shape[1], H, W)

    # ------------------------------------------------------------------ prior
    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Normalised log prior: N(0, I) coefficients, Gaussian levels, uniform globals."""
        levels, raw, z = self.split_theta(theta)
        lp = -0.5 * jnp.sum(z**2) - 0.5 * z.size * _LOG_2PI
        lp += jnp.sum(
            -0.5 * (levels / self._level_sigma) ** 2 - jnp.log(self._level_sigma) - 0.5 * _LOG_2PI
        )
        # Uniform in physical space: density of raw is sigmoid'(raw) (bounds width cancels).
        lp += jnp.sum(jax.nn.log_sigmoid(raw) + jax.nn.log_sigmoid(-raw))
        return lp

    def sample_prior(self, key: jax.Array, n: int) -> jnp.ndarray:
        """Draw ``n`` theta vectors from exactly the density of :meth:`log_prior`."""
        k1, k2, k3 = jax.random.split(key, 3)
        levels = jax.random.normal(k1, (n, self.n_fields)) * self._level_sigma
        raw = jax.random.logistic(k2, (n, self.n_global))
        z = jax.random.normal(k3, (n, self.n_modes_total))
        return jnp.concatenate([levels, raw, z], axis=1).astype(_default_float())

    def summary(self) -> str:
        """Human-readable description of the fields, modes and parameter count."""
        lines = [
            f"CorrelatedFieldMap {self.image_shape[0]}x{self.image_shape[1]} "
            f"(padded {self.grid_shape[0]}x{self.grid_shape[1]}), {self.n_params} parameters"
        ]
        for f in self.fields:
            lines.append(
                f"  field {f.name:18s} ell={f.corr_length:5.1f}px nu={f.nu:.2f} "
                f"s={f.amplitude:.3g} modes={f.n_modes} (x2 real, "
                f"{100 * f.variance_kept:.1f}% of prior variance) coupling={f.coupling:g}"
            )
        lines.append(f"  globals: {self.global_names or 'none'}; fixed: {self.fixed_params}")
        return "\n".join(lines)


def mass_map_from_image(
    obs,
    emulator: SPSEmulator,
    reference_params: Mapping[str, float],
    band: str | int | None = None,
    mass_param: str = "log_mass",
    smooth_sigma_px: float = 1.0,
    floor_frac: float = 1e-3,
) -> np.ndarray:
    """Log10 stellar mass per pixel implied by one band's image at reference SPS values.

    The emulator is linear in mass, so the flux per unit mass at
    ``reference_params`` converts a (lightly smoothed) image straight into a
    mass map.  Pixels below ``floor_frac`` of the peak are floored there.  This
    is the cheap "intelligent start" for the mass field's mean map; it ignores
    the PSF (a deconvolved solve is a planned refinement).

    Args:
        obs: ``ObservationCube`` (numpy or JAX arrays).
        emulator: Mass-linear SPS emulator.
        reference_params: Values for every emulator input except the mass.
        band: Band name or index; default the reddest (last) band.
        mass_param: Name of the mass input.
        smooth_sigma_px: Gaussian smoothing before the conversion (0 disables).
        floor_frac: Floor as a fraction of the smoothed peak flux.

    Returns:
        ``(H, W)`` float32 array of log10 mass per pixel.
    """
    names = list(emulator.param_names)
    b = (
        len(obs.band_names) - 1
        if band is None
        else (obs.band_names.index(band) if isinstance(band, str) else int(band))
    )
    row = np.array(
        [0.0 if p == mass_param else float(reference_params[p]) for p in names], np.float32
    )
    flux_per_msun = float(np.asarray(emulator.predict(jnp.asarray(row[None, :])))[0, b])
    img = np.asarray(obs.flux[b], np.float64)
    if smooth_sigma_px > 0:
        H, W = img.shape
        ky = np.fft.fftfreq(H)[:, None]
        kx = np.fft.fftfreq(W)[None, :]
        kernel = np.exp(-2.0 * (np.pi * smooth_sigma_px) ** 2 * (ky**2 + kx**2))
        img = np.real(np.fft.ifft2(np.fft.fft2(img) * kernel))
    floor = floor_frac * img.max()
    return np.log10(np.maximum(img, floor) / flux_per_msun).astype(np.float32)


def wiener_mass_map(
    obs,
    psf_model,
    emulator: SPSEmulator,
    reference_params: Mapping[str, float],
    mass_param: str = "log_mass",
    regularisation: float = 1e-2,
    floor_frac: float = 1e-3,
) -> np.ndarray:
    """Log10 mass per pixel from a joint multi-band Wiener deconvolution.

    At fixed reference SPS values every band is linear in the mass map ``M``::

        D_b = f_b * (P_b ⊛ M) + noise,   f_b = emulator flux per Msun in band b

    so the least-squares ``M`` is a per-wavenumber weighted sum over bands,
    regularised by ``regularisation`` times the largest data weight (a white
    Tikhonov term; larger values smooth more)::

        M(k) = Σ_b w_b f_b P_b*(k) D_b(k) / (Σ_b w_b f_b² |P_b(k)|² + λ)

    with ``w_b = 1 / median variance_b``.  The image is zero-padded by the
    kernel size so the deconvolution does not wrap.  Negative or tiny values
    are floored at ``floor_frac`` of the peak before the log.  This sharpens the
    start compared with :func:`mass_map_from_image`, which ignores the PSF.

    Args:
        obs: ``ObservationCube``.
        psf_model: ``PSFModel`` on the observation's pixel grid, bands in obs order.
        emulator: Mass-linear SPS emulator whose bands match ``obs.band_names``.
        reference_params: Values for every emulator input except the mass.
        mass_param: Name of the mass input.
        regularisation: Tikhonov strength relative to the peak data weight.
        floor_frac: Floor as a fraction of the peak deconvolved mass.

    Returns:
        ``(H, W)`` float32 array of log10 mass per pixel.
    """
    names = list(emulator.param_names)
    row = np.array(
        [0.0 if p == mass_param else float(reference_params[p]) for p in names], np.float32
    )
    f = np.asarray(emulator.predict(jnp.asarray(row[None, :])), np.float64)[0]  # (B,)
    flux = np.asarray(obs.flux, np.float64)
    var = np.asarray(obs.variance, np.float64)
    mask = np.asarray(obs.mask, np.float64)
    B, H, W = flux.shape
    kern = np.asarray(psf_model.kernels, np.float64)
    kh, kw = kern.shape[1:]
    Hp, Wp = H + kh, W + kw
    num = np.zeros((Hp, Wp), complex)
    den = np.zeros((Hp, Wp))
    for b in range(B):
        w_b = 1.0 / np.median(var[b][mask[b] > 0])
        k_img = np.zeros((Hp, Wp))
        k_img[:kh, :kw] = kern[b] / kern[b].sum()
        k_img = np.roll(k_img, (-(kh // 2), -(kw // 2)), axis=(0, 1))  # centre at (0, 0)
        pk = np.fft.fft2(k_img)
        d = np.zeros((Hp, Wp))
        d[:H, :W] = flux[b] * mask[b]
        num += w_b * f[b] * np.conj(pk) * np.fft.fft2(d)
        den += w_b * f[b] ** 2 * np.abs(pk) ** 2
    lam = regularisation * den.max()
    m = np.real(np.fft.ifft2(num / (den + lam)))[:H, :W]
    floor = floor_frac * m.max()
    return np.log10(np.maximum(m, floor)).astype(np.float32)
