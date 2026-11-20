"""Redshift-grid starting points for integrated-photometry (catalogue) fits.

A photometric-redshift posterior is multimodal in redshift (break confusions), and
a gradient optimiser or a variational fit started from one generic point settles in
whichever mode is nearest.  Nested sampling handles that but cannot be batched.
This module finds the candidate modes cheaply, before any optimiser runs:

1. evaluate the emulator once on a grid of redshifts times a small set of SED
   templates (dust, SFH shape, metallicity), at a reference mass;
2. for every galaxy and grid point solve the stellar mass analytically (the
   emulator's flux is exactly linear in mass) and compute chi-squared;
3. keep, per galaxy, the ``k`` lowest local minima of the chi-squared profile in
   redshift that are separated by ``min_separation`` in ln(1 + z).

The result is ``k`` starting points per galaxy (redshift, template, matched mass),
ordered by chi-squared, for multi-path Pathfinder or multi-chain sampling.  The
cost is one emulator call of ``n_templates * n_z`` rows per catalogue plus a few
array operations per batch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

__all__ = ["RedshiftGrid", "default_templates", "redshift_grid"]

_REF_LOG_MASS = 8.0


def default_templates(
    spec,
    av: Sequence[float] = (0.05, 0.4, 1.0, 2.0),
    sfh_slope: Sequence[float] = (0.0, 0.7, -0.7),
    log_metallicity: Sequence[float] = (-3.0, -2.0),
) -> list[dict[str, float]]:
    """SED templates spanning dust, SFH shape and metallicity.

    Every emulator input not varied here is held at ``spec.defaults``.  ``sfh_slope``
    is the value given to every SFH log-ratio at once: positive is a rising, negative
    a declining star-formation history.  Values are clipped into the emulator's
    domain; a quantity the emulator does not have is skipped.

    Args:
        spec: The emulator's :class:`~arachne.emulator.spec.EmulatorSpec`.
        av: A_V values [mag].
        sfh_slope: Common value of the SFH log-ratios.
        log_metallicity: log10 metallicity values.

    Returns:
        List of ``{parameter: value}`` dicts covering every emulator input except
        the redshift and the mass.
    """
    names = [p for p in spec.param_names if p not in (spec.redshift_param, spec.mass_param)]
    base = {p: spec.defaults[p] for p in names}

    def clip(name: str, value: float) -> float:
        if name not in spec.param_bounds:
            return float(value)
        lo, hi = spec.param_bounds[name]
        pad = 1e-3 * (hi - lo)
        return float(min(max(value, lo + pad), hi - pad))

    axes: list[list[dict[str, float]]] = []
    if "Av" in names:
        axes.append([{"Av": clip("Av", a)} for a in av])
    if spec.sfh_ratio_params:
        axes.append([{p: clip(p, s) for p in spec.sfh_ratio_params} for s in sfh_slope])
    if "log10metallicity" in names:
        axes.append([{"log10metallicity": clip("log10metallicity", m)} for m in log_metallicity])
    out = []
    for combo in product(*axes):
        t = dict(base)
        for part in combo:
            t.update(part)
        out.append(t)
    return out


@dataclass
class RedshiftGrid:
    """Emulator fluxes on a redshift x template grid, reusable for every galaxy.

    Build it once per catalogue with :func:`redshift_grid`, then call
    :meth:`starts` on each batch of photometry.

    Attributes:
        param_names: Emulator inputs, in ``predict`` order.
        redshifts: (Z,) grid redshifts.
        templates: (T, P) template parameter rows (redshift and mass columns unused).
        flux: (T, Z, B) emulator fluxes at the reference mass, for the fitted bands.
        i_z: Column of the redshift.
        i_mass: Column of the log mass.
        mass_bounds: ``(lo, hi)`` allowed log mass.
        min_separation: Minimum separation of two starts in ln(1 + z).
    """

    param_names: tuple[str, ...]
    redshifts: np.ndarray
    templates: np.ndarray
    flux: np.ndarray
    i_z: int
    i_mass: int
    mass_bounds: tuple[float, float]
    min_separation: float = 0.12

    def chi2(self, obs: np.ndarray, err: np.ndarray, valid: np.ndarray | None = None):
        """Chi-squared of every galaxy at every grid point, with the mass solved analytically.

        Args:
            obs: (N, B) observed fluxes, in the emulator's flux unit.
            err: (N, B) flux errors (any error floor already applied).
            valid: (N, B) boolean mask of usable bands (default: all).

        Returns:
            ``(chi2, log_mass)``, both (N, T, Z); ``chi2`` is ``inf`` where the matched
            mass falls outside ``mass_bounds`` or is not positive.
        """
        obs = np.asarray(obs, dtype=np.float64)
        w = 1.0 / np.asarray(err, dtype=np.float64) ** 2
        if valid is not None:
            w = np.where(valid, w, 0.0)
        flux = self.flux.astype(np.float64)
        num = np.einsum("nb,tzb->ntz", w * obs, flux, optimize=True)
        den = np.einsum("nb,tzb->ntz", w, flux**2, optimize=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(den > 0, num / den, 0.0)
            log_mass = _REF_LOG_MASS + np.log10(np.where(scale > 0, scale, np.nan))
        chi2 = np.einsum("nb,nb->n", w, obs**2)[:, None, None] - scale * num
        lo, hi = self.mass_bounds
        ok = np.isfinite(log_mass) & (log_mass > lo) & (log_mass < hi)
        return np.where(ok, chi2, np.inf), np.where(ok, log_mass, 0.5 * (lo + hi))

    def starts(
        self,
        obs: np.ndarray,
        err: np.ndarray,
        k: int = 6,
        valid: np.ndarray | None = None,
    ):
        """The ``k`` best separated redshift minima of every galaxy, as starting points.

        Args:
            obs: (N, B) observed fluxes.
            err: (N, B) flux errors.
            k: Starting points per galaxy.
            valid: (N, B) boolean mask of usable bands.

        Returns:
            ``(x, chi2)``: starting points (N, k, P) in physical parameters, best first,
            and their chi-squared (N, k).  When a galaxy has fewer than ``k`` separated
            minima the remaining slots repeat its best start (``chi2`` there is ``inf``).
        """
        chi2, log_mass = self.chi2(obs, err, valid)
        n = chi2.shape[0]
        t_best = chi2.argmin(axis=1)  # (N, Z) best template at each redshift
        rows = np.arange(n)[:, None]
        cols = np.arange(chi2.shape[2])[None, :]
        prof = chi2[rows, t_best, cols]  # (N, Z)
        pad = np.pad(prof, ((0, 0), (1, 1)), constant_values=np.inf)
        is_min = (prof <= pad[:, :-2]) & (prof <= pad[:, 2:]) & np.isfinite(prof)
        work = np.where(is_min, prof, np.inf)
        lnz = np.log1p(self.redshifts)
        x = np.zeros((n, k, len(self.param_names)))
        out_chi2 = np.full((n, k), np.inf)
        first = None
        for j in range(k):
            iz = work.argmin(axis=1)
            found = np.isfinite(work[np.arange(n), iz])
            if j == 0:
                # a galaxy with no finite grid point at all starts at the grid centre
                first = iz
            iz = np.where(found, iz, first)
            it = t_best[np.arange(n), iz]
            x[:, j] = self.templates[it]
            x[:, j, self.i_z] = self.redshifts[iz]
            x[:, j, self.i_mass] = log_mass[np.arange(n), it, iz]
            out_chi2[:, j] = np.where(found, prof[np.arange(n), iz], np.inf)
            near = np.abs(lnz[None, :] - lnz[iz][:, None]) < self.min_separation
            work = np.where(near, np.inf, work)
        return x, out_chi2


def redshift_grid(
    emulator,
    band_idx: Sequence[int],
    bounds: Mapping[str, Sequence[float]] | None = None,
    n_z: int = 140,
    z_max: float | None = None,
    templates: Sequence[Mapping[str, float]] | None = None,
    min_separation: float = 0.12,
    batch: int = 4096,
) -> RedshiftGrid:
    """Evaluate the emulator on the redshift x template grid.

    Args:
        emulator: An :class:`~arachne.emulator.base.SPSEmulator` (or a view of one).
        band_idx: Indices of the fitted bands in ``emulator.band_names``.
        bounds: ``{name: (lo, hi)}`` fit bounds; default: the emulator's domain.  The
            redshift grid spans the redshift bounds (uniform in ln(1 + z)) and matched
            masses outside the mass bounds are rejected.
        n_z: Number of grid redshifts.
        z_max: Upper end of the grid if lower than the redshift bound.
        templates: Template dicts (default: :func:`default_templates`).
        min_separation: Minimum separation of two starts in ln(1 + z).
        batch: Emulator rows per call.

    Returns:
        The :class:`RedshiftGrid`.
    """
    import jax.numpy as jnp

    spec = emulator.spec
    names = tuple(spec.param_names)
    if spec.redshift_param is None or spec.mass_param is None:
        raise ValueError("the emulator needs a redshift and a mass input for a redshift grid")
    b = {p: tuple(spec.param_bounds[p]) for p in names if p in spec.param_bounds}
    b.update({p: (float(v[0]), float(v[1])) for p, v in (bounds or {}).items() if p in names})
    i_z, i_mass = names.index(spec.redshift_param), names.index(spec.mass_param)
    z_lo, z_hi = b[spec.redshift_param]
    z_hi = min(z_hi, z_max) if z_max is not None else z_hi
    span = np.log1p(z_hi) - np.log1p(z_lo)
    redshifts = np.expm1(np.log1p(z_lo) + span * (np.arange(n_z) + 0.5) / n_z)
    tmpl = list(templates) if templates is not None else default_templates(spec)
    rows = np.zeros((len(tmpl), len(names)))
    for t, row in zip(tmpl, rows):
        for j, p in enumerate(names):
            if j not in (i_z, i_mass):
                row[j] = t[p]
    rows[:, i_mass] = _REF_LOG_MASS
    x = np.repeat(rows[:, None, :], n_z, axis=1)
    x[:, :, i_z] = redshifts[None, :]
    flat = x.reshape(-1, len(names)).astype(np.float32)
    idx = np.asarray(band_idx, dtype=int)
    pred = np.concatenate(
        [
            np.asarray(emulator.predict(jnp.asarray(flat[s : s + batch])))[:, idx]
            for s in range(0, len(flat), batch)
        ]
    )
    return RedshiftGrid(
        param_names=names,
        redshifts=redshifts,
        templates=rows,
        flux=pred.reshape(len(tmpl), n_z, len(idx)),
        i_z=i_z,
        i_mass=i_mass,
        mass_bounds=b[spec.mass_param],
        min_separation=float(min_separation),
    )
