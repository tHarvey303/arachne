"""Emulator metadata: what an emulator's inputs are, and where it is valid.

Fitting code should not hard-code a parameter list.  Every
:class:`~arachne.emulator.base.SPSEmulator` exposes ``emulator.spec``, an
:class:`EmulatorSpec` with the input names, the training domain of each input,
neutral values, default priors, units and the roles of the special inputs
(stellar mass, redshift, SFH ratios).  Checkpoints written by
:class:`~arachne.emulator.parrot_emulator_v2.ParrotEmulatorV2` carry this
information; for older checkpoints it is filled in from :data:`KNOWN_PARAMS`.

This module is pure Python (no JAX), so plotting and results code can use it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

__all__ = [
    "KNOWN_PARAMS",
    "EmulatorSpec",
    "ParamInfo",
    "param_info",
    "resolve_param_name",
]

_SFH_RATIO = re.compile(r"^logsfr_ratio_(\d+)$")
_STUDENT_T_CONTINUITY = {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3}
_UNIFORM = {"dist": "uniform"}


@dataclass(frozen=True)
class ParamInfo:
    """What arachne knows about one emulator input, independent of any checkpoint.

    Attributes:
        name: Canonical parameter name.
        role: ``"mass"``, ``"redshift"``, ``"sfh_ratio"`` or ``None``.
        bounds: Fallback ``(lo, hi)`` for checkpoints that do not record their
            training domain (the fit bounds used with the v6 library).
        default: Neutral value: the blind-start value, and the value a
            parameter is held at when a fit does not use it.
        prior: Default prior spec (see :mod:`arachne.priors.specs`).
        label: Plot label (mathtext allowed).
        unit: Unit string.
        aliases: Other names the same quantity has had in synference libraries.
    """

    name: str
    role: str | None = None
    bounds: tuple[float, float] | None = None
    default: float | None = None
    prior: dict = field(default_factory=lambda: dict(_UNIFORM))
    label: str | None = None
    unit: str = ""
    aliases: tuple[str, ...] = ()


#: Parameters of the synference libraries arachne emulators have been trained on.
#: ``bounds`` are fallbacks only; a checkpoint's own ``param_bounds`` win.
KNOWN_PARAMS: tuple[ParamInfo, ...] = (
    ParamInfo("redshift", role="redshift", bounds=(0.01, 14.0), label=r"$z$"),
    ParamInfo(
        "log_mass",
        role="mass",
        bounds=(4.0, 12.0),
        default=9.5,
        label=r"$\log M_*$",
        unit="dex",
        aliases=("log_stellar_mass",),
    ),
    ParamInfo(
        "dust_slope",
        bounds=(-0.3, 1.1),
        default=0.0,
        label="dust slope",
        aliases=("slope",),
    ),
    ParamInfo("dust_bump_amplitude", bounds=(0.0, 5.0), default=1.0, label="dust bump"),
    ParamInfo("fesc_lya", bounds=(0.0, 1.0), default=0.2, label=r"$f_{\rm esc}^{\rm Ly\alpha}$"),
    ParamInfo("fesc", bounds=(0.0, 1.0), default=0.1, label=r"$f_{\rm esc}$"),
    ParamInfo("f_birth", bounds=(0.0, 2.0), default=1.0, label=r"$f_{\rm birth}$"),
    ParamInfo(
        "dust_temperature", bounds=(20.0, 70.0), default=40.0, label=r"$T_{\rm dust}$", unit="K"
    ),
    ParamInfo("dust_emissivity", bounds=(1.2, 2.2), default=1.7, label=r"$\beta_{\rm dust}$"),
    ParamInfo("dust_mir_alpha", bounds=(1.0, 3.0), default=2.0, label=r"$\alpha_{\rm MIR}$"),
    ParamInfo(
        "log10metallicity", bounds=(-4.0, -1.39), default=-2.5, label=r"$\log Z$", unit="dex"
    ),
    ParamInfo("Av", bounds=(0.001, 5.0), default=0.5, label=r"$A_V$", unit="mag"),
)

_BY_NAME: dict[str, ParamInfo] = {}
for _info in KNOWN_PARAMS:
    _BY_NAME[_info.name] = _info
    for _alias in _info.aliases:
        _BY_NAME[_alias] = _info


def param_info(name: str) -> ParamInfo:
    """Look up a parameter by canonical name or alias.

    ``logsfr_ratio_<i>`` matches for any ``i``.  An unknown name returns a
    bare :class:`ParamInfo` (no bounds, no default, uniform prior), so new
    library parameters work without registering them here.

    Args:
        name: Parameter name as it appears in an emulator's ``param_names``.

    Returns:
        The matching :class:`ParamInfo`; its ``name`` is the canonical one.
    """
    if name in _BY_NAME:
        return _BY_NAME[name]
    if _SFH_RATIO.match(name):
        return ParamInfo(
            name,
            role="sfh_ratio",
            bounds=(-10.0, 10.0),
            default=0.0,
            prior=dict(_STUDENT_T_CONTINUITY),
            label=name,
            unit="dex",
        )
    return ParamInfo(name, label=name)


def resolve_param_name(name: str, available: Sequence[str]) -> str | None:
    """Find ``name`` in ``available``, allowing for aliases (``slope`` / ``dust_slope``).

    Args:
        name: Requested parameter name.
        available: Names an emulator actually has.

    Returns:
        The entry of ``available`` that means the same quantity, or ``None``.
    """
    if name in available:
        return name
    info = param_info(name)
    for candidate in (info.name, *info.aliases):
        if candidate in available:
            return candidate
    return None


def _pair(value: Iterable[float]) -> tuple[float, float]:
    lo, hi = (float(v) for v in value)
    return lo, hi


@dataclass(frozen=True)
class EmulatorSpec:
    """Description of an emulator's inputs and outputs.

    Attributes:
        param_names: Input names, in ``predict`` column order.
        band_names: Output band names, in ``predict`` column order.
        param_bounds: ``{name: (lo, hi)}``: the domain the emulator is valid on.
        defaults: ``{name: value}`` neutral values (always inside the bounds).
        priors: ``{name: prior spec}`` default priors.
        units: ``{name: unit string}``.
        labels: ``{name: plot label}``.
        mass_param: The input photometry is linear in (as log10 mass), or ``None``.
        redshift_param: The redshift input, or ``None``.
        sfh_ratio_params: The SFH log-ratio inputs, in order.
        meta: Free-form provenance: ``bounds_source`` (``"checkpoint"`` or
            ``"fallback"``), ``library``, and for views ``fixed`` / ``renamed``.
        band_errors: The emulator's per-band fractional flux error in redshift
            bins, measured on held-out library rows (see
            :mod:`arachne.emulator.validation`), or ``None`` if the checkpoint
            carries none.  Read it with :meth:`band_error`.
        simulator: Recipe of the SPS model the training library was generated
            with (see :mod:`arachne.postprocess.recipe`), or ``None``.  It lets
            :mod:`arachne.postprocess` rebuild spectra and derived quantities
            from a fit.  The recipe uses the library's own parameter names.
    """

    param_names: tuple[str, ...]
    band_names: tuple[str, ...]
    param_bounds: dict[str, tuple[float, float]]
    defaults: dict[str, float]
    priors: dict[str, dict]
    units: dict[str, str]
    labels: dict[str, str]
    mass_param: str | None
    redshift_param: str | None
    sfh_ratio_params: tuple[str, ...]
    meta: dict = field(default_factory=dict)
    band_errors: dict | None = None
    simulator: dict | None = None

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_names(
        cls,
        param_names: Sequence[str],
        band_names: Sequence[str],
        *,
        param_bounds: Mapping[str, Iterable[float]] | None = None,
        defaults: Mapping[str, float] | None = None,
        priors: Mapping[str, dict] | None = None,
        units: Mapping[str, str] | None = None,
        mass_param: str | None = None,
        redshift_param: str | None = None,
        meta: Mapping | None = None,
        band_errors: Mapping | None = None,
        simulator: Mapping | None = None,
    ) -> "EmulatorSpec":
        """Build a spec from names, filling every gap from :data:`KNOWN_PARAMS`.

        Args:
            param_names: Input names in column order.
            band_names: Output band names in column order.
            param_bounds: Recorded training domain for some or all inputs.
            defaults: Neutral values overriding the registry's.
            priors: Default prior specs overriding the registry's.
            units: Units overriding the registry's.
            mass_param: Mass input; default: the input with role ``"mass"``.
            redshift_param: Redshift input; default: the input with role ``"redshift"``.
            meta: Provenance entries.
            band_errors: Per-band error table (:mod:`arachne.emulator.validation`).
            simulator: Simulator recipe (:mod:`arachne.postprocess.recipe`).

        Returns:
            The completed spec.  A parameter with neither recorded nor known
            bounds gets none, and code that needs them raises a clear error.
        """
        names = tuple(str(p) for p in param_names)
        infos = {p: param_info(p) for p in names}
        given = {k: _pair(v) for k, v in (param_bounds or {}).items()}
        bounds = {
            p: given.get(p, infos[p].bounds)
            for p in names
            if p in given or infos[p].bounds is not None
        }

        def by_role(role: str) -> str | None:
            return next((p for p in names if infos[p].role == role), None)

        out_defaults: dict[str, float] = {}
        for p in names:
            value = (defaults or {}).get(p, infos[p].default)
            if value is None and p in bounds:
                value = 0.5 * (bounds[p][0] + bounds[p][1])
            if value is None:
                continue
            if p in bounds:
                lo, hi = bounds[p]
                margin = 0.01 * (hi - lo)
                value = min(max(float(value), lo + margin), hi - margin)
            out_defaults[p] = float(value)

        source = "checkpoint" if all(p in given for p in names) else "fallback"
        return cls(
            param_names=names,
            band_names=tuple(str(b) for b in band_names),
            param_bounds=bounds,
            defaults=out_defaults,
            priors={p: dict((priors or {}).get(p, infos[p].prior)) for p in names},
            units={p: str((units or {}).get(p, infos[p].unit)) for p in names},
            labels={p: infos[p].label or p for p in names},
            mass_param=mass_param if mass_param is not None else by_role("mass"),
            redshift_param=(redshift_param if redshift_param is not None else by_role("redshift")),
            sfh_ratio_params=tuple(p for p in names if infos[p].role == "sfh_ratio"),
            meta={"bounds_source": source, **dict(meta or {})},
            band_errors=dict(band_errors) if band_errors else None,
            simulator=dict(simulator) if simulator else None,
        )

    # -- queries -----------------------------------------------------------------
    def bounds(self, name: str) -> tuple[float, float]:
        """``(lo, hi)`` for one input.

        Raises:
            KeyError: If the emulator records no domain for ``name``.
        """
        if name not in self.param_bounds:
            raise KeyError(
                f"no bounds known for emulator parameter {name!r}; pass them explicitly or "
                "record them in the checkpoint (python -m arachne.emulator annotate)."
            )
        return self.param_bounds[name]

    def band_error(
        self,
        redshift: float | None = None,
        stat: str = "mad",
        z_bin: str | None = None,
        return_bin: bool = False,
    ):
        """The emulator's fractional flux error per band, in ``band_names`` order.

        Add it in quadrature to a fit's model-error floor.  The value comes from
        the narrowest well-populated redshift bin containing ``redshift``
        (``"all"`` if none, or if ``redshift`` is ``None``).

        The table was measured over the whole library.  If this spec belongs to
        a view that holds inputs fixed (``meta["fixed"]``), the error on that
        slice of parameter space can differ.

        Args:
            redshift: Redshift of the object being fitted.
            stat: ``"mad"`` (robust scatter, default), ``"std"``, ``"p95"`` or ``"bias"``.
            z_bin: Use this bin of the table instead of choosing by redshift.
            return_bin: Also return the name of the bin used.

        Returns:
            Array of shape ``(n_bands,)``, or ``(array, bin_name)`` with ``return_bin``.

        Raises:
            ValueError: If the checkpoint carries no table (``band_errors is
                None``) or the table lacks one of this spec's bands.
        """
        if self.band_errors is None:
            raise ValueError(
                "this emulator carries no band-error table; measure one with "
                "scripts/validate_parrot_emulator.py --embed or attach one with "
                "python -m arachne.emulator annotate CKPT --band-errors FILE.json"
            )
        from arachne.emulator.validation import select_band_errors

        values, name = select_band_errors(
            self.band_errors, self.band_names, redshift=redshift, stat=stat, z_bin=z_bin
        )
        return (values, name) if return_bin else values

    def clip_bounds(
        self, bounds: Mapping[str, Iterable[float]], names: Sequence[str] | None = None
    ) -> dict[str, tuple[float, float]]:
        """Intersect requested fit bounds with the emulator's domain.

        Use this in a driver that has its own preferred bounds: the fit then
        never leaves the region the emulator was trained on, whichever
        checkpoint is loaded.

        Args:
            bounds: ``{name: (lo, hi)}`` requested by the caller.  Names missing
                here get the emulator's own bounds.
            names: Parameters to return; default: every emulator input.

        Returns:
            ``{name: (lo, hi)}`` in ``names`` order.

        Raises:
            ValueError: If a requested interval does not overlap the domain.
        """
        out = {}
        for p in names if names is not None else self.param_names:
            if p not in bounds:
                out[p] = self.bounds(p)
                continue
            lo, hi = _pair(bounds[p])
            if p in self.param_bounds:
                dlo, dhi = self.param_bounds[p]
                lo, hi = max(lo, dlo), min(hi, dhi)
            if not lo < hi:
                raise ValueError(
                    f"{p}: requested bounds {tuple(bounds[p])} do not overlap the emulator "
                    f"domain {self.param_bounds.get(p)}"
                )
            out[p] = (lo, hi)
        return out

    def subset(
        self,
        param_names: Sequence[str] | None = None,
        band_names: Sequence[str] | None = None,
        rename: Mapping[str, str] | None = None,
        meta: Mapping | None = None,
    ) -> "EmulatorSpec":
        """Spec of a view exposing some inputs (optionally renamed) and some bands.

        Args:
            param_names: Inputs to keep, by their *current* names, in the new order.
            band_names: Bands to keep, in the new order.
            rename: ``{current name: new name}`` applied after selection.
            meta: Entries merged into ``meta``.

        Returns:
            The new spec.
        """
        keep = tuple(param_names) if param_names is not None else self.param_names
        missing = [p for p in keep if p not in self.param_names]
        if missing:
            raise ValueError(f"not emulator parameters: {missing}; have {list(self.param_names)}")
        new = dict(rename or {})

        def name(p: str | None) -> str | None:
            return None if p is None or p not in keep else new.get(p, p)

        def remap(d: Mapping) -> dict:
            return {new.get(p, p): d[p] for p in keep if p in d}

        return replace(
            self,
            param_names=tuple(new.get(p, p) for p in keep),
            band_names=tuple(band_names) if band_names is not None else self.band_names,
            param_bounds=remap(self.param_bounds),
            defaults=remap(self.defaults),
            priors=remap(self.priors),
            units=remap(self.units),
            labels=remap(self.labels),
            mass_param=name(self.mass_param),
            redshift_param=name(self.redshift_param),
            sfh_ratio_params=tuple(new.get(p, p) for p in keep if p in self.sfh_ratio_params),
            meta={**self.meta, **dict(meta or {})},
        )

    def describe(self) -> str:
        """Human-readable table of the inputs (name, bounds, default, prior, unit)."""
        lines = [
            f"{len(self.param_names)} inputs, {len(self.band_names)} bands; "
            f"mass: {self.mass_param}, redshift: {self.redshift_param}; "
            f"bounds from {self.meta.get('bounds_source', '?')}"
        ]
        if self.meta.get("library"):
            lines.append(f"library: {self.meta['library']}")
        for key in ("fixed", "renamed"):
            if self.meta.get(key):
                lines.append(f"{key}: {self.meta[key]}")
        if self.band_errors is None:
            lines.append("band errors: none recorded")
        else:
            table = self.band_errors["bands"]
            bins = list(next(iter(table.values())))
            lines.append(
                f"band errors: {len(table)} bands, {len(bins)} redshift bins "
                f"({self.band_errors.get('source', '') or 'no source recorded'})"
            )
        if self.simulator is None:
            lines.append("simulator: none recorded")
        else:
            sim = self.simulator
            lines.append(
                f"simulator: {sim.get('engine', '?')} recipe, grid {sim.get('grid_name', '?')}, "
                f"{len(sim.get('reference', {}).get('params', []))} reference rows"
            )
        for p in self.param_names:
            b = self.param_bounds.get(p)
            bounds = f"[{b[0]:g}, {b[1]:g}]" if b else "?"
            default = f"{self.defaults[p]:g}" if p in self.defaults else "?"
            prior = self.priors[p].get("dist", "uniform")
            lines.append(
                f"  {p:<22s} {bounds:<16s} default {default:<8s} prior {prior:<10s} "
                f"{self.units.get(p, '')}"
            )
        return "\n".join(lines)
