"""The simulator recipe: a JSON description of the SPS model an emulator was trained on.

An emulator checkpoint predicts photometry only.  The *recipe* records how the
training library was generated (SPS grid, emission model, dust, IGM, SFH and
metallicity parametrisation, cosmology), so that the exact model can be rebuilt
with Synthesizer after a fit (:mod:`arachne.postprocess.simulator`) to recover
spectra and derived quantities.  It is stored in the checkpoint
(``emulator.spec.simulator``; ``python -m arachne.emulator annotate CKPT
--simulator-library LIB.hdf5``) together with a few reference library rows that
let any installation check that its grid and Synthesizer version reproduce the
library (:func:`arachne.postprocess.simulator.check_recipe`).

This module is pure Python / numpy / h5py (no JAX, no Synthesizer).

Recipe layout (``version`` 1)::

    {
      "version": 1, "engine": "synthesizer", "library": "<file name>",
      "grid_name": "<Synthesizer grid name>",
      "cosmology": "<astropy cosmology, YAML>",
      "param_names": [...],                    # library inputs, library order
      "mass_param": "log_mass", "mass_kind": "formed",
      "redshift_param": "redshift",
      "emission_model": {"class", "factory", "kwargs", "components",
                         "model_fixed_parameters"},   # synference's description format
      "emission_model_key": "total",
      "stellar_params": {"tau_v": {"product": ["Av"], "scale": 0.9208}, "slope": "slope", ...},
          # a rule may carry "unit" (e.g. "K"): the simulator attaches it
      "igm": {"class": "Asada25", "module": "...", "params": {...}},
      "sfh": {"class": "Continuity", "logsfr_ratios": [...],
              "agebins": {"rule": "continuity", "n_bins": 6, "first_myr": 3, "second_myr": 10,
                          "last_fraction": 0.15, "z_form": 20}},
      "zdist": {"class": "DeltaConstant", "kwargs": {"log10metallicity": "log10metallicity"}},
      "reference": {"band_names", "params", "photometry_nJy", "derived"}
    }

String values inside ``emission_model`` name emitter attributes (Synthesizer's
convention); ``stellar_params`` says how each attribute is computed from the
library inputs.  Only declarative transforms are allowed: no code is stored or
executed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "AV_TO_TAU_V",
    "PRESETS",
    "continuity_agebins",
    "continuity_sfh",
    "check_recipe_structure",
    "recipe_from_checkpoint",
    "recipe_from_library",
    "recipe_hash",
    "resolve_params",
    "stellar_attributes",
]

RECIPE_VERSION = 1
AV_TO_TAU_V = 1.086  # A_V = 1.086 tau_V

_REQUIRED = (
    "version",
    "grid_name",
    "cosmology",
    "param_names",
    "mass_param",
    "redshift_param",
    "emission_model",
    "stellar_params",
    "igm",
    "sfh",
    "zdist",
)

#: What the library headers do not record declaratively, per library family.
#: ``agebins`` and ``stellar_params`` come from the library generation scripts
#: (synference ``examples/library_generation``); every preset is checked against
#: stored library rows when a recipe is built.
PRESETS: dict[str, dict] = {
    # BPASS, Continuity SFH, single-screen Calzetti dust (ParrotEmulatorV2 production library)
    "v6": {
        "emission_model": {
            "version": 1,
            "class": "PacmanEmissionNoEscapedWithDust",
            "factory": "PacmanEmission",
            "kwargs": {"tau_v": "tau_v", "fesc": 0.0, "fesc_ly_alpha": "fesc_lya"},
            "components": {
                "dust_curve": {
                    "class": "Calzetti2000",
                    "module": "synthesizer.emission_models.transformers.dust_attenuation",
                    "params": {"slope": "slope", "ampl": "dust_bump_amplitude"},
                },
                "dust_emission": {
                    "class": "Greybody",
                    "module": "synthesizer.emission_models.generators.dust.greybody",
                    "params": {"temperature": {"__unyt__": [40.0, "K"]}, "emissivity": 1.5},
                },
            },
            "model_fixed_parameters": {},
        },
        "igm": {
            "class": "Asada25",
            "module": "synthesizer.emission_models.transformers.igm",
            "params": {},
        },
        "agebins": {
            "rule": "continuity",
            "n_bins": 6,
            "first_myr": 3.0,
            "second_myr": 10.0,
            "last_fraction": 0.15,
            "z_form": 20.0,
        },
        "stellar_params": {
            "tau_v": {"product": ["Av"], "scale": 1.0 / AV_TO_TAU_V},
            "slope": "slope",
            "fesc_lya": "fesc_lya",
            "dust_bump_amplitude": "dust_bump_amplitude",
        },
    },
    # BPASS, Continuity SFH, two-component (birth cloud + ISM) dust, Casey12 dust emission
    "v7": {
        "agebins": {
            "rule": "continuity",
            "n_bins": 6,
            "first_myr": 3.0,
            "second_myr": 10.0,
            "last_fraction": 0.15,
            "z_form": 30.0,
        },
        "stellar_params": {
            "tau_v_ism": {"product": ["Av"], "scale": 1.0 / AV_TO_TAU_V},
            "tau_v_birth": {"product": ["f_birth", "Av"], "scale": 1.0 / AV_TO_TAU_V},
            "dust_slope": "dust_slope",
            "dust_bump_amplitude": "dust_bump_amplitude",
            "fesc": "fesc",
            "fesc_lya": "fesc_lya",
            "dust_temperature": {"product": ["dust_temperature"], "unit": "K"},
            "dust_emissivity": "dust_emissivity",
            "dust_mir_alpha": "dust_mir_alpha",
        },
    },
}


# ----------------------------------------------------------------------------- SFH


def continuity_agebins(redshift: float, cosmo: Any, rule: Mapping) -> np.ndarray:
    """Lookback-time bins of the library's Continuity SFH at one redshift.

    Bin 0 is ``[0, first_myr]``, bin 1 ends at ``second_myr``, the last bin is the
    oldest ``last_fraction`` of the time since ``z_form`` and the bins between are
    logarithmically spaced.

    Args:
        redshift: Galaxy redshift.
        cosmo: An ``astropy`` cosmology.
        rule: The recipe's ``sfh["agebins"]`` entry.

    Returns:
        Array of shape ``(n_bins, 2)`` of bin edges in Myr.
    """
    if rule.get("rule") != "continuity":
        raise ValueError(f"unknown age-bin rule {rule.get('rule')!r}")
    available = cosmo.age(float(redshift)).to_value("Myr") - cosmo.age(
        float(rule["z_form"])
    ).to_value("Myr")
    if available <= rule["second_myr"]:
        raise ValueError(f"redshift {redshift} is too close to z_form = {rule['z_form']}")
    middle = np.logspace(
        np.log10(rule["second_myr"]),
        np.log10(available * (1.0 - rule["last_fraction"])),
        int(rule["n_bins"]) - 2,
    )
    edges = np.concatenate(([0.0, float(rule["first_myr"])], middle, [available]))
    return np.stack([edges[:-1], edges[1:]], axis=1)


def continuity_sfh(log_mass: float, logsfr_ratios: Sequence[float], agebins_myr: np.ndarray):
    """Star-formation rate in each age bin of a Continuity SFH.

    ``logsfr_ratios[i] = log10(SFR_i / SFR_{i+1})`` with bin 0 the youngest (the
    Prospector convention used by Synthesizer).

    Args:
        log_mass: log10 of the total stellar mass formed [Msun].
        logsfr_ratios: ``n_bins - 1`` log ratios.
        agebins_myr: ``(n_bins, 2)`` bin edges in Myr.

    Returns:
        ``(sfr, mass)``: SFR per bin [Msun / yr] and mass formed per bin [Msun].
    """
    ratios = np.asarray(logsfr_ratios, dtype=float)
    widths = (agebins_myr[:, 1] - agebins_myr[:, 0]) * 1e6
    rel = 10.0 ** np.concatenate(([0.0], -np.cumsum(ratios)))
    mass = rel * widths
    mass *= 10.0 ** float(log_mass) / mass.sum()
    return mass / widths, mass


# ----------------------------------------------------------------------------- parameters


def resolve_params(recipe: Mapping, params: Mapping[str, float]) -> dict[str, float]:
    """Pick the recipe's library inputs out of a parameter dict, following name aliases.

    A fit may use an :class:`~arachne.emulator.view.EmulatorView` whose names
    differ from the library's (``dust_slope`` for ``slope``).

    Args:
        recipe: The recipe.
        params: ``{name: value}`` from a fit (extra entries are ignored).

    Returns:
        ``{library name: value}`` for every entry of ``recipe["param_names"]``.

    Raises:
        KeyError: If an input is missing under every known name.
    """
    from arachne.emulator.spec import param_info

    out, missing = {}, []
    for name in recipe["param_names"]:
        if name in params:
            out[name] = float(params[name])
            continue
        info = param_info(name)
        for alias in (info.name, *info.aliases):
            if alias in params:
                out[name] = float(params[alias])
                break
        else:
            missing.append(name)
    if missing:
        raise KeyError(f"parameters missing for the simulator: {missing}; have {sorted(params)}")
    return out


def stellar_attributes(recipe: Mapping, params: Mapping[str, float]) -> dict[str, float]:
    """Emitter attributes of the emission model, computed from library inputs.

    Args:
        recipe: The recipe.
        params: ``{library name: value}`` (see :func:`resolve_params`).

    Returns:
        ``{attribute: value}`` to pass to the stellar component, as plain floats
        (a rule's ``"unit"`` is attached by the simulator).
    """
    out = {}
    for attr, rule in recipe["stellar_params"].items():
        if isinstance(rule, str):
            out[attr] = float(params[rule])
        else:
            value = float(rule.get("scale", 1.0))
            for name in rule["product"]:
                value *= float(params[name])
            out[attr] = value
    return out


# ----------------------------------------------------------------------------- construction


def check_recipe_structure(recipe: Mapping) -> None:
    """Raise ``ValueError`` if ``recipe`` is not a usable version-1 recipe."""
    if not isinstance(recipe, Mapping):
        raise ValueError("simulator recipe must be a dict")
    missing = [k for k in _REQUIRED if k not in recipe]
    if missing:
        raise ValueError(f"simulator recipe lacks {missing}")
    if int(recipe["version"]) != RECIPE_VERSION:
        raise ValueError(f"simulator recipe version {recipe['version']} is not supported")
    names = list(recipe["param_names"])
    for key in ("mass_param", "redshift_param"):
        if recipe[key] not in names:
            raise ValueError(f"{key} {recipe[key]!r} is not one of the recipe's param_names")
    used = set(recipe["sfh"].get("logsfr_ratios", []))
    used.update(v for v in recipe["zdist"].get("kwargs", {}).values() if isinstance(v, str))
    for rule in recipe["stellar_params"].values():
        used.update([rule] if isinstance(rule, str) else rule["product"])
    unknown = sorted(used - set(names))
    if unknown:
        raise ValueError(f"simulator recipe refers to unknown parameters {unknown}")
    json.dumps(recipe)  # must be JSON-serialisable


def recipe_hash(recipe: Mapping) -> str:
    """Short hash of the physical content of a recipe (reference rows excluded)."""
    body = {k: v for k, v in recipe.items() if k != "reference"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


def _attr_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.size == 1:
        return str(value.reshape(-1)[0])
    return str(value)


def guess_preset(library: str | Path) -> str | None:
    """Preset name from a library file name (``..._v6_...`` -> ``"v6"``), or ``None``."""
    stem = Path(library).stem
    return next((p for p in PRESETS if f"_{p}" in stem), None)


def recipe_from_library(
    library: str | Path,
    preset: str | None = None,
    band_names: Sequence[str] | None = None,
    n_reference: int = 48,
    seed: int = 0,
) -> dict:
    """Build a recipe from a synference library header plus a preset.

    The header supplies the grid name, cosmology, parameter names, SFH and
    metallicity classes and (libraries written since 2026-11) the emission-model
    description and IGM model.  The preset supplies what headers do not hold in
    declarative form: the age-bin rule, the emitter-attribute transforms and, for
    old headers, the emission model and IGM.

    Args:
        library: Path to the synference HDF5 library.
        preset: Key of :data:`PRESETS`; default: guessed from the file name.
        band_names: Bands to keep in the reference rows (default: every library band).
        n_reference: Number of library rows to embed for :func:`check_recipe`.
        seed: Seed of the reference-row selection.

    Returns:
        The recipe (a JSON-serialisable dict).
    """
    import h5py

    library = Path(library)
    preset = preset or guess_preset(library)
    if preset not in PRESETS:
        raise ValueError(
            f"no simulator preset for {library.name}; pass one of {sorted(PRESETS)} "
            "or add an entry to arachne.postprocess.recipe.PRESETS"
        )
    extra = PRESETS[preset]
    with h5py.File(library, "r") as f:
        model = f["Model"]
        names = [str(x) for x in f.attrs["ParameterNames"]]
        codes = [str(x) for x in f.attrs["FilterCodes"]]
        sup_names = [str(x) for x in f.attrs.get("SupplementaryParameterNames", [])]
        em_attrs = dict(model["EmissionModel"].attrs)
        if "emission_model" in extra:
            emission = extra["emission_model"]
        elif "description" in em_attrs:
            emission = json.loads(_attr_str(em_attrs["description"]))
        else:
            raise ValueError(f"{library.name} has no emission-model description and no preset one")
        if "igm" in extra:
            igm = extra["igm"]
        elif "igm_class" in model.attrs:
            igm = {
                "class": _attr_str(model.attrs["igm_class"]),
                "module": _attr_str(model.attrs["igm_module"]),
                "params": json.loads(_attr_str(model.attrs.get("igm_params", "{}"))),
            }
        else:
            raise ValueError(f"{library.name} does not record its IGM model and no preset has one")
        sfh_class = _attr_str(model.attrs["sfh_class"])
        if sfh_class != "Continuity":
            raise NotImplementedError(f"SFH class {sfh_class!r} is not supported yet")
        zdist_class = _attr_str(model.attrs["metallicity_distribution_class"])
        if zdist_class != "DeltaConstant" or "log10metallicity" not in names:
            raise NotImplementedError(f"metallicity model {zdist_class!r} is not supported yet")
        ratios = sorted(
            (n for n in names if n.startswith("logsfr_ratio_")), key=lambda n: int(n[13:])
        )
        if len(ratios) != extra["agebins"]["n_bins"] - 1:
            raise ValueError(f"{len(ratios)} SFH ratios do not match the preset's age bins")

        n_rows = f["Grid/Parameters"].shape[1]
        rows = np.sort(np.random.default_rng(seed).choice(n_rows, 4 * n_reference, replace=False))
        params = f["Grid/Parameters"][:, rows].T
        phot = f["Grid/Photometry"][:, rows].T
        good = np.all(np.isfinite(params), axis=1) & np.all(np.isfinite(phot), axis=1)
        keep = np.flatnonzero(good)[:n_reference]
        bands = list(band_names) if band_names is not None else codes
        missing = [b for b in bands if b not in codes]
        if missing:
            raise ValueError(f"{library.name} lacks the bands {missing}")
        derived = {}
        if sup_names:
            sup = f["Grid/SupplementaryParameters"][:, rows].T[keep]
            for ours, theirs in _REFERENCE_DERIVED.items():
                if theirs in sup_names:
                    derived[ours] = [float(x) for x in sup[:, sup_names.index(theirs)]]
        recipe = {
            "version": RECIPE_VERSION,
            "engine": "synthesizer",
            "library": library.name,
            "preset": preset,
            "grid_name": _attr_str(model.attrs["grid_name"]),
            "cosmology": _attr_str(model.attrs["cosmology"]),
            "param_names": names,
            "mass_param": "log_mass",
            "mass_kind": "formed",
            "redshift_param": "redshift",
            "emission_model": emission,
            "emission_model_key": _attr_str(model.attrs.get("emission_model_key", "total")),
            "stellar_params": extra["stellar_params"],
            "igm": igm,
            "sfh": {"class": sfh_class, "logsfr_ratios": ratios, "agebins": extra["agebins"]},
            "zdist": {"class": zdist_class, "kwargs": {"log10metallicity": "log10metallicity"}},
            "reference": {
                "band_names": bands,
                "library_rows": [int(r) for r in rows[keep]],
                "params": [[float(x) for x in row] for row in params[keep]],
                "photometry_nJy": [
                    [float(x) for x in row[[codes.index(b) for b in bands]]] for row in phot[keep]
                ],
                "derived": derived,
            },
        }
    recipe = json.loads(json.dumps(recipe))
    check_recipe_structure(recipe)
    return recipe


#: our derived-quantity name -> the library's SupplementaryParameters name
_REFERENCE_DERIVED = {
    "sfr_10": "sfr_10",
    "sfr_100": "sfr_100",
    "age_mass_weighted": "mass_weighted_age",
    "log_mass_surviving": "log_surviving_mass",
}


def recipe_from_checkpoint(path: str | Path) -> dict | None:
    """Read the recipe out of a ``ParrotEmulatorV2`` checkpoint without importing JAX.

    Args:
        path: Checkpoint (``.eqx``, an ``npz`` archive).

    Returns:
        The recipe, or ``None`` if the checkpoint carries none.
    """
    with np.load(str(path), allow_pickle=False) as raw:
        if "v2_config" not in raw:
            return None
        cfg = json.loads(str(raw["v2_config"]))
    return cfg.get("simulator")
