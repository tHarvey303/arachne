"""Rebuild the SPS model behind an emulator with Synthesizer and evaluate it.

:class:`Simulator` turns a recipe (:mod:`arachne.postprocess.recipe`) into
Synthesizer objects and maps one set of library parameters to the observed
spectrum, photometry and derived quantities.  It is the slow, exact
counterpart of the emulator: about 15 ms per evaluation on one core, against
microseconds, so it is for post-processing posterior draws, not for fitting.

Everything that touches the Synthesizer API is in this module, so a Synthesizer
release that moves things needs changes here only.  Synthesizer is imported
lazily; it is installed with synference, which arachne already depends on.

:func:`check_recipe` re-simulates the reference library rows stored in the
recipe.  Run it (``python -m arachne.emulator check-simulator CKPT``) on every
new machine: a different grid file or Synthesizer version shows up there, not
in a fit.
"""

from __future__ import annotations

import importlib
import logging
import os
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .recipe import (
    check_recipe_structure,
    continuity_agebins,
    continuity_sfh,
    resolve_params,
    stellar_attributes,
)

logger = logging.getLogger(__name__)

__all__ = ["ADDITIVE", "DERIVED_NAMES", "DERIVED_UNITS", "Simulator", "check_recipe", "find_grid"]

#: Derived quantities returned by :meth:`Simulator.evaluate`, in order.
DERIVED_NAMES = (
    "log_mass_formed",
    "log_mass_surviving",
    "sfr_10",
    "sfr_100",
    "age_mass_weighted",
    "beta_uv",
    "d4000",
)
DERIVED_UNITS = {
    "log_mass_formed": "log10(Msun)",
    "log_mass_surviving": "log10(Msun)",
    "sfr_10": "Msun/yr",
    "sfr_100": "Msun/yr",
    "age_mass_weighted": "Myr",
    "beta_uv": "",
    "d4000": "",
}
#: How each derived quantity combines over pixels: ``"logsum"`` (log of a summed mass),
#: ``"sum"``, ``"mass_mean"`` (mean weighted by formed mass) or ``None`` (not additive).
ADDITIVE = {
    "log_mass_formed": "logsum",
    "log_mass_surviving": "logsum",
    "sfr_10": "sum",
    "sfr_100": "sum",
    "age_mass_weighted": "mass_mean",
    "beta_uv": None,
    "d4000": None,
}

# Where classes named in a recipe have lived across Synthesizer versions.
_CLASS_HOMES = (
    "synthesizer.emission_models.transformers.dust_attenuation",
    "synthesizer.emission_models.attenuation",
    "synthesizer.emission_models.transformers.igm",
    "synthesizer.emission_models.generators.dust",
    "synthesizer.emission_models.dust.emission",
    "synthesizer.emission_models",
)


def find_grid(grid_name: str, grid_dir: str | Path | None = None) -> Path:
    """Directory holding ``<grid_name>.hdf5``.

    Searched in order: ``grid_dir``, ``$ARACHNE_SPS_GRID_DIR``,
    ``$SYNTHESIZER_GRID_DIR``, Synthesizer's default grid directory.

    Raises:
        FileNotFoundError: If the grid is in none of them.  The grid must be the
            file the library was generated with; a grid of the same family but a
            different build changes nebular fluxes at the per-cent level.
    """
    candidates = [grid_dir, os.getenv("ARACHNE_SPS_GRID_DIR"), os.getenv("SYNTHESIZER_GRID_DIR")]
    try:
        from synthesizer import get_grids_dir

        candidates.append(get_grids_dir())
    except Exception:  # noqa: BLE001 - older Synthesizer has no default directory
        pass
    tried = []
    for cand in candidates:
        if not cand:
            continue
        cand = Path(cand).expanduser()
        tried.append(str(cand))
        if (cand / f"{grid_name}.hdf5").exists():
            return cand
    raise FileNotFoundError(
        f"SPS grid {grid_name}.hdf5 not found in {tried}; pass grid_dir or set ARACHNE_SPS_GRID_DIR"
    )


def _resolve_class(name: str, module: str | None = None) -> Any:
    for mod in (module, *_CLASS_HOMES):
        if not mod:
            continue
        try:
            obj = getattr(importlib.import_module(mod), name, None)
        except ImportError:
            continue
        if obj is not None:
            return obj
    raise ImportError(f"class {name!r} not found in this Synthesizer installation")


def _decode(value: Any) -> Any:
    """Undo the recipe's JSON encoding of unyt quantities."""
    if isinstance(value, Mapping) and "__unyt__" in value:
        from unyt import unyt_array, unyt_quantity

        val, unit = value["__unyt__"]
        return unyt_array(val, unit) if isinstance(val, list) else unyt_quantity(val, unit)
    return value


def _build_component(desc: Mapping) -> Any:
    cls = _resolve_class(desc["class"], desc.get("module"))
    return cls(**{k: _decode(v) for k, v in desc.get("params", {}).items()})


def _build_emission_model(grid: Any, desc: Mapping) -> Any:
    """Emission model from synference's description format (factory, kwargs, components)."""
    factory = _resolve_class(desc["factory"], "synthesizer.emission_models")
    kwargs = {k: _decode(v) for k, v in desc.get("kwargs", {}).items()}
    kwargs.pop("label", None)
    for role, comp in desc.get("components", {}).items():
        kwargs[role] = _build_component(comp)
    model = factory(grid=grid, **kwargs)
    for label, fixed in desc.get("model_fixed_parameters", {}).items():
        if label not in model._models:
            logger.warning("emission model has no component %r to restore parameters on", label)
            continue
        target = model._models[label].fixed_parameters
        for key, value in fixed.items():
            target[key] = _decode(value)
    if desc.get("class") and type(model).__name__ != desc["class"]:
        logger.warning(
            "emission model class is %s, the recipe expects %s", type(model).__name__, desc["class"]
        )
    return model


class Simulator:
    """The Synthesizer model an emulator was trained on.

    Args:
        recipe: Simulator recipe (``emulator.spec.simulator``).
        grid_dir: Directory of the SPS grid (see :func:`find_grid`).
        band_names: Filters for :meth:`evaluate` photometry, as SVO codes
            (``"JWST/NIRCam.F444W"``).  Curves come from Synthesizer's SVO cache
            (first use downloads them).  ``None``: no photometry.

    Attributes:
        recipe: The recipe.
        band_names: The photometry bands.
        cosmo: The library's ``astropy`` cosmology.
        synthesizer_version: Version of the Synthesizer in use.
    """

    def __init__(
        self,
        recipe: Mapping,
        grid_dir: str | Path | None = None,
        band_names: Sequence[str] | None = None,
    ):
        """Load the grid and build the emission model, IGM model and filters."""
        check_recipe_structure(recipe)
        warnings.filterwarnings("ignore", module="synthesizer")
        import synthesizer
        from astropy.cosmology import Cosmology
        from synthesizer.grid import Grid

        self.recipe = dict(recipe)
        self.synthesizer_version = str(synthesizer.__version__)
        self.grid_dir = find_grid(recipe["grid_name"], grid_dir)
        self.grid = Grid(recipe["grid_name"], str(self.grid_dir))
        self.cosmo = Cosmology.from_format(recipe["cosmology"], format="yaml")
        self.key = recipe.get("emission_model_key", "total")
        self.emission_model = _build_emission_model(self.grid, recipe["emission_model"])
        self.emission_model.save_spectra(self.key)
        igm = recipe["igm"]
        igm_cls = _resolve_class(igm["class"], igm.get("module"))
        params = {
            k: tuple(v) if isinstance(v, list) else v for k, v in igm.get("params", {}).items()
        }
        self.igm = igm_cls(**params) if params else igm_cls
        self.band_names = list(band_names) if band_names is not None else []
        self.filters = None
        if self.band_names:
            from synthesizer.instruments import FilterCollection

            self.filters = FilterCollection(filter_codes=self.band_names, new_lam=self.grid.lam)
            self.pivot_um = np.asarray(self.filters.pivot_lams.to("um").value, dtype=float)
        self._agebins: dict[float, np.ndarray] = {}

    # -- pieces ------------------------------------------------------------------
    def agebins(self, redshift: float) -> np.ndarray:
        """``(n_bins, 2)`` SFH bin edges in Myr at ``redshift`` (cached)."""
        z = float(redshift)
        if z not in self._agebins:
            if len(self._agebins) > 4096:
                self._agebins.clear()
            self._agebins[z] = continuity_agebins(z, self.cosmo, self.recipe["sfh"]["agebins"])
        return self._agebins[z]

    def sfh(self, params: Mapping[str, float]):
        """SFH of one parameter set in physical units.

        Returns:
            ``(agebins_myr, sfr, mass)``: bin edges (n_bins, 2), SFR per bin
            [Msun / yr] and mass formed per bin [Msun].
        """
        p = resolve_params(self.recipe, params)
        bins = self.agebins(p[self.recipe["redshift_param"]])
        ratios = [p[n] for n in self.recipe["sfh"]["logsfr_ratios"]]
        sfr, mass = continuity_sfh(p[self.recipe["mass_param"]], ratios, bins)
        return bins, sfr, mass

    def galaxy(self, params: Mapping[str, float]) -> Any:
        """Synthesizer ``Galaxy`` for one parameter set, with rest-frame and observed spectra.

        Args:
            params: ``{name: value}`` covering the recipe's ``param_names`` (aliases of
                :mod:`arachne.emulator.spec` are followed; extra entries are ignored).
        """
        from synthesizer.parametric import SFH, Stars, ZDist
        from synthesizer.parametric.galaxy import Galaxy
        from unyt import Msun, unyt_array, unyt_quantity

        r = self.recipe
        p = resolve_params(r, params)
        z = p[r["redshift_param"]]
        sfh = SFH.Continuity(
            logsfr_ratios=np.array([p[n] for n in r["sfh"]["logsfr_ratios"]]),
            agebins=unyt_array(self.agebins(z), "Myr"),
        )
        zdist = getattr(ZDist, r["zdist"]["class"])(
            **{k: p[v] for k, v in r["zdist"]["kwargs"].items()}
        )
        attrs: dict[str, Any] = stellar_attributes(r, p)
        for attr, rule in r["stellar_params"].items():
            if isinstance(rule, Mapping) and rule.get("unit"):
                attrs[attr] = unyt_quantity(attrs[attr], rule["unit"])
        stars = Stars(
            log10ages=self.grid.log10ages,
            metallicities=self.grid.metallicities,
            sf_hist=sfh,
            metal_dist=zdist,
            initial_mass=10.0 ** p[r["mass_param"]] * Msun,
            **attrs,
        )
        gal = Galaxy(stars=stars, redshift=z)
        gal.stars.get_spectra(self.emission_model)
        gal.get_observed_spectra(self.cosmo, igm=self.igm)
        return gal

    # -- evaluation --------------------------------------------------------------
    def evaluate(
        self, params: Mapping[str, float], spectrum: bool = True, derived: bool = True
    ) -> dict[str, Any]:
        """Observed spectrum, photometry and derived quantities of one parameter set.

        Args:
            params: ``{name: value}``; see :meth:`galaxy`.
            spectrum: Return the observed spectrum.
            derived: Return the derived quantities.

        Returns:
            Dict with ``"photometry"`` ((B,) nJy, if bands were given),
            ``"wavelength_um"`` and ``"spectrum"`` ((L,) observed-frame micron and
            f_nu in nJy) and ``"derived"`` ((len(DERIVED_NAMES),) in
            :data:`DERIVED_NAMES` order).
        """
        gal = self.galaxy(params)
        sed = gal.stars.spectra[self.key]
        out: dict[str, Any] = {}
        if self.filters is not None:
            phot = sed.get_photo_fnu(self.filters).photo_fnu
            out["photometry"] = np.asarray(phot.to("nJy").value, dtype=float)
        if spectrum:
            out["wavelength_um"] = np.asarray(sed.obslam.to("um").value, dtype=float)
            out["spectrum"] = np.asarray(sed.fnu.to("nJy").value, dtype=float)
        if derived:
            out["derived"] = self._derived(gal, sed, params)
        return out

    def _derived(self, gal: Any, sed: Any, params: Mapping[str, float]) -> np.ndarray:
        p = resolve_params(self.recipe, params)
        stars = gal.stars
        values = {
            "log_mass_formed": p[self.recipe["mass_param"]],
            "log_mass_surviving": np.log10(gal.get_surviving_mass(self.grid).to_value("Msun")),
            "sfr_10": stars.calculate_average_sfr((0, 1e7)).to_value("Msun/yr"),
            "sfr_100": stars.calculate_average_sfr((0, 1e8)).to_value("Msun/yr"),
            "age_mass_weighted": stars.get_mass_weighted_age().to_value("Myr"),
        }
        for name, method in (("beta_uv", "measure_beta"), ("d4000", "measure_d4000")):
            try:
                values[name] = float(getattr(sed, method)())
            except Exception:  # noqa: BLE001 - e.g. no flux in the windows
                values[name] = np.nan
        return np.array([float(values[n]) for n in DERIVED_NAMES])

    @property
    def wavelength_um_rest(self) -> np.ndarray:
        """Rest-frame wavelength grid of the spectra, in micron."""
        return np.asarray(self.grid.lam.to("um").value, dtype=float)


def check_recipe(
    recipe: Mapping,
    grid_dir: str | Path | None = None,
    simulator: Simulator | None = None,
    rest_window_um: tuple[float, float] = (0.13, 5.0),
    min_flux_njy: float = 1.0,
) -> dict[str, Any]:
    """Re-simulate the recipe's reference library rows and compare.

    Args:
        recipe: Recipe with a ``reference`` block.
        grid_dir: Directory of the SPS grid.
        simulator: Reuse this simulator (it must have been built with the
            reference bands).
        rest_window_um: Rest-frame window the verdict is based on.  Bluewards the
            comparison is sensitive to the Lyman-alpha and IGM treatment, redwards
            to the dust-emission model; both are reported separately.
        min_flux_njy: Library fluxes below this are ignored.

    Returns:
        Dict with ``status`` (``"exact"``: median deviation below 0.05 % and 90th
        percentile below 0.5 %; ``"approximate"``: median below 2 %; else
        ``"failed"``), the deviation statistics in and outside the window, per
        band medians, derived-quantity differences and the versions used.
    """
    ref = recipe.get("reference")
    if not ref:
        raise ValueError("the recipe has no reference rows to check against")
    bands = list(ref["band_names"])
    sim = simulator or Simulator(recipe, grid_dir=grid_dir, band_names=bands)
    if list(sim.band_names) != bands:
        raise ValueError("the simulator was not built with the recipe's reference bands")
    names = list(recipe["param_names"])
    params = np.asarray(ref["params"], dtype=float)
    lib = np.asarray(ref["photometry_nJy"], dtype=float)
    mine = np.zeros_like(lib)
    derived = np.zeros((len(params), len(DERIVED_NAMES)))
    for i, row in enumerate(params):
        out = sim.evaluate(dict(zip(names, row)), spectrum=False)
        mine[i], derived[i] = out["photometry"], out["derived"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(lib > min_flux_njy, np.abs(mine / lib - 1.0), np.nan)
    z = params[:, names.index(recipe["redshift_param"])]
    rest = sim.pivot_um[None, :] / (1.0 + z[:, None])
    inside = (rest >= rest_window_um[0]) & (rest <= rest_window_um[1])

    def stats(mask: np.ndarray) -> dict[str, float]:
        vals = dev[mask & np.isfinite(dev)]
        if vals.size == 0:
            return {"n": 0}
        return {
            "n": int(vals.size),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "max": float(vals.max()),
        }

    window = stats(inside)
    if window.get("n", 0) == 0:
        status = "failed"
    elif window["median"] < 5e-4 and window["p90"] < 5e-3:
        status = "exact"
    elif window["median"] < 2e-2:
        status = "approximate"
    else:
        status = "failed"
    derived_diff = {}
    for name, lib_vals in ref.get("derived", {}).items():
        ours = derived[:, DERIVED_NAMES.index(name)]
        lib_vals = np.asarray(lib_vals, dtype=float)
        if ADDITIVE[name] == "logsum":
            derived_diff[name] = {"max_abs_dex": float(np.nanmax(np.abs(ours - lib_vals)))}
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(ours / lib_vals - 1.0)
            derived_diff[name] = {"max_rel": float(np.nanmax(rel[np.isfinite(rel)]))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_band = {
            b: float(np.nanmedian(np.where(inside, dev, np.nan)[:, j])) for j, b in enumerate(bands)
        }
    return {
        "status": status,
        "n_rows": int(len(params)),
        "window_um": list(rest_window_um),
        "in_window": window,
        "blueward": stats(rest < rest_window_um[0]),
        "redward": stats(rest > rest_window_um[1]),
        "per_band_median": per_band,
        "derived": derived_diff,
        "synthesizer_version": sim.synthesizer_version,
        "grid": str(sim.grid_dir / f"{recipe['grid_name']}.hdf5"),
    }
