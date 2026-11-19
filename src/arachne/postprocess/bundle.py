"""Post-process a results bundle with the full SPS model.

:func:`postprocess_bundle` takes posterior draws of the parameter maps from a
bundle, evaluates the Synthesizer model of the emulator's training library on
every pixel of a region (:class:`~arachne.postprocess.simulator.Simulator`) and
stores, in a ``postprocess`` group of the same file:

- ``sed/<region>``: the observed-frame spectrum summed over each bundle region,
  per draw and as percentiles (the *intrinsic* light of the pixels: no PSF);
- ``photometry/<region>``: the same through the bundle's filters;
- ``maps/<name>``: per-pixel derived quantities (formed and surviving stellar
  mass, SFR over 10 and 100 Myr, mass-weighted age, UV slope, D4000);
- ``totals/<region>/<name>``: their region totals per draw;
- ``sfh/<region>``: the SFH in physical units (SFR per age bin);
- ``emulator_check``: simulator / emulator flux ratio per band over the pixels,
  a direct measure of the emulator error on this posterior.

:func:`load_postprocess` reads the group back.  No JAX is needed; Synthesizer and
the SPS grid are.  Cost: about 15 ms per pixel and draw on one core.
"""

from __future__ import annotations

import datetime as _dt
import logging
import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .recipe import continuity_agebins, recipe_from_checkpoint, recipe_hash
from .simulator import ADDITIVE, DERIVED_NAMES, DERIVED_UNITS, Simulator, check_recipe

logger = logging.getLogger(__name__)

__all__ = ["GROUP", "PostprocessResult", "load_postprocess", "postprocess_bundle"]

GROUP = "postprocess"
_PCT = {"p16": 16.0, "p50": 50.0, "p84": 84.0}

_SIM: Simulator | None = None


def _init_worker(recipe: Mapping, grid_dir: str | None, bands: Sequence[str]) -> None:
    global _SIM  # noqa: PLW0603 - one simulator per worker process
    _SIM = Simulator(recipe, grid_dir=grid_dir, band_names=bands)


def _work(task: tuple) -> tuple:
    """Evaluate one chunk of pixels: photometry, derived values and region spectrum sums."""
    draw, idx, params, member = task
    sim = _SIM
    names = sim.recipe["param_names"]
    phot = np.zeros((len(idx), len(sim.band_names)))
    derived = np.zeros((len(idx), len(DERIVED_NAMES)))
    sums, lam = None, None
    for n, row in enumerate(params):
        out = sim.evaluate(dict(zip(names, row)))
        phot[n], derived[n] = out["photometry"], out["derived"]
        if sums is None:
            lam = out["wavelength_um"]
            sums = np.zeros((member.shape[0], lam.size))
        sums[member[:, n]] += out["spectrum"]
    return draw, idx, phot, derived, sums, lam


def _param_sources(bundle: Any, recipe: Mapping) -> dict[str, tuple[str, str]]:
    """Where each recipe parameter comes from in the bundle: ``{name: (kind, bundle name)}``."""
    from arachne.emulator.spec import param_info

    have = {n: "map" for n in bundle.map_names}
    have.update({n: "scalar" for n in bundle.scalar_names})
    have.update({n: "fixed" for n in bundle.fixed_params})
    out, missing = {}, []
    for name in recipe["param_names"]:
        info = param_info(name)
        for cand in (name, info.name, *info.aliases):
            if cand in have:
                out[name] = (have[cand], cand)
                break
        else:
            missing.append(name)
    if missing:
        raise KeyError(
            f"the bundle does not hold the simulator inputs {missing} (maps {bundle.map_names}, "
            f"scalars {bundle.scalar_names}, fixed {sorted(bundle.fixed_params)})"
        )
    return out


def _param_cube(bundle: Any, recipe: Mapping, draws: np.ndarray | None) -> np.ndarray:
    """Library parameters on the pixel grid: (D, P, H, W) in recipe order."""
    src = _param_sources(bundle, recipe)
    h, w = bundle.image_shape
    n_d = 1 if draws is None else len(draws)
    cube = np.zeros((n_d, len(recipe["param_names"]), h, w))
    scalars = bundle.scalar_draws() if draws is not None else {}
    for j, name in enumerate(recipe["param_names"]):
        kind, key = src[name]
        if kind == "fixed":
            cube[:, j] = bundle.fixed_params[key]
        elif kind == "map":
            if draws is None:
                cube[0, j] = bundle.param_map(key, "map")
            else:
                cube[:, j] = np.asarray(bundle.param_draws(key))[draws]
        elif draws is None:
            best = np.atleast_1d(bundle._opt("params/scalars/map"))
            cube[0, j] = float(best[bundle.scalar_names.index(key)])
        else:
            cube[:, j] = np.asarray(scalars[key])[draws][:, None, None]
    return cube


def _combine(name: str, values: np.ndarray, weights: np.ndarray) -> float:
    """Region total of one derived quantity from per-pixel values (see ``ADDITIVE``)."""
    how = ADDITIVE[name]
    if how == "logsum":
        return float(np.log10(np.sum(10.0**values)))
    if how == "sum":
        return float(np.sum(values))
    if how == "mass_mean":
        return float(np.sum(values * weights) / np.sum(weights))
    return np.nan


def postprocess_bundle(  # noqa: C901
    bundle_path: str | Path,
    recipe: Mapping | None = None,
    emulator: str | Path | None = None,
    grid_dir: str | Path | None = None,
    region: str | None = None,
    extra_regions: Mapping[str, np.ndarray] | None = None,
    n_draws: int = 16,
    n_proc: int = 1,
    chunk: int = 64,
    check: bool = True,
    strict: bool = True,
    write: bool = True,
) -> "PostprocessResult":
    """Add simulator spectra and derived quantities to a results bundle.

    Args:
        bundle_path: The bundle (``results.h5``); the ``postprocess`` group is
            written into it (replacing an earlier one).
        recipe: Simulator recipe.  Default: read from ``emulator``.
        emulator: Emulator checkpoint the fit used (source of the recipe).
        grid_dir: Directory of the SPS grid
            (see :func:`arachne.postprocess.simulator.find_grid`).
        region: Bundle region whose pixels are processed.  Default: ``"segment"``
            if the bundle has one, else every pixel.  Spectra and totals are
            stored for every bundle region inside it.
        extra_regions: ``{name: (H, W) boolean mask}`` of further regions to store
            spectra, photometry, totals and SFHs for (apertures on a clump or the
            core, say).  Pixels outside the processed region are dropped.
        n_draws: Posterior draws to process, spread evenly over the stored draws
            (all of them if there are fewer).  A bundle without draws uses its MAP.
        n_proc: Worker processes.
        chunk: Pixels per task.
        check: First re-simulate the recipe's reference library rows.
        strict: Raise unless that check is ``"exact"``.
        write: Write the result into the bundle (``False``: only return it).

    Returns:
        The :class:`PostprocessResult`.

    Raises:
        RuntimeError: If ``strict`` and the simulator does not reproduce the library.
    """
    from arachne.results.reader import load_bundle

    bundle_path = Path(bundle_path)
    if recipe is None:
        if emulator is None:
            raise ValueError("pass a simulator recipe or the emulator checkpoint that holds one")
        recipe = recipe_from_checkpoint(emulator)
        if recipe is None:
            raise ValueError(
                f"{emulator} carries no simulator recipe; add one with "
                "python -m arachne.emulator annotate CKPT --simulator-library LIB.hdf5"
            )
    bundle = load_bundle(bundle_path)
    bands = list(bundle.band_names)
    h, w = bundle.image_shape

    report = None
    if check and recipe.get("reference"):
        report = check_recipe(recipe, grid_dir=grid_dir)
        logger.info("simulator check: %s (%s)", report["status"], report["in_window"])
        if strict and report["status"] != "exact":
            raise RuntimeError(
                f"the simulator does not reproduce the library ({report['status']}: "
                f"{report['in_window']}); check the grid file and Synthesizer version, or pass "
                "strict=False"
            )

    if region is None:
        region = "segment" if "segment" in bundle.regions else None
    mask = bundle.regions[region].mask if region else np.ones((h, w), dtype=bool)
    region_name = region or "all"
    regions = {region_name: mask}
    for name, reg in bundle.regions.items():
        if name != region_name and reg.mask.any() and not (reg.mask & ~mask).any():
            regions[name] = reg.mask
    for name, extra in (extra_regions or {}).items():
        extra = np.asarray(extra, dtype=bool) & mask
        if extra.shape != (h, w) or not extra.any():
            raise ValueError(f"extra region {name!r} has no pixel inside region {region_name!r}")
        regions[str(name)] = extra
    yy, xx = np.nonzero(mask)
    member = np.stack([m[yy, xx] for m in regions.values()])  # (R, Npix)

    if bundle.has_param_draws:
        stored = bundle.n_draws
        draws = np.unique(np.linspace(0, stored - 1, min(n_draws, stored)).round().astype(int))
    else:
        draws = None
    cube = _param_cube(bundle, recipe, draws)  # (D, P, H, W)
    n_d = cube.shape[0]
    flat = cube[:, :, yy, xx].transpose(0, 2, 1)  # (D, Npix, P)

    tasks = []
    for d in range(n_d):
        for start in range(0, len(yy), chunk):
            idx = np.arange(start, min(start + chunk, len(yy)))
            tasks.append((d, idx, flat[d, idx], member[:, idx]))
    logger.info(
        "post-processing %d draw(s) x %d pixels with %d process(es)", n_d, len(yy), max(n_proc, 1)
    )
    grid_arg = str(grid_dir) if grid_dir is not None else None
    if n_proc > 1:
        # one thread per worker: the per-pixel work is small and oversubscribes otherwise
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_proc, initializer=_init_worker, initargs=(recipe, grid_arg, bands)) as pool:
            results = list(pool.imap_unordered(_work, tasks))
    else:
        _init_worker(recipe, grid_arg, bands)
        results = [_work(t) for t in tasks]

    lam = results[0][5]
    phot = np.zeros((n_d, len(yy), len(bands)))
    derived = np.zeros((n_d, len(yy), len(DERIVED_NAMES)))
    spectra = np.zeros((n_d, len(regions), lam.size))
    for d, idx, ph, de, sums, _ in results:
        phot[d, idx], derived[d, idx] = ph, de
        spectra[d] += sums

    res = PostprocessResult(
        region=region_name,
        regions={k: v.copy() for k, v in regions.items()},
        draw_index=np.arange(1) if draws is None else draws,
        band_names=bands,
        wavelength_um=lam,
        attrs={
            "recipe_hash": recipe_hash(recipe),
            "library": str(recipe.get("library", "")),
            "grid_name": str(recipe["grid_name"]),
            "mass_kind": str(recipe.get("mass_kind", "formed")),
            "from_draws": draws is not None,
            "check_status": report["status"] if report else "not run",
            "check_median": float(report["in_window"].get("median", np.nan)) if report else np.nan,
            "synthesizer_version": report["synthesizer_version"] if report else "",
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
    )
    formed = 10.0 ** derived[..., DERIVED_NAMES.index("log_mass_formed")]
    for r, (name, m) in enumerate(regions.items()):
        sel = member[r]
        res.sed[name] = spectra[:, r]
        res.photometry[name] = phot[:, sel].sum(axis=1)
        res.totals[name] = {
            q: np.array([_combine(q, derived[d, sel, k], formed[d, sel]) for d in range(n_d)])
            for k, q in enumerate(DERIVED_NAMES)
            if ADDITIVE[q]
        }
    for k, q in enumerate(DERIVED_NAMES):
        full = np.full((n_d, h, w), np.nan, dtype=np.float32)
        full[:, yy, xx] = derived[..., k]
        res.maps[q] = full

    # SFH in physical units (analytic; the redshift is one number in image fits)
    names = list(recipe["param_names"])
    z_all = flat[..., names.index(recipe["redshift_param"])]
    if np.ptp(z_all) == 0:
        from astropy.cosmology import Cosmology

        cosmo = Cosmology.from_format(recipe["cosmology"], format="yaml")
        bins = continuity_agebins(float(z_all.flat[0]), cosmo, recipe["sfh"]["agebins"])
        ratios = flat[..., [names.index(n) for n in recipe["sfh"]["logsfr_ratios"]]]
        widths = (bins[:, 1] - bins[:, 0]) * 1e6
        rel = 10.0 ** np.concatenate(
            [np.zeros(ratios.shape[:-1] + (1,)), -np.cumsum(ratios, axis=-1)], axis=-1
        )
        mass_bin = rel * widths
        mass_bin *= (formed / mass_bin.sum(axis=-1))[..., None]  # (D, Npix, n_bins)
        res.sfh_agebins_myr = bins
        for r, name in enumerate(regions):
            res.sfh[name] = mass_bin[:, member[r]].sum(axis=1) / widths

    if draws is not None and bundle.has_intrinsic:
        emu = np.asarray(bundle.model_draws("intrinsic"))[draws][:, :, yy, xx].transpose(0, 2, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(emu > 0, phot / emu, np.nan)
        res.emulator_check = {
            k: np.nanpercentile(ratio.reshape(-1, len(bands)), q, axis=0) for k, q in _PCT.items()
        }
        res.emulator_check["total"] = np.median(phot.sum(axis=1) / emu.sum(axis=1), axis=0)
    bundle.close()
    if write:
        res.write(bundle_path)
    return res


@dataclass
class PostprocessResult:
    """Simulator spectra and derived quantities of one bundle.

    Arrays with a leading ``D`` axis are per processed draw (``draw_index`` gives
    the stored-draw index of each; ``D = 1`` and ``attrs["from_draws"] = False``
    for a MAP-only bundle).  Use :meth:`stat` for percentiles.

    Attributes:
        region: Name of the processed region.
        regions: ``{name: (H, W) mask}`` of the regions with spectra and totals.
        draw_index: (D,) indices into the bundle's stored draws.
        band_names: Bundle bands (B,).
        wavelength_um: (L,) observed-frame wavelengths of the spectra.
        sed: ``{region: (D, L)}`` f_nu in nJy summed over the region (no PSF).
        photometry: ``{region: (D, B)}`` the same through the filters.
        maps: ``{name: (D, H, W)}`` derived quantities per pixel (NaN outside).
        totals: ``{region: {name: (D,)}}`` region totals of the additive ones.
        sfh: ``{region: (D, n_bins)}`` SFR per age bin [Msun / yr].
        sfh_agebins_myr: (n_bins, 2) lookback-time bin edges, or ``None``.
        emulator_check: Simulator / emulator flux ratio per band: ``p16``,
            ``p50``, ``p84`` over pixels and draws, and ``total`` (region sums).
        attrs: Provenance (recipe hash, grid, library check, versions).
    """

    region: str
    regions: dict[str, np.ndarray]
    draw_index: np.ndarray
    band_names: list[str]
    wavelength_um: np.ndarray
    sed: dict[str, np.ndarray] = field(default_factory=dict)
    photometry: dict[str, np.ndarray] = field(default_factory=dict)
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    totals: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    sfh: dict[str, np.ndarray] = field(default_factory=dict)
    sfh_agebins_myr: np.ndarray | None = None
    emulator_check: dict[str, np.ndarray] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    units = DERIVED_UNITS

    @property
    def n_draws(self) -> int:
        """Number of processed draws."""
        return int(len(self.draw_index))

    @staticmethod
    def stat(draws: np.ndarray, stat: str = "p50") -> np.ndarray:
        """Percentile (``"p16"``, ``"p50"``, ``"p84"``), ``"mean"`` or ``"std"`` over draws."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if stat in _PCT:
                return np.nanpercentile(draws, _PCT[stat], axis=0)
            return getattr(np, f"nan{stat}")(draws, axis=0)

    def total(self, name: str, region: str | None = None) -> tuple[float, float, float]:
        """``(p16, p50, p84)`` of a region total (default: the processed region)."""
        vals = self.totals[region or self.region][name]
        return tuple(float(np.percentile(vals, q)) for q in _PCT.values())

    def write(self, bundle_path: str | Path) -> None:
        """Write the ``postprocess`` group into a bundle file, replacing an existing one."""
        import h5py

        with h5py.File(bundle_path, "a") as f:
            if GROUP in f:
                del f[GROUP]
            g = f.create_group(GROUP)
            for k, v in self.attrs.items():
                g.attrs[k] = v
            g.attrs["region"] = self.region
            g.attrs["band_names"] = [str(b) for b in self.band_names]
            g.attrs["derived_names"] = list(self.maps)
            g.create_dataset("draw_index", data=np.asarray(self.draw_index, dtype=np.int64))
            g.create_dataset("wavelength_um", data=self.wavelength_um.astype(np.float32))

            def put(path: str, draws: np.ndarray, **kw: Any) -> None:
                g.create_dataset(f"{path}/draws", data=draws.astype(np.float32), **kw)
                for s in _PCT:
                    g.create_dataset(f"{path}/{s}", data=self.stat(draws, s).astype(np.float32))

            for name, m in self.regions.items():
                g.create_dataset(f"regions/{name}", data=m.astype(np.uint8), compression="gzip")
                put(f"sed/{name}", self.sed[name], compression="gzip")
                put(f"photometry/{name}", self.photometry[name])
                for q, vals in self.totals[name].items():
                    g.create_dataset(f"totals/{name}/{q}/draws", data=vals)
                if name in self.sfh:
                    put(f"sfh/{name}", self.sfh[name])
            for q, cube in self.maps.items():
                put(f"maps/{q}", cube, compression="gzip")
                g[f"maps/{q}"].attrs["unit"] = DERIVED_UNITS.get(q, "")
            if self.sfh_agebins_myr is not None:
                g.create_dataset("sfh_agebins_myr", data=self.sfh_agebins_myr)
            for k, v in self.emulator_check.items():
                g.create_dataset(f"emulator_check/{k}", data=np.asarray(v, dtype=np.float32))


def load_postprocess(bundle_path: str | Path) -> PostprocessResult | None:
    """Read the ``postprocess`` group of a bundle.

    Args:
        bundle_path: The bundle file.

    Returns:
        The :class:`PostprocessResult`, or ``None`` if the bundle has not been
        post-processed.
    """
    import h5py

    def text(v: Any) -> Any:
        return v.decode() if isinstance(v, bytes) else v

    with h5py.File(bundle_path, "r") as f:
        if GROUP not in f:
            return None
        g = f[GROUP]
        attrs = {k: text(v) for k, v in g.attrs.items()}
        names = [text(x) for x in attrs.pop("derived_names")]
        res = PostprocessResult(
            region=str(attrs.pop("region")),
            regions={k: g[f"regions/{k}"][()].astype(bool) for k in g["regions"]},
            draw_index=g["draw_index"][()],
            band_names=[text(b) for b in attrs.pop("band_names")],
            wavelength_um=g["wavelength_um"][()].astype(float),
            attrs=attrs,
        )
        for name in res.regions:
            res.sed[name] = g[f"sed/{name}/draws"][()].astype(float)
            res.photometry[name] = g[f"photometry/{name}/draws"][()].astype(float)
            res.totals[name] = {q: g[f"totals/{name}/{q}/draws"][()] for q in g[f"totals/{name}"]}
            if f"sfh/{name}" in g:
                res.sfh[name] = g[f"sfh/{name}/draws"][()].astype(float)
        for q in names:
            res.maps[q] = g[f"maps/{q}/draws"][()]
        if "sfh_agebins_myr" in g:
            res.sfh_agebins_myr = g["sfh_agebins_myr"][()]
        if "emulator_check" in g:
            res.emulator_check = {k: g[f"emulator_check/{k}"][()] for k in g["emulator_check"]}
    return res
