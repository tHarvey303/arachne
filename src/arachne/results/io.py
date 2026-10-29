"""Write :class:`~arachne.results.schema.BundleContent` to an HDF5 bundle (numpy + h5py only).

The content is first flattened into a :class:`ContentTree` (dataset path -> array, group or
dataset path -> attrs) that mirrors the file exactly.  :func:`write_content` writes that tree;
:meth:`arachne.results.reader.ResultsBundle.from_content` reads the same tree from memory, so
in-memory and file-backed bundles behave identically.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
from dataclasses import dataclass, field
from importlib import metadata as _ilm
from pathlib import Path
from typing import Any

import numpy as np

from .regions import Region
from .schema import (
    PARAM_SUMMARY_KEYS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    BundleContent,
)

__all__ = ["ContentTree", "content_to_tree", "write_content", "round_mantissa", "STRUCTURAL_ATTRS"]

STRUCTURAL_ATTRS = (
    "schema",
    "schema_version",
    "kind",
    "grid",
    "forward_model",
    "spatial_model",
    "posterior_source",
    "n_draws",
    "degraded",
    "title",
    "redshift",
)
"""Root attributes that describe the bundle; every other root attr is provenance."""

_POSTERIOR_ATTR_KEYS = ("sampler", "n_chains", "n_samples_total", "log_evidence_laplace")


@dataclass
class ContentTree:
    """Flat mirror of the HDF5 layout.

    Attributes:
        data: Dataset path (no leading slash) -> numpy array or ``str`` (scalar string).
        attrs: Group or dataset path (``""`` = root) -> attribute dict.
        draws: Paths of draw datasets (chunked ``(1, ...)``, eligible for mantissa rounding).
    """

    data: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, dict[str, Any]] = field(default_factory=dict)
    draws: set[str] = field(default_factory=set)

    def put(self, path: str, value: Any, *, dtype: Any = None, draws: bool = False) -> None:
        """Add a dataset unless ``value`` is None."""
        if value is None:
            return
        if not isinstance(value, str):
            value = np.asarray(value, dtype=dtype)
        self.data[path] = value
        if draws:
            self.draws.add(path)

    def set_attrs(self, path: str, **attrs: Any) -> None:
        """Merge attributes into a group or dataset (None values are dropped)."""
        self.attrs.setdefault(path, {}).update({k: v for k, v in attrs.items() if v is not None})


def _f4(a: Any) -> np.ndarray | None:
    return None if a is None else np.asarray(a, dtype=np.float32)


def _summaries(draws: np.ndarray) -> dict[str, np.ndarray]:
    d = np.asarray(draws, dtype=np.float64)
    p16, p50, p84 = np.percentile(d, [16, 50, 84], axis=0)
    return {
        "p16": p16.astype(np.float32),
        "p50": p50.astype(np.float32),
        "p84": p84.astype(np.float32),
        "mean": d.mean(axis=0).astype(np.float32),
        "std": d.std(axis=0).astype(np.float32),
    }


def _attr_value(v: Any) -> Any:
    """Make a provenance value storable as an HDF5 attribute."""
    if isinstance(v, (str, bool, int, float, np.generic)):
        return v
    if isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
        return v
    try:
        return json.dumps(v, default=str)
    except TypeError:
        return str(v)


def _version(dist: str) -> str:
    try:
        return _ilm.version(dist)
    except _ilm.PackageNotFoundError:
        return ""


def default_provenance() -> dict[str, Any]:
    """Cheap provenance defaults that need neither git nor JAX.

    ``writer.py`` overrides them with ``collect_provenance()``.
    """
    return {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "arachne_version": _version("arachne"),
        "git_commit": "",
        "git_dirty": False,
        "command_line": "",
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "jax_version": _version("jax"),
        "numpy_version": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }


def content_to_tree(content: BundleContent) -> ContentTree:
    """Flatten validated content into the HDF5 layout, computing draw summaries.

    Args:
        content: Bundle content (validated here).

    Returns:
        The :class:`ContentTree`.
    """
    content.validate()
    c = content
    t = ContentTree()
    s = c.n_draws
    nb = len(c.band_names)
    h, w = np.shape(c.flux)[1:]

    root = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": "image",
        "grid": "single",
        "forward_model": c.forward_model,
        "spatial_model": c.spatial_model,
        "posterior_source": c.posterior_source,
        "n_draws": int(s),
        "degraded": bool(c.degraded),
        "title": c.title,
        "redshift": float(c.redshift),
    }
    prov = default_provenance()
    prov.update(c.provenance or {})
    for k, v in prov.items():
        if k not in root and v is not None:
            root[k] = _attr_value(v)
    t.set_attrs("", **root)

    # observation / likelihood
    t.set_attrs(
        "observation",
        band_names=[str(b) for b in c.band_names],
        pivot_um=np.asarray(c.pivot_um, dtype=np.float64).reshape(nb),
        pixel_scale_arcsec=float(c.pixel_scale),
        image_shape=np.asarray([h, w], dtype=np.int32),
        flux_unit="nJy",
        target_name="",
        ra=float("nan"),
        dec=float("nan"),
    )
    t.put("observation/flux", _f4(c.flux))
    t.put("observation/variance", _f4(c.variance))
    t.put("observation/mask", _f4(c.mask))
    t.put("observation/psf", _f4(c.psf))
    t.set_attrs(
        "likelihood",
        model_error_frac=np.asarray(c.model_error_frac, dtype=np.float64).reshape(nb),
        likelihood="gaussian",
    )

    # model
    t.set_attrs("model", draw_indices_source="posterior/draw_index")
    for frame, best, draws in (
        ("convolved", c.convolved_map, c.convolved_draws),
        ("intrinsic", c.intrinsic_map, c.intrinsic_draws),
    ):
        if best is None:
            continue
        t.put(f"model/{frame}/map", _f4(best))
        if draws is not None and s > 0:
            draws = _f4(draws)
            t.put(f"model/{frame}/draws", draws, draws=True)
            for stat, arr in _summaries(draws).items():
                t.put(f"model/{frame}/{stat}", arr)
    for key, arr in (c.nuisance or {}).items():
        t.put(f"model/nuisance/{key}", np.asarray(arr, dtype=np.float64))

    # params
    t.set_attrs(
        "params",
        map_names=[str(n) for n in c.map_names],
        scalar_names=[str(n) for n in c.scalar_names],
        fixed_names=[str(n) for n in c.fixed_params],
        fixed_values=np.asarray(list(c.fixed_params.values()), dtype=np.float64),
        mass_param=c.mass_param,
        mass_is_log10=True,
    )
    t.put("params/labels", json.dumps(dict(c.labels)))
    t.put("params/units", json.dumps(dict(c.units)))
    if c.map_names:
        t.put("params/maps/map", _f4(c.param_maps_map))
        computed: dict[str, np.ndarray] = {}
        if c.param_maps_draws is not None and s > 0:
            pdraws = _f4(c.param_maps_draws)
            t.put("params/maps/draws", pdraws, draws=True)
            computed = _summaries(pdraws)
        if c.prior_std_maps is not None:
            prior_std = _f4(c.prior_std_maps)
            t.put("params/maps/prior_std", prior_std)
            if "std" in computed:
                with np.errstate(divide="ignore", invalid="ignore"):
                    info = 1.0 - (computed["std"] / prior_std) ** 2
                computed["information"] = info.astype(np.float32)
        for key, arr in (c.param_summaries or {}).items():
            computed.setdefault(key, _f4(arr))
        for key in PARAM_SUMMARY_KEYS:
            if key in computed:
                t.put(f"params/maps/{key}", computed[key])
    if c.scalar_names:
        t.put("params/scalars/map", np.asarray(c.scalars_map, dtype=np.float64))
        if c.scalars_draws is not None and s > 0:
            t.put("params/scalars/draws", np.asarray(c.scalars_draws, dtype=np.float64))

    # components
    cc = c.components
    if cc is not None:
        k = len(cc.profile_names)
        t.set_attrs(
            "components",
            n_components=int(k),
            profile_names=[str(n) for n in cc.profile_names],
            param_names=[str(n) for n in cc.param_names],
            factorised=bool(cc.factorised),
            ordering=cc.ordering,
        )
        t.put("components/draw_index", cc.draw_index, dtype=np.int32)
        t.put("components/sed_intrinsic/map", cc.sed_intrinsic_map, dtype=np.float64)
        t.put("components/sed_intrinsic/draws", cc.sed_intrinsic_draws, dtype=np.float64)
        t.put("components/params/map", cc.params_map, dtype=np.float64)
        t.put("components/params/draws", cc.params_draws, dtype=np.float64)
        for name, best, draws in (
            ("profiles", cc.profiles_map, cc.profiles_draws),
            ("intrinsic", cc.intrinsic_map, cc.intrinsic_draws),
            ("convolved", cc.convolved_map, cc.convolved_draws),
        ):
            t.put(f"components/{name}/map", _f4(best))
            t.put(f"components/{name}/draws", _f4(draws), draws=True)

    # truth
    tr = c.truth
    if tr is not None:
        t.put("truth/convolved", _f4(tr.convolved))
        t.put("truth/intrinsic", _f4(tr.intrinsic))
        t.put("truth/param_maps", _f4(tr.param_maps))
        t.put("truth/scalars", tr.scalars, dtype=np.float64)
        t.put("truth/component_seds", tr.component_seds, dtype=np.float64)
        t.put("truth/theta", tr.theta)
        t.set_attrs("truth")

    # posterior
    diag = dict(c.diagnostics or {})
    n_chains = diag.pop("n_chains", None)
    if n_chains is None:
        n_chains = int(len(np.unique(c.chain_id))) if c.chain_id is not None else 1
    post_attrs = {
        "sampler": diag.pop("sampler", c.posterior_source),
        "n_chains": int(n_chains),
        "n_samples_total": int(diag.pop("n_samples_total", s)),
        "thinning": "even",
    }
    for key in _POSTERIOR_ATTR_KEYS:
        if key in diag:
            post_attrs[key] = diag.pop(key)
    t.set_attrs("posterior", **post_attrs)
    t.put("posterior/theta_map", c.theta_map)
    t.put("posterior/theta_draws", c.theta_draws if s > 0 else None)
    t.put("posterior/draw_index", c.draw_index, dtype=np.int64)
    t.put("posterior/chain_id", c.chain_id, dtype=np.int16)
    diag_scalars = {}
    for key, val in diag.items():
        arr = np.asarray(val)
        if arr.ndim == 0:
            diag_scalars[key] = _attr_value(arr.item() if isinstance(val, np.ndarray) else val)
        else:
            t.put(f"posterior/diagnostics/{key}", arr)
    t.set_attrs("posterior/diagnostics", **diag_scalars)

    # regions / metrics / extra
    for name, reg in c.regions.items():
        if not isinstance(reg, Region):
            reg = Region.from_mask(np.asarray(reg), name=name)
        path = f"regions/{name}"
        t.put(path, reg.mask.astype(np.uint8))
        t.set_attrs(path, kind=reg.kind, geometry=json.dumps(reg.geometry))
    if c.metrics is not None:
        t.put("metrics/json", json.dumps(c.metrics, default=_json_default))
    for name, arr in c.extra.items():
        t.put(f"extra/{name}", np.asarray(arr))
    return t


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def round_mantissa(arr: np.ndarray, bits: int) -> np.ndarray:
    """Round float32 values to ``bits`` explicit mantissa bits (round half up, in place safe).

    The relative error is at most ``2**-(bits + 1)``.  Non-finite values are unchanged.

    Args:
        arr: Array (converted to float32).
        bits: Mantissa bits to keep, 1..23.

    Returns:
        Rounded float32 copy.
    """
    if not 1 <= int(bits) <= 23:
        raise ValueError(f"mantissa_bits must be in 1..23, got {bits}")
    a = np.array(arr, dtype=np.float32, copy=True)
    drop = 23 - int(bits)
    if drop == 0:
        return a
    u = a.view(np.uint32)
    finite = np.isfinite(a)
    half = np.uint32(1 << (drop - 1))
    keep = np.uint32((0xFFFFFFFF << drop) & 0xFFFFFFFF)
    u[finite] = (u[finite] + half) & keep
    return a


def _write_attrs(obj: Any, attrs: dict[str, Any]) -> None:
    import h5py

    str_dt = h5py.string_dtype()
    for key, val in attrs.items():
        if isinstance(val, (list, tuple)):
            obj.attrs.create(key, np.asarray([str(v) for v in val], dtype=object), dtype=str_dt)
        elif isinstance(val, str):
            obj.attrs.create(key, val, dtype=str_dt)
        else:
            obj.attrs[key] = val


def write_content(
    path: str | Path,
    content: BundleContent,
    *,
    mantissa_bits: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a results bundle.

    Cubes are gzip(4)+shuffle compressed; draw datasets are chunked ``(1, ...)``.  With
    ``mantissa_bits`` the draw datasets (only) are rounded to that many mantissa bits after the
    summaries have been computed from the unrounded values.  The file is written to a
    temporary name and moved into place.

    Args:
        path: Output path (``.h5``).
        content: Bundle content.
        mantissa_bits: Optional lossy rounding of float32 draws (12 gives ~1e-4 relative).
        overwrite: Replace an existing file.

    Returns:
        The output path.
    """
    import h5py

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True)")
    tree = content_to_tree(content)
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype()
    try:
        with h5py.File(tmp, "w") as f:
            _write_attrs(f, tree.attrs.get("", {}))
            for dpath, val in tree.data.items():
                if isinstance(val, str):
                    ds = f.create_dataset(dpath, data=val, dtype=str_dt)
                else:
                    kw: dict[str, Any] = {}
                    is_draws = dpath in tree.draws
                    if is_draws and mantissa_bits is not None and val.dtype == np.float32:
                        val = round_mantissa(val, mantissa_bits)
                    if val.ndim >= 2 and val.size > 0 and val.dtype.kind in "fiu":
                        kw = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
                        if is_draws and val.ndim >= 3:
                            kw["chunks"] = (1, *val.shape[1:])
                    ds = f.create_dataset(dpath, data=val, **kw)
                _write_attrs(ds, tree.attrs.get(dpath, {}))
            for gpath, attrs in tree.attrs.items():
                if gpath == "" or gpath in tree.data:
                    continue
                grp = f.require_group(gpath)
                _write_attrs(grp, attrs)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
