"""Read ``arachne.results`` bundles (numpy and h5py only).

:func:`load_bundle` opens an HDF5 file; :meth:`ResultsBundle.from_content` wraps in-memory
:class:`~arachne.results.schema.BundleContent`.  Both go through the same flat layout
(:func:`arachne.results.io.content_to_tree`), so they behave identically.  Small datasets are
read eagerly; draw datasets are read on first use and memoised.

Plots and the viewer must only use the per-band accessors (:meth:`ResultsBundle.band_image`,
:meth:`ResultsBundle.band_draws`), never index a cube directly, so that the reserved
``grid="per_band"`` layout can be added later.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .io import STRUCTURAL_ATTRS, ContentTree, content_to_tree
from .metadata import DEFAULT_LABELS, short_band_name
from .regions import Region
from .schema import SCHEMA_NAME, SCHEMA_VERSION, SUMMARY_STATS, BundleContent, TruthContent

__all__ = ["ResultsBundle", "ComponentsView", "load_bundle"]

IMAGE_KINDS = (
    "data",
    "variance",
    "sigma",
    "mask",
    "model",
    "intrinsic",
    "residual",
    "chi",
    "truth_model",
    "truth_intrinsic",
    "truth_residual",
)
MODEL_STATS = ("map", *SUMMARY_STATS)
PARAM_STATS = (*MODEL_STATS, "truth", "diff", "z", "information", "prior_std")
_FALLBACK_STATS = ("p16", "p50", "p84", "mean")
_DATA_KINDS = ("data", "variance", "sigma", "residual", "chi", "truth_residual")


def _is_lazy(path: str) -> bool:
    return path.rsplit("/", 1)[-1] in ("draws", "theta_draws")


def _norm_attr(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, (list, tuple)):
        return [_norm_attr(x) for x in v]
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "OSU":
            return [x.decode() if isinstance(x, bytes) else str(x) for x in v.ravel()]
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


class _MemoryStore:
    """Store over an in-memory :class:`ContentTree`."""

    def __init__(self, tree: ContentTree):
        self._tree = tree
        self._paths = set(tree.data) | set(tree.attrs)

    def exists(self, path: str) -> bool:
        if path in self._paths:
            return True
        prefix = path + "/"
        return any(p.startswith(prefix) for p in self._paths)

    def is_dataset(self, path: str) -> bool:
        return path in self._tree.data

    def read(self, path: str) -> Any:
        return self._tree.data[path]

    def attrs(self, path: str) -> dict[str, Any]:
        return {k: _norm_attr(v) for k, v in self._tree.attrs.get(path, {}).items()}

    def children(self, group: str) -> list[str]:
        prefix = group + "/"
        names = []
        for p in sorted(self._paths):
            if p.startswith(prefix):
                head = p[len(prefix) :].split("/", 1)[0]
                if head and head not in names:
                    names.append(head)
        return names


class _H5Store:
    """Store over an HDF5 file: attrs and small datasets eager, draws lazy (file reopened)."""

    def __init__(self, path: Path):
        import h5py

        self.path = Path(path)
        self._attrs: dict[str, dict[str, Any]] = {}
        self._data: dict[str, Any] = {}
        self._lazy: set[str] = set()
        self._children: dict[str, list[str]] = {}
        with h5py.File(self.path, "r") as f:
            self._attrs[""] = {k: _norm_attr(v) for k, v in f.attrs.items()}
            self._children[""] = list(f.keys())

            def visit(name: str, obj: Any) -> None:
                self._attrs[name] = {k: _norm_attr(v) for k, v in obj.attrs.items()}
                if isinstance(obj, h5py.Group):
                    self._children[name] = list(obj.keys())
                elif _is_lazy(name):
                    self._lazy.add(name)
                elif h5py.check_string_dtype(obj.dtype) is not None:
                    val = obj.asstr()[()]
                    self._data[name] = val if isinstance(val, str) else np.asarray(val)
                else:
                    self._data[name] = obj[()]

            f.visititems(visit)

    def exists(self, path: str) -> bool:
        return path in self._attrs

    def is_dataset(self, path: str) -> bool:
        return path in self._data or path in self._lazy

    def read(self, path: str) -> Any:
        if path in self._data:
            return self._data[path]
        if path not in self._lazy:
            raise KeyError(path)
        import h5py

        with h5py.File(self.path, "r") as f:
            return f[path][()]

    def attrs(self, path: str) -> dict[str, Any]:
        return dict(self._attrs.get(path, {}))

    def children(self, group: str) -> list[str]:
        return list(self._children.get(group, []))


class ComponentsView:
    """Read-only view of the ``/components`` group of a bundle.

    Attributes:
        n_components: Number of components K.
        profile_names: (K,) profile names.
        param_names: (Pc,) parameter column names of :meth:`params`.
        factorised: True when component images are ``sed_k ⊗ profile_k``.
        ordering: ``"size"`` or ``"none"``.
        last_fallback: True when the last :meth:`image` call fell back to the best point.
    """

    def __init__(self, bundle: ResultsBundle):
        """Wrap the components group of ``bundle``."""
        self._b = bundle
        a = bundle._store.attrs("components")
        self.n_components = int(a.get("n_components", 0))
        self.profile_names = list(a.get("profile_names", []))
        self.param_names = list(a.get("param_names", []))
        self.factorised = bool(a.get("factorised", True))
        self.ordering = str(a.get("ordering", "none"))
        self.last_fallback = False

    def _has(self, path: str) -> bool:
        return self._b._store.is_dataset(f"components/{path}")

    def _read(self, path: str) -> np.ndarray:
        return self._b._read(f"components/{path}")

    @property
    def has_draws(self) -> bool:
        """True when per-draw component SEDs are stored."""
        return self._has("sed_intrinsic/draws")

    @property
    def draw_index(self) -> np.ndarray | None:
        """(Sc,) indices into the S draws used by the component cube draws."""
        return self._read("draw_index") if self._has("draw_index") else None

    def has_frame(self, frame: str) -> bool:
        """Whether best-point component images exist in ``frame``."""
        if frame == "intrinsic":
            return self._has("profiles/map") if self.factorised else self._has("intrinsic/map")
        if frame == "convolved":
            return self._has("convolved/map")
        raise ValueError(f"frame must be 'intrinsic' or 'convolved', got {frame!r}")

    def has_frame_draws(self, frame: str) -> bool:
        """Whether per-draw component images exist in ``frame``."""
        if frame == "intrinsic":
            if self.factorised:
                return (
                    self._has("profiles/draws") and self.has_draws and self.draw_index is not None
                )
            return self._has("intrinsic/draws")
        if frame == "convolved":
            return self._has("convolved/draws")
        raise ValueError(f"frame must be 'intrinsic' or 'convolved', got {frame!r}")

    def sed_intrinsic_map(self) -> np.ndarray:
        """Best-point intrinsic SEDs (K, B)."""
        return np.asarray(self._read("sed_intrinsic/map"), dtype=float)

    def sed_intrinsic(self) -> np.ndarray:
        """Intrinsic SEDs (S, K, B), or (1, K, B) best point without draws."""
        if self.has_draws:
            return np.asarray(self._read("sed_intrinsic/draws"), dtype=float)
        return self.sed_intrinsic_map()[None]

    def params(self) -> np.ndarray:
        """Component parameters (S, K, Pc), or (1, K, Pc) best point without draws."""
        if self._has("params/draws"):
            return np.asarray(self._read("params/draws"), dtype=float)
        return np.asarray(self._read("params/map"), dtype=float)[None]

    def _check_k(self, k: int) -> None:
        if not 0 <= k < self.n_components:
            raise IndexError(f"component {k} out of range [0, {self.n_components})")

    def image_draws(self, k: int, frame: str = "convolved") -> np.ndarray:
        """Per-draw image of component ``k``, shape (Sc, B, H, W).

        Raises:
            KeyError: If the frame has no draws.
        """
        self._check_k(k)
        if not self.has_frame_draws(frame):
            raise KeyError(f"No {frame} component draws in this bundle")
        if frame == "intrinsic" and self.factorised:
            seds = self.sed_intrinsic()[self.draw_index, k]  # (Sc, B)
            prof = np.asarray(self._read("profiles/draws")[:, k], dtype=float)  # (Sc, H, W)
            return seds[:, :, None, None] * prof[:, None]
        return np.asarray(self._read(f"{frame}/draws")[:, k], dtype=float)

    def image(self, k: int, frame: str = "intrinsic", stat: str = "p50") -> np.ndarray:
        """Image of component ``k`` (B, H, W).

        Args:
            k: Component index.
            frame: ``"intrinsic"`` or ``"convolved"`` (PSF + shifts, no sky).
            stat: One of ``map, p16, p50, p84, mean, std``; percentiles and mean fall back to
                the best point without draws (``last_fallback`` is set).

        Raises:
            KeyError: If the frame (or ``std`` without draws) is unavailable.
        """
        self._check_k(k)
        self.last_fallback = False
        if not self.has_frame(frame):
            raise KeyError(f"No {frame} component images in this bundle")
        if stat not in MODEL_STATS:
            raise ValueError(f"stat must be one of {MODEL_STATS}, got {stat!r}")
        if stat != "map" and self.has_frame_draws(frame):
            return _reduce(self.image_draws(k, frame), stat)
        if stat == "std":
            raise KeyError("No component draws: 'std' unavailable")
        self.last_fallback = stat != "map"
        if frame == "intrinsic" and self.factorised:
            sed = self.sed_intrinsic_map()[k]
            prof = np.asarray(self._read("profiles/map")[k], dtype=float)
            return sed[:, None, None] * prof[None]
        return np.asarray(self._read(f"{frame}/map")[k], dtype=float)


def _reduce(draws: np.ndarray, stat: str) -> np.ndarray:
    if stat in ("p16", "p50", "p84"):
        return np.percentile(draws, float(stat[1:]), axis=0)
    if stat == "mean":
        return draws.mean(axis=0)
    if stat == "std":
        return draws.std(axis=0)
    raise ValueError(stat)


class ResultsBundle:
    """A fitted model's results: observation, model cubes, parameter maps and draws.

    Use :func:`load_bundle` or :meth:`from_content`.  Attributes are documented in the design
    spec, section 3; the main entry points are :meth:`band_image`, :meth:`band_draws`,
    :meth:`param_map` and the functions of :mod:`arachne.results.regions`.

    Attributes:
        last_fallback: True when the last :meth:`band_image` / :meth:`param_map` call fell
            back from a draw summary to the best point (no draws).
    """

    def __init__(self, store: _MemoryStore | _H5Store, path: Path | None = None):
        """Build from a store; prefer :func:`load_bundle` or :meth:`from_content`."""
        self._store = store
        self.path = path
        self._cache: dict[str, Any] = {}
        self.last_fallback = False
        self.attrs: dict[str, Any] = store.attrs("")
        self._check_version()
        if self.attrs.get("grid", "single") != "single":
            raise NotImplementedError("grid='per_band' reserved in schema 1.0")
        if self.attrs.get("kind", "image") != "image":
            raise NotImplementedError(f"kind={self.attrs.get('kind')!r} reserved in schema 1.0")
        obs = store.attrs("observation")
        self.band_names: list[str] = list(obs.get("band_names", []))
        self.short_names: list[str] = [short_band_name(b) for b in self.band_names]
        self.pivot_um: np.ndarray = np.asarray(obs.get("pivot_um"), dtype=float)
        self.pixel_scale: float = float(obs.get("pixel_scale_arcsec", np.nan))
        self.n_bands: int = len(self.band_names)
        self.flux: np.ndarray = np.asarray(self._read("observation/flux"))
        self.variance: np.ndarray = np.asarray(self._read("observation/variance"))
        self.mask: np.ndarray = np.asarray(self._read("observation/mask"))
        self.image_shape: tuple[int, int] = tuple(int(n) for n in self.flux.shape[1:])
        lk = store.attrs("likelihood")
        self.model_error_frac: np.ndarray = np.asarray(
            lk.get("model_error_frac", np.zeros(self.n_bands)), dtype=float
        )
        self.schema_version: str = str(self.attrs.get("schema_version"))
        self.grid: str = str(self.attrs.get("grid", "single"))
        self.posterior_source: str = str(self.attrs.get("posterior_source", "map"))
        self.n_draws: int = int(self.attrs.get("n_draws", 0))
        self.degraded: bool = bool(self.attrs.get("degraded", False))
        self.title: str = str(self.attrs.get("title", ""))
        self.redshift: float = float(self.attrs.get("redshift", np.nan))
        self.spatial_model: str = str(self.attrs.get("spatial_model", ""))
        self.forward_model: str = str(self.attrs.get("forward_model", ""))
        self.provenance: dict[str, Any] = {
            k: v for k, v in self.attrs.items() if k not in STRUCTURAL_ATTRS
        }

        pa = store.attrs("params")
        self.map_names: list[str] = list(pa.get("map_names", []))
        self.scalar_names: list[str] = list(pa.get("scalar_names", []))
        self.mass_param: str = str(pa.get("mass_param", ""))
        fixed_vals = np.atleast_1d(pa.get("fixed_values", np.zeros(0)))
        self.fixed_params: dict[str, float] = {
            n: float(v) for n, v in zip(pa.get("fixed_names", []), fixed_vals)
        }
        stored_labels = self._json("params/labels") or {}
        self.labels: dict[str, str] = {
            **{n: DEFAULT_LABELS.get(n, n) for n in (*self.map_names, *self.scalar_names)},
            **stored_labels,
        }
        self.units: dict[str, str] = self._json("params/units") or {}
        self.metrics: dict = self._json("metrics/json") or {}

        self.regions: dict[str, Region] = {}
        for name in store.children("regions"):
            ra = store.attrs(f"regions/{name}")
            self.regions[name] = Region(
                name=name,
                mask=np.asarray(self._read(f"regions/{name}")).astype(bool),
                kind=str(ra.get("kind", "mask")),
                geometry=json.loads(ra.get("geometry", "{}") or "{}"),
            )
        self.truth: TruthContent | None = self._load_truth()
        self.components: ComponentsView | None = (
            ComponentsView(self) if store.exists("components") else None
        )

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_content(cls, content: BundleContent) -> ResultsBundle:
        """Wrap in-memory content (no file); behaves exactly like a loaded bundle."""
        return cls(_MemoryStore(content_to_tree(content)), None)

    def _check_version(self) -> None:
        name = self.attrs.get("schema")
        if name != SCHEMA_NAME:
            raise ValueError(f"Not an {SCHEMA_NAME} bundle (schema attr = {name!r})")
        version = str(self.attrs.get("schema_version", "0.0"))
        try:
            major, minor = (int(x) for x in version.split(".")[:2])
        except ValueError as err:
            raise ValueError(f"Malformed schema_version {version!r}") from err
        our_major, our_minor = (int(x) for x in SCHEMA_VERSION.split("."))
        if major != our_major:
            raise ValueError(
                f"Bundle schema version {version} is incompatible with reader {SCHEMA_VERSION}"
            )
        if minor > our_minor:
            warnings.warn(
                f"Bundle schema version {version} is newer than reader {SCHEMA_VERSION}; "
                "unknown content is ignored",
                stacklevel=3,
            )

    def _read(self, path: str) -> Any:
        if path not in self._cache:
            if not self._store.is_dataset(path):
                raise KeyError(f"{path} not in bundle")
            self._cache[path] = self._store.read(path)
        return self._cache[path]

    def _opt(self, path: str) -> Any:
        return self._read(path) if self._store.is_dataset(path) else None

    def _json(self, path: str) -> Any:
        raw = self._opt(path)
        if raw is None:
            return None
        return json.loads(raw if isinstance(raw, str) else str(raw))

    def _load_truth(self) -> TruthContent | None:
        if not self._store.exists("truth"):
            return None
        fields = ("convolved", "intrinsic", "param_maps", "scalars", "component_seds", "theta")
        return TruthContent(**{f: self._opt(f"truth/{f}") for f in fields})

    def close(self) -> None:
        """Drop memoised draws (the file is only open while reading)."""
        self._cache = {k: v for k, v in self._cache.items() if not _is_lazy(k)}

    def __enter__(self) -> ResultsBundle:
        """Context-manager entry."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Context-manager exit (calls :meth:`close`)."""
        self.close()

    def __repr__(self) -> str:
        """Short description."""
        src = self.path or "<memory>"
        return (
            f"ResultsBundle({src}, {self.posterior_source}, S={self.n_draws}, "
            f"B={self.n_bands}, shape={self.image_shape})"
        )

    # ------------------------------------------------------------------ flags

    @property
    def has_draws(self) -> bool:
        """True when model draws are stored (S > 0)."""
        return self.n_draws > 0 and self._store.is_dataset("model/convolved/draws")

    @property
    def has_intrinsic(self) -> bool:
        """True when the unconvolved model is stored."""
        return self._store.is_dataset("model/intrinsic/map")

    @property
    def has_truth(self) -> bool:
        """True for mocks with a ``/truth`` group."""
        return self.truth is not None

    @property
    def has_components(self) -> bool:
        """True for additive models with a ``/components`` group."""
        return self.components is not None

    @property
    def has_param_draws(self) -> bool:
        """True when parameter-map draws are stored."""
        return self._store.is_dataset("params/maps/draws")

    # ------------------------------------------------------------------ misc accessors

    @property
    def psf(self) -> np.ndarray | None:
        """PSF kernels (B, h, w), if stored."""
        return self._opt("observation/psf")

    @property
    def nuisance(self) -> dict[str, np.ndarray]:
        """Nuisance arrays keyed as in ``/model/nuisance`` (empty when none)."""
        return {
            k: np.asarray(self._read(f"model/nuisance/{k}"))
            for k in self._children("model/nuisance")
        }

    @property
    def posterior_attrs(self) -> dict[str, Any]:
        """Attributes of ``/posterior`` (sampler, n_chains, n_samples_total, ...)."""
        return self._store.attrs("posterior")

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Sampler diagnostics: scalar attrs plus per-parameter arrays (rhat, ess)."""
        out: dict[str, Any] = dict(self._store.attrs("posterior/diagnostics"))
        for k in self._children("posterior/diagnostics"):
            out[k] = np.asarray(self._read(f"posterior/diagnostics/{k}"))
        return out

    @property
    def theta_map(self) -> np.ndarray | None:
        """Best-point theta (P,), if stored."""
        return self._opt("posterior/theta_map")

    def theta_draws(self) -> np.ndarray | None:
        """Thinned theta draws (S, P), if stored (loaded on first use)."""
        return self._opt("posterior/theta_draws")

    @property
    def draw_index(self) -> np.ndarray | None:
        """Index of every stored draw in the original chain-concatenated samples."""
        return self._opt("posterior/draw_index")

    @property
    def chain_id(self) -> np.ndarray | None:
        """Chain of every stored draw."""
        return self._opt("posterior/chain_id")

    @property
    def extra_names(self) -> list[str]:
        """Names of the free-form ``/extra`` arrays."""
        return self._children("extra")

    def extra(self, name: str) -> np.ndarray:
        """Free-form array ``/extra/<name>``."""
        return np.asarray(self._read(f"extra/{name}"))

    def _children(self, group: str) -> list[str]:
        return [c for c in self._store.children(group) if self._store.is_dataset(f"{group}/{c}")]

    def noise_scale(self) -> np.ndarray:
        """Per-band noise scale ``exp(median log_noise_scale)`` (ones when not fitted)."""
        draws = self._opt("model/nuisance/log_noise_scale")
        if draws is not None and len(draws) > 0:
            return np.exp(np.median(np.asarray(draws, dtype=float), axis=0))
        best = self._opt("model/nuisance/log_noise_scale_map")
        if best is not None:
            return np.exp(np.asarray(best, dtype=float))
        return np.ones(self.n_bands)

    # ------------------------------------------------------------------ images

    def _model_cube(self, frame: str, stat: str) -> np.ndarray:
        if not self._store.is_dataset(f"model/{frame}/map"):
            raise KeyError(f"No {frame} model in this bundle")
        if stat not in MODEL_STATS:
            raise ValueError(f"stat must be one of {MODEL_STATS}, got {stat!r}")
        if stat == "map":
            return self._read(f"model/{frame}/map")
        path = f"model/{frame}/{stat}"
        if self._store.is_dataset(path):
            return self._read(path)
        if stat in _FALLBACK_STATS:
            self.last_fallback = True
            return self._read(f"model/{frame}/map")
        raise KeyError(f"No draws: model stat {stat!r} unavailable")

    def _truth_cube(self, name: str) -> np.ndarray:
        arr = None if self.truth is None else getattr(self.truth, name)
        if arr is None:
            raise KeyError(f"No truth {name} in this bundle")
        return arr

    def band_image(
        self, b: int, kind: str, stat: str = "p50", *, model_floor: bool = True
    ) -> np.ndarray:
        """One band of an observed, model or derived image.

        Args:
            b: Band index.
            kind: One of ``data, variance, sigma, mask, model, intrinsic, residual, chi,
                truth_model, truth_intrinsic, truth_residual``.
            stat: Model statistic, one of ``map, p16, p50, p84, mean, std``.  Without draws,
                ``p16/p50/p84/mean`` fall back to ``map`` (``last_fallback`` is set) and
                ``std`` raises KeyError.
            model_floor: For ``chi``: use ``var_eff = variance * exp(2 s) + (f * model)^2``
                as in the likelihood; otherwise the plain variance.

        Returns:
            Float array (H, W).  Data-derived kinds are NaN where ``mask == 0``.

        Raises:
            KeyError: If the requested product is not in the bundle.
        """
        self.last_fallback = False
        if not -self.n_bands <= b < self.n_bands:
            raise IndexError(f"band {b} out of range for {self.n_bands} bands")
        valid = self.mask[b] > 0
        flux = np.asarray(self.flux[b], dtype=float)
        var = np.asarray(self.variance[b], dtype=float)
        if kind == "mask":
            return np.asarray(self.mask[b], dtype=float)
        if kind == "model":
            return np.asarray(self._model_cube("convolved", stat)[b], dtype=float)
        if kind == "intrinsic":
            return np.asarray(self._model_cube("intrinsic", stat)[b], dtype=float)
        if kind == "truth_model":
            return np.asarray(self._truth_cube("convolved")[b], dtype=float)
        if kind == "truth_intrinsic":
            return np.asarray(self._truth_cube("intrinsic")[b], dtype=float)
        if kind == "data":
            img = flux
        elif kind == "variance":
            img = var
        elif kind == "sigma":
            img = np.sqrt(var)
        elif kind == "residual":
            img = flux - np.asarray(self._model_cube("convolved", stat)[b], dtype=float)
        elif kind == "chi":
            model = np.asarray(self._model_cube("convolved", stat)[b], dtype=float)
            if model_floor:
                var_eff = var * self.noise_scale()[b] ** 2 + (self.model_error_frac[b] * model) ** 2
            else:
                var_eff = var
            with np.errstate(divide="ignore", invalid="ignore"):
                img = (flux - model) / np.sqrt(var_eff)
        elif kind == "truth_residual":
            img = flux - np.asarray(self._truth_cube("convolved")[b], dtype=float)
        else:
            raise ValueError(f"kind must be one of {IMAGE_KINDS}, got {kind!r}")
        return np.where(valid, img, np.nan)

    def cube(self, kind: str, stat: str = "p50", **kw: Any) -> np.ndarray:
        """All bands of :meth:`band_image` stacked to (B, H, W) (single grid only)."""
        fallback = False
        planes = []
        for b in range(self.n_bands):
            planes.append(self.band_image(b, kind, stat, **kw))
            fallback = fallback or self.last_fallback
        self.last_fallback = fallback
        return np.stack(planes)

    def model_draws(self, frame: str = "convolved") -> np.ndarray:
        """Model draws (S, B, H, W), memoised.

        Raises:
            KeyError: If the frame has no draws.
        """
        if frame not in ("convolved", "intrinsic"):
            raise ValueError(f"frame must be 'convolved' or 'intrinsic', got {frame!r}")
        return self._read(f"model/{frame}/draws")

    def band_draws(self, b: int, frame: str = "convolved") -> np.ndarray:
        """Model draws of band ``b`` (S, H, W); KeyError if absent."""
        return self.model_draws(frame)[:, b]

    def chi2_red_per_band(self, *, model_floor: bool = True) -> np.ndarray:
        """Mean chi^2 over valid pixels per band for the p50 model (B,)."""
        out = np.empty(self.n_bands)
        for b in range(self.n_bands):
            chi = self.band_image(b, "chi", "p50", model_floor=model_floor)
            good = np.isfinite(chi)
            out[b] = np.mean(chi[good] ** 2) if good.any() else np.nan
        return out

    # ------------------------------------------------------------------ parameters

    def _map_index(self, name: str) -> int:
        try:
            return self.map_names.index(name)
        except ValueError:
            raise KeyError(f"{name!r} is not a parameter map (have {self.map_names})") from None

    def _param_stat(self, i: int, stat: str) -> np.ndarray:
        if stat == "map":
            return np.asarray(self._read("params/maps/map")[i], dtype=float)
        path = f"params/maps/{stat}"
        if self._store.is_dataset(path):
            return np.asarray(self._read(path)[i], dtype=float)
        if stat in _FALLBACK_STATS and not self.has_param_draws:
            self.last_fallback = True
            return np.asarray(self._read("params/maps/map")[i], dtype=float)
        raise KeyError(f"Parameter stat {stat!r} unavailable")

    def _truth_map(self, i: int) -> np.ndarray:
        if self.truth is None or self.truth.param_maps is None:
            raise KeyError("No truth parameter maps in this bundle")
        return np.asarray(self.truth.param_maps[i], dtype=float)

    def param_map(self, name: str, stat: str = "p50") -> np.ndarray:
        """A parameter map (H, W).

        Args:
            name: One of :attr:`map_names`.
            stat: ``map, p16, p50, p84, mean, std, truth, diff (p50 - truth),
                z ((p50 - truth) / std), information, prior_std``.  Without stored summaries
                ``p16/p50/p84/mean`` fall back to ``map`` (``last_fallback`` is set).

        Raises:
            KeyError: If the name or statistic is unavailable.
        """
        self.last_fallback = False
        i = self._map_index(name)
        if stat in ("map", *SUMMARY_STATS, "information", "prior_std"):
            return self._param_stat(i, stat)
        if stat == "truth":
            return self._truth_map(i)
        if stat == "diff":
            return self._param_stat(i, "p50") - self._truth_map(i)
        if stat == "z":
            truth = self._truth_map(i)
            p50 = self._param_stat(i, "p50")
            std = self._param_stat(i, "std")
            with np.errstate(divide="ignore", invalid="ignore"):
                return (p50 - truth) / std
        raise ValueError(f"stat must be one of {PARAM_STATS}, got {stat!r}")

    def available_param_stats(self) -> list[str]:
        """Parameter-map statistics actually stored (no fallbacks), in display order."""
        if not self.map_names:
            return []
        have = {s for s in (*SUMMARY_STATS, "information", "prior_std")}
        have = {s for s in have if self._store.is_dataset(f"params/maps/{s}")}
        truth = self.truth is not None and self.truth.param_maps is not None
        out = ["map"]
        out += [s for s in SUMMARY_STATS if s in have]
        if truth:
            out += ["truth", "diff"]
            if "std" in have:
                out.append("z")
        out += [s for s in ("information", "prior_std") if s in have]
        return out

    def param_draws(self, name: str) -> np.ndarray:
        """Draws of one parameter map (S, H, W); KeyError if absent."""
        i = self._map_index(name)
        return self._read("params/maps/draws")[:, i]

    def pixel_param_draws(self, y: int, x: int) -> dict[str, np.ndarray]:
        """Every map parameter at pixel ``[y, x]``: name -> (S,) draws, or (1,) best point."""
        if self.has_param_draws:
            d = self._read("params/maps/draws")
            return {n: np.asarray(d[:, i, y, x], dtype=float) for i, n in enumerate(self.map_names)}
        if not self.map_names:
            return {}
        m = self._read("params/maps/map")
        return {n: np.asarray(m[i, y, x], dtype=float)[None] for i, n in enumerate(self.map_names)}

    def scalar_draws(self) -> dict[str, np.ndarray]:
        """Scalar parameters: name -> (S,) draws, or (1,) best point without draws."""
        if not self.scalar_names:
            return {}
        d = self._opt("params/scalars/draws")
        if d is None:
            d = np.asarray(self._read("params/scalars/map"))[None]
        return {n: np.asarray(d[:, i], dtype=float) for i, n in enumerate(self.scalar_names)}


def load_bundle(path: str | Path) -> ResultsBundle:
    """Open an ``arachne.results`` HDF5 bundle.

    Args:
        path: Bundle path.

    Returns:
        A :class:`ResultsBundle` (draws are read lazily).

    Raises:
        ValueError: On a wrong schema or an incompatible major version.
        NotImplementedError: For the reserved ``grid="per_band"`` or ``kind="photometry"``.
    """
    path = Path(path)
    return ResultsBundle(_H5Store(path), path)
