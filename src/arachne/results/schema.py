"""In-memory content of an ``arachne.results`` bundle and structural validation.

The dataclasses here mirror the HDF5 layout of schema ``arachne.results`` v1.0 one to one (see
``docs/design/results_bundle_gui_spec.md``).  They hold plain numpy arrays and never import JAX.
Shape symbols: S draws, B bands, (H, W) image, Nm parameter maps, Ns scalars, K components,
Pc component parameters, Sc component-cube draws, P theta length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .regions import Region

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SUMMARY_STATS",
    "NUISANCE_KEYS",
    "TruthContent",
    "ComponentContent",
    "BundleContent",
    "validate",
]

SCHEMA_NAME = "arachne.results"
SCHEMA_VERSION = "1.0"
SUMMARY_STATS = ("p16", "p50", "p84", "mean", "std")
PARAM_SUMMARY_KEYS = (*SUMMARY_STATS, "information")
NUISANCE_KEYS = (
    "sky_map",
    "sky",
    "shifts_map",
    "shifts",
    "log_noise_scale_map",
    "log_noise_scale",
)
POSTERIOR_SOURCES = ("nuts", "laplace", "nss", "mclmc", "map")


@dataclass
class TruthContent:
    """Ground truth for mocks (all optional).

    Attributes:
        convolved: (B, H, W) noise-free observed-frame image.
        intrinsic: (B, H, W) unconvolved image.
        param_maps: (Nm, H, W) parameter maps.
        scalars: (Ns,) scalar parameters.
        component_seds: (K, B) intrinsic component SEDs.
        theta: (P,) true parameter vector.
    """

    convolved: np.ndarray | None = None
    intrinsic: np.ndarray | None = None
    param_maps: np.ndarray | None = None
    scalars: np.ndarray | None = None
    component_seds: np.ndarray | None = None
    theta: np.ndarray | None = None


@dataclass
class ComponentContent:
    """Per-component products of an additive model.

    Attributes:
        profile_names: (K,) profile names.
        param_names: (Pc,) names of the per-component parameter columns.
        factorised: True when each component is ``sed_k ⊗ profile_k``.
        sed_intrinsic_map: (K, B) best-point intrinsic SEDs.
        sed_intrinsic_draws: (S, K, B) or None.
        profiles_map: (K, H, W) normalised profiles (factorised) or None.
        profiles_draws: (Sc, K, H, W) or None.
        intrinsic_map: (K, B, H, W) (not factorised) or None.
        intrinsic_draws: (Sc, K, B, H, W) or None.
        convolved_map: (K, B, H, W) PSF + shifts, no sky, or None.
        convolved_draws: (Sc, K, B, H, W) or None.
        draw_index: (Sc,) indices into the S axis used for the cube draws, or None.
        params_map: (K, Pc) best-point parameters.
        params_draws: (S, K, Pc) or None.
        ordering: ``"size"`` or ``"none"``.
    """

    profile_names: list[str]
    param_names: list[str]
    factorised: bool
    sed_intrinsic_map: np.ndarray
    sed_intrinsic_draws: np.ndarray | None
    profiles_map: np.ndarray | None
    profiles_draws: np.ndarray | None
    intrinsic_map: np.ndarray | None
    intrinsic_draws: np.ndarray | None
    convolved_map: np.ndarray | None
    convolved_draws: np.ndarray | None
    draw_index: np.ndarray | None
    params_map: np.ndarray
    params_draws: np.ndarray | None
    ordering: str = "size"


@dataclass
class BundleContent:
    """Everything stored in a results bundle, as numpy arrays.

    See the module docstring for the shape symbols and the design spec for the meaning of
    every field.  ``param_summaries`` is only used when the map draws are absent (degraded
    bundles); its keys are among ``p16, p50, p84, mean, std, information``.
    """

    band_names: list[str]
    pivot_um: np.ndarray
    pixel_scale: float
    flux: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    model_error_frac: np.ndarray
    convolved_map: np.ndarray
    convolved_draws: np.ndarray | None = None
    intrinsic_map: np.ndarray | None = None
    intrinsic_draws: np.ndarray | None = None
    map_names: list[str] = field(default_factory=list)
    param_maps_map: np.ndarray | None = None
    param_maps_draws: np.ndarray | None = None
    prior_std_maps: np.ndarray | None = None
    param_summaries: dict[str, np.ndarray] | None = None
    scalar_names: list[str] = field(default_factory=list)
    scalars_map: np.ndarray | None = None
    scalars_draws: np.ndarray | None = None
    fixed_params: dict[str, float] = field(default_factory=dict)
    mass_param: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    nuisance: dict[str, np.ndarray] | None = None
    components: ComponentContent | None = None
    truth: TruthContent | None = None
    theta_map: np.ndarray | None = None
    theta_draws: np.ndarray | None = None
    draw_index: np.ndarray | None = None
    chain_id: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    regions: dict[str, Region] = field(default_factory=dict)
    metrics: dict | None = None
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    posterior_source: str = "map"
    spatial_model: str = ""
    forward_model: str = "ForwardModel"
    redshift: float = float("nan")
    title: str = ""
    degraded: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    psf: np.ndarray | None = None

    @property
    def n_draws(self) -> int:
        """Number of stored draws S (0 = best point only)."""
        return 0 if self.convolved_draws is None else int(np.shape(self.convolved_draws)[0])

    def validate(self) -> None:
        """Check shapes and consistency.

        Raises:
            ValueError: Listing every problem found.
        """
        problems = _content_problems(self)
        if problems:
            raise ValueError("Invalid BundleContent:\n  - " + "\n  - ".join(problems))


def _shape(a: Any) -> tuple[int, ...] | None:
    return None if a is None else tuple(np.shape(a))


def _content_problems(c: BundleContent) -> list[str]:
    p: list[str] = []
    fs = _shape(c.flux)
    if fs is None or len(fs) != 3:
        return [f"flux must be (B, H, W), got {fs}"]
    nb, h, w = fs
    s = c.n_draws

    def chk(name: str, arr: Any, expected: tuple[int, ...], optional: bool = True) -> None:
        shp = _shape(arr)
        if shp is None:
            if not optional:
                p.append(f"{name} is required")
            return
        if shp != expected:
            p.append(f"{name} has shape {shp}, expected {expected}")

    if len(c.band_names) != nb:
        p.append(f"band_names has {len(c.band_names)} entries, expected {nb}")
    chk("pivot_um", c.pivot_um, (nb,), optional=False)
    chk("variance", c.variance, fs, optional=False)
    chk("mask", c.mask, fs, optional=False)
    chk("model_error_frac", c.model_error_frac, (nb,), optional=False)
    chk("convolved_map", c.convolved_map, fs, optional=False)
    chk("convolved_draws", c.convolved_draws, (s, nb, h, w))
    chk("intrinsic_map", c.intrinsic_map, fs)
    if c.intrinsic_draws is not None and c.intrinsic_map is None:
        p.append("intrinsic_draws given without intrinsic_map")
    chk("intrinsic_draws", c.intrinsic_draws, (s, nb, h, w))
    if c.psf is not None and (np.ndim(c.psf) != 3 or np.shape(c.psf)[0] != nb):
        p.append(f"psf must be (B, h, w), got {_shape(c.psf)}")

    nm = len(c.map_names)
    if nm and c.param_maps_map is None:
        p.append("map_names given but param_maps_map is None")
    chk("param_maps_map", c.param_maps_map, (nm, h, w))
    chk("param_maps_draws", c.param_maps_draws, (s, nm, h, w))
    chk("prior_std_maps", c.prior_std_maps, (nm, h, w))
    for key, arr in (c.param_summaries or {}).items():
        if key not in PARAM_SUMMARY_KEYS:
            p.append(f"param_summaries key {key!r} not in {PARAM_SUMMARY_KEYS}")
        chk(f"param_summaries[{key!r}]", arr, (nm, h, w))
    if c.mass_param and c.mass_param not in c.map_names:
        p.append(f"mass_param {c.mass_param!r} is not a map name")

    ns = len(c.scalar_names)
    if ns and c.scalars_map is None:
        p.append("scalar_names given but scalars_map is None")
    chk("scalars_map", c.scalars_map, (ns,))
    chk("scalars_draws", c.scalars_draws, (s, ns))

    if c.theta_draws is not None:
        if np.ndim(c.theta_draws) != 2 or np.shape(c.theta_draws)[0] != s:
            p.append(f"theta_draws must be (S={s}, P), got {_shape(c.theta_draws)}")
        elif c.theta_map is not None and np.shape(c.theta_map) != np.shape(c.theta_draws)[1:]:
            p.append("theta_map and theta_draws have different lengths")
    chk("draw_index", c.draw_index, (s,))
    chk("chain_id", c.chain_id, (s,))

    for key, arr in (c.nuisance or {}).items():
        if key not in NUISANCE_KEYS:
            p.append(f"nuisance key {key!r} not in {NUISANCE_KEYS}")
            continue
        per_band = (nb, 2) if key.startswith("shifts") else (nb,)
        expected = per_band if key.endswith("_map") else (s, *per_band)
        chk(f"nuisance[{key!r}]", arr, expected)

    if c.components is not None:
        p.extend(_component_problems(c.components, s, nb, h, w))

    t = c.truth
    if t is not None:
        chk("truth.convolved", t.convolved, fs)
        chk("truth.intrinsic", t.intrinsic, fs)
        chk("truth.param_maps", t.param_maps, (nm, h, w))
        chk("truth.scalars", t.scalars, (ns,))
        if t.component_seds is not None and np.shape(t.component_seds)[-1:] != (nb,):
            p.append(f"truth.component_seds must be (K, B), got {_shape(t.component_seds)}")

    for name, reg in c.regions.items():
        mshape = np.shape(reg.mask if isinstance(reg, Region) else reg)
        if mshape != (h, w):
            p.append(f"region {name!r} has shape {mshape}, expected {(h, w)}")
    if c.posterior_source not in POSTERIOR_SOURCES:
        p.append(f"posterior_source {c.posterior_source!r} not in {POSTERIOR_SOURCES}")
    return p


def _component_problems(cc: ComponentContent, s: int, nb: int, h: int, w: int) -> list[str]:
    p: list[str] = []
    k = len(cc.profile_names)
    pc = len(cc.param_names)
    sc = None if cc.draw_index is None else int(np.shape(cc.draw_index)[0])

    def chk(name: str, arr: Any, expected: tuple[int, ...]) -> None:
        shp = _shape(arr)
        if shp is not None and shp != expected:
            p.append(f"components.{name} has shape {shp}, expected {expected}")

    chk("sed_intrinsic_map", cc.sed_intrinsic_map, (k, nb))
    chk("sed_intrinsic_draws", cc.sed_intrinsic_draws, (s, k, nb))
    chk("params_map", cc.params_map, (k, pc))
    chk("params_draws", cc.params_draws, (s, k, pc))
    chk("profiles_map", cc.profiles_map, (k, h, w))
    chk("intrinsic_map", cc.intrinsic_map, (k, nb, h, w))
    chk("convolved_map", cc.convolved_map, (k, nb, h, w))
    cube_draws = (cc.profiles_draws, cc.intrinsic_draws, cc.convolved_draws)
    if any(d is not None for d in cube_draws):
        if sc is None:
            p.append("components cube draws given without draw_index")
        else:
            chk("profiles_draws", cc.profiles_draws, (sc, k, h, w))
            chk("intrinsic_draws", cc.intrinsic_draws, (sc, k, nb, h, w))
            chk("convolved_draws", cc.convolved_draws, (sc, k, nb, h, w))
            if sc and (np.min(cc.draw_index) < 0 or np.max(cc.draw_index) >= s):
                p.append(f"components.draw_index out of range [0, {s})")
    if cc.factorised and cc.profiles_map is None:
        p.append("factorised components need profiles_map")
    if not cc.factorised and cc.intrinsic_map is None:
        p.append("non-factorised components need intrinsic_map")
    if cc.ordering not in ("size", "none"):
        p.append(f"components.ordering {cc.ordering!r} not in ('size', 'none')")
    return p


# ---------------------------------------------------------------------------
# HDF5 structural validation
# ---------------------------------------------------------------------------


def validate(h5: Any) -> list[str]:
    """Report structural problems of an HDF5 bundle.

    Args:
        h5: Path to the file, or an open ``h5py.File`` / group.

    Returns:
        List of human-readable problems (empty when the bundle is valid).
    """
    import h5py

    if isinstance(h5, (str, Path)):
        with h5py.File(h5, "r") as f:
            return validate(f)

    p: list[str] = []
    attrs = h5.attrs
    if attrs.get("schema") != SCHEMA_NAME:
        p.append(f"root attr schema is {attrs.get('schema')!r}, expected {SCHEMA_NAME!r}")
    version = str(attrs.get("schema_version", ""))
    if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        p.append(f"schema_version {version!r} incompatible with {SCHEMA_VERSION!r}")
    for key in ("kind", "grid", "posterior_source", "n_draws", "degraded"):
        if key not in attrs:
            p.append(f"missing root attr {key!r}")
    if attrs.get("grid", "single") != "single":
        p.append(f"grid {attrs.get('grid')!r} is reserved and not supported in schema 1.0")
    if attrs.get("kind", "image") != "image":
        p.append(f"kind {attrs.get('kind')!r} is reserved and not supported in schema 1.0")

    def has(path: str) -> bool:
        return path in h5

    def shp(path: str) -> tuple[int, ...] | None:
        return tuple(h5[path].shape) if has(path) else None

    for path in (
        "observation/flux",
        "observation/variance",
        "observation/mask",
        "likelihood",
        "model/convolved/map",
        "posterior",
    ):
        if not has(path):
            p.append(f"missing {path}")
    fs = shp("observation/flux")
    if fs is None or len(fs) != 3:
        p.append(f"observation/flux must be (B, H, W), got {fs}")
        return p
    nb, h, w = fs
    s = int(attrs.get("n_draws", 0))

    def expect(path: str, expected: tuple[int, ...]) -> None:
        got = shp(path)
        if got is not None and got != expected:
            p.append(f"{path} has shape {got}, expected {expected}")

    obs = h5["observation"].attrs
    for key, n in (("band_names", nb), ("pivot_um", nb)):
        if key not in obs:
            p.append(f"missing observation attr {key!r}")
        elif len(obs[key]) != n:
            p.append(f"observation attr {key} has length {len(obs[key])}, expected {n}")
    expect("observation/variance", fs)
    expect("observation/mask", fs)
    if has("likelihood"):
        mef = h5["likelihood"].attrs.get("model_error_frac")
        if mef is None or len(np.atleast_1d(mef)) != nb:
            p.append("likelihood attr model_error_frac must have length B")
    for frame in ("convolved", "intrinsic"):
        expect(f"model/{frame}/map", fs)
        expect(f"model/{frame}/draws", (s, nb, h, w))
        for stat in SUMMARY_STATS:
            expect(f"model/{frame}/{stat}", fs)
    if s > 0 and not has("model/convolved/draws"):
        p.append(f"n_draws={s} but model/convolved/draws is missing")
    if s == 0 and has("model/convolved/draws"):
        p.append("n_draws=0 but model/convolved/draws is present")

    if has("params"):
        pa = h5["params"].attrs
        nm = len(pa.get("map_names", []))
        ns = len(pa.get("scalar_names", []))
        expect("params/maps/map", (nm, h, w))
        expect("params/maps/draws", (s, nm, h, w))
        for stat in (*PARAM_SUMMARY_KEYS, "prior_std"):
            expect(f"params/maps/{stat}", (nm, h, w))
        expect("params/scalars/map", (ns,))
        expect("params/scalars/draws", (s, ns))
        if nm and not has("params/maps/map"):
            p.append("params attr map_names set but params/maps/map is missing")
    else:
        p.append("missing params")

    if has("posterior"):
        expect("posterior/draw_index", (s,))
        expect("posterior/chain_id", (s,))
        td = shp("posterior/theta_draws")
        if td is not None and td[0] != s:
            p.append(f"posterior/theta_draws has {td[0]} rows, expected {s}")

    if has("components"):
        ca = h5["components"].attrs
        k = int(ca.get("n_components", 0))
        pc = len(ca.get("param_names", []))
        sc = shp("components/draw_index")
        sc = sc[0] if sc is not None else None
        expect("components/sed_intrinsic/map", (k, nb))
        expect("components/sed_intrinsic/draws", (s, k, nb))
        expect("components/params/map", (k, pc))
        expect("components/params/draws", (s, k, pc))
        expect("components/profiles/map", (k, h, w))
        for frame in ("intrinsic", "convolved"):
            expect(f"components/{frame}/map", (k, nb, h, w))
        if sc is not None:
            expect("components/profiles/draws", (sc, k, h, w))
            for frame in ("intrinsic", "convolved"):
                expect(f"components/{frame}/draws", (sc, k, nb, h, w))
        if len(ca.get("profile_names", [])) != k:
            p.append("components attr profile_names must have n_components entries")

    if has("regions"):
        for name, ds in h5["regions"].items():
            if tuple(ds.shape) != (h, w):
                p.append(f"regions/{name} has shape {tuple(ds.shape)}, expected {(h, w)}")
    if has("truth"):
        expect("truth/convolved", fs)
        expect("truth/intrinsic", fs)
    return p
