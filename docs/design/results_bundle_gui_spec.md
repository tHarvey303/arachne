# Spec: arachne results bundles, static plots and interactive viewer

Status: agreed design for v1 (2026-10-29). Builders A (results), B (viz) and C (gui) work to this document. Changes to the shared interfaces in Section 3 go through Builder A, and this file is updated first.

## 0. Facts that drive the design

- Existing run outputs (`2026-10-29_bands/prior_medium`):
  - `maps.npz`: truth/start/map `(4,64,64)`, `model_image (17,64,64)` (PSF-convolved MAP), `data (17,64,64)`, `sigma (17,)` (one sigma per band, white noise), `theta_map (4440,)`, `fields (4,)`.
  - `laplace.npz`, `nuts.npz`: median/std/information `(4,64,64)`. Only the Laplace file has `theta`.
  - `metrics.json`: `bands`, `emulator` path, `psf`, `size_px`, `chi2_red_per_band`, plus `laplace`/`nuts` sub-dicts. Pixel scale and fixed params are not saved (they are demo constants).
  - No posterior draws of images or maps exist, and the Laplace Cholesky factor is not saved.
- `ForwardModel._model_image(theta)` already includes nuisance shifts and sky. `spatial_model.model_image(theta_spatial, emulator, (H,W))` gives the unconvolved image.
- `GaussianLikelihood` uses `var_eff = variance*exp(2s) + (f*model)^2`. `residual_summary` builds chi with the plain variance. The bundle therefore records `model_error_frac` and any noise scale, and chi comes in two flavours (with and without the floor).
- `arachne/__init__.py` imports JAX eagerly, so `import arachne.results` imports JAX. JAX is a core dependency (CPU only), so this is acceptable. The rule: **no module under `results/` (except `writer.py`), `viz/` or `gui/` may import jax or equinox.** A test enforces this with `ast`. The GUI CLI sets `JAX_PLATFORMS=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing arachne, so it never takes GPU memory on the H100 node.
- Environments:
  - Local (`astro_code` pixi): panel 1.9.4, bokeh 3.9.2, JupyterLab 4.6, pyviz_comms. `jupyter_bokeh` is not installed, so plain `bokeh.io.show(app)` probably will not render in JupyterLab; Panel will.
  - Remote: bokeh 3.10, no panel.
- Measured on the real `model_image`: gzip+shuffle compresses float32 cubes to ~0.81 of raw; keeping 12 mantissa bits gives ~0.51.

## 1. Decisions (recommended defaults)

| # | Question | Default |
|---|---|---|
| D1 | GUI toolkit | **Plain Bokeh models with Python callbacks**, served with `bokeh.server.server.Server` (bokeh only, so it runs on the node). In Jupyter, wrap in `panel.pane.Bokeh` when panel is installed, otherwise fall back to `bokeh.io.show(app)`. One code path, Panel optional. |
| D2 | Draws vs recompute | Store **S thinned draws** of the convolved and intrinsic cubes, plus the parameter maps. The GUI never needs the emulator. Theta draws are also stored, for provenance and recomputation. |
| D3 | Precision | Lossless float32 with gzip(4)+shuffle by default. Opt-in `mantissa_bits=12` rounding, on draws only (relative error ~1e-4, far below the 5% model floor). |
| D4 | Posteriors per file | **One** posterior per bundle (nuts > laplace > map). Laplace vs NUTS is compared with two bundles (GUI compare mode, `--bundle-all`). |
| D5 | Regions | Boolean masks on the model grid, plus optional geometry (pixel/box/circle/polygon in pixel coordinates) so they can be re-rasterised later (multires). |
| D6 | Multires | `grid="per_band"` reserved in the schema; v1 writer/reader raise `NotImplementedError`. B and C must only use the per-band accessors (`band_image`, `band_draws`), never index a cube directly. |
| D7 | Catalogue/photometry fits | Reserved as `kind="photometry"` (H=W=1 "image"); not built in v1. |
| D8 | Lazy `arachne/__init__` | Not in v1; revisit if GUI start-up is slow. |

## 2. Bundle format: HDF5 schema `arachne.results` v1.0

Conventions:
- Array index `[y, x]` = `[row, col]`; pixel centres at integer coordinates; displayed with `origin="lower"`.
- Fluxes in nJy, variances in nJy².
- Strings use `h5py.string_dtype()`. JSON goes in **scalar string datasets**, not attrs (attrs are limited to 64 KB).
- Cubes use `compression="gzip", compression_opts=4, shuffle=True`; draws are chunked `(1, B, H, W)`.
- S = draws, B = bands, P = theta length, Nm = number of parameter maps, Ns = number of scalars, K = number of components.
- `[opt]` means the dataset may be absent; readers must treat absence as "not available".

```
/  attrs:
   schema="arachne.results"  schema_version="1.0"
   kind="image"                      # reserved: "photometry"
   grid="single"                     # reserved: "per_band"
   forward_model="ForwardModel"  spatial_model="CorrelatedFieldMap"|"AdditiveComponentModel"|...
   posterior_source="nuts"|"laplace"|"nss"|"mclmc"|"map"
   n_draws=S (0 = MAP only)  degraded=bool  title=str
   redshift=float (nan if free/unknown)
   created_utc (ISO8601)  arachne_version  git_commit  git_dirty(bool)  command_line
   hostname  python_version  jax_version  numpy_version  slurm_job_id ("" if none)
/observation
   attrs: band_names[B] (vlen str), pivot_um[B] f8, pixel_scale_arcsec f8, image_shape[2] i4,
          flux_unit="nJy", target_name="", ra=nan, dec=nan
   flux (B,H,W) f4 · variance (B,H,W) f4 · mask (B,H,W) f4 (weights, >0 = valid)
   psf (B,h,w) f4 [opt] · wcs_header (scalar str) [opt]
/likelihood   attrs: model_error_frac[B] f8, likelihood="gaussian"
/model
   attrs: draw_indices_source="posterior/draw_index"
   convolved/map (B,H,W) f4                       # observed frame: PSF+shifts+sky, best point
   convolved/draws (S,B,H,W) f4 [opt, S>0]
   convolved/{p16,p50,p84,mean,std} (B,H,W) f4 [opt, S>0]
   intrinsic/...  same layout [opt, store_intrinsic]   # unconvolved, no sky
   nuisance/ [opt]  sky_map (B,) · sky (S,B) · shifts_map (B,2) · shifts (S,B,2)
                    log_noise_scale_map (B,) · log_noise_scale (S,B)
/params
   attrs: map_names[Nm], scalar_names[Ns], fixed_names[], fixed_values[] f8,
          mass_param ("log_mass" or ""), mass_is_log10=True
   labels (scalar str, JSON {name: latex}) · units (scalar str, JSON {name: unit})
   maps/map (Nm,H,W) f4 · maps/draws (S,Nm,H,W) f4 [opt]
   maps/{p16,p50,p84,mean,std} (Nm,H,W) f4 [opt]
   maps/prior_std (Nm,H,W) f4 [opt] · maps/information (Nm,H,W) f4 [opt; =1-(std/prior_std)^2]
   scalars/map (Ns,) f8 · scalars/draws (S,Ns) f8 [opt]      # globals + derived (e.g. total_log_mass)
/components [opt]   (additive models)
   attrs: n_components=K, profile_names[K], param_names[Pc], factorised=bool, ordering="size"|"none"
   draw_index (Sc,) i4                   # which of the S draws the component cubes use
   sed_intrinsic/map (K,B) f8 · sed_intrinsic/draws (S,K,B) f8
   profiles/map (K,H,W) f4 · profiles/draws (Sc,K,H,W) f4        # if factorised
   intrinsic/map (K,B,H,W) · intrinsic/draws (Sc,K,B,H,W)        # if not factorised
   convolved/map (K,B,H,W) f4 · convolved/draws (Sc,K,B,H,W) f4 [opt]   # PSF+shifts, NO sky
   params/map (K,Pc) f8 · params/draws (S,K,Pc) f8
       Pc names = ["mu_y","mu_x","size_y","size_x","rho","sersic_n", *emulator_param_names]
/truth [opt]  (mocks)
   convolved (B,H,W) f4 (noise-free) · intrinsic (B,H,W) f4 · param_maps (Nm,H,W) f4
   scalars (Ns,) f8 · component_seds (K,B) f8 · theta (P,)
/posterior
   attrs: sampler, n_chains, n_samples_total, thinning="even"
   theta_map (P,) · theta_draws (S,P) [opt, store_theta] (sampler dtype)
   draw_index (S,) i8 · chain_id (S,) i2
   diagnostics/ attrs scalar summaries (max_rhat, min_ess, ...); datasets rhat (P,), ess (P,) [opt]
   attrs (Laplace): log_evidence_laplace [opt]
/regions [opt]   one dataset per name: (H,W) u1; attrs kind, geometry (JSON str)
                 writer defaults: "all" (all valid pixels), "segment" (detection region)
/metrics/json  (scalar str)  [opt]
/extra/<name>  free-form arrays [opt] (e.g. start maps, alternative-posterior summaries)
```

Invariants (tested):
- `sum_k components/convolved + sky == model/convolved` (to float tolerance).
- `sum_k sed_k ⊗ profile_k == model/intrinsic` (factorised case).
- `model/convolved/draws[i] == fm._model_image(theta_draws[i])`.

Versioning: the reader accepts the same major version; a newer minor version gives a warning and unknown groups are ignored. `schema.validate(h5) -> list[str]` reports structural problems.

Size estimate (64×64, 17 bands, S=200, CorrelatedFieldMap P=4440):

| Item | Raw | gzip+shuffle | 12-bit |
|---|---|---|---|
| convolved draws (200,17,64,64) f4 | 53.1 MiB | ~43 | ~27 |
| intrinsic draws, same shape | 53.1 | ~43 | ~27 |
| param map draws (200,4,64,64) | 12.5 | ~10 | ~6.4 |
| theta draws (200,4440) f8 | 6.8 | ~6.3 | (f4: 3.4) |
| observation + summaries | ~4 | ~3.3 | ~3.3 |
| **Total** | **~130 MiB** | **~105 MiB** | **~70 MiB** |

Real 120×120×9 data at S=200 is ~160 MiB lossless; `--bundle-draws 100` halves that. The GUI keeps draws in RAM (≲250 MB); region sums over (200,17,4096) take ~10 ms.

## 3. Core API: `src/arachne/results/` (Builder A)

```
results/__init__.py   exports reader/regions/schema/display; write_bundle via lazy __getattr__
results/schema.py     SCHEMA_NAME, SCHEMA_VERSION="1.0", BundleContent, TruthContent, ComponentContent, validate()
results/metadata.py   PIVOT_UM: dict[str,float] (all NIRCam W/M/N, HST ACS/WFC3 broad, MIRI);
                      pivot_um(band_names)->np.ndarray (nan if unknown); DEFAULT_LABELS; short_band_name()
results/provenance.py collect_provenance(extra=None)->dict (git via subprocess, fail-safe; versions via importlib.metadata)
results/io.py         write_content(path, content, *, mantissa_bits=None, overwrite=False)->Path   (numpy+h5py only)
results/reader.py     ResultsBundle, load_bundle
results/regions.py    Region, RegionSED, region_sed, total_sed, pixel_sed, component_seds,
                      component_region_seds, region_param_summary, detection_region
results/display.py    percentile_limits, stretch, rgb_composite (numpy only; shared by viz and gui)
results/testing.py    toy_sed, make_synthetic_content, make_synthetic_bundle
results/writer.py     content_from_forward_model, write_bundle  (the ONLY JAX module here)
results/cli.py        `arachne-bundle info|validate <path>`
results/__main__.py   -> cli.main
```

**`schema.py`**: plain dataclasses of numpy arrays that mirror the HDF5 layout one to one.

```python
@dataclass
class TruthContent:
    convolved: np.ndarray | None = None; intrinsic: np.ndarray | None = None
    param_maps: np.ndarray | None = None; scalars: np.ndarray | None = None
    component_seds: np.ndarray | None = None; theta: np.ndarray | None = None

@dataclass
class ComponentContent:
    profile_names: list[str]; param_names: list[str]; factorised: bool
    sed_intrinsic_map: np.ndarray; sed_intrinsic_draws: np.ndarray | None
    profiles_map: np.ndarray | None; profiles_draws: np.ndarray | None
    intrinsic_map: np.ndarray | None; intrinsic_draws: np.ndarray | None
    convolved_map: np.ndarray | None; convolved_draws: np.ndarray | None
    draw_index: np.ndarray | None; params_map: np.ndarray; params_draws: np.ndarray | None
    ordering: str = "size"

@dataclass
class BundleContent:
    band_names: list[str]; pivot_um: np.ndarray; pixel_scale: float
    flux: np.ndarray; variance: np.ndarray; mask: np.ndarray
    model_error_frac: np.ndarray                       # (B,)
    convolved_map: np.ndarray
    convolved_draws: np.ndarray | None = None
    intrinsic_map: np.ndarray | None = None; intrinsic_draws: np.ndarray | None = None
    map_names: list[str] = field(default_factory=list); param_maps_map: np.ndarray | None = None
    param_maps_draws: np.ndarray | None = None; prior_std_maps: np.ndarray | None = None
    param_summaries: dict[str, np.ndarray] | None = None   # used when draws are absent (degraded):
                                                           # keys among p16,p50,p84,mean,std,information
    scalar_names: list[str] = field(default_factory=list); scalars_map: np.ndarray | None = None
    scalars_draws: np.ndarray | None = None
    fixed_params: dict[str, float] = field(default_factory=dict); mass_param: str = ""
    labels: dict[str, str] = field(default_factory=dict); units: dict[str, str] = field(default_factory=dict)
    nuisance: dict[str, np.ndarray] | None = None      # keys as in /model/nuisance
    components: ComponentContent | None = None; truth: TruthContent | None = None
    theta_map: np.ndarray | None = None; theta_draws: np.ndarray | None = None
    draw_index: np.ndarray | None = None; chain_id: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    regions: dict[str, "Region"] = field(default_factory=dict)
    metrics: dict | None = None; extra: dict[str, np.ndarray] = field(default_factory=dict)
    posterior_source: str = "map"; spatial_model: str = ""; forward_model: str = "ForwardModel"
    redshift: float = float("nan"); title: str = ""; degraded: bool = False
    provenance: dict[str, Any] = field(default_factory=dict); psf: np.ndarray | None = None
    def validate(self) -> None: ...                    # shape/consistency checks, raises ValueError
```

`write_content` computes the p16/p50/p84/mean/std summaries itself when draws are present.

**`reader.py`** (numpy and h5py only):

```python
def load_bundle(path: str | Path) -> ResultsBundle
class ResultsBundle:
    @classmethod
    def from_content(cls, content: BundleContent) -> "ResultsBundle"   # in-memory, used by tests
    path: Path | None; attrs: dict; schema_version: str; grid: str; posterior_source: str
    band_names: list[str]; short_names: list[str]; pivot_um: np.ndarray; pixel_scale: float
    image_shape: tuple[int, int]; n_bands: int; n_draws: int; degraded: bool
    has_draws: bool; has_intrinsic: bool; has_truth: bool; has_components: bool
    flux / variance / mask -> (B,H,W) np.ndarray; model_error_frac -> (B,)
    map_names: list[str]; scalar_names: list[str]; labels: dict[str, str]
    regions: dict[str, Region]; metrics: dict; provenance: dict; truth: TruthContent | None
    components: ComponentsView | None
    def band_image(self, b: int, kind: str, stat: str = "p50", *, model_floor: bool = True) -> np.ndarray  # (H,W)
        # kind ∈ {"data","variance","sigma","mask","model","intrinsic","residual","chi",
        #          "truth_model","truth_intrinsic","truth_residual"}
        # stat ∈ {"map","p16","p50","p84","mean","std"}; "p50" etc. fall back to "map" when S=0
        #        (the fallback is reported in self.last_fallback: bool)
        # residual = flux - model[stat]; chi = residual / sqrt(var_eff) with
        #   var_eff = variance*exp(2*median(log_noise_scale)) + (f*model)^2 if model_floor else variance
        # data-derived kinds are NaN where mask == 0
    def cube(self, kind: str, stat: str = "p50", **kw) -> np.ndarray      # (B,H,W), single grid only
    def band_draws(self, b: int, frame: str = "convolved") -> np.ndarray   # (S,H,W); KeyError if absent
    def model_draws(self, frame: str = "convolved") -> np.ndarray          # (S,B,H,W), memoised
    def param_map(self, name: str, stat: str = "p50") -> np.ndarray        # (H,W)
        # stat ∈ {"map","p16","p50","p84","mean","std","truth","diff"(p50-truth),
        #         "z"((p50-truth)/std),"information","prior_std"}; KeyError if unavailable
    def available_param_stats(self) -> list[str]
    def param_draws(self, name: str) -> np.ndarray                         # (S,H,W)
    def pixel_param_draws(self, y: int, x: int) -> dict[str, np.ndarray]   # name -> (S,) (or (1,) MAP)
    def scalar_draws(self) -> dict[str, np.ndarray]                        # name -> (S,)
    def chi2_red_per_band(self, *, model_floor: bool = True) -> np.ndarray # (B,), p50 model
    def close(self) -> None; __enter__/__exit__
class ComponentsView:
    n_components: int; profile_names: list[str]; param_names: list[str]
    def sed_intrinsic(self) -> np.ndarray                     # (S,K,B) or (1,K,B)
    def image(self, k: int, frame: str = "intrinsic", stat: str = "p50") -> np.ndarray  # (B,H,W)
    def image_draws(self, k: int, frame: str = "convolved") -> np.ndarray  # (Sc,B,H,W)
    def params(self) -> np.ndarray                            # (S,K,Pc)
```

Lifetimes: small datasets load eagerly and draws load on first use; `from_content` keeps everything in memory.

**`regions.py`** (numpy only; matplotlib `Path` imported lazily inside `polygon`):

```python
@dataclass(frozen=True)
class Region:
    name: str; mask: np.ndarray            # (H,W) bool on the model grid
    kind: str = "mask"                     # "mask"|"pixel"|"box"|"circle"|"polygon"
    geometry: dict = field(default_factory=dict)
        # pixel {"y","x"}; box {"y0","y1","x0","x1"} (centres inclusive); circle {"cy","cx","r"} (centre dist <= r);
        # polygon {"vertices": [[y,x],...]} (centres inside)
    @classmethod pixel(cls, shape, y, x, name=None) / box(...) / circle(...) / polygon(...) / from_mask(...)
    @property n_pix -> int

@dataclass(frozen=True)
class RegionSED:
    label: str; band_names: list[str]; pivot_um: np.ndarray
    obs_flux: np.ndarray | None            # (B,) sum of flux over region ∩ (mask_b>0)
    obs_err: np.ndarray | None             # (B,) sqrt(sum variance) * error_scale (* exp(s) if fitted)
    n_pix: np.ndarray                      # (B,) valid pixels per band
    model_draws: np.ndarray | None         # (S,B) convolved draws over the SAME pixels
    model_map: np.ndarray | None           # (B,)
    intrinsic_draws: np.ndarray | None     # (S,B) unconvolved, summed over region (mask ignored)
    intrinsic_map: np.ndarray | None
    truth_convolved: np.ndarray | None; truth_intrinsic: np.ndarray | None
    def model_percentiles(self, q=(16, 50, 84), frame="convolved") -> np.ndarray  # (3,B); MAP repeated if no draws
    def chi(self, include_model_spread: bool = True) -> np.ndarray
        # (obs - p50) / sqrt(obs_err^2 + [(p84-p16)/2]^2)
    def frac_residual(self) -> np.ndarray  # (obs - p50) / p50

def region_sed(bundle, region: Region | np.ndarray, *, label: str = "", error_scale: float | np.ndarray = 1.0,
               model_floor: bool = False) -> RegionSED
    # model_floor=True adds (f * sum model)^2 (a coherent floor within the aperture)
def total_sed(bundle, **kw) -> RegionSED            # regions["segment"] if present, else "all", else all pixels
def pixel_sed(bundle, y: int, x: int, **kw) -> RegionSED
def component_seds(bundle) -> list[RegionSED]       # intrinsic per component (obs_* None) + total intrinsic
def component_region_seds(bundle, region) -> list[RegionSED]  # convolved component cubes over the region
def region_param_summary(bundle, region) -> dict[str, np.ndarray]
    # per draw: mass param -> log10 sum 10**m; other fields -> mass-weighted mean; plus "n_pix"
def detection_region(flux, variance, mask, nsigma: float = 3.0, smooth_px: float = 1.0) -> np.ndarray
    # inverse-variance-weighted detection image, smoothed, thresholded, largest connected component (scipy.ndimage)
```

**`display.py`**:

```python
def percentile_limits(img, pct=(0.5, 99.5), *, symmetric=False, mask=None) -> tuple[float, float]
def stretch(img, kind: str = "asinh", lo: float | None = None, hi: float | None = None,
            *, asinh_a: float = 0.1) -> np.ndarray   # -> [0,1], NaN preserved; kind ∈ linear|asinh|log|sqrt
def rgb_composite(r, g, b, *, pct=99.5, asinh_a=0.1) -> np.ndarray  # (H,W,3) in [0,1], shared scale
```

**`writer.py`** (imports JAX). Thins first, then evaluates every product on the same draws:

```python
def content_from_forward_model(
    forward_model: ForwardModel, *, theta_map: ArrayLike | None, samples: ArrayLike | None = None,
    posterior_source: str = "map", n_draws: int = 200, chain_ids: ArrayLike | None = None,
    n_chains: int | None = None, store_intrinsic: bool = True, store_theta: bool = True,
    component_draws: int = 50, order_components: bool = True, truth: TruthContent | None = None,
    prior_std_maps: ArrayLike | None = None, regions: Mapping[str, Region | np.ndarray] | None = None,
    default_regions: bool = True, metrics: Mapping | None = None,
    diagnostics: Mapping[str, Any] | None = None, pivot_um: Sequence[float] | None = None,
    labels: Mapping[str, str] | None = None, title: str = "", command_line: str | None = None,
    extra: Mapping[str, ArrayLike] | None = None, chunk: int = 16,
) -> BundleContent
def write_bundle(path, forward_model, *, mantissa_bits: int | None = None, overwrite: bool = False,
                 **kwargs) -> Path   # = write_content(path, content_from_forward_model(...))
```

How the writer behaves:
- **Thinning:** `idx = unique(round(linspace(0, N-1, S)))`; `chain_id = idx // n_per_chain`.
- **Additive models:** vmap `order_components_by_size` over the samples first.
- **No `theta_map` but samples given:** use the best `log_posterior` among the thinned draws and set attr `best_point="best_draw"`.
- **Convolved cubes:** `posterior_predictive._chunked_map(fm._model_image, ...)`.
- **Intrinsic cubes:** chunked `spatial_model.model_image(split_theta(t)[0], emulator, (H,W))`.
- **Parameter adapters** (dispatched with `hasattr`):
  - `field_maps` (CorrelatedFieldMap): the maps are `field_maps`. Scalars are `global_values` (with `global_names`) plus the derived `total_log_mass`.
  - `component_params` (AdditiveComponentModel): no maps. Components come from `component_seds` + `profiles` (factorised) and `component_shapes` + `component_params` (params). Convolved components use `fm.convolver(comp_k, shifts)` with no sky.
  - Anything else: `decode` reshaped into maps, best effort with a warning.
- **Nuisance draws:** from `fm.nuisance.split`.
- **Multires:** `MultiResolutionForwardModel` raises `NotImplementedError("grid='per_band' reserved in schema 1.0")`.
- **Redshift:** from `fixed_params["redshift"]` when present.
- **Provenance:** from `collect_provenance()`.

**`testing.py`** (owned by A; fully specified here so B and C can write fixtures before A finishes):

```python
def toy_sed(log_mass, av, pivot_um, redshift=2.0) -> np.ndarray
    # (..., B) nJy: 10**(log_mass-8) * 10 * (lam/2)**0.5 * 10**(-0.4*av*(lam/(1+z)/0.55)**-1)
def make_synthetic_content(flavour: str = "field", *, shape=(16, 16), n_draws: int = 20,
                           with_truth: bool = True, seed: int = 0) -> BundleContent
def make_synthetic_bundle(path, flavour: str = "field", **kw) -> Path
```

Common to all flavours:
- Bands `["JWST/NIRCam.F115W","JWST/NIRCam.F200W","JWST/NIRCam.F444W"]`, pivots `[1.154, 1.988, 4.402]`, pixel_scale 0.03, z = 2.0, `model_error_frac = [0.05]*3`.
- `r` is the distance from `((H-1)/2, (W-1)/2)`.
- "PSF": `scipy.ndimage.gaussian_filter(..., sigma=[0.8, 1.0, 1.5][b], mode="constant")`.
- Noise: `sigma_b = 0.03 * max(clean convolved_b)`; flux = clean + N(0, sigma_b) from `np.random.default_rng(seed)`.
- `mask` all ones except `mask[0,0,0] = 0`.
- Regions: `"all"` (all ones) and `"core"` (r < 3, circle geometry).
- `posterior_source="nuts"`, `metrics={"chi2_red": <computed>}`, `title="synthetic <flavour>"`.

Per flavour:
- `"field"`:
  - `map_names=["log_mass","Av"]`, `mass_param="log_mass"`.
  - Truth: `log_mass = 8.0 - 0.15 r`, `Av = 0.2 + 0.6 exp(-r²/18)`.
  - Draws: truth + N(0, 0.05) per pixel, Av clipped at ≥ 0. MAP = truth + 0.02.
  - Intrinsic = `toy_sed` of the maps, transposed to (B,H,W). Convolved = blur.
  - `prior_std_maps = 0.5` everywhere. Scalars `["total_log_mass"]`.
  - Truth populated when `with_truth`.
- `"additive"`:
  - K=2, `profile_names=["gaussian","gaussian"]`, factorised.
  - Profiles: centred, normalised Gaussians with sigma 1.5 px and 4 px.
  - `sed_k = toy_sed([9.5, 10.0][k], [0.1, 0.8][k])`.
  - Draws perturb `log_mass` by N(0, 0.02) and `Av` by N(0, 0.03); `component_draws = S`.
  - Convolved components = blur of `sed_k ⊗ profile_k`.
  - No parameter maps (Nm=0). Pc names as in the schema, with `size = sigma` and `sersic_n = nan`.
- `"degraded"`: the field flavour with S=0. It keeps `convolved_map`, `param_maps_map` and `param_summaries={"p50","std","information"}`; no intrinsic; `degraded=True`, `posterior_source="map"`.

## 4. Static plots: `src/arachne/viz/` (Builder B)

Matplotlib only, no JAX. Every function takes a `ResultsBundle` (or a path, opened through `load_bundle`), accepts `ax=`/`fig=` where sensible, and **returns the `Figure`**.

```
viz/__init__.py  exports the functions below
viz/style.py     band colours by pivot, region colour cycle, rc defaults, label helpers
viz/sed.py       plot_sed, plot_total_sed, plot_region_seds, plot_component_seds
viz/maps.py      plot_param_maps, plot_regions
viz/images.py    plot_band_mosaic, plot_rgb
viz/diagnostics.py plot_chi_histograms, plot_chi2_per_band
viz/summary.py   plot_summary_page, save_all
viz/cli.py       `arachne-plots bundle.h5 --outdir figs [--format png|pdf] [--regions core,segment]`
viz/__main__.py
```

```python
def plot_sed(sed: RegionSED, *, fig=None, show_intrinsic: bool = False, show_truth: bool = True,
             show_map: bool = True, residuals: Sequence[str] = ("sigma", "frac"),
             logy: bool = True, color=None, label_bands: bool = True) -> Figure
    # top: obs points + errors; model p50 line+markers at pivots, 16-84 fill_between; MAP open markers;
    # intrinsic dashed (labelled "intrinsic (unconvolved, not aperture-comparable)"); truth crosses.
    # x log-scale micron, band short names on a twin top axis.
    # "sigma" panel: RegionSED.chi() with ±1/±3 lines; "frac" panel: frac residual (%) with model 16-84 band.
def plot_total_sed(bundle, **kw) -> Figure
def plot_region_seds(bundle, regions: Mapping[str, Region | np.ndarray] | None = None,
                     layout: str = "grid", **kw) -> Figure          # None -> bundle.regions
def plot_component_seds(bundle, *, include_total: bool = True) -> Figure   # no-op note figure if no components
def plot_param_maps(bundle, names: Sequence[str] | None = None,
                    stats: Sequence[str] | None = None, mask_region: str | None = "segment") -> Figure
    # default stats = [s for s in ("truth","p50","std","z","information") if available]; generalises demo plot_laplace
def plot_regions(bundle, regions=None, band: int = -1) -> Figure
def plot_band_mosaic(bundle, bands: Sequence[int] | None = None,
                     kinds: Sequence[str] = ("data", "model", "residual", "chi"),
                     stat: str = "p50", stretch: str = "asinh", model_floor: bool = True) -> Figure
    # data/model share per-band limits; residual symmetric at the data scale; chi fixed [-4,4] RdBu_r
def plot_rgb(bundle, bands: tuple[int, int, int] | None = None,
             kinds: Sequence[str] = ("data", "model", "residual")) -> Figure
def plot_chi_histograms(bundle, *, model_floor: bool = True) -> Figure  # per band, N(0,1) overlay, rms in title
def plot_chi2_per_band(bundle, *, model_floor: bool = True) -> Figure    # chi2_red vs pivot
def plot_summary_page(bundle) -> Figure   # RGB data/model/residual, total SED+residuals, chi2/band, p50 maps
def save_all(bundle, outdir, fmt: str = "png", dpi: int = 150) -> list[Path]
```

Everything must degrade cleanly: no draws means no 16–84 bands; no truth, intrinsic or components means those panels are omitted.

## 5. Interactive viewer: `src/arachne/gui/` (Builder C)

```
gui/__init__.py   ArachneViewer, view; friendly ImportError: "pip install 'arachne[gui]'"
gui/state.py      ViewerState (pure dataclass) + pure functions: image_for(bundle, state)->np.ndarray,
                  sed_for(bundle, state)->list[RegionSED], hist_for(bundle, state)->dict  (numpy only, testable)
gui/panels.py     bokeh builders: image_panel, sed_panel, residual_panel, hist_panel, controls
gui/app.py        ArachneViewer
gui/cli.py        arachne-gui
gui/__main__.py   (sets JAX_PLATFORMS=cpu, XLA_PYTHON_CLIENT_PREALLOCATE=false before importing arachne)
```

```python
class ArachneViewer:
    def __init__(self, bundle: ResultsBundle | str | Path, compare: ResultsBundle | str | Path | None = None): ...
    layout: bokeh.models.LayoutDOM          # built in __init__
    state: ViewerState
    # programmatic controller API (used by callbacks AND tests):
    def set_band(self, b: int) -> None
    def set_image_kind(self, panel: int, kind: str, stat: str = "p50") -> None
    def set_param(self, name: str, stat: str = "p50") -> None
    def select_pixel(self, y: int, x: int) -> None
    def select_indices(self, flat_indices: Sequence[int]) -> None   # box/lasso/poly selection
    def select_circle(self, cy: float, cx: float, r: float) -> None
    def select_region(self, name: str) -> None                      # bundle.regions
    def set_component(self, k: int | None) -> None
    def current_seds(self) -> list[RegionSED]
    def modify_doc(self, doc) -> None       # for the bokeh server
    def show(self): ...                     # Jupyter: panel.pane.Bokeh(layout) if panel else bokeh show(app)
    def serve(self, port: int = 5006, address: str = "127.0.0.1", open_browser: bool = True) -> None
def view(path, **kw) -> ArachneViewer
```

CLI: `arachne-gui bundle.h5 [--compare other.h5] [--port 5006] [--address 127.0.0.1] [--no-browser] [--allow-websocket-origin HOST:PORT ...]`. It prints the URL and the remote recipe: run with `--no-browser` on the node, then `ssh -L 5006:localhost:5006 node`.

Layout (v1):
- **Header:** title, posterior source, S, χ²_red, path; a degraded badge.
- **Left column (controls):**
  - Band `Select` (short names, sorted by pivot).
  - Panel 1 kind: `RadioButtonGroup` with data / model / intrinsic / residual / chi / truth_model.
  - Panel 2 kind (default chi).
  - Model stat `Select` (p50/map/mean/p16/p84/std).
  - Stretch `Select` (linear/asinh/log), percentile `RangeSlider` (0–100, default 0.5–99.5), colormap `Select`.
  - "χ includes model floor" `Toggle`.
  - Parameter `Select` (`map_names`; plus "component k fraction (band)" when components exist) with a parameter stat `Select` populated from `available_param_stats()`.
  - "Mask outside segment" `Toggle`.
  - Component `Select` (total, comp k), shown only when components exist.
  - Selection mode (tap-pixel / box / lasso / polygon / circle), circle radius `Slider`, named-regions `Select` (bundle regions).
  - `Button`s to clear and to keep the current selection as an overlay (up to 5 coloured overlays).
  - SED options `CheckboxGroup` (intrinsic, truth, MAP, log y, total in grey).
- **Centre:** three image figures sharing `x_range`/`y_range` (linked pan and zoom): Panel 1 (band, kind), Panel 2 (band, kind), Panel 3 (parameter map).
  - Each uses `image(image="img", x=-0.5, y=-0.5, dw=W, dh=H)` with a `LinearColorMapper` and `ColorBar`; stretched images are precomputed in numpy with `display.stretch`.
  - `HoverTool` shows (x, y, value).
  - Region outline: `rect` glyphs on the selected pixels, `fill_alpha=0.15`.
- **Right:** SED figure (log-x micron; observed `Whisker` + scatter; model p50 line, p16–p84 `varea`; MAP open markers; total in grey) and a residual figure (σ, ±1/±3 `Span`s), with a fractional tab.
- **Bottom:** a row of posterior histograms (`quad`) for the selected pixel or region: one per map param (region values from `region_param_summary`), plus scalars. Truth drawn as a `Span`; MAP as a `Span` when S=0.

How selections work:
- Each image figure carries an invisible `scatter` of pixel centres (alpha 0, `output_backend="webgl"`) with `BoxSelectTool`, `LassoSelectTool` and `PolySelectTool`. `source.selected.on_change("indices", ...)` calls `select_indices`.
- Tap uses `fig.on_event(events.Tap, ...)`: the pixel is `round(event.y), round(event.x)`, or a circle of the slider radius in circle mode.
- All recomputation goes through `state.py`; callbacks only push arrays into `ColumnDataSource.data`.

Compare mode (stretch goal C-S1): needs the same `image_shape`. The second bundle's model SED is overlaid in another colour for the same region, and a "bundle A/B" `Select` switches the image panels. Bands are matched by name.

## 6. Integration (Builder A)

**`examples/demo_correlated_field.py`:**
- `laplace_posterior` additionally returns `"thetas"` (S,P).
- `nuts_posterior` additionally returns `"theta_samples"` (N,P) and `"n_chains"`, with the chain-concatenated samples.
- `spiral_truth` also returns its arm mask, giving regions `"arms"` (`win*arm > 0.3`) and `"interarm"`. Both truths add a `"galaxy"` region (the `sel` used for the metrics).
- At the end of `main`: `write_bundle(outdir/"results.h5", fm, theta_map=post["theta"] if post else res.theta, samples=<nuts thetas> or <laplace thetas> or None, posterior_source=..., truth=TruthContent(convolved=clean, intrinsic=render_model.model_image(theta_truth, emulator, (n,n)), param_maps=truth_maps), prior_std_maps=pstd, regions=..., metrics=metrics, diagnostics=<rhat/ess>, pivot_um=PIVOT_UM, labels=LABELS, title=outdir.name)`.
- The existing npz/PNG outputs are unchanged.
- New flags:
  - `--no-bundle`
  - `--bundle-draws 200`
  - `--bundle-mantissa-bits` (default None)
  - `--bundle-no-intrinsic`
  - `--bundle-all` (also writes `results_laplace.h5` when NUTS ran)
  - `--bundle-plots` (phase 2: `arachne.viz.save_all` into `figures/bundle/`)

**`scripts/convert_correlated_field_run.py <run_dir> [--out results.h5] [--pixel-scale 0.03] [--emulator CKPT]`:**
- Imports the demo by file path (`importlib.util.spec_from_file_location`) to reuse `FIXED`, `SPS_PARAM_NAMES`, `NIRCAM_PIVOT_UM` and `LABELS`.
- Builds a `BundleContent` with:
  - `flux = data`, `variance = sigma²` broadcast, `mask = 1`, bands from `metrics["bands"]`;
  - `convolved_map = model_image`, `param_maps_map = map`;
  - `param_summaries` from `nuts.npz`, falling back to `laplace.npz` (median→p50, std, information);
  - the other posterior's summaries under `/extra/<laplace|nuts>_{p50,std,information}`;
  - truth param maps when `metrics["truth"]` is set, and `start` under `/extra/start_maps`;
  - `theta_map`, metrics, `degraded=True`, `posterior_source="map"`, `title=<truth>_<bands>`.
- `--emulator` (optional, needs JAX) adds `intrinsic_map` = emulator applied to the MAP field maps + `FIXED`. The production checkpoint is at `~/Documents/projects/arachne/emulators/parrot_emulator_v2.eqx`; "toy" runs use the demo's `ToySEDEmulator`.
- What a degraded bundle supports: SEDs and residuals for any region against the MAP model, and parameter median/std/information maps. It has **no** model uncertainty bands and no per-pixel posteriors (the GUI banner says so).
- Running it on the four existing run directories is part of acceptance.

## 7. Packaging, docs and tests

`pyproject.toml` (C edits; nobody else):
```toml
[project.optional-dependencies]
gui = ["bokeh>=3.4,<4"]
gui-jupyter = ["bokeh>=3.4,<4", "panel>=1.4"]
test = ["pytest", "ruff==0.11.7", "bokeh>=3.4,<4"]   # so CI runs the GUI smoke tests
[project.scripts]
arachne-gui = "arachne.gui.cli:main"
arachne-plots = "arachne.viz.cli:main"
arachne-bundle = "arachne.results.cli:main"
```

Tests (CPU, tiny, no conftest edits; each test file builds its own fixtures through `arachne.results.testing` and `tmp_path`):
- **A**
  - `tests/test_results_schema_io.py`:
    - round trip content → h5 → reader for all three flavours;
    - optional pieces absent;
    - version checks (major mismatch raises, minor warns);
    - `mantissa_bits=12` relative error < 3e-4;
    - metrics JSON larger than 64 KB;
    - `validate`.
  - `tests/test_results_regions.py`:
    - full mask equals manual sums;
    - masked pixel excluded per band;
    - variance propagation;
    - pixel/box/circle/polygon membership;
    - `RegionSED` percentiles, chi and frac;
    - `region_param_summary` mass sum;
    - `detection_region`.
  - `tests/test_results_writer.py`:
    - CorrelatedFieldMap 3 bands × 16×16 with the toy emulator (pattern from `tests/test_correlated_field.py`);
    - AdditiveComponentModel with `LinearMassEmulator` (as in `tests/test_additive_model.py`), with and without `NuisanceModel(fit_sky=True)`;
    - the three schema invariants;
    - thinning indices;
    - MAP-only;
    - multires raises.
  - `tests/test_results_convert.py`: fake run dir built with `np.savez` in `tmp_path`.
  - `tests/test_no_jax_imports.py`: `ast` scan of `results/{schema,io,reader,regions,display,testing,metadata,provenance,cli}.py`, `viz/*.py` and `gui/*.py` for `jax`/`equinox` imports.
- **B** — `tests/test_viz.py` (Agg):
  - every function returns a `Figure` for the field, additive and degraded flavours;
  - expected panel and axes counts (e.g. `plot_sed` with two residual panels → 3 axes + twin);
  - the `save_all` file list;
  - `cli.main([...])`.
- **C** — `tests/test_gui_state.py` (numpy only) and `tests/test_gui.py` (`pytest.importorskip("bokeh")`):
  - build `ArachneViewer` for all three flavours;
  - call each controller method and assert the source data equal `region_sed` / `band_image` outputs;
  - degraded mode hides the bands and histograms;
  - `cli.main(["--help"])`;
  - with `importorskip("panel")`: `pn.pane.Bokeh(viewer.layout).get_root()` succeeds;
  - missing bokeh (`monkeypatch` `sys.modules["bokeh"]=None`) gives the friendly ImportError;
  - never start a server in tests.

Docs:
- A adds a README section "Results bundles, plots and the interactive viewer" after "Diagnostics, model comparison and posterior predictive checks", with three subsections: *Bundles* (A), *Static plots* (B), *Interactive viewer* (C). B and C edit only their own subsection.
- At the end, A adds the `.claude/CLAUDE.md` module-map rows and the `docs/source/API.rst` automodule entries.

## 8. Work breakdown and ownership

| Builder | Owns (exclusive) | Phase |
|---|---|---|
| **A** | `src/arachne/results/**`, `src/arachne/__init__.py` (add `ResultsBundle, load_bundle, write_bundle, Region, RegionSED, region_sed, total_sed` + `__all__`; never viz/gui), `examples/demo_correlated_field.py`, `scripts/convert_correlated_field_run.py`, `tests/test_results_*.py`, `tests/test_no_jax_imports.py`, README section skeleton + Bundles subsection, `.claude/CLAUDE.md`, `docs/source/API.rst` | **P0 (first):** schema, metadata, io, reader, regions, display, testing. **P1:** provenance, writer, cli, demo, converter. **P2:** `--bundle-plots`, docs, convert the existing runs. |
| **B** | `src/arachne/viz/**`, `tests/test_viz.py`, README "Static plots" subsection | Codes against A's P0. |
| **C** | `src/arachne/gui/**`, `pyproject.toml`, `tests/test_gui*.py`, README "Interactive viewer" subsection | Codes against A's P0; `state.py` first, then panels and app. |

Interfaces between builders: B and C use only what Section 3 defines (`load_bundle`, `ResultsBundle` (+ `from_content`), `Region`, `RegionSED`, `region_sed`/`total_sed`/`pixel_sed`/`component_seds`/`component_region_seds`/`region_param_summary`, `display.*`, `testing.make_synthetic_*`). Any change to these goes through A, and this spec is updated first. C does not import `arachne.viz`, and B does not import `arachne.gui`.

Acceptance:
- **A**
  - Tests pass on CPU and ruff is clean.
  - `python examples/demo_correlated_field.py --truth prior --quick --laplace --float64` (toy emulator) writes `results.h5` that `arachne-bundle validate` passes.
  - Bundle size at 64×64×17, S=200 within 20% of the table.
  - The converter produces valid degraded bundles for all four existing run directories.
  - Bundle writing adds < 60 s on the H100.
- **B**
  - Tests pass.
  - `arachne-plots results.h5 --outdir figs` writes total SED (+σ and fractional residuals), region SEDs, param maps, band mosaic, RGB, χ histograms, χ² per band and a summary page, for both a full and a degraded bundle.
  - The SED residual panel matches `RegionSED.chi()` exactly.
- **C**
  - Tests pass.
  - Locally, `arachne-gui results.h5` opens a browser where:
    - band, kind or tap updates take < 300 ms;
    - box/lasso/polygon/circle selection updates the SED, residuals and histograms;
    - the three images pan and zoom together;
    - the components selector appears only for additive bundles;
    - a degraded bundle shows the banner and MAP-only SEDs.
  - Works on the node with bokeh only (`--no-browser` + `ssh -L`).
  - `ArachneViewer(path).show()` renders in local JupyterLab (Panel path).
  - Stretch goal: `--compare`.

## 9. Open questions (recommended defaults in brackets)

1. Default S: 200 or 100 [200 for 64² cutouts; the demo exposes `--bundle-draws`].
2. Lossy mantissa rounding by default [off; it is a knob].
3. Aperture errors for drizzled data with correlated noise [`error_scale=1`; per-band factors are a later addition].
4. Model-error floor in region SED error bars [off for error bars; per-pixel chi maps include it by default, matching the likelihood].
5. Saving GUI-drawn regions back to disk [v1: "export regions" to a sidecar `<bundle>.regions.json`; the bundle is never modified].
6. Lazy `arachne/__init__` to avoid importing JAX in the viewer [defer].
