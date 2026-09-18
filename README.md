# arachne

**JAX-native SED fitting and image-level forward modelling of galaxy populations.**

arachne trains fast neural emulators of stellar-population-synthesis (SPS) photometry and
uses them for GPU-accelerated Bayesian inference — from fitting a single integrated SED, to
fitting a catalogue of thousands of galaxies in parallel, to full spatially-resolved
image-level forward modelling with PSF convolution.

## Overview

Two use cases share the same emulator + inference core:

1. **Catalogue / single-galaxy SED fitting** (`scripts/fit_*.py`) — integrated photometry for
   one galaxy or a whole catalogue, with GPU-batched Pathfinder+NUTS for speed or gradient-free
   Nested Slice Sampling (NSS) for multimodal posteriors and evidence (`logZ`).
2. **Image-level forward modelling** — a galaxy as a sum of `K` additive light components
   (`AdditiveComponentModel`: a Gaussian, Sérsic or point-source surface-brightness profile
   carrying the emulator SED of its *own* SPS parameters and total stellar mass) or as a
   free-form per-pixel map; the full multi-band image forward-modelled through FFT PSF
   convolution and optional instrumental nuisances; initialised *blind* from the data; sampled
   with NSS, NUTS or MCLMC. Bands may share one pixel grid (`ForwardModel`) or each keep its own
   WCS grid (`MultiResolutionForwardModel`), and a sample of equal-shape cutouts can be fitted
   in one vmapped program (`BatchedForwardModel`).

The entire pipeline is pure JAX/Equinox — end-to-end differentiable, JIT-compilable to GPU,
and has no PyTorch dependency at inference time.

Full API documentation lives in `docs/` (Sphinx): `pip install -e ".[docs]" && cd docs && make html`.

## Emulators

Both emulators are trained directly from the HDF5 model library produced by
[synference](https://github.com/tHarvey303/synference).

### ParrotEmulatorV2 (recommended)

The current production emulator (`arachne.ParrotEmulatorV2` / `arachne.load_emulator`), built
on three validated improvements over the original Parrot design (see the module docstring in
[`src/arachne/emulator/parrot_emulator_v2.py`](src/arachne/emulator/parrot_emulator_v2.py) for
the full rationale and ablations):

1. **Analytic mass factorisation** — photometry is exactly linear in stellar mass, so the
   network predicts flux per 10⁹ M☉ and `log_mass` is reapplied analytically at inference.
   (Every blind-initialisation trick downstream depends on this exact linearity.)
2. **Fourier redshift features** — sin/cos encodings let the network represent the sharp
   Lyman-break cutoff without extreme depth (the single largest accuracy win, ~2×).
3. **arsinh-compressed SFH-ratio inputs** — preserves the informative core of the
   Student-t-distributed `logsfr_ratio_*` inputs that plain z-scoring squashes.

```bash
python scripts/train_parrot_emulator_v2.py --library galaxy_library.hdf5 \
    --output outputs/emulators/parrot_emulator_v2.eqx
python scripts/validate_parrot_emulator.py --emulator outputs/emulators/parrot_emulator_v2.eqx \
    --library galaxy_library.hdf5 --output-dir outputs/validation/
```

`load_emulator(path)` (`arachne.load_emulator`) loads either a V1 or V2 checkpoint
transparently — use it in downstream code instead of `ParrotEmulatorV2.load` directly. The
architecture-search harness that produced the V2 recipe (Optuna HPO, ablation sweeps, scaling
studies) lives in `scripts/experiments/` — see [Experiments](#experiments--training-harness).

### ParrotEmulator (legacy) and SPSMLPEmulator

`ParrotEmulator` (v1) is the original Parrot-style MLP
([Mathews et al. 2023](https://iopscience.iop.org/article/10.3847/1538-4357/ace720)) with
GELU activations and an arsinh-magnitude output transform; superseded by V2 above but kept for
compatibility. `SPSMLPEmulator` uses the [Speculator](https://arxiv.org/abs/1911.11778)
architecture (Alsing et al. 2020), trained in log10-flux space.

All emulators produce `.eqx` checkpoints — native Equinox pytrees, differentiable and
JIT-compilable with no PyTorch dependency.

## Installation

```bash
git clone https://github.com/tHarvey303/arachne
cd arachne
pip install -e ".[dev,test]"

# JAX with GPU support — install the CUDA plugin at the SAME version as jax/jaxlib,
# or JAX will silently fall back to CPU instead of erroring (see Gotchas below):
pip install "jax[cuda12]==<jaxlib version>"

# synference (provides the HDF5 model libraries used to train the emulator):
pip install -e /path/to/synference
```

Nested Slice Sampling requires `blackjax>=1.6.2` (nested sampling shipped upstream in
blackjax 1.6; no fork is needed any more — see [Samplers](#samplers)).

In a venv shared across projects (e.g. a department HPC venv), **do not upgrade or remove
packages without checking the blast radius first** — `pip`'s resolver will happily bump
transitive dependencies (numpy, jax, jaxlib, ...) that other projects pin to older versions.
See [Gotchas](#gotchas) for a concrete example of this breaking the environment.

## Catalogue / Single-Galaxy SED Fitting

Three scripts share the same forward model and likelihood machinery
(`scripts/fit_catalogue.py` is the shared driver — the others import from it); the
per-parameter prior specifications they use live in the library, in
[`arachne.priors.specs`](src/arachne/priors/specs.py) (see [Priors](#priors) below):

| Script | Use for |
|---|---|
| `fit_sed.py` | One mock galaxy, minimal reference example (no catalogue needed). |
| `fit_one_galaxy.py` | One real or mock galaxy with verbose diagnostics — debug fits here before scaling up. |
| `fit_catalogue.py` | Production catalogue fitting, thousands of galaxies. |

`fit_catalogue.py` supports two inference backends via `--sampler`:

```bash
EMU=outputs/emulators/parrot_emulator_v2.eqx
# GPU-batched Pathfinder + multi-chain NUTS (default; thousands of galaxies/min)
python scripts/fit_catalogue.py catalogue.fits bands_config.json --emulator $EMU \
    --sampler nuts --n-chains 2 --n-samples 500
# Nested slice sampling: gradient-free, multimodal, reports logZ; sequential per galaxy
# (~10-30x slower than the batched NUTS path), so use it for QA and hard cases.
python scripts/fit_catalogue.py catalogue.fits bands_config.json --emulator $EMU \
    --sampler nss --num-live 500 --num-inner-steps 24 --num-delete 50
```

`--print-config-template --emulator <ckpt>` prints a band-config JSON template (bands + default
priors); see `scripts/configs/*.json` for worked examples. `fit_catalogue.py` also defines
`SPS_PARAM_NAMES` and `PARAM_BOUNDS` — the 12-parameter emulator training domain (`redshift`,
`log_mass`, `slope`, `fesc_lya`, `dust_bump_amplitude`, `log10metallicity`, `Av`,
`logsfr_ratio_0..4`) that image-level fits should reuse as their bounds.
`diagnose_parrot_emulator.py` and `validate_parrot_emulator.py` round out the toolkit for
inspecting emulator accuracy before trusting it in a fit.

## Experiments / Training Harness

`scripts/experiments/` holds the emulator architecture-search harness used to derive the V2
recipe: `hpo_lab.py` (Optuna HPO), `emulator_lab.py` / `train_final_v2.py` (ablation training),
`bench_inference.py`, `make_summary_figs.py` and shell drivers. Results go to `scripts/outputs/`
(gitignored; only the harness code is tracked). `archive_gmm_blend/` holds retired diagnostics
from the old parameter-blending demo — kept for the record, they do not run against the current
code.

## Image-Level Forward Modelling

### Quick Start

A blind two-component fit, complete and runnable: it builds a toy mass-linear emulator (a
stand-in for a trained `ParrotEmulatorV2` checkpoint) and a synthetic three-band cutout, then
fits them blind. Swap the toy emulator for `load_emulator(...)` and the synthetic arrays for
`ObservationCube.from_fits(...)` / `PSFModel.from_fits(...)` and nothing else changes.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from arachne import (
    AdditiveComponentModel, ForwardModel, NSSSampler, ObservationCube, PSFModel,
    SPSEmulator, blind_initial_theta, multistart_map,
)
from arachne.priors import build_component_log_prior, resolve_prior_specs


class ToyEmulator(SPSEmulator, eqx.Module):
    """flux_b proportional to 10**(log_mass - 9), reddened by Av.  Real fits: load_emulator()."""

    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)

    @property
    def param_names(self):
        return self._param_names

    @property
    def band_names(self):
        return self._band_names

    def predict(self, params):
        """(K, N_params) physical rows -> (K, N_bands) fluxes in nJy."""
        mass = 10.0 ** (params[:, 0:1] - 9.0)
        lam = jnp.array([1.15, 2.0, 2.77])                       # micron, one per band
        redden = 10.0 ** (-0.4 * params[:, 2:3] / lam[None, :])
        return 3e3 * mass * redden * (1.0 + 0.2 * params[:, 1:2])


BANDS = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]
NAMES = ["log_mass", "log_age", "Av"]                            # emulator input order
BOUNDS = {"log_mass": (6.0, 12.0), "log_age": (7.0, 10.1), "Av": (0.0, 4.0)}
emulator = ToyEmulator(_param_names=NAMES, _band_names=BANDS)

# Per-band PSF kernels (real work: PSFModel.from_fits({band: path})).
yy, xx = np.mgrid[-6:7, -6:7]
kernels = np.stack([np.exp(-(yy**2 + xx**2) / (2 * s**2)) for s in (1.0, 1.3, 1.7)])
kernels = (kernels / kernels.sum(axis=(1, 2), keepdims=True)).astype(np.float32)
psf = PSFModel(kernels=kernels, band_names=BANDS)

H = W = 32
spatial_model = AdditiveComponentModel(
    n_components=2,                       # bulge + disk
    emulator_param_names=NAMES,
    param_bounds=BOUNDS,                  # = the emulator's training domain
    image_shape=(H, W),
    mass_param="log_mass",                # per-component TOTAL stellar mass
    sps_log_prior=build_component_log_prior(
        NAMES, resolve_prior_specs(NAMES, None, BOUNDS), BOUNDS
    ),
    profiles=["sersic", "gaussian"],      # compact bulge + smooth disk
    pixel_scale=0.031,                    # arcsec/px: mu and sigma are now in arcsec
    normalisation="analytic",             # unit flux over the whole plane
    oversample=3,                         # cuspy Sersic needs sub-pixel sampling
)

# Synthetic data: a compact n=4 bulge plus a bigger, dustier disk.  Raw (unconstrained) theta;
# the truth is used ONLY to make the image, never to start the fit.
truth = jnp.array([
    0.0, 0.0, np.log(0.08), np.log(0.07), 0.0, np.log(4.0),      # sersic shape block
    1.0, 0.6, -1.2,                                              # its log_mass, log_age, Av
    0.03, -0.02, np.log(0.25), np.log(0.18), 0.3,                # gaussian shape block
    0.7, -0.4, 0.4,
])
def cube(flux, variance):                # real work: ObservationCube.from_fits(...)
    flux = np.asarray(flux, np.float32)
    return ObservationCube(flux=flux, variance=np.full(flux.shape, variance, np.float32),
                           mask=np.ones(flux.shape, np.float32), band_names=BANDS,
                           pixel_scale=0.031)

clean = np.asarray(ForwardModel.build(obs=cube(np.zeros((3, H, W)), 1.0), psf_model=psf,
                                      spatial_model=spatial_model,
                                      emulator=emulator)._model_image(truth))
sigma = 0.05 * clean.max()
obs = cube(clean + np.random.default_rng(0).normal(0.0, sigma, clean.shape), sigma**2)

# model_error_frac adds a fractional model-error floor in quadrature: a few per cent of
# emulator systematics dominate photon noise once ~1e4 pixels are summed.
forward_model = ForwardModel.build(
    obs=obs, psf_model=psf, spatial_model=spatial_model, emulator=emulator,
    model_error_frac=0.05,
)

# Blind init: image moments -> neutral SPS values -> Adam with linear mass re-solves, from a
# few generic SPS archetypes; the best -log_posterior wins.  No truth, no zeros.
theta0 = blind_initial_theta(spatial_model, forward_model.observation)
map_result = multistart_map(forward_model, theta0, archetypes=[{}, {"Av": 0.3}, {"Av": 2.0}])
print("blind MAP -log_post", float(map_result.neg_log_posterior),
      "vs truth", float(-forward_model.log_posterior(truth)))

# Nested slice sampling: live points from the prior (no starting point), equal-weight samples
# plus the evidence, so fits with different K can be compared.  See "Samplers" for when to
# prefer arachne.inference.laplace.run_whitened_nuts instead.
result = NSSSampler(forward_model, num_live=100, n_samples_out=200).run(jax.random.PRNGKey(0))
print("logZ", result.logZ, "+-", result.logZ_err, "ESS", result.ess)

# Components are exchangeable: order compact-first before summarising.
samples = jax.vmap(spatial_model.order_components_by_size)(result.samples)
mu, size, rho, sps_phys = jax.vmap(spatial_model.component_params)(samples)
print(np.percentile(np.asarray(sps_phys[:, :, 0]), [16, 50, 84], axis=0))  # log_mass per component
result.to_hdf5("posterior.h5")
```

`theta` is a flat vector of length `Σ_k (n_shape_k + N_free) + N_shared`, laid out as `K`
component blocks `[shape_k..., sps_raw_k...]` followed by the shared raws; every SPS value is
mapped to its bounds by a sigmoid. Use `spatial_model.split_theta` / `join_theta` /
`component_params` / `shape_raw` / `sps_raw` rather than slicing by hand.

### Data Flow

```
synference HDF5 library ──▶ train_parrot_emulator_v2.py ──▶ emulator.eqx
                                     (mass-factorised MLP, Fourier-z, arsinh SFH inputs)
  (inference time)
theta (n_params,)  ← BlackJAX NSS / NUTS / MCLMC  (from blind_initial_theta → multistart_map)
   │  split_theta → (theta_spatial, theta_nuisance)
   ▼ SpatialModel.model_image(theta_spatial, emulator, (H, W))
   │    AdditiveComponentModel:  K SEDs (K, N_bands) ← emulator.predict,
   │                             × K normalised profiles (K, H, W) → Σ_k F_k ⊗ P_k
   │    FreeFormPixelMap / GMM:  decode → pixel_params (H*W, N_sps) → predict per pixel
   ▼ PSFConvolver (FFT; nuisance sub-pixel shifts as a phase ramp) + per-band nuisance sky
   ▼ GaussianLikelihood (model-error floor, nuisance log_noise_scale)
   │    + SpatialModel.log_prior() + NuisanceModel.log_prior()
log_likelihood(theta), log_prior(theta), log_posterior(theta)  ← scalar, differentiable
   ├─▶ NSS: log_prior + log_likelihood separately, live points from sample_prior → samples + logZ
   └─▶ NUTS / MCLMC: jax.grad(log_posterior) → leapfrog
```

### Spatial Models

| Model | Parameters | Best for |
|---|---|---|
| `AdditiveComponentModel` (**recommended**) | `Σ_k (n_shape_k + N_free) + N_shared` | Bulge/disk and other multi-component decompositions. Light is additive: each component has its own SED at its own total mass times a normalised profile; `K` emulator calls per likelihood, independent of image size. Proper prior with `sample_prior` (works with NSS); supports `fixed_params`, `shared_param_names`, mixed profiles, arcsec coordinates and analytic normalisation. |
| `GaussianMixtureSpatialModel` | `K × (5 + N_sps)` | **Caveat:** blends SPS *parameters* per pixel with softmax weights that sum to 1, so it has no surface-brightness profile — every pixel carries a full galaxy mass and a lone "disk" fills the whole frame at constant flux. Only for smooth parameter *gradients* across an already-resolved source; not for bulge/disk work. Kept for compatibility. |
| `FreeFormPixelMap` | `H × W × N_sps` | Maximum flexibility; per-pixel emulator calls, requires GPU; L2 smoothness prior. |

All models implement `decode` / `log_prior`; `model_image` (the hook `ForwardModel` calls) has
a per-pixel default and is overridden by `AdditiveComponentModel`. Note that
`AdditiveComponentModel.decode` returns a *summary* map (mass-weighted component parameters,
with log10 stellar-mass surface density in the mass column) for plotting — the likelihood
never uses it.

### Light profiles

Each component's shape is delegated to a `Profile`
([`src/arachne/spatial/profiles.py`](src/arachne/spatial/profiles.py)), selected per component
with `profiles="gaussian"` (one name for all) or `profiles=["sersic", "gaussian", "point"]`
(one per component; pre-built instances such as `SersicProfile(log_n_sd=0.3)` are accepted):

| Profile | Shape parameters | Notes |
|---|---|---|
| `"gaussian"` | 5: `mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho` | The historical parametrisation. `sigma = exp(log_sigma)`, `rho = tanh(atanh_rho)` clipped to ±0.99. |
| `"sersic"` | 6: the above `+ log_n` | `sigma` is the effective (half-light) radius along each axis; `n = exp(log_n)` is *softly* clamped into `n_bounds` (default `(0.3, 10)`) so it keeps a gradient at the wall; prior `Normal(log 2, 0.7)` on `log_n`. Use `oversample >= 3` — a single sample per pixel badly under-resolves `n ≳ 2`. |
| `"point"` | 2: `mu_y, mu_x` | An AGN / nuclear source: unit flux in a fixed circular Gaussian of width `point_sigma` (0.5 px by default), so after PSF convolution the component *is* the PSF at `mu`. |

Because profiles have different block lengths, `split_theta` returns a **list** of blocks for a
mixed-profile model (and a rectangular `(K, ·)` array when every component shares a profile), and
`order_components_by_size` only permutes components that share a profile.

Two more switches change what the coordinates and the normalisation mean. `pixel_scale=None`
(default) keeps `mu`/`sigma` in **pixel indices**, while `pixel_scale=0.031` switches to
**arcsec offsets from the frame centre** `((H-1)/2, (W-1)/2)` — which is what lets the same
component be rendered on another band's grid. `normalisation="frame"` (default) divides each
profile by its sum over the model's own grid; `normalisation="analytic"` normalises the integral
over the whole plane, so flux falling outside the cutout is correctly lost, and is **required**
for `model_image_on` (arbitrary grids) and for multi-resolution fitting.

### Initialisation

`arachne.inference.initialisation` gives a deterministic, truth-free start: `image_moments`
(centroid + rms size from the S/N-stacked image) → `blind_initial_theta` (all components at the
centroid on a geometric size ladder, neutral SPS values, Sérsic `n=4` for the most compact
component and `n=1` for the rest) → `find_map` / `multistart_map` (Adam on `-log_posterior`).
It works because the `K` component masses are **linear amplitudes** — flux ∝ `10**log_mass` —
so given every other parameter they solve a `K × K` weighted least-squares system
(`solve_component_masses`, which also accounts for a fitted sky, shifts and noise scale).
`find_map` re-solves them between Adam rounds, collapsing the dominant mass/size/colour
degeneracy far faster than gradient steps, and `multistart_map` starts from a few generic SPS
archetypes (low vs high dust) to avoid the wrong basin of the dust/age degeneracy. Components
are ordered compact-first at the end (`order_components_by_size`) to fix the label symmetry.

With nuisances attached use `blind_initial_full_theta(forward_model)` (or wrap a spatial vector
with `forward_model.initial_theta_from_spatial`) so theta has the right length;
`blind_initial_theta` also accepts a `MultiResolutionObservation`, where `ref_band=` picks the
band whose moments seed the centre and size. Very close to the mode the linear mass re-solve
maximises the *floor-free* likelihood and can therefore **raise** `-log_posterior`: polish with
`find_map(..., resolve_masses=False)` at a low learning rate (the demo's `polish_map` anneals
Adam, then takes a few damped Newton steps).

### Priors

Per-parameter physical-space priors are dict specs (`{"dist": "studentt", "df": 2, "loc": 0,
"scale": 0.3}`) handled by [`arachne.priors.specs`](src/arachne/priors/specs.py):
`DEFAULT_PRIORS` (Student-t on `logsfr_ratio_*` — the emulator training grid was drawn from
it — uniform elsewhere), `resolve_prior_specs` (merge user overrides, validate against bounds),
`build_log_prior` (catalogue fits, `(N,)` vector), `build_component_log_prior` (image fits,
`(K, N_emulator)` matrix; shared parameters counted once, fixed ones ignored),
`prior_config_template` and `sigmoid_log_jacobian`. Supported `dist` values: `uniform`,
`loguniform`, `normal`, `studentt`, `halfnormal`, `exponential`, `lognormal`. The spatial models
add their own sigmoid Jacobian (uniform in *physical* space) plus each profile's shape prior
(Gaussians on the centre, log sizes, `atanh_rho` and `log_n`), so
`AdditiveComponentModel.log_prior` is a proper density, `logZ` values are comparable across `K`,
and `sample_prior` draws from exactly that density (minus the optional `sps_log_prior` term) for
NSS's live points.

### Instrumental nuisances

Real imaging is never a clean realisation of the astrophysical model, and holding the
instrumental terms fixed at the wrong value biases the inferred stellar populations, so
`NuisanceModel` appends up to three per-band blocks to theta and marginalises over them:

```python
from arachne import NuisanceModel

nuisance = NuisanceModel(
    n_bands=len(BANDS),
    fit_sky=True,              # residual background pedestal, nJy/pixel
    fit_shifts=True,           # (dy, dx) astrometric registration, pixels
    fit_noise_scale=True,      # log multiplier on the noise sigma (variance x exp(2s))
    shift_reference_band=0,    # pin band 0's shift: see below
)
forward_model = ForwardModel.build(obs=obs, psf_model=psf, spatial_model=spatial_model,
                                   emulator=emulator, model_error_frac=0.05, nuisance=nuisance)
theta0 = blind_initial_full_theta(forward_model)
print(nuisance.param_names(BANDS))   # sky[...], dy[...], dx[...], log_noise_scale[...]
```

The sub-vector is ordered `sky` (N), then `[dy, dx]` for each *free* band, then
`log_noise_scale` (N), and every fitted parameter carries a normalised zero-mean Gaussian prior
(`sky_prior_sigma=1.0` nJy, `shift_prior_sigma=0.5` px, `noise_scale_prior_sigma=0.3`);
`nuisance.split(theta_n)` always returns all three blocks, zero-filled when disabled.

**Always pass `shift_reference_band=` when fitting shifts** — a shift common to every band is
exactly degenerate with moving every spatial component, so one band must be pinned and the free
parameter count is `2 * (n_bands - 1)`. `forward_model.split_theta`,
`initial_theta_from_spatial` and `sample_prior` all know about the nuisance tail, and
`solve_component_masses` is nuisance-aware.

### Multi-resolution fitting (optional)

NIRCam short-wavelength mosaics are natively 0.02–0.03″/px and long-wavelength ones 0.04–0.06″,
so a single-grid fit must drizzle everything onto a common grid — throwing away SW resolution or
correlating the LW noise. `MultiResolutionForwardModel` avoids that: **the data are never
resampled**. Each band keeps its own pixel grid and PSF, and the same physical components are
rendered analytically on that band's pixel centres through its WCS. Because `GaussianLikelihood`
is a plain sum over pixels, the joint log-likelihood is *exactly* the sum of the per-band ones.

```python
from arachne import MultiResolutionForwardModel, blind_initial_full_theta
from arachne.data.dja import fetch_dja_native_cutout

mro = fetch_dja_native_cutout(ra, dec, size_arcsec=6.0, filters=["f200w", "f444w"],
                              psfs={"JWST/NIRCam.F200W": (kernel_f200w, 0.03),
                                    "JWST/NIRCam.F444W": (kernel_f444w, 0.03)})
spatial_model = AdditiveComponentModel(..., pixel_scale=0.02, normalisation="analytic")
fm = MultiResolutionForwardModel.build(mro, spatial_model, emulator,
                                       model_error_frac=0.05, oversample=3)
theta0 = blind_initial_full_theta(fm)          # blind_initial_theta accepts the MRO directly
row, col = fm.sky_to_pixel("JWST/NIRCam.F444W", dy, dx)   # overlay centres on a band's image
```

`n_params`, `split_theta`, `initial_theta_from_spatial`, `sample_prior`, `log_prior`,
`log_likelihood` and `log_posterior` mean exactly what they do on `ForwardModel`, so the
samplers, `find_map`/`multistart_map` and the posterior-predictive helpers work unchanged; it
adds `model_images`, `component_images_per_band`, `band_profiles`, `chi_maps`, `chi2`, `n_data`
and `sky_to_pixel`/`pixel_to_sky`. The band loop is a Python loop over static indices (ragged
shapes cannot be vmapped), so it costs roughly 1.3× a single-grid fit on 9-band NIRCam; the
emulator is still evaluated exactly `K` times per likelihood call.

Two things to watch. The spatial model must be in arcsec mode *and*
`normalisation="analytic"`, and every band needs a PSF kernel on its own grid
(`psfs={band: (kernel, kernel_pixel_scale)}` when loading, or `PSFModel.resample`); its
`image_shape`/`pixel_scale` no longer describe any band and only set the scale of the default
centre and size priors, so pick the finest band's values. And this model works in the **sky
frame**, `mu_y` towards North and `mu_x` towards East, whereas a single-grid model in arcsec
mode uses `+y = +row`, `+x = +column` — with "North up, East left" `+column` points *West*, so
the two frames are mirror images in `x` and the sign of `rho` flips.

Use the plain `ForwardModel` whenever the bands really do share a grid — a DJA `thumb` cutout,
where the server has already resampled every filter onto one 0.05″/px grid
(`load_dja_cutout(...).to_observation_cube()`).

### Batched fitting

When a sample of cutouts shares the same bands and `(H, W)` — the usual outcome of a
fixed-size cutout service — every fit is the *same* XLA program with different data, so
`arachne.inference.batched` turns the sample into one vmapped program:

```python
from arachne import (BatchedForwardModel, batched_blind_initial_theta,
                     batched_multistart_map, batched_nuts, fit_batch_nss)

bfm = BatchedForwardModel.build(
    observations=[obs_0, obs_1, ...],          # identical bands and (H, W)
    psf_models=psf,                            # one shared PSFModel, or one per galaxy
    spatial_model=spatial_model, emulator=emulator, model_error_frac=0.05,
    fixed_params_per_galaxy={"redshift": z_spec},   # must already be a fixed param
    galaxy_ids=ids,
)
thetas0 = batched_blind_initial_theta(bfm)
maps = batched_multistart_map(bfm, thetas0, archetypes=[{}, {"Av": 2.0}])
result = batched_nuts(bfm, maps.theta, jax.random.PRNGKey(0), n_chains=4)
print(result.summary())
result.to_hdf5("batch_posterior.h5")
```

`bfm.forward_model(i)` extracts a single-galaxy `ForwardModel` view for plotting or diagnostics,
and per-galaxy redshifts go through `fixed_params_per_galaxy` (backed by
`AdditiveComponentModel.with_fixed_values`), not a tight prior. NSS cannot be batched — its
termination criterion is a per-problem Python `while` loop — so `fit_batch_nss` loops and each
galaxy pays its own XLA compile: batch the MAP stage and reserve NSS for the galaxies that
actually need an evidence.

### Diagnostics, model comparison and posterior predictive checks

```python
from arachne import residual_summary
from arachne.inference.model_comparison import compare_n_components, bayes_factor_table

print(result.summary())                  # step size, R-hat, ESS, divergences, tree depth
summary = residual_summary(forward_model, samples)
print(summary["chi2_red"], summary["chi2_red_per_band"], summary["frac_chi_gt_3"])
rows = compare_n_components(make_forward_model, ks=[1, 2, 3], rng_key=jax.random.PRNGKey(0))
print(bayes_factor_table(rows))          # logZ, logZ_err, chi2_red, Kass & Raftery verdict
```

- `diagnostics` — `split_rhat`, `ess` (Stan/ArviZ rank-normalised bulk ESS), `chain_movement`
  and `summarise_chains` on `(n_chains, n_samples, n_params)` arrays, JAX or numpy. Run ≥ 4
  chains and require `rhat < 1.01` and `ess > 100` per chain before trusting quantiles;
  `NUTSResult.summary()` prints them with explicit WARNING lines.
- `posterior_predictive` — `model_image_samples` (PSF-convolved, with sky and shifts),
  `component_image_samples` (unconvolved, per component), `residual_summary` (median model, chi
  map, `chi2_red`, per-band chi², `frac_chi_gt_3`), `predictive_bands` (16/50/84 envelopes) and
  `chi2_reduced(_samples)`, all chunked through `jax.vmap`. A `MultiResolutionForwardModel` is
  detected automatically and its products come back as per-band **lists** rather than stacked
  arrays (`is_multiresolution(fm)`).
- `model_comparison` — `compare_n_components(make_forward_model, ks, rng_key, sampler_kwargs=...,
  map_theta_fn=..., n_chi2_samples=64)` runs NSS once per `K`, returning
  `ModelComparisonRow(n_components, logZ, logZ_err, ess, n_params, chi2_red_map,
  chi2_red_median, runtime_s, result)`. **The caller owns prior comparability**:
  `make_forward_model(K)` must change nothing but `K`, or the Bayes factor measures the prior
  difference instead of the data's preference.

### Real data: JADES + DJA

`arachne.data` has clients for the two public products this pipeline was built against.

**Cutouts (DJA).** `fetch_dja_cutout(ra, dec, size_arcsec, filters, output="fits_weight",
cache_dir=...)` hits the grizli thumbnail service and caches a multi-extension FITS;
`load_dja_cutout` turns it into a `MultiResolutionObservation`. `size_arcsec` is the **full**
width (the service's own `size` is a half-width), so 6″ gives 120×120 at 0.05″/px — about 6 s
and 1.5 MB for nine filters. The thumbnail server resamples every filter onto that one grid, so
`.to_observation_cube()` is the right follow-up. For native resolution (SW 0.020″/px, LW
0.040″/px) `fetch_dja_native_cutout` downloads and caches the ~35 MB association sub-mosaics and
cuts them locally, returning a multi-resolution observation directly.

**Units caveats.** DJA pixels carry an authoritative `BUNIT = '10.0*nanoJansky'`
(`PHOTFNU = 1e-8`) *and* a `ZP` card that is **not** consistent with it (it is the original
mosaic's zeropoint from before the server rescaled the pixels, off by ≈ 2.29×); the loaders
prefer BUNIT and fall back to `ZP` only when no usable BUNIT is present. More importantly the
`WHT` maps are inverse variance **with no source Poisson term**, so a bright galaxy comes out
at S/N 10³–10⁴ per aperture: always fit with `model_error_frac ≈ 0.05–0.1`. Cutouts use
`Cutout2D(mode="partial", fill_value=0)`, so check the `mask`, not the `flux`, for coverage.

**Targets and redshifts (JADES).** `download_jades_dr4_specz()` / `load_jades_dr4_specz()` fetch
and standardise the JADES NIRSpec DR4 combined catalogue (5190 rows) into columns `id, ra, dec,
z_spec, z_flag, field, z_phot, nirspec_id, ...` (`z_flag` A/B highly robust, C secure, D
tentative); `select_targets(table, field=..., z_min=..., z_max=..., quality="best", n=...,
seed=...)` picks a sample. Empirical JADES PSFs (133×133) live outside this repo and their
headers carry **no pixel scale** — assume 0.03″/px and pass it explicitly, as
`PSFModel.resample(kernel, from_scale=0.03, to_scale=...)` or `psfs={band: (kernel, 0.03)}`.

```python
targets = select_targets(load_jades_dr4_specz(), field="GOODS-S", z_min=1.5, z_max=3.0, n=5)
ra, dec = targets["ra"][0], targets["dec"][0]
path = fetch_dja_cutout(ra, dec, size_arcsec=6.0, filters=["f115w", "f200w", "f444w"])
obs = load_dja_cutout(path, ref_ra=ra, ref_dec=dec).to_observation_cube()
```

[`examples/fit_jades_dja.py`](examples/fit_jades_dja.py) (with
[`examples/real_data_utils.py`](examples/real_data_utils.py)) is the worked end-to-end driver
*(CLI read from `--help` at 2026-09-18 15:10 BST; under active development)*: 9-band NIRCam DJA
cutouts, a JADES DR4 spec-z held fixed, empirical JADES DR3 PSFs, Sérsic (optionally
point-source) components with per-band sky nuisances, fitted blind with NSS or NUTS. The
catalogue photometry never enters the fit and is only used afterwards, in
`photometry_comparison.csv`.

```bash
python examples/fit_jades_dja.py --list-targets                       # the vetted GOODS-S sample
python examples/fit_jades_dja.py --target goods-s-mediumhst_12281 --quick
python examples/fit_jades_dja.py --target goods-s-mediumhst_12281 --k 1 2   # evidence over K
python examples/fit_jades_dja.py --ra 53.14104 --dec -27.76680 --z 1.8989
```

Useful flag groups: source isolation (`--mask-neighbours`, `--mask-clump-nsigma`,
`--mask-plume`, `--mask-radius`), noise (`--model-err-frac`, `--poisson-floor`,
`--fit-noise-scale`), geometry (`--profiles`, `--point`, `--n-bounds`, `--max-re-arcsec`),
`--multires` to fit the native DJA mosaics instead of the 0.05″/px thumbnails, and `--map-only`
for fast checks. Outputs land in `outputs/real_data/<target_id>/K<k>/`. It needs network access
and a GPU node.

### Resolved demo

[`examples/demo_resolved_sed_fitting.py`](examples/demo_resolved_sed_fitting.py) is the
end-to-end reference for the image-level pipeline *(CLI read from `--help` at 2026-09-18
15:10 BST; the script is under active development, so check `--help` before trusting a flag)*.
It renders a two-component mock — a compact, old, quenched Sérsic bulge plus an extended,
young, dusty Sérsic disk at a common fixed z = 2 — as a 10-band JWST/NIRCam image with the real
trained `ParrotEmulatorV2` checkpoint and real JADES empirical PSFs, contaminates it with
per-band sky pedestals, per-band sub-pixel shifts and Gaussian noise, and fits it back
**blind**: 34 spatial + 28 nuisance = **62 free parameters**, with the injected truth used only
to generate the data and grade the answer. F200W is the astrometric reference band, so the
fitted shifts are relative registrations.

The pipeline is `blind_initial_theta` → `multistart_map` → a MAP polish (annealed Adam with the
mass re-solve off, then modified-Newton steps) → `--sampler auto`, which means 4-chain NUTS in
whitened coordinates with a dense metric for the 62-parameter model and NSS (with `logZ`) for
the 34-parameter `--no-nuisance` model. It runs in float64 and needs a GPU node, the checkpoint
under `scripts/outputs/` and the JADES PSF directory on the cluster.

```bash
ssh mad06                                                     # a GPU node with fast fp64
python examples/demo_resolved_sed_fitting.py --quick          # smoke run, ~20 min on an A100
python examples/demo_resolved_sed_fitting.py                  # NUTS, 4 chains, ~1-3 h
python examples/demo_resolved_sed_fitting.py --no-nuisance    # 34 params, NSS + logZ
python examples/demo_resolved_sed_fitting.py --compare-k --quick
python examples/demo_resolved_sed_fitting.py --help           # the full, current flag list
```

Several flags exist to *reproduce* documented failure modes: `--no-whiten` (raw theta with an
identity metric: R-hat ~ 4000), `--diagonal-metric`, `--float32`, `--all-band-shifts` (the
exactly degenerate shift parameterisation). Outputs go to `--outdir` (default
`outputs/demo/resolved_sed_fitting/`): `truth.json`, `map.json`, `posterior.hdf5`,
`sampler_summary.txt`, `bayes_factors.txt` (with `--compare-k`) and `figures/` — per-band
truth/data/model and chi mosaics, SPS-parameter maps, bulge/disk SEDs, 1-D posteriors, sampler
diagnostics and component radial profiles. Read its module docstring for the assumptions and
known limitations.

## Samplers

| Sampler | Best for |
|---|---|
| `NSSSampler` (`blackjax.nss`) | Image models with a proper prior and **≲ 40 parameters**: gradient-free, live points drawn from `sample_prior` (no starting point), handles label/size/mass multimodality by construction, returns equal-weight samples plus `logZ`, `logZ_err`, ESS — the tool for comparing different `K`. Many more likelihood evaluations than NUTS; use a GPU. Checkpoints and resumes bit-exactly. |
| `NUTSSampler` / `fit_catalogue.py --sampler nuts` | GPU-batched catalogue fitting; image models above ~40 parameters, started from the polished blind MAP and sampled in **whitened** coordinates (see below) |
| `arachne.inference.laplace.run_whitened_nuts` | The practical way to run NUTS on a resolved fit: builds the Laplace covariance at the MAP, samples `theta = MAP + L z` with a dense metric adapted inside the whitened space, and returns a `NUTSResult` in `theta` coordinates |
| `batched_nuts` | Many equal-shape cutouts at once: vmapped warmup + sampling over (galaxy, chain), with per-galaxy diagnostics |
| `MCLMCSampler` | `FreeFormPixelMap` (high-d, ~45k–67k params); O(1) gradient evals per effective sample vs NUTS's O(d^{1/4}) |
| `fit_catalogue.py --sampler nss` (`blackjax.nss`) | Catalogue fits: multimodal posteriors, per-galaxy log-evidence (`logZ`); sequential, not GPU-batched |

`ForwardModel` exposes `log_likelihood`, `log_prior` and `log_posterior` separately; NSS uses
the first two, the gradient samplers the third. `run_pathfinder` provides a fast L-BFGS
warm-start (MAP position + diagonal inverse-mass-matrix estimate) that can be passed to
either gradient-based sampler to skip expensive warmup.

Two hard-won rules, measured on the 62-parameter demo in September 2026. **NSS from the prior
is impractical above ~40 parameters**: at 62 an outer step took over 3 s and the run was still
~1.5e6 nats below the MAP after 50 steps; below that threshold NSS is the best tool, because it
also gives `logZ`. And **NUTS needs a dense metric here**: the curvature eigenvalues at the mode
span 1e-1 to 4e7 and the stiff directions are correlated, so a diagonal mass matrix collapses
the warmup step size to ~1e-7, every trajectory saturates the tree-depth cap and R-hat reaches
the thousands. Sampling whitened coordinates `theta = MAP + L z`, with `L` the Cholesky factor
of the Laplace covariance at a properly polished MAP, lifts the step size by four orders of
magnitude, and a dense metric adapted *inside* the whitened space mops up the residual
mass-exchange ridge the Laplace approximation misses.
`arachne.inference.laplace` packages all of that — `hessian_neg_log_post`,
`laplace_covariance`, `WhitenedLogDensity`, `laplace_whitening` and the one-call
`run_whitened_nuts(fm, theta_map, key, n_chains=4, dense_mass_matrix=True)`, whose result comes
back in `theta` coordinates with R-hat and ESS recomputed there. Check them on every run.

## Gotchas

- **`pip install`ing anything that touches `jax`/`jaxlib` can silently break GPU support.**
  JAX requires the CUDA plugin (`jax-cuda12-plugin`, `jax-cuda12-pjrt`) to match `jaxlib`'s
  version *exactly*; if a resolver bumps `jaxlib` without bumping the plugin, JAX does not
  error — it silently falls back to CPU. After any such install, verify on an actual GPU node
  (`jax.devices()` on a login node proves nothing — there's no GPU there to detect):
  ```bash
  ssh gn004   # or any node from the GPU table in CLAUDE.md
  python -c "import jax; print(jax.devices())"   # must show CudaDevice, not CpuDevice
  ```
- **Never upgrade a shared/multi-project venv without checking who else depends on it.** A
  `blackjax`/`jax` upgrade drags `numpy` along and breaks any package with a tighter pin
  (astropy, numba, ...); `pip check`'s static comparison misses runtime breaks that declare no
  strict upper bound (astropy did not, but still broke on `numpy.in1d` removal), so actually
  `import` what you care about. Pin known-good versions in `pyproject.toml`.
- **Data S/N above the emulator's accuracy makes a resolved fit over-confident.** A resolved
  fit sums tens of thousands of pixels; with photon noise alone the posterior is far narrower
  than the emulator's few-per-cent systematics. Use `model_error_frac` (a `GaussianLikelihood`
  fractional floor added in quadrature, with the `log var` term) on every resolved fit — and
  note that DJA weight maps carry no source Poisson term at all.
- **Fitting a shift in every band is exactly degenerate** with moving every component. Pass
  `NuisanceModel(..., shift_reference_band=i)`.
- **Frame normalisation is not a surface brightness.** `normalisation="frame"` is defined only
  on the model's own grid and `model_image_on` will refuse anything else; build the model with
  `normalisation="analytic"` for multi-resolution work.
- **`float32` gradients are noisy near the mode.** The demo enables
  `jax.config.update("jax_enable_x64", True)` at import time, at no measurable cost on
  A100/H100 — but fp64 is slow or absent on consumer-class cards, so keep float64 jobs on the
  datacentre GPUs.

## License

GPLv3 — see [LICENSE.md](LICENSE.md).
