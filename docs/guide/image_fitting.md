# Image-level forward modelling with light components

The component pipeline: `AdditiveComponentModel`, PSF convolution, nuisances, blind starts, multi-resolution and batched fits. Per-pixel parameter maps are in [correlated_fields.md](correlated_fields.md). Part of the [arachne guide](README.md).

## Quick Start

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
    param_bounds=BOUNDS,                  # real emulator: emulator.spec.param_bounds
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

## Data Flow

```
synference HDF5 library ──▶ train_parrot_emulator_v2.py ──▶ emulator.eqx
                                     (mass-factorised MLP, Fourier-z, arsinh SFH inputs)
  (inference time)
theta (n_params,)  ← BlackJAX NSS / NUTS / MCLMC, NIFTy.re geoVI/MGVI
   │               (from blind_initial_theta → multistart_map, or a trust_region_newton mode)
   │  split_theta → (theta_spatial, theta_nuisance)
   ▼ SpatialModel.model_image(theta_spatial, emulator, (H, W))
   │    AdditiveComponentModel:  K SEDs (K, N_bands) ← emulator.predict,
   │                             × K normalised profiles (K, H, W) → Σ_k F_k ⊗ P_k
   │    CorrelatedFieldMap:      levels + globals + white Fourier coefficients → Matérn
   │                             field maps → softclip / copula → per-pixel emulator inputs
   │                             (mapped + derived) → predict per pixel
   │    FreeFormPixelMap / GMM:  decode → pixel_params (H*W, N_sps) → predict per pixel
   ▼ PSFConvolver (FFT; nuisance sub-pixel shifts as a phase ramp) + per-band nuisance sky
   ▼ GaussianLikelihood (model-error floor, nuisance log_noise_scale)
   │    + SpatialModel.log_prior() + NuisanceModel.log_prior()
log_likelihood(theta), log_prior(theta), log_posterior(theta)  ← scalar, differentiable
   ├─▶ NSS: log_prior + log_likelihood separately, live points from sample_prior → samples + logZ
   └─▶ NUTS / MCLMC: jax.grad(log_posterior) → leapfrog
```

## Spatial Models

| Model | Parameters | Best for |
|---|---|---|
| `AdditiveComponentModel` (**recommended**) | `Σ_k (n_shape_k + N_free) + N_shared` | Bulge/disk and other multi-component decompositions. Light is additive: each component has its own SED at its own total mass times a normalised profile; `K` emulator calls per likelihood, independent of image size. Proper prior with `sample_prior` (works with NSS); supports `fixed_params`, `shared_param_names`, mixed profiles, arcsec coordinates and analytic normalisation. |
| `CorrelatedFieldMap` (**recommended for resolved maps**) | `N_fields + N_global + N_hyper + N_coupling + Σ_k n_modes_k` | Per-pixel maps of the SPS parameters (stellar mass per pixel, A_V, Z, SFH) as Gaussian random fields around mean maps: Matérn priors on truncated white Fourier coefficients (a few hundred to a few thousand modes per map instead of `H × W`), free level offsets, optional fitted hyperparameters and couplings between fields, copula marginals, latent fields with derived emulator inputs. Proper prior with `sample_prior`; one emulator call per pixel. See [Correlated-field parameter maps](correlated_fields.md). |
| `GaussianMixtureSpatialModel` | `K × (5 + N_sps)` | **Caveat:** blends SPS *parameters* per pixel with softmax weights that sum to 1, so it has no surface-brightness profile — every pixel carries a full galaxy mass and a lone "disk" fills the whole frame at constant flux. Superseded by `CorrelatedFieldMap` for smooth parameter gradients; kept for compatibility only, never for bulge/disk work. |
| `FreeFormPixelMap` | `H × W × N_sps` | Maximum flexibility; per-pixel emulator calls, requires GPU; L2 smoothness prior. |

All models implement `decode` / `log_prior`; `model_image` (the hook `ForwardModel` calls) has
a per-pixel default and is overridden by `AdditiveComponentModel` and `CorrelatedFieldMap`. Note that
`AdditiveComponentModel.decode` returns a *summary* map (mass-weighted component parameters,
with log10 stellar-mass surface density in the mass column) for plotting — the likelihood
never uses it.

## Light profiles

Each component's shape is delegated to a `Profile`
([`src/arachne/spatial/profiles.py`](../../src/arachne/spatial/profiles.py)), selected per component
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

## Initialisation

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

## Priors

Per-parameter physical-space priors are dict specs (`{"dist": "studentt", "df": 2, "loc": 0,
"scale": 0.3}`) handled by [`arachne.priors.specs`](../../src/arachne/priors/specs.py):
`DEFAULT_PRIORS` (Student-t on `logsfr_ratio_*` — the v6 emulator training grid was drawn from
it — uniform elsewhere; `default_prior(name)` extends this to every known library parameter, and
`resolve_prior_specs(..., defaults=emulator.spec.priors)` lets a checkpoint supply its own), `resolve_prior_specs` (merge user overrides, validate against bounds),
`build_log_prior` (catalogue fits, `(N,)` vector), `build_component_log_prior` (image fits,
`(K, N_emulator)` matrix; shared parameters counted once, fixed ones ignored),
`prior_config_template` and `sigmoid_log_jacobian`. Supported `dist` values: `uniform`,
`loguniform`, `normal`, `studentt`, `halfnormal`, `exponential`, `lognormal`. The spatial models
add their own sigmoid Jacobian (uniform in *physical* space) plus each profile's shape prior
(Gaussians on the centre, log sizes, `atanh_rho` and `log_n`), so
`AdditiveComponentModel.log_prior` is a proper density, `logZ` values are comparable across `K`,
and `sample_prior` draws from exactly that density (minus the optional `sps_log_prior` term) for
NSS's live points. `CorrelatedFieldMap` takes the same specs through `global_priors` (a density
on a free global instead of uniform-in-bounds) and `field_marginals` (the per-pixel marginal of a
mapped field, applied as a copula so the spatial correlation and the `N(0, I)` prior on theta are
kept).

## Instrumental nuisances

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

## Multi-resolution fitting (optional)

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

## Batched fitting

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

## Diagnostics, model comparison and posterior predictive checks

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
