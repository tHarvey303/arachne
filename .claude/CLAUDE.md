# CLAUDE.md

Guidance for Claude Code when working in this repository. The HPC/venv/GPU-node setup lives in
the parent `/cosma7/data/dp276/dc-harv3/CLAUDE.md`; this file is about arachne itself.

## Project Overview

**arachne** is a pure-JAX/Equinox package for galaxy SED fitting with neural SPS emulators:

1. **Catalogue / single-galaxy fitting** (`scripts/fit_catalogue.py` and friends) — integrated
   photometry, GPU-batched Pathfinder+NUTS or nested slice sampling (NSS, gives `logZ`).
2. **Image-level forward modelling** — a galaxy as `K` additive light components
   (`AdditiveComponentModel`: Gaussian / Sérsic / point profiles) or a free-form pixel map,
   PSF-convolved per band, optionally with instrumental nuisances, on one grid
   (`ForwardModel`), on every band's native WCS grid (`MultiResolutionForwardModel`) or
   vmapped over a sample of cutouts (`BatchedForwardModel`); fitted blind with NSS / NUTS / MCLMC.

Emulators are trained from synference HDF5 libraries (`ParrotEmulatorV2` is the production
one). Everything downstream of the checkpoint is JAX: differentiable, jit-able, no PyTorch.

## Commands

```bash
source /cosma/apps/dp276/dc-harv3/venv_simformer/bin/activate   # shared venv: never pip install/upgrade
JAX_PLATFORMS=cpu python -m pytest tests -q -p no:cacheprovider  # full suite on CPU, ~9 min
JAX_PLATFORMS=cpu pytest tests/test_additive_model.py tests/test_multires_forward_model.py
pytest -m "not gpu"                       # skip GPU-marked tests
ruff check --fix src/ scripts/ tests/ && ruff format src/ scripts/ tests/
cd docs && make html      # needs the `docs` extra; sphinx is NOT in the shared venv

# Emulator training / validation
python scripts/train_parrot_emulator_v2.py --library <lib.hdf5> --output outputs/emulators/parrot_emulator_v2.eqx
python scripts/validate_parrot_emulator.py --emulator <ckpt.eqx> --library <lib.hdf5>

# Resolved mock demo / real-data fit (GPU node, see below)
python examples/demo_resolved_sed_fitting.py --quick
python examples/fit_jades_dja.py --list-targets
```

Test suite measured on the login node on 2026-09-18 15:21 BST:
**816 passed, 1 skipped, 2 warnings in 532.72 s (8 min 53 s)** with
`JAX_PLATFORMS=cpu python -m pytest tests -q -p no:cacheprovider`. Zero failures — the old
"known pre-existing `ParrotEmulator` metadata failures" paragraph no longer applies. (Wall time
swings with login-node load; an earlier identical run of 764 tests took 8 min 17 s.)

## Module map (`src/arachne/`)

| Module | Responsibility |
|---|---|
| `data/observation.py`, `data/psf.py` | `ObservationCube` (flux/variance/mask `(N_bands,H,W)` nJy, `from_fits`, `to_jax()`), `PSFModel` (per-band kernels, `from_fits`, static `resample(kernel, from_scale, to_scale)` flux-conserving, `resample_to`, `pad_to_image_size`) |
| `data/units.py` | `parse_bunit`, `flux_scale_to_nJy`, `flux_to_nJy`, `variance_to_nJy2`, `weight_to_variance`, `zeropoint_scale_to_nJy`. Everything internal is nJy / nJy². AB zp 31.4 ⇔ 1 nJy |
| `data/multires.py` | `BandImage` (one band on its own grid: flux/variance/mask, `pixel_scale`, `affine`, `ref_pixel`, `psf`, `sky_coords()`), `MultiResolutionObservation` (`from_fits`, `from_dja_fits`, `to_observation_cube`, `to_jax`), `canonical_band_name`, `tangent_plane_affine` |
| `data/dja.py` | DAWN JWST Archive client: `fetch_dja_cutout` / `load_dja_cutout` (thumb service), `fetch_dja_native_cutout` (assoc_mosaic), `dja_filter_names`, `server_reachable` |
| `data/jades.py` | JADES NIRSpec DR4 spec-z catalogue: `download_jades_dr4_specz`, `load_jades_dr4_specz`, `select_targets` |
| `emulator/parrot_emulator_v2.py` | `ParrotEmulatorV2` + `load_emulator` — **primary emulator**: analytic mass factorisation (flux exactly ∝ 10**log_mass), Fourier-z features, arsinh SFH-ratio inputs |
| `emulator/parrot_emulator.py`, `jax_mlp_emulator.py`, `jax_emulator.py` | `ParrotEmulator` (v1, legacy), `SPSMLPEmulator` (Speculator), `JAXFlowEmulator` (legacy flow export) |
| `emulator/fixed_param_wrapper.py` | `FixedParamEmulator` — pins one input; superseded for spatial work by `AdditiveComponentModel(fixed_params=...)` |
| `spatial/base.py` | `SpatialModel` ABC: `n_params`, `decode`, `log_prior`; optional `model_image` hook and `sample_prior` |
| `spatial/profiles.py` | `Profile` ABC + `GaussianProfile` (5), `SersicProfile` (6), `PointSourceProfile` (2); `PROFILES`, `get_profile`, `sersic_b`, `render_on_grid` / `log_render_on_grid` (analytic flux per pixel on **any** coordinates) |
| `spatial/additive.py` | `AdditiveComponentModel` — **recommended** for multi-component work (see below) |
| `spatial/gmm.py` | `GaussianMixtureSpatialModel` — **caveat**: blends SPS *parameters* per pixel with unit-sum softmax weights; no surface-brightness profile, every pixel carries a full galaxy mass. Only for smooth parameter gradients; never for bulge/disk |
| `spatial/pixel_map.py` | `FreeFormPixelMap` — per-pixel params, L2 smoothness prior, high-d |
| `psf/convolution.py` | `PSFConvolver(psf_model, image_shape, pad=True)` / `.from_kernels(...)`; `__call__(image (N,H,W), shifts (N,2)\|None)` — FFT convolution, exact Fourier phase ramp for sub-pixel shifts (+dy down, +dx right); `pad=True` = linear (zero-padded to an even 5-smooth size); `with_psf_ffts` for vmapping |
| `likelihood/gaussian.py` | `GaussianLikelihood(obs, model_error_frac)` / `.from_arrays(...)`; `__call__(model_image, log_noise_scale=None)`; `with_observation`. Per-band likelihoods add **exactly** to the joint one |
| `forward_model/nuisance.py` | `NuisanceModel` — per-band sky / (dy,dx) shifts / log noise scale appended to theta |
| `forward_model/pipeline.py` | `ForwardModel` (single shared grid) — `build`, `log_likelihood`, `log_prior`, `log_posterior` (all pure) |
| `forward_model/multires.py` | `MultiResolutionForwardModel` — every band on its own WCS grid, no data resampling; sky frame (+y North, +x East) |
| `inference/initialisation.py` | `image_moments`, `reference_band_index`, `blind_initial_theta`, `blind_initial_full_theta`, `solve_component_masses` (nuisance-aware), `find_map`, `multistart_map`, `MAPResult` |
| `inference/nss_sampler.py` | `NSSSampler` / `NSSResult` — `blackjax.nss` over `log_prior` + `log_likelihood`, live points from `sample_prior`; equal-weight samples, `logZ`, `logZ_err`, `ess`; bit-exact checkpoint/resume |
| `inference/nuts_sampler.py` | `NUTSSampler` / `NUTSResult` (`get_parameter_map`, `to_hdf5`/`from_hdf5`, `summary`); per-chain vmapped window adaptation, `inverse_mass_matrix=` |
| `inference/laplace.py` | `make_hessian_fn`, `hessian_neg_log_post`, `laplace_covariance`, `WhitenedLogDensity`, `laplace_whitening`, `run_whitened_nuts` — Laplace covariance at the MAP + whitened/dense-metric NUTS. NOT exported from `arachne/__init__.py`: import from `arachne.inference.laplace` |
| `inference/mclmc_sampler.py` | `MCLMCSampler` (blackjax 1.6.2 API) + `run_pathfinder`; for `FreeFormPixelMap` |
| `inference/diagnostics.py` | `split_rhat`, `ess`, `chain_movement`, `summarise_chains` on `(n_chains, n_samples, n_params)` |
| `inference/model_comparison.py` | `compare_n_components(make_forward_model, ks, key, ...) -> [ModelComparisonRow]`, `bayes_factor_table` |
| `inference/posterior_predictive.py` | `model_image_samples`, `component_image_samples`, `residual_summary`, `predictive_bands`, `chi2_reduced(_samples)`, `is_multiresolution`, `n_model_params` |
| `inference/batched.py` | `BatchedForwardModel` (vmapped equal-shape cutouts) + `batched_blind_initial_theta`, `batched_find_map`, `batched_multistart_map`, `batched_nuts`, `fit_batch_nss`, `BatchedMAPResult`, `BatchedNUTSResult` |

`scripts/`: `fit_catalogue.py` (shared catalogue driver; `SPS_PARAM_NAMES` / `PARAM_BOUNDS` are
the emulator training domain — redshift, log_mass, slope, fesc_lya, dust_bump_amplitude,
log10metallicity, Av, logsfr_ratio_0..4), `fit_mock_jwst.py` / `generate_mock_jwst.py`,
`fit_bulge_disk_fixed_z.py`, training/validation scripts, `experiments/` (emulator HPO harness;
`archive_gmm_blend/` = retired diagnostics from the old parameter-blending demo, do not resurrect).

## The additive model in one screen

```
I_b(y,x) = Σ_k F_kb(θ_k) · P_k(y,x)      F_k = emulator SED at component k's OWN total mass
                                          P_k = normalised surface-brightness profile
theta = concat([block_0 ... block_{K-1}, shared_raw])
block_k = [shape_k (n_shape_k,), sps_raw_k (N_free,)]
n_params = Σ_k (n_shape_k + N_free) + N_shared          (K*(n_shape+N_free)+N_shared if uniform)
phys = lo + (hi-lo)*sigmoid(raw);  sigma = exp(log_sigma);  rho = tanh(atanh_rho) clipped ±0.99
```

- **Profiles** (`profiles=` one name/instance for all, or a list of K):
  `"gaussian"` 5 shape params `[mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho]`;
  `"sersic"` 6 (`+ log_n`; `sigma` = effective radius per axis, `n = exp(log_n)` softly clamped
  into `n_bounds=(0.3,10)`, prior `Normal(log 2, 0.7)`); `"point"` 2 (`mu` only; a fixed circular
  Gaussian of width `point_sigma`, 0.5 px by default — after PSF convolution it *is* the PSF).
  Blocks then have different lengths, so `split_theta` returns a list rather than a `(K, ·)` array.
- **Coordinates**: `pixel_scale=None` → pixel indices. `pixel_scale=<arcsec/px>` → arcsec offsets
  from the frame centre `((H-1)/2, (W-1)/2)`; mu/sigma in arcsec; default log-size prior mean
  `log(max(H,W)*pixel_scale/16)`, centre prior `(0,0)` with sigma frame/4. A `"point"` asked for
  by name gets `point_sigma = 0.5 * pixel_scale`.
- **Normalisation**: `"frame"` (default, historical) divides by the sum over the model's own grid;
  `"analytic"` normalises the integral over the whole plane (flux outside the cutout is lost) and
  is **required** to render on foreign grids (`model_image_on`, multires). `oversample=3..5` for
  cuspy Sérsic components (1 is fine for Gaussians).
- **Roles**: every emulator parameter is **fixed** (`fixed_params`), **shared**
  (`shared_param_names`, one value for all components, e.g. redshift) or **per-component**
  (default; `mass_param` must be). Per-component free names = `sps_param_names`.
- **Prior** = sigmoid Jacobian (normalised uniform in physical space) + optional `sps_log_prior`
  on the `(K, N_emulator)` physical matrix (`build_component_log_prior`) + each profile's own
  shape prior (Gaussians on centre / log-size / atanh_rho / log_n). `sample_prior` draws from it
  without the `sps_log_prior` term (NSS needs this).
- **Helpers**: `component_slices`, `shared_slice`, `n_shape_params`, `shape_raw(theta,k)`,
  `sps_raw(theta,k)`, `set_sps_raw`, `split_theta`/`join_theta`, `component_shapes` (per-profile
  dicts incl. `n`, `b_n`), `component_params -> (mu (K,2), sigma (K,2), rho (K,), sps_phys
  (K,N_emulator))`, `component_seds`, `profiles(theta) -> (K,H,W)`, `component_images`,
  `model_image`, `model_image_on(theta, emulator, yy, xx, pixel_area, oversample=None)`,
  `coords`, `pixel_area`, `centre`, `order_components_by_size` (permutes **only within
  same-profile groups**; passes a nuisance tail through), `fixed_param_names` / `fixed_values` /
  `with_fixed_values(values)` (per-galaxy z inside a vmap), `decode` (summary map only).
- Costs `K` emulator evaluations per likelihood, independent of image size.

## Nuisance parameters

`NuisanceModel(n_bands, fit_sky=True, fit_shifts=False, fit_noise_scale=False,
sky_prior_sigma=1.0 nJy, shift_prior_sigma=0.5 px, noise_scale_prior_sigma=0.3,
shift_reference_band=None)`. Vector order: `sky (N)`, `shifts [dy,dx] per free band`,
`log_noise_scale (N)`; `n_params`, `param_names(band_names)`, `split(theta_n)` (zeros for
disabled blocks; `shifts` always `(N,2)`), `log_prior`, `initial_theta()`, `sample_prior`.

**Always pass `shift_reference_band=i` when fitting shifts** — a common shift of all bands is
exactly degenerate with moving every component, so `n_shift = 2*(n_bands-1)`.

`theta = concat(theta_spatial, theta_nuisance)`; `fm.split_theta`,
`fm.initial_theta_from_spatial(theta_spatial)`, `fm.sample_prior(key, n)` handle the join.
`solve_component_masses` is nuisance-aware (subtracts the fitted sky, shifts the templates,
scales the weights).

## Standard image-fit workflow

1. Bounds = emulator training domain (`scripts/fit_catalogue.py: PARAM_BOUNDS`).
2. `specs = resolve_prior_specs(free_names, None, bounds)`;
   `sps_log_prior = build_component_log_prior(names, specs, bounds, shared_param_names=..., fixed_param_names=...)`.
3. `AdditiveComponentModel(K, names, bounds, (H, W), fixed_params={"redshift": z}, mass_param="log_mass", sps_log_prior=..., profiles=..., pixel_scale=..., normalisation=...)`.
4. `nuisance = NuisanceModel(n_bands, fit_sky=True, fit_shifts=True, shift_reference_band=0)` (optional).
5. `fm = ForwardModel.build(obs=obs, psf_model=psf, spatial_model=model, emulator=emulator,
   model_error_frac=0.05, nuisance=nuisance)` (`obs_jax = fm.observation`).
6. `theta0 = blind_initial_full_theta(fm)` (or `blind_initial_theta(model, obs_jax)` +
   `fm.initial_theta_from_spatial`); `map = multistart_map(fm, theta0, [{}, {"Av": 0.3}, {"Av": 2.0}])`.
7. Sample (see the decision rule below), then
   `jax.vmap(model.order_components_by_size)(result.samples)` before any summary.
8. Check it: `residual_summary(fm, samples)`, `result.summary()`, `compare_n_components(...)`.

Variants:
- **Multi-resolution**: build the `AdditiveComponentModel` in **arcsec + analytic** mode and use
  `MultiResolutionForwardModel.build(mro, spatial_model, emulator, model_error_frac=...,
  nuisance=None, pad_psf=True, oversample=None)`. Every band needs a PSF on its own grid.
  `model_images`, `component_images_per_band`, `chi_maps`, `chi2`, `band_profiles`,
  `sky_to_pixel`/`pixel_to_sky`, `n_data`; `log_prior/log_likelihood/log_posterior`, `n_params`,
  `split_theta`, `initial_theta_from_spatial`, `sample_prior` mirror `ForwardModel`, so the
  samplers work unchanged. `blind_initial_theta(model, mro_jax, ref_band=...)` accepts a
  `MultiResolutionObservation`; `posterior_predictive.is_multiresolution(fm)` dispatches the
  products (per-band **lists** instead of stacked arrays). Cost ≈ 1.3× a single grid on 9-band
  NIRCam. **Sky frame: `mu_y` = +North, `mu_x` = +East** — mirrored in x relative to the
  single-grid arcsec frame (+column is usually West), and `rho`'s sign flips with the handedness.
- **Batched**: `BatchedForwardModel.build(observations, psf_models, spatial_model, emulator,
  ..., fixed_params_per_galaxy={"redshift": z_spec})` for N equal-shape cutouts, then
  `batched_blind_initial_theta` → `batched_multistart_map` → `batched_nuts`. `fit_batch_nss` is
  a sequential loop (NSS terminates on a per-problem Python `while`, so it cannot be batched).

## Sampler decision rule

| Situation | Use |
|---|---|
| ≲ 40 parameters, evidence wanted | `NSSSampler` (from the prior; gives `logZ`, handles multimodality) |
| > 40 parameters | `laplace.run_whitened_nuts(fm, theta_map, key, n_chains=4, dense_mass_matrix=True)` from the polished MAP |
| `FreeFormPixelMap` (10⁴–10⁵ params) | `MCLMCSampler` (optionally warm-started by `run_pathfinder`) |
| Many equal-shape cutouts | `batched_nuts` (one compiled program for all galaxies) |
| Comparing K | `compare_n_components` (NSS per K; `make_forward_model(K)` must change *nothing* but K) |

NSS from the prior is impractical above ~40 parameters: a 62-parameter demo took >3 s per outer
step and was still ~1.5e6 nats below the MAP after 50 steps.

## Lessons / design rules

- **Light is additive; SPS parameters are not.** Never average SPS parameters across components
  or pixels to build a model image. Each component gets its own emulator call at its own total
  mass. (This is the whole reason `AdditiveComponentModel` replaced the GMM blend in the demo.)
- **Always include the sigmoid Jacobian** when sampling bounded parameters in unconstrained
  space; otherwise the "uniform" prior is uniform in raw space and `logZ` is meaningless.
  `AdditiveComponentModel` drops the `log(hi-lo)` constant so its prior is properly normalised.
- **Use the library's priors** (`arachne.priors.specs`): the emulator training grid drew
  `logsfr_ratio_*` from Student-t(df=2, scale=0.3); a flat prior there is wrong and pushes the
  emulator out of domain.
- **Never seed a recovery demo at the truth.** Blind init works because masses are linear
  amplitudes (`solve_component_masses`); start from image moments + neutral values + archetypes.
- **Use a fractional model-error floor** (`model_error_frac` ≈ 0.05–0.1) whenever data S/N
  exceeds emulator accuracy (always, for resolved fits); keep the `log var_eff` term or the
  floor is exploitable.
- **NSS needs a proper prior** (`sample_prior`) and prior-distributed live points; passing MAP
  points as live points invalidates `logZ`.
- **Break label symmetry after sampling** (`order_components_by_size`), not with the prior.
- **NUTS with a diagonal metric fails on these posteriors** (2026-09-17): curvature eigenvalues
  at the mode span 1e-1..4e7 and the stiff directions are correlated, so warmup collapses the
  step size to ~1e-7, every trajectory saturates the tree cap and R-hat reaches the thousands.
  Sample **whitened** coordinates `theta = MAP + L z` (`L` = Cholesky of the Laplace covariance
  at the polished MAP) — `arachne.inference.laplace.run_whitened_nuts`, which also adapts a
  **dense** metric inside the whitened space to absorb the residual mass-exchange / mass–dust
  ridge the Laplace approximation misses. Two details in that module matter: negative curvature
  directions are floored by `|lambda|`, not by a max-variance clip (that alone took the whitened
  condition number from 4.4e4 to 350), and the Hessian is always eigen-decomposed in numpy
  float64. A 62-parameter quick run with plain whitening + a *diagonal* metric still gave max
  R-hat 4.5–7, min ESS 5 and saturated trajectories, so check R-hat/ESS on every run.
- **`find_map` re-solves masses linearly between Adam rounds**, which near the mode maximises
  the *floor-free* likelihood and can RAISE `-log_post`. Polish with the mass re-solve **off**
  (annealed Adam) and then a few damped Newton steps — the demo's `polish_map` does this and the
  blind MAP then beats the truth by ~6 nats on the mock.
- **Float64** (`jax.config.update("jax_enable_x64", True)`) removes gradient noise in the demo at
  no measurable cost on A100/H100. The RTX 6000 Pro (gn005) and the V100s have slow/absent fast
  fp64 — send float64 jobs to mad06/gn004 only.
- `log_posterior` must stay pure (no I/O, no Python branching on traced values); no numpy/scipy
  in the hot path.
- Create a new sampler object per model/image shape (compiled graphs are shape-specific).

## Data layer facts worth remembering

- **Units**: everything internal is nJy (variances nJy²). AB zeropoint 31.4 ⇔ 1 nJy.
  `parse_bunit` returns `(scale, kind)`; `kind="surface_brightness"` (e.g. `MJy/sr`) still needs
  multiplying by the pixel solid angle.
- **DJA thumbnails** (`fetch_dja_cutout`, grizli-cutout `/thumb`): every filter is resampled onto
  one 0.05"/px grid, so use `ObservationCube` (`load_dja_cutout(...).to_observation_cube()`).
  `size_arcsec` is the **full** width (the server's `size` is a half-width): 6" → 120×120;
  ~6 s / 1.5 MB for 9 bands. `BUNIT='10.0*nanoJansky'` (PHOTFNU 1e-8) is **authoritative**; the
  `ZP` card is the *original mosaic's* zeropoint and is inconsistent (2.29× off) — the layer
  already prefers BUNIT. WHT extensions are inverse variance and contain **no source Poisson
  term**, so bright galaxies come out at S/N 1e3–1e4 per aperture: always use
  `model_error_frac ≈ 0.05–0.1`.
- **DJA native mosaics** (`fetch_dja_native_cutout`, `assoc_mosaic`): SW 0.020"/px, LW 0.040"/px,
  ~35 MB gz per file, cached under `arachne_data/assoc_mosaic/`; feeds
  `MultiResolutionForwardModel` directly.
- Cutouts use `Cutout2D(mode="partial", fill_value=0)` — check `mask`, **not** `flux`, for coverage.
- **JADES**: `download_jades_dr4_specz` / `load_jades_dr4_specz` (`Combined_DR4_external_v1.2.1.fits`,
  ext 1 `Obs_info`, 5190 rows) → columns `id, ra, dec, z_spec, z_flag (A/B robust, C secure,
  D tentative, E none), field, nirspec_id, z_phot, ...`; `select_targets(table, field="GOODS-S",
  z_min, z_max, quality="best"|"secure"|..., n, seed)`.
- **Empirical JADES PSFs**: `/cosma/apps/dp276/dc-harv3/synference/priv/JADES-DR3-GS/{FILTER}_psf_norm.fits`
  — 133×133, float64, sum 0.98–1.00, and the header carries **no pixel scale**: assume
  **0.03"/px** and pass it explicitly to `PSFModel.resample` / `psfs={band: (kernel, 0.03)}`.
- Production emulator: `scripts/outputs/emulators/parrot_emulator_v2.eqx` (54 bands, 12 params)
  via `arachne.load_emulator`. Data cache: `/cosma7/data/dp276/dc-harv3/work/arachne_data/`
  (JADES DR4 catalogue, DJA cutouts, `target_scan/final.json` aperture fluxes for the vetted
  GOODS-S targets). Local JADES DR3 GS photometry with total fluxes:
  `/cosma7/data/dp276/dc-harv3/work/catalogs/JADES_DR3_GS_Matched_Specz_total_flux_good.fits`.

## Examples

*(module docstrings and `--help` read at **2026-09-18 15:10 BST**; both scripts are being
actively edited by other streams, so re-read `--help` before quoting a flag)*

`examples/demo_resolved_sed_fitting.py` — the blind-recovery reference. Two **Sérsic**
components (compact quenched bulge + extended dusty disk, common fixed z=2) rendered as a
10-band NIRCam mock with the real `ParrotEmulatorV2` checkpoint and real JADES PSFs (hard-coded
cosma paths, 0.03"/px, 4.8" cutout), contaminated with per-band sky (~N(0, 0.3 nJy/px)) and
sub-pixel shifts (~N(0, 0.3 px)) plus Gaussian noise, and fitted **blind**. Arcsec + analytic
mode, `oversample=3`, `NuisanceModel(fit_sky=True, fit_shifts=True,
shift_reference_band=F200W)` → `block_k = [shape (6), sps_raw (11)]`, theta = 34 spatial + 28
nuisance = **62 parameters**. Float64 by default. Pipeline: `blind_initial_theta` →
`multistart_map` → `polish_map` (annealed Adam with the mass re-solve OFF, then modified-Newton)
→ `--sampler auto` = whitened 4-chain NUTS with a **dense** metric at 62 params, NSS (logZ) for
the 34-param `--no-nuisance` model.
CLI: `--outdir --seed --quick --profiles {sersic,gaussian} --oversample --no-nuisance
--all-band-shifts --noise-scale --model-err-frac --sampler {auto,nss,nuts} --num-live
--num-inner-steps --termination --no-resume --n-chains --n-warmup --n-samples --max-doublings
--chain-jitter --float32 --diagonal-metric --no-whiten --n-newton --compare-k --compare-ks`.
The last few exist to *reproduce* documented failures (`--no-whiten` → R-hat ~ 4000,
`--all-band-shifts` → the degenerate shift parameterisation). Runtimes on a shared A100:
blind MAP ~5 min, `--quick` ~20 min, default ~1–3 h. Outputs in
`outputs/demo/resolved_sed_fitting/`: `truth.json`, `map.json`, `posterior.hdf5`,
`sampler_summary.txt`, `bayes_factors.txt`, `figures/*.png`.

`examples/fit_jades_dja.py` (+ `examples/real_data_utils.py`) — the real-data driver: 9-band DJA
thumbnail cutouts, a JADES DR4 spec-z held **fixed**, empirical JADES DR3 PSFs, Sérsic (and
optional point-source) components with per-band sky nuisances, fitted blind with NSS or NUTS.
The JADES DR3 catalogue photometry never enters the fit and is only compared afterwards
(`photometry_comparison.csv`). `--list-targets` prints the vetted GOODS-S sample; `--target`,
`--ra/--dec/--z` or `--all-targets` select one; `--k 1 2` runs an evidence comparison.
Flag groups: isolation (`--mask-neighbours --mask-clump-nsigma --mask-plume --mask-radius`),
noise (`--model-err-frac --poisson-floor --fit-noise-scale --sky-prior-sigma`), geometry
(`--profiles --point --n-bounds --size-prior-sd --max-re-arcsec --share-dust`), sampling
(`--sampler --num-live --inner-steps-factor --n-chains --whiten/--no-whiten --polish
--newton-steps --map-only --quick --float64`) and `--multires` (fit the native DJA mosaics,
~70 MB per band to fetch). Outputs in `outputs/real_data/<target_id>/K<k>/`.

Run both on a GPU node inside tmux (mad06/gn004 for float64):

```bash
ssh mad06 && tmux new -s demo
source /cosma/apps/dp276/dc-harv3/venv_simformer/bin/activate
cd /cosma/apps/dp276/dc-harv3/arachne && python examples/demo_resolved_sed_fitting.py --quick
```

First run pays XLA compile time (NSS: minutes).

## Conventions

- `ruff` (line-length 100, Google docstrings; `__init__.py` and `examples/` docstrings exempt).
- FITS in, HDF5 out; JAX arrays inside the forward model, numpy only at I/O boundaries.
- Units: fluxes nJy, log10 masses in Msun (component **totals** in the additive model),
  parameter names follow the synference/emulator conventions (`log_mass`, `Av`, `logsfr_ratio_*`, ...).
- Tests: CPU, 3 bands × 16×16, tiny dummy `SPSEmulator` subclasses (see `tests/conftest.py` and
  `LinearMassEmulator` in `tests/test_additive_model.py`); GPU-only tests marked `@pytest.mark.gpu`.
- Emulator checkpoints and run outputs live in `outputs/` / `scripts/outputs/` (gitignored).
- Docs are Sphinx (`docs/source/`), but **sphinx is not installed in the shared venv** and we must
  never `pip install` into it — validate RST by reading, and build elsewhere.
