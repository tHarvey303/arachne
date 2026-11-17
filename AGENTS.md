# AGENTS.md

Working notes for AI coding agents (and for people who want the detail). The short
introduction is [README.md](README.md); topic guides are in [docs/guide/](docs/guide/README.md);
design notes with the measurements behind each choice are in [docs/design/](docs/design/).
`.claude/CLAUDE.md` imports this file, so edit this one.

## Project overview

**arachne** is a pure-JAX/Equinox package for galaxy SED fitting with neural SPS emulators:

1. **Catalogue / single-galaxy fitting** (`scripts/fit_catalogue.py` and friends): integrated
   photometry, GPU-batched Pathfinder+NUTS or nested slice sampling (NSS, gives `logZ`).
2. **Image-level forward modelling**: a galaxy as `K` additive light components
   (`AdditiveComponentModel`: Gaussian / Sérsic / point profiles), as per-pixel parameter maps
   with Gaussian-random-field priors (`CorrelatedFieldMap`, 10³–10⁴ parameters) or as a
   free-form pixel map; PSF-convolved per band, optionally with instrumental nuisances, on one
   grid (`ForwardModel`), on every band's native WCS grid (`MultiResolutionForwardModel`) or
   vmapped over a sample of cutouts (`BatchedForwardModel`); fitted blind with NSS, whitened
   NUTS or geoVI/MGVI.

Emulators are trained from synference HDF5 libraries. Everything downstream of the checkpoint
is JAX: differentiable, jit-able, no PyTorch.

## Which document to read

| Working on | Read first |
|---|---|
| Loading, swapping or training an emulator; anything that names SPS parameters | [docs/guide/emulators.md](docs/guide/emulators.md) and "Emulators" below |
| `CorrelatedFieldMap`, `examples/demo_correlated_field.py`, `fit_correlated_field_dja.py` | [docs/guide/correlated_fields.md](docs/guide/correlated_fields.md), [docs/design/correlated_field_inference.md](docs/design/correlated_field_inference.md) |
| `AdditiveComponentModel`, nuisances, multi-resolution, batched fits | [docs/guide/image_fitting.md](docs/guide/image_fitting.md) and the sections below |
| Choosing or debugging a sampler | [docs/guide/samplers.md](docs/guide/samplers.md) |
| `arachne.results`, `arachne.viz`, `arachne.gui` | [docs/guide/results.md](docs/guide/results.md), [docs/design/results_bundle_gui_spec.md](docs/design/results_bundle_gui_spec.md) |
| DJA / JADES data, real-galaxy drivers | [docs/guide/real_data.md](docs/guide/real_data.md), "Data layer facts" below |
| Catalogue scripts | [docs/guide/catalogue_fitting.md](docs/guide/catalogue_fitting.md) |

## Commands

```bash
# Tests (CPU; the full suite takes ~8-9 min).  Always set JAX_PLATFORMS=cpu for tests.
JAX_PLATFORMS=cpu python -m pytest tests -q -p no:cacheprovider
JAX_PLATFORMS=cpu python -m pytest tests/test_emulator_spec.py tests/test_additive_model.py
pytest -m "not gpu"                       # skip GPU-marked tests
ruff check --fix src/ scripts/ tests/ examples/ && ruff format src/ scripts/ tests/   # ruff==0.11.7
cd docs && make html                      # needs the `docs` extra

# Emulators
python -m arachne.emulator describe <ckpt.eqx>
python scripts/train_parrot_emulator_v2.py --library <lib.hdf5> --output outputs/emulators/<name>.eqx
python scripts/validate_parrot_emulator.py --emulator <ckpt.eqx> --library <lib.hdf5>

# Demos (GPU node for anything but --quick toy runs)
python examples/demo_correlated_field.py --truth prior --quick          # toy emulator, CPU
python examples/demo_resolved_sed_fitting.py --quick
python examples/fit_jades_dja.py --list-targets
python -m arachne.results info results.h5
```

Where it runs:

| Place | Python | Notes |
|---|---|---|
| Tom's laptop, `~/Documents/software/arachne` | `pixi run --manifest-path ~/Documents/astro_code/pixi.toml python ...` (arachne installed editable); ruff via `pixi exec --spec ruff==0.11.7 ruff ...` | Never `pip install` into the env. Local data: `ARACHNE_EMULATOR`, `ARACHNE_PSF_DIR`, `ARACHNE_DATA_ROOT`, `ARACHNE_JADES_DR3_PHOT` point the examples at it |
| COSMA, `/cosma/apps/dp276/dc-harv3/arachne` | `source /cosma/apps/dp276/dc-harv3/venv_simformer/bin/activate` | Shared venv: never install or upgrade. Sphinx is not in it. The examples' default paths are the COSMA ones |
| MPCDF GPU nodes (A100 / H100) | a read-only conda env plus `PYTHONPATH=<checkout>/src` | Float64 fits go here. Never import JAX on a GPU node while a job runs there without `JAX_PLATFORMS=cpu` |

Test suite on 2026-11-17 (laptop, CPU): **1107 passed, 1 skipped** in about 9 min when the JADES/DJA servers are reachable (the network tests skip themselves offline).

## Emulators: rules

Full description: [docs/guide/emulators.md](docs/guide/emulators.md).

- **Never hard-code a parameter list, bounds or neutral values in library code.** Take them
  from the emulator: `emulator.param_names`, `emulator.spec.param_bounds`, `.defaults`,
  `.priors`, `.mass_param`, `.redshift_param`, `.sfh_ratio_params`.
- **Load with `arachne.load_emulator`**, never a class's `.load`. To run a fit written for one
  parameter list with a checkpoint that has another, pass `params=` / `fixed=` / `bands=`: the
  result is an `EmulatorView`, itself an `SPSEmulator`.
- **Bounds are the emulator's training domain.** A driver with its own preferred bounds
  intersects them: `spec.clip_bounds(my_bounds)`. Check `spec.meta["bounds_source"]`:
  `"fallback"` means the checkpoint recorded no domain and the bounds are the v6 fit bounds,
  which are wrong for other libraries. Fix it once with `python -m arachne.emulator annotate`.
- **The emulator's per-band error comes from the checkpoint**:
  `emulator.spec.band_error(redshift)`, added in quadrature to the model-error floor. Do not
  read a stand-alone `perband_fractional_error.json` for a checkpoint that carries a table (a
  v2 file used with a v3 emulator is the mistake this prevents). `spec.band_errors is None`
  means no table: measure one (`validate_parrot_emulator.py --embed`) or attach one
  (`python -m arachne.emulator annotate CKPT --band-errors FILE`). The tables on the two
  existing checkpoints only resolve z ≈ 2; elsewhere they return the all-redshift value.
- **Known parameter sets.** v6 library (checkpoint `parrot_emulator_v2.eqx`, production): 12
  inputs `redshift, log_mass, slope, fesc_lya, dust_bump_amplitude, log10metallicity, Av,
  logsfr_ratio_0..4`; z ≤ 14. v7 library (`parrot_emulator_v3.eqx`, candidate): 17 inputs,
  `slope` renamed `dust_slope`, plus `fesc, f_birth, dust_temperature, dust_emissivity,
  dust_mir_alpha`; z ≤ 20, log_mass 6–12.5, ratios within ±5. A new parameter needs an entry in
  `KNOWN_PARAMS` (`emulator/spec.py`) only if it should have a default, a label or an alias.
- **Fixing the extra v7 inputs is a modelling choice.** At defaults (`fesc 0.1`, `f_birth 1`)
  v3 and v2 NIRCam fluxes differ by 3–20 % at the same nominal parameters. Every measured
  result in the docs is with v2.
- **The example drivers still carry a 12-name `SPS_PARAM_NAMES` constant** (and the demos many
  name-keyed tables: `NEUTRAL`, `CORR_LENGTHS`, the SFH basis). They load any checkpoint through
  a view of those 12 names. Fitting one of the *new* v7 inputs needs those tables extended.
- `fesc_lya` and `dust_bump_amplitude` have no leverage in NIRCam at z ≈ 2 (< 0.3 % per unit):
  fix them there.
- Photometry is exactly linear in mass (`flux ∝ 10**log_mass`). Blind starts, the mass
  re-solve and the unbounded per-pixel mass field all rely on it; an emulator without a mass
  input (`spec.mass_param is None`) cannot be used with them.

## Module map (`src/arachne/`)

| Module | Responsibility |
|---|---|
| `data/observation.py`, `data/psf.py` | `ObservationCube` (flux/variance/mask `(N_bands,H,W)` nJy, `from_fits`, `to_jax()`), `PSFModel` (per-band kernels, `from_fits`, static `resample(kernel, from_scale, to_scale)` flux-conserving, `resample_to`, `pad_to_image_size`) |
| `data/units.py` | `parse_bunit`, `flux_scale_to_nJy`, `flux_to_nJy`, `variance_to_nJy2`, `weight_to_variance`, `zeropoint_scale_to_nJy`. Everything internal is nJy / nJy². AB zp 31.4 ⇔ 1 nJy |
| `data/multires.py` | `BandImage` (one band on its own grid: flux/variance/mask, `pixel_scale`, `affine`, `ref_pixel`, `psf`, `sky_coords()`), `MultiResolutionObservation` (`from_fits`, `from_dja_fits`, `to_observation_cube`, `to_jax`), `canonical_band_name`, `tangent_plane_affine` |
| `data/dja.py` | DAWN JWST Archive client: `fetch_dja_cutout` / `load_dja_cutout` (thumb service), `fetch_dja_native_cutout` (assoc_mosaic), `dja_filter_names`, `server_reachable` |
| `data/jades.py` | JADES NIRSpec DR4 spec-z catalogue: `download_jades_dr4_specz`, `load_jades_dr4_specz`, `select_targets` |
| `emulator/base.py`, `emulator/spec.py` | `SPSEmulator` ABC (`param_names`, `band_names`, `predict`, `spec`, `param_bounds`); `EmulatorSpec` (bounds, defaults, priors, units, labels, roles, `band_errors` / `band_error(z)`, `clip_bounds`, `subset`, `describe`), `KNOWN_PARAMS` / `param_info` (registry of library parameters and aliases; pure Python) |
| `emulator/io.py`, `emulator/view.py`, `emulator/__main__.py` | `load_emulator(path, params=, fixed=, bands=, rename=)`, `register_emulator_format`; `EmulatorView` (fix / rename / reorder inputs, select bands); `python -m arachne.emulator describe\|annotate` (`--library`, `--bounds`, `--band-errors`) |
| `emulator/validation.py` | `band_error_table(pred, true, z, bands)`, `select_band_errors`, `select_z_bin`, `table_z_bins`, `check_band_error_table` — per-band fractional-error tables in redshift bins (numpy only) |
| `emulator/parrot_emulator_v2.py` | `ParrotEmulatorV2` — **primary emulator class**: analytic mass factorisation (flux exactly ∝ 10**log_mass), Fourier-z features (period `z_scale` from the checkpoint), arsinh SFH-ratio inputs; the checkpoint records names, roles, training domain, units, library; `with_metadata` (domain, units, `band_errors`), `from_synference_library` |
| `emulator/parrot_emulator.py`, `jax_mlp_emulator.py`, `jax_emulator.py` | `ParrotEmulator` (v1, legacy), `SPSMLPEmulator` (Speculator), `JAXFlowEmulator` (legacy flow export) |
| `emulator/fixed_param_wrapper.py` | `FixedParamEmulator` — pins one input; kept for existing code, `EmulatorView` is the general form |
| `spatial/base.py` | `SpatialModel` ABC: `n_params`, `decode`, `log_prior`; optional `model_image` hook and `sample_prior` |
| `spatial/profiles.py` | `Profile` ABC + `GaussianProfile` (5), `SersicProfile` (6), `PointSourceProfile` (2); `PROFILES`, `get_profile`, `sersic_b`, `render_on_grid` / `log_render_on_grid` (analytic flux per pixel on **any** coordinates) |
| `spatial/additive.py` | `AdditiveComponentModel` — **recommended** for multi-component work (see below) |
| `spatial/correlated_field.py` | `CorrelatedFieldMap` — **recommended for resolved maps**: per-pixel emulator inputs as Matérn Gaussian random fields on truncated white Fourier coefficients; roles field / global / fixed / derived, `latent_fields`, couplings, copula marginals, `hyper_priors`; `split_theta` / `join_theta` (`ThetaBlocks`), `param_maps`, `field_maps`, `sample_prior`; `mass_map_from_image`, `wiener_mass_map`, `matern_spectrum` |
| `spatial/gmm.py` | `GaussianMixtureSpatialModel` — **caveat**: blends SPS *parameters* per pixel with unit-sum softmax weights; no surface-brightness profile, every pixel carries a full galaxy mass. Only for smooth parameter gradients; never for bulge/disk |
| `spatial/pixel_map.py` | `FreeFormPixelMap` — per-pixel params, L2 smoothness prior, high-d |
| `spatial/mass_start.py` | `template_mass_map(obs, emulator, ref, templates)` -> `MassStart` (baseline mass map + smoothed per-pixel colour-template correction; template 0 must be the reference `{}`), `default_sfh_dust_templates` |
| `psf/resolution.py` | `mtf_profile`, `psf_k_max(psf_model, mtf_threshold)` — wavenumber where the sharpest band's MTF drops below a threshold (field-basis truncation) |
| `psf/convolution.py` | `PSFConvolver(psf_model, image_shape, pad=True)` / `.from_kernels(...)`; `__call__(image (N,H,W), shifts (N,2)\|None)` — FFT convolution, exact Fourier phase ramp for sub-pixel shifts (+dy down, +dx right); `pad=True` = linear (zero-padded to an even 5-smooth size); `with_psf_ffts` for vmapping |
| `likelihood/gaussian.py` | `GaussianLikelihood(obs, model_error_frac)` / `.from_arrays(...)`; `__call__(model_image, log_noise_scale=None)`; `with_observation`. Per-band likelihoods add **exactly** to the joint one |
| `forward_model/nuisance.py` | `NuisanceModel` — per-band sky / (dy,dx) shifts / log noise scale appended to theta |
| `forward_model/pipeline.py` | `ForwardModel` (single shared grid) — `build`, `log_likelihood`, `log_prior`, `log_posterior` (all pure) |
| `forward_model/multires.py` | `MultiResolutionForwardModel` — every band on its own WCS grid, no data resampling; sky frame (+y North, +x East) |
| `inference/initialisation.py` | `image_moments`, `reference_band_index`, `blind_initial_theta`, `blind_initial_full_theta`, `solve_component_masses` (nuisance-aware), `find_map`, `multistart_map`, `MAPResult` |
| `inference/nss_sampler.py` | `NSSSampler` / `NSSResult` — `blackjax.nss` over `log_prior` + `log_likelihood`, live points from `sample_prior`; equal-weight samples, `logZ`, `logZ_err`, `ess`; bit-exact checkpoint/resume |
| `inference/nuts_sampler.py` | `NUTSSampler` / `NUTSResult` (`get_parameter_map`, `to_hdf5`/`from_hdf5`, `summary`); per-chain vmapped window adaptation, `inverse_mass_matrix=` |
| `inference/laplace.py` | `make_hessian_fn`, `hessian_neg_log_post`, `laplace_covariance`, `WhitenedLogDensity`, `laplace_whitening`, `run_whitened_nuts` — Laplace covariance at the MAP + whitened/dense-metric NUTS. NOT exported from `arachne/__init__.py`: import from `arachne.inference.laplace` |
| `inference/newton.py` | `trust_region_newton` (Steihaug-Toint truncated CG, HVPs only; escapes saddles along negative curvature; converged = interior step with predicted decrease < tol), `dense_preconditioner` / `diagonal_preconditioner`, `NewtonResult`; `block_newton` + `galaxy_block_indices` (exact eigen-modified Newton on the galaxy-level block of a correlated-field fit). Polishes an Adam MAP into a mode before Laplace / whitened MCMC. A mode is local: use several starts |
| `inference/mgvi.py` | `run_mgvi(fm, theta0, key, mode="geovi"\|"mgvi", ...)` -> `MGVIResult`, `mgvi_scale` — geoVI / MGVI via NIFTy.re for `CorrelatedFieldMap` fits (no model-error floor, no fitted noise scale) |
| `inference/standardise.py` | `StandardisingTransform` (theta = T(xi) with an exactly N(0, I) prior), `StandardisedLogDensity` (WhitenedLogDensity-like interface, Jacobian included) |
| `inference/field_precond.py` | `fourier_fisher_diagonal`, `fisher_preconditioner`, `DiagonalWhitenedLogDensity` — diagonal Fisher of the field basis; measured slower than identity for Newton on the toy |
| `inference/mclmc_sampler.py` | `MCLMCSampler` (blackjax 1.6.2 API) + `run_pathfinder`; for `FreeFormPixelMap` |
| `inference/diagnostics.py` | `split_rhat`, `ess`, `chain_movement`, `summarise_chains` on `(n_chains, n_samples, n_params)` |
| `inference/model_comparison.py` | `compare_n_components(make_forward_model, ks, key, ...) -> [ModelComparisonRow]`, `bayes_factor_table` |
| `inference/posterior_predictive.py` | `model_image_samples`, `component_image_samples`, `residual_summary`, `predictive_bands`, `chi2_reduced(_samples)`, `is_multiresolution`, `n_model_params` |
| `inference/batched.py` | `BatchedForwardModel` (vmapped equal-shape cutouts) + `batched_blind_initial_theta`, `batched_find_map`, `batched_multistart_map`, `batched_nuts`, `fit_batch_nss`, `BatchedMAPResult`, `BatchedNUTSResult` |
| `results/schema.py`, `results/io.py` | `BundleContent` / `ComponentContent` / `TruthContent` (numpy mirror of the HDF5 schema `arachne.results` 1.0), `validate(h5)`; `write_content(path, content, mantissa_bits=, overwrite=)` via `content_to_tree` (also used by `ResultsBundle.from_content`) |
| `results/writer.py` | `content_from_forward_model` / `write_bundle` — the ONLY JAX module in `results/`: thin (`thin_indices`), evaluate convolved/intrinsic cubes, maps, scalars, nuisances, components on the same draws (`_chunked_map`); adapters `field_maps` / `component_params` / `decode` |
| `results/reader.py` | `load_bundle`, `ResultsBundle` (`band_image(b, kind, stat)`, `cube`, `band_draws`, `model_draws`, `param_map(name, stat)`, `available_param_stats`, `pixel_param_draws`, `scalar_draws`, `chi2_red_per_band`, `noise_scale`, `last_fallback`), `ComponentsView`; draws lazy, MAP fallback without draws |
| `results/regions.py`, `results/display.py` | `Region` (pixel/box/circle/polygon/from_mask, geometry), `RegionSED` (`model_percentiles`, `chi`, `frac_residual`), `region_sed`, `total_sed`, `pixel_sed`, `component_seds`, `component_region_seds`, `region_param_summary`, `detection_region`; `percentile_limits`, `stretch`, `rgb_composite` |
| `results/metadata.py`, `results/provenance.py`, `results/testing.py`, `results/cli.py` | `PIVOT_UM`, `pivot_um`, `DEFAULT_LABELS`, `short_band_name`; `collect_provenance`; `make_synthetic_content/bundle("field"\|"additive"\|"degraded")`; `python -m arachne.results info\|validate` |
| `viz/` (`sed`, `maps`, `images`, `diagnostics`, `summary`, `style`, `cli`) | Matplotlib figures from a bundle, no JAX: `plot_sed`, `plot_total_sed`, `plot_region_seds`, `plot_component_seds`, `plot_param_maps`, `plot_band_mosaic`, `plot_rgb`, `plot_chi_histograms`, `plot_chi2_per_band`, `plot_summary_page`, `save_all`; `python -m arachne.viz` / `arachne-plots` |
| `gui/` (`app`, `panels`, `state`, `cli`) | Bokeh viewer for a bundle, no JAX: `view`, `ArachneViewer`; numpy-only logic in `gui/state.py`; `python -m arachne.gui` / `arachne-gui` |
| `priors/specs.py`, `priors/spatial.py`, `priors/physical.py` | `DEFAULT_PRIORS`, `default_prior`, `resolve_prior_specs(..., defaults=)`, `build_log_prior`, `build_component_log_prior`, `prior_config_template`, `sigmoid_log_jacobian`; spatial and physical prior helpers |

`scripts/`: `fit_catalogue.py` (shared catalogue driver; its `SPS_PARAM_NAMES` / `PARAM_BOUNDS`
are that script's fit set, narrowed to the loaded emulator's domain), `fit_catalogue_nss.py`,
`fit_one_galaxy.py`, `fit_sed.py`, `fit_mock_jwst.py` / `generate_mock_jwst.py`,
`fit_bulge_disk_fixed_z.py`, training/validation scripts, `convert_correlated_field_run.py`,
`experiments/` (emulator HPO harness; `emulator_lab_v3.py` is the library-agnostic one). The
old GMM parameter-blending demo and its diagnostics were deleted on 2026-11-17 (they are in
the git history); do not resurrect that approach.

`examples/`: `demo_correlated_field.py` (mock recovery with correlated fields; also imported
as a module by `fit_correlated_field_dja.py`, the real-galaxy driver, which sets its module
globals), `demo_resolved_sed_fitting.py` (component mock recovery), `fit_jades_dja.py` +
`real_data_utils.py` (component fits of real JADES galaxies).

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

1. `emulator = load_emulator(path, params=..., bands=...)`; bounds = `emulator.spec.param_bounds`
   (or `spec.clip_bounds(my_bounds)`), names = `emulator.param_names`.
2. `specs = resolve_prior_specs(free_names, None, bounds, defaults=emulator.spec.priors)`;
   `sps_log_prior = build_component_log_prior(names, specs, bounds, shared_param_names=..., fixed_param_names=...)`.
3. `AdditiveComponentModel(K, names, bounds, (H, W), fixed_params={spec.redshift_param: z}, mass_param=spec.mass_param, sps_log_prior=..., profiles=..., pixel_scale=..., normalisation=...)`.
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

## Correlated-field fits: rules

Details and numbers: [docs/guide/correlated_fields.md](docs/guide/correlated_fields.md).

- Pipeline: truth-free start → Adam MAP (float32) → `trust_region_newton` polish (float64) →
  saddle check → Laplace (whitening and quick look only) → `run_whitened_nuts`. Adam alone
  stops tens of nats above the mode; a Laplace covariance built there is wrong.
- **Quote maps from NUTS draws, not from Laplace.** Laplace under-covers the non-mass maps
  (68 % coverage ~0.1 for Z and SFH where NUTS gives ~0.7).
- **Expect several modes** (total mass against the oldest SFH bin). Laplace, multi-start
  Newton and geoVI are local. Run ≥ 4 chains from several starts; a large R-hat with good
  within-chain ESS is a second mode, not a tuning problem.
- **Never pick a mode by its Laplace evidence** (it prefers unconverged ones): lowest −log p
  among converged modes with a positive-definite Hessian.
- **Do not fit field hyperparameters (ℓ, amplitude) at a joint mode**: it overfits (funnel).
  Fix them, choose by evidence, or sample them. Couplings bias the total mass.
- **SFH**: at z ≈ 2 the photometry constrains ~2 SFH directions per pixel. Use `--sfh basis2`
  (two latent fields, all five ratios derived). Mapping some ratios and freeing others as
  globals gives confidently wrong globals. The `continuity` copula prior was worse than
  `legacy` for map03 on honest mocks.
- **MCLMC does not converge on these posteriors** (R-hat 2.4–20). Use whitened NUTS.
- **`run_mgvi` needs `model_error_frac=0` and no fitted noise scale**, so it does not apply to
  real data, where the floor matters.
- **Convergence at ~12k parameters is not solved.** On the honest 17-band 64×64 mocks none of
  three completed NUTS runs converged (R-hat 1.17–1.72, 3–10 h on an H100); basis2 still beat
  map03 on the spiral (total mass +2.1σ against +4.6σ, mass coverage 0.65 against 0.28). At
  ~5k parameters (8 bands) NUTS converges and is calibrated on a prior-draw mock.
- **Real data (first target, JADES 33151, z 2.13, DJA 0.05″ thumbnails)**: PSFs must come from
  unsaturated stars in the same thumbnails (bright stars give broad cores; χ²_red 3.9 against
  1.05 in F115W); use a per-band model-error floor (3 % LW wide, 5 % elsewhere, emulator term
  in quadrature). The fit converges (17k parameters, R-hat 1.014), but **the stellar mass is
  set by the metallicity prior at the 0.35 dex level**: with no prior log Z sits at the
  emulator floor. An open question, not a bug.

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

## Gotchas

- **`pip install`ing anything that touches `jax`/`jaxlib` can silently break GPU support.** The
  CUDA plugin (`jax-cuda12-plugin`, `jax-cuda12-pjrt`) must match `jaxlib` exactly; on a
  mismatch JAX does not error, it falls back to CPU. Verify on a GPU node with
  `python -c "import jax; print(jax.devices())"` (a login node proves nothing).
- **Never upgrade a shared venv.** A `blackjax`/`jax` upgrade drags `numpy` along and breaks
  packages with tighter pins; `pip check` misses runtime breaks, so `import` what you care about.
- **Data S/N above the emulator's accuracy makes a resolved fit over-confident.** Use
  `model_error_frac` on every resolved fit; DJA weight maps carry no source Poisson term.
- **Fitting a shift in every band is exactly degenerate** with moving every component: pass
  `NuisanceModel(..., shift_reference_band=i)`.
- **Frame normalisation is not a surface brightness.** `model_image_on` and multi-resolution
  fits need `normalisation="analytic"`.
- **`float32` gradients are noisy near the mode.** Polish and sample in float64 on datacentre
  GPUs. Checkpoints hold float32 weights and load under x64 (they promote).
- **Checkpoints without a recorded domain get fallback bounds** (see "Emulators: rules").
- **`EmulatorView` logs the inputs it holds at defaults.** Read that line: a silently fixed
  `fesc` or `f_birth` changes the SED.
- **NSS is forced to float32** (`inference/nss_sampler.py`); its intervals on the 34-parameter
  demo are ~0.65× too narrow and float32 is the prime suspect. Not yet tested in float64.
- **`examples/fit_correlated_field_dja.py` mutates `demo_correlated_field` module globals**
  (`FIXED`, `NEUTRAL`, `LEVEL_SIGMAS`, `PARAM_BOUNDS`, bands). Order matters; set them before
  building the model.
- **The results reader, `viz` and `gui` must not import JAX or Equinox**
  (`tests/test_no_jax_imports.py`). Only `results/writer.py` may. `emulator/spec.py` is pure
  Python for the same reason.
- **Create a new sampler object per model/image shape**; compiled graphs are shape-specific.
  The first run pays XLA compile time (NSS: minutes).

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
- Production emulator on COSMA: `scripts/outputs/emulators/parrot_emulator_v2.eqx` (54 bands,
  12 inputs) via `arachne.load_emulator`. Data cache: `/cosma7/data/dp276/dc-harv3/work/arachne_data/`
  (JADES DR4 catalogue, DJA cutouts, `target_scan/final.json` aperture fluxes for the vetted
  GOODS-S targets). Local JADES DR3 GS photometry with total fluxes:
  `/cosma7/data/dp276/dc-harv3/work/catalogs/JADES_DR3_GS_Matched_Specz_total_flux_good.fits`.

## Examples

*(flag lists last checked 2026-09-18 for the two component drivers; re-read `--help` before
quoting a flag. The correlated-field drivers are described in
[docs/guide/correlated_fields.md](docs/guide/correlated_fields.md).)*

`examples/demo_resolved_sed_fitting.py` — the blind-recovery reference. Two **Sérsic**
components (compact quenched bulge + extended dusty disk, common fixed z=2) rendered as a
10-band NIRCam mock with the real `ParrotEmulatorV2` checkpoint and real JADES PSFs (COSMA
paths by default; `ARACHNE_EMULATOR` / `ARACHNE_PSF_DIR` override them; 0.03"/px, 4.8" cutout), contaminated with per-band sky (~N(0, 0.3 nJy/px)) and
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
