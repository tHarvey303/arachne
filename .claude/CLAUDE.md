# CLAUDE.md

Guidance for Claude Code when working in this repository. The HPC/venv/GPU-node setup lives in
the parent `/cosma7/data/dp276/dc-harv3/CLAUDE.md`; this file is about arachne itself.

## Project Overview

**arachne** is a pure-JAX/Equinox package for galaxy SED fitting with neural SPS emulators:

1. **Catalogue / single-galaxy fitting** (`scripts/fit_catalogue.py` and friends) — integrated
   photometry, GPU-batched Pathfinder+NUTS or nested slice sampling (NSS, gives `logZ`).
2. **Image-level forward modelling** — a galaxy as `K` additive light components
   (`AdditiveComponentModel`) or a free-form pixel map, PSF-convolved per band, fitted blind
   with NSS / NUTS / MCLMC.

Emulators are trained from synference HDF5 libraries (`ParrotEmulatorV2` is the production
one). Everything downstream of the checkpoint is JAX: differentiable, jit-able, no PyTorch.

## Commands

```bash
source /cosma/apps/dp276/dc-harv3/venv_simformer/bin/activate   # shared venv: never pip install/upgrade
JAX_PLATFORMS=cpu pytest                  # full suite on CPU (login node is fine; ~minutes)
JAX_PLATFORMS=cpu pytest tests/test_additive_model.py tests/test_initialisation.py tests/test_nss_sampler.py
pytest -m "not gpu"                       # skip GPU-marked tests
ruff check --fix src/ scripts/ tests/ && ruff format src/ scripts/ tests/
cd docs && make html

# Emulator training / validation
python scripts/train_parrot_emulator_v2.py --library <lib.hdf5> --output outputs/emulators/parrot_emulator_v2.eqx
python scripts/validate_parrot_emulator.py --emulator <ckpt.eqx> --library <lib.hdf5>

# Resolved bulge+disk demo (GPU node, see below)
python examples/demo_resolved_sed_fitting.py --quick
```

**Known pre-existing test failures** (not regressions): the `TestFromSynerenceLibrary` /
`TestMetadataReaders` tests in `tests/test_parrot_emulator.py` (6 failures as of 2026-09-15;
`KeyError: "Requested parameter 'redshift' not found in library. Available: []"`) — the
`ParrotEmulator` synference-library metadata readers do not parse the synthetic v4 HDF5
layout the tests build. Everything else passes on CPU (316 passed, 1 skipped, ~4 min).

## Module map (`src/arachne/`)

| Module | Responsibility |
|---|---|
| `data/observation.py`, `data/psf.py` | `ObservationCube` (flux/variance/mask `(N_bands,H,W)` nJy; `to_jax()`), `PSFModel` (per-band kernels, `from_fits`) |
| `emulator/parrot_emulator_v2.py` | `ParrotEmulatorV2` + `load_emulator` — **primary emulator**: analytic mass factorisation (flux exactly ∝ 10**log_mass), Fourier-z features, arsinh SFH-ratio inputs |
| `emulator/parrot_emulator.py`, `jax_mlp_emulator.py`, `jax_emulator.py` | `ParrotEmulator` (v1, legacy), `SPSMLPEmulator` (Speculator), `JAXFlowEmulator` (legacy flow export) |
| `emulator/fixed_param_wrapper.py` | `FixedParamEmulator` — pins one input; superseded for spatial work by `AdditiveComponentModel(fixed_params=...)` |
| `spatial/base.py` | `SpatialModel` ABC: `n_params`, `decode`, `log_prior`; optional `model_image` hook (default per-pixel decode→predict) and `sample_prior` |
| `spatial/additive.py` | `AdditiveComponentModel` — **recommended** for multi-component work (see below) |
| `spatial/gmm.py` | `GaussianMixtureSpatialModel` — **caveat**: blends SPS *parameters* per pixel with unit-sum softmax weights; no surface-brightness profile, every pixel carries a full galaxy mass. Only for smooth parameter gradients; never for bulge/disk |
| `spatial/pixel_map.py` | `FreeFormPixelMap` — per-pixel params, L2 smoothness prior, high-d |
| `psf/convolution.py` | `PSFConvolver` — FFT convolution with pre-computed PSF FFTs |
| `likelihood/gaussian.py` | `GaussianLikelihood(obs, model_error_frac=0.0)` — chi-squared; optional fractional model-error floor in quadrature (adds the `log var_eff` term) |
| `priors/specs.py` | Prior spec dicts: `DEFAULT_PRIORS`, `resolve_prior_specs`, `build_log_prior` (catalogue, `(N,)`), `build_component_log_prior` (image, `(K,N_emulator)`, shared counted once / fixed ignored), `sigmoid_log_jacobian`, `prior_config_template` |
| `priors/physical.py`, `priors/spatial.py` | Simple vector priors; `GradientPenaltyPrior`, `TotalVariationPrior` |
| `forward_model/pipeline.py` | `ForwardModel` — calls `spatial_model.model_image` → convolver → likelihood; exposes `log_likelihood`, `log_prior`, `log_posterior` (all pure) |
| `inference/initialisation.py` | Blind init: `image_moments`, `blind_initial_theta`, `solve_component_masses`, `find_map`, `multistart_map`, `MAPResult` |
| `inference/nss_sampler.py` | `NSSSampler` / `NSSResult` — `blackjax.nss` over `log_prior` + `log_likelihood`, live points from `sample_prior`; equal-weight samples, `logZ`, `logZ_err`, `ess` |
| `inference/nuts_sampler.py` | `NUTSSampler` / `NUTSResult` (`get_parameter_map`, `to_hdf5`); `max_num_doublings` passed to warmup and kernel |
| `inference/mclmc_sampler.py` | `MCLMCSampler` (blackjax 1.6.2 API) + `run_pathfinder`; for `FreeFormPixelMap` |

`scripts/`: `fit_catalogue.py` (shared catalogue driver; imports priors from the library),
`fit_mock_jwst.py` / `generate_mock_jwst.py` (toy additive mock), `fit_bulge_disk_fixed_z.py`,
training/validation scripts, `experiments/` (emulator HPO harness; `archive_gmm_blend/` = retired
diagnostics from the old parameter-blending demo, do not resurrect).

## The additive model in one screen

```
I_b(y,x) = Σ_k F_kb(θ_k) · P_k(y,x)      F_k = emulator SED at component k's OWN total mass
                                          P_k = bivariate Gaussian, normalised to sum to 1
theta = concat([block_0 ... block_{K-1}, shared_raw]),  n_params = K*(5+N_free) + N_shared
block_k = [mu_y, mu_x, log_sigma_y, log_sigma_x, atanh_rho, sps_raw_k (N_free,)]
phys = lo + (hi-lo)*sigmoid(raw);  sigma = exp(log_sigma);  rho = tanh(atanh_rho) clipped ±0.99
```

- Roles: every emulator parameter is **fixed** (`fixed_params`), **shared** (`shared_param_names`,
  one value for all components, e.g. redshift) or **per-component** (default; `mass_param` must be).
- Prior = sigmoid Jacobian (normalised uniform in physical space) + optional `sps_log_prior`
  callable on the `(K, N_emulator)` physical matrix (use `build_component_log_prior`) +
  Gaussians on centre / log-size / atanh_rho. `sample_prior` draws from it (NSS needs this).
- Helpers: `split_theta`/`join_theta`, `component_params` → `(mu, sigma, rho, sps_phys)`,
  `profiles`, `component_seds`, `order_components_by_size` (label symmetry).
- `decode` is a *summary* map (mass-weighted params; mass column = log10 Σ_* per pixel) for
  plotting / `get_parameter_map`. The likelihood never calls it.
- Costs `K` emulator evaluations per likelihood, independent of image size.

## Standard image-fit workflow

1. Bounds = emulator training domain (`scripts/fit_catalogue.py: PARAM_BOUNDS`).
2. `specs = resolve_prior_specs(free_names, None, bounds)`;
   `sps_log_prior = build_component_log_prior(names, specs, bounds, shared_param_names=..., fixed_param_names=...)`.
3. `AdditiveComponentModel(K, names, bounds, (H, W), fixed_params={"redshift": z}, mass_param="log_mass", sps_log_prior=...)`.
4. `fm = ForwardModel.build(obs=obs, psf_model=psf, spatial_model=model, emulator=emulator, model_error_frac=0.05)`
   (`model_error_frac` is passed through to `GaussianLikelihood`; `obs_jax = fm.observation`).
5. `theta0 = blind_initial_theta(model, obs_jax)`; `map = multistart_map(fm, theta0, [{}, {"Av": 0.3}, {"Av": 2.0}])`.
6. `NSSSampler(fm, num_live=500).run(key)` (samples + logZ) or `NUTSSampler(fm).run(map.theta, key)`.
7. `jax.vmap(model.order_components_by_size)(result.samples)` before any summary.

The README Quick Start is this workflow verbatim and has been executed against a dummy emulator.

## Resolved demo

`examples/demo_resolved_sed_fitting.py`: 2-component (bulge + dusty disk, common fixed z=2)
10-band NIRCam mock rendered with the real `ParrotEmulatorV2` checkpoint
(`scripts/outputs/emulators/parrot_emulator_v2.eqx`) and real JADES PSFs (hard-coded cosma
path), noise added, fitted **blind** (truth only generates the data and grades the result).
CLI: `--sampler {nss,nuts}` (default nss), `--quick`, `--model-err-frac 0.05`, `--noise-scale`,
`--num-live`, `--num-inner-steps`, `--termination`, `--n-warmup`, `--n-samples`, `--seed`,
`--outdir`. Run it on a GPU node inside tmux:

```bash
ssh gn004 && tmux new -s demo
source /cosma/apps/dp276/dc-harv3/venv_simformer/bin/activate
cd /cosma/apps/dp276/dc-harv3/arachne && python examples/demo_resolved_sed_fitting.py --quick
```

Outputs under `outputs/demo/resolved_sed_fitting/`: `truth.json`, `map.json`, `posterior.hdf5`,
`figures/*.png`. First run pays XLA compile time.

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
- **Use a fractional model-error floor** (`model_error_frac`) whenever data S/N exceeds emulator
  accuracy (always, for resolved fits); keep the `log var_eff` term or the floor is exploitable.
- **NSS needs a proper prior** (`sample_prior`) and prior-distributed live points; passing MAP
  points as live points invalidates `logZ`.
- **Break label symmetry after sampling** (`order_components_by_size`), not with the prior.
- `log_posterior` must stay pure (no I/O, no Python branching on traced values); float32
  throughout; never numpy/scipy in the hot path.
- Create a new sampler object per model/image shape (compiled graphs are shape-specific).

## Conventions

- `ruff` (line-length 100, Google docstrings; `__init__.py` and `examples/` docstrings exempt).
- FITS in, HDF5 out; JAX arrays inside the forward model, numpy only at I/O boundaries.
- Units: fluxes nJy, log10 masses in Msun (component **totals** in the additive model),
  parameter names follow the synference/emulator conventions (`log_mass`, `Av`, `logsfr_ratio_*`, ...).
- Tests: CPU, 3 bands × 16×16, tiny dummy `SPSEmulator` subclasses (see
  `tests/test_additive_model.py`); GPU-only tests marked `@pytest.mark.gpu`.
- Emulator checkpoints and run outputs live in `outputs/` / `scripts/outputs/` (gitignored).
