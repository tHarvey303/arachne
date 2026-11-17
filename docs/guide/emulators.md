# Emulators

How emulators are trained, loaded, described and swapped. Part of the [arachne guide](README.md).

An emulator maps a matrix of SPS parameters `(N, n_params)` to photometry `(N, n_bands)` in
nJy. Everything else in arachne only needs the three things on
[`SPSEmulator`](../../src/arachne/emulator/base.py): `param_names`, `band_names` and
`predict`. No fitting code names an emulator class or a parameter list.

## Loading

```python
from arachne import load_emulator

emulator = load_emulator("parrot_emulator_v3.eqx")     # exactly as trained
print(emulator.spec.describe())                          # inputs, domain, defaults, priors
```

```bash
python -m arachne.emulator describe parrot_emulator_v3.eqx [--bands]
```

`load_emulator` reads any registered checkpoint format (`ParrotEmulatorV2`, and the legacy
`ParrotEmulator`). A new emulator class registers its own loader with
`register_emulator_format(archive_key, loader)`; `archive_key` is an `.npz` key only that
format writes.

## What an emulator says about itself: `emulator.spec`

`emulator.spec` is an [`EmulatorSpec`](../../src/arachne/emulator/spec.py):

| Field | Meaning |
|---|---|
| `param_names`, `band_names` | Inputs and outputs in `predict` column order |
| `param_bounds` | `{name: (lo, hi)}`, the domain the emulator is valid on. `emulator.param_bounds` is a shortcut |
| `defaults` | Neutral value per input: the blind-start value, and what an unused input is held at |
| `priors` | Default prior spec per input (see `arachne.priors.specs`) |
| `units`, `labels` | For tables and plots |
| `mass_param`, `redshift_param`, `sfh_ratio_params` | The inputs with special roles |
| `meta` | `bounds_source` (`"checkpoint"` or `"fallback"`), `library`, and for views `fixed` / `renamed` |
| `band_errors`, `band_error(redshift)` | The emulator's own fractional flux error per band; see [below](#emulator-error-per-band) |

Checkpoints trained since November 2026 record the min/max of every input over the training
library, its units and the library file name. An older checkpoint has none of this, so its
spec falls back to the registry of known library parameters (`KNOWN_PARAMS` in `spec.py`) and
says so in `meta["bounds_source"]`. The fallback bounds are the fit bounds used with the v6
library and are **wrong for other libraries**, so record the real domain once:

```bash
python -m arachne.emulator annotate old.eqx --library the_library_it_was_trained_on.hdf5
python -m arachne.emulator annotate old.eqx --bounds bounds.json     # {"name": [lo, hi], ...}
```

This rewrites only the metadata; the weights and predictions are unchanged
(`ParrotEmulatorV2.with_metadata` in Python).

Build a model from the spec, not from constants:

```python
spec = emulator.spec
bounds = spec.param_bounds                       # or spec.clip_bounds(my_bounds): mine, narrowed
priors = resolve_prior_specs(free_names, None, bounds, defaults=spec.priors)
model = AdditiveComponentModel(..., emulator_param_names=emulator.param_names,
                               param_bounds=bounds, mass_param=spec.mass_param,
                               fixed_params={spec.redshift_param: z_spec})
```

## Emulator error per band

A checkpoint carries its own accuracy: the fractional flux error per band
(`pred/true − 1` on held-out library rows with true flux above 1 nJy), in redshift bins. A
resolved fit adds it in quadrature to its model-error floor, and because the table travels
with the checkpoint, a fit cannot pick up another emulator's numbers.

```python
err = emulator.spec.band_error(redshift=2.13)             # (n_bands,), robust scatter ("mad")
err, z_bin = emulator.spec.band_error(2.13, return_bin=True)
floor = np.sqrt(model_err**2 + err**2)                    # per band, for model_error_frac
```

- Each band uses the narrowest bin containing the redshift in which it has at least 100
  rows; otherwise `"all"`. The returned label names the commonest bin and any fallbacks. `z_bin=` forces one; `stat=` selects `mad`, `std`, `p95`
  or `bias`.
- `emulator.spec.band_errors` is `None` when the checkpoint has no table, and `band_error`
  then raises. `examples/fit_correlated_field_dja.py` uses the checkpoint's table by default,
  warns and uses zero if there is none, and takes `--emulator-err-json` as an override.
- Making a table: `scripts/validate_parrot_emulator.py --emulator CKPT --library LIB --embed`
  measures it and stores it (on a random library subset, which can include training rows);
  `scripts/experiments/emulator_lab_v3.py --save-checkpoint` embeds the held-out one;
  `python -m arachne.emulator annotate CKPT --band-errors table.json` attaches an existing
  file. `arachne.emulator.validation.band_error_table(pred, true, z, bands)` is the function
  behind all three.
- **The table is measured over the whole library.** A view that holds inputs fixed sees one
  slice of parameter space, where the error can differ. Both production tables were
  re-measured on 2026-11-18 on held-out rows (v2: 50,000 of the v6 library; v3: 100,000 of
  the v7 library) with bins across the whole redshift range, as any newly made table has.
  Above the redshift where a band drops out there are no detected rows, so that band falls
  back to a wider bin.
- **Measure in full precision.** On A100/H100 the default float32 matmul (TF32) inflates the
  measured scatter by about 4 % (up to 9 %); `emulator_lab_v3.py` sets
  `jax_default_matmul_precision = "highest"` when scoring.

## Swapping emulators: `EmulatorView`

Libraries differ in their inputs. The v6 library has 12; the v7 library has 17, renames
`slope` to `dust_slope` and adds `fesc`, `f_birth`, `dust_temperature`, `dust_emissivity` and
`dust_mir_alpha`. To run the same fit with either, ask for the inputs and bands the fit uses:

```python
emulator = load_emulator(path, params=FIT_PARAMS, bands=FIT_BANDS)   # an EmulatorView
emulator = load_emulator(path, fixed={"redshift": 2.13, "fesc": 0.0}, bands=FIT_BANDS)
```

- `params`: the inputs the view exposes, in the fit's order. Aliases in `KNOWN_PARAMS` are
  matched (`slope` / `dust_slope`); `rename={checkpoint_name: exposed_name}` covers anything else.
- `fixed`: inputs held at given values (checked against the domain).
- Every remaining input is held at `spec.defaults` and logged. Pass `fill_defaults=False` to
  make that an error instead.
- `bands`: the output bands, in the fit's order.

The view is an `SPSEmulator` with a matching `spec` (`spec.meta["fixed"]` lists what is held
constant), so models, samplers and the results writer take it unchanged. `EmulatorView(inner,
...)` does the same for an emulator already in memory. `FixedParamEmulator` (one fixed input) is
the older, narrower form and is kept for existing code.

The example drivers (`scripts/fit_catalogue.py`, `examples/demo_correlated_field.py`,
`examples/fit_correlated_field_dja.py`, `examples/demo_resolved_sed_fitting.py`,
`examples/fit_jades_dja.py`) load through this path with their own 12-name list, and narrow
their bounds with `spec.clip_bounds`. They still carry that list as a module constant.

Holding the extra v7 inputs at defaults is a modelling choice, not a formality: with
`fesc = 0.1`, `f_birth = 1` the v3 emulator's NIRCam fluxes differ from v2's by 3–20 % at the
same nominal parameters (different dust model). Choose the fixed values deliberately.

## Training

Both production emulators are trained from the HDF5 model library produced by
[synference](https://github.com/tHarvey303/synference).

```bash
python scripts/train_parrot_emulator_v2.py --library galaxy_library.hdf5 \
    --output outputs/emulators/my_emulator.eqx
python scripts/validate_parrot_emulator.py --emulator outputs/emulators/my_emulator.eqx \
    --library galaxy_library.hdf5 --output-dir outputs/validation/
```

`ParrotEmulatorV2.from_synference_library` works on any library: `param_names="all"` takes the
library's own list, `mass_param` / `redshift_param` / `sfr_ratio_prefix` name the inputs with
special treatment, and `z_scale` (the Fourier redshift period) defaults to the library's
maximum redshift rounded up. The recipe, relative to the original Parrot design
([Mathews et al. 2023](https://iopscience.iop.org/article/10.3847/1538-4357/ace720)):

1. **Analytic mass factorisation.** Photometry is exactly linear in stellar mass, so the
   network predicts flux per 10⁹ M☉ and `log_mass` is reapplied analytically. Every
   blind-initialisation trick downstream depends on this exact linearity.
2. **Fourier redshift features.** sin/cos encodings represent the sharp Lyman-break cutoff
   without extreme depth (the largest single accuracy gain, about 2×).
3. **arsinh-compressed SFH-ratio inputs.** Keeps the informative core of heavy-tailed
   `logsfr_ratio_*` inputs that plain z-scoring squashes.

The module docstring of
[`parrot_emulator_v2.py`](../../src/arachne/emulator/parrot_emulator_v2.py) has the ablations.

## Checkpoints in use

| Checkpoint | Library | Inputs | Notes |
|---|---|---|---|
| `parrot_emulator_v2.eqx` | v6 (1M rows, z ≤ 14, Calzetti) | 12 | Production; every measured result so far. No recorded domain (fallback bounds). Error table: all 54 bands, 13 redshift bins (NIRCam at z ≈ 2: 0.40 %) |
| `parrot_emulator_v3.eqx` | v7 (10M rows, z ≤ 20, BimodalPacman + Casey12) | 17 | Candidate (`BW4_k128_full`): NIRCam error at z ≈ 2 0.34 % against 0.41 %, detected-band scatter 4× lower. Domain recorded. Error table: all 54 bands, 14 redshift bins. Not yet used for a science fit |

The copies with the domain and error table attached are the annotated ones (2026-11-18);
copies made before that lack them, which `python -m arachne.emulator describe` shows.

## Experiments / training harness

`scripts/experiments/` holds the architecture-search harness: `hpo_lab.py` (Optuna HPO),
`emulator_lab.py` / `train_final_v2.py` (v6 ablations, COSMA paths), `emulator_lab_v3.py`
(library-agnostic: any synference library, step budget, bright-weighted loss, per-band
fractional errors, writes V2-format checkpoints), `bench_inference.py`,
`make_summary_figs.py` and shell drivers. Results go to `scripts/outputs/` (gitignored).

## Legacy emulators

`ParrotEmulator` (v1) is the original Parrot-style MLP with GELU activations and an
arsinh-magnitude output, superseded by V2 and kept for old checkpoints. `SPSMLPEmulator` uses
the [Speculator](https://arxiv.org/abs/1911.11778) architecture (Alsing et al. 2020), trained
in log10-flux space. `JAXFlowEmulator` is a legacy flow export. All write `.eqx` checkpoints:
Equinox pytrees, differentiable and JIT-compilable, with no PyTorch dependency.
