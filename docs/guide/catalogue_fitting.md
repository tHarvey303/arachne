# Catalogue and single-galaxy SED fitting

Integrated photometry, one galaxy or a whole catalogue. Part of the [arachne guide](README.md).

Three scripts share the same forward model and likelihood machinery
(`scripts/fit_catalogue.py` is the shared driver — the others import from it); the
per-parameter prior specifications they use live in the library, in
[`arachne.priors.specs`](../../src/arachne/priors/specs.py) (see [Priors](image_fitting.md#priors) below):

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
priors); see `scripts/configs/*.json` for worked examples.

**Parameter set.** `fit_catalogue.py` fits the 12 parameters in its `SPS_PARAM_NAMES`
(`redshift`, `log_mass`, `slope`, `fesc_lya`, `dust_bump_amplitude`, `log10metallicity`, `Av`,
`logsfr_ratio_0..4`) within `PARAM_BOUNDS`. Any checkpoint can be passed to `--emulator`: it is
loaded with `load_emulator(path, params=SPS_PARAM_NAMES)`, which matches aliases (`slope` is
`dust_slope` in the v7 library), holds every other input at the emulator's default and prints
which, and `PARAM_BOUNDS` is narrowed to the emulator's training domain. `fit_sed.py` and
`fit_one_galaxy.py` load the same way but stop if their fixed bounds exceed the domain. New
code should take names and bounds from `emulator.spec` instead of these constants; see
[emulators.md](emulators.md).

`diagnose_parrot_emulator.py` and `validate_parrot_emulator.py` round out the toolkit for
inspecting emulator accuracy before trusting it in a fit.
