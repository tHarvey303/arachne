# arachne guide

Detailed reference, one file per topic. The top-level [README](../../README.md) is the short
introduction; [AGENTS.md](../../AGENTS.md) has the module map, design rules and gotchas.

| File | Read it for |
|---|---|
| [emulators.md](emulators.md) | Loading, describing, swapping and training emulators; `EmulatorSpec`, `EmulatorView` |
| [catalogue_fitting.md](catalogue_fitting.md) | Integrated-photometry fits: `fit_catalogue.py`, `fit_one_galaxy.py`, `fit_sed.py` |
| [image_fitting.md](image_fitting.md) | Light components (`AdditiveComponentModel`): quick start, data flow, profiles, blind starts, priors, nuisances, multi-resolution, batched fits, diagnostics |
| [correlated_fields.md](correlated_fields.md) | Per-pixel parameter maps (`CorrelatedFieldMap`): model, inference pipeline, measured results, demo flags |
| [samplers.md](samplers.md) | Which sampler when; whitening, mode polish, multimodality |
| [results.md](results.md) | Results bundles (HDF5), static plots (`arachne.viz`), browser viewer (`arachne.gui`) |
| [real_data.md](real_data.md) | DJA cutouts, JADES redshifts and PSFs, the real-data and resolved-demo drivers |

Design notes with the measurements behind the choices: [`../design/`](../design/).
