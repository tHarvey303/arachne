# Post-processing with the full SPS model

The emulator predicts photometry and nothing else. After a fit, `arachne.postprocess` rebuilds
the Synthesizer model the emulator's training library was generated with and evaluates it on
posterior draws. That gives:

- the **spectrum** behind the fitted photometry (lines, breaks, continuum), per region;
- **derived quantities** the emulator has no output for: surviving stellar mass, SFR over 10
  and 100 Myr, mass-weighted age, UV slope, D4000, and the SFH in physical units;
- a direct **check of the emulator** on the posterior itself (simulator / emulator flux ratio
  per band).

It is optional and slow compared with the fit (about 15 ms per pixel and draw on one core), so
it runs on a subset of draws.

## The pieces

| Piece | Where | Needs |
|---|---|---|
| Simulator recipe (JSON) | `emulator.spec.simulator`, stored in the checkpoint | nothing |
| `Simulator`, `check_recipe` | `arachne.postprocess.simulator` | Synthesizer, the SPS grid file |
| `postprocess_bundle`, `load_postprocess` | `arachne.postprocess.bundle` | the above, a results bundle |
| `plot_postprocess` | `arachne.viz.postprocess` | a post-processed bundle |

Synthesizer comes with synference, which arachne depends on. JAX is not used.

## The recipe

The recipe describes the library's model declaratively: SPS grid name, cosmology, emission
model (synference's description format: factory, arguments, dust curve and dust emission
components), IGM model, SFH class with its age-bin rule, metallicity model, how each emitter
attribute is computed from the library inputs, and a few dozen reference library rows
(parameters, photometry, some derived values). No code is stored or executed. The layout is in
the docstring of `arachne/postprocess/recipe.py`.

```bash
# write the recipe into a checkpoint (weights unchanged)
python -m arachne.emulator annotate CKPT.eqx --simulator-library LIBRARY.hdf5

# rebuild the model here and compare with the stored library rows; exit status 1 unless exact
python -m arachne.emulator check-simulator CKPT.eqx --grid-dir /path/to/grids
```

Library headers do not hold everything in declarative form, so `recipe_from_library` combines
the header with a preset (`arachne.postprocess.recipe.PRESETS`, chosen from the file name):

| Preset | Library | Supplied by the preset |
|---|---|---|
| `v6` | BPASS, Continuity SFH, Calzetti dust (the `ParrotEmulatorV2` production library) | emission model, IGM (Asada25), age bins (z_form = 20), `tau_v = Av / 1.086` |
| `v7` | BPASS, Continuity SFH, birth-cloud + ISM dust, Casey12 | age bins (z_form = 30), `tau_v_ism`, `tau_v_birth = f_birth Av / 1.086` |

A new library family needs a new preset. Whatever the source, `check-simulator` is the test
that the recipe is right.

**Run `check-simulator` on every machine.** It needs the *same grid file* the library was
made with. A grid of the same name but another build is not the same model: with the
`_updated-star-fraction` build of the BPASS grid the v6 photometry is off by about 1 % (up to
15 % in bands holding strong lines) and surviving masses by up to 0.02 dex. With the right
grid (Synthesizer 1.3.0, 2026-11-19): median |ΔF/F| 4e-5 at rest 0.13–5 µm, and SFRs, ages
and surviving masses equal to the library's to rounding. Bluewards of Lyman-alpha the
agreement is about 0.3 % (IGM and Lyman-alpha treatment).

Grid search order: `grid_dir` argument, `$ARACHNE_SPS_GRID_DIR`, `$SYNTHESIZER_GRID_DIR`,
Synthesizer's default directory.

## Post-processing a bundle

```bash
arachne-bundle postprocess results.h5 --emulator CKPT.eqx --grid-dir /path/to/grids \
    --draws 16 --nproc 8 --plot
```

or

```python
from arachne.postprocess.bundle import postprocess_bundle, load_postprocess

res = postprocess_bundle("results.h5", emulator="CKPT.eqx", n_draws=16, n_proc=8)
res.total("log_mass_surviving")          # (p16, p50, p84) over the processed region
res.stat(res.maps["sfr_100"], "p50")     # (H, W) map
res.sed["segment"]                       # (D, L) spectra in nJy; res.wavelength_um
```

It reads the parameter maps, scalars and fixed values from the bundle (names are matched
through the aliases of `arachne.emulator.spec`, so a fit that used an `EmulatorView` works),
processes the pixels of one region (`segment` by default) for `n_draws` evenly spaced stored
draws, and writes a `postprocess` group into the same file:

| Path | Content |
|---|---|
| `sed/<region>/{draws,p16,p50,p84}` | observed-frame f_ν [nJy] summed over the region, on `wavelength_um` |
| `photometry/<region>/…` | the same through the bundle's filters |
| `maps/<name>/…` | per-pixel `log_mass_formed`, `log_mass_surviving`, `sfr_10`, `sfr_100`, `age_mass_weighted`, `beta_uv`, `d4000` |
| `totals/<region>/<name>/draws` | region totals (masses and SFRs summed, age mass-weighted) |
| `sfh/<region>/…`, `sfh_agebins_myr` | SFR per age bin |
| `emulator_check/{p16,p50,p84,total}` | simulator / emulator flux ratio per band |

Spectra and totals are stored for every bundle region inside the processed one. Pass
`extra_regions={name: mask}` to `postprocess_bundle` for further apertures (a clump, the core). The spectra
are the *intrinsic* light of the pixels (no PSF), like the bundle's intrinsic model, so they are
not aperture-matched to the data. `plot_postprocess(bundle)` draws the spectrum, the SFH and
maps of surviving-mass and SFR surface density, age and sSFR.

Things to know:

- **`log_mass` is formed mass.** The emulators are trained on the mass formed, and the fitted
  maps and `total_log_mass` are therefore formed masses. `log_mass_surviving` is the mass in
  stars and remnants today; for the z ≈ 2 galaxies fitted so far it is about 0.25 dex lower.
- `sfr_10`, `sfr_100` and `age_mass_weighted` are Synthesizer's values (the SFH binned on the
  grid's ages), identical to the library's supplementary columns. The `sfh` arrays are the
  analytic piecewise-constant SFH; the two differ by a few per cent over 10 Myr.
- A MAP-only bundle is processed as one "draw" (`attrs["from_draws"]` is false).
- By default the run refuses to start unless the library check is `exact`
  (`--allow-approximate` / `strict=False` overrides).
- Cost: 8 draws of a 3,000-pixel segment take 2–3 minutes with 8 processes on a laptop.

## Synthesizer versions

Everything that touches the Synthesizer API is in `postprocess/simulator.py`
(`_resolve_class` knows where classes have lived). Tested with Synthesizer 1.3.0. A new
Synthesizer release should be accepted only after `check-simulator` passes with it.
