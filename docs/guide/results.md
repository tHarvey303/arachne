# Results bundles, plots and the interactive viewer

Part of the [arachne guide](README.md). The schema and viewer specification are in [`docs/design/results_bundle_gui_spec.md`](../design/results_bundle_gui_spec.md).

A fit is saved as one self-describing HDF5 **results bundle** (schema `arachne.results` 1.0):
the data, the MAP model, thinned posterior draws of the model cubes and parameter maps, scalars,
truth (for mocks), named regions, metrics and provenance. The static plots and the viewer read
only the bundle, never the emulator, so they run anywhere (no GPU, no checkpoint).

## Bundles

```python
from arachne import write_bundle, load_bundle, region_sed, total_sed, Region

# after a fit: thins `samples` to n_draws, evaluates every product on the same draws
write_bundle("results.h5", forward_model, theta_map=theta_map, samples=samples,
             posterior_source="nuts",        # or laplace, nss, mclmc, geovi, mgvi, map
             n_draws=200, n_chains=4, truth=None,
             regions={"core": core_mask}, metrics=metrics, title="my galaxy")

b = load_bundle("results.h5")            # draws are read lazily on first use
b.band_image(3, "chi")                   # data/model/intrinsic/residual/chi/truth_*, (H, W)
b.param_map("Av", "std")                 # map/p16/p50/p84/mean/std/truth/diff/z/information
sed = region_sed(b, Region.circle(b.image_shape, 31.5, 31.5, 4.0))
sed.model_percentiles(), sed.chi(), sed.frac_residual()
```

- **Contents.** `/observation` (flux, variance, mask, pivots), `/model/{convolved,intrinsic}`
  (`map`, `draws (S,B,H,W)`, p16/p50/p84/mean/std), `/params` (maps and their draws, prior std,
  information `1 - (std/prior_std)^2`, scalars), `/components` for additive models (SEDs,
  profiles, convolved components without sky), `/truth`, `/posterior` (theta, draw index,
  chain id, diagnostics), `/regions`, `/metrics/json`, `/extra`.
- **Adapters.** `CorrelatedFieldMap`: maps = `param_maps` (every per-pixel emulator input:
  mapped fields, then derived ones; latent SFH fields are left out), or `field_maps` for a
  model without derived parameters; scalars = free globals, fitted
  hyperparameters as `amplitude[Av]` / `corr_length[Av]`, couplings `c[Av<-log_mass]`, and
  `total_log_mass`. `AdditiveComponentModel`: no maps; components ordered compact-first; scalar
  `total_log_mass`. Anything else: `decode` reshaped into maps (best effort, with a warning).
  `MultiResolutionForwardModel` is not supported yet (`grid="per_band"` is reserved).
- **Size.** Lossless float32, gzip + shuffle: ~106 MiB for 64x64 x 17 bands, S=200, five maps.
  `mantissa_bits=12` (draws only, ~1e-4 relative error) brings that to ~66 MiB; fewer draws
  scale it linearly.
- **Regions.** `Region.pixel/box/circle/polygon/from_mask` (pixel-centre membership, geometry
  kept). The writer adds `all` and `segment` (`detection_region`: inverse-variance S/N image,
  smoothed, 3 sigma, largest component). `region_sed` sums the data and every model draw over
  the same valid pixels; `region_param_summary` gives region mass (log-sum) and mass-weighted
  means per draw.
- **No draws (MAP only, or degraded bundles converted from old runs).** p50/p16/p84/mean fall
  back to the MAP and set `bundle.last_fallback`; `std` raises `KeyError`. Use
  `available_param_stats()` to see what a bundle really holds.
- **In memory and tests.** `ResultsBundle.from_content(BundleContent(...))` behaves exactly
  like a loaded file; `arachne.results.testing.make_synthetic_bundle(path, "field" |
  "additive" | "degraded")` builds small fixtures without JAX.
- **CLI.** `python -m arachne.results info results.h5 [--json]` and
  `python -m arachne.results validate *.h5` (also `arachne-bundle` when installed).
- **Demo and old runs.** `examples/demo_correlated_field.py` writes `results.h5`
  (`--bundle-draws`, `--bundle-mantissa-bits`, `--bundle-no-intrinsic`, `--bundle-all`
  for `results_laplace.h5` / `results_geovi.h5` next to the NUTS bundle, `--bundle-plots`,
  `--no-bundle`).
  `scripts/convert_correlated_field_run.py RUN_DIR [--emulator CKPT|toy]` turns old 4-field
  run directories into degraded bundles (`results_degraded.h5`).

## Static plots

`arachne.viz` draws publication-style matplotlib figures from a bundle (matplotlib only, no
JAX). Every function takes a `ResultsBundle` or a path and returns a `Figure` built without
pyplot: call `fig.savefig(...)` in scripts; Jupyter displays it.

```python
from arachne import viz
from arachne.results import load_bundle, region_sed

b = load_bundle("results.h5")
viz.plot_total_sed(b).savefig("sed.pdf")              # total SED + sigma and % residuals
viz.plot_sed(region_sed(b, b.regions["core"]), show_intrinsic=True)
viz.plot_region_seds(b, ["core", "segment"])          # or layout="overlay"
viz.save_all(b, "figs", fmt="png")                    # every standard figure
```

```bash
python -m arachne.viz results.h5 --outdir figs [--format png|pdf|svg] [--regions core,segment]
arachne-plots results.h5 --outdir figs                # same, once installed
```

- **SEDs.** `plot_sed` shows the observed aperture fluxes (±1σ) against the PSF-convolved
  model over the same pixels: median line, 16–84% band, MAP as open diamonds, truth as
  crosses. The intrinsic (unconvolved) model is optional, dashed and labelled
  "not aperture-comparable". The x axis is log micron with band names on a top axis. The
  residual panels are χ (exactly `RegionSED.chi()`, with ±1/±3 guides) and
  (obs − model)/model in %. `plot_component_seds` gives the per-component SEDs of additive
  models: intrinsic, and observed frame over the total region.
- **Maps and images.**
  - `plot_param_maps`: rows are parameters; the columns are truth, median, σ,
    (median − truth)/σ and information, whichever are available.
  - `plot_band_mosaic`: data, model, residual and χ per band. Data and model share an asinh
    stretch with the colourbar in nJy; χ is fixed to [−4, 4].
  - `plot_rgb`: data, model and residual on one shared asinh scale. The residual image is
    grey where the fit is perfect.
  - `plot_regions`: region outlines drawn on the data.
- **Diagnostics.** `plot_chi_histograms` shows per-pixel χ per band against N(0, 1).
  `plot_chi2_per_band` shows reduced χ² against wavelength, with and without the model floor.
  `plot_summary_page` puts it all on one page.
- **Missing pieces.** Each band keeps one colour (blue to red by wavelength) in every
  figure. Figures leave out what a bundle lacks:
  - no draws: no 16–84% bands, and the model is labelled MAP;
  - no truth, intrinsic model or components: those panels are omitted.

  `save_all` skips `component_seds` and `param_maps` when they do not apply.

## Interactive viewer

`arachne.gui` is a browser viewer for a bundle: plain bokeh with Python callbacks, served by a
bokeh server (bokeh only, so it runs on the GPU nodes), or shown inline in JupyterLab through
Panel. Install with `pip install 'arachne[gui]'` (`'arachne[gui-jupyter]'` adds Panel).

```bash
arachne-gui results.h5                     # opens a browser tab at http://127.0.0.1:5006/
python -m arachne.gui results.h5 --port 5099 --no-browser   # same without the entry point
arachne-gui results.h5 --compare results_laplace.h5         # overlay a second posterior
```

```python
from arachne.gui import view
v = view("results.h5")   # ArachneViewer
v.show()                 # in Jupyter: a panel.pane.Bokeh (falls back to bokeh.io.show)
```

- **Remote use.** On the node, run `arachne-gui results.h5 --no-browser --port 5006`. On your
  laptop, run `ssh -L 5006:localhost:5006 <node>` (add `-J <gateway>` through a jump host) and
  open `http://localhost:5006/`. If the local port differs from the remote one, also pass
  `--allow-websocket-origin localhost:<local port>`. The CLI prints this recipe. It sets
  `JAX_PLATFORMS=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing arachne, so
  the viewer never takes GPU memory.
- **Images.** Three panels share pan and zoom: two band panels (data, model, intrinsic,
  residual, χ, truth model and truth residual, with a model statistic) and a parameter map
  with a statistic menu built from `available_param_stats()`. Additive bundles show each
  component's fraction of the model instead. Controls cover the stretch (linear, asinh, log,
  sqrt), percentile range and colormap. Residual and χ use a symmetric diverging scale. A
  toggle switches the χ model floor; another blanks the image outside `segment`. Hovering
  reads out the raw pixel value.
- **Selections.** Tap a pixel, tap a circle of adjustable radius, drag a box or lasso, draw a
  polygon (the box, lasso and polygon tools are in each image toolbar), or pick a named
  region. The SED panel then shows the observed fluxes (±1σ) against the convolved model
  summed over the same pixels: median line, 16–84% band, MAP markers, and optionally the
  intrinsic model and truth. The grey reference is the total SED. The σ and fractional
  residual panels match `RegionSED.chi()` and `frac_residual()`. Posterior histograms show
  region mass (log-sum) and mass-weighted means from `region_param_summary`, plus the global
  scalars, with truth marked. "Keep as overlay" pins up to five selections in colour.
  "Export regions" writes them to `<bundle>.regions.json`; the bundle itself is never
  modified.
- **Additive models.** A component selector restricts the model images, the SED component
  curves (convolved components over the selected pixels) and the histograms (that
  component's parameters) to one component.
- **MAP-only and degraded bundles.** A banner explains what is missing. SEDs use the MAP
  model, and the uncertainty bands and histograms are hidden.
- **Scripting.** The callbacks call public methods, which you can also use directly:
  `set_band`, `set_image_kind`, `set_param`, `select_pixel`, `select_circle`,
  `select_indices`, `select_region`, `set_component`, `keep_selection`, `current_seds()`.
  The numpy-only logic is in `arachne.gui.state`.
