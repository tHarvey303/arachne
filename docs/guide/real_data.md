# Real data (JADES + DJA) and the resolved demo

Part of the [arachne guide](README.md).

## Real data: JADES + DJA

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

[`examples/fit_jades_dja.py`](../../examples/fit_jades_dja.py) (with
[`examples/real_data_utils.py`](../../examples/real_data_utils.py)) is the worked end-to-end driver
*(flags checked against the script on 2026-11-01; under active development)*: 9-band NIRCam DJA
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

## Resolved demo

[`examples/demo_resolved_sed_fitting.py`](../../examples/demo_resolved_sed_fitting.py) is the
end-to-end reference for the *component* pipeline *(flags checked against the script on
2026-11-01; it is under active development, so check `--help` before trusting a flag)*; the
correlated-field counterpart is `examples/demo_correlated_field.py` above.
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
