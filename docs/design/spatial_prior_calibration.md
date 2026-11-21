# Empirical spatial priors for resolved SED fitting

These are population estimates for the **residual** stellar attenuation, stellar metallicity,
and recent star-formation fields after fitting a radial mean. They use the
`CorrelatedFieldMap` Matérn convention `[1 + (k ell)^2]^-(nu + 1)` with `nu=1.5`.
At this smoothness the unfiltered correlation is `(1 + r/ell) exp(-r/ell)`, and the
e-folding separation is `2.146 ell`. The estimates describe projected, two-dimensional maps.
The amplitudes are intrinsic residual standard deviations in the indicated coordinates.

## Measurements

| Sample and field | Break length in \(R_e\) | Break length in kpc | Residual amplitude | Two-sided lengths in training sample |
| --- | ---: | ---: | ---: | ---: |
| MaNGA stellar continuum \(A_V\) | 0.119 | 0.768 | 0.293 mag | 12/36 |
| MaNGA mass-weighted stellar \(\log_{10}Z\) | 0.046 | 0.334 | 0.353 dex | 4/36 |
| MaNGA 10–100 Myr / 100–1000 Myr log SFR contrast | 0.054 | 0.307 | 1.45 dex | 1/18 active |
| GalSyn effective rest-V attenuation | 0.292 | 0.392 | 0.499 mag | 20/39 |
| GalSyn mass-weighted stellar \(\log_{10}Z\) | 0.404 | 0.500 | 0.112 dex | 14/39 |
| GalSyn 100 Myr \(\log_{10}\mathrm{sSFR}\) | 0.278 | 0.353 | 0.291 dex | 12/39 |

The radius and kpc columns come from separate population fits. The MaNGA metallicity and SFH
numbers are sensitive to the resolution boundary and the treatment of continuum-fitting
scatter. Use them to characterize the limits of the IFU experiment. In GalSyn, the log-length
scatters in radius units are 0.47, 0.59 and 0.62, respectively, in natural logarithms.
The log-amplitude scatters are 0.52, 0.39 and 0.46. The held-out profile scores gave limited
support for any partition by activity or snapshot. For example, snapshot-specific
metallicity lengths improved the mean test profile score by 0.05 nats per galaxy with a
95% galaxy-bootstrap interval of −0.14 to +0.26. The radius-normalized pooled model is the
benchmark prescription. A pooled dust length in kpc scored +0.41 nats per GalSyn galaxy with
an interval of −0.12 to +0.97.

### MaNGA observations

The 48-galaxy IFU sample used pyPipe3D v3_1_1/3.1.1, QCFLAG 0, 127-fibre observations,
`0.012 <= z <= 0.055`, `6 <= Re_arc <= 17`, axis ratio above 0.25, and catalogued
stellar mass between \(10^9\) and \(10^{11.7}\) solar masses. The sample was stratified by
catalogued Hα sSFR at \(10^{-11}\,\mathrm{yr}^{-1}\), with 36 training and 12 test galaxies.
Metallicity comes from stellar continuum plane 9 in the SSP extension. Dust comes from
the stellar continuum attenuation plane. The SFH diagnostic converts light fractions to
relative template masses with the MaStar template normalization and applies a \(10^{-4}\)
fractional mass floor to both age intervals. Only active galaxies enter that diagnostic.
[pyPipe3D release and data model](https://www.sdss4.org/dr17/manga/manga-data/manga-pipe3d-value-added-catalog/),
[Sánchez et al.](https://arxiv.org/abs/2206.07062)

The fit uses one datum per continuum segment inside \(1.5R_e\), up to 160 per galaxy.
It projects the covariance through the FITS r-band PSF and the flux-weighted spaxel footprints
of the segments. The mean design has a constant and a linear slope in elliptical radius.
The restricted likelihood integrates out those two coefficients and profiles the intrinsic
amplitude and extra independent bin scatter on each trial length. Reported uncertainties enter
the diagonal with floors of 0.025 mag for attenuation and 0.025 dex for metallicity.
A 0.20 dex working floor is used for the template SFH contrast. Correlated spectral-fit errors
and the nonlinear response of the continuum fit to PSF mixing are outside this likelihood.

### GalSyn simulations

The 60 GalSyn DR1 TNG50 progenitors span snapshots 50, 33 and 25 (approximately \(z=1,2,3\)).
They use the first viewing angle and GalSyn's particle-based line-of-sight dust attenuation
with its SKIRT-like curve. The split has disjoint descendant lineages, stratified at the
median sSFR within each snapshot. Four galaxies had fewer than 25 suitable map pixels and
56 entered the spatial analysis. The aperture uses a projected half-stellar-mass radius
within 20 kpc. The map fits use pixels within \(1.5R_e\) with sufficient local mass to reduce
particle shot noise. The recent-sSFR diagnostic floors the maps at \(10^{-11}\,\mathrm{yr}^{-1}\).
Effective dust is the attenuation from the matched dust-on and dust-off maps in the NIRCam
band closest to rest-frame V. The photometry was rendered by GalSyn with FSPS and a
particle-based dust column, independently of the Arachne emulator.
[GalSyn DR1](https://github.com/aabdurrouf/GalSyn_dr1),
[Abdurro'uf et al.](https://arxiv.org/abs/2603.23986)

### Resolution check

The injection experiment generated continuous Matérn fields on a padded pixel grid, blurred
the images, and averaged them through irregular 2×2 continuum bins. The median fitted lengths
were 0.66, 2.19 and 4.20 pixels for injections of 0.65, 2 and 4 pixels. The shortest case hit
a searched length limit in 14 of 16 realizations. The MaNGA and GalSyn tables use both
profile bounds and a resolution threshold when counting two-sided lengths.

## Applying a calibration

The loader returns keyword mappings in the model's pixel and parameter units. Supply a projected
effective radius in pixels or a physical pixel size in kpc. Merge the per-field dictionaries
with your mass and SFH configurations.

```python
from arachne.priors import load_spatial_calibration

updates = load_spatial_calibration(
    "docs/design/spatial_priors/galsyn.json",
    redshift=2.0,
    re_pixels=5.0,
)
for key, values in updates.items():
    model_kwargs[key] = {**model_kwargs.get(key, {}), **values}
model = CorrelatedFieldMap(**model_kwargs)
```

For sampled fits, `sample_hyperparameters=True` uses the population scatter for
`log_corr_length` and `log_amplitude`. The benchmark uses fixed medians to compare MAP
fits with the same optimization budget. Keep the physical calibration tied to its sample
and the mean-map model. Use physical or photometric information for trends such as the
radial metallicity gradient that was fitted out during calibration.

The GalSyn SFH test rotates the emulator's two-dimensional SFH basis so that the first
direction is locally most sensitive to \(\log_{10}\mathrm{sSFR}_{100}\). The slope at constant
SFR converts the measured log-sSFR scatter into a Gaussian latent amplitude. The conversion
is local, depends on bandpasses and redshift, and compares per-formed-mass sSFR from the emulator
with the simulator's per-surviving-mass sSFR. The benchmark reports both absolute and
mean-subtracted spatial errors. The general calibration loader returns the direct attenuation
and metallicity maps. The experimental SFH transfer is in the GalSyn benchmark.

## Reproducing the measurements

The [MaNGA sample](spatial_priors/manga_sample.json) and
[GalSyn sample](spatial_priors/galsyn_sample.json) record the IDs, data paths and splits.
The [MaNGA calibration](spatial_priors/manga.json),
[GalSyn calibration](spatial_priors/galsyn.json) and
[injection measurements](spatial_priors/injections.json) are small JSON files.
Commands below run from the repository root with the normal Arachne dependencies.

```sh
run=outputs/spatial_priors
manga=$run/measurement
galsyn=$run/galsyn/measurement

python scripts/experiments/calibrate_pipe3d_spatial_priors.py fetch \
  --cache "$run/pipe3d" --outdir "$manga"
OPENBLAS_NUM_THREADS=1 python scripts/experiments/calibrate_pipe3d_spatial_priors.py measure \
  --cache "$run/pipe3d" --outdir "$manga"
python scripts/experiments/pool_spatial_measurements.py \
  "$manga/measurements.json" "$manga/calibration.json"

python scripts/experiments/calibrate_galsyn_spatial_priors.py fetch \
  --cache "$run/galsyn/cache" --outdir "$galsyn"
OPENBLAS_NUM_THREADS=1 python scripts/experiments/calibrate_galsyn_spatial_priors.py measure \
  --cache "$run/galsyn/cache" --outdir "$galsyn"
python scripts/experiments/pool_spatial_measurements.py \
  "$galsyn/measurements.json" "$galsyn/calibration.json"

OPENBLAS_NUM_THREADS=1 python scripts/experiments/validate_spatial_covariance.py
```

The MaNGA benchmark forwards held-out continuum maps through Parrot V2 in eight fitted and
two withheld bands. The known maps retain MaNGA's seeing and continuum-segment resolution.
The GalSyn benchmark uses the released dust-attenuated images, empirical JADES DR3 PSFs
at 0.03 arcsec per pixel, and two held-out medium bands. Both use the same noise draw
and optimizer budget within each pair. The checkpoint must carry its simulator recipe
and band-error table.

```sh
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/experiments/benchmark_spatial_calibration.py \
  --cache "$run/pipe3d" --sample "$manga/sample.json" \
  --calibration "$manga/calibration.json" --emulator "$ARACHNE_EMULATOR" \
  --output "$manga/h100-manga.json"

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/experiments/benchmark_galsyn_spatial_priors.py \
  --cache "$run/galsyn/cache" --measurements "$galsyn/measurements.json" \
  --calibration "$galsyn/calibration.json" --emulator "$ARACHNE_EMULATOR" \
  --psf-dir "$ARACHNE_PSF_DIR" --size 24 --newton-steps 100 \
  --output "$galsyn/h100-galsyn.json"
```

This project makes use of the MaNGA-Pipe3D dataproducts. We thank the IA-UNAM MaNGA team
for creating this catalogue, and the Conacyt Project CB-285080 for supporting them.
