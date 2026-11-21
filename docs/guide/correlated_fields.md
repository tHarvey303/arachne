# Correlated-field parameter maps

Per-pixel SPS parameter maps with Gaussian-random-field priors (`CorrelatedFieldMap`). The measurements behind each choice are in [`docs/design/correlated_field_inference.md`](../design/correlated_field_inference.md). Part of the [arachne guide](README.md).

`CorrelatedFieldMap` ([`src/arachne/spatial/correlated_field.py`](../../src/arachne/spatial/correlated_field.py))
fits per-pixel SPS parameter maps of one resolved galaxy without decomposing it into components.
The [empirical spatial priors study](../design/spatial_prior_calibration.md) provides optional
residual Matérn scales measured from MaNGA and GalSyn maps, plus held-out recovery benchmarks.
Every emulator parameter plays one of four roles: a **field** (a map with a Gaussian-random-field
prior; the mass parameter must be one, `log_mass` *per pixel*, unbounded because the emulator is
exactly linear in mass), a **global** (one scalar for every pixel, uniform within its bounds or
with a `global_priors` density), **fixed** (the spec-z; and at z ≈ 2 also `fesc_lya` and
`dust_bump_amplitude`, which change the NIRCam fluxes by < 0.3 % per unit), or **derived** (a
per-pixel linear combination of field maps, soft-clipped to its bounds, with no theta entries and
no prior terms).

Field `k` is `u_k(x) = m_k(x) + a_k + s_k g_k(x) + Σ_j c_kj d_j(x)`: a mean map `m_k`, a free
level `a_k ~ N(0, level_sigma²)` (it replaces the DC mode), a unit-variance stationary field
`g_k` with a Matérn spectrum (`corr_lengths` in pixels, slope `nus`) built from white Fourier
coefficients `z ~ N(0, I)` on a zero-padded grid, and optional couplings to earlier fields (a
resolved Σ*–A_V relation, dust following young stars: `couplings` fixed, `free_couplings`
fitted; the *smooth* part of any relation belongs in `mean_maps`). Only the modes carrying most
of the prior variance are kept (`variance_fraction=0.99`, or an explicit `k_max` per field, e.g.
the wavenumber where the sharpest PSF's MTF drops below a threshold, `psf_k_max`), so a 64×64
map costs a few hundred to ~2000 parameters rather than 4096. The physical map is
`softclip(u_k, bounds)`, or a copula transform (`field_marginals`, e.g. the library's
Student-t(2) on `logsfr_ratio_*`; `copula_level="gaussian"|"physical"` decides whether the
level is added before or after it) that keeps the spatial correlation and the exact `N(0, I)`
prior on theta. Amplitude, correlation length and ν can be fitted with log-normal `hyper_priors`
(non-centred: the basis is fixed, the spectrum re-evaluated in-graph). `latent_fields` plus
`derived_params` implement a reduced SFH: two latent maps whose fixed 2 × 5 basis (the top right-
singular vectors of the fit emulator's `d ln F / d logsfr_ratio` at a neutral point, the demo's
`sfh_basis`) derives all five `logsfr_ratio_i` per pixel. The reason is identifiability: at
z = 2 the photometry constrains only ~2 SFH directions per pixel (≈ `r1 − r2` is unconstrained),
and mapping more ratios, or mixing mapped ratios with free global ones, leaves a null space that
a Laplace step fills confidently and that the stacked pixels then push to confidently wrong
global values. The prior is proper (`sample_prior`), theta is laid out as
`[levels, globals_raw, hyper, couplings, z_field_0, z_field_1, ...]` (use `split_theta` /
`join_theta`), and `param_maps(theta)` returns every per-pixel emulator input (mapped fields,
then derived ones) named by `param_map_names`; `field_maps` returns the fields themselves.

```python
from arachne import CorrelatedFieldMap, mass_map_from_image

spec = emulator.spec                                           # see emulators.md
names, bounds = emulator.param_names, spec.param_bounds        # the emulator's training domain
neutral = {p: spec.defaults[p] for p in names if p not in ("redshift", "log_mass")}
B = sfh_basis(emulator)                                        # (2, 5), examples/demo_correlated_field.py
model = CorrelatedFieldMap(
    image_shape=(64, 64), emulator_param_names=names, param_bounds=bounds,
    field_params=["log_mass", "Av", "log10metallicity", "sfh_a", "sfh_b"],
    mass_param="log_mass",
    fixed_params={"redshift": 2.0, "fesc_lya": 0.2, "dust_bump_amplitude": 1.0},
    corr_lengths={"log_mass": 3.0, "Av": 5.0, "log10metallicity": 12.0, "sfh_a": 4.0, "sfh_b": 6.0},
    amplitudes={"log_mass": 0.25, "Av": 0.3, "log10metallicity": 0.15, "sfh_a": 0.5, "sfh_b": 0.5},
    level_sigmas={"log_mass": 1.0, "Av": 1.0, "log10metallicity": 0.5, "sfh_a": 1.0, "sfh_b": 1.0},
    nus={"log_mass": 1.5, "Av": 1.0, "log10metallicity": 1.0, "sfh_a": 1.0, "sfh_b": 1.0},
    # blind mean maps: log mass from the smoothed reddest band at neutral SPS values, constants elsewhere
    mean_maps={"log_mass": mass_map_from_image(obs, emulator, {"redshift": 2.0, **neutral}),
               "Av": 0.5, "log10metallicity": -2.5, "sfh_a": 0.0, "sfh_b": 0.0},
    latent_fields={"sfh_a": (-10.0, 10.0), "sfh_b": (-10.0, 10.0)},
    derived_params={f"logsfr_ratio_{i}": {"offset": 0.0, "weights": {"sfh_a": float(B[0, i]), "sfh_b": float(B[1, i])}}
                    for i in range(5)},
    pixel_scale=0.03,
)
fm = ForwardModel.build(obs=obs, psf_model=psf, spatial_model=model, emulator=emulator,
                        model_error_frac=0.05)
```

**Inference pipeline** (what `examples/demo_correlated_field.py` does; the design note has the
measurements behind each choice):

1. **Start, truth-free.** The mass mean map comes from the smoothed reddest band converted to
   mass at neutral SPS values (`mass_map_from_image`); `template_mass_map` (`--mass-start template`) corrects it per pixel with the best of a small grid of dust/recent-SFH templates
   chosen from the colours, because one neutral template over-weights young clumps (toy: 0.196 →
   0.163 dex rms). `wiener_mass_map` + `project_to_field` (`--mass-start wiener`) puts a Wiener
   deconvolution into the mass coefficients but gave no measurable gain. Levels start at zero,
   other means at neutral constants. `--n-starts K` adds K − 1 starts that redraw the non-mass
   levels and the globals from the prior.
2. **Adam MAP** (`find_map`, float32), or `block_newton` block-coordinate descent (`--block-newton`, off by default: it helped basis2 and hurt map03 on the toy).
3. **Mode polish in float64** with `arachne.inference.newton.trust_region_newton`
   (Steihaug-Toint truncated CG on Hessian-vector products, trust region measured in the norm
   of the eigen-decomposed dense Hessian `|λ|`, convergence confirmed by a tight CG re-solve),
   then a **saddle check** (`negative_curvature_step` along the negative eigenvectors of the
   Laplace Hessian, re-polish if it lowers −log p). Adam alone stops tens of nats above the
   mode and the old damped Newton-CG stopped early too; everything downstream expands around
   this point. The dense Hessian is the limit, at d ~ 3 × 10⁴. `field_precond.fourier_fisher_diagonal` reproduces the Hessian diagonal matrix-free to ~2 %, but as a Newton
   preconditioner it is *slower* than the identity (`--precond fourier`): the coefficients are
   already prior-whitened and the stiffness is each field's level against its low-k modes,
   which no diagonal scaling removes.
4. **Posterior.** `--posterior laplace`: `laplace_covariance` (|λ| flooring) plus the Laplace
   evidence, Gaussian map samples and coverage of the truth. It is the whitening for NUTS and
   a quick look, but **not the posterior**: on a prior-draw mock it under-covers Z and the SFH
   maps (68 % coverage 0.08–0.13 where NUTS gives 0.67–0.73). `--posterior geovi|mgvi`:
   `arachne.inference.mgvi.run_mgvi` (NIFTy.re `optimize_kl`) in the latents of
   `StandardisingTransform`, an exact elementwise map `theta = T(xi)` with `xi ~ N(0, I)`
   (uniform/Normal/Student-t globals through their truncated CDFs). It is matrix-free and
   scales past the dense-Hessian limit, but needs `model_error_frac=0` and no fitted noise
   scale (NIFTy's Gaussian takes a fixed inverse covariance), and it is local: different seeds
   end in different modes, so it does not replace multi-start. With several starts the kept
   mode is the lowest −log p among converged modes with a positive-definite Hessian, never
   the Laplace evidence, which favours unconverged modes.
5. **Whitened NUTS** (`run_whitened_nuts`, `--nuts`) in `z = L⁻¹(θ − mode)` with a dense metric
   adapted inside the whitened space; after VI the chains run in the standardised latents with
   the metric seeded from the VI scale. Chains start at every mode within 10 nats of the best,
   so R-hat exposes modes that do not mix instead of hiding them.

The SFH prior of mapped ratios (`--sfh-prior`): `continuity` (default; Student-t(2, 0.3)
copula marginals), `legacy` (a Gaussian field, no copula) or `bursty` (σ = 1, the distribution
the emulator was trained on; it traps whole-map levels at the bounds). On the 17-band honest
mocks `continuity` was *worse* than `legacy` for map03 (SFH coverage ≈ 0, total mass 6–13σ
high). None of this applies to the basis2 latent fields, which are plain Gaussian fields.

**What has been measured** (production `ParrotEmulatorV2`, empirical JADES PSFs, 64×64
mocks at S/N 30 on an H100; `--honest` mocks offset the truth's means and hyperparameters from
the fit's prior):

- Whitened NUTS at d ≈ 5000 (8 bands, five maps) converges on a prior-draw mock (R-hat ≤ 1.03,
  median ESS 3300) and is calibrated: 68 % coverage 0.67–0.83 for every map and
  the total log M* within 0.3σ. It takes ~3 h, half of it warmup.
- On the spiral mock (dust lanes on young arms, a radial Z gradient: the prior is wrong) the
  posterior is **bimodal**: total mass against the oldest SFH bin (+0.25 dex of mass hidden
  under young light), one chain in four in the second mode, R-hat 4.7 with 0.2 % of parameters
  above 1.01. Laplace and multi-start Newton never found that mode. The A_V lanes are missed.
- At d ≈ 12,000 (17 bands, ν = 1) NUTS had not converged after 4.3 h (R-hat 1.5, 58
  divergences) from a mode that turned out not to be converged; this is what the Newton
  convergence check and basis2 address.
- MCLMC (blackjax 1.7, per-chain or shared tuning) does not work on these posteriors: R-hat
  2.4–20, chains that do not move. Use NUTS.
- Fitting the field hyperparameters at a joint mode overfits (a funnel); pick them by
  evidence or sample them. Couplings bias the total mass.
- basis2 against map03 at 32 px: χ²_red 1.02 vs 1.79 at equal Adam steps, median |Δ log M*|
  0.077 vs 0.168 dex.

```bash
python examples/demo_correlated_field.py --truth prior --quick                     # CPU, toy emulator
python examples/demo_correlated_field.py --truth spiral --emulator emu.eqx --psf-dir psfs/ \
    --bands medium --honest --nu 1.0 --float64 --posterior laplace --nuts           # H100, ~12k params
python examples/demo_correlated_field.py --truth spiral --emulator emu.eqx --psf-dir psfs/ \
    --bands medium --honest --nu 1.0 --float64 --posterior geovi                    # NIFTy.re geoVI
```

Flags: `--truth {prior,spiral}`, `--bands {wide8,medium}` (8 or 17 NIRCam bands), `--size`,
`--snr`, `--honest`, `--nu`, `--hyper`, `--couple`, `--sfh {basis2,map03,fixed_truth,map_all,
merge0}`, `--sfh-prior`, `--free-globals`, `--mass-start {image,template,wiener}`,
`--k-max-mtf T`, `--n-starts K`, `--block-newton`, `--precond {dense,diag,fourier,none}`,
`--max-newton`, `--posterior {map,laplace,geovi,mgvi}` (`--laplace`, `--n-laplace`,
`--vi-iterations`, `--vi-samples`), `--nuts` with `--sampler {nuts,mclmc}`, `--nuts-warmup`,
`--nuts-samples`, `--nuts-chains`, and the bundle flags below. Outputs: `metrics.json` (fit
configuration, MAP and start statistics against the truth, per-start `modes`, `laplace`/
`geovi` and `nuts` blocks with coverage per map and the total mass), `maps.npz`,
`laplace.npz`, `figures/` and the results bundle `results.h5`.
