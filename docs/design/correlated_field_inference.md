# Correlated-field inference: what exists, what to try next

Status: working notes, 2026-10-30. Scope: `CorrelatedFieldMap` fits (10³–10⁵ parameters: field
levels, global SPS parameters, optional hyperparameters and couplings, and the white Fourier
coefficients of each map). Results so far are in the vault notes "arachne correlated-field
progress" and "arachne TODO" §1b.

## 1. Current pipeline (`examples/demo_correlated_field.py`)

1. **Start.** Mass mean map from the reddest band (optionally Wiener-deconvolved coefficients),
   SPS levels at neutral values. `--n-starts K` adds K−1 starts that redraw the non-mass levels and
   the globals from the prior (mass level and field coefficients kept).
2. **Adam MAP** (`find_map`, float32).
3. **Mode polish** (`arachne.inference.newton.trust_region_newton`, float64). Steihaug-Toint
   trust-region Newton with Hessian-vector products, preconditioned by the eigen-decomposed dense
   Hessian (`|λ|`, floored). Converged = an interior CG step whose predicted decrease is
   < 0.01 nat. Replaced a damped Newton-CG that handed indefinite systems to CG and stopped on a
   small last-step gain.
4. **Saddle check.** The Laplace Hessian at the polished point is eigen-decomposed; if a step
   along a negative-curvature eigenvector (`negative_curvature_step`) lowers −log p, polish again
   from there (≤ 3 times). Needed because CG stopping early on an interior step never explores weak
   negative directions.
5. **Laplace** (eigen-modified covariance, `|λ|`) and Laplace evidence. With several starts the
   best evidence is kept and `metrics.json["modes"]` records each start's log Z, −log p, whitened
   distance to the best mode, total mass and coverage.
6. **Whitened MCMC** (NUTS or MCLMC) in z = L⁻¹(θ − mode). With several starts, chains begin at
   the modes within 10 nats of the best (`run_whitened_nuts(initial_positions=...)`), so R-hat
   can reveal modes that do not mix instead of hiding them.

SFH prior: `--sfh-prior bursty` (default) puts Student-t(ν = 2, σ = 1) on every
`logsfr_ratio`. Free globals get the density directly; mapped ratios get it as their per-pixel
marginal through a Gaussian copula (`CorrelatedFieldMap(field_marginals=...)`), so the prior on θ
stays N(0, I) and the whitening and evidence stay exact.

## 2. Known weaknesses

- **One mode at a time.** Laplace, the Newton polish and the whitening all expand around a single
  point. SED fitting is multimodal (age–dust–metallicity, SFH bin trading), and the honest-mock
  SFH screen showed confidently wrong SFH ratios whose failure mode changes with the seed.
- **Point curvature.** Laplace widths come from the curvature at one point. Near-flat directions
  (ratio_0 at 12k parameters: posterior sd 0.9) and weak negative curvature make them depend on
  where the optimiser stopped. A properly converged mode had +45 nats more evidence than the old
  stopping point *and worse coverage* (spiral, map03, legacy prior).
- **Cost of the dense Hessian.** ~100 s and d² memory at d ≈ 12k; infeasible beyond ~3×10⁴.

## 3. Candidate methods

| method | handles multimodality | non-Gaussian shape | evidence | cost at d ≈ 10⁴ | role here |
|---|---|---|---|---|---|
| Mean-field Gaussian VI | no (reverse KL picks a mode) | no; drops correlations | ELBO (lower bound) | low | not useful: the bin trading *is* a correlation |
| Low-rank + diagonal Gaussian VI | no | no, but curvature averaged over the mass | ELBO | moderate (rank-r, stochastic gradients) | replacement for Laplace and a better whitening for NUTS |
| Normalising-flow VI / NeuTra-HMC | weakly | yes | ELBO / via MCMC | high to train at 10⁴ | flow as an HMC preconditioner, keeping MCMC exactness |
| Laplace mixture over multi-start modes | yes, if starts find the modes | per mode only | sum of Laplace Z | K × (polish + Hessian) | already almost there with `--n-starts`; weights unreliable when modes are non-Gaussian |
| Multi-path Pathfinder (+ PSIS) | yes | Gaussian per path | importance-weighted | low | mode *finder*; PSIS weights will likely degenerate at d = 10⁴ (check k̂). `run_pathfinder` exists in `mclmc_sampler.py` |
| Stacking of non-mixing chains (Yao et al. 2022) | yes | via MCMC | no | MCMC per mode | combine chains started at different modes |
| Tempered SMC with whitened HMC moves | yes | yes | yes | ~10–100× NUTS | reference answer for mode weights; blackjax adaptive tempering |
| Nested sampling (NSS) | yes | yes | yes | ≳ 2d inner steps per replacement: ~10⁹ calls | not for full maps (see TODO §1b) |
| INLA-style split | yes, in the low-d block | in the low-d block | yes | inner polish + log det per outer point | strongest candidate if the multimodality is galaxy-level (below) |
| SBI proposals (synference NPE) | yes | yes | no | ms | seeds for `--n-starts` covering the modes |

### INLA-style split

Separate the ~10–20 galaxy-level quantities φ (field levels, global SPS parameters,
hyperparameters, couplings) from the field coefficients z. Given φ, z should be close to
Gaussian, so

    p(φ | d) ∝ p(d, z*(φ), φ) |H_z(φ)|^(-1/2),   z*(φ) = argmax_z p(d, z, φ)

and φ can be explored with a method that handles multimodality (SMC or nested sampling in ~20
dimensions), while z is integrated out by Laplace. Each outer point needs:
- an inner polish, warm-started from the previous point (`trust_region_newton`, a few iterations);
- `log det H_z`, by stochastic Lanczos quadrature (~10³ HVPs, 10–30 s) rather than a dense
  Hessian (~100 s).

Risk: if the bin trading is also *pixel-local* (per-region SFH modes), the conditional of z is not
Gaussian and the split does not remove the problem. The `map_all` runs and the per-start mode table
show which case we are in.

### Tempered SMC

`blackjax.adaptive_tempered_smc(logprior_fn, loglikelihood_fn, mcmc_build_kernel, mcmc_init,
mcmc_parameters, resampling_fn, target_ess, num_mcmc_steps=...)`
([sampling book](https://blackjax-devs.github.io/sampling-book/algorithms/temperedsmc/)).
Particles move through the tempered sequence `π_λ ∝ prior × L^λ` from λ = 0 to 1. Each next λ
is chosen so the ESS of the incremental weights stays at `target_ess` (e.g. 0.5); particles are
then resampled and moved with a few MCMC steps (HMC/NUTS, step size and metric retuned per stage
via `inner_kernel_tuning`). The sum of the log mean incremental weights is log Z. Its main
strength is that a population of particles keeps every mode that carries prior × likelihood
mass, with the right relative weights.

Fit to this problem:
- **Prior start is easy.** The θ prior is N(0, I) on the coefficients plus simple densities on
  levels and globals, so `sample_prior` gives exact λ = 0 particles.
- **Cost from the prior is high.** The number of stages grows with the information gained from
  prior to posterior. Here that is thousands of nats (information ≈ 0.6–0.95 per map pixel,
  17 bands), i.e. hundreds of stages. Each stage moves ~10³ particles with several HMC
  trajectories. Each particle is a full 64×64×17 forward model, so the total is ~10⁷–10⁸
  likelihood calls: many hours on the H100. Use it as a reference run on one mock, not routinely.
- **Moves need per-stage whitening.** The Laplace whitening is right only at λ = 1. Use the
  particle cloud's diagonal or low-rank covariance per stage.
- **Bridge from an approximation instead of the prior.** Temper `q^(1−λ) × p(θ | d)^λ`, with q a
  normalised, samplable approximation: the evidence-weighted mixture of the multi-start Laplace
  modes, or SBI-seeded Gaussians. The number of stages then scales with KL(q → posterior), not
  KL(prior → posterior). log Z stays exact because q is normalised. It is the SMC analogue of the
  posterior repartitioning in TODO §1b. q must cover every mode (widen it, or mix in a little
  prior); this ties SMC to the mode-finding steps above.
- **Natural outer sampler for the INLA split.** With ~20 galaxy-level parameters and the field
  integrated out by Laplace, tempered SMC needs ~10³ particles and tens of stages. Each particle
  evaluation is an inner polish plus a log det, vmappable in principle but memory-heavy. This is
  probably its best role here.

### SBI-seeded starts

The amortised synference posteriors (same SPS library) run on the galaxy's integrated or
per-region photometry give multimodal proposals for φ in milliseconds. Drawing `--n-starts`
seeds from them should cover the modes far better than prior redraws. Pair with
`project_to_field` for per-region seeds.

## 4. Decision plan

1. Read the 3-start runs (`runs/2026-10-30_optimizer/*_ms3`) and the saddle-check runs:
   - Do different starts reach different modes (whitened distance ≫ 1, log Z differences of a few
     to tens of nats)?
   - Does the bursty SFH prior restore coverage on its own?
2. **Distinct modes found** → multi-path Pathfinder and SBI-seeded starts for mode finding; chains
   from each mode, combined by stacking; tempered SMC on one mock as the reference for the
   weights. If the modes live in φ, prototype the INLA split.
3. **One mode, still miscalibrated** → the problem is the Gaussian approximation and the prior,
   not mode finding: low-rank Gaussian VI in place of Laplace (and as the NUTS whitening), then
   revisit the SFH parametrisation (fewer bins, or a continuity prior across neighbouring bins).
4. Long term, beyond d ~ 3×10⁴: drop the dense Hessian (diagonal or low-rank preconditioner,
   stochastic log-determinants).
