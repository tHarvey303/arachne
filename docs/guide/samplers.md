# Samplers

Which sampler for which problem, and the rules that were learned the hard way. Part of the [arachne guide](README.md).

| Sampler | Best for |
|---|---|
| `NSSSampler` (`blackjax.nss`) | Image models with a proper prior and **≲ 40 parameters**: gradient-free, live points drawn from `sample_prior` (no starting point), handles label/size/mass multimodality by construction, returns equal-weight samples plus `logZ`, `logZ_err`, ESS — the tool for comparing different `K`. Many more likelihood evaluations than NUTS; use a GPU. Checkpoints and resumes bit-exactly. |
| `NUTSSampler` / `fit_catalogue.py --sampler nuts` | GPU-batched catalogue fitting; image models above ~40 parameters, started from the polished blind MAP and sampled in **whitened** coordinates (see below) |
| `arachne.inference.laplace.run_whitened_nuts` | The practical way to run NUTS on a resolved fit: builds the Laplace covariance at the MAP, samples `theta = MAP + L z` with a dense metric adapted inside the whitened space, and returns a `NUTSResult` in `theta` coordinates. The reference sampler for `CorrelatedFieldMap` fits (d ≈ 5k–12k; ~3 h at 5k on an H100), started from a `trust_region_newton` mode, not an Adam stop |
| `arachne.inference.mgvi.run_mgvi` | geoVI / MGVI through NIFTy.re for `CorrelatedFieldMap` fits: matrix-free, so it scales past the dense Laplace Hessian (d ≳ 3 × 10⁴); needs `model_error_frac=0` and no fitted noise scale; local (does not replace multi-start). Toy: mean within 0.3 nats of the Newton mode, sample sd / Laplace sd median ratio 1.03; not yet validated at production size |
| `batched_nuts` | Many equal-shape cutouts at once: vmapped warmup + sampling over (galaxy, chain), with per-galaxy diagnostics |
| `MCLMCSampler` | Designed for `FreeFormPixelMap` (high-d, ~45k–67k params; O(1) gradient evals per effective sample vs NUTS's O(d^{1/4})). **Did not converge on a 5k-parameter `CorrelatedFieldMap` posterior** with either per-chain or shared tuning (`tune()`, blackjax 1.7: R-hat > 2, tuned step sizes unstable from start to start); use whitened NUTS there |
| `fit_catalogue.py --sampler nss` (`blackjax.nss`) | Catalogue fits: multimodal posteriors, per-galaxy log-evidence (`logZ`); sequential, not GPU-batched |

`ForwardModel` exposes `log_likelihood`, `log_prior` and `log_posterior` separately; NSS uses
the first two, the gradient samplers the third. `run_pathfinder` provides a fast L-BFGS
warm-start (MAP position + diagonal inverse-mass-matrix estimate) that can be passed to
either gradient-based sampler to skip expensive warmup.

Two hard-won rules, measured on the 62-parameter demo in September 2026. **NSS from the prior
is impractical above ~40 parameters**: at 62 an outer step took over 3 s and the run was still
~1.5e6 nats below the MAP after 50 steps; below that threshold NSS is the best tool, because it
also gives `logZ`. And **NUTS needs a dense metric here**: the curvature eigenvalues at the mode
span 1e-1 to 4e7 and the stiff directions are correlated, so a diagonal mass matrix collapses
the warmup step size to ~1e-7, every trajectory saturates the tree-depth cap and R-hat reaches
the thousands. Sampling whitened coordinates `theta = MAP + L z`, with `L` the Cholesky factor
of the Laplace covariance at a properly polished MAP, lifts the step size by four orders of
magnitude, and a dense metric adapted *inside* the whitened space mops up the residual
mass-exchange ridge the Laplace approximation misses.
`arachne.inference.laplace` packages all of that — `hessian_neg_log_post`,
`laplace_covariance`, `WhitenedLogDensity`, `laplace_whitening` and the one-call
`run_whitened_nuts(fm, theta_map, key, n_chains=4, dense_mass_matrix=True)`, whose result comes
back in `theta` coordinates with R-hat and ESS recomputed there. Check them on every run.

At 10³–10⁴ parameters (`CorrelatedFieldMap`) three more rules, measured in October–November
2026 on the production emulator. **Polish the mode before whitening**: Adam stops tens of nats
above it, and a Laplace covariance or whitening built there is wrong
(`arachne.inference.newton.trust_region_newton` + the saddle check). **The Laplace
approximation under-covers the non-mass maps** (coverage ~0.1 for Z and the SFH where NUTS gives
~0.7), so quote maps from NUTS draws. And **expect several modes**: total mass trades against the
oldest SFH bin, and Laplace, multi-start Newton and geoVI are all local, so run ≥ 4 chains from
several starts and treat a large R-hat with good within-chain ESS as a second mode, not as a
tuning problem. Never pick a mode by its Laplace evidence; it prefers unconverged ones.
