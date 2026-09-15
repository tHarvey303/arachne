# Archived: GMM parameter-blending diagnostics

These five `diagnose_*.py` scripts and the three `best_*_theta.npy` MAP vectors
were written to debug the old `examples/demo_resolved_sed_fitting.py`, which used
`GaussianMixtureSpatialModel`: per-pixel SPS parameters blended with softmax
weights summing to 1, so the "disk" was a flat sheet filling the whole frame.
They import `demo_resolved_sed_fitting` functions that no longer exist
(`near_truth_initial_theta`, `FREE_SPS_PARAM_NAMES`, the `FixedParamEmulator`
pipeline) and will not run against the current demo.

Their conclusions (blind optimisation is impossible, NUTS step sizes of 1e-8,
"structural curvature", float32 limits) were artefacts of that model, not of the
problem; the additive-flux `AdditiveComponentModel` finds the true basin blind.
Kept for the record only.
