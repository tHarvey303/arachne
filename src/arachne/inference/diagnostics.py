"""Convergence diagnostics for MCMC chains (split-R-hat, ESS, summaries).

The functions here operate on plain arrays of shape ``(n_chains, n_samples,
n_params)`` and return ``numpy`` arrays of shape ``(n_params,)``.  They are
deliberately framework-agnostic: JAX arrays are accepted and converted with
``np.asarray`` so that the diagnostics can also be applied to samples loaded
back from disk.

Two estimators are provided:

``split_rhat``
    The Gelman-Rubin potential scale reduction factor computed on
    *split* chains (each chain is cut in half, doubling the number of
    sequences), which also detects within-chain non-stationarity.  This is
    the same estimator used by ``scripts/fit_catalogue.py``.

``ess``
    The bulk effective sample size following the Stan / ArviZ recipe:
    optional rank normalisation, split chains, FFT autocovariances, and
    Geyer's initial positive / initial monotone sequence truncation of the
    autocorrelation sum.

Interpretation (Vehtari et al. 2021): run at least 4 chains, require
``rhat < 1.01`` (``< 1.05`` as a loose screen) and ``ess > 100`` per chain
before trusting posterior quantiles.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "split_rhat",
    "ess",
    "summarise_chains",
    "chain_movement",
]


def _as_chain_array(chains) -> np.ndarray:
    """Validate and convert chains to a float64 ``(n_chains, n_samples, d)`` array.

    Args:
        chains: Array-like of shape ``(n_chains, n_samples, n_params)``.  A
            2-D input is interpreted as a single chain ``(n_samples, n_params)``.

    Returns:
        ``numpy`` float64 array of shape ``(n_chains, n_samples, n_params)``.

    Raises:
        ValueError: If the input is not 2- or 3-dimensional.
    """
    arr = np.asarray(chains, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"chains must have shape (n_chains, n_samples, n_params); got {arr.shape}")
    return arr


def _split(chains: np.ndarray) -> np.ndarray:
    """Split each chain in half, returning ``(2 * n_chains, n_samples // 2, d)``."""
    n_chains, n_samples, d = chains.shape
    half = n_samples // 2
    x = chains[:, : 2 * half, :].reshape(n_chains, 2, half, d)
    return x.reshape(2 * n_chains, half, d)


def split_rhat(chains) -> np.ndarray:
    """Gelman-Rubin split-R-hat per parameter.

    Each chain is split in half so that within-chain drift inflates R-hat.
    Values are ``1`` for perfectly mixed chains and grow as the between-chain
    variance exceeds the within-chain variance.  ``rhat < 1.01`` is the modern
    convergence target; ``> 1.05`` indicates the chains disagree.

    Args:
        chains: Array of shape ``(n_chains, n_samples, n_params)`` (a 2-D
            ``(n_samples, n_params)`` input is treated as one chain).

    Returns:
        Array of shape ``(n_params,)``.  Entries are ``NaN`` where the
        within-chain variance is zero (a stuck or constant parameter) and
        ``NaN`` for every parameter when fewer than 4 samples are supplied.
    """
    x = _as_chain_array(chains)
    d = x.shape[2]
    n = x.shape[1] // 2
    if n < 2:
        return np.full((d,), np.nan, dtype=np.float64)
    x = _split(x)  # (m, n, d)
    m = x.shape[0]

    chain_mean = x.mean(axis=1)  # (m, d)
    chain_var = x.var(axis=1, ddof=1)  # (m, d)
    grand_mean = chain_mean.mean(axis=0, keepdims=True)
    b_var = n * ((chain_mean - grand_mean) ** 2).sum(axis=0) / (m - 1)
    w = chain_var.mean(axis=0)
    var_plus = (n - 1) / n * w + b_var / n
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.sqrt(var_plus / np.where(w > 0, w, np.nan))
    return np.asarray(rhat, dtype=np.float64)


def _rank_normalise(x: np.ndarray) -> np.ndarray:
    """Rank-normalise draws per parameter (pooled over chains and draws).

    Applies the inverse normal CDF to the fractional ranks with the
    Blom offset ``(r - 3/8) / (N - 1/4)``, as in ArviZ's bulk-ESS.

    Args:
        x: Array of shape ``(n_chains, n_samples, d)``.

    Returns:
        Array of the same shape holding normal scores.
    """
    from scipy.special import ndtri
    from scipy.stats import rankdata

    n_chains, n_samples, d = x.shape
    flat = x.reshape(-1, d)
    n_total = flat.shape[0]
    # Average ranks for ties, so a frozen parameter stays frozen after the
    # transform instead of being spread out into artificial variation.
    ranks = rankdata(flat, axis=0)
    z = ndtri((ranks - 0.375) / (n_total + 0.25))
    return z.reshape(n_chains, n_samples, d)


def _autocov(x: np.ndarray) -> np.ndarray:
    """Biased FFT autocovariance along the draw axis.

    Args:
        x: Array of shape ``(m, n, d)``.

    Returns:
        Array of shape ``(m, n, d)`` with ``acov[..., t, :] = (1/n) *
        sum_i (x_i - xbar)(x_{i+t} - xbar)``.
    """
    m, n, d = x.shape
    n_pad = int(2 ** np.ceil(np.log2(max(2 * n, 2))))
    centred = x - x.mean(axis=1, keepdims=True)
    freq = np.fft.rfft(centred, n=n_pad, axis=1)
    acov = np.fft.irfft(freq * np.conjugate(freq), n=n_pad, axis=1)[:, :n, :]
    return acov / n


def ess(chains, rank_normalise: bool = True) -> np.ndarray:
    """Bulk effective sample size per parameter (Stan / ArviZ convention).

    Rank-normalises the draws (optional), splits each chain in half, then
    estimates the integrated autocorrelation time ``tau`` from the pooled
    autocorrelations using Geyer's initial positive sequence followed by the
    initial monotone sequence.  ``ESS = n_chains_split * n_draws_split / tau``.

    Sanity checks: i.i.d. draws give ``ESS ~ n_total``; an AR(1) process with
    lag-1 correlation ``rho`` gives ``ESS ~ n_total (1 - rho) / (1 + rho)``.

    Args:
        chains: Array of shape ``(n_chains, n_samples, n_params)`` (a 2-D
            input is treated as a single chain).
        rank_normalise: Apply the rank normalisation of Vehtari et al. (2021)
            before estimating autocorrelations.  This is what makes the
            estimator a *bulk* ESS and keeps it finite for heavy-tailed or
            non-normal marginals.  Set ``False`` for the plain ESS of the
            raw draws.

    Returns:
        Array of shape ``(n_params,)``.  Entries are ``NaN`` where a parameter
        is constant (zero variance) and ``NaN`` for every parameter when
        fewer than 8 draws per chain are supplied.
    """
    x = _as_chain_array(chains)
    d = x.shape[2]
    if x.shape[1] // 2 < 4:
        return np.full((d,), np.nan, dtype=np.float64)
    if rank_normalise:
        # Constant parameters have no ranks to normalise; guard them below.
        with np.errstate(invalid="ignore"):
            x = _rank_normalise(x)
    x = _split(x)  # (m, n, d)
    m, n, _ = x.shape

    acov = _autocov(x)  # (m, n, d)
    acov_mean = acov.mean(axis=0)  # (n, d)
    chain_mean = x.mean(axis=1)  # (m, d)

    mean_var = acov_mean[0] * n / (n - 1.0)  # within-chain variance, ddof=1
    var_plus = mean_var * (n - 1.0) / n
    if m > 1:
        var_plus = var_plus + chain_mean.var(axis=0, ddof=1)

    constant = ~(var_plus > 0) | ~np.isfinite(var_plus)
    safe_var_plus = np.where(constant, 1.0, var_plus)

    def rho_at(t: int) -> np.ndarray:
        return 1.0 - (mean_var - acov_mean[t]) / safe_var_plus

    rho = np.zeros((n, d), dtype=np.float64)
    rho[0] = 1.0
    rho[1] = rho_at(1)

    # --- Geyer initial positive sequence (vectorised over parameters) ---
    rho_even = np.ones(d, dtype=np.float64)
    rho_odd = rho[1].copy()
    active = np.ones(d, dtype=bool)
    max_t = np.full(d, -1, dtype=np.int64)
    t = 1
    while t < n - 3:
        cont = active & ((rho_even + rho_odd) > 0.0)
        max_t[active & ~cont] = t - 2
        active = cont
        if not active.any():
            break
        new_even = rho_at(t + 1)
        new_odd = rho_at(t + 2)
        assign = active & ((new_even + new_odd) >= 0.0)
        rho[t + 1] = np.where(assign, new_even, rho[t + 1])
        rho[t + 2] = np.where(assign, new_odd, rho[t + 2])
        rho_even = np.where(active, new_even, rho_even)
        rho_odd = np.where(active, new_odd, rho_odd)
        t += 2
    max_t[active] = t - 2
    # Improve the estimate at the truncation point, as Stan does.
    trunc = np.clip(max_t + 1, 0, n - 1)
    cols = np.arange(d)
    rho[trunc, cols] = np.where(rho_even > 0.0, rho_even, rho[trunc, cols])

    # --- Geyer initial monotone sequence ---
    t = 1
    t_stop = int(max_t.max()) if d > 0 else -1
    while t <= t_stop - 2:
        sel = max_t - 2 >= t
        if sel.any():
            pair_new = rho[t + 1] + rho[t + 2]
            pair_old = rho[t - 1] + rho[t]
            fix = sel & (pair_new > pair_old)
            half = pair_old / 2.0
            rho[t + 1] = np.where(fix, half, rho[t + 1])
            rho[t + 2] = np.where(fix, half, rho[t + 2])
        t += 2

    idx = np.arange(n)[:, None]
    in_sum = idx <= max_t[None, :]
    tail = idx == np.clip(max_t + 1, 0, n - 1)[None, :]
    tau = -1.0 + 2.0 * (rho * in_sum).sum(axis=0) + (rho * tail).sum(axis=0)

    n_total = float(m * n)
    tau = np.maximum(tau, 1.0 / np.log10(max(n_total, 11.0)))
    out = n_total / tau
    return np.where(constant, np.nan, out)


def chain_movement(chains) -> float:
    """Fraction of parameters that took at least one distinct value.

    A dimension whose draws are all identical means the sampler never moved
    in that direction (a frozen or degenerate coordinate), which is the
    cheapest possible check that a run did something at all.

    Args:
        chains: Array of shape ``(n_chains, n_samples, n_params)``.

    Returns:
        Fraction in ``[0, 1]`` of dimensions with more than one unique value
        across all chains and draws.  Returns ``0.0`` for an empty array.
    """
    x = _as_chain_array(chains)
    d = x.shape[2]
    if d == 0 or x.shape[1] == 0:
        return 0.0
    flat = x.reshape(-1, d)
    moved = np.any(flat != flat[0:1, :], axis=0)
    return float(np.mean(moved))


def summarise_chains(
    chains,
    param_names: list[str] | None = None,
    quantiles: tuple[float, ...] = (0.16, 0.5, 0.84),
) -> dict:
    """Per-parameter posterior summary and convergence diagnostics.

    Args:
        chains: Array of shape ``(n_chains, n_samples, n_params)`` (a 2-D
            input is treated as a single chain).
        param_names: Optional parameter names of length ``n_params``.  When
            given, the returned dict also has a ``"names"`` entry and a
            ``"table"`` entry (a formatted, ready-to-print string).
        quantiles: Quantiles in ``[0, 1]`` to report, ordered.

    Returns:
        Dict with keys ``n_chains``, ``n_samples``, ``n_params``, ``rhat``
        ``(d,)``, ``ess`` ``(d,)``, ``mean`` ``(d,)``, ``sd`` ``(d,)``,
        ``quantile_levels``, ``quantiles`` ``(n_q, d)``,
        ``fraction_dims_moved`` (float), ``max_rhat`` and ``min_ess``
        (floats, ``NaN``-safe), plus ``names``/``table`` when
        ``param_names`` is supplied.

    Raises:
        ValueError: If ``param_names`` has the wrong length.
    """
    x = _as_chain_array(chains)
    n_chains, n_samples, d = x.shape
    if param_names is not None and len(param_names) != d:
        raise ValueError(f"param_names has length {len(param_names)}, expected {d}")

    flat = x.reshape(-1, d)
    qs = np.asarray(quantiles, dtype=np.float64)
    r = split_rhat(x)
    e = ess(x)
    out = {
        "n_chains": int(n_chains),
        "n_samples": int(n_samples),
        "n_params": int(d),
        "rhat": r,
        "ess": e,
        "mean": flat.mean(axis=0),
        "sd": flat.std(axis=0, ddof=1) if flat.shape[0] > 1 else np.zeros(d),
        "quantile_levels": qs,
        "quantiles": np.quantile(flat, qs, axis=0),
        "fraction_dims_moved": chain_movement(x),
        "max_rhat": float(np.nanmax(r)) if np.any(np.isfinite(r)) else float("nan"),
        "min_ess": float(np.nanmin(e)) if np.any(np.isfinite(e)) else float("nan"),
    }
    if param_names is not None:
        out["names"] = list(param_names)
        out["table"] = _format_table(out)
    return out


def _format_table(summary: dict) -> str:
    """Render a ``summarise_chains`` dict as a fixed-width text table."""
    names = summary["names"]
    width = max(4, max(len(n) for n in names)) if names else 4
    qlev = summary["quantile_levels"]
    header = (
        f"{'name':<{width}}  {'mean':>10}  {'sd':>10}  "
        + "  ".join(f"{f'q{q:.2f}':>10}" for q in qlev)
        + f"  {'rhat':>7}  {'ess':>9}"
    )
    lines = [header, "-" * len(header)]
    for i, name in enumerate(names):
        qvals = "  ".join(f"{summary['quantiles'][j, i]:>10.4g}" for j in range(len(qlev)))
        lines.append(
            f"{name:<{width}}  {summary['mean'][i]:>10.4g}  {summary['sd'][i]:>10.4g}  "
            f"{qvals}  {summary['rhat'][i]:>7.3f}  {summary['ess'][i]:>9.1f}"
        )
    return "\n".join(lines)
