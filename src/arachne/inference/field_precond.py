r"""Matrix-free curvature estimates for correlated-field fits (no dense Hessian).

A :class:`~arachne.spatial.correlated_field.CorrelatedFieldMap` fit has two kinds of
parameters with very different curvature: ~20 galaxy-level ones (field levels,
globals, hyperparameters, couplings, nuisances) at 1e6-1e8, and 1e3-1e5 white Fourier
coefficients at 1-1e4.  The dense Hessian that the Laplace step and the dense
preconditioner use costs ``d^2`` memory and is out of reach beyond ``d ~ 3e4``.  This
module estimates the **diagonal** of the Hessian of ``-log p`` instead, at the cost of
``n_galaxy + n_fields`` Hessian-vector / Jacobian-vector products plus a few FFTs:

* **Galaxy-level block**: exact diagonal entries from HVPs along those unit vectors.
* **Field coefficients**: a Gauss-Newton (Fisher) estimate in Fourier space.

Derivation (field ``k``, retained mode ``m`` at frequency ``f_m = (fy, fx)`` cycles/px
on the field's padded ``Hp x Wp`` grid, ``N = Hp Wp``, ``theta_m = 2 pi f_m . x`` with
``x`` the padded-grid pixel position).  ``unit_field`` maps a unit real (imaginary)
coefficient to ``phi_m(x) = (2 a_m / N) cos theta_m`` (``-(2 a_m / N) sin theta_m``),
``a_m`` the mode amplitude, and the field enters the pre-transform map as
``u_k += s_k phi_m``.  The emulator is evaluated pixel by pixel, so the unconvolved
flux responds as ``dF_b(x) = S_b(x) s_k phi_m(x)`` with the per-pixel sensitivity
``S_b(x) = dF_b(x) / du_k(x)``.  The field level enters ``u_k`` exactly like
``s_k phi_m`` (additively, before the softclip / copula transform, and propagated
identically through any couplings to later fields), so ``S_b`` is **exactly** the JVP
of the unconvolved model image along the level's unit vector: one JVP per field.

The Fisher information of a Gaussian likelihood with ``var_eff = v + (f mu)^2`` is
``sum_b sum_x w_b(x) [dmu_b(x)]^2`` with ``w_b = mask (1 / var_eff + 2 f^4 mu^2 /
var_eff^2)`` (the second term is the information carried by the model-dependent
variance; it is zero without a model-error floor).  The convolved response to
``e^{i theta_m}`` is ``P_b * (S_b e^{i theta_m})`` and, since ``[Re z]^2 = |z|^2 / 2 +
Re(z^2) / 2``::

    F_m(re/im) = s_k^2 (2 a_m^2 / N^2) sum_b [ T_b(f_m)  +/-  O_b(f_m) ]
    T_b(f) = sum_x w_b |P_b * (S_b e^{2 pi i f.x})|^2         ("power" term)
    O_b(f) = Re sum_x w_b (P_b * (S_b e^{2 pi i f.x}))^2      ("phase" term)

Three approximations make these cheap:

1. **Power term, spectral spread** (``spectral_spread=True``).  With ``X_b = sqrt(w_b)
   S_b`` (``w_b`` assumed smooth on the PSF scale), Parseval gives ``T_b(f) = sum_y
   C_b(y) R_b(y) e^{-2 pi i f.y}``, ``C_b`` the PSF autocorrelation
   (``irfft2(|psf_fft|^2)``) and ``R_b`` the autocorrelation of ``X_b``: exact for
   uniform weights and an unbounded frame, one FFT per band on a doubled padded grid
   (where every ``f_m`` is a grid frequency).  The cruder ``|M_b(f_m)|^2 sum_x W_b``
   (``spectral_spread=False``; ``M_b`` the PSF transform bilinearly interpolated onto
   ``f_m``, ``W_b = w_b S_b^2``) ignores the spread of ``S_b``'s spectrum under a
   steep MTF and underestimates high-``k`` modes by up to ~5x.
2. **k -> 0 calibration** (``calibrate=True``).  For ``f -> 0`` the power term is exactly
   ``sum_x w_b (P_b * S_b)^2`` (one convolution per field); ``T_b`` is rescaled per band
   and field to match it, which absorbs most of the error from ``w_b`` varying on the
   PSF scale (strong with a model-error floor, where ``w ~ 1 / mu^2``).
3. **Phase term, stationary** (``phase_correction=True``).  ``O_b(f) ~ Re(M_b(f)^2
   conj(W^_b(2 f)))`` with ``W^_b(q) = sum_x W_b(x) e^{-2 pi i q.x}`` read off one FFT
   of ``W_b`` on the padded grid (shifts add their phase ramp to ``M_b``).  It is what
   separates a real coefficient from its imaginary partner: for the longest modes of a
   compact galaxy one of them peaks on the galaxy (twice the average information) and
   the other is nearly blind.  ``phase_correction=False`` gives both ``T_b`` (``cos^2``
   and ``sin^2`` averaged to 1/2), which is right only for modes much shorter than
   the galaxy.

The prior adds 1 per coefficient (``include_prior``).  What is neglected: off-diagonal
curvature (irrelevant for a diagonal), frame-edge truncation of the convolution, the
variation of ``w_b`` across a PSF width beyond the k -> 0 calibration, and the
residual-weighted second-derivative terms of the exact Hessian (zero in expectation
at the true parameters, small at a good MAP).  Measured agreement with the exact
Hessian diagonal is quoted in :func:`fourier_fisher_diagonal`.

:func:`fisher_preconditioner` packages the estimate for
:func:`~arachne.inference.newton.trust_region_newton`, and
:class:`DiagonalWhitenedLogDensity` uses it as a diagonal whitening ``theta = mean +
scale z`` (``scale = 1 / sqrt(diag)``) for NUTS / MCLMC at ``d ~ 1e5``, where the
dense Cholesky of :class:`~arachne.inference.laplace.WhitenedLogDensity` cannot be
formed.

**Caveat: a good diagonal is not a good preconditioner here.**  The coefficients are
already prior-whitened (``N(0, I)``), so in raw theta most Hessian eigenvalues sit in
a cluster at 1 with a few stiff outliers, which CG resolves cheaply.  Any diagonal or
block-diagonal rescaling (measured: this estimate, the *exact* Jacobi diagonal, a
dense galaxy block plus per-frequency cross-field blocks, and ``diag^alpha`` for
``alpha`` = 0.1-0.5) spreads that cluster and exposes the near-degeneracy of each
field level with its own low-``k`` coefficients over a compact galaxy (level/
coefficient Hessian correlation up to 0.993): whitened eigenvalues drop to 1e-4-1e-6.
On the 16x16 toy, trust-region Newton therefore needs 2-6x *more* HVPs with
:func:`fisher_preconditioner` than with the identity (S/N 300: ~4300 vs ~750 to the
mode), and the diagonally whitened Hessian has a condition number of ~3e7 against
~5e7 for raw theta.  Use the diagonal for per-coefficient curvature / conditional
widths, for step-size scales and as a starting metric to be adapted; for the stiff
galaxy-level block use :func:`~arachne.inference.newton.block_newton`.  A
preconditioner that actually helps has to keep the prior cluster (identity on the
coefficients) and remove the level / low-``k`` coupling, e.g. a reparametrisation
that makes the low-``k`` modes orthogonal to the level over the galaxy, or a
low-rank (Lanczos) correction.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from arachne.inference.laplace import _model_dtype
from arachne.inference.newton import Preconditioner, _block_hessian_fn, galaxy_block_indices

__all__ = [
    "DiagonalWhitenedLogDensity",
    "fisher_preconditioner",
    "fourier_fisher_diagonal",
]


def _check_forward_model(fm: Any) -> Any:
    """Return ``fm.spatial_model`` after checking ``fm`` is a single-grid field model."""
    spatial = getattr(fm, "spatial_model", None)
    if spatial is None or not hasattr(spatial, "z_offset") or not hasattr(spatial, "fields"):
        raise TypeError("fourier_fisher_diagonal needs a ForwardModel with a CorrelatedFieldMap")
    conv = getattr(fm, "convolver", None)
    if conv is None or not hasattr(conv, "psf_ffts"):
        raise TypeError("fourier_fisher_diagonal needs a ForwardModel with a PSFConvolver")
    return spatial


def _interp_psf_fft(psf_ffts: np.ndarray, Wc: int, fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of rfft2 PSF spectra ``(B, Hc, Wc//2+1)`` at ``(fy, fx)``.

    ``fy`` (any sign) indexes the full ``fftfreq(Hc)`` axis periodically, so no
    symmetry of the PSF is assumed; ``fx >= 0`` indexes ``rfftfreq(Wc)``.

    Returns:
        Complex ``(B, n)`` array.
    """
    _, Hc, nx = psf_ffts.shape
    ry = np.asarray(fy, np.float64) * Hc
    iy0 = np.floor(ry).astype(np.int64)
    ty = ry - iy0
    rx = np.clip(np.asarray(fx, np.float64) * Wc, 0.0, nx - 1)
    ix0 = np.minimum(np.floor(rx).astype(np.int64), nx - 1)
    tx = rx - ix0
    ix1 = np.minimum(ix0 + 1, nx - 1)
    y0, y1 = iy0 % Hc, (iy0 + 1) % Hc
    return (
        (1 - ty) * (1 - tx) * psf_ffts[:, y0, ix0]
        + (1 - ty) * tx * psf_ffts[:, y0, ix1]
        + ty * (1 - tx) * psf_ffts[:, y1, ix0]
        + ty * tx * psf_ffts[:, y1, ix1]
    )


def _pixel_weights(fm: Any, theta: jnp.ndarray) -> np.ndarray:
    """Per-pixel Fisher weights ``w_b(x)`` ``(B, H, W)`` of the Gaussian likelihood at theta."""
    lik = fm.likelihood
    obs = lik.obs
    _, theta_n = fm.split_theta(theta)
    nuis = fm._nuisance_blocks(theta_n)
    mu = np.asarray(fm._model_image(theta), np.float64)
    var = np.asarray(obs.variance, np.float64)
    mask = np.asarray(obs.mask, np.float64)
    if nuis.get("log_noise_scale") is not None:
        var = var * np.exp(2.0 * np.asarray(nuis["log_noise_scale"], np.float64))[:, None, None]
    frac = np.asarray(lik.model_error_frac, np.float64)
    frac = frac[:, None, None] if frac.ndim == 1 else frac
    fm2 = (frac * mu) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        var_eff = var + fm2 + 1e-30
        w = mask * (1.0 / var_eff + 2.0 * (frac**2 * mu) ** 2 / var_eff**2)
    return np.where(mask > 0, w, 0.0)


def _psf_autocorrelation(psf_ffts: np.ndarray, conv_shape: tuple, image_shape: tuple) -> dict:
    """PSF autocorrelation ``C_b(y) = sum_u p_b(u) p_b(u + y)`` for ``|y| < (H, W)``.

    Computed as ``irfft2(|psf_fft|^2)`` on the convolver's grid, which is at least
    ``H + kh - 1`` wide, so the lags needed here (``|y| < H``) never alias.

    Returns:
        ``{"c": (B, 2H-1, 2W-1) array, "lag_y": (2H-1,), "lag_x": (2W-1,)}``.
    """
    Hc, Wc = conv_shape
    H, W = image_shape
    c_full = np.fft.irfft2(np.abs(psf_ffts) ** 2, s=(Hc, Wc))  # lag 0 at [0, 0]
    lag_y, lag_x = np.arange(-(H - 1), H), np.arange(-(W - 1), W)
    c = c_full[:, (lag_y % Hc)[:, None], (lag_x % Wc)[None, :]]
    return {"c": c, "lag_y": lag_y, "lag_x": lag_x}


def _spread_sum(psf_ac: dict, x: np.ndarray, grid: tuple, iy, ix) -> tuple:
    """``sum_b sum_x |P_b * (x_b e^{i theta_m})|^2`` for each mode, via autocorrelations.

    By Parseval, ``sum_x |P * (X e^{i k.x})|^2 = sum_y C(y) R(y) e^{-i k.y}`` with ``C``
    the PSF and ``R`` the ``X`` autocorrelation: exact for an unbounded frame, and it
    keeps the spread of ``X``'s spectrum under a steep MTF that ``|M(k)|^2 sum X^2``
    misses.  Evaluated by one FFT per band on a ``(2 Hp, 2 Wp)`` grid, where both the
    lags (``|y| < H``) fit without wrapping and every mode frequency ``f_m`` is a grid
    frequency (index ``2 iy``, ``2 ix``).
    """
    Hp, Wp = grid
    B, H, W = x.shape
    Hb, Wb = 2 * Hp, 2 * Wp
    big = np.zeros((B, Hb, Wb))
    big[:, :H, :W] = x  # the frame offset ``pad`` is a pure phase and drops out of R
    spec = np.fft.rfft2(big)
    r = np.fft.irfft2(np.abs(spec) ** 2, s=(Hb, Wb))  # (B, Hb, Wb), lag 0 at [0, 0]
    ly, lx = psf_ac["lag_y"] % Hb, psf_ac["lag_x"] % Wb
    prod = np.zeros((B, Hb, Wb))
    prod[:, ly[:, None], lx[None, :]] = psf_ac["c"] * r[:, ly[:, None], lx[None, :]]
    full = np.fft.fft2(prod)
    f_hat = full[:, (2 * np.asarray(iy)) % Hb, (2 * np.asarray(ix)) % Wb].real  # (B, n_modes)
    return f_hat, full[:, 0, 0].real


def fourier_fisher_diagonal(
    fm: Any,
    theta,
    *,
    include_prior: bool = True,
    phase_correction: bool = True,
    spectral_spread: bool = True,
    calibrate: bool = True,
    chunk_size: int | None = 32,
) -> np.ndarray:
    """Approximate ``diag`` of the Hessian of ``-log_posterior`` without a dense matrix.

    See the module docstring for the derivation.  Galaxy-level entries (indices of
    :func:`~arachne.inference.newton.galaxy_block_indices`: everything before the field
    coefficients, plus any nuisance parameters) are exact; field-coefficient entries
    are the Fourier-space Fisher estimate.  Cost: ``n_galaxy`` HVPs (vmapped,
    ``chunk_size`` at a time), ``n_fields`` JVPs of the unconvolved model image, and
    ``O(n_bands * n_fields)`` FFTs of the padded grid (numpy, outside jit); memory
    ``O(d + n_fields * n_bands * H * W)``, no ``d x d`` matrix.

    Measured against the exact Hessian diagonal (``make_hessian_fn``) on 16x16 toys
    (mass + Av fields, one global, 3 bands, Gaussian PSFs of 2/2.5/3 px FWHM, noisy
    data at the truth, float64; ``include_prior=False`` so the +1 does not flatter
    the comparison): galaxy-level entries agree to 1e-15.  Field coefficients, ratio
    approx/exact (median; 5-95 %; log-log Pearson):

    * no floor, peak S/N 30-300, 99-99.9 % of the prior variance kept: 1.00-1.02;
      0.97-1.21; >= 0.9998.
    * 5 % model-error floor: 0.92-0.96; 0.76-1.0; >= 0.9997.
    * 10 % floor at peak S/N 300, 99.9 % kept: 0.75; 0.24-0.99; 0.995 (high-``k``
      modes far below their prior curvature, where ``w ~ 1 / mu^2`` varies fastest).
    * Without ``spectral_spread`` the medians drop to 0.47-0.95 (5 % quantile down to
      0.07); without ``phase_correction`` single long modes are off by up to 3.4x.

    Args:
        fm: :class:`~arachne.forward_model.pipeline.ForwardModel` whose
            ``spatial_model`` is a ``CorrelatedFieldMap`` and whose ``convolver`` is a
            ``PSFConvolver``.
        theta: Full parameter vector ``(d,)`` (spatial + nuisance).
        include_prior: Include the prior curvature: exact for the galaxy block
            (HVPs of ``-log_posterior`` instead of ``-log_likelihood``), ``+1`` for
            every coefficient.
        phase_correction: Keep the phase term that tells a real coefficient from its
            imaginary partner (module docstring, item 3).
        spectral_spread: Power term from the PSF and sensitivity autocorrelations
            (item 1) instead of ``|M(f)|^2 sum_x W``.
        calibrate: Rescale the power term to its exact ``k -> 0`` limit (item 2;
            only with ``spectral_spread``).
        chunk_size: HVPs per vmapped batch for the galaxy block.

    Returns:
        ``numpy`` float64 array of shape ``(d,)``.

    Raises:
        TypeError: If ``fm`` is not a single-grid forward model of a correlated field.
    """
    spatial = _check_forward_model(fm)
    dtype = _model_dtype(fm)
    theta = jnp.asarray(theta, dtype)
    d = int(theta.size)
    out = np.zeros(d, np.float64)

    # Galaxy-level block (and nuisances): exact diagonal entries.
    gal = galaxy_block_indices(fm)
    if include_prior:
        target = lambda t: -fm.log_posterior(t)  # noqa: E731
    else:
        target = lambda t: -fm.log_likelihood(t)  # noqa: E731
    if gal.size:
        out[gal] = np.diag(_block_hessian_fn(target, gal, chunk_size)(theta))

    # Per-pixel sensitivity to each field's pre-transform value: JVP along its level.
    theta_s, theta_n = fm.split_theta(theta)
    H, W = spatial.image_shape
    n_fields = spatial.n_fields
    emulator = fm.emulator

    def image(ts):
        return spatial.model_image(ts, emulator, (H, W))

    tangents = (
        jnp.zeros((n_fields, theta_s.size), dtype)
        .at[jnp.arange(n_fields), jnp.arange(n_fields)]
        .set(1.0)
    )
    sens = jax.jit(jax.vmap(lambda v: jax.jvp(image, (theta_s,), (v,))[1]))(tangents)
    if spectral_spread and calibrate:
        sens_conv = np.asarray(jax.vmap(fm.convolver)(sens), np.float64)
    sens = np.asarray(sens, np.float64)  # (n_fields, B, H, W)
    weights = _pixel_weights(fm, theta)  # (B, H, W)

    conv = fm.convolver
    psf_ffts = np.asarray(conv.psf_ffts)
    Wc = int(conv.padded_shape[1])
    shifts = fm._nuisance_blocks(theta_n).get("shifts") if hasattr(fm, "_nuisance_blocks") else None
    Hp, Wp = spatial.grid_shape
    n_pad = float(Hp * Wp)
    P = int(spatial.pad)
    fy_grid, fx_grid = np.fft.fftfreq(Hp), np.fft.rfftfreq(Wp)
    if spectral_spread:
        psf_autocorr = _psf_autocorrelation(psf_ffts, tuple(conv.padded_shape), (H, W))
    for k, f in enumerate(spatial.fields):
        if f.n_modes == 0:
            continue
        s_k, ell, nu = spatial.field_hyperparameters(theta_s, k)
        amp = np.asarray(spatial._mode_amplitudes(k, ell, nu, dtype), np.float64)
        s_k = float(s_k)
        iy, ix = np.asarray(f.iy), np.asarray(f.ix)
        fy, fx = fy_grid[iy], fx_grid[ix]
        mtf = _interp_psf_fft(psf_ffts, Wc, fy, fx)  # (B, n_modes)
        if shifts is not None:
            sh = np.asarray(shifts, np.float64)
            mtf = mtf * np.exp(-2j * np.pi * (sh[:, 0:1] * fy[None] + sh[:, 1:2] * fx[None]))
        wmap = weights * sens[k] ** 2  # (B, H, W)
        if spectral_spread:
            per_band, at_zero = _spread_sum(
                psf_autocorr, np.sqrt(weights) * sens[k], (Hp, Wp), iy, ix
            )
            if calibrate:
                exact0 = np.sum(weights * sens_conv[k] ** 2, axis=(1, 2))
                per_band = per_band * (exact0 / np.where(at_zero > 0, at_zero, 1.0))[:, None]
            base = per_band.sum(axis=0)
        else:
            base = np.sum(np.abs(mtf) ** 2 * wmap.sum(axis=(1, 2))[:, None], axis=0)
        if phase_correction:
            padded = np.zeros((wmap.shape[0], Hp, Wp))
            padded[:, P : P + H, P : P + W] = wmap
            w_hat = np.fft.fft2(padded)[:, (2 * iy) % Hp, (2 * ix) % Wp]  # (B, n_modes)
            osc = np.sum(np.real(mtf**2 * np.conj(w_hat)), axis=0)
        else:
            osc = 0.0
        pref = s_k**2 * 2.0 * amp**2 / n_pad**2
        z0 = spatial.z_offset + f.slice.start
        out[z0 : z0 + f.n_modes] = pref * (base + osc)
        out[z0 + f.n_modes : z0 + 2 * f.n_modes] = pref * (base - osc)
        if include_prior:
            out[z0 : z0 + 2 * f.n_modes] += 1.0
    return out


def fisher_preconditioner(fm: Any, theta, rel_floor: float = 1e-3, **kwargs) -> Preconditioner:
    """Diagonal ``Preconditioner`` built from :func:`fourier_fisher_diagonal`.

    ``|diag|`` is floored at ``rel_floor`` times its median (the exact galaxy-level
    entries may be negative away from a mode).  Read the module docstring's caveat
    first: on correlated-field posteriors this has been measured to be *slower* than
    the identity metric.  Pass it to
    :func:`~arachne.inference.newton.trust_region_newton` as ``preconditioner`` (and
    ``refresh_preconditioner=lambda t: fisher_preconditioner(fm, t)`` to rebuild it as
    the iterate moves); it needs no dense Hessian, so it scales to ``d ~ 1e5``.

    Args:
        fm: See :func:`fourier_fisher_diagonal`.
        theta: Point to build it at.
        rel_floor: Floor as a fraction of the median.
        **kwargs: Passed to :func:`fourier_fisher_diagonal`.

    Returns:
        Preconditioner of kind ``"diag"`` in the forward model's dtype.
    """
    diag = np.abs(fourier_fisher_diagonal(fm, theta, **kwargs))
    lam = np.maximum(diag, rel_floor * float(np.median(diag)))
    return Preconditioner("diag", jnp.asarray(lam, _model_dtype(fm)))


class DiagonalWhitenedLogDensity:
    """``log_posterior`` in diagonally whitened coordinates ``theta = mean + scale * z``.

    Same duck-type interface as :class:`~arachne.inference.laplace.WhitenedLogDensity`
    (``log_posterior``, ``log_prob``, ``to_theta``, ``from_theta``, ``n_params``,
    ``spatial_model``, ``forward_model``, ``mean``), so the NUTS / MCLMC samplers and
    ``run_whitened_nuts(whitened=...)`` accept it, but it stores only a ``(d,)``
    scale: no ``d x d`` Cholesky, so it works at ``d ~ 1e5``.  The natural choice is
    ``scale = 1 / sqrt(fourier_fisher_diagonal(fm, mode))``.  The constant Jacobian
    ``sum log scale`` is dropped, as in ``WhitenedLogDensity``.

    A diagonal whitening removes the scale mismatch between parameter blocks but not
    the correlations between them (e.g. a level against the low-``k`` coefficients of
    the same field), and with ``scale = 1 / sqrt(Fisher diag)`` it turns those into
    near-flat directions (module docstring caveat); adapt a metric inside the
    whitened space and check R-hat / ESS.

    Attributes:
        forward_model: The wrapped model.
        mean: ``(d,)`` centre (usually the MAP).
        scale: ``(d,)`` positive scales.
        n_params: ``d``.
        spatial_model: Passed through from the wrapped model.
    """

    def __init__(self, fm: Any, mean, scale) -> None:
        """Build the whitened density.

        Args:
            fm: Object exposing ``log_posterior(theta) -> scalar``.
            mean: ``(d,)`` centre.
            scale: ``(d,)`` positive per-coordinate scales.

        Raises:
            ValueError: If the shapes disagree or a scale is not positive and finite.
        """
        dtype = _model_dtype(fm)
        mean_np = np.asarray(mean, np.float64)
        scale_np = np.asarray(scale, np.float64)
        if mean_np.ndim != 1 or scale_np.shape != mean_np.shape:
            raise ValueError(
                f"mean and scale must both be (d,); got {mean_np.shape} and {scale_np.shape}"
            )
        if not np.all(np.isfinite(scale_np) & (scale_np > 0)):
            raise ValueError("scale must be positive and finite")
        self.forward_model = fm
        self.mean = jnp.asarray(mean_np, dtype)
        self.scale = jnp.asarray(scale_np, dtype)
        self.n_params = int(mean_np.size)
        self.spatial_model = getattr(fm, "spatial_model", None)

    @classmethod
    def from_fisher(cls, fm: Any, mean, **kwargs) -> DiagonalWhitenedLogDensity:
        """Whiten with ``scale = 1 / sqrt(fisher_preconditioner(fm, mean).lam)``.

        Args:
            fm: See :func:`fourier_fisher_diagonal`.
            mean: Centre (and the point where the curvature is estimated).
            **kwargs: Passed to :func:`fisher_preconditioner`.

        Returns:
            The whitened density.
        """
        lam = np.asarray(fisher_preconditioner(fm, mean, **kwargs).lam, np.float64)
        return cls(fm, mean, 1.0 / np.sqrt(lam))

    def to_theta(self, z: jnp.ndarray) -> jnp.ndarray:
        """Map whitened ``z`` (``(d,)`` or ``(..., d)``) to ``theta``."""
        return self.mean + self.scale * jnp.asarray(z)

    def from_theta(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Map ``theta`` (``(d,)`` or ``(..., d)``) to whitened ``z``."""
        return (jnp.asarray(theta) - self.mean) / self.scale

    def log_posterior(self, z: jnp.ndarray) -> jnp.ndarray:
        """Log posterior at whitened ``z`` (up to the constant ``sum log scale``)."""
        return self.forward_model.log_posterior(self.to_theta(z))

    #: Alias, for callers that think in terms of a log density.
    log_prob = log_posterior

    def __repr__(self) -> str:
        """One-line representation naming the wrapped model and the dimension."""
        inner = type(self.forward_model).__name__
        return f"DiagonalWhitenedLogDensity(n_params={self.n_params}, model={inner})"
