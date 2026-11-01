"""Tests for arachne.inference.field_precond (matrix-free curvature for correlated fields).

The Fourier-space Fisher diagonal is checked against the exact Hessian diagonal from
``make_hessian_fn`` on a 16x16, 3-band toy with Gaussian PSFs of 2-3 px FWHM.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import spearmanr

from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.nuisance import NuisanceModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.field_precond import (
    DiagonalWhitenedLogDensity,
    fisher_preconditioner,
    fourier_fisher_diagonal,
)
from arachne.inference.laplace import make_hessian_fn
from arachne.inference.newton import (
    block_newton,
    dense_preconditioner,
    galaxy_block_indices,
    trust_region_newton,
)
from arachne.spatial.correlated_field import CorrelatedFieldMap

NAMES = ["redshift", "log_mass", "log10metallicity", "Av", "fesc"]
BOUNDS = {"log10metallicity": (-2.0, 0.3), "Av": (0.0, 3.0), "fesc": (0.0, 1.0)}
BANDS = ["F115W", "F200W", "F444W"]
LAM = jnp.array([1.15, 2.0, 4.44])


class ToySEDEmulator(SPSEmulator, eqx.Module):
    """Mass-linear toy: metallicity tilts the colour, Av reddens, fesc ignored."""

    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)

    @property
    def param_names(self):
        """Input names."""
        return self._param_names

    @property
    def band_names(self):
        """Band names."""
        return self._band_names

    def predict(self, params):
        """(N, 5) -> (N, 3) nJy."""
        z, logm, logz, av = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
        beta = 0.5 + 0.8 * (logz + 0.5)
        tau = av * (LAM[None, :] / (0.55 * (1.0 + z))) ** -1.0
        return 10.0 ** (logm - 6.0) * LAM[None, :] ** beta * 10.0 ** (-0.4 * tau)


@pytest.fixture(autouse=True)
def _x64():
    """Run every test in float64 and restore the previous setting."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _psf(fwhms=(2.0, 2.5, 3.0)) -> PSFModel:
    kern = []
    for fwhm in fwhms:
        s = fwhm / 2.3548
        y, x = np.mgrid[-5:6, -5:6]
        k = np.exp(-(x**2 + y**2) / (2 * s**2))
        kern.append(k / k.sum())
    return PSFModel(kernels=np.stack(kern), band_names=BANDS)


def build_toy(frac=0.0, snr=30.0, seed=0, nuisance=None, fwhms=(2.0, 2.5, 3.0), **model_kw):
    """Toy fm: compact galaxy (Gaussian mass blob, sigma 3 px), data = truth + noise."""
    H = W = 16
    yy, xx = np.mgrid[:H, :W]
    blob = np.exp(-((yy - 7.6) ** 2 + (xx - 8.3) ** 2) / (2 * 3.0**2))
    kw = dict(
        image_shape=(H, W),
        emulator_param_names=NAMES,
        param_bounds=BOUNDS,
        field_params=["log_mass", "Av"],
        fixed_params={"redshift": 2.0, "fesc": 0.0},
        corr_lengths={"log_mass": 2.0, "Av": 3.0},
        amplitudes={"log_mass": 0.3, "Av": 0.3},
        mean_maps={"log_mass": np.log10(blob + 0.02) + 7.0, "Av": 0.8},
    )
    kw.update(model_kw)
    model = CorrelatedFieldMap(**kw)
    emu = ToySEDEmulator(_param_names=NAMES, _band_names=BANDS)
    psf = _psf(fwhms)
    theta_s = model.sample_prior(jax.random.PRNGKey(seed), 1)[0]
    img = np.asarray(model.model_image(theta_s, emu, (H, W)))
    ones = np.ones_like(img)
    scratch = ObservationCube(flux=img, variance=ones, mask=ones, band_names=BANDS, pixel_scale=1)
    fm0 = ForwardModel.build(scratch, psf, model, emu, nuisance=nuisance)
    theta = fm0.initial_theta_from_spatial(theta_s)
    if nuisance is not None:
        theta = theta.at[model.n_params :].set(
            0.2 * nuisance.sample_prior(jax.random.PRNGKey(seed + 1), 1)[0]
        )
    conv = np.asarray(fm0._model_image(theta))
    sig = conv.max(axis=(1, 2), keepdims=True) / snr * ones
    flux = conv + np.random.default_rng(seed).normal(size=conv.shape) * sig
    obs = ObservationCube(flux=flux, variance=sig**2, mask=ones, band_names=BANDS, pixel_scale=1)
    fm = ForwardModel.build(obs, psf, model, emu, model_error_frac=frac, nuisance=nuisance)
    return fm, model, theta


def _compare(fm, model, theta):
    """(galaxy approx, galaxy exact, ratio, Pearson, Spearman) of likelihood-only diagonals."""
    exact = np.diag(make_hessian_fn(fm)(theta))
    approx = fourier_fisher_diagonal(fm, theta)
    gal = galaxy_block_indices(fm)
    coef = np.setdiff1d(np.arange(fm.n_params), gal)
    # Compare the likelihood parts: the +1 prior would flatter weakly constrained modes.
    a, e = approx[coef] - 1.0, exact[coef] - 1.0
    ratio = a / e
    la, le = np.log(a), np.log(np.abs(e))
    return approx[gal], exact[gal], ratio, np.corrcoef(la, le)[0, 1], spearmanr(la, le)[0]


@pytest.mark.parametrize("frac", [0.0, 0.05])
def test_matches_exact_hessian_diagonal(frac):
    """Galaxy entries exact; coefficients within ~10 % in the median, log-correlation ~1."""
    fm, model, theta = build_toy(frac=frac)
    a_gal, e_gal, ratio, pearson, spear = _compare(fm, model, theta)
    print(
        f"\nfrac={frac}: d={fm.n_params}, coefficient ratio approx/exact median "
        f"{np.median(ratio):.3f} (5-95% {np.percentile(ratio, 5):.2f}-"
        f"{np.percentile(ratio, 95):.2f}), log Pearson {pearson:.4f}, Spearman {spear:.4f}"
    )
    np.testing.assert_allclose(a_gal, e_gal, rtol=1e-5)
    assert 0.3 <= np.median(ratio) <= 3.0
    assert 0.85 <= np.median(ratio) <= 1.15  # what the derivation actually achieves
    assert pearson > 0.8 and spear > 0.8
    assert pearson > 0.99


def test_variants_and_normalisation():
    """With a near-delta PSF every variant matches the exact diagonal (normalisation check)."""
    fm, model, theta = build_toy(fwhms=(0.3, 0.3, 0.3))
    exact = np.diag(make_hessian_fn(fm)(theta))[model.z_offset :] - 1.0
    for kw in ({}, {"spectral_spread": False}, {"calibrate": False}):
        approx = fourier_fisher_diagonal(fm, theta, include_prior=False, **kw)
        ratio = approx[model.z_offset :] / exact
        assert abs(np.median(ratio) - 1.0) < 0.02, kw
        assert np.max(np.abs(np.log(ratio))) < np.log(1.1), kw
    # Without the phase term single long modes of the compact galaxy are off by a lot.
    flat = fourier_fisher_diagonal(fm, theta, include_prior=False, phase_correction=False)
    assert np.max(np.abs(np.log(flat[model.z_offset :] / exact))) > np.log(1.5)


def test_nuisance_hyperparameters_couplings_copula():
    """The general layout (nuisances, fitted hyperparameters, coupling, copula) still works."""
    nuisance = NuisanceModel(3, fit_sky=True, fit_shifts=True, fit_noise_scale=True,
                             shift_reference_band=0)  # fmt: skip
    fm, model, theta = build_toy(
        frac=0.05,
        nuisance=nuisance,
        hyper_priors={"Av": {"log_amplitude": {"loc": np.log(0.3), "scale": 0.3},
                             "log_corr_length": {"loc": np.log(3.0), "scale": 0.2}}},
        free_couplings={"Av": {"log_mass": 0.5}},
        field_marginals={"Av": {"dist": "normal", "loc": 0.8, "scale": 0.4}},
    )  # fmt: skip
    gal = galaxy_block_indices(fm)
    assert gal.size == model.z_offset + nuisance.n_params
    a_gal, e_gal, ratio, pearson, _ = _compare(fm, model, theta)
    np.testing.assert_allclose(a_gal, e_gal, rtol=1e-5)
    assert 0.5 <= np.median(ratio) <= 2.0
    assert pearson > 0.95


@pytest.mark.parametrize("snr", [30.0, 300.0])
def test_fisher_preconditioner_in_trust_region_newton(snr):
    """Both metrics reach the mode; on this toy the identity needs fewer HVPs (documented).

    The coefficients are already prior-whitened, so the identity metric has most
    eigenvalues clustered at 1; diagonal curvature scaling spreads that cluster and
    exposes the near-degeneracy of each field level with its low-k coefficients
    (whitened eigenvalues down to ~1e-6).  Measured HVPs to reach the mode: identity
    ~200-750, Fisher diagonal ~400-4300.  The identity run also checks the
    tight-residual convergence confirmation: without it, it "converged" ~10 nats
    above the mode at S/N 300.
    """
    fm, model, theta = build_toy(snr=snr)

    def nlp(t):
        return -fm.log_posterior(t)

    dense = dense_preconditioner(make_hessian_fn(fm)(theta))
    ref = trust_region_newton(nlp, theta, preconditioner=dense, tol=1e-6, verbose=False)
    assert ref.converged
    common = dict(max_iter=400, max_cg=500, tol=1e-3, verbose=False)
    res_id = trust_region_newton(nlp, theta, **common)
    M = fisher_preconditioner(fm, theta)
    assert M.kind == "diag" and M.lam.shape == (fm.n_params,) and bool(jnp.all(M.lam > 0))
    res_fi = trust_region_newton(nlp, theta, preconditioner=M, **common)
    print(
        f"\nS/N {snr}: dense ref {ref.n_hvp} HVPs; identity {res_id.reason} {res_id.n_hvp} HVPs "
        f"(+{res_id.neg_log_post - ref.neg_log_post:.2g} nat); Fisher diag {res_fi.reason} "
        f"{res_fi.n_hvp} HVPs (+{res_fi.neg_log_post - ref.neg_log_post:.2g} nat)"
    )
    for res in (res_id, res_fi):
        assert res.converged
        assert res.neg_log_post < ref.neg_log_post + 0.05


class TestDiagonalWhitening:
    """``DiagonalWhitenedLogDensity``."""

    def test_round_trip_and_log_posterior(self):
        """to_theta / from_theta invert each other and log_posterior matches fm's."""
        fm, model, theta = build_toy()
        rng = np.random.default_rng(3)
        scale = np.exp(rng.normal(size=fm.n_params))
        white = DiagonalWhitenedLogDensity(fm, theta, scale)
        assert white.n_params == fm.n_params and white.spatial_model is model
        assert white.forward_model is fm
        z = jnp.asarray(rng.normal(size=fm.n_params))
        np.testing.assert_allclose(white.from_theta(white.to_theta(z)), z, atol=1e-12)
        zs = jnp.asarray(rng.normal(size=(2, 3, fm.n_params)))
        np.testing.assert_allclose(white.from_theta(white.to_theta(zs)), zs, atol=1e-12)
        th = white.to_theta(z)
        np.testing.assert_allclose(th, theta + scale * z, rtol=1e-12)
        assert float(white.log_posterior(z)) == pytest.approx(float(fm.log_posterior(th)))
        assert float(white.log_prob(z)) == float(white.log_posterior(z))
        assert "DiagonalWhitenedLogDensity" in repr(white)

    def test_from_fisher_whitens_curvature(self):
        """With scale = 1/sqrt(Fisher diag) the whitened Hessian diagonal is ~1."""
        fm, model, theta = build_toy()
        white = DiagonalWhitenedLogDensity.from_fisher(fm, theta)
        hz = np.diag(make_hessian_fn(white)(jnp.zeros(fm.n_params)))
        assert np.all(np.isfinite(hz))
        assert 0.8 < np.median(hz) < 1.25
        np.testing.assert_allclose(hz[: model.z_offset], 1.0, rtol=1e-5)

    def test_bad_inputs_raise(self):
        """Mismatched shapes and non-positive scales are rejected."""
        fm, _, theta = build_toy()
        with pytest.raises(ValueError, match="shape|both"):
            DiagonalWhitenedLogDensity(fm, theta, np.ones(3))
        with pytest.raises(ValueError, match="positive"):
            DiagonalWhitenedLogDensity(fm, theta, np.zeros(fm.n_params))


def test_rejects_non_field_models():
    """A forward model without a correlated field is refused with a clear error."""

    class _NotAField:
        spatial_model = object()

    with pytest.raises(TypeError, match="CorrelatedFieldMap"):
        fourier_fisher_diagonal(_NotAField(), np.zeros(3))


def test_block_newton_on_toy_leaves_other_coordinates_untouched():
    """Galaxy-block Newton lowers -log p; field coefficients stay bit-identical."""
    fm, model, theta = build_toy()
    start = theta.at[: model.n_fields].add(0.1).at[model.n_fields].add(0.5)

    def nlp(t):
        return -fm.log_posterior(t)

    idx = galaxy_block_indices(model)
    np.testing.assert_array_equal(idx, np.arange(model.z_offset))
    res = block_newton(nlp, start, idx, max_iter=20, tol=1e-6)
    assert res.neg_log_post < float(nlp(start)) - 1.0
    assert res.converged, res.summary()
    np.testing.assert_array_equal(
        np.asarray(res.theta[model.z_offset :]), np.asarray(start[model.z_offset :])
    )
    assert res.n_hvp == model.z_offset * sum(h["cg_exit"] == "block" for h in res.history)
    assert "block Newton" in res.summary()
