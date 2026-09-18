"""Tests for evidence-based model comparison over the component count K.

Uses the same tiny linear-emulator + AdditiveComponentModel mock as
``tests/test_nss_sampler.py``, with deliberately small ``num_live`` and
``max_steps``: the point is that the plumbing works and returns finite
evidences, *not* that a 16x16 toy image can resolve K=1 vs K=2.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.data.observation import ObservationCube
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.model_comparison import (
    ModelComparisonRow,
    _verdict,
    bayes_factor_table,
    compare_n_components,
)
from arachne.inference.nss_sampler import NSSResult
from arachne.inference.posterior_predictive import chi2_reduced
from arachne.psf.convolution import PSFConvolver
from arachne.spatial.additive import AdditiveComponentModel

blackjax = pytest.importorskip("blackjax", reason="blackjax not installed")

N_BANDS = 3
H = 16
W = 16
BAND_NAMES = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]
SPS_PARAM_NAMES = ["log_stellar_mass", "log_age", "tau_v"]
PARAM_BOUNDS = {
    "log_stellar_mass": (6.0, 12.0),
    "log_age": (7.0, 10.1),
    "tau_v": (0.0, 4.0),
}
MASS = "log_stellar_mass"

# Small enough to finish in a few seconds per K on CPU.
SAMPLER_KWARGS = dict(
    num_live=40,
    num_inner_steps=4,
    num_delete=4,
    termination=5.0,
    n_samples_out=32,
    max_steps=25,
)


class LinearMassEmulator(SPSEmulator, eqx.Module):
    """flux_b = 10**(logM - 9) * (b + 1) * (1 + 0.1 * tau_v)."""

    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)

    @property
    def param_names(self) -> list[str]:
        """SPS parameter names."""
        return self._param_names

    @property
    def band_names(self) -> list[str]:
        """Band names."""
        return self._band_names

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Mass-linear flux with a mild tau_v colour term."""
        m9 = 10.0 ** (params[:, 0:1] - 9.0)
        colour = 1.0 + 0.1 * params[:, 2:3]
        bands = jnp.arange(1, len(self._band_names) + 1, dtype=jnp.float32)[None, :]
        return m9 * colour * bands


def _component_theta(y: float, x: float, log_mass: float) -> list[float]:
    """One component's 8 raw parameters (position, size, rho, then SPS)."""
    lo, hi = PARAM_BOUNDS[MASS]
    u = (log_mass - lo) / (hi - lo)
    return [y, x, float(np.log(2.0)), float(np.log(2.5)), 0.1, float(np.log(u / (1 - u))), 0.0, 0.0]


_CACHE: dict = {}


@pytest.fixture
def factory(gaussian_psf):
    """``(make_forward_model, observation, theta_true_k2)`` for a K=2 truth.

    ``make_forward_model(K)`` varies **only** the component count: the
    observation, PSF, emulator and parameter bounds are shared objects, which
    is exactly the prior comparability that ``compare_n_components`` documents
    as the caller's responsibility.
    """
    if "value" in _CACHE:
        return _CACHE["value"]
    emulator = LinearMassEmulator(_param_names=SPS_PARAM_NAMES, _band_names=BAND_NAMES)
    truth_model = AdditiveComponentModel(
        n_components=2,
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(H, W),
        mass_param=MASS,
    )
    theta_true = jnp.asarray(
        _component_theta(5.5, 6.0, 9.5) + _component_theta(10.0, 10.5, 9.2),
        dtype=jnp.float32,
    )
    conv = PSFConvolver(gaussian_psf, image_shape=(H, W))
    truth = np.asarray(conv(truth_model.model_image(theta_true, emulator, (H, W))))
    sigma = truth.max(axis=(1, 2)) / 10.0
    rng = np.random.default_rng(11)
    flux = (truth + rng.normal(size=truth.shape) * sigma[:, None, None]).astype(np.float32)
    variance = np.broadcast_to(sigma[:, None, None] ** 2, truth.shape).astype(np.float32)
    obs = ObservationCube(flux, variance, np.ones_like(flux), BAND_NAMES, 0.031, None)

    models = {
        k: AdditiveComponentModel(
            n_components=k,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(H, W),
            mass_param=MASS,
        )
        for k in (1, 2)
    }
    built: dict[int, ForwardModel] = {}

    def make_forward_model(k: int) -> ForwardModel:
        if k not in built:
            built[k] = ForwardModel.build(obs, gaussian_psf, models[k], emulator)
        return built[k]

    _CACHE["value"] = (make_forward_model, obs, theta_true)
    return _CACHE["value"]


@pytest.fixture
def rows(factory):
    """Comparison rows for K = 1, 2 from one (cached) pair of NSS runs."""
    if "rows" not in _CACHE:
        make_fm, _, _ = factory
        _CACHE["rows"] = compare_n_components(
            make_fm,
            (1, 2),
            jax.random.PRNGKey(0),
            sampler_kwargs=SAMPLER_KWARGS,
            n_chi2_samples=8,
        )
    return _CACHE["rows"]


class TestCompareNComponents:
    """compare_n_components end-to-end on the tiny mock."""

    def test_one_row_per_k_in_order(self, rows):
        """Rows come back in the requested order, one per K."""
        assert [r.n_components for r in rows] == [1, 2]
        assert all(isinstance(r, ModelComparisonRow) for r in rows)

    def test_logz_finite(self, rows):
        """Every model has a finite evidence and a non-negative error bar."""
        for r in rows:
            assert np.isfinite(r.logZ), f"K={r.n_components} logZ not finite"
            assert np.isfinite(r.logZ_err) and r.logZ_err >= 0.0
            assert np.isfinite(r.ess) and r.ess >= 1.0

    def test_n_params_tracks_k(self, rows, factory):
        """n_params is the fitted model's own parameter count."""
        make_fm, _, _ = factory
        for r in rows:
            assert r.n_params == make_fm(r.n_components).n_params
        assert rows[1].n_params > rows[0].n_params

    def test_chi2_and_runtime_populated(self, rows):
        """chi2_red_map / chi2_red_median are finite and runtime is positive."""
        for r in rows:
            assert np.isfinite(r.chi2_red_map) and r.chi2_red_map > 0.0
            assert np.isfinite(r.chi2_red_median) and r.chi2_red_median > 0.0
            # The best sample cannot fit worse than a typical one.
            assert r.chi2_red_map <= r.chi2_red_median * 1.001
            assert r.runtime_s > 0.0

    def test_result_is_full_nss_result(self, rows):
        """The row carries the whole NSSResult, with sampler_kwargs honoured."""
        for r in rows:
            assert isinstance(r.result, NSSResult)
            assert r.result.samples.shape == (
                SAMPLER_KWARGS["n_samples_out"],
                r.n_params,
            )
            assert r.result.n_steps >= 1

    def test_map_theta_fn_is_used(self, factory):
        """A supplied map_theta_fn sets chi2_red_map exactly."""
        make_fm, _, theta_true = factory
        fm2 = make_fm(2)
        rows2 = compare_n_components(
            make_fm,
            [2],
            jax.random.PRNGKey(1),
            sampler_kwargs=SAMPLER_KWARGS,
            map_theta_fn=lambda fm: theta_true,
            n_chi2_samples=4,
        )
        assert len(rows2) == 1
        assert rows2[0].chi2_red_map == pytest.approx(chi2_reduced(fm2, theta_true), rel=1e-5)

    def test_empty_ks_raises(self, factory):
        """An empty K list is a programming error, not an empty table."""
        make_fm, _, _ = factory
        with pytest.raises(ValueError, match="at least one"):
            compare_n_components(make_fm, [], jax.random.PRNGKey(0))

    def test_keys_differ_per_k(self, factory, monkeypatch):
        """Each K is fitted with its own PRNG key split from rng_key."""
        seen: list[tuple[int, bytes]] = []
        make_fm, _, _ = factory

        import arachne.inference.model_comparison as mc

        class FakeSampler:
            def __init__(self, fm, **kwargs):
                self.fm = fm

            def run(self, key):
                seen.append(np.asarray(jax.random.key_data(key)).tobytes())
                n = self.fm.n_params
                return NSSResult(
                    samples=jnp.zeros((3, n)),
                    infos=None,
                    spatial_model=self.fm.spatial_model,
                    logZ=-1.0,
                    logZ_err=0.1,
                    ess=3.0,
                    n_steps=1,
                    n_dead=1,
                )

        monkeypatch.setattr(mc, "NSSSampler", FakeSampler)
        mc.compare_n_components(make_fm, (1, 2), jax.random.PRNGKey(7), n_chi2_samples=2)
        assert len(seen) == 2 and seen[0] != seen[1]


def _row(k: int, logz: float, err: float = 0.1) -> ModelComparisonRow:
    """A minimal row for table-formatting tests."""
    return ModelComparisonRow(
        n_components=k,
        logZ=logz,
        logZ_err=err,
        ess=100.0,
        n_params=8 * k,
        chi2_red_map=1.0,
        chi2_red_median=1.1,
        runtime_s=1.0,
        result=None,  # type: ignore[arg-type]
    )


class TestVerdict:
    """Kass & Raftery thresholds expressed in ln K."""

    @pytest.mark.parametrize(
        "ln_k,expected",
        [
            (0.0, "inconclusive"),
            (0.99, "inconclusive"),
            (1.0, "positive"),
            (2.9, "positive"),
            (3.0, "strong"),
            (4.9, "strong"),
            (5.0, "very strong"),
            (50.0, "very strong"),
            (float("nan"), "undefined"),
        ],
    )
    def test_thresholds(self, ln_k, expected):
        """Boundaries sit at ln K = 1, 3, 5."""
        assert _verdict(ln_k) == expected


class TestBayesFactorTable:
    """bayes_factor_table formatting and reference model choice."""

    def test_best_model_has_zero_ln_k(self):
        """The highest-evidence model is the reference, flagged "best"."""
        table = bayes_factor_table([_row(1, -110.0), _row(2, -100.0), _row(3, -105.0)])
        lines = table.splitlines()
        assert "K=2" in lines[0]
        body = [ln for ln in lines if ln.strip().startswith(("1 ", "2 ", "3 "))]
        assert len(body) == 3
        assert body[1].endswith("best")
        assert "0.000" in body[1]

    def test_ln_k_values_and_verdicts(self):
        """The reported ln K is logZ_best - logZ_model, with a Jeffreys verdict."""
        table = bayes_factor_table([_row(1, -110.0), _row(2, -100.0)])
        assert "10.000" in table
        assert "very strong" in table

    def test_mc_noise_flagged(self):
        """A ln K inside its own MC error is called out as noise."""
        table = bayes_factor_table([_row(1, -100.2, err=1.0), _row(2, -100.0, err=1.0)])
        assert "inconclusive (MC noise)" in table

    def test_non_finite_logz_tolerated(self):
        """A failed run (logZ = -inf / NaN) does not break the table."""
        table = bayes_factor_table([_row(1, float("-inf")), _row(2, -100.0)])
        assert "K=2" in table.splitlines()[0]
        assert "inf" in table

    def test_all_non_finite_falls_back_to_first_row(self):
        """With no usable evidence the first row becomes the reference."""
        table = bayes_factor_table([_row(1, float("nan")), _row(2, float("nan"))])
        assert "K=1" in table.splitlines()[0]

    def test_empty_rows_raises(self):
        """An empty table is a programming error."""
        with pytest.raises(ValueError, match="at least one"):
            bayes_factor_table([])

    def test_caveat_line_present(self):
        """The prior-comparability caveat is printed with the table."""
        assert "identical priors apart from K" in bayes_factor_table([_row(1, -1.0)])


def test_package_exports():
    """Model-comparison API is re-exported at package level."""
    import arachne

    assert arachne.compare_n_components is compare_n_components
    assert arachne.bayes_factor_table is bayes_factor_table
    assert arachne.ModelComparisonRow is ModelComparisonRow
    for name in ("compare_n_components", "bayes_factor_table", "ModelComparisonRow"):
        assert name in arachne.__all__


# ---------------------------------------------------------------------------
# Multi-resolution forward models
# ---------------------------------------------------------------------------

_MULTIRES_SPEC = [
    dict(shape=(16, 16), scale=0.02, psf_sigma_px=1.5),
    dict(shape=(8, 8), scale=0.04, psf_sigma_px=1.2),
    dict(shape=(8, 8), scale=0.04, psf_sigma_px=1.5),
]


def _kernel(sigma: float) -> np.ndarray:
    """Normalised 9x9 Gaussian PSF kernel."""
    y, x = np.mgrid[-4:5, -4:5]
    k = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return (k / k.sum()).astype(np.float32)


def _multires_obs(flux=None, variance=None):
    """Three bands on two different pixel grids about one reference position."""
    from arachne.data.multires import BandImage, MultiResolutionObservation

    bands = []
    for b, cfg in enumerate(_MULTIRES_SPEC):
        h, w = cfg["shape"]
        s = cfg["scale"]
        bands.append(
            BandImage(
                band_name=BAND_NAMES[b],
                flux=np.zeros((h, w), np.float32) if flux is None else np.asarray(flux[b]),
                variance=(
                    np.ones((h, w), np.float32) if variance is None else np.asarray(variance[b])
                ),
                mask=np.ones((h, w), np.float32),
                pixel_scale=s,
                affine=np.array([[s, 0.0], [0.0, -s]]),
                ref_pixel=((h - 1) / 2.0, (w - 1) / 2.0),
                psf=_kernel(cfg["psf_sigma_px"]),
            )
        )
    return MultiResolutionObservation(bands=bands, ref_ra=53.0, ref_dec=-27.0)


def _multires_model(k: int) -> AdditiveComponentModel:
    """Arcsec-mode, analytically normalised model matching the multi-res mock."""
    return AdditiveComponentModel(
        n_components=k,
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(16, 16),
        mass_param=MASS,
        pixel_scale=0.02,
        normalisation="analytic",
    )


class TestMultiResolutionComparison:
    """``make_forward_model(K)`` may return a MultiResolutionForwardModel."""

    def test_compare_runs_and_reports_chi2(self):
        """K = 1, 2 both produce finite evidences and chi-squared columns."""
        from arachne.forward_model.multires import MultiResolutionForwardModel

        emulator = LinearMassEmulator(_param_names=SPS_PARAM_NAMES, _band_names=BAND_NAMES)
        theta_true = jnp.asarray(
            [0.02, -0.02, float(np.log(0.05)), float(np.log(0.05)), 0.0]
            + _component_theta(0.0, 0.0, 9.5)[5:],
            dtype=jnp.float32,
        )
        clean = MultiResolutionForwardModel.build(_multires_obs(), _multires_model(1), emulator)
        truth = [np.asarray(im) for im in clean.model_images(theta_true)]
        rng = np.random.default_rng(4)
        sigma = [float(t.max()) / 10.0 for t in truth]
        flux = [
            (truth[b] + rng.normal(size=truth[b].shape) * sigma[b]).astype(np.float32)
            for b in range(3)
        ]
        variance = [np.full(truth[b].shape, sigma[b] ** 2, np.float32) for b in range(3)]
        obs = _multires_obs(flux, variance)

        def make_forward_model(k: int):
            return MultiResolutionForwardModel.build(obs, _multires_model(k), emulator)

        rows = compare_n_components(
            make_forward_model, (1, 2), jax.random.PRNGKey(2), sampler_kwargs=SAMPLER_KWARGS
        )
        assert [r.n_components for r in rows] == [1, 2]
        for row in rows:
            assert np.isfinite(row.logZ)
            assert np.isfinite(row.chi2_red_map)
            assert np.isfinite(row.chi2_red_median)
            assert row.n_params == make_forward_model(row.n_components).n_params
        assert "K=" in bayes_factor_table(rows).splitlines()[0]
