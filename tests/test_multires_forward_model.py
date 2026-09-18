"""Tests for the optional WCS-level multi-resolution forward model.

Three bands share one tiny mock: band 0 is 32x32 at 0.02 arcsec/px and bands 1
and 2 are 16x16 at 0.04 arcsec/px, all covering the same 0.64 arcsec field
about one reference position, with different PSF widths and (for band 1) a
rotated WCS.  Everything runs on CPU in a few seconds.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.data.multires import BandImage, MultiResolutionObservation
from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.multires import MultiResolutionForwardModel
from arachne.forward_model.nuisance import NuisanceModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.initialisation import (
    blind_initial_full_theta,
    blind_initial_theta,
    multistart_map,
    reference_band_index,
)
from arachne.inference.posterior_predictive import (
    chi2_reduced,
    component_image_samples,
    model_image_samples,
    residual_summary,
)
from arachne.spatial.additive import AdditiveComponentModel

BAND_NAMES = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]
SPS_PARAM_NAMES = ["log_stellar_mass", "log_age", "tau_v"]
PARAM_BOUNDS = {
    "log_stellar_mass": (6.0, 12.0),
    "log_age": (7.0, 10.1),
    "tau_v": (0.0, 4.0),
}
MASS = "log_stellar_mass"

#: Band specification of the genuine multi-resolution mock: a 0.02 arcsec/px
#: band and two 0.04 arcsec/px bands, all spanning the same 0.64 arcsec field.
BAND_SPEC = [
    dict(shape=(32, 32), scale=0.02, rotation=0.0, psf_sigma_px=1.5),
    dict(shape=(16, 16), scale=0.04, rotation=30.0, psf_sigma_px=1.2),
    dict(shape=(16, 16), scale=0.04, rotation=0.0, psf_sigma_px=1.5),
]
TRUE_LOG_M = (9.3, 9.8)
TRUE_MU = ((0.05, -0.04), (-0.03, 0.06))  # sky-frame (dy North, dx East), arcsec


class ColourLinearEmulator(SPSEmulator, eqx.Module):
    """flux_b = 10**(logM - 9) * (b + 1) * exp(-0.4 * tau_v * (2 - b)).

    Exactly linear in ``10**logM`` (so the linear mass solve is exact) and
    colour-carrying through ``tau_v``; ``log_age`` has no effect.
    """

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
        """Mass-linear flux with a band-dependent attenuation term."""
        m9 = 10.0 ** (params[:, 0:1] - 9.0)
        b = jnp.arange(len(self._band_names), dtype=jnp.float32)[None, :]
        atten = jnp.exp(-0.4 * params[:, 2:3] * (2.0 - b))
        return m9 * (b + 1.0) * atten


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emulator() -> ColourLinearEmulator:
    return ColourLinearEmulator(_param_names=SPS_PARAM_NAMES, _band_names=BAND_NAMES)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    r = size // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    k = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return (k / k.sum()).astype(np.float32)


def _affine(rotation_deg: float, scale: float) -> np.ndarray:
    """North-up/East-left affine (``dx`` flips sign), rotated by ``rotation_deg``."""
    t = np.deg2rad(rotation_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return rot @ np.array([[scale, 0.0], [0.0, -scale]])


def _raw(value: float, lo: float, hi: float) -> float:
    u = (value - lo) / (hi - lo)
    return float(np.log(u) - np.log1p(-u))


def _block(mu_y, mu_x, sy, sx, rho, log_m, log_age, tau_v) -> list[float]:
    """One Gaussian component block: 5 shape entries then 3 SPS raws."""
    return [
        mu_y,
        mu_x,
        float(np.log(sy)),
        float(np.log(sx)),
        float(np.arctanh(rho)),
        _raw(log_m, *PARAM_BOUNDS[MASS]),
        _raw(log_age, *PARAM_BOUNDS["log_age"]),
        _raw(tau_v, *PARAM_BOUNDS["tau_v"]),
    ]


def _multires_model(k: int = 2) -> AdditiveComponentModel:
    """Arcsec-mode, analytically normalised additive model for the mock field."""
    return AdditiveComponentModel(
        n_components=k,
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(32, 32),
        mass_param=MASS,
        pixel_scale=0.02,
        normalisation="analytic",
    )


def _multires_observation(flux=None, variance=None, psfs=True) -> MultiResolutionObservation:
    """Build the three-band mock observation, optionally with data attached."""
    bands = []
    for b, cfg in enumerate(BAND_SPEC):
        h, w = cfg["shape"]
        bands.append(
            BandImage(
                band_name=BAND_NAMES[b],
                flux=np.zeros((h, w), np.float32) if flux is None else np.asarray(flux[b]),
                variance=(
                    np.ones((h, w), np.float32) if variance is None else np.asarray(variance[b])
                ),
                mask=np.ones((h, w), np.float32),
                pixel_scale=cfg["scale"],
                affine=_affine(cfg["rotation"], cfg["scale"]),
                ref_pixel=((h - 1) / 2.0, (w - 1) / 2.0),
                psf=_gaussian_kernel(9, cfg["psf_sigma_px"]) if psfs else None,
            )
        )
    return MultiResolutionObservation(bands=bands, ref_ra=53.0, ref_dec=-27.0)


_MOCK: dict = {}


@pytest.fixture
def multires_mock():
    """``(fm, theta_true, noiseless_images)`` for the genuine multi-resolution mock.

    The truth is rendered with the model itself on the three native grids, then
    Gaussian noise at peak S/N ~ 30 per band is added and a fresh forward model
    is built on the noisy data.  Cached so every test shares one set of
    XLA-compiled kernels.
    """
    if "value" in _MOCK:
        return _MOCK["value"]
    model = _multires_model(2)
    emulator = _emulator()
    theta_true = jnp.asarray(
        _block(*TRUE_MU[0], 0.05, 0.05, 0.0, TRUE_LOG_M[0], 8.5, 0.5)
        + _block(*TRUE_MU[1], 0.12, 0.09, 0.2, TRUE_LOG_M[1], 9.5, 2.5),
        dtype=jnp.float32,
    )
    fm_clean = MultiResolutionForwardModel.build(_multires_observation(), model, emulator)
    truth = [np.asarray(im) for im in fm_clean.model_images(theta_true)]
    rng = np.random.default_rng(11)
    sigma = [float(t.max()) / 30.0 for t in truth]
    flux = [
        (truth[b] + rng.normal(size=truth[b].shape) * sigma[b]).astype(np.float32)
        for b in range(len(truth))
    ]
    variance = [np.full(truth[b].shape, sigma[b] ** 2, np.float32) for b in range(len(truth))]
    fm = MultiResolutionForwardModel.build(_multires_observation(flux, variance), model, emulator)
    _MOCK["value"] = (fm, theta_true, truth)
    return _MOCK["value"]


# ---------------------------------------------------------------------------
# 1. Equivalence with the single-grid ForwardModel
# ---------------------------------------------------------------------------


class TestSingleGridEquivalence:
    """A degenerate multi-resolution model reproduces ForwardModel exactly."""

    @staticmethod
    def _pair(nuisance: NuisanceModel | None):
        """Same 2-component mock as ForwardModel and as MultiResolutionForwardModel."""
        h = w = 16
        scale = 0.03
        emulator = _emulator()
        model = AdditiveComponentModel(
            n_components=2,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(h, w),
            mass_param=MASS,
            pixel_scale=scale,
            normalisation="analytic",
        )
        kernels = np.stack([_gaussian_kernel(9, s) for s in (1.0, 1.2, 1.5)])
        psf_model = PSFModel(kernels=kernels, band_names=BAND_NAMES)

        rng = np.random.default_rng(5)
        flux = rng.normal(5.0, 1.0, (3, h, w)).astype(np.float32)
        variance = np.full((3, h, w), 1.0, np.float32)
        mask = np.ones((3, h, w), np.float32)
        cube = ObservationCube(flux, variance, mask, BAND_NAMES, scale, None)
        fm_single = ForwardModel.build(
            cube, psf_model, model, emulator, model_error_frac=0.05, nuisance=nuisance
        )

        # affine = diag(scale, scale) and ref_pixel at the frame centre make the
        # sky frame coincide with the model's own pixel-centred arcsec frame.
        affine = np.array([[scale, 0.0], [0.0, scale]])
        ref_pixel = ((h - 1) / 2.0, (w - 1) / 2.0)
        bands = [
            BandImage(
                band_name=BAND_NAMES[b],
                flux=flux[b],
                variance=variance[b],
                mask=mask[b],
                pixel_scale=scale,
                affine=affine,
                ref_pixel=ref_pixel,
                psf=kernels[b],
            )
            for b in range(3)
        ]
        fm_multi = MultiResolutionForwardModel.build(
            MultiResolutionObservation(bands=bands, ref_ra=53.0, ref_dec=-27.0),
            model,
            emulator,
            model_error_frac=0.05,
            nuisance=nuisance,
        )
        return fm_single, fm_multi, model

    def test_log_probabilities_agree(self):
        """log_likelihood / log_prior / log_posterior match to rtol 1e-4."""
        nuisance = NuisanceModel(3, fit_sky=True, fit_shifts=True)
        fm_single, fm_multi, _ = self._pair(nuisance)
        assert fm_multi.n_params == fm_single.n_params
        assert fm_multi.band_names == fm_single.observation.band_names
        assert fm_multi.band_indices == [0, 1, 2]

        thetas = fm_single.sample_prior(jax.random.PRNGKey(3), 5)
        for theta in thetas:
            for name in ("log_likelihood", "log_prior", "log_posterior"):
                a = float(getattr(fm_single, name)(theta))
                b = float(getattr(fm_multi, name)(theta))
                np.testing.assert_allclose(b, a, rtol=1e-4, err_msg=name)

    def test_model_images_match_per_band(self):
        """model_images equals _model_image band by band, sky and shifts included."""
        nuisance = NuisanceModel(3, fit_sky=True, fit_shifts=True)
        fm_single, fm_multi, _ = self._pair(nuisance)
        theta = fm_single.sample_prior(jax.random.PRNGKey(17), 1)[0]
        single = np.asarray(fm_single._model_image(theta))
        multi = fm_multi.model_images(theta)
        assert len(multi) == 3
        for b in range(3):
            assert multi[b].shape == single[b].shape
            np.testing.assert_allclose(
                np.asarray(multi[b]), single[b], rtol=1e-4, atol=1e-5 * abs(single[b]).max()
            )

    def test_without_nuisance(self):
        """The nuisance-free configuration agrees too, and theta is spatial-only."""
        fm_single, fm_multi, model = self._pair(None)
        assert fm_multi.n_params == model.n_params
        theta = fm_single.sample_prior(jax.random.PRNGKey(1), 1)[0]
        np.testing.assert_allclose(
            float(fm_multi.log_posterior(theta)),
            float(fm_single.log_posterior(theta)),
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# 2. Genuine multi-resolution rendering and recovery
# ---------------------------------------------------------------------------


class TestGenuineMultiResolution:
    """Bands on different native grids, one of them with a rotated WCS."""

    def test_geometry_and_shapes(self, multires_mock):
        """Per-band shapes, pixel areas and the sky/pixel round trip."""
        fm, theta_true, _ = multires_mock
        assert fm.n_bands == 3
        assert fm.shapes == [(32, 32), (16, 16), (16, 16)]
        np.testing.assert_allclose(fm.pixel_areas, [0.02**2, 0.04**2, 0.04**2], rtol=1e-12)
        for b, cfg in enumerate(BAND_SPEC):
            h, w = cfg["shape"]
            assert fm.model_images(theta_true)[b].shape == (h, w)
            # the reference position is the frame centre of every band
            row, col = fm.sky_to_pixel(b, 0.0, 0.0)
            np.testing.assert_allclose([float(row), float(col)], [(h - 1) / 2, (w - 1) / 2])
            dy, dx = fm.pixel_to_sky(b, 3.0, 5.0)
            back_row, back_col = fm.sky_to_pixel(b, dy, dx)
            np.testing.assert_allclose([float(back_row), float(back_col)], [3.0, 5.0], atol=1e-9)
        # band 1 is rotated by 30 degrees, so its affine is not diagonal
        assert abs(np.asarray(fm.observation[1].affine)[0, 1]) > 1e-3
        # East-left: +column is West, i.e. d(dx)/d(col) < 0 for the unrotated bands
        assert np.asarray(fm.observation[0].affine)[1, 1] < 0

    def test_truth_has_unit_chi2(self, multires_mock):
        """chi2_red at the truth is ~1 across the three ragged grids."""
        fm, theta_true, _ = multires_mock
        assert fm.n_data == 32 * 32 + 2 * 16 * 16
        chi2, n_data = fm.chi2(theta_true)
        assert n_data == fm.n_data
        assert 0.85 < float(chi2) / n_data < 1.15
        assert 0.85 < chi2_reduced(fm, theta_true) < 1.15
        chi = fm.chi_maps(theta_true)
        assert [c.shape for c in chi] == fm.shapes
        assert abs(float(jnp.mean(chi[0]))) < 0.2

    def test_component_images_sum_to_the_model(self, multires_mock):
        """Unconvolved per-component images sum to the unconvolved band image."""
        fm, theta_true, _ = multires_mock
        comps = fm.component_images_per_band(theta_true)
        seds = fm.spatial_model.component_seds(theta_true, fm.emulator)
        for b in range(fm.n_bands):
            assert comps[b].shape == (2, *fm.shapes[b])
            direct = jnp.einsum(
                "k,khw->hw", seds[:, fm.band_indices[b]], fm.band_profiles(theta_true, b)
            )
            np.testing.assert_allclose(
                np.asarray(comps[b].sum(axis=0)), np.asarray(direct), rtol=1e-5
            )

    def test_blind_map_recovers_masses_and_sky_centres(self, multires_mock):
        """Blind init + multistart MAP: masses to 0.1 dex, centres to 0.02 arcsec."""
        fm, theta_true, _ = multires_mock
        model = fm.spatial_model
        theta0 = blind_initial_full_theta(fm)
        assert theta0.shape == (fm.n_params,)
        result = multistart_map(
            fm,
            theta0,
            [{"tau_v": 0.3}, {"tau_v": 3.0}],
            n_rounds=2,
            steps_per_round=200,
            final_steps=200,
        )
        mu, _, _, sps = model.component_params(result.theta)
        np.testing.assert_allclose(np.asarray(sps[:, 0]), TRUE_LOG_M, atol=0.1)
        np.testing.assert_allclose(np.asarray(mu), np.asarray(TRUE_MU), atol=0.02)
        assert result.neg_log_posterior <= -float(fm.log_posterior(theta_true)) + 5.0
        # the recovered sky centre lands on the right pixel of the rotated band
        row, col = fm.sky_to_pixel(1, float(mu[0, 0]), float(mu[0, 1]))
        true_row, true_col = fm.sky_to_pixel(1, *TRUE_MU[0])
        np.testing.assert_allclose([float(row), float(col)], [true_row, true_col], atol=0.5)

    def test_blind_initial_theta_reference_band(self, multires_mock):
        """The reference band is the highest-S/N one and can be named explicitly."""
        fm, _, _ = multires_mock
        obs = fm.observation
        auto = reference_band_index(obs)
        assert auto == reference_band_index(obs, BAND_NAMES[auto])
        assert reference_band_index(obs, 1) == 1
        with pytest.raises(ValueError):
            reference_band_index(obs, "not-a-band")
        with pytest.raises(ValueError):
            reference_band_index(obs, 9)

        theta = blind_initial_theta(fm.spatial_model, obs, ref_band=0)
        mu, sigma, _, _ = fm.spatial_model.component_params(theta)
        # both components start on the light centroid, within a pixel of the truth
        np.testing.assert_allclose(np.asarray(mu[0]), np.asarray(mu[1]), atol=1e-7)
        assert abs(float(mu[0, 0])) < 0.06 and abs(float(mu[0, 1])) < 0.06
        assert float(sigma[0, 0]) < float(sigma[1, 0])

    def test_blind_initial_theta_needs_arcsec_model(self, multires_mock):
        """A pixel-index model cannot be initialised from a multi-resolution cube."""
        fm, _, _ = multires_mock
        pixel_model = AdditiveComponentModel(
            n_components=1,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(32, 32),
            mass_param=MASS,
        )
        with pytest.raises(ValueError, match="arcsec mode"):
            blind_initial_theta(pixel_model, fm.observation)


# ---------------------------------------------------------------------------
# 3. Samplers run unchanged
# ---------------------------------------------------------------------------


class TestSamplers:
    """NSS and NUTS only touch the log_* / sample_prior / n_params interface."""

    def test_nss_returns_finite_evidence(self, multires_mock):
        """A tiny nested-sampling run produces a finite logZ."""
        pytest.importorskip("blackjax", reason="blackjax not installed")
        from arachne.inference.nss_sampler import NSSSampler

        fm, _, _ = multires_mock
        sampler = NSSSampler(
            fm,
            num_live=30,
            num_inner_steps=3,
            num_delete=3,
            termination=5.0,
            n_samples_out=16,
            max_steps=20,
        )
        result = sampler.run(jax.random.PRNGKey(0))
        assert np.isfinite(result.logZ)
        assert result.samples.shape[1] == fm.n_params

    def test_nuts_runs_two_chains(self, multires_mock):
        """NUTS gives (n_chains, n_samples, n_params) and the usual diagnostics."""
        pytest.importorskip("blackjax", reason="blackjax not installed")
        from arachne.inference.nuts_sampler import NUTSSampler

        fm, theta_true, _ = multires_mock
        sampler = NUTSSampler(fm, n_warmup=20, n_samples=10)
        result = sampler.run(theta_true, jax.random.PRNGKey(1), n_chains=2)
        assert result.chains.shape == (2, 10, fm.n_params)
        for key in (
            "rhat",
            "ess",
            "n_divergent",
            "mean_tree_depth",
            "step_size",
            "acceptance_rate",
            "fraction_dims_moved",
            "warmup_per_chain",
            "max_num_doublings",
        ):
            assert key in result.diagnostics


# ---------------------------------------------------------------------------
# 4. Posterior-predictive products
# ---------------------------------------------------------------------------


class TestProducts:
    """Per-band lists out of the posterior-predictive helpers."""

    def test_residual_summary_per_band(self, multires_mock):
        """Ragged lists of the right shapes and chi2_red ~ 1 at the truth."""
        fm, theta_true, _ = multires_mock
        samples = jnp.tile(theta_true[None, :], (6, 1))
        summary = residual_summary(fm, samples, n_max=4)
        assert isinstance(summary["median_model"], list)
        assert [m.shape for m in summary["median_model"]] == fm.shapes
        assert [c.shape for c in summary["chi"]] == fm.shapes
        assert summary["n_data"] == fm.n_data
        assert summary["n_params"] == fm.n_params
        assert summary["dof"] == fm.n_data - fm.n_params
        assert 0.85 < summary["chi2_red"] < 1.15
        assert summary["chi2_red_per_band"].shape == (3,)
        assert np.all(summary["chi2_red_per_band"] < 1.5)
        assert 0.0 <= summary["frac_chi_gt_3"] < 0.05
        assert summary["band_names"] == BAND_NAMES
        assert summary["n_samples_used"] == 4

    def test_image_samples_per_band(self, multires_mock):
        """model_image_samples / component_image_samples return per-band stacks."""
        fm, theta_true, _ = multires_mock
        samples = jnp.tile(theta_true[None, :], (5, 1))
        images = model_image_samples(fm, samples, n_max=3, chunk=2)
        assert [im.shape for im in images] == [(3, *s) for s in fm.shapes]
        comps = component_image_samples(fm, samples, n_max=3, chunk=2)
        assert [c.shape for c in comps] == [(3, 2, *s) for s in fm.shapes]
        direct = fm.model_images(theta_true)
        for b in range(fm.n_bands):
            reference = np.asarray(direct[b])
            np.testing.assert_allclose(
                np.asarray(images[b][0]),
                reference,
                rtol=1e-4,
                atol=1e-6 * np.abs(reference).max(),
            )


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Configurations that cannot produce a physical multi-resolution model."""

    def test_frame_normalisation_rejected(self):
        """normalisation='frame' is meaningless on another band's grid."""
        model = AdditiveComponentModel(
            n_components=1,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(32, 32),
            mass_param=MASS,
            pixel_scale=0.02,
            normalisation="frame",
        )
        with pytest.raises(ValueError, match="analytic"):
            MultiResolutionForwardModel.build(_multires_observation(), model, _emulator())

    def test_pixel_mode_rejected(self):
        """pixel_scale=None means the sizes are in one grid's pixels."""
        model = AdditiveComponentModel(
            n_components=1,
            emulator_param_names=SPS_PARAM_NAMES,
            param_bounds=PARAM_BOUNDS,
            image_shape=(32, 32),
            mass_param=MASS,
        )
        with pytest.raises(ValueError, match="arcsec mode"):
            MultiResolutionForwardModel.build(_multires_observation(), model, _emulator())

    def test_missing_psf_rejected(self):
        """Every band needs a PSF on its own pixel grid."""
        with pytest.raises(ValueError, match="psf=None"):
            MultiResolutionForwardModel.build(
                _multires_observation(psfs=False), _multires_model(1), _emulator()
            )

    def test_unknown_band_rejected(self):
        """The emulator must predict every observed band."""
        emulator = ColourLinearEmulator(
            _param_names=SPS_PARAM_NAMES, _band_names=["JWST/NIRCam.F444W"]
        )
        with pytest.raises(ValueError, match="does not predict"):
            MultiResolutionForwardModel.build(_multires_observation(), _multires_model(1), emulator)

    def test_bad_arguments_rejected(self):
        """Wrong-length model_error_frac / oversample and a non-additive model."""
        obs = _multires_observation()
        with pytest.raises(ValueError, match="model_error_frac"):
            MultiResolutionForwardModel.build(
                obs, _multires_model(1), _emulator(), model_error_frac=[0.1, 0.2]
            )
        with pytest.raises(ValueError, match="oversample"):
            MultiResolutionForwardModel.build(
                obs, _multires_model(1), _emulator(), oversample=[1, 2]
            )
        with pytest.raises(ValueError, match="oversample"):
            MultiResolutionForwardModel.build(obs, _multires_model(1), _emulator(), oversample=0)

    def test_non_additive_model_rejected(self, gmm_model):
        """Only AdditiveComponentModel can be rendered on foreign grids."""
        with pytest.raises(TypeError, match="AdditiveComponentModel"):
            MultiResolutionForwardModel.build(_multires_observation(), gmm_model, _emulator())


def test_package_export():
    """The class is exported from the top-level package."""
    import arachne

    assert arachne.MultiResolutionForwardModel is MultiResolutionForwardModel
    assert "MultiResolutionForwardModel" in arachne.__all__
