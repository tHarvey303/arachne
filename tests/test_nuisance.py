"""Tests for NuisanceModel and its integration into ForwardModel."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy.stats import norm

from arachne.data.observation import ObservationCube
from arachne.forward_model.nuisance import NuisanceModel
from arachne.forward_model.pipeline import ForwardModel

BANDS = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]


class TestNuisanceLayout:
    """Parameter counts, names and the empty configuration."""

    @pytest.mark.parametrize(
        ("fit_sky", "fit_shifts", "fit_noise", "expected"),
        [
            (True, False, False, 3),
            (False, True, False, 6),
            (False, False, True, 3),
            (True, True, False, 9),
            (True, True, True, 12),
            (False, False, False, 0),
        ],
    )
    def test_n_params_combinations(self, fit_sky, fit_shifts, fit_noise, expected):
        """n_params counts each enabled block once per band (twice for shifts)."""
        nm = NuisanceModel(3, fit_sky=fit_sky, fit_shifts=fit_shifts, fit_noise_scale=fit_noise)
        assert nm.n_params == expected

    def test_param_names_default(self):
        """Default names use band_i placeholders in vector order."""
        nm = NuisanceModel(2, fit_sky=True, fit_shifts=True, fit_noise_scale=True)
        names = nm.param_names()
        assert names == [
            "sky[band_0]",
            "sky[band_1]",
            "dy[band_0]",
            "dx[band_0]",
            "dy[band_1]",
            "dx[band_1]",
            "log_noise_scale[band_0]",
            "log_noise_scale[band_1]",
        ]
        assert len(names) == nm.n_params

    def test_param_names_with_band_names(self):
        """Supplied band names appear inside the brackets."""
        nm = NuisanceModel(3, fit_sky=True)
        assert nm.param_names(BANDS)[1] == "sky[JWST/NIRCam.F200W]"

    def test_param_names_wrong_length_raises(self):
        """A band_names list of the wrong length is rejected."""
        nm = NuisanceModel(3)
        with pytest.raises(ValueError, match="n_bands"):
            nm.param_names(["a", "b"])

    def test_empty_model_everything_works(self):
        """With no blocks enabled every method still returns sane empty results."""
        nm = NuisanceModel(4, fit_sky=False)
        assert nm.n_params == 0
        assert nm.param_names() == []
        assert nm.initial_theta().shape == (0,)
        assert float(nm.log_prior(jnp.zeros(0))) == 0.0
        assert nm.sample_prior(jax.random.PRNGKey(0), 5).shape == (5, 0)
        blocks = nm.split(jnp.zeros(0))
        assert blocks["sky"].shape == (4,)
        assert blocks["shifts"].shape == (4, 2)
        assert blocks["log_noise_scale"].shape == (4,)

    def test_invalid_n_bands_raises(self):
        """n_bands must be positive."""
        with pytest.raises(ValueError, match="n_bands"):
            NuisanceModel(0)

    def test_non_positive_sigma_raises(self):
        """An enabled block needs a strictly positive prior sigma."""
        with pytest.raises(ValueError, match="sky_prior_sigma"):
            NuisanceModel(2, fit_sky=True, sky_prior_sigma=0.0)


class TestNuisanceSplit:
    """split() returns the right values and zeros for disabled blocks."""

    def test_split_all_blocks(self):
        """Values land in the right block in the documented order."""
        nm = NuisanceModel(2, fit_sky=True, fit_shifts=True, fit_noise_scale=True)
        theta = jnp.arange(nm.n_params, dtype=jnp.float32)
        blocks = nm.split(theta)
        np.testing.assert_allclose(blocks["sky"], [0.0, 1.0])
        np.testing.assert_allclose(blocks["shifts"], [[2.0, 3.0], [4.0, 5.0]])
        np.testing.assert_allclose(blocks["log_noise_scale"], [6.0, 7.0])

    def test_split_zeros_for_disabled_blocks(self):
        """Blocks that are not fitted come back as zeros of the right shape."""
        nm = NuisanceModel(3, fit_sky=False, fit_shifts=True, fit_noise_scale=False)
        blocks = nm.split(jnp.arange(6, dtype=jnp.float32))
        np.testing.assert_allclose(blocks["sky"], np.zeros(3))
        np.testing.assert_allclose(blocks["log_noise_scale"], np.zeros(3))
        np.testing.assert_allclose(blocks["shifts"], np.arange(6).reshape(3, 2))

    def test_split_jit_safe(self):
        """split() traces cleanly under jax.jit (all config is static)."""
        nm = NuisanceModel(2, fit_sky=True, fit_shifts=True)
        fn = jax.jit(lambda t: nm.split(t)["shifts"])
        out = fn(jnp.arange(nm.n_params, dtype=jnp.float32))
        assert out.shape == (2, 2)


class TestNuisancePrior:
    """Prior normalisation and sampling."""

    def test_log_prior_matches_gaussian_sum(self):
        """log_prior equals the sum of normalised Gaussian log-densities."""
        nm = NuisanceModel(
            3,
            fit_sky=True,
            fit_shifts=True,
            fit_noise_scale=True,
            sky_prior_sigma=2.0,
            shift_prior_sigma=0.5,
            noise_scale_prior_sigma=0.3,
        )
        rng = np.random.default_rng(0)
        theta = rng.normal(size=nm.n_params).astype(np.float32)
        expected = (
            norm.logpdf(theta[:3], scale=2.0).sum()
            + norm.logpdf(theta[3:9], scale=0.5).sum()
            + norm.logpdf(theta[9:], scale=0.3).sum()
        )
        assert float(nm.log_prior(jnp.asarray(theta))) == pytest.approx(expected, rel=1e-5)

    def test_log_prior_peaks_at_zero(self):
        """The mode of the prior is the zero vector."""
        nm = NuisanceModel(2, fit_sky=True)
        assert float(nm.log_prior(jnp.zeros(2))) > float(nm.log_prior(jnp.ones(2)))

    def test_log_prior_differentiable(self):
        """jax.grad of the prior is finite and points back to zero."""
        nm = NuisanceModel(2, fit_sky=True, sky_prior_sigma=1.0)
        grad = jax.grad(nm.log_prior)(jnp.array([1.0, -2.0]))
        np.testing.assert_allclose(np.asarray(grad), [-1.0, 2.0], rtol=1e-5)

    def test_sample_prior_shape_and_scale(self):
        """sample_prior returns (n, n_params) with the right per-block spread."""
        nm = NuisanceModel(
            2,
            fit_sky=True,
            fit_shifts=True,
            sky_prior_sigma=3.0,
            shift_prior_sigma=0.25,
        )
        samples = nm.sample_prior(jax.random.PRNGKey(1), 20000)
        assert samples.shape == (20000, nm.n_params)
        std = np.asarray(samples).std(axis=0)
        np.testing.assert_allclose(std[:2], 3.0, rtol=0.05)
        np.testing.assert_allclose(std[2:], 0.25, rtol=0.05)

    def test_initial_theta_is_zeros(self):
        """initial_theta() starts every nuisance parameter at its prior mean."""
        nm = NuisanceModel(3, fit_sky=True, fit_noise_scale=True)
        theta = nm.initial_theta()
        assert theta.shape == (nm.n_params,)
        assert bool(jnp.all(theta == 0.0))


# ---------------------------------------------------------------------------
# ForwardModel integration
# ---------------------------------------------------------------------------


def _flat_obs(flux, H=16, W=16):
    """Constant-flux ObservationCube; ``flux`` is a scalar or per-band sequence."""
    flux_arr = jnp.broadcast_to(
        jnp.asarray(flux, dtype=jnp.float32).reshape(-1, 1, 1), (len(BANDS), H, W)
    )
    ones = jnp.ones((len(BANDS), H, W), dtype=jnp.float32)
    return ObservationCube(
        flux=jnp.asarray(flux_arr),
        variance=ones,
        mask=ones,
        band_names=BANDS,
        pixel_scale=0.031,
    )


@pytest.fixture
def nuisance_sky_shifts():
    """NuisanceModel fitting sky and shifts for the 3 test bands."""
    return NuisanceModel(3, fit_sky=True, fit_shifts=True)


@pytest.fixture
def fm_with_nuisance(tiny_observation, gaussian_psf, mock_emulator, gmm_model, nuisance_sky_shifts):
    """ForwardModel with a GMM spatial model plus sky and shift nuisances."""
    return ForwardModel.build(
        obs=tiny_observation,
        psf_model=gaussian_psf,
        spatial_model=gmm_model,
        emulator=mock_emulator,
        nuisance=nuisance_sky_shifts,
    )


class TestForwardModelNuisance:
    """ForwardModel with a NuisanceModel attached."""

    def test_n_params_sums_blocks(self, fm_with_nuisance, gmm_model, nuisance_sky_shifts):
        """n_params = spatial + nuisance."""
        assert fm_with_nuisance.n_params == gmm_model.n_params + nuisance_sky_shifts.n_params

    def test_split_theta_round_trip(self, fm_with_nuisance, gmm_model, nuisance_sky_shifts):
        """split_theta then concatenate recovers the original vector."""
        theta = jnp.arange(fm_with_nuisance.n_params, dtype=jnp.float32)
        ts, tn = fm_with_nuisance.split_theta(theta)
        assert ts.shape == (gmm_model.n_params,)
        assert tn.shape == (nuisance_sky_shifts.n_params,)
        np.testing.assert_allclose(np.asarray(jnp.concatenate([ts, tn])), np.asarray(theta))

    def test_initial_theta_from_spatial_length(self, fm_with_nuisance, gmm_model):
        """initial_theta_from_spatial appends the zeroed nuisance block."""
        theta_spatial = jnp.ones(gmm_model.n_params)
        full = fm_with_nuisance.initial_theta_from_spatial(theta_spatial)
        assert full.shape == (fm_with_nuisance.n_params,)
        np.testing.assert_allclose(np.asarray(full[: gmm_model.n_params]), 1.0)
        assert bool(jnp.all(full[gmm_model.n_params :] == 0.0))

    def test_log_posterior_finite_and_grad_finite(self, fm_with_nuisance):
        """log_posterior and its gradient are finite over the full vector."""
        theta = jnp.zeros(fm_with_nuisance.n_params).at[-6:].set(0.1)
        lp = fm_with_nuisance.log_posterior(theta)
        assert lp.shape == ()
        assert bool(jnp.isfinite(lp))
        grad = jax.grad(fm_with_nuisance.log_posterior)(theta)
        assert grad.shape == theta.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        # The nuisance block genuinely enters the posterior.
        assert bool(jnp.any(grad[-6:] != 0.0))

    def test_log_posterior_equals_like_plus_prior(self, fm_with_nuisance):
        """The fused posterior matches likelihood + prior exactly."""
        rng = np.random.default_rng(3)
        theta = jnp.asarray(rng.normal(0.0, 0.2, fm_with_nuisance.n_params).astype(np.float32))
        lp = float(fm_with_nuisance.log_posterior(theta))
        expected = float(fm_with_nuisance.log_likelihood(theta)) + float(
            fm_with_nuisance.log_prior(theta)
        )
        assert lp == pytest.approx(expected, rel=1e-5)

    def test_jit_compiles_with_nuisance(self, fm_with_nuisance):
        """The whole pipeline still compiles under jax.jit."""
        theta = jnp.zeros(fm_with_nuisance.n_params)
        assert bool(jnp.isfinite(jax.jit(fm_with_nuisance.log_posterior)(theta)))

    def test_sample_prior_propagates_not_implemented(self, fm_with_nuisance):
        """A spatial model without a proper prior still raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            fm_with_nuisance.sample_prior(jax.random.PRNGKey(0), 4)

    def test_no_nuisance_defaults(self, tiny_observation, gaussian_psf, mock_emulator, gmm_model):
        """With nuisance=None theta is exactly the spatial vector."""
        fm = ForwardModel.build(
            obs=tiny_observation,
            psf_model=gaussian_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
        )
        assert fm.nuisance is None
        assert fm.n_params == gmm_model.n_params
        theta = jnp.zeros(gmm_model.n_params)
        ts, tn = fm.split_theta(theta)
        assert ts.shape == theta.shape
        assert tn.shape == (0,)
        np.testing.assert_allclose(
            np.asarray(fm.initial_theta_from_spatial(theta)), np.asarray(theta)
        )

    def test_sky_offsets_model_image(self, tiny_observation, delta_psf, mock_emulator, gmm_model):
        """A non-zero sky adds a constant to the corresponding band only."""
        nm = NuisanceModel(3, fit_sky=True)
        fm = ForwardModel.build(
            obs=tiny_observation,
            psf_model=delta_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
            nuisance=nm,
        )
        theta0 = jnp.zeros(fm.n_params)
        theta1 = theta0.at[gmm_model.n_params].set(5.0)
        diff = np.asarray(fm._model_image(theta1) - fm._model_image(theta0))
        np.testing.assert_allclose(diff[0], 5.0, atol=1e-4)
        np.testing.assert_allclose(diff[1:], 0.0, atol=1e-4)

    def test_noise_scale_enters_likelihood(
        self, tiny_observation, delta_psf, mock_emulator, gmm_model
    ):
        """Inflating the noise scale changes the log-likelihood (log-det included)."""
        nm = NuisanceModel(3, fit_sky=False, fit_noise_scale=True)
        fm = ForwardModel.build(
            obs=tiny_observation,
            psf_model=delta_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
            nuisance=nm,
        )
        theta0 = jnp.zeros(fm.n_params)
        theta1 = theta0.at[gmm_model.n_params :].set(0.5)
        assert float(fm.log_likelihood(theta0)) != pytest.approx(
            float(fm.log_likelihood(theta1)), rel=1e-6
        )


class TestSkyRecovery:
    """End-to-end MAP recovery of an injected sky pedestal."""

    def test_map_recovers_injected_sky(self, delta_psf, mock_emulator, gmm_model):
        """A +3 nJy sky in band 0 is recovered by Adam over the nuisance block.

        The mock emulator at theta_spatial = 0 predicts 10 nJy in every band,
        so an observation of (13, 10, 10) nJy differs from the model only by
        the injected sky.
        """
        obs = _flat_obs([13.0, 10.0, 10.0])
        nm = NuisanceModel(3, fit_sky=True, sky_prior_sigma=10.0)
        fm = ForwardModel.build(
            obs=obs,
            psf_model=delta_psf,
            spatial_model=gmm_model,
            emulator=mock_emulator,
            nuisance=nm,
        )
        theta_spatial = jnp.zeros(gmm_model.n_params)

        def neg_log_post(theta_n):
            return -fm.log_posterior(jnp.concatenate([theta_spatial, theta_n]))

        opt = optax.adam(0.1)
        params = nm.initial_theta()
        state = opt.init(params)
        grad_fn = jax.jit(jax.grad(neg_log_post))
        for _ in range(400):
            updates, state = opt.update(grad_fn(params), state)
            params = optax.apply_updates(params, updates)

        sky = np.asarray(nm.split(params)["sky"])
        assert sky[0] == pytest.approx(3.0, abs=0.05)
        assert sky[1] == pytest.approx(0.0, abs=0.05)
        assert sky[2] == pytest.approx(0.0, abs=0.05)


class TestShiftReferenceBand:
    """A reference band pins one (dy, dx) pair to zero to break the shift/centre degeneracy."""

    def test_counts_names_and_split(self):
        """Counts, names, split and sampling skip the reference band."""
        nm = NuisanceModel(3, fit_sky=False, fit_shifts=True, shift_reference_band=1)
        assert nm.n_shift == 4 and nm.n_params == 4
        assert nm.shift_bands == [0, 2]
        assert nm.param_names(["a", "b", "c"]) == ["dy[a]", "dx[a]", "dy[c]", "dx[c]"]
        blocks = nm.split(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(blocks["shifts"], [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
        assert nm.initial_theta().shape == (4,)
        assert nm.sample_prior(jax.random.PRNGKey(0), 5).shape == (5, 4)

    def test_log_prior_counts_only_free_shifts(self):
        """The prior normalisation counts only the free shift parameters."""
        nm = NuisanceModel(3, fit_sky=False, fit_shifts=True, shift_reference_band=0)
        theta = jnp.asarray([0.1, -0.2, 0.3, 0.4])
        expected = -0.5 * float(jnp.sum((theta / 0.5) ** 2)) - 4 * (
            np.log(0.5) + 0.5 * np.log(2 * np.pi)
        )
        np.testing.assert_allclose(float(nm.log_prior(theta)), expected, rtol=1e-6)
        assert jnp.isfinite(jax.grad(nm.log_prior)(theta)).all()

    def test_split_is_jittable(self):
        """Split works under jit and zeroes the reference row."""
        nm = NuisanceModel(4, fit_sky=True, fit_shifts=True, shift_reference_band=3)
        theta = jnp.arange(nm.n_params, dtype=jnp.float32)
        out = jax.jit(lambda t: nm.split(t)["shifts"])(theta)
        assert out.shape == (4, 2)
        np.testing.assert_allclose(out[3], [0.0, 0.0])

    def test_reference_out_of_range_raises(self):
        """An out-of-range reference band is rejected."""
        with pytest.raises(ValueError):
            NuisanceModel(3, fit_shifts=True, shift_reference_band=3)
