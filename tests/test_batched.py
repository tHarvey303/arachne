"""Tests for GPU-batched fitting of many equal-shape cutouts (`inference/batched.py`)."""

from __future__ import annotations

import math
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.forward_model.nuisance import NuisanceModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.batched import (
    BatchedForwardModel,
    batched_blind_initial_theta,
    batched_multistart_map,
    batched_nuts,
    fit_batch_nss,
)
from arachne.inference.initialisation import blind_initial_full_theta, find_map
from arachne.spatial.additive import AdditiveComponentModel

# Mirror tests/conftest.py constants (tests/ is not a package).
N_BANDS = 3
H = 16
W = 16
BAND_NAMES = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]
MASS = "log_stellar_mass"
FULL_NAMES = [MASS, "log_age", "tau_v", "redshift"]
FULL_BOUNDS = {
    MASS: (6.0, 12.0),
    "log_age": (7.0, 10.1),
    "tau_v": (0.0, 4.0),
    "redshift": (0.0, 10.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RedshiftMassEmulator(SPSEmulator, eqx.Module):
    """flux_b = 10**(logM - 9) * (b + 1) * (1 + 0.1 tau_v + 0.05 z).

    Exactly linear in ``10**log_mass`` (so the linear mass solve is exact) and
    genuinely sensitive to the *fixed* redshift, which is what makes the
    per-galaxy ``fixed_params_per_galaxy`` machinery testable.
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
        """Mass-linear flux with tau_v and redshift colour terms."""
        m9 = 10.0 ** (params[:, 0:1] - 9.0)
        colour = 1.0 + 0.1 * params[:, 2:3] + 0.05 * params[:, 3:4]
        bands = jnp.arange(1, len(self._band_names) + 1, dtype=jnp.float32)[None, :]
        return m9 * colour * bands


def _gaussian_psf(sigma: float) -> PSFModel:
    """Three 9x9 Gaussian kernels of widths ``sigma``, ``1.2 sigma``, ``1.5 sigma``."""
    kernels = []
    for s in (sigma, 1.2 * sigma, 1.5 * sigma):
        y, x = np.mgrid[-4:5, -4:5]
        k = np.exp(-(x**2 + y**2) / (2.0 * s**2)).astype(np.float32)
        kernels.append(k / k.sum())
    return PSFModel(kernels=np.stack(kernels), band_names=BAND_NAMES)


def _logit(value: float, lo: float, hi: float) -> float:
    """Raw (unconstrained) value whose sigmoid maps to ``value`` in ``(lo, hi)``."""
    u = (float(value) - lo) / (hi - lo)
    u = min(max(u, 1e-3), 1 - 1e-3)
    return math.log(u) - math.log1p(-u)


def _spatial_model(n_components: int = 1, redshift: float = 0.0) -> AdditiveComponentModel:
    """Additive model with redshift declared fixed (a per-galaxy placeholder)."""
    return AdditiveComponentModel(
        n_components=n_components,
        emulator_param_names=FULL_NAMES,
        param_bounds=FULL_BOUNDS,
        image_shape=(H, W),
        fixed_params={"redshift": redshift},
        mass_param=MASS,
    )


def _truth_theta(
    model: AdditiveComponentModel,
    mu: tuple[float, float],
    log_sigma: float,
    log_mass: float,
    log_age: float = 9.0,
    tau_v: float = 0.5,
) -> jnp.ndarray:
    """Hand-built single-component theta at known physical values."""
    sps = [
        _logit(log_mass, *FULL_BOUNDS[MASS]),
        _logit(log_age, *FULL_BOUNDS["log_age"]),
        _logit(tau_v, *FULL_BOUNDS["tau_v"]),
    ]
    theta = np.zeros(model.n_params, dtype=np.float32)
    theta[:5] = [mu[0], mu[1], log_sigma, log_sigma, 0.0]
    theta[5:8] = sps
    return jnp.asarray(theta)


@pytest.fixture(scope="module")
def emulator() -> RedshiftMassEmulator:
    """Mass-linear, redshift-sensitive emulator over the 4 test parameters."""
    return RedshiftMassEmulator(_param_names=FULL_NAMES, _band_names=BAND_NAMES)


@pytest.fixture(scope="module")
def sample() -> dict:
    """Three 3-band 16x16 cutouts with different noise, PSF width and redshift."""
    redshifts = [1.0, 2.0, 3.0]
    psfs = [_gaussian_psf(s) for s in (1.0, 1.3, 1.7)]
    cubes = []
    for i in range(3):
        rng = np.random.default_rng(100 + i)
        cubes.append(
            ObservationCube(
                flux=rng.normal(5.0, 1.0, (N_BANDS, H, W)).astype(np.float32),
                variance=np.full((N_BANDS, H, W), 1.0 + 0.25 * i, dtype=np.float32),
                mask=np.ones((N_BANDS, H, W), dtype=np.float32),
                band_names=BAND_NAMES,
                pixel_scale=0.03,
            ).to_jax()
        )
    return {"cubes": cubes, "psfs": psfs, "redshifts": redshifts}


def _build_batched(sample: dict, emulator, nuisance=None, n_components: int = 1):
    """BatchedForwardModel over the whole sample with per-galaxy spec-z."""
    return BatchedForwardModel.build(
        observations=sample["cubes"],
        psf_models=sample["psfs"],
        spatial_model=_spatial_model(n_components),
        emulator=emulator,
        model_error_frac=0.05,
        nuisance=nuisance,
        fixed_params_per_galaxy={"redshift": np.asarray(sample["redshifts"])},
        galaxy_ids=[f"mock_{i}" for i in range(len(sample["cubes"]))],
    )


def _single_forward_model(sample: dict, emulator, i: int, nuisance=None, n_components: int = 1):
    """An independently built per-galaxy ForwardModel (the equality reference)."""
    return ForwardModel.build(
        obs=sample["cubes"][i],
        psf_model=sample["psfs"][i],
        spatial_model=_spatial_model(n_components, redshift=sample["redshifts"][i]),
        emulator=emulator,
        model_error_frac=0.05,
        nuisance=nuisance,
    )


# ---------------------------------------------------------------------------
# 1. Equality with per-galaxy ForwardModels
# ---------------------------------------------------------------------------


class TestBatchedEqualsSingle:
    """The batched log-posterior must equal the per-galaxy one to float32."""

    @pytest.mark.parametrize("with_nuisance", [False, True])
    def test_log_posterior_matches(self, sample, emulator, with_nuisance):
        """N=3 galaxies, different noise / PSF width / fixed z, with and without nuisance."""
        nuisance = (
            NuisanceModel(N_BANDS, fit_sky=True, fit_shifts=True, fit_noise_scale=True)
            if with_nuisance
            else None
        )
        bfm = _build_batched(sample, emulator, nuisance=nuisance)
        thetas = jax.random.normal(jax.random.PRNGKey(3), (3, bfm.n_params)) * 0.5

        batched_post = np.asarray(bfm.log_posterior(thetas))
        batched_like = np.asarray(bfm.log_likelihood(thetas))
        batched_prior = np.asarray(bfm.log_prior(thetas))

        for i in range(3):
            fm = _single_forward_model(sample, emulator, i, nuisance=nuisance)
            assert batched_post[i] == pytest.approx(float(fm.log_posterior(thetas[i])), rel=1e-5)
            assert batched_like[i] == pytest.approx(float(fm.log_likelihood(thetas[i])), rel=1e-5)
            assert batched_prior[i] == pytest.approx(float(fm.log_prior(thetas[i])), rel=1e-5)

        # log_posterior == log_likelihood + log_prior, batched too.
        np.testing.assert_allclose(batched_post, batched_like + batched_prior, rtol=1e-5)

    def test_model_images_match(self, sample, emulator):
        """The batched model images equal the per-galaxy ones."""
        bfm = _build_batched(sample, emulator)
        thetas = jax.random.normal(jax.random.PRNGKey(4), (3, bfm.n_params)) * 0.4
        images = np.asarray(bfm.model_images(thetas))
        assert images.shape == (3, N_BANDS, H, W)
        for i in range(3):
            fm = _single_forward_model(sample, emulator, i)
            np.testing.assert_allclose(
                images[i], np.asarray(fm._model_image(thetas[i])), rtol=1e-4, atol=1e-5
            )

    def test_fixed_z_actually_differs(self, sample, emulator):
        """Giving every galaxy the same z changes the answer: the batch axis is live."""
        bfm = _build_batched(sample, emulator)
        same_z = BatchedForwardModel.build(
            observations=sample["cubes"],
            psf_models=sample["psfs"],
            spatial_model=_spatial_model(1, redshift=sample["redshifts"][0]),
            emulator=emulator,
            model_error_frac=0.05,
        )
        thetas = jnp.zeros((3, bfm.n_params), dtype=jnp.float32)
        per_galaxy = np.asarray(bfm.log_posterior(thetas))
        shared = np.asarray(same_z.log_posterior(thetas))
        assert per_galaxy[0] == pytest.approx(shared[0], rel=1e-5)  # galaxy 0 has that z
        assert not np.allclose(per_galaxy[1:], shared[1:])

    def test_forward_model_view_and_helpers(self, sample, emulator):
        """``forward_model(i)``, ``log_posterior_single`` and the plumbing helpers."""
        nuisance = NuisanceModel(N_BANDS, fit_sky=True)
        bfm = _build_batched(sample, emulator, nuisance=nuisance)
        assert bfm.n_galaxies == 3
        assert bfm.image_shape == (H, W)
        assert bfm.band_names == BAND_NAMES
        assert bfm.n_params == bfm.n_spatial_params + nuisance.n_params
        assert "BatchedForwardModel" in repr(bfm)

        theta = jnp.zeros(bfm.n_params, dtype=jnp.float32)
        fm = bfm.forward_model(1)
        assert isinstance(fm, ForwardModel)
        assert float(bfm.log_posterior_single(1, theta)) == pytest.approx(
            float(fm.log_posterior(theta)), rel=1e-6
        )

        spatial, nuis = bfm.split_theta(jnp.zeros((3, bfm.n_params)))
        assert spatial.shape == (3, bfm.n_spatial_params)
        assert nuis.shape == (3, nuisance.n_params)
        assert bfm.initial_theta_from_spatial(jnp.zeros(bfm.n_spatial_params)).shape == (
            3,
            bfm.n_params,
        )
        assert bfm.sample_prior(jax.random.PRNGKey(0), 5).shape == (3, 5, bfm.n_params)

    def test_shared_psf_and_stacked_arrays(self, sample, emulator):
        """A single shared PSFModel and a stacked-array mapping both work."""
        stacked = {
            "flux": jnp.stack([c.flux for c in sample["cubes"]]),
            "variance": jnp.stack([c.variance for c in sample["cubes"]]),
            "mask": jnp.stack([c.mask for c in sample["cubes"]]),
            "band_names": BAND_NAMES,
            "pixel_scale": 0.03,
        }
        bfm = BatchedForwardModel.build(
            observations=stacked,
            psf_models=sample["psfs"][0],
            spatial_model=_spatial_model(1),
            emulator=emulator,
            model_error_frac=0.05,
        )
        assert bfm.in_axes["psf_ffts"] is None
        thetas = jnp.zeros((3, bfm.n_params), dtype=jnp.float32)
        values = np.asarray(bfm.log_posterior(thetas))
        for i in range(3):
            fm = ForwardModel.build(
                obs=sample["cubes"][i],
                psf_model=sample["psfs"][0],
                spatial_model=_spatial_model(1),
                emulator=emulator,
                model_error_frac=0.05,
            )
            assert values[i] == pytest.approx(float(fm.log_posterior(thetas[i])), rel=1e-5)

    def test_build_validation(self, sample, emulator):
        """Bad inputs raise informative errors."""
        with pytest.raises(ValueError, match="at least one galaxy"):
            BatchedForwardModel.build([], sample["psfs"][0], _spatial_model(), emulator)
        with pytest.raises(ValueError, match="but there are 3 galaxies"):
            BatchedForwardModel.build(
                sample["cubes"], sample["psfs"][:2], _spatial_model(), emulator
            )
        with pytest.raises(ValueError, match="not fixed parameters"):
            BatchedForwardModel.build(
                sample["cubes"],
                sample["psfs"],
                _spatial_model(),
                emulator,
                fixed_params_per_galaxy={"tau_v": np.zeros(3)},
            )


# ---------------------------------------------------------------------------
# 2. Blind initialisation + batched multistart MAP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_batch(emulator) -> dict:
    """Three noisy mocks rendered from known masses, with per-galaxy z and PSF."""
    redshifts = [1.0, 2.0, 3.0]
    psfs = [_gaussian_psf(s) for s in (1.0, 1.3, 1.6)]
    log_masses = [10.2, 10.6, 10.9]
    centres = [(7.5, 8.2), (8.3, 7.4), (7.8, 8.0)]
    truths = []
    cubes = []
    for i in range(3):
        model = _spatial_model(1, redshift=redshifts[i])
        theta = _truth_theta(model, centres[i], math.log(2.0 + 0.3 * i), log_masses[i])
        truths.append(theta)
        clean_cube = ObservationCube(
            flux=np.zeros((N_BANDS, H, W), dtype=np.float32),
            variance=np.full((N_BANDS, H, W), 0.04, dtype=np.float32),
            mask=np.ones((N_BANDS, H, W), dtype=np.float32),
            band_names=BAND_NAMES,
            pixel_scale=0.03,
        )
        fm = ForwardModel.build(
            obs=clean_cube,
            psf_model=psfs[i],
            spatial_model=model,
            emulator=emulator,
            model_error_frac=0.05,
        )
        clean = np.asarray(fm._model_image(theta))
        rng = np.random.default_rng(500 + i)
        noisy = clean + rng.normal(0.0, 0.2, clean.shape)
        cubes.append(
            ObservationCube(
                flux=noisy.astype(np.float32),
                variance=np.full((N_BANDS, H, W), 0.04, dtype=np.float32),
                mask=np.ones((N_BANDS, H, W), dtype=np.float32),
                band_names=BAND_NAMES,
                pixel_scale=0.03,
            ).to_jax()
        )
    return {
        "cubes": cubes,
        "psfs": psfs,
        "redshifts": redshifts,
        "log_masses": log_masses,
        "truths": truths,
    }


def _recovered_log_masses(bfm: BatchedForwardModel, thetas: jnp.ndarray) -> np.ndarray:
    """Physical log mass of component 0 for every galaxy."""
    model = bfm.template.spatial_model
    mass_col = model.emulator_param_names.index(MASS)
    spatial, _ = bfm.split_theta(thetas)
    return np.array(
        [float(model.component_params(spatial[i])[3][0, mass_col]) for i in range(len(spatial))]
    )


class TestBatchedMAP:
    """Blind init + batched multistart MAP recover the mock masses."""

    def test_recovers_masses_and_beats_truth(self, mock_batch, emulator):
        """Masses within 0.1 dex and log_post at least as good as the truth's."""
        bfm = _build_batched(mock_batch, emulator)
        theta0 = batched_blind_initial_theta(bfm)
        assert theta0.shape == (3, bfm.n_params)
        assert np.all(np.isfinite(np.asarray(theta0)))

        result = batched_multistart_map(
            bfm,
            theta0,
            archetypes=[{"tau_v": 2.0}],
            n_rounds=2,
            steps_per_round=150,
            lr=0.05,
            final_steps=150,
            final_lr=0.01,
        )
        assert result.theta.shape == (3, bfm.n_params)
        assert result.log_post.shape == (3,)
        assert result.archetype_index.shape == (3,)
        assert result.history.shape[0] == 3
        assert len(result.summary(bfm.galaxy_ids).splitlines()) == 4  # header + 3 galaxies

        recovered = _recovered_log_masses(bfm, result.theta)
        truth_masses = np.asarray(mock_batch["log_masses"])
        assert np.max(np.abs(recovered - truth_masses)) < 0.1, (recovered, truth_masses)

        truth_log_post = np.asarray(bfm.log_posterior(jnp.stack(mock_batch["truths"])))
        assert np.all(np.asarray(result.log_post) > truth_log_post - 5.0), (
            np.asarray(result.log_post),
            truth_log_post,
        )

    def test_matches_sequential_find_map_and_is_faster(self, emulator):
        """Wall-clock sanity: one batched N=8 MAP beats 8 sequential find_map calls."""
        n = 8
        rng = np.random.default_rng(7)
        cubes, psfs, redshifts = [], [], []
        for i in range(n):
            psf = _gaussian_psf(1.0 + 0.05 * i)
            z = 1.0 + 0.2 * i
            model = _spatial_model(1, redshift=z)
            theta = _truth_theta(model, (7.6, 8.1), math.log(2.0), 10.4)
            clean_cube = ObservationCube(
                flux=np.zeros((N_BANDS, H, W), dtype=np.float32),
                variance=np.full((N_BANDS, H, W), 0.04, dtype=np.float32),
                mask=np.ones((N_BANDS, H, W), dtype=np.float32),
                band_names=BAND_NAMES,
                pixel_scale=0.03,
            )
            fm = ForwardModel.build(
                obs=clean_cube,
                psf_model=psf,
                spatial_model=model,
                emulator=emulator,
                model_error_frac=0.05,
            )
            clean = np.asarray(fm._model_image(theta))
            cubes.append(
                ObservationCube(
                    flux=(clean + rng.normal(0.0, 0.2, clean.shape)).astype(np.float32),
                    variance=np.full((N_BANDS, H, W), 0.04, dtype=np.float32),
                    mask=np.ones((N_BANDS, H, W), dtype=np.float32),
                    band_names=BAND_NAMES,
                    pixel_scale=0.03,
                ).to_jax()
            )
            psfs.append(psf)
            redshifts.append(z)

        bfm = BatchedForwardModel.build(
            observations=cubes,
            psf_models=psfs,
            spatial_model=_spatial_model(1),
            emulator=emulator,
            model_error_frac=0.05,
            fixed_params_per_galaxy={"redshift": np.asarray(redshifts)},
        )
        map_kwargs = dict(n_rounds=1, steps_per_round=100, lr=0.05, final_steps=100, final_lr=0.01)
        theta0 = batched_blind_initial_theta(bfm)

        # Compile once, then time the second call.
        batched_multistart_map(bfm, theta0, archetypes=(), **map_kwargs)
        t0 = time.perf_counter()
        batched = batched_multistart_map(bfm, theta0, archetypes=(), **map_kwargs)
        t_batched = time.perf_counter() - t0

        singles = []
        t0 = time.perf_counter()
        for i in range(n):
            fm = bfm.forward_model(i)
            singles.append(find_map(fm, blind_initial_full_theta(fm), **map_kwargs))
        t_sequential = time.perf_counter() - t0

        speedup = t_sequential / max(t_batched, 1e-9)
        print(
            f"\nN={n} MAP wall clock: batched {t_batched:.2f} s (post-compile) vs "
            f"sequential {t_sequential:.2f} s -> {speedup:.1f}x"
        )
        # Not asserted strictly (CPU test runners are noisy), only that both
        # paths land in the same basin.
        batched_masses = _recovered_log_masses(bfm, batched.theta)
        single_masses = np.array([_recovered_log_masses(bfm, r.theta[None, :])[0] for r in singles])
        np.testing.assert_allclose(batched_masses, single_masses, atol=0.1)


# ---------------------------------------------------------------------------
# 3. Batched NUTS
# ---------------------------------------------------------------------------


class TestBatchedNUTS:
    """Shapes, diagnostics and reporting of the batched sampler."""

    def test_shapes_diagnostics_and_summary(self, mock_batch, emulator, tmp_path):
        """Short run: correct shapes, finite diagnostics, N summary rows, HDF5 groups."""
        bfm = _build_batched(mock_batch, emulator)
        theta_init = batched_blind_initial_theta(bfm)
        result = batched_nuts(
            bfm,
            theta_init,
            jax.random.PRNGKey(11),
            n_warmup=30,
            n_samples=20,
            n_chains=2,
            chain_jitter=0.1,
            max_num_doublings=5,
        )
        d = bfm.n_params
        assert result.chains.shape == (3, 2, 20, d)
        assert result.samples.shape == (3, 40, d)
        assert np.all(np.isfinite(np.asarray(result.chains)))

        diag = result.diagnostics
        assert diag["rhat"].shape == (3, d)
        assert diag["ess"].shape == (3, d)
        assert diag["n_divergent"].shape == (3,)
        assert diag["mean_tree_depth"].shape == (3,)
        assert diag["acceptance_rate"].shape == (3,)
        assert diag["step_size"].shape == (3, 2)
        for key in ("rhat", "ess", "mean_tree_depth", "acceptance_rate", "step_size"):
            assert np.all(np.isfinite(np.asarray(diag[key], dtype=float))), key

        lines = result.summary().splitlines()
        assert len(lines) == 4  # header + one line per galaxy
        for gid in bfm.galaxy_ids:
            assert any(line.startswith(gid) for line in lines)

        per_galaxy = result.result(1)
        assert per_galaxy.chains.shape == (2, 20, d)
        assert per_galaxy.diagnostics["rhat"].shape == (d,)

        path = tmp_path / "batched_nuts.h5"
        result.to_hdf5(path)
        import h5py

        with h5py.File(path, "r") as f:
            assert f.attrs["n_galaxies"] == 3
            for gid in bfm.galaxy_ids:
                assert f[f"{gid}/chains"].shape == (2, 20, d)
                assert f[f"{gid}/theta"].shape == (40, d)
                assert np.isfinite(f[gid].attrs["acceptance_rate"])

    def test_explicit_per_chain_inits(self, mock_batch, emulator):
        """A (N, C, d) ``theta_init`` is used as given."""
        bfm = _build_batched(mock_batch, emulator)
        theta_init = batched_blind_initial_theta(bfm)
        inits = jnp.stack([theta_init, theta_init + 0.05], axis=1)  # (3, 2, d)
        result = batched_nuts(
            bfm,
            inits,
            jax.random.PRNGKey(12),
            n_warmup=20,
            n_samples=10,
            n_chains=2,
            max_num_doublings=4,
        )
        assert result.chains.shape == (3, 2, 10, bfm.n_params)


# ---------------------------------------------------------------------------
# 4. Sequential NSS helper
# ---------------------------------------------------------------------------


class TestFitBatchNSS:
    """The sequential nested-sampling helper returns one result per galaxy."""

    def test_returns_finite_logz(self, mock_batch, emulator):
        """N=2 with tiny settings; both results carry a finite logZ."""
        two = {key: value[:2] for key, value in mock_batch.items()}
        bfm = _build_batched(two, emulator)
        results = fit_batch_nss(
            bfm,
            jax.random.PRNGKey(21),
            num_live=60,
            num_inner_steps=5,
            num_delete=6,
            termination=0.5,
            n_samples_out=40,
            max_steps=200,
        )
        assert len(results) == 2
        for result in results:
            assert np.isfinite(result.logZ)
            assert np.isfinite(result.logZ_err)
            assert result.samples.shape == (40, bfm.n_params)
