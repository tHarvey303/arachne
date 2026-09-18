"""Tests for NSSSampler / NSSResult (tiny CPU run on a K=1 additive model).

Includes the checkpoint / resume contract: an interrupted run (stopped by
``max_steps``) that is resumed from its checkpoint with a larger ``max_steps``
reproduces an uninterrupted run **bit-exactly**, because the PRNG key for the
next step is checkpointed together with the sampler state.
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
from arachne.inference.nss_sampler import (
    NSSResult,
    NSSSampler,
    _checkpoint_file,
    _load_checkpoint,
)
from arachne.inference.nuts_sampler import NUTSResult
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

N_LIVE = 50
N_OUT = 120


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


@pytest.fixture
def model_k1() -> AdditiveComponentModel:
    """K=1 additive model (8 parameters)."""
    return AdditiveComponentModel(
        n_components=1,
        emulator_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(H, W),
        mass_param=MASS,
    )


@pytest.fixture
def fm(model_k1, gaussian_psf):
    """ForwardModel on a moderate-S/N K=1 mock."""
    emulator = LinearMassEmulator(_param_names=SPS_PARAM_NAMES, _band_names=BAND_NAMES)
    lo, hi = PARAM_BOUNDS[MASS]
    u = (9.5 - lo) / (hi - lo)
    theta_true = jnp.array(
        [7.5, 8.0, np.log(2.0), np.log(2.5), 0.1, np.log(u / (1 - u)), 0.0, 0.0],
        dtype=jnp.float32,
    )
    conv = PSFConvolver(gaussian_psf, image_shape=(H, W))
    truth = np.asarray(conv(model_k1.model_image(theta_true, emulator, (H, W))))
    sigma = truth.max(axis=(1, 2)) / 10.0
    rng = np.random.default_rng(3)
    flux = (truth + rng.normal(size=truth.shape) * sigma[:, None, None]).astype(np.float32)
    variance = np.broadcast_to(sigma[:, None, None] ** 2, truth.shape).astype(np.float32)
    obs = ObservationCube(flux, variance, np.ones_like(flux), BAND_NAMES, 0.031, None)
    return ForwardModel.build(obs, gaussian_psf, model_k1, emulator)


@pytest.fixture(scope="module")
def _result_cache():
    return {}


@pytest.fixture
def nss_result(fm, _result_cache) -> NSSResult:
    """One short NSS run, shared across tests in this module."""
    if "result" not in _result_cache:
        sampler = NSSSampler(
            fm,
            num_live=N_LIVE,
            num_inner_steps=6,
            num_delete=5,
            termination=0.5,
            n_samples_out=N_OUT,
            max_steps=150,
        )
        _result_cache["result"] = sampler.run(jax.random.PRNGKey(0))
    return _result_cache["result"]


class TestDefaults:
    """Constructor defaults."""

    def test_defaults(self, fm, model_k1):
        """num_inner_steps = 3 * n_params, num_delete = num_live // 10."""
        s = NSSSampler(fm)
        assert s.num_live == 500
        assert s.num_inner_steps == 3 * model_k1.n_params
        assert s.num_delete == 50
        assert s.termination == 1e-3 and s.n_samples_out == 1000 and s.max_steps == 100_000

    def test_num_delete_validation(self, fm):
        """num_delete must be < num_live."""
        with pytest.raises(ValueError):
            NSSSampler(fm, num_live=10, num_delete=10)


class TestRun:
    """End-to-end short run."""

    def test_result_type_and_shapes(self, nss_result, model_k1):
        """NSSResult subclassing NUTSResult with equal-weight samples."""
        assert isinstance(nss_result, NSSResult)
        assert isinstance(nss_result, NUTSResult)
        assert nss_result.samples.shape == (N_OUT, model_k1.n_params)
        assert nss_result.n_samples == N_OUT
        assert jnp.all(jnp.isfinite(nss_result.samples))
        assert nss_result.spatial_model is model_k1

    def test_evidence_diagnostics(self, nss_result):
        """Finite logZ / err / ESS and consistent counts."""
        assert np.isfinite(nss_result.logZ)
        assert np.isfinite(nss_result.logZ_err) and nss_result.logZ_err >= 0
        assert np.isfinite(nss_result.ess) and nss_result.ess > 1
        assert nss_result.n_steps >= 1
        assert nss_result.n_dead == nss_result.n_steps * 5
        assert nss_result.log_weights.shape == (nss_result.n_dead + N_LIVE,)
        assert nss_result.infos is not None
        assert nss_result.infos.particles.position.shape[0] == nss_result.n_dead + N_LIVE

    def test_samples_concentrate_near_truth(self, nss_result, model_k1):
        """Posterior mass sits near the true centre (7.5, 8.0)."""
        mu, _, _, _ = jax.vmap(model_k1.component_params)(nss_result.samples)
        med = np.median(np.asarray(mu[:, 0, :]), axis=0)
        assert abs(med[0] - 7.5) < 1.0 and abs(med[1] - 8.0) < 1.0

    def test_get_parameter_map(self, nss_result, model_k1):
        """NUTSResult machinery works on the resampled thetas."""
        maps = nss_result.get_parameter_map((H, W), percentiles=[50])
        assert set(maps) == set(model_k1.sps_param_names)
        assert maps[MASS].shape == (1, H, W)

    def test_hdf5_roundtrip(self, nss_result, model_k1, tmp_path):
        """to_hdf5 stores logZ etc.; from_hdf5 restores them."""
        import h5py

        path = tmp_path / "nss.h5"
        nss_result.to_hdf5(path)
        with h5py.File(path, "r") as f:
            assert f.attrs["sampler"] == "nss"
            assert np.isclose(f.attrs["logZ"], nss_result.logZ)
            assert np.isclose(f.attrs["logZ_err"], nss_result.logZ_err)
            assert f.attrs["n_steps"] == nss_result.n_steps
            assert f["diagnostics/log_weights"].shape == nss_result.log_weights.shape
        loaded = NSSResult.from_hdf5(path, model_k1)
        assert isinstance(loaded, NSSResult)
        np.testing.assert_allclose(np.asarray(loaded.samples), np.asarray(nss_result.samples))
        assert np.isclose(loaded.logZ, nss_result.logZ)
        assert loaded.ess == pytest.approx(nss_result.ess)
        assert loaded.n_dead == nss_result.n_dead
        np.testing.assert_allclose(loaded.log_weights, nss_result.log_weights)

    def test_explicit_initial_live_points(self, fm, model_k1):
        """initial_theta of shape (num_live, n_params) is accepted; bad shapes raise."""
        live = model_k1.sample_prior(jax.random.PRNGKey(5), 20)
        sampler = NSSSampler(
            fm,
            num_live=20,
            num_inner_steps=4,
            num_delete=2,
            termination=2.0,
            n_samples_out=10,
            max_steps=20,
        )
        result = sampler.run(jax.random.PRNGKey(1), initial_theta=live)
        assert result.samples.shape == (10, model_k1.n_params)
        with pytest.raises(ValueError):
            sampler.run(jax.random.PRNGKey(2), initial_theta=jnp.zeros(model_k1.n_params))

    def test_sample_prior_required(self, tiny_observation, gaussian_psf, mock_emulator, gmm_model):
        """A spatial model without sample_prior gives an informative NotImplementedError."""
        fm = ForwardModel.build(tiny_observation, gaussian_psf, gmm_model, mock_emulator)
        sampler = NSSSampler(fm, num_live=10, num_inner_steps=2, num_delete=1)
        with pytest.raises(NotImplementedError, match="sample_prior"):
            sampler.run(jax.random.PRNGKey(0))


CKPT_KWARGS = dict(
    num_live=N_LIVE,
    num_inner_steps=6,
    num_delete=5,
    termination=0.5,
    n_samples_out=40,
)


class TestCheckpointing:
    """run(checkpoint_path=..., checkpoint_every=..., resume=...)."""

    def test_resume_reproduces_uninterrupted_run(self, fm, tmp_path):
        """30 steps + resume to 60 == an uninterrupted 60-step run, bit-exactly.

        The checkpoint stores the PRNG key *as the next step will consume it*,
        so the resumed run replays the identical key sequence.  Everything else
        (state pytree, dead particles) round-trips through ``.npz`` as float32,
        hence the equality is exact rather than statistical.
        """
        key = jax.random.PRNGKey(0)
        uninterrupted = NSSSampler(fm, max_steps=60, **CKPT_KWARGS).run(key)
        assert uninterrupted.n_steps == 60  # max_steps-limited, not converged

        interrupted = NSSSampler(fm, max_steps=30, **CKPT_KWARGS).run(
            key, checkpoint_path=tmp_path, checkpoint_every=10
        )
        assert interrupted.n_steps == 30
        assert _checkpoint_file(tmp_path).exists()

        resumed = NSSSampler(fm, max_steps=60, **CKPT_KWARGS).run(
            key, checkpoint_path=tmp_path, checkpoint_every=10, resume=True
        )
        assert resumed.n_steps == 60
        assert resumed.logZ == uninterrupted.logZ
        assert resumed.logZ_err == uninterrupted.logZ_err
        assert resumed.n_dead == uninterrupted.n_dead
        np.testing.assert_array_equal(
            np.asarray(resumed.samples), np.asarray(uninterrupted.samples)
        )
        np.testing.assert_array_equal(resumed.log_weights, uninterrupted.log_weights)

    def test_checkpoint_contents(self, fm, tmp_path):
        """The checkpoint carries the state, dead list, step count and geometry."""
        sampler = NSSSampler(fm, max_steps=12, **CKPT_KWARGS)
        sampler.run(jax.random.PRNGKey(1), checkpoint_path=tmp_path, checkpoint_every=5)
        ckpt = _load_checkpoint(tmp_path)
        assert ckpt is not None
        # A final checkpoint is written on exit even off the 5-step cadence.
        assert ckpt["n_steps"] == 12
        assert len(ckpt["dead"]) == 12
        assert ckpt["num_live"] == N_LIVE
        assert ckpt["num_delete"] == 5
        assert ckpt["state"].particles.position.shape == (N_LIVE, fm.n_params)
        assert np.isfinite(float(ckpt["state"].integrator.logZ))

    def test_atomic_write_leaves_no_temp_files(self, fm, tmp_path):
        """Only the checkpoint itself remains in the directory."""
        NSSSampler(fm, max_steps=6, **CKPT_KWARGS).run(
            jax.random.PRNGKey(2), checkpoint_path=tmp_path, checkpoint_every=2
        )
        assert sorted(p.name for p in tmp_path.iterdir()) == ["nss_checkpoint.npz"]

    def test_explicit_npz_path(self, fm, tmp_path):
        """checkpoint_path may be a .npz file rather than a directory."""
        file = tmp_path / "nested" / "run.npz"
        NSSSampler(fm, max_steps=6, **CKPT_KWARGS).run(
            jax.random.PRNGKey(3), checkpoint_path=file, checkpoint_every=3
        )
        assert file.exists()
        assert _checkpoint_file(file) == file
        assert _load_checkpoint(file)["n_steps"] == 6

    def test_resume_without_checkpoint_starts_fresh(self, fm, tmp_path):
        """resume=True with nothing on disk is a no-op, not an error."""
        result = NSSSampler(fm, max_steps=6, **CKPT_KWARGS).run(
            jax.random.PRNGKey(4),
            checkpoint_path=tmp_path / "missing",
            checkpoint_every=3,
            resume=True,
        )
        assert result.n_steps == 6
        assert np.isfinite(result.logZ)

    def test_load_checkpoint_missing_returns_none(self, tmp_path):
        """_load_checkpoint reports absence rather than raising."""
        assert _load_checkpoint(tmp_path) is None

    def test_no_checkpoint_path_writes_nothing(self, fm, tmp_path):
        """Checkpointing is opt-in."""
        NSSSampler(fm, max_steps=4, **CKPT_KWARGS).run(jax.random.PRNGKey(5))
        assert list(tmp_path.iterdir()) == []

    def test_checkpoint_every_validated(self, fm, tmp_path):
        """checkpoint_every must be positive."""
        with pytest.raises(ValueError, match="checkpoint_every"):
            NSSSampler(fm, max_steps=2, **CKPT_KWARGS).run(
                jax.random.PRNGKey(6), checkpoint_path=tmp_path, checkpoint_every=0
            )


def test_package_exports():
    """NSSSampler / NSSResult exported at package level."""
    import arachne

    assert arachne.NSSSampler is NSSSampler and "NSSSampler" in arachne.__all__
    assert arachne.NSSResult is NSSResult and "NSSResult" in arachne.__all__
