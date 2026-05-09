"""Tests for SPSMLPEmulator (native JAX/Equinox MLP emulator)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.emulator.jax_mlp_emulator import AlsingLayer, SPSMLPEmulator

N_PARAMS = 3
N_BANDS = 3
PARAM_NAMES = ["log_stellar_mass", "log_age", "tau_v"]
BAND_NAMES = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]


@pytest.fixture
def tiny_emulator():
    """Small SPSMLPEmulator with 2 hidden layers of width 16."""
    return SPSMLPEmulator(
        param_names=PARAM_NAMES,
        band_names=BAND_NAMES,
        hidden_sizes=[16, 16],
        in_mean=np.zeros(N_PARAMS),
        in_std=np.ones(N_PARAMS),
        out_mean=np.zeros(N_BANDS),
        out_std=np.ones(N_BANDS),
        key=jax.random.PRNGKey(0),
    )


# ---------------------------------------------------------------------------
# AlsingLayer tests
# ---------------------------------------------------------------------------


class TestAlsingLayer:
    """Tests for the Alsing activation layer."""

    def test_output_shape(self):
        """AlsingLayer output has correct shape."""
        layer = AlsingLayer(8, 16, jax.random.PRNGKey(0))
        x = jnp.ones(8)
        out = layer(x)
        assert out.shape == (16,)

    def test_batched_output_shape(self):
        """AlsingLayer with vmap produces (N, out) shape."""
        layer = AlsingLayer(8, 16, jax.random.PRNGKey(0))
        x = jnp.ones((10, 8))
        out = jax.vmap(layer)(x)
        assert out.shape == (10, 16)

    def test_output_finite(self):
        """AlsingLayer output is finite for zero input."""
        layer = AlsingLayer(4, 8, jax.random.PRNGKey(1))
        x = jnp.zeros(4)
        out = layer(x)
        assert jnp.all(jnp.isfinite(out))

    def test_differentiable(self):
        """jax.grad passes through AlsingLayer."""
        layer = AlsingLayer(4, 4, jax.random.PRNGKey(2))

        def loss(x):
            return jnp.sum(layer(x))

        grad = jax.grad(loss)(jnp.zeros(4))
        assert grad.shape == (4,)
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# SPSMLPEmulator tests
# ---------------------------------------------------------------------------


class TestSPSMLPEmulator:
    """Tests for SPSMLPEmulator."""

    def test_predict_shape(self, tiny_emulator):
        """predict() returns (N_pixels, N_bands)."""
        params = jnp.ones((16 * 16, N_PARAMS))
        out = tiny_emulator.predict(params)
        assert out.shape == (16 * 16, N_BANDS)

    def test_predict_positive(self, tiny_emulator):
        """predict() returns positive flux values (10^x > 0)."""
        params = jnp.zeros((10, N_PARAMS))
        out = tiny_emulator.predict(params)
        assert jnp.all(out > 0)

    def test_predict_finite(self, tiny_emulator):
        """predict() returns finite values."""
        params = jnp.zeros((10, N_PARAMS))
        out = tiny_emulator.predict(params)
        assert jnp.all(jnp.isfinite(out))

    def test_predict_grad(self, tiny_emulator):
        """jax.grad differentiates through predict()."""

        def loss(params):
            return jnp.sum(tiny_emulator.predict(params))

        params = jnp.ones((4, N_PARAMS))
        grad = jax.grad(loss)(params)
        assert grad.shape == params.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_predict_jit(self, tiny_emulator):
        """eqx.filter_jit compiles predict() (correct Equinox JIT pattern)."""
        import equinox as eqx

        predict_jit = eqx.filter_jit(tiny_emulator.predict)
        params = jnp.zeros((8, N_PARAMS))
        out = predict_jit(params)
        assert out.shape == (8, N_BANDS)

    def test_param_names(self, tiny_emulator):
        """param_names and band_names are correct."""
        assert tiny_emulator.param_names == PARAM_NAMES
        assert tiny_emulator.band_names == BAND_NAMES

    def test_n_params_n_bands(self, tiny_emulator):
        """n_params and n_bands match name list lengths."""
        assert tiny_emulator.n_params == N_PARAMS
        assert tiny_emulator.n_bands == N_BANDS

    def test_save_load_roundtrip(self, tiny_emulator, tmp_path):
        """Emulator saved and loaded produces identical predictions."""
        path = tmp_path / "emulator.eqx"
        tiny_emulator.save(str(path))
        assert path.exists()

        loaded = SPSMLPEmulator.load(
            str(path),
            param_names=PARAM_NAMES,
            band_names=BAND_NAMES,
            hidden_sizes=[16, 16],
        )

        params = jnp.zeros((5, N_PARAMS))
        out_orig = tiny_emulator.predict(params)
        out_loaded = loaded.predict(params)
        np.testing.assert_allclose(np.asarray(out_orig), np.asarray(out_loaded), rtol=1e-5)

    def test_normalisation_applied(self, tiny_emulator):
        """Changing in_mean shifts the predictions (normalisation is active)."""
        params = jnp.ones((4, N_PARAMS))
        out1 = tiny_emulator.predict(params)

        # Build a copy with different in_mean
        shifted = SPSMLPEmulator(
            param_names=PARAM_NAMES,
            band_names=BAND_NAMES,
            hidden_sizes=[16, 16],
            in_mean=np.ones(N_PARAMS) * 2.0,  # shifted
            in_std=np.ones(N_PARAMS),
            out_mean=np.zeros(N_BANDS),
            out_std=np.ones(N_BANDS),
            key=jax.random.PRNGKey(0),
        )
        out2 = shifted.predict(params)
        # Predictions should differ because the normalised input differs
        assert not jnp.allclose(out1, out2)


# ---------------------------------------------------------------------------
# from_synference_library test (uses synthetic HDF5)
# ---------------------------------------------------------------------------


def _make_synthetic_library(path: Path, n_models: int = 500) -> None:
    """Create a minimal synthetic synference-format HDF5 library for testing."""
    import h5py

    rng = np.random.default_rng(42)
    params = rng.uniform(0, 1, (N_PARAMS, n_models)).astype(np.float32)
    # Photometry: simple linear function of params so the network can fit it
    phot = (10.0 ** (params[0] * 2 + 5)).reshape(1, n_models) * np.ones(
        (N_BANDS, n_models), dtype=np.float32
    )
    phot = phot.astype(np.float32)

    with h5py.File(path, "w") as f:
        g = f.create_group("Grid")
        g.create_dataset("Parameters", data=params)
        g.create_dataset("Photometry", data=phot)
        f.attrs["ParameterNames"] = np.bytes_(PARAM_NAMES)
        f.attrs["FilterCodes"] = np.bytes_(BAND_NAMES)
        f.attrs["PhotometryUnits"] = "nJy"


def test_from_synference_library(tmp_path):
    """from_synference_library trains on a synthetic HDF5 and returns an emulator."""
    pytest.importorskip("h5py")
    pytest.importorskip("optax")

    lib_path = tmp_path / "library.h5"
    _make_synthetic_library(lib_path, n_models=200)

    emulator = SPSMLPEmulator.from_synference_library(
        library_path=str(lib_path),
        param_names=PARAM_NAMES,
        band_names=BAND_NAMES,
        hidden_sizes=[16, 16],
        n_epochs=5,
        batch_size=64,
        log_interval=5,
    )

    assert isinstance(emulator, SPSMLPEmulator)
    params = jnp.zeros((4, N_PARAMS))
    out = emulator.predict(params)
    assert out.shape == (4, N_BANDS)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(out > 0)


def test_from_synference_library_missing_param_raises(tmp_path):
    """from_synference_library raises KeyError for unknown param name."""
    pytest.importorskip("h5py")
    pytest.importorskip("optax")

    lib_path = tmp_path / "library.h5"
    _make_synthetic_library(lib_path)

    with pytest.raises(KeyError, match="not_a_param"):
        SPSMLPEmulator.from_synference_library(
            library_path=str(lib_path),
            param_names=["not_a_param"],
            band_names=BAND_NAMES,
        )


def test_output_layer_key_independent_of_n_hidden():
    """Output layer uses a distinct PRNGKey from every hidden layer.

    Regression test for fix 3: the original __init__ computed the output-layer
    key as ``jax.random.split(key, 1)[0]``, which equals ``keys[0]`` (the key
    used for hidden_layers[0]) regardless of n_hidden.  With the fix, the
    output layer gets ``keys[-1]`` from a split of size ``n_hidden + 1``, so
    it changes as n_hidden changes.

    Two emulators with the same seed but different numbers of hidden layers
    must have different output-layer weight matrices — under the old code they
    would be identical.
    """
    key = jax.random.PRNGKey(0)
    common_kwargs = dict(
        param_names=PARAM_NAMES,
        band_names=BAND_NAMES,
        in_mean=np.zeros(N_PARAMS),
        in_std=np.ones(N_PARAMS),
        out_mean=np.zeros(N_BANDS),
        out_std=np.ones(N_BANDS),
        key=key,
    )
    em2 = SPSMLPEmulator(hidden_sizes=[16, 16], **common_kwargs)
    em3 = SPSMLPEmulator(hidden_sizes=[16, 16, 16], **common_kwargs)

    assert not jnp.allclose(em2.output_layer.weight, em3.output_layer.weight), (
        "Output layer weights are identical for 2 vs 3 hidden layers with the same key. "
        "This indicates the key-collision bug (fix 3) has been reintroduced."
    )


def test_from_synference_library_grad(tmp_path):
    """Emulator trained from library is differentiable."""
    pytest.importorskip("h5py")
    pytest.importorskip("optax")

    lib_path = tmp_path / "library.h5"
    _make_synthetic_library(lib_path, n_models=200)

    emulator = SPSMLPEmulator.from_synference_library(
        library_path=str(lib_path),
        param_names=PARAM_NAMES,
        band_names=BAND_NAMES,
        hidden_sizes=[16, 16],
        n_epochs=2,
        batch_size=64,
    )

    def loss(params):
        return jnp.sum(emulator.predict(params))

    params = jnp.ones((4, N_PARAMS))
    grad = jax.grad(loss)(params)
    assert jnp.all(jnp.isfinite(grad))
