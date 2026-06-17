"""Tests for ParrotEmulator (Parrot-style Parrot MLP with arsinh output transform)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.emulator.parrot_emulator import (
    ParrotEmulator,
    _flux_to_asinh_mag_np,
    _read_band_names,
    _read_param_names,
    asinh_mag_to_flux,
    flux_to_asinh_mag,
)

N_PARAMS = 4
N_BANDS = 5
PARAM_NAMES = ["redshift", "log10metallicity", "Av", "tau_v"]
BAND_NAMES = [
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F444W",
]


@pytest.fixture
def tiny_emulator():
    """Small ParrotEmulator with 2 hidden layers of width 16."""
    return ParrotEmulator(
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
# arsinh transform tests
# ---------------------------------------------------------------------------


class TestArsinhTransform:
    """Tests for the flux ↔ arsinh-magnitude transforms."""

    def test_roundtrip_positive_flux(self):
        """flux_to_asinh_mag ∘ asinh_mag_to_flux is identity for positive flux."""
        flux = jnp.array([1e-5, 1.0, 100.0, 1e6, 1e10])
        recovered = asinh_mag_to_flux(flux_to_asinh_mag(flux))
        np.testing.assert_allclose(np.asarray(recovered), np.asarray(flux), rtol=1e-5)

    def test_zero_flux_finite(self):
        """Zero flux maps to a finite arsinh magnitude (no divergence)."""
        mag = flux_to_asinh_mag(jnp.array([0.0]))
        assert jnp.all(jnp.isfinite(mag))

    def test_negative_flux_finite(self):
        """Negative (unphysical) flux also maps to finite arsinh magnitude."""
        mag = flux_to_asinh_mag(jnp.array([-1.0, -100.0]))
        assert jnp.all(jnp.isfinite(mag))

    def test_very_small_flux(self):
        """Near-zero flux (e.g. Lyman-break dropout 1e-50 nJy) is handled."""
        flux = jnp.array([1e-50])
        mag = flux_to_asinh_mag(flux)
        assert jnp.all(jnp.isfinite(mag))

    def test_numpy_asinh_matches_jax(self):
        """Numpy and JAX arsinh transforms agree."""
        flux_np = np.array([0.0, 1.0, 1e6], dtype=np.float32)
        flux_jax = jnp.array(flux_np)
        mag_np = _flux_to_asinh_mag_np(flux_np)
        mag_jax = np.asarray(flux_to_asinh_mag(flux_jax))
        # float32: flux=1 nJy maps to asinh-mag ≈ 0 by construction; allow small abs error
        np.testing.assert_allclose(mag_np, mag_jax, atol=1e-4)

    def test_differentiable(self):
        """jax.grad passes through the arsinh transform."""

        def loss(flux):
            return jnp.sum(flux_to_asinh_mag(flux))

        grad = jax.grad(loss)(jnp.ones(5))
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# ParrotEmulator unit tests
# ---------------------------------------------------------------------------


class TestParrotEmulator:
    """Tests for the ParrotEmulator class."""

    def test_predict_shape(self, tiny_emulator):
        """predict() returns (N_pixels, N_bands)."""
        params = jnp.ones((16 * 16, N_PARAMS))
        out = tiny_emulator.predict(params)
        assert out.shape == (16 * 16, N_BANDS)

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
        """eqx.filter_jit compiles predict()."""
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

        loaded = ParrotEmulator.load(str(path))

        params = jnp.zeros((5, N_PARAMS))
        out_orig = tiny_emulator.predict(params)
        out_loaded = loaded.predict(params)
        np.testing.assert_allclose(np.asarray(out_orig), np.asarray(out_loaded), rtol=1e-5)

    def test_zero_input_finite(self, tiny_emulator):
        """Zero-valued inputs produce finite predictions."""
        params = jnp.zeros((4, N_PARAMS))
        out = tiny_emulator.predict(params)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# from_synference_library with synthetic v4-format HDF5
# ---------------------------------------------------------------------------


def _make_synthetic_v4_library(path: Path, n_models: int = 500) -> None:
    """Create a minimal synthetic synference v4-format HDF5 library.

    Uses the ``Model.attrs`` metadata format (``varying_param_names``,
    ``stellar_params``, ``filters``) rather than root-level attrs.
    """
    import h5py

    rng = np.random.default_rng(42)
    params_np = rng.uniform(0, 1, (N_PARAMS, n_models)).astype(np.float32)
    # Simple test signal: photometry scales with first parameter
    phot_np = (10.0 ** (params_np[0] * 2 + 3)).reshape(1, n_models) * np.ones(
        (N_BANDS, n_models), dtype=np.float32
    )

    varying = PARAM_NAMES[:3]  # first 3 = varying
    stellar = PARAM_NAMES[3:]  # last 1 = stellar

    with h5py.File(path, "w") as f:
        g = f.create_group("Grid")
        g.create_dataset("Parameters", data=params_np)
        g.create_dataset("Photometry", data=phot_np)

        m = f.create_group("Model")
        m.attrs["varying_param_names"] = np.array(varying, dtype="S")
        m.attrs["stellar_params"] = np.array(stellar, dtype="S")
        m.attrs["filters"] = np.array(BAND_NAMES, dtype="S")


def _make_synthetic_legacy_library(path: Path, n_models: int = 500) -> None:
    """Create a minimal synthetic synference legacy-format HDF5 library.

    Uses root-level ``ParameterNames`` / ``FilterCodes`` attrs.
    """
    import h5py

    rng = np.random.default_rng(42)
    params_np = rng.uniform(0, 1, (N_PARAMS, n_models)).astype(np.float32)
    phot_np = (10.0 ** (params_np[0] * 2 + 3)).reshape(1, n_models) * np.ones(
        (N_BANDS, n_models), dtype=np.float32
    )

    with h5py.File(path, "w") as f:
        g = f.create_group("Grid")
        g.create_dataset("Parameters", data=params_np)
        g.create_dataset("Photometry", data=phot_np)
        f.attrs["ParameterNames"] = np.array(PARAM_NAMES, dtype="S")
        f.attrs["FilterCodes"] = np.array(BAND_NAMES, dtype="S")


class TestFromSynerenceLibrary:
    """Tests for ParrotEmulator.from_synference_library."""

    @pytest.fixture(autouse=True)
    def _skip_deps(self):
        pytest.importorskip("h5py")
        pytest.importorskip("optax")

    def test_trains_on_v4_library(self, tmp_path):
        """from_synference_library trains on v4-format HDF5."""
        lib_path = tmp_path / "library_v4.h5"
        _make_synthetic_v4_library(lib_path, n_models=300)

        emulator = ParrotEmulator.from_synference_library(
            library_path=str(lib_path),
            param_names=PARAM_NAMES,
            band_names=BAND_NAMES,
            hidden_sizes=[16, 16],
            n_epochs=3,
            batch_size=64,
            early_stopping_patience=999,
            log_interval=1,
        )

        assert isinstance(emulator, ParrotEmulator)
        out = emulator.predict(jnp.zeros((4, N_PARAMS)))
        assert out.shape == (4, N_BANDS)
        assert jnp.all(jnp.isfinite(out))

    def test_trains_on_legacy_library(self, tmp_path):
        """from_synference_library trains on legacy-format HDF5."""
        lib_path = tmp_path / "library_legacy.h5"
        _make_synthetic_legacy_library(lib_path, n_models=300)

        emulator = ParrotEmulator.from_synference_library(
            library_path=str(lib_path),
            param_names=PARAM_NAMES,
            band_names=BAND_NAMES,
            hidden_sizes=[16, 16],
            n_epochs=3,
            batch_size=64,
            early_stopping_patience=999,
            log_interval=1,
        )

        assert isinstance(emulator, ParrotEmulator)

    def test_missing_param_raises(self, tmp_path):
        """KeyError raised for an unknown parameter name."""
        lib_path = tmp_path / "library.h5"
        _make_synthetic_v4_library(lib_path)

        with pytest.raises(KeyError, match="not_a_param"):
            ParrotEmulator.from_synference_library(
                library_path=str(lib_path),
                param_names=["not_a_param"],
                band_names=BAND_NAMES,
                n_epochs=1,
            )

    def test_missing_band_raises(self, tmp_path):
        """KeyError raised for an unknown band name."""
        lib_path = tmp_path / "library.h5"
        _make_synthetic_v4_library(lib_path)

        with pytest.raises(KeyError, match="JWST/MIRI.F560W"):
            ParrotEmulator.from_synference_library(
                library_path=str(lib_path),
                param_names=PARAM_NAMES,
                band_names=["JWST/MIRI.F560W"],
                n_epochs=1,
            )

    def test_gradient_after_training(self, tmp_path):
        """Emulator trained from library is differentiable."""
        lib_path = tmp_path / "library.h5"
        _make_synthetic_v4_library(lib_path, n_models=300)

        emulator = ParrotEmulator.from_synference_library(
            library_path=str(lib_path),
            param_names=PARAM_NAMES,
            band_names=BAND_NAMES,
            hidden_sizes=[16, 16],
            n_epochs=2,
            batch_size=64,
            early_stopping_patience=999,
        )

        def loss(params):
            return jnp.sum(emulator.predict(params))

        grad = jax.grad(loss)(jnp.ones((4, N_PARAMS)))
        assert jnp.all(jnp.isfinite(grad))

    def test_checkpoint_saved(self, tmp_path):
        """Safety checkpoint is written during training."""
        lib_path = tmp_path / "library.h5"
        _make_synthetic_v4_library(lib_path, n_models=300)
        ckpt = tmp_path / "best.eqx"

        ParrotEmulator.from_synference_library(
            library_path=str(lib_path),
            param_names=PARAM_NAMES,
            band_names=BAND_NAMES,
            hidden_sizes=[16, 16],
            n_epochs=3,
            batch_size=64,
            early_stopping_patience=999,
            checkpoint_path=str(ckpt),
        )

        assert ckpt.exists()

    def test_early_stopping(self, tmp_path):
        """Training stops early when val loss does not improve."""
        lib_path = tmp_path / "library.h5"
        _make_synthetic_v4_library(lib_path, n_models=300)

        # patience=1 ensures early stopping triggers quickly
        ParrotEmulator.from_synference_library(
            library_path=str(lib_path),
            param_names=PARAM_NAMES,
            band_names=BAND_NAMES,
            hidden_sizes=[16, 16],
            n_epochs=100,
            batch_size=64,
            early_stopping_patience=1,
            log_interval=100,
        )
        # If we get here without hanging, early stopping worked.


# ---------------------------------------------------------------------------
# Metadata reader tests
# ---------------------------------------------------------------------------


class TestMetadataReaders:
    """Tests for _read_param_names and _read_band_names."""

    @pytest.fixture(autouse=True)
    def _skip_h5py(self):
        pytest.importorskip("h5py")

    def test_v4_param_names(self, tmp_path):
        """_read_param_names reads v4-format varying + stellar params."""
        import h5py

        path = tmp_path / "lib.h5"
        with h5py.File(path, "w") as f:
            m = f.create_group("Model")
            m.attrs["varying_param_names"] = np.array(["redshift", "Av"], dtype="S")
            m.attrs["stellar_params"] = np.array(["tau_v"], dtype="S")

        with h5py.File(path, "r") as f:
            names = _read_param_names(f)

        assert names == ["redshift", "Av", "tau_v"]

    def test_v4_band_names(self, tmp_path):
        """_read_band_names reads v4-format Model.attrs['filters']."""
        import h5py

        path = tmp_path / "lib.h5"
        with h5py.File(path, "w") as f:
            m = f.create_group("Model")
            m.attrs["filters"] = np.array(["JWST/NIRCam.F200W", "HST/ACS_WFC.F814W"], dtype="S")

        with h5py.File(path, "r") as f:
            bands = _read_band_names(f)

        assert bands == ["JWST/NIRCam.F200W", "HST/ACS_WFC.F814W"]

    def test_legacy_param_names(self, tmp_path):
        """_read_param_names falls back to root ParameterNames attr."""
        import h5py

        path = tmp_path / "lib.h5"
        with h5py.File(path, "w") as f:
            f.attrs["ParameterNames"] = np.array(["log_stellar_mass", "tau_v"], dtype="S")

        with h5py.File(path, "r") as f:
            names = _read_param_names(f)

        assert names == ["log_stellar_mass", "tau_v"]

    def test_legacy_band_names(self, tmp_path):
        """_read_band_names falls back to root FilterCodes attr."""
        import h5py

        path = tmp_path / "lib.h5"
        with h5py.File(path, "w") as f:
            f.attrs["FilterCodes"] = np.array(["JWST/NIRCam.F115W"], dtype="S")

        with h5py.File(path, "r") as f:
            bands = _read_band_names(f)

        assert bands == ["JWST/NIRCam.F115W"]
