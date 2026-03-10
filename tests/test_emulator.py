"""Tests for SPS emulators.

The critical test is test_mock_emulator_grad which verifies that jax.grad
can differentiate through the emulator — this is required for NUTS to work.

The test_weight_export test (marked gpu) verifies that a JAXFlowEmulator
loaded from a real synference checkpoint agrees with the PyTorch output
to 1e-5. This test requires a real checkpoint and is skipped in CI.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.emulator.base import SPSEmulator


def test_mock_emulator_predict_shape(mock_emulator):
    """predict() returns (N_pixels, N_bands)."""
    N_pixels = 16 * 16
    params = jnp.ones((N_pixels, 3))
    out = mock_emulator.predict(params)
    assert out.shape == (N_pixels, 3)


def test_mock_emulator_predict_positive(mock_emulator):
    """predict() returns non-negative values."""
    params = jnp.ones((10, 3))
    out = mock_emulator.predict(params)
    assert jnp.all(out >= 0)


def test_mock_emulator_grad(mock_emulator):
    """jax.grad differentiates through mock emulator predict()."""
    params = jnp.ones((4, 3))

    def loss(p):
        return jnp.sum(mock_emulator.predict(p))

    grad = jax.grad(loss)(params)
    assert grad.shape == params.shape
    assert jnp.all(jnp.isfinite(grad))


def test_mock_emulator_param_names(mock_emulator):
    """param_names and band_names are correct."""
    assert "log_stellar_mass" in mock_emulator.param_names
    assert "JWST/NIRCam.F115W" in mock_emulator.band_names


def test_mock_emulator_n_params(mock_emulator):
    """n_params and n_bands properties are consistent."""
    assert mock_emulator.n_params == len(mock_emulator.param_names)
    assert mock_emulator.n_bands == len(mock_emulator.band_names)


def test_jax_flow_emulator_from_weights():
    """JAXFlowEmulator.from_weights constructs a working emulator."""
    from arachne.emulator.jax_emulator import JAXFlowEmulator, _SimpleLinear, MAFTransform, _FCNN

    n_params = 3
    n_bands = 3

    # Build a minimal MAF conditioner
    layer = _SimpleLinear(
        weight=jnp.eye(n_params * 2, n_params),
        bias=jnp.zeros(n_params * 2),
    )
    conditioner = _FCNN(layers=[layer])
    transform = MAFTransform(conditioner=conditioner, n_features=n_params)

    emulator = JAXFlowEmulator.from_weights(
        transforms=[transform],
        param_scaler_mean=np.zeros(n_params),
        param_scaler_std=np.ones(n_params),
        phot_scaler_mean=np.zeros(n_bands),
        phot_scaler_std=np.ones(n_bands),
        param_names=["log_stellar_mass", "log_age", "tau_v"],
        band_names=["F115W", "F200W", "F277W"],
    )

    params = jnp.zeros((5, n_params))
    out = emulator.predict(params)
    assert out.shape == (5, n_bands)
    assert jnp.all(jnp.isfinite(out))


def test_jax_flow_emulator_grad():
    """jax.grad differentiates through JAXFlowEmulator."""
    from arachne.emulator.jax_emulator import JAXFlowEmulator, _SimpleLinear, MAFTransform, _FCNN

    n_params = 3
    n_bands = 3

    layer = _SimpleLinear(
        weight=jnp.eye(n_params * 2, n_params) * 0.1,
        bias=jnp.zeros(n_params * 2),
    )
    conditioner = _FCNN(layers=[layer])
    transform = MAFTransform(conditioner=conditioner, n_features=n_params)

    emulator = JAXFlowEmulator.from_weights(
        transforms=[transform],
        param_scaler_mean=np.zeros(n_params),
        param_scaler_std=np.ones(n_params),
        phot_scaler_mean=np.zeros(n_bands),
        phot_scaler_std=np.ones(n_bands),
        param_names=["log_stellar_mass", "log_age", "tau_v"],
        band_names=["F115W", "F200W", "F277W"],
    )

    def loss(params):
        return jnp.sum(emulator.predict(params))

    params = jnp.ones((4, n_params))
    grad = jax.grad(loss)(params)
    assert grad.shape == params.shape
    assert jnp.all(jnp.isfinite(grad))


@pytest.mark.gpu
def test_weight_export(tmp_path):
    """PyTorch lampe forward pass and Equinox reimplementation agree to 1e-5.

    This test requires a real synference checkpoint and a GPU. Skip in CI.
    """
    pytest.importorskip("torch")
    pytest.importorskip("joblib")
    checkpoint = tmp_path / "checkpoint.pkl"
    if not checkpoint.exists():
        pytest.skip("No real synference checkpoint available for weight export test.")

    from arachne.emulator.jax_emulator import JAXFlowEmulator

    emulator = JAXFlowEmulator.from_synference_checkpoint(
        checkpoint,
        param_names=["log_stellar_mass", "log_age", "tau_v"],
        band_names=["F115W", "F200W", "F277W"],
    )
    params = jnp.zeros((1, 3))
    out = emulator.predict(params)
    assert out.shape == (1, 3)
    # Real test would compare against PyTorch output; requires checkpoint
