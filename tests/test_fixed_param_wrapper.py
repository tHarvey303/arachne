"""Tests for FixedParamEmulator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.emulator.fixed_param_wrapper import FixedParamEmulator


def test_param_names_drops_fixed(mock_emulator):
    """The fixed parameter is removed from ``param_names``; order is preserved."""
    wrapper = FixedParamEmulator(mock_emulator, "log_age", 8.5)
    assert wrapper.param_names == ["log_stellar_mass", "tau_v"]
    assert wrapper.band_names == mock_emulator.band_names


@pytest.mark.parametrize(
    "fixed_name,fixed_value",
    [
        ("log_stellar_mass", 9.0),  # index 0 -- matches the real redshift case
        ("log_age", 8.5),  # interior index
        ("tau_v", 1.2),  # last index
    ],
)
def test_predict_matches_inner_with_fixed_value_spliced_in(mock_emulator, fixed_name, fixed_value):
    """Wrapper prediction equals the inner prediction with the fixed value spliced in."""
    wrapper = FixedParamEmulator(mock_emulator, fixed_name, fixed_value)
    rng = np.random.default_rng(0)
    free = jnp.asarray(rng.normal(size=(5, 2)), dtype=jnp.float32)

    got = wrapper.predict(free)

    idx = mock_emulator.param_names.index(fixed_name)
    full = np.zeros((5, 3), dtype=np.float32)
    cols = [c for c in range(3) if c != idx]
    full[:, cols] = np.asarray(free)
    full[:, idx] = fixed_value
    expected = mock_emulator.predict(jnp.asarray(full))

    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_grad_flows_through_free_params(mock_emulator):
    """Gradients with respect to the free parameters are finite and non-zero."""
    wrapper = FixedParamEmulator(mock_emulator, "log_age", 8.5)
    free = jnp.array([1.0, 0.5])

    def loss(p):
        return jnp.sum(wrapper.predict(p[None, :]))

    grad = jax.grad(loss)(free)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.linalg.norm(grad) > 0


def test_invalid_fixed_param_name_raises(mock_emulator):
    """An unknown fixed parameter name raises ValueError."""
    with pytest.raises(ValueError, match="not_a_param"):
        FixedParamEmulator(mock_emulator, "not_a_param", 1.0)


def test_band_subsetting(mock_emulator):
    """Band subsetting returns only the requested bands in the requested order."""
    subset = mock_emulator.band_names[:2]
    wrapper = FixedParamEmulator(mock_emulator, "log_age", 8.5, band_names=subset)
    assert wrapper.band_names == subset

    free = jnp.array([[1.0, 0.5]])
    got = wrapper.predict(free)
    assert got.shape == (1, 2)

    full = jnp.array([[1.0, 8.5, 0.5]])
    expected_full = mock_emulator.predict(full)
    np.testing.assert_allclose(got, expected_full[:, :2], rtol=1e-6)


def test_invalid_band_name_raises(mock_emulator):
    """An unknown band name raises ValueError."""
    with pytest.raises(ValueError, match="not_a_band"):
        FixedParamEmulator(mock_emulator, "log_age", 8.5, band_names=["not_a_band"])
