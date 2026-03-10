"""Tests for ObservationCube."""

import numpy as np
import pytest

from arachne.data.observation import ObservationCube


def test_observation_shapes(tiny_observation):
    """ObservationCube has consistent array shapes."""
    obs = tiny_observation
    assert obs.flux.shape == (3, 16, 16)
    assert obs.variance.shape == (3, 16, 16)
    assert obs.mask.shape == (3, 16, 16)
    assert len(obs.band_names) == 3


def test_observation_n_bands(tiny_observation):
    """n_bands property returns correct count."""
    assert tiny_observation.n_bands == 3


def test_observation_image_shape(tiny_observation):
    """image_shape property returns (H, W)."""
    assert tiny_observation.image_shape == (16, 16)


def test_to_jax_returns_jax_arrays(tiny_observation_numpy):
    """to_jax() converts numpy arrays to JAX arrays."""
    import jax.numpy as jnp

    obs_jax = tiny_observation_numpy.to_jax()
    assert isinstance(obs_jax.flux, jnp.ndarray)
    assert isinstance(obs_jax.variance, jnp.ndarray)
    assert isinstance(obs_jax.mask, jnp.ndarray)


def test_to_jax_preserves_values(tiny_observation_numpy):
    """to_jax() does not change array values."""
    import numpy as np

    obs = tiny_observation_numpy
    obs_jax = obs.to_jax()
    np.testing.assert_allclose(np.asarray(obs_jax.flux), obs.flux, rtol=1e-6)


def test_band_names_preserved(tiny_observation):
    """Band names list is preserved."""
    assert tiny_observation.band_names[0] == "JWST/NIRCam.F115W"


def test_shape_mismatch_raises():
    """Mismatched flux/variance shapes raise ValueError."""
    flux = np.ones((3, 16, 16), dtype=np.float32)
    variance = np.ones((3, 8, 8), dtype=np.float32)
    mask = np.ones((3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="variance"):
        ObservationCube(
            flux=flux,
            variance=variance,
            mask=mask,
            band_names=["a", "b", "c"],
            pixel_scale=0.031,
        )


def test_band_name_count_mismatch_raises():
    """Mismatched band_names count raises ValueError."""
    flux = np.ones((3, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="band names"):
        ObservationCube(
            flux=flux,
            variance=flux.copy(),
            mask=flux.copy(),
            band_names=["a", "b"],  # Only 2, should be 3
            pixel_scale=0.031,
        )


def test_mask_default_all_valid():
    """When mask is all ones, all pixels are valid."""
    flux = np.ones((2, 4, 4), dtype=np.float32)
    obs = ObservationCube(
        flux=flux,
        variance=flux.copy(),
        mask=np.ones_like(flux),
        band_names=["F115W", "F200W"],
        pixel_scale=0.031,
    )
    import jax.numpy as jnp

    obs_jax = obs.to_jax()
    assert jnp.all(obs_jax.mask == 1.0)
