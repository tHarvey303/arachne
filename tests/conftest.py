"""Shared pytest fixtures for arachne tests.

All fixtures use small synthetic arrays (3 bands, 16×16 pixels) so the
test suite runs on CPU without any GPU or real checkpoint.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.spatial.gmm import GaussianMixtureSpatialModel
from arachne.spatial.pixel_map import FreeFormPixelMap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BANDS = 3
H = 16
W = 16
N_SPS = 3
BAND_NAMES = ["JWST/NIRCam.F115W", "JWST/NIRCam.F200W", "JWST/NIRCam.F277W"]
SPS_PARAM_NAMES = ["log_stellar_mass", "log_age", "tau_v"]
PARAM_BOUNDS = {
    "log_stellar_mass": (6.0, 12.0),
    "log_age": (7.0, 10.1),
    "tau_v": (0.0, 4.0),
}

# ---------------------------------------------------------------------------
# Observation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_observation() -> ObservationCube:
    """3-band 16×16 synthetic ObservationCube with JAX arrays."""
    rng = np.random.default_rng(42)
    flux = rng.normal(10.0, 1.0, size=(N_BANDS, H, W)).astype(np.float32)
    variance = np.full((N_BANDS, H, W), 1.0, dtype=np.float32)
    mask = np.ones((N_BANDS, H, W), dtype=np.float32)
    obs = ObservationCube(
        flux=flux,
        variance=variance,
        mask=mask,
        band_names=BAND_NAMES,
        pixel_scale=0.031,
        wcs=None,
    )
    return obs.to_jax()


@pytest.fixture
def tiny_observation_numpy() -> ObservationCube:
    """3-band 16×16 synthetic ObservationCube with numpy arrays (pre-to_jax)."""
    rng = np.random.default_rng(42)
    flux = rng.normal(10.0, 1.0, size=(N_BANDS, H, W)).astype(np.float32)
    variance = np.full((N_BANDS, H, W), 1.0, dtype=np.float32)
    mask = np.ones((N_BANDS, H, W), dtype=np.float32)
    return ObservationCube(
        flux=flux,
        variance=variance,
        mask=mask,
        band_names=BAND_NAMES,
        pixel_scale=0.031,
        wcs=None,
    )


# ---------------------------------------------------------------------------
# PSF fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian_psf() -> PSFModel:
    """Analytical 9×9 Gaussian PSF kernels, one per band."""
    kernels = []
    for sigma in [1.0, 1.2, 1.5]:  # slightly different per band
        y, x = np.mgrid[-4:5, -4:5]
        k = np.exp(-(x**2 + y**2) / (2 * sigma**2)).astype(np.float32)
        k /= k.sum()
        kernels.append(k)
    return PSFModel(
        kernels=np.stack(kernels, axis=0),
        band_names=BAND_NAMES,
    )


@pytest.fixture
def delta_psf() -> PSFModel:
    """Delta-function (identity) PSF: 9×9 kernel with 1 at centre, 0 elsewhere."""
    kernel = np.zeros((9, 9), dtype=np.float32)
    kernel[4, 4] = 1.0
    kernels = np.stack([kernel] * N_BANDS, axis=0)
    return PSFModel(kernels=kernels, band_names=BAND_NAMES)


# ---------------------------------------------------------------------------
# Mock emulator
# ---------------------------------------------------------------------------


class _MockTransform(eqx.Module):
    """Identity-like transform for testing: inverse(z) = z * stellar_mass_scale."""

    scale: jnp.ndarray

    def inverse(self, z: jnp.ndarray) -> jnp.ndarray:
        """Scale the first element and return all as photometry proxy."""
        return jnp.ones(N_BANDS) * jnp.abs(z[0]) * self.scale


class MockEmulator(SPSEmulator, eqx.Module):
    """Mock emulator that returns ones scaled by stellar mass.

    predict(params) = |log_stellar_mass| * ones(N_bands)

    This is trivially differentiable and sufficient for testing the full
    pipeline without a real normalising flow checkpoint.
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
        """Return flux proportional to abs(log_stellar_mass).

        Args:
            params: (N_pixels, N_params) array.

        Returns:
            (N_pixels, N_bands) flux array in arbitrary units.
        """
        # Use the first parameter (log_stellar_mass) as a proxy
        scale = jnp.abs(params[:, 0:1]) + 1.0  # (N_pixels, 1)
        return jnp.broadcast_to(scale, (params.shape[0], len(self._band_names)))


@pytest.fixture
def mock_emulator() -> MockEmulator:
    """MockEmulator with 3 SPS params and 3 bands."""
    return MockEmulator(_param_names=SPS_PARAM_NAMES, _band_names=BAND_NAMES)


# ---------------------------------------------------------------------------
# Spatial model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gmm_model() -> GaussianMixtureSpatialModel:
    """K=2 GMM spatial model for a 16×16 image with 3 SPS params."""
    return GaussianMixtureSpatialModel(
        n_components=2,
        sps_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        image_shape=(H, W),
    )


@pytest.fixture
def pixel_map_model() -> FreeFormPixelMap:
    """FreeFormPixelMap for a 16×16 image with 3 SPS params."""
    return FreeFormPixelMap(
        image_shape=(H, W),
        sps_param_names=SPS_PARAM_NAMES,
        param_bounds=PARAM_BOUNDS,
        smoothness_strength=0.1,
    )
