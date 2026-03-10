"""Abstract base class for SPS emulators."""

from abc import ABC, abstractmethod

import jax.numpy as jnp


class SPSEmulator(ABC):
    """Abstract base class for stellar population synthesis emulators.

    An emulator maps per-pixel SPS parameter vectors to predicted photometry.
    All implementations must be JAX-differentiable so that gradients can
    flow through the emulator during NUTS sampling.
    """

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Names of the SPS input parameters.

        Returns:
            List of parameter name strings, e.g.
            ["log_stellar_mass", "log_age", "log_metallicity", "tau_v"].
        """
        ...

    @property
    @abstractmethod
    def band_names(self) -> list[str]:
        """Names of the photometric output bands.

        Returns:
            List of band name strings, e.g. ["JWST/NIRCam.F115W", ...].
        """
        ...

    @abstractmethod
    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Predict photometry from SPS parameters.

        Args:
            params: SPS parameter array of shape (N_pixels, N_params).
                Units follow synference conventions: log10 stellar mass (Msun),
                log10 age (yr), log10 metallicity (Z/Zsun), tau_v (mag).

        Returns:
            Predicted photometry of shape (N_pixels, N_bands) in nJy.
        """
        ...

    @property
    def n_params(self) -> int:
        """Number of SPS input parameters."""
        return len(self.param_names)

    @property
    def n_bands(self) -> int:
        """Number of photometric output bands."""
        return len(self.band_names)
