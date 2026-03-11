"""Abstract base class for SPS emulators."""

import equinox as eqx
import jax.numpy as jnp


class SPSEmulator(eqx.Module):
    """Abstract base class for stellar population synthesis emulators.

    Inherits from ``eqx.Module`` so that all concrete subclasses are
    Equinox pytrees without a metaclass conflict.  Abstract methods raise
    ``NotImplementedError`` at runtime rather than at class-definition time.

    An emulator maps per-pixel SPS parameter vectors to predicted photometry.
    All implementations must be JAX-differentiable so that gradients can
    flow through the emulator during NUTS sampling.
    """

    @property
    def param_names(self) -> list[str]:
        """Names of the SPS input parameters.

        Returns:
            List of parameter name strings, e.g.
            ["log_stellar_mass", "log_age", "log_metallicity", "tau_v"].
        """
        raise NotImplementedError

    @property
    def band_names(self) -> list[str]:
        """Names of the photometric output bands.

        Returns:
            List of band name strings, e.g. ["JWST/NIRCam.F115W", ...].
        """
        raise NotImplementedError

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Predict photometry from SPS parameters.

        Args:
            params: SPS parameter array of shape (N_pixels, N_params).
                Units follow synference conventions: log10 stellar mass (Msun),
                log10 age (yr), log10 metallicity (Z/Zsun), tau_v (mag).

        Returns:
            Predicted photometry of shape (N_pixels, N_bands) in nJy.
        """
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        """Number of SPS input parameters."""
        return len(self.param_names)

    @property
    def n_bands(self) -> int:
        """Number of photometric output bands."""
        return len(self.band_names)
