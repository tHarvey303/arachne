"""Emulator wrapper that fixes one input parameter to a constant."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from arachne.emulator.base import SPSEmulator


class FixedParamEmulator(SPSEmulator):
    """Wraps an SPSEmulator, fixing one of its input parameters to a constant.

    Exposes ``inner.param_names`` minus ``fixed_param_name`` as its own
    ``param_names`` (order otherwise preserved), and re-inserts the fixed
    value at the correct column index before delegating to ``inner.predict``.
    Optionally also restricts ``inner.band_names`` to a given subset.

    Typical use: hold one emulator input (e.g. a spectroscopic redshift)
    fixed for a spatial model that has no notion of fixed parameters, such
    as ``FreeFormPixelMap`` or the legacy ``GaussianMixtureSpatialModel``.
    ``AdditiveComponentModel`` supports ``fixed_params`` and
    ``shared_param_names`` natively, so this wrapper is not needed there.
    """

    inner: SPSEmulator
    fixed_param_name: str = eqx.field(static=True)
    fixed_value: float = eqx.field(static=True)
    _param_names: tuple[str, ...] = eqx.field(static=True)
    _insert_idx: int = eqx.field(static=True)
    _band_names: tuple[str, ...] = eqx.field(static=True)
    _band_idx: tuple[int, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        inner: SPSEmulator,
        fixed_param_name: str,
        fixed_value: float,
        band_names: list[str] | None = None,
    ):
        """Wrap ``inner``, fixing ``fixed_param_name`` to ``fixed_value``.

        Args:
            inner: Emulator whose ``predict`` takes the full parameter vector.
            fixed_param_name: Name (in ``inner.param_names``) of the input to fix.
            fixed_value: Constant value substituted for that input.
            band_names: Optional subset of ``inner.band_names`` to return, in
                this order.  ``None`` keeps all bands.
        """
        if fixed_param_name not in inner.param_names:
            raise ValueError(f"{fixed_param_name!r} not in inner.param_names: {inner.param_names}")
        self.inner = inner
        self.fixed_param_name = fixed_param_name
        self.fixed_value = float(fixed_value)
        self._insert_idx = inner.param_names.index(fixed_param_name)
        self._param_names = tuple(p for p in inner.param_names if p != fixed_param_name)

        if band_names is not None:
            missing = [b for b in band_names if b not in inner.band_names]
            if missing:
                raise ValueError(f"bands not found in inner.band_names: {missing}")
            self._band_names = tuple(band_names)
            self._band_idx = tuple(inner.band_names.index(b) for b in band_names)
        else:
            self._band_names = tuple(inner.band_names)
            self._band_idx = None

    @property
    def param_names(self) -> list[str]:
        """Free parameter names (``inner.param_names`` minus the fixed one)."""
        return list(self._param_names)

    @property
    def band_names(self) -> list[str]:
        """Band names returned by ``predict`` (subset of ``inner.band_names``)."""
        return list(self._band_names)

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Free params (N, n_params-1) -> photometry (N, n_bands) nJy."""
        params = jnp.atleast_2d(params)
        n = params.shape[0]
        fixed_col = jnp.full((n, 1), self.fixed_value, dtype=params.dtype)
        i = self._insert_idx
        full = jnp.concatenate([params[:, :i], fixed_col, params[:, i:]], axis=1)
        flux = self.inner.predict(full)
        if self._band_idx is not None:
            flux = flux[:, jnp.array(self._band_idx)]
        return flux
