"""Present an emulator under a chosen parameter list and band list.

Different libraries give emulators different inputs (the v6 library has 12,
the v7 library 17, with ``slope`` renamed ``dust_slope``).  A fit usually
wants a fixed, smaller set.  :class:`EmulatorView` wraps any
:class:`~arachne.emulator.base.SPSEmulator` and exposes exactly the inputs and
bands asked for: the remaining inputs are held at constants, inputs can be
renamed and reordered, and bands can be selected.  The view is itself an
``SPSEmulator`` with a matching :attr:`spec`, so every model and sampler takes
it unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp

from arachne.emulator.base import SPSEmulator
from arachne.emulator.spec import EmulatorSpec, resolve_param_name
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

__all__ = ["EmulatorView"]


class EmulatorView(SPSEmulator):
    """An emulator with some inputs fixed, renamed or reordered, and some bands selected.

    Example:
        Use a 17-input v7 checkpoint in code written for the 12 v6 inputs::

            view = EmulatorView(
                emulator,
                params=V6_NAMES,
                bands=my_bands,
            )
            view.spec.meta[
                "fixed"
            ]  # {"fesc": 0.1, "f_birth": 1.0, ...}

        ``slope`` is matched to ``dust_slope`` through the alias table, and the
        five inputs the v6 list lacks are held at the emulator's defaults.
    """

    inner: SPSEmulator
    _param_names: tuple[str, ...] = eqx.field(static=True)
    _band_names: tuple[str, ...] = eqx.field(static=True)
    _source: tuple[int, ...] = eqx.field(static=True)  # per inner input: view column, or -1
    _constant: tuple[float, ...] = eqx.field(static=True)  # per inner input: fixed value (or 0)
    _band_idx: tuple[int, ...] | None = eqx.field(static=True)
    _fixed: tuple[tuple[str, float], ...] = eqx.field(static=True)
    _renamed: tuple[tuple[str, str], ...] = eqx.field(static=True)  # (inner name, view name)

    def __init__(
        self,
        inner: SPSEmulator,
        params: Sequence[str] | None = None,
        fixed: Mapping[str, float] | None = None,
        bands: Sequence[str] | None = None,
        rename: Mapping[str, str] | None = None,
        fill_defaults: bool = True,
    ):
        """Wrap ``inner``.

        Args:
            inner: The emulator to present.
            params: Input names the view exposes, in this order.  Each must be
                an input of ``inner``, a known alias of one, or a value of
                ``rename``.  ``None``: every input of ``inner`` that is not fixed.
            fixed: ``{name: value}`` inputs of ``inner`` held constant (names as
                ``inner`` spells them, or aliases).
            bands: Bands to return, in this order.  ``None`` keeps all.
            rename: ``{inner name: view name}`` for names the alias table does
                not cover.
            fill_defaults: Hold every input of ``inner`` that is neither in
                ``params`` nor in ``fixed`` at ``inner.spec.defaults`` (logged).
                If ``False`` such inputs raise.

        Raises:
            ValueError: On a name that cannot be matched, an input that is both
                exposed and fixed, an input left over with ``fill_defaults=False``
                or without a default, or an unknown band.
        """
        inner_names = list(inner.param_names)
        spec = inner.spec
        rename = dict(rename or {})
        unknown = [k for k in rename if k not in inner_names]
        if unknown:
            raise ValueError(f"rename: not emulator parameters: {unknown}; have {inner_names}")
        view_to_inner = {v: k for k, v in rename.items()}

        fixed_inner: dict[str, float] = {}
        for name, value in (fixed or {}).items():
            match = resolve_param_name(name, inner_names)
            if match is None:
                raise ValueError(
                    f"fixed: {name!r} is not an emulator parameter; have {inner_names}"
                )
            fixed_inner[match] = float(value)

        if params is None:
            params = [rename.get(p, p) for p in inner_names if p not in fixed_inner]
        params = [str(p) for p in params]
        if len(set(params)) != len(params):
            raise ValueError(f"duplicate parameter names: {params}")

        source = [-1] * len(inner_names)
        renamed = []
        for col, name in enumerate(params):
            match = view_to_inner.get(name) or resolve_param_name(name, inner_names)
            if match is None:
                raise ValueError(
                    f"{name!r} is not an input of this emulator (nor an alias of one); "
                    f"have {inner_names}"
                )
            if match in fixed_inner:
                raise ValueError(f"{name!r} is both exposed and fixed")
            if source[inner_names.index(match)] >= 0:
                raise ValueError(f"{name!r} maps to {match!r}, which is already exposed")
            source[inner_names.index(match)] = col
            if match != name:
                renamed.append((match, name))

        constant = [0.0] * len(inner_names)
        filled = {}
        for i, name in enumerate(inner_names):
            if source[i] >= 0:
                continue
            if name in fixed_inner:
                constant[i] = fixed_inner[name]
                continue
            if not fill_defaults:
                raise ValueError(
                    f"emulator input {name!r} is neither exposed nor fixed; add it to `fixed` "
                    "or pass fill_defaults=True"
                )
            if name not in spec.defaults:
                raise ValueError(
                    f"emulator input {name!r} is not exposed and has no default value; "
                    "add it to `fixed`"
                )
            constant[i] = filled[name] = fixed_inner[name] = float(spec.defaults[name])
        if filled:
            logger.info(f"EmulatorView: inputs held at their defaults: {filled}")

        for name, value in fixed_inner.items():
            if name in spec.param_bounds:
                lo, hi = spec.param_bounds[name]
                if not lo <= value <= hi:
                    raise ValueError(
                        f"fixed {name} = {value} is outside the emulator domain [{lo}, {hi}]"
                    )

        if bands is not None:
            missing = [b for b in bands if b not in inner.band_names]
            if missing:
                raise ValueError(f"bands not found in the emulator: {missing}")
            self._band_names = tuple(bands)
            self._band_idx = tuple(inner.band_names.index(b) for b in bands)
        else:
            self._band_names = tuple(inner.band_names)
            self._band_idx = None

        self.inner = inner
        self._param_names = tuple(params)
        self._source = tuple(source)
        self._constant = tuple(constant)
        self._fixed = tuple(sorted(fixed_inner.items()))
        self._renamed = tuple(renamed)

    @property
    def param_names(self) -> list[str]:
        """Exposed input names, in ``predict`` column order."""
        return list(self._param_names)

    @property
    def band_names(self) -> list[str]:
        """Bands returned by ``predict``."""
        return list(self._band_names)

    @property
    def fixed_params(self) -> dict[str, float]:
        """``{inner name: value}`` of the inputs held constant."""
        return dict(self._fixed)

    @property
    def spec(self) -> EmulatorSpec:
        """The inner spec restricted to the exposed inputs and bands, with view names."""
        inner_names = list(self.inner.param_names)
        order = sorted((col, inner_names[i]) for i, col in enumerate(self._source) if col >= 0)
        meta = {}
        fixed = {**self.inner.spec.meta.get("fixed", {}), **dict(self._fixed)}
        if fixed:
            meta["fixed"] = fixed
        if self._renamed:
            meta["renamed"] = {**self.inner.spec.meta.get("renamed", {}), **dict(self._renamed)}
        return self.inner.spec.subset(
            param_names=[name for _, name in order],
            band_names=self._band_names,
            rename=dict(self._renamed),
            meta=meta,
        )

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        """Exposed inputs ``(N, n_params)`` -> photometry ``(N, n_bands)`` in nJy."""
        params = jnp.atleast_2d(params)
        columns = [
            params[:, col] if col >= 0 else jnp.full(params.shape[:1], value, dtype=params.dtype)
            for col, value in zip(self._source, self._constant)
        ]
        flux = self.inner.predict(jnp.stack(columns, axis=1))
        if self._band_idx is not None:
            flux = flux[:, jnp.array(self._band_idx)]
        return flux
