"""Load emulator checkpoints of any registered format.

A checkpoint format is identified by a key in its ``.npz`` archive
(``v2_config`` for :class:`~arachne.emulator.parrot_emulator_v2.ParrotEmulatorV2`).
:func:`load_emulator` finds the matching loader, so fitting code never names an
emulator class; :func:`register_emulator_format` adds a new one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from arachne.emulator.base import SPSEmulator

__all__ = ["load_emulator", "register_emulator_format"]

# (archive key, loader), tried in order; ``None`` matches any archive (the fallback).
_FORMATS: list[tuple[str | None, Callable[[str | Path], SPSEmulator]]] = []


def register_emulator_format(
    archive_key: str | None, loader: Callable[[str | Path], SPSEmulator]
) -> None:
    """Register a checkpoint loader.

    Args:
        archive_key: An ``.npz`` key that only this format writes, or ``None``
            for a fallback tried after every keyed format.
        loader: ``loader(path) -> SPSEmulator``.
    """
    entry = (archive_key, loader)
    _FORMATS[:] = [f for f in _FORMATS if f[0] != archive_key]
    if archive_key is None:
        _FORMATS.append(entry)
    else:
        _FORMATS.insert(0, entry)


def _load_v2(path):
    from arachne.emulator.parrot_emulator_v2 import ParrotEmulatorV2

    return ParrotEmulatorV2.load(path)


def _load_v1(path):
    from arachne.emulator.parrot_emulator import ParrotEmulator

    return ParrotEmulator.load(path)


register_emulator_format(None, _load_v1)
register_emulator_format("v2_config", _load_v2)


def load_emulator(
    path: str | Path,
    *,
    params: Sequence[str] | None = None,
    fixed: Mapping[str, float] | None = None,
    bands: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    fill_defaults: bool = True,
) -> SPSEmulator:
    """Load an emulator checkpoint, optionally presented under a chosen input/band list.

    With no keyword arguments this returns the emulator exactly as trained.
    With any of them it returns an :class:`~arachne.emulator.view.EmulatorView`:

    * ``params``: the inputs the fit uses, in the fit's order (aliases such as
      ``slope`` / ``dust_slope`` are matched);
    * ``fixed``: inputs held at given values;
    * every other input is held at ``emulator.spec.defaults`` (``fill_defaults``);
    * ``bands``: the output bands, in the fit's order.

    ``emulator.spec`` then gives the bounds, neutral values and default priors
    of exactly the exposed inputs, so the same fitting code runs with any
    checkpoint.

    Args:
        path: Checkpoint file.
        params: Exposed input names; ``None`` keeps every non-fixed input.
        fixed: ``{name: value}`` inputs to hold constant.
        bands: Output bands; ``None`` keeps all.
        rename: ``{checkpoint name: exposed name}`` beyond the alias table.
        fill_defaults: See :class:`~arachne.emulator.view.EmulatorView`.

    Returns:
        The emulator, or a view of it.
    """
    with np.load(str(path), allow_pickle=False) as raw:
        keys = set(raw.keys())
    loader = next(fn for key, fn in _FORMATS if key is None or key in keys)
    emulator = loader(path)
    if params is None and fixed is None and bands is None and rename is None:
        return emulator
    from arachne.emulator.view import EmulatorView

    return EmulatorView(
        emulator,
        params=params,
        fixed=fixed,
        bands=bands,
        rename=rename,
        fill_defaults=fill_defaults,
    )
