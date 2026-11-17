"""Per-band emulator accuracy tables.

An emulator's fractional flux error per band, measured on held-out library rows
in redshift bins, belongs with the checkpoint: a fit adds it in quadrature to
its model-error floor.  :func:`band_error_table` measures it,
``ParrotEmulatorV2.with_metadata(band_errors=...)`` stores it in the
checkpoint, and ``emulator.spec.band_error(redshift)`` reads it back.

Table format (JSON-compatible; also the format of the older stand-alone
``perband_fractional_error.json`` files, which lack ``z_bins``)::

    {"source": "...", "definition": "...",
     "z_bins": {"z1.8-2.5": [1.8, 2.5], "all": null, ...},
     "bands": {band: {bin: {"bias", "mad", "std", "p95", "n"}}}}

This module is pure numpy (no JAX).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_Z_EDGES",
    "band_error_table",
    "check_band_error_table",
    "select_band_errors",
    "select_z_bin",
    "table_z_bins",
]

#: Edges of the default redshift bins (clipped to the redshifts present).
DEFAULT_Z_EDGES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0)
#: Extra bins kept for continuity with the tables made before 2026-11-18.
_LEGACY_BINS: dict[str, tuple[float, float]] = {"z1.8-2.5": (1.8, 2.5), "z1.5-3.0": (1.5, 3.0)}
_BIN_NAME = re.compile(r"^z(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$")
_STATS = ("bias", "mad", "std", "p95", "n")
_DEFINITION = (
    "fractional flux error pred/true - 1 for true flux > {min_flux:g} nJy; "
    "bias = median, mad = 1.4826*MAD"
)


def _default_bins(redshift: np.ndarray) -> dict[str, tuple[float, float] | None]:
    bins: dict[str, tuple[float, float] | None] = dict(_LEGACY_BINS)
    z_max = float(np.max(redshift))
    for lo, hi in zip(DEFAULT_Z_EDGES[:-1], DEFAULT_Z_EDGES[1:]):
        if lo < z_max:
            bins[f"z{lo:g}-{hi:g}"] = (lo, hi)
    bins["all"] = None
    return bins


def band_error_table(
    pred_flux: np.ndarray,
    true_flux: np.ndarray,
    redshift: np.ndarray,
    band_names: Sequence[str],
    z_bins: Mapping[str, tuple[float, float] | None] | None = None,
    min_flux: float = 1.0,
    source: str = "",
) -> dict:
    """Measure the fractional flux error per band in redshift bins.

    Args:
        pred_flux: Emulator fluxes ``(N, n_bands)`` in nJy for held-out rows.
        true_flux: Library fluxes ``(N, n_bands)`` in nJy for the same rows.
        redshift: Redshift of each row, ``(N,)``.
        band_names: Band names, in column order.
        z_bins: ``{name: (lo, hi)}``, with ``None`` meaning every row.  Default:
            bins on :data:`DEFAULT_Z_EDGES` up to the largest redshift, the two
            bins of the older tables, and ``"all"``.
        min_flux: Only rows with a true flux above this (nJy) count in a band.
        source: Free-text provenance stored in the table.

    Returns:
        The table (see the module docstring).  A bin with no rows has ``None``
        statistics and ``n = 0``.
    """
    pred_flux = np.asarray(pred_flux, dtype=np.float64)
    true_flux = np.asarray(true_flux, dtype=np.float64)
    redshift = np.asarray(redshift, dtype=np.float64)
    if pred_flux.shape != true_flux.shape or pred_flux.shape != (len(redshift), len(band_names)):
        raise ValueError(
            f"shapes disagree: pred {pred_flux.shape}, true {true_flux.shape}, "
            f"{len(redshift)} redshifts, {len(band_names)} bands"
        )
    bins = dict(z_bins) if z_bins is not None else _default_bins(redshift)
    in_bin = {
        name: np.ones(len(redshift), bool)
        if rng is None
        else (redshift >= rng[0]) & (redshift <= rng[1])
        for name, rng in bins.items()
    }
    bands: dict[str, dict] = {}
    for i, band in enumerate(band_names):
        detected = true_flux[:, i] > min_flux
        bands[band] = {}
        for name, rows in in_bin.items():
            m = rows & detected
            if not m.any():
                bands[band][name] = {"bias": None, "mad": None, "std": None, "p95": None, "n": 0}
                continue
            e = pred_flux[m, i] / true_flux[m, i] - 1.0
            med = float(np.median(e))
            bands[band][name] = {
                "bias": med,
                "mad": float(1.4826 * np.median(np.abs(e - med))),
                "std": float(e.std()),
                "p95": float(np.percentile(np.abs(e), 95)),
                "n": int(m.sum()),
            }
    return {
        "source": str(source),
        "definition": _DEFINITION.format(min_flux=min_flux),
        "z_bins": {k: (None if v is None else [float(v[0]), float(v[1])]) for k, v in bins.items()},
        "bands": bands,
    }


def table_z_bins(table: Mapping) -> dict[str, tuple[float, float] | None]:
    """Redshift range of every bin of a table.

    Uses ``table["z_bins"]`` if present; otherwise parses names of the form
    ``z<lo>-<hi>`` and treats anything else (``"all"``) as every redshift.
    """
    names: list[str] = []
    for per_band in table["bands"].values():
        names.extend(n for n in per_band if n not in names)
    recorded = table.get("z_bins") or {}
    out: dict[str, tuple[float, float] | None] = {}
    for name in names:
        if name in recorded:
            rng = recorded[name]
            out[name] = None if rng is None else (float(rng[0]), float(rng[1]))
            continue
        match = _BIN_NAME.match(name)
        out[name] = (float(match.group(1)), float(match.group(2))) if match else None
    return out


def check_band_error_table(table: Mapping, band_names: Sequence[str] | None = None) -> None:
    """Raise ``ValueError`` if ``table`` is not a usable band-error table.

    Args:
        table: Candidate table.
        band_names: If given, every band of the table must be one of these.
    """
    if not isinstance(table, Mapping) or not isinstance(table.get("bands"), Mapping):
        raise ValueError('a band-error table needs a "bands" mapping {band: {bin: stats}}')
    if not table["bands"]:
        raise ValueError("the band-error table has no bands")
    for band, per_bin in table["bands"].items():
        if band_names is not None and band not in band_names:
            raise ValueError(f"band-error table band {band!r} is not an emulator band")
        if not isinstance(per_bin, Mapping) or not per_bin:
            raise ValueError(f"band-error table: {band!r} has no redshift bins")
        for name, stats in per_bin.items():
            if not isinstance(stats, Mapping) or "mad" not in stats:
                raise ValueError(f"band-error table: {band!r}/{name!r} lacks a 'mad' entry")


def select_z_bin(
    table: Mapping,
    redshift: float | None,
    bands: Sequence[str] | None = None,
    stat: str = "mad",
    min_rows: int = 100,
) -> str:
    """Name of the bin to use at ``redshift``.

    The narrowest bin that contains ``redshift`` and has, in every requested
    band, a value for ``stat`` from at least ``min_rows`` rows; otherwise the
    widest bin (normally ``"all"``).  ``redshift=None`` gives the widest bin.

    Args:
        table: Band-error table.
        redshift: Target redshift, or ``None``.
        bands: Bands that must be populated; default: every band of the table.
        stat: Statistic that must be present.
        min_rows: Minimum rows per band for a bin to qualify.

    Returns:
        A bin name of the table.
    """
    bins = table_z_bins(table)
    use = list(bands) if bands is not None else list(table["bands"])

    def width(name: str) -> float:
        return np.inf if bins[name] is None else bins[name][1] - bins[name][0]

    def populated(name: str, rows: int) -> bool:
        for band in use:
            stats = table["bands"].get(band, {}).get(name)
            if not stats or stats.get(stat) is None or int(stats.get("n", rows)) < rows:
                return False
        return True

    fallback = max(bins, key=width)
    if redshift is None:
        return fallback
    candidates = [
        n for n, rng in bins.items() if rng is None or rng[0] <= float(redshift) <= rng[1]
    ]
    for name in sorted(candidates, key=width):
        if populated(name, min_rows):
            return name
    return fallback


def select_band_errors(
    table: Mapping,
    band_names: Sequence[str],
    redshift: float | None = None,
    stat: str = "mad",
    z_bin: str | None = None,
    min_rows: int = 100,
) -> tuple[np.ndarray, str]:
    """Per-band value of ``stat`` for ``band_names`` at ``redshift``.

    Args:
        table: Band-error table.
        band_names: Bands wanted, in order.
        redshift: Target redshift; picks the bin (see :func:`select_z_bin`).
        stat: ``"mad"`` (default), ``"std"``, ``"p95"`` or ``"bias"``.
        z_bin: Use this bin instead of choosing one from ``redshift``.
        min_rows: Minimum rows per band when choosing a bin.

    Returns:
        ``(values, bin_name)`` with ``values`` of shape ``(len(band_names),)``.

    Raises:
        ValueError: If a band is not in the table, ``z_bin`` is unknown, or the
            chosen bin has no value for a band.
    """
    missing = [b for b in band_names if b not in table["bands"]]
    if missing:
        raise ValueError(f"the band-error table has no entry for {missing}")
    name = z_bin or select_z_bin(table, redshift, band_names, stat, min_rows)
    values = []
    for band in band_names:
        stats = table["bands"][band].get(name)
        if stats is None:
            raise ValueError(f"the band-error table has no bin {name!r} for {band!r}")
        if stats.get(stat) is None:
            raise ValueError(f"the band-error table has no {stat!r} for {band!r} in bin {name!r}")
        values.append(float(stats[stat]))
    return np.asarray(values, dtype=np.float64), name
