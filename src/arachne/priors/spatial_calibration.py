"""Load empirically calibrated Matérn priors for a resolved SED fit."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def load_spatial_calibration(
    path: str | Path,
    *,
    redshift: float,
    re_pixels: float | None = None,
    kpc_per_pixel: float | None = None,
    fields: Sequence[str] | None = None,
    sample_hyperparameters: bool = False,
    allow_extrapolation: bool = False,
) -> dict:
    """Convert calibrated spatial residuals to CorrelatedFieldMap keyword arguments.

    Args:
        path: Version-1 population calibration JSON.
        redshift: Redshift of the target galaxy, used to check calibration support.
        re_pixels: Projected effective radius in model pixels. Selects the pooled
            radius-normalized length distribution.
        kpc_per_pixel: Physical pixel size in kpc. Selects the pooled physical length
            distribution. Supply exactly one of this and re_pixels.
        fields: Subset of the transferable fields to load. The default selects all of them.
        sample_hyperparameters: Include population scatter as Gaussian priors on log
            length and amplitude. Otherwise use respective population medians.
        allow_extrapolation: Permit use outside the redshift interval in the file.

    Returns:
        Keyword mappings for the calibrated fields. Merge each mapping with any
        field-specific settings for other fields in the model.
    """
    data = json.loads(Path(path).read_text())
    if data.get("version") != 1:
        raise ValueError("Expected version 1 spatial calibration")
    if not math.isfinite(redshift) or redshift < 0:
        raise ValueError("redshift must be finite and nonnegative")
    z_min, z_max = data["redshift_range"]
    if not allow_extrapolation and not z_min <= redshift <= z_max:
        raise ValueError(f"z={redshift:g} is outside the calibration range [{z_min:g}, {z_max:g}]")
    if (re_pixels is None) == (kpc_per_pixel is None):
        raise ValueError("Specify exactly one of re_pixels and kpc_per_pixel")
    if re_pixels is not None:
        length_unit, factor = "re", _positive(re_pixels, "re_pixels")
    else:
        length_unit, factor = "kpc", 1 / _positive(kpc_per_pixel, "kpc_per_pixel")

    result = {"corr_lengths": {}, "amplitudes": {}, "nus": {}}
    if sample_hyperparameters:
        result["hyper_priors"] = {}
    nu = _positive(data["nu"], "nu")
    names = data["transferable_fields"] if fields is None else fields
    unsupported = set(names) - set(data["transferable_fields"])
    if unsupported:
        raise ValueError(f"Fields without a direct calibration: {sorted(unsupported)}")
    for name in names:
        field = data["fields"][name]
        length = field["length"][length_unit]
        amplitude = field["amplitude"]
        result["corr_lengths"][name] = _positive(length["median"], "length median") * factor
        result["amplitudes"][name] = _positive(amplitude["median"], "amplitude median")
        result["nus"][name] = nu
        if sample_hyperparameters:
            result["hyper_priors"][name] = {
                "log_corr_length": {
                    "loc": math.log(result["corr_lengths"][name]),
                    "scale": _positive(length["log_scatter"], "log length scatter"),
                },
                "log_amplitude": {
                    "loc": math.log(result["amplitudes"][name]),
                    "scale": _positive(amplitude["log_scatter"], "log amplitude scatter"),
                },
            }
    return result
