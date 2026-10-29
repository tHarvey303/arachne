"""Results bundles: a self-describing HDF5 record of a fit, its reader and region SEDs.

Everything here is numpy/h5py only, except ``write_bundle`` / ``content_from_forward_model``
(``arachne.results.writer``, which evaluates a JAX forward model).  Those two are loaded lazily
on first attribute access so that importing this package never imports the writer.
"""

from __future__ import annotations

import importlib
from typing import Any

from . import display, metadata, schema, testing
from .display import percentile_limits, rgb_composite, stretch
from .io import write_content
from .metadata import DEFAULT_LABELS, PIVOT_UM, pivot_um, short_band_name
from .reader import ComponentsView, ResultsBundle, load_bundle
from .regions import (
    Region,
    RegionSED,
    component_region_seds,
    component_seds,
    detection_region,
    pixel_sed,
    region_param_summary,
    region_sed,
    total_sed,
)
from .schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    BundleContent,
    ComponentContent,
    TruthContent,
    validate,
)
from .testing import make_synthetic_bundle, make_synthetic_content, toy_sed

_LAZY_WRITER = ("write_bundle", "content_from_forward_model")

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "BundleContent",
    "ComponentContent",
    "TruthContent",
    "validate",
    "write_content",
    "ResultsBundle",
    "ComponentsView",
    "load_bundle",
    "Region",
    "RegionSED",
    "region_sed",
    "total_sed",
    "pixel_sed",
    "component_seds",
    "component_region_seds",
    "region_param_summary",
    "detection_region",
    "percentile_limits",
    "stretch",
    "rgb_composite",
    "PIVOT_UM",
    "DEFAULT_LABELS",
    "pivot_um",
    "short_band_name",
    "toy_sed",
    "make_synthetic_content",
    "make_synthetic_bundle",
    "display",
    "metadata",
    "schema",
    "testing",
    # add *_LAZY_WRITER once writer.py exists (P1); listing them now would break `import *`
]


def __getattr__(name: str) -> Any:
    """Load the JAX-dependent writer on first access to its public names."""
    if name in _LAZY_WRITER:
        try:
            writer = importlib.import_module(f"{__name__}.writer")
        except ModuleNotFoundError as err:
            if err.name != f"{__name__}.writer":
                raise
            raise ImportError(
                f"arachne.results.{name} is not available: arachne.results.writer has not "
                "been implemented yet (phase P1)"
            ) from err
        return getattr(writer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
