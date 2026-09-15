"""Priors: physical-parameter priors, spatial regularisers, and prior specs.

* :mod:`arachne.priors.specs` -- dict-based per-parameter prior specifications
  (``{"dist": "studentt", "df": 2, ...}``), their defaults, validation and the
  JAX log-prior builders used by the catalogue scripts and spatial models.
* :mod:`arachne.priors.physical` -- simple vector priors on physical parameters.
* :mod:`arachne.priors.spatial` -- smoothness priors on parameter maps.
"""

from arachne.priors.physical import IndependentUniformPrior, LogNormalPrior
from arachne.priors.spatial import GradientPenaltyPrior, TotalVariationPrior
from arachne.priors.specs import (
    DEFAULT_PRIORS,
    SUPPORTED_DISTS,
    build_component_log_prior,
    build_log_prior,
    log_prior_1d,
    prior_config_template,
    resolve_prior_specs,
    sigmoid_log_jacobian,
    validate_prior_spec,
)

__all__ = [
    "DEFAULT_PRIORS",
    "SUPPORTED_DISTS",
    "GradientPenaltyPrior",
    "IndependentUniformPrior",
    "LogNormalPrior",
    "TotalVariationPrior",
    "build_component_log_prior",
    "build_log_prior",
    "log_prior_1d",
    "prior_config_template",
    "resolve_prior_specs",
    "sigmoid_log_jacobian",
    "validate_prior_spec",
]
