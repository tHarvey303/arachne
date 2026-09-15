"""arachne: image-level forward modelling of galaxy populations."""

from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator
from arachne.emulator.jax_emulator import JAXFlowEmulator
from arachne.emulator.jax_mlp_emulator import SPSMLPEmulator
from arachne.emulator.parrot_emulator import ParrotEmulator
from arachne.emulator.parrot_emulator_v2 import ParrotEmulatorV2, load_emulator
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.initialisation import (
    MAPResult,
    blind_initial_theta,
    find_map,
    image_moments,
    multistart_map,
    solve_component_masses,
)
from arachne.inference.mclmc_sampler import MCLMCSampler, run_pathfinder
from arachne.inference.nss_sampler import NSSResult, NSSSampler
from arachne.inference.nuts_sampler import NUTSResult, NUTSSampler
from arachne.likelihood.gaussian import GaussianLikelihood
from arachne.priors.physical import IndependentUniformPrior, LogNormalPrior
from arachne.priors.spatial import GradientPenaltyPrior, TotalVariationPrior
from arachne.psf.convolution import PSFConvolver
from arachne.spatial.additive import AdditiveComponentModel
from arachne.spatial.base import SpatialModel
from arachne.spatial.gmm import GaussianMixtureSpatialModel
from arachne.spatial.pixel_map import FreeFormPixelMap

__version__ = "0.1.0"

__all__ = [
    "ObservationCube",
    "PSFModel",
    "SPSEmulator",
    "SPSMLPEmulator",
    "ParrotEmulator",
    "ParrotEmulatorV2",
    "load_emulator",
    "JAXFlowEmulator",
    "ForwardModel",
    "NUTSSampler",
    "MCLMCSampler",
    "run_pathfinder",
    "NUTSResult",
    "NSSSampler",
    "NSSResult",
    "MAPResult",
    "blind_initial_theta",
    "find_map",
    "multistart_map",
    "solve_component_masses",
    "image_moments",
    "GaussianLikelihood",
    "IndependentUniformPrior",
    "LogNormalPrior",
    "GradientPenaltyPrior",
    "TotalVariationPrior",
    "PSFConvolver",
    "SpatialModel",
    "AdditiveComponentModel",
    "GaussianMixtureSpatialModel",
    "FreeFormPixelMap",
]
