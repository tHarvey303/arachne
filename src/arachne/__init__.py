"""arachne: image-level forward modelling of galaxy populations."""

from arachne.data.dja import fetch_dja_cutout, load_dja_cutout
from arachne.data.jades import load_jades_dr4_specz, select_targets
from arachne.data.multires import BandImage, MultiResolutionObservation
from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.data.units import flux_to_nJy
from arachne.emulator.base import SPSEmulator
from arachne.emulator.jax_emulator import JAXFlowEmulator
from arachne.emulator.jax_mlp_emulator import SPSMLPEmulator
from arachne.emulator.parrot_emulator import ParrotEmulator
from arachne.emulator.parrot_emulator_v2 import ParrotEmulatorV2, load_emulator
from arachne.forward_model.multires import MultiResolutionForwardModel
from arachne.forward_model.nuisance import NuisanceModel
from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.batched import (
    BatchedForwardModel,
    BatchedMAPResult,
    BatchedNUTSResult,
    batched_blind_initial_theta,
    batched_find_map,
    batched_multistart_map,
    batched_nuts,
    fit_batch_nss,
)
from arachne.inference.diagnostics import ess, split_rhat, summarise_chains
from arachne.inference.initialisation import (
    MAPResult,
    blind_initial_full_theta,
    blind_initial_theta,
    find_map,
    image_moments,
    multistart_map,
    reference_band_index,
    solve_component_masses,
)
from arachne.inference.mclmc_sampler import MCLMCSampler, run_pathfinder
from arachne.inference.model_comparison import (
    ModelComparisonRow,
    bayes_factor_table,
    compare_n_components,
)
from arachne.inference.nss_sampler import NSSResult, NSSSampler
from arachne.inference.nuts_sampler import NUTSResult, NUTSSampler
from arachne.inference.posterior_predictive import (
    component_image_samples,
    model_image_samples,
    residual_summary,
)
from arachne.likelihood.gaussian import GaussianLikelihood
from arachne.priors.physical import IndependentUniformPrior, LogNormalPrior
from arachne.priors.spatial import GradientPenaltyPrior, TotalVariationPrior
from arachne.psf.convolution import PSFConvolver
from arachne.spatial.additive import AdditiveComponentModel
from arachne.spatial.base import SpatialModel
from arachne.spatial.gmm import GaussianMixtureSpatialModel
from arachne.spatial.pixel_map import FreeFormPixelMap
from arachne.spatial.profiles import (
    PROFILES,
    GaussianProfile,
    PointSourceProfile,
    Profile,
    SersicProfile,
    get_profile,
    render_on_grid,
    sersic_b,
)

__version__ = "0.1.0"

__all__ = [
    "ObservationCube",
    "MultiResolutionObservation",
    "BandImage",
    "fetch_dja_cutout",
    "load_dja_cutout",
    "load_jades_dr4_specz",
    "select_targets",
    "flux_to_nJy",
    "PSFModel",
    "SPSEmulator",
    "SPSMLPEmulator",
    "ParrotEmulator",
    "ParrotEmulatorV2",
    "load_emulator",
    "JAXFlowEmulator",
    "ForwardModel",
    "MultiResolutionForwardModel",
    "NuisanceModel",
    "NUTSSampler",
    "MCLMCSampler",
    "run_pathfinder",
    "NUTSResult",
    "NSSSampler",
    "NSSResult",
    "MAPResult",
    "blind_initial_theta",
    "blind_initial_full_theta",
    "reference_band_index",
    "find_map",
    "multistart_map",
    "solve_component_masses",
    "image_moments",
    "split_rhat",
    "ess",
    "summarise_chains",
    "compare_n_components",
    "bayes_factor_table",
    "ModelComparisonRow",
    "BatchedForwardModel",
    "BatchedMAPResult",
    "BatchedNUTSResult",
    "batched_blind_initial_theta",
    "batched_find_map",
    "batched_multistart_map",
    "batched_nuts",
    "fit_batch_nss",
    "model_image_samples",
    "component_image_samples",
    "residual_summary",
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
    "Profile",
    "GaussianProfile",
    "SersicProfile",
    "PointSourceProfile",
    "PROFILES",
    "get_profile",
    "render_on_grid",
    "sersic_b",
]
