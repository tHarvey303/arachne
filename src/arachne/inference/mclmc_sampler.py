"""MCLMC sampler and Pathfinder initialiser for arachne forward models.

Microcanonical Langevin Monte Carlo (MCLMC, Robnik et al. 2022/2023) is
preferred over NUTS for the high-dimensional ``FreeFormPixelMap`` regime
(d ~ 45k–67k parameters).  Unlike NUTS, MCLMC uses *persistent momentum*
and does not resample momenta or build binary trees each step, giving
roughly O(1) gradient evaluations per independent sample rather than
NUTS's O(d^{1/4}).  Benchmarks show 5–50× fewer gradient evaluations at
d = 1,000–10,000, with the gap increasing with dimension.

This module provides:

* ``MCLMCSampler`` — drop-in replacement for ``NUTSSampler`` using
  ``blackjax.mclmc`` with automatic L and step-size tuning
  (``blackjax.mclmc_find_L_and_step_size``) and diagonal preconditioning.
* ``run_pathfinder`` — convenience wrapper around ``blackjax.pathfinder``
  that returns an improved starting position and a diagonal inverse-mass-
  matrix estimate via L-BFGS.  Can be used to warm-start either
  ``MCLMCSampler`` or ``NUTSSampler``, replacing the expensive
  ``window_adaptation`` warmup.

Typical workflow for a large pixel map
---------------------------------------
.. code-block:: python

    # 1. Quick MAP + mass-matrix estimate via Pathfinder
    (
        theta_init,
        inv_mass,
    ) = run_pathfinder(
        forward_model.log_posterior,
        theta_zero,
        rng_key,
    )

    # 2. MCLMC sampling (much cheaper per effective sample than NUTS at high d)
    sampler = MCLMCSampler(
        forward_model,
        n_warmup=1000,
        n_samples=500,
    )
    result = sampler.run(
        theta_init,
        rng_key,
        inverse_mass_matrix=inv_mass,
    )

References:
----------
* Robnik, De Luca & Guth (2022) "Microcanonical Hamiltonian Monte Carlo"
  arXiv:2212.08549
* Robnik & Seljak (2023) "Optimal Tuning of Microcanonical Hamiltonian
  Monte Carlo" arXiv:2309.07202
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

from arachne.forward_model.pipeline import ForwardModel
from arachne.inference.nuts_sampler import NUTSResult
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)


# ---------------------------------------------------------------------------
# Pathfinder initialiser
# ---------------------------------------------------------------------------


def run_pathfinder(
    log_posterior_fn,
    theta_init: jnp.ndarray,
    rng_key: jnp.ndarray,
    num_samples: int = 200,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Approximate the posterior with Pathfinder (L-BFGS variational inference).

    Runs ``blackjax.pathfinder`` to find a MAP-like starting position and a
    diagonal inverse-mass-matrix estimate in a small number of gradient
    evaluations.  The returned values can be passed directly to
    ``MCLMCSampler.run`` or ``NUTSSampler`` to skip the expensive warmup
    phase.

    Args:
        log_posterior_fn: Callable ``theta -> scalar``.  Should be the
            ``forward_model.log_posterior`` method.
        theta_init: Initial unconstrained parameter vector, shape (N,).
        rng_key: JAX random key.
        num_samples: Number of L-BFGS samples used to estimate the inverse
            Hessian diagonal.  Defaults to 200.

    Returns:
        Tuple ``(position, inverse_mass_matrix)`` where:
        * ``position`` — best position found by L-BFGS, shape (N,).
          A better starting point than the prior mean for MCMC.
        * ``inverse_mass_matrix`` — diagonal inverse-mass-matrix estimate,
          shape (N,).  Pass to ``MCLMCSampler.run`` or use as the initial
          ``inverse_mass_matrix`` for ``blackjax.nuts``.

    Raises:
        ImportError: If ``blackjax`` is not installed.
    """
    try:
        import blackjax.vi.pathfinder as pf_mod
    except ImportError as e:
        raise ImportError(
            "blackjax is required for Pathfinder. Install with: pip install blackjax"
        ) from e

    logger.info(f"Running Pathfinder (num_samples={num_samples}, d={theta_init.shape[0]})...")
    pf_state, _ = pf_mod.approximate(
        rng_key=rng_key,
        logdensity_fn=log_posterior_fn,
        initial_position=theta_init,
        num_samples=num_samples,
    )
    logger.info("Pathfinder complete.")
    # pf_state.position: MAP-like estimate; pf_state.alpha: diag inv-Hessian
    return pf_state.position, pf_state.alpha


# ---------------------------------------------------------------------------
# MCLMC sampler
# ---------------------------------------------------------------------------


class MCLMCSampler:
    """MCLMC sampler with automatic L / step-size tuning.

    Uses ``blackjax.mclmc`` (Robnik et al. 2022) with
    ``blackjax.mclmc_find_L_and_step_size`` for warmup.  Significantly more
    efficient than NUTS for the ``FreeFormPixelMap`` regime
    (d ~ 45k–67k):

    * **Persistent momentum** — no per-step momentum resampling, better
      exploration of spatially correlated posteriors.
    * **No tree doubling** — O(1) gradient evaluations per step, predictable
      GPU memory use.  The ``max_num_doublings`` cap in ``NUTSSampler`` is
      not needed.
    * **Diagonal preconditioning** — ``mclmc_find_L_and_step_size`` jointly
      adapts L, step size *and* a diagonal inverse-mass-matrix, which
      partially compensates for the spatial correlations introduced by the
      smoothness prior.

    The sampling loop uses ``jax.lax.scan`` for full GPU utilisation,
    identical to ``NUTSSampler``.

    Returns a ``NUTSResult`` (same interface as ``NUTSSampler``) for
    compatibility with ``get_parameter_map``, ``to_hdf5``, etc.

    Attributes:
        forward_model: ForwardModel whose ``log_posterior`` is sampled.
        n_warmup: Number of integration steps used for tuning L and step size.
        n_samples: Number of posterior samples to draw.
        diagonal_preconditioning: Whether to adapt a diagonal inverse-mass-
            matrix during warmup.  Recommended ``True`` for pixel maps.
        desired_energy_var: Target energy variance for the MCLMC integrator.
            Lower values → smaller step size → more accurate but slower.
            Default ``5e-4`` matches the BlackJAX recommendation.
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        n_warmup: int = 1000,
        n_samples: int = 500,
        diagonal_preconditioning: bool = True,
        desired_energy_var: float = 5e-4,
    ) -> None:
        """Initialise the MCLMC sampler.

        Args:
            forward_model: Assembled ForwardModel.
            n_warmup: Number of integration steps for tuning.  1000 is a
                reasonable default; increase for very high-d problems.
            n_samples: Number of posterior samples to collect after warmup.
            diagonal_preconditioning: Adapt diagonal inverse-mass-matrix
                during warmup.  Defaults to True.
            desired_energy_var: Energy variance target for the integrator.
                Defaults to 5e-4.
        """
        self.forward_model = forward_model
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.diagonal_preconditioning = diagonal_preconditioning
        self.desired_energy_var = desired_energy_var

    def run(
        self,
        theta_init: jnp.ndarray,
        rng_key: jnp.ndarray,
        inverse_mass_matrix: Optional[jnp.ndarray] = None,
    ) -> NUTSResult:
        """Run MCLMC warmup (tuning) and sampling.

        Args:
            theta_init: Initial unconstrained parameter vector, shape (N,).
                For best results use the output of ``run_pathfinder``.
            rng_key: JAX random key.
            inverse_mass_matrix: Optional diagonal inverse-mass-matrix
                (shape (N,)) to use as the initial guess for tuning.  If
                ``None``, defaults to a vector of ones (isotropic).  Pass
                the ``inverse_mass_matrix`` returned by ``run_pathfinder``
                to skip most of the mass-matrix adaptation.

        Returns:
            ``NUTSResult`` containing posterior samples (shape
            ``(n_samples, N)``), MCLMC step info, and the spatial model
            reference.  The ``acceptance_rate`` attribute reflects the
            Metropolis acceptance rate of the underlying integrator.
        """
        try:
            import blackjax
            import blackjax.mcmc.integrators as integrators
            import blackjax.mcmc.mclmc as mclmc_mod
            from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
        except ImportError as e:
            raise ImportError(
                "blackjax is required for MCLMC sampling. Install with: pip install blackjax"
            ) from e

        logpost = jax.jit(self.forward_model.log_posterior)
        d = theta_init.shape[0]

        logger.info(
            f"Starting MCLMC: {self.n_warmup} warmup steps + {self.n_samples} samples, "
            f"d={d}, diagonal_preconditioning={self.diagonal_preconditioning}"
        )

        k_init, k_tune, k_sample = jax.random.split(rng_key, 3)

        # 1. Initialise MCLMC state (samples initial momentum from key)
        state = mclmc_mod.init(theta_init, logpost, k_init)

        # 2. Build kernel factory (callable: inverse_mass_matrix -> kernel)
        def kernel_factory(imm):
            return mclmc_mod.build_kernel(
                logdensity_fn=logpost,
                inverse_mass_matrix=imm,
                integrator=integrators.isokinetic_mclachlan,
            )

        # 3. Set initial adaptation params (L, step_size, inverse_mass_matrix)
        init_imm = inverse_mass_matrix if inverse_mass_matrix is not None else jnp.ones(d)
        init_params = MCLMCAdaptationState(
            L=jnp.sqrt(float(d)),
            step_size=jnp.sqrt(float(d)) * 0.25,
            inverse_mass_matrix=init_imm,
        )

        # 4. Auto-tune L, step_size, and (optionally) inverse_mass_matrix
        state, params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel_factory,
            num_steps=self.n_warmup,
            state=state,
            rng_key=k_tune,
            diagonal_preconditioning=self.diagonal_preconditioning,
            desired_energy_var=self.desired_energy_var,
            params=init_params,
        )
        logger.info(
            f"Warmup complete. L={float(params.L):.4g}, step_size={float(params.step_size):.4g}"
        )

        # 5. Build final kernel with tuned parameters
        kernel = kernel_factory(params.inverse_mass_matrix)

        # 6. Sampling via jax.lax.scan (full GPU efficiency, no Python overhead)
        def one_step(carry: Any, rng_key: jnp.ndarray) -> tuple[Any, tuple[jnp.ndarray, Any]]:
            state, info = kernel(rng_key, carry, params.step_size, params.L)
            return state, (state.position, info)

        sample_keys = jax.random.split(k_sample, self.n_samples)
        final_state, (samples, infos) = jax.lax.scan(one_step, state, sample_keys)

        logger.info(f"Sampling complete. Samples shape: {samples.shape}")

        return NUTSResult(
            samples=samples,
            infos=infos,
            spatial_model=self.forward_model.spatial_model,
        )
