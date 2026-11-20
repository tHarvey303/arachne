"""Exact posterior curvature with the per-pixel emulator derivatives cached."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from arachne.forward_model.pipeline import ForwardModel
from arachne.spatial.correlated_field import CorrelatedFieldMap


def field_hvp_builder(fm):
    """Return a function that builds a Hessian-vector product at a fixed position.

    For the built-in correlated-field pipeline, compute the emulator Jacobian
    and a likelihood-weighted emulator Hessian at each pixel once. Every HVP
    then differentiates only the field decode, instrument and likelihood.
    Other model types can use automatic differentiation of their full posterior.
    """
    if type(fm) is not ForwardModel or type(fm.spatial_model) is not CorrelatedFieldMap:
        return None

    model = fm.spatial_model
    height, width = model.image_shape
    active = jnp.asarray(
        [i for i, p in enumerate(model.emulator_param_names) if p not in model.fixed_params]
    )

    def data_loss(theta, flux):
        _, theta_nuisance = fm.split_theta(theta)
        nuis = fm._nuisance_blocks(theta_nuisance)
        image = flux.T.reshape(flux.shape[1], height, width)
        return -fm.likelihood(fm._apply_instrument(image, nuis), nuis["log_noise_scale"])

    def emulator_derivatives(row_and_weight):
        params, band_weight = row_and_weight

        def predict(free):
            return fm.emulator.predict(params.at[active].set(free)[None, :])[0]

        free = params[active]
        jacobian = jax.jacfwd(predict)(free)
        weighted_hess = jax.hessian(lambda free: band_weight @ predict(free))(free)
        return jacobian, weighted_hess

    @jax.jit
    def build_hvp(theta):
        pixel_params = model.decode(fm.split_theta(theta)[0])
        pixel_flux = fm.emulator.predict(pixel_params)
        band_weight = jax.grad(data_loss, argnums=1)(theta, pixel_flux)
        jacobian, weighted_hess = jax.lax.map(
            emulator_derivatives, (pixel_params, band_weight), batch_size=32
        )
        at = pixel_params[:, active]

        def local_loss(t):
            delta = model.decode(fm.split_theta(t)[0])[:, active] - at
            flux = pixel_flux + jnp.einsum("pbi,pi->pb", jacobian, delta)
            # The linearised flux carries all field, PSF and likelihood terms.
            # Add the missing second derivative of the emulator contracted
            # with d(data_loss)/d(pixel_flux), including its sign off the mode.
            curvature = 0.5 * jnp.einsum("pi,pij,pj->", delta, weighted_hess, delta)
            return data_loss(t, flux) - fm.log_prior(t) + curvature

        return jax.linearize(jax.grad(local_loss), theta)[1]

    return build_hvp
