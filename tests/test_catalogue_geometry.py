"""Tests of the catalogue change of variables and Gauss–Newton metric."""

import jax
import jax.numpy as jnp
import numpy as np

from arachne.inference.catalogue_geometry import CatalogueCoordinates, gauss_newton_metric


def test_coordinates_roundtrip_and_jacobian():
    """The new chart preserves bounded physical values and their densities."""
    names = ["z", "sfh", "dust"]
    bounds = {"z": (0.01, 14.0), "sfh": (-8.0, 3.0), "dust": (0.0, 2.0)}
    priors = {"z": {}, "sfh": {"dist": "studentt", "scale": 0.3, "loc": 0.7}, "dust": {}}
    chart = CatalogueCoordinates(names, bounds, priors)
    theta = jnp.array([[-0.6, -1.7, 2.0], [1.9, 1.4, -0.1], [0.0, 0.0, 0.0]])
    x = chart.to_physical(theta)
    np.testing.assert_allclose(chart.from_physical(x), theta, atol=2e-6)
    np.testing.assert_allclose(chart.from_sigmoid(chart.to_sigmoid(theta)), theta, atol=5e-6)
    ordinary_x = chart.lo + (chart.hi - chart.lo) * jax.nn.sigmoid(chart.to_sigmoid(theta))
    np.testing.assert_allclose(ordinary_x, x, atol=2e-6)
    jac = jax.vmap(jax.jacfwd(chart.to_physical))(theta)
    np.testing.assert_allclose(
        jax.vmap(chart.log_det)(theta), jnp.linalg.slogdet(jac)[1], atol=2e-6
    )


def test_gauss_newton_metric_keeps_prior_scale_in_null_directions():
    """A rank-deficient observation Jacobian still gives a proper dense metric."""
    jac = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float32)
    sd = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    expected = np.linalg.inv(jac.T @ jac + np.diag(1.0 / sd**2))
    np.testing.assert_allclose(gauss_newton_metric(jac, sd), expected, atol=2e-7)
    np.testing.assert_allclose(gauss_newton_metric(np.zeros_like(jac), sd), np.diag(sd**2))
