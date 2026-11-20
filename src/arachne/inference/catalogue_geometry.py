"""Prior-scaled coordinates and a Gauss–Newton metric for catalogue NUTS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from arachne.priors.specs import sigmoid_log_jacobian


class CatalogueCoordinates:
    """Bounded parameter chart that spreads the core of each Student-t prior.

    Student-t parameters use a sigmoid in ``asinh((x - loc) / scale)``.
    The other parameters use the usual sigmoid in physical space. The
    Jacobian preserves the original physical prior and likelihood.
    """

    def __init__(self, names, bounds, prior_specs) -> None:
        """Build the chart in the emulator's parameter order."""
        self.lo, self.hi = jnp.asarray([bounds[name] for name in names], jnp.float32).T
        specs = [prior_specs[name] for name in names]
        student = [p.get("dist", "uniform").lower() == "studentt" for p in specs]
        self.is_student = jnp.asarray(student)
        self.loc = jnp.asarray(
            [float(p.get("loc", 0.0)) if s else 0.0 for p, s in zip(specs, student, strict=True)],
            jnp.float32,
        )
        self.scale = jnp.asarray(
            [float(p["scale"]) if s else 1.0 for p, s in zip(specs, student, strict=True)],
            jnp.float32,
        )
        self.lower = self._to_chart(self.lo)
        self.upper = self._to_chart(self.hi)

    def _to_chart(self, x):
        return jnp.where(self.is_student, jnp.arcsinh((x - self.loc) / self.scale), x)

    @staticmethod
    def _logit_in_bounds(x, lo, hi):
        frac = (x - lo) / (hi - lo)
        eps = jnp.finfo(frac.dtype).eps
        frac = jnp.clip(frac, eps, 1 - eps)
        return jnp.log(frac) - jnp.log1p(-frac)

    def from_physical(self, x):
        """Encode bounded physical values in the chart's unconstrained coordinates."""
        return self._logit_in_bounds(self._to_chart(x), self.lower, self.upper)

    def from_sigmoid(self, theta):
        """Translate ordinary sigmoid coordinates into the chart."""
        x = self.lo + (self.hi - self.lo) * jax.nn.sigmoid(theta)
        return jnp.where(self.is_student, self.from_physical(x), theta)

    def to_sigmoid(self, theta):
        """Translate draws back to the catalogue HDF5 sigmoid coordinates."""
        bounded = self._logit_in_bounds(self.to_physical(theta), self.lo, self.hi)
        return jnp.where(self.is_student, bounded, theta)

    def to_physical(self, theta):
        """Decode a vector or batch of chart coordinates."""
        t = self.lower + (self.upper - self.lower) * jax.nn.sigmoid(theta)
        warped = self.loc + self.scale * jnp.sinh(jnp.where(self.is_student, t, 0.0))
        return jnp.where(self.is_student, warped, t)

    def log_det(self, theta):
        """Return the log determinant of the map to physical space."""
        t = self.lower + (self.upper - self.lower) * jax.nn.sigmoid(theta)
        log_cosh = jnp.logaddexp(t, -t) - np.log(2.0)
        return sigmoid_log_jacobian(theta, self.lower, self.upper) + jnp.sum(
            jnp.where(self.is_student, jnp.log(self.scale) + log_cosh, 0.0)
        )


def gauss_newton_metric(jacobian, prior_sd):
    """Invert ``J.T @ J + diag(1 / prior_sd**2)`` in prior-scaled coordinates.

    ``jacobian`` differentiates flux divided by observational error, with
    masked rows set to zero. Its null directions retain the prior scale.
    SVD avoids cancellation in the weak directions of ``J.T @ J``.
    """
    scaled = jacobian * prior_sd
    _, singular, vt = jnp.linalg.svd(scaled, full_matrices=True)
    padded = jnp.pad(singular, (0, scaled.shape[1] - singular.size))
    vectors = prior_sd[:, None] * vt.T
    return (vectors / (1.0 + padded**2)) @ vectors.T
