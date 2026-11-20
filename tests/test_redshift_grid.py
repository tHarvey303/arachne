"""Redshift-grid starting points for catalogue fits, on a toy emulator with a spectral break."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.emulator.base import SPSEmulator
from arachne.inference.redshift_grid import default_templates, redshift_grid

LAM = np.array([0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.2])  # band wavelengths [micron]


class BreakEmulator(SPSEmulator, eqx.Module):
    """Flux linear in mass with a soft break at rest 0.4 micron and dust reddening."""

    @property
    def param_names(self):
        """Inputs."""
        return ["redshift", "log_mass", "Av", "logsfr_ratio_0"]

    @property
    def band_names(self):
        """Bands."""
        return [f"b{i}" for i in range(len(LAM))]

    def predict(self, params):
        """(N, 4) -> (N, 8) fluxes."""
        z, logm, av, r = (params[:, i : i + 1] for i in range(4))
        rest = jnp.asarray(LAM)[None, :] / (1.0 + z)
        step = 0.3 + 0.7 * jnp.asarray(1.0) / (1.0 + jnp.exp(-(rest - 0.4) / 0.05))
        dust = 10.0 ** (-0.4 * av * (0.55 / rest))
        slope = rest ** (-0.5 * r)
        return 10.0 ** (logm - 8.0) * step * dust * slope / (1.0 + z) ** 2


BOUNDS = {"redshift": (0.01, 6.0), "log_mass": (6.0, 12.0)}


def _observe(emu, truth, noise=0.01, seed=0):
    flux = np.asarray(emu.predict(jnp.asarray(truth, jnp.float32)))
    rng = np.random.default_rng(seed)
    err = noise * flux
    return flux + rng.normal(0, 1, flux.shape) * err, err


def test_default_templates_cover_the_axes():
    """Templates vary dust and SFH, hold the rest at defaults and stay inside the domain."""
    spec = BreakEmulator().spec
    tmpl = default_templates(spec)
    assert len(tmpl) == 4 * 3  # no metallicity input in this emulator
    assert all(set(t) == {"Av", "logsfr_ratio_0"} for t in tmpl)
    lo, hi = spec.param_bounds["Av"]
    assert all(lo < t["Av"] < hi for t in tmpl)


def test_grid_recovers_redshift_and_mass():
    """The best start is at the true redshift with the analytically matched mass."""
    emu = BreakEmulator()
    truth = np.array([[0.6, 9.5, 0.4, 0.0], [2.1, 10.5, 1.0, 0.7], [3.4, 11.0, 0.05, -0.7]])
    obs, err = _observe(emu, truth)
    grid = redshift_grid(emu, range(len(LAM)), bounds=BOUNDS, n_z=200)
    assert grid.flux.shape == (12, 200, len(LAM))
    x, chi2 = grid.starts(obs, err, k=3)
    assert x.shape == (3, 3, 4) and chi2.shape == (3, 3)
    assert np.all(np.diff(chi2, axis=1)[np.isfinite(np.diff(chi2, axis=1))] >= 0)  # best first
    np.testing.assert_allclose(x[:, 0, 0], truth[:, 0], rtol=0.03)
    np.testing.assert_allclose(x[:, 0, 1], truth[:, 1], atol=0.05)
    # the starts of one galaxy are separated in ln(1 + z)
    sep = np.abs(np.diff(np.sort(np.log1p(x[:, :, 0]), axis=1), axis=1))
    assert np.all(sep[np.isfinite(chi2[:, 1:])] >= grid.min_separation - 1e-9)


def test_masked_bands_and_mass_bounds():
    """Masked bands are ignored, and a mass outside the bounds is never proposed."""
    emu = BreakEmulator()
    truth = np.array([[1.2, 9.0, 0.4, 0.0]])
    obs, err = _observe(emu, truth)
    bad = obs.copy()
    bad[0, 2] = 1e6
    valid = np.ones_like(obs, dtype=bool)
    valid[0, 2] = False
    grid = redshift_grid(emu, range(len(LAM)), bounds=BOUNDS, n_z=200)
    x, _ = grid.starts(bad, err, k=2, valid=valid)
    assert x[0, 0, 0] == pytest.approx(1.2, rel=0.03)
    tight = redshift_grid(emu, range(len(LAM)), bounds={**BOUNDS, "log_mass": (6.0, 8.0)}, n_z=50)
    x, chi2 = tight.starts(obs, err, k=2)
    assert np.all((x[..., 1] >= 6.0) & (x[..., 1] <= 8.0))
    assert np.all(np.isfinite(x))


def test_grid_needs_redshift_and_mass():
    """An emulator without a redshift input cannot use the grid."""

    class NoZ(BreakEmulator):
        @property
        def param_names(self):
            return ["log_mass", "Av"]

    with pytest.raises(ValueError, match="redshift"):
        redshift_grid(NoZ(), [0, 1])
