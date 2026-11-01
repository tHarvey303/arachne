"""Tests for PSF MTF diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from arachne.data.psf import PSFModel
from arachne.psf.resolution import mtf_profile, psf_k_max


def _gauss(sigma, n=33):
    y, x = np.mgrid[:n, :n] - n // 2
    k = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return (k / k.sum()).astype(np.float32)


def test_mtf_profile_gaussian():
    """MTF is 1 at k=0 and follows exp(-k^2 s^2 / 2)."""
    k, m = mtf_profile(_gauss(1.5), n_bins=32)
    assert k.shape == m.shape == (32,)
    s = 1.5
    sel = k < 1.0
    np.testing.assert_allclose(m[sel], np.exp(-0.5 * k[sel] ** 2 * s**2), atol=0.05)
    assert m[0] > 0.97


@pytest.mark.parametrize("sigma", [1.0, 1.8, 2.5])
def test_k_max_gaussian(sigma):
    """k_max matches sqrt(-2 ln t)/sigma within 5%."""
    t = 0.1
    psf = PSFModel(kernels=_gauss(sigma)[None], band_names=["a"])
    expected = np.sqrt(-2 * np.log(t)) / sigma
    assert psf_k_max(psf, t) == pytest.approx(expected, rel=0.05)


def test_sharpest_band_and_override():
    """The sharpest band wins unless a band is requested."""
    psf = PSFModel(kernels=np.stack([_gauss(2.5), _gauss(1.5)]), band_names=["wide", "sharp"])
    sharp = psf_k_max(psf, band="sharp")
    wide = psf_k_max(psf, band="wide")
    assert sharp > wide
    assert psf_k_max(psf) == pytest.approx(sharp)
    assert psf_k_max(psf, band=0) == pytest.approx(wide)


def test_nyquist_when_never_below():
    """A delta-function PSF never falls below the threshold."""
    k = np.zeros((1, 15, 15), np.float32)
    k[0, 7, 7] = 1.0
    assert psf_k_max(PSFModel(kernels=k, band_names=["d"])) == pytest.approx(np.pi)
