"""Tests for the template-based mass start."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from arachne.data.observation import ObservationCube
from arachne.emulator.base import SPSEmulator
from arachne.spatial.correlated_field import mass_map_from_image
from arachne.spatial.mass_start import (
    MassStart,
    default_sfh_dust_templates,
    template_mass_map,
)

NAMES = ["redshift", "log_mass", "Av", "logsfr_ratio_0"]
BANDS = ["F115W", "F150W", "F200W", "F277W", "F356W", "F444W"]
LAM = jnp.array([1.15, 1.5, 2.0, 2.77, 3.56, 4.44])
REF = {"redshift": 2.0, "Av": 0.4, "logsfr_ratio_0": 0.0}
TEMPLATES = [{}, *default_sfh_dust_templates()]  # index 0 = the reference itself


class ToyEmu(SPSEmulator, eqx.Module):
    """Mass-linear toy: Av reddens, a high recent SFR ratio makes the SED bluer/brighter."""

    _param_names: list[str] = eqx.field(static=True)
    _band_names: list[str] = eqx.field(static=True)

    @property
    def param_names(self):
        """Input names."""
        return self._param_names

    @property
    def band_names(self):
        """Band names."""
        return self._band_names

    def predict(self, params):
        """(N, 4) -> (N, 6) nJy."""
        z, logm, av, r = (params[:, i : i + 1] for i in range(4))
        beta = 0.8 - 1.0 * r
        tau = av * (LAM[None, :] / (0.55 * (1.0 + z))) ** -1.0
        return (
            10.0 ** (logm - 6.0) * LAM[None, :] ** beta * 10.0 ** (-0.4 * tau) * 10.0 ** (0.3 * r)
        )


def _mock(seed=0, H=48, W=48, snr_peak=50.0):
    emu = ToyEmu(_param_names=NAMES, _band_names=BANDS)
    y, x = np.mgrid[:H, :W]
    rr = np.hypot(x - W / 2, y - H / 2)
    logm = 8.0 + np.log10(np.exp(-rr / 8.0))
    av = 0.2 + 1.0 * (x / W)
    r = -1.0 + 2.0 * (y / H)
    rows = np.stack([np.full(H * W, 2.0), logm.ravel(), av.ravel(), r.ravel()], axis=1).astype(
        np.float32
    )
    flux = np.asarray(emu.predict(jnp.asarray(rows))).T.reshape(len(BANDS), H, W)
    sigma = flux[-1].max() / snr_peak
    rng = np.random.default_rng(seed)
    obs = ObservationCube(
        flux=(flux + rng.normal(0, sigma, flux.shape)).astype(np.float32),
        variance=np.full(flux.shape, sigma**2, np.float32),
        mask=np.ones(flux.shape, np.float32),
        band_names=BANDS,
        pixel_scale=0.03,
    )
    return emu, obs, logm, flux


def test_default_templates():
    """Grid size and keys."""
    t = default_sfh_dust_templates()
    assert len(t) == 9 and set(t[0]) == {"Av", "logsfr_ratio_0"}
    assert len(default_sfh_dust_templates((0.1,), (0.0, 1.0), "r")) == 2


def test_shapes_and_dtypes():
    """Output container shapes and dtypes."""
    emu, obs, _, _ = _mock()
    out = template_mass_map(obs, emu, REF, TEMPLATES)
    assert isinstance(out, MassStart)
    for a in (out.log_mass, out.correction, out.template_index, out.chi2_best):
        assert a.shape == (48, 48)
    assert out.log_mass.dtype == np.float32
    assert np.issubdtype(out.template_index.dtype, np.integer)
    assert len(out.templates) == 10
    base = mass_map_from_image(obs, emu, REF)
    np.testing.assert_allclose(out.log_mass, base + out.correction, atol=1e-5)


def test_template_start_beats_single_template():
    """Per-pixel templates cut the rms log-mass error over bright pixels."""
    emu, obs, logm, flux = _mock()
    out = template_mass_map(obs, emu, REF, TEMPLATES, snr_min=5.0)
    base = mass_map_from_image(obs, emu, REF)
    bright = flux[-1] > 0.02 * flux[-1].max()
    rms_base = np.sqrt(np.mean((base - logm)[bright] ** 2))
    rms_tmpl = np.sqrt(np.mean((out.log_mass - logm)[bright] ** 2))
    print(f"rms log-mass error: baseline {rms_base:.3f} dex, template {rms_tmpl:.3f} dex")
    assert rms_tmpl < rms_base


def test_correction_zero_at_low_snr():
    """Pixels below the S/N cut get no correction and template 0."""
    emu, obs, _, _ = _mock(snr_peak=50.0)
    out = template_mass_map(obs, emu, REF, TEMPLATES, snr_min=5.0)
    snr = np.sum(np.asarray(obs.flux), 0) / np.sqrt(len(BANDS) * float(obs.variance[0, 0, 0]))
    low = snr <= 5.0
    assert low.any()
    assert np.all(out.correction[low] == 0.0)
    assert np.all(out.template_index[low] == 0)
    # an unreachable cut disables the correction everywhere
    off = template_mass_map(obs, emu, REF, TEMPLATES, snr_min=1e9)
    assert np.all(off.correction == 0.0)
