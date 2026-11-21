"""Checks for transferring physical Matérn calibration into model pixels."""

import json
import math

import pytest

from arachne.priors import load_spatial_calibration


@pytest.fixture
def calibration(tmp_path):
    """Investigation result with one usable field and a diagnostic SFH measurement."""
    result = {
        "version": 1,
        "nu": 1.5,
        "redshift_range": [1, 3],
        "transferable_fields": ["Av"],
        "fields": {
            "Av": {
                "amplitude": {"median": 0.35, "log_scatter": 0.6},
                "length": {
                    "re": {"median": 0.25, "log_scatter": 0.9},
                    "kpc": {"median": 0.5, "log_scatter": 0.4},
                },
            },
            "logssfr_100myr": {},
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(result))
    return path


def test_radius_and_physical_units(calibration):
    """Match radius-normalized and physical lengths for a two-kpc galaxy."""
    local = load_spatial_calibration(calibration, redshift=2, re_pixels=16)
    physical = load_spatial_calibration(calibration, redshift=2, kpc_per_pixel=0.125)
    assert (
        local
        == physical
        == {
            "corr_lengths": {"Av": 4.0},
            "amplitudes": {"Av": 0.35},
            "nus": {"Av": 1.5},
        }
    )


def test_population_scatter_in_log_units(calibration):
    """A change in pixel scale shifts log length without rescaling log scatter."""
    spatial = load_spatial_calibration(
        calibration, redshift=2, re_pixels=16, sample_hyperparameters=True
    )
    assert spatial["hyper_priors"]["Av"] == {
        "log_corr_length": {"loc": math.log(4), "scale": 0.9},
        "log_amplitude": {"loc": math.log(0.35), "scale": 0.6},
    }


def test_requires_explicit_extrapolation_and_valid_units(calibration):
    """Reject an unsupported target and inconsistent geometry input."""
    with pytest.raises(ValueError, match="outside"):
        load_spatial_calibration(calibration, redshift=4, re_pixels=4)
    assert load_spatial_calibration(calibration, redshift=4, re_pixels=4, allow_extrapolation=True)[
        "corr_lengths"
    ] == {"Av": 1}
    with pytest.raises(ValueError, match="exactly one"):
        load_spatial_calibration(calibration, redshift=2, re_pixels=8, kpc_per_pixel=0.2)
    with pytest.raises(ValueError, match="positive"):
        load_spatial_calibration(calibration, redshift=2, kpc_per_pixel=0)
    with pytest.raises(ValueError, match="without a direct calibration"):
        load_spatial_calibration(calibration, redshift=2, re_pixels=8, fields=["logssfr_100myr"])
