"""Static band and parameter metadata for results bundles (numpy only).

Pivot wavelengths are approximate (~0.5 per cent) and meant for plotting and for ordering bands,
not for synthetic photometry.  Band names follow the SVO convention used throughout arachne
(``"JWST/NIRCam.F115W"``); lookups also accept a bare filter name (``"F115W"``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_NIRCAM = {
    # values shared with examples/demo_correlated_field.py (NIRCAM_PIVOT_UM)
    "F070W": 0.704, "F090W": 0.902, "F115W": 1.154, "F150W": 1.501, "F162M": 1.627,
    "F182M": 1.845, "F200W": 1.988, "F210M": 2.096, "F250M": 2.503, "F277W": 2.776,
    "F300M": 2.996, "F335M": 3.365, "F356W": 3.564, "F410M": 4.092, "F430M": 4.280,
    "F444W": 4.402, "F460M": 4.630, "F480M": 4.817,
    # remaining wide / medium / narrow bands (JDox)
    "F140M": 1.404, "F150W2": 1.659, "F164N": 1.644, "F187N": 1.874, "F212N": 2.120,
    "F322W2": 3.247, "F323N": 3.237, "F360M": 3.621, "F405N": 4.055, "F466N": 4.654,
    "F470N": 4.708,
}  # fmt: skip
_MIRI = {
    "F560W": 5.635, "F770W": 7.639, "F1000W": 9.953, "F1130W": 11.309, "F1280W": 12.810,
    "F1500W": 15.064, "F1800W": 17.984, "F2100W": 20.795, "F2550W": 25.365,
}  # fmt: skip
_WFC3_IR = {"F105W": 1.0552, "F110W": 1.1534, "F125W": 1.2486, "F140W": 1.3923, "F160W": 1.5369}
_ACS_WFC = {
    "F435W": 0.4329, "F475W": 0.4747, "F555W": 0.5361, "F606W": 0.5922, "F625W": 0.6311,
    "F775W": 0.7693, "F814W": 0.8045, "F850LP": 0.9031,
}  # fmt: skip
_WFC3_UVIS = {
    "F225W": 0.2359, "F275W": 0.2704, "F336W": 0.3355, "F390W": 0.3924, "F438W": 0.4326,
    "F475W": 0.4773, "F555W": 0.5308, "F606W": 0.5889, "F625W": 0.6242, "F775W": 0.7651,
    "F814W": 0.8039,
}  # fmt: skip

# Order matters for bare-filter lookups: earlier instruments win name clashes (e.g. F814W).
_INSTRUMENTS = (
    ("JWST/NIRCam", _NIRCAM),
    ("JWST/MIRI", _MIRI),
    ("HST/WFC3_IR", _WFC3_IR),
    ("HST/ACS_WFC", _ACS_WFC),
    ("HST/WFC3_UVIS", _WFC3_UVIS),
)

PIVOT_UM: dict[str, float] = {
    f"{inst}.{filt}": value for inst, table in _INSTRUMENTS for filt, value in table.items()
}
"""Pivot wavelength in micron keyed by SVO-style band name."""

_BY_SHORT: dict[str, float] = {}
for _inst, _table in _INSTRUMENTS:
    for _filt, _value in _table.items():
        _BY_SHORT.setdefault(_filt.upper(), _value)

DEFAULT_LABELS: dict[str, str] = {
    "log_mass": r"$\log M_*$",
    "total_log_mass": r"$\log M_{*,\rm tot}$",
    "Av": r"$A_V$",
    "log10metallicity": r"$\log Z$",
    "redshift": r"$z$",
    "slope": "slope",
    "fesc_lya": r"$f_{\rm esc}^{\rm Ly\alpha}$",
    "dust_bump_amplitude": "dust bump",
    **{f"logsfr_ratio_{i}": f"logsfr_ratio_{i}" for i in range(5)},
    "mu_y": r"$\mu_y$",
    "mu_x": r"$\mu_x$",
    "size_y": r"$\sigma_y$",
    "size_x": r"$\sigma_x$",
    "rho": r"$\rho$",
    "sersic_n": r"$n$",
}
"""Fallback LaTeX labels for common parameter names."""


def short_band_name(name: str) -> str:
    """Return the bare filter name of a band (``"JWST/NIRCam.F115W"`` -> ``"F115W"``).

    Args:
        name: Band name, SVO style or already short.

    Returns:
        The text after the last ``"."`` (or ``"/"`` when there is no dot).
    """
    name = str(name)
    if "." in name:
        return name.rsplit(".", 1)[1]
    return name.rsplit("/", 1)[-1]


def pivot_um(band_names: Sequence[str]) -> np.ndarray:
    """Look up pivot wavelengths for a list of bands.

    Args:
        band_names: Band names, SVO style (``"JWST/NIRCam.F115W"``) or bare (``"F115W"``).

    Returns:
        Array of shape (B,) in micron, NaN for unknown bands.
    """
    out = np.full(len(band_names), np.nan)
    for i, name in enumerate(band_names):
        if name in PIVOT_UM:
            out[i] = PIVOT_UM[name]
        else:
            out[i] = _BY_SHORT.get(short_band_name(name).upper(), np.nan)
    return out
