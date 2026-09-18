"""Client for the JADES NIRSpec DR4 spectroscopic redshift catalogue.

The catalogue lives at https://jades.herts.ac.uk/DR4/ .  The combined external
table ``Combined_DR4_external_v1.2.1.fits`` (release v5.1.1, 1 October 2025) is
a five-extension FITS file; extension 1, ``Obs_info``, holds the targeting
metadata and the best redshifts and is the one used here.  Extensions 2-5 hold
emission-line fluxes from the PRISM and R~1000 spectra.

Column mapping (source -> standardised)
---------------------------------------
======================  ==========  ==================================================
Source column           Standard    Notes
======================  ==========  ==================================================
``Unique_ID``           ``id``      Unique per target/tier; ``NIRSpec_ID`` is *not*
                                    unique and is kept as ``nirspec_id``.
``RA_TARG``             ``ra``      Degrees.
``Dec_TARG``            ``dec``     Degrees.
``z_Spec``              ``z_spec``  Best NIRSpec redshift; ``-1`` means "no redshift".
``z_Spec_flag``         ``z_flag``  A/B highly robust, C secure, D tentative, E none.
``Field``               ``field``   ``'GS'`` -> ``'GOODS-S'``, ``'GN'`` -> ``'GOODS-N'``.
======================  ==========  ==================================================

``z_phot``, ``TIER``, ``PID`` and ``NIRCam_DR3_ID``/``NIRCam_DR5_ID`` are carried
through as ``z_phot``, ``tier``, ``pid``, ``nircam_dr3_id`` and
``nircam_dr5_id`` when present.

Contents of the v1.2.1 file: 5190 rows; 3387 in GOODS-S and 1803 in GOODS-N;
flags A 2651, B 207, C 439, D 493, E 1400; 2858 rows have a flag of A or B with
a positive redshift (1795 in GOODS-S, 1063 in GOODS-N); redshifts run to 14.18.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy.table import Table

from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

#: Base URL of the JADES DR4 data release.
JADES_DR4_BASE_URL = "https://jades.herts.ac.uk/DR4"
#: Combined external catalogue used by :func:`download_jades_dr4_specz`.
JADES_DR4_SPECZ_URL = f"{JADES_DR4_BASE_URL}/Combined_DR4_external_v1.2.1.fits"
#: Human-readable description of the catalogue columns.
JADES_DR4_README_URL = f"{JADES_DR4_BASE_URL}/Readme_DR4_catalogues.md"

#: Default cache directory for the downloaded catalogue.
DEFAULT_CACHE_DIR = Path("~/.cache/arachne/jades").expanduser()

#: Redshift-quality flags considered secure, by ``quality`` keyword.
QUALITY_FLAGS = {
    "best": ("A", "B"),
    "secure": ("A", "B", "C"),
    "tentative": ("A", "B", "C", "D"),
    "any": ("A", "B", "C", "D", "E"),
}

#: Field-code to full-name mapping.
FIELD_NAMES = {"GS": "GOODS-S", "GN": "GOODS-N"}

_COLUMN_MAP = {
    "Unique_ID": "id",
    "RA_TARG": "ra",
    "Dec_TARG": "dec",
    "z_Spec": "z_spec",
    "z_Spec_flag": "z_flag",
    "Field": "field",
}
_EXTRA_COLUMNS = {
    "NIRSpec_ID": "nirspec_id",
    "PID": "pid",
    "TIER": "tier",
    "z_phot": "z_phot",
    "NIRCam_DR3_ID": "nircam_dr3_id",
    "NIRCam_DR5_ID": "nircam_dr5_id",
}


def _as_str_array(column) -> np.ndarray:
    """Decode a possibly byte-typed table column to stripped unicode strings.

    Args:
        column: Table column.

    Returns:
        Numpy array of stripped ``str``.
    """
    values = np.asarray(column)
    if values.dtype.kind == "S":
        values = np.char.decode(values, "utf-8")
    return np.char.strip(values.astype(str))


def download_jades_dr4_specz(
    cache_dir: Optional[Union[str, Path]] = None,
    url: str = JADES_DR4_SPECZ_URL,
    timeout: float = 600.0,
    overwrite: bool = False,
    session=None,
) -> Path:
    """Download the JADES DR4 combined spectroscopic catalogue, once.

    Args:
        cache_dir: Directory to cache the FITS file in.
        url: Catalogue URL.
        timeout: Request timeout in seconds.
        overwrite: Re-download even if a cached copy exists.
        session: Optional :class:`requests.Session`.

    Returns:
        Path to the cached FITS file.

    Raises:
        RuntimeError: If the download fails.
    """
    import requests

    directory = Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / url.rsplit("/", 1)[-1]
    if target.exists() and not overwrite and target.stat().st_size > 0:
        logger.info(f"Using cached JADES catalogue {target}")
        return target

    logger.info(f"Downloading JADES DR4 catalogue from {url}")
    getter = session.get if session is not None else requests.get
    try:
        response = getter(
            url,
            timeout=timeout,
            stream=True,
            headers={"User-Agent": "arachne/0.1.0 (https://github.com/arachne-project/arachne)"},
        )
        response.raise_for_status()
        tmp = target.with_suffix(target.suffix + ".part")
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
        tmp.replace(target)
    except Exception as exc:
        raise RuntimeError(f"Failed to download the JADES DR4 catalogue from {url}: {exc}") from exc

    logger.info(f"Cached JADES catalogue to {target} ({target.stat().st_size / 1e6:.2f} MB)")
    return target


def load_jades_dr4_specz(
    path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    hdu: int = 1,
    download: bool = True,
) -> Table:
    """Load the JADES DR4 redshift catalogue with standardised column names.

    Args:
        path: Path to an already-downloaded catalogue.  If None, the cached copy
            is used or downloaded.
        cache_dir: Cache directory used when ``path`` is None.
        hdu: FITS extension to read.  Extension 1 (``Obs_info``) carries the
            redshifts and positions.
        download: Whether to download the catalogue when it is not cached.

    Returns:
        Table with columns ``id, ra, dec, z_spec, z_flag, field`` plus any of
        ``nirspec_id, pid, tier, z_phot, nircam_dr3_id, nircam_dr5_id`` present
        in the source file.  ``field`` uses the full names ``'GOODS-S'`` and
        ``'GOODS-N'``, and ``z_spec`` is NaN where the source records ``-1``.

    Raises:
        FileNotFoundError: If no catalogue is available and ``download`` is False.
        KeyError: If the file lacks the expected columns.
    """
    if path is None:
        directory = Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_CACHE_DIR
        candidate = directory / JADES_DR4_SPECZ_URL.rsplit("/", 1)[-1]
        if candidate.exists():
            path = candidate
        elif download:
            path = download_jades_dr4_specz(cache_dir=cache_dir)
        else:
            raise FileNotFoundError(
                f"No cached JADES catalogue at {candidate}; pass path= or set download=True."
            )

    raw = Table.read(path, hdu=hdu)
    missing = [c for c in _COLUMN_MAP if c not in raw.colnames]
    if missing:
        raise KeyError(
            f"JADES catalogue {path} (hdu={hdu}) is missing expected columns {missing}. "
            f"Available: {raw.colnames}"
        )

    out = Table()
    out["id"] = _as_str_array(raw["Unique_ID"])
    out["ra"] = np.asarray(raw["RA_TARG"], dtype=np.float64)
    out["dec"] = np.asarray(raw["Dec_TARG"], dtype=np.float64)

    z = np.asarray(raw["z_Spec"], dtype=np.float64).copy()
    z[z <= 0] = np.nan  # the catalogue writes -1 for "no redshift"
    out["z_spec"] = z

    out["z_flag"] = _as_str_array(raw["z_Spec_flag"])
    field = _as_str_array(raw["Field"])
    out["field"] = np.array([FIELD_NAMES.get(f, f) for f in field])

    for source, name in _EXTRA_COLUMNS.items():
        if source in raw.colnames:
            column = raw[source]
            if np.asarray(column).dtype.kind in ("S", "U"):
                out[name] = _as_str_array(column)
            else:
                out[name] = np.asarray(column)

    out.meta["source_file"] = str(path)
    out.meta["source_hdu"] = hdu
    logger.info(f"Loaded JADES DR4 catalogue: {len(out)} rows from {path}")
    return out


def select_targets(
    table: Table,
    field: Optional[str] = "GOODS-S",
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    quality: str = "best",
    n: Optional[int] = 10,
    seed: int = 0,
) -> Table:
    """Select secure-redshift targets from a standardised JADES table.

    Args:
        table: Table from :func:`load_jades_dr4_specz`.
        field: Field name (``'GOODS-S'``, ``'GOODS-N'``, or the short ``'GS'``
            / ``'GN'``).  None keeps both fields.
        z_min: Optional lower redshift bound (inclusive).
        z_max: Optional upper redshift bound (inclusive).
        quality: One of ``'best'`` (flags A, B), ``'secure'`` (A, B, C),
            ``'tentative'`` (A-D) or ``'any'``.
        n: Number of targets to return.  None returns all matches.  When fewer
            than ``n`` match, all of them are returned.
        seed: Seed for the deterministic random subsample.

    Returns:
        Table of the selected rows, sorted by ``id``.

    Raises:
        ValueError: If ``quality`` is not recognised.
    """
    if quality not in QUALITY_FLAGS:
        raise ValueError(f"quality must be one of {sorted(QUALITY_FLAGS)}, got {quality!r}.")

    z = np.asarray(table["z_spec"], dtype=np.float64)
    keep = np.isfinite(z)
    keep &= np.isfinite(np.asarray(table["ra"], dtype=np.float64))
    keep &= np.isfinite(np.asarray(table["dec"], dtype=np.float64))
    keep &= np.isin(_as_str_array(table["z_flag"]), QUALITY_FLAGS[quality])

    if field is not None:
        wanted = FIELD_NAMES.get(str(field).upper(), str(field))
        keep &= _as_str_array(table["field"]) == wanted
    if z_min is not None:
        keep &= z >= float(z_min)
    if z_max is not None:
        keep &= z <= float(z_max)

    selected = table[keep]
    logger.info(
        f"select_targets: {len(selected)} rows match field={field}, quality={quality}, "
        f"z in [{z_min}, {z_max}]"
    )
    if n is not None and len(selected) > n:
        rng = np.random.default_rng(seed)
        index = rng.choice(len(selected), size=int(n), replace=False)
        selected = selected[np.sort(index)]

    selected.sort("id")
    return selected
