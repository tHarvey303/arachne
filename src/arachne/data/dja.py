"""Client for the DAWN JWST Archive (DJA) cutout and mosaic services.

Two routes to pixels are provided.

**Thumbnail cutouts** (:func:`fetch_dja_cutout`) hit
``https://grizli-cutout.herokuapp.com/thumb``.  This is fast and needs no
authentication, but the server resamples *every* filter onto one common
0.05 arcsec/pixel grid, so the result is effectively single-resolution.  Probed
on 2026-09-16, the endpoint ignores ``pixscale``, ``pixel_scale``, ``scale``,
``pixscale_mas`` and ``native`` query parameters, and the API documentation
lists no pixel-scale option.

**Native sub-mosaics** (:func:`fetch_dja_native_cutout`) use the ``assoc_mosaic``
endpoint, which lists per-association drizzled mosaics at their native scales:
20 mas for NIRCam short-wavelength and 40 mas for long-wavelength in DJA v7.
Those are whole ``.fits.gz`` sub-mosaics of roughly 35 MB each (about 4800x4800
pixels), so this route downloads and caches the full files and then cuts out the
region of interest locally.

What a DJA thumbnail FITS actually contains
-------------------------------------------
For ``?ra=53.1625&dec=-27.7914&size=3&filters=f115w-clear,f200w-clear,f444w-clear
&output=fits_weight``:

* Six HDUs: the *primary* HDU is the first filter's science image (it is not
  empty), followed by five ``ImageHDU``s.
* ``EXTNAME`` is the filter, upper case with the pupil: ``'F115W-CLEAR'``.
  ``EXTVER`` is the string ``'SCI'`` or ``'WHT'`` -- not an integer.
* ``WHT`` extensions are inverse-variance maps in ``(pixel value)^-2``.
* ``BUNIT = '10.0*nanoJansky'`` and ``PHOTFNU = 1e-8`` (Jy per pixel value) in
  every extension; full TAN WCS with a CD matrix; identical CRPIX/CRVAL across
  filters.
* All three filters came back 120x120 pixels at 0.05000 arcsec/pixel.
* ``size`` is a **half**-width in arcsec: ``size=3`` gives 6 arcsec (120 px at
  0.05), ``size=2`` gives 4 arcsec (80 px).  :func:`fetch_dja_cutout` takes the
  **full** width in ``size_arcsec`` and halves it for the query.
* The ``ZP`` card is **inconsistent** with ``BUNIT``: it equals
  ``-2.5*log10(OPHOTFNU) + 8.9``, the zeropoint of the original mosaic before
  the server rescaled the pixels (F115W 27.998, F200W 28.002, F444W 26.529 vs
  the 28.900 implied by ``PHOTFNU``).  Use BUNIT, not ZP.
"""

from __future__ import annotations

import csv
import hashlib
import io
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Union
from urllib.parse import urlencode

import numpy as np

from arachne.data.multires import MultiResolutionObservation, canonical_band_name
from arachne.utils.logging import setup_named_logger

logger = setup_named_logger(__name__)

#: Base URL of the DJA cutout / query service.
DJA_BASE_URL = "https://grizli-cutout.herokuapp.com"
#: Thumbnail cutout endpoint.
DJA_THUMB_URL = f"{DJA_BASE_URL}/thumb"
#: Association-mosaic query endpoint.
DJA_ASSOC_MOSAIC_URL = f"{DJA_BASE_URL}/assoc_mosaic"

#: Default cache directory for downloaded DJA products.
DEFAULT_CACHE_DIR = Path("~/.cache/arachne/dja").expanduser()

#: User agent identifying arachne to the DJA servers.
USER_AGENT = "arachne/0.1.0 (https://github.com/arachne-project/arachne)"

# The DJA service is a single small Heroku dyno; serialise our requests.
_REQUEST_LOCK = threading.Lock()


def _cache_dir(cache_dir: Optional[Union[str, Path]]) -> Path:
    """Resolve and create the cache directory.

    Args:
        cache_dir: Explicit directory, or None for :data:`DEFAULT_CACHE_DIR`.

    Returns:
        Existing directory path.
    """
    path = Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def server_reachable(url: str = DJA_BASE_URL, timeout: float = 5.0) -> bool:
    """Probe whether an HTTP(S) endpoint answers, for skipping network tests.

    Tries a HEAD request and falls back to a ranged GET, since some servers
    (and S3) reject HEAD or answer it differently.

    Args:
        url: URL to probe.
        timeout: Timeout in seconds.

    Returns:
        True if the server answered with any HTTP status, False on any error.
    """
    try:
        import requests
    except ImportError:  # pragma: no cover - requests is a hard dependency here
        return False
    headers = {"User-Agent": USER_AGENT}
    for method in ("head", "get"):
        try:
            kwargs = {"timeout": timeout, "headers": dict(headers), "allow_redirects": True}
            if method == "get":
                kwargs["headers"]["Range"] = "bytes=0-0"
                kwargs["stream"] = True
            response = getattr(requests, method)(url, **kwargs)
            response.close()
            return response.status_code < 500
        except Exception:
            continue
    return False


def dja_filter_names(band_names: Union[str, Iterable[str]]) -> Union[str, list[str]]:
    """Convert arachne band names to DJA filter strings.

    ``'JWST/NIRCam.F200W'`` becomes ``'f200w-clear'``.  A string that already
    looks like a DJA filter is lower-cased and returned unchanged.  NIRCam
    filters gain the ``-clear`` pupil suffix; MIRI filters do not.

    Args:
        band_names: A single band name or an iterable of them.

    Returns:
        The corresponding DJA filter string, or a list of them if an iterable
        was given.
    """
    if isinstance(band_names, str):
        return _dja_filter_name(band_names)
    return [_dja_filter_name(b) for b in band_names]


def _dja_filter_name(band_name: str) -> str:
    """Convert one band name to a DJA filter string.

    Args:
        band_name: e.g. ``'JWST/NIRCam.F200W'`` or ``'f200w-clear'``.

    Returns:
        DJA filter string, e.g. ``'f200w-clear'``.
    """
    name = str(band_name).strip()
    if "/" in name:
        name = name.split(".")[-1]
    name = name.lower()
    if "-" in name:
        return name
    if name.startswith("f") and "miri" not in str(band_name).lower():
        return f"{name}-clear"
    return name


def band_names_from_dja_filters(filters: Union[str, Iterable[str]]) -> Union[str, list[str]]:
    """Convert DJA filter strings back to arachne band names.

    ``'f200w-clear'`` becomes ``'JWST/NIRCam.F200W'``.

    Args:
        filters: A single DJA filter string or an iterable of them.

    Returns:
        Canonical band name, or a list of them.
    """
    if isinstance(filters, str):
        return canonical_band_name(filters)
    return [canonical_band_name(f) for f in filters]


def _get(
    url: str,
    params: Optional[dict] = None,
    timeout: float = 120.0,
    session=None,
    max_retries: int = 4,
    stream: bool = False,
):
    """Perform a polite GET with retries and exponential backoff on 5xx.

    Requests are serialised through a module-level lock so that arachne never
    hammers the DJA dyno with concurrent queries.

    Args:
        url: URL to fetch.
        params: Optional query parameters.
        timeout: Per-request timeout in seconds.
        session: Optional :class:`requests.Session` to reuse.
        max_retries: Number of attempts before giving up.
        stream: Whether to stream the response body.

    Returns:
        The :class:`requests.Response`.

    Raises:
        RuntimeError: If every attempt fails.
    """
    import requests

    getter = session.get if session is not None else requests.get
    delay = 2.0
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with _REQUEST_LOCK:
                response = getter(
                    url,
                    params=params,
                    timeout=timeout,
                    headers={"User-Agent": USER_AGENT},
                    stream=stream,
                )
            if response.status_code >= 500:
                last_error = RuntimeError(f"HTTP {response.status_code} from {response.url}")
                response.close()
                logger.warning(
                    f"DJA returned {response.status_code} (attempt {attempt}/{max_retries}); "
                    f"retrying in {delay:.0f}s."
                )
            else:
                response.raise_for_status()
                return response
        except Exception as exc:  # network error or 4xx
            last_error = exc
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                if exc.response.status_code < 500:
                    raise
            logger.warning(
                f"DJA request failed (attempt {attempt}/{max_retries}): {exc}; "
                f"retrying in {delay:.0f}s."
            )
        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError(f"DJA request to {url} failed after {max_retries} attempts: {last_error}")


def fetch_dja_cutout(
    ra: float,
    dec: float,
    size_arcsec: float,
    filters: Iterable[str],
    output: str = "fits_weight",
    cache_dir: Optional[Union[str, Path]] = None,
    timeout: float = 120.0,
    session=None,
    overwrite: bool = False,
) -> Path:
    """Download a multi-extension FITS cutout from the DJA thumbnail server.

    The result is cached under ``cache_dir`` keyed by a hash of the query, so a
    repeated call with the same arguments does no network I/O.

    Args:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        size_arcsec: **Full** width of the cutout in arcsec.  The DJA ``size``
            parameter is a half-width, so ``size_arcsec / 2`` is sent.
        filters: Band names (``'JWST/NIRCam.F200W'``) or DJA filter strings
            (``'f200w-clear'``); band names are converted automatically.
        output: ``"fits_weight"`` for SCI + inverse-variance WHT extensions, or
            ``"fits"`` for science only.
        cache_dir: Directory for the cached FITS file.
        timeout: Per-request timeout in seconds.
        session: Optional :class:`requests.Session`.
        overwrite: Re-download even if a cached file exists.

    Returns:
        Path to the cached FITS file.
    """
    filter_list = [_dja_filter_name(f) for f in filters]
    params = {
        "ra": f"{float(ra):.7f}",
        "dec": f"{float(dec):.7f}",
        "size": f"{float(size_arcsec) / 2.0:.4f}",
        "filters": ",".join(filter_list),
        "output": output,
    }
    key = hashlib.sha1(urlencode(sorted(params.items())).encode()).hexdigest()[:16]
    target = _cache_dir(cache_dir) / f"dja_thumb_{key}.fits"
    if target.exists() and not overwrite and target.stat().st_size > 0:
        logger.info(f"Using cached DJA cutout {target}")
        return target

    logger.info(
        f"Fetching DJA cutout at ({ra}, {dec}), {size_arcsec} arcsec wide, filters {filter_list}"
    )
    response = _get(DJA_THUMB_URL, params=params, timeout=timeout, session=session)
    content = response.content
    if not content.startswith(b"SIMPLE"):
        raise RuntimeError(
            f"DJA did not return a FITS file for {response.url} (first bytes: {content[:80]!r})"
        )
    tmp = target.with_suffix(".part")
    tmp.write_bytes(content)
    tmp.replace(target)
    logger.info(f"Cached DJA cutout to {target} ({target.stat().st_size / 1e6:.2f} MB)")
    return target


def load_dja_cutout(
    path: Union[str, Path],
    ref_ra: float,
    ref_dec: float,
    psfs: Optional[dict[str, tuple[np.ndarray, float]]] = None,
    band_name_map: Optional[dict[str, str]] = None,
    size_arcsec: Optional[float] = None,
) -> MultiResolutionObservation:
    """Load a downloaded DJA cutout into a :class:`MultiResolutionObservation`.

    Args:
        path: Path returned by :func:`fetch_dja_cutout`.
        ref_ra: Reference right ascension in degrees.
        ref_dec: Reference declination in degrees.
        psfs: Optional mapping band name -> ``(kernel, psf_pixel_scale)``.
        band_name_map: Optional explicit EXTNAME -> band name mapping.
        size_arcsec: Optional full cutout width in arcsec to trim to.

    Returns:
        MultiResolutionObservation in nJy.
    """
    return MultiResolutionObservation.from_dja_fits(
        path,
        ref_ra=ref_ra,
        ref_dec=ref_dec,
        psfs=psfs,
        band_name_map=band_name_map,
        size_arcsec=size_arcsec,
    )


# ---------------------------------------------------------------------------
# Native-resolution sub-mosaics
# ---------------------------------------------------------------------------


def query_dja_assoc_mosaic(
    ra: float,
    dec: float,
    filters: Iterable[str],
    timeout: float = 120.0,
    session=None,
) -> list[dict]:
    """Query the ``assoc_mosaic`` endpoint for native-resolution sub-mosaics.

    Args:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        filters: Band names or DJA filter strings.
        timeout: Request timeout in seconds.
        session: Optional :class:`requests.Session`.

    Returns:
        List of row dicts with keys including ``file``, ``filter``,
        ``pixscale_mas``, ``exptime`` and ``version``.
    """
    filter_list = [_dja_filter_name(f).upper() for f in filters]
    params = {
        "coords": f"{float(ra):.7f},{float(dec):.7f}",
        "filters": ",".join(filter_list),
        "output": "csv",
    }
    response = _get(DJA_ASSOC_MOSAIC_URL, params=params, timeout=timeout, session=session)
    rows = list(csv.DictReader(io.StringIO(response.text)))
    logger.info(f"assoc_mosaic returned {len(rows)} rows for filters {filter_list}")
    return rows


def fetch_dja_mosaic_file(
    url: str,
    cache_dir: Optional[Union[str, Path]] = None,
    timeout: float = 600.0,
    session=None,
    overwrite: bool = False,
) -> Path:
    """Download and cache one native sub-mosaic ``.fits.gz`` file.

    These files are roughly 35 MB each; the science and weight images of one
    filter therefore cost about 70 MB.

    Args:
        url: S3 URL of the ``_sci.fits.gz`` or ``_wht.fits.gz`` file.
        cache_dir: Cache directory.
        timeout: Request timeout in seconds.
        session: Optional :class:`requests.Session`.
        overwrite: Re-download even if cached.

    Returns:
        Path to the cached file.
    """
    target = _cache_dir(cache_dir) / "assoc_mosaic" / url.rsplit("/", 1)[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite and target.stat().st_size > 0:
        logger.info(f"Using cached mosaic {target}")
        return target

    logger.info(f"Downloading native sub-mosaic {url}")
    response = _get(url, timeout=timeout, session=session, stream=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with open(tmp, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            handle.write(chunk)
    tmp.replace(target)
    logger.info(f"Cached {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def fetch_dja_native_cutout(
    ra: float,
    dec: float,
    size_arcsec: float,
    filters: Iterable[str],
    cache_dir: Optional[Union[str, Path]] = None,
    timeout: float = 600.0,
    session=None,
    psfs: Optional[dict[str, tuple[np.ndarray, float]]] = None,
    with_weights: bool = True,
) -> MultiResolutionObservation:
    """Build a genuinely multi-resolution observation from DJA native sub-mosaics.

    For each requested filter the deepest overlapping association is selected
    from :func:`query_dja_assoc_mosaic`, its ``_sci.fits.gz`` (and matching
    ``_wht.fits.gz``) sub-mosaic is downloaded and cached, and the requested
    region is cut out locally.  NIRCam short-wavelength filters come back at
    0.02 arcsec/pixel and long-wavelength filters at 0.04 arcsec/pixel, so the
    bands really do sit on different grids.

    Each sub-mosaic is about 35 MB compressed, so a two-filter request with
    weights downloads roughly 140 MB the first time and nothing thereafter.

    Args:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        size_arcsec: Full width of the cutout in arcsec.
        filters: Band names or DJA filter strings.
        cache_dir: Cache directory for the downloaded mosaics.
        timeout: Per-request timeout in seconds.
        session: Optional :class:`requests.Session`.
        psfs: Optional mapping band name -> ``(kernel, psf_pixel_scale)``.
        with_weights: Also download the inverse-variance ``_wht`` images.

    Returns:
        MultiResolutionObservation with one band per requested filter.

    Raises:
        RuntimeError: If a requested filter has no sub-mosaic at that position.
    """
    rows = query_dja_assoc_mosaic(ra, dec, filters, timeout=timeout, session=session)
    wanted = [_dja_filter_name(f).upper() for f in filters]

    flux_paths: dict[str, Union[str, Path]] = {}
    weight_paths: dict[str, Union[str, Path]] = {}
    for filt in wanted:
        candidates = [r for r in rows if str(r.get("filter", "")).upper() == filt]
        if not candidates:
            raise RuntimeError(
                f"No DJA assoc_mosaic sub-mosaic for filter {filt} at ({ra}, {dec})."
            )
        best = max(candidates, key=lambda r: float(r.get("exptime") or 0.0))
        band = canonical_band_name(filt)
        logger.info(
            f"{band}: using association {best.get('assoc_name')} "
            f"({best.get('pixscale_mas')} mas, exptime {best.get('exptime')} s)"
        )
        flux_paths[band] = fetch_dja_mosaic_file(
            best["file"], cache_dir=cache_dir, timeout=timeout, session=session
        )
        if with_weights:
            weight_paths[band] = fetch_dja_mosaic_file(
                best["file"].replace("_sci.fits.gz", "_wht.fits.gz"),
                cache_dir=cache_dir,
                timeout=timeout,
                session=session,
            )

    return MultiResolutionObservation.from_fits(
        flux_paths=flux_paths,
        ref_ra=ra,
        ref_dec=dec,
        size_arcsec=size_arcsec,
        weight_paths=weight_paths or None,
        psfs=psfs,
    )
