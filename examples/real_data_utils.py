#!/usr/bin/env python3
r"""Helpers for ``examples/fit_jades_dja.py``: real-data plumbing and figures.

Everything here is presentation or data-wrangling: target resolution, DJA
cutout loading, PSF resampling, the JADES DR3 catalogue cross-match, and the
matplotlib figures.  The physics (model, priors, likelihood, sampling) lives in
``fit_jades_dja.py``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from arachne.data.dja import fetch_dja_cutout, load_dja_cutout
from arachne.data.observation import ObservationCube
from arachne.data.psf import PSFModel
from arachne.emulator.base import SPSEmulator

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

ARACHNE_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ARACHNE_ROOT / "scripts/outputs/emulators/parrot_emulator_v2.eqx"
PSF_DIR = Path("/cosma/apps/dp276/dc-harv3/synference/priv/JADES-DR3-GS")
DATA_ROOT = Path("/cosma7/data/dp276/dc-harv3/work/arachne_data")
DJA_CACHE = DATA_ROOT / "target_scan/dja_cache9"  # populated by the target scan
JADES_DR4_FILE = DATA_ROOT / "Combined_DR4_external_v1.2.1.fits"
JADES_DR3_PHOT = Path(
    "/cosma7/data/dp276/dc-harv3/work/catalogs/JADES_DR3_GS_Matched_Specz_total_flux_good.fits"
)

# The nine NIRCam bands the DJA GOODS-S mosaics, the emulator and PSF_DIR share.
DEFAULT_BANDS = [
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F115W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
    "JWST/NIRCam.F277W",
    "JWST/NIRCam.F335M",
    "JWST/NIRCam.F356W",
    "JWST/NIRCam.F410M",
    "JWST/NIRCam.F444W",
]

PIVOT_UM = {
    "F090W": 0.901, "F115W": 1.154, "F150W": 1.501, "F200W": 1.990, "F277W": 2.786,
    "F335M": 3.365, "F356W": 3.563, "F410M": 4.092, "F444W": 4.421,
}  # fmt: skip

# JADES DR4 flag A/B spec-z galaxies in GOODS-S, vetted for the demo: isolated,
# resolved, covered in all nine bands (see the Phase 2 target scan).
VETTED_TARGETS: dict[str, tuple[float, float, float]] = {
    # Showcase pair: small, smooth, isolated bulge+envelope systems (Phase 2b
    # small-galaxy scan; F200W 3.1 and 5.4 uJy, r_half 0.25", peak S/N 1300-1700).
    "goods-s-mediumjwst_57578": (53.1397386, -27.7632830, 2.2263),
    "goods-s-mediumjwst_47870": (53.1493036, -27.7885994, 1.9068),
    # The hard case: bright, clumpy, with a tidal plume running off the frame.
    "goods-s-mediumhst_12281": (53.1410369, -27.7668037, 1.8989),
    # The smooth control.
    "goods-s-mediumhst_44832": (53.1326553, -27.7323573, 1.5482),
    # Harder / messier: 208134 is a clumpy interacting system and 37881 sits
    # under a bright foreground spiral that floods the frame.
    "goods-s-mediumjwst_208134": (53.1556640, -27.7793702, 1.8468),
    "goods-s-mediumjwst_60217631": (53.1329263, -27.7458694, 1.6128),
    "goods-s-mediumjwst_37881": (53.1421626, -27.8138370, 2.3460),
    # spares
    "goods-s-mediumjwst_30347": (53.1676502, -27.8304101, 1.8974),
    "goods-s-mediumjwst_52091": (53.1999545, -27.7776820, 2.8312),
}
DEFAULT_TARGET_ORDER = list(VETTED_TARGETS)[:4]


@dataclass(frozen=True)
class Target:
    """A galaxy to fit: identifier, sky position and spectroscopic redshift."""

    id: str
    ra: float
    dec: float
    z: float
    source: str = "vetted"


def short_band(band: str) -> str:
    return band.split(".")[-1]


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def resolve_target(
    target_id: str | None, ra: float | None, dec: float | None, z: float | None
) -> Target:
    """Resolve a target from the vetted table, the JADES DR4 catalogue or explicit coordinates."""
    if ra is not None and dec is not None and z is not None:
        return Target(target_id or f"radec_{ra:.6f}_{dec:+.6f}", ra, dec, z, source="explicit")
    if target_id is None:
        raise ValueError("give --target, or all of --ra --dec --z")
    if target_id in VETTED_TARGETS:
        r, d, zz = VETTED_TARGETS[target_id]
        return Target(target_id, r, d, z if z is not None else zz)

    from arachne.data.jades import load_jades_dr4_specz

    path = JADES_DR4_FILE if JADES_DR4_FILE.exists() else None
    table = load_jades_dr4_specz(path)
    ids = np.asarray(table["id"]).astype(str)
    match = np.flatnonzero(ids == target_id)
    if match.size == 0:
        raise ValueError(f"target {target_id!r} is not in the vetted list nor in JADES DR4")
    row = table[int(match[0])]
    z_row = float(row["z_spec"]) if z is None else float(z)
    if not np.isfinite(z_row):
        raise ValueError(f"JADES DR4 has no spec-z for {target_id!r}; pass --z")
    return Target(target_id, float(row["ra"]), float(row["dec"]), z_row, source="jades_dr4")


# ---------------------------------------------------------------------------
# Observation, PSF, emulator
# ---------------------------------------------------------------------------


def load_dja_observation(target: Target, size_arcsec: float, bands: list[str], cache_dir: Path):
    """Fetch and load the DJA thumbnail cutout as a single-grid ObservationCube.

    The *full* nine-band set is always requested and the result subset
    afterwards, because the DJA cache key includes the filter list: asking for
    two bands would miss a cached nine-band cutout and go to the network, which
    the GPU nodes cannot reach.
    """
    request = list(dict.fromkeys(list(DEFAULT_BANDS) + list(bands)))
    path = fetch_dja_cutout(
        target.ra, target.dec, size_arcsec, request, cache_dir=cache_dir, timeout=180
    )
    multires = load_dja_cutout(path, target.ra, target.dec)
    available = list(multires.band_names)
    missing = [b for b in bands if b not in available]
    if missing:
        print(f"  WARNING: DJA returned no data for {missing}")
    got = [b for b in bands if b in available]
    obs = multires.to_observation_cube()
    if got != list(obs.band_names):
        idx = [list(obs.band_names).index(b) for b in got]
        obs = ObservationCube(
            flux=np.asarray(obs.flux)[idx],
            variance=np.asarray(obs.variance)[idx],
            mask=np.asarray(obs.mask)[idx],
            band_names=got,
            pixel_scale=float(obs.pixel_scale),
            wcs=obs.wcs,
        )
    return obs, got, path


def native_psf_kernels(bands: list[str]) -> dict[str, np.ndarray]:
    """The empirical JADES DR3 GOODS-S kernels on their native 0.03"/px grid."""
    paths = {b: PSF_DIR / f"{short_band(b)}_psf_norm.fits" for b in bands}
    missing = [b for b, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"no PSF file for {missing} under {PSF_DIR}")
    native = PSFModel.from_fits(paths)
    return {b: np.asarray(native.kernels[i], dtype=np.float64) for i, b in enumerate(bands)}


def _overlap_matrix(n_in: int, n_out: int, ratio: float) -> np.ndarray:
    """Fractional overlap of each output pixel with each input pixel, 1-D.

    Both grids are centred on their own midpoint; ``ratio`` is
    ``to_scale / from_scale`` (input pixels per output pixel).  Entry
    ``(j, i)`` is the length of the intersection of output pixel ``j`` with
    input pixel ``i``, measured in input pixels, so ``W @ arr`` integrates the
    piecewise-constant surface implied by ``arr`` over the output pixels and
    conserves flux exactly except where the output grid does not cover the
    input one.
    """
    centres = (np.arange(n_out) - (n_out - 1) / 2.0) * ratio + (n_in - 1) / 2.0
    lo_out = centres[:, None] - 0.5 * ratio
    hi_out = centres[:, None] + 0.5 * ratio
    idx = np.arange(n_in)[None, :]
    return np.clip(np.minimum(hi_out, idx + 0.5) - np.maximum(lo_out, idx - 0.5), 0.0, None)


def resample_psf(
    kernel: np.ndarray,
    from_scale: float,
    to_scale: float,
    method: str = "spline",
    broaden_arcsec: float = 0.0,
    out_size: int | None = None,
) -> np.ndarray:
    """Resample one PSF kernel to ``to_scale`` arcsec/px, optionally broadened.

    Two resamplers, so the choice can be tested rather than assumed:

    ``"spline"``
        :meth:`PSFModel.resample` — cubic-spline interpolation of the kernel
        treated as a point-sampled surface, times the pixel-area ratio, then
        renormalised.  This is the right operation if the stored kernel samples
        the PSF *surface*, and it preserves the centroid to <1e-4 arcsec, but on
        the critically sampled blue kernels it mis-integrates: the raw sum
        before renormalisation is 1.043 for F090W (1.001 for F444W), and the
        second moment shrinks by 2.5% (0.3%).
    ``"area"``
        Exact area-weighted rebinning (:func:`_overlap_matrix` on each axis) of
        the kernel treated as *pixel-integrated* fluxes.  Flux and first moment
        are conserved analytically; the price is the extra ``to_scale`` top-hat
        blur that this reconstruction implies.

    ``broaden_arcsec`` convolves the resampled kernel with a Gaussian of that
    sigma.  Its use here is the pixel-integration mismatch: the JADES kernels
    are integrated over 0.03" pixels while the DJA thumb's pixels are 0.05",
    a variance difference of ``(0.05**2 - 0.03**2) / 12`` i.e. sigma = 0.0115".
    Whether the DJA resampling really adds that blur depends on its drizzle
    kernel, so this is offered as a knob and scanned, not assumed.

    Args:
        kernel: 2-D kernel on the ``from_scale`` grid.
        from_scale: Pixel scale of ``kernel``, arcsec/px.
        to_scale: Target pixel scale, arcsec/px.
        method: ``"spline"`` or ``"area"``.
        broaden_arcsec: Sigma of an extra Gaussian blur, arcsec (0 = none).
        out_size: Optional odd output size in pixels.

    Returns:
        2-D float32 kernel on the target grid, normalised to sum 1.

    Raises:
        ValueError: On an unknown ``method``.
    """
    arr = np.asarray(kernel, dtype=np.float64)
    if method == "spline":
        out = np.asarray(
            PSFModel.resample(arr, from_scale, to_scale, out_size=out_size), dtype=np.float64
        )
    elif method == "area":
        ratio = float(to_scale) / float(from_scale)
        h_in, w_in = arr.shape
        if out_size is None:
            h_out = int(np.ceil(h_in / ratio)) | 1
            w_out = int(np.ceil(w_in / ratio)) | 1
        else:
            h_out = w_out = int(out_size)
        out = _overlap_matrix(h_in, h_out, ratio) @ arr @ _overlap_matrix(w_in, w_out, ratio).T
    else:
        raise ValueError(f"unknown PSF resampling method {method!r} (spline|area)")
    if broaden_arcsec > 0:
        from scipy.ndimage import gaussian_filter

        out = gaussian_filter(out, broaden_arcsec / float(to_scale), mode="constant", cval=0.0)
    return (out / out.sum()).astype(np.float32)


def build_psf_model(
    bands: list[str],
    to_scale: float,
    psf_scale: float,
    method: str = "spline",
    broaden_arcsec: float = 0.0,
) -> PSFModel:
    """Load the empirical JADES DR3 PSFs and resample them to the cutout pixel scale."""
    native = native_psf_kernels(bands)
    kernels = np.stack(
        [resample_psf(native[b], psf_scale, to_scale, method, broaden_arcsec) for b in bands]
    ).astype(np.float32)
    return PSFModel(kernels=kernels, band_names=list(bands))


class BandSubsetEmulator(SPSEmulator):
    """Restrict an emulator's output to a subset of its bands (same inputs)."""

    inner: SPSEmulator
    _band_names: tuple[str, ...] = eqx.field(static=True)
    _band_idx: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, inner: SPSEmulator, band_names: list[str]) -> None:
        missing = [b for b in band_names if b not in inner.band_names]
        if missing:
            raise ValueError(f"bands not in emulator: {missing}")
        self.inner = inner
        self._band_names = tuple(band_names)
        self._band_idx = tuple(inner.band_names.index(b) for b in band_names)

    @property
    def param_names(self) -> list[str]:
        return list(self.inner.param_names)

    @property
    def band_names(self) -> list[str]:
        return list(self._band_names)

    def predict(self, params: jnp.ndarray) -> jnp.ndarray:
        return self.inner.predict(params)[:, jnp.array(self._band_idx)]


# ---------------------------------------------------------------------------
# Catalogue photometry
# ---------------------------------------------------------------------------


def catalogue_photometry(target: Target, bands: list[str], max_sep_arcsec: float = 0.3) -> dict:
    """Total fluxes (nJy) of the nearest JADES DR3 GS source, or an empty dict.

    The DR4 ``nircam_dr3_id`` values do not appear in the local DR3 subset's
    ``UNIQUE_ID`` column, so the match is positional on ``Ra_MATCH/Dec_MATCH``.
    """
    if not JADES_DR3_PHOT.exists():
        return {}
    from astropy.table import Table

    table = Table.read(JADES_DR3_PHOT)
    ra = np.asarray(table["Ra_MATCH"], dtype=float)
    dec = np.asarray(table["Dec_MATCH"], dtype=float)
    sep = np.hypot((ra - target.ra) * np.cos(np.radians(target.dec)), dec - target.dec) * 3600.0
    i = int(np.nanargmin(sep))
    if not sep[i] < max_sep_arcsec:
        print(f'  no JADES DR3 counterpart within {max_sep_arcsec}" (nearest {sep[i]:.2f}")')
        return {}
    row = table[i]
    out = {"dr3_unique_id": int(row["UNIQUE_ID"]), "sep_arcsec": float(sep[i]), "flux": {}}
    for band in bands:
        key = f"FLUX_TOTAL_{short_band(band)}"
        if key in table.colnames:
            out["flux"][band] = (float(row[key]), float(row[f"FLUXERR_TOTAL_{short_band(band)}"]))
    return out


def write_photometry_csv(path: Path, rows: list[dict]) -> None:
    """Write the band-by-band model vs catalogue photometry comparison."""
    fields = [
        "band",
        "pivot_um",
        "catalogue_nJy",
        "catalogue_err_nJy",
        "model_total_p16",
        "model_total_p50",
        "model_total_p84",
        "model_in_frame_p50",
        "data_frame_sum",
        "ratio_total_over_cat",
        "ratio_in_frame_over_cat",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _asinh(img: np.ndarray, scale: float) -> np.ndarray:
    return np.arcsinh(img / scale)


def plot_band_mosaic(plt, path: Path, bands, data, model, chi, title: str) -> Path:
    """Per-band data / model / chi mosaic, one row per band."""
    nb = len(bands)
    fig, axes = plt.subplots(nb, 3, figsize=(8.4, 2.5 * nb))
    axes = np.atleast_2d(axes)
    for b, band in enumerate(bands):
        scale = max(float(np.nanmax(data[b])) / 50.0, 1e-3)
        vmax = _asinh(np.nanmax(data[b]), scale)
        for c, img in enumerate((data[b], model[b])):
            ax = axes[b, c]
            ax.imshow(_asinh(img, scale), origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        im = axes[b, 2].imshow(chi[b], origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
        axes[b, 2].set_xticks([])
        axes[b, 2].set_yticks([])
        fig.colorbar(im, ax=axes[b, 2], fraction=0.046, pad=0.02)
        axes[b, 0].set_ylabel(
            f"{short_band(band)}\npeak {np.nanmax(data[b]):.0f} nJy/px", fontsize=8
        )
    for c, lab in enumerate(("data", "median model", r"$\chi_{\rm eff}$")):
        axes[0, c].set_title(lab, fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_component_images(plt, path: Path, bands, comp_median, centres_px, show_bands) -> Path:
    """Median per-component (unconvolved) images in the requested bands.

    ``comp_median[k][b]`` is component ``k``'s image in band ``b`` (a (K,
    N_bands, H, W) array on one grid, or nested lists when the bands have
    different shapes), and ``centres_px[k][b]`` is that component's centre in
    that band's pixel indices.
    """
    K = len(comp_median)
    cols = [bands.index(b) for b in show_bands if b in bands]
    fig, axes = plt.subplots(K, len(cols), figsize=(4.2 * len(cols), 3.9 * K), squeeze=False)
    for k in range(K):
        for j, b in enumerate(cols):
            img = np.asarray(comp_median[k][b])
            scale = max(float(np.nanmax(img)) / 50.0, 1e-4)
            ax = axes[k][j]
            ax.imshow(_asinh(img, scale), origin="lower", cmap="magma")
            for kk in range(K):
                ax.plot(
                    centres_px[kk][b][1],
                    centres_px[kk][b][0],
                    marker="+" if kk == k else "x",
                    color="cyan" if kk == k else "white",
                    ms=9,
                    mew=1.5,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_title(short_band(bands[b]), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"component {k}\n(sum {img.sum():.0f} nJy)", fontsize=9)
    fig.suptitle("Posterior-median component images (before PSF convolution)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_component_seds(plt, path: Path, bands, sed_samples, total_samples, cat, data_sum) -> Path:
    """Component SEDs with 16-84% bands, the total model, and catalogue totals."""
    wl = np.array([PIVOT_UM[short_band(b)] for b in bands])
    K = sed_samples.shape[1]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colours = ["tab:orange", "tab:blue", "tab:green", "tab:purple"]
    for k in range(K):
        lo, med, hi = np.percentile(sed_samples[:, k, :], [16, 50, 84], axis=0)
        ax.fill_between(wl, lo, hi, color=colours[k % len(colours)], alpha=0.25)
        ax.plot(wl, med, "o-", color=colours[k % len(colours)], ms=4, label=f"component {k}")
    lo, med, hi = np.percentile(total_samples, [16, 50, 84], axis=0)
    ax.fill_between(wl, lo, hi, color="k", alpha=0.18)
    ax.plot(wl, med, "s-", color="k", ms=5, label="total model (whole plane)")
    ax.plot(wl, data_sum, "^", color="tab:red", ms=6, label="data, summed over cutout")
    if cat.get("flux"):
        cwl, cf, ce = [], [], []
        for b in bands:
            if b in cat["flux"]:
                cwl.append(PIVOT_UM[short_band(b)])
                cf.append(cat["flux"][b][0])
                ce.append(cat["flux"][b][1])
        ax.errorbar(cwl, cf, yerr=ce, fmt="D", color="tab:gray", ms=6, label="JADES DR3 FLUX_TOTAL")
    ax.set_xlabel(r"pivot wavelength [$\mu$m]")
    ax.set_ylabel("flux [nJy]")
    ax.set_yscale("log")
    ax.set_title("Component SEDs vs catalogue photometry")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_posterior_1d(plt, path: Path, panels, title: str) -> Path:
    """One histogram per (label, samples) entry, laid out on a grid."""
    n = len(panels)
    ncol = min(6, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.1 * nrow), squeeze=False)
    for i, (label, values) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        values = np.asarray(values, dtype=float)
        ax.hist(values, bins=30, color="tab:blue", alpha=0.75)
        q16, q50, q84 = np.percentile(values, [16, 50, 84])
        ax.axvline(q50, color="k", lw=1)
        ax.axvspan(q16, q84, color="k", alpha=0.10)
        ax.set_title(f"{label}\n{q50:.3g} [{q16:.3g}, {q84:.3g}]", fontsize=7)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_nss_diagnostics(plt, path: Path, result) -> Path:
    """NSS log-likelihood / weight trace of the dead points."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True)
    if result.infos is not None:
        logl = np.asarray(result.infos.particles.loglikelihood)
        it = np.arange(logl.size)
        axes[0].plot(it, logl, lw=0.6)
        lo = np.percentile(logl, 5)
        axes[0].set_ylim(lo, logl.max() + 0.05 * (logl.max() - lo))
    axes[0].set_ylabel("log L of dead point")
    if result.log_weights is not None:
        lw = np.asarray(result.log_weights)
        axes[1].plot(np.arange(lw.size), np.exp(lw - lw.max()), lw=0.6, color="tab:orange")
    axes[1].set_ylabel("relative posterior weight")
    axes[1].set_xlabel("dead-point index")
    fig.suptitle(
        f"NSS: logZ = {result.logZ:.2f} +/- {result.logZ_err:.2f}, ESS = {result.ess:.0f}, "
        f"{result.n_steps} steps, {result.n_dead} dead points",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_nuts_diagnostics(plt, path: Path, result, param_names) -> Path:
    """NUTS R-hat / ESS / tree-depth summary."""
    diag = result.diagnostics or {}
    rhat = np.asarray(diag.get("rhat", []), dtype=float)
    ess = np.asarray(diag.get("ess", []), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    idx = np.arange(rhat.size)
    axes[0].bar(idx, rhat, color="tab:blue")
    axes[0].axhline(1.01, color="k", ls="--", lw=1)
    axes[0].set_ylabel(r"split $\hat{R}$")
    axes[1].bar(np.arange(ess.size), ess, color="tab:green")
    axes[1].set_ylabel("ESS")
    for ax in axes[:2]:
        ax.set_xticks(np.arange(min(len(param_names), rhat.size)))
        ax.set_xticklabels(param_names[: rhat.size], rotation=90, fontsize=5)
    samples = np.asarray(result.samples)
    axes[2].plot(samples[:, : min(6, samples.shape[1])], lw=0.4)
    axes[2].set_xlabel("sample")
    axes[2].set_ylabel("first theta components")
    fig.suptitle(
        "NUTS diagnostics: divergences = "
        f"{diag.get('n_divergent', 'n/a')}, mean tree depth = "
        f"{diag.get('mean_tree_depth', float('nan')):.1f}, "
        f"acceptance = {diag.get('acceptance_rate', float('nan')):.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Neighbour masking
# ---------------------------------------------------------------------------


def detection_stack(flux: np.ndarray, variance: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inverse-variance-weighted multi-band detection image, in units of sigma.

    ``S/N = Σ_b f_b / v_b / sqrt(Σ_b 1 / v_b)`` is the matched-filter
    significance of a flat-spectrum source, so it can be thresholded directly
    at ``n`` sigma without a separate background estimate.
    """
    weight = np.where(mask > 0, 1.0 / np.maximum(variance, 1e-12), 0.0)
    num = np.sum(flux * weight, axis=0)
    den = np.sqrt(np.maximum(np.sum(weight, axis=0), 1e-12))
    return num / den


def _clipped_stats(values: np.ndarray, nsig: float = 3.0, iters: int = 5) -> tuple[float, float]:
    """Sigma-clipped (median, standard deviation) of a finite array."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    for _ in range(iters):
        med, sd = float(np.median(v)), float(np.std(v))
        if sd <= 0:
            break
        keep = np.abs(v - med) < nsig * sd
        if keep.all() or keep.sum() < 16:
            break
        v = v[keep]
    return float(np.median(v)), float(np.std(v))


def _disc(radius: int) -> np.ndarray:
    """Boolean disc structuring element of the given radius in pixels."""
    r = int(radius)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (yy**2 + xx**2) <= r * r


def neighbour_mask(
    flux: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    nsigma: float = 1.5,
    npixels: int = 10,
    dilate: int = 3,
    *,
    pixel_scale: float = 0.05,
    centre: tuple[float, float] | None = None,
    smooth_px: float = 1.0,
    clump_nsigma: float = 0.0,
    clump_scale_arcsec: float = 0.5,
    clump_dilate: int = 2,
    plume_radius_arcsec: float | None = None,
    aperture_radius_arcsec: float | None = None,
    protect_radius_arcsec: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Zero the mask on everything that is not the galaxy being fitted.

    ``skimage`` is absent from the shared venv (so ``photutils.deblend_sources``
    cannot run) and cannot be installed, so the segmentation is built from
    :mod:`scipy.ndimage` alone:

    1. **Detection image.** :func:`detection_stack` gives the inverse-variance
       weighted multi-band matched-filter significance, smoothed with a
       Gaussian of ``smooth_px`` pixels (roughly the PSF sigma, which is the
       matched filter for point-like neighbours).
    2. **Threshold.** The background level and noise of the smoothed stack are
       its sigma-clipped median and standard deviation, so ``nsigma`` means
       "above this cutout's own background" even though the stack is already in
       sigma units (the target's halo lifts the whole 6" frame above zero).
    3. **Segmentation.** ``scipy.ndimage.label`` with 8-connectivity; segments
       smaller than ``npixels`` are discarded.  The target is the segment
       containing ``centre`` (the nearest labelled pixel within 5 px if the
       centre itself is unlabelled, else the segment with the largest summed
       significance).  Every *other* segment is dilated by ``dilate`` pixels and
       masked.  This is deblending only by connectivity — two touching sources
       stay one segment, which is what steps 4-6 exist for.
    4. **Clump / companion peel** (``clump_nsigma > 0``).  A compact source
       sitting *on* the galaxy is never a separate segment (in the high-pass
       image it is simply connected to the galaxy's own cusp), so clumps are
       found as *local maxima* instead: the smoothed stack minus its median
       filter over ``clump_scale_arcsec`` is compared with its own maximum
       filter over the same scale, and every local maximum *inside the target
       segment* that is above ``clump_nsigma`` times the clipped noise and
       further than ``protect_radius_arcsec`` from the centre is masked out to
       ``clump_dilate + clump_scale / 2`` pixels.  Restricting the search to the
       target segment is deliberate: a compact source on blank sky is already a
       segment of its own and was masked in step 3.  A smooth
       monotonically declining profile has exactly one local maximum — the
       nucleus, which the radius cut protects — so this flags real substructure
       and not the galaxy itself.
    5. **Plume mask** (``plume_radius_arcsec``).  Masks the part of the *target's
       own* segment beyond that radius: tidal streams and merger plumes are
       physically connected to the galaxy and no segmentation will ever split
       them off, yet no Sersic profile describes them.  Blank sky beyond the
       radius stays unmasked, so it still constrains the sky pedestal.
    6. **Hard aperture** (``aperture_radius_arcsec``).  Masks *everything*
       beyond that radius.  Blunter than 5, and it leaves the sky nuisance
       parameters much less constrained.

    Pixels within ``protect_radius_arcsec`` of the centre are never masked.

    Args:
        flux: (N_bands, H, W) flux cube, nJy.
        variance: Matching variance cube.
        mask: Matching validity mask (1 = use).
        nsigma: Detection threshold above the background, in clipped sigma.
        npixels: Minimum connected pixels for a segment.
        dilate: Dilation of neighbour segments, pixels.
        pixel_scale: Arcsec per pixel.
        centre: (row, col) of the target; defaults to the frame centre.
        smooth_px: Gaussian smoothing of the detection stack, pixels.
        clump_nsigma: High-pass clump threshold (0 disables step 4).
        clump_scale_arcsec: Median-filter scale of the high-pass filter.
        clump_dilate: Dilation of clump detections, pixels.
        plume_radius_arcsec: Radius beyond which the target segment is masked.
        aperture_radius_arcsec: Radius beyond which everything is masked.
        protect_radius_arcsec: Radius that is never masked.

    Returns:
        ``(new_mask (N,H,W), segmentation (H,W), info)`` where ``info`` records
        the masked fraction of each step.
    """
    from scipy.ndimage import (
        binary_dilation,
        gaussian_filter,
        label,
        maximum_filter,
        median_filter,
    )

    snr = detection_stack(flux, variance, mask)
    h, w = snr.shape
    cy, cx = (
        ((h - 1) / 2.0, (w - 1) / 2.0) if centre is None else (float(centre[0]), float(centre[1]))
    )
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(yy - cy, xx - cx) * float(pixel_scale)

    smooth = gaussian_filter(snr, float(smooth_px)) if smooth_px > 0 else snr
    bkg, rms = _clipped_stats(smooth)
    detected = smooth > bkg + float(nsigma) * rms
    labels, n_lab = label(detected, structure=np.ones((3, 3), dtype=bool))
    if n_lab:  # drop segments below the minimum area
        counts = np.bincount(labels.ravel())
        small = np.flatnonzero(counts < int(npixels))
        if small.size:
            labels[np.isin(labels, small)] = 0
    segmentation = labels.astype(np.int32)

    info = {"n_segments": int(len(np.unique(segmentation)) - 1), "background": bkg, "rms": rms}
    target = int(segmentation[int(round(cy)), int(round(cx))])
    if target == 0:
        near = segmentation[max(0, int(cy) - 5) : int(cy) + 6, max(0, int(cx) - 5) : int(cx) + 6]
        present = [int(v) for v in np.unique(near) if v != 0]
        if present:
            target = present[0]
        else:
            allz = [int(v) for v in np.unique(segmentation) if v != 0]
            target = int(max(allz, key=lambda v: snr[segmentation == v].sum())) if allz else 0
    info["target_label"] = target
    in_target = segmentation == target

    bad = (segmentation != 0) & ~in_target
    if dilate > 0 and bad.any():
        bad = binary_dilation(bad, structure=np.ones((2 * int(dilate) + 1,) * 2, dtype=bool))
    bad &= ~in_target
    info["frac_neighbours"] = float(bad.mean())

    if clump_nsigma > 0:
        size = max(3, int(round(float(clump_scale_arcsec) / float(pixel_scale))) | 1)
        high_pass = smooth - median_filter(smooth, size=size, mode="nearest")
        _, hp_rms = _clipped_stats(high_pass)
        local_max = high_pass >= maximum_filter(high_pass, size=size, mode="nearest")
        peaks = (
            local_max
            & in_target
            & (high_pass > float(clump_nsigma) * hp_rms)
            & (radius > float(protect_radius_arcsec))
        )
        clumps = np.zeros_like(bad)
        if peaks.any():
            clumps = binary_dilation(peaks, structure=_disc(max(1, int(clump_dilate) + size // 2)))
        before = bad.mean()
        bad |= clumps
        info["frac_clumps"] = float(bad.mean() - before)
        info["n_clumps"] = int(peaks.sum())

    if plume_radius_arcsec is not None:
        before = bad.mean()
        bad |= in_target & (radius > float(plume_radius_arcsec))
        info["frac_plume"] = float(bad.mean() - before)
    if aperture_radius_arcsec is not None:
        before = bad.mean()
        bad |= radius > float(aperture_radius_arcsec)
        info["frac_aperture"] = float(bad.mean() - before)

    bad &= radius > float(protect_radius_arcsec)
    new_mask = np.where(bad[None, :, :], 0.0, mask).astype(np.float32)
    info["frac_masked"] = float(bad.mean())
    info["frac_valid"] = float((new_mask[0] > 0).mean())
    return new_mask, segmentation, info


def save_mask_products(path_png, path_fits, snr, segm, bad, target_id: str) -> None:
    """Write the detection stack / segmentation / final mask as FITS and a PNG."""
    from astropy.io import fits

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.asarray(snr, dtype=np.float32), name="SNR_STACK"),
            fits.ImageHDU(np.asarray(segm, dtype=np.int32), name="SEGMENTATION"),
            fits.ImageHDU(np.asarray(bad, dtype=np.uint8), name="MASKED"),
        ]
    )
    hdul.writeto(path_fits, overwrite=True)

    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(np.arcsinh(snr / 3.0), origin="lower", cmap="viridis")
    axes[0].set_title("S/N detection stack")
    axes[1].imshow(segm, origin="lower", cmap="tab20", interpolation="nearest")
    axes[1].set_title("segmentation (deblended)")
    axes[2].imshow(bad, origin="lower", cmap="gray_r", interpolation="nearest")
    axes[2].set_title(f"masked neighbours ({100 * bad.mean():.1f}% of pixels)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{target_id}: neighbour mask", fontsize=11)
    fig.tight_layout()
    fig.savefig(path_png, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-resolution helpers
# ---------------------------------------------------------------------------


def mirror_mask_to_band(bad: np.ndarray, ref_band, band) -> np.ndarray:
    """Nearest-neighbour transfer of a boolean mask between two band grids.

    Both grids carry ``affine`` (``(row - ref_row, col - ref_col) -> (dy, dx)``
    arcsec, North/East) about the *same* reference sky position, so the mask is
    resampled in sky coordinates: every pixel of ``band`` is mapped to a
    fractional pixel of ``ref_band`` and takes that pixel's value.  Pixels that
    fall outside ``ref_band`` are left unmasked.

    Args:
        bad: (H_ref, W_ref) boolean "masked" map on ``ref_band``'s grid.
        ref_band: BandImage the mask was built on.
        band: BandImage to transfer it to.

    Returns:
        (H_b, W_b) boolean array.
    """
    h, w = band.shape
    rows, cols = np.mgrid[0:h, 0:w]
    off = np.asarray(band.affine, dtype=np.float64) @ np.stack(
        [rows.ravel() - band.ref_pixel[0], cols.ravel() - band.ref_pixel[1]]
    )
    inv = np.linalg.inv(np.asarray(ref_band.affine, dtype=np.float64))
    ref_rc = inv @ off
    rr = np.rint(ref_rc[0] + ref_band.ref_pixel[0]).astype(int)
    cc = np.rint(ref_rc[1] + ref_band.ref_pixel[1]).astype(int)
    hr, wr = bad.shape
    inside = (rr >= 0) & (rr < hr) & (cc >= 0) & (cc < wr)
    out = np.zeros(rr.size, dtype=bool)
    out[inside] = bad[rr[inside], cc[inside]]
    return out.reshape(h, w)


def apply_mask_to_mro(mro, bad_maps: dict[str, np.ndarray], psfs: dict | None = None):
    """Copy of a MultiResolutionObservation with extra pixels masked (and PSFs replaced)."""
    import dataclasses as _dc

    from arachne.data.multires import MultiResolutionObservation

    bands = []
    for band in mro.bands:
        updates = {}
        bad = bad_maps.get(band.band_name)
        if bad is not None:
            updates["mask"] = np.where(bad, 0.0, np.asarray(band.mask, dtype=np.float32)).astype(
                np.float32
            )
        if psfs is not None and band.band_name in psfs:
            updates["psf"] = np.asarray(psfs[band.band_name], dtype=np.float32)
        bands.append(_dc.replace(band, **updates) if updates else band)
    return MultiResolutionObservation(bands=bands, ref_ra=mro.ref_ra, ref_dec=mro.ref_dec)


# ---------------------------------------------------------------------------
# Laplace / whitened-NUTS helpers
#
# FALLBACK COPY of the pattern in ``examples/demo_resolved_sed_fitting.py``
# (``newton_hessian`` / ``laplace_covariance`` / the whitened ``run_nuts``).
# ``arachne.inference.laplace`` landed during Phase 2b and is strictly better
# (it uses ``|eigenvalue|`` instead of clipping negative-curvature directions
# and adapts a dense metric inside the whitened space), so ``fit_jades_dja.py``
# calls ``laplace.run_whitened_nuts`` whenever ``import_laplace()`` finds it
# and only falls back to the functions below on an older checkout.  ``hessian_fn``
# is still used directly by the damped-Newton MAP polish.
# ---------------------------------------------------------------------------

MAX_LAPLACE_VAR = 9.0  # cap on a Laplace metric variance (raw units; ~the prior width)


def hessian_fn(fm):
    """Jitted ``theta -> symmetrised Hessian of -log_posterior`` (float64 numpy).

    Build it once per forward model: ``jax.hessian`` re-traces on every call and
    tracing this one costs tens of seconds.
    """
    import jax

    jitted = jax.jit(jax.hessian(lambda t: -fm.log_posterior(t)))

    def at(theta):
        h = np.asarray(jitted(jnp.asarray(theta)), dtype=np.float64)
        return 0.5 * (h + h.T)

    return at


def laplace_covariance(hess: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Eigen-clipped Laplace covariance and its Cholesky factor.

    Directions with numerically zero or (at an imperfect mode) negative
    curvature are given the widest variance the model allows,
    ``MAX_LAPLACE_VAR``, so the sampler explores them freely instead of being
    told they are infinitely wide.

    Args:
        hess: (d, d) Hessian of ``-log_posterior`` at the MAP.

    Returns:
        ``(covariance, cholesky, info)``.
    """
    evals, evecs = np.linalg.eigh(hess)
    floor = 1.0 / MAX_LAPLACE_VAR
    n_bad = int((evals < floor).sum())
    cov = (evecs / np.clip(evals, floor, None)) @ evecs.T
    cov = 0.5 * (cov + cov.T)
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0])
    sd = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    info = {
        "curvature_min": float(evals.min()),
        "curvature_max": float(evals.max()),
        "n_floored": n_bad,
        "marginal_sd_min": float(sd.min()),
        "marginal_sd_max": float(sd.max()),
    }
    return cov, chol, info


class WhitenedModel:
    """``log_posterior`` in whitened coordinates ``theta = theta0 + L z``.

    These posteriors have curvature eigenvalues spanning ~1e-1 to ~4e7 with the
    stiff directions strongly correlated, so a diagonal metric collapses the
    NUTS step size to ~1e-7.  Sampling ``z`` with ``L`` the Cholesky factor of
    the Laplace covariance makes the target approximately ``N(0, I)``.
    ``log p(z) = log p(theta0 + L z) + log|L|``; the constant is dropped.  Only
    ``log_posterior`` and ``spatial_model`` are used by ``NUTSSampler``.
    """

    def __init__(self, fm, theta0, chol: np.ndarray) -> None:
        self.fm = fm
        self.theta0 = jnp.asarray(theta0)
        self.chol = jnp.asarray(chol, dtype=self.theta0.dtype)
        self.spatial_model = fm.spatial_model
        self.n_params = int(self.theta0.size)

    def to_theta(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.theta0 + z @ self.chol.T

    def log_posterior(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.fm.log_posterior(self.theta0 + self.chol @ z)


def import_laplace():
    """Return ``arachne.inference.laplace`` if the library module has landed."""
    try:
        import arachne.inference.laplace as module  # noqa: PLC0415
    except ImportError:
        return None
    return module


# ---------------------------------------------------------------------------
# Size guess (sets the log-size prior and the blind starting size)
# ---------------------------------------------------------------------------


def half_light_radius_arcsec(
    flux: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    pixel_scale: float,
    centre: tuple[float, float] | None = None,
    max_radius_arcsec: float | None = None,
) -> float:
    """Circularised half-light radius of the masked multi-band S/N stack.

    A curve of growth is far more robust than the second moment
    :func:`~arachne.inference.initialisation.image_moments` returns: on a 6"
    frame the second moment of a bright galaxy with a tidal plume is dominated
    by the plume and comes out several times the true size, which then sets a
    log-size prior centred on the wrong scale.  The stack's sigma-clipped
    median is removed first, the result clipped at zero, and the radius that
    encloses half of the remaining flux inside ``max_radius_arcsec`` returned.

    Args:
        flux: (N_bands, H, W) flux cube.
        variance: Matching variance cube.
        mask: Matching validity mask (1 = use).
        pixel_scale: Arcsec per pixel.
        centre: (row, col) of the target; defaults to the frame centre.
        max_radius_arcsec: Aperture for the curve of growth; defaults to a
            quarter of the frame width.

    Returns:
        Half-light radius in arcsec (never below one pixel).
    """
    stack = detection_stack(flux, variance, mask)
    h, w = stack.shape
    cy, cx = (
        ((h - 1) / 2.0, (w - 1) / 2.0) if centre is None else (float(centre[0]), float(centre[1]))
    )
    if max_radius_arcsec is None:
        max_radius_arcsec = 0.25 * max(h, w) * float(pixel_scale)
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(yy - cy, xx - cx) * float(pixel_scale)
    bkg, _ = _clipped_stats(stack)
    valid = (mask[0] > 0) & (radius <= max_radius_arcsec)
    values = np.clip(stack - bkg, 0.0, None) * valid
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(values.ravel()[order])
    if cumulative[-1] <= 0:
        return float(pixel_scale)
    r_half = float(radius.ravel()[order][int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))])
    return max(r_half, float(pixel_scale))


def log_size_prior_from_guess(
    r_guess_arcsec: float, sd: float = 0.5, max_re_arcsec: float | None = None, n_sigma: float = 3.0
) -> tuple[float, float]:
    """Gaussian prior on ``log sigma`` centred on the measured size.

    ``--max-re-arcsec`` is implemented as a *narrower prior*, not a hard bound:
    the sd is reduced until ``mu + n_sigma * sd <= log(max_re)``, so a component
    larger than the cap is strongly disfavoured but the prior stays a proper,
    normalised Gaussian that :meth:`AdditiveComponentModel.sample_prior` draws
    from — which a hard wall or an extra penalty term would not, and nested
    sampling needs prior-distributed live points for ``logZ`` to mean anything.

    Args:
        r_guess_arcsec: Measured half-light radius.
        sd: Prior sd in natural log, before any cap.
        max_re_arcsec: Optional soft cap on the effective radius.
        n_sigma: How many sd from the mean the cap should sit at.

    Returns:
        ``(mu, sd)`` for ``AdditiveComponentModel(log_size_prior=...)``.
    """
    mu = float(np.log(max(float(r_guess_arcsec), 1e-3)))
    sd = float(sd)
    if max_re_arcsec is not None:
        cap = float(np.log(float(max_re_arcsec)))
        if cap <= mu:
            raise ValueError(
                f"--max-re-arcsec {max_re_arcsec} is below the measured size "
                f'{np.exp(mu):.3f}"; the prior would exclude the galaxy'
            )
        sd = min(sd, (cap - mu) / float(n_sigma))
    return mu, sd


def cached_mosaic_paths(
    cache_root: Path, bands: list[str], ra: float, dec: float
) -> tuple[dict, dict]:
    """Find already-downloaded DJA association sub-mosaics covering a position.

    The GPU nodes on this system have **no outbound network**, so
    :func:`arachne.data.dja.fetch_dja_native_cutout` — which queries the DJA
    API before touching its cache — can only run on the login node.  This
    function reproduces the "pick the deepest association that covers the
    target" choice from the files already in ``<cache_root>/assoc_mosaic``:
    candidates are matched by filter from the file name, their WCS footprints
    tested against the target, and the one with the largest ``EXPTIME`` kept.

    Args:
        cache_root: Directory whose ``assoc_mosaic`` subdirectory holds the files.
        bands: Canonical band names, e.g. ``"JWST/NIRCam.F200W"``.
        ra: Target right ascension, degrees.
        dec: Target declination, degrees.

    Returns:
        ``(flux_paths, weight_paths)``, each band -> path, for the bands found.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS

    directory = Path(cache_root) / "assoc_mosaic"
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    flux_paths: dict[str, Path] = {}
    weight_paths: dict[str, Path] = {}
    for band in bands:
        filt = short_band(band).lower()
        best, best_exptime = None, -1.0
        for path in sorted(directory.glob(f"*-{filt}-clear_drc_sci.fits.gz")):
            with fits.open(path) as hdul:
                header = hdul[0].header
                wcs = WCS(header)
                try:
                    covered = bool(wcs.footprint_contains(target))
                except Exception:  # noqa: BLE001 - a malformed WCS is just "no match"
                    covered = False
                exptime = float(header.get("EXPTIME", 0.0) or 0.0)
            if covered and exptime > best_exptime:
                best, best_exptime = path, exptime
        if best is None:
            continue
        flux_paths[band] = best
        weight = best.with_name(best.name.replace("_sci.fits.gz", "_wht.fits.gz"))
        if weight.exists():
            weight_paths[band] = weight
    return flux_paths, weight_paths
