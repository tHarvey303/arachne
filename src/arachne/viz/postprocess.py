"""Figure of a bundle's simulator post-processing: spectrum, SFH and derived maps.

Needs a bundle that :func:`arachne.postprocess.bundle.postprocess_bundle` has run on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure, SubFigure

from ..results.reader import ResultsBundle
from ..results.regions import region_sed
from .style import (
    INTRINSIC_COLOR,
    OBS_COLOR,
    as_bundle,
    new_figure,
    root_figure,
    set_log_flux_axis,
    set_micron_axis,
    style_context,
)

__all__ = ["plot_postprocess"]

_SPEC_COLOR = "0.3"


def _kpc_per_pixel(bundle: ResultsBundle) -> float:
    from astropy.cosmology import Planck18

    return float(Planck18.kpc_proper_per_arcmin(bundle.redshift).value / 60.0 * bundle.pixel_scale)


def plot_postprocess(
    bundle: ResultsBundle | str | Path,
    result: Any = None,
    region: str | None = None,
    *,
    fig: Figure | SubFigure | None = None,
) -> Figure:
    """Spectrum, SFH and derived maps from the simulator post-processing.

    Left: the observed-frame spectrum of the region (median and 16-84 % over
    draws; the intrinsic light of its pixels, no PSF) with the simulator
    photometry and the observed aperture fluxes, and below it the SFH in
    physical units.  Right: surviving stellar mass and SFR surface densities,
    mass-weighted age and specific SFR.

    Args:
        bundle: Bundle or path.
        result: A ``PostprocessResult``; default: read from the bundle file.
        region: Region for the spectrum and SFH (default: the processed region).
        fig: Figure or SubFigure to draw into.

    Returns:
        The figure.

    Raises:
        ValueError: If the bundle has not been post-processed.
    """
    bundle = as_bundle(bundle)
    if result is None:
        from ..postprocess.bundle import load_postprocess

        result = load_postprocess(bundle.path) if bundle.path is not None else None
    if result is None:
        raise ValueError("the bundle has no postprocess group; run arachne-bundle postprocess")
    region = region or result.region
    stat = result.stat
    lam = result.wavelength_um
    piv = np.asarray(bundle.pivot_um, dtype=float)
    with style_context():
        fig = new_figure(fig, (12.0, 5.6))
        gs = fig.add_gridspec(2, 4, width_ratios=[2.1, 2.1, 1, 1], height_ratios=[1.35, 1])

        ax = fig.add_subplot(gs[0, :2])
        sel = (lam > 0.55 * piv.min()) & (lam < 1.6 * piv.max())
        to_ujy = 1e-3
        sed = result.sed[region][:, sel] * to_ujy
        if result.n_draws > 1:
            ax.fill_between(
                lam[sel], stat(sed, "p16"), stat(sed, "p84"), color=_SPEC_COLOR, alpha=0.3, lw=0
            )
        ax.plot(lam[sel], stat(sed, "p50"), color=_SPEC_COLOR, lw=0.7, label="simulator spectrum")
        ax.plot(
            piv,
            stat(result.photometry[region], "p50") * to_ujy,
            "o",
            ms=7,
            mfc="none",
            mec=INTRINSIC_COLOR,
            label="simulator photometry (intrinsic)",
        )
        obs = region_sed(bundle, bundle.regions[region]) if region in bundle.regions else None
        if obs is not None and obs.obs_flux is not None:
            ax.errorbar(
                piv,
                obs.obs_flux * to_ujy,
                yerr=obs.obs_err * to_ujy,
                fmt="s",
                ms=3.5,
                color=OBS_COLOR,
                lw=0.8,
                label="observed, summed in the region",
            )
        ax.set_yscale("log")
        med = stat(sed, "p50")
        ax.set_ylim(0.5 * np.nanmin(med[med > 0]), 2.5 * np.nanpercentile(med, 99))
        set_micron_axis(ax, [lam[sel].min(), lam[sel].max()])
        set_log_flux_axis(ax)
        ax.set_ylabel(r"$f_\nu$ [$\mu$Jy]")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        tot = result.totals[region]
        kind = "draws" if result.attrs.get("from_draws", True) else "MAP"

        def pm(name: str, fmt: str = "{:.2f}") -> str:
            lo, mid, hi = (float(np.percentile(tot[name], q)) for q in (16, 50, 84))
            if result.n_draws < 2:
                return fmt.format(mid)
            return f"{fmt.format(mid)}$^{{+{fmt.format(hi - mid)}}}_{{-{fmt.format(mid - lo)}}}$"

        ax.set_title(
            f"{bundle.title or 'bundle'} — region '{region}', {result.n_draws} {kind}:  "
            rf"$\log M_*$ surviving {pm('log_mass_surviving')} (formed {pm('log_mass_formed')}),  "
            rf"SFR$_{{100}}$ {pm('sfr_100', '{:.1f}')} M$_\odot$/yr",
            fontsize=9,
            loc="left",
        )

        ax2 = fig.add_subplot(gs[1, :2])
        if region in result.sfh and result.sfh_agebins_myr is not None:
            edges = np.concatenate((result.sfh_agebins_myr[:, 0], result.sfh_agebins_myr[-1:, 1]))
            edges[0] = max(edges[0], 1.0)
            sfh = result.sfh[region]

            def steps(v: np.ndarray) -> np.ndarray:
                return np.concatenate((v, v[-1:]))

            if result.n_draws > 1:
                ax2.fill_between(
                    edges,
                    steps(stat(sfh, "p16")),
                    steps(stat(sfh, "p84")),
                    step="post",
                    color=_SPEC_COLOR,
                    alpha=0.3,
                    lw=0,
                )
            ax2.step(edges, steps(stat(sfh, "p50")), where="post", color=_SPEC_COLOR, lw=1.2)
            ax2.set(xscale="log", yscale="log", xlim=(edges[0], edges[-1]))
            set_log_flux_axis(ax2)
        ax2.set_xlabel("lookback time [Myr]")
        ax2.set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")

        area = 2.0 * np.log10(_kpc_per_pixel(bundle))
        m_surv = stat(result.maps["log_mass_surviving"], "p50")
        sfr = stat(result.maps["sfr_100"], "p50")
        with np.errstate(divide="ignore", invalid="ignore"):
            panels = [
                (m_surv - area, r"$\log\,\Sigma_*$ surviving [M$_\odot$ kpc$^{-2}$]"),
                (
                    np.log10(sfr) - area,
                    r"$\log\,\Sigma_{\rm SFR,100}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]",
                ),
                (stat(result.maps["age_mass_weighted"], "p50") / 1e3, "mass-weighted age [Gyr]"),
                (np.log10(sfr) - m_surv, r"$\log$ sSFR$_{100}$ [yr$^{-1}$]"),
            ]
        for k, (img, label) in enumerate(panels):
            a = fig.add_subplot(gs[k // 2, 2 + k % 2])
            good = img[np.isfinite(img)]
            lo, hi = np.percentile(good, [2, 98]) if good.size else (0.0, 1.0)
            cmap = "magma_r" if k < 2 else "viridis"
            im = a.imshow(img, origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            cb = root_figure(fig).colorbar(im, ax=a, location="top", shrink=0.92, pad=0.02)
            cb.set_label(label, fontsize=8)
            cb.ax.tick_params(labelsize=7)
    return root_figure(fig)
