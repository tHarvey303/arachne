#!/usr/bin/env python3
"""Front-page figure for the arachne README: the three ways the code fits galaxies.

(a) catalogue fitting: integrated photometry of COSMOS2020 galaxies, nested sampling;
(b) light components: a bulge + disk decomposition of a real JADES galaxy;
(c) correlated fields: per-pixel parameter maps of a real JADES spiral.

Inputs are not in the repository.  They are the catalogue (COSMOS2020 Farmer, from IRSA)
and the outputs of ``scripts/fit_catalogue.py --sampler nss``, ``examples/fit_jades_dja.py``
(``figure_products.npz``) and ``examples/fit_correlated_field_dja.py`` (``results.h5``).
Set ``ARACHNE_FRONTPAGE_CATALOGUE``, ``ARACHNE_FRONTPAGE_CATALOGUE_FIT``,
``ARACHNE_FRONTPAGE_COMPONENTS`` and ``ARACHNE_FRONTPAGE_FIELDS`` to their paths.

Usage:  python make_frontpage.py [OUTPUT.png]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
from astropy.cosmology import Planck18
from astropy.table import Table
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from arachne.results import load_bundle

_DEFAULTS = {
    "catalogue": "catalogue/cosmos2020_farmer_cone.fits",
    "catalogue_fit": "catalogue/fits_nss.h5",
    "components": "components/figure_products.npz",
    "fields": "fields/results.h5",
}
INPUTS = {
    k: Path(os.environ.get(f"ARACHNE_FRONTPAGE_{k.upper()}", v)) for k, v in _DEFAULTS.items()
}

# ---- house style (Tom's matplotlib style: Charter, boxed frame, printed-ink colours) ----
INK, INK2, MUTED = "#111111", "#333333", "#8a8a8a"
INDIGO, CELADON, OXBLOOD, SKY, MUSTARD = "#2f3b73", "#7fa08a", "#7a1f2b", "#7fb3d5", "#c9a84c"
ACCENT = "#e34234"
EMBER = [
    "#fbe6b0",
    "#f2b26a",
    "#e98052",
    "#d75a5a",
    "#b64a6f",
    "#8c417b",
    "#62367a",
    "#3c2a63",
    "#1c1b44",
]
DIV = [
    "#a4252a",
    "#c4594f",
    "#dc9487",
    "#edc9c2",
    "#f1f0ee",
    "#c2cfea",
    "#8fa6dc",
    "#5c7dcb",
    "#2d55b0",
]
ember = LinearSegmentedColormap.from_list("ember", EMBER)
ember_r = LinearSegmentedColormap.from_list("ember_r", EMBER[::-1])
ember_div = LinearSegmentedColormap.from_list("ember_div", DIV)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Charter", "STIX Two Text", "DejaVu Serif"],
        "font.size": 8,
        "mathtext.fontset": "stix",
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)

# SFH bins of the emulator's library at z ~ 2 (Myr); logsfr_ratio_i = log10(SFR_i / SFR_{i+1}).
SFH_EDGES_MYR = np.array([0.0, 3.0, 10.0, 64.0, 411.0, 2600.0, 3100.0])


PIVOT_UM = {"F090W": 0.901, "F115W": 1.154, "F150W": 1.501, "F200W": 1.990, "F277W": 2.786,
            "F335M": 3.365, "F356W": 3.563, "F410M": 4.092, "F444W": 4.421}  # fmt: skip


def rgb_image(r, g, b, hi, soft=0.05):
    """Asinh RGB composite on one fixed scale ``hi`` (so panels can be compared)."""
    chans = np.stack([np.asarray(c, float) for c in (r, g, b)], axis=-1)
    out = np.arcsinh(np.clip(chans, 0, None) / (soft * hi)) / np.arcsinh(1.0 / soft)
    return np.clip(np.nan_to_num(out), 0, 1)


def image_axes(ax):
    """An image panel: no ticks, thin frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def caption(ax, text, color="white", size=7):
    """Short label inside the top-left corner of an image."""
    ax.text(
        0.05, 0.94, text, transform=ax.transAxes, ha="left", va="top", color=color, fontsize=size
    )


def scale_bar(ax, npix, pixel_scale, arcsec, color="white"):
    """Angular scale bar in the lower-left corner of an image of ``npix`` pixels."""
    length = arcsec / pixel_scale
    x0, y0 = 0.07 * npix, 0.08 * npix
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=1.2, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + 0.035 * npix, f'{arcsec:g}"', color=color, fontsize=6.5,
            ha="center", va="bottom")  # fmt: skip


def colorbar_below(fig, ax, im, label, ticks=None):
    """Thin horizontal colour bar under an image, with its label beneath."""
    box = ax.get_position()
    cax = fig.add_axes([box.x0 + 0.08 * box.width, box.y0 - 0.022, 0.84 * box.width, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1.5)
    cax.minorticks_off()
    cax.set_xlabel(label, fontsize=7, labelpad=1.5)
    return cb


# ---------------------------------------------------------------------------------------
# (c) correlated-field maps
# ---------------------------------------------------------------------------------------


def mass_weighted_age(ratios):
    """Mass-weighted age (Gyr) from the five log SFR ratios, shape (5, ...)."""
    log_sfr = np.zeros((6,) + ratios.shape[1:])
    for i in range(4, -1, -1):  # oldest bin is the reference
        log_sfr[i] = log_sfr[i + 1] + ratios[i]
    dt = np.diff(SFH_EDGES_MYR).reshape((6,) + (1,) * (ratios.ndim - 1))
    mid = (0.5 * (SFH_EDGES_MYR[1:] + SFH_EDGES_MYR[:-1])).reshape(dt.shape)
    mass = 10.0**log_sfr * dt
    return (mass * mid).sum(0) / mass.sum(0) / 1e3


def panel_fields(fig, spec):
    """Data, model, deconvolved model and three posterior-median parameter maps."""
    b = load_bundle(INPUTS["fields"])
    names = list(b.short_names)
    i_r, i_g, i_b = names.index("F444W"), names.index("F277W"), names.index("F150W")
    seg = b.regions["segment"].mask
    n = b.image_shape[0]
    kpc_per_px = b.pixel_scale * Planck18.kpc_proper_per_arcmin(b.redshift).value / 60.0

    hi = np.nanpercentile(np.stack([b.band_image(i, "data") for i in (i_r, i_g, i_b)]), 99.7)

    def rgb(kind):
        scale = hi * (2.5 if kind == "intrinsic" else 1.0)  # the deconvolved peak is sharper
        return rgb_image(b.band_image(i_r, kind), b.band_image(i_g, kind),
                         b.band_image(i_b, kind), scale)  # fmt: skip

    sigma = np.where(seg, b.param_map("log_mass", "p50") - 2 * np.log10(kpc_per_px), np.nan)
    av = np.where(seg, b.param_map("Av", "p50"), np.nan)
    ratios = np.stack([b.param_draws(f"logsfr_ratio_{i}") for i in range(5)])  # (5, S, H, W)
    age = np.where(seg, np.median(mass_weighted_age(ratios), axis=0), np.nan)

    gs = GridSpecFromSubplotSpec(1, 6, subplot_spec=spec, wspace=0.06)
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    for ax, kind, text in zip(
        axes[:3], ("data", "model", "intrinsic"), ("JWST/NIRCam", "model", "model, PSF removed")
    ):
        ax.imshow(rgb(kind), origin="lower", interpolation="nearest")
        caption(ax, text)
        image_axes(ax)
    scale_bar(axes[0], n, b.pixel_scale, 1.0)
    maps = [
        (sigma, r"$\log\,\Sigma_\star$  [M$_\odot$ kpc$^{-2}$]", ember, (7.0, 9.8), [7, 8, 9]),
        (av, r"$A_V$  [mag]", ember, (0.0, 2.6), [0, 1, 2]),
        (age, "mass-weighted age  [Gyr]", ember, (0.5, 2.7), [1, 2]),
    ]
    out = []
    for ax, (m, label, cmap, lim, ticks) in zip(axes[3:], maps):
        lo, hi = lim if lim else np.nanpercentile(m, [2, 98])
        ax.set_facecolor("white")
        im = ax.imshow(m, origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        image_axes(ax)
        out.append((ax, im, label, ticks))
    info = {
        "id": "JADES 33151",
        "z": b.redshift,
        "n_params": int(np.asarray(b.theta_map).size),
        "bands": b.n_bands,
        "total_log_mass": float(np.median(b.scalar_draws()["total_log_mass"])),
    }
    return axes, out, info


# ---------------------------------------------------------------------------------------
# (b) light components: bulge + disk
# ---------------------------------------------------------------------------------------


def panel_components(fig, spec):
    """Data, model, the two PSF-convolved components, residual, and the component SEDs."""
    d = np.load(INPUTS["components"])
    bands = [str(x).split(".")[-1] for x in d["bands"]]
    i_r, i_g, i_b = bands.index("F444W"), bands.index("F277W"), bands.index("F150W")
    n = d["flux"].shape[-1]
    half = int(round(1.6 / float(d["pixel_scale"])))  # show the central 3.2 arcsec
    c = n // 2
    cut = (slice(c - half, c + half), slice(c - half, c + half))
    sky = np.median(d["sky"], axis=0)[:, None, None]
    data = d["flux"] - sky
    comps = d["components_convolved"]  # (K, B, H, W), compact component first
    model = comps.sum(0)
    hi = np.percentile(np.stack([data[i][cut] for i in (i_r, i_g, i_b)]), 99.8)

    def rgb(cube):
        return rgb_image(cube[i_r][cut], cube[i_g][cut], cube[i_b][cut], hi)

    gs = GridSpecFromSubplotSpec(2, 5, subplot_spec=spec, height_ratios=[1.0, 1.25], hspace=0.16,
                                 wspace=0.05)  # fmt: skip
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    for ax, cube, text in zip(axes[:4], (data, model, comps[0], comps[1]),
                              ("JWST/NIRCam", "model", "bulge", "disk")):  # fmt: skip
        ax.imshow(rgb(cube), origin="lower", interpolation="nearest")
        caption(ax, text)
        image_axes(ax)
    scale_bar(axes[0], 2 * half, float(d["pixel_scale"]), 1.0)
    frac = 0.10
    sigma_eff = np.sqrt(d["variance"][i_g] + (frac * (model[i_g] + sky[i_g])) ** 2)
    chi = np.where(d["mask"][i_g] > 0, (data[i_g] - model[i_g]) / sigma_eff, np.nan)
    axes[4].set_facecolor("#dddddd")
    im = axes[4].imshow(chi[cut], origin="lower", cmap=ember_div, vmin=-4, vmax=4,
                        interpolation="nearest")  # fmt: skip
    axes[4].text(0.05, 0.94, r"residual / $\sigma$", transform=axes[4].transAxes, ha="left",
                 va="top", color=INK, fontsize=7,
                 bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))  # fmt: skip
    image_axes(axes[4])

    # component SEDs: posterior median and 16-84 % band, against the data summed in the frame
    ax = fig.add_subplot(gs[1, :])
    lam = np.array([PIVOT_UM[b] for b in bands])
    seds = d["component_seds"] / 1e3  # (S, K, B) in uJy
    styles = [("bulge", OXBLOOD, "-"), ("disk", INDIGO, "--")]
    for k, (name, color, ls) in enumerate(styles):
        lo, med, hi_ = np.percentile(seds[:, k], [16, 50, 84], axis=0)
        ax.fill_between(lam, lo, hi_, color=color, alpha=0.25, lw=0)
        ax.plot(lam, med, color=color, ls=ls, marker="o", ms=3, label=name)
    total = np.median(seds.sum(1), axis=0)
    ax.plot(lam, total, color=INK, lw=1.2, label="bulge + disk")
    valid = d["mask"] > 0
    obs = np.array([data[i][valid[i]].sum() for i in range(len(bands))]) / 1e3
    mod = np.array([model[i][valid[i]].sum() for i in range(len(bands))]) / 1e3
    ax.plot(lam, obs * total / mod, ls="none", marker="s", ms=4.5, mfc="white", mec=INK, mew=0.8,
            label="data (same pixels)", zorder=5)  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 14)
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(["1", "2", "5", "10"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([1, 1.5, 2, 3, 4])
    ax.set_xticklabels(["1", "1.5", "2", "3", "4"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.85, 4.75)
    ax.set_xlabel(r"observed wavelength  [$\mu$m]")
    ax.set_ylabel(r"flux density  [$\mu$Jy]")
    ax.legend(loc="lower right", ncol=2, handlelength=2.2, columnspacing=1.2)
    med = lambda x: np.median(x, axis=0)  # noqa: E731
    r_e = med(np.sqrt(d["sigma"][:, :, 0] * d["sigma"][:, :, 1]))
    kpc = Planck18.kpc_proper_per_arcmin(float(d["redshift"])).value / 60.0
    mass = med(d["sps"][:, :, list(d["sps_names"]).index("log_mass")])
    n_s = med(d["sersic_n"])
    txt = "\n".join(
        f"{name}:  $n$ = {n_s[k]:.1f},  $r_e$ = {r_e[k] * kpc:.1f} kpc,  "
        f"$\\log M_\\star$ = {mass[k]:.1f}"
        for k, (name, _, _) in enumerate(styles)
    )
    ax.text(0.025, 0.93, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK2,
            linespacing=1.5)  # fmt: skip
    info = {"id": "JADES 47870", "z": float(d["redshift"]), "bands": len(bands),
            "n_params": 43, "chi_im": im}  # fmt: skip
    return axes, ax, info


def panel_catalogue(fig, spec):
    """Panel (a): photometric redshifts of a catalogue against the survey's own."""
    cat = Table.read(INPUTS["catalogue"])
    with h5py.File(INPUTS["catalogue_fit"], "r") as f:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["param_names"]]
        lo, hi = f.attrs["param_bounds_lo"], f.attrs["param_bounds_hi"]
        ids = f["galaxy_id"][:]
        n_bands = len(f.attrs["band_names"]) if "band_names" in f.attrs else 12
        if "nss_samples" in f:  # nested sampling: physical draws (N, S, P)
            phys, rhat = f["nss_samples"][:], f["nss_rhat"][:]
        else:  # NUTS: unconstrained draws (N, chains, draws, P)
            theta, rhat = f["theta_samples"][:], f["rhat"][:]
            phys = (lo + (hi - lo) / (1.0 + np.exp(-theta))).reshape(len(ids), -1, len(names))
    id_col = next((c for c in cat.colnames if c.lower() == "id"), None)
    rows = np.arange(len(ids))  # fits are written in catalogue row order
    if id_col is not None:
        lookup = {int(v): i for i, v in enumerate(np.asarray(cat[id_col]))}
        rows = np.array([lookup[int(v)] for v in ids])
    z_ref = np.asarray(cat["lp_zpdf"], float)[rows]
    z16, z50, z84 = np.percentile(phys[:, :, names.index("redshift")], [16, 50, 84], axis=1)
    mass = np.median(phys[:, :, names.index("log_mass")], axis=1)
    ok = np.isfinite(z_ref) & (z_ref > 0)
    z_ref, z16, z50, z84, mass = (a[ok] for a in (z_ref, z16, z50, z84, mass))
    dz = (z50 - z_ref) / (1.0 + z_ref)
    nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    outliers = np.mean(np.abs(dz) > 0.15)

    ax = fig.add_subplot(spec)
    zmax = 4.0
    grid = np.linspace(0, zmax, 50)
    ax.plot(grid, grid, color=INK, lw=0.7, zorder=1)
    for sign in (-1, 1):
        ax.plot(
            grid, grid + sign * 0.15 * (1 + grid), color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1
        )
    ax.errorbar(z_ref, z50, yerr=[z50 - z16, z84 - z50], fmt="none", ecolor=MUTED, elinewidth=0.35,
                alpha=0.6, zorder=2)  # fmt: skip
    order = np.argsort(mass)
    sc = ax.scatter(z_ref[order], z50[order], c=mass[order], cmap=ember, vmin=8.0, vmax=11.5, s=9,
                    edgecolor="white", linewidth=0.25, zorder=3)  # fmt: skip
    ax.set_xlim(0, zmax)
    ax.set_ylim(0, zmax)
    ax.set_xlabel("Redshift, COSMOS2020 (LePhare)")
    ax.set_ylabel("Redshift, arachne")
    ax.text(0.05, 0.95, f"{ok.sum()} galaxies, {n_bands} bands\n"
            rf"$\sigma_{{\rm NMAD}}$ = {nmad:.3f}" f"\noutliers {100 * outliers:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK2,
            linespacing=1.4)  # fmt: skip
    info = {"n": int(ok.sum()), "bands": n_bands, "nmad": float(nmad), "outliers": float(outliers),
            "rhat_ok": float(np.mean(rhat.max(1) < 1.05))}  # fmt: skip
    return ax, sc, info


def main(output):
    """Draw the three panels and save the figure to ``output``."""
    fig = plt.figure(figsize=(10.0, 6.6))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.62, 1.0], hspace=0.27,
                     left=0.055, right=0.985, top=0.945, bottom=0.085)  # fmt: skip
    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.0, 2.05],
                                  wspace=0.17)  # fmt: skip
    ax_a, sc_a, info_a = panel_catalogue(fig, top[0])
    axes_b, ax_sed, info_b = panel_components(fig, top[1])
    axes_c, maps_c, info_c = panel_fields(fig, outer[1])
    fig.canvas.draw()
    for ax, im, label, ticks in maps_c:
        colorbar_below(fig, ax, im, label, ticks)
    box = axes_c[0].get_position()
    fig.text(box.x0, box.y1 + 0.012, "(c)  Correlated-field maps", fontweight="bold", fontsize=9,
             ha="left", va="bottom", color=INK)  # fmt: skip
    fig.text(
        axes_c[-1].get_position().x1, box.y1 + 0.013,
        f"{info_c['id']}, z = {info_c['z']:.2f}: {info_c['bands']} bands, "
        f"{info_c['n_params']:,} parameters, sampled with whitened NUTS",
        fontsize=7.5, ha="right", va="bottom", color=INK2,
    )  # fmt: skip
    box = axes_b[0].get_position()
    fig.text(box.x0, box.y1 + 0.012, "(b)  Light components", fontweight="bold", fontsize=9,
             ha="left", va="bottom", color=INK)  # fmt: skip
    fig.text(
        axes_b[-1].get_position().x1, box.y1 + 0.013,
        f"{info_b['id']}, z = {info_b['z']:.2f}: two S\u00e9rsic components, "
        f"{info_b['bands']} bands, nested sampling",
        fontsize=7.5, ha="right", va="bottom", color=INK2,
    )  # fmt: skip
    fig.canvas.draw()
    box_a = ax_a.get_position()
    y0 = ax_sed.get_position().y0
    ax_a.set_position([box_a.x0, y0, box_a.width, box.y1 - y0])  # align with panel (b)
    box_a = ax_a.get_position()
    fig.text(box_a.x0, box.y1 + 0.012, "(a)  Catalogues", fontweight="bold", fontsize=9,
             ha="left", va="bottom", color=INK)  # fmt: skip
    cax = fig.add_axes([box_a.x0 + 0.055 * box_a.width, box_a.y0 + 0.685 * box_a.height,
                        0.36 * box_a.width, 0.011])  # fmt: skip
    cb = fig.colorbar(sc_a, cax=cax, orientation="horizontal", ticks=[8, 9, 10, 11])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.4, which="both", top=False)
    cb.ax.minorticks_off()
    cb.ax.set_title(r"$\log_{10}\,M_\star/{\rm M}_\odot$", fontsize=7, pad=3)
    fig.savefig(output, dpi=200)
    print(f"saved {output}", info_a, info_c)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "arachne_frontpage.png")
