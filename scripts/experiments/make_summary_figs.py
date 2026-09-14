#!/usr/bin/env python3
"""Summary figures for the emulator optimization campaign."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path("/cosma/apps/dp276/dc-harv3/arachne/scripts/outputs/experiments")
OUT = D / "summary"
OUT.mkdir(exist_ok=True)

BLUE, AQUA, GRAY = "#2a78d6", "#1baf7a", "#6b6a63"


def load(name):
    with open(D / f"{name}.json") as fh:
        return json.load(fh)


plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d9d8d0", "axes.grid": True, "grid.color": "#eceae2",
    "grid.linewidth": 0.7, "axes.axisbelow": True,
    "text.color": "#1a1a19", "axes.labelcolor": "#1a1a19",
    "xtick.color": "#6b6a63", "ytick.color": "#6b6a63", "font.size": 10,
})

# ---- Fig 1: data-scaling curves --------------------------------------------
n = np.array([100_000, 300_000, 925_000])
new = [load("S_win_100k")["det_scatter"], load("S_win_300k")["det_scatter"],
       load("A3_mass_sfr_f16")["det_scatter"]]
base = [load("S_base_100k")["det_scatter"], load("S_base_300k")["det_scatter"],
        load("A0_base")["det_scatter"]]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.loglog(n, base, "-o", color=GRAY, lw=2, ms=6, label="original recipe")
ax.loglog(n, new, "-o", color=BLUE, lw=2, ms=6, label="V2 recipe (mass-norm + Fourier-z)")
ref = load("ref_checkpoint")["det_scatter"]
ax.axhline(ref, color=AQUA, lw=1.5, ls="--")
ax.annotate("production checkpoint (1M rows, full budget)", (1.05e5, ref * 1.05),
            fontsize=8.5, color="#3f6f5c")
for x, y in zip(n, new):
    ax.annotate(f"{y:.3f}", (x, y * 0.88), fontsize=8.5, color="#1c5cab", ha="center")
ax.set_xlabel("training rows")
ax.set_ylabel("detectable-flux scatter (asinh mag, median over bands)")
ax.set_title("Emulator error vs training-set size (fixed 180k-step budget)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig_scaling.png", dpi=150)
plt.close(fig)

# ---- Fig 2: staged improvements (ablation waterfall) ------------------------
runs = [
    ("ref_checkpoint", "production\n(original)"),
    ("A0_base", "original arch\n(800 ep)"),
    ("A2_mass_sfr", "+ mass-norm\n+ sfr-arsinh"),
    ("A3_mass_sfr_f16", "+ Fourier-z\n(k=16)"),
    ("F_res512_f64", "ResMLP 512x4\nFourier-64, long"),
    ("FINAL_v2", "FINAL V2\n(shipped ckpt)"),
]
labels, det, trans, bright = [], [], [], []
for key, lab in runs:
    try:
        r = load(key)
    except FileNotFoundError:
        continue
    labels.append(lab)
    det.append(r["det_scatter"])
    trans.append(r["trans_rms"])
    bright.append(100 * r["bright_flux_err"])

x = np.arange(len(labels))
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
for ax, vals, title, color in [
    (axes[0], det, "detectable scatter\n(asinh mag, >1 nJy)", BLUE),
    (axes[1], trans, "dropout-transition RMS\n(asinh mag)", BLUE),
    (axes[2], bright, "bright-flux error\n(% median, >10 nJy)", BLUE),
]:
    bars = ax.bar(x, vals, width=0.62, color=color)
    bars[0].set_color(GRAY)
    if len(vals) == len(runs):
        bars[-1].set_color(AQUA)
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.3f}" if v < 1 else f"{v:.2f}", (xi, v), ha="center",
                    va="bottom", fontsize=8, color="#1a1a19")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=7.5,
                       rotation=28, ha="right")
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="x", visible=False)
fig.suptitle("ParrotEmulator optimization: held-out test metrics", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "fig_improvements.png", dpi=150)
plt.close(fig)

print(f"figures written to {OUT}")
