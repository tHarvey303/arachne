"""Convert an old ``demo_correlated_field.py`` run directory into a degraded results bundle.

Old runs (before results bundles) saved ``maps.npz``, ``laplace.npz``, ``nuts.npz`` and
``metrics.json`` but no posterior draws.  The degraded bundle holds the data, the MAP model
cube and parameter maps, and the posterior median / std / information maps, so the static
plots and the viewer can show MAP-only SEDs, residuals and parameter maps for any region
(no model uncertainty bands, no per-pixel posteriors).

The constants describe the **old 4-field demo** (fields log_mass, Av, log10metallicity,
logsfr_ratio_0; everything else fixed) and are hard-coded on purpose: the current demo has
a different setup.

Usage::

    python scripts/convert_correlated_field_run.py RUN_DIR [--out results_degraded.h5]
        [--pixel-scale 0.03] [--emulator CKPT|toy]

``--emulator`` (needs JAX) adds the unconvolved MAP model (and the unconvolved truth): a
ParrotEmulatorV2 checkpoint, or ``toy`` for the demo's toy SED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from arachne.results import (
    BundleContent,
    Region,
    TruthContent,
    detection_region,
    pivot_um,
    write_content,
)

OLD_FIELDS = ["log_mass", "Av", "log10metallicity", "logsfr_ratio_0"]
OLD_FIXED = {
    "redshift": 2.0, "slope": 0.0, "fesc_lya": 0.2, "dust_bump_amplitude": 1.0,
    **{f"logsfr_ratio_{i}": 0.0 for i in range(1, 5)},
}  # fmt: skip
SPS_PARAM_NAMES = [
    "redshift", "log_mass", "slope", "fesc_lya", "dust_bump_amplitude", "log10metallicity",
    "Av", "logsfr_ratio_0", "logsfr_ratio_1", "logsfr_ratio_2", "logsfr_ratio_3",
    "logsfr_ratio_4",
]  # fmt: skip
WIDE8 = ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F410M", "F444W"]
LABELS = {
    "log_mass": r"$\log M_*$ / px",
    "Av": r"$A_V$",
    "log10metallicity": r"$\log Z$",
    "logsfr_ratio_0": r"logsfr_ratio_0 (0-3 / 3-10 Myr)",
}
UNITS = {"log_mass": "dex", "Av": "mag", "log10metallicity": "dex", "total_log_mass": "dex"}


def _band_names(metrics: dict, n_bands: int) -> list[str]:
    bands = metrics.get("bands")
    if bands is None:
        if n_bands != len(WIDE8):
            raise ValueError(f"metrics.json has no 'bands' and the data have {n_bands} bands")
        bands = [f"JWST/NIRCam.{f}" for f in WIDE8]
    return list(bands)


def _summaries(npz: dict) -> dict[str, np.ndarray]:
    return {"p50": npz["median"], "std": npz["std"], "information": npz["information"]}


def _load_npz(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _intrinsic(emulator_arg: str, band_names: list[str], maps: np.ndarray) -> np.ndarray:
    """Unconvolved image (B, H, W) of the maps (OLD_FIELDS order) with OLD_FIXED."""
    import jax.numpy as jnp

    nf, h, w = maps.shape
    rows = np.empty((h * w, len(SPS_PARAM_NAMES)))
    for j, name in enumerate(SPS_PARAM_NAMES):
        if name in OLD_FIELDS:
            rows[:, j] = maps[OLD_FIELDS.index(name)].ravel()
        else:
            rows[:, j] = OLD_FIXED[name]
    piv = pivot_um(band_names)
    if emulator_arg == "toy":
        # the old demo's ToySEDEmulator
        lam = piv[None, :]
        z, logm, logz, av, r0 = (rows[:, i : i + 1] for i in (0, 1, 5, 6, 7))
        rest = lam / (1.0 + z)
        beta = 0.4 + 0.5 * (logz + 2.5)
        young = 1.0 / (1.0 + np.exp(-r0))
        brk = np.where(rest < 0.40, 0.35 + 0.65 * young, 1.0)
        dust = 10.0 ** (-0.4 * av * (rest / 0.55) ** -1.0)
        flux = 2.0e-6 * 10.0**logm * lam**beta * brk * dust
    else:
        from arachne import load_emulator

        emu = load_emulator(emulator_arg)
        if list(emu.param_names) != SPS_PARAM_NAMES:
            raise ValueError(f"emulator inputs {emu.param_names} != {SPS_PARAM_NAMES}")
        idx = [list(emu.band_names).index(b) for b in band_names]
        flux = np.concatenate(
            [
                np.asarray(emu.predict(jnp.asarray(rows[i : i + 4096], jnp.float32)))[:, idx]
                for i in range(0, len(rows), 4096)
            ]
        )
    return flux.T.reshape(len(band_names), h, w)


def convert_run(
    run_dir: str | Path,
    out: str | Path | None = None,
    *,
    pixel_scale: float = 0.03,
    emulator: str | None = None,
) -> Path:
    """Convert one run directory.

    Args:
        run_dir: Directory with ``maps.npz``, ``metrics.json`` and optionally
            ``laplace.npz`` / ``nuts.npz``.
        out: Output path (default ``RUN_DIR/results_degraded.h5``).
        pixel_scale: Arcsec per pixel (not saved by the old demo).
        emulator: Checkpoint path or ``"toy"`` to add the unconvolved MAP model and truth.

    Returns:
        The bundle path.
    """
    run_dir = Path(run_dir)
    out = Path(out) if out is not None else run_dir / "results_degraded.h5"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    maps = _load_npz(run_dir / "maps.npz")
    if maps is None:
        raise FileNotFoundError(run_dir / "maps.npz")
    fields = [str(f) for f in maps.get("fields", OLD_FIELDS)]
    if fields != OLD_FIELDS:
        raise ValueError(f"expected the old 4-field run ({OLD_FIELDS}), got {fields}")
    data = np.asarray(maps["data"], np.float32)
    nb, h, w = data.shape
    band_names = _band_names(metrics, nb)
    sigma = np.asarray(maps["sigma"], np.float64)
    variance = np.broadcast_to((sigma**2)[:, None, None], data.shape).astype(np.float32)
    mask = np.ones_like(data)

    posts = {"nuts": _load_npz(run_dir / "nuts.npz"), "laplace": _load_npz(run_dir / "laplace.npz")}
    primary = "nuts" if posts["nuts"] is not None else "laplace"
    summaries = _summaries(posts[primary]) if posts[primary] is not None else None
    extra = {"start_maps": np.asarray(maps["start"])}
    for name, post in posts.items():
        if post is not None and name != primary:
            for key, arr in _summaries(post).items():
                extra[f"{name}_{key}"] = arr
    if posts["laplace"] is not None and "theta" in posts["laplace"]:
        extra["laplace_theta"] = posts["laplace"]["theta"]

    map_maps = np.asarray(maps["map"], np.float64)
    truth = regions = None
    regions = {"all": Region.from_mask(np.ones((h, w), bool), name="all")}
    seg = detection_region(data, variance, mask)
    if seg.any():
        regions["segment"] = Region.from_mask(seg, name="segment")
    intrinsic_map = None
    if emulator:
        intrinsic_map = _intrinsic(emulator, band_names, map_maps)
    if metrics.get("truth") and "truth" in maps:
        tmaps = np.asarray(maps["truth"], np.float64)
        weight = 10.0 ** tmaps[0]
        regions["galaxy"] = Region.from_mask(weight > 0.02 * weight.max(), name="galaxy")
        truth = TruthContent(
            param_maps=tmaps,
            scalars=np.array([np.log10(np.sum(10.0 ** tmaps[0]))]),
            intrinsic=_intrinsic(emulator, band_names, tmaps) if emulator else None,
        )

    n_bands_label = "medium" if nb == 17 else "wide8" if nb == 8 else f"{nb}bands"
    content = BundleContent(
        band_names=band_names,
        pivot_um=pivot_um(band_names),
        pixel_scale=float(pixel_scale),
        flux=data,
        variance=variance,
        mask=mask,
        model_error_frac=np.zeros(nb),
        convolved_map=np.asarray(maps["model_image"], np.float32),
        intrinsic_map=intrinsic_map,
        map_names=list(OLD_FIELDS),
        param_maps_map=map_maps,
        param_summaries=summaries,
        scalar_names=["total_log_mass"],
        scalars_map=np.array([np.log10(np.sum(10.0 ** map_maps[0]))]),
        fixed_params=dict(OLD_FIXED),
        mass_param="log_mass",
        labels=dict(LABELS),
        units=dict(UNITS),
        truth=truth,
        theta_map=np.asarray(maps["theta_map"]) if "theta_map" in maps else None,
        diagnostics={"converted_from": primary if summaries is not None else "map"},
        regions=regions,
        metrics=metrics,
        extra=extra,
        posterior_source="map",
        spatial_model="CorrelatedFieldMap",
        redshift=OLD_FIXED["redshift"],
        title=f"{metrics.get('truth', run_dir.name)}_{n_bands_label}",
        degraded=True,
        provenance={"converted_from": str(run_dir.resolve()), "emulator": emulator or ""},
    )
    return write_content(out, content, overwrite=True)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", help="run directories (maps.npz, metrics.json, ...)")
    ap.add_argument("--out", default=None, help="output path (one run dir only)")
    ap.add_argument("--pixel-scale", type=float, default=0.03)
    ap.add_argument("--emulator", default=None, help="checkpoint or 'toy' (adds intrinsic)")
    args = ap.parse_args(argv)
    if args.out and len(args.run_dirs) > 1:
        ap.error("--out needs a single run directory")
    for run_dir in args.run_dirs:
        path = convert_run(run_dir, args.out, pixel_scale=args.pixel_scale, emulator=args.emulator)
        print(f"{run_dir} -> {path} ({path.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
