"""Fit a galaxy image with a free-form per-pixel SPS parameter map.

For a 75x75 image with 12 SPS parameters this gives ~67,500 free parameters.
GPU with sufficient VRAM (>=16 GB) strongly recommended at this scale.

Run:
    python fit_pixel_map.py --flux-dir /path/to/data --checkpoint model.pkl
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from arachne import (
    ForwardModel,
    FreeFormPixelMap,
    JAXFlowEmulator,
    NUTSSampler,
    ObservationCube,
    PSFModel,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fit a galaxy with a free-form pixel map.")
    parser.add_argument("--flux-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("posterior_pixel_map.h5"))
    parser.add_argument("--n-warmup", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--smoothness", type=float, default=1.0,
                        help="L2 gradient penalty strength (higher = smoother)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    band_names = [
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F444W",
    ]

    flux_paths = [args.flux_dir / f"flux_{b.split('.')[-1].lower()}.fits" for b in band_names]
    var_paths = [args.flux_dir / f"var_{b.split('.')[-1].lower()}.fits" for b in band_names]
    psf_paths = {b: args.flux_dir / f"psf_{b.split('.')[-1].lower()}.fits" for b in band_names}

    print("Loading observations...")
    obs = ObservationCube.from_fits(flux_paths=flux_paths, variance_paths=var_paths, band_names=band_names)

    print("Loading PSF models...")
    psf = PSFModel.from_fits(psf_paths)

    print("Loading emulator...")
    param_names = ["log_stellar_mass", "log_age", "log_metallicity", "tau_v"]
    emulator = JAXFlowEmulator.from_synference_checkpoint(
        args.checkpoint, param_names=param_names, band_names=band_names, direction="forward"
    )

    H, W = obs.image_shape
    print(f"Image shape: {H}x{W}")

    param_bounds = {
        "log_stellar_mass": (6.0, 12.0),
        "log_age": (7.0, 10.1),
        "log_metallicity": (-2.0, 0.5),
        "tau_v": (0.0, 4.0),
    }

    print("Setting up free-form pixel map...")
    spatial_model = FreeFormPixelMap(
        image_shape=(H, W),
        sps_param_names=param_names,
        param_bounds=param_bounds,
        smoothness_strength=args.smoothness,
    )
    print(f"Free parameters: {spatial_model.n_params}")

    forward_model = ForwardModel.build(
        obs=obs, psf_model=psf, spatial_model=spatial_model, emulator=emulator
    )

    sampler = NUTSSampler(
        forward_model=forward_model,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        max_num_doublings=5,  # Cap to avoid OOM on large pixel maps
    )

    rng_key = jax.random.PRNGKey(args.seed)
    theta_init = jnp.zeros(spatial_model.n_params)

    print(f"Running NUTS ({args.n_warmup} warmup + {args.n_samples} samples)...")
    result = sampler.run(theta_init, rng_key)
    print(f"Done. Acceptance rate: {result.acceptance_rate:.3f}")

    result.to_hdf5(str(args.output))
    print(f"Saved posterior to {args.output}")

    param_maps = result.get_parameter_map(image_shape=(H, W))
    fig, axes = plt.subplots(2, len(param_names), figsize=(4 * len(param_names), 8))
    for i, name in enumerate(param_names):
        median_map = np.asarray(param_maps[name][1])
        sigma_map = np.asarray((param_maps[name][2] - param_maps[name][0]) / 2)
        im1 = axes[0, i].imshow(median_map, origin="lower", cmap="viridis")
        axes[0, i].set_title(f"{name} (median)")
        plt.colorbar(im1, ax=axes[0, i])
        im2 = axes[1, i].imshow(sigma_map, origin="lower", cmap="Reds")
        axes[1, i].set_title(f"{name} (1σ)")
        plt.colorbar(im2, ax=axes[1, i])
    plt.tight_layout()
    plot_path = args.output.with_suffix(".png")
    plt.savefig(plot_path, dpi=150)
    print(f"Parameter maps saved to {plot_path}")


if __name__ == "__main__":
    main()
