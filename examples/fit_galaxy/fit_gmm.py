"""Fit a galaxy image with a K-component GMM spatial model.

Example usage on a real JWST cutout. Requires:
- A synference forward-direction checkpoint
- Multi-band FITS flux/variance images
- Per-band PSF FITS files

Run:
    python fit_gmm.py --flux-dir /path/to/data --checkpoint model.pkl --n-components 3
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from arachne import (
    ForwardModel,
    GaussianMixtureSpatialModel,
    JAXFlowEmulator,
    NUTSSampler,
    ObservationCube,
    PSFModel,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fit a galaxy with a GMM spatial model.")
    parser.add_argument("--flux-dir", type=Path, required=True, help="Directory with FITS files")
    parser.add_argument("--checkpoint", type=Path, required=True, help="synference checkpoint")
    parser.add_argument("--output", type=Path, default=Path("posterior_gmm.h5"))
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--n-warmup", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=1000)
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
    obs = ObservationCube.from_fits(
        flux_paths=flux_paths,
        variance_paths=var_paths,
        band_names=band_names,
    )

    print("Loading PSF models...")
    psf = PSFModel.from_fits(psf_paths)

    print("Loading emulator...")
    param_names = ["log_stellar_mass", "log_age", "log_metallicity", "tau_v"]
    emulator = JAXFlowEmulator.from_synference_checkpoint(
        args.checkpoint,
        param_names=param_names,
        band_names=band_names,
        direction="forward",
    )

    H, W = obs.image_shape
    print(f"Image shape: {H}x{W}, {obs.n_bands} bands")

    param_bounds = {
        "log_stellar_mass": (6.0, 12.0),
        "log_age": (7.0, 10.1),
        "log_metallicity": (-2.0, 0.5),
        "tau_v": (0.0, 4.0),
    }

    print(f"Setting up GMM spatial model with K={args.n_components}...")
    spatial_model = GaussianMixtureSpatialModel(
        n_components=args.n_components,
        sps_param_names=param_names,
        param_bounds=param_bounds,
        image_shape=(H, W),
    )
    print(f"Free parameters: {spatial_model.n_params}")

    forward_model = ForwardModel.build(
        obs=obs, psf_model=psf, spatial_model=spatial_model, emulator=emulator
    )

    sampler = NUTSSampler(
        forward_model=forward_model,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        max_num_doublings=5,
    )

    rng_key = jax.random.PRNGKey(args.seed)
    theta_init = jnp.zeros(spatial_model.n_params)

    print(f"Running NUTS ({args.n_warmup} warmup + {args.n_samples} samples)...")
    result = sampler.run(theta_init, rng_key)
    print(f"Done. Acceptance rate: {result.acceptance_rate:.3f}")

    result.to_hdf5(str(args.output))
    print(f"Saved posterior to {args.output}")

    # Quick diagnostic plot
    param_maps = result.get_parameter_map(image_shape=(H, W))
    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))
    for ax, name in zip(axes, param_names):
        median_map = np.asarray(param_maps[name][1])  # 50th percentile
        im = ax.imshow(median_map, origin="lower", cmap="viridis")
        ax.set_title(name)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plot_path = args.output.with_suffix(".png")
    plt.savefig(plot_path, dpi=150)
    print(f"Parameter maps saved to {plot_path}")


if __name__ == "__main__":
    main()
