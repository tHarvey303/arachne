"""Check length recovery through irregular binning and a finite observing beam."""

from __future__ import annotations

import argparse
import json

import numpy as np
from field_covariance import BinGeometry, fit_profiles, matern32
from scipy.linalg import cholesky
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist


def validate(seed, repetitions):
    """Draw intrinsic maps before applying the beam and binning in image space."""
    rng = np.random.default_rng(seed)
    n, pad, beam = 26, 7, 1.55
    yy, xx = np.indices((n, n)) - (n - 1) / 2
    radius = np.hypot(xx, yy) / 9
    segmentation = 1 + np.arange(n * n).reshape(n, n) // (2 * n) * (n // 2)
    segmentation += np.arange(n)[None, :] // 2
    segmentation[(radius > 1.4) | (rng.uniform(size=radius.shape) < 0.07)] = 0
    labels = np.unique(segmentation)[1:]
    radius_pad = np.pad(radius, pad, mode="edge")
    xy = np.indices(radius_pad.shape).reshape(2, -1).T
    d2 = cdist(xy, xy, metric="sqeuclidean")
    weights = np.exp(-0.4 * radius)
    geometry = BinGeometry(segmentation, weights, labels, radius, beam)
    groups = [segmentation == label for label in labels]
    widths = np.geomspace(0.4, 12.0, 35)
    restored = []
    for injected_length in (0.65, 2.0, 4.0):
        covariance = matern32(d2, injected_length)
        root = cholesky(covariance, lower=True)
        dataset = {}
        for rep in range(repetitions):
            intrinsic = (root @ rng.standard_normal(len(xy))).reshape(radius_pad.shape)
            signal = 0.5 - 0.15 * radius_pad + 0.22 * intrinsic
            blurred = gaussian_filter(signal, beam)[pad:-pad, pad:-pad]
            report = np.array([np.average(blurred[mask], weights=weights[mask]) for mask in groups])
            documented = rng.uniform(0.015, 0.035, size=len(labels))
            report += rng.normal(scale=np.hypot(documented, 0.035))
            dataset[str(rep)] = (report, documented)
        fitted = fit_profiles(geometry, dataset, widths, dict.fromkeys(dataset, (0.006, 1.5)))
        values = list(fitted.values())
        recovered = np.array([widths[item["length_index"]] for item in values])
        inside = [
            widths[item["ci_low_index"]] <= injected_length <= widths[item["ci_high_index"]]
            for item in values
        ]
        restored.append(
            {
                "injected_length_pixels": injected_length,
                "injected_amplitude": 0.22,
                "recovered_length_percentiles": np.percentile(recovered, [16, 50, 84]).tolist(),
                "recovered_amplitude_median": float(
                    np.median([item["amplitude"] for item in values])
                ),
                "profile_interval_hits": int(sum(inside)),
                "profile_intervals_touching_grid_edge": sum(
                    item["ci_low_index"] == 0 or item["ci_high_index"] == len(widths) - 1
                    for item in values
                ),
                "repetitions": repetitions,
            }
        )
    return restored


def main():
    """Print the injection recovery measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=9423)
    parser.add_argument("--repetitions", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(validate(args.seed, args.repetitions), indent=2))


if __name__ == "__main__":
    main()
