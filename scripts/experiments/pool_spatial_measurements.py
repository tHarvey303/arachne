"""Pool galaxy-level Matérn likelihoods and evaluate population models on held-out galaxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from calibrate_pipe3d_spatial_priors import RELEASE
from pipe3d_field_measurements import FIELD_UNITS
from scipy.optimize import minimize
from scipy.special import ndtr


def length_weights(grid, loc, scale):
    """Normal probability per log-length cell with censored end cells."""
    return np.diff(ndtr((grid - loc) / scale), prepend=0, append=1)


def fit_lengths(rows, name, grid, unit):
    """Fit a log-normal distribution to the full profiles, including flat profiles."""
    factors = np.array([np.log(row["re_kpc"]) if unit == "kpc" else 0.0 for row in rows])
    lik = np.exp([row["fields"][name]["log_likelihood"] for row in rows])

    def objective(params):
        loc, log_scale = params
        grid_scaled = (grid[None, :] + factors[:, None] - loc) / np.exp(log_scale)
        probabilities = np.diff(ndtr(grid_scaled), prepend=0, append=1, axis=1)
        return -np.log(np.maximum(np.sum(lik * probabilities, axis=1), 1e-250)).sum()

    bounds = [
        (float(grid[0] + np.min(factors)), float(grid[-1] + np.max(factors))),
        (np.log(0.30), np.log(1.50)),
    ]
    result = min(
        (
            minimize(
                objective,
                (np.median(grid) + np.median(factors) + shift, np.log(0.6)),
                bounds=bounds,
                method="L-BFGS-B",
            )
            for shift in (-0.8, 0, 0.8)
        ),
        key=lambda r: r.fun,
    )
    return tuple(map(float, (result.x[0], np.exp(result.x[1]))))


def score_length(row, name, grid, prior, unit):
    """Integrated profile score for a single galaxy."""
    shift = np.log(row["re_kpc"]) if unit == "kpc" else 0
    weights = length_weights(grid + shift, *prior)
    return float(np.log(weights @ np.exp(row["fields"][name]["log_likelihood"])))


def resample_score(differences, rng):
    """Galaxy bootstrap uncertainty for a paired score difference."""
    values = np.asarray(differences)
    samples = rng.choice(values, (4000, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "bootstrap_95pct": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def summarize(measurements):
    """Compare pooled and activity-conditioned distributions in kpc and R_e."""
    rng = np.random.default_rng(measurements["seed"])
    exploratory_rng = np.random.default_rng(702)
    ell = np.asarray(measurements["ell_over_re_grid"])
    cells = 0.5 * (np.log(ell[1:]) + np.log(ell[:-1]))
    training_z = [g["z"] for g in measurements["galaxies"] if g["split"] == "train"]
    result = {
        "version": 1,
        "source": measurements.get("source", RELEASE),
        "nu": measurements["nu"],
        "redshift_range": [min(training_z), max(training_z)],
        "transferable_fields": measurements.get("transferable_fields", ["Av", "log10metallicity"]),
        "fields": {},
    }
    for name, unit in measurements.get("field_units", FIELD_UNITS).items():
        available = [g for g in measurements["galaxies"] if name in g["fields"]]
        observational_sfh = name == "sfh_10_100_over_100_1000_myr"
        if observational_sfh:
            available = [g for g in available if g["group"] == "active"]
        train = [g for g in available if g["split"] == "train"]
        test = [g for g in available if g["split"] == "test"]
        if len(train) < 3 or not test:
            raise ValueError(f"{name} needs at least three training and one test galaxy")
        models = {}
        for length_unit in ("re", "kpc"):
            prior = fit_lengths(train, name, cells, length_unit)
            scores = [score_length(g, name, cells, prior, length_unit) for g in test]
            models[length_unit] = {
                "median": float(np.exp(prior[0])),
                "log_scatter": prior[1],
                "test_mean_score": float(np.mean(scores)),
            }
            if length_unit == "re":
                prior_re = prior
                scores_re = scores
            else:
                models[length_unit]["test_score_delta_from_pooled_re"] = resample_score(
                    np.asarray(scores) - scores_re, rng
                )
        for key in ("group", "snapshot", "log_mass", "axis_ratio"):
            if key not in train[0] or (key == "group" and observational_sfh):
                continue
            threshold = (
                float(np.median([g[key] for g in train]))
                if key in ("log_mass", "axis_ratio")
                else None
            )

            def subgroup(galaxy):
                if threshold is None:
                    return galaxy[key]
                return "below_median" if galaxy[key] < threshold else "at_or_above_median"

            groups = {}
            for group in sorted({subgroup(g) for g in train}):
                group_train = [g for g in train if subgroup(g) == group]
                if len(group_train) >= 3:
                    groups[group] = fit_lengths(group_train, name, cells, "re")
            if len(groups) < 2 or any(subgroup(g) not in groups for g in test):
                continue
            scores = [score_length(g, name, cells, groups[subgroup(g)], "re") for g in test]
            comparison = {
                "groups": {
                    group: {"median": float(np.exp(loc)), "log_scatter": scale}
                    for group, (loc, scale) in groups.items()
                },
                "test_score_delta_from_pooled_re": resample_score(
                    np.asarray(scores) - scores_re, rng if threshold is None else exploratory_rng
                ),
            }
            if threshold is not None:
                comparison["training_median_threshold"] = threshold
            models["activity_re" if key == "group" else f"{key}_re"] = comparison
        amplitudes = []
        resolved = []
        for galaxy in train:
            m = galaxy["fields"][name]
            if m["delta_null"] >= 4.5:
                weights = length_weights(cells, *prior_re) * np.exp(m["log_likelihood"])
                amplitudes.append(np.average(np.log(m["amplitude_profile"]), weights=weights))
                if 0 < m["ci_low_index"] <= m["ci_high_index"] < len(ell) - 1 and ell[
                    m["ci_low_index"]
                ] >= max(0.25 * galaxy["beam_fwhm_over_re"], galaxy.get("pixel_over_re", 0)):
                    resolved.append(galaxy["id"])
        amplitude = None
        if amplitudes:
            p16, p50, p84 = np.quantile(amplitudes, [0.16, 0.5, 0.84])
            amplitude = {"median": float(np.exp(p50)), "log_scatter": float((p84 - p16) / 2)}
        else:
            result["transferable_fields"] = [f for f in result["transferable_fields"] if f != name]
        result["fields"][name] = {
            "unit": unit,
            "n_train": len(train),
            "n_test": len(test),
            "n_train_detected": len(amplitudes),
            "n_train_bounded": len(resolved),
            "amplitude": amplitude,
            "radial_slope_per_re_quantiles": np.quantile(
                [g["fields"][name]["slope_per_re"] for g in train], [0.16, 0.50, 0.84]
            ).tolist(),
            "length": models,
        }
    return result


def main():
    """Read per-galaxy measurements and write population summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.measurements.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
