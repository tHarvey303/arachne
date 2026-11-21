"""Restricted Matérn likelihood for binned maps with a Gaussian observing beam."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.special import roots_genlaguerre

NU = 1.5
_NODES, _WEIGHTS = roots_genlaguerre(64, NU - 1)
_WEIGHTS /= _WEIGHTS.sum()


def matern32(distance2, length, beam_sigma=0.0):
    """Unit intrinsic variance with Arachne's break length and a circular Gaussian PSF.

    A Matérn is a Gamma mixture of squared exponentials with kernel
    exp(-r²/(4 ell² t)). Filtering both arguments by the PSF adds its variance
    to ell² t and reduces the variance by ell² t / (ell² t + sigma²) in 2D.
    """
    if beam_sigma == 0:
        x = np.sqrt(distance2) / length
        return (1 + x) * np.exp(-x)
    scale = length**2 * _NODES
    blurred = scale + beam_sigma**2
    return np.exp(-np.asarray(distance2)[..., None] / (4 * blurred)) @ (_WEIGHTS * scale / blurred)


class BinGeometry:
    """Project a stationary spaxel covariance through weighted continuum bins."""

    def __init__(self, segmentation, weights, labels, radius, beam_sigma, max_points=12):
        """Build a sparse average over the pair separations of the bin footprints."""
        self.labels = np.asarray(labels)
        yy, xx = np.indices(segmentation.shape)
        xy = np.zeros((len(labels), max_points, 2), dtype=np.int32)
        w = np.zeros(xy.shape[:2])
        self.design = np.ones((len(labels), 2))
        blurred_radius = gaussian_filter(radius, beam_sigma, mode="nearest")
        for i, label in enumerate(labels):
            mask = segmentation == label
            bin_xy = np.column_stack((yy[mask], xx[mask]))
            bin_w = np.maximum(weights[mask], 0)
            if bin_w.sum() <= 0:
                bin_w = np.ones(len(bin_w))
            bin_w = bin_w / bin_w.sum()
            self.design[i, 1] = bin_w @ blurred_radius[mask]
            if len(bin_w) > max_points:
                indices = np.searchsorted(
                    np.cumsum(bin_w), (np.arange(max_points) + 0.5) / max_points
                )
                bin_xy, bin_w = bin_xy[indices], np.full(max_points, 1 / max_points)
            xy[i, : len(bin_w)], w[i, : len(bin_w)] = bin_xy, bin_w

        self.i, self.j = np.tril_indices(len(labels))
        dxy = xy[self.i, :, None] - xy[self.j, None, :]
        d2 = np.sum(dxy**2, axis=-1)
        pair_weights = w[self.i, :, None] * w[self.j, None, :]
        pair, left, right = np.nonzero(pair_weights)
        self.distance2, inverse = np.unique(d2[pair, left, right], return_inverse=True)
        self.projection = csr_matrix(
            (pair_weights[pair, left, right], (pair, inverse)),
            shape=(len(self.i), len(self.distance2)),
        )
        self.beam_sigma = beam_sigma

    def covariance(self, length):
        """Evaluate the observed covariance at unit intrinsic variance."""
        values = self.projection @ matern32(self.distance2, length, self.beam_sigma)
        matrix = np.empty((len(self.labels), len(self.labels)))
        matrix[self.i, self.j] = values
        matrix[self.j, self.i] = values
        return matrix


class RestrictedLikelihood:
    """Integrate out a free level and radial slope with a flat mean prior."""

    def __init__(self, y, sigma, trend):
        """Pre-whiten measurements and mean design by the diagonal noise."""
        self.y = y / sigma
        self.trend = trend / sigma[:, None]
        self.sigma = sigma

    def at_length(self, covariance):
        """Return a cheap profile function in log amplitude for this covariance."""
        eigenvalues, vectors = np.linalg.eigh(
            covariance / self.sigma[:, None] / self.sigma[None, :]
        )
        eigenvalues = np.maximum(eigenvalues, 0)
        y = vectors.T @ self.y
        trend = vectors.T @ self.trend

        def profile(log_amplitude):
            diagonal = 1 + np.exp(2 * log_amplitude) * eigenvalues
            weights = 1 / diagonal
            gram = (trend.T * weights) @ trend
            coeff = np.linalg.solve(gram, (trend.T * weights) @ y)
            residual = y - trend @ coeff
            loss = 0.5 * (
                np.log(diagonal).sum() + np.linalg.slogdet(gram)[1] + (residual**2) @ weights
            )
            return loss, coeff

        return profile


def fit_profiles(geometry, maps, lengths, amplitude_bounds):
    """Profile the intrinsic amplitude at each trial break length for every map."""
    selected = {}
    for name, (values, errors) in maps.items():
        valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
        if valid.sum() >= 25:
            trend, y = geometry.design[valid], values[valid]
            detrended = y - trend @ np.linalg.lstsq(trend, y, rcond=None)[0]
            rms = max(np.std(detrended), np.median(errors[valid]))
            nuggets = rms * np.array([0, 0.15, 0.25, 0.38, 0.50, 0.63, 0.76, 0.89, 1.02, 1.2])
            likelihoods = [
                RestrictedLikelihood(y, np.hypot(errors[valid], jitter), trend)
                for jitter in nuggets
            ]
            selected[name] = (valid, likelihoods, nuggets, [])

    for length in lengths:
        covariance = geometry.covariance(length)
        for name, (valid, likelihoods, nuggets, rows) in selected.items():
            submatrix = covariance[np.ix_(valid, valid)]
            bounds = np.log(amplitude_bounds[name])
            candidates, nulls = [], []
            for likelihood, jitter in zip(likelihoods, nuggets, strict=True):
                profile = likelihood.at_length(submatrix)
                result = minimize_scalar(lambda x: profile(x)[0], bounds=bounds, method="bounded")
                loss, coeff = profile(result.x)
                null, _ = profile(-np.inf)
                normalization = np.log(likelihood.sigma).sum()
                candidates.append((loss + normalization, np.exp(result.x), *coeff, jitter))
                nulls.append(null + normalization)
            rows.append((*min(candidates, key=lambda row: row[0]), min(nulls)))

    result = {}
    for name, (valid, _, _, rows) in selected.items():
        loss, amp, level, slope, nugget, null = np.asarray(rows).T
        best = np.argmin(loss)
        delta = loss - loss[best]
        inside = np.flatnonzero(delta <= 1.92)
        result[name] = {
            "n_bins": int(valid.sum()),
            "amplitude": float(amp[best]),
            "bin_scatter": float(nugget[best]),
            "level": float(level[best]),
            "slope_per_re": float(slope[best]),
            "length_index": int(best),
            "ci_low_index": int(inside[0]),
            "ci_high_index": int(inside[-1]),
            "delta_null": float(null[best] - loss[best]),
            "log_likelihood": (-delta).tolist(),
            "amplitude_profile": amp.tolist(),
        }
    return result
