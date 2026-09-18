"""CPU-only unit tests for the pure helpers of ``examples/fit_jades_dja.py``.

Only the deterministic, numpy-level helpers in ``examples/real_data_utils.py``
are covered: PSF resampling, the neighbour/plume mask, the half-light size
guess, the log-size prior construction, the sky-frame mask transfer and the
photometry CSV.  Anything that needs the emulator checkpoint, a GPU or the DJA
server is deliberately out of scope.

Run with::

    JAX_PLATFORMS=cpu python -m pytest tests/test_real_data_example.py -q -p no:cacheprovider
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import real_data_utils as rdu  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: a synthetic three-band cutout with a central galaxy, a neighbour,
# a clump riding on the galaxy and a plume running off one edge.
# ---------------------------------------------------------------------------


def _gaussian(shape, cy, cx, sigma, amp):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))


@pytest.fixture
def scene():
    """(flux, variance, mask) for a 61x61, 3-band synthetic scene."""
    shape = (61, 61)
    galaxy = _gaussian(shape, 30, 30, 4.0, 100.0)
    neighbour = _gaussian(shape, 10, 50, 2.0, 40.0)
    clump = _gaussian(shape, 36, 30, 1.2, 25.0)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    plume = 8.0 * np.exp(-((yy - 30) ** 2) / (2 * 3.0**2)) * (xx > 40) * (xx < 58)
    image = galaxy + neighbour + clump + plume
    rng = np.random.default_rng(0)
    flux = np.stack([image + rng.normal(0, 0.3, shape) for _ in range(3)]).astype(np.float32)
    variance = np.full_like(flux, 0.09)
    mask = np.ones_like(flux)
    return flux, variance, mask


# ---------------------------------------------------------------------------
# PSF resampling
# ---------------------------------------------------------------------------


class TestResamplePSF:
    """PSF resampling: normalisation, centroid, width and broadening."""

    def _gaussian_kernel(self, size=133, sigma_px=2.0):
        yy, xx = np.mgrid[0:size, 0:size]
        c = (size - 1) / 2.0
        k = np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * sigma_px**2))
        return k / k.sum()

    @pytest.mark.parametrize("method", ["spline", "area"])
    def test_normalised_and_odd(self, method):
        """Normalised and odd."""
        out = rdu.resample_psf(self._gaussian_kernel(), 0.03, 0.05, method=method)
        assert out.ndim == 2
        assert out.shape[0] % 2 == 1 and out.shape[1] % 2 == 1
        assert out.sum() == pytest.approx(1.0, rel=1e-5)

    @pytest.mark.parametrize("method", ["spline", "area"])
    def test_centroid_preserved(self, method):
        """Centroid preserved."""
        out = rdu.resample_psf(self._gaussian_kernel(), 0.03, 0.05, method=method)
        h, w = out.shape
        y = (np.arange(h) - (h - 1) / 2) * 0.05
        x = (np.arange(w) - (w - 1) / 2) * 0.05
        cy = float(out.sum(axis=1) @ y)
        cx = float(out.sum(axis=0) @ x)
        assert abs(cy) < 1e-6
        assert abs(cx) < 1e-6

    def test_spline_preserves_width_of_a_well_sampled_gaussian(self):
        """0.06" sigma is 2 px in, 1.2 px out; the second moment must survive."""
        out = rdu.resample_psf(self._gaussian_kernel(sigma_px=2.0), 0.03, 0.05)
        h, _ = out.shape
        y = (np.arange(h) - (h - 1) / 2) * 0.05
        sigma_out = float(np.sqrt(out.sum(axis=1) @ y**2))
        assert sigma_out == pytest.approx(0.06, rel=0.02)

    def test_area_rebin_conserves_flux_before_normalisation(self):
        """Area rebin conserves flux before normalisation."""
        kernel = self._gaussian_kernel()
        w_row = rdu._overlap_matrix(133, 81, 0.05 / 0.03)
        raw = w_row @ kernel @ w_row.T
        assert raw.sum() == pytest.approx(1.0, rel=1e-6)

    def test_broadening_widens_the_kernel(self):
        """Broadening widens the kernel."""
        narrow = rdu.resample_psf(self._gaussian_kernel(), 0.03, 0.05)
        wide = rdu.resample_psf(self._gaussian_kernel(), 0.03, 0.05, broaden_arcsec=0.05)
        h, _ = narrow.shape
        y = (np.arange(h) - (h - 1) / 2) * 0.05

        def sigma(k):
            return float(np.sqrt(k.sum(axis=1) @ y**2))

        assert sigma(wide) > sigma(narrow) * 1.3
        assert wide.sum() == pytest.approx(1.0, rel=1e-5)

    def test_unknown_method_raises(self):
        """Unknown method raises."""
        with pytest.raises(ValueError, match="unknown PSF resampling method"):
            rdu.resample_psf(self._gaussian_kernel(), 0.03, 0.05, method="nearest")


# ---------------------------------------------------------------------------
# Neighbour / clump / plume masking
# ---------------------------------------------------------------------------


class TestNeighbourMask:
    """Segmentation-based masking of neighbours, clumps, plumes and apertures."""

    def test_masks_the_neighbour_not_the_target(self, scene):
        """Masks the neighbour not the target."""
        flux, variance, mask = scene
        new_mask, segm, info = rdu.neighbour_mask(
            flux, variance, mask, nsigma=2.0, npixels=5, dilate=2, pixel_scale=0.05
        )
        assert new_mask[0, 30, 30] == 1.0  # target core kept
        assert new_mask[0, 10, 50] == 0.0  # neighbour masked
        assert 0.0 < info["frac_masked"] < 0.5
        assert info["n_segments"] >= 2
        assert segm[30, 30] == info["target_label"]

    def test_mask_is_identical_in_every_band(self, scene):
        """Mask is identical in every band."""
        flux, variance, mask = scene
        new_mask, _, _ = rdu.neighbour_mask(
            flux, variance, mask, nsigma=2.0, npixels=5, pixel_scale=0.05
        )
        assert np.array_equal(new_mask[0], new_mask[1])
        assert np.array_equal(new_mask[0], new_mask[2])

    def test_clump_peel_masks_a_clump_on_the_galaxy(self, scene):
        """Clump peel masks a clump on the galaxy."""
        flux, variance, mask = scene
        plain, _, _ = rdu.neighbour_mask(
            flux, variance, mask, nsigma=2.0, npixels=5, pixel_scale=0.05, clump_nsigma=0.0
        )
        peeled, _, info = rdu.neighbour_mask(
            flux,
            variance,
            mask,
            nsigma=2.0,
            npixels=5,
            pixel_scale=0.05,
            clump_nsigma=3.0,
            clump_scale_arcsec=0.25,
        )
        assert plain[0, 36, 30] == 1.0
        assert peeled[0, 36, 30] == 0.0
        assert info["frac_clumps"] > 0
        assert peeled[0, 30, 30] == 1.0  # never the core

    def test_plume_radius_masks_the_outskirts_of_the_target_segment(self, scene):
        """Plume radius masks the outskirts of the target segment."""
        flux, variance, mask = scene
        masked, _, info = rdu.neighbour_mask(
            flux,
            variance,
            mask,
            nsigma=2.0,
            npixels=5,
            pixel_scale=0.05,
            plume_radius_arcsec=0.5,  # 10 px
        )
        assert masked[0, 30, 30] == 1.0
        assert masked[0, 30, 50] == 0.0  # the plume, 20 px out
        assert info["frac_plume"] > 0

    def test_aperture_masks_everything_outside(self, scene):
        """Aperture masks everything outside."""
        flux, variance, mask = scene
        masked, _, info = rdu.neighbour_mask(
            flux, variance, mask, nsigma=2.0, npixels=5, pixel_scale=0.05,
            aperture_radius_arcsec=0.5,
        )  # fmt: skip
        assert masked[0, 30, 30] == 1.0
        assert masked[0, 0, 0] == 0.0
        assert info["frac_aperture"] > 0
        kept = masked[0] > 0
        yy, xx = np.mgrid[0:61, 0:61]
        assert np.hypot(yy - 30, xx - 30)[kept].max() * 0.05 <= 0.5 + 1e-9

    def test_protect_radius_is_never_masked(self, scene):
        """Protect radius is never masked."""
        flux, variance, mask = scene
        masked, _, _ = rdu.neighbour_mask(
            flux, variance, mask, nsigma=2.0, npixels=5, pixel_scale=0.05,
            aperture_radius_arcsec=0.01, protect_radius_arcsec=0.15,
        )  # fmt: skip
        yy, xx = np.mgrid[0:61, 0:61]
        inside = np.hypot(yy - 30, xx - 30) * 0.05 <= 0.15
        assert (masked[0][inside] == 1.0).all()

    def test_blank_frame_returns_everything(self):
        """Blank frame returns everything."""
        rng = np.random.default_rng(1)
        flux = rng.normal(0, 0.3, (2, 41, 41)).astype(np.float32)
        variance = np.full_like(flux, 0.09)
        mask = np.ones_like(flux)
        new_mask, _, info = rdu.neighbour_mask(
            flux, variance, mask, nsigma=5.0, npixels=20, pixel_scale=0.05
        )
        assert info["frac_masked"] == 0.0
        assert (new_mask == 1.0).all()


# ---------------------------------------------------------------------------
# Detection stack and size guess
# ---------------------------------------------------------------------------


class TestDetectionStackAndSize:
    """The multi-band detection stack and the half-light size guess."""

    def test_detection_stack_is_matched_filter_significance(self):
        """Detection stack is matched filter significance."""
        flux = np.ones((4, 8, 8), dtype=np.float32)
        variance = np.ones_like(flux)
        mask = np.ones_like(flux)
        stack = rdu.detection_stack(flux, variance, mask)
        assert stack == pytest.approx(np.full((8, 8), 2.0))  # sqrt(4) * 1

    def test_masked_bands_do_not_contribute(self):
        """Masked bands do not contribute."""
        flux = np.stack([np.ones((6, 6)), 1e6 * np.ones((6, 6))]).astype(np.float32)
        variance = np.ones_like(flux)
        mask = np.stack([np.ones((6, 6)), np.zeros((6, 6))]).astype(np.float32)
        assert rdu.detection_stack(flux, variance, mask) == pytest.approx(np.ones((6, 6)))

    def test_half_light_radius_recovers_a_gaussian(self, scene):
        """Half light radius recovers a gaussian."""
        flux, variance, mask = scene
        r_half = rdu.half_light_radius_arcsec(flux, variance, mask, 0.05)
        # A circular Gaussian of sigma 4 px has r_half = 1.177 sigma = 4.7 px
        # = 0.235"; the neighbour and the plume are outside the default aperture.
        assert 0.15 < r_half < 0.35

    def test_half_light_radius_never_below_one_pixel(self):
        """Half light radius never below one pixel."""
        flux = np.zeros((2, 21, 21), dtype=np.float32)
        flux[:, 10, 10] = 100.0
        variance = np.ones_like(flux)
        mask = np.ones_like(flux)
        assert rdu.half_light_radius_arcsec(flux, variance, mask, 0.05) == pytest.approx(0.05)


class TestLogSizePrior:
    """Construction of the Gaussian log-size prior and its soft cap."""

    def test_centred_on_the_guess(self):
        """Centred on the guess."""
        mu, sd = rdu.log_size_prior_from_guess(0.25, sd=0.5)
        assert np.exp(mu) == pytest.approx(0.25)
        assert sd == pytest.approx(0.5)

    def test_cap_narrows_the_prior(self):
        """Cap narrows the prior."""
        mu, sd = rdu.log_size_prior_from_guess(0.25, sd=0.5, max_re_arcsec=0.5, n_sigma=3.0)
        assert np.exp(mu + 3 * sd) == pytest.approx(0.5)
        assert sd < 0.5

    def test_loose_cap_leaves_the_prior_alone(self):
        """Loose cap leaves the prior alone."""
        _, sd = rdu.log_size_prior_from_guess(0.25, sd=0.5, max_re_arcsec=100.0)
        assert sd == pytest.approx(0.5)

    def test_cap_below_the_guess_raises(self):
        """Cap below the guess raises."""
        with pytest.raises(ValueError, match="below the measured size"):
            rdu.log_size_prior_from_guess(0.6, sd=0.5, max_re_arcsec=0.4)


# ---------------------------------------------------------------------------
# Sky-frame mask transfer between band grids
# ---------------------------------------------------------------------------


@dataclass
class _Grid:
    affine: np.ndarray
    ref_pixel: tuple[float, float]
    shape: tuple[int, int]


class TestMirrorMaskToBand:
    """Sky-frame transfer of a mask between two band grids."""

    def test_identity_grid_is_a_no_op(self):
        """Identity grid is a no op."""
        grid = _Grid(np.array([[0.05, 0.0], [0.0, -0.05]]), (10.0, 10.0), (21, 21))
        bad = np.zeros((21, 21), dtype=bool)
        bad[3:6, 7:9] = True
        out = rdu.mirror_mask_to_band(bad, grid, grid)
        assert np.array_equal(out, bad)

    def test_finer_grid_doubles_the_masked_area(self):
        """Finer grid doubles the masked area."""
        coarse = _Grid(np.array([[0.04, 0.0], [0.0, -0.04]]), (20.0, 20.0), (41, 41))
        fine = _Grid(np.array([[0.02, 0.0], [0.0, -0.02]]), (40.0, 40.0), (81, 81))
        bad = np.zeros((41, 41), dtype=bool)
        bad[18:23, 18:23] = True  # 5x5 coarse pixels = 0.2" x 0.2"
        out = rdu.mirror_mask_to_band(bad, coarse, fine)
        assert out[40, 40]
        # 0.2" square at 0.02"/px is 10x10 fine pixels, +/- one row of rounding.
        assert 81 <= out.sum() <= 121

    def test_pixels_outside_the_reference_grid_stay_unmasked(self):
        """Pixels outside the reference grid stay unmasked."""
        small = _Grid(np.array([[0.05, 0.0], [0.0, -0.05]]), (5.0, 5.0), (11, 11))
        big = _Grid(np.array([[0.05, 0.0], [0.0, -0.05]]), (30.0, 30.0), (61, 61))
        bad = np.ones((11, 11), dtype=bool)
        out = rdu.mirror_mask_to_band(bad, small, big)
        assert out[30, 30]
        assert not out[0, 0]
        assert out.sum() == 121

    def test_east_west_flip_is_respected(self):
        """A grid with the opposite x handedness must have its mask mirrored."""
        north_up = _Grid(np.array([[0.05, 0.0], [0.0, -0.05]]), (10.0, 10.0), (21, 21))
        flipped = _Grid(np.array([[0.05, 0.0], [0.0, 0.05]]), (10.0, 10.0), (21, 21))
        bad = np.zeros((21, 21), dtype=bool)
        bad[10, 4] = True  # 6 columns left of the reference pixel
        out = rdu.mirror_mask_to_band(bad, north_up, flipped)
        assert out[10, 16]
        assert not out[10, 4]


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------


class TestPhotometryCSV:
    """The photometry comparison CSV writer."""

    def test_round_trip(self, tmp_path):
        """Round trip."""
        rows = [
            {"band": "F200W", "pivot_um": 1.99, "catalogue_nJy": 100.0, "model_total_p50": 110.0},
            {"band": "F444W", "pivot_um": 4.42, "catalogue_nJy": 200.0, "model_total_p50": 210.0},
        ]
        path = tmp_path / "phot.csv"
        rdu.write_photometry_csv(path, rows)
        with open(path) as handle:
            back = list(csv.DictReader(handle))
        assert [r["band"] for r in back] == ["F200W", "F444W"]
        assert float(back[0]["model_total_p50"]) == 110.0
        assert back[0]["ratio_total_over_cat"] == ""  # missing keys become empty


class TestShortBand:
    """Band-name shortening."""

    @pytest.mark.parametrize(
        ("full", "short"),
        [("JWST/NIRCam.F200W", "F200W"), ("F090W", "F090W"), ("HST/ACS_WFC.F814W", "F814W")],
    )
    def test_short_band(self, full, short):
        """Short band."""
        assert rdu.short_band(full) == short


class TestClippedStats:
    """Sigma-clipped median and standard deviation."""

    def test_outliers_are_rejected(self):
        """Outliers are rejected."""
        rng = np.random.default_rng(3)
        values = rng.normal(5.0, 1.0, 5000)
        values[:50] = 1000.0
        median, sd = rdu._clipped_stats(values)
        assert median == pytest.approx(5.0, abs=0.1)
        assert sd == pytest.approx(1.0, abs=0.15)
