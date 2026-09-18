"""Tests for arachne.spatial.profiles (Gaussian, Sersic and point-source profiles)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.spatial.profiles import (
    PROFILES,
    GaussianProfile,
    PointSourceProfile,
    Profile,
    SersicProfile,
    get_profile,
    log_render_on_grid,
    render_on_grid,
    sersic_b,
)

CENTRE = jnp.zeros(2, dtype=jnp.float32)
CENTRE_SIGMA = 4.0
LOG_SIZE_PRIOR = (0.0, 1.0)
RHO_SIGMA = 1.0
PRIOR_ARGS = (CENTRE, CENTRE_SIGMA, LOG_SIZE_PRIOR, RHO_SIGMA)


def _grid(extent: float, step: float):
    """Cell-centred square grid on [-extent, extent], returning (yy, xx, cell area)."""
    c = np.arange(-extent + step / 2, extent, step, dtype=np.float64)
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return jnp.asarray(yy, dtype=jnp.float32), jnp.asarray(xx, dtype=jnp.float32), step * step


def _raw(profile: Profile, **overrides) -> jnp.ndarray:
    """Build a shape block from named defaults, overriding by shape-parameter name."""
    defaults = {"mu_y": 0.0, "mu_x": 0.0, "log_sigma_y": 0.0, "log_sigma_x": 0.0}
    defaults.update({"atanh_rho": 0.0, "log_n": math.log(1.0)})
    defaults.update(overrides)
    return jnp.asarray([defaults[n] for n in profile.shape_param_names], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """PROFILES / get_profile."""

    @pytest.mark.parametrize("name", ["gaussian", "sersic", "point"])
    def test_get_profile_by_name(self, name):
        """Names resolve to instances whose ``name`` and ``n_shape`` agree."""
        p = get_profile(name)
        assert isinstance(p, PROFILES[name])
        assert p.name == name
        assert len(p.shape_param_names) == p.n_shape
        assert repr(p)

    def test_get_profile_passthrough(self):
        """An instance is returned unchanged, so tuned profiles survive."""
        p = SersicProfile(log_n_sd=0.3)
        assert get_profile(p) is p

    def test_get_profile_errors(self):
        """Unknown names raise ValueError, bad types TypeError."""
        with pytest.raises(ValueError):
            get_profile("moffat")
        with pytest.raises(TypeError):
            get_profile(3)

    def test_base_class_is_abstract(self):
        """The base class refuses to do any work."""
        with pytest.raises(NotImplementedError):
            Profile().parse(jnp.zeros(1))
        with pytest.raises(NotImplementedError):
            Profile().log_surface_brightness(jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))
        with pytest.raises(NotImplementedError):
            Profile().log_prior(jnp.zeros(1), *PRIOR_ARGS)
        with pytest.raises(NotImplementedError):
            Profile().sample_prior(jax.random.PRNGKey(0), *PRIOR_ARGS)
        with pytest.raises(NotImplementedError):
            Profile().size_statistic(jnp.zeros(1))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestSersicB:
    """Ciotti & Bertin b_n expansion."""

    @pytest.mark.parametrize("n,expected", [(1.0, 1.678), (4.0, 7.669), (0.5, 0.6934)])
    def test_known_values(self, n, expected):
        """b_1 = 1.678, b_4 = 7.669 to 1e-3."""
        assert abs(float(sersic_b(n)) - expected) < 1e-3

    def test_half_light_definition(self):
        """b_n really halves the enclosed flux at r = 1 (gamma(2n, b_n) = Gamma(2n)/2)."""
        from scipy.special import gammainc

        for n in [0.5, 1.0, 2.0, 4.0]:
            assert abs(float(gammainc(2 * n, float(sersic_b(n)))) - 0.5) < 2e-4


class TestNormalisation:
    """surface_brightness integrates to 1 over the plane."""

    def test_gaussian(self):
        """A correlated Gaussian integrates to 1 on a fine grid."""
        p = GaussianProfile()
        yy, xx, area = _grid(12.0, 0.02)
        total = float(jnp.sum(p.surface_brightness(_raw(p, atanh_rho=0.3), yy, xx))) * area
        np.testing.assert_allclose(total, 1.0, rtol=0.02)

    @pytest.mark.parametrize("n", [0.7, 1.0, 2.0])
    def test_sersic(self, n):
        """Sersic n = 0.7, 1, 2 integrate to 1 on a fine grid."""
        p = SersicProfile()
        yy, xx, area = _grid(15.0, 0.01)
        raw = _raw(p, atanh_rho=0.3, log_n=math.log(n))
        total = float(jnp.sum(p.surface_brightness(raw, yy, xx))) * area
        np.testing.assert_allclose(total, 1.0, rtol=0.02)

    def test_sersic_n4_needs_a_big_grid(self):
        """Sersic n = 4 has huge wings: a +-25 Re grid at 0.025 Re holds ~99% of the flux."""
        p = SersicProfile()
        yy, xx, area = _grid(25.0, 0.025)
        raw = _raw(p, log_n=math.log(4.0))
        total = float(jnp.sum(p.surface_brightness(raw, yy, xx))) * area
        np.testing.assert_allclose(total, 1.0, rtol=0.02)

    def test_point_source(self):
        """The point source carries unit flux."""
        p = PointSourceProfile()
        yy, xx, area = _grid(6.0, 0.01)
        total = float(jnp.sum(p.surface_brightness(jnp.zeros(2), yy, xx))) * area
        np.testing.assert_allclose(total, 1.0, rtol=0.02)

    def test_sersic_half_is_a_gaussian(self):
        """Sersic n = 0.5 equals a Gaussian of sigma / sqrt(2 b_n) with the same rho."""
        s, g = SersicProfile(), GaussianProfile()
        raw_s = jnp.asarray([0.3, -0.2, 0.2, -0.1, 0.4, math.log(0.5)], dtype=jnp.float32)
        parsed = s.parse(raw_s)
        assert abs(float(parsed["n"]) - 0.5) < 1e-4  # smooth clamp is inactive here
        shift = 0.5 * math.log(2.0 * float(parsed["b_n"]))
        raw_g = jnp.asarray([0.3, -0.2, 0.2 - shift, -0.1 - shift, 0.4], dtype=jnp.float32)
        c = np.linspace(-4.0, 4.0, 17)
        yy, xx = (jnp.asarray(a, dtype=jnp.float32) for a in np.meshgrid(c, c, indexing="ij"))
        np.testing.assert_allclose(
            s.surface_brightness(raw_s, yy, xx), g.surface_brightness(raw_g, yy, xx), rtol=1e-4
        )

    def test_sersic_n_smoothly_clamped(self):
        """The index is confined to n_bounds with a non-zero gradient at the wall."""
        s = SersicProfile()
        for log_n, lo, hi in [(-8.0, 0.29, 0.35), (8.0, 8.5, 10.0)]:
            n = float(s.parse(_raw(s, log_n=log_n))["n"])
            assert lo <= n <= hi
        # at the wall itself the softplus clamp still passes half the gradient
        grad = jax.grad(lambda r: s.parse(r)["n"])(_raw(s, log_n=math.log(0.3)))
        assert np.isfinite(float(grad[5])) and float(grad[5]) > 0.0


# ---------------------------------------------------------------------------
# Rendering / oversampling
# ---------------------------------------------------------------------------


class TestRendering:
    """render_on_grid and oversampling."""

    def test_oversample_one_is_plain_evaluation(self):
        """oversample=1 is surface_brightness * pixel_area."""
        p = GaussianProfile()
        raw = _raw(p, log_sigma_y=math.log(2.0), log_sigma_x=math.log(3.0), atanh_rho=0.2)
        c = np.arange(-6.0, 6.0)
        yy, xx = (jnp.asarray(a, dtype=jnp.float32) for a in np.meshgrid(c, c, indexing="ij"))
        np.testing.assert_allclose(
            render_on_grid(p, raw, yy, xx, 0.25, 1),
            p.surface_brightness(raw, yy, xx) * 0.25,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            log_render_on_grid(p, raw, yy, xx, 0.25, 1),
            jnp.log(render_on_grid(p, raw, yy, xx, 0.25, 1)),
            rtol=1e-5,
        )

    def test_bad_oversample(self):
        """An oversample below 1 raises."""
        with pytest.raises(ValueError):
            render_on_grid(GaussianProfile(), jnp.zeros(5), jnp.zeros(3), jnp.zeros(3), 1.0, 0)

    def test_oversampling_fixes_the_cusp(self):
        """For n=4 oversampling changes the central pixel a lot and the wings not at all."""
        s = SersicProfile()
        raw = _raw(s, log_sigma_y=math.log(3.0), log_sigma_x=math.log(3.0), log_n=math.log(4.0))
        c = np.arange(-60.0, 61.0)
        yy, xx = (jnp.asarray(a, dtype=jnp.float32) for a in np.meshgrid(c, c, indexing="ij"))
        r1 = np.asarray(render_on_grid(s, raw, yy, xx, 1.0, 1))
        r9 = np.asarray(render_on_grid(s, raw, yy, xx, 1.0, 9))
        core = np.asarray(np.meshgrid(c, c, indexing="ij"))
        outer = (core[0] ** 2 + core[1] ** 2) > (2 * 3.0) ** 2
        # the central pixel of a single-sampled cusp is wrong by a large factor
        assert r1.max() > 5.0 * r9.max()
        # ... while the resolved wings are untouched
        np.testing.assert_allclose(r1[outer].sum(), r9[outer].sum(), rtol=5e-3)
        # ... and the oversampled total recovers the analytic unit flux
        np.testing.assert_allclose(r9.sum(), 1.0, rtol=0.03)

    def test_oversampling_is_a_noop_for_smooth_profiles(self):
        """A well-resolved Gaussian is insensitive to oversampling."""
        g = GaussianProfile()
        raw = _raw(g, log_sigma_y=math.log(3.0), log_sigma_x=math.log(3.0))
        c = np.arange(-20.0, 21.0)
        yy, xx = (jnp.asarray(a, dtype=jnp.float32) for a in np.meshgrid(c, c, indexing="ij"))
        r1 = render_on_grid(g, raw, yy, xx, 1.0, 1)
        r5 = render_on_grid(g, raw, yy, xx, 1.0, 5)
        np.testing.assert_allclose(float(r1.sum()), float(r5.sum()), rtol=2e-3)
        np.testing.assert_allclose(float(r1.sum()), 1.0, rtol=2e-3)

    def test_point_source_is_contained(self):
        """A point source deposits essentially all its flux within 2 px of mu."""
        p = PointSourceProfile()
        raw = jnp.asarray([0.3, -0.4], dtype=jnp.float32)
        c = np.arange(-8.0, 9.0)
        yy, xx = np.meshgrid(c, c, indexing="ij")
        flux = np.asarray(
            render_on_grid(
                p,
                raw,
                jnp.asarray(yy, dtype=jnp.float32),
                jnp.asarray(xx, dtype=jnp.float32),
                1.0,
                3,
            )
        )
        near = (np.abs(yy - 0.3) <= 2.0) & (np.abs(xx + 0.4) <= 2.0)
        np.testing.assert_allclose(flux.sum(), 1.0, rtol=1e-3)
        assert flux[near].sum() > 0.999 * flux.sum()

    def test_off_grid_component_is_finite_in_log_space(self):
        """A profile centred far outside the grid gives finite logs, not NaNs."""
        for p, raw in [
            (GaussianProfile(), jnp.asarray([500.0, -300.0, 0.0, 0.0, 0.0])),
            (SersicProfile(), jnp.asarray([500.0, -300.0, 0.0, 0.0, 0.0, 0.0])),
        ]:
            c = np.arange(-4.0, 5.0)
            yy, xx = (jnp.asarray(a, dtype=jnp.float32) for a in np.meshgrid(c, c, indexing="ij"))
            logs = log_render_on_grid(p, raw, yy, xx, 1.0, 2)
            assert jnp.all(jnp.isfinite(logs))

    def test_jit_grad_vmap(self):
        """Rendering is jit / grad / vmap safe for every profile."""
        c = np.arange(-5.0, 6.0)
        yy, xx = (jnp.asarray(a, dtype=jnp.float32) for a in np.meshgrid(c, c, indexing="ij"))
        for p in [GaussianProfile(), SersicProfile(), PointSourceProfile()]:
            raw = _raw(p, log_n=math.log(2.5))
            f = jax.jit(lambda r, p=p: render_on_grid(p, r, yy, xx, 1.0, 2).sum())
            assert jnp.isfinite(f(raw))
            g = jax.grad(f)(raw)
            assert g.shape == (p.n_shape,) and jnp.all(jnp.isfinite(g))
            batch = jax.vmap(f)(jnp.stack([raw, raw + 0.05]))
            assert batch.shape == (2,) and jnp.all(jnp.isfinite(batch))


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


class TestPriors:
    """log_prior and sample_prior."""

    @pytest.mark.parametrize("name", ["gaussian", "sersic", "point"])
    def test_log_prior_finite_and_grad_safe(self, name):
        """Finite scalar with a finite gradient."""
        p = get_profile(name)
        raw = _raw(p)
        lp = p.log_prior(raw, *PRIOR_ARGS)
        assert lp.shape == () and jnp.isfinite(lp)
        g = jax.grad(lambda r: p.log_prior(r, *PRIOR_ARGS))(raw)
        assert g.shape == (p.n_shape,) and jnp.all(jnp.isfinite(g))

    def test_gaussian_log_prior_is_normalised(self):
        """At the prior mode the density equals the analytic normalising constants."""
        g = GaussianProfile()
        raw = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        expected = -2 * (np.log(CENTRE_SIGMA) + 0.5 * np.log(2 * np.pi))
        expected += -2 * (np.log(LOG_SIZE_PRIOR[1]) + 0.5 * np.log(2 * np.pi))
        expected += -(np.log(RHO_SIGMA) + 0.5 * np.log(2 * np.pi))
        np.testing.assert_allclose(float(g.log_prior(raw, *PRIOR_ARGS)), expected, rtol=1e-5)

    def test_sersic_adds_only_the_log_n_term(self):
        """Sersic log_prior = Gaussian log_prior + Normal(log 2, 0.7) on log_n."""
        s, g = SersicProfile(), GaussianProfile()
        raw = jnp.asarray([0.5, -0.5, 0.2, 0.1, 0.3, math.log(3.0)], dtype=jnp.float32)
        extra = float(s.log_prior(raw, *PRIOR_ARGS) - g.log_prior(raw[:5], *PRIOR_ARGS))
        z = (math.log(3.0) - s.log_n_mu) / s.log_n_sd
        expected = -0.5 * z**2 - math.log(s.log_n_sd) - 0.5 * math.log(2 * math.pi)
        np.testing.assert_allclose(extra, expected, rtol=1e-5)

    def test_point_prior_ignores_size_and_rho(self):
        """Changing the size / rho prior settings does not move a point source's prior."""
        p = PointSourceProfile()
        raw = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
        a = float(p.log_prior(raw, CENTRE, CENTRE_SIGMA, (0.0, 1.0), 1.0))
        b = float(p.log_prior(raw, CENTRE, CENTRE_SIGMA, (3.0, 0.2), 5.0))
        np.testing.assert_allclose(a, b, rtol=1e-6)

    @pytest.mark.parametrize("name", ["gaussian", "sersic", "point"])
    def test_sample_prior_shapes(self, name):
        """Single draws are (n_shape,) and batches are (n, n_shape)."""
        p = get_profile(name)
        single = p.sample_prior(jax.random.PRNGKey(0), *PRIOR_ARGS)
        assert single.shape == (p.n_shape,)
        batch = p.sample_prior(jax.random.PRNGKey(0), *PRIOR_ARGS, batch_shape=(7,))
        assert batch.shape == (7, p.n_shape)
        assert jnp.all(jnp.isfinite(batch))

    def test_sample_prior_matches_log_prior_hyperparameters(self):
        """Sample moments reproduce the Gaussian hyper-parameters used by log_prior."""
        s = SersicProfile()
        x = np.asarray(
            s.sample_prior(jax.random.PRNGKey(3), *PRIOR_ARGS, batch_shape=(40000,)),
            dtype=np.float64,
        )
        np.testing.assert_allclose(x[:, 0:2].mean(0), 0.0, atol=0.06)
        np.testing.assert_allclose(x[:, 0:2].std(0), CENTRE_SIGMA, rtol=0.03)
        np.testing.assert_allclose(x[:, 2:4].mean(0), LOG_SIZE_PRIOR[0], atol=0.02)
        np.testing.assert_allclose(x[:, 2:4].std(0), LOG_SIZE_PRIOR[1], rtol=0.03)
        np.testing.assert_allclose(x[:, 4].std(), RHO_SIGMA, rtol=0.03)
        np.testing.assert_allclose(x[:, 5].mean(), s.log_n_mu, atol=0.02)
        np.testing.assert_allclose(x[:, 5].std(), s.log_n_sd, rtol=0.03)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


class TestSummaries:
    """parse, centre and size_statistic."""

    def test_parse_common_keys(self):
        """Every profile exposes mu, sigma and rho."""
        for p in [GaussianProfile(), SersicProfile(), PointSourceProfile()]:
            d = p.parse(_raw(p, log_sigma_y=math.log(2.0), atanh_rho=0.5))
            assert d["mu"].shape == (2,) and d["sigma"].shape == (2,)
            assert jnp.ndim(d["rho"]) == 0
        assert "n" in SersicProfile().parse(_raw(SersicProfile()))

    def test_point_source_sigma_and_rho_are_fixed(self):
        """A point source reports its fixed width and zero ellipticity."""
        p = PointSourceProfile(point_sigma=0.4)
        d = p.parse(jnp.asarray([1.0, 2.0], dtype=jnp.float32))
        np.testing.assert_allclose(d["sigma"], [0.4, 0.4])
        np.testing.assert_allclose(float(d["rho"]), 0.0)
        np.testing.assert_allclose(float(p.size_statistic(jnp.zeros(2))), math.log(0.4))
        with pytest.raises(ValueError):
            PointSourceProfile(point_sigma=0.0)

    def test_size_statistic_and_centre(self):
        """size_statistic is the log geometric-mean size; centre reads mu."""
        for p in [GaussianProfile(), SersicProfile()]:
            raw = _raw(p, mu_y=3.0, mu_x=-1.0, log_sigma_y=math.log(2.0), log_sigma_x=math.log(8.0))
            np.testing.assert_allclose(float(p.size_statistic(raw)), math.log(4.0), rtol=1e-6)
            np.testing.assert_allclose(p.centre(raw), [3.0, -1.0])
        # a point source is smaller than any resolved component
        assert float(PointSourceProfile().size_statistic(jnp.zeros(2))) < float(
            GaussianProfile().size_statistic(_raw(GaussianProfile()))
        )
