"""Tests for arachne.priors.specs (dict-based per-parameter prior specifications)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.priors.specs import (
    DEFAULT_PRIORS,
    SUPPORTED_DISTS,
    build_component_log_prior,
    build_log_prior,
    log_prior_1d,
    prior_config_template,
    resolve_prior_specs,
    sigmoid_log_jacobian,
    validate_prior_spec,
)

# A spec of every supported dist, each on a domain where it is valid.
_EVERY_DIST = {
    "uniform": ({"dist": "uniform"}, (-1.0, 1.0)),
    "loguniform": ({"dist": "loguniform"}, (0.1, 10.0)),
    "normal": ({"dist": "normal", "loc": 0.2, "scale": 0.5}, (-1.0, 1.0)),
    "studentt": ({"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3}, (-10.0, 10.0)),
    "halfnormal": ({"dist": "halfnormal", "scale": 1.0}, (0.0, 5.0)),
    "exponential": ({"dist": "exponential", "scale": 0.5}, (0.0, 5.0)),
    "lognormal": ({"dist": "lognormal", "loc": 0.0, "scale": 0.5}, (0.1, 10.0)),
}

NAMES = ["a", "b", "c"]
BOUNDS = {"a": (0.1, 10.0), "b": (-10.0, 10.0), "c": (0.0, 5.0)}
SPECS = {
    "a": {"dist": "loguniform"},
    "b": {"dist": "studentt", "df": 2.0, "loc": 0.0, "scale": 0.3},
    "c": {"dist": "exponential", "scale": 0.5},
}


class TestLogPrior1d:
    """Per-distribution scalar log-densities."""

    def test_every_supported_dist_covered(self):
        """The test table covers exactly SUPPORTED_DISTS."""
        assert set(_EVERY_DIST) == set(SUPPORTED_DISTS)

    @pytest.mark.parametrize("dist", sorted(SUPPORTED_DISTS))
    def test_finite_inside_bounds(self, dist):
        """Each dist is finite (and grad-finite) everywhere strictly inside its domain."""
        spec, (lo, hi) = _EVERY_DIST[dist]
        xs = jnp.linspace(lo + 1e-3, hi - 1e-3, 25)
        vals = jax.vmap(lambda x: log_prior_1d(spec, x, lo, hi))(xs)
        grads = jax.vmap(jax.grad(lambda x: log_prior_1d(spec, x, lo, hi)))(xs)
        assert vals.shape == (25,)
        assert bool(jnp.all(jnp.isfinite(vals)))
        assert bool(jnp.all(jnp.isfinite(grads)))

    def test_uniform_is_zero(self):
        """Uniform contributes exactly zero."""
        assert float(log_prior_1d({"dist": "uniform"}, jnp.float32(0.3), -1.0, 1.0)) == 0.0

    def test_missing_dist_means_uniform(self):
        """A spec without a 'dist' key is treated as uniform."""
        assert float(log_prior_1d({}, jnp.float32(0.3), -1.0, 1.0)) == 0.0

    def test_dist_is_case_insensitive(self):
        """'Normal' and 'normal' evaluate identically."""
        a = log_prior_1d({"dist": "Normal", "scale": 0.5}, jnp.float32(0.2), -1.0, 1.0)
        b = log_prior_1d({"dist": "normal", "scale": 0.5}, jnp.float32(0.2), -1.0, 1.0)
        assert float(a) == float(b)

    def test_normal_matches_closed_form(self):
        """Normal includes its full normalising constant."""
        loc, scale, x = 0.2, 0.5, 0.7
        expected = -0.5 * math.log(2 * math.pi) - math.log(scale) - 0.5 * ((x - loc) / scale) ** 2
        got = log_prior_1d({"dist": "normal", "loc": loc, "scale": scale}, jnp.float32(x), -1, 1)
        assert float(got) == pytest.approx(expected, rel=1e-6)

    def test_studentt_matches_closed_form(self):
        """Student-t includes its untruncated normalising constant."""
        df, loc, scale, x = 2.0, 0.0, 0.3, 0.45
        c = (
            math.lgamma(0.5 * (df + 1))
            - math.lgamma(0.5 * df)
            - 0.5 * math.log(df * math.pi)
            - math.log(scale)
        )
        expected = c - 0.5 * (df + 1) * math.log1p(((x - loc) / scale) ** 2 / df)
        spec = {"dist": "studentt", "df": df, "loc": loc, "scale": scale}
        got = log_prior_1d(spec, jnp.float32(x), -10, 10)
        assert float(got) == pytest.approx(expected, rel=1e-6)

    def test_exponential_is_unnormalised_shape(self):
        """Exponential is -x/scale with no constant (historical contract)."""
        got = log_prior_1d({"dist": "exponential", "scale": 0.5}, jnp.float32(1.0), 0.0, 5.0)
        assert float(got) == pytest.approx(-2.0)

    def test_unhandled_dist_raises(self):
        """An unsupported dist raises at evaluation time."""
        with pytest.raises(ValueError, match="unhandled dist"):
            log_prior_1d({"dist": "cauchy"}, jnp.float32(0.0), -1.0, 1.0)


class TestValidatePriorSpec:
    """validate_prior_spec raises on malformed specs."""

    @pytest.mark.parametrize("dist", sorted(SUPPORTED_DISTS))
    def test_valid_specs_pass(self, dist):
        """Every entry of the reference table validates."""
        spec, bounds = _EVERY_DIST[dist]
        validate_prior_spec(dist, spec, bounds)

    def test_unknown_dist(self):
        """Unknown dist names are rejected."""
        with pytest.raises(ValueError, match="unknown prior dist"):
            validate_prior_spec("p", {"dist": "cauchy"}, (0.0, 1.0))

    @pytest.mark.parametrize(
        "dist", ["normal", "studentt", "halfnormal", "exponential", "lognormal"]
    )
    @pytest.mark.parametrize("scale", [0.0, -1.0])
    def test_nonpositive_scale(self, dist, scale):
        """Reject scale <= 0 for every scale-bearing dist."""
        with pytest.raises(ValueError, match="scale > 0"):
            validate_prior_spec("p", {"dist": dist, "df": 2.0, "scale": scale}, (0.1, 1.0))

    def test_missing_scale(self):
        """A missing scale counts as 0 and is rejected."""
        with pytest.raises(ValueError, match="scale > 0"):
            validate_prior_spec("p", {"dist": "normal"}, (0.0, 1.0))

    @pytest.mark.parametrize("df", [0.0, -2.0])
    def test_nonpositive_df(self, df):
        """Student-t requires df > 0."""
        with pytest.raises(ValueError, match="df > 0"):
            validate_prior_spec("p", {"dist": "studentt", "df": df, "scale": 1.0}, (-1.0, 1.0))

    @pytest.mark.parametrize("dist", ["loguniform", "lognormal"])
    @pytest.mark.parametrize("lo", [0.0, -1.0])
    def test_log_dists_need_positive_lower_bound(self, dist, lo):
        """Reject loguniform / lognormal when lo <= 0."""
        with pytest.raises(ValueError, match="lower bound > 0"):
            validate_prior_spec("p", {"dist": dist, "scale": 1.0}, (lo, 1.0))

    def test_halfnormal_negative_domain_only_warns(self, capsys):
        """Half-normal on a domain with lo < 0 prints a warning but does not raise."""
        validate_prior_spec("p", {"dist": "halfnormal", "scale": 1.0}, (-1.0, 1.0))
        assert "shape may be unintended" in capsys.readouterr().out


class TestResolvePriorSpecs:
    """Merging user overrides over defaults."""

    def test_defaults_for_known_params(self):
        """Without overrides, every SPS parameter gets its DEFAULT_PRIORS entry."""
        names = list(DEFAULT_PRIORS)
        bounds = {p: (0.001, 10.0) for p in names}
        out = resolve_prior_specs(names, None, bounds)
        assert out == DEFAULT_PRIORS
        assert list(out) == names
        # copies, not the shared default dicts
        assert all(out[p] is not DEFAULT_PRIORS[p] for p in names)

    def test_unknown_param_defaults_to_uniform(self):
        """Parameters absent from DEFAULT_PRIORS resolve to uniform."""
        out = resolve_prior_specs(["not_a_default"], {}, {"not_a_default": (0.0, 1.0)})
        assert out == {"not_a_default": {"dist": "uniform"}}

    def test_override_replaces_default(self):
        """A user spec fully replaces the default for that parameter."""
        user = {"logsfr_ratio_0": {"dist": "normal", "loc": 0.0, "scale": 1.0}}
        bounds = {"logsfr_ratio_0": (-10.0, 10.0), "Av": (0.001, 5.0)}
        out = resolve_prior_specs(["logsfr_ratio_0", "Av"], user, bounds)
        assert out["logsfr_ratio_0"] == user["logsfr_ratio_0"]
        assert out["logsfr_ratio_0"] is not user["logsfr_ratio_0"]
        assert out["Av"] == {"dist": "uniform"}

    def test_override_for_unknown_param_raises(self):
        """Overrides naming a parameter outside param_names are an error."""
        with pytest.raises(ValueError, match="unknown parameter"):
            resolve_prior_specs(["Av"], {"typo": {"dist": "uniform"}}, {"Av": (0.001, 5.0)})

    def test_invalid_override_raises(self):
        """Overrides are validated against the parameter bounds."""
        with pytest.raises(ValueError, match="lower bound > 0"):
            resolve_prior_specs(["b"], {"b": {"dist": "loguniform"}}, {"b": (-1.0, 1.0)})

    def test_empty_and_none_overrides_equivalent(self):
        """None, {} and a falsy value all mean 'no overrides'."""
        bounds = {"Av": (0.001, 5.0)}
        assert resolve_prior_specs(["Av"], None, bounds) == resolve_prior_specs(["Av"], {}, bounds)

    def test_prior_config_template(self):
        """Template returns defaults (uniform for unregistered names), validated."""
        names = ["Av", "logsfr_ratio_0", "extra"]
        bounds = {"Av": (0.001, 5.0), "logsfr_ratio_0": (-10.0, 10.0), "extra": (0.0, 1.0)}
        tpl = prior_config_template(names, bounds)
        assert tpl == {
            "Av": {"dist": "uniform"},
            "logsfr_ratio_0": DEFAULT_PRIORS["logsfr_ratio_0"],
            "extra": {"dist": "uniform"},
        }
        assert prior_config_template(names) == tpl


class TestBuildLogPrior:
    """Vector log-prior over a physical parameter vector."""

    def test_uniform_only_is_zero_and_grad_safe(self):
        """All-uniform specs give exactly 0 with zero gradient, under jit."""
        specs = {p: {"dist": "uniform"} for p in NAMES}
        fn = build_log_prior(NAMES, specs, BOUNDS)
        x = jnp.array([1.0, 0.5, 2.0], dtype=jnp.float32)
        assert float(jax.jit(fn)(x)) == 0.0
        g = jax.jit(jax.grad(fn))(x)
        assert g.shape == x.shape
        assert bool(jnp.all(g == 0.0))
        assert jax.jit(fn)(x).dtype == x.dtype

    def test_matches_sum_of_1d(self):
        """build_log_prior equals the sum of log_prior_1d over parameters."""
        fn = build_log_prior(NAMES, SPECS, BOUNDS)
        x = jnp.array([1.7, -0.4, 0.9], dtype=jnp.float32)
        expected = sum(float(log_prior_1d(SPECS[p], x[i], *BOUNDS[p])) for i, p in enumerate(NAMES))
        assert float(fn(x)) == pytest.approx(expected, rel=1e-6)

    def test_jit_grad_vmap(self):
        """The built function composes with jit, grad and vmap and stays finite."""
        fn = build_log_prior(NAMES, SPECS, BOUNDS)
        X = jnp.array([[1.0, 0.1, 0.3], [5.0, -2.0, 4.0]], dtype=jnp.float32)
        vals = jax.jit(jax.vmap(fn))(X)
        grads = jax.jit(jax.vmap(jax.grad(fn)))(X)
        assert vals.shape == (2,)
        assert grads.shape == X.shape
        assert bool(jnp.all(jnp.isfinite(vals))) and bool(jnp.all(jnp.isfinite(grads)))


class TestSigmoidLogJacobian:
    """Log-Jacobian of the bounded sigmoid transform."""

    def test_matches_autodiff_derivative(self):
        """Autodiff sum log|dx/draw| equals sigmoid_log_jacobian."""
        lows = jnp.array([0.001, -10.0, 4.0], dtype=jnp.float32)
        highs = jnp.array([5.0, 10.0, 12.0], dtype=jnp.float32)
        raw = jnp.array([-2.3, 0.4, 3.1], dtype=jnp.float32)

        def transform(r):
            return lows + (highs - lows) * jax.nn.sigmoid(r)

        jac = jax.jacfwd(transform)(raw)  # diagonal
        expected = float(jnp.sum(jnp.log(jnp.abs(jnp.diag(jac)))))
        assert float(sigmoid_log_jacobian(raw, lows, highs)) == pytest.approx(expected, rel=1e-5)

    def test_matches_finite_difference(self):
        """Central finite differences of the transform agree with the closed form."""
        lo, hi, r, h = 0.5, 3.5, 0.7, 1e-6
        x = lambda t: lo + (hi - lo) / (1.0 + math.exp(-t))  # noqa: E731
        num = math.log(abs((x(r + h) - x(r - h)) / (2 * h)))
        got = float(sigmoid_log_jacobian(jnp.float32(r), jnp.float32(lo), jnp.float32(hi)))
        assert got == pytest.approx(num, rel=1e-5)

    def test_closed_form_and_jit(self):
        """Equals sum(log(hi-lo) + log_sigmoid(raw) + log_sigmoid(-raw)) and jits."""
        lows = jnp.array([0.0, 1.0], dtype=jnp.float32)
        highs = jnp.array([1.0, 3.0], dtype=jnp.float32)
        raw = jnp.array([0.0, 2.0], dtype=jnp.float32)
        expected = jnp.sum(
            jnp.log(highs - lows) + jax.nn.log_sigmoid(raw) + jax.nn.log_sigmoid(-raw)
        )
        assert float(jax.jit(sigmoid_log_jacobian)(raw, lows, highs)) == pytest.approx(
            float(expected)
        )


class TestBuildComponentLogPrior:
    """Multi-component (K, N) log-prior with shared / fixed parameters."""

    def test_shared_counted_once(self):
        """Shared parameter is counted once (row 0); per-component params summed over rows."""
        fn = build_component_log_prior(NAMES, SPECS, BOUNDS, shared_param_names=["a"])
        X = jnp.array([[1.5, -0.3, 0.7], [1.5, 2.2, 3.1]], dtype=jnp.float32)
        one = lambda p, v: float(log_prior_1d(SPECS[p], v, *BOUNDS[p]))  # noqa: E731
        expected = (
            one("a", X[0, 0])
            + one("b", X[0, 1])
            + one("b", X[1, 1])
            + one("c", X[0, 2])
            + one("c", X[1, 2])
        )
        assert float(fn(X)) == pytest.approx(expected, rel=1e-6)

    def test_no_shared_is_sum_of_rows(self):
        """Without shared/fixed names it equals the sum of build_log_prior over rows."""
        fn = build_component_log_prior(NAMES, SPECS, BOUNDS)
        row_fn = build_log_prior(NAMES, SPECS, BOUNDS)
        X = jnp.array([[1.5, -0.3, 0.7], [4.0, 2.2, 3.1], [0.2, 0.0, 1.0]], dtype=jnp.float32)
        expected = sum(float(row_fn(X[k])) for k in range(3))
        assert float(fn(X)) == pytest.approx(expected, rel=1e-6)

    def test_fixed_contributes_nothing(self):
        """Fixed parameters add no prior, even when their spec would be non-zero."""
        fn = build_component_log_prior(NAMES, SPECS, BOUNDS, fixed_param_names=["c"])
        fn_ref = build_component_log_prior(["a", "b"], SPECS, BOUNDS)
        X = jnp.array([[1.5, -0.3, 0.7], [4.0, 2.2, 3.1]], dtype=jnp.float32)
        assert float(fn(X)) == pytest.approx(float(fn_ref(X[:, :2])), rel=1e-6)
        # fixed column need not have a spec / bounds at all
        fn2 = build_component_log_prior(
            NAMES,
            {k: SPECS[k] for k in ("a", "b")},
            {k: BOUNDS[k] for k in ("a", "b")},
            fixed_param_names=["c"],
        )
        assert float(fn2(X)) == pytest.approx(float(fn(X)))

    def test_all_shared_single_row_equivalent(self):
        """All parameters shared: independent of K, equals build_log_prior on row 0."""
        fn = build_component_log_prior(NAMES, SPECS, BOUNDS, shared_param_names=NAMES)
        row_fn = build_log_prior(NAMES, SPECS, BOUNDS)
        row = jnp.array([1.5, -0.3, 0.7], dtype=jnp.float32)
        X = jnp.stack([row, row, row])
        assert float(fn(X)) == pytest.approx(float(row_fn(row)), rel=1e-6)

    def test_jit_grad_vmap_safe(self):
        """Composes with jit, grad and an outer vmap over a batch of matrices."""
        fn = build_component_log_prior(NAMES, SPECS, BOUNDS, shared_param_names=["a"])
        XB = jnp.array(
            [[[1.5, -0.3, 0.7], [1.5, 2.2, 3.1]], [[0.4, 0.0, 0.1], [0.4, -1.0, 2.0]]],
            dtype=jnp.float32,
        )
        vals = jax.jit(jax.vmap(fn))(XB)
        grads = jax.jit(jax.vmap(jax.grad(fn)))(XB)
        assert vals.shape == (2,)
        assert grads.shape == XB.shape
        assert bool(jnp.all(jnp.isfinite(vals))) and bool(jnp.all(jnp.isfinite(grads)))
        # shared column gradient only flows through row 0
        assert bool(jnp.all(grads[:, 1, 0] == 0.0))
        assert bool(jnp.all(grads[:, 0, 0] != 0.0))

    def test_uniform_only_zero(self):
        """All-uniform specs give exactly 0 for any K."""
        specs = {p: {"dist": "uniform"} for p in NAMES}
        fn = build_component_log_prior(NAMES, specs, BOUNDS, shared_param_names=["b"])
        X = jnp.ones((4, 3), dtype=jnp.float32)
        assert float(fn(X)) == 0.0

    def test_unknown_names_raise(self):
        """Shared / fixed names must be emulator parameters and must not overlap."""
        with pytest.raises(ValueError, match="shared_param_names"):
            build_component_log_prior(NAMES, SPECS, BOUNDS, shared_param_names=["zz"])
        with pytest.raises(ValueError, match="fixed_param_names"):
            build_component_log_prior(NAMES, SPECS, BOUNDS, fixed_param_names=["zz"])
        with pytest.raises(ValueError, match="both shared and fixed"):
            build_component_log_prior(
                NAMES, SPECS, BOUNDS, shared_param_names=["a"], fixed_param_names=["a"]
            )


def test_default_priors_are_studentt_on_logsfr_ratios():
    """The emulator's training prior on logsfr_ratio_* is Student-t(df=2, scale=0.3)."""
    for i in range(5):
        assert DEFAULT_PRIORS[f"logsfr_ratio_{i}"] == {
            "dist": "studentt",
            "df": 2.0,
            "loc": 0.0,
            "scale": 0.3,
        }
    others = [p for p in DEFAULT_PRIORS if not p.startswith("logsfr_ratio_")]
    assert all(DEFAULT_PRIORS[p] == {"dist": "uniform"} for p in others)
    assert np.all([DEFAULT_PRIORS[p]["dist"] in SUPPORTED_DISTS for p in DEFAULT_PRIORS])
