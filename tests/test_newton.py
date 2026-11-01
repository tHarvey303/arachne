"""Tests for arachne.inference.newton (trust-region Newton with Steihaug-Toint CG)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arachne.inference.newton import (
    block_newton,
    dense_preconditioner,
    diagonal_preconditioner,
    galaxy_block_indices,
    negative_curvature_step,
    trust_region_newton,
)


@pytest.fixture(autouse=True)
def _x64():
    """Run every test in float64 and restore the previous setting."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _quadratic(d=50, cond=1e6, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    lam = np.logspace(0, np.log10(cond), d)
    A = jnp.asarray(Q @ np.diag(lam) @ Q.T)
    b = jnp.asarray(rng.normal(size=d))
    x_star = jnp.linalg.solve(A, b)
    return (lambda x: 0.5 * x @ A @ x - b @ x), A, x_star


def test_dense_preconditioned_quadratic_one_step():
    """A dense-preconditioned quadratic converges in a couple of steps."""
    f, A, x_star = _quadratic()
    res = trust_region_newton(f, jnp.zeros(50), preconditioner=dense_preconditioner(A),
                              radius0=1e6, verbose=False)  # fmt: skip
    assert res.converged
    assert res.n_iter <= 3
    np.testing.assert_allclose(res.theta, x_star, rtol=1e-6, atol=1e-8)


def test_unpreconditioned_quadratic_converges():
    """Without a preconditioner a moderately conditioned quadratic still converges."""
    f, _, x_star = _quadratic(cond=1e3)
    res = trust_region_newton(f, jnp.zeros(50), max_cg=200, tol=1e-10, verbose=False)
    assert res.converged
    np.testing.assert_allclose(res.theta, x_star, rtol=1e-4, atol=1e-6)


def test_escapes_saddle_along_negative_curvature():
    """Started at a saddle, CG meets negative curvature and the iterate reaches a minimum."""

    # Start exactly at the saddle of a double well in x0; plain Newton would stay put
    # (zero gradient along x0) or jump to the maximum.
    def f(x):
        return 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2 + 0.5 * jnp.sum(x[1:] ** 2) - 1e-6 * x[0]

    res = trust_region_newton(f, jnp.array([0.0, 1.0, -2.0]), tol=1e-12, verbose=False)
    assert res.converged
    assert abs(abs(float(res.theta[0])) - 1.0) < 1e-4
    np.testing.assert_allclose(res.theta[1:], 0.0, atol=1e-6)
    assert any(h["cg_exit"] == "negative_curvature" for h in res.history)


def test_rosenbrock():
    """The 10-d Rosenbrock function reaches its minimum at 1."""

    def f(x):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    res = trust_region_newton(f, jnp.full(10, -1.2), max_iter=500, tol=1e-12,
                              verbose=False)  # fmt: skip
    assert res.converged
    np.testing.assert_allclose(res.theta, 1.0, atol=1e-5)


def test_small_gain_is_not_convergence():
    """Stopping on the iteration cap is reported as not converged."""
    # A stalled start: max_iter=1 must report not converged rather than claiming a mode.
    f, A, _ = _quadratic(cond=1e8)
    res = trust_region_newton(f, jnp.zeros(50), max_iter=1, max_cg=2, verbose=False)
    assert not res.converged
    assert res.reason == "max_iter"


def test_diagonal_preconditioner_matches_diag():
    """Hutchinson is exact for a diagonal Hessian and preconditions it fully."""
    rng = np.random.default_rng(1)
    diag = jnp.asarray(np.logspace(0, 4, 30))

    def f(x):
        return 0.5 * jnp.sum(diag * x**2)

    M = diagonal_preconditioner(f, jnp.zeros(30), jax.random.PRNGKey(0), n_probe=4)
    np.testing.assert_allclose(M.lam, diag, rtol=1e-10)  # exact for a diagonal Hessian
    res = trust_region_newton(f, jnp.asarray(rng.normal(size=30)), preconditioner=M,
                              radius0=1e3, verbose=False)  # fmt: skip
    assert res.converged and res.n_iter <= 3


@pytest.mark.parametrize("kind", ["dense", "diag"])
def test_preconditioner_norm(kind):
    """Preconditioner.norm is the M norm."""
    rng = np.random.default_rng(2)
    B = rng.normal(size=(6, 6))
    A = B @ B.T + 6 * np.eye(6)
    M = dense_preconditioner(A)
    if kind == "diag":
        M = type(M)("diag", jnp.asarray(np.diag(A)))
        A = np.diag(np.diag(A))
    p = jnp.asarray(rng.normal(size=6))
    assert M.norm(p) == pytest.approx(float(np.sqrt(p @ A @ p)), rel=1e-10)


def test_negative_curvature_step_leaves_saddle():
    """From an exact saddle, the eigenvector search finds lower ground; at a minimum it does not."""

    def f(x):
        return 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2 + 0.5 * jnp.sum(x[1:] ** 2)

    x0 = jnp.zeros(3)
    evals, evecs = jnp.linalg.eigh(jax.hessian(f)(x0))
    cand, gain, n_neg = negative_curvature_step(f, x0, evals, evecs)
    assert n_neg == 1
    assert gain > 0.1
    res = trust_region_newton(f, cand, tol=1e-12, verbose=False)
    assert abs(abs(float(res.theta[0])) - 1.0) < 1e-4
    evals, evecs = jnp.linalg.eigh(jax.hessian(f)(res.theta))
    same, gain, n_neg = negative_curvature_step(f, res.theta, evals, evecs)
    assert n_neg == 0 and gain == 0.0
    np.testing.assert_array_equal(same, res.theta)


def test_block_newton_quadratic_one_step():
    """On a quadratic the exact block Newton step lands on the block minimiser at once."""
    f, A, _ = _quadratic(d=20, cond=1e6)
    idx = np.array([0, 3, 7, 11, 19])
    x0 = jnp.asarray(np.random.default_rng(4).normal(size=20))
    res = block_newton(f, x0, idx, tol=1e-10)
    assert res.converged and res.reason == "converged"
    accepted = [h for h in res.history if h["accepted"]]
    assert len(accepted) == 1 and res.n_iter == 2
    assert accepted[0]["rho"] == pytest.approx(1.0, rel=1e-6)
    # Block optimum: A_bb x_b = b_b - A_b,rest x_rest.
    An, b = np.asarray(A), np.asarray(jax.grad(f)(jnp.zeros(20)) * -1.0)
    rest = np.setdiff1d(np.arange(20), idx)
    x_b = np.linalg.solve(
        An[np.ix_(idx, idx)], b[idx] - An[np.ix_(idx, rest)] @ np.asarray(x0)[rest]
    )
    np.testing.assert_allclose(np.asarray(res.theta)[idx], x_b, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(np.asarray(res.theta)[rest], np.asarray(x0)[rest])
    assert res.n_hvp == 2 * idx.size


def test_block_newton_negative_curvature_and_chunks():
    """Indefinite block: |lambda| step plus line search still descends; chunking is exact."""

    def f(x):
        return 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2 + 0.5 * jnp.sum(x[1:] ** 2)

    x0 = jnp.array([0.1, 1.0, -2.0, 0.5])
    r1 = block_newton(f, x0, [0, 1, 2], tol=1e-12, max_iter=50)
    r2 = block_newton(f, x0, [0, 1, 2], tol=1e-12, max_iter=50, chunk_size=2)
    assert r1.converged and abs(abs(float(r1.theta[0])) - 1.0) < 1e-6
    np.testing.assert_allclose(r1.theta, r2.theta, rtol=1e-12)
    assert float(r1.theta[3]) == 0.5


def test_block_newton_rejects_bad_indices():
    """Empty, out-of-range and duplicate indices raise."""
    f, _, _ = _quadratic(d=5)
    for bad in ([], [5], [1, 1]):
        with pytest.raises(ValueError):
            block_newton(f, jnp.zeros(5), bad)


def test_galaxy_block_indices_duck_typing():
    """z_offset models give arange(z_offset); forward models add the nuisance tail."""

    class _Spatial:
        z_offset, n_params = 4, 30

    class _FM:
        spatial_model, n_params = _Spatial(), 33

    np.testing.assert_array_equal(galaxy_block_indices(_Spatial()), np.arange(4))
    np.testing.assert_array_equal(galaxy_block_indices(_FM()), [0, 1, 2, 3, 30, 31, 32])
    with pytest.raises(TypeError):
        galaxy_block_indices(object())
