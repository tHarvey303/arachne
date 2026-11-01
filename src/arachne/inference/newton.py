"""Trust-region Newton (Steihaug-Toint truncated CG) for polishing a MAP into a mode.

The Laplace approximation and the whitened samplers are only as good as the point
they expand around.  Adam stops well short of a mode in a 1e4-parameter posterior
with a ~1e8 condition number, and a damped Newton-CG that hands an *indefinite*
``H + mu I`` to plain conjugate gradients produces steps of arbitrary size and
direction (CG is only defined for positive-definite systems).  This module
replaces that with the standard remedy:

* **Steihaug-Toint CG** solves the trust-region subproblem
  ``min_p g.p + p.H.p / 2`` subject to ``|p|_M <= radius`` with Hessian-vector
  products only.  It stops at the boundary when it meets a direction of
  negative curvature, so saddle points are escaped along that direction rather
  than solved through.
* **Preconditioning** with a positive-definite ``M``: a dense eigen-decomposed
  Hessian (``|lambda|``, floored; built with :func:`dense_preconditioner`, the
  same object the Laplace step needs anyway) or a diagonal one
  (:func:`diagonal_preconditioner`, Hutchinson estimate, for problems too large
  for a dense Hessian).  The trust region is measured in the ``M`` norm, i.e. in
  units of the local posterior width.
* **Saddle check**: a CG solve that stops early on an interior step never explores
  weak directions of negative curvature, so a "converged" point can still be a
  saddle.  Whoever computes the dense Hessian there (the Laplace step does) should
  pass its eigen-decomposition to :func:`negative_curvature_step` and re-polish from
  the point it returns if that lowers ``-log p``.
* **Convergence** is declared when the CG step is interior (no negative
  curvature met, inside the trust region) and the *predicted* decrease of
  ``-log p`` (the Newton decrement for an exact solve) is below ``tol`` nats,
  confirmed by a re-solve to a tight residual (``verify_eta``): with the loose
  Eisenstat-Walker forcing, an identity-preconditioned CG that has only resolved
  the stiff directions dominating ``|g|`` predicts a tiny decrease while the soft
  ones are untouched (measured on a correlated-field toy: "converged" 10-180 nats
  above the mode without the check).  A small last-step gain is not convergence
  either: it is what a stalled optimiser looks like too.

* **Block Newton** (:func:`block_newton`): exact Newton on a small sub-block of
  the parameters (e.g. the ~20 galaxy-level parameters of a correlated-field fit,
  :func:`galaxy_block_indices`) with everything else held fixed.  The block
  Hessian costs one Hessian-vector product per block coordinate, so a stiff
  low-dimensional block (curvature 1e6-1e8) can be solved exactly while the
  field coefficients are left to a cheaper optimiser.

A converged result is a local mode.  It says nothing about other modes; run
several starts (see ``examples/demo_correlated_field.py --n-starts``) and compare
them by evidence, or let MCMC chains started at different modes tell you.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "CG_EXIT",
    "NewtonResult",
    "Preconditioner",
    "block_newton",
    "dense_preconditioner",
    "diagonal_preconditioner",
    "galaxy_block_indices",
    "negative_curvature_step",
    "trust_region_newton",
]

# CG exit codes (index into CG_EXIT).
# block_newton records cg_exit = "block" (no CG: the block Hessian is solved exactly).
CG_EXIT = ("interior", "boundary", "negative_curvature", "maxiter")
_INTERIOR, _BOUNDARY, _NEGATIVE, _MAXITER = range(4)


@dataclasses.dataclass(frozen=True)
class Preconditioner:
    """Positive-definite ``M`` for the CG solve, dense (eigen form) or diagonal.

    ``kind == "dense"``: ``M = V diag(lam) V^T``.  ``kind == "diag"``:
    ``M = diag(lam)`` and ``V`` is ``None``.  ``lam > 0`` always.
    """

    kind: str
    lam: jnp.ndarray
    V: jnp.ndarray | None = None

    def arrays(self) -> tuple:
        """The arrays handed to the jitted solver."""
        return (self.lam,) if self.V is None else (self.lam, self.V)

    def norm(self, p) -> float:
        """``|p|_M = sqrt(p^T M p)``."""
        if self.V is None:
            return float(jnp.sqrt(p @ (p * self.lam)))
        u = self.V.T @ p
        return float(jnp.sqrt(u @ (u * self.lam)))


def dense_preconditioner(hess, rel_floor: float = 1e-8, min_eig: float = 1e-6) -> Preconditioner:
    """``M = V |Lambda| V^T`` from a dense Hessian, eigenvalues floored.

    Args:
        hess: ``(d, d)`` Hessian of ``-log p`` (numpy or JAX).
        rel_floor: Floor as a fraction of the largest ``|lambda|``.
        min_eig: Absolute floor.

    Returns:
        :class:`Preconditioner` of kind ``"dense"``.
    """
    h = jnp.asarray(hess)
    lam, V = jnp.linalg.eigh(0.5 * (h + h.T))
    lam = jnp.abs(lam)
    lam = jnp.maximum(lam, jnp.maximum(rel_floor * lam.max(), min_eig))
    return Preconditioner("dense", lam, V)


def diagonal_preconditioner(
    neg_log_post: Callable,
    theta,
    key: jax.Array,
    n_probe: int = 32,
    rel_floor: float = 1e-3,
) -> Preconditioner:
    """Diagonal ``M`` from a Hutchinson estimate of ``diag(H)``: ``E[v * Hv]``.

    Uses ``n_probe`` Rademacher Hessian-vector products.  ``|diag|`` is floored
    at ``rel_floor`` times its median, which keeps noisy entries from turning
    into huge steps (the trust region bounds them anyway).
    """
    grad_fn = jax.grad(neg_log_post)
    theta = jnp.asarray(theta)

    @jax.jit
    def probe(k):
        v = jax.random.rademacher(k, theta.shape, dtype=theta.dtype)
        return v * jax.jvp(grad_fn, (theta,), (v,))[1]

    est = jnp.mean(jax.lax.map(probe, jax.random.split(key, n_probe)), axis=0)
    est = jnp.abs(est)
    return Preconditioner("diag", jnp.maximum(est, rel_floor * jnp.median(est)))


@dataclasses.dataclass
class NewtonResult:
    """Outcome of :func:`trust_region_newton`.

    Attributes:
        theta: Final point.
        neg_log_post: ``-log p`` there.
        converged: Interior CG step with predicted decrease below ``tol``.
        reason: Why it stopped (``"converged"``, ``"max_iter"``, ``"radius"``,
            ``"nonfinite"``; :func:`block_newton` also ``"rejected"``, a step the
            line search could not make decrease ``-log p``).
        n_iter: Outer iterations taken.
        n_hvp: Hessian-vector products used by CG (by the block Hessians for
            :func:`block_newton`).
        history: One dict per iteration: ``iter``, ``neg_log_post``, ``grad_norm``,
            ``radius``, ``rho``, ``cg_iters``, ``cg_exit``, ``predicted``, ``actual``,
            ``accepted``.
        runtime_s: Wall time.
    """

    theta: jnp.ndarray
    neg_log_post: float
    converged: bool
    reason: str
    n_iter: int
    n_hvp: int
    history: list
    runtime_s: float

    def summary(self) -> str:
        """One-line description."""
        h0, h1 = self.history[0], self.history[-1]
        # The trailing bookkeeping record (cg_exit "-") has no prediction; report the last step's.
        h_pred = [h for h in self.history if h["cg_exit"] != "-"][-1]
        exits = [h["cg_exit"] for h in self.history]
        n_neg = sum(e == "negative_curvature" for e in exits)
        label = "block Newton" if "block" in exits else "trust-region Newton"
        return (
            f"{label}: {self.reason} after {self.n_iter} iterations "
            f"({self.n_hvp} HVPs, {self.runtime_s:.0f} s); -log_post "
            f"{h0['neg_log_post']:.3f} -> {self.neg_log_post:.3f}; |grad| "
            f"{h0['grad_norm']:.3g} -> {h1['grad_norm']:.3g}; last predicted decrease "
            f"{h_pred['predicted']:.3g}; negative curvature met in {n_neg} steps"
        )


def _make_solver(
    neg_log_post: Callable, kind: str, max_cg: int, eta_max: float, abs_tol: float = 0.0
):
    """Jitted Steihaug-Toint CG: ``(theta, g, radius, *M arrays) -> (p, Hp, n_iter, exit)``.

    Stops on ``|r| <= max(min(eta_max, sqrt|g|) |g|, abs_tol)``.
    """
    grad_fn = jax.grad(neg_log_post)

    def solve(theta, g, radius, lam, V=None):
        if kind == "dense":

            def minv(r):
                return V @ ((V.T @ r) / lam)

            def mmul(p):
                return V @ ((V.T @ p) * lam)
        else:

            def minv(r):
                return r / lam

            def mmul(p):
                return p * lam

        def hvp(v):
            return jax.jvp(grad_fn, (theta,), (v,))[1]

        gnorm = jnp.linalg.norm(g)
        # Eisenstat-Walker style forcing: loose far away, tight near the mode.
        tol = jnp.maximum(jnp.minimum(eta_max, jnp.sqrt(gnorm)) * gnorm, abs_tol)

        def to_boundary(z, Mz, d, Md):
            # Positive root of |z + tau d|_M = radius.
            a = d @ Md
            b = 2.0 * (z @ Md)
            c = z @ Mz - radius**2
            return (-b + jnp.sqrt(jnp.maximum(b * b - 4.0 * a * c, 0.0))) / (2.0 * a)

        y0 = minv(g)
        d0 = -y0
        state0 = dict(
            z=jnp.zeros_like(g), r=g, y=y0, d=d0, ry=g @ y0, it=jnp.asarray(0),
            exit=jnp.asarray(_MAXITER), done=jnp.asarray(False),
        )  # fmt: skip

        def cond(s):
            return (~s["done"]) & (s["it"] < max_cg)

        def body(s):
            z, r, d, ry = s["z"], s["r"], s["d"], s["ry"]
            Hd = hvp(d)
            kappa = d @ Hd
            Mz, Md = mmul(z), mmul(d)
            tau = to_boundary(z, Mz, d, Md)
            neg = kappa <= 0.0
            alpha = ry / jnp.where(neg, 1.0, kappa)
            z_new = z + alpha * d
            out = (z_new @ mmul(z_new)) >= radius**2
            r_new = r + alpha * Hd
            small = jnp.linalg.norm(r_new) <= tol
            y_new = minv(r_new)
            ry_new = r_new @ y_new
            d_new = -y_new + (ry_new / ry) * d
            z_out = jnp.where(neg | out, z + tau * d, z_new)
            exit_code = jnp.where(
                neg, _NEGATIVE, jnp.where(out, _BOUNDARY, jnp.where(small, _INTERIOR, _MAXITER))
            )
            return dict(
                z=z_out, r=r_new, y=y_new, d=d_new, ry=ry_new, it=s["it"] + 1,
                exit=exit_code, done=neg | out | small,
            )  # fmt: skip

        s = jax.lax.while_loop(cond, body, state0)
        p = s["z"]
        return p, hvp(p), s["it"], s["exit"]

    return jax.jit(solve)


def trust_region_newton(
    neg_log_post: Callable,
    theta0,
    *,
    preconditioner: Preconditioner | None = None,
    refresh_preconditioner: Callable | None = None,
    refresh_every: int = 0,
    max_iter: int = 200,
    max_cg: int = 250,
    tol: float = 1e-2,
    radius0: float | None = None,
    max_radius: float = 1e4,
    min_radius: float = 1e-8,
    eta_max: float = 0.5,
    verify_eta: float = 1e-6,
    verbose: bool = True,
    print_fn: Callable = print,
) -> NewtonResult:
    """Minimise ``neg_log_post`` from ``theta0`` with preconditioned trust-region Newton.

    Args:
        neg_log_post: ``theta -> -log p(theta | d)`` (JAX-differentiable; float64 advised).
        theta0: Start, e.g. an Adam MAP.
        preconditioner: Positive-definite ``M``; ``None`` uses the identity.  For
            a ~1e4-parameter correlated-field fit pass
            ``dense_preconditioner(hessian(theta0))``: CG then needs a handful of
            HVPs per step instead of hundreds.
        refresh_preconditioner: Optional ``theta -> Preconditioner`` called every
            ``refresh_every`` accepted steps (and after a step that hit the CG
            iteration cap), e.g. to recompute the dense Hessian once the iterate
            has moved far from where ``M`` was built.
        refresh_every: See above; 0 disables periodic refreshes.
        max_iter: Outer iterations.
        max_cg: CG iterations per subproblem.
        tol: Converged when an interior CG step predicts a decrease below this (nats).
        radius0: Initial trust radius in the ``M`` norm (posterior-width units);
            default ``sqrt(d)`` -- one sigma per direction.
        max_radius: Largest trust radius.
        min_radius: Stop (``reason="radius"``) if the radius shrinks below this.
        eta_max: Largest CG forcing term (relative residual tolerance).
        verify_eta: Forcing term of the confirming re-solve: an interior step with
            predicted decrease below ``tol`` is re-solved to this relative residual
            (or to ``|r| <= 0.1 sqrt(tol)``, whichever is looser) before convergence
            is declared (a loose solve that has only resolved
            the stiff directions dominating ``|g|`` predicts a tiny decrease while
            the soft directions are far from converged).  ``>= eta_max`` disables it.
        verbose: Print one line per iteration.
        print_fn: Printer for those lines.

    Returns:
        :class:`NewtonResult`.
    """
    t_start = time.time()
    theta = jnp.asarray(theta0)
    d = theta.size
    f_fn = jax.jit(neg_log_post)
    g_fn = jax.jit(jax.grad(neg_log_post))
    if preconditioner is None:
        preconditioner = Preconditioner("diag", jnp.ones(d, theta.dtype))
    solvers: dict[tuple, Callable] = {}

    def solver_for(kind, eta=eta_max, abs_tol=0.0):
        if (kind, eta) not in solvers:
            solvers[kind, eta] = _make_solver(neg_log_post, kind, max_cg, eta, abs_tol)
        return solvers[kind, eta]

    radius = float(np.sqrt(d) if radius0 is None else radius0)
    f = float(f_fn(theta))
    g = g_fn(theta)
    history: list[dict] = []
    n_hvp, n_accepted = 0, 0
    converged, reason = False, "max_iter"
    for it in range(max_iter):
        if not np.isfinite(f):
            reason = "nonfinite"
            break
        gnorm = float(jnp.linalg.norm(g))
        p, Hp, n_cg, exit_code = solver_for(preconditioner.kind)(
            theta, g, jnp.asarray(radius, theta.dtype), *preconditioner.arrays()
        )
        n_cg, exit_code = int(n_cg), int(exit_code)
        n_hvp += n_cg + 1
        predicted = float(-(g @ p + 0.5 * (p @ Hp)))
        if exit_code == _INTERIOR and predicted < tol and verify_eta < eta_max:
            # Confirm with a tight solve before calling it a mode (see verify_eta).
            # The floor bounds the unexplored decrement |r|^2 / (2 lambda_min) by ~tol / 200
            # in prior-whitened coordinates (lambda_min >~ 1) without chasing round-off.
            p, Hp, n_cg2, exit_code = solver_for(
                preconditioner.kind, verify_eta, 0.1 * float(np.sqrt(tol))
            )(theta, g, jnp.asarray(radius, theta.dtype), *preconditioner.arrays())
            n_cg2, exit_code = int(n_cg2), int(exit_code)
            n_hvp += n_cg2 + 1
            n_cg += n_cg2 + 1
            predicted = float(-(g @ p + 0.5 * (p @ Hp)))
        if exit_code == _INTERIOR and predicted < tol:
            history.append(dict(iter=it, neg_log_post=f, grad_norm=gnorm, radius=radius,
                                rho=float("nan"), cg_iters=n_cg, cg_exit=CG_EXIT[exit_code],
                                predicted=predicted, actual=0.0, accepted=False))  # fmt: skip
            if verbose:
                print_fn(f"  TR-Newton {it}: converged, predicted decrease {predicted:.3g} "
                         f"< {tol:g}  |grad| {gnorm:.3g}  ({n_cg} CG)")  # fmt: skip
            converged, reason = True, "converged"
            break
        cand = theta + p
        f_new = float(f_fn(cand))
        actual = f - f_new if np.isfinite(f_new) else -np.inf
        rho = actual / predicted if predicted > 0 else -np.inf
        p_norm = preconditioner.norm(p)
        # Standard radius update (Nocedal & Wright, Alg. 4.1).
        if rho < 0.25:
            radius = 0.25 * min(radius, p_norm)
        elif rho > 0.75 and exit_code in (_BOUNDARY, _NEGATIVE):
            radius = min(2.0 * radius, max_radius)
        accepted = rho > 1e-4
        history.append(dict(iter=it, neg_log_post=f, grad_norm=gnorm, radius=radius,
                            rho=float(rho), cg_iters=n_cg, cg_exit=CG_EXIT[exit_code],
                            predicted=predicted, actual=float(actual),
                            accepted=bool(accepted)))  # fmt: skip
        if verbose:
            print_fn(
                f"  TR-Newton {it}: -log_post {f:.3f} -> {f_new:.3f}  pred {predicted:.3g}  "
                f"rho {rho:.2f}  |grad| {gnorm:.3g}  radius {radius:.3g}  "
                f"CG {n_cg} ({CG_EXIT[exit_code]}){'' if accepted else '  rejected'}"
            )
        if accepted:
            theta, f = cand, f_new
            g = g_fn(theta)
            n_accepted += 1
            if refresh_preconditioner is not None and (
                (refresh_every and n_accepted % refresh_every == 0) or exit_code == _MAXITER
            ):
                preconditioner = refresh_preconditioner(theta)
        if radius < min_radius:
            reason = "radius"
            break
    if not converged and history and history[-1]["accepted"]:
        # Record the final state's gradient for the summary.
        history.append(dict(iter=len(history), neg_log_post=f,
                            grad_norm=float(jnp.linalg.norm(g)), radius=radius,
                            rho=float("nan"), cg_iters=0, cg_exit="-", predicted=float("nan"),
                            actual=0.0, accepted=False))  # fmt: skip
    return NewtonResult(
        theta=theta, neg_log_post=f, converged=converged, reason=reason,
        n_iter=len([h for h in history if h["cg_exit"] != "-"]), n_hvp=n_hvp,
        history=history, runtime_s=time.time() - t_start,
    )  # fmt: skip


def negative_curvature_step(
    neg_log_post: Callable,
    theta,
    evals,
    evecs,
    *,
    n_dirs: int = 4,
    scales: tuple = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    rel_tol: float = 1e-9,
) -> tuple[jnp.ndarray, float, int]:
    """Search along the most negative-curvature eigenvectors of the Hessian at ``theta``.

    For each of the (up to ``n_dirs``) eigenvalues ``lambda < -rel_tol * max |lambda|``,
    tries ``theta +- a v / sqrt(|lambda|)`` for every ``a`` in ``scales`` (a step of ``a``
    curvature lengths) and keeps the lowest ``-log p``.

    Args:
        neg_log_post: ``theta -> -log p``.
        theta: Point where the Hessian was evaluated (typically a converged
            :func:`trust_region_newton` result).
        evals: Ascending eigenvalues of the Hessian there.
        evecs: Matching eigenvectors (columns).
        n_dirs: Most negative directions to try.
        scales: Step lengths in units of ``1 / sqrt(|lambda|)``.
        rel_tol: Eigenvalues above ``-rel_tol * max |lambda|`` count as zero.

    Returns:
        ``(theta_best, gain, n_negative)``: the best point found (``theta`` itself if
        nothing improves), the decrease of ``-log p`` there, and the number of
        eigenvalues counted as negative.
    """
    f = jax.jit(neg_log_post)
    theta = jnp.asarray(theta)
    evals = np.asarray(evals)
    thresh = -rel_tol * float(np.abs(evals).max())
    neg = np.flatnonzero(evals < thresh)
    f0 = float(f(theta))
    best, best_f = theta, f0
    for j in neg[:n_dirs]:
        v = jnp.asarray(np.asarray(evecs)[:, j], theta.dtype)
        base = 1.0 / np.sqrt(abs(float(evals[j])))
        for sign in (1.0, -1.0):
            for a in scales:
                cand = theta + (sign * a * base) * v
                fc = float(f(cand))
                if np.isfinite(fc) and fc < best_f:
                    best, best_f = cand, fc
    return best, f0 - best_f, int(neg.size)


def galaxy_block_indices(model) -> np.ndarray:
    """Indices of the galaxy-level parameters of a correlated-field model.

    For a :class:`~arachne.spatial.correlated_field.CorrelatedFieldMap`-like
    spatial model (anything with ``z_offset``) these are the field levels,
    globals, hyperparameters and couplings, ``arange(z_offset)``.  For a forward
    model wrapping one (``model.spatial_model.z_offset``) the nuisance
    parameters appended after the spatial block are included too, since they
    are equally few and equally stiff.

    Args:
        model: Spatial model with ``z_offset``, or a forward model whose
            ``spatial_model`` has one.

    Returns:
        ``int64`` array of parameter indices, ascending.

    Raises:
        TypeError: If neither ``model`` nor ``model.spatial_model`` has ``z_offset``.
    """
    if hasattr(model, "z_offset"):
        return np.arange(int(model.z_offset))
    spatial = getattr(model, "spatial_model", None)
    if spatial is not None and hasattr(spatial, "z_offset"):
        n_spatial, n_total = int(spatial.n_params), int(model.n_params)
        return np.concatenate([np.arange(int(spatial.z_offset)), np.arange(n_spatial, n_total)])
    raise TypeError(f"{type(model).__name__} has no z_offset (not a correlated-field model)")


def _block_hessian_fn(
    neg_log_post: Callable, idx, chunk_size: int | None = None
) -> Callable[[jnp.ndarray], np.ndarray]:
    """``theta -> H[idx][:, idx]`` (numpy float64) from one HVP per block coordinate.

    The HVPs along the unit vectors of ``idx`` are vmapped (``chunk_size`` at a
    time, the last chunk padded so only one shape is ever compiled) and only
    their ``idx`` entries are kept, so memory is ``O(chunk * d)``.
    """
    idx = np.asarray(idx, dtype=np.int64)
    n = idx.size
    chunk = n if chunk_size is None else max(1, min(int(chunk_size), n))
    grad_fn = jax.grad(neg_log_post)
    idx_j = jnp.asarray(idx)

    @jax.jit
    def rows(theta, cols):
        def one(i):
            v = jnp.zeros_like(theta).at[i].set(1.0)
            return jax.jvp(grad_fn, (theta,), (v,))[1][idx_j]

        return jax.vmap(one)(cols)

    def block_hessian(theta) -> np.ndarray:
        out = np.empty((n, n), dtype=np.float64)
        for start in range(0, n, chunk):
            sel = idx[start : start + chunk]
            m = sel.size
            cols = np.concatenate([sel, np.full(chunk - m, idx[0])])  # pad the ragged tail
            out[start : start + m] = np.asarray(rows(theta, jnp.asarray(cols)), np.float64)[:m]
        return 0.5 * (out + out.T)

    return block_hessian


def block_newton(
    neg_log_post: Callable,
    theta,
    idx,
    *,
    max_iter: int = 10,
    tol: float = 1e-3,
    rel_floor: float = 1e-8,
    min_eig: float = 1e-6,
    chunk_size: int | None = None,
    verbose: bool = False,
    print_fn: Callable = print,
) -> NewtonResult:
    """Exact (eigen-modified) Newton on the sub-block ``idx``, all other coordinates fixed.

    Each iteration builds the block Hessian ``H_bb`` from ``len(idx)``
    Hessian-vector products (jitted once), replaces its eigenvalues by
    ``max(|lambda|, max(rel_floor * max|lambda|, min_eig))`` (as
    :func:`dense_preconditioner` does, so negative curvature becomes a descent
    direction of the right length), and takes ``p = -H_mod^-1 g_b`` with an
    Armijo backtracking line search on the full objective (step halved up to
    10 times).  Coordinates outside ``idx`` are never touched (bit-identical).

    Stops when the predicted decrease ``g_b^T H_mod^-1 g_b / 2`` is below
    ``tol`` (converged), when no step length satisfies Armijo (``"rejected"``)
    or after ``max_iter`` iterations.  Typical use: the galaxy-level block of a
    correlated-field fit (``idx = galaxy_block_indices(model)``), whose
    curvature (1e6-1e8) is far above that of the field coefficients.

    Args:
        neg_log_post: ``theta -> -log p(theta | d)`` (JAX-differentiable).
        theta: Start, shape ``(d,)``.
        idx: Integer indices of the block (``(n,)``, no duplicates).
        max_iter: Newton iterations.
        tol: Converged when the predicted decrease is below this (nats).
        rel_floor: Eigenvalue floor relative to the largest ``|lambda|``.
        min_eig: Absolute eigenvalue floor.
        chunk_size: HVPs vmapped at once (default: all ``n``).
        verbose: Print one line per iteration.
        print_fn: Printer for those lines.

    Returns:
        :class:`NewtonResult` whose history entries have ``cg_exit == "block"``,
        ``cg_iters == len(idx)`` and ``rho`` = actual / predicted decrease of the
        accepted (possibly shortened) step.

    Raises:
        ValueError: If ``idx`` is empty, out of range or has duplicates.
    """
    t_start = time.time()
    theta = jnp.asarray(theta)
    d = theta.size
    idx = np.asarray(idx, dtype=np.int64).ravel()
    if idx.size == 0 or idx.min() < 0 or idx.max() >= d or np.unique(idx).size != idx.size:
        raise ValueError(f"idx must be distinct indices in [0, {d}); got {idx}")
    n = idx.size
    idx_j = jnp.asarray(idx)
    f_fn = jax.jit(neg_log_post)
    g_fn = jax.jit(lambda t: jax.grad(neg_log_post)(t)[idx_j])
    hess_fn = _block_hessian_fn(neg_log_post, idx, chunk_size)

    @jax.jit
    def moved(t, step):
        return t.at[idx_j].add(step.astype(t.dtype))

    f = float(f_fn(theta))
    history: list[dict] = []
    n_hvp = 0
    converged, reason = False, "max_iter"
    for it in range(max_iter):
        if not np.isfinite(f):
            reason = "nonfinite"
            break
        g = np.asarray(g_fn(theta), np.float64)
        hb = hess_fn(theta)
        n_hvp += n
        lam, V = np.linalg.eigh(hb)
        lam = np.abs(lam)
        lam = np.maximum(lam, max(rel_floor * float(lam.max()), min_eig))
        u = V.T @ g
        p = -V @ (u / lam)
        predicted = 0.5 * float(np.sum(u * u / lam))
        gnorm = float(np.linalg.norm(g))
        record = dict(iter=it, neg_log_post=f, grad_norm=gnorm, radius=float("nan"),
                      cg_iters=n, cg_exit="block", predicted=predicted)  # fmt: skip
        if predicted < tol:
            history.append(dict(record, rho=float("nan"), actual=0.0, accepted=False))
            if verbose:
                print_fn(f"  block Newton {it}: converged, predicted decrease {predicted:.3g} "
                         f"< {tol:g}  |grad_b| {gnorm:.3g}")  # fmt: skip
            converged, reason = True, "converged"
            break
        slope = float(g @ p)  # = -2 * predicted < 0
        alpha, accepted, f_new, cand = 1.0, False, np.inf, theta
        for _ in range(11):
            cand = moved(theta, jnp.asarray(alpha * p))
            f_new = float(f_fn(cand))
            if np.isfinite(f_new) and f_new <= f + 1e-4 * alpha * slope:
                accepted = True
                break
            alpha *= 0.5
        actual = f - f_new if np.isfinite(f_new) else -np.inf
        # Predicted decrease of the quadratic model along the shortened step.
        pred_alpha = alpha * predicted * (2.0 - alpha)
        rho = actual / pred_alpha if pred_alpha > 0 else float("nan")
        history.append(dict(record, rho=float(rho), actual=float(actual),
                            accepted=bool(accepted)))  # fmt: skip
        if verbose:
            print_fn(
                f"  block Newton {it}: -log_post {f:.3f} -> {f_new:.3f}  pred {predicted:.3g}  "
                f"step {alpha:.3g}  |grad_b| {gnorm:.3g}{'' if accepted else '  rejected'}"
            )
        if not accepted:
            reason = "rejected"
            break
        theta, f = cand, f_new
    if not converged and history and history[-1]["accepted"]:
        g = np.asarray(g_fn(theta), np.float64)
        history.append(dict(iter=len(history), neg_log_post=f,
                            grad_norm=float(np.linalg.norm(g)), radius=float("nan"),
                            rho=float("nan"), cg_iters=0, cg_exit="-", predicted=float("nan"),
                            actual=0.0, accepted=False))  # fmt: skip
    return NewtonResult(
        theta=theta, neg_log_post=f, converged=converged, reason=reason,
        n_iter=len([h for h in history if h["cg_exit"] != "-"]), n_hvp=n_hvp,
        history=history, runtime_s=time.time() - t_start,
    )  # fmt: skip
