"""The pure-JAX Krylov trio jNO implements itself (absent from the JAX ecosystem): FGMRES
(Saad 1993, Alg. 2.2), MINRES (Paige & Saunders 1975), Chebyshev (Saad 2003 §12.3).

Pins: correctness vs dense/scipy oracles on the structure each method targets (non-symmetric for
FGMRES, symmetric *indefinite* for MINRES, SPD for Chebyshev); FGMRES's distinguishing property
(an *iterative*, per-call-varying preconditioner — illegal for plain GMRES); restart smaller than
the Krylov dimension; the Chebyshev polynomial preconditioner accelerating CG; ``jit`` + ``vmap``
at the contract level; reverse-mode differentiability through the ``custom_linear_solve``
firewall (vs finite differences, non-symmetric so a transpose bug cannot hide); and end-to-end
use as ``fem.solve`` slots.
"""

from __future__ import annotations

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _spd(n=40, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    return Q @ Q.T + n * np.eye(n)


def _sym_indefinite(n=40, seed=1):
    """Symmetric with a genuinely mixed spectrum — CG diverges here, MINRES is the method."""
    rng = np.random.default_rng(seed)
    D = np.diag(np.concatenate([np.linspace(1.0, 10.0, n // 2), -np.linspace(1.0, 5.0, n - n // 2)]))
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return U @ D @ U.T


def _nonsym(n=40, seed=2):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)) + n * np.eye(n)


def _op(Ad):
    return jno.solve.LinearOperator(jsp.BCOO.fromdense(jnp.asarray(Ad)))


def _b(n, seed=3):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(n))


# ---------------------------------------------------------------------------
# correctness vs oracles
# ---------------------------------------------------------------------------


def test_minres_symmetric_indefinite_vs_scipy():
    Ad = _sym_indefinite()
    b = _b(Ad.shape[0])
    x = jno.solve.minres(tol=1e-12)(_op(Ad), b)
    assert np.abs(np.asarray(x) - np.linalg.solve(Ad, np.asarray(b))).max() < 1e-9
    sp = pytest.importorskip("scipy.sparse.linalg")
    x_sp, info = sp.minres(Ad, np.asarray(b), rtol=1e-12)
    assert info == 0
    assert np.abs(np.asarray(x) - x_sp).max() < 1e-7


def test_minres_spd_preconditioner():
    Ad = _spd(seed=4) + np.diag(np.linspace(0, 500, 40))  # spread the diagonal so Jacobi bites
    b = _b(40)
    op = _op(Ad)
    inv = 1.0 / jnp.asarray(np.diag(Ad))
    x = jno.solve.minres(tol=1e-12)(op, b, M=lambda v: inv * v)
    assert np.abs(np.asarray(x) - np.linalg.solve(Ad, np.asarray(b))).max() < 1e-9


def test_fgmres_nonsymmetric_and_restarted():
    Ad = _nonsym()
    b = _b(Ad.shape[0])
    x_ref = np.linalg.solve(Ad, np.asarray(b))
    x = jno.solve.fgmres(tol=1e-12)(_op(Ad), b)
    assert np.abs(np.asarray(x) - x_ref).max() < 1e-9
    # restart << n exercises the outer cycle loop
    x_r = jno.solve.fgmres(tol=1e-12, restart=7)(_op(Ad), b)
    assert np.abs(np.asarray(x_r) - x_ref).max() < 1e-9


def test_fgmres_flexible_iterative_preconditioner():
    """The defining FGMRES property: M is itself an inexact Krylov solve (varies per call)."""
    Ad = _spd(seed=5)
    b = _b(Ad.shape[0])
    op = _op(Ad)
    M_inner = lambda v: jax.scipy.sparse.linalg.cg(op.mv, v, tol=1e-2, maxiter=5)[0]
    x = jno.solve.fgmres(tol=1e-12)(op, b, M=M_inner)
    assert np.abs(np.asarray(x) - np.linalg.solve(Ad, np.asarray(b))).max() < 1e-9


def test_chebyshev_solver_true_and_estimated_bounds():
    Ad = _spd(seed=6)
    b = _b(Ad.shape[0])
    x_ref = np.linalg.solve(Ad, np.asarray(b))
    lam = np.linalg.eigvalsh(Ad)
    x = jno.solve.chebyshev(lmin=float(lam[0]), lmax=float(lam[-1]), tol=1e-12)(_op(Ad), b)
    assert np.abs(np.asarray(x) - x_ref).max() < 1e-9
    x_auto = jno.solve.chebyshev(tol=1e-12, maxiter=2000)(_op(Ad), b)  # power-iteration bounds
    assert np.abs(np.asarray(x_auto) - x_ref).max() < 1e-8


def test_chebyshev_polynomial_preconditioner_accelerates_cg():
    from jno.utils.solver.solver_api import PrecondContext, materialize_precond

    Ad = _spd(seed=7)
    b = _b(Ad.shape[0])
    op = _op(Ad)
    M = materialize_precond(jno.precond.chebyshev(degree=8), PrecondContext(op))
    x = jno.solve.cg(tol=1e-12)(op, b, M=M)
    assert np.abs(np.asarray(x) - np.linalg.solve(Ad, np.asarray(b))).max() < 1e-9
    # the fixed-degree application is linear in v (required for CG): p(A)(a v + w) = a p(A)v + p(A)w
    v, w = _b(40, seed=8), _b(40, seed=9)
    lin = np.asarray(M(2.0 * v + w) - 2.0 * M(v) - M(w))
    assert np.abs(lin).max() < 1e-10


# ---------------------------------------------------------------------------
# transforms: jit, vmap, grad (the custom_linear_solve firewall)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [lambda: jno.solve.fgmres(tol=1e-12), lambda: jno.solve.minres(tol=1e-12)])
def test_jit_and_vmap(factory):
    Ad = _sym_indefinite(seed=10)
    op = _op(Ad)
    solver = factory()
    b = _b(Ad.shape[0])
    x = jax.jit(lambda bb: solver(op, bb))(b)
    assert np.abs(np.asarray(x) - np.linalg.solve(Ad, np.asarray(b))).max() < 1e-8
    B = jnp.stack([b, -b, 3.0 * b + 1.0])
    X = jax.vmap(lambda bb: solver(op, bb))(B)
    assert np.abs(np.asarray(X) - np.linalg.solve(Ad, np.asarray(B).T).T).max() < 1e-8


def test_grad_through_firewall_nonsymmetric():
    """Reverse-mode through fgmres on a NON-symmetric parametric system vs finite differences —
    exercises the transpose solve of the custom_linear_solve wrapper."""
    A0 = jnp.asarray(_nonsym(n=12, seed=11))
    P = jnp.asarray(np.random.default_rng(12).standard_normal((12, 12)))
    b = _b(12, seed=13)
    solver = jno.solve.fgmres(tol=1e-13)

    def loss(theta):
        op = jno.solve.LinearOperator(A0 + theta * P)
        return jnp.sum(solver(op, b) ** 2)

    g = float(jax.grad(loss)(0.3))
    eps = 1e-6
    g_fd = (float(loss(0.3 + eps)) - float(loss(0.3 - eps))) / (2 * eps)
    assert abs(g - g_fd) / (abs(g_fd) + 1e-12) < 1e-5


# ---------------------------------------------------------------------------
# end-to-end as fem.solve slots
# ---------------------------------------------------------------------------


def test_fem_solve_with_new_krylov_slots():
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    u_ref = np.asarray(fem.solve())
    for solver, precond in [
        (jno.solve.fgmres(), jno.precond.jacobi()),
        (jno.solve.minres(), jno.precond.jacobi()),  # Poisson stiffness is SPD: MINRES applies
        (jno.solve.cg(), jno.precond.chebyshev(degree=6)),
        (jno.solve.chebyshev(maxiter=2000), None),
    ]:
        uu = np.asarray(fem.solve(linear=solver, precond=precond))
        assert np.abs(uu - u_ref).max() < 1e-6, f"{solver.name} deviates on Poisson"
