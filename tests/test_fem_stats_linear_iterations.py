"""``fem.stats["linear_iterations"]`` -- how many Krylov iterations each linear solve took (#104).

"How many Krylov iterations did each Newton step cost" is the number that says whether a preconditioner
works, and it was invisible: the ``jax.scipy`` loops count their steps and discard the count. The counted
CG / BiCGStab are those loops keeping it, and every jNO-owned Krylov solver returns its own.

Oracles:
* **exact counts** -- CG and FGMRES on a matrix with ``k`` distinct eigenvalues converge in exactly ``k``
  steps (the Krylov space is then invariant); the count must say so.
* **same arithmetic** -- the counted loops return the upstream ``x`` (bit-identical on CPU).
* **per Newton step** -- a matrix-free Newton reports one count per step, as many as ``nonlinear.steps``.
* **honest gaps** -- upstream GMRES is listed as uncounted rather than dropped; a direct factorisation
  reports ``None``; a march records nothing per step (``fem.stats["march"]`` is its record).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.krylov import bicgstab_counted, cg_counted, fgmres


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _distinct_eigs(k=5, n=60):
    """SPD diagonal operator with exactly ``k`` distinct eigenvalues, and a right-hand side touching all."""
    vals = np.repeat(np.linspace(1.0, 9.0, k), n // k)
    return jnp.asarray(vals), jnp.asarray(np.random.default_rng(0).standard_normal(n))


@pytest.mark.parametrize("k", [1, 3, 5])
def test_cg_and_fgmres_count_exactly_the_number_of_distinct_eigenvalues(k):
    d, b = _distinct_eigs(k)
    mv = lambda v: d * v  # noqa: E731
    x, its = jax.jit(lambda b: cg_counted(mv, b, tol=1e-12, maxiter=100))(b)
    assert int(its) == k
    assert float(jnp.max(jnp.abs(d * x - b))) < 1e-10
    x, its = fgmres(mv, b, tol=1e-12, restart=30, return_iters=True)
    assert int(its) == k


def _dirichlet_square(size=0.1):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    u, v = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    return d, u, v.bind(x=x, y=y), u.bind(x=x, y=y), (xb, yb)


def test_the_counted_loops_return_the_upstream_answer():
    _, u, vi, ui, (xb, yb) = _dirichlet_square()
    A, b = jno.fem([ui.x * vi.x + ui.y * vi.y + 3.0 * ui.x * vi - vi, u(xb, yb) - 0.0])._op
    b = jnp.asarray(b).reshape(-1)
    inv = 1.0 / jnp.asarray(A.todense()).diagonal()
    mv, M = (lambda v: A @ v), (lambda r: inv * r)  # noqa: E731
    x, k, broke = bicgstab_counted(mv, b, tol=1e-10, maxiter=5000, M=M)
    ref = jax.scipy.sparse.linalg.bicgstab(mv, b, tol=1e-10, atol=0.0, maxiter=5000, M=M)[0]
    assert float(jnp.max(jnp.abs(x - ref))) <= 1e-12 * float(jnp.max(jnp.abs(ref)))
    assert int(k) > 0 and not bool(broke)

    x, k = cg_counted(mv, b, tol=1e-10, maxiter=5000, M=M)  # non-symmetric A: only the arithmetic is compared
    ref = jax.scipy.sparse.linalg.cg(mv, b, tol=1e-10, atol=0.0, maxiter=5000, M=M)[0]
    assert np.array_equal(np.asarray(x), np.asarray(ref)) or float(jnp.max(jnp.abs(x - ref))) < 1e-12


@pytest.mark.parametrize(
    "slot, who",
    [
        ({}, "fem.solve default (Jacobi-preconditioned BiCGStab)"),
        ({"linear": "cg", "precond": "jacobi"}, "jno.solve.cg"),
        ({"linear": "bicgstab", "precond": "jacobi"}, "jno.solve.bicgstab"),
        ({"linear": "minres"}, "jno.solve.minres"),
        ({"linear": "fgmres"}, "jno.solve.fgmres"),
    ],
)
def test_a_steady_linear_solve_reports_its_count(slot, who):
    _, u, vi, ui, (xb, yb) = _dirichlet_square()
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - vi, u(xb, yb) - 0.0])
    kw = {}
    if "linear" in slot:
        kw["linear"] = getattr(jno.solve, slot["linear"])()
    if "precond" in slot:
        kw["precond"] = jno.precond.jacobi()
    fem.solve(**kw)
    rec = fem.stats["linear_iterations"]
    assert rec["by"] == [who]
    assert len(rec["iterations"]) == 1 and 0 < rec["iterations"][0] < 1000, rec
    assert rec["total"] == rec["max"] == rec["iterations"][0]


def test_gmres_is_listed_as_uncounted_not_dropped():
    _, u, vi, ui, (xb, yb) = _dirichlet_square()
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - vi, u(xb, yb) - 0.0])
    fem.solve(linear=jno.solve.gmres())
    rec = fem.stats["linear_iterations"]
    assert rec["iterations"] == [] and rec["uncounted"][0]["who"] == "jno.solve.gmres"
    assert len(rec["uncounted"]) == 1, "only the forward solve ran; a traced transpose is not a solve"


def test_matrix_free_newton_reports_one_count_per_newton_step():
    _, u, vi, ui, (xb, yb) = _dirichlet_square()
    fem = jno.fem([(1 + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 10.0 * vi, u(xb, yb) - 0.0])
    fem.solve(nonlinear=jno.solve.newton(direct=False))
    steps = fem.stats["nonlinear"]["steps"]
    rec = fem.stats["linear_iterations"]
    assert steps >= 2
    assert len(rec["iterations"]) == steps, (steps, rec)
    assert all(k > 0 for k in rec["iterations"])


def test_a_direct_solve_and_a_march_record_no_krylov_iterations():
    _, u, vi, ui, (xb, yb) = _dirichlet_square()
    fem = jno.fem([(1 + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 10.0 * vi, u(xb, yb) - 0.0])
    fem.solve(nonlinear=jno.solve.newton(direct=True))
    assert fem.stats["linear_iterations"] is None

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain(time=(0.0, 0.1, 6))
    u, v = d.fem_symbols()
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0])
    traj = np.asarray(fem.solve().fn())
    assert traj.shape[0] == 6
    from jno.utils.solver.solver_api import LAST_LINEAR_STATS

    assert LAST_LINEAR_STATS == [], "a march step must not call back to the host"
