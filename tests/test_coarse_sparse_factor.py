"""The sparse block-Cholesky factor of the coarse matrix (``deflation.SPARSE_MIN``; :mod:`block_cholesky`).

Above a size threshold a symmetric coarse matrix of several parts is factored over the parts' graph instead of
densely. Oracles: forced on at small sizes, it is the dense path's operator to rounding (a partitioned space, an
overlapping sparse space, GenEO); a kernel in the space still raises; no array of the factor is ``K x K``; the
block factor and solve alone match a dense solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver import deflation
from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

pytest.importorskip("pymetis", reason="partitions and the nested-dissection order use METIS: the [metis] extra")

R = jno.shape.rect


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _elasticity(size=0.12):
    d = R(0.0, 0.0, 3.0, 1.0, size=size).domain()
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    u, v = d.fem_symbols(value_shape=(2,))
    u_, v_ = u.bind(x=x, y=y), v.bind(x=x, y=y)
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    tr = jno.np.trace
    fem = jno.fem([2.0 * ddot(ε(u_), ε(v_)) + 1.5 * tr(ε(u_)) * tr(ε(v_)) + 0.1 * v_[1], u(xl, yl) - (0.0, 0.0)])
    return fem, (x, y)


def _matrix(spec, fem):
    spec.prepare(fem)
    A = fem._op[0]
    ap = materialize_precond(spec, PrecondContext(LinearOperator(A), fem))
    return np.asarray(jax.jit(jax.vmap(ap, in_axes=1, out_axes=1))(jnp.eye(A.shape[0])))


def _both(make, fem, monkeypatch):
    """The preconditioner's matrix with the dense factor, then with the sparse one forced on."""
    dense = make()
    B_dense = _matrix(dense, fem)
    monkeypatch.setattr(deflation, "SPARSE_MIN", 0)
    sparse = make()
    B_sparse = _matrix(sparse, fem)
    return B_dense, B_sparse, dense, sparse


def _coarse_of(spec):
    return spec._pattern if hasattr(spec, "space") else spec._pattern.coarse


@pytest.mark.parametrize("kind", ["rigid", "geneo", "schwarz-geneo"])
def test_the_sparse_factor_is_the_dense_one(kind, monkeypatch):
    fem, (x, y) = _elasticity()

    def make():
        S = jno.precond.schwarz(parts=12, coarse=False)
        if kind == "rigid":
            Q = jno.precond.coarse([(1, 0), (0, 1), (-y, x)], parts=S)
        elif kind == "geneo":
            Q = jno.precond.coarse(jno.space.geneo(nev=6), parts=S)
        else:
            return jno.precond.schwarz(parts=12, coarse=jno.space.geneo(nev=6))
        return Q

    B_dense, B_sparse, dense, sparse = _both(make, fem, monkeypatch)
    pd, ps = _coarse_of(dense), _coarse_of(sparse)
    assert pd.chol is None and ps.chol is not None and ps.Gt is None
    np.testing.assert_allclose(B_sparse, B_dense, rtol=1e-10, atol=1e-10 * np.abs(B_dense).max())


def test_a_bnn_solve_on_the_sparse_factor_gives_the_answer_and_the_count(monkeypatch):
    fem, (x, y) = _elasticity(0.08)
    ref = np.asarray(fem.solve(linear=jno.solve.lu()))

    def run():
        S = jno.precond.schwarz(parts=24, coarse=False)
        G = jno.precond.coarse(jno.space.geneo(), parts=S)
        sol = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-10, maxiter=2000), precond=G >> S >> G))
        return sol, fem.stats["linear_iterations"]["total"], G

    sol_d, its_d, _ = run()
    monkeypatch.setattr(deflation, "SPARSE_MIN", 0)
    sol_s, its_s, G = run()
    pat = G._pattern
    K = int(np.asarray(pat.kmask).sum())  # GenEO: a static structure, the kept directions masked
    assert pat.chol is not None
    assert pat.chol.nb * pat.k**2 < 0.5 * K**2  # stored blocks, not a K x K matrix
    assert abs(its_s - its_d) <= 1, (its_d, its_s)
    np.testing.assert_allclose(sol_s, ref, rtol=1e-7, atol=1e-9 * np.abs(ref).max())
    np.testing.assert_allclose(sol_d, ref, rtol=1e-7, atol=1e-9 * np.abs(ref).max())


def test_a_kernel_in_the_space_still_raises(monkeypatch):
    monkeypatch.setattr(deflation, "SPARSE_MIN", 0)
    d = R(0.0, 0.0, 1.0, 1.0, size=0.08).domain()
    x, y, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - (x - 0.5) * vi])  # pure Neumann: the constant is a kernel
    with pytest.raises(ValueError, match="KERNEL"):
        _matrix(jno.precond.coarse([1], parts=6), fem)


def test_the_block_factor_and_solve_alone_match_a_dense_solve():
    from jno.utils.solver.block_cholesky import block_cholesky, block_cholesky_plan, block_solve

    rng = np.random.default_rng(0)
    g, k = 9, 3
    p = g * g
    ei = [a * g + b for a in range(g) for b in range(g) if a + 1 < g] + [
        a * g + b for a in range(g) for b in range(g) if b + 1 < g
    ]
    ej = [i + g for i in ei[: len(ei) // 2]] + [i + 1 for i in ei[len(ei) // 2 :]]
    plan = block_cholesky_plan(p, ei, ej)
    E = np.zeros((p * k, p * k))
    for i, j in zip(ei, ej):
        B = rng.standard_normal((k, k))
        E[i * k : i * k + k, j * k : j * k + k] = B
        E[j * k : j * k + k, i * k : i * k + k] = B.T
    E += np.diag(np.abs(E).sum(axis=1) + 1.0)
    blocks = np.stack([E[r * k : r * k + k, c * k : c * k + k] for r, c in zip(plan.brow, plan.bcol)])
    b = rng.standard_normal((p, k))
    x = np.asarray(block_solve(plan, block_cholesky(plan, jnp.asarray(blocks)), jnp.asarray(b)))
    np.testing.assert_allclose(x.ravel(), np.linalg.solve(E, b.ravel()), rtol=1e-12, atol=1e-13)
    assert plan.nb < p * (p + 1) // 2  # nested dissection keeps the fill sparse
    # separators are eliminated as supernodes: per-column, this grid's elimination tree is 21 levels deep
    assert plan.nsuper < p and len(plan.levels) <= 2 * int(np.ceil(np.log2(p))) + 1
