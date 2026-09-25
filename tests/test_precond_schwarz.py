"""jno.precond.schwarz(): algebraic overlapping Schwarz (Toselli & Widlund 2005; Cai & Sarkis 1999; Nicolaides 1987).

Oracles: the default solve (a preconditioner never changes the answer), the defining property of the two-level
method (iterations do not grow with the number of parts; without the coarse space they do), and the partition's
own contract (contiguous parts).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.linear import sparse_matvec
from jno.utils.solver.schwarz import schwarz_apply, schwarz_factor
from jno.utils.solver.solver_api import LinearOperator, PrecondContext


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(size=0.02):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    x, y = d.variable("interior", split=True)[:2]
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - 10.0 * vi, u(cb[0], cb[1]) - 0.0])


def _leaf(sol):
    return np.asarray(jax.tree_util.tree_leaves(sol)[0]).reshape(-1)


def _pcg_iterations(A, b, spec):
    spec.build(A)
    pat = spec._pattern
    mv = sparse_matvec(A)

    def run(A, b):
        fac = schwarz_factor(pat, A, coarse=spec.coarse)
        M = schwarz_apply(pat, fac, restricted=False, mv=mv if spec.coarse else None)
        bn = jnp.linalg.norm(b)

        def body(s):
            x, r, z, p, rz, k = s
            Ap = mv(p)
            a = rz / (p @ Ap)
            x, r2 = x + a * p, r - a * Ap
            z2 = M(r2)
            return x, r2, z2, z2 + ((z2 @ (r2 - r)) / rz) * p, r2 @ z2, k + 1

        z0 = M(b)
        s0 = (jnp.zeros_like(b), b, z0, z0, b @ z0, 0)
        return jax.lax.while_loop(lambda s: (jnp.linalg.norm(s[1]) > 1e-8 * bn) & (s[5] < 2000), body, s0)

    out = jax.jit(run)(A, b)
    assert float(jnp.linalg.norm(b - mv(out[0])) / jnp.linalg.norm(b)) < 1e-7
    return int(out[5])


def test_the_answer_is_the_default_one():
    fem = _poisson()
    ref = _leaf(fem.solve())
    for spec, lin in (
        (jno.precond.schwarz(parts=32), jno.solve.cg(tol=1e-11)),
        (jno.precond.schwarz(parts=32, restricted=True), jno.solve.bicgstab(tol=1e-11)),
        (jno.precond.schwarz(parts=32, restricted=True), jno.solve.cg(tol=1e-11)),  # -> flexible CG
        (jno.precond.schwarz(parts=32, float32=True), jno.solve.cg(tol=1e-11)),
    ):
        np.testing.assert_allclose(_leaf(fem.solve(linear=lin, precond=spec)), ref, rtol=1e-7, atol=1e-10)


def test_the_coarse_space_keeps_iterations_flat_as_parts_grow():
    fem = _poisson(0.012)
    A, b = fem._op
    b = jnp.asarray(b).reshape(-1)
    one = [_pcg_iterations(A, b, jno.precond.schwarz(parts=p, coarse=False)) for p in (16, 256)]
    two = [_pcg_iterations(A, b, jno.precond.schwarz(parts=p, coarse=True)) for p in (16, 256)]
    assert one[1] > 1.5 * one[0], one  # one level: communication only between neighbours
    assert two[1] < 1.3 * two[0] and two[1] < 0.7 * one[1], (one, two)


def test_parts_are_contiguous_even_with_isolated_dirichlet_rows():
    from scipy.sparse.csgraph import connected_components

    from jno.utils.solver.amg import _to_scipy_csr

    A = _poisson()._op[0]
    spec = jno.precond.schwarz(parts=16, overlap=0)
    spec.build(A)
    S = _to_scipy_csr(A)
    G = (abs(S) + abs(S).T).tocsr()
    part = np.asarray(spec._pattern.part)
    diag_only = np.diff(G.indptr) == 1  # eliminated Dirichlet rows: isolated nodes
    for i in range(16):
        nodes = np.nonzero((part == i) & ~diag_only)[0]
        # BFS level-set bisection can leave a part in two pieces where a cut falls mid-level; the bug this
        # guards against left one in 568 (it ordered unreached nodes by raw index).
        assert connected_components(G[nodes][:, nodes])[0] <= 2


def test_a_transient_march():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.08).domain(time=(0.0, 0.1, 6))
    x, y, t = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    ic = u(ci[0], ci[1]) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [ci[0], ci[1]])
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(cb[0], cb[1]) - 0.0, ic])
    ref = np.asarray(fem.solve().fn())
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-12), precond=jno.precond.schwarz(parts=8)).fn())
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


def test_refusals():
    for kw in (dict(parts=0), dict(parts=2.5), dict(overlap=-1)):
        with pytest.raises(ValueError):
            jno.precond.schwarz(**kw)
    op = LinearOperator.from_matvec(lambda v: 2 * v, shape=(4, 4))
    with pytest.raises(TypeError, match="ASSEMBLED"):
        jno.precond.schwarz().materialize(PrecondContext(op, None))
