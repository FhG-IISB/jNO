"""jno.space: computed spaces for the space-taking consumers (coarse, amg(nullspace=), eigs(deflate=)).

Oracles: ``eigvecs(k)`` spans the ``k`` lowest eigenvectors of the operator the solve runs on (projector
distance against ``numpy.linalg.eigh``), and deflating with it removes exactly those eigenvalues from the
preconditioned operator (Tang, Nabben, Vuik & Erlangga 2009, Sec. 3); refusals for what is not supported.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(size=0.12):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - vi, u(xb, yb) - 0.0])


def test_eigvecs_spans_the_lowest_eigenvectors():
    fem = _poisson()
    k = 4
    Q = jno.precond.coarse(jno.space.eigvecs(k))
    Q.prepare(fem)
    Z = np.asarray(Q._pattern.null)
    A = np.asarray(fem._op[0].todense())
    _lam, V = np.linalg.eigh(A)
    # the operator's lowest eigenvectors include Dirichlet identity rows only above them on this problem
    Pz = Z @ np.linalg.pinv(Z)
    Pv = V[:, :k] @ V[:, :k].T
    assert np.abs(Pz - Pv).max() < 1e-5


def test_deflating_with_eigvecs_removes_those_eigenvalues():
    from jno.precond import _Spec

    class _Identity(_Spec):
        traceable = True
        key = ("identity",)

        def materialize(self, ctx):
            return lambda v: v

    fem = _poisson()
    A = np.asarray(fem._op[0].todense())
    lam = np.linalg.eigvalsh(A)
    k = 4
    spec = _Identity() >> jno.precond.coarse(jno.space.eigvecs(k))
    spec.prepare(fem)
    ap = materialize_precond(spec, PrecondContext(LinearOperator(fem._op[0]), fem))
    B = np.column_stack([np.asarray(ap(jnp.asarray(e))) for e in np.eye(A.shape[0])])
    got = np.sort(np.linalg.eigvals(B @ A).real)
    np.testing.assert_allclose(got, np.sort(np.concatenate([np.ones(k), lam[k:]])), rtol=1e-4, atol=1e-4)
    ref = np.asarray(fem.solve())
    sol = np.asarray(
        fem.solve(
            linear=jno.solve.fgmres(tol=1e-11), precond=jno.precond.jacobi() >> jno.precond.coarse(jno.space.eigvecs(k))
        )
    )
    np.testing.assert_allclose(sol, ref, rtol=1e-7, atol=1e-10)


def test_refusals():
    fem = _poisson(0.3)
    n = fem._op[0].shape[0]
    with pytest.raises(RuntimeError, match="LOBPCG"):
        jno.precond.coarse(jno.space.eigvecs(3, maxiter=1)).prepare(fem)
    mv_only = LinearOperator.from_matvec(lambda v: fem._op[0] @ v, shape=(n, n))
    with pytest.raises(ValueError, match="operator"):
        jno.precond.coarse(jno.space.eigvecs(2)).build(mv_only, fem)

    class _Recycled:
        def basis(self, ctx):
            return np.ones((ctx.n, 1))

        def update(self, *args):
            pass

    with pytest.raises(NotImplementedError, match="update"):
        jno.precond.coarse(_Recycled()).prepare(fem)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            jno.space.eigvecs(bad)
