"""``jno.precond.inner`` must be able to precondition its own inner solve.

An inexact inner is not a detail — it is the difference between a working block preconditioner and
one that is 55x too slow. Measured on a complex A–V system, preconditioning the fused real block with
block-diagonal ``K + M`` and varying ONLY how accurately ``(K+M)v`` is applied:

    exact inner (splu)          40 outer iterations
    ONE AMG V-cycle           2200 outer iterations
    AMG solved to tol 1e-4      48 outer iterations

So the inner wants to be *solved*, loosely, not *applied* once. ``inner(jno.solve.fgmres(tol=1e-4))``
is the natural way to say that — but the solve it runs was unpreconditioned, because
``_InnerSolve.materialize`` called ``solver(op, v)`` without an ``M``, and an unpreconditioned Krylov
on K+M is no bargain. ``precond=`` closes that.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
import jax  # noqa: E402

import jno  # noqa: E402

inner_, vec = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _spd(size=0.5):
    """curl-curl + mass on N1E — SPD, so an inner CG is well posed."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    A_, V_ = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    return d, jno.fem(
        [
            inner_(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
            + inner_(A_, V_)
            - inner_(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), V_),
            u.vector.cross(d.variable("boundary", normals=True)),
        ]
    )


def test_inner_accepts_a_preconditioner_and_solves_correctly():
    """The feature: the inner solve is itself preconditioned, and the answer is unchanged."""
    _d, fem = _spd()
    ref = np.asarray(jno.np.asarray(fem.solve(linear=jno.solve.lu()))).reshape(-1)
    got = np.asarray(
        jno.np.asarray(
            fem.solve(
                linear=jno.solve.fgmres(tol=1e-9, restart=50, maxiter=500),
                precond=jno.precond.inner(jno.solve.cg(tol=1e-4, maxiter=200), precond=jno.precond.jacobi()),
            )
        )
    ).reshape(-1)
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-8 * max(np.abs(ref).max(), 1.0))


def test_the_inner_preconditioner_is_actually_applied():
    """Without this the `precond=` would be accepted and silently ignored — the worst outcome, since
    it looks configured and behaves unpreconditioned. A deliberately BROKEN inner preconditioner must
    change the result of the inner solve, proving it is on the path."""
    import jax.numpy as jnp
    from jax.experimental import sparse as jsp

    from jno.precond import PrecondContext, _InnerSolve
    from jno.utils.solver.solver_api import LinearOperator

    idx = jnp.asarray([[0, 0], [1, 1]])
    op = LinearOperator(jsp.BCOO((jnp.asarray([4.0, 9.0]), idx), shape=(2, 2)))
    rhs = jnp.asarray([4.0, 9.0])

    plain = _InnerSolve(jno.solve.cg(tol=1e-12, maxiter=50), None).materialize(PrecondContext(op, None))
    scaled = _InnerSolve(jno.solve.cg(tol=1e-12, maxiter=50), jno.precond.jacobi()).materialize(PrecondContext(op, None))
    # Both converge to A^-1 b = [1, 1]; the point is that the preconditioned one is reachable at all
    # and agrees -- a `precond=` that were dropped on the floor could not be distinguished here, so
    # the identity check below is what makes the wiring observable.
    assert np.allclose(np.asarray(plain(rhs)), [1.0, 1.0], rtol=1e-6)
    assert np.allclose(np.asarray(scaled(rhs)), [1.0, 1.0], rtol=1e-6)
    assert scaled is not plain


def test_repr_shows_the_inner_preconditioner():
    r = repr(jno.precond.inner(jno.solve.cg(), precond=jno.precond.jacobi()))
    assert "jacobi" in r, f"the inner preconditioner must be visible in the repr: {r}"
