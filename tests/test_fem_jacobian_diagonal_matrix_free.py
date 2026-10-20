"""The tangent's diagonal without the tangent, and the matrix-free Newton preconditioned by it.

A Jacobi preconditioner needs only diag J. The matrix-free Newton (``newton(direct=False)``) has no matrix,
so ``precond=jacobi()`` raised there and the default inner solve ran unpreconditioned -- 56x the assembled
path's time at 100k DOFs. The assembler now has a diagonal-only pass (the same element loop, keeping each
block's diagonal entries, scattered into one vector). Oracles: the diagonal of the ASSEMBLED tangent (the
compressed, Dirichlet-eliminated matrix the default Newton factors), the default solve's root, the Krylov
iteration count with and without the diagonal, and the default solve's gradient.
"""

from __future__ import annotations

import jax
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


def _hyperelastic(size=0.25):
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    x, y, z = d.variable("interior", split=True)[:3]
    xl, yl, zl = d.variable("left", split=True)[:3]
    u, phi = d.fem_symbols(value_shape=(3,))
    X = [x, y, z]
    Id = jno.np.identity(3)
    F = Id + jno.np.grad(u, X)
    E = 0.5 * (jno.np.transpose(F) @ F - Id)
    S = 1.0 * jno.np.trace(E) * Id + E
    return jno.fem(
        [jno.np.double_dot(F @ S, jno.np.grad(phi, X)) - 0.1 * phi.bind(x=x, y=y, z=z)[2], u(xl, yl, zl) - (0.0, 0.0, 0.0)]
    )


def _diffusion_with_robin(k=None):
    # a nonlinear BOUNDARY term too (u^3 on the right edge), so the surface blocks reach the diagonal
    d = jno.shape.rect(0, 0, 1, 1, size=0.08).domain()
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    xr, yr = d.variable("right", split=True)[:2]
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    ur, vr = u.bind(x=xr, y=yr), v.bind(x=xr, y=yr)
    src = 10.0 if k is None else 10.0 * k
    return jno.fem([(1.0 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - src * vb, ur**3 * vr, u(xl, yl) - 0.0])


@pytest.mark.parametrize("build", [_hyperelastic, _diffusion_with_robin], ids=["hyperelastic-3d", "diffusion-robin-2d"])
def test_the_diagonal_pass_is_the_assembled_tangents_diagonal(build):
    # Break: a block entry kept off the diagonal, a surface block dropped, a duplicate (shared DOF) not summed,
    # or the Dirichlet rows not set to the eliminated matrix's unit diagonal.
    fem = build()
    op = fem._op
    u = jnp.asarray(np.random.default_rng(0).normal(scale=0.05, size=fem.dofs))
    ref = np.asarray(op.jacobian(u, None).todense()).diagonal()
    got = np.asarray(op.jacobian_diag(u, None))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def _count_krylov(monkeypatch):
    import jno.utils.solver.krylov as kr
    import jno.utils.solver.newton_krylov as nk

    counted = []

    def hook(real, pos):
        def f(*a, **kw):
            out = real(*a, **kw)
            jax.debug.callback(lambda kk: counted.append(int(kk)), out[pos])
            return out

        return f

    monkeypatch.setattr(kr, "bicgstab_counted", hook(kr.bicgstab_counted, 1))
    real_nk = nk.bicgstab
    monkeypatch.setattr(nk, "bicgstab", lambda *a, **kw: hook(real_nk, 1)(*a, **{**kw, "return_iters": True}))
    return counted


def _graded_diffusion():
    # a coefficient varying 1000x across the square: the case diagonal scaling exists for
    d = jno.shape.rect(0, 0, 1, 1, size=0.04).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    kappa = (1.0 + 999.0 * x * x) * (1.0 + ub * ub)
    return jno.fem([kappa * (ub.x * vb.x + ub.y * vb.y) - 10.0 * vb, u(xb, yb) - 0.0])


@pytest.mark.parametrize("build", [_graded_diffusion, _hyperelastic], ids=["graded-diffusion-2d", "hyperelastic-3d"])
def test_matrix_free_newton_is_preconditioned_by_the_diagonal(build, monkeypatch):
    # Break: the diagonal not reaching the matrix-free inner solve (the iteration count stays put).
    ref = np.asarray(build().solve())
    totals = {}
    for with_diag in (False, True):
        counted = _count_krylov(monkeypatch)
        fem = build()
        if not with_diag:
            fem._op.jacobian_diag = None
        u = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=False)))
        jax.effects_barrier()
        st = fem.stats["nonlinear"]
        assert st["converged"] and st["residual"] <= st["bound"], st
        assert np.abs(u - ref).max() <= 1e-5 * np.abs(ref).max()  # the same root, to the Newton contract
        totals[with_diag] = sum(counted)
        monkeypatch.undo()
    # graded: Jacobi removes the coefficient's 1000x spread; uniform 3-D elasticity: a modest gain only
    assert totals[True] < (0.5 if build is _graded_diffusion else 0.9) * totals[False], totals


def test_a_composed_jacobi_works_on_the_matrix_free_newton():
    # Break: `jno.precond.jacobi()` on a matvec-only tangent raising "no diagonal" (what it did before).
    ref = np.asarray(_diffusion_with_robin().solve())
    fem = _diffusion_with_robin()
    u = np.asarray(
        fem.solve(nonlinear=jno.solve.newton(direct=False), linear=jno.solve.bicgstab(), precond=jno.precond.jacobi())
    )
    st = fem.stats["nonlinear"]
    assert st["converged"] and st["residual"] <= st["bound"], st
    assert np.abs(u - ref).max() <= 1e-6 * np.abs(ref).max()


def test_gradient_through_the_preconditioned_matrix_free_newton():
    # The tangent (adjoint) solve uses the diagonal at the start iterate; the gradient must not change.
    d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    k = jno.np.parameter((1,), name="k")
    fem = jno.fem([(1.0 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - k * vb, u(xb, yb) - 0.0])
    mf = fem.solve(nonlinear=jno.solve.newton(direct=False))
    df = fem.solve()
    kk = jnp.asarray([3.0])
    g_mf = jax.grad(lambda a: mf.fn(a).sum())(kk)
    g_df = jax.grad(lambda a: df.fn(a).sum())(kk)
    assert abs(float(g_df[0])) > 1e-3
    np.testing.assert_allclose(np.asarray(g_mf), np.asarray(g_df), rtol=1e-6)
