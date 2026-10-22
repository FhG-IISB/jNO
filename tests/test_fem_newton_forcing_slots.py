"""Inexact Newton reaches a linear solver configured through the slots, not only the default one.

The forcing term (Eisenstat & Walker, SIAM J. Sci. Comput. 17 (1996) 16, choice 2) asks each step's linear
solve for ``rtol=eta``. Only the default Jacobi-BiCGStab took a per-call tolerance, so with ``linear=`` /
``precond=`` every Newton step was solved to the slot's own ``tol`` -- measured 6x the Krylov iterations for a
residual 100x below the bound anyone asked for. The contract is the outer one, ``||r|| <= atol + rtol ||r0||``,
so the oracle is that bound, the Krylov count with the forcing on and off, and the root itself.
"""

from __future__ import annotations

import jax
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


def _hyperelastic():
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.25).domain()
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


def _nonlinear_diffusion():
    d = jno.shape.rect(0, 0, 1, 1, size=0.05).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    return jno.fem([(1.0 + 10.0 * ub * ub) * (ub.x * vb.x + ub.y * vb.y) - 10.0 * vb, u(xb, yb) - 0.0])


def _solve(build, slots, forcing: bool, monkeypatch):
    """Solve with the forcing on or off; return the root, the Newton verdict and the total Krylov iterations
    (counted here: ``fem.stats`` does not see the inner solves of a compiled Newton loop)."""
    import jno.utils.solver.krylov as kr
    import jno.utils.solver.newton_krylov as nk

    monkeypatch.setattr(nk, "_FORCING", [forcing])
    counted = []
    for name in ("cg_counted", "bicgstab_counted"):
        real = getattr(kr, name)

        def counting(*a, _real=real, **kw):
            out = _real(*a, **kw)
            jax.debug.callback(lambda kk: counted.append(int(kk)), out[1])
            return out

        monkeypatch.setattr(kr, name, counting)
    fem = build()
    u = np.asarray(fem.solve(**slots()))
    jax.effects_barrier()
    return u, dict(fem.stats["nonlinear"]), sum(counted)


CASES = [
    (_hyperelastic, lambda: dict(linear=jno.solve.cg(), precond=jno.precond.jacobi())),
    (_nonlinear_diffusion, lambda: dict(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi())),
    (_nonlinear_diffusion, lambda: dict(linear=jno.solve.cg(), precond=jno.precond.amg())),
]


@pytest.mark.parametrize(
    "build,slots", CASES, ids=["hyperelastic-cg-jacobi", "diffusion-bicgstab-jacobi", "diffusion-cg-amg"]
)
def test_forcing_reaches_a_slot_composed_krylov_solver(build, slots, monkeypatch):
    # Break: the composed inner solve taking no per-call tolerance (every step to the slot's tol=1e-8), or a
    # Krylov loop that ignores the tolerance it was handed.
    u_ref, st_ref, k_ref = _solve(build, slots, False, monkeypatch)
    u, st, k = _solve(build, slots, True, monkeypatch)
    assert st["converged"] and st["residual"] <= st["bound"], st  # the contract
    assert k < 0.8 * k_ref, f"Krylov iterations {k} vs {k_ref} without forcing"
    assert st["steps"] <= st_ref["steps"] + 2, (st, st_ref)
    # Both are roots to within the same outer bound; their distance is bounded by that tolerance, not 0.
    assert np.abs(u - u_ref).max() <= 1e-4 * np.abs(u_ref).max()


def test_a_direct_linear_slot_is_still_solved_exactly(monkeypatch):
    # A direct solver has no tolerance to loosen: it takes no rtol and Newton solves each step exactly.
    import jno.utils.solver.newton_krylov as nk

    monkeypatch.setattr(nk, "_FORCING", [True])
    assert not getattr(jno.solve.lu(), "accepts_rtol", False)
    fem = _nonlinear_diffusion()
    u = np.asarray(fem.solve(linear=jno.solve.lu()))
    st = fem.stats["nonlinear"]
    assert st["converged"] and st["residual"] <= st["bound"], st
    ref = np.asarray(_nonlinear_diffusion().solve())
    assert np.abs(u - ref).max() <= 1e-4 * np.abs(ref).max()


def test_a_per_call_rtol_stops_a_krylov_solve_where_it_was_asked():
    # The tolerance is per call and may be traced; the spec's own tol applies when none is given.
    import jax.numpy as jnp

    n = 200
    A = jnp.diag(jnp.linspace(1.0, 1e3, n)) + jnp.diag(-0.4 * jnp.ones(n - 1), 1) + jnp.diag(-0.4 * jnp.ones(n - 1), -1)
    b = jnp.ones(n)
    cg = jno.solve.cg()
    assert cg.accepts_rtol
    rel = lambda x: float(jnp.linalg.norm(A @ x - b) / jnp.linalg.norm(b))  # noqa: E731
    from jno.utils.solver.solver_api import gate_suspended

    with gate_suspended():  # a loose solve is below the residual gate on purpose, as inside Newton
        loose = jax.jit(lambda r: cg(A, b, rtol=r))(1e-2)
    assert 1e-6 < rel(loose) <= 1e-2  # stopped near what was asked, not at the spec's 1e-8
    assert rel(cg(A, b)) <= 1e-8
