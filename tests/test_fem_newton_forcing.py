"""The default Newton solves each step's linear system only as accurately as that step needs.

Inexact Newton (Dembo, Eisenstat & Steihaug, SIAM J. Numer. Anal. 19 (1982) 400) with the forcing term of
Eisenstat & Walker (SIAM J. Sci. Comput. 17 (1996) 16, choice 2): far from the root a step's linear solve
to 1e-10 buys nothing the next Newton step does not redo. The CONTRACT is unchanged -- the loop still stops
on the true residual, ``||r|| <= atol + rtol ||r0||`` -- so the oracle is that bound, the Krylov count, and
a Newton count that does not grow beyond the quadratic phase's usual one or two extra steps.
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


def _solve(build, forcing: bool, monkeypatch):
    """Solve with the forcing on or off; return the root, the Newton verdict, and the total BiCGStab
    iterations (counted here: ``fem.stats`` does not record the inner solves of a compiled Newton loop)."""
    import jno.utils.solver.krylov as kr
    import jno.utils.solver.newton_krylov as nk

    monkeypatch.setattr(nk, "_FORCING", [forcing])
    counted = []
    real = kr.bicgstab_counted

    def bicgstab_counted(*a, **kw):
        x, k, broke = real(*a, **kw)
        jax.debug.callback(lambda kk: counted.append(int(kk)), k)
        return x, k, broke

    monkeypatch.setattr(kr, "bicgstab_counted", bicgstab_counted)
    fem = build()
    u = np.asarray(fem.solve())
    jax.effects_barrier()
    return u, dict(fem.stats["nonlinear"]), sum(counted)


@pytest.mark.parametrize("build", [_hyperelastic, _nonlinear_diffusion], ids=["hyperelastic-3d", "nonlinear-diffusion-2d"])
def test_forcing_cuts_krylov_work_and_keeps_the_newton_contract(build, monkeypatch):
    # Break: no forcing (every step to 1e-10), a forcing that never tightens (the bound is then missed or
    # Newton needs many more steps), or a final step solved looser than the outer bound needs.
    u_ref, st_ref, k_ref = _solve(build, False, monkeypatch)
    u, st, k = _solve(build, True, monkeypatch)
    assert st["converged"] and st["residual"] <= st["bound"], st  # the contract
    assert k < 0.8 * k_ref, f"Krylov iterations {k} vs {k_ref} without forcing"
    assert st["steps"] <= st_ref["steps"] + 2, (st, st_ref)
    # Both are roots to within the same outer bound; their distance is bounded by that tolerance, not 0.
    assert np.abs(u - u_ref).max() <= 1e-4 * np.abs(u_ref).max()


def test_toggling_the_forcing_does_not_reuse_the_compiled_solve():
    # Break: the compiled-solve cache keyed without the forcing switch -- the second solve then reruns the
    # program traced with exact steps and lands on its root to roundoff (an A/B in one process measured nothing).
    import jno.utils.solver.newton_krylov as nk

    fem = _nonlinear_diffusion()
    slots = lambda: dict(  # noqa: E731
        nonlinear=jno.solve.newton(direct=True), linear=jno.solve.fgmres(tol=1e-12), precond=jno.precond.jacobi()
    )
    prev = nk._FORCING[0]
    try:
        nk._FORCING[0] = False
        exact = np.asarray(fem.solve(**slots()))
        nk._FORCING[0] = True
        inexact = np.asarray(fem.solve(**slots()))
    finally:
        nk._FORCING[0] = prev
    diff = np.abs(exact - inexact).max() / np.abs(exact).max()
    # a rerun differs by GPU scatter roundoff (~1e-16); an inexact root by up to its outer tolerance
    assert diff > 1e-13, f"the forced solve reran the exact-step program ({diff:.1e} apart)"
    assert diff <= 1e-4  # both within the outer bound
