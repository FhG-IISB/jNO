"""A GenEO space that follows the operator: ``reuse=k`` on the coarse or Schwarz spec that holds it.

Default (no ``reuse``): the space is computed once, at setup; the coarse matrix and the local solves follow the
operator. ``reuse=k``: the space itself is recomputed every k-th solve; ``reuse=1`` also at every linearization
inside a compiled loop -- a Newton step, a staggered block step -- from the tangent and the iterate it was
assembled at.

Oracles: the preconditioner after a recompute equals a fresh spec's on that operator, and a kept one equals
the stale spec's (a structure that never changes gives both the same partition); inside Newton and the
staggered loop, the recompute runs on the traced tangent, with the other block's unknowns held out, and the
answer is the direct solve's; a cadence a single compiled loop cannot honour raises.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

pytest.importorskip("pymetis", reason="Schwarz subdomains use METIS: the [metis] extra")

R = jno.shape.rect


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _plate(h=0.1):
    d = R(0.0, 0.0, 1.0, 1.0).sized(h).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    return d, (x, y), (xb, yb)


def _elasticity(d, X, B, g, nonlinear=False):
    x, y = X
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    tr = jno.np.trace
    u, v = d.fem_symbols(value_shape=(2,))
    u_, v_ = u.bind(x=x, y=y), v.bind(x=x, y=y)
    # nonlinear: the energy eps:eps + (c/2)(eps:eps)^2 stiffens where it strains, so the tangent's stiff region
    # moves with u; it stays symmetric positive definite (a convex energy)
    shear = 2.0 + (2e4 * ddot(ε(u_), ε(u_)) if nonlinear else 0.0)
    return jno.fem([g * (shear * ddot(ε(u_), ε(v_)) + 1.5 * tr(ε(u_)) * tr(ε(v_))) - 0.1 * v_[1], u(*B) - (0.0, 0.0)])


def _band(X, at=0.5, width=0.05):
    return (1.0 - jno.np.exp(-jno.np.abs(X[1] - at) / width)) ** 2 + 1e-6


def _matrix(spec, fem):
    spec.prepare(fem)
    A = fem._op[0]
    ap = materialize_precond(spec, PrecondContext(LinearOperator(A), fem))
    return np.asarray(jax.jit(jax.vmap(ap, in_axes=1, out_axes=1))(jnp.eye(A.shape[0])))


def _bnn(reuse=None):
    S = jno.precond.schwarz(parts=6, coarse=False)
    kw = {} if reuse is None else {"reuse": reuse}
    Q = jno.precond.coarse(jno.space.geneo(nev=6), parts=S, **kw)
    return Q >> S >> Q


@pytest.mark.parametrize("reuse", [1, 2])
def test_reuse_k_recomputes_the_space_every_kth_solve(reuse):
    d, X, B = _plate()
    intact, cracked, moved = (_elasticity(d, X, B, g) for g in (1.0, _band(X), _band(X, at=0.3)))
    follow = _bnn(reuse)
    _matrix(follow, intact)  # set up on the intact plate
    B1 = _matrix(follow, cracked)
    B2 = _matrix(follow, moved)
    fresh1, fresh2 = _matrix(_bnn(), cracked), _matrix(_bnn(), moved)
    stale = _bnn()
    _matrix(stale, intact)
    tol = dict(rtol=1e-9, atol=1e-9 * np.abs(fresh1).max())
    if reuse == 1:  # every solve
        np.testing.assert_allclose(B1, fresh1, **tol)
        np.testing.assert_allclose(B2, fresh2, **tol)
    else:  # the second solve keeps the setup's space, the third recomputes
        np.testing.assert_allclose(B1, _matrix(stale, cracked), **tol)
        np.testing.assert_allclose(B2, fresh2, **tol)
    assert np.abs(B1 - _matrix(stale, cracked)).max() > 1e-6 * np.abs(B1).max() or reuse == 2


def test_the_space_follows_the_tangent_inside_newton(monkeypatch):
    from jno.utils.solver import geneo

    d, X, B = _plate()
    fem = _elasticity(d, X, B, 1.0, nonlinear=True)
    ref = np.asarray(fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.lu()))
    seen = []
    real = geneo.geneo_values

    def spy(plan, pat, A, vals, frozen=None):
        seen.append(isinstance(vals, jax.core.Tracer))
        return real(plan, pat, A, vals, frozen)

    monkeypatch.setattr(geneo, "geneo_values", spy)
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.cg(tol=1e-10), precond=_bnn(1)))
    assert any(seen), "the space was never recomputed from a traced tangent"
    np.testing.assert_allclose(sol, ref, rtol=0, atol=1e-6 * np.abs(ref).max())  # both within Newton's tolerance


def test_a_cadence_one_compiled_loop_cannot_honour_raises():
    d, X, B = _plate(0.2)
    fem = _elasticity(d, X, B, 1.0, nonlinear=True)
    with pytest.raises(NotImplementedError, match="reuse=3"):
        fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.cg(), precond=_bnn(3))


def test_a_staggered_block_step_holds_the_other_block_out(monkeypatch):
    from jno.utils.solver import geneo

    d, (x, y), (xb, yb) = _plate(0.15)
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    grad = lambda w: jno.np.grad(w, [x, y])  # noqa: E731
    u, v = d.fem_symbols(value_shape=(2,))
    c, q = d.fem_symbols()
    u_, v_, c_, q_ = u.bind(x=x, y=y), v.bind(x=x, y=y), c.bind(x=x, y=y), q.bind(x=x, y=y)
    g = (1.0 - c_) ** 2 + 1e-3  # a field degrading the stiffness, solved for alongside it
    fem = jno.fem(
        [
            g * (2.0 * ddot(ε(u_), ε(v_)) + 1.5 * jno.np.trace(ε(u_)) * jno.np.trace(ε(v_))) - 0.1 * v_[1],
            0.05 * jno.np.inner(grad(c_), grad(q_), 1) + c_ * q_ - 0.5 * jno.np.exp(-(((y - 0.5) / 0.1) ** 2)) * q_,
            u(xb, yb) - (0.0, 0.0),
        ]
    )
    alt = jno.solve.staggered([u, c], direct=True, rtol=1e-9, atol=1e-11)
    ref = np.asarray(fem.solve(nonlinear=alt, linear=jno.solve.lu()))
    held = []
    real = geneo.geneo_values

    def spy(plan, pat, A, vals, fz=None):
        held.append(None if fz is None else int(np.asarray(fz).sum()))
        return real(plan, pat, A, vals, fz)

    monkeypatch.setattr(geneo, "geneo_values", spy)
    sol = np.asarray(fem.solve(nonlinear=alt, linear=jno.solve.cg(tol=1e-11), precond=_bnn(1)))
    # each block step holds the OTHER field out: the u step the c unknowns, the c step the u unknowns (the
    # implicit-gradient solves at the root use the full tangent and hold nothing)
    nc = fem.dofs // 3
    assert {nc, 2 * nc} <= {h for h in held if h is not None}, held
    np.testing.assert_allclose(sol, ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max())
