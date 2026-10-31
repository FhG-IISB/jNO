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

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsparse

import jno
from jno.utils.solver.solver_api import _GATE_FAILURES, LinearOperator, PrecondContext, materialize_precond

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

    def spy(plan, pat, A, vals, frozen=None, **kw):
        seen.append(isinstance(vals, jax.core.Tracer))
        return real(plan, pat, A, vals, frozen, **kw)

    monkeypatch.setattr(geneo, "geneo_values", spy)
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.cg(tol=1e-10), precond=_bnn(1)))
    assert any(seen), "the space was never recomputed from a traced tangent"
    np.testing.assert_allclose(sol, ref, rtol=0, atol=1e-6 * np.abs(ref).max())  # both within Newton's tolerance


def test_a_saturated_space_says_so_once_not_at_every_newton_step(monkeypatch):
    d, X, B = _plate()
    fem = _elasticity(d, X, B, 1.0, nonlinear=True)
    said = []
    monkeypatch.setattr(logging.getLogger("jno"), "warning", lambda msg, *a, **k: said.append(msg))
    S = jno.precond.schwarz(parts=6, coarse=False)
    Q = jno.precond.coarse(jno.space.geneo(nev=1), parts=S, reuse=1)  # one column per subdomain: every one is full
    jax.block_until_ready(fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.cg(tol=1e-10), precond=Q >> S >> Q))
    jax.effects_barrier()
    assert fem.stats["nonlinear"]["steps"] > 1, "needs more than one recompute to mean anything"
    assert sum("Give a larger nev=" in m for m in said) == 1


def test_an_indefinite_tangent_at_a_recompute_is_named():
    """Setup refuses an operator it cannot factor; so must a recompute, or NaN spreads through the coarse level with
    no direction counted as bad."""
    d, X, B = _plate()
    fem = _elasticity(d, X, B, 1.0)
    S = jno.precond.schwarz(parts=6, coarse=False)
    Q = jno.precond.coarse(jno.space.geneo(nev=6), parts=S, reuse=1)
    M = Q >> S >> Q
    _matrix(M, fem)
    A = fem._op[0]
    with pytest.raises(ValueError, match="symmetric positive definite"):
        materialize_precond(M, PrecondContext(LinearOperator(-1.0 * A), fem))

    def traced(data):  # inside a compiled loop it cannot raise: the solve's gate does
        At = jsparse.BCOO((data, A.indices), shape=A.shape)
        return materialize_precond(M, PrecondContext(LinearOperator(At), fem))(jnp.ones(A.shape[0]))

    _GATE_FAILURES.clear()
    jax.block_until_ready(jax.jit(traced)(-A.data))
    jax.effects_barrier()
    assert any("symmetric positive definite" in m for m in _GATE_FAILURES)
    _GATE_FAILURES.clear()


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

    def spy(plan, pat, A, vals, fz=None, **kw):
        held.append(None if fz is None else int(np.asarray(fz).sum()))
        return real(plan, pat, A, vals, fz, **kw)

    monkeypatch.setattr(geneo, "geneo_values", spy)
    sol = np.asarray(fem.solve(nonlinear=alt, linear=jno.solve.cg(tol=1e-11), precond=_bnn(1)))
    # each block step holds the OTHER field out: the u step the c unknowns, the c step the u unknowns (the
    # implicit-gradient solves at the root use the full tangent and hold nothing)
    nc = fem.dofs // 3
    assert {nc, 2 * nc} <= {h for h in held if h is not None}, held
    np.testing.assert_allclose(sol, ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max())


def test_the_space_follows_the_tangent_on_a_periodic_tie(monkeypatch):
    from jno.utils.solver import geneo

    eps = 1e-9
    d = R(0.0, 0.0, 2.0, 1.0).sized(0.12).domain()
    d.tag("l", lambda x, y: x < eps)
    d.tag("r", lambda x, y: x > 2.0 - eps)
    d.tag("b", lambda x, y: (y < eps) & (x > eps))
    x, y, _ = d.variable("interior", split=True)
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    u, v = d.fem_symbols(value_shape=(2,))
    u_, v_ = u.bind(x=x, y=y), v.bind(x=x, y=y)
    shear = 2.0 + 2e4 * ddot(ε(u_), ε(u_))
    fem = jno.fem(
        [
            shear * ddot(ε(u_), ε(v_)) + 1.5 * jno.np.trace(ε(u_)) * jno.np.trace(ε(v_)) - 0.1 * v_[1],
            u(*d.variable("b", split=True)[:2]) - (0.0, 0.0),
            u(*d.variable("l", split=True)[:2]) - u(*d.variable("r", split=True)[:2]),
        ]
    )
    # a periodic nonlinear solve is a lazy node (it flows into crux); `.fn()` evaluates it
    ref = np.asarray(fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.lu()).fn())
    seen = []
    real = geneo.geneo_values

    def spy(plan, pat, A, vals, frozen=None, **kw):
        seen.append(isinstance(vals, jax.core.Tracer))
        return real(plan, pat, A, vals, frozen, **kw)

    monkeypatch.setattr(geneo, "geneo_values", spy)
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.cg(tol=1e-10), precond=_bnn(1)).fn())
    assert any(seen), "the space was never recomputed from a traced tangent"
    np.testing.assert_allclose(sol, ref, rtol=0, atol=1e-6 * np.abs(ref).max())


def _fracture(h=0.15, steps=3):
    """Second-order phase-field fracture of a notched plate: a load-path march with a history state, as in the
    tutorial, so its tangent reads the march's buffers."""
    n = jno.np
    E, nu = 210.0, 0.3
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    Gc, ell, eta = 2.7e-3, 0.3, 1e-6
    dom = (R(0.0, 0.0, 1.0, 1.0, size=h) - R(-0.01, 0.49, 0.5, 0.51, size=h)).domain(tau=(0.0, 1.0, steps))
    dom.tag("bot", lambda x, y: y < 1e-9)
    dom.tag("top", lambda x, y: y > 1 - 1e-9)
    co, cb, ct = (dom.variable(r, split=True) for r in ("interior", "bot", "top"))
    X = [co[0], co[1]]
    u, phi = dom.fem_symbols(value_shape=(2,))
    dm, q = dom.fem_symbols()
    Hs, _ = dom.fem_symbols(value_shape=())
    eps = lambda w: n.sym(n.grad(w, X))  # noqa: E731
    e_u = eps(u)
    tr_u = n.trace(e_u)
    dev = e_u - tr_u / 2 * n.identity(2)
    psi_p = 0.5 * (lam + mu) * n.maximum(tr_u, 0.0) ** 2 + mu * n.inner(dev, dev, 2)
    psi_m = 0.5 * (lam + mu) * n.minimum(tr_u, 0.0) ** 2
    sigma = n.diff(((1.0 - dm) ** 2 + eta) * psi_p + psi_m, e_u)
    fem = jno.fem(
        [
            n.inner(sigma, eps(phi), 2),
            (Gc / ell) * dm * q + Gc * ell * n.inner(n.grad(dm, X), n.grad(q, X), 1) - 2.0 * (1.0 - dm) * Hs.i(-1) * q,
            Hs.evolves(n.maximum(Hs.i(-1), psi_p)),
            u(*cb)[0] - 0.0,
            u(*cb)[1] - 0.0,
            u(*ct)[0] - 0.0,
            u(*ct)[1] - 2e-2 * ct[-1],
        ]
    )
    return fem, jno.solve.staggered([u, dm], direct=True, max_sweeps=100, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("reuse", [None, 1], ids=["once", "follow"])
def test_geneo_on_a_fracture_march_reads_the_marchs_history(reuse, monkeypatch):
    from jno.utils.solver import geneo

    fem, alt = _fracture()
    ref = np.asarray(fem.solve(nonlinear=alt, linear=jno.solve.lu()))
    seen = []
    real = geneo.geneo_values

    def spy(plan, pat, A, vals, frozen=None, **kw):
        seen.append((isinstance(vals, jax.core.Tracer), frozen is not None))
        return real(plan, pat, A, vals, frozen, **kw)

    monkeypatch.setattr(geneo, "geneo_values", spy)
    fem, alt = _fracture()
    sol = np.asarray(fem.solve(nonlinear=alt, linear=jno.solve.cg(tol=1e-9, maxiter=2000), precond=_bnn(reuse)))
    if reuse:
        assert any(t and f for t, f in seen), "no block step recomputed the space from a traced tangent"
    else:
        assert not any(t for t, _f in seen), "the space was to be computed once"
    np.testing.assert_allclose(sol, ref, rtol=0, atol=1e-6 * np.abs(ref).max())


def test_one_level_schwarz_sets_up_on_a_march_with_history():
    fem, alt = _fracture()
    ref = np.asarray(fem.solve(nonlinear=alt, linear=jno.solve.lu()))
    fem, alt = _fracture()
    sol = np.asarray(
        fem.solve(
            nonlinear=alt, linear=jno.solve.cg(tol=1e-9, maxiter=2000), precond=jno.precond.schwarz(parts=6, coarse=False)
        )
    )
    np.testing.assert_allclose(sol, ref, rtol=0, atol=1e-6 * np.abs(ref).max())


def test_a_design_parameter_with_no_values_yet_is_refused_by_name():
    d, (x, y), (xb, yb) = _plate(0.2)
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    rho = jno.np.parameter(s, name="rho")
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    u, v = d.fem_symbols(value_shape=(2,))
    u_, v_ = u.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([(1e-3 + rho**3) * jno.np.inner(ε(u_), ε(v_), n_contract=2) - 0.1 * v_[1], u(xb, yb) - (0.0, 0.0)])
    ncells = int(np.asarray(d.mesh.cells_dict["triangle"]).shape[0])
    A = fem._op.evaluate({"rho": jnp.full((ncells,), 0.5)})[0]
    S = jno.precond.schwarz(parts=4, coarse=False)
    with pytest.raises(ValueError, match="runtime parameters"):
        jno.precond.coarse(jno.space.geneo(), parts=S).build(A, fem)
