"""GenEO on a march: the step operator ``a M + b J`` is what the local eigenproblems are measured against.

The per-cell matrices of a transient block are the mass's and the spatial tangent's, combined as the step combines
them. Oracles: summed, they are the step operator (constant-operator theta step); on a stiff diffusion step with a
coefficient jump GenEO beats one-level Schwarz in CG iterations and the march keeps the sparse-direct answer; under
``reuse=1`` the space is recomputed from the traced tangent at every Newton step of a nonlinear march, whose answer is
the direct one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import jno
from jno.utils.solver.solver_api import LinearOperator, PrecondContext, _add_step_operator, materialize_precond, step_form

pytest.importorskip("pymetis", reason="Schwarz subdomains use METIS: the [metis] extra")

R = jno.shape.rect
DT, STEPS = 100.0, 3


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _layered_heat(contrast, h=0.06, dt=None):
    """u_t = div(k grad u) + 1 on four layers of conductivity 1 and 1/contrast, cold at the bottom. The default step
    is far longer than the diffusion time across a cell, so the operator is mostly the stiffness."""
    dt = DT if dt is None else dt
    soft = 1.0 / contrast
    d = (
        (
            R(0, 0.00, 2, 0.25).name("l0").attach(k=1.0)
            + R(0, 0.25, 2, 0.50).name("l1").attach(k=soft)
            + R(0, 0.50, 2, 0.75).name("l2").attach(k=1.0)
            + R(0, 0.75, 2, 1.00).name("l3").attach(k=soft)
        )
        .sized(h)
        .domain(time=(0.0, dt * STEPS, STEPS + 1))
    )
    x, y, t = d.variable("interior", split=True)
    u, w = d.fem_symbols()
    ui, wi = u.bind(x=x, y=y, t=t), w.bind(x=x, y=y, t=t)
    xb, yb, _tb = d.variable("bottom", split=True)
    return jno.fem(
        [ui.t * wi + d.k * (ui.x * wi.x + ui.y * wi.y) - wi, u(xb, yb) - 0.0, u(*d.variable("initial", split=True)) - 0.0]
    )


def test_the_cells_of_a_march_sum_to_its_step_operator():
    from jno.utils.solver.deflation import element_triplets

    fem = _layered_heat(1e3, h=0.15)
    block = fem._op
    s = 0.7 * float(block.dt)
    with step_form(1.0, s, 0.0):
        r, c, v, _cell = (np.asarray(a) for a in element_triplets(fem))
    n = int(block.M.shape[0])
    K = sp.coo_matrix((v, (r, c)), shape=(n, n)).toarray()
    Ad = np.asarray(_add_step_operator(block.M, block.A, s).todense())
    free = np.abs(Ad - np.diag(np.diag(Ad))).sum(axis=1) > 0  # rows with couplings: not eliminated
    np.testing.assert_allclose(K[np.ix_(free, free)], Ad[np.ix_(free, free)], rtol=1e-10, atol=1e-10 * np.abs(Ad).max())


def _value(sol):
    """A march is a lazy node where a parameter may flow through it; ``.fn()`` evaluates it."""
    return np.asarray(sol.fn() if hasattr(sol, "fn") else sol)


def _cg_count(A, B, b):
    its = []
    x, info = spla.cg(
        sp.csr_matrix(A),
        b,
        M=spla.LinearOperator(A.shape, matvec=lambda r: B @ r),
        rtol=1e-8,
        maxiter=2000,
        callback=lambda _x: its.append(1),
    )
    assert info == 0
    return len(its)


def test_geneo_on_a_march_beats_one_level_and_keeps_the_answer():
    fem = _layered_heat(1e4, h=0.03)
    ref = _value(fem.solve(linear=jno.solve.lu()))
    S = jno.precond.schwarz(parts=16, coarse=False)
    Q = jno.precond.coarse(jno.space.geneo(), parts=S)
    M = Q >> S >> Q
    sol = _value(fem.solve(linear=jno.solve.cg(tol=1e-10, maxiter=2000), precond=M))
    np.testing.assert_allclose(sol, ref, rtol=0, atol=1e-6 * np.abs(ref).max())

    block = fem._op
    Astep = _add_step_operator(block.M, block.A, float(block.dt))
    n = int(Astep.shape[0])
    eye = jnp.eye(n)
    ctx = PrecondContext(LinearOperator(Astep), fem)
    B2 = np.asarray(jax.vmap(materialize_precond(M, ctx), in_axes=1, out_axes=1)(eye))
    B1 = np.asarray(jax.vmap(materialize_precond(S, ctx), in_axes=1, out_axes=1)(eye))
    Ad, b = np.asarray(Astep.todense()), np.asarray(jax.random.normal(jax.random.PRNGKey(0), (n,)))
    one, two = _cg_count(Ad, B1, b), _cg_count(Ad, B2, b)
    assert two < 0.6 * one, (one, two)


def test_the_space_follows_the_tangent_inside_a_nonlinear_march(monkeypatch):
    from jno.utils.solver import geneo

    d = R(0.0, 0.0, 1.0, 1.0).sized(0.12).domain(time=(0.0, 0.2, 3))
    x, y, t = d.variable("interior", split=True)
    u, v = d.fem_symbols(value_shape=(2,))
    u_, v_ = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    shear = 2.0 + 2e4 * ddot(ε(u_), ε(u_))  # a convex energy: the tangent stays symmetric positive definite
    xb, yb, _tb = d.variable("bottom", split=True)
    terms = [
        u_.t[0] * v_[0]
        + u_.t[1] * v_[1]
        + shear * ddot(ε(u_), ε(v_))
        + 1.5 * jno.np.trace(ε(u_)) * jno.np.trace(ε(v_))
        - 0.1 * v_[1],
        u(xb, yb) - (0.0, 0.0),
        u(*d.variable("initial", split=True)) - (0.0, 0.0),
    ]
    fem = jno.fem(terms)
    ref = _value(fem.solve(nonlinear=jno.solve.newton(direct=True), linear=jno.solve.lu()))
    seen = []
    real = geneo.geneo_values

    def spy(plan, pat, A, vals, frozen=None):
        seen.append(isinstance(vals, jax.core.Tracer))
        return real(plan, pat, A, vals, frozen)

    monkeypatch.setattr(geneo, "geneo_values", spy)
    S = jno.precond.schwarz(parts=6, coarse=False)
    Q = jno.precond.coarse(jno.space.geneo(nev=6), parts=S, reuse=1)
    sol = _value(fem.solve(nonlinear=jno.solve.newton(direct=True), linear=jno.solve.cg(tol=1e-10), precond=Q >> S >> Q))
    assert any(seen), "the space was never recomputed from a traced tangent"
    np.testing.assert_allclose(sol, ref, rtol=0, atol=1e-5 * np.abs(ref).max())


def test_a_step_too_short_to_need_a_coarse_space_gets_an_empty_one():
    fem = _layered_heat(1e4, h=0.1, dt=1e-7)
    ref = _value(fem.solve(linear=jno.solve.lu()))
    S = jno.precond.schwarz(parts=6, coarse=False)
    G = jno.space.geneo()
    Q = jno.precond.coarse(G, parts=S)
    sol = _value(fem.solve(linear=jno.solve.cg(tol=1e-10, maxiter=500), precond=Q >> S >> Q))
    assert int(G.counts.sum()) == 0
    np.testing.assert_allclose(sol, ref, rtol=0, atol=1e-6 * np.abs(ref).max())
