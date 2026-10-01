"""A weak term is evaluated with its test function as a seeded trial field, where it is provably linear.

The evaluator carried the test basis through every product as an extra axis (n_local x n_comp one-hot
columns for a vector field), evaluating each term once per test DOF. A term linear in ``v`` is
``r_a = ∂/∂s_a Σ_q w_q I(s)`` with ``v = Σ_a φ_a s_a``: one reverse pass, the integrand a scalar per point.
Measured on a 3-D residual-based VMS residual (442k DOFs, RTX 3070): 71.7 -> 53.6 ms.

Oracles: the seeded and the per-DOF evaluation assemble the same residual and the same tangent on forms
that read the test function every way the evaluator knows -- value, gradient, a P2 Hessian, a component
gradient, a ``cellwise`` projection, a ``where`` selection; a test function under ``lag`` (a stop-gradient,
whose gradient would be zero) keeps its load, so ``u - lag(v)`` still projects 1 onto 1; and a term not
linear in ``v`` keeps the per-DOF number.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
import jno.utils.solver.fem_native as fn
import jno.utils.solver.fem_utils as fu
import jno.utils.solver.time_route as tr


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


grad, inner, lap, trace, where = jno.np.grad, jno.np.inner, jno.np.laplacian, jno.np.trace, jno.np.where
dot = lambda a, b: inner(a, b, n_contract=1)  # noqa: E731
ddot = lambda A, B: inner(A, B, n_contract=2)  # noqa: E731


def _flow():
    """A stabilised Navier-Stokes residual (P1/P1, 3-D): vector and scalar test functions, value and
    gradient, products with tau(u) and the strong residual."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=3).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    x, y, z = d.variable("interior", split=True)[:3]
    X = [x, y, z]
    ui, vi, pi, qi = (f.bind(x=x, y=y, z=z) for f in (u, v, p, q))
    gu, gv, gp, gq = grad(u, X), grad(v, X), grad(p, X), grad(q, X)
    G = d.cell_metric
    tau = (1.0 + dot(ui, dot(G, ui))) ** -0.5
    R = dot(gu, ui) - 0.01 * lap(u, X) + gp
    return [dot(dot(gu, ui), vi) + 0.01 * ddot(gu, gv) - pi * trace(gv) + tau * dot(dot(gv, ui), R)
            - tau**2 * dot(R, dot(gv, R)) + qi * trace(gu) + tau * dot(gq, R) + 0.1 * dot(ui, vi) + pi * qi]  # fmt: skip


def _p2_hessian():
    """A scalar P2 form reading the test function's Laplacian and gradient, with a nonlinear coefficient."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=3).domain()
    u, v = d.fem_symbols(order=2, names=("u", "v"))
    x, y = d.variable("interior", split=True)[:2]
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    gu, gv = grad(u, [x, y]), grad(v, [x, y])
    return [(1 + ui * ui) * dot(gu, gv) + 0.1 * lap(u, [x, y]) * lap(v, [x, y]) + ui * vi]


def _component_cellwise_where():
    """Component gradients ``v[i].d(x)``, a B-bar ``cellwise`` on the test side, a ``where`` selection."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=3).domain()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"))
    x, y = d.variable("interior", split=True)[:2]
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    gu, gv = grad(u, [x, y]), grad(v, [x, y])
    cu, cv = jno.np.cellwise(trace(gu)), jno.np.cellwise(trace(gv))
    return [ui[0].d(x) * vi[0].d(x) + (1 + ui[1] ** 2) * ui[1].d(y) * vi[1].d(y) + cu * cv
            + where(x > 0.5, dot(ui, vi), 0.0 * dot(ui, vi)) + dot(ui, vi)]  # fmt: skip


def _assemble(terms_of, seeded, monkeypatch):
    monkeypatch.setattr(fn, "_SEEDED_TEST", [seeded])
    # The element-kernel caches key on the form, not on this switch: start each side from empty.
    monkeypatch.setattr(fu, "_ELEM_MAP_CACHE", type(fu._ELEM_MAP_CACHE)())
    monkeypatch.setattr(fu, "_ELEM_MAP_CONTENT", type(fu._ELEM_MAP_CONTENT)())
    fem = jno.fem(terms_of())
    u0 = jnp.asarray(np.random.default_rng(0).standard_normal(fem.dofs) * 0.3)
    J = fem.jacobian(u0)
    return np.asarray(fem.residual(u0)), np.asarray(J.todense() if hasattr(J, "todense") else J)


@pytest.mark.parametrize("terms_of", [_flow, _p2_hessian, _component_cellwise_where])
def test_seeded_and_per_dof_assemble_the_same_residual_and_tangent(terms_of, monkeypatch):
    proven, depth = [], [0]
    real = tr.linear_degree

    def spy(*a, **k):  # the walk recurses through this name: keep only the outermost answer per piece
        depth[0] += 1
        try:
            d = real(*a, **k)
        finally:
            depth[0] -= 1
        if depth[0] == 0 and k.get("strict"):
            proven.append(d)
        return d

    monkeypatch.setattr(tr, "linear_degree", spy)
    r1, J1 = _assemble(terms_of, True, monkeypatch)
    assert proven and all(d == 1 for d in proven), f"a piece did not take the seeded path: {proven}"
    r0, J0 = _assemble(terms_of, False, monkeypatch)
    assert np.abs(r0).max() > 1e-3 and np.abs(J0).max() > 1e-3
    assert np.allclose(r1, r0, rtol=1e-12, atol=1e-13 * np.abs(r0).max())
    assert np.allclose(J1, J0, rtol=1e-12, atol=1e-13 * np.abs(J0).max())


def test_a_test_function_under_lag_keeps_its_load():
    """``lag`` is a stop-gradient: seeding through it would differentiate to zero and drop the load.
    The walk refuses it, so ``∫ u v = ∫ lag(v)`` still has the solution ``u = 1``."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain()
    u, v = d.fem_symbols(names=("u", "v"))
    x, y = d.variable("interior", split=True)[:2]
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    sol = np.asarray(jno.fem([ui * vi - jno.lag(vi)]).solve())
    assert np.allclose(sol, 1.0, atol=1e-8), (sol.min(), sol.max())


def test_a_term_not_linear_in_its_test_function_keeps_the_per_dof_number(monkeypatch):
    """Not a weak form, but it used to evaluate to something: it still does, the same thing."""

    def terms():
        d = jno.shape.rect(0, 0, 1, 1).structured(n=3).domain()
        u, v = d.fem_symbols(names=("u", "v"))
        x, y = d.variable("interior", split=True)[:2]
        ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
        return [ui * ui * vi + 0.5 * ui * vi * vi]

    # Seeding the v*v piece would differentiate it to zero at s = 0; per DOF it is 0.5 u φ_a².
    r1, _ = _assemble(terms, True, monkeypatch)
    r0, _ = _assemble(terms, False, monkeypatch)
    assert np.abs(r0).max() > 1e-3
    assert np.allclose(r1, r0, rtol=1e-13, atol=1e-15)
