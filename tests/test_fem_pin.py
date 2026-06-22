"""Gauge-fix pin ``p.pin()`` for an incompressible pressure / pure-Neumann scalar null space.

``p.pin()`` removes a field's constant null space by fixing one arbitrary DOF -- it is
gauge-fixing, not a boundary condition. ``jno.fem`` lowers it to a single-node Dirichlet
``p(node) - value`` at a deterministic vertex (nearest the mesh min-corner), i.e. the *same*
essential path as the explicit ``domain.point_region`` + ``p(xpn, ypn) - 0`` form. These tests
pin equivalence to that explicit form (steady-linear AND transient-nonlinear -- the latter is
the path where a synthesized value node could silently be read as time-varying), the gauge
shift, the two misuse errors, and idempotency across two ``jno.fem`` calls on one domain.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402

inner, grad, trace, where = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.where


@pytest.fixture(autouse=True)
def _x64():
    """feax assembly is float64; opt into x64 per-test and restore (session default is x64-off)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


# ---------------------------------------------------------------------------
# Steady Taylor-Hood Stokes (linear): u* = (x, -y), p* = x, f = (1, 0). Pure-Dirichlet
# velocity -> pressure null space; pinned two ways and compared.
# ---------------------------------------------------------------------------
def _stokes(pin_mode: str, pin_value: float = 0.0):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
    p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq, vv = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    cons = [
        inner(gu, gv, n_contract=2) - pp * trace(gv) - 1.0 * vv[0],  # momentum, f = (1, 0)
        -qq * trace(gu),  # continuity
        u(xb, yb)[0] - xb,
        u(xb, yb)[1] - (-1.0 * yb),
    ]
    if pin_mode == "explicit":
        d.point_region("ppin", (0.0, 0.0))
        xpn, ypn, _ = d.variable("ppin", split=True)
        cons.append(p(xpn, ypn) - 0.0)
    else:
        cons.append(p.pin(pin_value))
    return d, jno.fem(cons)


def test_pin_matches_explicit_point_region_steady():
    _, fem_e = _stokes("explicit")
    d_p, fem_p = _stokes("pin")
    assert fem_p.is_linear
    # the pin lowers to a single-node pressure Dirichlet on an auto-tagged region
    assert any(c.startswith("dirichlet@_gauge_pin_") for c in fem_p.classification)
    assert [t for t in d_p._boundary_regions if t.startswith("_gauge_pin_")]  # region was registered

    off_e, off_p = fem_e.offsets, fem_p.offsets
    assert off_e == off_p  # identical block structure
    s_e = np.linalg.solve(_dense(fem_e.A), np.asarray(fem_e.b).reshape(-1))
    s_p = np.linalg.solve(_dense(fem_p.A), np.asarray(fem_p.b).reshape(-1))
    # both pin the (0, 0) min-corner vertex -> byte-identical systems -> match to machine eps,
    # velocity AND pressure (same gauge node, not merely up-to-a-constant)
    assert np.allclose(s_e[off_e[0] : off_e[1]], s_p[off_p[0] : off_p[1]], atol=1e-9)  # velocity
    assert np.allclose(s_e[off_e[1] :], s_p[off_p[1] :], atol=1e-9)  # pressure


def test_pin_value_shifts_pressure_gauge():
    c = 3.5
    _, fem0 = _stokes("pin", 0.0)
    _, femc = _stokes("pin", c)
    off = fem0.offsets
    s0 = np.linalg.solve(_dense(fem0.A), np.asarray(fem0.b).reshape(-1))
    sc = np.linalg.solve(_dense(femc.A), np.asarray(femc.b).reshape(-1))
    assert np.allclose(s0[off[0] : off[1]], sc[off[0] : off[1]], atol=1e-9)  # velocity unchanged
    assert np.allclose(sc[off[1] :] - s0[off[1] :], c, atol=1e-9)  # pressure shifted by exactly c


# ---------------------------------------------------------------------------
# Transient + nonlinear (lid-driven cavity, coarse): the path where a synthesized Dirichlet
# value node could be mis-read as time-varying. Pin two ways, run backward-Euler+Newton, compare.
# ---------------------------------------------------------------------------
def _cavity(pin_mode: str, nu: float = 0.05):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.34, time=(0.0, 0.3, 3))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    conv = inner(gu, ub, n_contract=1)  # (u.grad)u nonlinearity
    momentum = (
        inner(ub.t, vb, n_contract=1) + inner(conv, vb, n_contract=1) + nu * inner(gu, gv, n_contract=2) - pp * trace(gv)
    )
    cons = [
        momentum,
        -qq * trace(gu),
        u(xb, yb)[0] - where(yb > 1 - 1e-6, 1.0, 0.0),  # lid on top, no-slip elsewhere
        u(xb, yb)[1] - 0.0,
        u(ci[0], ci[1])[0] - 0.0,  # start from rest
        u(ci[0], ci[1])[1] - 0.0,
    ]
    if pin_mode == "explicit":
        d.point_region("ppin", (0.0, 0.0))
        xpn, ypn, _ = d.variable("ppin", split=True)
        cons.append(p(xpn, ypn) - 0.0)
    else:
        cons.append(p.pin())
    return jno.fem(cons)


def _run_transient(fem):
    dn = lambda X: jnp.asarray(X.todense()) if hasattr(X, "todense") else jnp.asarray(X)  # noqa: E731
    blk = fem.operator
    M, dt = dn(blk.mass(0.0, {})), float(blk.dt)
    nsteps = round((float(fem.t1) - float(fem.t0)) / dt)
    w = jnp.asarray(blk.state0)
    for step in range(nsteps):
        w_prev, t_next = w, (step + 1) * dt
        for _ in range(12):  # backward Euler + Newton
            G = M @ (w - w_prev) / dt + jnp.asarray(blk.residual(w, t_next, {})).reshape(-1)
            J = M / dt + dn(blk.jacobian(w, t_next, {}))
            dw = jnp.linalg.solve(J, -G)
            w = w + dw
            if float(jnp.linalg.norm(dw)) < 1e-11:
                break
    return np.asarray(w)


def test_pin_transient_nonlinear_matches_explicit():
    fem_e = _cavity("explicit")
    fem_p = _cavity("pin")
    assert fem_e.is_transient and not fem_e.is_linear
    assert fem_p.is_transient and not fem_p.is_linear  # pin did not change the routing
    w_e = _run_transient(fem_e)
    w_p = _run_transient(fem_p)
    # identical mesh + identical pin node (min-corner == (0, 0)) -> identical implicit steps
    assert np.allclose(w_e, w_p, atol=1e-8)


# ---------------------------------------------------------------------------
# Misuse + idempotency
# ---------------------------------------------------------------------------
def test_pin_on_test_function_raises():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.5)
    _, q = d.fem_symbols(names=("p", "q"))
    with pytest.raises(ValueError, match="trial symbol"):
        q.pin()


def test_pin_on_vector_field_raises():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.5)
    u, _ = d.fem_symbols(value_shape=(2,), names=("u", "v"))
    with pytest.raises(ValueError, match="scalar"):
        u.pin()


def test_pin_idempotent_across_two_fem_calls():
    # Two jno.fem(...) calls on ONE domain reusing the SAME symbols (stable field_key -> stable
    # pin tag). Coordinates are rebuilt each call so the weak-term retag does not leak between
    # calls; the pin region must be registered exactly once and both systems must match.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)

    def build():
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
        pp, qq, vv = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        return jno.fem(
            [
                inner(gu, gv, n_contract=2) - pp * trace(gv) - 1.0 * vv[0],
                -qq * trace(gu),
                u(xb, yb)[0] - xb,
                u(xb, yb)[1] - (-1.0 * yb),
                p.pin(),
            ]
        )

    fem1 = build()
    fem2 = build()
    pin_tags = [t for t in d._boundary_regions if t.startswith("_gauge_pin_")]
    assert len(pin_tags) == 1  # registered once, not accumulated across the two fem() calls
    s1 = np.linalg.solve(_dense(fem1.A), np.asarray(fem1.b).reshape(-1))
    s2 = np.linalg.solve(_dense(fem2.A), np.asarray(fem2.b).reshape(-1))
    assert np.allclose(s1, s2, atol=1e-9)
