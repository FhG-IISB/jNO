"""Constant unknowns: ``d.unknown.scalar(constant=True)`` is ONE value over the domain, solved for with the fields.

Its trial is seen identically by every element; its test function is one, so a weak term carrying it is an
integral row (``∫_Γ (u·n - Q/|Γ|) W ds = 0`` is a prescribed flow rate). A tie ``u(region) - U`` makes every
DOF of ``u`` on the region the same unknown as ``U`` -- eliminated exactly by a prolongation, with the tied
rows summed into U's row (the virtual work of the constraint), so a Neumann load on the region becomes U's
equation. ``U - g`` pins it.

Oracles (closed form):

* a floating boundary value with a prescribed total flux: ``-u'' = 1`` on a strip, ``u(0) = 0``, ``u(1) = U``
  free, ``∫ ∂u/∂n = Q`` on the right -> ``u = -x²/2 + (Q/H + 1) x``, ``U = Q/H + 1/2``;
* Stokes channel with a prescribed flow rate ``Q`` and the inlet pressure as the constant: Poiseuille,
  ``P_in = 12 Q L / H³`` (unit viscosity, do-nothing outlet);
* the tie with ``U`` pinned reproduces the hand-built uniform Dirichlet condition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno

J = jno.np
inner = J.inner
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _value(out):
    """A solve result as an array (a reduced nonlinear solve returns a deferred node)."""
    return np.asarray(out.eval() if isinstance(out, jno.trace.Placeholder) else out)


H = 0.5


def _floating(Q, *, extra=(), dirichlet=None, nonlinear=False, order=2):
    """-u'' = 1 on [0, 1] x [0, H], insulated top/bottom, u(0) = 0, the right edge tied to U, flux Q there."""
    d = jno.shape.rect(0.0, 0.0, 1.0, H).structured(n=4).domain()
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    xr, yr = d.variable("right", split=True)[:2]
    u = d.unknown(order=order)
    v = u.test()
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    k = (1.0 + 0.0 * ui**2) if nonlinear else 1.0  # nonlinear in form only: routes to the Newton path
    terms = [k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, -(Q / H) * v.bind(x=xr, y=yr), u(xl, yl) - 0.0]
    if dirichlet is not None:
        return jno.fem(terms + [u(xr, yr) - dirichlet]), None
    U = d.unknown.scalar(constant=True, name="U")
    terms += [u(xr, yr) - U] + [e(U) for e in extra]
    return jno.fem(terms), U


def test_floating_boundary_value_with_prescribed_flux():
    Q = 0.8
    fem, U = _floating(Q)
    assert fem.classification[-1] == "tie@right->U"
    assert fem.dofs == fem.offsets[1] + 1  # one DOF for the constant, appended as the last block
    sol = _value(fem.solve(linear=jno.solve.lu()))
    P = np.asarray(fem.field_points[0])
    np.testing.assert_allclose(sol[fem.blocks[fem.block_index(U)]], [Q / H + 0.5], atol=1e-12)
    np.testing.assert_allclose(sol[fem.blocks[0]], -(P[:, 0] ** 2) / 2 + (Q / H + 1) * P[:, 0], atol=1e-12)


def test_stokes_channel_prescribed_flow_rate():
    L, Hc, Q = 2.0, 1.0, 0.3
    d = jno.shape.rect(0.0, 0.0, L, Hc).structured(n=4).domain()
    d.tag("walls", lambda x, y: (y < 1e-9) | (y > Hc - 1e-9))
    x, y = d.variable("interior", split=True)[:2]
    xi, yi = d.variable("left", split=True)[:2]
    xw, yw = d.variable("walls", split=True)[:2]
    u = d.unknown.vector(2, order=2)
    p = d.unknown.scalar(name="p")
    P = d.unknown.scalar(constant=True, name="P_in")
    v, q, W = u.test(), p.test(), P.test()
    gu, gv = J.jacobian(u, [x, y]), J.jacobian(v, [x, y])
    pi, qi = p.bind(x=x, y=y), q.bind(x=x, y=y)
    vin, uin = v.bind(x=xi, y=yi), u.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            inner(gu, gv, n_contract=2) - pi * J.trace(gv),
            -qi * J.trace(gu),
            P * (-vin[0]),  # inlet traction -P n, with n = (-1, 0)
            (uin[0] - Q / Hc) * W,  # its equation: the flow rate, an integral row over the inlet
            u(xw, yw) - 0.0,
            u(xi, yi)[1] - 0.0,
        ]
    )
    sol = _value(fem.solve(linear=jno.solve.lu()))
    G = 12 * Q / Hc**3
    np.testing.assert_allclose(sol[fem.blocks[fem.block_index(P)]], [G * L], atol=1e-11)
    Pu, Pp = np.asarray(fem.field_points[0]), np.asarray(fem.field_points[1])
    vel = sol[fem.blocks[0]].reshape(-1, 2)
    np.testing.assert_allclose(vel[:, 0], G / 2 * Pu[:, 1] * (Hc - Pu[:, 1]), atol=1e-12)
    np.testing.assert_allclose(vel[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(sol[fem.blocks[1]], G * (L - Pp[:, 0]), atol=1e-11)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_pinned_tie_reproduces_a_uniform_dirichlet(nonlinear):
    kw = {"nonlinear": jno.solve.newton(direct=True, rtol=1e-12, atol=1e-13)} if nonlinear else {"linear": jno.solve.lu()}
    fem_t, U = _floating(0.8, extra=[lambda U: U - 0.7], nonlinear=nonlinear)
    fem_d, _ = _floating(0.8, dirichlet=0.7, nonlinear=nonlinear)
    assert "dirichlet@U" in fem_t.classification
    a = _value(fem_t.solve(**kw))
    b = _value(fem_d.solve(**kw))
    np.testing.assert_allclose(a[fem_t.blocks[0]], b, atol=1e-12)
    np.testing.assert_allclose(a[fem_t.blocks[1]], [0.7], atol=1e-13)


def test_vector_constant_tie():
    """u(right) - U for a 2-vector, U pinned: the same as the vector Dirichlet value."""
    out = []
    for mode in ("tie", "dirichlet"):
        d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=4).domain()
        x, y = d.variable("interior", split=True)[:2]
        xl, yl = d.variable("left", split=True)[:2]
        xr, yr = d.variable("right", split=True)[:2]
        u = d.unknown.vector(2)
        v = u.test()
        terms = [inner(J.jacobian(u, [x, y]), J.jacobian(v, [x, y]), n_contract=2), u(xl, yl) - 0.0]
        g = J.stack([0.3 + 0 * yr, -0.2 + 0 * yr], axis=-1)
        if mode == "tie":
            U = d.unknown.vector(2, constant=True, name="U")
            terms += [u(xr, yr) - U, U[0] - 0.3, U[1] - (-0.2)]
        else:
            terms += [u(xr, yr) - g]
        fem = jno.fem(terms)
        out.append(_value(fem.solve(linear=jno.solve.lu()))[: fem.offsets[1]])
    np.testing.assert_allclose(out[0], out[1], atol=1e-12)


def test_bounds_on_a_constant():
    """U = Q/H + 1/2 = 2.1 free; with U.bounds(0, 1.5) the bound is active and the field is the Dirichlet
    solution at 1.5; with U.bounds(0, 3) it is inactive and nothing changes."""
    active, U = _floating(0.8, extra=[lambda U: U.bounds(0.0, 1.5)], nonlinear=True)
    s = _value(active.solve())
    ref, _ = _floating(0.8, dirichlet=1.5)
    np.testing.assert_allclose(s[active.blocks[1]], [1.5], atol=1e-12)
    np.testing.assert_allclose(s[active.blocks[0]], _value(ref.solve(linear=jno.solve.lu())), atol=1e-10)
    loose, _ = _floating(0.8, extra=[lambda U: U.bounds(0.0, 3.0)], nonlinear=True)
    np.testing.assert_allclose(_value(loose.solve())[loose.blocks[1]], [2.1], atol=1e-10)


def test_gradient_through_a_runtime_parameter():
    """dL/dQ for L = (U - 1)², U = Q/H + 1/2: exactly 2 (U - 1) / H, as jno.core computes it."""
    Q = jno.np.parameter((1,), name="Q").initialize(lambda *a, **k: jnp.array([0.8]))
    Q.dtype(jnp.float64)
    fem, _ = _floating(Q)
    node = fem.solve(linear=jno.solve.lu())
    U_node = node[fem.offsets[1]]
    loss = ((U_node - 1.0) ** 2).mean
    lr = 1e-3
    Q.optimizer(optax.sgd(lr))
    crux = jno.core([loss], domain=_DUMMY)
    q0, U0 = (float(np.asarray(crux.eval([e])).reshape(-1)[0]) for e in (Q, U_node))
    crux.solve(1)
    q1 = float(np.asarray(crux.eval([Q])).reshape(-1)[0])
    assert abs(U0 - 2.1) < 1e-12
    np.testing.assert_allclose((q0 - q1) / lr, 2 * (U0 - 1.0) / H, rtol=1e-9)


def test_mean_value_multiplier_is_a_volume_integral_row():
    """Pure Neumann -Δu = f with a constant multiplier λ for ∫u = 0: the multiplier's row is ∫ u W dΩ, a
    volume integral, and λ is the (quadrature) mean of f -- zero up to the quadrature error of ∫f."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=8).domain()
    x, y = d.variable("interior", split=True)[:2]
    u = d.unknown(order=2)
    lam = d.unknown.scalar(constant=True, name="lam")
    v, W = u.test(), lam.test()
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    f = J.cos(np.pi * x) * J.cos(np.pi * y)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi + lam * vi, ui * W])
    sol = _value(fem.solve(linear=jno.solve.lu()))
    uu = sol[fem.blocks[0]]
    P = np.asarray(fem.field_points[0])
    exact = np.cos(np.pi * P[:, 0]) * np.cos(np.pi * P[:, 1]) / (2 * np.pi**2)
    assert abs(sol[fem.blocks[1]][0]) < 1e-7
    assert np.abs(uu - exact).max() < 2e-4  # P2, h = 1/8
    assert abs(float(fem.eval(u.bind(x=x, y=y), sol))) < 1e-13  # ∫ u dΩ = 0 exactly


def test_refusals():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=3).domain()
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    with pytest.raises(NotImplementedError, match="takes no order"):
        d.unknown.scalar(constant=True, order=2)
    U = d.unknown.scalar(constant=True, name="U")
    with pytest.raises(ValueError, match="every unknown in this form is a constant"):
        jno.fem([(U - 1.0) * U.test() + 0 * x])
    u = d.unknown()
    ui, vi = u.bind(x=x, y=y), u.test().bind(x=x, y=y)
    with pytest.raises(ValueError, match="names a region"):
        jno.fem([ui.x * vi.x + ui.y * vi.y, u(x, y) - U])
    with pytest.raises(ValueError, match="written `U - g`"):
        jno.fem([ui.x * vi.x + ui.y * vi.y, u(xl, yl) - U, U + 0.2])
    with pytest.raises(ValueError, match="same value shape"):
        V = d.unknown.vector(2, constant=True)
        jno.fem([ui.x * vi.x + ui.y * vi.y, u(xl, yl) - V])
    xb, yb = d.variable("boundary", split=True)[:2]
    with pytest.raises(NotImplementedError, match="FEM-only"):
        jno.fdm([ui.xx + ui.yy - 1.0, u(xb, yb) - U])


def test_a_tie_on_a_transient_form_is_refused():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=3).domain(time=(0.0, 0.1, 3))
    x, y, t = d.variable("interior", split=True)[:3]
    xr, yr, _ = d.variable("right", split=True)[:3]
    ci = d.variable("initial", split=True)
    u = d.unknown()
    U = d.unknown.scalar(constant=True)
    ui, vi = u.bind(x=x, y=y, t=t), u.test().bind(x=x, y=y, t=t)
    with pytest.raises(NotImplementedError, match="steady linear and nonlinear"):
        jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xr, yr) - U, U - 1.0, u(*ci) - 0.0])



def _tilted_strip(theta, *, slip_reaches_outlet=False):
    """The strip [0, 1] x [0, H] turned by ``theta``, tagged in its own frame (s along, η across). The slip
    walls stop short of the inlet corners (a slip node carrying a Dirichlet value on a tilted wall is a
    separate limitation, refused with or without a tie); ``outlet_open`` is the outlet without its corners."""
    c, s_ = np.cos(theta), np.sin(theta)
    shape = jno.shape.rect(0.0, 0.0, 1.0, H, size=0.2).rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), theta)
    d = shape.domain()

    def loc(x, y):
        return c * x + s_ * y, -s_ * x + c * y

    on_wall = lambda x, y: (np.abs(loc(x, y)[1]) < 1e-9) | (np.abs(loc(x, y)[1] - H) < 1e-9)  # noqa: E731
    d.tag("walls", lambda x, y: on_wall(x, y) & (loc(x, y)[0] > 1e-9))
    d.tag("inlet", lambda x, y: np.abs(loc(x, y)[0]) < 1e-9)
    d.tag("outlet", lambda x, y: np.abs(loc(x, y)[0] - 1.0) < 1e-9)
    d.tag("outlet_open", lambda x, y: (np.abs(loc(x, y)[0] - 1.0) < 1e-9) & ~on_wall(x, y))
    return d, loc, (c, s_)


def test_slip_and_a_tie_compose_exactly():
    """-Δu = (1, b) on the strip [0, 1] x [0, H]: slip on the long walls, u_x = 0 at the inlet, flux Q of u_x
    through the outlet, where u_x is tied to U. Exact: u_x = -x²/2 + (Q/H + 1) x, U = Q/H + 1/2, and
    u_y = b/2 y (H - y) -- held only by the slip walls, so a slip that is lost or doubled shows there, and a
    lost tie shows in U. The outlet corners are on both: the slip takes u_y there, the tie u_x."""
    Q, b = 0.8, 0.6
    d = jno.shape.rect(0.0, 0.0, 1.0, H).structured(n=4).domain()
    d.tag("walls", lambda x, y: (y < 1e-9) | (y > H - 1e-9))
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    xr, yr = d.variable("right", split=True)[:2]
    cw = d.variable("walls", normals=True, split=True)
    xw, yw, nx, ny = cw[0], cw[1], cw[-2], cw[-1]
    u = d.unknown.vector(2, order=2)
    v = u.test()
    U = d.unknown.scalar(constant=True, name="U")
    vi = v.bind(x=x, y=y)
    fem = jno.fem(
        [
            inner(J.jacobian(u, [x, y]), J.jacobian(v, [x, y]), n_contract=2) - vi[0] - b * vi[1],
            -(Q / H) * v.bind(x=xr, y=yr)[0],
            nx * u(xw, yw)[0] + ny * u(xw, yw)[1] - 0.0,
            u(xl, yl)[0] - 0.0,
            u(xr, yr)[0] - U,
        ]
    )
    assert "slip@walls" in fem.classification and "tie@right->U" in fem.classification
    sol = _value(fem.solve(linear=jno.solve.lu()))
    P = np.asarray(fem.field_points[0])
    vel = sol[fem.blocks[0]].reshape(-1, 2)
    np.testing.assert_allclose(sol[fem.blocks[fem.block_index(U)]], [Q / H + 0.5], atol=1e-12)
    np.testing.assert_allclose(vel[:, 0], -(P[:, 0] ** 2) / 2 + (Q / H + 1) * P[:, 0], atol=1e-12)
    np.testing.assert_allclose(vel[:, 1], b / 2 * P[:, 1] * (H - P[:, 1]), atol=1e-12)


def test_a_weighted_slip_and_a_vector_tie_compose_exactly():
    """The same strip turned by 0.3 rad, so the slip rows are weighted (the wall normal is no axis), and the
    whole outlet velocity tied to a vector constant. Exact: u = u_s(s) t̂ with u_s = -s²/2 + (Q/H + 1) s, so
    U = (Q/H + 1/2) t̂ -- the composed reduction must reproduce a quadratic it can represent, to round-off."""
    Q = 0.8
    d, loc, t_hat = _tilted_strip(0.3)
    x, y = d.variable("interior", split=True)[:2]
    xi, yi = d.variable("inlet", split=True)[:2]
    xo, yo = d.variable("outlet", split=True)[:2]
    xoo, yoo = d.variable("outlet_open", split=True)[:2]
    cw = d.variable("walls", normals=True, split=True)
    xw, yw, nx, ny = cw[0], cw[1], cw[-2], cw[-1]
    u = d.unknown.vector(2, order=2)
    v = u.test()
    U = d.unknown.vector(2, constant=True, name="U")
    vi, vo = v.bind(x=x, y=y), v.bind(x=xo, y=yo)
    fem = jno.fem(
        [
            inner(J.jacobian(u, [x, y]), J.jacobian(v, [x, y]), n_contract=2) - (t_hat[0] * vi[0] + t_hat[1] * vi[1]),
            -(Q / H) * (t_hat[0] * vo[0] + t_hat[1] * vo[1]),
            nx * u(xw, yw)[0] + ny * u(xw, yw)[1] - 0.0,
            u(xi, yi) - 0.0,
            u(xoo, yoo) - U,
        ]
    )
    sol = _value(fem.solve(linear=jno.solve.lu()))
    P = np.asarray(fem.field_points[0])
    s = loc(P[:, 0], P[:, 1])[0]
    exact = (-(s**2) / 2 + (Q / H + 1) * s)[:, None] * np.asarray(t_hat)
    np.testing.assert_allclose(sol[fem.blocks[fem.block_index(U)]], (Q / H + 0.5) * np.asarray(t_hat), atol=1e-12)
    np.testing.assert_allclose(sol[fem.blocks[0]].reshape(-1, 2), exact, atol=1e-12)


def test_a_node_both_slipping_and_tied_is_refused():
    """Tying the WHOLE outlet velocity, corners included, puts the corner nodes on the slip wall too: their
    normal component would be eliminated by the slip and pinned to U by the tie. Refused, naming it."""
    d, _loc, _t = _tilted_strip(0.3)
    x, y = d.variable("interior", split=True)[:2]
    xi, yi = d.variable("inlet", split=True)[:2]
    xo, yo = d.variable("outlet", split=True)[:2]
    cw = d.variable("walls", normals=True, split=True)
    xw, yw, nx, ny = cw[0], cw[1], cw[-2], cw[-1]
    u = d.unknown.vector(2, order=2)
    v = u.test()
    U = d.unknown.vector(2, constant=True, name="U")
    with pytest.raises(NotImplementedError, match="sits on a slip surface"):
        jno.fem(
            [
                inner(J.jacobian(u, [x, y]), J.jacobian(v, [x, y]), n_contract=2) - v.bind(x=x, y=y)[0],
                nx * u(xw, yw)[0] + ny * u(xw, yw)[1] - 0.0,
                u(xi, yi) - 0.0,
                u(xo, yo) - U,
            ]
        )
