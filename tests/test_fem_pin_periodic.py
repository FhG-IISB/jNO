"""``p.pin()`` and single-node values on a PERIODIC domain.

A tie ``u(A) - u(B)`` eliminates the ``A`` side, and the min-corner vertex -- where ``p.pin()`` used to
land -- is on the ``A`` side of every tie of a periodic box. Two failures followed:

* coupled (``u`` + ``p``): ``jno.fem`` raised "prescribed DOF ... was eliminated by a tie", because the
  coupled reduction never kept prescribed DOFs out of the elimination. The same error stopped every
  coupled problem with a Dirichlet value on a tied face, e.g. a periodic channel with no-slip walls;
* single field: it built and solved, but the pinned corner was kept out of the tie on its own, so it
  held 0 while its three periodic images held -0.037 (u ~ 1): the tie was torn at the pin, silently.

The pin now uses the vertex nearest the min-corner that lies on no tied face; a value written by hand on
a tied node is carried to all of its images; and a prescribed value on a tie target no longer breaks a
nonlinear transient march. Oracles: a manufactured doubly periodic Poisson solution, the 2-D
Taylor-Green vortex (an exact Navier-Stokes solution), Poiseuille flow (exact in P2), and the same
problem gauged at an interior point, which never touched a tie.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

inner, grad, trace, sin, cos = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.sin, jno.np.cos


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _box(L, n, **dkw):
    d = jno.shape.rect(0.0, 0.0, L, L).structured(n=n).domain(**dkw)
    e = 1e-9 * L
    for nm, pr in {
        "left": lambda x, y: x < e,
        "right": lambda x, y: x > L - e,
        "bottom": lambda x, y: y < e,
        "top": lambda x, y: y > L - e,
    }.items():
        d.tag(nm, pr)
    return d


def _at(d, tag):
    return d.variable(tag, split=True)[:2]


def _node_value(d, gauge):
    """``u(node) - 0`` at the vertex nearest ``gauge`` (a hand-written single-node value)."""
    return lambda f: (d.point_region("g0", gauge), f(*_at(d, "g0")) - 0.0)[1]


def _pin_vertex(d):
    tags = [t for t in d._boundary_regions if t.startswith("_gauge_pin_")]
    assert len(tags) == 1, tags
    return np.asarray(d._boundary_regions[tags[0]].points).reshape(-1)


def _load(d):
    """``L_i = int phi_i dx`` on P1: ``L @ u`` is the integral of the FE field."""
    a, b = d.fem_symbols(order=1, names=("_la", "_lb"))
    xi, yi, _ = d.variable("interior", split=True)
    return np.asarray(jno.fem([a.bind(x=xi, y=yi) * b.bind(x=xi, y=yi) - 1.0 * b.bind(x=xi, y=yi)]).b).reshape(-1)


# ---------------------------------------------------------------------------
# Single field: -lap u = f on the doubly periodic unit square, u* = cos(2 pi x) cos(2 pi y), int f = 0.
# The solution is defined up to a constant; every gauge must give u* + const + O(h^2).
# ---------------------------------------------------------------------------
def _poisson(gauge, n=16):
    d = _box(1.0, n)
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, _ = d.variable("interior", split=True)
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = 8 * np.pi**2 * cos(2 * np.pi * xi) * cos(2 * np.pi * yi)
    terms = [ub.x * vb.x + ub.y * vb.y - f * vb, u(*_at(d, "left")) - u(*_at(d, "right"))]
    terms.append(u(*_at(d, "bottom")) - u(*_at(d, "top")))
    if gauge == "pin":
        terms.append(u.pin())
    elif gauge == "mean":
        terms.append(u.pin(mean=True))
    else:  # a hand-written value at the vertex nearest `gauge`
        terms.append(_node_value(d, gauge)(u))
    fem = jno.fem(terms)
    s = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)
    pts = np.asarray(fem.points)[:, :2]
    return d, fem, s, pts, np.cos(2 * np.pi * pts[:, 0]) * np.cos(2 * np.pi * pts[:, 1])


def _corner_images(pts, L=1.0):
    on = lambda a: (np.abs(a) < 1e-9) | (np.abs(a - L) < 1e-9)  # noqa: E731
    idx = np.flatnonzero(on(pts[:, 0]) & on(pts[:, 1]))
    assert idx.size == 4
    return idx


def test_pin_on_a_doubly_periodic_scalar_holds_and_matches_the_manufactured_solution():
    d, fem, s, pts, ue = _poisson("pin")
    # the pin sits off every tied face, and its value holds there
    pv = _pin_vertex(d)
    assert 1e-9 < pv[0] < 1 - 1e-9 and 1e-9 < pv[1] < 1 - 1e-9, pv
    k = int(np.argmin(((pts - pv) ** 2).sum(axis=1)))
    assert abs(s[k]) < 1e-12
    # the field is periodic everywhere, including the corner the old pin tore (0 vs -0.037)
    corners = _corner_images(pts)
    assert np.ptp(s[corners]) < 1e-12, s[corners]
    # u* up to a constant, to the P1 nodal error (measured 1.27e-02 at n=16)
    err = s - ue
    assert np.abs(err - err.mean()).max() < 2.0e-2
    # and the SAME discrete solution as a gauge that never touches a tie, up to a constant
    _, _, s_int, _, _ = _poisson((0.5, 0.5))
    assert np.ptp(s - s_int) < 1e-10


def test_pin_mean_on_a_doubly_periodic_scalar_is_mean_zero_and_converges():
    errs = []
    for n in (8, 16):
        d, fem, s, pts, ue = _poisson("mean", n=n)
        L = _load(d)
        assert abs(float(L @ s)) < 1e-12  # int u dx == 0, the gauge it asked for
        errs.append(np.abs(s - ue).max())  # u* has zero mean: no constant left to remove
    assert errs[1] < 1.3e-2, errs
    assert errs[0] / errs[1] > 3.5, errs  # second order: the gauge adds no O(1) constant


def test_a_hand_written_value_on_a_tied_corner_holds_at_every_image():
    """``u(0, 0) - 0`` names the corner every tie eliminates. The value is carried to the one unknown the
    four corners share, instead of tearing the tie at that node (which it used to do, silently)."""
    _, _, s, pts, ue = _poisson((0.0, 0.0))
    corners = _corner_images(pts)
    np.testing.assert_allclose(s[corners], 0.0, atol=1e-12)
    _, _, s_int, _, _ = _poisson((0.5, 0.5))
    assert np.ptp(s - s_int) < 1e-10


def test_two_different_values_on_one_periodic_unknown_are_refused():
    d = _box(1.0, 4)
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, _ = d.variable("interior", split=True)
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    d.point_region("c00", (0.0, 0.0))
    d.point_region("c11", (1.0, 1.0))
    terms = [
        ub.x * vb.x + ub.y * vb.y - 0.0 * vb,
        u(*_at(d, "left")) - u(*_at(d, "right")),
        u(*_at(d, "bottom")) - u(*_at(d, "top")),
        u(*_at(d, "c00")) - 0.0,
        u(*_at(d, "c11")) - 1.0,  # the same unknown as (0, 0) under both ties
    ]
    with pytest.raises(ValueError, match="identifies as one unknown"):
        jno.fem(terms)


# ---------------------------------------------------------------------------
# Coupled: a channel periodic in x with no-slip walls at y = 0, 1 and f = (1, 0). The wall corners are
# prescribed AND on a tied face -- the case the coupled reduction could not build at all, with or
# without a pin. Steady Stokes: Poiseuille u = (y (1 - y) / 2, 0), p = const. Transient Navier-Stokes:
# u = (1 + t) (y (1 - y) / 2, 0) under f = (y (1 - y) / 2 + 1 + t, 0), a parallel flow, so (u.grad)u = 0.
# Both are quadratic in y and the transient is linear in t, so Taylor-Hood P2/P1 with backward Euler
# reproduces them to round-off.
# ---------------------------------------------------------------------------
def _channel(gauge, transient=False):
    dkw = {"time": (0.0, 0.1, 4)} if transient else {}
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=6).domain(**dkw)
    for nm, pr in {
        "left": lambda x, y: x < 1e-9,
        "right": lambda x, y: x > 1 - 1e-9,
        "wall": lambda x, y: (y < 1e-9) | (y > 1 - 1e-9),
    }.items():
        d.tag(nm, pr)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    bind = {"x": xi, "y": yi, **({"t": ti} if transient else {})}
    ub, vv, pp, qq = u.bind(**bind), v.bind(**bind), p.bind(**bind), q.bind(**bind)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    momentum = inner(gu, gv, n_contract=2) - pp * trace(gv)
    if transient:
        momentum = momentum + inner(ub.t, vv, n_contract=1) + inner(inner(gu, ub, n_contract=1), vv, n_contract=1)
        momentum = momentum - (0.5 * yi * (1.0 - yi) + 1.0 + ti) * vv[0]
    else:
        momentum = momentum - 1.0 * vv[0]
    xw, yw = _at(d, "wall")
    terms = [
        momentum,
        -qq * trace(gu),
        u(xw, yw)[0] - 0.0,
        u(xw, yw)[1] - 0.0,
        u(*_at(d, "left")) - u(*_at(d, "right")),
        p(*_at(d, "left")) - p(*_at(d, "right")),
        p.pin() if gauge == "pin" else _node_value(d, gauge)(p),
    ]
    if transient:
        ci = d.variable("initial", split=True)
        terms += [u(*ci)[0] - 0.5 * ci[1] * (1.0 - ci[1]), u(*ci)[1] - 0.0]
        fem = jno.fem(terms)
        s = fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-12, atol=1e-13), linear=jno.solve.lu(backend="host"))
        s = np.asarray(s.fn() if hasattr(s, "fn") else s)[-1]
    else:
        fem = jno.fem(terms)
        s = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)
    U = s[fem.blocks[fem.block_index(u)]].reshape(-1, 2)
    P = s[fem.blocks[fem.block_index(p)]]
    return d, U, P, np.asarray(d._fem_native_dof_points_all[fem.block_index(u)])[:, :2]


@pytest.mark.parametrize("transient", [False, True], ids=["steady-stokes", "transient-navier-stokes"])
@pytest.mark.parametrize("gauge", ["pin", (0.0, 0.5)], ids=["pin", "value-on-eliminated-face"])
def test_coupled_periodic_channel_with_walls_on_the_tied_faces(gauge, transient):
    """Wall values on the tied faces' corners, gauged by ``p.pin()`` or by a hand-written ``p(0, 0.5) - 0``
    on the eliminated face: every combination builds and gives Poiseuille flow to round-off, with the
    walls held on both sides of the tie and the pressure at its gauge."""
    _, U, P, xu = _channel(gauge, transient)
    y = xu[:, 1]
    amp = 1.1 if transient else 1.0  # (1 + t) at t = 0.1
    np.testing.assert_allclose(U[:, 0], amp * 0.5 * y * (1 - y), atol=1e-11)
    np.testing.assert_allclose(U[:, 1], 0.0, atol=1e-11)
    np.testing.assert_allclose(P, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Coupled, transient, nonlinear: the 2-D Taylor-Green vortex on the doubly periodic [0, 2 pi]^2,
#   u = (sin x cos y, -cos x sin y) e^{-2 nu t},  p = (cos 2x + cos 2y) / 4 e^{-4 nu t}.
# Taylor-Hood P2/P1, backward Euler, Newton on the assembled tangent.
# ---------------------------------------------------------------------------
NU, T_END, N_STEPS = 0.1, 0.2, 4


def _taylor_green(gauge):
    L = 2 * np.pi
    d = _box(L, 12, time=(0.0, T_END, N_STEPS + 1))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pp, qq = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    momentum = (
        inner(ub.t, vb, n_contract=1)
        + inner(inner(gu, ub, n_contract=1), vb, n_contract=1)
        + NU * inner(gu, gv, n_contract=2)
        - pp * trace(gv)
    )
    terms = [momentum, -qq * trace(gu)]
    for a, b in (("left", "right"), ("bottom", "top")):
        terms += [u(*_at(d, a)) - u(*_at(d, b)), p(*_at(d, a)) - p(*_at(d, b))]
    terms += [u(*ci)[0] - sin(ci[0]) * cos(ci[1]), u(*ci)[1] - (-cos(ci[0]) * sin(ci[1]))]
    terms.append(p.pin() if gauge == "pin" else _node_value(d, gauge)(p))
    fem = jno.fem(terms)
    traj = fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-11, atol=1e-12), linear=jno.solve.lu(backend="host"))
    traj = np.asarray(traj.fn() if hasattr(traj, "fn") else traj)
    U = traj[-1][fem.blocks[fem.block_index(u)]].reshape(-1, 2)
    P = traj[-1][fem.blocks[fem.block_index(p)]]
    xu = np.asarray(d._fem_native_dof_points_all[fem.block_index(u)])[:, :2]
    xp = np.asarray(d._fem_native_dof_points_all[fem.block_index(p)])[:, :2]
    return U, P, xu, xp


def test_coupled_periodic_navier_stokes_pin_matches_an_interior_gauge_and_taylor_green():
    L = 2 * np.pi
    U_i, P_i, xu, xp = _taylor_green((L / 2, L / 2))  # interior: never touches a tie
    for gauge in ("pin", (0.0, 0.0), (L, L)):  # the pin; the eliminated corner; the retained corner
        U, P, _, _ = _taylor_green(gauge)
        np.testing.assert_allclose(U, U_i, atol=1e-9, err_msg=str(gauge))  # the velocity cannot move
        assert np.ptp(P - P_i) < 1e-9, (gauge, np.ptp(P - P_i))  # the pressure moves by a constant only
    # ...and the common answer is the Taylor-Green vortex, to discretisation error. Measured at n=12:
    # velocity 9.6e-03 (a march that did not decay at all would be 4.1e-02 off), mean-free pressure 9.2e-02.
    x, y = xu[:, 0], xu[:, 1]
    Ue = np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)], axis=1) * np.exp(-2 * NU * T_END)
    assert np.linalg.norm(U_i - Ue) / np.linalg.norm(Ue) < 1.5e-2
    Pe = (np.cos(2 * xp[:, 0]) + np.cos(2 * xp[:, 1])) / 4 * np.exp(-4 * NU * T_END)
    dp = (P_i - P_i.mean()) - (Pe - Pe.mean())
    assert np.linalg.norm(dp) / np.linalg.norm(Pe - Pe.mean()) < 0.15
