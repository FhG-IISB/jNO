"""Steady coupling of problems on their own meshes (jno._coupling.driver).

Two slabs in series, [0, 1] with conductivity k_a and [1, 2] with k_b, held at 0 on the far left and 1 on the far
right, insulated top and bottom: the temperature is piecewise linear, the flux through both the same,
q = 1 / (1/k_a + 1/k_b), and the interface sits at q/k_a. Every discretisation here holds a linear field exactly,
so the coupled solution must equal it to the coupling tolerance, whatever the meshes on either side.
"""

import jax
import numpy as np
import pytest

import jno
import jno.jnp_ops as jnn
from jno._coupling.driver import CouplingSpec, couple

KA, KB = 1.0, 10.0
Q = 1.0 / (1.0 / KA + 1.0 / KB)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def exact(x):
    return np.where(x <= 1.0, Q * x / KA, Q / KA + Q * (x - 1.0) / KB)


def _strong_slab(make, x0, k, h, outer_tag, outer_value, iface_tag):
    d = jno.shape.rect(x0, 0.0, x0 + 1.0, 1.0, size=h).domain()
    x, y, _ = d.variable("interior", split=True)
    xo, yo, _ = d.variable(outer_tag, split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    cons = [-k * (ui.xx + ui.yy), u(xo, yo) - outer_value]
    for tag in ("top", "bottom"):
        xt, yt, _ = d.variable(tag, split=True)
        cons.append(u.bind(x=xt, y=yt).d(d.variable(tag, normals=True)) - 0.0)
    xi, yi, _ = d.variable(iface_tag, split=True)
    ub = u.bind(x=xi, y=yi)
    return make(cons), ub, ub.d(d.variable(iface_tag, normals=True)), d


def _fem_slab(x0, k, h, outer_tag, outer_value, iface_tag):
    d = jno.shape.rect(x0, 0.0, x0 + 1.0, 1.0, size=h).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xo, yo, _ = d.variable(outer_tag, split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xs, ys, _ = d.variable(iface_tag, split=True)
    ub = u.bind(x=xs, y=ys)
    return (
        jno.fem([k * (ui.x * vi.x + ui.y * vi.y), u(xo, yo) - outer_value]),
        ub,
        ub.d(d.variable(iface_tag, normals=True)),
        d,
    )


def _coords(prob, d):
    if hasattr(prob, "centroids"):
        return np.asarray(prob.centroids)[:, 0]
    return np.asarray(d.mesh_connectivity["points"])[:, 0]


def _slabs(left, right):
    a = left(0.0, KA, 0.1, "left", 0.0, "right")
    b = right(1.0, KB, 0.07, "right", 1.0, "left")
    return a, b


KINDS = {
    "fvm": lambda *a: _strong_slab(jno.fvm, *a),
    "fdm": lambda *a: _strong_slab(jno.fdm, *a),
    "fem": _fem_slab,
}


@pytest.mark.parametrize("left, right", [("fvm", "fem"), ("fem", "fvm"), ("fdm", "fvm")])
def test_two_slabs_on_their_own_meshes_match_the_series_resistance(left, right):
    (pa, ua, fa, da), (pb, ub, fb, db) = _slabs(KINDS[left], KINDS[right])
    # tol 1e-8: each problem solves with its own default, and the iterative ones (fdm, fem) are not more accurate
    c = couple([pa, pb], [ua - ub, KA * fa + KB * fb], CouplingSpec(tol=1e-8, max_iter=40))
    sa, sb = c.solve()
    # the low-conductance side takes the temperature (Giles 1997): the iteration then contracts by k_a/k_b
    assert c.parts[c.assign[0]].problem is pa
    np.testing.assert_allclose(sa, exact(_coords(pa, da)), atol=1e-7)
    np.testing.assert_allclose(sb, exact(_coords(pb, db)), atol=1e-7)
    assert c.iterations <= 12


def test_the_wrong_dirichlet_side_diverges_unrelaxed_and_converges_with_aitken():
    (pa, ua, fa, da), (pb, ub, fb, db) = _slabs(KINDS["fvm"], KINDS["fem"])
    conds = [ua - ub, KA * fa + KB * fb]
    with pytest.raises(RuntimeError, match="did not converge"):  # amplification k_b/k_a = 10 per sweep
        couple([pa, pb], conds, CouplingSpec(dirichlet=pb, accel=None, max_iter=8)).solve()
    c = couple([pa, pb], conds, CouplingSpec(dirichlet=pb, accel="aitken", tol=1e-8, max_iter=40))
    sa, sb = c.solve()
    np.testing.assert_allclose(sa, exact(_coords(pa, da)), atol=1e-7)
    np.testing.assert_allclose(sb, exact(_coords(pb, db)), atol=1e-7)


def test_jno_core_couples_problems_on_their_own_meshes():
    (pa, ua, fa, da), (pb, ub, fb, db) = _slabs(KINDS["fvm"], KINDS["fem"])
    core = jno.core([pa, pb, ua - ub, KA * fa + KB * fb])
    sa, sb = core.solve(coupling=jno.solve.coupling(tol=1e-8))
    np.testing.assert_allclose(sa, exact(_coords(pa, da)), atol=1e-7)
    np.testing.assert_allclose(sb, exact(_coords(pb, db)), atol=1e-7)
    assert core.coupled.iterations <= 12


@pytest.mark.parametrize("alpha, scheme", [("auto", "gauss-seidel"), (3.0, "jacobi")])
def test_robin_robin_reaches_the_series_resistance(alpha, scheme):
    (pa, ua, fa, da), (pb, ub, fb, db) = _slabs(KINDS["fvm"], KINDS["fem"])
    c = couple([pa, pb], [ua - ub, KA * fa + KB * fb], CouplingSpec(alpha=alpha, scheme=scheme, tol=1e-8, max_iter=60))
    sa, sb = c.solve()
    assert len(c.conditions) == 2 and {c.assign[0], c.assign[1]} == {0, 1}  # each side takes a Robin condition
    np.testing.assert_allclose(sa, exact(_coords(pa, da)), atol=1e-7)
    np.testing.assert_allclose(sb, exact(_coords(pb, db)), atol=1e-7)


def test_three_problems_in_series():
    # fvm [0, 1] (k 1) | fem [1, 2] (k 10) | fdm [2, 3] (k 2): q = 1 / (1 + 1/10 + 1/2), interfaces at q and 1.1 q
    ks = (1.0, 10.0, 2.0)
    q = 1.0 / sum(1.0 / k for k in ks)

    def exact3(x):
        return np.where(x <= 1, q * x, np.where(x <= 2, q + q * (x - 1) / ks[1], q * 1.1 + q * (x - 2) / ks[2]))

    pa, ua, fa, da = _strong_slab(jno.fvm, 0.0, ks[0], 0.1, "left", 0.0, "right")
    pc, uc, fc, dc = _strong_slab(jno.fdm, 2.0, ks[2], 0.09, "right", 1.0, "left")
    db = jno.shape.rect(1.0, 0.0, 2.0, 1.0, size=0.07).domain()
    T, v = db.fem_symbols()
    xi, yi, _ = db.variable("interior", split=True)
    pb = jno.fem([ks[1] * (T.bind(x=xi, y=yi).x * v.bind(x=xi, y=yi).x + T.bind(x=xi, y=yi).y * v.bind(x=xi, y=yi).y)])
    sides = {}
    for tag in ("left", "right"):
        xs, ys, _ = db.variable(tag, split=True)
        tb = T.bind(x=xs, y=ys)
        sides[tag] = (tb, tb.d(db.variable(tag, normals=True)))
    conds = [
        ua - sides["left"][0],
        ks[0] * fa + ks[1] * sides["left"][1],
        sides["right"][0] - uc,
        ks[1] * sides["right"][1] + ks[2] * fc,
    ]
    c = couple([pa, pb, pc], conds, CouplingSpec(tol=1e-8, max_iter=60))
    sa, sb, sc = c.solve()
    # the stiff middle holds no value condition of its own: given both fluxes it would float (a singular steady
    # problem), so one of its pairs is Robin-Robin -- the fdm side's, whose stiffness is the closer to its own
    robin = [cond for cond in c.conditions if cond.receiver is not None]
    assert len(robin) == 2 and all(set(cond.owners) == {1, 2} for cond in robin)
    np.testing.assert_allclose(sa, exact3(_coords(pa, da)), atol=1e-7)
    np.testing.assert_allclose(sb, exact3(_coords(pb, db)), atol=1e-7)
    np.testing.assert_allclose(sc, exact3(_coords(pc, dc)), atol=1e-7)


GAMMA, T_L, T_R = 1.4, 1.0, 2.0


def _resting_gas(robin=False):
    """A gas at rest in a tube 0 < x < 1 (a strip of cells): the Euler equations with heat conduction (KA) of an
    algebraic temperature T = p/rho in the energy row, fed by a reservoir at rest (p = 1, T = T_L) on the left,
    closed by a wall on the right. ``robin``: that wall is a slab [1, 2] of conductivity KB held at T_R, written as
    the condition it puts on the gas, KA dT/dn + KB (T - T_R) = 0."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 0.1).structured(n=(10, 1)).quad().domain()
    x, y, _ = d.variable("interior", split=True)
    q, T = d.unknown.vector(3), d.unknown()  # (rho, m, E) and T
    qi, Ti = q.bind(x=x, y=y), T.bind(x=x, y=y)
    rho, m, En = qi[0], qi[1], qi[2]
    u = m / rho
    p = (GAMMA - 1.0) * (En - 0.5 * m * u)
    F = jnn.stack([m, m * u + p, u * (En + p)], axis=-1)
    # Roe: it holds a stationary contact (a density jump at one pressure) all but exactly; HLL diffuses it into a flow
    cons = [F.d(x, scheme=jno.fv(flux="roe")) - jnn.stack([0, 0, KA * Ti.xx], axis=-1), Ti - p / rho]
    xl, yl, _ = d.variable("left", split=True)
    cons += [q(xl, yl) - jnn.stack([1.0 / T_L + 0.0 * xl, 0.0 * xl, 1.0 / (GAMMA - 1.0) + 0.0 * xl], axis=-1)]
    cons += [T(xl, yl) - T_L]
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    cons.append(q.bind(x=xr, y=yr)[1] - 0.0)
    Tr = T.bind(x=xr, y=yr)
    if robin:
        cons.append(KA * Tr.d(nr) + KB * (Tr - T_R))
    gas = jno.fvm(cons)
    n = gas.centroids.shape[0]
    at_rest = np.concatenate([np.full(n, 1.0 / T_L), np.zeros(n), np.full(n, 1.0 / (GAMMA - 1.0)), np.full(n, T_L)])
    return gas, Tr, Tr.d(nr), at_rest


def test_a_steady_gas_couples_from_a_starting_state():
    # Name the break: a steady participant is solved inside a compiled program from a traced state, and its
    # Jacobian's pattern was checked at zeros, where a gas has no density; the stiffness probe solved it from zeros
    # and with its wall at 0 too. Started at rest (x0=), the gas and a steady wall [1, 2] held at T_R reach what the
    # gas alone reaches with the wall as its Robin condition: the wall holds a linear field exactly and its interface
    # temperature is uniform along the face, so the two agree to the coupling tolerance.
    gas, Tg, fg, at_rest = _resting_gas()
    dw = jno.shape.rect(1.0, 0.0, 2.0, 0.1, size=0.05).domain()
    u, v = dw.fem_symbols()
    xw, yw, _ = dw.variable("interior", split=True)
    ui, vi = u.bind(x=xw, y=yw), v.bind(x=xw, y=yw)
    xo, yo, _ = dw.variable("right", split=True)
    wall = jno.fem([KB * (ui.x * vi.x + ui.y * vi.y), u(xo, yo) - T_R])
    xs, ys, _ = dw.variable("left", split=True)
    Ts = u.bind(x=xs, y=ys)
    core = jno.core([gas, wall, Tg - Ts, KA * fg + KB * Ts.d(dw.variable("left", normals=True))])
    sg, sw = core.solve(coupling=jno.solve.coupling(tol=1e-10, x0={gas: at_rest}))
    ref_gas, *_ = _resting_gas(robin=True)
    ref = np.asarray(ref_gas.solve(x0=at_rest)).reshape(np.shape(sg))
    np.testing.assert_allclose(sg, ref, atol=1e-8)
    pts = np.asarray(dw.mesh_connectivity["points"])[:, 0]
    sw = np.asarray(sw)
    Ti = sw[np.isclose(pts, 1.0)].mean()
    assert T_L < Ti < T_R
    np.testing.assert_allclose(sw, Ti + (T_R - Ti) * (pts - 1.0), atol=1e-8)  # the wall: linear


def test_a_steady_gas_without_a_starting_state_says_so():
    # From zeros a gas has no density: whichever check meets it first (the stencil check, Newton) raises and names
    # the remedy, a starting state.
    gas, *_ = _resting_gas()
    with pytest.raises((ValueError, RuntimeError), match="x0"):
        gas.solve()
