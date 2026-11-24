"""Transient coupling of problems on their own meshes, window by window (jno._coupling.driver).

Two solids brought into contact, one at 0 and one at 1: while neither feels its far end, the contact
temperature is constant at once, T_c = (e1 T1 + e2 T2) / (e1 + e2) with the effusivities e = sqrt(k rho c), and
each side is an erf profile about it (Carslaw & Jaeger 1959, §2.15). Here e1 = 1 (k 1) and e2 = 2 (k 4), so T_c = 2/3.
"""

import jax
import numpy as np
import pytest
from scipy.special import erf

import jno
from jno._coupling.driver import CouplingSpec, couple

K1, K2 = 1.0, 4.0
TC = (np.sqrt(K1) * 0.0 + np.sqrt(K2) * 1.0) / (np.sqrt(K1) + np.sqrt(K2))
T_END, N_TIME = 0.02, 21  # dt 1e-3; the diffusion length sqrt(4 K2 T) = 0.57 stays inside both unit bars


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def exact(x, t):
    left = TC + (0.0 - TC) * erf(-x / (2.0 * np.sqrt(K1 * t)))
    right = TC + (1.0 - TC) * erf(x / (2.0 * np.sqrt(K2 * t)))
    return np.where(x < 0.0, left, right)


def _fem_cold(h):
    d = jno.shape.rect(-1.0, 0.0, 0.0, 0.1, size=h).domain(time=(0.0, T_END, N_TIME))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ub.t * vb + K1 * (ub.x * vb.x + ub.y * vb.y), u(xl, yl) - 0.0, u(*ci) - 0.0 * ci[0]])
    xr, yr, _ = d.variable("right", split=True)
    ur = u.bind(x=xr, y=yr)
    return fem, ur, ur.d(d.variable("right", normals=True)), d


def _fvm_hot(h):
    d = jno.shape.rect(0.0, 0.0, 1.0, 0.1, size=h).domain(time=(0.0, T_END, N_TIME))
    x, y, t = d.variable("interior", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    prob = jno.fvm([ui.t - K2 * (ui.xx + ui.yy), u(xr, yr) - 1.0, u(xi, yi) - (1.0 + 0.0 * xi)])
    xl, yl, _ = d.variable("left", split=True)
    ul = u.bind(x=xl, y=yl)
    return prob, ul, ul.d(d.variable("left", normals=True)), d


@pytest.mark.parametrize("accel", ["aitken", "iqn-ils"])
def test_two_solids_in_contact_hold_the_contact_temperature(accel):
    pa, ua, fa, da = _fem_cold(0.02)
    pb, ub, fb, db = _fvm_hot(0.015)
    c = couple([pa, pb], [ua - ub, K1 * fa + K2 * fb], CouplingSpec(accel=accel, tol=1e-8, max_iter=40))
    ta, tb = c.solve()
    assert ta.shape[0] == tb.shape[0] == N_TIME and len(c.times) == N_TIME
    # the low-effusivity (cold, k 1) side takes the temperature, probed over one window
    assert c.parts[c.assign[0]].problem is pa
    pts = np.asarray(da.mesh_connectivity["points"])
    iface = np.flatnonzero(np.abs(pts[:, 0]) < 1e-12)
    # backward Euler and a mesh of h ~ 0.02 against an initial jump: ~1 % at the interface, about the profile
    np.testing.assert_allclose(ta[-1][iface], TC, atol=0.012)
    xb = np.asarray(pb.centroids)[:, 0]
    np.testing.assert_allclose(tb[-1], exact(xb, T_END), atol=0.02)
    np.testing.assert_allclose(ta[-1], exact(pts[:, 0], T_END), atol=0.02)


def test_an_imex_participant_takes_its_cfl_substeps_in_every_window():
    # jno.solve.imex(cfl=) decides its substeps from the state; inside a coupling's compiled window march the
    # state is traced, so the participant decides them per window (as for erk) -- it raised a concretization error
    pa, ua, fa, da = _fem_cold(0.02)
    pb, ub, fb, db = _fvm_hot(0.03)  # explicit diffusion: 4x fewer substeps than at 0.015
    spec = CouplingSpec(tol=1e-8, max_iter=40, time={pb: jno.solve.imex("ars222", cfl=0.5), pa: None})
    ta, tb = couple([pa, pb], [ua - ub, K1 * fa + K2 * fb], spec).solve()
    pts = np.asarray(da.mesh_connectivity["points"])
    iface = np.flatnonzero(np.abs(pts[:, 0]) < 1e-12)
    # the same bounds as backward Euler on that side: the mesh and the cold side's step set them
    np.testing.assert_allclose(ta[-1][iface], TC, atol=0.012)
    np.testing.assert_allclose(tb[-1], exact(np.asarray(pb.centroids)[:, 0], T_END), atol=0.02)


def test_loose_coupling_takes_one_sweep_per_window_and_stays_close():
    pa, ua, fa, da = _fem_cold(0.02)
    pb, ub, fb, db = _fvm_hot(0.015)
    conds = [ua - ub, K1 * fa + K2 * fb]
    loose = couple([pa, pb], conds, CouplingSpec(implicit=False))
    ta, tb = loose.solve()
    assert loose.window_iterations == [1] * (N_TIME - 1)
    tight = couple([pa, pb], conds, CouplingSpec(tol=1e-8, max_iter=40))
    ra, rb = tight.solve()
    # one sweep per window lags the interface data by a window: a first-order splitting error, small at
    # dt = 1e-3 but not round-off
    gap = np.abs(tb[-1] - rb[-1]).max()
    assert 1e-6 < gap < 0.05


def test_a_steady_problem_couples_quasi_steadily_to_a_transient_one():
    # a capacity-free slab [-1, 0] (k_a, held at 0 on its far side) on a heating slab [0, 1] (k_b, rho c 1, held at 1):
    # the steady slab answers every window at once, exactly a Robin condition k_b dT/dn = (k_a/L_a)(0 - T) on the
    # transient one -- which one jno.fem problem on the same mesh solves directly
    ka, kb = 2.0, 1.0
    da = jno.shape.rect(-1.0, 0.0, 0.0, 0.1, size=0.05).domain()
    x, y, _ = da.variable("interior", split=True)
    xl, yl, _ = da.variable("left", split=True)
    ua = da.unknown()
    pa = jno.fvm([-ka * (ua.bind(x=x, y=y).xx + ua.bind(x=x, y=y).yy), ua(xl, yl) - 0.0])
    xa, ya, _ = da.variable("right", split=True)
    uar = ua.bind(x=xa, y=ya)

    def wall(robin):
        d = jno.shape.rect(0.0, 0.0, 1.0, 0.1, size=0.04).domain(time=(0.0, 0.1, 21))
        T, v = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        xr, yr, _ = d.variable("right", split=True)
        ci = d.variable("initial", split=True)
        tb, vb = T.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
        xs, ys, _ = d.variable("left", split=True)
        cons = [tb.t * vb + kb * (tb.x * vb.x + tb.y * vb.y), T(xr, yr) - 1.0, T(*ci) - (1.0 + 0.0 * ci[0])]
        if robin:
            cons.append((ka / 1.0) * T.bind(x=xs, y=ys) * v.bind(x=xs, y=ys))
        Tl = T.bind(x=xs, y=ys)
        return jno.fem(cons), Tl, Tl.d(d.variable("left", normals=True)), d

    pb, ub, fb, db = wall(False)
    c = couple(
        [pa, pb], [uar - ub, ka * uar.d(da.variable("right", normals=True)) + kb * fb], CouplingSpec(tol=1e-10, max_iter=40)
    )
    _, tb = c.solve()
    ref, *_ = wall(True)
    tref = np.asarray(ref.solve().fn())
    # the fvm flux is piecewise constant over its faces; projected onto the FEM's P1 space it differs from the P1
    # Robin term by O(h²) (h 0.04-0.05 here: a few 1e-5), the transfer's own discretisation, not the coupling's
    np.testing.assert_allclose(tb, tref, atol=2e-4)
    assert np.abs(tb[-1] - tb[0]).max() > 0.1  # the wall did cool


def _heat_of_fem(d, u):
    """∫ u dA of a P1 field: the mean of its corner values times each triangle's area."""
    pts = np.asarray(d.mesh_connectivity["points"])[:, :2]
    tri = np.asarray(d.mesh_connectivity["triangles"])
    e1, e2 = pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]]
    return np.asarray(u)[..., tri].mean(axis=-1) @ (0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]))


def _contact_heat_drift(n_b):
    """Two insulated bars brought into contact (heat conserved): the fem bar [-1, 0] at 0 (k 1) one backward
    Euler step per window, the fvm bar [0, 1] at 1 (k 0.25, the softer side: it takes the temperature) on its own
    ``n_b - 1`` explicit steps over the same time. The largest drift of their total heat, over the heat moved."""
    t_end, kb = 0.02, 0.25
    da = jno.shape.rect(-1.0, 0.0, 0.0, 0.1, size=0.02).domain(time=(0.0, t_end, 11))
    u, v = da.fem_symbols()
    xi, yi, ti = da.variable("interior", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pa = jno.fem([ub.t * vb + K1 * (ub.x * vb.x + ub.y * vb.y), u(*da.variable("initial", split=True)) - 0.0])
    xr, yr, _ = da.variable("right", split=True)
    ua = u.bind(x=xr, y=yr)
    db = jno.shape.rect(0.0, 0.0, 1.0, 0.1, size=0.015).domain(time=(0.0, t_end, n_b))
    x, y, t = db.variable("interior", split=True)
    xb0, yb0, _ = db.variable("initial", split=True)
    w = db.unknown()
    wi = w.bind(x=x, y=y, t=t)
    pb = jno.fvm([wi.t - kb * (wi.xx + wi.yy), w(xb0, yb0) - (1.0 + 0.0 * xb0)])
    xl, yl, _ = db.variable("left", split=True)
    wl = w.bind(x=xl, y=yl)
    c = couple(
        [pa, pb],
        [ua - wl, K1 * ua.d(da.variable("right", normals=True)) + kb * wl.d(db.variable("left", normals=True))],
        CouplingSpec(tol=1e-10, max_iter=40, time={pb: jno.solve.imex("ars222", cfl=0.5), pa: None}),
    )
    ta, tb = c.solve()
    assert c.parts[c.assign[0]].problem is pb  # the softer bar takes the temperature, the fem bar the flux
    heat = _heat_of_fem(da, ta) + np.asarray(tb) @ np.asarray(pb.volumes)
    return float(np.abs(heat - heat[0]).max() / abs(_heat_of_fem(da, ta)[-1] - _heat_of_fem(da, ta)[0]))


def test_a_substepping_sender_passes_its_window_mean_flux_and_heat_is_conserved():
    # The flux the fem bar takes over a window must be what the fvm bar gave out: its mean over the window, not
    # its end value. After the contact jump the flux falls fast within a window; the end value lost 24 % of the
    # heat moved. The mean, by the trapezoid rule over the sender's steps, is off only on the first step after
    # the jump, where the flux is singular, and shrinks with that step.
    coarse, fine = _contact_heat_drift(101), _contact_heat_drift(201)
    # measured 2.4 % and 0.8 % (ten and twenty sender steps per window; the end value: 24 %): the first-step
    # error falls faster than the step
    assert coarse < 0.03 and fine < 0.5 * coarse, (coarse, fine)


@pytest.mark.parametrize("implicit", [True, False], ids=["one-way", "loose"])
def test_a_window_of_one_sweep_runs_as_one_compiled_program(implicit):
    # Name the break: a window that takes one sweep (a one-way coupling, or a loose one) is one compiled program,
    # traced once for every window after the first (eager) one. Run eagerly window by window, the participants'
    # Python glue between their compiled pieces dominated a long march (the nozzle start-up: 80 windows, 5.4 min).
    # Oracle: the same coupling with compile=False, window by window as before.

    def coupled(compile):  # fresh problems for each run (git-bug 44958ea: a fem build re-tags shared Variables)
        pa, ua, fa, _ = _fem_cold(0.04)
        pb, ub, fb, _ = _fvm_hot(0.03)
        if implicit:  # cooling of the cold side by the hot one's surface temperature: data flow one way only
            conds = [K1 * fa + 5.0 * (ua - ub)]
        else:  # the contact pair, one sweep per window
            conds = [ua - ub, K1 * fa + K2 * fb]
        return couple([pa, pb], conds, CouplingSpec(implicit=implicit, compile=compile, tol=1e-8, max_iter=40))

    ra, rb = coupled(False).solve()
    c = coupled(True)
    ta, tb = c.solve()
    assert c.compiles == 1 and len(c.times) == N_TIME
    # the same arithmetic, compiled: round-off of the step solves
    np.testing.assert_allclose(ta, ra, atol=1e-12)
    np.testing.assert_allclose(tb, rb, atol=1e-12)
    assert np.abs(ta[-1] - ta[0]).max() > 0.01  # the cold side did warm
