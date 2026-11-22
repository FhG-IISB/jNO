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
