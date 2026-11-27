"""Spectral problems in ``jno.core``'s partitioned coupling.

Two slabs side by side, [0, 1] x [0, 1] (k_a) and [1, 2] x [0, 1] (k_b), T = 0 on the top, bottom and far left,
T = sin(πy) on the far right. The exact solution is a sinh/cosh profile times sin(πy) -- not polynomial, so the
interface data are what a piecewise-linear transfer gets wrong at O(h²) and a spectral one does not:
two spectral sides with different node counts must keep spectral accuracy across the interface.
"""

import jax
import jno.numpy as jnn
import numpy as np
import pytest

import jno
from jno._coupling.driver import CouplingSpec, couple

KA, KB = 1.0, 4.0
PI = np.pi
A = 1.0 / (np.sinh(PI) * np.cosh(PI) * (1.0 + KA / KB))
C = A * np.sinh(PI)
B = (KA / KB) * A * np.cosh(PI)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def exact(x, y):
    left = A * np.sinh(PI * x)
    right = B * np.sinh(PI * (x - 1)) + C * np.cosh(PI * (x - 1))
    return np.where(x <= 1.0, left, right) * np.sin(PI * y)


def _spectral_slab(x0, k, order, outer_tag, outer_value, iface_tag):
    d = jno.shape.rect(x0, 0.0, x0 + 1.0, 1.0).domain()
    U = d.unknown(space="spectral", order=order)
    x, y, _ = d.variable("interior", split=True)
    xo, yo, _ = d.variable(outer_tag, split=True)
    u = U.bind(x=x, y=y)
    cons = [-k * (u.xx + u.yy), U(xo, yo) - outer_value(xo, yo)]
    for tag in ("top", "bottom"):
        xt, yt, _ = d.variable(tag, split=True)
        cons.append(U(xt, yt) - 0.0)
    xi, yi, _ = d.variable(iface_tag, split=True)
    ub = U.bind(x=xi, y=yi)
    return jno.spectral(cons), ub, ub.d(d.variable(iface_tag, normals=True)), U


def _fem_slab(x0, k, h, outer_tag, outer_value, iface_tag):
    d = jno.shape.rect(x0, 0.0, x0 + 1.0, 1.0, size=h).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xo, yo, _ = d.variable(outer_tag, split=True)
    xt, yt, _ = d.variable("top", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    xs, ys, _ = d.variable(iface_tag, split=True)
    ub = u.bind(x=xs, y=ys)
    prob = jno.fem([k * (ui.x * vi.x + ui.y * vi.y), u(xo, yo) - outer_value(xo, yo), u(xt, yt) - 0.0, u(xb, yb) - 0.0])
    return prob, ub, ub.d(d.variable(iface_tag, normals=True)), d


def test_two_spectral_slabs_with_different_orders_keep_spectral_accuracy():
    pa, ua, fa, Ua = _spectral_slab(0.0, KA, (16, 14), "left", lambda x, y: 0.0 * x, "right")
    pb, ub, fb, Ub = _spectral_slab(1.0, KB, (14, 19), "right", lambda x, y: jnn.sin(PI * y), "left")
    c = couple([pa, pb], [ua - ub, KA * fa + KB * fb], CouplingSpec(tol=1e-10, max_iter=60))
    sa, sb = c.solve()
    Pa, Pb = Ua.basis.nodes, Ub.basis.nodes
    err = max(np.abs(np.asarray(sa) - exact(*Pa.T)).max(), np.abs(np.asarray(sb) - exact(*Pb.T)).max())
    assert err < 1e-9, err


def test_spectral_and_fem_slabs_through_jno_core():
    pa, ua, fa, Ua = _spectral_slab(0.0, KA, (16, 16), "left", lambda x, y: 0.0 * x, "right")
    pb, ub, fb, db = _fem_slab(1.0, KB, 0.05, "right", lambda x, y: jnn.sin(PI * y), "left")
    core = jno.core([pa, pb, ua - ub, KA * fa + KB * fb])
    sa, sb = core.solve(coupling=jno.solve.coupling(tol=1e-9, max_iter=60))
    Pa = Ua.basis.nodes
    Pb = np.asarray(db.mesh_connectivity["points"])[:, :2]
    ea = np.abs(np.asarray(sa) - exact(*Pa.T)).max()
    eb = np.abs(np.asarray(sb).reshape(-1) - exact(*Pb.T)).max()
    assert eb < 2e-3 and ea < 2e-3, (ea, eb)  # the fem side's O(h²) sets the error
