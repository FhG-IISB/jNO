"""What a coupling reads from each kind of problem (jno._coupling.participant).

u = 1 + 2x on the unit square (u = 1 on the left, u = 3 on the right, insulated top and bottom) is linear, so
every discretisation holds it exactly: on the right edge a participant must read the value 3 and the outward
flux du/dn = 2 at every interface node, from its own discrete flux (the finite-volume boundary rule, the
finite-difference boundary stencil, the finite-element consistent reaction).
"""

import jax
import numpy as np
import pytest

import jno
from jno._coupling.participant import participant


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _strong(make):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain()
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui, ur = u.bind(x=x, y=y), u.bind(x=xr, y=yr)
    cons = [-(ui.xx + ui.yy), u(xl, yl) - 1.0, u(xr, yr) - 3.0]
    for tag in ("top", "bottom"):
        xt, yt, _ = d.variable(tag, split=True)
        cons.append(u.bind(x=xt, y=yt).d(d.variable(tag, normals=True)) - 0.0)
    return make(cons), ur, ur.d(nr)


def _fem():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    ui, vi, ur = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), u.bind(x=xr, y=yr)
    return jno.fem([ui.x * vi.x + ui.y * vi.y, u(xl, yl) - 1.0, u(xr, yr) - 3.0]), ur, ur.d(nr)


CASES = {
    "fdm": lambda: _strong(jno.fdm),
    "fvm-cell": lambda: _strong(jno.fvm),
    "fvm-vertex": lambda: _strong(lambda c: jno.fvm(c, centering="vertex")),
    "fem": _fem,
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_participant_reads_value_and_flux_of_a_linear_solution(case):
    prob, value, flux = CASES[case]()
    p = participant(prob)
    state = p.solve(prob, {})
    iface = p.interface("right")
    assert p.owns(value) and not p.owns(jno.shape.rect(0, 0, 1, 1).domain().unknown())
    np.testing.assert_allclose(np.asarray(p.sample(prob, state, value, "right", iface)), 3.0, atol=1e-10)
    # each problem solves with its own default: the iterative ones (fdm, fem) stop at a relative residual near
    # 1e-8, which the flux, a gradient over h = 0.1, carries as a few 1e-7
    np.testing.assert_allclose(np.asarray(p.sample(prob, state, flux, "right", iface)), 2.0, atol=1e-6)
