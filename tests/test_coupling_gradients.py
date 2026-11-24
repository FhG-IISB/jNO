"""Gradients through a partitioned coupling (jno._coupling.driver): the coupled solution differentiated by a trainable
parameter of one participant, implicitly at the interface fixed point.

Steady: two slabs in series, [0, 1] (fvm, conductivity k_a, held at 0) and [1, 2] (fem, k_b, held at 1): the
interface value is k_b / (k_a + k_b). Transient: two solids brought into contact (as in test_coupling_transient),
differentiated by the hot side's conductivity, against a central difference of the coupled solve.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
from jno._coupling.driver import CouplingSpec, couple


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _trainable(name, value, stored=True):
    """A trainable scalar initialised to ``value``; ``stored``: its stored value set too (``initialize`` acts only in
    a jno.core setup, until then the parameter holds 0)."""
    k = jno.np.parameter((1,), name=name)
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(value))
    k.optimizer(optax.adam(1e-2))
    if stored:
        k.model.module = eqx.tree_at(lambda m: m.value, k.model.module, jnp.asarray([value]))
    return k


def _fvm_slab(k, x0, h, outer_tag, outer_value, iface_tag, time=None, make=jno.fvm):
    """-k Δu = 0 (or u.t = k Δu from u = outer_value) on [x0, x0 + 1] x [0, h_y], held at ``outer_value`` on
    ``outer_tag``, insulated top and bottom; its value and flux views on ``iface_tag``. ``make``: jno.fvm or
    jno.fdm."""
    d = jno.shape.rect(x0, 0.0, x0 + 1.0, 0.1 if time else 1.0, size=h).domain(**({"time": time} if time else {}))
    x, y, t = d.variable("interior", split=True)
    xo, yo, _ = d.variable(outer_tag, split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    cons = [-k * (ui.xx + ui.yy), u(xo, yo) - outer_value]
    if time:
        xi, yi, _ = d.variable("initial", split=True)
        cons = [ui.t + cons[0], cons[1], u(xi, yi) - (outer_value + 0.0 * xi)]
    for tag in ("top", "bottom"):
        xt, yt, _ = d.variable(tag, split=True)
        cons.append(u.bind(x=xt, y=yt).d(d.variable(tag, normals=True)) - 0.0)
    xs, ys, _ = d.variable(iface_tag, split=True)
    ub = u.bind(x=xs, y=ys)
    return make(cons), ub, ub.d(d.variable(iface_tag, normals=True)), d


def _fem_slab(k, x0, h, outer_tag, outer_value, iface_tag, time=None):
    d = jno.shape.rect(x0, 0.0, x0 + 1.0, 0.1 if time else 1.0, size=h).domain(**({"time": time} if time else {}))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xo, yo, _ = d.variable(outer_tag, split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    cons = [k * (ui.x * vi.x + ui.y * vi.y), u(xo, yo) - outer_value]
    if time:
        ci = d.variable("initial", split=True)
        cons = [ui.t * vi + cons[0], cons[1], u(*ci) - (outer_value + 0.0 * ci[0])]
    xs, ys, _ = d.variable(iface_tag, split=True)
    ub = u.bind(x=xs, y=ys)
    return jno.fem(cons), ub, ub.d(d.variable(iface_tag, normals=True)), d


def test_a_steady_coupling_differentiates_by_a_conductivity_like_the_series_resistance():
    # Name the break: the coupled root is differentiated implicitly (custom_root at the interface fixed point). An
    # eager iteration under a trace raises (float of a traced error); a gradient through the unrolled sweeps would
    # carry the iteration's error. Closed form: fvm [0, 1] (k_a, u = 0 on the left) and fem [1, 2] (k_b = 10, u = 1
    # on the right) in series: the interface value T_i = q / k_a with q = 1 / (1/k_a + 1/k_b), so
    # T_i = k_b / (k_a + k_b) and dT_i/dk_a = -k_b / (k_a + k_b)².
    ka0, kb = 1.0, 10.0
    k = _trainable("ka", ka0)
    pa, ua, fa, _ = _fvm_slab(k, 0.0, 0.1, "left", 0.0, "right")
    pb, ub, fb, db = _fem_slab(kb, 1.0, 0.07, "right", 1.0, "left")
    c = couple([pa, pb], [ua - ub, k * fa + kb * fb], CouplingSpec(tol=1e-12, max_iter=60))
    pts = np.asarray(db.mesh_connectivity["points"])
    iface = jnp.asarray(np.isclose(pts[:, 0], 1.0))
    lid = k.model.layer_id

    def t_i(ka):
        _, sb = c.solve(params={lid: ka})
        return jnp.sum(jnp.where(iface, jnp.asarray(sb), 0.0)) / jnp.sum(iface)

    ka = jnp.asarray([ka0])
    # direct participant solves, the coupling to 1e-12: round-off
    np.testing.assert_allclose(float(t_i(ka)), kb / (ka0 + kb), atol=1e-10)
    np.testing.assert_allclose(float(jax.grad(t_i)(ka)[0]), -kb / (ka0 + kb) ** 2, rtol=1e-8)


def test_a_transient_coupling_differentiates_by_a_conductivity_like_a_central_difference():
    # Name the break: every window's fixed point is differentiated implicitly and the windows chain (a window starts
    # from the last one's states, which carry the gradient). Oracle: a central difference of the coupled solve in
    # k_b, each solve converged to 1e-12 with direct steps (the difference's error, O(h²), is ~1e-9 relative).
    k1, kb0, n_time = 1.0, 4.0, 6
    k = _trainable("kb", kb0)
    pa, ua, fa, da = _fem_slab(k1, -1.0, 0.04, "left", 0.0, "right", time=(0.0, 0.005, n_time))
    pb, ub, fb, _ = _fvm_slab(k, 0.0, 0.03, "right", 1.0, "left", time=(0.0, 0.005, n_time))
    c = couple([pa, pb], [ua - ub, k1 * fa + k * fb], CouplingSpec(tol=1e-12, max_iter=60))
    lid = k.model.layer_id

    def heat(kb):  # the cold side's mean temperature at the end
        ta, _ = c.solve(params={lid: kb})
        return jnp.mean(jnp.asarray(ta)[-1])

    kb = jnp.asarray([kb0])
    g = float(jax.grad(heat)(kb)[0])
    h = 1e-4 * kb0
    fd = (float(heat(kb + h)) - float(heat(kb - h))) / (2 * h)
    assert abs(fd) > 1e-3  # the conductivity does move the contact
    np.testing.assert_allclose(g, fd, rtol=1e-6)


def test_jno_core_trains_a_conductivity_through_a_coupling():
    # Name the break: with a trainable parameter in a coupled problem, core.solve(coupling=...) returns the coupled
    # solution as trace nodes of that parameter, so a jno.core loss on it trains the parameter through the coupling
    # (as fem.solve() and jno.dd's coupled solves do). Returned as arrays, the loss has no parameter to train.
    # Target: the fem slab's exact solution at k_a = 2, T = T_i + (1 - T_i)(x - 1) with T_i = k_b/(k_a + k_b).
    from jno.trace import FunctionCall

    ka_true, kb = 2.0, 10.0
    k = _trainable("ka", 1.0, stored=False)  # as a user writes it: jno.core applies the initializer
    k.optimizer(optax.adam(1e-1))
    pa, ua, fa, _ = _fvm_slab(k, 0.0, 0.1, "left", 0.0, "right")
    pb, ub, fb, db = _fem_slab(kb, 1.0, 0.07, "right", 1.0, "left")
    sa, sb = jno.core([pa, pb, ua - ub, k * fa + kb * fb]).solve(coupling=jno.solve.coupling(tol=1e-12, max_iter=60))
    assert isinstance(sb, FunctionCall)
    x = np.asarray(db.mesh_connectivity["points"])[:, 0]
    t_i = kb / (ka_true + kb)
    crux = jno.core([(sb - jnp.asarray(t_i + (1.0 - t_i) * (x - 1.0))).mse])
    crux.solve(120)
    rec = float(np.asarray(crux.eval([k])).reshape(-1)[0])
    assert abs(rec - ka_true) < 2e-2, f"the coupling did not train k_a: {rec:.4f}"


def test_three_kinds_in_series_differentiate_by_the_middle_conductivity():
    # Name the break: the fixed point of three participants of three kinds (fvm | fem | fdm), the floating fem middle
    # on Robin-Robin pairs, differentiates through all three solves. Closed form: slabs [0,1], [1,2], [2,3] with k_1,
    # k_2, k_3, held at 0 and 1 at the ends: q = 1/S with S = Σ 1/k_i, the second interface T_2 = (1/k_1 + 1/k_2)/S,
    # and dT_2/dk_2 = -(1/k_2²)(1/k_3)/S².
    k1, k20, k3 = 1.0, 10.0, 2.0
    k = _trainable("k2", k20)
    pa, ua, fa, _ = _fvm_slab(k1, 0.0, 0.1, "left", 0.0, "right")
    pc, uc, fc, dc = _fvm_slab(k3, 2.0, 0.09, "right", 1.0, "left", make=jno.fdm)
    db = jno.shape.rect(1.0, 0.0, 2.0, 1.0, size=0.07).domain()
    T, v = db.fem_symbols()
    xi, yi, _ = db.variable("interior", split=True)
    pb = jno.fem([k * (T.bind(x=xi, y=yi).x * v.bind(x=xi, y=yi).x + T.bind(x=xi, y=yi).y * v.bind(x=xi, y=yi).y)])
    sides = {}
    for tag in ("left", "right"):
        xs, ys, _ = db.variable(tag, split=True)
        tb = T.bind(x=xs, y=ys)
        sides[tag] = (tb, tb.d(db.variable(tag, normals=True)))
    conds = [ua - sides["left"][0], k1 * fa + k * sides["left"][1], sides["right"][0] - uc]
    conds.append(k * sides["right"][1] + k3 * fc)
    c = couple([pa, pb, pc], conds, CouplingSpec(tol=1e-12, max_iter=80))
    left = jnp.asarray(np.isclose(np.asarray(dc.mesh_connectivity["points"])[:, 0], 2.0))
    lid = k.model.layer_id

    def t2(k2):
        _, _, sc = c.solve(params={lid: k2})
        return jnp.sum(jnp.where(left, jnp.asarray(sc).reshape(-1), 0.0)) / jnp.sum(left)

    S = 1.0 / k1 + 1.0 / k20 + 1.0 / k3
    k2 = jnp.asarray([k20])
    np.testing.assert_allclose(float(t2(k2)), (1.0 / k1 + 1.0 / k20) / S, atol=1e-9)
    np.testing.assert_allclose(float(jax.grad(t2)(k2)[0]), -(1.0 / k20**2) * (1.0 / k3) / S**2, rtol=1e-7)


def test_a_coupling_probes_a_parameter_at_its_initial_value_before_jno_core_sets_it():
    # Name the break: a parameter given only .initialize(...) holds 0 until a jno.core setup, and the coupling's
    # structural probes (the interface stiffness) solved at 0: a singular slab (-0 Δu), no stiffness, the value
    # condition on the wrong side. They read the parameter's initializer now. With k_a = 1 against k_b = 10 the
    # softer fvm slab takes the value (Giles 1997), as with a literal k_a = 1.
    ka0, kb = 1.0, 10.0
    k = _trainable("ka", ka0, stored=False)
    pa, ua, fa, _ = _fvm_slab(k, 0.0, 0.1, "left", 0.0, "right")
    pb, ub, fb, db = _fem_slab(kb, 1.0, 0.07, "right", 1.0, "left")
    c = couple([pa, pb], [ua - ub, k * fa + kb * fb], CouplingSpec(tol=1e-12, max_iter=60))
    assert c.parts[c.assign[0]].problem is pa
    np.testing.assert_allclose(sorted(c.stiffness.values()), [ka0, kb], rtol=1e-6)  # k/L of each unit slab
    _, sb = c.solve(params={k.model.layer_id: jnp.asarray([ka0])})
    iface = np.isclose(np.asarray(db.mesh_connectivity["points"])[:, 0], 1.0)
    np.testing.assert_allclose(np.asarray(sb)[iface], kb / (ka0 + kb), atol=1e-10)
