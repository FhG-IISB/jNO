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


def _steady_gas_2d():
    """A steady conducting gas in a box: a reservoir at rest (p = 1, T = 1) on the left, slip walls elsewhere, the
    right one held at T = 2; the temperature is an algebraic unknown T = p/rho. And a state at rest to start from."""
    import jno.jnp_ops as jnn

    gamma = 1.4
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(8, 8)).quad().domain()
    x, y, _ = d.variable("interior", split=True)
    q, T = d.unknown.vector(4), d.unknown()
    qi, Ti = q.bind(x=x, y=y), T.bind(x=x, y=y)
    rho, mx, my, En = qi[0], qi[1], qi[2], qi[3]
    u, v = mx / rho, my / rho
    p = (gamma - 1.0) * (En - 0.5 * (mx * u + my * v))
    fv = jno.fv(flux="roe")
    Fx = jnn.stack([mx, mx * u + p, my * u, u * (En + p)], axis=-1)
    Fy = jnn.stack([my, mx * v, my * v + p, v * (En + p)], axis=-1)
    cons = [Fx.d(x, scheme=fv) + Fy.d(y, scheme=fv) - jnn.stack([0, 0, 0, 0.1 * (Ti.xx + Ti.yy)], axis=-1)]
    cons.append(Ti - p / rho)
    xl, yl, _ = d.variable("left", split=True)
    rest = jnn.stack([1.0 + 0.0 * xl, 0.0 * xl, 0.0 * xl, 1.0 / (gamma - 1.0) + 0.0 * xl], axis=-1)
    cons += [q(xl, yl) - rest, T(xl, yl) - 1.0]
    for tag in ("right", "top", "bottom"):
        xe, ye, _ = d.variable(tag, split=True)
        ne = d.variable(tag, normals=True)
        qe = q.bind(x=xe, y=ye)
        cons.append(qe[1] * ne[..., 0] + qe[2] * ne[..., 1])
    xr, yr, _ = d.variable("right", split=True)
    cons.append(T(xr, yr) - 2.0)
    gas = jno.fvm(cons)
    n = gas.centroids.shape[0]
    return gas, np.concatenate([np.ones(n), np.zeros(n), np.zeros(n), np.full(n, 1.0 / (gamma - 1.0)), np.ones(n)])


def test_a_steady_gas_participant_solves_from_its_start():
    # Name the break: a steady participant's solve is compiled with its start traced, and the stencil of its
    # Jacobian was checked at zeros, where a gas has no density ("could not assemble"). Its concrete start is
    # recorded for that check now: the participant's solve is the eager solve from the same start.
    gas, at_rest = _steady_gas_2d()
    coupled = np.asarray(participant(gas).solve(gas, {}, x0=at_rest))
    alone, _ = _steady_gas_2d()
    np.testing.assert_allclose(coupled, np.asarray(alone.solve(x0=at_rest)).reshape(-1), atol=1e-12)


def test_a_steady_roe_gas_solves_on_a_cpu_with_two_threads():
    # Name the break: on a CPU with four threads or fewer (a CI runner) XLA:CPU (JAX 0.10.1) deadlocked this solve.
    # The small per-face blocks (the Roe |A| inverses, the face Newton steps) went to batched LAPACK kernels; run side
    # by side, each split its batch over the same thread pool and waited for it, every thread blocked. A fresh
    # process pinned to two threads, since JAX sizes its CPU thread pool when it starts.
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import os, sys; os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2]); "
        f"sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        "import jax; jax.config.update('jax_enable_x64', True); import numpy as np; "
        "from test_coupling_participants import _steady_gas_2d; gas, x0 = _steady_gas_2d(); "
        "print(float(np.abs(np.asarray(gas.solve(x0=x0))).max()))"
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    try:
        run = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=400)
    except subprocess.TimeoutExpired:
        pytest.fail("a steady Roe gas on 8 x 8 cells did not finish in 400 s on two CPU threads (an XLA:CPU deadlock)")
    assert run.returncode == 0, run.stderr[-3000:]
    assert np.isfinite(float(run.stdout.split()[-1]))


def _k_strip(kind, k0=2.0, transient=False):
    """-k Δu = 0 on the strip [0, 1] x [0, 0.2], u = 0 on the left, k ∂u/∂n = q on the right (q a data field of the
    participant), insulated top and bottom: u = q x / k, linear, so every discretisation holds it exactly. k is a
    trainable parameter (stored value k0: ``initialize`` acts only inside jno.core). ``transient``: u.t = k Δu from
    u = 0 instead, over t in [0, 0.04] in 4 steps."""
    import equinox as eqx
    import jax.numpy as jnp
    import optax

    d = jno.shape.rect(0.0, 0.0, 1.0, 0.2, size=0.1).domain(**({"time": (0.0, 0.04, 5)} if transient else {}))
    k = jno.np.parameter((1,), name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(k0))
    k.optimizer(optax.adam(1e-2))
    k.model.module = eqx.tree_at(lambda m: m.value, k.model.module, jnp.asarray([k0]))
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    if kind == "fem":
        u, v = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
        cons = [k * (ui.x * vi.x + ui.y * vi.y), u(xl, yl) - 0.0]
        if transient:
            ci = d.variable("initial", split=True)
            cons = [ui.t * vi + cons[0], cons[1], u(*ci) - 0.0 * ci[0]]
        part = participant(jno.fem(cons))
        q, setter = part.data("q")
        prob = part.rebuild([-1.0 * q * v.bind(x=xr, y=yr)])  # ∫ k ∇u·∇v - ∫_right q v: k ∂u/∂n = q
    else:
        make = {"fdm": jno.fdm, "fvm-cell": jno.fvm, "fvm-vertex": lambda c: jno.fvm(c, centering="vertex")}[kind]
        x, y, t = d.variable("interior", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y, t=t)
        cons = [-k * (ui.xx + ui.yy), u(xl, yl) - 0.0]
        if transient:
            xi, yi, _ = d.variable("initial", split=True)
            cons = [ui.t + cons[0], cons[1], u(xi, yi) - 0.0 * xi]
        for tag in ("top", "bottom"):
            xt, yt, _ = d.variable(tag, split=True)
            cons.append(u.bind(x=xt, y=yt).d(d.variable(tag, normals=True)) - 0.0)
        part = participant(make(cons))
        q, setter = part.data("q")
        prob = part.rebuild([k * u.bind(x=xr, y=yr).d(d.variable("right", normals=True)) - q])
    return part, prob, k.model.layer_id, setter, u.bind(x=xr, y=yr)


@pytest.mark.parametrize("case", list(CASES))
def test_a_participant_solves_under_jit_and_differentiates_by_its_data_and_a_parameter(case):
    # Name the break: a participant's solve and samples take the interface data and the trainable parameters as
    # traced inputs. Under jax.jit a stored-value route (a module swap, numpy spreading of the data) raises or reads
    # stale values, and a parameter read from its stored value has no gradient. On the right edge u = q/k:
    # 1.5 at q = 3, k = 2; d/dk = -q/k² = -0.75; along a uniform change of q, 1/k = 0.5.
    import jax.numpy as jnp

    part, prob, lid, setter, value = _k_strip(case)
    iface = part.interface("right")

    def right_mean(q, k):
        data = setter(part.nodal(iface, q), prob)
        state = part.solve(prob, data, params={lid: k})
        vals = part.sample(prob, state, value, "right", iface, data, params={lid: k})
        return jnp.sum(jnp.asarray(iface.weights) * vals) / iface.total

    q, k = jnp.full(iface.points.shape[0], 3.0), jnp.asarray([2.0])
    # direct solves of exact discretisations: round-off (the fem default is an iterative solve to ~1e-10)
    np.testing.assert_allclose(float(jax.jit(right_mean)(q, k)), 1.5, atol=1e-8)
    gq, gk = jax.jit(jax.grad(right_mean, argnums=(0, 1)))(q, k)
    np.testing.assert_allclose(float(gk[0]), -0.75, atol=1e-7)
    np.testing.assert_allclose(float(jnp.sum(gq)), 0.5, atol=1e-7)


@pytest.mark.parametrize("case", list(CASES))
def test_a_participant_marches_a_window_under_jit_and_differentiates_by_a_parameter(case):
    # Name the break: a window's march takes the data and the parameters as traced inputs (a coupling window compiled
    # once, its fixed point differentiated). Traced, a structural probe that converts a value raises; a stored-value
    # route returns the eager result at the stored k, whatever k is passed. Oracles: the eager march at the same
    # values, and a central difference in k (the problem is linear: the difference is exact to round-off).
    import jax.numpy as jnp

    part, prob, lid, setter, value = _k_strip(case, transient=True)
    iface = part.interface("right")
    start = part.initial(prob)

    def right_mean(q, k):
        data = setter(part.nodal(iface, q), prob)
        state, _ = part.march(prob, start, 0.0, 0.04, data, params={lid: k})
        vals = part.sample(prob, state, value, "right", iface, data, params={lid: k})
        return jnp.sum(jnp.asarray(iface.weights) * vals) / iface.total

    q, k = jnp.full(iface.points.shape[0], 3.0), jnp.asarray([2.0])
    traced = float(jax.jit(right_mean)(q, k))  # first: every structural decision is made inside the trace
    gk = float(jax.jit(jax.grad(right_mean, argnums=1))(q, k)[0])
    eager = float(right_mean(q, k))
    assert eager > 0.05  # the flux did heat the edge
    np.testing.assert_allclose(traced, eager, rtol=1e-10)
    h = 1e-4  # linear in q, smooth in k: the central difference's error is O(h²) ~ 1e-8 relative
    fd = (float(right_mean(q, k + h)) - float(right_mean(q, k - h))) / (2 * h)
    np.testing.assert_allclose(gk, fd, rtol=1e-6)


def _heated_strip(n_steps, k=2.0):
    """u.t = k Δu on the strip [0, 1] x [0, 0.2] from u = 0 over t in [0, 0.04], u = 0 on the left, k ∂u/∂n = q (a data
    field) on the right, insulated top and bottom: the heat the strip takes in is q times the right edge's length times
    the time, less what leaves through the left edge. Returns the participant, the problem, the data setter and the
    flux ``k ∂u/∂n`` on the left."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 0.2, size=0.1).domain(time=(0.0, 0.04, n_steps))
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    cons = [ui.t - k * (ui.xx + ui.yy), u(xl, yl) - 0.0, u(xi, yi) - 0.0 * xi]
    for tag in ("top", "bottom"):
        xt, yt, _ = d.variable(tag, split=True)
        cons.append(u.bind(x=xt, y=yt).d(d.variable(tag, normals=True)) - 0.0)
    part = participant(jno.fvm(cons))
    q, setter = part.data("q")
    prob = part.rebuild([k * u.bind(x=xr, y=yr).d(d.variable("right", normals=True)) - q])
    return part, prob, setter, k * u.bind(x=xl, y=yl).d(d.variable("left", normals=True))


@pytest.mark.parametrize("scheme", ["ssprk3", "imex"])
@pytest.mark.parametrize("n_steps", [5, 9, 17])
def test_an_explicit_march_hands_over_the_window_mean_flux_of_its_own_stages(scheme, n_steps):
    # Name the break: the window mean of the flux a sender gives out was the trapezoid rule over its frames, off at
    # the first step after a jump (here: u = 0 against the left edge's own u = 0 while q heats the strip, a
    # singular start). Oracle: the heat balance of the strip, which the scheme's own stage sum of the flux closes to
    # the transfer's lumping (1e-8, whatever the step), where the trapezoid is off by 5e-4 and falls as dt².
    time = jno.solve.erk("ssprk3", cfl=0.5) if scheme == "ssprk3" else jno.solve.imex("ars222", cfl=0.5)
    part, prob, setter, expr = _heated_strip(n_steps)
    left, right = part.interface("left"), part.interface("right")
    data = setter(part.nodal(right, np.full(right.points.shape[0], 3.0)), prob)
    start = part.initial(prob)
    end, frames, means = part.march(prob, start, 0.0, 0.04, data, time=time, flux=[(expr, "left", left)])
    moved = np.asarray(prob.volumes) @ (np.asarray(end) - np.asarray(start))
    inflow = 3.0 * right.total * 0.04
    w = np.asarray(left.weights)
    stage = inflow + 0.04 * (w @ np.asarray(means[0]))
    m = frames.shape[0] - 1
    tw = np.full(m + 1, 1.0 / m)
    tw[0] = tw[-1] = 0.5 / m
    sampled = [np.asarray(part.sample(prob, frames[k], expr, "left", left, data)) for k in range(m + 1)]
    trap = inflow + 0.04 * (w @ sum(a * b for a, b in zip(tw, sampled)))
    assert abs(stage - moved) < 1e-6 * moved
    assert abs(trap - moved) > 1e3 * abs(stage - moved)


def test_the_stage_mean_flux_of_a_window_differentiates_by_its_data():
    # The accumulator rides the traced march a coupling's gradient goes through: the derivative of the left edge's
    # window-mean heat out by the uniform load q (the problem is linear in q) against a central difference.
    import jax.numpy as jnp

    part, prob, setter, expr = _heated_strip(41)  # dt = 1e-3, inside the explicit limit: no cfl substeps (reverse mode
    # does not go through a traced substep count)
    left, right = part.interface("left"), part.interface("right")
    start, time = part.initial(prob), jno.solve.erk("ssprk3")
    w = jnp.asarray(left.weights)

    def out(q, start):  # the start is an argument too: a coupling's windows after the first start from traced states
        data = setter(part.nodal(right, jnp.full(right.points.shape[0], 1.0) * q), prob)
        return jnp.sum(w * part.march(prob, start, 0.0, 0.04, data, time=time, flux=[(expr, "left", left)])[2][0])

    q = jnp.asarray(3.0)
    g = float(jax.jit(jax.grad(out))(q, start))
    h = 1e-3
    fd = (float(jax.jit(out)(q + h, start)) - float(jax.jit(out)(q - h, start))) / (2 * h)
    assert abs(g) > 1e-3
    np.testing.assert_allclose(g, fd, rtol=1e-6)
