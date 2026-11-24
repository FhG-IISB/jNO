"""Marching a transient problem over a window from a given state: ``solve(time=..., x0=state, span=(t0, t1))``.

A partitioned coupling advances every participant window by window and re-runs a window while it iterates,
so a march must start from any state at any time. Marching ``[0, T]`` in two windows, the second from the
state the first ended in, must give the same trajectory as one march, to the round-off of the step solves:
the same steps are taken on the same grid.
"""

import jax
import numpy as np
import pytest

import jno

T, N_TIME = 0.1, 11  # dt = 0.01


def _arr(out):
    """The trajectory as an array: a real transient ``fem.solve()`` is a lazy node, evaluated by ``.fn()``."""
    return np.asarray(out.fn() if hasattr(out, "fn") else out)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _strong_heat(make):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain(time=(0.0, T, N_TIME))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    u0 = jno.np.sin(np.pi * xi) * jno.np.sin(np.pi * yi)
    return make([ui.t - (ui.xx + ui.yy), u(xb, yb) - 0.0, u(xi, yi) - u0])


def _fem_heat():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain(time=(0.0, T, N_TIME))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    return jno.fem([ub.t * vb + ub.x * vb.x + ub.y * vb.y, u(xb, yb) - 0.0, u(*ci) - u0])


CASES = {
    "fdm-euler": (lambda: _strong_heat(jno.fdm), None),
    "fdm-cn": (lambda: _strong_heat(jno.fdm), lambda: jno.solve.theta(0.5)),
    "fvm-euler": (lambda: _strong_heat(jno.fvm), None),
    "fvm-ssprk3": (lambda: _strong_heat(jno.fvm), lambda: jno.solve.erk("ssprk3", cfl=0.5)),
    "fem-euler": (_fem_heat, None),
    "fem-cn": (_fem_heat, lambda: jno.solve.theta(0.5)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_two_windows_reproduce_one_march(case):
    make, scheme = CASES[case]
    prob = make()
    time = None if scheme is None else scheme()
    full = _arr(prob.solve(time=time))
    first = _arr(prob.solve(time=time, span=(0.0, 0.05)))
    second = _arr(prob.solve(time=time, x0=first[-1], span=(0.05, T)))
    assert first.shape[0] == 6 and second.shape[0] == 6  # the grid points of each window, both ends included
    assert np.abs(full[-1]).max() > 0.05  # the march decays but has not died out
    scale = np.abs(full).max()
    # the same steps on the same grid: equal up to the round-off of the step solves
    np.testing.assert_allclose(first, full[:6], atol=1e-11 * scale)
    np.testing.assert_allclose(second, full[5:], atol=1e-10 * scale)


@pytest.mark.parametrize("case", ["fdm-euler", "fvm-euler", "fem-euler"])
def test_x0_replaces_the_initial_condition(case):
    make, _ = CASES[case]
    prob = make()
    full = _arr(prob.solve())
    twice = _arr(prob.solve(x0=2.0 * full[0]))  # a linear problem: twice the state, twice the march
    np.testing.assert_allclose(twice, 2.0 * full, atol=1e-10 * np.abs(full).max())


@pytest.mark.parametrize("case", ["fdm-euler", "fvm-euler", "fem-euler"])
def test_a_window_after_the_start_needs_its_state(case):
    make, _ = CASES[case]
    with pytest.raises(ValueError, match="x0="):
        _arr(make().solve(span=(0.05, T)))


@pytest.mark.parametrize(
    "case, window", [("fdm-euler", 0.05), ("fvm-euler", 0.05), ("fvm-ssprk3", 0.01), ("fem-euler", 0.05), ("fem-cn", 0.05)]
)
def test_a_participant_marches_window_by_window_like_one_march(case, window):
    from jno._coupling.participant import participant

    make, scheme = CASES[case]
    prob = make()
    time = None if scheme is None else scheme()
    full = _arr(prob.solve(time=time))
    p = participant(prob)
    state, t = full[0], 0.0
    steps = int(round(window / 0.01))
    for k in range(int(round(T / window))):
        state = np.asarray(p.march(prob, state, t, window, {}, time=time)[0])  # (end, frames)
        t += window
        # an autonomous problem marches every window as [0, window]: one compiled march for all of them
        np.testing.assert_allclose(state, full[(k + 1) * steps], atol=1e-10 * np.abs(full).max())
