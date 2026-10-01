"""``fem.solve(k=2.0)`` solves at ``k = 2`` on every path that marches, not only on the steady ones.

``fem.solve`` turns a keyword naming a runtime parameter into ``values={...}``. The steady solves read it;
the others did not:

* a transient raised ``TypeError: SemidiscreteTimeBlock.solve() got an unexpected keyword 'values'``;
* a load-path (``.i(k)``) march dropped it and ran at the parameters' STORED values -- a wall prescribed
  as ``u = p`` came back at 0.000 for ``p = 0.25``, silently;
* the remeshing transient (``adapt=remesh(...)``) swallowed it the same way: ``k = 2`` marched as ``k = 0``,
  so the field never decayed.

Oracles: the named-value solve equals the same form with the parameter replaced by that constant (a path
known to be right), and equals the differentiable trace node evaluated at the value; an unknown or a
missing name raises.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


E = 1e-9


def _param(name):
    return jno.np.parameter((1,), name=name)


# ---------------------------------------------------------------------------------------------------
# first-order transient
# ---------------------------------------------------------------------------------------------------
def _heat(k, nonlinear, size=None):
    s = jno.shape.rect(0, 0, 1, 1, size=size) if size else jno.shape.rect(0, 0, 1, 1).structured(n=5)
    d = s.domain(time=(0.0, 0.1, 5))
    d.tag("wall", lambda x, y: (x < E) | (x > 1 - E) | (y < E) | (y > 1 - E))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    c = (1 + ub * ub) if nonlinear else 1.0
    ic = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    return jno.fem([ub.t * vb + k * c * (ub.x * vb.x + ub.y * vb.y), u(xw, yw) - 0.0, u(*ci) - ic])


def _eager(out):
    return np.asarray(out.fn() if hasattr(out, "fn") else out)


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_a_transient_marches_at_the_named_value(nonlinear):
    got = np.asarray(_heat(_param("k"), nonlinear).solve(k=2.0))
    twin = _eager(_heat(2.0, nonlinear).solve())
    node = np.asarray(_heat(_param("k"), nonlinear).solve().fn(jnp.array([2.0])))
    assert got.shape == twin.shape
    assert np.abs(twin[-1]).max() < 0.5 * np.abs(twin[0]).max(), "the oracle must actually decay"
    assert np.abs(got - twin).max() <= 1e-12
    assert np.abs(got - node).max() <= 1e-12


def test_a_transient_refuses_an_unknown_or_a_missing_name():
    with pytest.raises(TypeError, match="'q'"):  # not a parameter of this form: it is not swallowed
        _heat(_param("k"), False).solve(k=1.0, q=2.0)
    fem = jno.fem(_heat_two_params())
    with pytest.raises(ValueError, match="no value was given"):
        fem.solve(k=1.0)


def _heat_two_params():
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(time=(0.0, 0.1, 3))
    d.tag("wall", lambda x, y: (x < E) | (x > 1 - E) | (y < E) | (y > 1 - E))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    return [
        ub.t * vb + _param("k") * (ub.x * vb.x + ub.y * vb.y) - _param("q") * vb,
        u(xw, yw) - 0.0,
        u(*ci) - 0.0 * ci[0],
    ]


# ---------------------------------------------------------------------------------------------------
# load-path march: the default grid, an explicit schedule, and the adaptive pilot
# ---------------------------------------------------------------------------------------------------
def _ramp(p):
    """A displacement-controlled ramp with a parameter-valued grip: ``u = p`` on x = 0, ``u = 0.5 + τ²`` on
    x = 1, ``(1 + u²) ∇u·∇v`` with a step-history term so the form marches."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(tau=(0.0, 1.0, 4))
    d.tag("left", lambda x, y: x < E)
    d.tag("right", lambda x, y: x > 1 - E)
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, tr = d.variable("right", split=True)
    ub, vb = u.bind(x=xi, y=yi, tau=ti), v.bind(x=xi, y=yi, tau=ti)
    return jno.fem(
        [
            (1 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) + 1e-3 * (ub - ub.i(-1)) * vb,
            u(xl, yl) - p,
            u(xr, yr) - (0.5 + tr * tr),
        ]
    )


@pytest.mark.parametrize(
    "tau",
    [None, np.array([0.0, 0.2, 0.45, 0.7, 1.0]), "adaptive"],
    ids=["grid", "explicit_schedule", "adaptive_pilot"],
)
def test_a_load_path_march_runs_at_the_named_value(tau):
    kw = {} if tau is None else {"tau": jno.solve.adaptive(limit=0.2) if isinstance(tau, str) else tau}
    fem = _ramp(_param("p"))
    got = np.asarray(fem.solve(p=0.25, **kw))
    twin = np.asarray(_ramp(0.25).solve(**kw))
    x = np.asarray(fem.points)[:, 0]
    assert np.abs(got[:, x < E] - 0.25).max() < 1e-10, "the grip was not held at the named value"
    assert np.abs(got - twin).max() <= 1e-9 * np.abs(twin).max()


def test_the_load_path_node_still_resolves_through_crux_values():
    node = _ramp(_param("p")).solve()
    eager = np.asarray(_ramp(_param("p")).solve(p=0.25))
    assert np.abs(np.asarray(node.fn(jnp.array([0.25]))) - eager).max() <= 1e-12


# ---------------------------------------------------------------------------------------------------
# remeshing transient
# ---------------------------------------------------------------------------------------------------
def test_a_remeshing_transient_marches_at_the_named_value():
    pytest.importorskip("mmgpy", reason="mmgpy required for adaptive remeshing")

    def spec():
        return jno.solve.remesh(every=2, max_dofs=200)

    got = _heat(_param("k"), False, size=0.2).solve(adapt=spec(), k=2.0)
    twin = _heat(2.0, False, size=0.2).solve(adapt=spec())
    peaks = [float(np.abs(np.asarray(s)).max()) for s in got.states]
    want = [float(np.abs(np.asarray(s)).max()) for s in twin.states]
    assert want[-1] < 0.2 * want[0], "the oracle must actually decay"
    assert np.allclose(peaks, want, rtol=1e-9, atol=1e-12)


def test_a_remeshing_transient_refuses_a_parametric_form_without_values():
    pytest.importorskip("mmgpy", reason="mmgpy required for adaptive remeshing")
    with pytest.raises(NotImplementedError, match="Name the values"):
        _heat(_param("k"), False, size=0.2).solve(adapt=jno.solve.remesh(every=2, max_dofs=200))

