"""Trainables in a transient initial condition.

* A ``jno.np.parameter`` in an initial condition (``u(initial) - a * u0(x)``) was read once, at its stored value:
  jno.core never saw it (``No model for Model N``) and it could not train. A compound network IC
  (``(a + b x) * u0``) was refused, and a network IC on a nonlinear form was refused by a guard the marches no
  longer need. All three now re-form the initial state from the runtime args.
* An initial state given as a nodal DATA field marched from zero: its initializer is lazy, and the build read
  the field before it ran.
* Where the initial state IS fixed at build (1-D lines, non-nodal families, ``u_tt``), a trainable there is
  refused by name instead of silently frozen.

Oracle: the gradient jno.core takes (one SGD step) against central FD of the same loss.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jno
from tests.test_fem_exponential_trainable import T, _heat, _net_gradient, _scalar_gradient, _x64  # noqa: F401
from tests.test_fem_trainables_gaps import PI, _evaluate, _param, _rel, sin


def _line(a):
    d = jno.domain(constructor=jno.domain.line(mesh_size=0.1), time=T)
    u, v = d.fem_symbols()
    co, cb = d.variable("interior", split=True), d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=co[0], t=co[-1]), v.bind(x=co[0], t=co[-1])
    return [ui.t * vi + ui.x * vi.x, u(*cb[:-1]) - 0.0, u(ci[0]) - a * sin(PI * ci[0])]


_FIXED = {"be": lambda: jno.solve.theta(1.0), "sdirk3": lambda: jno.solve.sdirk(3)}


@pytest.mark.parametrize("scheme", list(_FIXED))
@pytest.mark.parametrize("cubic", [False, True], ids=["linear", "nonlinear"])
def test_a_parameter_in_the_initial_condition_trains(scheme, cubic):
    def build(a):
        return _heat(ic=lambda x, y: a * sin(PI * x) * sin(PI * y), cubic=cubic)

    g, fd, loss0 = _scalar_gradient(build, _FIXED[scheme])
    assert loss0 > 0 and abs(g) > 0
    # a Newton step converges to rtol 1e-8, and the FD of the converged march carries that noise
    assert _rel(g, fd) < (1e-5 if cubic else 1e-6), f"crux {g:.10e} vs FD {fd:.10e}"


def test_a_compound_network_initial_condition_trains():
    def build(k):
        return _heat(ic=lambda x, y: k(x) * sin(PI * x) * sin(PI * y))

    g, fd, loss0 = _net_gradient(build, _FIXED["be"])
    assert loss0 > 0 and abs(g) > 0
    assert _rel(g, fd) < 1e-6, f"crux {g:.10e} vs FD {fd:.10e}"


def test_a_network_initial_condition_on_a_nonlinear_form_trains():
    def build(k):
        return _heat(ic=lambda x, y: k(x), cubic=True)

    g, fd, loss0 = _net_gradient(build, _FIXED["be"])
    assert loss0 > 0 and abs(g) > 0
    assert _rel(g, fd) < 1e-6, f"crux {g:.10e} vs FD {fd:.10e}"


@pytest.mark.parametrize(
    "build",
    [
        lambda a: _heat(ic=lambda x, y: a * sin(PI * x) * sin(PI * y), space="Hermite"),
        _line,
        lambda a: _heat(ic=lambda x, y: a * sin(PI * x) * sin(PI * y), tt=True),
    ],
    ids=["hermite_2d", "line_1d", "second_order"],
)
def test_a_trainable_initial_condition_where_it_cannot_thread_is_refused(build):
    with pytest.raises(NotImplementedError, match="initial condition reads the trainable"):
        jno.fem(build(_param("a", 1.3)))


def test_an_optimizerless_data_field_initial_condition_is_not_a_trainable():
    """A known initial state handed in as a nodal field stays a data value, on every path."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.3).domain(time=T)
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pts = np.asarray(d.mesh.points)[:, :2]
    vals = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    g = jno.np.parameter(v, name="u0").initialize(lambda *a, **k: jnp.asarray(vals))
    g.dtype(jnp.float64)
    terms = [ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0]
    got = np.asarray(_evaluate(jno.fem(terms + [u(ci[0], ci[1]) - g]).solve()))
    ref = np.asarray(_evaluate(jno.fem(terms + [u(ci[0], ci[1]) - sin(PI * ci[0]) * sin(PI * ci[1])]).solve()))
    np.testing.assert_allclose(got, ref, atol=1e-10)
