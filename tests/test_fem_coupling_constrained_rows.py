"""A nonlocal ``Coupling`` is zeroed on EVERY essential row, not only the constant-valued ones.

The coupling's contribution is added to the assembled residual and must stay off the Dirichlet rows, whose
residual is ``u - g``: a contribution ``c`` left there holds the wall at ``g - c``. Its row set was read off
the domain's CONSTANT pairs only, so a wall whose value moves with τ or t, or rides the runtime args (a
parameter or a net in the value), took the contribution -- measured: a coupling of 0.1 held a wall at 0.4
where 0.5 was prescribed, on steady, load-path and transient forms alike, with nothing said.

Also: a coupling on a LINEAR form promotes it to a residual operator, and that operator now states its
essential rows (``dirichlet_dofs``) as every other residual operator does, so an extrapolating driver
(``staggered(over_relax>1)``) leaves them alone.

Oracles: every wall dof holds its prescribed value; a parameter-valued wall evaluated at ``g`` gives the
same field as the constant wall ``g`` under the same coupling; the coupling still acts in the interior.
"""

from __future__ import annotations

import os
import types

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


LOAD = 0.1  # the coupling: a constant nodal load on every row
G = 0.5
E = 1e-9


def _load(u):
    return LOAD * jnp.ones_like(u)


def _square(**grid):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain(**grid)
    d.tag("left", lambda x, y: x < E)
    d.tag("right", lambda x, y: x > 1 - E)
    return d


def _walls(fem):
    x = np.asarray(fem.points)[:, 0]
    return x < E, x > 1 - E


def _array(out, *values):
    if values:
        return np.asarray(out.fn(*[jnp.array([v]) for v in values]))
    return np.asarray(out.fn() if hasattr(out, "fn") else out)


# ---------------------------------------------------------------------------------------------------
# steady: a parameter-valued wall, linear (promoted by the coupling) and nonlinear
# ---------------------------------------------------------------------------------------------------
def _steady(right, nonlinear, coupled=True):
    d = _square()
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, *_ = d.variable("interior", split=True)
    xl, yl, *_ = d.variable("left", split=True)
    xr, yr, *_ = d.variable("right", split=True)
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    c = (1 + ub * ub) if nonlinear else 1.0
    terms = [c * (ub.x * vb.x + ub.y * vb.y), u(xl, yl) - 0.25, u(xr, yr) - right]
    if coupled:
        terms.append(jno.Coupling(_load, name="load"))
    return jno.fem(terms)


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_a_parameter_valued_wall_holds_under_a_coupling(nonlinear):
    fem = _steady(jno.np.parameter((1,), name="g"), nonlinear)
    got = _array(fem.solve(), G)
    left, right = _walls(fem)
    assert np.abs(got[right] - G).max() < 1e-10, "the coupling shifted the parameter-valued wall"
    assert np.abs(got[left] - 0.25).max() < 1e-10

    twin = _array(_steady(G, nonlinear).solve())  # the constant wall: the path that was always right
    assert np.abs(got - twin).max() <= 1e-9

    free = _array(_steady(G, nonlinear, coupled=False).solve())
    assert np.abs(got - free).max() > 1e-3, "the coupling must still act on the free rows"


# ---------------------------------------------------------------------------------------------------
# load path: a τ-ramped wall (this branch's regression -- the coupling used to be refused on a history form)
# ---------------------------------------------------------------------------------------------------
def test_a_tau_ramped_wall_holds_under_a_coupling_on_the_load_path():
    d = _square(tau=(0.0, 1.0, 3))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, tr = d.variable("right", split=True)
    ub, vb = u.bind(x=xi, y=yi, tau=ti), v.bind(x=xi, y=yi, tau=ti)
    fem = jno.fem(
        [
            (1 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) + 1e-3 * (ub - ub.i(-1)) * vb,
            u(xl, yl) - 0.25,
            u(xr, yr) - (0.5 + tr),
            jno.Coupling(_load, name="load"),
        ]
    )
    ys = _array(fem.solve())
    left, right = _walls(fem)
    for k, tau in enumerate([0.0, 0.5, 1.0]):
        assert np.abs(ys[k, right] - (0.5 + tau)).max() < 1e-10, f"step {k}: the coupling shifted the ramped wall"
        assert np.abs(ys[k, left] - 0.25).max() < 1e-10


# ---------------------------------------------------------------------------------------------------
# first-order transient: a g(x, t) wall and a parameter-valued wall, linear and nonlinear
# ---------------------------------------------------------------------------------------------------
def _transient(right, nonlinear):
    d = _square(time=(0.0, 0.2, 3))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, tr = d.variable("right", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    c = (1 + ub * ub) if nonlinear else 1.0
    g = right(tr) if isinstance(right, types.FunctionType) else right  # a parameter is callable too
    return jno.fem(
        [
            ub.t * vb + c * (ub.x * vb.x + ub.y * vb.y),
            u(xl, yl) - 0.25,
            u(xr, yr) - g,
            u(*ci) - 0.0 * ci[0],
            jno.Coupling(_load, name="load"),
        ]
    )


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_a_time_varying_wall_holds_under_a_coupling(nonlinear):
    fem = _transient(lambda t: 0.5 + t, nonlinear)
    ys = _array(fem.solve())
    left, right = _walls(fem)
    for k, t in [(1, 0.1), (2, 0.2)]:
        assert np.abs(ys[k, right] - (0.5 + t)).max() < 1e-10, f"step {k}: the coupling shifted g(x, t)"
        assert np.abs(ys[k, left] - 0.25).max() < 1e-10


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_a_parameter_valued_wall_holds_under_a_coupling_on_a_transient(nonlinear):
    fem = _transient(jno.np.parameter((1,), name="g"), nonlinear)
    ys = _array(fem.solve(), G)
    _, right = _walls(fem)
    assert np.abs(ys[1:, right] - G).max() < 1e-10, "the coupling shifted the parameter-valued wall"
    twin = _array(_transient(G, nonlinear).solve())
    assert np.abs(ys - twin).max() <= 1e-9


# ---------------------------------------------------------------------------------------------------
# the promoted linear operator states its essential rows
# ---------------------------------------------------------------------------------------------------
def test_a_coupling_on_a_linear_form_keeps_the_dirichlet_dof_set():
    fem = _steady(G, nonlinear=False)
    assert fem._mode == "nonlinear"  # promoted by the coupling
    left, right = _walls(fem)
    want = np.flatnonzero(left | right)
    got = np.sort(np.asarray(fem._op.dirichlet_dofs))
    assert np.array_equal(got, want)
