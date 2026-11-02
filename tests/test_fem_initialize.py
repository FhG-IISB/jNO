"""u.initialize(value): a steady solve's starting point, and the initial state of a time-dependent problem.

Oracles: a problem with two solutions (u^2 = 1 on the whole domain: the start picks +1 or -1); the same start
given as a formula and as its nodal values (identical Newton iterates); the initial-condition term it stands for
(identical trajectories); a linear constant-unknown problem whose start does not change the answer.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

import jno

config.update("jax_enable_x64", True)
NEWTON = dict(direct=True, rtol=1e-12, atol=0.0, max_steps=40)


def _square(h=0.1, **kw):
    return jno.shape.rect(0, 0, 1, 1).sized(h).domain(**kw)


def _two_roots(d):
    """eps grad u . grad v + (u^2 - 1) v = 0 with natural boundaries: u = +1 and u = -1 both solve it."""
    x, y, _ = d.variable("interior", split=True)
    u, v = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    return u, x, y, jno.fem([1e-3 * (ui.d(x) * vi.d(x) + ui.d(y) * vi.d(y)) + (ui**2 - 1.0) * vi])


@pytest.mark.parametrize("start, root", [(0.5, 1.0), (-0.5, -1.0)])
def test_the_start_picks_the_root(start, root):
    d = _square()
    u, _, _, fem = _two_roots(d)
    u.initialize(start)
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(**NEWTON)))
    np.testing.assert_allclose(sol, root, atol=1e-10)


def test_it_is_read_at_each_solve():
    d = _square()
    u, _, _, fem = _two_roots(d)
    u.initialize(0.5)
    plus = np.asarray(fem.solve(nonlinear=jno.solve.newton(**NEWTON)))
    u.initialize(-0.5)
    minus = np.asarray(fem.solve(nonlinear=jno.solve.newton(**NEWTON)))
    assert plus.min() > 0.99 and minus.max() < -0.99


def test_a_formula_is_evaluated_at_the_nodes():
    d = _square()
    u, x, y, fem = _two_roots(d)
    u.initialize(0.3 + 0.2 * jno.np.sin(3.0 * x) * y)
    p = np.asarray(fem.points)
    np.testing.assert_allclose(np.asarray(fem._initialized_x0()), 0.3 + 0.2 * np.sin(3.0 * p[:, 0]) * p[:, 1], atol=1e-14)
    assert float(np.asarray(fem.solve(nonlinear=jno.solve.newton(**NEWTON))).min()) > 0.99  # and starts there: root +1


def test_wrong_sizes_and_complex_values_raise():
    d = _square()
    u, _, _, fem = _two_roots(d)
    u.initialize(np.ones(7))
    with pytest.raises(ValueError, match="7 values"):
        fem.solve(nonlinear=jno.solve.newton(**NEWTON))
    u.initialize(1.0 + 1.0j)
    with pytest.raises(ValueError, match="real field"):
        fem.solve(nonlinear=jno.solve.newton(**NEWTON))


def test_a_constant_unknown_and_a_complex_field_start_where_told():
    d = _square(0.25)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    A, W = d.fem_symbols(names=("A", "W"), complex=True)
    I = d.unknown.scalar(constant=True, name="I")  # noqa: E741
    Ai, Wi = A.bind(x=x, y=y), W.bind(x=x, y=y)
    fem = jno.fem(
        [Ai.d(x) * Wi.d(x) + Ai.d(y) * Wi.d(y) + 1j * Ai * Wi - I * Wi, A(xb, yb) - 0.0, (abs(Ai) ** 2).integrate() - 1e-4]
    )
    a0 = (0.01 + 0.02j) * np.ones(len(np.asarray(fem.field_points[0])))
    A.initialize(a0)
    I.initialize(0.7)
    x0 = np.asarray(fem._initialized_x0())
    start = fem.split(x0)
    np.testing.assert_allclose(start["A"], a0)
    assert float(start["I"]) == 0.7
    assert start.A is start["A"] and start.I is start["I"]  # fields read as attributes too
    assert {"A", "I"} <= set(dir(start))
    with pytest.raises(AttributeError, match="no field 'B'.*'A', 'I'"):
        start.B
    sol = fem.split(fem.solve(nonlinear=jno.solve.newton(**NEWTON)))
    assert abs(float(sol["I"])) > 0  # it solved from there


def _heat(d, with_term):
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    terms = [ui.t * vi + ui.d(x) * vi.d(x) + ui.d(y) * vi.d(y), u(xb, yb) - 0.0]
    if with_term:
        xi, yi = d.variable("initial", split=True)[:2]
        terms.append(u(xi, yi) - jno.np.sin(jnp.pi * xi) * jno.np.sin(jnp.pi * yi))
    else:
        u.initialize(jno.np.sin(jnp.pi * x) * jno.np.sin(jnp.pi * y))
    return u, x, y, terms


def test_the_initial_state_of_a_transient_problem_is_the_initial_condition_term():
    ref = jno.fem(_heat(_square(0.1, time=(0.0, 0.02, 6)), True)[3])
    u, _, _, terms = _heat(_square(0.1, time=(0.0, 0.02, 6)), False)
    fem = jno.fem(terms)
    assert fem.is_transient
    s_ref = np.asarray(ref.state0)
    assert np.ptp(s_ref) > 0.9  # the sine mode, not zero
    np.testing.assert_allclose(np.asarray(fem.state0), s_ref, rtol=0, atol=1e-14)
    # nodal values give the same state
    p = np.asarray(fem.points)
    u2, _, _, terms2 = _heat(_square(0.1, time=(0.0, 0.02, 6)), False)
    u2.initialize(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    np.testing.assert_allclose(np.asarray(jno.fem(terms2).state0), s_ref, rtol=0, atol=1e-14)


def test_a_transient_initial_state_given_twice_or_changed_after_the_build_raises():
    d = _square(0.2, time=(0.0, 0.02, 3))
    u, x, y, terms = _heat(d, True)
    u.initialize(0.0)
    with pytest.raises(ValueError, match="given twice"):
        jno.fem(terms)
    d = _square(0.2, time=(0.0, 0.02, 3))
    u, x, y, terms = _heat(d, False)
    fem = jno.fem(terms)
    u.initialize(1.0)
    with pytest.raises(ValueError, match="changed after jno.fem was built"):
        fem.solve()


@pytest.mark.parametrize("start, root", [(0.5, 1.0), (-0.5, -1.0)])
def test_an_unknown_from_d_unknown_starts_where_told(start, root):
    """``u = d.unknown()`` is a valued parameter for jno.fdm; its .initialize is the field's start in jno.fem,
    not a network's weight initializer."""
    d = _square()
    x, y, _ = d.variable("interior", split=True)
    u = d.unknown.scalar("u")
    v = u.test()
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([1e-3 * (ui.d(x) * vi.d(x) + ui.d(y) * vi.d(y)) + (ui**2 - 1.0) * vi])
    assert u.initialize(start) is u
    np.testing.assert_allclose(np.asarray(fem.solve(nonlinear=jno.solve.newton(**NEWTON))), root, atol=1e-10)


def test_initialize_refuses_weights():
    d = _square(0.25)
    u = d.unknown.scalar("u")
    with pytest.raises(TypeError, match="weight initializers are for networks"):
        u.initialize(lambda key, shape, dtype: 0.0)


def _burgers(nu=0.02):
    """Steady viscous Burgers u u_x = nu lap u, u = +-1 at the walls: Newton needs a start with the layer's sign
    change (from zero it has no wind)."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    (xl, yl, _), (xr, yr, _) = d.variable("left", split=True), d.variable("right", split=True)
    terms = [ui * ui.d(x, scheme=jno.fd(upwind=ui, order=2)) - nu * (ui.xx + ui.yy), u(xl, yl) - 1.0, u(xr, yr) + 1.0]
    for wall in ("bottom", "top"):
        xw, yw, _ = d.variable(wall, split=True)
        terms.append(u.bind(x=xw, y=yw).d(d.variable(wall, normals=True)) - 0.0)
    return d, u, x, terms


def test_fdm_starts_where_initialize_says():
    d, u, x, terms = _burgers()
    X = np.asarray(d.mesh_connectivity["points"])[:, 0]
    newton = jno.solve.newton(direct=True, line_search=True)
    ref = np.asarray(jno.fdm(terms).solve(x0=1.0 - 2.0 * X, nonlinear=newton, linear=jno.solve.lu())).reshape(-1)
    for start in (1.0 - 2.0 * x, 1.0 - 2.0 * X):  # a formula, its nodal values
        u.initialize(start)
        fd = jno.fdm(terms)
        np.testing.assert_allclose(np.asarray(fd._initialized_x0()), 1.0 - 2.0 * X, atol=1e-14)
        sol = np.asarray(fd.solve(nonlinear=newton, linear=jno.solve.lu())).reshape(-1)
        np.testing.assert_allclose(sol, ref, atol=1e-12)
    u.initialize(np.ones(5))
    with pytest.raises(ValueError, match="5 values"):
        jno.fdm(terms).solve(nonlinear=newton, linear=jno.solve.lu())


def _fdm_heat(with_term):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).structured().domain(time=(0.0, 0.02, 5))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    terms = [ui.t - (ui.xx + ui.yy), u(xb, yb) - 0.0]
    if with_term:
        xi, yi, _ = d.variable("initial", split=True)
        terms.append(u(xi, yi) - jno.np.sin(jnp.pi * xi) * jno.np.sin(jnp.pi * yi))
    else:
        u.initialize(jno.np.sin(jnp.pi * x) * jno.np.sin(jnp.pi * y))
    return u, terms


def test_fdm_initial_state_is_the_initial_condition_term():
    ref = np.asarray(jno.fdm(_fdm_heat(True)[1]).solve())
    u, terms = _fdm_heat(False)
    fd = jno.fdm(terms)
    assert fd._transient
    traj = np.asarray(fd.solve())
    assert np.ptp(traj[0]) > 0.9
    np.testing.assert_allclose(traj, ref, rtol=0, atol=1e-12)
    u.initialize(0.0)
    with pytest.raises(ValueError, match="changed after jno.fdm was built"):
        fd.solve()
    u2, terms2 = _fdm_heat(True)
    u2.initialize(0.0)
    with pytest.raises(ValueError, match="given twice"):
        jno.fdm(terms2)
