"""``jno.fd(...)``: a finite-difference stencil described by its mathematics, passed as ``scheme=``.

Oracles: Fornberg weights against the textbook stencils; the observed order of a Poisson solve and of a
network's derivative against its exact (AD) derivative; polynomial exactness; the refusals."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
import jno.jnp_ops as jnn
from jno.differential_operators import DifferentialOperators as D
from jno.stencils import FDStencil, fornberg


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize(
    "offsets, deriv, expected",
    [
        ((-1, 0, 1), 1, [-0.5, 0.0, 0.5]),
        ((-1, 0, 1), 2, [1.0, -2.0, 1.0]),
        ((-2, -1, 0, 1, 2), 1, [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12]),
        ((-2, -1, 0, 1, 2), 2, [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
        ((0, 1, 2), 1, [-1.5, 2.0, -0.5]),
    ],
)
def test_fornberg_reproduces_the_textbook_stencils(offsets, deriv, expected):
    np.testing.assert_allclose(fornberg(offsets, deriv), expected, atol=1e-14)


def test_a_spec_is_a_scheme_string():
    s = jno.fd(order=4)
    assert isinstance(s, str) and isinstance(s, FDStencil) and s == "finite_difference:order=4"


def _poisson_error(h, *, scheme, via_bind):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    if via_bind:
        ui = u.bind(x=x, y=y, scheme=scheme)
        lap = ui.xx + ui.yy
    else:
        ui = u.bind(x=x, y=y)
        lap = ui.d2(x, scheme=scheme) + ui.d2(y, scheme=scheme)
    f = -2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    sol = np.asarray(jno.fdm([lap - f, u(xb, yb) - 0.0]).solve()).reshape(-1)
    P = np.asarray(d.mesh_connectivity["points"])[:, :2]
    return float(np.abs(sol - np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])).max())


@pytest.mark.parametrize("via_bind", [False, True])
@pytest.mark.parametrize("order, rate", [(4, 3.7), (6, 5.5)])
def test_higher_order_poisson(order, rate, via_bind):
    """Measured: order 4 → 5.7e-5, 6.3e-6, 4.2e-7 (rate 3.9); order 6 → 5.5e-6, 3.6e-8, 4.4e-10."""
    e = [_poisson_error(h, scheme=jno.fd(order=order), via_bind=via_bind) for h in (0.1, 0.05, 0.025)]
    assert np.log2(e[1] / e[2]) > rate, e


def test_order_4_on_a_network_field():
    """A PINN loss on a field: a network evaluated on the grid, differentiated with jno.fd(order=4),
    converges at fourth order to the network's exact derivative (the default stencil: first order in the
    max norm, from its first-order edges)."""
    import foundax

    net = jno.nn.wrap(foundax.mlp(2, output_dim=1, hidden_dims=16, num_layers=2, key=jax.random.PRNGKey(0)))
    errs = []
    for n in (16, 32, 64):
        d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / n).structured().domain()
        x, y, _ = d.variable("interior", split=True)
        u = net(x, y)
        crux = jno.core([(u - 0.0).mse])
        ad, fd4 = (np.asarray(v).reshape(-1) for v in crux.eval([u.d(x), u.d(x, scheme=jno.fd(order=4))]))
        errs.append(np.abs(fd4 - ad).max() / np.abs(ad).max())
    assert np.log2(errs[1] / errs[2]) > 3.7, errs


def test_points_and_weights_are_exact_on_polynomials():
    """points=(-1, 0, 1, 2) is exact on cubics; weights={-1: -0.5, 1: 0.5} is the central difference."""
    n, h = 11, 0.1
    X = np.linspace(0.0, 1.0, n)
    U = jnp.asarray(X**3)
    np.testing.assert_allclose(D._grid_stencil_diff(U, h, 0, 1, jno.fd(points=(-1, 0, 1, 2)), False), 3 * X**2, atol=1e-11)
    central = D._grid_stencil_diff(jnp.asarray(X**2), h, 0, 1, jno.fd(weights={-1: -0.5, 1: 0.5}), False)
    np.testing.assert_allclose(central[1:-1], 2 * X[1:-1], atol=1e-12)


def test_periodic_axis_wraps():
    """On a periodic axis the interior stencil wraps: d/dx sin(2πx) at 4th order."""
    errs = []
    for n in (16, 32):
        X = np.linspace(0.0, 1.0, n + 1)  # the last node repeats the first
        d = D._grid_stencil_diff(jnp.asarray(np.sin(2 * np.pi * X)), 1.0 / n, 0, 1, jno.fd(order=4), True)
        errs.append(np.abs(np.asarray(d) - 2 * np.pi * np.cos(2 * np.pi * X)).max())
    assert np.log2(errs[0] / errs[1]) > 3.8, errs


def test_refusals():
    with pytest.raises(ValueError, match="one of"):
        jno.fd(order=4, points=(-1, 0, 1))
    with pytest.raises(ValueError, match="even order"):
        jno.fd(order=3)
    with pytest.raises(ValueError, match="sum to zero"):
        jno.fd(weights={-1: 1.0, 1: 1.0})
    with pytest.raises(ValueError, match="at least"):
        fornberg((0, 1), 2)
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    with pytest.raises(NotImplementedError, match="structured grid"):
        jno.fdm([ui.d2(x, scheme=jno.fd(order=4)) + ui.yy, u(xb, yb) - 0.0]).solve()


# The conservative form of (κ·u.x).x. Chaining two central differences reads every second node — a
# 2h-wide stencil that decouples odd and even nodes: layered electrostatics was first order (0.14 off at
# h = 0.1, only the even nodes wrong). The compact flux difference with κ at the half-points is exact.


def _layered_potential(h, eps1=1.0, eps2=10.0):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    phi = d.unknown()
    ph = phi.bind(x=x, y=y)
    eps = jnn.where(x < 0.5, eps1, eps2)
    (xl, yl, _), (xr, yr, _) = d.variable("left", split=True), d.variable("right", split=True)
    terms = [(eps * ph.x).x + (eps * ph.y).y, phi(xl, yl) - 0.0, phi(xr, yr) - 1.0]
    for wall in ("bottom", "top"):
        xw, yw, _ = d.variable(wall, split=True)
        terms.append(phi.bind(x=xw, y=yw).d(d.variable(wall, normals=True)) - 0.0)
    sol = np.asarray(jno.fdm(terms).solve()).reshape(-1)
    X = np.asarray(d.mesh_connectivity["points"])[:, 0]
    q = 1.0 / (0.5 / eps1 + 0.5 / eps2)
    return float(np.abs(sol - np.where(X < 0.5, q * X / eps1, q * 0.5 / eps1 + q * (X - 0.5) / eps2)).max())


def test_layered_dielectric_is_exact():
    """ε = 1 | 10: the potential is piecewise linear, and the conservative stencil with ε evaluated at the
    half-points reproduces it to solver tolerance (it was 0.14 off at h = 0.1)."""
    assert _layered_potential(0.1) < 1e-9 and _layered_potential(0.05) < 1e-9


def _smooth_error(h, *, average=None, data_coefficient=False):
    import equinox as eqx

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    P = np.asarray(d.mesh_connectivity["points"])[:, :2]
    if data_coefficient:  # κ known only at the nodes
        k = jno.np.parameter((len(P),), name="kappa")
        k.model.module = eqx.tree_at(lambda m: m.value, k.model.module, jnp.asarray(1 + P[:, 0] ** 2))
    else:
        k = 1 + x**2
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    S = jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    f = -(2 * x * np.pi * jnn.cos(np.pi * x) * jnn.sin(np.pi * y) - 2 * (1 + x**2) * np.pi**2 * S)  # −∇·(κ∇u)
    sch = jno.fd(average=average) if average else "finite_difference"
    lhs = -((k * ui.x).d(x, scheme=sch) + (k * ui.y).d(y, scheme=sch))
    sol = np.asarray(jno.fdm([lhs - f, u(xb, yb) - 0.0]).solve()).reshape(-1)
    return float(np.abs(sol - np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])).max())


@pytest.mark.parametrize("average", [None, "arithmetic", "harmonic"])
@pytest.mark.parametrize("data_coefficient", [False, True])
def test_conservative_form_is_second_order(average, data_coefficient):
    """κ = 1 + x²: second order with every averaging, as a formula or as data known only at the nodes
    (which used to fail with "No model": a data field was only honoured in boundary values)."""
    e = [_smooth_error(h, average=average, data_coefficient=data_coefficient) for h in (0.1, 0.05)]
    assert e[1] < 2.5e-3 and e[0] / e[1] > 3.7, e


def test_exact_average_refuses_stored_coefficients():
    with pytest.raises(ValueError, match="half-points"):
        _smooth_error(0.1, average="exact", data_coefficient=True)
    with pytest.raises(ValueError, match="average"):
        jno.fd(average="geometric")


# Upwinding: jno.fd(upwind=b, order=k) reads the side the wind comes from, per node. The wind is an
# expression — a number, a formula, a field, the unknown itself.


@pytest.mark.parametrize("order, rate", [(1, 0.9), (2, 1.8), (3, 2.7)])
def test_upwind_order_on_a_smooth_field(order, rate):
    """d/dx sin(2πx) with the wind changing sign at x = ½: observed order k, both wind directions."""
    from jno.fdm import _unwrap
    from jno.trace_evaluator import TraceEvaluator

    errs = []
    for n in (32, 64):
        d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / n).structured().domain()
        x, y, _ = d.variable("interior", split=True)
        u = d.unknown()
        ub = u.bind(x=x, y=y)
        f = jno.fdm([ub.xx + ub.yy])
        X = np.asarray(d.mesh_connectivity["points"])[:, 0]
        ev = TraceEvaluator(params={**f._params_scope(), **f._inject(jnp.asarray(np.sin(2 * np.pi * X)))})
        du = ub.d(x, scheme=jno.fd(upwind=x - 0.5, order=order))
        got = np.asarray(ev.evaluate(_unwrap(du), context=f._eval_context({"interior"}), var_bindings={})).reshape(-1)
        errs.append(np.abs(got - 2 * np.pi * np.cos(2 * np.pi * X)).max())
    assert np.log2(errs[0] / errs[1]) > rate, errs


def _boundary_layer(h, scheme):
    b, D = 1.0, 0.01
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    (xl, yl, _), (xr, yr, _) = d.variable("left", split=True), d.variable("right", split=True)
    terms = [b * ui.d(x, scheme=scheme) - D * (ui.xx + ui.yy), u(xl, yl) - 0.0, u(xr, yr) - 1.0]
    for wall in ("bottom", "top"):
        xw, yw, _ = d.variable(wall, split=True)
        terms.append(u.bind(x=xw, y=yw).d(d.variable(wall, normals=True)) - 0.0)
    return np.asarray(jno.fdm(terms).solve(linear=jno.solve.lu())).reshape(-1)


def test_first_order_upwinding_is_monotone_where_central_oscillates():
    """−0.01 u'' + u' = 0, cell Péclet 5 at h = 0.05: central undershoots to −0.43; upwind stays in [0, 1]."""
    assert _boundary_layer(0.05, "finite_difference").min() < -0.3
    up = _boundary_layer(0.05, jno.fd(upwind=1.0, order=1))
    assert up.min() > -1e-10 and up.max() < 1.0 + 1e-10


def test_the_wind_can_be_the_unknown():
    """Steady viscous Burgers u·u_x = ν Δu, u = ±1 at the walls: the wind is u itself and changes sign at
    the layer. Matrix-free and assembled-tangent Newton agree (the tangent covers both upwind sides)."""
    nu = 0.02
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    (xl, yl, _), (xr, yr, _) = d.variable("left", split=True), d.variable("right", split=True)
    terms = [ui * ui.d(x, scheme=jno.fd(upwind=ui, order=2)) - nu * (ui.xx + ui.yy), u(xl, yl) - 1.0, u(xr, yr) + 1.0]
    for wall in ("bottom", "top"):
        xw, yw, _ = d.variable(wall, split=True)
        terms.append(u.bind(x=xw, y=yw).d(d.variable(wall, normals=True)) - 0.0)
    X = np.asarray(d.mesh_connectivity["points"])[:, 0]
    free = np.asarray(jno.fdm(terms).solve(x0=1.0 - 2.0 * X, nonlinear=jno.solve.newton(line_search=True))).reshape(-1)
    direct = np.asarray(
        jno.fdm(terms).solve(
            x0=1.0 - 2.0 * X, nonlinear=jno.solve.newton(direct=True, line_search=True), linear=jno.solve.lu()
        )
    ).reshape(-1)
    assert np.abs(free - direct).max() < 1e-8
    assert np.abs(free + np.tanh((X - 0.5) / (2 * nu))).max() < 0.15  # the layer, at h = 0.05 (0.118 measured)


def test_upwind_refuses_a_mesh_and_other_stencil_options():
    with pytest.raises(ValueError, match="upwind"):
        jno.fd(upwind=1.0, points=(-1, 0))
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    with pytest.raises(NotImplementedError, match="structured grid"):
        jno.fdm([ui.d(x, scheme=jno.fd(upwind=1.0)) - ui.xx - ui.yy, u(xb, yb) - 0.0]).solve()
