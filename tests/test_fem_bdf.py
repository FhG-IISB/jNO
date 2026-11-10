"""``jno.solve.bdf()``: variable-step, variable-order BDF / NDF (orders 1-5) with error control.

Oracle: a fine fixed-step SDIRK3 march (1600 steps) on the same mesh, whose grid contains every save time,
so what is measured is the time error of the scheme under test (Shampine & Reichelt, SIAM J. Sci. Comput.
18 (1997) for the method; Hairer & Wanner II, Sec. III.5 for variable-coefficient BDF).

Run with x64 (the FEM assembly is float64).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
T1 = 0.05
N_SAVE = 81
SAVE = jnp.linspace(0.0, T1, N_SAVE)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _form(nsteps, *, cubic=False, param=False, mesh_size=0.25):
    """u_t = k lap u (- u^3 when ``cubic``) on the unit square, zero walls, two decay modes in u0."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, T1, nsteps))
    u, phi = d.fem_symbols()
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    k = jno.np.parameter((1,), name="k") if param else 1.0
    s = jno.np.sin
    u0 = 3.0 * (s(PI * ci[0]) * s(PI * ci[1]) + s(2 * PI * ci[0]) * s(3 * PI * ci[1]))
    weak = ui.t * vi + k * (ui.x * vi.x + ui.y * vi.y)
    if cubic:
        weak = weak + ui**3 * vi
    return jno.fem([weak, u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])


def _run(scheme, fem, save=SAVE, args=None):
    return scheme.integrate(fem.operator, args or {}, save, linear_solve=None, nonlinear_solve=None)


def _reference(cubic):
    fem = _form(1601, cubic=cubic)
    full = np.asarray(_run(jno.solve.sdirk(3), fem, jnp.linspace(0.0, T1, 1601)))
    return full[::20]


@pytest.fixture(scope="module")
def ref_linear():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        return _reference(False)
    finally:
        jax.config.update("jax_enable_x64", prev)


def _worst(out, ref):
    return max(np.max(np.abs(out[i] - ref[i])) / np.max(np.abs(ref[i])) for i in range(1, N_SAVE))


def test_bdf_error_follows_the_tolerance(ref_linear):
    fem = _form(6)
    errs = []
    for rtol in (1e-4, 1e-6):
        out = np.asarray(_run(jno.solve.bdf().adaptive(rtol=rtol, atol=1e-3 * rtol), fem))
        assert not np.any(np.isnan(out))
        errs.append(_worst(out, ref_linear))
        assert errs[-1] < 20 * rtol, f"rtol={rtol}: error {errs[-1]:.2e}"
    assert errs[1] < errs[0] / 10, f"tightening the tolerance 100x gained only {errs[0] / errs[1]:.1f}x"


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("ndf", [True, False], ids=["ndf", "bdf"])
def test_every_fixed_order_reaches_the_tolerance(ref_linear, order, ndf):
    out = np.asarray(_run(jno.solve.bdf(order=order, ndf=ndf).adaptive(rtol=1e-4, atol=1e-7), _form(6)))
    assert not np.any(np.isnan(out))
    assert _worst(out, ref_linear) < 50 * 1e-4


def test_higher_order_takes_fewer_steps():
    """At rtol = 1e-6 the variable-order march reaches t1 within 100 attempts; held at order 2 it does not
    (it needs a few hundred), and backward Euler would need thousands. The budget is a step count."""
    fem = _form(6)
    tight = dict(rtol=1e-6, atol=1e-9, max_steps=100)
    assert not np.any(np.isnan(np.asarray(_run(jno.solve.bdf().adaptive(**tight), fem, SAVE[-1:]))))
    assert np.all(np.isnan(np.asarray(_run(jno.solve.bdf(order=2).adaptive(**tight), fem, SAVE[-1:]))))


def test_fixed_grid_orders_one_and_two_are_backward_euler_and_bdf2():
    fem = _form(21)
    grid = jnp.linspace(0.0, T1, 21)
    assert np.array_equal(
        np.asarray(_run(jno.solve.bdf(order=2), fem, grid)), np.asarray(_run(jno.solve.bdf2(), fem, grid))
    )
    assert np.array_equal(
        np.asarray(_run(jno.solve.bdf(order=1), fem, grid)), np.asarray(_run(jno.solve.theta(1.0), fem, grid))
    )


def test_bdf2_adaptive_is_the_bdf_march_at_order_two(ref_linear):
    out = np.asarray(_run(jno.solve.bdf2().adaptive(rtol=1e-4, atol=1e-7), _form(6)))
    assert _worst(out, ref_linear) < 50 * 1e-4


def test_bdf_marches_a_nonlinear_block():
    """u_t = lap u - u^3: every BDF step is a Newton solve (the backward-Euler step from the shifted state)."""
    ref = _reference(True)
    out = np.asarray(_run(jno.solve.bdf().adaptive(rtol=1e-5, atol=1e-8), _form(6, cubic=True)))
    assert not np.any(np.isnan(out))
    assert _worst(out, ref) < 2e-4


def test_bdf_is_differentiable_through_the_march():
    fem = _form(6, param=True, mesh_size=0.3)
    (name,) = fem.operator.runtime_parameter_exprs
    sch = jno.solve.bdf().adaptive(rtol=1e-5, atol=1e-8)

    def loss(kv):
        return jnp.sum(_run(sch, fem, SAVE[::10], {name: jnp.asarray([kv])}) ** 2)

    g = float(jax.grad(loss)(0.8))
    fd = float((loss(0.8 + 1e-5) - loss(0.8 - 1e-5)) / 2e-5)
    assert np.isfinite(g)
    assert abs(g - fd) <= 1e-2 * max(abs(fd), 1.0), f"AD {g} vs FD {fd}"


def test_bdf_rejections_keep_the_gradient_finite():
    """An oversized first step is rejected; its discarded state must give a zero, not NaN, cotangent."""
    fem = _form(6, param=True, mesh_size=0.3)
    (name,) = fem.operator.runtime_parameter_exprs
    sch = jno.solve.bdf().adaptive(rtol=1e-5, atol=1e-8, dt0=0.02)
    g = jax.grad(lambda kv: jnp.sum(_run(sch, fem, SAVE[::10], {name: jnp.asarray([kv])}) ** 2))(0.8)
    assert np.isfinite(float(g))


@pytest.mark.parametrize("scheme", [jno.solve.bdf(), jno.solve.bdf(order=2), jno.solve.bdf(order=1).adaptive()])
def test_bdf_refuses_a_second_order_in_time_block(scheme):
    """A ``u_tt`` block is assembled undamped (theta = 1/2); a BDF march would damp the wave silently."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4, time=(0.0, T1, 6))
    u, phi = d.fem_symbols()
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem([ui.tt * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
    with pytest.raises(NotImplementedError, match="second-order-in-time"):
        _run(scheme, fem)


def test_bdf_composes_through_fem_solve_complex_and_periodic():
    """The step is the block's own backward-Euler step, so the complex (2n real block) and periodic (reduced)
    paths and a state-dependent mass with a time-varying wall march through ``fem.solve`` unchanged. The
    oracle is a tight SDIRK3 march of the same form."""
    from tests.test_fem_adaptive_timestep import _complex_heat

    def ev(x):
        return np.asarray(x.fn() if hasattr(x, "fn") else jnp.asarray(x))

    save = jnp.linspace(0.0, T1, 6)
    ref = jno.solve.sdirk(3).adaptive(rtol=1e-7, atol=1e-10)
    sch = jno.solve.bdf().adaptive(rtol=1e-4, atol=1e-7)

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, T1, 6))
    u, phi = d.fem_symbols()
    x, y, t = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    mass_wall = jno.fem(
        [(1 + ui**2) * ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - jno.np.exp(-2 * tb), u(ci[0], ci[1]) - 1.0]
    )

    for label, fem, cplx in [
        ("c(u) u_t + g(t)", mass_wall, False),
        ("complex", _complex_heat(mesh_size=0.3)[1], True),
        ("complex periodic", _complex_heat(mesh_size=0.3, periodic=True)[1], True),
    ]:
        got, want = ev(fem.solve(save_ts=save, time=sch)), ev(fem.solve(save_ts=save, time=ref))
        assert np.iscomplexobj(got) == cplx and not np.any(np.isnan(got)), label
        rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
        assert rel < 2e-3, f"{label}: {rel:.2e}"


def test_bdf_refuses_a_bad_order():
    for bad in (0, 6, 2.5, True):
        with pytest.raises(ValueError, match="order must be"):
            jno.solve.bdf(order=bad)
