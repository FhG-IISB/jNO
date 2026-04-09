import jax
import jax.numpy as jnp
import pytest

from jno import fn
from jno.trace_evaluator import TraceEvaluator
from tests.conftest import make_var, MockDomain


def _eval(expr, context):
    ev = TraceEvaluator({})
    return ev.evaluate(expr, context, {}, key=jax.random.PRNGKey(0))


def _make_2d_vars():
    d = MockDomain(tags=["xy"], dim=2)
    from jno.trace import Variable

    x = Variable("xy", [0, 1], domain=d)
    y = Variable("xy", [1, 2], domain=d)
    return x, y


def _make_3d_vars():
    d = MockDomain(tags=["xyz"], dim=3)
    from jno.trace import Variable

    x = Variable("xyz", [0, 1], domain=d)
    y = Variable("xyz", [1, 2], domain=d)
    z = Variable("xyz", [2, 3], domain=d)
    return x, y, z


def test_poisson_residual_1d():
    x = make_var("x")
    u = x**3
    pts = jnp.linspace(0.1, 0.9, 11).reshape(-1, 1)

    residual = fn.poisson(u, 0.0, x)
    got = jnp.ravel(_eval(residual, {"x": pts}))
    expected = -6.0 * pts[:, 0]

    assert jnp.allclose(got, expected, atol=1e-4)


def test_heat_residual_1d():
    x = make_var("x")
    t = make_var("t")

    u = t * (x**2)
    residual = fn.heat(u, t, x, diffusivity=0.5)

    pts_x = jnp.linspace(0.1, 0.9, 9).reshape(-1, 1)
    pts_t = jnp.linspace(0.2, 1.0, 9).reshape(-1, 1)
    got = jnp.ravel(_eval(residual, {"x": pts_x, "t": pts_t}))
    expected = pts_x[:, 0] ** 2 - pts_t[:, 0]

    assert jnp.allclose(got, expected, atol=1e-4)


def test_wave_residual_1d():
    x = make_var("x")
    t = make_var("t")

    u = (t**2) * x
    residual = fn.wave(u, t, x, speed=3.0)

    pts_x = jnp.linspace(0.1, 0.9, 8).reshape(-1, 1)
    pts_t = jnp.linspace(0.2, 1.0, 8).reshape(-1, 1)
    got = jnp.ravel(_eval(residual, {"x": pts_x, "t": pts_t}))
    expected = 2.0 * pts_x[:, 0]

    assert jnp.allclose(got, expected, atol=1e-4)


def test_burgers_1d_residual():
    x = make_var("x")
    t = make_var("t")

    u = x
    residual = fn.burgers_1d(u, x, t, viscosity=0.1)

    pts_x = jnp.linspace(0.1, 0.9, 10).reshape(-1, 1)
    pts_t = jnp.linspace(0.2, 1.1, 10).reshape(-1, 1)
    got = jnp.ravel(_eval(residual, {"x": pts_x, "t": pts_t}))
    expected = pts_x[:, 0]

    assert jnp.allclose(got, expected, atol=1e-4)


def test_burgers_inviscid_1d_residual():
    x = make_var("x")
    t = make_var("t")

    u = x + t
    residual = fn.burgers_inviscid_1d(u, x, t)

    pts_x = jnp.linspace(0.1, 0.9, 10).reshape(-1, 1)
    pts_t = jnp.linspace(0.2, 0.8, 10).reshape(-1, 1)
    got = jnp.ravel(_eval(residual, {"x": pts_x, "t": pts_t}))
    expected = 1.0 + pts_x[:, 0] + pts_t[:, 0]
    assert jnp.allclose(got, expected, atol=1e-4)


def test_biharmonic_residual_1d():
    x = make_var("x")
    u = x**4

    residual = fn.biharmonic(u, 0.0, x)
    pts = jnp.linspace(0.1, 0.9, 9).reshape(-1, 1)
    got = jnp.ravel(_eval(residual, {"x": pts}))
    expected = jnp.full_like(got, 24.0)
    assert jnp.allclose(got, expected, atol=1e-4)


def test_adr_and_continuity_2d():
    x, y = _make_2d_vars()

    u = x**2 + y
    residual = fn.advection_diffusion_reaction(
        u,
        [2.0, -1.0],
        x,
        y,
        diffusivity=0.5,
        reaction_coeff=3.0,
    )

    pts = jnp.stack(
        [jnp.linspace(0.1, 0.9, 7), jnp.linspace(0.2, 1.0, 7)],
        axis=-1,
    )
    got = jnp.ravel(_eval(residual, {"xy": pts}))
    expected = 3.0 * pts[:, 0] ** 2 + 4.0 * pts[:, 0] + 3.0 * pts[:, 1] - 2.0
    assert jnp.allclose(got, expected, atol=1e-4)

    vx = x**2
    vy = y**3
    c = fn.continuity([vx, vy], x, y)
    got_c = jnp.ravel(_eval(c, {"xy": pts}))
    expected_c = 2.0 * pts[:, 0] + 3.0 * pts[:, 1] ** 2
    assert jnp.allclose(got_c, expected_c, atol=1e-4)


def test_advection_and_diffusion_residuals():
    x = make_var("x")
    t = make_var("t")

    u = x + t
    adv = fn.advection(u, [2.0], x, t=t)
    pts_x = jnp.linspace(0.1, 0.9, 6).reshape(-1, 1)
    pts_t = jnp.linspace(0.2, 0.7, 6).reshape(-1, 1)
    got_adv = jnp.ravel(_eval(adv, {"x": pts_x, "t": pts_t}))
    assert jnp.allclose(got_adv, jnp.full_like(got_adv, 3.0), atol=1e-4)

    u2 = t * (x**2)
    diff = fn.diffusion(u2, x, t=t, diffusivity=0.5)
    got_diff = jnp.ravel(_eval(diff, {"x": pts_x, "t": pts_t}))
    expected_diff = pts_x[:, 0] ** 2 - pts_t[:, 0]
    assert jnp.allclose(got_diff, expected_diff, atol=1e-4)


def test_fisher_kpp_residual_1d():
    x = make_var("x")
    t = make_var("t")

    u = x + t
    residual = fn.fisher_kpp(u, t, x, diffusivity=0.5, growth_rate=2.0)

    pts_x = jnp.linspace(0.1, 0.8, 7).reshape(-1, 1)
    pts_t = jnp.linspace(0.05, 0.35, 7).reshape(-1, 1)
    got = jnp.ravel(_eval(residual, {"x": pts_x, "t": pts_t}))
    s = pts_x[:, 0] + pts_t[:, 0]
    expected = 1.0 - 2.0 * s * (1.0 - s)
    assert jnp.allclose(got, expected, atol=1e-4)


def test_cahn_hilliard_residual_1d():
    x = make_var("x")
    t = make_var("t")

    u = x + t
    residual = fn.cahn_hilliard(u, t, x, epsilon=0.2, mobility=0.5)

    pts_x = jnp.linspace(0.1, 0.9, 8).reshape(-1, 1)
    pts_t = jnp.linspace(0.2, 0.8, 8).reshape(-1, 1)
    got = jnp.ravel(_eval(residual, {"x": pts_x, "t": pts_t}))
    expected = 1.0 - 3.0 * (pts_x[:, 0] + pts_t[:, 0])
    assert jnp.allclose(got, expected, atol=1e-4)


def test_navier_stokes_incompressible_2d_residuals():
    x, y = _make_2d_vars()
    p = x * 0.0
    u = x
    v = -y

    mom_x, mom_y, mass = fn.navier_stokes_incompressible_2d(
        u,
        v,
        p,
        x,
        y,
        viscosity=0.1,
    )

    pts = jnp.stack(
        [jnp.linspace(0.1, 0.9, 6), jnp.linspace(0.2, 1.0, 6)],
        axis=-1,
    )
    got_mx = jnp.ravel(_eval(mom_x, {"xy": pts}))
    got_my = jnp.ravel(_eval(mom_y, {"xy": pts}))
    got_mass = jnp.ravel(_eval(mass, {"xy": pts}))

    assert jnp.allclose(got_mx, pts[:, 0], atol=1e-4)
    assert jnp.allclose(got_my, pts[:, 1], atol=1e-4)
    assert jnp.allclose(got_mass, jnp.zeros_like(got_mass), atol=1e-4)


def test_maxwell_3d_residuals():
    x, y, z = _make_3d_vars()
    t = make_var("t")

    ex = x
    ey = y
    ez = z
    bx = x * 0.0
    by = y * 0.0
    bz = z * 0.0

    (
        faraday_x,
        faraday_y,
        faraday_z,
        ampere_x,
        ampere_y,
        ampere_z,
        gauss_e,
        gauss_b,
    ) = fn.maxwell_3d(ex, ey, ez, bx, by, bz, x, y, z, t)

    pts_xyz = jnp.stack(
        [
            jnp.linspace(0.1, 0.6, 6),
            jnp.linspace(0.2, 0.7, 6),
            jnp.linspace(0.3, 0.8, 6),
        ],
        axis=-1,
    )
    pts_t = jnp.linspace(0.0, 0.5, 6).reshape(-1, 1)
    ctx = {"xyz": pts_xyz, "t": pts_t}

    assert jnp.allclose(jnp.ravel(_eval(faraday_x, ctx)), jnp.zeros(6), atol=1e-4)
    assert jnp.allclose(jnp.ravel(_eval(faraday_y, ctx)), jnp.zeros(6), atol=1e-4)
    assert jnp.allclose(jnp.ravel(_eval(faraday_z, ctx)), jnp.zeros(6), atol=1e-4)
    assert jnp.allclose(jnp.ravel(_eval(ampere_x, ctx)), jnp.zeros(6), atol=1e-4)
    assert jnp.allclose(jnp.ravel(_eval(ampere_y, ctx)), jnp.zeros(6), atol=1e-4)
    assert jnp.allclose(jnp.ravel(_eval(ampere_z, ctx)), jnp.zeros(6), atol=1e-4)
    assert jnp.allclose(jnp.ravel(_eval(gauss_e, ctx)), jnp.full((6,), 3.0), atol=1e-4)
    assert jnp.allclose(jnp.ravel(_eval(gauss_b, ctx)), jnp.zeros(6), atol=1e-4)


def test_adr_requires_matching_velocity_and_variables():
    x = make_var("x")
    with pytest.raises(ValueError):
        fn.advection_diffusion_reaction(x, [1.0, 2.0], x)


def test_continuity_requires_matching_lengths():
    x = make_var("x")
    y = make_var("y")
    with pytest.raises(ValueError):
        fn.continuity([x], x, y)


def test_advection_requires_at_least_one_spatial_variable():
    x = make_var("x")
    with pytest.raises(ValueError):
        fn.advection(x, [1.0])
