"""Arbitrary jno.np ops inside a FEM weak form.

A jno.np op of the *unknown* (exp(u), sqrt(u), 1/u, u**2.5, ...) makes the weak form nonlinear, so
jno.fem routes it to the residual operator and the Jacobian comes from JAX autodiff through the op
-- so the op only has to be JAX-differentiable. A jno.np op of the *coordinates* is just a linear
coefficient field. These tests pin down that breadth (write Helmholtz / reaction / Schrodinger-style
nonlinearities natively, no special API).

Run with x64 (the feax assembly is float64): ``JAX_ENABLE_X64=1``.
"""

import pytest

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for the box domain")
pytest.importorskip("scipy", reason="scipy.optimize for the Newton solve")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.optimize as spo  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# op(u) as a NONLINEAR reaction on the trial (all must be detected nonlinear + Jacobian-differentiable)
NONLINEAR_OPS = {
    "exp": lambda z: jno.np.exp(z),
    "sqrt": lambda z: jno.np.sqrt(z),
    "log": lambda z: jno.np.log(z),
    "tanh": lambda z: jno.np.tanh(z),
    "pow_2_5": lambda z: z**2.5,
    "reciprocal": lambda z: 1.0 / z,
    "cube": lambda z: z**3,
}


@pytest.mark.parametrize("name", list(NONLINEAR_OPS))
def test_nonlinear_trial_op_recovers_manufactured(name):
    """-lap u + op(u) = f with a positive manufactured u* = 1 + 0.3 sin(pi x) sin(pi y) (so
    sqrt/log/1/u are well-defined). The form routes to the residual operator; Newton (residual +
    autodiff Jacobian through op) recovers u*."""
    op = NONLINEAR_OPS[name]
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.12)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    g = jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    u_star = 1.0 + 0.3 * g
    f = 0.3 * 2 * PI**2 * g + op(u_star)  # -lap u* + op(u*)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + op(u) * vi - f * vi, u(xb, yb) - 1.0], quad_degree=3)
    assert not fem.is_linear, f"{name}: op(u) must classify as nonlinear"

    pts = np.asarray(fem.points)
    sol = spo.root(
        lambda v: np.asarray(fem.residual(jnp.asarray(v))),
        np.ones(fem.dofs),
        jac=lambda v: np.asarray(dense(fem.jacobian(jnp.asarray(v)))),
        method="hybr",
    )
    u_ex = 1.0 + 0.3 * np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    rel = float(np.linalg.norm(sol.x - u_ex) / np.linalg.norm(u_ex))
    assert sol.success and rel < 5e-3, f"{name}: converged={sol.success} rel_L2={rel:.2e}"


def test_coordinate_ops_are_linear_coefficients():
    """jno.np ops of the COORDINATES are ordinary (linear) coefficient fields, evaluated at the
    quadrature points: exp/sqrt/log/tanh of position assemble a linear system fine."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    kx = jno.np.exp(0.3 * xi) + jno.np.sqrt(1.0 + yi) + jno.np.tanh(xi - yi) + jno.np.log(2.0 + xi)  # > 0
    fem = jno.fem([kx * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem.is_linear
    u_h = np.linalg.solve(np.asarray(dense(fem.A)), np.asarray(fem.b).reshape(-1))
    assert np.all(np.isfinite(u_h)) and np.linalg.norm(u_h) > 0.0


def test_navier_stokes_convective_term_is_nonlinear_and_recovers_manufactured():
    """The Navier-Stokes convective term ``inner(grad u, u)`` is the unknown contracted with itself,
    so it must classify as NONLINEAR -- unlike the bilinear ``inner(grad u, grad v)`` (trial x test),
    which stays linear. Manufactured steady NS with the Taylor-Green field
    ``u* = (cos x sin y, -sin x cos y)``, ``p* = -1/4(cos 2x + cos 2y)``: there ``(u*.grad)u* +
    grad p* = 0`` and ``lap u* = -2 u*``, so the forcing is ``f = 2 nu u*``. Newton (residual +
    autodiff Jacobian through the convection) recovers ``u*`` -- which it could not if the convective
    term were dropped/linearised."""
    inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    nu = 0.1
    ux = lambda x, y: jno.np.cos(x) * jno.np.sin(y)  # noqa: E731
    uy = lambda x, y: -jno.np.sin(x) * jno.np.cos(y)  # noqa: E731
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.13)
    d.point_region("ppin", (0.0, 0.0))  # p*(0, 0) = -1/2
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xpn, ypn, _ = d.variable("ppin", split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    conv = inner(gu, ub, n_contract=1)  # (u.grad)u
    fx, fy = 2 * nu * ux(xi, yi), 2 * nu * uy(xi, yi)
    momentum = inner(conv, vb, n_contract=1) + nu * inner(gu, gv, n_contract=2) - pp * trace(gv) - fx * vb[0] - fy * vb[1]
    fem = jno.fem([momentum, -qq * trace(gu), u(xb, yb)[0] - ux(xb, yb), u(xb, yb)[1] - uy(xb, yb), p(xpn, ypn) - (-0.5)])
    assert not fem.is_linear, "convective term inner(grad u, u) must classify as nonlinear"

    off = fem.problem.offset
    pts = np.asarray(fem.problem.mesh[0].points)
    sol = spo.root(
        lambda w: np.asarray(fem.residual(jnp.asarray(w))),
        np.zeros(fem.dofs),
        jac=lambda w: np.asarray(dense(fem.jacobian(jnp.asarray(w)))),
        method="hybr",
    )
    uu = sol.x[off[0] : off[1]].reshape(-1, 2)
    ref = np.stack([np.cos(pts[:, 0]) * np.sin(pts[:, 1]), -np.sin(pts[:, 0]) * np.cos(pts[:, 1])], 1)
    rel = float(np.linalg.norm(uu - ref) / np.linalg.norm(ref))
    assert sol.success and rel < 5e-3, f"manufactured NS recovery: converged={sol.success} rel_L2={rel:.2e}"
