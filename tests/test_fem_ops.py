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
