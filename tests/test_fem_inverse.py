"""Differentiable FEM forward solve for inverse problems, authored entirely
through ``jno.fem([...])`` (no ``init_fem`` / ``assemble``).

``FEM.solve`` hosts a *real* parametric solve in the trace so ``crux.solve``
recovers a ``jno.np.parameter`` from data. The solver is the user's own callable
(jNO writes none): the linear default is ``jnp.linalg.solve`` (a ``lineax`` backend
is exercised too); the nonlinear default is an ``optimistix`` Newton ``root_find``
(implicit-diff, so the gradient reaches the parameter without unrolling Newton).

Run with x64 (the feax assembly is float64): ``JAX_ENABLE_X64=1``.
"""

import pytest

pytest.importorskip("feax", reason="feax required for FEM inverse tests")
pytest.importorskip("shapely", reason="shapely required for the box domain")
pytest.importorskip("optimistix", reason="optimistix required for the nonlinear solve")
pytest.importorskip("lineax", reason="lineax required for the lineax-backend test")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import lineax  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import optimistix as optx  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
TOL = 0.05  # adam recovery of a scalar parameter from clean data

# The inverse loss has no spatial Variable (the FEM solve is global), so crux needs
# an explicit domain to drive its loop. See jno.core(domain=).
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


@pytest.fixture(autouse=True)
def _x64():
    """feax assembly is float64, so these tests need x64. Set it per-test with
    save/restore — the global flag is shared across modules and other suites flip it
    at import (e.g. the periodic-parametric tests force it False)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _alpha(start=2.0, lr=5e-2):
    a = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="alpha")
    a.initialize(jax.nn.initializers.constant(start))  # start far from truth = 1
    a.dtype(jnp.float64)
    a.optimizer(optax.adam(lr))
    return a


def _recover(u_node, alpha, u_obs, n=120):
    crux = jno.core([(u_node - u_obs).mse], domain=_DUMMY)
    crux.solve(n)
    return float(np.asarray(crux.eval([alpha])[0]).reshape(-1)[0])


def _linear_fem(alpha, mesh_size=0.2):
    """Parametric Poisson  -alpha * lap u = f,  exact u = x(1-x)y(1-y) at alpha=1."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    weak = alpha * (ui.x * vi.x + ui.y * vi.y) - f * vi
    return jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)


def _nonlinear_fem(alpha, mesh_size=0.2):
    """Parametric reaction-diffusion  -lap u + alpha * u^3 = f."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    ss = jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    f = 2.0 * PI**2 * ss + 1.0 * ss**3  # source for alpha_true = 1
    weak = ui.x * vi.x + ui.y * vi.y + alpha * (u * u * u) * phi - f * vi
    return jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)


def test_linear_recovers_default_solver():
    alpha = _alpha()
    fem = _linear_fem(alpha)
    assert fem.is_linear
    sys = fem.operator
    assert sys.is_parametric and list(sys.runtime_parameter_exprs) == ["alpha"]

    A1, b1 = sys.evaluate({"alpha": 1.0})
    u_obs = jnp.linalg.solve(jnp.asarray(A1), jnp.asarray(b1).reshape(-1))

    rec = _recover(fem.solve(), alpha, u_obs)
    assert abs(rec - 2.0) > 0.5, "parameter did not move -- gradient did not reach it"
    assert abs(rec - 1.0) < TOL, f"linear (default solver): recovered alpha={rec:.4f}"


def test_linear_recovers_with_lineax_backend():
    alpha = _alpha()
    fem = _linear_fem(alpha)
    A1, b1 = fem.operator.evaluate({"alpha": 1.0})
    u_obs = jnp.linalg.solve(jnp.asarray(A1), jnp.asarray(b1).reshape(-1))

    u_node = fem.solve(lambda A, b: lineax.linear_solve(lineax.MatrixLinearOperator(A), b).value)
    rec = _recover(u_node, alpha, u_obs)
    assert abs(rec - 1.0) < TOL, f"linear (lineax backend): recovered alpha={rec:.4f}"


def test_nonlinear_recovers_optimistix():
    alpha = _alpha()
    fem = _nonlinear_fem(alpha)
    assert not fem.is_linear
    op = fem.operator
    assert type(op).__name__ == "FemResidualOperator" and op.is_parametric

    u0 = jnp.zeros((int(op.size),), dtype=jnp.float64)
    u_obs = optx.root_find(
        lambda uu, _a: op.residual(uu, {"alpha": 1.0}),
        optx.Newton(rtol=1e-8, atol=1e-8),
        u0,
        args=None,
        max_steps=100,
    ).value

    rec = _recover(fem.solve(), alpha, u_obs)
    assert abs(rec - 2.0) > 0.5, "parameter did not move -- implicit-diff gradient did not reach it"
    assert abs(rec - 1.0) < TOL, f"nonlinear (optimistix): recovered alpha={rec:.4f}"


def test_nonaffine_scalar_recovers_via_reassembly():
    """A parameter inside a nonlinear function (``exp(logk)``) can't be factored into
    a constant basis, so the operator is re-assembled each call with the parameter
    threaded as feax InternalVars. The feax kernel is JAX, so the gradient still
    reaches the parameter."""
    logk = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="logk")
    logk.initialize(jax.nn.initializers.constant(0.7))  # k = e^0.7 ~ 2 (truth: logk=0)
    logk.dtype(jnp.float64)
    logk.optimizer(optax.adam(5e-2))

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))  # -k lap u = f at k=1
    weak = jno.np.exp(logk) * (ui.x * vi.x + ui.y * vi.y) - f * vi
    fem = jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)

    assert fem.is_linear
    assert fem.operator.metadata.get("nonaffine_operator") is True  # re-assembly route
    assert list(fem.operator.runtime_parameter_exprs) == ["logk"]

    A1, b1 = fem.operator.evaluate({"logk": 0.0})  # truth k = exp(0) = 1
    u_obs = jnp.linalg.solve(jnp.asarray(A1), jnp.asarray(b1).reshape(-1))

    rec = _recover(fem.solve(), logk, u_obs, n=150)
    assert abs(rec - 0.7) > 0.3, "parameter did not move -- gradient did not reach it through re-assembly"
    assert abs(rec) < TOL, f"non-affine (exp(logk)): recovered logk={rec:.4f} (truth 0)"


def test_global_solve_runs_once_per_step_not_per_node():
    """Freeze the invariant: the global FEM solve is evaluated ~once per optimizer
    step, not vmapped once per mesh node."""
    alpha = _alpha()
    fem = _linear_fem(alpha)
    A1, b1 = fem.operator.evaluate({"alpha": 1.0})
    u_obs = jnp.linalg.solve(jnp.asarray(A1), jnp.asarray(b1).reshape(-1))
    n_nodes = int(fem.dofs)

    calls = [0]

    def counting_solve(A, b):
        jax.debug.callback(lambda: calls.__setitem__(0, calls[0] + 1))
        return jnp.linalg.solve(A, b)

    u_node = fem.solve(counting_solve)
    crux = jno.core([(u_node - u_obs).mse], domain=_DUMMY)
    calls[0] = 0
    steps = 20
    crux.solve(steps)
    assert calls[0] <= 4 * steps, (
        f"solve ran {calls[0]}x for {steps} steps on a {n_nodes}-node mesh "
        "-- the global solve appears to be vmapped per node"
    )


def test_nodal_field_parameter_recovers_via_crux():
    """jno.np.parameter(phi) -> a P1 nodal coefficient field k(x): trainable nodal
    values, interpolated to quad points during a re-assembled solve. Recover a smooth
    k(x) end-to-end through crux.solve."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))

    k = jno.np.parameter(phi, name="k")  # nodal field on the trial's FE space
    assert getattr(k.model, "_fem_field", None) == "node"
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem.is_linear
    assert fem.operator.metadata.get("nonaffine_operator") is True
    assert list(fem.operator.runtime_parameter_exprs) == ["k"]

    nodes = np.asarray(d.built_mesh.points)[:, :2]
    k_true = jnp.asarray(0.6 + 0.8 * nodes[:, 0] + 0.5 * nodes[:, 1])  # smooth, positive

    # Correctness: a *linear* nodal field must assemble the same operator as the same
    # coordinate-function coefficient (P1 interpolation of a linear field is exact) --
    # this catches any gather/node-order error.
    fem_ref = jno.fem(
        [(0.6 + 0.8 * xi + 0.5 * yi) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0],
        quad_degree=3,
    )
    A_field, _ = fem.operator.evaluate({"k": k_true})
    A_ref = np.asarray(fem_ref.A.todense() if hasattr(fem_ref.A, "todense") else fem_ref.A)
    assert np.max(np.abs(np.asarray(A_field) - A_ref)) < 1e-9, "nodal interpolation/gather mismatch"

    A_t, b = fem.operator.evaluate({"k": k_true})
    u_obs = jnp.linalg.solve(jnp.asarray(A_t), jnp.asarray(b).reshape(-1))

    A_t, b = fem.operator.evaluate({"k": k_true})
    u_obs = jnp.linalg.solve(jnp.asarray(A_t), jnp.asarray(b).reshape(-1))

    # Recover the full nodal field k(x) through crux.solve (the differentiable
    # re-assembled solve). Well-posed enough here from full-field data; field
    # inversion in general needs regularization (jno.fn.regularize), which composes
    # as an extra loss term.
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))  # start k=1 everywhere (nonsingular)
    k.optimizer(optax.adam(2e-2))
    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(400)

    # crux.eval([single_op]) returns the array itself (a (n_nodes, 1) field), NOT a
    # one-element list -- so reshape it; do not index [0] (that reads a single node).
    rec = np.asarray(crux.eval([k])).reshape(-1)
    assert rec.shape[0] == int(k_true.shape[0]), f"expected the full nodal field, got {rec.shape}"
    assert (rec.max() - rec.min()) > 1e-2, "recovered field is uniform -- not trained per node"
    rel = float(np.linalg.norm(rec - np.asarray(k_true)) / np.linalg.norm(np.asarray(k_true)))
    assert rel < 0.1, f"nodal field recovery via crux rel-err {rel:.3e}"
