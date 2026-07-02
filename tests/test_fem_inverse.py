"""Differentiable FEM forward solve for inverse problems, authored entirely
through ``jno.fem([...])`` (no ``init_fem`` / ``assemble``).

``FEM.solve`` hosts a *real* parametric solve in the trace so ``crux.solve``
recovers a ``jno.np.parameter`` from data. The solver is the user's own callable
or the built-in default (the differentiable sparse-direct ``sparse_lu_solve`` for
linear, matrix-free Newton-Krylov for nonlinear); a bring-your-own dense solver is
exercised too. Implicit-diff lets the gradient reach the parameter without unrolling.

Run with x64 (assembly runs in float64): ``JAX_ENABLE_X64=1``.
"""

import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
TOL = 0.05  # adam recovery of a scalar parameter from clean data

# The inverse loss has no spatial Variable (the FEM solve is global), so crux needs
# an explicit domain to drive its loop. See jno.core(domain=).
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly/solves run in float64, so these tests need x64. Set it per-test with
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
    u_obs = jnp.linalg.solve(A1.todense(), jnp.asarray(b1).reshape(-1))

    rec = _recover(fem.solve(), alpha, u_obs)
    assert abs(rec - 2.0) > 0.5, "parameter did not move -- gradient did not reach it"
    assert abs(rec - 1.0) < TOL, f"linear (default solver): recovered alpha={rec:.4f}"


def test_linear_recovers_with_byo_dense_solver():
    alpha = _alpha()
    fem = _linear_fem(alpha)
    A1, b1 = fem.operator.evaluate({"alpha": 1.0})
    u_obs = jnp.linalg.solve(A1.todense(), jnp.asarray(b1).reshape(-1))

    # bring-your-own dense solver: solve_fn receives the raw BCOO operator, so densify it
    u_node = fem.solve(lambda A, b: jnp.linalg.solve(A.todense() if hasattr(A, "todense") else A, b))
    rec = _recover(u_node, alpha, u_obs)
    assert abs(rec - 1.0) < TOL, f"linear (BYO dense): recovered alpha={rec:.4f}"


def _argyris_biharmonic_fem(alpha, mesh_size=0.34):
    """Parametric conforming biharmonic ``alpha * Δ²u = f`` on the Argyris C¹ element; exact ``u = x⁴+y⁴``
    at ``alpha=1`` (``f = Δ²(x⁴+y⁴) = 48``), clamped to the manufactured trace. The 4th-order inverse
    analogue of ``_linear_fem`` -- it exercises the non-nodal *parametric* assembler."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols(space="Argyris")
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    lap = jno.np.laplacian
    f = 48.0 + 0.0 * xi
    g = xb**4 + yb**4
    weak = alpha * (lap(ui, [xi, yi]) * lap(vi, [xi, yi])) - f * vi
    return jno.fem([weak, u(xb, yb) - g])


def test_argyris_biharmonic_inverse_recovers_scalar():
    """A **4th-order** inverse problem: recover the plate stiffness ``alpha`` in ``alpha·Δ²u = f`` from a
    steady deflection, through the conforming C¹ Argyris element. Validates the non-nodal *parametric* path
    (a differentiable ``FemLinearSystem`` whose operator re-assembles at each ``alpha``) end-to-end via
    ``crux.solve`` -- the capability the non-nodal assembler previously lacked."""
    alpha = _alpha()  # starts at 2.0; truth is 1.0
    fem = _argyris_biharmonic_fem(alpha)
    assert fem.is_linear
    sys = fem.operator
    assert sys.is_parametric and list(sys.runtime_parameter_exprs) == ["alpha"]

    dense = lambda A: A.todense() if hasattr(A, "todense") else A  # noqa: E731 (the non-nodal operator is dense)
    A1, b1 = sys.evaluate({"alpha": 1.0})
    u_obs = jnp.linalg.solve(jnp.asarray(dense(A1)), jnp.asarray(b1).reshape(-1))

    u_node = fem.solve(lambda A, b: jnp.linalg.solve(jnp.asarray(dense(A)), jnp.asarray(b).reshape(-1)))
    rec = _recover(u_node, alpha, u_obs)
    assert abs(rec - 2.0) > 0.5, "parameter did not move -- gradient did not reach alpha through the C¹ solve"
    assert abs(rec - 1.0) < TOL, f"Argyris biharmonic inverse: recovered alpha={rec:.4f}"


def test_argyris_nonlinear_biharmonic_parametric_operator():
    """The non-nodal STEADY-NONLINEAR parametric branch: ``α·Δ²u + u³ = f`` on Argyris builds a parametric
    ``FemResidualOperator`` whose residual threads ``α``. (A full crux recovery is validated for the linear
    case above; optimising the ill-conditioned biharmonic Newton–Krylov is impractically slow for a unit
    test, so here we assert the branch is wired parametrically and ``α`` reaches the residual.)"""
    alpha = _alpha()
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4)
    u, phi = d.fem_symbols(space="Argyris")
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    lap = jno.np.laplacian
    weak = alpha * (lap(ui, [xi, yi]) * lap(vi, [xi, yi])) + (u * u * u) * phi - (48.0 + 0.0 * xi) * vi
    fem = jno.fem([weak, u(xb, yb) - (xb**4 + yb**4)])
    assert not fem.is_linear
    op = fem.operator
    assert type(op).__name__ == "FemResidualOperator" and op.is_parametric
    assert list(op.runtime_parameter_exprs) == ["alpha"]
    u_test = 0.1 * jnp.ones((int(op.size),), dtype=jnp.float64)
    r1 = jnp.asarray(op.residual(u_test, {"alpha": 1.0})).reshape(-1)
    r2 = jnp.asarray(op.residual(u_test, {"alpha": 2.0})).reshape(-1)
    assert float(jnp.max(jnp.abs(r1 - r2))) > 1e-6, "alpha did not thread into the nonlinear residual"


def _hermite_transient_fem(alpha, mesh_size=0.3):
    """Parametric transient diffusion ``u_t = alpha·Δu`` (weak ``∫u_t v + alpha·∫∇u·∇v``), homogeneous
    Dirichlet, IC ``sin(πx)sin(πy)`` ⇒ ``u(t) = exp(-alpha·2π²t)·IC``. A Hermite (C⁰) field exercises the
    non-nodal *parametric time block* (linear, ``operator_fn(t, args)``), well-conditioned so the default
    integrator's Krylov step converges fast (an Argyris transient inverse works the same way but is slow,
    like the ill-conditioned biharmonic)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, 0.02, 5))
    u, phi = d.fem_symbols(space="Hermite")
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    psi0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    weak = ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y)
    return jno.fem([weak, u(xb, yb) - 0.0, u(ci[0], ci[1]) - psi0])


def test_argyris_field_parameter_operator_matches_coordinate_coeff():
    """A P1 **field** parameter ``k(x)`` on an Argyris biharmonic (``k·Δu·Δv``): the coefficient is a P1
    field independent of the C¹ trial, gathered at the mesh vertices and interpolated with P1 shape
    functions. For a *linear* ``k`` the P1 interpolation is exact, so the field-parameter operator must
    equal the operator built from the same coordinate-function coefficient — the definitive check that the
    gather / node-order / interpolation on the non-nodal element is correct."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.35)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    lap = jno.np.laplacian
    f = 48.0 + 0.0 * xi
    g = xb**4 + yb**4

    kf, _ = d.fem_symbols()  # a P1 coefficient field, independent of the Argyris trial
    k = jno.np.parameter(kf, name="k")
    assert getattr(k.model, "_fem_field", None) == "node"
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([k * (lap(ui, [xi, yi]) * lap(vi, [xi, yi])) - f * vi, u(xb, yb) - g])
    assert fem.is_linear and list(fem.operator.runtime_parameter_exprs) == ["k"]

    nodes = np.asarray(d.built_mesh.points)[:, :2]
    k_true = jnp.asarray(0.6 + 0.8 * nodes[:, 0] + 0.5 * nodes[:, 1])  # smooth, positive, LINEAR

    u2, p2 = d.fem_symbols(space="Argyris")
    ux, vx = u2.bind(x=xi, y=yi), p2.bind(x=xi, y=yi)
    fem_ref = jno.fem([(0.6 + 0.8 * xi + 0.5 * yi) * (lap(ux, [xi, yi]) * lap(vx, [xi, yi])) - f * vx, u2(xb, yb) - g])
    dense = lambda A: np.asarray(A.todense() if hasattr(A, "todense") else A)  # noqa: E731
    A_field, _b = fem.operator.evaluate({"k": k_true})
    assert np.max(np.abs(dense(A_field) - dense(fem_ref.A))) < 1e-8, "P1 field-param interpolation/gather mismatch"


def test_hermite_field_parameter_recovers():
    """A **field** inverse: recover a spatially-varying coefficient ``k(x)`` in ``-∇·(k∇u) = f`` on a Hermite
    field, end-to-end through ``crux.solve``. Validates the gradient to the full P1 nodal field through the
    non-nodal parametric solve (Poisson-like, so well-posed from full-field data — no regularization needed)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    kf, _ = d.fem_symbols()
    k = jno.np.parameter(kf, name="k")
    u, phi = d.fem_symbols(space="Hermite")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
    assert fem.is_linear and list(fem.operator.runtime_parameter_exprs) == ["k"]

    nodes = np.asarray(d.built_mesh.points)[:, :2]
    k_true = jnp.asarray(0.6 + 0.8 * nodes[:, 0] + 0.5 * nodes[:, 1])
    dense = lambda A: A.todense() if hasattr(A, "todense") else A  # noqa: E731
    A_t, b = fem.operator.evaluate({"k": k_true})
    u_obs = jnp.linalg.solve(jnp.asarray(dense(A_t)), jnp.asarray(b).reshape(-1))

    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(optax.adam(2e-2))
    solver = lambda A, b: jnp.linalg.solve(jnp.asarray(dense(A)), jnp.asarray(b).reshape(-1))  # noqa: E731
    crux = jno.core([(fem.solve(solver) - u_obs).mse], domain=_DUMMY)
    crux.solve(400)
    rec = np.asarray(crux.eval([k])).reshape(-1)
    rel = float(np.linalg.norm(rec - np.asarray(k_true)) / np.linalg.norm(np.asarray(k_true)))
    assert rel < 0.05, f"field parameter k(x) not recovered: rel {rel:.3e}"


def test_hermite_transient_inverse_recovers_scalar():
    """A **transient** inverse: recover the diffusivity ``alpha`` from a decay trajectory through the
    non-nodal *parametric time block* — the operator ``A(alpha)`` is re-assembled each step and stays
    differentiable through the time-stepping. Validates the transient inverse (previously rejected) for the
    vertex/edge element families."""
    from jno.utils.solver.backend_blocks import _block_time_grid, _default_transient_integrate

    alpha = _alpha()  # starts at 2.0; truth is 1.0
    fem = _hermite_transient_fem(alpha)
    assert fem.is_transient
    block = fem.operator
    assert list(block.runtime_parameter_exprs) == ["alpha"]
    save_ts = _block_time_grid(block)
    u_obs = _default_transient_integrate(block, {"alpha": 1.0}, save_ts)  # observed trajectory at the truth

    rec = _recover(fem.solve(), alpha, u_obs, n=150)
    assert abs(rec - 2.0) > 0.3, "parameter did not move -- gradient did not reach alpha through the time-stepping"
    assert abs(rec - 1.0) < TOL, f"transient inverse: recovered alpha={rec:.4f}"


def test_nonlinear_recovers():
    from jno.utils.solver.newton_krylov import newton_krylov

    alpha = _alpha()
    fem = _nonlinear_fem(alpha)
    assert not fem.is_linear
    op = fem.operator
    assert type(op).__name__ == "FemResidualOperator" and op.is_parametric

    u0 = jnp.zeros((int(op.size),), dtype=jnp.float64)
    u_obs = newton_krylov(lambda uu: jnp.asarray(op.residual(uu, {"alpha": 1.0})).reshape(-1), u0)

    rec = _recover(fem.solve(), alpha, u_obs)  # default: matrix-free Newton-Krylov, implicit-diff
    assert abs(rec - 2.0) > 0.5, "parameter did not move -- implicit-diff gradient did not reach it"
    assert abs(rec - 1.0) < TOL, f"nonlinear: recovered alpha={rec:.4f}"


def test_nonaffine_scalar_recovers_via_reassembly():
    """A parameter inside a nonlinear function (``exp(logk)``) can't be factored into
    a constant basis, so the operator is re-assembled each call with the parameter
    threaded as InternalVars. The kernel is JAX, so the gradient still
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
    u_obs = jnp.linalg.solve(A1.todense(), jnp.asarray(b1).reshape(-1))

    rec = _recover(fem.solve(), logk, u_obs, n=150)
    assert abs(rec - 0.7) > 0.3, "parameter did not move -- gradient did not reach it through re-assembly"
    assert abs(rec) < TOL, f"non-affine (exp(logk)): recovered logk={rec:.4f} (truth 0)"


def test_global_solve_runs_once_per_step_not_per_node():
    """Freeze the invariant: the global FEM solve is evaluated ~once per optimizer
    step, not vmapped once per mesh node."""
    alpha = _alpha()
    fem = _linear_fem(alpha)
    A1, b1 = fem.operator.evaluate({"alpha": 1.0})
    u_obs = jnp.linalg.solve(A1.todense(), jnp.asarray(b1).reshape(-1))
    n_nodes = int(fem.dofs)

    calls = [0]

    def counting_solve(A, b):
        jax.debug.callback(lambda: calls.__setitem__(0, calls[0] + 1))
        return jnp.linalg.solve(A.todense() if hasattr(A, "todense") else A, b)

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
    assert np.max(np.abs(np.asarray(A_field.todense()) - A_ref)) < 1e-9, "nodal interpolation/gather mismatch"

    A_t, b = fem.operator.evaluate({"k": k_true})
    u_obs = jnp.linalg.solve(A_t.todense(), jnp.asarray(b).reshape(-1))

    A_t, b = fem.operator.evaluate({"k": k_true})
    u_obs = jnp.linalg.solve(A_t.todense(), jnp.asarray(b).reshape(-1))

    # Recover the full nodal field k(x) through crux.solve (the differentiable
    # re-assembled solve). Well-posed enough here from full-field data; field
    # inversion in general needs regularization (k.regularize(...)), which composes
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


def test_nodal_field_h1_regularizer():
    """k.regularize('h1seminorm') is the exact FE H1 seminorm integral|grad k|^2 = k^T L k
    (L = the stiffness/discrete Laplacian on the field's space): a differentiable smoothness
    loss term that composes through crux.solve."""
    from jno._fem import _assemble_h1_stiffness

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)  # unit square, area = 1
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    k = jno.np.parameter(phi, name="k")
    jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    n = int(d.built_mesh.points.shape[0])

    # L sanity: symmetric Laplacian with a constant null space.
    L = np.asarray(_assemble_h1_stiffness(d))
    assert L.shape == (n, n)
    assert np.allclose(L, L.T, atol=1e-9)
    assert np.max(np.abs(L @ np.ones(n))) < 1e-9

    # Exact H1 seminorm for a linear field: integral|grad k|^2 = (a^2 + b^2) * area.
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    a, b = 0.7, -1.3
    klin = jnp.asarray(a * nodes[:, 0] + b * nodes[:, 1])
    sem = float(klin @ (jnp.asarray(L) @ klin))
    assert abs(sem - (a * a + b * b)) < 1e-9, f"H1 seminorm {sem} != {a * a + b * b}"

    # The reg term composes + flattens a rough field through crux.solve (grad reaches k).
    k2 = jno.np.parameter(phi, name="k2")
    k2.dtype(jnp.float64)
    k2.initialize(jax.nn.initializers.normal(1.0))  # rough random per-node init
    k2.optimizer(optax.adam(5e-2))
    reg = k2.regularize("h1seminorm")
    crux = jno.core([reg.mean], domain=_DUMMY)
    k0 = np.asarray(crux.eval([k2])).reshape(-1)
    s0 = float(k0 @ (L @ k0))
    crux.solve(300)
    kf = np.asarray(crux.eval([k2])).reshape(-1)
    sf = float(kf @ (L @ kf))
    assert not np.allclose(k0, kf), "reg gradient did not reach the field"
    assert sf < 1e-2 * s0, f"pure-reg did not flatten the field: {s0} -> {sf}"

    # Guards: not a field parameter, and an unknown kind.
    with pytest.raises(ValueError):
        jno.np.parameter((1,), name="s").regularize("h1seminorm")
    with pytest.raises(ValueError):
        k.regularize("nope")


def test_nodal_field_regularizers():
    """The full field-parameter regularizer set on k.regularize(kind): l2/tikhonov
    (mass-weighted integral k^2), tv (total variation integral|grad k|), nonneg, bounded.
    Checked directly on each node's fn (FE-exact / pointwise, deterministic)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)  # unit square, area = 1
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    k = jno.np.parameter(phi, name="k")
    jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    ones = jnp.ones(nodes.shape[0])

    # l2 / tikhonov: integral k^2 = c^2 * area (mass-weighted); a reference shifts it.
    l2 = k.regularize("l2")
    assert abs(float(np.asarray(l2.fn(2.0 * ones)).sum()) - 4.0) < 1e-9  # c=2 -> 4*area
    assert abs(float(np.asarray(k.regularize("tikhonov").fn(ones)).sum()) - 1.0) < 1e-9  # integral 1 = area
    l2r = k.regularize("l2", ref=2.0)
    assert float(np.asarray(l2r.fn(2.0 * ones)).sum()) < 1e-9  # k == ref -> 0

    # tv: integral|grad k| = |a| * area for k = a*x; ~0 for a constant field.
    tv = k.regularize("tv")
    assert abs(float(np.asarray(tv.fn(jnp.asarray(1.5 * nodes[:, 0]))).sum()) - 1.5) < 1e-6
    assert float(np.asarray(tv.fn(ones)).sum()) < 1e-3  # constant -> ~0 (eps floor)

    # nonneg: zero for positive values, positive penalty for negative.
    nn = k.regularize("nonneg")
    assert float(np.asarray(nn.fn(ones)).sum()) == 0.0
    assert float(np.asarray(nn.fn(-ones)).sum()) > 0.0

    # bounded: zero inside [lo, hi], positive outside.
    bd = k.regularize("bounded", lo=0.0, hi=1.0)
    assert float(np.asarray(bd.fn(0.5 * ones)).sum()) == 0.0
    assert float(np.asarray(bd.fn(2.0 * ones)).sum()) > 0.0
    with pytest.raises(ValueError):
        k.regularize("bounded")  # missing lo/hi


# --------------------------------------------------------------------------- transient
# A *time-dependent* inverse problem: recover a parameter in a transient weak form from a
# u(t) trajectory. FEM.solve hosts the integrator as a trace node, so the gradient flows
# through the time integration to the parameter. The integrator is the user's callable
# (default: a backward-Euler lax.scan over the block's assembled dt); pass solve_fn= to bring
# your own stepper built from the block's M / A / residual. A Dirichlet problem zeroes M's
# Dirichlet rows -> a DAE, so the implicit backward-Euler default suits Dirichlet BCs.
TRANSIENT_TOL = 0.05


def _transient_heat_fem(alpha, *, mesh_size=0.2, time=(0.0, 0.1, 11)):
    """Parametric transient heat  u_t = alpha * lap u  on the unit square, homogeneous
    Dirichlet, IC sin(pi x) sin(pi y) (the mode decays at rate alpha * 2 pi^2)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=time)
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem([ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
    return d, fem


def _grid_ts(block):
    n_steps = int(round((float(block.t1) - float(block.t0)) / float(block.dt)))
    return jnp.linspace(float(block.t0), float(block.t1), n_steps + 1)


def test_transient_recovers_default_scan():
    """Transient inverse: recover the scalar alpha in u_t = alpha*lap u from the u(t)
    trajectory through the default backward-Euler scan in FEM.solve -- the gradient flows
    through the time integration (lax.scan) to alpha. Also freezes the fully-parametric
    operator Dirichlet-identity fix (without it M + dt A is singular) and the trajectory
    readback shape (n_save, n_dofs)."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    alpha = _alpha()
    d, fem = _transient_heat_fem(alpha)
    assert fem.is_transient
    block = fem.operator
    assert block.is_linear() and list(block.runtime_parameter_exprs) == ["alpha"]

    # Regression guard for the fully-parametric-operator fix: the entire operator is
    # parametric (no static term), so without restoring the Dirichlet identity in A the
    # backward-Euler matrix M + dt A would be singular (rank-deficient on the bc rows).
    M = np.asarray(block.M.todense() if hasattr(block.M, "todense") else block.M)  # block.M is a BCOO -> densify
    dt = float(block.dt)
    S = M + dt * np.asarray(block.operator_fn(dt, {"alpha": 1.0}).todense())
    assert np.linalg.matrix_rank(S) == S.shape[0], "M + dt A is singular -- Dirichlet identity missing from A"

    n_dofs = int(M.shape[0])
    save_ts = _grid_ts(block)
    u_obs = _default_transient_integrate(block, {"alpha": 1.0}, save_ts)  # truth trajectory at alpha=1
    assert u_obs.shape == (len(save_ts), n_dofs)
    assert not bool(jnp.isnan(u_obs).any())

    # Forward physics: the recovery loss alone cannot catch a *self-consistent* forward bug
    # (u_obs and fem.solve() share the integrator, so e.g. a 2x operator scale cancels and
    # still "recovers" 1.0). The IC is a single Laplacian eigenmode sin(pi x) sin(pi y)
    # (eigenvalue 2 pi^2), so the exact solution decays as u(t) = exp(-alpha*2*pi^2*t) u(0).
    # Check the backward-Euler trajectory decays at that rate -- order-safe (uses only the
    # state vector). BE is O(dt) here (~10%); a 2x operator-scale bug would be ~77%.
    decay = np.exp(-1.0 * 2.0 * PI**2 * float(save_ts[-1]))
    ref = decay * np.asarray(u_obs[0])
    fwd_rel = float(np.linalg.norm(np.asarray(u_obs[-1]) - ref) / np.linalg.norm(ref))
    assert fwd_rel < 0.15, f"forward backward-Euler heat decay rel-err {fwd_rel:.3f} -- physics wrong?"

    u_node = fem.solve()
    crux = jno.core([(u_node - u_obs).mse], domain=_DUMMY)
    crux.solve(200)
    rec = float(np.asarray(crux.eval([alpha])).reshape(-1)[0])
    # crux.eval([single_op]) returns the array itself: the full trajectory, NOT a list and
    # NOT a single time-slice -- so its shape is (n_save, n_dofs), do not index [0].
    traj = np.asarray(crux.eval([u_node]))
    assert traj.shape == (len(save_ts), n_dofs), f"trajectory readback {traj.shape}"

    assert abs(rec - 2.0) > 0.5, "alpha did not move -- gradient did not reach it through the integrator"
    assert abs(rec - 1.0) < TRANSIENT_TOL, f"transient (default scan): recovered alpha={rec:.4f}"


def test_transient_save_ts_decouples_from_dt():
    """save_ts only samples the output; the step is the block's assembled dt. A coarse
    save_ts keeps full-dt accuracy (sampled by interpolation) and its shape follows save_ts
    -- the step size is never an accident of how the output is sampled."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    _, fem = _transient_heat_fem(_alpha())
    block = fem.operator
    n_dofs = int(block.M.shape[0])  # BCOO or dense — both expose .shape

    fine = _default_transient_integrate(block, {"alpha": 1.0}, _grid_ts(block))
    coarse_ts = jnp.array([float(block.t0), float(block.t1)])  # only 2 output points
    coarse = _default_transient_integrate(block, {"alpha": 1.0}, coarse_ts)

    assert coarse.shape == (2, n_dofs)  # shape follows save_ts
    # end state at t1 is integrated at block.dt regardless of how few points we save:
    assert float(jnp.linalg.norm(coarse[-1] - fine[-1])) < 1e-10


def test_transient_solve_fn_override():
    """fem.solve(my_integrator) -- bring-your-own transient integrator (same role as the steady
    (A,b)->u escape hatch): fem.solve routes through the user-supplied stepper and the gradient
    still flows through it to recover alpha."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    alpha = _alpha()
    _, fem = _transient_heat_fem(alpha)
    block = fem.operator
    u_obs = _default_transient_integrate(block, {"alpha": 1.0}, _grid_ts(block))

    def my_integrator(blk, args, ts):
        # a user-supplied stepper handed to fem.solve (here, the built-in backward-Euler);
        # extract M / A / residual from the block and integrate however you like.
        return _default_transient_integrate(blk, args, ts)

    rec = _recover(fem.solve(my_integrator), alpha, u_obs, n=200)
    assert abs(rec - 1.0) < TRANSIENT_TOL, f"transient (solve_fn override): recovered alpha={rec:.4f}"


def test_transient_nonlinear_recovers_via_inner_newton():
    """The default transient integrator handles a NONLINEAR block too: backward Euler with the
    matrix-free Newton-Krylov root find per step (implicit-diff). Recover a scalar alpha in
    u_t = lap u - alpha*u^3 from the trajectory through fem.solve() -- the gradient flows
    through the per-step Newton to alpha without unrolling it."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    alpha = _alpha()
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15, time=(0.0, 0.05, 11))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem(
        [ui.t * vi + (ui.x * vi.x + ui.y * vi.y) + alpha * (u * u * u) * vi, u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0]
    )
    assert fem.is_transient and not fem.operator.is_linear()
    u_obs = _default_transient_integrate(fem.operator, {"alpha": 1.0}, _grid_ts(fem.operator))
    assert u_obs.ndim == 2 and not bool(jnp.isnan(u_obs).any())
    rec = _recover(fem.solve(), alpha, u_obs, n=200)
    assert abs(rec - 1.0) < TRANSIENT_TOL, f"nonlinear-transient (inner Newton): recovered alpha={rec:.4f}"


# --------------------------------------------------------- non-homogeneous Dirichlet + parametric
# A parametric operator (e.g. k * stiffness) lifts a non-homogeneous Dirichlet value g into a
# parameter-scaled RHS term; jno.fem now carries that term (b(theta) = b0 + sum theta*bK) instead
# of rejecting it. (Only a parameter that scales the Dirichlet *value* itself stays unsupported.)


def test_steady_nonhomog_dirichlet_parametric_recovers():
    """Steady: -k lap u = f with u = 1 on the boundary (non-homogeneous g). Manufactured
    u* = 1 + x(1-x)y(1-y); recover the scalar k (was 'Runtime Dirichlet parameters not supported')."""
    k = _alpha()  # trainable scalar named "alpha", start 2.0
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 1.0], quad_degree=3)
    A1, b1 = fem.operator.evaluate({"alpha": 1.0})
    u_obs = jnp.linalg.solve(A1.todense(), jnp.asarray(b1).reshape(-1))
    rec = _recover(fem.solve(), k, u_obs)
    assert abs(rec - 1.0) < TOL, f"steady non-homogeneous parametric: recovered k={rec:.4f}"


def test_transient_nonhomog_dirichlet_parametric_recovers():
    """Transient: u_t = alpha lap u with u = 1 on the boundary (non-homogeneous g), IC
    1 + sin(pi x) sin(pi y). The lifting is threaded into the time-block forcing; recover alpha
    from the u(t) trajectory through fem.solve()."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    alpha = _alpha()
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.12, time=(0.0, 0.1, 21))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    u0 = 1.0 + jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem([ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 1.0, u(ci[0], ci[1]) - u0])
    u_obs = _default_transient_integrate(fem.operator, {"alpha": 1.0}, _grid_ts(fem.operator))
    bdry = np.asarray(fem.points)
    on_bdry = np.isclose(bdry[:, 0], 0) | np.isclose(bdry[:, 0], 1) | np.isclose(bdry[:, 1], 0) | np.isclose(bdry[:, 1], 1)
    assert np.allclose(np.asarray(u_obs[-1])[on_bdry], 1.0, atol=1e-6)  # non-homog g=1 held
    rec = _recover(fem.solve(), alpha, u_obs, n=200)
    assert abs(rec - 1.0) < TRANSIENT_TOL, f"transient non-homogeneous parametric: recovered alpha={rec:.4f}"


def test_transient_field_parameter_recovers():
    """A jno.np.parameter(phi) FIELD coefficient in a *transient* form: u_t = div(k(x) grad u).
    The transient non-affine route threads the node field as InternalVars (per-cell gather +
    shape-fn interpolation), the same machinery as the steady field route. Recover the full
    nodal k(x) field from the u(t) trajectory through fem.solve()."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2, time=(0.0, 0.05, 11))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    k = jno.np.parameter(phi, name="k")  # nodal FIELD parameter
    assert getattr(k.model, "_fem_field", None) == "node"
    fem = jno.fem([ui.t * vi + k * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
    assert fem.is_transient

    nodes = np.asarray(d.built_mesh.points)[:, :2]
    k_true = jnp.asarray(1.0 + 0.5 * nodes[:, 0] + 0.3 * nodes[:, 1])  # smooth, positive
    u_obs = _default_transient_integrate(fem.operator, {"k": k_true}, _grid_ts(fem.operator))
    assert u_obs.shape[0] == 11 and not bool(jnp.isnan(u_obs).any())

    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(optax.adam(3e-2))
    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(300)
    rec = np.asarray(crux.eval([k])).reshape(-1)  # the recovered nodal field
    assert rec.shape[0] == int(k_true.shape[0]), f"expected the full nodal field, got {rec.shape}"
    rel = float(np.linalg.norm(rec - np.asarray(k_true)) / np.linalg.norm(np.asarray(k_true)))
    assert rel < 0.1, f"transient field k(x) recovery rel_L2 {rel:.3e}"


# --------------------------------------------------- coupled (multi-field) parametric transient
# A runtime parameter inside a COUPLED nonlinear transient used to be rejected by
# _assemble_multifield (NotImplementedError "parametric=True"); the native block already threads
# scalar parameters through `args`, so the gate now lets the coupled case through.


def _coupled_transient_fem(k, *, mesh_size=0.25, time=(0.0, 0.05, 6)):
    """Two coupled fields u, w with a shared nonlinear reaction k*u*w -- a multi-field, nonlinear,
    *parametric* transient (the case _assemble_multifield used to reject)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=time)
    u, pu = d.fem_symbols(names=("u", "pu"))
    w, pw = d.fem_symbols(names=("w", "pw"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, pui = u.bind(x=xi, y=yi, t=ti), pu.bind(x=xi, y=yi, t=ti)
    wi, pwi = w.bind(x=xi, y=yi, t=ti), pw.bind(x=xi, y=yi, t=ti)
    u0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    return d, jno.fem(
        [
            ui.t * pui + (ui.x * pui.x + ui.y * pui.y) + k * ui * wi * pui,
            wi.t * pwi + (wi.x * pwi.x + wi.y * pwi.y) + k * ui * wi * pwi,
            u(xb, yb) - 0.0,
            w(xb, yb) - 0.0,
            u(ci[0], ci[1]) - u0,
            w(ci[0], ci[1]) - u0,
        ]
    )


def test_coupled_transient_parametric_recovers():
    """Multi-field (u, w) NONLINEAR TRANSIENT with a runtime scalar parameter k -- the coupled case
    _assemble_multifield used to reject with NotImplementedError. Regression: it now assembles
    natively, and crux.solve differentiates through the coupled backward-Euler scan to recover k
    from the joint (u, w) trajectory."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    k = jno.np.parameter((1,), name="k")
    _, fem = _coupled_transient_fem(k)
    assert fem.is_transient and not fem.operator.is_linear()
    assert list(fem.operator.runtime_parameter_exprs) == ["k"]

    u_obs = _default_transient_integrate(fem.operator, {"k": 1.0}, _grid_ts(fem.operator))  # truth at k=1
    assert u_obs.ndim == 2 and not bool(jnp.isnan(u_obs).any())

    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(2.0))  # start far from truth = 1
    k.optimizer(optax.adam(5e-2))
    crux = jno.core([(fem.solve() - u_obs).mse], domain=_DUMMY)
    crux.solve(250)
    rec = float(np.asarray(crux.eval([k])).reshape(-1)[0])
    assert abs(rec - 2.0) > 0.3, "k did not move -- gradient did not reach it through the coupled integrator"
    assert abs(rec - 1.0) < TRANSIENT_TOL, f"coupled transient: recovered k={rec:.4f}"


# ------------------------------------------------------------------- parameter naming ergonomics


def test_unnamed_parameters_get_unique_names():
    """Two unnamed jno.np.parameter() used to both default to the name 'value' and the assembler
    rejected the form ("Multiple runtime parameter models use the name 'value'"). Each now gets a
    unique id-based name, so several unnamed parameters coexist in one weak form."""
    from jno.utils.solver.parametric_helpers import _collect_runtime_parameter_exprs

    a = jno.np.parameter((1,))
    b = jno.np.parameter((1,))
    na, nb = a.model._parameter_name, b.model._parameter_name
    assert na and nb and na != nb, f"unnamed parameters collide: {na!r} == {nb!r}"

    collected = _collect_runtime_parameter_exprs(a * b)  # a tree containing both -> no collision raise
    assert sorted(collected) == sorted([na, nb])


def test_name_method_sets_parameter_identity():
    """`.name(label)` on a parameter sets its *solver identity* (_parameter_name), not just a log
    label -- so jno.np.parameter((1,)).name('k') is equivalent to jno.np.parameter((1,), name='k').
    On a non-parameter expression `.name()` stays a pure label and must not grow _parameter_name."""
    from jno.utils.solver.parametric_helpers import _parameter_name

    p = jno.np.parameter((1,)).name("kappa")
    assert p.model._parameter_name == "kappa"
    assert _parameter_name(p) == "kappa"

    labelled = (p * 2.0).name("loss")  # a BinaryOp, not a parameter node
    assert getattr(labelled, "_user_name", None) == "loss"
    assert not hasattr(labelled, "_parameter_name")
