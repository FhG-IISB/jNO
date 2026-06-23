"""Differentiable FEM forward solve for inverse problems, authored entirely
through ``jno.fem([...])`` (no ``init_fem`` / ``assemble``).

``FEM.solve`` hosts a *real* parametric solve in the trace so ``crux.solve``
recovers a ``jno.np.parameter`` from data. The solver is the user's own callable
or the built-in default (the differentiable sparse-direct ``sparse_lu_solve`` for
linear, matrix-free Newton-Krylov for nonlinear); a bring-your-own dense solver is
exercised too. Implicit-diff lets the gradient reach the parameter without unrolling.

Run with x64 (the feax assembly is float64): ``JAX_ENABLE_X64=1``.
"""

import pytest

pytest.importorskip("feax", reason="feax required for FEM inverse tests")
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


def test_linear_recovers_with_byo_dense_solver():
    alpha = _alpha()
    fem = _linear_fem(alpha)
    A1, b1 = fem.operator.evaluate({"alpha": 1.0})
    u_obs = jnp.linalg.solve(jnp.asarray(A1), jnp.asarray(b1).reshape(-1))

    u_node = fem.solve(lambda A, b: jnp.linalg.solve(A, b))  # bring-your-own dense solver
    rec = _recover(u_node, alpha, u_obs)
    assert abs(rec - 1.0) < TOL, f"linear (BYO dense): recovered alpha={rec:.4f}"


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
    S = M + dt * np.asarray(block.operator_fn(dt, {"alpha": 1.0}))
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
    n_dofs = int(np.asarray(block.M).shape[0])

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
    u_obs = jnp.linalg.solve(jnp.asarray(A1), jnp.asarray(b1).reshape(-1))
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
