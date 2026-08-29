"""The ``jno.fem([...])`` driver: classify traced residuals (test-function
presence x region) and assemble into an ``FEM`` container of
matrices/operators (no solve).

Covers:
* steady linear Poisson (Dirichlet) assembles + solves to the manufactured
  solution, and matches the legacy ``init_fem`` + ``assemble(target="fem_system")``
  path byte-for-byte;
* a nonlinear form classifies as a residual operator (``residual``/``jacobian``);
* a mixed Dirichlet + Neumann problem buckets correctly and the Neumann term
  contributes to the load;
* error cases (test-function-free interior term, empty input) raise;
* the ``FEM`` accessor guards (``.A`` on nonlinear, ``.residual`` on linear).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly is float64; opt into x64 per-test (the session default may be x64-off when
    co-run with test_periodic). Save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_solve_fn_receives_the_operator_as_assembled_not_densified():
    """``solve_fn`` is *your* solver, so it gets the operator exactly as assembled — the BCOO.

    It used to be densified first, which made bringing your own SPARSE solver pointless at the sizes
    where you would want one: densifying is ``O(n²)`` on an ``O(nnz)`` operator — 418 MiB against 1.3
    MiB at n=7407, and 20.5 GiB against 9.4 MiB at n=51843, which OOMs before the solver runs.

    Pinned by capturing what the callable is handed, because this is the kind of contract that is
    easy to reverse by accident and impossible to notice from a correct answer."""

    seen = {}

    def capture(A, b):
        seen["sparse"] = hasattr(A, "indices")
        seen["nbytes"] = int(A.data.nbytes + A.indices.nbytes) if seen["sparse"] else int(A.nbytes)
        seen["n"] = int(A.shape[0])
        return jnp.linalg.solve(_dense(A), jnp.asarray(b).reshape(-1))

    _d, fem = _poisson_fem()
    u = np.asarray(fem.solve(solve_fn=capture)).reshape(-1)

    assert seen["sparse"], "solve_fn was handed a densified operator, not the assembled BCOO"
    n = seen["n"]
    assert seen["nbytes"] < n * n * 8, f"the operator carries {seen['nbytes']} B, no better than dense"
    assert np.all(np.isfinite(u))
    # and it still solves the system the assembler built
    A, b = fem.operator
    rel = float(jnp.linalg.norm(jnp.asarray(A @ jnp.asarray(u)) - jnp.asarray(b).reshape(-1)))
    assert rel / max(1e-30, float(jnp.linalg.norm(jnp.asarray(b)))) < 1e-10


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _poisson_fem(mesh_size=0.12):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    weak = ui.x * vi.x + ui.y * vi.y - f * vi
    bc = u(xb, yb) - 0.0
    return d, jno.fem([weak, bc], quad_degree=3)


def test_poisson_dirichlet_assembles_and_solves():
    d, fem = _poisson_fem()
    assert fem.is_linear
    A, b = _dense(fem.A), np.asarray(fem.b).reshape(-1)
    assert A.shape == (fem.dofs, fem.dofs)
    assert np.allclose(A, A.T, atol=1e-8)  # symmetric stiffness
    u_sol = np.linalg.solve(A, b)
    c = np.asarray(d.mesh.points)[:, :2]
    u_exact = c[:, 0] * (1 - c[:, 0]) * c[:, 1] * (1 - c[:, 1])
    rel = np.linalg.norm(u_exact - u_sol) / np.linalg.norm(u_exact)
    assert rel < 1e-2


def test_linear_solve_default_and_custom_solver_match():
    """A non-parametric steady linear FEM is solvable through ``fem.solve()`` -- both the default
    (matrix-free BiCGStab on the BCOO operator, to solver tolerance) and a user
    ``solve_fn=(A, b) -> u`` (dense, exact) -- and both equal the direct ``A^-1 b``."""
    _, fem = _poisson_fem()
    direct = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
    u_default = np.asarray(fem.solve())
    # `solve_fn` receives the operator AS ASSEMBLED (a BCOO), never a densified copy -- densifying an
    # O(nnz) operator would be O(n^2) and would defeat bringing your own sparse solver. A dense solver
    # therefore densifies itself, which is the one line below.
    u_custom = np.asarray(fem.solve(solve_fn=lambda A, b: jnp.linalg.solve(_dense(A), b)))
    assert np.allclose(u_default, direct, atol=1e-6)  # iterative: converged to BiCGStab tol
    assert np.allclose(u_custom, direct, atol=1e-10)  # dense direct: exact


def test_nonlinear_form_is_residual_operator():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.18)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    weak = 0.01 * (ui.x * vi.x + ui.y * vi.y) + (u * u * u - u) * phi
    fem = jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)
    assert not fem.is_linear
    assert callable(fem.residual) and callable(fem.jacobian)
    R = np.asarray(fem.residual(np.zeros(fem.dofs)))
    assert R.shape == (fem.dofs,)


def test_mixed_dirichlet_neumann_buckets_and_contributes():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    weak = ui.x * vi.x + ui.y * vi.y - 1.0 * vi
    flux = 0.5 * phi.bind(x=xr, y=yr)
    bc = u(xl, yl) - 0.0

    fem = jno.fem([weak, flux, bc], quad_degree=3)
    assert any("surface@right" in s for s in fem.classification)
    assert any("dirichlet@left" in s for s in fem.classification)

    fem_noflux = jno.fem([weak, bc], quad_degree=3)
    assert not np.allclose(np.asarray(fem.b).reshape(-1), np.asarray(fem_noflux.b).reshape(-1))


def test_test_function_free_interior_term_raises():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, _ = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    with pytest.raises(ValueError):
        jno.fem([u(xi, yi) - 1.0])  # interior, trial-only -> forgot the test function


def test_empty_constraints_raises():
    with pytest.raises(ValueError):
        jno.fem([])


def test_fem_accessor_guards():
    _, fem = _poisson_fem(mesh_size=0.3)
    assert fem.is_linear
    with pytest.raises(AttributeError):
        _ = fem.residual
    with pytest.raises(AttributeError):
        _ = fem.jacobian


@pytest.mark.parametrize("kind", ["jno_fn", "arith"])
def test_position_dependent_dirichlet(kind):
    # Manufactured u = x^2 + y^2  ->  -lap(u) = -4, with u = x^2+y^2 on the boundary.
    # The position-dependent value is evaluated through the existing TraceEvaluator.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    weak = ui.x * vi.x + ui.y * vi.y + 4.0 * vi
    if kind == "jno_fn":
        g = jno.fn(lambda x, y: x**2 + y**2, [xb, yb])
    else:
        g = xb**2 + yb**2

    fem = jno.fem([weak, u(xb, yb) - g], quad_degree=3)
    A = _dense(fem.A)
    u_sol = np.linalg.solve(A, np.asarray(fem.b).reshape(-1))
    c = np.asarray(d.mesh.points)[:, :2]
    u_exact = c[:, 0] ** 2 + c[:, 1] ** 2
    rel = np.linalg.norm(u_exact - u_sol) / np.linalg.norm(u_exact)
    assert rel < 1e-2


# ---------------------------------------------------------------------------
# transient (semidiscrete) assembly — M + operator, state0 from the IC,
# integration window from jno.domain(time=...)
# ---------------------------------------------------------------------------
def _heat_fem(mesh_size=0.12, n_time=6):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, 0.1, n_time))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    nu = 0.1
    weak = ui.t * vi + nu * (ui.x * vi.x + ui.y * vi.y)
    bc = u(xb, yb) - 0.0
    ic = u(xi0, yi0) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [xi0, yi0])
    return d, jno.fem([weak, bc, ic], quad_degree=3), nu


def test_transient_assembles_to_mass_and_operator():
    d, fem, _ = _heat_fem()
    assert fem.is_transient and fem.is_linear
    n = fem.dofs
    assert _dense(fem.M).shape == (n, n)
    assert np.asarray(fem.state0).shape == (n,)
    # time window comes from jno.domain(time=(0, 0.1, 6)) -> dt = 0.1/5
    assert (fem.t0, fem.t1) == (0.0, 0.1)
    assert abs(fem.dt - 0.02) < 1e-9


def test_transient_state0_from_initial_condition():
    d, fem, _ = _heat_fem()
    c = np.asarray(d.mesh.points)[:, :2]
    ic_exact = np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])
    assert np.allclose(np.asarray(fem.state0), ic_exact, atol=1e-6)


def test_transient_backward_euler_step_decays():
    # one implicit step of M u_dot + nu K u = 0 must decay like 1/(1 + 2 nu pi^2 dt)
    d, fem, nu = _heat_fem()
    M, A = _dense(fem.M), _dense(fem.operator.A)
    u0, dt = np.asarray(fem.state0), float(fem.dt)
    u1 = np.linalg.solve(M + dt * A, M @ u0)
    ratio = np.max(np.abs(u1)) / np.max(np.abs(u0))
    analytic = 1.0 / (1.0 + 2 * nu * np.pi**2 * dt)
    assert abs(ratio - analytic) < 0.02


def test_initial_condition_without_time_derivative_raises():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 0.1, 6))
    u, phi = d.fem_symbols()
    xi, yi, _ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    steady_weak = ui.x * vi.x + ui.y * vi.y - 1.0 * vi  # no time derivative
    with pytest.raises(ValueError):
        jno.fem([steady_weak, u(xb, yb) - 0.0, u(xi0, yi0) - 0.0])


def test_transient_accessor_guards():
    _, fem, _ = _heat_fem(mesh_size=0.3)
    assert fem.is_transient
    with pytest.raises(AttributeError):
        _ = fem.A  # steady-only accessor
    _, steady = _poisson_fem(mesh_size=0.3)
    with pytest.raises(AttributeError):
        _ = steady.M  # transient-only accessor


def _elasticity_fem(eps_of, mesh_size=0.15):
    """Plane-strain elasticity, clamped on the left, under gravity -- built through whichever
    spelling of the symmetric gradient ``eps_of`` uses."""
    inner, trace = jno.np.inner, jno.np.trace
    lam = mu = 1.0
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    xi, yi, _ = d.variable("interior", split=True)
    xL, yL, _ = d.variable("left", split=True)
    u, phi = d.fem_symbols(value_shape=(2,))
    eu, ep = eps_of(u, [xi, yi]), eps_of(phi, [xi, yi])
    weak = lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2) - inner((0.0, -1.0), phi, n_contract=1)
    return jno.fem([weak, u(xL, yL) - (0.0, 0.0)])


def test_sym_of_grad_is_linear_like_the_fused_symgrad():
    """``sym(grad(u))`` and ``symgrad(u, coords)`` are the same tensor, so they must classify the
    same way. They did not: ``sym`` was missing from the linear-wrapper set the unknown-dependence
    heuristic consults, so the second spelling of an ordinary linear elasticity was routed to
    ``fem_residual`` and paid a Newton solve for an assembled linear system.

    ``sym(A) = 0.5*(A + A^T)`` is linear in ``A`` -- and ``symgrad`` IS this composition fused, so
    the two agreeing is the oracle: same mode, and the same solution to the last bit."""
    symgrad, sym, grad = jno.np.symgrad, jno.np.sym, jno.np.grad

    fused = _elasticity_fem(lambda w, c: symgrad(w, c))
    composed = _elasticity_fem(lambda w, c: sym(grad(w, c)))

    assert fused.is_linear, "symgrad elasticity is linear (the case that already worked)"
    assert composed.is_linear, "sym(grad(u)) is the same tensor, so it is linear too"

    x_fused = np.asarray(fused.solve()).reshape(-1)
    x_composed = np.asarray(composed.solve()).reshape(-1)
    assert np.linalg.norm(x_fused) > 1e-6  # not the trivial all-zero solution
    assert np.allclose(x_fused, x_composed, rtol=1e-10, atol=1e-12)


def test_antisym_is_linear_in_the_unknown():
    """``antisym(A) = 0.5*(A - A^T)`` is the sibling of ``sym`` and just as linear; it was missing
    from the same set. The rotation term below is linear, so the form must assemble."""
    inner, antisym, grad = jno.np.inner, jno.np.antisym, jno.np.grad
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    xi, yi, _ = d.variable("interior", split=True)
    xL, yL, _ = d.variable("left", split=True)
    u, phi = d.fem_symbols(value_shape=(2,))
    wu, wp = antisym(grad(u, [xi, yi])), antisym(grad(phi, [xi, yi]))
    weak = (
        inner(wu, wp, n_contract=2)
        + inner(grad(u, [xi, yi]), grad(phi, [xi, yi]), n_contract=2)
        - inner((0.0, -1.0), phi, n_contract=1)
    )
    fem = jno.fem([weak, u(xL, yL) - (0.0, 0.0)])
    assert fem.is_linear
