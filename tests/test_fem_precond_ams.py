"""``jno.precond.ams()`` — the H(curl) auxiliary-space Maxwell preconditioner, through ``fem.solve``.

M4a: the spec is exercised on the REAL curl-curl+β·mass system via the ordinary steady-linear path
(no complex-solve changes yet). It must reproduce the direct solve and converge in the handful of
iterations M1/M2 established — where the gradient near-null-space makes an un-preconditioned CG crawl.
"""

import contextlib

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import pytest

import jno

inner, vec = jno.np.inner, jno.np.vector
# The lu auxiliary runs on CPU (jax-spsolve/cuSolver is flaky on GPU — the same reason the complex
# eddy path is CPU-pinned); the GPU-native aux is a multigrid inner solver (M5). Pin the solves.
_CPU = next(iter(jax.devices("cpu")), None)
_ON_CPU = jax.default_device(_CPU) if _CPU else contextlib.nullcontext()


def _solve(fem, **kw):
    with _ON_CPU:
        return np.asarray(jnp.asarray(fem.solve(**kw))).reshape(-1)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _curlcurl_source(mesh_size, beta):
    """A driven real curl-curl (+ β·mass) N1E problem: inner(curl u, curl v) + β⟨u,v⟩ − ⟨Js, v⟩."""
    d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    Js = vec(0.0 * x, 0.0 * y, 1.0 + 0.0 * z)
    term = inner(cu, cv) + beta * inner(ui, vi) - inner(Js, vi)
    return jno.fem([term])


def test_ams_matches_direct_solve_through_fem_solve():
    """fem.solve(linear=cg, precond=ams()) reproduces the sparse-direct solve — the spec is wired
    end-to-end (prepare builds G/Π from the mesh, materialize assembles the nodal aux, CG converges)."""
    fem = _curlcurl_source(0.3, beta=1e-4)
    x_lu = _solve(fem, linear=jno.solve.lu())
    x_ams = _solve(fem, linear=jno.solve.cg(tol=1e-9, maxiter=400), precond=jno.precond.ams())
    assert np.linalg.norm(x_ams - x_lu) / np.linalg.norm(x_lu) < 1e-7


def test_ams_converges_in_a_small_iteration_budget():
    """The whole point: AMS collapses the count. It reaches tol in a budget (60) far below the
    hundreds an un-preconditioned/Jacobi CG needs on this gradient-dominated (β→small) operator."""
    fem = _curlcurl_source(0.3, beta=1e-5)
    x_lu = _solve(fem, linear=jno.solve.lu())
    x_ams = _solve(fem, linear=jno.solve.cg(tol=1e-7, maxiter=60), precond=jno.precond.ams())
    assert np.linalg.norm(x_ams - x_lu) / np.linalg.norm(x_lu) < 1e-5  # converged within the budget
    with pytest.raises(Exception):  # Jacobi cannot, in the same budget — the null-space is undamped
        _solve(fem, linear=jno.solve.cg(tol=1e-7, maxiter=60), precond=jno.precond.jacobi())


def test_ams_custom_aux_solver():
    """The auxiliary nodal solves are a pluggable ``jno.solve`` inner solver (default lu()); an
    iterative inner solver works too — the hook M5 swaps a multigrid solver into."""
    fem = _curlcurl_source(0.3, beta=1e-4)
    x_lu = _solve(fem, linear=jno.solve.lu())
    spec = jno.precond.ams(aux=jno.solve.cg(tol=1e-10, maxiter=500))
    x = _solve(fem, linear=jno.solve.cg(tol=1e-9, maxiter=400), precond=spec)
    assert np.linalg.norm(x - x_lu) / np.linalg.norm(x_lu) < 1e-6


def test_ams_requires_a_gauge_on_pure_curl_curl():
    """A bare curl-curl has GᵀAG ≡ 0 (curl∘grad = 0) — no coercivity on the gradient space. The spec
    must refuse with actionable guidance rather than silently forming a singular auxiliary solve."""
    fem = _curlcurl_source(0.3, beta=0.0)
    with pytest.raises(ValueError, match="curl-curl"):
        _solve(fem, linear=jno.solve.cg(maxiter=50), precond=jno.precond.ams())


def test_ams_needs_the_owning_fem():
    """Materialised on a bare operator (no ctx.fem → no edge topology) it errors clearly, since G/Π
    are mesh objects, not recoverable from the matrix alone."""
    from jno.utils.solver.solver_api import LinearOperator, PrecondContext

    n = 6
    op = LinearOperator(jsparse.BCOO.fromdense(jnp.eye(n)))
    with pytest.raises(TypeError, match="owning FEM"):
        jno.precond.ams().materialize(PrecondContext(op, None))
