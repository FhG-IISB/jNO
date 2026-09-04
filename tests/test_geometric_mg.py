"""Geometric multigrid — ``jno.precond.gmg()`` / ``build_vcycle``, a structured-grid V-cycle
preconditioner for the constant-coefficient Poisson operators ``jno.fdm`` produces on a regular grid.

The defining test is **grid-independent convergence**: a correct multigrid reduces the residual by a
constant factor per V-cycle regardless of grid size (O(N) solve)."""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np
import pytest

import jno  # noqa: E402
from jno.utils.solver.geometric_mg import _interior_mask, _neg_laplacian, build_vcycle  # noqa: E402
from jno.utils.solver.solver_api import LinearOperator, PrecondContext  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64. The session default is x64-off (see tests/conftest.py), and this
    flag is process-wide -- save/restore keeps it from leaking to whatever module runs next."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(size, dim=2):
    """Structured -Δu = f, u = Π sin(π x_i), homogeneous Dirichlet. Returns (grid, A_mv, b, exact)."""
    shp = (
        jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size)
        if dim == 2
        else jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=size)
    )
    d = jno.domain(shp.structured())
    grid = d.mesh_connectivity["grid"]
    shape, spacing = grid["shape"], grid["spacing"]
    p = np.asarray(d.mesh_connectivity["points"])[:, :dim]
    interior = _interior_mask(shape)
    int_flat = jnp.asarray(np.asarray(interior).reshape(-1) > 0.5)
    exact = np.prod([np.sin(np.pi * p[:, i]) for i in range(dim)], axis=0)

    def A_mv(u):
        nl = _neg_laplacian(u.reshape(shape), spacing, interior).reshape(-1)
        return jnp.where(int_flat, nl, u)  # identity boundary rows

    b = jnp.asarray(np.where(np.asarray(int_flat), dim * np.pi**2 * exact, 0.0))
    return grid, A_mv, b, exact, np.asarray(int_flat)


def _iterate_vcycle(A_mv, b, apply, n=25):
    """Standalone V-cycle iteration ``u ← u + M⁻¹(b − A u)``; returns (avg residual factor, u)."""
    u = jnp.zeros_like(b)
    r0 = float(jnp.linalg.norm(b - A_mv(u)))
    prev, facs, it = r0, [], 0
    for it in range(n):
        r = b - A_mv(u)
        rn = float(jnp.linalg.norm(r))
        if it > 0:
            facs.append(rn / prev)
        prev = rn
        if rn / r0 < 1e-10:
            break
        u = u + apply(r)
    return float(np.mean(facs[2:8])) if len(facs) > 8 else float(np.mean(facs)), u, it


def test_vcycle_grid_independent_convergence():
    """The multigrid property: ~0.1 residual reduction per V-cycle, independent of grid size."""

    def run(size):
        grid, A_mv, b, exact, _ = _poisson(size)
        apply, nlev = build_vcycle(grid["shape"], grid["spacing"])
        assert nlev >= 3
        fac, u, _ = _iterate_vcycle(A_mv, b, apply)
        return fac, float(np.linalg.norm(np.asarray(u) - exact) / np.linalg.norm(exact))

    f_coarse, e_coarse = run(0.05)  # 20 cells → 3 levels
    f_fine, e_fine = run(0.025)  # 40 cells → 4 levels
    assert f_coarse < 0.2 and f_fine < 0.2  # textbook multigrid factor
    assert abs(f_coarse - f_fine) < 0.08  # grid-INDEPENDENT (the hallmark)
    assert e_coarse < 1e-2 and e_fine < 3e-3  # solves to discretisation accuracy


def test_vcycle_3d():
    grid, A_mv, b, exact, _ = _poisson(0.1, dim=3)  # 10 cells → coarsenable
    apply, nlev = build_vcycle(grid["shape"], grid["spacing"])
    assert nlev >= 2
    fac, u, iters = _iterate_vcycle(A_mv, b, apply)
    assert fac < 0.25
    assert float(np.linalg.norm(np.asarray(u) - exact) / np.linalg.norm(exact)) < 2e-2


def test_gmg_precond_solves_poisson():
    """jno.precond.gmg() materializes from ctx.grid and preconditions GMRES to solve Poisson."""
    grid, A_mv, b, exact, _ = _poisson(0.05)
    op = LinearOperator.from_matvec(A_mv, shape=(b.shape[0], b.shape[0]))
    applier = jno.precond.gmg().materialize(PrecondContext(op, grid=grid))
    sol = jno.solve.gmres(maxiter=50)(op, b, M=applier)
    assert float(np.linalg.norm(np.asarray(sol) - exact) / np.linalg.norm(exact)) < 1e-2


def test_gmg_preconditioned_solve_is_differentiable():
    """A GMG-preconditioned GMRES solve is reverse-mode differentiable w.r.t. the RHS scale."""
    grid, A_mv, b, exact, int_flat = _poisson(0.05)
    op = LinearOperator.from_matvec(A_mv, shape=(b.shape[0], b.shape[0]))
    obs = jnp.asarray(exact)

    def loss(scale):
        applier = jno.precond.gmg().materialize(PrecondContext(op, grid=grid))
        sol = jno.solve.gmres(maxiter=50)(op, scale * b, M=applier)
        return jnp.mean((sol - obs) ** 2)

    g = float(jax.grad(loss)(1.3))
    assert np.isfinite(g) and g > 0.0  # scale=1.3 (> true 1.0) → loss increases


def test_gmg_rejects_no_grid():
    op = LinearOperator.from_matvec(lambda v: v, shape=(9, 9))
    with pytest.raises(ValueError, match="structured grid"):
        jno.precond.gmg().materialize(PrecondContext(op))  # no grid, no fem


def test_gmg_rejects_uncoarsenable_grid():
    """A grid with an odd cell count can't be halved → gmg() raises (the auto FDM path falls back to GMRES)."""
    grid = {"shape": (26, 26), "spacing": (0.04, 0.04), "origin": (0.0, 0.0)}  # 25 cells (odd)
    op = LinearOperator.from_matvec(lambda v: v, shape=(26 * 26, 26 * 26))
    with pytest.raises(ValueError, match="coarsen"):
        jno.precond.gmg().materialize(PrecondContext(op, grid=grid))
