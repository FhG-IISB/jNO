"""Heterogeneous domain decomposition — Stage 0: the coupling-mechanism gate.

The design target (see ``plans/heterogeneous-domain-decomposition.md``) is to solve part of a domain
with one solver and the rest with another, composed through ``jno.core``. Before any driver/API is
built, this test pins down the load-bearing claim with a **throwaway, hand-rolled overlapping Schwarz
loop**: on one shared mesh, two overlapping regions solved by finite differences and coupled only by
Dirichlet exchange on the overlap must reproduce the **monolithic single-mesh solve** to solver
tolerance. That "monolithic-equivalence" is what proves the coupling is real rather than plausible; the
future ``jno.core`` coupled-solve mode has to keep this green.

Run with x64 (the FD solve accumulates in float64)."""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

import jno  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _region_mask(points, geometry):
    """Boolean mask of the mesh nodes inside ``geometry`` (a shapely region) — a subdomain's nodes."""
    g = geometry.buffer(1e-9)
    return np.array([g.contains(Point(float(x), float(y))) for x, y in points])


def test_overlapping_schwarz_matches_monolithic():
    """Two overlapping FD regions on one shared mesh, coupled by Dirichlet exchange on the overlap,
    converge to the monolithic single-mesh FD solution (‖Δ‖/‖u‖ ~ 1e-10). -Δu = f, u = 0 on ∂Ω,
    manufactured u* = sin(πx)sin(πy). The two boxes union to the unit square and overlap in x∈[0.4,0.6]."""
    b1, b2 = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)
    d = jno.domain(b1.union(b2), mesh_size=0.05)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    n = p.shape[0]

    bnd = np.zeros(n, bool)
    bnd[np.asarray(d.mesh_connectivity["boundary_indices"])] = True
    in_a, in_b = _region_mask(p, b1), _region_mask(p, b2)
    overlap = in_a & in_b

    f = jnp.asarray(2 * np.pi**2 * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    lap = lambda u: jno.fdm.laplacian(u, d, method="cotangent")  # noqa: E731
    newton = jno.solve.newton()

    def solve_pinned(mask, vals):  # solve -Δu = f with `mask` nodes pinned to `vals`
        mask_j, vals_j = jnp.asarray(mask), jnp.asarray(vals)
        return np.asarray(newton(lambda u: jnp.where(mask_j, u - vals_j, -lap(u) - f), jnp.zeros(n)))

    # monolithic reference: the whole domain in one solve
    u_mono = solve_pinned(bnd, np.zeros(n))
    assert float(np.linalg.norm(u_mono - exact) / np.linalg.norm(exact)) < 1e-2  # it's a real solution

    # hand-rolled overlapping Schwarz: each region pins its complement to the neighbour's current field
    u_a = u_b = np.zeros(n)
    iters, jump = 0, np.inf
    for iters in range(1, 51):
        u_a = solve_pinned((~in_a) | bnd, np.where(bnd, 0.0, u_b))  # A: complement ← B
        u_b = solve_pinned((~in_b) | bnd, np.where(bnd, 0.0, u_a))  # B: complement ← A
        jump = float(np.max(np.abs(u_a[overlap] - u_b[overlap])))
        if jump < 1e-10:
            break
    u_dd = np.where(in_a, u_a, u_b)  # combine (the two agree on the overlap)

    assert jump < 1e-10, f"Schwarz did not converge on the overlap (jump={jump:.2e})"
    equiv = float(np.linalg.norm(u_dd - u_mono) / np.linalg.norm(u_mono))
    assert equiv < 1e-8, f"coupled solution must match the monolithic solve, got {equiv:.2e} in {iters} iters"


def test_heterogeneous_fem_fdm_coupling():
    """Stage 1: a **FEM** region and an **FDM** region on one shared mesh, coupled by overlapping
    Schwarz (Dirichlet exchange), converge to the correct solution — the first *heterogeneous* coupling
    (two different discretizations). -Δu = f, u = 0 on ∂Ω, u* = sin(πx)sin(πy); FDM on box1, FEM on box2.

    Note: heterogeneous Dirichlet–Dirichlet Schwarz converges *slower* than the homogeneous FDM+FDM case
    above (the operators differ at the interface) — the convergence-rate knob the design flags as a
    research risk (optimized/Robin transmission is the later mitigation). The gate here is **correctness**
    (matches the analytic field to discretization accuracy) plus convergence below discretization error."""
    import jax.numpy as jnp

    import jno.jnp_ops as jnn

    b1, b2 = box(0.0, 0.0, 0.6, 1.0), box(0.4, 0.0, 1.0, 1.0)
    d = jno.domain(b1.union(b2), mesh_size=0.05)  # union = unit square, overlap x∈[0.4,0.6]
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    n = p.shape[0]
    bnd = np.zeros(n, bool)
    bnd[np.asarray(d.mesh_connectivity["boundary_indices"])] = True
    in_a, in_b = _region_mask(p, b1), _region_mask(p, b2)  # FDM region, FEM region
    overlap = in_a & in_b
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    f = jnp.asarray(2 * np.pi**2 * exact)

    # FEM engine: raw stiffness + load assembled on the whole mesh (subdomain BCs pinned per iteration)
    xi, yi, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fn = 2 * np.pi**2 * jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - fn * vi])  # weak form of -Δu = f, no BC
    a_fem, b_fem = jnp.asarray(fem.A), jnp.asarray(fem.b).reshape(-1)

    def fem_pinned(mask, vals):  # FEM solve with `mask` nodes pinned to `vals` (row-replacement)
        m, v = jnp.asarray(mask), jnp.asarray(vals)
        a = jnp.where(m[:, None], 0.0, a_fem)
        a = a.at[jnp.arange(n), jnp.arange(n)].set(jnp.where(m, 1.0, jnp.diag(a)))
        return np.asarray(jnp.linalg.solve(a, jnp.where(m, v, b_fem)))

    lap = lambda u: jno.fdm.laplacian(u, d, method="cotangent")  # noqa: E731
    newton = jno.solve.newton()

    def fdm_pinned(mask, vals):
        m, v = jnp.asarray(mask), jnp.asarray(vals)
        return np.asarray(newton(lambda u: jnp.where(m, u - v, -lap(u) - f), jnp.zeros(n)))

    u_a = u_b = np.zeros(n)
    iters, jump = 0, np.inf
    for iters in range(1, 81):
        u_a = fdm_pinned((~in_a) | bnd, np.where(bnd, 0.0, u_b))  # FDM on region A
        u_b = fem_pinned((~in_b) | bnd, np.where(bnd, 0.0, u_a))  # FEM on region B
        jump = float(np.max(np.abs(u_a[overlap] - u_b[overlap])))
        if jump < 1e-3:
            break
    u_dd = np.where(in_a, u_a, u_b)

    assert jump < 1e-3, f"heterogeneous Schwarz did not converge below discretization (jump={jump:.2e})"
    rel = float(np.linalg.norm(u_dd - exact) / np.linalg.norm(exact))
    assert rel < 5e-3, f"coupled FEM+FDM solution must match the analytic field, got rel-L2={rel:.2e}"
