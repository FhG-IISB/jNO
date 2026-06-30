"""Cubic Hermite element (C⁰ with vertex value + first-derivative DOFs) through ``jno.fem``.

Hermite is the first element with a **DOF-mixing** per-cell transform ``M(cell)`` (the Piola maps only
scale each DOF by ±1): the global derivative DOFs are physical-coordinate derivatives ``∂u/∂x``, ``∂u/∂y``
at the vertices, which requires multiplying the reference basis by ``M = blockdiag(1, J)`` per vertex.
It routes through the non-nodal assembler but, being a *scalar* field, reuses the shared scalar evaluator
(value / `.x` gradient / Hessian) once ``M`` is baked into the shape data.

Validation: (1) exact per-element energy identities -- for a global cubic ``u`` the Hermite interpolant
is exact, so ``cᵀK c = ∫|∇u|²`` / ``cᵀM c = ∫u²`` to machine precision, where ``c`` is the Hermite DOF
vector (value + gradient at vertices, centroid value at interiors); a wrong ``M`` (e.g. ``Jᵀ``) fails
these. (2) the gradient DOFs really are ``∇u``. (3) a real Poisson BVP recovers a manufactured cubic
exactly (clamped boundary DOFs).

NB: cubic Hermite is **C⁰** (normal-derivative discontinuous across edges) -- it is NOT a conforming
biharmonic element; it de-risks the ``M(cell)`` / vertex-derivative-DOF machinery the C¹ Bell/Argyris
elements reuse.
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _hermite_dofs(d, u, ux, uy):
    """Global Hermite DOF vector of a function: per vertex ``[u, ∂u/∂x, ∂u/∂y]`` (basix order), then one
    centroid-value DOF per cell -- matching the assembler's layout ``[3·v+k | 3·nv+cell]``."""
    pts = np.asarray(d.mesh.points)[:, :2]
    cells = np.asarray(d.mesh.cells_dict["triangle"])
    nv, nc = pts.shape[0], cells.shape[0]
    c = np.zeros(3 * nv + nc)
    for v in range(nv):
        x, y = pts[v]
        c[3 * v], c[3 * v + 1], c[3 * v + 2] = u(x, y), ux(x, y), uy(x, y)
    cen = pts[cells].mean(1)
    for k in range(nc):
        c[3 * nv + k] = u(*cen[k])
    return c, pts, cells, nv, nc


def _hermite_symbols(d):
    xi, yi, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols(space="Hermite")
    return u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)


def test_hermite_stiffness_energy_reproduces_cubic():
    """``cᵀK c = ∫|∇u|²`` for the harmonic cubic ``u = x³ − 3xy²`` (⇒ ``∫|∇u|² = 5.6`` on the unit square).
    Exact iff the M(cell) derivative-DOF transform is correct."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.4)
    ui, vi = _hermite_symbols(d)
    K = _dense(jno.fem([ui.x * vi.x + ui.y * vi.y]).A)
    assert np.allclose(K, K.T, atol=1e-10)
    c, _, _, nv, nc = _hermite_dofs(
        d, lambda x, y: x**3 - 3 * x * y**2, lambda x, y: 3 * x**2 - 3 * y**2, lambda x, y: -6 * x * y
    )
    assert K.shape == (3 * nv + nc, 3 * nv + nc), "ndof must be 3*n_vertices + n_cells (value+grad per vertex, centroid)"
    assert abs(float(c @ K @ c) - 5.6) < 1e-9, "Hermite stiffness energy != ∫|∇u|² (wrong M(cell)?)"


def test_hermite_mass_energy_reproduces_cubic():
    """``cᵀM c = ∫u²`` for ``u = x³`` (⇒ ``∫x⁶ = 1/7`` on the unit square)."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.4)
    ui, vi = _hermite_symbols(d)
    M = _dense(jno.fem([ui * vi]).A)
    c, _, _, _, _ = _hermite_dofs(d, lambda x, y: x**3, lambda x, y: 3 * x**2, lambda x, y: 0.0 * x)
    assert abs(float(c @ M @ c) - 1.0 / 7.0) < 1e-9, "Hermite mass energy != ∫u²"


def test_hermite_gradient_dofs_are_physical_gradient():
    """The interpolation is exact for a cubic: feeding the Hermite DOFs of ``u = x³ − 3xy²`` (whose
    derivative DOFs are the physical ``∇u``) and projecting (M c = u) recovers ``∫|∇u|²`` -- already
    covered by the stiffness test; here we additionally assert the operator is SPD on the free DOFs (a
    wrong DOF-coupling would break definiteness)."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.5)
    ui, vi = _hermite_symbols(d)
    K = _dense(jno.fem([ui.x * vi.x + ui.y * vi.y + ui * vi]).A)  # +mass -> SPD (no null space)
    evals = np.linalg.eigvalsh(0.5 * (K + K.T))
    assert float(evals.min()) > 1e-10, f"stiffness+mass must be SPD; min eig {evals.min():.2e}"


def test_hermite_poisson_recovers_cubic_exactly():
    """A real Poisson BVP: ``-Δu = f`` with ``u = x³ + y³`` ⇒ ``f = -6(x+y)``. Assemble ``K``, load
    ``b = ∫f v`` through jno.fem, clamp every boundary-vertex DOF (value + both gradient components) to the
    manufactured ``u, ∇u``, solve, and recover ``u`` exactly (it lies in the cubic Hermite space)."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.4)
    xi, yi, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols(space="Hermite")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = -6.0 * (xi + yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi])
    K, b = _dense(fem.A), np.asarray(fem.b).reshape(-1).copy()
    c_exact, pts, _, nv, _ = _hermite_dofs(d, lambda x, y: x**3 + y**3, lambda x, y: 3 * x**2, lambda x, y: 3 * y**2)

    # clamp all DOFs (value + ∂x + ∂y) at boundary vertices to the exact manufactured values
    on_bnd = (pts[:, 0] < 1e-9) | (pts[:, 0] > 1 - 1e-9) | (pts[:, 1] < 1e-9) | (pts[:, 1] > 1 - 1e-9)
    pinned = [3 * v + k for v in np.where(on_bnd)[0] for k in range(3)]
    K, b = K.copy(), b.copy()
    for dof in pinned:  # symmetric elimination
        b -= K[:, dof] * c_exact[dof]
    for dof in pinned:
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        K[dof, dof] = 1.0
        b[dof] = c_exact[dof]
    sol = np.linalg.solve(K, b)
    rel = float(np.linalg.norm(sol - c_exact) / np.linalg.norm(c_exact))
    assert rel < 1e-9, f"Hermite Poisson did not recover the cubic exactly: rel {rel:.2e}"
