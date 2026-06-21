"""Non-nodal (RT, N1E) elements through the ``jno.fem`` weak-form DSL.

`jno.fem([... space="RT"/"N1E" ...])` routes to the native push-forward assembler
(`fem_nonnodal.assemble_fem_nonnodal`), reusing the shared integrand evaluator's
space-guarded branches. RT (H(div), contravariant Piola) and N1E (H(curl), covariant
Piola) share the edge topology and DOF map; both recover a constant field exactly
(constants lie in RT0 / N1E0), which also pins the assembly edge-orientation sign.
Dense solves run on host (GPU-memory independent).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("feax", reason="feax required for jno.fem")
pytest.importorskip("pygmsh", reason="pygmsh required for 2D meshing")
pytest.importorskip("basix", reason="basix required for RT tabulation")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_nonnodal import (  # noqa: E402
    assemble_mixed_poisson_rt,
    n1e_field_at_centroids,
    rt_flux_at_centroids,
)
from jno.utils.solver.fem_topology import build_edge_topology  # noqa: E402

inner, grad, trace, sin = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.sin


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))


def _rt_domain(mesh_size=0.4):
    d = jno.domain(box(0, 0, 1, 1), mesh_size=mesh_size)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    xi, yi, _ = d.variable("interior", split=True)
    return d, u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)


def _mesh(d):
    return np.asarray(d.mesh.points)[:, :2], np.asarray(d.mesh.cells_dict["triangle"])


def test_rt_mass_matrix_via_dsl_matches_direct_assembler():
    d, ui, vi = _rt_domain()
    A = _dense(jno.fem([inner(ui, vi)]).A)  # residual inner(u,v) -> A = RT mass, b = 0
    pts, cells = _mesh(d)
    A_dir, _, top, _ = assemble_mixed_poisson_rt(pts, cells, lambda x, y: 0.0 * x)
    M = np.asarray(A_dir)[: top.n_edges, : top.n_edges]
    np.testing.assert_allclose(A, M, atol=1e-12)
    np.testing.assert_allclose(A, A.T, atol=1e-12)  # RT mass is symmetric


def test_rt_projection_of_constant_is_exact():
    # constant g=(1,0) lies in RT0 -> the L2 projection recovers it exactly.
    d, ui, vi = _rt_domain()
    fem = jno.fem([inner(ui, vi) - vi[0]])  # residual ∫u·v - ∫g·v, g=(1,0)
    A, b = _dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1)
    uu = np.linalg.solve(A, b)
    pts, cells = _mesh(d)
    flux = np.asarray(rt_flux_at_centroids(pts, cells, build_edge_topology(cells), jnp.asarray(uu)))
    np.testing.assert_allclose(flux, np.tile([1.0, 0.0], (flux.shape[0], 1)), atol=1e-10)


def test_fem_offsets_expose_nonnodal_block_layout():
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.4)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    p, q = d.fem_symbols(names=("p", "q"), space="P0")
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi, pp, qq = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    divu, divv = trace(grad(ui, [xi, yi])), trace(grad(vi, [xi, yi]))
    f = 2 * jnp.pi**2 * sin(jnp.pi * xi) * sin(jnp.pi * yi)
    pts, cells = _mesh(d)
    ne = build_edge_topology(cells).n_edges
    # mixed RT-P0: [0, n_edges, n_edges + n_cells]
    fem = jno.fem([inner(ui, vi) - pp * divv, qq * divu - f * qq], quad_degree=4)
    assert fem.offsets == [0, ne, ne + cells.shape[0]]
    # single RT field: [0, n_edges]
    assert jno.fem([inner(ui, vi)]).offsets == [0, ne]


def test_rt_normal_flux_bc_pins_boundary_dofs():
    # essential u·n = g (constant) pins each boundary edge DOF to -sign_topo * g * |edge| (locked sign).
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.5)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
    ui, vi, ub = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), u.bind(x=xb, y=yb)
    g = 1.5
    fem = jno.fem([inner(ui, vi), ub[0] * nx + ub[1] * ny - g])  # mass system + u·n = g
    sol = np.linalg.solve(_dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1))
    pts, cells = _mesh(d)
    top = build_edge_topology(cells)
    counts = np.bincount(np.asarray(top.cell_edges).reshape(-1), minlength=top.n_edges)
    loc = {int(top.cell_edges[c, k]): (c, k) for c in range(cells.shape[0]) for k in range(3)}
    pinned = 0
    for e, (c, k) in loc.items():
        if counts[e] != 1:  # boundary edges are single-use
            continue
        va, vb = top.edge_vertices[e]
        length = float(np.linalg.norm(pts[vb] - pts[va]))
        np.testing.assert_allclose(sol[e], -int(top.cell_edge_signs[c, k]) * g * length, atol=1e-10)
        pinned += 1
    assert pinned == 8  # the unit-square boundary at mesh_size 0.5


def test_rt_normal_flux_bc_varying_g():
    # u·n = g(x,y) varying: each boundary DOF == -sign * ∫_edge g ds (linear g is midpoint-exact).
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.5)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
    ui, vi, ub = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), u.bind(x=xb, y=yb)
    fem = jno.fem([inner(ui, vi), ub[0] * nx + ub[1] * ny - (xb + 2.0 * yb)])  # u·n = x + 2y
    sol = np.linalg.solve(_dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1))
    pts, cells = _mesh(d)
    top = build_edge_topology(cells)
    counts = np.bincount(np.asarray(top.cell_edges).reshape(-1), minlength=top.n_edges)
    loc = {int(top.cell_edges[c, k]): (c, k) for c in range(cells.shape[0]) for k in range(3)}
    for e, (c, k) in loc.items():
        if counts[e] != 1:
            continue
        va, vb = top.edge_vertices[e]
        pa, pb = pts[va], pts[vb]
        length = float(np.linalg.norm(pb - pa))
        mid = (pa + pb) / 2
        np.testing.assert_allclose(sol[e], -int(top.cell_edge_signs[c, k]) * length * (mid[0] + 2.0 * mid[1]), atol=1e-10)


def test_natural_pressure_bc_mixed_poisson_exact():
    # p=x -> u=-grad p=(-1,0) (exact in RT0), f=0; natural pressure BC p_D=x via the weak term x*(v·n).
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.25)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    p, q = d.fem_symbols(names=("p", "q"), space="P0")
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
    ui, vi, pp, qq = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    vb = v.bind(x=xb, y=yb)
    divu, divv = trace(grad(ui, [xi, yi])), trace(grad(vi, [xi, yi]))
    fem = jno.fem([inner(ui, vi) - pp * divv, qq * divu, xb * (vb[0] * nx + vb[1] * ny)], quad_degree=4)
    off = fem.offsets
    sol = np.linalg.solve(_dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1))
    pts, cells = _mesh(d)
    cent = pts[cells].mean(1)
    flux = np.asarray(rt_flux_at_centroids(pts, cells, build_edge_topology(cells), jnp.asarray(sol[off[0] : off[1]])))
    np.testing.assert_allclose(flux, np.tile([-1.0, 0.0], (flux.shape[0], 1)), atol=1e-10)  # u exact in RT0
    np.testing.assert_allclose(sol[off[1] : off[2]], cent[:, 0], atol=1e-10)  # p = x at centroids


def test_natural_pressure_bc_uses_true_edge_average_not_midpoint():
    # Discriminating quadrature test. The natural term is ∮ p_D (v·n) ds; since the RT0 normal trace is
    # 1/L it reduces to the edge *average* ⟨p_D⟩. A NONLINEAR p_D = x² makes ⟨p_D⟩ ≠ midpoint(p_D) on
    # horizontal boundary edges, so a midpoint shortcut would fail this where the linear p_D=x test cannot.
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.5)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    p, q = d.fem_symbols(names=("p", "q"), space="P0")
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
    ui, vi, pp, qq = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    vb = v.bind(x=xb, y=yb)
    divu, divv = trace(grad(ui, [xi, yi])), trace(grad(vi, [xi, yi]))
    fem = jno.fem([inner(ui, vi) - pp * divv, qq * divu, (xb * xb) * (vb[0] * nx + vb[1] * ny)], quad_degree=4)
    b = np.asarray(jnp.asarray(fem.b)).reshape(-1)
    off = fem.offsets
    pts, cells = _mesh(d)
    top = build_edge_topology(cells)
    counts = np.bincount(np.asarray(top.cell_edges).reshape(-1), minlength=top.n_edges)
    loc = {int(top.cell_edges[c, k]): (c, k) for c in range(cells.shape[0]) for k in range(3)}
    discriminating = 0
    for e, (c, k) in loc.items():
        if counts[e] != 1:  # boundary edges are single-use
            continue
        a, bx = pts[top.edge_vertices[e][0]][0], pts[top.edge_vertices[e][1]][0]  # edge endpoint x-coords
        dd = bx - a
        avg = a * a + a * dd + dd * dd / 3.0  # ∫₀¹ (a + t·dd)² dt — the true edge average of x²
        sign = int(top.cell_edge_signs[c, k])
        np.testing.assert_allclose(b[off[0] + e], sign * avg, atol=1e-12)  # b = sign · ⟨x²⟩_edge
        if abs(dd) > 1e-9:  # non-vertical edge: average differs from the midpoint shortcut by dd²/12
            assert abs(sign * avg - sign * ((a + bx) / 2.0) ** 2) > 1e-3  # would catch a midpoint impl
            discriminating += 1
    assert discriminating > 0  # the unit-square's horizontal boundary edges exercise the nonlinear case


def test_mixed_poisson_rt_p0_via_dsl_matches_direct_assembler():
    # Full RT-P0 mixed Poisson written through jno.fem must assemble the SAME (A, b) as the proven
    # direct assembler (which is convergence-tested in test_fem_nonnodal). div = trace(grad(.)).
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.3)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    p, q = d.fem_symbols(names=("p", "q"), space="P0")
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi, pp, qq = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    divu, divv = trace(grad(ui, [xi, yi])), trace(grad(vi, [xi, yi]))
    f = 2 * jnp.pi**2 * sin(jnp.pi * xi) * sin(jnp.pi * yi)
    fem = jno.fem([inner(ui, vi) - pp * divv, qq * divu - f * qq], quad_degree=4)
    A, b = _dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1)

    pts, cells = _mesh(d)
    src = lambda x, y: 2 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)  # noqa: E731
    A_dir, b_dir, _, _ = assemble_mixed_poisson_rt(pts, cells, src, quad_degree=4)
    np.testing.assert_allclose(A, np.asarray(A_dir), atol=1e-11)
    np.testing.assert_allclose(b, np.asarray(b_dir), atol=1e-11)


def _n1e_domain(mesh_size=0.4):
    d = jno.domain(box(0, 0, 1, 1), mesh_size=mesh_size)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="N1E")
    xi, yi, _ = d.variable("interior", split=True)
    return d, u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)


def test_n1e_mass_matrix_symmetric_positive_definite():
    # The H(curl) mass matrix ∫u·v on N1E: symmetric and positive-definite (the curl-curl problem's
    # coercive `+u` block). Same edge-DOF layout as RT (one DOF per global edge).
    d, ui, vi = _n1e_domain()
    A = _dense(jno.fem([inner(ui, vi)]).A)
    pts, cells = _mesh(d)
    ne = build_edge_topology(cells).n_edges
    assert A.shape == (ne, ne)
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(A).min()) > 0  # positive definite


def test_n1e_projection_of_constant_is_exact():
    # a constant g=(2,-1) lies in N1E0 -> the L2 projection recovers it exactly. This also pins the
    # assembly edge-orientation sign: a wrong per-edge sign cannot reproduce a constant across cells.
    d, ui, vi = _n1e_domain()
    fem = jno.fem([inner(ui, vi) - (2.0 * vi[0] - vi[1])])  # residual ∫u·v - ∫g·v, g=(2,-1)
    A, b = _dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1)
    uu = np.linalg.solve(A, b)
    pts, cells = _mesh(d)
    field = np.asarray(n1e_field_at_centroids(pts, cells, build_edge_topology(cells), jnp.asarray(uu)))
    np.testing.assert_allclose(field, np.tile([2.0, -1.0], (field.shape[0], 1)), atol=1e-10)


def test_n1e_fem_offsets_single_field():
    d, ui, vi = _n1e_domain()
    pts, cells = _mesh(d)
    ne = build_edge_topology(cells).n_edges
    assert jno.fem([inner(ui, vi)]).offsets == [0, ne]  # single edge-DOF field
