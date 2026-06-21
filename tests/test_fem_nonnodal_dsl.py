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


def test_rt_div_via_view_matches_trace_grad():
    # The `.div()` view sugar builds component-first `u[i].d(v)`; for a non-nodal field this now lowers
    # via the whole-field physical gradient and must assemble *identically* to the proven trace(grad(.))
    # idiom -- an exact anchor for the component-gradient evaluator branch.
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.5)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    p, q = d.fem_symbols(names=("p", "q"), space="P0")
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi, pp, qq = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    A_view = _dense(jno.fem([inner(ui, vi) - pp * v.vector.div(xi, yi), qq * u.vector.div(xi, yi)]).A)
    A_trace = _dense(jno.fem([inner(ui, vi) - pp * trace(grad(vi, [xi, yi])), qq * trace(grad(ui, [xi, yi]))]).A)
    np.testing.assert_allclose(A_view, A_trace, atol=1e-12)


def test_n1e_curl_curl_form_symmetric_pd_and_bilinear_exact():
    # H(curl) curl-curl through the `.curl()` view sugar. With the coercive +mass the operator is
    # symmetric positive-definite; the pure curl-curl block reproduces ∫(curl u)² exactly for a field in
    # N1E0 (u=(-y,x), curl=2 -> ∫(curl u)² = 2²·area = 4 on the unit square).
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.4)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="N1E")
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    cu, cv = u.vector.curl(xi, yi), v.vector.curl(xi, yi)
    pts, cells = _mesh(d)
    ne = build_edge_topology(cells).n_edges
    A = _dense(jno.fem([inner(ui, vi) + cu * cv]).A)  # curl-curl + mass -> coercive on all of H(curl)
    assert A.shape == (ne, ne)
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(A).min()) > 0  # positive definite (the +u block; pure curl-curl is singular)
    # exact bilinear form: project u=(-y,x) (in N1E0) for its DOFs, then uᵀ K u == ∫(curl u)² = 4
    M = _dense(jno.fem([inner(ui, vi)]).A)
    b = np.asarray(jnp.asarray(jno.fem([inner(ui, vi) - (-yi * vi[0] + xi * vi[1])]).b)).reshape(-1)
    u_dof = np.linalg.solve(M, b)
    K = _dense(jno.fem([cu * cv]).A)  # pure curl-curl (no mass)
    np.testing.assert_allclose(u_dof @ K @ u_dof, 4.0, atol=1e-9)


def test_bind_aware_div_curl_match_explicit_form():
    # u.bind(x=xi, y=yi).curl() / .div() pull the coords from the bind, so they assemble identically to
    # the explicit u.vector.curl(xi, yi) / .div(xi, yi) -- the ergonomic no-arg form for bound FEM symbols.
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.4)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="N1E")
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    A_noarg = _dense(jno.fem([inner(ui, vi) + ui.curl() * vi.curl()]).A)
    A_expl = _dense(jno.fem([inner(ui, vi) + u.vector.curl(xi, yi) * v.vector.curl(xi, yi)]).A)
    np.testing.assert_allclose(A_noarg, A_expl, atol=1e-13)


def _tangential_sign(pts, top, e, c):
    # geometric N1E tangential pin sign: orientation of the +90° rotation of the edge vector relative to
    # the outward direction (away from the opposite vertex of the single incident cell). Mirrors _apply_flux_bcs.
    va, vb = (int(x) for x in top.edge_vertices[e])
    pa, pb = pts[va], pts[vb]
    vc = (set(int(x) for ek in top.cell_edges[c] for x in top.edge_vertices[ek]) - {va, vb}).pop()
    rot90 = np.array([-(pb[1] - pa[1]), pb[0] - pa[0]])
    return 1.0 if float(np.dot(rot90, 0.5 * (pa + pb) - pts[vc])) > 0 else -1.0


def test_n1e_tangential_bc_pins_boundary_dofs():
    # Essential tangential trace u×n = g (constant) on N1E pins each boundary edge DOF to sgn·g·|edge|.
    # Written via the outward normal (domain.variable(..., normals=True)): u[0]*ny - u[1]*nx - g, the same
    # `dot(u, normal-data) - g` gesture as the RT u·n BC -- no new API. The sign is the geometric tangential
    # reconciliation (verified against the exact projection of a constant field).
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.5)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="N1E")
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
    ui, vi, ub = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), u.bind(x=xb, y=yb)
    g = 0.7
    fem = jno.fem([inner(ui, vi), ub[0] * ny - ub[1] * nx - g])  # mass + tangential trace u×n = g
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
        np.testing.assert_allclose(sol[e], _tangential_sign(pts, top, e, c) * g * length, atol=1e-10)
        pinned += 1
    assert pinned == 8  # the unit-square boundary at mesh_size 0.5


def test_n1e_curl_curl_dirichlet_converges():
    # The canonical H(curl) validation: curl curl u + u = f with the homogeneous tangential BC u×n = 0.
    # Manufactured u=(sin πy, sin πx) has u×n=0 on the unit square and curl curl u = π²u, so f=(π²+1)u.
    # Lowest-order N1E is O(h) in L²: check the centroid error drops at ~rate 1 under refinement.
    PI = float(np.pi)

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="N1E")
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
        ui, vi, ub = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), u.bind(x=xb, y=yb)
        fac = PI**2 + 1.0
        fx, fy = fac * sin(PI * yi), fac * sin(PI * xi)  # f = (π²+1)·u_exact
        fem = jno.fem([inner(ui, vi) + ui.curl() * vi.curl() - (fx * vi[0] + fy * vi[1]), ub[0] * ny - ub[1] * nx - 0.0])
        sol = np.linalg.solve(_dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1))
        pts, cells = _mesh(d)
        fc = np.asarray(n1e_field_at_centroids(pts, cells, build_edge_topology(cells), jnp.asarray(sol)))
        cent = pts[cells].mean(1)
        exact = np.stack([np.sin(PI * cent[:, 1]), np.sin(PI * cent[:, 0])], -1)
        return float(np.sqrt(np.mean(np.sum((fc - exact) ** 2, 1))))

    e = [solve(h) for h in (0.5, 0.25, 0.125)]
    assert e[2] < e[1] < e[0], f"not monotone under refinement: {e}"
    assert np.log2(e[1] / e[2]) > 0.85, f"curl-curl Dirichlet rate {np.log2(e[1] / e[2]):.2f} below O(h)"


def test_rt_mixed_bc_natural_pressure_and_essential_flux():
    # MIXED boundary conditions in one problem -- the stress test that everything still registers when
    # kinds are combined across regions. Natural pressure p=x on left+right AND essential flux u·n=0 on
    # top+bottom (built-in box edge tags carry normals). The classifier must route each kind to its own
    # region; exact recovery of u=(-1,0)∈RT0, p=x proves both register and target correctly. Extremes
    # exercised: homogeneous flux g=0 (top/bottom) and a zero natural value p_D=0 (the left edge, x=0).
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.25)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    p, q = d.fem_symbols(names=("p", "q"), space="P0")
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _, nlx, nly = d.variable("left", normals=True, split=True)
    xr, yr, _, nrx, nry = d.variable("right", normals=True, split=True)
    xt, yt, _, ntx, nty = d.variable("top", normals=True, split=True)
    xbo, ybo, _, nbx, nby = d.variable("bottom", normals=True, split=True)
    ui, vi, pp, qq = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    vl, vr, ut, ubo = v.bind(x=xl, y=yl), v.bind(x=xr, y=yr), u.bind(x=xt, y=yt), u.bind(x=xbo, y=ybo)
    divu, divv = trace(grad(ui, [xi, yi])), trace(grad(vi, [xi, yi]))
    fem = jno.fem(
        [
            inner(ui, vi) - pp * divv,
            qq * divu,
            xl * (vl[0] * nlx + vl[1] * nly),  # natural pressure p_D=x on left (p_D=0 there)
            xr * (vr[0] * nrx + vr[1] * nry),  # natural pressure p_D=x on right (p_D=1)
            ut[0] * ntx + ut[1] * nty - 0.0,  # essential flux u·n=0 on top
            ubo[0] * nbx + ubo[1] * nby - 0.0,  # essential flux u·n=0 on bottom
        ],
        quad_degree=4,
    )
    off = fem.offsets
    sol = np.linalg.solve(_dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1))
    pts, cells = _mesh(d)
    cent = pts[cells].mean(1)
    flux = np.asarray(rt_flux_at_centroids(pts, cells, build_edge_topology(cells), jnp.asarray(sol[off[0] : off[1]])))
    np.testing.assert_allclose(flux, np.tile([-1.0, 0.0], (flux.shape[0], 1)), atol=1e-9)  # u exact in RT0
    np.testing.assert_allclose(sol[off[1] : off[2]], cent[:, 0], atol=1e-9)  # p = x at centroids


def test_n1e_tangential_bc_distinct_values_on_two_subregions():
    # Two tangential traces with DIFFERENT values on two edge tags must register independently: left edges
    # pin to g_left, right edges to g_right, and the untagged top/bottom stay natural. Per-region targeting
    # for the N1E essential trace, with g_right<0 an extreme (sign of the prescribed value).
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.5)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="N1E")
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _, nlx, nly = d.variable("left", normals=True, split=True)
    xr, yr, _, nrx, nry = d.variable("right", normals=True, split=True)
    ui, vi, ul, ur = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), u.bind(x=xl, y=yl), u.bind(x=xr, y=yr)
    gl, gr = 0.9, -0.4
    fem = jno.fem([inner(ui, vi), ul[0] * nly - ul[1] * nlx - gl, ur[0] * nry - ur[1] * nrx - gr])
    sol = np.linalg.solve(_dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1))
    pts, cells = _mesh(d)
    top = build_edge_topology(cells)
    counts = np.bincount(np.asarray(top.cell_edges).reshape(-1), minlength=top.n_edges)
    loc = {int(top.cell_edges[c, k]): (c, k) for c in range(cells.shape[0]) for k in range(3)}
    nl = nr = 0
    for e, (c, k) in loc.items():
        if counts[e] != 1:  # boundary edges
            continue
        va, vb = top.edge_vertices[e]
        pa, pb = pts[va], pts[vb]
        length = float(np.linalg.norm(pb - pa))
        sgn = _tangential_sign(pts, top, e, c)
        if abs(pa[0]) < 1e-9 and abs(pb[0]) < 1e-9:  # left edge (x=0)
            np.testing.assert_allclose(sol[e], sgn * gl * length, atol=1e-10)
            nl += 1
        elif abs(pa[0] - 1.0) < 1e-9 and abs(pb[0] - 1.0) < 1e-9:  # right edge (x=1)
            np.testing.assert_allclose(sol[e], sgn * gr * length, atol=1e-10)
            nr += 1
    assert nl == 2 and nr == 2  # left & right each split into 2 sub-edges at mesh_size 0.5


def test_rt_mixed_bc_via_user_defined_tags():
    # The mixed-region BC again, but on USER-defined domain.tag regions instead of the built-in box edges
    # -- exercises the tag-normals fix end to end: a pure-boundary tag must carry outward normals for the
    # flux/natural BCs to register. Natural pressure p=x on a custom 'lr', essential flux u·n=0 on 'tb'.
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.25)
    d.tag("lr", lambda x, y: (x < 1e-6) | (x > 1 - 1e-6))
    d.tag("tb", lambda x, y: (y < 1e-6) | (y > 1 - 1e-6))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    p, q = d.fem_symbols(names=("p", "q"), space="P0")
    xi, yi, _ = d.variable("interior", split=True)
    xlr, ylr, _, nlx, nly = d.variable("lr", normals=True, split=True)
    xtb, ytb, _, ntx, nty = d.variable("tb", normals=True, split=True)
    ui, vi, pp, qq = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    vlr, utb = v.bind(x=xlr, y=ylr), u.bind(x=xtb, y=ytb)
    divu, divv = trace(grad(ui, [xi, yi])), trace(grad(vi, [xi, yi]))
    fem = jno.fem(
        [
            inner(ui, vi) - pp * divv,
            qq * divu,
            xlr * (vlr[0] * nlx + vlr[1] * nly),  # natural pressure p_D=x on the custom left+right tag
            utb[0] * ntx + utb[1] * nty - 0.0,  # essential flux u·n=0 on the custom top+bottom tag
        ],
        quad_degree=4,
    )
    off = fem.offsets
    sol = np.linalg.solve(_dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1))
    pts, cells = _mesh(d)
    cent = pts[cells].mean(1)
    flux = np.asarray(rt_flux_at_centroids(pts, cells, build_edge_topology(cells), jnp.asarray(sol[off[0] : off[1]])))
    np.testing.assert_allclose(flux, np.tile([-1.0, 0.0], (flux.shape[0], 1)), atol=1e-9)  # u exact in RT0
    np.testing.assert_allclose(sol[off[1] : off[2]], cent[:, 0], atol=1e-9)  # p = x at centroids
