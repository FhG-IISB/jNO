"""Periodic ties on **unstructured** (non-matching) meshes via interpolatory prolongation.

The structured/conforming case ties slave≡master node-to-node (exact 0/1 ``P``). When the two
periodic faces carry *different* node layouts, a slave with no master node within ``tol`` is tied to
the master *facet* it lands on by node-to-segment interpolation (linear P1 / quadratic P2). The
primary correctness gate is a **patch test**: the prolongation reproduces a constant and a
linear-along-the-face field exactly (P1), a quadratic exactly (P2) — partition of unity + facet
completeness. (Interpolatory master--slave elimination, not full dual-mortar; see
``_periodic_facet_weights``.)

These exercise ``build_periodic_prolongation`` directly (no FEM assembly). x64 so the float64
reduction reproduces the fields exactly.
"""

import numpy as np
import pytest

pytest.importorskip("feax", reason="feax required for the periodic solver utilities")

import jax  # noqa: E402

from jno.utils.solver.feax_utils import build_periodic_prolongation  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _two_faces(n_master, n_slave, *, order=1):
    """Right (master, x=1) and left (slave, x=0) faces with independent node layouts.

    Returns ``(points, tag_indices, facets)``. For ``order==2`` the master face carries edge
    midpoints and the facets are 3-node (a, b, mid)."""
    ym = np.linspace(0.0, 1.0, n_master)
    master = np.column_stack([np.ones_like(ym), ym])
    ys = np.linspace(0.0, 1.0, n_slave)
    slave = np.column_stack([np.zeros_like(ys), ys])
    if order == 1:
        pts = np.vstack([master, slave])
        m_ids = np.arange(n_master)
        s_ids = np.arange(n_master, n_master + n_slave)
        edges = np.column_stack([m_ids[:-1], m_ids[1:]])  # (a, b)
        return pts, {"right": m_ids, "left": s_ids}, {"right": edges}
    # P2: insert master edge midpoints
    mids = 0.5 * (ym[:-1] + ym[1:])
    master_mid = np.column_stack([np.ones_like(mids), mids])
    pts = np.vstack([master, master_mid, slave])
    m_ids = np.arange(n_master)
    mid_ids = np.arange(n_master, n_master + len(mids))
    s_ids = np.arange(n_master + len(mids), len(pts))
    edges = np.column_stack([m_ids[:-1], m_ids[1:], mid_ids])  # (a, b, mid)
    return pts, {"right": m_ids, "left": s_ids}, {"right": edges}


def test_patch_test_p1_reproduces_constant_and_linear():
    """Non-matching faces (5 master vs 9 slave nodes): P-rows are a partition of unity and the
    prolongation reproduces a constant and a linear-in-y field exactly."""
    pts, tags, facets = _two_faces(5, 9, order=1)
    res = build_periodic_prolongation(pts, [("right", "left")], tags, facets=facets)
    P = np.asarray(res["P_node"])
    kept = np.asarray(res["kept_nodes"])

    assert res["n_red"] == 5  # all 9 left nodes eliminated; the 5 right nodes are retained
    assert np.allclose(P.sum(axis=1), 1.0)  # partition of unity on every row

    # constant
    u_red = np.ones(len(kept))
    assert np.allclose(P @ u_red, 1.0)
    # linear in the transverse coord y -> reproduced exactly at every slave node
    for a, b in [(2.0, 0.3), (-1.5, 1.0)]:
        f = lambda y: a * y + b  # noqa: E731
        u_full = P @ f(pts[kept, 1])
        assert np.allclose(u_full, f(pts[:, 1]), atol=1e-10)


def test_patch_test_p2_reproduces_quadratic():
    """With 3-node (a, b, mid) master facets the interpolation reproduces a quadratic exactly."""
    pts, tags, facets = _two_faces(3, 9, order=2)
    res = build_periodic_prolongation(pts, [("right", "left")], tags, facets=facets)
    P = np.asarray(res["P_node"])
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    f = lambda y: 0.7 * y**2 - 0.2 * y + 0.1  # noqa: E731
    u_full = P @ f(pts[kept, 1])
    assert np.allclose(u_full, f(pts[:, 1]), atol=1e-10)


def test_patch_test_3d_triangle_reproduces_linear():
    """3D triangular facets: a slave face (z=0) ties to a master face (z=1) by point-in-triangle
    barycentric interpolation. The prolongation reproduces a constant and a linear-in-(x,y) field
    exactly (partition of unity + barycentric completeness)."""
    # master face z=1: unit square as two triangles over its 4 corners
    master = np.array([[0, 0, 1.0], [1, 0, 1.0], [1, 1, 1.0], [0, 1, 1.0]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])  # local -> these are master node ids 0..3
    # slave face z=0: interior points (fall inside the master triangles)
    slave = np.array([[0.3, 0.2, 0.0], [0.7, 0.6, 0.0], [0.5, 0.5, 0.0], [0.2, 0.8, 0.0], [0.9, 0.1, 0.0]])
    pts = np.vstack([master, slave])
    tags = {"top": np.arange(4), "bot": np.arange(4, 4 + len(slave))}
    res = build_periodic_prolongation(pts, [("top", "bot")], tags, facets={"top": tris})
    P = np.asarray(res["P_node"])
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)  # partition of unity
    # reproduce constant and a linear field a·x + b·y + c exactly at every slave node
    for a, b, c in [(0.0, 0.0, 1.0), (2.0, -1.5, 0.3)]:
        field = a * pts[:, 0] + b * pts[:, 1] + c
        assert np.allclose(P @ field[kept], field, atol=1e-10)


def test_patch_test_3d_triangle_p2_reproduces_quadratic():
    """3D P2 triangular facets (6-node: 3 vertices + 3 edge midpoints) reproduce a quadratic-in-(x,y)
    field exactly via the quadratic-triangle shape functions."""
    # master face z=1: 4 corners + 5 edge midpoints, two 6-node triangles
    c = np.array([[0, 0, 1.0], [1, 0, 1.0], [1, 1, 1.0], [0, 1, 1.0]])  # ids 0..3
    mids = np.array(
        [[0.5, 0, 1.0], [1, 0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 1, 1.0], [0, 0.5, 1.0]]
    )  # m01,m12,m20,m23,m30 -> ids 4..8
    slave = np.array([[0.3, 0.2, 0.0], [0.6, 0.5, 0.0], [0.5, 0.5, 0.0], [0.2, 0.7, 0.0]])  # ids 9..12
    pts = np.vstack([c, mids, slave])
    tris = np.array([[0, 1, 2, 4, 5, 6], [0, 2, 3, 6, 7, 8]])  # (a,b,c, mab,mbc,mca)
    tags = {"top": np.arange(9), "bot": np.arange(9, 13)}
    res = build_periodic_prolongation(pts, [("top", "bot")], tags, facets={"top": tris})
    P = np.asarray(res["P_node"])
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    q = lambda x, y: 0.5 * x**2 - 0.3 * y**2 + 0.7 * x * y + x - 0.2 * y + 0.1  # noqa: E731
    field = q(pts[:, 0], pts[:, 1])
    assert np.allclose(P @ field[kept], field, atol=1e-9)


def test_conforming_stays_exact_0_1_permutation():
    """Equal node layout on both faces -> exact node-to-node 0/1 P (no interpolation, no facets)."""
    pts, tags, _ = _two_faces(5, 5, order=1)
    res = build_periodic_prolongation(pts, [("right", "left")], tags)  # no facets needed
    P = np.asarray(res["P_node"])
    assert res["n_red"] == 5
    assert set(np.unique(P).tolist()) <= {0.0, 1.0}  # pure permutation, no fractional weights
    assert np.allclose(P.sum(axis=1), 1.0)


def test_non_matching_without_facets_raises():
    """Non-matching faces and no facet connectivity -> a clear error (can't interpolate)."""
    pts, tags, _ = _two_faces(5, 9, order=1)
    with pytest.raises(ValueError, match="no master facet connectivity"):
        build_periodic_prolongation(pts, [("right", "left")], tags)


def test_periodic_poisson_2d_p1_nonconforming():
    """End-to-end: steady Poisson ``-Δu = f`` periodic in x (``u(left) - u(right)`` tie) and Dirichlet
    in y, on a **non-conforming** mesh (fine left half, coarse right half -> the x=0 / x=1 faces carry
    different node layouts). The manufactured solution has a *nonzero, matching* flux at the periodic
    faces, so the natural (zero-flux) BC would be wrong -- the tie is doing real work. P1 elements."""
    from shapely.geometry import box

    import jno

    pi = np.pi
    dom = jno.domain({"fine": box(0, 0, 0.5, 1), "coarse": box(0.5, 0, 1, 1)}).build_mesh(0.10, sizes={"fine": 0.045})
    # periodic faces are the OPEN edges (corners belong to the Dirichlet top/bottom, not the tie)
    dom.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("bottom", lambda x, y: y < 1e-6)
    dom.tag("top", lambda x, y: y > 1 - 1e-6)
    nl = len(np.asarray(dom.tag_indices["left"]).ravel())
    nr = len(np.asarray(dom.tag_indices["right"]).ravel())
    assert nl != nr, f"faces must be non-matching to exercise interpolation (got {nl} vs {nr})"

    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    hh = jno.np.cos(2 * jno.np.pi * xi) + 0.5 * jno.np.sin(2 * jno.np.pi * xi)
    f = 5 * jno.np.pi**2 * hh * jno.np.sin(jno.np.pi * yi)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y - f * vi,  # -Δu = f
            u(xb, yb) - 0.0,
            u(xt, yt) - 0.0,  # homogeneous Dirichlet on y = 0, 1
            u(xl, yl) - u(xr, yr),  # periodic in x
        ]
    )
    assert fem._periodic is not None, "the u(left)-u(right) tie must be recognised and reduce the system"
    assert fem._periodic["n_red"] < fem._periodic["n_full"], "the tie must eliminate the slave-face DOFs"

    uh = np.asarray(fem.solve())
    pts = np.asarray(fem.points)
    u_exact = (np.cos(2 * pi * pts[:, 0]) + 0.5 * np.sin(2 * pi * pts[:, 0])) * np.sin(pi * pts[:, 1])
    rel = float(np.linalg.norm(uh - u_exact) / np.linalg.norm(u_exact))
    assert rel < 0.05, f"periodic Poisson L2 relative error too large: {rel:.3f}"


def test_periodic_1d_reaction_diffusion():
    """1D periodic is the degenerate (node-to-node) case: ``-u'' + u = f`` on [0,1] with ``u(left) -
    u(right)``. The reaction term makes the all-periodic problem well-posed (no null space); the tie
    is exact (one endpoint eliminated) so the recovered ``u = cos(2πx)`` is near machine-accurate."""
    import jno

    d = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    f = (4 * jno.np.pi**2 + 1) * jno.np.cos(2 * jno.np.pi * xi)
    fem = jno.fem([ui.x * vi.x + ui * vi - f * vi, u(xl) - u(xr)])
    assert fem._periodic is not None and fem._periodic["n_red"] == fem._periodic["n_full"] - 1
    uh = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.points).reshape(-1)
    rel = float(np.linalg.norm(uh - np.cos(2 * np.pi * pts)) / np.linalg.norm(np.cos(2 * np.pi * pts)))
    assert rel < 1e-3, f"1D periodic reaction-diffusion error too large: {rel:.2e}"


def test_tie_to_non_boundary_region_raises():
    """A tie whose region is not a boundary (e.g. an interior tag) is rejected with a clear error."""
    from shapely.geometry import box

    import jno

    dom = jno.domain(box(0, 0, 1, 1)).build_mesh(0.2)
    dom.tag("blob", lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.04)  # interior region (no boundary facets)
    dom.tag("left", lambda x, y: x < 1e-6)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xbl, ybl, _ = dom.variable("blob", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    with pytest.raises(ValueError, match="distinct boundary regions"):
        jno.fem([ui.x * vi.x + ui.y * vi.y - vi, u(xbl, ybl) - u(xl, yl)])


def test_only_plain_minus_is_a_tie():
    """`u(A)+u(B)` (anti-periodic) and `2*u(A)-u(B)` (scaled) must NOT be silently read as a periodic
    tie -- they raise a clear error rather than quietly meaning `u(A)=u(B)`."""
    from shapely.geometry import box

    import jno

    dom = jno.domain(box(0, 0, 1, 1)).build_mesh(0.25)
    dom.tag("left", lambda x, y: x < 1e-6)
    dom.tag("right", lambda x, y: x > 1 - 1e-6)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    weak = ui.x * vi.x + ui.y * vi.y - vi
    with pytest.raises(ValueError, match="must be `u\\(A\\) - u\\(B\\)`"):
        jno.fem([weak, u(xl, yl) + u(xr, yr)])  # anti-periodic, not a tie
    with pytest.raises(ValueError, match="must be `u\\(A\\) - u\\(B\\)`"):
        jno.fem([weak, 2.0 * u(xl, yl) - u(xr, yr)])  # scaled, not a tie


def test_doubly_periodic_reaction_diffusion():
    """Multi-direction periodicity (a doubly-periodic cell): ``-Δu + u = f`` periodic in **both** x
    and y, via two ties ``u(left)-u(right)`` and ``u(bottom)-u(top)``. The reaction term makes the
    all-periodic problem well-posed. This exercises the general transitive corner resolution -- the
    four corners are each a slave in two directions and must all collapse onto one kept master.
    Manufactured ``u = cos(2πx) cos(2πy)``."""
    from shapely.geometry import box

    import jno

    pi = np.pi
    dom = jno.domain(box(0, 0, 1, 1)).build_mesh(0.06)
    for nm, pred in {
        "left": lambda x, y: x < 1e-6,
        "right": lambda x, y: x > 1 - 1e-6,
        "bottom": lambda x, y: y < 1e-6,
        "top": lambda x, y: y > 1 - 1e-6,
    }.items():
        dom.tag(nm, pred)
    # the box triangulation is non-conforming across opposite faces -> this exercises interpolation
    # AND corner resolution together (not just exact node-to-node ties)
    lefty = np.sort(np.asarray(dom.mesh.points)[np.asarray(dom.tag_indices["left"]).ravel(), 1])
    righty = np.sort(np.asarray(dom.mesh.points)[np.asarray(dom.tag_indices["right"]).ravel(), 1])
    assert not (len(lefty) == len(righty) and np.allclose(lefty, righty)), "expected non-conforming faces"
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = (8 * jno.np.pi**2 + 1) * jno.np.cos(2 * jno.np.pi * xi) * jno.np.cos(2 * jno.np.pi * yi)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + ui * vi - f * vi,  # -Δu + u = f
            u(xl, yl) - u(xr, yr),  # periodic in x
            u(xb, yb) - u(xt, yt),  # periodic in y
        ]
    )
    assert fem._periodic is not None and fem._periodic["n_red"] < fem._periodic["n_full"]
    uh = np.asarray(fem.solve())
    pts = np.asarray(fem.points)
    u_exact = np.cos(2 * pi * pts[:, 0]) * np.cos(2 * pi * pts[:, 1])
    rel = float(np.linalg.norm(uh - u_exact) / np.linalg.norm(u_exact))
    assert rel < 0.05, f"doubly-periodic reaction-diffusion L2 relative error too large: {rel:.3f}"

    # the four corners are identified -> they must carry the same solution value
    corner = lambda cx, cy: int(np.argmin(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))  # noqa: E731
    cvals = [uh[corner(cx, cy)] for cx in (0.0, 1.0) for cy in (0.0, 1.0)]
    assert np.allclose(cvals, cvals[0], atol=1e-9), f"the four periodic corners must be identified: {cvals}"


def test_boundary_facets_extractor_p1_p2():
    """The assembly-mesh facet extractor returns 2-node edges for P1 and 3-node (edge+midpoint) for
    P2, with the midpoint at the average of the two endpoint vertices."""
    from shapely.geometry import box

    import jno
    from jno._fem import _boundary_facets
    from jno.utils.solver.fem_native import _get_mesh

    for order, k in [(1, 2), (2, 3)]:
        dom = jno.domain(box(0, 0, 1, 1)).build_mesh(0.4)
        # The assembly mesh the DOFs live on (vertices, plus edge midpoints for P2).
        _, _, pts_f, cells_f = _get_mesh(dom, 2, order)
        pts = np.asarray(pts_f)
        bf = _boundary_facets(pts, np.asarray(cells_f), 2, order)
        assert bf.shape[1] == k, f"order={order} should give {k}-node facets, got {bf.shape}"
        if order == 2:
            assert all(np.allclose(pts[r[2]], 0.5 * (pts[r[0]] + pts[r[1]])) for r in bf), "col2 must be the edge midpoint"


def test_periodic_poisson_2d_p2_nonconforming():
    """Same single-direction periodic Poisson as the P1 case, but with **P2** elements: the periodic
    face now carries midpoint nodes, tied through quadratic edge interpolation."""
    from shapely.geometry import box

    import jno

    pi = np.pi
    dom = jno.domain({"fine": box(0, 0, 0.5, 1), "coarse": box(0.5, 0, 1, 1)}).build_mesh(0.12, sizes={"fine": 0.07})
    dom.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("bottom", lambda x, y: y < 1e-6)
    dom.tag("top", lambda x, y: y > 1 - 1e-6)
    u, phi = dom.fem_symbols(order=2)
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    hh = jno.np.cos(2 * jno.np.pi * xi) + 0.5 * jno.np.sin(2 * jno.np.pi * xi)
    f = 5 * jno.np.pi**2 * hh * jno.np.sin(jno.np.pi * yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0, u(xt, yt) - 0.0, u(xl, yl) - u(xr, yr)])
    assert fem._periodic is not None and fem._periodic["n_red"] < fem._periodic["n_full"]
    uh = np.asarray(fem.solve())
    pts = np.asarray(fem.points)
    u_exact = (np.cos(2 * pi * pts[:, 0]) + 0.5 * np.sin(2 * pi * pts[:, 0])) * np.sin(pi * pts[:, 1])
    rel = float(np.linalg.norm(uh - u_exact) / np.linalg.norm(u_exact))
    assert rel < 0.05, f"P2 periodic Poisson L2 relative error too large: {rel:.3f}"


def test_triply_periodic_3d_cube():
    """3D: a **triply-periodic** unit cube (`-Δu + u = f`, periodic in x, y AND z) -- the full
    corner→edge→face hierarchy. Exercises the transitive `_expand` composition (a master triangle's
    edge/corner node is itself a slave on adjacent faces). Manufactured `u = cos2πx·cos2πy·cos2πz`;
    assert the solve converges and the 8 cube corners are all identified."""
    import jno

    pi = np.pi
    e = 1e-6
    dom = jno.domain(constructor=jno.domain.cube(mesh_size=0.13))
    faces = {
        "xlo": lambda x, y, z: x < e,
        "xhi": lambda x, y, z: x > 1 - e,
        "ylo": lambda x, y, z: y < e,
        "yhi": lambda x, y, z: y > 1 - e,
        "zlo": lambda x, y, z: z < e,
        "zhi": lambda x, y, z: z > 1 - e,
    }
    for nm, p in faces.items():
        dom.tag(nm, p)
    u, phi = dom.fem_symbols()
    xi, yi, zi, _ = dom.variable("interior", split=True)
    g = lambda t: dom.variable(t, split=True)  # noqa: E731
    xlo, ylo, zlo, _ = g("xlo")
    xhi, yhi, zhi, _ = g("xhi")
    xa, ya, za, _ = g("ylo")
    xb, yb, zb, _ = g("yhi")
    xc, yc, zc, _ = g("zlo")
    xd, yd, zd, _ = g("zhi")
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    f = (
        (12 * jno.np.pi**2 + 1)
        * jno.np.cos(2 * jno.np.pi * xi)
        * jno.np.cos(2 * jno.np.pi * yi)
        * jno.np.cos(2 * jno.np.pi * zi)
    )
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + ui.z * vi.z + ui * vi - f * vi,
            u(xlo, ylo, zlo) - u(xhi, yhi, zhi),  # periodic in x
            u(xa, ya, za) - u(xb, yb, zb),  # periodic in y
            u(xc, yc, zc) - u(xd, yd, zd),  # periodic in z
        ]
    )
    assert fem._periodic is not None and fem._periodic["n_red"] < fem._periodic["n_full"]
    uh = np.asarray(fem.solve())
    pts = np.asarray(fem.points)
    # the cube triangulation is non-conforming across opposite faces -> the 3D triangle interpolation
    # AND the _expand composition (shared edge/corner nodes) are genuinely exercised end-to-end
    xlo = pts[pts[:, 0] < 1e-6][:, 1:]
    xhi = pts[pts[:, 0] > 1 - 1e-6][:, 1:]
    xlo, xhi = xlo[np.lexsort(xlo.T)], xhi[np.lexsort(xhi.T)]
    assert not (xlo.shape == xhi.shape and np.allclose(xlo, xhi)), "expected non-conforming cube faces"
    u_exact = np.cos(2 * pi * pts[:, 0]) * np.cos(2 * pi * pts[:, 1]) * np.cos(2 * pi * pts[:, 2])
    rel = float(np.linalg.norm(uh - u_exact) / np.linalg.norm(u_exact))
    assert rel < 0.12, f"3D triply-periodic L2 relative error too large: {rel:.3f}"
    cv = [
        uh[int(np.argmin(np.sum((pts - np.array([cx, cy, cz])) ** 2, axis=1)))]
        for cx in (0.0, 1.0)
        for cy in (0.0, 1.0)
        for cz in (0.0, 1.0)
    ]
    assert np.allclose(cv, cv[0], atol=1e-9), f"the 8 cube corners must be identified: {cv}"


def test_triply_periodic_3d_cube_p2():
    """3D triply-periodic cube with **P2** (TET10) elements -- face nodes carry edge midpoints tied
    through quadratic-triangle interpolation. P2 is far more accurate than P1 even on a coarse mesh."""
    import jno

    pi = np.pi
    e = 1e-6
    dom = jno.domain(constructor=jno.domain.cube(mesh_size=0.28))
    for nm, p in {
        "xlo": lambda x, y, z: x < e,
        "xhi": lambda x, y, z: x > 1 - e,
        "ylo": lambda x, y, z: y < e,
        "yhi": lambda x, y, z: y > 1 - e,
        "zlo": lambda x, y, z: z < e,
        "zhi": lambda x, y, z: z > 1 - e,
    }.items():
        dom.tag(nm, p)
    u, phi = dom.fem_symbols(order=2)
    xi, yi, zi, _ = dom.variable("interior", split=True)
    g = lambda t: dom.variable(t, split=True)  # noqa: E731
    xlo, ylo, zlo, _ = g("xlo")
    xhi, yhi, zhi, _ = g("xhi")
    xa, ya, za, _ = g("ylo")
    xb, yb, zb, _ = g("yhi")
    xc, yc, zc, _ = g("zlo")
    xd, yd, zd, _ = g("zhi")
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    f = (
        (12 * jno.np.pi**2 + 1)
        * jno.np.cos(2 * jno.np.pi * xi)
        * jno.np.cos(2 * jno.np.pi * yi)
        * jno.np.cos(2 * jno.np.pi * zi)
    )
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + ui.z * vi.z + ui * vi - f * vi,
            u(xlo, ylo, zlo) - u(xhi, yhi, zhi),
            u(xa, ya, za) - u(xb, yb, zb),
            u(xc, yc, zc) - u(xd, yd, zd),
        ]
    )
    assert fem._periodic is not None and fem._periodic["n_red"] < fem._periodic["n_full"]
    uh = np.asarray(fem.solve())
    pts = np.asarray(fem.points)
    u_exact = np.cos(2 * pi * pts[:, 0]) * np.cos(2 * pi * pts[:, 1]) * np.cos(2 * pi * pts[:, 2])
    rel = float(np.linalg.norm(uh - u_exact) / np.linalg.norm(u_exact))
    assert rel < 0.12, f"3D P2 triply-periodic L2 relative error too large: {rel:.3f}"


def test_periodic_nonlinear_reaction_diffusion():
    """Periodic ties on a **nonlinear** problem: ``-Δu + u + 0.2u³ = f``, periodic in x + Dirichlet y
    on a non-conforming mesh. FEM.solve reduces the Newton residual (``r_red = Pᵀ r(P·u_red)``) so the
    tie is enforced exactly. Manufactured ``u = cos(2πx)·sin(πy)``."""
    from shapely.geometry import box

    import jno

    pi = np.pi
    dom = jno.domain({"fine": box(0, 0, 0.5, 1), "coarse": box(0.5, 0, 1, 1)}).build_mesh(0.1, sizes={"fine": 0.06})
    dom.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("bottom", lambda x, y: y < 1e-6)
    dom.tag("top", lambda x, y: y > 1 - 1e-6)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    ue = jno.np.cos(2 * jno.np.pi * xi) * jno.np.sin(jno.np.pi * yi)
    f = (5 * jno.np.pi**2 + 1) * ue + 0.2 * ue**3
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + ui * vi + 0.2 * (u * u * u) * vi - f * vi,  # -Δu + u + 0.2u³ = f
            u(xb, yb) - 0.0,
            u(xt, yt) - 0.0,
            u(xl, yl) - u(xr, yr),  # periodic in x
        ]
    )
    assert fem._mode == "nonlinear" and fem._periodic is not None
    node = fem.solve()  # nonlinear solve is a traced node -> evaluate via a throwaway crux
    crux = jno.core([node.mean], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    arr = np.asarray(crux.eval([node])).reshape(-1)
    pts = np.asarray(fem.points)
    u_exact = np.cos(2 * pi * pts[:, 0]) * np.sin(pi * pts[:, 1])
    rel = float(np.linalg.norm(arr - u_exact) / np.linalg.norm(u_exact))
    assert rel < 0.05, f"nonlinear periodic L2 relative error too large: {rel:.3f}"


def test_periodic_transient_heat():
    """Periodic ties on a **transient** problem: heat ``u_t = Δu``, periodic in x + Dirichlet y. The
    time route reduces M/A from the tie's prolongation at assembly time; solve the reduced block and
    prolong. Manufactured decay ``u = exp(-5π²t)·cos(2πx)·sin(πy)``."""
    import jax.numpy as jnp
    from shapely.geometry import box

    import jno

    pi = np.pi
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.1, time=(0.0, 0.01, 2))
    d.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    d.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    d.tag("bottom", lambda x, y: y < 1e-6)
    d.tag("top", lambda x, y: y > 1 - 1e-6)
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.cos(2 * jno.np.pi * ci[0]) * jno.np.sin(jno.np.pi * ci[1])
    fem = jno.fem(
        [
            ui.t * vi + ui.x * vi.x + ui.y * vi.y,  # u_t = Δu
            u(xb, yb) - 0.0,
            u(xt, yt) - 0.0,
            u(xl, yl) - u(xr, yr),  # periodic in x
            u(ci[0], ci[1]) - ic,  # initial condition
        ]
    )
    assert fem._mode == "transient" and fem._periodic is not None
    assert fem._periodic["n_red"] < fem._periodic["n_full"], "the time block must be reduced"
    # the block is assembled reduced (M, A, state0); step it (backward Euler from the flat block
    # matrices -- (M + dt A) u_next = M u) then prolong (u_full = P u_red) to the full layout.
    P = jnp.asarray(fem._periodic["P"])
    blk = fem.operator
    M = jnp.asarray(blk.M.todense() if hasattr(blk.M, "todense") else blk.M)
    A = jnp.asarray(blk.A.todense() if hasattr(blk.A, "todense") else blk.A)
    state = jnp.asarray(blk.state0)
    dt = float(fem.t1) / 200
    for _ in range(200):
        state = jnp.linalg.solve(M + dt * A, M @ state)
    full = np.asarray(state @ P.T).reshape(-1)
    pts = np.asarray(fem.points)
    u_exact = np.exp(-5 * pi**2 * float(fem.t1)) * np.cos(2 * pi * pts[:, 0]) * np.sin(pi * pts[:, 1])
    rel = float(np.linalg.norm(full - u_exact) / np.linalg.norm(u_exact))
    assert rel < 0.05, f"transient periodic heat L2 relative error too large: {rel:.3f}"


def test_multidirection_requires_tagged_faces():
    """Multidirectional periodicity on **auto-generated** tags (no domain.tag predicate) cannot recover
    the shared corners, so it is rejected with a clear error rather than silently mis-solving."""
    from shapely.geometry import box

    import jno

    dom = jno.domain(box(0, 0, 1, 1)).build_mesh(0.2)  # auto left/right/bottom/top, no domain.tag
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    with pytest.raises(NotImplementedError, match="multidirectional periodicity requires"):
        jno.fem([ui.x * vi.x + ui.y * vi.y + ui * vi - vi, u(xl, yl) - u(xr, yr), u(xb, yb) - u(xt, yt)])
