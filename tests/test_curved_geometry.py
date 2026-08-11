"""Curved (isoparametric) meshes — ``emit.build(shape, order=2)``.

jNO's geometry is affine at every basis order: the cell Jacobian is built from P1 vertices, and the
higher-order nodes are *synthesised* by interpolating reference points through that affine map
(``fem_native._get_mesh`` -> ``_promote_to_degree``). So a P2 midside node lands on the straight-edge
midpoint and the domain stays a polygon however high the element order goes.

That polygonal approximation carries an **O(h²)** domain error regardless of basis order, which is what
caps P2/P3 at second order on any curved boundary, and leaves facet normals O(h) wrong. ``order=2`` asks
gmsh to place those nodes on the actual CAD surface instead.

This file covers the **geometry only** — that the nodes land where they should. Using them (the
per-quadrature-point Jacobian) is the next step; until then the assembler still treats every cell as
affine, so nothing here changes a solve.
"""

import numpy as np
import pytest

import jno
from jno.geometry.emit import build


def _disk(size, order):
    return build(jno.Shape.disk(0.0, 0.0, 1.0, size=size), order=order)[0]


def test_order_one_is_unchanged():
    """The default must stay exactly what it was — first-order blocks, no curving."""
    mesh = _disk(0.35, 1)
    assert set(mesh.cells_dict) == {"triangle", "line"}
    assert mesh.cells_dict["triangle"].shape[1] == 3
    assert mesh.cells_dict["line"].shape[1] == 2


def test_curved_mesh_emits_second_order_blocks():
    mesh = _disk(0.35, 2)
    assert set(mesh.cells_dict) == {"triangle6", "line3"}
    assert mesh.cells_dict["triangle6"].shape[1] == 6
    assert mesh.cells_dict["line3"].shape[1] == 3


def test_boundary_nodes_lie_on_the_true_circle():
    """Both the vertices and the MIDSIDE nodes, to machine precision — the midsides are the whole
    point, since those are the ones jNO would otherwise place on the chord."""
    mesh = _disk(0.35, 2)
    pts = np.asarray(mesh.points)[:, :2]
    edges = mesh.cells_dict["line3"]
    for cols, what in ((edges[:, :2], "vertices"), (edges[:, 2:3], "midsides")):
        r = np.linalg.norm(pts[np.unique(cols)], axis=1)
        assert np.abs(r - 1.0).max() < 1e-12, what


def test_the_synthesised_midpoint_carries_the_o_h2_domain_error():
    """The reason the whole feature exists, measured rather than asserted.

    A curved node sits on the circle exactly at every refinement. The straight-edge midpoint jNO
    synthesises today is off by ``R(1 - cos(θ/2)) = O(h²)`` — it quarters as h halves, which is exactly
    the second-order domain error that caps P2 at O(h²) no matter how good the basis is."""
    errs = {}
    for size in (0.6, 0.3, 0.15):
        mesh = _disk(size, 2)
        pts = np.asarray(mesh.points)[:, :2]
        e = mesh.cells_dict["line3"]
        curved = np.abs(np.linalg.norm(pts[e[:, 2]], axis=1) - 1.0).max()
        straight = np.abs(np.linalg.norm(0.5 * (pts[e[:, 0]] + pts[e[:, 1]]), axis=1) - 1.0).max()
        assert curved < 1e-12, f"curved nodes must be exact at size={size}"
        errs[size] = straight

    assert errs[0.6] > 1e-2, "the straight-midpoint error must be significant to begin with"
    for coarse, fine in ((0.6, 0.3), (0.3, 0.15)):
        rate = errs[coarse] / errs[fine]
        assert 2.8 < rate < 5.5, f"expected ~4x per halving (O(h^2)), got {rate:.2f}"


def test_a_polygon_is_unaffected_by_curving():
    """Affine is the special case: a straight-sided domain has no curvature to add, so the curved
    midside nodes must coincide with the chord midpoints. If they drift, the curving is inventing
    geometry rather than following the CAD."""
    mesh = build(jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3), order=2)[0]
    pts = np.asarray(mesh.points)[:, :2]
    e = mesh.cells_dict["line3"]
    assert np.allclose(pts[e[:, 2]], 0.5 * (pts[e[:, 0]] + pts[e[:, 1]]), atol=1e-12)


def test_curving_works_in_3d():
    mesh = build(jno.Shape.sphere(0.0, 0.0, 0.0, 1.0, size=0.6), order=2)[0]
    assert mesh.cells_dict["tetra10"].shape[1] == 10
    assert mesh.cells_dict["triangle6"].shape[1] == 6
    surf = np.unique(mesh.cells_dict["triangle6"])
    r = np.linalg.norm(np.asarray(mesh.points)[surf], axis=1)
    assert np.abs(r - 1.0).max() < 1e-12


def _curved_disk_domain(size=0.3):
    return jno.Shape.disk(0.0, 0.0, 1.0, size=size).curved().domain()


def test_the_domain_tags_the_whole_curved_facet():
    """The tag machinery derives a boundary node set by walking the P1 edge chain, which returns only
    the facet VERTICES. A curved facet also carries midside nodes, and those are genuine boundary DOFs
    — without them a Dirichlet condition pins each facet's corners and leaves its interior free."""
    d = _curved_disk_domain()
    mesh = d.built_mesh
    tagged = set(np.asarray(d.tag_indices["boundary"]).reshape(-1).tolist())
    facet_nodes = set(np.unique(mesh.cells_dict["line3"]).tolist())
    assert tagged == facet_nodes, "the boundary tag must cover the whole facet, not just its vertices"
    r = np.linalg.norm(np.asarray(mesh.points)[sorted(tagged), :2], axis=1)
    assert np.abs(r - 1.0).max() < 1e-12


def test_mesh_order_survives_the_csg_operators():
    """`.curved()` must not be silently dropped by a later boolean or transform — every Shape
    constructor rebuilds the dataclass, and each one that forgot the flag would lose the curving with
    no indication."""
    disk = jno.Shape.disk(0.0, 0.0, 1.0, size=0.4).curved()
    hole = jno.Shape.disk(0.0, 0.0, 0.3, size=0.4)
    for shape, what in (
        (disk - hole, "cut"),
        (disk | hole, "fuse"),
        (disk.translate((0.1, 0.0)), "translate"),
        (disk.sized(0.5), "sized"),
        (disk.name("part"), "name"),
    ):
        assert shape._mesh_order == 2, what


def _poisson_terms(d, order):
    u, v = d.fem_symbols(order=order)
    c = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    ui, vi = u.bind(x=c[0], y=c[1]), v.bind(x=c[0], y=c[1])
    return [ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(b[0], b[1]) - 0.0]


def test_a_p1_basis_on_a_curved_mesh_is_refused():
    """Isoparametric means geometry order == basis order. A curved mesh under a P1 basis puts the
    midside DOF coordinates (on the arc) and the geometric map (from the chord) in disagreement — an
    inconsistent discretisation, not merely a coarse one."""
    d = _curved_disk_domain(0.35)
    with pytest.raises(ValueError, match="order-2 geometry but this field is P1"):
        jno.fem(_poisson_terms(d, 1))


def test_the_assembler_refuses_curved_geometry_for_now():
    """The assembler still builds one constant Jacobian per cell from its vertices. Solving on a
    curved mesh with that map would use chord geometry with arc-positioned DOFs — wrong in a way no
    test would flag — so it refuses until the per-quadrature-point Jacobian lands."""
    d = _curved_disk_domain(0.35)
    with pytest.raises(NotImplementedError, match="per-quadrature-point Jacobian"):
        jno.fem(_poisson_terms(d, 2))


@pytest.mark.parametrize("size", [1.2, 0.8])
def test_a_very_coarse_curved_boundary_is_still_exact(size):
    """Extreme: so few edges that each spans a large arc. The nodes are placed by the CAD kernel, not
    interpolated, so coarseness does not degrade them — only the element count does."""
    mesh = _disk(size, 2)
    pts = np.asarray(mesh.points)[:, :2]
    e = mesh.cells_dict["line3"]
    assert len(e) >= 3
    assert np.abs(np.linalg.norm(pts[e[:, 2]], axis=1) - 1.0).max() < 1e-12
