"""Local refinement of a **hexahedral** mesh, and the two kinds of hanging node it creates.

The 3-D half of hanging-node adaptivity, and the only h-adaptivity a hex mesh can have: mmg adapts
simplices, and rebuilding a ``Shape`` plan at a finer size field needs a mesher to rebuild *with* --
gmsh's ``Recombine3DAll`` on a plain box returns tetrahedra and no hexahedra, so there is nothing to
remesh to. Splitting a hex into 8 needs neither.

What 3-D adds over the quadrilateral case is a second kind of constrained node. A 2:1 face interface
leaves the coarse face's four **edge midpoints** hanging on 2 parents each, *and* its **centre** hanging
on all 4 corners at 1/4 each. Both are the same ``u = sum_i w_i u_parent_i`` relation and go through the
same prolongation; what differs is that they are not interchangeable when deciding which facets the
interface covers -- see ``test_a_refined_face_on_the_domain_boundary_stays_boundary``.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.domain.geometries import Geometries
from jno.utils.solver.fem_refine import (
    HEX_CORNER_OFFSETS,
    balance_marks,
    boundary_facets,
    hanging_nodes,
    refine_domain,
    refine_hexes,
)

HEX = "hexahedron"


@pytest.fixture
def x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _grid(n=4):
    m, _, _ = Geometries.equi_distant_box(x_range=(0, 1), y_range=(0, 1), z_range=(0, 1), nx=n, ny=n, nz=n, cell="hex")(
        None
    )
    b = {c.type: np.asarray(c.data) for c in m.cells}
    return np.asarray(m.points), b["hexahedron"]


def _volume(pts, hexes):
    """Total volume, as the signed sum of five tetrahedra per hex. Signed on purpose: it is the check
    that each child came out in VTK vertex order, which a positive total alone would not catch if two
    children were inverted symmetrically."""
    v = pts[hexes]
    tot = 0.0
    for a, b, c, d in [(0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)]:
        tot += np.einsum("ij,ij->i", np.cross(v[:, b] - v[:, a], v[:, c] - v[:, a]), v[:, d] - v[:, a]).sum() / 6
    return tot


def _cell_near(pts, hexes, target):
    return int(np.argmin(np.linalg.norm(pts[hexes].mean(axis=1) - np.asarray(target), axis=1)))


# ------------------------------------------------------------------------------------ the split


def test_one_hex_becomes_eight_with_the_right_lattice():
    pts, hexes = _grid()
    c = _cell_near(pts, hexes, (0.375, 0.375, 0.375))
    p1, h1 = refine_hexes(pts, hexes, [c])
    assert len(h1) == len(hexes) + 7
    # a split hex is a 3x3x3 lattice: 27 nodes, of which 8 corners already existed
    assert len(p1) == len(pts) + 19  # 12 edge midpoints + 6 face centres + 1 body centre


def test_the_split_conserves_volume_and_orientation():
    """Volume is the cheap global check; its SIGN is the one that catches a child whose corners were
    emitted in the wrong order, which would still tile the parent exactly and then assemble with a
    negative Jacobian."""
    pts, hexes = _grid()
    v0 = _volume(pts, hexes)
    p1, h1 = refine_hexes(pts, hexes, [0, 5, 21])
    assert _volume(p1, h1) == pytest.approx(v0, rel=1e-12)
    assert _volume(p1, h1) > 0


def test_children_are_emitted_in_vtk_vertex_order():
    """Each child must be listed in the same corner order as its parent, since that is the order every
    facet, quadrature and basis table in jNO reads a hex in."""
    pts, hexes = _grid(2)
    p1, h1 = refine_hexes(pts, hexes, [0])
    child = p1[h1[len(hexes) - 1 + 1]]  # the first emitted child
    lo = child.min(axis=0)
    span = child.max(axis=0) - lo
    for k, off in enumerate(HEX_CORNER_OFFSETS):
        np.testing.assert_allclose((child[k] - lo) / span, off, atol=1e-12)


def test_nodes_are_shared_between_neighbouring_split_cells():
    """Two adjacent hexes split in the same round must agree on the 9 nodes of their shared face --
    creating them per cell duplicates them and disconnects the two while every count still looks
    right."""
    pts, hexes = _grid()
    p1, h1 = refine_hexes(pts, hexes, [0, 1])
    assert len(p1) == len(np.unique(np.round(p1, 12), axis=0)), "a node was duplicated"


def test_refining_twice_reuses_the_hanging_nodes(x64):
    """The 3-D form of the bug the quadrilateral path hit: a node already at an edge midpoint or a face
    centre is the hanging node a previous round left there, and refining this cell promotes it. Face and
    edge topology cannot see it once the mesh is non-conforming."""
    pts, hexes = _grid()
    p1, h1 = refine_hexes(pts, hexes, [_cell_near(pts, hexes, (0.4, 0.4, 0.4))])
    p2, h2 = refine_hexes(p1, h1, [_cell_near(p1, h1, (0.4, 0.4, 0.4))])
    assert len(p2) == len(np.unique(np.round(p2, 12), axis=0)), "a node was duplicated at the interface"
    assert _volume(p2, h2) == pytest.approx(_volume(pts, hexes), rel=1e-12)


# ------------------------------------------------------------- hanging nodes: both kinds, at once


def test_a_2to1_interface_hangs_on_edge_midpoints_AND_face_centres():
    """The headline of the 3-D case. One interior hex refined leaves 18 constrained nodes: its 12 edge
    midpoints on 2 parents each, and its 6 face centres on 4 parents each."""
    pts, hexes = _grid()
    p1, h1 = refine_hexes(pts, hexes, [_cell_near(pts, hexes, (0.375, 0.375, 0.375))])
    hang = hanging_nodes(p1, h1, HEX)
    by_arity = {}
    for parents in hang.values():
        by_arity[len(parents)] = by_arity.get(len(parents), 0) + 1
    assert by_arity == {2: 12, 4: 6}, by_arity


def test_hanging_weights_are_a_partition_of_unity_at_the_parents_centroid():
    """A face centre's Q1 weights are exactly 1/4 each because the centre is the parameter point
    (1/2, 1/2) of the bilinear face -- true however the face is warped, which is why no Newton
    inversion is needed here (unlike the general non-matching tie)."""
    pts, hexes = _grid()
    p1, h1 = refine_hexes(pts, hexes, [0, 21])
    for n, parents in hanging_nodes(p1, h1, HEX).items():
        assert sum(w for _, w in parents) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(p1[n], sum(w * p1[i] for i, w in parents), atol=1e-12)


def test_a_uniformly_refined_hex_mesh_has_no_hanging_nodes():
    pts, hexes = _grid(2)
    p1, h1 = refine_hexes(pts, hexes, np.arange(len(hexes)))
    assert hanging_nodes(p1, h1, HEX) == {}
    assert len(h1) == 8 * len(hexes)


def test_repeated_refinement_stays_balanced_and_unchained():
    """No hanging node may have a hanging PARENT — the property the whole scheme rests on, in 3-D where
    a cell has 12 edges and 6 faces to get it wrong on."""
    pts, hexes = _grid()
    for rnd in range(3):
        mk = np.where(np.linalg.norm(pts[hexes].mean(axis=1) - 0.35, axis=1) < 0.3)[0]
        pts, hexes = refine_hexes(pts, hexes, mk)
        hang = hanging_nodes(pts, hexes, HEX)
        chained = [n for par in hang.values() for n, _ in par if n in hang]
        assert not chained, f"round {rnd + 1}: {len(chained)} hanging nodes have a hanging parent"


def test_the_balance_is_by_edge_so_edge_only_neighbours_are_covered():
    """Two hexes touching along an EDGE alone share no face, so a face-neighbour walk never compares
    them and they can drift two levels apart. The balance is tested on edges, which covers it."""
    pts, hexes = _grid()
    p1, h1 = refine_hexes(pts, hexes, [_cell_near(pts, hexes, (0.375, 0.375, 0.375))])
    # ask to refine again; whatever the closure adds, the result must stay unchained
    p2, h2 = refine_hexes(p1, h1, [_cell_near(p1, h1, (0.3, 0.3, 0.3))])
    hang = hanging_nodes(p2, h2, HEX)
    assert not [n for par in hang.values() for n, _ in par if n in hang]


def test_balance_marks_only_grows_the_set():
    pts, hexes = _grid()
    asked = [0]
    closed = balance_marks(pts, hexes, asked, HEX)
    assert closed.sum() >= len(asked)
    assert closed[0]


# ---------------------------------------------------------------------------------- the boundary


def test_an_interior_refinement_does_not_change_the_boundary():
    """The cube's surface is 6 x 16 = 96 faces, and refining a cell in the middle must leave every one
    of them: the 2:1 interface it creates is interior, and the topological rule cannot tell."""
    pts, hexes = _grid()
    assert len(boundary_facets(pts, hexes, None, HEX)) == 96
    p1, h1 = refine_hexes(pts, hexes, [_cell_near(pts, hexes, (0.375, 0.375, 0.375))])
    bf = boundary_facets(p1, h1, None, HEX)
    assert len(bf) == 96
    assert _all_on_the_cube_surface(p1, bf)


def test_a_refined_face_on_the_domain_boundary_stays_boundary():
    """The 3-D-only trap, and the one that makes the covering rule subtler than in 2-D.

    A corner hex has 3 faces on the cube's surface. Splitting it turns each into 4 sub-faces, all still
    boundary: 96 - 3 + 12 = 105. But those sub-faces have hanging EDGE MIDPOINTS as corners -- the edges
    are shared with unrefined neighbours, so the midpoints genuinely hang -- and a rule that treats any
    hanging node as proof of interiority deletes them. Measured when it did: 96 faces, silently losing 9
    faces of the domain's own surface. Only a hanging node with as many parents as the facet has
    vertices (a face CENTRE) means the facet was covered.
    """
    pts, hexes = _grid()
    corner = _cell_near(pts, hexes, (0.125, 0.125, 0.125))
    p1, h1 = refine_hexes(pts, hexes, [corner])
    bf = boundary_facets(p1, h1, None, HEX)
    assert len(bf) == 105
    assert _all_on_the_cube_surface(p1, bf)
    # and the surface area is unchanged, which is the statement that matters physically
    assert _surface_area(p1, bf) == pytest.approx(6.0, rel=1e-12)


def _all_on_the_cube_surface(pts, faces):
    return all(
        any(np.all(np.isclose(pts[f][:, d], 0)) or np.all(np.isclose(pts[f][:, d], 1)) for d in range(3)) for f in faces
    )


def _surface_area(pts, faces):
    v = pts[faces]
    return float(np.linalg.norm(np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 1]), axis=1).sum() / 2)


# ------------------------------------------------------------------------------ end to end solve


def _box(n):
    return jno.Shape.box(0, 0, 0, 1, 1, 1).structured(n=n).quad().domain(compute_mesh_connectivity=False)


def _poisson(dom):
    u, v = dom.fem_symbols()
    xi, yi, zi = dom.variable("interior", split=True)[:3]
    xb, yb, zb = dom.variable("boundary", split=True)[:3]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u(xb, yb, zb) - 0.0])
    return np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).ravel()


def _centre(dom, sol):
    p = np.asarray(dom.mesh.points)[:, :3]
    return float(sol[int(np.argmin(np.linalg.norm(p - 0.5, axis=1)))])


def test_a_warped_hex_keeps_its_constraints_exact_but_not_its_volume():
    """The one place 3-D genuinely differs from 2-D, measured rather than assumed.

    A quadrilateral's edges are straight, so its four children tile it exactly and the area is conserved
    to rounding. A hexahedron's FACES are bilinear and in general non-planar, so the children -- whose
    own faces are bilinear over different corners -- do not reproduce the parent's geometry exactly.
    Measured on a mesh whose interior nodes are displaced by up to 0.06 against a cell size of 0.25:
    splitting one hex moves the total volume by 3.9e-04, and the gap SHRINKS as the mesh refines
    (1.2e-04, 8.8e-05, 6.2e-05 over three uniform rounds) -- the O(h^2) geometry error of straight-edged
    elements, not a bookkeeping mistake. On an affine mesh (any lattice) it is exact.

    The constraints are unaffected: a face centre is the mean of its four corners for any warping, and
    the Q1 weights at that parameter point are 1/4 whatever the cell looks like.
    """
    pts, hexes = _grid()
    rng = np.random.default_rng(0)
    inner = np.all((pts > 1e-9) & (pts < 1 - 1e-9), axis=1)
    warped = pts.copy()
    warped[inner] += 0.06 * rng.uniform(-1, 1, (int(inner.sum()), 3))

    v0 = _volume(warped, hexes)
    p1, h1 = refine_hexes(warped, hexes, [21])
    assert abs(_volume(p1, h1) - v0) < 1e-3, "the geometry error should be O(h^2), not O(1)"
    assert abs(_volume(p1, h1) - v0) > 1e-9, "a warped hex mesh is NOT volume-exact; if it is, this test lies"

    for n, parents in hanging_nodes(p1, h1, HEX).items():
        assert sum(w for _, w in parents) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(p1[n], sum(w * p1[i] for i, w in parents), atol=1e-12)


def test_a_refined_hex_solve_converges(x64):
    """-Lap u = 1 on the unit cube against a uniform 16^3 reference. Two rounds into the centre from a
    4^3 must beat the 4^3 it came from, and beat a uniform 8^3 that costs more nodes.

    Measured: 4^3 err 6.0e-03 at 125 nodes, uniform 8^3 1.1e-03 at 729, two refinement rounds 2.7e-04
    at 1277 -- a quarter the error of the uniform mesh for 1.75x its nodes.
    """
    ref = _centre(_box(16), _poisson(_box(16)))
    d = _box(4)
    coarse = abs(_centre(d, _poisson(d)) - ref)
    uniform8 = abs(_centre(_box(8), _poisson(_box(8))) - ref)

    for _ in range(2):
        p = np.asarray(d.mesh.points)[:, :3]
        h = np.asarray(d.mesh.cells_dict[HEX])
        d = refine_domain(d, np.where(np.linalg.norm(p[h].mean(axis=1) - 0.5, axis=1) < 0.35)[0])
    got = abs(_centre(d, _poisson(d)) - ref)

    assert got < coarse, f"refining made it worse ({got:.2e} vs {coarse:.2e})"
    assert got < uniform8, f"local refinement lost to a uniform mesh ({got:.2e} vs {uniform8:.2e})"


def test_the_hex_constraint_holds_on_the_solved_field(x64):
    """Both kinds of constraint, on the assembled and solved system: exact, because the DOF is
    eliminated -- so this asserts the elimination reached the solve at all."""
    d = _box(4)
    p = np.asarray(d.mesh.points)[:, :3]
    h = np.asarray(d.mesh.cells_dict[HEX])
    d = refine_domain(d, np.where(np.linalg.norm(p[h].mean(axis=1) - 0.5, axis=1) < 0.35)[0])
    sol = _poisson(d)
    hang = d._fem_hanging_nodes
    assert {len(v) for v in hang.values()} == {2, 4}, "both kinds must be present for this to test both"
    for n, parents in hang.items():
        assert sol[n] == pytest.approx(sum(w * sol[i] for i, w in parents), abs=1e-12)


def test_the_hex_interface_is_not_pinned_as_a_boundary(x64):
    """The 3-D form of the blocker: the aggregate ``boundary`` is what a Dirichlet condition resolves
    through, so an interface mistaken for boundary is pinned in the middle of the domain."""
    d = _box(4)
    p = np.asarray(d.mesh.points)[:, :3]
    h = np.asarray(d.mesh.cells_dict[HEX])
    d = refine_domain(d, np.where(np.linalg.norm(p[h].mean(axis=1) - 0.5, axis=1) < 0.35)[0])
    p = np.asarray(d.mesh.points)[:, :3]
    bn = {int(i) for i in np.asarray((getattr(d, "tag_indices", {}) or {})["boundary"]).reshape(-1)}
    on_surface = {i for i in range(len(p)) if np.isclose(p[i], 0).any() or np.isclose(p[i], 1).any()}
    assert bn == on_surface
    assert not (bn & set(d._fem_hanging_nodes)), "a hanging node was called boundary"
