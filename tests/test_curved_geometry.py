"""Curved (isoparametric) meshes — ``emit.build(shape, order=2)``.

jNO's geometry is affine at every basis order: the cell Jacobian is built from P1 vertices, and the
higher-order nodes are *synthesised* by interpolating reference points through that affine map
(``fem_native._get_mesh`` -> ``_promote_to_degree``). So a P2 midside node lands on the straight-edge
midpoint and the domain stays a polygon however high the element order goes.

That polygonal approximation carries an **O(h²)** domain error regardless of basis order, which is what
caps P2/P3 at second order on any curved boundary, and leaves facet normals O(h) wrong. ``order=2`` asks
gmsh to place those nodes on the actual CAD surface instead.

The assembler forms the Jacobian **per quadrature point** on such a mesh, since the map is no longer
affine. The gate is a convergence RATE, not an error number: straight-sided P2 is capped at O(h^2) by
the polygonal domain, curved P2 recovers its own O(h^3).

Deliberately out of scope, and refused rather than approximated: an order mismatch (a P1 basis on
order-2 geometry), a 4th-order form (the physical-Hessian transform is derived for an affine cell), and
the non-nodal families. Facet normals are still straight-facet, so the O(h) normal error is untouched.
"""

import jax
import numpy as np
import pytest

import jno
from jno.geometry.emit import build


@pytest.fixture(autouse=True)
def _x64():
    """The rate study resolves errors down to ~7e-7 on a solution of size 0.25 — below what float32
    can represent, where the curved and straight results stop being distinguishable (measured: the
    curved error floors at 5e-5 and moves 4% under a quadrature refinement that should not move it)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


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


def _poisson_l2(curved, size, quad_degree=None):
    """-Δu = 1 on the unit disk (exact ``u = (1-r²)/4``), P2. Returns the RMS nodal error."""
    sh = jno.Shape.disk(0.0, 0.0, 1.0, size=size)
    d = (sh.curved() if curved else sh).domain()
    u, v = d.fem_symbols(order=2)
    c = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    ui, vi = u.bind(x=c[0], y=c[1]), v.bind(x=c[0], y=c[1])
    kw = {} if quad_degree is None else {"quad_degree": quad_degree}
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(b[0], b[1]) - 0.0], **kw)
    sol = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.points)[:, :2]
    return float(np.sqrt(np.mean((sol - (1.0 - (pts**2).sum(1)) / 4.0) ** 2)))


def test_curved_geometry_lifts_p2_from_second_to_third_order():
    """The gate for the whole feature, and the only unambiguous one: a RATE, not an error number.

    Straight-sided P2 on a curved domain is capped at O(h²) by the polygonal approximation — the basis
    can do better and the geometry will not let it. With curved geometry the same basis recovers its
    O(h³). Measured 4.05x / 3.93x per halving against 9.95x / 10.43x."""
    sizes = (0.4, 0.2, 0.1)
    straight = [_poisson_l2(False, s) for s in sizes]
    curved = [_poisson_l2(True, s) for s in sizes]

    for lo, hi in zip(straight, straight[1:]):
        assert 3.0 < lo / hi < 5.5, f"straight-sided must stay O(h^2), got {lo / hi:.2f}x"
    for lo, hi in zip(curved, curved[1:]):
        assert lo / hi > 7.0, f"curved must reach O(h^3), got {lo / hi:.2f}x"
    assert curved[-1] < straight[-1] / 100.0, "curved must be dramatically more accurate at the finest h"


def test_the_result_is_not_limited_by_quadrature():
    """A curved map makes the integrand rational, so no rule is exact and under-integration would look
    exactly like a geometry bug. Refining the rule must not move the answer."""
    base = _poisson_l2(True, 0.2)
    for qd in (6, 8):
        assert abs(_poisson_l2(True, 0.2, quad_degree=qd) - base) < 1e-3 * base


def test_a_p1_basis_on_a_curved_mesh_is_refused():
    """Isoparametric means geometry order == basis order. A curved mesh under a P1 basis puts the
    midside DOF coordinates (on the arc) and the geometric map (from the chord) in disagreement — an
    inconsistent discretisation, not merely a coarse one."""
    d = _curved_disk_domain(0.35)
    with pytest.raises(ValueError, match="order-2 geometry but this field is P1"):
        jno.fem(_poisson_terms(d, 1))


def test_a_fourth_order_form_on_a_curved_cell_is_refused():
    """The physical-Hessian push-forward is derived for an AFFINE cell (``∂²ξ/∂x² ≡ 0``). On a curved
    cell it gains a curvature term this does not carry, so Argyris / Morley / phase-field / plates
    would be wrong with nothing to flag it."""
    from jno.utils.solver.fem_lagrange import identity_pushforward_hess

    ref_hess = np.zeros((3, 6, 1, 2, 2))
    identity_pushforward_hess(ref_hess, np.eye(2))  # affine: fine
    with pytest.raises(NotImplementedError, match="AFFINE cell"):
        identity_pushforward_hess(ref_hess, np.broadcast_to(np.eye(2), (3, 2, 2)))


@pytest.mark.parametrize("size", [1.2, 0.8])
def test_a_very_coarse_curved_boundary_is_still_exact(size):
    """Extreme: so few edges that each spans a large arc. The nodes are placed by the CAD kernel, not
    interpolated, so coarseness does not degrade them — only the element count does."""
    mesh = _disk(size, 2)
    pts = np.asarray(mesh.points)[:, :2]
    e = mesh.cells_dict["line3"]
    assert len(e) >= 3
    assert np.abs(np.linalg.norm(pts[e[:, 2]], axis=1) - 1.0).max() < 1e-12
