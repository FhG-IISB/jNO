"""`.structured()` on a plan of several axis-aligned boxes: a rectilinear lattice.

Every face of every box becomes a grid plane, the spacing between two planes is the finest size of the
regions that slab crosses, and each cell belongs to the first region that contains its centroid. So
material interfaces are resolved exactly, opposite faces match node for node, and every cell is a
voxel (or its six Kuhn tets) -- no slivers.
"""

import jax
import numpy as np
import pytest

import jno

Bx, R = jno.shape.box, jno.shape.rect


def _volumes(pts, cells):
    V = pts[cells]
    return np.abs(np.linalg.det(np.stack([V[:, k] - V[:, 0] for k in (1, 2, 3)], axis=2))) / 6


def _stack(**kw):
    """quartz | film (with a hole) | air, the film thinner than the bulk size."""
    return (
        Bx(0.3, 0.3, 0.0, 0.5, 0.5, 0.06).name("hole").attach(k=1.0).size(0.1)
        + Bx(0.0, 0.0, 0.0, 0.8, 0.8, 0.06).name("film").attach(k=3.0).size(0.03)
        + Bx(0.0, 0.0, -0.3, 0.8, 0.8, 0.0).name("quartz").attach(k=2.0)
        + Bx(0.0, 0.0, 0.06, 0.8, 0.8, 0.4).name("air").attach(k=1.0)
    ).sized(0.2)


def test_the_lattice_planes_are_the_box_faces_and_the_slab_sizes():
    d = _stack().structured().domain()
    pts = np.asarray(d.mesh.points)
    xs, zs = np.unique(pts[:, 0]), np.unique(pts[:, 2])
    for p in (0.0, 0.3, 0.5, 0.8):
        assert np.isclose(xs, p).any(), f"x = {p} is not a grid plane"
    for p in (-0.3, 0.0, 0.06, 0.4):
        assert np.isclose(zs, p).any(), f"z = {p} is not a grid plane"
    # x: every slab crosses the film (size 0.03), so it is at most 0.03 apart everywhere
    assert np.diff(xs).max() <= 0.03 + 1e-12
    # z: the film slab at 0.03, the quartz and air slabs at the bulk 0.2 -- not dragged down to 0.03
    dz = np.diff(zs)
    assert np.all(dz[(zs[:-1] >= 0) & (zs[1:] <= 0.06 + 1e-12)] <= 0.03 + 1e-12)
    assert np.all(dz[zs[1:] <= 1e-12] > 0.1) and np.all(dz[zs[:-1] >= 0.06 - 1e-12] > 0.1)
    assert len(np.unique(pts[:, 1])) == len(xs)  # the same lattice along y


def test_each_region_gets_exactly_its_volume():
    d = _stack().structured().domain()
    pts = np.asarray(d.mesh.points)
    tets = np.asarray(d.mesh.cells_dict["tetra"])
    vol = _volumes(pts, tets)
    want = {"hole": 0.2 * 0.2 * 0.06, "film": 0.8 * 0.8 * 0.06 - 0.2 * 0.2 * 0.06, "quartz": 0.64 * 0.3, "air": 0.64 * 0.34}
    sets = d.mesh.cell_sets
    for name, v in want.items():
        assert np.isclose(vol[np.asarray(sets[name][0])].sum(), v, rtol=1e-12), name
    assert np.isclose(vol.sum(), 0.64 * 0.7, rtol=1e-12)
    assert np.unique(np.concatenate([np.asarray(sets[n][0]) for n in want])).size == len(tets)  # a partition


def test_interfaces_outer_faces_and_their_orientation():
    d = _stack().structured().domain()
    pts = np.asarray(d.mesh.points)
    for name, (axis, value) in {
        "film|quartz": (2, 0.0),
        "left": (0, 0.0),
        "right": (0, 0.8),
        "front": (1, 0.0),
        "back": (1, 0.8),
        "bottom": (2, -0.3),
        "top": (2, 0.4),
    }.items():
        tri = d.tag_facets(name)
        assert tri is not None and len(tri), name
        np.testing.assert_allclose(pts[tri][..., axis], value, atol=1e-12, err_msg=name)
    # the hole's walls and floor/lid are interfaces, not boundary
    walls = d.tag_facets("film|hole")
    assert np.isclose(pts[walls].reshape(-1, 3)[:, 0].min(), 0.3) and np.isclose(pts[walls].reshape(-1, 3)[:, 0].max(), 0.5)
    assert d.tag_facets("hole|quartz") is not None and d.tag_facets("air|hole") is not None
    bnd = d.tag_facets("boundary")
    c = pts[bnd].mean(axis=1)
    on_box = np.isclose(c[:, 0], 0) | np.isclose(c[:, 0], 0.8) | np.isclose(c[:, 1], 0) | np.isclose(c[:, 1], 0.8)
    on_box |= np.isclose(c[:, 2], -0.3) | np.isclose(c[:, 2], 0.4)
    assert on_box.all(), "an interface facet leaked into 'boundary'"
    # outward: the triangle normal points away from the box centre
    P = pts[bnd]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    assert (np.einsum("ij,ij->i", n, c - np.array([0.4, 0.4, 0.05])) > 0).all()


def test_opposite_faces_match_node_for_node():
    d = _stack().structured().domain()
    pts = np.asarray(d.mesh.points)
    L = pts[np.unique(d.tag_facets("left"))][:, 1:]
    Rr = pts[np.unique(d.tag_facets("right"))][:, 1:]
    assert sorted(map(tuple, np.round(L, 12))) == sorted(map(tuple, np.round(Rr, 12)))


def test_a_union_that_does_not_fill_its_box_drops_the_empty_cells():
    """An L-shaped union of two boxes: cells outside it are dropped, and no node is left orphaned."""
    d = (Bx(0, 0, 0, 1, 1, 0.5) | Bx(0, 0, 0.5, 0.5, 1, 1)).sized(0.25).structured().domain()
    pts = np.asarray(d.mesh.points)
    tets = np.asarray(d.mesh.cells_dict["tetra"])
    assert np.isclose(_volumes(pts, tets).sum(), 0.5 + 0.25, rtol=1e-12)
    assert np.unique(tets).size == len(pts)
    # the step face at x = 0.5, z > 0.5 is outer boundary, named by its outward normal
    right = d.tag_facets("right")
    xr = pts[right][..., 0]
    assert np.isclose(xr, 0.5).any() and np.isclose(xr, 1.0).any()


def test_hexahedra_and_a_2d_stack():
    d = (
        (Bx(0, 0, 0, 1, 1, 0.2).name("a").size(0.1) + Bx(0, 0, 0.2, 1, 1, 1).name("b"))
        .sized(0.25)
        .structured()
        .quad()
        .domain()
    )
    assert "hexahedron" in d.mesh.cells_dict and d.tag_facets("a|b").shape[1] == 4
    d2 = (R(0, 0, 1, 0.1).name("a").size(0.05) + R(0, 0.1, 1, 1).name("b")).sized(0.25).structured().domain()
    pts = np.asarray(d2.mesh.points)
    e = d2.tag_facets("a|b")
    np.testing.assert_allclose(pts[e][..., 1], 0.1, atol=1e-12)
    tri = np.asarray(d2.mesh.cells_dict["triangle"])
    A = pts[tri][:, :, :2]
    e1, e2 = A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]
    area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    assert (area > 0).all(), "triangles must be counterclockwise"
    assert np.isclose(area[np.asarray(d2.mesh.cell_sets["a"][0])].sum(), 0.1)


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_a_two_layer_conduction_is_exact_on_the_lattice(x64):
    """-(k u')' = 0 through two layers, u = 0 at the bottom and 1 at the top: piecewise linear in z, so
    P1 reproduces it to round-off when the interface is a grid plane and k reaches each region."""
    d = (
        (
            Bx(0, 0, 0, 0.5, 0.5, 0.3).name("lo").attach(k=4.0).size(0.1)
            + Bx(0, 0, 0.3, 0.5, 0.5, 1).name("hi").attach(k=1.0)
        )
        .sized(0.25)
        .structured()
        .domain()
    )
    u, v = d.fem_symbols()
    x, y, z = d.variable("interior", split=True)[:3]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    bot, top = d.variable("bottom", split=True)[:3], d.variable("top", split=True)[:3]
    fem = jno.fem([d.k * (ui.x * vi.x + ui.y * vi.y + ui.z * vi.z), u(*bot) - 0.0, u(*top) - 1.0])
    sol = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1)
    zz = np.asarray(d.mesh.points)[:, 2]
    # slopes a (lower) and b (upper): equal flux 4 a = 1 b, and a 0.3 + b 0.7 = 1
    a = 1.0 / (0.3 + 4 * 0.7)
    exact = np.where(zz <= 0.3, a * zz, a * 0.3 + 4 * a * (zz - 0.3))
    np.testing.assert_allclose(sol, exact, atol=1e-9)


def test_periodic_edge_ties_keep_the_lattice():
    """N1E periodic ties re-mesh a gmsh plan conforming; a lattice already is, so it must be kept (a gmsh
    re-mesh would silently drop the structured mesh)."""
    d = _stack().structured().domain()
    n_before = len(d.mesh.points)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    xi, yi, zi = d.variable("interior", split=True)[:3]
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)

    def face(nm):
        c = d.variable(nm, split=True)
        return u.bind(x=c[0], y=c[1], z=c[2])

    vol = jno.np.inner(cu, cv) + jno.np.inner(u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi))
    fem = jno.fem([vol, face("left") - face("right"), face("front") - face("back")])
    assert len(d.mesh.points) == n_before
    assert fem._periodic is not None and fem._periodic["n_red"] < fem._periodic["n_full"]


@pytest.mark.parametrize(
    "plan, match",
    [
        (
            lambda: (Bx(0, 0, 0, 1, 1, 1).name("a") + jno.shape.sphere(0.5, 0.5, 0.5, 0.2).name("b")).sized(0.2),
            "axis-aligned rect/box",
        ),
        (
            lambda: (Bx(0, 0, 0, 1, 1, 1).name("a") + Bx(0, 0, 1, 1, 1, 2).name("b")).sized(lambda x, y, z: 0.1 + 0 * x),
            "graded",
        ),
        (
            lambda: (
                Bx(0, 0, 0, 1, 1, 1).rotate((0, 0, 0), (0, 0, 1), 0.3).name("a") + Bx(0, 0, 1, 1, 1, 2).name("b")
            ).sized(0.2),
            "axis-aligned",
        ),
    ],
    ids=["sphere", "graded-size", "rotated"],
)
def test_what_a_lattice_cannot_mesh_is_refused_by_name(plan, match):
    with pytest.raises(NotImplementedError, match=match):
        plan().structured().domain()


def test_cell_counts_are_refused_on_a_multi_box_plan():
    with pytest.raises(ValueError, match="size"):
        (Bx(0, 0, 0, 1, 1, 1).name("a") + Bx(0, 0, 1, 1, 1, 2).name("b")).sized(0.2).structured(n=4).domain()


def test_a_non_conforming_plan_is_refused():
    plan = jno.shape.regions(a=Bx(0, 0, 0, 1, 1, 1), b=Bx(0, 0, 1, 1, 1, 2), conforming=False)
    with pytest.raises(NotImplementedError, match="conforming"):
        plan.sized(0.2).structured().domain()


# ---------------------------------------------------------------- grading between slabs of different size


def _z_spacing(d):
    return np.diff(np.unique(np.asarray(d.mesh.points)[:, 2]))


def _graded_stack():
    return (Bx(0, 0, 0, 0.4, 0.4, 0.03).name("film").size(0.01) + Bx(0, 0, -1, 0.4, 0.4, 0).name("below")).sized(0.2)


def test_without_grading_the_spacing_jumps_at_the_plane():
    dz = _z_spacing(_graded_stack().structured().domain())
    assert np.maximum(dz[1:] / dz[:-1], dz[:-1] / dz[1:]).max() > 10  # 0.2 below straight to 0.01 in the film


@pytest.mark.parametrize("r", [1.2, 1.5])
def test_grading_caps_the_growth_between_neighbouring_cells(r):
    """Cells grow by at most about r per cell away from the finer slab, the planes stay grid planes,
    the fine slab keeps its own spacing and the coarse slab still reaches its own size far away."""
    d = _graded_stack().structured(grading=r).domain()
    zs = np.unique(np.asarray(d.mesh.points)[:, 2])
    dz = np.diff(zs)
    assert np.isclose(zs, 0.0).any() and np.isclose(zs, 0.03).any() and np.isclose(zs, -1.0).any()
    ratio = np.maximum(dz[1:] / dz[:-1], dz[:-1] / dz[1:])
    assert ratio.max() <= r * 1.1, f"neighbouring cells grow by {ratio.max():.3f} > {r}"
    assert np.all(dz[zs[:-1] >= -1e-12] <= 0.01 + 1e-12)  # the film is untouched
    below = dz[zs[1:] <= 1e-12]
    assert below.max() > 0.12 and below.max() <= 0.2 + 1e-12  # coarse again far from the film
    # graded, so more cells below than uniform 0.2 (5), fewer than the film's 0.01 everywhere (100)
    assert 5 < below.size < 40


def test_grading_must_be_a_growth_ratio_above_one():
    with pytest.raises(ValueError, match="grading"):
        _graded_stack().structured(grading=1.0)
    with pytest.raises(ValueError, match="grading"):
        _graded_stack().structured(grading=-2)
