"""A mesh file's named volume regions as a per-region COEFFICIENT.

``domain.by_region({"steel": 16.0, "air": 0.026})`` is the per-region material primitive, and it
resolved a region only from a shapely geometry part, a ``domain.tag`` predicate, or a
``Shape.regions`` sub-region. A mesh loaded from a ``.msh`` keeps its gmsh physical volumes in
``mesh.cell_sets``, which none of those cover -- so the one kind of domain that has materials
declared *in the file* was the one kind that could not name them in a weak form.

That gap is what forces a mesh-backed problem to re-derive materials from coordinate predicates,
which is fragile in a specific way: the predicate is evaluated at cell CENTROIDS for one purpose and
at QUADRATURE POINTS for another, and the two disagree on cells straddling a material boundary. A
``RegionMask`` is one 0/1 value per cell (``fem_nonnodal._cell_masks``), so it has no such split.
"""

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")
import jno  # noqa: E402


def _two_material_square(tmp_path, n=4):
    """Unit square split into a LEFT and a RIGHT named cell region, written to a mesh file."""
    g = np.linspace(0, 1, n + 1)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    tris, left = [], []
    for i in range(n):
        for j in range(n):
            a, b = i * (n + 1) + j, i * (n + 1) + j + 1
            c, d = (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1
            for t in ([a, b, d], [a, d, c]):
                left.append(i < n // 2)
                tris.append(t)
    tris = np.asarray(tris, dtype=np.int64)
    left = np.asarray(left)
    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", tris)],
        cell_sets={
            "left": [np.flatnonzero(left).astype(np.int64)],
            "right": [np.flatnonzero(~left).astype(np.int64)],
        },
    )
    path = tmp_path / "two_material.inp"
    meshio.write(path, mesh, file_format="abaqus")
    return jno.domain(str(path), compute_mesh_connectivity=False)


def _integral_of(d, coeff):
    """int coeff dV over the whole domain, via a load vector that sums to it."""
    _p, q = d.fem_symbols(names=("p", "q"))
    ai = d.variable("interior", split=True)
    fem = jno.fem([_p.bind(x=ai[0], y=ai[1]) * q.bind(x=ai[0], y=ai[1]) - coeff * q.bind(x=ai[0], y=ai[1])])
    return float(np.asarray(jno.np.asarray(fem.b)).sum())


def test_the_fixture_has_the_regions():
    d = _two_material_square(pytest.importorskip("pathlib") and __import__("pathlib").Path("/tmp"))
    assert {"left", "right"} <= set(d.avaiable_mesh_tags)


def test_by_region_accepts_a_mesh_file_region(tmp_path):
    d = _two_material_square(tmp_path)
    k = d.by_region({"left": 3.0, "right": 1.0})
    assert k is not None


def test_the_per_region_coefficient_integrates_correctly(tmp_path):
    """Each region is half the unit square, so int k dV = 3*0.5 + 1*0.5 = 2."""
    d = _two_material_square(tmp_path)
    total = _integral_of(d, d.by_region({"left": 3.0, "right": 1.0}))
    assert np.isclose(total, 2.0, rtol=1e-9), f"int k dV = {total}, expected 2.0"


def test_overlapping_regions_raise_instead_of_double_counting(tmp_path):
    """`by_region` desugars to sum_r RegionMask(r) * value[r], so two keys covering the same cell add
    together -- and `default` is subtracted twice over. gmsh groups legitimately nest, so this must
    be caught rather than summed."""
    d = _two_material_square(tmp_path)
    d.mesh.cell_sets["all"] = [np.arange(len(d.mesh.cells_dict["triangle"]), dtype=np.int64)]
    with pytest.raises(ValueError, match="overlap"):
        d.by_region({"left": 3.0, "all": 1.0})


def test_a_tag_predicate_of_the_same_name_still_wins(tmp_path):
    """Mesh regions resolve LAST, so nothing that worked before changes behaviour."""
    d = _two_material_square(tmp_path)
    d.tag("left", lambda x, y: x > 0.5)  # deliberately the opposite half
    total = _integral_of(d, d.by_region({"left": 1.0}))
    assert np.isclose(total, 0.5, rtol=1e-9), f"the tag predicate did not win: {total}"


def _two_material_cube(tmp_path):
    """A cube split into a LOWER and an UPPER named cell region, as a 3-D mesh file."""
    pts, tets, lower = [], [], []
    idx = {}
    n = 2
    g = np.linspace(0.0, 1.0, n + 1)
    for i, xv in enumerate(g):
        for j, yv in enumerate(g):
            for k, zv in enumerate(g):
                idx[(i, j, k)] = len(pts)
                pts.append([xv, yv, zv])
    # 6 tets per cube cell (the standard Kuhn subdivision)
    KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 4, 5, 7), (0, 4, 6, 7), (0, 2, 6, 7), (0, 2, 3, 7)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [idx[(i + (v & 1), j + ((v >> 1) & 1), k + ((v >> 2) & 1))] for v in range(8)]
                for t in KUHN:
                    tets.append([c[t[0]], c[t[1]], c[t[2]], c[t[3]]])
                    lower.append(k < n // 2)
    tets, lower = np.asarray(tets, dtype=np.int64), np.asarray(lower)
    mesh = meshio.Mesh(
        points=np.asarray(pts, dtype=float),
        cells=[("tetra", tets)],
        cell_sets={
            "lower": [np.flatnonzero(lower).astype(np.int64)],
            "upper": [np.flatnonzero(~lower).astype(np.int64)],
        },
    )
    path = tmp_path / "cube.inp"
    meshio.write(path, mesh, file_format="abaqus")
    return jno.domain(str(path), compute_mesh_connectivity=False)


def test_a_mesh_region_coefficient_works_on_a_non_nodal_n1e_form(tmp_path):
    """The non-nodal assembler threads `RegionMask` through `_cell_masks`, one 0/1 per cell -- which
    is why this is the right mechanism for an edge-element form and a P0 parameter is not (that path
    has no cell-field branch and would read a per-cell array as P1 vertex data)."""
    inner, vec = jno.np.inner, jno.np.vector
    d = _two_material_cube(tmp_path)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    nu = d.by_region({"lower": 2.0, "upper": 1.0})
    fem = jno.fem(
        [
            nu * inner(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
            + inner(u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z))
            - inner(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), v.bind(x=x, y=y, z=z)),
            u.vector.cross(d.variable("boundary", normals=True)),
        ]
    )
    assert fem.A is not None


def test_a_region_touching_the_outer_boundary_still_works(tmp_path):
    """Both regions of the fixture reach the outer boundary, so each also registers as a BOUNDARY
    region -- after which the shared classifier reads a *volume term* on it as a face-less surface
    term assembling to zero. A `RegionMask` coefficient is not a region-restricted integration
    domain and must be immune, and the outer region of any real device touches the box."""
    d = _two_material_square(tmp_path)
    assert np.isclose(_integral_of(d, d.by_region({"left": 1.0})), 0.5, rtol=1e-9)
    assert np.isclose(_integral_of(d, d.by_region({"right": 1.0})), 0.5, rtol=1e-9)


def test_the_volume_term_guard_still_refuses_domain_variable(tmp_path):
    """`by_region` is the route that works; `d.variable("<mesh region>")` in a VOLUME term stays
    refused. The two must not collide -- RegionMask names are collected separately from coordinate
    variable tags -- and the guard's advice is only honest while the coefficient route exists."""
    d = _two_material_square(tmp_path)
    p, q = d.fem_symbols(names=("p", "q"))
    ai = d.variable("interior", split=True)
    r = d.variable("left", split=True)
    with pytest.raises(ValueError, match="volume term on mesh cell region"):
        jno.fem([p.bind(x=ai[0], y=ai[1]) * q.bind(x=ai[0], y=ai[1]), -1.0 * q.bind(x=r[0], y=r[1])])


def test_attach_works_on_a_mesh_file_region(tmp_path):
    """`domain.attach` documents itself as the way to give properties to a mesh-file domain, "which
    has no Shape to declare them on" -- but its target classifier enumerated the same three sources,
    so it raised `unknown target` on exactly that case. A mesh region owns cells by definition and is
    classified `volume` directly: routing it through the facets-vs-cells test would call the outer
    region of any real device ambiguous, since it reaches the boundary."""
    d = _two_material_square(tmp_path)
    d.attach("left", k=3.0)
    d.attach("right", k=1.0)
    assert np.isclose(_integral_of(d, d.k), 2.0, rtol=1e-9), "int k dV over the unit square"


def test_attach_still_refuses_an_unknown_target(tmp_path):
    d = _two_material_square(tmp_path)
    with pytest.raises(ValueError, match="unknown target"):
        d.attach("no_such_region", k=1.0)
