"""A VOLUME term on a named mesh cell region must RAISE, not quietly integrate somewhere else.

``d.variable("<tag>")`` restricts a SURFACE term -- that is how every Dirichlet / Neumann term in
jno.fem works -- and the front end resolves a cell region correctly too: ``d.variable("core")``
returns exactly that region's coordinates. But the volume assembly path never used it, and the term
went one of two ways, both silent:

  * no ``_subregion_id`` (a mesh-file group is not a shapely part or a tag predicate)
    -> falls through to whole-domain and integrates over the ENTIRE mesh;
  * a physical volume also registers a node set, so it can land in ``_boundary_regions``
    -> read as a face-less surface term and assembles to exactly ZERO.

Measured on a four-material mesh before the guard: a term on ``air`` integrated to the full
69,694 mm^3 rather than the air's 66,445, and one on ``core`` produced b = 0. Both look like
answers. Per-region materials belong in the COEFFICIENT -- ``d.by_region({...})``, which takes a
named region of the mesh file and restricts exactly, one value per cell.
"""

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")
import jno  # noqa: E402


def _two_material_mesh(tmp_path):
    """Unit square, 8 triangles, split into two named cell regions -- the minimum that reproduces it."""
    gx, gy = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3), indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    tris = []
    for i in range(2):
        for j in range(2):
            a, b = i * 3 + j, i * 3 + j + 1
            c, d = (i + 1) * 3 + j, (i + 1) * 3 + j + 1
            tris += [[a, b, d], [a, d, c]]
    tris = np.asarray(tris, dtype=np.int64)
    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", tris)],
        cell_sets={
            "steel": [np.arange(0, 4, dtype=np.int64)],
            "copper": [np.arange(4, len(tris), dtype=np.int64)],
        },
    )
    path = tmp_path / "two_material.inp"
    meshio.write(path, mesh, file_format="abaqus")
    return jno.domain(str(path), compute_mesh_connectivity=False)


def _mass_plus_load_on(d, region):
    p, q = d.fem_symbols(names=("p", "q"))
    ai = d.variable("interior", split=True)
    mass = p.bind(x=ai[0], y=ai[1]) * q.bind(x=ai[0], y=ai[1])
    r = d.variable(region, split=True)
    return jno.fem([mass, -1.0 * q.bind(x=r[0], y=r[1])])


def test_volume_term_on_a_named_cell_region_raises(tmp_path):
    d = _two_material_mesh(tmp_path)
    assert {"steel", "copper"} <= set(d.avaiable_mesh_tags)
    for region in ("steel", "copper"):
        with pytest.raises(ValueError, match="volume term on mesh cell region"):
            _mass_plus_load_on(d, region)


def test_the_message_names_the_working_alternative(tmp_path):
    """A guard that only says 'no' costs the reader the afternoon this one cost to find."""
    d = _two_material_mesh(tmp_path)
    with pytest.raises(ValueError) as ei:
        _mass_plus_load_on(d, "steel")
    msg = str(ei.value)
    assert "steel" in msg
    assert "by_region" in msg, "must point at the coefficient route that works"
    assert "COEFFICIENT" in msg, "must say the fix is the coefficient, not the quadrature domain"
    assert "SURFACE" in msg, "must say why the surface case still works, so the asymmetry is not a mystery"


def test_interior_is_unaffected(tmp_path):
    """The whole-domain volume region is the legitimate case and must keep working."""
    d = _two_material_mesh(tmp_path)
    fem = _mass_plus_load_on(d, "interior")
    total = float(np.asarray(jno.np.asarray(fem.b)).sum())
    assert np.isclose(total, 1.0, rtol=1e-9), f"int 1 dV over the unit square = {total}"


def test_surface_terms_still_restrict(tmp_path):
    """`d.variable(tag)` on a boundary is the path that always worked; the guard must not touch it."""
    d = _two_material_mesh(tmp_path)
    p, q = d.fem_symbols(names=("p", "q"))
    ai = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    fem = jno.fem([p.bind(x=ai[0], y=ai[1]) * q.bind(x=ai[0], y=ai[1]), p.bind(x=b[0], y=b[1]) - 1.0])
    assert fem is not None
