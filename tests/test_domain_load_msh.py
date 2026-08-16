"""Loading a mesh built somewhere else — deriving its regions, and refusing what jNO cannot assemble.

`jno.domain("part.msh")` used to load a file into a domain you could not write an equation against:
an external mesh carries gmsh *physical groups* under whatever names its author chose, or nothing at
all, so `interior` and `boundary` — the two tags every weak form needs — simply did not exist. This
was measured on 13 real files from gmsh's own benchmark suite and meshio's corpus: eight loaded and
none could solve.

These tests generate their meshes rather than shipping fixtures, but they mirror what those files
actually contain: physical groups under arbitrary names, no surface block at all, an internal void,
mixed cell types, and a surface mesh with no volume. The acceptance criterion is a SOLVE, not a
non-empty tag dict.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

meshio = pytest.importorskip("meshio")
from jno.domain.geometries import Geometries  # noqa: E402


def _write(tmp, name, mesh, fmt=None):
    p = tmp / name
    meshio.write(str(p), mesh, file_format=fmt) if fmt else meshio.write(str(p), mesh)
    return str(p)


def _blocks(mesh):
    return {c.type: np.asarray(c.data) for c in mesh.cells}


def _poisson(d):
    """-Δu = 1 with u = 0 on the derived boundary. The point of the whole feature."""
    dim = int(d.dimension)
    u, v = d.fem_symbols()
    ci = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    ui = u.bind(**dict(zip("xyz", ci[:dim])))
    vi = v.bind(**dict(zip("xyz", ci[:dim])))
    lap = ui.x * vi.x + ui.y * vi.y + (ui.z * vi.z if dim == 3 else 0.0 * ui * vi)
    return np.asarray(jno.fem([lap - 1.0 * vi, u(*cb[:dim]) - 0.0]).solve()).ravel()


@pytest.fixture(scope="module")
def bare_2d(tmp_path_factory):
    """A triangle mesh with NO cell_sets whatsoever — the commonest external file."""
    m, _, _ = Geometries.equi_distant_rect(nx=4, ny=4)(None)
    tmp = tmp_path_factory.mktemp("bare")
    plain = meshio.Mesh(points=m.points, cells=[("triangle", _blocks(m)["triangle"])])
    return _write(tmp, "bare.inp", plain, "abaqus")


# ------------------------------------------------------------------- the tags are derived, and work


def test_a_tagless_mesh_gains_interior_and_boundary(bare_2d):
    d = jno.domain(bare_2d, compute_mesh_connectivity=False)
    assert "interior" in d.tag_indices and "boundary" in d.tag_indices
    assert len(np.atleast_1d(d.tag_indices["interior"])) == len(d.mesh.points)
    assert len(np.atleast_1d(d.tag_indices["boundary"])) == 16  # a 4x4 grid's perimeter nodes


def test_a_tagless_mesh_solves(bare_2d):
    """The acceptance criterion: a file jNO has never seen supports a weak form with no setup."""
    sol = _poisson(jno.domain(bare_2d, compute_mesh_connectivity=False))
    assert np.isfinite(sol).all()
    assert sol.max() > 0 and np.isclose(sol.min(), 0.0, atol=1e-12)  # Dirichlet held on the boundary


def test_derived_tags_are_identical_to_native_ones(bare_2d):
    """A derived tag must not be second-class. Same geometry built both ways, same node sets —
    which is what makes it safe to feed the derivation through the existing tag machinery rather
    than teaching every consumer about a new kind of tag."""
    native = jno.domain(constructor=Geometries.equi_distant_rect(nx=4, ny=4), compute_mesh_connectivity=False)
    derived = jno.domain(bare_2d, compute_mesh_connectivity=False)
    for tag in ("interior", "boundary"):
        a = np.sort(np.atleast_1d(native.tag_indices[tag]))
        b = np.sort(np.atleast_1d(derived.tag_indices[tag]))
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- what the file says wins


def test_a_tag_the_file_defines_is_never_overridden(tmp_path):
    """A physical group genuinely named `boundary` keeps its own meaning — here deliberately only
    two edges of sixteen. Only the MISSING name is derived."""
    m, _, _ = Geometries.equi_distant_rect(nx=4, ny=4)(None)
    b = _blocks(m)
    own = meshio.Mesh(
        points=m.points,
        cells=[("triangle", b["triangle"]), ("line", b["line"])],
        cell_sets={"boundary": [np.array([], dtype=np.int64), np.array([0, 1], dtype=np.int64)]},
    )
    d = jno.domain(_write(tmp_path, "own.inp", own, "abaqus"), compute_mesh_connectivity=False)
    assert len(np.atleast_1d(d.tag_indices["boundary"])) == 3, "the file's own boundary was overwritten"
    assert "interior" in d.tag_indices, "the missing tag should still be derived"


def test_physical_groups_survive_derivation(tmp_path):
    """Named groups are the reason to load an external mesh at all; deriving must not disturb them."""
    m, _, _ = Geometries.equi_distant_rect(nx=4, ny=4)(None)
    tri = _blocks(m)["triangle"]
    named = meshio.Mesh(
        points=m.points,
        cells=[("triangle", tri)],
        cell_sets={"steel": [np.arange(10, dtype=np.int64)], "copper": [np.arange(10, len(tri), dtype=np.int64)]},
    )
    d = jno.domain(_write(tmp_path, "named.inp", named, "abaqus"), compute_mesh_connectivity=False)
    assert {"steel", "copper", "interior", "boundary"} <= set(d.avaiable_mesh_tags)
    assert len(np.atleast_1d(d.tag_indices["steel"])) > 0


# ------------------------------------------------------------------------ the boundary can be split


def test_an_internal_void_becomes_its_own_boundary_tag(tmp_path):
    """Connected components of the boundary, so a cavity can be addressed separately from the outer
    surface. Measured on a box with a spherical void: two shells, the inner much smaller."""
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.add("void")
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        ball = gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.25)
        gmsh.model.occ.cut([(3, box)], [(3, ball)])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        p = str(tmp_path / "void.msh")
        gmsh.write(p)
    finally:
        gmsh.finalize()

    d = jno.domain(p, compute_mesh_connectivity=False)
    shells = sorted(t for t in d.avaiable_mesh_tags if t.startswith("boundary_"))
    assert len(shells) == 2, f"expected an outer surface and a cavity, got {shells}"
    sizes = sorted(len(np.atleast_1d(d.tag_indices[t])) for t in shells)
    assert sizes[0] < sizes[1], "the cavity should be the smaller shell"
    assert sizes[0] + sizes[1] == len(np.atleast_1d(d.tag_indices["boundary"]))


def test_a_solid_part_gains_no_numbered_shells(bare_2d):
    """One component means the numbered tags would be pure noise, so they are not emitted."""
    d = jno.domain(bare_2d, compute_mesh_connectivity=False)
    assert not [t for t in d.avaiable_mesh_tags if t.startswith("boundary_")]


# ------------------------------------------------------------------------------------- the refusals


def test_a_mixed_cell_mesh_is_refused_with_its_numbers(tmp_path):
    """jNO assembles on one element family and would otherwise take the first block it recognises —
    measured on a real file as 70 % of the domain, silently. The refusal names what would be lost."""
    mb, _, _ = Geometries.equi_distant_box(nx=2, ny=2, nz=2)(None)
    qb, _, _ = Geometries.equi_distant_box(nx=2, ny=2, nz=2, cell="hex")(None)
    mixed = meshio.Mesh(
        points=mb.points,
        cells=[("tetra", _blocks(mb)["tetra"]), ("hexahedron", _blocks(qb)["hexahedron"])],
    )
    p = _write(tmp_path, "mixed.vtu", mixed)
    with pytest.raises(NotImplementedError, match="mixes cell types"):
        jno.domain(p, compute_mesh_connectivity=False)
    with pytest.raises(NotImplementedError, match="% of the domain"):
        jno.domain(p, compute_mesh_connectivity=False)


def test_a_surface_mesh_is_refused_as_a_shell(tmp_path):
    """Triangles in 3-D with no volume behind them — a routine CAD/STL export. This used to die
    inside numpy with `zero-size array to reduction`, which says nothing about the file."""
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([np.cos(ang), np.sin(ang), np.linspace(0.0, 1.0, 12)])
    tri = np.array([[i, (i + 1) % 12, (i + 2) % 12] for i in range(12)])
    shell = meshio.Mesh(points=pts, cells=[("triangle", tri)])
    with pytest.raises(NotImplementedError, match="SURFACE"):
        jno.domain(_write(tmp_path, "shell.vtu", shell), compute_mesh_connectivity=False)


# ---------------------------------------------------------------------------------- 3-D end to end


def test_a_tagless_3d_mesh_solves(tmp_path):
    """The 3-D case, which is where external meshes actually come from."""
    m, _, _ = Geometries.equi_distant_box(nx=3, ny=3, nz=3)(None)
    plain = meshio.Mesh(points=m.points, cells=[("tetra", _blocks(m)["tetra"])])
    d = jno.domain(_write(tmp_path, "solid.vtu", plain), compute_mesh_connectivity=False)
    assert int(d.dimension) == 3
    assert len(np.atleast_1d(d.tag_indices["interior"])) == len(m.points)
    sol = _poisson(d)
    assert np.isfinite(sol).all() and sol.max() > 0
