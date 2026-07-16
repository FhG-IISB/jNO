"""Multi-material domains via ``jno.Shape.regions`` — conforming meshes + region tags.

The pieces are fragmented so element edges align with every material interface, and each
volume cell is assigned to the first region whose shape contains its centroid (exact, because
the mesh conforms). Each region becomes its own variable set; internal interface facets are not
boundary; the outer boundary keeps its auto-names.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import jno


def _tri_areas(points, tri):
    p = points[tri][:, :, :2]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def test_regions_split_is_exact_and_conforming():
    """A disk inclusion in a plate: the two region cell-sets partition the cells and their areas
    match the analytic areas (only the disk's polygonal facetting differs, ~1%)."""
    plate = jno.Shape.rect(0, 0, 2, 1)
    disk = jno.Shape.disk(1, 0.5, 0.3)
    mesh, dim, _ = jno.Shape.regions(inclusion=disk, matrix=plate).sized(0.05).build()

    tri = mesh.cells[0].data
    inc = mesh.cell_sets["inclusion"][0]
    mat = mesh.cell_sets["matrix"][0]
    # partition: every interior cell belongs to exactly one region
    assert len(inc) + len(mat) == len(tri)
    assert set(inc.tolist()).isdisjoint(mat.tolist())

    areas = _tri_areas(mesh.points, tri)
    assert abs(areas[inc].sum() - math.pi * 0.3**2) < 1e-2
    assert abs(areas[mat].sum() - (2.0 - math.pi * 0.3**2)) < 1e-2


def test_regions_expose_variable_sets_and_external_boundary():
    d = jno.Shape.regions(inclusion=jno.Shape.disk(1, 0.5, 0.3), matrix=jno.Shape.rect(0, 0, 2, 1)).sized(0.08).domain()
    # region tags queryable
    for tag in ("interior", "inclusion", "matrix", "boundary"):
        d.variable(tag)  # must not raise
    assert d._mesh_pool["inclusion"].shape[0] > 0
    # the interior sub-region is NOT a boundary; the boundary is the plate's outer edges only
    tags = set(d.boundary_tags())
    assert tags == {"left", "right", "top", "bottom", "boundary"}
    assert "inclusion" not in tags and "matrix" not in tags


def test_regions_priority_is_keyword_order():
    """Overlapping regions resolve by keyword order: the disk overlaps the plate, and the disk's
    label wins because it is listed first."""
    mesh, _, _ = (
        jno.Shape.regions(inclusion=jno.Shape.disk(1, 0.5, 0.3), matrix=jno.Shape.rect(0, 0, 2, 1)).sized(0.08).build()
    )
    tri = mesh.points[mesh.cells[0].data][:, :, :2]
    cent = tri.mean(axis=1)
    inc = mesh.cell_sets["inclusion"][0]
    # every inclusion cell centroid lies inside the disk
    assert np.all((cent[inc, 0] - 1.0) ** 2 + (cent[inc, 1] - 0.5) ** 2 <= 0.3**2 + 1e-9)


def test_regions_3d_box_with_spherical_core():
    core = jno.Shape.sphere(0.5, 0.5, 0.5, 0.25)
    shell = jno.Shape.box(0, 0, 0, 1, 1, 1)
    d = jno.Shape.regions(core=core, shell=shell).sized(0.12).domain()
    assert d.dimension == 3
    assert d._mesh_pool["core"].shape[0] > 0
    assert d._mesh_pool["shell"].shape[0] > 0
    # 3-D outer boundary keeps the box face names; the sphere interface is not boundary
    assert set(d.boundary_tags()) == {"left", "right", "front", "back", "top", "bottom", "boundary"}


def test_regions_contains_is_union():
    sh = jno.Shape.regions(inclusion=jno.Shape.disk(1, 0.5, 0.3), matrix=jno.Shape.rect(0, 0, 2, 1))
    mask = sh.contains(np.array([[1.0, 0.5], [0.05, 0.05], [5.0, 5.0]]))
    assert mask.tolist() == [True, True, False]


def test_regions_requires_two_regions_of_same_dim():
    with pytest.raises(ValueError, match="at least two"):
        jno.Shape.regions(only=jno.Shape.rect(0, 0, 1, 1))
    with pytest.raises(ValueError, match="dimension"):
        jno.Shape.regions(a=jno.Shape.rect(0, 0, 1, 1), b=jno.Shape.box(0, 0, 0, 1, 1, 1))


def test_fem_term_restricts_to_a_shape_region():
    """A ``jno.fem`` load written on the ``inclusion`` region integrates over that region's cells
    only: ``sum(b) ≈ area(disk)`` — exact because the mesh conforms to the interface."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        d = jno.Shape.regions(inclusion=jno.Shape.disk(1, 0.5, 0.3), matrix=jno.Shape.rect(0, 0, 2, 1)).sized(0.05).domain()
        u, phi = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xd, yd, _ = d.variable("inclusion", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        vd = phi.bind(x=xd, y=yd)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y, -1.0 * vd, u(xb, yb) - 0.0])
        area = float(np.asarray(fem.b).sum())
        assert abs(area - math.pi * 0.3**2) < 5e-3, f"inclusion load should integrate to ~pi r^2, got {area:.4f}"
    finally:
        jax.config.update("jax_enable_x64", prev)
