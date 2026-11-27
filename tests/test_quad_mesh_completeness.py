"""A quadrilateral mesh covers the whole shape, and cavity radiation can be built on one.

gmsh's Blossom recombination cannot pair the triangles of a surface with an odd count; those triangles used to be left
out of the quad block without a word (a triangle meshed at h = 0.21 kept 0.394 of its 0.45 area). The mesher now meshes
such a plan again at twice the size and splits every cell into quads. The enclosure builder read triangle cells only."""

import numpy as np
import pytest
from jax import config

import jno
from jno.domain.radiation import radiating_faces


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64; the session default is x64-off (tests/conftest.py), so opt in per test."""
    prev = config.jax_enable_x64
    config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        config.update("jax_enable_x64", prev)


def _quad_area(d):
    p = np.asarray(d.mesh.points)[:, :2]
    q = np.asarray(d.mesh.cells_dict["quad"])
    x, y = p[q, 0], p[q, 1]
    return float(np.sum(0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, 1))))


@pytest.mark.parametrize("h", [0.21, 0.17, 0.13])
def test_a_quad_mesh_keeps_every_cell(h):
    d = jno.shape.polygon([(0, 0), (1, 0), (0.3, 0.9)]).quad().sized(h).domain()
    d.variable("interior")
    assert {c.type for c in d.mesh.cells} == {"quad", "line"}
    assert _quad_area(d) == pytest.approx(0.45, rel=1e-12)


def test_a_triangle_build_after_a_quad_build_is_triangles():
    jno.shape.rect(0, 0, 1, 1).quad().sized(0.3).domain().variable("interior")
    d = jno.shape.rect(0, 0, 1, 1).sized(0.3).domain()
    d.variable("interior")
    assert "triangle" in d.mesh.cells_dict and "quad" not in d.mesh.cells_dict


@pytest.mark.parametrize("cell", ["tri", "quad"])
def test_cavity_radiation_on_quads_matches_the_concentric_cylinders(cell, tmp_path, monkeypatch):
    """Two long concentric cylinders: everything the outer one's inside sends reaches the inner one with the share R1/R2."""
    monkeypatch.setenv("JNO_CACHE", str(tmp_path))
    R1, R2, R3, disk = 0.3, 0.6, 0.8, jno.shape.disk
    plan = (
        disk(0, 0, R1).name("core").attach(eps=0.6, kappa=1e9)
        + (disk(0, 0, R3) - disk(0, 0, R2)).name("shell").attach(eps=0.8, kappa=1e9)
        + disk(0, 0, R2).name("gas").attach(kappa=0.0)
    )
    d = (plan.quad() if cell == "quad" else plan).sized(0.04).domain()
    enc = radiating_faces(d).parts[0]
    F, A, tag = np.asarray(enc.view_factor), np.asarray(enc.areas), np.asarray(enc.element_tags)
    shell, core = tag == "shell", tag == "core"
    F21 = float((A[shell, None] * F[np.ix_(shell, core)]).sum() / A[shell].sum())
    F12 = float((A[core, None] * F[np.ix_(core, shell)]).sum() / A[core].sum())
    assert F12 == pytest.approx(1.0, abs=0.02)
    assert F21 == pytest.approx(R1 / R2, abs=0.02)


def _two_parts(quad):
    plan = jno.shape.rect(0, 0, 0.1, 0.2).name("a").attach(k=2.0) + jno.shape.rect(0.1, 0, 0.2, 0.2).name("b").attach(
        k=20.0
    )
    d = (plan.quad() if quad else plan).sized(0.02).domain()
    d.tag("e", lambda x, y: (np.abs(x - 0.1) < 1e-9) & (y > 0.05) & (y < 0.15), region="a")
    return d


@pytest.mark.parametrize("quad", [False, True])
def test_the_faces_of_a_part_inside_the_mesh_are_found_on_quads(quad):
    """The facets between part a and part b: the interface x = 0.1, 0.2 long (on quads these were never found)."""
    d = _two_parts(quad)
    d.variable("interior")
    f = d._region_surface_facets("a")
    assert f.shape[1:] == (2, 2)
    assert np.linalg.norm(f[:, 1] - f[:, 0], axis=1).sum() == pytest.approx(0.2, rel=1e-12)
    assert np.allclose(f[..., 0], 0.1)


def test_a_term_on_a_tag_inside_a_quad_mesh_is_refused_by_name():
    """It used to integrate over nothing (no facet found), then failed with a bare KeyError; the assembler that
    integrates interior faces works on simplices only."""
    d = _two_parts(True)
    T = d.unknown.scalar("T")
    v = T.test()
    x, y, _ = d.variable("interior", split=True)
    xs, ys, _ = d.variable("e", split=True)
    with pytest.raises(NotImplementedError, match="tag inside the mesh .*e:"):
        jno.fem(
            [
                d.k * (T.bind(x=x, y=y).d(x) * v.bind(x=x, y=y).d(x)) - v.bind(x=x, y=y),
                10.0 * T.bind(x=xs, y=ys) * v.bind(x=xs, y=ys),
                T(*d.variable("right", split=True)[:2]) - 0.0,
            ]
        )
