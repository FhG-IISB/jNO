"""Tests for general region definition on the Shapely-backed PolygonDomain:

* axis-aligned ``box`` domains auto-register ``left``/``right``/``top``/``bottom``
  boundary tags (so per-edge BCs are addressable, matching ``jno.domain.rect``),
* the auto edges are samplable and produce working FEM location functions,
* the auto edges survive ``build_mesh`` (re-derived from ``_polygon_tags``),
* ``domain.region(name, where=...)`` accepts a predicate, a shapely geometry,
  and a tag alias,
* non-rectangular polygons get no spurious edge tags.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for PolygonDomain regions")
from shapely.geometry import LineString, Polygon, box  # noqa: E402


def _unit_box(mesh_size=None):
    return jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)


def test_box_auto_registers_edge_tags():
    d = _unit_box()
    for tag in ("left", "right", "top", "bottom"):
        assert tag in d.avaiable_mesh_tags
        assert tag in d._boundary_regions
        assert d._make_tag_location_fn(tag) is not None


def test_box_edge_location_fns_are_geometrically_correct():
    d = _unit_box()
    # midpoint of each edge is contained only by that edge's tag
    cases = {
        "left": jnp.array([0.0, 0.5]),
        "right": jnp.array([1.0, 0.5]),
        "bottom": jnp.array([0.5, 0.0]),
        "top": jnp.array([0.5, 1.0]),
    }
    for tag, pt in cases.items():
        fn = d._make_tag_location_fn(tag)
        assert bool(fn(pt)) is True
        for other, opt in cases.items():
            if other != tag:
                assert bool(d._make_tag_location_fn(other)(pt)) is False


def test_box_edges_are_samplable():
    d = _unit_box()
    d.variable("right", split=True)
    pts = np.asarray(d.context["right"])
    assert pts.shape[-1] == 2 and pts.shape[-2] >= 1
    # all sampled points lie on x == 1
    assert np.allclose(pts.reshape(-1, 2)[:, 0], 1.0)


def test_box_edges_survive_build_mesh():
    pytest.importorskip("pygmsh", reason="pygmsh required for build_mesh")
    d = jno.domain(box(0.0, 0.0, 2.0, 1.0))
    d.build_mesh(0.3)
    for tag in ("left", "right", "top", "bottom"):
        assert tag in d.avaiable_mesh_tags
        assert d._make_tag_location_fn(tag) is not None


def test_predicate_region():
    d = _unit_box()
    d.region("right_top", where=lambda x, y: jnp.isclose(x, 1.0) | jnp.isclose(y, 1.0))
    assert "right_top" in d.avaiable_mesh_tags
    fn = d._make_tag_location_fn("right_top")
    assert bool(fn(jnp.array([1.0, 0.5]))) is True  # right edge
    assert bool(fn(jnp.array([0.5, 1.0]))) is True  # top edge
    assert bool(fn(jnp.array([0.0, 0.5]))) is False  # left edge
    assert bool(fn(jnp.array([0.5, 0.0]))) is False  # bottom edge


def test_predicate_region_selecting_nothing_raises():
    d = _unit_box()
    with pytest.raises(ValueError):
        d.region("empty", where=lambda x, y: jnp.isclose(x, 5.0))


def test_geometry_region_and_alias():
    d = jno.domain(box(0.0, 0.0, 2.0, 1.0))
    d.region("diag", where=LineString([(0.0, 0.0), (2.0, 1.0)]))
    assert "diag" in d.avaiable_mesh_tags
    d.region("diag_alias", where="diag")
    assert "diag_alias" in d.avaiable_mesh_tags


def test_non_box_polygon_has_no_edge_tags():
    d = jno.domain(Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]))
    assert not any(t in d.avaiable_mesh_tags for t in ("left", "right", "top", "bottom"))
