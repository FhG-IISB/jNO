"""Tutorial: lazy polygon CSG domains.

This example uses the Shapely-backed ``jno.PolygonDomain`` class to build a
2D computational domain from ordered polygon vertices. Unlike the historical
mesh-backed ``jno.domain.polygon(...)`` constructor, ``PolygonDomain`` keeps the
geometry analytic and samples points only when ``variable(..., sample=...)`` is
called.

The example constructs a small chamber with an inlet extension and a square
obstacle removed from the interior:

    computational domain = (chamber union inlet) difference obstacle

It then samples interior points, the full boundary, and a named obstacle edge
with exact normal vectors.
"""

from __future__ import annotations

import numpy as np

import jno

INTERIOR_SAMPLES = 256
BOUNDARY_SAMPLES = 96
EDGE_SAMPLES = 64


def build_polygon_csg_domain() -> jno.PolygonDomain:
    """Build a named CSG geometry from ordered vertex loops."""
    chamber = jno.PolygonDomain(
        [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (0.0, 1.0),
        ],
        name="chamber",
    )
    inlet = jno.PolygonDomain(
        [
            (2.0, 0.35),
            (2.5, 0.35),
            (2.5, 0.65),
            (2.0, 0.65),
        ],
        name="inlet",
    )
    obstacle = jno.PolygonDomain(
        [
            (0.8, 0.35),
            (1.2, 0.35),
            (1.2, 0.65),
            (0.8, 0.65),
        ],
        name="obstacle",
    )

    # ``+`` and ``|`` are true geometric union for PolygonDomain objects.
    # ``-`` is geometric difference, producing an analytic hole here.
    return (chamber + inlet) - obstacle


def main() -> None:
    np.random.seed(0)
    domain = build_polygon_csg_domain()

    # PolygonDomain tags are lazy: the geometry exists immediately, but no
    # coordinate arrays are materialized before variable(..., sample=...).
    assert "interior" not in domain.context
    assert "boundary" in domain.boundary_tags()
    assert "boundary_obstacle_0" in domain.boundary_tags()

    # Sample the active CSG region.
    x, y, t = domain.variable("interior", sample=(INTERIOR_SAMPLES, None))
    interior_points = domain.context[x.tag]
    assert interior_points.shape == (1, 1, INTERIOR_SAMPLES, 2)
    assert t.tag == "__time__"

    # Sample all boundary components, including the outer boundary and the hole.
    xb, yb, tb, nx, ny = domain.variable("boundary", sample=(BOUNDARY_SAMPLES, None), normals=True)
    boundary_points = domain.context[xb.tag]
    boundary_normals = domain.context[nx.tag]
    assert boundary_points.shape == (1, 1, BOUNDARY_SAMPLES, 2)
    assert boundary_normals.shape == (1, 1, BOUNDARY_SAMPLES, 2)
    assert tb.tag == "__time__"
    assert ny.tag == nx.tag

    # Source edge tags follow input order. This samples the bottom edge of the
    # obstacle polygon; after subtraction, its normals point into the obstacle
    # hole, i.e. outward from the remaining computational material.
    xo, yo, _, nxo, nyo = domain.variable("boundary_obstacle_0", sample=(EDGE_SAMPLES, None), normals=True)
    obstacle_edge_points = domain.context[xo.tag][0, 0]
    obstacle_edge_normals = domain.context[nxo.tag][0, 0]
    assert obstacle_edge_points.shape == (EDGE_SAMPLES, 2)
    assert obstacle_edge_normals.shape == (EDGE_SAMPLES, 2)
    assert np.allclose(obstacle_edge_points[:, 1], 0.35, atol=1e-12)
    assert np.allclose(obstacle_edge_normals, np.array([0.0, 1.0]), atol=1e-7)
    assert nyo.tag == nxo.tag

    # Named source regions remain available. The chamber region is clipped by
    # the active CSG result, so the obstacle hole is excluded automatically.
    xc, yc, _ = domain.variable("interior_chamber", sample=(64, None))
    assert domain.context[xc.tag].shape == (1, 1, 64, 2)
    assert yc.tag == xc.tag

    print("PolygonDomain CSG tutorial completed")
    print(f"  interior: {interior_points.shape}")
    print(f"  boundary: {boundary_points.shape}")
    print(f"  obstacle edge normals: {obstacle_edge_normals[:3]}")


if __name__ == "__main__":
    main()
