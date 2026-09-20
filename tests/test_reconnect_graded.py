"""``alpha_reconnect`` takes a length scale PER NODE, so it works on a graded mesh.

The filter keeps a triangle when its circumradius is below ``alpha * h``. With one global ``h`` a
graded mesh is wrong at both ends: in the coarse region every cell exceeds the threshold and its
nodes come back as "free particles", and in the fine region the coarse value fuses surfaces that are
genuinely apart. Both of the thresholds the filter applies -- the circumradius test and the
long-edge/short-pair node management -- are per-cell or per-edge quantities already, so they take a
per-node field directly.

This is what a weld needs. Two 1.6 mm rod ends meeting across a 40 um gap cannot be meshed uniformly:
resolving the gap everywhere costs ~15000 nodes, while a mesh coarse enough to afford has a filter
threshold six times the gap and fuses the joint on contact with the mesh rather than with the metal.
Graded -- 50 um at the joint, 500 um down the rod -- it is 363 nodes and the rods stay apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from jno.utils.solver.reconnect import alpha_reconnect


def _graded_block(x0, x1, h_fine, h_coarse, fine_at):
    """A block of points whose spacing grows smoothly from ``h_fine`` (at ``fine_at``) to ``h_coarse``.

    Structured rather than random so the Delaunay triangulation is well conditioned and the test is
    about the length-scale field, not about a pathological point cloud.
    """
    xs, x = [], float(x0)
    while x <= x1 + 1e-12:
        xs.append(x)
        t = min(1.0, abs(x - fine_at) / 1.0)
        x += h_fine + (h_coarse - h_fine) * t
    pts = []
    for xv in xs:
        t = min(1.0, abs(xv - fine_at) / 1.0)
        hy = h_fine + (h_coarse - h_fine) * t
        ny = max(2, int(round(1.0 / hy)))
        for yv in np.linspace(-0.5, 0.5, ny + 1):
            pts.append((xv, yv))
    pts = np.asarray(pts)
    t = np.minimum(1.0, np.abs(pts[:, 0] - fine_at) / 1.0)
    return pts, h_fine + (h_coarse - h_fine) * t


def _two_blocks(gap, h_fine=0.05, h_coarse=0.25):
    """Two graded blocks facing each other across ``gap``, fine on the facing sides."""
    a, ha = _graded_block(-1.0 - gap / 2, -gap / 2, h_fine, h_coarse, -gap / 2)
    b, hb = _graded_block(gap / 2, 1.0 + gap / 2, h_fine, h_coarse, gap / 2)
    return np.vstack([a, b]), np.concatenate([ha, hb])


def _n_bodies(pts, cells):
    n = len(pts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in cells:
        r = find(int(tri[0]))
        for v in tri[1:]:
            rv = find(int(v))
            if rv != r:
                parent[rv] = r
    return len({find(int(v)) for tri in cells for v in tri})


def test_a_scalar_h_cannot_serve_a_graded_mesh_but_a_per_node_field_can():
    """The scalar mean strands the coarse nodes; the per-node field triangulates every one."""
    pts, h = _two_blocks(gap=0.30)
    with pytest.raises(ValueError, match="free particles"):
        alpha_reconnect(pts, float(h.mean()), 0.8, manage=False)

    _, cells, _ = alpha_reconnect(pts, h, 0.8, manage=False)
    used = np.zeros(len(pts), dtype=bool)
    used[np.asarray(cells).reshape(-1)] = True
    assert used.all(), f"{int((~used).sum())} nodes left out of the triangulation"


def test_the_per_node_field_keeps_bodies_apart_that_a_coarse_scalar_would_fuse():
    """A gap wider than the LOCAL threshold must survive, however coarse the rest of the mesh is."""
    pts, h = _two_blocks(gap=0.30)
    _, cells, _ = alpha_reconnect(pts, h, 0.8, manage=False)
    assert _n_bodies(pts, cells) == 2, "the filter bridged a gap well beyond its local threshold"


def test_node_management_carries_the_length_scale_onto_the_new_nodes():
    """An inserted midpoint must inherit a scale, or the next filter pass indexes a stale field."""
    pts, h = _two_blocks(gap=0.30)
    moved, cells, _ = alpha_reconnect(pts, h, 0.8, manage=True)
    used = np.zeros(len(moved), dtype=bool)
    used[np.asarray(cells).reshape(-1)] = True
    assert used.all(), "management left nodes out of the triangulation"
    assert _n_bodies(moved, cells) == 2, "management fused the two bodies"


def test_a_mismatched_field_is_refused_by_name():
    pts, h = _two_blocks(gap=0.30)
    with pytest.raises(ValueError, match="(?i)per point"):
        alpha_reconnect(pts, h[:-3], 0.8, manage=False)
