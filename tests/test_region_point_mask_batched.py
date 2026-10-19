"""Resolving a geometric boundary region over many nodes stays memory-bounded.

A region test compares a point with every facet of the region. Resolving a Dirichlet region did that for
every boundary node in ONE vmap, so the build held (boundary nodes x region facets) intermediates --
quadratic in the mesh, 6 GB of host memory at 206k DOFs on a 3-D box and the peak of the whole build.
Batched by a byte budget, the answer must not change: the oracle is the plane x = 0 itself.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno
from jno.domain import boundary_region as br


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _box():
    return jno.shape.box(0, 0, 0, 1, 1, 1, size=0.2).domain()


def test_point_chunk_respects_the_budget():
    # Break: a fixed batch size (the old 512 points, whatever the facet count).
    for n_ent in (1, 10, 1_000, 100_000, 10**8):
        c = br.point_chunk(n_ent, budget=1 << 20)
        assert c >= 1
        assert c == 1 or c * n_ent * br._BYTES_PER_PAIR <= 1 << 20


def _square_at_x0(n=8):
    """The unit square on the plane x = 0, as 2 n^2 triangles: a facet region like the assembler's."""
    g = np.linspace(0.0, 1.0, n + 1)
    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = ((0, g[i], g[j]), (0, g[i + 1], g[j]), (0, g[i + 1], g[j + 1]), (0, g[i], g[j + 1]))
            tris += [(a, b, c), (a, c, d)]
    tris = np.asarray(tris, dtype=float)
    return br.BoundaryRegion.from_facets("x0", 3, tris.reshape(-1, 3), tris)


@pytest.mark.parametrize("budget", [1, 40_000, 1 << 30], ids=["one-point-batches", "uneven-batches", "one-batch"])
def test_batched_region_mask_is_the_square(budget):
    # Break: a padding row kept in the result, a batch dropped, or the last batch misaligned.
    region = _square_at_x0()
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.5, 1.5, size=(997, 3))  # a prime count: the last batch is short
    pts[::3, 0] = 0.0  # every third point on the plane
    mask = br.map_points(region.contains, pts, region.n_entities, budget=budget).astype(bool)
    inside = (pts[:, 0] == 0.0) & np.all((pts[:, 1:] >= 0.0) & (pts[:, 1:] <= 1.0), axis=1)
    assert inside.sum() > 50
    np.testing.assert_array_equal(mask, inside)


def test_a_dirichlet_region_resolves_in_bounded_batches(monkeypatch):
    # Break: the assembler going back to one vmap over every boundary node (map_points not used, or used
    # with the whole node set as one batch).
    batches = []
    real = br.map_points

    def spy(fn, pts, n_entities, **kw):
        batches.append((len(pts), br.point_chunk(n_entities, kw.get("budget"))))
        return real(fn, pts, n_entities, **kw)

    monkeypatch.setattr(br, "map_points", spy)
    monkeypatch.setattr(br, "_POINT_BUDGET_BYTES", 64 * 1024)  # force several batches on this small mesh
    d = _box()
    x, y, z = d.variable("interior", split=True)[:3]
    xl, yl, zl = d.variable("left", split=True)[:3]
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    fem = jno.fem([ub.x * vb.x + ub.y * vb.y + ub.z * vb.z - vb, u(xl, yl, zl) - 0.0])
    assert batches and all(chunk < n for n, chunk in batches), batches
    sol = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(d.mesh.points)[:, :3]
    on_left = np.abs(pts[:, 0]) < 1e-9
    assert on_left.any() and np.all(sol[on_left] == 0.0)  # pinned exactly where x = 0 ...
    assert np.all(sol[~on_left] > 0.0)  # ... and nowhere else: -lap u = 1 with u = 0 only on that face
