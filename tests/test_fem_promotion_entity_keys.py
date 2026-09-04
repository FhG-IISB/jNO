"""Higher-order node promotion keys on the TOPOLOGICAL ENTITY, not the physical coordinate.

`_promote_to_degree` synthesises a P{k} node mesh from a P1 one and has to decide when two synthesised
nodes are the same node. Keying on the coordinate is the right conformity test for ONE body and the
wrong one for two: a `Shape.regions(..., conforming=False)` interface is coincident *on purpose*, so
every node the promotion added there was merged across the bodies and welded them -- silently. It was
refused outright rather than allowed to be wrong, which is why P2 was unavailable on a non-conforming
domain, and with it Taylor-Hood (P2 velocity / P1 pressure) on independently meshed bodies.

A reference point's non-zero barycentric weights name exactly the P1 vertices spanning the entity it
lies on -- one for a vertex, two for an edge, three for a face. Keying on `(sorted global vertex ids,
the weights in that order)` is therefore orientation-independent (so a shared entity still collapses
from either neighbouring cell) while separating entities that merely coincide in space.

The two properties that matter pull in opposite directions, so both are asserted:

* **conforming meshes must not move.** The promoted node count is exactly `n_vertices + n_edges` for
  P2 on a simplex mesh -- a combinatorial identity that fails if dedup either over- or under-merges.
* **non-conforming bodies must not weld.** Zero nodes shared between the two bodies.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _region_nodes(d, name):
    from jno.utils.solver.fem_utils import _cell_region_mask

    cells = np.asarray(d._fem_native_assembly_cells_all[0])
    return np.unique(cells[np.asarray(_cell_region_mask(d, name)).reshape(-1) > 0])


def _build(d):
    """A trivial P2 vector problem, just to force the promotion and publish the layout."""
    u, phi = d.fem_symbols(value_shape=(2,), names=("u", "phi"), order=2)
    ci = d.variable("interior", split=True)
    ub, zb = u.bind(x=ci[0], y=ci[1]), phi.bind(x=ci[0], y=ci[1])
    ob = d.variable("ob", where=lambda x, y: (x < 1e-9) | (x > 2.0 - 1e-9), split=True)
    jno.fem([ub.x[0] * zb.x[0] + ub.y[0] * zb.y[0] + ub.x[1] * zb.x[1] + ub.y[1] * zb.y[1], u(ob[0], ob[1]) - 0.0])
    return d


def test_a_conforming_p2_mesh_has_exactly_vertices_plus_edges():
    """The identity that catches BOTH failure modes: over-merging loses nodes, under-merging adds them.

    On a simplex mesh a P2 node sits on every vertex and every edge, and nowhere else -- so the promoted
    count is fixed by the topology alone, independent of how dedup is keyed."""
    d = _build(jno.Shape.rect(0.0, 0.0, 2.0, 1.0).sized(0.3).domain())
    p1 = np.asarray(d.built_mesh.points)
    tri = np.asarray(d.built_mesh.cells_dict["triangle"])
    edges = {frozenset((int(a), int(b))) for t in tri for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    n_p2 = len(np.asarray(d._fem_native_dof_points_all[0]))
    assert n_p2 == len(p1) + len(edges), (n_p2, len(p1), len(edges))


def test_two_independently_meshed_bodies_are_not_welded_at_p2():
    """The defect. Before entity keying this reported 37 shared nodes, all on the interface -- harmless
    for a tie, and wrong for contact, where those DOFs could then never separate."""
    d = _build(
        jno.Shape.regions(
            left=jno.Shape.rect(0.0, 0.0, 1.0, 1.0),
            right=jno.Shape.rect(1.0, 0.0, 2.0, 1.0),
            conforming=False,
        )
        .sized(0.34)
        .domain()
    )
    nl, nr = _region_nodes(d, "left"), _region_nodes(d, "right")
    assert len(nl) > 0 and len(nr) > 0
    assert len(np.intersect1d(nl, nr)) == 0, "the two bodies must not share a single node"

    # ...and the coincident interface carries TWO nodes at each location, one per body
    pts = np.asarray(d._fem_native_dof_points_all[0])
    on_if = np.flatnonzero(np.abs(pts[:, 0] - 1.0) < 1e-9)
    assert len(on_if) == 2 * len(np.unique(np.round(pts[on_if, 1], 9))), "each side keeps its own nodes"


def test_a_conforming_interface_still_shares_its_nodes():
    """The other direction, and the one entity keying could plausibly break: `+` composition is
    CONFORMING, so the two regions must still meet on one shared node set."""
    d = _build(
        (
            jno.Shape.rect(0.0, 0.0, 1.0, 1.0).name("left").sized(0.34)
            + jno.Shape.rect(1.0, 0.0, 2.0, 1.0).name("right").sized(0.34)
        ).domain()
    )
    nl, nr = _region_nodes(d, "left"), _region_nodes(d, "right")
    shared = np.intersect1d(nl, nr)
    pts = np.asarray(d._fem_native_dof_points_all[0])
    assert len(shared) > 0, "a conforming interface MUST share nodes"
    assert np.abs(pts[shared, 0] - 1.0).max() < 1e-9, "and they all lie on the interface"


def test_taylor_hood_now_builds_on_independently_meshed_bodies():
    """What the refusal cost: mixed-order elements were unavailable on a non-conforming domain, so
    P2/P1 Taylor-Hood could not be posed there at all -- and P1/P1 is inf-sup unstable."""
    d = (
        jno.Shape.regions(
            left=jno.Shape.rect(0.0, 0.0, 1.0, 1.0),
            right=jno.Shape.rect(1.0, 0.0, 2.0, 1.0),
            conforming=False,
        )
        .sized(0.34)
        .domain()
    )
    v, psi = d.fem_symbols(value_shape=(2,), names=("v", "psi"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    ci = d.variable("interior", split=True)
    vb, wb = v.bind(x=ci[0], y=ci[1]), psi.bind(x=ci[0], y=ci[1])
    pb, qb = p.bind(x=ci[0], y=ci[1]), q.bind(x=ci[0], y=ci[1])
    ob = d.variable("ob", where=lambda x, y: (x < 1e-9) | (x > 2.0 - 1e-9), split=True)
    fem = jno.fem(
        [
            vb.x[0] * wb.x[0] + vb.y[0] * wb.y[0] + vb.x[1] * wb.x[1] + vb.y[1] * wb.y[1] - pb * (wb.x[0] + wb.y[1]),
            qb * (vb.x[0] + vb.y[1]),
            v(ob[0], ob[1]) - 0.0,
        ]
    )
    assert len(fem.offsets) == 3, fem.offsets
    n_v, n_p = len(np.asarray(fem.field_points[0])), len(np.asarray(fem.field_points[1]))
    assert n_v > n_p, "P2 velocity must carry more nodes than P1 pressure"
