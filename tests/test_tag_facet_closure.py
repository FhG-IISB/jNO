"""A predicate tag's NODES are the closure of the facets it selected.

The tag picks its boundary facets by the predicate at facet centroids; node-based uses (pinning an
edge needs both its nodes) re-asked the predicate at the vertices. Where they disagree -- a predicate
written to keep a neighbouring face out, on a wall that MEETS that face -- the junction vertices were
dropped, and every edge along the junction went unpinned without a word (measured: 8 % in an
inductance on a mirror-cell eddy problem).

The closure is for FACET-based uses (`closure=True`, what the non-nodal path asks for). A node-based
use keeps the predicate at the vertices, so a nodal Dirichlet value or tie can still exclude a corner."""

import numpy as np

import jno


def test_tag_nodes_include_the_closure_of_its_facets():
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.34).domain()
    d.tag("walls_and_top", lambda x, y, z: z > 1e-9)  # everything but the bottom face
    P = np.asarray(d.mesh.points)
    m = np.asarray(d.tag_node_mask("walls_and_top", P, closure=True), bool)
    nodal = np.asarray(d.tag_node_mask("walls_and_top", P), bool)
    x, y, z = P.T
    on_side = (np.abs(x) < 1e-9) | (np.abs(x - 1) < 1e-9) | (np.abs(y) < 1e-9) | (np.abs(y - 1) < 1e-9)
    junction = on_side & (np.abs(z) < 1e-9)  # bottom edges of the side walls
    bottom_inside = (np.abs(z) < 1e-9) & ~on_side
    assert junction.any() and bottom_inside.any()
    assert m[junction].all(), "the walls' vertices on the bottom edge belong to the tag"
    assert not m[bottom_inside].any(), "but the bottom face's own nodes do not"
    assert not nodal[junction].any(), "a node-based use keeps the predicate at the vertices: z > 0 excludes them"
