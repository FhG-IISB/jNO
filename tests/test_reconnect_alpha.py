"""Alpha-shape reconnection: two bodies whose gap closes become one mesh, and the nodes stay usable.

Merging oracle: two unit squares of lattice nodes (spacing h), side by side with a gap g. A triangle
bridging the gap is right-angled with legs g and h, so its circumradius is sqrt(g^2 + h^2)/2, and it passes
the filter ``R < alpha h`` exactly when ``g < h sqrt(4 alpha^2 - 1)`` (0.218 for h = 0.1, alpha = 1.2).
Merged, the two inner walls (10 edges each) leave the boundary and the two gap edges join it:
80 - 20 + 2 = 62 edges.

The filter tests pass ``manage=False`` to isolate it. Node management is tested on its own below:
re-triangulating cannot fix a bad point DISTRIBUTION, since Delaunay already maximises the minimum angle
for the points it is given, so nodes have to be inserted and dropped as the body deforms.
"""

import numpy as np
import pytest

from jno.utils.solver.reconnect import alpha_reconnect, n_components

H, ALPHA = 0.1, 1.2


def _two_squares(gap):
    g = np.linspace(0.0, 1.0, 11)
    a = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    return np.concatenate([a, a + [1.0 + gap, 0.0]])


def _lattice(n, spacing):
    g = np.arange(n) * spacing
    return np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)


def _edges(X, cells):
    e = np.unique(np.sort(np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1), axis=0)
    return np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)


def _area(X, cells):
    t = X[cells]
    a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return float(np.abs(0.5 * np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])))


@pytest.mark.parametrize("gap, bodies, edges", [(0.30, 2, 80), (0.15, 1, 62)])
def test_two_bodies_merge_exactly_when_the_gap_closes(gap, bodies, edges):
    X = _two_squares(gap)
    _pts, cells, bnd = alpha_reconnect(X, H, ALPHA, manage=False)
    assert n_components(len(X), cells) == bodies
    assert len(bnd) == edges


def test_the_threshold_is_where_the_bridging_circumradius_says():
    g_star = H * np.sqrt(4.0 * ALPHA**2 - 1.0)
    for factor, bodies in ((0.98, 1), (1.02, 2)):
        X = _two_squares(factor * g_star)
        assert n_components(len(X), alpha_reconnect(X, H, ALPHA, manage=False)[1]) == bodies


def test_the_nodes_are_kept_and_the_cells_are_counter_clockwise():
    X = _two_squares(0.15)
    pts, cells, _ = alpha_reconnect(X, H, ALPHA, manage=False)
    assert pts is X or np.array_equal(pts, X)  # `manage=False` touches no node
    assert np.array_equal(np.unique(cells), np.arange(len(X)))  # every node, same numbering
    t = X[cells]
    p, q = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    area = 0.5 * (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0])
    assert (area > 0).all()
    assert area.sum() == pytest.approx(2.0 + 0.15, rel=1e-12)  # the two squares plus the bridged gap


def test_hysteresis_holds_a_bridge_that_a_fresh_filter_would_drop():
    """A march re-decides the triangulation every step, so a cell sitting near ``alpha h`` would drop out
    and come back -- the surface flickering wedges in and out. An existing cell is therefore held to a
    wider threshold: here the bodies stay joined as the gap widens past the merge threshold."""
    g_star = H * np.sqrt(4.0 * ALPHA**2 - 1.0)
    merged = alpha_reconnect(_two_squares(0.98 * g_star), H, ALPHA, manage=False)[1]
    wider = _two_squares(1.05 * g_star)  # a gap a fresh filter would NOT bridge
    assert n_components(len(wider), alpha_reconnect(wider, H, ALPHA, manage=False)[1]) == 2
    held = alpha_reconnect(wider, H, ALPHA, previous=merged, manage=False)[1]
    assert n_components(len(wider), held) == 1, "hysteresis did not hold the existing bridge"


def test_hysteresis_still_lets_a_far_cell_go():
    """It widens the threshold; it does not disable it. A gap well beyond `hysteresis * alpha * h` splits."""
    far = _two_squares(4.0 * H * np.sqrt(4.0 * ALPHA**2 - 1.0))
    merged = alpha_reconnect(_two_squares(0.5 * H), H, ALPHA, manage=False)[1]
    assert n_components(len(far), alpha_reconnect(far, H, ALPHA, previous=merged, manage=False)[1]) == 2


def test_a_hysteresis_below_one_is_refused():
    with pytest.raises(ValueError, match="must be >= 1"):
        alpha_reconnect(_two_squares(0.15), H, ALPHA, hysteresis=0.9)


def test_a_stretched_body_gains_nodes_on_its_long_edges():
    """A body that has stretched has edges longer than the mesh it was built with; they gain midpoints,
    and the longest edge comes down. Delaunay alone cannot do this -- it only reconnects what is there."""
    X = _lattice(6, 1.6 * H)  # every edge 1.6 h: too long, but still inside the alpha filter
    before = alpha_reconnect(X, H, ALPHA, manage=False)
    after = alpha_reconnect(X, H, ALPHA)
    n_long = lambda pts, cells: int((_edges(pts, cells) > 1.5 * H).sum())  # noqa: E731
    assert len(after[0]) > len(X), "no nodes were inserted"
    assert n_long(*after[:2]) < n_long(X, before[1]), "the long edges were not reduced"
    assert _area(*after[:2]) == pytest.approx(_area(X, before[1]), rel=1e-12), "insertion moved the outline"
    # Insertion is CAPPED per call (`max_growth`), so one pass need not clear every long edge -- a
    # stretching body must not be allowed to grow its mesh without bound. Lifting the cap clears them.
    once = alpha_reconnect(X, H, ALPHA, max_growth=10.0)
    assert n_long(*once[:2]) == 0, "with the cap lifted, long edges should all be split"


def test_a_crowded_interior_node_is_dropped_and_the_outline_is_not():
    """Nodes that pile up cost DOFs and make slivers, so an interior one goes. A boundary node never does:
    removing it would cut a corner off the body and lose liquid."""
    X = _lattice(6, H)
    crowd = np.concatenate([X, [X[7] + [0.25 * H, 0.15 * H]]])  # an interior node's near-twin
    kept = alpha_reconnect(crowd, H, ALPHA, long_f=1e9)[0]  # insertion off, removal only
    assert len(kept) < len(crowd), "the crowding node was kept"
    outline = X[(X[:, 0] == 0) | (X[:, 1] == 0) | (X[:, 0] == 5 * H) | (X[:, 1] == 5 * H)]
    for corner in outline:
        assert np.isclose(kept, corner).all(axis=1).any(), f"boundary node {corner} was dropped"


def test_a_free_particle_is_refused():
    X = np.concatenate([_two_squares(0.15), [[5.0, 5.0]]])
    with pytest.raises(ValueError, match="in no triangle"):
        alpha_reconnect(X, H, ALPHA)


def test_3d_is_refused():
    with pytest.raises(NotImplementedError, match="2-D only"):
        alpha_reconnect(np.random.default_rng(0).random((20, 3)), H, ALPHA)
