"""Alpha-shape reconnection: two bodies whose gap closes become one mesh, on the same nodes.

Oracle: two unit squares of lattice nodes (spacing h), side by side with a gap g. A triangle bridging the
gap is right-angled with legs g and h, so its circumradius is sqrt(g^2 + h^2)/2, and it passes the filter
``R < alpha h`` exactly when ``g < h sqrt(4 alpha^2 - 1)`` (0.218 for h = 0.1, alpha = 1.2). Merged, the
two inner walls (10 edges each) leave the boundary and the two gap edges join it: 80 - 20 + 2 = 62 edges.
"""

import numpy as np
import pytest

from jno.utils.solver.reconnect import alpha_reconnect, n_components

H, ALPHA = 0.1, 1.2


def _two_squares(gap):
    g = np.linspace(0.0, 1.0, 11)
    a = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    return np.concatenate([a, a + [1.0 + gap, 0.0]])


@pytest.mark.parametrize("gap, bodies, edges", [(0.30, 2, 80), (0.15, 1, 62)])
def test_two_bodies_merge_exactly_when_the_gap_closes(gap, bodies, edges):
    X = _two_squares(gap)
    cells, bnd = alpha_reconnect(X, H, ALPHA)
    assert n_components(len(X), cells) == bodies
    assert len(bnd) == edges


def test_the_threshold_is_where_the_bridging_circumradius_says():
    g_star = H * np.sqrt(4.0 * ALPHA**2 - 1.0)
    assert n_components(242, alpha_reconnect(_two_squares(0.98 * g_star), H, ALPHA)[0]) == 1
    assert n_components(242, alpha_reconnect(_two_squares(1.02 * g_star), H, ALPHA)[0]) == 2


def test_the_nodes_are_kept_and_the_cells_are_counter_clockwise():
    X = _two_squares(0.15)
    cells, _ = alpha_reconnect(X, H, ALPHA)
    assert np.array_equal(np.unique(cells), np.arange(len(X)))  # every node, same numbering
    t = X[cells]
    p, q = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    area = 0.5 * (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0])
    assert (area > 0).all()
    assert area.sum() == pytest.approx(2.0 + 0.15, rel=1e-12)  # the two squares plus the bridged gap


def test_a_free_particle_is_refused():
    X = np.concatenate([_two_squares(0.15), [[5.0, 5.0]]])
    with pytest.raises(ValueError, match="in no triangle"):
        alpha_reconnect(X, H, ALPHA)


def test_3d_is_refused():
    with pytest.raises(NotImplementedError, match="2-D only"):
        alpha_reconnect(np.random.default_rng(0).random((20, 3)), H, ALPHA)
