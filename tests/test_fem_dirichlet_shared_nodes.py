"""Two Dirichlet conditions that pin the same node.

Where two boundary regions meet -- the corner shared by `right` and `top` -- both conditions list the
corner node. Each listing used to add its own unit diagonal to the pinned row (duplicate BCOO entries
are summed) while the load kept only the last value, so the corner was solved as `2 u = g`: half the
prescribed value, with nothing raised. The exact answers below are constant fields, so any corner
error shows up directly.
"""

import numpy as np
import pytest

import jno


def _square():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).sized(0.25).domain()
    u, v = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    edges = {e: d.variable(e, split=True)[:2] for e in ("left", "right", "bottom", "top")}
    return d, u, ui.x * vi.x + ui.y * vi.y, edges


def test_equal_values_on_a_shared_corner_pin_it_once():
    """u = 1 on right and on top (Neumann elsewhere): the exact solution is u = 1 everywhere."""
    d, u, laplace, e = _square()
    U = np.asarray(jno.fem([laplace, u(*e["right"]) - 1.0, u(*e["top"]) - 1.0]).solve())
    np.testing.assert_allclose(U, 1.0, atol=1e-6)  # the corner was 0.5


def test_all_four_edges_at_one_value():
    """Every corner is shared by two of the four conditions; the exact solution is u = 3."""
    d, u, laplace, e = _square()
    U = np.asarray(jno.fem([laplace] + [u(*e[k]) - 3.0 for k in e]).solve())
    # 1e-6 is the iterative solve's tolerance (pinned rows are solved too, as s*u = s*g); the bug was 50 %.
    np.testing.assert_allclose(U, 3.0, atol=1e-6)


def test_conflicting_values_on_a_shared_node_raise():
    """right = 1 and top = 3 disagree at (1, 1); there is no right answer to pick, so it must refuse."""
    _d, u, laplace, e = _square()
    with pytest.raises(ValueError, match=r"same node.*\(1(\.0*)?, 1(\.0*)?\).*1\.0.*3\.0"):
        jno.fem([laplace, u(*e["right"]) - 1.0, u(*e["top"]) - 3.0]).solve()
