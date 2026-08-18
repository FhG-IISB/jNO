"""A mesh condition written as an inequality: ``criterion=jno.le(d.cell_aspect(), 2.0)``.

A plain criterion is a *ranking* -- Dörfler marks the worst fraction of it, which says where to refine
but never whether anything is bad enough to bother. A **constraint** states a condition, so its signed
margin says both: positive exactly on the cells that break it, and nowhere at all once they do not.
That is why the trigger needs no cadence or threshold argument of its own.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

meshio = pytest.importorskip("meshio")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _aspect(dom):
    node = dom.cell_aspect()
    return np.asarray(jno.core([node], domain=dom).eval([node], domain=dom)).reshape(-1)


def _poisson(skew=0.0, size=0.35):
    d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=size).domain()
    if skew:
        p = np.asarray(d.mesh.points)
        p[:, 1] = p[:, 1] * (1.0 - skew * p[:, 0] / 2.0 * p[:, 1])  # stretch the elements
        d.mesh.points = p
    u, phi = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0], quad_degree=3)


def test_a_satisfied_constraint_marks_nothing_and_leaves_the_mesh_alone():
    d, fem = _poisson()
    assert _aspect(d).max() < 10.0
    n0 = int(np.asarray(d.mesh.points).shape[0])
    fem.solve(adapt=jno.solve.remesh(criterion=lambda dm: jno.le(dm.cell_aspect(), 10.0), max_iters=3))
    assert [e["n_marked"] for e in fem.adapt_history] == [0]
    assert int(np.asarray(fem.domain.mesh.points).shape[0]) == n0


def test_a_violated_constraint_marks_exactly_the_offending_cells():
    """Not a Dörfler fraction of them -- every one, which is what a condition means."""
    d, fem = _poisson(skew=0.45)
    q = _aspect(d)
    thr = 1.4
    expected = int((q > thr).sum())
    assert 0 < expected < q.size, f"the fixture must have some but not all cells bad: {expected}/{q.size}"
    fem.solve(adapt=jno.solve.remesh(criterion=lambda dm: jno.le(dm.cell_aspect(), thr), max_iters=1))
    assert fem.adapt_history[0]["n_marked"] == expected


def test_the_march_stops_once_the_condition_holds_everywhere():
    """The stopping rule is the condition itself: `nothing_marked` already breaks the loop."""
    d, fem = _poisson(skew=0.45)
    assert _aspect(d).max() > 2.0
    fem.solve(adapt=jno.solve.remesh(criterion=lambda dm: jno.le(dm.cell_aspect(), 2.0), max_iters=6))
    assert _aspect(fem.domain).max() <= 2.0, "the constraint is still violated after the march"
    marks = [e["n_marked"] for e in fem.adapt_history]
    assert marks[-1] == 0, f"the march did not settle: {marks}"
    assert fem.adapt_history[-1]["estimate"] <= 0.0, "a satisfied constraint must report a margin <= 0"


def test_ge_works_without_the_driver_branching_on_sense():
    """`Constraint` normalises to `g <= 0` either way, so `ge` needs no separate path."""
    d, fem = _poisson()
    fem.solve(adapt=jno.solve.remesh(criterion=lambda dm: jno.ge(dm.cell_volume(), 1e9), max_iters=1))
    n_cells = int(np.asarray(d.mesh.cells_dict["triangle"]).shape[0])
    assert fem.adapt_history[0]["n_marked"] == n_cells, "every cell breaks an impossible lower bound"


def test_theta_with_a_constraint_is_refused():
    d, fem = _poisson()
    with pytest.raises(ValueError, match="no bulk fraction to choose"):
        fem.solve(adapt=jno.solve.remesh(criterion=jno.le(d.cell_aspect(), 2.0), theta=0.9, max_iters=1))


def test_a_boolean_criterion_is_refused_by_name():
    """`q > 6.0` is a valid trace node, so it would otherwise be Dörfler-marked over True/False --
    silently picking a FRACTION of the violators."""
    d, fem = _poisson()
    with pytest.raises(ValueError, match=r"criterion is a comparison"):
        fem.solve(adapt=jno.solve.remesh(criterion=(d.cell_aspect() > 1.0), max_iters=1))


def test_a_stale_geometry_node_says_it_is_stale():
    """A geometry node captures the cell table when constructed, so one built before a refinement
    keeps answering for the old mesh. That must be named, not read as a shape mistake."""
    d, fem = _poisson(skew=0.45)
    node = d.cell_aspect()  # built ONCE, deliberately not rebuilt per round
    with pytest.raises(ValueError, match="built on an EARLIER mesh"):
        fem.solve(adapt=jno.solve.remesh(criterion=jno.le(node, 1.05), max_iters=4))


def test_a_multicomponent_geometry_criterion_is_refused_by_name():
    d, fem = _poisson()
    with pytest.raises(ValueError, match="one value per cell"):
        fem.solve(adapt=jno.solve.remesh(criterion=lambda dm: dm.cell_angles(), max_iters=1))
