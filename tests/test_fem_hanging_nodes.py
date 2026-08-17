"""Solving on a locally refined mesh, with the hanging nodes constrained.

The mesh machinery is tested in ``tests/test_fem_refine_quads.py`` and the constraint *weights* in
``tests/test_fem_hex_tie.py``; this file is the two joined to the solve. A hanging node is constrained
to the coarse edge it lies on, ``u_h = sum_i w_i u_parent_i`` -- the same relation a periodic tie and a
mortar coupling impose -- so it rides the SAME prolongation, and ``P^T A P`` is the whole mechanism.

Two things had to be true before that worked, and both were silent when they were not:

* the **boundary** must survive non-conformity. jNO derives it topologically, and across a 2:1
  interface the coarse edge and both half-edges each belong to exactly one cell, so the interface reads
  as boundary and a Dirichlet condition PINS it;
* refining an already-refined mesh must **reuse** the hanging node sitting at the edge midpoint rather
  than create a second node there, which edge topology cannot see once the mesh is non-conforming.

Both are pinned below by the number they moved, because neither changed the area, the winding or the
2:1 balance -- the properties the mesh tests already checked.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.utils.solver import fem_refine
from jno.utils.solver.fem_refine import hanging_nodes, hanging_prolongation, refine_domain

# -Lap u = 1 on the unit square, u = 0 on the boundary: the centre value of the classic series
# solution, u(1/2,1/2) = (32/pi^3) sum_{k odd} (-1)^((k-1)/2) / (k^3 cosh(k pi / 2)).
# Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells* (2nd ed., 1959), Art. 30.
POISSON_CENTRE = 0.07367135


@pytest.fixture
def x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _grid(n=4):
    return jno.Shape.rect(0, 0, 1, 1).quad().structured(n=n).domain(compute_mesh_connectivity=False)


def _mark_near(dom, centre, radius):
    p = np.asarray(dom.mesh.points)[:, :2]
    q = np.asarray(dom.mesh.cells_dict["quad"])
    return np.where(np.linalg.norm(p[q].mean(axis=1) - np.asarray(centre), axis=1) < radius)[0]


def _poisson(dom, f=1.0):
    """-Lap u = f, u = 0 on the boundary. Solved DIRECTLY: the default Jacobi-BiCGStab stops at its own
    tolerance, which is larger than the differences this file measures."""
    u, v = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])
    return np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).ravel()


def _centre(dom, sol):
    p = np.asarray(dom.mesh.points)[:, :2]
    return float(sol[int(np.argmin(np.linalg.norm(p - 0.5, axis=1)))])


# --------------------------------------------------------------------------- it solves, and correctly


def test_a_locally_refined_solve_lands_between_the_two_uniform_meshes(x64):
    """The headline. Refining the middle four cells of a 4x4 grid must buy accuracy there: the answer
    has to sit between the coarse mesh it came from and the uniform mesh it is locally as fine as."""
    coarse = _centre(_grid(4), _poisson(_grid(4)))
    fine = _centre(_grid(8), _poisson(_grid(8)))
    d = refine_domain(_grid(4), [5, 6, 9, 10])
    got = _centre(d, _poisson(d))
    assert len(d._fem_hanging_nodes) == 8
    assert min(coarse, fine) < got < max(coarse, fine), f"{got} is not between {coarse} and {fine}"
    assert abs(got - POISSON_CENTRE) < abs(coarse - POISSON_CENTRE), "refining made it worse"


def test_repeated_refinement_converges(x64):
    """Four rounds into the centre. The error must fall every round -- the property that caught the
    duplicated-midpoint bug, which converged to the WRONG value while every mesh invariant held."""
    d = _grid(4)
    errs = []
    for _ in range(4):
        d = refine_domain(d, _mark_near(d, (0.5, 0.5), 0.3))
        errs.append(abs(_centre(d, _poisson(d)) - POISSON_CENTRE))
    assert all(b < a for a, b in zip(errs, errs[1:])), f"not monotone: {errs}"
    assert errs[-1] < 1e-4, f"four rounds should be well converged, got {errs[-1]:.2e}"


def test_the_constraint_is_what_makes_it_right(x64, monkeypatch):
    """The discriminating test: with the SAME mesh and the SAME boundary, dropping only the constraint
    must break it. Patching the prolongation rather than clearing the domain's hanging nodes, because
    those also drive the boundary derivation -- clearing them changes two things at once and the
    comparison stops being about the constraint.

    Measured against a reference of 0.073671: constrained 0.075063 (error 1.4e-03), unconstrained
    0.090093 (error 1.6e-02) -- an 11.9x worse answer, and the unconstrained one is worse than the 4x4
    grid it was refined FROM, so the refinement is not merely wasted but harmful without the tie.
    """
    from jno.utils.solver.fem_utils import prolongation_from_ties

    d = refine_domain(_grid(4), [5, 6, 9, 10])
    good = _centre(d, _poisson(d))

    # the same call, but tying nothing: P is the identity, so the mesh, the boundary and the assembly
    # are untouched and the ONLY difference is that the hanging DOFs stay free
    monkeypatch.setattr(
        fem_refine,
        "hanging_prolongation",
        lambda points, quads, **kw: prolongation_from_ties(len(points), {}, {}, vec=kw.get("vec", 1)),
    )
    loose = _centre(d, _poisson(d))

    coarse = _centre(_grid(4), _poisson(_grid(4)))
    assert abs(good - POISSON_CENTRE) < 0.5 * abs(coarse - POISSON_CENTRE), "the constraint bought nothing"
    assert abs(loose - POISSON_CENTRE) > 5.0 * abs(good - POISSON_CENTRE), (
        f"dropping the constraint changed almost nothing (constrained {good:.6f}, free {loose:.6f}) -- "
        "the tie is not load-bearing, so it is not being applied"
    )
    # and without it the refined mesh is WORSE than the grid it came from
    assert abs(loose - POISSON_CENTRE) > abs(coarse - POISSON_CENTRE)


def test_every_hanging_node_equals_its_parents_average(x64):
    """The constraint itself, on the solved field. Exact rather than approximate: the DOF is
    eliminated, so it is satisfied by construction -- this asserts the elimination actually happened."""
    d = refine_domain(_grid(4), _mark_near(_grid(4), (0.25, 0.25), 0.3))
    sol = _poisson(d)
    hang = d._fem_hanging_nodes
    assert hang
    for n, ((a, wa), (b, wb)) in hang.items():
        assert sol[n] == pytest.approx(wa * sol[a] + wb * sol[b], abs=1e-12)


# ------------------------------------------------------------------------------------ the boundary


def test_the_two_to_one_interface_is_not_pinned_as_a_boundary(x64):
    """The bug that blocked this for a whole phase. The topological rule makes the interface look like
    a boundary, and the aggregate ``boundary`` region is what a Dirichlet condition resolves through --
    so the interface gets pinned, silently, in the middle of the domain.

    Measured before the fix: 32 identity rows on a mesh whose perimeter is 16 nodes.
    """
    d = refine_domain(_grid(4), [5, 6, 9, 10])
    p = np.asarray(d.mesh.points)[:, :2]
    bn = np.asarray((getattr(d, "tag_indices", {}) or {})["boundary"]).reshape(-1)
    perimeter = {i for i in range(len(p)) if min(p[i][0], p[i][1], 1 - p[i][0], 1 - p[i][1]) < 1e-9}
    assert set(int(i) for i in bn) == perimeter
    assert not (set(int(i) for i in bn) & set(d._fem_hanging_nodes)), "a hanging node was called boundary"


def test_a_uniformly_refined_mesh_reproduces_the_matching_uniform_grid(x64):
    """Refining EVERY cell is conforming again, so it must agree with the uniform grid of that size to
    the last digit -- the degenerate case that separates a refinement bug from a constraint bug."""
    d = refine_domain(_grid(4), np.arange(16))
    assert d._fem_hanging_nodes == {}
    assert _centre(d, _poisson(d)) == pytest.approx(_centre(_grid(8), _poisson(_grid(8))), rel=1e-12)


# ------------------------------------------------------------------------- the prolongation itself


def test_the_prolongation_eliminates_exactly_the_hanging_nodes():
    d = refine_domain(_grid(4), [5, 6, 9, 10])
    p = np.asarray(d.mesh.points)[:, :2]
    q = np.asarray(d.mesh.cells_dict["quad"])
    red = hanging_prolongation(p, q)
    assert red["n_full"] == len(p)
    assert red["n_red"] == len(p) - len(d._fem_hanging_nodes)
    assert red["coupling"] == "hanging"
    # a partition of unity: a constant field must prolong to the same constant
    P = np.asarray(red["P"].todense())
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)


def test_the_prolongation_reproduces_a_linear_field_exactly():
    """The patch test at prolongation level. A field the coarse edge can represent must come through
    the constraint untouched -- weights that sum to 1 at the wrong point still pass the check above."""
    d = refine_domain(_grid(4), [5, 6, 9, 10])
    p = np.asarray(d.mesh.points)[:, :2]
    red = hanging_prolongation(p, np.asarray(d.mesh.cells_dict["quad"]))
    P = np.asarray(red["P"].todense())
    kept = np.asarray(red["kept_nodes"])
    field = 1.3 + 2.0 * p[:, 0] - 0.7 * p[:, 1]
    np.testing.assert_allclose(P @ field[kept], field, atol=1e-12)


def test_refining_twice_reuses_the_hanging_node_instead_of_duplicating_it():
    """A node already sitting at an edge midpoint is that edge's hanging node, and refining the cell is
    what promotes it to a regular one. Edge topology cannot see it (the coarse edge and the two
    half-edges are different edges once the mesh is non-conforming), so a second node appears at the
    same coordinate and the mesh silently splits along the interface -- with the area, the winding and
    the 2:1 balance all still correct."""
    d = refine_domain(_grid(4), _mark_near(_grid(4), (0.3, 0.3), 0.3))
    d = refine_domain(d, _mark_near(d, (0.25, 0.25), 0.2))
    p = np.asarray(d.mesh.points)[:, :2]
    assert len(p) == len(np.unique(np.round(p, 12), axis=0)), "a node was duplicated at an interface"


# ------------------------------------------------------------------------------------- refusals


def test_a_hanging_node_on_a_tied_interface_is_refused_by_name():
    """Two prolongations on the same DOF: the hanging constraint eliminates it onto its coarse edge,
    the tie onto the other interface, and the order changes the answer. Refused rather than half-done."""
    d = refine_domain(_grid(4), [5, 6, 9, 10])
    p = np.asarray(d.mesh.points)[:, :2]
    q = np.asarray(d.mesh.cells_dict["quad"])
    victim = next(iter(hanging_nodes(p, q)))
    with pytest.raises(NotImplementedError, match="composes two prolongations"):
        hanging_prolongation(p, q, tied_nodes=[victim])


def test_local_refinement_of_a_non_quadrilateral_mesh_is_refused_by_name():
    d = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain(compute_mesh_connectivity=False)
    with pytest.raises(NotImplementedError, match="quadrilateral"):
        refine_domain(d, [0])
