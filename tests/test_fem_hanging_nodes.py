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
from jno.utils.solver.fem_refine import hanging_dofs, hanging_nodes, hanging_prolongation, refine_domain

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


def _refined_centre(n=4, radius=0.3):
    d = _grid(n)
    p = np.asarray(d.mesh.points)[:, :2]
    q = np.asarray(d.mesh.cells_dict["quad"])
    return refine_domain(d, np.where(np.linalg.norm(p[q].mean(axis=1) - 0.5, axis=1) < radius)[0])


def _poisson_order(dom, order):
    u, v = dom.fem_symbols(order=order)
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    return np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)


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


# ------------------------------------------------------- what a refined mesh must not quietly lose


def test_a_named_boundary_region_survives_the_split(x64):
    """A Neumann term bound to a named region must still find its facets after refining.

    The refined mesh is built carrying only ``interior`` and ``boundary`` cell sets, so ``left`` /
    ``right`` / any ``.tag()`` region has to be re-derived from its spatial predicate -- otherwise a
    flux term silently integrates over nothing, or a Dirichlet condition fails by name. The adaptive
    driver already re-tags after a remesh; measured before `refine_domain` did the same, a direct call
    raised "Tag 'left' is not in the mesh pool or context".

    ``-Lap u = 0`` with ``u = 0`` on ``left`` and a unit flux on ``right``: refining must not change the
    answer, since the coarse mesh already resolves it.
    """

    def solve_flux(dom):
        u, v = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xl, yl, _ = dom.variable("left", split=True)
        xr, yr, _ = dom.variable("right", split=True)
        ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * v.bind(x=xr, y=yr), u(xl, yl) - 0.0])
        return np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).ravel()

    base = solve_flux(_grid(4))
    assert base.max() > 0.1, "the baseline flux term must do something, or this test proves nothing"

    d = _grid(4)
    p = np.asarray(d.mesh.points)[:, :2]
    q = np.asarray(d.mesh.cells_dict["quad"])
    refined = solve_flux(refine_domain(d, np.where(p[q].mean(axis=1)[:, 0] > 0.7)[0]))
    assert refined.max() == pytest.approx(base.max(), rel=1e-6), (
        f"the flux term changed across the split ({base.max():.6f} -> {refined.max():.6f}); "
        "a value of 0 means it lost its facets"
    )


def test_a_vector_field_is_constrained_component_by_component(x64):
    """``vec > 1``: the prolongation is the Kronecker expansion of the node map, so every component of a
    hanging node must satisfy the same relation. A P built on nodes but applied to a component-major
    layout would tie the wrong entries and fail here."""
    d = _grid(4)
    p = np.asarray(d.mesh.points)[:, :2]
    q = np.asarray(d.mesh.cells_dict["quad"])
    d = refine_domain(d, np.where(np.linalg.norm(p[q].mean(axis=1) - 0.5, axis=1) < 0.3)[0])
    n = len(np.asarray(d.mesh.points))

    u, v = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    a, t = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    lam, mu = 1.0, 1.0
    exx, eyy, exy = a[0].x, a[1].y, 0.5 * (a[0].y + a[1].x)
    tr = exx + eyy
    sxx, syy, sxy = lam * tr + 2 * mu * exx, lam * tr + 2 * mu * eyy, 2 * mu * exy
    fem = jno.fem([sxx * t[0].x + sxy * t[0].y + sxy * t[1].x + syy * t[1].y - 1.0 * t[1], u(xb, yb) - 0.0])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    assert sol.size == 2 * n
    s = sol.reshape(n, 2)
    assert np.abs(s).max() > 1e-6, "the body load did nothing; the test would pass vacuously"
    for node, parents in d._fem_hanging_nodes.items():
        for c in range(2):
            assert s[node, c] == pytest.approx(sum(w * s[i, c] for i, w in parents), abs=1e-12)


def test_order_2_is_constrained_correctly_and_beats_the_mesh_it_refined(x64):
    """Order 2 on a locally refined mesh, which needs the coarse edge's QUADRATIC basis.

    Order changes *which* DOFs hang, not only their weights, and getting that backwards is what made
    this wrong in both directions. jNO shares one DOF per coordinate, so the fine side's vertex at the
    coarse edge's midpoint IS that edge's own order-2 DOF -- free, not hanging -- while the DOFs that do
    hang sit at the quarter points and were left unconstrained. Measured with the P1 answer reused at
    order 2: centre off by 9.07e-03, i.e. 458x worse than order 1 on the same mesh.

    With the constraints built in the field's own DOF space: 1.16e-05, which also beats order 2 on the
    coarse mesh it was refined from (1.98e-05) -- the point of refining at all.
    """
    coarse_err = abs(_centre(_grid(4), _poisson_order(_grid(4), 2)) - POISSON_CENTRE)

    d = _refined_centre()
    sol = _poisson_order(d, 2)
    pts = np.asarray(d._fem_native_dof_points)[:, :2]
    got = float(sol[int(np.argmin(np.linalg.norm(pts - 0.5, axis=1)))])
    assert abs(got - POISSON_CENTRE) < coarse_err, (
        f"refining at order 2 did not beat the coarse mesh ({abs(got - POISSON_CENTRE):.2e} vs {coarse_err:.2e})"
    )

    hd = hanging_dofs(
        pts,
        np.asarray(d._fem_native_assembly_cells),
        np.asarray(d.mesh.points)[:, :2],
        np.asarray(d._fem_hanging_cells),
        "quad",
        2,
    )
    assert len(hd) > len(d._fem_hanging_nodes), "order 2 must constrain MORE DOFs than the vertex answer"
    assert {len(p) for p in hd.values()} == {3}, "an order-2 edge constraint has 3 parents"
    # the quadratic weights are not a convex combination -- a scheme assuming positivity would be wrong
    assert min(w for par in hd.values() for _, w in par) < 0.0
    for k, parents in hd.items():
        assert sum(w for _, w in parents) == pytest.approx(1.0, abs=1e-12)
        assert sol[k] == pytest.approx(sum(w * sol[i] for i, w in parents), abs=1e-10)


def test_order_2_on_a_refined_HEX_mesh_is_refused_by_name(x64):
    """A hex's 2:1 interface also constrains DOFs lying on a FACE, which needs that face's order-2
    (9-node) basis rather than the edge basis. Refused rather than left partly constrained."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1).structured(n=2).quad().domain(compute_mesh_connectivity=False)
    d = refine_domain(d, [0])
    u, v = d.fem_symbols(order=2)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    with pytest.raises(NotImplementedError, match="hexahedral"):
        jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u(xb, yb, zb) - 0.0]).solve(
            linear=jno.solve.lu(backend="host")
        )


def test_order_2_still_works_on_a_uniformly_refined_mesh(x64):
    """Refining EVERY cell stays conforming, so there are no hanging
    nodes and a P2 space is perfectly fine on the result."""
    d = refine_domain(_grid(4), np.arange(16))
    assert d._fem_hanging_nodes == {}
    u, v = d.fem_symbols(order=2)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)
    pts = np.asarray(fem.points)[:, :2]
    centre = float(sol[int(np.argmin(np.linalg.norm(pts - 0.5, axis=1)))])
    assert abs(centre - POISSON_CENTRE) < 1e-4, f"P2 on a uniformly refined mesh should be accurate, got {centre}"


# --------------------------------------------------------------- the field types that share a mesh


def _field_major(sol, n, k):
    """Coupled blocks are concatenated per FIELD; a vector field is node-major. Reading one as the
    other silently reports a violated constraint on a correct solution — which is how this was first
    misdiagnosed."""
    return np.asarray(sol).reshape(-1).reshape(k, n).T


def test_a_coupled_two_field_problem_constrains_both_fields(x64):
    """A coupled system reduces block-wise, so it needs one P per FIELD. Handing it a single field's P
    reached JAX as a bare shape error -- "contracting dimensions ... (41,) and (82,)" -- naming neither
    the fields nor the refinement."""
    d = _refined_centre()
    n = len(np.asarray(d.mesh.points))
    T, q = d.fem_symbols(names=("T", "q"))
    c, p = d.fem_symbols(names=("c", "p"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    Ti, qi = T.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    ci, pi = c.bind(x=xi, y=yi), p.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            Ti.x * qi.x + Ti.y * qi.y - ci * qi,  # T driven by c
            ci.x * pi.x + ci.y * pi.y - 1.0 * pi,
            T(xb, yb) - 0.0,
            c(xb, yb) - 0.0,
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)
    assert sol.size == 2 * n
    s = _field_major(sol, n, 2)
    assert np.abs(s).max() > 1e-6, "both fields are zero; the test would pass vacuously"
    for node, parents in d._fem_hanging_nodes.items():
        for f in range(2):
            assert s[node, f] == pytest.approx(sum(w * s[i, f] for i, w in parents), abs=1e-12)


def test_a_complex_field_constrains_both_its_real_and_imaginary_parts(x64):
    """jNO carries a complex field as two coupled real fields, so it arrives at the reduction as a
    two-field problem and rides the same per-field blocks. Both halves must satisfy the constraint,
    which is what the relation means for a complex field."""
    d = _refined_centre()
    n = len(np.asarray(d.mesh.points))
    u, w = d.fem_symbols(complex=True)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, wi = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
    weak = ui.x * wi.x + ui.y * wi.y - 4.0 * ui * wi - (1.0 + 0.5j) * wi
    fem = jno.fem([weak.real, u.real(xb, yb) - 0.0, u.imag(xb, yb) - 0.0])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)
    assert sol.size == 2 * n
    s = _field_major(sol, n, 2)
    # a genuinely complex source: neither half may be trivial, or half the assertion is vacuous
    assert np.abs(s[:, 0]).max() > 1e-6 and np.abs(s[:, 1]).max() > 1e-6
    for node, parents in d._fem_hanging_nodes.items():
        for part in range(2):
            assert s[node, part] == pytest.approx(sum(w * s[i, part] for i, w in parents), abs=1e-12)


def test_a_3d_vector_field_is_constrained_on_a_refined_hex_mesh(x64):
    """vec = 3 on hexes: the Kronecker expansion again, in the dimension where both hanging kinds
    (edge midpoints and face centres) occur at once."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1).structured(n=4).quad().domain(compute_mesh_connectivity=False)
    p = np.asarray(d.mesh.points)[:, :3]
    h = np.asarray(d.mesh.cells_dict["hexahedron"])
    d = refine_domain(d, np.where(np.linalg.norm(p[h].mean(axis=1) - 0.5, axis=1) < 0.3)[0])
    n = len(np.asarray(d.mesh.points))

    u, v = d.fem_symbols(value_shape=(3,))
    cs = d.variable("interior", split=True)[:3]
    bs = d.variable("boundary", split=True)[:3]
    gu, gv = jno.np.grad(u, list(cs)), jno.np.grad(v, list(cs))
    vv = v.bind(x=cs[0], y=cs[1], z=cs[2])
    fem = jno.fem([jno.np.inner(gu, gv, n_contract=2) - 1.0 * vv[2], u(*bs) - 0.0])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)
    assert sol.size == 3 * n
    s = sol.reshape(n, 3)  # vector fields are node-major
    assert np.abs(s).max() > 1e-6
    assert {len(par) for par in d._fem_hanging_nodes.values()} == {2, 4}
    for node, parents in d._fem_hanging_nodes.items():
        for comp in range(3):
            assert s[node, comp] == pytest.approx(sum(w * s[i, comp] for i, w in parents), abs=1e-12)


def test_order_3_is_constrained_correctly(x64):
    """Order 3, and the reason it is worth having a test of its own rather than trusting the order-2 one.

    The constraint machinery is order-general (a cell's own edge DOFs at k/order, a 2:1 neighbour's at
    k/(2*order), Lagrange through the former), and at order 3 it built the right 24 constraints with a
    passing patch test -- yet the solve was wrong by 5.53e-02. The cause was elsewhere and dimensional: a
    2-D edge facet at order 3 has FOUR columns (2 vertices + 2 on-edge nodes), and the covering filter
    identified facets by column count, reading it as a 3-D quadrilateral FACE. It then dropped nothing,
    the 2:1 interface was handed to the solve as boundary, and a Dirichlet condition pinned it mid-domain
    -- the same 0.018-vs-0.074 signature as the original blocker. Orders 1 and 2 were unaffected only
    because their column counts fell on the right side of the guess.
    """
    coarse_err = abs(_centre(_grid(4), _poisson_order(_grid(4), 3)) - POISSON_CENTRE)
    d = _refined_centre()
    sol = _poisson_order(d, 3)
    pts = np.asarray(d._fem_native_dof_points)[:, :2]
    got = float(sol[int(np.argmin(np.linalg.norm(pts - 0.5, axis=1)))])
    assert abs(got - POISSON_CENTRE) < 1e-5, f"order 3 on a refined mesh is off by {abs(got - POISSON_CENTRE):.2e}"
    assert abs(got - POISSON_CENTRE) < coarse_err

    hd = hanging_dofs(
        pts,
        np.asarray(d._fem_native_assembly_cells),
        np.asarray(d.mesh.points)[:, :2],
        np.asarray(d._fem_hanging_cells),
        "quad",
        3,
    )
    assert {len(p) for p in hd.values()} == {4}, "an order-3 edge constraint has 4 parents"
    for k, parents in hd.items():
        assert sum(w for _, w in parents) == pytest.approx(1.0, abs=1e-12)
        assert sol[k] == pytest.approx(sum(w * sol[i] for i, w in parents), abs=1e-10)


def test_the_covering_filter_is_told_the_facet_kind_not_left_to_guess():
    """A 2-D edge and a 3-D quadrilateral face can both arrive with 4 columns, so the number of columns
    does not identify the facet — the mesh's DIMENSION does. Pinned directly, because the failure it
    caused was three steps away from its cause."""
    from jno.utils.solver.fem_refine import drop_covered_facets

    # one hanging node with 2 parents (an edge midpoint): edge (0,1) is covered, and so is any edge
    # touching node 9
    hang = {9: [(0, 0.5), (1, 0.5)]}
    order3_edges = np.array([[0, 1, 5, 6], [2, 3, 7, 8]], dtype=np.int64)  # 4 columns, but EDGES
    kept = drop_covered_facets(order3_edges, hang, n_v=2)
    assert len(kept) == 1 and kept[0][0] == 2, "the covered order-3 edge was not dropped"
    # read as a 4-vertex face, the same rows are not recognised at all
    assert len(drop_covered_facets(order3_edges, hang, n_v=4)) == 2
