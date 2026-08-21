"""``relocate(...).remesh(criterion=...)`` — r-adaptivity that adds nodes when moving them is not enough.

Relocation moves a FIXED node set. When the mesh has to stretch further than its elements allow there
is nothing it can do: ``quality_floor`` is a line search that rejects the step, and rejecting a step
never adds a node. This says what "too far" means, as an inequality in the same traced language the
objective is written in, and remeshes when relocation can no longer honour it.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

meshio = pytest.importorskip("meshio")
pytest.importorskip("mmgpy", reason="mmgpy required for adaptive remeshing")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _aspect(dom):
    tc = list(getattr(dom, "_trainable_coords", None) or [])
    dom._trainable_coords = []
    try:
        return np.asarray(dom.cell_aspect().eval()).reshape(-1)
    finally:
        dom._trainable_coords = tc


def _peak(size=0.2):
    """Poisson with an off-centre peak; a central box of vertices is movable."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xm, ym, _ = d.variable("mov", where=lambda x, y: (x > 0.15) & (x < 0.85) & (y > 0.15) & (y < 0.85), split=True)
    xm.trainable(name="ix")
    ym.trainable(name="iy")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = jno.np.exp(-60.0 * ((xi - 0.62) ** 2 + (yi - 0.35) ** 2))
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)


def test_the_chained_spec_carries_the_nested_one():
    spec = jno.solve.relocate(objective="energy", max_iters=7).remesh(criterion=None, max_iters=3)
    assert spec.relocate is True and spec.max_iters == 7
    assert spec.remesh_spec is not None and spec.remesh_spec.max_iters == 3
    assert jno.solve.relocate().remesh_spec is None, "a plain relocate must be untouched"


def test_the_march_honours_the_condition_it_was_given():
    """The point of the feature, and the thing an earlier design got wrong.

    Checking the condition AFTER each round only reports the damage: one accepted step can carry the
    mesh far past the bound, and breach margins over a march grew +0.019 -> +0.555 -> +1.070 -> +3.375
    that way, ending on a mesh at aspect 3.73 against a bound of 1.7. The condition is therefore tested
    inside the line search, on the candidate, so an inadmissible step is simply never taken.
    """
    d, fem = _peak()
    bound = 1.7
    assert _aspect(d).max() < bound, "the fixture should start admissible"
    fem.solve(
        adapt=jno.solve.relocate(objective="equidistribution", max_iters=40, lr=6e-3).remesh(
            criterion=lambda dm: jno.le(dm.cell_aspect(), bound), max_iters=4
        )
    )
    assert _aspect(fem.domain).max() <= bound + 1e-9, "the march ended on a mesh that breaks its own condition"


def test_it_adds_nodes_and_keeps_going_afterwards():
    """A remesh destroys the node set the descent was moving. The movable vertices are re-derived from
    the REGION each one was tagged on -- indices do not survive a remesh, a region name does."""
    d, fem = _peak()
    n0 = int(np.asarray(d.mesh.points).shape[0])
    fem.solve(
        adapt=jno.solve.relocate(objective="equidistribution", max_iters=40, lr=6e-3).remesh(
            criterion=lambda dm: jno.le(dm.cell_aspect(), 1.7), max_iters=4
        )
    )
    n1 = int(np.asarray(fem.domain.mesh.points).shape[0])
    assert n1 > n0, f"no nodes were added: {n0} -> {n1}"
    hist = fem.adapt_history
    assert len(hist) > 1, "the descent did not resume after the remesh"
    objs = [e["objective"] for e in hist]
    assert min(objs) < objs[0], f"the objective did not improve across the march: {objs[0]:.3e}"


def test_a_bound_the_mesher_cannot_reach_is_reported_not_run_away_from():
    """Refining does not repair every shape. Before the line search enforced the condition this ran
    44 -> 61 -> 110 -> 243 -> 503 -> 1161 -> 2911 -> 7104 -> 15709 vertices and died inside the solver
    with an out-of-memory error that named nothing."""
    d, fem = _peak()
    with pytest.raises(RuntimeError, match="still breaks the condition"):
        fem.solve(
            adapt=jno.solve.relocate(objective="equidistribution", max_iters=40, lr=6e-3).remesh(
                criterion=lambda dm: jno.le(dm.cell_aspect(), 1.02), max_iters=2
            )
        )


def test_a_ranking_criterion_is_refused_by_name():
    """A plain expression says which cells are worst, never whether any is bad enough to act on."""
    d, fem = _peak()
    with pytest.raises(ValueError, match="must be a CONDITION"):
        fem.solve(adapt=jno.solve.relocate(max_iters=2).remesh(criterion=lambda dm: dm.cell_aspect()))


def test_a_solution_criterion_is_refused_by_name():
    """The interleaved check runs on the moved vertices with no solve, so there is nothing to read."""
    d, fem = _peak()
    xi, _yi, _ = d.variable("interior", split=True)
    with pytest.raises(ValueError, match="MESH-GEOMETRY"):
        fem.solve(adapt=jno.solve.relocate(max_iters=2).remesh(criterion=jno.le(1.0 * xi, 0.5)))


def test_plain_relocate_is_unchanged():
    d, fem = _peak()
    out = np.asarray(fem.solve(adapt=jno.solve.relocate(objective="equidistribution", max_iters=6, lr=6e-3)))
    assert np.isfinite(out).all() and fem.adapt_history


def _free_surface_channel(size=0.34):
    """Stokes channel with a traction-free top whose vertices may slide vertically -- the free-surface
    problem, whose objective is a SURFACE integral and so exercises the region resolution."""
    d = jno.Shape.rect(0.0, 0.0, 3.0, 1.0, size=size).domain()
    p = np.asarray(d.mesh.points)
    p[:, 1] = p[:, 1] * (1.0 - 0.22 * p[:, 0] / 3.0 * p[:, 1])  # a deliberately wrong wall
    d.mesh.points = p
    wall = lambda X, Y: Y > 1.0 - 0.22 * X / 3.0 - 1e-9  # noqa: E731
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    pr, q = d.fem_symbols(names=("p", "q"), order=1)
    x, y, _ = d.variable("interior", split=True)
    cin = d.variable("inlet", where=lambda X, Y: X < 1e-9, split=True)
    cbot = d.variable("bottom", where=lambda X, Y: Y < 1e-9, split=True)
    ct = d.variable("top", where=wall, normals=True, split=True)
    cmov = d.variable("tmov", where=lambda X, Y: wall(X, Y) & (X > 1e-9), split=True)
    cmov[1].trainable(name="ty")
    eu, ev = jno.np.symgrad(u, [x, y]), jno.np.symgrad(v, [x, y])
    dd = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    pp, qq = pr.bind(x=x, y=y), q.bind(x=x, y=y)
    vt, ut = v.bind(x=ct[0], y=ct[1]), u.bind(x=ct[0], y=ct[1])
    nx, ny = ct[-2], ct[-1]
    fem = jno.fem(
        [
            2.0 * dd(eu, ev) - pp * jno.np.trace(ev),
            -qq * jno.np.trace(eu),
            0.0 * vt[0],
            u(cin[0], cin[1])[0] - 1.0,
            u(cin[0], cin[1])[1] - 0.0,
            u(cbot[0], cbot[1])[0] - 0.0,
            u(cbot[0], cbot[1])[1] - 0.0,
            pr.pin(),
        ],
        quad_degree=3,
    )
    return d, fem, (ut[0] * nx + ut[1] * ny) ** 2 * vt[0]


def test_a_surface_objective_survives_the_remesh():
    """A free surface AND a mesh condition, together -- the case the two features exist for.

    The objective is a surface integral, so its region has to be resolved to boundary facets. That
    resolution walks `tag_node_mask`, which intersects the tag with the catch-all "boundary" region
    under `jax.vmap`; run from inside the traced objective it yields a traced mask, and converting one
    raises. It used to resolve cleanly twice (50 points, then 62 eagerly after the remesh) and fail the
    third time, inside the trace -- `FEM.eval` builds a fresh term list per call, so it never hit the
    assembler's own memo. The facet selection is a property of the mesh, so it is now memoized per
    assembly and the traced path finds it already computed.
    """
    d, fem, flux = _free_surface_channel()
    n0 = int(np.asarray(d.mesh.points).shape[0])
    fem.solve(
        adapt=jno.solve.relocate(objective=flux, max_iters=60, lr=3e-2).remesh(
            criterion=lambda dm: jno.le(dm.cell_aspect(), 1.9), max_iters=4
        )
    )
    n1 = int(np.asarray(fem.domain.mesh.points).shape[0])
    assert n1 > n0, f"no remesh happened, so this pins nothing: {n0} -> {n1}"
    objs = [e["objective"] for e in fem.adapt_history]
    assert min(objs) < objs[0] / 2.0, f"through-flow was not reduced: {objs[0]:.3e} -> {min(objs):.3e}"
    assert _aspect(fem.domain).max() <= 1.9 + 1e-9
