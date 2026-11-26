"""``fem.solve(adapt=jno.solve.refine(...))`` -- h-adaptivity by splitting cells.

The adaptive loop could only *remesh*: rebuild the mesh at a finer size field, which needs a mesher and
therefore a geometry to rebuild from. ``refine`` splits the marked cells instead, so it works on a mesh
loaded from a file, keeps every node, and -- the part that matters most -- works in 3-D, where there is
no all-hex mesher to remesh to and ``remesh`` refuses by name.

Everything upstream of the mesh change is shared with ``remesh``: the ZZ estimator, the traced
``criterion``, Dörfler marking, and the ``max_iters`` / ``max_dofs`` / ``tol`` / ``eps`` budgets. What
this file checks is that the loop *converges* through the new mechanism and that the constraints
survive re-assembly each round -- not that the plumbing runs.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _peaked_2d(n=8):
    """-Lap u = a narrow Gaussian at the centre: a local feature, so refinement should be local."""
    d = jno.shape.rect(0, 0, 1, 1).quad().structured(n=n).domain(compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = jno.np.exp(-(((xi - 0.5) ** 2 + (yi - 0.5) ** 2) / 0.003))
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0]), xi, yi


def _peaked_3d(n=4):
    d = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=n).quad().domain(compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    f = jno.np.exp(-(((xi - 0.5) ** 2 + (yi - 0.5) ** 2 + (zi - 0.5) ** 2) / 0.02))
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - f * vi, u(xb, yb, zb) - 0.0])


def _cell_sizes(dom, cell_type):
    p = np.asarray(dom.mesh.points)[:, : int(dom.dimension)]
    c = np.asarray(dom.mesh.cells_dict[cell_type])
    return np.linalg.norm(p[c].max(axis=1) - p[c].min(axis=1), axis=1)


# ------------------------------------------------------------------------------------- 2-D quads


def test_the_loop_refines_and_the_estimate_falls(x64):
    """The headline. Each round must add DOFs and lower the ZZ estimate — a loop that ran but never
    marked, or marked but never changed the mesh, would still return a solution."""
    d, fem, _xi, _yi = _peaked_2d()
    n0 = len(d.mesh.points)
    fem.solve(adapt=jno.solve.refine(theta=0.5, max_iters=4))

    hist = fem.adapt_history
    assert len(hist) == 4
    dofs = [h["n_dofs"] for h in hist]
    ests = [h["estimate"] for h in hist]
    assert dofs == sorted(dofs) and dofs[-1] > n0, f"the mesh did not grow: {dofs}"
    assert all(b < a for a, b in zip(ests, ests[1:])), f"the estimate did not fall: {ests}"


def test_it_refines_locally_rather_than_everywhere(x64):
    """Splitting is worth having only if it is LOCAL: a loop that split every cell would also grow the
    mesh and lower the estimate, so the discriminating checks are that a *range* of cell sizes appears
    and that the finest cells are a minority, sitting where the source is.

    The depth is one level per round that marks, so it tracks ``max_iters``: measured 2x / 4x / 8x size
    spread at 3 / 5 / 7 rounds, with the finest cells 21% / 35% / 27% of the mesh.
    """
    d, fem, _xi, _yi = _peaked_2d()
    fem.solve(adapt=jno.solve.refine(theta=0.4, max_iters=5))
    h = _cell_sizes(fem.domain, "quad")
    assert h.max() / h.min() >= 4.0, f"refinement did not deepen (h spread {h.max() / h.min():.2f})"
    finest = (h < h.min() * 1.01).sum()
    assert finest < 0.6 * len(h), f"refinement was global, not local ({finest} of {len(h)} cells at the finest size)"
    # and the small cells are where the source is
    p = np.asarray(fem.domain.mesh.points)[:, :2]
    centres = p[np.asarray(fem.domain.mesh.cells_dict["quad"])].mean(axis=1)
    assert np.linalg.norm(centres[np.argmin(h)] - 0.5) < np.linalg.norm(centres[np.argmax(h)] - 0.5)


def test_the_hanging_constraints_survive_every_round(x64):
    """The mesh is re-assembled from scratch each round, so the constraint set has to be re-derived on
    the new mesh — a stale one from the previous round would leave real DOFs free."""
    d, fem, _xi, _yi = _peaked_2d()
    fem.solve(adapt=jno.solve.refine(theta=0.5, max_iters=3))
    hang = fem.domain._fem_hanging_nodes
    assert hang, "a locally refined mesh must have hanging nodes"
    chained = [n for par in hang.values() for n, _ in par if n in hang]
    assert not chained, "a hanging node acquired a hanging parent across the rounds"


def test_a_criterion_drives_the_split_loop(x64):
    """Part 1 x Part 2: the traced criterion composes with the new mechanism, because marking is shared.
    The ridge is deliberately NOT where the ZZ estimator would refine this problem."""
    d, fem, xi, yi = _peaked_2d()
    fem.solve(adapt=jno.solve.refine(criterion=jno.np.exp(-(((xi + yi - 1.0) / 0.05) ** 2)), theta=0.4, max_iters=3))
    p = np.asarray(fem.domain.mesh.points)[:, :2]
    c = np.asarray(fem.domain.mesh.cells_dict["quad"])
    centres = p[c].mean(axis=1)
    h = _cell_sizes(fem.domain, "quad")
    on = np.abs(centres.sum(axis=1) - 1.0) < 0.1
    assert h[~on].mean() / h[on].mean() > 1.5, "the mesh did not follow the criterion onto the ridge"


def test_max_dofs_stops_the_loop(x64):
    d, fem, _xi, _yi = _peaked_2d()
    fem.solve(adapt=jno.solve.refine(theta=0.6, max_iters=20, max_dofs=200))
    assert len(fem.domain.mesh.points) < 600, "the budget did not stop the loop"
    assert len(fem.adapt_history) < 20, "the loop ran to max_iters despite the budget"


# ------------------------------------------------------------------------------------- 3-D hexes


def test_a_hex_mesh_adapts_at_all(x64):
    """The capability this unlocks. ``remesh`` refuses on hexes -- no all-hex mesher exists, so there is
    nothing to remesh to -- and before this there was no other h-adaptive path for them."""
    d, fem = _peaked_3d()
    n0 = len(d.mesh.points)
    fem.solve(adapt=jno.solve.refine(theta=0.4, max_iters=3))

    assert len(fem.domain.mesh.points) > n0
    ests = [h["estimate"] for h in fem.adapt_history]
    assert all(b < a for a, b in zip(ests, ests[1:])), f"the estimate did not fall: {ests}"
    hang = fem.domain._fem_hanging_nodes
    assert {len(v) for v in hang.values()} == {2, 4}, "both hanging kinds must appear on a refined hex mesh"
    assert not [n for par in hang.values() for n, _ in par if n in hang]


def test_the_hex_loop_refines_locally(x64):
    d, fem = _peaked_3d()
    fem.solve(adapt=jno.solve.refine(theta=0.4, max_iters=3))
    h = _cell_sizes(fem.domain, "hexahedron")
    assert h.max() / h.min() >= 2.0, f"the hex refinement was global (h spread {h.max() / h.min():.2f})"


# --------------------------------------------------------------------------------------- refusals


def test_a_simplex_mesh_is_refused_by_name(x64):
    """Splitting a simplex is a different algorithm (and mmg already adapts them locally), so this says
    so rather than silently doing nothing."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.3).domain(compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    with pytest.raises(NotImplementedError, match="quadrilateral and hexahedral"):
        fem.solve(adapt=jno.solve.refine(max_iters=2))


def test_anisotropic_refinement_is_refused_by_name(x64):
    """A split is isotropic by construction, so there is no direction for a metric to stretch along."""
    d, fem, _xi, _yi = _peaked_2d()
    spec = jno.solve.refine(max_iters=2)
    spec.anisotropic = True
    with pytest.raises(NotImplementedError, match="no direction"):
        fem.solve(adapt=spec)


def test_the_spec_carries_the_split_flag_and_no_refine_factor():
    """`refine` and `remesh` build the same spec type; the flag is what selects the mechanism."""
    assert jno.solve.refine().split is True
    assert jno.solve.remesh().split is False
    assert jno.solve.refine(criterion=42).criterion == 42
    assert jno.solve.refine(theta=0.3).theta == 0.3


# ------------------------------------------------------------------------------ multi-box lattice


def _two_box_lattice():
    """Two unit boxes a|b side by side, different conductivity in each (1 and 2), and the manufactured
    solution sin(pi x/2) sin(pi y): its x-derivative vanishes at the interface x = 1, so the flux is
    continuous there for ANY piecewise-constant coefficient and the oracle is exact for both regions."""
    R = jno.shape.rect
    plan = (R(0, 0, 1, 1).name("a") + R(1, 0, 2, 1).name("b")).sized(0.25).structured().quad()
    d = plan.domain(compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    xa, ya, _ = d.variable("a", split=True)
    xb, yb, _ = d.variable("b", split=True)
    xw, yw, _ = d.variable("boundary", split=True)
    ua, va = u.bind(x=xa, y=ya), v.bind(x=xa, y=ya)
    ub, vb = u.bind(x=xb, y=yb), v.bind(x=xb, y=yb)
    w = 1.25 * np.pi**2

    def exact(x, y):
        return jno.np.sin(np.pi * x / 2) * jno.np.sin(np.pi * y)

    fem = jno.fem(
        [
            ua.x * va.x + ua.y * va.y - w * exact(xa, ya) * va,
            2.0 * (ub.x * vb.x + ub.y * vb.y) - 2.0 * w * exact(xb, yb) * vb,
            u(xw, yw) - 0.0,
        ]
    )
    return d, fem


def _max_error(dom, sol):
    p = np.asarray(dom.mesh.points)[:, :2]
    return float(np.abs(sol[: len(p)] - np.sin(np.pi * p[:, 0] / 2) * np.sin(np.pi * p[:, 1])).max())


def test_hanging_node_refinement_on_a_multi_box_lattice_keeps_regions_tags_and_converges(x64):
    """#267: `refine` on a `.structured().quad()` lattice of two named boxes. Each region's cells must follow
    the split (a child belongs to the region of its parent: both stay on their own side, covering their own
    area), the interface and outer tags must be re-derived on the refined mesh, and the solution must
    converge to the analytic one."""
    d, fem = _two_box_lattice()
    n0 = len(d.mesh.points)
    err0 = _max_error(d, np.asarray(fem.solve()).reshape(-1))
    sol = np.asarray(fem.solve(adapt=jno.solve.refine(theta=0.9, max_iters=3))).reshape(-1)
    dom = fem.domain
    assert len(dom.mesh.points) > n0 and dom._fem_hanging_nodes, "the lattice was not refined with hanging nodes"

    p = np.asarray(dom.mesh.points)[:, :2]
    cells = np.asarray(dom.mesh.cells_dict["quad"])
    sets = dom.mesh.cell_sets_dict
    lo, hi = p[cells].min(axis=1), p[cells].max(axis=1)
    area = np.prod(hi - lo, axis=1)
    a, b = np.asarray(sets["a"]["quad"]), np.asarray(sets["b"]["quad"])
    assert len(np.intersect1d(a, b)) == 0 and len(a) + len(b) == len(cells), "regions overlap or miss cells"
    assert hi[a, 0].max() <= 1.0 + 1e-12 and lo[b, 0].min() >= 1.0 - 1e-12, "a child left its parent's box"
    np.testing.assert_allclose([area[a].sum(), area[b].sum()], [1.0, 1.0], rtol=1e-12)

    # tags re-derived on the new mesh: each tag's points are exactly the refined mesh's nodes on its line
    # (so the interface carries the NEW nodes on x = 1, not the stale coarse ones)
    for name, x_at in (("a|b", 1.0), ("left", 0.0), ("right", 2.0)):
        xt, yt, _ = dom.variable(name, split=True)
        xs, ys = np.asarray(xt.eval()).reshape(-1), np.asarray(yt.eval()).reshape(-1)
        assert np.allclose(xs, x_at), f"tag {name!r} is not on x = {x_at}"
        on_line = np.unique(np.round(p[np.isclose(p[:, 0], x_at), 1], 12))
        np.testing.assert_allclose(np.unique(np.round(ys, 12)), on_line, err_msg=f"tag {name!r}")

    err = _max_error(dom, sol)
    assert err < 0.5 * err0, f"no convergence against the analytic oracle: {err0:.3e} -> {err:.3e}"
