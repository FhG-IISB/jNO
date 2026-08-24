"""``relocate(objective=<expression>)`` — the mesh objective as a weak form, and the surface readout
it needs.

The three string objectives (``"energy"``, ``"equidistribution"``, ``"huang"``) are mesh-QUALITY
functionals: they see the solution only through a monitor, so they can ask for resolution but cannot
state a goal that names the physics. An expression can — ``(u·n)²`` on a wall is a free surface.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno
from jno import np as J

meshio = pytest.importorskip("meshio")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _peak(size=0.14, movable=True):
    """Poisson with a sharp off-center peak; a central box of interior nodes is movable."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    if movable:
        xm, ym, _ = d.variable("mov", where=lambda x, y: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8), split=True)
        xm.trainable(name="ix")
        ym.trainable(name="iy")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = J.exp(-40.0 * ((xi - 0.62) ** 2 + (yi - 0.35) ** 2))
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3), ui


def test_a_traced_volume_objective_runs_and_descends():
    d, fem, ui = _peak()
    p0 = np.asarray(d.mesh.points)[:, :2].copy()
    fem.solve(adapt=jno.solve.relocate(objective=ui * ui, max_iters=10, lr=3e-3))
    hist = fem.adapt_history
    objs = [e["objective"] for e in hist]
    assert len(objs) > 1
    assert min(objs) < objs[0], f"the traced objective did not descend: {objs}"
    moved = np.abs(np.asarray(fem.domain.mesh.points)[:, :2] - p0).max()
    assert moved > 0.0, "no vertex moved, so nothing was optimised"


def test_the_traced_objective_gradient_matches_finite_differences():
    """The keystone: descent is only meaningful if d(objective)/d(vertex) is right. A nonzero gradient
    is not evidence — a wrong one is also nonzero — so it is checked against central differences."""
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _criterion_weak_terms
    from jno.utils.solver.linear import sparse_lu_solve

    d, fem, ui = _peak(size=0.25)
    weak, _ = _criterion_weak_terms(fem, ui * ui)
    specs = d._trainable_coords
    pts = np.asarray(d.mesh.points)
    vals = {s["name"]: jnp.asarray(pts[np.asarray(s["ids"], int), s["axis"]]) for s in specs}
    op = fem.operator

    def scalar(v):
        a, b = op.evaluate(v)
        rhs = jnp.asarray(b).reshape(-1)
        uu = sparse_lu_solve(a, rhs) if hasattr(a, "indices") else jnp.linalg.solve(jnp.asarray(a), rhs)
        return jnp.sum(jnp.asarray(fem.eval(weak, uu, args=v)))

    _, grad = jax.value_and_grad(scalar)(vals)
    h = 1e-6
    for key in vals:
        fd = []
        for j in range(len(vals[key])):
            vp = {k: (v.at[j].add(h) if k == key else v) for k, v in vals.items()}
            vm = {k: (v.at[j].add(-h) if k == key else v) for k, v in vals.items()}
            fd.append((float(scalar(vp)) - float(scalar(vm))) / (2 * h))
        fd, ad = np.array(fd), np.asarray(grad[key])
        rel = np.linalg.norm(fd - ad) / max(np.linalg.norm(fd), 1e-30)
        assert rel < 1e-5, f"{key}: autodiff disagrees with finite differences (rel {rel:.2e})"


def test_the_string_objectives_still_work():
    for name in ("equidistribution", "energy", "huang"):
        _, fem, _ = _peak(size=0.25)
        fem.solve(adapt=jno.solve.relocate(objective=name, max_iters=3, lr=2e-3))
        assert fem.adapt_history, name


def _channel_with_a_surface_term():
    """Poisson on [0,2]x[0,1] carrying ONE surface term, so the facet quadrature tables get built."""
    d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.25).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ct = d.variable("top", where=lambda x, y: y > 1.0 - 1e-9, split=True)
    cl = d.variable("left", where=lambda x, y: x < 1e-9, split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    vt, vl = phi.bind(x=ct[0], y=ct[1]), phi.bind(x=cl[0], y=cl[1])
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, 0.3 * vl, u(xb, yb) - 0.0], quad_degree=3)
    return fem, vi, vt


def test_fem_eval_assembles_a_surface_term_exactly():
    """``fem.eval`` used to refuse every boundary term. The integral has a known value, so this pins
    the assembly rather than merely its shape: summing over the DOFs gives ``int_R 1 dS`` because a
    Lagrange basis is a partition of unity."""
    fem, vi, vt = _channel_with_a_surface_term()
    sol = fem.solve()
    assert np.asarray(fem.eval(1.0 * vt, sol)).sum() == pytest.approx(2.0, rel=1e-10)  # top edge length
    assert np.asarray(fem.eval(1.0 * vi, sol)).sum() == pytest.approx(2.0, rel=1e-10)  # area
    # volume and surface terms in ONE call, each assembled on its own support
    assert np.asarray(fem.eval([1.0 * vi, 1.0 * vt], sol)).sum() == pytest.approx(4.0, rel=1e-10)


def test_a_surface_term_without_facet_tables_refuses_by_name():
    """The tables are tabulated at BUILD time and only when the form itself carries a surface term, so
    a surface readout on a purely-volume problem has nothing to integrate against. It must say that
    rather than fail on `NoneType` unpacking six frames inside the element kernel."""
    d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.3).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ct = d.variable("top", where=lambda x, y: y > 1.0 - 1e-9, split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    vt = phi.bind(x=ct[0], y=ct[1])
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0], quad_degree=3)
    sol = fem.solve()
    with pytest.raises(NotImplementedError, match="facet quadrature tables were never built"):
        fem.eval(1.0 * vt, sol)


def test_an_unknown_string_objective_is_still_refused():
    _, fem, _ = _peak(size=0.3)
    with pytest.raises(ValueError, match="must be 'energy', 'equidistribution', 'huang', or a weak-form"):
        fem.solve(adapt=jno.solve.relocate(objective="nonsense", max_iters=1))


def _stokes_channel(size=0.5):
    """Stokes in a channel whose top wall is traction-free -- NOT a no-flux condition, so a wrongly
    shaped wall carries material through it and ``u.n`` on that wall is the free-surface residual.

    The bottom is **no-slip**, and that is not decoration. With a symmetry bottom instead, uniform flow
    ``u = (1,0)``, ``p = 0`` satisfies every equation and every boundary condition for ANY shape of the
    top wall -- MEASURED: the solution stayed uniform to 2.3e-14 and moved by 9.8e-14 when the wall was
    displaced by 0.1. The objective was then purely geometric, so the problem exercised the normals and
    the facet measure but never the solve, which is the half that matters here. No-slip couples them:
    the flow must develop a profile, so the channel's shape sets it (``max|du| = 7.2e-02`` for the same
    0.1 displacement) and ``d(objective)/d(vertex)`` genuinely runs through the solve.
    """
    d = jno.Shape.rect(0.0, 0.0, 3.0, 1.0, size=size).domain()
    # Deform BEFORE tagging. A region's predicate is re-evaluated against the CURRENT points (both by
    # `tag_node_mask` and by relocate, which re-applies every predicate after moving the mesh), so a
    # mesh deformed after tagging leaves the tags describing a geometry that no longer exists -- here
    # "top" kept 1 node of 7 and the movable set kept none, and the rebuild then failed with a term
    # whose region no longer resolved. The wall is straight, so its predicate stays exact.
    pts = np.asarray(d.mesh.points)
    pts[:, 1] = pts[:, 1] * (1.0 - 0.22 * pts[:, 0] / 3.0 * pts[:, 1])  # a deliberately wrong wall
    d.mesh.points = pts
    _wall = lambda X, Y: Y > 1.0 - 0.22 * X / 3.0 - 1e-9  # noqa: E731  (the deformed top, exactly)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    x, y, _ = d.variable("interior", split=True)
    cin = d.variable("inlet", where=lambda X, Y: X < 1e-9, split=True)
    cbot = d.variable("bottom", where=lambda X, Y: Y < 1e-9, split=True)
    ct = d.variable("top", where=_wall, normals=True, split=True)
    cmov = d.variable("tmov", where=lambda X, Y: _wall(X, Y) & (X > 1e-9), split=True)
    cmov[1].trainable(name="ty")
    eu, ev = jno.np.symgrad(u, [x, y]), jno.np.symgrad(v, [x, y])
    dd = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    pp, qq = p.bind(x=x, y=y), q.bind(x=x, y=y)
    vt, ut = v.bind(x=ct[0], y=ct[1]), u.bind(x=ct[0], y=ct[1])
    nx, ny = ct[-2], ct[-1]
    fem = jno.fem(
        [
            2.0 * dd(eu, ev) - pp * jno.np.trace(ev),
            -qq * jno.np.trace(eu),
            0.0 * vt[0],  # traction-free top; also what builds the facet quadrature tables
            u(cin[0], cin[1])[0] - 1.0,
            u(cin[0], cin[1])[1] - 0.0,
            u(cbot[0], cbot[1])[0] - 0.0,  # no-slip: see the docstring -- this is what couples u to the wall
            u(cbot[0], cbot[1])[1] - 0.0,
            p.pin(),
        ],
        quad_degree=3,
    )
    return d, fem, (ut[0] * nx + ut[1] * ny) ** 2, vt


def test_a_surface_objective_carrying_its_own_test_function_is_differentiable_in_the_wall():
    """The free-surface objective, and the reason the TEST function may be supplied by hand.

    ``int_wall (u.n)^2`` driven to zero IS the free-surface condition. What matters is that its
    gradient with respect to the WALL VERTICES is right -- the descent is nothing but that gradient --
    so it is checked against central differences. The facet normals are rebuilt from the moving
    vertices, so this also pins that ``n`` follows the mesh rather than staying at its initial value.
    """
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _criterion_weak_terms
    from jno.utils.solver.linear import sparse_lu_solve

    d, fem, flux, vt = _stokes_channel()
    weak, _ = _criterion_weak_terms(fem, flux * vt[0])  # test function supplied, so none is bound here
    vals = {
        s["name"]: jnp.asarray(np.asarray(d.mesh.points)[np.asarray(s["ids"], int), s["axis"]]) for s in d._trainable_coords
    }
    op = fem.operator

    def scalar(vv):
        a, b = op.evaluate(vv)
        rhs = jnp.asarray(b).reshape(-1)
        uu = sparse_lu_solve(a, rhs) if hasattr(a, "indices") else jnp.linalg.solve(jnp.asarray(a), rhs)
        return jnp.sum(jnp.asarray(fem.eval(weak, uu, args=vv)))

    val, grad = jax.value_and_grad(scalar)(vals)
    ad = np.asarray(grad["ty"])
    assert float(val) > 0.0, "the deliberately-wrong wall should carry through-flow"
    assert np.linalg.norm(ad) > 1e-6, "a free surface cannot be found from a vanishing gradient"
    h = 1e-6
    fd = []
    for j in range(ad.size):
        vp = {k: (v.at[j].add(h) if k == "ty" else v) for k, v in vals.items()}
        vm = {k: (v.at[j].add(-h) if k == "ty" else v) for k, v in vals.items()}
        fd.append((float(scalar(vp)) - float(scalar(vm))) / (2 * h))
    fd = np.array(fd)
    rel = np.linalg.norm(fd - ad) / max(np.linalg.norm(fd), 1e-30)
    assert rel < 1e-5, f"autodiff disagrees with finite differences on the wall gradient (rel {rel:.2e})"


def test_a_traced_surface_objective_finds_the_free_surface():
    """The motivating case, end to end: minimising the through-flow on a wall MOVES that wall.

    No exact answer is asserted -- with a no-slip bottom the free surface is not analytic (it is flat
    only in the decoupled symmetry-bottom variant the fixture docstring rejects). What is asserted is
    that the residual the free surface is DEFINED by falls substantially, and that the mesh survives.
    MEASURED at these settings: 2.88e-01 -> 2.54e-02, 11.4x (12.5x at 120 rounds, so this is not yet
    converged -- it is a descent, not a root-find).
    """
    d, fem, flux, vt = _stokes_channel()
    p0 = np.asarray(d.mesh.points)[:, :2].copy()
    fem.solve(adapt=jno.solve.relocate(objective=flux * vt[0], max_iters=60, lr=3e-2))
    objs = [e["objective"] for e in fem.adapt_history]
    assert min(objs) < objs[0] / 5.0, f"through-flow was not reduced: {objs[0]:.3e} -> {min(objs):.3e}"
    p1 = np.asarray(fem.domain.mesh.points)[:, :2]
    assert np.abs(p1 - p0).max() > 1e-3, "the wall did not move"
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    e = p1[cells]
    det = (e[:, 1, 0] - e[:, 0, 0]) * (e[:, 2, 1] - e[:, 0, 1]) - (e[:, 2, 0] - e[:, 0, 0]) * (e[:, 1, 1] - e[:, 0, 1])
    assert np.all(np.sign(det) == np.sign(det[0])), "the moved mesh is tangled"


def test_an_unbindable_objective_says_how_to_fix_it():
    """When the criterion reaches its region only through a bound view, the test function cannot be
    auto-bound (identity, not tag, is what the binder compares). That must arrive as an instruction,
    not as a trace-level 'coord binding conflict'."""
    from jno.utils.solver.fem_adapt import _criterion_weak_terms

    _d, fem, flux, _vt = _stokes_channel()
    with pytest.raises(ValueError, match=r"Carry the test function yourself"):
        _criterion_weak_terms(fem, flux, 1)  # field 1 = the scalar pressure test


def test_a_surface_objective_may_omit_the_test_function():
    """The test function is supplied automatically when the objective does not carry one -- including
    for an objective built from a BOUND field plus normals, which exposes no free coordinates of its
    own (a bound view absorbs them). That case has to recover the region's coordinate OBJECTS rather
    than re-fetch them: `domain.variable()` mints new Variables on every call and the binder compares
    identity, not tag, so a re-fetch raises 'coord binding conflict for x' between two Variables that
    both read `gauss_top`."""
    from jno.utils.solver.fem_adapt import _criterion_weak_terms

    for size in (0.5, 0.4, 0.34):
        _d, fem, flux, vt = _stokes_channel(size=size)
        bare, _ = _criterion_weak_terms(fem, flux)  # no test function supplied
        given, _ = _criterion_weak_terms(fem, flux * vt[0])  # one supplied by hand
        assert bare is not None and given is not None, size
