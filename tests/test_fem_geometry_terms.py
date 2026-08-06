"""Mesh motion written as a **term**: ``coord.d(t) - velocity`` in the ``jno.fem([...])`` list.

A coordinate is one of exactly three things, and each is an existing spelling — no new method:

===============  ==========================  ==========================================
a coordinate is  you write                   who moves it
===============  ==========================  ==========================================
fixed            nothing                     nobody
free             ``coord.trainable()``       an optimiser, or ``jno.solve.relocate()``
determined       ``coord.d(t) - v`` (a term) the march
===============  ==========================  ==========================================

These cases pin down the **classification**: which residuals are geometry terms, which are emphatically
not, and that nothing about it is boundary-specific — an interior region, a boundary and a ``where=``
predicate all resolve the same way, per axis.
"""

import numpy as np
import pytest

import jno
from jno.trace import mesh_velocity


def _dom(size=0.3, t=(0.0, 0.2, 5)):
    return jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=t)


def test_a_coordinate_time_derivative_is_a_geometry_term():
    """``yb.d(tb) - v`` is recognised structurally — by containing d(spatial)/d(temporal), so the existing
    ``Variable.d`` is all that is needed to express it."""
    d = _dom()
    xb, yb, tb = d.variable("boundary", split=True)
    found = mesh_velocity(yb.d(tb) - 0.5)
    assert found is not None, "a coordinate time-derivative term must classify as geometry"
    coord, tvar, _jac = found
    assert coord.tag == "boundary"
    assert coord.dim[0] == 1, "must identify the AXIS -- y is column 1, and tagging is per-axis"
    assert tvar.axis == "temporal"


def test_geometry_terms_are_not_boundary_specific():
    """The generality that matters: an interior region and a ``where=`` predicate classify identically.
    ``domain.variable`` resolves interior / boundary / predicate the same way, so mesh motion is not a
    boundary feature that happens to be reusable — it is a coordinate feature."""
    d = _dom()
    xi, yi, ti = d.variable("interior", split=True)
    xc, yc, tc = d.variable("core", where=lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.04, split=True)

    interior = mesh_velocity(xi.d(ti) - 0.3 * (yi - 0.5))
    core = mesh_velocity(yc.d(tc) - 1.0)
    assert interior is not None and interior[0].tag == "interior"
    assert interior[0].dim[0] == 0, "xi moves the x column"
    assert core is not None and core[0].tag == "core"
    assert core[0].dim[0] == 1, "yc moves the y column"


def test_a_velocity_may_read_the_solved_field():
    """The point of putting the law in the term list: it is ordinary traced math, so an interface law can
    reference the solution (a Stefan front ``v_n = -(k/L)·∇T·n``) instead of a Python callback."""
    d = _dom()
    u, _v = d.fem_symbols()
    xb, yb, tb, nx, ny = d.variable("boundary", normals=True, split=True)  # (x, y, t, nx, ny)
    tf = u.bind(x=xb, y=yb).freeze(np.zeros(len(d.mesh.points)))
    term = yb.d(tb) - (-(0.5) * (tf.x * nx + tf.y * ny)) * ny
    found = mesh_velocity(term)
    assert found is not None, "a state-dependent velocity is still a geometry term"
    assert found[0].dim[0] == 1


def test_ordinary_terms_are_never_claimed_as_geometry():
    """The classifier must not steal constraints. A weak form is poisoned by its test function even though
    its integrand mentions a coordinate derivative; a Dirichlet BC and a plain expression carry no
    coordinate time-derivative at all."""
    d = _dom()
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)

    for label, term in [
        ("weak form", ui.t * vi + ui.x * vi.x + ui.y * vi.y),
        ("Dirichlet", u(xb, yb) - 0.0),
        ("initial condition", u(ci[0], ci[1]) - 1.0),
        ("plain coordinate expression", yb - 0.5),
    ]:
        assert mesh_velocity(term) is None, f"{label} must not classify as a geometry term"


def test_one_geometry_term_moves_one_coordinate():
    """``xb.d(tb) + yb.d(tb) - 1`` under-determines the motion — one residual, two unknowns. Fail loud and
    say the fix, rather than silently moving whichever coordinate the walk happened to find first."""
    d = _dom()
    xb, yb, tb = d.variable("boundary", split=True)
    with pytest.raises(ValueError, match="may move ONE coordinate"):
        mesh_velocity(xb.d(tb) + yb.d(tb) - 1.0)


def test_the_rest_of_the_problem_is_unaffected():
    """A geometry term is pulled out BEFORE weak-form / Dirichlet classification, so the FE problem it
    accompanies has exactly the DOFs and mode it would have had on its own."""
    d = _dom()
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    physics = [ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0]

    plain = jno.fem(list(physics))
    moving = jno.fem([*physics, yb.d(tb) - 0.5])

    assert moving._mode == plain._mode == "transient"
    assert moving.dofs == plain.dofs, "a geometry term must not add or consume an FE unknown"
    assert len(moving._geometry) == 1 and len(plain._geometry) == 0


# ── the motion driver ────────────────────────────────────────────────────────────────────────────────


def _heat(d, *geometry, kappa=0.05):
    """A plain heat problem plus whatever geometry terms are given."""
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    return jno.fem(
        [
            ui.t * vi + kappa * (ui.x * vi.x + ui.y * vi.y),
            u(xb, yb) - 0.0,
            u(ci[0], ci[1]) - 1.0,
            *geometry,
        ]
    )


def test_prescribed_motion_converges_first_order_to_the_analytic_domain():
    """``yb.d(tb) - 0.5*yb`` is ``dy/dt = y/2``, so the box must stretch to ``exp(0.5·T)``. The driver is
    explicit in the velocity, so it should reproduce forward Euler exactly and converge at FIRST order —
    that is the documented scheme, and asserting the *rate* is what would catch it silently changing."""
    T, errs = 0.4, []
    for nt in (9, 17, 33):
        d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, T, nt))
        _xb, yb, tb = d.variable("boundary", split=True)
        traj = _heat(d, yb.d(tb) - 0.5 * yb).solve()
        ymax = float(traj.meshes[-1][0][:, 1].max())
        assert ymax == pytest.approx((1 + 0.5 * T / (nt - 1)) ** (nt - 1), rel=1e-6), "not forward Euler"
        errs.append(abs(ymax - np.exp(0.5 * T)))

    assert errs[0] > errs[1] > errs[2], f"not converging under refinement: {errs}"
    rate = errs[0] / errs[2]
    assert 3.0 < rate < 5.0, f"expected ~4x (first order over 4x refinement), got {rate:.2f}"


def test_the_velocity_is_re_evaluated_on_the_MOVED_mesh():
    """Regression, and it bites silently. A tag's coordinates are cached in ``domain.context`` and
    ``move_mesh`` moves ``domain.mesh.points`` without touching it, so ``0.5*yb`` kept reading the ORIGINAL
    y — exponential growth collapsed to a single Euler step over the whole interval (1.2000 rather than
    1.2184), with no error anywhere. Anything less than compounding growth means the re-sample was lost."""
    T, nt = 0.4, 9
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, T, nt))
    _xb, yb, tb = d.variable("boundary", split=True)
    ymax = float(_heat(d, yb.d(tb) - 0.5 * yb).solve().meshes[-1][0][:, 1].max())

    stale = 1.0 + 0.5 * T  # what a velocity frozen at the initial mesh would give
    assert ymax > stale + 1e-3, f"velocity looks frozen at the initial mesh: {ymax:.6f} vs stale {stale:.6f}"


def test_motion_is_per_axis_and_holds_the_untagged_column():
    """Tagging is literal: a term on ``yb`` moves the y column and leaves x alone. That is the lever for
    sliding a node within a wall instead of pushing it through one."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.3, 7))
    _xb, yb, tb = d.variable("boundary", split=True)
    traj = _heat(d, yb.d(tb) - 0.4).solve()
    p0, p1 = traj.meshes[0][0], traj.meshes[-1][0]

    assert np.allclose(p0[:, 0], p1[:, 0], atol=1e-12), "x moved, but only y was given a velocity"
    assert p1[:, 1].max() > p0[:, 1].max() + 0.05, "y did not move"


def test_an_interior_region_moves_too():
    """The generality the classifier promises, end to end: a ``where=`` region in the middle of the domain
    is driven, and the mesh around it (including the outer boundary) relaxes harmonically to accommodate."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain(time=(0.0, 0.2, 5))
    xc, _yc, tc = d.variable("core", where=lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.05, split=True)
    core_ids = np.asarray(jno.trace.Variable._region_vertex_ids(d, "core", np.asarray(d.mesh.points)), dtype=int)
    assert core_ids.size > 0, "the test needs a non-empty interior region"

    traj = _heat(d, xc.d(tc) - 0.3).solve()
    p0, p1 = traj.meshes[0][0], traj.meshes[-1][0]
    assert np.allclose(p1[core_ids, 0] - p0[core_ids, 0], 0.3 * 0.2, atol=1e-9), "the core did not move as prescribed"
    assert len(p1) == len(p0), "mesh motion is connectivity-preserving"


def test_the_velocity_may_read_the_solved_field():
    """A state-dependent law: the boundary speed is proportional to the solution's own boundary gradient,
    the shape a Stefan condition takes. The frozen field is re-pinned to the live state each step."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.2, 6))
    u, _v = d.fem_symbols()
    parts = d.variable("boundary", normals=True, split=True)
    xb, yb, tb, nx, ny = parts[0], parts[1], parts[2], parts[-2], parts[-1]
    tf = u.bind(x=xb, y=yb).freeze(np.zeros(len(d.mesh.points)))

    traj = _heat(d, yb.d(tb) - 0.05 * (tf.x * nx + tf.y * ny) * ny).solve()
    p0, p1 = traj.meshes[0][0], traj.meshes[-1][0]
    assert len(p1) == len(p0)
    assert not np.allclose(p0[:, 1], p1[:, 1]), "a state-dependent velocity produced no motion at all"


def test_a_geometry_term_needs_a_transient_problem():
    """``coord.d(t) - v`` moves the mesh *in time*. On a steady problem there is no time to move through,
    and the term would be silently inert."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0, yb.d(tb) - 0.5])
    with pytest.raises(NotImplementedError, match="needs a transient problem"):
        fem.solve()


def test_motion_does_not_silently_share_the_march():
    """The motion driver owns the march and re-assembles each step, so it cannot also be owned by adapt=
    or a solver slot. Refuse rather than let one of them win quietly."""
    d = _dom()
    _xb, yb, tb = d.variable("boundary", split=True)
    fem = _heat(d, yb.d(tb) - 0.1)
    with pytest.raises(NotImplementedError, match="does not compose"):
        fem.solve(adapt=jno.solve.remesh())


def test_two_terms_may_not_prescribe_the_same_coordinate():
    """Regions overlap freely — a corner belongs to both edges — so two terms naming the same vertex *and*
    the same axis is easy to write by accident. Scattering in list order would silently let the last one
    win, which is a wrong velocity with no symptom."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, 0.2, 5))
    _xl, yl, tl = d.variable("lo", where=lambda x, y: y < 0.5, split=True)
    _xa, ya, ta = d.variable("all", where=lambda x, y: np.ones_like(x, dtype=bool), split=True)
    with pytest.raises(ValueError, match="only one velocity"):
        _heat(d, yl.d(tl) - 0.1, ya.d(ta) - 0.2).solve()


def test_two_terms_may_drive_different_axes_of_one_region():
    """The complement: x and y of the same region are separate degrees of freedom, so a term each is a
    perfectly well-posed 2-D velocity and must be allowed."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.2, 5))
    xb, yb, tb = d.variable("boundary", split=True)
    traj = _heat(d, xb.d(tb) - 0.2, yb.d(tb) - 0.3).solve()
    p0, p1 = traj.meshes[0][0], traj.meshes[-1][0]
    shift = p1[:, :2] - p0[:, :2]
    assert np.allclose(shift[:, 0], 0.2 * 0.2, atol=1e-9), "x did not translate as prescribed"
    assert np.allclose(shift[:, 1], 0.3 * 0.2, atol=1e-9), "y did not translate as prescribed"


def test_a_motion_that_would_tangle_the_mesh_raises():
    """Connectivity-preserving only: a motion large enough to invert elements must fail loud, not return a
    solve on a tangled mesh.

    Note the velocity has to be *differential*. Driving the whole boundary at one constant speed is a rigid
    translation — the interior follows harmonically and nothing inverts however far it goes. ``-4y`` pulls
    the top edge down while the bottom stays put, which folds the domain through itself."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 1.0, 3))
    _xb, yb, tb = d.variable("boundary", split=True)
    with pytest.raises(ValueError, match="inverts or collapses"):
        _heat(d, yb.d(tb) + 4.0 * yb).solve()
