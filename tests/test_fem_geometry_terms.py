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

import jax
import jax.numpy as jnp
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


def yb_of(d):
    return d.variable("boundary", split=True)[1]


def tb_of(d):
    return d.variable("boundary", split=True)[2]


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


def test_a_position_independent_region_marches_too():
    """The boundary tag does not depend on a coordinate window, so it marches however far it goes.

    A ``where=`` region that moves out of its own predicate does NOT — it silently stops being driven. That
    is a known defect (see the note in ``run_mesh_motion``); it is not asserted here because pinning the
    current wrong behaviour into a test would make the fix look like a regression.
    """
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.4, 6))
    _xb, yb, tb = d.variable("boundary", split=True)
    traj = _heat(d, yb.d(tb) - 0.5).solve()
    p0, p1 = traj.meshes[0][0], traj.meshes[-1][0]
    assert p1[:, 1].max() > p0[:, 1].max() + 0.15, "the boundary did not move"


def test_the_SOLUTION_stays_finite_while_the_mesh_moves():
    """The assertion whose absence shipped a silent NaN.

    Every other motion test checks mesh positions and node counts. For a PRESCRIBED velocity the mesh does
    not depend on the field, so the mesh can be exactly right while the solution is entirely NaN — which is
    what happened when the driver stopped re-assembling per step: the parametric branch's step operator
    made the block's default BiCGStab break down. Seventeen tests passed on a NaN field.

    So: assert the field. Finite at every frame, obeying the maximum principle (this is heat with u=0 on
    the boundary and u=1 initially, so nothing may exceed those bounds), and actually decaying.
    """
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.4, 9))
    _xb, yb, tb = d.variable("boundary", split=True)
    traj = _heat(d, yb.d(tb) - 0.5 * yb).solve()
    states = [np.asarray(s) for s in traj.states]

    for k, s in enumerate(states):
        s = s.astype(np.float64)
        assert np.isfinite(s).all(), f"solution is not finite at frame {k} (the mesh being right proves nothing)"
        # Bounded, not monotone: a CONSISTENT-mass P1 Galerkin step is not a discrete maximum principle,
        # and this initial condition is discontinuous (1 in the interior, 0 on the boundary), so a small
        # overshoot is the scheme's, not a defect. Measured ~2.3%; the bound is there to catch divergence.
        assert -0.1 < s.min() and s.max() < 1.1, f"frame {k} left a sane envelope: [{s.min():.4f}, {s.max():.4f}]"
    assert states[-1].max() < states[0].max(), "heat did not diffuse — the field is not being marched"


def test_the_solution_matches_a_fixed_mesh_when_the_mesh_does_not_move():
    """A zero velocity must reproduce the ordinary fixed-mesh march exactly. This pins the whole moving-mesh
    path — parametric assembly, per-step solve, state transfer — against the plain solver, so a defect in
    any of them shows up as a number rather than as a plausible-looking trajectory."""

    def _mk(with_motion):
        d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.3, 7))
        _xb, yb, tb = d.variable("boundary", split=True)
        geom = (yb.d(tb) - 0.0,) if with_motion else ()
        return d, _heat(d, *geom)

    d_m, fem_m = _mk(True)
    moving = np.asarray(fem_m.solve().states[-1])

    d_f, fem_f = _mk(False)
    sol = fem_f.solve()
    fixed = np.asarray(jno.core([sol.mean], domain=d_f).eval([sol]))[-1]

    assert np.isfinite(moving).all(), "zero-velocity moving-mesh march is not finite"
    assert moving == pytest.approx(fixed, rel=1e-6, abs=1e-8), (
        f"zero motion should reproduce the fixed-mesh march: max|d| = {np.abs(moving - fixed).max():.3e}"
    )


def test_the_march_can_be_run_twice():
    """Solving twice must work, and continue from where the first march left the mesh.

    It raised. The driver registers the whole mesh as a coordinate parameter to keep the assembly
    parametric, and its own registration from the first call then tripped the guard meant to catch a
    *user's* hand-tagged coordinate — so a second `solve()` complained about a tag the user never wrote.
    Only a coordinate the user tagged is a conflict.

    Continuing rather than restarting is the right semantics: like the h-refinement loop, this driver
    mutates the domain in place, so afterwards ``fem`` and its domain refer to the moved mesh. Each march
    therefore applies the same increment from wherever it starts.
    """
    V, T = 0.3, 0.2
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, T, 5))
    _xb, yb, tb = d.variable("boundary", split=True)
    fem = _heat(d, yb.d(tb) - V)

    y0 = np.asarray(d.mesh.points)[:, 1].copy()
    y1 = fem.solve().meshes[-1][0][:, 1]
    y2 = fem.solve().meshes[-1][0][:, 1]

    assert np.allclose(y1 - y0, V * T, atol=1e-9), "the first march did not move by velocity x horizon"
    assert np.allclose(y2 - y1, V * T, atol=1e-9), "the second march did not apply the same increment from the moved mesh"


# ── differentiability: r-adaptivity moves nodes at fixed connectivity, so gradients must flow ────────


def _heat_kappa(d, *geometry):
    """The same heat problem, but with the diffusivity as a runtime parameter."""
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    kap = jno.np.parameter((1,), name="kap")
    return jno.fem([ui.t * vi + kap * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0, *geometry])


def test_a_runtime_parameter_reaches_the_moving_mesh_march():
    """A parameter's VALUE must reach the assembly, not just its seed.

    ``**kwargs`` was accepted by the driver and dropped on the floor: the block exposes the parameter in
    ``runtime_parameter_exprs`` and the assembly reads it from ``args``, but ``args`` only ever carried the
    mesh-motion axes. A moving-mesh solve therefore used the seed value and said nothing — the silent-wrong
    -answer shape, not a missing feature."""
    d = _dom()
    fem = _heat_kappa(d, yb_of(d).d(tb_of(d)) - 0.5)
    hot = np.asarray(fem.solve(kap=2.0).states[-1])
    cold = np.asarray(fem.solve(kap=0.01).states[-1])
    assert np.all(np.isfinite(hot)) and np.all(np.isfinite(cold))
    assert np.abs(hot - cold).max() > 1e-4, (
        f"kappa never reached the assembly: kap=2.0 and kap=0.01 agree to {np.abs(hot - cold).max():.2e}"
    )


def test_an_unknown_solve_kwarg_raises_rather_than_being_swallowed():
    """The ordinary transient path raises on an unknown kwarg; the motion driver used to accept anything."""
    d = _dom()
    fem = _heat_kappa(d, yb_of(d).d(tb_of(d)) - 0.5)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        fem.solve(not_a_parameter=1.0)


def test_the_field_gradient_flows_through_a_moving_mesh():
    """``d(field)/d(kappa)`` through the march, against central differences.

    r-adaptivity moves nodes at FIXED connectivity, so the march is smooth in its parameters and this
    gradient has to exist. Asserted non-zero as well as correct — a silently vanishing gradient passes any
    relative-error check on its own."""
    d = _dom()
    fem = _heat_kappa(d, yb_of(d).d(tb_of(d)) - 0.5)

    def loss(k):
        return jnp.sum(jnp.asarray(fem.solve(kap=k).states[-1]) ** 2)

    k0 = 0.05
    g = float(jax.grad(loss)(k0))
    assert abs(g) > 1e-8, "d(field)/d(kappa) vanished — the parameter never reached the assembly"
    h = 1e-3
    fd = (float(loss(k0 + h)) - float(loss(k0 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=2e-2), f"AD {g:.6e} vs FD {fd:.6e}"


# ── the velocity as a traced function of the vertex positions ────────────────────────────────────────


def _vel_case(term_fn, size=0.3, state_fn=None):
    """Build a moving-mesh problem and return (spec, domain, state) for a velocity comparison."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, 0.2, 5))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb, nx, ny = d.variable("boundary", normals=True, split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    tf = u.bind(x=xb, y=yb).freeze(np.zeros(len(d.mesh.points)))
    fem = jno.fem(
        [
            ui.t * vi + 0.05 * (ui.x * vi.x + ui.y * vi.y),
            u(xb, yb) - 0.0,
            u(ci[0], ci[1]) - 1.0,
            term_fn(yb, tb, tf, nx, ny),
        ]
    )
    from jno.utils.solver.fem_adapt import _geometry_motion_specs

    pts = np.asarray(d.mesh.points)
    state = np.asarray(fem._op.state0).reshape(-1).astype(np.float64)
    state = (state + 0.7 * np.sin(3.0 * pts[:, 0]) + 0.4 * pts[:, 1]).astype(np.float32)
    return _geometry_motion_specs(fem, d)[0], d, state, fem


@pytest.mark.parametrize(
    "label,term_fn",
    [
        ("prescribed", lambda yb, tb, tf, nx, ny: yb.d(tb) - 0.5 * yb),
        ("constant", lambda yb, tb, tf, nx, ny: yb.d(tb) - 0.4),
        ("scaled derivative", lambda yb, tb, tf, nx, ny: 2.0 * yb.d(tb) - 1.0),
        ("reads the solved field", lambda yb, tb, tf, nx, ny: yb.d(tb) - 0.2 * tf.x),
    ],
)
def test_the_traced_velocity_reproduces_the_host_evaluation(label, term_fn):
    """``_geometry_velocity_fn`` must agree with ``_geometry_velocity`` — on the seed mesh AND on a moved one.

    The traced form is what lets the march be scanned: it reads the vertex positions it is *handed*, where
    the host form re-samples ``domain.context`` (host state that cannot be touched inside ``lax.scan``).
    The moved case is the one that matters; agreeing only on the seed mesh would prove nothing.

    A state-reading law is checked on a NON-TRIVIAL state on purpose — with the zero state both routes
    return zero and the comparison passes without exercising anything.
    """
    from jno.utils.solver.fem_adapt import _geometry_velocity, _geometry_velocity_fn, harmonic_extension, move_mesh

    sp, d, state, fem = _vel_case(term_fn)
    fn = _geometry_velocity_fn(sp, d)
    dim = int(d.dimension)
    pts0 = jnp.asarray(np.asarray(d.mesh.points)[:, :dim])

    host0 = np.asarray(_geometry_velocity(sp, d, state))
    assert np.abs(host0).max() > 1e-6, f"{label}: the reference velocity is ~0 — the test would prove nothing"
    assert np.asarray(fn(pts0, state)) == pytest.approx(host0, abs=2e-6), f"{label}: seed mesh"

    # move, then re-sample the host context (the host route needs that to see the move; the traced one does not)
    disp = np.zeros((pts0.shape[0], dim))
    disp[np.asarray(sp["ids"]), 1] = 0.08
    prescribed = np.isin(np.arange(pts0.shape[0]), np.asarray(sp["ids"]))
    move_mesh(d, np.asarray(harmonic_extension(d, disp, prescribed=prescribed)), copy=False)
    d.variable("boundary", normals=True, split=True)
    from jno.utils.solver.fem_adapt import _geometry_motion_specs

    host1 = np.asarray(_geometry_velocity(_geometry_motion_specs(fem, d)[0], d, state))
    pts1 = jnp.asarray(np.asarray(d.mesh.points)[:, :dim])
    assert np.asarray(fn(pts1, state)) == pytest.approx(host1, abs=2e-6), f"{label}: moved mesh"
    assert not np.allclose(pts0, pts1), "the move did nothing — the moved-mesh half is vacuous"


def test_the_traced_velocity_is_differentiable_in_the_vertex_positions():
    """``∂v/∂X`` must exist and be non-zero — this is the gradient the geometry chain is built on."""
    from jno.utils.solver.fem_adapt import _geometry_velocity_fn

    sp, d, state, _fem = _vel_case(lambda yb, tb, tf, nx, ny: yb.d(tb) - 0.5 * yb)
    fn = _geometry_velocity_fn(sp, d)
    pts = jnp.asarray(np.asarray(d.mesh.points)[:, : int(d.dimension)])
    g = jax.grad(lambda p: jnp.sum(fn(p, state) ** 2))(pts)
    assert bool(jnp.isfinite(g).all()), "∂v/∂X is not finite"
    assert float(jnp.linalg.norm(g)) > 1e-8, "∂v/∂X vanished — the velocity does not see the positions"


# ── the state transfer, without scipy ────────────────────────────────────────────────────────────────


def _ring_case(size=0.12, disp_scale=0.02, seed=0):
    from jno.utils.solver.fem_adapt import _one_ring_cells, _simplex_cell_key

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    dim = int(d.dimension)
    pts = np.asarray(d.mesh.points)[:, :dim].astype(np.float64)
    cells = np.asarray(d.mesh.cells_dict[_simplex_cell_key(dim)]).astype(np.int64)
    rng = np.random.default_rng(seed)
    new = pts + disp_scale * rng.standard_normal(pts.shape)
    cand, mask = _one_ring_cells(cells, pts.shape[0])
    return pts, cells, new, cand, mask


def _interior_mask(cells, n):
    from jno.utils.solver.fem_adapt import _boundary_edges_from_triangles

    on_b = np.zeros(n, dtype=bool)
    on_b[np.unique(np.asarray(_boundary_edges_from_triangles(cells)).reshape(-1))] = True
    return ~on_b


def test_the_one_ring_transfer_reproduces_the_kdtree_route():
    """The scipy-free transfer must give the same FIELD, not merely the same cell.

    A query point on a shared face is legitimately inside two simplices, so comparing chosen cell indices
    would flag a difference that does not exist. What has to agree is the interpolated value."""
    from jno.utils.solver.fem_adapt import _locate_barycentric, _locate_in_one_ring_jax

    pts, cells, new, cand, mask = _ring_case()
    field = np.sin(2.4 * pts[:, 0]) + 0.6 * pts[:, 1] ** 2

    idx_h, w_h, inside_h = _locate_barycentric(pts, cells, new, tol=1e-9, k=32)
    idx_j, w_j, esc = _locate_in_one_ring_jax(jnp.asarray(pts), cells, cand, mask, jnp.asarray(new))
    val_h = np.einsum("qk,qk->q", w_h, field[idx_h])
    val_j = np.einsum("qk,qk->q", np.asarray(w_j), field[np.asarray(idx_j)])

    both = inside_h & ~np.asarray(esc)
    assert both.sum() > 0.5 * len(pts), "too few points located by both routes for the comparison to mean anything"
    assert val_j[both] == pytest.approx(val_h[both], abs=1e-6)


def test_a_moderate_step_never_escapes_an_interior_vertex_s_one_ring():
    """The one-ring is a *candidate set*, so it has to actually contain the answer.

    An interior vertex cannot leave the mesh, so an interior escape means the step outran the local
    element size. A BOUNDARY vertex moving outward legitimately leaves the old mesh entirely — the
    kd-tree route clamps those to the nearest simplex too, so they are expected, not a failure."""
    from jno.utils.solver.fem_adapt import _locate_barycentric, _locate_in_one_ring_jax

    pts, cells, new, cand, mask = _ring_case(disp_scale=0.02)
    _, _, esc = _locate_in_one_ring_jax(jnp.asarray(pts), cells, cand, mask, jnp.asarray(new))
    esc, interior = np.asarray(esc), _interior_mask(cells, len(pts))
    _, _, inside_h = _locate_barycentric(pts, cells, new, tol=1e-9, k=32)

    assert int((esc & interior).sum()) == 0, "a moderate step escaped an interior vertex's one-ring"
    assert int((esc & ~interior).sum()) == int((~inside_h).sum()), (
        "boundary escapes should be exactly the points the kd-tree route also finds outside the old mesh"
    )


def test_an_over_large_step_is_detectable_rather_than_silently_clamped():
    """The failure the one-ring makes visible. The kd-tree route discards its own `inside` flag, so an
    over-large step is absorbed by a nearest-simplex clamp without a word; here it is reportable."""
    from jno.utils.solver.fem_adapt import _locate_in_one_ring_jax

    pts, cells, new, cand, mask = _ring_case(disp_scale=0.06)
    _, _, esc = _locate_in_one_ring_jax(jnp.asarray(pts), cells, cand, mask, jnp.asarray(new))
    assert int((np.asarray(esc) & _interior_mask(cells, len(pts))).sum()) > 0


def test_a_zero_displacement_transfer_is_the_identity():
    """Not moving must not diffuse the field — the weights have to land on the vertex itself."""
    from jno.utils.solver.fem_adapt import _locate_in_one_ring_jax

    pts, cells, _new, cand, mask = _ring_case()
    field = np.sin(2.4 * pts[:, 0]) + 0.6 * pts[:, 1] ** 2
    idx, w, esc = _locate_in_one_ring_jax(jnp.asarray(pts), cells, cand, mask, jnp.asarray(pts))
    got = np.einsum("qk,qk->q", np.asarray(w), field[np.asarray(idx)])
    assert not np.asarray(esc).any(), "a vertex escaped its own one-ring at zero displacement"
    # 1e-6, not 1e-9: the barycentric solve runs in the default float32, and an exact vertex hit still
    # lands ~3.5e-08 out. The quantity being excluded is P1 interpolation DIFFUSION, which on this field
    # would be O(h^2) ~ 1e-2 -- four orders above this bar, so the test still has all its teeth.
    assert got == pytest.approx(field, abs=1e-6)


def test_the_transfer_is_differentiable_in_the_vertex_positions():
    """``∂(transferred field)/∂X`` — the transfer sits inside the march, so it must carry a gradient."""
    from jno.utils.solver.fem_adapt import _locate_in_one_ring_jax

    pts, cells, new, cand, mask = _ring_case()
    field = jnp.asarray(np.sin(2.4 * pts[:, 0]) + 0.6 * pts[:, 1] ** 2)

    def transferred(p):
        i, w, _ = _locate_in_one_ring_jax(p, cells, cand, mask, jnp.asarray(new))
        return jnp.sum(jnp.einsum("qk,qk->q", w, field[i]) ** 2)

    g = jax.grad(transferred)(jnp.asarray(pts))
    assert bool(jnp.isfinite(g).all()) and float(jnp.linalg.norm(g)) > 1e-8


# ── the geometry chain: the mesh trajectory itself carries a gradient ────────────────────────────────


def _stefan(size=0.25, steps=5):
    """A state-reading law, so the interface position depends on the PDE coefficient."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, 0.2, steps + 1))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb, nx, ny = d.variable("boundary", normals=True, split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    tf = u.bind(x=xb, y=yb).freeze(np.zeros(len(d.mesh.points)))
    kap = jno.np.parameter((1,), name="kap")
    return jno.fem(
        [
            ui.t * vi + kap * (ui.x * vi.x + ui.y * vi.y),
            u(xb, yb) - 0.0,
            u(ci[0], ci[1]) - 1.0,
            yb.d(tb) - 0.4 * (tf.x * nx + tf.y * ny) * ny,
        ]
    )


def test_the_mesh_trajectory_is_a_traced_array_not_host_numpy():
    """The frames must come back on the device, or nothing downstream can differentiate through them.

    This is the whole difference the scan buys: the eager driver returned `np.ndarray` meshes, which made
    the mesh a CONSTANT to autodiff no matter what was differentiated."""
    d = _dom()
    _xb, yb, tb = d.variable("boundary", split=True)
    traj = _heat(d, yb.d(tb) - 0.5 * yb).solve()
    assert isinstance(traj.meshes[-1][0], jnp.ndarray), (
        f"mesh frame is {type(traj.meshes[-1][0]).__name__} — the geometry chain is broken"
    )
    assert len(traj.meshes) == len(traj.states) == len(traj.times)


def test_the_interface_position_is_differentiable_in_the_pde_coefficient():
    """``∂(final mesh)/∂κ`` — the gradient that did not exist before the scan.

    The chain is κ → field → Stefan velocity → mesh motion → final vertex positions, so it exercises the
    whole geometry path. A SMOOTH functional of the mesh is used deliberately: ``max`` has a switching
    argmax, and central differences across the switch are meaningless (measured: it reports the wrong
    sign). Small h for the same reason — the transfer's simplex choice is piecewise constant, so a wide
    stencil can straddle a kink.

    The problem is rebuilt inside the objective on purpose: ``solve()`` leaves the domain on the *moved*
    mesh (see ``test_the_march_can_be_run_twice``), so reusing one ``fem`` would start each finite-
    difference evaluation from the previous one's final geometry. Measured, that alone turns FD from
    5.20 into 1.8e+03."""

    def objective(k):
        return jnp.sum(jnp.asarray(_stefan().solve(kap=k).meshes[-1][0])[:, 1] ** 2)

    k0, h = 0.05, 5e-4
    g = float(jax.grad(objective)(k0))
    assert abs(g) > 1e-6, "∂(mesh)/∂κ vanished — the geometry chain is not connected"
    fd = (float(objective(k0 + h)) - float(objective(k0 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=5e-2), f"AD {g:.6e} vs FD {fd:.6e}"


# ── the interface law itself as a design variable ────────────────────────────────────────────────────


def _law_param_fem(size=0.3, steps=4):
    """``yb.d(tb) - v0*yb`` with the law's rate as an ordinary runtime parameter."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, 0.2, steps + 1))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    v0 = jno.np.parameter((1,), name="v0")
    return jno.fem(
        [
            ui.t * vi + 0.05 * (ui.x * vi.x + ui.y * vi.y),
            u(xb, yb) - 0.0,
            u(ci[0], ci[1]) - 1.0,
            yb.d(tb) - v0 * yb,
        ]
    )


@pytest.mark.parametrize("v0", [0.0, 0.5, 1.0])
def test_a_parameter_in_the_law_actually_sets_the_motion(v0):
    """A parameter written into a geometry term must drive the march.

    It was INERT, and silently: a geometry term is pulled out before the weak-form assembly, so its
    parameter never reached the block's ``runtime_parameter_exprs`` and the march ran at the parameter's
    seed — measured, no motion at all and no error. ``dy/dt = v0*y`` under forward Euler gives exactly
    ``(1 + v0*dt)^n``, so the check is against the analytic stepper, not a fixture."""
    ymax = float(jnp.max(jnp.asarray(_law_param_fem().solve(v0=v0).meshes[-1][0])[:, 1]))
    assert ymax == pytest.approx((1.0 + v0 * 0.05) ** 4, rel=1e-5)


def test_the_interface_law_is_differentiable_in_its_own_parameter():
    """``∂(final mesh)/∂(law parameter)`` — the law is a design variable, which is the point of writing it
    in the term list rather than as a callback. Rebuilt per evaluation because ``solve()`` leaves the
    domain on the moved mesh."""

    def objective(p):
        return jnp.sum(jnp.asarray(_law_param_fem().solve(v0=p).meshes[-1][0])[:, 1] ** 2)

    p0, h = 0.5, 1e-3
    g = float(jax.grad(objective)(p0))
    assert abs(g) > 1e-6, "∂(mesh)/∂(law parameter) vanished — the parameter never reached the velocity"
    fd = (float(objective(p0 + h)) - float(objective(p0 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-3), f"AD {g:.6e} vs FD {fd:.6e}"


def test_an_unknown_kwarg_still_raises_when_the_law_has_a_parameter():
    """The accepted set is the union of weak-form and law parameters — not a licence to accept anything."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _law_param_fem().solve(v0=0.5, nope=1.0)
