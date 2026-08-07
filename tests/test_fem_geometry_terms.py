"""Mesh motion written as a **term**: ``coord.d(t) - velocity`` in the ``jno.fem([...])`` list.

A coordinate is one of exactly three things, and each is an existing spelling — no new method:

=================  ==========================  ==========================================
a coordinate is    you write                   who moves it
=================  ==========================  ==========================================
fixed              nothing                     nobody
free               ``coord.trainable()``       an optimiser, or ``jno.solve.relocate()``
determined         ``coord.d(t) - v`` (a term) the march
free *and*         both                        the march, from a design-variable start
determined
=================  ==========================  ==========================================

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


@pytest.fixture(autouse=True)
def _x64():
    """Mesh motion requires x64 and raises without it (see ``run_mesh_motion``): the transfer locates
    quadrature points in the previous mesh, which in float32 carries ~4e-4 — enough for a mesh that never
    moves to drift 1.5e-3 from the fixed-mesh march, against 2.6e-10 here."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


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


# ── the state transfer: conservative L2 projection ───────────────────────────────────────────────────


def _bump_march(n_steps, v=0.5, size=0.1):
    """A Gaussian bump on a mesh translating rigidly in y, with kappa ~ 0 and NO Dirichlet BC. The exact
    answer is 'the field never changes at a fixed spatial point', so any change is pure transfer error."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, 0.2, n_steps + 1))
    u, vv = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    _xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), vv.bind(x=xi, y=yi)
    bump = jno.np.exp(-40.0 * ((ci[0] - 0.5) ** 2 + (ci[1] - 0.5) ** 2))
    fem = jno.fem([ui.t * vi + 1e-12 * (ui.x * vi.x + ui.y * vi.y), u(ci[0], ci[1]) - bump, yb.d(tb) - v])
    return fem.solve()


def test_the_transfer_does_not_get_worse_as_the_step_shrinks():
    """The property the pointwise re-interpolation did not have.

    It was applied once per step, so its error ACCUMULATED: the same total displacement cost -27.6 % of
    the peak over 2 steps and -33.0 % over 16, i.e. refining ``dt`` made the answer worse — the opposite
    of what a first-order scheme should do. The conservative L2 projection reverses the trend (-11.5 % ->
    -9.1 %). Asserted as a trend, not a threshold, because the absolute loss is dominated by how well the
    bump is resolved at this ``h``."""
    peaks = {n: float(np.asarray(_bump_march(n).states[-1]).max()) for n in (2, 16)}
    assert peaks[16] >= peaks[2] - 1e-9, (
        f"refining dt made the transfer worse: peak {peaks[2]:.5f} at 2 steps vs {peaks[16]:.5f} at 16"
    )


def test_a_still_mesh_transfers_the_field_untouched():
    """Zero velocity must be a bit-for-bit identity, at every frame. With the meshes equal the projection
    is ``M u_new = M u_old``, so anything but equality means the load vector and the mass disagree."""
    traj = _bump_march(8, v=0.0)
    first = np.asarray(traj.states[0])
    for k, s in enumerate(traj.states):
        assert np.asarray(s) == pytest.approx(first, rel=1e-9, abs=1e-12), f"frame {k} drifted on a still mesh"


def test_the_transfer_conserves_the_integral_better_than_pointwise_sampling():
    """``int u`` over a domain that cannot change (interior vertices jittered, boundary held).

    The projection conserves whatever its load vector integrated — ``sum_i phi_i = 1`` — so the residual is
    quadrature error on an integrand with kinks, not a lost invariant. Measured ~2e-4 relative against the
    pointwise route's 3e-3 to 9e-3, i.e. more than an order of magnitude. Compared against the route it
    replaced rather than against zero, since exact conservation needs a supermesh."""
    from jno.utils.solver.fem_adapt import (
        _l2_transfer_jax,
        _locate_in_one_ring_jax,
        _mesh_boundary_facets,
        _one_ring_cells,
        _simplex_measure_divisor,
    )

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.08).domain()
    P = jnp.asarray(np.asarray(d.mesh.points)[:, :2])
    C = np.asarray(d.mesh.cells_dict["triangle"]).astype(np.int64)
    n = P.shape[0]
    xy = np.asarray(P)
    u = jnp.asarray(np.exp(-40.0 * ((xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2)))

    on_b = np.zeros(n, dtype=bool)
    on_b[np.unique(np.asarray(_mesh_boundary_facets(d)[1]).reshape(-1))] = True
    disp = np.zeros((n, 2))
    disp[~on_b] = np.random.default_rng(0).normal(0.0, 0.012, ((~on_b).sum(), 2))
    Xn = P + jnp.asarray(disp)

    def integral(vals, pts):
        v = np.asarray(pts)[C]
        meas = np.abs(np.linalg.det(v[:, 1:, :] - v[:, :1, :])) / _simplex_measure_divisor(2)
        return float(np.sum(meas * np.asarray(vals)[C].mean(axis=1)))

    i0 = integral(u, P)
    l2, esc = _l2_transfer_jax(P, Xn, C, 2, u, [0, n])
    assert not bool(jnp.any(esc)), "an interior jitter should never leave its own cell patch"

    cand, cmask = _one_ring_cells(C, n)
    idx, w, _e, _c = _locate_in_one_ring_jax(P, C, cand, cmask, Xn)
    pw = jnp.einsum("qk,qk->q", jnp.asarray(w, dtype=u.dtype), u[idx])

    d_l2 = abs(integral(l2, Xn) - i0) / i0
    d_pw = abs(integral(pw, Xn) - i0) / i0
    assert d_l2 < d_pw / 5.0, f"L2 drift {d_l2:.2e} is not clearly better than pointwise {d_pw:.2e}"
    assert d_l2 < 5e-3, f"L2 drift {d_l2:.2e} is larger than the quadrature error should allow"


def test_the_transfer_between_identical_meshes_is_the_identity():
    """The sharp self-test for the projection machinery: with ``X_new == X_old`` the system is
    ``M u_new = M u_old``, so a mismatch between how the load vector and the mass are assembled — a
    transposed element Jacobian, a mis-ordered quadrature gather — shows up immediately. It did: the first
    version contracted the Jacobian on the wrong axis and returned an error of 5.4e-01 here."""
    from jno.utils.solver.fem_adapt import _l2_transfer_jax

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    P = jnp.asarray(np.asarray(d.mesh.points)[:, :2])
    C = np.asarray(d.mesh.cells_dict["triangle"]).astype(np.int64)
    xy = np.asarray(P)
    u = jnp.asarray(np.sin(3.0 * xy[:, 0]) + xy[:, 1])

    out, esc = _l2_transfer_jax(P, P, C, 2, u, [0, P.shape[0]])
    assert not bool(jnp.any(esc)), "a zero displacement cannot escape its own cell"
    assert np.asarray(out) == pytest.approx(np.asarray(u), rel=1e-9, abs=1e-10)


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


@pytest.mark.parametrize(
    "law, expected, stale",
    [
        # v = 1 + t over [0, 0.4] in 4 steps of 0.1: forward Euler sums dt*(1 + t_n) for t_n = 0, .1, .2, .3
        ("affine", 1.0 + 0.1 * sum(1.0 + 0.1 * k for k in range(4)), 1.4),
        # v = t: pure time dependence, so a frozen t gives NO MOTION AT ALL
        ("pure", 1.0 + 0.1 * sum(0.1 * k for k in range(4)), 1.0),
    ],
)
def test_a_velocity_may_depend_explicitly_on_time(law, expected, stale):
    """``t`` in a velocity law is the step's time, not the seed grid's first entry.

    A temporal variable does not live in the tag's own pool — that is (B, T, N, D) with D = dim, purely
    spatial — it resolves from the separate ``__time__`` context key. The march rebuilt the tag entry from
    the carried positions and left ``__time__`` at the seed grid, so every step evaluated at t = ts[0]:
    ``yb.d(tb) - tb`` produced no motion whatever, and ``- (1 + tb)`` gave 1.40000 against forward Euler's
    1.46000. No error either way.

    Checked against the closed form rather than against the eager oracle, which takes no ``t``."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, 0.4, 5))
    _xb, yb, tb = d.variable("boundary", split=True)
    v = (1.0 + tb) if law == "affine" else tb
    ymax = float(_heat(d, yb.d(tb) - v).solve().meshes[-1][0][:, 1].max())

    assert ymax == pytest.approx(expected, rel=1e-6), f"expected forward Euler {expected:.5f}, got {ymax:.5f}"
    assert abs(ymax - stale) > 1e-3, f"the velocity still reads t = t0: {ymax:.5f} vs the frozen answer {stale:.5f}"


def test_a_velocity_may_read_a_SECOND_regions_coordinates():
    """The driver refreshes every region the law reads, not only the one it drives.

    ``velocity()`` rebuilt the driven tag's context entry and left the rest of the context on the seed
    mesh for the whole march. Two tags over the IDENTICAL vertex set make that visible with nothing else
    changing: reading its own coordinate compounded correctly, reading the twin returned exactly the
    frozen-seed answer (1.40000 against 1.46410) with no error."""

    def run(read_twin):
        # A FRESH domain per solve: `solve()` leaves the domain on the final moved mesh, so a second solve
        # here would re-resolve `y > 1 - 1e-6` against vertices that have already moved past it.
        d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, 0.4, 5))
        d.tag("top", lambda x, n, names: x[:, 1] > 1.0 - 1e-6)
        d.tag("twin", lambda x, n, names: x[:, 1] > 1.0 - 1e-6)  # the same vertices, a different name
        _xt, yt, tt = d.variable("top", split=True)
        _xw, yw, _tw = d.variable("twin", split=True)
        return float(_heat(d, yt.d(tt) - (yw if read_twin else yt)).solve().meshes[-1][0][:, 1].max())

    own, other = run(False), run(True)

    assert own == pytest.approx(other, rel=1e-9), (
        f"reading a second region froze the law at the seed mesh: {other:.5f} vs {own:.5f}"
    )
    assert own > 1.0 + 0.4 + 1e-3, f"dy/dt = y must compound; {own:.5f} is the single-Euler-step answer"


def test_a_law_reading_a_region_the_driver_cannot_move_fails_loud():
    """The complement of the fix: a region whose samples are NOT mesh vertices cannot be carried with the
    mesh, so its values would go stale exactly as the second-tag defect did. Refuse instead."""
    from jno.utils.solver.fem_adapt import _geometry_motion_specs, _geometry_velocity_fn

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, 0.2, 5))
    _xb, yb, tb = d.variable("boundary", split=True)
    fem = _heat(d, yb.d(tb) - 0.5)
    spec = _geometry_motion_specs(fem, d)[0]

    # a context entry that exists but whose points are nowhere near a vertex
    d.context["floaty"] = np.full((7, 2), 50.0)
    spec["term"] = spec["term"]  # unchanged; patch the walker's answer instead
    import jno.utils.solver.fem_adapt as fa

    real = fa._tags_read
    fa._tags_read = lambda _e: ["boundary", "floaty"]
    try:
        with pytest.raises(NotImplementedError, match="not mesh vertices"):
            _geometry_velocity_fn(spec, d)
    finally:
        fa._tags_read = real


def test_save_ts_is_rejected_rather_than_ignored():
    """It used to be a named parameter of the driver that was never read, so it both did nothing AND
    slipped past the unknown-keyword check. A request for 3 frames returned the grid's 4."""
    d = _dom()
    fem = _heat(d, yb_of(d).d(tb_of(d)) - 0.1)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        fem.solve(save_ts=np.array([0.0, 0.1, 0.2]))


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
    idx_j, w_j, esc, _c = _locate_in_one_ring_jax(jnp.asarray(pts), cells, cand, mask, jnp.asarray(new))
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
    _, _, esc, _c = _locate_in_one_ring_jax(jnp.asarray(pts), cells, cand, mask, jnp.asarray(new))
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
    _, _, esc, _c = _locate_in_one_ring_jax(jnp.asarray(pts), cells, cand, mask, jnp.asarray(new))
    assert int((np.asarray(esc) & _interior_mask(cells, len(pts))).sum()) > 0


def test_a_zero_displacement_transfer_is_the_identity():
    """Not moving must not diffuse the field — the weights have to land on the vertex itself."""
    from jno.utils.solver.fem_adapt import _locate_in_one_ring_jax

    pts, cells, _new, cand, mask = _ring_case()
    field = np.sin(2.4 * pts[:, 0]) + 0.6 * pts[:, 1] ** 2
    idx, w, esc, _c = _locate_in_one_ring_jax(jnp.asarray(pts), cells, cand, mask, jnp.asarray(pts))
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
        i, w, _, _c = _locate_in_one_ring_jax(p, cells, cand, mask, jnp.asarray(new))
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


# ── where the mesh STARTS as a design variable ───────────────────────────────────────────────────────


def _init_mesh_fem(size=0.3, steps=4):
    """A moving-mesh problem whose interior region's y-coordinates are a `.trainable()` design variable.

    Completes the coordinate table: the region is FREE (its start is a design variable) and the boundary
    is DETERMINED (the march moves it). Those two used to be mutually exclusive."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, 0.2, steps + 1))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    ym = d.variable("mov", where=lambda x, y: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8), split=True)[1]
    ym.trainable(name="Y0")
    ids = np.asarray(d._trainable_coords[0]["ids"], dtype=int)
    fem = jno.fem(
        [
            ui.t * vi + 0.05 * (ui.x * vi.x + ui.y * vi.y),
            u(xb, yb) - 0.0,
            u(ci[0], ci[1]) - 1.0,
            yb.d(tb) - 0.5 * yb,
        ]
    )
    return fem, np.asarray(d.mesh.points)[ids, 1], ids


def test_a_trainable_coordinate_composes_with_a_geometry_term():
    """`.trainable()` and `coord.d(t) - v` used to be mutually exclusive — the driver raised outright.

    They are different roles: the tag says where the mesh STARTS, the term says where it GOES. The tag
    seeds the march's initial geometry and is never put in `args`, so it cannot race the driver's own
    per-step axis parameters."""
    fem, y0, ids = _init_mesh_fem()
    moved = np.asarray(y0) + 0.05
    traj = fem.solve(Y0=moved)
    assert np.asarray(traj.meshes[0][0])[ids, 1] == pytest.approx(moved, abs=1e-6), (
        "the supplied starting coordinates did not become the march's first frame"
    )
    assert np.all(np.isfinite(np.asarray(traj.states[-1])))


def test_the_march_is_differentiable_in_the_initial_mesh():
    """``∂(field)/∂X₀`` — the last of the three target gradients.

    It flows through every downstream step: the harmonic extension, the state transfer's simplex
    selection and each θ-step. Rebuilt per evaluation because ``solve()`` leaves the domain moved."""

    def objective(Y):
        fem, _y0, _ids = _init_mesh_fem()
        return jnp.sum(jnp.asarray(fem.solve(Y0=Y).states[-1]) ** 2)

    _fem, y0, _ids = _init_mesh_fem()
    g = jax.grad(objective)(jnp.asarray(y0))
    assert bool(jnp.isfinite(g).all()), "∂/∂X₀ is not finite"
    assert float(jnp.linalg.norm(g)) > 1e-6, "∂/∂X₀ vanished — the initial mesh never reached the march"

    j = int(np.argmax(np.abs(np.asarray(g))))
    h = 1e-3

    def bumped(dv):
        return float(objective(jnp.asarray(y0).at[j].add(dv)))

    fd = (bumped(h) - bumped(-h)) / (2 * h)
    assert float(g[j]) == pytest.approx(fd, rel=5e-2), f"AD {float(g[j]):.6e} vs FD {fd:.6e}"


def test_the_traced_normals_are_outward_and_agree_with_the_domain_s_own():
    """The normals a moving tag reads must be **outward**, and the domain must report the same ones.

    Agreement alone is not the property. This test used to assert only that, and it passed while both
    routes carried the same wrong rule — orient away from the mesh centroid, which is exact for a
    star-shaped domain and wrong anywhere else. So it is now an absolute check first (against a direction
    known from the geometry) and an agreement check second."""
    from jno.utils.solver.fem_adapt import _facet_outward_sign, _tag_facet_vertex_ids, _vertex_normals_jax

    # This exact aspect ratio / resolution exposed an earlier flip (a tall narrow strip, 6 front
    # vertices): on a FLAT tag `n . (facet_centre - tag_centroid)` is zero to round-off (1.1e-16) and
    # carries no sign information at all.
    H = 0.620063
    d = jno.Shape.rect(0.0, 0.0, 0.35, H, size=0.07).domain(time=(0.0, 0.2, 5))
    d.tag("top", lambda x, n, names: x[:, 1] > H - 1e-6)
    d.tag("bottom", lambda x, n, names: x[:, 1] < 1e-6)
    d.variable("top", normals=True, split=True)
    d.variable("bottom", normals=True, split=True)
    pts = jnp.asarray(np.asarray(d.mesh.points)[:, :2])

    for tag, expect in (("top", +1.0), ("bottom", -1.0)):
        facets = _tag_facet_vertex_ids(d, tag, 2)
        ids = np.unique(np.asarray(facets).reshape(-1))
        assert ids.size >= 3, f"{tag}: only {ids.size} facet vertices — the check would be vacuous"
        got = np.asarray(_vertex_normals_jax(pts, facets, 2, _facet_outward_sign(d, facets, 2)))
        assert np.allclose(got[ids, 1], expect, atol=1e-5), f"{tag}: ny should be {expect}, got {got[ids, 1]}"
        assert np.allclose(np.asarray(d.normals_by_tag[tag])[:, 1], expect, atol=1e-5)


def test_the_normals_are_outward_on_a_CONCAVE_boundary():
    """The case the agreement-only test could not see: an annulus, whose inner ring is concave.

    Outward-from-the-material on the inner ring points *into the hole*, i.e. **towards** the mesh centroid
    — the exact opposite of what the centroid rule concludes. Every producer got it wrong and they all
    agreed, so nothing failed. Measured before the fix, ``n·r̂`` on the inner ring: the traced route and
    ``_facet_normals`` returned +1 (should be −1), and the k-NN PCA fit behind ``get_boundary_normals``
    returned values scattered from −0.346 to +0.311 — not a normal at all.

    Asserted on **both** rings so neither half is vacuous."""
    from jno.domain.mesh_utils import MeshUtils
    from jno.utils.solver.fem_adapt import _facet_outward_sign, _tag_facet_vertex_ids, _vertex_normals_jax

    C = np.array([0.5, 0.5])
    d = (jno.Shape.disk(0.5, 0.5, 0.4) - jno.Shape.disk(0.5, 0.5, 0.15)).domain(size=0.06)

    def radial(p):
        r = np.asarray(p)[:, :2] - C
        return r / np.linalg.norm(r, axis=1, keepdims=True), np.linalg.norm(r, axis=1)

    # 1) the vertex normals `domain.variable(tag, normals=True)` is built from
    n_host, ids = MeshUtils.get_boundary_normals(d.mesh)
    rh, rn = radial(np.asarray(d.mesh.points)[ids])
    dot = np.sum(np.asarray(n_host)[:, :2] * rh, axis=1)
    inner = rn < 0.25
    assert inner.sum() >= 5 and (~inner).sum() >= 5, "both rings must be represented or the check is vacuous"
    assert np.allclose(dot[inner], -1.0, atol=1e-5), f"inner ring is not outward-from-material: {dot[inner]}"
    assert np.allclose(dot[~inner], +1.0, atol=1e-5), f"outer ring is not outward: {dot[~inner]}"

    # 2) the traced route the march reads, on the same concave ring
    d.tag("hole", lambda x, n, names: ((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2) < 0.2**2)
    d.variable("hole", normals=True, split=True)
    facets = _tag_facet_vertex_ids(d, "hole", 2)
    hid = np.unique(np.asarray(facets).reshape(-1))
    traced = np.asarray(
        _vertex_normals_jax(jnp.asarray(np.asarray(d.mesh.points)[:, :2]), facets, 2, _facet_outward_sign(d, facets, 2))
    )
    rh_h, _rn_h = radial(np.asarray(d.mesh.points)[hid])
    assert np.allclose(np.sum(traced[hid] * rh_h, axis=1), -1.0, atol=1e-5), "traced normals point into the solid"


# ── the generality matrix: what already works, and where the walls are ───────────────────────────────


def test_a_geometry_term_marches_in_3D():
    """Nothing in the driver is 2-D, and nothing tested it: every other motion test is a unit square.
    A box driven on z must translate by exactly ``v*T``, since a uniform boundary velocity is a rigid
    translation the harmonic extension reproduces exactly."""
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.45).domain(time=(0.0, 0.2, 4))
    u, v = d.fem_symbols()
    xi, yi, zi, ti = d.variable("interior", split=True)
    xb, yb, zb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi, t=ti), v.bind(x=xi, y=yi, z=zi)
    fem = jno.fem(
        [
            ui.t * vi + 0.05 * (ui.x * vi.x + ui.y * vi.y + ui.z * vi.z),
            u(xb, yb, zb) - 0.0,
            u(ci[0], ci[1], ci[2]) - 1.0,
            zb.d(tb) - 0.1,
        ]
    )
    traj = fem.solve()
    p0, p1 = np.asarray(traj.meshes[0][0]), np.asarray(traj.meshes[-1][0])
    assert p1.shape == p0.shape and p1.shape[1] == 3
    assert np.allclose(p1[:, 2] - p0[:, 2], 0.1 * 0.2, atol=1e-9), "z did not translate as prescribed"
    assert np.allclose(p1[:, :2], p0[:, :2], atol=1e-9), "x/y moved, but only z was given a velocity"
    assert np.isfinite(np.asarray(traj.states[-1])).all()


def test_a_geometry_term_marches_a_NONLINEAR_transient():
    """A state-dependent coefficient puts the block on its Newton branch, which the motion driver never
    exercised. It works — and the solution must genuinely respond to the moved mesh, so this compares a
    moving march against a still one rather than only asserting finiteness."""

    def run(vel):
        d = _dom(size=0.4)
        u, v = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        xb, yb, tb = d.variable("boundary", split=True)
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
        fem = jno.fem(
            [
                ui.t * vi + 0.05 * (1.0 + ui**2) * (ui.x * vi.x + ui.y * vi.y),
                u(xb, yb) - 0.0,
                u(ci[0], ci[1]) - 1.0,
                yb.d(tb) - vel,
            ]
        )
        traj = fem.solve()
        return np.asarray(traj.states[-1]), np.asarray(traj.meshes[-1][0])

    moving, Xm = run(1.0)
    still, Xs = run(0.0)
    assert np.isfinite(moving).all(), "a nonlinear moving-mesh march is not finite"
    assert np.abs(Xm - Xs).max() > 0.1, "the mesh did not move"
    assert np.abs(moving - still).max() > 1e-6, "the nonlinear solve ignored the moved mesh"


def test_a_geometry_term_marches_COUPLED_scalar_fields():
    """The DOF guard allows N scalar-P1 fields, and the transfer loops over the blocks. Untested until
    now; a coupled pair is what a Stefan problem with a solute field looks like."""
    d = _dom(size=0.4)
    a, qa = d.fem_symbols(names=("a", "qa"))
    b, qb = d.fem_symbols(names=("b", "qb"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ai, bi = a.bind(x=xi, y=yi, t=ti), b.bind(x=xi, y=yi, t=ti)
    vai, vbi = qa.bind(x=xi, y=yi), qb.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            ai.t * vai + 0.05 * (ai.x * vai.x + ai.y * vai.y) - bi * vai,
            bi.t * vbi + 0.05 * (bi.x * vbi.x + bi.y * vbi.y),
            a(xb, yb) - 0.0,
            b(xb, yb) - 0.0,
            a(ci[0], ci[1]) - 1.0,
            b(ci[0], ci[1]) - 0.5,
            yb.d(tb) - 0.1,
        ]
    )
    traj = fem.solve()
    n_verts = len(np.asarray(traj.meshes[0][0]))
    final = np.asarray(traj.states[-1])
    assert final.shape == (2 * n_verts,), f"expected two scalar blocks, got {final.shape}"
    assert np.isfinite(final).all()
    # the two fields must stay distinct -- a shared block would make them identical
    assert np.abs(final[:n_verts] - final[n_verts:]).max() > 1e-6


# ── any nodal-Lagrange order and value shape ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("dim, order", [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)])
def test_the_in_trace_lagrange_basis_matches_basix_and_differentiates_at_zero(dim, order):
    """The traced basis must agree with basix in VALUE and stay finite in GRADIENT.

    The gradient half is not redundant. The first version built the monomials as ``xi ** e``, whose value
    is right everywhere — but whose derivative is taken by the general power rule as ``0 · xi**(-1)``,
    i.e. ``0 · inf = NaN``, at ``xi == 0``. A reference coordinate is exactly zero whenever a quadrature
    point lands on a cell edge, so this put NaNs into ``d(march)/dX₀`` while every forward test stayed
    green. Sample points ON the edges and vertices are included here deliberately."""
    from jno.utils.solver.fem_adapt import _eval_lagrange_traced, _lagrange_monomial_coeffs, _tabulate_lagrange_at

    exps, coeffs = _lagrange_monomial_coeffs(dim, order)
    rng = np.random.default_rng(0)
    xs = rng.random((60, dim))
    xs = xs[xs.sum(axis=1) <= 1.0]
    edges = np.zeros((3, dim))  # the origin, and points on two coordinate edges: exact zeros
    edges[1, 0] = 0.5
    edges[2, -1] = 0.5
    xs = np.vstack([xs, edges])

    got = np.asarray(_eval_lagrange_traced(jnp.asarray(xs), exps, coeffs))
    assert got == pytest.approx(_tabulate_lagrange_at(dim, order, xs), abs=1e-12), "traced basis != basix"
    assert np.allclose(got.sum(axis=1), 1.0, atol=1e-12), "the basis is not a partition of unity"

    g = jax.jacrev(lambda p: _eval_lagrange_traced(p, exps, coeffs).sum())(jnp.asarray(xs))
    assert bool(jnp.isfinite(g).all()), "the basis gradient is not finite (xi**0 differentiates to NaN at 0)"
    # sum(phi) is identically 1, so its derivative is identically 0 -- an independent check on the whole route
    assert float(jnp.abs(g).max()) < 1e-10, "d(sum phi)/d(xi) should vanish"


def _order_march(order=1, shape=(), vel=0.1, size=0.4, nt=4):
    """A moving-mesh heat problem at a chosen element order and value shape."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, 0.2, nt))
    u, v = d.fem_symbols(order=order, value_shape=shape)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    if shape:
        weak = jno.np.inner(ui.t, vi) + 0.05 * (
            jno.np.inner(jno.np.grad(ui, xi), jno.np.grad(vi, xi)) + jno.np.inner(jno.np.grad(ui, yi), jno.np.grad(vi, yi))
        )
        terms = [
            weak,
            u(xb, yb)[0] - 0.0,
            u(xb, yb)[1] - 0.0,
            u(ci[0], ci[1])[0] - 1.0,
            u(ci[0], ci[1])[1] - 0.5,
        ]
    else:
        terms = [ui.t * vi + 0.05 * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0]
    return d, jno.fem([*terms, yb.d(tb) - vel])


@pytest.mark.parametrize(
    "order, shape, label", [(1, (), "P1"), (2, (), "P2"), (3, (), "P3"), (1, (2,), "vector P1"), (2, (2,), "vector P2")]
)
def test_a_moving_mesh_carries_any_lagrange_order_and_value_shape(order, shape, label):
    """The P1 wall is gone: the transfer is basis-aware, so a field of any nodal-Lagrange order and value
    shape marches. It used to raise for everything but scalar P1.

    Two things make the higher-order case cheap rather than a rewrite. The mesh geometry is P1 whatever
    the field order — a moved simplex stays straight-sided — so the quadrature map and the point location
    are shared by every field. And the P{k} connectivity is unchanged by a topology-preserving move, so
    the seed assembly's `cells_f` stays valid and the moved DOF *coordinates* are never needed."""
    d, fem = _order_march(order=order, shape=shape)
    traj = fem.solve()
    final = np.asarray(traj.states[-1])
    n_comp = shape[0] if shape else 1

    assert np.isfinite(final).all(), f"{label}: the march is not finite"
    assert final.shape[0] > len(np.asarray(traj.meshes[0][0])) * n_comp or order == 1, (
        f"{label}: a P{order} field should carry more DOFs than there are vertices"
    )
    p0, p1 = np.asarray(traj.meshes[0][0]), np.asarray(traj.meshes[-1][0])
    assert np.allclose(p1[:, 1] - p0[:, 1], 0.1 * 0.2, atol=1e-9), f"{label}: the mesh did not translate as prescribed"
    # the maximum principle still holds: u = 0 on the boundary, <= 1 initially
    assert -0.2 < final.min() and final.max() < 1.2, f"{label}: left a sane envelope [{final.min()}, {final.max()}]"


@pytest.mark.parametrize("order", [1, 2, 3])
def test_a_still_mesh_reproduces_the_fixed_mesh_march_at_any_order(order):
    """The anchor that pins the whole path — parametric assembly, per-step solve, basis-aware transfer —
    against the ordinary solver, at each order. A defect in any of them shows up as a number rather than
    as a plausible-looking trajectory."""

    def mk(with_motion):
        d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, 0.3, 7))
        u, v = d.fem_symbols(order=order)
        xi, yi, ti = d.variable("interior", split=True)
        xb, yb, tb = d.variable("boundary", split=True)
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
        geom = (yb.d(tb) - 0.0,) if with_motion else ()
        return d, jno.fem([ui.t * vi + 0.05 * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0, *geom])

    _dm, fem_m = mk(True)
    moving = np.asarray(fem_m.solve().states[-1])
    d_f, fem_f = mk(False)
    sol = fem_f.solve()
    fixed = np.asarray(jno.core([sol.mean], domain=d_f).eval([sol]))[-1]

    assert moving == pytest.approx(fixed, rel=1e-6, abs=1e-8), (
        f"P{order}: zero motion should reproduce the fixed-mesh march; max|d| = {np.abs(moving - fixed).max():.3e}"
    )


def test_a_mixed_order_coupled_system_moves_as_one():
    """Two fields of DIFFERENT order in one system — the shape a Taylor-Hood pair takes. Each field's own
    order, connectivity and DOF count drive its own projection; they share the mesh, the quadrature map
    and the point location."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain(time=(0.0, 0.2, 4))
    a, qa = d.fem_symbols(names=("a", "qa"), order=2)
    b, qb = d.fem_symbols(names=("b", "qb"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ai, bi = a.bind(x=xi, y=yi, t=ti), b.bind(x=xi, y=yi, t=ti)
    va, vb = qa.bind(x=xi, y=yi), qb.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            ai.t * va + 0.05 * (ai.x * va.x + ai.y * va.y) - bi * va,
            bi.t * vb + 0.05 * (bi.x * vb.x + bi.y * vb.y),
            a(xb, yb) - 0.0,
            b(xb, yb) - 0.0,
            a(ci[0], ci[1]) - 1.0,
            b(ci[0], ci[1]) - 0.5,
            yb.d(tb) - 0.1,
        ]
    )
    traj = fem.solve()
    off = [int(x) for x in fem.offsets]
    n_verts = len(np.asarray(traj.meshes[0][0]))

    assert off[1] - off[0] > n_verts, "the P2 block should carry more DOFs than there are vertices"
    assert off[2] - off[1] == n_verts, "the P1 block's DOFs are the vertices"
    assert np.isfinite(np.asarray(traj.states[-1])).all()
    # each field resamples onto a reference mesh through its OWN basis
    for f in (0, 1):
        r = np.asarray(traj.resample(d, field=f))
        assert r.shape == (len(traj), n_verts), f"field {f} resampled to {r.shape}"
        assert np.isfinite(r).all()


def test_a_non_nodal_family_fails_loud():
    """The transfer tabulates a NODAL Lagrange basis. A Nedelec / RT / Argyris DOF is an edge circulation
    or a normal moment, and there is nothing sensible the projection could do with one, so the wall moved
    from 'scalar P1 only' to 'nodal Lagrange only' rather than disappearing.

    Asserted at `_field_layout`, which is where the driver asks the question. Driving a non-nodal
    *transient* to `solve()` would be stopped earlier by the assembler's own gate, testing that instead of
    this one."""
    from jno.utils.solver.fem_adapt import _field_layout

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain()
    u, v = d.fem_symbols(space="Morley")
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    Hu, Hv = jno.np.hessian(ui, [xi, yi]), jno.np.hessian(vi, [xi, yi])
    fem = jno.fem([jno.np.inner(Hu, Hv, n_contract=2) - 1.0 * vi, u(xb, yb) - 0.0])

    with pytest.raises(NotImplementedError, match="nodal-Lagrange"):
        _field_layout(fem)


def test_mesh_motion_requires_x64():
    """The transfer locates quadrature points in the previous mesh — interior points, where the
    barycentric solve is far weaker than at a vertex. Measured in float32: located weights off by 3.9e-04,
    and a mesh that never moves drifting 1.5e-03 from the fixed-mesh march (2.6e-10 under x64). Refuse
    rather than return that quietly."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        d = _dom(size=0.4)
        fem = _heat(d, yb_of(d).d(tb_of(d)) - 0.1)
        with pytest.raises(NotImplementedError, match="jax_enable_x64"):
            fem.solve()
    finally:
        jax.config.update("jax_enable_x64", prev)


# ── the scheme itself: what order is it, actually? ───────────────────────────────────────────────────
#
# Everything above tests the MECHANISM -- that the velocity is read correctly, the normals point the
# right way, the transfer is conservative. None of it measures the SCHEME. `run_mesh_motion` documents
# "first-order in the step"; these measure it.
#
# The manufactured solution is chosen so no part of the driver's stated scope is exercised by accident:
#
#     Omega(t) = [0,1] x [ct, 1+ct]      driven by `yb.d(tb) - c`
#     u*(x,y,t) = sin(pi x) sin(pi (y - ct)) exp(-lambda t),   lambda = 2 pi^2 kappa
#     f = u*_t - kappa lap(u*) = -pi c exp(-lambda t) sin(pi x) cos(pi (y - ct))
#
# u* vanishes on all four MOVING edges at every time, so the Dirichlet data stays homogeneous and never
# has to follow the boundary -- which the driver says it cannot do. A constant velocity is reproduced
# EXACTLY by the explicit (forward-Euler) mesh update, so the mesh is exact at every step and what is
# measured is the solution alone. And the source is a function of the coordinates, so it exercises the
# forcing-on-the-moved-mesh path that nothing else here touches.

_MMS_KAPPA = 0.05
_MMS_LAMBDA = 2.0 * np.pi**2 * _MMS_KAPPA


def _mms(size, nt, c, t_end=0.2, order=1):
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, t_end, nt))
    u, v = d.fem_symbols(order=order)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    src = -np.pi * c * jno.np.exp(-_MMS_LAMBDA * ti) * jno.np.sin(np.pi * xi) * jno.np.cos(np.pi * (yi - c * ti))
    u0 = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    return jno.fem(
        [
            ui.t * vi + _MMS_KAPPA * (ui.x * vi.x + ui.y * vi.y) - src * vi,
            u(xb, yb) - 0.0,
            u(ci[0], ci[1]) - u0,
            yb.d(tb) - c,
        ]
    )


def _l2_on(vals, X, cells):
    """``||vals||_L2`` over the P1 mesh, by quadrature."""
    from basix import CellType, make_quadrature

    from jno.utils.solver.fem_adapt import _tabulate_lagrange_at

    qp, qw = make_quadrature(CellType.triangle, 2)
    phi = _tabulate_lagrange_at(2, 1, np.asarray(qp))
    v = np.asarray(X)[np.asarray(cells)][:, :, :2]
    detJ = np.abs(np.linalg.det(np.stack([v[:, i + 1] - v[:, 0] for i in range(2)], axis=2)))
    eq = np.einsum("qn,cn->cq", phi, np.asarray(vals)[np.asarray(cells)])
    return float(np.sqrt(np.sum(np.asarray(qw)[None, :] * detJ[:, None] * eq**2)))


@pytest.mark.slow
@pytest.mark.parametrize("c, label", [(0.5, "moving"), (0.0, "still")])
def test_the_march_is_first_order_in_the_step(c, label):
    """The documented claim, measured: the operator-split ALE scheme is **first order in dt**.

    Measured against a fine-dt reference ON THE SAME MESH rather than against ``u*``, which is what makes
    this a clean measurement: the spatial error is then identical in both solutions and cancels out of the
    difference. Comparing against ``u*`` directly instead mixes in an O(h²) floor, and the temporal and
    spatial errors have OPPOSITE SIGNS — so the naive sweep shows rates of +1.4 then −0.4 as they cancel
    and separate again, which looks like a broken scheme and is only a broken measurement.

    Observed rates at h = 0.06 (512-step reference): 0.99 / 1.01 / 1.04 / 1.10 still, and
    1.14 / 1.12 / 1.12 / 1.10 moving. The motion multiplies the error CONSTANT by ~3x — that is the state
    transfer — but leaves the ORDER intact, which is the property being asserted."""
    ref = np.asarray(_mms(0.08, 129, c).solve().states[-1])
    errs, ns = [], (5, 9, 17)
    for nt in ns:
        traj = _mms(0.08, nt, c).solve()
        u = np.asarray(traj.states[-1])
        errs.append(_l2_on(u - ref, traj.meshes[-1][0], traj.meshes[-1][1]))

    rates = [np.log(errs[i] / errs[i + 1]) / np.log(2.0) for i in range(len(errs) - 1)]
    assert all(e > 0 for e in errs), f"{label}: degenerate errors {errs}"
    assert all(0.75 < r < 1.45 for r in rates), f"{label}: expected first order, got rates {rates} from {errs}"


@pytest.mark.slow
def test_a_moving_march_converges_in_space_and_higher_order_pays():
    """Refining the mesh must converge on a MOVING domain too, and a higher-order field must be worth
    its DOFs there — the projection happens every step, so it is not obvious that it is.

    Measured at h = 0.05, 128+ steps: P1 reaches 1.84e-03 and P2 1.00e-03 -> 1.00e-04, i.e. **~18x more
    accurate at the same mesh**. P1's spatial rate on the moving domain measures 1.51 then 1.76,
    approaching the expected 2."""
    errs = {}
    for order, h in ((1, 0.2), (1, 0.1), (2, 0.2), (2, 0.1)):
        traj = _mms(h, 129, 0.5, order=order).solve()
        X, cells = np.asarray(traj.meshes[-1][0]), np.asarray(traj.meshes[-1][1])
        # compare on the P1 vertices, which every order shares (they are the first DOFs of the block)
        n_v = X.shape[0]
        u = np.asarray(traj.states[-1])[:n_v]
        ex = np.sin(np.pi * X[:, 0]) * np.sin(np.pi * (X[:, 1] - 0.5 * 0.2)) * np.exp(-_MMS_LAMBDA * 0.2)
        errs[(order, h)] = _l2_on(u - ex, X, cells)

    for order in (1, 2):
        assert errs[(order, 0.1)] < errs[(order, 0.2)], f"P{order} did not converge in h: {errs}"
    assert errs[(2, 0.1)] < errs[(1, 0.1)], f"P2 is not more accurate than P1 at the same mesh: {errs}"
