"""r-adaptivity via the adapt slot — ``FEM.solve(adapt=jno.solve.relocate())``.

Relocates the mesh vertices tagged with :meth:`Variable.trainable` down the **equidistribution** gradient (through the
differentiable solve) with a backtracking mesh-validity line search — the built-in companion of h-refinement
(``run_adaptive_relocate``). Checks that it reduces the objective at **fixed DOF** without tangling across
**scalar, vector, nonlinear, transient, periodic, and complex** problems, and demands at least one
``.trainable()`` coordinate.
"""

import jax
import numpy as np
import pytest

import jno
import jno.jnp_ops as J


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _min_detj(pts, cells):
    v = pts[cells]
    a, b = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    return float(np.min(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))


def _peak_scalar(size=0.14, movable=True):
    """Poisson with a sharp off-center peak source; interior nodes (a central box) tagged trainable."""
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
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)


def test_relocate_reduces_the_defect_and_stays_valid():
    d, fem = _peak_scalar()
    pts0 = np.asarray(d.mesh.points)[:, :2].copy()
    n0 = len(pts0)
    sol = np.asarray(fem.solve(adapt=jno.solve.relocate(max_iters=40, lr=3e-3, quality_floor=0.1))).reshape(-1)
    hist = fem.adapt_history
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    pts_r = np.asarray(fem.domain.mesh.points)[:, :2]

    assert len(hist) >= 5, "relocation should take several steps"
    assert hist[-1]["objective"] < hist[0]["objective"], "relocation must reduce the equidistribution defect"
    # each step records the moved vertices (so a relocation run can be animated); the last is the final mesh
    assert hist[0]["points"].shape == (n0, 2)
    assert np.allclose(hist[-1]["points"], pts_r), "the last recorded mesh should be the final relocated mesh"
    assert len(pts_r) == n0 and sol.shape[0] == n0, "r-adaptivity adds no DOFs (fixed connectivity)"
    assert _min_detj(pts_r, cells) > 0.0, "the relocated mesh must stay valid (no inverted elements)"
    assert np.linalg.norm(pts_r - pts0) > 1e-3, "the interior vertices should actually move toward the feature"


def test_relocate_requires_trainable_coordinates():
    _, fem = _peak_scalar(movable=False)
    with pytest.raises(ValueError, match="no trainable mesh coordinates"):
        fem.solve(adapt=jno.solve.relocate())


def test_relocate_vector_field():
    """Generality: a vector problem relocates too (the monitor sums over components)."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.16).domain()
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xm, ym, _ = d.variable("mov", where=lambda x, y: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8), split=True)
    xm.trainable(name="ix")
    ym.trainable(name="iy")
    vi = phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
    weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2) - (
        f * vi.component(0) + 0.5 * f * vi.component(1)
    )
    fem = jno.fem([weak, u(xb, yb) - 0.0])
    n0 = len(d.mesh.points)
    sol = np.asarray(fem.solve(adapt=jno.solve.relocate(max_iters=25, lr=2e-3))).reshape(-1)
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    assert fem.adapt_history[-1]["objective"] <= fem.adapt_history[0]["objective"], (
        "vector relocation should not raise the defect"
    )
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0, "vector relocation must stay valid"
    assert sol.shape[0] == 2 * n0, "vector solution has 2 DOFs per node, unchanged by relocation"


def _mov(d):
    xm, ym, _ = d.variable("mov", where=lambda x, y: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8), split=True)
    xm.trainable(name="ix")
    ym.trainable(name="iy")


def test_relocate_nonlinear():
    """A steady *nonlinear* problem relocates (the objective's solve is a differentiable Newton solve)."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.18).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    _mov(d)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 10.0 * J.exp(-40.0 * ((xi - 0.6) ** 2 + (yi - 0.35) ** 2))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui * ui * ui * vi - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem._mode == "nonlinear"
    fem.solve(adapt=jno.solve.relocate(max_iters=20, lr=2e-3))
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    assert fem.adapt_history[-1]["objective"] <= fem.adapt_history[0]["objective"]
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0


def test_relocate_transient():
    """A *transient* problem relocates for the whole trajectory (time-averaged state; the coord gradient
    flows through the marched block)."""
    from shapely.geometry import box

    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.18, time=(0.0, 0.3, 11))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    _mov(d)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0])
    assert fem.is_transient
    fem.solve(adapt=jno.solve.relocate(max_iters=15, lr=2e-3))
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    h = fem.adapt_history
    assert h[-1]["objective"] < h[0]["objective"], "transient relocation should reduce the time-averaged defect"
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0


def test_relocate_periodic():
    """A *periodic* problem relocates: interior relocation never touches the boundary ties."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.18).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", where=lambda x, y: x < 1e-6, split=True)
    xr, yr, _ = d.variable("right", where=lambda x, y: x > 1 - 1e-6, split=True)
    bt = d.variable("bt", where=lambda x, y: (y < 1e-6) | (y > 1 - 1e-6), split=True)
    _mov(d)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = J.exp(-40.0 * ((xi - 0.5) ** 2 + (yi - 0.35) ** 2))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xl, yl) - u(xr, yr), u(bt[0], bt[1]) - 0.0])
    fem.solve(adapt=jno.solve.relocate(max_iters=20, lr=2e-3))
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    assert fem.adapt_history[-1]["objective"] <= fem.adapt_history[0]["objective"]
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0


def test_relocate_complex():
    """A *complex* problem relocates: complex is two real blocks (real + imag), and the monitor sums both."""
    from shapely.geometry import box

    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.16)
    u, w = d.fem_symbols(complex=True)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    _mov(d)
    ub, wb = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
    c = 1.0 + 0.5j
    f = jno.complex(
        10.0 * J.exp(-40.0 * ((xi - 0.6) ** 2 + (yi - 0.35) ** 2)),
        8.0 * J.exp(-40.0 * ((xi - 0.35) ** 2 + (yi - 0.6) ** 2)),
    )
    weak = (ub.x * wb.x + ub.y * wb.y) - c * (ub * wb) - f * wb
    fem = jno.fem([weak.real, u.real(xb, yb) - 0.0, u.imag(xb, yb) - 0.0])
    assert fem._mode == "linear" and len(fem.offsets) == 3  # a real 2N block system (real + imag)
    sol = np.asarray(fem.solve(adapt=jno.solve.relocate(max_iters=20, lr=2e-3))).reshape(-1)
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    n0 = len(fem.domain.mesh.points)
    assert fem.adapt_history[-1]["objective"] <= fem.adapt_history[0]["objective"]
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0
    assert sol.shape[0] == 2 * n0, "complex solution = real + imaginary blocks"


def test_relocate_complex_transient_fails_loud():
    """Complex-*transient* cannot carry a trainable coordinate yet — its assembly builds static real Re/Im
    blocks and does not thread runtime parameters — so it fails loud at build (not silently mis-assembled)."""
    from shapely.geometry import box

    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.3, time=(0.0, 0.05, 6))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    _mov(d)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    with pytest.raises(NotImplementedError, match="complex-transient"):
        jno.fem([ui.t * vi + (0.5 + 1j) * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#114 -- relocate() writes back a mesh that is not the geometry it validated. This test "
        "re-solves through `final_err`, which used to dispatch on the PRE-adapt operator (FEM.solve "
        "restored `_op` unconditionally) and so scored relocation on a geometry the caller never "
        "receives. Solving honestly on `fem.domain.mesh.points` gives 1.141e-01 against the uniform "
        "mesh's 9.379e-02 -- relocation LOSES here -- and an independent `jno.fem` built on those "
        "same points agrees to 15+ digits, so it is the loop's 0.51 ratio that is the outlier. "
        "xfail(strict) so this reports the day the two geometries agree; do not relax the assertion "
        "to make it pass."
    ),
)
def test_relocate_beats_a_uniform_mesh_on_an_underresolved_front():
    """The property the old suite never checked: relocation must improve the **answer**, not just an objective.

    Relocation used to descend the FE Dirichlet energy. That is not equidistribution — for a non-convex
    functional the mesh can lower the energy by *under-resolving* the layer, and it did: on this problem the
    energy objective cut the energy 4.949 -> 4.422 while making the final-time error 10.7x WORSE than a
    uniform mesh at the same node count. The objective is now the equidistribution defect of an arclength
    monitor, which targets resolution directly.

    An Allen-Cahn front of width ~2.2*sqrt(2)*eps on an h=0.06 mesh: eps=0.03 spans ~1.5 cells, so a uniform
    mesh under-resolves it and there is something for relocation to win. Measured ratios (relocated/uniform
    final rel-L2) across the sharpness sweep: 4.99 at eps=0.15 (front ~8 cells -- already over-resolved,
    nothing to redistribute), 0.96 at eps=0.06, 0.51 at eps=0.03.

    Energy descent was not deleted for this, only dethroned: it remains ``objective="energy"`` and is the
    right choice on a *fixed singularity*, where it IS the error norm. The two are pinned against each
    other in ``test_the_two_objectives_each_win_on_their_own_problem`` below -- removing the option
    outright is what left the L-shape tutorial asserting a property no reachable setting delivered (#109).
    """
    T, NSTEP, SIZE, EPS = 2.0, 24, 0.06, 0.03

    def build(movable):
        d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=SIZE).domain(time=(0.0, T, NSTEP))
        d.tag("right_edge", lambda x, y: x > 1.0 - 1e-9)
        u, v = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right_edge", split=True)
        ci = d.variable("initial", split=True)
        if movable:  # a SEPARATE movable region, so the boundary vertices are never relocated
            xm, ym, _ = d.variable("mov", where=lambda x, y: (x > 0.02) & (x < 0.98) & (y > 0.02) & (y < 0.98), split=True)
            xm.trainable(name="ix")
            ym.trainable(name="iy")
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
        return d, jno.fem(
            [
                ui.t * vi + EPS**2 * (ui.x * vi.x + ui.y * vi.y) + (u**3 - u) * vi,
                u(xl, yl) - (-1.0),
                u(xr, yr) - (+1.0),
                u(ci[0], ci[1]) - jno.np.tanh((ci[0] - 0.5) / (np.sqrt(2.0) * 0.30)),
            ]
        )

    def final_err(fem, dom):
        sol = fem.solve(nonlinear=jno.solve.newton(direct=True))
        traj = np.asarray(jno.core([sol.mean], domain=dom).eval([sol]))
        p = np.asarray(dom.mesh.points)[:, :2]
        exact = np.tanh((p[:, 0] - 0.5) / (np.sqrt(2.0) * EPS))
        return float(np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact))

    d0, fem0 = build(False)
    err_uniform = final_err(fem0, d0)

    d1, fem1 = build(True)
    n0 = len(d1.mesh.points)
    fem1.solve(adapt=jno.solve.relocate(lr=5e-3, max_iters=30), nonlinear=jno.solve.newton(direct=True))
    err_moved = final_err(fem1, fem1.domain)

    assert len(fem1.domain.mesh.points) == n0, "r-adaptivity must not change the node count"
    assert err_moved < err_uniform, f"relocation should beat a uniform mesh here: {err_moved:.3e} vs {err_uniform:.3e}"
    assert err_moved < 0.75 * err_uniform, f"expected a clear win, got ratio {err_moved / err_uniform:.2f}"


# ── Huang's equidistribution+alignment functional (AdaptSpec.objective="huang") ──────────────────────
#
# Unit tests of the functional itself.  The conventions are the whole risk here: `J` is the Jacobian of
# the INVERSE map (∂ξ/∂x) and the integral is over the PHYSICAL domain, so a sign/inversion slip yields a
# functional that still looks plausible and equidistributes backwards.  `test_..._inverse_map_convention`
# is the test that actually pins that down.

_UNIT_SQUARE = (
    np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    np.array([[0, 1, 2], [0, 2, 3]]),
)


def _huang(pts, pts0, u_nodal, cells, **kw):
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _huang_ea_jax

    return float(_huang_ea_jax(jnp.asarray(pts), jnp.asarray(pts0), jnp.asarray(u_nodal), jnp.asarray(cells), 2, **kw))


def test_huang_functional_matches_its_closed_form_at_the_identity_map():
    """At ``x = ξ`` with a uniform monitor both conditions are exactly met, and the two terms collapse
    onto the same value: ``I = (1−θ)·d^(dp/2)·ρ^(1−p)·|Ω|``.

    Worth pinning as a number rather than a property: it fixes the *normalisation* of both terms and
    their relative weight, which is where ``θ`` vs ``1−2θ`` would silently go wrong."""
    pts, cells = _UNIT_SQUARE
    u = pts[:, 0].copy()  # linear ⇒ |∇u|² is the same in both cells ⇒ ρ = sqrt(2) everywhere
    theta, p, dim, area = 1.0 / 3.0, 1.5, 2, 1.0
    rho = np.sqrt(2.0)
    expected = (1.0 - theta) * dim ** (dim * p / 2.0) * rho ** (1.0 - p) * area
    assert _huang(pts, pts, u, cells) == pytest.approx(expected, rel=1e-10)


def test_huang_functional_scales_as_the_inverse_map_convention_requires():
    """Uniformly scale the *physical* mesh by ``s`` at a fixed computational mesh: ``J = ∂ξ/∂x = s⁻¹I``,
    ``|K| ∝ s^d``, so ``I(s) = I(1)·s^(d(1−p))`` — for ``d=2, p=3/2`` that is ``1/s``.

    This is the convention test.  Had ``J`` been taken as ``∂x/∂ξ`` (the form quoted by several secondary
    sources) the exponent would be ``d(1+p) = +5`` instead of ``−1`` — a factor of 10⁶ apart at ``s=10``,
    while every identity-map check above still passes.  The monitor is scale-free by construction
    (``|∇u|²`` normalised by its own mesh mean), so it does not contaminate the exponent."""
    pts, cells = _UNIT_SQUARE
    u = pts[:, 0].copy()
    base = _huang(pts, pts, u, cells)
    for s in (0.5, 2.0, 5.0):
        assert _huang(s * pts, pts, u, cells) == pytest.approx(base * s ** (2 * (1.0 - 1.5)), rel=1e-9)


def test_huang_functional_is_a_barrier_against_a_flattening_element():
    """``det J → ∞`` as a physical element flattens, and the equidistribution term carries ``(det J)^p``
    against only one factor of ``|K| ∝ det E``, so the functional diverges like ``(det E)^(1−p)``.

    This is the coercivity that is meant to replace the backtracking ``det J`` line search (M2) — so it
    must hold *before* the line search is removed, not after."""
    pts0, cells = _UNIT_SQUARE
    u = pts0[:, 0].copy()
    vals = []
    for gap in (1e-1, 1e-2, 1e-3, 1e-4):
        pts = pts0.copy()
        # slide vertex 3 perpendicularly onto the 0–2 diagonal, so cell [0,2,3] *approaches* flatness
        # (landing it exactly on the diagonal gives det E = 0 and a NaN, not a limit)
        pts[3] = [0.5 - gap / np.sqrt(2.0), 0.5 + gap / np.sqrt(2.0)]
        vals.append(_huang(pts, pts0, u, cells))
    assert np.isfinite(vals).all(), f"non-finite before degeneracy: {vals}"
    assert all(b > a for a, b in zip(vals, vals[1:])), f"not monotone as the element flattens: {vals}"
    assert vals[-1] > 1e3 * vals[0], f"barrier too weak: {vals[0]:.3e} -> {vals[-1]:.3e}"


def test_huang_functional_gradient_is_finite_and_points_somewhere():
    """The driver descends ``jax.grad`` of this — a NaN or an identically-zero gradient would stall
    relocation silently rather than loudly."""
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _huang_ea_jax

    pts0, cells = _UNIT_SQUARE
    u = np.tanh((pts0[:, 0] - 0.5) / 0.1)  # a front, so the monitor is genuinely non-uniform
    g = jax.grad(lambda p: _huang_ea_jax(p, jnp.asarray(pts0), jnp.asarray(u), jnp.asarray(cells), 2))(
        jnp.asarray(pts0 * 1.0)
    )
    g = np.asarray(g)
    assert np.isfinite(g).all(), "non-finite mesh gradient"
    assert np.abs(g).max() > 1e-8, "gradient vanished — relocation would not move"


def test_relocate_with_the_huang_objective_runs_and_keeps_the_mesh_valid():
    """Driver-level: the alternative objective is wired through ``AdaptSpec.objective`` and descends."""
    from jno.utils.solver.fem_adapt import AdaptSpec

    d, fem = _peak_scalar()
    cells = np.asarray(d.mesh.cells_dict["triangle"])
    n0 = len(d.mesh.points)
    fem.solve(adapt=AdaptSpec(relocate=True, max_iters=25, lr=3e-3, objective="huang"))
    h = fem.adapt_history
    assert len(h) > 0
    assert h[-1]["objective"] < h[0]["objective"], f"objective did not fall: {h[0]['objective']} -> {h[-1]['objective']}"
    assert len(fem.domain.mesh.points) == n0, "r-adaptivity must not change the node count"
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0, "mesh tangled"


def test_relocate_rejects_an_unknown_objective():
    from jno.utils.solver.fem_adapt import AdaptSpec

    _d, fem = _peak_scalar()
    with pytest.raises(ValueError, match="objective must be"):
        fem.solve(adapt=AdaptSpec(relocate=True, max_iters=2, objective="dirichlet-energy"))


# ── Monge-Ampere relocation (AdaptSpec.relocate_method="monge_ampere") ───────────────────────────────
#
# x = xi + grad(phi) for a mesh potential phi solving  m(x)*det(I + H(phi)) = theta.
# The displacement being a GRADIENT is what makes the map non-folding, so these tests check that
# structurally -- no line search is involved anywhere below.


def _mesh_grid(n=13):
    """Structured n x n triangulation of the unit square."""
    s = np.linspace(0.0, 1.0, n)
    gx, gy = np.meshgrid(s, s, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    cells = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            cells += [[a, c, d], [a, d, b]]  # positively oriented -- [a,b,d] would give det = -h²
    cells = np.asarray(cells)
    # gmsh hands out positively-oriented cells, so a hand-rolled grid that does not would make every
    # det-J assertion below read as "totally inverted" and mean nothing.
    assert _min_detj(pts, cells) > 0.0, "the reference grid itself must be positively oriented"
    return pts, cells


def _ma_disp(monitor, pts, cells, **kw):
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _monge_ampere_displacement, _p1_operators

    ops = _p1_operators(pts, cells, 2)
    return np.asarray(_monge_ampere_displacement(jnp.asarray(monitor), ops, cells, 2, **kw))


def test_monge_ampere_leaves_an_already_equidistributed_mesh_alone():
    """A constant monitor means the uniform mesh already equidistributes it, so ``φ`` is constant and the
    displacement vanishes. The cheapest possible check that the normalisation ``θ`` is right — get ``θ``
    wrong and a uniform monitor drives a spurious global drift."""
    pts, cells = _mesh_grid()
    d = _ma_disp(np.ones(len(pts)), pts, cells)
    assert np.abs(d).max() < 1e-9, f"uniform monitor moved the mesh by {np.abs(d).max():.2e}"


def test_monge_ampere_pulls_nodes_toward_the_feature():
    """A monitor peaked on the line ``x = 0.5`` must draw nodes toward it — the direction check that a
    sign slip in eq. (3.7) would invert (and which the physical-vertex descent got wrong)."""
    pts, cells = _mesh_grid(15)
    monitor = 1.0 + 12.0 * np.exp(-(((pts[:, 0] - 0.5) / 0.08) ** 2))
    moved = pts + _ma_disp(monitor, pts, cells, n_relax=120, dt=0.3)
    before = float(np.mean(np.abs(pts[:, 0] - 0.5)))
    after = float(np.mean(np.abs(moved[:, 0] - 0.5)))
    assert after < before, f"nodes moved AWAY from the feature: {before:.4f} -> {after:.4f}"


def test_monge_ampere_never_folds_the_mesh_without_a_line_search():
    """The displacement is ``∇φ``, so the map is a gradient and cannot fold. No ``det J`` step control is
    involved — this is the property that lets the line search be deleted, which is what unblocks
    differentiation.

    **The claim is about the WHOLE map**, applied to every vertex — which is what this asserts. Hold a
    subset (as the driver does for every untagged vertex) and the truncated map carries no such guarantee:
    on this same problem, freezing the boundary reaches ``min det J = -1.2e-03`` at settings where the full
    map stays positive throughout. That is why :func:`run_adaptive_relocate` still validity-checks every
    round, and it is covered by :func:`test_monge_ampere_truncated_by_a_frozen_boundary_can_fold`.

    Driven hard: a 13:1 monitor demands a √13 ≈ 3.6× linear compression, and nodes move ~2 cells. Note
    ``dt`` stays inside the relaxation's stability window — divergence is a *separate* failure mode from
    folding, and is covered by :func:`test_monge_ampere_divergence_is_caught_not_silently_accepted`."""
    pts, cells = _mesh_grid(15)
    monitor = 1.0 + 12.0 * np.exp(-(((pts[:, 0] - 0.5) / 0.10) ** 2))
    disp = _ma_disp(monitor, pts, cells, n_relax=400, dt=0.05)
    moved = pts + disp
    assert np.abs(disp).max() > 1.5 / 14, "not actually driven hard -- nodes barely moved"
    assert _min_detj(moved, cells) > 0.0, f"mesh folded: min det J = {_min_detj(moved, cells):.3e}"


def test_monge_ampere_truncated_by_a_frozen_boundary_can_fold():
    """The other half of the guarantee, stated so it cannot be mistaken for a bug later: applying ``∇φ`` to
    only *some* vertices is no longer the Monge-Ampère map, and the result can invert. Same monitor and
    same settings as the whole-map case above, which stays valid — the only difference is the freeze."""
    pts, cells = _mesh_grid(21)
    monitor = np.sqrt(1.0 + 9.0 * np.exp(-(((pts[:, 0] + pts[:, 1] - 1.0) / 0.12) ** 2)))
    disp = _ma_disp(monitor, pts, cells, n_relax=300, dt=0.05)
    bd = (pts[:, 0] < 1e-12) | (pts[:, 0] > 1 - 1e-12) | (pts[:, 1] < 1e-12) | (pts[:, 1] > 1 - 1e-12)
    frozen = pts.copy()
    frozen[~bd] += disp[~bd]
    assert _min_detj(pts + disp, cells) > 0.0, "the whole map must stay valid — that is the guarantee"
    assert _min_detj(frozen, cells) < 0.0, (
        "expected the frozen-boundary truncation to invert here; if the method improved, keep the "
        "distinction documented rather than deleting it — the driver's validity check depends on it"
    )


def test_monge_ampere_divergence_is_caught_not_silently_accepted():
    """The relaxation is explicit in ``Δt`` and *will* diverge past its stability limit (measured: it
    survives ``dt=0.3`` on a 13:1 monitor and blows up by ``dt=0.4``). A diverged solve returns a
    non-finite displacement, and the driver's validity guard must reject that rather than move the mesh
    to NaN — ``nan <= 0`` is ``False``, so a naive ``<= 0`` test lets it straight through."""
    pts, cells = _mesh_grid(15)
    monitor = 1.0 + 60.0 * np.exp(-(((pts[:, 0] - 0.5) / 0.05) ** 2))
    disp = _ma_disp(monitor, pts, cells, n_relax=200, dt=0.4)
    assert not np.isfinite(disp).all(), "expected this setting to diverge; restore the guard test if it no longer does"
    assert not (_min_detj(pts + disp, cells) > 0.0), "a non-finite mesh must not read as valid"


def test_monge_ampere_displacement_is_differentiable_in_the_monitor():
    """The point of this route: the mesh solve sits inside ``jax.grad`` with nothing to trace around.

    There is no concrete branch anywhere in the relaxation — no validity test, no backtracking — so the
    whole thing is a plain ``lax.scan`` over constant pre-factorized operators."""
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _monge_ampere_displacement, _p1_operators

    pts, cells = _mesh_grid(11)
    ops = _p1_operators(pts, cells, 2)

    def spread(scale):
        m = 1.0 + scale * jnp.exp(-(((jnp.asarray(pts[:, 0]) - 0.5) / 0.1) ** 2))
        return jnp.sum(_monge_ampere_displacement(m, ops, cells, 2, n_relax=40, dt=0.2) ** 2)

    g = float(jax.grad(spread)(8.0))
    assert np.isfinite(g), "non-finite gradient through the Monge-Ampere solve"
    assert abs(g) > 1e-10, "gradient vanished — the monitor does not reach the mesh"


def test_relocate_monge_ampere_holds_the_boundary_and_keeps_the_node_count():
    from jno.utils.solver.fem_adapt import AdaptSpec

    d, fem = _peak_scalar()
    pts0 = np.asarray(d.mesh.points)[:, :2].copy()
    cells = np.asarray(d.mesh.cells_dict["triangle"])
    fem.solve(adapt=AdaptSpec(relocate=True, relocate_method="monge_ampere", max_iters=4))
    p1 = np.asarray(fem.domain.mesh.points)[:, :2]
    on_edge = (pts0[:, 0] < 1e-9) | (pts0[:, 0] > 1 - 1e-9) | (pts0[:, 1] < 1e-9) | (pts0[:, 1] > 1 - 1e-9)
    moved = np.linalg.norm(p1 - pts0, axis=1)
    assert len(p1) == len(pts0), "r-adaptivity must not change the node count"
    assert moved[on_edge].max() == 0.0, f"boundary drifted by {moved[on_edge].max():.2e}"
    assert moved[~on_edge].max() > 1e-4, "no interior motion at all"
    assert _min_detj(p1, cells) > 0.0, "mesh tangled"
    assert len(fem.adapt_history) > 0


def test_relocate_rejects_an_unknown_method():
    from jno.utils.solver.fem_adapt import AdaptSpec

    _d, fem = _peak_scalar()
    with pytest.raises(ValueError, match="relocate_method must be"):
        fem.solve(adapt=AdaptSpec(relocate=True, max_iters=2, relocate_method="mmpde"))


def test_relocate_method_is_selectable_from_the_public_spec():
    """Both methods are reachable through ``jno.solve.relocate`` — the ``AdaptSpec`` route is internal."""
    assert jno.solve.relocate().relocate_method == "descent", "descent is the default (it wins on accuracy)"
    assert jno.solve.relocate(method="monge_ampere").relocate_method == "monge_ampere"
    spec = jno.solve.relocate(method="monge_ampere", relax=25, relax_step=0.02)
    assert (spec.ma_relax, spec.ma_dt) == (25, 0.02), "the Monge-Ampere knobs must reach the spec"
    with pytest.raises(ValueError, match="expected 'descent' or 'monge_ampere'"):
        jno.solve.relocate(method="mmpde")


def test_relocate_monge_ampere_returns_its_best_mesh_not_its_last():
    """Monge-Ampère does not descend the objective — it solves a different problem, and truncating its map
    to the tagged vertices can make a round *worse*. Measured: on a periodic problem with 13 of 57 vertices
    free, running to the last round raised the defect ~10%. The outer loop therefore keeps the best mesh
    visited, and trims the history so its final entry is the mesh actually handed back."""
    d, fem = _peak_scalar()
    fem.solve(adapt=jno.solve.relocate(method="monge_ampere", max_iters=10))
    hist = fem.adapt_history
    objs = [h["objective"] for h in hist]
    pts_final = np.asarray(fem.domain.mesh.points)[:, :2]

    assert objs[-1] == min(objs), f"returned a worse mesh than one it visited: {objs}"
    assert objs[-1] <= objs[0], "relocation must never hand back a mesh worse than the one it started from"
    assert np.allclose(hist[-1]["points"], pts_final), "the last history entry must BE the returned mesh"


# ── the monitor must read VERTEX values, whatever the DOF layout ─────────────────────────────────────


def _peak_field(order, shape, size=0.14):
    """The same Poisson peak at a chosen element order and value shape; every component carries the
    identical field, which is what makes the scalar and vector runs comparable below."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    xm, ym = d.variable("mov", where=lambda x, y: (x > 0.05) & (x < 0.95) & (y > 0.05) & (y < 0.95), split=True)[:2]
    xm.trainable(name="ix")
    ym.trainable(name="iy")
    u, phi = d.fem_symbols(order=order, value_shape=shape)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = J.exp(-60.0 * ((xi - 0.5) ** 2 + (yi - 0.5) ** 2))
    if shape:
        weak = J.inner(J.grad(ui, xi), J.grad(vi, xi)) + J.inner(J.grad(ui, yi), J.grad(vi, yi)) - f * (vi[0] + vi[1])
        bcs = [u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0]
    else:
        weak = ui.x * vi.x + ui.y * vi.y - f * vi
        bcs = [u(xb, yb) - 0.0]
    return d, jno.fem([weak, *bcs])


def test_a_higher_order_vector_field_relocates_on_its_vertex_values():
    """The monitor reads VERTEX values out of each solution block, and it must find them whatever the DOF
    layout is.

    It did not. ``vec`` was guessed from the block LENGTH — ``nb // n_verts if nb % n_verts == 0 else 1``
    — and for a P2 vector field that modulus is non-zero (634 DOFs against 88 vertices), so it fell to
    ``blk[:n_verts]``: the first ``n_verts/vec`` NODES times their components, read as if it were one
    value per vertex. Measured 1.4e-02 away from the true component-0 vertex values, the scale of the
    solution itself. Nothing raised; the mesh relocated against a misread array, and this was the one
    configuration where element quality got WORSE — min |det J| 8.7e-03 -> 4.7e-03, where every other
    order/shape improved it.

    Both components here carry the identical field, so the vector run must land on the SAME mesh as the
    scalar one — the defect is then exactly twice the scalar's, and the descent direction is
    RMS-normalised. That equality is the sharp assertion: a misread block cannot satisfy it by accident.
    """
    d_s, fem_s = _peak_field(2, ())
    fem_s.solve(adapt=jno.solve.relocate(max_iters=8))
    scalar_mesh = np.asarray(d_s.mesh.points)[:, :2].copy()

    d_v, fem_v = _peak_field(2, (2,))
    fem_v.solve(adapt=jno.solve.relocate(max_iters=8))
    vector_mesh = np.asarray(d_v.mesh.points)[:, :2]

    assert vector_mesh == pytest.approx(scalar_mesh, abs=1e-12), (
        f"a P2 vector field relocated to a different mesh than the equivalent scalar: "
        f"max|dX| = {np.abs(vector_mesh - scalar_mesh).max():.3e}"
    )


@pytest.mark.parametrize("order, shape", [(1, ()), (2, ()), (3, ()), (1, (2,)), (2, (2,))])
def test_relocation_improves_element_quality_at_any_order_and_shape(order, shape):
    """Relocation must not make the mesh worse. Parametrised over the orders and value shapes the driver
    accepts, because the P2-vector case silently degraded quality while every other combination improved
    it — a shape of failure a single-configuration test cannot see."""
    d, fem = _peak_field(order, shape)
    cells = np.asarray(d.mesh.cells_dict["triangle"])
    before = _min_detj(np.asarray(d.mesh.points)[:, :2], cells)

    fem.solve(adapt=jno.solve.relocate(max_iters=8))
    after = _min_detj(np.asarray(d.mesh.points)[:, :2], cells)

    assert after > 0.0, f"P{order} shape={shape}: relocation tangled the mesh (min detJ {after:.3e})"
    assert after >= 0.9 * before, f"P{order} shape={shape}: element quality fell, {before:.3e} -> {after:.3e}"


# ------------------------------------------------------- the objective is a CHOICE, and both directions matter

L_SHAPE_R = [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)]


def _corner_problem(movable, size=0.12):
    """Poisson on an L-shape: a FIXED singularity at the re-entrant corner, nothing moving."""
    d = jno.Shape.polygon(L_SHAPE_R, size=size).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    if movable:
        eps = 1e-9
        xm, ym, _ = d.variable("mov", where=lambda x, y: (x > eps) & (x < 1 - eps) & (y > eps) & (y < 1 - eps), split=True)
        xm.trainable(name="ix")
        ym.trainable(name="iy")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])


def _dirichlet_energy_of(d, sol):
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _dirichlet_energy_jax

    pts = jnp.asarray(np.asarray(d.mesh.points)[:, :2])
    cells = jnp.asarray(np.asarray(d.mesh.cells_dict["triangle"]))
    return float(_dirichlet_energy_jax(pts, jnp.asarray(np.asarray(sol).reshape(-1)), cells, 2))


def test_energy_descent_cuts_the_error_at_fixed_dofs_on_a_fixed_singularity():
    """The measurement that lived only in a tutorial, which is how issue #109 survived three releases.

    Relocation must improve the ANSWER at a fixed node count. For a Ritz method
    ``E_h - E_exact = 1/2 ||u - u_h||_E^2``, so the Dirichlet energy against a fine reference IS the
    energy-norm error, and `objective="energy"` descends exactly it. Measured: 4.459e-03 -> 1.991e-03,
    a 55% cut at +0 DOFs, where the default monitor objective gives -12% on this problem.
    """
    d_ref, fem_ref = _corner_problem(False, size=0.03)
    e_ref = _dirichlet_energy_of(d_ref, fem_ref.solve())

    d0, fem0 = _corner_problem(False)
    err0 = _dirichlet_energy_of(d0, fem0.solve()) - e_ref

    d, fem = _corner_problem(True)
    n0 = len(d.mesh.points)
    sol = fem.solve(adapt=jno.solve.relocate(objective="energy", max_iters=60, lr=3e-3))
    err = _dirichlet_energy_of(fem.domain, sol) - e_ref

    assert len(fem.domain.mesh.points) == n0, "r-adaptivity must not change the node count"
    assert err < 0.75 * err0, f"energy descent did not cut the error at fixed DOFs: {err:.3e} vs {err0:.3e}"


def test_the_two_objectives_each_win_on_their_own_problem():
    """Neither objective dominates, which is why both exist and why the default is a judgement call.

    On a FIXED singularity the energy is the error norm and descending it wins. On an UNDER-RESOLVED
    FRONT the energy can be lowered by under-resolving the layer -- measured at 10.7x worse than uniform
    when it was the default, which is what motivated the switch. The monitor targets resolution instead.
    Pinning both directions means neither can be silently dropped again.
    """
    d_ref, fem_ref = _corner_problem(False, size=0.03)
    e_ref = _dirichlet_energy_of(d_ref, fem_ref.solve())
    out = {}
    for obj in ("energy", "equidistribution"):
        d, fem = _corner_problem(True)
        sol = fem.solve(adapt=jno.solve.relocate(objective=obj, max_iters=60, lr=3e-3))
        out[obj] = _dirichlet_energy_of(fem.domain, sol) - e_ref
    assert out["energy"] < out["equidistribution"], f"on a fixed singularity the energy objective must win: {out}"


def test_the_objective_is_reachable_from_the_public_slot_and_validated():
    """`AdaptSpec.objective` existed with two values while `jno.solve.relocate()` exposed no way to set
    it -- a documented knob unreachable from the front door, which is half of why #109 was hard to
    diagnose."""
    assert jno.solve.relocate(objective="energy").objective == "energy"
    assert jno.solve.relocate().objective == "equidistribution"  # unchanged default
    d, fem = _corner_problem(True)
    with pytest.raises(ValueError, match="must be 'energy', 'equidistribution', 'huang', or a weak-form"):
        fem.solve(adapt=jno.solve.relocate(objective="nonsense", max_iters=1))


def _stokes_with_a_pressure_gauge(size=0.5):
    """Taylor-Hood Stokes in a channel, with `p.pin()` fixing the pressure gauge, and its top wall
    vertices free to slide vertically."""
    d = jno.Shape.rect(0.0, 0.0, 3.0, 1.0, size=size).domain()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    x, y, _ = d.variable("interior", split=True)
    cin = d.variable("inlet", where=lambda X, Y: X < 1e-9, split=True)
    cbot = d.variable("bottom", where=lambda X, Y: Y < 1e-9, split=True)
    cmov = d.variable("tmov", where=lambda X, Y: (Y > 1.0 - 1e-9) & (X > 1e-9), split=True)
    cmov[1].trainable(name="ty")
    eu, ev = jno.np.symgrad(u, [x, y]), jno.np.symgrad(v, [x, y])
    dd = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    pp, qq = p.bind(x=x, y=y), q.bind(x=x, y=y)
    fem = jno.fem(
        [
            2.0 * dd(eu, ev) - pp * jno.np.trace(ev),
            -qq * jno.np.trace(eu),
            u(cin[0], cin[1])[0] - 1.0,
            u(cin[0], cin[1])[1] - 0.0,
            u(cbot[0], cbot[1])[0] - 0.0,
            u(cbot[0], cbot[1])[1] - 0.0,
            p.pin(),
        ],
        quad_degree=3,
    )
    return d, fem


def test_relocate_survives_a_pressure_gauge_pin():
    """r-adaptivity on a Taylor-Hood saddle -- which is to say, on any incompressible flow.

    `move_mesh` resets the custom-tag state, and `p.pin()`'s single-vertex region is minted by the
    front end rather than held in `_tag_predicates`, so nothing re-derived it after the move: the
    rebuilt form carried a trial-without-test term whose region no longer existed and `jno.fem`
    refused it as a whole-domain volume. Relocate was therefore unusable on every problem carrying a
    pressure (or all-Neumann) gauge -- it raised rather than returning a wrong answer, but it raised
    from the mesh rebuild with nothing pointing at the pin.
    """
    d, fem = _stokes_with_a_pressure_gauge()
    p0 = np.asarray(d.mesh.points)[:, :2].copy()
    u = np.asarray(fem.solve(adapt=jno.solve.relocate(max_iters=6, lr=3e-3)))
    assert fem.adapt_history, "no relocation rounds ran"
    assert np.isfinite(u).all() and u.size > 0, "the re-solve on the moved mesh gave no solution"
    p1 = np.asarray(fem.domain.mesh.points)[:, :2]
    assert np.abs(p1 - p0).max() > 0.0, "no vertex moved"
    assert _min_detj(p1, np.asarray(fem.domain.mesh.cells_dict["triangle"])) > 0.0, "mesh tangled"
