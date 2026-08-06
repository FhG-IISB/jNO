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

    Driven hard: a 13:1 monitor demands a √13 ≈ 3.6× linear compression, and nodes move ~2 cells. Note
    ``dt`` stays inside the relaxation's stability window — divergence is a *separate* failure mode from
    folding, and is covered by :func:`test_monge_ampere_divergence_is_caught_not_silently_accepted`."""
    pts, cells = _mesh_grid(15)
    monitor = 1.0 + 12.0 * np.exp(-(((pts[:, 0] - 0.5) / 0.10) ** 2))
    disp = _ma_disp(monitor, pts, cells, n_relax=400, dt=0.05)
    moved = pts + disp
    assert np.abs(disp).max() > 1.5 / 14, "not actually driven hard -- nodes barely moved"
    assert _min_detj(moved, cells) > 0.0, f"mesh folded: min det J = {_min_detj(moved, cells):.3e}"


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
