"""Adaptive-FEM (``jno.utils.solver.fem_adapt``) tests.

The load-bearing check here is not "the mesh got denser" but "the *remeshed* domain
still assembles and solves a FEM problem correctly" -- that is the real integration
risk in the adapt-then-solve architecture (a remeshed mesh has to survive
``_extract_points_from_mesh`` -> ``_build_simplex_pools`` -> ``build_native_fem_context``
and still resolve Dirichlet nodes geometrically).
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("mmgpy", reason="mmgpy required for adaptive remeshing")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from shapely.geometry import Polygon, box  # noqa: E402

import jno.jnp_ops as J  # noqa: E402
from jno.utils.solver.fem_adapt import (  # noqa: E402
    AdaptSpec,
    _solve_vertex_values,
    dorfler_mark,
    hessian_metric,
    remesh_with_mmg,
    run_adaptive_inverse,
    zz_error_indicators,
)


def _mean_aspect_ratio(d):
    pts = np.asarray(d.mesh.points)[:, :2]
    tris = np.asarray(d.mesh.cells_dict["triangle"])
    e = np.stack([np.linalg.norm(pts[tris[:, a]] - pts[tris[:, b]], axis=1) for a, b in [(0, 1), (1, 2), (2, 0)]], axis=1)
    return float(np.mean(e.max(axis=1) / e.min(axis=1)))


def _mod(a, m):
    # jno.np has no `mod`; build it from `floor` (trace-aware).
    return a - m * J.floor(a / m)


def _u_singular(x, y, xp, mod):
    """Harmonic L-shape corner mode ``u = r^(2/3) sin(2*phi/3)`` about (0.5, 0.5).

    ``phi`` runs over the 3*pi/2 material wedge (the missing quadrant is x>0.5, y>0.5).
    ``u`` is harmonic, so with ``g=u`` on the boundary the only error source is the
    reentrant-corner singularity -- the canonical adaptive-FEM benchmark. ``xp``/``mod``
    are numpy (evaluation) or jno.np (symbolic boundary data)."""
    X, Y = x - 0.5, y - 0.5
    r = xp.sqrt(X * X + Y * Y)
    th = mod(xp.arctan2(Y, X), 2.0 * np.pi)
    phi = mod(th - np.pi / 2.0, 2.0 * np.pi)
    return (r ** (2.0 / 3.0)) * xp.sin(2.0 / 3.0 * phi)


def _build_singular_laplace(d):
    """``-lap u = 0`` with the singular corner mode as Dirichlet data on the whole boundary."""
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    return [ui.x * vi.x + ui.y * vi.y, u(xb, yb) - _u_singular(xb, yb, J, _mod)]


def _true_error(fem):
    uv = _solve_vertex_values(fem)
    c = np.asarray(fem.points)[:, :2]
    ue = _u_singular(c[:, 0], c[:, 1], np, np.mod)
    return np.linalg.norm(uv - ue) / np.linalg.norm(ue)


def _loglog_slope(dofs, errs):
    return float(np.polyfit(np.log(np.asarray(dofs)), np.log(np.asarray(errs)), 1)[0])


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _l_shape_polygon(size: float = 1.0) -> Polygon:
    return Polygon([(0, 0), (size, 0), (size, size / 2), (size / 2, size / 2), (size / 2, size), (0, size)])


def _l_shape_domain(mesh_size: float = 0.15):
    return jno.Shape.polygon([(0, 0), (1.0, 0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0, 1.0)], size=mesh_size).domain()


def _corner_focused_size(points: np.ndarray, corner=(0.5, 0.5), fine=0.03, coarse=0.3) -> np.ndarray:
    r = np.linalg.norm(points[:, :2] - np.asarray(corner), axis=1)
    # smooth radial ramp fine->coarse so the request is gentle (respects hgrad)
    return np.clip(fine + (coarse - fine) * (r / 0.5), fine, coarse)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def test_remesh_refines_at_corner_and_preserves_geometry():
    d = _l_shape_domain(mesh_size=0.2)
    pts0 = np.asarray(d.mesh.points)[:, :2]
    n0 = len(pts0)

    size = _corner_focused_size(pts0)
    d2 = remesh_with_mmg(d, size)
    pts1 = np.asarray(d2.mesh.points)[:, :2]

    # (1) refinement happened and concentrated at the reentrant corner
    assert len(pts1) > n0
    r0 = np.linalg.norm(pts0 - (0.5, 0.5), axis=1)
    r1 = np.linalg.norm(pts1 - (0.5, 0.5), axis=1)
    frac0 = np.mean(r0 < 0.15)
    frac1 = np.mean(r1 < 0.15)
    assert frac1 > frac0

    # (2) the reentrant corner vertex survived exactly (no smoothing/drift)
    assert r1.min() < 1e-9

    # (3) no boundary node drifted off the L-shape geometry
    poly = _l_shape_polygon()
    signed = np.array(
        [poly.exterior.distance(_p_to_shapely(p)) if not poly.covers(_p_to_shapely(p)) else 0.0 for p in pts1]
    )
    assert signed.max() < 1e-6


def _p_to_shapely(p):
    from shapely.geometry import Point

    return Point(float(p[0]), float(p[1]))


def test_solve_on_remeshed_domain_recovers_linear_solution():
    """The gate: assemble+solve Laplace on the *remeshed* domain and recover u=x.

    ``u = x`` is harmonic, so with Dirichlet ``g = x`` on the whole boundary the
    exact FEM solution is ``u = x`` to solver tolerance -- a strong end-to-end check
    that the remeshed mesh assembles and that boundary nodes are still tagged.
    """
    d = _l_shape_domain(mesh_size=0.18)
    size = _corner_focused_size(np.asarray(d.mesh.points)[:, :2])
    d2 = remesh_with_mmg(d, size)

    u, phi = d2.fem_symbols()
    xi, yi, _ = d2.variable("interior", split=True)
    xb, yb, _ = d2.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    # -lap u = 0 with u = x on the whole boundary  =>  u = x
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - xb])

    A = _dense(fem.A)
    b = np.asarray(fem.b).reshape(-1)
    sol = np.linalg.solve(A, b)

    coords = np.asarray(fem.points)[:, :2]
    err = np.linalg.norm(sol - coords[:, 0]) / np.linalg.norm(coords[:, 0])
    assert err < 1e-8, f"u=x not recovered on remeshed mesh (rel err {err:.2e})"


def test_zz_indicator_concentrates_at_reentrant_corner():
    d = _l_shape_domain(mesh_size=0.12)
    fem = jno.fem(_build_singular_laplace(d))
    u_vertex = _solve_vertex_values(fem)
    eta, est = zz_error_indicators(d, u_vertex)

    assert est > 0.0
    assert eta.shape[0] == len(d.mesh.cells_dict["triangle"])

    cent = np.asarray(d.mesh.points)[:, :2][np.asarray(d.mesh.cells_dict["triangle"])].mean(axis=1)
    rc = np.linalg.norm(cent - (0.5, 0.5), axis=1)
    # the singularity is at the corner, so indicators there must dominate the far field
    assert eta[rc < 0.2].mean() > 5.0 * eta[rc > 0.4].mean()

    # Dörfler marks a strict, non-empty subset for a concentrated error
    marked = dorfler_mark(eta, theta=0.5)
    assert 0 < marked.size < eta.shape[0]


def test_domain_refine_in_place():
    d = _l_shape_domain(mesh_size=0.2)
    n0 = len(d.mesh.points)
    size = _corner_focused_size(np.asarray(d.mesh.points)[:, :2])
    ret = d.refine(size)
    assert ret is d  # in place, chainable
    assert len(d.mesh.points) > n0


def test_fem_solve_adapt_drives_loop_and_rebinds():
    d = _l_shape_domain(mesh_size=0.2)
    fem = jno.fem(_build_singular_laplace(d))
    n0 = len(d.mesh.points)
    sol = np.asarray(fem.solve(adapt=AdaptSpec(theta=0.6, max_iters=3, refine_factor=1.6))).reshape(-1)

    # the domain was refined in place and the FEM rebound to the final mesh
    assert len(d.mesh.points) > n0
    assert sol.shape[0] == len(fem.points)
    # history has one record per round, DOFs monotonically non-decreasing
    hist = fem.adapt_history
    assert 1 <= len(hist) <= 3
    dofs = [h["n_dofs"] for h in hist]
    assert dofs == sorted(dofs)


def _poisson_on(d, alpha):
    """Parametric Poisson ``-alpha*lap u = f`` (exact ``u = x(1-x)y(1-y)`` at alpha=1)."""
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    return jno.fem([alpha * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)


def test_differentiable_inverse_on_adapted_mesh():
    """Differentiability requirement: adapt the mesh (forward) -> freeze -> recover a
    parameter on the frozen adapted mesh via crux/implicit-diff.

    The refinement itself is non-differentiable (discrete), but once the mesh is frozen
    the ordinary parametric solve is fully differentiable, so the gradient reaches the
    ``jno.np.parameter`` unchanged."""
    # forward: adapt at a fixed coefficient to produce a refined mesh (in place)
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    n0 = len(d.mesh.points)
    _poisson_on(d, 1.0).solve(adapt=AdaptSpec(theta=0.6, max_iters=3, refine_factor=1.6))
    assert len(d.mesh.points) > n0

    # inverse on the frozen adapted mesh
    alpha = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="alpha")
    alpha.initialize(jax.nn.initializers.constant(2.0))  # start far from truth = 1
    alpha.dtype(jnp.float64)
    alpha.optimizer(optax.adam(5e-2))
    fem_inv = _poisson_on(d, alpha)

    a1, b1 = fem_inv.operator.evaluate({"alpha": 1.0})
    u_obs = jnp.linalg.solve(a1.todense(), jnp.asarray(b1).reshape(-1))

    crux = jno.core([(fem_inv.solve() - u_obs).mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    crux.solve(120)
    rec = float(np.asarray(crux.eval([alpha])[0]).reshape(-1)[0])
    assert abs(rec - 2.0) > 0.5, "parameter did not move -- gradient did not reach it through the adapted mesh"
    assert abs(rec - 1.0) < 0.05, f"recovered alpha={rec:.4f} on adapted mesh"


_KAPPA_TRUE = 5.0


def _reaction_diffusion_fem(d, kappa):
    """``-lap u + kappa*u = kappa_true*u_singular`` on the L-shape, Dirichlet ``g = u_singular``.

    Since ``u_singular`` is harmonic, at ``kappa = kappa_true`` the exact solution *is*
    ``u_singular`` -- a value-singular corner state.  ``kappa`` multiplies the mass
    (reaction) bilinear term, so it enters through the operator and the inverse solve is
    differentiable via implicit diff (unlike a parameter in the Dirichlet data)."""
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = _KAPPA_TRUE * _u_singular(xi, yi, J, _mod)
    return jno.fem(
        [ui.x * vi.x + ui.y * vi.y + kappa * (ui * vi) - f * vi, u(xb, yb) - _u_singular(xb, yb, J, _mod)],
        quad_degree=4,
    )


def _corner_obs(d):
    """Closed-form observations ``u_singular`` at nodes, weighted to the corner (r < 0.2)."""
    nodes = np.asarray(d.mesh.points)[:, :2]
    s = jnp.asarray(_u_singular(nodes[:, 0], nodes[:, 1], np, np.mod))
    r = np.linalg.norm(nodes - (0.5, 0.5), axis=1)
    return s, jnp.asarray((r < 0.2).astype(np.float64))


def _fresh_kappa(seed, init=2.0):
    k = jno.np.parameter((1,), key=jax.random.PRNGKey(seed), name=f"kappa{seed}")
    k.initialize(jax.nn.initializers.constant(init))
    k.dtype(jnp.float64)
    k.optimizer(optax.adam(1e-1))
    return k


@pytest.mark.slow
def test_adaptive_inverse_beats_uniform_on_l_shape():
    """Adaptive mesh refinement wrapped around the inverse solve recovers the parameter
    more accurately per DOF than uniform refinement -- the *minimal mesh for the recovered
    design*.

    Each round differentiably recovers ``kappa`` on the current (frozen) mesh, then the ZZ
    estimator refines the corner where the singular state is under-resolved; the recovered
    ``kappa`` de-biases toward the truth as the mesh adapts."""
    dummy = jno.domain.from_array({"_": np.zeros((1, 1))})
    kappa = _fresh_kappa(0)
    best: dict = {}

    def build_inverse(dd):
        if "k" in best:  # warm-start from the previous (coarser) round
            kappa.initialize(jax.nn.initializers.constant(best["k"]))
        s, w = _corner_obs(dd)
        fem = _reaction_diffusion_fem(dd, kappa)
        return jno.core([(w * (fem.solve() - s)).mse], domain=dummy), fem.solve()

    def readout(crux):
        v = float(np.asarray(crux.eval([kappa])).reshape(-1)[0])
        best["k"] = v
        return v

    d = _l_shape_domain(mesh_size=0.2)
    hist = run_adaptive_inverse(
        d, build_inverse, AdaptSpec(theta=0.6, max_iters=5, refine_factor=1.6), n_opt=200, readout=readout
    )
    k_adapt = hist[-1]["params"]
    dof_adapt = hist[-1]["n_dofs"]

    # the inverse moved kappa off its init toward the truth, and the mesh grew monotonically
    assert abs(k_adapt - _KAPPA_TRUE) < abs(2.0 - _KAPPA_TRUE)
    assert [h["n_dofs"] for h in hist] == sorted(h["n_dofs"] for h in hist)

    # uniform baseline on a mesh with at LEAST as many DOFs as the adaptive result
    du = _l_shape_domain(mesh_size=0.09)
    assert len(du.mesh.points) >= dof_adapt, "uniform baseline must match or exceed the adaptive DOFs"
    s, w = _corner_obs(du)
    ku = _fresh_kappa(1)
    fem_u = _reaction_diffusion_fem(du, ku)
    cru = jno.core([(w * (fem_u.solve() - s)).mse], domain=dummy)
    cru.solve(200)
    k_unif = float(np.asarray(cru.eval([ku])).reshape(-1)[0])

    # minimal mesh: adaptive recovers kappa more accurately with FEWER DOFs than uniform
    assert abs(k_adapt - _KAPPA_TRUE) < abs(k_unif - _KAPPA_TRUE), (
        f"adaptive |k-5|={abs(k_adapt - _KAPPA_TRUE):.3f} @ {dof_adapt} dofs did not beat "
        f"uniform |k-5|={abs(k_unif - _KAPPA_TRUE):.3f} @ {len(du.mesh.points)} dofs"
    )


def test_adapt_inverse_eps_requires_readout():
    """eps needs a readout to measure parameter convergence -- guard it explicitly."""
    d = _l_shape_domain(mesh_size=0.3)
    with pytest.raises(ValueError, match="readout"):
        run_adaptive_inverse(d, lambda dd: (None, None), AdaptSpec(max_iters=3, eps=0.01), n_opt=1)


def test_adapt_forward_eps_stops_on_plateau():
    """AdaptSpec.eps stops the forward loop once the error estimate stops improving."""
    d = _l_shape_domain(mesh_size=0.2)
    fem = jno.fem(_build_singular_laplace(d))
    # loose eps: consecutive estimate changes are well under 30%, so the plateau guard
    # (patience=2) trips within a few rounds instead of running all 15
    fem.solve(adapt=AdaptSpec(theta=0.6, max_iters=15, refine_factor=1.5, eps=0.3))
    hist = fem.adapt_history
    assert 3 <= len(hist) < 15, f"eps did not stop the forward loop early (ran {len(hist)} rounds)"


@pytest.mark.slow
def test_adapt_inverse_eps_stops_on_convergence():
    """AdaptSpec.eps stops the inverse loop once the recovered parameter stops moving,
    and the patience=2 guard means the stop reflects two consecutive converged rounds."""
    dummy = jno.domain.from_array({"_": np.zeros((1, 1))})
    kappa = _fresh_kappa(0)
    best: dict = {}

    def build_inverse(dd):
        if "k" in best:
            kappa.initialize(jax.nn.initializers.constant(best["k"]))
        s, w = _corner_obs(dd)
        fem = _reaction_diffusion_fem(dd, kappa)
        return jno.core([(w * (fem.solve() - s)).mse], domain=dummy), fem.solve()

    def readout(crux):
        best["k"] = float(np.asarray(crux.eval([kappa])).reshape(-1)[0])
        return best["k"]

    d = _l_shape_domain(mesh_size=0.2)
    hist = run_adaptive_inverse(
        d, build_inverse, AdaptSpec(theta=0.6, max_iters=12, refine_factor=1.5, eps=0.05), n_opt=150, readout=readout
    )
    ks = [h["params"] for h in hist]

    # stopped early on convergence, not the iteration cap
    assert len(ks) < 12, f"eps did not stop the inverse loop early (ran {len(ks)} rounds)"
    # the stop is genuine: the last TWO round-to-round changes are both under eps (patience=2)
    assert abs(ks[-1] - ks[-2]) / abs(ks[-1]) < 0.05
    assert abs(ks[-2] - ks[-3]) / abs(ks[-2]) < 0.05


def test_hessian_metric_yields_anisotropic_mesh():
    """hessian_metric on a thin-layer field produces a stretched (anisotropic) mesh."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
    nodes = np.asarray(d.mesh.points)[:, :2]
    field = np.tanh((nodes[:, 0] - 0.5) / 0.04)  # thin vertical layer at x=0.5

    metric = hessian_metric(d, field, target_complexity=3.0 * len(nodes), hmin=0.004, hmax=0.2)
    assert metric.shape == (len(nodes), 3)  # (m11, m12, m22) tensor per vertex

    d.refine(metric, hgrad=3.0)
    # elements are stretched along the layer, not near-equilateral like an isotropic mesh
    assert _mean_aspect_ratio(d) > 3.0


def _oblique_u(x, y, xp):
    return xp.tanh((x + y - 1.0) / 0.03)  # thin layer along the x+y=1 diagonal


def _oblique_layer_fem(d):
    """-lap u = f with u = tanh((x+y-1)/eps): an OBLIQUE layer isotropic refinement handles
    poorly (it must refine a wide diagonal band) but anisotropic resolves with stretched
    elements aligned to the diagonal."""
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    t = J.tanh((xi + yi - 1.0) / 0.03)
    f = (4.0 / 0.03**2) * (1.0 - t * t) * t
    return jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - _oblique_u(xb, yb, J)])


@pytest.mark.slow
def test_anisotropic_adapt_beats_isotropic_on_oblique_layer():
    """On an oblique layer, anisotropic (Hessian-metric) refinement reaches a far lower
    error estimate per DOF than isotropic ZZ + Dörfler."""
    # both run to a comparable DOF budget; anisotropic reaches a far lower estimate
    di = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    _oblique_layer_fem(di).solve(adapt=AdaptSpec(theta=0.7, max_iters=9, refine_factor=1.7, max_dofs=3000))
    # fem is rebound to the final mesh; rebuild on the adapted domain for a fresh estimate
    _, iso_est = zz_error_indicators(di, _solve_vertex_values(_oblique_layer_fem(di)))
    iso_dofs = len(di.mesh.points)

    da = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    _oblique_layer_fem(da).solve(adapt=AdaptSpec(anisotropic=True, max_iters=8, refine_factor=1.6, max_dofs=3000))
    _, aniso_est = zz_error_indicators(da, _solve_vertex_values(_oblique_layer_fem(da)))
    aniso_dofs = len(da.mesh.points)

    assert _mean_aspect_ratio(da) > 3.0, "anisotropic mesh is not stretched"
    # comparable DOFs, far lower estimate (empirically ~5-10x; require >=3x with margin)
    assert aniso_dofs <= 1.5 * iso_dofs, f"anisotropic used {aniso_dofs} dofs vs isotropic {iso_dofs}"
    assert aniso_est < iso_est / 3.0, f"anisotropic est {aniso_est:.3f} not < iso est {iso_est:.3f}/3"


def _cube(mesh_size=0.35):
    return jno.Shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain()


def test_remesh_and_solve_3d_recovers_linear_solution():
    """3D gate: remesh a tet cube, preserve its geometry, and recover u=x on the new mesh."""
    d = _cube(0.35)
    n0 = len(d.mesh.points)
    d2 = remesh_with_mmg(d, np.full(n0, 0.18))
    pts = np.asarray(d2.mesh.points)[:, :3]

    assert len(pts) > n0  # refined
    assert pts.min() > -1e-9 and pts.max() < 1.0 + 1e-9  # geometry preserved (stays in the cube)

    u, phi = d2.fem_symbols()
    xi, yi, zi, _ = d2.variable("interior", split=True)
    xb, yb, zb, _ = d2.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    # -lap u = 0 with u = x on the whole boundary => u = x
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, u(xb, yb, zb) - xb])
    sol = _solve_vertex_values(fem)
    coords = np.asarray(fem.points)[:, :3]
    err = np.linalg.norm(sol - coords[:, 0]) / np.linalg.norm(coords[:, 0])
    assert err < 1e-7, f"u=x not recovered on remeshed 3D mesh (rel err {err:.2e})"


def _layer_3d_fem(d, eps=0.08):
    """-lap u = f with u = tanh((x-0.5)/eps): a thin planar layer in the unit cube."""
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    t = J.tanh((xi - 0.5) / eps)
    f = (2.0 / eps**2) * (1.0 - t * t) * t
    return jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - f * vi, u(xb, yb, zb) - J.tanh((xb - 0.5) / eps)])


@pytest.mark.slow
def test_adaptive_loop_3d_refines_at_layer():
    """The 3D adaptive loop (solve -> estimate -> mark -> refine) drives the error estimate
    down, grows the mesh, and concentrates DOFs at the planar layer."""
    d = _cube(0.3)
    fem = _layer_3d_fem(d)
    fem.solve(adapt=AdaptSpec(theta=0.6, max_iters=4, refine_factor=1.6, max_dofs=2500))
    hist = fem.adapt_history

    dofs = [h["n_dofs"] for h in hist]
    ests = [h["estimate"] for h in hist]
    assert dofs == sorted(dofs) and dofs[-1] > dofs[0]  # mesh grew monotonically
    assert ests[-1] < ests[0]  # estimate fell as the layer was resolved
    # refinement concentrated at the layer plane x=0.5
    pts = np.asarray(d.mesh.points)[:, :3]
    assert np.mean(np.abs(pts[:, 0] - 0.5) < 0.1) > 0.3


def _mean_tet_aspect(d):
    pts = np.asarray(d.mesh.points)[:, :3]
    tets = np.asarray(d.mesh.cells_dict["tetra"])
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    e = np.stack([np.linalg.norm(pts[tets[:, a]] - pts[tets[:, b]], axis=1) for a, b in pairs], axis=1)
    return float(np.mean(e.max(axis=1) / e.min(axis=1)))


@pytest.mark.slow
def test_anisotropic_adapt_3d_stretches_and_beats_isotropic():
    """3D anisotropic (Hessian-metric) refinement stretches tetrahedra along a planar layer
    and reaches a far lower error estimate than isotropic at comparable DOFs."""
    di = _cube(0.28)
    _layer_3d_fem(di, eps=0.05).solve(adapt=AdaptSpec(theta=0.6, max_iters=5, refine_factor=1.6, max_dofs=6000))
    _, iso_est = zz_error_indicators(di, _solve_vertex_values(_layer_3d_fem(di, eps=0.05)))
    iso_dofs = len(di.mesh.points)

    da = _cube(0.28)
    _layer_3d_fem(da, eps=0.05).solve(adapt=AdaptSpec(anisotropic=True, max_iters=5, refine_factor=1.8, max_dofs=6000))
    _, aniso_est = zz_error_indicators(da, _solve_vertex_values(_layer_3d_fem(da, eps=0.05)))
    aniso_dofs = len(da.mesh.points)

    assert _mean_tet_aspect(da) > 3.0, "3D anisotropic mesh is not stretched"
    # metric-based DOF control is approximate, so allow up to ~2.5x isotropic's DOFs; the win is
    # the far lower estimate (stretched tets resolving the layer that isotropic cannot cheaply)
    assert aniso_dofs <= 2.5 * iso_dofs, f"anisotropic used {aniso_dofs} dofs vs isotropic {iso_dofs}"
    assert aniso_est < iso_est / 3.0, f"3D anisotropic est {aniso_est:.3f} not < iso est {iso_est:.3f}/3"


@pytest.mark.slow
def test_adaptive_beats_uniform_on_l_shape():
    """Derisk gate: adaptive refinement reaches lower error per DOF than uniform.

    Compared apples-to-apples on the ZZ global estimate (slope), plus a matched-DOF
    check on the *true* error against the exact singular solution.
    """
    # uniform refinement sequence
    u_dofs, u_true, u_est = [], [], []
    for ms in [0.2, 0.14, 0.1, 0.07, 0.05, 0.035]:
        d = _l_shape_domain(mesh_size=ms)
        fem = jno.fem(_build_singular_laplace(d))
        uv = _solve_vertex_values(fem)
        _, est = zz_error_indicators(d, uv)
        u_dofs.append(len(uv))
        u_true.append(_true_error(fem))
        u_est.append(est)

    # one adaptive run from a coarse mesh, driven through the public FEM.solve(adapt=) API
    d0 = _l_shape_domain(mesh_size=0.2)
    fem_a = jno.fem(_build_singular_laplace(d0))
    sol = np.asarray(fem_a.solve(adapt=AdaptSpec(theta=0.7, max_iters=9, refine_factor=1.6))).reshape(-1)
    a_dofs = [h["n_dofs"] for h in fem_a.adapt_history]
    a_est = [h["estimate"] for h in fem_a.adapt_history]
    a_final_dof = a_dofs[-1]
    # fem_a now refers to the final adapted mesh; ``sol`` is the solution on it
    c = np.asarray(fem_a.points)[:, :2]
    ue = _u_singular(c[:, 0], c[:, 1], np, np.mod)
    a_final_true = np.linalg.norm(sol - ue) / np.linalg.norm(ue)

    # (1) the estimate converges faster per DOF under adaptivity
    assert _loglog_slope(a_dofs, a_est) < _loglog_slope(u_dofs, u_est) - 0.05

    # (2) matched-DOF true error: adaptive beats the *finest* uniform mesh with >= its DOFs
    finer = [(n, e) for n, e in zip(u_dofs, u_true) if n >= a_final_dof]
    assert finer, "uniform sequence should include a mesh at least as fine as the adaptive result"
    uniform_true_at_matched = min(e for _, e in finer)
    assert a_final_true < uniform_true_at_matched
