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
    remesh_with_mmg,
    zz_error_indicators,
)


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
    return jno.domain(jno.domain.l_shape(size=1.0, mesh_size=mesh_size))


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
