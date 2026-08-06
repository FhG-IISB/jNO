"""Adaptivity on a **1D line domain** — native refine, solution transfer, and both AFEM loops.

mmg has no 1-D mode and needs none: an interval mesh is a sorted vertex list, so honouring a size
field is subdivision rather than remeshing (exact where mmg is approximate, and no optional
dependency). ``remesh_with_mmg`` routes dimension 1 to that native path behind the same signature, so
the steady refine loop and the transient re-mesher are dimension-agnostic above it.

The point-location core (:func:`_locate_in_cells`) was already dimension-general — an interval IS a
1-simplex and its "barycentric weights" are the two linear hat values — so ``transfer_solution`` in 1D
is the same code, not a parallel implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("scipy", reason="point location needs scipy.spatial")

import jax  # noqa: E402

from jno.utils.solver.fem_adapt import remesh_with_mmg, transfer_solution  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _line(ms=0.1, **kw):
    return jno.domain(constructor=jno.domain.line(mesh_size=ms), **kw)


def _x(d):
    return np.sort(np.asarray(d.mesh.points)[:, 0])


def _peaked_size(x, floor=0.01, ceil=0.1, at=0.5, width=0.08):
    """A size field asking for `floor` near `at` and `ceil` far from it."""
    return ceil - (ceil - floor) * np.exp(-(((x - at) / width) ** 2))


# ==========================================================================
# native 1D remesh
# ==========================================================================
def test_remesh_1d_honours_the_size_field_and_grades():
    """Subdivision to meet a per-vertex target size, with mmg's gradation rule (the ratio between
    ADJACENT edge sizes is capped at ``hgrad``) imposed by two monotone sweeps. Without gradation a
    sharply peaked estimator produces a 100x jump between neighbouring elements."""
    d = _line(0.1)
    x0 = np.asarray(d.mesh.points)[:, 0]
    hgrad = 1.4
    d2 = remesh_with_mmg(d, _peaked_size(x0), hgrad=hgrad, copy=True)

    x = _x(d2)
    dx = np.diff(x)
    assert len(x) > len(x0), "the mesh must refine"
    assert np.all(dx > 1e-12), "vertices must stay sorted and unique"
    near = np.abs(x[:-1] - 0.5) < 0.03
    assert dx[near].mean() < 0.4 * dx[[0, -1]].mean(), "the mesh is not graded toward the requested peak"
    ratios = np.maximum(dx[1:] / dx[:-1], dx[:-1] / dx[1:])
    assert np.max(ratios) < hgrad * 1.2, f"gradation cap violated: max adjacent ratio {np.max(ratios):.2f}"


def test_remesh_1d_preserves_the_geometry():
    """The endpoints are never moved, so the domain's extent survives any number of remeshes."""
    d = _line(0.1)
    lo, hi = float(np.min(_x(d))), float(np.max(_x(d)))
    for _ in range(3):
        d = remesh_with_mmg(d, _peaked_size(np.asarray(d.mesh.points)[:, 0]), copy=False)
        assert abs(float(np.min(_x(d))) - lo) < 1e-14
        assert abs(float(np.max(_x(d))) - hi) < 1e-14


def test_remesh_1d_keeps_the_boundary_regions_usable():
    """``left`` / ``right`` / ``boundary`` must survive the remesh — a Dirichlet *and* a Neumann
    condition bound to them have to resolve on the new mesh, or the remeshed problem silently loses a
    boundary condition. ``-u''=0`` with ``u(0)=0`` and ``u'(1)=1`` recovers ``u = x``."""
    d = _line(0.1)
    d2 = remesh_with_mmg(d, _peaked_size(np.asarray(d.mesh.points)[:, 0]), copy=True)
    u, phi = d2.fem_symbols()
    xi = d2.variable("interior", split=True)[0]
    xl = d2.variable("left", split=True)[0]
    xr = d2.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x, -1.0 * phi.bind(x=xr), u(xl) - 0.0])
    assert "surface@right" in fem.classification and "dirichlet@left" in fem.classification
    sol = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.points).reshape(-1)
    assert np.max(np.abs(sol - pts)) < 1e-10


def test_remesh_1d_rejects_a_bad_size_field():
    """Extremes: a non-positive or wrongly-sized metric field must raise, not produce a garbled mesh."""
    d = _line(0.2)
    n = int(np.asarray(d.mesh.points).shape[0])
    with pytest.raises(ValueError, match="strictly positive"):
        remesh_with_mmg(d, np.zeros(n), copy=True)
    with pytest.raises(ValueError, match="entries but the mesh has"):
        remesh_with_mmg(d, np.ones(n + 3), copy=True)


# ==========================================================================
# 1D solution transfer
# ==========================================================================
def test_transfer_1d_is_exact_on_a_linear_field():
    """P1 interpolation reproduces a linear field exactly — the invariant that pins the stencil and
    the weights (a wrong containing interval, or weights that do not sum to one, breaks it)."""
    src, tgt = _line(0.1), _line(0.037)
    xs, xt = np.asarray(src.mesh.points)[:, 0], np.asarray(tgt.mesh.points)[:, 0]
    got = np.asarray(transfer_solution(src, 3.0 * xs - 1.0, tgt))
    assert np.max(np.abs(got - (3.0 * xt - 1.0))) < 1e-12


def test_transfer_1d_is_second_order_on_a_smooth_field():
    src_errs = []
    for ms in (0.1, 0.05, 0.025):
        src, tgt = _line(ms), _line(0.0037)
        xs, xt = np.asarray(src.mesh.points)[:, 0], np.asarray(tgt.mesh.points)[:, 0]
        got = np.asarray(transfer_solution(src, np.sin(np.pi * xs), tgt))
        src_errs.append(float(np.max(np.abs(got - np.sin(np.pi * xt)))))
    rates = [np.log2(src_errs[i] / src_errs[i + 1]) for i in range(len(src_errs) - 1)]
    assert all(abs(r - 2.0) < 0.3 for r in rates), f"interpolation is not O(h^2): {rates}"


def test_transfer_1d_preserves_shape_and_dtype():
    """Trailing axes (a vector field) and a complex dtype ride through unchanged — the transfer is a
    weighted gather, so it must not flatten or realify its payload."""
    src, tgt = _line(0.1), _line(0.05)
    xs, xt = np.asarray(src.mesh.points)[:, 0], np.asarray(tgt.mesh.points)[:, 0]

    vec = np.stack([xs, 2.0 * xs], axis=1)
    got = np.asarray(transfer_solution(src, vec, tgt))
    assert got.shape == (len(xt), 2)
    assert np.max(np.abs(got[:, 1] - 2.0 * xt)) < 1e-12

    cx = np.asarray(transfer_solution(src, (1 + 2j) * xs, tgt))
    assert np.iscomplexobj(cx)
    assert np.max(np.abs(cx - (1 + 2j) * xt)) < 1e-12


def test_transfer_1d_is_differentiable_in_the_source_values():
    """The interpolation apply stays differentiable (point location is host, the gather is not), which
    is what lets a state be carried across a remesh inside a differentiable march."""
    import jax.numpy as jnp

    src, tgt = _line(0.1), _line(0.05)
    xs = np.asarray(src.mesh.points)[:, 0]
    g = jax.grad(lambda v: jnp.sum(transfer_solution(src, v, tgt) ** 2))(jnp.asarray(3.0 * xs))
    g = np.asarray(g)
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


def test_transfer_1d_across_a_remesh():
    """The actual use: carry a field from a mesh onto its own refinement."""
    src = _line(0.1)
    xs = np.asarray(src.mesh.points)[:, 0]
    tgt = remesh_with_mmg(src, _peaked_size(xs), copy=True)
    xt = np.asarray(tgt.mesh.points)[:, 0]
    got = np.asarray(transfer_solution(src, 3.0 * xs - 1.0, tgt))
    assert np.max(np.abs(got - (3.0 * xt - 1.0))) < 1e-12


# ==========================================================================
# the AFEM loops
# ==========================================================================
_EPS = 0.02


def _layer_exact(x):
    return (np.exp((x - 1.0) / _EPS) - np.exp(-1.0 / _EPS)) / (1.0 - np.exp(-1.0 / _EPS))


def _layer_problem(ms):
    """``-eps u'' + u' = 0``, ``u(0)=0``, ``u(1)=1`` — flat until it turns up in a layer of width ~eps
    at ``x=1``. Uniform refinement spends its dofs on the flat region; adaptive should not."""
    d = _line(ms)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    return d, jno.fem([_EPS * ui.x * vi.x + ui.x * vi, u(xl) - 0.0, u(xr) - 1.0])


def test_steady_afem_1d_beats_uniform_refinement():
    """The point of h-adaptivity: at a comparable dof count the adapted mesh must be far more accurate
    than the uniform one, and the extra dofs must actually sit in the layer."""
    _d, fem_u = _layer_problem(0.0125)  # 81 dofs uniform
    sol_u = np.asarray(fem_u.solve()).reshape(-1)
    pts_u = np.asarray(fem_u.points).reshape(-1)
    err_u = float(np.max(np.abs(sol_u - _layer_exact(pts_u))))

    _d2, fem_a = _layer_problem(0.1)  # 11 dofs, adapted from there
    sol_a = np.asarray(fem_a.solve(adapt=jno.solve.remesh(max_iters=6, theta=0.6))).reshape(-1)
    pts_a = np.asarray(fem_a.points).reshape(-1)
    err_a = float(np.max(np.abs(sol_a - _layer_exact(pts_a))))

    assert len(pts_a) < 3 * len(pts_u), "the adaptive run must stay in the same dof ballpark"
    assert err_a < err_u / 5.0, f"adaptive {err_a:.2e} did not beat uniform {err_u:.2e}"

    x = np.sort(pts_a)
    dx = np.diff(x)
    h_layer = dx[x[:-1] > 0.9].mean()
    h_far = dx[x[:-1] < 0.5].mean()
    assert h_layer < h_far / 2.0, f"the mesh did not concentrate in the layer ({h_layer:.4f} vs {h_far:.4f})"


def test_transient_afem_1d_tracks_a_moving_pulse():
    """Transient adaptive remeshing in 1D: a pulse advected at speed ``c`` must still arrive at
    ``x0 + cT`` after the mesh has been rebuilt and the state carried across it several times. The
    trajectory keeps one mesh per frame and ``resample`` projects them onto a reference mesh."""
    c, nu, T = 1.0, 2e-3, 0.3
    d = _line(0.02, time=(0.0, T, 31))
    u, phi = d.fem_symbols()
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[-1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    u0 = jno.np.exp(-(((ci - 0.2) / 0.05) ** 2))
    fem = jno.fem([ui.t * vi + c * ui.x * vi + nu * ui.x * vi.x, u(xb) - 0.0, u(ci) - u0])

    traj = fem.solve(adapt=jno.solve.remesh(every=5))
    assert len(traj) == 31
    sizes = [int(np.asarray(m[0]).shape[0]) for m in traj.meshes]
    assert sizes[-1] > sizes[0], "the mesh never adapted"

    state, (pts, _cells) = traj.final()
    x = np.asarray(pts).reshape(-1)
    centre = float(x[int(np.argmax(np.asarray(state).reshape(-1)))])
    assert abs(centre - (0.2 + c * T)) < 0.03, f"pulse centre {centre:.3f}, expected {0.2 + c * T:.3f}"

    # every frame projects onto one reference mesh for post-processing
    ref = _line(0.005)
    R = np.asarray(traj.resample(ref))
    assert R.shape == (31, int(np.asarray(ref.mesh.points).shape[0]))
    xr = np.asarray(ref.mesh.points)[:, 0]
    assert abs(float(xr[int(np.argmax(np.abs(R[-1])))]) - (0.2 + c * T)) < 0.03
