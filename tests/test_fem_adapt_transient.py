"""Transient adaptive remeshing — ``FEM.solve(adapt=...)`` on a ``u.t`` problem (run_adaptive_transient).

The mesh is remeshed every ``spec.every`` steps and the state carried across each remesh
(``transfer_solution``), so the mesh tracks the evolving feature. Oracle: the decaying heat eigenmode

    u(x, y, t) = sin(πx) sin(πy) · exp(-2κπ² t)

is analytic, so the adaptive trajectory (resampled onto a fixed reference mesh) must match it — which
proves the **time march**, the **remesh**, and the **state transfer** across meshes are all correct.
Plus: the mesh actually adapts, ``resample`` shape, and the fail-loud scope guards.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.fem_adapt import AdaptiveTrajectory

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)  # FEM assembly/solves are float64
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat_fem(mesh_size=0.1, kappa=0.1, t_end=0.3, nt=21, order=1):
    """Scalar heat equation u_t = κΔu on the unit square, mode-(1,1) IC, homogeneous Dirichlet."""
    d = jno.Shape.rect(0, 0, 1, 1, size=mesh_size).domain(time=(0.0, t_end, nt))
    u, phi = d.fem_symbols(order=order)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    weak = ui.t * vi + kappa * (ui.x * vi.x + ui.y * vi.y)
    u_ic = u(xi0, yi0) - jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [xi0, yi0])
    return jno.fem([weak, u(xb, yb) - 0.0, u_ic]), d


def test_transient_adaptive_matches_analytic_heat_decay():
    """The headline oracle: the adaptive trajectory reproduces the analytic decaying eigenmode."""
    kappa = 0.1
    fem, _ = _heat_fem(mesh_size=0.1, kappa=kappa, t_end=0.3, nt=21)
    traj = fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=5, max_dofs=6000))

    assert isinstance(traj, AdaptiveTrajectory)
    assert len(traj) >= 5 and len(traj.states) == len(traj) == len(traj.meshes)
    times = np.asarray(traj.times)
    assert abs(times[0]) < 1e-12 and abs(times[-1] - 0.3) < 1e-9  # spans [0, t_end], no drift
    assert fem.adapt_history, "expected at least one remesh during the march"

    ref = jno.Shape.rect(0, 0, 1, 1, size=0.08).domain()
    ys = np.asarray(traj.resample(ref))  # (n_save, n_ref) uniform array
    xr = np.asarray(ref.mesh.points)[:, :2]
    base = np.sin(PI * xr[:, 0]) * np.sin(PI * xr[:, 1])
    worst = 0.0
    for k, t in enumerate(times):
        exact = np.exp(-2 * kappa * PI**2 * t) * base
        worst = max(worst, np.linalg.norm(ys[k] - exact) / max(np.linalg.norm(exact), 1e-12))
    assert worst < 0.08, f"analytic heat-decay mismatch across the moving mesh: worst rel L2 = {worst:.3f}"


def test_transient_adaptive_mesh_actually_changes():
    fem, _ = _heat_fem(mesh_size=0.12, t_end=0.2, nt=21)
    traj = fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=4, max_dofs=5000))
    sizes = [m[0].shape[0] for m in traj.meshes]
    assert len(set(sizes)) > 1, "the adapted mesh never changed size across the march"


def test_transient_adaptive_holds_a_constant_budget():
    """The transient driver targets a CONSTANT complexity each remesh (redistribute DOFs to follow the
    feature) — it must NOT ratchet the mesh up toward max_dofs every remesh like the steady loop does."""
    fem, _ = _heat_fem(mesh_size=0.1, t_end=0.24, nt=16)
    traj = fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=3, max_dofs=4000))
    post = [m[0].shape[0] for m in traj.meshes][4:]  # sizes after the first remesh settles
    assert max(post) < 1.5 * min(post), f"vertex count ratcheted instead of holding a budget: {min(post)}..{max(post)}"


def test_resample_shape_and_final():
    fem, _ = _heat_fem(mesh_size=0.14, t_end=0.15, nt=13)
    traj = fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=4))
    ref = jno.Shape.rect(0, 0, 1, 1, size=0.1).domain()
    ys = np.asarray(traj.resample(ref))
    assert ys.shape == (len(traj), len(np.asarray(ref.mesh.points)))
    state_final, (pts, cells) = traj.final()
    assert np.asarray(state_final).shape[0] == pts.shape[0]  # final state aligned with the final mesh


# ── fail-loud scope guards ────────────────────────────────────────────────────
def test_solve_fn_with_transient_adapt_raises():
    fem, _ = _heat_fem(mesh_size=0.15, t_end=0.1, nt=7)
    with pytest.raises(NotImplementedError, match="owns the time march|solve_fn"):
        fem.solve(adapt=jno.AdaptSpec(every=3), solve_fn=lambda A, b: b)


def test_higher_order_transient_adapt_raises():
    # P2 has more DOFs than vertices → trips the scalar-P1 scope guard.
    fem, _ = _heat_fem(mesh_size=0.2, t_end=0.1, nt=7, order=2)
    with pytest.raises(NotImplementedError, match="scalar-P1 field"):
        fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=3))


def test_metric_field_out_of_range_raises():
    fem, _ = _heat_fem(mesh_size=0.2, t_end=0.1, nt=7)
    with pytest.raises(ValueError, match="out of range"):
        fem.solve(adapt=jno.AdaptSpec(every=3, metric_field=5))


# ── coupled multifield (scalar-P1) ────────────────────────────────────────────
def test_transient_adaptive_two_coupled_scalar_fields():
    """Two heat fields with DIFFERENT κ (block/multifield). If the per-field transfer mixed them, the
    decay rates would be wrong — so matching each field's analytic decay proves the multifield state is
    split, transferred, and re-assembled correctly. Also exercises the (n_save, n_fields, n_ref) resample."""
    ku, kw = 0.12, 0.04
    d = jno.Shape.rect(0, 0, 1, 1, size=0.1).domain(time=(0.0, 0.3, 21))
    u, v = d.fem_symbols(names=("u", "v"))
    w, q = d.fem_symbols(names=("w", "q"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    wi, qi = w.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    ic = jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [xi0, yi0])
    fem = jno.fem(
        [
            ui.t * vi + ku * (ui.x * vi.x + ui.y * vi.y),
            wi.t * qi + kw * (wi.x * qi.x + wi.y * qi.y),
            u(xb, yb) - 0.0,
            w(xb, yb) - 0.0,
            u(xi0, yi0) - ic,
            w(xi0, yi0) - ic,
        ]
    )
    traj = fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=5, max_dofs=6000, metric_field=0))
    assert isinstance(traj, AdaptiveTrajectory)

    ref = jno.Shape.rect(0, 0, 1, 1, size=0.08).domain()
    ys = np.asarray(traj.resample(ref))  # (n_save, n_fields=2, n_ref)
    assert ys.ndim == 3 and ys.shape[1] == 2
    xr = np.asarray(ref.mesh.points)[:, :2]
    base = np.sin(PI * xr[:, 0]) * np.sin(PI * xr[:, 1])
    wu = ww = 0.0
    for kk, t in enumerate(np.asarray(traj.times)):
        eu, ew = np.exp(-2 * ku * PI**2 * t) * base, np.exp(-2 * kw * PI**2 * t) * base
        wu = max(wu, np.linalg.norm(ys[kk, 0] - eu) / max(np.linalg.norm(eu), 1e-12))
        ww = max(ww, np.linalg.norm(ys[kk, 1] - ew) / max(np.linalg.norm(ew), 1e-12))
    assert wu < 0.08 and ww < 0.08, f"multifield decay mismatch (fields not kept separate?): u={wu:.3f}, w={ww:.3f}"
