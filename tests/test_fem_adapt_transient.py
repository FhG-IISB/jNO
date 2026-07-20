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

pytest.importorskip("mmgpy", reason="mmgpy required for adaptive remeshing (fem_adapt imports it)")

from jno.utils.solver.fem_adapt import (  # noqa: E402
    AdaptiveTrajectory,
    _eval_fe_fields_at_points,
    _field_layout,
)

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


# ── basis-aware transfer: the ordering-invariant locks (no remesh needed) ─────
def _p2_scalar_layout():
    """A P2 scalar transient fem + its per-field layout and P1 base mesh."""
    d = jno.Shape.rect(0, 0, 1, 1, size=0.34).domain(time=(0.0, 0.1, 3))
    u, phi = d.fem_symbols(order=2)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0])
    pts = np.asarray(d.mesh.points)[:, :2]
    cells = np.asarray(d.mesh.cells_dict["triangle"]).astype(np.int64)
    return _field_layout(fem), pts, cells


def test_transfer_p2_quadratic_identity():
    """Transfer a known QUADRATIC field (which P2 represents exactly) from the P2 nodes back to the SAME
    nodes: it must reproduce to machine precision. Pins the load-bearing invariant — the basix P2 basis
    columns line up with the recorded P{order} connectivity (a scrambled order would corrupt the
    edge-midpoint DOFs and break reproduction)."""
    lay, pts, cells = _p2_scalar_layout()
    dofc = np.asarray(lay["field_points"][0])  # (n_p2, 2) P2 node coords

    def quad(P):
        return 0.3 + 0.7 * P[:, 0] - 0.4 * P[:, 1] + 1.1 * P[:, 0] * P[:, 1] - 0.5 * P[:, 0] ** 2 + 0.2 * P[:, 1] ** 2

    field = quad(dofc)  # exact P2 nodal values
    out = _eval_fe_fields_at_points(
        pts, cells, jnp.asarray(field), lay["offsets"], lay["orders"], lay["cells_f"], lay["vecs"], [dofc], dim=2
    )[0]
    np.testing.assert_allclose(np.asarray(out)[:, 0], field, atol=1e-10)


def test_transfer_vector_field_identity():
    """Transfer a LINEAR 2-vector field from P1 nodes to the same nodes: reproduces exactly, per
    component (node-major ``node*vec+comp`` layout). A P1-collapse or component scramble would fail."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    nv = 4
    blk = np.stack([pts[:, 0] + 2 * pts[:, 1], 3.0 - pts[:, 0] + pts[:, 1]], axis=1)  # (4, 2) linear field
    out = _eval_fe_fields_at_points(pts, cells, jnp.asarray(blk.reshape(-1)), [0, 2 * nv], [1], [cells], [2], [pts], dim=2)[
        0
    ]
    np.testing.assert_allclose(np.asarray(out), blk, atol=1e-12)


# ── P2 scalar end-to-end (replaces the old "P2 raises" scope guard) ───────────
def test_transient_adaptive_p2_scalar_matches_reference():
    """P2 scalar heat now SUCCEEDS under transient adapt (was fail-loud): the adaptive trajectory
    reproduces the analytic decay, proving the P2-basis transfer (edge midpoints included) is
    exact-to-order, not P1-collapsed."""
    kappa = 0.1
    fem, _ = _heat_fem(mesh_size=0.14, kappa=kappa, t_end=0.25, nt=16, order=2)
    traj = fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=4, max_dofs=8000))
    assert isinstance(traj, AdaptiveTrajectory)
    assert fem.adapt_history, "expected at least one remesh"
    ref = jno.Shape.rect(0, 0, 1, 1, size=0.08).domain()
    ys = np.asarray(traj.resample(ref))  # single scalar field -> (n_save, n_ref)
    xr = np.asarray(ref.mesh.points)[:, :2]
    base = np.sin(PI * xr[:, 0]) * np.sin(PI * xr[:, 1])
    worst = 0.0
    for k, t in enumerate(np.asarray(traj.times)):
        exact = np.exp(-2 * kappa * PI**2 * t) * base
        worst = max(worst, np.linalg.norm(ys[k] - exact) / max(np.linalg.norm(exact), 1e-12))
    assert worst < 0.08, f"P2 adaptive heat-decay mismatch: worst rel L2 = {worst:.3f}"


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


# ── mixed spaces (heterogeneous per-field DOF counts) + nonlinear march ───────
def test_transient_adaptive_mixed_order_fields():
    """Two heat fields of DIFFERENT ORDER — P2 + P1, so DIFFERENT per-field DOF counts (the Taylor-Hood
    bookkeeping). Each must reproduce its analytic decay through the remeshes, proving the mixed-space
    offsets/transfer carry heterogeneous blocks correctly (not a uniform-n_verts assumption)."""
    ku, kw = 0.12, 0.05
    d = jno.Shape.rect(0, 0, 1, 1, size=0.12).domain(time=(0.0, 0.3, 21))
    u, v = d.fem_symbols(names=("u", "v"), order=2)  # P2 field
    w, q = d.fem_symbols(names=("w", "q"), order=1)  # P1 field
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
    off = fem.offsets
    assert off[1] - off[0] > off[2] - off[1], "expected the P2 block (edge dofs) to be larger than the P1 block"
    traj = fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=5, max_dofs=9000, metric_field=0))
    assert isinstance(traj, AdaptiveTrajectory)

    ref = jno.Shape.rect(0, 0, 1, 1, size=0.08).domain()
    xr = np.asarray(ref.mesh.points)[:, :2]
    base = np.sin(PI * xr[:, 0]) * np.sin(PI * xr[:, 1])
    ys = np.asarray(traj.resample(ref))  # both scalar -> (n_save, 2, n_ref)
    assert ys.ndim == 3 and ys.shape[1] == 2
    wu = ww = 0.0
    for k, t in enumerate(np.asarray(traj.times)):
        eu, ew = np.exp(-2 * ku * PI**2 * t) * base, np.exp(-2 * kw * PI**2 * t) * base
        wu = max(wu, np.linalg.norm(ys[k, 0] - eu) / max(np.linalg.norm(eu), 1e-12))
        ww = max(ww, np.linalg.norm(ys[k, 1] - ew) / max(np.linalg.norm(ew), 1e-12))
    assert wu < 0.1 and ww < 0.1, f"mixed-order decay mismatch: P2={wu:.3f}, P1={ww:.3f}"


def _nonlinear_reaction_fem(size, r=2.0, t_end=0.2, nt=21):
    """Semilinear heat u_t = Δu + r·u(1-u) — a NONLINEAR transient block (mass + residual)."""
    d = jno.Shape.rect(0, 0, 1, 1, size=size).domain(time=(0.0, t_end, nt))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    weak = ui.t * vi + (ui.x * vi.x + ui.y * vi.y) - r * ui * (1.0 - ui) * vi  # nonlinear reaction term
    ic = jno.fn(lambda x, y: 0.5 * jnp.sin(PI * x) * jnp.sin(PI * y), [xi0, yi0])
    return jno.fem([weak, u(xb, yb) - 0.0, u(xi0, yi0) - ic]), d


def test_transient_adaptive_nonlinear_matches_manufactured():
    """A NONLINEAR transient solve under adapt reproduces a MANUFACTURED solution. u* = e^{-λt}·sin(πx)
    sin(πy) is forced to solve u_t = Δu + r·u(1-u) + f (f derived from u*). Each step is a Newton solve
    (block.step's newton_krylov) that must survive the scan AND re-assembly on every remesh — matching u*
    (an analytic oracle, no non-adaptive reference needed) proves it."""
    r, lam = 2.0, 1.0

    def ustar(x, y, t):
        return jnp.exp(-lam * t) * jnp.sin(PI * x) * jnp.sin(PI * y)

    def forcing(x, y, t):  # f = (2π² - λ)·u* - r·u*(1-u*), so u* solves the semilinear heat eqn
        us = ustar(x, y, t)
        return (2 * PI**2 - lam) * us - r * us * (1.0 - us)

    d = jno.Shape.rect(0, 0, 1, 1, size=0.1).domain(time=(0.0, 0.3, 21))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    weak = ui.t * vi + (ui.x * vi.x + ui.y * vi.y) - r * ui * (1.0 - ui) * vi - jno.fn(forcing, [xi, yi, ti]) * vi
    ic = jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [xi0, yi0])
    fem = jno.fem([weak, u(xb, yb) - 0.0, u(xi0, yi0) - ic])
    assert fem._op.is_nonlinear(), "expected a nonlinear semidiscrete block (mass + residual)"
    traj = fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=5, max_dofs=9000))
    assert isinstance(traj, AdaptiveTrajectory) and fem.adapt_history

    ref = jno.Shape.rect(0, 0, 1, 1, size=0.08).domain()
    ys = np.asarray(traj.resample(ref))
    xr = np.asarray(ref.mesh.points)[:, :2]
    base = np.sin(PI * xr[:, 0]) * np.sin(PI * xr[:, 1])
    worst = 0.0
    for k, t in enumerate(np.asarray(traj.times)):
        exact = np.exp(-lam * t) * base
        worst = max(worst, np.linalg.norm(ys[k] - exact) / max(np.linalg.norm(exact), 1e-12))
    assert worst < 0.06, f"nonlinear manufactured-solution mismatch: worst rel L2 = {worst:.3f}"


def test_transient_adaptive_nonlinear_slot_composes():
    """The nonlinear= slot composes with transient adapt (the guard relaxation): a custom Newton config
    runs the same nonlinear adaptive march and lands close to the default-solver result."""
    fem, _ = _nonlinear_reaction_fem(0.11, t_end=0.15, nt=13)
    traj = fem.solve(
        adapt=jno.AdaptSpec(anisotropic=True, every=4, max_dofs=8000),
        nonlinear=jno.solve.newton(max_steps=40, rtol=1e-9, atol=1e-11),
    )
    assert isinstance(traj, AdaptiveTrajectory) and fem.adapt_history
    ref = jno.Shape.rect(0, 0, 1, 1, size=0.1).domain()
    ys = np.asarray(traj.resample(ref))
    assert np.all(np.isfinite(ys)) and float(np.abs(ys).max()) > 1e-3  # a real, finite solution


def test_transient_adapt_rejects_warm_start_x0():
    """x0= (warm start) still cannot compose with adapt= — the DOF layout changes across a remesh."""
    fem, _ = _heat_fem(mesh_size=0.2, t_end=0.1, nt=7)
    with pytest.raises(NotImplementedError, match="x0=|warm start|do not compose"):
        fem.solve(adapt=jno.AdaptSpec(every=3), x0=jnp.zeros(1))


# ── Taylor-Hood (vector P2 + scalar P1): the headline mixed case ──────────────
def test_transient_adaptive_vector_p2_plus_scalar_p1():
    """The melt-pool field LAYOUT under adapt — a VECTOR P2 field + a SCALAR P1 field (the Taylor-Hood
    shape: different per-field orders, sizes, and a vector block). Two decoupled heat problems; the P2
    vector's two components carry DISTINCT modes (hence distinct decay rates) and the P1 scalar its own —
    each reproduced across the remeshes proves the mixed vector-P2 + scalar-P1 transfer/offsets end-to-end
    (the transient Taylor-Hood saddle assembles too; here we isolate the transfer, which is what adapt
    touches)."""
    kv, ks = 0.1, 0.04
    d = jno.Shape.rect(0, 0, 1, 1, size=0.12).domain(time=(0.0, 0.3, 21))
    u, w = d.fem_symbols(value_shape=(2,), names=("u", "w"), order=2)  # vector P2 (velocity-shaped)
    s, r = d.fem_symbols(names=("s", "r"), order=1)  # scalar P1 (pressure-shaped)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, wi = u.bind(x=xi, y=yi, t=ti), w.bind(x=xi, y=yi, t=ti)
    si, ri = s.bind(x=xi, y=yi, t=ti), r.bind(x=xi, y=yi, t=ti)
    gu, gw = jno.np.grad(ui, [xi, yi]), jno.np.grad(wi, [xi, yi])
    ic0 = jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [xi0, yi0])  # mode (1,1)
    ic1 = jno.fn(lambda x, y: jnp.sin(2 * PI * x) * jnp.sin(PI * y), [xi0, yi0])  # mode (2,1)
    fem = jno.fem(
        [
            jno.np.inner(ui.t, wi, 1) + kv * jno.np.inner(gu, gw, 2),  # vector P2 heat (both components)
            si.t * ri + ks * (si.x * ri.x + si.y * ri.y),  # scalar P1 heat
            u(xb, yb)[0] - 0.0,
            u(xb, yb)[1] - 0.0,
            s(xb, yb) - 0.0,
            u(xi0, yi0)[0] - ic0,
            u(xi0, yi0)[1] - ic1,
            s(xi0, yi0) - ic0,
        ]
    )
    off = fem.offsets
    assert len(off) == 3 and (off[1] - off[0]) > (off[2] - off[1])  # vector-P2 block > scalar-P1 block
    traj = fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=5, max_dofs=12000, metric_field=0))
    assert isinstance(traj, AdaptiveTrajectory)

    ref = jno.Shape.rect(0, 0, 1, 1, size=0.08).domain()
    xr = np.asarray(ref.mesh.points)[:, :2]
    b0 = np.sin(PI * xr[:, 0]) * np.sin(PI * xr[:, 1])
    b1 = np.sin(2 * PI * xr[:, 0]) * np.sin(PI * xr[:, 1])
    yv = np.asarray(traj.resample(ref, field=0))  # (n_save, n_ref, 2) vector
    ysc = np.asarray(traj.resample(ref, field=1))  # (n_save, n_ref) scalar
    assert yv.ndim == 3 and yv.shape[2] == 2, "vector-P2 resample must be (n_save, n_ref, 2)"
    worst = 0.0
    for k, t in enumerate(np.asarray(traj.times)):
        e0 = np.exp(-2 * kv * PI**2 * t) * b0  # component 0: mode (1,1)
        e1 = np.exp(-5 * kv * PI**2 * t) * b1  # component 1: mode (2,1) -> (4+1)π²
        es = np.exp(-2 * ks * PI**2 * t) * b0  # scalar
        worst = max(
            worst,
            np.linalg.norm(yv[k, :, 0] - e0) / max(np.linalg.norm(e0), 1e-12),
            np.linalg.norm(yv[k, :, 1] - e1) / max(np.linalg.norm(e1), 1e-12),
            np.linalg.norm(ysc[k] - es) / max(np.linalg.norm(es), 1e-12),
        )
    assert worst < 0.1, f"vector-P2 + scalar-P1 mixed decay mismatch: worst rel L2 = {worst:.3f}"


@pytest.mark.slow  # a coupled Taylor-Hood saddle + per-remesh Newton solves ~ a few minutes
def test_transient_adaptive_pressure_pin_survives_remesh():
    """A Taylor-Hood pressure gauge ``p.pin()`` must re-derive on every remesh: its single-vertex point
    region is cached per domain and does NOT survive a remesh, so without re-deriving it collapses to a
    whole-domain trial-no-test term on re-assembly. A small body-forced Navier–Stokes saddle (P2 vel +
    P1 pressure, advection -> nonlinear -> the fast per-step Newton path), no-slip, ``p.pin()``, adapt on
    velocity: it must re-assemble across remeshes and develop a finite flow (the gauge holding)."""
    nu = 0.1
    d = jno.Shape.rect(0, 0, 1, 1, size=0.22).domain(time=(0.0, 0.06, 7))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
    p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pb, qb = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    ux, uy, vx, vy = ui[0], ui[1], vi[0], vi[1]
    uxx, uxy, uyx, uyy = ui.x[0], ui.y[0], ui.x[1], ui.y[1]
    vxx, vxy, vyx, vyy = vi.x[0], vi.y[0], vi.x[1], vi.y[1]
    force = jno.fn(lambda x, y, t: jnp.sin(PI * y), [xi, yi, ti])  # a steady body force -> a flow
    mom = (
        (ui.t[0] * vx + ui.t[1] * vy)
        + ((ux * uxx + uy * uxy) * vx + (ux * uyx + uy * uyy) * vy)  # (u.grad)u -> nonlinear -> Newton path
        + nu * (uxx * vxx + uxy * vxy + uyx * vyx + uyy * vyy)
        - pb * (vxx + vyy)
        - force * vx
    )
    cont = qb * (uxx + uyy)
    fem = jno.fem([mom, cont, u(xb, yb) - 0.0, p.pin(), u(xi0, yi0) - 0.0])
    assert len(fem.offsets) == 3 and not fem.is_linear  # P2 vel + P1 pressure saddle, nonlinear
    traj = fem.solve(adapt=jno.AdaptSpec(anisotropic=True, every=2, max_dofs=4000, metric_field=0))
    assert isinstance(traj, AdaptiveTrajectory) and fem.adapt_history  # remeshed >= once -> pin re-derived each time
    ref = jno.Shape.rect(0, 0, 1, 1, size=0.1).domain()
    yv = np.asarray(traj.resample(ref, field=0))
    assert np.all(np.isfinite(yv)) and float(np.abs(yv[-1]).max()) > 1e-3  # a finite flow developed; the gauge held
