# --8<-- [start:code]
r"""Soft p-enrichment on the L-shape: the order distribution as a differentiable variable.

Same problem as ``adaptive_l_shape.py``: ``-lap u = f`` on an L-shape, ``u = 0`` on the boundary, ``f`` a
C^3 bump in the lower-right arm, so the field has a re-entrant-corner singularity and a smooth bump.
One scalar ``space="cover"`` field on a fixed coarse mesh.

``jno.solve.enrich`` decides per node, with a boolean mask, whether its interpolation cover is on.
Here each node gets a weight ``w_i in [0, 1]`` instead, through a diagonal penalty on its cover DOFs
(``jno/utils/solver/fem_soft_cover.py``), so the Ritz energy

    Pi(w) = 1/2 u(w).A u(w) - b.u(w),     2 (Pi - Pi_exact) = |u - u_exact|_a^2

is differentiable in ``w`` and one compiled program serves every ``w``. Three results:

A. The AD indicator ``-dPi/dw`` at ``w = 0`` against its closed form and against a gradient-magnitude
   criterion, as one-shot rankings at equal enriched-node budgets.
B. Budgeted optimisation ``min Pi(w)`` s.t. ``sum w = k``, ``0 <= w <= 1``, by projected gradient, for a
   linear gate and a SIMP gate ``w^3``; thresholded to ``k`` nodes and compared with the greedy
   ``jno.solve.enrich`` loop at the SAME enriched counts (the budgets ARE that loop's rounds).
C. XLA compilations: the greedy loop against the whole soft optimisation.
D. Uniform P1 on finer meshes at the same active DOFs.

The soft weights live in the private module ``jno.utils.solver.fem_soft_cover``; they are not exposed
through ``jno.solve`` yet. The closing asserts check what the tables show: the soft energy equals the
hard-mask energy at ``w in {0, 1}``, the AD indicator equals its closed form, the budgeted optimum beats
the greedy loop and uniform P1 at every budget, and the whole optimisation compiles fewer programs.

Run::

    python docs/tutorial_examples/08_fem_and_varpinns/soft_enrich_l_shape.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_soft_cover import (  # noqa: E402
    cover_dof_index,
    hierarchical_indicator,
    operator_diagonal,
    ritz_energy,
    soft_solve,
)
from jno.utils.solver.linear import sparse_lu_solve  # noqa: E402

L_SHAPE = [(0, 0), (1.0, 0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0, 1.0)]
BUMP = (0.74, 0.22, 0.17, 60.0)  # cx, cy, R, amplitude
H = 0.075  # the coarse mesh, as in adaptive_l_shape.py
H_REF = 0.015  # fully enriched (third-order) reference, as in adaptive_l_shape.py
STEPS = 300  # projected-gradient steps per budget
SIMP_P = 3.0


def source(x, y, xp):
    cx, cy, R, A = BUMP
    s = ((x - cx) ** 2 + (y - cy) ** 2) / R**2
    return A * xp.where(s < 1.0, 1.0 - s, 0.0) ** 4


def build(size, mask=None):
    d = jno.shape.polygon(L_SHAPE, size=size).domain()
    if mask is not None:
        d._fem_enriched_nodes = np.asarray(mask, dtype=bool)
    u, phi = d.fem_symbols(space="cover")
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - source(xi, yi, jno.np) * vi, u(xb, yb) - 0.0])
    return d, fem, ui


def solve_fn(a, b):
    return sparse_lu_solve(a, b)


# --- compile counter (C) -------------------------------------------------------------------------------
_compiles = [0]


def _on_event(name, *_a, **_k):
    if "backend_compile" in name:
        _compiles[0] += 1


jax.monitoring.register_event_duration_secs_listener(_on_event)

# --- reference and the coarse fully enriched operator ---------------------------------------------------
d_ref, fem_ref, _ = build(H_REF)
A_ref, b_ref = fem_ref.operator
PI_REF = float(ritz_energy(A_ref, b_ref, sparse_lu_solve(A_ref, b_ref)))

d0, fem0, ui0 = build(H)
A, b = fem0.operator
b = jnp.asarray(b).reshape(-1)
pts = np.asarray(d0.mesh.points)[:, :2]
tris = np.asarray(d0.mesh.cells_dict["triangle"])
N = pts.shape[0]
IDX = cover_dof_index(N, 2)
C = operator_diagonal(A, IDX)


def err(w):
    """Energy-norm error squared / 2 of the softly enriched solution at weights ``w``."""
    return float(ritz_energy(A, b, soft_solve(A, b, IDX, C, jnp.asarray(w, dtype=float)))) - PI_REF


# Active DOFs of a mask: free DOFs of the full build, minus the free cover DOFs of unenriched nodes. A
# boundary node's covers are partly pinned already, so enriching it costs fewer DOFs than an interior one.
_PINNED = {int(i) for i, _ in d0._fem_native_dirichlet_pairs}
_FREE_COV = np.array([[int(k) not in _PINNED for k in row] for row in IDX])  # (N, 2)


def active_dofs(mask):
    n_nodal_free = sum(1 for i in range(N) if int(IDX[i, 0] - 1) not in _PINNED)
    return n_nodal_free + int(_FREE_COV[np.asarray(mask, dtype=bool)].sum())


def top_k(score, k):
    m = np.zeros(N, dtype=bool)
    m[np.argsort(-np.asarray(score))[:k]] = True
    return m


E_P1, E_FULL = err(np.zeros(N)), err(np.ones(N))
print(f"L-shape, H={H}: {N} nodes, {A.shape[0]} padded DOFs; reference cover h={H_REF}, Pi_ref={PI_REF:.8f}")
print(f"  error (Pi - Pi_ref):  P1 {E_P1:.4e}    full cover {E_FULL:.4e}")

# --- greedy enrich loop: the existing path, and the budgets every comparison uses -----------------------
d_g, fem_g, ui_g = build(H)
c0 = _compiles[0]
fem_g.solve(solve_fn, adapt=jno.solve.enrich(criterion=jno.np.sqrt(ui_g.x**2 + ui_g.y**2), theta=0.5, max_iters=6))
compiles_greedy = _compiles[0] - c0
rounds = fem_g.adapt_history
masks_g = [np.asarray(r["enriched"], dtype=bool) for r in rounds] + [np.asarray(d_g._fem_enriched_nodes, dtype=bool)]
masks_g = [m for i, m in enumerate(masks_g) if m.sum() > 0 and (i == 0 or m.sum() != masks_g[i - 1].sum())]
budgets = [int(m.sum()) for m in masks_g]

# hard-path cross-check of the soft energy on the loop's final mask (the loop rebinds fem_g to it)
A_m, b_m = fem_g.operator
E_hard = float(ritz_energy(A, b, sparse_lu_solve(A_m, b_m))) - PI_REF
E_soft = err(masks_g[-1].astype(float))
print(f"  hard mask vs soft w in {{0,1}} on the loop's final mask: {E_hard:.6e} vs {E_soft:.6e}")

# --- A. indicators at w = 0 ------------------------------------------------------------------------------
pi = lambda w: ritz_energy(A, b, soft_solve(A, b, IDX, C, w))  # noqa: E731
grad_pi = jax.jit(jax.grad(pi))
g0 = np.asarray(grad_pi(jnp.zeros(N)))
h0 = np.asarray(hierarchical_indicator(A, b, IDX, C, soft_solve(A, b, IDX, C, jnp.zeros(N))))
live = np.abs(h0) > 1e-10 * np.abs(h0).max()
print(
    f"\nA. AD gradient at w=0 vs closed form sum r^2/A_kk: worst rel diff {np.max(np.abs(g0 - h0)[live] / np.abs(h0[live])):.1e}"
)

# gradient-magnitude criterion on the P1 field: area-weighted mean of the cell gradients at each node
u_p1 = np.asarray(soft_solve(A, b, IDX, C, jnp.zeros(N)))[IDX[:, 0] - 1]
v = pts[tris]
e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
Jinv = np.stack([np.stack([e2[:, 1], -e1[:, 1]], -1), np.stack([-e2[:, 0], e1[:, 0]], -1)], -1) / det[:, None, None]
du = np.stack([u_p1[tris[:, 1]] - u_p1[tris[:, 0]], u_p1[tris[:, 2]] - u_p1[tris[:, 0]]], -1)
gcell = np.einsum("cij,cj->ci", np.transpose(Jinv, (0, 2, 1)), du)
area = 0.5 * np.abs(det)
gnode, wnode = np.zeros(N), np.zeros(N)
for a in range(3):
    np.add.at(gnode, tris[:, a], area * np.linalg.norm(gcell, axis=1))
    np.add.at(wnode, tris[:, a], area)
gnode /= wnode

# --- B. budgeted projected gradient -----------------------------------------------------------------------


def project(v_, k):
    """Euclidean projection onto {0 <= w <= 1, sum w = k}: bisection on the shift tau."""
    lo, hi = jnp.min(v_) - 1.0, jnp.max(v_)

    def body(_, lh):
        lo_, hi_ = lh
        mid = 0.5 * (lo_ + hi_)
        s = jnp.sum(jnp.clip(v_ - mid, 0.0, 1.0))
        return jnp.where(s > k, mid, lo_), jnp.where(s > k, hi_, mid)

    lo, hi = jax.lax.fori_loop(0, 60, body, (lo, hi))
    return jnp.clip(v_ - 0.5 * (lo + hi), 0.0, 1.0)


def make_optimiser(p):
    obj = (lambda w: pi(w**p)) if p != 1.0 else pi
    vg = jax.value_and_grad(obj)

    @jax.jit
    def run(w0, k):
        def step(i, w):
            _, g = vg(w)
            eta = 0.3 * (1.0 - i / STEPS) + 0.01  # normalised step, decaying
            return project(w - eta * g / (jnp.max(jnp.abs(g)) + 1e-300), k)

        return jax.lax.fori_loop(0, STEPS, step, w0)

    return run


opt_lin, opt_simp = make_optimiser(1.0), make_optimiser(SIMP_P)
c0 = _compiles[0]
rows = []
for m_g, k in zip(masks_g, budgets):
    w_start = jnp.full(N, k / N)
    w_lin = np.asarray(opt_lin(w_start, float(k)))
    w_simp = np.asarray(opt_simp(w_start, float(k)))
    m_lin = top_k(w_lin, k)
    binary = lambda w: float(np.mean((w < 0.05) | (w > 0.95)))  # noqa: E731
    rows.append(
        {
            "k": k,
            "greedy": err(m_g.astype(float)),
            "dofs_g": active_dofs(m_g),
            "grad1": err(top_k(gnode, k).astype(float)),
            "ad1": err(top_k(-g0, k).astype(float)),
            "lin": err(m_lin.astype(float)),
            "dofs_lin2": active_dofs(m_lin),
            "dofs_ad": active_dofs(top_k(-g0, k)),
            "lin_relaxed": err(w_lin),
            "lin_bin": binary(w_lin),
            "simp": err(top_k(w_simp, k).astype(float)),
            "simp_bin": binary(w_simp),
        }
    )

# spread over random starts, at the middle budget
k_mid = budgets[len(budgets) // 2]
rng = np.random.default_rng(0)
spread = {"lin": [], "simp": []}
for _ in range(3):
    w_r = project(jnp.asarray(rng.random(N)), float(k_mid))
    spread["lin"].append(err(top_k(np.asarray(opt_lin(w_r, float(k_mid))), k_mid).astype(float)))
    spread["simp"].append(err(top_k(np.asarray(opt_simp(w_r, float(k_mid))), k_mid).astype(float)))
compiles_soft = _compiles[0] - c0

print("\nB. Energy error Pi - Pi_ref at equal enriched-node counts k (lower is better; all binary masks)")
print(
    f"{'k':>5} {'dofs g/AD/lin':>15} {'%nodes':>7} {'greedy':>11} {'|grad| 1x':>11} {'AD 1x':>11} {'opt lin':>11} {'opt SIMP':>11}   relaxed lin   binary lin/SIMP"
)
for r in rows:
    print(
        f"{r['k']:>5} {str(r['dofs_g']) + '/' + str(r['dofs_ad']) + '/' + str(r['dofs_lin2']):>15} {100 * r['k'] / N:>6.0f}% {r['greedy']:>11.4e} {r['grad1']:>11.4e} {r['ad1']:>11.4e} "
        f"{r['lin']:>11.4e} {r['simp']:>11.4e}   {r['lin_relaxed']:>11.4e}   {r['lin_bin']:.0%}/{r['simp_bin']:.0%}"
    )
print(f"  (P1 {E_P1:.4e}, full cover {E_FULL:.4e} at {N} nodes)")
print(
    f"  random starts at k={k_mid}: opt lin {min(spread['lin']):.4e}..{max(spread['lin']):.4e}, "
    f"opt SIMP {min(spread['simp']):.4e}..{max(spread['simp']):.4e}"
)

print("\nC. XLA compilations")
print(f"  greedy enrich loop, {len(rounds)} rounds:                       {compiles_greedy}")
print(f"  soft: {len(budgets)} budgets x 2 gates x {STEPS} steps + 6 random-start runs: {compiles_soft}")

# --- D. uniform P1 at equal DOFs: does h beat p here? ---------------------------------------------------------
# P1 on a finer mesh is the cover space with every weight 0 there. Its error is interpolated (log-log) to the
# active DOFs of the budgeted masks of B.
p1_curve = []
for fct in (1.0, 0.85, 0.75, 0.65, 0.55):
    d_h, fem_h, _ = build(H * fct)
    A_h, b_h = fem_h.operator
    b_h = jnp.asarray(b_h).reshape(-1)
    n_h = np.asarray(d_h.mesh.points).shape[0]
    pins_h = {int(i) for i, _ in d_h._fem_native_dirichlet_pairs}
    dof_h = sum(1 for i in range(n_h) if 3 * i not in pins_h)
    idx_h = cover_dof_index(n_h, 2)
    e_h = float(ritz_energy(A_h, b_h, soft_solve(A_h, b_h, idx_h, operator_diagonal(A_h, idx_h), jnp.zeros(n_h)))) - PI_REF
    p1_curve.append((dof_h, e_h))
p1_dofs, p1_err = (np.array(v) for v in zip(*sorted(p1_curve)))
print(
    "\nD. Uniform P1 at the same active DOFs (log-log interpolation over meshes",
    ", ".join(f"{d}" for d in p1_dofs),
    "DOFs)",
)
print(f"{'k':>5} {'DOFs':>6} {'opt lin':>11} {'greedy':>11} {'uniform P1':>11}")
for r in rows:
    t = r["dofs_lin2"]
    e_u = (
        float(np.exp(np.interp(np.log(t), np.log(p1_dofs), np.log(p1_err))))
        if p1_dofs[0] <= t <= p1_dofs[-1]
        else float("nan")
    )
    print(f"{r['k']:>5} {t:>6} {r['lin']:>11.4e} {r['greedy']:>11.4e} {e_u:>11.4e}")

# --- the claims the tables make ------------------------------------------------------------------------------
assert abs(E_hard - E_soft) <= 1e-5 * abs(E_hard), "soft weights in {0, 1} must reproduce the hard mask"
assert np.max(np.abs(g0 - h0)[live] / np.abs(h0[live])) < 1e-3, "AD indicator at w = 0 != its closed form"
assert all(r["lin"] <= r["greedy"] for r in rows), "the budgeted optimum lost to greedy marking"
for r in rows:
    t = r["dofs_lin2"]
    if p1_dofs[0] <= t <= p1_dofs[-1]:
        assert r["lin"] < float(np.exp(np.interp(np.log(t), np.log(p1_dofs), np.log(p1_err)))), "uniform P1 won"
assert compiles_soft < compiles_greedy, "the soft optimisation should compile fewer programs than the greedy loop"
print("\nall checks passed")
# --8<-- [end:code]
