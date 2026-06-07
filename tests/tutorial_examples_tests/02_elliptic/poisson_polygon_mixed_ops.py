"""02 — 2-D Poisson on a disk with a hybrid AD-outer / FD-inner residual.

Problem
-------
    −Δu(x, y) = 4,        (x, y) in the unit disk x² + y² ≤ 1
        u(x, y) = 0,      on the disk boundary

Analytical solution
-------------------
    u_exact(x, y) = 1 − x² − y²

which satisfies Δu = −4 and vanishes on the boundary. This is the
simplest polynomial Dirichlet problem on the disk — perfect for
verification because the boundary residual is exactly zero with a
multiplicative hard-BC ansatz `u = net(x, y) · (1 − x² − y²)`.

Why this tutorial — the hybrid AD+FD pattern
--------------------------------------------
A single ``jno.core(...)`` ``.solve(...)`` call drives **two** PDE
residuals at once:

  - **Outer region** (sparse mesh) — Laplacian via
    ``scheme="automatic_differentiation"``. Exact derivatives, no
    discretisation error, and only a moderate per-point cost on a small
    network. Sparse meshes mean few points to evaluate AD over.

  - **Inner region** (fine mesh, the rectangle in the middle) — Laplacian
    via ``scheme="finite_difference:cotangent"``. No AD overhead per
    point — the FD stencil is a precomputed sparse linear operator
    applied to the field values. The fine mesh keeps the O(h¹) FD error
    small in the inner region exactly where we have lots of points.

Both residuals reference the **same** network so the parameters are
coupled through training. The hard-BC ansatz eliminates the boundary
residual, so this tutorial is purely about demonstrating the per-
residual scheme split.

Note on FD connectivity
-----------------------
The FD operator at ``trace_evaluator.py:1128`` uses the *full* mesh's
``mesh_connectivity``, not just the inner region's nodes. The per-region
mesh sizing (`region_mesh_sizes={"inner": 0.02}`) gives FD genuinely
fine neighbours in the inner region; at the inner/outer interface the
stencil reaches into the coarser outer mesh but those contributions
contribute negligibly when the outer mesh is much sparser than the
inner stencil reach.
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

# ── Geometry: 64-vertex disk approximation + axis-aligned inner rectangle ───
N_DISK_VERTS = 64
theta = np.linspace(0.0, 2.0 * np.pi, N_DISK_VERTS, endpoint=False)
disk_verts = [(float(np.cos(t)), float(np.sin(t))) for t in theta]
rect_verts = [(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)]

dom = jno.domain.csg.from_polygons({"outer": disk_verts, "inner": rect_verts})
dom.build_mesh(mesh_size=0.08, region_mesh_sizes={"inner": 0.02})

# ── Sample interior tags from each region separately ────────────────────────
x_o, y_o, _ = dom.variable("interior_outer")
x_i, y_i, _ = dom.variable("interior_inner")

# ── Network: small tanh MLP, 2 → 32 → 32 → 1 ─────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(
        in_features=2,
        hidden_dims=32,
        num_layers=3,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(
    optax.adam(
        optax.exponential_decay(
            init_value=1e-3,
            transition_steps=500,
            decay_rate=0.5,
            end_value=1e-5,
        )
    )
)

# ── Hard-BC ansatz on each region — (1 − x² − y²) vanishes on x²+y² = 1 ─────
u_outer = net(x_o, y_o) * (1.0 - x_o**2 - y_o**2)
u_inner = net(x_i, y_i) * (1.0 - x_i**2 - y_i**2)

# ── Two PDE residuals, different schemes per region ─────────────────────────
# −Δu = 4 in both regions.
res_outer = (-u_outer.laplacian(x_o, y_o, scheme="automatic_differentiation") - 4.0).mse
res_inner = (-u_inner.laplacian(x_i, y_i, scheme="finite_difference:cotangent") - 4.0).mse

crux = jno.core([res_outer, res_inner], dom)
history = crux.solve(5000)

# ── Evaluate against the analytic solution on both regions ──────────────────
u_target_outer = 1.0 - x_o**2 - y_o**2
u_target_inner = 1.0 - x_i**2 - y_i**2
u_pred_outer, u_exact_outer, u_pred_inner, u_exact_inner = crux.eval([u_outer, u_target_outer, u_inner, u_target_inner])

rel_l2_outer = float(jnp.linalg.norm(u_pred_outer - u_exact_outer) / (jnp.linalg.norm(u_exact_outer) + 1e-8))
rel_l2_inner = float(jnp.linalg.norm(u_pred_inner - u_exact_inner) / (jnp.linalg.norm(u_exact_inner) + 1e-8))

# ── Write result to tracking file (per the 02_elliptic/poisson_2d.py convention) ──
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"02_elliptic/poisson_polygon_mixed_ops.py | epochs=5000 | "
        f"AD_outer_rel_L2={rel_l2_outer:.6e} | FD_inner_rel_L2={rel_l2_inner:.6e}\n"
    )

assert rel_l2_outer < 0.10, f"AD-outer relative L² error too large: {rel_l2_outer:.3e}"
assert rel_l2_inner < 0.10, f"FD-inner relative L² error too large: {rel_l2_inner:.3e}"
