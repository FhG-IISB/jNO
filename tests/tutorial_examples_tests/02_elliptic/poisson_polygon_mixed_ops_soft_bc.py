"""02 — Hybrid AD/FD Poisson on a disk with **soft** Dirichlet BC.

Soft-BC variant of ``poisson_polygon_mixed_ops.py``. The PDE setup is
identical:

    −Δu(x, y) = 4,   in the unit disk
        u(x, y) = 0, on the disk boundary

What's different
----------------
Where the companion tutorial uses a **hard** Dirichlet ansatz
``u = net(x, y) · (1 − x² − y²)`` to enforce ``u = 0`` on ``∂Ω`` by
construction, this version uses the raw network output ``u = net(x, y)``
and asks the optimiser to satisfy the boundary condition through an
additional **boundary residual** in the loss:

    L = ‖ −Δu_outer − 4 ‖²  +  ‖ −Δu_inner − 4 ‖²  +  ‖ u_boundary − 0 ‖²
        \\__________ AD __________/   \\____ FD ____/   \\____ soft BC ____/

So three residuals share the same network parameters. The BC residual
samples ``boundary_outer`` — the combined CSG tag covering all 64
polygon edges of the disk approximation.

When to prefer soft vs hard
---------------------------
- **Hard ansatz** (companion tutorial): exact BC by construction, BC
  residual is zero, training only needs to satisfy the PDE. Best when a
  simple closed-form distance function to ``∂Ω`` exists (here the disk).
- **Soft residual** (this tutorial): more general — works for any
  geometry without needing a closed-form ansatz, supports mixed/Neumann/
  Robin BCs trivially. Trades a small accuracy penalty + a balancing
  challenge for the BC weight.

The hybrid AD-outer + FD-inner split is unchanged: a single ``net`` is
shared across all three residuals and the optimiser drives them
simultaneously through ``jno.core(...).solve(...)``.
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

# ── Geometry: 64-vertex disk + axis-aligned inner rectangle ─────────────────
N_DISK_VERTS = 64
theta = np.linspace(0.0, 2.0 * np.pi, N_DISK_VERTS, endpoint=False)
disk_verts = [(float(np.cos(t)), float(np.sin(t))) for t in theta]
rect_verts = [(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)]

dom = jno.domain.csg.from_polygons({"outer": disk_verts, "inner": rect_verts})
dom.build_mesh(mesh_size=0.08, region_mesh_sizes={"inner": 0.02})

# ── Region tags ─────────────────────────────────────────────────────────────
x_o, y_o, _ = dom.variable("interior_outer")
x_i, y_i, _ = dom.variable("interior_inner")
# `boundary_outer` is the combined polygon-approximated disk boundary
# (all 64 edges of the 64-gon). Per-edge tags `boundary_outer_0` ...
# `boundary_outer_63` are also available if you ever need to apply
# different BCs on different arcs.
x_b, y_b, _ = dom.variable("boundary_outer")

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

# ── Raw network output — no multiplicative ansatz here ──────────────────────
u_outer = net(x_o, y_o)
u_inner = net(x_i, y_i)
u_bc = net(x_b, y_b)

# ── Three residuals: AD-PDE, FD-PDE, soft Dirichlet BC ─────────────────────
res_outer = (-u_outer.laplacian(x_o, y_o, scheme="automatic_differentiation") - 4.0).mse
res_inner = (-u_inner.laplacian(x_i, y_i, scheme="finite_difference:cotangent") - 4.0).mse
# u_exact = 1 − x² − y² → vanishes on the disk boundary, so the BC target is 0.
res_bc = (u_bc - 0.0).mse

crux = jno.core([res_outer, res_inner, res_bc], dom)
history = crux.solve(5000)

# ── Evaluate against the analytic solution on both interior regions ─────────
u_target_outer = 1.0 - x_o**2 - y_o**2
u_target_inner = 1.0 - x_i**2 - y_i**2
u_pred_outer, u_exact_outer, u_pred_inner, u_exact_inner = crux.eval([u_outer, u_target_outer, u_inner, u_target_inner])

rel_l2_outer = float(jnp.linalg.norm(u_pred_outer - u_exact_outer) / (jnp.linalg.norm(u_exact_outer) + 1e-8))
rel_l2_inner = float(jnp.linalg.norm(u_pred_inner - u_exact_inner) / (jnp.linalg.norm(u_exact_inner) + 1e-8))

# ── Write result to tracking file ──────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"02_elliptic/poisson_polygon_mixed_ops_soft_bc.py | epochs=5000 | "
        f"AD_outer_rel_L2={rel_l2_outer:.6e} | FD_inner_rel_L2={rel_l2_inner:.6e}\n"
    )

# Soft BC tolerates a wider band than hard BC — the BC residual competes
# with the PDE residuals during training. 15% threshold is roughly 1.5×
# the hard-BC tutorial's 10% to absorb this.
assert rel_l2_outer < 0.15, f"AD-outer rel L² too large: {rel_l2_outer:.3e}"
assert rel_l2_inner < 0.15, f"FD-inner rel L² too large: {rel_l2_inner:.3e}"
