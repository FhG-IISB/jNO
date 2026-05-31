"""06 — Boundary flux integrals and the divergence theorem

Equation:  −∇²u = 2π² sin(πx) sin(πy),   (x,y) ∈ [0,1]²,   u = 0 on ∂Ω
Exact:     u*(x,y) = sin(πx) sin(πy)

Trains the network with only the PDE residual, then verifies Gauss's
divergence theorem post-training:

    ∫_∂Ω ∇u · n dS  =  ∫_Ω Δu dΩ

Outward normals are accessed via domain.variable("boundary", normals=True),
which appends per-component outward-normal Variables after the spatial ones.
"""

import foundax
import jax
import jax.numpy as jnp
import optax

import jno
from jno import LearningRateSchedule as lrs

π = jno.np.pi

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05), compute_mesh_connectivity=True)

x, y, _ = domain.variable("interior")
x_b, y_b, _, nx, ny = domain.variable("boundary", normals=True)

# ── Forcing and exact solution ────────────────────────────────────────────────
forcing = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)

# ── Network ───────────────────────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(
        in_features=2,
        hidden_dims=32,
        num_layers=3,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.9, 3_000, 1e-5))

u = net(jno.np.concat([x, y], axis=-1)) * x * (1 - x) * y * (1 - y)
u_bnd = net(jno.np.concat([x_b, y_b], axis=-1)) * x_b * (1 - x_b) * y_b * (1 - y_b)

# ── PDE residual ──────────────────────────────────────────────────────────────
pde = -u.laplacian(x, y) - forcing

# ── Solve ─────────────────────────────────────────────────────────────────────
EPOCHS = 10_000
crux = jno.core([pde.mse], domain)
crux.solve(EPOCHS)

# ── Evaluate pointwise accuracy ───────────────────────────────────────────────
u_pred, u_ref = crux.eval([u, u_exact])
rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))
assert rel_l2 < 0.15, f"Pointwise rel. L2 error too large: {rel_l2:.3e}"

# ── Verify divergence theorem: ∫_∂Ω ∇u·n dS = ∫_Ω Δu dΩ ─────────────────────
vol = u.laplacian(x, y).integrate()
flux = (u_bnd.d(x_b) * nx + u_bnd.d(y_b) * ny).integrate()

vol_val, flux_val = crux.eval([vol, flux])
vol_s = float(jnp.squeeze(vol_val))
flux_s = float(jnp.squeeze(flux_val))

rel_disc = abs(vol_s - flux_s) / (abs(vol_s) + 1e-8)
assert rel_disc < 0.05, (
    f"Divergence theorem violated: ∫Δu={vol_s:.4f}, ∫∇u·n={flux_s:.4f}, relative discrepancy={rel_disc:.4f}"
)
