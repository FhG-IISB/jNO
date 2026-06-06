"""06 — Boundary flux integrals and the divergence theorem

Problem
-------
    −∇²u(x,y) = f(x,y),   (x,y) ∈ [0,1]²,   u = 0 on ∂Ω

Analytical solution
-------------------
    u(x,y) = sin(πx) sin(πy)
    f(x,y) = 2π² sin(πx) sin(πy)

Divergence theorem check
------------------------
    ∫_∂Ω ∇u · n dS  =  ∫_Ω Δu dΩ  =  −2π² · 4/π²  =  −8

Outward normal Variables are obtained by passing ``normals=True`` to
``domain.variable()``.  The extra returned Variables are one per spatial
dimension and resolve to the outward unit normal components at the boundary
mesh nodes.  They are valid inside any ``.integrate()`` expression:

    x_b, y_b, _, nx, ny = domain.variable("boundary", normals=True)
    flux = (u.d(x_b) * nx + u.d(y_b) * ny).integrate()   # ∫_∂Ω ∇u · n dS

This is the 2-D analogue of the 1-D endpoint formula u'(1) − u'(0).

API demonstrated
----------------
    domain.variable(tag, normals=True)   — returns coordinate Variables followed
                                           by D outward-normal Variables.  Requires
                                           ``compute_mesh_connectivity=True``.
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05), compute_mesh_connectivity=True)

x, y, _ = domain.variable("interior")
# normals=True appends per-component outward-normal Variables after the spatial ones.
x_b, y_b, _, nx, ny = domain.variable("boundary", normals=True)
domain.summary()

# ── Analytical values ─────────────────────────────────────────────────────────
forcing = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
# Exact Δu = −2π² sin(πx)sin(πy);  ∫_Ω Δu dΩ = −8
LAPLACIAN_INTEGRAL = -8.0  # analytic: (−2π²) · (4/π²) = −8

# ── Network and hard-BC ansatz ────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(
        in_features=2,
        hidden_dims=32,
        num_layers=3,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 5_000, 0.9, end_value=1e-5)))

u = net(jno.np.concat([x, y], axis=-1)) * x * (1 - x) * y * (1 - y)  # hard Dirichlet BCs
u_bnd = net(jno.np.concat([x_b, y_b], axis=-1)) * x_b * (1 - x_b) * y_b * (1 - y_b)

# ── PDE residual ──────────────────────────────────────────────────────────────
pde = -u.laplacian(x, y) - forcing

# ── Solve ─────────────────────────────────────────────────────────────────────
EPOCHS = 10_000
crux = jno.core([pde.mse], domain).print_shapes()
history = crux.solve(EPOCHS)

# ── Post-training: divergence theorem verification ────────────────────────────
# Volume integral:   ∫_Ω Δu dΩ
# Boundary flux:     ∫_∂Ω ∇u · n dS   (should equal volume integral by Gauss)
volume_laplacian = u.laplacian(x, y).integrate()
boundary_flux = (u_bnd.d(x_b) * nx + u_bnd.d(y_b) * ny).integrate()

vol_val, flux_val = crux.eval([volume_laplacian, boundary_flux])
vol_s = float(jnp.squeeze(vol_val))
flux_s = float(jnp.squeeze(flux_val))

print(f"∫_Ω  Δu dΩ     = {vol_s:.4f}   (analytic: {LAPLACIAN_INTEGRAL:.4f})")
print(f"∫_∂Ω ∇u · n dS = {flux_s:.4f}   (should equal ∫Δu by Gauss's theorem)")
print(f"Relative discrepancy: {abs(vol_s - flux_s) / abs(vol_s):.4f}")

# ── Pointwise accuracy ────────────────────────────────────────────────────────
u_pred, u_ref = crux.eval([u, jno.np.sin(π * x) * jno.np.sin(π * y)])
rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))
print(f"Pointwise rel. L2 error: {rel_l2:.4e}   (exact: u = sin(πx)sin(πy))")

# ── Record result ─────────────────────────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f_out:
    f_out.write(
        f"06_integration/boundary_flux_divergence.py | epochs={EPOCHS}"
        f" | rel_L2={rel_l2:.6e}"
        f" | vol_laplacian={vol_s:.4f}"
        f" | boundary_flux={flux_s:.4f}\n"
    )

assert rel_l2 < 0.15, f"Relative L2 error too large: {rel_l2:.3e}"
assert abs(vol_s - flux_s) / abs(vol_s) < 0.05, (
    f"Divergence theorem: ∫Δu={vol_s:.4f}, ∫∇u·n={flux_s:.4f}, discrepancy={abs(vol_s - flux_s) / abs(vol_s):.4f}"
)
