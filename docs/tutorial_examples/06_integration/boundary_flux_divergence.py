"""06 — Boundary flux & the divergence theorem  (shapely + view-API dot product)"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax
from shapely.geometry import box

import jno

π = jno.np.pi
LAPLACIAN_INTEGRAL = -8.0  # analytic: (−2π²) · (4/π²) = −8

# --8<-- [start:setup]
domain = jno.domain(box(0, 0, 1, 1), mesh_size=0.05, compute_mesh_connectivity=True)
x, y, _ = domain.variable("interior")
x_b, y_b, _, nx, ny = domain.variable("boundary", normals=True, split=True)

forcing = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)
# --8<-- [end:setup]

# --8<-- [start:residual]
net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 1000, 0.5, end_value=1e-5)))

u = (net(jno.np.concat([x, y], axis=-1)) * x * (1 - x) * y * (1 - y)).scalar.bind(x=x, y=y)
u_bnd = (net(jno.np.concat([x_b, y_b], axis=-1)) * x_b * (1 - x_b) * y_b * (1 - y_b)).scalar.bind(x=x_b, y=y_b)
pde = -(u.xx + u.yy) - forcing
# --8<-- [end:residual]

# --8<-- [start:solve]
crux = jno.core([pde.mse])
crux.solve(5_000)
# --8<-- [end:solve]

# --8<-- [start:eval]
volume_laplacian = (u.xx + u.yy).integrate()
boundary_flux = jno.np.vector(u_bnd.x, u_bnd.y).dot(jno.np.vector(nx, ny)).integrate()  # ∫_∂Ω ∇u · n dS

vol_val, flux_val, u_pred, u_ref = crux.eval([volume_laplacian, boundary_flux, u, jno.np.sin(π * x) * jno.np.sin(π * y)])
vol_s = float(jnp.squeeze(vol_val))
flux_s = float(jnp.squeeze(flux_val))
rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))

print(f"∫_Ω  Δu dΩ     = {vol_s:.4f}   (analytic: {LAPLACIAN_INTEGRAL:.4f})")
print(f"∫_∂Ω ∇u · n dS = {flux_s:.4f}   (should equal ∫Δu by Gauss's theorem)")
print(f"Relative discrepancy:    {abs(vol_s - flux_s) / abs(vol_s):.4f}")
print(f"Pointwise rel. L2 error: {rel_l2:.4e}")
# --8<-- [end:eval]

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"06_integration/boundary_flux_divergence.py | epochs=5000"
        f" | rel_L2={rel_l2:.6e} | vol_laplacian={vol_s:.4f} | boundary_flux={flux_s:.4f}\n"
    )

# --8<-- [start:assert]
assert rel_l2 < 0.15, f"Relative L2 error too large: {rel_l2:.3e}"
assert abs(vol_s - flux_s) / abs(vol_s) < 0.05, (
    f"Divergence theorem: ∫Δu={vol_s:.4f}, ∫∇u·n={flux_s:.4f}, discrepancy={abs(vol_s - flux_s) / abs(vol_s):.4f}"
)
# --8<-- [end:assert]
