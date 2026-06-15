"""07 — Gradient and sensitivity analysis with u.grad(net)"""

import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.domain.line(mesh_size=0.001)
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

# ── Exact solution (for validation only, not used in training) ─────────────────
u_exact = jno.np.sin(π * x) / π**2

# ── Network with hard-enforced BCs ─────────────────────────────────────────────
u_net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=32,
        num_layers=3,
        key=jax.random.PRNGKey(0),
    )
).optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.5, end_value=1e-5)))

u = u_net(x)
u_xx = u.d2(x)
pde = -u_xx - jno.np.sin(π * x)  # residual — should be 0


ub = u_net(xb)

# ── In-training cosine similarity tracker ─────────────────────────────────────
# Build a boolean mask selecting only the output-layer weight matrix.
# This makes the Jacobian fast to compute — P_out_weight ≪ P_total.
all_false = jax.tree_util.tree_map(lambda _: False, u_net.params)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)

# Symbolic Jacobian restricted to the masked parameters.
# J shape at eval time: (N, P_out_weight)
J1 = pde.mse.grad(u_net.mask(output_mask))
J2 = ub.mse.grad(u_net.mask(output_mask))

cos_tracker = jno.np.dot(J1, J2).tracker(100)

# ── Solve ──────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, ub.mse, cos_tracker])
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jnp.linalg.norm(_u - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L² error: {rel_l2:.3e}")
assert rel_l2 < 1e-1, f"solution error too large: {rel_l2:.3e}"

# ── Post-training: gradient alignment between PDE and BC loss ─────────────────
# crux.eval([single_expr]) returns the raw array without a batch dimension.
[g_pde] = crux.eval([J1])  # (P_out_weight,) — gradient of PDE loss
[g_bc] = crux.eval([J2])  # (P_out_weight,) — gradient of BC  loss
cos_sim = float(jnp.dot(g_pde, g_bc) / (jnp.linalg.norm(g_pde) * jnp.linalg.norm(g_bc) + 1e-12))
print("\nGradient alignment (PDE vs BC, output-layer params)")
print(f"  dot(g_pde, g_bc) = {float(jnp.dot(g_pde, g_bc)):.4f}")
print(f"  cos_sim          = {cos_sim:.4f}")
if cos_sim > 0.5:
    print("  → Strongly aligned: PDE and BC losses reinforce each other.")
elif cos_sim > 0:
    print("  → Weakly aligned: compatible but partially independent.")
else:
    print("  → Conflict: PDE and BC losses are pulling parameters in opposite directions!")

assert cos_sim > -1.0, f"cos_sim out of range: {cos_sim:.4f}"

# ── Post-training: full Jacobian + Neural Tangent Kernel ──────────────────────
# Clear the output-layer mask so we get the full (N, P_total) Jacobian.
[J_full] = crux.eval([u.grad(u_net.mask(None))])  # (N, P_total)
N, P_total = J_full.shape
print(f"\nFull Jacobian  J  shape: {J_full.shape}  ({P_total} parameters)")

K = J_full @ J_full.T  # (N, N)
# Clip small negative eigenvalues (numerical noise from semi-definite K).
eigvals = jnp.maximum(jnp.sort(jnp.linalg.eigvalsh(K))[::-1], 0.0)

eff_rank = float(jnp.sum(eigvals) ** 2 / (jnp.sum(eigvals**2) + 1e-12))
cond = float(eigvals[0] / (eigvals[-1] + 1e-12))

print(f"\nNeural Tangent Kernel  K  ({N}×{N})")
print(f"  λ_max        = {float(eigvals[0]):.4f}")
print(f"  λ_min        = {float(eigvals[-1]):.4f}")
print(f"  Eff. rank    = {eff_rank:.2f}  (trace² / ‖K‖²_F)")
print(f"  Cond. number = {cond:.1f}")
