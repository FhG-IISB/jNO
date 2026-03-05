import jax
import jax.numpy as jnp
import optax
import jno
import jno.numpy as jnn
import matplotlib.pyplot as plt
"""
API usage is still crude and is to be improved after the integration with jax fem 
- the fem _gauss tag can be removed
- definition of phi can be removed
"""
π = jnn.pi

# =============================================================================
# 1. Setup Domain & Initialize FEM
# =============================================================================
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.1))
domain.init_fem(element_type="TRI3", quad_degree=2, dirichlet_tags=["boundary"])

# Sample Quadrature Points 
x, y, _ = domain.variable("fem_gauss", split=True)

# =============================================================================
# 2. Define Neural Network & HARD Constraint
# =============================================================================
key = jax.random.PRNGKey(0)
u_net = jnn.nn.mlp(2, hidden_dims=64, num_layers=3, key=key)

# EXACT Boundary Ansatz (Strictly satisfies Dirichlet boundaries)
u = u_net(x, y) * x * (1 - x) * y * (1 - y)

# =============================================================================
# 3. Symbolic Weak Form
# =============================================================================
f = 2 * π**2 * jnn.sin(π * x) * jnn.sin(π * y)
phi = jno.TestFunction(tag="fem_gauss")

# \int (grad(u) * grad(phi) - f * phi) dx
weak_pde = (
    jnn.grad(u, x) * jnn.grad(phi, x) + 
    jnn.grad(u, y) * jnn.grad(phi, y) - 
    f * phi
)

# Assemble the global residual vector
residual = weak_pde.assemble(domain=domain, tag="fem_gauss")

# =============================================================================
# 4. Solve!
# =============================================================================
# No soft constraints needed, just the PDE residual!
crux = jno.core(constraints=[residual.mse], domain=domain)

u_net.optimizer(optax.adam, lr=1e-3)
crux.solve(epochs=2000)

print("\nVPINN Training Complete!")

# =============================================================================
# 5. Evaluation and Visualization
# =============================================================================
print("Evaluating solutions on a dense grid...")

eval_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.02))
x_eval, y_eval, _ = eval_domain.variable("interior", split=True)

# Must apply the exact same ansatz for evaluation!
u_pred_expr = u_net(x_eval, y_eval) * x_eval * (1 - x_eval) * y_eval * (1 - y_eval)
u_true_expr = jnn.sin(π * x_eval) * jnn.sin(π * y_eval)

u_pred_val = crux.eval(u_pred_expr, domain=eval_domain).flatten()
u_true_val = crux.eval(u_true_expr, domain=eval_domain).flatten()
X_coords = crux.eval(x_eval, domain=eval_domain).flatten()
Y_coords = crux.eval(y_eval, domain=eval_domain).flatten()

abs_error = jnp.abs(u_true_val - u_pred_val)
l2_error = jnp.linalg.norm(abs_error) / jnp.linalg.norm(u_true_val)
print(f"Final Relative L2 Error: {l2_error:.4e}")

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(16, 4))


sc0 = axes[0].tricontourf(X_coords, Y_coords, u_true_val, levels=50, cmap='viridis')
axes[0].set_title("True Solution")
axes[0].set_aspect('equal')
fig.colorbar(sc0, ax=axes[0])

sc1 = axes[1].tricontourf(X_coords, Y_coords, u_pred_val, levels=50, cmap='viridis')
axes[1].set_title("Predicted Solution (VPINN)")
axes[1].set_aspect('equal')
fig.colorbar(sc1, ax=axes[1])

sc2 = axes[2].tricontourf(X_coords, Y_coords, abs_error, levels=50, cmap='magma')
axes[2].set_title(f"Absolute Error (L2: {l2_error:.2e})")
axes[2].set_aspect('equal')
fig.colorbar(sc2, ax=axes[2])

plt.tight_layout()
plt.savefig("vpinn_poisson_hard_constraint.png", dpi=300)
print("Saved plot to vpinn_poisson_hard_constraint.png")