import jax
# Enable 64-bit precision for exact L-BFGS gradients
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs
"""
API usage is still crude and is to be improved after the integration with jax fem 
- definition of phi can be removed
"""


π = jnn.pi
sin = jnn.sin

# =============================================================================
# 1. Setup Domain & Initialize FEM 
# =============================================================================
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
domain.init_fem(element_type="TRI3", quad_degree=3, dirichlet_tags=["boundary"])

# 🛡️ THE TRUE SHIELD: Pad 2D+ FEM arrays so the compiler sets Batch Size B=1
# for k in ["dN_dx_flat", "N_flat", "JxW", "cells"]:
#     if k in domain.context:
#         domain.context[k] = domain.context[k][None, None, ...]

# Sample Quadrature Points
xg, yg, _ = domain.variable("fem_gauss", split=True)
x_int, y_int, _ = domain.variable("interior", split=True)

# =============================================================================
# 2. Define Neural Network & HARD Constraint
# =============================================================================
key = jax.random.PRNGKey(0)
u_net = jnn.nn.mlp(2, hidden_dims=64, num_layers=3, activation=jax.nn.swish, key=key)

def apply_hard_bc(u_pred, x_val, y_val):
    return u_pred * x_val * (1 - x_val) * y_val * (1 - y_val)

u_gauss = apply_hard_bc(u_net(xg, yg), xg, yg)
u_interior = apply_hard_bc(u_net(x_int, y_int), x_int, y_int)

def exact_u(x_val, y_val): return sin(π * x_val) * sin(π * y_val)
def source_f(x_val, y_val): return 2 * π**2 * sin(π * x_val) * sin(π * y_val)

# =============================================================================
# 3. Clean Symbolic Weak Form
# =============================================================================
phi = jnn.test()

# jnn.grad natively returns the correct (N_q, 1) Jacobian now!
du_dx = jnn.grad(u_gauss, xg)
du_dy = jnn.grad(u_gauss, yg)
f_gauss = source_f(xg, yg)

# Natively fetch shape gradients directly in the trace!
phi_x = jnn.grad(phi, xg)
phi_y = jnn.grad(phi, yg)

weak_pde = du_dx * phi_x + du_dy * phi_y - f_gauss * phi
residual = weak_pde.assemble(domain=domain, tag="fem_gauss")

# =============================================================================
# 4. Training Strategy (Adam Cosine → L-BFGS)
# =============================================================================
val_error = jnn.abs(u_interior - exact_u(x_int, y_int))
error_tracker = jnn.tracker(val_error, interval=50)
sse_loss = jnn.sum(jnn.square(residual))
# Compile normally!
crux = jno.core(constraints=[sse_loss, error_tracker], domain=domain)

# --- STAGE 1: Adam ---
learning_rate = lrs.warmup_cosine(2000, 200, 1e-3, 1e-4)
u_net.optimizer(optax.adam, lr=learning_rate)
crux.solve(epochs=2000)

# --- STAGE 2: L-BFGS ---
u_net.optimizer(optax.lbfgs)
crux.solve(epochs=1000)


# =============================================================================
# 6. Evaluation on Dense 0.01 Grid
# =============================================================================
eval_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))
x_eval, y_eval, _ = eval_domain.variable("interior", split=True)

u_pred_val = crux.eval(apply_hard_bc(u_net(x_eval, y_eval), x_eval, y_eval), domain=eval_domain)
u_true_val = crux.eval(exact_u(x_eval, y_eval), domain=eval_domain)

l2_error = jnp.linalg.norm(u_true_val - u_pred_val) / jnp.linalg.norm(u_true_val)
print(f"\nFinal Relative L2 Error: {l2_error:.4e}")