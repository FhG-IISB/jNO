import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs
import lineax as lx

pi = jnn.pi
sin = jnn.sin

# ============================================================
# 1. Domain + FEM/variational init
# ============================================================
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))

# fem_solver=True is needed if you also want A,b from target="fem_system"
domain.init_fem(element_type="TRI3", quad_degree=3, dirichlet_tags=["boundary"],  neumann_tags=[], fem_solver=True,)
# Generic variational symbols
u, phi = domain.fem_symbols()

# Variational sample points
xg, yg, _ = domain.variable("fem_gauss", split=True)

# Interior points for validation
x_int, y_int, _ = domain.variable("interior", split=True)

# ============================================================
# 2. Neural network + hard BC
# ============================================================
key = jax.random.PRNGKey(0)
net = jnn.nn.mlp(2,hidden_dims=64, num_layers=3, activation=jax.nn.swish, key=key,)

def apply_hard_bc(u_pred, x, y): return u_pred * x * (1.0 - x) * y * (1.0 - y)

def exact_u(x, y): return sin(pi * x) * sin(pi * y)

def source_f(x, y): return 2.0 * pi**2 * sin(pi * x) * sin(pi * y)

# This is the VPINN realization you want to substitute in ONLY for target="vpinn"
u_gauss = apply_hard_bc(net(xg, yg), xg, yg)

# For validation at interior points
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)

# ============================================================
# 3. One symbolic weak form
# ============================================================
du_dx = jnn.grad(u, xg)
du_dy = jnn.grad(u, yg)

phi_x = jnn.grad(phi, xg)
phi_y = jnn.grad(phi, yg)

f_gauss = source_f(xg, yg)

weak = du_dx * phi_x + du_dy * phi_y - f_gauss * phi


# ============================================================
# 4. Assemble BOTH targets from the SAME weak form
# ============================================================
# VPINN path: symbolic u should be replaced internally by u_gauss
pde = weak.assemble(domain, u_net=u_gauss, target="vpinn")

# FEM path: symbolic u should stay TrialFunction, so A,b are assembled
A, b = weak.assemble(domain, target="fem_system")

print("VPINN assembled object:", pde)
print("A shape:", A.shape)
print("b shape:", b.shape)
A_dense = jnp.asarray(A.toarray())
b_dense = jnp.asarray(b)


op = lx.MatrixLinearOperator(A_dense)
sol = lx.linear_solve(op, b_dense,  solver=lx.AutoLinearSolver(well_posed=True),)
u_fem = sol.value
lin_res = jnp.linalg.norm(A_dense @ u_fem - b_dense) / jnp.linalg.norm(b_dense)
print(f"Linear solve residual: {lin_res:.6e}")
print("u_fem shape:", u_fem.shape)
# ============================================================
# 5. Loss + tracker
# ============================================================
val_error = jnn.abs(u_int - exact_u(x_int, y_int))
error_tracker = jnn.tracker(val_error, interval=50)

crux = jno.core(constraints=[pde.mse, error_tracker], domain=domain)

# ============================================================
# 6. Training
# ============================================================
learning_rate = lrs.warmup_cosine(4000, 200, 1e-3, 1e-5)
net.optimizer(optax.adam, lr=learning_rate)
crux.solve(epochs=4000)


# ============================================================
# 7. Dense evaluation
# ============================================================
eval_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))
x_eval, y_eval, _ = eval_domain.variable("interior", split=True)

u_pred_val = crux.eval(apply_hard_bc(net(x_eval, y_eval), x_eval, y_eval), domain=eval_domain)
u_true_val = crux.eval(exact_u(x_eval, y_eval), domain=eval_domain)

rel_l2 = jnp.linalg.norm(u_true_val - u_pred_val) / jnp.linalg.norm(u_true_val)
print(f"Final Relative L2 Error: {rel_l2:.6e}")

# ------------------------------------------------------------
# 7b. Compare FEM nodal solution against exact solution
# ------------------------------------------------------------
def exact_u_num(x, y):
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
coords_fe = domain.mesh.points
x_fe = jnp.asarray(coords_fe[:, 0:1])
y_fe = jnp.asarray(coords_fe[:, 1:2])

u_exact_fe = exact_u_num(x_fe, y_fe).reshape(-1)
u_fem_vec = jnp.asarray(u_fem).reshape(-1)

rel_l2_fem = jnp.linalg.norm(u_exact_fe - u_fem_vec) / jnp.linalg.norm(u_exact_fe)
abs_err_fem = jnp.abs(u_exact_fe - u_fem_vec)

print(f"FEM Relative L2 Error (nodal): {rel_l2_fem:.6e}")
print(f"FEM Max Abs Error (nodal): {jnp.max(abs_err_fem):.6e}")
print(f"VPINN Relative L2 Error: {rel_l2:.6e}")
print(f"FEM   Relative L2 Error: {rel_l2_fem:.6e}")

# ============================================================
# 8. Plot
# ============================================================
# Dense evaluation mesh (for exact + VPINN)
coords_eval = eval_domain.mesh.points
x_eval_pts = coords_eval[:, 0]
y_eval_pts = coords_eval[:, 1]

# FEM mesh
coords_fe = domain.mesh.points
x_fe_pts = coords_fe[:, 0]
y_fe_pts = coords_fe[:, 1]

abs_error_vpinn = jnp.abs(u_true_val - u_pred_val)

fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharex=False, sharey=False)

def plot_component(ax, x, y, triangles, val, title, cmap="viridis"):
    v = np.array(val).reshape(-1)
    im = ax.tripcolor(x, y, triangles, v, cmap=cmap, shading="flat")
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

# Eval mesh triangles
tri_eval = eval_domain.mesh.cells_dict["triangle"]

# FEM mesh triangles
tri_fe = domain.mesh.cells_dict["triangle"]

plot_component(axes[0], x_eval_pts, y_eval_pts, tri_eval, u_true_val, "Exact solution")
plot_component(axes[1], x_eval_pts, y_eval_pts, tri_eval, u_pred_val, "VPINN prediction")
plot_component(axes[2], x_eval_pts, y_eval_pts, tri_eval, abs_error_vpinn, "VPINN absolute error", cmap="magma")
plot_component(axes[3], x_fe_pts, y_fe_pts, tri_fe, u_fem, "FEM solution")

plt.tight_layout()
plt.savefig("poisson_new_api_same_weak_form_with_fem.png", dpi=300)
