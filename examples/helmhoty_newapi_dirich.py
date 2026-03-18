import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import optax
import lineax as lx
import numpy as np

import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs

pi = jnn.pi
sin = jnn.sin

k_val = 4.0

def exact_u(x, y):
    return sin(pi * x) * sin(pi * y)

def source_f(x, y):
    return (2.0 * pi**2 - k_val**2) * sin(pi * x) * sin(pi * y)

def exact_u_num(x, y):
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


# ============================================================
# 1. COARSE DOMAIN FOR VPINN
# ============================================================
train_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
train_domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        train_domain.dirichlet(["left", "right", "bottom", "top"], 0.0),
    ],
    fem_solver=True,
)

u, phi = train_domain.fem_symbols()
xg, yg, _ = train_domain.variable("fem_gauss", split=True)
x_int, y_int, _ = train_domain.variable("interior", split=True)

key = jax.random.PRNGKey(0)
net = jnn.nn.mlp(
    2,
    hidden_dims=128,
    num_layers=4,
    activation=jax.nn.tanh,
    key=key,
)

def apply_hard_bc(u_pred, x, y):
    return x * (1.0 - x) * y * (1.0 - y) * u_pred

u_gauss = apply_hard_bc(net(xg, yg), xg, yg)
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)

du_dx = jnn.grad(u, xg)
du_dy = jnn.grad(u, yg)
phi_x = jnn.grad(phi, xg)
phi_y = jnn.grad(phi, yg)

k_sq = 0.0 * xg + k_val**2
#k_sq_int = 0.0 * x_int + k_val**2
vol_integrand = (
    du_dx * phi_x
    + du_dy * phi_y
    - k_sq * u * phi
    - source_f(xg, yg) * phi
)

weak_train = vol_integrand

pde = weak_train.assemble(train_domain, u_net=u_gauss, target="vpinn")

#strong_pde = -jnn.laplacian(u_int, [x_int, y_int], scheme="automatic_differentiation") - k_sq_int * u_int - source_f(x_int, y_int)
val_error = jnn.abs(u_int - exact_u(x_int, y_int))
error_tracker = jnn.tracker(val_error, interval=100)

#crux = jno.core(constraints=[strong_pde.mse, error_tracker], domain=train_domain)
crux = jno.core(constraints=[pde.mse, error_tracker], domain=train_domain)

learning_rate = lrs.warmup_cosine(4000, 200, 1e-3, 1e-5)
net.optimizer(optax.adam, lr=learning_rate)
crux.solve(epochs=4000)

# net.optimizer(optax.lbfgs(1e-3))
# crux.solve(epochs=1000)


# ============================================================
# 2. FINE DOMAIN FOR FEM
# ============================================================
fem_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))
fem_domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        fem_domain.dirichlet(["left", "right", "bottom", "top"], 0.0),
    ],
    fem_solver=True,
)

u_fem_sym, phi_fem_sym = fem_domain.fem_symbols()
xg_fem, yg_fem, _ = fem_domain.variable("fem_gauss", split=True)

du_dx_fem = jnn.grad(u_fem_sym, xg_fem)
du_dy_fem = jnn.grad(u_fem_sym, yg_fem)
phi_x_fem = jnn.grad(phi_fem_sym, xg_fem)
phi_y_fem = jnn.grad(phi_fem_sym, yg_fem)

k_sq_fem = 0.0 * xg_fem + k_val**2

vol_integrand_fem = (
    du_dx_fem * phi_x_fem
    + du_dy_fem * phi_y_fem
    - k_sq_fem * u_fem_sym * phi_fem_sym
    - source_f(xg_fem, yg_fem) * phi_fem_sym
)

weak_fem = vol_integrand_fem
A, b = weak_fem.assemble(fem_domain, target="fem_system")

A_dense = jnp.asarray(A.toarray())
b_dense = jnp.asarray(b)

op = lx.MatrixLinearOperator(A_dense)
sol = lx.linear_solve(op, b_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem = sol.value.reshape(-1)

lin_res = jnp.linalg.norm(A_dense @ u_fem - b_dense) / (jnp.linalg.norm(b_dense) + 1e-14)
print(f"Fine FEM linear solve residual: {lin_res:.6e}")


# ============================================================
# 3. EVALUATE VPINN ON THE SAME FINE DOMAIN
# ============================================================
x_eval, y_eval, _ = fem_domain.variable("interior", split=True)

u_vpinn_eval = crux.eval(apply_hard_bc(net(x_eval, y_eval), x_eval, y_eval), domain=fem_domain)
u_true_eval = crux.eval(exact_u(x_eval, y_eval), domain=fem_domain)

rel_l2_vpinn = jnp.linalg.norm(u_true_eval - u_vpinn_eval) / jnp.linalg.norm(u_true_eval)
print(f"VPINN Relative L2 Error on fine domain: {rel_l2_vpinn:.6e}")


# ============================================================
# 4. FEM ERROR ON THE SAME FINE DOMAIN
# ============================================================
coords_fem = np.asarray(fem_domain.mesh.points)[:, :2]
x_f = jnp.asarray(coords_fem[:, 0:1])
y_f = jnp.asarray(coords_fem[:, 1:2])

u_exact_fem = exact_u_num(x_f, y_f).reshape(-1)
u_fem_vec = jnp.asarray(u_fem).reshape(-1)

rel_l2_fem = jnp.linalg.norm(u_exact_fem - u_fem_vec) / jnp.linalg.norm(u_exact_fem)
max_abs_fem = jnp.max(jnp.abs(u_exact_fem - u_fem_vec))

print(f"Fine FEM Relative L2 Error: {rel_l2_fem:.6e}")
print(f"Fine FEM Max Abs Error:     {max_abs_fem:.6e}")
import matplotlib.pyplot as plt

coords_eval = np.asarray(fem_domain.mesh.points)[:, :2]
x_eval_pts = coords_eval[:, 0]
y_eval_pts = coords_eval[:, 1]
tri_eval = fem_domain.mesh.cells_dict["triangle"]

u_true_np = np.asarray(u_true_eval).reshape(-1)
u_vpinn_np = np.asarray(u_vpinn_eval).reshape(-1)
u_fem_np = np.asarray(u_fem).reshape(-1)

abs_err_vpinn = np.abs(u_vpinn_np - u_true_np)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

def plot_field(ax, values, title, cmap="viridis"):
    im = ax.tripcolor(
        x_eval_pts,
        y_eval_pts,
        tri_eval,
        values,
        shading="flat",
        cmap=cmap,
    )
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

plot_field(axes[0], u_true_np, "True solution")
plot_field(axes[1], u_vpinn_np, "VPINN solution")
plot_field(axes[2], abs_err_vpinn, "VPINN absolute error", cmap="magma")
plot_field(axes[3], u_fem_np, "FEM solution")

plt.tight_layout()
plt.savefig("helmholtz_true_vpinn_error_fem_solution.png", dpi=300)