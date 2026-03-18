import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
import lineax as lx

import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs

pi = jnn.pi
sin = jnn.sin
cos = jnn.cos


# ============================================================
# 1. Problem setup
# ============================================================
k_val = 4.0

def exact_u(x, y):
    return sin(pi * x) * (cos(pi * y) + y)

def exact_u_num(x, y):
    return jnp.sin(jnp.pi * x) * (jnp.cos(jnp.pi * y) + y)

def source_f(x, y):
    return (
        pi**2 * sin(pi * x) * (2.0 * cos(pi * y) + y)
        - (k_val**2) * sin(pi * x) * (cos(pi * y) + y)
    )

def exact_flux_right(x, y):
    # u_x = pi cos(pi x) (cos(pi y) + y)
    # right boundary x=1 -> -pi (cos(pi y) + y)
    return -pi * (cos(pi * y) + y)

def exact_flux_top(x, y):
    # u_y = sin(pi x)(-pi sin(pi y) + 1)
    # top boundary y=1 -> sin(pi x)
    return sin(pi * x)


# ============================================================
# 2. Training domain (coarse)
# ============================================================
train_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))

train_domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        train_domain.dirichlet("left", 0.0),
        train_domain.dirichlet("bottom", lambda p: jnp.sin(jnp.pi * p[0])),
        train_domain.neumann(["right", "top"]),
    ],
    fem_solver=True,
)

u, phi = train_domain.fem_symbols()

xg, yg, _ = train_domain.variable("fem_gauss", split=True)
xr, yr, _ = train_domain.variable("gauss_right", split=True)
xt, yt, _ = train_domain.variable("gauss_top", split=True)
x_int, y_int, _ = train_domain.variable("interior", split=True)


# ============================================================
# 3. Network + hard BC
# left  : u=0
# bottom: u=sin(pi x)
# ============================================================
key = jax.random.PRNGKey(0)

net = jnn.nn.mlp(
    2,
    hidden_dims=128,
    num_layers=4,
    activation=jax.nn.tanh,
    key=key,
)

def apply_hard_bc(u_pred, x, y):
    # left satisfied by factor x
    # bottom satisfied because y=0 kills correction term
    return sin(pi * x) + x * y * u_pred

u_gauss = apply_hard_bc(net(xg, yg), xg, yg)
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)


# ============================================================
# 4. Mixed Helmholtz weak form
# ∫ (grad u·grad phi - k^2 u phi - f phi) dΩ - ∫ g phi dΓ = 0
# ============================================================
du_dx = jnn.grad(u, xg)
du_dy = jnn.grad(u, yg)
phi_x = jnn.grad(phi, xg)
phi_y = jnn.grad(phi, yg)

# IMPORTANT: anchor k^2 to volume bucket
k_sq = 0.0 * xg + k_val**2

vol_integrand = (
    du_dx * phi_x
    + du_dy * phi_y
    - k_sq * u * phi
    - source_f(xg, yg) * phi
)

# IMPORTANT: Neumann terms must contain the correct boundary variables
neumann_right = exact_flux_right(xr, yr) * phi
neumann_top = exact_flux_top(xt, yt) * phi

weak = vol_integrand - neumann_right - neumann_top


# ============================================================
# 5. Assemble VPINN and FEM on coarse domain
# ============================================================
pde = weak.assemble(train_domain, u_net=u_gauss, target="vpinn")
A_coarse, b_coarse = weak.assemble(train_domain, target="fem_system")

A_coarse_dense = jnp.asarray(A_coarse.toarray())
b_coarse_dense = jnp.asarray(b_coarse)

op = lx.MatrixLinearOperator(A_coarse_dense)
sol = lx.linear_solve(op, b_coarse_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem_coarse = sol.value.reshape(-1)

lin_res = jnp.linalg.norm(A_coarse_dense @ u_fem_coarse - b_coarse_dense) / (jnp.linalg.norm(b_coarse_dense) + 1e-14)
print(f"Coarse FEM linear solve residual: {lin_res:.6e}")


# ============================================================
# 6. Train VPINN
# ============================================================
val_error = jnn.abs(u_int - exact_u(x_int, y_int))
error_tracker = jnn.tracker(val_error, interval=100)

crux = jno.core(constraints=[pde.mse, error_tracker], domain=train_domain)

learning_rate = lrs.warmup_cosine(4000, 200, 1e-3, 1e-5)
net.optimizer(optax.adam, lr=learning_rate)
crux.solve(epochs=4000)

# net.optimizer(optax.lbfgs(1e-3))
# crux.solve(epochs=1000)


# ============================================================
# 7. Fine FEM domain for reference solve
# ============================================================
fem_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))

fem_domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        fem_domain.dirichlet("left", 0.0),
        fem_domain.dirichlet("bottom", lambda p: jnp.sin(jnp.pi * p[0])),
        fem_domain.neumann(["right", "top"]),
    ],
    fem_solver=True,
)

u_fem_sym, phi_fem_sym = fem_domain.fem_symbols()
xg_fem, yg_fem, _ = fem_domain.variable("fem_gauss", split=True)
xr_fem, yr_fem, _ = fem_domain.variable("gauss_right", split=True)
xt_fem, yt_fem, _ = fem_domain.variable("gauss_top", split=True)

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

neumann_right_fem = exact_flux_right(xr_fem, yr_fem) * phi_fem_sym
neumann_top_fem = exact_flux_top(xt_fem, yt_fem) * phi_fem_sym

weak_fem = vol_integrand_fem - neumann_right_fem - neumann_top_fem

A_fine, b_fine = weak_fem.assemble(fem_domain, target="fem_system")

A_fine_dense = jnp.asarray(A_fine.toarray())
b_fine_dense = jnp.asarray(b_fine)

op_fine = lx.MatrixLinearOperator(A_fine_dense)
sol_fine = lx.linear_solve(op_fine, b_fine_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem_fine = sol_fine.value.reshape(-1)

lin_res_fine = jnp.linalg.norm(A_fine_dense @ u_fem_fine - b_fine_dense) / (jnp.linalg.norm(b_fine_dense) + 1e-14)
print(f"Fine FEM linear solve residual: {lin_res_fine:.6e}")


# ============================================================
# 8. Evaluate VPINN on fine domain
# ============================================================
x_eval, y_eval, _ = fem_domain.variable("interior", split=True)

u_vpinn_eval = crux.eval(apply_hard_bc(net(x_eval, y_eval), x_eval, y_eval), domain=fem_domain)
u_true_eval = crux.eval(exact_u(x_eval, y_eval), domain=fem_domain)

rel_l2_vpinn = jnp.linalg.norm(u_true_eval - u_vpinn_eval) / jnp.linalg.norm(u_true_eval)
print(f"VPINN Relative L2 Error on fine domain: {rel_l2_vpinn:.6e}")


# ============================================================
# 9. Fine FEM error
# ============================================================
coords_fem = np.asarray(fem_domain.mesh.points)[:, :2]
x_f = jnp.asarray(coords_fem[:, 0:1])
y_f = jnp.asarray(coords_fem[:, 1:2])

u_exact_fem = exact_u_num(x_f, y_f).reshape(-1)
u_fem_vec = jnp.asarray(u_fem_fine).reshape(-1)

rel_l2_fem = jnp.linalg.norm(u_exact_fem - u_fem_vec) / jnp.linalg.norm(u_exact_fem)
max_abs_fem = jnp.max(jnp.abs(u_exact_fem - u_fem_vec))

print(f"Fine FEM Relative L2 Error: {rel_l2_fem:.6e}")
print(f"Fine FEM Max Abs Error:     {max_abs_fem:.6e}")


# ============================================================
# 10. Plot true, VPINN abs error, FEM solution
# ============================================================
coords_eval = np.asarray(fem_domain.mesh.points)[:, :2]
x_eval_pts = coords_eval[:, 0]
y_eval_pts = coords_eval[:, 1]
tri_eval = fem_domain.mesh.cells_dict["triangle"]

u_true_np = np.asarray(u_true_eval).reshape(-1)
u_vpinn_np = np.asarray(u_vpinn_eval).reshape(-1)
u_fem_np = np.asarray(u_fem_fine).reshape(-1)

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
plt.savefig("helmholtz_QUAD4_mixed_true_vpinn_error_fem.png", dpi=300)
plt.show()