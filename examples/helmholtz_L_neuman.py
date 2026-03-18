import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import optax
import lineax as lx
import numpy as np
import matplotlib.pyplot as plt

import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs


pi = jnn.pi
k_val = 4.0


# ============================================================
# 1. Manufactured exact solution
# ============================================================
# u(x,y) = x y sin(pi x) cos(pi y)
#
# This gives:
#   u(0,y)=0    on left
#   u(x,0)=0    on bottom
#
# PDE:
#   -Δu - k^2 u = f
# Weak form:
#   ∫ (grad u · grad phi - k^2 u phi - f phi) dΩ - ∫ g phi dΓ = 0
# ============================================================
def exact_u(x, y):
    return x * y * jnn.sin(pi * x) * jnn.cos(pi * y)


def exact_u_num(x, y):
    return x * y * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)


def source_f(x, y):
    sx = jnn.sin(pi * x)
    cx = jnn.cos(pi * x)
    sy = jnn.sin(pi * y)
    cy = jnn.cos(pi * y)

    # Δu = 2*pi*y*cy*cx - 2*pi*x*sx*sy - 2*pi^2*x*y*sx*cy
    # f = -Δu - k^2 u
    return (
        -2.0 * pi * y * cy * cx
        + 2.0 * pi * x * sx * sy
        + (2.0 * pi**2 - k_val**2) * x * y * sx * cy
    )


# ============================================================
# 2. Exact Neumann fluxes on L-shape tags
# ============================================================
# L-shape with separate_boundary=True has:
#   left, bottom   -> Dirichlet
#   right_lower, inner_horizontal, inner_vertical, top -> Neumann
#
# outward normals for this geometry:
#   top               : (0,  1)
#   right_lower       : (1,  0)
#   inner_horizontal  : (0,  1)
#   inner_vertical    : (1,  0)
# ============================================================
def flux_top(x, y):
    # du/dy at y = 1
    # u_y = x sin(pi x) [cos(pi y) - pi y sin(pi y)]
    # at y=1 => -x sin(pi x)
    return -x * jnn.sin(pi * x)


def flux_right_lower(x, y):
    # du/dx at x = 1
    # u_x = y cos(pi y) [sin(pi x) + pi x cos(pi x)]
    # at x=1 => -pi y cos(pi y)
    return -pi * y * jnn.cos(pi * y)


def flux_inner_horizontal(x, y):
    # du/dy at y = 0.5
    # cos(pi/2)=0, sin(pi/2)=1
    # => -(pi/2) x sin(pi x)
    return -0.5 * pi * x * jnn.sin(pi * x)


def flux_inner_vertical(x, y):
    # du/dx at x = 0.5
    # sin(pi/2)=1, cos(pi/2)=0
    # => y cos(pi y)
    return y * jnn.cos(pi * y)


# ============================================================
# 3. Coarse training domain (L-shape)
# ============================================================
train_domain = jno.domain(constructor=jno.domain.l_shape(size=1.0, mesh_size=0.08, separate_boundary=True))

train_domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        train_domain.dirichlet("left", 0.0),
        train_domain.dirichlet("bottom", 0.0),
        train_domain.neumann(["right_lower", "inner_horizontal", "inner_vertical", "top"]),
    ],
    fem_solver=True,
)

u, phi = train_domain.fem_symbols()

xg, yg, _ = train_domain.variable("fem_gauss", split=True)
xr, yr, _ = train_domain.variable("gauss_right_lower", split=True)
xih, yih, _ = train_domain.variable("gauss_inner_horizontal", split=True)
xiv, yiv, _ = train_domain.variable("gauss_inner_vertical", split=True)
xt, yt, _ = train_domain.variable("gauss_top", split=True)

x_int, y_int, _ = train_domain.variable("interior", split=True)


# ============================================================
# 4. Neural network + hard BC ansatz
# ============================================================
key = jax.random.PRNGKey(0)

net = jnn.nn.mlp(2, hidden_dims=128,num_layers=4, activation=jax.nn.swish,key=key,)


def apply_hard_bc(u_pred, x, y):
    # left:   x=0 -> 0
    # bottom: y=0 -> 0
    return x * y * u_pred


u_gauss = apply_hard_bc(net(xg, yg), xg, yg)
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)


# ============================================================
# 5. Helmholtz weak form
# ============================================================
du_dx = jnn.grad(u, xg)
du_dy = jnn.grad(u, yg)
phi_x = jnn.grad(phi, xg)
phi_y = jnn.grad(phi, yg)

k_sq = 0.0 * xg + k_val**2

vol_integrand = ( du_dx * phi_x  + du_dy * phi_y - k_sq * u * phi - source_f(xg, yg) * phi)

# IMPORTANT:
# each boundary term must contain the correct boundary variables
neumann_right_lower = flux_right_lower(xr, yr) * phi
neumann_inner_horizontal = flux_inner_horizontal(xih, yih) * phi
neumann_inner_vertical = flux_inner_vertical(xiv, yiv) * phi
neumann_top = flux_top(xt, yt) * phi

weak = (
    vol_integrand
    - neumann_right_lower
    - neumann_inner_horizontal
    - neumann_inner_vertical
    - neumann_top
)


# ============================================================
# 6. Assemble coarse VPINN and coarse FEM
# ============================================================
pde = weak.assemble(train_domain, u_net=u_gauss, target="vpinn")
A_coarse, b_coarse = weak.assemble(train_domain, target="fem_system")

A_coarse_dense = jnp.asarray(A_coarse.toarray())
b_coarse_dense = jnp.asarray(b_coarse)

op = lx.MatrixLinearOperator(A_coarse_dense)
sol = lx.linear_solve(op, b_coarse_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem_coarse = sol.value.reshape(-1)

lin_res = jnp.linalg.norm(A_coarse_dense @ u_fem_coarse - b_coarse_dense) / (
    jnp.linalg.norm(b_coarse_dense) + 1e-14
)
print(f"Coarse FEM linear solve residual: {lin_res:.6e}")


# ============================================================
# 7. Train VPINN
# ============================================================
val_error = jnn.abs(u_int - exact_u(x_int, y_int))
error_tracker = jnn.tracker(val_error, interval=100)

crux = jno.core(constraints=[pde.mse, error_tracker], domain=train_domain)

learning_rate = lrs.warmup_cosine(4000, 200, 1e-3, 1e-5)
net.optimizer(optax.adam, lr=learning_rate)
crux.solve(epochs=4000)


# ============================================================
# 8. Fine FEM domain
# ============================================================
fem_domain = jno.domain(
    constructor=jno.domain.l_shape(size=1.0, mesh_size=0.03, separate_boundary=True)
)

fem_domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        fem_domain.dirichlet("left", 0.0),
        fem_domain.dirichlet("bottom", 0.0),
        fem_domain.neumann(["right_lower", "inner_horizontal", "inner_vertical", "top"]),
    ],
    fem_solver=True,
)

u_fem_sym, phi_fem_sym = fem_domain.fem_symbols()

xg_fem, yg_fem, _ = fem_domain.variable("fem_gauss", split=True)
xr_fem, yr_fem, _ = fem_domain.variable("gauss_right_lower", split=True)
xih_fem, yih_fem, _ = fem_domain.variable("gauss_inner_horizontal", split=True)
xiv_fem, yiv_fem, _ = fem_domain.variable("gauss_inner_vertical", split=True)
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

neumann_right_lower_fem = flux_right_lower(xr_fem, yr_fem) * phi_fem_sym
neumann_inner_horizontal_fem = flux_inner_horizontal(xih_fem, yih_fem) * phi_fem_sym
neumann_inner_vertical_fem = flux_inner_vertical(xiv_fem, yiv_fem) * phi_fem_sym
neumann_top_fem = flux_top(xt_fem, yt_fem) * phi_fem_sym

weak_fem = (
    vol_integrand_fem
    - neumann_right_lower_fem
    - neumann_inner_horizontal_fem
    - neumann_inner_vertical_fem
    - neumann_top_fem
)

A_fine, b_fine = weak_fem.assemble(fem_domain, target="fem_system")

A_fine_dense = jnp.asarray(A_fine.toarray())
b_fine_dense = jnp.asarray(b_fine)

op_fine = lx.MatrixLinearOperator(A_fine_dense)
sol_fine = lx.linear_solve(op_fine, b_fine_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem_fine = sol_fine.value.reshape(-1)

lin_res_fine = jnp.linalg.norm(A_fine_dense @ u_fem_fine - b_fine_dense) / (
    jnp.linalg.norm(b_fine_dense) + 1e-14
)
print(f"Fine FEM linear solve residual: {lin_res_fine:.6e}")

# ============================================================
# 9. Evaluate VPINN on fine domain
# ============================================================
x_eval, y_eval, _ = fem_domain.variable("interior", split=True)

u_vpinn_eval = crux.eval(
    apply_hard_bc(net(x_eval, y_eval), x_eval, y_eval),
    domain=fem_domain,
)
u_true_eval = crux.eval(
    exact_u(x_eval, y_eval),
    domain=fem_domain,
)

u_vpinn_eval = jnp.asarray(u_vpinn_eval).reshape(-1)
u_true_eval = jnp.asarray(u_true_eval).reshape(-1)

rel_l2_vpinn = jnp.linalg.norm(u_true_eval - u_vpinn_eval) / (jnp.linalg.norm(u_true_eval) + 1e-14)
max_abs_vpinn = jnp.max(jnp.abs(u_true_eval - u_vpinn_eval))

print(f"VPINN Relative L2 Error on fine domain: {rel_l2_vpinn:.6e}")
print(f"VPINN Max Abs Error on fine domain:     {max_abs_vpinn:.6e}")


# ============================================================
# 10. Fine FEM error
# ============================================================
coords_fem = np.asarray(fem_domain.mesh.points)[:, :2]
x_f = jnp.asarray(coords_fem[:, 0:1])
y_f = jnp.asarray(coords_fem[:, 1:2])

u_exact_fem = exact_u_num(x_f, y_f).reshape(-1)
u_fem_vec = jnp.asarray(u_fem_fine).reshape(-1)

rel_l2_fem = jnp.linalg.norm(u_exact_fem - u_fem_vec) / (jnp.linalg.norm(u_exact_fem) + 1e-14)
max_abs_fem = jnp.max(jnp.abs(u_exact_fem - u_fem_vec))

print(f"Fine FEM Relative L2 Error: {rel_l2_fem:.6e}")
print(f"Fine FEM Max Abs Error:     {max_abs_fem:.6e}")


# ============================================================
# 11. Plot
# ============================================================
# ============================================================
# 11. Plot
# ============================================================
tri_eval = fem_domain.mesh.cells_dict["triangle"]

# --- mesh-node coordinates for FEM/exact plots ---
coords_mesh = np.asarray(fem_domain.mesh.points)[:, :2]
x_mesh = coords_mesh[:, 0]
y_mesh = coords_mesh[:, 1]

# --- evaluate interior coordinates numerically for VPINN scatter plots ---
x_int_plot = np.asarray(crux.eval(x_eval, domain=fem_domain)).reshape(-1)
y_int_plot = np.asarray(crux.eval(y_eval, domain=fem_domain)).reshape(-1)

# --- values on mesh nodes ---
u_true_mesh_np = np.asarray(u_exact_fem).reshape(-1)
u_fem_np = np.asarray(u_fem_vec).reshape(-1)
abs_err_fem = np.abs(u_fem_np - u_true_mesh_np)

# --- values on interior sampled points ---
u_true_int_np = np.asarray(u_true_eval).reshape(-1)
u_vpinn_np = np.asarray(u_vpinn_eval).reshape(-1)
abs_err_vpinn = np.abs(u_vpinn_np - u_true_int_np)

print("mesh nodes:", len(x_mesh), len(y_mesh), len(u_true_mesh_np), len(u_fem_np))
print("interior points:", len(x_int_plot), len(y_int_plot), len(u_true_int_np), len(u_vpinn_np))

fig, axes = plt.subplots(1, 5, figsize=(24, 5))


def plot_mesh_field(ax, x, y, tri, values, title, cmap="viridis"):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    values = np.asarray(values).reshape(-1)

    im = ax.tripcolor(
        x,
        y,
        tri,
        values,
        shading="flat",
        cmap=cmap,
    )
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)


def plot_scatter_field(ax, x, y, values, title, cmap="viridis"):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    values = np.asarray(values).reshape(-1)

    assert len(x) == len(y) == len(values), (
        f"Scatter length mismatch: len(x)={len(x)}, len(y)={len(y)}, len(values)={len(values)}"
    )

    im = ax.scatter(
        x,
        y,
        c=values,
        s=12,
        cmap=cmap,
    )
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)


plot_mesh_field(axes[0], x_mesh, y_mesh, tri_eval, u_true_mesh_np, "True solution")
plot_scatter_field(axes[1], x_int_plot, y_int_plot, u_vpinn_np, "VPINN solution", cmap="viridis")
plot_scatter_field(axes[2], x_int_plot, y_int_plot, abs_err_vpinn, "VPINN abs error", cmap="magma")
plot_mesh_field(axes[3], x_mesh, y_mesh, tri_eval, u_fem_np, "Fine FEM solution")
plot_mesh_field(axes[4], x_mesh, y_mesh, tri_eval, abs_err_fem, "FEM abs error", cmap="magma")

plt.tight_layout()
plt.savefig("helmholtz_Lshape_manufactured.png", dpi=300)