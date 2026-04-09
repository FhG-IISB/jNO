import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import lineax as lx
import optax

import jno

import numpy as np
from jno import LearningRateSchedule as lrs

"""02 - 2-D Helmholtz equation with FEM and variational PINNs

Problem
-------
    -Delta u - k^2 u = f    in [0, 1]^2

Boundary conditions
-------------------
    u = 0                  on x = 0
    u = sin(pi x)          on y = 0
    du/dn = g              on x = 1 and y = 1

Analytical solution
-------------------
    u(x, y) = sin(pi x) (cos(pi y) + y)
"""
pi = jno.np.pi
sin = jno.np.sin
cos = jno.np.cos
k_val = 4.0


# -----------------------------------------------------------------------------
# Manufactured solution
# -----------------------------------------------------------------------------
def exact_u(x, y):
    return sin(pi * x) * (cos(pi * y) + y)


def exact_u_num(x, y):
    return jnp.sin(jnp.pi * x) * (jnp.cos(jnp.pi * y) + y)


def source_f(x, y):
    return pi**2 * sin(pi * x) * (2.0 * cos(pi * y) + y) - (k_val**2) * sin(pi * x) * (cos(pi * y) + y)


def exact_flux_right(x, y):
    return -pi * (cos(pi * y) + y)


def exact_flux_top(x, y):
    return sin(pi * x)


# -----------------------------------------------------------------------------
# Training domain (coarse)
# -----------------------------------------------------------------------------
train_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.22))
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
# -----------------------------------------------------------------------------
# Neural network with hard Dirichlet BCs
# -----------------------------------------------------------------------------
net = jno.np.nn.mlp(
    2,
    hidden_dims=32,
    num_layers=4,
    activation=jax.nn.tanh,
    key=jax.random.PRNGKey(0),
)


def apply_hard_bc(u_pred, x, y):
    return sin(pi * x) + x * y * u_pred


u_gauss = apply_hard_bc(net(xg, yg), xg, yg)
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)

# -----------------------------------------------------------------------------
# Weak form and Vartiational PINN Training
# -----------------------------------------------------------------------------
du_dx = jno.np.grad(u, xg)
du_dy = jno.np.grad(u, yg)
phi_x = jno.np.grad(phi, xg)
phi_y = jno.np.grad(phi, yg)

vol_integrand = du_dx * phi_x + du_dy * phi_y - (k_val**2) * u * phi - source_f(xg, yg) * phi

neumann_right = exact_flux_right(xr, yr) * phi
neumann_top = exact_flux_top(xt, yt) * phi
weak = vol_integrand - neumann_right - neumann_top

pde = weak.assemble(train_domain, u_net=u_gauss, target="vpinn")
crux = jno.core(constraints=[pde.mse], domain=train_domain)

net.optimizer(optax.adam, lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))
crux.solve(epochs=10)
# -----------------------------------------------------------------------------
# Reference FEM solve on a finer mesh
# -----------------------------------------------------------------------------
fem_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.22))
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

du_dx_fem = jno.np.grad(u_fem_sym, xg_fem)
du_dy_fem = jno.np.grad(u_fem_sym, yg_fem)
phi_x_fem = jno.np.grad(phi_fem_sym, xg_fem)
phi_y_fem = jno.np.grad(phi_fem_sym, yg_fem)
k_sq_fem = 0.0 * xg_fem + k_val**2

vol_integrand_fem = du_dx_fem * phi_x_fem + du_dy_fem * phi_y_fem - k_sq_fem * u_fem_sym * phi_fem_sym - source_f(xg_fem, yg_fem) * phi_fem_sym

neumann_right_fem = exact_flux_right(xr_fem, yr_fem) * phi_fem_sym
neumann_top_fem = exact_flux_top(xt_fem, yt_fem) * phi_fem_sym
weak_fem = vol_integrand_fem - neumann_right_fem - neumann_top_fem

A_fine, b_fine = weak_fem.assemble(fem_domain, target="fem_system")
A_fine_dense = jnp.asarray(A_fine.todense())
b_fine_dense = jnp.asarray(b_fine)

fine_op = lx.MatrixLinearOperator(A_fine_dense)
fine_sol = lx.linear_solve(fine_op, b_fine_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem_fine = fine_sol.value.reshape(-1)

lin_res_fine = jnp.linalg.norm(A_fine_dense @ u_fem_fine - b_fine_dense) / (jnp.linalg.norm(b_fine_dense) + 1e-14)
print(f"Fine FEM linear solve residual: {lin_res_fine:.6e}")
# -----------------------------------------------------------------------------
# Compare VPINN and FEM on the fine domain
# -----------------------------------------------------------------------------
x_eval, y_eval, _ = fem_domain.variable("interior", split=True)
u_vpinn_eval = crux.eval(apply_hard_bc(net(x_eval, y_eval), x_eval, y_eval), domain=fem_domain)
u_true_eval = crux.eval(exact_u(x_eval, y_eval), domain=fem_domain)

u_vpinn_eval = jnp.asarray(u_vpinn_eval).reshape(-1)
u_true_eval = jnp.asarray(u_true_eval).reshape(-1)

rel_l2_vpinn = jnp.linalg.norm(u_true_eval - u_vpinn_eval) / (jnp.linalg.norm(u_true_eval) + 1e-14)
max_abs_vpinn = jnp.max(jnp.abs(u_true_eval - u_vpinn_eval))
print(f"VPINN Relative L2 Error on fine domain: {rel_l2_vpinn:.6e}")
print(f"VPINN Max Abs Error on fine domain:     {max_abs_vpinn:.6e}")

coords_fem = np.asarray(fem_domain.mesh.points)[:, :2]
x_f = jnp.asarray(coords_fem[:, 0:1])
y_f = jnp.asarray(coords_fem[:, 1:2])
u_exact_fem = exact_u_num(x_f, y_f).reshape(-1)
u_fem_vec = jnp.asarray(u_fem_fine).reshape(-1)

rel_l2_fem = jnp.linalg.norm(u_exact_fem - u_fem_vec) / (jnp.linalg.norm(u_exact_fem) + 1e-14)
max_abs_fem = jnp.max(jnp.abs(u_exact_fem - u_fem_vec))
print(f"Fine FEM Relative L2 Error: {rel_l2_fem:.6e}")
print(f"Fine FEM Max Abs Error:     {max_abs_fem:.6e}")
assert float(rel_l2_vpinn) < 1.1, f"VPINN relative L2 error too large: {float(rel_l2_vpinn):.3e}"
assert float(rel_l2_fem) < 0.5, f"FEM relative L2 error too large: {float(rel_l2_fem):.3e}"
