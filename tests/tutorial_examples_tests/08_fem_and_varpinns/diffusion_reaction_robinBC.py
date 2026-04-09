import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import lineax as lx
import optax

import jno

import numpy as np
from jno import LearningRateSchedule as lrs

"""
2-D diffusion-reaction equation with mixed Dirichlet + Robin BCs
(FEM + VPINN comparison)

Problem
-------
    -Delta u + sigma * u = f    in [0, 1]^2

Boundary conditions
-------------------
    u = y    on x = 0   (left)
    u = x    on y = 0   (bottom)

    du/dn + alpha_right * u = r_right   on x = 1
    du/dn + alpha_top   * u = r_top     on y = 1

Manufactured solution
---------------------
    u(x, y) = x + y

Then
----
    Delta u = 0
    f(x, y) = sigma * (x + y)

Robin data
----------
On x = 1:
    du/dn = u_x = 1
    r_right(y) = 1 + alpha_right * (1 + y)

On y = 1:
    du/dn = u_y = 1
    r_top(x) = 1 + alpha_top * (x + 1)

Why this example?
-----------------
- clean Robin-boundary demo for both FEM and VPINN
- easier and more stable than Helmholtz
- exact solution is simple and easy to verify
"""
pi = jno.np.pi
sin = jno.np.sin
cos = jno.np.cos

k_val = 4.0
alpha_right = 2.0
alpha_top = 3.0


# ============================================================
# Exact solution and forcing:  u(x,y) = x sin(pi y) + y
# ============================================================
def exact_u(x, y):
    return x * sin(pi * y) + y


def exact_u_num(x, y):
    return x * jnp.sin(jnp.pi * y) + y


def source_f(x, y):
    return x * (pi**2) * sin(pi * y) - (k_val**2) * (x * sin(pi * y) + y)


def robin_rhs_right(x, y):  # du/dn + alpha u on x=1
    return sin(pi * y) + alpha_right * (sin(pi * y) + y)


def robin_rhs_top(x, y):
    # du/dn + alpha u on y=1
    # u_y(x,1)=1-pi x, u(x,1)=1
    return 1.0 - pi * x + alpha_top


# ============================================================
# Training domain
# ============================================================
train_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.20))
train_domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        train_domain.dirichlet("left", lambda p: p[1]),
        train_domain.dirichlet("bottom", 0.0),
        train_domain.neumann(["right", "top"]),
    ],
    fem_solver=True,
)

u, phi = train_domain.fem_symbols()

# Volume quadrature
xg, yg, _ = train_domain.variable("fem_gauss", split=True)

# Boundary quadrature
xr, yr, _ = train_domain.variable("gauss_right", split=True)
xt, yt, _ = train_domain.variable("gauss_top", split=True)

# Interior points for tracking
x_int, y_int, _ = train_domain.variable("interior", split=True)

# ============================================================
# Network + hard BC ansatz
#
# lifting(x,y) = y
# correction = x*y*NN(x,y)
# ============================================================
key = jax.random.PRNGKey(0)

net = jno.np.nn.mlp(
    2,
    hidden_dims=32,
    num_layers=2,
    activation=jax.nn.tanh,
    key=key,
)


def apply_hard_bc(raw, x, y):
    return y + x * y * raw


# network evaluated on all supports
u_gauss = apply_hard_bc(net(xg, yg), xg, yg)
u_right = apply_hard_bc(net(xr, yr), xr, yr)
u_top = apply_hard_bc(net(xt, yt), xt, yt)
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)

# ============================================================
# Variational PINN weakfrom on coarse mesh weak form (symbolic u)
# ============================================================
du_dx = jno.np.grad(u, xg)
du_dy = jno.np.grad(u, yg)
phi_x = jno.np.grad(phi, xg)
phi_y = jno.np.grad(phi, yg)

k_sq = 0.0 * xg + k_val**2
alpha_r = 0.0 * xr + alpha_right
alpha_t = 0.0 * xt + alpha_top

vol_integrand_fem = du_dx * phi_x + du_dy * phi_y - k_sq * u * phi - source_f(xg, yg) * phi

robin_right_fem = alpha_r * u * phi - robin_rhs_right(xr, yr) * phi
robin_top_fem = alpha_t * u * phi - robin_rhs_top(xt, yt) * phi

weak_form = vol_integrand_fem + robin_right_fem + robin_top_fem
pde = weak_form.assemble(train_domain, u_net=u_gauss, target="vpinn")

# ============================================================
# Train VPINN
# ============================================================
crux = jno.core(constraints=[pde.mse], domain=train_domain)

learning_rate = lrs.warmup_cosine(
    5,
    1,
    1e-3,
    1e-5,
)

net.optimizer(optax.adam, lr=learning_rate)
crux.solve(epochs=5)

# ============================================================
# Fine FEM domain for reference / evaluation
# ============================================================
fem_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.20))
fem_domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        fem_domain.dirichlet("left", lambda p: p[1]),
        fem_domain.dirichlet("bottom", 0.0),
        fem_domain.neumann(["right", "top"]),
    ],
    fem_solver=True,
)

u_fem_sym, phi_fem_sym = fem_domain.fem_symbols()

xg_f, yg_f, _ = fem_domain.variable("fem_gauss", split=True)
xr_f, yr_f, _ = fem_domain.variable("gauss_right", split=True)
xt_f, yt_f, _ = fem_domain.variable("gauss_top", split=True)

du_dx_f = jno.np.grad(u_fem_sym, xg_f)
du_dy_f = jno.np.grad(u_fem_sym, yg_f)
phi_x_f = jno.np.grad(phi_fem_sym, xg_f)
phi_y_f = jno.np.grad(phi_fem_sym, yg_f)

k_sq_f = 0.0 * xg_f + k_val**2
alpha_r_f = 0.0 * xr_f + alpha_right
alpha_t_f = 0.0 * xt_f + alpha_top

vol_integrand_f = du_dx_f * phi_x_f + du_dy_f * phi_y_f - k_sq_f * u_fem_sym * phi_fem_sym - source_f(xg_f, yg_f) * phi_fem_sym

robin_right_f = alpha_r_f * u_fem_sym * phi_fem_sym - robin_rhs_right(xr_f, yr_f) * phi_fem_sym
robin_top_f = alpha_t_f * u_fem_sym * phi_fem_sym - robin_rhs_top(xt_f, yt_f) * phi_fem_sym

weak_fine_fem = vol_integrand_f + robin_right_f + robin_top_f

A_fine, b_fine = weak_fine_fem.assemble(fem_domain, target="fem_system")

A_fine_dense = jnp.asarray(A_fine.todense())
b_fine_dense = jnp.asarray(b_fine)

op_fine = lx.MatrixLinearOperator(A_fine_dense)
sol_fine = lx.linear_solve(op_fine, b_fine_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem_fine = sol_fine.value.reshape(-1)

lin_res_fine = jnp.linalg.norm(A_fine_dense @ u_fem_fine - b_fine_dense) / (jnp.linalg.norm(b_fine_dense) + 1e-14)
print(f"Fine FEM linear solve residual: {lin_res_fine:.6e}")
# ============================================================
# Evaluate VPINN on fine domain
# ============================================================
x_eval, y_eval, _ = fem_domain.variable("interior", split=True)

u_vpinn_eval = crux.eval(
    apply_hard_bc(net(x_eval, y_eval), x_eval, y_eval),
    domain=fem_domain,
)
u_true_eval = crux.eval(exact_u(x_eval, y_eval), domain=fem_domain)

rel_l2_vpinn = jnp.linalg.norm(u_true_eval - u_vpinn_eval) / (jnp.linalg.norm(u_true_eval) + 1e-14)
max_abs_vpinn = jnp.max(jnp.abs(u_true_eval - u_vpinn_eval))

print(f"VPINN Relative L2 Error on fine domain: {rel_l2_vpinn:.6e}")
print(f"VPINN Max Abs Error on fine domain:     {max_abs_vpinn:.6e}")

# ============================================================
# FEM nodal error on fine mesh
# ============================================================
coords = np.asarray(fem_domain.mesh.points)[:, :2]
x_nodes = jnp.asarray(coords[:, 0:1])
y_nodes = jnp.asarray(coords[:, 1:2])

u_exact_nodes = exact_u_num(x_nodes, y_nodes).reshape(-1)

rel_l2_fem = jnp.linalg.norm(u_exact_nodes - u_fem_fine) / (jnp.linalg.norm(u_exact_nodes) + 1e-14)
max_abs_fem = jnp.max(jnp.abs(u_exact_nodes - u_fem_fine))

print(f"Fine FEM Relative L2 Error: {rel_l2_fem:.6e}")
print(f"Fine FEM Max Abs Error:     {max_abs_fem:.6e}")
assert float(rel_l2_vpinn) < 1.1, f"VPINN relative L2 error too large: {float(rel_l2_vpinn):.3e}"
assert float(rel_l2_fem) < 0.5, f"FEM relative L2 error too large: {float(rel_l2_fem):.3e}"
# ============================================================
# Plot / smoke test
# ============================================================
