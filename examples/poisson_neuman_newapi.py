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

pi = jnn.pi; sin = jnn.sin; cos = jnn.cos

# ============================================================
# 1. Exact solution / forcing / Neumann data
# ============================================================
def exact_u(x, y): return sin(pi * x) * cos(pi * y)


def source_f(x, y): return 2.0 * pi**2 * sin(pi * x) * cos(pi * y)


def exact_flux_x(x, y): return pi * cos(pi * x) * cos(pi * y)# du/dx


def exact_flux_y(x, y): return -pi * sin(pi * x) * sin(pi * y)# du/dy


def exact_u_num(x, y): return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)

# ============================================================
# 2. Domain + FEM/variational init
# ============================================================
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))

domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        domain.dirichlet("left", 0.0),
        domain.dirichlet("bottom", lambda p: jnp.sin(jnp.pi * p[0])),
        domain.neumann(["right", "top"]),
    ],
    fem_solver=True,
)

u, phi = domain.fem_symbols()

# Volume quadrature
xg, yg, _ = domain.variable("fem_gauss", split=True)

# Boundary quadrature
xr, yr, _ = domain.variable("gauss_right", split=True)
xt, yt, _ = domain.variable("gauss_top", split=True)

# Interior points for VPINN tracker
x_int, y_int, _ = domain.variable("interior", split=True)


# ============================================================
# 3. Neural network + hard BC
# ============================================================
key = jax.random.PRNGKey(0)
net = jnn.nn.mlp(2,hidden_dims=128,num_layers=4,activation=jax.nn.tanh,key=key,)

# Enforces:
#   left   (x=0): u = 0;  bottom (y=0): u = sin(pi*x)
def apply_hard_bc(u_pred, x, y): return sin(pi * x) + u_pred * x * y

u_gauss = apply_hard_bc(net(xg, yg), xg, yg)
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)

# ============================================================
# 4. One symbolic weak form
# ============================================================
du_dx = jnn.grad(u, xg)
du_dy = jnn.grad(u, yg)

phi_x = jnn.grad(phi, xg)
phi_y = jnn.grad(phi, yg)

vol_integrand = du_dx * phi_x + du_dy * phi_y - source_f(xg, yg) * phi

# right boundary outward normal = (1,0)
neumann_right = exact_flux_x(xr, yr) * phi

# top boundary outward normal = (0,1)
neumann_top = exact_flux_y(xt, yt) * phi

weak = vol_integrand - neumann_right - neumann_top

# ============================================================
# 5. Assemble BOTH targets from the SAME weak form
# ============================================================
pde = weak.assemble(domain, u_net=u_gauss, target="vpinn")
A, b = weak.assemble(domain, target="fem_system")

print("VPINN assembled object:", pde)
print("A shape:", A.shape)
print("b shape:", b.shape)

A_dense = jnp.asarray(A.todense())
b_dense = jnp.asarray(b)
# solve usign jax lineax
op = lx.MatrixLinearOperator(A_dense)
sol = lx.linear_solve(op, b_dense, solver=lx.AutoLinearSolver(well_posed=True),)
u_fem = sol.value

lin_res = jnp.linalg.norm(A_dense @ u_fem - b_dense) / (jnp.linalg.norm(b_dense) + 1e-14)
print(f"Linear solve residual: {lin_res:.6e}")
print("u_fem shape:", u_fem.shape)

# ============================================================
# 6. VPINN loss + tracker
# ============================================================
val_error = jnn.abs(u_int - exact_u(x_int, y_int))
error_tracker = jnn.tracker(val_error, interval=50)

crux = jno.core(constraints=[pde.mse, error_tracker], domain=domain)

# ============================================================
# 7. Training
# ============================================================
learning_rate = lrs.warmup_cosine(4000, 200, 1e-3, 1e-5)
net.optimizer(optax.adam, lr=learning_rate)
crux.solve(epochs=4000)

net.optimizer(optax.lbfgs(1e-6))
crux.solve(epochs=1000)

# ============================================================
# 8. Dense evaluation for VPINN
# ============================================================
eval_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))
x_eval, y_eval, _ = eval_domain.variable("interior", split=True)

u_pred_val = crux.eval(
    apply_hard_bc(net(x_eval, y_eval), x_eval, y_eval),
    domain=eval_domain,
)
u_true_val = crux.eval(exact_u(x_eval, y_eval), domain=eval_domain)

rel_l2_vpinn = jnp.linalg.norm(u_true_val - u_pred_val) / (jnp.linalg.norm(u_true_val) + 1e-14)
abs_error_vpinn = jnp.abs(u_true_val - u_pred_val)

print(f"VPINN Relative L2 Error: {rel_l2_vpinn:.6e}")

# ============================================================
# 9. FEM nodal comparison
# ============================================================
coords_fe = np.asarray(domain.mesh.points)
xy_fe = coords_fe[:, :2]

x_fe = jnp.asarray(xy_fe[:, 0:1])
y_fe = jnp.asarray(xy_fe[:, 1:2])

u_exact_fe = exact_u_num(x_fe, y_fe).reshape(-1)
u_fem_vec = jnp.asarray(u_fem).reshape(-1)

rel_l2_fem = jnp.linalg.norm(u_exact_fe - u_fem_vec) / (jnp.linalg.norm(u_exact_fe) + 1e-14)
abs_err_fem = jnp.abs(u_exact_fe - u_fem_vec)

print(f"FEM Relative L2 Error (nodal): {rel_l2_fem:.6e}")
print(f"FEM Max Abs Error (nodal): {jnp.max(abs_err_fem):.6e}")

# ============================================================
# 10. Plot
# ============================================================
coords_eval = np.asarray(eval_domain.mesh.points)
xy_eval = coords_eval[:, :2]
x_eval_pts = xy_eval[:, 0]
y_eval_pts = xy_eval[:, 1]

x_fe_pts = xy_fe[:, 0]
y_fe_pts = xy_fe[:, 1]

tri_eval = eval_domain.mesh.cells_dict["triangle"]
tri_fe = domain.mesh.cells_dict["triangle"]

fig, axes = plt.subplots(1, 4, figsize=(24, 5))


def plot_component(ax, x, y, triangles, val, title, cmap="viridis"):
    v = np.array(val).reshape(-1)
    im = ax.tripcolor(x, y, triangles, v, cmap=cmap, shading="flat")
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


plot_component(axes[0], x_eval_pts, y_eval_pts, tri_eval, u_true_val, "Exact solution")
plot_component(axes[1], x_eval_pts, y_eval_pts, tri_eval, u_pred_val, "VPINN prediction")
plot_component(axes[2], x_eval_pts, y_eval_pts, tri_eval, abs_error_vpinn, "VPINN absolute error", cmap="magma")
plot_component(axes[3], x_fe_pts, y_fe_pts, tri_fe, u_fem, "FEM solution")

plt.tight_layout()
plt.savefig("poisson_neumann_new_api_with_fem_corrected.png", dpi=300)