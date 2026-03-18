import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import optax

import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs

# ============================================================
# 0. Symbolic vs numeric math
# ============================================================
# symbolic math for weak forms
sqrt = jnn.sqrt
tanh = jnn.tanh

# numeric math for postprocessing
sqrt_num = jnp.sqrt
tanh_num = jnp.tanh

# Allen–Cahn parameter
eps = 0.05


# ============================================================
# 1. Exact interface solution (true Allen–Cahn solution)
# ============================================================
# u(x,y) = tanh((x-0.5)/(sqrt(2)*eps))
# This solves:
#   -eps^2 Δu + (u^3 - u) = 0
# exactly, since u depends only on x and is the 1D heteroclinic profile.

def u_exact(x, y):
    return tanh((x - 0.5) / (sqrt(2.0) * eps))

def u_exact_num(x, y):
    return jnp.tanh((x - 0.5) / (jnp.sqrt(2.0) * eps))

# left/right Dirichlet values
u_left = float(u_exact_num(jnp.array([[0.0]]), jnp.array([[0.0]])).reshape(()))
u_right = float(u_exact_num(jnp.array([[1.0]]), jnp.array([[0.0]])).reshape(()))


# ============================================================
# 2. Domain
# ============================================================
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.03))

domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        domain.dirichlet("left", u_left),
        domain.dirichlet("right", u_right),
    ],
    fem_solver=True,
)

u, phi = domain.fem_symbols()
xg, yg, _ = domain.variable("fem_gauss", split=True)

ux = jnn.grad(u, xg)
uy = jnn.grad(u, yg)
phix = jnn.grad(phi, xg)
phiy = jnn.grad(phi, yg)

# Weak form:
# ∫ eps^2 grad(u).grad(phi) + (u^3 - u) phi dΩ = 0
weak = (
    eps**2 * (ux * phix + uy * phiy)
    + (u**3 - u) * phi
)


# ============================================================
# 3. Nonlinear FEM solve via SciPy root
# ============================================================
op = weak.assemble(domain, target="fem_residual")

coords = np.asarray(domain.mesh.points)[:, :2]
x_nodes = jnp.asarray(coords[:, 0:1])
y_nodes = jnp.asarray(coords[:, 1:2])

# Use exact profile as initial guess to stay on the intended branch
u0 = u_exact_num(x_nodes, y_nodes).reshape(-1)

R0 = op.residual(u0)
print("Initial FEM residual norm:", float(jnp.linalg.norm(R0)))

def residual_np(u_np):
    u_jax = jnp.asarray(u_np)
    r_jax = op.residual(u_jax)
    return np.asarray(r_jax)

def jacobian_np(u_np):
    u_jax = jnp.asarray(u_np)
    J = op.jacobian(u_jax)

    if hasattr(J, "todense"):      # JAX BCOO
        return np.asarray(J.todense())
    elif hasattr(J, "toarray"):    # SciPy sparse
        return np.asarray(J.toarray())
    else:
        return np.asarray(J)

sol_root = spo.root(
    residual_np,
    np.asarray(u0),
    jac=jacobian_np,
    method="hybr",
)

u_fem = jnp.asarray(sol_root.x)
R_fem = op.residual(u_fem)

print("SciPy root success:", sol_root.success)
print("SciPy root status :", sol_root.status)
print("SciPy root msg    :", sol_root.message)
print("Final FEM residual norm:", float(jnp.linalg.norm(R_fem)))


# ============================================================
# 4. VPINN solve on same weak form
# ============================================================
x_int, y_int, _ = domain.variable("interior", split=True)

key = jax.random.PRNGKey(0)

net = jnn.nn.mlp(
    2,
    hidden_dims=128,
    num_layers=4,
    activation=jax.nn.tanh,
    key=key,
)

# Hard Dirichlet ansatz only on left/right:
# g(x) matches the exact left/right boundary values,
# and x(1-x)*NN vanishes on x=0 and x=1.
def lifting_lr(x):
    return (1.0 - x) * u_left + x * u_right

def apply_hard_bc(u_pred, x, y):
    return lifting_lr(x) + x * (1.0 - x) * u_pred

u_gauss = apply_hard_bc(net(xg, yg), xg, yg)
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)

pde = weak.assemble(domain, u_net=u_gauss, target="vpinn")

val_error = jnn.abs(u_int - u_exact(x_int, y_int))
error_tracker = jnn.tracker(val_error, interval=100)

crux = jno.core(constraints=[pde.mse, error_tracker], domain=domain)

learning_rate = lrs.warmup_cosine(4000, 200, 1e-3, 1e-5)
net.optimizer(optax.adam, lr=learning_rate)
crux.solve(epochs=4000)

u_vpinn_int = crux.eval(
    apply_hard_bc(net(x_int, y_int), x_int, y_int),
    domain=domain,
)
u_true_int = crux.eval(u_exact(x_int, y_int), domain=domain)

rel_l2_vpinn_int = jnp.linalg.norm(u_true_int - u_vpinn_int) / jnp.linalg.norm(u_true_int)
print(f"VPINN relative L2 error (interior sample): {rel_l2_vpinn_int:.6e}")


# ============================================================
# 5. Compare both against exact solution at mesh nodes
# ============================================================
u_exact_nodes = u_exact_num(x_nodes, y_nodes).reshape(-1)
u_fem_nodes = u_fem.reshape(-1)

u_vpinn_nodes = crux.eval(
    apply_hard_bc(net(x_nodes, y_nodes), x_nodes, y_nodes),
    domain=domain,
).reshape(-1)

def compute_errors(u_sol, name):
    rel_l2 = jnp.linalg.norm(u_exact_nodes - u_sol) / jnp.linalg.norm(u_exact_nodes)
    max_abs = jnp.max(jnp.abs(u_exact_nodes - u_sol))
    print(f"{name:>18} | rel L2 = {float(rel_l2):.6e} | max abs = {float(max_abs):.6e}")
    return float(rel_l2), float(max_abs)

print("\n==============================")
print("Error summary at mesh nodes")
print("==============================")
err_fem = compute_errors(u_fem_nodes, "FEM + SciPy root")
err_vpinn = compute_errors(u_vpinn_nodes, "VPINN")


# ============================================================
# 6. Plot exact / FEM / VPINN / errors
# ============================================================
tri = domain.mesh.cells_dict["triangle"]

u_exact_np = np.asarray(u_exact_nodes)
u_fem_np = np.asarray(u_fem_nodes)
u_vpinn_np = np.asarray(u_vpinn_nodes)

err_fem_np = np.abs(u_fem_np - u_exact_np)
err_vpinn_np = np.abs(u_vpinn_np - u_exact_np)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

def plot_field(ax, vals, title, cmap="viridis"):
    im = ax.tripcolor(
        coords[:, 0],
        coords[:, 1],
        tri,
        vals,
        shading="flat",
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

plot_field(axes[0, 0], u_exact_np,   "Exact interface solution")
plot_field(axes[0, 1], u_fem_np,     "Allen–Cahn FEM (SciPy root)")
plot_field(axes[0, 2], u_vpinn_np,   "Allen–Cahn VPINN")

plot_field(axes[1, 0], err_fem_np,   "FEM abs error", cmap="magma")
plot_field(axes[1, 1], err_vpinn_np, "VPINN abs error", cmap="magma")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig("allen_cahn_interface_fem_vpinn_compare.png", dpi=200)