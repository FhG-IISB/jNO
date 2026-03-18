import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import lineax as lx
import equinox as eqx

import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs
from jno.architectures.linear import Linear


# ============================================================
# 0. Problem setup: 2D linear elasticity on unit square
# ============================================================
# Strong form:
#   -div(sigma(u)) = f   in Omega
#              u   = 0   on boundary
#
# Weak form:
#   ∫ sigma(u) : eps(v) dΩ - ∫ f · v dΩ = 0
#
# Constitutive law:
#   sigma(u) = lambda * tr(eps(u)) * I + 2 * mu * eps(u)
#   eps(u)   = symgrad(u)

pi = jnn.pi
sin = jnn.sin
cos = jnn.cos

pi_num = jnp.pi

lam = 2.0
mu = 1.0

I2 = jnp.eye(2)
e1 = jnp.array([1.0, 0.0])
e2 = jnp.array([0.0, 1.0])


# ============================================================
# 1. Manufactured exact solution (zero on all boundaries)
# ============================================================
# u1(x,y) = sin(pi x) sin(pi y)
# u2(x,y) = 0.5 sin(2 pi x) sin(pi y)
#
# These vanish on x=0,1 and y=0,1.

def u1_exact(x, y):
    return sin(pi * x) * sin(pi * y)

def u2_exact(x, y):
    return 0.5 * sin(2.0 * pi * x) * sin(pi * y)

def exact_u_num(x, y):
    u1 = jnp.sin(pi_num * x) * jnp.sin(pi_num * y)
    u2 = 0.5 * jnp.sin(2.0 * pi_num * x) * jnp.sin(pi_num * y)
    return jnp.concatenate([u1, u2], axis=-1)


# ============================================================
# 2. Analytic body force f = -div(sigma(u_exact))
# ============================================================
# For the chosen manufactured solution:
#
# f1 = (lambda + 3 mu) pi^2 sin(pi x) sin(pi y)
#      - (lambda + mu) pi^2 cos(2 pi x) cos(pi y)
#
# f2 = -(lambda + mu) pi^2 cos(pi x) cos(pi y)
#      + (lambda/2 + 3 mu) pi^2 sin(2 pi x) sin(pi y)

def f1_sym(x, y):
    return (
        (lam + 3.0 * mu) * pi**2 * sin(pi * x) * sin(pi * y)
        - (lam + mu) * pi**2 * cos(2.0 * pi * x) * cos(pi * y)
    )

def f2_sym(x, y):
    return (
        -(lam + mu) * pi**2 * cos(pi * x) * cos(pi * y)
        + (0.5 * lam + 3.0 * mu) * pi**2 * sin(2.0 * pi * x) * sin(pi * y)
    )
def body_force(x, y):
    return jnn.stack([f1_sym(x, y), f2_sym(x, y)], axis=-1)


# ============================================================
# 3. Geometry / domain
# ============================================================
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.06))

domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        domain.dirichlet(["left", "right", "bottom", "top"], (0.0, 0.0)),
    ],
    fem_solver=True,
    vec=2,
)
# ============================================================
# 4. Symbolic weak form
# ============================================================
u, phi = domain.fem_symbols(value_shape=(2,))
xg, yg, _ = domain.variable("fem_gauss", split=True)
x_int, y_int, _ = domain.variable("interior", split=True)

eps_u = jnn.symgrad(u, [xg, yg])
eps_phi = jnn.symgrad(phi, [xg, yg])


tr_eps_u = jnn.trace(eps_u)
sigma_u = lam * jnn.einsum("q,ij->qij", tr_eps_u, I2) + 2.0 * mu * eps_u

weak = (
    jnn.inner(sigma_u, eps_phi, n_contract=2)
    - jnn.inner(body_force(xg, yg), phi)
)


# ============================================================
# 5. FEM solve
# ============================================================
A, b = weak.assemble(domain, target="fem_system")

A_dense = jnp.asarray(A.todense())
b_dense = jnp.asarray(b)

op = lx.MatrixLinearOperator(A_dense)
sol = lx.linear_solve(op, b_dense, solver=lx.AutoLinearSolver(well_posed=True))

u_fem_flat = jnp.asarray(sol.value)
lin_res = jnp.linalg.norm(A_dense @ u_fem_flat - b_dense) / (jnp.linalg.norm(b_dense) + 1e-14)
print(f"FEM linear solve residual: {float(lin_res):.6e}")

coords = np.asarray(domain.mesh.points)[:, :2]
x_nodes = jnp.asarray(coords[:, 0:1])
y_nodes = jnp.asarray(coords[:, 1:2])

u_fem_nodes = u_fem_flat.reshape(-1, 2)
u_exact_nodes = exact_u_num(x_nodes, y_nodes)


# ============================================================
# 6. VPINN model (vector output)
# ============================================================
class VectorMLP(eqx.Module):
    dense1: Linear
    dense2: Linear
    dense3: Linear

    def __init__(self, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.dense1 = Linear(2, 64, key=k1)
        self.dense2 = Linear(64, 64, key=k2)
        self.dense3 = Linear(64, 2, key=k3)

    def __call__(self, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        h = jnp.tanh(self.dense1(h))
        h = jnp.tanh(self.dense2(h))
        return self.dense3(h)


key = jax.random.PRNGKey(0)
net = jnn.nn.wrap(VectorMLP(key=key))

def apply_hard_bc(u_pred, x, y):
    # homogeneous Dirichlet on all four sides
    lift = x * (1.0 - x) * y * (1.0 - y)
    return lift * u_pred

u_gauss = apply_hard_bc(net(xg, yg), xg, yg)
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)


# ============================================================
# 7. VPINN assembly + training
# ============================================================
pde = weak.assemble(domain, u_net=u_gauss, target="vpinn")

crux = jno.core(constraints=[pde.mse], domain=domain)

learning_rate = lrs.warmup_cosine(4000, 200, 1e-3, 1e-5)
net.optimizer(optax.adam, lr=learning_rate)
crux.solve(epochs=4000)

u_vpinn_nodes = crux.eval(
    apply_hard_bc(net(x_nodes, y_nodes), x_nodes, y_nodes),
    domain=domain,
)

u_vpinn_nodes = jnp.asarray(u_vpinn_nodes)
if u_vpinn_nodes.ndim == 3 and u_vpinn_nodes.shape[0] == 1:
    u_vpinn_nodes = u_vpinn_nodes[0]
# ============================================================
# 8. Error summaries
# ============================================================
def rel_l2_vec(u_ref, u_pred):
    return jnp.linalg.norm((u_ref - u_pred).reshape(-1)) / (jnp.linalg.norm(u_ref.reshape(-1)) + 1e-14)

def comp_rel_l2(u_ref, u_pred, comp):
    u_ref = jnp.asarray(u_ref)
    u_pred = jnp.asarray(u_pred)

    if u_ref.ndim == 3 and u_ref.shape[0] == 1:
        u_ref = u_ref[0]
    if u_pred.ndim == 3 and u_pred.shape[0] == 1:
        u_pred = u_pred[0]

    return jnp.linalg.norm(u_ref[:, comp] - u_pred[:, comp]) / (
        jnp.linalg.norm(u_ref[:, comp]) + 1e-14
    )

def max_abs_vec(u_ref, u_pred):
    return jnp.max(jnp.abs(u_ref - u_pred))

print("\n==============================")
print("Error summary at mesh nodes")
print("==============================")

fem_rel = rel_l2_vec(u_exact_nodes, u_fem_nodes)
vpinn_rel = rel_l2_vec(u_exact_nodes, u_vpinn_nodes)
vpinn_vs_fem = rel_l2_vec(u_fem_nodes, u_vpinn_nodes)

print(f"FEM   vs exact | rel L2 = {float(fem_rel):.6e} | max abs = {float(max_abs_vec(u_exact_nodes, u_fem_nodes)):.6e}")
print(f"VPINN vs exact | rel L2 = {float(vpinn_rel):.6e} | max abs = {float(max_abs_vec(u_exact_nodes, u_vpinn_nodes)):.6e}")
print(f"VPINN vs FEM   | rel L2 = {float(vpinn_vs_fem):.6e} | max abs = {float(max_abs_vec(u_fem_nodes, u_vpinn_nodes)):.6e}")

print("\nComponent-wise relative L2:")
print(f"u_x FEM   vs exact : {float(comp_rel_l2(u_exact_nodes, u_fem_nodes,   0)):.6e}")
print(f"u_y FEM   vs exact : {float(comp_rel_l2(u_exact_nodes, u_fem_nodes,   1)):.6e}")
print(f"u_x VPINN vs exact : {float(comp_rel_l2(u_exact_nodes, u_vpinn_nodes, 0)):.6e}")
print(f"u_y VPINN vs exact : {float(comp_rel_l2(u_exact_nodes, u_vpinn_nodes, 1)):.6e}")


# ============================================================
# 9. Plot component fields
# ============================================================
tri = domain.mesh.cells_dict["triangle"]

u_exact_np = np.asarray(u_exact_nodes)
u_fem_np = np.asarray(u_fem_nodes)
u_vpinn_np = np.asarray(u_vpinn_nodes)

err_fem_np = np.abs(u_fem_np - u_exact_np)
err_vpinn_np = np.abs(u_vpinn_np - u_exact_np)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

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

plot_field(axes[0, 0], u_exact_np[:, 0], "Exact $u_x$")
plot_field(axes[0, 1], u_fem_np[:, 0],   "FEM $u_x$")
plot_field(axes[0, 2], u_vpinn_np[:, 0], "VPINN $u_x$")
plot_field(axes[0, 3], err_vpinn_np[:, 0], "VPINN abs err $u_x$", cmap="magma")

plot_field(axes[1, 0], u_exact_np[:, 1], "Exact $u_y$")
plot_field(axes[1, 1], u_fem_np[:, 1],   "FEM $u_y$")
plot_field(axes[1, 2], u_vpinn_np[:, 1], "VPINN $u_y$")
plot_field(axes[1, 3], err_vpinn_np[:, 1], "VPINN abs err $u_y$", cmap="magma")

plt.tight_layout()
plt.savefig("linear_elasticity_fem_vpinn_compare.png", dpi=220)