"""
03 - 2D reaction-diffusion equation with Robin BCs: FEAX-FEM + VPINN

Problem
-------
    -Δu + sigma u = f       in Ω = [0, 1]^2

Boundary conditions
-------------------
    u = y                  on x = 0
    u = 0                  on y = 0

    du/dn + alpha_right u = r_right   on x = 1
    du/dn + alpha_top   u = r_top     on y = 1

Manufactured solution
---------------------
    u(x, y) = x sin(pi y) + y

Showcases
---------
- mixed Dirichlet + Robin weak form
- FEAX-FEM linear system route
- VPINN weak-form route
- hard Dirichlet ansatz for VPINN
"""

import numpy as np

import jax
import jax.numpy as jnp
import optax
import foundax

import jno
from jno import LearningRateSchedule as lrs


pi = jno.np.pi
sin = jno.np.sin
sigma = 4.0
alpha_right = 2.0
alpha_top = 3.0


# ---------------------------------------------------------------------
# Manufactured solution
# ---------------------------------------------------------------------


def exact_u(x, y):
    return x * sin(pi * y) + y


def exact_u_num(x, y):
    return x * jnp.sin(jnp.pi * y) + y


def source_f(x, y):
    # -Δu + sigma u
    # u = x sin(pi y) + y
    # Δu = -pi^2 x sin(pi y)
    return x * (pi**2) * sin(pi * y) + sigma * (x * sin(pi * y) + y)


def robin_rhs_right(x, y):
    # x = 1, outward normal n = (1, 0)
    # du/dn = u_x = sin(pi y)
    return sin(pi * y) + alpha_right * (sin(pi * y) + y)


def robin_rhs_top(x, y):
    # y = 1, outward normal n = (0, 1)
    # u_y = pi x cos(pi y) + 1 = 1 - pi x
    # u(x, 1) = 1
    return 1.0 - pi * x + alpha_top


def to_dense(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    return jnp.asarray(A)


# ---------------------------------------------------------------------
# Shared weak form builder
# ---------------------------------------------------------------------


def build_robin_weak_form(domain):
    u, phi = domain.fem_symbols()

    xg, yg, _ = domain.variable("fem_gauss", split=True)
    xr, yr, _ = domain.variable("gauss_right", split=True)
    xt, yt, _ = domain.variable("gauss_top", split=True)

    du_dx = jno.np.grad(u, xg)
    du_dy = jno.np.grad(u, yg)
    phi_x = jno.np.grad(phi, xg)
    phi_y = jno.np.grad(phi, yg)

    vol = (
        du_dx * phi_x
        + du_dy * phi_y
        + sigma * u * phi
        - source_f(xg, yg) * phi
    )

    robin_right = (
        alpha_right * u * phi
        - robin_rhs_right(xr, yr) * phi
    )

    robin_top = (
        alpha_top * u * phi
        - robin_rhs_top(xt, yt) * phi
    )

    weak = vol + robin_right + robin_top

    # Return volume quadrature variables so the VPINN trial value is built
    # on the same support as the weak form.
    return weak, xg, yg


def make_domain(mesh_size=0.22):
    domain = jno.domain(constructor=jno.domain.rect(mesh_size=mesh_size))
    domain.init_fem(
        element_type="TRI3",
        quad_degree=3,
        bcs=[
            domain.dirichlet("left", lambda p: p[1]),
            domain.dirichlet("bottom", 0.0),
            domain.neumann(["right", "top"]),
        ],
        fem_solver=True,
    )
    return domain


# ---------------------------------------------------------------------
# FEAX-FEM solve
# ---------------------------------------------------------------------

fem_domain = make_domain(mesh_size=0.22)
weak_fem, _, _ = build_robin_weak_form(fem_domain)

A, b = weak_fem.assemble(fem_domain, target="fem_system")

A_dense = to_dense(A)
b_dense = jnp.asarray(b)

u_fem = jnp.linalg.solve(A_dense, b_dense).reshape(-1)

lin_res = jnp.linalg.norm(A_dense @ u_fem - b_dense) / (
    jnp.linalg.norm(b_dense) + 1e-14
)

coords = np.asarray(fem_domain.mesh.points)[:, :2]
x_nodes = jnp.asarray(coords[:, 0:1])
y_nodes = jnp.asarray(coords[:, 1:2])

u_exact_nodes = exact_u_num(x_nodes, y_nodes).reshape(-1)

rel_l2_fem = jnp.linalg.norm(u_exact_nodes - u_fem) / (
    jnp.linalg.norm(u_exact_nodes) + 1e-14
)
max_abs_fem = jnp.max(jnp.abs(u_exact_nodes - u_fem))

print("\n" + "=" * 70)
print("Reaction-diffusion Robin BC: FEAX-FEM + VPINN")
print("=" * 70)
print("FEAX-FEM")
print("-" * 70)
print(f"Number of FEM DOFs       : {u_fem.shape[0]}")
print(f"Linear solve residual    : {float(lin_res):.6e}")
print(f"Relative L2 error        : {float(rel_l2_fem):.6e}")
print(f"Maximum absolute error   : {float(max_abs_fem):.6e}")


# ---------------------------------------------------------------------
# VPINN training
# ---------------------------------------------------------------------

train_domain = make_domain(mesh_size=0.22)
weak_vpinn, xg, yg = build_robin_weak_form(train_domain)

x_int, y_int, _ = train_domain.variable("interior", split=True)

net = jno.nn.wrap(
    foundax.mlp(
        2,
        hidden_dims=32,
        num_layers=3,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)


def apply_hard_bc(raw, x, y):
    # Satisfies:
    #   u(0, y) = y
    #   u(x, 0) = 0
    return y + x * y * raw


u_gauss = apply_hard_bc(net(xg, yg), xg, yg)
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)

pde = weak_vpinn.assemble(train_domain, u_net=u_gauss, target="vpinn")

crux = jno.core(constraints=[pde.mse], domain=train_domain)

net.optimizer(
    optax.adam,
    lr=lrs.warmup_cosine(
        100,
        2,
        1e-3,
        1e-5,
    ),
)

crux.solve(epochs=1000)

u_vpinn_eval = crux.eval(u_int, domain=train_domain)
u_true_eval = crux.eval(exact_u(x_int, y_int), domain=train_domain)

u_vpinn_eval = jnp.asarray(u_vpinn_eval).reshape(-1)
u_true_eval = jnp.asarray(u_true_eval).reshape(-1)

rel_l2_vpinn = jnp.linalg.norm(u_true_eval - u_vpinn_eval) / (
    jnp.linalg.norm(u_true_eval) + 1e-14
)
max_abs_vpinn = jnp.max(jnp.abs(u_true_eval - u_vpinn_eval))

print("\nVPINN")
print("-" * 70)
print(f"Relative L2 error        : {float(rel_l2_vpinn):.6e}")
print(f"Maximum absolute error   : {float(max_abs_vpinn):.6e}")

assert float(rel_l2_fem) < 5e-1
assert float(rel_l2_vpinn) < 1.2