import sys

try:
    import feax  # noqa: F401
except ImportError:
    sys.exit(0)

import jax

jax.config.update("jax_enable_x64", False)

import foundax
import jax.numpy as jnp
import numpy as np
import optax

import jno
from jno import LearningRateSchedule as lrs

"""02 - 2-D Helmholtz equation with FEAX-FEM and variational PINNs

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

Showcases
---------
- mixed Dirichlet + Neumann weak form
- weak.assemble(target="vpinn")
- weak.assemble(target="fem_system")
- FEAX-backed FEM reference solve
- variational PINN training with hard Dirichlet ansatz
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
    # x = 1, outward normal n = (1, 0)
    # u_x(1,y) = pi*cos(pi)*[cos(pi*y)+y] = -pi[cos(pi*y)+y]
    return -pi * (cos(pi * y) + y)


def exact_flux_top(x, y):
    # y = 1, outward normal n = (0, 1)
    # u_y(x,1) = sin(pi*x)*[-pi*sin(pi) + 1] = sin(pi*x)
    return sin(pi * x)


def to_dense(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    return jnp.asarray(A)


# -----------------------------------------------------------------------------
# Domain helper
# -----------------------------------------------------------------------------


def make_domain(mesh_size=0.22):
    domain = jno.domain(constructor=jno.domain.rect(mesh_size=mesh_size))
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
    return domain


# -----------------------------------------------------------------------------
# Shared weak form builder
# -----------------------------------------------------------------------------


def build_helmholtz_weak_form(domain):
    u, phi = domain.fem_symbols()

    xg, yg, _ = domain.variable("fem_gauss", split=True)
    xr, yr, _ = domain.variable("gauss_right", split=True)
    xt, yt, _ = domain.variable("gauss_top", split=True)

    du_dx = jno.np.grad(u, xg)
    du_dy = jno.np.grad(u, yg)
    phi_x = jno.np.grad(phi, xg)
    phi_y = jno.np.grad(phi, yg)

    k_sq = 0.0 * xg + k_val**2

    vol_integrand = du_dx * phi_x + du_dy * phi_y - k_sq * u * phi - source_f(xg, yg) * phi

    neumann_right = exact_flux_right(xr, yr) * phi
    neumann_top = exact_flux_top(xt, yt) * phi

    weak = vol_integrand - neumann_right - neumann_top

    # Return volume variables so VPINN trial value is built on the same
    # quadrature support as the weak form.
    return weak, xg, yg


# -----------------------------------------------------------------------------
# Training domain: VPINN
# -----------------------------------------------------------------------------

train_domain = make_domain(mesh_size=0.22)

weak_vpinn, xg, yg = build_helmholtz_weak_form(train_domain)
x_int, y_int, _ = train_domain.variable("interior", split=True)


# -----------------------------------------------------------------------------
# Neural network with hard Dirichlet BCs
# -----------------------------------------------------------------------------

net = jno.nn.wrap(
    foundax.mlp(
        2,
        hidden_dims=32,
        num_layers=4,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)


def apply_hard_bc(u_pred, x, y):
    # Satisfies:
    #   u(0,y) = 0
    #   u(x,0) = sin(pi*x)
    return sin(pi * x) + x * y * u_pred


u_gauss = apply_hard_bc(net(xg, yg), xg, yg)
u_int = apply_hard_bc(net(x_int, y_int), x_int, y_int)


# -----------------------------------------------------------------------------
# Variational PINN training
# -----------------------------------------------------------------------------

pde = weak_vpinn.assemble(train_domain, u_net=u_gauss, target="vpinn")

crux = jno.core(constraints=[pde.mse], domain=train_domain)

net.optimizer(
    optax.adam,
    lr=lrs.warmup_cosine(
        10,
        1,
        1e-3,
        1e-5,
    ),
)

crux.solve(epochs=10)


# -----------------------------------------------------------------------------
# Reference FEAX-FEM solve
# -----------------------------------------------------------------------------

fem_domain = make_domain(mesh_size=0.22)

weak_fem, _, _ = build_helmholtz_weak_form(fem_domain)

A_fem, b_fem = weak_fem.assemble(fem_domain, target="fem_system")

A_fem_dense = to_dense(A_fem)
b_fem_dense = jnp.asarray(b_fem)

u_fem = jnp.linalg.solve(A_fem_dense, b_fem_dense).reshape(-1)

lin_res_fem = jnp.linalg.norm(A_fem_dense @ u_fem - b_fem_dense) / (jnp.linalg.norm(b_fem_dense) + 1e-14)

print("\n" + "=" * 70)
print("2D Helmholtz mixed BC: FEAX-FEM + VPINN")
print("=" * 70)
print(f"FEM linear solve residual: {float(lin_res_fem):.6e}")


# -----------------------------------------------------------------------------
# Compare VPINN and FEM
# -----------------------------------------------------------------------------

x_eval, y_eval, _ = fem_domain.variable("interior", split=True)

u_vpinn_eval = crux.eval(
    apply_hard_bc(net(x_eval, y_eval), x_eval, y_eval),
    domain=fem_domain,
)
u_true_eval = crux.eval(exact_u(x_eval, y_eval), domain=fem_domain)

u_vpinn_eval = jnp.asarray(u_vpinn_eval).reshape(-1)
u_true_eval = jnp.asarray(u_true_eval).reshape(-1)

rel_l2_vpinn = jnp.linalg.norm(u_true_eval - u_vpinn_eval) / (jnp.linalg.norm(u_true_eval) + 1e-14)
max_abs_vpinn = jnp.max(jnp.abs(u_true_eval - u_vpinn_eval))

coords_fem = np.asarray(fem_domain.mesh.points)[:, :2]
x_f = jnp.asarray(coords_fem[:, 0:1])
y_f = jnp.asarray(coords_fem[:, 1:2])

u_exact_fem = exact_u_num(x_f, y_f).reshape(-1)
u_fem_vec = jnp.asarray(u_fem).reshape(-1)

rel_l2_fem = jnp.linalg.norm(u_exact_fem - u_fem_vec) / (jnp.linalg.norm(u_exact_fem) + 1e-14)
max_abs_fem = jnp.max(jnp.abs(u_exact_fem - u_fem_vec))

print("\nVPINN")
print("-" * 70)
print(f"Relative L2 Error on FEM domain: {float(rel_l2_vpinn):.6e}")
print(f"Max Abs Error on FEM domain    : {float(max_abs_vpinn):.6e}")

print("\nFEAX-FEM")
print("-" * 70)
print(f"Relative L2 Error             : {float(rel_l2_fem):.6e}")
print(f"Max Abs Error                 : {float(max_abs_fem):.6e}")

assert float(rel_l2_vpinn) < 1.1, f"VPINN relative L2 error too large: {float(rel_l2_vpinn):.3e}"
assert float(rel_l2_fem) < 0.5, f"FEM relative L2 error too large: {float(rel_l2_fem):.3e}"
