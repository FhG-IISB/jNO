import jax

# jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import scipy.optimize as spo

import jno

import numpy as np

"""03 - Stationary Allen–Cahn equation (nonlinear FEM only)

Problem
-------
    -eps^2 Delta u + (u^3 - u) = 0    in [0, 1]^2

Boundary conditions
-------------------
    u = tanh((x - 0.5) / (sqrt(2) eps))   on x = 0 and x = 1

Analytical solution
-------------------
    u(x, y) = tanh((x - 0.5) / (sqrt(2) eps))
"""

sqrt = jno.np.sqrt
tanh = jno.np.tanh
eps = 0.05


# -----------------------------------------------------------------------------
# Exact interface profile
# -----------------------------------------------------------------------------
def u_exact(x, y):
    return tanh((x - 0.5) / (sqrt(2.0) * eps))


def u_exact_num(x, y):
    return jnp.tanh((x - 0.5) / (jnp.sqrt(2.0) * eps))


u_left = float(u_exact_num(jnp.array([[0.0]]), jnp.array([[0.0]])).reshape(()))
u_right = float(u_exact_num(jnp.array([[1.0]]), jnp.array([[0.0]])).reshape(()))

# -----------------------------------------------------------------------------
# Domain and weak form
# -----------------------------------------------------------------------------
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.12))
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

ux = jno.np.grad(u, xg)
uy = jno.np.grad(u, yg)
phix = jno.np.grad(phi, xg)
phiy = jno.np.grad(phi, yg)

weak = eps**2 * (ux * phix + uy * phiy) + (u**3 - u) * phi
op = weak.assemble(domain, target="fem_residual")

coords = np.asarray(domain.mesh.points)[:, :2]
x_nodes = jnp.asarray(coords[:, 0:1])
y_nodes = jnp.asarray(coords[:, 1:2])
u0 = u_exact_num(x_nodes, y_nodes).reshape(-1)

R0 = op.residual(u0)
print("Initial FEM residual norm:", float(jnp.linalg.norm(R0)))


def residual_np(u_np):
    return np.asarray(op.residual(jnp.asarray(u_np)))


def jacobian_np(u_np):
    J = op.jacobian(jnp.asarray(u_np))
    return np.asarray(J.todense())


sol_root = spo.root(
    residual_np,
    np.asarray(u0),
    jac=jacobian_np,
    method="hybr",
)

u_fem = jnp.asarray(sol_root.x).reshape(-1)
R_fem = op.residual(u_fem)

print("SciPy root success:", sol_root.success)
print("SciPy root status :", sol_root.status)
print("SciPy root msg    :", sol_root.message)
print("Final FEM residual norm:", float(jnp.linalg.norm(R_fem)))

u_exact_nodes = u_exact_num(x_nodes, y_nodes).reshape(-1)
rel_l2 = jnp.linalg.norm(u_exact_nodes - u_fem) / (jnp.linalg.norm(u_exact_nodes) + 1e-14)
max_abs = jnp.max(jnp.abs(u_exact_nodes - u_fem))
print(f"FEM Relative L2 Error: {rel_l2:.6e}")
print(f"FEM Max Abs Error:     {max_abs:.6e}")
assert float(rel_l2) < 0.5, f"FEM relative L2 error too large: {float(rel_l2):.3e}"
