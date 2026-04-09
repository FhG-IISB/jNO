import jax

# jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import lineax as lx

import jno

import numpy as np

"""
2-D Poisson equation with manufactured polynomial solution (FEM only)

Problem
-------
    -Delta u = f    in [0, 1]^2
    u = 0           on the boundary

Analytical solution
-------------------
    u(x, y) = x (1 - x) y (1 - y)

Then
----
    -Delta u = 2 [x (1 - x) + y (1 - y)]

Why this example?
-----------------
- very clean FEM-only verification example
- smooth non-oscillatory exact solution
- homogeneous Dirichlet BCs on all boundaries
"""


# -----------------------------------------------------------------------------
# Manufactured solution
# -----------------------------------------------------------------------------
def exact_u(x, y):
    return x * (1.0 - x) * y * (1.0 - y)


def exact_u_num(x, y):
    return x * (1.0 - x) * y * (1.0 - y)


def source_f(x, y):
    return 2.0 * (x * (1.0 - x) + y * (1.0 - y))


# -----------------------------------------------------------------------------
# FEM domain and weak form
# -----------------------------------------------------------------------------
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.18))
domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        domain.dirichlet(["left", "right", "bottom", "top"], 0.0),
    ],
    fem_solver=True,
)

u, phi = domain.fem_symbols()
xg, yg, _ = domain.variable("fem_gauss", split=True)

du_dx = jno.np.grad(u, xg)
du_dy = jno.np.grad(u, yg)
phi_x = jno.np.grad(phi, xg)
phi_y = jno.np.grad(phi, yg)

# Weak form:
# ∫_Omega grad(u)·grad(phi) dOmega = ∫_Omega f phi dOmega
vol_integrand = du_dx * phi_x + du_dy * phi_y - source_f(xg, yg) * phi
weak = vol_integrand

A, b = weak.assemble(domain, target="fem_system")
A_dense = jnp.asarray(A.todense())
b_dense = jnp.asarray(b)

op = lx.MatrixLinearOperator(A_dense)
sol = lx.linear_solve(op, b_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem = sol.value.reshape(-1)

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
lin_res = jnp.linalg.norm(A_dense @ u_fem - b_dense) / (jnp.linalg.norm(b_dense) + 1e-14)
print(f"FEM linear solve residual: {lin_res:.6e}")

coords = np.asarray(domain.mesh.points)[:, :2]
x = jnp.asarray(coords[:, 0:1])
y = jnp.asarray(coords[:, 1:2])

u_exact = exact_u_num(x, y).reshape(-1)
abs_err = jnp.abs(u_exact - u_fem)
rel_l2 = jnp.linalg.norm(u_exact - u_fem) / (jnp.linalg.norm(u_exact) + 1e-14)
max_abs = jnp.max(abs_err)
mean_abs = jnp.mean(abs_err)

print(f"FEM Relative L2 Error: {rel_l2:.6e}")
print(f"FEM Mean Abs Error:    {mean_abs:.6e}")
print(f"FEM Max Abs Error:     {max_abs:.6e}")
assert float(rel_l2) < 0.5, f"FEM relative L2 error too large: {float(rel_l2):.3e}"
