"""
02 - Stationary Allen-Cahn equation with nonlinear FEAX-FEM residual route

Problem
-------
    -eps^2 Δu + (u^3 - u) = 0      in Ω = [0, 1]^2

Boundary conditions
-------------------
    u = tanh((x - 0.5) / (sqrt(2) eps))  on x = 0 and x = 1

Manufactured profile
--------------------
    u(x, y) = tanh((x - 0.5) / (sqrt(2) eps))

Showcases
---------
- weak.assemble(target="fem_residual")
- FemResidualOperator
- residual_fn(u), jacobian_fn(u)
- external nonlinear solve using scipy.optimize.root
"""

import numpy as np
import scipy.optimize as spo

import jax.numpy as jnp

import jno


eps = 0.05


# ---------------------------------------------------------------------
# Exact interface profile
# ---------------------------------------------------------------------


def exact_u_num(x, y):
    del y
    return jnp.tanh((x - 0.5) / (jnp.sqrt(2.0) * eps))


u_left = float(exact_u_num(jnp.array([[0.0]]), jnp.array([[0.0]])).reshape(()))
u_right = float(exact_u_num(jnp.array([[1.0]]), jnp.array([[0.0]])).reshape(()))


def to_dense(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    return jnp.asarray(A)


# ---------------------------------------------------------------------
# Domain and weak form
# ---------------------------------------------------------------------

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

# Weak form:
#   ∫ eps^2 grad(u)·grad(phi) dΩ + ∫ (u^3 - u) phi dΩ = 0
weak = eps**2 * (ux * phix + uy * phiy) + (u**3 - u) * phi

op = weak.assemble(domain, target="fem_residual")

coords = np.asarray(domain.mesh.points)[:, :2]
x_nodes = jnp.asarray(coords[:, 0:1])
y_nodes = jnp.asarray(coords[:, 1:2])

u0 = exact_u_num(x_nodes, y_nodes).reshape(-1)

R0 = op.residual(u0)
print("\n" + "=" * 70)
print("Stationary Allen-Cahn nonlinear FEAX-FEM example")
print("=" * 70)
print(f"Number of FEM DOFs       : {op.size}")
print(f"Initial residual norm    : {float(jnp.linalg.norm(R0)):.6e}")


def residual_np(u_np):
    return np.asarray(op.residual(jnp.asarray(u_np)))


def jacobian_np(u_np):
    J = op.jacobian(jnp.asarray(u_np))
    return np.asarray(to_dense(J))


sol = spo.root(
    residual_np,
    np.asarray(u0),
    jac=jacobian_np,
    method="hybr",
)

u_fem = jnp.asarray(sol.x).reshape(-1)
R_fem = op.residual(u_fem)

u_exact = exact_u_num(x_nodes, y_nodes).reshape(-1)

rel_l2 = jnp.linalg.norm(u_exact - u_fem) / (jnp.linalg.norm(u_exact) + 1e-14)
max_abs = jnp.max(jnp.abs(u_exact - u_fem))

print(f"SciPy root success       : {sol.success}")
print(f"SciPy root status        : {sol.status}")
print(f"Final residual norm      : {float(jnp.linalg.norm(R_fem)):.6e}")
print(f"Relative L2 error        : {float(rel_l2):.6e}")
print(f"Maximum absolute error   : {float(max_abs):.6e}")

assert sol.success
assert float(rel_l2) < 5e-1