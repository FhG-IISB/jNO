"""
01 - 2D Poisson equation with FEAX-FEM assembly

Problem
-------
    -Δu = f      in Ω = [0, 1]^2
     u = 0      on ∂Ω

Manufactured solution
---------------------
    u(x, y) = x(1 - x)y(1 - y)

Then
----
    f(x, y) = 2[x(1 - x) + y(1 - y)]

Showcases
---------
- domain.init_fem(...)
- weak.assemble(target="fem_system")
- FEAX-backed linear FEM system A u = b
"""

import numpy as np

import jax.numpy as jnp

import jno


# ---------------------------------------------------------------------
# Manufactured solution
# ---------------------------------------------------------------------


def exact_u_num(x, y):
    return x * (1.0 - x) * y * (1.0 - y)


def source_f(x, y):
    return 2.0 * (x * (1.0 - x) + y * (1.0 - y))


def to_dense(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    return jnp.asarray(A)


# ---------------------------------------------------------------------
# Domain and FEAX-FEM setup
# ---------------------------------------------------------------------

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
#   ∫ grad(u)·grad(phi) dΩ - ∫ f phi dΩ = 0
weak = du_dx * phi_x + du_dy * phi_y - source_f(xg, yg) * phi

A, b = weak.assemble(domain, target="fem_system")

A_dense = to_dense(A)
b_dense = jnp.asarray(b)

u_fem = jnp.linalg.solve(A_dense, b_dense).reshape(-1)

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

lin_res = jnp.linalg.norm(A_dense @ u_fem - b_dense) / (
    jnp.linalg.norm(b_dense) + 1e-14
)

coords = np.asarray(domain.mesh.points)[:, :2]
x = jnp.asarray(coords[:, 0:1])
y = jnp.asarray(coords[:, 1:2])

u_exact = exact_u_num(x, y).reshape(-1)

rel_l2 = jnp.linalg.norm(u_exact - u_fem) / (jnp.linalg.norm(u_exact) + 1e-14)
max_abs = jnp.max(jnp.abs(u_exact - u_fem))

print("\n" + "=" * 70)
print("2D Poisson FEAX-FEM example")
print("=" * 70)
print(f"Number of FEM DOFs       : {u_fem.shape[0]}")
print(f"Linear solve residual    : {float(lin_res):.6e}")
print(f"Relative L2 error        : {float(rel_l2):.6e}")
print(f"Maximum absolute error   : {float(max_abs):.6e}")

assert float(lin_res) < 1e-5
assert float(rel_l2) < 5e-1