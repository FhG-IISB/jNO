import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import lineax as lx
import numpy as np

import jno
import jno.numpy as jnn

pi = jnp.pi
sin = jnp.sin
cos = jnp.cos


# ------------------------------------------------------------
# 1. Domain
# ------------------------------------------------------------
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.04))

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
xg, yg, _ = domain.variable("fem_gauss", split=True)

# ------------------------------------------------------------
# 2. Manufactured exact solution and coefficients
# ------------------------------------------------------------
def exact_u(x, y):
    return sin(pi * x) * sin(pi * y)

# scalar spatial diffusivity (this answers your "K = 2*x*y?" question)
kappa = 1.0 + xg + yg

# constant advection vector as tensor tag
beta = domain.variable("beta", jnp.array([[1.0, -0.5]]))   # shape (1, 2)

# constant scalar reaction as tensor tag
c = domain.variable("c", jnp.array([[0.2]]))               # shape (1, 1)

def source_f(x, y):
    return (
        -1.5 * pi * sin(pi * x) * cos(pi * y)
        + (2.0 * pi**2 * (1.0 + x + y) + 0.2) * sin(pi * x) * sin(pi * y)
    )

# ------------------------------------------------------------
# 3. Weak form
# ------------------------------------------------------------
grad_u = jnn.grad(u, (xg, yg))
grad_phi = jnn.grad(phi, (xg, yg))

diff_term  = kappa * jnn.inner(grad_u, grad_phi)
adv_term   = jnn.inner(beta, grad_u) * phi
react_term = c * u * phi
rhs_term   = source_f(xg, yg) * phi

weak = diff_term + adv_term + react_term - rhs_term

# ------------------------------------------------------------
# 4. Assemble and solve
# ------------------------------------------------------------
A, b = weak.assemble(domain, target="fem_system")

A_dense = jnp.asarray(A.toarray())
b_dense = jnp.asarray(b)

op = lx.MatrixLinearOperator(A_dense)
sol = lx.linear_solve(op, b_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_h = sol.value.reshape(-1)

# ------------------------------------------------------------
# 5. Error check at mesh nodes
# ------------------------------------------------------------
coords = np.asarray(domain.mesh.points)[:, :2]
x = jnp.asarray(coords[:, 0:1])
y = jnp.asarray(coords[:, 1:2])

u_ex = exact_u(x, y).reshape(-1)
rel_l2 = jnp.linalg.norm(u_ex - u_h) / jnp.linalg.norm(u_ex)
max_abs = jnp.max(jnp.abs(u_ex - u_h))

print(f"Relative L2 error: {rel_l2:.6e}")
print(f"Max abs error    : {max_abs:.6e}")