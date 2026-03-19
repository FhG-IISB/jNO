import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import lineax as lx
import numpy as np

import jno
import jno.numpy as jnn

# Symbolic ops for weak form construction
pi = jnn.pi
sin = jnn.sin
cos = jnn.cos


# ------------------------------------------------------------
# 1. Domain
# ------------------------------------------------------------
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.04))

domain.init_fem(
    element_type="TRI3",
    quad_degree=3,
    bcs=[
        domain.dirichlet("left", 0.0),
        domain.dirichlet("bottom",0.0),
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


# ------------------------------------------------------------
# 2. Manufactured exact solution and coefficients
# ------------------------------------------------------------
def exact_u(x, y):
    # symbolic version
    return sin(pi * x) * sin(pi * y)


def exact_u_num(x, y):
    # numeric version
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


# scalar spatial diffusivity
kappa = 1.0 + xg + yg

# constant advection vector as tensor tag
beta = domain.variable("beta", jnp.array([[1.0, -0.5]]))   # shape (1, 2)

# constant scalar reaction as tensor tag
c = domain.variable("c", jnp.array([[0.2]]))               # shape (1, 1)


def source_f(x, y):
    # For u = sin(pi x) sin(pi y),
    # -div(kappa grad u) + beta.grad(u) + c u = f
    #
    # kappa = 1 + x + y
    # beta = (1, -0.5)
    # c = 0.2
    return (
        -pi * sin(pi * y) * cos(pi * x)
        -pi * sin(pi * x) * cos(pi * y)
        + 2.0 * pi**2 * (1.0 + x + y) * sin(pi * x) * sin(pi * y)
        + pi * cos(pi * x) * sin(pi * y)
        - 0.5 * pi * sin(pi * x) * cos(pi * y)
        + 0.2 * sin(pi * x) * sin(pi * y)
    )


def flux_right(x, y):
    # symbolic Neumann flux on x=1:
    # g = kappa * grad(u).n = (2+y) * (-pi sin(pi y))
    return (2.0 + y) * (-pi * sin(pi * y))


def flux_top(x, y):
    # symbolic Neumann flux on y=1:
    # g = kappa * grad(u).n = (2+x) * (-pi sin(pi x))
    return (2.0 + x) * (-pi * sin(pi * x))


# ------------------------------------------------------------
# 3. Weak form
# ------------------------------------------------------------
grad_u = jnn.grad(u, (xg, yg))
grad_phi = jnn.grad(phi, (xg, yg))

diff_term = kappa * jnn.inner(grad_u, grad_phi)
adv_term = jnn.inner(beta, grad_u) * phi
react_term = c * u * phi
rhs_term = source_f(xg, yg) * phi

# Neumann boundary contributions
neumann_right = flux_right(xr, yr) * phi
neumann_top = flux_top(xt, yt) * phi

weak = diff_term + adv_term + react_term - rhs_term - neumann_right - neumann_top


# ------------------------------------------------------------
# 4. Assemble and solve
# ------------------------------------------------------------
A, b = weak.assemble(domain, target="fem_system")


def to_dense(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    return jnp.asarray(A)


A_dense = to_dense(A)
b_dense = jnp.asarray(b)

op = lx.MatrixLinearOperator(A_dense)
sol = lx.linear_solve(op, b_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_h = jnp.asarray(sol.value).reshape(-1)

lin_res = jnp.linalg.norm(A_dense @ u_h - b_dense) / (jnp.linalg.norm(b_dense) + 1e-14)
print(f"Linear solve residual: {lin_res:.6e}")


# ------------------------------------------------------------
# 5. Error check at mesh nodes
# ------------------------------------------------------------
coords = np.asarray(domain.mesh.points)[:, :2]
x = jnp.asarray(coords[:, 0:1])
y = jnp.asarray(coords[:, 1:2])

u_ex = exact_u_num(x, y).reshape(-1)
rel_l2 = jnp.linalg.norm(u_ex - u_h) / (jnp.linalg.norm(u_ex) + 1e-14)
max_abs = jnp.max(jnp.abs(u_ex - u_h))

print(f"Relative L2 error: {rel_l2:.6e}")
print(f"Max abs error    : {max_abs:.6e}")