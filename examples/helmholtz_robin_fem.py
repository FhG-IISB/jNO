import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np
import lineax as lx
import matplotlib.pyplot as plt

import jno
import jno.numpy as jnn

pi = jnn.pi
sin = jnn.sin
cos = jnn.cos

k_val = 4.0
alpha_right = 2.0
alpha_top = 3.0


# ============================================================
# Exact solution and forcing
# u(x,y) = x sin(pi y) + y
# ============================================================
def exact_u(x, y):
    return x * sin(pi * y) + y


def exact_u_num(x, y):
    return x * jnp.sin(jnp.pi * y) + y


def source_f(x, y):
    return x * (pi**2) * sin(pi * y) - (k_val**2) * (x * sin(pi * y) + y)


def robin_rhs_right(x, y):
    # du/dn + alpha u on x=1
    return sin(pi * y) + alpha_right * (sin(pi * y) + y)


def robin_rhs_top(x, y):
    # du/dn + alpha u on y=1
    # u_y(x,1)=1-pi x, u(x,1)=1
    return 1.0 - pi * x + alpha_top


def to_dense(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    return jnp.asarray(A)


# ============================================================
# FEM domain
# ============================================================
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))
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

u, phi = domain.fem_symbols()

# Volume quadrature
xg, yg, _ = domain.variable("fem_gauss", split=True)

# Boundary quadrature
xr, yr, _ = domain.variable("gauss_right", split=True)
xt, yt, _ = domain.variable("gauss_top", split=True)

du_dx = jnn.grad(u, xg)
du_dy = jnn.grad(u, yg)
phi_x = jnn.grad(phi, xg)
phi_y = jnn.grad(phi, yg)

k_sq = 0.0 * xg + k_val**2
alpha_r = 0.0 * xr + alpha_right
alpha_t = 0.0 * xt + alpha_top

# ============================================================
# Unified weak form
#
# Strong form:
#   -Δu - k^2 u = f
#
# Robin on Γ_R:
#   du/dn + α u = r
#
# Weak form:
#   ∫Ω (∇u·∇phi - k^2 u phi - f phi) dΩ
# + ∫Γ_R (α u phi - r phi) dΓ
# = 0
# ============================================================
vol_integrand = (
    du_dx * phi_x
    + du_dy * phi_y
    - k_sq * u * phi
    - source_f(xg, yg) * phi
)

robin_right = alpha_r * u * phi - robin_rhs_right(xr, yr) * phi
robin_top = alpha_t * u * phi - robin_rhs_top(xt, yt) * phi

weak = vol_integrand + robin_right + robin_top

A, b = weak.assemble(domain, target="fem_system")

A_dense = to_dense(A)
b_dense = jnp.asarray(b)

op = lx.MatrixLinearOperator(A_dense)
sol = lx.linear_solve(op, b_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem = sol.value.reshape(-1)

lin_res = jnp.linalg.norm(A_dense @ u_fem - b_dense) / (jnp.linalg.norm(b_dense) + 1e-14)
print(f"Robin FEM linear solve residual: {lin_res:.6e}")

# ============================================================
# Error
# ============================================================
coords = np.asarray(domain.mesh.points)[:, :2]
x = jnp.asarray(coords[:, 0:1])
y = jnp.asarray(coords[:, 1:2])

u_exact = exact_u_num(x, y).reshape(-1)
rel_l2 = jnp.linalg.norm(u_exact - u_fem) / (jnp.linalg.norm(u_exact) + 1e-14)
max_abs = jnp.max(jnp.abs(u_exact - u_fem))

print(f"Robin FEM Relative L2 Error: {rel_l2:.6e}")
print(f"Robin FEM Max Abs Error:     {max_abs:.6e}")

# ============================================================
# Plot
# ============================================================
tri = domain.mesh.cells_dict["triangle"]
u_exact_np = np.asarray(u_exact).reshape(-1)
u_fem_np = np.asarray(u_fem).reshape(-1)
abs_err = np.abs(u_fem_np - u_exact_np)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_field(ax, vals, title, cmap="viridis"):
    im = ax.tripcolor(coords[:, 0], coords[:, 1], tri, vals, shading="flat", cmap=cmap)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax)

plot_field(axes[0], u_exact_np, "Exact")
plot_field(axes[1], u_fem_np, "FEM solution")
plot_field(axes[2], abs_err, "FEM abs error", cmap="magma")

plt.tight_layout()
plt.savefig("helmholtz_robin_fem_only.png", dpi=300)