import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np
import optax
import lineax as lx
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs


# ============================================================
# 1. Geometry: 3D letter F
# ============================================================
def letter_F_3d(depth=1.0, mesh_size=0.18):
    """
    3D letter F by:
    1) defining one 2D F-shaped polygon in the x-y plane
    2) extruding it in z

    Tags:
      - interior
      - bottom
      - top
      - wall
      - boundary
    """

    def construct(geo):
        z0 = 0.0

        outline_xy = [
            (0.0, 0.0),
            (0.35, 0.0),
            (0.35, 0.90),
            (0.90, 0.90),
            (0.90, 1.20),
            (0.35, 1.20),
            (0.35, 1.65),
            (1.20, 1.65),
            (1.20, 2.00),
            (0.0, 2.00),
        ]

        pts = [
            geo.add_point([x, y, z0], mesh_size=mesh_size)
            for (x, y) in outline_xy
        ]

        lines = [
            geo.add_line(pts[i], pts[(i + 1) % len(pts)])
            for i in range(len(pts))
        ]

        loop = geo.add_curve_loop(lines)
        bottom_surface = geo.add_plane_surface(loop)

        extruded = geo.extrude(bottom_surface, [0.0, 0.0, depth])

        def _flatten_entities(items):
            flat = []
            for item in items:
                if isinstance(item, (list, tuple)):
                    flat.extend(_flatten_entities(item))
                else:
                    flat.append(item)
            return flat

        flat_extruded = _flatten_entities(extruded)

        surfaces = [e for e in flat_extruded if hasattr(e, "dim") and e.dim == 2]
        volumes = [e for e in flat_extruded if hasattr(e, "dim") and e.dim == 3]

        if len(volumes) != 1:
            raise RuntimeError(f"Expected exactly one volume from extrusion, got {len(volumes)}")

        if len(surfaces) < 1:
            raise RuntimeError("No surface entities returned by extrusion.")

        volume = volumes[0]

        # In this pygmsh version, first returned surface is the translated top surface
        top_surface = surfaces[0]
        side_surfaces = surfaces[1:]

        all_surfaces = [bottom_surface, top_surface] + side_surfaces

        geo.add_physical(volume, "interior")
        geo.add_physical(all_surfaces, "boundary")
        geo.add_physical([bottom_surface], "bottom")
        geo.add_physical([top_surface], "top")
        geo.add_physical(side_surfaces, "wall")

        return geo, 3, mesh_size

    return construct


# ============================================================
# 2. Manufactured Helmholtz solution
# ============================================================
# u(z) = z + alpha * sin(pi z)
# PDE: -Δu - k^2 u = f
# Since u depends only on z:
#   u''(z) = -alpha*pi^2*sin(pi z)
# => -Δu = alpha*pi^2*sin(pi z)
# => f = alpha*pi^2*sin(pi z) - k^2 * (z + alpha*sin(pi z))
#
# Bottom Dirichlet:
#   u(0)=0
# Top Neumann:
#   du/dn = du/dz at z=1 = 1 - alpha*pi
# Wall Neumann:
#   du/dn = 0 on vertical wall
# ============================================================
alpha = 0.2
k_val = 4.0


def exact_u(x, y, z): return z + alpha * jnn.sin(jnn.pi * z)


def exact_u_num(x, y, z): return z + alpha * jnp.sin(jnp.pi * z)


def source_f(x, y, z): return ( alpha * (jnn.pi ** 2) * jnn.sin(jnn.pi * z) - (k_val ** 2) * (z + alpha * jnn.sin(jnn.pi * z)) )


def flux_top(x, y, z): return 0.0 * x + (1.0 - alpha * jnn.pi)


def flux_wall(x, y, z): return 0.0 * x


# ============================================================
# 3. Helpers
# ============================================================
def extract_boundary_faces_from_tetra(mesh):
    pts = np.asarray(mesh.points)
    tet = np.asarray(mesh.cells_dict["tetra"], dtype=int)

    faces = np.vstack([
        tet[:, [0, 1, 2]],
        tet[:, [0, 1, 3]],
        tet[:, [0, 2, 3]],
        tet[:, [1, 2, 3]],
    ])

    faces_sorted = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(faces_sorted, axis=0, return_counts=True)
    boundary_faces = unique_faces[counts == 1]

    return pts[:, :3], boundary_faces


def plot_boundary_scalar(ax, pts, faces, nodal_values, title, cmap="viridis"):
    nodal_values = np.asarray(nodal_values).reshape(-1)
    verts = pts[faces]
    face_vals = nodal_values[faces].mean(axis=1)

    poly = Poly3DCollection(
        verts,
        linewidth=0.15,
        edgecolor="none",
        alpha=1.0,
        cmap=cmap,
    )
    poly.set_array(face_vals)

    ax.add_collection3d(poly)

    xmin, ymin, zmin = pts.min(axis=0)[:3]
    xmax, ymax, zmax = pts.max(axis=0)[:3]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=35)

    return poly


# ============================================================
# 4. Coarse training domain
# ============================================================
train_domain = jno.domain(
    constructor=letter_F_3d(depth=1.0, mesh_size=0.18)
)

train_domain.init_fem(
    element_type="TET4",
    quad_degree=2,
    bcs=[
        train_domain.dirichlet("bottom", 0.0),
        train_domain.neumann(["top", "wall"]),
    ],
    fem_solver=True,
)

u, phi = train_domain.fem_symbols()

xg, yg, zg, _ = train_domain.variable("fem_gauss", split=True)
xt, yt, zt, _ = train_domain.variable("gauss_top", split=True)
xw, yw, zw, _ = train_domain.variable("gauss_wall", split=True)

x_int, y_int, z_int, _ = train_domain.variable("interior", split=True)


# ============================================================
# 5. Neural network + hard BC ansatz
# ============================================================
key = jax.random.PRNGKey(0)

net = jnn.nn.mlp( 3, hidden_dims=128, num_layers=4,  activation=jax.nn.tanh,  key=key,)


def apply_hard_bc(u_pred, x, y, z):
    # bottom face z=0 -> u=0
    return z * u_pred


u_gauss = apply_hard_bc(net(xg, yg, zg), xg, yg, zg)
u_int = apply_hard_bc(net(x_int, y_int, z_int), x_int, y_int, z_int)


# ============================================================
# 6. Helmholtz weak form
# ============================================================
du_dx = jnn.grad(u, xg)
du_dy = jnn.grad(u, yg)
du_dz = jnn.grad(u, zg)

phi_x = jnn.grad(phi, xg)
phi_y = jnn.grad(phi, yg)
phi_z = jnn.grad(phi, zg)

k_sq = 0.0 * xg + k_val**2

vol_integrand = (
    du_dx * phi_x
    + du_dy * phi_y
    + du_dz * phi_z
    - k_sq * u * phi
    - source_f(xg, yg, zg) * phi
)

neumann_top = flux_top(xt, yt, zt) * phi
neumann_wall = flux_wall(xw, yw, zw) * phi

weak = vol_integrand - neumann_top - neumann_wall


# ============================================================
# 7. Assemble coarse VPINN and coarse FEM
# ============================================================
pde = weak.assemble(train_domain, u_net=u_gauss, target="vpinn")
A_coarse, b_coarse = weak.assemble(train_domain, target="fem_system")

A_coarse_dense = jnp.asarray(A_coarse.todense())
b_coarse_dense = jnp.asarray(b_coarse)

op = lx.MatrixLinearOperator(A_coarse_dense)
sol = lx.linear_solve(op, b_coarse_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem_coarse = sol.value.reshape(-1)

lin_res = jnp.linalg.norm(A_coarse_dense @ u_fem_coarse - b_coarse_dense) / (
    jnp.linalg.norm(b_coarse_dense) + 1e-14
)
print(f"Coarse FEM linear solve residual: {lin_res:.6e}")


# ============================================================
# 8. Train VPINN
# ============================================================
val_error = jnn.abs(u_int - exact_u(x_int, y_int, z_int))
error_tracker = jnn.tracker(val_error, interval=100)

crux = jno.core(constraints=[pde.mse, error_tracker], domain=train_domain)

learning_rate = lrs.warmup_cosine(3000, 200, 1e-3, 1e-5)
net.optimizer(optax.adam, lr=learning_rate)
crux.solve(epochs=3000)


# ============================================================
# 9. Fine FEM domain
# ============================================================
fem_domain = jno.domain(
    constructor=letter_F_3d(depth=1.0, mesh_size=0.10)
)

fem_domain.init_fem(
    element_type="TET4",
    quad_degree=2,
    neumann_tags=["top", "wall"],
    dirichlet_tags=["bottom"],
    dirichlet_value_fns={
        "bottom": lambda p: 0.0,
    },
    fem_solver=True,
)

u_fem_sym, phi_fem_sym = fem_domain.fem_symbols()

xg_fem, yg_fem, zg_fem, _ = fem_domain.variable("fem_gauss", split=True)
xt_fem, yt_fem, zt_fem, _ = fem_domain.variable("gauss_top", split=True)
xw_fem, yw_fem, zw_fem, _ = fem_domain.variable("gauss_wall", split=True)

du_dx_fem = jnn.grad(u_fem_sym, xg_fem)
du_dy_fem = jnn.grad(u_fem_sym, yg_fem)
du_dz_fem = jnn.grad(u_fem_sym, zg_fem)

phi_x_fem = jnn.grad(phi_fem_sym, xg_fem)
phi_y_fem = jnn.grad(phi_fem_sym, yg_fem)
phi_z_fem = jnn.grad(phi_fem_sym, zg_fem)

k_sq_fem = 0.0 * xg_fem + k_val**2

vol_integrand_fem = (
    du_dx_fem * phi_x_fem
    + du_dy_fem * phi_y_fem
    + du_dz_fem * phi_z_fem
    - k_sq_fem * u_fem_sym * phi_fem_sym
    - source_f(xg_fem, yg_fem, zg_fem) * phi_fem_sym
)

neumann_top_fem = flux_top(xt_fem, yt_fem, zt_fem) * phi_fem_sym
neumann_wall_fem = flux_wall(xw_fem, yw_fem, zw_fem) * phi_fem_sym

weak_fem = vol_integrand_fem - neumann_top_fem - neumann_wall_fem

A_fine, b_fine = weak_fem.assemble(fem_domain, target="fem_system")

A_fine_dense = jnp.asarray(A_fine.todense())
b_fine_dense = jnp.asarray(b_fine)

op_fine = lx.MatrixLinearOperator(A_fine_dense)
sol_fine = lx.linear_solve(op_fine, b_fine_dense, solver=lx.AutoLinearSolver(well_posed=True))
u_fem_fine = sol_fine.value.reshape(-1)

lin_res_fine = jnp.linalg.norm(A_fine_dense @ u_fem_fine - b_fine_dense) / (
    jnp.linalg.norm(b_fine_dense) + 1e-14
)
print(f"Fine FEM linear solve residual: {lin_res_fine:.6e}")


# ============================================================
# 10. Evaluate VPINN on fine domain
# ============================================================
x_eval, y_eval, z_eval, _ = fem_domain.variable("interior", split=True)

u_vpinn_eval = crux.eval(
    apply_hard_bc(net(x_eval, y_eval, z_eval), x_eval, y_eval, z_eval),
    domain=fem_domain,
)
u_true_eval = crux.eval(
    exact_u(x_eval, y_eval, z_eval),
    domain=fem_domain,
)

u_vpinn_eval = jnp.asarray(u_vpinn_eval).reshape(-1)
u_true_eval = jnp.asarray(u_true_eval).reshape(-1)

rel_l2_vpinn = jnp.linalg.norm(u_true_eval - u_vpinn_eval) / (jnp.linalg.norm(u_true_eval) + 1e-14)
max_abs_vpinn = jnp.max(jnp.abs(u_true_eval - u_vpinn_eval))

print(f"VPINN Relative L2 Error on fine domain: {rel_l2_vpinn:.6e}")
print(f"VPINN Max Abs Error on fine domain:     {max_abs_vpinn:.6e}")


# ============================================================
# 11. Fine FEM error
# ============================================================
coords_mesh = np.asarray(fem_domain.mesh.points)[:, :3]
x_m = jnp.asarray(coords_mesh[:, 0:1])
y_m = jnp.asarray(coords_mesh[:, 1:2])
z_m = jnp.asarray(coords_mesh[:, 2:3])

u_exact_fem = exact_u_num(x_m, y_m, z_m).reshape(-1)
u_fem_vec = jnp.asarray(u_fem_fine).reshape(-1)

rel_l2_fem = jnp.linalg.norm(u_exact_fem - u_fem_vec) / (jnp.linalg.norm(u_exact_fem) + 1e-14)
max_abs_fem = jnp.max(jnp.abs(u_exact_fem - u_fem_vec))

print(f"Fine FEM Relative L2 Error: {rel_l2_fem:.6e}")
print(f"Fine FEM Max Abs Error:     {max_abs_fem:.6e}")


# ============================================================
# 12. Prepare plotting values on mesh boundary
# ============================================================
pts, boundary_faces = extract_boundary_faces_from_tetra(fem_domain.mesh)

# exact/FEM directly on mesh nodes
u_true_mesh = np.asarray(u_exact_fem).reshape(-1)
u_fem_mesh = np.asarray(u_fem_vec).reshape(-1)
abs_err_fem = np.abs(u_fem_mesh - u_true_mesh)

# VPINN is evaluated on interior sample points; map to mesh nodes by nearest neighbor
x_int_plot = np.asarray(crux.eval(x_eval, domain=fem_domain)).reshape(-1)
y_int_plot = np.asarray(crux.eval(y_eval, domain=fem_domain)).reshape(-1)
z_int_plot = np.asarray(crux.eval(z_eval, domain=fem_domain)).reshape(-1)

coords_int = np.column_stack([x_int_plot, y_int_plot, z_int_plot])
tree = cKDTree(coords_int)

_, nn_idx = tree.query(pts[:, :3], k=1)
u_vpinn_mesh = np.asarray(u_vpinn_eval).reshape(-1)[nn_idx]
abs_err_vpinn = np.abs(u_vpinn_mesh - u_true_mesh)


# ============================================================
# 13. Plot
# ============================================================
fig = plt.figure(figsize=(20, 5))

axes = [
    fig.add_subplot(1, 4, 1, projection="3d"),
    fig.add_subplot(1, 4, 2, projection="3d"),
    fig.add_subplot(1, 4, 3, projection="3d"),
    fig.add_subplot(1, 4, 4, projection="3d"),
]

m0 = plot_boundary_scalar(axes[0], pts, boundary_faces, u_true_mesh, "True solution", cmap="viridis")
m1 = plot_boundary_scalar(axes[1], pts, boundary_faces, u_vpinn_mesh, "VPINN solution", cmap="viridis")
m2 = plot_boundary_scalar(axes[2], pts, boundary_faces, abs_err_vpinn, "VPINN abs error", cmap="magma")
m3 = plot_boundary_scalar(axes[3], pts, boundary_faces, u_fem_mesh, "FEM solution", cmap="viridis")

fig.colorbar(m0, ax=axes[0], shrink=0.75)
fig.colorbar(m1, ax=axes[1], shrink=0.75)
fig.colorbar(m2, ax=axes[2], shrink=0.75)
fig.colorbar(m3, ax=axes[3], shrink=0.75)

plt.tight_layout()
plt.savefig("helmholtz_F3D_vpinn_fem.png", dpi=300)