import jax

# jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import lineax as lx
import optax
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

import jno

import foundax
import numpy as np
from jno import LearningRateSchedule as lrs

"""04 - 3-D Helmholtz equation on an F-shaped domain with FEM and variational PINNs

Problem
-------
    -Delta u - k^2 u = f    in Omega

Geometry
--------
    Omega is a 3-D extrusion of a 2-D letter F.

Boundary conditions
-------------------
    u = 0      on the bottom face
    du/dn = g  on the top face and side walls

Analytical solution
-------------------
    u(x, y, z) = z + alpha sin(pi z)
"""
alpha = 0.2
k_val = 4.0


# -----------------------------------------------------------------------------
# Geometry helper
# -----------------------------------------------------------------------------
def letter_F_3d(depth=1.0, mesh_size=0.18):
    def construct(geo):
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

        pts = [geo.add_point([x, y, 0.0], mesh_size=mesh_size) for (x, y) in outline_xy]
        lines = [geo.add_line(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        loop = geo.add_curve_loop(lines)
        bottom_surface = geo.add_plane_surface(loop)
        extruded = geo.extrude(bottom_surface, [0.0, 0.0, depth])

        def flatten(items):
            out = []
            for item in items:
                if isinstance(item, (list, tuple)):
                    out.extend(flatten(item))
                else:
                    out.append(item)
            return out

        flat = flatten(extruded)
        surfaces = [e for e in flat if hasattr(e, "dim") and e.dim == 2]
        volumes = [e for e in flat if hasattr(e, "dim") and e.dim == 3]

        if len(volumes) != 1:
            raise RuntimeError(f"Expected one volume, got {len(volumes)}")
        if len(surfaces) < 1:
            raise RuntimeError("No surface entities returned by extrusion.")

        top_surface = surfaces[0]
        side_surfaces = surfaces[1:]
        all_surfaces = [bottom_surface, top_surface] + side_surfaces

        geo.add_physical(volumes[0], "interior")
        geo.add_physical(all_surfaces, "boundary")
        geo.add_physical([bottom_surface], "bottom")
        geo.add_physical([top_surface], "top")
        geo.add_physical(side_surfaces, "wall")
        return geo, 3, mesh_size

    return construct


# -----------------------------------------------------------------------------
# Manufactured solution
# -----------------------------------------------------------------------------
def exact_u(x, y, z):
    return z + alpha * jno.np.sin(jno.np.pi * z)


def exact_u_num(x, y, z):
    return z + alpha * jnp.sin(jnp.pi * z)


def source_f(x, y, z):
    return alpha * (jno.np.pi**2) * jno.np.sin(jno.np.pi * z) - (k_val**2) * (z + alpha * jno.np.sin(jno.np.pi * z))


def flux_top(x, y, z):
    return 0.0 * x + (1.0 - alpha * jno.np.pi)


def flux_wall(x, y, z):
    return 0.0 * x


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def extract_boundary_faces_from_tetra(mesh):
    pts = np.asarray(mesh.points)
    tet = np.asarray(mesh.cells_dict["tetra"], dtype=int)
    faces = np.vstack(
        [
            tet[:, [0, 1, 2]],
            tet[:, [0, 1, 3]],
            tet[:, [0, 2, 3]],
            tet[:, [1, 2, 3]],
        ]
    )
    faces_sorted = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(faces_sorted, axis=0, return_counts=True)
    boundary_faces = unique_faces[counts == 1]
    return pts[:, :3], boundary_faces


def plot_boundary_scalar(ax, pts, faces, nodal_values, title, cmap="viridis"):
    nodal_values = np.asarray(nodal_values).reshape(-1)
    verts = pts[faces]
    face_vals = nodal_values[faces].mean(axis=1)

    poly = Poly3DCollection(verts, linewidth=0.15, edgecolor="none", alpha=1.0, cmap=cmap)
    poly.set_array(face_vals)
    ax.add_collection3d(poly)

    xmin, ymin, zmin = pts.min(axis=0)[:3]
    xmax, ymax, zmax = pts.max(axis=0)[:3]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=35)
    return poly


# -----------------------------------------------------------------------------
# Coarse training domain
# -----------------------------------------------------------------------------
train_domain = jno.domain(constructor=letter_F_3d(depth=1.0, mesh_size=0.55))
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

net = jno.nn.wrap(foundax.mlp(
    3,
    hidden_dims=32,
    num_layers=4,
    activation=jax.nn.tanh,
    key=jax.random.PRNGKey(0),
))


def apply_hard_bc(u_pred, x, y, z):
    return z * u_pred


u_gauss = apply_hard_bc(net(xg, yg, zg), xg, yg, zg)
u_int = apply_hard_bc(net(x_int, y_int, z_int), x_int, y_int, z_int)

du_dx = jno.np.grad(u, xg)
du_dy = jno.np.grad(u, yg)
du_dz = jno.np.grad(u, zg)
phi_x = jno.np.grad(phi, xg)
phi_y = jno.np.grad(phi, yg)
phi_z = jno.np.grad(phi, zg)

vol_integrand = du_dx * phi_x + du_dy * phi_y + du_dz * phi_z - (k_val**2) * u * phi - source_f(xg, yg, zg) * phi

neumann_top = flux_top(xt, yt, zt) * phi
neumann_wall = flux_wall(xw, yw, zw) * phi
weak = vol_integrand - neumann_top - neumann_wall

pde = weak.assemble(train_domain, u_net=u_gauss, target="vpinn")

crux = jno.core(constraints=[pde.mse], domain=train_domain)
net.optimizer(optax.adam, lr=lrs.warmup_cosine(3, 1, 1e-3, 1e-5))
crux.solve(epochs=3)

u_pred, u_true = crux.eval([u_int, exact_u(x_int, y_int, z_int)])
rel_l2 = float(jnp.linalg.norm(u_pred - u_true) / (jnp.linalg.norm(u_true) + 1e-8))
assert rel_l2 < 1.1, f"relative L2 error too large: {rel_l2:.3e}"
