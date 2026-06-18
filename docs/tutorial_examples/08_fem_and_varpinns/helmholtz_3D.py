"""05 - 3D screened-Helmholtz solve on an extruded F-shaped domain via ``jno.fem``.

    -Delta u + sigma u = f      on a 3D "F" prism (gmsh extrusion)      (sigma = 4)
    u = 0           on the bottom face        (Dirichlet)
    du/dn = 1 - alpha pi  on the top face      (Neumann)
    du/dn = 0       on the side walls          (natural)

Shows TET4 assembly on a non-trivial 3-D geometry with mixed Dirichlet / Neumann faces.
Manufactured solution  u(x, y, z) = z + alpha sin(pi z).
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"  # FEM surface arrays are CPU-pinned

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jno  # noqa: E402

alpha, sigma, MS = 0.20, 4.0, 0.40
exact = lambda z: z + alpha * jnp.sin(jnp.pi * z)  # noqa: E731
dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731


def letter_F_3d(depth=1.0, mesh_size=MS):
    """A jNO/gmsh constructor for a 3-D extruded "F" (tags: interior, bottom, top, wall)."""
    outline = [
        (0, 0),
        (0.35, 0),
        (0.35, 0.9),
        (0.9, 0.9),
        (0.9, 1.2),
        (0.35, 1.2),
        (0.35, 1.65),
        (1.2, 1.65),
        (1.2, 2.0),
        (0, 2.0),
    ]

    def construct(geo):
        pts = [geo.add_point([x, y, 0.0], mesh_size=mesh_size) for x, y in outline]
        loop = geo.add_curve_loop([geo.add_line(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))])
        base = geo.add_plane_surface(loop)
        flat = []  # flatten the extrusion result (volume + side/top surfaces)

        def _flat(xs):
            [(_flat(x) if isinstance(x, (list, tuple)) else flat.append(x)) for x in xs]

        _flat(geo.extrude(base, [0.0, 0.0, depth], num_layers=8))  # 8 layers resolve sin(pi z) in z
        surfaces = [e for e in flat if getattr(e, "dim", None) == 2]
        volume = next(e for e in flat if getattr(e, "dim", None) == 3)
        geo.add_physical(volume, "interior")
        geo.add_physical([base] + surfaces, "boundary")
        geo.add_physical([base], "bottom")
        geo.add_physical([surfaces[0]], "top")
        geo.add_physical(surfaces[1:], "wall")
        return geo, 3, mesh_size

    return construct


d = jno.domain(constructor=letter_F_3d(mesh_size=MS), compute_mesh_connectivity=True)
u, phi = d.fem_symbols()
ci, cb, ct = d.variable("interior", split=True), d.variable("bottom", split=True), d.variable("top", split=True)
xi, yi, zi = ci[0], ci[1], ci[2]
ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

f = alpha * np.pi**2 * jno.np.sin(np.pi * zi) + sigma * (zi + alpha * jno.np.sin(np.pi * zi))  # -Delta u* + sigma u*
volume = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z + sigma * u * vi - f * vi
top_neumann = -(1.0 - alpha * np.pi) * phi.bind(x=ct[0], y=ct[1], z=ct[2])  # du/dn given on the top face
fem = jno.fem([volume, top_neumann, u(cb[0], cb[1], cb[2]) - 0.0], element_type="TET4", quad_degree=2)

u_fem = jnp.linalg.solve(dense(fem.A), jnp.asarray(fem.b).reshape(-1))
pts = np.asarray(fem.points)
rel_l2 = float(jnp.linalg.norm(exact(pts[:, 2]) - u_fem) / jnp.linalg.norm(exact(pts[:, 2])))
print(f"\n3D F-domain screened Helmholtz via jno.fem: dofs={fem.dofs}  rel_L2={rel_l2:.3e}")
assert fem.is_linear and rel_l2 < 1.0e-2
