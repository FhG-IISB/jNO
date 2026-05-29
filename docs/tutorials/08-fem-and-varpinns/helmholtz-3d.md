# Helmholtz 3D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/helmholtz_3D.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

This is the most geometrically complex example in the tutorial set: a 3D Helmholtz problem on an extruded letter-F style geometry.

## Problem Setup

The script works with mixed boundary conditions on a 3D domain and compares VPINN and FEM-style weak-form ideas on a nontrivial mesh.

## Step 1: Build a Custom 3D Geometry

A custom geometry constructor creates the volume and tagged surfaces required for boundary conditions.

```python
def letter_F_3d(depth=1.0, mesh_size=0.55):
    def construct(geo):
        outline_xy = [
            (0.0, 0.0), (0.35, 0.0), (0.35, 0.90), (0.90, 0.90),
            (0.90, 1.20), (0.35, 1.20), (0.35, 1.65), (1.20, 1.65),
            (1.20, 2.00), (0.0, 2.00),
        ]
        pts    = [geo.add_point([x, y, 0.0], mesh_size=mesh_size) for (x, y) in outline_xy]
        lines  = [geo.add_line(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        loop   = geo.add_curve_loop(lines)
        bottom = geo.add_plane_surface(loop)
        extruded = geo.extrude(bottom, [0.0, 0.0, depth])
        # tag physical groups
        geo.add_physical(volumes[0], "interior")
        geo.add_physical([bottom, top] + side_surfaces, "boundary")
        geo.add_physical([bottom], "bottom")
        geo.add_physical([top], "top")
        geo.add_physical(side_surfaces, "wall")
        return geo, 3, mesh_size
    return construct

domain = jno.domain(
    constructor=letter_F_3d(depth=1.0, mesh_size=0.55),
    compute_mesh_connectivity=True,
)
domain.init_fem(
    element_type="TET4",
    quad_degree=2,
    bcs=[
        domain.dirichlet("bottom", 0.0),
        domain.neumann(["top", "wall"]),
    ],
    fem_solver=True,
)
```

## Step 2: Assemble Weak-Form Quantities in 3D

The example uses tetrahedral-style weak-form machinery rather than a pointwise PDE residual.

```python
u, phi = domain.fem_symbols()
xg, yg, zg, _ = domain.variable("fem_gauss", split=True)
xt, yt, zt, _ = domain.variable("gauss_top",  split=True)
xw, yw, zw, _ = domain.variable("gauss_wall", split=True)

ux = jno.np.grad(u, xg); uy = jno.np.grad(u, yg); uz = jno.np.grad(u, zg)
phix = jno.np.grad(phi, xg); phiy = jno.np.grad(phi, yg); phiz = jno.np.grad(phi, zg)

volume        = ux*phix + uy*phiy + uz*phiz + sigma*u*phi - source_f(xg, yg, zg)*phi
top_boundary  = flux_top(xt, yt, zt) * phi
wall_boundary = flux_wall(xw, yw, zw) * phi
weak = volume - top_boundary - wall_boundary

A, b  = weak.assemble(domain, target="fem_system")
u_fem = jnp.linalg.solve(to_dense(A), jnp.asarray(b)).reshape(-1)
```

## Step 3: Visualize the 3D Result

The script includes a surface or boundary visualization pipeline so the final field can be interpreted geometrically.

```python
coords  = np.asarray(domain.mesh.points[:, :3])
x_nodes = jnp.asarray(coords[:, 0:1])
y_nodes = jnp.asarray(coords[:, 1:2])
z_nodes = jnp.asarray(coords[:, 2:3])

u_exact = exact_u_jax(x_nodes, y_nodes, z_nodes).reshape(-1)
rel_l2  = jnp.linalg.norm(u_fem - u_exact) / (jnp.linalg.norm(u_exact) + 1e-14)
print(f"Relative L2 error (FEM): {float(rel_l2):.6e}")
```

## What To Notice

- This is a high-end tutorial example rather than a first learning example.
- Complex geometry handling is one of the main reasons weak-form approaches become valuable.
- The workflow shows how jNO scales beyond unit-interval and unit-square toy problems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/helmholtz_3D.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to 08 FEM and Variational PINNs</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/helmholtz_3D.py"
```
