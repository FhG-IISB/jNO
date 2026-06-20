# 3D Helmholtz on an F-shaped Domain (FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/helmholtz_3D.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

A 3D screened-Helmholtz solve $-\Delta u + \sigma u = f$ on a non-trivial geometry — an
extruded "F" prism meshed with `gmsh` into TET4 elements — with a Dirichlet bottom, a Neumann
top, and natural side walls.

## Same API, one more coordinate

3D uses the identical workflow with a `z` axis: bind `z=zi`, take `ui.z`, and write the
Dirichlet condition over three coordinates. The geometry is any jNO/`gmsh` constructor:

```python
d = jno.domain(constructor=letter_F_3d(mesh_size=0.4), compute_mesh_connectivity=True)
ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
volume = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z + sigma * u * vi - f * vi
fem = jno.fem([volume, top_neumann, u(cb[0], cb[1], cb[2]) - 0.0], element_type="TET4", quad_degree=2)
```

## What to notice

- 3D is the same `jno.fem` API with `z` added — no special path.
- The extrusion uses `num_layers=8` so the through-thickness mode is resolved.
- Recovers $u^\*=z+\alpha\sin(\pi z)$ to rel-$L^2 \approx 2\times10^{-3}$.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/helmholtz_3D.py"
```
