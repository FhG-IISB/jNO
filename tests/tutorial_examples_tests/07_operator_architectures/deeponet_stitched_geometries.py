"""
DeepONet with stitched geometries (triangle + disk + rectangle).

This example shows how to build a mixed-geometry dataset by sampling
interior points from different meshes and stitching domains together
with the + operator. The operator here is synthetic: f(x,y) -> u(x,y)
with u = f + bias(x,y).
"""

from __future__ import annotations

import jax
import jno

import numpy as np
import optax

KEY = jax.random.PRNGKey(0)
SEED = 0
SAMPLES_PER_GEOM = 60
TARGET_POINTS = 256
N_MODES = 3
EPOCHS = 300
BATCH = 30
MESH_SIZE = 0.05
def triangle_geometry(vertices, mesh_size=MESH_SIZE):
    """Return a pygmsh constructor for a triangle."""

    def constructor(geo):
        pts = [geo.add_point([x, y], mesh_size=mesh_size) for x, y in vertices]
        lines = [
            geo.add_line(pts[0], pts[1]),
            geo.add_line(pts[1], pts[2]),
            geo.add_line(pts[2], pts[0]),
        ]
        loop = geo.add_curve_loop(lines)
        surface = geo.add_plane_surface(loop)

        geo.add_physical(surface, "interior")
        geo.add_physical(lines, "boundary")

        return geo, 2, mesh_size

    return constructor
def available_points(constructor) -> int:
    temp = jno.domain(constructor=constructor, compute_mesh_connectivity=False)
    pts = temp._mesh_pool["interior"]
    return int(pts.shape[0]) if pts.ndim == 2 else int(pts.shape[1])
def random_fourier_field(points: np.ndarray, rng: np.random.Generator, n_modes: int) -> np.ndarray:
    """Generate a smooth random field on points (B, N, 2)."""
    batch, n_pts, _ = points.shape
    coeffs = rng.standard_normal((batch, n_modes, n_modes)).astype(np.float32) * 0.3

    x = points[..., 0:1]
    y = points[..., 1:2]
    field = np.zeros((batch, n_pts), dtype=np.float32)
    for i in range(n_modes):
        for j in range(n_modes):
            basis = np.sin((i + 1) * np.pi * x) * np.sin((j + 1) * np.pi * y)
            field += coeffs[:, i, j][:, None] * basis[..., 0]
    return field
def build_domain(constructor, geom_id: int, rng: np.random.Generator, n_points: int) -> jno.domain:
    base = jno.domain(constructor=constructor, compute_mesh_connectivity=False)
    dom = SAMPLES_PER_GEOM * base

    dom.variable("interior", sample=(n_points, None))
    pts = dom.context["interior"][:, 0]  # (B, N, 2)

    f = random_fourier_field(pts, rng, N_MODES)
    bias = 0.2 * np.sin(2 * np.pi * pts[..., 0]) * np.cos(2 * np.pi * pts[..., 1])
    u = f + bias

    dom.variable("_f", f[:, None, :, None])
    dom.variable("_u", u[:, None, :, None])
    dom.add_tensor_tag("geom_id", np.full((SAMPLES_PER_GEOM, 1), geom_id, dtype=np.float32))

    return dom
def main():
    rng = np.random.default_rng(SEED)

    rect_ctor = jno.domain.rect(x_range=(0.0, 1.0), y_range=(0.0, 1.0), mesh_size=MESH_SIZE)
    disk_ctor = jno.domain.disk(center=(0.0, 0.0), radius=0.6, mesh_size=MESH_SIZE, num_points=48)
    tri_ctor = triangle_geometry(vertices=[(0.0, 0.0), (1.0, 0.0), (0.25, 0.9)], mesh_size=MESH_SIZE)

    n_points = min(
        TARGET_POINTS,
        available_points(rect_ctor),
        available_points(disk_ctor),
        available_points(tri_ctor),
    )

    dom_rect = build_domain(rect_ctor, geom_id=0, rng=rng, n_points=n_points)
    dom_disk = build_domain(disk_ctor, geom_id=1, rng=rng, n_points=n_points)
    dom_tri = build_domain(tri_ctor, geom_id=2, rng=rng, n_points=n_points)

    # Stitch domains together: mixed geometries in one dataset
    domain = dom_rect + dom_disk + dom_tri

    _f = domain.variable("_f")
    _u = domain.variable("_u")
    x, y, _ = domain.variable("interior")

    coords = jno.np.concat([x, y])[:, 0]  # (B, N, 2)
    f_vals = _f[:, 0]  # (B, N, 1)
    u_vals = _u[:, 0]  # (B, N, 1)

    model = jno.np.nn.deeponet(
        branch_type="mlp",
        trunk_type="mlp",
        combination_type="dot",
        n_sensors=n_points,
        sensor_channels=1,
        coord_dim=2,
        n_outputs=1,
        basis_functions=64,
        hidden_dim=128,
        n_layers=4,
        key=KEY,
    )

    # Evaluate DeepONet per-sample because coordinates differ per geometry
    u_pred = jax.vmap(lambda f_i, y_i: model(f_i, y_i), in_axes=(0, 0))(f_vals, coords)
    if u_pred.ndim == 2:
        u_pred = u_pred[..., None]

    crux = jno.core([(u_vals - u_pred).mse], domain)

    model.optimizer(
        optax.chain(optax.clip_by_global_norm(1e-3), optax.adamw(1e-3, weight_decay=1e-6)),
        lr=jno.schedule.learning_rate.cosine(EPOCHS, 5e-4, 1e-6),
    )
    crux.solve(
        epochs=EPOCHS,
        constraint_weights=jno.schedule.constraint([1]),
        batchsize=BATCH,
        checkpoint_gradients=False,
        offload_data=False,
)
