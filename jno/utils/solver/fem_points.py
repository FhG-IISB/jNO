"""Points inside a Lagrange mesh: which element holds a point, and that element's shape functions there.

This is what a point load ``P * v(px, py, pz)`` and a point value ``u(px, py, pz)`` are made of: the
point's containing element is found on the host (it is structure -- an integer), and the point's
reference coordinates and the shape-function values are computed in jax from the element's vertex
positions, so they are differentiable in the point and in the mesh.

Scope: order-1 Lagrange cells (interval, triangle, quadrilateral, tetrahedron, hexahedron), the local
vertex order basix uses. The derivative with respect to the point is continuous INSIDE an element and
jumps across element faces, as it does for any point load on C0 elements.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# basix's reference vertices: the local order `cells_f` is written in
_REF = {
    "interval": np.array([[0.0], [1.0]]),
    "triangle": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    "quadrilateral": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    "tetrahedron": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    "hexahedron": np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=float),
}
_SIMPLEX = {"interval", "triangle", "tetrahedron"}
_MESHIO = {"line": "interval", "tetra": "tetrahedron", "quad": "quadrilateral"}  # meshio's names -> basix's


def _shape(cell_type: str, xi):
    """Order-1 shape functions ``(n, nv)`` and their reference gradients ``(n, nv, dim)`` at ``xi`` ``(n, dim)``."""
    if cell_type in _SIMPLEX:
        dim = xi.shape[1]
        N = jnp.concatenate([1.0 - xi.sum(1, keepdims=True), xi], axis=1)
        dN = jnp.concatenate([-jnp.ones((1, dim)), jnp.eye(dim)], axis=0)  # constant
        return N, jnp.broadcast_to(dN, (xi.shape[0],) + dN.shape)
    R = jnp.asarray(_REF[cell_type])  # (nv, dim) of 0/1: tensor-product factors
    f = jnp.where(R[None] > 0.5, xi[:, None, :], 1.0 - xi[:, None, :])  # (n, nv, dim)
    df = jnp.where(R[None] > 0.5, 1.0, -1.0) * jnp.ones_like(f)
    N = jnp.prod(f, axis=-1)
    dim = xi.shape[1]
    grads = []
    for a in range(dim):  # d/dxi_a of the product: the factor a differentiated, the rest kept
        g = df[..., a]
        for b in range(dim):
            if b != a:
                g = g * f[..., b]
        grads.append(g)
    return N, jnp.stack(grads, axis=-1)


class PointShape:
    """``(cell, N)`` for points of a fixed mesh: ``cell`` ``(n,)`` the containing element's index into
    ``cells_basix``, ``N`` ``(n, nv)`` its shape functions at the points.

    ``pts`` ``(n_nodes, >=dim)`` the vertex coordinates, ``cells_vtk`` / ``cells_basix`` the same cells
    in the mesh's own vertex order (for the search) and in basix order (for the shape functions)."""

    def __init__(self, cell_type: str, pts, cells_vtk, cells_basix, dim: int):
        cell_type = _MESHIO.get(cell_type, cell_type)
        if cell_type not in _REF:
            raise ValueError(f"jno.fem: a point term needs order-1 Lagrange cells; this mesh has {cell_type!r}.")
        nv = _REF[cell_type].shape[0]
        if np.asarray(cells_basix).shape[1] != nv:
            raise ValueError(
                f"jno.fem: a point term is implemented for ORDER-1 Lagrange fields only (a {cell_type} cell "
                f"has {nv} vertices; this field has {np.asarray(cells_basix).shape[1]} local nodes)."
            )
        self.cell_type, self.dim = cell_type, int(dim)
        self.pts = np.asarray(pts, float)[:, : self.dim]
        self.cells_vtk = np.asarray(cells_vtk)
        self.cells_basix = jnp.asarray(cells_basix, dtype=jnp.int32)
        self.lo, self.hi = self.pts.min(0), self.pts.max(0)

    def _locate(self, pos):
        """Host: the containing cell of every point; raises for a point outside the mesh."""
        from .fem_adapt import _locate_in_cells

        pos = np.asarray(pos, float).reshape(-1, self.dim)
        cell, _w, _ref, inside = _locate_in_cells(self.pts, self.cells_vtk, pos, tol=1e-9, k=16)
        if not np.all(inside):
            bad = pos[~np.asarray(inside)]
            raise ValueError(
                f"jno.fem: {len(bad)} point(s) lie outside the mesh, e.g. {bad[0].tolist()}; the mesh spans "
                f"{self.lo.tolist()} .. {self.hi.tolist()}. A point load or a point value needs a point inside it."
            )
        return np.asarray(cell, dtype=np.int32)

    def __call__(self, pos, pts=None):
        """``(cell, N)`` at ``pos`` ``(n, dim)`` (traced ok). ``pts``: the vertex positions to use (traced
        for a moving mesh), default the build's."""
        pos = jnp.asarray(pos)
        n = pos.shape[0]
        cell = jax.pure_callback(
            self._locate, jax.ShapeDtypeStruct((n,), jnp.int32), jax.lax.stop_gradient(pos), vmap_method="sequential"
        )
        P = jnp.asarray(self.pts if pts is None else pts)[:, : self.dim]
        X = P[self.cells_basix[cell]]  # (n, nv, dim)
        if self.cell_type in _SIMPLEX:
            # affine: x = X0 + (X_i - X0) xi, one linear solve
            T = jnp.swapaxes(X[:, 1:, :] - X[:, :1, :], 1, 2)  # (n, dim, dim)
            xi = jnp.linalg.solve(T, (pos - X[:, 0, :])[..., None])[..., 0]
        else:
            # bi/trilinear: Newton from the centre; exact in one step on an axis-aligned box
            xi = jnp.full((n, self.dim), 0.5, dtype=pos.dtype)
            for _ in range(8):
                N, dN = _shape(self.cell_type, xi)
                x = jnp.einsum("nv,nvd->nd", N, X)
                J = jnp.einsum("nva,nvd->nda", dN, X)  # dx_d / dxi_a
                xi = xi - jnp.linalg.solve(J, (x - pos)[..., None])[..., 0]
        N, _ = _shape(self.cell_type, xi)
        return cell, N
