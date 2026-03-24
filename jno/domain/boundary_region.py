from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoundaryRegion:
    tag: str
    dim: int
    points: np.ndarray
    edges: Optional[np.ndarray] = None
    triangles: Optional[np.ndarray] = None
    tol: float = 1e-8

    def contains(self, p):
        import jax.numpy as jnp

        p = jnp.asarray(p)[: self.dim]

        # 2D: segment membership
        if self.dim == 2 and self.edges is not None and len(self.edges) > 0:
            a = jnp.asarray(self.edges[:, 0, :])  # (E,2)
            b = jnp.asarray(self.edges[:, 1, :])  # (E,2)

            ab = b - a
            ap = p[None, :] - a

            ab_len2 = jnp.sum(ab * ab, axis=1)
            ab_len2 = jnp.maximum(ab_len2, 1e-30)

            t = jnp.sum(ap * ab, axis=1) / ab_len2
            t = jnp.clip(t, 0.0, 1.0)

            proj = a + t[:, None] * ab
            dist2 = jnp.sum((proj - p[None, :]) ** 2, axis=1)
            return jnp.any(dist2 <= self.tol * self.tol)

        # 3D: triangle membership
        if self.dim == 3 and self.triangles is not None and len(self.triangles) > 0:
            a = jnp.asarray(self.triangles[:, 0, :])  # (T,3)
            b = jnp.asarray(self.triangles[:, 1, :])
            c = jnp.asarray(self.triangles[:, 2, :])

            ab = b - a
            ac = c - a
            ap = p[None, :] - a

            n = jnp.cross(ab, ac)
            n_norm = jnp.linalg.norm(n, axis=1)
            n_norm = jnp.maximum(n_norm, 1e-30)

            plane_dist = jnp.abs(jnp.sum(ap * n, axis=1)) / n_norm

            d00 = jnp.sum(ab * ab, axis=1)
            d01 = jnp.sum(ab * ac, axis=1)
            d11 = jnp.sum(ac * ac, axis=1)
            d20 = jnp.sum(ap * ab, axis=1)
            d21 = jnp.sum(ap * ac, axis=1)

            denom = d00 * d11 - d01 * d01
            denom = jnp.maximum(denom, 1e-30)

            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w

            inside = (u >= -1e-8) & (v >= -1e-8) & (w >= -1e-8)
            return jnp.any((plane_dist <= self.tol) & inside)

        # fallback only if no explicit entities are available
        pts = jnp.asarray(self.points[:, : self.dim])
        d = jnp.linalg.norm(pts - p[None, :], axis=1)
        return jnp.any(d <= self.tol)
