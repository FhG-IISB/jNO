from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# How many vertices a facet has decides which field it belongs in. A 3-D boundary facet is a triangle
# on a tetrahedral mesh and a quadrilateral on a hexahedral one, so the store cannot be picked from
# the dimension -- which is what left every hexahedral boundary region with no entities at all.
_FIELD_BY_ARITY = {2: "edges", 3: "triangles", 4: "quads"}

# Slack on the dimensionless edge-side test for a quad, and on the barycentric one for a triangle.
# The quad slack is the looser of the two because its test runs on quantities of order h^2 rather
# than on barycentric coordinates of order 1.
_SLACK_TRI = 1e-8
_SLACK_QUAD = 1e-6


@dataclass
class BoundaryRegion:
    tag: str
    dim: int
    points: np.ndarray
    edges: Optional[np.ndarray] = None
    triangles: Optional[np.ndarray] = None
    quads: Optional[np.ndarray] = None
    tol: float = 1e-8

    @classmethod
    def from_facets(cls, tag: str, dim: int, points: np.ndarray, ents, tol: float = 1e-8) -> "BoundaryRegion":
        """Build a region from one facet array ``(E, k, dim)``, storing it by its vertex count ``k``.

        The three callers that build a sub-region from a parent's facets used to choose the field with
        ``edges if dim == 2 else triangles``, so a quadrilateral facet was written into ``triangles``
        and read back as a triangle. Dispatching in one place makes that impossible.
        """
        ents = np.asarray(ents)
        field = _FIELD_BY_ARITY.get(int(ents.shape[1])) if ents.ndim == 3 else None
        if field is None:
            raise ValueError(
                f"BoundaryRegion.from_facets({tag!r}): expected facets of shape (E, k, dim) with "
                f"k in {sorted(_FIELD_BY_ARITY)}, got {np.shape(ents)}."
            )
        return cls(tag=tag, dim=dim, points=points, tol=tol, **{field: ents})

    @property
    def facets(self) -> Optional[np.ndarray]:
        """This region's facets, whatever their vertex count — the mirror of ``domain.tag_facets``."""
        for ents in (self.edges, self.triangles, self.quads):
            if ents is not None and len(ents):
                return ents
        return None

    def _in_quads(self, p):
        """Membership in a quadrilateral facet: near its plane, and inside all four edge half-planes.

        Splitting the quad into triangles instead — either along a diagonal or as a centroid fan — puts
        an interior triangle edge inside the face, and a point on one lands at a barycentric coordinate
        of exactly zero reached by cancelling two numbers of order 1/2. That is below the float32
        roundoff floor: measured on a 3x3x3 hexahedral box, the centroid fan missed **36 of 216** facet
        edge midpoints where the triangle path missed none. Testing the edges directly removes every
        interior edge, so the only near-zero left is a point on the facet's own boundary — the same case
        a triangular mesh has, at a magnitude of order h^2 rather than a cancellation.
        """
        import jax.numpy as jnp

        q = jnp.asarray(self.quads)  # (Q, 4, 3)
        # The two diagonals give the normal of a quad that need not be planar -- the same rule the
        # facet tables use (`fem_facets.compute_face_normals`).
        n = jnp.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
        scale = jnp.linalg.norm(n, axis=1)  # ~ 2 * facet area: the natural size of the tests below
        n = n / jnp.maximum(scale, 1e-30)[:, None]

        plane_dist = jnp.abs(jnp.sum((p[None, :] - q[:, 0]) * n, axis=1))
        side = jnp.stack(
            [jnp.sum(jnp.cross(q[:, (i + 1) % 4] - q[:, i], p[None, :] - q[:, i]) * n, axis=1) for i in range(4)],
            axis=0,
        )  # (4, Q), positive inside for a cyclically ordered facet
        inside = jnp.all(side >= -_SLACK_QUAD * jnp.maximum(scale, 1e-30)[None, :], axis=0)
        return jnp.any((plane_dist <= self.tol) & inside)

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

        # 3D: quadrilateral membership (a hexahedral mesh's facet)
        if self.dim == 3 and self.quads is not None and len(self.quads) > 0:
            return self._in_quads(p)

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

            inside = (u >= -_SLACK_TRI) & (v >= -_SLACK_TRI) & (w >= -_SLACK_TRI)
            return jnp.any((plane_dist <= self.tol) & inside)

        # fallback only if no explicit entities are available
        pts = jnp.asarray(self.points[:, : self.dim])
        d = jnp.linalg.norm(pts - p[None, :], axis=1)
        return jnp.any(d <= self.tol)
