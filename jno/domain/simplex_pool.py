"""Static simplex pools for in-JIT collocation-point sampling.

A ``SimplexPool`` is a domain-side precompute: at register / mesh-load time we
decompose each samplable tag into a flat list of simplices (segments for 1-D
interiors and 2-D boundaries, triangles for 2-D interiors) and store them as
JAX arrays.  The in-JIT sampler then draws fresh collocation points by

1. Picking a simplex index via measure-weighted ``jax.random.choice``.
2. Drawing barycentric coordinates inside the picked simplex.
3. Mapping those barycentrics back to world coordinates.

This keeps the per-step cost on the GPU (no Shapely round-trips) and removes
the host-side rebuild that currently dominates the resampling pipeline.

The dataclass is frozen and JAX-static — the arrays are jnp arrays but their
shapes are baked into the trace at JIT-compile time and never change during
training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SimplexPool:
    """Flat list of simplices for in-JIT uniform sampling.

    Attributes
    ----------
    simplices:
        Vertex coordinates, shape ``(T, V, D)``.  ``V`` is the number of
        vertices per simplex (2 for segments, 3 for triangles).  ``D`` is the
        spatial dimension.
    weights:
        Per-simplex measure (length for V=2, area for V=3), shape ``(T,)``.
        Not normalised; the sampler divides by ``weights.sum()`` at use time so
        downstream code can rely on the raw measures.
    kind:
        Either ``"lerp"`` (V=2, uniform-on-segment via linear interpolation)
        or ``"barycentric"`` (V=3, uniform-on-triangle via the sqrt trick).
    normals:
        Optional ``(T, D)`` per-simplex outward unit normal.  Populated only
        for boundary tags so the sampler can simultaneously emit ``n_<tag>``
        alongside the spatial coordinates.
    """

    simplices: jnp.ndarray
    weights: jnp.ndarray
    kind: str
    normals: Optional[jnp.ndarray] = None

    @classmethod
    def from_segments(
        cls,
        segments: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> "SimplexPool":
        """Build a V=2 pool from segment endpoints.

        ``segments`` has shape ``(T, 2, D)``.  Optional ``normals`` has shape
        ``(T, D)``.  Per-segment weight is the Euclidean length.
        """
        segments = np.asarray(segments, dtype=np.float32)
        if segments.ndim != 3 or segments.shape[1] != 2:
            raise ValueError(f"segments must have shape (T, 2, D); got {segments.shape}")
        vectors = segments[:, 1, :] - segments[:, 0, :]
        lengths = np.linalg.norm(vectors, axis=1).astype(np.float32)
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float32)
            if normals.shape != (segments.shape[0], segments.shape[2]):
                raise ValueError(f"normals must have shape ({segments.shape[0]}, {segments.shape[2]}); got {normals.shape}")
            normals_j = jnp.asarray(normals)
        else:
            normals_j = None
        return cls(
            simplices=jnp.asarray(segments),
            weights=jnp.asarray(lengths),
            kind="lerp",
            normals=normals_j,
        )

    @classmethod
    def from_triangles(cls, triangles: np.ndarray) -> "SimplexPool":
        """Build a V=3 pool from triangle vertex coordinates.

        ``triangles`` has shape ``(T, 3, D)`` with ``D=2`` (we only support
        2-D interiors at this layer; 3-D tets land via a separate factory).
        Per-triangle weight is the Shoelace area.
        """
        triangles = np.asarray(triangles, dtype=np.float32)
        if triangles.ndim != 3 or triangles.shape[1] != 3:
            raise ValueError(f"triangles must have shape (T, 3, D); got {triangles.shape}")
        if triangles.shape[2] != 2:
            raise ValueError(
                f"SimplexPool.from_triangles supports D=2; got D={triangles.shape[2]}. "
                "3-D tet pools land via a future factory."
            )
        # Shoelace: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        x = triangles[:, :, 0]
        y = triangles[:, :, 1]
        areas = 0.5 * np.abs(
            x[:, 0] * (y[:, 1] - y[:, 2]) + x[:, 1] * (y[:, 2] - y[:, 0]) + x[:, 2] * (y[:, 0] - y[:, 1])
        ).astype(np.float32)
        return cls(
            simplices=jnp.asarray(triangles),
            weights=jnp.asarray(areas),
            kind="barycentric",
            normals=None,
        )

    def __repr__(self) -> str:
        t, v, d = self.simplices.shape
        nrm = "+normals" if self.normals is not None else ""
        return f"SimplexPool(T={t}, V={v}, D={d}, kind={self.kind!r}{nrm})"
