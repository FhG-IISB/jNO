"""Finite-volume operators on an :class:`~jno._fvm.geometry.FVGraph`, as ``jax`` functions.

Based on:
  Moukalled, F., Mangani, L., & Darwish, M. (2016). The Finite Volume Method in Computational Fluid
  Dynamics. Springer -- ch. 8 (least-squares gradient), ch. 9 (face gradient with the non-orthogonal
  correction along the centroid line).

Every operator takes per-volume arrays ``(n, c)`` (``c`` components) and, where a boundary is involved,
per-boundary-face arrays ``(n_bf, c)`` of face values. Nothing here knows which PDE is solved.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["FVOps"]


class FVOps:
    """Geometry as device arrays, and the gradient, interpolation and divergence operators on it.

    Args:
        graph: the control volumes and faces.
        dtype: floating-point type of the geometry arrays.
    """

    def __init__(self, graph, dtype=None):
        dtype = dtype or jnp.result_type(float)
        g = graph
        self.graph = g
        self.n, self.dim = g.n, g.dim
        to = lambda a: jnp.asarray(np.asarray(a, dtype=float), dtype=dtype)  # noqa: E731
        self.owner = jnp.asarray(g.owner, dtype=jnp.int32)
        self.neigh = jnp.asarray(g.neigh, dtype=jnp.int32)
        self.b_owner = jnp.asarray(g.b_owner, dtype=jnp.int32)
        self.volume = to(g.volume)
        self.centroid = to(g.centroid)

        smag = np.linalg.norm(g.area, axis=1)
        self.S, self.Smag, self.nhat = to(g.area), to(smag), to(g.area / smag[:, None])
        self.fcentroid = to(g.fcentroid)
        dmag = np.linalg.norm(g.delta, axis=1)
        self.delta, self.dmag, self.dhat = to(g.delta), to(dmag), to(g.delta / dmag[:, None])
        # linear interpolation weight of the neighbour: the face centroid projected on the centroid line
        off = g.fcentroid - g.centroid[g.owner]
        self.gw = to(np.clip(np.einsum("fd,fd->f", off, g.delta) / dmag**2, 0.0, 1.0))

        bmag = np.linalg.norm(g.b_area, axis=1) if g.b_area.size else np.zeros(0)
        bn = g.b_area / np.maximum(bmag, np.finfo(float).tiny)[:, None]
        self.bS, self.bSmag, self.bn = to(g.b_area), to(bmag), to(bn)
        self.bcentroid = to(g.b_centroid)
        r = g.b_centroid - g.centroid[g.b_owner]
        dperp = np.einsum("fd,fd->f", r, bn)
        if dperp.size and dperp.min() <= 0:
            raise ValueError("jno.fvm: a boundary face lies behind its cell centroid (an inverted cell).")
        self.br, self.bdperp = to(r), to(dperp)
        self.br_t = to(r - dperp[:, None] * bn)  # the tangential offset of the face centroid
        # The owner value at the foot of a boundary normal uses the volume gradient, which itself reads the
        # boundary values: a fixed point, alternated a few times where boundary cells are skewed (each pass
        # gains about a factor 25 on a triangulation; 6 passes reach round-off for a linear field), once
        # where every boundary normal runs through its cell centroid.
        rt = np.linalg.norm(r - dperp[:, None] * bn, axis=1) if r.size else np.zeros(0)
        from .trace import _BOUNDARY_PASSES

        self.boundary_passes = 1 if not rt.size or rt.max() <= 1e-12 * dperp.max() else _BOUNDARY_PASSES

        # least squares on the neighbours (and boundary faces), weights 1/|d|^2; the normal matrix per volume
        w = 1.0 / dmag**2
        wb = 1.0 / np.maximum(np.einsum("fd,fd->f", r, r), np.finfo(float).tiny)
        A = np.zeros((g.n, g.dim, g.dim))
        outer = w[:, None, None] * g.delta[:, :, None] * g.delta[:, None, :]
        np.add.at(A, g.owner, outer)
        np.add.at(A, g.neigh, outer)
        np.add.at(A, g.b_owner, wb[:, None, None] * r[:, :, None] * r[:, None, :])
        if np.any(np.abs(np.linalg.det(A)) <= 1e-12 * np.abs(A).reshape(g.n, -1).max(axis=1) ** g.dim):
            raise ValueError("jno.fvm: a control volume has too few neighbours for a least-squares gradient.")
        self.lsq_w, self.lsq_wb, self.lsq_inv = to(w), to(wb), to(np.linalg.inv(A))

    # -- sums over faces ------------------------------------------------------------------------------
    def _sum(self, values, index):
        return jax.ops.segment_sum(values, index, num_segments=self.n)

    def divergence(self, flux_i, flux_b):
        """``Σ_faces flux`` per volume: interior fluxes ``(n_if, c)`` leave the owner and enter the
        neighbour, boundary fluxes ``(n_bf, c)`` leave their owner."""
        out = self._sum(flux_i, self.owner) - self._sum(flux_i, self.neigh)
        if flux_b is not None and self.b_owner.shape[0]:
            out = out + self._sum(flux_b, self.b_owner)
        return out

    # -- gradients ------------------------------------------------------------------------------------
    def cell_gradient(self, T, Tb):
        """Least-squares gradient ``(n, c, dim)`` of volume values ``T`` ``(n, c)``, with the boundary face
        values ``Tb`` ``(n_bf, c)`` as extra neighbours."""
        dT = T[self.neigh] - T[self.owner]
        contrib = (self.lsq_w[:, None, None] * dT[:, :, None]) * self.delta[:, None, :]
        b = self._sum(contrib, self.owner) + self._sum(contrib, self.neigh)
        if self.b_owner.shape[0]:
            dTb = Tb - T[self.b_owner]
            b = b + self._sum((self.lsq_wb[:, None, None] * dTb[:, :, None]) * self.br[:, None, :], self.b_owner)
        return jnp.einsum("nij,ncj->nci", self.lsq_inv, b)

    def green_gauss(self, T, Tb):
        """Green-Gauss gradient ``(n, c, dim)``: ``(1/V) Σ_f T_f S_f`` with linearly interpolated face values."""
        Tf = self.face_value(T)
        contrib = Tf[:, :, None] * self.S[:, None, :]
        b = self._sum(contrib, self.owner) - self._sum(contrib, self.neigh)
        if self.b_owner.shape[0]:
            b = b + self._sum(Tb[:, :, None] * self.bS[:, None, :], self.b_owner)
        return b / self.volume[:, None, None]

    def multidimensional_limiter(self, T, Tb, G, kind, k=0.3):
        """Per-volume factor ``(n, c)`` that keeps ``T_P + G·(x_f − x_P)`` within the extrema of the volume and
        its neighbours at every face: Barth-Jespersen's ``min(1, Δ_max/Δ)``, or Venkatakrishnan's smooth
        version of it (``ε² = (k h)³``)."""
        lo, hi = self.neighbour_extrema(T, Tb)
        h = self.volume ** (1.0 / self.dim)
        eps2 = ((k * h) ** 3)[:, None]

        def factor(cell, r):
            d2 = jnp.einsum("fcd,fd->fc", G[cell], r)
            dmax = jnp.where(d2 > 0, hi[cell] - T[cell], lo[cell] - T[cell])
            if kind == "barth":
                safe = jnp.where(jnp.abs(d2) > 0, d2, 1.0)
                return jnp.where(jnp.abs(d2) > 0, jnp.minimum(1.0, dmax / safe), 1.0)
            e2 = eps2[cell]
            return (dmax**2 + e2 + 2 * d2 * dmax) / (dmax**2 + 2 * d2**2 + d2 * dmax + e2)

        rL = self.fcentroid - self.centroid[self.owner]
        rR = self.fcentroid - (self.centroid[self.owner] + self.delta)
        phi = jnp.minimum(
            jax.ops.segment_min(factor(self.owner, rL), self.owner, num_segments=self.n),
            jax.ops.segment_min(factor(self.neigh, rR), self.neigh, num_segments=self.n),
        )
        if self.b_owner.shape[0]:
            phi = jnp.minimum(phi, jax.ops.segment_min(factor(self.b_owner, self.br), self.b_owner, num_segments=self.n))
        return jnp.clip(phi, 0.0, 1.0)

    def face_value(self, T):
        """Linear interpolation of volume values to the interior faces."""
        return T[self.owner] + self.gw[:, None] * (T[self.neigh] - T[self.owner])

    def face_gradient(self, T, G):
        """Interior face gradient ``(n_if, c, dim)``: the interpolated volume gradient, with its component
        along the centroid line replaced by the two-point difference (compact on an orthogonal mesh, and
        corrected for non-orthogonality otherwise)."""
        Gf = G[self.owner] + self.gw[:, None, None] * (G[self.neigh] - G[self.owner])
        along = (T[self.neigh] - T[self.owner]) / self.dmag[:, None]
        corr = along - jnp.einsum("fcd,fd->fc", Gf, self.dhat)
        return Gf + corr[:, :, None] * self.dhat[:, None, :]

    def owner_at_face(self, T, G):
        """The owner value carried to the foot of the face normal, ``T_P + G_P·r_t`` ``(n_bf, c)``: what the
        two-point relation ``∂T/∂n = (T_f - T_P')/d⊥`` compares a boundary face value with."""
        return T[self.b_owner] + jnp.einsum("fcd,fd->fc", G[self.b_owner], self.br_t)

    def boundary_gradient(self, T, G, Tb):
        """Boundary face gradient ``(n_bf, c, dim)``: the owner's gradient with its normal component from the
        face value, ``(T_f - T_P')/d⊥``."""
        Gp = G[self.b_owner]
        dn = (Tb - self.owner_at_face(T, G)) / self.bdperp[:, None]
        return Gp + (dn - jnp.einsum("fcd,fd->fc", Gp, self.bn))[:, :, None] * self.bn[:, None, :]

    # -- reconstruction -------------------------------------------------------------------------------
    def reconstruct(self, T, G):
        """Left and right states at the interior faces ``(T_L, T_R)`` and the inner state at the boundary
        faces, extrapolated with the (possibly limited) volume gradient ``G``."""
        Gl = G
        rL = self.fcentroid - self.centroid[self.owner]
        rR = self.fcentroid - (self.centroid[self.owner] + self.delta)  # periodic-safe: neighbour seen from owner
        TL = T[self.owner] + jnp.einsum("fcd,fd->fc", Gl[self.owner], rL)
        TR = T[self.neigh] + jnp.einsum("fcd,fd->fc", Gl[self.neigh], rR)
        Tb = T[self.b_owner] + jnp.einsum("fcd,fd->fc", Gl[self.b_owner], self.br)
        return TL, TR, Tb

    def neighbour_extrema(self, T, Tb):
        """Per-volume minimum and maximum of the volume's own value, its face neighbours' and its boundary
        face values ``(n, c)`` -- the bounds a multidimensional limiter keeps the reconstruction within."""
        lo = jnp.minimum(
            jax.ops.segment_min(T[self.neigh], self.owner, num_segments=self.n),
            jax.ops.segment_min(T[self.owner], self.neigh, num_segments=self.n),
        )
        hi = jnp.maximum(
            jax.ops.segment_max(T[self.neigh], self.owner, num_segments=self.n),
            jax.ops.segment_max(T[self.owner], self.neigh, num_segments=self.n),
        )
        if self.b_owner.shape[0] and Tb is not None:
            lo = jnp.minimum(lo, jax.ops.segment_min(Tb, self.b_owner, num_segments=self.n))
            hi = jnp.maximum(hi, jax.ops.segment_max(Tb, self.b_owner, num_segments=self.n))
        return jnp.minimum(lo, T), jnp.maximum(hi, T)
