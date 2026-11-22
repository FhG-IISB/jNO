"""Interface data between two non-matching discretisations of one interface.

Based on:
    de Boer, A., van Zuijlen, A. H. & Bijl, H. (2007). Review of coupling methods for non-matching meshes.
    Computer Methods in Applied Mechanics and Engineering 196, 1515-1525.
    Farhat, C., Lesoinne, M. & Le Tallec, P. (1998). Load and motion transfer algorithms for fluid/structure
    interaction problems with non-matching discrete interfaces. Computer Methods in Applied Mechanics and
    Engineering 157, 95-114.

Each side of an interface is a set of mesh nodes on its own boundary facets (:class:`Interface`). Data move as
nodal fields:

- **values** (a temperature, a displacement) by the *consistent* map ``H(t <- s)``: every target node reads the
  source's piecewise-linear field at the closest point of the source's facets. Constants and linear fields
  pass exactly where the two discretisations describe the same surface.
- **fluxes and loads** (a heat flux, a traction) by the *conservative* map ``W_t^-1 H(s <- t)^T W_s``, the
  transpose of the consistent map the other way, with ``W`` the lumped facet weights ``w_i = ∫ φ_i dμ``. The
  total ``Σ w q`` is the same on both sides to round-off, whatever the meshes, which is what a coupled
  energy or force balance needs.

``measure`` sets ``dμ``: plain length/area, ``"axisymmetric"`` (``2π r ds`` with ``r`` the second coordinate,
for an ``(x, r)`` meridian plane), or any callable of the points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# Gauss rules on the reference segment [0, 1] and triangle, exact for the degree-2 integrands φ_i · μ
# (a linear measure times a linear shape function) that the weights need.
_SEG_X = np.array([0.5 - np.sqrt(15.0) / 10.0, 0.5, 0.5 + np.sqrt(15.0) / 10.0])
_SEG_W = np.array([5.0, 8.0, 5.0]) / 18.0
_TRI_L = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])  # barycentric points
_TRI_W = np.full(3, 1.0 / 3.0)


def _measure_fn(measure):
    if measure is None:
        return lambda p: np.ones(p.shape[0])
    if isinstance(measure, str):
        if measure != "axisymmetric":
            raise ValueError(f"measure={measure!r}: use None, 'axisymmetric' or a callable of the points (n, dim).")
        return lambda p: 2.0 * np.pi * p[:, 1]
    if callable(measure):
        return lambda p: np.asarray(measure(p), dtype=float).reshape(p.shape[0])
    raise TypeError(f"measure={measure!r}: use None, 'axisymmetric' or a callable of the points (n, dim).")


def _simplices(facets):
    """Facets as simplices: segments and triangles stay, a quadrilateral face splits into two triangles (a
    piecewise-linear field on the split reproduces linear data exactly)."""
    facets = np.asarray(facets)
    if facets.shape[1] == 4:
        return np.concatenate([facets[:, [0, 1, 2]], facets[:, [0, 2, 3]]], axis=0)
    return facets


@dataclass(frozen=True)
class Interface:
    """One side's discretisation of an interface.

    Attributes:
        nodes: ``(m,)`` mesh node ids of the side's interface nodes.
        points: ``(m, dim)`` their coordinates.
        facets: ``(f, k)`` the boundary facets, as indices into ``nodes`` (segments, triangles or quadrilaterals).
        weights: ``(m,)`` the lumped facet weights ``w_i = ∫ φ_i dμ``.
    """

    nodes: np.ndarray
    points: np.ndarray
    facets: np.ndarray
    weights: np.ndarray

    @classmethod
    def of(cls, domain, tag, measure=None):
        """The interface a boundary ``tag`` of ``domain`` carries."""
        facets = domain.tag_facets(tag)
        if facets is None:
            raise ValueError(f"interface: the domain has no boundary facets tagged {tag!r}.")
        dim = int(getattr(domain, "dimension", np.asarray(domain.mesh_connectivity["points"]).shape[1]))
        pts = np.asarray(domain.mesh_connectivity["points"], dtype=float)[:, :dim]
        nodes, local = np.unique(np.asarray(facets), return_inverse=True)
        local = local.reshape(np.shape(facets))
        return cls.from_facets(nodes, pts[nodes], local, measure)

    @classmethod
    def from_facets(cls, nodes, points, facets, measure=None):
        """An interface from its nodes, their coordinates and its facets (indices into ``nodes``)."""
        points = np.asarray(points, dtype=float)
        facets = np.asarray(facets)
        mu = _measure_fn(measure)
        w = np.zeros(points.shape[0])
        if facets.shape[1] == 2:  # segments: ∫ φ_a μ ds over each
            a, b = points[facets[:, 0]], points[facets[:, 1]]
            length = np.linalg.norm(b - a, axis=1)
            for x, wq in zip(_SEG_X, _SEG_W):
                m = mu(a + x * (b - a)) * length * wq
                np.add.at(w, facets[:, 0], (1.0 - x) * m)
                np.add.at(w, facets[:, 1], x * m)
        else:  # triangles (a quadrilateral face as two)
            tri = _simplices(facets)
            v = points[tri]  # (f, 3, dim)
            area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1)
            for lam, wq in zip(_TRI_L, _TRI_W):
                m = mu(np.einsum("j,fjd->fd", lam, v)) * area * wq
                for j in range(3):
                    np.add.at(w, tri[:, j], lam[j] * m)
        return cls(nodes=np.asarray(nodes), points=points, facets=facets, weights=w)

    @property
    def total(self):
        """``∫ dμ`` over the interface."""
        return float(self.weights.sum())


def _closest_on_segments(p, a, b):
    """Closest point of each query ``p`` (q, d) on each candidate segment ``a``-``b`` (q, k, d): the parameter
    (q, k) and the squared distance (q, k)."""
    ab = b - a
    t = np.einsum("qkd,qkd->qk", p[:, None, :] - a, ab) / np.maximum(np.einsum("qkd,qkd->qk", ab, ab), 1e-300)
    t = np.clip(t, 0.0, 1.0)
    c = a + t[..., None] * ab
    return t, np.einsum("qkd,qkd->qk", p[:, None, :] - c, p[:, None, :] - c)


def _closest_on_triangles(p, v0, v1, v2):
    """Barycentric coordinates (q, k, 3) of the closest point of ``p`` on each candidate triangle, and the squared
    distance (q, k). Inside the projection the barycentric solve; outside, the closest of the three edges."""
    e1, e2, r = v1 - v0, v2 - v0, p[:, None, :] - v0
    g11, g12, g22 = (np.einsum("qkd,qkd->qk", x, y) for x, y in ((e1, e1), (e1, e2), (e2, e2)))
    r1, r2 = np.einsum("qkd,qkd->qk", r, e1), np.einsum("qkd,qkd->qk", r, e2)
    det = np.maximum(g11 * g22 - g12 * g12, 1e-300)
    s = (g22 * r1 - g12 * r2) / det
    t = (g11 * r2 - g12 * r1) / det
    lam = np.stack([1.0 - s - t, s, t], axis=-1)
    inside = (lam >= 0.0).all(axis=-1)
    c = v0 + s[..., None] * e1 + t[..., None] * e2
    best_d = np.where(inside, np.einsum("qkd,qkd->qk", p[:, None, :] - c, p[:, None, :] - c), np.inf)
    best = np.where(inside[..., None], lam, 0.0)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        vs = (v0, v1, v2)
        u, d = _closest_on_segments(p, vs[i], vs[j])
        edge = np.zeros_like(lam)
        edge[..., i], edge[..., j] = 1.0 - u, u
        take = d < best_d
        best_d = np.where(take, d, best_d)
        best = np.where(take[..., None], edge, best)
    return best, best_d


def consistent(target: Interface, source: Interface, k: int = 8):
    """``H (m_t, m_s)``: ``H @ u_s`` is the source's piecewise-linear nodal field read at the target's nodes, each
    at the closest point of the source's facets. Rows sum to one, so constants pass exactly; linear fields pass
    exactly where the two sides describe the same surface."""
    from scipy.spatial import cKDTree

    simp = _simplices(source.facets)
    v = source.points[simp]  # (f, k_f, d)
    kk = int(min(k, simp.shape[0]))
    _, cand = cKDTree(v.mean(axis=1)).query(target.points, k=kk)
    cand = np.asarray(cand).reshape(target.points.shape[0], kk)
    vc = v[cand]  # (q, kk, k_f, d)
    if simp.shape[1] == 2:
        t, d2 = _closest_on_segments(target.points, vc[:, :, 0], vc[:, :, 1])
        lam = np.stack([1.0 - t, t], axis=-1)
    else:
        lam, d2 = _closest_on_triangles(target.points, vc[:, :, 0], vc[:, :, 1], vc[:, :, 2])
    pick = np.argmin(d2, axis=1)
    q = np.arange(target.points.shape[0])
    cols = simp[cand[q, pick]]  # (q, k_f) source-local node indices
    vals = lam[q, pick]
    rows = np.repeat(q, cols.shape[1])
    H = sp.csr_matrix((vals.ravel(), (rows, cols.ravel())), shape=(target.points.shape[0], source.points.shape[0]))
    H.sum_duplicates()
    return H


def conservative(target: Interface, source: Interface, k: int = 8):
    """``C (m_t, m_s) = W_t^-1 H(s <- t)^T W_s``: maps a nodal flux density so that ``Σ w_t (C q_s) = Σ w_s q_s``
    exactly (the rows of ``H(s <- t)`` sum to one)."""
    Hst = consistent(source, target, k=k)  # (m_s, m_t)
    if np.any(target.weights <= 0.0):
        raise ValueError("conservative transfer: a target node carries no weight (a zero or negative measure there).")
    return sp.diags(1.0 / target.weights) @ Hst.T @ sp.diags(source.weights)


def to_jax(M):
    """A scipy sparse map as a JAX ``BCOO`` (``M @ x`` differentiable in ``x``)."""
    import jax.experimental.sparse as jsparse
    import jax.numpy as jnp

    M = sp.coo_matrix(M)
    idx = np.stack([M.row, M.col], axis=1).astype(np.int32)
    return jsparse.BCOO((jnp.asarray(M.data), jnp.asarray(idx)), shape=M.shape)
