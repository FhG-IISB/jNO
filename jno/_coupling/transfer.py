"""Interface data between two non-matching discretisations of one interface.

Based on:
    de Boer, A., van Zuijlen, A. H. & Bijl, H. (2007). Review of coupling methods for non-matching meshes.
    Computer Methods in Applied Mechanics and Engineering 196, 1515-1525.
    Farhat, C., Lesoinne, M. & Le Tallec, P. (1998). Load and motion transfer algorithms for fluid/structure
    interaction problems with non-matching discrete interfaces. Computer Methods in Applied Mechanics and
    Engineering 157, 95-114.
    Wathen, A. J. (1987). Realistic eigenvalue bounds for the Galerkin mass matrix. IMA Journal of Numerical
    Analysis 7, 449-457 (the Jacobi-scaled mass is well conditioned: CG for the projection).

Each side of an interface is a set of mesh nodes on its own boundary facets (:class:`Interface`). Data move as
nodal fields:

- **values** (a temperature, a displacement) by the *consistent* map ``H(t <- s)``: every target node reads the
  source's piecewise-linear field at the closest point of the source's facets. Constants and linear fields
  pass exactly where the two discretisations describe the same surface.
- **fluxes and loads** (a heat flux, a traction) by the *conservative* map, the L2 (mortar) projection
  ``M_tt^-1 M_ts``: ``M_tt`` the target's interface mass matrix, ``M_ts`` the mixed one, integrated over the
  source's facets with the target's shape functions read at the closest point. The target's shape functions
  sum to one, so the total ``Σ w q`` is the same on both sides to round-off, whatever the meshes; and in 2-D
  the source segments are split where the target's nodes fall, so the integrals are exact and a linear flux
  crosses a shared straight interface unchanged. (The transpose of the consistent map is conservative too,
  but turns a uniform flux into a non-uniform one across non-matching nodes, measured at 3 % in a slab.)

``measure`` sets ``dμ``: plain length/area, ``"axisymmetric"`` (``2π r ds`` with ``r`` the second coordinate,
for an ``(x, r)`` meridian plane; ``"axisymmetric:0"`` with ``r`` the first, an ``(r, z)`` one), or any callable of the
points.
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
# Dunavant's degree-4 rule (6 points) for the mass matrices: φ_i φ_j μ is cubic for a linear measure
_D4_A, _D4_B = 0.445948490915965, 0.091576213509771
_D4_L = np.array(
    [
        [_D4_A, _D4_A, 1 - 2 * _D4_A],
        [_D4_A, 1 - 2 * _D4_A, _D4_A],
        [1 - 2 * _D4_A, _D4_A, _D4_A],
        [_D4_B, _D4_B, 1 - 2 * _D4_B],
        [_D4_B, 1 - 2 * _D4_B, _D4_B],
        [1 - 2 * _D4_B, _D4_B, _D4_B],
    ]
)
_D4_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def radial_axis(measure):
    """The coordinate that is ``r`` in an axisymmetric measure: ``"axisymmetric"`` the second (an ``(x, r)``
    meridian), ``"axisymmetric:0"`` the first (an ``(r, z)`` one); ``None`` for any other measure."""
    if isinstance(measure, str) and measure.startswith("axisymmetric"):
        tail = measure[len("axisymmetric") :]
        if tail in ("", ":1"):
            return 1
        if tail == ":0":
            return 0
        raise ValueError(f"measure={measure!r}: use 'axisymmetric' (r the second coordinate) or 'axisymmetric:0'.")
    return None


def _measure_fn(measure):
    if measure is None:
        return lambda p: np.ones(p.shape[0])
    if isinstance(measure, str):
        k = radial_axis(measure)
        if k is None:
            raise ValueError(f"measure={measure!r}: use None, 'axisymmetric' or a callable of the points (n, dim).")
        return lambda p: 2.0 * np.pi * p[:, k]
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
        measure: the ``measure`` the weights were integrated with.
    """

    nodes: np.ndarray
    points: np.ndarray
    facets: np.ndarray
    weights: np.ndarray
    measure: object = None

    @classmethod
    def of(cls, domain, tag, measure=None):
        """The interface a boundary ``tag`` of ``domain`` carries."""
        facets = domain.tag_facets(tag)
        if facets is None:
            # a tag made by a predicate (`domain.tag(...)`) knows its mesh NODES (`tag_node_mask`; its
            # `tag_indices` index a sampling pool, not the mesh): its facets are the boundary facets whose
            # nodes all carry it
            boundary = domain.tag_facets("boundary")
            if boundary is not None:
                pts_all = np.asarray(domain.mesh_connectivity["points"], dtype=float)
                nodes = np.flatnonzero(np.asarray(domain.tag_node_mask(tag, pts_all)))
                mine = np.isin(np.asarray(boundary), nodes).all(axis=1)
                facets = np.asarray(boundary)[mine] if mine.any() else None
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
        return cls(nodes=np.asarray(nodes), points=points, facets=facets, weights=w, measure=measure)

    @property
    def total(self):
        """``∫ dμ`` over the interface."""
        return float(self.weights.sum())

    @property
    def mass(self):
        """The interface mass matrix ``M_ij = ∫ φ_i φ_j dμ`` (sparse); its row sums are :attr:`weights`."""
        cached = self.__dict__.get("_mass")
        if cached is not None:
            return cached
        mu = _measure_fn(self.measure)
        rows, cols, vals = [], [], []
        if self.facets.shape[1] == 2:
            a, b = self.points[self.facets[:, 0]], self.points[self.facets[:, 1]]
            length = np.linalg.norm(b - a, axis=1)
            for x, wq in zip(_SEG_X, _SEG_W):
                m = mu(a + x * (b - a)) * length * wq
                phi = (1.0 - x, x)
                for i in range(2):
                    for j in range(2):
                        rows.append(self.facets[:, i])
                        cols.append(self.facets[:, j])
                        vals.append(phi[i] * phi[j] * m)
        else:
            tri = _simplices(self.facets)
            v = self.points[tri]
            area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1)
            for lam, wq in zip(_D4_L, _D4_W):
                m = mu(np.einsum("j,fjd->fd", lam, v)) * area * wq
                for i in range(3):
                    for j in range(3):
                        rows.append(tri[:, i])
                        cols.append(tri[:, j])
                        vals.append(lam[i] * lam[j] * m)
        n = self.points.shape[0]
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        object.__setattr__(self, "_mass", M)
        return M

    def project(self, loads):
        """Nodal densities whose loads ``∫ q φ_i dμ`` are ``loads``: ``M^-1 loads``, by conjugate gradients on the
        sparse interface mass with a Jacobi preconditioner, so it runs inside a compiled program and differentiates
        in ``loads`` (a host LU did neither), in O(nnz) memory. Jacobi-scaled, a linear-element mass has its
        eigenvalues in [1/2, 3/2] on segments, [1/2, 2] on triangles, [1/4, 9/4] on bilinear quads (Wathen 1987):
        CG halves the error at least every iteration and reaches round-off in a few tens of them."""
        import jax.numpy as jnp
        from jax.experimental import sparse as jsparse
        from jax.scipy.sparse.linalg import cg

        from .participant import _compiled

        ops = self.__dict__.get("_mass_ops")
        if ops is None:
            coo = sp.coo_matrix(self.mass)
            index = jnp.asarray(np.stack([coo.row, coo.col], axis=1))
            A = jsparse.BCOO((jnp.asarray(coo.data), index), shape=coo.shape)
            dinv = jnp.asarray(1.0 / self.mass.diagonal())

            def solve(loads):
                maxiter = 4 * loads.shape[0] + 50
                return cg(lambda v: A @ v, loads, M=lambda r: dinv * r, tol=1e-14, atol=0.0, maxiter=maxiter)[0]

            ops = (solve, dinv.dtype, {})
            object.__setattr__(self, "_mass_ops", ops)
        solve, dtype, cache = ops
        return _compiled(cache, None, solve, jnp.asarray(loads, dtype=dtype))  # compiled once, eagerly


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


def _stencil(iface: Interface, qpts, k: int = 8):
    """Each query point at the closest point of ``iface``'s facets: ``(nodes (q, k_f), shape values (q, k_f),
    facet (q,), parameter)`` -- ``parameter`` the position along the segment in 2-D (``None`` in 3-D)."""
    from scipy.spatial import cKDTree

    qpts = np.asarray(qpts, dtype=float)
    simp = _simplices(iface.facets)
    v = iface.points[simp]  # (f, k_f, d)
    kk = int(min(k, simp.shape[0]))
    _, cand = cKDTree(v.mean(axis=1)).query(qpts, k=kk)
    cand = np.asarray(cand).reshape(qpts.shape[0], kk)
    vc = v[cand]  # (q, kk, k_f, d)
    t = None
    if simp.shape[1] == 2:
        tt, d2 = _closest_on_segments(qpts, vc[:, :, 0], vc[:, :, 1])
        lam = np.stack([1.0 - tt, tt], axis=-1)
    else:
        lam, d2 = _closest_on_triangles(qpts, vc[:, :, 0], vc[:, :, 1], vc[:, :, 2])
    pick = np.argmin(d2, axis=1)
    q = np.arange(qpts.shape[0])
    facet = cand[q, pick]
    if simp.shape[1] == 2:
        t = tt[q, pick]
    return simp[facet], lam[q, pick], facet, t


def consistent(target: Interface, source: Interface, k: int = 8):
    """``H (m_t, m_s)``: ``H @ u_s`` is the source's piecewise-linear nodal field read at the target's nodes, each
    at the closest point of the source's facets. Rows sum to one, so constants pass exactly; linear fields pass
    exactly where the two sides describe the same surface."""
    cols, vals, _, _ = _stencil(source, target.points, k)
    rows = np.repeat(np.arange(target.points.shape[0]), cols.shape[1])
    H = sp.csr_matrix((vals.ravel(), (rows, cols.ravel())), shape=(target.points.shape[0], source.points.shape[0]))
    H.sum_duplicates()
    return H


def mixed_mass(target: Interface, source: Interface, k: int = 8, refine: int = 4):
    """``M_ts (m_t, m_s) = ∫_source φ_i^t φ_j^s dμ``, the target's shape functions read at the closest point of its
    facets. In 2-D every source segment is split where the target's nodes fall onto it, so each piece is a product
    of two linear functions (times a linear measure) and Gauss-Legendre integrates it exactly; in 3-D each source
    triangle is subdivided ``refine`` times per edge (exact totals, linear data to the subdivision's accuracy)."""
    mu = _measure_fn(source.measure)
    qp, qw, sj, sphi = [], [], [], []  # quadrature points, weights, source nodes (q, k), source shapes (q, k)
    if source.facets.shape[1] == 2:
        a_id, b_id = source.facets[:, 0], source.facets[:, 1]
        A, B = source.points[a_id], source.points[b_id]
        # where the target's nodes fall on each source segment
        _, _, fac, tpar = _stencil(source, target.points, k)
        cuts = [[0.0, 1.0] for _ in range(A.shape[0])]
        for f, t in zip(fac, tpar):
            if 0.0 < t < 1.0:
                cuts[f].append(float(t))
        for f in range(A.shape[0]):
            c = np.unique(np.asarray(cuts[f]))
            lo, hi = c[:-1], c[1:]
            for x, wq in zip(_SEG_X, _SEG_W):
                t = lo + x * (hi - lo)
                pts = A[f] + t[:, None] * (B[f] - A[f])
                qp.append(pts)
                qw.append(wq * (hi - lo) * np.linalg.norm(B[f] - A[f]) * mu(pts))
                sj.append(np.repeat([[a_id[f], b_id[f]]], t.size, axis=0))
                sphi.append(np.stack([1.0 - t, t], axis=1))
    else:
        tri = _simplices(source.facets)
        v = source.points[tri]
        area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1)
        n = int(refine)
        subs = []  # sub-triangles of the reference triangle, as barycentric vertex triples
        for i in range(n):
            for j in range(n - i):
                p0 = np.array([i, j]) / n
                p1, p2 = p0 + np.array([1, 0]) / n, p0 + np.array([0, 1]) / n
                subs.append((p0, p1, p2))
                if i + j < n - 1:
                    subs.append((p1, p0 + np.array([1, 1]) / n, p2))
        for s0, s1, s2 in subs:
            for lam_r, wq in zip(_D4_L, _D4_W):
                xi = lam_r[0] * s0 + lam_r[1] * s1 + lam_r[2] * s2  # reference (ξ, η)
                bary = np.array([1.0 - xi[0] - xi[1], xi[0], xi[1]])
                pts = np.einsum("j,fjd->fd", bary, v)
                qp.append(pts)
                qw.append(wq * area / (n * n) * mu(pts))
                sj.append(tri)
                sphi.append(np.broadcast_to(bary, (tri.shape[0], 3)))
    qp, qw = np.concatenate(qp), np.concatenate(qw)
    sj, sphi = np.concatenate(sj), np.concatenate(sphi)
    tj, tphi, _, _ = _stencil(target, qp, k)
    rows = np.repeat(tj[:, :, None], sj.shape[1], axis=2).ravel()
    cols = np.repeat(sj[:, None, :], tj.shape[1], axis=1).ravel()
    vals = (tphi[:, :, None] * sphi[:, None, :] * qw[:, None, None]).ravel()
    M = sp.csr_matrix((vals, (rows, cols)), shape=(target.points.shape[0], source.points.shape[0]))
    M.sum_duplicates()
    return M


class _Projection:
    """``M_tt^-1 M_ts``, applied without forming the inverse; traceable (the mixed mass as a sparse JAX matrix, the
    target's projection by CG)."""

    def __init__(self, target, mixed):
        self.target, self.mixed = target, mixed
        self.shape = mixed.shape
        self._mixed = None

    def __matmul__(self, q):
        import jax.numpy as jnp

        if self._mixed is None:
            self._mixed = to_jax(self.mixed)
        return self.target.project(self._mixed @ jnp.asarray(q, dtype=self._mixed.dtype))


def conservative(target: Interface, source: Interface, k: int = 8):
    """The L2 (mortar) projection of a nodal flux density onto the target: ``Σ w_t (C q_s) = Σ w_s q_s`` exactly,
    and a linear density on a shared straight interface (2-D) passes unchanged."""
    return _Projection(target, mixed_mass(target, source, k))


def to_jax(M):
    """A scipy sparse map as a JAX ``BCOO`` (``M @ x`` differentiable in ``x``)."""
    import jax.experimental.sparse as jsparse
    import jax.numpy as jnp

    M = sp.coo_matrix(M)
    idx = np.stack([M.row, M.col], axis=1).astype(np.int32)
    return jsparse.BCOO((jnp.asarray(M.data), jnp.asarray(idx)), shape=M.shape)
