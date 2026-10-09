"""Auxiliary-space building blocks for H(curl) (Nédélec / N1E) preconditioning.

The auxiliary-space Maxwell solver **AMS** (Hiptmair & Xu, *SIAM J. Numer. Anal.* 45(6):2483–2509,
2007, §5; Kolev & Vassilevski, *J. Comput. Math.* 27(5):604–623, 2009) preconditions a curl-curl
system by correcting its near-null-space on a cheaper *nodal* auxiliary problem. The first ingredient
is the **discrete gradient** ``G`` — the node→edge incidence matrix whose columns span exactly the
kernel of the curl-curl operator (``∇×∇φ = 0`` discretely). Plain point/Jacobi smoothing cannot damp
that gradient sub-space, so its condition number leaks into the iteration count; the AMS correction
``G (GᵀAG)⁻¹ Gᵀ`` restores it. This module builds ``G`` from the N1E edge topology the non-nodal
assembler stashes on the domain.
"""

from __future__ import annotations

from typing import Mapping

import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np


def discrete_gradient(topology: Mapping) -> jsparse.BCOO:
    """Discrete gradient ``G`` (node→edge incidence) for a Nédélec first-kind (N1E) space.

    Row ``e`` of ``G`` maps a nodal field ``φ`` to the tangential moment of ``∇φ`` on edge ``e``. With
    the canonical ``edge_vertices[e] = (lo, hi)`` orientation the N1E assembler uses — the lo→hi edge
    tangent — that moment is ``φ(hi) − φ(lo)``, so ``G[e, lo] = -1`` and ``G[e, hi] = +1``. The columns
    of ``G`` therefore span the discrete gradient space, i.e. the kernel of the curl-curl operator
    (``curl(G φ) = 0``) — the near-null-space AMS corrects on a nodal auxiliary problem.

    Args:
        topology: the ``domain._fem_nonnodal_topology`` dict the N1E assembler stashes; needs
            ``n_edges``, ``n_verts`` and the canonical ``edge_vertices`` pairs.

    Returns:
        The ``(n_edges, n_verts)`` incidence as a ``BCOO`` — so it drops straight into a traced
        auxiliary operator ``Gᵀ A G`` without a host round-trip.
    """
    if not topology.get("lowest_order_n1e", True):
        return high_order_transfer(topology)[0]
    n_edges = int(topology["n_edges"])
    n_verts = int(topology["n_verts"])
    ev = np.asarray(topology["edge_vertices"], dtype=np.int64)  # (n_edges, 2) canonical (lo, hi)
    if ev.shape != (n_edges, 2):
        raise ValueError(f"discrete_gradient: edge_vertices has shape {ev.shape}, expected {(n_edges, 2)}.")
    rows = np.repeat(np.arange(n_edges, dtype=np.int64), 2)
    cols = ev.reshape(-1)  # [lo_0, hi_0, lo_1, hi_1, ...]
    data = np.tile(np.asarray([-1.0, 1.0]), n_edges)  # -φ(lo) + φ(hi)
    indices = jnp.asarray(np.stack([rows, cols], axis=1))
    return jsparse.BCOO((jnp.asarray(data), indices), shape=(n_edges, n_verts))


def nodal_vector_interpolation(topology: Mapping) -> tuple[jsparse.BCOO, jsparse.BCOO, jsparse.BCOO]:
    """Nodal→edge vector interpolation ``(Π_x, Π_y, Π_z)`` for a Nédélec first-kind (N1E) space.

    AMS corrects a *second* near-null-space — the solenoidal (divergence-free) modes the discrete
    gradient misses — on an auxiliary **vector nodal** problem. The link is ``Π``, which maps a
    piecewise-linear nodal vector field to N1E edge DOFs: the DOF on edge ``e`` is the circulation
    ``∫_e v·dl ≈ v(mid)·t_e`` with ``t_e = x_hi − x_lo`` (midpoint rule), and ``v(mid) = ½(v_lo + v_hi)``
    for a linear field, so ``Π_α[e, lo] = Π_α[e, hi] = ½ t_e[α]``. Following Kolev & Vassilevski
    (*J. Comput. Math.* 27(5):604–623, 2009, §3) the three scalar components are kept separate — each
    gets its own scalar auxiliary solve ``(Π_αᵀ A Π_α)⁻¹`` — which is cheaper than one coupled 3n-vector
    solve and works as well in practice.

    A key consistency property, exact by construction, is that ``Π`` reproduces constant vector fields
    and ties back to the discrete gradient: ``Π_α · 1 = G · coords[:, α] = t_e[α]``.

    Args:
        topology: the ``domain._fem_nonnodal_topology`` dict; needs ``n_edges``, ``n_verts``, the
            canonical ``edge_vertices`` pairs and ``vertex_points``.

    Returns:
        A 3-tuple of ``(n_edges, n_verts)`` BCOO blocks — one per Cartesian component.
    """
    if not topology.get("lowest_order_n1e", True):
        return high_order_transfer(topology)[1]
    n_edges = int(topology["n_edges"])
    n_verts = int(topology["n_verts"])
    ev = np.asarray(topology["edge_vertices"], dtype=np.int64)  # (n_edges, 2) canonical (lo, hi)
    vpts = np.asarray(topology["vertex_points"], dtype=float)
    lo, hi = ev[:, 0], ev[:, 1]
    t = vpts[hi] - vpts[lo]  # (n_edges, 3) edge vectors x_hi − x_lo
    rows = np.repeat(np.arange(n_edges, dtype=np.int64), 2)
    cols = np.stack([lo, hi], axis=1).reshape(-1)  # [lo_0, hi_0, lo_1, hi_1, ...]
    indices = jnp.asarray(np.stack([rows, cols], axis=1))
    blocks = tuple(
        jsparse.BCOO((jnp.asarray(np.repeat(0.5 * t[:, a], 2)), indices), shape=(n_edges, n_verts))
        for a in range(3)  # both endpoints of an edge share ½ t_α
    )
    return blocks


def _lagrange_degree_for(family: str, degree: int) -> int:
    """The H¹ space whose gradients are EXACTLY the curl-free part of the H(curl) space (the discrete de
    Rham sequence): P_k for first-kind N1E_k, P_{k+1} for second-kind N2E_k (whose fields are full P_k)."""
    if family == "N1E":
        return int(degree)
    if family == "N2E":
        return int(degree) + 1
    raise ValueError(f"AMS transfer operators are defined for H(curl) fields (N1E/N2E); got {family!r}.")


def high_order_transfer(topology: Mapping, *, chunk: int = 20000):
    """``(G, (Π_x, Π_y[, Π_z]))`` for an H(curl) field of ANY degree, from basix interpolation.

    ``G`` maps the H¹ Lagrange space ``P_m`` (``m = k`` for N1E_k, ``k+1`` for N2E_k) into the H(curl)
    space: column ``j`` holds the H(curl) DOFs of ``∇φ_j``. Because the covariant Piola map pulls
    ``∇ₓφ`` back to ``∇_ξφ̂``, the reference block ``Ĝ[i, j] = ℓ̂_i(∇_ξ ψ̂_j)`` is the same on every cell
    (basix ``interpolation_matrix`` applied to the tabulated reference gradients); the cell block is
    ``B_N⁻ᵀ Ĝ B_Lᵀ``, with the two elements' DOF transforms (:mod:`fem_dofmap`). ``Π_α`` interpolates
    the vector field ``e_α φ_j`` (``φ_j ∈ P_m``): its pull-back ``Jᵀe_α ψ̂_j`` is cell-dependent through
    the row ``J[α, :]``, ``Π̂_α = Σ_m J[α, m] P̂_m``. Global entries are SET (not summed) from the first
    cell that owns them -- a conforming interpolant gives every owning cell the same value, which the
    test-suite checks. These are the high-order AMS ingredients of Hiptmair & Xu (2007) and of
    Kolev & Vassilevski's AMS for arbitrary-order Nédélec spaces (hypre/MFEM use the same pair).

    Exactness: ``G`` is exact (``curl G = 0`` and ``range G`` = the whole discrete kernel); ``Π`` is exact
    on ``(P_m)^d``. Built on the host in chunks of ``chunk`` cells; returns BCOO blocks."""
    import jax.experimental.sparse as jsparse

    from .fem_dofmap import build_dofmap

    dmN = topology["dofmap"]
    cells = np.asarray(topology["cells"], dtype=np.int64)
    pts = np.asarray(topology["vertex_points"], dtype=float)
    tdim = dmN.tdim
    cache = topology.get("_ams_cache") if isinstance(topology, dict) else None
    if cache is not None:
        return cache
    m = _lagrange_degree_for(dmN.family, dmN.degree)
    dmL = build_dofmap(cells, "Lagrange", m, n_verts=pts.shape[0])
    eN, eL = dmN.element, dmL.element
    X = np.asarray(eN.points)  # (npts, tdim) H(curl) interpolation points
    Imat = np.asarray(eN.interpolation_matrix)  # (nN, vs * npts), component-major
    tabL = eL.tabulate(1, X)  # (1 + tdim, npts, nL, 1)
    npts = X.shape[0]
    # P̂_m[i, j] = Σ_p I[i, m*npts + p] ψ̂_j(x_p);  Ĝ = Σ_m I_m ∂_m ψ̂ (the gradient's m-th component)
    Pm = np.stack([Imat[:, a * npts : (a + 1) * npts] @ tabL[0][:, :, 0] for a in range(tdim)])  # (tdim, nN, nL)
    Gref = sum(Imat[:, a * npts : (a + 1) * npts] @ tabL[1 + a][:, :, 0] for a in range(tdim))  # (nN, nL)

    def _full_B(dm, cs):
        B = np.broadcast_to(np.eye(dm.ndof_local), (len(cs), dm.ndof_local, dm.ndof_local)).copy()
        if dm.is_diagonal:
            return B * dm.signs[cs][:, :, None]
        for d, blk in dm.blocks.items():
            for kk, idx in enumerate(dm.entity_dofs[d]):
                B[:, np.asarray(idx)[:, None], np.asarray(idx)[None, :]] = blk[kk][dm.orient[d][cs, kk].astype(np.int64)]
        return B

    J_all = pts[cells[:, 1:]] - pts[cells[:, :1]]  # (n_cells, tdim, gdim): rows are v_k - v_0
    J_all = np.swapaxes(J_all, 1, 2)  # J[c][:, k] = v_{k+1} - v_0
    rows_l, cols_l, g_l = [], [], []
    p_l = [[] for _ in range(tdim)]
    for s0 in range(0, cells.shape[0], chunk):
        cs = np.arange(s0, min(s0 + chunk, cells.shape[0]))
        BN, BL = _full_B(dmN, cs), _full_B(dmL, cs)
        BNinvT = np.linalg.inv(np.swapaxes(BN, 1, 2))
        BLT = np.swapaxes(BL, 1, 2)
        Gc = BNinvT @ Gref[None] @ BLT  # (n, nN, nL)
        Jc = J_all[cs]
        for a in range(tdim):
            Pa = np.einsum("nm,mij->nij", Jc[:, a, :], Pm)  # Σ_m J[α, m] P̂_m
            p_l[a].append((BNinvT @ Pa @ BLT).reshape(-1))
        r = np.broadcast_to(dmN.cell_dofs[cs][:, :, None], Gc.shape).reshape(-1)
        c = np.broadcast_to(dmL.cell_dofs[cs][:, None, :], Gc.shape).reshape(-1)
        rows_l.append(r)
        cols_l.append(c)
        g_l.append(Gc.reshape(-1))
    rows, cols = np.concatenate(rows_l), np.concatenate(cols_l)
    key = rows * np.int64(dmL.n_dofs) + cols
    _u, first = np.unique(key, return_index=True)  # SET semantics: one value per (row, col), first owner
    rows, cols = rows[first], cols[first]
    idx = np.stack([rows, cols], axis=1)
    shape = (int(dmN.n_dofs), int(dmL.n_dofs))

    def _bcoo(vals):
        v = vals[first]
        tol = 1e-13 * max(float(np.abs(v).max()) if v.size else 0.0, 1.0)
        keep = np.abs(v) > tol
        return jsparse.BCOO((jnp.asarray(v[keep]), jnp.asarray(idx[keep])), shape=shape)

    G = _bcoo(np.concatenate(g_l))
    Pis = tuple(_bcoo(np.concatenate(p_l[a])) for a in range(tdim))
    out = (G, Pis, dmL)
    if isinstance(topology, dict):
        topology["_ams_cache"] = out
    return out
