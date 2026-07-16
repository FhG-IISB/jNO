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
