"""Interpolation-cover enrichment: p-refinement that never touches the mesh.

The enriched approximation of Kim & Bathe (*Comput. Struct.* **115** (2013) 1-11), as used for
topology optimisation on a deformable mesh by Jung, Yun & Kim (*Comput. Struct.* **331** (2026)
108403, eq. 7-11). Each node carries, besides its ordinary value, a local polynomial **cover**
multiplied by that node's hat function::

    u_h(x) = Σ_i h_i(x) · ( u_i + Σ_m p_m(x) a_im )

with ``p_m`` the first-order cover ``(x - x_i)_m`` in **physical** coordinates. Two properties do
all the work:

* **Conformity is automatic.** ``h_i`` vanishes on the far side of node ``i``'s patch, so an
  enriched node sitting next to an unenriched one simply blends. No edge-mode matching, no
  constraint equations between different-order cells -- which is the whole reason this is the
  cheap route to variable ``p``.
* **The cover must be PHYSICAL, not referential.** ``h_i(ξ)·(ξ - ξ_i)`` would be discontinuous
  across a cell boundary, because two cells sharing node ``i`` disagree about ``ξ``. Physical
  ``(x - x_i)`` agrees, so the enriched space stays C⁰. That is why these tables are built per
  cell here rather than tabulated once on the reference cell.

**What the enrichment buys, exactly.** On a simplex the span of ``{h_i} ∪ {h_i (x-x_i)_m}`` is
exactly ``P2``: ``dim+1`` nodal functions plus ``(dim+1)·dim`` enrichment functions, minus the
null space below, equals ``dim(dim+3)/2 + 1`` = dim P2. Measured: a first-order cover reproduces
every global quadratic to ``≤ 4e-15``.

**The null space is structural, not numerical, and this module hands it to the caller.** Because
``Σ_i h_i(x) = 1`` and ``Σ_i h_i(x)·x_i = x`` on a simplex mesh, ``Σ_i h_i(x)·(x - x_i) ≡ 0``
identically. More generally the per-edge condition is ``(x_j - x_i)·(a_i - a_j) = 0``, whose
solutions are ``a_i = S·x_i + c`` with ``S`` **skew** -- constants plus rotations. So the
enrichment block is rank-deficient by ``dim(dim+1)/2`` per scalar component, *independent of the
mesh*: 3 in 2-D, 6 in 3-D. Measured constant at 16/36/81 nodes (2-D) and 27/64/125 (3-D).
:func:`cover_null_modes` returns exactly those vectors so the caller can remove them.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

# First order is what the reference implements and what this module supports; the span argument
# above and the null-space count both assume it. A quadratic cover changes both.
COVER_ORDER = 1


def cover_count(dim: int, cover_order: int = COVER_ORDER) -> int:
    """Number of cover functions per node, ``M``. First order: one per spatial direction."""
    if int(cover_order) != 1:
        raise NotImplementedError(
            f"interpolation covers: only first-order covers are implemented (got cover_order="
            f"{cover_order}). A higher cover changes the spanned space and the null-space count, "
            "both of which this module's tests pin."
        )
    return int(dim)


def cover_block(dim: int, cover_order: int = COVER_ORDER) -> int:
    """DOFs per node per component: the value plus its covers, ``1 + M``."""
    return 1 + cover_count(dim, cover_order)


def nodal_scale(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Per-node length scale: the mean length of the edges incident on each node.

    The covers are ``(x - x_i)``, which carry the dimension of length, so without a scale the
    enrichment columns shrink like ``h`` relative to the nodal ones and the block's conditioning
    drifts with refinement. Dividing by a nodal length makes every column O(1). Measured effect on
    the enrichment map's condition number is modest (7.1 -> 6.7 at 16 nodes, 19.8 -> 16.4 at 81),
    so this is hygiene rather than a rescue -- but it costs nothing and it makes the scale of an
    enrichment coefficient interpretable (it is a directional derivative, not a derivative times a
    mesh size).
    """
    pts = np.asarray(points, dtype=float)
    cel = np.asarray(cells, dtype=np.int64)
    acc = np.zeros(pts.shape[0])
    cnt = np.zeros(pts.shape[0])
    n_loc = cel.shape[1]
    for a in range(n_loc):
        for b in range(n_loc):
            if a == b:
                continue
            ia, ib = cel[:, a], cel[:, b]
            np.add.at(acc, ia, np.linalg.norm(pts[ia] - pts[ib], axis=1))
            np.add.at(cnt, ia, 1.0)
    scale = acc / np.maximum(cnt, 1.0)
    # An isolated node (no incident cell) would give 0 and then divide by zero downstream. It
    # cannot carry an enrichment either, so 1.0 is the harmless value -- and jno.domain already
    # removes orphan nodes on load, so this is belt-and-braces.
    scale[scale <= 0.0] = 1.0
    return scale


def expand_cover(
    phi: jnp.ndarray,
    dphi: jnp.ndarray,
    xq: jnp.ndarray,
    node_pts: jnp.ndarray,
    scale: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Enrich one cell's P1 tables with first-order covers.

    Args:
        phi: ``(n_quad, n_local)`` hat values at this cell's quadrature points.
        dphi: ``(n_quad, n_local, dim)`` **physical** hat gradients.
        xq: ``(n_quad, dim)`` physical quadrature points.
        node_pts: ``(n_local, dim)`` physical coordinates of this cell's nodes.
        scale: ``(n_local,)`` nodal length scale (see :func:`nodal_scale`).

    Returns:
        ``(phi_e, dphi_e)`` of shapes ``(n_quad, n_local*(1+M))`` and
        ``(n_quad, n_local*(1+M), dim)``, ordered **node-major**: node ``i`` owns
        ``[i*(1+M)]`` (the value) then ``[i*(1+M)+1+m]`` (cover ``m``). That ordering is not
        cosmetic -- it is what makes the global DOF map ``offs + node*(1+M)*vec + m*vec + c``
        agree with the local ravel the assembler already produces.

    Differentiable in ``node_pts`` and ``xq``, so a trainable-coordinate mesh still gets ``∂/∂X``
    through the enrichment as well as through ``|det J|``.
    """
    n_q, n_loc = phi.shape
    dim = int(node_pts.shape[-1])
    m = cover_count(dim)
    blk = 1 + m

    rel = (xq[:, None, :] - node_pts[None, :, :]) / scale[None, :, None]  # (n_q, n_local, dim)

    # values: [h_i , h_i * rel_i0 , ... , h_i * rel_i,dim-1]
    val = jnp.concatenate([phi[:, :, None], phi[:, :, None] * rel], axis=-1)  # (n_q, n_local, 1+M)

    # gradients: ∇(h_i) for the value; ∇(h_i · rel_im) = ∇h_i · rel_im + h_i · e_m / s_i
    grad_val = dphi[:, :, None, :]  # (n_q, n_local, 1, dim)
    grad_cov = dphi[:, :, None, :] * rel[..., None] + (
        phi[:, :, None, None] * jnp.eye(dim)[None, None, :, :] / scale[None, :, None, None]
    )  # (n_q, n_local, M, dim)
    grad = jnp.concatenate([grad_val, grad_cov], axis=2)  # (n_q, n_local, 1+M, dim)

    return val.reshape(n_q, n_loc * blk), grad.reshape(n_q, n_loc * blk, dim)


def cover_null_modes(points: np.ndarray, dim: int, n_comp: int = 1, block_stride: int | None = None) -> np.ndarray:
    """The enrichment block's exact null space, as vectors over the **padded** DOF layout.

    Returns ``(n_modes, n_nodes * (1+M) * n_comp)`` with
    ``n_modes = n_comp · dim(dim+1)/2`` -- the ``dim`` constant modes ``a_i = c`` plus the
    ``dim(dim-1)/2`` rotational modes ``a_i = S·x_i`` with ``S`` skew. Nodal-value slots are zero
    in every mode: the deficiency lives entirely in the enrichment.

    These are *exact* algebraic identities of the basis, not near-null directions, so the
    assembled operator is singular to machine precision without removing them. The count is
    mesh-independent, which is what makes a check on it a real test rather than a tolerance.
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    m = cover_count(dim)
    blk = 1 + m
    stride = blk * n_comp if block_stride is None else int(block_stride)
    width = n * stride

    modes = []
    for comp in range(n_comp):
        # constants: a_{i,k} = 1 for every node i
        for k in range(m):
            v = np.zeros(width)
            v[np.arange(n) * stride + (1 + k) * n_comp + comp] = 1.0
            modes.append(v)
        # rotations: S = e_p e_q^T - e_q e_p^T, so a_{i,p} = x_{i,q}, a_{i,q} = -x_{i,p}
        for p in range(dim):
            for q in range(p + 1, dim):
                v = np.zeros(width)
                v[np.arange(n) * stride + (1 + p) * n_comp + comp] = pts[:, q]
                v[np.arange(n) * stride + (1 + q) * n_comp + comp] = -pts[:, p]
                modes.append(v)

    out = np.stack(modes, axis=0)
    expect = n_comp * dim * (dim + 1) // 2
    if out.shape[0] != expect:
        raise AssertionError(f"cover_null_modes built {out.shape[0]} modes, expected {expect}")
    return out
