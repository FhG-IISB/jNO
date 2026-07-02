"""Lagrange element factories for the native 2D/3D assembler.

Wraps basix tabulation into :class:`~fem_elements.ElementSpec` for scalar Lagrange
families (P1, P2, P3, … — any degree) on triangles and tetrahedra, and exposes
:func:`identity_pushforward` — the isoparametric chain-rule map from reference to physical
gradients:

    ``∂φ/∂x = ∂φ/∂ξ J⁻¹``

This is the Lagrange analogue of the contravariant and covariant Piola maps in
:mod:`fem_elements`. The basis is tabulated and the assembly-mesh nodes are placed via the
**same** basix element (:func:`_lagrange_basix`), so the DOF order of the promoted P{k} mesh
(:func:`fem_utils._promote_to_degree`, which puts :func:`lagrange_interp_points` on each cell)
always matches the tabulated shape functions.

References
----------
P.G. Ciarlet, *The Finite Element Method for Elliptic Problems*, SIAM (2002),
§§ 2.2–2.3 (Lagrange elements on simplices, isoparametric mapping).
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .fem_elements import ElementSpec

# ``BASIX_TET_EDGES`` (defined in fem_topology) is the basix tetrahedron edge order, which is also
# the P2 edge-midpoint DOF order: the 6 midpoints (after the 4 vertices) sit at vertex pairs
# (2,3),(1,3),(1,2),(0,3),(0,2),(0,1). ``_promote_to_quadratic`` appends midpoint nodes in this order,
# so a mismatch silently scrambles the P2 local DOFs against the tabulated basis.
from .fem_topology import BASIX_TET_EDGES, BASIX_TRIANGLE_EDGES


def _lagrange_basix(cell_type, degree: int):
    """basix Lagrange element of ``degree`` on ``cell_type``.

    Degree > 2 *requires* an explicit Lagrange variant (basix raises ``Lagrange elements of degree > 2
    need to be given a variant`` otherwise); we use **equispaced** nodes so the interpolation points are
    the natural ``1/k``-spaced lattice that :func:`fem_utils._promote_to_degree` places on the global
    mesh. Degree <= 2 keeps the default (``unset``), so P1/P2 are byte-identical to before. The mesh
    node generator and the basis tabulation BOTH go through this one builder, so their DOF order and
    node positions always agree."""
    import basix
    from basix import ElementFamily, LagrangeVariant

    variant = LagrangeVariant.equispaced if degree > 2 else LagrangeVariant.unset
    return basix.create_element(ElementFamily.P, cell_type, degree, variant)


def _ref_hessian_from_tab(tab: np.ndarray, dim: int) -> np.ndarray:
    """Symmetric reference Hessian ``(n_quad, n_dof, 1, dim, dim)`` from a basix ``tabulate(2, qp)`` array.

    The second-derivative blocks are addressed by their derivative multi-index via ``basix.index`` (NOT a
    hardcoded position): entry ``(i, j)`` is the block whose multi-index has a +1 in axes ``i`` and ``j``
    (e.g. 2D ``(0,0) -> index(2,0)=∂ξ₀²``, ``(0,1) -> index(1,1)=∂ξ₀∂ξ₁``)."""
    import basix

    nq, nd = int(tab.shape[1]), int(tab.shape[2])
    H = np.zeros((nq, nd, 1, dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            mi = [0] * dim
            mi[i] += 1
            mi[j] += 1
            blk = np.asarray(tab[basix.index(*mi)])[..., 0]  # (n_quad, n_dof)
            H[:, :, 0, i, j] = blk
            H[:, :, 0, j, i] = blk
    return H


def lagrange_interp_points(dim: int, degree: int) -> np.ndarray:
    """Reference interpolation points of the degree-``k`` Lagrange simplex, in basix DOF order
    (vertices, then per-edge, per-face, interior nodes). :func:`fem_utils._promote_to_degree` maps these
    through each cell's affine geometry to place the global P{k} mesh nodes; the order matches the basis
    tabulated by :func:`lagrange_triangle` / :func:`lagrange_tet` (same builder)."""
    from basix import CellType

    cell = CellType.triangle if dim == 2 else CellType.tetrahedron
    return np.asarray(_lagrange_basix(cell, degree).points)


def lagrange_triangle(degree: int, quad_degree: Optional[int] = None) -> ElementSpec:
    """Lagrange P{degree} element on a reference triangle, tabulated via basix.

    P1 (``degree=1``) has 3 DOFs (vertex nodes); P2 (``degree=2``) has 6 DOFs
    (3 vertex + 3 edge-midpoint nodes, in BASIX_TRIANGLE_EDGES order).

    The returned :class:`~fem_elements.ElementSpec` carries:

    * ``ref_values``  ``(n_quad, n_dof, 1)`` — scalar shape values at quad points.
    * ``ref_grads``   ``(n_quad, n_dof, 1, 2)`` — reference partial derivatives
      ``(∂φ/∂ξ₀, ∂φ/∂ξ₁)`` at quad points.

    Map to physical element data with :func:`identity_pushforward`.
    """
    import basix
    from basix import CellType

    if degree < 1:
        raise ValueError(f"lagrange_triangle: degree must be >= 1; got {degree}.")
    qd = quad_degree if quad_degree is not None else 2 * degree + 1
    elem = _lagrange_basix(CellType.triangle, degree)
    qp, qw = basix.make_quadrature(CellType.triangle, qd)
    tab = elem.tabulate(2, qp)  # (n_blocks, n_quad, n_dof, 1) -- values, 1st and 2nd reference derivatives
    ref_values = np.asarray(tab[0])  # (n_quad, n_dof, 1)
    # Stack ∂φ/∂ξ₀ (tab[1]) and ∂φ/∂ξ₁ (tab[2]) into the last axis
    ref_grads = np.stack([np.asarray(tab[1]), np.asarray(tab[2])], axis=-1)  # (n_quad, n_dof, 1, 2)
    return ElementSpec(
        family=f"Lagrange-P{degree}",
        n_dof=int(elem.dim),
        value_size=1,
        quad_points=np.asarray(qp),
        quad_weights=np.asarray(qw),
        ref_values=ref_values,
        ref_div=None,
        ref_grads=ref_grads,
        local_edges=BASIX_TRIANGLE_EDGES,
        ref_hess=_ref_hessian_from_tab(tab, 2),  # (n_quad, n_dof, 1, 2, 2) for 4th-order weak forms
    )


def lagrange_tet(degree: int, quad_degree: Optional[int] = None) -> ElementSpec:
    """Lagrange P{degree} element on a reference tetrahedron, tabulated via basix.

    P1 has 4 DOFs (vertex nodes); P2 has 10 DOFs (4 vertex + 6 edge-midpoint nodes,
    in BASIX_TET_EDGES order).

    The returned :class:`~fem_elements.ElementSpec` carries:

    * ``ref_values``  ``(n_quad, n_dof, 1)`` — scalar shape values.
    * ``ref_grads``   ``(n_quad, n_dof, 1, 3)`` — reference partial derivatives
      ``(∂φ/∂ξ₀, ∂φ/∂ξ₁, ∂φ/∂ξ₂)``.
    """
    import basix
    from basix import CellType

    if degree < 1:
        raise ValueError(f"lagrange_tet: degree must be >= 1; got {degree}.")
    qd = quad_degree if quad_degree is not None else 2 * degree + 1
    elem = _lagrange_basix(CellType.tetrahedron, degree)
    qp, qw = basix.make_quadrature(CellType.tetrahedron, qd)
    tab = elem.tabulate(2, qp)  # (n_blocks, n_quad, n_dof, 1) -- values, 1st and 2nd reference derivatives
    ref_values = np.asarray(tab[0])  # (n_quad, n_dof, 1)
    ref_grads = np.stack([np.asarray(tab[i]) for i in range(1, 4)], axis=-1)  # (n_quad, n_dof, 1, 3)
    return ElementSpec(
        family=f"Lagrange-P{degree}-Tet",
        n_dof=int(elem.dim),
        value_size=1,
        quad_points=np.asarray(qp),
        quad_weights=np.asarray(qw),
        ref_values=ref_values,
        ref_div=None,
        ref_grads=ref_grads,
        local_edges=BASIX_TET_EDGES,
        ref_hess=_ref_hessian_from_tab(tab, 3),  # (n_quad, n_dof, 1, 3, 3) for 4th-order weak forms
    )


def identity_pushforward(
    ref_values: jnp.ndarray,
    ref_grads: jnp.ndarray,
    J: jnp.ndarray,
    detJ: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Isoparametric push-forward of scalar Lagrange basis data to a physical cell.

    Scalar nodal values are coordinate-invariant (the push-forward is the identity
    on values); gradients transform via the chain rule:

        ``∂φ/∂x = ∂φ/∂ξ J⁻¹``

    Parameters
    ----------
    ref_values : ``(n_quad, n_dof, 1)``  scalar shape values from basix tabulation.
    ref_grads  : ``(n_quad, n_dof, 1, tdim)``  ``∂φ/∂ξ`` from basix (value_size=1).
    J          : ``(tdim, tdim)``  affine cell Jacobian (columns = physical edge vectors).
    detJ       : scalar, unused (kept for API symmetry with the Piola push-forwards).

    Returns
    -------
    phi       : ``(n_quad, n_dof)``  physical shape values (``= ref_values[..., 0]``).
    dphi_phys : ``(n_quad, n_dof, tdim)``  physical gradients ``∂φ/∂x``.
    """
    K = jnp.linalg.inv(J)  # J⁻¹, (tdim, tdim)
    phi = ref_values[..., 0]  # (n_quad, n_dof)
    dphi_ref = ref_grads[..., 0, :]  # (n_quad, n_dof, tdim)
    dphi_phys = jnp.einsum("qnd,dD->qnD", dphi_ref, K)  # (n_quad, n_dof, tdim)
    return phi, dphi_phys


def identity_pushforward_hess(ref_hess: jnp.ndarray, J: jnp.ndarray) -> jnp.ndarray:
    """Physical Hessian of a scalar Lagrange basis on an **affine** simplex.

    The geometry is always P1 (straight-sided), so the reference→physical map ``ξ ↦ x`` is affine with a
    **constant** Jacobian ``J``; thus ``∂²ξ/∂x² ≡ 0`` and the second derivatives transform by the clean
    chain rule with no curvature term::

        ``∂²φ/∂x_a∂x_b = K_ia K_jb ∂²φ/∂ξ_i∂ξ_j``,   ``K = J⁻¹``.

    Parameters
    ----------
    ref_hess : ``(n_quad, n_dof, 1, tdim, tdim)``  reference second derivatives ``∂²φ/∂ξ∂ξ`` (value_size=1).
    J        : ``(tdim, tdim)``  affine cell Jacobian.

    Returns
    -------
    hess_phys : ``(n_quad, n_dof, tdim, tdim)``  physical Hessian ``∂²φ/∂x∂x`` (symmetric).
    """
    K = jnp.linalg.inv(J)  # J⁻¹
    return jnp.einsum("qnij,ia,jb->qnab", ref_hess[..., 0, :, :], K, K)
