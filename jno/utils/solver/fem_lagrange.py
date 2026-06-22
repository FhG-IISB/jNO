"""Lagrange element factories for the native 2D/3D assembler.

Wraps basix tabulation into :class:`~fem_elements.ElementSpec` for scalar Lagrange
families (P1, P2) on triangles and tetrahedra, and exposes :func:`identity_pushforward`
— the isoparametric chain-rule map from reference to physical gradients:

    ``∂φ/∂x = ∂φ/∂ξ J⁻¹``

This is the Lagrange analogue of the contravariant and covariant Piola maps in
:mod:`fem_elements`.  The DOF ordering produced by :func:`lagrange_triangle` (degree 2)
matches the edge-midpoint layout of :func:`feax_utils._promote_to_quadratic` with
``edge_local = BASIX_TRIANGLE_EDGES``, so a promoted P2 mesh indexes correctly into the
shape functions.

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
from .fem_topology import BASIX_TRIANGLE_EDGES

# Basix edge ordering for a tetrahedron (6 edges, one P2 midpoint DOF each).
# Matches basix entity_dofs for P, CellType.tetrahedron, degree=2:
# edges (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) in that order.
BASIX_TET_EDGES: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


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
    from basix import CellType, ElementFamily

    if degree not in (1, 2):
        raise NotImplementedError(f"lagrange_triangle: degree must be 1 or 2; got {degree}.")
    qd = quad_degree if quad_degree is not None else 2 * degree + 1
    elem = basix.create_element(ElementFamily.P, CellType.triangle, degree)
    qp, qw = basix.make_quadrature(CellType.triangle, qd)
    tab = elem.tabulate(1, qp)  # (1 + tdim, n_quad, n_dof, 1)
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
    from basix import CellType, ElementFamily

    if degree not in (1, 2):
        raise NotImplementedError(f"lagrange_tet: degree must be 1 or 2; got {degree}.")
    qd = quad_degree if quad_degree is not None else 2 * degree + 1
    elem = basix.create_element(ElementFamily.P, CellType.tetrahedron, degree)
    qp, qw = basix.make_quadrature(CellType.tetrahedron, qd)
    tab = elem.tabulate(1, qp)  # (1 + tdim, n_quad, n_dof, 1)
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
