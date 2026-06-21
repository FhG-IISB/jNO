"""Reference tabulation + per-cell push-forward for non-nodal element families.

basix tabulates each family on the *reference* cell (eager numpy, constant); this
module wraps that into an :class:`ElementSpec` and pairs it with the per-cell
push-forward that maps the reference basis to a physical triangle. The map is pure
linear algebra in the cell Jacobian, so it is JAX-friendly (vmappable over cells,
differentiable in the geometry).

The nodal-Lagrange path needs none of this — its only map is the isoparametric
gradient chain rule already in feax. The edge/derivative-DOF families do:

* Raviart–Thomas (H(div)): contravariant Piola
  ``Phi_phys = (1/detJ) J Phi_ref``, ``div Phi_phys = (1/detJ) div Phi_ref``.
* Nédélec (H(curl)): covariant Piola ``Phi_phys = J^{-T} Phi_ref``,
  ``curl Phi_phys = (1/detJ) curl Phi_ref`` (2-D scalar curl).
* Argyris (C1): per-cell Kirby ``M(cell)`` on derivative DOFs  *(later)*.

A per-DOF orientation sign (from :mod:`fem_topology`) multiplies the whole basis
function so the two cells sharing an edge agree on its sign. The Piola formula and
the ``div_phys`` identity are validated against ``basix``'s own ``push_forward``.

References
----------
P.-A. Raviart, J.-M. Thomas, *A mixed finite element method for 2nd order elliptic
problems*, in Mathematical Aspects of FEM, Lecture Notes in Math. 606 (1977).
J.-C. Nédélec, *Mixed finite elements in* R^3, Numer. Math. 35 (1980) 315–341
(first-kind H(curl) edge elements; the 2-D triangle restriction used here).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .fem_topology import BASIX_TRIANGLE_EDGES


class ElementSpec(NamedTuple):
    """Reference-cell data for one non-nodal element family on triangles.

    ``ref_values``  basis values at quad points, ``(n_quad, n_dof, value_size)``.
    ``ref_div``     reference divergence ``(n_quad, n_dof)`` for H(div) families
                    (``None`` otherwise).
    ``ref_grads``   reference gradient ``d(Phi_ref)_k / d(xi)_m`` of the vector basis,
                    ``(n_quad, n_dof, value_size, tdim)`` — used by the divergence/gradient
                    push-forward; ``None`` for families that don't need it.
    ``local_edges`` the family's reference edge ordering (DOF k <-> edge k for the
                    lowest-order edge elements), matching :mod:`fem_topology`.
    ``ref_curl``    reference (2-D scalar) curl ``(n_quad, n_dof)`` for H(curl) families
                    (``None`` otherwise).
    """

    family: str
    n_dof: int
    value_size: int
    quad_points: np.ndarray  # (n_quad, tdim)
    quad_weights: np.ndarray  # (n_quad,)
    ref_values: np.ndarray  # (n_quad, n_dof, value_size)
    ref_div: Optional[np.ndarray]  # (n_quad, n_dof) for H(div), else None
    ref_grads: Optional[np.ndarray]  # (n_quad, n_dof, value_size, tdim), reference d Phi_k / d xi_m
    local_edges: Tuple[Tuple[int, int], ...]
    ref_curl: Optional[np.ndarray] = None  # (n_quad, n_dof) for H(curl), else None


def raviart_thomas_triangle(degree: int = 1, quad_degree: int = 2) -> ElementSpec:
    """Lowest-order (``degree=1``) Raviart–Thomas on a triangle, tabulated via basix.

    RT(degree 1) has 3 DOFs, one per edge (``num_entity_dofs == [[0,0,0],[1,1,1],[0]]``),
    a vector value (``value_size == 2``) and a per-basis-constant divergence.
    """
    import basix
    from basix import CellType, ElementFamily

    elem = basix.create_element(ElementFamily.RT, CellType.triangle, degree)
    qp, qw = basix.make_quadrature(CellType.triangle, quad_degree)
    tab = elem.tabulate(1, qp)  # (1 + tdim, n_quad, n_dof, value_size)
    ref_values = np.asarray(tab[0])  # (n_quad, n_dof, 2)
    # reference gradient d(Phi_ref)_k / d(xi)_m: tab[1] = d/dxi0, tab[2] = d/dxi1 -> stack on m
    ref_grads = np.stack([np.asarray(tab[1]), np.asarray(tab[2])], axis=-1)  # (n_quad, n_dof, 2, 2)
    # divergence = d(Phi_x)/dxi0 + d(Phi_y)/dxi1 (trace of the reference gradient)
    ref_div = ref_grads[:, :, 0, 0] + ref_grads[:, :, 1, 1]  # (n_quad, n_dof)
    return ElementSpec(
        family="RT",
        n_dof=elem.dim,
        value_size=elem.value_size,
        quad_points=np.asarray(qp),
        quad_weights=np.asarray(qw),
        ref_values=ref_values,
        ref_div=ref_div,
        ref_grads=ref_grads,
        local_edges=BASIX_TRIANGLE_EDGES,
    )


def piola_contravariant(
    ref_values: jnp.ndarray, ref_div: jnp.ndarray, J: jnp.ndarray, detJ: jnp.ndarray, signs: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Map RT reference data to one physical triangle (contravariant Piola + sign).

    ``ref_values`` ``(n_quad, n_dof, 2)``, ``ref_div`` ``(n_quad, n_dof)``, ``J`` the
    ``(2, 2)`` affine Jacobian, ``detJ`` its determinant, ``signs`` the per-DOF edge
    orientation ``(n_dof,)``. Returns physical ``(values (n_quad, n_dof, 2), div
    (n_quad, n_dof))``. The orientation sign multiplies the whole basis function, so
    both the value and its divergence carry it.
    """
    s = signs[None, :]  # (1, n_dof)
    values = jnp.einsum("ij,qnj->qni", J, ref_values) / detJ * s[:, :, None]
    div = ref_div / detJ * s
    return values, div


def piola_contravariant_grad(ref_grads: jnp.ndarray, J: jnp.ndarray, detJ: jnp.ndarray, signs: jnp.ndarray) -> jnp.ndarray:
    """Physical gradient of the contravariant-Piola RT basis on one (affine) triangle.

    For ``Phi_phys = (1/detJ) J Phi_ref`` the chain rule gives
    ``d(Phi_phys)_i / dx_l = (1/detJ) J_ik d(Phi_ref)_k/d(xi)_m K_ml`` with ``K = J^{-1}``.
    ``ref_grads`` is ``(n_quad, n_dof, value_size, tdim)``; returns the physical gradient
    ``(n_quad, n_dof, value_size i, tdim l)`` with the per-DOF orientation ``signs`` applied.
    Tracing over ``(i, l)`` recovers the RT divergence (the invariant the test pins).
    """
    K = jnp.linalg.inv(J)
    grad = jnp.einsum("ik,qnkm,ml->qnil", J, ref_grads, K) / detJ
    return grad * signs[None, :, None, None]


def nedelec_triangle(degree: int = 1, quad_degree: int = 2) -> ElementSpec:
    """Lowest-order (``degree=1``) Nédélec first-kind (edge) element on a triangle, via basix.

    N1E(degree 1) has 3 DOFs, one tangential moment per edge (``num_entity_dofs ==
    [[0,0,0],[1,1,1],[0]]``), a vector value (``value_size == 2``) and a per-basis-constant
    (2-D scalar) curl. The H(curl) counterpart of :func:`raviart_thomas_triangle`.
    """
    import basix
    from basix import CellType, ElementFamily

    elem = basix.create_element(ElementFamily.N1E, CellType.triangle, degree)
    qp, qw = basix.make_quadrature(CellType.triangle, quad_degree)
    tab = elem.tabulate(1, qp)  # (1 + tdim, n_quad, n_dof, value_size)
    ref_values = np.asarray(tab[0])  # (n_quad, n_dof, 2)
    ref_grads = np.stack([np.asarray(tab[1]), np.asarray(tab[2])], axis=-1)  # (n_quad, n_dof, 2, 2)
    # 2-D scalar curl = d(Phi_y)/dxi0 - d(Phi_x)/dxi1 (the antisymmetric part of the reference gradient)
    ref_curl = ref_grads[:, :, 1, 0] - ref_grads[:, :, 0, 1]  # (n_quad, n_dof)
    return ElementSpec(
        family="N1E",
        n_dof=elem.dim,
        value_size=elem.value_size,
        quad_points=np.asarray(qp),
        quad_weights=np.asarray(qw),
        ref_values=ref_values,
        ref_div=None,
        ref_grads=ref_grads,
        local_edges=BASIX_TRIANGLE_EDGES,
        ref_curl=ref_curl,
    )


def piola_covariant(
    ref_values: jnp.ndarray, ref_curl: jnp.ndarray, J: jnp.ndarray, detJ: jnp.ndarray, signs: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Map Nédélec reference data to one physical triangle (covariant Piola + sign).

    ``Phi_phys = J^{-T} Phi_ref`` (preserves the tangential trace); the 2-D scalar curl transforms
    as ``curl Phi_phys = (1/detJ) curl Phi_ref`` (signed ``detJ``). ``ref_values`` ``(n_quad, n_dof,
    2)``, ``ref_curl`` ``(n_quad, n_dof)``, ``J`` the ``(2, 2)`` affine Jacobian, ``signs`` the
    per-DOF edge orientation ``(n_dof,)``. Returns physical ``(values (n_quad, n_dof, 2), curl
    (n_quad, n_dof))``; the orientation sign multiplies both.
    """
    K = jnp.linalg.inv(J)  # J^{-1}; the covariant map J^{-T} gives Phi_phys_i = K_ji Phi_ref_j
    s = signs[None, :]  # (1, n_dof)
    values = jnp.einsum("ji,qnj->qni", K, ref_values) * s[:, :, None]
    curl = ref_curl / detJ * s
    return values, curl


def piola_covariant_grad(ref_grads: jnp.ndarray, J: jnp.ndarray, detJ: jnp.ndarray, signs: jnp.ndarray) -> jnp.ndarray:
    """Physical gradient of the covariant-Piola Nédélec basis on one (affine) triangle.

    For ``Phi_phys_i = K_ji Phi_ref_j`` (``K = J^{-1}``) the chain rule gives
    ``d(Phi_phys)_i / dx_l = K_ji d(Phi_ref)_j/d(xi)_m K_ml`` — no ``detJ`` (the covariant value has
    none). ``ref_grads`` is ``(n_quad, n_dof, value_size, tdim)``; returns ``(n_quad, n_dof, i, l)``
    with the per-DOF ``signs`` applied. The off-diagonal ``grad[..., 1, 0] - grad[..., 0, 1]`` recovers
    the physical curl ``(1/detJ) curl_ref`` (the invariant the test pins).
    """
    K = jnp.linalg.inv(J)
    grad = jnp.einsum("ji,qnjm,ml->qnil", K, ref_grads, K)
    return grad * signs[None, :, None, None]
