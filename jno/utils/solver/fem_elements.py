"""Reference tabulation + per-cell push-forward for non-nodal element families.

basix tabulates each family on the *reference* cell (eager numpy, constant); this
module wraps that into an :class:`ElementSpec` and pairs it with the per-cell
push-forward that maps the reference basis to a physical triangle. The map is pure
linear algebra in the cell Jacobian, so it is JAX-friendly (vmappable over cells,
differentiable in the geometry).

The nodal-Lagrange path needs none of this — its only map is the isoparametric
gradient chain rule. The edge/derivative-DOF families do:

* Raviart–Thomas (H(div)): contravariant Piola
  ``Phi_phys = (1/detJ) J Phi_ref``, ``div Phi_phys = (1/detJ) div Phi_ref``.
* Nédélec (H(curl)): covariant Piola ``Phi_phys = J^{-T} Phi_ref``,
  ``curl Phi_phys = (1/detJ) curl Phi_ref`` (2-D scalar curl).
* Hermite (C0, vertex value + first-derivative DOFs): per-cell ``M(cell)`` DOF-transform
  (``M = blockdiag(1, J)`` per vertex) -- the first DOF-*mixing* map (see :func:`hermite_M`). The
  C1 Bell/Argyris elements extend the same machinery to second-derivative + edge-normal DOFs *(later)*.

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

from typing import Any, NamedTuple, Optional, Tuple

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
    ``ref_hess``    reference second derivatives ``d²(Phi_ref) / d(xi)_i d(xi)_j``,
                    ``(n_quad, n_dof, value_size, tdim, tdim)`` (symmetric) — used to assemble
                    4th-order (biharmonic) weak forms via :func:`identity_pushforward_hess`;
                    ``None`` for families that don't tabulate it.
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
    ref_hess: Optional[np.ndarray] = None  # (n_quad, n_dof, value_size, tdim, tdim), reference d²Phi/dxi dxi
    ref_aux: Optional[Any] = None  # element-specific extra reference data (e.g. Argyris nodal tabulations for M(cell))


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


# ---------------------------------------------------------------------------
# Cubic Hermite (C0, vertex value + first-derivative DOFs) -- the foundation for the
# derivative-DOF / DOF-mixing transform that C1 elements (Bell/Argyris) build on.
# ---------------------------------------------------------------------------


def hermite_triangle(quad_degree: int = 6) -> ElementSpec:
    """Cubic Hermite element on the reference triangle, via basix (``ElementFamily.Hermite``, degree 3).

    A *scalar* element with **derivative DOFs**. basix DOF order (verified): per vertex
    ``(value, ∂/∂ξ₀, ∂/∂ξ₁)`` for vertices 0,1,2, then one interior (centroid value) DOF -> ``n_dof=10``.
    The reference basis is mapped to a physical cell by the per-cell **DOF-transform** ``M(cell)``
    (:func:`hermite_M`) -- a DOF-*mixing* matrix (the value-Piola maps only scale each DOF), so the global
    derivative DOFs are physical-coordinate derivatives ``∂/∂x, ∂/∂y``. Carries ``ref_hess`` so a 4th-order
    form assembles (note: cubic Hermite is C⁰, so it is non-conforming for biharmonic -- the C¹ Bell/Argyris
    elements reuse this machinery)."""
    import basix
    from basix import CellType, ElementFamily

    from .fem_lagrange import _ref_hessian_from_tab

    elem = basix.create_element(ElementFamily.Hermite, CellType.triangle, 3)
    qp, qw = basix.make_quadrature(CellType.triangle, quad_degree)
    tab = elem.tabulate(2, qp)  # (n_blocks, n_quad, n_dof, 1)
    return ElementSpec(
        family="Hermite-Tri",
        n_dof=int(elem.dim),
        value_size=1,
        quad_points=np.asarray(qp),
        quad_weights=np.asarray(qw),
        ref_values=np.asarray(tab[0]),
        ref_div=None,
        ref_grads=np.stack([np.asarray(tab[1]), np.asarray(tab[2])], axis=-1),  # (n_quad, 10, 1, 2)
        local_edges=(),
        ref_hess=_ref_hessian_from_tab(tab, 2),  # (n_quad, 10, 1, 2, 2)
    )


def hermite_M(J: jnp.ndarray) -> jnp.ndarray:
    """Per-cell DOF-transform ``M(cell)`` (10×10) for cubic Hermite.

    Block-diagonal over the three vertices: the value DOF is unchanged and the two first-derivative DOFs
    transform by the cell Jacobian ``J`` (so a global derivative DOF is ``∂u/∂x``/``∂u/∂y`` at the vertex,
    not the reference ``∂u/∂ξ``); the interior centroid-value DOF is unchanged. Derived and numerically
    validated (the derivative block is ``J``, not ``Jᵀ``). The physical basis is ``Φ = M φ̂``."""
    M = jnp.eye(10, dtype=J.dtype)
    for vb in (0, 3, 6):  # each vertex block: DOF vb=value, vb+1/vb+2 = ∂ξ₀/∂ξ₁
        M = M.at[vb + 1 : vb + 3, vb + 1 : vb + 3].set(J)
    return M


def hermite_pushforward(ref_values, ref_grads, ref_hess, J, detJ, signs):
    """Map the reference Hermite basis to a physical cell: ``Φ = M φ̂`` (value), the chain-ruled physical
    gradient/Hessian then left-multiplied by ``M`` on the DOF axis. Scalar field -> shapes match nodal
    Lagrange ``(n_quad, n_dof)`` / ``(…, tdim)`` / ``(…, tdim, tdim)``, so the shared evaluator treats a
    Hermite field exactly like a scalar Lagrange one (the ``signs`` arg is unused -- Hermite has no edge
    orientation -- kept for a uniform push-forward signature)."""
    M = hermite_M(J)
    K = jnp.linalg.inv(J)
    phi = jnp.einsum("ab,qb->qa", M, ref_values[..., 0])  # (n_quad, n_dof)
    dphys = jnp.einsum("qbi,id->qbd", ref_grads[..., 0, :], K)  # reference grad -> physical (chain rule)
    grad = jnp.einsum("ab,qbd->qad", M, dphys)  # (n_quad, n_dof, tdim)
    hphys = jnp.einsum("qbij,ia,jc->qbac", ref_hess[..., 0, :, :], K, K)  # Kᵀ H_ref K
    hess = jnp.einsum("ab,qbij->qaij", M, hphys)  # (n_quad, n_dof, tdim, tdim)
    return phi, grad, hess


# ---------------------------------------------------------------------------
# Argyris (C¹ conforming, quintic, 21 DOF) — the real conforming biharmonic element.
#
# DOFs (per the classical TUBA-6 / Argyris element): at each of the 3 vertices the value, the two
# first derivatives and the three second derivatives (6 each = 18), plus the normal derivative at each
# of the 3 edge midpoints (= 3). The edge-normal DOFs are what make the element C¹ on an *unstructured*
# mesh where the simpler vertex-only reduced-quintic (Bell) fails: the cross-cell normal derivative is
# pinned by a DOF shared (with a globally consistent normal) by the two incident cells. basix has no
# Argyris family, so the reference dual basis is built here from the monomials (pure numpy, a constant),
# and the per-cell physical element uses the affine-equivalence DOF-transform ``M(cell)`` (Kirby 2018) —
# a true DOF-*mixing* map that, unlike Hermite's ``blockdiag(1, J)``, couples the second-derivative DOFs
# (``KᵀHK``) and scales the edge-normal DOFs by the *physical* edge normal.
#
# References
# ----------
# J.H. Argyris, I. Fried, D.W. Scharpf, "The TUBA family of plate elements for the matrix displacement
#   method", Aeronautical Journal 72 (1968) 701–709 — the original quintic C¹ triangle.
# R.C. Kirby, "A general approach to transforming finite elements", SMAI J. Comput. Math. 4 (2018)
#   197–224 — the functional/affine transform ``M[m,k] = ℓ_m^phys(φ̂_k)`` used to map the reference dual
#   basis to a physical cell while preserving C¹ conformity.
# ---------------------------------------------------------------------------

# Reference geometry (standard triangle) and edge ordering, shared with the edge topology so DOF k ↔ the
# global edge id ``cell_edges[c, k]``.
_ARG_REF_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
_ARG_REF_EDGE_MIDS = np.array(
    [
        0.5 * (_ARG_REF_VERTS[a] + _ARG_REF_VERTS[b])  # midpoint of reference edge (a, b)
        for (a, b) in BASIX_TRIANGLE_EDGES  # ((1,2),(0,2),(0,1))
    ]
)


def _arg_mono_exps() -> Tuple[Tuple[int, int], ...]:
    """Exponents of the 21 P5 monomials ``x^i y^j`` (``i+j ≤ 5``), ascending total degree."""
    return tuple((i, total - i) for total in range(6) for i in range(total + 1))


_ARG_EXPS = _arg_mono_exps()  # length 21


def _arg_mono_value(pt: np.ndarray) -> np.ndarray:
    x, y = pt
    return np.array([x**i * y**j for (i, j) in _ARG_EXPS])  # (21,)


def _arg_mono_grad(pt: np.ndarray) -> np.ndarray:
    x, y = pt
    g = np.zeros((21, 2))
    for k, (i, j) in enumerate(_ARG_EXPS):
        g[k, 0] = i * (x ** (i - 1) if i >= 1 else 0.0) * (y**j)
        g[k, 1] = j * (x**i) * (y ** (j - 1) if j >= 1 else 0.0)
    return g  # (21, 2) columns ∂x, ∂y


def _arg_mono_hess(pt: np.ndarray) -> np.ndarray:
    x, y = pt
    H = np.zeros((21, 2, 2))
    for k, (i, j) in enumerate(_ARG_EXPS):
        H[k, 0, 0] = i * (i - 1) * (x ** (i - 2) if i >= 2 else 0.0) * (y**j)
        H[k, 0, 1] = i * j * (x ** (i - 1) if i >= 1 else 0.0) * (y ** (j - 1) if j >= 1 else 0.0)
        H[k, 1, 0] = H[k, 0, 1]
        H[k, 1, 1] = j * (j - 1) * (x**i) * (y ** (j - 2) if j >= 2 else 0.0)
    return H  # (21, 2, 2)


def _arg_ref_normals() -> np.ndarray:
    """Reference unit edge normals (only fix the intermediate reference dual basis; the final physical
    element is independent of their sign — the physical edge normal is what enters ``M(cell)``)."""
    out = []
    for a, b in BASIX_TRIANGLE_EDGES:
        d = _ARG_REF_VERTS[b] - _ARG_REF_VERTS[a]
        n = np.array([-d[1], d[0]])  # +90° rotation
        out.append(n / np.linalg.norm(n))
    return np.array(out)


def _arg_ref_functionals(coeffs: np.ndarray, K: np.ndarray, edge_normals: np.ndarray) -> np.ndarray:
    """Apply the 21 (physical) Argyris functionals to a field given by its reference-monomial coefficient
    vector ``coeffs`` (a function of ξ). The physical field is ``φ̂(x) = field(F⁻¹(x))``; chain rule
    ``∇ₓ = Kᵀ ∇_ξ``, ``Hₓ = Kᵀ H_ξ K``. Vertex functionals at the reference vertices, edge functionals at
    the reference edge midpoints dotted with the supplied (physical) edge normal. With ``K = I`` and the
    reference normals this builds the reference dual basis; with the cell ``K`` and physical normals it
    builds ``M(cell)``."""
    KT = K.T
    out = []
    for v in _ARG_REF_VERTS:
        val = coeffs @ _arg_mono_value(v)
        grad = KT @ (_arg_mono_grad(v).T @ coeffs)
        H = KT @ np.einsum("k,kab->ab", coeffs, _arg_mono_hess(v)) @ K
        out += [val, grad[0], grad[1], H[0, 0], H[0, 1], H[1, 1]]
    for mid, n in zip(_ARG_REF_EDGE_MIDS, edge_normals):
        grad = KT @ (_arg_mono_grad(mid).T @ coeffs)
        out.append(n @ grad)
    return np.array(out)


def _arg_reference_coeffs() -> np.ndarray:
    """Reference dual-basis monomial coefficients ``S`` (21×21): ``V_ref[i,j] = ℓ_i(monomial_j)``,
    ``S = V_ref⁻¹``, so basis ``ψ_k = Σ_p S[p,k] monomial_p`` satisfies ``ℓ_i(ψ_k) = δ_ik``."""
    refn = _arg_ref_normals()
    cols = []
    for j in range(21):
        ej = np.zeros(21)
        ej[j] = 1.0
        cols.append(_arg_ref_functionals(ej, np.eye(2), refn))
    V_ref = np.column_stack(cols)  # V_ref[:, j] = functionals of monomial j
    return np.linalg.inv(V_ref)


def argyris_triangle(quad_degree: int = 12) -> ElementSpec:
    """Argyris C¹ quintic triangle (21 DOF), built from the monomials (basix has no Argyris family).

    Returns the reference dual basis ``ψ_k`` (and its gradient/Hessian) tabulated at a quintic-exact
    triangle quadrature, plus — in ``ref_aux`` — the reference nodal tabulations (value/grad/Hessian at the
    3 vertices, gradient at the 3 edge midpoints) that :func:`argyris_pushforward` needs to assemble the
    per-cell DOF-transform ``M(cell)``. ``local_edges = BASIX_TRIANGLE_EDGES`` so the edge-normal DOF ``k``
    aligns with the global edge ``cell_edges[c, k]``."""
    import basix
    from basix import CellType

    S = _arg_reference_coeffs()  # (21, 21)
    qp, qw = basix.make_quadrature(CellType.triangle, quad_degree)
    qp = np.asarray(qp)
    nq = qp.shape[0]
    rv = np.zeros((nq, 21, 1))
    rg = np.zeros((nq, 21, 1, 2))
    rh = np.zeros((nq, 21, 1, 2, 2))
    for q in range(nq):
        rv[q, :, 0] = _arg_mono_value(qp[q]) @ S
        rg[q, :, 0, :] = np.einsum("pl,pk->kl", _arg_mono_grad(qp[q]), S)
        rh[q, :, 0, :, :] = np.einsum("pab,pk->kab", _arg_mono_hess(qp[q]), S)
    # reference nodal tabulations of ψ_k for building M(cell)
    nv_val = np.array([_arg_mono_value(v) @ S for v in _ARG_REF_VERTS])  # (3, 21)
    nv_grad = np.array([np.einsum("pl,pk->kl", _arg_mono_grad(v), S) for v in _ARG_REF_VERTS])  # (3, 21, 2)
    nv_hess = np.array([np.einsum("pab,pk->kab", _arg_mono_hess(v), S) for v in _ARG_REF_VERTS])  # (3, 21, 2, 2)
    ne_grad = np.array([np.einsum("pl,pk->kl", _arg_mono_grad(m), S) for m in _ARG_REF_EDGE_MIDS])  # (3, 21, 2)
    return ElementSpec(
        family="Argyris-Tri",
        n_dof=21,
        value_size=1,
        quad_points=qp,
        quad_weights=np.asarray(qw),
        ref_values=rv,
        ref_div=None,
        ref_grads=rg,
        local_edges=BASIX_TRIANGLE_EDGES,
        ref_hess=rh,
        ref_aux=(nv_val, nv_grad, nv_hess, ne_grad),
    )


def argyris_pushforward(ref_values, ref_grads, ref_hess, J, detJ, edge_normals, nodal):
    """Map the reference Argyris basis to a physical cell via the affine DOF-transform ``M(cell)``.

    ``M[m,k] = ℓ_m^phys(ψ̂_k)`` applies the 21 *physical* functionals to the mapped reference basis: the
    vertex value rows are ``ψ_k(ξ_i)``; the gradient rows ``Kᵀ ∇_ξψ_k(ξ_i)``; the Hessian rows the three
    independent entries of ``Kᵀ H_ξψ_k(ξ_i) K``; the edge rows ``n_e · Kᵀ ∇_ξψ_k(ξ_mid_e)`` with ``n_e``
    the **physical, globally-oriented** edge normal (the cross-cell C¹ ingredient). The physical basis is
    ``φ_n = Σ_k C[k,n] ψ̂_k`` with ``C = M⁻¹``; tabulating it at the quad points gives ``shape_vals``,
    and the chain rule (``∇ₓ = Kᵀ∇_ξ``, ``Hₓ = Kᵀ H_ξ K``) gives ``shape_grads`` / ``shape_hess``. A scalar
    field → shapes match nodal Lagrange, so the shared evaluator serves it once ``M`` is baked in.

    ``nodal = (nv_val, nv_grad, nv_hess, ne_grad)`` are the reference nodal tabulations from
    :func:`argyris_triangle`; ``edge_normals`` is the cell's ``(3, 2)`` physical edge normals (one per local
    edge ``k``, globally oriented)."""
    nv_val, nv_grad, nv_hess, ne_grad = nodal
    K = jnp.linalg.inv(J)
    # M(cell): assemble the 21 functional rows column-blockwise over the basis index k.
    vg = jnp.einsum("lm,vkl->vkm", K, nv_grad)  # (3,21,2) Kᵀ∇ at each vertex
    vh = jnp.einsum("pa,vkpq,qb->vkab", K, nv_hess, K)  # (3,21,2,2) Kᵀ H K at each vertex
    eg = jnp.einsum("lm,ekl->ekm", K, ne_grad)  # (3,21,2) Kᵀ∇ at each edge midpoint
    en = jnp.einsum("em,ekm->ek", edge_normals, eg)  # (3,21) n_e · Kᵀ∇
    rows = []
    for v in range(3):
        rows += [nv_val[v], vg[v, :, 0], vg[v, :, 1], vh[v, :, 0, 0], vh[v, :, 0, 1], vh[v, :, 1, 1]]
    for e in range(3):
        rows.append(en[e])
    M = jnp.stack(rows, axis=0)  # (21, 21)
    C = jnp.linalg.inv(M)
    rv = ref_values[..., 0]  # (n_quad, 21)
    rg = ref_grads[..., 0, :]  # (n_quad, 21, 2)
    rh = ref_hess[..., 0, :, :]  # (n_quad, 21, 2, 2)
    phi = jnp.einsum("qk,kn->qn", rv, C)  # (n_quad, 21)
    grad = jnp.einsum("qkl,kn,lm->qnm", rg, C, K)  # (n_quad, 21, 2) physical gradient
    hess = jnp.einsum("qkij,kn,ia,jb->qnab", rh, C, K, K)  # (n_quad, 21, 2, 2) physical Hessian
    return phi, grad, hess
