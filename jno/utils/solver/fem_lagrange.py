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

References:
    P.G. Ciarlet, *The Finite Element Method for Elliptic Problems*, SIAM (2002),
    §§ 2.2–2.3 (Lagrange elements on simplices, isoparametric mapping).
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .fem_elements import ElementSpec

# ``BASIX_TET_EDGES`` (defined in fem_topology) is the basix tetrahedron edge order, which is also
# the P2 edge-midpoint DOF order: the 6 midpoints (after the 4 vertices) sit at vertex pairs
# (2,3),(1,3),(1,2),(0,3),(0,2),(0,1). ``_promote_to_quadratic`` appends midpoint nodes in this order,
# so a mismatch silently scrambles the P2 local DOFs against the tabulated basis.
from .fem_topology import BASIX_TET_EDGES, BASIX_TRIANGLE_EDGES
from .small_linalg import small_inv


def _lagrange_basix(cell_type, degree: int):
    """basix Lagrange element of ``degree`` on ``cell_type``.

    Degree > 2 *requires* an explicit Lagrange variant (basix raises ``Lagrange elements of degree > 2
    need to be given a variant`` otherwise). On a **quadrilateral or hexahedron** it is the
    Gauss-Lobatto-Legendre (GLL) node set, ``gll_warped`` (the tensor product of the 1-D GLL points): the
    nodes of a spectral element, whose interpolation stays well conditioned as the degree grows (an
    equispaced Q6 interpolant has a Lebesgue constant of ~30, the GLL one ~2.7) -- the F4 item of
    plans/fem-basis-function-gaps.md. On a simplex it stays **equispaced**, the natural ``1/k``-spaced
    lattice. Degree <= 2 keeps the default (``unset``), so P1/P2 and Q1/Q2 are byte-identical to before. The
    mesh node generator (:func:`fem_utils._promote_to_degree`, through :func:`lagrange_interp_points`) and
    the basis tabulation BOTH go through this one builder, so their DOF order and node positions always
    agree."""
    import basix
    from basix import CellType, ElementFamily, LagrangeVariant

    if degree <= 2:
        variant = LagrangeVariant.unset
    elif cell_type in (CellType.quadrilateral, CellType.hexahedron):
        variant = LagrangeVariant.gll_warped
    else:
        variant = LagrangeVariant.equispaced
    return basix.create_element(ElementFamily.P, cell_type, degree, variant)


def edge_node_parameters(cell_type: str, degree: int) -> np.ndarray:
    """Parameters in ``[0, 1]``, ascending, of a degree-``degree`` Lagrange element's nodes along one edge of a
    ``"quad"`` / ``"hexahedron"`` cell (any other name: a simplex).

    ``k / degree`` up to degree 2 and on a simplex; the Gauss-Lobatto-Legendre points on a quadrilateral or
    hexahedron above. Read off the element :func:`_lagrange_basix` builds (its nodes on the reference edge
    ``y = z = 0``), so whoever places or looks for DOFs along an edge -- the hanging-node constraints -- follows
    the node set the basis really has."""
    from basix import CellType

    cell = {"quad": CellType.quadrilateral, "hexahedron": CellType.hexahedron}.get(cell_type, CellType.triangle)
    pts = np.asarray(_lagrange_basix(cell, int(degree)).points, dtype=float)
    on_edge = np.all(np.abs(pts[:, 1:]) < 1e-12, axis=1)
    return np.sort(pts[on_edge, 0])


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


def lagrange_interp_points(dim: int, degree: int, cell_type: Optional[str] = None) -> np.ndarray:
    """Reference interpolation points of the degree-``k`` Lagrange element, in basix DOF order
    (vertices first, then per-edge, per-face, interior nodes).

    :func:`fem_utils._promote_to_degree` maps these through each cell's geometry to place the global
    P{k} / Q{k} mesh nodes, and the order matches the basis tabulated by :func:`lagrange_on` — the
    same builder, which is what keeps the node positions and the DOF numbering from drifting apart.

    ``cell_type`` names the cell; without it the dimension picks the simplex, which is what every
    caller predating tensor-product cells means. Note a tensor-product cell has interpolation points
    a simplex does not: a Q2 quadrilateral carries a cell-INTERIOR node (and a Q2 hexahedron a
    face-interior one too), where a P2 triangle has none.
    """
    from basix import CellType

    cell = basix_cell(cell_type)[0] if cell_type else (CellType.triangle if dim == 2 else CellType.tetrahedron)
    return np.asarray(_lagrange_basix(cell, degree).points)


def lagrange_interval(degree: int, quad_degree: Optional[int] = None) -> ElementSpec:
    """Lagrange P{degree} element on a reference interval, tabulated via basix.

    The 1-D sibling of :func:`lagrange_triangle` / :func:`lagrange_tet`, built by the same
    :func:`_lagrange_basix` builder so a P{k} interval and a P{k} triangle agree on what degree ``k``
    means. P1 has 2 DOFs (the endpoints); degree ``k`` adds ``k-1`` interior nodes, endpoints first.

    ``local_edges`` is the single "edge" of an interval — the cell itself — which keeps the
    ``ElementSpec`` contract uniform for consumers that read it.
    """
    import basix
    from basix import CellType

    if degree < 1:
        raise ValueError(f"lagrange_interval: degree must be >= 1; got {degree}.")
    qd = quad_degree if quad_degree is not None else 2 * degree + 1
    elem = _lagrange_basix(CellType.interval, degree)
    qp, qw = basix.make_quadrature(CellType.interval, qd)
    tab = elem.tabulate(2, qp)  # (n_blocks, n_quad, n_dof, 1) -- values, 1st and 2nd reference derivatives
    return ElementSpec(
        family=f"Lagrange-P{degree}-Interval",
        n_dof=int(elem.dim),
        value_size=1,
        quad_points=np.asarray(qp),
        quad_weights=np.asarray(qw),
        ref_values=np.asarray(tab[0]),  # (n_quad, n_dof, 1)
        ref_div=None,
        ref_grads=np.asarray(tab[1])[..., None],  # (n_quad, n_dof, 1, 1): the single ∂φ/∂ξ
        local_edges=((0, 1),),
        ref_hess=_ref_hessian_from_tab(tab, 1),  # (n_quad, n_dof, 1, 1, 1)
    )


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


#: meshio/VTK cell name -> (basix CellType name, topological dimension).
_BASIX_CELL = {
    "line": ("interval", 1),
    "interval": ("interval", 1),
    "triangle": ("triangle", 2),
    "tetra": ("tetrahedron", 3),
    "tetrahedron": ("tetrahedron", 3),
    "quad": ("quadrilateral", 2),
    "quadrilateral": ("quadrilateral", 2),
    "hexahedron": ("hexahedron", 3),
    "hex": ("hexahedron", 3),
}


def basix_cell(cell_type: str):
    """The basix ``CellType`` for a meshio cell name, and its topological dimension."""
    from basix import CellType

    entry = _BASIX_CELL.get(cell_type)
    if entry is None:
        raise NotImplementedError(
            f"cell type {cell_type!r} has no basix element in jNO (line / triangle / tetra / quad / hexahedron)."
        )
    name, tdim = entry
    return getattr(CellType, name), tdim


def lagrange_on(cell_type: str, degree: int, quad_degree: Optional[int] = None) -> ElementSpec:
    """Lagrange Q/P{degree} ``ElementSpec`` on any supported cell, tabulated via basix.

    The generic form of :func:`lagrange_interval` / :func:`lagrange_triangle` / :func:`lagrange_tet`
    -- those three differed only in the cell they named and how many derivative blocks they stacked,
    which is ``tdim``. Writing it once is what lets a quadrilateral or hexahedron join without a
    fourth and fifth near-copy, and it keeps one guarantee that matters: the mesh node generator
    (:func:`lagrange_interp_points`) and the basis tabulation go through the same builder, so their
    DOF order cannot drift apart.

    **Quadrature degree on a tensor-product cell.** The ``2*degree + 1`` default is chosen to be
    exact for an affine simplex, where ``detJ`` is constant. A bilinear quad or trilinear hex has a
    non-constant Jacobian, so ``1/detJ`` makes the integrand rational and no polynomial rule is
    exact -- the same situation curved cells are already in, and the caller raises the degree for
    them. The default is kept here so the two paths stay comparable, and the bump lives with the
    caller that knows the geometry.
    """
    import basix

    if degree < 1:
        raise ValueError(f"lagrange_on: degree must be >= 1; got {degree}.")
    cell, tdim = basix_cell(cell_type)
    qd = quad_degree if quad_degree is not None else 2 * degree + 1
    elem = _lagrange_basix(cell, degree)
    from .fem_utils import _QUAD_RULE

    if _QUAD_RULE[0] == "gll" and cell in (basix.CellType.quadrilateral, basix.CellType.hexahedron):
        # the GLL rule at the element's own nodes (degree >= 1: k + 1 points per axis, exact to 2k - 1): every
        # basis function is cardinal at the quadrature points, so the mass matrix is diagonal
        qp, qw = basix.make_quadrature(cell, max(2 * degree - 1, 1), rule=basix.QuadratureType.gll)
        # basix lists the GLL points end points first; in lexicographic order (as its Gauss rule already is)
        # the sum-factorised kernels need no permutation of the quadrature points, forward or transposed
        lex = np.lexsort(tuple(np.round(qp[:, a], 12) for a in range(qp.shape[1] - 1, -1, -1)))
        qp, qw = qp[lex], qw[lex]
    else:
        qp, qw = basix.make_quadrature(cell, qd)
    tab = elem.tabulate(2, qp)  # (n_blocks, n_quad, n_dof, 1): values, 1st and 2nd ref derivatives
    return ElementSpec(
        family=f"Lagrange-P{degree}-{basix.CellType(cell).name}",
        n_dof=int(elem.dim),
        value_size=1,
        quad_points=np.asarray(qp),
        quad_weights=np.asarray(qw),
        ref_values=np.asarray(tab[0]),  # (n_quad, n_dof, 1)
        ref_div=None,
        ref_grads=np.stack([np.asarray(tab[i]) for i in range(1, tdim + 1)], axis=-1),
        local_edges=tuple(tuple(e) for e in basix.topology(cell)[1]) if tdim > 1 else ((0, 1),),
        ref_hess=_ref_hessian_from_tab(tab, tdim),
    )


def _invert_tensor_map(V: np.ndarray, x: np.ndarray, cell: str, *, iters: int = 12):
    """Reference coordinates of ``x`` inside trilinear cells ``V``, by Newton on the isoparametric map.

    A simplex map is affine, so its inverse is one linear solve — which is why the branch opposite is
    exact and non-iterative. A quadrilateral or hexahedron maps ``x(xi) = sum_a N_a(xi) x_a`` with
    ``N`` bi/trilinear, so the inverse has no closed form and is found by Newton from the cell centre:
    ``xi <- xi - J^-1 (x(xi) - x)`` with ``J = sum_a x_a (dN_a/dxi)``.

    ``V`` is ``(..., n_local, dim)`` in **basix** vertex order and ``x`` is ``(..., dim)``. Returns
    ``(xi, N)`` — the reference coordinates ``(..., tdim)`` and the shape-function values there
    ``(..., n_local)``, so a caller gets both the coordinates it needs for a higher-order tabulation
    and the weights it needs for a Q1 stencil.

    Newton converges quadratically on a convex cell and is started at the centre, so 12 iterations is
    far more than needed for a mesh whose cells are not degenerate; it is a fixed count rather than a
    tolerance loop so the whole batch stays vectorised.
    """
    from .fem_lagrange import _lagrange_basix, basix_cell

    bcell, tdim = basix_cell(cell)
    elem = _lagrange_basix(bcell, 1)

    def tab(xi_flat):
        t = np.asarray(elem.tabulate(1, xi_flat))  # (1 + tdim, Q, n_local, 1)
        return t[0, :, :, 0], np.moveaxis(t[1 : tdim + 1, :, :, 0], 0, -1)  # N (Q,A), dN (Q,A,T)

    shp = x.shape[:-1]
    Vf = V.reshape(-1, V.shape[-2], V.shape[-1])
    xf = x.reshape(-1, x.shape[-1])
    xi = np.full((Vf.shape[0], tdim), 0.5)
    for _ in range(iters):
        N, dN = tab(xi)
        J = np.einsum("cai,cak->cik", Vf, dN)  # (C, dim, tdim)
        r = np.einsum("ca,cai->ci", N, Vf) - xf  # residual x(xi) - x
        det = np.linalg.det(J)
        Jsafe = np.where((np.abs(det) > 1e-300)[:, None, None], J, np.eye(tdim))
        xi = xi - np.linalg.solve(Jsafe, r[..., None])[..., 0]
        xi = np.clip(xi, -1.0, 2.0)  # keep a bad start from diverging out of the basis' useful range
    N, _ = tab(xi)
    return xi.reshape(*shp, tdim), N.reshape(*shp, Vf.shape[1])


def vtk_to_basix_vertex_perm(cell_type: str) -> np.ndarray:
    """Permutation taking a cell's meshio/VTK vertex order to basix's, as basix-index -> VTK-index.

    The two libraries number a reference cell differently, and only for the tensor-product cells:
    an interval, triangle and tetrahedron have the same vertex order in both, but VTK walks a
    quadrilateral's vertices **around its perimeter** (0,0), (1,0), (1,1), (0,1) while basix orders
    them **lexicographically** (0,0), (1,0), (0,1), (1,1) -- so VTK's nodes 2 and 3 are basix's 3
    and 2. On a hexahedron the same disagreement applies to both of its faces.

    Feeding a VTK-ordered cell to a basix-tabulated basis without this permutation does not fail
    loudly: it silently evaluates the basis with two corners swapped, which for a quad is a
    self-intersecting (bow-tie) map with a Jacobian that changes sign inside the cell.

    Derived from the reference geometries rather than written out, so it cannot drift if either
    library renumbers.
    """
    import basix

    cell, tdim = basix_cell(cell_type)
    ref = np.asarray(basix.geometry(cell))  # (n_vertices, tdim), basix order
    if cell_type in ("quad", "quadrilateral"):
        vtk = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elif cell_type in ("hexahedron", "hex"):
        base = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        vtk = np.array([b + [z] for z in (0.0, 1.0) for b in base])
    else:
        return np.arange(len(ref), dtype=np.int64)  # simplices already agree
    # basix DOF b sits at ref[b]; find which VTK slot holds that same corner.
    matches = np.all(np.isclose(vtk[None, :, :], ref[:, None, :], atol=1e-12), axis=-1)
    perm = np.argmax(matches, axis=1)
    if not matches.any(axis=1).all() or len(set(perm.tolist())) != len(ref):
        raise RuntimeError(f"could not match basix and VTK vertices for {cell_type!r}")
    return perm.astype(np.int64)


def identity_pushforward(
    ref_values: jnp.ndarray,
    ref_grads: jnp.ndarray,
    J: jnp.ndarray,
    detJ: jnp.ndarray,
    K: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Isoparametric push-forward of scalar Lagrange basis data to a physical cell.

    Scalar nodal values are coordinate-invariant (the push-forward is the identity
    on values); gradients transform via the chain rule:

        ``∂φ/∂x = ∂φ/∂ξ J⁻¹``

    Args:
        ref_values (``(n_quad, n_dof, 1)``  scalar shape values from basix tabulation.)
        ref_grads (``(n_quad, n_dof, 1, tdim)``  ``∂φ/∂ξ`` from basix (value_size=1).)
        J (``(tdim, tdim)`` affine cell Jacobian (columns = physical edge vectors), **or**): ``(n_quad,
            tdim, tdim)`` for **curved (isoparametric)** geometry, where the map is no longer affine and
            ``J`` varies over the cell.
        detJ (scalar or ``(n_quad,)``, unused (kept for API symmetry with the Piola push-forwards).)

    Returns:
        ``(n_quad, n_dof)``  physical shape values (``= ref_values[..., 0]``).: phi
        ``(n_quad, n_dof, tdim)``  physical gradients ``∂φ/∂x``.: dphi_phys
    """
    if K is None:  # a caller with a static mesh passes its cached J⁻¹ (see fem_native `_static_geometry`)
        K = small_inv(J)  # J⁻¹, (tdim, tdim) or (n_quad, tdim, tdim)
    phi = ref_values[..., 0]  # (n_quad, n_dof)
    dphi_ref = ref_grads[..., 0, :]  # (n_quad, n_dof, tdim)
    # One inverse per quadrature point when the geometry is curved; one for the whole cell when affine.
    # Written as multiply-and-sum, NOT an einsum: vmapped over cells, the einsum lowers to a batched GEMM of
    # (n_dof x tdim) @ (tdim x tdim) blocks with layout transposes around it -- a 4x3 @ 3x3 product per cell,
    # the worst shape a GEMM kernel can get. The broadcast form is an elementwise loop XLA fuses with its
    # neighbours.
    Kq = K[:, None, :, :] if K.ndim == 3 else K[None, None, :, :]  # (q|1, 1, d, D)
    dphi_phys = jnp.sum(dphi_ref[..., :, None] * Kq, axis=-2)  # (n_quad, n_dof, tdim)
    return phi, dphi_phys


def _refuse_curved_hessian(J: jnp.ndarray) -> None:
    """A curved cell has ``∂²ξ/∂x² ≠ 0``, which this push-forward's derivation drops."""
    if getattr(J, "ndim", 2) == 3:
        raise NotImplementedError(
            "A 4th-order weak form (Argyris / Morley / Hermite, phase-field, plates) needs the physical "
            "Hessian, whose transform assumes an AFFINE cell -- on curved geometry it gains a curvature "
            "term this does not carry, and the result would be wrong with nothing to flag it. Drop "
            "shape.curved() for a 4th-order form, or use a straight-sided domain."
        )


def identity_pushforward_hess(ref_hess: jnp.ndarray, J: jnp.ndarray) -> jnp.ndarray:
    """Physical Hessian of a scalar Lagrange basis on an **affine** simplex.

    The geometry is always P1 (straight-sided), so the reference→physical map ``ξ ↦ x`` is affine with a
    **constant** Jacobian ``J``; thus ``∂²ξ/∂x² ≡ 0`` and the second derivatives transform by the clean
    chain rule with no curvature term::

        ``∂²φ/∂x_a∂x_b = K_ia K_jb ∂²φ/∂ξ_i∂ξ_j``,   ``K = J⁻¹``.

    Args:
        ref_hess (``(n_quad, n_dof, 1, tdim, tdim)``  reference second derivatives ``∂²φ/∂ξ∂ξ``
            (value_size=1).)
        J (``(tdim, tdim)``  affine cell Jacobian.)

    Returns:
        ``(n_quad, n_dof, tdim, tdim)``  physical Hessian ``∂²φ/∂x∂x`` (symmetric).: hess_phys
        A P1 basis has an identically zero reference Hessian, and it is tabulated anyway. The
            push-forward
        of a zero table is returned as a zero CONSTANT, not an einsum: the einsum was the most expensive
        kernel of a 3-D stabilised Navier-Stokes residual (``lap(u)`` in the strong residual, ~9 of 26
            ms
        at 55k DOFs), and a constant lets XLA drop every contraction against it. The zero terms stay
        exactly zero; the sums around them may fuse in a different order (round-off, 1e-16 relative).
    """
    _refuse_curved_hessian(J)
    if _is_zero_table(ref_hess):
        n = jnp.shape(J)[-2]
        return jnp.zeros((*jnp.shape(ref_hess)[:2], n, n), jnp.result_type(ref_hess, J))
    K = small_inv(J)  # J⁻¹
    return jnp.einsum("qnij,ia,jb->qnab", ref_hess[..., 0, :, :], K, K)


def _is_zero_table(a) -> bool:
    """Whether ``a`` is a CONCRETE all-zero table (a tracer -- a table being differentiated -- is not)."""
    if isinstance(a, jax.core.Tracer):
        return False
    return not np.any(np.asarray(a))


# ---------------------------------------------------------------------------------------------------------
# Sum factorisation on tensor-product cells
# ---------------------------------------------------------------------------------------------------------


def _lagrange_1d(nodes: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Values and first derivatives ``(len(x), len(nodes))`` at ``x`` of the Lagrange basis on ``nodes``
    (barycentric form; a point on a node gives the exact cardinal row)."""
    from ..._spectral.bases import _barycentric_weights

    nodes, x = np.asarray(nodes, float), np.asarray(x, float)
    w = _barycentric_weights(nodes)
    diff = x[:, None] - nodes[None, :]
    hit = np.abs(diff) < 1e-13
    c = w[None, :] / np.where(hit, 1.0, diff)
    B = c / c.sum(axis=1, keepdims=True)
    on = hit.any(axis=1)
    B[on] = hit[on].astype(float)
    dn = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(dn, 1.0)
    Dn = (w[None, :] / w[:, None]) / dn
    np.fill_diagonal(Dn, 0.0)
    np.fill_diagonal(Dn, -Dn.sum(axis=1))  # the derivative of a constant is zero
    return B, B @ Dn


# The lowest degree evaluated by sum factorisation. Q1 and Q2 keep the dense tables (8 and 27 DOFs: the
# tables are as cheap, and their programs stay as they were).
SUMFACT_MIN_DEGREE = 3


class TensorFactors:
    """The 1-D factors of a Lagrange tabulation on a quadrilateral or hexahedron.

    The basis of ``Q_k`` at a tensor quadrature rule is a product of 1-D tables,
    ``φ_n(ξ_q) = Π_a B_a[q_a, n_a]`` and ``∂φ_n/∂ξ_a = D_a[q_a, n_a] Π_{b≠a} B_b[q_b, n_b]``. A field's values and
    gradients at the quadrature points are then one small matrix applied along each axis of the cell's DOF
    tensor: ``O(k^(d+1))`` work per cell where contracting the dense ``(n_quad, n_dof)`` tables costs
    ``O(k^(2d))`` (Orszag 1980). With the quadrature at the element's own nodes (GLL) ``B_a`` is the identity
    and is skipped.

    Everything is a linear map of ``cell_sol`` written in JAX, so forward mode gives the element tangent and
    reverse mode the transposed contractions that test a flux against the basis.

    Attributes:
        B, D: per axis, the ``(m_a, p_a)`` value and derivative tables.
        dof: ``(n_dof,)`` element DOF index of each lexicographic tensor DOF.
        quad: ``(n_quad,)`` lexicographic index of each quadrature point, ``None`` when already in that order.
    """

    __slots__ = ("B", "D", "dof", "quad", "p", "m", "_ident")

    def __init__(self, B, D, dof, quad):
        self.B, self.D = tuple(B), tuple(D)
        self.dof, self.quad = np.asarray(dof), quad
        self.p = tuple(int(b.shape[1]) for b in self.B)
        self.m = tuple(int(b.shape[0]) for b in self.B)
        self._ident = tuple(b.shape[0] == b.shape[1] and np.array_equal(b, np.eye(b.shape[0])) for b in self.B)

    @staticmethod
    def _along(M, T, axis):
        # Full float32 precision, stated: on an Ampere-or-later GPU a float32 ``dot_general`` otherwise runs
        # in TensorFloat-32 (a 10-bit mantissa). A stiffness apply differences neighbouring values, so that
        # loss is amplified with the mesh: measured on an RTX 3070, ``A x`` in float32 was off by 5.5e-4 at
        # 227k DOFs and 5.6e-2 at 2M (CG then stalled at a relative residual of 1.1), against 2.2e-6 here.
        M = jnp.asarray(M, _real_of(T))
        return jnp.moveaxis(jnp.tensordot(M, T, axes=(1, axis), precision=jax.lax.Precision.HIGHEST), 0, axis)

    def _lex(self, cell_sol):
        # a permutation: said so, its transpose is a plain scatter, not a scatter-add
        lex = cell_sol.at[self.dof].get(unique_indices=True, mode="promise_in_bounds")
        return lex.reshape(self.p + cell_sol.shape[1:])

    def _flat(self, T):
        out = T.reshape((-1,) + T.shape[len(self.m) :])
        return out if self.quad is None else out[self.quad]

    def values(self, cell_sol):
        """``(n_quad, vec)``: the field at the quadrature points from its cell DOFs ``(n_dof, vec)``."""
        T = self._lex(cell_sol)
        for a, B in enumerate(self.B):
            if not self._ident[a]:
                T = self._along(B, T, a)
        return self._flat(T)

    def ref_grads(self, cell_sol):
        """``(n_quad, vec, tdim)``: the reference gradient ``∂u_c/∂ξ_a`` at the quadrature points."""
        T = self._lex(cell_sol)
        out = []
        for a in range(len(self.B)):
            G = self._along(self.D[a], T, a)
            for b, B in enumerate(self.B):
                if b != a and not self._ident[b]:
                    G = self._along(B, G, b)
            out.append(self._flat(G))
        return jnp.stack(out, axis=-1)

    def grads(self, cell_sol, K):
        """``(n_quad, vec, dim)``: the physical gradient, ``∂u/∂x = ∂u/∂ξ J⁻¹`` with ``K = J⁻¹`` per cell
        ``(tdim, dim)`` or per quadrature point ``(n_quad, tdim, dim)``."""
        g = self.ref_grads(cell_sol)
        return jnp.sum(g[..., :, None] * self.per_point(K), axis=-2)

    @staticmethod
    def per_point(K):
        """``K`` broadcast against ``(n_quad, vec, tdim, dim)``. A caller with several fields on one cell does
        this once and hands the result to every :meth:`grads`: one array, kept once by a linearisation."""
        return K if K.ndim == 4 else (K[:, None, :, :] if K.ndim == 3 else K[None, None, :, :])


def _real_of(T):
    dt = jnp.result_type(T, jnp.float32)
    return jnp.finfo(dt).dtype


def tensor_factors(cell_type: str, degree: int, quad_points, ref_values, ref_grads, *, tol: float = 1e-9):
    """The :class:`TensorFactors` of a Lagrange ``Q_degree`` tabulation on ``cell_type``, or ``None``.

    The factors are rebuilt from the element's node coordinates and the quadrature points, then checked against
    the tabulation they stand in for (values and reference gradients, to ``tol``): a cell, node set or rule
    that is not a full tensor product returns ``None`` and the dense tables stay in use.
    """
    cell, tdim = basix_cell(cell_type)
    import basix

    if cell not in (basix.CellType.quadrilateral, basix.CellType.hexahedron):
        return None
    X = np.asarray(_lagrange_basix(cell, int(degree)).points, float)
    Q = np.asarray(quad_points, float)
    V = np.asarray(ref_values, float)[..., 0]
    dV = np.asarray(ref_grads, float)[..., 0, :]
    if V.shape != (Q.shape[0], X.shape[0]):
        return None
    B, D, ni, qi = [], [], [], []
    for a in range(tdim):
        _xn, ia = np.unique(np.round(X[:, a], 11), return_inverse=True)
        _xq, ja = np.unique(np.round(Q[:, a], 11), return_inverse=True)
        xn = np.array([X[ia == j, a].mean() for j in range(_xn.size)])
        xq = np.array([Q[ja == j, a].mean() for j in range(_xq.size)])
        b, d = _lagrange_1d(xn, xq)
        B.append(b), D.append(d), ni.append(ia), qi.append(ja)
    p, m = tuple(b.shape[1] for b in B), tuple(b.shape[0] for b in B)
    if int(np.prod(p)) != X.shape[0] or int(np.prod(m)) != Q.shape[0]:
        return None
    lex_n, lex_q = np.ravel_multi_index(ni, p), np.ravel_multi_index(qi, m)
    if np.unique(lex_n).size != lex_n.size or np.unique(lex_q).size != lex_q.size:
        return None
    # the tabulation the factors must reproduce
    parts = [B[a][qi[a]][:, ni[a]] for a in range(tdim)]  # (n_quad, n_dof) each
    if np.abs(np.prod(parts, axis=0) - V).max() > tol:
        return None
    for a in range(tdim):
        g = np.prod([D[b][qi[b]][:, ni[b]] if b == a else parts[b] for b in range(tdim)], axis=0)
        if np.abs(g - dV[..., a]).max() > tol * max(1.0, np.abs(dV).max()):
            return None
    quad = None if np.array_equal(lex_q, np.arange(lex_q.size)) else lex_q
    return TensorFactors(B, D, np.argsort(lex_n), quad)
