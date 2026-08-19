"""Native Lagrange assembler for ``jno.fem``.

Implements the full assembly pipeline for scalar/vector Lagrange P1/P2 fields on 2D
triangle and 3D tetrahedral meshes (single- and multi-field, linear/nonlinear/transient),
mirroring the contract of :func:`fem_1d.assemble_fem_1d` and
:func:`fem_nonnodal.assemble_fem_nonnodal`. The assembler is dimension-generic: the cell
Jacobian, element factory and facet machinery all key off ``dim``.

Key components re-used without change:

* :func:`fem_utils._eval_integrand` — the DSL integrand evaluator.
* :func:`fem_1d._integrate_term` — weighted sum over quad points.
* :func:`fem_1d._apply_dirichlet_*` — Dirichlet enforcement (symmetric/row/transient).
* :func:`fem_utils._promote_to_quadratic` — P1→P2 mesh promotion.
* :func:`fem_utils._cell_region_mask` — per-cell sub-region indicator.

New components (this module only):

* :func:`fem_lagrange.lagrange_triangle` / :func:`fem_lagrange.lagrange_tet` /
  :func:`fem_lagrange.identity_pushforward` — basix-backed Lagrange reference tabulation +
  isoparametric gradient map.
* :func:`fem_facets.build_facet_connectivity` / :func:`fem_facets.compute_face_normals`
  — boundary face connectivity + outward normals for surface integration.

References
----------
Matrix extraction via ``jax.jacfwd(residual)(zeros)`` follows Griewank & Walther,
*Evaluating Derivatives*, SIAM (2008), §3.5 — the same pattern as :mod:`fem_1d`
and :mod:`fem_nonnodal`.

Scope
-----
Lagrange P1/P2 fields on 2D triangle and 3D tetrahedral meshes (single- and multi-field,
linear, nonlinear, and transient), with Dirichlet and Neumann/Robin boundary conditions
(2D edge / 3D tet-face surface quadrature).  Niches outside this scope raise a clear
``NotImplementedError`` from ``jno.fem`` rather than assembling silently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

from .fem_1d import (
    _apply_dirichlet_projected,
    _apply_dirichlet_symmetric,
    _apply_dirichlet_transient,
    _integrate_term,
    _line_quadrature,
    _region_node_ids,
    dirichlet_projection,
)
from .fem_cover import COVER_ORDER as COVER_DEGREE
from .fem_cover import cover_block, cover_null_modes, expand_cover, nodal_scale
from .fem_facets import _LOCAL_FACES_TET, _face_table, build_facet_connectivity, compute_face_normals
from .fem_lagrange import (
    _lagrange_basix,
    identity_pushforward,
    identity_pushforward_hess,
    lagrange_interp_points,
    lagrange_interval,
    lagrange_tet,
    lagrange_triangle,
)
from .fem_utils import (
    _CHUNK_CONSUMED,
    _CHUNK_OVERRIDE,
    _cell_region_mask,
    _collect_region_mask_names,
    _collect_tag_mask_names,
    _eval_integrand,
    _gather_temporal_tags,
    _infer_fields,
    _lower_statefield_to_trial,
    _promote_to_degree,
    _test_field_index,
    apply_compress_plan,
    bcoo_eliminate_dirichlet,
    bcoo_set_dirichlet_rows,
    bcoo_zero_rows,
    bcoo_zero_rows_cols,
    cell_chunk,
    compress_eager,
    compress_plan,
    elem_map,
)
from .parametric_helpers import _collect_runtime_parameter_exprs
from .weak_form import (
    _apply_sign,
    _contains_temporal_derivative,
    _is_obviously_nonlinear_in_unknown,
    _split_additive_terms,
)

# Reference simplex vertex coordinates (basix convention).
_REF_TRI_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # v0=(0,0), v1=(1,0), v2=(0,1)
_REF_TET_VERTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# Local face ordering for a triangle: entry k = (local_node_a, local_node_b, opp_node).
_LOCAL_FACES_TRI = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def _gmsh_to_basix_perm(pts: np.ndarray, cells_c: np.ndarray, dim: int, order: int) -> np.ndarray:
    """Column permutation taking gmsh's higher-order node order to basix's DOF order.

    The two disagree, and silently. For a P2 triangle the reference nodes come out::

        basix:  (0,0) (1,0) (0,1) | (0.5,0.5) (0,0.5)   (0.5,0)
        gmsh:   (0,0) (1,0) (0,1) | (0.5,0)   (0.5,0.5) (0,0.5)

    so using gmsh's cells directly pairs each geometry node with the wrong shape function. Nothing
    errors -- the Jacobian is simply built from a scrambled map, and the only symptom is a wrecked
    convergence rate (measured: L2 got *worse* than straight-sided, converging at 1.8x per halving
    instead of 8x).

    Derived numerically rather than hard-coded from two conventions: each node's reference coordinate
    is recovered through the cell's own affine vertex map and matched to
    :func:`lagrange_interp_points`. A curved cell displaces its midside nodes by O(h²), far below the
    O(1) spacing of the reference points, so the nearest match stays unambiguous. Several cells are
    checked and must agree, which is what turns a silent mis-order into a loud failure.
    """
    ref = np.asarray(lagrange_interp_points(dim, order), dtype=float)  # (n_loc, dim), basix DOF order
    n_loc = ref.shape[0]
    if cells_c.shape[1] != n_loc:
        raise ValueError(f"curved cells have {cells_c.shape[1]} nodes but the P{order} element wants {n_loc}.")

    perm = None
    for cell in cells_c[: min(8, len(cells_c))]:
        v = pts[cell[: dim + 1]]
        jac = np.column_stack([v[i + 1] - v[0] for i in range(dim)])
        local = np.linalg.solve(jac, (pts[cell] - v[0]).T).T  # (n_loc, dim) reference coords
        near = np.argmin(((ref[:, None, :] - local[None, :, :]) ** 2).sum(-1), axis=1)  # basix -> gmsh
        if sorted(near.tolist()) != list(range(n_loc)):
            raise ValueError("curved cell nodes do not match the reference element one-to-one.")
        if perm is None:
            perm = near
        elif not np.array_equal(perm, near):
            raise ValueError("gmsh's higher-order node order is not consistent across cells.")
    return np.asarray(perm)


def _is_nonconforming_side(tag: str) -> bool:
    """Is ``tag`` one *side* of a non-conforming interface (``"a|b.a"``), rather than a conforming
    ``"a|b"`` or one of its disjoint components ``"a|b.0"``? The suffix must be one of the pair's own
    region names, which is what distinguishes a duplicated surface from a shared one."""
    head, dot, tail = str(tag).rpartition(".")
    return bool(dot) and "|" in head and tail in head.split("|")


#: meshio volume-block name per dimension, simplex first. A mesh has exactly one of these.
_VOLUME_BLOCKS = {1: ("line",), 2: ("triangle", "quad"), 3: ("tetra", "hexahedron")}

#: The cells whose reference map is NOT affine, so the Jacobian must be formed per quadrature point.
TENSOR_PRODUCT_CELLS = ("quad", "hexahedron")


def mesh_cell_type(domain, dim: int) -> str:
    """The meshio name of ``domain``'s volume cell: ``"triangle"``/``"quad"``, ``"tetra"``/``"hexahedron"``.

    The assembler used to derive this from the dimension alone, which is exactly the assumption a
    tensor-product mesh breaks. Curved blocks report their first-order base name, as everywhere else.
    """
    cd = domain.mesh.cells_dict
    for name in _VOLUME_BLOCKS.get(int(dim), ()):
        if name in cd:
            return name
    curved = {1: "line3", 2: "triangle6"}.get(int(dim), "tetra10")
    if curved in cd:
        return _VOLUME_BLOCKS[int(dim)][0]
    raise NotImplementedError(
        f"no supported volume cell block for a {dim}-D mesh; expected one of {_VOLUME_BLOCKS.get(int(dim), ())}, "
        f"found {sorted(cd)}."
    )


def _basix_ordered(cells: np.ndarray, cell_type: str) -> np.ndarray:
    """``cells`` with each cell's vertices reordered from meshio/VTK order into basix's.

    A no-op for simplices (the two libraries agree); a real permutation for a quad or hex, where
    skipping it silently evaluates the basis on a bow-tie. Topology (facets, region masks, normals)
    keeps the mesh's own order -- only the arrays that meet a tabulated BASIS are permuted.
    """
    from .fem_lagrange import vtk_to_basix_vertex_perm

    perm = vtk_to_basix_vertex_perm(cell_type)
    return cells if np.array_equal(perm, np.arange(len(perm))) else np.asarray(cells)[:, perm]


def _refuse_nonconforming_promotion(domain, order: int) -> None:
    """Refuse order > 1 on a ``Shape.regions(..., conforming=False)`` domain.

    `_promote_to_degree` dedups synthesised nodes by physical COORDINATE, which is the right
    conformity test for one body and the wrong one for two: a non-conforming interface is coincident
    *on purpose*, so every higher-order node the promotion adds there is merged across the two bodies
    and welds them. Measured on a two-body bar: 37 nodes referenced by BOTH bodies, all at the
    interface. Harmless for a tie (it wanted continuity anyway) and wrong for contact, where those
    DOFs can then never separate -- silent either way. See plans/p2-promotion-entity-keys.md for the
    fix (key on the topological entity instead); refused until then. Shared by the simplex and
    tensor-product promotion paths, since the dedup they run is the same one.
    """
    if any(_is_nonconforming_side(t) for t in (getattr(domain, "_interface_registry", {}) or {})):
        raise NotImplementedError(
            f"order-{order} elements on a Shape.regions(..., conforming=False) domain: the higher-order "
            "node promotion deduplicates by coordinate, so it would silently WELD the two bodies at "
            "every interface node it adds -- which a contact gap could then never open. Use order-1 "
            "elements for a non-conforming interface, or Shape.curved() (which reads gmsh's nodes "
            "instead of synthesising them, and is unaffected)."
        )


def _get_mesh(domain, dim: int, order: int):
    """P1 base mesh + optionally promoted P{order} mesh, both as NumPy arrays.

    Returns ``(pts_p1, cells_p1, pts_f, cells_f)`` where:

    * ``pts_p1, cells_p1`` — the original P1 mesh (used for region masks and facets).
    * ``pts_f, cells_f`` — same as P1 when ``order=1``; promoted P2 when ``order=2``.

    ``cells_p1`` stays in the mesh's own (meshio/VTK) vertex order because the topology machinery
    is written against it; ``cells_f`` is what the tabulated basis sees and is therefore in basix
    order. The two differ only for quadrilaterals and hexahedra.
    """
    cell_type = mesh_cell_type(domain, dim)
    if cell_type in TENSOR_PRODUCT_CELLS:
        cells_p1 = np.asarray(domain.mesh.cells_dict[cell_type], dtype=np.int64)
        pts_all = np.asarray(domain.mesh.points)[:, :dim]
        cells_b = _basix_ordered(cells_p1, cell_type)  # the order a tabulated basis is written in
        if order == 1:
            return pts_all, cells_p1, pts_all, cells_b
        _refuse_nonconforming_promotion(domain, order)
        pts_f, cells_f = _promote_to_degree(pts_all, cells_b, lagrange_interp_points(dim, order, cell_type), cell_type)
        # Both arrays share one id space, as on the curved path: `_promote_to_degree` keeps the
        # original vertices at ids 0..nv-1, so the P1 connectivity still indexes `pts_f` correctly.
        # That matters because the geometry gather reads the ASSEMBLY connectivity against `pts_p1`
        # on a non-affine cell -- returning the vertex-only array here would index a Q{k}-wide
        # connectivity into it.
        return pts_f, cells_p1, pts_f, cells_f
    # meshio names the simplex cell block "triangle" (2D) / "tetra" (3D) -- distinct from the basix
    # CellType name "tetrahedron" the facet machinery uses.
    meshio_key = {1: "line", 2: "triangle"}.get(dim, "tetra")
    curved_key = {1: "line3", 2: "triangle6"}.get(dim, "tetra10")
    cd = domain.mesh.cells_dict
    pts_all = np.asarray(domain.mesh.points)[:, :dim]

    if curved_key in cd:
        # CURVED (isoparametric) mesh from `Shape.curved()`: gmsh already placed the higher-order nodes
        # on the CAD surface, so they must be USED, never re-synthesised. The first dim+1 columns of a
        # curved cell are its vertices, giving the P1 sub-mesh the facet/region machinery wants -- kept
        # in the SAME id space as the curved mesh so a node id means one thing everywhere (the promoted
        # path below builds a second array precisely because it has no such nodes to point at).
        cells_c = np.asarray(cd[curved_key], dtype=np.int64)
        cells_v = cells_c[:, : dim + 1]
        if order != 2:
            # Isoparametric means geometry order == basis order. A curved mesh under a P1 basis puts
            # the midside DOF coordinates (on the arc) and the geometric map (from the chord) in
            # disagreement -- an inconsistent discretisation, not merely a coarse one.
            raise ValueError(
                f"Shape.curved() gives order-2 geometry but this field is P{order}. Isoparametric "
                f"geometry needs a matching basis: pass element_type='{'TRI6' if dim == 2 else 'TET10'}' "
                "(or order=2) to jno.fem, or drop .curved() to mesh straight-sided."
            )
        return pts_all, cells_v, pts_all, cells_c[:, _gmsh_to_basix_perm(pts_all, cells_c, dim, order)]

    pts_p1 = pts_all
    cells_p1 = np.asarray(cd[meshio_key], dtype=np.int64)
    if order == 1:
        return pts_p1, cells_p1, pts_p1, cells_p1
    # `_promote_to_degree` dedups synthesised nodes by physical COORDINATE, which is the right
    # conformity test for one body and the wrong one for two: a `conforming=False` interface is
    # coincident *on purpose*, so every P2+ node the promotion adds there is merged across the two
    # bodies and welds them. Measured on a two-body bar: 37 nodes referenced by BOTH bodies, all at the
    # interface. Harmless for a tie (it wanted continuity anyway) but wrong for contact, where those
    # DOFs can then never separate -- and silent either way. See plans/p2-promotion-entity-keys.md for
    # the fix (key on the topological entity instead); refused until then.
    _refuse_nonconforming_promotion(domain, order)
    if dim not in (1, 2, 3):
        raise NotImplementedError(f"Dimension {dim} not supported by native assembler.")
    # P{order} node mesh: place the element's reference interpolation points (basix DOF order) on each
    # cell and dedup by coordinate. One code path for P2 and P3+ (the P2 midpoints are the k=2 case).
    pts_f, cells_f = _promote_to_degree(pts_p1, cells_p1, lagrange_interp_points(dim, order))
    return pts_p1, cells_p1, pts_f, cells_f


def _lagrange_simplex(dim: int, degree: int, quad_degree: Any = None, cell_type: Any = None):
    """The Lagrange ``ElementSpec`` for a cell: interval / triangle / tetrahedron / quad / hexahedron.

    One dispatcher, because the assembler and the VPINN context builder both need it and a second
    per-site conditional is how a dimension gets silently left out. ``cell_type`` names the cell
    directly; without it the dimension picks the simplex, which is what every existing caller means.
    """
    if cell_type is not None:
        from .fem_lagrange import lagrange_on

        return lagrange_on(cell_type, degree, quad_degree)
    builder = {1: lagrange_interval, 2: lagrange_triangle}.get(int(dim), lagrange_tet)
    return builder(degree, quad_degree)


def _real_dirichlet_values(gs: Any, region: str) -> np.ndarray:
    """Essential values as float64, refusing a genuinely complex one instead of dropping its imaginary part.

    A complex weak form assembles as two real legs sharing one Dirichlet row set, and these values feed
    both. That is well posed for a **real** ``g``: the fused block imposes ``x_r - x_i = g`` and
    ``x_r + x_i = g``, i.e. ``Re u = g`` with ``Im u = 0``. For a complex ``g`` it is not expressible —
    pinning ``Im u = g_i`` needs the imaginary leg's Dirichlet rows zeroed rather than set to identity,
    and the symmetric elimination's *column* lift is cross-leg (the real equation's known-column term is
    ``A_r[:,j] g_r - A_i[:,j] g_i``, which no per-leg elimination can produce).

    These call sites used to cast with ``float(...)`` / ``.astype(float)``, so ``Im(g)`` vanished behind
    a numpy ``ComplexWarning`` and the solve returned a plausible, wrong field: measured 8.9e-1 relative
    error on ``u = (1+2j)x`` over a unit square, with no error raised. Refuse instead.
    """
    arr = np.asarray(gs)
    if np.iscomplexobj(arr) and np.any(np.abs(arr.imag) > 0.0):
        raise NotImplementedError(
            f"jno.fem: a COMPLEX essential value on region {region!r} is not supported — a complex form's "
            "Re/Im legs share one Dirichlet row set, which can impose Re u = g with Im u = 0 but not a "
            "prescribed Im u. Use a real essential value and carry the complex part in the operator or "
            "the source."
        )
    # Returned UNCHANGED (not cast): each call site keeps its own conversion, so this guard adds a check
    # and changes nothing else — a real value still takes exactly the path it always did.
    return gs


def _region_node_ids_from_pts(domain, region: str, pts_all: np.ndarray) -> List[int]:
    """Node ids in ``pts_all`` for ``region`` — a **geometric interior sub-region**
    (``domain.region(name, polygon)``) by point-in-polygon, else the region's location function.

    A shapely polygon is not jax-traceable, so an interior sub-region cannot go through the jax
    ``_make_tag_location_fn`` path below; it is resolved here in numpy. This is what lets a subdomain
    solve (domain decomposition) restrict/pin on a named sub-region."""
    ptags = getattr(domain, "_polygon_tags", {})
    if region in ptags and ptags[region][0] == "interior":
        pts = np.asarray(pts_all)
        try:
            from shapely import contains_xy  # vectorized (shapely >= 2.0.2)

            hits = np.asarray(contains_xy(ptags[region][1].buffer(1e-9), pts[:, 0], pts[:, 1]))
        except (ImportError, AttributeError):
            from shapely.geometry import Point

            g = ptags[region][1].buffer(1e-9)
            hits = np.array([g.contains(Point(float(q[0]), float(q[1]))) for q in pts])
        return list(np.where(hits.reshape(-1))[0])

    # Via `tag_node_mask`, which evaluates a `domain.tag` predicate in numpy float64. Reading the
    # coordinates into a JAX array here instead truncated them to float32 (x64 off), so a tag
    # tolerance finer than float32 eps selected nothing -- the same defect the 1D assembler had.
    # The boundary path (`_boundary_node_ids`) already applied `_tag_predicates` in numpy; this is
    # the fallback path (no facets, or an interior pin), which did not.
    mask = domain.tag_node_mask(region, np.asarray(pts_all))
    if mask is None:
        raise ValueError(f"jno.fem (native): region {region!r} has no location function.")
    return list(np.where(mask)[0])


# ---------------------------------------------------------------------------
# Face (edge) pre-tabulation for surface integration
# ---------------------------------------------------------------------------


#: A quadrilateral facet's nodes come out of the topology tables in PERIMETER-CYCLIC order
#: (a, b, c, d around the face), while basix numbers a quadrilateral's vertices lexicographically
#: ((0,0), (1,0), (0,1), (1,1)). This reindexes the former into the latter. Pairing cyclic corners
#: with a basix quad basis without it yields a BOW-TIE whose Jacobian changes sign inside the face.
_CYCLIC_TO_BASIX_QUAD = [0, 1, 3, 2]


def _refuse_tensor_product(cell_type: Any, what: str, because: str) -> None:
    """Refuse a still-simplex-only path on a quad/hex mesh by NAME, not by shape error.

    These paths compute cell or facet geometry with simplex formulae (one Jacobian per cell from
    ``verts[i+1] - verts[0]``, one normal and area element per facet). Handed a tensor-product cell
    they do not produce a wrong answer -- they produce a bare broadcasting error between two basis
    sizes, which says nothing about what is missing. Returns ``False`` when it does not refuse, so
    it can be used inline in a conditional.
    """
    if cell_type in TENSOR_PRODUCT_CELLS:
        raise NotImplementedError(
            f"{what} on a {cell_type} mesh is not supported yet: {because} Volume terms and Dirichlet "
            "conditions do work on quad/hex meshes -- use those, or a simplicial mesh for this problem."
        )
    return False


def _refuse_tensor_product_surface(cell_type: Any) -> None:
    """Surface integrals: supported on quadrilaterals, still refused on hexahedra.

    The split is geometric, not incidental. Restricted to one edge a bilinear map is LINEAR, so a
    quadrilateral's facet is a straight segment with a constant tangent and a single normal -- the
    facet machinery needs only the right basis and a Jacobian formed from the geometry basis. A
    hexahedron's facet is a bilinear SURFACE: its normal and area element vary across the facet and
    need Nanson's formula per quadrature point, and the frozen one-normal-per-facet orientation the
    assembler carries has no per-point analogue yet.
    """
    return False


def _build_face_tables(elem_degree: int, quad_degree: int, dim: int = 2, cell_type: str = ""):
    """Pre-tabulate the parent-cell Lagrange basis at the quad points of each local facet.

    Dimension-generic: a 2D triangle's facets are its 3 edges (1-D Gauss quadrature); a 3D tet's
    facets are its 4 triangular faces (2-D triangle quadrature). The facet ordering matches
    ``build_facet_connectivity`` (``_LOCAL_FACES_TRI`` / ``_LOCAL_FACES_TET``) so a connectivity
    ``local_face`` index ``k`` selects the right table.

    Returns ``(face_phi, face_dphi_ref, face_ref_qp, face_ref_tangs, face_w)``:

    * ``face_phi``       ``(n_faces, n_q, n_dof)``       parent basis values at facet qp.
    * ``face_dphi_ref``  ``(n_faces, n_q, n_dof, dim)``  reference-domain gradients.
    * ``face_ref_qp``    ``(n_faces, n_q, dim)``         parent-reference coords of facet qp.
    * ``face_ref_tangs`` ``(n_faces, dim-1, dim)``       the ``dim-1`` reference tangent vectors that
      span each facet (one edge tangent in 2D; two face tangents in 3D). The physical area element is
      ``|J·t|`` (2D edge length) or ``|（J·t0) × (J·t1)|`` (3D face area), formed in ``_surf_elem_res``.
    * ``face_w``         ``(n_q,)``                      reference-facet quadrature weights (1-D Gauss
      on [0, 1] summing to 1 in 2D; triangle weights summing to 1/2 in 3D).
    """
    import basix
    from basix import CellType

    from .fem_facets import local_faces_in_basix_order

    if cell_type in ("hexahedron", "hex"):
        # A hexahedron's facet is a QUADRILATERAL, so the facet rule is a quad rule and the facet is
        # parameterised bilinearly. On the REFERENCE hex every face is a unit square (verified for
        # all six), so its two tangents are constant -- all the variation that makes a hex face a
        # curved surface enters later, through the per-quadrature-point cell Jacobian.
        #
        # `local_faces_in_basix_order` lists a face's nodes in PERIMETER-CYCLIC order, while a basix
        # quadrilateral numbers its vertices lexicographically. `_CYCLIC_TO_BASIX_QUAD` reconciles
        # the two; pairing them without it makes the face a bow-tie whose normal flips inside it.
        cell = CellType.hexahedron
        ref_verts = np.asarray(basix.geometry(cell))
        local_faces, _nfn = local_faces_in_basix_order(cell_type)
        qp_quad, face_w = (np.asarray(x) for x in basix.make_quadrature(CellType.quadrilateral, quad_degree))
        _fq = _lagrange_basix(CellType.quadrilateral, 1)
        _tabq = np.asarray(_fq.tabulate(1, qp_quad))  # (1+2, n_q, 4, 1)

        def _facet_qp_tangs(nodes):
            V = ref_verts[list(np.asarray(nodes)[_CYCLIC_TO_BASIX_QUAD])]  # (4, 3) basix quad order
            ref_qp = _tabq[0, :, :, 0] @ V  # (n_q, 3): x(s,t) = sum_a N_a(s,t) v_a
            return ref_qp, np.stack([V[1] - V[0], V[2] - V[0]])  # constant on a reference face
    elif cell_type in ("quad", "quadrilateral"):
        # A quadrilateral's facet is a straight 2-node edge, exactly as a triangle's is: restricted
        # to one edge the bilinear map is LINEAR in the edge parameter, so the edge stays straight
        # and its tangent is constant. Only the reference cell, its vertices and the basis change.
        cell = CellType.quadrilateral
        ref_verts = np.asarray(basix.geometry(cell))
        local_faces, _nfn = local_faces_in_basix_order(cell_type)
        gp_1d, face_w = (np.asarray(x) for x in _line_quadrature(quad_degree))

        def _facet_qp_tangs(nodes):
            va, vb = ref_verts[nodes[0]], ref_verts[nodes[1]]
            ref_qp = va[None, :] * (1.0 - gp_1d[:, None]) + vb[None, :] * gp_1d[:, None]
            return ref_qp, np.stack([vb - va])
    elif dim == 2:
        cell, ref_verts, local_faces = CellType.triangle, _REF_TRI_VERTS, _LOCAL_FACES_TRI
        gp_1d, face_w = (np.asarray(x) for x in _line_quadrature(quad_degree))

        def _facet_qp_tangs(nodes):  # an edge between two vertices
            va, vb = ref_verts[nodes[0]], ref_verts[nodes[1]]
            ref_qp = va[None, :] * (1.0 - gp_1d[:, None]) + vb[None, :] * gp_1d[:, None]  # (n_q, 2)
            return ref_qp, np.stack([vb - va])  # tangs (1, 2)
    else:
        cell, ref_verts, local_faces = CellType.tetrahedron, _REF_TET_VERTS, _LOCAL_FACES_TET
        qp_tri, face_w = (np.asarray(x) for x in basix.make_quadrature(CellType.triangle, quad_degree))

        def _facet_qp_tangs(nodes):  # a triangular face spanned by three vertices
            va, vb, vc = ref_verts[nodes[0]], ref_verts[nodes[1]], ref_verts[nodes[2]]
            xi, eta = qp_tri[:, 0], qp_tri[:, 1]
            ref_qp = va[None] * (1 - xi - eta)[:, None] + vb[None] * xi[:, None] + vc[None] * eta[:, None]
            return ref_qp, np.stack([vb - va, vc - va])  # tangs (2, 3)

    elem = _lagrange_basix(cell, elem_degree)
    phi_list, dphi_list, qp_list, tang_list = [], [], [], []
    # A facet's vertex count is a property of the CELL, not the dimension: `dim` is right for a
    # simplex facet and for a quad edge, and takes three of a hexahedron's four face nodes.
    _n_fv = _face_table(cell_type)[1] if cell_type else dim
    for entry in local_faces:
        ref_qp, tangs = _facet_qp_tangs(entry[:_n_fv])  # the facet's vertex local ids
        tab = elem.tabulate(1, ref_qp)  # (1 + dim, n_q, n_dof, 1)
        phi_list.append(tab[0, :, :, 0])  # (n_q, n_dof)
        dphi_list.append(np.stack([tab[1 + d, :, :, 0] for d in range(dim)], axis=-1))  # (n_q, n_dof, dim)
        qp_list.append(ref_qp)
        tang_list.append(tangs)

    return (
        jnp.asarray(np.stack(phi_list)),  # (n_faces, n_q, n_dof)
        jnp.asarray(np.stack(dphi_list)),  # (n_faces, n_q, n_dof, dim)
        jnp.asarray(np.stack(qp_list)),  # (n_faces, n_q, dim)
        jnp.asarray(np.stack(tang_list)),  # (n_faces, dim-1, dim)
        jnp.asarray(face_w),  # (n_q,)
    )


def _GSPEC(K):
    """einsum spec pushing reference gradients forward by ``K = J⁻¹``, per cell or per quad point.

    The surface twin of :func:`fem_lagrange.identity_pushforward`'s rank dispatch: a simplex facet
    has one ``K`` for the whole cell, a tensor-product facet one per quadrature point.
    """
    return "qnd,qdD->qnD" if getattr(K, "ndim", 2) == 3 else "qnd,dD->qnD"


def _facet_nanson_normal(J, tangs, sign):
    """Unit outward normal at each facet quadrature point, by Nanson's formula.

    The physical tangents ``T_i = J · t_i`` span the facet's tangent plane *at that point*, so their
    cross product is the (unnormalised) normal there — which is exactly the quantity
    :func:`_facet_area_element` already forms to take its magnitude. On a straight facet it is
    constant and this reproduces the frozen per-facet normal; on a hexahedron's **bilinear** facet it
    genuinely varies, which is the whole reason this exists.

    ``sign`` is the frozen ±1 outward orientation of the facet. It stays a per-facet constant even
    when the direction does not: the sign is a discrete choice tied to the facet's vertex ordering,
    and a non-inverted facet's normal cannot swing into the opposite hemisphere within the facet.

    Returns ``(n_q, dim)``. Nanson, *Messenger of Mathematics* 7 (1878) 182.
    """
    T = jnp.einsum("td,qDd->qtD", tangs, J) if J.ndim == 3 else (tangs @ J.T)[None, ...]
    n = jnp.cross(T[:, 0], T[:, 1]) if T.shape[1] == 2 else jnp.stack([T[:, 0, 1], -T[:, 0, 0]], axis=-1)
    return sign * n / (jnp.linalg.norm(n, axis=-1, keepdims=True) + 1e-300)


def _facet_area_element(J, tangs):
    """Physical facet measure element from the reference tangents ``tangs`` (``dim-1, dim``) pushed
    forward by the cell Jacobian ``J`` (``dim, dim``): the edge length ``|J·t|`` in 2D, the face area
    ``|(J·t0) × (J·t1)|`` in 3D. Multiplying it by the reference-facet weights gives ``dS``."""
    if tangs.shape[0] == 0:  # 1-D: a "facet" is a point, whose measure is 1 (dS = 1, not a length)
        return jnp.asarray(1.0, dtype=J.dtype)
    if J.ndim == 3:
        # One Jacobian per facet quadrature point (a tensor-product cell): the measure follows it and
        # becomes a (n_q,) array, which multiplies the reference weights exactly as the scalar does.
        T = jnp.einsum("td,qDd->qtD", tangs, J)  # (n_q, dim-1, dim) physical tangents
        return (
            jnp.linalg.norm(T[:, 0], axis=-1) if T.shape[1] == 1 else jnp.linalg.norm(jnp.cross(T[:, 0], T[:, 1]), axis=-1)
        )
    T = tangs @ J.T  # (dim-1, dim) physical tangents
    return jnp.linalg.norm(T[0]) if T.shape[0] == 1 else jnp.linalg.norm(jnp.cross(T[0], T[1]))


def _face_normals_jax(points, facet_verts, sign):
    """Differentiable outward unit boundary-facet normals ``(n_bfaces, dim)`` from (traced) vertex
    positions -- the JAX companion of the host-numpy :func:`fem_facets.compute_face_normals`, so a facet's
    normal re-evaluates (and stays differentiable) when its vertices move (trainable coordinates / ALE).

    ``facet_verts`` is ``(n_bfaces, dim)`` P1 vertex ids per facet (``conn.face_nodes``); ``sign`` is a
    precomputed ``±1`` per facet fixing the outward orientation. The raw normal is the 90°-rotated edge
    tangent (2D) or the edge cross product (3D); the orientation sign is **frozen** because it is locally
    constant -- it only flips at element inversion (tangling), the same validity envelope as ``detJ``. See
    plans/differentiable-r-adaptivity.md (Feature 3)."""
    v = points[facet_verts]  # (n_bfaces, n_face_nodes, dim)
    dim = v.shape[-1]
    if dim == 2:  # edge -> rotate the tangent 90°
        t = v[:, 1] - v[:, 0]
        n_raw = jnp.stack([t[:, 1], -t[:, 0]], axis=1)
    else:  # triangular face -> cross product of two edges
        n_raw = jnp.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n = sign[:, None] * n_raw
    return n / jnp.linalg.norm(n, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Native fem_context for the VPINN / grouped-weak-form path
# ---------------------------------------------------------------------------


def build_native_fem_context(domain, *, element_type, quad_degree, vec=1, neumann_tags=(), dirichlet_node_ids=None):
    """Build ``domain.fem_context`` for the VPINN / grouped-weak-form evaluator
    (``trace_evaluator._eval_grouped_assembly``).

    Returns ``(fem_context, vol_quad_points, surface_quad_by_tag, surface_normals_by_tag)``,
    computed from the native Lagrange element + facet machinery. The geometry is affine-simplex
    (P1 vertices), so the cell Jacobian, ``JxW`` and physical gradients are exact for P1/P2 nodal
    bases.
    """
    dim = int(domain.dimension)
    # element-type label -> polynomial order: TRI6/TET10 == P2; generic "TRI-P{k}"/"TET-P{k}" carries k.
    order = 2 if element_type in ("TRI6", "TET10") else (int(element_type.split("-P")[1]) if "-P" in element_type else 1)
    quad_degree = max(quad_degree, 2 * order)

    _refuse_tensor_product(
        mesh_cell_type(domain, dim),
        "the VPINN / grouped-weak-form context",
        "this second assembler forms ONE Jacobian per cell from `verts[i+1] - verts[0]`, a simplex "
        "formula, where a bilinear cell needs one per quadrature point.",
    )
    pts_p1, cells_p1, pts_f, cells_f = _get_mesh(domain, dim, order)
    spec = _lagrange_simplex(dim, order, quad_degree)
    ref_vals = jnp.asarray(spec.ref_values)  # (n_q, n_dof, 1)
    ref_grads = jnp.asarray(spec.ref_grads)  # (n_q, n_dof, 1, dim)
    qp = jnp.asarray(spec.quad_points)  # (n_q, dim)
    qw = jnp.asarray(spec.quad_weights)  # (n_q,)
    n_q, n_dof = int(qw.shape[0]), int(ref_vals.shape[1])
    test_vec = int(vec)

    pts_j = jnp.asarray(pts_p1)
    cells_p1_j = jnp.asarray(cells_p1, dtype=jnp.int32)
    cells_f_j = jnp.asarray(cells_f, dtype=jnp.int32)
    n_cells = int(cells_f.shape[0])
    num_total_nodes = int(pts_f.shape[0])

    def _cell(c):
        verts = pts_j[cells_p1_j[c]]  # (dim+1, dim) — P1 geometry vertices
        J = jnp.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)  # (dim, dim)
        detJ = jnp.linalg.det(J)
        phi, dphi = identity_pushforward(ref_vals, ref_grads, J, detJ)  # (n_q,n_dof), (n_q,n_dof,dim)
        JxW = qw * jnp.abs(detJ)  # (n_q,)
        xq = verts[0] + qp @ J.T  # (n_q, dim)
        return phi, dphi, JxW, xq

    phis, dphis, JxWs, xqs = jax.vmap(_cell)(jnp.arange(n_cells))

    N_flat = phis.reshape(-1, n_dof)  # (n_cells*n_q, n_dof)
    dN_dx_flat = dphis.reshape(-1, n_dof, dim)  # (n_cells*n_q, n_dof, dim)
    # v_grads_JxW = physical test gradient * JxW, broadcast over the test-vec component axis
    vg = (dphis * JxWs[:, :, None, None])[:, :, :, None, :]  # (n_cells,n_q,n_dof,1,dim)
    v_grads_JxW_flat = jnp.broadcast_to(vg, (n_cells, n_q, n_dof, test_vec, dim)).reshape(-1, n_dof, test_vec, dim)
    quad_points = xqs.reshape(-1, dim)

    local_areas = jnp.einsum("cq,cqa->ca", JxWs, phis)  # lumped nodal areas
    global_areas = jax.ops.segment_sum(local_areas.reshape(-1), cells_f_j.reshape(-1), num_segments=num_total_nodes)

    dirichlet_nodes = (
        jnp.asarray(sorted(set(int(i) for i in dirichlet_node_ids)), dtype=jnp.int32)
        if dirichlet_node_ids
        else jnp.asarray([], dtype=jnp.int32)
    )

    fem_context = {
        "cells": cells_f_j,
        "flat_cells": cells_f_j,
        "global_areas": global_areas,
        "N_flat": N_flat,
        "dN_dx_flat": dN_dx_flat,
        "v_grads_JxW_flat": v_grads_JxW_flat,
        "JxW": JxWs,
        "quad_points": quad_points,
        "test_vec": test_vec,
        "num_total_nodes": num_total_nodes,
        "dirichlet_nodes": dirichlet_nodes,
        "surface_data": {},
    }

    # ---- surface_data per Neumann tag (boundary weak terms) ----
    surface_quad_by_tag: dict = {}
    surface_normals_by_tag: dict = {}
    if neumann_tags:
        cell_key = {1: "interval", 2: "triangle"}.get(dim, "tetrahedron")
        conn = build_facet_connectivity(cells_p1, cell_key)
        normals_all = compute_face_normals(pts_p1, conn, cells_p1, cell_key) if conn.n_bfaces > 0 else np.zeros((0, dim))
        fp_phi, fp_dphi_ref, fp_qp, fp_tangs, gw_face = _build_face_tables(order, quad_degree, dim)
        for tag in neumann_tags:
            region_nodes = {int(n) for n in _region_node_ids_from_pts(domain, tag, pts_p1)}
            face_ids = [
                fi
                for fi in range(conn.n_bfaces)
                if all(int(conn.face_nodes[fi, j]) in region_nodes for j in range(conn.face_nodes.shape[1]))
            ]
            if not face_ids:
                continue
            face_ids_j = jnp.asarray(face_ids, dtype=jnp.int32)
            parent = jnp.asarray(conn.parent_cell, dtype=jnp.int32)[face_ids_j]
            lface = jnp.asarray(conn.local_face, dtype=jnp.int32)[face_ids_j]
            normals_j = jnp.asarray(normals_all)[face_ids_j]

            def _face(c, k, n_vec, _fp=fp_phi, _fd=fp_dphi_ref, _fq=fp_qp, _ft=fp_tangs, _gw=gw_face):
                verts = pts_j[cells_p1_j[c]]
                J = jnp.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)
                K = jnp.linalg.inv(J)
                phi_f = _fp[k]  # (n_fq, n_dof)
                dphi_f = jnp.einsum("qnd,dD->qnD", _fd[k], K)  # (n_fq, n_dof, dim)
                jac_f = _facet_area_element(J, _ft[k])  # edge length (2D) / face area (3D)
                nanson = _gw * jac_f  # (n_fq,)
                xq_f = verts[0] + _fq[k] @ J.T  # (n_fq, dim)
                return phi_f, dphi_f, nanson, xq_f

            phi_fs, dphi_fs, nanson_fs, xq_fs = jax.vmap(_face)(parent, lface, normals_j)
            # (n_faces, n_fq, n_dof), (n_faces, n_fq, n_dof, dim), (n_faces, n_fq), (n_faces, n_fq, dim)
            n_fq = int(phi_fs.shape[1])
            parent_nodes = cells_f_j[parent]  # (n_faces, n_loc) global parent-cell node ids
            local_b_areas = jnp.einsum("fq,fqn->fn", nanson_fs, phi_fs)
            global_b_areas = jax.ops.segment_sum(
                local_b_areas.reshape(-1), parent_nodes.reshape(-1), num_segments=num_total_nodes
            )
            quad_pts_flat = xq_fs.reshape(-1, dim)
            # outward normals broadcast to every face quad point
            quad_normals = jnp.broadcast_to(normals_j[:, None, :], (len(face_ids), n_fq, dim)).reshape(-1, dim)

            fem_context["surface_data"][tag] = {
                "flat_parent_nodes": parent_nodes.reshape(-1),
                "face_shape_vals": phi_fs,
                "face_shape_grads": dphi_fs,
                "nanson_scale": nanson_fs,
                "global_boundary_areas": global_b_areas,
                "quad_points": quad_pts_flat,
                "quad_normals": quad_normals,
            }
            surface_quad_by_tag[tag] = np.asarray(quad_pts_flat)
            surface_normals_by_tag[tag] = np.asarray(quad_normals)

    return fem_context, np.asarray(quad_points), surface_quad_by_tag, surface_normals_by_tag


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


class _CellFieldData(dict):
    """One field's per-cell data, with ``shape_hess`` built only if a term actually reads it.

    ``ref_hess`` is tabulated for **every** Lagrange element (``lagrange_tri``/``lagrange_tet`` fill
    it unconditionally), so the old ``if ref_hess is not None`` guard never fired: a plain P1 Poisson
    computed physical shape Hessians it has no way to use. That is not free -- the push-forward is a
    three-operand ``einsum`` over the batched cell axis, and it measured **467 ms of the 3716 ms**
    spent compiling one ``jno.fem()`` build, 12.6% of the total, on a problem with no second
    derivative anywhere.

    Only a 4th-order weak form (Argyris/Morley plates, biharmonic) reads it, through
    ``fem_utils._field_hess``, which already uses ``.get`` and raises a clear error on ``None``. So
    the value is produced on first read instead of always: identical for the forms that need it,
    absent for the ones that do not.

    ``get`` is overridden as well as ``__missing__`` because ``_field_hess`` reaches the key through
    ``.get`` -- and ``dict.get`` does not consult ``__missing__``, so overriding only the latter would
    have deferred the work and then silently reported "this element has no second derivatives".
    """

    __slots__ = ("_hess_fn",)

    def __init__(self, base, hess_fn):
        super().__init__(base)
        self._hess_fn = hess_fn

    def __missing__(self, key):
        if key != "shape_hess" or self._hess_fn is None:
            raise KeyError(key)
        self["shape_hess"] = value = self._hess_fn()
        return value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def assemble_fem_native(
    domain,
    volume_terms: List[Any],
    boundary_terms: Dict[str, List[Any]],
    dirichlet_raw: List[Tuple],
    ic_residuals: List[Any],
    *,
    vec: int,
    quad_degree: int,
    evolution: Optional[Dict[Any, Any]] = None,
    bounded: bool = False,
    tv_dirichlet_external: bool = False,
) -> Tuple[Any, str]:
    """Assemble a Lagrange FEM system into ``(op, mode, offs)`` for :class:`FEM`.

    ``mode`` is ``"linear"``, ``"nonlinear"``, or ``"transient"``; ``op`` matches the
    return-type contract of :func:`fem_1d.assemble_fem_1d` and
    :func:`fem_nonnodal.assemble_fem_nonnodal`.

    Scope: scalar/vector Lagrange P1/P2 fields on 2D triangle and 3D tetrahedral meshes
    (single- and multi-field), with Dirichlet and Neumann/Robin boundary conditions (2D edge /
    3D tet-face surface quadrature).
    """
    from ...trace import FemResidualOperator

    dim = int(domain.dimension)
    if dim not in (2, 3):
        raise NotImplementedError(f"assemble_fem_native: only dim=2 and dim=3 are supported; got dim={dim}.")
    # The facet machinery's name for this cell. Derived from the MESH, not the dimension: a 2-D mesh
    # is triangles or quads, a 3-D one tets or hexes, and naming the simplex regardless was what fed
    # a triangle's 3-entry facet table to a quadrilateral's 4 local faces.
    _cell_type = mesh_cell_type(domain, dim)
    cell_key = {"triangle": "triangle", "tetra": "tetrahedron"}.get(_cell_type, _cell_type)

    ctx = dict(getattr(domain, "context", {}) or {})
    ctx.pop("cell_size", None)  # `dom.cell_size` placeholder; the real per-cell h is packed per volume element below
    # Same for every `u.gap(secondary, main)` placeholder: dropping it means a gap that assembly has not
    # packed raises as an unresolved symbol instead of silently evaluating to the zero placeholder --
    # which would read as "everywhere exactly in contact" and be believed.
    for _k in [k for k in ctx if str(k).startswith("gap_")]:
        ctx.pop(_k, None)

    # -------------------------------------------------------------------------
    # Field layout inference
    # -------------------------------------------------------------------------

    fields: List[Dict] = []
    field_index: Dict[Any, int] = {}
    for bare in volume_terms:
        for _, sub in _split_additive_terms(domain, bare):
            lowered = _lower_statefield_to_trial(sub, {})
            fs, _ = _infer_fields(lowered)
            for f in fs:
                if f["field_key"] not in field_index:
                    field_index[f["field_key"]] = len(fields)
                    fields.append(f)

    if not fields:
        raise ValueError("assemble_fem_native: no trial fields found in volume_terms.")

    for f in fields:
        sp = f.get("space", "Lagrange")
        if sp not in ("Lagrange", "", "cover"):
            raise NotImplementedError(
                f"assemble_fem_native: only Lagrange and cover fields are supported; got space={sp!r}. "
                "Use assemble_fem_nonnodal for RT/N1E/P0 fields."
            )

    # Interpolation-cover enrichment (space="cover"): each node carries its value plus `M` cover
    # coefficients, so the field's DOF nodes are the mesh nodes REPEATED (1+M) times. That keeps the
    # global map `offs + node*vec + comp` and the rectangular `cells_f` intact -- every formula below
    # is unchanged -- at the cost of allocating the slots everywhere (docs/fem.md, Known limitations).
    # Scope limits are raised here rather than discovered downstream.
    _cover = [str(f.get("space", "") or "").lower() == "cover" for f in fields]
    if any(_cover):
        if dim not in (2, 3):
            raise NotImplementedError(
                f"space='cover': interpolation covers are implemented on 2-D triangles and 3-D "
                f"tetrahedra; this domain is {dim}-D."
            )
        for _i, _is in enumerate(_cover):
            if _is and int(fields[_i]["order"]) != 1:
                raise NotImplementedError(
                    f"space='cover' is built on the P1 hats and takes order=1; got order="
                    f"{fields[_i]['order']}. The cover supplies the extra order, not `order=`."
                )
    _cblk = [cover_block(dim) if c else 1 for c in _cover]
    # Selective enrichment: a boolean over the REAL mesh nodes saying which carry covers, stashed on
    # the domain by `jno.solve.enrich(...)` between adaptive rounds. The slots exist either way (the
    # padded layout is uniform); an unenriched node simply has its cover DOFs pinned to zero, which
    # is the same mechanism a Dirichlet condition uses and costs the solve nothing after elimination.
    # None means "every node", the plain `space="cover"` case.
    _cover_mask = getattr(domain, "_fem_enriched_nodes", None)
    _cover_mask = np.asarray(_cover_mask, dtype=bool).reshape(-1) if (_cover_mask is not None and any(_cover)) else None
    # Record WHICH mask this build baked in, so a caller can tell whether an existing FEM already
    # matches the space it wants and skip re-assembling. `jno.solve.enrich` resumes from the mask it
    # finds, so its first round is exactly the case where the answer is yes.
    domain._fem_cover_mask_built = None if _cover_mask is None else _cover_mask.copy()

    # -------------------------------------------------------------------------
    # Per-field mesh data
    # -------------------------------------------------------------------------

    mesh_data = [_get_mesh(domain, dim, f["order"]) for f in fields]

    def _pad_for_cover(md, blk):
        """A cover field's DOF nodes are the mesh nodes repeated ``blk = 1+M`` times.

        Node ``i`` owns synthetic nodes ``i*blk + m``: ``m = 0`` is the ordinary value, ``m >= 1``
        the cover coefficients. The ordering matches :func:`expand_cover`'s node-major tables, which
        is what makes the local ravel and the global ``cdofs`` agree."""
        if blk == 1:
            return md
        p1c, c1c, pf, cf = md
        pf_a, cf_a = np.asarray(pf), np.asarray(cf)
        pf2 = np.repeat(pf_a, blk, axis=0)
        cf2 = (cf_a[:, :, None] * blk + np.arange(blk)[None, None, :]).reshape(cf_a.shape[0], -1)
        return (p1c, c1c, pf2, cf2)

    # The padded arrays are a DOF layout, not a mesh: a cover field's connectivity has (1+M) columns
    # per vertex and its point array repeats coordinates, so anything TOPOLOGICAL (boundary facets)
    # or coordinate-based (region predicates) must run on the unpadded pair and map the answer back.
    # Keeping both is the whole cost of the padded layout; forgetting to is a silently wrong answer,
    # which is exactly what a boundary-facet walk over 9-column "cells" produces.
    mesh_data_real = mesh_data
    mesh_data = [_pad_for_cover(md, _cblk[i]) for i, md in enumerate(mesh_data)]
    pts_f_real = [np.asarray(md[2]) for md in mesh_data_real]  # unpadded: real DOF nodes per field
    cells_f_real = [np.asarray(md[3]) for md in mesh_data_real]  # unpadded: a genuine simplex mesh
    pts_p1 = mesh_data[0][0]  # (n_pts_p1, dim)    — P1 node coordinates (shared)
    cells_p1 = mesh_data[0][1]  # (n_cells, dim+1)  — P1 simplex connectivity (shared)

    pts_f_all = [d[2] for d in mesh_data]  # per-field node coords (P2 or P1)
    cells_f_all = [d[3] for d in mesh_data]  # per-field connectivity
    # CURVED (isoparametric) geometry: the reference->physical map is the order-2 nodal map, so its
    # Jacobian VARIES over the cell and must be formed per quadrature point. `_get_mesh` already
    # refused an order mismatch, so every field here is order 2 and any of them serves as the geometry.
    # A TENSOR-PRODUCT cell is in the same position for a different reason: a bilinear quad or
    # trilinear hex has a Jacobian that varies over the cell even when the cell is "straight-sided",
    # so it takes the same per-quadrature-point branch that curving introduced. (For a rectangle or
    # a box the map happens to be affine, but detecting that is an optimisation, not a correctness
    # condition, and it would have to hold for every cell in the mesh.)
    _tensor_product = _cell_type in TENSOR_PRODUCT_CELLS
    if any(_cover) and _tensor_product:
        raise NotImplementedError(
            "space='cover': simplices only. A quadrilateral/hexahedral cell has a non-constant "
            "Jacobian, and the cover's gradient term assumes the affine map."
        )
    # Whether a FACET of this cell is curved. A quadrilateral's edge is straight (a bilinear map
    # restricted to an edge is linear), so only a hexahedron's bilinear face needs a normal that
    # varies across it.
    _curved_facet = _cell_type in ("hexahedron", "hex")
    _curved = {1: "line3", 2: "triangle6"}.get(dim, "tetra10") in domain.mesh.cells_dict
    _nonaffine = _curved or _tensor_product
    _geom_field = 0
    n_nodes_f = [d[2].shape[0] for d in mesh_data]  # number of DOF nodes per field
    vecs = [int(f["vec"]) for f in fields]

    # Global DOF block offsets: [0, n0, n0+n1, ...]
    offs = [0]
    for i in range(len(fields)):
        offs.append(offs[-1] + n_nodes_f[i] * vecs[i])
    total = offs[-1]

    # Per-node cover length scale (host, constant): makes the enrichment columns O(1) instead of
    # O(h) relative to the nodal ones, so an enrichment coefficient reads as a directional
    # derivative rather than one times a mesh size.
    _cover_scale_j = jnp.asarray(nodal_scale(pts_p1, cells_p1)) if any(_cover) else None

    # Tell region-mask machinery which mesh to classify against
    domain._fem_assembly_points = pts_p1
    domain._fem_assembly_cells = cells_p1

    # The DOF coordinates the flat solution lives on (vertices + edge midpoints for P2).
    # ``FEM.points`` reads ``[0]`` so the solution can be interpreted at the right coordinates --
    # the first field's nodes. The full per-field list backs ``FEM.field_points`` for coupled
    # problems (e.g. Taylor-Hood velocity vs pressure nodes).
    domain._fem_native_dof_points = np.asarray(pts_f_all[0])
    domain._fem_native_dof_points_all = [np.asarray(p) for p in pts_f_all]

    # Field-0 assembly cells + element order, for the periodic-tie reduction (``_build_periodic_
    # reduction`` reads the assembly mesh's cells to extract boundary facets); ``_finalize`` reads
    # these. The full per-field lists back the heterogeneous-order coupled periodic reduction
    # (Taylor-Hood: per-field P_i from each field's own cells/order, matched to its ties by field_key).
    domain._fem_native_assembly_cells = np.asarray(cells_f_real[0])
    domain._fem_native_assembly_order = int(fields[0]["order"])
    domain._fem_native_assembly_cells_all = [np.asarray(cf) for cf in cells_f_all]
    domain._fem_native_field_orders = [int(f["order"]) for f in fields]
    domain._fem_native_field_keys = [f["field_key"] for f in fields]
    domain._fem_native_field_shapes = [tuple(f["value_shape"]) for f in fields]

    # -------------------------------------------------------------------------
    # Element specs and JAX constants
    # -------------------------------------------------------------------------

    # On a CURVED cell the map is not affine, so `1/detJ` makes the integrand RATIONAL and no rule is
    # exact any more -- the degree stops being a correctness setting and becomes an accuracy one. Two
    # extra degrees by default, with `jno.fem(quad_degree=...)` still overriding. Measured on the disk
    # rate study this changes the answer by <0.01% (7.6721e-06 -> 7.6719e-06 at degree 8), so it is
    # insurance against a form whose coefficients vary more sharply, not a fix for anything observed.
    _qd = (quad_degree + 2) if _nonaffine else quad_degree
    if any(_cover):
        # A cover basis function is h_i * (x - x_i): degree 2, not 1. The entry-side floor
        # (`_fem.py`, max(quad_degree, 2*order)) sees only `order`, which stays 1 here, so the bump
        # has to happen where the enrichment is known. Without it the mass matrix silently
        # under-integrates and only a convergence study would notice.
        _qd = max(_qd, 2 * (1 + COVER_DEGREE))
    specs = [_lagrange_simplex(dim, f["order"], _qd, cell_type=_cell_type) for f in fields]
    # All specs share the same simplex quadrature rule (basix is deterministic)
    qp_shared = jnp.asarray(specs[0].quad_points)  # (n_quad, dim)
    qw_shared = jnp.asarray(specs[0].quad_weights)  # (n_quad,)

    pts_j = jnp.asarray(pts_p1)
    cells_j = jnp.asarray(cells_p1, dtype=jnp.int32)
    n_cells = int(cells_p1.shape[0])

    ref_vals_all = [jnp.asarray(s.ref_values) for s in specs]  # list of (n_quad, n_dof_i, 1)
    ref_grads_all = [jnp.asarray(s.ref_grads) for s in specs]  # list of (n_quad, n_dof_i, 1, dim)
    # reference Hessians (for 4th-order / biharmonic weak forms); None if a spec doesn't tabulate them
    ref_hess_all = [None if s.ref_hess is None else jnp.asarray(s.ref_hess) for s in specs]
    cells_f_j = [jnp.asarray(cf, dtype=jnp.int32) for cf in cells_f_all]  # list of (n_cells, n_local_i)

    # Per-field cell DOF index arrays: (n_cells, n_local_i * vec_i)
    cdofs = []
    for i in range(len(fields)):
        comp = jnp.arange(vecs[i])
        cd = offs[i] + cells_f_j[i][:, :, None] * vecs[i] + comp[None, None, :]
        cdofs.append(cd.reshape(n_cells, -1))

    # -------------------------------------------------------------------------
    # Per-region masks (collected from all volume terms)
    # -------------------------------------------------------------------------

    def _collect_masks(terms):
        return tuple(
            sorted(
                {
                    r
                    for bare in terms
                    for _, sub in _split_additive_terms(domain, bare)
                    for r in _collect_region_mask_names(_lower_statefield_to_trial(sub, {}))
                }
            )
        )

    region_mask_names: Tuple[str, ...] = _collect_masks(volume_terms)
    region_mask_arrays = [
        jnp.asarray(_cell_region_mask(domain, r), dtype=qw_shared.dtype).reshape(-1) for r in region_mask_names
    ]
    _region_mask_index = {r: i for i, r in enumerate(region_mask_names)}  # O(1) lookup vs list().index() per cell

    # Temporal variable tags (e.g. "__time__") used inside the weak form's coefficients -- a
    # time-dependent source s(x,t) or operator. The residual/Jacobian builders thread the runtime time
    # `t` into the kernel's volume_vars at the matching slots so `_eval_integrand` resolves them;
    # the packing order is [temporal..., runtime_param..., region_mask...] (see _make_internal_vars).
    _temporal_tag_set: set = set()
    for bare in volume_terms:
        _gather_temporal_tags(bare, _temporal_tag_set)
    for _exprs in boundary_terms.values():
        for bare in _exprs:
            _gather_temporal_tags(bare, _temporal_tag_set)
    temporal_tags: Tuple[str, ...] = tuple(sorted(_temporal_tag_set))

    # Runtime parameters (trainable ``jno.np.parameter(...)`` coefficients, e.g. an unknown diffusivity
    # in an inverse problem). Their values arrive at solve time in an ``args`` dict; the builders pack
    # them into volume_vars right AFTER the temporal slots so ``_eval_integrand`` resolves each
    # parameter node (layout [temporal..., runtime_param..., region_mask...]). A SCALAR parameter is
    # broadcast; a nodal FIELD parameter k(x) (``jno.np.parameter(phi)``) has its per-cell nodal values
    # gathered and interpolated to the quad points via the field's shape functions.
    from .parametric_helpers import _is_fem_field_parameter

    _rt_param_exprs: Dict[str, Any] = {}
    for bare in volume_terms:
        _collect_runtime_parameter_exprs(bare, _rt_param_exprs)
    for _exprs in boundary_terms.values():
        for bare in _exprs:
            _collect_runtime_parameter_exprs(bare, _rt_param_exprs)
    runtime_parameter_tags: Tuple[str, ...] = tuple(sorted(_rt_param_exprs))
    _field_param_names: set = {n for n, expr in _rt_param_exprs.items() if _is_fem_field_parameter(expr)}
    # P0 field parameters carry ONE value per element rather than per node, so they gather by cell
    # index and need no interpolation -- the kernel's scalar branch broadcasts them over the quad
    # points. The design variable of density-based topology optimisation is exactly this.
    from .parametric_helpers import _fem_field_kind

    _cell_field_names: set = {n for n, expr in _rt_param_exprs.items() if _fem_field_kind(expr) == "cell"}

    # Neural coefficients (``jno.nn.wrap(net)`` called inside the weak form, e.g. ``net(x,y)*u.dx*v.dx``).
    # Unlike scalar/nodal parameters they never enter the per-cell ``volume_vars`` -- a weight pytree is
    # cell-independent -- the kernel instead re-evaluates the network at the quad points from the
    # {name: module} table (``neural_local_table``) threaded via ``loc["neural_coefficients"]``. A neural
    # coefficient needs NO per-field resolution (unlike a nodal FIELD parameter, which gathers on one
    # field's mesh): the net is evaluated at the shared physical quad points, and a trial-input
    # ``net(u_i)`` resolves its field through ``_field_data`` (op_id/field_key) inside the kernel -- so a
    # coupled (multi-field) form threads it unchanged. The collect / crux-delivery / kernel-table
    # mechanism lives in ``parametric_helpers`` (shared with the non-nodal assembler).
    from .parametric_helpers import collect_neural_slots, neural_local_table, neural_operator_exprs

    _neural = collect_neural_slots(volume_terms, boundary_terms, runtime_parameter_tags=runtime_parameter_tags)
    neural_param_names, _neural_models = _neural.param_names, _neural.models
    _param_and_neural_exprs = neural_operator_exprs(_rt_param_exprs, _neural)

    # Trainable mesh-coordinate parameters (geometry design variables) registered by
    # ``Variable.trainable()`` on a spatial coordinate: their value is scattered into the P1 geometry
    # points before the cell Jacobian is formed (``_apply_coord_params`` below), so ``∂(solve)/∂X``
    # flows through the ordinary assembly. They ride ``runtime_parameter_exprs`` (so crux discovers them
    # and their value arrives in ``args``) but stay OUT of ``runtime_parameter_tags`` -- they are not term
    # coefficients (``_runtime_vals`` must not pack them). See plans/differentiable-r-adaptivity.md (Feature 2).
    _coord_specs: List[Tuple[Any, int, str]] = []
    for _cspec in getattr(domain, "_trainable_coords", None) or []:
        _cname = str(_cspec["name"])
        _param_and_neural_exprs = {**_param_and_neural_exprs, _cname: _cspec["expr"]}
        _coord_specs.append((jnp.asarray(_cspec["ids"], dtype=jnp.int32), int(_cspec["axis"]), _cname))

    # Frozen fields (ui.freeze(values)): KNOWN nodal vectors whose value/gradient are delivered at the
    # quad points (e.g. as neural-coefficient inputs). Collected once; their per-cell nodal slice is a
    # compile-time constant gathered below and threaded via loc["frozen_fields"].
    def _collect_frozen_fields(terms):
        from ...trace import FrozenField
        from .solver_helper import iter_children

        found: Dict[Any, Any] = {}
        seen: set = set()
        stack = list(terms)
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            if isinstance(n, FrozenField):
                found[n.frozen_id] = n
                continue
            stack.extend(iter_children(n))
        return found

    _frozen_nodes = _collect_frozen_fields(list(volume_terms) + list(boundary_terms))

    # Per-quadrature-point STEP HISTORY (``v.i(k)``): scan the terms for HistoryRef nodes and record, per
    # base variable, how many past states to buffer (the most-negative offset). The buffer itself lives on
    # the runtime ``args`` (so it UPDATES each load step without re-assembly and rides the driver's scan
    # carry differentiably); here we only fix the layout so the driver can allocate and thread it. Buffer
    # shape per variable: ``(n_cells, n_quad, depth, *value_shape)`` -- per Gauss point, exactly `depth`
    # deep, so it is memory-minimal. Presence of any history forces the args-threading (parametric) path.
    from ...trace import history_variables as _history_variables

    # Evolution updates (``state.evolves(formula)``) advance internal states between load steps. Their
    # formulas are walked here too, so a ``state.i(-1)`` that appears ONLY inside an evolution formula
    # (not in the weak form) still allocates its buffer with the right depth.
    _evolution = dict(evolution or {})  # {history_key: StateUpdate}
    # WHERE each step-history state lives: a state read at ``.i(k)`` inside a BOUNDARY term buffers on that
    # region's face quadrature points (a *surface* state — e.g. a friction slip on the contact face);
    # otherwise it buffers on the cell quadrature points (a *volume* state — e.g. a plastic strain). Each
    # state's evolution formula is walked together with the reads it belongs to. (Boundary terms were
    # previously not scanned at all — ``list(boundary_terms)`` yielded the region keys, not the terms.)
    _bterm_list = [t for terms in boundary_terms.values() for t in terms]
    _surf_read_regions: Dict[Any, str] = {}  # history key -> the boundary region it is read on
    for _R, _rterms in boundary_terms.items():
        for _k in _history_variables(_rterms):
            _surf_read_regions[_k] = _R
    _surf_read_keys = set(_surf_read_regions)  # history keys read on ANY boundary
    _vol_evo = [su.formula for k, su in _evolution.items() if k not in _surf_read_keys]
    _surf_evo = [su.formula for k, su in _evolution.items() if k in _surf_read_keys]
    _vol_history_raw = _history_variables(list(volume_terms) + _vol_evo)  # {key: (base, depth)}
    _surf_history_raw = _history_variables(_bterm_list + _surf_evo)
    _both = set(_vol_history_raw) & set(_surf_history_raw)
    if _both:
        raise ValueError(
            "jno.fem: a step-history state is read at `.i(k)` on BOTH a volume and a boundary term; a "
            "state lives on one quadrature set (cells or faces). Split it into separate states."
        )
    history_specs = {
        key: {
            "name": str(getattr(base, "name", "hist")),
            "depth": int(depth),
            "value_shape": tuple(getattr(base, "value_shape", ())),
            "shape": (n_cells, int(qp_shared.shape[0]), int(depth)) + tuple(getattr(base, "value_shape", ())),
        }
        for key, (base, depth) in _vol_history_raw.items()
    }
    # Surface-state buffer specs are allocated below, once the boundary facet tables (face count +
    # per-face quadrature width) are built; kept here so the role/readout pass sees every state.
    _history_raw = {**_vol_history_raw, **_surf_history_raw}

    # Per-history-key READOUT + role, for the load-step march. Every buffered state advances one of two
    # ways between steps: (1) a *primary unknown* read at ``.i(-1)`` (its base is one of the solved fields
    # — e.g. a BDF2 ``u.i(-1)``) auto-buffers the just-solved ``u``, so its readout is the bare field
    # interpolated to the quad points; (2) an *internal state* (``ep``) advances by its
    # ``state.evolves(formula)`` update. A state read at ``.i(-1)`` that is NEITHER solved NOR has an
    # ``.evolves`` would leave its buffer frozen at zero — a silently wrong (deformation-theory) result —
    # so that is a hard build error (never a silent freeze). ``readout_formulas`` maps each key to the
    # trace expression the march evaluates per quad point to produce that state's next value.
    _solved_field_keys = {f["field_key"] for f in fields}
    _is_march = bool(getattr(domain, "_is_pseudo_time", False))
    history_roles: Dict[Any, str] = {}
    readout_formulas: Dict[Any, Any] = {}
    for key, (base, _depth) in _history_raw.items():
        if key in _evolution:
            history_roles[key] = "internal"
            readout_formulas[key] = _lower_statefield_to_trial(_evolution[key].formula, {})
        elif getattr(base, "field_key", None) in _solved_field_keys:
            history_roles[key] = "primary"  # auto-buffered from the solved unknown (the bare field at QPs)
            readout_formulas[key] = _lower_statefield_to_trial(base, {})
        elif _is_march:
            # A ``tau=`` domain signals a load-step MARCH: a buffered internal state with no ``.evolves``
            # would stay frozen at zero every step (a silently wrong, deformation-theory result). Fail
            # loud. (On a plain domain the same read is allowed — a residual you thread history into by
            # hand, e.g. to verify the zero-history reduction — so this only fires when marching.)
            raise ValueError(
                f"jno.fem: internal state {str(getattr(base, 'name', 'state'))!r} is read at `.i(-1)` but "
                "has no `.evolves(...)` update — on a `domain(tau=...)` march its history buffer would stay "
                "frozen at zero (a silently wrong, deformation-theory result). Add "
                "`state.evolves(<formula>)` to the `jno.fem([...])` list describing how it advances; or, if "
                "it is really the primary unknown, solve for it (give it a test function)."
            )
        else:
            history_roles[key] = "frozen"  # plain-domain history read, threaded by hand; not marchable

    # A trainable DIRICHLET VALUE ``u(region) - net(x)`` (an unknown boundary profile). The net is not an
    # integrand coefficient -- it is evaluated at the boundary NODES to form the Dirichlet lift -- so it is
    # collected here from ``dirichlet_raw`` (a bare net node; the front-end already rejected compound values)
    # and joins ``_param_and_neural_exprs`` as its own ``ModelWeights`` slot. The lift is (re-)built from the
    # runtime args in ``_dirichlet_pairs_at`` so ``∂b/∂weights`` flows through the solve.
    from ..._fem import _bare as _bare_node
    from ..._fem import _essential_spec as _essential_spec_node
    from ...trace import ModelCall, ModelWeights
    from .parametric_helpers import _is_neural_coefficient, _neural_coefficient_name

    _dir_net_models: Dict[str, Any] = {}
    for _fk, _rg, _comp, _val, _vnode in dirichlet_raw:
        _vn = _bare_node(_vnode) if _vnode is not None else None
        if _vn is not None and _is_neural_coefficient(_vn):
            _dir_net_models[_neural_coefficient_name(_vn)] = _vn.model
    # A trainable ``jno.np.parameter`` in an ESSENTIAL value -- ``u(top) - g``, or ``u(top) -
    # g*profile(x)``. Exactly the coord-params pattern above: it rides ``runtime_parameter_exprs``
    # (crux discovers it, its value arrives in ``args``) but stays OUT of ``runtime_parameter_tags``
    # -- it is not a term coefficient, and ``_runtime_vals`` must not pack it per cell. The rows are
    # recorded so ``_dirichlet_pairs_at`` re-forms the lift from args (``∂b/∂g`` flows through the
    # symmetric elimination, the same contract the net-valued branch documents) and
    # ``_build_dirichlet_pairs`` skips them instead of freezing the stored value behind ``float(g)``.
    #
    # The one exclusion: an optimizer-less FIELD-sized parameter stays the nodal DATA-field value the
    # branch in ``_build_dirichlet_pairs`` gathers per node (a neighbour's field in a DD solve). A
    # SCALAR optimizer-less parameter is runtime-parametric here -- the old path crashed on it
    # (IndexError gathering a length-1 value by node id) rather than meaning anything.
    _dir_param_exprs: Dict[str, Any] = {}
    _dir_param_rows: set = set()
    for _i, (_fk, _rg, _comp, _val, _vnode) in enumerate(dirichlet_raw):
        _vn = _bare_node(_vnode) if _vnode is not None else None
        if _vn is None or _is_neural_coefficient(_vn):
            continue
        if (
            isinstance(_vn, ModelCall)
            and getattr(_vn.model, "_is_parameter", False)
            and getattr(_vn.model, "_opt_fn", None) is None
            and np.asarray(_vn.model.module.value).size > 1
        ):
            continue  # nodal data field -- the per-node gather branch owns it
        _found: Dict[str, Any] = {}
        _collect_runtime_parameter_exprs(_vnode, _found)
        if _found:
            from ..._fem import _is_temporal_value_node as _is_tv

            if _is_tv(_vnode):
                # `u(top) - g * tau`: the value is BOTH parametric and time/τ-dependent. The parametric
                # branch would hold it constant in τ (silently un-ramping the load); the temporal branch
                # would freeze the parameter at its stored value (silently un-training it). Neither is
                # right, so refuse until the two held-value mechanisms compose.
                raise NotImplementedError(
                    f"jno.fem: the essential value on {_rg!r} is BOTH runtime-parametric "
                    f"({sorted(_found)}) and time/τ-dependent. A trainable parameter in a t/τ-varying "
                    "essential value is not supported yet -- train the amplitude through a Neumann/body "
                    "term written as a function of τ, or fix one of the two."
                )
            _dir_param_exprs.update(_found)
            _dir_param_rows.add(_i)
    if _dir_param_exprs:
        _param_and_neural_exprs = {**_param_and_neural_exprs, **_dir_param_exprs}

    def _dir_static_args() -> Dict[str, Any]:
        """Stored-value args for every args-dependent Dirichlet slot (net modules + parameter values) --
        what the static placeholders and dof-layout probes evaluate at, in place of runtime args."""
        return {n: m.module for n, m in _dir_net_models.items()} | {
            n: jnp.asarray(nd.model.module.value) for n, nd in _dir_param_exprs.items()
        }

    #: any Dirichlet condition whose held value must be (re-)formed from runtime ``args``.
    _dir_args_dependent = bool(_dir_net_models) or bool(_dir_param_rows)
    # A net-valued INITIAL condition ``u(initial) - net(x)`` (a trainable starting state, recovered from a
    # trajectory): its weights join the runtime slots the same way, and the initial state is (re-)formed
    # from the runtime args in ``_state0_at`` so ``∂traj/∂weights`` flows through the IC.
    _ic_net_models: Dict[str, Any] = {}
    for _ic in ic_residuals:
        _icv = _essential_spec_node(_bare_node(_ic))[1]
        _vn = _bare_node(_icv) if _icv is not None else None
        if _vn is not None and _is_neural_coefficient(_vn):
            _ic_net_models[_neural_coefficient_name(_vn)] = _vn.model
    if _dir_net_models or _ic_net_models:
        _param_and_neural_exprs = {
            **_param_and_neural_exprs,
            **{n: ModelWeights(m) for n, m in _dir_net_models.items()},
            **{n: ModelWeights(m) for n, m in _ic_net_models.items()},
        }
    # A nodal FIELD parameter k(x) interpolates on one field's FE space. Single field -> field 0. For a
    # coupled (multi-field) problem, associate it with the field whose test function appears in the
    # term(s) that reference it (e.g. mu(x)*(grad u . grad v) -> the velocity field), so its nodal values
    # gather/interpolate on THAT field's mesh. All field params must resolve to one field (a single shared
    # ``shape_vals`` threads the interpolation), else it is rejected.
    _field_param_field_idx = 0
    if _field_param_names and len(fields) > 1:
        from .parametric_helpers import _contains_fem_field_parameter

        _pf_idxs: set = set()
        for bare in volume_terms:
            for sign, sub in _split_additive_terms(domain, bare):
                coeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
                if _contains_fem_field_parameter(coeff):
                    tfi = _test_field_index(coeff, field_index)
                    if tfi is not None:
                        _pf_idxs.add(int(tfi))
        if len(_pf_idxs) == 1:
            _field_param_field_idx = _pf_idxs.pop()
        else:
            # A field parameter shared across several fields' terms (a material property common to coupled
            # equations, e.g. a conductivity in both a thermal and a coupling term) is allowed WHEN those
            # fields share ONE FE space: same element order on the one mesh -> identical nodes, connectivity
            # and shape_vals, so k(x) interpolates the same regardless of which field's space we pick. Only
            # DIFFERING orders are ambiguous (no shared node set) and stay rejected.
            _orders = {int(fields[i]["order"]) for i in _pf_idxs}
            if not _pf_idxs or len(_orders) != 1:
                raise NotImplementedError(
                    "jno.fem (native): a FEM field parameter k(x) on a coupled (multi-field) problem must "
                    "appear in the terms of fields sharing ONE FE space (same element order) — its nodal "
                    "values interpolate on that shared space. Resolved to fields "
                    f"{sorted(_pf_idxs)} with orders {sorted(_orders)} (differing orders share no node set)."
                )
            _field_param_field_idx = min(_pf_idxs)

    # -------------------------------------------------------------------------
    # Surface integration setup
    # -------------------------------------------------------------------------

    # Per-field facet tables (one set per distinct element order). These tabulate the parent basis on
    # the simplex facets for surface (Neumann/Robin) integration -- a triangle's 3 edges in 2D, a tet's
    # 4 triangular faces in 3D -- so they are skipped when there are no boundary terms.
    face_tables_per_field = (
        [_build_face_tables(f["order"], quad_degree, dim, _cell_type) for f in fields]
        if (boundary_terms and not _refuse_tensor_product_surface(_cell_type))
        else [None] * len(fields)
    )
    # face_tables_per_field[i] = (face_phi, face_dphi_ref, face_ref_qp, face_ref_tangs, face_w);
    # shapes: (n_faces, n_q, n_dof_i), (..., dim), (n_faces, n_q, dim), (n_faces, dim-1, dim), (n_q,)

    conn = build_facet_connectivity(cells_p1, cell_key)
    normals_np = compute_face_normals(pts_p1, conn, cells_p1, cell_key) if conn.n_bfaces > 0 else np.zeros((0, dim))
    # Frozen outward-orientation sign per boundary facet, so the normals can be recomputed differentiably
    # from moved vertices (``_face_normals_jax``) when coordinates are trainable (Feature 3). The sign is
    # locally constant (flips only at element inversion), so freezing it keeps the normal smooth on valid
    # meshes; the raw (unsigned) normal is built the same way as ``compute_face_normals``.
    if conn.n_bfaces > 0:
        _fv = np.asarray(pts_p1)[conn.face_nodes]  # (n_bfaces, n_face_nodes, dim)
        if dim == 2:
            _nraw = np.stack([(_fv[:, 1] - _fv[:, 0])[:, 1], -(_fv[:, 1] - _fv[:, 0])[:, 0]], axis=1)
        else:
            _nraw = np.cross(_fv[:, 1] - _fv[:, 0], _fv[:, 2] - _fv[:, 0])
        _facet_sign_j = jnp.asarray(np.where(np.sum(_nraw * np.asarray(normals_np), axis=1) >= 0, 1.0, -1.0))
        _facet_verts_j = jnp.asarray(conn.face_nodes, dtype=jnp.int32)
    else:
        _facet_sign_j = jnp.zeros((0,))
        _facet_verts_j = jnp.zeros((0, dim), dtype=jnp.int32)

    # Surface step-history buffer layout (now that the facet tables give the per-face quadrature width). A
    # state read at ``.i(k)`` on a boundary term (e.g. a friction slip on the contact face) lives on the
    # boundary FACE quadrature points: shape ``(n_bfaces, n_quad_surf, depth, *value_shape)``, indexed by
    # the global boundary-face id in ``_surf_elem_res`` (faces outside the term's region keep unused,
    # zeroed slots -- cheap, and avoids per-region local re-indexing). Threaded on ``args`` under a key
    # distinct from the volume history so the two never collide.
    _n_quad_surf = int(face_tables_per_field[0][4].shape[0]) if face_tables_per_field[0] is not None else 0
    surface_history_specs = {
        key: {
            "name": str(getattr(base, "name", "hist")),
            "depth": int(depth),
            "value_shape": tuple(getattr(base, "value_shape", ())),
            "shape": (int(conn.n_bfaces), _n_quad_surf, int(depth)) + tuple(getattr(base, "value_shape", ())),
            "surface": True,
        }
        for key, (base, depth) in _surf_history_raw.items()
    }
    # ---- which boundary facets a named region owns -------------------------------------------------
    # ONE resolver, shared by every consumer below (surface terms, surface states, contact gaps). It used
    # to be the same all-nodes-in-region mask copied three times, and that mask is wrong for the two
    # sides of a non-conforming interface: `_region_node_ids` resolves a tag through a *coordinate*
    # predicate, and the two sides sit at identical coordinates, so it returns their UNION. Measured on
    # a 2-D stack, the 'cap' side selected 17 facets -- 12 of the cap (normal [0,-1]) and 5 of the base
    # ([0,+1]) -- so a traction meant for one body was applied to both, and the gap projected the
    # interface onto itself (`g0 == 0` identically). See `plans/contact-main-reaction.md`.
    #
    # The mesh already knows the answer: `domain.tag_facets` carries each tag's facets by node id, per
    # side (measured: 0 shared nodes between the two sides), whatever their vertex count -- picking the
    # store by dimension instead returned nothing at all for a hexahedral mesh, whose boundary facets are
    # quadrilaterals, and the term silently fell back to the all-nodes-in-region mask below. Matching on
    # facet identity is exact, so prefer it and keep the mask only for tags the mesh does not name.
    _facet_lookup: Dict[frozenset, int] = {
        frozenset(int(conn.face_nodes[fi, j]) for j in range(conn.face_nodes.shape[1])): fi for fi in range(conn.n_bfaces)
    }

    _region_faces_cache: Dict[str, np.ndarray] = {}

    def _region_faces(region: str) -> np.ndarray:
        """Boundary-face ids owned by ``region`` (``(n,) int32``, possibly empty).

        Memoized for this assembly. The selection is a property of the MESH, not of any solution, so
        recomputing it can only ever return the same answer -- but *when* it is computed matters: the
        resolution walks `tag_node_mask`, which intersects the tag with the catch-all "boundary"
        region under `jax.vmap`. Run once at build time that is concrete; run again from inside a
        traced objective -- which is what happens, because `FEM.eval` builds a fresh term list on each
        call and so never hits `_preprocess_terms`'s own memo -- it produces a traced mask, and
        converting one raises `TracerArrayConversionError`. That is what stopped a SURFACE objective
        (a free surface, `(u.n)^2` on a moving wall) from surviving the rebuild after a remesh: the
        region resolved cleanly twice, then a third time under trace. Caching here removes the third.

        The cache lives in this assembly's closure, so a rebuilt problem starts with an empty one and
        fills it from its own build -- it cannot serve a stale mesh's facets.
        """
        _hit_cached = _region_faces_cache.get(region)
        if _hit_cached is not None:
            return _hit_cached
        out = _region_faces_uncached(region)
        _region_faces_cache[region] = out
        return out

    def _region_faces_uncached(region: str) -> np.ndarray:
        _named = domain.tag_facets(region) if hasattr(domain, "tag_facets") else None
        if _named is not None and region != "boundary":
            # `"boundary"` keeps the mask: it is the catch-all, with its own interface-dropping semantics.
            _hit = [_facet_lookup[k] for f in np.asarray(_named) if (k := frozenset(int(x) for x in f)) in _facet_lookup]
            # No hit at all means these facets are not boundary facets of the assembly mesh (a CONFORMING
            # interface is interior), or the numbering differs -- fall through rather than silently
            # selecting nothing, which would drop the term without a word.
            if _hit:
                return np.unique(np.asarray(_hit, dtype=np.int32))
        _owner = (getattr(domain, "_tag_regions", {}) or {}).get(region)
        _pred = (getattr(domain, "_tag_predicates", {}) or {}).get(region)
        if _owner is not None and _pred is not None:
            # The raw predicate over this mesh's boundary-facet nodes, exactly as `_boundary_node_ids`
            # does it -- NOT `_region_node_ids`, whose location function intersects with the "boundary"
            # region, and a non-conforming interface is deliberately kept out of that region. Going
            # through it here selected nothing at all on the very interface the tag names.
            _bn = np.unique(np.asarray(conn.face_nodes).reshape(-1))
            _co = np.asarray(pts_p1)[_bn]
            _hit_p = np.asarray(_pred(*(_co[:, i] for i in range(dim))), dtype=bool).reshape(-1)
            _rnodes = {int(x) for x in _bn[_hit_p]}
        else:
            _rnodes = {int(n) for n in _region_node_ids(domain, region)}
        # `d.tag(..., region=...)`: the predicate selects BOTH coincident sides, and ownership is the
        # only thing that separates them -- exactly as `_boundary_node_ids` does it for Dirichlet nodes.
        # Without this the documented way to name one side of an interface still resolves to both.
        if _owner is not None:
            # A body is a VOLUME region and has no boundary location function, so its nodes come from
            # the mesh's own set (`tag_indices`), not from a predicate.
            _own = (getattr(domain, "tag_indices", {}) or {}).get(_owner)
            if _own is None:
                raise ValueError(
                    f"tag({region!r}, region={_owner!r}): {_owner!r} has no nodes on this mesh, so the "
                    "two sides of the interface cannot be told apart. `region=` must name a body "
                    f"(a Shape.regions name). Known: {sorted(getattr(domain, 'tag_indices', {}) or {})}."
                )
            _rnodes &= {int(n) for n in np.asarray(_own).reshape(-1)}
        _mask = np.array(
            [
                all(int(conn.face_nodes[fi, j]) in _rnodes for j in range(conn.face_nodes.shape[1]))
                for fi in range(conn.n_bfaces)
            ]
        )
        return np.where(_mask)[0].astype(np.int32)

    # Boundary-face ids per region that carries a surface state (the faces its readout advances).
    _surf_region_faces: Dict[str, np.ndarray] = {}
    if surface_history_specs and conn.n_bfaces > 0:
        for _R in set(_surf_read_regions.values()):
            _surf_region_faces[_R] = _region_faces(_R)

    # ---- per-tag facet masks: the surface twin of `region_mask_arrays` -----------------------------
    # `domain.by_tag({tag: value})` desugars to `sum_t TagMask(t) * value`, which lets ONE boundary
    # term carry a coefficient that varies across tags -- the surface mirror of `by_region`.
    #
    # The mask comes from `_region_faces`, the assembler's OWN facet selection, not from re-evaluating
    # the tag predicate. Two reasons: a `TagMask("wall")` then covers exactly the facets a Dirichlet
    # condition on "wall" pins (one selection rule, not a second one that can disagree), and no
    # tolerance-tight predicate is re-run under float32, where `x > 1 - 1e-9` rounds to `x > 1.0` and
    # matches nothing at all (`domain.tag_node_mask` moved the node path off JAX for exactly this).
    #
    # Indexed by the GLOBAL boundary-face id `fi`, which is what `_surf_elem_res` already receives, so
    # no alignment against a per-region `fids` slice is needed.
    _tag_mask_names: Tuple[str, ...] = tuple(
        sorted(
            {
                t
                for _terms in (boundary_terms or {}).values()
                for bare in _terms
                for _, sub in _split_additive_terms(domain, bare)
                for t in _collect_tag_mask_names(_lower_statefield_to_trial(sub, {}))
            }
        )
    )
    _tag_mask_arrays: Dict[str, Any] = {}
    for _t in _tag_mask_names:
        _faces_t = np.asarray(_region_faces(_t), dtype=np.int64) if conn.n_bfaces > 0 else np.zeros(0, dtype=np.int64)
        if _faces_t.size == 0:
            raise ValueError(
                f"domain.by_tag: tag {_t!r} owns no boundary facet on this mesh, so its term would "
                f"integrate over nothing. Check the tag's predicate, or drop it from the mapping."
            )
        _m = np.zeros(int(conn.n_bfaces), dtype=np.float64)
        _m[_faces_t] = 1.0
        _tag_mask_arrays[_t] = jnp.asarray(_m, dtype=qw_shared.dtype)

    # ---- contact gaps: u.gap(secondary, main) -> per-face tables, built once, on the host -----------
    # For every secondary face this precomputes where its quadrature points land on the main surface:
    # the main nodes each point reads (`ids`), their shape weights (`w`), and the initial along-normal
    # separation (`g0`). At solve time the moving part is a plain gather, so the gap is differentiable
    # in the DOFs; the pairing itself is frozen, which is what limits this to SMALL SLIDING.
    _gap_tables: Dict[str, dict] = {}
    _contact_pairs = dict(getattr(domain, "_contact_pairs", {}) or {})
    if _contact_pairs and conn.n_bfaces > 0:
        from .fem_utils import interface_gap_data

        for _key, (_secondary, _main, _fkey) in _contact_pairs.items():
            _fidx = field_index.get(_fkey)
            if _fidx is not None and face_tables_per_field[_fidx] is None:
                # Face tables are only built when the form HAS boundary terms. A gap that no term ever
                # reads has nothing to pack -- and crashing on the missing tables would blame the
                # assembler for what is really a declared-but-unused `u.gap`.
                raise ValueError(
                    f"u.gap({_secondary!r}, {_main!r}) was declared but no boundary term reads it, so there "
                    "is no surface to evaluate it on. Use the gap in a term on the secondary face (a contact "
                    "traction), or drop the u.gap call."
                )
            if _fidx is None:  # the gap's field is not in this system -> nothing to pack
                continue
            _sf = np.asarray(_region_faces(_secondary), dtype=np.int32)
            _mfaces = np.asarray(_region_faces(_main), dtype=np.int32)
            _mf = np.asarray(conn.face_nodes, dtype=np.int64)[_mfaces]
            if _sf.size == 0 or _mf.size == 0:
                raise ValueError(
                    f"u.gap({_secondary!r}, {_main!r}): found {_sf.size} secondary and {len(_mf)} main boundary "
                    "facets. Both faces must select whole boundary facets -- check the tag predicates."
                )
            if np.intersect1d(_sf, _mfaces).size:
                # The two sides must be DISJOINT facet sets. They are not when a tag resolves through a
                # coordinate predicate, because the two sides of a non-conforming interface are
                # coincident -- the gap would then project the secondary face onto itself and read g0 == 0
                # everywhere, which looks like a perfectly tied interface instead of a bug.
                raise ValueError(
                    f"u.gap({_secondary!r}, {_main!r}): the two sides share boundary facets, so they are not "
                    "two distinct surfaces. This happens when a tag is defined by a coordinate predicate "
                    "over an interface whose sides are coincident -- name the sides by their mesh tags "
                    "(domain.interface_tags()) or pass `region=` to domain.tag() to say which body owns each."
                )
            # Physical quadrature points of each secondary face, formed exactly as `_surf_elem_res` does.
            _fp_qp = np.asarray(face_tables_per_field[_fidx][2])  # (n_faces_local, n_q, dim)
            _pc, _lk = np.asarray(conn.parent_cell)[_sf], np.asarray(conn.local_face)[_sf]
            _verts = pts_f_real[_fidx][cells_f_real[_fidx][_pc][:, : dim + 1]]
            _J = np.stack([_verts[:, i + 1] - _verts[:, 0] for i in range(dim)], axis=-1)  # (n_s, dim, dim)
            _xq = _verts[:, 0][:, None, :] + np.einsum("fqd,fDd->fqD", _fp_qp[_lk], _J)
            _nrm = np.broadcast_to(np.asarray(normals_np)[_sf][:, None, :], _xq.shape)
            _ids, _w, _g0 = interface_gap_data(_xq, _mf, np.asarray(pts_f_all[_fidx]), _nrm)
            _g0_full = np.zeros((conn.n_bfaces, np.asarray(_g0).shape[1]))
            _g0_full[_sf] = np.asarray(_g0)
            # Scattered up to ALL boundary faces for the same reason ``_gap_gather`` does it: a term's
            # face list is built separately from the gap's, so the reaction indexes by global face id
            # rather than assuming the two orders agree.
            _ids_full = np.zeros((conn.n_bfaces,) + np.asarray(_ids).shape[1:], dtype=np.int64)
            _w_full = np.zeros((conn.n_bfaces,) + np.asarray(_w).shape[1:])
            _ids_full[_sf], _w_full[_sf] = np.asarray(_ids), np.asarray(_w)
            _gap_tables[_key] = {
                "faces": jnp.asarray(_sf, dtype=jnp.int32),
                "ids": jnp.asarray(np.asarray(_ids), dtype=jnp.int32),  # (n_s, n_q, k) main node ids
                "w": jnp.asarray(np.asarray(_w)),  # (n_s, n_q, k)
                "ids_full": jnp.asarray(_ids_full, dtype=jnp.int32),  # (n_bfaces, n_q, k)
                "w_full": jnp.asarray(_w_full),  # (n_bfaces, n_q, k), zero off the secondary face
                "g0_full": jnp.asarray(_g0_full),  # (n_bfaces, n_q), zero off the secondary face
                "field": int(_fidx),
                "secondary": _secondary,
            }

    # -------------------------------------------------------------------------
    # Cell-level field data builder (called inside vmap'd kernels)
    # -------------------------------------------------------------------------

    def _apply_coord_params(pts, args):
        """Scatter trainable mesh-coordinate parameters (``Variable.trainable()`` on a spatial coordinate)
        into the P1 geometry points, so the cell Jacobian and quad-point coordinates become differentiable
        in them. A no-op (returns ``pts`` unchanged) when there are no coordinate parameters. Called once
        per residual/Jacobian evaluation; the resulting dynamic points thread down into ``_cell_fields``."""
        if not _coord_specs or args is None:
            return pts
        for _ids, _axis, _name in _coord_specs:
            if _name in args:
                pts = pts.at[_ids, _axis].set(jnp.asarray(args[_name], dtype=pts.dtype).reshape(-1))
        return pts

    def _cell_fields(c, cell_sols, pts=pts_j):
        """Per-field ``(phi, dphi_phys, cell_sol)`` and shared ``(xq, meas)`` for cell c.

        ``cell_sols`` is a list of this cell's local DOF values per field, shape
        ``(n_local_i, vec_i)``. The residual path gathers them from the global state; the
        per-cell Jacobian path passes a *differentiated* local slice so ``jax.jacfwd`` sees
        an element-sized (not global) input — keeping the AD intermediate O(n_local), not
        O(n_dofs). ``pts`` is the (possibly coordinate-parameter-scattered) P1 geometry points;
        it defaults to the static mesh and is overridden per-eval when coordinates are trainable."""
        if _nonaffine:
            # x(ξ) = Σ_a x_a N_a(ξ) over the geometry nodes, so J_dn(ξ) = Σ_a x_a[d] ∂N_a/∂ξ_n is a
            # function of ξ. Everything downstream that was one number per cell -- detJ, the
            # measure, the push-forward's J⁻¹ -- becomes one per quadrature point. This branch is
            # the general isoparametric map: it serves an order-2 curved simplex and a
            # bilinear/trilinear tensor-product cell without knowing which it has.
            gverts = pts[cells_f_j[_geom_field][c]]  # (n_geom, dim)
            J = jnp.einsum("ad,qan->qdn", gverts, ref_grads_all[_geom_field][..., 0, :])  # (n_q, dim, dim)
            detJ = jnp.linalg.det(J)  # (n_q,)
            xq = ref_vals_all[_geom_field][..., 0] @ gverts  # (n_q, dim)
        else:
            verts = pts[cells_j[c]]  # (dim+1, dim)
            J = jnp.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)  # (dim, dim) columns = edges
            detJ = jnp.linalg.det(J)
            xq = verts[0][None, :] + qp_shared @ J.T  # (n_quad, dim) physical qp
        meas = jnp.abs(detJ)  # scalar (affine) or (n_quad,) (curved)

        per = []
        for i in range(len(fields)):
            phi, dphi = identity_pushforward(ref_vals_all[i], ref_grads_all[i], J, detJ)
            if _cblk[i] > 1:
                # Enrich in PHYSICAL coordinates -- h_i(ξ)(ξ - ξ_i) would be discontinuous across a
                # shared face, because the two cells disagree about ξ. Tagged "Lagrange" below so
                # the shared integrand evaluator serves the wider tables unchanged, exactly as the
                # M(cell)-transform families do.
                _cn = cells_j[c]
                phi, dphi = expand_cover(phi, dphi, xq, pts[_cn], _cover_scale_j[_cn])
            fd = _CellFieldData(
                {"shape_vals": phi, "shape_grads": dphi, "cell_sol": cell_sols[i], "space": "Lagrange"},
                # Deferred: only a 4th-order weak form ever reads this. See _CellFieldData.
                None if ref_hess_all[i] is None else (lambda _rh=ref_hess_all[i], _J=J: identity_pushforward_hess(_rh, _J)),
            )
            per.append(fd)

        return per, xq, meas

    # Cell-local DOF bookkeeping for per-cell element-Jacobian assembly. ``cell_all_dofs[c]`` lists
    # every global DOF (all fields, node-major) the cell couples, so an element matrix's columns map
    # straight back to the global matrix; ``loc_seg`` splits a gathered local vector per field.
    n_local_f = [int(cells_f_j[i].shape[1]) for i in range(len(fields))]
    loc_seg = [0]
    for i in range(len(fields)):
        loc_seg.append(loc_seg[-1] + n_local_f[i] * vecs[i])
    cell_all_dofs = jnp.concatenate(cdofs, axis=1) if len(cdofs) > 1 else cdofs[0]  # (n_cell, n_local_all)

    # A LOAD-PATH field (``freeze_path``) is a FrozenField whose nodal values vary per load step: split it
    # out of the compile-time frozen gather, keep only its per-cell connectivity, and let the load-step
    # driver deliver each step's nodal slice through ``args["__loadpath__"]`` (like ``__history__``).
    from ...trace import LoadPathField as _LoadPathField

    _path_nodes = {fid: n for fid, n in _frozen_nodes.items() if isinstance(n, _LoadPathField)}
    _frozen_nodes = {fid: n for fid, n in _frozen_nodes.items() if not isinstance(n, _LoadPathField)}

    # Per-cell gather of each frozen field's nodal slice (n_cell, n_local, 1) -- a compile-time constant
    # (no args threading, no jacfwd tangent), gathered on the frozen field's own FE space via the same
    # connectivity as the live state, so its shape-gradient contraction matches the trial gradient.
    _frozen_gathered: Dict[Any, Any] = {}
    for _fid, _fnode in _frozen_nodes.items():
        _ffidx = field_index[_fnode.field_key]
        _fconn = cells_f_j[_ffidx]  # (n_cell, n_local)
        _fvals = jnp.asarray(_fnode.values)
        # scalar frozen field (n_nodes,) -> per-cell (n_local, 1); VECTOR (n_nodes, vec) -> (n_local, vec).
        # The kernel interpolation ``shape_vals . cell_nodal`` handles either (the trailing axis is carried).
        _frozen_gathered[_fid] = (
            _fvals[_fconn].reshape(_fconn.shape[0], _fconn.shape[1], 1) if _fvals.ndim == 1 else _fvals[_fconn]
        )

    # Load-path fields are scalar P1 fields on the mesh vertices (a temperature history, say) that are not
    # among the solved unknowns, so they have no assembled basis of their own. They borrow the nodal basis
    # and vertex connectivity of a P1 Lagrange field already in the problem (both live on the same mesh
    # vertices): we alias the load-path field's key to that P1 field's index so the kernel resolves its
    # shape functions, and gather its per-cell nodal slice on the same connectivity. Values arrive per step
    # from args; a spec (the full frame stack) rides the driver's scan.
    _path_conn: Dict[Any, Any] = {}
    path_specs: Dict[Any, Any] = {}
    if _path_nodes:
        _p1_idx = next(
            (i for i, f in enumerate(fields) if int(f["order"]) == 1 and str(f.get("space", "Lagrange")) == "Lagrange"),
            None,
        )
        if _p1_idx is None:
            raise NotImplementedError(
                "freeze_path(...): the load-path field is scalar P1 on the mesh vertices and borrows the "
                "nodal basis of a P1 Lagrange field in the problem, but this form has none. Give the primary "
                "unknown order=1 (P1)."
            )
        for _fid, _fnode in _path_nodes.items():
            # scalar and VECTOR load-path fields both borrow the P1 nodal basis (per-component interpolation
            # uses the same shape functions); the driver delivers the per-step slice (n_nodes[, vec]).
            field_index[_fnode.field_key] = _p1_idx  # resolve the load-path field's basis to the P1 field
            _path_conn[_fid] = cells_f_j[_p1_idx]  # scalar P1 vertex connectivity (n_cell, n_local)
            path_specs[_fid] = {"name": _fnode.name, "frames": jnp.asarray(_fnode.path_frames), "n_steps": _fnode.n_steps}

    if path_specs and not (_is_march and history_specs):
        # A load-path field's per-step slice is delivered by the load-step driver; without a march (a
        # `tau=` grid + step-history to drive it) it would never be supplied. Fail loud, name the fix.
        raise ValueError(
            "jno.fem: a `freeze_path(...)` load-path field requires a load-step march — build the domain "
            "with `domain(tau=(start, end, n))` and include step-history (a `.i(-1)` state advanced by "
            "`.evolves`, e.g. the plastic strain εₚ) so `fem.solve()` marches the load path and delivers "
            "each step's field slice. On a plain/steady domain the per-step values are never threaded."
        )

    def _add_loadpath_fields(loc, c, args):
        """Merge this load step's per-cell nodal slice for each load-path field into
        ``loc['frozen_fields']`` — so the FrozenField kernel path interpolates it to the quad points.
        The per-step nodal values come from the driver on ``args['__loadpath__']`` (like ``__history__``);
        without them (e.g. a non-march assembly) the field is simply absent, and a build-time guard has
        already required a march when a load-path field is present."""
        if not _path_conn or not isinstance(args, dict):
            return
        pbuf = args.get("__loadpath__")
        if not pbuf:
            return
        fz = dict(loc.get("frozen_fields", {}))
        for _fid, _conn in _path_conn.items():
            if _fid in pbuf:
                _arr = jnp.asarray(pbuf[_fid])
                # scalar field: (n_nodes,) -> per-cell (n_local, 1); vector field (prev-state mass):
                # (n_nodes, vec) -> per-cell (n_local, vec). The kernel interpolation handles either.
                if _arr.ndim <= 1:
                    fz[_fid] = _arr.reshape(-1)[_conn[c]].reshape(_conn.shape[1], 1)
                else:
                    fz[_fid] = _arr[_conn[c]]
        loc["frozen_fields"] = fz

    def _split_cell_local(local_vals):
        """Split a cell's gathered all-field local vector into per-field ``(n_local_i, vec_i)``."""
        return [local_vals[loc_seg[i] : loc_seg[i + 1]].reshape(n_local_f[i], vecs[i]) for i in range(len(fields))]

    def _gather_cell_local(u_blocks, c):
        """This cell's local DOFs across all fields, concatenated (matches ``cell_all_dofs[c]``)."""
        return jnp.concatenate([u_blocks[i][cells_f_j[i][c]].reshape(-1) for i in range(len(fields))])

    def _runtime_vals(c, t, args, dtype):
        """Cell ``c``'s runtime values for the kernel's volume_vars prefix, ordered
        ``[temporal..., runtime_param...]`` (region masks follow). Temporal + scalar parameters are
        single ``(1,)`` values (read back as scalars); a nodal FIELD parameter contributes this cell's
        local nodal slice ``(n_local,)`` which ``_runtime_parameter_value_from_internal_vars``
        interpolates to the quad points. Empty prefix when the form is autonomous and non-parametric."""
        tv = tuple(jnp.reshape(jnp.asarray(t, dtype=dtype), (-1,))[:1] for _ in temporal_tags)
        a = args or {}
        pv = []
        for name in runtime_parameter_tags:
            if name not in a:
                # Parameter not supplied for this assembly (e.g. the mass matrix, which references no
                # parameter): pack a zero placeholder of the right width. It is only ever read back if
                # the term actually contains the parameter node, in which case args carries its value.
                _w = (
                    n_local_f[_field_param_field_idx]
                    if (name in _field_param_names and name not in _cell_field_names)
                    else 1
                )
                pv.append(jnp.zeros((_w,), dtype))
                continue
            flat = jnp.reshape(jnp.asarray(a[name], dtype=dtype), (-1,))
            if name in _cell_field_names:
                # P0 -> this cell's single value. Width 1, so the kernel takes its scalar branch and
                # broadcasts it over the quad points -- which is what a per-element constant is.
                pv.append(jnp.reshape(flat[c], (1,)))
            else:
                # Nodal field parameter -> this cell's local nodal values on its field's mesh (field 0
                # for a single-field problem; the resolved field for a coupled one).
                pv.append(flat[cells_f_j[_field_param_field_idx][c]] if name in _field_param_names else flat[:1])
        return tv + tuple(pv)

    # -------------------------------------------------------------------------
    # Generic residual builder (volume + optional surface terms)
    # -------------------------------------------------------------------------

    # Surface connectivity (hoisted: shared by the residual and Jacobian builders).
    normals_j = jnp.asarray(normals_np)
    parent_j = jnp.asarray(conn.parent_cell, dtype=jnp.int32)
    lface_j = jnp.asarray(conn.local_face, dtype=jnp.int32)

    def _surface_normals(pts):
        """Outward unit facet normals for the current geometry ``pts``: the frozen static normals when no
        coordinates are trainable (fast path), else recomputed differentiably from the moved vertices."""
        if conn.n_bfaces == 0 or not _coord_specs:
            return normals_j
        return _face_normals_jax(pts, _facet_verts_j, _facet_sign_j)

    def _vol_elem_res(c, local_all, coeff, tfi, rnames, t=0.0, args=None, pts=None):
        """Element residual of one volume term on cell ``c`` as a function of that cell's gathered
        all-field local DOFs ``local_all`` -> ``(n_test_dofs_tfi,)``. Driving the AD off this
        element-sized input (not the global state) is what keeps the per-cell Jacobian's intermediate
        O(n_local) instead of O(n_dofs). ``t`` / ``args`` carry the runtime time and parameters, packed
        per cell into volume_vars BEFORE the region masks (layout [temporal..., runtime_param...,
        region_mask...]). ``pts`` is the coordinate-parameter-scattered geometry (``None`` -> static mesh)."""
        cell_sols = _split_cell_local(local_all)
        per, xq, meas = _cell_fields(c, cell_sols, pts_j if pts is None else pts)
        # Element size h = |detJ|^(1/dim) at the quad points -> the `dom.cell_size` symbol (SUPG/GLS).
        # Constant w.r.t. the cell DOFs (geometry only), so the per-cell Jacobian sees it as a constant.
        h_qp = jnp.broadcast_to(
            jnp.reshape(meas ** (1.0 / dim), (-1, 1)), (qw_shared.shape[0], 1)
        )  # meas: scalar (affine) or per-qp (curved)
        cell_masks = tuple(region_mask_arrays[_region_mask_index[r]][c] for r in rnames)
        loc = {
            "physical_quad_points": xq,
            "fields": per,
            "field_index": field_index,
            "tag": "fem_gauss",
            "surface": False,
            "domain_context": {**ctx, "cell_size": h_qp},
            "temporal_tags": temporal_tags,
            "runtime_parameter_tags": runtime_parameter_tags,
            "region_mask_names": rnames,
            "volume_vars": _runtime_vals(c, t, args, local_all.dtype) + cell_masks,
            "trial_value_shape": fields[tfi]["value_shape"],
            "trial_vec": vecs[tfi],
        }
        if _field_param_names:
            # The field parameter's nodal slice is interpolated to the quad points with its field's shape
            # functions (field 0 single-field; the resolved field for a coupled problem).
            # _runtime_parameter_value_from_internal_vars reads this top-level shape_vals.
            loc["shape_vals"] = per[_field_param_field_idx]["shape_vals"]
        _nt = neural_local_table(_neural, args)
        if _nt is not None:  # trainable nets ride args (crux weights); frozen/placeholder -> stored module
            loc["neural_coefficients"] = _nt
        if _frozen_gathered:  # known-field (ui.freeze) per-cell nodal slices for this cell
            loc["frozen_fields"] = {fid: g[c] for fid, g in _frozen_gathered.items()}
        if history_specs and args is not None:
            # This cell's per-quad-point history slice (n_quad, depth, *shape), gathered from the buffers on
            # ``args`` -- a plain per-cell index, so ``jacfwd`` treats it as a frozen constant.
            hbuf = args.get("__history__") if isinstance(args, dict) else None
            if hbuf:
                loc["qp_history"] = {k: hbuf[k][c] for k in history_specs if k in hbuf}
        _add_loadpath_fields(loc, c, args)  # per-step load-path field slices -> loc["frozen_fields"]
        return _integrate_term(domain, coeff, loc, qw_shared * meas)

    def _vol_elem_readout(c, local_all, formula, t=0.0, args=None):
        """Per-quadrature-point VALUE of an evolution formula on cell ``c`` -> ``(n_quad, *value_shape)``.

        Same field / parameter / frozen / history ``loc`` as :func:`_vol_elem_res` (so the formula reads
        the solved unknown through ``ε(u)`` and the previous state through ``ep.i(-1)``), but the formula
        carries NO test function, so it is *evaluated* at the quad points (``_eval_integrand``) rather than
        integrated. This is the internal-state update the load-step march applies after each solve.
        Reverse-mode differentiable in ``local_all`` (the solved DOFs) and the history buffers."""
        cell_sols = _split_cell_local(local_all)
        per, xq, meas = _cell_fields(c, cell_sols)
        h_qp = jnp.broadcast_to(
            jnp.reshape(meas ** (1.0 / dim), (-1, 1)), (qw_shared.shape[0], 1)
        )  # meas: scalar (affine) or per-qp (curved)
        loc = {
            "physical_quad_points": xq,
            "fields": per,
            "field_index": field_index,
            "tag": "fem_gauss",
            "surface": False,
            "domain_context": {**ctx, "cell_size": h_qp},
            "temporal_tags": temporal_tags,
            "runtime_parameter_tags": runtime_parameter_tags,
            "region_mask_names": (),
            "volume_vars": _runtime_vals(c, t, args, local_all.dtype),
            "trial_value_shape": fields[0]["value_shape"],
            "trial_vec": vecs[0],
        }
        if _field_param_names:
            loc["shape_vals"] = per[_field_param_field_idx]["shape_vals"]
        _nt = neural_local_table(_neural, args)
        if _nt is not None:
            loc["neural_coefficients"] = _nt
        if _frozen_gathered:
            loc["frozen_fields"] = {fid: g[c] for fid, g in _frozen_gathered.items()}
        if history_specs and args is not None:
            hbuf = args.get("__history__") if isinstance(args, dict) else None
            if hbuf:
                loc["qp_history"] = {k: hbuf[k][c] for k in history_specs if k in hbuf}
        _add_loadpath_fields(loc, c, args)  # per-step load-path field slices -> loc["frozen_fields"]
        return _eval_integrand(domain, formula, loc)

    def state_readout(u_flat, t=0.0, args=None):
        """Advance every buffered state one load step: evaluate each key's readout formula at the
        quadrature points, given the just-solved ``u_flat`` and the current history buffers (on
        ``args['__history__']``). Returns ``{history_key: (n_cells, n_quad, *value_shape)}`` — the value
        that becomes each state's ``.i(-1)`` at the NEXT step. The load-step march rolls these into the
        depth buffers. Whole-domain: the readout runs on every cell (sub-region-restricted plasticity is
        not wired — a future masked readout)."""
        local_all = u_flat[cell_all_dofs]  # (n_cell, n_local_all)
        out: Dict[Any, Any] = {}
        for key, formula in readout_formulas.items():
            if key not in history_specs:  # VOLUME states only; surface states advance in surface_state_readout
                continue
            # Chunked exactly like the residual/jacobian element maps -- a plain ``jax.vmap`` leaves the
            # batched intermediate uncapped, and a coupled form gathers every field's DOFs into one
            # ``local_all`` row, so the readout's per-cell working set grows with the field count. It also
            # runs inside the march's differentiated scan, so those intermediates are retained for the
            # backward pass. One test-DOF's worth of cost per cell is the honest proxy: the readout carries
            # no test function, so there is no element block -- only the per-quadrature-point value.
            out[key] = _elem_map(
                lambda c, la, _f=formula: _vol_elem_readout(c, la, _f, t, args),
                (jnp.arange(n_cells), local_all),
                _cell_chunk(n_cells, 1, cell_all_dofs.shape[1]),
            )
        return out

    def _gap_gather(u_flat, key):
        """``u_m . Phi`` at every secondary-face quadrature point: ``(n_secondary_faces, n_q, vec)``.

        Done GLOBALLY, outside the per-face vmap, because the main nodes live on the other body's
        cells -- they are not in the secondary face's parent-cell DOF slice. A plain weighted gather, so
        ``jax.linearize`` (the default JFNK path) picks up the main coupling in the tangent exactly,
        with no assembled Jacobian block needed. (The ASSEMBLED tangent builds its own explicit
        nonlocal blocks from these same tables -- see the gap emission in `_make_jacobian`.)
        """
        tb = _gap_tables[key]
        fidx, vt = tb["field"], vecs[tb["field"]]
        dofs = offs[fidx] + tb["ids"][..., None] * vt + jnp.arange(vt)  # (n_s, n_q, k, vec)
        um = jnp.einsum("sqk,sqkv->sqv", tb["w"], u_flat[dofs])
        # Scattered up to ALL boundary faces so a region's slice is a plain index -- the gap's own face
        # list and a term's ``surface_work`` face list are built separately, and assuming they agree in
        # order would be exactly the kind of silent mismatch this file keeps warning about.
        return jnp.zeros((conn.n_bfaces,) + um.shape[1:], um.dtype).at[tb["faces"]].set(um)

    def _gaps_in(expr, region):
        """Gap keys whose SECONDARY face is ``region`` and which ``expr`` actually reads.

        A plain Neumann load on the secondary face is an *external* force and must not be mirrored onto
        the main body -- only a traction written in terms of the gap is an interface traction.
        """
        from ...trace import Variable
        from .solver_helper import iter_children

        def _refs(node, tag):
            if isinstance(node, Variable) and getattr(node, "tag", None) == tag:
                return True
            return any(_refs(c, tag) for c in iter_children(node) or ())

        return [k for k, tb in _gap_tables.items() if tb["secondary"] == region and _refs(expr, k)]

    def _contact_reaction(R, keys, fids, lv, gslice, bcoeff, btfi, region, t, args, pts_dyn, normals_dyn):
        """Add the equal-and-opposite interface traction to the MAIN body's DOFs.

        The secondary face carries ``R_a = sum_q w_q tau_q . N_a(x_q)``. The main's share is the same
        integrand tested against its projected trace, so it is ``-sum_q w_q tau_q . Phi_b(x_q)`` with
        ``Phi`` the mortar weights the gap already built. Testing against ``eye(n_q)`` returns the
        weighted traction ``w_q tau_q`` per quadrature point, which is what the scatter needs: the
        contributing main node varies *with the quadrature point* (a secondary facet may overlap several
        main facets), so this is a per-``(q, b)`` scatter and not a nodal shape-table contraction.

        Without this the main body feels nothing -- contact against a rigid obstacle, and a two-body
        problem that violates Newton's third law. See ``plans/contact-main-reaction.md``.
        """
        n_q = int(np.asarray(face_tables_per_field[btfi][4]).shape[0])
        tau = _elem_map(
            lambda fi, la, gp: _surf_elem_res(
                fi, la, bcoeff, btfi, region, t, args, pts_dyn, normals_dyn, gp, test_vals=jnp.eye(n_q, dtype=lv.dtype)
            ),
            (fids, lv, gslice),
            _cell_chunk(int(fids.shape[0]), n_q * vecs[btfi], cell_all_dofs.shape[1]),
        )
        for k in keys:
            tb = _gap_tables[k]
            fidx, vt = tb["field"], vecs[tb["field"]]
            if fidx != btfi:
                raise NotImplementedError(
                    f"u.gap({tb['secondary']!r}): the traction term tests a different field than the gap's "
                    f"({fields[btfi]['name']!r} vs {fields[fidx]['name']!r}). The reaction on the main "
                    "body is only defined for a traction tested against the contacting field."
                )
            tau_f = tau.reshape(tau.shape[0], n_q, vt)
            w_f, ids_f = tb["w_full"][fids], tb["ids_full"][fids]  # (n_face, n_q, k)
            dofs = offs[fidx] + ids_f[..., None] * vt + jnp.arange(vt)  # (n_face, n_q, k, vec)
            R = R.at[dofs.reshape(-1)].add(-jnp.einsum("fqk,fqv->fqkv", w_f, tau_f).reshape(-1))
        return R

    def _gap_slices(region, fids, gap_um):
        """``{key: (g0, u_m)}`` for every gap whose SECONDARY face is this region, aligned with ``fids``."""
        return {
            k: (_gap_tables[k]["g0_full"][fids], gap_um[k][fids])
            for k in _gap_tables
            if _gap_tables[k]["secondary"] == region
        } or None

    def _pack_gaps(loc, per_f, n_vec, gaps):
        """Write each contact gap's per-quadrature-point value into ``loc["domain_context"]``.

        ``g = g0 - n . (u_s - u_m . Phi)``. Shared by the residual and by the surface-state readout, so
        an augmented-Lagrangian update ``lam.evolves(max(0, lam.i(-1) - c*g))`` sees exactly the gap the
        traction term saw -- if the two drifted apart the multiplier would converge to the wrong
        pressure, and nothing would report it.
        """
        for _k, (_g0_f, _um_f) in (gaps or {}).items():
            _gfi = _gap_tables[_k]["field"]
            _us = per_f[_gfi]["shape_vals"] @ per_f[_gfi]["cell_sol"]  # (n_q, vec)
            _jump = _us.reshape(_us.shape[0], -1) - _um_f.reshape(_um_f.shape[0], -1)
            # Explicit shape check: `jnp.einsum("d,qd->q", n, jump)` silently BROADCASTS a size-1
            # component axis, so a scalar field would contract to `sum(n) * jump` -- right only when the
            # normal happens to sum to 1, wrong on any tilted interface, and never an error.
            if _jump.shape[1] != n_vec.shape[-1]:
                raise ValueError(
                    f"u.gap: the field has {_jump.shape[1]} component(s) but the interface normal has "
                    f"{n_vec.shape[-1]}. A normal gap needs a vector field with one component per "
                    "dimension -- use fem_symbols(value_shape=(dim,))."
                )
            # `n_vec` is (dim,) on a straight facet and (n_q, dim) on a curved one; `atleast_2d`
            # makes both broadcast against `_jump` (n_q, vec) without a shape branch.
            loc["domain_context"][_k] = _g0_f - (_jump * jnp.atleast_2d(n_vec)).sum(axis=1)

    def _facet_geometry(c, k, pts_src):
        """``(J, K, xq)`` for facet ``k`` of cell ``c`` -- one per quadrature point when the cell's
        map is not affine.

        A simplex cell has a single Jacobian, formed from its edge vectors. A tensor-product cell
        does not: its map is bi/trilinear, so J is evaluated the same way the volume path evaluates
        it, by contracting the GEOMETRY basis gradients (tabulated here at the facet's quadrature
        points) against the cell's vertices. The geometry connectivity is the assembly one, which is
        in basix vertex order -- the same array the basis was tabulated against.
        """
        if _tensor_product:
            gverts = pts_src[cells_f_j[_geom_field][c]]  # (n_geom, dim), basix order
            _, fd_g, _, _, _ = face_tables_per_field[_geom_field]
            Jq = jnp.einsum("ad,qan->qdn", gverts, fd_g[k])  # (n_q, dim, dim)
            xq = face_tables_per_field[_geom_field][0][k] @ gverts  # (n_q, dim): x = sum_a N_a x_a
            return Jq, jnp.linalg.inv(Jq), xq
        verts = pts_src[cells_j[c]]
        Jc = jnp.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)  # (dim, dim)
        return Jc, jnp.linalg.inv(Jc), None  # xq is formed by the caller from its own facet points

    def _surf_elem_res(
        fi, local_all, bcoeff, btfi, region, t=0.0, args=None, pts=None, normals=None, gaps=None, test_vals=None
    ):
        """Element residual of one surface term on boundary face ``fi`` as a function of the parent
        cell's gathered all-field local DOFs ``local_all`` -> ``(n_test_dofs_btfi,)``. ``pts`` / ``normals``
        are the coordinate-parameter-scattered geometry and its facet normals (``None`` -> static mesh).

        ``test_vals`` substitutes the test field's shape table (the trial values keep the real one). The
        contact reaction passes ``eye(n_q)``, which makes the return the per-quadrature-point weighted
        traction ``w_q * tau_q`` instead of a nodal residual -- see :func:`_contact_reaction`."""
        c = parent_j[fi]
        k = lface_j[fi]
        n_vec = (normals_j if normals is None else normals)[fi]  # (dim,) outward unit normal
        cell_sols = _split_cell_local(local_all)
        _pts_src = pts_j if pts is None else pts
        verts = _pts_src[cells_j[c]]
        J, K, _xq_tp = _facet_geometry(c, k, _pts_src)
        if _curved_facet:
            # A hexahedron's facet is a bilinear SURFACE: its normal turns across the facet, so one
            # frozen vector per facet is not enough. Nanson's formula gives it at each quadrature
            # point from the same physical tangents the area element already forms. Straight facets
            # (simplices, and a quad's edges) keep the per-facet vector -- it is exact there, and
            # leaving its shape alone keeps every existing consumer untouched.
            n_vec = _facet_nanson_normal(J, face_tables_per_field[btfi][3][k], _facet_sign_j[fi])

        # All-field surface data (needed for coupled Robin terms).
        per_f = []
        for i in range(len(fields)):
            fp_i, fd_i, _, _, _ = face_tables_per_field[i]
            per_f.append(
                {
                    "shape_vals": fp_i[k],
                    "shape_grads": jnp.einsum(_GSPEC(K), fd_i[k], K),
                    "cell_sol": cell_sols[i],
                    "space": "Lagrange",
                }
            )

        _, _, fp_qp, fp_tangs, face_w = face_tables_per_field[btfi]
        jac_f = _facet_area_element(J, fp_tangs[k])  # physical edge length (2D) / face area (3D)
        xq_f = _xq_tp if _xq_tp is not None else verts[0] + fp_qp[k] @ J.T  # (n_q, dim)
        loc = {
            "physical_quad_points": xq_f,
            "fields": per_f,
            "field_index": field_index,
            "tag": f"gauss_{region}",
            "surface": True,
            "domain_context": {**ctx, f"n_{region}": n_vec},
            "temporal_tags": temporal_tags,
            "runtime_parameter_tags": runtime_parameter_tags,
            "region_mask_names": (),
            "volume_vars": _runtime_vals(c, t, args, local_all.dtype),
            # This face's 0/1 value for every tag the boundary terms reference, so `TagMask` resolves
            # to a scalar here exactly as `RegionMask` does per cell. `fi` is the global face id.
            "tag_masks": {_t: _arr[fi] for _t, _arr in _tag_mask_arrays.items()},
            "trial_value_shape": fields[btfi]["value_shape"],
            "trial_vec": vecs[btfi],
        }
        # Contact gap g = g0 - n . (u_s - u_m . Phi) at this face's quadrature points. ``gaps`` carries
        # this face's frozen g0 and its already-gathered main values -- see :func:`_gap_gather`.
        _pack_gaps(loc, per_f, n_vec, gaps)
        if test_vals is not None:
            loc["test_shape_vals"] = {btfi: test_vals}
        if _field_param_names:
            loc["shape_vals"] = per_f[_field_param_field_idx]["shape_vals"]
        _nt = neural_local_table(_neural, args)
        if _nt is not None:
            loc["neural_coefficients"] = _nt
        if _frozen_gathered:  # known-field (ui.freeze) per-cell nodal slices for the parent cell
            loc["frozen_fields"] = {fid: g[c] for fid, g in _frozen_gathered.items()}
        if surface_history_specs and args is not None:
            # This face's per-quad-point surface-history slice (n_quad_surf, depth, *shape), gathered from
            # the buffers on ``args`` by the global boundary-face id -- a per-face constant, so ``jacfwd``
            # treats it as frozen (the tangent is ``∂t_fric/∂u`` with the slip history held, exactly like
            # the volume return map holds the plastic strain).
            sbuf = args.get("__surface_history__") if isinstance(args, dict) else None
            if sbuf:
                loc["qp_history"] = {k: sbuf[k][fi] for k in surface_history_specs if k in sbuf}
        return _integrate_term(domain, bcoeff, loc, face_w * jac_f)

    def _surf_elem_readout(fi, local_all, formula, region, t=0.0, args=None, gaps=None):
        """Per-quad-point VALUE of a surface evolution formula on boundary face ``fi`` -> (n_q, *shape).

        The surface analogue of :func:`_vol_elem_readout`: the same surface ``loc`` as ``_surf_elem_res``
        (fields, outward normal, per-face surface history), but the formula carries no test function, so it
        is *evaluated* (``_eval_integrand``), not integrated -- the advance for a surface state (a slip)."""
        c = parent_j[fi]
        k = lface_j[fi]
        n_vec = normals_j[fi]
        cell_sols = _split_cell_local(local_all)
        verts = pts_j[cells_j[c]]
        J, Kmat, _xq_tp = _facet_geometry(c, k, pts_j)
        per_f = []
        for i in range(len(fields)):
            fp_i, fd_i, _, _, _ = face_tables_per_field[i]
            per_f.append(
                {
                    "shape_vals": fp_i[k],
                    "shape_grads": jnp.einsum(_GSPEC(Kmat), fd_i[k], Kmat),
                    "cell_sol": cell_sols[i],
                    "space": "Lagrange",
                }
            )
        _, _, fp_qp, _fp_tangs, _fw = face_tables_per_field[0]
        xq_f = _xq_tp if _xq_tp is not None else verts[0] + fp_qp[k] @ J.T
        loc = {
            "physical_quad_points": xq_f,
            "fields": per_f,
            "field_index": field_index,
            "tag": f"gauss_{region}",
            "surface": True,
            "domain_context": {**ctx, f"n_{region}": n_vec},
            "temporal_tags": temporal_tags,
            "runtime_parameter_tags": runtime_parameter_tags,
            "region_mask_names": (),
            "volume_vars": _runtime_vals(c, t, args, local_all.dtype),
            "trial_value_shape": fields[0]["value_shape"],
            "trial_vec": vecs[0],
        }
        _pack_gaps(loc, per_f, n_vec, gaps)  # so an AL multiplier update can read the same gap
        if _field_param_names:
            loc["shape_vals"] = per_f[_field_param_field_idx]["shape_vals"]
        _nt = neural_local_table(_neural, args)
        if _nt is not None:
            loc["neural_coefficients"] = _nt
        if _frozen_gathered:
            loc["frozen_fields"] = {fid: g[c] for fid, g in _frozen_gathered.items()}
        if surface_history_specs and args is not None:
            sbuf = args.get("__surface_history__") if isinstance(args, dict) else None
            if sbuf:
                loc["qp_history"] = {kk: sbuf[kk][fi] for kk in surface_history_specs if kk in sbuf}
        return _eval_integrand(domain, formula, loc)

    def surface_state_readout(u_flat, t=0.0, args=None):
        """Advance each SURFACE state one load step: evaluate its evolves formula on its region's faces.

        Returns ``{key: (n_bfaces, n_quad_surf, *value_shape)}`` -- the region's faces filled, every other
        boundary face zero (unused). The march rolls these into the surface depth buffers."""
        out: Dict[Any, Any] = {}
        for key, spec in surface_history_specs.items():
            formula = readout_formulas.get(key)
            region = _surf_read_regions[key]
            faces = _surf_region_faces.get(region)
            full = jnp.zeros(
                (int(spec["shape"][0]), int(spec["shape"][1])) + tuple(spec["value_shape"]), dtype=u_flat.dtype
            )
            if formula is None or faces is None or len(faces) == 0:
                out[key] = full
                continue
            fids = jnp.asarray(faces, dtype=jnp.int32)
            lv = u_flat[cell_all_dofs[parent_j[fids]]]  # (n_face_R, n_local_all)
            # An augmented-Lagrangian update `lam.evolves(max(0, lam.i(-1) - c*g))` reads the gap here,
            # so the same main-side gather the residual uses has to reach the readout.
            gsl = _gap_slices(region, fids, {k: _gap_gather(u_flat, k) for k in _gap_tables})
            vals = jax.vmap(lambda fi, la, gp, _f=formula, _r=region: _surf_elem_readout(fi, la, _f, _r, t, args, gp))(
                fids, lv, gsl if gsl else {}
            )
            # Normalize to the state's declared per-face shape (n_faces, n_quad_surf, *value_shape): a
            # scalar update written with `inner(dir, u.bind(...), 1)` keeps a spurious trailing size-1 axis
            # (harmless in the residual, where it contracts with the test) that must be squeezed here.
            vals = vals.reshape((fids.shape[0], int(spec["shape"][1])) + tuple(spec["value_shape"]))
            out[key] = full.at[fids].set(vals)
        return out

    def _classify_one(coeff, where: str) -> List[Tuple[Any, int]]:
        """``[(coeff, test_field_idx), ...]`` for one lowered term. Normally one entry; a term that
        welds several test fields inside a product (the real part of a ``complex=True`` form, e.g.
        ``c·(u_r·w_r − u_i·w_i)``) is distributed over its sums into single-test sub-terms, so one
        complex form lowers onto the coupled blocks."""
        from ...trace import BinaryOp, Literal
        from .fem_utils import _expand_product_terms

        tfi = _test_field_index(coeff, field_index)
        if tfi is not None:
            return [(coeff, tfi)]
        expanded = _expand_product_terms(coeff)
        if len(expanded) > 1:
            split: List[Tuple[Any, int]] = []
            for s, sub in expanded:
                sub_signed = sub if s >= 0 else BinaryOp("*", Literal(-1.0), sub)
                sfi = _test_field_index(sub_signed, field_index)
                if sfi is None:
                    split = None
                    break
                split.append((sub_signed, sfi))
            if split is not None:
                return split
        raise ValueError(
            f"jno.fem (native): each {where} weak-form term must contain exactly one test field "
            "(it determines the equation block)."
        )

    _preprocess_cache: Dict[Tuple[int, int], Tuple[Any, Any]] = {}

    def _preprocess_terms(terms, bterms):
        """``(typed_with_masks, surface_work)``: lower each additive sub-term to
        ``(coeff, test_field_idx[, mask_names])`` and bucket boundary faces per region.

        Memoized because ``_make_residual`` and ``_make_jacobian`` are always built from the SAME term
        list back to back (``:2148``/``:2149``, and likewise for the mass and spatial pairs), so this
        ran twice per build for one answer. The second run is not cheap: the region loop below is a
        pure-Python double loop over every boundary facet and its nodes, i.e. it grows with the mesh.

        The cache is local to this ``assemble_fem_native`` call, so it cannot go stale across builds --
        it is thrown away with the closure. Keyed on the identity of the term containers, which is
        what "the same list, twice" means; a caller that mutated ``terms`` between the two calls would
        defeat it, but that would be a bug in its own right (the residual and the Jacobian must come
        from one form)."""
        _ck = (id(terms), id(bterms))
        _hit = _preprocess_cache.get(_ck)
        if _hit is not None:
            return _hit
        typed: List[Tuple[Any, int]] = []
        for bare in terms:
            for sign, sub in _split_additive_terms(domain, bare):
                coeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
                typed.extend(_classify_one(coeff, "volume"))
        typed_with_masks = [(coeff, tfi, tuple(sorted(_collect_region_mask_names(coeff)))) for coeff, tfi in typed]

        surface_work: List[Tuple[str, np.ndarray, List[Tuple[Any, int]]]] = []
        if bterms and conn.n_bfaces > 0:
            for region, bexprs in bterms.items():
                face_ids = _region_faces(region)
                if len(face_ids) == 0:
                    continue
                btyped = []
                for bexpr in bexprs:
                    for sign, sub in _split_additive_terms(domain, bexpr):
                        bcoeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
                        btyped.extend(_classify_one(bcoeff, f"boundary ({region!r})"))
                surface_work.append((region, np.asarray(face_ids, dtype=np.int32), btyped))
        _preprocess_cache[_ck] = (typed_with_masks, surface_work)
        return typed_with_masks, surface_work

    # --- element-loop chunking -----------------------------------------------------------------
    # A single `vmap` over every cell materialises the whole batched intermediate at once, and on a 3-D
    # mesh that intermediate -- not the assembled operator -- is what sets the memory ceiling. Measured
    # on a 31k-DOF 3-D nonlinear problem: the jacfwd tangent tensor `f64[n_cells, 4, 4, 4]` was 82.2 MiB
    # against a 6.8 MiB operator, and the residual's own temp (182 MiB) is the unit every other cost is
    # a multiple of -- each Krylov matvec, and the linearization the matrix-free inner solve holds live.
    #
    # `lax.map(..., batch_size=C)` vmaps C cells at a time and scans over the chunks, so the batched
    # intermediate is capped at C cells regardless of mesh size. Remainders are handled (verified at
    # several non-dividing C) and gradients match the unchunked form exactly.
    # An explicit `jno.fem(chunk=...)` is captured HERE, once, rather than read when the closures run:
    # they are called at solve time, long outside the scope that set it. The policy itself lives in
    # `fem_utils` so the non-nodal assembler applies exactly the same one.
    _chunk_setting = _CHUNK_OVERRIDE[0]
    _CHUNK_CONSUMED[0] = True

    def _cell_chunk(n_items: int, n_test: int, n_local: int):
        return cell_chunk(n_items, n_test, n_local, _chunk_setting)

    _elem_map = elem_map

    def _make_residual(terms, bterms=None):
        """Build the free global residual ``R(u_flat) -> (total,)`` (volume + optional surface).

        ``bterms`` is an optional ``{region: [exprs]}`` dict for surface (Neumann/Robin) terms; pass
        ``None`` (the default) to assemble volume terms only — used for the transient mass matrix,
        where boundary contributions must not appear."""
        typed_with_masks, surface_work = _preprocess_terms(terms, bterms)

        def residual(u_flat, t=0.0, args=None):
            R = jnp.zeros(total, dtype=u_flat.dtype)
            local_all = u_flat[cell_all_dofs]  # (n_cell, n_local_all)
            pts_dyn = _apply_coord_params(pts_j, args)  # trainable coords -> differentiable geometry

            for coeff, tfi, rnames in typed_with_masks:
                elem = _elem_map(
                    lambda c, la, _e=coeff, _t=tfi, _r=rnames: _vol_elem_res(c, la, _e, _t, _r, t, args, pts_dyn),
                    (jnp.arange(n_cells), local_all),
                    _cell_chunk(n_cells, cdofs[tfi].shape[1], cell_all_dofs.shape[1]),
                )
                R = R.at[cdofs[tfi].reshape(-1)].add(elem.reshape(-1))

            normals_dyn = _surface_normals(pts_dyn)  # differentiable facet normals under coordinate motion
            # Main-side values for every contact gap, gathered ONCE over the whole secondary face: the
            # nodes read live on the other body's cells, so this cannot happen inside the per-face map.
            gap_um = {k: _gap_gather(u_flat, k) for k in _gap_tables}
            for region, face_ids, btyped in surface_work:
                fids = jnp.asarray(face_ids, dtype=jnp.int32)
                pcells = parent_j[fids]
                lv = u_flat[cell_all_dofs[pcells]]  # (n_face, n_local_all)
                gslice = _gap_slices(region, fids, gap_um)  # {key: (g0, u_m)} for THIS region's faces
                for bcoeff, btfi in btyped:
                    contribs = _elem_map(
                        lambda fi, la, gp, _e=bcoeff, _t=btfi, _r=region: _surf_elem_res(
                            fi, la, _e, _t, _r, t, args, pts_dyn, normals_dyn, gp
                        ),
                        (fids, lv, gslice),
                        _cell_chunk(int(fids.shape[0]), cdofs[btfi].shape[1], cell_all_dofs.shape[1]),
                    )
                    R = R.at[cdofs[btfi][pcells].reshape(-1)].add(contribs.reshape(-1))
                    _rk = _gaps_in(bcoeff, region)
                    if _rk:
                        R = _contact_reaction(R, _rk, fids, lv, gslice, bcoeff, btfi, region, t, args, pts_dyn, normals_dyn)
            return R

        return residual

    def _make_jacobian(terms, bterms=None):
        """Build the dense Jacobian ``J(u_flat) -> (total, total)`` by *per-element* forward-mode AD.

        Each cell's (and boundary face's) element matrix is ``jacfwd`` of its element residual w.r.t.
        that element's local DOFs — an ``(n_test, n_local)`` block — then scatter-added into the global
        matrix. The AD never sees the global state, so the intermediate is element-sized; this is what
        a single global ``jacfwd(residual)`` cannot do (it materialises an ``O(n_dofs × n_cells)``
        tangent tensor and OOMs on any non-trivial mesh). The dense result is entry-for-entry
        identical to that global ``jacfwd``, just assembled within a per-element memory budget."""
        typed_with_masks, surface_work = _preprocess_terms(terms, bterms)

        # --- hoist the (row, col) pattern out of the trace -------------------------------------
        # The triplet INDICES come exclusively from host-static mesh connectivity (`cdofs`,
        # `cell_all_dofs`, `parent_j`, `face_ids`) and the term list; only the element blocks `Ke`
        # depend on `u_flat`/`t`/`args`. So the pattern -- and therefore the compressed nonzero count
        # -- is the same at every state, time and parameter value this is ever traced at.
        #
        # Building it once here is what lets the TRACED assemblies compress: `sum_duplicates` needs a
        # static `nse` under jit, and inferring one requires concrete indices it does not have inside
        # the trace. This mirrors `fem_nonnodal._make_sparse_assembler`, which already hoists its
        # `_vol_idx` the same way. Order must match the append order in `jacobian` exactly.
        def _gap_key_of(region):
            """The (single) gap key whose SECONDARY face is this region, or ``None``."""
            ks = [k for k in _gap_tables if _gap_tables[k]["secondary"] == region]
            return ks[0] if ks else None

        _gap_static_cache: Dict[Any, Any] = {}

        def _gap_static(region, face_ids, btfi):
            """Concrete index/weight geometry of a region's gap blocks — computed ONCE from the frozen
            pairing tables and shared by the pattern hoist and the traced assembly, so the two cannot
            drift. The three nonlocal blocks, flat index arrays in emission order:

            * ``(s,m)``: secondary test rows x main columns (one column per (q, mortar-node, comp));
            * ``(m,s)``: reaction rows (main dofs, one per (q, mortar-node, comp)) x parent-local cols;
            * ``(m,m)``: reaction rows x main columns.
            """
            k = _gap_key_of(region)
            cache_key = (region, int(btfi))
            hit = _gap_static_cache.get(cache_key)
            if hit is not None:
                return hit
            tb = _gap_tables[k]
            fidx_m, vt = tb["field"], vecs[tb["field"]]
            fids_np = np.asarray(face_ids, dtype=np.int64)
            ids_f = np.asarray(tb["ids_full"])[fids_np]  # (n_face, n_q, K)
            w_f = jnp.asarray(np.asarray(tb["w_full"])[fids_np])  # (n_face, n_q, K)
            n_face, n_q, K = ids_f.shape
            pc = np.asarray(parent_j)[fids_np]
            rows_test = np.asarray(cdofs[btfi])[pc]  # (n_face, n_test)
            n_test = rows_test.shape[1]
            dof_m = (np.asarray(offs[fidx_m]) + ids_f[..., None] * vt + np.arange(vt)).astype(np.int64)
            nqKv = n_q * K * vt
            dof_m_flat = dof_m.reshape(n_face, nqKv)
            cols_parent = np.asarray(cell_all_dofs)[pc]  # (n_face, n_local_all)
            n_local = cols_parent.shape[1]
            sh_sm = (n_face, n_test, n_q, K, vt)
            rows_sm = np.broadcast_to(rows_test[:, :, None, None, None], sh_sm).reshape(-1)
            cols_sm = np.broadcast_to(dof_m[:, None, :, :, :], sh_sm).reshape(-1)
            sh_ms = (n_face, nqKv, n_local)
            rows_ms = np.broadcast_to(dof_m_flat[:, :, None], sh_ms).reshape(-1)
            cols_ms = np.broadcast_to(cols_parent[:, None, :], sh_ms).reshape(-1)
            sh_mm = (n_face, nqKv, nqKv)
            rows_mm = np.broadcast_to(dof_m_flat[:, :, None], sh_mm).reshape(-1)
            cols_mm = np.broadcast_to(dof_m_flat[:, None, :], sh_mm).reshape(-1)
            out = {
                "rows_sm": jnp.asarray(rows_sm, dtype=jnp.int32),
                "cols_sm": jnp.asarray(cols_sm, dtype=jnp.int32),
                "rows_ms": jnp.asarray(rows_ms, dtype=jnp.int32),
                "cols_ms": jnp.asarray(cols_ms, dtype=jnp.int32),
                "rows_mm": jnp.asarray(rows_mm, dtype=jnp.int32),
                "cols_mm": jnp.asarray(cols_mm, dtype=jnp.int32),
                "w_f": w_f,
                "n_q": n_q,
                "K": K,
                "vt": vt,
                "key": k,
            }
            _gap_static_cache[cache_key] = out
            return out

        _idx_rows, _idx_cols = [], []
        for _coeff_s, _tfi_s, _rn_s in typed_with_masks:
            _sh = (n_cells, int(cdofs[_tfi_s].shape[1]), int(cell_all_dofs.shape[1]))
            _idx_rows.append(jnp.broadcast_to(cdofs[_tfi_s][:, :, None], _sh).reshape(-1))
            _idx_cols.append(jnp.broadcast_to(cell_all_dofs[:, None, :], _sh).reshape(-1))
        for _region_s, _face_ids_s, _btyped_s in surface_work:
            _pc = parent_j[jnp.asarray(_face_ids_s, dtype=jnp.int32)]
            _fcols = cell_all_dofs[_pc]
            for _bcoeff_s, _btfi_s in _btyped_s:
                _sh = (int(_pc.shape[0]), int(cdofs[_btfi_s].shape[1]), int(cell_all_dofs.shape[1]))
                _idx_rows.append(jnp.broadcast_to(cdofs[_btfi_s][_pc][:, :, None], _sh).reshape(-1))
                _idx_cols.append(jnp.broadcast_to(_fcols[:, None, :], _sh).reshape(-1))
                # The gap's nonlocal blocks, in the SAME append order the traced assembly emits
                # them: (s,m) always when the region carries a gap; (m,s) and (m,m) when this term
                # also drives the main-side reaction. The indices come from the frozen pairing
                # tables, so the pattern stays static; inactive contact contributes zeros in DATA.
                if _gap_key_of(_region_s) is not None:
                    _gs = _gap_static(_region_s, _face_ids_s, _btfi_s)
                    _idx_rows.append(_gs["rows_sm"])
                    _idx_cols.append(_gs["cols_sm"])
                    if _gaps_in(_bcoeff_s, _region_s):
                        _idx_rows.append(_gs["rows_ms"])
                        _idx_cols.append(_gs["cols_ms"])
                        _idx_rows.append(_gs["rows_mm"])
                        _idx_cols.append(_gs["cols_mm"])
        _blk_sizes = [int(r.shape[0]) for r in _idx_rows]  # per-term flat lengths, in append order
        _idx_static = (
            jnp.stack([jnp.concatenate(_idx_rows).astype(jnp.int32), jnp.concatenate(_idx_cols).astype(jnp.int32)], axis=1)
            if _idx_rows
            else None
        )
        try:
            _plan = compress_plan(_idx_static) if _idx_static is not None else None
        except Exception:  # noqa: BLE001 -- a traced pattern would break the static-count invariant
            _idx_static, _plan = None, None  # fall back to the uncompressed (still correct) path

        def jacobian(u_flat, t=0.0, args=None):
            # Assemble into COO triplets and build a BCOO -- never materialises the dense (total, total)
            # matrix (O(nnz), GPU-able at large N). Each per-element block is element-sized; duplicate
            # (i, j) triplets from neighbouring cells are summed by BCOO on matvec / todense, so the
            # per-cell blocks are simply concatenated (no pre-summation).
            # With a plan in force each element block is scattered STRAIGHT into its compressed slots
            # and then dropped, so the concatenated raw-triplet array is never built. That array and
            # the transposed copy XLA made of it were the two largest buffers in the compiled jacobian
            # after the element blocks themselves (61.6 MiB each on a 31k-DOF 3-D problem, against a
            # 6.8 MiB operator), and every per-term `Ke` had to stay alive waiting for the concatenate.
            # The row/column arrays are skipped for the same reason: the plan already has the pattern.
            _acc = [None]
            _off = [0]
            _nblk = [0]
            rows_l, cols_l, data_l = [], [], []
            local_all = u_flat[cell_all_dofs]  # (n_cell, n_local_all)
            pts_dyn = _apply_coord_params(pts_j, args)  # trainable coords -> differentiable geometry

            def _emit(flat, rows_fn, cols_fn):
                if _plan is None:
                    data_l.append(flat)
                    rows_l.append(rows_fn())
                    cols_l.append(cols_fn())
                    return
                _inv, _nse = _plan[1], _plan[2]
                k = _blk_sizes[_nblk[0]]
                part = jax.ops.segment_sum(flat, _inv[_off[0] : _off[0] + k], num_segments=_nse)
                _acc[0] = part if _acc[0] is None else _acc[0] + part
                _off[0] += k
                _nblk[0] += 1

            for coeff, tfi, rnames in typed_with_masks:

                def _ke(c, la, _e=coeff, _t=tfi, _r=rnames, _p=pts_dyn):
                    return jax.jacfwd(lambda v: _vol_elem_res(c, v, _e, _t, _r, t, args, _p))(la)

                Ke = _elem_map(  # (n_cell, n_test_tfi, n_local_all)
                    _ke,
                    (jnp.arange(n_cells), local_all),
                    _cell_chunk(n_cells, cdofs[tfi].shape[1], cell_all_dofs.shape[1]),
                )
                _emit(
                    Ke.reshape(-1),
                    lambda _K=Ke, _t=tfi: jnp.broadcast_to(cdofs[_t][:, :, None], _K.shape).reshape(-1),
                    lambda _K=Ke: jnp.broadcast_to(cell_all_dofs[:, None, :], _K.shape).reshape(-1),
                )

            normals_dyn = _surface_normals(pts_dyn)  # differentiable facet normals under coordinate motion
            # Main-side values for every contact gap -- the residual's own gather, reused so the
            # assembled tangent linearizes the SAME function the residual evaluates.
            gap_um_j = {k: _gap_gather(u_flat, k) for k in _gap_tables}
            for region, face_ids, btyped in surface_work:
                fids = jnp.asarray(face_ids, dtype=jnp.int32)
                pcells = parent_j[fids]
                lv = u_flat[cell_all_dofs[pcells]]  # (n_face, n_local_all)
                fcols = cell_all_dofs[pcells]  # (n_face, n_local_all)
                gslice = _gap_slices(region, fids, gap_um_j)  # {key: (g0, u_m)} or None
                for bcoeff, btfi in btyped:

                    def _kef(fi, la, gp=None, _e=bcoeff, _t=btfi, _r=region, _p=pts_dyn, _n=normals_dyn):
                        # gaps PACKED: the local block then carries d(traction)/du_s THROUGH the gap,
                        # exactly as `jax.linearize` of the residual would.
                        return jax.jacfwd(lambda v: _surf_elem_res(fi, v, _e, _t, _r, t, args, _p, _n, gp))(la)

                    if gslice:
                        Kef = _elem_map(
                            _kef,
                            (fids, lv, gslice),
                            _cell_chunk(int(fids.shape[0]), cdofs[btfi].shape[1], cell_all_dofs.shape[1]),
                        )
                    else:
                        Kef = _elem_map(  # (n_face, n_test_btfi, n_local_all)
                            _kef,
                            (fids, lv),
                            _cell_chunk(int(fids.shape[0]), cdofs[btfi].shape[1], cell_all_dofs.shape[1]),
                        )
                    _emit(
                        Kef.reshape(-1),
                        lambda _K=Kef, _t=btfi, _p=pcells: jnp.broadcast_to(cdofs[_t][_p][:, :, None], _K.shape).reshape(
                            -1
                        ),
                        lambda _K=Kef, _f=fcols: jnp.broadcast_to(_f[:, None, :], _K.shape).reshape(-1),
                    )

                    if gslice:
                        # ---- the gap's NONLOCAL blocks (same append order as the pattern hoist) ----
                        gs = _gap_static(region, face_ids, btfi)
                        n_q, vt, gk = gs["n_q"], gs["vt"], gs["key"]
                        w_f = gs["w_f"]
                        g0_sl, um_sl = gslice[gk]

                        # (s,m): jacfwd of the SAME face residual w.r.t. the gathered main values,
                        # chained through the frozen mortar weights to global main columns.
                        def _kem(fi, la, g0f, umf, _e=bcoeff, _t=btfi, _r=region, _p=pts_dyn, _n=normals_dyn, _k=gk):
                            return jax.jacfwd(
                                lambda um: _surf_elem_res(fi, la, _e, _t, _r, t, args, _p, _n, {_k: (g0f, um)})
                            )(umf)

                        Kem = _elem_map(  # (n_face, n_test, n_q, vt)
                            _kem,
                            (fids, lv, g0_sl, um_sl),
                            _cell_chunk(int(fids.shape[0]), cdofs[btfi].shape[1], n_q * vt),
                        )
                        d_sm = jnp.einsum("frqv,fqk->frqkv", Kem.reshape(Kem.shape[0], Kem.shape[1], n_q, vt), w_f)
                        _emit(d_sm.reshape(-1), lambda _g=gs: _g["rows_sm"], lambda _g=gs: _g["cols_sm"])

                        if _gaps_in(bcoeff, region):
                            # Reaction-row tangent: tau (the per-QP weighted traction via the identity
                            # test-table substitution -- the residual's own trick) linearized w.r.t.
                            # the parent-local dofs (m,s) and the gathered main values (m,m), scattered
                            # through the same -w weights the residual's reaction uses.
                            def _tau(fi, la, g0f, umf, _e=bcoeff, _t=btfi, _r=region, _k=gk):
                                out = _surf_elem_res(
                                    fi,
                                    la,
                                    _e,
                                    _t,
                                    _r,
                                    t,
                                    args,
                                    pts_dyn,
                                    normals_dyn,
                                    {_k: (g0f, umf)},
                                    test_vals=jnp.eye(n_q, dtype=lv.dtype),
                                )
                                return jnp.asarray(out).reshape(n_q, vt)

                            n_local_all = int(cell_all_dofs.shape[1])
                            Dls = _elem_map(
                                lambda fi, la, g0f, umf: jax.jacfwd(lambda v: _tau(fi, v, g0f, umf))(la),
                                (fids, lv, g0_sl, um_sl),
                                _cell_chunk(int(fids.shape[0]), n_q * vt, n_local_all),
                            )
                            Dum = _elem_map(
                                lambda fi, la, g0f, umf: jax.jacfwd(lambda um: _tau(fi, la, g0f, um))(umf),
                                (fids, lv, g0_sl, um_sl),
                                _cell_chunk(int(fids.shape[0]), n_q * vt, n_q * vt),
                            )
                            n_face = int(fids.shape[0])
                            # (m,s): K[dof(f,q,K,v), parent_local] -= w[f,q,K] * dtau[f,q,v,local]
                            d_ms = -jnp.einsum("fqk,fqvl->fqkvl", w_f, Dls.reshape(n_face, n_q, vt, n_local_all))
                            _emit(d_ms.reshape(-1), lambda _g=gs: _g["rows_ms"], lambda _g=gs: _g["cols_ms"])
                            # (m,m): K[dof(q,K,v), dof(q2,K2,v2)] -= w[f,q,K] dtau[q,v,q2,v2] w[f,q2,K2]
                            d_mm = -jnp.einsum("fqk,fqvpw,fpl->fqkvplw", w_f, Dum.reshape(n_face, n_q, vt, n_q, vt), w_f)
                            _emit(d_mm.reshape(-1), lambda _g=gs: _g["rows_mm"], lambda _g=gs: _g["cols_mm"])

            if _plan is not None:
                if _acc[0] is None:  # no terms -> empty operator
                    return jsparse.BCOO((jnp.zeros((0,), u_flat.dtype), jnp.zeros((0, 2), jnp.int32)), shape=(total, total))
                # ~12x fewer stored triplets, INSIDE the trace -- so it reaches the nonlinear Jacobian
                # and the per-step/parametric re-assemblies, not just eager assembly. The host-decided
                # plan makes this an O(nnz) scatter-add rather than an O(nnz log nnz) sort, which
                # matters because it runs once per Newton step / timestep / parameter value.
                return jsparse.BCOO((_acc[0], _plan[0]), shape=(total, total))
            if not data_l:  # no terms -> empty operator
                return jsparse.BCOO((jnp.zeros((0,), u_flat.dtype), jnp.zeros((0, 2), jnp.int32)), shape=(total, total))
            data = jnp.concatenate(data_l)
            if _idx_static is not None:
                idx = _idx_static
            else:  # pattern could not be hoisted -> rebuild it in-trace, uncompressed but correct
                idx = jnp.stack(
                    [jnp.concatenate(rows_l).astype(jnp.int32), jnp.concatenate(cols_l).astype(jnp.int32)], axis=1
                )
            return jsparse.BCOO((data, idx), shape=(total, total))

        # Published so a wrapper that APPENDS triplets (the Dirichlet row-replacement below) can
        # derive its own plan from the same pattern instead of re-deriving one that might disagree.
        # It must describe what `jacobian` ACTUALLY RETURNS -- the COMPRESSED pattern when a plan is
        # in force, not the raw one it was derived from. Publishing the raw pattern here silently
        # mismatched the wrapper's `inverse` against the compressed data length: the recurring shape
        # of bug in this repo is a representation changing while one of its readers does not move.
        jacobian._jno_static_idx = _plan[0] if _plan is not None else _idx_static  # type: ignore[attr-defined]
        return jacobian

    def _dirichlet_jac_rows(jac_fn, pairs):
        """Wrap an assembled-Jacobian callable so the constrained system is eliminated symmetrically —
        the matrix-level analogue of :func:`_apply_dirichlet_projected`.

        Two things must line up with that residual or Newton silently solves a different system: the
        tangent is evaluated at the **projected** state ``P(u)`` (which is where the residual samples
        the free form), and the constrained rows *and columns* are zeroed with a unit diagonal (which
        is what differentiating through the projection gives)."""
        if not pairs:
            return jac_fn
        dofs, _vals, _project = dirichlet_projection(pairs)

        # `bcoo_eliminate_dirichlet` zeroes the constrained rows and then APPENDS one (d, d, 1) triplet
        # per constrained DOF, so its output carries up to `len(dofs)` duplicates however well the
        # inner Jacobian was compressed. That count is static too: the union of the inner pattern with
        # the Dirichlet diagonal. Derived from the inner assembler's own published pattern so the two
        # cannot disagree; without it, fall back to leaving the appended duplicates in place.
        _inner_idx = getattr(jac_fn, "_jno_static_idx", None)
        _dir_plan = None
        if _inner_idx is not None:
            try:
                _d_np = np.asarray(dofs, dtype=np.int64).reshape(-1)
                _dir_plan = compress_plan(
                    np.concatenate([np.asarray(_inner_idx), np.stack([_d_np, _d_np], axis=1)], axis=0)
                )
            except Exception:  # noqa: BLE001 -- no static plan available; correctness is unaffected
                _dir_plan = None

        def jac(u_flat):
            A = bcoo_eliminate_dirichlet(jac_fn(_project(u_flat)), dofs)
            # Same host-decided plan, so this is an O(nnz) scatter-add per Newton step, not a sort.
            return apply_compress_plan(A.data, _dir_plan, A.shape) if _dir_plan is not None else A

        return jac

    # -------------------------------------------------------------------------
    # Dirichlet pair builder
    # -------------------------------------------------------------------------

    def _drop_interface_only_nodes(bf: np.ndarray, bnodes: np.ndarray, pts_all: np.ndarray) -> np.ndarray:
        """Remove nodes that sit **only** on a non-conforming interface from the catch-all boundary.

        ``Shape.regions(conforming=False)`` meshes each body independently, so the two sides of an
        interface are each a facet of exactly one cell -- topologically boundary, and correctly so.
        Semantically they are internal: a tie glues them. Without this filter a plain
        ``u(boundary) - g`` pins the interface, which silently solves two disconnected bodies. Measured
        on a tied 1x1x2 bar: ``u`` was **0.000000 at every interface node** and the peak sat at the
        unit-cube value 0.0555 instead of the correct 0.0705, converging to the wrong answer at every
        refinement rather than erroring.

        The test is *at least one non-interface facet*, not *not on an interface facet*: the ring where
        the interface meets the outer wall belongs to both and must stay pinned. A facet counts as
        interface when all its **vertices** (the first ``dim`` columns) lie on a registered ``"a|b.x"``
        region -- matched by coordinate, since a P2 assembly mesh renumbers relative to the P1 mesh the
        tags were built on, and its edge midpoints are absent from the tag point cloud entirely.
        """

        # ONLY the per-side tags of a non-conforming interface, ``"a|b.a"`` -- the suffix must be one of
        # the pair's own region names. A conforming ``"a|b"`` (or its disjoint components ``"a|b.0"``)
        # is shared by two cells and never appears among boundary facets anyway, but excluding it here
        # makes this filter a provable no-op on every existing domain rather than one that relies on
        # that: a wall triangle whose three vertices all happen to sit on the interface seam would
        # otherwise be dropped from the Dirichlet set.
        def _is_side(t: str) -> bool:
            head, dot, tail = t.rpartition(".")
            return bool(dot) and "|" in head and tail in head.split("|")

        sides = [t for t in (getattr(domain, "_interface_registry", {}) or {}) if _is_side(t)]
        regions = getattr(domain, "_boundary_regions", {}) or {}
        clouds = [np.asarray(regions[t].points) for t in sides if t in regions and regions[t].points is not None]
        if not clouds:
            return bnodes
        tol = 1.0e-9 * max(float(np.ptp(pts_all)) if pts_all.size else 1.0, 1.0)
        key = lambda a: {tuple(r) for r in np.round(np.asarray(a)[:, :dim] / max(tol, 1e-300)).astype(np.int64)}  # noqa: E731
        iface_keys = set().union(*(key(c) for c in clouds))
        node_keys = np.round(np.asarray(pts_all)[:, :dim] / max(tol, 1e-300)).astype(np.int64)
        on_iface = np.array([tuple(r) in iface_keys for r in node_keys], dtype=bool)
        # How many leading columns are the facet's VERTICES depends on the cell, not the dimension:
        # a simplex facet has `dim` of them, a quad edge 2 (= dim in 2-D, which is why this read
        # correctly until now), and a hexahedron's facet FOUR against a dim of 3. Taking `[:, :dim]`
        # there silently drops a corner, so a facet lying wholly on an interface could still test as
        # partly outside it and stay pinned.
        _n_fv = _face_table(cell_key)[1]
        verts = np.asarray(bf)[:, :_n_fv]
        outer = np.asarray(bf)[~on_iface[verts].all(axis=1)]
        if outer.size == 0:  # every boundary facet is an interface -> nothing to pin; leave as-is
            return bnodes
        return np.intersect1d(bnodes, np.unique(outer.reshape(-1)))

    def _boundary_node_ids(fidx: int, region: str) -> List[int]:
        """Boundary DOF-node ids of ``region`` for field ``fidx``, in the padded layout.

        Resolution runs on the field's REAL mesh (see ``mesh_data_real``) because the padded
        connectivity is not a mesh and a facet walk over it is meaningless; each real node found is
        then expanded to the ``1+M`` slots it owns. A Dirichlet condition therefore reaches the
        value and its covers alike -- which is what :func:`_cover_g` then tells apart, giving the
        covers zero rather than ``g``."""
        real = _boundary_node_ids_real(fidx, region)
        blk = _cblk[fidx]
        if blk == 1:
            return real
        # On a straight boundary facet through node i, (x - x_i) is TANGENTIAL, so only the
        # tangential cover components enter the trace: pinning them keeps u = g; the NORMAL
        # component is the ∂u/∂n freedom and pinning it too is what capped the L2 rate at P1's
        # (measured: 1.86 with everything pinned, 3.02 when the pin is harmless). A cover slot m is
        # freed only when e_m is orthogonal to EVERY region tangent at the node -- an oblique facet
        # therefore frees nothing, which over-constrains but never violates the condition.
        tang = _region_tangent_dirs(fidx, region, real)
        out = []
        for r in real:
            out.append(int(r) * blk)  # the value slot, always
            ts = tang.get(int(r))
            for m in range(1, blk):
                e = np.zeros(dim)
                e[m - 1] = 1.0
                if ts is None or ts.size == 0 or float(np.abs(ts @ e).max()) > 1e-9:
                    out.append(int(r) * blk + m)
        return out

    def _region_tangent_dirs(fidx: int, region: str, real_ids) -> dict:
        """Unit tangent directions of ``region``'s boundary facets, per real node.

        Facets are taken from the UNPADDED mesh and restricted to those lying wholly in the region's
        node set. A node with no incident region facet (an interior point pin) gets no tangents, so
        all of its covers stay free -- a point constraint constrains the value, nothing else."""
        from ..._fem import _boundary_facets

        node_set = set(int(r) for r in real_ids)
        out: dict = {}
        bf = _boundary_facets(pts_f_real[fidx], cells_f_real[fidx], dim, fields[fidx]["order"], _cell_type)
        if bf is None:
            return out
        pts_r = pts_f_real[fidx]
        for facet in np.asarray(bf):
            vs = [int(v) for v in facet[:dim]]  # 2-D edge: 2 vertices; 3-D triangle: first 3
            if dim == 3:
                vs = [int(v) for v in facet[:3]]
            if not all(v in node_set for v in vs):
                continue
            edges = [pts_r[vs[a]] - pts_r[vs[0]] for a in range(1, len(vs))]
            for v in vs:
                acc = out.setdefault(v, [])
                for e in edges:
                    n = float(np.linalg.norm(e))
                    if n > 0:
                        acc.append(e / n)
        return {k: np.asarray(v) for k, v in out.items()}

    def _boundary_node_ids_real(fidx: int, region: str) -> List[int]:
        """Robust boundary-DOF-node ids of ``region`` for field ``fidx``, on the UNPADDED mesh.

        Boundary nodes are taken from the assembly mesh's boundary FACETS (a node on a boundary
        facet -- P2 edge-midpoints attached by coordinate), not a geometric containment test: the
        latter can miss a P2 midpoint sitting exactly on a face (the discrete proximity test catches
        the P1 vertices but not the new midpoint). The catch-all ``"boundary"`` region is every
        boundary-facet node; a named region filters those by its spatial predicate (exact even for an
        on-face midpoint) or, lacking one, by geometric containment -- but only ever among true
        boundary nodes, so an on-boundary node is never lost to a flaky test. Falls back to the
        plain predicate-over-all-nodes finder when there are no facets (degenerate mesh) or the
        boundary set is empty (e.g. an interior pin), which keeps interior Dirichlet points working.
        """
        from ..._fem import _boundary_facets

        pts_all = pts_f_real[fidx]
        # A named interior SUB-REGION (`domain.region(name, poly)`) pins its WHOLE node set (interior +
        # boundary), by point-in-polygon — not just its boundary nodes (which is empty for an interior
        # region and would silently drop the pin). This is the subdomain / domain-decomposition pin.
        ptags = getattr(domain, "_polygon_tags", {})
        if region in (getattr(domain, "_source_regions", {}) or {}) and ptags.get(region, (None,))[0] == "interior":
            return list(_region_node_ids_from_pts(domain, region, pts_all))
        bf = _boundary_facets(pts_all, cells_f_real[fidx], dim, fields[fidx]["order"], _cell_type)
        if bf is None:
            return list(_region_node_ids_from_pts(domain, region, pts_all))
        # A locally refined (hanging-node) mesh is NON-CONFORMING, and "belongs to exactly one cell" is
        # false on it: across a 2:1 interface the coarse edge and both half-edges each belong to one
        # cell. Left in, they make the interface a boundary -- and since the aggregate "boundary" is
        # what a Dirichlet condition resolves through, the interface gets PINNED. Measured on a 4x4 grid
        # with four cells refined: 32 identity rows where 16 are the perimeter, and -Lap u = 1 came back
        # with a centre value of 0.0194 against 0.0737, with no error anywhere.
        _hang = getattr(domain, "_fem_hanging_nodes", None)
        if _hang:
            from .fem_refine import drop_covered_facets

            # n_v from the DIMENSION, not from the column count: an order-3 edge also has 4 columns
            bf = drop_covered_facets(np.asarray(bf), _hang, n_v=2 if dim == 2 else 4)
        bnodes = np.unique(np.asarray(bf).reshape(-1))
        if region == "boundary":
            bnodes = _drop_interface_only_nodes(np.asarray(bf), bnodes, pts_all)
        if region != "boundary":
            coords = pts_all[bnodes]
            pred = getattr(domain, "_tag_predicates", {}).get(region)
            if pred is not None:
                mask = np.asarray(pred(*(coords[:, i] for i in range(dim))), dtype=bool).reshape(-1)
            else:
                loc = domain._make_tag_location_fn(region)
                if loc is None:
                    return []
                mask = np.asarray(jax.vmap(loc)(jnp.asarray(coords)), dtype=bool).reshape(-1)
            bnodes = bnodes[mask]
        # `d.tag(..., region=...)`: keep only the nodes belonging to that body. Two coincident faces of
        # a non-conforming interface have identical coordinates, so the predicate above selects BOTH;
        # ownership is the only thing that separates them, and it lives in the node ids.
        _owner = (getattr(domain, "_tag_regions", {}) or {}).get(region)
        if _owner is not None and bnodes.size:
            own = np.asarray(_region_node_ids(domain, _owner), dtype=np.int64)
            if own.size == 0:
                raise ValueError(f"tag region {_owner!r} has no nodes on this mesh; cannot restrict {region!r}.")
            if int(fields[fidx]["order"]) != 1:
                raise NotImplementedError(
                    f"tag({region!r}, region={_owner!r}) is resolved against the P1 node numbering, but "
                    f"this field is P{fields[fidx]['order']}. Use order-1 elements for a region-restricted "
                    "tag, or tag the two sides by their auto names from domain.interface_tags()."
                )
            bnodes = np.intersect1d(bnodes, own)
            if bnodes.size == 0:
                raise ValueError(
                    f"tag({region!r}, region={_owner!r}) selected no nodes: the predicate and the body "
                    "do not overlap. Check the predicate reaches that body's surface."
                )
        if bnodes.size == 0:  # interior pin (no boundary facet matched) -> predicate over all nodes
            return list(_region_node_ids_from_pts(domain, region, pts_all))
        return [int(n) for n in bnodes]

    def _build_dirichlet_pairs() -> List[Tuple[int, float]]:
        from ..._fem import _eval_value_node_at, _is_temporal_value_node
        from ...trace import ModelCall

        pairs: List[Tuple[int, float]] = []
        tv_stash: List[Tuple[Any, Any, Any]] = []  # (dofs, value_node, coords) for time-varying g(x,t)
        for _row_i, (field_key, region, comp, value, value_node) in enumerate(dirichlet_raw):
            fidx = field_index.get(field_key)
            if fidx is None:
                continue
            if _row_i in _dir_param_rows:
                continue  # args-dependent value: (re-)formed per args in _dirichlet_pairs_at, never frozen here
            # Time-varying Dirichlet g(x,t): no constant pair — stash (dofs, value_node, coords) so a
            # transient caller (e.g. the second-order augmented block) writes g(x_d, t) each step.
            if value_node is not None and _is_temporal_value_node(value_node):
                vt = vecs[fidx]
                pts_all = np.asarray(pts_f_all[fidx])
                nids = list(_boundary_node_ids(fidx, region))
                coords = jnp.asarray(pts_all[np.asarray(nids, dtype=int)]) if nids else jnp.zeros((0, dim))
                for c in range(vt) if comp is None else [int(comp)]:
                    dofs = jnp.asarray([offs[fidx] + nid * vt + c for nid in nids], dtype=jnp.int32)
                    tv_stash.append((dofs, value_node, coords))
                continue
            _vn = _bare_node(value_node) if value_node is not None else None
            # A nodal DATA-field value (a `jno.np.parameter` carrying a field with NO optimizer — e.g. a
            # neighbour's field in a coupled/domain-decomposition solve) → gather its per-node values by
            # node index. Checked before the neural-coefficient branch so a bare data-field is a value,
            # not a runtime net profile.
            _field_vals = None
            if (
                isinstance(_vn, ModelCall)
                and getattr(_vn.model, "_is_parameter", False)
                and getattr(_vn.model, "_opt_fn", None) is None
            ):
                _field_vals = np.asarray(_vn.model.module.value).reshape(-1)
            elif _vn is not None and _is_neural_coefficient(_vn):
                continue  # a net-valued Dirichlet is (re-)built per args in _dirichlet_pairs_at
            vt = vecs[fidx]
            pts_all = pts_f_all[fidx]
            nids = list(_boundary_node_ids(fidx, region))
            # Evaluate g on ALL of the region's boundary nodes at once. Doing it per node meant one
            # traced evaluation and one device->host sync per node -- 14.6k of each on a 75k-node
            # mesh, and the largest single cost in assembly. ``_eval_value_node_at`` already takes a
            # batch of points (the initial-condition path below passes the whole array).
            pts = np.asarray(pts_all)[nids] if nids else np.zeros((0, np.asarray(pts_all).shape[-1]))
            if _field_vals is not None:
                gs = _real_dirichlet_values(np.asarray(_field_vals)[nids], region).astype(float)
            elif value_node is not None:
                raw = np.asarray(jnp.asarray(_eval_value_node_at(value_node, jnp.asarray(pts))))
                _real_dirichlet_values(raw, region)  # refuse a complex g before any cast drops Im(g)
                # Does the result scale with the number of points? A CONSTANT profile returns the
                # same thing for any batch -- including a constant VECTOR like (gx, gy), whose size
                # can coincide with the node count -- so shape alone cannot tell. One extra
                # single-point evaluation settles it and costs nothing.
                one = np.asarray(jnp.asarray(_eval_value_node_at(value_node, jnp.asarray(pts[:1]))))
                if raw.shape == one.shape or len(nids) == 0:
                    gs = np.full(len(nids), float(np.real(one).reshape(-1)[0]))  # constant over the region
                else:
                    gs = np.real(raw).reshape(len(nids), -1)[:, 0].astype(float)  # first component per node
            elif callable(value):
                gs = np.array([float(_real_dirichlet_values(value(p), region)) for p in pts], dtype=float)
            else:
                gs = np.full(len(nids), float(_real_dirichlet_values(value, region)))
            comps_range = range(vt) if comp is None else [int(comp)]
            for nid, g in zip(nids, gs):
                for c in comps_range:
                    pairs.append((offs[fidx] + nid * vt + c, _cover_g(fidx, nid, g)))
        # Expose the (dof, value) pairs for callers that compose their own system from native blocks
        # (e.g. the second-order-in-time augmented [u, v] block applies them to the 2N system itself).
        # The time-varying entries ride a companion stash: the caller writes g(x_d, t) (and, for a
        # second-order block, the velocity ġ) per step.
        domain._fem_native_dirichlet_pairs = pairs
        domain._fem_native_dirichlet_tv = tv_stash
        _mask_cover_pins(pairs)
        _gauge_cover_modes(pairs)
        return pairs

    def _cover_g(fidx, nid, g):
        """The Dirichlet value for one padded DOF slot.

        A Dirichlet condition fixes the field VALUE. On a cover field the same mesh node also owns
        its cover coefficients, and those must go to **zero**, not to ``g`` -- otherwise the trace on
        the boundary is ``g + (x - x_i)·g/s`` rather than ``g``. The padded layout repeats each
        node's coordinates, so the region lookup finds every slot; this is where they are told
        apart, by their position within the node's block."""
        blk = _cblk[fidx]
        # NOT float(g): a net-valued condition (`u(top) - net(x)`) arrives as a traced JAX scalar, and
        # casting it raises ConcretizationTypeError -- which took down every `test_dirichlet_net_*`
        # case, plain Lagrange included, since blk == 1 came through here too. Pass the value along
        # and let the pair carry whatever it is; only the COVER slots need forcing, and to zero.
        return g if blk == 1 or (int(nid) % blk) == 0 else 0.0

    _cover_gauge_pins: List[Any] = []
    _cover_gauge_done = [False]

    def _mask_cover_pins(pairs):
        """Pin every cover DOF of a node the enrichment mask excludes.

        This is what makes ``p`` vary across the mesh. It needs no constraint equations and no
        interface bookkeeping: because enrichment rides the partition of unity, an enriched node
        beside an unenriched one already blends correctly, so switching a node off is just fixing
        its coefficients at zero. That is the whole reason interpolation covers are the cheap route
        to variable ``p`` -- a hierarchical basis would need the edge modes reconciled."""
        if _cover_mask is None:
            return
        for fidx, is_cov in enumerate(_cover):
            if not is_cov:
                continue
            n_real = int(np.asarray(pts_f_real[fidx]).shape[0])
            if _cover_mask.size != n_real:
                raise ValueError(
                    f"the enrichment mask has {_cover_mask.size} entries but field {fidx} has {n_real} "
                    f"nodes. The mask is one flag per MESH NODE (domain._fem_enriched_nodes, written by "
                    "jno.solve.enrich); a mismatch means it was built against a different mesh."
                )
            blk, vt = _cblk[fidx], int(vecs[fidx])
            off = set(int(i) for i in np.flatnonzero(~_cover_mask))
            for r in off:
                for m in range(1, blk):
                    base = offs[fidx] + (r * blk + m) * vt
                    for c in range(vt):
                        pairs.append((base + c, 0.0))

    def _gauge_cover_modes(pairs):
        """Remove the enrichment's exact null modes that the boundary conditions leave alive.

        The cover basis satisfies ``Σ h_i (x - x_i) ≡ 0``, so ``a_i = S·x_i + c`` (``S`` skew) is
        the zero FUNCTION -- ``dim(dim+1)/2`` modes per component. A mode dies when any pinned DOF
        carries a nonzero component of it; whatever survives makes the system singular without
        changing the field at all. So the fix is a GAUGE, exactly like a pressure pin: for each
        surviving mode, pin one more cover DOF (chosen by Gaussian pivoting so the set is
        independent) to zero. Every member of the solution family ``x* + Σ α_k v_k`` is the same
        physical field, and the gauge merely selects one -- measured and asserted in the tests, not
        assumed. The pins are recorded once and appended to every later pair collection."""
        if _cover_gauge_done[0]:
            pairs.extend(_cover_gauge_pins)
            return
        _cover_gauge_done[0] = True
        pinned = np.asarray(sorted({int(i) for i, _ in pairs}), dtype=np.int64)
        for fidx, is_cov in enumerate(_cover):
            if not is_cov:
                continue
            modes = cover_null_modes(pts_f_real[fidx], dim, n_comp=int(vecs[fidx])).astype(float)
            lo, hi = offs[fidx], offs[fidx + 1]
            loc_pin = pinned[(pinned >= lo) & (pinned < hi)] - lo
            # survivors: modes invisible to every existing pin
            alive = modes[np.abs(modes[:, loc_pin]).max(axis=1) < 1e-12] if loc_pin.size else modes
            if alive.shape[0] == 0:
                continue
            forbidden = set(int(x) for x in loc_pin)
            work = alive.copy()
            for _k in range(work.shape[0]):
                row = work[_k]
                cand = np.abs(row)
                if forbidden:
                    cand = cand.copy()
                    cand[list(forbidden)] = 0.0
                j = int(np.argmax(cand))
                if cand[j] < 1e-12:
                    raise ValueError(
                        "space='cover': could not complete the null-space gauge -- a surviving zero "
                        "mode has no free cover DOF left to pin. This should be unreachable; please "
                        "report the mesh and boundary conditions."
                    )
                _cover_gauge_pins.append((int(lo + j), 0.0))
                forbidden.add(j)
                for _l in range(_k + 1, work.shape[0]):  # eliminate so later picks stay independent
                    work[_l] = work[_l] - (work[_l][j] / row[j]) * row
        pairs.extend(_cover_gauge_pins)

    _static_dirichlet_cache: Dict[str, Any] = {}

    def _dirichlet_pairs_at(args):
        """Dirichlet ``(dof, value)`` pairs with the net-valued profiles evaluated from the runtime
        ``args`` (an unknown BC ``u(region) - net(x)``): the net is called on the region's boundary-node
        coordinates, so the value stays a differentiable JAX scalar and ``∂b/∂weights`` flows through the
        symmetric elimination. Non-net conditions reuse the concrete ``_build_dirichlet_pairs`` values."""
        a = args or {}
        # The concrete (non-net, non-parametric) pairs are ARG-INDEPENDENT, and rebuilding them here
        # is not just wasted work: this runs inside the traced Newton residual (`_np_hold(args)`),
        # where the host-side tag resolution in `_build_dirichlet_pairs` can meet traced state a
        # contact assembly stashed (measured: TracerArrayConversionError from the tag location fn on
        # a gap-carrying form). Built ONCE, on the first call -- which is always the eager dof-layout
        # probe `_dirichlet_pairs_at(_dir_static_args())`, outside any trace.
        if "pairs" not in _static_dirichlet_cache:
            _static_dirichlet_cache["pairs"] = _build_dirichlet_pairs()
        pairs = list(_static_dirichlet_cache["pairs"])
        for _row_i, (field_key, region, comp, value, value_node) in enumerate(dirichlet_raw):
            fidx = field_index.get(field_key)
            if fidx is None:
                continue
            _vn = _bare_node(value_node) if value_node is not None else None
            if _row_i in _dir_param_rows:
                # A trainable parameter in the value (`u(top) - g`, `u(top) - g*profile(x)`): evaluate
                # the value NODE at the boundary nodes with the args-substituted parameter, so the held
                # value stays a traced JAX scalar and ∂b/∂g (steady) / ∂step/∂g (transient) flows --
                # the same contract as the net branch below. `_eval_value_node_at(params=...)` is the
                # existing parametric-coefficient evaluator; no bespoke walker.
                #
                # The node LAYOUT is static per (field, region) and memoized from the first (eager)
                # call: this body re-runs inside the traced Newton residual, where the host tag
                # resolution behind `_boundary_node_ids` can meet traced state (measured on a
                # gap-carrying form: TracerArrayConversionError out of the tag location fn).
                from ..._fem import _eval_value_node_at as _eval_at

                vt = vecs[fidx]
                _lk = ("layout", fidx, region)
                if _lk not in _static_dirichlet_cache:
                    pts_all = np.asarray(pts_f_all[fidx])
                    nid_list = list(_boundary_node_ids(fidx, region))
                    _static_dirichlet_cache[_lk] = (
                        nid_list,
                        pts_all[np.asarray(nid_list, dtype=np.int64)] if nid_list else np.zeros((0, dim)),
                    )
                node_ids, pts = _static_dirichlet_cache[_lk]
                raw = jnp.asarray(_eval_at(value_node, jnp.asarray(pts), params=a)).reshape(-1)
                # Constant vs spatial profile: a constant returns the same shape for ANY batch (the
                # static-path trick) -- one extra single-point evaluation decides, shapes are static.
                one = jnp.asarray(_eval_at(value_node, jnp.asarray(pts[:1]), params=a)).reshape(-1)
                if raw.shape == one.shape or not node_ids:
                    gvals = jnp.broadcast_to(raw.reshape(-1)[:1], (len(node_ids),))
                else:
                    gvals = raw.reshape(len(node_ids), -1)[:, 0]
                for i, nid in enumerate(node_ids):
                    for c in range(vt) if comp is None else [int(comp)]:
                        pairs.append((offs[fidx] + nid * vt + c, _cover_g(fidx, nid, gvals[i])))
                continue
            if _vn is None or not _is_neural_coefficient(_vn):
                continue
            vt = vecs[fidx]
            pts_all = np.asarray(pts_f_all[fidx])
            node_ids = _boundary_node_ids(fidx, region)
            module = a.get(_neural_coefficient_name(_vn), _vn.model.module)
            coords = jnp.asarray(pts_all[np.asarray(node_ids, dtype=np.int64)])  # (n_bnodes, dim)
            n_in = len(_vn.args)  # net(xb, yb[, zb]) -> per-coordinate columns (foundax MLP arity)
            gvals = jnp.asarray(module(*[coords[:, i : i + 1] for i in range(n_in)])).reshape(-1)
            for i, nid in enumerate(node_ids):
                for c in range(vt) if comp is None else [int(comp)]:
                    pairs.append((offs[fidx] + nid * vt + c, _cover_g(fidx, nid, gvals[i])))
        return pairs

    def _build_dirichlet_tv_entries():
        """Time-varying Dirichlet ``g(x, t)`` entries: a list of ``(dofs, value_node, coords)`` for each
        ``dirichlet_raw`` whose value carries the temporal variable. The transient block evaluates
        ``g`` at ``coords`` and time ``t`` each step (``_eval_value_node_at_time``) and writes it onto
        ``dofs`` in the forcing. The constant-valued conditions are returned separately as ordinary
        pairs (their ``t``-independent value goes in the affine bias)."""
        from ..._fem import _eval_value_node_at, _is_temporal_value_node

        const_pairs: List[Tuple[int, float]] = []
        tv_entries: List[Tuple[Any, Any, Any]] = []
        for field_key, region, comp, value, value_node in dirichlet_raw:
            fidx = field_index.get(field_key)
            if fidx is None:
                continue
            vt = vecs[fidx]
            pts_all = np.asarray(pts_f_all[fidx])
            nids = _boundary_node_ids(fidx, region)
            comps_range = range(vt) if comp is None else [int(comp)]
            if value_node is not None and _is_temporal_value_node(value_node):
                coords = jnp.asarray(pts_all[np.asarray(nids, dtype=int)]) if nids else jnp.zeros((0, dim))
                for c in comps_range:
                    dofs = jnp.asarray([offs[fidx] + nid * vt + c for nid in nids], dtype=jnp.int32)
                    tv_entries.append((dofs, value_node, coords))
                continue
            for nid in nids:
                p = pts_all[nid]
                if value_node is not None:
                    g = float(jnp.asarray(_eval_value_node_at(value_node, jnp.asarray(p)[None])).reshape(-1)[0])
                elif callable(value):
                    g = float(value(p))
                else:
                    g = float(value)
                for c in comps_range:
                    const_pairs.append((offs[fidx] + nid * vt + c, _cover_g(fidx, nid, g)))
        return const_pairs, tv_entries

    # -------------------------------------------------------------------------
    # Mode detection
    # -------------------------------------------------------------------------

    all_terms = list(volume_terms) + [t for ts in boundary_terms.values() for t in ts]
    zeros = jnp.zeros(total)

    # === transient (Mu̇ + Au = c or M u̇ + R(u) = 0) ===
    if ic_residuals or any(_contains_temporal_derivative(t) for t in all_terms):
        from ..._fem import _bare, _essential_spec, _eval_value_node_at, _field_key_of
        from .backend_blocks import SemidiscreteTimeBlock
        from .time_route import (
            _infer_time_window,
            _replace_temporal_with_backward_euler,
            _strip_temporal_trial_derivative,
        )

        sub_signed = [
            _apply_sign(domain, sign, sub) for bare in volume_terms for sign, sub in _split_additive_terms(domain, bare)
        ]
        temporal = [t for t in sub_signed if _contains_temporal_derivative(t)]
        spatial = [t for t in sub_signed if not _contains_temporal_derivative(t)]
        if not temporal:
            raise ValueError(
                "jno.fem (native): an initial condition was provided but no temporal term "
                "(e.g. ``inner(u.t, v)``) was found in the volume weak form."
            )
        # A trainable net on the u̇ term. A COORDINATE ``net(x)`` (an unknown density ``rho(x)*u_t``) keeps
        # the mass a *matrix* -- just parametric in the weights -- so it is re-assembled from ``args`` each
        # step (``mass_fn`` below). A SOLUTION-DEPENDENT ``net(u)`` would make the mass itself nonlinear
        # (``C(u)*u_t``), which the semidiscrete matrix form cannot express -- reject that.
        _parametric_mass = collect_neural_slots(temporal).any_trainable
        if _parametric_mass and any(_is_obviously_nonlinear_in_unknown(domain, t) for t in temporal):
            raise NotImplementedError(
                "jno.fem (native): a solution-dependent neural coefficient net(u) on the mass (u_t) term is a "
                "nonlinear mass C(u)*u_t, which the semidiscrete matrix form cannot express. A coordinate "
                "net(x) mass coefficient (an unknown density) is supported."
            )

        mass_terms = [_strip_temporal_trial_derivative(t) for t in temporal]
        # Mass matrix: volume only (no boundary); spatial residual: volume + boundary
        _mass_jac = _make_jacobian(mass_terms)
        M = _mass_jac(zeros)
        spatial_res = _make_residual(spatial, boundary_terms)
        spatial_jac = _make_jacobian(spatial, boundary_terms)

        t0, t1, dt = _infer_time_window(domain)
        common = dict(
            backend="transient",
            mode="implicit",
            time_order=1,
            spatial_kind="weak_form",
            state0=None,
            t0=t0,
            t1=t1,
            dt=dt,
            eval_context=getattr(domain, "_fem_eval_context", {}) or {},
        )

        # --- initial state: nodal interpolation (exact for Lagrange). ``params`` re-forms a net-valued IC
        # ``u(initial) - net(x)`` from the runtime weights; ``None`` (no IC net) is byte-identical to the
        # old eager build. When an IC net is present the closure also rides the block as ``state0_fn`` so
        # ``∂traj/∂weights`` flows through the initial state. ---
        def _cover_state0(s0, fidx, comp, u0_node, params):
            """Interpolate the IC INTO THE ENRICHED SPACE rather than onto the node values alone.

            A cover node carries its value ``u_i`` AND coefficients ``a_i`` multiplying
            ``(x - x_i)/s_i``, and the element's own identity -- the one its patch test asserts -- is
            ``a_i = 1/2 grad g(x_i) * s_i``. The plain nodal path below writes ``g(x_i)`` into every
            slot of the node, which hands the cover coefficients a function value where a scaled
            GRADIENT belongs. That is not a small inconsistency: measured on the heat benchmark it
            starts the march at L2 1.9e-01 where this interpolation reaches 5.2e-03, and worse than
            simply leaving the covers at zero (8.5e-02). It is unrecoverable rather than merely
            inaccurate, because the enriched DOFs are stiff -- the generalized spectrum splits into 2
            value-dominated modes (lambda 2.0-5.0) and 55 cover-dominated ones (up to 184.6), so
            backward Euler damps the bad start by 1/(1 + dt*lambda) and annihilates it in a single
            step (measured: the cover coefficients collapse to 0.17 of their initial size while the
            values correctly go to 0.8516 against an exact 0.8624).

            The gradient comes from a JVP, not a symbolic derivative: the IC value is evaluated
            POINTWISE, so a tangent moving every node along ``e_k`` returns ``dg/dx_k`` at every node
            in one pass -- and it keeps the initial state differentiable in a net-valued IC's weights,
            which is the contract ``state0_fn`` exists to honour.
            """
            blk, vv = _cblk[fidx], int(vecs[fidx])
            xr = jnp.asarray(pts_f_real[fidx])[:, :dim]
            n_real = int(xr.shape[0])

            def _at(P):
                return jnp.reshape(jnp.asarray(_eval_value_node_at(u0_node, P, params=params)), (-1,))

            def _as_nodal(v):
                if v.size == 1:
                    return jnp.full((n_real, vv), v.reshape(-1)[0])
                if v.size == vv:
                    return jnp.broadcast_to(v.reshape(1, vv), (n_real, vv))
                return v.reshape(n_real, vv)

            scale = jnp.asarray(_cover_scale_j).reshape(-1)[:n_real, None]
            slots = [_as_nodal(_at(xr))]
            for k in range(dim):
                tangent = jnp.zeros_like(xr).at[:, k].set(1.0)
                slots.append(0.5 * _as_nodal(jax.jvp(_at, (xr,), (tangent,))[1]) * scale)
            block = jnp.stack(slots, axis=1)  # (n_real, blk, vv) -- node-major, slot, component
            if comp is None:
                return s0.at[offs[fidx] : offs[fidx + 1]].set(block.reshape(-1))
            idx = offs[fidx] + (jnp.arange(n_real)[:, None] * blk + jnp.arange(blk)[None, :]) * vv + int(comp)
            return s0.at[idx.reshape(-1)].set(block[:, :, int(comp)].reshape(-1))

        def _state0_at(params=None):
            s0 = zeros
            for ic in ic_residuals:
                comp, u0_node = _essential_spec(_bare(ic))
                fidx = field_index.get(_field_key_of(ic))
                if fidx is None:
                    raise ValueError("jno.fem (native): IC does not match any known trial field.")
                if _cover[fidx]:
                    # An enriched field's slots are not all values -- see `_cover_state0`.
                    s0 = _cover_state0(s0, fidx, comp, u0_node, params)
                    continue
                pts_ic = pts_f_all[fidx]  # (n_nodes_f[fidx], 2)
                nn, vv = n_nodes_f[fidx], vecs[fidx]
                raw = jnp.reshape(jnp.asarray(_eval_value_node_at(u0_node, jnp.asarray(pts_ic), params=params)), (-1,))
                if comp is not None:
                    # Per-component IC (e.g. ``u(initial)[0] - g0``): set just component ``comp`` at every
                    # node of the field. ``raw`` is the per-node value (or a single constant to broadcast).
                    vals = jnp.broadcast_to(raw, (nn,)) if raw.size == 1 else raw.reshape(nn)
                    idx = offs[fidx] + jnp.arange(nn) * vv + int(comp)
                    s0 = s0.at[idx].set(vals)
                else:
                    # Whole-field IC. A constant evaluates to a single value (no coordinate Variables to
                    # sample) -> broadcast to every node; a per-component constant broadcasts across nodes;
                    # otherwise it is the per-node field already.
                    if raw.size == 1:
                        u0_vals = jnp.full((nn, vv), raw[0])
                    elif raw.size == vv:
                        u0_vals = jnp.broadcast_to(raw[None, :], (nn, vv))
                    else:
                        u0_vals = raw.reshape(nn, vv)
                    s0 = s0.at[offs[fidx] : offs[fidx + 1]].set(u0_vals.reshape(-1))
            return s0

        common["state0"] = _state0_at({n: m.module for n, m in _ic_net_models.items()} if _ic_net_models else None)
        if _ic_net_models:
            common["state0_fn"] = _state0_at

        dirichlet_pairs = _build_dirichlet_pairs()
        d_dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32) if dirichlet_pairs else None
        d_vals = jnp.asarray([p[1] for p in dirichlet_pairs], dtype=zeros.dtype) if dirichlet_pairs else None

        # ---- STATE-DEPENDENT (nonlinear) MASS: ``c(u)·u_t`` with a coefficient depending on the unknown.
        # The fixed ``M = _mass_jac(zeros)`` freezes ``c`` at ``u=0`` (silently wrong; see jno-fem-hard-limits).
        # Reformulate each temporal term to backward-Euler *residual* form ``c(u)·(u − u_prev)·v`` — with
        # ``u_prev`` the previous step's nodal values delivered per step on the load-path channel — so the
        # ordinary residual/Jacobian assembly captures the exact mass action ``M(u)(u−u_prev)`` AND its exact
        # ``∂/∂u`` (both the ``M`` block and the ``∫c′(u)(u−u_prev)·v`` coefficient coupling). The ``1/dt``
        # factor is applied by the stepper. Backward Euler (θ=1) only; scalar fields only (load-path is scalar).
        _nonlinear_mass = (not _parametric_mass) and any(
            _is_obviously_nonlinear_in_unknown(domain, mt) for mt in mass_terms
        )
        prev_state_slices: List[Tuple[int, int, int, int]] = []  # (frozen_id, dof_start, dof_stop, n_components)
        mass_res_bc = mass_jac_bc = None
        if _nonlinear_mass:
            from ...trace import PrevStateField as _PrevStateField

            _prev_by_field: Dict[Any, Any] = {}

            def _prev_for(trial, _cache=_prev_by_field):
                fkey = trial.field_key
                pf = _cache.get(fkey)
                if pf is None:
                    fidx = field_index[fkey]
                    pf = _PrevStateField(trial)
                    _cache[fkey] = pf
                    # The prev-state field carries the source field's OWN key/basis, so it resolves the field's
                    # own shape data (P1 or P2, scalar or vector) — no P1 aliasing (unlike a load-path field).
                    _path_conn[pf.frozen_id] = cells_f_j[fidx]  # the field's own vertex connectivity
                    # (frozen_id, dof-slice into the flat state, n_components) — the step delivers this slice
                    # reshaped to (n_nodes, vec) on the load-path channel each backward-Euler step.
                    prev_state_slices.append((int(pf.frozen_id), int(offs[fidx]), int(offs[fidx + 1]), int(vecs[fidx])))
                return pf

            temporal_be = [_replace_temporal_with_backward_euler(t, _prev_for) for t in temporal]
            _mass_res_raw = _make_residual(temporal_be)  # ∫ c(u)·(u − u_prev)·v  (volume only; mass has no boundary)
            _mass_jac_raw = _make_jacobian(temporal_be)

            def mass_res_bc(u, t, args=None, _d=d_dofs, _f=_mass_res_raw):
                R = jnp.asarray(_f(jnp.asarray(u), t, args)).reshape(-1)
                return R if _d is None else R.at[_d].set(0.0)  # a constrained DOF carries no mass equation

            def mass_jac_bc(u, t, args=None, _d=d_dofs, _f=_mass_jac_raw):
                J = _f(jnp.asarray(u), t, args)
                return J if _d is None else bcoo_zero_rows(J, _d)

        # Parametric mass ``mass_fn(t, args)`` (unknown density net(x)*u_t): re-assemble M from args each
        # step with the Dirichlet rows/cols zeroed (a constrained DOF carries no time derivative). ``None``
        # keeps the static ``M_bc`` for a non-parametric mass.
        def _mass_cb(t, args=None, _d=d_dofs):
            Mt = _mass_jac(zeros, t, args)
            return Mt if _d is None else bcoo_zero_rows_cols(Mt, _d)

        # A mass-only nonlinearity (state-dependent mass) also requires the nonlinear step path, even when
        # every spatial term is linear — the mass action lives in the residual there (``mass_residual``).
        nonlinear = _nonlinear_mass or any(_is_obviously_nonlinear_in_unknown(domain, t) for t in spatial)
        if nonlinear:
            if _ic_net_models:
                raise NotImplementedError(
                    "jno.fem: a net-valued initial condition u(initial) - net(x) on a *nonlinear transient* "
                    "form is not wired yet (state0_fn threads only the linear stepper). Use a linear "
                    "transient form (a net IC threads there)."
                )
            if _dir_args_dependent and _nonlinear_mass:
                raise NotImplementedError(
                    "jno.fem: a net-valued Dirichlet with a state-dependent (nonlinear) mass c(u)·u_t on a "
                    "transient form is not supported (the mass residual holds a static Dirichlet dof set). "
                    "Use a linear/parametric mass."
                )

            if _dir_args_dependent:
                # net- or parameter-valued Dirichlet: the held value is re-formed from the args each
                # Newton residual (mirrors the nonlinear STEADY path ``res_p``); the dof set is static, only
                # the held values ride the args, and ``∂/∂args`` flows through the step's custom_root.
                _tnpd = jnp.asarray(
                    [p[0] for p in _dirichlet_pairs_at(_dir_static_args())],
                    dtype=jnp.int32,
                )

                def _tnp_hold(args):
                    return jnp.stack([jnp.asarray(p[1]).reshape(()) for p in _dirichlet_pairs_at(args)])

                def res_bc(u, t, args=None, _d=_tnpd):
                    R = spatial_res(jnp.asarray(u), t, args)
                    return R.at[_d].set(jnp.asarray(u)[_d] - _tnp_hold(args))

                def jac_bc(u, t, args=None, _d=_tnpd):
                    return bcoo_set_dirichlet_rows(spatial_jac(jnp.asarray(u), t, args), _d)

                _mdofs = _tnpd
            else:
                # Row-replacement Dirichlet (constant g), threaded through the runtime time t AND the
                # runtime args so a time-dependent / parametric spatial coefficient is re-evaluated each step.
                def res_bc(u, t, args=None, _d=d_dofs, _g=d_vals):
                    R = spatial_res(jnp.asarray(u), t, args)
                    return R if _d is None else R.at[_d].set(jnp.asarray(u)[_d] - _g)

                def jac_bc(u, t, args=None, _d=d_dofs):
                    J = spatial_jac(jnp.asarray(u), t, args)
                    return J if _d is None else bcoo_set_dirichlet_rows(J, _d)

                _mdofs = d_dofs

            M_bc = M if _mdofs is None else bcoo_zero_rows_cols(M, _mdofs)
            return (
                SemidiscreteTimeBlock(
                    # A state-dependent mass carries no fixed matrix; the mass action is in mass_residual.
                    mass=None
                    if _nonlinear_mass
                    else (_mass_cb if (_parametric_mass or _coord_specs) else (lambda t, args=None, _M=M_bc: _M)),
                    mass_residual=mass_res_bc,
                    mass_residual_jac=mass_jac_bc,
                    residual=res_bc,
                    jacobian=jac_bc,
                    runtime_parameter_exprs=dict(_param_and_neural_exprs),
                    metadata={"prev_state_slices": prev_state_slices} if _nonlinear_mass else {},
                    **common,
                ),
                "transient",
                offs,
            )

        # ---- linear parametric transient: the operator A(t, args) is re-evaluated each step.
        # Row-replacement Dirichlet (rows -> identity, columns kept) needs no args-dependent lift for a
        # CONSTANT g -- the held value sits in the affine bias. A net-valued Dirichlet u(∂Ω)-net(x) has an
        # args-dependent held value (differentiable in the weights): its whole held vector rides the
        # forcing each step instead (mirrors the g(x,t) path), so the constant bias drops to zero. ----
        # ``_coord_specs`` belongs here for the same reason it does in the steady gate below: a trainable
        # mesh coordinate makes the operator (and the mass) a function of ``args``. Without it a transient
        # falls to the static branch, where A and M are built once with ``args=None`` and
        # ``_apply_coord_params`` short-circuits -- so ``block.step(..., args={coord: X})`` silently ignores
        # X and ``du/dX`` is exactly ZERO. That is a wrong gradient with no symptom, not a missing feature.
        if runtime_parameter_tags or neural_param_names or _dir_args_dependent or _ic_net_models or _coord_specs:
            if _dir_args_dependent:
                if getattr(domain, "_fem_native_dirichlet_tv", None):
                    raise NotImplementedError(
                        "jno.fem: a net-valued Dirichlet combined with a time-varying g(x, t) Dirichlet on a "
                        "transient form is not supported yet (the net value rides the forcing; the g(x, t) lift "
                        "needs the temporal evaluator on those same rows). Use one or the other."
                    )
                # const + net Dirichlet dofs (static boundary-node layout); held values re-formed from args.
                _dd = jnp.asarray(
                    [p[0] for p in _dirichlet_pairs_at(_dir_static_args())],
                    dtype=jnp.int32,
                )

                def _dhold(args):  # held value on every Dirichlet dof (net entries live in the weights)
                    return jnp.stack([jnp.asarray(p[1]).reshape(()) for p in _dirichlet_pairs_at(args)])
            else:
                _dd = d_dofs
            M_bc = M if _dd is None else bcoo_zero_rows_cols(M, _dd)
            free_mask = jnp.ones((total,), dtype=zeros.dtype)
            if _dd is not None:
                free_mask = free_mask.at[_dd].set(0.0)

            def operator_fn(t, args=None, _d=_dd):
                A = spatial_jac(zeros, t, args)
                return A if _d is None else bcoo_set_dirichlet_rows(A, _d)

            if _dir_args_dependent:
                c_bias = zeros  # every held value (const + net + parameter) rides the forcing

                def forcing_vector_fn(t, args=None, _mask=free_mask, _d=_dd):
                    f = _mask * (-spatial_res(zeros, t, args))
                    return f.at[_d].set(_dhold(args))
            else:
                c_bias = zeros if d_dofs is None else zeros.at[d_dofs].set(d_vals)

                def forcing_vector_fn(t, args=None, _mask=free_mask):
                    return _mask * (-spatial_res(zeros, t, args))

            return (
                SemidiscreteTimeBlock(
                    M=M_bc,
                    # Parametric mass for an unknown density net(x)*u_t -- and for a trainable mesh
                    # coordinate, because the mass is ∫φᵢφⱼ dx ∝ |K|: hold it static while the mesh moves
                    # and the u̇ term keeps the volumes of the mesh you started from.
                    mass_fn=_mass_cb if (_parametric_mass or _coord_specs) else None,
                    operator_fn=operator_fn,
                    affine_bias=c_bias,
                    forcing_vector_fn=forcing_vector_fn,
                    runtime_parameter_exprs=dict(_param_and_neural_exprs),
                    # The operator is re-assembled at each (t, args) -- a general (non-affine) operator,
                    # so it covers a parameter inside a nonlinear coefficient (e.g. exp(logk)) too.
                    metadata={
                        "runtime_parameter_names": list(runtime_parameter_tags)
                        + list(neural_param_names)
                        + list(_dir_net_models)
                        + list(_ic_net_models),
                        "nonaffine_operator": True,
                    },
                    **common,
                ),
                "transient",
                offs,
            )

        # ---- time-varying Dirichlet g(x, t) (linear, non-parametric): row-replacement Dirichlet whose
        # held value is supplied by the forcing each step. Constant conditions go to the affine bias;
        # the constrained dofs carry no time derivative (their mass row is zeroed) and the held value
        # u[d] = g(x_d, t) is written into forcing_vector_fn(t) (the per-step Dirichlet lift). ----
        _const_pairs, _tv_entries = _build_dirichlet_tv_entries()
        if _tv_entries:
            from ..._fem import _eval_value_node_at_time

            _cd = jnp.asarray([p[0] for p in _const_pairs], dtype=jnp.int32) if _const_pairs else jnp.zeros((0,), jnp.int32)
            _cv = jnp.asarray([p[1] for p in _const_pairs], dtype=zeros.dtype) if _const_pairs else zeros[:0]
            _tvd = jnp.concatenate([e[0] for e in _tv_entries])
            _all_d = jnp.concatenate([_cd, _tvd])
            # Compress AFTER the Dirichlet edit: bcoo_set_unit_diag appends its own (d, d, 1)
            # triplets, which must merge with whatever the assembly already put on that diagonal.
            A_tv = compress_eager(bcoo_set_dirichlet_rows(spatial_jac(zeros, 0.0), _all_d))
            M_tv = compress_eager(bcoo_zero_rows(M, _all_d))
            c_tv = zeros.at[_cd].set(_cv)
            free_tv = jnp.ones((total,), dtype=zeros.dtype).at[_all_d].set(0.0)

            def forcing_vector_fn(t, args=None, _mask=free_tv, _tv=_tv_entries):
                f = _mask * (-spatial_res(zeros, t))  # source load on the free rows
                for dofs, vnode, coords in _tv:
                    f = f.at[dofs].set(jnp.asarray(_eval_value_node_at_time(vnode, coords, t)).reshape(-1))
                return f

            return (
                SemidiscreteTimeBlock(M=M_tv, A=A_tv, affine_bias=c_tv, forcing_vector_fn=forcing_vector_fn, **common),
                "transient",
                offs,
            )

        # linear transient.  The operator A is assembled at t=0 (autonomous operator); a
        # time-dependent SOURCE is carried by forcing_vector_fn(t).  The constant Dirichlet lift +
        # the t=0 load go into the affine bias via symmetric elimination; forcing_vector_fn supplies
        # only the time-varying increment on the free rows (Dirichlet rows handled by the bias).
        A = spatial_jac(zeros, 0.0)
        c0 = -spatial_res(zeros, 0.0)
        M, A, c = _apply_dirichlet_transient(M, A, c0, dirichlet_pairs)
        # Both operators are applied on EVERY timestep, so the ~19x triplet redundancy is paid once
        # per step for the whole march. Compressing here is the single highest-leverage site.
        M, A = compress_eager(M), compress_eager(A)
        if temporal_tags:
            free_mask = jnp.ones((total,), dtype=zeros.dtype)
            if d_dofs is not None:
                free_mask = free_mask.at[d_dofs].set(0.0)

            def forcing_vector_fn(t, args=None, _c0=c0, _mask=free_mask):
                return _mask * (-spatial_res(zeros, t) - _c0)

            return (
                SemidiscreteTimeBlock(M=M, A=A, affine_bias=c, forcing_vector_fn=forcing_vector_fn, **common),
                "transient",
                offs,
            )
        return SemidiscreteTimeBlock(M=M, A=A, affine_bias=c, **common), "transient", offs

    # === steady ===
    dirichlet_pairs = _build_dirichlet_pairs()
    # τ-dependent essential values (`u(top) - delta*tau`): collected as (dofs, node, coords) rather than
    # constant pairs, because their held value changes every load step. The march threads them below.
    _tv_dirichlet = list(getattr(domain, "_fem_native_dirichlet_tv", []) or [])
    residual = _make_residual(volume_terms, boundary_terms)
    # Publish the FREE (pre-Dirichlet) residual factory so `FEM.eval` can assemble an arbitrary weak
    # term at a solution. Every solve path elimination-mutates its own copy -- symmetric elimination for
    # the linear system, row replacement for Newton -- which zeroes exactly the rows a reaction/flux
    # readout needs. Snapshotted onto the FEM in `_finalize`, like the field keys and DOF points.
    domain._fem_native_term_residual = _make_residual
    # Whether the FACET tables exist. They are tabulated only when the FORM carries a surface term
    # (see `face_tables_per_field`), so a later `fem.eval` of a surface term on a problem with no
    # boundary terms has nothing to integrate against -- it must say so rather than fail deep inside
    # the element kernel on `NoneType` unpacking.
    domain._fem_native_has_facet_tables = bool(boundary_terms)
    jacobian = _make_jacobian(volume_terms, boundary_terms)
    nonlinear = any(_is_obviously_nonlinear_in_unknown(domain, t) for t in all_terms)
    # A form that is LINEAR in the unknown but READS step history is still a march: every load step is a
    # different linear system whose coefficients the buffers set. The linear branch below builds a
    # ``FemLinearSystem`` from ``_assemble_at(args)``, which has no ``__history__`` to thread — so the
    # history read raised at BUILD time, from deep inside the integrand evaluator. Route it through the
    # residual operator instead: Newton on a linear residual converges in one step, so the answer is the
    # same linear solve and there is exactly one march path to maintain. The AT1 phase-field damage
    # equation with a lagged driving force is precisely this shape, so it is on the critical path.
    if history_specs or surface_history_specs:
        nonlinear = True
    # A BOX CONSTRAINT (`u.bounds(lo, hi)`) makes the problem a variational inequality even when the
    # operator is linear -- the obstacle problem is a linear operator whose answer is decided by the free
    # boundary. Its KKT conditions are the root of a min-map residual, so it needs the residual path, not
    # a matrix/rhs pair. Same reason as history above, different cause.
    if bounded:
        nonlinear = True
    s_d_dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32) if dirichlet_pairs else None
    s_d_vals = jnp.asarray([p[1] for p in dirichlet_pairs], dtype=zeros.dtype) if dirichlet_pairs else None

    # ---- runtime-parametric (inverse): the operator/residual is re-evaluated at the runtime args
    # each call, kept differentiable in args -- the parameter flows as a JAX array through the kernel
    # coefficient into the per-cell assembly (no float() cast). The same re-assembly handles affine,
    # non-affine and (scalar) parameters uniformly. ----
    if (
        runtime_parameter_tags
        or neural_param_names
        or _dir_args_dependent
        or history_specs
        or surface_history_specs
        or _coord_specs
    ):
        from ...trace import FemLinearSystem

        if nonlinear:
            # ``t`` carries the pseudo-time (load) coordinate τ for the history march — the load written
            # as a function of τ in the weak form varies through it. Defaults to 0.0, so the ordinary
            # (non-marching) parametric/inverse call sites are unchanged.
            if _dir_args_dependent:
                # net- or parameter-valued Dirichlet: the held value is a differentiable function of the
                # args (net weights or a trainable boundary value), so the row-replacement value is
                # re-evaluated from args each residual call (mirrors the linear parametric path's
                # ``_dirichlet_pairs_at``). The dof set is static; only the held values ride the args.
                _npd = jnp.asarray(
                    [p[0] for p in _dirichlet_pairs_at(_dir_static_args())],
                    dtype=jnp.int32,
                )

                def _np_hold(args):  # held value on every Dirichlet dof (const + net), net entries live
                    return jnp.stack([jnp.asarray(p[1]).reshape(()) for p in _dirichlet_pairs_at(args)])

                def _np_project(u, args, _d=_npd):
                    return jnp.asarray(u).at[_d].set(_np_hold(args))

                def res_p(u, args=None, t=0.0, _d=_npd):
                    u = jnp.asarray(u)
                    R = residual(_np_project(u, args), t, args)
                    return R.at[_d].set(u[_d] - _np_hold(args))

                def jac_p(u, args=None, t=0.0, _d=_npd):
                    return bcoo_eliminate_dirichlet(jacobian(_np_project(u, args), t, args), _d)

                _constrained = _npd  # the dof set is static here; only the HELD VALUES ride the weights
            elif _tv_dirichlet:
                # A τ-DEPENDENT essential value on the load path -- `u(top)[1] - delta*tau`, i.e.
                # DISPLACEMENT CONTROL, which is how a softening test is driven at all (under load
                # control the specimen snaps at the peak and there is no branch to follow). The value is
                # not a constant pair, so it is re-evaluated at this step's τ and written into the same
                # row-replacement the constant pairs use. Before this it was collected and then dropped:
                # the constraint simply vanished and the solve returned u = 0, which looks entirely
                # plausible. The dof set is static, so only the held VALUES ride τ.
                from ..._fem import _eval_value_node_at_time

                _tvd = jnp.concatenate([d for d, _n, _c in _tv_dirichlet])
                _all_d = _tvd if s_d_dofs is None else jnp.concatenate([s_d_dofs, _tvd])

                def _tv_hold(t):
                    return jnp.concatenate(
                        [jnp.reshape(jnp.asarray(_eval_value_node_at_time(n, c, t)), (-1,)) for _d, n, c in _tv_dirichlet]
                    )

                def _tv_project(u, t, _d=s_d_dofs, _g=s_d_vals, _t=_tvd):
                    u = jnp.asarray(u)
                    if _d is not None:
                        u = u.at[_d].set(_g.astype(u.dtype))
                    return u.at[_t].set(_tv_hold(t).astype(u.dtype))

                def res_p(u, args=None, t=0.0, _d=s_d_dofs, _g=s_d_vals, _t=_tvd):
                    u = jnp.asarray(u)
                    R = residual(_tv_project(u, t), t, args)
                    if _d is not None:
                        R = R.at[_d].set(u[_d] - _g)
                    return R.at[_t].set(u[_t] - _tv_hold(t))

                def jac_p(u, args=None, t=0.0, _d=_all_d):
                    return bcoo_eliminate_dirichlet(jacobian(_tv_project(u, t), t, args), _d)

                _constrained = _all_d
            else:

                def _s_project(u, _d=s_d_dofs, _g=s_d_vals):
                    u = jnp.asarray(u)
                    return u if _d is None else u.at[_d].set(_g.astype(u.dtype))

                def res_p(u, args=None, t=0.0, _d=s_d_dofs, _g=s_d_vals):
                    u = jnp.asarray(u)
                    R = residual(_s_project(u), t, args)
                    return R if _d is None else R.at[_d].set(u[_d] - _g)

                def jac_p(u, args=None, t=0.0, _d=s_d_dofs):
                    J = jacobian(_s_project(u), t, args)
                    return J if _d is None else bcoo_eliminate_dirichlet(J, _d)

                _constrained = s_d_dofs

            _op = FemResidualOperator(res_p, jac_p, total, runtime_parameter_exprs=dict(_param_and_neural_exprs))
            # Which DOFs carry an ESSENTIAL condition rather than an equation. A solver that extrapolates
            # (``staggered(over_relax>1)``) must leave these alone: the sub-solve already puts them exactly
            # on the prescribed value, and stepping past it makes the constraint oscillate as (1-omega)^k
            # while every other field is solved against the wrong boundary value.
            _op.dirichlet_dofs = _constrained
            _op.history_specs = history_specs  # VOLUME step-history buffer layout for the load-step driver
            _op.surface_history_specs = surface_history_specs  # SURFACE (per-face) step-history layout
            _op.history_roles = history_roles  # {key: "primary" | "internal"} — how each state advances
            _op.state_readout = state_readout  # (u, t, args) -> {key: next per-QP VOLUME state}; march driver
            _op.surface_state_readout = surface_state_readout  # (u, t, args) -> {key: next per-FACE state}
            _op.path_specs = path_specs  # {fid: {frames (n_steps, n_nodes), ...}} — per-step load-path fields
            return (_op, "nonlinear", offs)

        def _assemble_at(args):
            A = jacobian(zeros, 0.0, args)
            b = -residual(zeros, 0.0, args)
            # a net- or parameter-valued Dirichlet re-forms the lift from args each call; else static.
            pairs = _dirichlet_pairs_at(args) if (_dir_net_models or _dir_param_rows) else dirichlet_pairs
            if pairs:
                A, b = _apply_dirichlet_symmetric(A, b, pairs)
            return A, b

        # Static placeholder for .A/.b: scalar params at 0, networks (coefficient + Dirichlet) at stored
        # weights, Dirichlet-value parameters at their STORED value (like the nets: `fem.b` then reflects
        # the initialized boundary value rather than an arbitrary zero condition).
        a0, b0 = _assemble_at(
            {n: 0.0 for n in runtime_parameter_tags}
            | {n: _neural_models[n].module for n in neural_param_names}
            | {n: m.module for n, m in _dir_net_models.items()}
            | {n: jnp.asarray(nd.model.module.value) for n, nd in _dir_param_exprs.items()}
        )
        op = FemLinearSystem(
            a0,
            b0,
            operator_fn=lambda args=None: _assemble_at(args)[0],
            rhs_fn=lambda args=None: _assemble_at(args)[1],
            runtime_parameter_exprs=dict(_param_and_neural_exprs),
            # The native parametric path re-assembles the operator at each args (it builds no affine
            # parameter basis), so every runtime parameter -- affine or not -- takes the re-assembly route.
            metadata={"nonaffine_operator": True},
        )
        return op, "linear", offs

    # A τ/t-dependent essential value that no branch above threaded would be silently DROPPED here --
    # the constraint disappears and the solve returns a plausible-looking wrong answer. Fail instead.
    # EXCEPT when the caller declared it consumes the tv stash itself (`tv_dirichlet_external=True`):
    # the second-order u_tt block calls this assembler for the spatial operator and the Dirichlet
    # stashes, then writes g(x_d, t) and the compatible ġ(x_d, t) onto its augmented [u, v] system per
    # step -- a legitimate consumer this guard was firing on (found by the pre-push suite: two wave
    # oracles that pass on origin/main NotImplementedError'd from the guard's own commit onward).
    if _tv_dirichlet and not tv_dirichlet_external:
        raise NotImplementedError(
            "jno.fem: a time/τ-dependent essential value (e.g. `u(top) - delta*tau`) is threaded on the "
            "steady residual path -- the load-path march and the runtime-parametric solve -- and by the "
            "linear transient stepper. This form assembled through neither. Use a constant essential "
            "value, or drive the load through a Neumann/body term written as a function of τ."
        )

    # nonlinear (non-parametric)
    if nonlinear:
        res_bc = _apply_dirichlet_projected(residual, dirichlet_pairs)
        jac = _dirichlet_jac_rows(jacobian, dirichlet_pairs)
        return (
            FemResidualOperator(
                lambda u, args=None: res_bc(jnp.asarray(u)),
                lambda u, args=None: jac(jnp.asarray(u)),
                total,
            ),
            "nonlinear",
            offs,
        )

    # linear (non-parametric)
    A = jacobian(zeros)
    b = -residual(zeros)
    if dirichlet_pairs:
        A, b = _apply_dirichlet_symmetric(A, jnp.asarray(b).reshape(-1), dirichlet_pairs)
    # Collapse duplicate triplets ONCE, after Dirichlet (which appends its own). The assembly emits
    # one block per term and never pre-sums, and each interior DOF pair gets a contribution per
    # incident element -- ~19x redundancy on a 3-D P1 mesh, paid again on every matvec.
    A = compress_eager(A)
    return (A, b), "linear", offs
