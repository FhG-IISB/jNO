"""Known fields on any element space: their per-cell evaluation, and integrals of them over a domain.

A *known* field is a frozen field (``e.bind(...).freeze(values)``) or a field parameter of an element space.
Per cell it is pushed forward with its own space's basis, DOF transform and Piola map, exactly as an unknown
of that space is, and handed to the evaluator as a field slot ``{shape_vals, shape_grads, cell_sol, space}``
with the known values in place of the solution. The non-nodal assembler uses this for coefficients;
:func:`domain_functional` uses it for ``expr.integrate(d)``, the integral of an expression made of known fields
(frozen data, frozen SOLUTIONS of any number of forms, parameters, coordinates, region masks) over a domain.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import jax.numpy as jnp

_EDGE_FAMILIES = ("RT", "N1E", "N2E")


def element_of(space, order):
    """``(family, degree)`` of a symbol's element, as ``build_dofmap`` names it (P0 is DG of degree 0)."""
    sp = str(space).upper()
    if sp == "P0":
        return "DG", 0
    return {"LAGRANGE": "Lagrange", "DG": "DG"}.get(sp, sp), int(order)


def coefficient_spec(fam, k, vs, dofmap, quad_points, dim) -> dict:
    """DOF map, reference basis at ``quad_points`` and Piola map of one known-field element."""
    from .fem_elements import piola_contravariant, piola_contravariant_grad, piola_covariant, piola_covariant_grad

    tab = dofmap.element.tabulate(1, np.asarray(quad_points))
    edge = fam in _EDGE_FAMILIES
    ncomp = 1 if edge else int(np.prod(vs)) if vs else 1
    if edge:
        ref = (jnp.asarray(tab[0]), jnp.asarray(np.stack([tab[1 + a] for a in range(dim)], axis=-1)))
        pf = (piola_contravariant, piola_contravariant_grad) if fam == "RT" else (piola_covariant, piola_covariant_grad)
    else:
        ref = (jnp.asarray(tab[0][..., 0]), jnp.asarray(np.stack([tab[1 + a][..., 0] for a in range(dim)], axis=-1)))
        pf = None
    return {
        "fam": fam,
        "k": k,
        "vs": vs,
        "ncomp": ncomp,
        "n_vals": int(dofmap.n_dofs) * ncomp,
        "cdofs": jnp.asarray(dofmap.cell_dofs, dtype=jnp.int32),
        "ref": ref,
        "piola": pf,
        "signs": jnp.asarray(dofmap.signs) if (edge and dofmap.is_diagonal) else None,
        "tables": None if dofmap.is_diagonal else (dofmap.jax_tables(), int(dofmap.ndof_local)),
    }


def coefficient_slot(sp, values, c, J, detJ) -> dict:
    """Cell ``c``'s field slot of a known field with global ``values`` (the `_cell_fields` dict layout)."""
    from .fem_dofmap import transform_from_tables
    from .small_linalg import small_inv

    loc = sp["cdofs"][c]
    cv = values[loc] if sp["ncomp"] == 1 else jnp.reshape(values, (-1, sp["ncomp"]))[loc]
    rv, rg = sp["ref"]
    if sp["tables"] is not None:  # degree k: the per-cell DOF transform mixes each entity's DOFs
        Bc = transform_from_tables(sp["tables"][0], sp["tables"][1], c)
        rv = jnp.einsum("ij,qj...->qi...", Bc, rv)
        rg = jnp.einsum("ij,qj...->qi...", Bc, rg)
    if sp["piola"] is not None:  # H(curl) / H(div): Piola map with the orientation signs
        sg = sp["signs"][c] if sp["signs"] is not None else jnp.ones((rv.shape[1],), rv.dtype)
        pf, pgf = sp["piola"]
        phi, _d = pf(rv, jnp.zeros(rv.shape[:2], rv.dtype), J, detJ, sg)
        return {"shape_vals": phi, "shape_grads": pgf(rg, J, detJ, sg), "cell_sol": cv, "space": sp["fam"]}
    return {  # Lagrange / DG: scalar basis, the isoparametric chain rule for the gradient
        "shape_vals": rv,
        "shape_grads": rg @ small_inv(J),
        "cell_sol": cv[:, None] if cv.ndim == 1 else cv,
        "space": "Lagrange",
    }


def collect_frozen(expr) -> Dict[int, Any]:
    """Every FrozenField in ``expr``, by ``frozen_id``."""
    from ...trace import FrozenField
    from .solver_helper import iter_children

    found, seen, stack = {}, set(), [expr]
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


def domain_functional(domain, expr) -> Tuple[Callable, List[Tuple[Any, Any]]]:
    """``(fn, inputs)``: ``fn(values) -> ()`` integrates ``expr`` over the whole volume of ``domain``;
    ``inputs`` lists ``(key, node)`` for every value it needs at evaluation time -- a frozen field whose values
    are a trace node (a solve), keyed ``("frozen", id)``, and every runtime parameter, keyed by its name.

    The integrand may hold known fields of any element space, runtime parameters (scalar, P0, P1, element
    space), coordinates and region masks -- but no unknown (no trial or test function: that is a form).
    Quadrature: the simplex rule of degree ``2 k`` (at least 2), with ``k`` the highest field degree, so a
    product of two fields of degree ``k`` is exact (times a cell-wise constant coefficient). Differentiable in
    every input value; the mesh is the static one."""
    import basix

    from ...trace import FrozenField, TestFunction, TrialFunction
    from .fem_dofmap import build_dofmap
    from .fem_topology import BASIX_TET_EDGES, BASIX_TRIANGLE_EDGES, build_edge_topology
    from .fem_utils import _CHUNK_OVERRIDE, _cell_region_mask, _collect_region_mask_names, cell_chunk, elem_map
    from .fem_1d import _integrate_term
    from .parametric_helpers import _collect_runtime_parameter_exprs, _fem_field_kind
    from .small_linalg import small_det
    from .solver_helper import contains_node_type

    if contains_node_type(expr, TrialFunction) or contains_node_type(expr, TestFunction):
        raise ValueError(
            "expr.integrate(domain): the integrand holds an unknown (a trial or test function). An integral over "
            "a domain integrates KNOWN fields -- freeze a solution with `u.bind(...).freeze(fem.solve())`, or "
            "integrate over its form with `expr.integrate(fem)`."
        )
    dim = int(domain.dimension)
    kind = {3: "tetra", 2: "triangle"}.get(dim)
    if kind is None or kind not in domain.mesh.cells_dict:
        raise NotImplementedError(f"expr.integrate(domain): simplex meshes in 2-D / 3-D only (got dimension {dim}).")
    cells_np = np.asarray(domain.mesh.cells_dict[kind], dtype=np.int64)
    pts = jnp.asarray(np.asarray(domain.mesh.points)[:, :dim])
    cells = jnp.asarray(cells_np, dtype=jnp.int32)
    n_cells, n_verts = cells_np.shape[0], int(pts.shape[0])
    top = build_edge_topology(cells_np, BASIX_TET_EDGES if dim == 3 else BASIX_TRIANGLE_EDGES)
    _dm = {}

    def dofmap(fam, k):  # the assemblers' numbering (same edge topology), so solution vectors line up
        if (fam, k) not in _dm:
            _dm[(fam, k)] = build_dofmap(cells_np, fam, k, n_verts=n_verts, edges=(top.cell_edges, top.edge_vertices))
        return _dm[(fam, k)]

    frozen = collect_frozen(expr)
    params: dict = {}
    _collect_runtime_parameter_exprs(expr, params)
    tags = tuple(sorted(params))
    pkind = {n: _fem_field_kind(e) for n, e in params.items()}
    elems = {("frozen", f): element_of(n.space, n.order) + (tuple(n.value_shape),) for f, n in frozen.items()}
    elems.update({("param", n): tuple(params[n].model._fem_space) for n in tags if pkind[n] == "space"})
    kmax = max([k for (_f, k, _v) in elems.values()] + [1 if "node" in pkind.values() else 0])
    rule = basix.make_quadrature(basix.CellType.tetrahedron if dim == 3 else basix.CellType.triangle, max(2, 2 * kmax))
    qp, qw = np.asarray(rule[0]), jnp.asarray(rule[1])
    specs = {key: coefficient_spec(fam, k, vs, dofmap(fam, k), qp, dim) for key, (fam, k, vs) in elems.items()}
    p1 = jnp.asarray(np.column_stack([1.0 - qp.sum(axis=1), qp]))

    inputs: List[Tuple[Any, Any]] = []
    const_vals: dict = {}
    for f, n in frozen.items():
        key = ("frozen", f)
        if getattr(n, "values_node", None) is not None:
            inputs.append((key, n.values_node))
        else:
            const_vals[key] = jnp.reshape(jnp.asarray(n.values), (-1,))
            if const_vals[key].shape[0] != specs[key]["n_vals"]:
                raise ValueError(
                    f"expr.integrate(domain): the frozen {specs[key]['fam']}{specs[key]['k']} field {n.name} has "
                    f"{const_vals[key].shape[0]} values; its space has {specs[key]['n_vals']}."
                )
    inputs += [(n, params[n]) for n in tags]
    rnames = tuple(sorted(_collect_region_mask_names(expr)))
    masks = [jnp.asarray(_cell_region_mask(domain, r)).reshape(-1) for r in rnames]
    ctx = getattr(domain, "context", {}) or {}

    def fn(values):
        fvals = dict(const_vals)
        for key, _node in inputs:
            if isinstance(key, tuple):
                v = jnp.reshape(jnp.asarray(values[key]), (-1,))
                blk = frozen[key[1]].values_block
                if blk is not None:
                    v = v[blk[0] : blk[1]]
                if v.shape[0] != specs[key]["n_vals"]:
                    raise ValueError(
                        f"expr.integrate(domain): a frozen solution has {v.shape[0]} values; its "
                        f"{specs[key]['fam']}{specs[key]['k']} space has {specs[key]['n_vals']} (a form with "
                        "several fields: freeze the block of this field)."
                    )
                fvals[key] = v
        pvals = {n: jnp.reshape(jnp.asarray(values[n]), (-1,)) for n in tags}
        for n in tags:
            want = {"cell": n_cells, "node": n_verts}.get(pkind[n])
            if pkind[n] == "space":
                want = specs[("param", n)]["n_vals"]
                fvals[("param", n)] = pvals[n]
            if want is not None and pvals[n].shape[0] != want:
                raise ValueError(f"expr.integrate(domain): parameter {n!r} has {pvals[n].shape[0]} values, needs {want}.")

        def cell(c):
            verts = pts[cells[c]]
            J = jnp.stack([verts[k] - verts[0] for k in range(1, dim + 1)], axis=1)
            detJ = small_det(J)
            vol_vars = tuple(
                pvals[n][:1]
                if pkind[n] is None
                else jnp.reshape(pvals[n][c], (1,))
                if pkind[n] == "cell"
                else pvals[n][cells[c]]
                if pkind[n] == "node"
                else jnp.zeros((1,), pvals[n].dtype)
                for n in tags
            )
            local = {
                "physical_quad_points": verts[0][None, :] + jnp.asarray(qp) @ J.T,
                "fields": [],
                "field_index": {},
                "tag": "fem_gauss",
                "surface": False,
                "domain_context": ctx,
                "temporal_tags": (),
                "runtime_parameter_tags": tags,
                "region_mask_names": rnames,
                "volume_vars": vol_vars + tuple(m[c] for m in masks),
                "shape_vals": p1,
                "coef_fields": {key: coefficient_slot(sp, fvals[key], c, J, detJ) for key, sp in specs.items()},
            }
            val = _integrate_term(domain, expr, local, qw * jnp.abs(detJ))
            if val.shape != (1,):
                raise ValueError(
                    f"expr.integrate(domain): the integrand is not a scalar per point (shape {tuple(val.shape)} per "
                    "cell). Contract it, e.g. with inner(...)."
                )
            return val

        return jnp.sum(elem_map(cell, (jnp.arange(n_cells),), cell_chunk(n_cells, 1, 1, _CHUNK_OVERRIDE[0])))

    return fn, inputs
