from __future__ import annotations

"""
Internal FEM assembly utilities.

This module contains low-level helpers shared by the steady FEM route and the
transient semidiscrete-time route. It is intentionally not a public API.

Responsibilities:
- lower jNO weak-form symbols into assembly kernels,
- evaluate symbolic expressions at quadrature points inside volume/surface kernels,
- normalize Dirichlet data and prepare residual/Jacobian runtime objects.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ...trace import (
    BinaryOp,
    Constant,
    FrozenField,
    FunctionCall,
    Hessian,
    Jacobian,
    Literal,
    ModelCall,
    OperationCall,
    OperationDef,
    RegionMask,
    StateField,
    TensorTag,
    TestFunction,
    Tracker,
    TrialFunction,
    Variable,
)
from .solver_helper import contains_node_type, iter_children


def _default_float_dtype():
    return jnp.asarray(0.0).dtype


def _lower_statefield_to_trial(expr, trial_cache=None):
    """Final safety pass: replace any remaining StateField with one shared TrialFunction.

    This is backend-side insurance for the Phase-1 NN-first weak route.
    """
    if trial_cache is None:
        trial_cache = {}

    if expr is None:
        return None

    if isinstance(expr, StateField):
        key = (int(expr.state_id), str(expr.name), tuple(expr.value_shape))
        if key not in trial_cache:
            trial_cache[key] = TrialFunction(name=expr.name, value_shape=expr.value_shape)
        return trial_cache[key]

    if isinstance(expr, BinaryOp):
        left = _lower_statefield_to_trial(expr.left, trial_cache)
        right = _lower_statefield_to_trial(expr.right, trial_cache)
        if left is not expr.left or right is not expr.right:
            return BinaryOp(expr.op, left, right)
        return expr

    if isinstance(expr, FunctionCall):
        new_args = []
        changed = False
        for a in expr.args:
            if isinstance(
                a,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                na = _lower_statefield_to_trial(a, trial_cache)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            return expr.copy_with_args(new_args)
        return expr

    if isinstance(expr, ModelCall):
        new_args = []
        changed = False
        for a in expr.args:
            if isinstance(
                a,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                na = _lower_statefield_to_trial(a, trial_cache)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            rebuilt = ModelCall(expr.model, new_args)
            rebuilt.op_id = expr.op_id
            return rebuilt
        return expr

    if isinstance(expr, OperationDef):
        new_expr = _lower_statefield_to_trial(expr.expr, trial_cache)
        if new_expr is not expr.expr:
            rebuilt = OperationDef.__new__(OperationDef)
            rebuilt.expr = new_expr
            rebuilt.input_vars = expr.input_vars
            rebuilt.name = getattr(expr, "name", None)
            rebuilt.op_id = expr.op_id
            return rebuilt
        return expr

    if isinstance(expr, OperationCall):
        new_args = []
        changed = False
        for a in expr.args:
            if isinstance(
                a,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                na = _lower_statefield_to_trial(a, trial_cache)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            rebuilt = OperationCall(expr.operation, tuple(new_args))
            rebuilt.op_id = expr.op_id
            return rebuilt
        return expr

    if isinstance(expr, Jacobian):
        new_target = _lower_statefield_to_trial(expr.target, trial_cache)
        new_vars = []
        changed = new_target is not expr.target
        for v in expr.variables:
            if isinstance(
                v,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                nv = _lower_statefield_to_trial(v, trial_cache)
            else:
                nv = v
            changed = changed or (nv is not v)
            new_vars.append(nv)
        if changed:
            return Jacobian(new_target, new_vars, expr.scheme)
        return expr

    if isinstance(expr, Hessian):
        new_target = _lower_statefield_to_trial(expr.target, trial_cache)
        new_vars = []
        changed = new_target is not expr.target
        for v in expr.variables:
            if isinstance(
                v,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                nv = _lower_statefield_to_trial(v, trial_cache)
            else:
                nv = v
            changed = changed or (nv is not v)
            new_vars.append(nv)
        if changed:
            return Hessian(new_target, new_vars, expr.scheme, trace=expr.trace)
        return expr

    return expr


def _const_bc_fn(value):
    value = float(value)
    return lambda p, c=value: c


def _normalize_dirichlet_value(value, vec: int):
    """
    Normalize user Dirichlet data into value functions.

    Accepts None, scalar, callable, component list/tuple, or component dict.
    Returns one callable for scalar fields, a length-``vec`` list of callables for
    a fully specified vector field, or a partial ``{component_index: callable}`` dict
    when only some components are constrained (e.g. a roller/symmetry BC
    ``{"y": 0.0}`` pins only ``u_y`` and leaves the other components free).
    """
    if value is None:
        value = 0.0

    if vec < 1:
        raise ValueError(f"'vec' must be >= 1, got {vec}.")

    if callable(value):
        if vec == 1:
            return value
        return [value for _ in range(vec)]

    if np.isscalar(value):
        fn = _const_bc_fn(value)
        if vec == 1:
            return fn
        return [fn for _ in range(vec)]

    if isinstance(value, (list, tuple)):
        if len(value) != vec:
            raise ValueError(f"Dirichlet BC has {len(value)} entries, but vec={vec}.")
        out = []
        for v in value:
            if callable(v):
                out.append(v)
            elif np.isscalar(v):
                out.append(_const_bc_fn(v))
            else:
                raise TypeError("Dirichlet list/tuple entries must be callables or scalars.")
        if vec == 1:
            return out[0]
        return out

    if isinstance(value, dict):
        keymap = {"x": 0, "y": 1, "z": 2}
        # Partial spec: only the named components are constrained; the rest stay
        # free. (Zero-filling here would silently clamp the other components, which
        # breaks roller/symmetry BCs and conflicts at shared corner nodes.)
        out: dict[int, Any] = {}
        for k, v in value.items():
            c = keymap[k.lower()] if isinstance(k, str) else int(k)
            if c < 0 or c >= vec:
                raise ValueError(f"Component index {c} out of range for vec={vec}.")
            if callable(v):
                out[c] = v
            elif np.isscalar(v):
                out[c] = _const_bc_fn(v)
            else:
                raise TypeError("Dirichlet dict entries must be callables or scalars.")
        if vec == 1:
            return out.get(0, _const_bc_fn(0.0))
        return out

    raise TypeError(f"Unsupported Dirichlet BC value type: {type(value).__name__}")


# --------------------------------
# small expression-inspection helpers
# --------------------------------


def _strip_test_function_factor(expr):
    factors = []

    def collect_mul_factors(node):
        if isinstance(node, BinaryOp) and node.op == "*":
            collect_mul_factors(node.left)
            collect_mul_factors(node.right)
        else:
            factors.append(node)

    collect_mul_factors(expr)
    test_factors = [f for f in factors if isinstance(f, TestFunction)]
    if len(test_factors) != 1:
        return None

    coeff_factors = [f for f in factors if not isinstance(f, TestFunction)]
    if len(coeff_factors) == 0:
        return Literal(1.0)

    coeff = coeff_factors[0]
    for f in coeff_factors[1:]:
        coeff = BinaryOp("*", coeff, f)
    return coeff


def _is_simple_neumann_load(expr):
    if not contains_node_type(expr, TestFunction):
        return False
    if contains_node_type(expr, TrialFunction):
        return False
    if contains_node_type(expr, Jacobian):
        return False
    coeff = _strip_test_function_factor(expr)
    return coeff is not None


def _value_shape_num_components(value_shape) -> int:
    if value_shape is None or len(value_shape) == 0:
        return 1
    n = 1
    for s in value_shape:
        n *= int(s)
    return n


def _reshape_components_last(arr, value_shape):
    if value_shape is None or len(value_shape) == 0:
        return arr
    return jnp.reshape(arr, arr.shape[:-1] + tuple(value_shape))


def _expand_test_shape_vals(shape_vals, n_comp):
    if n_comp == 1:
        return shape_vals
    eye = jnp.eye(n_comp, dtype=shape_vals.dtype)
    return shape_vals[:, :, None, None] * eye[None, None, :, :]


def _infer_trial_metadata(expr) -> Dict[str, Any]:
    """
    Infer the FEM unknown metadata from TrialFunction nodes.

    Returns the unique trial symbol, its value shape, vector size, and whether
    the expression contains a TrialFunction.
    """
    trial_nodes = {}

    def walk(node):
        if node is None:
            return

        if isinstance(node, TrialFunction):
            trial_nodes[node.op_id] = node
            return

        for child in iter_children(node):
            walk(child)

    walk(expr)

    unique_trials = list(trial_nodes.values())
    if len(unique_trials) > 1:
        raise NotImplementedError(
            "The FEM assembler currently supports exactly one TrialFunction "
            "(scalar or vector valued). Multiple coupled FEM unknowns will "
            "come in the next refactor step."
        )

    trial = unique_trials[0] if unique_trials else None
    value_shape = getattr(trial, "value_shape", ()) if trial is not None else ()
    vec = _value_shape_num_components(value_shape)

    return {
        "trial": trial,
        "value_shape": value_shape,
        "vec": vec,
        "has_trial": trial is not None,
    }


def _infer_fields(expr) -> Tuple[List[Dict[str, Any]], Dict[Any, int]]:
    """All distinct trial *fields* in ``expr``, ordered by first appearance.

    A field is one coupled unknown (one ``fem_symbols()`` call — its trial and test
    share a ``field_key``). Returns ``(fields, field_key->index)`` where each field is
    ``{field_key, value_shape, vec, order}``. The index order is the single source of
    truth threaded into the per-field DOF blocks and the multi-field kernel.
    """
    fields: List[Dict[str, Any]] = []
    seen: Dict[Any, int] = {}

    def walk(node):
        if node is None:
            return
        if isinstance(node, TrialFunction):
            key = getattr(node, "field_key", node.op_id)
            if key not in seen:
                seen[key] = len(fields)
                vs = getattr(node, "value_shape", ())
                fields.append(
                    {
                        "field_key": key,
                        "value_shape": vs,
                        "vec": _value_shape_num_components(vs),
                        "order": int(getattr(node, "order", 1)),
                        "space": str(getattr(node, "space", "Lagrange")),
                    }
                )
            return
        for child in iter_children(node):
            walk(child)

    walk(expr)
    return fields, dict(seen)


def _test_field_index(expr, field_index: Dict[Any, int]) -> Optional[int]:
    """Index of the single test field in an additive weak term (or ``None``).

    Each additive term must contain exactly one test field (it determines the
    equation/row block); a term with zero or several distinct test fields is
    ambiguous and the caller errors."""
    keys = set()

    def walk(node):
        if node is None:
            return
        if isinstance(node, TestFunction):
            keys.add(getattr(node, "field_key", node.op_id))
            return
        for child in iter_children(node):
            walk(child)

    walk(expr)
    if len(keys) != 1:
        return None
    return field_index.get(next(iter(keys)))


def _expand_product_terms(node, sign: float = 1.0):
    """Distribute products/quotients over sums → a list of ``(sign, product_node)``.

    Used only as a fallback when a coupled weak term carries several test fields
    welded inside a product — e.g. the **real part of a complex weak form**, where
    a coefficient multiplies a sum that straddles two test fields,
    ``c·(u_r·p_r − u_i·q_i)``. Fully distributing yields ``c·u_r·p_r`` and
    ``−c·u_i·q_i``, each of which classifies to a single test field/equation block.
    The cross-product can grow, but weak-form terms are small and this runs only on
    the (rare) terms that would otherwise be rejected."""
    if isinstance(node, BinaryOp):
        if node.op == "+":
            return _expand_product_terms(node.left, sign) + _expand_product_terms(node.right, sign)
        if node.op == "-":
            return _expand_product_terms(node.left, sign) + _expand_product_terms(node.right, -sign)
        if node.op == "*":
            left = _expand_product_terms(node.left, 1.0)
            right = _expand_product_terms(node.right, 1.0)
            return [(sign * sl * sr, BinaryOp("*", lt, rt)) for sl, lt in left for sr, rt in right]
        if node.op == "/":
            # distribute the numerator only; the denominator is treated as a coefficient
            return [(s, BinaryOp("/", t, node.right)) for s, t in _expand_product_terms(node.left, sign)]
    return [(sign, node)]


def _gather_runtime_parameter_tags(node, out=None):
    """Collect names of trainable runtime parameters (ModelCall) used in a coeff."""
    from .parametric_helpers import _is_runtime_scalar_parameter, _parameter_name

    if out is None:
        out = []
    if _is_runtime_scalar_parameter(node):
        name = _parameter_name(node)
        if name not in out:
            out.append(name)
        return out
    for child in iter_children(node) or ():
        _gather_runtime_parameter_tags(child, out)
    return out


def _collect_region_mask_names(node, out=None):
    """Sorted-unique region names appearing as ``RegionMask`` leaves in a lowered expression/IR.

    Used so the volume kernel knows the per-cell mask layout and the assembler builds the matching
    masks (see ``_cell_region_mask``). Order is fixed by sorting so kernel and assembler agree."""
    if out is None:
        out = set()
    if isinstance(node, RegionMask):
        out.add(node.region)
    for child in iter_children(node) or ():
        _collect_region_mask_names(child, out)
    return out


def _cell_region_mask(domain, region):
    """``(num_cells,)`` 0/1 indicator: a mesh cell is in ``region`` iff its **centroid** is.

    ``region`` is a geometry part (``domain._source_regions`` shapely polygon) or a ``domain.tag``
    predicate. Concrete (numpy/shapely), built once per assembly -- exact when the mesh respects the
    region boundaries (gmsh meshes each part separately, so no cell straddles a material interface).

    Classifies against the **assembly mesh** (the cell order the kernel vmaps over, stashed by
    ``_build_fem_problem``) so the per-cell mask aligns with the kernel's cells; falls back to the
    domain mesh only if no assembly mesh is recorded yet."""
    dim = int(domain.dimension)
    a_pts = getattr(domain, "_fem_assembly_points", None)
    a_cells = getattr(domain, "_fem_assembly_cells", None)
    if a_pts is not None and a_cells is not None:
        pts = np.asarray(a_pts)[:, :dim]
        cells = np.asarray(a_cells)
    else:
        mesh = domain.mesh
        pts = np.asarray(mesh.points)[:, :dim]
        cells = np.asarray(mesh.cells_dict["triangle" if dim == 2 else "tetra"])
    centroids = pts[cells].mean(axis=1)  # (num_cells, dim)
    src = getattr(domain, "_source_regions", {}) or {}
    preds = getattr(domain, "_tag_predicates", {}) or {}
    if region in src:
        from shapely import contains_xy

        m = np.asarray(contains_xy(src[region], centroids[:, 0], centroids[:, 1]))
    elif region in preds:
        m = np.asarray(preds[region](*[centroids[:, i] for i in range(dim)]))
    else:
        raise ValueError(
            f"jno.fem per-region integration: unknown region {region!r}. Define it with "
            f"domain.tag(name, predicate) or as a geometry part (domain._source_regions)."
        )
    return np.asarray(m, dtype=bool).astype(np.float64)


def _gather_temporal_tags(node, out=None):
    """
    Collect temporal Variable tags used inside the kernels.

    These tags determine which time values must be passed through the
    batched volume-variable arrays during transient assembly.
    """
    if out is None:
        out = set()

    if isinstance(node, Variable) and getattr(node, "axis", None) == "temporal":
        out.add(str(node.tag))
        return out

    if isinstance(node, BinaryOp):
        _gather_temporal_tags(node.left, out)
        _gather_temporal_tags(node.right, out)
        return out

    if isinstance(node, FunctionCall):
        for a in node.args:
            if isinstance(
                a,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                _gather_temporal_tags(a, out)
        return out

    if isinstance(node, Jacobian):
        _gather_temporal_tags(node.target, out)
        for v in node.variables:
            if isinstance(
                v,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                _gather_temporal_tags(v, out)
        return out

    if isinstance(node, Hessian):
        _gather_temporal_tags(node.target, out)
        for v in node.variables:
            if isinstance(
                v,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                _gather_temporal_tags(v, out)
        return out

    return out


def _runtime_parameter_value_from_internal_vars(local, name):
    """Read a runtime parameter scalar from volume_vars.

    Parameter values are packed AFTER the temporal values, so the volume_vars
    layout is:  [ temporal_tags ... , runtime_parameter_tags ... ].
    """
    temporal_tags = local.get("temporal_tags", ())
    param_tags = local.get("runtime_parameter_tags", ())
    volume_vars = local.get("volume_vars", ())
    if name not in param_tags:
        return None
    idx = len(temporal_tags) + param_tags.index(name)
    if idx >= len(volume_vars):
        return None
    arr = jnp.asarray(volume_vars[idx])
    flat = jnp.reshape(arr, (-1,))
    # Scalar coefficient (incl. per-cell-constant): one value -> broadcast to quad.
    if flat.shape[0] == 1:
        return flat[0]
    # Node-based field coefficient: the kernel has gathered the cell's local nodal values
    # (size = nodes-per-element); interpolate to quadrature points with the field's
    # shape functions, mirroring the solution interpolation (shape_vals . nodal).
    # Returns (n_quad, 1) like a field value.
    shape_vals = local.get("shape_vals")  # (n_quad, n_local)
    cell_nodal = flat.reshape(flat.shape[0], 1)  # (n_local, 1)
    return jnp.sum(shape_vals[:, :, None] * cell_nodal[None, :, :], axis=1)


def _temporal_value_from_internal_vars(local, tag, dim_start=0, dim_end=1):
    """
    Read a temporal variable value from the batched volume-variable arrays.

    Returns None when the requested temporal tag is not part of the current
    kernel call.
    """
    temporal_tags = local.get("temporal_tags", ())
    volume_vars = local.get("volume_vars", ())

    if tag not in temporal_tags:
        return None

    idx = temporal_tags.index(tag)
    if idx >= len(volume_vars):
        raise IndexError(
            f"Temporal variable tag '{tag}' mapped to slot {idx}, but only {len(volume_vars)} volume_vars were provided."
        )

    arr = jnp.asarray(volume_vars[idx])
    # scalar / (1,) / (1,1) -> one scalar time for this assembly call
    t_scalar = jnp.reshape(arr, (-1,))[0]
    out = jnp.asarray([t_scalar])
    return out[dim_start:dim_end]


# --------------------------------
# Expression evaluation helpers
# --------------------------------


def _prefix_align(a, b):
    """Broadcast-align two kernel quantities for an elementwise op.

    In the kernel, trial-derived quantities are laid out as ``(n_quad,
    *value)`` while test-derived ones carry extra leading test-DOF axes
    ``(n_quad, *dof, *value)`` (the per-DOF basis expansion). Their shared axis
    is the leading quadrature axis and their value axes are trailing, so when the
    ranks differ we pad the lower-rank operand with singleton axes **right after
    the quad axis** until the ranks match. The trailing value axes then align by
    normal right-broadcasting and the test-DOF axes broadcast against the inserted
    singletons. This mirrors the prefix-padding ``jno.np.inner`` already does, so
    arbitrary tensor algebra between trial- and test-derived quantities works
    (e.g. ``div(u) * div(phi)``), not just explicit contractions.

    Only activates when ranks differ (and both operands are arrays), so
    equal-rank expressions keep their exact current broadcasting.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.ndim == b.ndim or a.ndim == 0 or b.ndim == 0:
        return a, b
    if a.ndim < b.ndim:
        pad = (1,) * (b.ndim - a.ndim)
        a = jnp.reshape(a, a.shape[:1] + pad + a.shape[1:])
    else:
        pad = (1,) * (a.ndim - b.ndim)
        b = jnp.reshape(b, b.shape[:1] + pad + b.shape[1:])
    return a, b


def _field_data(local, node):
    """``(shape_vals, shape_grads, cell_sol)`` for ``node``'s field.

    A multi-field kernel puts per-field arrays in ``local["fields"]`` (indexed by
    field index) and a ``field_key -> index`` map in ``local["field_index"]``. A
    single-field kernel has neither, so this falls back to the flat ``local`` entries
    — leaving the single-field evaluation path byte-identical."""
    fields = local.get("fields")
    if fields is None:
        return local["shape_vals"], local.get("shape_grads"), local.get("cell_sol")
    key = getattr(node, "field_key", getattr(node, "op_id", None))
    fd = fields[local["field_index"][key]]
    return fd["shape_vals"], fd["shape_grads"], fd["cell_sol"]


def _frozen_cell_values(local, node):
    """This cell's frozen nodal slice ``(n_local, vec)`` for a :class:`FrozenField`.

    The native kernel gathers each frozen field's per-cell nodal values (a compile-time
    constant -- no ``args`` threading, no ``jacfwd`` tangent) into
    ``local["frozen_fields"][frozen_id]``; here we just read it back to interpolate the
    known value / contract the known gradient with the field's shape data."""
    table = local.get("frozen_fields")
    if table is None or node.frozen_id not in table:
        raise NotImplementedError(
            "A frozen field (ui.freeze(values)) was used in a weak form assembled by a path that does "
            "not thread frozen fields. It is currently wired for the native steady/linear volume path."
        )
    return table[node.frozen_id]


def _field_hess(local, node):
    """Physical shape-function Hessian ``(n_quad, n_dof, dim, dim)`` for ``node``'s field, or ``None`` if
    its element does not tabulate second derivatives (only nodal Lagrange does). Mirrors :func:`_field_data`."""
    fields = local.get("fields")
    if fields is None:
        return local.get("shape_hess")
    key = getattr(node, "field_key", getattr(node, "op_id", None))
    return fields[local["field_index"][key]].get("shape_hess")


def _field_space(local, node):
    """Element family of ``node``'s field (``"Lagrange"`` default).

    Non-nodal families (``"RT"``, ...) are assembled by the native push-forward path
    (:mod:`fem_nonnodal`), which tags each ``local["fields"]`` entry with its ``space`` and
    supplies *physical* (push-forward) shape data. The value branches below switch on this so
    the Lagrange path stays byte-identical (a single-field Lagrange kernel has no ``fields``)."""
    fields = local.get("fields")
    if fields is None:
        return local.get("space", "Lagrange")
    key = getattr(node, "field_key", getattr(node, "op_id", None))
    return fields[local["field_index"][key]].get("space", "Lagrange")


def _eval_frozen_coefficient(domain, model, local):
    """Value of a ``.freeze()``d (known) coefficient at the kernel's quadrature points.

    The known value is supplied via ``.initialize`` and read here at assembly time:

    * a **coordinate function** ``(x, y[, z]) -> value`` (``.initialize(lambda x, y: ...)``) is called
      on the physical quadrature coordinates — identical to ``jno.fn`` — and may return a scalar or a
      vector (tuple / last-axis array) field;
    * a **scalar constant** (``.initialize(0.8)``) is returned as-is.

    A raw per-node value array is *not* supported (turning scattered nodal data into a coefficient
    needs mesh interpolation — use ``jno.fn`` or a function instead); a frozen parameter with no value
    fails loud."""
    fn = getattr(model, "_initializer_fn", None)
    if fn is not None:  # coordinate function, evaluated at the quad points (like jno.fn)
        import inspect

        try:
            params = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            params = set()
        if {"key", "shape"} & params:  # a JAX initializer (key, shape, dtype), not a coordinate fn
            raise ValueError(
                "jno.fem: a frozen (.freeze()d) coefficient takes a *scalar* or a coordinate function "
                "(x, y[, z]) -> value — not a JAX initializer. For a uniform value use .initialize(<number>); "
                "JAX initializers (jax.nn.initializers.*) are for *trainable* parameters."
            )
        qp = local["physical_quad_points"]
        dim = int(getattr(domain, "dimension", qp.shape[-1]))
        return jnp.asarray(fn(*(qp[..., i] for i in range(dim))))

    weight = getattr(model, "_weight_tree", None)
    if weight is not None:
        arr = jnp.asarray(weight)
        if arr.ndim == 0:  # scalar constant
            return arr
        raise NotImplementedError(
            "jno.fem: a frozen (.freeze()d) coefficient must be a constant or a (x, y[, z]) -> value "
            "function — a raw per-node array is not supported (use jno.fn(...) or a function)."
        )

    raise ValueError(
        f"jno.fem: frozen parameter {getattr(model, '_parameter_name', '<param>')!r} has no value — "
        "call .initialize(<scalar or (x, y[, z]) -> value function>) before .freeze()."
    )


def _call_neural_coefficient(model, module, arg_vals, local):
    """Evaluate one neural coefficient (``jno.nn.wrap(net)`` inside a weak form) at the quad points.

    ``arg_vals`` are the network's arguments already evaluated by ``_eval_integrand`` — coordinate
    Variables arrive as ``(n_quad, 1)`` slices of the physical quadrature points, an interpolated
    trial value as ``(n_quad, c)``, its gradient as ``(n_quad, len(dims))`` — so ``k(x)``, ``k(u)``
    and ``k(∇u)`` all share this call. Everything is normalised to a ``(n_quad, feat)`` batch (the
    same convention as the point-cloud evaluator ``TraceEvaluator._eval_flax_module_call``, which
    foundax MLPs concatenate internally), cast to the model's explicit ``.dtype`` opt-in when set,
    and the network output is returned as ``(n_quad, k)`` — the shape a nodal field parameter
    returns, so it composes with test-expanded factors via ``_prefix_align`` identically.

    ``module`` is the *current* weight pytree threaded through the runtime ``args`` (a
    ``ModelWeights`` slot), so under ``jax.jacfwd`` w.r.t. the cell DOFs a trial-dependent argument
    carries tangents straight through the network — ∂k(u)/∂u enters the element Jacobian
    automatically — and under the outer training loop the weights are the crux-recombined
    trainable leaves, so gradients reach the optimizer.

    Neural-coefficient support follows the unsupervised coefficient/constitutive-recovery setting
    of NN-EUCLID (M. Flaschel, S. Kumar, L. De Lorenzis, "NN-EUCLID: Deep-learning hyperelasticity
    without stress data", J. Mech. Phys. Solids 165 (2022) 105076, §2.2–2.3) and Tartakovsky et
    al., "Learning Parameters and Constitutive Relationships with Physics-Informed Deep Neural
    Networks" (Water Resour. Res. 56, 2020, §2).
    """
    n_quad = int(local["physical_quad_points"].shape[0])

    def _as_quad_batch(v):
        v = jnp.asarray(v)
        if v.ndim == 0:
            return jnp.broadcast_to(v.reshape(1, 1), (n_quad, 1))
        if v.ndim == 1:
            if v.shape[0] == n_quad:
                return v.reshape(n_quad, 1)
            if v.shape[0] == 1:
                return jnp.broadcast_to(v.reshape(1, 1), (n_quad, 1))
        if v.ndim == 2 and v.shape[0] == n_quad:
            return v
        if v.ndim == 2 and v.shape[0] == 1:
            return jnp.broadcast_to(v, (n_quad, v.shape[1]))
        raise NotImplementedError(
            "jno.fem: a neural coefficient's arguments must be per-quad-point values (coordinates, "
            f"the trial or its derivatives, scalars); got an argument of shape {v.shape}. A test "
            "function cannot appear inside a network argument."
        )

    vals = [_as_quad_batch(v) for v in arg_vals]

    # Mixed precision: honour an explicit Model.dtype() opt-in exactly like the point evaluator —
    # cast floating inputs so the net computes in its declared dtype; a plain-f32 net under x64
    # promotes through the surrounding arithmetic instead.
    compute_dtype = getattr(model, "_dtype", None)
    if compute_dtype is not None:
        vals = [
            v.astype(compute_dtype) if jnp.issubdtype(v.dtype, jnp.floating) and v.dtype != compute_dtype else v
            for v in vals
        ]

    out = module(*vals)
    if hasattr(out, "output"):  # structured foundation-model outputs
        out = out.output
    out = jnp.asarray(out)
    if out.ndim == 0:
        return jnp.broadcast_to(out.reshape(1, 1), (n_quad, 1))
    if out.ndim == 1 and out.shape[0] == n_quad:
        return out.reshape(n_quad, 1)
    if out.ndim == 2 and out.shape[0] == n_quad:
        return out
    raise NotImplementedError(
        f"jno.fem: a neural coefficient must return one value (or a feature vector) per quadrature "
        f"point — got output shape {out.shape} for {n_quad} quad points."
    )


def _eval_integrand(domain, node, local):
    """
    Evaluate a jNO symbolic expression inside a local kernel.

    The `local` dictionary contains quadrature coordinates, shape values,
    shape gradients, local cell DOFs, domain context, and the optional batched
    temporal volume-variable arrays. This evaluator supports literals, constants,
    variables, tensor tags, TrialFunction/TestFunction values, their Jacobians,
    binary operations, and FunctionCall nodes.
    """
    if not isinstance(
        node,
        (
            Literal,
            Constant,
            TensorTag,
            RegionMask,
            Variable,
            TestFunction,
            TrialFunction,
            FrozenField,
            Jacobian,
            Hessian,
            BinaryOp,
            FunctionCall,
            ModelCall,
        ),
    ):
        try:
            return jnp.asarray(node)
        except Exception:
            pass

    if isinstance(node, Literal):
        return jnp.asarray(node.value)

    if isinstance(node, Constant):
        return jnp.asarray(node.value)

    if isinstance(node, RegionMask):
        # Per-cell sub-region indicator: read the current cell's 0/1 value from the constant per-cell
        # volume_var packed AFTER the temporal + runtime-parameter slots. Fail loud (never silently
        # integrate over the whole domain) if this assembly path did not thread the mask.
        mask_names = local.get("region_mask_names", ())
        volume_vars = local.get("volume_vars", ())
        idx = (
            len(local.get("temporal_tags", ()))
            + len(local.get("runtime_parameter_tags", ()))
            + (list(mask_names).index(node.region) if node.region in mask_names else -1)
        )
        if node.region not in mask_names or idx >= len(volume_vars):
            raise NotImplementedError(
                f"jno.fem per-region integration: the per-cell mask for region '{node.region}' was not "
                f"threaded into this assembly path. Sub-region terms are currently wired for the steady "
                f"linear single-field path; nonlinear / transient / multifield / parametric are not yet."
            )
        return jnp.reshape(jnp.asarray(volume_vars[idx]), (-1,))[0]

    if isinstance(node, TensorTag):
        if node.tag not in local["domain_context"]:
            raise KeyError(f"TensorTag '{node.tag}' not found in FEM domain context.")
        tensor = jnp.asarray(local["domain_context"][node.tag])
        if tensor.ndim >= 1 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.ndim >= 1 and tensor.shape[0] > 1:
            raise NotImplementedError(
                "The FEM assembler currently supports singleton-batch TensorTag coefficients only. "
                f"Got shape {tensor.shape} for tag '{node.tag}'."
            )
        if node.dim_index is not None and tensor.ndim >= 1:
            tensor = tensor[..., node.dim_index]
        return tensor

    if isinstance(node, Variable):
        dim_start, dim_end = node.dim

        # Local quadrature coordinates
        if local.get("surface", False):
            if isinstance(node.tag, str) and node.tag.startswith("gauss_"):
                return local["physical_quad_points"][..., dim_start:dim_end]
        else:
            if node.tag == "fem_gauss":
                return local["physical_quad_points"][..., dim_start:dim_end]

        # Temporal variable in assembly:
        # prefer the batched volume_vars arrays (pure JAX / no domain mutation),
        # then fall back to domain.context for the steady-context path.
        if getattr(node, "axis", None) == "temporal":
            from_iv = _temporal_value_from_internal_vars(
                local,
                str(node.tag),
                dim_start=dim_start,
                dim_end=dim_end,
            )
            if from_iv is not None:
                return from_iv

            if node.tag not in local["domain_context"]:
                raise KeyError(f"Temporal Variable tag '{node.tag}' not found in local/domain context.")
            arr = jnp.asarray(local["domain_context"][node.tag])
            t_scalar = jnp.reshape(arr, (-1,))[0]
            out = jnp.asarray([t_scalar])
            return out[dim_start:dim_end]

        # Fallback to stored tensor/point-data context
        if node.tag in local["domain_context"]:
            arr = jnp.asarray(local["domain_context"][node.tag])
            if arr.ndim >= 1 and arr.shape[0] == 1:
                arr = arr[0]
            return arr[..., dim_start:dim_end]

        raise KeyError(f"Variable tag '{node.tag}' not found in local/domain context.")

    if isinstance(node, TestFunction):
        shape_vals, _, _ = _field_data(local, node)
        if _field_space(local, node) != "Lagrange":
            # non-nodal: shape_vals is already the per-DOF *physical* basis (n_quad, n_dof, *value)
            return shape_vals
        n_comp = _value_shape_num_components(getattr(node, "value_shape", ()))
        return _expand_test_shape_vals(shape_vals, n_comp)

    if isinstance(node, TrialFunction):
        vals, _, cell_sol = _field_data(local, node)
        if _field_space(local, node) != "Lagrange":
            # non-nodal: u = sum_n cell_sol[n] Phi_n(x). Vector basis (RT) is (n_quad, n_dof, value_size);
            # scalar basis (P0/DG) is (n_quad, n_dof).
            if vals.ndim == 2:
                return jnp.einsum("qn,n->q", vals, cell_sol)
            return jnp.einsum("qnc,n->qc", vals, cell_sol)
        flat_interp = jnp.sum(vals[:, :, None] * cell_sol[None, :, :], axis=1)
        value_shape = getattr(node, "value_shape", ())
        if len(value_shape) == 0:
            return flat_interp
        return _reshape_components_last(flat_interp, value_shape)

    if isinstance(node, FrozenField):
        # KNOWN field: interpolate its frozen nodal slice at the quad points, exactly like the
        # TrialFunction value branch but with frozen values instead of the live cell solution.
        if _field_space(local, node) != "Lagrange":
            raise NotImplementedError("ui.freeze(values): frozen fields are supported for nodal Lagrange only.")
        vals, _, _ = _field_data(local, node)
        fz = _frozen_cell_values(local, node)  # (n_local, vec)
        flat_interp = jnp.sum(vals[:, :, None] * fz[None, :, :], axis=1)
        value_shape = getattr(node, "value_shape", ())
        if len(value_shape) == 0:
            return flat_interp
        return _reshape_components_last(flat_interp, value_shape)

    if isinstance(node, Jacobian):
        dims = []
        for var in node.variables:
            if not isinstance(var, Variable):
                raise NotImplementedError(
                    "The FEM assembler expects Jacobian variables to be domain.variable(...) placeholders."
                )
            dims.append(var.dim[0])
        if len(dims) == 0:
            raise ValueError("Jacobian node has no differentiation variables")

        if isinstance(node.target, TestFunction):
            _, grads, _ = _field_data(local, node.target)
            if _field_space(local, node.target) != "Lagrange":
                # non-nodal: grads is the per-DOF *physical* gradient (n_quad, n_dof, n_comp, n_dims);
                # pick the requested directions -> (n_quad, n_dof, n_comp[, len(dims)]). trace() then gives div.
                g = jnp.stack([grads[..., d] for d in dims], axis=-1)
                return g[..., 0] if len(dims) == 1 else g
            n_comp = _value_shape_num_components(getattr(node.target, "value_shape", ()))
            if n_comp == 1:
                comps = [grads[..., dim0] for dim0 in dims]
                return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)
            eye = jnp.eye(n_comp, dtype=grads.dtype)
            comps = [grads[..., dim0][:, :, None, None] * eye[None, None, :, :] for dim0 in dims]
            if len(comps) == 1:
                return comps[0]
            return jnp.stack(comps, axis=-1)

        if isinstance(node.target, FrozenField):
            # gradient of a KNOWN field: contract the SAME physical shape gradients with the frozen
            # nodal slice instead of the live cell solution (byte-identical to the TrialFunction case).
            if _field_space(local, node.target) != "Lagrange":
                raise NotImplementedError("ui.freeze(values): frozen-field gradients are nodal-Lagrange only.")
            _, grads, _ = _field_data(local, node.target)
            fz = _frozen_cell_values(local, node.target)  # (n_local, vec)
            grad_list = [jnp.sum(grads[:, :, dim0 : dim0 + 1] * fz[None, :, :], axis=1) for dim0 in dims]
            flat = grad_list[0] if len(dims) == 1 else jnp.stack(grad_list, axis=-1)
            value_shape = getattr(node.target, "value_shape", ())
            if len(value_shape) == 0:
                return flat
            if len(dims) == 1:
                return _reshape_components_last(flat, value_shape)
            return jnp.reshape(flat, flat.shape[:1] + tuple(value_shape) + (len(dims),))

        if isinstance(node.target, TrialFunction):
            _, grads, cell_sol = _field_data(local, node.target)
            if _field_space(local, node.target) != "Lagrange":
                # non-nodal: du_i/dx_l = sum_n cell_sol[n] grad[n, i, l] -> (n_quad, n_comp[, len(dims)])
                g = jnp.stack([grads[..., d] for d in dims], axis=-1)
                contracted = jnp.einsum("qn...,n->q...", g, cell_sol)
                return contracted[..., 0] if len(dims) == 1 else contracted
            grad_list = [jnp.sum(grads[:, :, dim0 : dim0 + 1] * cell_sol[None, :, :], axis=1) for dim0 in dims]
            flat = grad_list[0] if len(dims) == 1 else jnp.stack(grad_list, axis=-1)
            value_shape = getattr(node.target, "value_shape", ())
            if len(value_shape) == 0:
                return flat
            if len(dims) == 1:
                return _reshape_components_last(flat, value_shape)
            return jnp.reshape(flat, flat.shape[:1] + tuple(value_shape) + (len(dims),))

        # Component-of-field gradient: ``u[i].d(x)`` lowers to ``Jacobian(getitem(field, i), [x])``. A
        # non-nodal field's value-component cannot be differentiated directly (that is the rejected
        # "Jacobian-of-getitem"), but ``d(u_i)/dx_l`` IS the (component i, direction l) entry of the
        # whole-field *physical* gradient -- so select that row. This is what makes the existing
        # ``.div()`` / ``.curl()`` / ``.x``-partial sugar (all of which build ``u[i].d(v)``) work for RT
        # and N1E. Nodal FEM keeps the ``trace(grad(u, [x, y]))`` idiom: a nodal field's getitem-gradient
        # has a different tensor structure and stays out of scope here.
        tgt = node.target
        if isinstance(tgt, FunctionCall) and getattr(tgt, "getitem_key", None) is not None and len(tgt.args) == 1:
            field = tgt.args[0]
            ints = [k for k in tgt.getitem_key if isinstance(k, int)]
            if ints and isinstance(field, (TrialFunction, TestFunction)) and _field_space(local, field) != "Lagrange":
                comp = ints[-1]
                _, grads, cell_sol = _field_data(local, field)  # grads: (n_quad, n_dof, n_comp, n_dims)
                gc = grads[:, :, comp, :]  # physical gradient of component `comp` -> (n_quad, n_dof, n_dims)
                g = jnp.stack([gc[..., d] for d in dims], axis=-1)  # (n_quad, n_dof, len(dims))
                g = g[..., 0] if len(dims) == 1 else g
                if isinstance(field, TestFunction):
                    return g  # per-DOF directional derivative of the component
                return jnp.einsum("qn...,n->q...", g, cell_sol)  # contract the trial DOFs

        raise NotImplementedError("The FEM assembler supports gradients of TrialFunction/TestFunction only.")

    if isinstance(node, Hessian):
        dims = []
        for var in node.variables:
            if not isinstance(var, Variable):
                raise NotImplementedError(
                    "The FEM assembler expects Hessian variables to be domain.variable(...) placeholders."
                )
            if getattr(var, "axis", None) == "temporal":
                raise NotImplementedError(
                    "A temporal second derivative (u_tt) is handled by the second-order-in-time route, "
                    "not shape-Hessian assembly."
                )
            dims.append(var.dim[0])
        if len(dims) == 0:
            raise ValueError("Hessian node has no differentiation variables")
        if not isinstance(node.target, (TestFunction, TrialFunction)):
            raise NotImplementedError("The FEM assembler supports Hessians of TrialFunction/TestFunction only.")
        if _field_space(local, node.target) != "Lagrange":
            raise NotImplementedError("Second derivatives are assembled for nodal Lagrange fields only.")
        if _value_shape_num_components(getattr(node.target, "value_shape", ())) != 1:
            raise NotImplementedError("Hessian/Laplacian assembly currently supports scalar fields only.")
        hess = _field_hess(local, node.target)  # (n_quad, n_dof, dim, dim) physical shape Hessian
        if hess is None:
            raise NotImplementedError(
                "This element does not tabulate second derivatives -- use an order>=2 Lagrange field "
                "(a P1 Hessian is identically zero)."
            )
        da = jnp.asarray(dims)
        hsub = jnp.take(jnp.take(hess, da, axis=2), da, axis=3)  # (n_quad, n_dof, L, L) over requested dirs
        is_test = isinstance(node.target, TestFunction)
        if node.trace:  # Laplacian = sum over the selected diagonal directions
            lap = jnp.einsum("qnii->qn", hsub)  # (n_quad, n_dof) per-DOF Laplacian
            if is_test:
                return lap
            _, _, cell_sol = _field_data(local, node.target)
            return jnp.sum(lap[:, :, None] * cell_sol[None, :, :], axis=1)  # (n_quad, 1) trial Laplacian
        if is_test:
            return hsub  # (n_quad, n_dof, L, L) per-DOF Hessian (e.g. inner(hessian(u), hessian(v)))
        _, _, cell_sol = _field_data(local, node.target)
        return jnp.einsum("qnij,n->qij", hsub, cell_sol[:, 0])  # (n_quad, L, L) trial Hessian

    if isinstance(node, BinaryOp):
        a = _eval_integrand(domain, node.left, local)
        b = _eval_integrand(domain, node.right, local)
        a, b = _prefix_align(a, b)
        if node.op == "+":
            return a + b
        if node.op == "-":
            return a - b
        if node.op == "*":
            return a * b
        if node.op == "/":
            return a / b
        if node.op == "**":
            return a**b
        raise NotImplementedError(f"Unsupported binary operator: {node.op}")

    if isinstance(node, ModelCall):
        from .parametric_helpers import (
            _is_frozen_parameter,
            _is_runtime_scalar_parameter,
            _neural_coefficient_name,
            _parameter_name,
        )

        if _is_frozen_parameter(node):
            # A .freeze()d (known) coefficient: evaluate its constant / coordinate function at the
            # quadrature points -- exactly like jno.fn -- so it needs no runtime-parameter threading
            # and works in every assembly path (steady/transient/nonlinear/coupled).
            return _eval_frozen_coefficient(domain, node.model, local)
        if _is_runtime_scalar_parameter(node):
            name = _parameter_name(node)
            val = _runtime_parameter_value_from_internal_vars(local, name)
            if val is None:
                raise KeyError(
                    f"Runtime parameter '{name}' not supplied to the kernel. "
                    "Ensure it was registered in runtime_parameter_tags and packed "
                    "into InternalVars.volume_vars."
                )
            return val
        # Neural coefficient (``jno.nn.wrap(net)`` called inside the weak form): the assembler
        # threads a {name: module} table into ``local`` (trainable modules arrive through the
        # runtime ``args``; frozen ones fall back to their stored weights) and the network is
        # (re-)evaluated here on its kernel-evaluated arguments at the quadrature points.
        neural_modules = local.get("neural_coefficients")
        if neural_modules is not None:
            module = neural_modules.get(_neural_coefficient_name(node))
            if module is not None:
                arg_vals = [_eval_integrand(domain, a, local) for a in node.args]
                return _call_neural_coefficient(node.model, module, arg_vals, local)
        raise NotImplementedError(
            "jno.fem: a neural coefficient (jno.nn.wrap(net) inside the weak form) is supported on the "
            "native 2D/3D Lagrange assembler (single or coupled) and the scalar C¹ non-nodal families "
            "(Argyris/Morley/Hermite) — this assembly path does not thread network weights yet."
        )

    if isinstance(node, FunctionCall):
        args = [_eval_integrand(domain, arg, local) for arg in node.args]
        kwargs = node.kwargs if node.kwargs else {}
        return node.fn(*args, **kwargs)

    raise NotImplementedError(f"Unsupported weak-form node for the FEM assembler: {type(node).__name__}")


# --------------------------------
# Kernel builders
# --------------------------------


def _eval_volume_integrand(
    domain,
    expr,
    value_shape,
    cell_sol_flat,
    physical_quad_points,
    cell_shape_grads,
    cell_JxW,
    cell_v_grads_JxW,
    temporal_tags,
    runtime_parameter_tags,
    region_mask_names,
    problem_ref,
    *cell_internal_vars,
):
    """
    Evaluate and integrate one volume weak-form expression on one cell.

    Returns the flattened cell residual contribution the assembler expects.
    """
    num_nodes = cell_shape_grads.shape[1]
    vec = _value_shape_num_components(value_shape)
    cell_sol = cell_sol_flat.reshape(num_nodes, vec)

    problem = problem_ref["problem"]
    if problem is None:
        raise RuntimeError("The assembly problem_ref['problem'] was not initialized before kernel evaluation.")

    shape_vals = problem.fes[0].shape_vals
    local = {
        "physical_quad_points": physical_quad_points,
        "shape_vals": shape_vals,
        "shape_grads": cell_shape_grads,
        "cell_sol": cell_sol,
        "tag": "fem_gauss",
        "surface": False,
        "domain_context": domain.context,
        "trial_value_shape": value_shape,
        "trial_vec": vec,
        "temporal_tags": tuple(temporal_tags),
        "runtime_parameter_tags": tuple(runtime_parameter_tags),
        "region_mask_names": tuple(region_mask_names),
        "volume_vars": tuple(cell_internal_vars),
    }

    val = _eval_integrand(domain, expr, local)
    weights = cell_JxW[0]
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    weighted = val * weights.reshape(wshape)
    return ravel_pytree(jnp.sum(weighted, axis=0))[0]


def _lookup_boundary_normals(domain, tag, physical_surface_quad_points):
    """Outward normals at the face quad points for ``tag`` (or ``None``).

    Nearest-neighbour lookup against ``domain.normals_by_tag`` (keyed by the sampled
    ``gauss_<tag>`` or ``<tag>``), matching the single-field surface path. Shared by
    the single- and multi-field surface integrands so normal-dependent boundary terms
    (e.g. a pressure traction ``p_ext * n``) behave identically in coupled problems."""
    if not hasattr(domain, "normals_by_tag"):
        return None
    normal_lookup_tag = f"gauss_{tag}" if f"gauss_{tag}" in domain.normals_by_tag else tag
    if normal_lookup_tag in domain.normals_by_tag and normal_lookup_tag in getattr(domain, "_mesh_pool", {}):
        normal_pts = jnp.asarray(np.asarray(domain._mesh_pool[normal_lookup_tag])[:, : domain.dimension])
        normal_vals = jnp.asarray(np.asarray(domain.normals_by_tag[normal_lookup_tag])[:, : domain.dimension])
        if len(normal_pts) > 0 and len(normal_pts) == len(normal_vals):
            x_use = physical_surface_quad_points[:, : domain.dimension]
            d2 = jnp.sum((normal_pts[None, :, :] - x_use[:, None, :]) ** 2, axis=-1)
            nn_idx = jnp.argmin(d2, axis=1)
            return normal_vals[nn_idx]
    return None


def _eval_surface_integrand(
    domain,
    expr,
    tag,
    value_shape,
    cell_sol_flat,
    physical_surface_quad_points,
    face_shape_vals,
    face_shape_grads,
    face_nanson_scale,
    temporal_tags,
    runtime_parameter_tags,
    *cell_internal_vars_surface,
):
    """
    Evaluate and integrate one boundary weak-form expression on one face.

    Returns the flattened surface residual contribution the assembler expects.
    """
    vec = _value_shape_num_components(value_shape)

    cell_sol_flat = jnp.asarray(cell_sol_flat)
    physical_surface_quad_points = jnp.asarray(physical_surface_quad_points)
    face_shape_vals = jnp.asarray(face_shape_vals)
    face_shape_grads = jnp.asarray(face_shape_grads)
    face_nanson_scale = jnp.asarray(face_nanson_scale)

    if cell_sol_flat.ndim != 1:
        cell_sol_flat = cell_sol_flat.reshape(-1)

    if cell_sol_flat.size % vec != 0:
        raise ValueError(f"Surface kernel DOF size {cell_sol_flat.size} is not divisible by vec={vec} for tag '{tag}'.")

    n_parent_nodes = cell_sol_flat.size // vec
    cell_sol = cell_sol_flat.reshape(n_parent_nodes, vec)

    if face_shape_vals.ndim != 2:
        raise ValueError(f"Expected face_shape_vals.ndim == 2, got shape {face_shape_vals.shape} for tag '{tag}'.")
    if face_shape_grads.ndim != 3:
        raise ValueError(f"Expected face_shape_grads.ndim == 3, got shape {face_shape_grads.shape} for tag '{tag}'.")
    if physical_surface_quad_points.ndim != 2:
        raise ValueError(
            f"Expected physical_surface_quad_points.ndim == 2, got shape {physical_surface_quad_points.shape} for tag '{tag}'."
        )

    nq = face_shape_vals.shape[0]
    if face_shape_vals.shape[1] != n_parent_nodes:
        raise ValueError(
            f"Boundary shape/node mismatch on '{tag}': "
            f"face_shape_vals.shape={face_shape_vals.shape}, "
            f"but cell_sol implies n_parent_nodes={n_parent_nodes}."
        )
    if face_shape_grads.shape[0] != nq or face_shape_grads.shape[1] != n_parent_nodes:
        raise ValueError(
            f"Boundary grad shape mismatch on '{tag}': "
            f"face_shape_grads.shape={face_shape_grads.shape}, "
            f"expected (nq={nq}, n_parent_nodes={n_parent_nodes}, dim)."
        )
    if physical_surface_quad_points.shape[0] != nq:
        raise ValueError(
            f"Boundary quadrature mismatch on '{tag}': "
            f"physical_surface_quad_points.shape={physical_surface_quad_points.shape}, "
            f"face_shape_vals.shape={face_shape_vals.shape}."
        )

    if face_nanson_scale.ndim == 2:
        weights = face_nanson_scale[0]
    elif face_nanson_scale.ndim == 1:
        weights = face_nanson_scale
    else:
        raise ValueError(f"Unsupported face_nanson_scale shape {face_nanson_scale.shape} for tag '{tag}'.")

    if weights.shape[0] != nq:
        raise ValueError(f"Boundary weight/quadrature mismatch on '{tag}': weights.shape={weights.shape}, nq={nq}.")

    boundary_normals = _lookup_boundary_normals(domain, tag, physical_surface_quad_points)

    local = {
        "physical_quad_points": physical_surface_quad_points,
        "shape_vals": face_shape_vals,
        "shape_grads": face_shape_grads,
        "cell_sol": cell_sol,
        "tag": tag,
        "surface": True,
        "domain_context": domain.context,
        "trial_value_shape": value_shape,
        "trial_vec": vec,
        "boundary_normals": boundary_normals,
        "temporal_tags": tuple(temporal_tags),
        "volume_vars": tuple(cell_internal_vars_surface),
        "runtime_parameter_tags": tuple(runtime_parameter_tags),
    }

    val = _eval_integrand(domain, expr, local)
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    weighted = val * weights.reshape(wshape)
    return ravel_pytree(jnp.sum(weighted, axis=0))[0]


def _make_universal_volume_kernel(
    domain, expr, value_shape, temporal_tags, runtime_parameter_tags, region_mask_names, problem_ref
):
    """
    Create the universal volume kernel for a lowered weak-form expression.
    """

    def kernel(
        cell_sol_flat,
        physical_quad_points,
        cell_shape_grads,
        cell_JxW,
        cell_v_grads_JxW,
        *cell_internal_vars,
    ):
        return _eval_volume_integrand(
            domain,
            expr,
            value_shape,
            cell_sol_flat,
            physical_quad_points,
            cell_shape_grads,
            cell_JxW,
            cell_v_grads_JxW,
            temporal_tags,
            runtime_parameter_tags,
            region_mask_names,
            problem_ref,
            *cell_internal_vars,
        )

    return kernel


def _eval_multifield_volume_integrand(
    domain,
    term_list,
    fields,
    field_index,
    temporal_tags,
    runtime_parameter_tags,
    region_mask_names,
    problem_ref,
    cell_sol_flat,
    physical_quad_points,
    cell_shape_grads,
    cell_JxW,
    cell_v_grads_JxW,
    *cell_internal_vars,
):
    """Evaluate all coupled volume terms on one cell -> flat block-local residual.

    Splits the cell DOFs per field (``unflatten_fn_dof``), evaluates each additive
    term ``(coeff, test_field_index)`` with per-field shape data (reusing
    ``_eval_integrand``), and accumulates into its test field's residual slot.
    ``ravel_pytree`` concatenates the per-field residuals in field order (matching
    ``unflatten_fn_dof``) so the assembler autodiffs the full block matrix."""
    problem = problem_ref["problem"]
    if problem is None:
        raise RuntimeError("The assembly problem_ref['problem'] was not initialized before kernel evaluation.")

    cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
    nfields = len(fields)
    nnodes = [int(problem.fes[i].shape_vals.shape[1]) for i in range(nfields)]
    nc = [0]
    for n in nnodes:
        nc.append(nc[-1] + n)
    per_field = [
        {
            "shape_vals": problem.fes[i].shape_vals,
            "shape_grads": cell_shape_grads[:, nc[i] : nc[i + 1], :],
            "cell_sol": cell_sol_list[i],
        }
        for i in range(nfields)
    ]
    local = {
        "physical_quad_points": physical_quad_points,
        "fields": per_field,
        "field_index": field_index,
        "tag": "fem_gauss",
        "surface": False,
        "domain_context": domain.context,
        "temporal_tags": tuple(temporal_tags),
        "runtime_parameter_tags": tuple(runtime_parameter_tags),
        "region_mask_names": tuple(region_mask_names),
        "volume_vars": tuple(cell_internal_vars),
    }

    residuals = [jnp.zeros((nnodes[i] * int(fields[i]["vec"]),), dtype=cell_sol_flat.dtype) for i in range(nfields)]
    for coeff, test_idx in term_list:
        val = _eval_integrand(domain, coeff, local)
        weights = cell_JxW[test_idx]
        wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
        contrib = ravel_pytree(jnp.sum(val * weights.reshape(wshape), axis=0))[0]
        residuals[test_idx] = residuals[test_idx] + contrib
    return ravel_pytree(residuals)[0]


def _make_multifield_volume_kernel(
    domain, term_list, fields, field_index, temporal_tags, runtime_parameter_tags, region_mask_names, problem_ref
):
    """Universal volume kernel for a coupled (multi-field) weak form."""

    def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
        return _eval_multifield_volume_integrand(
            domain,
            term_list,
            fields,
            field_index,
            temporal_tags,
            runtime_parameter_tags,
            region_mask_names,
            problem_ref,
            cell_sol_flat,
            physical_quad_points,
            cell_shape_grads,
            cell_JxW,
            cell_v_grads_JxW,
            *cell_internal_vars,
        )

    return kernel


def _eval_multifield_surface_integrand(
    domain,
    term_list,
    fields,
    field_index,
    tag,
    temporal_tags,
    runtime_parameter_tags,
    problem_ref,
    cell_sol_flat,
    physical_surface_quad_points,
    face_shape_vals,
    face_shape_grads,
    face_nanson_scale,
    *cell_internal_vars_surface,
):
    """Evaluate all coupled boundary terms on one face -> flat block-local residual.

    The surface analogue of :func:`_eval_multifield_volume_integrand`. The kernel receives the
    full multi-field parent-cell DOFs and the per-field face shape data concatenated
    along the node axis (assembler concatenates in ``problem.fes`` order, identical to
    the volume ``shape_grads``), so we split DOFs with ``unflatten_fn_dof`` and slice the
    face shape arrays per field. Each term ``(coeff, test_field_index)`` accumulates into
    its test field's residual slot; ``ravel_pytree`` concatenates in field order so the assembler
    autodiffs the surface contribution into the right block(s) of the load and matrix."""
    problem = problem_ref["problem"]
    if problem is None:
        raise RuntimeError("The assembly problem_ref['problem'] was not initialized before kernel evaluation.")

    cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
    nfields = len(fields)
    nnodes = [int(problem.fes[i].shape_vals.shape[1]) for i in range(nfields)]
    nc = [0]
    for n in nnodes:
        nc.append(nc[-1] + n)
    per_field = [
        {
            "shape_vals": face_shape_vals[:, nc[i] : nc[i + 1]],
            "shape_grads": face_shape_grads[:, nc[i] : nc[i + 1], :],
            "cell_sol": cell_sol_list[i],
        }
        for i in range(nfields)
    ]
    local = {
        "physical_quad_points": physical_surface_quad_points,
        "fields": per_field,
        "field_index": field_index,
        "tag": tag,
        "surface": True,
        "domain_context": domain.context,
        "boundary_normals": _lookup_boundary_normals(domain, tag, jnp.asarray(physical_surface_quad_points)),
        "temporal_tags": tuple(temporal_tags),
        "runtime_parameter_tags": tuple(runtime_parameter_tags),
        "volume_vars": tuple(cell_internal_vars_surface),
    }

    face_nanson_scale = jnp.asarray(face_nanson_scale)
    weights = face_nanson_scale[0] if face_nanson_scale.ndim == 2 else face_nanson_scale

    residuals = [jnp.zeros((nnodes[i] * int(fields[i]["vec"]),), dtype=cell_sol_flat.dtype) for i in range(nfields)]
    for coeff, test_idx in term_list:
        val = _eval_integrand(domain, coeff, local)
        wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
        contrib = ravel_pytree(jnp.sum(val * weights.reshape(wshape), axis=0))[0]
        residuals[test_idx] = residuals[test_idx] + contrib
    return ravel_pytree(residuals)[0]


def _make_multifield_surface_kernel(
    domain, term_list, fields, field_index, tag, temporal_tags, runtime_parameter_tags, problem_ref
):
    """Universal surface kernel for the coupled boundary terms on one tag."""

    def kernel(
        cell_sol_flat,
        physical_surface_quad_points,
        face_shape_vals,
        face_shape_grads,
        face_nanson_scale,
        *cell_internal_vars_surface,
    ):
        return _eval_multifield_surface_integrand(
            domain,
            term_list,
            fields,
            field_index,
            tag,
            temporal_tags,
            runtime_parameter_tags,
            problem_ref,
            cell_sol_flat,
            physical_surface_quad_points,
            face_shape_vals,
            face_shape_grads,
            face_nanson_scale,
            *cell_internal_vars_surface,
        )

    return kernel


def _make_universal_surface_kernel(domain, expr, tag, value_shape, runtime_parameter_tags, temporal_tags):
    def kernel(
        cell_sol_flat,
        physical_surface_quad_points,
        face_shape_vals,
        face_shape_grads,
        face_nanson_scale,
        *cell_internal_vars_surface,
    ):
        return _eval_surface_integrand(
            domain,
            expr,
            tag,
            value_shape,
            cell_sol_flat,
            physical_surface_quad_points,
            face_shape_vals,
            face_shape_grads,
            face_nanson_scale,
            temporal_tags,
            runtime_parameter_tags,
            *cell_internal_vars_surface,
        )

    return kernel


# --------------------------------
# Problem assembly
# --------------------------------


def _meshio_type_for_element(element_type: str) -> str:
    meshio_type_map = {
        "TRI3": "triangle",
        "TRI6": "triangle6",
        "QUAD4": "quad",
        "QUAD8": "quad8",
        "TET4": "tetra",
        "TET10": "tetra10",
        "HEX8": "hexahedron",
        "HEX20": "hexahedron20",
        "HEX27": "hexahedron27",
    }
    if element_type not in meshio_type_map:
        raise KeyError(f"Unsupported element type '{element_type}'.")
    return meshio_type_map[element_type]


# Quadratic (P2) meshio cell type -> (linear source type, per-element edge node list
# in meshio ordering). jno's domain machinery assumes linear cells, so the domain mesh
# stays linear (P1) and we promote *only* the assembly mesh here: insert edge-midpoint
# nodes, vertices preserved (so a P1 field on a P2 problem -- e.g. Taylor-Hood pressure
# -- is just the vertex block).
_P2_FROM_P1 = {
    "triangle6": ("triangle", [(0, 1), (1, 2), (2, 0)]),
    "tetra10": ("tetra", [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]),
}


def _promote_to_quadratic(points, cells_p1, edge_local):
    """Promote a linear simplex mesh to quadratic (P1 -> P2).

    Inserts one node at each *unique* edge midpoint (canonical sorted-vertex-pair
    key, so an edge shared by two cells maps to a single node), preserving the
    original vertices (indices ``0..nverts-1``) and appending the edge nodes.
    Returns ``(points_p2, cells_p2)`` with cells in meshio ``*6``/``*10`` ordering
    (corners first, then ``edge_local`` midpoints)."""
    points = np.asarray(points)
    cells_p1 = np.asarray(cells_p1)
    ncorner = cells_p1.shape[1]
    edge_map: dict[tuple[int, int], int] = {}
    edge_nodes: List[Any] = []
    cells_p2 = np.zeros((cells_p1.shape[0], ncorner + len(edge_local)), dtype=np.int64)
    cells_p2[:, :ncorner] = cells_p1
    nid = points.shape[0]
    for c in range(cells_p1.shape[0]):
        for k, (i, j) in enumerate(edge_local):
            a, b = int(cells_p1[c, i]), int(cells_p1[c, j])
            key = (a, b) if a < b else (b, a)
            n = edge_map.get(key)
            if n is None:
                n = nid
                edge_map[key] = nid
                edge_nodes.append(0.5 * (points[a] + points[b]))
                nid += 1
            cells_p2[c, ncorner + k] = n
    pts = np.vstack([points, np.asarray(edge_nodes)]) if edge_nodes else points
    return pts, cells_p2


def _promote_to_degree(points, cells_p1, ref_pts):
    """Promote a linear simplex mesh to a degree-``k`` Lagrange node mesh (P1 -> P{k}, any ``k``).

    ``ref_pts`` are the element's reference interpolation points in **basix DOF order** (shape
    ``(n_dof, tdim)``; the first ``ncorner`` are the cell vertices). Each cell's nodes are the affine
    image of ``ref_pts`` through that cell's P1 geometry (a barycentric combination of its vertices);
    nodes are deduplicated by **physical coordinate** (a scale-aware grid hash). So an edge/face shared
    by two cells collapses to one global node *regardless of the cells' local orientation* -- for C0
    Lagrange a DOF is a point value, so coordinate coincidence on a shared entity is exactly the
    conformity condition (no orientation sign or per-edge node ordering, unlike RT/Nedelec; this is what
    lets a single midpoint suffice at P2 and a clean generalisation hold at P3+). Original vertices keep
    ids ``0..nv-1`` (so a P1 field on a P{k} problem is the leading vertex block, e.g. Taylor-Hood
    pressure). Returns ``(points_k, cells_k)`` with ``cells_k`` columns in the same DOF order as
    ``ref_pts`` -- hence as the basis tabulated from the same basix element."""
    points = np.asarray(points, dtype=float)
    cells_p1 = np.asarray(cells_p1)
    ref_pts = np.asarray(ref_pts, dtype=float)
    ncell, ncorner = cells_p1.shape
    ndof = ref_pts.shape[0]
    # barycentric weights of each reference point: l0 = 1 - sum(xi), l_i = xi_i
    bary = np.empty((ndof, ncorner), dtype=float)
    bary[:, 0] = 1.0 - ref_pts.sum(axis=1)
    bary[:, 1:] = ref_pts
    phys = np.einsum("dc,ncg->ndg", bary, points[cells_p1])  # (ncell, ndof, gdim) physical node coords
    # scale-aware coordinate hash: coincident nodes differ only by FP roundoff (<< tol); distinct nodes
    # are separated by ~mesh spacing (>> tol).
    extent = float(np.max(points.max(axis=0) - points.min(axis=0))) if points.shape[0] else 1.0
    tol = 1e-7 * (extent or 1.0)

    def _key(p):
        return tuple(np.round(np.asarray(p) / tol).astype(np.int64))

    coord_to_gid: dict = {}
    out_pts: List[Any] = []
    for vid in range(points.shape[0]):  # seed with the original vertices so they keep ids 0..nv-1
        coord_to_gid[_key(points[vid])] = vid
        out_pts.append(points[vid])
    cells_k = np.zeros((ncell, ndof), dtype=np.int64)
    nid = points.shape[0]
    for c in range(ncell):
        for d in range(ndof):
            k = _key(phys[c, d])
            gid = coord_to_gid.get(k)
            if gid is None:
                gid, coord_to_gid[k] = nid, nid
                out_pts.append(phys[c, d])
                nid += 1
            cells_k[c, d] = gid
    return np.asarray(out_pts), cells_k


def _zero_mass_dirichlet_rows(M, bc):
    """Zero a mass matrix's Dirichlet **rows** so ``M u̇ + A u = c`` reads ``u[d]=g``.

    Symmetric Dirichlet (identity rows) is applied to *every* assembled matrix — correct
    for a stiffness operator, but wrong for the **mass**: a constrained DOF must carry no
    time derivative. Zeroing the Dirichlet rows (``A``'s Dirichlet rows are identity, ``c``
    carries ``g``) makes the Dirichlet row of ``(M + dt A) w = M w_old + dt c`` reduce to
    ``u[d] = g``. The Dirichlet *columns* are deliberately kept: they couple a free row to a
    constrained DOF's time derivative (``M_fd·ġ``), which the stepper's ``M(w_new−w_old)``
    captures — essential for **time-varying** Dirichlet and harmless when ``ġ=0`` (the
    constant case). So M is asymmetric here, by design."""
    rows = None if bc is None else getattr(bc, "bc_rows", None)
    if rows is None:
        return M
    rows = jnp.asarray(rows).reshape(-1)
    if rows.shape[0] == 0:
        return M
    return jnp.asarray(M).at[rows, :].set(0.0)


def _zero_mass_dirichlet_rows_sparse(M, bc):
    """BCOO version of :func:`_zero_mass_dirichlet_rows` — zeros the entries in Dirichlet rows
    without densifying, so the transient mass stays sparse (matrix-free matvec, O(nnz) memory)."""
    rows = None if bc is None else getattr(bc, "bc_rows", None)
    if rows is None:
        return M
    rows = jnp.asarray(rows).reshape(-1)
    if rows.shape[0] == 0:
        return M
    keep = jnp.logical_not(jnp.isin(M.indices[:, 0], rows)).astype(M.data.dtype)
    return type(M)((M.data * keep, M.indices), shape=M.shape)


def _zero_forcing_dirichlet_rows(forcing_fn, bc):
    """Wrap a forcing callback ``f(t)`` to zero its Dirichlet rows.

    The transient forcing is the *raw* source load (assembled without Dirichlet
    elimination), so it has entries at constrained DOFs. The Dirichlet rows must read
    ``u[d]=g`` from the load ``c`` alone — a source contributes only to free DOFs — so the
    forcing is zeroed there before it enters the stepper ``M w_old + dt·(c + f(t))``."""
    rows = None if bc is None else getattr(bc, "bc_rows", None)
    if forcing_fn is None or rows is None:
        return forcing_fn
    rows = jnp.asarray(rows).reshape(-1)
    if rows.shape[0] == 0:
        return forcing_fn

    def cleaned(t, args=None):
        return jnp.asarray(forcing_fn(t, args)).reshape(-1).at[rows].set(0.0)

    return cleaned


def _dense_array(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    try:
        return jnp.asarray(A)
    except Exception:
        return jnp.asarray(np.asarray(A))


def _region_mask_arrays_for_domain(domain):
    """Per-cell sub-region masks for the regions this domain's FEM problem uses, in the kernel's fixed
    (sorted) order -- the order ``_build_fem_problem`` recorded in ``domain._fem_region_mask_order``.

    Each is a constant ``(num_cells, 1)`` 0/1 array (the kernel slices a scalar per cell). Cached per region
    on the domain (depends only on the mesh + region geometry), so transient steps and parametric
    re-evaluations don't rebuild them. Returns ``()`` when the problem has no sub-region terms."""
    names = tuple(getattr(domain, "_fem_region_mask_order", ()) or ())
    if not names:
        return ()
    a_cells = getattr(domain, "_fem_assembly_cells", None)
    n_cells = int(np.asarray(a_cells).shape[0]) if a_cells is not None else None
    cache = getattr(domain, "_fem_region_mask_cache", None)
    if cache is None or cache.get("__ncells__") != n_cells:  # invalidate if the assembly mesh changed
        cache = {"__ncells__": n_cells}
        domain._fem_region_mask_cache = cache
    out = []
    for r in names:
        if r not in cache:
            cache[r] = jnp.asarray(_cell_region_mask(domain, r), dtype=_default_float_dtype()).reshape(-1, 1)
        out.append(cache[r])
    return tuple(out)


def _make_internal_vars(
    fe_module,
    temporal_tags,
    t,
    *,
    n_cells: int,
    dtype=None,
    runtime_parameter_tags=(),
    runtime_parameter_values=None,
    region_mask_arrays=(),
    extra_volume_vars=(),
):
    """
    Build the batched volume-variable arrays the kernel slices per cell.

    Volume-var layout (the order every kernel's ``local`` indexing assumes):
    ``[ temporal ... , runtime_parameter ... , region_mask ... , extra ... ]``. Each temporal variable
    is broadcast to shape (n_cells, 1); region masks are constant per-cell arrays (see
    ``_region_mask_arrays_for_domain``) so any path that threads them gets per-region integration.
    """
    vol = []

    if temporal_tags:
        t0 = jnp.asarray(t, dtype=dtype)
        t_batched = jnp.full((int(n_cells), 1), t0, dtype=t0.dtype)
        vol.extend([t_batched for _ in temporal_tags])
    # Parameter values, in runtime_parameter_tags order, broadcast per cell.
    rpv = runtime_parameter_values or {}
    for name in runtime_parameter_tags:
        p = jnp.asarray(rpv[name], dtype=dtype)
        flat = p.reshape(-1)
        if flat.shape[0] == 1:
            # scalar parameter -> same value in every cell (broadcast to quad in-kernel)
            vol.append(jnp.full((int(n_cells), 1), flat[0], dtype=p.dtype))
        else:
            # field parameter (node- or cell-based) -> pass the global array through;
            # the per-cell gather slices it, the kernel interpolates.
            vol.append(flat)
    # Sub-region masks, after the runtime parameters (matches the evaluator's RegionMask index).
    for m in region_mask_arrays:
        vol.append(jnp.asarray(m, dtype=dtype).reshape(int(n_cells), 1))
    for v in extra_volume_vars:
        arr = jnp.asarray(v, dtype=dtype)
        if arr.ndim == 0:
            arr = jnp.full((int(n_cells), 1), arr, dtype=arr.dtype)
        vol.append(arr)

    return fe_module.InternalVars(volume_vars=tuple(vol))


def _periodic_facet_weights(
    t_query: np.ndarray,
    facet_node_ids: np.ndarray,
    pts: np.ndarray,
    transverse: List[int],
) -> List[Tuple[int, float]] | None:
    """Interpolation weights for a slave at transverse coord ``t_query`` on the
    master boundary facets (node-to-segment / mortar-lite identification).

    ``facet_node_ids`` is ``(n_facets, k)`` of global node ids. **2D** (transverse
    1-D): facets are edges -- columns 0,1 are the vertices, optional column 2 the
    midside node (``k == 3`` ⇒ P2). **3D** (transverse 2-D): facets are triangles --
    columns 0,1,2 the vertices, optional columns 3,4,5 the edge midpoints (``k == 6``
    ⇒ P2). Returns ``[(node_id, weight), ...]`` whose weights sum to 1 (partition of
    unity ⇒ constants reproduced; linear/quadratic-on-the-facet reproduced exactly).
    """
    tq = np.atleast_1d(np.asarray(t_query, dtype=float))
    facet_node_ids = np.asarray(facet_node_ids, dtype=int)
    if facet_node_ids.ndim != 2 or facet_node_ids.shape[0] == 0:
        return None
    k = facet_node_ids.shape[1]

    if tq.shape[0] == 1:  # 2D: locate the master edge spanning the slave's transverse coord
        t = float(tq[0])
        tr = transverse[0]
        a_ids, b_ids = facet_node_ids[:, 0], facet_node_ids[:, 1]
        ta, tb = pts[a_ids, tr], pts[b_ids, tr]
        lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
        span = hi - lo
        eps = 1.0e-9 * (float(np.max(span)) if span.size else 1.0)
        inside = (t >= lo - eps) & (t <= hi + eps)
        if inside.any():
            idx = int(np.argmax(inside))
        else:  # outside every edge (rounding at a face end) -> nearest edge
            idx = int(np.argmin(np.minimum(np.abs(t - lo), np.abs(t - hi))))
        a, b = int(a_ids[idx]), int(b_ids[idx])
        L = float(tb[idx] - ta[idx])
        xi = 0.0 if abs(L) < eps else (t - float(ta[idx])) / L  # local coord, a:0 -> b:1
        xi = min(1.0, max(0.0, xi))
        if k < 3:  # P1 edge: linear
            return [(a, 1.0 - xi), (b, xi)]
        m = int(facet_node_ids[idx, 2])  # P2 edge (a, b, mid): quadratic Lagrange at xi = 0, 1, 0.5
        return [(a, 2.0 * (xi - 0.5) * (xi - 1.0)), (b, 2.0 * xi * (xi - 0.5)), (m, -4.0 * xi * (xi - 1.0))]

    if tq.shape[0] == 2:  # 3D: locate the master triangle containing the slave, barycentric weights
        tr = transverse
        pa = pts[facet_node_ids[:, 0]][:, tr]
        pb = pts[facet_node_ids[:, 1]][:, tr]
        pc = pts[facet_node_ids[:, 2]][:, tr]
        v0, v1, v2 = pb - pa, pc - pa, tq[None, :] - pa
        d00 = (v0 * v0).sum(1)
        d01 = (v0 * v1).sum(1)
        d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1)
        d21 = (v2 * v1).sum(1)
        denom = d00 * d11 - d01 * d01
        denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
        l1 = (d11 * d20 - d01 * d21) / denom  # vertex b
        l2 = (d00 * d21 - d01 * d20) / denom  # vertex c
        l0 = 1.0 - l1 - l2  # vertex a
        # the containing triangle (all barycentrics >= 0); else the least-violating one (shared edge / rounding)
        viol = np.maximum(0.0, -l0) + np.maximum(0.0, -l1) + np.maximum(0.0, -l2)
        idx = int(np.argmin(viol))
        a, b, c = (int(facet_node_ids[idx, j]) for j in range(3))
        L0, L1, L2 = float(l0[idx]), float(l1[idx]), float(l2[idx])
        if k < 6:  # P1 triangle: barycentric
            return [(a, L0), (b, L1), (c, L2)]
        # P2 triangle (a, b, c, mab, mbc, mca): quadratic shape functions in barycentric coords
        mab, mbc, mca = (int(facet_node_ids[idx, j]) for j in range(3, 6))
        return [
            (a, L0 * (2.0 * L0 - 1.0)),
            (b, L1 * (2.0 * L1 - 1.0)),
            (c, L2 * (2.0 * L2 - 1.0)),
            (mab, 4.0 * L0 * L1),
            (mbc, 4.0 * L1 * L2),
            (mca, 4.0 * L2 * L0),
        ]
    return None

    raise NotImplementedError("3D periodic interpolation (triangle facets) is milestone M2.")


def build_periodic_prolongation(
    points: np.ndarray,
    pairs: Sequence[Tuple[str, str]],
    tag_indices: Dict[str, np.ndarray],
    *,
    vec: int = 1,
    tol: float | None = None,
    facets: Dict[str, np.ndarray] | None = None,
    phases: Sequence[complex] | None = None,
) -> Dict[str, object]:
    """Build the node-level periodic prolongation matrix ``P``.

    Identifies each slave-face node with the master face. When the two faces have
    the **same node layout** (structured / conforming) this is an exact 0/1
    node-to-node map. When they **don't** (unstructured / non-matching), a slave
    that has no master node within ``tol`` is instead tied to the master *facet*
    it lands on by **node-to-segment interpolation** (linear for P1, quadratic for
    P2) — master–slave MPC elimination, consistent (partition of unity) though not
    a full dual-mortar coupling. See :func:`_periodic_facet_weights`.

    Parameters
    ----------
    points:
        ``(n_nodes, dim)`` array of FEM node coordinates (the assembly mesh).
    pairs:
        Ordered ``(master_tag, slave_tag)`` boundary pairings, e.g.
        ``[("left", "right"), ("bottom", "top")]``.
    tag_indices:
        Mapping from boundary tag name to the global node ids on that tag.
    vec:
        Number of scalar components per node. ``vec > 1`` returns the
        node-major expansion ``kron(P_node, I_vec)``.
    tol:
        Coordinate-matching tolerance for the transverse coordinates. When
        ``None`` it is derived from the bounding-box diagonal.
    facets:
        Optional ``{master_tag: (n_facets, k) node-id array}`` of the master
        boundary facets, required only for the non-matching (interpolatory) path.

    Returns
    -------
    dict with keys:
        ``P``               : ``(n_full, n_red)`` dense jnp prolongation matrix.
        ``P_node``          : node-level prolongation (``vec == 1`` form).
        ``kept_nodes``      : sorted global ids of the retained (master/free) nodes.
        ``slave_to_master`` : resolved slave-node -> master-node mapping (exact ties).
        ``n_full``          : full node count.
        ``n_red``           : reduced node count.
        ``vec``             : component count used.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_nodes = pts.shape[0]
    facets = facets or {}
    if phases is None:
        phases = [1.0] * len(pairs)
    # Bloch/quasi-periodic: a non-unit phase makes P complex; a plain periodic cell keeps P real 0/1
    # so the fast (selection) reduction path is untouched.
    is_bloch = any(abs(complex(p) - 1.0) > 1e-12 for p in phases)

    if tol is None:
        span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        tol = max(span, 1.0) * 1.0e-6

    # slave -> master node (exact tie, weight = Bloch phase); slave -> [(master node, weight)] (interp).
    slave_to_master: Dict[int, int] = {}
    slave_phase: Dict[int, complex] = {}
    slave_interp: Dict[int, List[Tuple[int, float]]] = {}

    for (master_tag, slave_tag), ph in zip(pairs, phases):
        if master_tag not in tag_indices or slave_tag not in tag_indices:
            raise KeyError(
                f"Periodic pair ({master_tag!r}, {slave_tag!r}) refers to a tag "
                f"that is not present in the mesh. Known tags: {sorted(tag_indices)}."
            )

        m_ids = np.asarray(tag_indices[master_tag], dtype=int).reshape(-1)
        s_ids = np.asarray(tag_indices[slave_tag], dtype=int).reshape(-1)
        m_pts = pts[m_ids]
        s_pts = pts[s_ids]

        # Periodic axis = coordinate whose tag means differ the most.
        axis = int(np.argmax(np.abs(m_pts.mean(axis=0) - s_pts.mean(axis=0))))
        transverse = [d for d in range(pts.shape[1]) if d != axis]

        m_trans = m_pts[:, transverse]
        s_trans = s_pts[:, transverse]

        # Nearest transverse master node for every slave node.
        d2 = np.sum((s_trans[:, None, :] - m_trans[None, :, :]) ** 2, axis=-1)
        nn = np.argmin(d2, axis=1) if m_ids.size else np.zeros(len(s_ids), dtype=int)
        dist = np.sqrt(d2[np.arange(len(s_ids)), nn]) if m_ids.size and len(s_ids) else np.zeros(len(s_ids))

        for k, sid in enumerate(s_ids):
            if m_ids.size and dist[k] <= tol:  # conforming: exact node-to-node (corners land here too)
                slave_to_master[int(sid)] = int(m_ids[nn[k]])
                slave_phase[int(sid)] = complex(ph)
                continue
            # non-matching: tie to the master facet by interpolation (weights scaled by the Bloch phase)
            w = _periodic_facet_weights(s_trans[k], facets.get(master_tag), pts, transverse) if facets else None
            if w is None:
                raise ValueError(
                    f"Periodic matching for ({master_tag!r}, {slave_tag!r}) failed at slave node {int(sid)}: "
                    f"nearest master node is {float(dist[k]):.3e} away (tol {tol:.3e}) and no master facet "
                    "connectivity was supplied for interpolation. Pass `facets=` (unstructured) or use a "
                    "conforming mesh."
                )
            slave_interp[int(sid)] = [(int(m), complex(ph) * wt) for (m, wt) in w]

    slave_set = set(slave_to_master) | set(slave_interp)

    # Each slave is a linear combination of other nodes (exact: one master, weight 1; interpolated:
    # facet shape-function weights). Those nodes may themselves be slaves — a corner is a slave in
    # several directions, and an interpolation can land on a master edge whose endpoint is itself a
    # slave — so resolve every slave **transitively** to kept (master) nodes. This handles any number
    # of periodic directions (e.g. a doubly-periodic cell) with a single general mechanism.
    raw: Dict[int, List[Tuple[int, complex]]] = {
        sid: [(m, slave_phase.get(sid, 1.0))] for sid, m in slave_to_master.items()
    }
    raw.update({sid: list(ws) for sid, ws in slave_interp.items()})
    resolved: Dict[int, Dict[int, float]] = {}

    def _expand(node: int, stack: frozenset) -> Dict[int, float]:
        if node not in slave_set:
            return {node: 1.0}
        if node in resolved:
            return resolved[node]
        if node in stack:
            raise ValueError(f"Periodic identification is cyclic at node {node}; check the tie directions.")
        down = stack | {node}
        out: Dict[int, float] = {}
        for n2, w in raw[node]:
            for kept_node, wk in _expand(n2, down).items():
                out[kept_node] = out.get(kept_node, 0.0) + w * wk
        resolved[node] = out
        return out

    kept_nodes: List[int] = [i for i in range(n_nodes) if i not in slave_set]
    reduced_index = {node: r for r, node in enumerate(kept_nodes)}
    n_red = len(kept_nodes)

    # The prolongation is a SELECTION matrix (0/1, plus interpolation weights for non-matching faces)
    # -- it must be SPARSE, never dense: O(n_full) nonzeros vs O(n_full * n_red) dense (GBs at large
    # N). Collect (row, col, weight) entries node-by-node, then a BCOO; a vector field expands each
    # node entry to ``vec`` component entries (a Kronecker of the node map with I_vec).
    import jax.experimental.sparse as jsparse

    rows: List[int] = []
    cols: List[int] = []
    data: List[complex] = []
    for i in range(n_nodes):
        if i in slave_set:
            for kept_node, weight in _expand(i, frozenset()).items():
                rows.append(i)
                cols.append(reduced_index[kept_node])
                data.append(weight if is_bloch else float(np.real(weight)))
        else:
            rows.append(i)
            cols.append(reduced_index[i])
            data.append(1.0)

    # Informational exact-chain map (single kept master per exact slave); interpolated slaves omitted.
    final_master = {sid: next(iter(_expand(sid, frozenset()))) for sid in slave_to_master}

    rows_a = np.asarray(rows, dtype=np.int64)
    cols_a = np.asarray(cols, dtype=np.int64)
    data_a = np.asarray(data, dtype=np.complex128 if is_bloch else np.float64)
    if vec > 1:  # kron(P_node, I_vec): each node entry -> vec component entries
        comp = np.arange(vec, dtype=np.int64)
        rows_a = (rows_a[:, None] * vec + comp[None, :]).reshape(-1)
        cols_a = (cols_a[:, None] * vec + comp[None, :]).reshape(-1)
        data_a = np.repeat(data_a, vec)
    n_full, n_red_full = int(n_nodes * vec), int(n_red * vec)
    P = jsparse.BCOO((jnp.asarray(data_a), jnp.asarray(np.stack([rows_a, cols_a], axis=1))), shape=(n_full, n_red_full))

    return {
        "P": P,
        "P_node": P,  # sparse; equals the node-level map when vec == 1
        "kept_nodes": np.asarray(kept_nodes, dtype=np.int64),
        "slave_to_master": final_master,
        "n_full": n_full,
        "n_red": n_red_full,
        "vec": int(vec),
        # whether P is a one-master-per-slave selection (conforming) -> the sparse remap reduction is
        # exact; computed once here (eager, concrete P) so the reduce path never inspects P under trace.
        "is_selection": _is_selection(P),
        # Bloch/quasi-periodic: P is complex, so the reduction is Hermitian (P^H A P) and the reduced
        # complex system can't be split into independent real/imag legs.
        "is_bloch": bool(is_bloch),
    }


def build_periodic_prolongation_nonnodal(
    n_verts: int,
    n_edges: int,
    vertex_points: np.ndarray,
    edge_midpoints: np.ndarray,
    edge_normals: np.ndarray,
    vtags: Dict[str, np.ndarray],
    etags: Dict[str, np.ndarray],
    pairs: Sequence[Tuple[str, str]],
    *,
    tol: float | None = None,
) -> Dict[str, object]:
    """DOF-level periodic prolongation for a **Morley** (C¹ non-nodal) field.

    Morley DOFs: one *value* DOF per vertex at ids ``[0, n_verts)``; one *normal-derivative* DOF per
    global edge at ids ``[n_verts, n_verts + n_edges)``. The node-based ``build_periodic_prolongation``
    can't represent this (it assumes DOFs = points × components). Here the two DOF blocks are tied
    separately and block-diagonally combined:

    * **value DOFs** — tie by vertex coordinate, weight ``+1`` (delegated to the nodal builder on the
      vertex points, so corners / transitive chains are handled identically to Lagrange);
    * **edge normal-derivative DOFs** — tie boundary-edge → boundary-edge by *midpoint*, weight
      ``sign(n_slave · n_master)`` where each edge's normal is its globally-oriented reference normal
      (``fem_nonnodal``: ``n = R90·(P[hi] − P[lo])``). For axis-aligned periodic boundaries this dot is
      ``±1`` — the tie **sign is derived from geometry, not assumed** (and gated by an MMS test).

    Non-conforming periodic boundaries (a slave edge with no transverse-matching master edge) raise a
    clear error rather than silently mis-coupling (mortar interpolation of derivative DOFs is out of
    scope). Returns the same dict shape as :func:`build_periodic_prolongation`.
    """
    import jax.experimental.sparse as jsparse

    # --- value-DOF block: reuse the nodal builder on the vertex points (weight +1) ---
    vred = build_periodic_prolongation(np.asarray(vertex_points, dtype=np.float64), pairs, vtags, vec=1, tol=tol)
    Pv = vred["P"]
    n_vred = int(vred["n_red"])
    vkept = np.asarray(vred["kept_nodes"], dtype=np.int64)

    # --- edge-derivative block: signed boundary-edge ties by midpoint ---
    emid = np.asarray(edge_midpoints, dtype=np.float64).reshape(n_edges, -1)
    enrm = np.asarray(edge_normals, dtype=np.float64).reshape(n_edges, -1)
    span = float(np.linalg.norm(emid.max(0) - emid.min(0))) if n_edges else 1.0
    etol = tol if tol is not None else max(span, 1.0) * 1.0e-6
    e_slave_master: Dict[int, Tuple[int, float]] = {}
    for mtag, stag in pairs:
        me = np.asarray(etags.get(mtag, []), dtype=int).reshape(-1)
        se = np.asarray(etags.get(stag, []), dtype=int).reshape(-1)
        if me.size == 0 or se.size == 0:
            continue
        mp, sp = emid[me], emid[se]
        axis = int(np.argmax(np.abs(mp.mean(0) - sp.mean(0))))
        trans = [d for d in range(mp.shape[1]) if d != axis]
        d2 = np.sum((sp[:, None, :][:, :, trans] - mp[None, :, :][:, :, trans]) ** 2, axis=-1)
        nn = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(se)), nn])
        for kk, sid in enumerate(se):
            if dist[kk] > etol:
                raise ValueError(
                    f"jno.fem periodic (non-nodal C¹): slave edge {int(sid)} on {stag!r} has no "
                    f"transverse-matching master edge on {mtag!r} (nearest {dist[kk]:.3e} > tol {etol:.3e}). "
                    "Periodic C¹ needs a conforming (matching) periodic boundary; non-matching edge "
                    "coupling (mortar interpolation of derivative DOFs) is not supported."
                )
            mid = int(me[nn[kk]])
            dot = float(np.dot(enrm[int(sid)], enrm[mid]))
            e_slave_master[int(sid)] = (mid, 1.0 if dot >= 0.0 else -1.0)

    def _resolve_edge(e: int) -> Tuple[int, float]:
        seen: set = set()
        sign = 1.0
        while e in e_slave_master:
            if e in seen:
                raise ValueError(f"jno.fem periodic (non-nodal C¹): cyclic edge tie at edge {e}.")
            seen.add(e)
            e, s = e_slave_master[e]
            sign *= s
        return e, sign

    ekept = [e for e in range(n_edges) if e not in e_slave_master]
    e_red_idx = {e: r for r, e in enumerate(ekept)}
    n_ered = len(ekept)

    # --- combined DOF-level BCOO  [[Pv, 0], [0, Pe]]  (edge rows/cols offset past the value block) ---
    Pv_idx = np.asarray(Pv.indices)
    Pv_dat = np.asarray(Pv.data)
    rows = list(Pv_idx[:, 0])
    cols = list(Pv_idx[:, 1])
    data = list(Pv_dat)
    for e in range(n_edges):
        m, sign = _resolve_edge(e)
        rows.append(n_verts + e)
        cols.append(n_vred + e_red_idx[m])
        data.append(sign)
    n_full = n_verts + n_edges
    n_red = n_vred + n_ered
    P = jsparse.BCOO(
        (
            jnp.asarray(np.asarray(data, dtype=np.float64)),
            jnp.asarray(np.stack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)], axis=1)),
        ),
        shape=(n_full, n_red),
    )
    kept = np.concatenate([vkept, n_verts + np.asarray(ekept, dtype=np.int64)]) if n_ered else vkept
    return {
        "P": P,
        "P_node": P,
        "kept_nodes": kept,
        "slave_to_master": {},
        "n_full": n_full,
        "n_red": n_red,
        "vec": 1,
        # signed edge ties (weight −1) are not a 0/1 selection -> take the general PᵀAP reduce path.
        "is_selection": bool(np.all(np.abs(np.asarray(data)) == 1.0) and np.all(np.asarray(data) >= 0.0)),
    }


# ---------------------------------------------------------------------------
# Sparse (BCOO) Dirichlet row/column operations
#
# The native assembler returns the global matrix as a BCOO (``O(nnz)``, never the
# dense ``O(n^2)`` array). These keep the three dense Dirichlet primitives
# (``.at[d,:].set(0)``, ``.at[:,d].set(0)``, ``.at[d,d].set(1)``) sparse: a row/col
# is "zeroed" by masking the stored values (nse unchanged — zeroed entries stay as
# explicit-zero triplets, harmless to matvec); a unit diagonal is appended as
# ``(d, d, 1)`` triplets (BCOO sums duplicates on matvec / todense / sum_duplicates,
# so the zeroed original ``(d,d)`` plus the appended ``1`` is exactly ``1``). All are
# static-``nse``, ``jit``-safe and differentiable in the stored values.
# ---------------------------------------------------------------------------


def bcoo_zero_rows(A, dofs):
    """``A.at[dofs, :].set(0)`` for a BCOO ``A`` — mask out every stored entry on a ``dofs`` row."""
    isd = jnp.zeros(A.shape[0], A.data.dtype).at[dofs].set(1.0)
    keep = 1.0 - isd[A.indices[:, 0]]
    return jsparse.BCOO((A.data * keep, A.indices), shape=A.shape)


def bcoo_zero_rows_cols(A, dofs):
    """``A.at[dofs, :].set(0).at[:, dofs].set(0)`` for a BCOO ``A`` (symmetric elimination, no diagonal)."""
    isd = jnp.zeros(A.shape[0], A.data.dtype).at[dofs].set(1.0)
    keep = (1.0 - isd[A.indices[:, 0]]) * (1.0 - isd[A.indices[:, 1]])
    return jsparse.BCOO((A.data * keep, A.indices), shape=A.shape)


def bcoo_set_unit_diag(A, dofs):
    """Append unit-diagonal triplets ``(d, d, 1)`` for ``d in dofs`` to a BCOO ``A`` (after the rows
    were zeroed, this makes ``A[d, d] == 1`` exactly — duplicate indices are summed)."""
    eye_idx = jnp.stack([dofs, dofs], axis=1).astype(A.indices.dtype)
    eye_dat = jnp.ones(jnp.asarray(dofs).shape[0], A.data.dtype)
    return jsparse.BCOO((jnp.concatenate([A.data, eye_dat]), jnp.concatenate([A.indices, eye_idx])), shape=A.shape)


def bcoo_set_dirichlet_rows(A, dofs):
    """``A.at[dofs, :].set(0).at[dofs, dofs].set(1)`` for a BCOO ``A`` — row-replacement (identity row,
    columns kept): the matrix-level analogue of the Newton row-replacement residual."""
    return bcoo_set_unit_diag(bcoo_zero_rows(A, dofs), dofs)


# ---------------------------------------------------------------------------
# Operator / state reduction and prolongation
# ---------------------------------------------------------------------------


def _is_selection(P):
    """True iff BCOO ``P`` has exactly one nonzero per full row — a periodic *selection* (each slave
    DOF equals a single master), for which the remap-sum reduction is exact. A **nonconforming** tie
    builds an *interpolation* ``P`` (several weighted masters per slave row), which needs a genuine
    ``P^T M P`` (the dense fallback). ``P.indices`` is static (built from connectivity), so this is a
    trace-time constant even when ``reduce_matrix`` runs inside a jitted ``operator_fn``."""
    if not hasattr(P, "indices"):
        return False
    rows = np.asarray(P.indices[:, 0])
    n = int(P.shape[0])
    return rows.shape[0] == n and int(np.bincount(rows, minlength=n).max()) == 1


def _selection_maps(P, dtype):
    """For a periodic selection ``P`` (BCOO, one nonzero per full row), return ``(master, pval)``:
    ``master[i]`` = the reduced DOF that full DOF ``i`` maps to; ``pval[i]`` = the tie coefficient
    (``1``, or ``-1`` for an antiperiodic tie)."""
    master = jnp.zeros(P.shape[0], P.indices.dtype).at[P.indices[:, 0]].set(P.indices[:, 1])
    pval = jnp.zeros(P.shape[0], dtype).at[P.indices[:, 0]].set(jnp.asarray(P.data, dtype))
    return master, pval


def _remap_bcoo(mat, m_row, p_row, m_col, p_col, shape):
    """Sparse Galerkin remap: send each BCOO triplet ``(r, c, v)`` of ``mat`` to
    ``(m_row[r], m_col[c], v·p_row[r]·p_col[c])`` and let BCOO sum duplicates. This is exactly
    ``P_row^T mat P_col`` for selection matrices, in ``O(nnz(mat))`` and **without** ever forming a
    dense ``n_full × n_full`` intermediate (``nnz`` is static = ``nnz(mat)`` → ``jit``-safe)."""
    r, c = mat.indices[:, 0], mat.indices[:, 1]
    ridx = jnp.stack([m_row[r], m_col[c]], axis=1)
    rdata = mat.data * p_row[r] * p_col[c]
    return jsparse.BCOO((rdata, ridx), shape=shape)


def _remap_bcoo_weighted(mat, P, conj=False):
    """Sparse Galerkin reduction ``P^T mat P`` (or ``P^H mat P`` when ``conj``) for an *interpolation*
    (nonconforming) periodic ``P``
    -- a few masters per slave, weighted. Generalises :func:`_remap_bcoo` (one master, weight 1) by
    spreading each ``mat`` triplet ``(r, c, v)`` across the ``D x D`` master pairs of its row and
    column with the interpolation weights: ``v -> v · w_r[a] · w_c[b]`` at ``(master_r[a], master_c[b])``.
    Stays sparse (no dense ``n_full × n_full`` intermediate); ``D = max masters/slave`` (small: the
    nodes of a master facet). Returns ``None`` if ``P``'s indices are not concrete (built under trace),
    so the caller falls back to the dense product -- the nonconforming reduction is built eagerly."""
    try:
        pidx = np.asarray(P.indices)  # concrete only; a tracer raises
        pdat = np.asarray(P.data)
    except Exception:
        return None
    n_full, n_red = int(mat.shape[0]), int(P.shape[1])
    rows, cols = pidx[:, 0], pidx[:, 1]
    order = np.argsort(rows, kind="stable")
    rows, cols, wts = rows[order], cols[order], pdat[order]
    if len(rows) == 0:
        return jsparse.BCOO((jnp.zeros(0), jnp.zeros((0, 2), jnp.int32)), shape=(n_red, n_red))
    is_new = np.concatenate([[True], rows[1:] != rows[:-1]])
    slot = np.arange(len(rows)) - np.maximum.accumulate(np.where(is_new, np.arange(len(rows)), 0))
    D = int(slot.max()) + 1
    master = np.zeros((n_full, D), np.int64)
    master[rows, slot] = cols
    weight = np.zeros((n_full, D), np.complex128 if np.iscomplexobj(pdat) else np.float64)
    weight[rows, slot] = wts
    master, weight = jnp.asarray(master), jnp.asarray(weight)
    r, c, v = mat.indices[:, 0], mat.indices[:, 1], mat.data
    mr, wr, mc, wc = master[r], weight[r], master[c], weight[c]  # (nnz, D)
    if conj:  # P^H mat P conjugates the row (left) weights
        wr = jnp.conj(wr)
    nnz = r.shape[0]
    a = jnp.broadcast_to(mr[:, :, None], (nnz, D, D)).reshape(-1)
    b = jnp.broadcast_to(mc[:, None, :], (nnz, D, D)).reshape(-1)
    data = (v[:, None, None] * wr[:, :, None] * wc[:, None, :]).reshape(-1)
    idx = jnp.stack([a, b], axis=1).astype(mat.indices.dtype)
    return jsparse.BCOO((data, idx), shape=(n_red, n_red)).sum_duplicates()


def reduce_matrix(P, mat, is_selection=None, conj=False):
    """Galerkin reduction ``P^T mat P`` (or the Hermitian ``P^H mat P`` when ``conj`` — for a complex
    Bloch/quasi-periodic ``P``, where the left factor must be conjugated).

    When ``mat`` is BCOO and ``P`` is a BCOO *selection* (conforming/structured tie, one master per
    slave — e.g. the doubly-periodic PEB) the reduction remaps ``mat``'s triplets to their master
    indices: it stays sparse and never materialises the dense ``n_full × n_full`` matrix (``O(nnz)``;
    the reduction is otherwise the dominant memory peak of a periodic solve). An *interpolation* ``P``
    (nonconforming tie) or a dense ``P``/``mat`` (1D path) falls back to the exact dense ``P^T mat P``.

    ``is_selection`` (whether ``P`` is a one-master-per-slave selection) is passed precomputed by the
    periodic builders so the reduction never inspects ``P.indices`` at run time — that matters because
    a parametric ``operator_fn`` reduces inside a jitted ``scan`` where ``P.indices`` is a tracer. When
    ``None`` (a direct/eager call) it is computed once here."""
    if is_selection is None:
        is_selection = _is_selection(P)
    _dt = lambda x: x.data.dtype if hasattr(x, "data") else np.asarray(x).dtype  # noqa: E731
    pdtype = np.result_type(_dt(P), _dt(mat))
    if hasattr(mat, "indices") and is_selection:
        master, pval = _selection_maps(P, pdtype)
        prow = jnp.conj(pval) if conj else pval  # P^H remap conjugates the row (left) factor
        return _remap_bcoo(mat, master, prow, master, pval, (int(P.shape[1]), int(P.shape[1])))
    if hasattr(mat, "indices") and hasattr(P, "indices"):
        # nonconforming (interpolation) tie: weighted triplet-remap, stays sparse (falls back to the
        # dense product below only if P was built under trace, which the nonconforming path never is)
        remapped = _remap_bcoo_weighted(mat, P, conj=conj)
        if remapped is not None:
            return remapped
    mat = mat.todense() if hasattr(mat, "todense") else mat
    Pd = jnp.asarray(P.todense() if hasattr(P, "todense") else P, pdtype)
    left = jnp.conj(Pd).T if conj else Pd.T
    return left @ jnp.asarray(mat, pdtype) @ Pd


def reduce_vector(P, vec, conj=False):
    """Reduce a full-space load/bias vector via ``P^T vec`` (or ``P^H vec`` when ``conj``)."""
    Pd = P if hasattr(P, "todense") else jnp.asarray(P)
    left = (jnp.conj(Pd).T if conj else Pd.T) if not hasattr(Pd, "indices") else None
    if left is not None:
        return left @ jnp.asarray(vec, Pd.dtype).reshape(-1)
    Pc = jsparse.BCOO((jnp.conj(Pd.data), Pd.indices), shape=Pd.shape) if conj else Pd
    return Pc.T @ jnp.asarray(vec, Pc.dtype).reshape(-1)


def restrict_state(P, state_full, kept_nodes, vec: int = 1):
    """Restrict a full initial state to the reduced master DOFs.

    For a consistent periodic initial condition (matching values on opposite
    faces) gathering the kept-node entries is exact and avoids the doubling
    that ``P^T`` would introduce on master nodes.
    """
    state_full = jnp.asarray(state_full).reshape(-1)
    kept = np.asarray(kept_nodes, dtype=int)
    if vec == 1:
        return state_full[jnp.asarray(kept)]
    dof = (kept[:, None] * vec + np.arange(vec)[None, :]).reshape(-1)
    return state_full[jnp.asarray(dof)]


def prolong(P, reduced):
    """Map reduced DOFs back to the full space: ``u_full = P @ u_red``.

    Accepts a single ``(n_red,)`` vector or a batched ``(..., n_red)`` array
    (e.g. a Diffrax ``solution.ys`` trajectory of shape ``(T, n_red)``).
    """
    P = P if hasattr(P, "todense") else jnp.asarray(P)  # keep a BCOO P sparse
    reduced = jnp.asarray(reduced, P.dtype)
    if reduced.ndim == 1:
        return P @ reduced
    # (..., n_red) @ (n_red, n_full) -> (..., n_full)
    return reduced @ P.T


# ---------------------------------------------------------------------------
# Periodic reduction over a (possibly multifield) block.  A `periodic` dict is
# either single-field (legacy `{"P", "kept_nodes", "vec"}`) or multifield
# (`{"blocks": [{"P", "kept", "vec"}, ...], "off_full": [...], "off_red": [...]}`).
# The Galerkin reduction of a blocked operator decomposes per field-pair,
# `reduced[i,j] = P_i^T M[i,j] P_j`, so we NEVER materialise a block-diagonal P
# (which would be O(F^2) dense and force one global vec/order); each field keeps
# its own P_i / kept / vec and only one P_i is held at a time.
# ---------------------------------------------------------------------------


def _periodic_blocks(periodic):
    """Normalise a periodic dict to ``(blocks, off_full, off_red)``; a legacy single-field
    dict is wrapped as a single block (so the multifield path is the general case)."""
    if "blocks" in periodic:
        return periodic["blocks"], np.asarray(periodic["off_full"]), np.asarray(periodic["off_red"])
    P = periodic["P"]
    nf, nr = int(P.shape[0]), int(P.shape[1])
    block = [
        {
            "P": P,
            "kept": periodic["kept_nodes"],
            "vec": periodic.get("vec", 1),
            "is_selection": periodic.get("is_selection"),
        }
    ]
    return block, np.array([0, nf]), np.array([0, nr])


def build_periodic_prolongation_n1e(n_edges, edge_midpoints, edge_dirs, etags, pairs, phases=None, tol=None):
    """DOF-level periodic (Floquet/Bloch) prolongation for a lowest-order Nédélec (N1E) edge field.

    Each edge carries one tangential-moment DOF. A slave-face edge ties to the master-face edge at the same
    transverse position; the tie weight is the Bloch phase times an orientation **sign** (+1 if the two
    edges point the same way along their lo→hi canonical direction, −1 if opposed — the tangential moment
    flips with the edge orientation). Corner edges shared by two periodic faces are tied twice; the chain is
    resolved to a single retained master DOF. Returns a legacy single-field prolongation dict."""
    mid = np.asarray(edge_midpoints, dtype=np.float64)
    dirs = np.asarray(edge_dirs, dtype=np.float64)
    if phases is None:
        phases = [1.0] * len(pairs)
    is_bloch = any(abs(complex(p) - 1.0) > 1e-12 for p in phases)
    if tol is None:
        span = float(np.linalg.norm(mid.max(axis=0) - mid.min(axis=0)))
        tol = max(span, 1.0) * 1.0e-6

    s2m: Dict[int, int] = {}  # slave edge -> master edge
    weight: Dict[int, complex] = {}  # slave edge -> tie weight (orientation sign × Bloch phase)
    for (mtag, stag), ph in zip(pairs, phases):
        me, se = np.asarray(etags[mtag], dtype=int), np.asarray(etags[stag], dtype=int)
        if me.size == 0 or se.size == 0:
            raise ValueError(f"periodic N1E: boundary tag {mtag!r}/{stag!r} has no edges to tie.")
        mm, sm = mid[me], mid[se]
        axis = int(np.argmax(np.abs(mm.mean(axis=0) - sm.mean(axis=0))))  # periodic axis = largest mean gap
        tr = [d for d in range(mid.shape[1]) if d != axis]
        mt, st = mm[:, tr], sm[:, tr]
        d2 = np.sum((st[:, None, :] - mt[None, :, :]) ** 2, axis=-1)
        nn = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(len(se)), nn])
        for k, s in enumerate(se):
            if dist[k] > tol:
                raise ValueError(
                    f"periodic N1E: slave edge {int(s)} has no transverse master match "
                    f"(nearest {dist[k]:.2e} > tol {tol:.2e}); a conforming periodic mesh is required."
                )
            m = int(me[nn[k]])
            sign = 1.0 if float(dirs[int(s)] @ dirs[m]) >= 0.0 else -1.0
            s2m[int(s)], weight[int(s)] = m, sign * complex(ph)

    slaves = set(s2m)
    kept = np.array([e for e in range(n_edges) if e not in slaves], dtype=int)
    col = {int(e): j for j, e in enumerate(kept)}
    P = np.zeros((n_edges, len(kept)), dtype=(np.complex128 if is_bloch else np.float64))
    for j, e in enumerate(kept):
        P[e, j] = 1.0
    for s in slaves:  # resolve a possibly-chained tie (corner edge tied via two faces) to a kept master
        m, w = s2m[s], weight[s]
        while m in s2m:
            w *= weight[m]
            m = s2m[m]
        P[s, col[m]] = w if is_bloch else w.real  # non-Bloch weight is a real ±1 sign
    return {
        "P": jnp.asarray(P),
        "kept_nodes": kept,
        "vec": 1,
        "is_selection": False,  # ±1 signs (and complex phases) → not a pure 0/1 selection
        "is_bloch": is_bloch,
        "n_full": int(n_edges),
        "n_red": int(len(kept)),
    }


def reduce_matrix_periodic(periodic, mat, conj=False):
    """``P^T mat P`` (or Hermitian ``P^H mat P`` when ``conj``) for a single- or multi-field reduction.

    When ``mat`` and every field's ``P_i`` are BCOO, the whole blocked reduction
    ``reduced[i,j] = P_i^T mat[i,j] P_j`` is one global triplet-remap (the per-field offsets are
    folded into a single full→reduced master map), so it stays sparse and never densifies — the fix
    for the ``O(n^2)`` reduction peak at large ``N``. A dense ``mat`` falls back to the per-block
    dense reduction (no block-diagonal ``P`` is ever materialised)."""
    blocks, off_f, off_r = _periodic_blocks(periodic)

    def _sel(b):  # prefer the precomputed flag (built eagerly); compute only when called eagerly
        return b["is_selection"] if b.get("is_selection") is not None else _is_selection(b["P"])

    if len(blocks) == 1:
        return reduce_matrix(blocks[0]["P"], mat, is_selection=_sel(blocks[0]), conj=conj)  # single-field
    if hasattr(mat, "indices") and all(_sel(b) for b in blocks):
        n_full, n_red = int(off_f[-1]), int(off_r[-1])
        pdtype = np.result_type(*[b["P"].data.dtype for b in blocks], mat.data.dtype)
        gmaster = jnp.zeros(n_full, mat.indices.dtype)
        gpval = jnp.zeros(n_full, pdtype)
        for i, b in enumerate(blocks):
            Pi = b["P"]
            full_local = Pi.indices[:, 0]  # field-i local full DOF
            gmaster = gmaster.at[int(off_f[i]) + full_local].set(
                (int(off_r[i]) + Pi.indices[:, 1]).astype(mat.indices.dtype)
            )
            gpval = gpval.at[int(off_f[i]) + full_local].set(jnp.asarray(Pi.data, pdtype))
        prow = jnp.conj(gpval) if conj else gpval
        return _remap_bcoo(mat, gmaster, prow, gmaster, gpval, (n_red, n_red))
    mat = jnp.asarray(mat.todense()) if hasattr(mat, "todense") else jnp.asarray(mat)
    pdtype = np.result_type(
        *[np.asarray(b["P"].todense() if hasattr(b["P"], "todense") else b["P"]).dtype for b in blocks], mat.dtype
    )
    out = jnp.zeros((int(off_r[-1]), int(off_r[-1])), pdtype)
    _sp = lambda P: P if hasattr(P, "todense") else jnp.asarray(P, pdtype)  # keep BCOO sparse  # noqa: E731
    for i, bi in enumerate(blocks):
        Pi = _sp(bi["P"])
        left = (jnp.conj(Pi).T if conj else Pi.T) if not hasattr(Pi, "indices") else Pi.T
        for j, bj in enumerate(blocks):
            Pj = _sp(bj["P"])
            blk = jnp.asarray(mat[off_f[i] : off_f[i + 1], off_f[j] : off_f[j + 1]], pdtype)
            out = out.at[off_r[i] : off_r[i + 1], off_r[j] : off_r[j + 1]].set(left @ blk @ Pj)
    return out


def reduce_vector_periodic(periodic, vec, conj=False):
    """``P^T vec`` (or ``P^H vec`` when ``conj``) for a single- or multi-field reduction (per-field block)."""
    blocks, off_f, _ = _periodic_blocks(periodic)
    if len(blocks) == 1:
        return reduce_vector(blocks[0]["P"], vec, conj=conj)
    vec = jnp.asarray(vec).reshape(-1)
    _sp = lambda P: P if hasattr(P, "todense") else jnp.asarray(P, vec.dtype)  # keep BCOO sparse  # noqa: E731
    out = []
    for i, b in enumerate(blocks):
        Pi = _sp(b["P"])
        Pc = jsparse.BCOO((jnp.conj(Pi.data), Pi.indices), shape=Pi.shape) if (conj and hasattr(Pi, "indices")) else Pi
        out.append(Pc.T @ vec[off_f[i] : off_f[i + 1]])
    return jnp.concatenate(out)


def restrict_state_periodic(periodic, state):
    """Restrict a full state to the reduced master DOFs (per-field block)."""
    blocks, off_f, _ = _periodic_blocks(periodic)
    if len(blocks) == 1:
        return restrict_state(blocks[0]["P"], state, blocks[0]["kept"], blocks[0]["vec"])
    state = jnp.asarray(state).reshape(-1)
    return jnp.concatenate(
        [restrict_state(b["P"], state[off_f[i] : off_f[i + 1]], b["kept"], b["vec"]) for i, b in enumerate(blocks)]
    )


def prolong_periodic(periodic, reduced):
    """Prolong reduced DOFs to the full space (per-field block); supports a batched leading axis."""
    blocks, _, off_r = _periodic_blocks(periodic)
    if len(blocks) == 1:
        return prolong(blocks[0]["P"], reduced)
    reduced = jnp.asarray(reduced)
    parts = [prolong(b["P"], reduced[..., off_r[i] : off_r[i + 1]]) for i, b in enumerate(blocks)]
    return jnp.concatenate(parts, axis=-1)
