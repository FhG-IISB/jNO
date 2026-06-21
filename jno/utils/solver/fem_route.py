from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from ...trace import (
    FemLinearSystem,
    FemResidualOperator,
)
from .feax_utils import (
    _build_feax_problem,
    _default_float_dtype,
    _dense_array,
    _normalize_dirichlet_value,
    _prepare_feax_runtime,
    _region_mask_arrays_for_domain,
)
from .parametric_helpers import (
    _contains_runtime_parameter,
    _make_ir,
    _make_zero_ir_like,
    _runtime_scalar_arg,
    _split_parametric_operator_ir,
)
from .solver_helper import contains_trialfunction as _contains_trialfunction

# --------------------------------
# FEM boundary-condition helpers
# --------------------------------


@dataclass(frozen=True)
class DirichletBC:
    """
    Essential boundary-condition descriptor for FEM/FEAX assembly.

    Instances are created through `dirichlet(...)` and later normalized by
    `expand_bcs(...)` during `domain.init_fem(...)`.

    Parameters
    ----------
    tags:
        Boundary tag names on which the Dirichlet condition is applied.
    values:
        Boundary value specification. Supported forms are handled by
        `_normalize_dirichlet_value(...)` and include `None`, scalars,
        callables, component lists/tuples, and component dictionaries.
    """

    tags: tuple[str, ...]
    values: object = None


@dataclass(frozen=True)
class NeumannBC:
    """
    Natural boundary-condition descriptor for FEM/FEAX assembly.

    Instances are created through `neumann(...)`. The tags mark boundary
    regions whose weak-form boundary terms should be included in FEAX surface
    assembly.

    Parameters
    ----------
    tags:
        Boundary tag names treated as natural/surface regions.
    """

    tags: tuple[str, ...]


@dataclass(frozen=True)
class PeriodicBC:
    """
    Periodic boundary-condition descriptor.

    Each pair identifies a master and slave boundary whose degrees of freedom
    must be identified through a FEAX prolongation matrix.
    """

    pairs: tuple[tuple[str, str], ...]


def periodic(*pairs):
    """
    Create periodic boundary-condition pairings.

    Example
    -------
    periodic(
        ("left", "right"),
        ("bottom", "top"),
    )
    """
    normalized = []

    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("Each periodic pair must be `(master_tag, slave_tag)`.")

        normalized.append(
            (
                str(pair[0]),
                str(pair[1]),
            )
        )

    if len(normalized) == 0:
        raise ValueError("At least one periodic boundary pair is required.")

    return PeriodicBC(
        pairs=tuple(normalized),
    )


def _as_tags(tags) -> tuple[str, ...]:
    """
    Normalize one boundary tag or a sequence of tags into a non-empty tuple.

    Parameters
    ----------
    tags:
        Either a string tag or a sequence of tag-like objects.

    Returns
    -------
    tuple[str, ...]
        Normalized tuple of boundary tag strings.
    """
    if isinstance(tags, str):
        return (tags,)
    if isinstance(tags, Sequence):
        out = tuple(str(t) for t in tags)
        if len(out) == 0:
            raise ValueError("Boundary tag list cannot be empty.")
        return out
    raise TypeError(f"Boundary tags must be a string or a sequence of strings, got {type(tags).__name__}.")


def dirichlet(tags, values=None):
    """
    Create a Dirichlet boundary-condition descriptor.

    This is the public helper used in FEM setup, for example:

        domain.init_fem(
            bcs=[
                domain.dirichlet("left", 0.0),
                domain.dirichlet(["bottom", "top"], {"x": 0.0, "y": 1.0}),
            ]
        )

    Parameters
    ----------
    tags:
        Boundary tag or list of boundary tags.
    values:
        Boundary value specification. For scalar unknowns, this can be a scalar
        or callable. For vector unknowns, this can also be a component list,
        tuple, or dictionary.

    Returns
    -------
    DirichletBC
        Boundary-condition descriptor consumed by `expand_bcs(...)`.
    """
    return DirichletBC(tags=_as_tags(tags), values=values)


def neumann(tags):
    """
    Create a Neumann/natural boundary-condition descriptor.

    This marks boundary regions whose weak-form boundary terms should be
    assembled through FEAX surface kernels.

    Parameters
    ----------
    tags:
        Boundary tag or list of boundary tags.

    Returns
    -------
    NeumannBC
        Boundary-condition descriptor consumed by `expand_bcs(...)`.
    """
    return NeumannBC(tags=_as_tags(tags))


def expand_bcs(bcs, vec: int):
    """
    Normalize user boundary-condition descriptors for FEM initialization.

    Returns
    -------
    tuple
        ``(dirichlet_tags, dirichlet_value_fns, neumann_tags, periodic_pairs)``.
        ``periodic_pairs`` is a list of ``(master_tag, slave_tag)`` tuples.
    """
    dirichlet_tags = []
    dirichlet_value_fns = {}
    neumann_tags = []
    periodic_pairs = []

    for bc in bcs:
        if isinstance(bc, DirichletBC):
            for tag in bc.tags:
                if tag not in dirichlet_tags:
                    dirichlet_tags.append(tag)
                dirichlet_value_fns[tag] = _normalize_dirichlet_value(bc.values, vec)
        elif isinstance(bc, NeumannBC):
            for tag in bc.tags:
                if tag not in neumann_tags:
                    neumann_tags.append(tag)
        elif isinstance(bc, PeriodicBC):
            periodic_pairs.extend(bc.pairs)
        else:
            raise TypeError(
                f"Unsupported BC entry '{type(bc).__name__}'. Use dirichlet(...), neumann(...) or periodic(...)."
            )

    return dirichlet_tags, dirichlet_value_fns, neumann_tags, periodic_pairs


# --------------------------------
# public FEAX-backed entry points
# --------------------------------


def _assemble_fem_residual_from_ir(domain, ir, **kwargs):
    """
    Assemble a steady nonlinear FEM residual operator from lowered weak-form IR.

    Parameter-free weak forms preserve the existing FEAX path.

    Affine runtime parameters are lowered into:

        R(u, args) = R0(u) + sum_i args[name_i] * R_i(u)
        J(u, args) = J0(u) + sum_i args[name_i] * J_i(u)

    Supported runtime pattern:
        parameter * weak_term

    The static contribution is assembled with Dirichlet enforcement. Parameter
    basis contributions are assembled without reapplying Dirichlet enforcement,
    so essential boundary rows are imposed exactly once.
    """
    import feax as fe

    symmetric_bc = kwargs.get("symmetric_bc", True)
    fields_override = kwargs.get("fields_override", None)

    has_runtime_parameters = any(_contains_runtime_parameter(term.coeff) for term in ir.terms)

    # ------------------------------------------------------------
    # Original parameter-free route
    # ------------------------------------------------------------
    if not has_runtime_parameters:
        problem, bc = _build_feax_problem(domain, ir, fields_override=fields_override)
        # Sub-region terms: thread the per-cell masks (order set by _build_feax_problem) so a nonlinear
        # weak form restricted to a sub-region integrates over that region's cells only.
        _masks = _region_mask_arrays_for_domain(domain)
        internal_vars = fe.InternalVars(volume_vars=_masks) if _masks else fe.InternalVars()

        res_bc = fe.create_res_bc_function(problem, bc)
        jac_bc = fe.create_J_bc_function(
            problem,
            bc,
            symmetric=symmetric_bc,
        )
        size = int(problem.num_total_dofs_all_vars)

        def residual_fn(u_flat, args=None):
            del args
            u_flat = jnp.asarray(
                u_flat,
                dtype=_default_float_dtype(),
            ).reshape(-1)
            return jnp.asarray(
                res_bc(u_flat, internal_vars),
                dtype=_default_float_dtype(),
            ).reshape(-1)

        def jacobian_fn(u_flat, args=None):
            del args
            u_flat = jnp.asarray(
                u_flat,
                dtype=_default_float_dtype(),
            ).reshape(-1)
            return jac_bc(u_flat, internal_vars)

        return FemResidualOperator(
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            size=size,
        )

    # ------------------------------------------------------------
    # Parameter-aware affine route
    # ------------------------------------------------------------
    static_ir, parameter_irs, runtime_parameter_exprs, _ = _split_parametric_operator_ir(ir)

    if len(parameter_irs) == 0:
        raise RuntimeError("Runtime parameters were detected, but no affine FEM basis terms were generated.")

    # If all physical terms are parameter-dependent, create a structural zero
    # contribution so the static runtime still imposes Dirichlet conditions.
    if len(static_ir.terms) == 0:
        first_basis_ir = next(iter(parameter_irs.values()))
        static_ir = _make_zero_ir_like(first_basis_ir)

    # Each runtime gets its own InternalVars carrying that IR's sub-region masks (the order is set by
    # the _build_feax_problem inside _prepare_feax_runtime, so capture immediately after each call).
    def _iv_for_current():
        masks = _region_mask_arrays_for_domain(domain)
        return fe.InternalVars(volume_vars=masks) if masks else fe.InternalVars()

    static_rt = _prepare_feax_runtime(
        domain,
        static_ir,
        apply_dirichlet=True,
        need_jacobian=True,
        symmetric_bc=symmetric_bc,
    )
    static_iv = _iv_for_current()

    basis_runtimes = {}

    for name, basis_ir in parameter_irs.items():
        zero_basis_ir = _make_zero_ir_like(basis_ir)

        basis_rt = _prepare_feax_runtime(
            domain,
            basis_ir,
            apply_dirichlet=True,
            need_jacobian=True,
            symmetric_bc=symmetric_bc,
        )
        basis_iv = _iv_for_current()

        zero_rt = _prepare_feax_runtime(
            domain,
            zero_basis_ir,
            apply_dirichlet=True,
            need_jacobian=True,
            symmetric_bc=symmetric_bc,
        )
        zero_iv = _iv_for_current()

        basis_runtimes[name] = {
            "basis": basis_rt,
            "zero": zero_rt,
            "basis_iv": basis_iv,
            "zero_iv": zero_iv,
        }

    if static_rt["jac_bc"] is None:
        raise ValueError("Static FEM residual runtime did not produce a Jacobian.")

    for name, pair in basis_runtimes.items():
        rt = pair["basis"]
        zero_rt = pair["zero"]

        if rt["jac_bc"] is None or zero_rt["jac_bc"] is None:
            raise ValueError(f"FEM residual basis {name!r} did not produce a Jacobian.")

        if int(rt["size"]) != int(static_rt["size"]):
            raise ValueError(f"FEM residual basis {name!r} has size {rt['size']}, expected {static_rt['size']}.")

    dtype = static_rt["dtype"]
    size = int(static_rt["size"])

    def residual_fn(u_flat, args=None):
        u_flat = jnp.asarray(u_flat, dtype=dtype).reshape(-1)

        out = jnp.asarray(
            static_rt["res_bc"](u_flat, static_iv),
            dtype=dtype,
        ).reshape(-1)

        for name, pair in basis_runtimes.items():
            coeff = _runtime_scalar_arg(
                args,
                name,
                dtype=dtype,
            )

            basis_res = jnp.asarray(
                pair["basis"]["res_bc"](
                    u_flat,
                    pair["basis_iv"],
                ),
                dtype=dtype,
            ).reshape(-1) - jnp.asarray(
                pair["zero"]["res_bc"](
                    u_flat,
                    pair["zero_iv"],
                ),
                dtype=dtype,
            ).reshape(-1)

            out = out + coeff * basis_res

        return out

    def jacobian_fn(u_flat, args=None):
        u_flat = jnp.asarray(u_flat, dtype=dtype).reshape(-1)

        out = jnp.asarray(
            _dense_array(static_rt["jac_bc"](u_flat, static_iv)),
            dtype=dtype,
        )

        for name, pair in basis_runtimes.items():
            coeff = _runtime_scalar_arg(
                args,
                name,
                dtype=dtype,
            )

            basis_jac = jnp.asarray(
                _dense_array(
                    pair["basis"]["jac_bc"](
                        u_flat,
                        pair["basis_iv"],
                    )
                ),
                dtype=dtype,
            )
            -jnp.asarray(
                _dense_array(
                    pair["zero"]["jac_bc"](
                        u_flat,
                        pair["zero_iv"],
                    )
                ),
                dtype=dtype,
            )

            out = out + coeff * basis_jac

        return out

    return FemResidualOperator(
        residual_fn=residual_fn,
        jacobian_fn=jacobian_fn,
        size=size,
        runtime_parameter_exprs=runtime_parameter_exprs,
        residual_basis=basis_runtimes,
        metadata={
            "dynamic_residual": True,
            "runtime_parameter_names": sorted(runtime_parameter_exprs),
            "lowering": "R(u,args) = R0(u) + sum_i args[name_i] * R_i(u)",
        },
    )


def _split_trial_and_load_ir(ir):
    """Split one lowered FEM IR into operator and source/load terms."""
    operator_terms = []
    load_terms = []

    for term in ir.terms:
        if _contains_trialfunction(term.coeff):
            operator_terms.append(term)
        else:
            load_terms.append(term)

    return (
        _make_ir(ir.domain, operator_terms),
        _make_ir(ir.domain, load_terms),
    )


def _assemble_static_source_vector_from_ir(domain, src_ir, *, dtype):
    """Assemble a steady volume-source or Neumann-load vector."""
    import feax as fe

    if src_ir is None or len(src_ir.terms) == 0:
        return None

    rt = _prepare_feax_runtime(
        domain,
        src_ir,
        apply_dirichlet=False,
        need_jacobian=False,
        symmetric_bc=True,
    )

    u_zero = jnp.zeros((rt["size"],), dtype=rt["dtype"])
    iv = fe.InternalVars()

    return -jnp.asarray(
        rt["res_bc"](u_zero, iv),
        dtype=dtype,
    ).reshape(-1)


def _assemble_fem_system_concrete(
    domain,
    ir,
    *,
    apply_dirichlet=True,
    symmetric_bc=True,
    fields_override=None,
    store_on_domain=True,
):
    """
    Assemble one concrete steady linear FEAX contribution.

    This helper deliberately receives an IR that no longer contains runtime
    parameter ModelCall nodes. ``fields_override`` forces the multi-field block
    layout (used by the coupled transient route so mass and operator blocks share
    one field ordering). ``apply_dirichlet=False`` returns the *raw* matrix (no
    Dirichlet rows/cols eliminated) — used to keep the transient mass's Dirichlet
    columns for the time-varying-Dirichlet coupling; ``store_on_domain=False`` keeps
    that scratch problem off the domain.
    """
    import feax as fe

    problem, bc = _build_feax_problem(
        domain,
        ir,
        apply_dirichlet=apply_dirichlet,
        fields_override=fields_override,
        store_on_domain=store_on_domain,
    )

    # Per-region (sub-domain) terms: thread one constant per-cell 0/1 mask per region (cached, in the
    # order _build_feax_problem just recorded). The kernel multiplies each sub-region term's integrand
    # by its mask, so it integrates over that region's cells only.
    region_masks = _region_mask_arrays_for_domain(domain)
    internal_vars = fe.InternalVars(volume_vars=region_masks) if region_masks else fe.InternalVars()

    try:
        u0 = fe.zero_like_initial_guess(problem, bc)
    except Exception:
        u0 = jnp.zeros(
            (problem.num_total_dofs_all_vars,),
            dtype=_default_float_dtype(),
        )

    u0 = jnp.asarray(u0, dtype=_default_float_dtype())

    res_bc = fe.create_res_bc_function(problem, bc)
    jac_bc = fe.create_J_bc_function(
        problem,
        bc,
        symmetric=symmetric_bc,
    )

    # FEAX gives the correction system:
    #     A du = -r(u0)
    # Convert to the full-state system:
    #     A u = A u0 - r(u0)
    A = jac_bc(u0, internal_vars)
    r0 = jnp.asarray(
        res_bc(u0, internal_vars),
        dtype=_default_float_dtype(),
    )

    if hasattr(A, "__matmul__"):
        b = A @ u0 - r0
    else:
        A_dense = jnp.asarray(A.todense() if hasattr(A, "todense") else A.toarray())
        b = A_dense @ u0 - r0
        A = A_dense

    return A, b


def _assemble_fem_system_from_ir(domain, ir, **kwargs):
    """Assemble ``A(args) u = b(args)`` for affine runtime weak-form terms."""
    symmetric_bc = kwargs.get("symmetric_bc", True)
    fields_override = kwargs.get("fields_override", None)
    apply_dirichlet = kwargs.get("apply_dirichlet", True)
    store_on_domain = kwargs.get("store_on_domain", True)

    has_runtime_parameters = any(_contains_runtime_parameter(term.coeff) for term in ir.terms)

    if not has_runtime_parameters:
        return _assemble_fem_system_concrete(
            domain,
            ir,
            apply_dirichlet=apply_dirichlet,
            symmetric_bc=symmetric_bc,
            fields_override=fields_override,
            store_on_domain=store_on_domain,
        )

    static_ir, parameter_irs, runtime_parameter_exprs, nonaffine_ir = _split_parametric_operator_ir(
        ir, allow_nonaffine=True
    )
    has_nonaffine = len(nonaffine_ir.terms) > 0

    operator_basis = {}
    rhs_basis = {}

    for name, basis_ir in parameter_irs.items():
        op_basis_ir, rhs_basis_ir = _split_trial_and_load_ir(basis_ir)

        # With any non-affine operator parameter present the whole operator is
        # re-assembled each call (below), so skip the affine operator basis here to
        # avoid double-counting the affine operator parameters.
        if (not has_nonaffine) and len(op_basis_ir.terms) > 0:
            zero_basis_ir = _make_zero_ir_like(op_basis_ir)
            K_bc, bK_bc = _assemble_fem_system_concrete(
                domain, op_basis_ir, apply_dirichlet=True, symmetric_bc=symmetric_bc
            )
            K_zero_bc, bK_zero_bc = _assemble_fem_system_concrete(
                domain, zero_basis_ir, apply_dirichlet=True, symmetric_bc=symmetric_bc
            )
            K = jnp.asarray(_dense_array(K_bc)) - jnp.asarray(_dense_array(K_zero_bc))
            bK = jnp.asarray(bK_bc).reshape(-1) - jnp.asarray(bK_zero_bc).reshape(-1)
            operator_basis[name] = K
            # A non-homogeneous Dirichlet value g lifted by this parametric operator basis
            # appears as a parameter-scaled RHS term: bK = -theta * K_ib * g on the interior
            # rows, and *zero on the Dirichlet rows* (g itself is fixed, carried by b0). Carry
            # it so b(theta) = b0 + sum theta * bK. If bK is non-zero on the Dirichlet rows the
            # Dirichlet *value* would scale with theta -- a genuine runtime-Dirichlet-value
            # parameter, which is still unsupported.
            if not np.allclose(np.asarray(bK), 0.0, atol=1.0e-8):
                _dir = np.asarray(_dense_array(K_zero_bc)).diagonal() > 0.5  # bc-identity rows
                if not np.allclose(np.asarray(bK)[_dir], 0.0, atol=1.0e-7):
                    raise NotImplementedError(
                        "Runtime Dirichlet *value* parameters are not supported (the prescribed "
                        "Dirichlet value scales with the parameter)."
                    )
                rhs_basis[name] = rhs_basis.get(name, jnp.zeros_like(bK)) + bK

        if len(rhs_basis_ir.terms) > 0:
            rhs_vec = _assemble_static_source_vector_from_ir(domain, rhs_basis_ir, dtype=_default_float_dtype())
            if rhs_vec is not None:
                rv = jnp.asarray(rhs_vec)
                rhs_basis[name] = rhs_basis.get(name, jnp.zeros_like(rv)) + rv

    if len(static_ir.terms) == 0:
        structural_op_ir = None
        for basis_ir in parameter_irs.values():
            op_basis_ir, _ = _split_trial_and_load_ir(basis_ir)
            if len(op_basis_ir.terms) > 0:
                structural_op_ir = op_basis_ir
                break
        if structural_op_ir is None and has_nonaffine:
            na_op_ir, _ = _split_trial_and_load_ir(nonaffine_ir)
            if len(na_op_ir.terms) > 0:
                structural_op_ir = na_op_ir
        if structural_op_ir is None:
            raise ValueError(
                "A static fem_system requires at least one operator term. Only runtime source/load terms were found."
            )
        static_ir = _make_zero_ir_like(structural_op_ir)

    A0, b0 = _assemble_fem_system_concrete(domain, static_ir, apply_dirichlet=True, symmetric_bc=symmetric_bc)
    A0 = jnp.asarray(_dense_array(A0))
    b0 = jnp.asarray(b0, dtype=A0.dtype).reshape(-1)

    operator_fn = None
    if has_nonaffine:
        # Re-assembly route: a non-affine operator parameter (e.g. exp(k), k**2)
        # cannot be factored into a constant basis, so re-run the feax operator
        # assembly each call with ALL operator parameters threaded as InternalVars
        # (the parameter stays inside the integrand). The feax kernel is pure-JAX,
        # so operator_fn is differentiable in the parameters and Dirichlet is
        # applied once by the single assembly. Slower per call than the affine
        # A0 + sum theta*K basis, but exact for non-affine dependence. Mirrors the
        # transient non-affine path (time_route._na_full).
        import feax as fe

        from .feax_utils import _make_internal_vars
        from .parametric_helpers import _collect_runtime_parameter_exprs

        op_only_ir, _ = _split_trial_and_load_ir(ir)
        op_rt = _prepare_feax_runtime(
            domain, op_only_ir, apply_dirichlet=True, need_jacobian=True, symmetric_bc=symmetric_bc
        )
        if op_rt["jac_bc"] is None:
            raise ValueError("Non-affine FEM operator runtime did not produce a Jacobian.")
        _op_tags = list(op_rt["runtime_parameter_tags"])
        _op_dt = op_rt["dtype"]
        _op_u0 = jnp.zeros((int(op_rt["size"]),), dtype=_op_dt)
        # Sub-region masks for this operator IR, captured now (the order was just set by op_rt's
        # _build_feax_problem) so a non-affine parameter on a sub-region integrates over its cells only.
        _op_masks = _region_mask_arrays_for_domain(domain)

        def operator_fn(args=None, _rt=op_rt, _tags=_op_tags, _dt=_op_dt, _u0=_op_u0, _masks=_op_masks):
            # Keep the raw shape: a scalar parameter stays 0-d; a field parameter
            # (a nodal array) is threaded whole and gathered/interpolated per cell.
            _a = args or {}
            values = {name: jnp.asarray(_a[name], dtype=_dt) for name in _tags}
            iv = _make_internal_vars(
                fe,
                (),
                0.0,
                n_cells=_rt["n_cells"],
                dtype=_dt,
                runtime_parameter_tags=_tags,
                runtime_parameter_values=values,
                region_mask_arrays=_masks,
            )
            return jnp.asarray(_dense_array(_rt["jac_bc"](_u0, iv)), dtype=_dt)

        for term in nonaffine_ir.terms:
            _collect_runtime_parameter_exprs(term.coeff, runtime_parameter_exprs)

    elif operator_basis:

        def operator_fn(args=None):
            A = A0
            for name, K in operator_basis.items():
                coeff = _runtime_scalar_arg(args, name, dtype=A0.dtype)
                A = A + coeff * jnp.asarray(K, dtype=A0.dtype)
            return A

    rhs_fn = None
    if rhs_basis:

        def rhs_fn(args=None):
            b = b0
            for name, f_vec in rhs_basis.items():
                coeff = _runtime_scalar_arg(args, name, dtype=A0.dtype)
                b = b + coeff * jnp.asarray(f_vec, dtype=A0.dtype)
            return b

    return FemLinearSystem(
        A=A0,
        b=b0,
        operator_fn=operator_fn,
        rhs_fn=rhs_fn,
        runtime_parameter_exprs=runtime_parameter_exprs,
        operator_basis=operator_basis,
        rhs_basis=rhs_basis,
        metadata={
            "dynamic_operator": bool(operator_basis) or has_nonaffine,
            "dynamic_rhs": bool(rhs_basis),
            "nonaffine_operator": has_nonaffine,
            "runtime_parameter_names": sorted(runtime_parameter_exprs),
            "operator_parameter_names": sorted(operator_basis),
            "rhs_parameter_names": sorted(rhs_basis),
            "lowering": (
                "A(args) re-assembled with parameters as InternalVars (non-affine)"
                if has_nonaffine
                else "A(args) = A0 + sum_i args[name_i] * K_i; b(args) = b0 + sum_j args[name_j] * f_j"
            ),
        },
    )
