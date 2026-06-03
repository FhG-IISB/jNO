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

    has_runtime_parameters = any(_contains_runtime_parameter(term.coeff) for term in ir.terms)

    # ------------------------------------------------------------
    # Original parameter-free route
    # ------------------------------------------------------------
    if not has_runtime_parameters:
        problem, bc = _build_feax_problem(domain, ir)
        internal_vars = fe.InternalVars()

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
    static_ir, parameter_irs, runtime_parameter_exprs = _split_parametric_operator_ir(ir)

    if len(parameter_irs) == 0:
        raise RuntimeError("Runtime parameters were detected, but no affine FEM basis terms were generated.")

    # If all physical terms are parameter-dependent, create a structural zero
    # contribution so the static runtime still imposes Dirichlet conditions.
    if len(static_ir.terms) == 0:
        first_basis_ir = next(iter(parameter_irs.values()))
        static_ir = _make_zero_ir_like(first_basis_ir)

    static_rt = _prepare_feax_runtime(
        domain,
        static_ir,
        apply_dirichlet=True,
        need_jacobian=True,
        symmetric_bc=symmetric_bc,
    )

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

        zero_rt = _prepare_feax_runtime(
            domain,
            zero_basis_ir,
            apply_dirichlet=True,
            need_jacobian=True,
            symmetric_bc=symmetric_bc,
        )

        basis_runtimes[name] = {
            "basis": basis_rt,
            "zero": zero_rt,
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
    internal_vars = fe.InternalVars()

    def residual_fn(u_flat, args=None):
        u_flat = jnp.asarray(u_flat, dtype=dtype).reshape(-1)

        out = jnp.asarray(
            static_rt["res_bc"](u_flat, internal_vars),
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
                    internal_vars,
                ),
                dtype=dtype,
            ).reshape(-1) - jnp.asarray(
                pair["zero"]["res_bc"](
                    u_flat,
                    internal_vars,
                ),
                dtype=dtype,
            ).reshape(-1)

            out = out + coeff * basis_res

        return out

    def jacobian_fn(u_flat, args=None):
        u_flat = jnp.asarray(u_flat, dtype=dtype).reshape(-1)

        out = jnp.asarray(
            _dense_array(static_rt["jac_bc"](u_flat, internal_vars)),
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
                        internal_vars,
                    )
                ),
                dtype=dtype,
            )
            -jnp.asarray(
                _dense_array(
                    pair["zero"]["jac_bc"](
                        u_flat,
                        internal_vars,
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
):
    """
    Assemble one concrete steady linear FEAX contribution.

    This helper deliberately receives an IR that no longer contains runtime
    parameter ModelCall nodes.
    """
    import feax as fe

    problem, bc = _build_feax_problem(
        domain,
        ir,
        apply_dirichlet=apply_dirichlet,
    )
    internal_vars = fe.InternalVars()

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

    has_runtime_parameters = any(_contains_runtime_parameter(term.coeff) for term in ir.terms)

    if not has_runtime_parameters:
        return _assemble_fem_system_concrete(
            domain,
            ir,
            apply_dirichlet=True,
            symmetric_bc=symmetric_bc,
        )

    static_ir, parameter_irs, runtime_parameter_exprs = _split_parametric_operator_ir(ir)

    operator_basis = {}
    rhs_basis = {}

    for name, basis_ir in parameter_irs.items():
        op_basis_ir, rhs_basis_ir = _split_trial_and_load_ir(basis_ir)

        if len(op_basis_ir.terms) > 0:
            zero_basis_ir = _make_zero_ir_like(op_basis_ir)
            K_bc, bK_bc = _assemble_fem_system_concrete(
                domain, op_basis_ir, apply_dirichlet=True, symmetric_bc=symmetric_bc
            )
            K_zero_bc, bK_zero_bc = _assemble_fem_system_concrete(
                domain, zero_basis_ir, apply_dirichlet=True, symmetric_bc=symmetric_bc
            )
            K = jnp.asarray(_dense_array(K_bc)) - jnp.asarray(_dense_array(K_zero_bc))
            bK = jnp.asarray(bK_bc).reshape(-1) - jnp.asarray(bK_zero_bc).reshape(-1)
            if not np.allclose(np.asarray(bK), 0.0, atol=1.0e-8):
                raise NotImplementedError(
                    "A runtime operator basis produced a non-zero RHS contribution. "
                    "Runtime Dirichlet parameters are not supported yet."
                )
            operator_basis[name] = K

        if len(rhs_basis_ir.terms) > 0:
            rhs_vec = _assemble_static_source_vector_from_ir(domain, rhs_basis_ir, dtype=_default_float_dtype())
            if rhs_vec is not None:
                rhs_basis[name] = jnp.asarray(rhs_vec)

    if len(static_ir.terms) == 0:
        structural_op_ir = None
        for basis_ir in parameter_irs.values():
            op_basis_ir, _ = _split_trial_and_load_ir(basis_ir)
            if len(op_basis_ir.terms) > 0:
                structural_op_ir = op_basis_ir
                break
        if structural_op_ir is None:
            raise ValueError(
                "A static fem_system requires at least one operator term. Only runtime source/load terms were found."
            )
        static_ir = _make_zero_ir_like(structural_op_ir)

    A0, b0 = _assemble_fem_system_concrete(domain, static_ir, apply_dirichlet=True, symmetric_bc=symmetric_bc)
    A0 = jnp.asarray(_dense_array(A0))
    b0 = jnp.asarray(b0, dtype=A0.dtype).reshape(-1)

    operator_fn = None
    if operator_basis:

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
            "dynamic_operator": bool(operator_basis),
            "dynamic_rhs": bool(rhs_basis),
            "runtime_parameter_names": sorted(runtime_parameter_exprs),
            "operator_parameter_names": sorted(operator_basis),
            "rhs_parameter_names": sorted(rhs_basis),
            "lowering": ("A(args) = A0 + sum_i args[name_i] * K_i; b(args) = b0 + sum_j args[name_j] * f_j"),
        },
    )
