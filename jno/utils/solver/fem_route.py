from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import jax.numpy as jnp

from ...trace import (
    FemResidualOperator,
)

from .feax_utils import (
    _default_float_dtype,
    _normalize_dirichlet_value,
    _build_feax_problem,
)
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

    Parameters
    ----------
    bcs:
        Iterable containing `DirichletBC` and `NeumannBC` descriptors.
    vec:
        Number of scalar components of the FEM unknown. For scalar problems this
        is `1`; for vector-valued problems this is the flattened component count.

    Returns
    -------
    tuple
        `(dirichlet_tags, dirichlet_value_fns, neumann_tags)`, where:

        - `dirichlet_tags` is an ordered list of essential-BC boundary tags.
        - `dirichlet_value_fns` maps each Dirichlet tag to FEAX-compatible value
          callable(s).
        - `neumann_tags` is an ordered list of natural/surface boundary tags.

    Raises
    ------
    TypeError
        If an unsupported BC descriptor is provided.
    """
    dirichlet_tags = []
    dirichlet_value_fns = {}
    neumann_tags = []

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
        else:
            raise TypeError(f"Unsupported BC entry '{type(bc).__name__}'. Use dirichlet(...) or neumann(...).")

    return dirichlet_tags, dirichlet_value_fns, neumann_tags


# --------------------------------
# public FEAX-backed entry points
# --------------------------------

def _assemble_fem_residual_from_ir(domain, ir, **kwargs):
    """
    Assemble a steady nonlinear FEM residual operator from lowered weak-form IR.

    This target is used by:

        weak_expr.assemble(target="fem_residual")

    It builds a FEAX problem from the lowered weak-form IR and returns callable
    residual and Jacobian functions for external Newton-like solvers.

    Parameters
    ----------
    domain:
        jNO domain with initialized FEM/FEAX context.
    ir:
        Lowered weak-form IR produced by `lower_weak_form(..., for_target="fem")`.
    **kwargs:
        Optional backend settings. Currently supports `symmetric_bc`.

    Returns
    -------
    FemResidualOperator
        Object containing:

        - `residual_fn(u_flat)`: evaluates the FEAX residual vector `r(u)`.
        - `jacobian_fn(u_flat)`: evaluates the FEAX Jacobian matrix `J(u)`.
        - `size`: total number of scalar FEM degrees of freedom.

    Notes
    -----
    This route is intended for nonlinear steady weak forms. Linear steady weak
    forms should usually use `_assemble_fem_system_from_ir(...)`, which returns
    the direct system `(A, b)`.
    """
    import feax as fe

    problem, bc = _build_feax_problem(domain, ir)
    internal_vars = fe.InternalVars()

    res_bc = fe.create_res_bc_function(problem, bc)
    jac_bc = fe.create_J_bc_function(problem, bc, symmetric=kwargs.get("symmetric_bc", True))
    size = int(problem.num_total_dofs_all_vars)

    def residual_fn(u_flat):
        u_flat = jnp.asarray(u_flat, dtype=_default_float_dtype())
        return jnp.asarray(res_bc(u_flat, internal_vars))

    def jacobian_fn(u_flat):
        u_flat = jnp.asarray(u_flat, dtype=_default_float_dtype())
        return jac_bc(u_flat, internal_vars)

    return FemResidualOperator(
        residual_fn=residual_fn,
        jacobian_fn=jacobian_fn,
        size=size,
    )


def _assemble_fem_system_from_ir(domain, ir, **kwargs):
    """
    Assemble a steady linear FEM system from lowered weak-form IR.

    This target is used by:

        weak_expr.assemble(target="fem_system")

    and is also the default auto-selected target for linear steady weak forms.

    Parameters
    ----------
    domain:
        jNO domain with initialized FEM/FEAX context.
    ir:
        Lowered weak-form IR produced by `lower_weak_form(..., for_target="fem")`.
    **kwargs:
        Optional backend settings. Currently supports `symmetric_bc`.

    Returns
    -------
    tuple
        `(A, b)` such that the physical FEM unknown satisfies:

            A @ u = b

    Notes
    -----
    FEAX naturally assembles a Newton/correction system:

        A @ du = -r(u0)

    This route converts that correction system into the full-state linear
    system expected by jNO examples and external linear solvers:

        A @ u = A @ u0 - r(u0)
    """
    import feax as fe

    problem, bc = _build_feax_problem(domain, ir)
    internal_vars = fe.InternalVars()

    try:
        u0 = fe.zero_like_initial_guess(problem, bc)
    except Exception:
        u0 = jnp.zeros((problem.num_total_dofs_all_vars,), dtype=_default_float_dtype())

    u0 = jnp.asarray(u0, dtype=_default_float_dtype())

    res_bc = fe.create_res_bc_function(problem, bc)
    jac_bc = fe.create_J_bc_function(problem, bc, symmetric=kwargs.get("symmetric_bc", True))

    # FEAX gives the correction system:
    #     A du = -r(u0)
    # Convert to the full-state system expected by jNO examples:
    #     A u = A u0 - r(u0)
    A = jac_bc(u0, internal_vars)
    r0 = jnp.asarray(res_bc(u0, internal_vars), dtype=_default_float_dtype())

    if hasattr(A, "__matmul__"):
        b = A @ u0 - r0
    else:
        A_dense = jnp.asarray(A.todense() if hasattr(A, "todense") else A.toarray())
        b = A_dense @ u0 - r0
        A = A_dense

    return A, b
