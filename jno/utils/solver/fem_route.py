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
    tags: tuple[str, ...]
    values: object = None


@dataclass(frozen=True)
class NeumannBC:
    tags: tuple[str, ...]


def _as_tags(tags) -> tuple[str, ...]:
    if isinstance(tags, str):
        return (tags,)
    if isinstance(tags, Sequence):
        out = tuple(str(t) for t in tags)
        if len(out) == 0:
            raise ValueError("Boundary tag list cannot be empty.")
        return out
    raise TypeError(f"Boundary tags must be a string or a sequence of strings, got {type(tags).__name__}.")


def dirichlet(tags, values=None):
    return DirichletBC(tags=_as_tags(tags), values=values)


def neumann(tags):
    return NeumannBC(tags=_as_tags(tags))



def expand_bcs(bcs, vec: int):
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
