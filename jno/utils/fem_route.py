"""
Backward-compatibility shim for FEM/FEAX routing.

The real implementation now lives in:

    jno.utils.solver.fem_route
    jno.utils.solver.feax_utils

This module is kept so old imports still work, for example:

    from jno.utils.fem_route import dirichlet, neumann
    from jno.utils.fem_route import _assemble_fem_system_from_ir
"""

from __future__ import annotations

# Public FEM boundary-condition API
from .solver.fem_route import (
    DirichletBC,
    NeumannBC,
    dirichlet,
    neumann,
    expand_bcs,
    _assemble_fem_residual_from_ir,
    _assemble_fem_system_from_ir,
)

# Temporary/private backward-compatible FEAX helpers.
# These should eventually be imported directly from
# jno.utils.solver.feax_utils by internal modules.
from .solver.feax_utils import (
    _default_float_dtype,
    _lower_statefield_to_trial,
    _const_bc_fn,
    _normalize_dirichlet_value,
    _value_shape_num_components,
    _reshape_components_last,
    _expand_test_shape_vals,
    _infer_trial_metadata,
    _collect_temporal_tags_for_feax,
    _temporal_value_from_internal_vars,
    _eval_expr_for_feax,
    _eval_volume_integrand,
    _eval_surface_integrand,
    _make_universal_volume_kernel,
    _make_universal_surface_kernel,
    _meshio_type_for_element,
    _build_feax_mesh,
    _make_feax_dirichlet_specs,
    _build_feax_problem,
)

__all__ = [
    # Public BC API
    "DirichletBC",
    "NeumannBC",
    "dirichlet",
    "neumann",
    "expand_bcs",

    # FEM assembly entry points
    "_assemble_fem_residual_from_ir",
    "_assemble_fem_system_from_ir",

    # Temporary FEAX helper compatibility exports
    "_default_float_dtype",
    "_lower_statefield_to_trial",
    "_const_bc_fn",
    "_normalize_dirichlet_value",
    "_value_shape_num_components",
    "_reshape_components_last",
    "_expand_test_shape_vals",
    "_infer_trial_metadata",
    "_collect_temporal_tags_for_feax",
    "_temporal_value_from_internal_vars",
    "_eval_expr_for_feax",
    "_eval_volume_integrand",
    "_eval_surface_integrand",
    "_make_universal_volume_kernel",
    "_make_universal_surface_kernel",
    "_meshio_type_for_element",
    "_build_feax_mesh",
    "_make_feax_dirichlet_specs",
    "_build_feax_problem",
]