"""
Solver utility layer for jNO backend routing.

This package contains shared helpers used by FEM/FEAX, weak-form routing,
time routing, and solver adapters.
"""

from .solver_helper import (
    iter_children,
    iter_placeholder_children,
    contains_node_type,
    contains_testfunction,
    contains_trialfunction,
    contains_model_call,
    contains_model_eval,
    depends_on_domain_variables,
    contains_subexpr,
    unique_by_id,
    sum_terms,
    apply_sign,
    is_temporal_var,
    max_temporal_derivative_order,
    collect_temporal_tags,
    contains_temporal_derivative,
)

from .fem_route import (
    DirichletBC,
    NeumannBC,
    dirichlet,
    neumann,
    expand_bcs,
)

from .weak_form import (
    LoweredChannelTerm,
    LoweredWeakForm,
    lower_weak_form,
    assemble_weak_form,
)

from .backend_blocks import (
    DiffraxBlock,
    FeaxTimeBlock,
    FeaxPipelineBlock,
)

from .time_adapters import (
    make_diffrax_block,
    make_feax_pipeline,
)

from .weak_form_helpers import (
    split_weak_additive_terms,
)

__all__ = [
    "iter_children",
    "iter_placeholder_children",
    "contains_node_type",
    "contains_testfunction",
    "contains_trialfunction",
    "contains_model_call",
    "contains_model_eval",
    "depends_on_domain_variables",
    "contains_subexpr",
    "unique_by_id",
    "sum_terms",
    "apply_sign",
    "is_temporal_var",
    "max_temporal_derivative_order",
    "collect_temporal_tags",
    "contains_temporal_derivative",
    "DirichletBC",
    "NeumannBC",
    "dirichlet",
    "neumann",
    "expand_bcs",
    "LoweredChannelTerm",
    "LoweredWeakForm",
    "lower_weak_form",
    "assemble_weak_form",
    "DiffraxBlock",
    "FeaxTimeBlock",
    "FeaxPipelineBlock",
    "make_diffrax_block",
    "make_feax_pipeline",
    "split_weak_additive_terms",
]