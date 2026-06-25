"""
Solver utility layer for jNO backend routing.

This package contains shared helpers used by FEM, weak-form routing,
time routing, and solver adapters.
"""

from .backend_blocks import (
    SemidiscreteTimeBlock,
)
from .fem_route import (
    DirichletBC,
    NeumannBC,
    dirichlet,
    expand_bcs,
    neumann,
)
from .solver_helper import (
    apply_sign,
    collect_temporal_tags,
    contains_model_call,
    contains_model_eval,
    contains_node_type,
    contains_subexpr,
    contains_temporal_derivative,
    contains_testfunction,
    contains_trialfunction,
    depends_on_domain_variables,
    is_temporal_var,
    iter_children,
    iter_placeholder_children,
    max_temporal_derivative_order,
    sum_terms,
    unique_by_id,
)
from .weak_form import (
    LoweredChannelTerm,
    LoweredWeakForm,
    assemble_weak_form,
    lower_weak_form,
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
    "SemidiscreteTimeBlock",
    "split_weak_additive_terms",
]
