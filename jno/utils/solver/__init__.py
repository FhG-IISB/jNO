"""
Solver utility layer for jNO backend routing.

This package contains shared helpers used by FEM/FEAX, time routing,
and solver adapters.
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
    "split_weak_additive_terms",
    "is_temporal_var",
    "max_temporal_derivative_order",
    "collect_temporal_tags",
    "contains_temporal_derivative",
]