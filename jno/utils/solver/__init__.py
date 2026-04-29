"""
Solver utility layer for jNO backend routing.

This package contains shared helpers used by FEM/FEAX, time routing,
and solver adapters. It should not contain frontend DSL code.
"""

from .solver_helper import (
    contains_node_type,
    contains_testfunction,
    contains_trialfunction,
    sum_terms,
    apply_sign,
)

from .weak_form_helpers import (
    split_weak_additive_terms,
)

__all__ = [
    "contains_node_type",
    "contains_testfunction",
    "contains_trialfunction",
    "sum_terms",
    "apply_sign",
    "split_weak_additive_terms",
]