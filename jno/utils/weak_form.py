"""
Backward-compatible weak-form import path.

The real implementation lives in:
    jno.utils.solver.weak_form
"""

from .solver.weak_form import *  # noqa: F401,F403

# Explicit private-name compatibility for old internal imports.
from .solver.weak_form import (
    _sum_terms,
    _apply_sign,
    _contains_node_type,
    _contains_testfunction,
    _contains_trialfunction,
    _is_obviously_nonlinear_in_unknown,
    _assemble_vpinn_from_ir,
)