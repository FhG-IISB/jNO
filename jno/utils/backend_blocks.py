"""
Backward-compatibility shim for solver backend blocks.

Real implementation lives in jno.utils.solver.backend_blocks and
jno.utils.solver.time_adapters.
"""

from __future__ import annotations

from .solver.backend_blocks import (  # noqa: F401
    DiffraxBlock,
    FeaxTimeBlock,
    FeaxPipelineBlock,
)

from .solver.time_adapters import (  # noqa: F401
    make_diffrax_block,
    make_feax_pipeline,
)

__all__ = [
    "DiffraxBlock",
    "FeaxTimeBlock",
    "FeaxPipelineBlock",
    "make_diffrax_block",
    "make_feax_pipeline",
]