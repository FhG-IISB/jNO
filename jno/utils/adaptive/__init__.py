"""
jno.utils.adaptive
==================
Adaptive scheduling package for PINNs.

Use new-style imports only:

    from jno.utils.adaptive.lrscheduler import LearningRateSchedule, DLRS, dlrs
    from jno.utils.adaptive.weights import (
        WeightSchedule,
        ReLoBRaLo, relobralo,
        LbPINNsLossBalancing, lbpinns_loss_balancing,
    )
"""

from .lrscheduler import LearningRateSchedule
from .weights import WeightSchedule

__all__ = [
    "LearningRateSchedule",
    "WeightSchedule",
]