"""PINO - Physics-Informed Neural Operators."""

from .core import core
from .domain import domain
from .resampling import sampler
from .trace import Variable, Placeholder, OperationDef, OperationCall, FlaxModule, Hessian, Jacobian
from .utils.adaptive import LearningRateSchedule, WeightSchedule
from .utils import callbacks, Logger, init_default_logger as logger, IREEModel as iree
from .trace_evaluator import TraceEvaluator
from . import resampling

__version__ = "0.1.0"


class ScheduleWrapper:

    constraint = WeightSchedule
    learning_rate = LearningRateSchedule


schedule = ScheduleWrapper()


__all__ = [
    "schedule",
    "core",
    "sampler",
    "domain",
    "FlaxModule",
    "Variable",
    "Placeholder",
    "OperationDef",
    "OperationCall",
    "resampling",
    "LearningRateSchedule",
    "WeightSchedule",
    "callbacks",
    "logger",
    "TraceEvaluator",
    "iree",
]
