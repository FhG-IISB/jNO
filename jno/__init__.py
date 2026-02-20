"""PINO - Physics-Informed Neural Operators."""

from .core import core
from .domain import domain
from .resampling import sampler
from .trace import Variable, Placeholder, OperationDef, OperationCall, FlaxModule
from .utils.adaptive import LearningRateSchedule, WeightSchedule
from .utils import callbacks, Logger, init_default_logger as logger, IREEModel as iree
from .utils import create_rank_dict
from .trace_evaluator import TraceEvaluator
from . import resampling
from contextlib import contextmanager

__version__ = "0.1.0"


class ScheduleWrapper:

    constraint = WeightSchedule
    learning_rate = LearningRateSchedule


schedule = ScheduleWrapper()


@contextmanager
def operations():
    """Context manager for defining pino operations.

    Operations defined within this block can use the simpler syntax
    without explicit pino.operation() wrapping. The wrapping happens
    automatically when expressions are called or passed to solve().

    Example:
        with pino.operations():
            u = pnp.concat([x, y]) >> pino.dense(64) >> pino.dense(1)
            pde = pnp.laplacian(u(x, y), [x, y]) - f(x, y)

        sol = pino.solve([pde], ...)
    """
    yield


__all__ = [
    "ParallelConfig",
    "VmapConfig",
    "schedule",
    "tuner",
    "core",
    "sampler",
    "solve",
    "domain",
    "operation",
    "operations",
    "dense",
    "model",
    "FlaxModule",
    "laplace",
    "grad",
    "concat",
    "Variable",
    "Placeholder",
    "OperationDef",
    "OperationCall",
    "variable",
    "resampling",
    "LearningRateSchedule",
    "WeightSchedule",
    "Inferer",
    "callbacks",
    "logger",
    "TraceEvaluator",
    "create_rank_dict",
    "iree",
]
