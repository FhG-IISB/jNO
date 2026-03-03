"""
jNO: Physics-Informed Neural Operators.

.. warning::
    This is a research-level repository. It may contain bugs and is subject
    to continuous change without notice.
"""

from .core import core
from .domain import domain
from .resampling import sampler
from .trace import Variable, Placeholder, OperationDef, OperationCall, Model, Hessian, Jacobian
from .utils.adaptive import LearningRateSchedule, WeightSchedule
from .utils import callbacks, Logger, init_default_logger as logger, IREEModel as iree
from .utils.config import (
    load_config,
    get_config,
    get_config_path,
    get_runs_base_dir,
    get_rsa_public_key,
    get_rsa_private_key,
    get_seed,
    setup,
)
from .trace_evaluator import TraceEvaluator
from . import resampling
from .utils.load_save import save, load

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
    "Model",
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
    "save",
    "load",
    "setup",
    "load_config",
    "get_config",
    "get_runs_base_dir",
    "get_rsa_public_key",
    "get_rsa_private_key",
    "get_seed",
]
