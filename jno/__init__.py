"""
jNO: Physics-Informed Neural Operators.

.. warning::
    This is a research-level repository. It may contain bugs and is subject
    to continuous change without notice.
"""

import sys

from .core import core
from .domain import domain
from .utils.fem_route import dirichlet, neumann
from .resampling import sampler
from .trace import Variable, Placeholder, OperationDef, OperationCall, Model, Hessian, Jacobian, TestFunction, TrialFunction, Assembly, FemLinearSystem, GroupedAssembly, FemResidualOperator
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
from .trace_compiler import TraceCompiler
from .differential_operators import DifferentialOperators
from . import resampling
from . import jnp_ops as np
from .utils.load_save import save, load
from .architectures.models import nn, parameter

# Mirror the submodule on the package namespace and add a short alias.
numpy = np
# Backward compatibility: allow `import jno.numpy` after renaming internals.
sys.modules[__name__ + ".numpy"] = np
do = domain


__version__ = "0.1.0"


class ScheduleWrapper:

    constraint = WeightSchedule
    learning_rate = LearningRateSchedule


schedule = ScheduleWrapper()


class _CallbackNamespace:
    """Namespace for callback constructors: ``jno.callback.checkpoint(...)``."""

    from .utils.callbacks import Callback as base  # noqa: F401
    from .utils.callbacks import CheckpointCallback as checkpoint  # noqa: F401


callback = _CallbackNamespace()


__all__ = [
    "schedule",
    "core",
    "sampler",
    "domain",
    "do",
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
    "TraceCompiler",
    "DifferentialOperators",
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
    "TestFunction",
    "TrialFunction",
    "Assembly",
    "FemLinearSystem",
    "GroupedAssembly",
    "FemResidualOperator",
    "dirichlet",
    "neumann",
    "numpy",
    "nn",
    "np",
    "callback",
]
