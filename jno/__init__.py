"""
jNO: Physics-Informed Neural Operators.

.. warning::
    This is a research-level repository. It may contain bugs and is subject
    to continuous change without notice.
"""

import sys

from . import fn, lora
from . import jnp_ops as np
from .noise import noise
from .architectures.models import nn, parameter
from .core import core
from .differential_operators import DifferentialOperators
from .domain import PolygonDomain, domain
from .integration_operators import IntegrationOperators
from .trace import (
    Assembly,
    FemLinearSystem,
    FemResidualOperator,
    GroupedAssembly,
    Hessian,
    Integral,
    Jacobian,
    Model,
    NetworkGradient,
    OperationCall,
    OperationDef,
    Placeholder,
    StateField,
    TestFunction,
    TrialFunction,
    Variable,
)
from .trace_compiler import TraceCompiler
from .trace_evaluator import TraceEvaluator
from .utils import IREEModel as iree
from .utils import Logger
from .utils import init_default_logger as logger
from .utils.adaptive import LearningRateSchedule, WeightSchedule, callbacks, sampler
from .utils.config import (
    get_config,
    get_config_path,
    get_rsa_private_key,
    get_rsa_public_key,
    get_runs_base_dir,
    get_seed,
    load_config,
    setup,
)
from .utils.load_save import load, save
from .utils.solver.fem_route import dirichlet, neumann

# Mirror the submodule on the package namespace and add a short alias.
numpy = np
# Backward compatibility: allow `import jno.numpy` after renaming internals.
sys.modules[__name__ + ".numpy"] = np
do = domain


__version__ = "0.2.1"


class ScheduleWrapper:
    constraint = WeightSchedule
    learning_rate = LearningRateSchedule


schedule = ScheduleWrapper()

__all__ = [
    "schedule",
    "core",
    "sampler",
    "domain",
    "do",
    "PolygonDomain",
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
    "IntegrationOperators",
    "Integral",
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
    "fn",
    "lora",
    "noise",
    "callback",
    "StateField",
]
