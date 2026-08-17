"""
jNO: Physics-Informed Neural Operators.

.. warning::
    This is a research-level repository. It may contain bugs and is subject
    to continuous change without notice.
"""

import sys

from . import bayesian, fn, litho, lora, optimizers, precond, solve, trackers
from . import jnp_ops as np
from ._fem import Coupling, fem, lag
from .architectures.models import nn, parameter
from .core import core
from .differential_operators import DifferentialOperators
from .domain import domain
from .fdm import fdm
from .geometry import Path, Shape
from .integration_operators import IntegrationOperators
from .noise import noise
from .rcwa import Rcwa, RcwaError, rcwa
from .trace import (
    Assembly,
    Constraint,
    FemLinearSystem,
    FemResidualOperator,
    GroupedAssembly,
    Hessian,
    Integral,
    IntegralTime,
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
    units,
)
from .trace.views import (
    ComplexPair,
    ComplexView,
    MatrixView,
    NamedComplexViewWithPartials,
    NamedMatrixView,
    NamedMatrixViewWithPartials,
    NamedScalarViewWithPartials,
    NamedVectorView,
    NamedVectorViewWithPartials,
    NamedVoigtViewWithPartials,
    ScalarView,
    VectorView,
    VoigtView,
)
from .trace_compiler import TraceCompiler
from .trace_evaluator import TraceEvaluator
from .utils import IREEModel as iree
from .utils import Logger
from .utils import init_default_logger as logger
from .utils.adaptive import LearningRateSchedule, WeightSchedule, callbacks, sampler
from .utils.config import (
    _auto_compile_cache,
    disable_compile_cache,
    enable_compile_cache,
    get_config,
    get_config_path,
    get_rsa_private_key,
    get_rsa_public_key,
    get_runs_base_dir,
    get_seed,
    load_config,
    setup,
    wandb_finish,
)
from .utils.load_save import load, save
from .utils.solver.fem_route import dirichlet, neumann

# Mirror the submodule on the package namespace and add a short alias.
numpy = np
# Backward compatibility: allow `import jno.numpy` after renaming internals.
sys.modules[__name__ + ".numpy"] = np
do = domain


# `jno.complex(re, im)` builds a complex quantity from two real parts (the explicit form of
# `re + 1j*im`) — for complex *data*/coefficients in a weak form, e.g. a complex forcing
# `J = jno.complex(Jr, Ji)`. Complex *fields* come from `domain.fem_symbols(..., complex=True)`.
def complex(re, im=0.0):  # noqa: A001  (intentionally shadows builtin on the jno namespace)
    """A complex quantity as two real parts → :class:`ComplexPair` (``re + 1j*im``)."""
    return ComplexPair(re, im)


# Single source of truth: pyproject.toml. importlib.metadata reads the
# installed-package metadata so __version__ stays aligned with the wheel
# without manual edits. The fallback covers running from a source checkout
# without `pip install -e .`.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("jax-numerical-operators")
except (ImportError, PackageNotFoundError):
    __version__ = "unknown"


class ScheduleWrapper:
    constraint = WeightSchedule
    learning_rate = LearningRateSchedule


schedule = ScheduleWrapper()


def le(expr, bound=0.0):
    """An inequality constraint ``expr <= bound``, as an entry in the ``jno.core`` list.

    It is evaluated and differentiated exactly like any other entry -- so its value and gradient
    reach a constrained optimiser -- but it is **not summed into the loss**. Adding a constraint to
    the objective makes it a soft penalty, which double-counts against an optimiser that already
    handles it in the dual (MMA, OC, SQP)::

        jno.core([compliance,                       # minimised
                  jno.le(volume / v_star, 1.0),     # constraint, handed to the optimiser
                  1e-3 * reg.mean])                 # a genuine penalty, summed

    With no constrained optimiser attached this raises at ``solve()`` rather than silently
    ignoring the constraint.
    """
    return Constraint(expr, bound, "le")


def ge(expr, bound=0.0):
    """An inequality constraint ``expr >= bound``. See :func:`le`."""
    return Constraint(expr, bound, "ge")


__all__ = [
    "schedule",
    "core",
    "le",
    "ge",
    "Constraint",
    "sampler",
    "domain",
    "Shape",
    "Path",
    "do",
    "fem",
    "fdm",
    "rcwa",
    "Rcwa",
    "RcwaError",
    "litho",
    "Coupling",
    "Model",
    "Variable",
    "Placeholder",
    "OperationDef",
    "OperationCall",
    "LearningRateSchedule",
    "WeightSchedule",
    "callbacks",
    "trackers",
    "optimizers",
    "logger",
    "TraceEvaluator",
    "TraceCompiler",
    "DifferentialOperators",
    "IntegrationOperators",
    "Integral",
    "IntegralTime",
    "iree",
    "save",
    "load",
    "disable_compile_cache",
    "enable_compile_cache",
    "setup",
    "wandb_finish",
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
    "units",
    "StateField",
    "ScalarView",
    "VectorView",
    "ComplexView",
    "MatrixView",
    "NamedMatrixView",
    "NamedVectorView",
    "VoigtView",
    "NamedScalarViewWithPartials",
    "NamedVectorViewWithPartials",
    "NamedComplexViewWithPartials",
    "NamedMatrixViewWithPartials",
    "NamedVoigtViewWithPartials",
]

# Persistent XLA compilation cache, ON by default (opt out: JNO_COMPILE_CACHE=0, `[jno]
# compile_cache = false`, or jno.setup(compile_cache=False)). Measured 2.1x-4.3x on warm builds;
# see `jno.utils.config.enable_compile_cache` for the numbers and the first-run populate cost.
_auto_compile_cache()
