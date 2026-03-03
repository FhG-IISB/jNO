"""
Tracing system for pino operations.

Variables are placeholders that get filled during solve iterations.
Operations trace computations and return callable placeholders.
"""

from typing import List, Callable, Any, Union, Dict, Tuple, Optional, Type
from dataclasses import dataclass, field
from .tuner import Arch, ArchSpace
import jax
import jax.numpy as jnp
import equinox as eqx
from pathlib import Path
import json

__all__ = [
    "Placeholder",
    "FunctionCall",
    "Literal",
    "ConstantNamespace",
    "Constant",
    "Variable",
    "TensorTag",
    "BinaryOp",
    "Tracker",
    "Model",
    "TunableModule",
    "TunableModuleCall",
    "ModelCall",
    "OperationDef",
    "OperationCall",
    "Hessian",
    "Jacobian",
    "collect_operations",
    "collect_tags",
    "get_primary_tag",
    "dump_tree",
    "cse",
]

# Global counter for unique operation IDs
_operation_counter = 0


def _next_op_id() -> int:
    global _operation_counter
    _operation_counter += 1
    return _operation_counter


class Placeholder:
    """Base node for the traced DSL graph.

    Placeholders behave like symbolic tensors: arithmetic and composition
    operators create new traced nodes instead of executing eagerly. Calling a
    placeholder (`u(x)`) auto-wraps it in an `OperationDef` so it can be reused
    with different inputs. Concrete values are only produced when evaluated by
    the solver/visualizer.

    Note: ``__eq__`` and ``__hash__`` use object identity so that Placeholder
    instances can safely be stored in sets and used as dict keys.  For
    element-wise symbolic equality comparisons use ``Placeholder.equal(other)``.
    """

    # -- identity-based equality so Placeholders work in sets/dicts -----------
    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)

    # -- symbolic comparison operators (return traced FunctionCall nodes) ------
    def equal(self, other):
        """Element-wise symbolic equality (traced, not Python ``==``)."""
        return FunctionCall(jnp.equal, [self, other])

    def not_equal(self, other):
        """Element-wise symbolic inequality (traced, not Python ``!=``)."""
        return FunctionCall(jnp.not_equal, [self, other])

    def __gt__(self, other):
        return FunctionCall(jnp.greater, [self, other])

    def __lt__(self, other):
        return FunctionCall(jnp.less, [self, other])

    def __ge__(self, other):
        return FunctionCall(jnp.greater_equal, [self, other])

    def __le__(self, other):
        return FunctionCall(jnp.less_equal, [self, other])

    def _wrap(self, other):
        """Wrap non-Placeholder types."""
        if isinstance(other, Placeholder):
            return other
        return Literal(other)

    def __add__(self, other):
        return BinaryOp("+", self, self._wrap(other))

    def __radd__(self, other):
        return BinaryOp("+", self._wrap(other), self)

    def __sub__(self, other):
        return BinaryOp("-", self, self._wrap(other))

    def __rsub__(self, other):
        return BinaryOp("-", self._wrap(other), self)

    def __mul__(self, other):
        return BinaryOp("*", self, self._wrap(other))

    def __rmul__(self, other):
        return BinaryOp("*", self._wrap(other), self)

    def __truediv__(self, other):
        return BinaryOp("/", self, self._wrap(other))

    def __rtruediv__(self, other):
        return BinaryOp("/", self._wrap(other), self)

    def __neg__(self):
        return BinaryOp("*", Literal(-1.0), self)

    def __pow__(self, other):
        return BinaryOp("**", self, self._wrap(other))

    def __rpow__(self, other):
        return BinaryOp("**", self._wrap(other), self)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        concrete_key = tuple(None if k is None else k for k in key)
        return FunctionCall(lambda x, k=concrete_key: x[k], [self], name="getitem")

    def __call__(self, *args):
        """Call this expression with different variables (auto-wraps in operation)."""
        # Auto-wrap this expression in an OperationDef if not already one
        if not hasattr(self, "_auto_op"):
            self._auto_op = OperationDef(self)
        return self._auto_op(*args)

    def __matmul__(self, other):
        """Matrix multiplication: self @ other"""
        other = self._wrap(other)

        # symbolic path
        if isinstance(other, Placeholder):
            return FunctionCall(lambda a, b: a @ b, [self, other])

        # eager path (other is ndarray)
        return FunctionCall(lambda a: a @ other, [self])

    def __rmatmul__(self, other):
        """Matrix multiplication: other @ self"""
        other = self._wrap(other)

        if isinstance(other, Placeholder):
            return FunctionCall(lambda a, b: a @ b, [other, self])

        return FunctionCall(lambda b: other @ b, [self])

    def reshape(self, *shape):
        """Reshape this placeholder to a new shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return FunctionCall(lambda x, s=shape: x.reshape(s), [self], name="reshape")

    @property
    def shape(self):
        return FunctionCall(lambda x: jnp.ones(x.shape, dtype="bool"), [self], "shape", True)

    @property
    def mean(self):
        return FunctionCall(lambda x: jnp.squeeze(x.mean()), [self], "mean", True)

    @property
    def sum(self):
        return FunctionCall(lambda x: jnp.squeeze(jnp.sum(x)), [self], "sum", True)

    @property
    def min(self):
        return FunctionCall(lambda x: jnp.squeeze(jnp.min(x)), [self], "min", True)

    @property
    def max(self):
        return FunctionCall(lambda x: jnp.squeeze(jnp.max(x)), [self], "max", True)

    @property
    def std(self):
        return FunctionCall(lambda x: jnp.squeeze(jnp.std(x)), [self], "std", True)

    @property
    def mse(self):
        def fn(x):
            return jnp.squeeze(jnp.mean(jnp.square(x)))

        return FunctionCall(fn, [self], "mse", True)

    @property
    def mae(self):
        return FunctionCall(lambda x: jnp.squeeze(jnp.mean(jnp.abs(x))), [self], "mae", True)

    @property
    def T(self):
        return FunctionCall(lambda x: x.T, [self], "transpose", True)


class FunctionCall(Placeholder):
    """Call to a pure function over traced args."""

    def __init__(
        self,
        fn: Callable,
        args: tuple,
        name: str = None,
        reduces_axis: int = None,
        kwargs: Dict = None,
    ):
        self.fn = fn
        self.args = args if isinstance(args, (list, tuple)) else [args]
        self._name = name
        self.reduces_axis = reduces_axis
        self.kwargs = kwargs

    def __repr__(self):
        name = self._name or getattr(self.fn, "__name__", str(self.fn))
        args_str = ", ".join(str(a) for a in self.args)
        return f"{name}({args_str})"

    def copy_with_args(self, new_args):
        """Create a new instance with different args."""
        return FunctionCall(fn=self.fn, args=new_args, name=self._name, reduces_axis=self.reduces_axis, kwargs=self.kwargs)

    def __call__(self, args):
        """Return a new FunctionCall with the given args."""
        return self.copy_with_args([args])


class Literal(Placeholder):
    """Concrete scalar/array embedded in the trace (no trainable params)."""

    def __init__(self, value):
        self.value = jnp.asarray(value)

    def __repr__(self):
        return f"Literal({self.value})"


class ConstantNamespace:
    """Wrapper that allows attribute access to constants (P.k syntax).

    Supports loading from:
    - dict: Direct dictionary of key-value pairs
    - str/Path: File path to .json, .yaml, .yml, .toml, .pkl, .pickle files

    All numeric values are pre-converted to JAX arrays at load time.
    Nested dictionaries become nested ConstantNamespace objects.
    """

    def __init__(self, tag: str, data: Union[dict, str, Path], _parent_tag: str = None):
        self._tag = tag
        self._full_tag = f"{_parent_tag}.{tag}" if _parent_tag else tag
        self._data = self._load_and_convert(data)
        self._constants = {}

    def _load_and_convert(self, data: Union[dict, str, Path]) -> dict:
        """Load data from dict or file path and convert values to JAX arrays."""
        raw_data = self._load_data(data)
        return self._convert_to_jax(raw_data, self._full_tag)

    @staticmethod
    def _convert_to_jax(data: dict, parent_tag: str = None) -> dict:
        """Recursively convert all numeric values to JAX arrays and dicts to namespaces."""
        converted = {}
        for key, value in data.items():
            converted[key] = ConstantNamespace._convert_value(value, key, parent_tag)
        return converted

    @staticmethod
    def _convert_value(value: Any, key: str = None, parent_tag: str = None) -> Any:
        """Convert a single value to JAX array if numeric, or ConstantNamespace if dict."""
        # Nested dictionary -> nested ConstantNamespace
        if isinstance(value, dict):
            return ConstantNamespace(key, value, _parent_tag=parent_tag)

        # Already a JAX array
        if isinstance(value, jnp.ndarray):
            return value

        # NumPy array -> JAX array
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return jnp.asarray(value)
        except ImportError:
            pass

        # Numeric scalar (int, float)
        if isinstance(value, (int, float)):
            return jnp.asarray(value)

        # List/tuple - check if numeric or contains dicts
        if isinstance(value, (list, tuple)):
            # Check if it contains dicts (don't convert to array)
            if any(isinstance(item, dict) for item in value):
                # Convert each dict to ConstantNamespace, keep others as-is
                return [
                    (ConstantNamespace(f"{key}[{i}]", item, _parent_tag=parent_tag) if isinstance(item, dict) else ConstantNamespace._convert_value(item, f"{key}[{i}]", parent_tag))
                    for i, item in enumerate(value)
                ]
            # Check if it's numeric (could be nested arrays)
            if ConstantNamespace._is_numeric_sequence(value):
                return jnp.asarray(value)
            # Otherwise keep as-is (e.g., list of strings)
            return value

        # Non-numeric (strings, etc.) -> keep as-is
        return value

    @staticmethod
    def _is_numeric_sequence(seq) -> bool:
        """Check if a sequence contains only numeric values (possibly nested)."""
        if not seq:
            return True

        for item in seq:
            if isinstance(item, (int, float)):
                continue
            elif isinstance(item, (list, tuple)):
                if not ConstantNamespace._is_numeric_sequence(item):
                    return False
            else:
                # numpy arrays are numeric
                try:
                    import numpy as np

                    if isinstance(item, (np.ndarray, np.generic)):
                        continue
                except ImportError:
                    pass

                # JAX arrays are numeric
                if isinstance(item, jnp.ndarray):
                    continue

                return False
        return True

    def _load_data(self, data: Union[dict, str, Path]) -> dict:
        """Load data from dict or file path."""
        # Already a dict
        if isinstance(data, dict):
            return data

        # Convert to Path
        path = Path(data)

        if not path.exists():
            raise FileNotFoundError(f"Constant file not found: {path}")

        suffix = path.suffix.lower()

        # JSON
        if suffix == ".json":
            return self._load_json(path)

        # YAML
        elif suffix in (".yaml", ".yml"):
            return self._load_yaml(path)

        # TOML
        elif suffix == ".toml":
            return self._load_toml(path)

        # Pickle (for numpy arrays, etc.)
        elif suffix in (".pkl", ".pickle"):
            return self._load_pickle(path)

        # NumPy .npz
        elif suffix == ".npz":
            return self._load_npz(path)

        # NumPy .npy (single array - wrap in dict)
        elif suffix == ".npy":
            return self._load_npy(path)

        else:
            raise ValueError(f"Unsupported file format: '{suffix}'. " f"Supported formats: .json, .yaml, .yml, .toml, .pkl, .pickle, .npz, .npy")

    @staticmethod
    def _load_json(path: Path) -> dict:
        """Load JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Load YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load .yaml/.yml files. " "Install with: pip install pyyaml")

        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_toml(path: Path) -> dict:
        """Load TOML file."""
        try:
            # Python 3.11+ has tomllib in stdlib
            import tomllib

            with open(path, "rb") as f:
                return tomllib.load(f)
        except ImportError:
            try:
                import toml

                with open(path, "r") as f:
                    return toml.load(f)
            except ImportError:
                raise ImportError("toml package is required to load .toml files. " "Install with: pip install toml")

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        """Load pickle file."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Pickle file must contain a dict, got {type(data).__name__}")
        return data

    @staticmethod
    def _load_npz(path: Path) -> dict:
        """Load NumPy .npz file."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy is required to load .npz files.")

        npz = np.load(path, allow_pickle=True)
        return dict(npz)

    @staticmethod
    def _load_npy(path: Path) -> dict:
        """Load NumPy .npy file (single array, wrapped in dict with filename as key)."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy is required to load .npy files.")

        arr = np.load(path, allow_pickle=True)
        key = path.stem  # Use filename without extension as key
        return {key: arr}

    def __getattr__(self, key: str):
        # Avoid recursion for private attributes
        if key.startswith("_"):
            raise AttributeError(key)

        if key not in self._data:
            available = list(self._data.keys())
            raise AttributeError(f"Constant '{self._full_tag}' has no key '{key}'. " f"Available keys: {available}")

        value = self._data[key]

        # If it's already a ConstantNamespace, return it directly
        if isinstance(value, ConstantNamespace):
            return value

        # Lazy creation of Constant objects for leaf values
        if key not in self._constants:
            self._constants[key] = Constant(self._full_tag, key, value)

        return self._constants[key]

    def __getitem__(self, key: str):
        """Support P["key"] syntax as well."""
        try:
            return self.__getattr__(key)
        except AttributeError as e:
            raise KeyError(str(e))

    def __contains__(self, key: str) -> bool:
        """Support 'key in P' syntax."""
        return key in self._data

    def __iter__(self):
        """Iterate over keys."""
        return iter(self._data.keys())

    def keys(self):
        """Return available keys."""
        return self._data.keys()

    def values(self):
        """Return Constant objects or nested namespaces for all keys."""
        return [self.__getattr__(k) for k in self._data.keys()]

    def items(self):
        """Return (key, Constant/ConstantNamespace) pairs."""
        return [(k, self.__getattr__(k)) for k in self._data.keys()]

    def to_dict(self) -> dict:
        """Recursively convert back to a plain dictionary."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, ConstantNamespace):
                result[key] = value.to_dict()
            elif isinstance(value, jnp.ndarray):
                result[key] = value
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, ConstantNamespace) else item for item in value]
            else:
                result[key] = value
        return result

    def __repr__(self):
        def format_keys(data, indent=0):
            lines = []
            for key, value in data.items():
                if isinstance(value, ConstantNamespace):
                    lines.append(f"{'  ' * indent}{key}:")
                    lines.extend(format_keys(value._data, indent + 1))
                else:
                    lines.append(f"{'  ' * indent}{key}")
            return lines

        keys_repr = ", ".join(format_keys(self._data))
        return f"ConstantNamespace({self._full_tag}, keys=[{keys_repr}])"

    def __len__(self):
        return len(self._data)


class Constant(Placeholder):
    """Concrete scalar/array embedded in the trace.

    Values are pre-converted to JAX arrays at creation time.
    """

    def __init__(self, tag: str, key: str, value: Any):
        self.tag = tag
        self.key = key
        self.value = value  # Already a jnp.ndarray from ConstantNamespace

    def __repr__(self):
        # Truncate large arrays for display
        if hasattr(self.value, "shape"):
            if self.value.shape == ():
                val_repr = f"{float(self.value)}"
            else:
                val_repr = f"array{self.value.shape}"
        elif isinstance(self.value, (list, tuple)) and len(self.value) > 5:
            val_repr = f"{type(self.value).__name__}[{len(self.value)}]"
        else:
            val_repr = repr(self.value)
        return f"Constant({self.tag}.{self.key}={val_repr})"


class Variable(Placeholder):
    """Independent variable placeholder (e.g., `x`, `y`, `t`).

    Carries the domain tag and dimension index so the solver can bind sampled
    coordinates when evaluating traced expressions.

    For time-dependent problems, spatial variables (``axis='spatial'``) index
    into the spatial context array ``context[tag]`` shaped ``(N, D_spatial)``
    (after the outer B and T vmaps peel off their axes).  The temporal
    variable (``axis='temporal'``) reads from a separate
    ``context["__time__"]`` entry that is a scalar (after the T vmap).
    """

    def __init__(self, tag: str, dim: list, domain=None, axis: str = "spatial"):
        self.tag = tag
        self.dim = dim
        self.axis = axis  # 'spatial' or 'temporal'
        if tag in domain.context.keys():
            self.size = dim[1] - dim[0] if dim[1] is not None else domain.context[tag].shape[-1]
        else:
            raise KeyError(f"Variable tag '{tag}' not found in domain.context. Available: {list(domain.context.keys())}")
        self._domain = domain  # Reference to parent domain for inference

    def __repr__(self):
        if self.axis == "temporal":
            return f"Var(t)"
        return f"Var({self.tag}[{self.dim}])"


class TensorTag(Placeholder):
    """Named tensor tag used for parametric inputs or coefficients.

    Backed by arrays attached to the domain; when domains merge, tags stack
    along the batch dimension.
    """

    def __init__(self, tag: str, domain=None, dim_index: int = None):
        self.tag = tag
        self._domain = domain
        self.dim_index = dim_index  # For slicing multi-dimensional tensors

    def __repr__(self):
        if self.dim_index is not None:
            return f"Tensor({self.tag})[{self.dim_index}]"
        return f"Tensor({self.tag})"


class BinaryOp(Placeholder):
    """Binary arithmetic/elementwise op (e.g., +, -, *, /, **).

    Stores the operator string plus left/right operands so evaluation and
    visualization can rebuild the expression tree.
    """

    def __init__(self, op: str, left: Placeholder, right: Placeholder):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class Tracker(Placeholder):
    """Wraps an expression to be monitored during training without contributing to the loss."""

    def __init__(self, expr: Placeholder, interval: int = 1):
        self.expr = expr
        self.interval = interval

    def __repr__(self):
        return f"Tracker({self.expr!r}, interval={self.interval})"


class Model(Placeholder):
    """Wrapper for user-defined Equinox models.

    Allows using any Equinox module within the PINO tracing system.
    The module is initialized lazily when the input dimension is known.

    Example - Direct call style (module takes separate arguments):
        class MLP(eqx.Module):
            ...
            def __call__(self, x, y, *, key=None):
                z = jnp.concat([x, y], axis=-1)
                ...
                return z

        uv_net = pnp.nn.wrap(MLP(..., key=key))
        u = uv_net(x, y)[..., 0]
    """

    def __init__(self, module, name: str = "", weight_path: str = None):
        """Create a Model wrapper.

        Args:
            module: An Equinox module instance (already constructed), or a
                    callable / Flax nn.Module for backward compatibility.
        """
        self.module = module
        self.name = name
        self.input_dim = None
        self.weight_path = weight_path
        self.layer_id = _next_op_id()
        self.show = True  # Wether or not to print the model

        # ── training config (plain Python, not JAX arrays) ──
        self._frozen: bool = False
        self._lora_config = None  # (rank, alpha) or None
        self._opt_fn = None  # optax optimizer factory / instance
        self._lr = None  # LearningRateSchedule or None
        self._dtype = None  # target dtype (e.g. jnp.bfloat16) or None
        self._tunable_opts: Dict[str, list] = {}  # per-model tunable options for sweeps

    # ── public API ───────────────────────────────────────────

    def __call__(self, *args):
        """Call this module with variables - creates a ModelCall."""
        return ModelCall(self, list(args))

    def __repr__(self):
        return f"Model({type(self.module).__name__})"

    def dont_show(self):
        """If called will NOT display the network architecture."""
        self.show = False
        return self

    # ── finetuning helpers ───────────────────────────────────

    def freeze(self):
        """Mark this model as frozen (not trained)."""
        self._frozen = True
        return self

    def unfreeze(self):
        """Unfreeze this model so it is trained normally."""
        self._frozen = False
        return self

    def lora(self, rank: int = 4, alpha: float = 1.0):
        """Enable LoRA fine-tuning for this model.

        Only the low-rank adapters are trained; base weights are frozen.

        Args:
            rank:  LoRA rank.
            alpha: LoRA scaling factor.
        """
        self._lora_config = (rank, alpha)
        return self

    def merge_lora(self):
        """Merge LoRA adapters into base weights and disable LoRA."""
        from .architectures.lora_linear import merge_lora as _merge_lora

        # The actual merge is done by core.solve() after training;
        # calling this just sets a flag.
        self._lora_config = None
        self._merge_lora_flag = True
        return self

    def optimizer(self, opt_fn, *, lr=None):
        """Attach an optimizer (and optional LR schedule) to this model.

        Args:
            opt_fn: An optax optimizer factory, e.g. ``optax.adam``,
                    or an already-constructed transform.
            lr:     A ``LearningRateSchedule`` (or float) for this model.
                    If *None*, a constant schedule of 1e-3 is used.
        """
        from .utils.adaptive import LearningRateSchedule

        self._opt_fn = opt_fn
        if lr is None:
            self._lr = LearningRateSchedule(1e-3)
        elif isinstance(lr, (int, float)):
            self._lr = LearningRateSchedule(float(lr))
        else:
            self._lr = lr
        return self

    def initialize(self, weight_path: str):
        """Load pretrained weights into this model at init time.

        For **Equinox** modules the path should point to a file written by
        ``eqx.tree_serialise_leaves``.

        For **Flax** models wrapped via ``FlaxModelWrapper`` the path should
        point to a Flax ``.msgpack`` weights file.  Weights are merged with
        the current (freshly initialised) parameters – shape-mismatched
        arrays are re-initialised automatically.

        Args:
            weight_path: Path to the pretrained weights file.
        """

        # if isinstance(weight_path, dict):

        # make a temporary file here to saev the params

        # elif # weight path ends in pkl

        # type(weight_path.models[1])
        # <class 'jno.architectures.common.FlaxModelWrapper'>
        # -> crux.models[1].params

        self.weight_path = weight_path
        return self

    def dtype(self, dtype):
        """Cast all floating-point parameters to *dtype* before training.

        Reduces memory usage (e.g. ``jnp.bfloat16`` halves float32 memory)
        while maintaining training stability for many models.

        The cast is applied **after** pretrained weights are loaded (if any)
        and before the training loop starts.  Integer arrays (e.g. indices)
        are left unchanged.

        Args:
            dtype: Target JAX dtype, e.g. ``jnp.bfloat16`` or
                ``jnp.float16``.

        Example::

            u = nn.walrus((1,1,128,128,1), num_out_channels=1)
            u.initialize("walrus.msgpack")
            u.dtype(jnp.bfloat16)
        """
        self._dtype = dtype
        return self

    @property
    def model_key(self) -> str:
        """Stable key identifying this model in a sweep space."""
        return self.name if self.name else f"model_{self.layer_id}"

    def tune(self, *, freeze=None, lora=None, optimizer=None, lr=None, dtype=None):
        """Declare per-model tunable options for hyperparameter sweeps.

        Each argument accepts a list of candidate values.  During a sweep
        the tuner searches over all combinations.

        Args:
            freeze: List of bool, e.g. ``[True, False]``.
            lora: List of ``(rank, alpha)`` tuples **or** ``None`` values,
                e.g. ``[(4, 1.0), (8, 1.0), None]``.
            optimizer: List of optax factories, e.g. ``[optax.adam]``.
            lr: List of :class:`LearningRateSchedule` objects.
            dtype: List of dtypes, e.g. ``[jnp.float32, jnp.bfloat16]``.

        Returns:
            self (for chaining).

        Example::

            backbone = nn.poseidon(...)
            backbone.initialize("weights.msgpack")
            backbone.tune(
                freeze=[True, False],
                lora=[(4, 1.0), None],
                optimizer=[optax.adam],
                lr=[lrs.constant(1e-4), lrs.constant(1e-5)],
            )
        """
        self._tunable_opts = {}
        if freeze is not None:
            self._tunable_opts["freeze"] = list(freeze)
        if lora is not None:
            self._tunable_opts["lora"] = list(lora)
        if optimizer is not None:
            self._tunable_opts["optimizer"] = list(optimizer)
        if lr is not None:
            self._tunable_opts["lr"] = list(lr)
        if dtype is not None:
            self._tunable_opts["dtype"] = list(dtype)
        return self

    def reset(self):
        """Reset all training configuration to defaults.

        .. note:: This does **not** clear ``_tunable_opts`` — those
           persist across trials so that the tuner can re-apply them.
        """
        self._frozen = False
        self._lora_config = None
        self._opt_fn = None
        self._lr = None
        self._dtype = None
        self._merge_lora_flag = False
        return self


class ModelCall(Placeholder):
    """Represents a call to a Model with specific arguments.

    This is created when you call a Model directly with variables:
        uv_net = pnp.nn.wrap(MLP())
        result = uv_net(x, y)  # Creates ModelCall

    All training-configuration methods (``freeze``, ``lora``, ``optimizer``,
    ``dtype``, ``initialize``, ``tune``) are proxied to the underlying
    :class:`Model` so you can chain them after the call::

        u = nn.mlp(2, 64, 1)(x, y).freeze()
    """

    def __init__(self, model: Model, args: list):
        self.model = model
        self.args = args
        self.op_id = _next_op_id()

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.model}({args_str})"

    # ── proxied helpers (delegate to Model) ─────────────

    def dont_show(self):
        """If called will NOT display the network architecture."""
        self.model.show = False
        return self

    def freeze(self):
        self.model.freeze()
        return self

    def unfreeze(self):
        self.model.unfreeze()
        return self

    def lora(self, rank: int = 4, alpha: float = 1.0):
        self.model.lora(rank, alpha)
        return self

    def optimizer(self, opt_fn, *, lr=None):
        self.model.optimizer(opt_fn, lr=lr)
        return self

    def initialize(self, weight_path: str):
        self.model.initialize(weight_path)
        return self

    def dtype(self, dtype):
        self.model.dtype(dtype)
        return self

    def tune(self, **kwargs):
        """Proxy for :meth:`Model.tune`."""
        self.model.tune(**kwargs)
        return self


class TunableModule(Placeholder):
    """
    Wraps a Flax module CLASS + ArchSpace.
    Behaves like Model but with lazy instantiation.
    """

    def __init__(self, module_cls: Type, space: "ArchSpace"):
        self.module_cls = module_cls
        self.space = space
        self.layer_id = _next_op_id()
        self._current_instance: Optional[Model] = None

    def __call__(self, *args):
        """Call with variables - creates ModelCall."""
        # If we have a current instance (during solve), use it
        if self._current_instance is not None:
            return self._current_instance(*args)
        # Otherwise create a placeholder call
        return TunableModuleCall(self, list(args))

    def instantiate(self, arch: "Arch", *, key=None):
        """Create module instance with given architecture.

        For equinox modules, pass ``key`` for random initialization.
        Falls back to ``module_cls(arch=arch)`` if the constructor
        does not accept a ``key`` keyword.
        """
        if key is not None:
            try:
                return self.module_cls(arch=arch, key=key)
            except TypeError:
                return self.module_cls(arch=arch)
        return self.module_cls(arch=arch)


class TunableModuleCall(Placeholder):
    """Call to a TunableModule - resolved at solve time."""

    def __init__(self, model: TunableModule, args: list):
        self.model = model
        self.args = args
        self.op_id = _next_op_id()

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.model}({args_str})"

    def dont_show(self):
        """If called will NOT display the network architecture."""
        self.model._show = False
        return self


class OperationDef(Placeholder):
    """An operation definition - traces a computation graph.

    When called with variables, returns an OperationCall that can be
    evaluated during solve iterations.
    """

    def __init__(self, expr: Placeholder, input_vars: List[Variable] = None):
        self.expr = expr
        self.input_vars = input_vars or []
        self.op_id = _next_op_id()

        # Collect all variables from the expression to determine input signature
        self._collected_vars = self._collect_variables(expr)

        # Check if this operation has trainable layers
        self.has_trainable = self._has_trainable_layers(expr)

    def _collect_variables(self, expr) -> List[Variable]:
        """Collect all Variable and TensorTag placeholders from expression."""
        vars_found = []
        seen_ids = set()

        def visit(node):
            if isinstance(node, (Variable, TensorTag)):
                if id(node) not in seen_ids:
                    seen_ids.add(id(node))
                    vars_found.append(node)
            elif isinstance(node, ModelCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, BinaryOp):
                visit(node.left)
                visit(node.right)
            elif isinstance(node, FunctionCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, OperationCall):
                for arg in node.args:
                    visit(arg)

        visit(expr)
        return vars_found

    def _has_trainable_layers(self, expr) -> bool:
        """Check if expression contains Model nodes."""

        def visit(node):
            if isinstance(node, Model):
                return True
            elif isinstance(node, ModelCall):
                return True  # ModelCall contains a trainable Model
            elif isinstance(node, BinaryOp):
                return visit(node.left) or visit(node.right)
            elif isinstance(node, FunctionCall):
                return any(visit(arg) for arg in node.args if isinstance(arg, Placeholder))
            elif isinstance(node, OperationCall):
                return node.operation.has_trainable
            return False

        return visit(expr)

    def __call__(self, *args) -> "OperationCall":
        """Call operation with specific variables.

        If no arguments passed, uses original variables (no substitution).
        If fewer arguments passed than variables, fills remaining from original variables.
        Example: For Op with vars (x, t, a), calling Op(x0, t0) gives (x0, t0, a).
        """
        n_vars = len(self._collected_vars)
        if len(args) > n_vars:
            var_names = [str(v) for v in self._collected_vars]
            raise ValueError(f"Op[{self.op_id}] has {n_vars} variable(s) {var_names}, " f"but {len(args)} argument(s) were passed: {[str(a) for a in args]}")
        # Fill in missing args from original variables
        if len(args) < n_vars:
            args = args + tuple(self._collected_vars[len(args) :])
        return OperationCall(self, args)

    def __repr__(self):
        return f"Op[{self.op_id}]"


class OperationCall(Placeholder):
    """A call to an operation with specific input variables.

    Example: u(x, y) where u is an OperationDef and x, y are Variables.
    """

    def __init__(self, operation: OperationDef, args: tuple):
        self.operation = operation
        self.args = args  # Variables or other OperationCalls

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.operation}({args_str})"


class Hessian(Placeholder):
    """Second-order differential operator.

    When ``trace=False`` (default), computes the full Hessian matrix
    H[i,j] = d²u / (dxᵢ dxⱼ).

    When ``trace=True``, computes the Laplacian (trace of the Hessian):
    ∇²u = Σᵢ d²u/dxᵢ².
    """

    def __init__(
        self,
        target: OperationCall,
        variables: List[Variable],
        scheme: str = "automatic_differentiation",
        trace: bool = False,
    ):
        self.target = target
        self.variables = variables if isinstance(variables, list) else [variables]
        self.scheme = scheme
        self.trace = trace  # True → Laplacian (sum of diagonal)

    def __repr__(self):
        var_names = ", ".join(str(v) for v in self.variables)
        kind = "∇²" if self.trace else "Hessian"
        return f"{kind}({self.target}, [{var_names}])"


class Jacobian(Placeholder):
    """First-order differential operator.

    Computes J[i] = du/dxᵢ for each variable xᵢ.  When only one
    variable is supplied this is equivalent to a partial derivative
    (gradient), and the result is squeezed to a scalar per point.
    """

    def __init__(
        self,
        target: OperationCall,
        variables: List[Variable],
        scheme: str = "automatic_differentiation",
    ):
        self.target = target
        self.variables = variables if isinstance(variables, list) else [variables]
        self.scheme = scheme

    def __repr__(self):
        var_names = ", ".join(str(v) for v in self.variables)
        return f"Jacobian({self.target}, [{var_names}])"


# =============================================================================
# Tree optimisation — Common Sub-expression Elimination (CSE)
# =============================================================================


def cse(expr: Placeholder) -> Placeholder:
    """Eliminate common sub-expressions in a traced computation tree.

    Walks *expr* bottom-up and replaces structurally identical sub-trees
    with a single shared Python object.  Two nodes are considered
    identical when they have the same type, the same static attributes
    (operator, function identity, op_id, …) **and** the same children
    (by ``id``).

    The pass is safe to run multiple times and never changes semantics.

    What gets deduplicated:

    * ``OperationCall`` — same ``OperationDef`` + same argument
      ``Variable``/``TensorTag`` objects (by identity).
    * ``BinaryOp`` — same operator + same (already-deduped) children.
    * ``FunctionCall`` — same ``fn`` + same (already-deduped) args.
    * ``Jacobian`` / ``Hessian`` — same target + same variables.
    * ``ModelCall`` — same model + same args.

    Returns:
        A (possibly shared) tree with duplicates collapsed.
    """
    # Maps a structural key → canonical node that was already built.
    _canon: dict = {}

    def _key(node):
        """Return a hashable key for *node* assuming children are already canonical."""
        if isinstance(node, Variable):
            return ("Var", id(node))
        if isinstance(node, TensorTag):
            return ("Tag", id(node))
        if isinstance(node, (Constant, Literal)):
            return ("Const", id(node))
        if isinstance(node, BinaryOp):
            return ("Bin", node.op, id(node.left), id(node.right))
        if isinstance(node, FunctionCall):
            arg_ids = tuple(id(a) for a in node.args)
            return ("Fn", id(node.fn), node._name, arg_ids)
        if isinstance(node, ModelCall):
            arg_ids = tuple(id(a) for a in node.args)
            return ("Call", node.model.layer_id, arg_ids)
        if isinstance(node, OperationCall):
            arg_ids = tuple(id(a) for a in node.args)
            return ("OpCall", node.operation.op_id, arg_ids)
        if isinstance(node, OperationDef):
            return ("OpDef", node.op_id, id(node.expr))
        if isinstance(node, Jacobian):
            var_ids = tuple(id(v) for v in node.variables)
            return ("Jac", id(node.target), var_ids, node.scheme)
        if isinstance(node, Hessian):
            var_ids = tuple(id(v) for v in node.variables)
            return ("Hess", id(node.target), var_ids, node.trace, node.scheme)
        if isinstance(node, Tracker):
            return ("Track", id(node.expr), node.interval)
        # Fallback — use object identity (no dedup possible)
        return ("?", id(node))

    def _visit(node):
        """Post-order walk: canonicalise children first, then self."""
        # Leaves — always canonical
        if isinstance(node, (Variable, TensorTag, Constant, Literal)):
            return node

        # ── recurse into children and rebuild if anything changed ──
        if isinstance(node, BinaryOp):
            l = _visit(node.left)
            r = _visit(node.right)
            if l is not node.left or r is not node.right:
                node = BinaryOp(node.op, l, r)
        elif isinstance(node, FunctionCall):
            new_args = [_visit(a) if isinstance(a, Placeholder) else a for a in node.args]
            if any(n is not o for n, o in zip(new_args, node.args)):
                node = FunctionCall(node.fn, new_args, node._name, node.reduces_axis, node.kwargs)
        elif isinstance(node, ModelCall):
            new_args = [_visit(a) if isinstance(a, Placeholder) else a for a in node.args]
            if any(n is not o for n, o in zip(new_args, node.args)):
                new_node = ModelCall(node.model, new_args)
                new_node.op_id = node.op_id
                node = new_node
        elif isinstance(node, OperationDef):
            new_expr = _visit(node.expr)
            if new_expr is not node.expr:
                new_node = OperationDef.__new__(OperationDef)
                new_node.expr = new_expr
                new_node.input_vars = node.input_vars
                new_node.op_id = node.op_id
                new_node._collected_vars = node._collected_vars
                new_node.has_trainable = node.has_trainable
                node = new_node
        elif isinstance(node, OperationCall):
            new_op = _visit(node.operation)
            new_args = tuple(_visit(a) if isinstance(a, Placeholder) else a for a in node.args)
            if new_op is not node.operation or any(n is not o for n, o in zip(new_args, node.args)):
                node = OperationCall(new_op, new_args)
        elif isinstance(node, Jacobian):
            new_target = _visit(node.target)
            if new_target is not node.target:
                node = Jacobian(new_target, node.variables, node.scheme)
        elif isinstance(node, Hessian):
            new_target = _visit(node.target)
            if new_target is not node.target:
                node = Hessian(new_target, node.variables, node.scheme, node.trace)
        elif isinstance(node, Tracker):
            new_expr = _visit(node.expr)
            if new_expr is not node.expr:
                node = Tracker(new_expr, node.interval)

        # ── dedup: if we've seen an identical node, return the earlier one ──
        k = _key(node)
        if k in _canon:
            return _canon[k]
        _canon[k] = node
        return node

    return _visit(expr)


# =============================================================================
# Evaluation engine
# =============================================================================


def collect_operations(expr: Placeholder) -> List[OperationDef]:
    """Collect all OperationDef instances from an expression."""
    ops = []
    seen = set()

    def visit(node):
        if isinstance(node, OperationDef):
            if node.op_id not in seen:
                seen.add(node.op_id)
                ops.append(node)
            visit(node.expr)
        elif isinstance(node, OperationCall):
            if node.operation.op_id not in seen:
                seen.add(node.operation.op_id)
                ops.append(node.operation)
            visit(node.operation.expr)
            for arg in node.args:
                visit(arg)
        elif isinstance(node, ModelCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, TunableModuleCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, BinaryOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, (Hessian, Jacobian)):
            visit(node.target)
            for v in node.variables:
                visit(v)
        elif isinstance(node, Tracker):
            visit(node.expr)

    visit(expr)
    return ops


def collect_tags(expr: Placeholder) -> set:
    """Collect all unique tags from Variables in the expression tree."""
    tags = set()

    def visit(node):
        if isinstance(node, Variable):
            tags.add(node.tag)
        elif isinstance(node, TensorTag):
            tags.add(node.tag)
        elif isinstance(node, BinaryOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, OperationDef):
            visit(node.expr)
        elif isinstance(node, OperationCall):
            visit(node.operation.expr)
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, ModelCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, TunableModuleCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, (Hessian, Jacobian)):
            visit(node.target)
            for v in node.variables:
                visit(v)
        elif isinstance(node, Tracker):
            visit(node.expr)

    visit(expr)
    return tags


def get_primary_tag(expr: Placeholder) -> str:
    """Return the first Variable tag found in the expression tree."""
    tags = collect_tags(expr)
    return next(iter(tags)) if tags else None


def dump_tree(expr, indent: int = 0, seen: set = None) -> str:
    """Return a human-readable indented string of the expression tree.

    Args:
        expr:   Any trace node (Placeholder subclass).
        indent: Current indentation level (used by recursion).
        seen:   Set of already-visited ``OperationDef.op_id`` values to
                avoid infinite recursion on shared sub-graphs.

    Returns:
        Multi-line string with the full computation tree.

    Example::

        tree_str = dump_tree(pde)
        print(tree_str)
        # or
        with open("tree.txt", "w") as f:
            f.write(tree_str)
    """
    if seen is None:
        seen = set()
    pad = "  " * indent
    lines: list[str] = []

    def _node_label(node) -> str:
        """One-line label for a node (no children)."""
        if isinstance(node, Variable):
            return f"Variable({node.tag}[{node.dim}])"
        if isinstance(node, TensorTag):
            return f"TensorTag({node.tag})"
        if isinstance(node, Constant):
            val = node.value
            if hasattr(val, "shape") and val.shape == ():
                val = float(val)
            return f"Constant({node.tag}.{node.key}={val})"
        if isinstance(node, Literal):
            return f"Literal({node.value})"
        if isinstance(node, Model):
            return f"Model(id={node.layer_id}, {type(node.module).__name__})"
        if isinstance(node, (int, float)):
            return str(node)
        return type(node).__name__

    def _visit(node, depth):
        p = "  " * depth
        if isinstance(node, Variable):
            lines.append(f"{p}{_node_label(node)}")
        elif isinstance(node, TensorTag):
            lines.append(f"{p}{_node_label(node)}")
        elif isinstance(node, (Constant, Literal)):
            lines.append(f"{p}{_node_label(node)}")
        elif isinstance(node, BinaryOp):
            lines.append(f"{p}BinaryOp({node.op})")
            _visit(node.left, depth + 1)
            _visit(node.right, depth + 1)
        elif isinstance(node, FunctionCall):
            name = node._name or getattr(node.fn, "__name__", "fn")
            lines.append(f"{p}FunctionCall({name})")
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    _visit(arg, depth + 1)
                else:
                    lines.append(f"{p}  {arg}")
        elif isinstance(node, ModelCall):
            lines.append(f"{p}ModelCall({_node_label(node.model)})")
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    _visit(arg, depth + 1)
                else:
                    lines.append(f"{p}  {arg}")
        elif isinstance(node, TunableModuleCall):
            lines.append(f"{p}TunableModuleCall(id={node.model.layer_id})")
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    _visit(arg, depth + 1)
        elif isinstance(node, OperationDef):
            if node.op_id in seen:
                vars_str = ", ".join(str(v) for v in node._collected_vars)
                lines.append(f"{p}Op[{node.op_id}]({vars_str})  [already shown]")
                return
            seen.add(node.op_id)
            vars_str = ", ".join(str(v) for v in node._collected_vars)
            lines.append(f"{p}OperationDef[{node.op_id}] vars=({vars_str})")
            _visit(node.expr, depth + 1)
        elif isinstance(node, OperationCall):
            args_str = ", ".join(str(a) for a in node.args)
            lines.append(f"{p}OperationCall[{node.operation.op_id}]({args_str})")
            _visit(node.operation, depth + 1)
        elif isinstance(node, Hessian):
            kind = "Laplacian" if node.trace else "Hessian"
            vars_str = ", ".join(str(v) for v in node.variables)
            lines.append(f"{p}{kind}([{vars_str}])")
            _visit(node.target, depth + 1)
        elif isinstance(node, Jacobian):
            vars_str = ", ".join(str(v) for v in node.variables)
            lines.append(f"{p}Jacobian([{vars_str}])")
            _visit(node.target, depth + 1)
        elif isinstance(node, Tracker):
            lines.append(f"{p}Tracker(interval={node.interval})")
            _visit(node.expr, depth + 1)
        elif isinstance(node, ConstantNamespace):
            lines.append(f"{p}ConstantNamespace({node._full_tag})")
        elif isinstance(node, Placeholder):
            # Fallback for any unknown Placeholder subclass
            lines.append(f"{p}{repr(node)}")
        else:
            lines.append(f"{p}{node}")

    _visit(expr, indent)
    return "\n".join(lines)
