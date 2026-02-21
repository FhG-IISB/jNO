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
from flax import linen as nn
from pathlib import Path
import json

# Global counter for unique operation IDs
_operation_counter = 0


def _next_op_id() -> int:
    global _operation_counter
    _operation_counter += 1
    return _operation_counter


class Debug:

    def __init__(self, name):
        """
        This is the debugger
        """
        self.name: str = name
        self._val: bool = False
        self._shape: bool = False
        self._mean: bool = False
        self._max: bool = False
        self._min: bool = False
        self._flops: bool = True

    def count(self):
        cou = 0
        if self._val:
            cou += 1
        if self._shape:
            cou += 1
        if self._mean:
            cou += 1
        if self._max:
            cou += 1
        if self._min:
            cou += 1
        return cou

    def __call__(self, expr):
        num = self.count()

        if num == 0:
            return expr

        # If only one option is enabled, print with name
        if num == 1:
            if self._val:
                jax.debug.print("{name} -> {expr}", name=self.name, expr=expr)
            if self._shape:
                jax.debug.print(
                    "{name}.shape -> {shape}", name=self.name, shape=expr.shape
                )
            if self._mean:
                jax.debug.print(
                    "{name}.mean -> {mean}", name=self.name, mean=jnp.mean(expr)
                )
            if self._max:
                jax.debug.print(
                    "{name}.max -> {max}", name=self.name, max=jnp.max(expr)
                )
            if self._min:
                jax.debug.print(
                    "{name}.min -> {min}", name=self.name, min=jnp.min(expr)
                )
        else:
            # Multiple options enabled - print name once, then values without name
            jax.debug.print("{name} ->", name=self.name)
            if self._val:
                jax.debug.print("  value: {expr}", expr=expr)
            if self._shape:
                jax.debug.print("  shape: {shape}", shape=expr.shape)
            if self._mean:
                jax.debug.print("  mean: {mean}", mean=jnp.mean(expr))
            if self._max:
                jax.debug.print("  max: {max}", max=jnp.max(expr))
            if self._min:
                jax.debug.print("  min: {min}", min=jnp.min(expr))

        return expr

    # Builder methods for fluent API
    def print(self):
        self._print = True
        return self

    def shape(self):
        self._shape = True
        return self

    def mean(self):
        self._mean = True
        return self

    def max(self):
        self._max = True
        return self

    def min(self):
        self._min = True
        return self


class NewAxis:
    def __repr__(self):
        return "None"


class Placeholder:
    """Base node for the traced DSL graph.

    Placeholders behave like symbolic tensors: arithmetic and composition
    operators create new traced nodes instead of executing eagerly. Calling a
    placeholder (`u(x)`) auto-wraps it in an `OperationDef` so it can be reused
    with different inputs. Concrete values are only produced when evaluated by
    the solver/visualizer.
    """

    def __gt__(self, other):
        return FunctionCall(jnp.greater, [self, other])

    def __lt__(self, other):
        return FunctionCall(jnp.less, [self, other])

    def __ge__(self, other):
        return FunctionCall(jnp.greater_equal, [self, other])

    def __le__(self, other):
        return FunctionCall(jnp.less_equal, [self, other])

    def __eq__(self, other):
        return FunctionCall(jnp.equal, [self, other])

    def __ne__(self, other):
        return FunctionCall(jnp.not_equal, [self, other])

    def __rgt__(self, other):
        return FunctionCall(jnp.less, [other, self])

    def __rlt__(self, other):
        return FunctionCall(jnp.greater, [other, self])

    def __rge__(self, other):
        return FunctionCall(jnp.less_equal, [other, self])

    def __rle__(self, other):
        return FunctionCall(jnp.greater_equal, [other, self])

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

        normalized = tuple(NewAxis() if k is None else k for k in key)

        return Slice(self, normalized)

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
        """Reshape this placeholder to a new shape.

        Args:
            *shape: New shape as separate arguments or a single tuple

        Returns:
            Reshape node representing the reshaped tensor
        """
        # Handle both reshape(2, 3) and reshape((2, 3)) syntax
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape(self, shape)

    @property
    def shape(self):
        return FunctionCall(
            lambda x: jnp.ones(x.shape, dtype="bool"), [self], "shape", True
        )

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
        return FunctionCall(
            lambda x: jnp.squeeze(jnp.mean(jnp.square(x))), [self], "mse", True
        )

    @property
    def mae(self):
        return FunctionCall(
            lambda x: jnp.squeeze(jnp.mean(jnp.abs(x))), [self], "mae", True
        )

    @property
    def T(self):
        return FunctionCall(lambda x: x.T, [self], "transpose", True)


class Reshape(Placeholder):
    """Reshape a traced tensor to a new shape."""

    def __init__(self, target: Placeholder, shape: tuple):
        self.target = target
        self.shape = shape
        self.tag = target.tag if hasattr(target, "tag") else None

    def __repr__(self):
        return f"{self.target}.reshape({self.shape})"


class Slice(Placeholder):
    def __init__(self, target: Placeholder, key):
        self.target = target
        self.key = key
        self.tag = target.tag if hasattr(target, "tag") else None

    def __repr__(self):
        return f"{self.target}[{', '.join(map(str, self.key))}]"


class Concat(Placeholder):
    """Concatenate multiple traced tensors along a given axis."""

    def __init__(self, items: List, axis: int = -1):
        self.items = items
        self.axis = axis

    def __repr__(self):
        items_str = ", ".join(str(i) for i in self.items)
        return f"Concat([{items_str}])"


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
        return FunctionCall(
            fn=self.fn, args=new_args, name=self._name, reduces_axis=self.reduces_axis
        )

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
                    (
                        ConstantNamespace(f"{key}[{i}]", item, _parent_tag=parent_tag)
                        if isinstance(item, dict)
                        else ConstantNamespace._convert_value(
                            item, f"{key}[{i}]", parent_tag
                        )
                    )
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
            raise ValueError(
                f"Unsupported file format: '{suffix}'. "
                f"Supported formats: .json, .yaml, .yml, .toml, .pkl, .pickle, .npz, .npy"
            )

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
            raise ImportError(
                "PyYAML is required to load .yaml/.yml files. "
                "Install with: pip install pyyaml"
            )

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
                raise ImportError(
                    "toml package is required to load .toml files. "
                    "Install with: pip install toml"
                )

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        """Load pickle file."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError(
                f"Pickle file must contain a dict, got {type(data).__name__}"
            )
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
            raise AttributeError(
                f"Constant '{self._full_tag}' has no key '{key}'. "
                f"Available keys: {available}"
            )

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
                result[key] = [
                    item.to_dict() if isinstance(item, ConstantNamespace) else item
                    for item in value
                ]
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
    """

    def __init__(self, tag: str, dim: list, domain=None):
        self.tag = tag
        self.dim = dim
        if tag in domain.sampled_points.keys():
            self.size = (
                dim[1] - dim[0]
                if dim[1] is not None
                else domain.sampled_points[tag].shape[-1]
            )
        else:
            self.size = (
                dim[1] - dim[0]
                if dim[1] is not None
                else domain.tensor_tags[tag].shape[-1]
            )
        self._domain = domain  # Reference to parent domain for inference

    def __repr__(self):
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
        self.debug = Debug(f"({self.left} {self.op} {self.right})")

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class Tracker(Placeholder):
    def __init__(self, op: BinaryOp, interval: int):
        self.op = op
        self.interval = interval

    def __repr__(self):
        return f"tracked_BinaryOp[{self.interval}]"


class FlaxModule(Placeholder):
    """Wrapper for user-defined Flax nn.Module models.

    Allows using any Flax model within the PINO tracing system.
    The module is initialized lazily when the input dimension is known.

    Example 2 - Direct call style (module takes separate arguments):
        class MLP(nn.Module):
            @nn.compact
            def __call__(self, x, y):
                z = jnp.concat([x, y], axis=-1)
                z = nn.Dense(64)(z)
                z = nn.tanh(z)
                z = nn.Dense(2)(z)
                return z

        uv_net = pnp.nn.wrap(MLP())
        u = uv_net(x, y)[..., 0]
    """

    def __init__(self, module: nn.Module, name: str = "", weight_path: str = None):
        """Create a FlaxModule wrapper.

        Args:
            module: A Flax nn.Module instance (already constructed)
            input_dim: Optional input dimension. If None, inferred from arguments.
        """
        self.module = module
        self.name = name
        self.input_dim = None
        self.weight_path = weight_path
        self.layer_id = _next_op_id()
        self.show = True  # Wether or not to print the model

    def __call__(self, *args):
        """Call this module with variables - creates a FlaxModuleCall."""
        return FlaxModuleCall(self, list(args))

    def __repr__(self):
        return f"FlaxModule({type(self.module).__name__})"

    def dont_show(self):
        """If called will NOT display the network architecture."""
        self.show = False
        return self


class TunableModule(Placeholder):
    """
    Wraps a Flax module CLASS + ArchSpace.
    Behaves like FlaxModule but with lazy instantiation.
    """

    def __init__(self, module_cls: Type, space: "ArchSpace"):
        self.module_cls = module_cls
        self.space = space
        self.layer_id = _next_op_id()
        self._current_instance: Optional[FlaxModule] = None

    def __call__(self, *args):
        """Call with variables - creates FlaxModuleCall."""
        # If we have a current instance (during solve), use it
        if self._current_instance is not None:
            return self._current_instance(*args)
        # Otherwise create a placeholder call
        return TunableModuleCall(self, list(args))

    def instantiate(self, arch: "Arch"):
        """Create module instance with given architecture."""
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


class FlaxModuleCall(Placeholder):
    """Represents a call to a FlaxModule with specific arguments.

    This is created when you call a FlaxModule directly with variables:
        uv_net = pnp.nn.wrap(MLP())
        result = uv_net(x, y)  # Creates FlaxModuleCall

    Call .dont_show to not display the model
    """

    def __init__(self, model: FlaxModule, args: list):
        self.model = model
        self.args = args
        self.op_id = _next_op_id()

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.model}({args_str})"

    def dont_show(self):
        """If called will NOT display the network architecture."""
        self.model.show = False
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
            elif isinstance(node, FlaxModuleCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, BinaryOp):
                visit(node.left)
                visit(node.right)
            elif isinstance(node, Concat):
                for item in node.items:
                    visit(item)
            elif isinstance(node, FunctionCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, OperationCall):
                for arg in node.args:
                    visit(arg)
            elif isinstance(node, Slice):
                visit(node.target)

        visit(expr)
        return vars_found

    def _has_trainable_layers(self, expr) -> bool:
        """Check if expression contains FlaxModule nodes."""

        def visit(node):
            if isinstance(node, FlaxModule):
                return True
            elif isinstance(node, FlaxModuleCall):
                return True  # FlaxModuleCall contains a trainable FlaxModule
            elif isinstance(node, BinaryOp):
                return visit(node.left) or visit(node.right)
            elif isinstance(node, Concat):
                return any(visit(item) for item in node.items)
            elif isinstance(node, FunctionCall):
                return any(
                    visit(arg) for arg in node.args if isinstance(arg, Placeholder)
                )
            elif isinstance(node, OperationCall):
                return node.operation.has_trainable
            elif isinstance(node, Slice):
                return visit(node.target)
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
            raise ValueError(
                f"Op[{self.op_id}] has {n_vars} variable(s) {var_names}, "
                f"but {len(args)} argument(s) were passed: {[str(a) for a in args]}"
            )
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


class Laplacian(Placeholder):
    """Laplacian operator on an operation call."""

    def __init__(
        self,
        target: OperationCall,
        variables: List[Variable],
        scheme: str = "automatic_differentiation",
    ):
        self.target = target
        self.variables = variables
        self.scheme = scheme  # 'automatic_differentiation' or 'finite_difference'

    def __repr__(self):
        vars_str = ", ".join(str(v) for v in self.variables)
        return f"∇²({self.target}, [{vars_str}])"


class Gradient(Placeholder):
    """Gradient operator on an operation call."""

    def __init__(
        self,
        target: OperationCall,
        variable: Variable,
        scheme: str = "automatic_differentiation",
    ):
        self.target = target
        self.variable = variable
        self.scheme = scheme  # 'automatic_differentiation' or 'finite_difference'

    def __repr__(self):
        return f"∇({self.target}, {self.variable})"


class Hessian(Placeholder):
    """Hessian matrix operator (matrix of second derivatives)."""

    def __init__(
        self,
        target: OperationCall,
        variables: List[Variable],
        scheme: str = "automatic_differentiation",
    ):
        self.target = target
        self.variables = variables
        self.scheme = scheme  # 'automatic_differentiation' or 'finite_difference'

    def __repr__(self):
        var_names = ", ".join(str(v) for v in self.variables)
        return f"Hessian({self.target}, [{var_names}])"


class Jacobian(Placeholder):
    """Jacobian matrix operator (matrix of second derivatives)."""

    def __init__(
        self,
        target: OperationCall,
        variables: List[Variable],
        scheme: str = "automatic_differentiation",
    ):
        self.target = target
        self.variables = variables
        self.scheme = scheme  # 'automatic_differentiation' or 'finite_difference'

    def __repr__(self):
        var_names = ", ".join(str(v) for v in self.variables)
        return f"Jacobian({self.target}, [{var_names}])"


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
        elif isinstance(node, FlaxModuleCall):
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
        elif isinstance(node, Concat):
            for item in node.items:
                visit(item)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, Laplacian):
            visit(node.target)
        elif isinstance(node, Gradient):
            visit(node.target)
        elif isinstance(node, Hessian):
            visit(node.target)
        elif isinstance(node, Slice):
            visit(node.target)

    visit(expr)
    return ops


def collect_tags(expr: Placeholder) -> set:
    """Collect all tags from Variable placeholders."""
    tags = set()

    def visit(node):
        if isinstance(node, Variable):
            tags.add(node.tag)
        elif isinstance(node, OperationDef):
            visit(node.expr)
        elif isinstance(node, FlaxModuleCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, BinaryOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, Concat):
            for item in node.items:
                visit(item)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, OperationCall):
            for arg in node.args:
                visit(arg)
            visit(node.operation.expr)
        elif isinstance(node, Laplacian):
            visit(node.target)
            for v in node.variables:
                visit(v)
        elif isinstance(node, Gradient):
            visit(node.target)
            visit(node.variable)
        elif isinstance(node, Hessian):
            visit(node.target)
            for v in node.variables:
                visit(v)
        elif isinstance(node, Slice):
            visit(node.target)

    visit(expr)
    return tags


def get_primary_tag(expr: Placeholder) -> Optional[str]:
    """Get the primary tag for a constraint."""
    tags = collect_tags(expr)
    return next(iter(tags)) if tags else None
