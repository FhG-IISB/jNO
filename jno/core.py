import gc
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import paramax as _paramax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import bayesian as jno_bayesian
from .architectures.lora import (
    LoRAWrapper as _LoRAWrapper,
)
from .architectures.lora import (
    apply_lora as _apply_lora,
)
from .architectures.lora import (
    lora_trainable_filter as _lora_trainable_filter,
)
from .architectures.lora import (
    merge_lora as _merge_lora,
)
from .architectures.lora import (
    partial_lora_trainable_filter as _partial_lora_trainable_filter,
)
from .domain import DomainData, domain
from .trace import (
    BinaryOp,
    Choice,
    FunctionCall,
    Hessian,
    IntegralTime,
    Jacobian,
    Model,
    ModelCall,
    OperationCall,
    OperationDef,
    Placeholder,
    Tracker,
    TunableModule,
    TunableModuleCall,
    collect_operations,
    collect_tags,
    cse,
    dump_tree,
    get_primary_tag,
)
from .trace_compiler import TraceCompiler
from .trace_evaluator import TraceEvaluator
from .tuner import ArchSpace, DeviceConfig, Tuner
from .utils import (
    LearningRateSchedule,
    get_logger,
    get_seed,
    statistics,
)
from .utils.config import get_wandb_run, wandb_alert, wandb_log, wandb_log_model


def _find_temporal_variable(expr: Placeholder):
    """Walk an expression tree and return the first temporal Variable, or None."""
    from .trace import Variable as _Variable

    seen: set = set()

    def visit(node):
        if node is None or id(node) in seen:
            return None
        seen.add(id(node))
        if isinstance(node, _Variable) and getattr(node, "axis", "spatial") == "temporal":
            return node
        for attr in ("target", "left", "right", "expr", "operation"):
            child = getattr(node, attr, None)
            if isinstance(child, Placeholder):
                hit = visit(child)
                if hit is not None:
                    return hit
        for attr in ("args", "variables", "options"):
            for child in getattr(node, attr, []):
                if isinstance(child, Placeholder):
                    hit = visit(child)
                    if hit is not None:
                        return hit
        return None

    return visit(expr)


def _active_model_lids(exprs):
    """Return layer_ids of Model nodes reachable without crossing stop_gradient.

    Used by substep training to determine which models have non-zero gradient
    potential for a given subset of constraints.
    """
    active: set = set()
    seen: set = set()

    def _walk(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        # stop_gradient boundary — everything inside is frozen, skip
        if isinstance(node, FunctionCall) and node.fn is jax.lax.stop_gradient:
            return
        if isinstance(node, ModelCall):
            active.add(node.model.layer_id)
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    _walk(arg)
            return
        if isinstance(node, Model):
            active.add(node.layer_id)
            return
        if isinstance(node, BinaryOp):
            _walk(node.left)
            _walk(node.right)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    _walk(arg)
        elif isinstance(node, OperationCall):
            _walk(node.operation.expr)
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    _walk(arg)
        else:
            for attr in ("target", "expr"):
                child = getattr(node, attr, None)
                if isinstance(child, Placeholder):
                    _walk(child)
            for seq_attr in ("args", "variables"):
                for child in getattr(node, seq_attr, []):
                    if isinstance(child, Placeholder):
                        _walk(child)

    for expr in exprs:
        _walk(expr)
    return active


def _parse_substep_spec(spec):
    """Normalise one substep entry → (indices: list[int], n_steps: int).

    Accepts:
      [0, 1]           → ([0, 1], 1)   plain index list, 1 step
      ([0, 1], 2)      → ([0, 1], 2)   index list + step count
    """
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and isinstance(spec[1], int) and not isinstance(spec[0], int):
            return list(spec[0]), int(spec[1])
        return [int(i) for i in spec], 1
    raise TypeError(f"substep must be a list or ([list], int) tuple, got {type(spec)}")


class core:
    """core solver using traced operations."""

    def __init__(
        self,
        constraints: List[Placeholder],
        domain: domain,
        mesh: Optional[Tuple[int, ...]] = (1, 1),
        resume_from: Optional[str] = None,
    ):
        """
        Initialize core solver.

        Args:
            constraints: List of constraint expressions defining the problem to solve.
                Each constraint represents an equation or condition that should be
                minimized during training (e.g., PDE residuals, boundary conditions,
                data fitting terms).

            domain: Domain object containing the computational domain and sampled points.
                Defines the spatial/temporal coordinates where constraints are evaluated,
                along with any tensor data (e.g., input functions for operator learning).

            rng_seed: Random seed for reproducibility. Controls parameter initialization
                and any stochastic operations during training.
                If ``None`` (default), reads ``[jno] seed`` from ``.jno.toml`` (or
                ``~/.jno/config.toml``); falls back to ``21`` when no config is found.

                To pin the seed for your whole project, add to ``.jno.toml``::

                    [jno]
                    seed = 42

            mesh: Shape of the device mesh for hybrid parallelism as a tuple (batch, model).
                Controls how computation is distributed across multiple GPUs/TPUs.

                - First dimension (batch): Number of devices for data parallelism.
                Data is split across these devices, each processes different samples.
                Parameters are replicated on all devices.

                - Second dimension (model): Number of devices for model parallelism.
                Model parameters are sharded across these devices.
                Use when model is too large to fit on a single device.

                Examples:
                    - (1, 1): No parallelism, single device (default)
                    - (2, 1): Pure data parallelism on 2 GPUs - 2x throughput
                    - (1, 2): Pure model parallelism on 2 GPUs - fit 2x larger models
                    - (4, 1): Data parallelism on 4 GPUs - 4x throughput
                    - (2, 2): Hybrid parallelism on 4 GPUs - 2x data, 2x model
                    - (4, 2): Hybrid parallelism on 8 GPUs - 4x data, 2x model

                Note: batch * model must equal the total number of available devices.

                Recommendations:
                    - Model fits on 1 GPU: Use (n_devices, 1) for maximum throughput
                    - Model doesn't fit on 1 GPU: Use (1, n_devices) for model sharding
                    - Large model + large data: Use hybrid, e.g., (2, 2) on 4 GPUs

                Default: (1, 1), automatically expanded to (n_devices, 1) for pure
                data parallelism when multiple devices are available.

            resume_from: Path to a checkpoint directory written by
                :class:`~jno.utils.callbacks.CheckpointCallback`.  When
                provided, model parameters, optimizer states, and the RNG
                key are restored from the latest checkpoint at the start
                of the next ``solve()`` call.  Requires the optional
                ``orbax-checkpoint`` package.
        """
        self.log = get_logger()
        self.constraints: List[Placeholder] = constraints

        self.domain = domain
        self.models: Dict[int, Any] = {}
        self._trained_ops: Dict[int, Any] = {}
        self.training_logs: List[Dict[str, jnp.ndarray]] = []
        self.dots: List = []
        self.all_ops: List[OperationDef] = []
        self._resume_from: Optional[str] = resume_from

        super().__init__()

        self._total_epochs = 0
        seed_cfg = get_seed()
        seed = int(seed_cfg) if seed_cfg is not None else 21
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.log.info(f"RNG seed: {seed}")

        self.log.info("Initializing Model/s and compiling constraints")

        # If any constraint references a PDEformer-2 backbone, build its PDE
        # graph automatically from the symbolic expressions before compile().
        try:
            from .architectures.pdeformer2_bridge import maybe_attach_pdeformer2_graphs

            maybe_attach_pdeformer2_graphs(self.constraints, self.domain)
        except ImportError:
            pass  # jax_pdeformer2 / networkx not installed → silently skip

        # Early validation: temporal Variables require a time-dependent domain.
        # Without this, compilation succeeds and the failure surfaces later as
        # a cryptic KeyError on '__time__' deep in trace_evaluator.
        if not getattr(self.domain, "_is_time_dependent", False):
            for expr in self.constraints:
                tvar = _find_temporal_variable(expr)
                if tvar is not None:
                    raise ValueError(
                        f"Constraint uses a temporal Variable (tag='{tvar.tag}', axis='temporal') "
                        f"but the domain is not time-dependent. Pass time=(t0, t1, n_t) when "
                        f"constructing the domain, e.g. jno.domain.line(time=(0.0, 1.0, 50))."
                    )

        self.compile(mesh)

        self.log.info(f"Using {len(self.devices)} device(s): {self.devices}")

        return None

    def _setup_parallelism(self, mesh_shape: Optional[Tuple[int, ...]]):
        """Setup device mesh and sharding specifications."""
        self.devices = jax.devices()
        n_devices = len(self.devices)

        # Default mesh: all devices for data parallelism, no model parallelism
        if mesh_shape is None or mesh_shape == (1, 1):
            mesh_shape = (n_devices, 1)  # Pure data parallelism by default

        if mesh_shape[0] * mesh_shape[1] != n_devices:
            self.log.warning(f"mesh_shape {mesh_shape} doesn't match {n_devices} devices -> default back to (n_devices, 1)")
            mesh_shape = (n_devices, 1)

        self.mesh = Mesh(
            mesh_utils.create_device_mesh(mesh_shape, devices=self.devices),
            axis_names=("batch", "model"),
        )

        # Params sharded along model axis (replicated if model dim is 1)
        self.param_sharding = NamedSharding(self.mesh, P(None, "model"))
        # Data sharded along batch axis
        self.data_sharding = NamedSharding(self.mesh, P("batch", None))

        self.log.info(f"Device mesh: {self.mesh} (shape: {mesh_shape})")
        return None

    def _shard_params(self, params: Dict) -> Dict:
        """Apply sharding to model parameters."""

        model_dim = self.mesh.shape["model"]

        if model_dim > 1:
            self.log.info("Parameters sharded across devices")

        # Use P() (fully replicated) for all non-sharded arrays.
        # Important: P() is canonical — JAX's optimizer outputs use P(),
        # so using P(None,) or P(None, None) here would cause a sharding
        # mismatch on the next step and trigger a recompilation.
        replicated = P()

        def shard_leaf(x):
            # Handle JAX arrays
            if isinstance(x, (jnp.ndarray, jax.Array)):
                if model_dim == 1:
                    spec = replicated
                else:
                    if x.ndim <= 1:
                        spec = replicated
                    else:
                        spec = P(*([None] * (x.ndim - 1)), "model")
                return jax.device_put(x, NamedSharding(self.mesh, spec))
            # Handle numpy arrays (convert first)
            elif isinstance(x, np.ndarray):
                x = jnp.array(x)
                if model_dim == 1:
                    spec = replicated
                else:
                    if x.ndim <= 1:
                        spec = replicated
                    else:
                        spec = P(*([None] * (x.ndim - 1)), "model")
                return jax.device_put(x, NamedSharding(self.mesh, spec))
            return x

        return jax.tree_util.tree_map(shard_leaf, params)

    def _shard_data(self, data: Dict) -> Dict:
        """Apply sharding to training data.

        Spatial arrays ``(B, T, N, D)`` are sharded along the batch axis.
        The shared ``__time__`` array ``(T, 1)`` is fully replicated.
        """

        def shard_leaf(key, x):
            if isinstance(x, jnp.ndarray):
                if x.ndim == 0:
                    return x
                # __time__ is shared across batches — replicate
                if key == "__time__":
                    spec = P(*([None] * x.ndim))
                elif x.ndim == 1:
                    spec = P("batch")
                else:
                    spec = P("batch", *([None] * (x.ndim - 1)))
                return jax.device_put(x, NamedSharding(self.mesh, spec))
            return x

        return {k: shard_leaf(k, v) for k, v in data.items()}

    def _replicate_for_devices(self, data: Dict, n_devices: int) -> Dict:
        """Tile data to have leading dimension matching device count for data parallelism."""

        def tile_if_needed(x):
            if isinstance(x, jnp.ndarray) and x.ndim >= 1:
                # Check if we need to tile along batch dimension
                if x.shape[0] < n_devices:
                    reps = (n_devices // x.shape[0],) + (1,) * (x.ndim - 1)
                    return jnp.tile(x, reps)
            return x

        return jax.tree_util.tree_map(tile_if_needed, data)

    def wrap_constraints(self, constraints: List) -> List:
        """Auto-wrap raw expressions in OperationDef."""
        wrapped: List[Any] = []
        for expr in constraints:
            if isinstance(expr, (OperationDef, OperationCall)):
                wrapped.append(expr)
            elif isinstance(expr, Hessian) and isinstance(expr.target, (OperationDef, OperationCall)):
                wrapped.append(expr)
            elif isinstance(expr, Placeholder):
                wrapped.append(OperationDef(expr))
            else:
                wrapped.append(expr)
        return wrapped

    def collect_unique_operations(self, constraints: List) -> List:
        """Collect all unique operations from constraints."""
        all_ops = []
        seen_ops = set()
        for expr in constraints:
            for op in collect_operations(expr):
                if op.op_id not in seen_ops:
                    seen_ops.add(op.op_id)
                    all_ops.append(op)
        return all_ops

    def _collect_flax_modules(self) -> Dict[int, Model]:
        """Return ``{layer_id: Model}`` for every model in the problem."""
        from .trace_compiler import TraceCompiler

        result = {}
        for op in self.all_ops:
            for layer, _ in TraceCompiler.collect_dense_layers(op.expr):
                if isinstance(layer, Model) and layer.layer_id not in result:
                    result[layer.layer_id] = layer
        return result

    def set_optimizer(self, opt_fn, *, lr=None):
        """Set the same optimizer (and LR schedule) on **all** models.

        Useful after ``core.load()`` when original Python variables are
        no longer connected to the loaded expression tree.

        Args:
            opt_fn: Optimizer factory, e.g. ``optax.adam``.
            lr:     ``LearningRateSchedule`` or float.
        """
        for fm in self._collect_flax_modules().values():
            fm.optimizer(opt_fn, lr=lr)
        return self

    def get_constraint_tags(self, constraints: List) -> List[str]:
        """Get the primary tag for each constraint."""
        tags = []
        for expr in constraints:
            tag = get_primary_tag(expr)
            tags.append(tag if tag is not None else "default")
        return tags

    @staticmethod
    def _strip_reduction_for_resampling(expr: Placeholder) -> Placeholder:
        """Unwrap terminal reduction calls to recover pointwise residuals.

        If the constraint is ``residual.mse`` (or similar reduction),
        resampling needs the unreduced ``residual`` field to score points.

        Also handles ``weight * residual.mse`` patterns produced by adaptive
        loss balancers: walks through ``BinaryOp`` multiplication nodes and
        strips the reduction from the operand that contains it.

        Transparently unwraps ``OperationDef`` envelopes that
        ``wrap_constraints`` adds around every constraint expression.
        """
        # Unwrap OperationDef envelope if present.
        wrapped_in_opdef = isinstance(expr, OperationDef)
        if wrapped_in_opdef:
            node = cast(Placeholder, getattr(expr, "expr"))
        else:
            node = expr

        # Walk through BinaryOp wrappers (e.g. w0 * pde.mse) to find the
        # operand that carries the pointwise reduction.
        if isinstance(node, BinaryOp) and node.op == "*":
            left_stripped = core._strip_reduction_inner(node.left)
            right_stripped = core._strip_reduction_inner(node.right)
            # Prefer the operand whose strip actually changed something
            # (i.e. it had a reduction to unwrap).
            if left_stripped is not node.left:
                result = left_stripped
            elif right_stripped is not node.right:
                result = right_stripped
            else:
                # Neither side had a reduction — nothing to strip
                result = node
        else:
            result = core._strip_reduction_inner(node)

        # Re-wrap so the compiled expression list stays consistent.
        if wrapped_in_opdef and result is not node:
            return OperationDef(result)
        return expr if result is node else result

    @staticmethod
    def _strip_reduction_inner(node: Placeholder) -> Placeholder:
        """Peel off terminal FunctionCall nodes that reduce an axis."""
        while isinstance(node, FunctionCall) and getattr(node, "reduces_axis", False) and len(node.args) == 1:
            node = node.args[0]
        return node

    def compute_tensor_dims(self, domain) -> Dict[str, Tuple]:
        """Compute input dimensions for each context entry."""
        tensor_dims = {}
        if hasattr(domain, "context"):
            for name, tensor in domain.context.items():
                if isinstance(tensor, dict) or not hasattr(tensor, "shape"):
                    continue
                tensor_dims[name] = tensor.shape[
                    2:
                ]  # was 1 but I think 2 is right for the (B, T, ...) shape of context tensors
        return tensor_dims

    def _populate_missing_context_tags(self, domain) -> None:
        """Populate missing tags on a new eval domain.

        For tags originally created via ``domain.variable(..., sample=...)``,
        prefer re-sampling from the provided domain so the points reflect that
        domain's geometry. If a tag cannot be re-sampled there, fall back to
        copying the already-materialized context from ``self.domain``. Existing
        tags on the provided domain are never overwritten.
        """
        if self.domain is None or domain is self.domain:
            return

        source_context = getattr(self.domain, "context", None)
        target_context = getattr(domain, "context", None)
        if not source_context or target_context is None:
            return

        sample_records = getattr(self.domain, "sample_dict", None) or []
        for record in sample_records:
            if isinstance(record, dict):
                source_tag = record.get("source_tag")
                resolved_tag = record.get("resolved_tag", source_tag)
                sample = record.get("sample")
                resampling_strategy = record.get("resampling_strategy")
                normals = bool(record.get("normals", False))
                reverse_normals = bool(record.get("reverse_normals", False))
                view_factor = bool(record.get("view_factor", False))
            else:
                source_tag = record[0] if len(record) > 0 else None
                resolved_tag = source_tag
                sample = record[1] if len(record) > 1 else None
                resampling_strategy = record[2] if len(record) > 2 else None
                normals = bool(record[3]) if len(record) > 3 else False
                reverse_normals = False
                view_factor = bool(record[4]) if len(record) > 4 else False

            if source_tag is None or resolved_tag in target_context:
                continue

            if source_tag in getattr(domain, "_mesh_pool", {}) and isinstance(sample, tuple):
                try:
                    domain.variable(
                        source_tag,
                        sample=sample,
                        resampling_strategy=resampling_strategy,
                        normals=normals,
                        reverse_normals=reverse_normals,
                        view_factor=view_factor,
                    )
                    continue
                except Exception as exc:
                    self.log.warning(
                        f"Falling back to copied context for '{resolved_tag}': could not sample on provided domain ({exc})"
                    )

        for tag, value in source_context.items():
            if tag in target_context:
                continue

            if isinstance(value, dict):
                target_context[tag] = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), value)
            else:
                target_context[tag] = np.asarray(value).copy()

        if hasattr(self.domain, "_param_tags") and hasattr(domain, "_param_tags"):
            domain._param_tags.update(tag for tag in self.domain._param_tags if tag in domain.context)

    def prepare_domain_data(self, domain) -> DomainData:
        """Convert domain data to JAX arrays for training."""
        if domain is None:
            raise ValueError("domain required")

        self._populate_missing_context_tags(domain)

        context = {}
        # List of tags that are MESH METADATA and should never be batched
        metadata_tags = [
            "JxW",
            "flat_cells",
            "global_areas",
            "N_flat",
            "dN_dx_flat",
            "dirichlet_nodes",
            "cells",
            "quad_points",
            "boundary_nodes",
            "surface_data",
            "v_grads_JxW_flat",
            "__time__",
        ]
        if hasattr(domain, "context"):
            for tag, arr in domain.context.items():
                # If it's a nested dictionary (like our VPINN surface_data), map safely
                if isinstance(arr, dict):
                    # 1. Convert leaves to arrays
                    arr = jax.tree_util.tree_map(jnp.asarray, arr)
                    # 2. Add the batch dimension [None, ...] to every array in the dict, FEM/static metadata dicts must stay unbatched
                    if tag in metadata_tags:
                        context[tag] = arr
                    else:
                        context[tag] = jax.tree_util.tree_map(lambda x: x[None, ...], arr)
                    # 3. Skip the rest of the loop for dictionaries!
                    continue
                    # Standard behavior for everything else (preserves backward compatibility)
                arr = jnp.asarray(arr)

                if tag in metadata_tags:
                    context[tag] = arr
                elif hasattr(arr, "ndim") and arr.ndim >= 2:
                    context[tag] = arr
                else:
                    context[tag] = arr[None, ...]

        return DomainData(
            context=context,
            dimension=domain.dimension,
        )

    # Vpinn helpers
    def _is_marked_weak(self, node) -> bool:
        return isinstance(node, Placeholder) and bool(getattr(node, "_is_weak_expr", False))

    def _materialize_marked_weak(self, node, parent_is_weak: bool = False):
        from .utils.solver.weak_form import assemble_weak_form

        if node is None:
            return None

        is_weak = self._is_marked_weak(node)

        # Recurse through ordinary wrappers first.
        if isinstance(node, BinaryOp):
            left = self._materialize_marked_weak(node.left, parent_is_weak=is_weak)
            right = self._materialize_marked_weak(node.right, parent_is_weak=is_weak)
            if left is not node.left or right is not node.right:
                rebuilt_binary = BinaryOp(node.op, left, right)
                if is_weak:
                    setattr(rebuilt_binary, "_is_weak_expr", True)
                    setattr(
                        rebuilt_binary,
                        "_weak_root_id",
                        getattr(node, "_weak_root_id", None),
                    )
                node = rebuilt_binary

        elif isinstance(node, FunctionCall):
            new_args: List[Any] = []
            changed = False
            for a in node.args:
                if isinstance(a, Placeholder):
                    na = self._materialize_marked_weak(a, parent_is_weak=is_weak)
                    changed = changed or (na is not a)
                    new_args.append(na)
                else:
                    new_args.append(a)
            if changed:
                rebuilt_function = FunctionCall(node.fn, new_args, node._name, node.reduces_axis, node.kwargs)
                if is_weak:
                    setattr(rebuilt_function, "_is_weak_expr", True)
                    setattr(
                        rebuilt_function,
                        "_weak_root_id",
                        getattr(node, "_weak_root_id", None),
                    )
                node = rebuilt_function

        elif isinstance(node, OperationDef):
            new_expr = self._materialize_marked_weak(node.expr, parent_is_weak=is_weak)
            if new_expr is not node.expr:
                rebuilt_opdef = OperationDef(new_expr)
                rebuilt_opdef.name = getattr(node, "name", None)
                if is_weak:
                    setattr(rebuilt_opdef, "_is_weak_expr", True)
                    setattr(
                        rebuilt_opdef,
                        "_weak_root_id",
                        getattr(node, "_weak_root_id", None),
                    )
                node = rebuilt_opdef

        elif isinstance(node, Tracker):
            new_expr = self._materialize_marked_weak(node.expr, parent_is_weak=is_weak)
            if new_expr is not node.expr:
                rebuilt_tracker = Tracker(new_expr, interval=node.interval)
                if is_weak:
                    setattr(rebuilt_tracker, "_is_weak_expr", True)
                    setattr(
                        rebuilt_tracker,
                        "_weak_root_id",
                        getattr(node, "_weak_root_id", None),
                    )
                node = rebuilt_tracker

        # Replace only the OUTERMOST marked weak subtree, after wrapper recursion.
        if self._is_marked_weak(node) and not parent_is_weak:
            return assemble_weak_form(self.domain, node, target="vpinn")

        return node

    # Training
    def _make_loss_fn(
        self,
        compiled_constraints_fn,
        batchsize,
        frozen,
        static,
        checkpoint_gradients=False,
        min_consecutive=1,
    ):
        """Create loss function — evaluates ALL constraints in one combined call."""

        def loss_fn(trainable, context, rng):
            full_models = eqx.combine(trainable, frozen, static)
            full_models = _paramax.unwrap(full_models)

            if checkpoint_gradients:
                _fn, _bs = compiled_constraints_fn, batchsize

                # Equinox wrapper avoids JAX export false-positives in type stubs.
                @eqx.filter_checkpoint
                def _remat_eval(models, ctx, key):
                    return _fn(
                        models,
                        ctx,
                        batchsize=_bs,
                        key=key,
                        min_consecutive=min_consecutive,
                    )

                all_residuals = _remat_eval(full_models, context, rng)
            else:
                # One call → one JAX function → XLA applies CSE across constraints
                all_residuals = compiled_constraints_fn(
                    full_models,
                    context,
                    batchsize=batchsize,
                    key=rng,
                    min_consecutive=min_consecutive,
                )

            # all_residuals is a list of (B, T, ...) arrays — one per constraint
            losses = jnp.stack([jnp.mean(r) for r in all_residuals])
            return jnp.mean(losses), losses

        return loss_fn

    def _make_track_fn(self, compiled_trackers, batchsize, frozen, static):
        """Create tracking function that evaluates monitored expressions.

        Returns a JIT-friendly function that evaluates *all* trackers.
        Interval-based gating is handled by the Python training loop.
        """

        def track_fn(trainable, context, rng):
            full_models = eqx.combine(trainable, frozen, static)
            full_models = _paramax.unwrap(full_models)
            results = []
            for _, fn in compiled_trackers:
                results.append(jnp.mean(fn(full_models, context, batchsize=batchsize, key=rng)))
            return results

        return track_fn

    def make_step_fn(
        self,
        per_model_opts,
        batchsize,
        frozen,
        static,
        lr_schedules,
        group_lr_schedules=None,
        checkpoint_gradients=False,
        min_consecutive=1,
        compiled_constraints_fn=None,
        bayesian_handles=None,
    ):
        """Build a single JIT-compiled training step.

        Returns a function with signature::

            step(trainable, opt_states, rng, context, epoch, prev_losses)
                -> (trainable, opt_states, rng, next_epoch, total_loss, individual_losses)

        The training loop is a plain Python ``for`` loop which:
        * enables buffer donation at every step boundary,
        * allows host-resident data to be streamed per step,
        * makes progress logging trivial (no ``io_callback``).

        Args:
            per_model_opts: ``{layer_id_str: optax_chain}`` per-model optimizers.
            lr_schedules:   ``{layer_id_str: LearningRateSchedule}``.
            checkpoint_gradients: Wrap constraint evaluations in ``jax.checkpoint``.
            compiled_constraints_fn: Override the default combined constraint function.
                Used by ``substeps`` to supply per-substep compiled functions.
            bayesian_handles: Optional ``{layer_id_str: _KernelHandle}`` for
                models configured with ``.bayesian()``.  Their per-step update
                runs a blackjax MCMC kernel instead of the optax chain.
        """
        _compiled_fn = compiled_constraints_fn if compiled_constraints_fn is not None else self.compiled_constraints_fn
        loss_fn = self._make_loss_fn(
            _compiled_fn,
            batchsize,
            frozen,
            static,
            checkpoint_gradients=checkpoint_gradients,
            min_consecutive=min_consecutive,
        )

        bayesian_handles = bayesian_handles or {}
        # per_model_opts and bayesian_handles are disjoint by construction:
        # each model has exactly one backend (optax OR blackjax kernel).
        opt_keys = sorted(per_model_opts.keys())
        bay_keys = sorted(bayesian_handles.keys())
        base_epoch = self._total_epochs
        _group_lr = group_lr_schedules or {}  # {k: [(mask, sched), ..., (None, global_sched)]}

        def step(trainable, opt_states, rng, context, start_epoch, prev_losses):
            rng, step_rng = jax.random.split(rng)

            def loss_wrapper(p):
                return loss_fn(p, context, step_rng)

            (total_loss, individual_losses), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(trainable)

            # ── per-model optimizer step (optax models only) ──
            for k in opt_keys:
                lid = int(k)
                model_grads = grads[lid]
                model_params = trainable[lid]

                updates, new_state = per_model_opts[k].update(
                    model_grads,
                    opt_states[k],
                    model_params,
                    value=total_loss,
                    grad=model_grads,
                    value_fn=lambda p, _lid=lid: loss_fn({**trainable, _lid: p}, context, step_rng)[0],
                )

                # Update LR — either per-group (masked chain) or single global
                if k in _group_lr:
                    # new_state is a tuple: (masked_g0, masked_g1, ..., masked_default)
                    # Each MaskedState has .inner_state = (base_opt_state, inject_scale_state)
                    for i, sched in enumerate(_group_lr[k]):
                        lr_val = sched(base_epoch + start_epoch, individual_losses)
                        new_state[i].inner_state[-1].hyperparams["step_size"] = jnp.asarray(
                            lr_val,
                            dtype=new_state[i].inner_state[-1].hyperparams["step_size"].dtype,
                        )
                else:
                    lr_val = lr_schedules[k](base_epoch + start_epoch, individual_losses)
                    new_state[-1].hyperparams["step_size"] = jnp.asarray(
                        lr_val, dtype=opt_states[k][-1].hyperparams["step_size"].dtype
                    )

                trainable = {
                    **trainable,
                    lid: optax.apply_updates(model_params, updates),
                }
                opt_states = {**opt_states, k: new_state}

            # ── per-model Bayesian step (blackjax kernels) ──
            # Set of Bayesian lids — chain-indexed slicing applies to these
            # other Bayesian models inside each chain's logdensity closure.
            _bay_lid_set = {int(_k) for _k in bay_keys}

            def _pos_of(_state, _kind):
                # K-leading current position of a kernel state.  SG-MCMC
                # states ARE the position; HMC-family states carry it on
                # ``.position``; VI states carry the variational mean on
                # ``.mu`` (used as a per-step representative — actual VI
                # samples are drawn post-solve from the fitted state).
                if _kind == "vi":
                    return _state.mu
                return _state if _kind == "grad_estimator" else _state.position

            for k in bay_keys:
                lid = int(k)
                handle = bayesian_handles[k]
                rng, kernel_key = jax.random.split(rng)

                if handle.num_chains == 1:
                    # ── K=1 backward-compat path ──
                    # Closure form bit-identical to pre-multi-chain code
                    # (no extra factory indirection, no chain-index gather).
                    # ``bayesian.step`` recognises K=1 and avoids vmap;
                    # ``init_state`` keeps state without a K axis.  This
                    # preserves JAX donation/sharding/JIT-trace behaviour.
                    def logdensity_fn(p, _lid=lid, _h=handle):
                        full = {**trainable, _lid: p}
                        nll, _ = loss_fn(full, context, step_rng)
                        return -nll + _h.prior_fn(p)

                    def grad_estimator(p, minibatch, _lid=lid, _h=handle):
                        def neg_log_post(pp):
                            full = {**trainable, _lid: pp}
                            nll, _ = loss_fn(full, minibatch, step_rng)
                            return -nll + _h.prior_fn(pp)

                        return jax.grad(neg_log_post)(p)

                    new_state, new_position = jno_bayesian.step(
                        handle,
                        kernel_key,
                        opt_states[k],
                        trainable[lid],
                        logdensity_fn,
                        grad_estimator,
                        context,
                    )
                    trainable = {**trainable, lid: new_position}
                    opt_states = {**opt_states, k: new_state}
                    continue

                # ── K>1 multi-chain path ──
                # Snapshot per-chain positions of every Bayesian model
                # from ``opt_states`` (K-leading by construction).
                # Iterating ``bay_keys`` in sorted order gives
                # deterministic Gibbs cycling.
                _bay_positions_K = {int(_bk): _pos_of(opt_states[_bk], bayesian_handles[_bk].kind) for _bk in bay_keys}

                def logdensity_factory(p, k_idx, _lid=lid, _h=handle, _bay=_bay_lid_set, _bp=_bay_positions_K):
                    per_chain = {}
                    for _olid, _ov in trainable.items():
                        if _olid == _lid:
                            per_chain[_olid] = p
                        elif _olid in _bay:
                            per_chain[_olid] = jax.tree_util.tree_map(lambda x, _ki=k_idx: x[_ki], _bp[_olid])
                        else:
                            per_chain[_olid] = _ov
                    nll, _ = loss_fn(per_chain, context, step_rng)
                    return -nll + _h.prior_fn(p)

                def grad_estimator_factory(
                    p, minibatch, k_idx, _lid=lid, _h=handle, _bay=_bay_lid_set, _bp=_bay_positions_K
                ):
                    def neg_log_post(pp):
                        per_chain = {}
                        for _olid, _ov in trainable.items():
                            if _olid == _lid:
                                per_chain[_olid] = pp
                            elif _olid in _bay:
                                per_chain[_olid] = jax.tree_util.tree_map(lambda x, _ki=k_idx: x[_ki], _bp[_olid])
                            else:
                                per_chain[_olid] = _ov
                        nll, _ = loss_fn(per_chain, minibatch, step_rng)
                        return -nll + _h.prior_fn(pp)

                    return jax.grad(neg_log_post)(p)

                new_state, new_position = jno_bayesian.step(
                    handle,
                    kernel_key,
                    opt_states[k],
                    trainable[lid],
                    logdensity_factory,
                    grad_estimator_factory,
                    context,
                )

                # K>1: trainable keeps chain-0 as the representative for
                # the outer ``value_and_grad`` / ``individual_losses``
                # logging.  Mixed-mode caveat: optax-trained parameters
                # compute gradients against chain 0 of Bayesian models.
                trainable = {**trainable, lid: jax.tree_util.tree_map(lambda x: x[0], new_position)}
                opt_states = {**opt_states, k: new_state}

            next_epoch = start_epoch + jnp.asarray(1, dtype=start_epoch.dtype)
            return trainable, opt_states, rng, next_epoch, total_loss, individual_losses

        return step

    def make_mcmc_scan_fn(
        self,
        bayesian_handles,
        batchsize,
        frozen,
        static,
        compiled_constraints_fn=None,
        checkpoint_gradients=False,
        min_consecutive=1,
        warmup: int = 0,
        keep: int = 0,
        thin: int = 1,
    ):
        """Build a JIT-compatible scan function for the pure-Bayesian fastpath.

        Closes the three performance gaps in pure-Bayesian solves:

        1. No outer ``value_and_grad`` — only the Bayesian kernels compute
           their gradients (the slow-path discards the outer ``grads`` for
           pure-Bayesian anyway).
        2. ``warmup`` steps run inside a ``jax.lax.fori_loop`` (no sample
           accumulation); ``keep * thin`` post-warmup steps run inside a
           single ``jax.lax.scan`` (one XLA program, stacked sample output).
        3. Samples are stacked inside XLA and returned as one
           ``(keep, *param)`` (K=1) / ``(keep, K, *param)`` (K>1) tensor
           per Bayesian lid — one host transfer per scan-call, not one per
           step.

        Signature of the returned function::

            scan_fn(trainable, opt_states, rng, context)
                -> (trainable, opt_states, rng, samples)

        where ``samples`` is a dict keyed by ``bayesian_handles`` keys, each
        entry stacked along a leading axis of length ``keep``.
        """
        if compiled_constraints_fn is None:
            compiled_constraints_fn = self.compiled_constraints_fn
        loss_fn = self._make_loss_fn(
            compiled_constraints_fn,
            batchsize,
            frozen,
            static,
            checkpoint_gradients=checkpoint_gradients,
            min_consecutive=min_consecutive,
        )
        bay_keys = sorted(bayesian_handles.keys())
        _bay_lid_set = {int(_k) for _k in bay_keys}

        def _pos_of(_state, _kind):
            if _kind == "vi":
                return _state.mu
            return _state if _kind == "grad_estimator" else _state.position

        def _one_step(trainable, opt_states, rng, context):
            """Single Bayesian Gibbs cycle — no outer value_and_grad."""
            rng, step_rng = jax.random.split(rng)
            for k in bay_keys:
                lid = int(k)
                handle = bayesian_handles[k]
                rng, kernel_key = jax.random.split(rng)

                if handle.num_chains == 1:
                    # K=1 OLD-style closures — bit-identical to slow-path.
                    def logdensity_fn(p, _lid=lid, _h=handle):
                        full = {**trainable, _lid: p}
                        nll, _ = loss_fn(full, context, step_rng)
                        return -nll + _h.prior_fn(p)

                    def grad_estimator(p, minibatch, _lid=lid, _h=handle):
                        def neg_log_post(pp):
                            full = {**trainable, _lid: pp}
                            nll, _ = loss_fn(full, minibatch, step_rng)
                            return -nll + _h.prior_fn(pp)

                        return jax.grad(neg_log_post)(p)

                    new_state, new_position = jno_bayesian.step(
                        handle,
                        kernel_key,
                        opt_states[k],
                        trainable[lid],
                        logdensity_fn,
                        grad_estimator,
                        context,
                    )
                    trainable = {**trainable, lid: new_position}
                else:
                    # K>1 chain-indexed factories — same as slow-path.
                    _bay_positions_K = {int(_bk): _pos_of(opt_states[_bk], bayesian_handles[_bk].kind) for _bk in bay_keys}

                    def logdensity_factory(p, k_idx, _lid=lid, _h=handle, _bay=_bay_lid_set, _bp=_bay_positions_K):
                        per_chain = {}
                        for _olid, _ov in trainable.items():
                            if _olid == _lid:
                                per_chain[_olid] = p
                            elif _olid in _bay:
                                per_chain[_olid] = jax.tree_util.tree_map(lambda x, _ki=k_idx: x[_ki], _bp[_olid])
                            else:
                                per_chain[_olid] = _ov
                        nll, _ = loss_fn(per_chain, context, step_rng)
                        return -nll + _h.prior_fn(p)

                    def grad_estimator_factory(
                        p, minibatch, k_idx, _lid=lid, _h=handle, _bay=_bay_lid_set, _bp=_bay_positions_K
                    ):
                        def neg_log_post(pp):
                            per_chain = {}
                            for _olid, _ov in trainable.items():
                                if _olid == _lid:
                                    per_chain[_olid] = pp
                                elif _olid in _bay:
                                    per_chain[_olid] = jax.tree_util.tree_map(lambda x, _ki=k_idx: x[_ki], _bp[_olid])
                                else:
                                    per_chain[_olid] = _ov
                            nll, _ = loss_fn(per_chain, minibatch, step_rng)
                            return -nll + _h.prior_fn(pp)

                        return jax.grad(neg_log_post)(p)

                    new_state, new_position = jno_bayesian.step(
                        handle,
                        kernel_key,
                        opt_states[k],
                        trainable[lid],
                        logdensity_factory,
                        grad_estimator_factory,
                        context,
                    )
                    # K>1: trainable keeps chain-0 representative.
                    trainable = {**trainable, lid: jax.tree_util.tree_map(lambda x: x[0], new_position)}
                opt_states = {**opt_states, k: new_state}

            return trainable, opt_states, rng

        def scan_fn(trainable, opt_states, rng, context):
            """Run ``warmup`` warmup steps (no collection) + ``keep`` outer
            iterations each running ``thin`` inner steps (one sample per
            outer iter).  Returns (trainable, opt_states, rng, samples).
            """
            # --- Phase A: warmup (fori_loop, no sample accumulation) ---
            if warmup > 0:

                def _warmup_body(_i, carry):
                    tr, os_, rg = carry
                    return _one_step(tr, os_, rg, context)

                trainable, opt_states, rng = jax.lax.fori_loop(0, warmup, _warmup_body, (trainable, opt_states, rng))

            # --- Phase B: scan over `keep` outer iters × `thin` inner steps ---
            def _outer_body(carry, _):
                tr, os_, rg = carry

                def _inner_body(_j, c):
                    return _one_step(c[0], c[1], c[2], context)

                if thin > 1:
                    tr, os_, rg = jax.lax.fori_loop(0, thin, _inner_body, (tr, os_, rg))
                else:
                    tr, os_, rg = _one_step(tr, os_, rg, context)

                # Sample = full position per Bayesian/VI lid.  For masked
                # handles (Phase 11) the kernel state holds only the
                # masked subset, but ``tr[lid]`` is the post-step full
                # reassembled position (done in ``jno_bayesian.step``),
                # so we read directly from ``tr`` for masked handles.
                sample = {}
                for k in bay_keys:
                    handle = bayesian_handles[k]
                    if handle.param_mask is not None:
                        sample[k] = tr[int(k)]
                    else:
                        sample[k] = _pos_of(os_[k], handle.kind)
                return (tr, os_, rg), sample

            if keep > 0:
                (trainable, opt_states, rng), samples = jax.lax.scan(
                    _outer_body, (trainable, opt_states, rng), None, length=keep
                )
            else:
                samples = {k: jnp.zeros((0,)) for k in bay_keys}

            return trainable, opt_states, rng, samples

        return scan_fn

    def make_grad_fn(
        self,
        batchsize,
        frozen,
        static,
        checkpoint_gradients=False,
        min_consecutive=1,
    ):
        """Build a function that computes gradients without an optimizer update.

        Returns a function with signature::

            grad_fn(trainable, rng, context)
                -> (grads, total_loss, individual_losses)

        Used by gradient accumulation to compute gradients on multiple
        micro-batches before averaging and applying a single update.
        """
        loss_fn = self._make_loss_fn(
            self.compiled_constraints_fn,
            batchsize,
            frozen,
            static,
            checkpoint_gradients=checkpoint_gradients,
            min_consecutive=min_consecutive,
        )

        def grad_fn(trainable, rng, context):
            rng, step_rng = jax.random.split(rng)

            def loss_wrapper(p):
                return loss_fn(p, context, step_rng)

            (total_loss, individual_losses), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(trainable)
            return grads, rng, total_loss, individual_losses

        return grad_fn

    def make_apply_fn(
        self,
        per_model_opts,
        lr_schedules,
        group_lr_schedules=None,
    ):
        """Build a function that applies pre-computed gradients via the optimizer.

        Returns a function with signature::

            apply_fn(trainable, opt_states, grads, epoch, prev_losses)
                -> (trainable, opt_states)

        Used together with :meth:`make_grad_fn` for gradient accumulation.
        """
        lid_keys = sorted(per_model_opts.keys())
        base_epoch = self._total_epochs
        _group_lr = group_lr_schedules or {}

        def apply_fn(trainable, opt_states, grads, epoch, prev_losses):
            for k in lid_keys:
                lid = int(k)
                model_grads = grads[lid]
                model_params = trainable[lid]

                updates, new_state = per_model_opts[k].update(
                    model_grads,
                    opt_states[k],
                    model_params,
                )

                # Update LR — either per-group or single global
                if k in _group_lr:
                    for i, sched in enumerate(_group_lr[k]):
                        lr_val = sched(base_epoch + epoch, prev_losses)
                        new_state[i].inner_state[-1].hyperparams["step_size"] = jnp.asarray(
                            lr_val,
                            dtype=new_state[i].inner_state[-1].hyperparams["step_size"].dtype,
                        )
                else:
                    lr_val = lr_schedules[k](base_epoch + epoch, prev_losses)
                    new_state[-1].hyperparams["step_size"] = jnp.asarray(
                        lr_val,
                        dtype=opt_states[k][-1].hyperparams["step_size"].dtype,
                    )

                trainable = {
                    **trainable,
                    lid: optax.apply_updates(model_params, updates),
                }
                opt_states = {**opt_states, k: new_state}

            return trainable, opt_states

        return apply_fn

    def print_tree(self, file: Optional[str] = None):
        """Print the computation tree for every constraint and tracker.

        Call this **after** constructing the ``core`` object (which calls
        ``compile`` internally) so that ``self.constraints`` is populated.

        Args:
            file: Optional path.  When given the tree is written to that
                file; otherwise it is printed to stdout.

        Example::

            crux = jno.core([pde.mse, ini.mse], domain)
            crux.print_tree("tree.txt")
        """
        constraints = self.wrap_constraints(self.constraints)
        parts: list[str] = []
        for i, expr in enumerate(constraints):
            if isinstance(expr, OperationDef) and isinstance(expr.expr, Tracker):
                parts.append(f"=== Tracker {i} ===")
                parts.append(dump_tree(expr))
            elif isinstance(expr, Tracker):
                parts.append(f"=== Tracker {i} ===")
                parts.append(dump_tree(expr))
            else:
                parts.append(f"=== Constraint {i} ===")
                parts.append(dump_tree(expr))
            parts.append("")

        text = "\n".join(parts)
        if file is not None:
            from pathlib import Path as _P

            _P(file).parent.mkdir(parents=True, exist_ok=True)
            _P(file).write_text(text)
            self.log.info(f"Computation tree written to {file}")
        else:
            self.log.info(text)

        return self

    def compile(self, mesh: Optional[Tuple[int, ...]] = (1, 1)):

        # === Parallelism ===
        self._setup_parallelism(mesh)

        # === Preprocessing ===
        # Check if its Vpinn and routes accordingly
        constraints_in = [self._materialize_marked_weak(c) for c in self.constraints]
        constraints = self.wrap_constraints(constraints_in)

        # === Collect operations and tags ===
        self.all_ops = self.collect_unique_operations(constraints)

        # === CSE: deduplicate shared sub-expressions ===
        constraints = [cse(c) for c in constraints]
        self.all_ops = self.collect_unique_operations(constraints)

        # === Prepare domain data ===
        self.domain_data = self.prepare_domain_data(self.domain)
        tensor_dims = self.compute_tensor_dims(self.domain)

        # === Initialize models ===
        self.models, self.rng = TraceCompiler.init_layer_params(
            self.all_ops, self.domain_data.dimension, tensor_dims, self.rng, self.log
        )

        # === Apply sharding to model arrays ===
        self.models = self._shard_params(self.models)

        # === Compile constraints and trackers ===
        self.compiled_trackers = []
        self._constraint_exprs = []  # raw expressions for shape tracing
        # Original (pre-wrap, pre-CSE) constraint expressions in the same order
        # as ``_constraint_exprs`` — used by callbacks like residual_stats /
        # hessian_spectrum that accept a ``constraints=`` subset and need to
        # identity-match the user's original Python objects.
        self._user_constraint_exprs = []
        self._tracker_exprs = []
        constraint_exprs = []

        for orig_expr, expr in zip(self.constraints, constraints):
            inner = expr
            tracker_interval = None
            if isinstance(expr, OperationDef) and isinstance(expr.expr, Tracker):
                tracker_interval = expr.expr.interval
                inner = OperationDef(expr.expr.expr)
            elif isinstance(expr, Tracker):
                tracker_interval = expr.interval
                inner = expr.expr

            if tracker_interval is not None:
                fn_expr = TraceCompiler.compile_traced_expression(inner, self.all_ops)
                self.compiled_trackers.append((tracker_interval, fn_expr))
                self._tracker_exprs.append(inner)
            else:
                constraint_exprs.append(inner)
                self._constraint_exprs.append(inner)
                self._user_constraint_exprs.append(orig_expr)

        # Compile all normal constraints in ONE combined function so XLA
        # can apply CSE across shared sub-expressions.
        self.compiled_constraints_fn = TraceCompiler.compile_multi_expression(constraint_exprs, self.all_ops)
        self.n_constraints = len(constraint_exprs)

        # Keep tag metadata and a pointwise residual function for adaptive
        # resampling. The normal training loss still uses reduced constraints
        # in ``self.compiled_constraints_fn``.
        self._resample_exprs = [self._strip_reduction_for_resampling(expr) for expr in self._constraint_exprs]
        # Derive tags from the *stripped* expressions so that adaptive-weight
        # wrappers (which reference all losses) don't contaminate the tag set.
        self._constraint_tags = self.get_constraint_tags(self._resample_exprs)
        self.compiled_resample_constraints_fn = TraceCompiler.compile_multi_expression(self._resample_exprs, self.all_ops)

        # self.log.info(f"There are a total of {self.count(self.models)} trainable parameters in the network/s.")
        return None

    def solve(
        self,
        epochs: int = 1000,
        batchsize: Optional[int] = None,
        checkpoint_gradients: bool = False,
        offload_data: bool = False,
        inner_steps: int = 1,
        accumulation_steps: int = 1,
        min_consecutive: Optional[int] = 1,
        profile: bool = False,
        callbacks: Optional[List] = None,
        substeps=None,
    ):
        """Train using per-model optimizers attached via ``model.optimizer()``.

        Every model used in the constraints **must** have an optimizer
        attached before calling ``solve()``.  Models can optionally be
        frozen (``model.freeze()``) or have LoRA enabled
        (``model.lora(rank, alpha)``).

        Args:
            epochs: Number of training epochs.
            batchsize: Mini-batch size (``None`` for full-batch).
            checkpoint_gradients: If ``True``, wrap each constraint's
                forward pass in ``jax.checkpoint`` (gradient
                checkpointing / activation rematerialisation).  Trades
                ~30 % extra compute for significantly lower activation
                memory.  Default ``False``.
            offload_data: If ``True``, keep the full training dataset in
                host (CPU) memory and stream only the current mini-batch
                to the device each step.  Requires ``batchsize`` to be
                set.  Default ``False``.
            inner_steps: Number of gradient steps to fuse into a single
                ``jax.lax.fori_loop`` call, amortising Python dispatch
                overhead.  Must evenly divide *epochs*.  Default ``1``.
            accumulation_steps: Number of micro-batches whose gradients
                are averaged before a single optimizer update.  The
                effective batch size becomes
                ``batchsize * accumulation_steps`` while peak activation
                memory stays proportional to *batchsize*.  Requires
                ``batchsize`` to be set.  Default ``1``.
            min_consecutive: Minimum number of consecutive time steps
                fed to each constraint evaluation. ``None`` means use all
                available time steps. Default ``1``.
            profile: If ``True``, capture a JAX profiler trace for a
                short window of steady-state training steps.  The trace
                is written to ``<logger.path>/traces``.  Default
                ``False``.
            callbacks: Optional list of :class:`~jno.utils.callbacks.Callback`
                instances.  ``on_epoch_end`` is called after every outer
                step; ``on_training_end`` is called once after the loop
                finishes.
            substeps: Optional list of substep specs for alternating
                optimisation.  Each entry is either a plain list of constraint
                indices ``[i, j, ...]`` (1 gradient step) or a tuple
                ``([i, j, ...], n)`` (``n`` gradient steps sharing the same
                optimizer state).  Each substep runs sequentially per outer
                epoch and has its own independent optimizer states, so Adam
                momentum accumulates only for actively trained models.

                Example — HyCo alternating::

                    crux = jno.core(
                        [L_pde, beta * L_int_phy, alpha * L_data, beta * L_int_syn],
                        domain,
                    )
                    crux.solve(1_500, substeps=[[0, 1], [2, 3]])
                    # 1500 outer epochs × 2 substeps = 3000 effective gradient steps

        Returns:
            statistics: Training history with ``.plot()`` convenience.
        """
        from contextlib import nullcontext

        from jax._src import profiler as _jax_profiler

        if not self.constraints:
            raise ValueError(
                "solve() requires at least one constraint. "
                "Pass a non-empty list to jno.core([...], domain) — typically a "
                "PDE residual .mse, a boundary-condition loss, or a data-fitting term."
            )

        _profiling = _jax_profiler._profile_state.profile_session is not None
        _trace = jax.profiler.TraceAnnotation if _profiling else lambda name, **_: nullcontext()

        batchsize = batchsize if batchsize is not None else self.domain.total_samples

        # Validate accumulation_steps
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
        if accumulation_steps > 1 and batchsize >= self.domain.total_samples:
            self.log.warning(
                "accumulation_steps > 1 has no effect with full-batch training; falling back to accumulation_steps=1"
            )
            accumulation_steps = 1

        # Guard: IntegralTime requires min_consecutive >= 2
        def _has_integral_time(node):
            if isinstance(node, IntegralTime):
                return True
            for attr in ("target", "left", "right", "expr"):
                child = getattr(node, attr, None)
                if isinstance(child, Placeholder) and _has_integral_time(child):
                    return True
            for attr in ("args", "variables"):
                for child in getattr(node, attr, []):
                    if isinstance(child, Placeholder) and _has_integral_time(child):
                        return True
            return False

        if min_consecutive is not None and min_consecutive < 2:
            for expr in getattr(self, "_constraint_exprs", []):
                if _has_integral_time(expr):
                    raise ValueError(
                        f"IntegralTime (.integrate(t)) requires min_consecutive >= 2 "
                        f"(trapezoidal integration over a single time step is identically zero). "
                        f"Got min_consecutive={min_consecutive} (the default is 1). "
                        f"Pass min_consecutive=None to use all T time steps, "
                        f"or min_consecutive=2 for the minimum valid windowed integration."
                    )

        if (
            min_consecutive == 1
            and getattr(self.domain, "_is_time_dependent", False)
            and not getattr(self, "_min_consec_nudged", False)
        ):
            self.log.info(
                "Time-dependent domain with min_consecutive=1: each step sees a single "
                "time slice. Pass min_consecutive=None (all T) or >=2 for true "
                "spatiotemporal training."
            )
            self._min_consec_nudged = True

        # Validate substeps
        _use_substeps = substeps is not None
        if _use_substeps:
            if accumulation_steps > 1:
                raise ValueError("substeps is not compatible with accumulation_steps > 1.")
            if inner_steps > 1:
                raise ValueError(
                    "substeps is not compatible with inner_steps > 1. Use the n_steps tuple form instead: ([i, j], n)."
                )
            _parsed_substeps = [_parse_substep_spec(s) for s in substeps]
            for si, (indices, _) in enumerate(_parsed_substeps):
                for idx in indices:
                    if idx < 0 or idx >= self.n_constraints:
                        raise ValueError(
                            f"substeps[{si}] references constraint index {idx}, "
                            f"but there are only {self.n_constraints} constraints (indices 0–{self.n_constraints - 1})."
                        )

        # Adaptive resampling metadata
        strategies = getattr(self.domain, "_resampling_strategies", {})
        has_resampling = bool(strategies)
        if has_resampling and inner_steps > 1:
            self.log.warning("Adaptive resampling with inner_steps > 1 is applied at outer-step boundaries only.")

        tag_to_constraint_indices: Dict[str, List[int]] = {}
        # Map each constraint to *every* tag it touches so that resampling
        # strategies always find matching constraints, even when adaptive
        # weights or multi-variable expressions make get_primary_tag fragile.
        resample_exprs = getattr(self, "_resample_exprs", getattr(self, "_constraint_exprs", []))
        for i, expr in enumerate(resample_exprs):
            for tag in collect_tags(expr):
                tag_to_constraint_indices.setdefault(tag, []).append(i)

        def _infer_total_samples(ctx: Dict[str, np.ndarray]) -> int:
            candidates = [
                v.shape[0] for k, v in ctx.items() if k != "__time__" and hasattr(v, "shape") and len(v.shape) >= 1
            ]
            if candidates:
                return int(max(candidates))
            fallback = [v.shape[0] for v in ctx.values() if hasattr(v, "shape") and len(v.shape) >= 1]
            return int(max(fallback)) if fallback else batchsize

        def _collapse_residual_for_tag(residual: jax.Array, n_points: int, n_batch: int) -> Optional[jax.Array]:
            """Reduce residual to shape (B, N) for strategy scoring."""
            arr = jnp.abs(jnp.asarray(residual))

            if arr.ndim == 0:
                return None

            # No explicit batch axis.
            if arr.ndim == 1:
                if arr.shape[0] != n_points:
                    return None
                return jnp.broadcast_to(arr[None, :], (n_batch, n_points))

            has_batch_axis = arr.shape[0] == n_batch
            search_start = 1 if has_batch_axis else 0
            candidate_axes = [ax for ax in range(search_start, arr.ndim) if arr.shape[ax] == n_points]
            if not candidate_axes:
                return None

            # Prefer trailing point axis first (common layouts: B,T,N or B,T,N,C).
            point_axis = candidate_axes[-1]

            if has_batch_axis:
                reduce_axes = tuple(ax for ax in range(arr.ndim) if ax not in (0, point_axis))
                collapsed = jnp.mean(arr, axis=reduce_axes) if reduce_axes else arr
                if point_axis != 1:
                    collapsed = jnp.moveaxis(collapsed, -1, 1)
                return collapsed

            reduce_axes = tuple(ax for ax in range(arr.ndim) if ax != point_axis)
            collapsed = jnp.mean(arr, axis=reduce_axes) if reduce_axes else arr
            return jnp.broadcast_to(collapsed[None, :], (n_batch, n_points))

        def _rebuild_runtime_contexts(
            full_ctx: Dict[str, jax.Array],
            offload_enabled: bool,
            n_devices_local: int,
            total_samples_local: int,
        ):
            """Rebuild per-step runtime context after host-side mutations."""
            if offload_enabled:
                host_ctx = {k: np.asarray(v) for k, v in full_ctx.items()}
                total_samples_local = _infer_total_samples(host_ctx)
                return host_ctx, None, total_samples_local

            replicated_ctx = DomainData(
                context=self._replicate_for_devices(full_ctx, n_devices_local),
                dimension=self.domain_data.dimension,
            )
            sharded_ctx = DomainData(
                context=self._shard_data(replicated_ctx.context),
                dimension=replicated_ctx.dimension,
            )
            return None, sharded_ctx.context, total_samples_local

        # ── 0. Validate offload_data ──
        if offload_data and (batchsize is None or batchsize >= self.domain.total_samples):
            self.log.warning("offload_data requires batchsize < total_samples; ignoring offload_data for this run.")
            offload_data = False

        self.log.info("Paramax auto-unwrap enabled: wrappers are unwrapped before each forward evaluation")

        # ── 1. Collect Model metadata ──
        flax_mods = self._collect_flax_modules()  # {layer_id: Model}

        # Validate: every non-frozen model must have either an optimizer or
        # a Bayesian sampler attached.  A frozen model that has LoRA active
        # is also "effectively trainable" (LoRA overrides freeze) and
        # therefore needs one of the two.
        for lid, fm in flax_mods.items():
            needs_update = (not fm._frozen) or (fm._lora_config is not None)
            has_global_backend = (
                fm._opt_fn is not None or fm._bayesian_cfg is not None or getattr(fm, "_vi_cfg", None) is not None
            )
            # Phase 11: a masked .bayesian() / .vi() / .optimizer() call
            # registers a group entry without setting a global backend.
            # Treat that as a valid backend for the model.
            has_group_backend = any(
                g.get("opt_fn") is not None or g.get("backend") in ("bayesian", "vi") for g in fm._param_groups
            )
            has_backend = has_global_backend or has_group_backend
            if needs_update and not has_backend:
                raise ValueError(
                    f"Model '{fm.name or type(fm.module).__name__}' (layer {lid}) "
                    f"has no optimizer. Attach one with model.optimizer(...), "
                    f"sample its posterior with model.bayesian(...), fit a "
                    f"variational approximation with model.vi(...), or freeze "
                    f"the model with model.freeze().\n"
                    f"Example setup:\n"
                    f"    import jno, optax\n"
                    f"    model = jno.nn.wrap(my_eqx_module)\n"
                    f"    model.optimizer(optax.adam, lr=1e-3)"
                )

        # ── 2. Apply LoRA transforms ──
        models = dict(self.models)
        lora_param_counts: Dict[int, Any] = {}  # Track LoRA params per model for logging
        for lid, fm in flax_mods.items():
            if fm._lora_config is not None:
                self.rng, key = jax.random.split(self.rng)
                n_params_before = sum(param.size for param in jax.tree_util.tree_leaves(models[lid]) if eqx.is_array(param))

                models[lid] = _apply_lora(
                    models[lid],
                    key=key,
                    specs=fm._lora_config,
                    param_mask=getattr(fm, "_lora_param_mask", None),
                )

                model_after = models[lid]
                n_params_after = sum(param.size for param in jax.tree_util.tree_leaves(model_after) if eqx.is_array(param))
                n_lora_params = n_params_after - n_params_before
                lora_param_counts[lid] = n_lora_params

                is_lora = lambda x: isinstance(x, _LoRAWrapper)
                lora_leaves = [
                    leaf
                    for leaf in jax.tree_util.tree_leaves(model_after, is_leaf=is_lora)
                    if isinstance(leaf, _LoRAWrapper)
                ]
                n_lora_layers = len(lora_leaves)

                # Group by (wrapper type, rank, alpha) for reporting
                rank_groups: Dict[tuple, int] = {}
                for ll in lora_leaves:
                    rk = (type(ll).__name__, ll.rank, ll.alpha)
                    rank_groups[rk] = rank_groups.get(rk, 0) + 1

                if len(rank_groups) == 1:
                    (cls_name, r, a), cnt = next(iter(rank_groups.items()))
                    self.log.info(
                        f"LoRA applied to model {lid} ({cls_name}, rank={r}, alpha={a}): "
                        f"{cnt} adapter layers, "
                        f"Params: {n_params_before:,}→{n_params_after:,}"
                    )
                else:
                    parts = [f"{cls_name}(r={r}/a={a})×{cnt}" for (cls_name, r, a), cnt in sorted(rank_groups.items())]
                    self.log.info(
                        f"LoRA applied to model {lid}: {n_lora_layers} layers "
                        f"[{', '.join(parts)}], "
                        f"Params: {n_params_before:,}→{n_params_after:,}"
                    )

                if n_lora_layers == 0:
                    self.log.warning(f"LoRA: No layers were adapted for model {lid}! LoRA has NO EFFECT on this model.")

        # ── 3. Build trainable filter ──
        filter_spec = {}
        for lid, model in models.items():
            fm = flax_mods.get(lid)
            if fm is not None and fm._lora_config is not None:
                # LoRA modes:
                # 1) fm._frozen=True  -> freeze everything; only LoRA adapters trainable.
                #    (freeze().lora() or freeze().mask(M).lora() both land here — base
                #    params outside LoRA-wrapped layers are frozen too.)
                # 2) otherwise        -> partial LoRA: adapters trainable, wrapped bases
                #    frozen, every other param stays trainable (mask(M).lora() case).
                if fm._frozen:
                    filter_spec[lid] = _lora_trainable_filter(model)
                else:
                    filter_spec[lid] = _partial_lora_trainable_filter(model)
            elif fm is not None and fm._frozen:
                # Whole model frozen – no arrays trainable
                filter_spec[lid] = jax.tree_util.tree_map(lambda leaf: False, model)
            elif fm is not None and fm._trainable_param_mask is not None:
                # Partial mask — only leaves marked True in the mask are trained.
                # Non-array leaves (e.g. activation functions kept as module
                # attributes) are always False so equinox does not misinterpret
                # them as sub-filter callables.
                filter_spec[lid] = jax.tree_util.tree_map(
                    # Only train floating/complex arrays; integer/bool arrays
                    # (e.g. RNG/state tensors in wrapped modules) must stay frozen.
                    lambda arr, m: bool(m) if eqx.is_inexact_array(arr) else False,
                    model,
                    fm._trainable_param_mask,
                )
            else:
                # Normal – every array trainable, non-arrays (e.g. activation
                # functions stored as attributes) must be False, not the
                # original value — equinox interprets callables in the
                # filter spec as sub-filters.
                filter_spec[lid] = jax.tree_util.tree_map(
                    # Gradients are defined only for inexact dtypes.
                    lambda leaf: True if eqx.is_inexact_array(leaf) else False,
                    model,
                )

        # ── 4. Three-way partition ──
        trainable, rest = eqx.partition(models, filter_spec)
        frozen_arrays, static = eqx.partition(rest, eqx.is_array)

        # Stash for restore_checkpoint()
        self._last_frozen_arrays = frozen_arrays
        self._last_static = static

        # ── 4b. Log parameter counts ──
        def _count_params(pytree):
            """Count total parameters in a pytree."""
            return sum(param.size for param in jax.tree_util.tree_leaves(pytree) if eqx.is_array(param))

        n_trainable_params = _count_params(trainable)
        n_frozen_params = _count_params(frozen_arrays)
        n_total_params = n_trainable_params + n_frozen_params
        n_lora_params_total = sum(lora_param_counts.values())

        self.log.info("Parameter summary:")
        self.log.info(f"    Trainable parameters:  {n_trainable_params:>12,}")
        self.log.info(f"    Frozen parameters:     {n_frozen_params:>12,}")
        self.log.info(f"    Total parameters:      {n_total_params:>12,}")
        if n_lora_params_total > 0:
            self.log.info(f"    LoRA parameters:       {n_lora_params_total:>12,} (included in trainable)")
            self.log.info(f"    LoRA % of total:       {100.0 * n_lora_params_total / n_total_params:>11.2f}%")

        # ── Per-model policy report ──
        for lid, model in models.items():
            fm = flax_mods.get(lid)
            if fm is None:
                continue
            mname = fm.name or type(fm.module).__name__
            m_total = sum(param.size for param in jax.tree_util.tree_leaves(model) if eqx.is_array(param))
            fspec = filter_spec.get(lid)
            if fspec is not None:
                m_train = sum(
                    param.size
                    for param, f in zip(
                        jax.tree_util.tree_leaves(model),
                        jax.tree_util.tree_leaves(fspec),
                    )
                    if eqx.is_array(param) and f is True
                )
            else:
                m_train = m_total
            m_frozen = m_total - m_train
            m_lora = lora_param_counts.get(lid, 0)
            policy = "LoRA" if fm._lora_config else ("frozen" if fm._frozen else "full fine-tune")
            if fm._trainable_param_mask is not None and fm._lora_config:
                policy = "LoRA + partial base"
            self.log.info(f"    [{mname}] policy={policy}  train={m_train:,}  frozen={m_frozen:,}  lora={m_lora:,}")

        # Shard trainable params
        trainable = self._shard_params(trainable)

        # ── 5. Build per-model optimizers and Bayesian kernels ──
        per_model_opts: Dict[str, optax.GradientTransformation] = {}  # {str(lid): optax chain}
        lr_schedules: Dict[str, Any] = {}  # {str(lid): LearningRateSchedule} — global only
        group_lr_schedules: Dict[str, Any] = {}  # {str(lid): [sched_per_masked_group]} — when groups present
        bayesian_handles: Dict[str, Any] = {}  # {str(lid): _KernelHandle} for .bayesian() models
        zeros = jnp.zeros(self.n_constraints)

        _has_bayesian = any(fm._bayesian_cfg is not None for fm in flax_mods.values())
        if accumulation_steps > 1 and _has_bayesian:
            raise ValueError(
                "Combining accumulation_steps>1 with .bayesian() is not supported in this release. "
                "Bayesian models are dispatched through the blackjax kernel; the optax-style "
                "gradient accumulation does not apply to them."
            )

        def _build_opt_chain(opt_fn, lr_sched):
            """Build an optax chain with inject_hyperparams LR scaling."""
            if opt_fn is None:
                raise ValueError("Optimizer function cannot be None for trainable models.")

            if callable(opt_fn) and not isinstance(opt_fn, optax.GradientTransformation):
                try:
                    base = opt_fn(1.0)
                except TypeError:
                    base = opt_fn
            else:
                base = opt_fn

            if not isinstance(base, optax.GradientTransformation):
                raise TypeError(f"Unsupported optimizer type: {type(base)}")

            if lr_sched is None:
                lr_sched = LearningRateSchedule(1e-3)

            scale = optax.inject_hyperparams(optax.scale)(step_size=lr_sched(0, zeros))
            return optax.chain(base, scale)

        for lid, fm in flax_mods.items():
            # Skip only if truly frozen with no LoRA override.
            if fm._frozen and fm._lora_config is None:
                continue
            k = str(lid)

            # Split parameter groups by backend tag (Phase 11).  Existing
            # `.mask().optimizer()` entries default to backend="optax" so
            # this is backwards-compatible.
            _non_optax_groups = [g for g in fm._param_groups if g.get("backend") in ("bayesian", "vi")]
            _optax_groups = [g for g in fm._param_groups if g.get("backend", "optax") == "optax"]

            # Bayesian models: route through bayesian_handles only — they
            # are NOT added to per_model_opts.  The step dispatcher checks
            # both dicts; opt_states carries either an optax state (for
            # optax models) or a kernel state (for Bayesian models).
            if fm._bayesian_cfg is not None:
                if fm._param_groups:
                    raise ValueError(
                        f"Model (layer {lid}): cannot combine global .bayesian() with .mask() "
                        "groups.  Either drop the global .bayesian() and use one or more "
                        ".mask(M).bayesian() calls, or drop the masks."
                    )
                handle = jno_bayesian.build_kernel_handle(fm._bayesian_cfg)
                bayesian_handles[k] = handle
                self.log.info(
                    f"Model {lid}: Bayesian sampling via "
                    f"{getattr(handle.factory, '__name__', handle.factory)!r} "
                    f"(kind={handle.kind}, warmup={handle.warmup}, keep={handle.keep}, thin={handle.thin})"
                )
                continue

            # VI models: route through the same ``bayesian_handles`` dict
            # (the per-step dispatch in ``make_step_fn`` / ``make_mcmc_scan_fn``
            # branches on ``handle.kind == 'vi'`` internally).  VI handles
            # are excluded from the buffer-collection block and instead
            # produce ``posterior_samples`` via a post-loop draw from the
            # fitted variational distribution.
            if getattr(fm, "_vi_cfg", None) is not None:
                if fm._param_groups:
                    raise ValueError(
                        f"Model (layer {lid}): cannot combine global .vi() with .mask() "
                        "groups.  Use .mask(M).vi(...) instead."
                    )
                handle = jno_bayesian.build_vi_handle(fm._vi_cfg)
                bayesian_handles[k] = handle
                self.log.info(
                    f"Model {lid}: variational inference via "
                    f"{getattr(handle.factory, '__name__', handle.factory)!r} "
                    f"(kind=vi, num_samples={handle.vi_num_samples}, "
                    f"posterior_draws={handle.vi_posterior_draws})"
                )
                continue

            # Masked .bayesian() / .vi() — a single group entry tagged
            # with backend in {"bayesian", "vi"} (Phase 11 v1 supports
            # at most one non-optax group per model).
            if _non_optax_groups:
                if len(_non_optax_groups) > 1:
                    raise NotImplementedError(
                        f"Model (layer {lid}): multiple masked .bayesian()/.vi() groups "
                        "on the same model are not yet supported (Phase 11 v1).  "
                        "Use one masked group at a time."
                    )
                _g = _non_optax_groups[0]
                if _g["backend"] == "bayesian":
                    handle = jno_bayesian.build_kernel_handle(_g["bayesian_cfg"])
                else:
                    handle = jno_bayesian.build_vi_handle(_g["vi_cfg"])
                handle.param_mask = _g["mask"]
                bayesian_handles[k] = handle
                self.log.info(
                    f"Model {lid}: masked {_g['backend']} via "
                    f"{getattr(handle.factory, '__name__', handle.factory)!r} "
                    f"(restricted to .mask() scope)"
                )
                # If no optax-tagged groups and no global optimiser, the
                # complement of the mask is implicitly frozen — done.
                if not _optax_groups and fm._opt_fn is None:
                    continue
                # Otherwise fall through to the optax block (which uses
                # _optax_groups + the global _opt_fn for the "rest").

            if _optax_groups and fm._lora_config is None:
                # ── Per-group optimizer via chained optax.masked transforms ──
                # Build one masked transform per group, plus a "default" for
                # any trainable params not covered by an explicit group.
                global_opt_fn = fm._opt_fn
                global_lr = fm._lr if fm._lr is not None else LearningRateSchedule(1e-3)

                if global_opt_fn is None:
                    raise ValueError(
                        f"Model (layer {lid}) has parameter groups but no global optimizer. "
                        f"Call  model.optimizer(optax.adam)  as a fallback for ungrouped params."
                    )

                masked_transforms = []
                group_scheds = []

                # Align each user-supplied group mask to the *trainable* tree,
                # where frozen/static leaves are represented as None.
                group_masks_norm = []
                for g in _optax_groups:
                    gmask_norm = jax.tree_util.tree_map(
                        lambda p, m: bool(m) if p is not None else False,
                        trainable[lid],
                        g["mask"],
                        is_leaf=lambda x: x is None,
                    )
                    group_masks_norm.append(gmask_norm)

                # Diagnostics over group masks: per-group coverage + overlap + uncovered
                array_flags = [x is not None for x in jax.tree_util.tree_leaves(trainable[lid])]
                group_leaf_masks = [
                    [bool(x) if isinstance(x, bool) else False for x in jax.tree_util.tree_leaves(gm)]
                    for gm in group_masks_norm
                ]

                group_counts = []
                for g, gmask in zip(_optax_groups, group_leaf_masks):
                    count = sum(1 for m, is_arr in zip(gmask, array_flags) if is_arr and m)
                    group_counts.append((g["target"], count))
                    if count == 0:
                        self.log.warning(f"Model {lid}: parameter group target={g['target']!r} matched 0 parameters.")

                overlap_count = 0
                uncovered_count = 0
                for leaf_i, is_arr in enumerate(array_flags):
                    if not is_arr:
                        continue
                    n_hit = sum(1 for gmask in group_leaf_masks if leaf_i < len(gmask) and gmask[leaf_i])
                    if n_hit > 1:
                        overlap_count += 1
                    if n_hit == 0:
                        uncovered_count += 1

                if overlap_count > 0:
                    self.log.warning(
                        f"Model {lid}: parameter groups overlap on {overlap_count} array leaves. "
                        "Update order will follow optax.chain mask order."
                    )

                self.log.info(
                    f"Model {lid}: parameter groups summary — groups={len(_optax_groups)}, "
                    f"overlap={overlap_count}, uncovered_by_groups={uncovered_count}"
                )
                self.log.quiet(f"Parameter Group Diagnostic Report for model {lid}")
                self.log.quiet(f"groups={len(_optax_groups)}, overlap={overlap_count}, uncovered={uncovered_count}")
                for tgt, cnt in group_counts:
                    self.log.quiet(f"  target={tgt!r}: matched_arrays={cnt}")

                for g, gmask_norm in zip(_optax_groups, group_masks_norm):
                    g_opt = g["opt_fn"] or global_opt_fn
                    g_lr = g["lr"] if g["lr"] is not None else global_lr
                    chain = _build_opt_chain(g_opt, g_lr)
                    masked_transforms.append(optax.masked(chain, gmask_norm))
                    group_scheds.append(g_lr)

                # "default" group: negate all group masks to cover remaining params
                def _default_mask(params, _group_masks=group_masks_norm):
                    """True for leaves in no explicit group."""
                    combined = jax.tree_util.tree_map(lambda _: False, params)
                    for gmask in _group_masks:
                        combined = jax.tree_util.tree_map(
                            lambda c, m: c or (m if isinstance(m, bool) else False),
                            combined,
                            gmask,
                            is_leaf=lambda x: x is None,
                        )
                    return jax.tree_util.tree_map(lambda c: not c, combined, is_leaf=lambda x: x is None)

                default_chain = _build_opt_chain(global_opt_fn, global_lr)
                masked_transforms.append(optax.masked(default_chain, _default_mask(trainable[lid])))
                group_scheds.append(global_lr)

                per_model_opts[k] = optax.chain(*masked_transforms)
                group_lr_schedules[k] = group_scheds
                self.log.info(
                    f"Model {lid}: {len(_optax_groups)} parameter group(s) + default — using per-group optimizers"
                )
            else:
                # ── Single global optimizer (original behaviour) ──
                opt_fn = fm._opt_fn
                lr_sched = fm._lr if fm._lr is not None else LearningRateSchedule(1e-3)

                if opt_fn is None:
                    raise ValueError(
                        f"Model (layer {lid}) has no optimizer. "
                        f"Call model.optimizer(optax.adam, lr=...) before solve(), "
                        f"or freeze the model with model.freeze()."
                    )

                if callable(opt_fn) and not isinstance(opt_fn, optax.GradientTransformation):
                    try:
                        base_opt = opt_fn(1.0)
                    except TypeError:
                        base_opt = opt_fn
                else:
                    base_opt = opt_fn

                if not isinstance(base_opt, optax.GradientTransformation):
                    raise TypeError(f"Unsupported optimizer type for model {lid}: {type(base_opt)}")

                scale = optax.inject_hyperparams(optax.scale)(step_size=lr_sched(0, zeros))
                per_model_opts[k] = optax.chain(base_opt, scale)
                lr_schedules[k] = lr_sched

        # Validate that all Bayesian models in this solve share the same
        # num_chains — the joint K-axis is shared across chains so mixing
        # K values would be ambiguous (Metropolis-within-Gibbs needs one K).
        if bayesian_handles:
            _k_set = {h.num_chains for h in bayesian_handles.values()}
            if len(_k_set) > 1:
                raise ValueError(
                    f"All .bayesian() models in one solve() must share the same num_chains; got {sorted(_k_set)}."
                )
            _num_chains_global = next(iter(_k_set))
        else:
            _num_chains_global = 1

        # Initialise optimizer / kernel states and place on mesh.  We
        # iterate the union of optax and Bayesian keys; each model has
        # exactly one of the two backends.
        opt_states = {}
        for k in sorted({*per_model_opts.keys(), *bayesian_handles.keys()}):
            lid = int(k)
            if k in bayesian_handles:
                # Per-chain jitter is drawn from a key split off self.rng so
                # multi-chain runs get reproducible over-dispersion.
                self.rng, _bay_init_key = jax.random.split(self.rng)
                state = jno_bayesian.init_state(bayesian_handles[k], trainable[lid], _bay_init_key)
            else:
                state = per_model_opts[k].init(trainable[lid])
            # Copy every array leaf so that aliased buffers (e.g. from
            # L-BFGS zero-initialised history arrays that share the same
            # underlying allocation) become distinct.  Without this,
            # donate_argnums will fail with "Attempt to donate the same
            # buffer twice".
            # Then place on the mesh with P() so shardings are canonical
            # and match what the step function will produce.
            opt_states[k] = jax.tree_util.tree_map(
                lambda x: (
                    jax.device_put(jnp.copy(x), NamedSharding(self.mesh, P()))
                    if isinstance(x, (jnp.ndarray, jax.Array))
                    else x
                ),
                state,
            )

        self._log_constraint_shapes(batchsize, min_consecutive=min_consecutive)

        # ── 5b. Resume from checkpoint (optional) ──
        if self._resume_from is not None:
            try:
                import orbax.checkpoint as _ocp
            except ImportError as exc:
                raise ImportError(
                    "orbax-checkpoint is required for resume_from=. Install it with:  pip install orbax-checkpoint"
                ) from exc

            _ckpt_mgr = _ocp.CheckpointManager(
                os.path.abspath(self._resume_from),
                options=_ocp.CheckpointManagerOptions(read_only=True),
            )
            _ckpt_step = getattr(self, "_resume_step", None) or _ckpt_mgr.latest_step()
            if _ckpt_step is None:
                raise FileNotFoundError(f"No checkpoints found in {self._resume_from}")

            # Build the target tree matching the live partition/opt structure
            # so Orbax restores arrays into the correct Equinox pytree shape.
            _target_state = {
                "trainable": trainable,
                "opt_states": opt_states,
                "rng": self.rng,
            }
            _restored = _ckpt_mgr.restore(
                _ckpt_step,
                args=_ocp.args.Composite(
                    state=_ocp.args.StandardRestore(_target_state),
                    metadata=_ocp.args.JsonRestore(),
                ),
            )
            trainable = _restored.state["trainable"]
            opt_states = _restored.state["opt_states"]
            self.rng = _restored.state["rng"]

            _ckpt_meta = _restored.metadata
            if _ckpt_meta is not None and "epoch" in _ckpt_meta:
                self._total_epochs = int(_ckpt_meta["epoch"])
            self.log.info(f"Resumed from checkpoint {self._resume_from} at step {_ckpt_step}")
            _ckpt_mgr.close()
            self._resume_from = None  # only resume once

        # ── 6. Prepare data ──
        n_devices = len(self.devices)
        full_context = self.domain_data.context

        if offload_data:
            # Keep full dataset as numpy on host — only a mini-batch is
            # transferred to the device each step.
            host_context = {k: np.asarray(v) for k, v in full_context.items()}
            total_samples = _infer_total_samples(host_context)
            effective_batchsize = None  # data is already pre-sliced
            self.log.info(
                f"Data offloading enabled: {total_samples} total samples, streaming batches of {batchsize} from host"
            )
        else:
            # Replicate / shard full dataset on device (original behaviour)
            domain_data = DomainData(context=full_context, dimension=self.domain_data.dimension)
            domain_data = DomainData(
                context=self._replicate_for_devices(domain_data.context, n_devices),
                dimension=domain_data.dimension,
            )
            domain_data = DomainData(
                context=self._shard_data(domain_data.context),
                dimension=domain_data.dimension,
            )
            on_device_context = domain_data.context
            effective_batchsize = batchsize

        # ── 6b. Window adaptation for HMC-family Bayesian models ──
        # Runs once before the main loop using the current trainable + the
        # full domain context.  Replaces step_size + inverse_mass_matrix on
        # the matching handles, seeds opt_states with the adapted state,
        # and zeroes handle.warmup so the main loop collects from epoch 0.
        # Window adaptation runs against the full constraint set.  With
        # substeps, the Bayesian kernel sees only a substep-local subset,
        # so an adapter tuned to the full loss would mis-tune.  Force
        # users to set adapt=False (or skip substeps) in that case.
        _adapt_candidates = [(k, h) for k, h in bayesian_handles.items() if jno_bayesian.adapt_is_applicable(h)]
        if _use_substeps and _adapt_candidates:
            raise ValueError(
                "Combining substeps= with .bayesian(..., adapt=True) is not supported. "
                "When using substeps, set adapt=False on each Bayesian model — window "
                "adaptation would tune against the full loss, but the kernel in a "
                "substep sees only the substep's constraints."
            )
        if _adapt_candidates:
            _adapt_loss_fn = self._make_loss_fn(
                self.compiled_constraints_fn,
                effective_batchsize,
                frozen_arrays,
                static,
                checkpoint_gradients=checkpoint_gradients,
                min_consecutive=min_consecutive,
            )
            _adapt_ctx = on_device_context if not offload_data else self.domain_data.context
            # A fixed PRNG key keeps the logdensity deterministic across
            # adaptation steps — required for the HMC integrator's geometry.
            _adapt_key_for_loss = jax.random.PRNGKey(0)
            for _bk, _handle in _adapt_candidates:
                _lid = int(_bk)

                def _ld_fn(p, _lid=_lid, _h=_handle, _key=_adapt_key_for_loss):
                    full = {**trainable, _lid: p}
                    nll, _ = _adapt_loss_fn(full, _adapt_ctx, _key)
                    return -nll + _h.prior_fn(p)

                self.rng, _adapt_key = jax.random.split(self.rng)
                _adapt_out = jno_bayesian.run_window_adaptation(_handle, trainable[_lid], _ld_fn, _adapt_key)
                if _adapt_out is None:
                    continue
                _adapted_state, _adapted_kwargs = _adapt_out
                _handle.extra_kwargs = _adapted_kwargs
                _K_bay = _handle.num_chains
                if _K_bay == 1:
                    # K=1: keep the adapted state as-is (no K axis).
                    opt_states[_bk] = _adapted_state
                else:
                    # Multi-chain: broadcast adapted state across K
                    # chains (PyMC convention).
                    opt_states[_bk] = jax.tree_util.tree_map(
                        lambda x, _K=_K_bay: jnp.stack([x] * _K, axis=0) if isinstance(x, (jnp.ndarray, jax.Array)) else x,
                        _adapted_state,
                    )
                # main loop collects from epoch 0
                _handle.warmup = 0
                self.log.info(f"Model {_lid}: window adaptation done — step_size={_adapted_kwargs['step_size']:.4g}")

        # ── 7. Build JIT-compiled step function ──
        step_fn = self.make_step_fn(
            per_model_opts=per_model_opts,
            batchsize=effective_batchsize,
            frozen=frozen_arrays,
            static=static,
            lr_schedules=lr_schedules,
            group_lr_schedules=group_lr_schedules,
            checkpoint_gradients=checkpoint_gradients,
            min_consecutive=min_consecutive,
            bayesian_handles=bayesian_handles,
        )

        # Optionally amortise Python dispatch overhead by running multiple
        # gradient steps inside a single XLA program via fori_loop.
        # Only valid when context is fixed on-device (offload_data=False).
        if inner_steps > 1:
            if offload_data:
                self.log.warning("inner_steps > 1 is not compatible with offload_data=True; falling back to inner_steps=1")
                inner_steps = 1
            else:
                _K = inner_steps
                _single = step_fn

                def step_fn(trainable, opt_states, rng, context, start_epoch, prev_losses):
                    def body(i, carry):
                        tr, opt, rn, ep, _total, _indv = carry
                        tr, opt, rn, ep_next, total, indv = _single(tr, opt, rn, context, ep, _indv)
                        return tr, opt, rn, ep_next, total, indv

                    init = (
                        trainable,
                        opt_states,
                        rng,
                        start_epoch,
                        jnp.zeros(()),
                        prev_losses,
                    )
                    return jax.lax.fori_loop(0, _K, body, init)

        # ── 7b. Build gradient accumulation functions (if needed) ──
        _use_accumulation = accumulation_steps > 1
        if _use_accumulation:
            _grad_fn = self.make_grad_fn(
                batchsize=effective_batchsize,
                frozen=frozen_arrays,
                static=static,
                checkpoint_gradients=checkpoint_gradients,
                min_consecutive=min_consecutive,
            )
            _apply_fn = self.make_apply_fn(
                per_model_opts=per_model_opts,
                lr_schedules=lr_schedules,
                group_lr_schedules=group_lr_schedules,
            )
            self.log.info(
                f"Gradient accumulation enabled: {accumulation_steps} micro-batches "
                f"per update (effective batch = {batchsize} × {accumulation_steps} "
                f"= {batchsize * accumulation_steps})"
            )

        # Optional: build JIT-compiled tracker function
        has_trackers = len(self.compiled_trackers) > 0
        if has_trackers:
            track_fn = self._make_track_fn(
                self.compiled_trackers,
                effective_batchsize,
                frozen_arrays,
                static,
            )
            tracker_intervals = [intv for intv, _ in self.compiled_trackers]

        with self.mesh:
            # ── Derive input shardings from actual arrays ──
            # This tells jax.jit the canonical sharding for every input.
            # Without this, outputs from step N may carry different
            # sharding annotations than the original inputs, causing
            # an expensive recompilation at step N+1.
            def _leaf_sharding(x):
                if isinstance(x, jax.Array) and hasattr(x, "sharding"):
                    return x.sharding
                return None

            if offload_data:
                trace_context = {k: jnp.zeros((batchsize,) + tuple(v.shape[1:])) for k, v in host_context.items()}
                trace_context = self._shard_data(trace_context)
            else:
                trace_context = on_device_context

            # Canonical replicated sharding — must match what the step
            # function outputs, otherwise JAX will recompile.
            replicated = NamedSharding(self.mesh, P())

            # Place scalars on the mesh so their output sharding matches.
            self.rng = jax.device_put(self.rng, replicated)
            prev_losses = jax.device_put(jnp.zeros(self.n_constraints), replicated)

            in_shardings = (
                jax.tree_util.tree_map(_leaf_sharding, trainable),  # trainable
                jax.tree_util.tree_map(_leaf_sharding, opt_states),  # opt_states
                replicated,  # rng
                jax.tree_util.tree_map(_leaf_sharding, trace_context),  # context
                replicated,  # epoch (scalar)
                replicated,  # prev_losses
            )

            # Buffer donation: reuse trainable (0) and opt_states (1)
            # buffers in-place since the step returns updated versions.
            # rng (2) is also donated (small but correct).
            #
            # out_shardings mirrors the in_shardings for the three outputs
            # that are fed back as inputs (trainable, opt_states, rng), and
            # pins the remaining scalars to replicated.  Without this, JAX
            # returns outputs with SingleDeviceSharding which mismatches the
            # NamedSharding in in_shardings, triggering a device_put on every
            # call to fix the sharding before dispatch.
            out_shardings = (
                jax.tree_util.tree_map(_leaf_sharding, trainable),  # trainable
                jax.tree_util.tree_map(_leaf_sharding, opt_states),  # opt_states
                replicated,  # rng
                replicated,  # epoch (scalar)
                replicated,  # total_loss
                replicated,  # individual_losses  (→ prev_losses next step)
            )
            jit_step = jax.jit(
                step_fn,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                donate_argnums=(0, 1, 2),
            )

            # JIT-compile gradient accumulation functions when enabled.
            if _use_accumulation:
                _trainable_sharding = jax.tree_util.tree_map(_leaf_sharding, trainable)
                _ctx_sharding = jax.tree_util.tree_map(_leaf_sharding, trace_context)

                jit_grad = jax.jit(
                    _grad_fn,
                    in_shardings=(
                        _trainable_sharding,  # trainable (read-only)
                        replicated,  # rng
                        _ctx_sharding,  # context
                    ),
                    out_shardings=(
                        _trainable_sharding,  # grads (same tree as trainable)
                        replicated,  # rng
                        replicated,  # total_loss
                        replicated,  # individual_losses
                    ),
                )

                _opt_sharding = jax.tree_util.tree_map(_leaf_sharding, opt_states)
                jit_apply = jax.jit(
                    _apply_fn,
                    in_shardings=(
                        _trainable_sharding,  # trainable
                        _opt_sharding,  # opt_states
                        _trainable_sharding,  # grads
                        replicated,  # epoch
                        replicated,  # prev_losses
                    ),
                    out_shardings=(
                        _trainable_sharding,  # trainable
                        _opt_sharding,  # opt_states
                    ),
                    donate_argnums=(2,),  # donate grads (freshly accumulated)
                )

            if has_trackers:
                jit_track = jax.jit(track_fn)

            # ── 7b. Build per-substep JIT step functions ──
            if _use_substeps:
                _substep_jit_steps = []
                _substep_opt_states_list = []
                _substep_n_steps_list = []
                _substep_n_constraints_list = []

                for _si, (_indices, _n_steps_i) in enumerate(_parsed_substeps):
                    _sub_exprs = [self._constraint_exprs[i] for i in _indices]
                    _active_lids = _active_model_lids(_sub_exprs)

                    _per_model_opts_i = {k: v for k, v in per_model_opts.items() if int(k) in _active_lids}
                    _lr_schedules_i = {k: v for k, v in lr_schedules.items() if int(k) in _active_lids}
                    _group_lr_i = {k: v for k, v in group_lr_schedules.items() if int(k) in _active_lids}
                    # Per-substep Bayesian handles — only kernels for models
                    # whose params receive non-zero gradient from this
                    # substep's constraints participate in this substep.
                    _bayesian_handles_i = {k: v for k, v in bayesian_handles.items() if int(k) in _active_lids}

                    # Fresh optimizer / kernel states — isolated from other substeps
                    _opt_states_i: Dict[str, Any] = {}
                    for _k in sorted({*_per_model_opts_i.keys(), *_bayesian_handles_i.keys()}):
                        _lid_i = int(_k)
                        if _k in _bayesian_handles_i:
                            self.rng, _bay_init_key_i = jax.random.split(self.rng)
                            _state_i = jno_bayesian.init_state(_bayesian_handles_i[_k], trainable[_lid_i], _bay_init_key_i)
                        else:
                            _state_i = _per_model_opts_i[_k].init(trainable[_lid_i])
                        _opt_states_i[_k] = jax.tree_util.tree_map(
                            lambda x: (
                                jax.device_put(jnp.copy(x), NamedSharding(self.mesh, P()))
                                if isinstance(x, (jnp.ndarray, jax.Array))
                                else x
                            ),
                            _state_i,
                        )

                    _compiled_fn_i = TraceCompiler.compile_multi_expression(_sub_exprs, self.all_ops)
                    _n_constraints_i = len(_sub_exprs)

                    _step_fn_i = self.make_step_fn(
                        per_model_opts=_per_model_opts_i,
                        batchsize=effective_batchsize,
                        frozen=frozen_arrays,
                        static=static,
                        lr_schedules=_lr_schedules_i,
                        group_lr_schedules=_group_lr_i if _group_lr_i else None,
                        checkpoint_gradients=checkpoint_gradients,
                        min_consecutive=min_consecutive,
                        compiled_constraints_fn=_compiled_fn_i,
                        bayesian_handles=_bayesian_handles_i,
                    )

                    _zeros_i = jax.device_put(jnp.zeros(_n_constraints_i), replicated)
                    _in_shardings_i = (
                        jax.tree_util.tree_map(_leaf_sharding, trainable),
                        jax.tree_util.tree_map(_leaf_sharding, _opt_states_i),
                        replicated,
                        jax.tree_util.tree_map(_leaf_sharding, trace_context),
                        replicated,
                        replicated,
                    )
                    _out_shardings_i = (
                        jax.tree_util.tree_map(_leaf_sharding, trainable),
                        jax.tree_util.tree_map(_leaf_sharding, _opt_states_i),
                        replicated,
                        replicated,
                        replicated,
                        replicated,
                    )
                    _jit_step_i = jax.jit(
                        _step_fn_i,
                        in_shardings=_in_shardings_i,
                        out_shardings=_out_shardings_i,
                        donate_argnums=(0, 1, 2),
                    )

                    _substep_jit_steps.append(_jit_step_i)
                    _substep_opt_states_list.append(_opt_states_i)
                    _substep_n_steps_list.append(_n_steps_i)
                    _substep_n_constraints_list.append(_n_constraints_i)
                    self.log.info(
                        f"Substep {_si}: constraints={_indices}, n_steps={_n_steps_i}, active_models={sorted(_active_lids)}"
                    )

            self.log.info("JIT compiling step function with mesh sharding — this might take a while")

            # ── Enable persistent XLA compilation cache ──
            # On the first run XLA compiles and writes artifacts to disk.
            # Subsequent runs with the same network/batchsize/dtype reload
            # from disk and skip compilation entirely, saving minutes for
            # large models.  The cache is keyed on the full XLA program hash
            # so stale entries are never loaded for a different graph.
            _cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "jno", "xla_cache")
            os.makedirs(_cache_dir, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", _cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

            # Trigger AOT compilation so the first real step is fast.

            if _use_substeps:
                # AOT compile each substep's step function
                for _si, (_jit_step_i, _opt_states_i, _n_constraints_i) in enumerate(
                    zip(_substep_jit_steps, _substep_opt_states_list, _substep_n_constraints_list)
                ):
                    _zeros_i = jax.device_put(jnp.zeros(_n_constraints_i), replicated)
                    _ = _jit_step_i.lower(
                        trainable,
                        _opt_states_i,
                        self.rng,
                        trace_context,
                        jax.device_put(jnp.int32(0), replicated),
                        _zeros_i,
                    ).compile()
            elif _use_accumulation:
                # AOT compile grad and apply separately
                _ = jit_grad.lower(trainable, self.rng, trace_context).compile()
                _zero_grads = jax.tree_util.tree_map(jnp.zeros_like, trainable)
                _ = jit_apply.lower(
                    trainable,
                    opt_states,
                    _zero_grads,
                    jax.device_put(jnp.int32(0), replicated),
                    prev_losses,
                ).compile()
                del _zero_grads
            else:
                _ = jit_step.lower(
                    trainable,
                    opt_states,
                    self.rng,
                    trace_context,
                    jax.device_put(jnp.int32(0), replicated),
                    prev_losses,
                ).compile()

            # Pre-compile tracker JIT as well so profile windows focus on
            # steady-state train-step behavior instead of one-time compile work.
            if has_trackers:
                _ = jit_track.lower(trainable, trace_context, self.rng).compile()

            # Warmup: run a few real dispatches on throw-away copies so
            # the GPU's buffer allocator, CUDA kernel instruction cache,
            # and cuDNN workspaces are fully initialised before any
            # profiling starts. Without this the first 1-2 profiled steps
            # are anomalously slow, making the trace misleading.
            _tw = jax.tree_util.tree_map(jnp.copy, trainable)
            _rw = jnp.copy(self.rng)
            _ew = jax.device_put(jnp.int32(0), replicated)
            if _use_substeps:
                _substep_ow_list = [jax.tree_util.tree_map(jnp.copy, _os) for _os in _substep_opt_states_list]
                for _ in range(3):
                    for _si, (_jit_step_i, _n_constraints_i) in enumerate(
                        zip(_substep_jit_steps, _substep_n_constraints_list)
                    ):
                        _pl_i = jax.device_put(jnp.zeros(_n_constraints_i), replicated)
                        _tw, _substep_ow_list[_si], _rw, _ew, _, _ = _jit_step_i(
                            _tw, _substep_ow_list[_si], _rw, trace_context, _ew, _pl_i
                        )
                del _substep_ow_list
            else:
                _ow = jax.tree_util.tree_map(jnp.copy, opt_states)
                _pl = jax.device_put(jnp.zeros(self.n_constraints), replicated)
                if _use_accumulation:
                    for _ in range(3):
                        _gw, _rw, _, _ = jit_grad(_tw, _rw, trace_context)
                        _tw, _ow = jit_apply(_tw, _ow, _gw, _ew, _pl)
                    del _gw
                else:
                    for _ in range(3):
                        _tw, _ow, _rw, _ew, _, _pl = jit_step(_tw, _ow, _rw, trace_context, _ew, _pl)
                del _ow, _pl

            if has_trackers:
                _ = jit_track(_tw, trace_context, _rw)

            jax.effects_barrier()
            del _tw, _rw, _ew
            # self.log.info("Skipping AOT compile/warmup; first training step will JIT normally.")

            # Notify callbacks that setup is complete so they can build and
            # pre-compile their own JIT functions (e.g. explainability callbacks).
            if callbacks:
                for _cb in callbacks:
                    _cb.on_solve_begin(
                        compiled_constraints_fn=self.compiled_constraints_fn,
                        n_constraints=self.n_constraints,
                        batchsize=batchsize,
                        frozen=frozen_arrays,
                        static=static,
                        trainable=trainable,
                        context=trace_context,
                        rng=self.rng,
                        min_consecutive=min_consecutive,
                        constraint_exprs=self._user_constraint_exprs,
                        all_ops=self.all_ops,
                        domain=self.domain,
                    )

            # ── 8. Training loop ──

            print_rate = max(1, epochs // 10 if epochs < 100_000 else epochs // 1000)
            prev_losses = jax.device_put(jnp.zeros(self.n_constraints), replicated)

            # Substep state — one prev_losses array per substep + a global step counter
            if _use_substeps:
                _substep_prev_losses_list = [
                    jax.device_put(jnp.zeros(_n), replicated) for _n in _substep_n_constraints_list
                ]
                _global_substep_counter = 0

            # Log buffers
            log_epochs = []
            log_losses = []
            log_total_loss = []
            log_timestamps = []
            log_track_stats = []

            # Bayesian sample buffers — one per Bayesian model.  We append the
            # post-update position (a device array pytree) every `thin` outer
            # epochs after `warmup`, capped at `keep` samples.  Storing device
            # arrays keeps the loop async; the actual host transfer happens at
            # the end of solve() via jnp.stack + jax.device_get.
            _bayesian_buffers: Dict[str, list] = {k: [] for k in bayesian_handles}

            rng_np = np.random.default_rng(int(jax.device_get(self.rng[0])))
            st = time.time()
            epoch_jnp = jax.device_put(jnp.int32(0), replicated)

            n_outer = epochs // inner_steps
            if epochs % inner_steps != 0:
                self.log.warning(
                    f"epochs={epochs} is not divisible by inner_steps={inner_steps}; "
                    f"running {n_outer * inner_steps} epochs instead."
                )
            print_rate = max(10, n_outer // 10 if n_outer < 100_000 else n_outer // 1000)

            # Freeze all surviving Python objects (model params, opt states, etc.)
            # so Python's cyclic GC never has to scan them during the hot loop.
            # Without this, GC kicks in mid-step every ~700 allocations, adding
            # random multi-ms pauses visible as long unflatten spans in xprof.
            gc.disable()  # prevent cyclic GC from interrupting the hot loop;
            # JAX pytrees/dicts are acyclic so refcounting handles them correctly

            # Profile a short steady-state window: skip the very first outer
            # step (which can still include one-time runtime setup), then
            # capture a handful of outer steps to keep traces focused.
            _profile_skip_steps = 1 if profile else 0
            _profile_steps = min(50, max(0, n_outer - _profile_skip_steps)) if profile else 0
            _profile_start = _profile_skip_steps
            _profile_stop = _profile_start + _profile_steps
            _profile_active = False
            _profile_ctx: Any = nullcontext()

            # --- wandb: cache run reference and build model name map ---
            _wandb_run = get_wandb_run()
            _wandb_model_names: dict = {}
            if _wandb_run is not None:
                for _lid, _fm in flax_mods.items():
                    _k = str(_lid)
                    _wandb_model_names[_k] = _fm.name or type(_fm.module).__name__
                # Log config to wandb
                _wandb_run.config.update(
                    {
                        "epochs": epochs,
                        "inner_steps": inner_steps,
                        "n_constraints": self.n_constraints,
                        "n_trackers": len(self.compiled_trackers),
                        "trainable_params": n_trainable_params,
                        "frozen_params": n_frozen_params,
                        "total_params": n_total_params,
                        "seed": self.seed,
                    },
                    allow_val_change=True,
                )

            _wandb_nan_alerted = False

            # ── MCMC fastpath gate ──
            # Pure-Bayesian solves (no optax, no substeps, no streaming,
            # no inner_steps amortisation, no gradient accumulation, no
            # trackers, no resampling) take a scan-based fastpath: all
            # ``keep * thin`` post-warmup steps run inside a single
            # JIT-compiled XLA program per chunk of ``print_rate`` outer
            # iterations.  Closes three Bayesian perf gaps:
            #
            # 1. No outer ``value_and_grad`` (kernels do their own grads).
            # 2. ``lax.scan`` over chunked steps — one XLA dispatch per
            #    print_rate steps instead of per epoch.
            # 3. Samples stacked inside XLA — one host transfer per chunk.
            #
            # Falls through to the per-epoch Python loop otherwise.
            _fastpath_enabled = bool(
                bayesian_handles
                and not per_model_opts
                and not _use_substeps
                and inner_steps == 1
                and accumulation_steps == 1
                and not offload_data
                and not self.compiled_trackers
                and not (has_resampling and strategies)
            )
            if _fastpath_enabled and bayesian_handles:
                # All MCMC handles must share warmup/keep/thin so a
                # single scan length is well-defined.  VI handles are
                # excluded from this check — they don't have a
                # per-step "keep" concept; posterior_draws are taken
                # from the fitted q post-solve.
                _mcmc_handles_only = {k: h for k, h in bayesian_handles.items() if h.kind != "vi"}
                _shared = {(h.warmup, h.keep, h.thin) for h in _mcmc_handles_only.values()}
                if len(_shared) > 1:
                    _fastpath_enabled = False
                    self.log.info(
                        "MCMC fastpath disabled: Bayesian (MCMC) models have mixed "
                        "warmup/keep/thin; falling back to per-epoch loop."
                    )

            if _fastpath_enabled:
                # If any MCMC handles exist, use their shared (warmup,
                # keep, thin); otherwise (pure-VI solve) derive scan
                # length from ``epochs`` so VI gets ``epochs`` ELBO
                # optimisation steps without per-epoch dispatch.
                _mcmc_for_shape = [h for h in bayesian_handles.values() if h.kind != "vi"]
                if _mcmc_for_shape:
                    _ref_h = _mcmc_for_shape[0]
                    _fp_warmup, _fp_keep, _fp_thin = _ref_h.warmup, _ref_h.keep, _ref_h.thin
                else:
                    # Pure-VI: each outer iter = one ELBO step.
                    _fp_warmup, _fp_keep, _fp_thin = 0, n_outer, 1
                self.log.info(
                    f"MCMC fastpath: scan over {_fp_keep} samples × thin={_fp_thin} "
                    f"+ {_fp_warmup} warmup, chunked at print_rate={print_rate}."
                )

                # Chunk size in *outer* iterations (each = ``thin`` MCMC
                # steps).  print_rate gives the desired progress cadence;
                # divide by thin to keep the chunk's collected sample
                # count near print_rate.
                _chunk_keep = max(1, min(_fp_keep, max(1, print_rate // max(_fp_thin, 1))))
                _n_chunks = (_fp_keep + _chunk_keep - 1) // _chunk_keep
                # All but possibly the last chunk are `_chunk_keep` long.
                _last_chunk_keep = _fp_keep - (_n_chunks - 1) * _chunk_keep

                # Build (and JIT) the scan function.  Two compilations
                # at most — one for the steady chunk size, one for the
                # last (smaller) chunk.  Warmup folds into chunk 0.
                def _build_jit_scan(_chunk_warmup, _chunk_keep_arg):
                    _sf = self.make_mcmc_scan_fn(
                        bayesian_handles=bayesian_handles,
                        batchsize=effective_batchsize,
                        frozen=frozen_arrays,
                        static=static,
                        checkpoint_gradients=checkpoint_gradients,
                        min_consecutive=min_consecutive,
                        warmup=_chunk_warmup,
                        keep=_chunk_keep_arg,
                        thin=_fp_thin,
                    )
                    return jax.jit(_sf, donate_argnums=(0, 1, 2))

                # Loss eval for per-chunk wandb / progress (one forward
                # pass — no grad, no accumulation).
                _loss_fn_for_log = self._make_loss_fn(
                    self.compiled_constraints_fn,
                    effective_batchsize,
                    frozen_arrays,
                    static,
                    checkpoint_gradients=False,
                    min_consecutive=min_consecutive,
                )
                _jit_loss_eval = jax.jit(_loss_fn_for_log)

                # Context for the scan body — on-device tensor.
                _fp_ctx = on_device_context if not offload_data else self.domain_data.context

                # Profile the very first chunk only (fastpath chunks are
                # large so one chunk is a representative sample).
                if (not _profile_active) and _profile_steps > 0:
                    _profile_ctx = jax.profiler.trace(f"{self.log.path}/traces", create_perfetto_trace=True)
                    _profile_ctx.__enter__()
                    _profile_active = True

                _epoch_counter = 0
                for _chunk_idx in range(_n_chunks):
                    _this_keep = _chunk_keep if _chunk_idx < _n_chunks - 1 else _last_chunk_keep
                    # Warmup steps fold into chunk 0; subsequent chunks
                    # don't need to re-do warmup.
                    _this_warmup = _fp_warmup if _chunk_idx == 0 else 0
                    _jit_scan = _build_jit_scan(_this_warmup, _this_keep)

                    trainable, opt_states, self.rng, _samples_chunk = _jit_scan(trainable, opt_states, self.rng, _fp_ctx)

                    # Append the (this_keep, K, *param) chunk to each
                    # Bayesian model's buffer.  One device→host copy
                    # per chunk (vs one per step in slow path).
                    for _bk in bayesian_handles:
                        _bayesian_buffers[_bk].append(
                            jax.tree_util.tree_map(
                                lambda x: jnp.copy(x) if isinstance(x, (jnp.ndarray, jax.Array)) else x,
                                _samples_chunk[_bk],
                            )
                        )

                    # Advance epoch counter and emit a per-chunk loss /
                    # progress snapshot.
                    _epoch_counter += _this_warmup + _this_keep * _fp_thin
                    self.rng, _loss_key = jax.random.split(self.rng)
                    _total_loss_chunk, _individual_losses_chunk = _jit_loss_eval(trainable, _fp_ctx, _loss_key)
                    _total_np = float(_total_loss_chunk)
                    _ind_np = np.asarray(_individual_losses_chunk)

                    # Append to history buffers at chunk-boundary cadence.
                    log_epochs.append(_epoch_counter - 1)
                    log_total_loss.append(_total_np)
                    log_losses.append(_ind_np)
                    log_timestamps.append(time.time())

                    # Progress message.
                    _msg = " | ".join([f"C{ci}: {float(v):.4e}" for ci, v in enumerate(_ind_np)])
                    self.log.info(f"Epoch {_epoch_counter - 1:>6}/{epochs}| L:{_total_np:.4e} | {_msg}")

                    # Wandb (per chunk) — mirror the slow-path block:
                    # n_samples, n_chains, plus a running posterior mean
                    # for scalar parameters.
                    if _wandb_run is not None:
                        _wb = {"total_loss": _total_np, "epoch": _epoch_counter - 1}
                        for _ci, _cv in enumerate(_ind_np):
                            _wb[f"constraint_{_ci}"] = float(_cv)
                        for _bk_w, _h_w in bayesian_handles.items():
                            _fm_w = flax_mods.get(int(_bk_w))
                            if _fm_w is None:
                                continue
                            _name_w = _fm_w.name or f"model{_bk_w}"
                            _bufw = _bayesian_buffers[_bk_w]
                            # Each entry is shape ``(chunk_keep, *)`` — sum
                            # leading axis across chunks gives total
                            # samples collected so far.
                            _so_far = sum(int(jax.tree_util.tree_leaves(s)[0].shape[0]) for s in _bufw)
                            _wb[f"posterior/{_name_w}/n_samples"] = _so_far
                            _wb[f"posterior/{_name_w}/n_chains"] = _h_w.num_chains
                            if getattr(_fm_w, "_is_jno_scalar_parameter", False) and _bufw:
                                _stacked = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *_bufw)
                                _stacked_v = _stacked.value
                                _wb[f"posterior/{_name_w}/mean"] = float(jnp.mean(_stacked_v))
                        wandb_log(_wb, step=_epoch_counter - 1)
                        if (not _wandb_nan_alerted) and not np.isfinite(_total_np):
                            wandb_alert(
                                "NaN/Inf loss detected",
                                f"total_loss became {_total_np} at epoch {_epoch_counter - 1}",
                                level="ERROR",
                            )
                            _wandb_nan_alerted = True

                # Stop the profile window opened above.
                if _profile_active:
                    _profile_ctx.__exit__(None, None, None)
                    _profile_active = False
                    _profile_ctx = nullcontext()

            for outer_epoch in range(0 if _fastpath_enabled else n_outer):
                if (not _profile_active) and _profile_steps > 0 and outer_epoch == _profile_start:
                    _profile_ctx = jax.profiler.trace(f"{self.log.path}/traces", create_perfetto_trace=True)
                    _profile_ctx.__enter__()
                    _profile_active = True

                epoch = outer_epoch * inner_steps  # first epoch of this outer step

                # --- adaptive host-side resampling at outer-step boundaries ---
                if has_resampling and strategies:
                    due = [(tag, strat) for tag, strat in strategies.items() if strat.should_resample(epoch)]
                    if due:
                        full_models = eqx.combine(trainable, frozen_arrays, static)
                        full_models = _paramax.unwrap(full_models)

                        residuals_all = self.compiled_resample_constraints_fn(
                            full_models,
                            full_context,
                            batchsize=None,
                            key=self.rng,
                            min_consecutive=min_consecutive,
                        )

                        updated = False
                        for tag, strategy in due:
                            tag_points = full_context.get(tag, None)
                            if tag_points is None:
                                self.log.warning(f"Resampling skipped for tag '{tag}': tag not found in context")
                                continue

                            if not hasattr(tag_points, "ndim") or tag_points.ndim != 4:
                                self.log.warning(
                                    f"Resampling skipped for tag '{tag}': expected 4-D point array (B, T, N, D), "
                                    f"got shape {tuple(getattr(tag_points, 'shape', '?'))}"
                                )
                                continue

                            # Spatial slice: coordinates are identical across timesteps.
                            T = tag_points.shape[1]
                            points_bn = jnp.asarray(tag_points[:, 0, :, :])
                            n_batch, n_points = points_bn.shape[0], points_bn.shape[1]

                            idxs = tag_to_constraint_indices.get(tag, [])
                            if not idxs:
                                self.log.warning(
                                    f"Resampling skipped for tag '{tag}': no constraints associated with this tag"
                                )
                                continue

                            scored = []
                            for idx in idxs:
                                collapsed = _collapse_residual_for_tag(residuals_all[idx], n_points, n_batch)
                                if collapsed is not None:
                                    scored.append(collapsed)

                            if not scored:
                                self.log.warning(f"Resampling skipped for tag '{tag}': no compatible pointwise residuals")
                                continue

                            # Normalize each constraint to [0, 1] then take per-point max.
                            stacked = jnp.stack(scored, axis=0)  # (C, B, N)
                            per_max = jnp.max(stacked, axis=-1, keepdims=True)  # (C, B, 1)
                            normalized = stacked / (per_max + 1e-12)
                            combined = jnp.max(normalized, axis=0)  # (B, N)

                            # Draw candidates once — reused by every batch and for normals.
                            candidates_pts, candidates_nrms = self.domain.draw_candidates(tag)

                            new_batches = []
                            for b in range(n_batch):
                                self.rng, rs_key = jax.random.split(self.rng)
                                b_key = jax.random.fold_in(rs_key, b)
                                new_batches.append(
                                    strategy.resample(
                                        points_bn[b],
                                        combined[b],
                                        self.domain,
                                        tag,
                                        epoch,
                                        b_key,
                                        candidates=candidates_pts,
                                    )
                                )

                            new_points_bn = jnp.stack(new_batches, axis=0)
                            full_context[tag] = jnp.tile(new_points_bn[:, None, :, :], (1, T, 1, 1))
                            self.domain.context[tag] = np.asarray(full_context[tag])

                            # Update normals atomically when the candidate pool provides them.
                            normal_tag = f"n_{tag}"
                            if normal_tag in full_context and candidates_pts is not None and candidates_nrms is not None:
                                cand_pts_j = jnp.array(candidates_pts)  # (N_pool, D)
                                cand_nrm_j = jnp.array(candidates_nrms)  # (N_pool, D)
                                new_nrm_batches = []
                                for b in range(n_batch):
                                    # Each new point is an exact row from the pool → argmin recovers its index.
                                    diffs = new_points_bn[b, :, None, :] - cand_pts_j[None, :, :]
                                    nearest = jnp.argmin(jnp.sum(diffs**2, axis=-1), axis=-1)
                                    new_nrm_batches.append(cand_nrm_j[nearest])
                                new_nrms_bn = jnp.stack(new_nrm_batches, axis=0)  # (B, N, D)
                                full_context[normal_tag] = jnp.tile(new_nrms_bn[:, None, :, :], (1, T, 1, 1))
                                self.domain.context[normal_tag] = np.asarray(full_context[normal_tag])
                            strategy.update_epoch(epoch)
                            self.log.info(f"Resampled {tag} points (epoch {epoch + 1})")
                            updated = True

                        if updated:
                            # Keep canonical domain_data in sync with updated points.
                            self.domain_data = self.prepare_domain_data(self.domain)
                            full_context = self.domain_data.context

                            host_context_new, on_device_context_new, total_samples = _rebuild_runtime_contexts(
                                full_context,
                                offload_data,
                                n_devices,
                                total_samples if offload_data else 0,
                            )
                            if offload_data:
                                host_context = host_context_new
                            else:
                                on_device_context = on_device_context_new

                # --- prepare context and step ---
                if _use_substeps:
                    # Each substep: own compiled function, own optimizer state.
                    # `trainable` is shared and passes between substeps so a
                    # later substep sees the freshly updated params from earlier ones.
                    if offload_data:
                        if host_context is None:
                            raise RuntimeError("offload_data=True but host_context is not available")
                        indices = rng_np.choice(total_samples, batchsize, replace=False)
                        batch_np = {
                            k: (
                                v
                                if k == "__time__"
                                else (np.broadcast_to(v, (batchsize,) + v.shape[1:]) if v.shape[0] == 1 else v[indices])
                            )
                            for k, v in host_context.items()
                        }
                        context = self._shard_data(jax.device_put(batch_np))
                    else:
                        context = on_device_context

                    _last_total = None
                    _last_indiv = None
                    for _si, _jit_step_i in enumerate(_substep_jit_steps):
                        _n_steps_i = _substep_n_steps_list[_si]
                        for _ in range(_n_steps_i):
                            (
                                trainable,
                                _substep_opt_states_list[_si],
                                self.rng,
                                _,
                                _last_total,
                                _last_indiv,
                            ) = _jit_step_i(
                                trainable,
                                _substep_opt_states_list[_si],
                                self.rng,
                                context,
                                jax.device_put(jnp.int32(_global_substep_counter), replicated),
                                _substep_prev_losses_list[_si],
                            )
                            _substep_prev_losses_list[_si] = _last_indiv
                            _global_substep_counter += 1

                    # Representative metrics from the final substep
                    total_loss = _last_total
                    individual_losses = _last_indiv
                    epoch_jnp = epoch_jnp + jnp.asarray(1, dtype=epoch_jnp.dtype)
                elif _use_accumulation:
                    # Gradient accumulation: N micro-batch forward/backward
                    # passes, then one averaged optimizer update.
                    _acc_grads = None
                    _acc_total = 0.0
                    _acc_losses = jax.device_put(jnp.zeros(self.n_constraints), replicated)
                    _inv_accum = 1.0 / accumulation_steps

                    for _ai in range(accumulation_steps):
                        # Each micro-batch gets a fresh random sample
                        if offload_data:
                            if host_context is None:
                                raise RuntimeError("offload_data=True but host_context is not available")
                            indices = rng_np.choice(total_samples, batchsize, replace=False)
                            batch_np = {
                                k: (
                                    v
                                    if k == "__time__"
                                    else (np.broadcast_to(v, (batchsize,) + v.shape[1:]) if v.shape[0] == 1 else v[indices])
                                )
                                for k, v in host_context.items()
                            }
                            micro_ctx = self._shard_data(jax.device_put(batch_np))
                        else:
                            micro_ctx = on_device_context

                        _micro_grads, self.rng, _micro_loss, _micro_indiv = jit_grad(
                            trainable,
                            self.rng,
                            micro_ctx,
                        )

                        if _acc_grads is None:
                            _acc_grads = jax.tree_util.tree_map(lambda g: g * _inv_accum, _micro_grads)
                        else:
                            _acc_grads = jax.tree_util.tree_map(
                                lambda a, g: a + g * _inv_accum,
                                _acc_grads,
                                _micro_grads,
                            )
                        _acc_total = _acc_total + float(jax.device_get(_micro_loss)) * _inv_accum
                        _acc_losses = _acc_losses + _micro_indiv * _inv_accum

                    # Single optimizer update with averaged gradients
                    trainable, opt_states = jit_apply(
                        trainable,
                        opt_states,
                        _acc_grads,
                        epoch_jnp,
                        prev_losses,
                    )
                    total_loss = jax.device_put(jnp.float32(_acc_total), replicated)
                    individual_losses = _acc_losses
                    epoch_jnp = epoch_jnp + jnp.asarray(1, dtype=epoch_jnp.dtype)
                    context = micro_ctx  # keep last micro-batch for tracker evaluation
                else:
                    if offload_data:
                        if host_context is None:
                            raise RuntimeError("offload_data=True but host_context is not available")
                        indices = rng_np.choice(total_samples, batchsize, replace=False)
                        batch_np = {
                            k: (
                                v
                                if k == "__time__"
                                else (np.broadcast_to(v, (batchsize,) + v.shape[1:]) if v.shape[0] == 1 else v[indices])
                            )
                            for k, v in host_context.items()
                        }
                        context = self._shard_data(jax.device_put(batch_np))
                    else:
                        context = on_device_context

                    # --- step ---
                    (
                        trainable,
                        opt_states,
                        self.rng,
                        epoch_jnp,
                        total_loss,
                        individual_losses,
                    ) = jit_step(
                        trainable,
                        opt_states,
                        self.rng,
                        context,
                        epoch_jnp,
                        prev_losses,
                    )

                if not _use_substeps:
                    prev_losses = individual_losses

                # --- collect Bayesian samples (post-warmup, thinned, capped at keep) ---
                # The per-chain (K-leading) position lives on ``opt_states[_bk]``
                # — either the state itself (SG-MCMC) or ``.position``
                # (HMC-family). ``trainable[_lid]`` only carries a chain-0
                # representative, so we read from opt_states here.
                if bayesian_handles:
                    for _bk, _handle in bayesian_handles.items():
                        # VI handles skip per-step collection — proper
                        # posterior samples are drawn from the fitted
                        # variational state in the post-solve block.
                        if _handle.kind == "vi":
                            continue
                        _idx = outer_epoch
                        if _idx < _handle.warmup:
                            continue
                        _post = _idx - _handle.warmup
                        if _post % _handle.thin != 0:
                            continue
                        _buf = _bayesian_buffers[_bk]
                        if len(_buf) >= _handle.keep:
                            continue
                        _lid_int = int(_bk)
                        if _handle.param_mask is not None:
                            # Masked Bayesian (Phase 11): the kernel state
                            # holds only the masked subset; the full
                            # reassembled position lives in trainable[lid]
                            # (updated post-step inside the jit by
                            # ``jno_bayesian.step``).
                            if _handle.num_chains != 1:
                                raise NotImplementedError(
                                    "Masked .bayesian()/.vi() with num_chains>1 is not "
                                    "supported in Phase 11 v1. Use num_chains=1 with .mask()."
                                )
                            _pos_K = trainable[_lid_int]
                        else:
                            _state_for_sample = opt_states[_bk]
                            _pos_K = _state_for_sample if _handle.kind == "grad_estimator" else _state_for_sample.position
                        # Detach from buffer donation: a jnp.copy of every
                        # array leaf gives us our own buffers that survive
                        # the next jit_step's donate_argnums.  We add a
                        # leading length-1 chunk axis so the slow-path
                        # ``(1, *param)`` entries concatenate uniformly
                        # with the MCMC-fastpath ``(chunk_keep, *param)``
                        # entries at flush time.
                        _sample = jax.tree_util.tree_map(
                            lambda x: jnp.copy(x)[None, ...] if isinstance(x, (jnp.ndarray, jax.Array)) else x,
                            _pos_K,
                        )
                        _buf.append(_sample)

                # Stop profiling after the requested steady-state window.
                if _profile_active and outer_epoch + 1 >= _profile_stop:
                    _profile_ctx.__exit__(None, None, None)
                    _profile_active = False
                    _profile_ctx = nullcontext()

                # --- wandb: log every epoch ---
                displayed_epoch = epoch + inner_steps - 1
                if _wandb_run is not None:
                    _wb_losses, _wb_total = jax.device_get((individual_losses, total_loss))
                    _wb_metrics: dict = {
                        "total_loss": float(_wb_total),
                        "epoch": displayed_epoch,
                    }
                    for _ci, _cl in enumerate(np.asarray(_wb_losses)):
                        _wb_metrics[f"constraint_{_ci}"] = float(_cl)
                    # Learning rates (one per model)
                    for _wk in sorted(opt_states.keys()):
                        _wst = opt_states[_wk]
                        try:
                            _lr = float(jax.device_get(_wst[-1].hyperparams["step_size"]))
                        except (IndexError, KeyError, AttributeError):
                            try:
                                _lr = float(jax.device_get(_wst[0].inner_state[-1].hyperparams["step_size"]))
                            except (IndexError, KeyError, AttributeError, TypeError):
                                _lr = None
                        if _lr is not None:
                            _model_name = _wandb_model_names.get(_wk, _wk)
                            _wb_metrics[f"lr/{_model_name}"] = _lr
                    # Per-Bayesian-model chain stats: running posterior
                    # mean over all (K, N) collected draws so far.  For
                    # multi-chain runs the per-chain "last sample" doesn't
                    # have a single meaning, so we drop the legacy /last
                    # field and instead expose /n_chains so dashboards can
                    # group panels by chain.  Acceptance probability would
                    # need plumbing the kernel info from the step; deferred.
                    for _bk, _handle in bayesian_handles.items():
                        _lid_int = int(_bk)
                        _fm = flax_mods.get(_lid_int)
                        if _fm is None:
                            continue
                        _name = _fm.name or f"model{_lid_int}"
                        _buf = _bayesian_buffers.get(_bk, [])
                        if not _buf:
                            continue
                        # Stack the buffer once for the running mean — cheap
                        # because the buffer holds device arrays and we only
                        # hit this at print_rate cadence.  Each entry is
                        # already (K, *param); stack gives (N, K, *param).
                        _stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *_buf)
                        _wb_metrics[f"posterior/{_name}/n_samples"] = len(_buf)
                        _wb_metrics[f"posterior/{_name}/n_chains"] = _handle.num_chains
                        if getattr(_fm, "_is_jno_scalar_parameter", False):
                            _stacked_v = _stacked.value  # (N, K, *param_shape)
                            _wb_metrics[f"posterior/{_name}/mean"] = float(jnp.mean(_stacked_v))

                    wandb_log(_wb_metrics, step=displayed_epoch)

                    # NaN / Inf alert (only fire once)
                    if not _wandb_nan_alerted and not np.isfinite(_wb_total):
                        wandb_alert(
                            "NaN/Inf loss detected",
                            f"total_loss became {_wb_total} at epoch {displayed_epoch}",
                            level="ERROR",
                        )
                        _wandb_nan_alerted = True

                # --- logging: sync only at print interval ---
                should_print = (outer_epoch % print_rate == 0) or (outer_epoch == n_outer - 1)
                if should_print:
                    losses_np, total_np_arr = jax.device_get((individual_losses, total_loss))
                    losses_np = np.asarray(losses_np)
                    total_np = float(total_np_arr)

                    log_epochs.append(displayed_epoch)
                    log_losses.append(losses_np)
                    log_total_loss.append(total_np)
                    log_timestamps.append(time.time())

                    # Trackers
                    track_stats_np = None
                    if has_trackers and any(outer_epoch % (max(1, intv // inner_steps)) == 0 for intv in tracker_intervals):
                        track_vals = jit_track(trainable, context, self.rng)
                        track_stats_np = [float(v) for v in jax.device_get(track_vals)]
                        log_track_stats.append(track_stats_np)
                        # Log trackers to wandb
                        if _wandb_run is not None:
                            _wb_track = {}
                            for _ti, _tv in enumerate(track_stats_np):
                                _wb_track[f"tracker_{_ti}"] = _tv
                            wandb_log(_wb_track, step=displayed_epoch)

                    # Progress line
                    loss_strs = " | ".join(f"C{i}: {v:>10.4e}" for i, v in enumerate(losses_np))
                    if track_stats_np is not None:
                        track_strs = " | ".join(f"T{i}: {v:>10.4e}" for i, v in enumerate(track_stats_np))
                        self.log.info(
                            f"Epoch {displayed_epoch:>6}/{epochs}| L:{total_np:>10.4e} | {loss_strs} | {track_strs}"
                        )
                    else:
                        self.log.info(f"Epoch {displayed_epoch:>6}/{epochs}| L:{total_np:>10.4e} | {loss_strs}")

                # --- callbacks ---
                if callbacks:
                    cb_info = {
                        "epoch": epoch + inner_steps - 1,
                        "trainable": trainable,
                        "opt_states": opt_states,
                        "rng": self.rng,
                        "total_loss": total_loss,
                        "individual_losses": individual_losses,
                        "log": self.log,
                        "context": context,
                    }
                    _stop_requested = False
                    for cb in callbacks:
                        if cb.on_epoch_end(**cb_info):
                            _stop_requested = True
                    if _stop_requested:
                        break

            if _profile_active:
                _profile_ctx.__exit__(None, None, None)

            et = time.time()

            gc.enable()  # restore GC after training loop

            # ── 9. Reconstruct models ──
            trained_models = eqx.combine(trainable, frozen_arrays, static)

            # Merge LoRA if requested
            for lid, fm in flax_mods.items():
                if fm._lora_config is not None:
                    trained_models[lid] = _merge_lora(trained_models[lid])
                    self.log.info(f"LoRA merged for model {lid}")

            self.models = trained_models

            # ── 9b. Sync Model.module refs with trained weights ──
            # The expression tree (self.constraints / self.all_ops) holds
            # Model objects whose .module still points to the
            # *pre-training* arrays.  Buffer donation deletes those
            # arrays, so pickling the expression tree would fail.
            # Update every Model to point at the trained model.
            for lid, fm in flax_mods.items():
                fm.module = trained_models[lid]

            # ── 9c. Flush Bayesian sample buffers ──
            # Each buffer entry is a module-shaped pytree with a leading
            # *chunk* axis: ``(1, *param)`` for slow-path single-step
            # collection, ``(chunk_keep, *param)`` (K=1) or
            # ``(chunk_keep, K, *param)`` (K>1) for the MCMC fastpath.
            # Concatenating along axis 0 unifies both cases into
            # ``(N, *param)`` (K=1) or ``(N, K, *param)`` (K>1); the K
            # axis is then either added (K=1) or swapped to front (K>1)
            # to produce the canonical arviz ``(chain, draw, *)`` layout.
            for _bk, _buf in _bayesian_buffers.items():
                _lid_int = int(_bk)
                _fm = flax_mods.get(_lid_int)
                if _fm is None:
                    continue
                if len(_buf) == 0:
                    _fm._posterior_samples_pytree = None
                    continue
                _K_flush = bayesian_handles[_bk].num_chains
                # Step 1: concatenate all chunks along axis 0.
                joined = jax.tree_util.tree_map(
                    lambda *xs: jnp.concatenate(xs, axis=0),
                    *_buf,
                )
                # Step 2: reshape to the user-facing ``(K, N, *param)``.
                if _K_flush == 1:
                    _fm._posterior_samples_pytree = jax.tree_util.tree_map(
                        lambda x: x[None, ...],
                        joined,
                    )
                else:
                    _fm._posterior_samples_pytree = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(x, 0, 1),
                        joined,
                    )

            # ── 9d. Variational posterior draws ──
            # For each VI handle, draw ``vi_posterior_draws`` i.i.d. samples
            # from the fitted variational distribution and overwrite the
            # per-step buffer with these proper posterior samples.  Layout
            # ``(1, posterior_draws, *param)`` matches the MCMC posterior
            # shape so downstream code (crux.eval, rhat, ess) is uniform.
            for _bk, _handle in bayesian_handles.items():
                if _handle.kind != "vi":
                    continue
                _lid_int = int(_bk)
                _fm = flax_mods.get(_lid_int)
                if _fm is None:
                    continue
                self.rng, _vi_draw_key = jax.random.split(self.rng)
                _samples = jno_bayesian.vi_sample(_handle, opt_states[_bk], _vi_draw_key, _handle.vi_posterior_draws)
                # _samples is a pytree with leading axis = posterior_draws.
                # For masked VI: ``_samples`` only covers the masked
                # subset; reassemble each draw with the unmasked
                # snapshot (constant across draws).
                if _handle.param_mask is not None:
                    _unmasked_snap = eqx.filter(trainable[_lid_int], _handle.param_mask, inverse=True)
                    _samples = jax.vmap(lambda p: eqx.combine(p, _unmasked_snap))(_samples)
                # Add the canonical K-axis at position 0.
                _fm._posterior_samples_pytree = jax.tree_util.tree_map(
                    lambda x: jnp.copy(x)[None, ...] if isinstance(x, (jnp.ndarray, jax.Array)) else x,
                    _samples,
                )

            # ── 10. Build log dict ──
            logs = {
                "epoch": np.array(log_epochs),
                "total_loss": np.array(log_total_loss),
                "losses": np.stack(log_losses) if log_losses else np.array([]),
                "timestamps": np.array(log_timestamps),
                "training_time": et - st,
                "trainable_params": n_trainable_params,
                "frozen_params": n_frozen_params,
                "total_params": n_total_params,
                "lora_params": n_lora_params_total,
            }
            if log_track_stats:
                logs["track_stats"] = np.array(log_track_stats)
            self.training_logs.append(logs)
            _t = int(logs["training_time"])
            self.log.info(f"Training took {_t // 3600}h {(_t % 3600) // 60}m {_t % 60}s")

            # --- wandb: log training summary ---
            if _wandb_run is not None:
                _wandb_run.summary.update(
                    {
                        "training_time": logs["training_time"],
                        "final_total_loss": float(log_total_loss[-1]) if log_total_loss else None,
                    }
                )
                wandb_log_model(self)

        self._total_epochs += epochs

        # --- callbacks: on_training_end ---
        if callbacks:
            for cb in callbacks:
                cb.on_training_end()

        return statistics(self.training_logs)

    def restore_checkpoint(self, directory: str, step: Optional[int] = None):
        """Restore model parameters and optimizer state from an Orbax checkpoint.

        Reads checkpoint metadata (epoch counter) immediately and schedules
        a full weight restore for the next :meth:`solve` call.  The actual
        array restore is deferred because Orbax needs the live Equinox /
        Optax tree structures as a target, and those are only available
        inside ``solve()`` after the three-way partition and optimizer
        initialisation.

        Args:
            directory: Path to the checkpoint directory written by
                :class:`~jno.utils.callbacks.CheckpointCallback`.
            step: Checkpoint step to restore.  ``None`` (default)
                restores the latest available checkpoint.

        Returns:
            self, for chaining.
        """
        try:
            import orbax.checkpoint as ocp
        except ImportError as exc:
            raise ImportError(
                "orbax-checkpoint is required for restore_checkpoint(). Install it with:  pip install orbax-checkpoint"
            ) from exc

        manager = ocp.CheckpointManager(
            os.path.abspath(directory),
            options=ocp.CheckpointManagerOptions(read_only=True),
        )
        if step is None:
            step = manager.latest_step()
        if step is None:
            raise FileNotFoundError(f"No checkpoints found in {directory}")

        # Read only metadata so we can set the epoch counter now.
        restored = manager.restore(
            step,
            args=ocp.args.Composite(
                metadata=ocp.args.JsonRestore(),
            ),
        )
        metadata = restored.metadata

        if metadata is not None and "epoch" in metadata:
            self._total_epochs = int(metadata["epoch"])

        self.log.info(f"Restored checkpoint from {directory} at step {step}")
        manager.close()

        # Defer actual weight / opt-state restore to solve(), where the
        # correct Equinox and Optax tree structures are available.
        self._resume_from = os.path.abspath(directory)
        self._resume_step = step
        return self

    @property
    def _unwrapped_models(self):
        """Models with all paramax wrappers resolved — safe for inference / shape queries."""
        return _paramax.unwrap(self.models)

    def _log_constraint_shapes(self, batchsize, min_consecutive: Optional[int] = 1):
        """Log the output shape of each constraint by doing a test evaluation.

        When the log level is DEBUG, prints a full shape-annotated tree
        for each constraint so users can see how shapes evolve through
        every node of the expression.
        """

        # Create dummy inputs for shape inference
        test_rng = jax.random.PRNGKey(0)

        # Use jax.eval_shape to get output shape without computation
        _models = self._unwrapped_models
        out_shape = jax.eval_shape(
            lambda: self.compiled_constraints_fn(
                _models,
                self.domain_data.context,
                batchsize=batchsize,
                key=test_rng,
                min_consecutive=min_consecutive,
            )
        )

        # For each constraint, also get the shape *before* the final
        # reduction (e.g. .mse) so the log shows the residual geometry.
        # Constraints are stored as OperationDef(inner_expr); unwrap first.
        constraint_exprs = getattr(self, "_constraint_exprs", [])

        def _unwrap(expr):
            """Unwrap OperationDef to get the inner expression."""
            inner = expr.expr if isinstance(expr, OperationDef) else expr
            if isinstance(inner, FunctionCall):
                return inner.args[0], inner._name
            return inner, None

        parent_exprs = [_unwrap(expr) for expr in constraint_exprs]
        # Only compile the parent layer if at least one expr has a parent
        if any(name is not None for _, name in parent_exprs):
            parent_fn = TraceCompiler.compile_multi_expression([e for e, _ in parent_exprs], self.all_ops)
            parent_shape = jax.eval_shape(
                lambda: parent_fn(
                    _models,
                    self.domain_data.context,
                    batchsize=batchsize,
                    key=test_rng,
                    min_consecutive=min_consecutive,
                )
            )
        else:
            parent_shape = [None] * len(out_shape)

        for i, (const, (_, op_name)) in enumerate(zip(out_shape, parent_exprs)):
            p_shape = parent_shape[i]
            if op_name is not None and p_shape is not None:
                self.log.info(f"Constraint {i}: Shape = {p_shape.shape} → .{op_name}() → {const.shape}")
            else:
                self.log.info(f"Constraint {i}: Shape = {const.shape}")

        for i, (_, fn) in enumerate(self.compiled_trackers):
            # Use jax.eval_shape to get output shape without computation
            out_shape = jax.eval_shape(
                lambda: fn(
                    _models,
                    self.domain_data.context,
                    batchsize=batchsize,
                    key=test_rng,
                    min_consecutive=min_consecutive,
                )
            )

            # Also get the pre-reduction shape for tracker expressions
            tracker_exprs = getattr(self, "_tracker_exprs", [])
            tracker_expr = tracker_exprs[i] if i < len(tracker_exprs) else None
            # Trackers may also be wrapped in OperationDef
            if tracker_expr is not None and isinstance(tracker_expr, OperationDef):
                tracker_expr = tracker_expr.expr
            if tracker_expr is not None and isinstance(tracker_expr, FunctionCall):
                t_parent_fn = TraceCompiler.compile_multi_expression([tracker_expr.args[0]], self.all_ops)
                t_parent_shape = jax.eval_shape(
                    lambda: t_parent_fn(
                        _models,
                        self.domain_data.context,
                        batchsize=batchsize,
                        key=test_rng,
                        min_consecutive=min_consecutive,
                    )
                )
                t_shape = t_parent_shape[0]
                op_name = tracker_expr._name
            else:
                t_shape = None
                op_name = None

            if not isinstance(out_shape, tuple):
                if t_shape is not None:
                    self.log.info(f"Tracker {i}: Shape = {t_shape.shape} → .{op_name}() → {out_shape.shape}")
                else:
                    self.log.info(f"Tracker {i}: Shape = {out_shape.shape}")
            else:
                self.log.info(f"Tracker {i}: {out_shape}")

        # === Detailed shape trace (logged at DEBUG level) ===
        is_enabled_for = getattr(self.log, "isEnabledFor", None)
        if callable(is_enabled_for) and bool(is_enabled_for(10)):
            self._log_shape_traces(min_consecutive=min_consecutive)

        return None

    def _build_shape_context(self, min_consecutive: Optional[int] = 1) -> dict:
        """Build a single-sample runtime context for shape tracing.

        The compiled expression uses ``vmap(B) → scan(T) → eval(...)``.
        This method strips B and keeps a temporal window of size
        ``min_consecutive`` (clamped to available T), or full T when
        ``min_consecutive`` is ``None``, so shape tracing mirrors
        what the evaluator receives at runtime.

        Returns a plain dict mapping tag → array.
        """
        t_total = 1
        if "__time__" in self.domain_data.context:
            _t_arr = jnp.asarray(self.domain_data.context["__time__"])
            if _t_arr.ndim >= 1:
                t_total = int(_t_arr.shape[0])
        if min_consecutive is None:
            w_global = t_total
        else:
            w_global = max(1, min(int(min_consecutive), t_total))

        ctx_single = {}
        for tag, arr in self.domain_data.context.items():
            arr = jnp.asarray(arr)
            if tag == "__time__":
                # (T, 1) → (W, 1) or scalar-step (1,)
                w = max(1, min(w_global, int(arr.shape[0]) if arr.ndim >= 1 else 1))
                ctx_single[tag] = arr[:w] if w > 1 else arr[0]
            elif arr.ndim >= 3:
                # (B, T, ...) → strip batch and keep a W-step temporal window.
                # Covers (B,T,N,D), (B,T,H,W,C), (B,T,1,H,W,C), etc.
                t_steps = int(arr.shape[1])
                # If tensor stores only one step (e.g. initial condition),
                # mirror runtime behavior and broadcast it to the global window.
                if t_steps == 1 and w_global > 1:
                    ctx_single[tag] = jnp.broadcast_to(arr[0, 0], (w_global, *arr.shape[2:]))
                else:
                    w = max(1, min(w_global, t_steps))
                    ctx_single[tag] = arr[0, :w] if w > 1 else arr[0, 0]
            elif arr.ndim == 2:
                # (B, F) parametric → (F,)
                ctx_single[tag] = arr[0]
            else:
                ctx_single[tag] = arr
        return ctx_single

    def _log_shape_traces(self, min_consecutive: Optional[int] = 1):
        """Emit per-node shape trees for constraints and trackers.

        Called automatically when log level is DEBUG, or on demand via
        ``core.print_shapes()``.
        """
        ctx_single = self._build_shape_context(min_consecutive=min_consecutive)
        evaluator = TraceEvaluator(self._unwrapped_models)

        all_exprs = getattr(self, "_constraint_exprs", [])
        all_tracker_exprs = getattr(self, "_tracker_exprs", [])

        for i, expr in enumerate(all_exprs):
            try:
                tree = evaluator.trace_shapes(expr, ctx_single, key=jax.random.PRNGKey(0))
                self.log.debug(f"Constraint {i} shape trace:\n{tree}")
            except Exception as exc:
                self.log.debug(f"Constraint {i} shape trace failed: {exc}")

        for i, expr in enumerate(all_tracker_exprs):
            try:
                tree = evaluator.trace_shapes(expr, ctx_single, key=jax.random.PRNGKey(0))
                self.log.debug(f"Tracker {i} shape trace:\n{tree}")
            except Exception as exc:
                self.log.debug(f"Tracker {i} shape trace failed: {exc}")

    def print_shapes(self, min_consecutive: Optional[int] = 1):
        """Print shape-annotated expression trees to stdout.

        Can be called any time after ``compile()`` or ``solve()`` has
        run.  Useful for troubleshooting shape mismatches::

            crux = jno.core([pde.mse, ini.mse], domain)
            crux.print_shapes()
        """
        ctx_single = self._build_shape_context(min_consecutive=min_consecutive)
        evaluator = TraceEvaluator(self._unwrapped_models)

        all_exprs = getattr(self, "_constraint_exprs", [])
        all_tracker_exprs = getattr(self, "_tracker_exprs", [])

        for i, expr in enumerate(all_exprs):
            try:
                tree = evaluator.trace_shapes(expr, ctx_single, key=jax.random.PRNGKey(0))
                self.log.info(f"═══ Constraint {i} ═══")
                self.log.info(tree)
                self.log.info("")
            except Exception as exc:
                self.log.info(f"═══ Constraint {i} ═══  FAILED: {exc}")

        for i, expr in enumerate(all_tracker_exprs):
            try:
                tree = evaluator.trace_shapes(expr, ctx_single, key=jax.random.PRNGKey(0))
                self.log.info(f"═══ Tracker {i} ═══")
                self.log.info(tree)
                self.log.info("")
            except Exception as exc:
                self.log.info(f"═══ Tracker {i} ═══  FAILED: {exc}")

        return self

    def sweep(
        self,
        space: ArchSpace,
        optimizer: Union[str, type],
        budget: int,
        devices: Union[None, int, str, List[int], DeviceConfig] = None,
    ):
        """Run architecture and hyperparameter search with optional parallelism.

        Args:
            space: ArchSpace defining the search space (architecture + training params)
            optimizer: Nevergrad optimizer name (e.g., "NGOpt", "OnePlusOne", "CMA"),
                      class, or None for exhaustive grid search
            budget: Number of configurations to try (ignored for grid search)
            devices: Device specification for parallel execution:
                - None: auto-detect and use all available devices
                - int: use this many devices
                - str: device type ("gpu", "cpu", "tpu")
                - List[int]: specific device indices to use
                - DeviceConfig: explicit device configuration

        Returns:
            Training statistics from the best configuration
        """
        tuner = Tuner(self)
        stats = tuner.sweep(space, optimizer, budget, devices)
        return stats

    def _find_tunable_modules(self):
        """Collect unique TunableModule instances from constraints/trackers."""
        modules = []
        seen = set()

        def visit(node):
            if isinstance(node, TunableModule):
                if id(node) not in seen:
                    seen.add(id(node))
                    modules.append(node)
            elif isinstance(node, TunableModuleCall):
                tm = node.model
                if id(tm) not in seen:
                    seen.add(id(tm))
                    modules.append(tm)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, Choice):
                for opt in node.options:
                    if isinstance(opt, Placeholder):
                        visit(opt)
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
            elif isinstance(node, (Hessian, Jacobian)):
                visit(node.target)
                for v in node.variables:
                    if isinstance(v, Placeholder):
                        visit(v)
            elif isinstance(node, Tracker):
                visit(node.expr)

        for expr in getattr(self, "_constraint_exprs", []):
            visit(expr)
        for expr in getattr(self, "_tracker_exprs", []):
            visit(expr)

        return modules

    def _find_choice_nodes(self):
        """Collect unique Choice nodes from constraints/trackers."""
        choices = []
        seen = set()

        def visit(node):
            if isinstance(node, Choice):
                if id(node) not in seen:
                    seen.add(id(node))
                    choices.append(node)
                for opt in node.options:
                    if isinstance(opt, Placeholder):
                        visit(opt)
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
            elif isinstance(node, OperationDef):
                visit(node.expr)
            elif isinstance(node, OperationCall):
                visit(node.operation.expr)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, (Hessian, Jacobian)):
                visit(node.target)
                for v in node.variables:
                    if isinstance(v, Placeholder):
                        visit(v)
            elif isinstance(node, Tracker):
                visit(node.expr)

        for expr in getattr(self, "_constraint_exprs", []):
            visit(expr)
        for expr in getattr(self, "_tracker_exprs", []):
            visit(expr)

        return choices

    def eval(
        self,
        operation: Union[List[BinaryOp], BinaryOp],
        domain: Optional[domain] = None,
        min_consecutive: Optional[int] = 1,
        key=None,
        samples: str = "auto",
    ):
        """Evaluate an operation (or list of operations) on the current models.

        Args:
            operation: Expression(s) to evaluate.
            domain:    Override the stored domain.
            min_consecutive: Consecutive-time-step window for time-dependent
                expressions.
            key: Optional PRNG key for stochastic ops.
            samples: How to handle Bayesian models in the dependency graph:

                * ``"auto"`` (default) — per expression: if any model it
                  depends on has ``posterior_samples`` set, ``vmap`` the
                  evaluator over the chain (output shape
                  ``(n_samples, *original_shape)``); otherwise evaluate at
                  the point value.
                * ``"chain"`` — force chain evaluation; raises if no Bayesian
                  model appears in any expression's dependency graph.
                * ``"point"`` — force point evaluation for every expression
                  (last sample for Bayesian models, trained value for optax
                  models).  Use for a quick look without paying the vmap cost.

                The default flips to chain automatically because a single
                last-sample evaluation of a nonlinear function of Bayesian
                weights is, in general, *not* a meaningful summary of the
                posterior (``f(mean(θ)) ≠ mean(f(θ))``).
        """

        if isinstance(operation, Placeholder):
            operation = [operation]

        if samples not in ("auto", "chain", "point"):
            raise ValueError(f"crux.eval(samples=...) expects 'auto' | 'chain' | 'point', got {samples!r}.")

        domain_data = self.domain_data if domain is None else self.prepare_domain_data(domain)
        _models = self._unwrapped_models
        ctx = domain_data.context

        def _point_eval(op):
            fn = TraceCompiler.compile_traced_expression(op, self.all_ops)
            return fn(_models, ctx, batchsize=None, key=key, min_consecutive=min_consecutive)

        if samples == "point":
            results = [_point_eval(op) for op in operation]
            return results[0] if len(results) == 1 else results

        # samples == "auto" or "chain": walk each expression to discover its
        # Bayesian dependencies, then chain-eval or point-eval per expression.
        def _collect_posterior_lids(op) -> Dict[int, Any]:
            posterior_by_lid: Dict[int, Any] = {}

            def _record(m):
                raw = getattr(m, "_posterior_samples_pytree", None)
                if raw is not None:
                    posterior_by_lid[m.layer_id] = raw

            def _walk(node, _seen):
                if id(node) in _seen:
                    return
                _seen.add(id(node))
                if isinstance(node, Model):
                    _record(node)
                    return
                if isinstance(node, ModelCall):
                    _record(node.model)
                    for a in node.args:
                        if isinstance(a, Placeholder):
                            _walk(a, _seen)
                    return
                if isinstance(node, BinaryOp):
                    _walk(node.left, _seen)
                    _walk(node.right, _seen)
                    return
                if isinstance(node, FunctionCall):
                    for a in node.args:
                        if isinstance(a, Placeholder):
                            _walk(a, _seen)
                    return
                if isinstance(node, OperationCall):
                    _walk(node.operation.expr, _seen)
                    for a in node.args:
                        if isinstance(a, Placeholder):
                            _walk(a, _seen)
                    return
                if isinstance(node, OperationDef):
                    _walk(node.expr, _seen)
                    return

            _walk(op, set())
            return posterior_by_lid

        per_op_chains = [_collect_posterior_lids(op) for op in operation]
        any_bayesian = any(per_op_chains)

        if samples == "chain" and not any_bayesian:
            raise ValueError(
                "samples='chain' was requested but no models in the given expression(s) "
                "carry posterior_samples. Configure one or more parameters with "
                ".bayesian(...) and run crux.solve() before calling eval(samples='chain')."
            )

        def _chain_eval(op, posterior_by_lid):
            chain_part, static_part = jno_bayesian.chain_params_for_eval(_models, posterior_by_lid)
            fn = TraceCompiler.compile_traced_expression(op, self.all_ops)

            def _one(_chain_p, _fn=fn):
                params = {**static_part, **_chain_p}
                return _fn(params, ctx, batchsize=None, key=key, min_consecutive=min_consecutive)

            # ``chain_part`` carries the arviz-shaped (K, N, *param) leading
            # axes from solve(); nested vmap pushes the K outer axis and the
            # N inner axis through the evaluator so the result lands as
            # ``(K, N, *expr_shape)``.
            return jax.vmap(jax.vmap(_one))(chain_part)

        results = []
        for op, posterior_by_lid in zip(operation, per_op_chains):
            if posterior_by_lid or (samples == "chain"):
                results.append(_chain_eval(op, posterior_by_lid))
            else:
                results.append(_point_eval(op))

        return results[0] if len(results) == 1 else results

    def __getstate__(self):
        """Prepare state for pickling - remove unpicklable objects."""
        state = self.__dict__.copy()
        state["_mesh_shape"] = tuple(self.mesh.shape.values())
        state["devices"] = None
        state["mesh"] = None
        state["data_sharding"] = None
        state["param_sharding"] = None

        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)

        # Restore mesh and sharding
        mesh_shape = state.get("_mesh_shape")
        self._setup_parallelism(mesh_shape)
