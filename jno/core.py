from typing import List, Callable, Dict, Optional, Tuple, Union, Any
import os
import gc
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import optax
import cloudpickle
import numpy as np
import time
from .trace import (
    Placeholder,
    Variable,
    TensorTag,
    BinaryOp,
    Model,
    TunableModule,
    TunableModuleCall,
    ModelCall,
    OperationDef,
    OperationCall,
    Hessian,
    Jacobian,
    FunctionCall,
    Literal,
    Constant,
    ConstantNamespace,
    Tracker,
    collect_operations,
    collect_tags,
    get_primary_tag,
    dump_tree,
    cse,
)
from .utils import LearningRateSchedule, WeightSchedule, statistics, get_logger, get_seed
from .utils.monitor import HardwareMonitor
from .domain import domain, DomainData
from .trace_evaluator import TraceEvaluator
from .trace_compiler import TraceCompiler
from .tuner import ArchSpace, DeviceConfig, Tuner
from .architectures.lora_linear import (
    apply_lora as _apply_lora,
    merge_lora as _merge_lora,
    lora_trainable_filter as _lora_trainable_filter,
)
import equinox as eqx

try:
    import paramax as _paramax
except Exception:  # pragma: no cover - optional dependency
    _paramax = None


class core:
    """core solver using traced operations."""

    def __init__(
        self,
        constraints: List[Placeholder],
        domain: domain,
        mesh: Optional[Tuple[int, ...]] = (1, 1),
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
        """
        self.log = get_logger()
        self.constraints: List[Placeholder] = constraints

        self.domain = domain
        self.models: Dict[int, Any] = {}  # full equinox models (pytrees with arrays + static)
        self._trained_ops: Dict[int, Any] = {}
        self.training_logs: List[Dict[str, jnp.ndarray]] = []
        self.dots: List = []
        self.checkpoints: List[Dict] = []
        self.all_ops: List[OperationDef] = []

        super().__init__()

        self._total_epochs = 0
        seed_cfg = get_seed()
        seed = int(seed_cfg) if seed_cfg is not None else 21
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.log.info(f"RNG seed: {seed}")

        self.log.info(f"Initializing Model/s and compiling constraints")

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
        wrapped = []
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
        """
        node = expr
        while isinstance(node, FunctionCall) and getattr(node, "reduces_axis", False) and len(node.args) == 1:
            node = node.args[0]
        return node

    def compute_tensor_dims(self, domain) -> Dict[str, Tuple]:
        """Compute input dimensions for each context entry."""
        tensor_dims = {}
        if hasattr(domain, "context"):
            for name, tensor in domain.context.items():
                tensor_dims[name] = tensor.shape[1:]
        return tensor_dims

    def prepare_domain_data(self, domain) -> DomainData:
        """Convert domain data to JAX arrays for training."""
        if domain is None:
            raise ValueError("domain required")

        context = {}
        if hasattr(domain, "context"):
            for tag, arr in domain.context.items():
                arr = jnp.asarray(arr)
                # Ensure batch dimension exists
                if arr.ndim >= 2:
                    context[tag] = arr
                else:
                    context[tag] = arr[None, ...]

        return DomainData(
            context=context,
            dimension=domain.dimension,
        )

    # Training
    def _make_loss_fn(self, compiled_constraints_fn, n_constraints, batchsize, frozen, static, checkpoint_gradients=False, min_consecutive=1):
        """Create loss function — evaluates ALL constraints in one combined call."""

        def loss_fn(trainable, context, rng):
            full_models = eqx.combine(trainable, frozen, static)
            if _paramax is not None:
                # Always unwrap Paramax wrappers before model evaluation.
                full_models = _paramax.unwrap(full_models)

            if checkpoint_gradients:
                _fn, _bs = compiled_constraints_fn, batchsize

                # Equinox wrapper avoids JAX export false-positives in type stubs.
                @eqx.filter_checkpoint
                def _remat_eval(models, ctx, key):
                    return _fn(models, ctx, batchsize=_bs, key=key, min_consecutive=min_consecutive)

                all_residuals = _remat_eval(full_models, context, rng)
            else:
                # One call → one JAX function → XLA applies CSE across constraints
                all_residuals = compiled_constraints_fn(full_models, context, batchsize=batchsize, key=rng, min_consecutive=min_consecutive)

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
            if _paramax is not None:
                # Keep tracker evaluation consistent with training forward path.
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
    ):
        """Build a single JIT-compiled training step.

        Returns a function with signature::

            step(trainable, opt_states, rng, context, epoch, prev_losses)
                -> (trainable, opt_states, rng, total_loss, individual_losses)

        The training loop is a plain Python ``for`` loop which:
        * enables buffer donation at every step boundary,
        * allows host-resident data to be streamed per step,
        * makes progress logging trivial (no ``io_callback``).

        Args:
            per_model_opts: ``{layer_id_str: optax_chain}`` per-model optimizers.
            lr_schedules:   ``{layer_id_str: LearningRateSchedule}``.
            checkpoint_gradients: Wrap constraint evaluations in ``jax.checkpoint``.
        """
        loss_fn = self._make_loss_fn(
            self.compiled_constraints_fn,  # combined fn (replaces list)
            self.n_constraints,
            batchsize,
            frozen,
            static,
            checkpoint_gradients=checkpoint_gradients,
            min_consecutive=min_consecutive,
        )

        lid_keys = sorted(per_model_opts.keys())  # deterministic order
        base_epoch = self._total_epochs
        _group_lr = group_lr_schedules or {}  # {k: [(mask, sched), ..., (None, global_sched)]}

        def step(trainable, opt_states, rng, context, start_epoch, prev_losses):
            rng, step_rng = jax.random.split(rng)

            def loss_wrapper(p):
                return loss_fn(p, context, step_rng)

            (total_loss, individual_losses), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(trainable)

            # ── per-model optimizer step ──
            for k in lid_keys:
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
                        new_state[i].inner_state[-1].hyperparams["step_size"] = jnp.asarray(lr_val, dtype=new_state[i].inner_state[-1].hyperparams["step_size"].dtype)
                else:
                    lr_val = lr_schedules[k](base_epoch + start_epoch, individual_losses)
                    new_state[-1].hyperparams["step_size"] = jnp.asarray(lr_val, dtype=opt_states[k][-1].hyperparams["step_size"].dtype)

                trainable = {**trainable, lid: optax.apply_updates(model_params, updates)}
                opt_states = {**opt_states, k: new_state}

            return trainable, opt_states, rng, total_loss, individual_losses

        return step

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
        constraints = self.wrap_constraints(self.constraints)

        # === Collect operations and tags ===
        self.all_ops = self.collect_unique_operations(constraints)

        # === CSE: deduplicate shared sub-expressions ===
        constraints = [cse(c) for c in constraints]
        self.all_ops = self.collect_unique_operations(constraints)

        # === Prepare domain data ===
        self.domain_data = self.prepare_domain_data(self.domain)
        tensor_dims = self.compute_tensor_dims(self.domain)

        # === Initialize models ===
        self.models, self.rng = TraceCompiler.init_layer_params(self.all_ops, self.domain_data.dimension, tensor_dims, self.rng, self.log)

        # === Apply sharding to model arrays ===
        self.models = self._shard_params(self.models)

        # === Compile constraints and trackers ===
        self.compiled_trackers = []
        self._constraint_exprs = []  # raw expressions for shape tracing
        self._tracker_exprs = []
        constraint_exprs = []

        for expr in constraints:
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

        # Compile all normal constraints in ONE combined function so XLA
        # can apply CSE across shared sub-expressions.
        self.compiled_constraints_fn = TraceCompiler.compile_multi_expression(constraint_exprs, self.all_ops)
        self.n_constraints = len(constraint_exprs)

        # Keep tag metadata and a pointwise residual function for adaptive
        # resampling. The normal training loss still uses reduced constraints
        # in ``self.compiled_constraints_fn``.
        self._constraint_tags = self.get_constraint_tags(self._constraint_exprs)
        self._resample_exprs = [self._strip_reduction_for_resampling(expr) for expr in self._constraint_exprs]
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
        min_consecutive: int = 1,
        profile: bool = False,
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

        Returns:
            statistics: Training history with ``.plot()`` convenience.
        """
        from contextlib import nullcontext
        from jax._src import profiler as _jax_profiler

        _profiling = _jax_profiler._profile_state.profile_session is not None
        _trace = jax.profiler.TraceAnnotation if _profiling else lambda name, **_: nullcontext()

        batchsize = batchsize if batchsize is not None else self.domain.total_samples

        # Adaptive resampling metadata
        strategies = getattr(self.domain, "_resampling_strategies", {})
        has_resampling = bool(strategies)
        if has_resampling and inner_steps > 1:
            self.log.warning("Adaptive resampling with inner_steps > 1 is applied at outer-step boundaries only.")

        constraint_tags = getattr(self, "_constraint_tags", self.get_constraint_tags(getattr(self, "_constraint_exprs", [])))
        tag_to_constraint_indices: Dict[str, List[int]] = {}
        for i, tag in enumerate(constraint_tags):
            tag_to_constraint_indices.setdefault(tag, []).append(i)

        def _infer_total_samples(ctx: Dict[str, np.ndarray]) -> int:
            candidates = [v.shape[0] for k, v in ctx.items() if k != "__time__" and hasattr(v, "shape") and len(v.shape) >= 1]
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
            self.log.warning("offload_data requires batchsize < total_samples; " "ignoring offload_data for this run.")
            offload_data = False

        if _paramax is not None:
            self.log.info("Paramax auto-unwrap enabled: wrappers are unwrapped before each forward evaluation")

        # ── 1. Collect Model metadata ──
        flax_mods = self._collect_flax_modules()  # {layer_id: Model}

        # Validate: every non-frozen model must have an optimizer.
        # A frozen model that has LoRA active is also "effectively trainable"
        # (LoRA overrides freeze) and therefore also needs an optimizer.
        for lid, fm in flax_mods.items():
            needs_optimizer = (not fm._frozen) or (fm._lora_config is not None)
            if needs_optimizer and fm._opt_fn is None:
                raise ValueError(f"Model '{fm.name or type(fm.module).__name__}' (layer {lid}) " f"has no optimizer. Call  model.optimizer(optax.adam, lr=...)  " f"before solve(), or freeze it with  model.freeze().")

        # ── 1b. Mask diagnostics (always logged; warns on empty matches) ──
        for lid, fm in flax_mods.items():
            meta = getattr(fm, "_mask_meta", None)
            if meta is None:
                continue
            target = meta.get("target")
            matched = int(meta.get("matched", 0))
            total_arrays = int(meta.get("total_arrays", 0))
            sample_paths = meta.get("sample_paths", [])

            self.log.info(f"Model {lid}: mask(target={target!r}) matched {matched}/{total_arrays} array leaves")
            self.log.quiet(f"Mask Diagnostic Report for model {lid}")
            self.log.quiet(f"target={target!r}")
            self.log.quiet(f"matched={matched} / total_arrays={total_arrays}")
            if sample_paths:
                self.log.quiet("sample matched paths:")
                for p in sample_paths:
                    self.log.quiet(f"  {p}")

            if matched == 0:
                self.log.warning(f"Model {lid}: mask(target={target!r}) matched 0 parameters. " "No parameters will be selected by this target.")

        # ── 2. Apply LoRA transforms ──
        models = dict(self.models)
        lora_param_counts: Dict[int, Any] = {}  # Track LoRA params per model for logging
        for lid, fm in flax_mods.items():
            if fm._lora_config is not None:
                rank, alpha, lora_target = fm._lora_config
                self.rng, key = jax.random.split(self.rng)
                model_before = models[lid]
                models[lid] = _apply_lora(models[lid], rank, alpha, key=key, target=(lora_target if lora_target is not None else ""))
                model_after = models[lid]

                # ── LoRA diagnostic logging ──
                from .architectures.lora_linear import LoRALinear, FlaxLoRAWrapper

                n_arrays_before = sum(1 for l in jax.tree_util.tree_leaves(model_before) if eqx.is_array(l))
                n_arrays_after = sum(1 for l in jax.tree_util.tree_leaves(model_after) if eqx.is_array(l))
                n_params_before = sum(l.size for l in jax.tree_util.tree_leaves(model_before) if eqx.is_array(l))
                n_params_after = sum(l.size for l in jax.tree_util.tree_leaves(model_after) if eqx.is_array(l))
                n_lora_params = n_params_after - n_params_before
                lora_param_counts[lid] = n_lora_params

                if isinstance(model_after, FlaxLoRAWrapper):
                    # Count LoRA layers from the Flax lora_params dict
                    n_lora_layers = sum(1 for l in jax.tree_util.tree_leaves(model_after.lora_params) if eqx.is_array(l)) // 2  # each layer has lora_a + lora_b
                    self.log.info(f"LoRA (Flax) applied to model {lid} (rank={rank}, alpha={alpha}): " f"{n_lora_layers} kernel layers adapted, " f"{n_lora_params:,} new LoRA params")
                else:
                    from .architectures.linear import Linear as JNOLinear

                    is_lora = lambda x: isinstance(x, LoRALinear)
                    lora_leaves_after = [l for l in jax.tree_util.tree_leaves(model_after, is_leaf=is_lora) if isinstance(l, LoRALinear)]
                    n_lora_layers = len(lora_leaves_after)
                    self.log.info(f"LoRA applied to model {lid} (rank={rank}, alpha={alpha}): " f"{n_lora_layers} LoRALinear layers, " f"Params: {n_params_before:,}→{n_params_after:,}")
                    if n_lora_layers == 0:
                        self.log.warning(f"LoRA: No layers were adapted for model {lid}! " f"LoRA has NO EFFECT on this model.")

                # Write detailed LoRA diagnostics to log file (file-only, no console noise)
                # self.log.quiet(f"LoRA Diagnostic Report for model {lid}")
                # self.log.quiet(f"{'='*60}")
                # self.log.quiet(f"Requested: rank={rank}, alpha={alpha}")
                # self.log.quiet(f"Model type: {type(model_before).__name__}")
                # self.log.quiet(f"Wrapper type: {type(model_after).__name__}")
                # self.log.quiet(f"LoRA layers adapted:        {n_lora_layers}")
                # self.log.quiet(f"Total arrays before LoRA:   {n_arrays_before}")
                # self.log.quiet(f"Total arrays after LoRA:    {n_arrays_after}")
                # self.log.quiet(f"Total params before LoRA:   {n_params_before:,}")
                # self.log.quiet(f"Total params after LoRA:    {n_params_after:,}")
                # self.log.quiet(f"New LoRA params:            {n_lora_params:,}")
                # if isinstance(model_after, FlaxLoRAWrapper):
                #    self.log.quiet("Adapted kernels (lora_a / lora_b shapes):")
                #    flat, _ = jax.tree_util.tree_flatten_with_path(model_after.lora_params)
                #    for path, leaf in flat:
                #        if eqx.is_array(leaf):
                #            self.log.quiet(f"  {'/'.join(str(k) for k in path)}: {leaf.shape} {leaf.dtype}")
                # else:
                #    self.log.quiet("Full pytree paths (after LoRA):")
                #    flat, _ = jax.tree_util.tree_flatten_with_path(model_after)
                #    for path, leaf in flat:
                #        if eqx.is_array(leaf):
                #            self.log.quiet(f"  {'/'.join(str(k) for k in path)}: {leaf.shape} {leaf.dtype}")

        # ── 3. Build trainable filter ──
        filter_spec = {}
        for lid, model in models.items():
            fm = flax_mods.get(lid)
            if fm is not None and fm._lora_config is not None:
                # LoRA modes:
                # 1) fm._frozen=True  -> freeze all base params, train LoRA only
                # 2) fm._param_mask   -> freeze only masked target params in base,
                #                        train non-target base + LoRA params
                # 3) otherwise        -> default LoRA behaviour (freeze all base)
                if fm._frozen:
                    filter_spec[lid] = _lora_trainable_filter(model)
                elif fm._param_mask is not None:
                    filter_spec[lid] = _lora_trainable_filter(
                        model,
                        base_param_mask=fm._param_mask,
                        freeze_base=False,
                    )
                else:
                    filter_spec[lid] = _lora_trainable_filter(model)
            elif fm is not None and fm._frozen:
                # Whole model frozen – no arrays trainable
                filter_spec[lid] = jax.tree_util.tree_map(lambda l: False, model)
            elif fm is not None and fm._param_mask is not None:
                # Partial mask — only leaves marked True in the mask are trained.
                # Non-array leaves (e.g. activation functions kept as module
                # attributes) are always False so equinox does not misinterpret
                # them as sub-filter callables.
                filter_spec[lid] = jax.tree_util.tree_map(
                    lambda arr, m: bool(m) if eqx.is_array(arr) else False,
                    model,
                    fm._param_mask,
                )
            else:
                # Normal – every array trainable, non-arrays (e.g. activation
                # functions stored as attributes) must be False, not the
                # original value — equinox interprets callables in the
                # filter spec as sub-filters.
                filter_spec[lid] = jax.tree_util.tree_map(lambda l: True if eqx.is_array(l) else False, model)

        # ── 4. Three-way partition ──
        trainable, rest = eqx.partition(models, filter_spec)
        frozen_arrays, static = eqx.partition(rest, eqx.is_array)

        # ── 4b. Log parameter counts ──
        def _count_params(pytree):
            """Count total parameters in a pytree."""
            return sum(l.size for l in jax.tree_util.tree_leaves(pytree) if eqx.is_array(l))

        n_trainable_params = _count_params(trainable)
        n_frozen_params = _count_params(frozen_arrays)
        n_total_params = n_trainable_params + n_frozen_params
        n_lora_params_total = sum(lora_param_counts.values())

        self.log.info(f"Parameter summary:")
        self.log.info(f"    Trainable parameters:  {n_trainable_params:>12,}")
        self.log.info(f"    Frozen parameters:     {n_frozen_params:>12,}")
        self.log.info(f"    Total parameters:      {n_total_params:>12,}")
        if n_lora_params_total > 0:
            self.log.info(f"    LoRA parameters:       {n_lora_params_total:>12,} (included in trainable)")
            self.log.info(f"    LoRA % of total:       {100.0 * n_lora_params_total / n_total_params:>11.2f}%")

        # Shard trainable params
        trainable = self._shard_params(trainable)

        # ── 5. Build per-model optimizers ──
        per_model_opts = {}  # {str(lid): optax chain}
        lr_schedules: Dict[str, Any] = {}  # {str(lid): LearningRateSchedule} — global only
        group_lr_schedules: Dict[str, Any] = {}  # {str(lid): [sched_per_masked_group]} — when groups present
        zeros = jnp.zeros(self.n_constraints)

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

            if fm._param_groups and fm._lora_config is None:
                # ── Per-group optimizer via chained optax.masked transforms ──
                # Build one masked transform per group, plus a "default" for
                # any trainable params not covered by an explicit group.
                global_opt_fn = fm._opt_fn
                global_lr = fm._lr if fm._lr is not None else LearningRateSchedule(1e-3)

                if global_opt_fn is None:
                    raise ValueError(f"Model (layer {lid}) has parameter groups but no global optimizer. " f"Call  model.optimizer(optax.adam)  as a fallback for ungrouped params.")

                masked_transforms = []
                group_scheds = []

                # Diagnostics over group masks: per-group coverage + overlap + uncovered
                array_flags = [eqx.is_array(x) for x in jax.tree_util.tree_leaves(models[lid])]
                group_leaf_masks = [[bool(x) if isinstance(x, bool) else False for x in jax.tree_util.tree_leaves(g["mask"])] for g in fm._param_groups]

                group_counts = []
                for g, gmask in zip(fm._param_groups, group_leaf_masks):
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
                    self.log.warning(f"Model {lid}: parameter groups overlap on {overlap_count} array leaves. " "Update order will follow optax.chain mask order.")

                self.log.info(f"Model {lid}: parameter groups summary — groups={len(fm._param_groups)}, " f"overlap={overlap_count}, uncovered_by_groups={uncovered_count}")
                self.log.quiet(f"Parameter Group Diagnostic Report for model {lid}")
                self.log.quiet(f"groups={len(fm._param_groups)}, overlap={overlap_count}, uncovered={uncovered_count}")
                for tgt, cnt in group_counts:
                    self.log.quiet(f"  target={tgt!r}: matched_arrays={cnt}")

                for g in fm._param_groups:
                    g_opt = g["opt_fn"] or global_opt_fn
                    g_lr = g["lr"] if g["lr"] is not None else global_lr
                    chain = _build_opt_chain(g_opt, g_lr)
                    masked_transforms.append(optax.masked(chain, g["mask"]))
                    group_scheds.append(g_lr)

                # "default" group: negate all group masks to cover remaining params
                def _default_mask(params, _groups=fm._param_groups):
                    """True for leaves in no explicit group."""
                    combined = jax.tree_util.tree_map(lambda _: False, params)
                    for g in _groups:
                        combined = jax.tree_util.tree_map(
                            lambda c, m: c or (m if isinstance(m, bool) else False),
                            combined,
                            g["mask"],
                        )
                    return jax.tree_util.tree_map(lambda c: not c, combined)

                default_chain = _build_opt_chain(global_opt_fn, global_lr)
                masked_transforms.append(optax.masked(default_chain, _default_mask(trainable[lid])))
                group_scheds.append(global_lr)

                per_model_opts[k] = optax.chain(*masked_transforms)
                group_lr_schedules[k] = group_scheds
                self.log.info(f"Model {lid}: {len(fm._param_groups)} parameter group(s) + default — " f"using per-group optimizers")
            else:
                # ── Single global optimizer (original behaviour) ──
                opt_fn = fm._opt_fn
                lr_sched = fm._lr if fm._lr is not None else LearningRateSchedule(1e-3)

                if opt_fn is None:
                    raise ValueError(f"Model (layer {lid}) has no optimizer.")

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

        # Initialise optimizer states and place on mesh
        opt_states = {}
        for k in per_model_opts:
            lid = int(k)
            state = per_model_opts[k].init(trainable[lid])
            # Copy every array leaf so that aliased buffers (e.g. from
            # L-BFGS zero-initialised history arrays that share the same
            # underlying allocation) become distinct.  Without this,
            # donate_argnums will fail with "Attempt to donate the same
            # buffer twice".
            # Then place on the mesh with P() so shardings are canonical
            # and match what the step function will produce.
            opt_states[k] = jax.tree_util.tree_map(
                lambda x: jax.device_put(jnp.copy(x), NamedSharding(self.mesh, P())) if isinstance(x, (jnp.ndarray, jax.Array)) else x,
                state,
            )

        self._log_constraint_shapes(batchsize, min_consecutive=min_consecutive)

        # ── 6. Prepare data ──
        n_devices = len(self.devices)
        full_context = self.domain_data.context

        if offload_data:
            # Keep full dataset as numpy on host — only a mini-batch is
            # transferred to the device each step.
            host_context = {k: np.asarray(v) for k, v in full_context.items()}
            total_samples = _infer_total_samples(host_context)
            effective_batchsize = None  # data is already pre-sliced
            self.log.info(f"Data offloading enabled: {total_samples} total samples, " f"streaming batches of {batchsize} from host")
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
        )

        # Optionally amortise Python dispatch overhead by running multiple
        # gradient steps inside a single XLA program via fori_loop.
        # Only valid when context is fixed on-device (offload_data=False).
        if inner_steps > 1:
            if offload_data:
                self.log.warning("inner_steps > 1 is not compatible with offload_data=True; " "falling back to inner_steps=1")
                inner_steps = 1
            else:
                _K = inner_steps
                _nc = self.n_constraints
                _single = step_fn

                def step_fn(trainable, opt_states, rng, context, start_epoch, prev_losses):
                    def body(i, carry):
                        tr, opt, rn, _total, _indv = carry
                        tr, opt, rn, total, indv = _single(tr, opt, rn, context, start_epoch + i, _indv)
                        return tr, opt, rn, total, indv

                    init = (trainable, opt_states, rng, jnp.zeros(()), prev_losses)
                    return jax.lax.fori_loop(0, _K, body, init)

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
                replicated,  # total_loss
                replicated,  # individual_losses  (→ prev_losses next step)
            )
            jit_step = jax.jit(
                step_fn,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                donate_argnums=(0, 1, 2),
            )

            if has_trackers:
                jit_track = jax.jit(track_fn)

            self.log.info("JIT compiling step function with mesh sharding — " "this might take a while")

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

            _ = jit_step.lower(
                trainable,
                opt_states,
                self.rng,
                trace_context,
                jax.device_put(jnp.int32(0), replicated),
                prev_losses,
            ).compile()

            # Warmup: run a few real dispatches on throw-away copies so
            # the GPU's buffer allocator, CUDA kernel instruction cache,
            # and cuDNN workspaces are fully initialised before any
            # profiling starts. Without this the first 1-2 profiled steps
            # are anomalously slow, making the trace misleading.
            _tw = jax.tree_util.tree_map(jnp.copy, trainable)
            _ow = jax.tree_util.tree_map(jnp.copy, opt_states)
            _rw = jnp.copy(self.rng)
            _pl = jax.device_put(jnp.zeros(self.n_constraints), replicated)
            _ep0 = jax.device_put(jnp.int32(0), replicated)
            for _ in range(3):
                _tw, _ow, _rw, _, _pl = jit_step(_tw, _ow, _rw, trace_context, _ep0, _pl)
            jax.effects_barrier()
            del _tw, _ow, _rw, _pl, _ep0

            # ── 8. Training loop ──
            # hw_monitor = HardwareMonitor(logger=self.log, interval=0.5)
            # hw_monitor.start()

            print_rate = max(1, epochs // 10 if epochs < 100_000 else epochs // 1000)
            prev_losses = jax.device_put(jnp.zeros(self.n_constraints), replicated)

            # Log buffers
            log_epochs = []
            log_losses = []
            log_total_loss = []
            log_timestamps = []
            log_track_stats = []

            rng_np = np.random.default_rng(int(jax.device_get(self.rng[0])))
            st = time.time()
            epoch_jnp = jax.device_put(jnp.int32(0), replicated)
            _epoch_step = jax.device_put(jnp.int32(inner_steps), replicated)

            n_outer = epochs // inner_steps
            if epochs % inner_steps != 0:
                self.log.warning(f"epochs={epochs} is not divisible by inner_steps={inner_steps}; " f"running {n_outer * inner_steps} epochs instead.")
            print_rate = max(10, n_outer // 10 if n_outer < 100_000 else n_outer // 1000)

            # Freeze all surviving Python objects (model params, opt states, etc.)
            # so Python's cyclic GC never has to scan them during the hot loop.
            # Without this, GC kicks in mid-step every ~700 allocations, adding
            # random multi-ms pauses visible as long unflatten spans in xprof.
            gc.disable()  # prevent cyclic GC from interrupting the hot loop;
            # JAX pytrees/dicts are acyclic so refcounting handles them correctly

            _profile_steps = 50 if profile else 0  # number of outer steps to profile; 0 = disabled
            _profiling_active = _profile_steps > 0
            _profile_ctx = jax.profiler.trace(f"{self.log.path}/traces") if _profiling_active else nullcontext()
            # Re-evaluate _trace now that we know whether we'll open a session
            # if _profiling_active:
            #    _trace = jax.profiler.TraceAnnotation
            # (else: _trace is already nullcontext lambda from above)

            with _profile_ctx:
                for outer_epoch in range(n_outer):
                    epoch = outer_epoch * inner_steps  # first epoch of this outer step

                    # --- adaptive host-side resampling at outer-step boundaries ---
                    if has_resampling and strategies:
                        due = [(tag, strat) for tag, strat in strategies.items() if strat.should_resample(epoch)]
                        if due:
                            full_models = eqx.combine(trainable, frozen_arrays, static)
                            if _paramax is not None:
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

                                # Current strategies are designed for steady-state point
                                # sets represented as (B, 1, N, D). Keep T fixed at 1.
                                if not hasattr(tag_points, "ndim") or tag_points.ndim != 4 or tag_points.shape[1] != 1:
                                    self.log.warning(f"Resampling skipped for tag '{tag}': expected point shape (B, 1, N, D), " f"got {tuple(tag_points.shape)}")
                                    continue

                                points_bn = jnp.asarray(tag_points[:, 0, :, :])
                                n_batch, n_points = points_bn.shape[0], points_bn.shape[1]

                                idxs = tag_to_constraint_indices.get(tag, [])
                                if not idxs:
                                    self.log.warning(f"Resampling skipped for tag '{tag}': no constraints associated with this tag")
                                    continue

                                scored = []
                                for idx in idxs:
                                    collapsed = _collapse_residual_for_tag(residuals_all[idx], n_points, n_batch)
                                    if collapsed is not None:
                                        scored.append(collapsed)

                                if not scored:
                                    self.log.warning(f"Resampling skipped for tag '{tag}': no compatible pointwise residuals")
                                    continue

                                combined = jnp.mean(jnp.stack(scored, axis=0), axis=0)  # (B, N)

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
                                        )
                                    )

                                new_points_bn = jnp.stack(new_batches, axis=0)
                                full_context[tag] = new_points_bn[:, None, :, :]
                                self.domain.context[tag] = np.asarray(full_context[tag])
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

                    # --- prepare context for this step ---

                    if offload_data:
                        if host_context is None:
                            raise RuntimeError("offload_data=True but host_context is not available")
                        indices = rng_np.choice(total_samples, batchsize, replace=False)
                        batch_np = {k: (v if k == "__time__" else (np.broadcast_to(v, (batchsize,) + v.shape[1:]) if v.shape[0] == 1 else v[indices])) for k, v in host_context.items()}
                        context = self._shard_data(jax.device_put(batch_np))
                    else:
                        context = on_device_context

                    (trainable, opt_states, self.rng, total_loss, individual_losses) = jit_step(trainable, opt_states, self.rng, context, epoch_jnp, prev_losses)

                    epoch_jnp = epoch_jnp + _epoch_step
                    prev_losses = individual_losses

                    # Stop profiling after the requested number of steps to avoid
                    # the in-memory trace buffer growing and adding per-step overhead.
                    if _profiling_active and outer_epoch + 1 >= _profile_steps:
                        _profiling_active = False
                        _trace = lambda name, **_: nullcontext()  # noqa: E731
                        _profile_ctx.__exit__(None, None, None)
                        _profile_ctx = nullcontext()

                    # --- logging: sync only at print interval ---
                    should_print = (outer_epoch % print_rate == 0) or (outer_epoch == n_outer - 1)
                    if should_print:
                        losses_np, total_np_arr = jax.device_get((individual_losses, total_loss))
                        losses_np = np.asarray(losses_np)
                        total_np = float(total_np_arr)

                        displayed_epoch = epoch + inner_steps - 1  # last epoch completed in this outer step
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

                        # Progress line
                        loss_strs = " | ".join(f"C{i}: {l:>10.4e}" for i, l in enumerate(losses_np))
                        if track_stats_np is not None:
                            track_strs = " | ".join(f"T{i}: {v:>10.4e}" for i, v in enumerate(track_stats_np))
                            self.log.info(f"Epoch {displayed_epoch:>6}/{epochs}| " f"L:{total_np:>10.4e} | {loss_strs} | {track_strs}")
                        else:
                            self.log.info(f"Epoch {displayed_epoch:>6}/{epochs}| " f"L:{total_np:>10.4e} | {loss_strs}")

            et = time.time()
            # hw_monitor.stop(logger=self.log)

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

        self._total_epochs += epochs

        # Checkpoint
        self.checkpoints.append(
            self.checkpoint(
                models=self.models,
                opt_state=opt_states,
                rng=self.rng,
            )
        )

        return statistics(self.training_logs)

    def checkpoint(self, models, opt_state, rng, lora_params=None):
        """Snapshot current model weights and optimiser state."""
        import copy

        payload = {
            "step": int(self._total_epochs),
            "time": time.time(),
            "models": copy.deepcopy(models),
            "opt_state": copy.deepcopy(opt_state),
            "rng": copy.deepcopy(rng),
        }
        if lora_params is not None:
            payload["lora_params"] = copy.deepcopy(lora_params)
        return payload

    def _log_constraint_shapes(self, batchsize, min_consecutive: int = 1):
        """Log the output shape of each constraint by doing a test evaluation.

        When the log level is DEBUG, prints a full shape-annotated tree
        for each constraint so users can see how shapes evolve through
        every node of the expression.
        """

        # Create dummy inputs for shape inference
        test_rng = jax.random.PRNGKey(0)

        # Use jax.eval_shape to get output shape without computation
        out_shape = jax.eval_shape(
            lambda: self.compiled_constraints_fn(
                self.models,
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
                    self.models,
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
                self.log.info(f"Constraint {i}: Shape = {p_shape.shape}" f" → .{op_name}() → {const.shape}")
            else:
                self.log.info(f"Constraint {i}: Shape = {const.shape}")

        for i, (_, fn) in enumerate(self.compiled_trackers):
            # Use jax.eval_shape to get output shape without computation
            out_shape = jax.eval_shape(
                lambda: fn(
                    self.models,
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
                        self.models,
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
                    self.log.info(f"Tracker {i}: Shape = {t_shape.shape}" f" → .{op_name}() → {out_shape.shape}")
                else:
                    self.log.info(f"Tracker {i}: Shape = {out_shape.shape}")
            else:
                self.log.info(f"Tracker {i}: {out_shape}")

        # === Detailed shape trace (logged at DEBUG level) ===
        is_enabled_for = getattr(self.log, "isEnabledFor", None)
        if callable(is_enabled_for) and bool(is_enabled_for(10)):
            self._log_shape_traces(min_consecutive=min_consecutive)

        return None

    def _build_shape_context(self, min_consecutive: int = 1) -> dict:
        """Build a single-sample runtime context for shape tracing.

        The compiled expression uses ``vmap(B) → scan(T) → eval(...)``.
        This method strips B and keeps a temporal window of size
        ``min_consecutive`` (clamped to available T) so shape tracing mirrors
        what the evaluator receives at runtime.

        Returns a plain dict mapping tag → array.
        """
        t_total = 1
        if "__time__" in self.domain_data.context:
            _t_arr = jnp.asarray(self.domain_data.context["__time__"])
            if _t_arr.ndim >= 1:
                t_total = int(_t_arr.shape[0])
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

    def _log_shape_traces(self, min_consecutive: int = 1):
        """Emit per-node shape trees for constraints and trackers.

        Called automatically when log level is DEBUG, or on demand via
        ``core.print_shapes()``.
        """
        ctx_single = self._build_shape_context(min_consecutive=min_consecutive)
        evaluator = TraceEvaluator(self.models)

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

    def print_shapes(self, min_consecutive: int = 1):
        """Print shape-annotated expression trees to stdout.

        Can be called any time after ``compile()`` or ``solve()`` has
        run.  Useful for troubleshooting shape mismatches::

            crux = jno.core([pde.mse, ini.mse], domain)
            crux.print_shapes()
        """
        ctx_single = self._build_shape_context(min_consecutive=min_consecutive)
        evaluator = TraceEvaluator(self.models)

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

    def eval(self, operation: BinaryOp, domain: Optional[domain] = None, min_consecutive: int = 1, key=None):
        """
        Evaluates an operation.
        """
        domain_data = self.domain_data if domain is None else self.prepare_domain_data(domain)

        fn = TraceCompiler.compile_traced_expression(operation, self.all_ops)
        result = fn(
            self.models,
            domain_data.context,
            batchsize=None,
            key=key,
            min_consecutive=min_consecutive,
        )
        return result

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
