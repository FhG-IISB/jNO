from typing import List, Callable, Dict, Optional, Tuple, Union
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


class core:
    """core solver using traced operations."""

    def __init__(
        self,
        constraints: List[Placeholder],
        domain: domain,
        rng_seed: int | None = None,
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
        self.constraints: List[BinaryOp] = constraints

        self.domain = domain
        self.models = {}  # full equinox models (pytrees with arrays + static)
        self._trained_ops = {}
        self.training_logs: List[Dict[str, jnp.ndarray]] = []
        self.dots: List = []
        self.checkpoints: List[Dict] = []
        self.all_ops: List[BinaryOp] = []

        super().__init__()

        self._total_epochs = 0
        seed = get_seed() if get_seed() is not None else 21
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

    def init_schedules(
        self,
        learning_rate: Optional[LearningRateSchedule],
        constraint_weights: Optional[WeightSchedule],
        n_constraints: int,
    ) -> Tuple[LearningRateSchedule, WeightSchedule]:
        """Initialize learning rate and weight schedules with defaults."""
        if learning_rate is None:
            learning_rate = LearningRateSchedule(1e-2)
        if constraint_weights is None:
            constraint_weights = WeightSchedule([1.0] * n_constraints)

        if constraint_weights(0, jnp.array(n_constraints)).size != n_constraints:
            raise ValueError(f"WeightSchedule has {constraint_weights(0, jnp.array(n_constraints)).size} weights but " f"{n_constraints} constraints were provided.")

        return learning_rate, constraint_weights

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
    def _make_loss_fn(self, compiled_constraints_fn, n_constraints, batchsize, frozen, static, checkpoint_gradients=False):
        """Create loss function — evaluates ALL constraints in one combined call."""

        def loss_fn(trainable, context, tag_weights, rng):
            full_models = eqx.combine(trainable, frozen, static)

            if checkpoint_gradients:
                _fn, _bs = compiled_constraints_fn, batchsize

                @jax.checkpoint
                def _remat_eval(models, ctx, key):
                    return _fn(models, ctx, batchsize=_bs, key=key)

                all_residuals = _remat_eval(full_models, context, rng)
            else:
                # One call → one JAX function → XLA applies CSE across constraints
                all_residuals = compiled_constraints_fn(full_models, context, batchsize=batchsize, key=rng)

            # all_residuals is a list of (B, T, ...) arrays — one per constraint
            losses = jnp.stack([jnp.mean(r) for r in all_residuals])
            weighted_loss = jnp.dot(tag_weights, losses)
            return weighted_loss, losses

        return loss_fn

    def _make_track_fn(self, compiled_trackers, batchsize, frozen, static):
        """Create tracking function that evaluates monitored expressions.

        Returns a JIT-friendly function that evaluates *all* trackers.
        Interval-based gating is handled by the Python training loop.
        """

        def track_fn(trainable, context, rng):
            full_models = eqx.combine(trainable, frozen, static)
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
        checkpoint_gradients=False,
    ):
        """Build a single JIT-compiled training step.

        Returns a function with signature::

            step(trainable, opt_states, rng, context, epoch, prev_losses)
                -> (trainable, opt_states, rng, total_loss, individual_losses, tag_weights)

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
        )

        lid_keys = sorted(per_model_opts.keys())  # deterministic order
        base_epoch = self._total_epochs

        def step(trainable, opt_states, rng, context, epoch, prev_losses):
            tag_weights = self.constraint_weights(base_epoch + epoch, prev_losses)

            rng, step_rng = jax.random.split(rng)

            def loss_wrapper(p):
                return loss_fn(p, context, tag_weights, step_rng)

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
                    value_fn=lambda p, _lid=lid: loss_fn({**trainable, _lid: p}, context, tag_weights, step_rng)[0],
                )

                # Update LR for this model
                lr_val = lr_schedules[k](base_epoch + epoch, individual_losses)
                new_state[-1].hyperparams["step_size"] = jnp.asarray(lr_val, dtype=opt_states[k][-1].hyperparams["step_size"].dtype)

                trainable = {**trainable, lid: optax.apply_updates(model_params, updates)}
                opt_states = {**opt_states, k: new_state}

            return trainable, opt_states, rng, total_loss, individual_losses, tag_weights

        return step

    def print_tree(self, file: str = None):
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
            print(text)

        return self

    def compile(self, mesh: tuple = (1, 1)):

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

        # self.log.info(f"There are a total of {self.count(self.models)} trainable parameters in the network/s.")
        return None

    def solve(
        self,
        epochs: int = 1000,
        constraint_weights: WeightSchedule = None,
        batchsize: int = None,
        checkpoint_gradients: bool = False,
        offload_data: bool = False,
    ):
        """Train using per-model optimizers attached via ``model.optimizer()``.

        Every model used in the constraints **must** have an optimizer
        attached before calling ``solve()``.  Models can optionally be
        frozen (``model.freeze()``) or have LoRA enabled
        (``model.lora(rank, alpha)``).

        Args:
            epochs: Number of training epochs.
            constraint_weights: Weight schedule for constraints.
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
        batchsize = batchsize if batchsize is not None else self.domain.total_samples

        self.constraint_weights = constraint_weights if constraint_weights is not None else WeightSchedule([1.0] * self.n_constraints)

        # ── 0. Validate offload_data ──
        if offload_data and (batchsize is None or batchsize >= self.domain.total_samples):
            self.log.warning("offload_data requires batchsize < total_samples; " "ignoring offload_data for this run.")
            offload_data = False

        # ── 1. Collect Model metadata ──
        flax_mods = self._collect_flax_modules()  # {layer_id: Model}

        # Validate: every non-frozen model must have an optimizer.
        # A frozen model that has LoRA active is also "effectively trainable"
        # (LoRA overrides freeze) and therefore also needs an optimizer.
        for lid, fm in flax_mods.items():
            needs_optimizer = (not fm._frozen) or (fm._lora_config is not None)
            if needs_optimizer and fm._opt_fn is None:
                raise ValueError(f"Model '{fm.name or type(fm.module).__name__}' (layer {lid}) " f"has no optimizer. Call  model.optimizer(optax.adam, lr=...)  " f"before solve(), or freeze it with  model.freeze().")

        # ── 2. Apply LoRA transforms ──
        models = dict(self.models)
        lora_param_counts = {}  # Track LoRA params per model for logging
        for lid, fm in flax_mods.items():
            if fm._lora_config is not None:
                rank, alpha = fm._lora_config
                self.rng, key = jax.random.split(self.rng)
                model_before = models[lid]
                models[lid] = _apply_lora(models[lid], rank, alpha, key=key)
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
                    self.log.info(f"LoRA (Flax) applied to model {lid} (rank={rank}, alpha={alpha}): " f"{n_lora_layers} kernel layers adapted, " f"{n_lora_params:,} new LoRA params, " f"base frozen at {n_params_before:,} params")
                else:
                    from .architectures.linear import Linear as JNOLinear

                    is_lora = lambda x: isinstance(x, LoRALinear)
                    lora_leaves_after = [l for l in jax.tree_util.tree_leaves(model_after, is_leaf=is_lora) if isinstance(l, LoRALinear)]
                    n_lora_layers = len(lora_leaves_after)
                    self.log.info(f"LoRA applied to model {lid} (rank={rank}, alpha={alpha}): " f"{n_lora_layers} LoRALinear layers, " f"Params: {n_params_before:,}→{n_params_after:,}")
                    if n_lora_layers == 0:
                        self.log.warning(f"LoRA: No layers were adapted for model {lid}! " f"LoRA has NO EFFECT on this model.")

                # Write detailed LoRA diagnostics to log file (file-only, no console noise)
                self.log.quiet(f"LoRA Diagnostic Report for model {lid}")
                self.log.quiet(f"{'='*60}")
                self.log.quiet(f"Requested: rank={rank}, alpha={alpha}")
                self.log.quiet(f"Model type: {type(model_before).__name__}")
                self.log.quiet(f"Wrapper type: {type(model_after).__name__}")
                self.log.quiet(f"LoRA layers adapted:        {n_lora_layers}")
                self.log.quiet(f"Total arrays before LoRA:   {n_arrays_before}")
                self.log.quiet(f"Total arrays after LoRA:    {n_arrays_after}")
                self.log.quiet(f"Total params before LoRA:   {n_params_before:,}")
                self.log.quiet(f"Total params after LoRA:    {n_params_after:,}")
                self.log.quiet(f"New LoRA params:            {n_lora_params:,}")
                if isinstance(model_after, FlaxLoRAWrapper):
                    self.log.quiet("Adapted kernels (lora_a / lora_b shapes):")
                    flat, _ = jax.tree_util.tree_flatten_with_path(model_after.lora_params)
                    for path, leaf in flat:
                        if eqx.is_array(leaf):
                            self.log.quiet(f"  {'/'.join(str(k) for k in path)}: {leaf.shape} {leaf.dtype}")
                else:
                    self.log.quiet("Full pytree paths (after LoRA):")
                    flat, _ = jax.tree_util.tree_flatten_with_path(model_after)
                    for path, leaf in flat:
                        if eqx.is_array(leaf):
                            self.log.quiet(f"  {'/'.join(str(k) for k in path)}: {leaf.shape} {leaf.dtype}")

        # ── 3. Build trainable filter ──
        filter_spec = {}
        for lid, model in models.items():
            fm = flax_mods.get(lid)
            if fm is not None and fm._lora_config is not None:
                # LoRA takes highest priority — base frozen by LoRA itself.
                # Explicitly calling .freeze() before .lora() is therefore a
                # no-op: LoRA still wins so callers can freely chain
                # .mask(...).freeze().lora(...) without unexpected behaviour.
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
        self.log.info(f"  Trainable parameters:  {n_trainable_params:>12,}")
        self.log.info(f"  Frozen parameters:     {n_frozen_params:>12,}")
        self.log.info(f"  Total parameters:      {n_total_params:>12,}")
        if n_lora_params_total > 0:
            self.log.info(f"  LoRA parameters:       {n_lora_params_total:>12,} (included in trainable)")
            self.log.info(f"  LoRA % of total:       {100.0 * n_lora_params_total / n_total_params:>11.2f}%")

        # Shard trainable params
        trainable = self._shard_params(trainable)

        # ── 5. Build per-model optimizers ──
        per_model_opts = {}  # {str(lid): optax chain}
        lr_schedules = {}  # {str(lid): LearningRateSchedule}
        zeros = jnp.zeros(self.n_constraints)

        for lid, fm in flax_mods.items():
            # Skip only if truly frozen with no LoRA override.
            # If LoRA is active, we need an optimizer even when _frozen=True
            # because LoRA takes priority and its adapter params are trainable.
            if fm._frozen and fm._lora_config is None:
                continue
            k = str(lid)

            opt_fn = fm._opt_fn
            lr_sched = fm._lr

            # Instantiate optimizer (handle callables like optax.adam)
            if isinstance(opt_fn, Callable) and not isinstance(opt_fn, optax.GradientTransformation):
                try:
                    base_opt = opt_fn(1.0)
                except TypeError:
                    base_opt = opt_fn
            else:
                base_opt = opt_fn

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

        self._log_constraint_shapes(batchsize)

        # ── 6. Prepare data ──
        n_devices = len(self.devices)

        if offload_data:
            # Keep full dataset as numpy on host — only a mini-batch is
            # transferred to the device each step.
            host_context = {k: np.asarray(v) for k, v in self.domain_data.context.items()}
            total_samples = max(v.shape[0] for v in host_context.values())
            effective_batchsize = None  # data is already pre-sliced
            self.log.info(f"Data offloading enabled: {total_samples} total samples, " f"streaming batches of {batchsize} from host")
        else:
            # Replicate / shard full dataset on device (original behaviour)
            domain_data = self.domain_data
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
            checkpoint_gradients=checkpoint_gradients,
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
            jit_step = jax.jit(
                step_fn,
                in_shardings=in_shardings,
                donate_argnums=(0, 1, 2),
            )

            if has_trackers:
                jit_track = jax.jit(track_fn)

            self.log.info("JIT compiling step function with mesh sharding — " "this might take a while")

            # Trigger AOT compilation so the first real step is fast.
            _ = jit_step.lower(
                trainable,
                opt_states,
                self.rng,
                trace_context,
                jax.device_put(jnp.int32(0), replicated),
                prev_losses,
            ).compile()

            # ── 8. Training loop ──
            # hw_monitor = HardwareMonitor(logger=self.log, interval=0.5)
            # hw_monitor.start()

            print_rate = max(1, epochs // 100 if epochs < 100_000 else epochs // 1000)
            prev_losses = jax.device_put(jnp.zeros(self.n_constraints), replicated)

            # Log buffers
            log_epochs = []
            log_losses = []
            log_total_loss = []
            log_weights = []
            log_timestamps = []
            log_track_stats = []

            rng_np = np.random.default_rng(int(jax.device_get(self.rng[0])))
            st = time.time()

            for epoch in range(epochs):
                # --- prepare context for this step ---
                if offload_data:
                    indices = rng_np.choice(total_samples, batchsize, replace=False)
                    batch_np = {k: np.broadcast_to(v, (batchsize,) + v.shape[1:]) if v.shape[0] == 1 else v[indices] for k, v in host_context.items()}
                    context = self._shard_data(jax.device_put(batch_np))
                else:
                    context = on_device_context

                epoch_jnp = jax.device_put(jnp.int32(epoch), replicated)

                (trainable, opt_states, self.rng, total_loss, individual_losses, tag_weights) = jit_step(trainable, opt_states, self.rng, context, epoch_jnp, prev_losses)

                prev_losses = individual_losses

                # --- logging (only every print_rate epochs) ---
                should_log = (epoch % print_rate == 0) or (epoch == epochs - 1)
                if should_log:
                    # Synchronise once per log interval
                    losses_np = np.asarray(jax.device_get(individual_losses))
                    total_np = float(jax.device_get(total_loss))
                    weights_np = np.asarray(jax.device_get(tag_weights))

                    log_epochs.append(epoch)
                    log_losses.append(losses_np)
                    log_total_loss.append(total_np)
                    log_weights.append(weights_np)
                    log_timestamps.append(time.time())

                    # Trackers
                    track_stats_np = None
                    if has_trackers and any(epoch % intv == 0 for intv in tracker_intervals):
                        track_vals = jit_track(trainable, context, self.rng)
                        track_stats_np = [float(jax.device_get(v)) for v in track_vals]
                        log_track_stats.append(track_stats_np)

                    # Progress line
                    loss_strs = " | ".join(f"C{i}: {l:>10.4e}" for i, l in enumerate(losses_np))
                    if track_stats_np is not None:
                        track_strs = " | ".join(f"T{i}: {v:>10.4e}" for i, v in enumerate(track_stats_np))
                        print(
                            f"\rEpoch {epoch:>6}/{epochs}| " f"L:{total_np:>10.4e} | {loss_strs} | {track_strs}",
                            end="\n",
                            flush=True,
                        )
                    else:
                        print(
                            f"\rEpoch {epoch:>6}/{epochs}| " f"L:{total_np:>10.4e} | {loss_strs}",
                            end="\n",
                            flush=True,
                        )

            et = time.time()
            # hw_monitor.stop(logger=self.log)

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
                "weights": np.stack(log_weights) if log_weights else np.array([]),
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
            self.log.info(f"Training took {(logs['training_time'] / 60):.2f} minutes")

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

    def _log_constraint_shapes(self, batchsize):
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
                )
            )
        else:
            parent_shape = [None] * len(out_shape)

        for i, (const, (_, op_name)) in enumerate(zip(out_shape, parent_exprs)):
            if op_name is not None and parent_shape[i] is not None:
                self.log.info(f"Constraint {i}: Shape = {parent_shape[i].shape}" f" → .{op_name}() → {const.shape}")
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
        if hasattr(self.log, "isEnabledFor") and self.log.isEnabledFor(10):
            self._log_shape_traces()

        return None

    def _build_shape_context(self) -> dict:
        """Build a single-sample, single-timestep context for shape tracing.

        The compiled expression uses ``vmap(B) → scan(T) → eval(...)``.
        This method strips the B and T axes so that shape tracing sees
        the same per-sample shapes the evaluator sees at runtime.

        Returns a plain dict mapping tag → array.
        """
        ctx_single = {}
        for tag, arr in self.domain_data.context.items():
            arr = jnp.asarray(arr)
            if tag == "__time__":
                # (T, 1) → first time step → (1,)
                ctx_single[tag] = arr[0]
            elif arr.ndim >= 3:
                # (B, T, ...) → strip batch + time → (...)
                # Covers (B,T,N,D), (B,T,C,H,W,C_out), etc.
                ctx_single[tag] = arr[0, 0]
            elif arr.ndim == 2:
                # (B, F) parametric → (F,)
                ctx_single[tag] = arr[0]
            else:
                ctx_single[tag] = arr
        return ctx_single

    def _log_shape_traces(self):
        """Emit per-node shape trees for constraints and trackers.

        Called automatically when log level is DEBUG, or on demand via
        ``core.print_shapes()``.
        """
        ctx_single = self._build_shape_context()
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

    def print_shapes(self):
        """Print shape-annotated expression trees to stdout.

        Can be called any time after ``compile()`` or ``solve()`` has
        run.  Useful for troubleshooting shape mismatches::

            crux = jno.core([pde.mse, ini.mse], domain)
            crux.print_shapes()
        """
        ctx_single = self._build_shape_context()
        evaluator = TraceEvaluator(self.models)

        all_exprs = getattr(self, "_constraint_exprs", [])
        all_tracker_exprs = getattr(self, "_tracker_exprs", [])

        for i, expr in enumerate(all_exprs):
            try:
                tree = evaluator.trace_shapes(expr, ctx_single, key=jax.random.PRNGKey(0))
                print(f"═══ Constraint {i} ═══")
                print(tree)
                print()
            except Exception as exc:
                print(f"═══ Constraint {i} ═══  FAILED: {exc}")

        for i, expr in enumerate(all_tracker_exprs):
            try:
                tree = evaluator.trace_shapes(expr, ctx_single, key=jax.random.PRNGKey(0))
                print(f"═══ Tracker {i} ═══")
                print(tree)
                print()
            except Exception as exc:
                print(f"═══ Tracker {i} ═══  FAILED: {exc}")

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

    def eval(self, operation: BinaryOp, domain: domain = None):
        """
        Evaluates an operation.
        """
        domain_data = self.domain_data if domain is None else self.prepare_domain_data(domain)

        fn = TraceCompiler.compile_traced_expression(operation, self.all_ops)
        result = fn(
            self.models,
            domain_data.context,
            batchsize=None,
            key=self.rng,
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
