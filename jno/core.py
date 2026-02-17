from typing import List, Callable, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import optax
import lox
import cloudpickle
import numpy as np
import time
from .trace import *
from .utils import LearningRateSchedule, WeightSchedule, statistics, get_logger
from .domain import domain, DomainData
from .trace_evaluator import TraceEvaluator
from .core_utilities import CoreUtilities
from .tuner import ArchSpace, DeviceConfig, Tuner
from .utils.lora import LoRA


class core(CoreUtilities):
    """core solver using traced operations."""

    def __init__(
        self,
        constraints: List[Placeholder],
        domain: domain,
        rng_seed: int = 21,
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
                and any stochastic operations during training. Default: 21.

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
        self.params = {}
        self.layer_info = {}
        self._trained_ops = {}
        self.training_logs: List[Dict[str, jnp.ndarray]] = []
        self.dots: List = []
        self.checkpoints: List[Dict] = []
        self.all_ops: List[BinaryOp] = []

        super().__init__()

        self.errors.set_domain(domain)
        self._total_epochs = 0
        self.rng = jax.random.PRNGKey(rng_seed)

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

        self.mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape, devices=self.devices), axis_names=("batch", "model"))

        # Params sharded along model axis (replicated if model dim is 1)
        self.param_sharding = NamedSharding(self.mesh, P(None, "model"))
        # Data sharded along batch axis
        self.data_sharding = NamedSharding(self.mesh, P("batch", None))

        self.log.info(f"Device mesh: {self.mesh} (shape: {mesh_shape})")
        return None

    def _shard_params(self, params: Dict) -> Dict:
        """Apply sharding to model parameters."""

        model_dim = self.mesh.shape["model"]

        def shard_leaf(x):
            # Handle JAX arrays
            if isinstance(x, (jnp.ndarray, jax.Array)):
                if model_dim == 1:
                    spec = P(*([None] * x.ndim))
                else:
                    if x.ndim == 0:
                        spec = P()
                    elif x.ndim == 1:
                        spec = P(None)
                    else:
                        spec = P(*([None] * (x.ndim - 1)), "model")
                return jax.device_put(x, NamedSharding(self.mesh, spec))
            # Handle numpy arrays (convert first)
            elif isinstance(x, np.ndarray):
                x = jnp.array(x)
                if model_dim == 1:
                    spec = P(*([None] * x.ndim))
                else:
                    if x.ndim == 0:
                        spec = P()
                    elif x.ndim == 1:
                        spec = P(None)
                    else:
                        spec = P(*([None] * (x.ndim - 1)), "model")
                return jax.device_put(x, NamedSharding(self.mesh, spec))
            return x

        return jax.tree_util.tree_map(shard_leaf, params)

    def _shard_data(self, data: Dict) -> Dict:
        """Apply sharding to training data."""

        def shard_leaf(x):
            if isinstance(x, jnp.ndarray):
                if x.ndim == 0:
                    # Scalars: no sharding
                    return x
                elif x.ndim == 1:
                    # 1D arrays: shard along the single axis (batch)
                    spec = P("batch")
                else:
                    # 2D+ arrays: shard along first axis (batch), replicate rest
                    spec = P("batch", *([None] * (x.ndim - 1)))
                return jax.device_put(x, NamedSharding(self.mesh, spec))
            return x

        return jax.tree_util.tree_map(shard_leaf, data)

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
            elif isinstance(expr, Laplacian) and isinstance(expr.target, (OperationDef, OperationCall)):
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

    def get_constraint_tags(self, constraints: List) -> List[str]:
        """Get the primary tag for each constraint."""
        tags = []
        for expr in constraints:
            tag = get_primary_tag(expr)
            tags.append(tag if tag is not None else "default")
        return tags

    def init_schedules(self, learning_rate: Optional[LearningRateSchedule], constraint_weights: Optional[WeightSchedule], n_constraints: int) -> Tuple[LearningRateSchedule, WeightSchedule]:
        """Initialize learning rate and weight schedules with defaults."""
        if learning_rate is None:
            learning_rate = LearningRateSchedule(1e-2)
        if constraint_weights is None:
            constraint_weights = WeightSchedule([1.0] * n_constraints)

        if constraint_weights(0, jnp.array(n_constraints)).size != n_constraints:
            raise ValueError(f"WeightSchedule has {constraint_weights(0, jnp.array(n_constraints)).size} weights but " f"{n_constraints} constraints were provided.")

        return learning_rate, constraint_weights

    def compute_tensor_dims(self, domain) -> Dict[str, Tuple]:
        """Compute input dimensions for each tensor tag."""
        tensor_dims = {}
        if hasattr(domain, "tensor_tags"):
            for name, tensor in domain.tensor_tags.items():
                tensor_dims[name] = tensor.shape[1:]

        if hasattr(domain, "sampled_points"):
            for name, tensor in domain.sampled_points.items():
                tensor_dims[name] = tensor.shape[1:]

        return tensor_dims

    def prepare_domain_data(self, domain) -> DomainData:
        """Convert domain data to JAX arrays for training."""
        if domain is None:
            raise ValueError("domain required")

        points_by_tag = {}
        n_batches = 1

        if hasattr(domain, "sampled_points"):
            for tag, pts in domain.sampled_points.items():
                pts = jnp.asarray(pts)
                if pts.ndim == 3:
                    n_batches = max(n_batches, pts.shape[0])
                    points_by_tag[tag] = pts
                else:
                    points_by_tag[tag] = pts[None, ...]

        tensor_tags = {}
        tensor_batch_size = 1
        if hasattr(domain, "tensor_tags"):
            for name, tensor in domain.tensor_tags.items():
                tensor = jnp.asarray(tensor)
                tensor_tags[name] = tensor
                # Track max batch size from tensor tags
                if tensor.ndim > 0:
                    tensor_batch_size = max(tensor_batch_size, tensor.shape[0])

        # For operator learning: use the larger batch size but DON'T tile
        # Let vmap handle broadcasting (in_axes=None) to avoid copies
        n_batches = max(n_batches, tensor_batch_size)

        ordered_tags = tuple(points_by_tag.keys())
        points_arrays = tuple(points_by_tag[tag] for tag in ordered_tags)

        return DomainData(tensor_tags=tensor_tags, dimension=domain.dimension, points_by_tag=dict(zip(ordered_tags, points_arrays)))

    # Training
    def _make_loss_fn(self, compiled_constraints, points_by_tag, tensor_tags, batchsize):
        """Create loss function - rng passed at call time."""

        def loss_fn(params, tag_weights, rng):
            losses = []
            for fn in compiled_constraints:
                residual = fn(params, points_by_tag, tensor_tags, batchsize=batchsize, key=rng)
                mean_squared_residual = jnp.mean(residual)
                losses.append(mean_squared_residual)

            losses = jnp.stack(losses)
            weighted_loss = jnp.dot(tag_weights, losses)
            return weighted_loss, losses

        return loss_fn

    def _make_track_fn(self, compiled_trackers, points_by_tag, tensor_tags, batchsize):
        """Create tracking function with conditional evaluation."""

        # Pre-build individual conditional functions
        def make_conditional_tracker(interval, fn):
            def conditional_fn(params, rng, epoch):
                return jax.lax.cond(
                    epoch % interval == 0,
                    lambda: fn(params, points_by_tag, tensor_tags, batchsize=batchsize, key=rng),
                    lambda: jnp.zeros(batchsize),  # TODO tracker does not have to be a scaler
                )

            return conditional_fn

        conditional_trackers = [make_conditional_tracker(interval, fn) for interval, fn in compiled_trackers]

        def track_fn(params, rng, epoch):
            return [tracker(params, rng, epoch) for tracker in conditional_trackers]

        return track_fn

    def make_train_fn(self, optimizer, compiled_constraints, compiled_trackers, domain_data, epochs, batchsize):
        points_by_tag = domain_data.points_by_tag
        tensor_tags = domain_data.tensor_tags
        n_constraints = len(compiled_constraints)
        print_rate = int(epochs / 100) if epochs < 100_000 else int(epochs / 1000)

        if len(compiled_trackers) > 0:
            track_fn = self._make_track_fn(compiled_trackers, points_by_tag, tensor_tags, batchsize)
        loss_fn = self._make_loss_fn(compiled_constraints, points_by_tag, tensor_tags, batchsize)

        def print_progress(epoch, individual_losses, track_stats, loss):
            """Host-side callback for progress printing."""
            loss_strs = " | ".join([f"C{i}: {float(l):>10.4e}" for i, l in enumerate(individual_losses)])
            if track_stats is not None:
                track_strs = " | ".join([f"T{i}: {float(jnp.mean(l)):>10.4e}" for i, l in enumerate(track_stats)])
                print(f"\rEpoch {int(epoch):>6}/{epochs}| L:{float(loss):>10.4e} | {loss_strs} | {track_strs}", end="\n", flush=True)
            else:
                print(f"\rEpoch {int(epoch):>6}/{epochs}| L:{float(loss):>10.4e} | {loss_strs}", end="\n", flush=True)
            return None

        def train_n_steps(params, opt_state, rng):
            original_lr_dtype = opt_state[-1].hyperparams["step_size"].dtype

            def body_fn(_, carry):
                params, opt_state, rng, total_loss, individual_losses, epoch = carry
                tag_weights = self.constraint_weights(self._total_epochs + epoch, individual_losses)

                rng, step_rng = jax.random.split(rng)

                def loss_wrapper(p):
                    return loss_fn(p, tag_weights, step_rng)

                track_stats = track_fn(params, step_rng, epoch) if len(compiled_trackers) > 0 else None

                (total_loss, individual_losses), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(params)
                updates, opt_state = optimizer.update(grads, opt_state, params, value=total_loss, grad=grads, value_fn=lambda p: loss_wrapper(p)[0])

                lr = self.learning_rate(self._total_epochs + epoch, individual_losses)
                opt_state[-1].hyperparams["step_size"] = jnp.asarray(lr, dtype=original_lr_dtype)

                params = optax.apply_updates(params, updates)

                # Progress callback (unordered, works with multi-device)
                jax.lax.cond(
                    epoch % print_rate == 0,
                    lambda: jax.experimental.io_callback(print_progress, None, epoch, individual_losses, track_stats, total_loss, ordered=False),
                    lambda: None,
                )

                lox.log(
                    {
                        "epoch": epoch,
                        "learning_rate": opt_state[-1].hyperparams["step_size"],
                        "losses": individual_losses,
                        "weights": tag_weights,
                        "total_loss": jnp.sum(individual_losses),
                        "track_stats": track_stats if len(compiled_trackers) > 0 else [0],
                    }
                )

                return params, opt_state, rng, total_loss, individual_losses, epoch + 1

            init_carry = (params, opt_state, rng, 0.0, jnp.zeros(n_constraints), 0)
            return jax.lax.fori_loop(0, epochs, body_fn, init_carry)

        return train_n_steps

    def compile(self, mesh: tuple = (1, 1)):

        # === Parallelism ===
        self._setup_parallelism(mesh)

        # === Preprocessing ===
        constraints = self.wrap_constraints(self.constraints)

        # === Collect operations and tags ===
        self.all_ops = self.collect_unique_operations(constraints)

        # === Prepare domain data ===
        self.domain_data = self.prepare_domain_data(self.domain)
        tensor_dims = self.compute_tensor_dims(self.domain)

        # === Initialize parameters ===
        self.params, self.layer_info, self.rng = TraceEvaluator.init_layer_params(self.all_ops, self.domain_data.dimension, tensor_dims, self.rng, self.log)

        # === Apply sharding to params ===
        self.params = self._shard_params(self.params)
        self.log.info("Parameters sharded across devices")

        # === Compile constraints ===
        self.compiled_constraints = []
        self.compiled_trackers = []
        for expr in constraints:
            fn_expr = TraceEvaluator.compile_traced_expression(expr, self.all_ops, self.layer_info)
            if hasattr(expr, "expr"):
                if isinstance(expr.expr, Tracker):
                    self.compiled_trackers.append((expr.expr.interval, fn_expr))
                else:
                    self.compiled_constraints.append(fn_expr)
            else:
                self.compiled_constraints.append(fn_expr)

        self.log.info(f"There are a total of {self.count(self.params)} trainable parameters in the network/s.")
        return None

    def solve(
        self,
        epochs: int = 1000,
        optimizer: optax.GradientTransformation = optax.adam(1.0),
        learning_rate: LearningRateSchedule = None,
        constraint_weights: WeightSchedule = None,
        batchsize: int = None,
        lora: Optional[Dict] = None,
    ):
        """
        Solve using traced constraints with optional LoRA fine-tuning.

        Args:
            epochs: Number of training epochs.
            optimizer: Optax optimizer.
            learning_rate: Learning rate schedule.
            constraint_weights: Weight schedule for constraints.
            batchsize: Mini-batch size (None for full-batch).
            lora: Optional rank dictionary for LoRA fine-tuning.
                Structure mirrors params with (rank, alpha) tuples at leaves.
                Use float('nan') to skip a layer.

                For multi-model (params has int keys):
                    lora = {
                        0: {  # Model 0
                            'params': {
                                'Dense_0': {'kernel': (8, 1.0)},
                                'Dense_1': {'kernel': float('nan')},
                            }
                        },
                        # Model 1 not included = no LoRA for it
                    }

                For single model:
                    lora = {
                        'params': {
                            'Dense_0': {'kernel': (8, 1.0)},
                        }
                    }

                Use create_rank_dict() helper to generate this automatically.

        Returns:
            statistics: Training history with visualization methods.
        """
        n_constraints = len(self.compiled_constraints)
        batchsize = batchsize if batchsize is not None else self.domain.total_samples

        self.learning_rate = learning_rate if learning_rate is not None else LearningRateSchedule(1.0)
        self.constraint_weights = constraint_weights if constraint_weights is not None else WeightSchedule([1.0 for _ in range(n_constraints)])

        if isinstance(optimizer, Callable):
            optimizer = optimizer(1.0)
        else:
            self.log.warning("Optimizer should have learning rate 1.0 -> rates are set via learning_rate argument.")

        scale = optax.inject_hyperparams(optax.scale)(step_size=self.learning_rate(0, jnp.zeros(n_constraints)))
        optimizer = optax.chain(optimizer, scale)

        # === LoRA Setup ===
        use_lora = lora is not None

        if use_lora:
            self._lora = LoRA(lora, self.params)
            self.rng, lora_key = jax.random.split(self.rng)

            # Initialize LoRA params
            self._lora_params, lora_param_count, lora_layer_count = self._lora.init(lora_key, self.params)
            self._lora_params = self._shard_params(self._lora_params)

            base_count = self.count(self.params)
            self.log.info(f"LoRA: {lora_layer_count} layers, {lora_param_count:,} params ({100*lora_param_count/base_count:.2f}% of base)")

            # Optimizer on LoRA params only
            trainable_params = self._lora_params
            opt_state = optimizer.init(trainable_params)
        else:
            self._lora = None
            self._lora_params = None

            trainable_params = self.params
            opt_state = optimizer.init(trainable_params)

        self._log_constraint_shapes(batchsize)

        # Prepare data with sharding
        domain_data = self.domain_data
        n_devices = len(self.devices)
        domain_data = DomainData(
            tensor_tags=self._replicate_for_devices(domain_data.tensor_tags, n_devices),
            dimension=domain_data.dimension,
            points_by_tag=self._replicate_for_devices(domain_data.points_by_tag, n_devices),
        )
        domain_data = DomainData(
            tensor_tags=self._shard_data(domain_data.tensor_tags),
            dimension=domain_data.dimension,
            points_by_tag=self._shard_data(domain_data.points_by_tag),
        )

        # Select training function
        if use_lora:
            train_fn = self._lora.make_train_fn(
                optimizer=optimizer,
                compiled_constraints=self.compiled_constraints,
                compiled_trackers=self.compiled_trackers,
                domain_data=domain_data,
                epochs=epochs,
                batchsize=batchsize,
                constraint_weights=self.constraint_weights,
                learning_rate=self.learning_rate,
                total_epochs=self._total_epochs,
                all_params=self.params,
            )
        else:
            train_fn = self.make_train_fn(optimizer, self.compiled_constraints, self.compiled_trackers, domain_data, epochs, batchsize)

        with self.mesh:
            train = jax.jit(lox.spool(train_fn))
            self.log.info("Jit Tracing Training Function with mesh sharding - this might take a while")
            lowered = train.lower(trainable_params, opt_state, self.rng)
            compiled_train = lowered.compile()
            self._cost_analysis(compiled_train)

            st = time.time()
            (trainable_params, opt_state, self.rng, loss, individual_losses, _), logs = train(trainable_params, opt_state, self.rng)
            et = time.time()

            # Update params
            if use_lora:
                self._lora_params = trainable_params
                # Merge LoRA into full params for eval/inference
                self.params = self._lora.get_full_params(self._lora_params, self.params)
            else:
                self.params = trainable_params

            logs = self._to_plain_dict(logs)
            logs["training_time"] = et - st
            logs["lora"] = use_lora
            self.training_logs.append(logs)
            self.log.info(f"Training took {(logs['training_time'] / 60):.2f} minutes")

        self._total_epochs += epochs

        # Checkpoint
        checkpoint_data = {"params": self.params, "opt_state": opt_state, "rng": self.rng, "lora_params": self._lora_params if use_lora else None}

        self.checkpoints.append(self.checkpoint(**checkpoint_data))

        return statistics(self.training_logs)

    def _log_constraint_shapes(self, batchsize):
        """Log the output shape of each constraint by doing a test evaluation."""

        # Create dummy inputs for shape inference
        test_rng = jax.random.PRNGKey(0)

        for i, fn in enumerate(self.compiled_constraints):
            # Use jax.eval_shape to get output shape without computation
            out_shape = jax.eval_shape(lambda: fn(self.params, self.domain_data.points_by_tag, self.domain_data.tensor_tags, batchsize=batchsize, key=test_rng))
            self.log.info(f"Constraint {i}: Shape = {out_shape.shape}")

        for i, (_, fn) in enumerate(self.compiled_trackers):
            # Use jax.eval_shape to get output shape without computation
            out_shape = jax.eval_shape(lambda: fn(self.params, self.domain_data.points_by_tag, self.domain_data.tensor_tags, batchsize=batchsize, key=test_rng))
            self.log.info(f"Tracker {i}: Shape = {out_shape.shape}")

        return None

    def _cost_analysis(self, compiled_train):
        cost_analysis_list = compiled_train.cost_analysis()

        if cost_analysis_list is None:
            return

        # Handle both single dict and list of dicts
        if isinstance(cost_analysis_list, dict):
            # Single aggregated result
            flops = cost_analysis_list.get("flops", 0)
            bytes_accessed = cost_analysis_list.get("bytes accessed", 0)
            self.log.info(f"Performance : megaFLOPS: {flops / 1_000_000:.3f} | megaBYTES: {bytes_accessed / 1_000_000:.3f}")
        elif isinstance(cost_analysis_list, list):
            # Per-device results
            for i, cost_analysis in enumerate(cost_analysis_list):
                device = jax.devices()[i] if i < len(jax.devices()) else f"device_{i}"
                flops = cost_analysis.get("flops", 0)
                bytes_accessed = cost_analysis.get("bytes accessed", 0)
                self.log.info(f"Performance: {device} | megaFLOPS: {flops / 1_000_000:.3f} | megaBYTES: {bytes_accessed / 1_000_000:.3f}")

    def sweep(self, space: ArchSpace, optimizer: Union[str, type], budget: int, devices: Union[None, int, str, List[int], DeviceConfig] = None):
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

        # evaluator = TraceEvaluator(self.params, self.layer_info)

        fn = TraceEvaluator.compile_traced_expression(operation, self.all_ops, self.layer_info)
        result = fn(self.params, domain_data.points_by_tag, domain_data.tensor_tags, batchsize=None, key=self.rng)
        return result

    # Save and Load

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

    def save(self, filepath: str):
        """Save the trained core model to a file."""
        with open(filepath, "wb") as f:
            cloudpickle.dump(self, f)
        self.log.info(f"Model saved to: {filepath}")
        return None

    @classmethod
    def load(cls, filepath: str) -> "core":
        """Load a trained core model from a file."""
        with open(filepath, "rb") as f:
            instance = cloudpickle.load(f)
        return instance
