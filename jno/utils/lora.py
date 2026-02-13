# lora.py

from typing import Dict, Optional, Tuple, Union, List
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import optax
import lox


class LoRA:
    """
    LoRA wrapper for Flax parameters supporting multiple models.

    Handles params structured as {model_id: {layer: params, ...}, ...}
    where model_id is an integer key.

    rank_dict structure mirrors params with (rank, alpha) tuples at leaves.
    Use float('nan') to skip a layer.

    Example rank_dict for multi-model setup:
        {
            0: {  # First model
                'params': {
                    'Dense_0': {'kernel': (8, 1.0)},
                    'Dense_1': {'kernel': (16, 0.5)},
                    'Dense_2': {'kernel': float('nan')},  # skip
                }
            },
            1: {  # Second model - no LoRA
                'params': {
                    'Dense_0': {'kernel': float('nan')},
                }
            }
        }

    Or for single model (backward compatible):
        {
            'params': {
                'Dense_0': {'kernel': (8, 1.0)},
                'Dense_1': {'kernel': (16, 0.5)},
            }
        }
    """

    def __init__(self, rank_dict: Dict, params: Dict):
        """
        Args:
            rank_dict: Dictionary specifying LoRA config per layer.
                       Structure should mirror params with (rank, alpha) tuples.
            params: Full model parameters dict (may have int keys for multi-model).
        """
        self.rank_dict = rank_dict
        self._lora_config: Dict[str, Tuple[int, float]] = {}  # flat path -> (rank, alpha)

        # Detect if params has integer keys (multi-model setup)
        self._is_multi_model = self._check_multi_model(params)

        # Store base params for all models that have LoRA applied
        self._base_params = self._extract_base_params(params)

        # Track which model IDs have LoRA
        self._lora_model_ids: List[int] = []

    def _check_multi_model(self, params: Dict) -> bool:
        """Check if params uses integer keys (multi-model setup)."""
        if not params:
            return False
        first_key = next(iter(params.keys()))
        return isinstance(first_key, int)

    def _extract_base_params(self, params: Dict) -> Dict:
        """
        Extract base params for models that will have LoRA applied.
        Returns the subset of params that matches rank_dict structure.
        """
        if self._is_multi_model:
            # Multi-model: rank_dict should have int keys too
            base = {}
            for key in params.keys():
                if key in self.rank_dict:
                    base[key] = params[key]
                elif isinstance(key, int) and key in self.rank_dict:
                    base[key] = params[key]
            return base if base else params
        else:
            # Single model
            return params

    def _parse_rank_dict(self, params: Dict) -> Dict[str, Tuple[int, float]]:
        """
        Flatten rank_dict and extract (rank, alpha) configs.
        Handles both multi-model and single-model structures.
        """
        config = {}

        if self._is_multi_model:
            # Process each model separately
            for model_id, model_rank_dict in self.rank_dict.items():
                if not isinstance(model_id, int):
                    continue
                if model_id not in params:
                    continue

                # Flatten this model's rank dict with model_id prefix
                flat = flatten_dict(model_rank_dict, sep="/")

                for path, value in flat.items():
                    full_path = f"{model_id}/{path}"
                    parsed = self._parse_value(value)
                    if parsed is not None:
                        config[full_path] = parsed
                        if model_id not in self._lora_model_ids:
                            self._lora_model_ids.append(model_id)
        else:
            # Single model
            flat = flatten_dict(self.rank_dict, sep="/")
            for path, value in flat.items():
                parsed = self._parse_value(value)
                if parsed is not None:
                    config[path] = parsed

        return config

    def _parse_value(self, value) -> Optional[Tuple[int, float]]:
        """Parse a rank_dict value into (rank, alpha) tuple or None."""
        # Skip nan values
        if isinstance(value, float) and np.isnan(value):
            return None

        # Handle (rank, alpha) tuple
        if isinstance(value, (tuple, list)) and len(value) == 2:
            rank, alpha = value
            if isinstance(rank, (int, float)) and not np.isnan(rank) and rank > 0:
                return (int(rank), float(alpha))
        # Handle just rank (default alpha=1.0)
        elif isinstance(value, (int, float)) and not np.isnan(value) and value > 0:
            return (int(value), 1.0)

        return None

    def _flatten_params(self, params: Dict) -> Dict[str, jnp.ndarray]:
        """Flatten params handling multi-model structure."""
        if self._is_multi_model:
            flat = {}
            for model_id, model_params in params.items():
                model_flat = flatten_dict(unfreeze(model_params), sep="/")
                for path, value in model_flat.items():
                    flat[f"{model_id}/{path}"] = value
            return flat
        else:
            return flatten_dict(unfreeze(params), sep="/")

    def _unflatten_params(self, flat: Dict[str, jnp.ndarray]) -> Dict:
        """Unflatten params handling multi-model structure."""
        if self._is_multi_model:
            # Group by model_id
            by_model = {}
            for path, value in flat.items():
                parts = path.split("/", 1)
                if len(parts) == 2:
                    model_id = int(parts[0])
                    rest = parts[1]
                    if model_id not in by_model:
                        by_model[model_id] = {}
                    by_model[model_id][rest] = value

            # Unflatten each model
            result = {}
            for model_id, model_flat in by_model.items():
                result[model_id] = freeze(unflatten_dict(model_flat, sep="/"))
            return result
        else:
            return freeze(unflatten_dict(flat, sep="/"))

    def init(self, key: random.PRNGKey, params: Dict) -> Tuple[Dict, int, int]:
        """
        Initialize LoRA A and B matrices.

        Args:
            key: JAX random key
            params: Full model parameters

        Returns:
            Tuple of (lora_params, total_param_count, num_layers)
        """
        self._lora_config = self._parse_rank_dict(params)
        flat_params = self._flatten_params(params)

        lora_params = {}
        keys = random.split(key, len(self._lora_config) + 1)

        total_lora_params = 0
        num_layers = 0

        for i, (path, (rank, alpha)) in enumerate(self._lora_config.items()):
            if path not in flat_params:
                continue

            param = flat_params[path]
            if param.ndim != 2:
                continue

            out_features, in_features = param.shape
            actual_rank = min(rank, min(in_features, out_features))

            if actual_rank <= 0:
                continue

            k1, k2 = random.split(keys[i])
            std = 1.0 / np.sqrt(in_features)
            lora_A = random.normal(k1, (actual_rank, in_features)) * std
            lora_B = jnp.zeros((out_features, actual_rank))

            lora_params[path + "/lora_A"] = lora_A
            lora_params[path + "/lora_B"] = lora_B

            # Update config with actual rank used
            self._lora_config[path] = (actual_rank, alpha)
            total_lora_params += actual_rank * (in_features + out_features)
            num_layers += 1

        return self._unflatten_params(lora_params), total_lora_params, num_layers

    def merge(self, lora_params: Dict, params: Optional[Dict] = None) -> Dict:
        """
        Merge LoRA into base params: W_new = W + (alpha / rank) * B @ A

        Args:
            lora_params: LoRA parameters (A and B matrices)
            params: Optional full params to merge into (uses stored base if None)

        Returns:
            Merged parameters with same structure as input params
        """
        base = params if params is not None else self._base_params

        flat_params = self._flatten_params(base)
        flat_lora = self._flatten_params(lora_params)

        merged = {}
        for path, param in flat_params.items():
            lora_A_path = path + "/lora_A"
            lora_B_path = path + "/lora_B"

            if lora_A_path in flat_lora and lora_B_path in flat_lora:
                A = flat_lora[lora_A_path]
                B = flat_lora[lora_B_path]
                rank, alpha = self._lora_config.get(path)
                scale = alpha / rank
                merged[path] = param + scale * (B @ A)
            else:
                merged[path] = param

        return self._unflatten_params(merged)

    def get_full_params(self, lora_params: Dict, all_params: Dict) -> Dict:
        """
        Get full params dict with LoRA merged for applicable models.
        Non-LoRA models are passed through unchanged.

        Args:
            lora_params: LoRA parameters
            all_params: Full params dict (may include models without LoRA)

        Returns:
            Full params dict with LoRA merged where applicable
        """
        if not self._is_multi_model:
            return self.merge(lora_params, all_params)

        # Multi-model: merge only for models with LoRA
        result = {}
        merged = self.merge(lora_params)

        for model_id, model_params in all_params.items():
            if model_id in merged:
                result[model_id] = merged[model_id]
            else:
                result[model_id] = model_params

        return result

    # === Training Function ===

    def _make_track_fn(self, compiled_trackers, points_by_tag, tensor_tags, batchsize):
        """Create tracking function with conditional evaluation."""

        def make_conditional_tracker(interval, fn):
            def conditional_fn(params, rng, epoch):
                return jax.lax.cond(
                    epoch % interval == 0,
                    lambda: fn(params, points_by_tag, tensor_tags, batchsize=batchsize, key=rng),
                    lambda: jnp.zeros(batchsize if batchsize else 1),
                )

            return conditional_fn

        conditional_trackers = [make_conditional_tracker(interval, fn) for interval, fn in compiled_trackers]

        def track_fn(params, rng, epoch):
            return [tracker(params, rng, epoch) for tracker in conditional_trackers]

        return track_fn

    def make_train_fn(
        self,
        optimizer,
        compiled_constraints,
        compiled_trackers,
        domain_data,
        epochs: int,
        batchsize: Optional[int],
        constraint_weights,
        learning_rate,
        total_epochs: int,
        all_params: Dict,
    ):
        """
        Create training function that only updates LoRA params.

        Args:
            optimizer: Optax optimizer
            compiled_constraints: List of compiled constraint functions
            compiled_trackers: List of (interval, fn) tuples for tracking
            domain_data: DomainData with points and tensors
            epochs: Number of epochs
            batchsize: Mini-batch size or None
            constraint_weights: WeightSchedule for constraints
            learning_rate: LearningRateSchedule
            total_epochs: Current total epochs (for schedule offset)
            all_params: Full params dict (for non-LoRA models)
        """
        points_by_tag = domain_data.points_by_tag
        tensor_tags = domain_data.tensor_tags
        n_constraints = len(compiled_constraints)
        print_rate = max(1, int(epochs / 100) if epochs < 100_000 else int(epochs / 1000))

        if len(compiled_trackers) > 0:
            track_fn = self._make_track_fn(compiled_trackers, points_by_tag, tensor_tags, batchsize)

        # Capture references for closure
        lora_merge = self.merge
        get_full = self.get_full_params
        base_params = self._base_params
        is_multi = self._is_multi_model

        def loss_fn(lora_params, tag_weights, rng):
            # Merge LoRA and get full params for forward pass
            if is_multi:
                merged_params = get_full(lora_params, all_params)
            else:
                merged_params = lora_merge(lora_params)

            losses = []
            for fn in compiled_constraints:
                residual = fn(merged_params, points_by_tag, tensor_tags, batchsize=batchsize, key=rng)
                mean_squared_residual = jnp.mean(jnp.square(residual))
                losses.append(mean_squared_residual)

            losses = jnp.stack(losses)
            weighted_loss = jnp.dot(tag_weights, losses)
            return weighted_loss, losses

        def print_progress(epoch, individual_losses, track_stats):
            loss_strs = " | ".join([f"C{i}: {float(l):>10.4e}" for i, l in enumerate(individual_losses)])
            if track_stats is not None:
                track_strs = " | ".join([f"T{i}: {float(jnp.mean(l)):>10.4e}" for i, l in enumerate(track_stats)])
                print(f"\rEpoch {int(epoch):>6}/{epochs} | {loss_strs} | {track_strs} [LoRA]", end="\n", flush=True)
            else:
                print(f"\rEpoch {int(epoch):>6}/{epochs} | {loss_strs} [LoRA]", end="\n", flush=True)
            return None

        def train_n_steps(lora_params, opt_state, rng):
            original_lr_dtype = opt_state[-1].hyperparams["step_size"].dtype

            def body_fn(_, carry):
                lora_params, opt_state, rng, total_loss, individual_losses, epoch = carry
                tag_weights = constraint_weights(total_epochs + epoch, individual_losses)

                rng, step_rng = jax.random.split(rng)

                def loss_wrapper(p):
                    return loss_fn(p, tag_weights, step_rng)

                # Track with merged params
                if is_multi:
                    merged_for_track = get_full(lora_params, all_params)
                else:
                    merged_for_track = lora_merge(lora_params)
                track_stats = track_fn(merged_for_track, step_rng, epoch) if len(compiled_trackers) > 0 else None

                (total_loss, individual_losses), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(lora_params)
                updates, opt_state = optimizer.update(grads, opt_state, lora_params, value=total_loss, grad=grads, value_fn=lambda p: loss_wrapper(p)[0])

                lr = learning_rate(total_epochs + epoch, individual_losses)
                opt_state[-1].hyperparams["step_size"] = jnp.asarray(lr, dtype=original_lr_dtype)

                lora_params = optax.apply_updates(lora_params, updates)

                jax.lax.cond(
                    epoch % print_rate == 0,
                    lambda: jax.experimental.io_callback(print_progress, None, epoch, individual_losses, track_stats, ordered=False),
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

                return lora_params, opt_state, rng, total_loss, individual_losses, epoch + 1

            init_carry = (lora_params, opt_state, rng, 0.0, jnp.zeros(n_constraints), 0)
            return jax.lax.fori_loop(0, epochs, body_fn, init_carry)

        return train_n_steps


def create_rank_dict(
    params: Dict,
    rank: int = 8,
    alpha: float = 1.0,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    model_ids: Optional[List[int]] = None,
) -> Dict:
    """
    Helper to create a rank_dict from params structure.

    Args:
        params: Model parameters (single or multi-model)
        rank: Default rank for all layers
        alpha: Default alpha for all layers
        include: Patterns to include (None = all 2D params)
        exclude: Patterns to exclude
        model_ids: For multi-model, which model IDs to apply LoRA to.
                   None means all models.

    Returns:
        rank_dict ready for LoRA()

    Example:
        # Single model
        rank_dict = create_rank_dict(params, rank=8, include=["Dense_0", "Dense_1"])

        # Multi-model, only apply to model 0
        rank_dict = create_rank_dict(params, rank=8, model_ids=[0])
    """
    # Check if multi-model
    is_multi = params and isinstance(next(iter(params.keys())), int)

    if is_multi:
        rank_dict = {}
        for model_id, model_params in params.items():
            # Skip if model_ids specified and this isn't in the list
            if model_ids is not None and model_id not in model_ids:
                continue

            flat = flatten_dict(unfreeze(model_params), sep="/")
            model_rank_dict = {}

            for path, param in flat.items():
                if param.ndim != 2:
                    continue

                should_include = True
                if include is not None:
                    should_include = any(p in path for p in include)
                if exclude is not None and any(p in path for p in exclude):
                    should_include = False

                model_rank_dict[path] = (rank, alpha) if should_include else float("nan")

            if model_rank_dict:
                rank_dict[model_id] = unflatten_dict(model_rank_dict, sep="/")

        return rank_dict
    else:
        # Single model
        flat = flatten_dict(unfreeze(params), sep="/")
        rank_dict = {}

        for path, param in flat.items():
            if param.ndim != 2:
                continue

            should_include = True
            if include is not None:
                should_include = any(p in path for p in include)
            if exclude is not None and any(p in path for p in exclude):
                should_include = False

            rank_dict[path] = (rank, alpha) if should_include else float("nan")

        return unflatten_dict(rank_dict, sep="/")


def print_params_structure(params: Dict, max_depth: int = 4) -> None:
    """
    Pretty print params structure to help create rank_dict.

    Args:
        params: Model parameters
        max_depth: Maximum depth to print
    """

    def _print_tree(d, prefix="", depth=0):
        if depth >= max_depth:
            print(f"{prefix}...")
            return

        if isinstance(d, dict):
            for i, (k, v) in enumerate(d.items()):
                is_last = i == len(d) - 1
                connector = "└── " if is_last else "├── "
                extension = "    " if is_last else "│   "

                if isinstance(v, dict):
                    print(f"{prefix}{connector}{k}/")
                    _print_tree(v, prefix + extension, depth + 1)
                elif hasattr(v, "shape"):
                    shape_str = str(v.shape)
                    dtype_str = str(v.dtype) if hasattr(v, "dtype") else ""
                    print(f"{prefix}{connector}{k}: {shape_str} {dtype_str}")
                else:
                    print(f"{prefix}{connector}{k}: {type(v).__name__}")

    print("Params structure:")
    _print_tree(params)
