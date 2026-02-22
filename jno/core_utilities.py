from typing import List, Callable, Dict, Union
import jax
import jax.numpy as jnp
import numpy as np

from .utils import get_logger
import time
import equinox as eqx
from .trace import (
    Placeholder,
    Variable,
    TensorTag,
    FlaxModule,
    TunableModule,
    TunableModuleCall,
    OperationDef,
    OperationCall,
    Hessian,
    Jacobian,
    BinaryOp,
    Concat,
    FunctionCall,
    Slice,
)
from .trace_evaluator import TraceEvaluator
from .domain import domain


class CoreUtilities:
    """Base class for prediction and utility methods."""

    def __init__(self):
        self.log = get_logger()
        self.dots = []
        return None

    def count(self, models):
        leaves = jax.tree_util.tree_leaves(eqx.filter(models, eqx.is_array))
        return sum(leaf.size for leaf in leaves)

    def _find_tunable_modules(self) -> List[TunableModule]:
        """Find all TunableModules in constraints."""
        tunable = []
        seen = set()

        def visit(node):
            if isinstance(node, TunableModule):
                if id(node) not in seen:
                    seen.add(id(node))
                    tunable.append(node)
            elif isinstance(node, TunableModuleCall):
                # The TunableModuleCall holds reference to TunableModule
                if id(node.model) not in seen:
                    seen.add(id(node.model))
                    tunable.append(node.model)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, FlaxModule):
                # Check if the underlying module is tunable
                pass
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
                visit(node.operation.expr)
                for arg in node.args:
                    visit(arg)
            elif isinstance(node, OperationDef):
                visit(node.expr)
            elif isinstance(node, (Hessian, Jacobian)):
                visit(node.target)
            elif isinstance(node, Slice):
                visit(node.target)

        for constraint in self.constraints:
            visit(constraint)

        return tunable

    def checkpoint(self, models, opt_state, rng, lora_params=None):
        # lazily initialize
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

    def _to_plain_dict(self, logs):
        """Convert lox logs to plain Python dict."""
        if hasattr(logs, "to_dict"):
            return logs.to_dict()
        elif hasattr(logs, "items"):
            return {k: jax.device_get(v) if hasattr(v, "device") else v for k, v in logs.items()}
        else:
            return dict(logs)

    def print_solve_info(self, logs, epochs):
        logs_plain = dict(logs)
        logs = logs_plain
        logs_plain = jax.tree_util.tree_map(jax.device_get, logs_plain)
        self.training_logs.append(logs_plain)

        # Print training phase summary
        final_epoch = logs["epoch"][-1] if "epoch" in logs else epochs
        final_loss = logs["total_loss"][-1] if "total_loss" in logs else 0.0
        final_lr = logs["learning_rate"][-1] if "learning_rate" in logs else 0.0
        training_time = logs.get("training_time", 0.0)
        final_losses = logs["losses"][-1]
        loss_str = " | ".join([f"L{i}: {l:.6e}" for i, l in enumerate(final_losses)])
        self.log.info(f"Epoch {self._total_epochs} -> {self._total_epochs + final_epoch + 1} | Loss: {final_loss:.4e} | LR: {final_lr:.4e} > Constraint losses: {loss_str}")

    def predict(self, points: jnp.ndarray = None, operation: OperationDef = None, context: Dict[str, jnp.ndarray] = None) -> jnp.ndarray:
        """Predict using trained operation.

        Args:
            points: Input points (N, D) or (B, N, D)
            operation: The operation/expression to evaluate (uses main trainable op if None)
            context: Optional context dict to override domain's context
        """
        if operation is None:
            operation = getattr(self, "_main_op", None)
        if operation is None:
            raise ValueError("No operation available for prediction")

        # Auto-wrap raw Placeholder expressions
        if isinstance(operation, Placeholder) and not isinstance(operation, (OperationDef, OperationCall)):
            if hasattr(operation, "_auto_op"):
                operation = operation._auto_op
            else:
                operation = OperationDef(operation)

        # Handle OperationCall by getting its underlying operation
        if isinstance(operation, OperationCall):
            operation = operation.operation

        # Build context from domain
        vars_in_op = operation._collected_vars
        if not vars_in_op:
            raise ValueError("Operation has no input variables")

        tag = vars_in_op[0].tag

        # Start from domain's context
        ctx = {}
        if self.domain is not None and hasattr(self.domain, "context"):
            ctx = dict(self.domain.context)

        # Override with user-provided context entries
        if context is not None:
            ctx.update(context)

        # Ensure points are (B, N, D) shape
        was_2d = False
        if points is not None:
            was_2d = points.ndim == 2
            if was_2d:
                points = points[None, :, :]
            ctx[tag] = points

        # Determine batch size
        batch_sizes = [v.shape[0] for v in ctx.values() if hasattr(v, "shape") and v.ndim >= 1]
        B = max(batch_sizes) if batch_sizes else 1

        evaluator = TraceEvaluator(self.models)

        if B == 1:
            # Single batch: evaluate directly without vmap
            ctx_single = {k: (v[0] if v.ndim >= 2 and v.shape[0] == 1 else v) for k, v in ctx.items()}
            result = evaluator.evaluate(operation.expr, ctx_single)
            result = result[None, ...]
        else:
            # Multiple batches: use vmap
            tag_order = tuple(sorted(ctx.keys()))
            ctx_tuple = tuple(ctx[k] for k in tag_order)

            def eval_batch_tuple(ctx_vals):
                ctx_dict = dict(zip(tag_order, ctx_vals))
                return evaluator.evaluate(operation.expr, ctx_dict)

            ctx_in_axes = tuple(0 if (a.ndim >= 1 and a.shape[0] == B) else None for a in ctx_tuple)
            vmapped_fn = jax.vmap(eval_batch_tuple, in_axes=(ctx_in_axes,))
            result = vmapped_fn(ctx_tuple)

        # Squeeze batch dimension if input was 2D
        if was_2d and result.ndim >= 2:
            result = result[0]

        return result
