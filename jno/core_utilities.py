from typing import List, Callable, Dict, Union
import jax
import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
import graphviz
import time

from .utils import get_logger
from .trace import *
from .trace_evaluator import TraceEvaluator
from .domain import domain


class ErrorMetrics:
    def __init__(self, predict_fn, log=None):
        """
        Args:
            domain: domain object containing points
            predict_fn: function(test_pts, operation) -> predictions
            log: optional logger
        """
        self.predict = predict_fn
        self.log = log

    def set_domain(self, domain):
        self.domain = domain

    # ----------------- Internal helpers -----------------
    def _prepare_points(self, test_pts=None):
        """
        Returns a dict of points per tag and tensor_tags dict
        """
        tensor_tags = {}

        if test_pts is None:
            # Use training domain - check if it's batched
            if hasattr(self.domain, "sampled_points") and self.domain.sampled_points:
                # Check if batched by looking at first sampled_points entry
                first_key = next(iter(self.domain.sampled_points.keys()))
                first_val = self.domain.sampled_points[first_key]
                if first_val.ndim == 3:  # Batched: (B, N, D)
                    # Use all available points from points_by_tag and tile for each batch
                    batch_size = first_val.shape[0]
                    points_by_tag = {}
                    for k, v in self.domain.points_by_tag.items():
                        # Tile points to (B, N, D) - same points for each batch
                        pts = jnp.array(v)
                        len_shape = [batch_size] + pts.ndim * [1]
                        points_by_tag[k] = jnp.tile(pts[None, ...], tuple(len_shape))
                else:  # Not batched: use all available points from points_by_tag
                    points_by_tag = {k: jnp.array(v) for k, v in self.domain.points_by_tag.items()}
            else:
                # No sampled_points, use all available points from points_by_tag
                points_by_tag = {k: jnp.array(v) for k, v in self.domain.points_by_tag.items()}

            if hasattr(self.domain, "tensor_tags"):
                tensor_tags = {k: jnp.asarray(v) for k, v in self.domain.tensor_tags.items()}
        elif isinstance(test_pts, jnp.ndarray):
            points_by_tag = {"all": test_pts}
        elif isinstance(test_pts, dict):
            points_by_tag = test_pts
        elif isinstance(test_pts, domain):
            # Extract sampled points from domain (take first batch if batched)
            # Sample all the needed points first
            test_pts._verbose = False
            if self.domain.sampled_points.keys() != test_pts.sampled_points.keys():
                for sample in self.domain.sample_dict:
                    if sample[0] in test_pts.avaiable_mesh_tags:  # Check if the tag is in the other domain (inference and train domains are of different types)
                        test_pts.variable(*sample)

            points_by_tag = {k: jnp.array(v[0] if v.ndim == 3 else v) for k, v in test_pts.sampled_points.items() if not k.startswith("n_")}
            # Extract tensor tags
            if hasattr(test_pts, "tensor_tags"):
                tensor_tags = {k: jnp.asarray(v) for k, v in test_pts.tensor_tags.items()}

        return points_by_tag, tensor_tags

    def _compute_pointwise(self, predictions, references, test_pts=None):
        """
        Compute pointwise errors per operation, separated by tag.
        Returns: (errors_dict, exacts_dict) where errors[tag][op_idx] -> array of pointwise errors
                 and exacts[tag][op_idx] -> array of exact values
        """

        points_by_tag, tensor_tags = self._prepare_points(test_pts)

        # Check if we have batched data (multiple spatial domains) OR operator learning (single domain, multiple parameters)
        first_key = next(iter(points_by_tag.keys()))
        first_val = points_by_tag[first_key]
        is_batched = first_val.ndim == 3

        # Check for operator learning: single spatial domain but multiple tensor batches
        tensor_batch_sizes = [v.shape[0] for v in tensor_tags.values() if hasattr(v, "shape") and v.ndim > 0]
        tensor_batch_size = min(tensor_batch_sizes) if tensor_batch_sizes else 1
        is_operator_learning = (not is_batched) and tensor_batch_size > 1

        if is_operator_learning:
            # Handle operator learning: single domain, multiple parameter batches
            # Compute errors separately for each parameter value
            errors = {}
            exacts = {}
            all_points = jnp.vstack([val for tag, val in points_by_tag.items() if tag[1] != "_"])

            for batch_idx in range(self.domain.total_samples):
                # Extract tensor_tags for this batch

                batch_tensor_tags = {}
                for k, v in tensor_tags.items():
                    if "_" not in k:
                        batch_tensor_tags[k] = v[batch_idx : batch_idx + 1]
                    else:
                        batch_tensor_tags[k] = v

                # Compute for this batch
                batch_exactions = [self.predict(all_points, operation=ref, tensor_tags=batch_tensor_tags).flatten() for ref in references]
                # batch_exactions = [self._call_reference(ref, all_points, batch_tensor_tags) for ref in references]
                # Flatten batch_exactions in case they have extra dimensions from broadcasting with tensor_tags
                # batch_exactions = [jnp.ravel(ex) for ex in batch_exactions]
                preds = [self.predict(all_points, operation=op, tensor_tags=batch_tensor_tags).flatten() for op in predictions]

                # Store errors per tag
                start = 0
                for tag, pts in points_by_tag.items():
                    n_pts = pts.shape[0]
                    if tag not in errors:
                        errors[tag] = {}
                        exacts[tag] = {}
                    for op_idx, (pred, exact) in enumerate(zip(preds, batch_exactions)):
                        diff = pred[start : start + n_pts] - exact[start : start + n_pts]
                        exact_vals = exact[start : start + n_pts]
                        if op_idx not in errors[tag]:
                            errors[tag][op_idx] = []
                            exacts[tag][op_idx] = []
                        errors[tag][op_idx].append(diff)
                        exacts[tag][op_idx].append(exact_vals)
                    start += n_pts

            # Stack batch results to get (B, N) arrays
            for tag in errors:
                for op_idx in errors[tag]:
                    errors[tag][op_idx] = jnp.stack(errors[tag][op_idx])  # (B, N)
                    exacts[tag][op_idx] = jnp.stack(exacts[tag][op_idx])  # (B, N)

        elif is_batched:
            # Handle batched evaluation: (B, N, D) - multiple spatial domains
            batch_size = first_val.shape[0]
            errors = {}
            exacts = {}

            for batch_idx in range(batch_size):
                # Extract points for this batch
                batch_points_by_tag = {k: v[batch_idx] for k, v in points_by_tag.items()}
                batch_all_points = jnp.vstack(list(batch_points_by_tag.values()))

                # Extract tensor_tags for this batch
                batch_tensor_tags = {k: v[batch_idx : batch_idx + 1] for k, v in tensor_tags.items()} if tensor_tags else {}

                # Compute for this batch
                batch_exactions = [self.predict(batch_all_points, operation=ref, tensor_tags=batch_tensor_tags if batch_tensor_tags else None).flatten() for ref in references]
                preds = [self.predict(batch_all_points, operation=op, tensor_tags=batch_tensor_tags if batch_tensor_tags else None).flatten() for op in predictions]

                # Store errors per tag
                start = 0
                for tag, pts in batch_points_by_tag.items():
                    n_pts = pts.shape[0]
                    if tag not in errors:
                        errors[tag] = {}
                        exacts[tag] = {}
                    for op_idx, (pred, exact) in enumerate(zip(preds, batch_exactions)):
                        diff = pred[start : start + n_pts] - exact[start : start + n_pts]
                        exact_vals = exact[start : start + n_pts]
                        if op_idx not in errors[tag]:
                            errors[tag][op_idx] = []
                            exacts[tag][op_idx] = []
                        errors[tag][op_idx].append(diff)
                        exacts[tag][op_idx].append(exact_vals)
                    start += n_pts

            # Concatenate batch results
            for tag in errors:
                for op_idx in errors[tag]:
                    errors[tag][op_idx] = jnp.concatenate(errors[tag][op_idx])
                    exacts[tag][op_idx] = jnp.concatenate(exacts[tag][op_idx])
        else:
            # Non-batched evaluation
            all_points = jnp.vstack([val for tag, val in points_by_tag.items() if tag[1] != "_"])
            exactions = [self.predict(all_points, operation=ref, tensor_tags=tensor_tags if tensor_tags else None).flatten() for ref in references]
            preds = [self.predict(all_points, operation=op, tensor_tags=tensor_tags if tensor_tags else None).flatten() for op in predictions]

            errors = {}
            exacts = {}
            start = 0
            for tag, pts in points_by_tag.items():
                n_pts = pts.shape[0]
                errors[tag] = {}
                exacts[tag] = {}
                for op_idx, (pred, exact) in enumerate(zip(preds, exactions)):
                    diff = pred[start : start + n_pts] - exact[start : start + n_pts]
                    exact_vals = exact[start : start + n_pts]
                    errors[tag][op_idx] = diff
                    exacts[tag][op_idx] = exact_vals
                start += n_pts

        return errors, exacts

    # ----------------- Metric computations -----------------
    def l2(self, predictions, references, test_pts=None, relative=False):
        errors, exacts = self._compute_pointwise(predictions, references, test_pts)
        results = {}
        for tag, ops in errors.items():
            if ops[0].shape != (0,):
                results[tag] = []
                for op_idx, diff in ops.items():
                    # Check if we have multiple batches (B, N) vs single batch (N,)
                    if diff.ndim == 2:  # (B, N) - compute per batch
                        batch_results = []
                        for batch_idx in range(diff.shape[0]):
                            val = jnp.sqrt(jnp.mean(diff[batch_idx] ** 2))
                            if relative:
                                exact_vals = exacts[tag][op_idx][batch_idx]
                                val /= jnp.maximum(jnp.sqrt(jnp.mean(exact_vals**2)), 1e-10)
                            batch_results.append(val)
                        results[tag].append(batch_results)
                    if diff.ndim == 2:  # (B, N) - compute per batch
                        batch_results = []
                        for batch_idx in range(diff.shape[0]):
                            val = jnp.sqrt(jnp.mean(diff[batch_idx] ** 2))
                            if relative:
                                exact_vals = exacts[tag][op_idx][batch_idx]
                                val /= jnp.maximum(jnp.sqrt(jnp.mean(exact_vals**2)), 1e-10)
                            batch_results.append(val)
                        results[tag].append(batch_results)
                    else:  # (N,) - single result
                        val = jnp.sqrt(jnp.mean(diff**2))
                        if relative:
                            exact_vals = exacts[tag][op_idx]
                            val /= jnp.maximum(jnp.sqrt(jnp.mean(exact_vals**2)), 1e-10)
                        results[tag].append(val)
        self._log_results(results, "L2", relative)
        return results

    def l1(self, predictions, references, test_pts=None, relative=False):
        errors, exacts = self._compute_pointwise(predictions, references, test_pts)
        results = {}
        for tag, ops in errors.items():
            if ops[0].shape != (0,):
                results[tag] = []
                for op_idx, diff in ops.items():
                    # Check if we have multiple batches (B, N) vs single batch (N,)
                    if diff.ndim == 2:  # (B, N) - compute per batch
                        batch_results = []
                        for batch_idx in range(diff.shape[0]):
                            val = jnp.mean(jnp.abs(diff[batch_idx]))
                            if relative:
                                exact_vals = exacts[tag][op_idx][batch_idx]
                                val /= jnp.maximum(jnp.mean(jnp.abs(exact_vals)), 1e-10)
                            batch_results.append(val)
                        results[tag].append(batch_results)
                    else:  # (N,) - single result
                        val = jnp.mean(jnp.abs(diff))
                        if relative:
                            exact_vals = exacts[tag][op_idx]
                            val /= jnp.maximum(jnp.mean(jnp.abs(exact_vals)), 1e-10)
                        results[tag].append(val)
        self._log_results(results, "L1", relative)
        return results

    def linf(self, predictions, references, test_pts=None, relative=False):
        errors, exacts = self._compute_pointwise(predictions, references, test_pts)
        results = {}
        for tag, ops in errors.items():
            if ops[0].shape != (0,):
                results[tag] = []
                for op_idx, diff in ops.items():
                    val = jnp.max(jnp.abs(diff))
                    if relative:
                        exact_vals = exacts[tag][op_idx]
                        val /= jnp.maximum(jnp.max(jnp.abs(exact_vals)), 1e-10)
                    results[tag].append(val)
        self._log_results(results, "Linf", relative)
        return results

    def rmse(self, predictions, references, test_pts=None, relative=None):
        # root mean squared error, same as l2 but without sqrt(…)?
        return self.l2(predictions, references, test_pts, relative=False)

    # ----------------- All metrics -----------------
    def all(self, predictions, references, test_pts=None, relative=True, save_path="./", plot=False):
        """
        Compute all metrics at once, return pointwise errors, optionally save and plot.
        """

        pointwise, _ = self._compute_pointwise(predictions, references, test_pts)

        metrics = ["l2", "l1", "linf", "rmse"]
        results = {}

        for metric in metrics:
            func = getattr(self, metric)
            results[metric] = func(predictions, references, test_pts=test_pts, relative=relative)

        if save_path is not None:
            # flatten pointwise errors for npz
            flat = {}
            for tag, ops in pointwise.items():
                for idx, err in ops.items():
                    flat[f"{tag}_op{idx}"] = np.array(err)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path + "errors.npz", **flat)

        if plot:
            self._plot_pointwise_errors(pointwise, save_path)

        return results

    # ----------------- Logging -----------------
    def _log_results(self, results, metric, relative):
        if self.log is None:
            return
        suffix = " (relative)" if relative else ""
        for tag, vals in results.items():
            for idx, val in enumerate(vals):
                # Check if val is a list (multiple batches) or scalar
                if isinstance(val, list):
                    for batch_idx, batch_val in enumerate(val):
                        self.log.info(f"{tag} - op{idx} - batch{batch_idx} - {metric}{suffix}: {batch_val:.6e}")
                else:
                    self.log.info(f"{tag} - op{idx} - {metric}{suffix}: {val:.6e}")

    # ----------------- Plotting -----------------
    def _plot_pointwise_errors(self, pointwise_errors, save_path):
        for tag, ops in pointwise_errors.items():
            pts = np.array(self.domain.points_by_tag[tag])
            for op_idx, err in ops.items():
                plt.figure()
                plt.scatter(pts[:, 0], pts[:, 1], c=np.array(err), s=1, cmap="hot")
                plt.colorbar(label="Error")
                plt.title(f"{tag} - op{op_idx} - Pointwise Error")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.savefig(f"{save_path}/{tag}_{op_idx}.png")


class CoreUtilities:
    """Base class for analysis, plotting, and prediction methods."""

    def __init__(self):
        self.log = get_logger()
        self.dots = []
        self.errors = ErrorMetrics(self.predict, log=self.log)
        return None

    def count(self, params):
        leaves, _ = jax.tree_util.tree_flatten(params)
        return sum(leaf.size for leaf in leaves if jnp.issubdtype(leaf.dtype, jnp.floating))

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
            elif isinstance(node, (Gradient, Laplacian, Hessian)):
                visit(node.target)
            elif isinstance(node, Slice):
                visit(node.target)

        for constraint in self.constraints:
            visit(constraint)

        return tunable

    def checkpoint(self, params, opt_state, rng, lora_params):
        # lazily initialize
        import copy

        payload = {
            "step": int(self._total_epochs),
            "time": time.time(),
            "params": copy.deepcopy(params),
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

    def predict(self, points: jnp.ndarray = None, operation: OperationDef = None, tensor_tags: Dict[str, jnp.ndarray] = None) -> jnp.ndarray:
        """Predict using trained operation.

        Args:
            points: Input points (N, D) or (B, N, D)
            operation: The operation/expression to evaluate (uses main trainable op if None)
            tensor_tags: Optional tensor tags to pass to evaluation (overrides domain's)
        """
        if operation is None:
            operation = getattr(self, "_main_op", None)
        if operation is None:
            raise ValueError("No operation available for prediction")

        # Auto-wrap raw Placeholder expressions
        if isinstance(operation, Placeholder) and not isinstance(operation, (OperationDef, OperationCall)):
            # Check for cached auto-op first
            if hasattr(operation, "_auto_op"):
                operation = operation._auto_op
            else:
                operation = OperationDef(operation)

        # Handle OperationCall by getting its underlying operation
        if isinstance(operation, OperationCall):
            operation = operation.operation

        # Build points_by_tag - use first collected variable's tag
        vars_in_op = operation._collected_vars
        if not vars_in_op:
            raise ValueError("Operation has no input variables")

        tag = vars_in_op[0].tag
        points_by_tag = self.domain.points_by_tag
        if self.domain is not None:
            for k, v in self.domain.sampled_points.items():
                points_by_tag[k] = v

        # Ensure points are (B, N, D) shape
        if points is not None:
            was_2d = points.ndim == 2
            if was_2d:
                points = points[None, :, :]  # (N, D) -> (1, N, D)
            points_by_tag[tag] = points

        # Get tensor tags from domain or use provided ones
        if tensor_tags is None:
            tensor_tags = {}
            if self.domain is not None and hasattr(self.domain, "tensor_tags"):
                for name, tensor in self.domain.tensor_tags.items():
                    tensor_tags[name] = tensor

        # Check if we have operator learning scenario: single domain, multiple parameters
        tensor_batch_sizes = [v.shape[0] for v in tensor_tags.values() if hasattr(v, "shape") and v.ndim > 0]
        tensor_batch_size = min(tensor_batch_sizes) if tensor_batch_sizes else 1

        if points is None:
            # Multiple batches: use vmap
            evaluator = TraceEvaluator(self.params, self.layer_info)

            def eval_single(tens_dict_single):
                return evaluator.evaluate(operation.expr, {}, {}, tens_dict_single)

            tensor_order = tuple(tensor_tags.keys())
            tensors_tuple = tuple(tensor_tags.get(tag) for tag in tensor_order)

            def eval_batch_tuple(tensor_vals):
                tens_dict = {tag: tens for tag, tens in zip(tensor_order, tensor_vals)}
                return eval_single(tens_dict)

            tensors_in_axes = tuple(0 if t.ndim > 0 else None for t in tensors_tuple)

            vmapped_fn = jax.vmap(eval_batch_tuple, in_axes=(tensors_in_axes,))
            result = vmapped_fn(tensors_tuple)

            return result

        if points.shape[0] == 1 and tensor_batch_size == 1:
            # Single batch for both: evaluate directly without vmap
            points_single = {tag: pts[0] for tag, pts in points_by_tag.items()}
            tensor_tags_single = {k: v[0] if v.ndim > 0 and v.shape[0] == 1 else v for k, v in tensor_tags.items()}
            evaluator = TraceEvaluator(self.params, self.layer_info)
            result = evaluator.evaluate(operation.expr, points_single, {}, tensor_tags_single)
            # Result is (N,) or (N, K), wrap in batch dimension
            result = result[None, ...]  # (N, ...) -> (1, N, ...)
        elif points.shape[0] == 1 and tensor_batch_size > 1:
            # Operator learning: single domain, multiple parameters - vmap over tensors only
            points_single = {tag: pts[0] for tag, pts in points_by_tag.items()}  # Squeeze to (N, D)
            evaluator = TraceEvaluator(self.params, self.layer_info)

            def eval_single(tens_dict_single):
                return evaluator.evaluate(operation.expr, points_single, {}, tens_dict_single)

            # Convert tensor_tags to tuple for vmapping over batch dimension
            tensor_order = tuple(sorted(tensor_tags.keys()))
            tensors_tuple = tuple(tensor_tags[k] for k in tensor_order)

            def eval_with_dict(tensor_vals):
                tens_dict = dict(zip(tensor_order, tensor_vals))
                return eval_single(tens_dict)

            # Vmap only over tensor dimension (points are broadcast)
            vmapped_fn = jax.vmap(eval_with_dict, in_axes=(tuple(0 if t.ndim > 0 else None for t in tensors_tuple),))
            result = vmapped_fn(tensors_tuple)  # Result is (B, N, ...)

        else:
            # Multiple batches: use vmap
            evaluator = TraceEvaluator(self.params, self.layer_info)

            def eval_single(pts_dict_single, tens_dict_single):
                return evaluator.evaluate(operation.expr, pts_dict_single, {}, tens_dict_single)

            # Convert to tuples for vmapping
            tag_order = tuple(points_by_tag.keys())
            points_tuple = tuple(points_by_tag[tag] for tag in tag_order)

            tensor_order = tuple(tensor_tags.keys())
            tensors_tuple = tuple(tensor_tags.get(tag, jnp.array(0.0)) for tag in tensor_order)

            def eval_batch_tuple(points_vals, tensor_vals):
                pts_dict = {tag: pts for tag, pts in zip(tag_order, points_vals)}
                tens_dict = {tag: tens for tag, tens in zip(tensor_order, tensor_vals)}
                return eval_single(pts_dict, tens_dict)

            points_in_axes = tuple(0 for _ in points_tuple)
            tensors_in_axes = tuple(0 if t.ndim > 0 else None for t in tensors_tuple)

            vmapped_fn = jax.vmap(eval_batch_tuple, in_axes=(points_in_axes, tensors_in_axes))
            result = vmapped_fn(points_tuple, tensors_tuple)

        # Squeeze batch dimension if input was 2D
        if was_2d and result.ndim >= 2:
            result = result[0]  # (1, N, ...) -> (N, ...)

        return result

    def _plot_operator_learning(self, operation, points, tensor_tags, tensor_batch_size, title, exact):
        """Create subplots for each tensor batch in operator learning scenarios."""
        import matplotlib.pyplot as plt
        import numpy as np

        points_jax = jnp.array(points)
        dim = points.shape[1]

        if dim != 2:
            raise ValueError(f"Plotting operator learning only supported for 2D domains, got {dim}D")

        # Create subplots - one for each parameter batch
        ncols = min(3, tensor_batch_size)
        nrows = (tensor_batch_size + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes = axes.flatten()

        # Get tensor tag names and values for titles
        tensor_names = list(tensor_tags.keys())

        for batch_idx in range(tensor_batch_size):
            ax = axes[batch_idx]

            # Extract batch-specific tensor_tags
            batch_tensor_tags = {k: v[batch_idx : batch_idx + 1] for k, v in tensor_tags.items()}

            # Predict for this batch
            pred = np.array(self.predict(points_jax, operation=operation, tensor_tags=batch_tensor_tags))
            if exact is not None:
                exact_pts = np.array(self.predict(points_jax, operation=exact, tensor_tags=batch_tensor_tags))
            if pred.ndim > 1:
                pred = pred[:, 0]
                exact_pts = exact_pts[:, 0]

            # Create scatter plot
            if exact is None:
                if not self.domain._is_time_dependent:
                    scatter = ax.scatter(points[:, 0], points[:, 1], c=pred, cmap="viridis", s=5)
                else:
                    scatter = ax.scatter(points[:, 1], pred, s=5)
            else:
                scatter = ax.scatter(points[:, 0], points[:, 1], c=pred - exact_pts, cmap="viridis", s=5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            # ax.set_aspect("equal")

            # Create title with parameter values
            # param_str = ", ".join([f"{k}={v[batch_idx].item():.2f}" for k, v in tensor_tags.items()])
            plt.colorbar(scatter, ax=ax)

        # Hide unused subplots
        for idx in range(tensor_batch_size, len(axes)):
            axes[idx].axis("off")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        return fig

    def plot(self, operation: OperationDef = None, tag: str = None, title: str = "", exact: Callable = None, test_pts: Union[jnp.ndarray, domain] = None, convex: bool = True):
        """Plot the solution over the domain.

        Args:
            operation: Operation to plot (uses main trainable op if None)
            tag: domain tag to use for points (e.g., 'interior')
            title: Plot title
            exact: Optional exact solution function f(points) -> values
            convex: If True, interpolate between points and show actual nodes as small grey points
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if operation is None:
            operation = getattr(self, "_main_op", None)
        if operation is None:
            raise ValueError("No operation available for plotting")

        # Auto-wrap raw Placeholder expressions
        if isinstance(operation, Placeholder) and not isinstance(operation, (OperationDef, OperationCall)):
            if hasattr(operation, "_auto_op"):
                operation = operation._auto_op
            else:
                operation = OperationDef(operation)

        if isinstance(operation, OperationCall):
            operation = operation.operation

        # Get tag from operation or use provided
        if tag is None:
            vars_in_op = operation._collected_vars
            if vars_in_op:
                tag = vars_in_op[0].tag
            else:
                tag = list(self.domain.points_by_tag.keys())[0]

        # Get points and tensor_tags
        points_by_tag, tensor_tags = self.errors._prepare_points(test_pts)

        # Check for operator learning: single domain but multiple tensor batches
        tensor_batch_sizes = [v.shape[0] for v in tensor_tags.values() if hasattr(v, "shape") and v.ndim > 0]
        tensor_batch_size = min(tensor_batch_sizes) if tensor_batch_sizes else 1

        # For batched domains or operator learning, handle multiple batches
        if tag in points_by_tag:
            pts = points_by_tag[tag]
            if pts.ndim == 3:  # Spatially batched domains
                points = np.asarray(pts[0])
                # Extract first batch of tensor_tags too
                tensor_tags = {k: v[0:1] for k, v in tensor_tags.items()} if tensor_tags else None
                tensor_batch_size = 1  # Override since we're only plotting first spatial batch
            else:
                points = np.asarray(pts)
        else:
            if tag in self.domain.points_by_tag:
                points = np.asarray(self.domain.points_by_tag[tag])
            else:
                points = np.asarray(self.domain.tensor_tags[tag])

        # If operator learning with multiple parameter batches, create subplots
        if tensor_batch_size > 1:
            return self._plot_operator_learning(operation, points, tensor_tags, tensor_batch_size, title, exact)

        # Single batch plotting (original behavior)
        points_jax = jnp.array(points)

        extra_tags = {**tensor_tags, **points_by_tag} if tensor_tags is not None else {**points_by_tag}
        pred = np.array(self.predict(points_jax, operation=operation, tensor_tags=extra_tags))
        if pred.ndim > 1:
            pred = np.squeeze(pred)

        dim = points.shape[1]

        is_time_dependent = False
        if isinstance(test_pts, domain):
            is_time_dependent = test_pts._is_time_dependent

        if dim == 2:

            if self.domain._is_time_dependent or is_time_dependent:
                # 1D time dependent domains

                plt.figure(figsize=(8, 6))
                plt.scatter(points[:, 1], pred, s=5)
                return fig

            if convex:
                from matplotlib.tri import Triangulation

                tri = Triangulation(points[:, 0], points[:, 1])

            if exact is not None:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                if convex:
                    # Interpolated plot for predicted
                    tpc1 = axes[0].tripcolor(tri, pred, cmap="viridis", shading="gouraud")
                    axes[0].scatter(points[:, 0], points[:, 1], c="grey", s=1, alpha=0.3, zorder=5)
                    plt.colorbar(tpc1, ax=axes[0])
                else:
                    scatter1 = axes[0].scatter(points[:, 0], points[:, 1], c=pred, cmap="viridis", s=5)
                    plt.colorbar(scatter1, ax=axes[0])

                axes[0].set_xlabel("x")
                axes[0].set_ylabel("y")
                axes[0].set_title("Predicted")
                axes[0].set_aspect("equal")

                exact_vals = np.array(self.predict(points_jax, operation=exact, tensor_tags=tensor_tags))
                if exact_vals.ndim > 1:
                    exact_vals = exact_vals[:, 0]

                if convex:
                    # Interpolated plot for exact
                    tpc2 = axes[1].tripcolor(tri, exact_vals, cmap="viridis", shading="gouraud")
                    axes[1].scatter(points[:, 0], points[:, 1], c="grey", s=1, alpha=0.3, zorder=5)
                    plt.colorbar(tpc2, ax=axes[1])
                else:
                    scatter2 = axes[1].scatter(points[:, 0], points[:, 1], c=exact_vals, cmap="viridis", s=5)
                    plt.colorbar(scatter2, ax=axes[1])

                axes[1].set_xlabel("x")
                axes[1].set_ylabel("y")
                axes[1].set_title("Exact")
                axes[1].set_aspect("equal")
            else:
                fig, ax = plt.subplots(figsize=(8, 6))

                if convex:
                    tpc = ax.tripcolor(tri, pred, cmap="viridis", shading="gouraud")
                    ax.scatter(points[:, 0], points[:, 1], c="grey", s=1, alpha=0.3, zorder=5)
                    plt.colorbar(tpc, ax=ax, label="u(x, y)")
                else:
                    scatter = ax.scatter(points[:, 0], points[:, 1], c=pred, cmap="viridis", s=5)
                    plt.colorbar(scatter, ax=ax, label="u(x, y)")

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect("equal")

            plt.suptitle(title)
            plt.tight_layout()

        elif dim == 1:
            if exact is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                exact_vals = np.array(exact(points_jax))
                if exact_vals.ndim > 1:
                    exact_vals = exact_vals[:, 0]
                order = np.argsort(points[:, 0])
                ax.plot(points[order, 0], exact_vals[order], "b-", label="Exact", linewidth=2)
                ax.plot(points[order, 0], pred[order], "r--", label="Predicted", linewidth=2)

                if convex:
                    # Show actual nodes as small grey points
                    ax.scatter(points[:, 0], exact_vals, c="grey", s=1, alpha=0.3, zorder=5)
                    ax.scatter(points[:, 0], pred, c="grey", s=1, alpha=0.3, zorder=5)

                ax.legend()
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                order = np.argsort(points[:, 0])
                ax.plot(points[order, 0], pred[order], "b-", linewidth=2)

                if convex:
                    # Show actual nodes as small grey points
                    ax.scatter(points[:, 0], pred, c="grey", s=1, alpha=0.3, zorder=5)

            ax.set_xlabel("x")
            ax.set_ylabel("u(x)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        elif dim == 3:
            # 3D: slice through z-dimension and show 9 slices in 3x3 grid
            z_coords = points[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            z_slices = np.linspace(z_min, z_max, 9)

            if exact is not None:
                # Show predicted and exact side by side
                fig, axes = plt.subplots(3, 6, figsize=(18, 9))
                exact_vals = np.array(exact(points_jax))
                if exact_vals.ndim > 1:
                    exact_vals = exact_vals[:, 0]

                for i, z_slice in enumerate(z_slices):
                    row = i // 3
                    col_base = (i % 3) * 2

                    # Find points near this z-slice
                    tolerance = (z_max - z_min) / 20
                    mask = np.abs(z_coords - z_slice) < tolerance
                    slice_points = points[mask]
                    slice_pred = pred[mask]
                    slice_exact = exact_vals[mask]

                    if len(slice_points) > 0:
                        ax_pred = axes[row, col_base]
                        ax_exact = axes[row, col_base + 1]

                        if convex and len(slice_points) >= 3:
                            from matplotlib.tri import Triangulation

                            try:
                                tri = Triangulation(slice_points[:, 0], slice_points[:, 1])
                                # Predicted
                                tpc_pred = ax_pred.tripcolor(tri, slice_pred, cmap="viridis", shading="gouraud")
                                ax_pred.scatter(slice_points[:, 0], slice_points[:, 1], c="grey", s=1, alpha=0.3, zorder=5)
                                plt.colorbar(tpc_pred, ax=ax_pred)
                                # Exact
                                tpc_exact = ax_exact.tripcolor(tri, slice_exact, cmap="viridis", shading="gouraud")
                                ax_exact.scatter(slice_points[:, 0], slice_points[:, 1], c="grey", s=1, alpha=0.3, zorder=5)
                                plt.colorbar(tpc_exact, ax=ax_exact)
                            except Exception:
                                # Fallback to scatter if triangulation fails
                                scatter = ax_pred.scatter(slice_points[:, 0], slice_points[:, 1], c=slice_pred, cmap="viridis", s=5)
                                plt.colorbar(scatter, ax=ax_pred)
                                scatter = ax_exact.scatter(slice_points[:, 0], slice_points[:, 1], c=slice_exact, cmap="viridis", s=5)
                                plt.colorbar(scatter, ax=ax_exact)
                        else:
                            # Predicted
                            scatter = ax_pred.scatter(slice_points[:, 0], slice_points[:, 1], c=slice_pred, cmap="viridis", s=5)
                            plt.colorbar(scatter, ax=ax_pred)
                            # Exact
                            scatter = ax_exact.scatter(slice_points[:, 0], slice_points[:, 1], c=slice_exact, cmap="viridis", s=5)
                            plt.colorbar(scatter, ax=ax_exact)

                        ax_pred.set_xlabel("x")
                        ax_pred.set_ylabel("y")
                        ax_pred.set_title(f"z={z_slice:.2f}")
                        ax_pred.set_aspect("equal")

                        ax_exact.set_xlabel("x")
                        ax_exact.set_ylabel("y")
                        ax_exact.set_title(f"z={z_slice:.2f} (exact)")
                        ax_exact.set_aspect("equal")
                    else:
                        axes[row, col_base].axis("off")
                        axes[row, col_base + 1].axis("off")

                plt.suptitle(f"{title} - Predicted (left) vs Exact (right)")
            else:
                # Show only predicted
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))

                for i, z_slice in enumerate(z_slices):
                    row = i // 3
                    col = i % 3

                    # Find points near this z-slice
                    tolerance = (z_max - z_min) / 20
                    mask = np.abs(z_coords - z_slice) < tolerance
                    slice_points = points[mask]
                    slice_pred = pred[mask]

                    if len(slice_points) > 0:
                        ax = axes[row, col]

                        if convex and len(slice_points) >= 3:
                            from matplotlib.tri import Triangulation

                            try:
                                tri = Triangulation(slice_points[:, 0], slice_points[:, 1])
                                tpc = ax.tripcolor(tri, slice_pred, cmap="viridis", shading="gouraud")
                                ax.scatter(slice_points[:, 0], slice_points[:, 1], c="grey", s=1, alpha=0.3, zorder=5)
                                plt.colorbar(tpc, ax=ax)
                            except Exception:
                                # Fallback to scatter if triangulation fails
                                scatter = ax.scatter(slice_points[:, 0], slice_points[:, 1], c=slice_pred, cmap="viridis", s=5)
                                plt.colorbar(scatter, ax=ax)
                        else:
                            scatter = ax.scatter(slice_points[:, 0], slice_points[:, 1], c=slice_pred, cmap="viridis", s=5)
                            plt.colorbar(scatter, ax=ax)

                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_title(f"z={z_slice:.2f}")
                        ax.set_aspect("equal")
                    else:
                        axes[row, col].axis("off")

                plt.suptitle(title)

            plt.tight_layout()

        else:
            raise ValueError(f"Plotting not supported for {dim}D domains")

        return fig

    def analyze(self, operation: OperationDef = None, sample_points: jnp.ndarray = None, tensor_tags: Dict[str, jnp.ndarray] = None, save_to: str = None, visualize: bool = False):
        """Analyze the JAX computation graph for an operation.

        Uses jax.make_jaxpr to show all operations and tensor shapes in the forward pass.

        Args:
            operation: Operation to analyze (uses main trainable op if None)
            sample_points: Sample input points for tracing. If None, uses 10 random points from domain.
            tensor_tags: Optional tensor tags dict. If None, uses domain's tensor_tags.
            save_to: Optional file path to save JAXPR text (e.g., "jaxpr.txt")
            visualize: If True, generates a DOT graph and saves as PNG (requires graphviz)

        Returns:
            The JAXPR object (also prints it)
        """
        if operation is None:
            operation = getattr(self, "_main_op", None)
        if operation is None:
            raise ValueError("No operation available for analysis")

        # Auto-wrap raw Placeholder expressions
        if isinstance(operation, Placeholder) and not isinstance(operation, (OperationDef, OperationCall)):
            if hasattr(operation, "_auto_op"):
                operation = operation._auto_op
            else:
                operation = OperationDef(operation)

        if isinstance(operation, OperationCall):
            operation = operation.operation

        # Get sample points
        if sample_points is None:
            # Get points from domain
            vars_in_op = operation._collected_vars
            if vars_in_op:
                tag = vars_in_op[0].tag
                if tag in self.domain.points_by_tag:
                    all_pts = self.domain.points_by_tag[tag]
                    sample_points = jnp.array(all_pts)  # First 10 points

        # Get tensor tags
        if tensor_tags is None:
            tensor_tags = {}
            if hasattr(self.domain, "tensor_tags"):
                for name, tensor in self.domain.tensor_tags.items():
                    # Use first batch element for tracing
                    tensor_tags[name] = jnp.array(tensor)

        # Define forward pass for tracing
        def forward_pass(points, *tag_values):
            tags_dict = {k: v for k, v in zip(tensor_tags.keys(), tag_values)}
            return self.predict(points, operation=operation, tensor_tags=tags_dict)

        # Build args for make_jaxpr
        tag_arrays = tuple(tensor_tags.values())

        header = []
        header.append("=" * 70)
        header.append("JAX Computation Graph (JAXPR)")
        header.append("=" * 70)
        header.append(f"Operation: {operation}")
        header.append(f"Sample points shape: {sample_points.shape}")
        if tensor_tags:
            header.append(f"Tensor tags: {list(tensor_tags.keys())}")
        header.append("-" * 70)

        jaxpr = jax.make_jaxpr(forward_pass)(sample_points, *tag_arrays)
        jaxpr_str = str(jaxpr)

        # Print to console
        self.log.info("\n" + "\n".join(header))
        self.log.info(jaxpr_str)

        # Save to file if requested
        if save_to:
            with open(save_to, "w") as f:
                f.write("\n".join(header) + "\n")
                f.write(jaxpr_str)
            self.log.info(f"\nSaved JAXPR to: {save_to}")

        # Visualize as graph if requested
        if visualize:
            self._visualize_jaxpr(jaxpr, save_to or "jaxpr")

        return jaxpr

    def _visualize_jaxpr(self, jaxpr, base_name: str = "jaxpr"):
        """Generate a DOT graph visualization of the JAXPR.

        Args:
            jaxpr: The closed JAXPR to visualize
            base_name: Base filename for output (without extension)
        """
        try:
            import graphviz
        except ImportError:
            self.log.info("\nTo visualize JAXPR as a graph, install graphviz:")
            self.log.info("  pip install graphviz")
            self.log.info("  Also install Graphviz system package: https://graphviz.org/download/")
            return

        # Friendly variable names (Greek letters + common math symbols)
        friendly_names = [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
            "lambda",
            "mu",
            "nu",
            "xi",
            "omicron",
            "pi",
            "rho",
            "sigma",
            "tau",
            "upsilon",
            "phi",
            "chi",
            "psi",
            "omega",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
        ]

        dot = graphviz.Digraph(comment="JAXPR Computation Graph")
        dot.attr(rankdir="TB", fontsize="10", dpi="150")
        dot.attr("node", shape="box", fontsize="9")

        # Extract the jaxpr from ClosedJaxpr
        inner_jaxpr = jaxpr.jaxpr
        consts = jaxpr.consts

        # Build a mapping from original var to node ID and friendly name
        var_to_node_id = {}  # Maps var key -> simple node ID for graphviz
        var_to_label = {}  # Maps var key -> display label
        node_counter = 0
        name_idx = 0

        def get_var_key(var):
            """Get a unique key for a variable."""
            return id(var)

        def register_var(var, friendly_name, shape_str):
            """Register a variable with its node ID and label."""
            nonlocal node_counter
            var_key = get_var_key(var)
            node_id = f"n{node_counter}"
            node_counter += 1
            var_to_node_id[var_key] = node_id
            var_to_label[var_key] = f"{friendly_name}\\n{shape_str}"
            return node_id

        def get_node_id(var):
            """Get the node ID for a variable."""
            return var_to_node_id.get(get_var_key(var))

        def get_shape_str(var):
            """Get shape string for a variable."""
            if hasattr(var, "aval") and hasattr(var.aval, "shape"):
                return str(var.aval.shape)
            return "scalar"

        def get_next_friendly_name():
            """Get the next available friendly name."""
            nonlocal name_idx
            if name_idx < len(friendly_names):
                name = friendly_names[name_idx]
            else:
                name = f"v{name_idx}"
            name_idx += 1
            return name

        # Register constants (weights/biases)
        weight_idx = 0
        bias_idx = 0
        for const_var, const_val in zip(inner_jaxpr.constvars, consts):
            shape = const_val.shape if hasattr(const_val, "shape") else ()
            if len(shape) == 2:
                name = f"W{weight_idx}"
                weight_idx += 1
            elif len(shape) == 1:
                name = f"b{bias_idx}"
                bias_idx += 1
            else:
                name = f"const{node_counter}"
            register_var(const_var, name, str(shape))

        # Register input variables
        for i, invar in enumerate(inner_jaxpr.invars):
            shape_str = get_shape_str(invar)
            dtype_str = str(invar.aval.dtype) if hasattr(invar.aval, "dtype") else ""
            register_var(invar, f"input_{i}", f"{shape_str}\\n{dtype_str}")

        # Add constant nodes (weights/biases) - gray
        for const_var, const_val in zip(inner_jaxpr.constvars, consts):
            node_id = get_node_id(const_var)
            label = var_to_label[get_var_key(const_var)]
            dot.node(node_id, label, style="filled", fillcolor="lightgray")

        # Add input nodes
        for invar in inner_jaxpr.invars:
            node_id = get_node_id(invar)
            label = var_to_label[get_var_key(invar)]
            dot.node(node_id, label, style="filled", fillcolor="lightblue")

        # Readable operation names
        op_names = {
            "dot_general": "MatMul",
            "broadcast_in_dim": "Broadcast",
            "concatenate": "Concat",
        }

        # Add equation nodes
        for i, eqn in enumerate(inner_jaxpr.eqns):
            prim_name = eqn.primitive.name
            display_name = op_names.get(prim_name, prim_name)

            # Get output shape info
            out_shapes = []
            for outvar in eqn.outvars:
                if hasattr(outvar.aval, "shape"):
                    out_shapes.append(str(outvar.aval.shape))

            shape_info = out_shapes[0] if len(out_shapes) == 1 else ", ".join(out_shapes)
            label = f"{display_name}\\n{shape_info}"

            # Color code by operation type
            if "dot" in prim_name:
                color = "lightgreen"
            elif prim_name in ["tanh", "relu", "sigmoid", "sin", "cos", "exp"]:
                color = "khaki"
            elif prim_name in ["add", "mul", "sub", "div"]:
                color = "lightpink"
            elif prim_name in ["broadcast_in_dim", "reshape", "squeeze", "slice"]:
                color = "lavender"
            else:
                color = "white"

            eqn_node_id = f"eqn_{i}"
            dot.node(eqn_node_id, label, style="filled", fillcolor=color)

            # Add edges from inputs
            for invar in eqn.invars:
                if hasattr(invar, "count"):  # It's a variable
                    in_node_id = get_node_id(invar)
                    if in_node_id:
                        dot.edge(in_node_id, eqn_node_id)

            # Register and connect outputs with friendly names
            for outvar in eqn.outvars:
                friendly = get_next_friendly_name()
                shape_str = get_shape_str(outvar)
                out_node_id = register_var(outvar, friendly, shape_str)
                out_label = var_to_label[get_var_key(outvar)]
                dot.node(out_node_id, out_label, shape="ellipse", fontsize="8")
                dot.edge(eqn_node_id, out_node_id)

        # Add output nodes
        for i, outvar in enumerate(inner_jaxpr.outvars):
            shape_str = get_shape_str(outvar)
            out_node_id = get_node_id(outvar)
            dot.node(f"out_{i}", f"Output\\n{shape_str}", style="filled", fillcolor="lightcoral")
            if out_node_id:
                dot.edge(out_node_id, f"out_{i}")

        # Save
        dot_file = f"{base_name}.dot"
        png_file = f"{base_name}.png"

        dot.save(dot_file)
        self.log.info(f"\nSaved DOT file: {dot_file}")

        try:
            dot.render(base_name, format="png", cleanup=False)
            self.log.info(f"Saved PNG visualization: {png_file}")
        except Exception as e:
            self.log.info(f"Could not render PNG (graphviz may not be installed system-wide): {e}")
            self.log.info(f"You can render manually: dot -Tpng {dot_file} -o {png_file}")

    def visualize_trace(self, expression: Union[Placeholder, List[Placeholder]] = None, show_shapes: bool = False):
        """Visualize the PINO DSL computation graph (your variable names).

        This shows your DSL-level operations like x, y, u(x,y), grad, etc.
        Neural network layers are grouped together in clusters.

        Args:
            expression: The expression to visualize. If None, uses all constraints.
            save_to: Output file path for DOT file.
            show_shapes: If True, show tensor shapes at each node (B=batch, N=points).
        """
        count = 0
        expressions = [expression] if isinstance(expression, Placeholder) else expression

        for expression in expressions:
            count += 1
            dot = graphviz.Digraph(comment="PINO Trace Graph")
            # Layout settings for better readability
            dot.attr(
                rankdir="TB",
                fontsize="10",
                dpi="150",
                splines="ortho",  # Orthogonal (right-angle) edges
                nodesep="0.5",  # Horizontal spacing between nodes
                ranksep="0.6",  # Vertical spacing between ranks
                concentrate="true",  # Merge edges going to same node
            )
            dot.attr("node", shape="box", fontsize="9")
            dot.attr("edge", arrowsize="0.7")  # Smaller arrowheads

            # Build shape info if requested
            node_shapes = {}  # id(node) -> shape string
            if show_shapes:
                # Infer tensor dimensions from domain
                tensor_dims = {}
                for tag, arr in self.domain.tensor_tags.items():
                    tensor_dims[tag] = arr.shape[-1] if arr.ndim > 1 else 1
                # Get domain dimension (total including time if applicable)
                domain_dim = self.domain.dimension

                # Helper to infer output dimension at a node
                def infer_dim(node, upstream_dim=None) -> int:
                    if isinstance(node, Variable):
                        return 1
                    elif isinstance(node, TensorTag):
                        return tensor_dims.get(node.tag, 1)
                    elif isinstance(node, Literal):
                        val = jnp.asarray(node.value)
                        return val.shape[-1] if val.ndim > 0 else 1
                    elif isinstance(node, Concat):
                        return sum(infer_dim(item) for item in node.items)
                    elif isinstance(node, BinaryOp):
                        return max(infer_dim(node.left), infer_dim(node.right))
                    elif isinstance(node, FunctionCall):
                        # Check if this is a reduction operation along the last axis
                        if hasattr(node, "reduces_axis") and node.reduces_axis == -1:
                            return 1  # Reduction along last axis produces dim 1
                        # Otherwise preserve input shape
                        for arg in node.args:
                            if isinstance(arg, Placeholder):
                                return infer_dim(arg)
                        return 1
                    elif isinstance(node, OperationCall):
                        return infer_dim(node.operation.expr)
                    elif isinstance(node, OperationDef):
                        return infer_dim(node.expr)
                    elif isinstance(node, Slice):
                        return 1
                    elif isinstance(node, (Gradient, Laplacian, Hessian)):
                        return 1
                    return upstream_dim if upstream_dim else domain_dim

                def get_shape_str(node) -> str:
                    """Get shape string for a node: (B, N, dim)."""
                    dim = infer_dim(node)
                    if isinstance(node, TensorTag):
                        return f"(B,1,{dim})"
                    elif isinstance(node, Variable):
                        return f"(B,N,1)"
                    elif isinstance(node, Literal):
                        return f"({dim},)"
                    elif isinstance(node, (Gradient, Laplacian, Hessian)):
                        return f"(B,N,1)"
                    else:
                        return f"(B,N,{dim})"

                # Cache shape strings for all nodes we'll visit
                def cache_shapes(node):
                    if node is None or id(node) in node_shapes:
                        return
                    node_shapes[id(node)] = get_shape_str(node)
                    if isinstance(node, BinaryOp):
                        cache_shapes(node.left)
                        cache_shapes(node.right)
                    elif isinstance(node, Concat):
                        for item in node.items:
                            cache_shapes(item)
                    elif isinstance(node, FunctionCall):
                        for arg in node.args:
                            if isinstance(arg, Placeholder):
                                cache_shapes(arg)
                    elif isinstance(node, OperationCall):
                        cache_shapes(node.operation.expr)
                        for arg in node.args:
                            if isinstance(arg, Placeholder):
                                cache_shapes(arg)
                    elif isinstance(node, OperationDef):
                        cache_shapes(node.expr)
                    elif isinstance(node, (Gradient, Laplacian, Hessian)):
                        cache_shapes(node.target)
                    elif isinstance(node, Slice):
                        cache_shapes(node.target)

                # Pre-cache shapes for all constraints
                if expression is None:
                    for c in self._constraints:
                        if isinstance(c, OperationDef):
                            cache_shapes(c.expr)
                        else:
                            cache_shapes(c)
                else:
                    cache_shapes(expression)

            def shape_suffix(node) -> str:
                """Get shape suffix for label if show_shapes is enabled."""
                if show_shapes and id(node) in node_shapes:
                    return f"\\n{node_shapes[id(node)]}"
                return ""

            # Track visited nodes to avoid duplicates
            visited = set()
            node_ids = {}  # id(node) -> node_id string
            node_counter = [0]
            edges = []  # Collect edges to add after all nodes
            clusters = {}  # op_id -> cluster subgraph

            def get_node_id(node):
                """Get or create a node ID."""
                key = id(node)
                if key not in node_ids:
                    node_ids[key] = f"n{node_counter[0]}"
                    node_counter[0] += 1
                return node_ids[key]

            def visit(node, parent_graph=None):
                """Recursively visit nodes and build graph."""
                if node is None:
                    return None

                node_id = get_node_id(node)
                graph = parent_graph if parent_graph else dot

                # Skip if already fully processed
                if id(node) in visited:
                    return node_id
                visited.add(id(node))

                # Handle different node types
                if isinstance(node, Variable):
                    label = f"Var({node.tag}[{node.dim}]){shape_suffix(node)}"
                    dot.node(node_id, label, style="filled", fillcolor="lightblue")

                elif isinstance(node, TensorTag):
                    label = f"Tensor({node.tag}){shape_suffix(node)}"
                    dot.node(node_id, label, style="filled", fillcolor="lightblue")

                elif isinstance(node, Literal):
                    val = node.value
                    if hasattr(val, "shape"):
                        label = f"Literal\\n{val.shape}{shape_suffix(node)}"
                    else:
                        label = f"{val}{shape_suffix(node)}"
                    dot.node(node_id, label, style="filled", fillcolor="lightgray")

                elif isinstance(node, BinaryOp):
                    label = f"{node.op}{shape_suffix(node)}"
                    dot.node(node_id, label, style="filled", fillcolor="lightpink", shape="circle")
                    left_id = visit(node.left, graph)
                    right_id = visit(node.right, graph)
                    if left_id:
                        edges.append((left_id, node_id))
                    if right_id:
                        edges.append((right_id, node_id))

                elif isinstance(node, FunctionCall):
                    fn_name = node._name if hasattr(node, "_name") and node._name else getattr(node.fn, "__name__", str(node.fn))
                    label = f"{fn_name}(){shape_suffix(node)}"
                    dot.node(node_id, label, style="filled", fillcolor="khaki")
                    for arg in node.args:
                        if isinstance(arg, Placeholder):
                            arg_id = visit(arg, graph)
                            if arg_id:
                                edges.append((arg_id, node_id))

                elif isinstance(node, Laplacian):
                    label = f"laplacian(){shape_suffix(node)}"
                    dot.node(node_id, label, style="filled", fillcolor="orange")
                    target_id = visit(node.target, graph)
                    if target_id:
                        edges.append((target_id, node_id))
                    # Also show variables
                    for v in node.variables:
                        v_id = visit(v, graph)

                elif isinstance(node, Gradient):
                    label = f"grad(){shape_suffix(node)}"
                    dot.node(node_id, label, style="filled", fillcolor="orange")
                    target_id = visit(node.target, graph)
                    if target_id:
                        edges.append((target_id, node_id))
                    var_id = visit(node.variable, graph)

                elif isinstance(node, OperationDef):
                    # Create cluster for operations to show grouping
                    cluster_name = f"cluster_op{node.op_id}"
                    if node.op_id not in clusters:
                        if node.has_trainable:
                            # BinaryOp-based operation with neural nets (e.g., attention * x * (1-x)) - create cluster
                            with dot.subgraph(name=cluster_name) as sub:
                                sub.attr(label=f"Op[{node.op_id}]", style="rounded,filled", color="darkgreen", fillcolor="honeydew")
                                clusters[node.op_id] = sub
                                # Visit the internal expression inside the cluster
                                expr_id = self._visit_expr_for_trace(node.expr, sub, dot, visited, node_ids, node_counter, edges, clusters, node_shapes=node_shapes)
                                # Add output node for the operation
                                sub.node(node_id, f"Op[{node.op_id}]\\noutput", style="filled", fillcolor="palegreen")
                                # Connect internal expression to output
                                if expr_id:
                                    edges.append((expr_id, node_id))
                        else:
                            # Non-trainable operation (e.g., source terms like f = sin(x)*sin(y))
                            # Show as single compact node instead of full expression tree - NO cluster
                            dot.node(node_id, f"Op[{node.op_id}]\\n(math expr)", style="filled,rounded", fillcolor="thistle")
                            clusters[node.op_id] = True  # Mark as processed but no cluster
                            # Still need to visit variables used in the expression for edges
                            self._collect_vars_for_edges(node.expr, node_id, dot, visited, node_ids, node_counter, edges)
                    else:
                        # Already processed, just return the id
                        pass

                elif isinstance(node, OperationCall):
                    # Operation call: u(x, y) - each CALL is a distinct node
                    # First ensure the operation itself is shown (creates cluster/node for Op)
                    op_output_id = visit(node.operation, graph)

                    # Create a node for THIS specific call
                    call_node_id = get_node_id(node)  # Uses id(OperationCall) which is unique per call
                    op_label = f"Op[{node.operation.op_id}]"
                    args_str = ", ".join(str(a) for a in node.args)
                    dot.node(call_node_id, f"{op_label}({args_str})", style="filled,rounded", fillcolor="honeydew")

                    # Connect operation output to call node
                    if op_output_id:
                        edges.append((op_output_id, call_node_id))

                    # Check if call arguments differ from the operation's original variables
                    # If same (no substitution), skip redundant edges - they're already in the computation tree
                    # If different (substitution like u(x0, t0) for u defined with x, t), show the edges
                    original_vars = node.operation._collected_vars
                    args_differ = len(node.args) != len(original_vars) or any(id(arg) != id(orig) for arg, orig in zip(node.args, original_vars))

                    if args_differ:
                        # Substitution happening - show argument edges
                        for arg in node.args:
                            arg_id = visit(arg, graph)
                            if arg_id:
                                edges.append((arg_id, call_node_id))
                    return call_node_id  # Return this call's node ID

                elif isinstance(node, Concat):
                    label = f"Concat(axis={node.axis})"
                    dot.node(node_id, label, style="filled", fillcolor="white")
                    for item in node.items:
                        item_id = visit(item, graph)
                        if item_id:
                            edges.append((item_id, node_id))

                elif isinstance(node, Slice):
                    label = f"[{node.key}]"
                    dot.node(node_id, label, style="filled", fillcolor="lavender")
                    target_id = visit(node.target, graph)
                    if target_id:
                        edges.append((target_id, node_id))

                else:
                    # Generic placeholder
                    label = repr(node)[:30]
                    dot.node(node_id, label)

                return node_id

            # If no expression given, visualize all constraints
            if expression is None:
                if not hasattr(self, "_constraints") or not self._constraints:
                    self.log.info("No constraints stored. Call solve() first or pass an expression.")
                    return None
                for i, constraint in enumerate(self._constraints):
                    constraint_id = f"constraint_{i}"
                    dot.node(constraint_id, f"Constraint {i}\\n== 0", style="filled", fillcolor="lightcoral", shape="doubleoctagon")
                    # Always unwrap the top-level OperationDef to get the actual expression
                    if isinstance(constraint, OperationDef):
                        expr_id = visit(constraint.expr, dot)
                    else:
                        expr_id = visit(constraint, dot)
                    if expr_id:
                        edges.append((expr_id, constraint_id))
            else:
                visit(expression, dot)

            # Add all edges
            for src, dst in edges:
                dot.edge(src, dst)

            self.dots.append(dot)

        return self.dots[-1] if count > 1 else self.dots[0]

    def _visit_pipeline_for_trace(self, node, graph, main_graph, visited, node_ids, node_counter, edges, node_shapes=None):
        """Helper to visit pipeline nodes inside an operation cluster.

        Args:
            graph: The subgraph (cluster) to add layer nodes to
            main_graph: The main dot graph (for Variables which stay outside clusters)
            node_shapes: Dict mapping id(node) -> shape string (optional)
        """
        if node is None:
            return None

        def shape_suffix(n):
            if node_shapes and id(n) in node_shapes:
                return f"\\n{node_shapes[id(n)]}"
            return ""

        def get_node_id(n):
            key = id(n)
            if key not in node_ids:
                node_ids[key] = f"n{node_counter[0]}"
                node_counter[0] += 1
            return node_ids[key]

        if isinstance(node, Concat):
            node_id = get_node_id(node)
            label = f"Concat{shape_suffix(node)}"
            graph.node(node_id, label, style="filled", fillcolor="white")
            for item in node.items:
                item_id = self._visit_pipeline_for_trace(item, graph, main_graph, visited, node_ids, node_counter, edges, node_shapes)
                if item_id:
                    edges.append((item_id, node_id))
            return node_id

        elif isinstance(node, Variable):
            node_id = get_node_id(node)
            if id(node) not in visited:
                visited.add(id(node))
                label = f"Var({node.tag}[{node.dim}]){shape_suffix(node)}"
                # Variables go in main graph, not cluster
                main_graph.node(node_id, label, style="filled", fillcolor="lightblue")
            return node_id

        elif isinstance(node, FunctionCall):
            node_id = get_node_id(node)
            fn_name = node._name if hasattr(node, "_name") and node._name else getattr(node.fn, "__name__", str(node.fn))
            label = f"{fn_name}(){shape_suffix(node)}"
            graph.node(node_id, label, style="filled", fillcolor="khaki")
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    arg_id = self._visit_pipeline_for_trace(arg, graph, main_graph, visited, node_ids, node_counter, edges, node_shapes)
                    if arg_id:
                        edges.append((arg_id, node_id))
            return node_id

        return None

    def _visit_expr_for_trace(self, node, graph, main_graph, visited, node_ids, node_counter, edges, clusters, pipeline_counter=None, node_shapes=None):
        """Helper to visit expression nodes inside an operation cluster.

        Args:
            graph: The subgraph (cluster) to add nodes to
            main_graph: The main dot graph (for Variables/TensorTags which stay outside clusters)
            clusters: Dict of existing clusters (for nested OperationCalls)
            pipeline_counter: Mutable counter for pipeline cluster IDs
            node_shapes: Dict mapping id(node) -> shape string (optional)
        """
        if node is None:
            return None

        if pipeline_counter is None:
            pipeline_counter = [0]

        def shape_suffix(n):
            if node_shapes and id(n) in node_shapes:
                return f"\\n{node_shapes[id(n)]}"
            return ""

        def get_node_id(n):
            key = id(n)
            if key not in node_ids:
                node_ids[key] = f"n{node_counter[0]}"
                node_counter[0] += 1
            return node_ids[key]

        node_id = get_node_id(node)

        # Skip if already visited
        if id(node) in visited:
            return node_id
        visited.add(id(node))

        if isinstance(node, BinaryOp):
            label = f"{node.op}{shape_suffix(node)}"
            graph.node(node_id, label, style="filled", fillcolor="lightpink", shape="circle")
            left_id = self._visit_expr_for_trace(node.left, graph, main_graph, visited, node_ids, node_counter, edges, clusters, pipeline_counter, node_shapes)
            right_id = self._visit_expr_for_trace(node.right, graph, main_graph, visited, node_ids, node_counter, edges, clusters, pipeline_counter, node_shapes)
            if left_id:
                edges.append((left_id, node_id))
            if right_id:
                edges.append((right_id, node_id))
            return node_id

        elif isinstance(node, Variable):
            # Variables go in main graph, not cluster
            label = f"Var({node.tag}[{node.dim}]){shape_suffix(node)}"
            main_graph.node(node_id, label, style="filled", fillcolor="lightblue")
            return node_id

        elif isinstance(node, TensorTag):
            # TensorTags go in main graph, not cluster
            label = f"Tensor({node.tag}){shape_suffix(node)}"
            main_graph.node(node_id, label, style="filled", fillcolor="lightblue")
            return node_id

        elif isinstance(node, Literal):
            val = node.value
            if hasattr(val, "shape"):
                label = f"Literal\\n{val.shape}{shape_suffix(node)}"
            else:
                label = f"{val}{shape_suffix(node)}"
            graph.node(node_id, label, style="filled", fillcolor="lightgray")
            return node_id

        elif isinstance(node, FunctionCall):
            fn_name = node._name if hasattr(node, "_name") and node._name else getattr(node.fn, "__name__", str(node.fn))
            label = f"{fn_name}(){shape_suffix(node)}"
            graph.node(node_id, label, style="filled", fillcolor="khaki")
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    arg_id = self._visit_expr_for_trace(arg, graph, main_graph, visited, node_ids, node_counter, edges, clusters, pipeline_counter, node_shapes)
                    if arg_id:
                        edges.append((arg_id, node_id))
            return node_id

        elif isinstance(node, OperationCall):
            # Nested operation call - visit the operation (which may create its own cluster)
            # Then create a call node referencing it
            op = node.operation
            cluster_name = f"cluster_op{op.op_id}"

            # Ensure the operation's cluster exists
            if op.op_id not in clusters and op.has_trainable:
                op_node_id = get_node_id(op)
                with main_graph.subgraph(name=cluster_name) as sub:
                    sub.attr(label=f"Op[{op.op_id}]", style="rounded,filled", color="darkgreen", fillcolor="honeydew")
                    clusters[op.op_id] = sub
                    expr_id = self._visit_expr_for_trace(op.expr, sub, main_graph, visited, node_ids, node_counter, edges, clusters, pipeline_counter, node_shapes)
                    sub.node(op_node_id, f"Op[{op.op_id}]\\noutput", style="filled", fillcolor="palegreen")
                    if expr_id:
                        edges.append((expr_id, op_node_id))

            # Create call node in current graph
            op_label = f"Op[{op.op_id}]"
            args_str = ", ".join(str(a) for a in node.args)
            graph.node(node_id, f"{op_label}({args_str})", style="filled,rounded", fillcolor="honeydew")

            # Connect operation output to call
            op_output_id = get_node_id(op)
            edges.append((op_output_id, node_id))
            return node_id

        elif isinstance(node, Concat):
            label = f"Concat(axis={node.axis}){shape_suffix(node)}"
            graph.node(node_id, label, style="filled", fillcolor="white")
            for item in node.items:
                item_id = self._visit_expr_for_trace(item, graph, main_graph, visited, node_ids, node_counter, edges, clusters, pipeline_counter, node_shapes)
                if item_id:
                    edges.append((item_id, node_id))
            return node_id

        elif isinstance(node, Slice):
            label = f"[{node.key}]{shape_suffix(node)}"
            graph.node(node_id, label, style="filled", fillcolor="lavender")
            target_id = self._visit_expr_for_trace(node.target, graph, main_graph, visited, node_ids, node_counter, edges, clusters, pipeline_counter, node_shapes)
            if target_id:
                edges.append((target_id, node_id))
            return node_id

        return None

    def _collect_vars_for_edges(self, expr, target_node_id, graph, visited, node_ids, node_counter, edges):
        """Collect variables from an expression and create edges to target node.

        Used for compact math expression nodes - just shows variable dependencies
        without the full expression tree.
        """

        def get_node_id(n):
            key = id(n)
            if key not in node_ids:
                node_ids[key] = f"n{node_counter[0]}"
                node_counter[0] += 1
            return node_ids[key]

        def collect(node):
            if node is None:
                return
            if isinstance(node, Variable):
                var_id = get_node_id(node)
                if id(node) not in visited:
                    visited.add(id(node))
                    label = f"Var({node.tag}[{node.dim}])"
                    graph.node(var_id, label, style="filled", fillcolor="lightblue")
                edges.append((var_id, target_node_id))
            elif isinstance(node, TensorTag):
                tag_id = get_node_id(node)
                if id(node) not in visited:
                    visited.add(id(node))
                    label = f"Tensor({node.tag})"
                    graph.node(tag_id, label, style="filled", fillcolor="lightblue")
                edges.append((tag_id, target_node_id))
            elif isinstance(node, BinaryOp):
                collect(node.left)
                collect(node.right)
            elif isinstance(node, FunctionCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        collect(arg)
            elif isinstance(node, Concat):
                for item in node.items:
                    collect(item)
            elif isinstance(node, Slice):
                collect(node.target)
            elif isinstance(node, OperationCall):
                for arg in node.args:
                    collect(arg)

        collect(expr)

    def _visit_pipeline_inside_cluster(self, node, graph, main_graph, visited, node_ids, node_counter, edges, node_shapes=None):
        """Visit pipeline nodes inside a neural network cluster.

        Similar to _visit_pipeline_for_trace but for nested pipelines in expressions.
        """
        if node is None:
            return None

        def shape_suffix(n):
            if node_shapes and id(n) in node_shapes:
                return f"\\n{node_shapes[id(n)]}"
            return ""

        def get_node_id(n):
            key = id(n)
            if key not in node_ids:
                node_ids[key] = f"n{node_counter[0]}"
                node_counter[0] += 1
            return node_ids[key]

        if isinstance(node, Concat):
            node_id = get_node_id(node)
            if id(node) not in visited:
                visited.add(id(node))
                label = f"Concat(axis={node.axis}){shape_suffix(node)}"
                graph.node(node_id, label, style="filled", fillcolor="white")
                for item in node.items:
                    item_id = self._visit_pipeline_inside_cluster(item, graph, main_graph, visited, node_ids, node_counter, edges, node_shapes)
                    if item_id:
                        edges.append((item_id, node_id))
            return node_id

        elif isinstance(node, Variable):
            node_id = get_node_id(node)
            if id(node) not in visited:
                visited.add(id(node))
                label = f"Var({node.tag}[{node.dim}]){shape_suffix(node)}"
                main_graph.node(node_id, label, style="filled", fillcolor="lightblue")
            return node_id

        elif isinstance(node, TensorTag):
            node_id = get_node_id(node)
            if id(node) not in visited:
                visited.add(id(node))
                label = f"Tensor({node.tag}){shape_suffix(node)}"
                main_graph.node(node_id, label, style="filled", fillcolor="lightblue")
            return node_id

        elif isinstance(node, FunctionCall):
            node_id = get_node_id(node)
            if id(node) not in visited:
                visited.add(id(node))
                fn_name = node._name if hasattr(node, "_name") and node._name else getattr(node.fn, "__name__", str(node.fn))
                label = f"{fn_name}(){shape_suffix(node)}"
                graph.node(node_id, label, style="filled", fillcolor="khaki")
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        arg_id = self._visit_pipeline_inside_cluster(arg, graph, main_graph, visited, node_ids, node_counter, edges, node_shapes)
                        if arg_id:
                            edges.append((arg_id, node_id))
            return node_id

        return None
