"""Graph compilation and parameter initialisation utilities.

This module is responsible for the **one-time setup phase**:

- Traversing the expression tree to discover ``Model`` nodes
  (:meth:`TraceCompiler.collect_dense_layers`).
- Initialising / loading parameters for every layer
  (:meth:`TraceCompiler.init_layer_params`,
  :meth:`TraceCompiler.build_single_layer_params`).
- Compiling traced expressions into JAX-compatible vmapped/scanned
  callables (:meth:`TraceCompiler.compile_traced_expression`,
  :meth:`TraceCompiler.compile_multi_expression`).

The hot-path **evaluation** code lives in :mod:`jno.trace_evaluator`
(:class:`~jno.trace_evaluator.TraceEvaluator`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from .trace import (
    BinaryOp,
    FunctionCall,
    Hessian,
    Jacobian,
    Model,
    ModelCall,
    OperationCall,
    OperationDef,
    Placeholder,
    TunableModule,
    TunableModuleCall,
    Variable,
)


# Lazy import: TraceEvaluator imports from this module at package level,
# so we defer the import to avoid any potential circular dependency.
# It is resolved at function-call time, well after both modules are loaded.
def _get_evaluator_class():
    from .trace_evaluator import TraceEvaluator  # noqa: PLC0415

    return TraceEvaluator


def _default_float_dtype():
    """Return JAX's current default floating dtype (float32 or float64)."""
    return jnp.asarray(0.0).dtype


class TraceCompiler:
    """One-time graph-compilation and parameter-initialisation utilities.

    All methods are *static* — instantiation is never required.  The class
    exists purely as a namespace to group compilation-phase helpers that
    would otherwise clutter :class:`~jno.trace_evaluator.TraceEvaluator`.

    Typical call order when setting up a solve::

        # 1. Discover learnable layers
        layers = TraceCompiler.collect_dense_layers(expr)

        # 2. Initialise / load weights
        params, rng = TraceCompiler.init_layer_params(
            all_ops, domain_dim, tensor_dims, rng, logger
        )

        # 3. Compile to a JAX function (vmap + scan, JIT-ready)
        fn = TraceCompiler.compile_traced_expression(expr, all_ops)
        loss = jax.value_and_grad(lambda p: fn(p, context).mean())(params)
    """

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------

    @staticmethod
    def collect_dense_layers(expr: Placeholder) -> List:
        """Collect all Model nodes and their call arguments from expression tree.

        Traverses depth-first so that dependencies (modules whose outputs feed
        into other modules) are collected before the modules that consume them.

        Returns:
            List of ``(Model, call_args | None)`` tuples.
            ``call_args`` is ``None`` for standalone parameter modules.
        """
        layers = []
        seen = set()

        def visit(node):
            if isinstance(node, Model):
                if node.layer_id not in seen:
                    seen.add(node.layer_id)
                    layers.append((node, None))

            elif isinstance(node, TunableModule):
                if node._current_instance is not None:
                    inst = node._current_instance
                    if inst.layer_id not in seen:
                        seen.add(inst.layer_id)
                        layers.append((inst, None))

            elif isinstance(node, TunableModuleCall):
                # Visit args first (dependency order)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
                tunable = node.model
                if tunable._current_instance is not None:
                    flax_mod = tunable._current_instance
                    if flax_mod.layer_id not in seen:
                        seen.add(flax_mod.layer_id)
                        layers.append((flax_mod, node.args))

            elif isinstance(node, ModelCall):
                # Visit args first (dependency order)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
                flax_mod = node.model
                if flax_mod.layer_id not in seen:
                    seen.add(flax_mod.layer_id)
                    layers.append((flax_mod, node.args))

            elif isinstance(node, BinaryOp):
                visit(node.left)
                visit(node.right)
            elif isinstance(node, FunctionCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, OperationCall):
                visit(node.operation.expr)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, (Hessian, Jacobian)):
                visit(node.target)

        visit(expr)
        return layers

    # ------------------------------------------------------------------
    # Shape inference (for legacy Flax modules only)
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_arg_shapes(call_args: List, tensor_dims: Dict[str, tuple], existing_params: Dict) -> List[tuple]:
        """Infer the *normalised* argument shapes for a ModelCall.

        Only needed for legacy Flax modules that require dummy inputs
        for ``module.init()``.  Equinox modules are constructed eagerly
        and never reach this code path.
        """
        TraceEvaluator = _get_evaluator_class()
        abstract_ctx = {tag: jax.ShapeDtypeStruct(tuple(shape), _default_float_dtype()) for tag, shape in tensor_dims.items()}

        def eval_and_normalize(context):
            evaluator = TraceEvaluator(existing_params)
            ctx = evaluator._EvalCtx(context, {}, jax.random.PRNGKey(0))

            arg_values = []
            arg_sources = []
            for arg in call_args:
                val = evaluator._dispatch(arg, ctx)
                arg_values.append(val)
                is_spatial = isinstance(arg, Variable) and arg.tag in context
                arg_sources.append(is_spatial)

            N = 1
            for val, is_spatial in zip(arg_values, arg_sources):
                if is_spatial:
                    val = jnp.asarray(val)
                    if val.ndim >= 1:
                        N = max(N, val.shape[0])

            def normalize_arg(val, is_spatial):
                val = jnp.asarray(val)
                if is_spatial:
                    if val.ndim == 0:
                        return jnp.full((N, 1), val)
                    elif val.ndim == 1:
                        return val[:, jnp.newaxis]
                    else:
                        return val
                else:
                    if val.ndim == 0:
                        return val[jnp.newaxis]
                    else:
                        return val

            return tuple(normalize_arg(v, s) for v, s in zip(arg_values, arg_sources))

        abstract_results = jax.eval_shape(eval_and_normalize, abstract_ctx)
        return [r.shape for r in abstract_results]

    # ------------------------------------------------------------------
    # Weight utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cast_model_dtype(model, dtype, logger):
        """Cast all floating-point arrays in *model* to *dtype*.

        Works for both Equinox modules and ``FlaxModelWrapper`` (which
        wraps a Flax param dict).  Integer arrays are left unchanged.
        """

        def cast_leaf(x):
            if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(dtype)
            return x

        model = jax.tree_util.tree_map(cast_leaf, model)
        logger.info(f"Cast model parameters to {dtype}")
        return model

    @staticmethod
    def merge_pretrained_params(pretrained_params: dict, new_params: dict, logger) -> dict:
        """
        Merge pretrained weights with new params, replacing embedding/recovery layers
        when shapes don't match (for different channel dimensions).

        A concise summary (counts) is logged to the main logger.  Detailed
        per-parameter information is written to ``weight_merge.log`` in the
        same directory as the logger's output path.
        """
        stats = {"matched": 0, "replaced": 0}
        details: list = []  # collect per-param detail lines

        def count_params(arr):
            return arr.size if hasattr(arr, "size") else 0

        def merge(pretrained, new, path=""):
            if isinstance(pretrained, dict) and isinstance(new, dict):
                result = {}
                all_keys = set(list(pretrained.keys()) + list(new.keys()))

                for key in all_keys:
                    current_path = f"{path}/{key}" if path else key

                    if key in pretrained and key in new:
                        if isinstance(pretrained[key], dict):
                            result[key] = merge(pretrained[key], new[key], current_path)
                        else:
                            if pretrained[key].shape == new[key].shape:
                                result[key] = pretrained[key]
                                stats["matched"] += count_params(pretrained[key])
                                details.append(f"  MATCHED  {current_path}  " f"shape={pretrained[key].shape}  " f"params={count_params(pretrained[key]):,}")
                            else:
                                result[key] = new[key]
                                n = count_params(new[key])
                                stats["replaced"] += n
                                details.append(f"  MISMATCH {current_path}  " f"{pretrained[key].shape} -> {new[key].shape}  " f"params={n:,}  (reinitialized)")
                    elif key in pretrained:
                        result[key] = pretrained[key]
                        if not isinstance(pretrained[key], dict):
                            n = count_params(pretrained[key])
                            stats["matched"] += n
                            details.append(f"  MATCHED  {current_path}  " f"shape={pretrained[key].shape}  " f"params={n:,}  (pretrained only)")
                    else:
                        result[key] = new[key]
                        if not isinstance(new[key], dict):
                            n = count_params(new[key])
                            stats["replaced"] += n
                            details.append(f"  NEW      {current_path}  " f"shape={new[key].shape}  " f"params={n:,}  (new only)")

                return result
            else:
                if hasattr(pretrained, "shape") and hasattr(new, "shape"):
                    if pretrained.shape == new.shape:
                        stats["matched"] += count_params(pretrained)
                        return pretrained
                    else:
                        stats["replaced"] += count_params(new)
                        return new
                return new if new is not None else pretrained

        merged = merge(pretrained_params, new_params)

        total = stats["matched"] + stats["replaced"]
        pct = 100 * stats["matched"] / total if total else 0
        n_mismatch = sum(1 for d in details if "MISMATCH" in d)
        n_new = sum(1 for d in details if "NEW" in d)

        summary = f"Pretrained weights: {stats['matched']:,}/{total:,} params matched " f"({pct:.4f}%), {stats['replaced']:,} reinitialized " f"({n_mismatch} shape mismatches, {n_new} new)"
        logger.info(summary)

        # Write detailed per-parameter report to a text file next to log.txt
        log_dir = getattr(logger, "path", None)
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            detail_path = log_dir / "weight_merge.log"
            with open(detail_path, "w") as f:
                f.write(summary + "\n\n")
                f.write("\n".join(details) + "\n")

        return merged

    @staticmethod
    def build_single_layer_params(layer, arg_shapes, rng, logger):
        """Retrieve or construct the model for a single layer.

        The model was already fully constructed at factory time — we just
        return ``layer.module``, optionally loading pretrained weights.
        """
        if not isinstance(layer, Model):
            raise ValueError(f"Unknown layer type: {type(layer)}")

        module = layer.module
        init_mask = getattr(layer, "_initialize_mask", None)

        # ---- Flax NNX wrapper path ----------------------------------
        from .architectures.common import FlaxModelWrapper, FlaxNNXWrapper

        if isinstance(module, FlaxNNXWrapper) and getattr(layer, "_weight_tree", None) is not None:
            pretrained = layer._weight_tree
            try:
                from flax import nnx as _nnx

                if isinstance(pretrained, _nnx.Module):
                    # Extract state from the live NNX module
                    _, pretrained_state = _nnx.split(pretrained)
                else:
                    # Assume it already is an nnx.State (or compatible pytree)
                    pretrained_state = pretrained
            except ImportError:
                pretrained_state = pretrained
            logger.info("Loading pretrained NNX weights from pytree")
            module = FlaxNNXWrapper.__new__(FlaxNNXWrapper)
            object.__setattr__(module, "graphdef", layer.module.graphdef)
            if init_mask is not None:
                state_mask = init_mask.state if hasattr(init_mask, "state") else init_mask
                state = jax.tree_util.tree_map(
                    lambda src, dst, m: src if bool(m) else dst,
                    pretrained_state,
                    layer.module.state,
                    state_mask,
                )
            else:
                state = jax.tree_util.tree_map(lambda src, _: src, pretrained_state, layer.module.state)
            object.__setattr__(module, "state", state)
            object.__setattr__(module, "post_fn", layer.module.post_fn)
            object.__setattr__(module, "default_kwargs", layer.module.default_kwargs)
            if layer.show:
                logger.info(f"  FlaxNNXWrapper: {module._param_count():,} parameters")
            return module

        # ---- Flax Linen wrapper path (msgpack weights) ---------------

        if isinstance(module, FlaxModelWrapper) and (layer.weight_path is not None or getattr(layer, "_weight_tree", None) is not None):
            if layer.weight_path is not None:
                logger.info(f"Loading pretrained Flax weights from {layer.weight_path}")
                from flax.serialization import from_bytes

                with open(layer.weight_path, "rb") as f:
                    pretrained_params = from_bytes(module.params, f.read())
            else:
                # Pytree supplied directly — must be a dict of Flax params
                pretrained_params = layer._weight_tree
                logger.info("Loading pretrained Flax weights from pytree")

            # Merge pretrained weights with fresh params
            merged = TraceCompiler.merge_pretrained_params(
                pretrained_params,
                module.params,
                logger,
            )
            # Optional masked initialise: copy only selected leaves from
            # pretrained params; keep fresh init elsewhere.
            if init_mask is not None:
                params_mask = init_mask.params if hasattr(init_mask, "params") else init_mask
                merged = jax.tree_util.tree_map(
                    lambda pre, fresh, m: pre if bool(m) else fresh,
                    merged,
                    module.params,
                    params_mask,
                )
                logger.info("Applied masked initialise (Flax): loaded target subset only")

            # Return a new FlaxModelWrapper with merged params
            model = FlaxModelWrapper(
                module.apply_fn,
                merged,
                post_fn=module.post_fn,
                **module.default_kwargs,
            )

            # ---- optional dtype cast --------------------------------
            if getattr(layer, "_dtype", None) is not None:
                model = TraceCompiler._cast_model_dtype(model, layer._dtype, logger)

            if layer.show:
                leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
                total = sum(l.size for l in leaves)
                logger.info(f"  {type(model).__name__}: {total:,} parameters")

            return model

        # ---- Equinox path (normal) ----------------------------------
        if isinstance(module, eqx.Module):
            model = module

            if layer.weight_path is not None:
                logger.info(f"Loading pretrained weights from {layer.weight_path}")
                model = eqx.tree_deserialise_leaves(layer.weight_path, model)
            elif getattr(layer, "_weight_tree", None) is not None:
                # Pytree supplied directly — copy array leaves from the tree
                # onto the freshly-initialised model.
                logger.info("Loading pretrained weights from pytree")
                model = jax.tree_util.tree_map(
                    lambda src, _: src,
                    layer._weight_tree,
                    model,
                )

            if init_mask is not None:
                model = jax.tree_util.tree_map(
                    lambda pre, fresh, m: pre if bool(m) else fresh,
                    model,
                    module,
                    init_mask,
                )
                logger.info("Applied masked initialise: loaded target subset only")

            # ---- optional dtype cast --------------------------------
            if getattr(layer, "_dtype", None) is not None:
                model = TraceCompiler._cast_model_dtype(model, layer._dtype, logger)

            if layer.show:
                leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
                total = sum(l.size for l in leaves)
                logger.info(f"  {type(model).__name__}: {total:,} parameters")

            return model

    # ------------------------------------------------------------------
    # Parameter initialisation for all layers
    # ------------------------------------------------------------------

    @staticmethod
    def init_layer_params(all_ops: List, domain_dim: int, tensor_dims: Dict[str, Tuple], rng: jax.Array, logger) -> Tuple[Dict, jax.Array]:
        """Collect / initialise models for all layers.

        For equinox modules (the normal path), the model was already
        constructed eagerly at factory time — we just store it directly
        (with optional pretrained weight loading).

        For legacy Flax modules (ScOT / Poseidon), we fall back to
        shape inference via ``jax.eval_shape`` and ``module.init``.

        Returns:
            all_models: Dict mapping layer_id -> callable model
            rng: Updated RNG key
        """
        all_models: Dict[int, Any] = {}
        seen = set()

        for op in all_ops:
            layers_with_args = TraceCompiler.collect_dense_layers(op.expr)
            for layer, call_args in layers_with_args:
                if layer.layer_id in seen:
                    continue
                seen.add(layer.layer_id)

                rng, init_rng = jax.random.split(rng)

                # Fast path: equinox modules are already fully constructed
                if isinstance(layer.module, eqx.Module):
                    model = TraceCompiler.build_single_layer_params(layer, None, init_rng, logger)
                else:
                    # Legacy Flax: need shape inference first
                    if call_args is not None:
                        arg_shapes = TraceCompiler._infer_arg_shapes(call_args, tensor_dims, all_models)
                    else:
                        arg_shapes = None
                    model = TraceCompiler.build_single_layer_params(layer, arg_shapes, init_rng, logger)

                all_models[layer.layer_id] = model

        return all_models, rng

    # ------------------------------------------------------------------
    # Expression compilation
    # ------------------------------------------------------------------

    @staticmethod
    def compile_traced_expression(expr: Placeholder, all_ops: List[OperationDef]) -> Callable:
        """Compile traced expression into a JAX-compatible function.

        The compiled function handles the (B, T, N, D) data layout:

        1. ``vmap`` over B (batch dimension)
        2. ``jax.lax.scan`` over T (time steps — T=1 for steady-state)
        3. Evaluate the expression on ``(N, D_spatial)`` context

        The ``"__time__"`` context entry (shape ``(T, 1)``) is **not**
        batched — it is shared and scanned over T together with the
        spatial arrays.
        """
        TraceEvaluator = _get_evaluator_class()
        TIME_TAG = "__time__"

        def evaluate_single_point_set(params, context_single, key):
            """Evaluate for a single (N, D) context — no batch or time."""
            evaluator = TraceEvaluator(params)
            return evaluator.evaluate(expr, context_single, {}, key)

        def compiled_fn(params, context=None, batchsize=None, key=None, min_consecutive=1):
            """
            Evaluate the compiled expression.

            Args:
                params: Model parameters
                context: Unified dictionary — spatial tags have shape
                    ``(B, T, N, D)``, ``"__time__"`` has shape ``(T, 1)``
                    (absent for steady-state), parametric tags ``(B, F)``.
                batchsize: If provided, randomly select this many samples
                    from the batch dimension.
                key: JAX random key for mini-batch and stochastic ops.
                min_consecutive: Minimum number of consecutive time steps
                    passed to the evaluator in one call.  Setting this >= T
                    passes all time steps at once (no loop, 2 AD passes).
                    Values > 1 require the model to accept a leading time
                    dimension (shape ``(W, N, D)`` instead of ``(N, D)``).
            """
            context = context or {}

            # ----- tag ordering (stable across calls) -----------------
            all_tags = set()
            for op in all_ops:
                if hasattr(op, "_collected_vars"):
                    for var in op._collected_vars:
                        all_tags.add(var.tag)

            tag_order = tuple(
                sorted(
                    context.keys(),
                    key=lambda t: (t not in all_tags, t),
                )
            )
            ctx_tuple = tuple(context[tag] for tag in tag_order) if tag_order else ()

            # ----- determine batch size B -----------------------------
            # Spatial arrays are (B, T, N, D) — first dim is B.
            # __time__ is (T, 1) — skip it when finding B.
            batched_sizes = []
            for tag, arr in zip(tag_order, ctx_tuple):
                if tag == TIME_TAG:
                    continue
                if hasattr(arr, "ndim") and arr.ndim >= 1:
                    batched_sizes.append(arr.shape[0])

            if not batched_sizes:
                return evaluate_single_point_set(params, context, key=key)

            B = max(batched_sizes)

            # ----- mini-batch subset selection ------------------------
            if batchsize is not None:
                if key is None:
                    raise ValueError("A JAX random key must be provided when " "batchsize is specified.")
                if batchsize > B:
                    indices = jax.random.choice(key, B, shape=(batchsize,), replace=True)
                    indices = jnp.sort(indices)
                elif batchsize < B:
                    indices = jax.random.choice(key, B, shape=(batchsize,), replace=False)
                    indices = jnp.sort(indices)
                else:
                    indices = jnp.arange(0, B, 1)

                def subset_entry(tag_name, arr):
                    if tag_name == TIME_TAG:
                        return arr  # not batched
                    if hasattr(arr, "ndim") and arr.ndim >= 1 and arr.shape[0] == B:
                        return arr[indices]
                    return arr

                ctx_tuple = tuple(subset_entry(t, a) for t, a in zip(tag_order, ctx_tuple))
                B = batchsize

            # ----- separate __time__ from batched arrays ---------------
            # After vmap peels B, spatial arrays become (T, N, D).
            # __time__ is (T, 1) and must NOT be vmapped — pass via
            # closure instead.
            time_arr = None  # will be set if __time__ is present
            time_idx_in_order = None

            spatial_tag_order = []
            spatial_ctx = []
            for i, (tag, arr) in enumerate(zip(tag_order, ctx_tuple)):
                if tag == TIME_TAG:
                    time_arr = jnp.asarray(arr)  # (T, 1)
                    time_idx_in_order = i
                else:
                    spatial_tag_order.append(tag)
                    spatial_ctx.append(arr)

            spatial_tag_order = tuple(spatial_tag_order)
            spatial_ctx = tuple(spatial_ctx)

            # ----- normalize for vmap (batch axis) --------------------
            def normalize_entry(arg):
                if not hasattr(arg, "ndim") or arg.ndim == 0:
                    return arg, None
                bs = arg.shape[0]
                if bs == B:
                    return arg, 0
                elif bs == 1:
                    return jnp.squeeze(arg, axis=0), None
                else:
                    return arg, None

            new_ctx = []
            ctx_in_axes = []
            for a in spatial_ctx:
                a2, ax = normalize_entry(a)
                new_ctx.append(a2)
                ctx_in_axes.append(ax)
            spatial_ctx = tuple(new_ctx)
            ctx_in_axes = tuple(ctx_in_axes)

            # ----- inner: single sampled temporal window per sample ----------
            def scan_over_time(spatial_vals, rng_key):
                """Evaluate one consecutive W-step window for this sample.

                W is controlled by ``min_consecutive`` and clamped to ``T``.
                When ``T > W``, we sample a random start index per sample (using
                ``rng_key``) and evaluate only that window. This keeps temporal
                context while avoiding a full pass over all windows each step.
                """
                # T and W are static Python ints — resolved from shapes at trace time
                T = 1
                for v in spatial_vals:
                    if hasattr(v, "ndim") and v.ndim >= 3:
                        T = max(T, v.shape[0])

                W = max(1, min(min_consecutive, T))  # window size
                idx_dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
                zero_idx = jnp.asarray(0, dtype=idx_dtype)

                if T > W:
                    if rng_key is None:
                        start = zero_idx
                    else:
                        start = jax.random.randint(rng_key, shape=(), minval=0, maxval=T - W + 1)
                        start = start.astype(idx_dtype)
                else:
                    start = zero_idx

                def eval_window(windowed_ctx, t_wind):
                    """Evaluate on one window of W steps.
                    windowed_ctx: tuple of (W, N, D) or non-spatial arrays.
                    t_wind: (W, 1) time slice, or dummy scalar when time_arr is None.
                    """
                    ctx_dict = {}
                    for tag, arr in zip(spatial_tag_order, windowed_ctx):
                        if W == 1 and hasattr(arr, "ndim") and arr.ndim >= 2:
                            ctx_dict[tag] = arr[0]  # (1, N, D) → (N, D) — scalar-step compat
                        else:
                            ctx_dict[tag] = arr  # (W, N, D)
                    if time_arr is not None:
                        ctx_dict[TIME_TAG] = t_wind[0] if W == 1 else t_wind
                    return evaluate_single_point_set(params, ctx_dict, key=rng_key)

                # Slice one temporal window: (T, ...) → (W, ...)
                windowed_list = []
                for arr in spatial_vals:
                    if hasattr(arr, "ndim") and arr.ndim >= 3 and arr.shape[0] == T:
                        slice_sizes = (W,) + tuple(arr.shape[1:])
                        start_idx = (start,) + (zero_idx,) * (arr.ndim - 1)
                        windowed_list.append(jax.lax.dynamic_slice(arr, start_idx, slice_sizes))
                    elif hasattr(arr, "ndim") and arr.ndim >= 3 and arr.shape[0] < T:
                        # Broadcast static/short temporal inputs (e.g. initial condition)
                        # to the selected window length.
                        windowed_list.append(jnp.broadcast_to(arr, (W, *arr.shape[1:])))
                    else:
                        windowed_list.append(arr)

                if time_arr is not None:
                    t_windowed = jax.lax.dynamic_slice(time_arr, (start, zero_idx), (W, 1))
                else:
                    t_windowed = jnp.zeros((W, 1))  # dummy — never read when time_arr is None

                return eval_window(tuple(windowed_list), t_windowed)

            # ----- outer: vmap over B ---------------------------------
            if key is not None:
                keys = jax.random.split(key, B)
                vmapped_fn = jax.vmap(
                    scan_over_time,
                    in_axes=(ctx_in_axes, 0),
                )
                return vmapped_fn(spatial_ctx, keys)
            else:

                def scan_over_time_no_key(spatial_vals):
                    return scan_over_time(spatial_vals, rng_key=None)

                vmapped_fn = jax.vmap(
                    scan_over_time_no_key,
                    in_axes=(ctx_in_axes,),
                )
                return vmapped_fn(spatial_ctx)

        return compiled_fn

    @staticmethod
    def compile_multi_expression(exprs: List[Placeholder], all_ops: List[OperationDef]) -> Callable:
        """Compile multiple constraint expressions into a SINGLE function.

        All expressions are evaluated by the same ``TraceEvaluator`` instance,
        so JAX/XLA sees them in one compilation unit and can apply CSE across
        constraints.  Individual residual arrays are returned as a list so
        ``_make_loss_fn`` can still compute per-constraint losses.

        Mirrors ``compile_traced_expression`` exactly — only
        ``evaluate_single_point_set`` changes.
        """
        TraceEvaluator = _get_evaluator_class()
        TIME_TAG = "__time__"

        def evaluate_single_point_set(params, context_single, key):
            """Evaluate ALL expressions on one (N, D) context — shared evaluator."""
            evaluator = TraceEvaluator(params)
            # One evaluator → one JAX trace → XLA sees all constraints together
            return [evaluator.evaluate(expr, context_single, {}, key) for expr in exprs]

        # Everything below is identical to compile_traced_expression.
        def compiled_fn(params, context=None, batchsize=None, key=None, min_consecutive=1):
            context = context or {}

            all_tags = set()
            for op in all_ops:
                if hasattr(op, "_collected_vars"):
                    for var in op._collected_vars:
                        all_tags.add(var.tag)

            tag_order = tuple(sorted(context.keys(), key=lambda t: (t not in all_tags, t)))
            ctx_tuple = tuple(context[tag] for tag in tag_order) if tag_order else ()

            batched_sizes = []
            for tag, arr in zip(tag_order, ctx_tuple):
                if tag == TIME_TAG:
                    continue
                if hasattr(arr, "ndim") and arr.ndim >= 1:
                    batched_sizes.append(arr.shape[0])

            if not batched_sizes:
                return evaluate_single_point_set(params, context, key=key)

            B = max(batched_sizes)

            if batchsize is not None:
                if key is None:
                    raise ValueError("A JAX random key must be provided when batchsize is specified.")
                if batchsize > B:
                    indices = jax.random.choice(key, B, shape=(batchsize,), replace=True)
                    indices = jnp.sort(indices)
                elif batchsize < B:
                    indices = jax.random.choice(key, B, shape=(batchsize,), replace=False)
                    indices = jnp.sort(indices)
                else:
                    indices = jnp.arange(0, B, 1)

                def subset_entry(tag_name, arr):
                    if tag_name == TIME_TAG:
                        return arr
                    if hasattr(arr, "ndim") and arr.ndim >= 1 and arr.shape[0] == B:
                        return arr[indices]
                    return arr

                ctx_tuple = tuple(subset_entry(t, a) for t, a in zip(tag_order, ctx_tuple))
                B = batchsize

            time_arr = None
            spatial_tag_order = []
            spatial_ctx = []
            for tag, arr in zip(tag_order, ctx_tuple):
                if tag == TIME_TAG:
                    time_arr = jnp.asarray(arr)
                else:
                    spatial_tag_order.append(tag)
                    spatial_ctx.append(arr)

            spatial_tag_order = tuple(spatial_tag_order)
            spatial_ctx = tuple(spatial_ctx)

            def normalize_entry(arg):
                if not hasattr(arg, "ndim") or arg.ndim == 0:
                    return arg, None
                bs = arg.shape[0]
                if bs == B:
                    return arg, 0
                elif bs == 1:
                    return jnp.squeeze(arg, axis=0), None
                else:
                    return arg, None

            new_ctx, ctx_in_axes = [], []
            for a in spatial_ctx:
                a2, ax = normalize_entry(a)
                new_ctx.append(a2)
                ctx_in_axes.append(ax)
            spatial_ctx = tuple(new_ctx)
            ctx_in_axes = tuple(ctx_in_axes)

            def scan_over_time(spatial_vals, rng_key):
                T = 1
                for v in spatial_vals:
                    if hasattr(v, "ndim") and v.ndim >= 3:
                        T = max(T, v.shape[0])

                W = max(1, min(min_consecutive, T))
                idx_dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
                zero_idx = jnp.asarray(0, dtype=idx_dtype)

                if T > W:
                    if rng_key is None:
                        start = zero_idx
                    else:
                        start = jax.random.randint(rng_key, shape=(), minval=0, maxval=T - W + 1)
                        start = start.astype(idx_dtype)
                else:
                    start = zero_idx

                def eval_window(windowed_ctx, t_wind):
                    ctx_dict = {}
                    for tag, arr in zip(spatial_tag_order, windowed_ctx):
                        if W == 1 and hasattr(arr, "ndim") and arr.ndim >= 2:
                            ctx_dict[tag] = arr[0]
                        else:
                            ctx_dict[tag] = arr
                    if time_arr is not None:
                        ctx_dict[TIME_TAG] = t_wind[0] if W == 1 else t_wind
                    return evaluate_single_point_set(params, ctx_dict, key=rng_key)

                windowed_list = []
                for arr in spatial_vals:
                    if hasattr(arr, "ndim") and arr.ndim >= 3 and arr.shape[0] == T:
                        slice_sizes = (W,) + tuple(arr.shape[1:])
                        start_idx = (start,) + (zero_idx,) * (arr.ndim - 1)
                        windowed_list.append(jax.lax.dynamic_slice(arr, start_idx, slice_sizes))
                    elif hasattr(arr, "ndim") and arr.ndim >= 3 and arr.shape[0] < T:
                        windowed_list.append(jnp.broadcast_to(arr, (W, *arr.shape[1:])))
                    else:
                        windowed_list.append(arr)

                if time_arr is not None:
                    t_windowed = jax.lax.dynamic_slice(time_arr, (start, zero_idx), (W, 1))
                else:
                    t_windowed = jnp.zeros((W, 1))

                return eval_window(tuple(windowed_list), t_windowed)

            if key is not None:
                keys = jax.random.split(key, B)
                vmapped_fn = jax.vmap(scan_over_time, in_axes=(ctx_in_axes, 0))
                return vmapped_fn(spatial_ctx, keys)
            else:

                def scan_over_time_no_key(spatial_vals):
                    return scan_over_time(spatial_vals, rng_key=None)

                vmapped_fn = jax.vmap(scan_over_time_no_key, in_axes=(ctx_in_axes,))
                return vmapped_fn(spatial_ctx)

        return compiled_fn
