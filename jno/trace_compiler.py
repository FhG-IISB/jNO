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

import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .trace import (
    Assembly,
    BinaryOp,
    Choice,
    FunctionCall,
    GroupedAssembly,
    Hessian,
    Integral,
    IntegralTime,
    Jacobian,
    Model,
    ModelCall,
    ModelWeights,
    NetworkGradient,
    OperationCall,
    OperationDef,
    Placeholder,
    TunableModule,
    TunableModuleCall,
    Variable,
    collect_tags,
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


def _collect_temporal_derivative_targets(expr_or_exprs):
    """Walk the trace graph and return ``(target, id(target))`` pairs for every
    distinct :class:`TemporalDerivative` target, in **post-order**.

    Post-order ensures that inner targets appear before outer ones so the
    compiler can populate the temporal-FD cache incrementally — when an outer
    ``TemporalDerivative`` is pre-computed, its target (an inner
    ``TemporalDerivative``) already has its own window in the cache.

    The traversal follows the same attribute conventions as
    :func:`jno.trace._contains_node_type_local`: ``"left"``, ``"right"``,
    ``"target"``, ``"expr"``, ``"operation"``, ``"model"``, plus the iterable
    fields ``"args"`` and ``"variables"``.
    """
    from .trace import Placeholder, TemporalDerivative  # noqa: PLC0415

    exprs = expr_or_exprs if isinstance(expr_or_exprs, (list, tuple)) else [expr_or_exprs]
    seen_nodes: set[int] = set()
    seen_targets: set[int] = set()
    results: list = []

    def walk(node):
        if not isinstance(node, Placeholder):
            return
        if id(node) in seen_nodes:
            return
        seen_nodes.add(id(node))
        # Visit children first (post-order)
        for attr in ("left", "right", "target", "expr", "operation", "model"):
            child = getattr(node, attr, None)
            if isinstance(child, Placeholder):
                walk(child)
        for attr in ("args", "variables"):
            vals = getattr(node, attr, None) or []
            for v in vals:
                if isinstance(v, (list, tuple)):
                    for vv in v:
                        if isinstance(vv, Placeholder):
                            walk(vv)
                elif isinstance(v, Placeholder):
                    walk(v)
        # After children: append this node's target
        if isinstance(node, TemporalDerivative):
            tid = id(node.target)
            if tid not in seen_targets:
                seen_targets.add(tid)
                results.append((node.target, tid))

    for e in exprs:
        walk(e)
    return results


def _find_fd_domain(expr):
    """Walk expression tree to find a domain that uses FD + sub-domains.

    Returns the domain if FD derivatives are present and the domain has
    sub-domains with ``_batch_domain_map``, otherwise ``None``.
    """
    visited = set()
    domain = None

    def _walk(node):
        nonlocal domain
        if domain is not None:
            return
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)

        if isinstance(node, (Jacobian, Hessian)):
            if getattr(node, "scheme", "").startswith("finite_difference"):
                # Find domain from variable nodes
                for v in getattr(node, "variables", []):
                    d = getattr(v, "_domain", None)
                    if d is not None and getattr(d, "_sub_domains", []):
                        domain = d
                        return

        # Recurse into children
        for attr in ("target", "left", "right", "expr", "volume_expr"):
            child = getattr(node, attr, None)
            if child is not None:
                _walk(child)
        for attr in ("variables", "args", "boundary_exprs"):
            children = getattr(node, attr, None)
            if isinstance(children, (list, tuple)):
                for c in children:
                    if c is not None:
                        _walk(c)

    if isinstance(expr, (list, tuple)):
        for e in expr:
            _walk(e)
    else:
        _walk(expr)
    return domain


import numpy as _np  # used only for _grouped_vmap index manipulation


def _grouped_vmap(
    scan_fn,
    spatial_ctx,
    ctx_in_axes,
    spatial_tag_order,
    fd_domain,
    batch_domain_map,
    B,
    key,
    is_multi,
):
    """Run vmap in per-domain groups so each group uses correct mesh_connectivity.

    For each unique domain index in *batch_domain_map*, the function:
    1. Subsets the context arrays to only that group's batch indices.
    2. Temporarily swaps ``fd_domain.mesh_connectivity`` to the group's mesh.
    3. Runs ``jax.vmap(scan_fn)`` on the subset.
    4. Restores the original ``mesh_connectivity``.
    5. Concatenates all group outputs in the original batch order.

    Parameters
    ----------
    scan_fn : callable
        The per-sample evaluation function ``(spatial_vals, rng_key) -> result``.
    spatial_ctx : tuple of arrays
        Context arrays (potentially batched along axis 0).
    ctx_in_axes : tuple
        Per-array vmap in_axes (0 or None).
    spatial_tag_order : tuple of str
        Tag names corresponding to ``spatial_ctx``.
    fd_domain : domain
        The domain with ``_sub_domains`` and ``_domain_mesh_connectivities``.
    batch_domain_map : ndarray of int, shape (B,)
        Maps each batch index to its source domain index.
    B : int
        Total batch size.
    key : jax random key or None
    is_multi : bool
        If True, scan_fn returns a list of arrays (multi-expression).
    """
    connectivities = fd_domain._domain_mesh_connectivities
    original_mc = fd_domain.mesh_connectivity
    unique_domains = sorted(set(int(d) for d in batch_domain_map))

    # Pre-split keys if needed
    all_keys = jax.random.split(key, B) if key is not None else None

    group_results = {}  # domain_idx -> result (array or list of arrays)
    group_indices = {}  # domain_idx -> numpy array of original batch indices

    for d_idx in unique_domains:
        mask = _np.array(batch_domain_map) == d_idx
        indices = _np.where(mask)[0]
        group_indices[d_idx] = indices

        # Subset context arrays for this group
        grp_ctx = []
        grp_in_axes = []
        for arr, ax in zip(spatial_ctx, ctx_in_axes):
            if ax == 0:
                grp_ctx.append(arr[indices])
                grp_in_axes.append(0)
            else:
                grp_ctx.append(arr)
                grp_in_axes.append(ax)
        grp_ctx = tuple(grp_ctx)
        grp_in_axes = tuple(grp_in_axes)

        # Swap mesh_connectivity
        mc = connectivities[d_idx]
        fd_domain.mesh_connectivity = mc

        if key is not None:
            grp_keys = all_keys[indices]
            vmapped = jax.vmap(scan_fn, in_axes=(grp_in_axes, 0))
            grp_result = vmapped(grp_ctx, grp_keys)
        else:

            def _scan_no_key(sv):
                return scan_fn(sv, rng_key=None)

            vmapped = jax.vmap(_scan_no_key, in_axes=(grp_in_axes,))
            grp_result = vmapped(grp_ctx)

        group_results[d_idx] = grp_result

    # Restore original mesh_connectivity
    fd_domain.mesh_connectivity = original_mc

    # Recombine in original batch order
    # Build a permutation: for each original batch index, find which
    # group it's in and what position within that group.
    group_offsets = {}
    offset = 0
    for d_idx in unique_domains:
        group_offsets[d_idx] = offset
        offset += len(group_indices[d_idx])

    # Build concatenation order: groups are concatenated in domain order,
    # then we permute to restore original batch order.
    concat_to_original = _np.empty(B, dtype=_np.int64)
    pos = 0
    for d_idx in unique_domains:
        for local_i, orig_i in enumerate(group_indices[d_idx]):
            concat_to_original[pos] = orig_i
            pos += 1

    # The inverse permutation: original_order[concat_to_original[i]] = i
    inv_perm = _np.argsort(concat_to_original)

    if is_multi:
        # grp_result is a list of arrays, one per expression
        # Concatenate along batch axis for each expression, then reorder
        n_exprs = len(group_results[unique_domains[0]])
        combined = []
        for expr_idx in range(n_exprs):
            parts = [group_results[d_idx][expr_idx] for d_idx in unique_domains]
            cat = jnp.concatenate(parts, axis=0)
            combined.append(cat[inv_perm])
        return combined
    else:
        parts = [group_results[d_idx] for d_idx in unique_domains]
        cat = jnp.concatenate(parts, axis=0)
        return cat[inv_perm]


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

            elif isinstance(node, ModelWeights):
                # A neural FEM coefficient: the solve node references the model's weights (not a
                # call), so register the Model itself — that is what makes it trainable.
                visit(node.model)

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
            elif isinstance(node, OperationCall):
                visit(node.operation.expr)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, (Hessian, Jacobian)):
                visit(node.target)
            elif isinstance(node, NetworkGradient):
                visit(node.target)  # registers the Model inside target
            elif isinstance(node, (Integral, IntegralTime)):
                visit(node.target)
            elif isinstance(node, Assembly):
                visit(node.expr)
            elif isinstance(node, GroupedAssembly):
                if node.volume_value_expr is not None:
                    visit(node.volume_value_expr)
                if node.volume_grad_expr is not None:
                    visit(node.volume_grad_expr)
                for bnd_expr in node.boundary_value_exprs.values():
                    visit(bnd_expr)

        visit(expr)
        return layers

    # ------------------------------------------------------------------
    # Shape inference (legacy — kept for offline tools)
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_arg_shapes(call_args: List, tensor_dims: Dict[str, tuple], existing_params: Dict) -> List[tuple]:
        """Infer the *normalised* argument shapes for a ModelCall."""
        TraceEvaluator = _get_evaluator_class()
        abstract_ctx = {
            tag: jax.ShapeDtypeStruct(tuple(shape), _default_float_dtype()) for tag, shape in tensor_dims.items()
        }

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

            normalized = tuple(normalize_arg(v, s) for v, s in zip(arg_values, arg_sources))

            # Keep shape inference consistent with TraceEvaluator._eval_flax_module_call.
            _model = None
            # In this function you only have call_args, not the model object.
            # So either pass model into _infer_arg_shapes later, or leave this alone
            # if this function is no longer used for Equinox-foundax initialization.
            return normalized

        abstract_results = jax.eval_shape(eval_and_normalize, abstract_ctx)
        return [r.shape for r in abstract_results]

    # ------------------------------------------------------------------
    # Weight utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cast_model_dtype(model, dtype, logger):
        """Cast all floating-point arrays in *model* to *dtype*.

        Integer arrays are left unchanged.
        """

        def cast_leaf(x):
            if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(dtype)
            return x

        model = jax.tree_util.tree_map(cast_leaf, model)
        logger.info(f"Cast model parameters to {dtype}")
        return model

    @staticmethod
    def _apply_callable_initializer(model, initializer: Callable, *, key: jax.Array, logger):
        """Apply a JAX initializer to every floating-point array leaf in *model*."""
        leaves, treedef = jax.tree_util.tree_flatten(model)
        out_leaves = []
        local_key = key
        n_init = 0

        for leaf in leaves:
            if eqx.is_inexact_array(leaf):
                local_key, subkey = jax.random.split(local_key)
                try:
                    init_leaf = initializer(subkey, leaf.shape, leaf.dtype)
                except TypeError:
                    init_leaf = initializer(subkey, leaf.shape)
                out_leaves.append(jnp.asarray(init_leaf, dtype=leaf.dtype))
                n_init += 1
            else:
                out_leaves.append(leaf)

        logger.info(f"Applied callable initializer to {n_init} floating-point parameter leaves")
        return jax.tree_util.tree_unflatten(treedef, out_leaves)

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
                                details.append(
                                    f"  MATCHED  {current_path}  "
                                    f"shape={pretrained[key].shape}  "
                                    f"params={count_params(pretrained[key]):,}"
                                )
                            else:
                                result[key] = new[key]
                                n = count_params(new[key])
                                stats["replaced"] += n
                                details.append(
                                    f"  MISMATCH {current_path}  "
                                    f"{pretrained[key].shape} -> {new[key].shape}  "
                                    f"params={n:,}  (reinitialized)"
                                )
                    elif key in pretrained:
                        result[key] = pretrained[key]
                        if not isinstance(pretrained[key], dict):
                            n = count_params(pretrained[key])
                            stats["matched"] += n
                            details.append(
                                f"  MATCHED  {current_path}  shape={pretrained[key].shape}  params={n:,}  (pretrained only)"
                            )
                    else:
                        result[key] = new[key]
                        if not isinstance(new[key], dict):
                            n = count_params(new[key])
                            stats["replaced"] += n
                            details.append(f"  NEW      {current_path}  shape={new[key].shape}  params={n:,}  (new only)")

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

        summary = (
            f"Pretrained weights: {stats['matched']:,}/{total:,} params matched "
            f"({pct:.4f}%), {stats['replaced']:,} reinitialized "
            f"({n_mismatch} shape mismatches, {n_new} new)"
        )
        logger.info(summary)

        # Write detailed per-parameter report to a text file next to log.log
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
    def _count_checkpoint_arrays(weight_path: str):
        """Count arrays and total params in an .eqx file by scanning npy headers."""
        import numpy as np

        path = Path(weight_path)
        if path.suffix == "":
            path = path.with_suffix(".eqx")
        if not path.exists():
            return None, None
        n_arrays = 0
        total_params = 0
        with open(path, "rb") as f:
            while True:
                try:
                    major, minor = np.lib.format.read_magic(f)
                    reader = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
                    shape, _fortran, dtype = reader(f)
                    n_elems = int(np.prod(shape)) if len(shape) > 0 else 1
                    f.seek(n_elems * dtype.itemsize, 1)
                    n_arrays += 1
                    total_params += n_elems
                except Exception:
                    break
        return n_arrays, total_params

    @staticmethod
    def _load_eqx_weights_partial(weight_path: str, model, logger):
        """Load an Equinox checkpoint, skipping leaves with incompatible shapes.

        Only compatible leaves are loaded; mismatched leaves keep their
        fresh initialisation.
        """

        stats = {"matched": 0, "skipped": 0, "matched_leaves": 0, "skipped_leaves": 0}

        def _count_params(arr):
            return int(arr.size) if hasattr(arr, "size") else 0

        def _filter_spec(f, x):
            loaded = eqx.default_deserialise_filter_spec(f, x)

            # For non-array leaves default_deserialise_filter_spec returns x
            # unchanged and does not read from the file.
            if not eqx.is_array(x) and not isinstance(x, jax.ShapeDtypeStruct):
                return loaded

            src_shape = tuple(loaded.shape) if hasattr(loaded, "shape") else None
            dst_shape = tuple(x.shape) if hasattr(x, "shape") else None
            src_dtype = getattr(loaded, "dtype", None)
            dst_dtype = getattr(x, "dtype", None)

            shape_ok = src_shape is not None and dst_shape is not None and src_shape == dst_shape
            dtype_ok = (src_dtype is None) or (dst_dtype is None) or (src_dtype == dst_dtype)

            if shape_ok and dtype_ok:
                stats["matched"] += _count_params(loaded)
                stats["matched_leaves"] += 1
                return loaded

            # Keep fresh init for incompatible leaves.
            stats["skipped"] += _count_params(x)
            stats["skipped_leaves"] += 1
            return x

        loaded_model = eqx.tree_deserialise_leaves(weight_path, model, filter_spec=_filter_spec)

        total = stats["matched"] + stats["skipped"]
        pct = 100 * stats["matched"] / total if total else 0.0
        logger.info(
            f"Equinox checkpoint: {stats['matched']:,}/{total:,} params matched "
            f"({pct:.4f}%), {stats['skipped']:,} kept fresh init "
            f"({stats['skipped_leaves']} mismatched leaves)"
        )

        # Report unused arrays in the checkpoint file.
        file_arrays, file_params = TraceCompiler._count_checkpoint_arrays(weight_path)
        if file_arrays is not None:
            used_leaves = stats["matched_leaves"] + stats["skipped_leaves"]
            unused_arrays = file_arrays - used_leaves
            unused_params = file_params - total
            if unused_arrays > 0:
                logger.warning(
                    f"Checkpoint file contains {file_arrays} arrays "
                    f"({file_params:,} params) but model only consumed "
                    f"{used_leaves} arrays — {unused_arrays} arrays "
                    f"({unused_params:,} params) unused"
                )
            else:
                logger.info(f"Checkpoint file: {file_arrays} arrays ({file_params:,} params total), all consumed by model")

        return loaded_model

    @staticmethod
    def _resolve_orbax_step_dir(weight_path: str) -> tuple[Path, str | None]:
        """Resolve an Orbax checkpoint root or step dir, with optional model-key selector.

        Supports paths of the form ``/path/to/checkpoints`` (latest step),
        ``/path/to/checkpoints/1234`` (specific step), and
        ``/path/to/checkpoints/1234::1`` (specific saved trainable model key).
        """
        raw_path = str(weight_path)
        selected_key = None
        if "::" in raw_path:
            raw_path, selected_key = raw_path.rsplit("::", 1)
            selected_key = selected_key.strip() or None

        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")

        if path.is_dir() and (path / "_CHECKPOINT_METADATA").exists() and (path / "state").exists():
            return path, selected_key

        if not path.is_dir():
            raise FileNotFoundError(f"Orbax checkpoint path must be a directory, got: {path}")

        try:
            import orbax.checkpoint as ocp
        except ImportError as exc:
            raise ImportError(
                "orbax-checkpoint is required to load Orbax checkpoints. Install it with: pip install orbax-checkpoint"
            ) from exc

        manager = ocp.CheckpointManager(
            os.path.abspath(path),
            options=ocp.CheckpointManagerOptions(read_only=True),
        )
        try:
            step = manager.latest_step()
        finally:
            manager.close()

        if step is None:
            raise FileNotFoundError(f"No Orbax checkpoints found in {path}")

        return path / str(step), selected_key

    @staticmethod
    def _discover_orbax_trainable_keys(step_dir: Path) -> list[str]:
        metadata_path = step_dir / "state" / "_METADATA"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Orbax state metadata not found: {metadata_path}")

        metadata_text = metadata_path.read_text()
        keys = sorted(set(re.findall(r"\('trainable', '([^']+)'", metadata_text)))
        if not keys:
            raise ValueError(f"No trainable model entries found in Orbax checkpoint: {step_dir}")
        return keys

    @staticmethod
    def _normalize_keypath(keypath: tuple[Any, ...]) -> tuple[Any, ...]:
        normalized: list[Any] = []
        for key in keypath:
            if hasattr(key, "idx"):
                normalized.append(key.idx)
            elif hasattr(key, "key"):
                normalized.append(key.key)
            elif hasattr(key, "name"):
                normalized.append(key.name)
            else:
                normalized.append(str(key))
        return tuple(normalized)

    @staticmethod
    def _count_shape_params(shape: tuple[int, ...]) -> int:
        return int(math.prod(shape)) if len(shape) > 0 else 1

    @staticmethod
    def _build_orbax_restore_plan(checkpoint_tree, model):
        checkpoint_by_path = {}
        checkpoint_arrays = 0
        checkpoint_params = 0
        for keypath, meta in jax.tree_util.tree_leaves_with_path(checkpoint_tree):
            normalized = TraceCompiler._normalize_keypath(keypath)
            checkpoint_by_path[normalized] = meta
            checkpoint_arrays += 1
            checkpoint_params += TraceCompiler._count_shape_params(tuple(meta.shape))

        stats = {
            "matched": 0,
            "skipped": 0,
            "matched_leaves": 0,
            "skipped_leaves": 0,
            "checkpoint_arrays": checkpoint_arrays,
            "checkpoint_params": checkpoint_params,
            "considered_checkpoint_arrays": 0,
            "considered_checkpoint_params": 0,
        }
        matching_paths: set[tuple[Any, ...]] = set()

        filtered_model = eqx.filter(model, eqx.is_array)
        for keypath, leaf in jax.tree_util.tree_leaves_with_path(filtered_model):
            normalized = TraceCompiler._normalize_keypath(keypath)
            meta = checkpoint_by_path.get(normalized)
            if meta is None:
                stats["skipped"] += int(leaf.size)
                stats["skipped_leaves"] += 1
                continue

            stats["considered_checkpoint_arrays"] += 1
            stats["considered_checkpoint_params"] += TraceCompiler._count_shape_params(tuple(meta.shape))

            shape_ok = tuple(meta.shape) == tuple(leaf.shape)
            dtype_ok = getattr(meta, "dtype", None) is None or meta.dtype == leaf.dtype
            if shape_ok and dtype_ok:
                stats["matched"] += int(leaf.size)
                stats["matched_leaves"] += 1
                matching_paths.add(normalized)
            else:
                stats["skipped"] += int(leaf.size)
                stats["skipped_leaves"] += 1

        stats["unused_arrays"] = stats["checkpoint_arrays"] - stats["considered_checkpoint_arrays"]
        stats["unused_params"] = stats["checkpoint_params"] - stats["considered_checkpoint_params"]
        return matching_paths, stats

    @staticmethod
    def _load_orbax_weights_partial(weight_path: str, model, logger):
        """Load a model subtree from an Orbax training checkpoint.

        Orbax checkpoints written by jNO store a full training state under the
        ``state`` item. This loader restores only the ``trainable`` subtree for
        the selected model key so that ``nn.initialize(...)`` can reuse those
        weights without restoring optimizer state.
        """
        try:
            import orbax.checkpoint as ocp
        except ImportError as exc:
            raise ImportError(
                "orbax-checkpoint is required to load Orbax checkpoints. Install it with: pip install orbax-checkpoint"
            ) from exc

        step_dir, selected_key = TraceCompiler._resolve_orbax_step_dir(weight_path)
        available_keys = TraceCompiler._discover_orbax_trainable_keys(step_dir)

        if selected_key is None:
            if len(available_keys) != 1:
                available_str = ", ".join(available_keys)
                raise ValueError(
                    "Orbax checkpoint contains multiple trainable models "
                    f"({available_str}). Pass '<checkpoint_path>::<model_key>' to select one."
                )
            selected_key = available_keys[0]
        elif selected_key not in available_keys:
            available_str = ", ".join(available_keys)
            raise ValueError(f"Orbax checkpoint model key '{selected_key}' not found. Available keys: {available_str}")

        metadata_tree = ocp.PyTreeCheckpointHandler().metadata(step_dir / "state").tree["trainable"][selected_key]
        matching_paths, stats = TraceCompiler._build_orbax_restore_plan(metadata_tree, model)

        restore_template = jax.tree_util.tree_map_with_path(
            lambda keypath, leaf: (
                leaf
                if eqx.is_array(leaf) and TraceCompiler._normalize_keypath(keypath) in matching_paths
                else ocp.PLACEHOLDER
            ),
            model,
            is_leaf=lambda leaf: leaf is None,
        )
        restore_item = {"trainable": {selected_key: restore_template}}

        construct_restore_args = getattr(getattr(ocp, "checkpoint_utils", None), "construct_restore_args", None)
        restore_args = construct_restore_args(restore_item) if callable(construct_restore_args) else None

        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        try:
            restored = checkpointer.restore(
                step_dir / "state",
                args=ocp.args.PyTreeRestore(
                    item=restore_item,
                    restore_args=restore_args,
                    partial_restore=True,
                ),
            )
        finally:
            close = getattr(checkpointer, "close", None)
            if callable(close):
                close()

        logger.info(f"Orbax checkpoint: restored trainable model key {selected_key} from {step_dir}")
        total = stats["matched"] + stats["skipped"]
        pct = 100 * stats["matched"] / total if total else 0.0
        skipped_leaf_label = "leaf" if stats["skipped_leaves"] == 1 else "leaves"
        logger.info(
            f"Orbax checkpoint: {stats['matched']:,}/{total:,} params matched "
            f"({pct:.4f}%), {stats['skipped']:,} kept fresh init "
            f"({stats['skipped_leaves']} model {skipped_leaf_label})"
        )
        if stats["unused_arrays"] > 0:
            logger.warning(
                f"Checkpoint file contains {stats['checkpoint_arrays']} arrays "
                f"({stats['checkpoint_params']:,} params) but model only consumed "
                f"{stats['considered_checkpoint_arrays']} arrays — {stats['unused_arrays']} arrays "
                f"({stats['unused_params']:,} params) unused"
            )
        elif stats["skipped_leaves"] > 0:
            logger.info(
                f"Checkpoint file: {stats['checkpoint_arrays']} arrays "
                f"({stats['checkpoint_params']:,} params total), all checkpoint arrays consumed; "
                f"model kept fresh init for {stats['skipped_leaves']} additional "
                f"{skipped_leaf_label} ({stats['skipped']:,} params)"
            )
        else:
            logger.info(
                f"Checkpoint file: {stats['checkpoint_arrays']} arrays "
                f"({stats['checkpoint_params']:,} params total), all consumed by model"
            )
        return jax.tree_util.tree_map(
            lambda restored_leaf, fresh_leaf: (
                fresh_leaf if restored_leaf is ocp.PLACEHOLDER or restored_leaf is None else restored_leaf
            ),
            restored["trainable"][selected_key],
            model,
            is_leaf=lambda leaf: leaf is ocp.PLACEHOLDER or leaf is None,
        )

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

        # ---- Equinox path (all models) ------------------------------
        if not isinstance(module, eqx.Module):
            raise TypeError(
                f"Expected an eqx.Module, got {type(module).__name__}. "
                f"Flax modules are no longer supported at runtime. "
                f"Use the Equinox version of your model."
            )

        model = module

        if layer.weight_path is not None:
            logger.info(f"Loading pretrained weights from {layer.weight_path}")
            if Path(str(layer.weight_path).rsplit("::", 1)[0]).is_dir():
                model = TraceCompiler._load_orbax_weights_partial(layer.weight_path, model, logger)
            else:
                model = TraceCompiler._load_eqx_weights_partial(layer.weight_path, model, logger)
        elif getattr(layer, "_weight_tree", None) is not None:
            # Pytree supplied directly — copy array leaves from the tree
            # onto the freshly-initialised model.
            logger.info("Loading pretrained weights from pytree")
            model = jax.tree_util.tree_map(
                lambda src, _: src,
                layer._weight_tree,
                model,
            )
        elif getattr(layer, "_initializer_fn", None) is not None:
            init_key = getattr(layer, "_initializer_key", None)
            if init_key is None:
                init_key = rng
            logger.info("Initializing model parameters from callable initializer")
            model = TraceCompiler._apply_callable_initializer(model, layer._initializer_fn, key=init_key, logger=logger)

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
            total = sum(leaf.size for leaf in leaves)
            logger.info(f"  {type(model).__name__}: {total:,} parameters")

        return model

    # ------------------------------------------------------------------
    # Parameter initialisation for all layers
    # ------------------------------------------------------------------

    @staticmethod
    def init_layer_params(
        all_ops: List,
        domain_dim: int,
        tensor_dims: Dict[str, Tuple],
        rng: jax.Array,
        logger,
    ) -> Tuple[Dict, jax.Array]:
        """Collect / initialise models for all layers.

        For equinox modules (the only supported path), the model was
        already constructed eagerly at factory time — we just store it
        directly (with optional pretrained weight loading).

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
                model = TraceCompiler.build_single_layer_params(layer, None, init_rng, logger)
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
        expr_tags = collect_tags(expr)
        METADATA_TAGS = {
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
        }

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
                    passed to the evaluator in one call.  ``None`` means use
                    all available steps (``W=T``). Setting this >= T
                    passes all time steps at once (no loop, 2 AD passes).
                    Values > 1 require the model to accept a leading time
                    dimension (shape ``(W, N, D)`` instead of ``(N, D)``).
            """
            context = context or {}

            # ----- tag ordering (stable across calls) -----------------
            all_tags = set(expr_tags)
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
                if tag == TIME_TAG or tag in METADATA_TAGS or tag not in expr_tags:
                    continue
                if hasattr(arr, "ndim") and arr.ndim >= 1:
                    batched_sizes.append(arr.shape[0])

            if not batched_sizes:
                # Inject __time_window__ for IntegralTime nodes that live in
                # time-only expressions (no spatial variables in expr_tags).
                if TIME_TAG in context:
                    context_with_window = dict(context)
                    context_with_window["__time_window__"] = jnp.asarray(context[TIME_TAG])
                    return evaluate_single_point_set(params, context_with_window, key=key)
                return evaluate_single_point_set(params, context, key=key)

            B = max(batched_sizes)

            # ----- mini-batch subset selection ------------------------
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
                    if tag_name == TIME_TAG or tag_name in METADATA_TAGS:
                        return arr
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
            passive_ctx = {}
            spatial_tag_order = []
            spatial_ctx = []
            metadata_ctx = {}

            for tag, arr in zip(tag_order, ctx_tuple):
                if tag == TIME_TAG:
                    time_arr = jnp.asarray(arr)  # (T, 1)
                elif hasattr(arr, "ndim") and arr.ndim >= 1 and arr.shape[0] in (B, 1):
                    # Include all arrays carrying a per-sample batch axis (size B or
                    # broadcastable 1), regardless of whether they appear in expr_tags.
                    # This covers FEM tensors (JxW, N_flat, …) that are accessed by the
                    # evaluator via ctx.context directly and are not Variable-tagged.
                    spatial_tag_order.append(tag)
                    spatial_ctx.append(arr)
                elif tag in METADATA_TAGS:
                    metadata_ctx[tag] = arr
                else:
                    passive_ctx[tag] = arr

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

                W = T if min_consecutive is None else max(1, min(min_consecutive, T))  # window size
                idx_dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
                zero_idx = jnp.asarray(0, dtype=idx_dtype)

                if T > W:
                    if rng_key is None:
                        start = zero_idx
                    else:
                        start = jax.random.randint(rng_key, shape=(), minval=0, maxval=T - W + 1).astype(idx_dtype)
                else:
                    start = zero_idx

                def eval_window(windowed_ctx, t_wind):
                    """Evaluate on one window of W steps.
                    windowed_ctx: tuple of (W, N, D) or non-spatial arrays.
                    t_wind: (W, 1) time slice, or dummy scalar when time_arr is None.
                    """
                    # Augment passive_ctx with the full time window so IntegralTime
                    # handlers can access all W time values even inside the per-step vmap.
                    augmented_passive_ctx = dict(passive_ctx)
                    augmented_passive_ctx["__time_window__"] = t_wind  # (W, 1)

                    # Pre-compute every TemporalDerivative.target on all W steps
                    # before the per-step vmap.  This avoids the W² cost that
                    # would arise from each per-step evaluation re-running the
                    # target W times internally.  Targets are processed in
                    # post-order (innermost first) so nested TDs see populated
                    # cache entries during their own pre-compute pass.
                    temporal_fd_cache: dict = {}
                    td_targets = _collect_temporal_derivative_targets(expr)
                    if td_targets:
                        TraceEvaluator = _get_evaluator_class()
                        for td_target, tid in td_targets:
                            captured_cache = dict(temporal_fd_cache)
                            in_axes_pc = []
                            for arr in windowed_ctx:
                                if hasattr(arr, "ndim") and arr.ndim >= 3:
                                    in_axes_pc.append(0)
                                else:
                                    in_axes_pc.append(None)
                            in_axes_pc.append(0)  # t_wind axis
                            in_axes_pc.append(0)  # step index axis
                            step_indices = jnp.arange(t_wind.shape[0], dtype=jnp.int32)

                            def _eval_target_at_step(*args, _target=td_target, _cache=captured_cache):
                                step_idx = args[-1]
                                step_t = args[-2]
                                step_spatial = args[:-2]
                                ctx_dict = dict(metadata_ctx)
                                ctx_dict.update(passive_ctx)
                                ctx_dict["__time_window__"] = t_wind
                                ctx_dict["__temporal_fd_cache__"] = _cache
                                ctx_dict["__step_index__"] = step_idx
                                for tag, step_arr in zip(spatial_tag_order, step_spatial):
                                    ctx_dict[tag] = step_arr
                                if time_arr is not None:
                                    ctx_dict[TIME_TAG] = step_t
                                return TraceEvaluator(params).evaluate(_target, ctx_dict, {}, rng_key)

                            u_window = jax.vmap(
                                _eval_target_at_step,
                                in_axes=tuple(in_axes_pc),
                            )(*windowed_ctx, t_wind, step_indices)
                            temporal_fd_cache[tid] = u_window

                        augmented_passive_ctx["__temporal_fd_cache__"] = temporal_fd_cache

                    if W > 1:
                        # Vmap over the time-window dimension so each step
                        # sees (N, D) spatial + (1,) time — identical to W=1.
                        in_axes_list = []
                        for arr in windowed_ctx:
                            if hasattr(arr, "ndim") and arr.ndim >= 3:
                                in_axes_list.append(0)
                            else:
                                in_axes_list.append(None)
                        in_axes_list.append(0)  # for t_wind
                        in_axes_list.append(0)  # for step indices
                        step_indices = jnp.arange(W, dtype=jnp.int32)

                        def _eval_single_step(*step_spatial_and_t_and_idx):
                            step_idx = step_spatial_and_t_and_idx[-1]
                            step_t = step_spatial_and_t_and_idx[-2]
                            step_spatial = step_spatial_and_t_and_idx[:-2]
                            ctx_dict = dict(metadata_ctx)
                            ctx_dict.update(augmented_passive_ctx)
                            ctx_dict["__step_index__"] = step_idx
                            active_spatial_n = None
                            for tag, step_arr in zip(spatial_tag_order, step_spatial):
                                ctx_dict[tag] = step_arr
                                if active_spatial_n is None and hasattr(step_arr, "ndim") and step_arr.ndim >= 1:
                                    active_spatial_n = int(step_arr.shape[0])
                            if active_spatial_n is not None:
                                ctx_dict["__active_spatial_n__"] = active_spatial_n
                            if time_arr is not None:
                                ctx_dict[TIME_TAG] = step_t
                            return evaluate_single_point_set(params, ctx_dict, key=rng_key)

                        return jax.vmap(
                            _eval_single_step,
                            in_axes=tuple(in_axes_list),
                        )(*windowed_ctx, t_wind, step_indices)
                    else:
                        ctx_dict = dict(metadata_ctx)
                        ctx_dict.update(augmented_passive_ctx)
                        ctx_dict["__step_index__"] = jnp.zeros((), dtype=jnp.int32)
                        active_spatial_n = None
                        for tag, arr in zip(spatial_tag_order, windowed_ctx):
                            if hasattr(arr, "ndim") and arr.ndim >= 2:
                                ctx_dict[tag] = arr[0]
                                if active_spatial_n is None and arr[0].ndim >= 1:
                                    active_spatial_n = int(arr[0].shape[0])
                            else:
                                ctx_dict[tag] = arr
                                if active_spatial_n is None and hasattr(arr, "ndim") and arr.ndim >= 2:
                                    active_spatial_n = int(arr.shape[0])
                        if active_spatial_n is not None:
                            ctx_dict["__active_spatial_n__"] = active_spatial_n
                        if time_arr is not None:
                            ctx_dict[TIME_TAG] = t_wind[0]
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
            fd_domain = _find_fd_domain(expr)
            batch_domain_map = getattr(fd_domain, "_batch_domain_map", None) if fd_domain is not None else None

            if batch_domain_map is not None:
                if len(batch_domain_map) != B:
                    raise ValueError(
                        "Mini-batching (batchsize) is not supported with "
                        "finite-difference derivatives on stacked domains. "
                        "Use batchsize=None (full-batch) or switch to "
                        "scheme='automatic_differentiation'."
                    )
                return _grouped_vmap(
                    scan_over_time,
                    spatial_ctx,
                    ctx_in_axes,
                    spatial_tag_order,
                    fd_domain,
                    batch_domain_map,
                    B,
                    key,
                    is_multi=False,
                )

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
        METADATA_TAGS = {
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
        }

        def evaluate_single_point_set(params, context_single, key):
            """Evaluate ALL expressions on one (N, D) context — shared evaluator."""
            evaluator = TraceEvaluator(params)
            # One evaluator → one JAX trace → XLA sees all constraints together
            return [evaluator.evaluate(expr, context_single, {}, key) for expr in exprs]

        # Collect tags from ALL expressions so batch inference is scoped.
        expr_tags = set()
        for _expr in exprs:
            expr_tags |= collect_tags(_expr)

        # Everything below mirrors compile_traced_expression.
        def compiled_fn(params, context=None, batchsize=None, key=None, min_consecutive=None):
            context = context or {}

            all_tags = set(expr_tags)
            for op in all_ops:
                if hasattr(op, "_collected_vars"):
                    for var in op._collected_vars:
                        all_tags.add(var.tag)

            tag_order = tuple(sorted(context.keys(), key=lambda t: (t not in all_tags, t)))
            ctx_tuple = tuple(context[tag] for tag in tag_order) if tag_order else ()

            batched_sizes = []
            for tag, arr in zip(tag_order, ctx_tuple):
                if tag == TIME_TAG or tag in METADATA_TAGS or tag not in expr_tags:
                    continue
                if hasattr(arr, "ndim") and arr.ndim >= 1:
                    batched_sizes.append(arr.shape[0])

            if not batched_sizes:
                # Inject __time_window__ for IntegralTime nodes that live in
                # time-only expressions (no spatial variables in expr_tags).
                if TIME_TAG in context:
                    context_with_window = dict(context)
                    context_with_window["__time_window__"] = jnp.asarray(context[TIME_TAG])
                    return evaluate_single_point_set(params, context_with_window, key=key)
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
                    if tag_name == TIME_TAG or tag_name in METADATA_TAGS or tag_name not in expr_tags:
                        return arr
                    if hasattr(arr, "ndim") and arr.ndim >= 1 and arr.shape[0] == B:
                        return arr[indices]
                    return arr

                ctx_tuple = tuple(subset_entry(t, a) for t, a in zip(tag_order, ctx_tuple))
                B = batchsize

            time_arr = None
            passive_ctx = {}
            spatial_tag_order = []
            spatial_ctx = []
            metadata_ctx = {}

            for tag, arr in zip(tag_order, ctx_tuple):
                if tag == TIME_TAG:
                    time_arr = jnp.asarray(arr)
                elif hasattr(arr, "ndim") and arr.ndim >= 1 and arr.shape[0] in (B, 1):
                    # Include both expression-tag arrays and any auxiliary arrays
                    # (e.g. FEM tensors not in expr_tags) that carry a matching batch
                    # axis — they must be vmapped over rather than passed as static
                    # passive context so per-sample evaluation sees the right slice.
                    spatial_tag_order.append(tag)
                    spatial_ctx.append(arr)
                elif tag in METADATA_TAGS:
                    metadata_ctx[tag] = arr
                else:
                    passive_ctx[tag] = arr

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

                W = T if min_consecutive is None else max(1, min(min_consecutive, T))
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
                    # Augment passive_ctx with the full time window so IntegralTime
                    # handlers can access all W time values even inside the per-step vmap.
                    augmented_passive_ctx = dict(passive_ctx)
                    augmented_passive_ctx["__time_window__"] = t_wind  # (W, 1)

                    # Pre-compute every TemporalDerivative.target on all W steps
                    # before the per-step vmap (post-order so nested TDs see their
                    # dependencies populated).  See the analogous block in
                    # compile_traced_expression for the rationale.
                    temporal_fd_cache: dict = {}
                    td_targets = _collect_temporal_derivative_targets(exprs)
                    if td_targets:
                        TraceEvaluator = _get_evaluator_class()
                        for td_target, tid in td_targets:
                            captured_cache = dict(temporal_fd_cache)
                            in_axes_pc = []
                            for arr in windowed_ctx:
                                if hasattr(arr, "ndim") and arr.ndim >= 3:
                                    in_axes_pc.append(0)
                                else:
                                    in_axes_pc.append(None)
                            in_axes_pc.append(0)  # t_wind axis
                            in_axes_pc.append(0)  # step index axis
                            step_indices = jnp.arange(t_wind.shape[0], dtype=jnp.int32)

                            def _eval_target_at_step(*args, _target=td_target, _cache=captured_cache):
                                step_idx = args[-1]
                                step_t = args[-2]
                                step_spatial = args[:-2]
                                ctx_dict = dict(metadata_ctx)
                                ctx_dict.update(passive_ctx)
                                ctx_dict["__time_window__"] = t_wind
                                ctx_dict["__temporal_fd_cache__"] = _cache
                                ctx_dict["__step_index__"] = step_idx
                                for tag, step_arr in zip(spatial_tag_order, step_spatial):
                                    ctx_dict[tag] = step_arr
                                if time_arr is not None:
                                    ctx_dict[TIME_TAG] = step_t
                                return TraceEvaluator(params).evaluate(_target, ctx_dict, {}, rng_key)

                            u_window = jax.vmap(
                                _eval_target_at_step,
                                in_axes=tuple(in_axes_pc),
                            )(*windowed_ctx, t_wind, step_indices)
                            temporal_fd_cache[tid] = u_window

                        augmented_passive_ctx["__temporal_fd_cache__"] = temporal_fd_cache

                    if W > 1:
                        # Vmap over the W (time-window) dimension so each
                        # step sees (N, D) spatial + (1,) time — identical
                        # to the W=1 case.  This ensures FD operators and
                        # model branch networks that flatten their input
                        # (e.g. DeepONet) work correctly.
                        in_axes_list = []
                        for arr in windowed_ctx:
                            if hasattr(arr, "ndim") and arr.ndim >= 3:
                                in_axes_list.append(0)
                            else:
                                in_axes_list.append(None)
                        in_axes_list.append(0)  # for t_wind
                        in_axes_list.append(0)  # for step indices
                        step_indices = jnp.arange(W, dtype=jnp.int32)

                        def _eval_single_step(*step_spatial_and_t_and_idx):
                            step_idx = step_spatial_and_t_and_idx[-1]
                            step_t = step_spatial_and_t_and_idx[-2]
                            step_spatial = step_spatial_and_t_and_idx[:-2]
                            ctx_dict = dict(metadata_ctx)
                            ctx_dict.update(augmented_passive_ctx)
                            ctx_dict["__step_index__"] = step_idx
                            active_spatial_n = None
                            for tag, step_arr in zip(spatial_tag_order, step_spatial):
                                ctx_dict[tag] = step_arr
                                if active_spatial_n is None and hasattr(step_arr, "ndim") and step_arr.ndim >= 1:
                                    active_spatial_n = int(step_arr.shape[0])
                            if active_spatial_n is not None:
                                ctx_dict["__active_spatial_n__"] = active_spatial_n
                            if time_arr is not None:
                                ctx_dict[TIME_TAG] = step_t  # (1,) per step
                            return evaluate_single_point_set(params, ctx_dict, key=rng_key)

                        return jax.vmap(
                            _eval_single_step,
                            in_axes=tuple(in_axes_list),
                        )(*windowed_ctx, t_wind, step_indices)
                    else:
                        # W == 1: squeeze the window dimension
                        ctx_dict = dict(metadata_ctx)
                        ctx_dict.update(augmented_passive_ctx)
                        ctx_dict["__step_index__"] = jnp.zeros((), dtype=jnp.int32)
                        active_spatial_n = None
                        for tag, arr in zip(spatial_tag_order, windowed_ctx):
                            if hasattr(arr, "ndim") and arr.ndim >= 2:
                                ctx_dict[tag] = arr[0]
                                if active_spatial_n is None and arr[0].ndim >= 1:
                                    active_spatial_n = int(arr[0].shape[0])
                            else:
                                ctx_dict[tag] = arr
                                if active_spatial_n is None and hasattr(arr, "ndim") and arr.ndim >= 2:
                                    active_spatial_n = int(arr.shape[0])
                        if active_spatial_n is not None:
                            ctx_dict["__active_spatial_n__"] = active_spatial_n
                        if time_arr is not None:
                            ctx_dict[TIME_TAG] = t_wind[0]
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

            fd_domain = _find_fd_domain(exprs)
            batch_domain_map = getattr(fd_domain, "_batch_domain_map", None) if fd_domain is not None else None

            if batch_domain_map is not None:
                if len(batch_domain_map) != B:
                    raise ValueError(
                        "Mini-batching (batchsize) is not supported with "
                        "finite-difference derivatives on stacked domains. "
                        "Use batchsize=None (full-batch) or switch to "
                        "scheme='automatic_differentiation'."
                    )
                return _grouped_vmap(
                    scan_over_time,
                    spatial_ctx,
                    ctx_in_axes,
                    spatial_tag_order,
                    fd_domain,
                    batch_domain_map,
                    B,
                    key,
                    is_multi=True,
                )

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
