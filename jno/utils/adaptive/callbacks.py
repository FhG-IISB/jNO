"""Callback classes for monitoring and checkpointing during training."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import jax
import numpy as np

from ..config import get_wandb_run, wandb_alert, wandb_log


class Callback:
    """Base callback class.

    Subclass and override the hooks you need.  Every hook receives
    keyword arguments whose contents depend on the hook and the
    caller.  This keeps the interface decoupled from any particular
    solver implementation.
    """

    def on_solve_begin(self, **kwargs) -> None:
        """Called once after ``solve()`` finishes setup, before the training loop.

        Keyword Args:
            compiled_constraints_fn: Combined compiled JAX function for all constraints.
            n_constraints (int): Number of constraint terms.
            batchsize: Mini-batch size (``None`` for full-batch).
            frozen: Frozen parameter pytree (from ``eqx.partition``).
            static: Static (non-array) pytree.
            trainable: Initial trainable parameter pytree (use for pre-compilation only).
            context: Domain context (use for pre-compilation only).
            rng: PRNG key.
            min_consecutive (int): Minimum consecutive time steps per constraint call.
            constraint_exprs: List of the solver's raw constraint placeholders
                (the exact Python objects passed to ``jno.core([...])``).
                Callbacks that take a ``constraints=`` arg validate user input
                against this list by Python identity.
            all_ops: List of OperationDef instances (the solver's compiled op set);
                callbacks may pass this to ``TraceCompiler.compile_multi_expression``
                to evaluate user-supplied placeholder expressions during training.
            domain: The solver's :class:`~jno.domain.domain` instance (read-only;
                use only for shape / metadata inspection during pre-compilation).
        """

    def on_epoch_end(self, **kwargs) -> bool:
        """Called at the end of every outer training step.

        Keyword Args:
            epoch (int): Current epoch number (0-indexed).
            trainable: Trainable parameter pytree (Equinox partition).
            opt_states: ``dict[str, optax.OptState]`` per-model optimizer states.
            rng: Current JAX PRNG key.
            total_loss: Scalar total loss (JAX array, still on device).
            individual_losses: Per-constraint losses (JAX array).
            log: Logger instance (when called from ``core.solve``).

        Returns:
            ``True`` to request early termination of the training loop,
            ``False`` (default) to continue.
        """
        return False

    def on_before_update(self, *, grads, trainable, context, rng, epoch, **kwargs):
        """Called between gradient computation and the optimizer update.

        Subclasses may return a modified ``grads`` dict to redirect the
        optimizer (e.g. to apply a preconditioner), or ``None`` to leave the
        gradients unchanged.  The returned dict must have the same pytree
        structure as ``grads``.

        Called only when the solver detects that at least one active callback
        overrides this method (it is a no-op in the base class).  Requires
        ``inner_steps=1`` and no Bayesian models.

        Keyword Args:
            grads: Parameter gradient pytree ``{lid: model_params}``.
            trainable: Current trainable parameter pytree.
            context: Domain context (current batch).
            rng: Current JAX PRNG key (Python-side; not consumed here).
            epoch (int): Current outer epoch (Python integer, 0-indexed).
            total_loss: The scalar loss at this step, as descended.
            individual_losses: Per-entry values, one per ``jno.core`` constraint entry --
                **including** any ``jno.le``/``jno.ge`` inequality constraints, which are
                evaluated with everything else but held out of ``total_loss``. A constrained
                optimiser reads its ``g_j`` here rather than recomputing them.

        Returns:
            Modified ``grads`` dict, or ``None`` to leave unchanged.
        """
        return None

    def on_training_end(self, **kwargs) -> None:
        """Called once after the training loop finishes."""


def _resolve_constraint_indices(user_constraints, solver_constraints, callback_name: str) -> list[int]:
    """Map user-supplied constraint placeholders to their solver-side indices.

    Matches by Python identity — i.e. the user must pass the *same* Python
    object that was given to ``jno.core([...])`` (assign the constraint to a
    variable rather than re-accessing ``.mse``, which returns a fresh
    placeholder every time).  Raises ``ValueError`` with a clear hint on
    mismatch.

    Args:
        user_constraints: List of placeholder expressions, or ``None`` for "all".
        solver_constraints: ``solver._constraint_exprs`` (the solver's stored
            constraint list).
        callback_name: Used in the error message so users know which callback
            raised.

    Returns:
        List of integer indices into ``solver_constraints``.  When
        ``user_constraints is None`` the full range ``[0, len(solver_constraints))``.
    """
    if user_constraints is None:
        return list(range(len(solver_constraints)))
    indices: list[int] = []
    for c in user_constraints:
        found = -1
        for i, ce in enumerate(solver_constraints):
            if c is ce:
                found = i
                break
        if found == -1:
            raise ValueError(
                f"{callback_name}: constraint {c!r} not found among the "
                f"solver's compiled constraints.  Pass the same Python "
                f"object that was given to jno.core([...]) — assign your "
                f"constraint to a variable rather than re-accessing .mse "
                f"(which returns a fresh placeholder each access)."
            )
        indices.append(found)
    return indices


# ---------------------------------------------------------------------------
# Orbax-based checkpoint callback
# ---------------------------------------------------------------------------


class CheckpointCallback(Callback):
    """Save training checkpoints to disk using `orbax-checkpoint`_.

    This callback periodically writes model parameters, optimizer states,
    and RNG state to a directory managed by an Orbax ``CheckpointManager``.
    Checkpoints are saved asynchronously by default and old checkpoints
    are automatically cleaned up.

    Install the optional dependency with::

        pip install orbax-checkpoint

    Args:
        directory: Root directory for checkpoints (created if needed).
            When ``None`` (default), uses ``<jno.setup() run dir>/checkpoints``.
        save_interval_epochs: Save a checkpoint every *n* epochs.
            Epochs refer to *outer* training steps (i.e. after
            ``inner_steps`` gradient updates each).
        max_to_keep: Maximum number of checkpoints retained on disk.
            Oldest are deleted first, unless *best_fn* is set.
        best_fn: Optional callable ``(metrics: dict) -> float`` used to
            rank checkpoints.  The checkpoint with the **lowest** returned
            value is considered the best and will always be kept.  For
            example, ``best_fn=lambda m: m['total_loss']`` keeps the
            checkpoint with the lowest total loss.
        async_checkpointing: If *True* (default), writes happen in a
            background thread so training is not blocked.

    When a W&B run is active (see :func:`jno.setup(wandb=True) <jno.utils.config.setup>`),
    each checkpoint is automatically uploaded as a versioned ``checkpoint``
    artifact.

    Example::

        cb = jno.callbacks.CheckpointCallback(
            directory=\"./runs/ckpt\",
            save_interval_epochs=500,
            max_to_keep=3,
            best_fn=lambda m: m[\"total_loss\"],
        )
        solver.solve(epochs=5000, callbacks=[cb])

        # Later, restore:
        restored = cb.restore()   # latest
        restored = cb.restore(step=2000)

    .. _orbax-checkpoint: https://github.com/google/orbax
    """

    def __init__(
        self,
        directory: Optional[str] = None,
        save_interval_epochs: int = 500,
        max_to_keep: int = 3,
        best_fn: Optional[Any] = None,
        async_checkpointing: bool = True,
    ) -> None:
        try:
            import orbax.checkpoint as ocp
        except ImportError as exc:
            raise ImportError(
                "orbax-checkpoint is required for CheckpointCallback. Install it with:  pip install orbax-checkpoint"
            ) from exc

        if directory is None:
            from ..logger import get_logger

            log = get_logger()
            log_path = getattr(log, "path", None)
            if log_path and str(log_path):
                directory = os.path.join(str(log_path), "checkpoints")
            else:
                raise ValueError(
                    "No directory given and no jno.setup() run directory found. "
                    "Either pass directory= or call jno.setup(__file__) first."
                )

        self._ocp = ocp
        self._directory = os.path.abspath(directory)
        self._save_interval = save_interval_epochs
        self._best_fn = best_fn

        opts_kwargs: Dict[str, Any] = dict(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_epochs,
            enable_async_checkpointing=async_checkpointing,
        )
        if best_fn is not None:
            opts_kwargs["best_fn"] = best_fn
            opts_kwargs["best_mode"] = "min"
        options = ocp.CheckpointManagerOptions(**opts_kwargs)
        self._manager = ocp.CheckpointManager(
            self._directory,
            options=options,
        )

    # -- hooks ---------------------------------------------------------------

    def on_epoch_end(self, **kwargs) -> None:
        ocp = self._ocp
        epoch: int = kwargs["epoch"]
        trainable = kwargs["trainable"]
        opt_states = kwargs["opt_states"]
        rng = kwargs["rng"]
        total_loss = kwargs["total_loss"]
        individual_losses = kwargs["individual_losses"]

        # Build a plain pytree of arrays (Orbax StandardSave needs this).
        # trainable and opt_states are already JAX-compatible pytrees.
        pytree = {
            "trainable": trainable,
            "opt_states": opt_states,
            "rng": rng,
        }
        metadata = {
            "epoch": int(epoch),
            "total_loss": float(jax.device_get(total_loss)),
            "individual_losses": [float(v) for v in jax.device_get(individual_losses)],
            "timestamp": time.time(),
            "checkpoint_dir": os.path.join(self._directory, str(epoch)),
        }

        # CheckpointManager.save internally checks should_save(step)
        # based on save_interval_steps, so we call it every epoch and
        # let the manager decide.
        saved = self._manager.save(
            epoch,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(pytree),
                metadata=ocp.args.JsonSave(metadata),
            ),
            metrics=metadata if self._best_fn is not None else None,
        )

        # Upload checkpoint as a W&B artifact when a save happened.
        if saved:
            self._upload_wandb_artifact(epoch, metadata=metadata)
            self._log_wandb_histograms(trainable, epoch)

    def on_training_end(self, **kwargs) -> None:
        self._manager.wait_until_finished()

    # -- wandb ---------------------------------------------------------------

    def _upload_wandb_artifact(self, epoch: int, metadata: Optional[Dict] = None) -> None:
        """Upload the latest checkpoint directory as a W&B artifact."""

        run = get_wandb_run()
        if run is None:
            return

        try:
            import wandb  # type: ignore[import-untyped]
        except ImportError:
            return

        self._manager.wait_until_finished()
        ckpt_path = os.path.join(self._directory, str(epoch))
        if not os.path.isdir(ckpt_path):
            return
        artifact = wandb.Artifact(
            f"checkpoint-{epoch}",
            type="checkpoint",
            metadata=metadata if metadata is not None else {"epoch": epoch},
        )
        artifact.add_dir(ckpt_path)
        run.log_artifact(artifact)

    def _log_wandb_histograms(self, trainable: Any, epoch: int) -> None:
        """Log per-layer weight histograms to W&B."""

        run = get_wandb_run()
        if run is None:
            return

        try:
            import wandb  # type: ignore[import-untyped]
        except ImportError:
            return

        histograms: dict = {}
        for model_key, model_params in trainable.items():
            leaves = jax.tree_util.tree_leaves_with_path(model_params)
            for path, leaf in leaves:
                name = "/".join(str(k) for k in path)
                arr = np.asarray(jax.device_get(leaf)).ravel()
                if arr.size > 0:
                    histograms[f"weights/{model_key}/{name}"] = wandb.Histogram(arr.tolist())
        if histograms:
            run.log(histograms, step=epoch)

    # -- public API ----------------------------------------------------------

    def restore(self, step: Optional[int] = None) -> Dict[str, Any]:
        """Restore a checkpoint.

        Args:
            step: Checkpoint step to restore.  ``None`` (default)
                restores the latest available checkpoint.

        Returns:
            Dictionary with keys ``trainable``, ``opt_states``, ``rng``,
            and ``metadata``.
        """
        if step is None:
            step = self._manager.latest_step()
        if step is None:
            raise FileNotFoundError(f"No checkpoints found in {self._directory}")

        restored = self._manager.restore(step)
        return {
            "trainable": restored.state["trainable"],
            "opt_states": restored.state["opt_states"],
            "rng": restored.state["rng"],
            "metadata": restored.metadata,
        }

    @property
    def latest_step(self) -> Optional[int]:
        """Return the latest checkpoint step, or ``None``."""
        return self._manager.latest_step()

    @property
    def all_steps(self):
        """Return a list of all available checkpoint steps."""
        return self._manager.all_steps()

    def close(self) -> None:
        """Close the checkpoint manager (waits for pending writes)."""
        self._manager.close()


# ---------------------------------------------------------------------------
# Early stopping callback
# ---------------------------------------------------------------------------


class EarlyStoppingCallback(Callback):
    """Stop training when a monitored metric stops improving.

    Monitors a scalar metric (by default the total loss) each epoch and
    signals the training loop to stop once the metric has not improved
    for *patience* consecutive checks.

    Three stopping strategies are available via the *mode* parameter:

    ``"min"``
        Improvement means the metric decreased by more than *min_delta*.
        Use for losses.
    ``"max"``
        Improvement means the metric increased by more than *min_delta*.
        Use for accuracy-like metrics.
    ``"rel"``
        Improvement means the metric decreased by a factor of at least
        *min_delta* relative to the best value so far
        (i.e. ``new < best * (1 - min_delta)``).  Useful when the loss
        spans many orders of magnitude, which is common in PINN training.

    Args:
        patience: Number of epochs with no improvement after which
            training is stopped.  Default ``500``.
        min_delta: Minimum change to qualify as an improvement.
            For ``"min"``/``"max"`` this is an absolute threshold;
            for ``"rel"`` it is a relative fraction.  Default ``0.0``.
        mode: One of ``"min"``, ``"max"``, or ``"rel"``.
            Default ``"min"``.
        metric_fn: Callable that extracts the scalar metric from the
            ``on_epoch_end`` keyword arguments.  Default extracts
            ``total_loss`` (transferred to host).
        baseline: An optional baseline value.  Training will stop if
            the metric never improves beyond this value.
        verbose: If ``True``, log a message when stopping.

    Example::

        cb = jno.callback.early_stopping(patience=1000, min_delta=1e-6)
        solver.solve(epochs=100_000, callbacks=[cb])

        print(cb.stopped_epoch)   # epoch at which training was halted
        print(cb.best_metric)     # best metric value observed
    """

    def __init__(
        self,
        patience: int = 500,
        min_delta: float = 0.0,
        mode: str = "min",
        metric_fn: Optional[Any] = None,
        baseline: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        if mode not in ("min", "max", "rel"):
            raise ValueError(f"mode must be 'min', 'max', or 'rel', got {mode!r}")

        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.verbose = verbose

        if metric_fn is None:
            self._metric_fn = lambda **kw: float(jax.device_get(kw["total_loss"]))
        else:
            self._metric_fn = metric_fn

        self.best_metric: Optional[float] = baseline
        self.stopped_epoch: Optional[int] = None
        self._wait = 0
        self._stopped = False

    # -- comparison helpers --------------------------------------------------

    def _is_improvement(self, current: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return current < self.best_metric - self.min_delta
        elif self.mode == "max":
            return current > self.best_metric + self.min_delta
        else:  # rel
            return current < self.best_metric * (1.0 - self.min_delta)

    # -- hooks ---------------------------------------------------------------

    def on_epoch_end(self, **kwargs) -> bool:
        current = self._metric_fn(**kwargs)
        epoch: int = kwargs["epoch"]

        if self._is_improvement(current):
            self.best_metric = current
            self._wait = 0
        else:
            self._wait += 1

        if self._wait >= self.patience:
            self._stopped = True
            self.stopped_epoch = epoch
            if self.verbose:
                log = kwargs.get("log")
                msg = (
                    f"Early stopping at epoch {epoch}: "
                    f"no improvement for {self.patience} epochs "
                    f"(best={self.best_metric:.6e})"
                )
                if log is not None:
                    log.info(msg)
            wandb_alert(
                "Early stopping",
                f"Stopped at epoch {epoch} — no improvement for {self.patience} epochs (best={self.best_metric:.6e})",
                level="WARN",
            )
            return True  # signal stop

        return False

    @property
    def has_stopped(self) -> bool:
        """Whether early stopping was triggered."""
        return self._stopped


# ---------------------------------------------------------------------------
# Live-value mixin — shared by all explainability trackers
# ---------------------------------------------------------------------------


class _LiveValue:
    """Expose the latest computed metric values to other in-loop components.

    Adaptive weight schemes (see :mod:`jno.utils.adaptive.weights`) read
    ``tracker.value`` inside their host callback to balance losses based on
    a tracker's current measurement (NTK eigenvalues, per-loss gradient
    norms, etc.).  Until the first time the metric fires, ``value`` is
    ``None`` and ``latest_epoch`` is ``None`` — consumers must handle that
    cold-start case (typically by falling back to uniform weights).
    """

    def _init_live_value(self) -> None:
        self._latest: Optional[Dict[str, Any]] = None
        self._latest_epoch: Optional[int] = None

    def _publish(self, epoch: int, value: Dict[str, Any]) -> None:
        self._latest = value
        self._latest_epoch = int(epoch)

    @property
    def value(self) -> Optional[Dict[str, Any]]:
        """Latest computed metric values (numpy dict) or ``None`` until the
        first interval fires."""
        return self._latest

    @property
    def latest_epoch(self) -> Optional[int]:
        """Epoch index when :attr:`value` was last updated, or ``None``."""
        return self._latest_epoch


# ---------------------------------------------------------------------------
# Shared base for gradient-analysis callbacks (1–3)
# ---------------------------------------------------------------------------


class _PerLossGradCallback(Callback, _LiveValue):
    """Base class for callbacks that require per-loss gradients.

    Builds and pre-compiles a single ``jacrev``-based function in
    ``on_solve_begin``; subclasses extract the metric they care about
    in ``on_epoch_end``.
    """

    def __init__(self, interval: int, mask) -> None:
        self._interval = interval
        self._mask = mask
        self._grad_fn = None
        self._epochs: list = []
        self._init_live_value()

    def on_solve_begin(self, **kwargs) -> None:
        from jno.utils.explainability import make_per_loss_grad_fn

        self._constraint_names: list = kwargs.get("constraint_names", [])
        self._grad_fn = jax.jit(
            make_per_loss_grad_fn(
                kwargs["compiled_constraints_fn"],
                kwargs["n_constraints"],
                kwargs["batchsize"],
                kwargs["frozen"],
                kwargs["static"],
                param_mask=self._mask,
                min_consecutive=kwargs.get("min_consecutive", 1),
            )
        )
        self._grad_fn.lower(kwargs["trainable"], kwargs["context"], kwargs["rng"]).compile()

    def _compute(self, epoch: int, trainable, context, rng):
        """Compute (norms, cos_matrix, alignment) if this epoch is due."""
        if self._grad_fn is None or epoch % self._interval != 0:
            return None
        return jax.device_get(self._grad_fn(trainable, context, rng))


# ---------------------------------------------------------------------------
# Callback 1 — Gradient norms
# ---------------------------------------------------------------------------


class GradientNormsCallback(_PerLossGradCallback):
    """Track the gradient norm of each loss term during training.

    At every ``interval`` outer steps, computes ``‖∇L_i‖₂`` for each
    constraint *i* by differentiating through the constraint function.

    Args:
        interval: Compute every *n* outer training steps.  Default ``100``.
        mask: Optional pytree of booleans matching ``crux.models`` structure.
            When set, gradients are computed only for the selected parameter
            subset, reducing cost for large models.

    Example::

        cb = jno.callbacks.gradient_norms(interval=100)
        crux.solve(5000, callbacks=[cb])
        print(cb.result["norms"])   # (n_samples, n_constraints)
    """

    def __init__(self, interval: int = 100, mask=None) -> None:
        super().__init__(interval, mask)
        self._norms: list = []

    def on_epoch_end(self, **kwargs) -> bool:
        out = self._compute(kwargs["epoch"], kwargs["trainable"], kwargs["context"], kwargs["rng"])
        if out is not None:
            norms, _, _ = out
            epoch = kwargs["epoch"]
            norms_np = np.asarray(norms)
            self._epochs.append(epoch)
            self._norms.append(norms_np)
            self._publish(epoch, {"norms": norms_np})
            if get_wandb_run() is not None:
                cn = getattr(self, "_constraint_names", [])
                wandb_log(
                    {
                        f"explainability/gradient_norm/{cn[i] if i < len(cn) and cn[i] else f'constraint_{i}'}": float(v)
                        for i, v in enumerate(norms)
                    },
                    step=epoch,
                )
        return False

    @property
    def result(self) -> dict:
        """Collected data as numpy arrays.

        Keys:
            ``epochs`` — ``(S,)`` int array of sampled outer steps.
            ``norms``  — ``(S, N)`` float32 array of per-loss gradient norms.
        """
        return {
            "epochs": np.array(self._epochs),
            "norms": np.stack(self._norms) if self._norms else np.zeros((0,)),
        }


# ---------------------------------------------------------------------------
# Callback 2 — Pairwise cosine similarity
# ---------------------------------------------------------------------------


class CosSimilarityCallback(_PerLossGradCallback):
    """Track pairwise gradient cosine similarity between loss terms.

    Computes the full ``(N × N)`` cosine similarity matrix (upper triangle
    carries the pairwise values; diagonal is always 1).

    Args:
        interval: Compute every *n* outer training steps.  Default ``100``.
        mask: Optional pytree of booleans — see :class:`GradientNormsCallback`.

    Example::

        cb = jno.callbacks.cos_similarity(interval=100)
        crux.solve(5000, callbacks=[cb])
        print(cb.result["cos_sim_matrix"])   # (n_samples, N, N)
    """

    def __init__(self, interval: int = 100, mask=None) -> None:
        super().__init__(interval, mask)
        self._cos: list = []

    def on_epoch_end(self, **kwargs) -> bool:
        out = self._compute(kwargs["epoch"], kwargs["trainable"], kwargs["context"], kwargs["rng"])
        if out is not None:
            _, cos_matrix, _ = out
            epoch = kwargs["epoch"]
            self._epochs.append(epoch)
            cos_np = np.asarray(cos_matrix)
            self._cos.append(cos_np)
            self._publish(epoch, {"cos_sim_matrix": cos_np})
            if get_wandb_run() is not None:
                N = cos_np.shape[0]
                cn = getattr(self, "_constraint_names", [])

                def _clabel(k):
                    return cn[k] if k < len(cn) and cn[k] else str(k)

                wb: dict = {
                    f"explainability/cos_sim/{_clabel(i)}_{_clabel(j)}": float(cos_np[i, j])
                    for i in range(N)
                    for j in range(i + 1, N)
                }
                try:
                    import matplotlib.pyplot as plt
                    import wandb as _wandb

                    fig, ax = plt.subplots(figsize=(max(3, N), max(3, N)))
                    im = ax.imshow(cos_np, vmin=-1, vmax=1, cmap="RdBu_r")
                    fig.colorbar(im, ax=ax)
                    ax.set_title(f"Gradient cosine similarity (epoch {epoch})")
                    wb["explainability/cos_sim_matrix"] = _wandb.Image(fig)
                    plt.close(fig)
                except ImportError:
                    pass
                wandb_log(wb, step=epoch)
        return False

    @property
    def result(self) -> dict:
        """Collected data as numpy arrays.

        Keys:
            ``epochs``         — ``(S,)`` int array of sampled outer steps.
            ``cos_sim_matrix`` — ``(S, N, N)`` float32 pairwise cosine similarity.
        """
        return {
            "epochs": np.array(self._epochs),
            "cos_sim_matrix": np.stack(self._cos) if self._cos else np.zeros((0,)),
        }


# ---------------------------------------------------------------------------
# Callback 3 — Total gradient alignment
# ---------------------------------------------------------------------------


class GradientAlignmentCallback(_PerLossGradCallback):
    """Track the total gradient alignment scalar during training.

    Computes ``2‖(1/N) Σ ĝᵢ‖² − 1`` with ``ĝᵢ = gᵢ / ‖gᵢ‖`` (Eq. 3.1,
    [2502.00604]), a value in ``[-1, 1]`` that measures how well all loss
    gradients point in the same direction.  Near ``1`` means perfect
    alignment; ``0`` means orthogonal; near ``-1`` means anti-aligned
    (gradients actively cancel).  For ``N = 2`` this reduces to the
    ordinary cosine similarity (Proposition 1 of the paper).

    Args:
        interval: Compute every *n* outer training steps.  Default ``100``.
        mask: Optional pytree of booleans — see :class:`GradientNormsCallback`.

    Example::

        cb = jno.callbacks.gradient_alignment(interval=100)
        crux.solve(5000, callbacks=[cb])
        print(cb.result["alignment"])   # (n_samples,)
    """

    def __init__(self, interval: int = 100, mask=None) -> None:
        super().__init__(interval, mask)
        self._align: list = []

    def on_epoch_end(self, **kwargs) -> bool:
        out = self._compute(kwargs["epoch"], kwargs["trainable"], kwargs["context"], kwargs["rng"])
        if out is not None:
            _, _, alignment = out
            epoch = kwargs["epoch"]
            self._epochs.append(epoch)
            align_v = float(alignment)
            self._align.append(align_v)
            self._publish(epoch, {"alignment": align_v})
            if get_wandb_run() is not None:
                wandb_log({"explainability/gradient_alignment": align_v}, step=epoch)
        return False

    @property
    def result(self) -> dict:
        """Collected data as numpy arrays.

        Keys:
            ``epochs``    — ``(S,)`` int array of sampled outer steps.
            ``alignment`` — ``(S,)`` float32 total gradient alignment scalar.
        """
        return {
            "epochs": np.array(self._epochs),
            "alignment": np.array(self._align),
        }


# ---------------------------------------------------------------------------
# Callback 4 — Loss landscape
# ---------------------------------------------------------------------------


class LossLandscapeCallback(Callback, _LiveValue):
    """Evaluate the 2-D loss landscape periodically during training.

    At every ``interval`` outer steps, samples two random filter-normalized
    directions in parameter space and evaluates the total loss on an
    ``(n_grid × n_grid)`` perturbation grid centred on the current
    parameters.  The directions change each call (stochastic sampling).

    This is expensive: each call requires ``n_grid²`` forward passes.
    Choose a large ``interval`` (e.g. 500–1000) for typical runs.

    Args:
        interval: Compute every *n* outer training steps.  Default ``500``.
        mask: Optional pytree of booleans matching ``crux.models`` structure.
            When set, only the selected parameters are perturbed; the rest
            are held fixed.  Strongly recommended for large models.
        n_grid: Number of grid points per axis.  Default ``15``.
        alpha_range: Perturbation scale in units of ``‖θ_selected‖``.
            Default ``1.0``.

    Example::

        cb = jno.callbacks.loss_landscape(interval=500, n_grid=15)
        crux.solve(5000, callbacks=[cb])
        landscapes = cb.result["landscapes"]   # (n_samples, 15, 15)
    """

    def __init__(
        self,
        interval: int = 500,
        mask=None,
        n_grid: int = 15,
        alpha_range: float = 1.0,
    ) -> None:
        self._interval = interval
        self._mask = mask
        self._n_grid = n_grid
        self._alpha_range = alpha_range
        self._landscape_fn = None
        self._epochs: list = []
        self._landscapes: list = []
        self._init_live_value()

    def on_solve_begin(self, **kwargs) -> None:
        from jno.utils.explainability import make_landscape_fn

        self._landscape_fn = jax.jit(
            make_landscape_fn(
                kwargs["compiled_constraints_fn"],
                kwargs["batchsize"],
                kwargs["frozen"],
                kwargs["static"],
                n_grid=self._n_grid,
                alpha_range=self._alpha_range,
                param_mask=self._mask,
                min_consecutive=kwargs.get("min_consecutive", 1),
            )
        )
        self._landscape_fn.lower(kwargs["trainable"], kwargs["context"], kwargs["rng"]).compile()

    def on_epoch_end(self, **kwargs) -> bool:
        if self._landscape_fn is None:
            return False
        epoch = kwargs["epoch"]
        if epoch % self._interval != 0:
            return False
        ls = jax.device_get(self._landscape_fn(kwargs["trainable"], kwargs["context"], kwargs["rng"]))
        self._epochs.append(epoch)
        ls_np = np.asarray(ls)
        self._landscapes.append(ls_np)
        self._publish(epoch, {"landscape": ls_np})
        if get_wandb_run() is not None:
            wb: dict = {}
            try:
                import matplotlib.pyplot as plt
                import wandb as _wandb

                fig, ax = plt.subplots()
                im = ax.imshow(ls_np, origin="lower", aspect="auto")
                fig.colorbar(im, ax=ax)
                ax.set_title(f"Loss landscape (epoch {epoch})")
                wb["explainability/loss_landscape"] = _wandb.Image(fig)
                plt.close(fig)
            except ImportError:
                wb["explainability/landscape_min"] = float(ls_np.min())
                wb["explainability/landscape_max"] = float(ls_np.max())
                wb["explainability/landscape_mean"] = float(ls_np.mean())
            wandb_log(wb, step=epoch)
        return False

    @property
    def result(self) -> dict:
        """Collected data as numpy arrays.

        Keys:
            ``epochs``     — ``(K,)`` int array of sampled outer steps.
            ``landscapes`` — ``(K, n_grid, n_grid)`` float32 loss landscape grids.
        """
        return {
            "epochs": np.array(self._epochs),
            "landscapes": np.stack(self._landscapes) if self._landscapes else np.zeros((0,)),
        }


# ---------------------------------------------------------------------------
# Per-constraint residual statistics
# ---------------------------------------------------------------------------


class ResidualStatsCallback(Callback, _LiveValue):
    """Track per-constraint residual distributions during training.

    At every ``interval`` outer steps, evaluates each constraint's
    *un-reduced* residual array (shape ``(B, T, ...)``) and records four
    summary statistics per constraint — mean, std, max, and 99th percentile
    — plus a histogram of the raw residual magnitudes (Sec. 3, [2207.10289]).

    A constraint whose ``max`` or ``p99`` stays orders of magnitude above
    the others indicates a region of the domain where the PDE is poorly
    satisfied; this complements :class:`GradientNormsCallback` which only
    reflects a constraint's *aggregated* contribution to the gradient.

    Args:
        interval: Compute every *n* outer training steps. Default ``100``.
        constraints: Optional list of constraint placeholders to scope the
            callback to a subset of the solver's constraints.  Pass the
            *same* Python objects that were given to ``jno.core([...])`` —
            assign your constraints to variables rather than re-accessing
            ``.mse`` (which returns a fresh placeholder each access).
            When ``None`` (default) all constraints are tracked.

    Example::

        pde_loss = pde.mse                       # assign once
        bc_loss  = bc.mse
        solver = jno.core([pde_loss, bc_loss])
        cb = jno.callbacks.residual_stats(interval=100, constraints=[pde_loss])
        crux.solve(5000, callbacks=[cb])
        print(cb.result["maxes"])   # (n_samples, 1)  — just pde_loss
    """

    def __init__(self, interval: int = 100, constraints=None) -> None:
        self._interval = interval
        self._user_constraints = constraints  # resolved in on_solve_begin
        self._indices: list[int] = []  # populated in on_solve_begin
        self._fn = None
        self._n_constraints = 0
        self._epochs: list = []
        self._means: list = []
        self._stds: list = []
        self._maxes: list = []
        self._p99: list = []
        self._init_live_value()

    def on_solve_begin(self, **kwargs) -> None:
        from jno.utils.explainability import make_residual_stats_fn

        self._n_constraints = kwargs["n_constraints"]
        self._constraint_names: list = kwargs.get("constraint_names", [])
        # Resolve the optional constraint subset to solver-side indices.
        # `constraint_exprs` is added by the solver hook; defensively fall back
        # to the full range if absent (older solvers without the hook).
        solver_constraints = kwargs.get("constraint_exprs")
        if solver_constraints is None:
            solver_constraints = [None] * self._n_constraints
        self._indices = _resolve_constraint_indices(self._user_constraints, solver_constraints, "residual_stats")

        self._fn = jax.jit(
            make_residual_stats_fn(
                kwargs["compiled_constraints_fn"],
                kwargs["n_constraints"],
                kwargs["batchsize"],
                kwargs["frozen"],
                kwargs["static"],
                min_consecutive=kwargs.get("min_consecutive", 1),
            )
        )
        self._fn.lower(kwargs["trainable"], kwargs["context"], kwargs["rng"]).compile()

    def on_epoch_end(self, **kwargs) -> bool:
        if self._fn is None:
            return False
        epoch = kwargs["epoch"]
        if epoch % self._interval != 0:
            return False
        means, stds, maxes, p99, raw = self._fn(kwargs["trainable"], kwargs["context"], kwargs["rng"])
        means_np = np.asarray(jax.device_get(means))
        stds_np = np.asarray(jax.device_get(stds))
        maxes_np = np.asarray(jax.device_get(maxes))
        p99_np = np.asarray(jax.device_get(p99))
        raw_np = [np.asarray(jax.device_get(r)) for r in raw]

        # Slice each scalar array to the requested constraint subset.  Raw
        # residuals are a list of variable-length arrays, so they're
        # picked-by-index rather than fancy-indexed.
        idx = self._indices
        means_sel = means_np[idx]
        stds_sel = stds_np[idx]
        maxes_sel = maxes_np[idx]
        p99_sel = p99_np[idx]
        raw_sel = [raw_np[i] for i in idx]

        self._epochs.append(epoch)
        self._means.append(means_sel)
        self._stds.append(stds_sel)
        self._maxes.append(maxes_sel)
        self._p99.append(p99_sel)
        self._publish(
            epoch,
            {
                "means": means_sel,
                "stds": stds_sel,
                "maxes": maxes_sel,
                "p99": p99_sel,
                "indices": np.array(idx, dtype=int),
            },
        )

        if get_wandb_run() is not None:
            wb: dict = {}
            cn = getattr(self, "_constraint_names", [])
            # Use the user name when available; fall back to solver-side index
            # so the dashboard remains stable when unrelated constraints are added.
            for slot, i in enumerate(idx):
                label = cn[i] if i < len(cn) and cn[i] else f"constraint_{i}"
                wb[f"explainability/residual/{label}/mean"] = float(means_sel[slot])
                wb[f"explainability/residual/{label}/std"] = float(stds_sel[slot])
                wb[f"explainability/residual/{label}/max"] = float(maxes_sel[slot])
                wb[f"explainability/residual/{label}/p99"] = float(p99_sel[slot])
            try:
                import wandb as _wandb

                for slot, i in enumerate(idx):
                    label = cn[i] if i < len(cn) and cn[i] else f"constraint_{i}"
                    wb[f"explainability/residual/{label}/histogram"] = _wandb.Histogram(raw_sel[slot])
            except ImportError:
                pass
            wandb_log(wb, step=epoch)
        return False

    @property
    def result(self) -> dict:
        """Collected data as numpy arrays.

        Keys:
            ``epochs`` — ``(S,)`` int array of sampled outer steps.
            ``means``  — ``(S, K)`` float32 per-constraint residual mean,
                where ``K = len(constraints)`` when a subset was selected,
                else ``N_constraints``.
            ``stds``   — ``(S, K)`` float32 per-constraint residual std.
            ``maxes``  — ``(S, K)`` float32 per-constraint residual max.
            ``p99``    — ``(S, K)`` float32 per-constraint 99th-percentile.
            ``indices`` — ``(K,)`` int array of solver-side constraint indices
                in the same order as the columns of ``means``/``stds``/etc.
        """
        empty = np.zeros((0,))
        return {
            "epochs": np.array(self._epochs),
            "means": np.stack(self._means) if self._means else empty,
            "stds": np.stack(self._stds) if self._stds else empty,
            "maxes": np.stack(self._maxes) if self._maxes else empty,
            "p99": np.stack(self._p99) if self._p99 else empty,
            "indices": np.array(self._indices, dtype=int),
        }


# ---------------------------------------------------------------------------
# Input sensitivity / saliency
# ---------------------------------------------------------------------------


class InputSensitivityCallback(Callback, _LiveValue):
    """Evaluate any placeholder expression at training collocation points.

    Most useful for input-saliency: pass ``u.d(x)`` (single coordinate) or
    ``jno.Jacobian(u, [x, y])`` (multi-coordinate) and the callback will
    record the per-point sensitivity of the network output to its inputs
    every ``interval`` outer steps.  Conceptually equivalent to the
    class-saliency map of Sec. 3 of [1312.6034] — high magnitude points
    correspond to spatial regions where the network response is most
    responsive to a perturbation of the input coordinate.

    The expression is compiled via
    :meth:`~jno.trace_compiler.TraceCompiler.compile_multi_expression`
    (the same machinery used by the solver's constraints and trackers),
    so any composite placeholder expression works — not just first
    derivatives.

    Args:
        expr: A :class:`~jno.trace.Placeholder` expression to evaluate.
            Common choices: ``u.d(x)``, ``jno.Jacobian(u, [x, y])``,
            ``jno.np.linalg.norm(u.d(x))``.
        interval: Compute every *n* outer training steps. Default ``100``.

    Example::

        cb = jno.callbacks.input_sensitivity(u.d(x), interval=100)
        crux.solve(5000, callbacks=[cb])
        print(cb.result["values"].shape)   # (n_samples, *expr_shape)
    """

    def __init__(self, expr, interval: int = 100) -> None:
        from jno.trace import Placeholder

        if not isinstance(expr, Placeholder):
            raise TypeError(
                f"input_sensitivity expects a jno Placeholder expression (e.g. u.d(x)), got {type(expr).__name__!r}."
            )
        self._expr = expr
        self._interval = interval
        self._fn = None
        self._epochs: list = []
        self._values: list = []
        self._init_live_value()

    def on_solve_begin(self, **kwargs) -> None:
        from jno.utils.explainability import make_expression_eval_fn

        if "all_ops" not in kwargs:
            raise RuntimeError(
                "InputSensitivityCallback requires the solver to pass `all_ops` "
                "via on_solve_begin (added by jno >= explainability v2)."
            )
        self._fn = jax.jit(
            make_expression_eval_fn(
                self._expr,
                kwargs["all_ops"],
                kwargs["batchsize"],
                kwargs["frozen"],
                kwargs["static"],
                min_consecutive=kwargs.get("min_consecutive", 1),
            )
        )
        self._fn.lower(kwargs["trainable"], kwargs["context"], kwargs["rng"]).compile()

    def on_epoch_end(self, **kwargs) -> bool:
        if self._fn is None:
            return False
        epoch = kwargs["epoch"]
        if epoch % self._interval != 0:
            return False
        values = np.asarray(jax.device_get(self._fn(kwargs["trainable"], kwargs["context"], kwargs["rng"])))
        self._epochs.append(epoch)
        self._values.append(values)
        self._publish(epoch, {"values": values})
        if get_wandb_run() is not None:
            abs_v = np.abs(values)
            wb: dict = {
                "explainability/saliency/mean_abs": float(abs_v.mean()),
                "explainability/saliency/max_abs": float(abs_v.max()),
                "explainability/saliency/std_abs": float(abs_v.std()),
            }
            try:
                import wandb as _wandb

                wb["explainability/saliency/histogram"] = _wandb.Histogram(abs_v.ravel())
            except ImportError:
                pass
            wandb_log(wb, step=epoch)
        return False

    @property
    def result(self) -> dict:
        """Collected data as numpy arrays.

        Keys:
            ``epochs`` — ``(S,)`` int array of sampled outer steps.
            ``values`` — ``(S, *expr_shape)`` array of expression evaluations.
        """
        return {
            "epochs": np.array(self._epochs),
            "values": np.stack(self._values) if self._values else np.zeros((0,)),
        }


# ---------------------------------------------------------------------------
# Empirical NTK spectrum
# ---------------------------------------------------------------------------


class NTKSpectrumCallback(Callback, _LiveValue):
    """Track the empirical Neural Tangent Kernel eigenvalue spectrum.

    For a :class:`~jno.trace.NetworkGradient` placeholder ``u.grad(net)``
    (per-point parameter Jacobian :math:`J \\in \\mathbb{R}^{N \\times P}`)
    constructs the empirical NTK :math:`K = J J^\\top` at every
    ``interval`` outer steps, subsampled to ``n_points`` collocation
    points, and reports the top-k eigenvalues, condition number, and full
    spectrum.

    The eigenvalue spread is the canonical diagnostic for PINN spectral
    bias (Sec. 3-4, [2007.14527]): widely separated eigenvalues mean some
    directions in parameter space train orders of magnitude faster than
    others, producing the characteristic PINN failure mode where
    high-frequency features lag low-frequency ones.

    Args:
        grad_expr: A :class:`~jno.trace.NetworkGradient` placeholder
            (the result of ``expr.grad(model)``).  Use ``model.mask(...)``
            to restrict to a parameter subset, e.g.
            ``u.grad(net.mask(output_only_mask))`` — masking lives in the
            placeholder rather than as a separate argument.
        n_points: Number of collocation points to subsample for the
            kernel.  Cost is ``O(n_points² × P)``; default ``256``.
        top_k: Number of largest eigenvalues to report.  Default ``10``.
        interval: Compute every *n* outer training steps.  Default ``500``
            (expensive — keep large for real runs).

    Example::

        cb = jno.callbacks.ntk_spectrum(u.grad(u_net), n_points=128, top_k=10)
        crux.solve(10_000, callbacks=[cb])
        print(cb.result["lambda_max"])         # (n_samples,)
        print(cb.result["condition_number"])   # (n_samples,)
    """

    def __init__(
        self,
        grad_expr,
        n_points: int = 256,
        top_k: int = 10,
        interval: int = 500,
    ) -> None:
        from jno.trace import NetworkGradient

        if not isinstance(grad_expr, NetworkGradient):
            raise TypeError(
                f"ntk_spectrum expects a NetworkGradient placeholder "
                f"(e.g. u.grad(net)), got {type(grad_expr).__name__!r}. "
                "Build one via ``expr.grad(model)``."
            )
        self._grad_expr = grad_expr
        self._n_points = n_points
        self._top_k = top_k
        self._interval = interval
        self._fn = None
        self._epochs: list = []
        self._top: list = []
        self._lam_min: list = []
        self._lam_max: list = []
        self._cond: list = []
        self._all: list = []
        self._init_live_value()

    def on_solve_begin(self, **kwargs) -> None:
        from jno.utils.explainability import make_ntk_spectrum_fn

        if "all_ops" not in kwargs:
            raise RuntimeError(
                "NTKSpectrumCallback requires the solver to pass `all_ops` "
                "via on_solve_begin (added by jno >= explainability v2)."
            )
        self._fn = jax.jit(
            make_ntk_spectrum_fn(
                self._grad_expr,
                kwargs["all_ops"],
                kwargs["batchsize"],
                kwargs["frozen"],
                kwargs["static"],
                n_points=self._n_points,
                top_k=self._top_k,
                min_consecutive=kwargs.get("min_consecutive", 1),
            )
        )
        self._fn.lower(kwargs["trainable"], kwargs["context"], kwargs["rng"]).compile()

    def on_epoch_end(self, **kwargs) -> bool:
        if self._fn is None:
            return False
        epoch = kwargs["epoch"]
        if epoch % self._interval != 0:
            return False
        top, lam_min, lam_max, cond, all_eigs = self._fn(kwargs["trainable"], kwargs["context"], kwargs["rng"])
        top_np = np.asarray(jax.device_get(top))
        lam_min_v = float(jax.device_get(lam_min))
        lam_max_v = float(jax.device_get(lam_max))
        cond_v = float(jax.device_get(cond))
        all_np = np.asarray(jax.device_get(all_eigs))

        self._epochs.append(epoch)
        self._top.append(top_np)
        self._lam_min.append(lam_min_v)
        self._lam_max.append(lam_max_v)
        self._cond.append(cond_v)
        self._all.append(all_np)
        self._publish(
            epoch,
            {
                "eigvals_topk": top_np,
                "lambda_min": lam_min_v,
                "lambda_max": lam_max_v,
                "condition_number": cond_v,
                "all_eigvals": all_np,
                "trace": float(all_np.sum()),
            },
        )

        if get_wandb_run() is not None:
            wb: dict = {f"explainability/ntk/eigval_{i}": float(v) for i, v in enumerate(top_np)}
            wb["explainability/ntk/lambda_max"] = lam_max_v
            wb["explainability/ntk/lambda_min"] = lam_min_v
            wb["explainability/ntk/condition_number"] = cond_v
            try:
                import wandb as _wandb

                wb["explainability/ntk/spectrum_hist"] = _wandb.Histogram(all_np)
            except ImportError:
                pass
            wandb_log(wb, step=epoch)
        return False

    @property
    def result(self) -> dict:
        """Collected data as numpy arrays."""
        empty = np.zeros((0,))
        return {
            "epochs": np.array(self._epochs),
            "eigvals_topk": np.stack(self._top) if self._top else empty,
            "lambda_min": np.array(self._lam_min),
            "lambda_max": np.array(self._lam_max),
            "condition_number": np.array(self._cond),
            "all_eigvals": np.stack(self._all) if self._all else empty,
        }


# ---------------------------------------------------------------------------
# Hessian eigenspectrum / sharpness
# ---------------------------------------------------------------------------


class HessianSpectrumCallback(Callback, _LiveValue):
    """Track the top-k Hessian eigenvalues of the total training loss.

    Constructs :math:`\\nabla^2_\\theta L` implicitly via Hessian-vector
    products (``jvp(grad(L), …, …)``) and runs ``n_iter`` Lanczos
    iterations with full reorthogonalisation.  The resulting tridiagonal
    is eigendecomposed host-side via ``scipy.linalg.eigh_tridiagonal``
    (Sec. 3.1-3.2, [1912.07145]) to obtain the top-k eigenvalues.

    The largest eigenvalue is the *sharpness* of the loss surface at the
    current iterate (Sec. 2.2, [1609.04836]); high values predict a sharp
    minimum, typically associated with worse generalisation.

    Args:
        k: Number of largest eigenvalues to report.  Default ``10``.
        n_iter: Number of Lanczos iterations.  Default ``30``.
        interval: Compute every *n* outer training steps.  Default ``500``.
        mask: Optional pytree of booleans matching the *trainable*
            structure.  Essential for large models.
        constraints: Optional list of constraint placeholders to scope the
            Hessian to the mean of those constraints' losses (instead of
            the total training loss).  Pass the same Python objects given
            to ``jno.core([...])`` — assign your constraints to variables
            rather than re-accessing ``.mse``.
    """

    def __init__(
        self,
        k: int = 10,
        n_iter: int = 30,
        interval: int = 500,
        mask=None,
        constraints=None,
    ) -> None:
        self._k = k
        self._n_iter = n_iter
        self._interval = interval
        self._mask = mask
        self._user_constraints = constraints
        self._indices: list[int] = []
        self._fn = None
        self._epochs: list = []
        self._eigvals: list = []
        self._sharpness: list = []
        self._init_live_value()

    def on_solve_begin(self, **kwargs) -> None:
        from jno.utils.explainability import make_hessian_spectrum_fn

        solver_constraints = kwargs.get("constraint_exprs")
        if solver_constraints is None:
            solver_constraints = [None] * kwargs["n_constraints"]
        self._indices = _resolve_constraint_indices(self._user_constraints, solver_constraints, "hessian_spectrum")

        constraint_indices = tuple(self._indices) if self._user_constraints is not None else None

        self._fn = make_hessian_spectrum_fn(
            kwargs["compiled_constraints_fn"],
            kwargs["batchsize"],
            kwargs["frozen"],
            kwargs["static"],
            param_mask=self._mask,
            min_consecutive=kwargs.get("min_consecutive", 1),
            k=self._k,
            n_iter=self._n_iter,
            constraint_indices=constraint_indices,
        )
        # Pre-warm — Lanczos is a Python driver, so it JIT-compiles the inner
        # HVP lazily on first call.
        self._fn(kwargs["trainable"], kwargs["context"], kwargs["rng"])

    def on_epoch_end(self, **kwargs) -> bool:
        if self._fn is None:
            return False
        epoch = kwargs["epoch"]
        if epoch % self._interval != 0:
            return False
        top, lambda_max, _all = self._fn(kwargs["trainable"], kwargs["context"], kwargs["rng"])
        top_np = np.asarray(top)
        sharp_v = float(lambda_max)
        self._epochs.append(epoch)
        self._eigvals.append(top_np)
        self._sharpness.append(sharp_v)
        self._publish(epoch, {"eigvals": top_np, "sharpness": sharp_v})

        if get_wandb_run() is not None:
            wb: dict = {f"explainability/hessian/eigval_{i}": float(v) for i, v in enumerate(top_np)}
            wb["explainability/hessian/sharpness"] = float(lambda_max)
            wb["explainability/hessian/n_iter"] = int(self._n_iter)
            wandb_log(wb, step=epoch)
        return False

    @property
    def result(self) -> dict:
        """Collected data as numpy arrays."""
        empty = np.zeros((0,))
        return {
            "epochs": np.array(self._epochs),
            "eigvals": np.stack(self._eigvals) if self._eigvals else empty,
            "sharpness": np.array(self._sharpness),
        }


# ---------------------------------------------------------------------------
# Energy Natural Gradient Descent (ENGD) callback
# ---------------------------------------------------------------------------


class ENGDCallback(Callback):
    """Precondition parameter gradients with the energy Gram matrix (ENGD).

    At every training step (or every ``gram_interval`` steps), assembles the
    energy Gram matrix

    .. math::

        G = \\sum_k \\frac{w_k}{N_k} J_k^\\top J_k \\in \\mathbb{R}^{P \\times P}

    where :math:`J_k \\in \\mathbb{R}^{N_k \\times P}` is the per-point
    parameter Jacobian of the :math:`k`-th residual expression and :math:`w_k`
    is its weight.  The natural gradient direction

    .. math::

        \\mathbf{d} = G^{-1} \\nabla_\\theta L

    replaces the standard gradient before the attached optimizer applies its
    update.  With a learning rate of 1 and a quadratic loss this is an exact
    Newton step; empirically it achieves several orders of magnitude lower
    error than first-order methods in far fewer iterations
    (Zeinhofer et al., ICML 2023, Sec. 3, arXiv:2302.13163).

    **Residual vs loss Jacobian.** Pass *raw residual* expressions — not
    ``.mse``-wrapped ones.  For a PDE ``r = u.d(x, x) + u.d(y, y) + f``,
    use ``r.grad(net)``, which gives the :math:`(N \\times P)` Jacobian of
    the per-point residual.  ``r.mse.grad(net)`` gives the scalar loss
    gradient — a very different object.

    **float64.** ENGD's accuracy benefit (reaching ~1e-7 vs ~1e-3 for Adam)
    requires float64. Enable it with
    ``jax.config.update("jax_enable_x64", True)`` before training.

    **Constraints.** Requires ``inner_steps=1`` (the hook fires between grad
    and optimizer, not inside the XLA loop).  Not compatible with
    ``.bayesian()`` models.

    **Line search.** Setting ``line_search=True`` enables a grid line search
    over 31 step sizes :math:`\\alpha \\in \\{0.5^0, 0.5^1, \\ldots, 0.5^{30}\\}`
    (Sec. 4.1, arXiv:2302.13163) to find the optimal per-step learning rate.
    This is essential for convergence when the Gram matrix is ill-conditioned
    (e.g. near initialisation), and allows reaching the paper's headline
    accuracy of ~2.4e-7.  Use ``optax.sgd(1.0)`` as the optimizer — the
    line search handles the step-size selection.  Without ``line_search=True``
    a fixed learning rate must be tuned manually.

    Args:
        gram_terms: List of ``(NetworkGradient_expr, weight)`` pairs.  Each
            expression must be a :class:`~jno.trace.NetworkGradient` — the
            result of ``residual.grad(model)`` — and all terms must reference
            the *same* model.  The weight mirrors the corresponding loss
            weight for correct metric scaling.
        gram_interval: Recompute the Gram matrix every *n* outer steps.
            Setting ``gram_interval > 1`` caches :math:`G` between
            recomputations and only re-solves with the new gradient — cheap
            but approximate when parameters move significantly.  Default
            ``1`` (full recomputation every step).
        rcond: Relative condition-number cutoff passed to
            ``jnp.linalg.lstsq``.  ``None`` (default) uses machine epsilon
            — correct for float64.  Increase to regularise a near-singular G.
        line_search: If ``True``, perform a 31-point grid line search
            :math:`\\alpha \\in \\{0.5^k : k=0,\\ldots,30\\}` each step to
            select the optimal step size.  The callback then returns
            :math:`\\alpha^* \\cdot G^{-1} \\nabla L` as the effective
            gradient so that an ``optax.sgd(1.0)`` optimizer applies the
            correct step.  Default ``False``.

    Example::

        pde = u.d(x, x) + u.d(y, y) + f          # raw residual
        bc  = u_bc                                  # raw residual

        engd = jno.callbacks.engd(
            gram_terms=[(pde.grad(net), 1.0), (bc.grad(net), 1.0)],
            line_search=True,                       # grid line search (paper §4.1)
        )
        net.optimizer(optax.sgd(1.0))              # lr=1.0; line search scales step
        crux.solve(500, callbacks=[engd])
    """

    def __init__(
        self,
        gram_terms: list,
        gram_interval: int = 1,
        rcond: Optional[float] = None,
        line_search: bool = False,
    ) -> None:
        from jno.trace import NetworkGradient

        if not gram_terms:
            raise ValueError("ENGDCallback: gram_terms must not be empty.")
        for i, (expr, w) in enumerate(gram_terms):
            if not isinstance(expr, NetworkGradient):
                raise TypeError(
                    f"ENGDCallback: gram_terms[{i}][0] must be a NetworkGradient "
                    f"placeholder (e.g. residual.grad(model)), got "
                    f"{type(expr).__name__!r}.  Build one via ``expr.grad(model)``."
                )

        # All terms must reference the same model.
        lids = [expr.model_node.layer_id for expr, _ in gram_terms]
        if len(set(lids)) != 1:
            raise ValueError(
                "ENGDCallback: all gram_terms must reference the same model "
                f"(found layer_ids {lids}).  Use a separate ENGDCallback per model."
            )

        self._gram_terms = list(gram_terms)
        self._gram_interval = gram_interval
        self._rcond = rcond
        self._line_search = line_search
        self._lid: int = lids[0]
        self._gram_and_solve_jit = None
        self._cached_solve_jit = None
        self._ls_jit = None
        self._unravel = None
        self._G_cache = None

    def on_solve_begin(self, **kwargs) -> None:
        import equinox as eqx
        import paramax as _paramax

        from jno.utils.explainability import make_engd_fn

        if "all_ops" not in kwargs:
            raise RuntimeError(
                "ENGDCallback requires the solver to pass `all_ops` via on_solve_begin (jno >= feat/engd-callback)."
            )

        trainable = kwargs["trainable"]
        if self._lid not in trainable:
            raise ValueError(
                f"ENGDCallback: model with layer_id={self._lid} not found in "
                f"trainable (keys: {list(trainable.keys())}).  Ensure the model "
                "referenced in gram_terms is a trainable model in this solver."
            )

        flat_template, self._unravel = jax.flatten_util.ravel_pytree(trainable[self._lid])
        P = flat_template.shape[0]

        gram_and_solve_fn, cached_solve_fn = make_engd_fn(
            gram_terms=self._gram_terms,
            all_ops=kwargs["all_ops"],
            batchsize=None,  # always use all context points for the Gram
            frozen=kwargs["frozen"],
            static=kwargs["static"],
            lid=self._lid,
            trainable_template=trainable,
            rcond=self._rcond,
            min_consecutive=kwargs.get("min_consecutive", 1),
        )

        self._gram_and_solve_jit = jax.jit(gram_and_solve_fn)
        self._cached_solve_jit = jax.jit(cached_solve_fn)

        # Warm up: pre-compile both JIT functions.
        flat_g = jax.numpy.zeros(P, dtype=flat_template.dtype)
        nat, G = self._gram_and_solve_jit(trainable, kwargs["context"], kwargs["rng"], flat_g)
        jax.block_until_ready((nat, G))
        _ = self._cached_solve_jit(G, flat_g)
        jax.block_until_ready(_)

        if self._line_search:
            # Grid line search (Sec. 4.1, arXiv:2302.13163):
            # evaluate total loss at α ∈ {0.5^0, …, 0.5^30} and pick the minimum.
            # Closed-over constants (captured once at solve-begin):
            _compiled_fn = kwargs["compiled_constraints_fn"]
            _frozen = kwargs["frozen"]
            _static = kwargs["static"]
            _batchsize = kwargs.get("batchsize")
            _min_consec = kwargs.get("min_consecutive", 1)
            _unravel = self._unravel
            _lid = self._lid

            def _ls_fn(trainable_inner, nat_flat_inner, context_inner, rng_inner):
                flat_p, _ = jax.flatten_util.ravel_pytree(trainable_inner[_lid])

                def _loss_at_alpha(alpha):
                    new_p = flat_p - alpha * nat_flat_inner
                    new_tr = {**trainable_inner, _lid: _unravel(new_p)}
                    full = eqx.combine(new_tr, _frozen, _static)
                    full = _paramax.unwrap(full)
                    residuals = _compiled_fn(
                        full,
                        context_inner,
                        batchsize=_batchsize,
                        key=rng_inner,
                        min_consecutive=_min_consec,
                    )
                    return jax.numpy.mean(jax.numpy.stack([jax.numpy.mean(r) for r in residuals]))

                steps = 0.5 ** jax.numpy.arange(31, dtype=flat_p.dtype)
                _, losses = jax.lax.scan(lambda c, a: (c, _loss_at_alpha(a)), None, steps)
                return steps[jax.numpy.argmin(losses)]

            self._ls_jit = jax.jit(_ls_fn)
            # Warm up.
            _ = self._ls_jit(trainable, nat, kwargs["context"], kwargs["rng"])
            jax.block_until_ready(_)

    def on_before_update(self, *, grads, trainable, context, rng, epoch, **kwargs):
        if self._gram_and_solve_jit is None or self._unravel is None:
            return None

        flat_g, _ = jax.flatten_util.ravel_pytree(grads[self._lid])

        if self._G_cache is None or epoch % self._gram_interval == 0:
            nat_flat, self._G_cache = self._gram_and_solve_jit(trainable, context, rng, flat_g)
        else:
            nat_flat = self._cached_solve_jit(self._G_cache, flat_g)

        if self._ls_jit is not None:
            best_alpha = self._ls_jit(trainable, nat_flat, context, rng)
            nat_flat = nat_flat * best_alpha

        return {**grads, self._lid: self._unravel(nat_flat)}


class callbacks:
    """Factory helpers for built-in adaptive training callbacks.

    The methods below mirror callback constructor signatures so users get
    accurate autocomplete, type hints, and inline documentation.
    """

    @staticmethod
    def checkpoint(
        directory: Optional[str] = None,
        save_interval_epochs: int = 500,
        max_to_keep: int = 3,
        best_fn: Optional[Any] = None,
        async_checkpointing: bool = True,
    ) -> CheckpointCallback:
        """Create a :class:`CheckpointCallback`.

        Args:
            directory: Root directory used to store checkpoints.
                When ``None``, defaults to
                ``<jno.setup() run dir>/checkpoints``.
            save_interval_epochs: Save frequency in outer training epochs.
            max_to_keep: Maximum number of checkpoints to retain on disk.
            best_fn: Optional metric selector ``(metrics: dict) -> float``
                used by Orbax to keep the best checkpoint.
            async_checkpointing: If ``True``, write checkpoints
                asynchronously.
        """
        return CheckpointCallback(
            directory=directory,
            save_interval_epochs=save_interval_epochs,
            max_to_keep=max_to_keep,
            best_fn=best_fn,
            async_checkpointing=async_checkpointing,
        )

    @staticmethod
    def early_stopping(
        patience: int = 500,
        min_delta: float = 0.0,
        mode: str = "min",
        metric_fn: Optional[Any] = None,
        baseline: Optional[float] = None,
        verbose: bool = True,
    ) -> EarlyStoppingCallback:
        """Create an :class:`EarlyStoppingCallback`.

        Args:
            patience: Number of non-improving epochs allowed before stopping.
            min_delta: Improvement threshold (absolute for ``min``/``max``,
                relative fraction for ``rel`` mode).
            mode: Improvement mode, one of ``"min"``, ``"max"``, or
                ``"rel"``.
            metric_fn: Optional callable extracting a scalar metric from
                ``on_epoch_end`` keyword arguments.
            baseline: Optional initial best metric value.
            verbose: If ``True``, emit a stop message to the logger.
        """
        return EarlyStoppingCallback(
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            metric_fn=metric_fn,
            baseline=baseline,
            verbose=verbose,
        )

    @staticmethod
    def gradient_norms(interval: int = 100, mask=None) -> GradientNormsCallback:
        """Create a :class:`GradientNormsCallback`.

        Tracks ``‖∇L_i‖₂`` for each loss term every *interval* outer steps.

        Args:
            interval: Compute every *n* outer training steps.
            mask: Optional pytree of booleans matching ``crux.models``
                structure.  Restricts gradient computation to the selected
                parameter subset — recommended for large models to reduce cost.
        """
        return GradientNormsCallback(interval=interval, mask=mask)

    @staticmethod
    def cos_similarity(interval: int = 100, mask=None) -> CosSimilarityCallback:
        """Create a :class:`CosSimilarityCallback`.

        Tracks the ``(N × N)`` pairwise cosine similarity matrix between
        per-loss gradients every *interval* outer steps.

        Args:
            interval: Compute every *n* outer training steps.
            mask: Optional pytree of booleans — see :func:`gradient_norms`.
        """
        return CosSimilarityCallback(interval=interval, mask=mask)

    @staticmethod
    def gradient_alignment(interval: int = 100, mask=None) -> GradientAlignmentCallback:
        """Create a :class:`GradientAlignmentCallback`.

        Tracks the total gradient alignment scalar
        ``2‖(1/N) Σ ĝᵢ‖² − 1`` (Eq. 3.1, [2502.00604]) every *interval*
        outer steps.  Range ``[-1, 1]``.

        Args:
            interval: Compute every *n* outer training steps.
            mask: Optional pytree of booleans — see :func:`gradient_norms`.
        """
        return GradientAlignmentCallback(interval=interval, mask=mask)

    @staticmethod
    def loss_landscape(
        interval: int = 500,
        mask=None,
        n_grid: int = 15,
        alpha_range: float = 1.0,
    ) -> LossLandscapeCallback:
        """Create a :class:`LossLandscapeCallback`.

        Evaluates the total loss on a 2-D perturbation grid (``n_grid²``
        forward passes) every *interval* outer steps.

        Args:
            interval: Compute every *n* outer training steps.  Use a large
                value (500–1000) since each call is expensive.
            mask: Optional pytree of booleans matching ``crux.models``
                structure.  Only selected parameters are perturbed; strongly
                recommended for large models.
            n_grid: Grid points per axis.  Total evaluations = ``n_grid²``.
            alpha_range: Perturbation range in units of ``‖θ_selected‖``.
        """
        return LossLandscapeCallback(interval=interval, mask=mask, n_grid=n_grid, alpha_range=alpha_range)

    @staticmethod
    def hessian_spectrum(
        k: int = 10,
        n_iter: int = 30,
        interval: int = 500,
        mask=None,
        constraints=None,
    ) -> HessianSpectrumCallback:
        """Create a :class:`HessianSpectrumCallback`.

        Tracks the top-k eigenvalues of the total training loss Hessian
        via Lanczos with HVPs (Sec. 3.1-3.2, [1912.07145]); the largest
        eigenvalue is the sharpness of [Keskar et al., Sec. 2.2,
        1609.04836].

        Args:
            k: Number of largest eigenvalues to report.
            n_iter: Number of Lanczos iterations.
            interval: Compute every *n* outer training steps.
            mask: Optional pytree of booleans matching ``crux.models``
                structure — essential for large models.
            constraints: Optional list of constraint placeholders to scope
                the Hessian to the mean of those constraints' losses
                instead of the full training loss.
        """
        return HessianSpectrumCallback(
            k=k,
            n_iter=n_iter,
            interval=interval,
            mask=mask,
            constraints=constraints,
        )

    @staticmethod
    def ntk_spectrum(
        grad_expr,
        n_points: int = 256,
        top_k: int = 10,
        interval: int = 500,
    ) -> NTKSpectrumCallback:
        """Create an :class:`NTKSpectrumCallback`.

        Tracks the empirical NTK eigenvalue spectrum (Sec. 3-4, [2007.14527])
        for a user-supplied :class:`~jno.trace.NetworkGradient` placeholder,
        e.g. ``u.grad(net)``.  Restrict to a parameter subset by chaining
        ``net.mask(mask)`` into the placeholder.

        Args:
            grad_expr: ``NetworkGradient`` placeholder (typically
                ``u.grad(net)`` or ``u.grad(net.mask(mask))``).
            n_points: Subsample cap for kernel rows.  Cost is
                ``O(n_points² × P)``.
            top_k: Number of largest eigenvalues to report.
            interval: Compute every *n* outer training steps.
        """
        return NTKSpectrumCallback(
            grad_expr=grad_expr,
            n_points=n_points,
            top_k=top_k,
            interval=interval,
        )

    @staticmethod
    def input_sensitivity(expr, interval: int = 100) -> InputSensitivityCallback:
        """Create an :class:`InputSensitivityCallback`.

        Evaluates a user-supplied placeholder expression at training
        collocation points every *interval* outer steps (Sec. 3, [1312.6034]).

        Args:
            expr: A :class:`~jno.trace.Placeholder` expression — commonly
                ``u.d(x)`` for input-gradient saliency, or
                ``jno.Jacobian(u, [x, y])`` for a multi-variable Jacobian.
            interval: Compute every *n* outer training steps.
        """
        return InputSensitivityCallback(expr=expr, interval=interval)

    @staticmethod
    def residual_stats(interval: int = 100, constraints=None) -> ResidualStatsCallback:
        """Create a :class:`ResidualStatsCallback`.

        Tracks per-constraint residual mean, std, max, and 99th-percentile —
        plus a histogram when W&B is active — every *interval* outer steps
        (Sec. 3, [2207.10289]).

        Args:
            interval: Compute every *n* outer training steps.
            constraints: Optional list of constraint placeholders to scope to a
                subset of the solver's constraints.  Pass the same Python
                objects given to ``jno.core([...])`` (assign your constraints
                to variables first).  Default ``None`` — all constraints.
        """
        return ResidualStatsCallback(interval=interval, constraints=constraints)

    @staticmethod
    def engd(
        gram_terms: list,
        gram_interval: int = 1,
        rcond: Optional[float] = None,
        line_search: bool = False,
    ) -> "ENGDCallback":
        """Create an :class:`ENGDCallback` for Energy Natural Gradient Descent.

        Preconditions parameter gradients with the inverse energy Gram matrix
        :math:`G^{-1}` before the optimizer update, converting standard
        gradient descent into a Newton-like method that can achieve
        orders-of-magnitude lower error in far fewer iterations than Adam or
        L-BFGS (Zeinhofer et al., ICML 2023, Sec. 3, arXiv:2302.13163).

        Requires ``inner_steps=1``, no Bayesian models, and **float64**
        (``jax.config.update("jax_enable_x64", True)``) for full accuracy.
        Use ``model.optimizer(optax.sgd(1.0))`` — the natural gradient
        direction already encodes the correct step scale.

        Args:
            gram_terms: List of ``(NetworkGradient_expr, weight)`` pairs.
                Build a :class:`~jno.trace.NetworkGradient` via
                ``residual.grad(model)`` on the *raw* residual expression
                (not ``.mse.grad``).  All terms must reference the same model.
            gram_interval: Recompute G every *n* outer steps; cache between
                recomputations.  Default ``1`` (recompute every step).
            rcond: Condition-number cutoff for ``jnp.linalg.lstsq``.
                ``None`` (default) → machine epsilon (best for float64).
            line_search: If ``True``, perform a 31-point grid line search
                :math:`\\alpha \\in \\{0.5^k : k=0,\\ldots,30\\}` each step
                (Sec. 4.1, arXiv:2302.13163).  Recommended for faithful
                reproduction of the paper's results.  Use with
                ``optax.sgd(1.0)``; the selected :math:`\\alpha^*` is
                folded into the returned gradient so the optimizer applies
                the full scaled natural gradient step.  Default ``False``.
        """
        return ENGDCallback(
            gram_terms=gram_terms,
            gram_interval=gram_interval,
            rcond=rcond,
            line_search=line_search,
        )
