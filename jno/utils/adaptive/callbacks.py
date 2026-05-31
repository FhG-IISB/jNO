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

        ckpt_path = os.path.join(self._directory, str(epoch))
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
# Shared base for gradient-analysis callbacks (1–3)
# ---------------------------------------------------------------------------


class _PerLossGradCallback(Callback):
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

    def on_solve_begin(self, **kwargs) -> None:
        from jno.utils.explainability import make_per_loss_grad_fn

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
            self._epochs.append(epoch)
            self._norms.append(np.asarray(norms))
            if get_wandb_run() is not None:
                wandb_log(
                    {f"explainability/gradient_norm/constraint_{i}": float(v) for i, v in enumerate(norms)},
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
            if get_wandb_run() is not None:
                N = cos_np.shape[0]
                wb: dict = {
                    f"explainability/cos_sim/{i}_{j}": float(cos_np[i, j]) for i in range(N) for j in range(i + 1, N)
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

    Computes ``‖Σgᵢ‖ / Σ‖gᵢ‖`` (Eq. 3.1, [2502.00604]), a value in
    ``[0, 1]`` that measures how well all loss gradients point in the same
    direction.  A value near 1 means perfect alignment; near 0 means
    destructive interference.

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
            self._align.append(float(alignment))
            if get_wandb_run() is not None:
                wandb_log({"explainability/gradient_alignment": float(alignment)}, step=epoch)
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


class LossLandscapeCallback(Callback):
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


class ResidualStatsCallback(Callback):
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
        solver = jno.core([pde_loss, bc_loss], domain)
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

    def on_solve_begin(self, **kwargs) -> None:
        from jno.utils.explainability import make_residual_stats_fn

        self._n_constraints = kwargs["n_constraints"]
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

        if get_wandb_run() is not None:
            wb: dict = {}
            # Use the *solver-side* index for the W&B key so the dashboard
            # remains stable when users add/remove unrelated constraints.
            for slot, i in enumerate(idx):
                wb[f"explainability/residual/constraint_{i}/mean"] = float(means_sel[slot])
                wb[f"explainability/residual/constraint_{i}/std"] = float(stds_sel[slot])
                wb[f"explainability/residual/constraint_{i}/max"] = float(maxes_sel[slot])
                wb[f"explainability/residual/constraint_{i}/p99"] = float(p99_sel[slot])
            try:
                import wandb as _wandb

                for slot, i in enumerate(idx):
                    wb[f"explainability/residual/constraint_{i}/histogram"] = _wandb.Histogram(raw_sel[slot])
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

        Tracks the total gradient alignment scalar ``‖Σgᵢ‖ / Σ‖gᵢ‖``
        (Eq. 3.1, [2502.00604]) every *interval* outer steps.

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
