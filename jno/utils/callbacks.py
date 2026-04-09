"""Callback classes for monitoring and checkpointing during training."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .config import wandb_alert


class Callback:
    """Base callback class.

    Subclass and override the hooks you need.  Every hook receives
    keyword arguments whose contents depend on the hook and the
    caller.  This keeps the interface decoupled from any particular
    solver implementation.
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
            raise ImportError("orbax-checkpoint is required for CheckpointCallback. " "Install it with:  pip install orbax-checkpoint") from exc

        if directory is None:
            from .logger import get_logger

            log = get_logger()
            log_path = getattr(log, "path", None)
            if log_path and str(log_path):
                directory = os.path.join(str(log_path), "checkpoints")
            else:
                raise ValueError("No directory given and no jno.setup() run directory found. " "Either pass directory= or call jno.setup(__file__) first.")

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
            self._upload_wandb_artifact(epoch)
            self._log_wandb_histograms(trainable, epoch)

    def on_training_end(self, **kwargs) -> None:
        self._manager.wait_until_finished()

    # -- wandb ---------------------------------------------------------------

    def _upload_wandb_artifact(self, epoch: int) -> None:
        """Upload the latest checkpoint directory as a W&B artifact."""
        from .config import get_wandb_run

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
            metadata={"epoch": epoch},
        )
        artifact.add_dir(ckpt_path)
        run.log_artifact(artifact)

    def _log_wandb_histograms(self, trainable: Any, epoch: int) -> None:
        """Log per-layer weight histograms to W&B."""
        from .config import get_wandb_run

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
        ocp = self._ocp
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
                msg = f"Early stopping at epoch {epoch}: " f"no improvement for {self.patience} epochs " f"(best={self.best_metric:.6e})"
                if log is not None:
                    log.info(msg)
            wandb_alert(
                "Early stopping",
                f"Stopped at epoch {epoch} — no improvement for " f"{self.patience} epochs (best={self.best_metric:.6e})",
                level="WARN",
            )
            return True  # signal stop

        return False

    @property
    def has_stopped(self) -> bool:
        """Whether early stopping was triggered."""
        return self._stopped
