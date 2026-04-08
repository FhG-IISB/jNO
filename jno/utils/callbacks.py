"""Callback classes for monitoring and checkpointing during training."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from ..core import core


class Callback:
    """Base callback class.

    Subclass and override the hooks you need.  Every hook receives the
    ``core`` solver instance as its first positional argument, followed by
    keyword arguments whose contents depend on the hook.
    """

    def on_epoch_end(self, state: "core", **kwargs) -> None:
        """Called at the end of every outer training step.

        Keyword Args:
            epoch (int): Current epoch number (0-indexed).
            trainable: Trainable parameter pytree (Equinox partition).
            opt_states: ``dict[str, optax.OptState]`` per-model optimizer states.
            rng: Current JAX PRNG key.
            total_loss: Scalar total loss (JAX array, still on device).
            individual_losses: Per-constraint losses (JAX array).
        """

    def on_training_end(self, state: "core", **kwargs) -> None:
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

        opts_kwargs = dict(
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

    def on_epoch_end(self, state: "core", **kwargs) -> None:
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
        self._manager.save(
            epoch,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(pytree),
                metadata=ocp.args.JsonSave(metadata),
            ),
            metrics=metadata if self._best_fn is not None else None,
        )

    def on_training_end(self, state: "core", **kwargs) -> None:
        self._manager.wait_until_finished()

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
