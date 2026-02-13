"""Callback classes for monitoring training."""

from typing import Dict, Any
import jax.numpy as jnp
from ..core import core


class Callback:
    """Base callback class."""

    def on_epoch_end(self, state: core, *args, **kwargs):
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of logs (e.g., loss, metrics)
        """
        pass

    def on_training_end(self, state: core, *args, **kwargs):
        """Called at the end of training.

        Args:
            logs: Final training logs
        """
        pass
