"""Batched Linear layer for equinox — handles arbitrary leading dimensions.

``eqx.nn.Linear`` computes ``weight @ x`` which only works for 1D input.
This module computes ``x @ weight.T + bias`` which broadcasts over any
leading dimensions: ``(D,)``, ``(N, D)``, ``(M, N, D)``, etc.
"""

from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx


class Linear(eqx.Module):
    """Drop-in replacement for ``eqx.nn.Linear`` that handles batched inputs.

    Maps ``(..., in_features) -> (..., out_features)`` for any number of
    leading dimensions, matching the behavior of PyTorch's ``nn.Linear``
    and Flax's ``nn.Dense``.
    """

    weight: jax.Array
    bias: Optional[jax.Array]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, use_bias: bool = True, *, key: jax.Array):
        wkey, bkey = jax.random.split(key)
        lim = 1 / jnp.sqrt(in_features)
        self.weight = jax.random.uniform(wkey, (out_features, in_features), minval=-lim, maxval=lim)
        self.bias = jax.random.uniform(bkey, (out_features,), minval=-lim, maxval=lim) if use_bias else None
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: ``x @ weight.T [+ bias]``.

        Args:
            x: Input array of shape ``(..., in_features)``.

        Returns:
            Output array of shape ``(..., out_features)``.
        """
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y
