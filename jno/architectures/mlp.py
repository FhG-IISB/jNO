from typing import Callable, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx
from .linear import Linear
from .common import BatchNorm


class MLP(eqx.Module):
    """
    Multi-Layer Perceptron with configurable architecture.
    """

    hidden_layers: list
    output_layer: Linear
    norm_layers: list
    activation: Callable = eqx.field(static=True)
    output_activation: Callable = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        output_dim: int = 1,
        activation: Callable = jnp.tanh,
        hidden_dims: int | Sequence[int] = 64,
        num_layers: int = 2,
        output_activation: Callable | None = None,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
        final_layer_bias: bool = True,
        *,
        key,
        **kwargs,
    ):
        if isinstance(hidden_dims, int):
            layer_widths = [hidden_dims] * num_layers
        else:
            layer_widths = list(hidden_dims)

        # Build hidden layers
        dims = [in_features] + layer_widths
        keys = jax.random.split(key, len(dims))  # one extra for output
        self.hidden_layers = [Linear(dims[i], dims[i + 1], use_bias=use_bias, key=keys[i]) for i in range(len(layer_widths))]

        # Output layer
        self.output_layer = Linear(
            layer_widths[-1] if layer_widths else in_features,
            output_dim,
            use_bias=final_layer_bias,
            key=keys[-1],
        )

        # Normalization layers
        self.norm_layers = []
        for w in layer_widths:
            if batch_norm:
                self.norm_layers.append(BatchNorm(w))
            elif layer_norm:
                self.norm_layers.append(eqx.nn.LayerNorm(w))
            else:
                self.norm_layers.append(None)

        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate

    def __call__(
        self,
        *inputs: jnp.ndarray,
        key=None,
        **kwargs,
    ) -> jnp.ndarray:
        # Concatenate all inputs along feature axis
        if len(inputs) == 1:
            h = inputs[0]
        else:
            h = jnp.concatenate(inputs, axis=-1)

        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)

            # Normalization
            if self.norm_layers[i] is not None:
                h = self.norm_layers[i](h)

            # Activation
            h = self.activation(h)

            # Dropout
            if self.dropout_rate > 0.0 and key is not None:
                key, subkey = jax.random.split(key)
                h = eqx.nn.Dropout(p=self.dropout_rate)(h, key=subkey)

        # Output layer
        output = self.output_layer(h)

        # Optional output activation
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output
