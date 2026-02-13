from typing import Callable, Sequence
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable architecture.
    """

    output_dim: int = 1
    activation: Callable = nn.tanh
    hidden_dims: int | Sequence[int] = 64
    num_layers: int = 2
    output_activation: Callable | None = None
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    dropout_rate: float = 0.0
    batch_norm: bool = False
    layer_norm: bool = False
    final_layer_bias: bool = True

    @nn.compact
    def __call__(
        self,
        *inputs: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        # Concatenate all inputs along feature axis
        if len(inputs) == 1:
            h = inputs[0]
        else:
            h = jnp.concatenate(inputs, axis=-1)

        if h.ndim == 1:
            h = h[:, None]

        # Determine hidden layer widths
        if isinstance(self.hidden_dims, int):
            layer_widths = [self.hidden_dims] * self.num_layers
        else:
            layer_widths = list(self.hidden_dims)

        # Hidden layers
        for i, width in enumerate(layer_widths):
            h = nn.Dense(
                features=width,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"hidden_{i}",
            )(h)

            # Normalization (applied before activation)
            if self.batch_norm:
                h = nn.BatchNorm(
                    use_running_average=not training,
                    name=f"bn_{i}",
                )(h)
            elif self.layer_norm:
                h = nn.LayerNorm(name=f"ln_{i}")(h)

            # Activation
            h = self.activation(h)

            # Dropout (applied after activation)
            if self.dropout_rate > 0.0:
                h = nn.Dropout(
                    rate=self.dropout_rate,
                    deterministic=not training,
                )(h)

        # Output layer
        output = nn.Dense(
            features=self.output_dim,
            use_bias=self.final_layer_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="output",
        )(h)

        # Optional output activation
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output
