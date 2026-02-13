from typing import Callable, Optional, List
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass, field


class PointNet(nn.Module):
    """
    PointNet-style CNN with 1D convolutions, sin activation, and global feature aggregation.
    """

    output_dim: int  # Output dimension (dO in original)
    hidden_dims: list = field(default_factory=[32, 16, 8, 4, 2, 2, 4, 8, 8])
    fixed_size: bool = True
    dropout_rate: float = 0.0
    feature_transform: Optional[Callable] = None  # Optional input transform
    act: Callable = nn.tanh
    use_bias: bool = True

    @nn.compact
    def __call__(self, *inputs: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape (batch, n_points, features)
            training: Whether in training mode (affects dropout)

        Returns:
            Output tensor of shape (batch, n_points, output_dim)
        """

        if len(inputs) == 1:
            h = inputs[0]
        else:
            h = jnp.concatenate(inputs, axis=-1)

        if h.ndim == 1:
            h = h[:, None]

        # Optional feature transform
        if self.feature_transform is not None:
            c = self.feature_transform(h)
        else:
            c = h
        n_points = c.shape[0]  # Get actual shape from input, not self.n_tot

        # Helper function for Conv1D + sin + Dropout block
        def conv_dropout(c, features, name):
            c = nn.Conv(features=int(features), kernel_size=(1,), use_bias=self.use_bias, name=name)(c)
            c = self.act(c)
            c = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(c)
            return c

        # Encoder path
        c = conv_dropout(c, self.hidden_dims[0], "conv1")
        c = conv_dropout(c, self.hidden_dims[1], "conv2")
        seg_part1 = c  # Save for skip connection

        c = conv_dropout(c, self.hidden_dims[2], "conv3")
        c = conv_dropout(c, self.hidden_dims[3], "conv4")
        c = conv_dropout(c, self.hidden_dims[4], "conv5")

        # Global feature extraction via max pooling
        # MaxPool over the points dimension, then tile back
        global_feature = jnp.max(c, axis=0, keepdims=True)  # (batch, 1, channels)
        global_feature = jnp.broadcast_to(global_feature, (n_points, global_feature.shape[-1]))  # (batch, n_points, channels)

        # Concatenate local and global features
        c = jnp.concatenate([seg_part1, global_feature], axis=-1)

        # Decoder path
        c = conv_dropout(c, self.hidden_dims[5], "conv6")
        c = conv_dropout(c, self.hidden_dims[6], "conv7")
        c = conv_dropout(c, self.hidden_dims[7], "conv8")
        c = conv_dropout(c, self.hidden_dims[8], "conv9")

        # Final output layer (linear activation)
        c = c[None, :, :]
        c = nn.Conv(features=self.output_dim, kernel_size=(1,), use_bias=self.use_bias, name="output_conv")(c)
        c = c[0]
        return c
