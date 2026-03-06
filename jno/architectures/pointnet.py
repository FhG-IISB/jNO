from typing import Callable, Optional, List
import jax
import jax.numpy as jnp
import equinox as eqx


class PointNet(eqx.Module):
    output_dim: int
    hidden_dims: list
    dropout_rate: float
    act: Callable = eqx.field(static=True)
    use_bias: bool

    # Encoder/decoder split index
    bottleneck_idx: int = eqx.field(static=True)
    skip_idx: int = eqx.field(static=True)

    enc_convs: list
    dec_convs: list
    output_conv: eqx.nn.Conv1d

    def __init__(
        self,
        in_features: int,
        output_dim: int,
        hidden_dims: list = None,
        bottleneck_idx: int = None,
        skip_idx: int = 1,
        dropout_rate: float = 0.0,
        act: Callable = jax.nn.tanh,
        use_bias: bool = True,
        *,
        key: jax.Array,
        **kwargs,
    ):
        """
        PointNet architecture with configurable depth.

        Args:
            in_features: Number of input features per point (C)
            output_dim: Number of output features per point
            hidden_dims: List of hidden dimensions for all conv layers
            bottleneck_idx: Index in hidden_dims where encoder ends and decoder begins.
                           If None, defaults to len(hidden_dims) // 2
            skip_idx: Index in encoder to take skip connection from (default: 1)
            dropout_rate: Dropout probability
            act: Activation function
            use_bias: Whether to use bias in conv layers
            key: PRNG key
        """
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 256, 128, 64]

        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.act = act
        self.use_bias = use_bias

        # Determine encoder/decoder split
        if bottleneck_idx is None:
            bottleneck_idx = len(hidden_dims) // 2
        self.bottleneck_idx = bottleneck_idx
        self.skip_idx = skip_idx

        enc_dims = hidden_dims[: bottleneck_idx + 1]
        dec_dims = hidden_dims[bottleneck_idx + 1 :]

        # Build encoder: in_features -> enc_dims[0] -> enc_dims[1] -> ... -> enc_dims[-1]
        enc_channels = [in_features] + [int(h) for h in enc_dims]
        enc_convs = []
        for i in range(len(enc_dims)):
            key, subkey = jax.random.split(key)
            enc_convs.append(
                eqx.nn.Conv1d(
                    enc_channels[i],
                    enc_channels[i + 1],
                    kernel_size=1,
                    use_bias=use_bias,
                    key=subkey,
                )
            )
        self.enc_convs = enc_convs

        # Build decoder: (skip_features + global_features) -> dec_dims[0] -> ... -> dec_dims[-1]
        skip_features = int(enc_dims[skip_idx])
        global_features = int(enc_dims[-1])
        dec_in = skip_features + global_features

        dec_channels = [dec_in] + [int(h) for h in dec_dims]
        dec_convs = []
        for i in range(len(dec_dims)):
            key, subkey = jax.random.split(key)
            dec_convs.append(
                eqx.nn.Conv1d(
                    dec_channels[i],
                    dec_channels[i + 1],
                    kernel_size=1,
                    use_bias=use_bias,
                    key=subkey,
                )
            )
        self.dec_convs = dec_convs

        # Output convolution
        key, subkey = jax.random.split(key)
        final_hidden = int(dec_dims[-1]) if dec_dims else dec_in
        self.output_conv = eqx.nn.Conv1d(
            final_hidden,
            output_dim,
            kernel_size=1,
            use_bias=use_bias,
            key=subkey,
        )

    def _forward_single(self, c, key=None):
        """
        Forward pass for a single sample.

        Args:
            c: Input of shape (N, C) - N points, C features
            key: Optional PRNG key for dropout

        Returns:
            Output of shape (N, output_dim)
        """
        n_points = c.shape[0]
        dropout = eqx.nn.Dropout(p=self.dropout_rate)

        def conv_block(c, conv, key):
            # c: (N, F) -> (F, N) for Conv1d
            c = c.T
            c = conv(c)
            c = c.T  # (N, H)
            c = self.act(c)

            if key is not None and self.dropout_rate > 0:
                key, subkey = jax.random.split(key)
                c = dropout(c, key=subkey)
                return c, key
            return c, key

        # Encoder
        skip_connection = None
        for i, conv in enumerate(self.enc_convs):
            c, key = conv_block(c, conv, key)
            if i == self.skip_idx:
                skip_connection = c

        # Global feature via max pooling
        global_feature = jnp.max(c, axis=0, keepdims=True)  # (1, H)
        global_feature = jnp.broadcast_to(global_feature, (n_points, global_feature.shape[-1]))

        # Concatenate skip connection with global feature
        c = jnp.concatenate([skip_connection, global_feature], axis=-1)

        # Decoder
        for conv in self.dec_convs:
            c, key = conv_block(c, conv, key)

        # Output projection (no activation)
        c = c.T  # (H, N)
        c = self.output_conv(c)  # (output_dim, N)
        c = c.T  # (N, output_dim)

        return c

    def __call__(self, c, key=None, **kwargs):
        """
        Forward pass for batched input.

        Args:
            c: Input of shape (B, N, C) - batch, points, features
            key: Optional PRNG key for dropout

        Returns:
            Output of shape (B, N, output_dim)
        """
        if c.ndim == 3:
            batch_size = c.shape[0]
            if key is not None:
                keys = jax.random.split(key, batch_size)
                return jax.vmap(self._forward_single)(c, keys)
            else:
                return jax.vmap(self._forward_single, in_axes=(0, None))(c, None)
        elif c.ndim == 2:
            return self._forward_single(c, key)
        else:
            raise ValueError(f"Expected input of shape (B, N, C) or (N, C), got {c.shape}")
