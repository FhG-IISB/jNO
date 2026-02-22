from typing import Callable, Optional, List
import jax
import jax.numpy as jnp
import equinox as eqx


class PointNet(eqx.Module):
    output_dim: int
    hidden_dims: list
    fixed_size: bool
    dropout_rate: float
    feature_transform: Optional[Callable]
    act: Callable = eqx.field(static=True)
    use_bias: bool

    enc_convs: list
    dec_convs: list
    output_conv: eqx.nn.Conv1d

    def __init__(
        self,
        in_features: int,
        output_dim: int,
        hidden_dims: list = None,
        fixed_size: bool = True,
        dropout_rate: float = 0.0,
        feature_transform: Optional[Callable] = None,
        act: Callable = jax.nn.tanh,
        use_bias: bool = True,
        *,
        key: jax.random.PRNGKey,
        **kwargs,
    ):
        if hidden_dims is None:
            hidden_dims = [32, 16, 8, 4, 2, 2, 4, 8, 8]
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)
        self.fixed_size = fixed_size
        self.dropout_rate = dropout_rate
        self.feature_transform = feature_transform
        self.act = act
        self.use_bias = use_bias

        # Encoder convolutions: in -> h0 -> h1 -> h2 -> h3 -> h4
        enc_channels = [in_features] + [int(h) for h in hidden_dims[:5]]
        enc_convs = []
        for i in range(5):
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

        # Decoder convolutions: (h1 + h4) -> h5 -> h6 -> h7 -> h8
        dec_in = int(hidden_dims[1]) + int(hidden_dims[4])
        dec_channels = [dec_in] + [int(h) for h in hidden_dims[5:9]]
        dec_convs = []
        for i in range(4):
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
        self.output_conv = eqx.nn.Conv1d(
            int(hidden_dims[8]),
            output_dim,
            kernel_size=1,
            use_bias=use_bias,
            key=subkey,
        )

    def __call__(self, *inputs, key=None, **kwargs):
        if len(inputs) == 1:
            h = inputs[0]
        else:
            h = jnp.concatenate(inputs, axis=-1)
        if self.feature_transform is not None:
            c = self.feature_transform(h)
        else:
            c = h
        n_points = c.shape[0]

        dropout = eqx.nn.Dropout(p=self.dropout_rate)

        def conv_dropout(c, conv, key):
            c = jax.vmap(conv)(c[:, None, :])[:, 0, :]
            c = self.act(c)
            if key is not None:
                key, subkey = jax.random.split(key)
                c = dropout(c, key=subkey)
            return c, key

        # Encoder
        c, key = conv_dropout(c, self.enc_convs[0], key)
        c, key = conv_dropout(c, self.enc_convs[1], key)
        seg_part1 = c
        c, key = conv_dropout(c, self.enc_convs[2], key)
        c, key = conv_dropout(c, self.enc_convs[3], key)
        c, key = conv_dropout(c, self.enc_convs[4], key)

        # Global feature
        global_feature = jnp.max(c, axis=0, keepdims=True)
        global_feature = jnp.broadcast_to(global_feature, (n_points, global_feature.shape[-1]))
        c = jnp.concatenate([seg_part1, global_feature], axis=-1)

        # Decoder
        c, key = conv_dropout(c, self.dec_convs[0], key)
        c, key = conv_dropout(c, self.dec_convs[1], key)
        c, key = conv_dropout(c, self.dec_convs[2], key)
        c, key = conv_dropout(c, self.dec_convs[3], key)

        # Output
        c = jax.vmap(self.output_conv)(c[:, None, :])[:, 0, :]
        return c
