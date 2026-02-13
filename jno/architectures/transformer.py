# https://github.com/voyager-jhk/JaxTransformer
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


class PositionalEncoding(nn.Module):
    max_len: int
    embed_dim: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        seq_len = inputs.shape[1]
        position = jnp.arange(self.max_len, dtype=jnp.float32)[jnp.newaxis, :]
        div_term = jnp.exp(jnp.arange(0, self.embed_dim, 2) * -(jnp.log(10000.0) / self.embed_dim))

        pe = jnp.zeros((self.max_len, self.embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        pe = pe[jnp.newaxis, :seq_len, :]

        return inputs + jnp.array(pe, dtype=inputs.dtype)


class MLP(nn.Module):
    features: int
    out_features: int
    dropout_rate: float = 0.1
    deterministic: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray):
        x = nn.Dense(self.features, name="wi")(inputs)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(x)
        x = nn.Dense(self.out_features, name="wo")(x)
        return x


class SelfAttention(nn.Module):
    num_heads: int
    qkv_features: int
    out_features: int
    dropout_rate: float
    deterministic: bool
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, mask=None):
        query = nn.Dense(self.qkv_features, name="query_proj")(inputs_q)
        key = nn.Dense(self.qkv_features, name="key_proj")(inputs_kv)
        value = nn.Dense(self.qkv_features, name="value_proj")(inputs_kv)

        attention_output = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dtype=self.dtype, deterministic=self.deterministic, dropout_rate=self.dropout_rate, name="attention_core")(query, key, value, mask)

        output = nn.Dense(self.out_features, name="output_proj")(attention_output)
        return output


# --- 2. Encoder & Decoder Blocks ---


class EncoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float = 0.1
    deterministic: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray = None):
        norm_inputs = nn.LayerNorm(name="norm1")(inputs)
        attention_output = SelfAttention(num_heads=self.num_heads, qkv_features=self.qkv_features, out_features=self.embed_dim, dropout_rate=self.dropout_rate, deterministic=self.deterministic, dtype=self.dtype, name="self_attention")(
            norm_inputs, norm_inputs, mask
        )
        x = inputs + attention_output

        norm_x = nn.LayerNorm(name="norm2")(x)
        mlp_output = MLP(features=self.mlp_features, out_features=self.embed_dim, dropout_rate=self.dropout_rate, deterministic=self.deterministic, name="mlp")(norm_x)
        return x + mlp_output


class DecoderBlock(nn.Module):
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float = 0.1
    deterministic: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, encoder_outputs: jnp.ndarray, self_attention_mask: jnp.ndarray = None, cross_attention_mask: jnp.ndarray = None):
        norm_inputs = nn.LayerNorm(name="norm1")(inputs)
        self_attention_output = SelfAttention(num_heads=self.num_heads, qkv_features=self.qkv_features, out_features=self.embed_dim, dropout_rate=self.dropout_rate, deterministic=self.deterministic, dtype=self.dtype, name="self_attention")(
            norm_inputs, norm_inputs, self_attention_mask
        )
        x = inputs + self_attention_output

        norm_x_cross = nn.LayerNorm(name="norm2")(x)
        cross_attention_output = SelfAttention(num_heads=self.num_heads, qkv_features=self.qkv_features, out_features=self.embed_dim, dropout_rate=self.dropout_rate, deterministic=self.deterministic, dtype=self.dtype, name="cross_attention")(
            norm_x_cross, encoder_outputs, cross_attention_mask
        )
        x = x + cross_attention_output

        norm_x_mlp = nn.LayerNorm(name="norm3")(x)
        mlp_output = MLP(features=self.mlp_features, out_features=self.embed_dim, dropout_rate=self.dropout_rate, deterministic=self.deterministic, name="mlp")(norm_x_mlp)
        return x + mlp_output


# --- 3. Stacks & Full Model ---


class TransformerEncoder(nn.Module):
    num_layers: int
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float
    deterministic: bool
    vocab_size: int
    max_len: int
    dtype: Any

    @nn.compact
    def __call__(self, input_tokens, attention_mask=None):
        x = nn.Embed(self.vocab_size, self.embed_dim, name="token_embeddings")(input_tokens)
        x = PositionalEncoding(self.max_len, self.embed_dim, name="pos_enc")(x)
        x = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(x)
        for i in range(self.num_layers):
            x = EncoderBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads, qkv_features=self.qkv_features, mlp_features=self.mlp_features, dropout_rate=self.dropout_rate, deterministic=self.deterministic, dtype=self.dtype, name=f"encoder_block_{i}"
            )(x, attention_mask)
        return nn.LayerNorm(name="encoder_final_norm")(x)


class TransformerDecoder(nn.Module):
    num_layers: int
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float
    deterministic: bool
    vocab_size: int
    max_len: int
    dtype: Any

    @nn.compact
    def __call__(self, target_tokens, encoder_outputs, decoder_self_attention_mask=None, cross_attention_mask=None):
        x = nn.Embed(self.vocab_size, self.embed_dim, name="token_embeddings")(target_tokens)
        x = PositionalEncoding(self.max_len, self.embed_dim, name="pos_enc")(x)
        x = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(x)
        for i in range(self.num_layers):
            x = DecoderBlock(
                embed_dim=self.embed_dim, num_heads=self.num_heads, qkv_features=self.qkv_features, mlp_features=self.mlp_features, dropout_rate=self.dropout_rate, deterministic=self.deterministic, dtype=self.dtype, name=f"decoder_block_{i}"
            )(x, encoder_outputs, decoder_self_attention_mask, cross_attention_mask)
        x = nn.LayerNorm(name="decoder_final_norm")(x)
        return nn.Dense(self.vocab_size, name="logits")(x)


class Transformer(nn.Module):
    encoder_num_layers: int
    decoder_num_layers: int
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float = 0.1
    deterministic: bool = False
    vocab_size: int = None
    max_len: int = 512
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, encoder_input_tokens, decoder_input_tokens, encoder_attention_mask=None, decoder_self_attention_mask=None, cross_attention_mask=None):
        encoder_outputs = TransformerEncoder(
            num_layers=self.encoder_num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            mlp_features=self.mlp_features,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            dtype=self.dtype,
            name="encoder",
        )(encoder_input_tokens, encoder_attention_mask)

        return TransformerDecoder(
            num_layers=self.decoder_num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            mlp_features=self.mlp_features,
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            dtype=self.dtype,
            name="decoder",
        )(decoder_input_tokens, encoder_outputs, decoder_self_attention_mask, cross_attention_mask)
