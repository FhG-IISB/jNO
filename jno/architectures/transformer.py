# https://github.com/voyager-jhk/JaxTransformer
import jax
import jax.numpy as jnp
import equinox as eqx
from .linear import Linear
from typing import Any, Optional


def _default_float_dtype():
    """Return JAX's current default floating dtype (float32 or float64)."""
    return jnp.asarray(0.0).dtype


class PositionalEncoding(eqx.Module):
    max_len: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(self, max_len, embed_dim, **kwargs):
        self.max_len = max_len
        self.embed_dim = embed_dim

    def __call__(self, inputs: jnp.ndarray, **kwargs) -> jnp.ndarray:
        seq_len = inputs.shape[1]
        position = jnp.arange(self.max_len, dtype=_default_float_dtype())[jnp.newaxis, :]
        div_term = jnp.exp(jnp.arange(0, self.embed_dim, 2) * -(jnp.log(10000.0) / self.embed_dim))
        pe = jnp.zeros((self.max_len, self.embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position.T * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position.T * div_term))
        pe = pe[jnp.newaxis, :seq_len, :]
        return inputs + jnp.array(pe, dtype=inputs.dtype)


class TransformerMLP(eqx.Module):
    wi: Linear
    wo: Linear
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, in_features, features, out_features, dropout_rate=0.1, *, key):
        k1, k2 = jax.random.split(key)
        self.wi = Linear(in_features, features, key=k1)
        self.wo = Linear(features, out_features, key=k2)
        self.dropout_rate = dropout_rate

    def __call__(self, x, *, key=None, **kwargs):
        x = jax.nn.relu(self.wi(x))
        if self.dropout_rate > 0 and key is not None:
            key, subkey = jax.random.split(key)
            x = eqx.nn.Dropout(p=self.dropout_rate)(x, key=subkey)
        x = self.wo(x)
        return x


class SelfAttention(eqx.Module):
    query_proj: Linear
    key_proj: Linear
    value_proj: Linear
    output_proj: Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, in_features, qkv_features, out_features, num_heads, dropout_rate=0.0, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.query_proj = Linear(in_features, qkv_features, key=k1)
        self.key_proj = Linear(in_features, qkv_features, key=k2)
        self.value_proj = Linear(in_features, qkv_features, key=k3)
        self.output_proj = Linear(qkv_features, out_features, key=k4)
        self.num_heads = num_heads
        self.head_dim = qkv_features // num_heads
        self.dropout_rate = dropout_rate

    def __call__(self, inputs_q, inputs_kv, mask=None, *, key=None, **kwargs):
        query = self.query_proj(inputs_q)
        k = self.key_proj(inputs_kv)
        value = self.value_proj(inputs_kv)

        def reshape_heads(x):
            return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim).swapaxes(-3, -2)

        q_h = reshape_heads(query)
        k_h = reshape_heads(k)
        v_h = reshape_heads(value)

        scale = jnp.sqrt(jnp.array(self.head_dim, dtype=q_h.dtype))
        attn_weights = jnp.matmul(q_h, k_h.swapaxes(-2, -1)) / scale

        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        if self.dropout_rate > 0 and key is not None:
            key, subkey = jax.random.split(key)
            attn_weights = eqx.nn.Dropout(p=self.dropout_rate)(attn_weights, key=subkey)

        attn_output = jnp.matmul(attn_weights, v_h)
        attn_output = attn_output.swapaxes(-3, -2).reshape(*attn_output.shape[:-3], -1, self.num_heads * self.head_dim)

        return self.output_proj(attn_output)


class EncoderBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    self_attention: SelfAttention
    mlp: TransformerMLP

    def __init__(self, embed_dim, num_heads, qkv_features, mlp_features, dropout_rate=0.1, *, key):
        k1, k2 = jax.random.split(key)
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)
        self.self_attention = SelfAttention(embed_dim, qkv_features, embed_dim, num_heads, dropout_rate=dropout_rate, key=k1)
        self.mlp = TransformerMLP(embed_dim, mlp_features, embed_dim, dropout_rate=dropout_rate, key=k2)

    def __call__(self, inputs, mask=None, *, key=None, **kwargs):
        k1, k2 = jax.random.split(key) if key is not None else (None, None)
        norm_inputs = jax.vmap(self.norm1)(inputs)
        attn_out = self.self_attention(norm_inputs, norm_inputs, mask, key=k1)
        x = inputs + attn_out
        norm_x = jax.vmap(self.norm2)(x)
        mlp_out = self.mlp(norm_x, key=k2)
        return x + mlp_out


class DecoderBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    norm3: eqx.nn.LayerNorm
    self_attention: SelfAttention
    cross_attention: SelfAttention
    mlp: TransformerMLP

    def __init__(self, embed_dim, num_heads, qkv_features, mlp_features, dropout_rate=0.1, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)
        self.norm3 = eqx.nn.LayerNorm(embed_dim)
        self.self_attention = SelfAttention(embed_dim, qkv_features, embed_dim, num_heads, dropout_rate=dropout_rate, key=k1)
        self.cross_attention = SelfAttention(embed_dim, qkv_features, embed_dim, num_heads, dropout_rate=dropout_rate, key=k2)
        self.mlp = TransformerMLP(embed_dim, mlp_features, embed_dim, dropout_rate=dropout_rate, key=k3)

    def __call__(self, inputs, encoder_outputs, self_attention_mask=None, cross_attention_mask=None, *, key=None, **kwargs):
        k1, k2, k3 = jax.random.split(key, 3) if key is not None else (None, None, None)
        norm_inputs = jax.vmap(self.norm1)(inputs)
        self_attn_out = self.self_attention(norm_inputs, norm_inputs, self_attention_mask, key=k1)
        x = inputs + self_attn_out
        norm_x = jax.vmap(self.norm2)(x)
        cross_attn_out = self.cross_attention(norm_x, encoder_outputs, cross_attention_mask, key=k2)
        x = x + cross_attn_out
        norm_x = jax.vmap(self.norm3)(x)
        mlp_out = self.mlp(norm_x, key=k3)
        return x + mlp_out


class TransformerEncoder(eqx.Module):
    token_embeddings: eqx.nn.Embedding
    pos_enc: PositionalEncoding
    blocks: list
    final_norm: eqx.nn.LayerNorm
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, num_layers, embed_dim, num_heads, qkv_features, mlp_features, dropout_rate, vocab_size, max_len, *, key):
        keys = jax.random.split(key, num_layers + 1)
        self.token_embeddings = eqx.nn.Embedding(vocab_size, embed_dim, key=keys[0])
        self.pos_enc = PositionalEncoding(max_len, embed_dim)
        self.blocks = [EncoderBlock(embed_dim, num_heads, qkv_features, mlp_features, dropout_rate=dropout_rate, key=keys[i + 1]) for i in range(num_layers)]
        self.final_norm = eqx.nn.LayerNorm(embed_dim)
        self.dropout_rate = dropout_rate

    def __call__(self, input_tokens, attention_mask=None, *, key=None, **kwargs):
        x = jax.vmap(self.token_embeddings)(input_tokens)
        x = self.pos_enc(x)
        if self.dropout_rate > 0 and key is not None:
            key, subkey = jax.random.split(key)
            x = eqx.nn.Dropout(p=self.dropout_rate)(x, key=subkey)
        for block in self.blocks:
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            x = block(x, attention_mask, key=subkey)
        return jax.vmap(self.final_norm)(x)


class TransformerDecoder(eqx.Module):
    token_embeddings: eqx.nn.Embedding
    pos_enc: PositionalEncoding
    blocks: list
    final_norm: eqx.nn.LayerNorm
    logits_layer: Linear
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, num_layers, embed_dim, num_heads, qkv_features, mlp_features, dropout_rate, vocab_size, max_len, *, key):
        keys = jax.random.split(key, num_layers + 2)
        self.token_embeddings = eqx.nn.Embedding(vocab_size, embed_dim, key=keys[0])
        self.pos_enc = PositionalEncoding(max_len, embed_dim)
        self.blocks = [DecoderBlock(embed_dim, num_heads, qkv_features, mlp_features, dropout_rate=dropout_rate, key=keys[i + 1]) for i in range(num_layers)]
        self.final_norm = eqx.nn.LayerNorm(embed_dim)
        self.logits_layer = Linear(embed_dim, vocab_size, key=keys[-1])
        self.dropout_rate = dropout_rate

    def __call__(self, target_tokens, encoder_outputs, decoder_self_attention_mask=None, cross_attention_mask=None, *, key=None, **kwargs):
        x = jax.vmap(self.token_embeddings)(target_tokens)
        x = self.pos_enc(x)
        if self.dropout_rate > 0 and key is not None:
            key, subkey = jax.random.split(key)
            x = eqx.nn.Dropout(p=self.dropout_rate)(x, key=subkey)
        for block in self.blocks:
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            x = block(x, encoder_outputs, decoder_self_attention_mask, cross_attention_mask, key=subkey)
        x = jax.vmap(self.final_norm)(x)
        return jax.vmap(self.logits_layer)(x)


class Transformer(eqx.Module):
    encoder: TransformerEncoder
    decoder: TransformerDecoder

    def __init__(self, encoder_num_layers, decoder_num_layers, embed_dim, num_heads, qkv_features, mlp_features, vocab_size, dropout_rate=0.1, max_len=512, *, key, **kwargs):
        k1, k2 = jax.random.split(key)
        self.encoder = TransformerEncoder(encoder_num_layers, embed_dim, num_heads, qkv_features, mlp_features, dropout_rate, vocab_size, max_len, key=k1)
        self.decoder = TransformerDecoder(decoder_num_layers, embed_dim, num_heads, qkv_features, mlp_features, dropout_rate, vocab_size, max_len, key=k2)

    def __call__(self, encoder_input_tokens, decoder_input_tokens, encoder_attention_mask=None, decoder_self_attention_mask=None, cross_attention_mask=None, *, key=None, **kwargs):
        k1, k2 = jax.random.split(key) if key is not None else (None, None)
        encoder_outputs = self.encoder(encoder_input_tokens, encoder_attention_mask, key=k1)
        return self.decoder(decoder_input_tokens, encoder_outputs, decoder_self_attention_mask, cross_attention_mask, key=k2)
