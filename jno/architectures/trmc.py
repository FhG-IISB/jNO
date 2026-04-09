import jax
import flax.linen as nn
import jax.numpy as jnp


class TRMCritic(nn.Module):
    hidden_dim: int
    num_blocks: int
    n: int
    T: int

    def setup(self, encoder_z, encoder_y):
        self.encoder_z = encoder_z  # SACEncoder(self.hidden_dim, self.num_blocks)
        self.encoder_y = encoder_y  # SACEncoder(self.hidden_dim, self.num_blocks)
        self.fc_out = nn.Dense(1)

    def latent_recursion(self, x, y, z):
        for _ in range(self.n):
            z_input = jnp.concatenate([x, y, z], axis=-1)
            z = self.encoder_z(z_input)

        y_input = jnp.concatenate([y, z], axis=-1)
        y = self.encoder_y(y_input)
        return y, z

    def deep_recursion(self, x, y, z):
        for _ in range(max(self.T - 1, 0)):
            y, z = self.latent_recursion(x, y, z)

        y = jax.lax.stop_gradient(y)
        z = jax.lax.stop_gradient(z)

        y, z = self.latent_recursion(x, y, z)
        return jax.lax.stop_gradient(y), jax.lax.stop_gradient(z), y

    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        batch_size = x.shape[0]
        y0 = jnp.zeros((batch_size, self.hidden_dim), dtype=x.dtype)
        z0 = jnp.zeros((batch_size, self.hidden_dim), dtype=x.dtype)

        _, _, y = self.deep_recursion(x, y0, z0)
        q = self.fc_out(y)
        return q
