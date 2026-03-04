"""Tests for jno.architectures.models — the neural network factory."""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from jno.trace import Model, TunableModule
from jno.architectures.models import nn, parameter


# ======================================================================
# parameter() helper
# ======================================================================
class TestParameter:
    def test_creates_flax_module(self):
        key = jax.random.PRNGKey(0)
        p = parameter((3, 2), key=key)
        assert isinstance(p, Model)

    def test_unique_ids(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        p1 = parameter((1,), key=k1)
        p2 = parameter((1,), key=k2)
        assert p1.layer_id != p2.layer_id

    def test_parameter_shape(self):
        key = jax.random.PRNGKey(0)
        p = parameter((3, 2), key=key)
        result = p.module()
        assert result.shape == (3, 2)


# ======================================================================
# nn.wrap
# ======================================================================
class TestNNWrap:
    def test_wrap_module_instance(self):
        key = jax.random.PRNGKey(0)
        module = eqx.nn.Linear(4, 2, key=key)
        wrapped = nn.wrap(module)
        assert isinstance(wrapped, Model)

    def test_wrap_with_name(self):
        key = jax.random.PRNGKey(0)
        module = eqx.nn.Linear(4, 2, key=key)
        wrapped = nn.wrap(module, name="my_dense")
        assert wrapped.name == "my_dense"


# ======================================================================
# nn.mlp
# ======================================================================
class TestNNMLP:
    def test_returns_flax_module(self):
        key = jax.random.PRNGKey(0)
        m = nn.mlp(3, output_dim=1, hidden_dims=16, num_layers=2, key=key)
        assert isinstance(m, Model)

    def test_different_configs(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        m1 = nn.mlp(3, output_dim=1, hidden_dims=32, num_layers=3, key=k1)
        m2 = nn.mlp(3, output_dim=2, hidden_dims=16, num_layers=1, key=k2)
        assert m1.layer_id != m2.layer_id

    def test_forward_shape(self):
        """Verify forward pass output shape."""
        key = jax.random.PRNGKey(0)
        m = nn.mlp(3, output_dim=4, hidden_dims=16, num_layers=2, key=key)
        x = jnp.ones((5, 3))
        y = m.module(x)
        assert y.shape == (5, 4)


# ======================================================================
# nn.fno1d
# ======================================================================
class TestFNO1D:
    def test_returns_flax_module(self):
        key = jax.random.PRNGKey(0)
        m = nn.fno1d(1, hidden_channels=8, n_modes=4, d_vars=1, key=key)
        assert isinstance(m, Model)

    @pytest.mark.slow
    def test_forward_shape(self):
        key = jax.random.PRNGKey(0)
        m = nn.fno1d(1, hidden_channels=8, n_modes=4, d_vars=1, n_layers=2, key=key)
        x = jnp.ones((1, 16, 1))  # (batch, spatial, channels)
        y = m.module(x, key=key)
        assert y.shape[0] == 1
        assert y.shape[1] == 16


# ======================================================================
# nn.deeponet (if available)
# ======================================================================
class TestDeepONet:
    def test_returns_flax_module(self):
        key = jax.random.PRNGKey(0)
        m = nn.deeponet(
            n_sensors=16,
            sensor_channels=1,
            coord_dim=1,
            n_outputs=1,
            basis_functions=8,
            hidden_dim=16,
            n_layers=2,
            key=key,
        )
        assert isinstance(m, Model)


# ======================================================================
# Import shadowing fix verification
# ======================================================================
class TestImportShadowing:
    """Verify pcno vs geofno compute_Fourier_modes aren't shadowed."""

    def test_both_functions_accessible(self):
        from jno.architectures.models import (
            pcno_compute_Fourier_modes,
            geofno_compute_Fourier_modes,
        )

        assert pcno_compute_Fourier_modes is not geofno_compute_Fourier_modes
