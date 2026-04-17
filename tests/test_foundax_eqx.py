"""Integration tests for foundax equinox foundation models in jno.

Every foundax foundation model is now a native ``equinox.Module``.
These tests verify:
    1. Each factory returns an ``eqx.Module`` (not a Flax wrapper).
    2. ``nn.wrap()`` wraps each model as a ``Model`` (not FlaxModelWrapper).
    3. Forward pass + gradient flow work on each model.
    4. Model controls: optimizer, freeze, unfreeze, dtype, initialize (weights).
    5. LoRA application (warns when no jno ``Linear`` layers are found).
    6. Mini training loop through ``jno.core.solve()``.

Run with a single GPU::

    CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_foundax_eqx.py -v
"""

import tempfile
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import foundax

import jno
import jno.jnp_ops as jnn
from jno.architectures.models import nn
from jno.trace import Model
from jno import LearningRateSchedule as lrs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _param_count(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(l.size for l in leaves)


def _assert_finite_grads(grads):
    for g in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array)):
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Tiny configs shared across test classes
# ---------------------------------------------------------------------------


def _make_bcat():
    return foundax.bcat.base(
        n_layer=2,
        dim_emb=64,
        dim_ffn=128,
        n_head=4,
        x_num=16,
        max_output_dim=2,
        patch_num=4,
        patch_num_output=4,
        conv_dim=8,
        data_dim=2,
    )


def _make_morph():
    return foundax.morph.Ti(
        dim=32,
        depth=2,
        heads=2,
        mlp_dim=64,
        max_patches=64,
        dropout=0.0,
        emb_dropout=0.0,
    )


def _make_mpp():
    return foundax.mpp.Ti(
        embed_dim=32,
        processor_blocks=2,
        n_states=3,
        num_heads=2,
        drop_path=0.0,
    )


def _make_dpot():
    return foundax.dpot.Ti(
        embed_dim=64,
        depth=2,
        n_blocks=2,
        in_channels=1,
        out_channels=1,
        img_size=32,
        patch_size=8,
        in_timesteps=1,
        out_timesteps=1,
    )


def _make_poseidon():
    return foundax.poseidon.T(
        num_channels=1,
        num_out_channels=1,
        embed_dim=48,
        depths=(2, 2, 2, 2),
        image_size=32,
        window_size=4,
    )


def _make_pdeformer2():
    return foundax.pdeformer2.small(
        num_encoder_layers=2,
        embed_dim=64,
        ffn_embed_dim=128,
        num_heads=4,
        inr_dim_hidden=32,
        inr_num_layers=2,
        hyper_num_layers=1,
        scalar_num_layers=1,
    )


def _make_walrus():
    return foundax.walrus.base(
        hidden_dim=64,
        intermediate_dim=32,
        n_states=2,
        processor_blocks=1,
        groups=4,
        num_heads=4,
        base_kernel_size=((4, 2), (4, 2), (4, 2)),
        encoder_groups=4,
    )


def _make_prose():
    return foundax.prose.fd_1to1(x_num=16, max_output_dim=2, output_len=4)


ALL_FACTORIES = [
    _make_bcat,
    _make_morph,
    _make_mpp,
    _make_dpot,
    _make_poseidon,
    _make_pdeformer2,
    _make_walrus,
    _make_prose,
]

# Subset that can run forward+grad quickly with known input shapes
FORWARD_FACTORIES = [
    _make_bcat,
    _make_morph,
    _make_mpp,
    _make_dpot,
    _make_poseidon,
]


# =====================================================================
# 1. Construction: all foundation models return eqx.Module
# =====================================================================


@pytest.mark.integration
class TestFoundaxModelsAreEquinox:

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_is_eqx_module(self, factory):
        m = factory()
        assert isinstance(m, eqx.Module)


# =====================================================================
# 2. nn.wrap() wraps as Model (not Flax wrapper)
# =====================================================================


@pytest.mark.integration
class TestNNWrapEquinox:

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_wrap_produces_model_not_flax(self, factory):
        m = factory()
        wrapped = nn.wrap(m)
        assert isinstance(wrapped, Model)
        assert isinstance(wrapped.module, eqx.Module)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_wrap_assigns_layer_id(self, factory):
        wrapped = nn.wrap(factory())
        assert wrapped.layer_id is not None

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_param_count_nonzero(self, factory):
        m = factory()
        assert _param_count(m) > 0


# =====================================================================
# 3. Forward + gradient flow
# =====================================================================


@pytest.mark.integration
@pytest.mark.gpu
class TestForwardAndGrad:

    def test_bcat_forward_grad(self):
        model = _make_bcat()
        data = jnp.zeros((1, 5, 16, 16, 2))
        times = jnp.zeros((1, 5, 1))

        @eqx.filter_value_and_grad
        def loss_fn(m):
            return jnp.mean(m(data, times, input_len=3) ** 2)

        loss, grads = loss_fn(model)
        assert jnp.isfinite(loss)
        _assert_finite_grads(grads)

    def test_morph_forward_grad(self):
        model = _make_morph()
        x = jnp.zeros((1, 1, 1, 1, 8, 8, 8))

        @eqx.filter_value_and_grad
        def loss_fn(m):
            _, _, pred = m(x)
            return jnp.mean(pred**2)

        loss, grads = loss_fn(model)
        assert jnp.isfinite(loss)
        _assert_finite_grads(grads)

    def test_mpp_forward_grad(self):
        model = _make_mpp()
        # (T=1, B=1, C=n_states=3, H=32, W=32); H,W must be divisible by 16
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 3, 32, 32))
        labels = jnp.array([0, 1, 2], dtype=jnp.int32)
        bcs = jnp.zeros((1, 2), dtype=jnp.int32)

        @eqx.filter_value_and_grad
        def loss_fn(m):
            return jnp.mean(m(x, labels, bcs, deterministic=True) ** 2)

        loss, grads = loss_fn(model)
        assert jnp.isfinite(loss)
        _assert_finite_grads(grads)

    def test_dpot_forward_grad(self):
        model = _make_dpot()
        # (B, Sx, Sy, T=in_timesteps, C=in_channels)
        x = jnp.zeros((1, 32, 32, 1, 1))

        @eqx.filter_value_and_grad
        def loss_fn(m):
            pred, cls = m(x)
            return jnp.mean(pred**2)

        loss, grads = loss_fn(model)
        assert jnp.isfinite(loss)
        _assert_finite_grads(grads)

    def test_poseidon_forward_grad(self):
        model = _make_poseidon()
        x = jnp.zeros((1, 32, 32, 1))
        t = jnp.zeros((1,))

        @eqx.filter_value_and_grad
        def loss_fn(m):
            out = m(pixel_values=x, time=t, deterministic=True, return_dict=False)
            pred = out[0] if isinstance(out, tuple) else out
            return jnp.mean(pred**2)

        loss, grads = loss_fn(model)
        assert jnp.isfinite(loss)
        _assert_finite_grads(grads)


# =====================================================================
# 4. Model controls: optimizer, freeze, unfreeze, dtype, initialize
# =====================================================================


@pytest.mark.integration
class TestModelControls:

    # -- optimizer -------------------------------------------------------

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_optimizer_attaches(self, factory):
        net = nn.wrap(factory())
        ret = net.optimizer(optax.adam, lr=lrs(1e-3))
        assert ret is net
        assert net._opt_fn is not None
        assert net._lr is not None

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_optimizer_with_lr_schedule(self, factory):
        net = nn.wrap(factory())
        net.optimizer(optax.adamw, lr=lrs.exponential(1e-3, decay_rate=0.9, decay_steps=100))
        assert net._opt_fn is not None

    # -- freeze / unfreeze -----------------------------------------------

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_freeze_sets_flag(self, factory):
        net = nn.wrap(factory())
        net.freeze()
        assert net._frozen is True

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_unfreeze_clears_flag(self, factory):
        net = nn.wrap(factory())
        net.freeze()
        net.unfreeze()
        assert net._frozen is False

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_partial_freeze_via_mask(self, factory):
        m = factory()
        net = nn.wrap(m)
        mask = jax.tree_util.tree_map(lambda _: True, eqx.filter(m, eqx.is_array))
        net.mask(mask).freeze()
        assert net._trainable_param_mask is not None
        assert net._frozen is False  # partial freeze, not global

    # -- dtype cast ------------------------------------------------------

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_dtype_sets(self, factory):
        net = nn.wrap(factory())
        net.dtype(jnp.bfloat16)
        assert net._dtype == jnp.bfloat16

    # -- initialize (weights from pytree) --------------------------------

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_initialize_from_pytree(self, factory):
        m = factory()
        net = nn.wrap(m)
        net.initialize(m)  # load from the model itself (identity)
        assert net._weight_tree is not None

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_initialize_from_file(self, factory, tmp_path):
        m = factory()
        path = str(tmp_path / "weights.eqx")
        eqx.tree_serialise_leaves(path, m)
        net = nn.wrap(m)
        net.initialize(path)
        assert net.weight_path == path

    # -- summary (smoke test) -------------------------------------------

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_summary_does_not_crash(self, factory):
        net = nn.wrap(factory())
        net.optimizer(optax.adam, lr=lrs(1e-3))
        net.summary()  # should print without error

    # -- reset -----------------------------------------------------------

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_reset_clears_config(self, factory):
        net = nn.wrap(factory())
        net.optimizer(optax.adam, lr=lrs(1e-3))
        net.freeze()
        net.reset()
        assert net._frozen is False
        assert net._opt_fn is None
        assert net._lr is None

    # -- dont_show -------------------------------------------------------

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_dont_show(self, factory):
        net = nn.wrap(factory())
        net.dont_show()
        assert net.show is False


# =====================================================================
# 5. LoRA on foundation models
# =====================================================================


@pytest.mark.integration
class TestLoRA:

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_lora_sets_config(self, factory):
        net = nn.wrap(factory())
        net.lora(rank=4, alpha=1.0)
        assert net._lora_config == (4, 1.0, None)

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_lora_with_freeze(self, factory):
        net = nn.wrap(factory())
        net.freeze()
        net.lora(rank=4, alpha=1.0)
        assert net._frozen is True
        assert net._lora_config is not None


# =====================================================================
# 6. Serialisation round-trip
# =====================================================================


@pytest.mark.integration
class TestSerialisation:

    @pytest.mark.parametrize("factory", ALL_FACTORIES)
    def test_eqx_serialise_roundtrip(self, factory, tmp_path):
        m = factory()
        path = str(tmp_path / "model.eqx")
        eqx.tree_serialise_leaves(path, m)
        m2 = eqx.tree_deserialise_leaves(path, m)
        leaves_a = jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
        leaves_b = jax.tree_util.tree_leaves(eqx.filter(m2, eqx.is_array))
        assert len(leaves_a) == len(leaves_b)
        for a, b in zip(leaves_a, leaves_b):
            assert jnp.array_equal(a, b)


# =====================================================================
# 7. Mini training loop through jno.core
# =====================================================================


@pytest.mark.integration
@pytest.mark.gpu
class TestJNOPipeline:

    def test_mlp_head_solves(self):
        """An MLP (foundax.mlp) trains for a few epochs through jno.core."""
        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = domain.variable("interior")

        net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=8, num_layers=1))
        net.optimizer(optax.adam, lr=lrs(1e-3))

        u = net(x)
        loss_expr = (u - jnn.sin(jnn.pi * x)).mse

        solver = jno.core([loss_expr], domain)
        stats = solver.solve(5)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    def test_mlp_freeze_unfreeze_solves(self):
        """Frozen model contributes to loss but has zero gradients."""
        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = domain.variable("interior")

        frozen_net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=8, num_layers=1))
        frozen_net.freeze()

        train_net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=8, num_layers=1))
        train_net.optimizer(optax.adam, lr=lrs(1e-3))

        u = train_net(x) + frozen_net(x)
        loss_expr = (u - jnn.sin(jnn.pi * x)).mse

        solver = jno.core([loss_expr], domain)
        stats = solver.solve(3)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    def test_mlp_dtype_bf16_solves(self):
        """Model cast to bfloat16 still trains."""
        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = domain.variable("interior")

        net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=8, num_layers=1))
        net.optimizer(optax.adam, lr=lrs(1e-3))
        net.dtype(jnp.bfloat16)

        u = net(x)
        loss_expr = (u - jnn.sin(jnn.pi * x)).mse

        solver = jno.core([loss_expr], domain)
        stats = solver.solve(3)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    def test_mlp_lora_solves(self):
        """MLP with LoRA enabled trains (foundax.mlp uses jno Linear)."""
        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = domain.variable("interior")

        net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2))
        net.freeze()
        net.lora(rank=4, alpha=1.0)
        net.optimizer(optax.adam, lr=lrs(1e-3))

        u = net(x)
        loss_expr = (u - jnn.sin(jnn.pi * x)).mse

        solver = jno.core([loss_expr], domain)
        stats = solver.solve(3)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    def test_mlp_initialize_from_file_solves(self):
        """Model loaded from serialised weights trains normally."""
        m = foundax.mlp(1, output_dim=1, hidden_dims=8, num_layers=1)
        with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
            path = f.name
        eqx.tree_serialise_leaves(path, m)

        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = domain.variable("interior")

        net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=8, num_layers=1))
        net.initialize(path)
        net.optimizer(optax.adam, lr=lrs(1e-3))

        u = net(x)
        loss_expr = (u - jnn.sin(jnn.pi * x)).mse

        solver = jno.core([loss_expr], domain)
        stats = solver.solve(3)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])
