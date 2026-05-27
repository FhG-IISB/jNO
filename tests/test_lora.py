"""Tests for jno.lora — LoRA adapter classes and utilities.

Covers:
  - Public namespace: jno.lora.<class>
  - Core utilities: apply_lora, merge_lora, lora_trainable_filter
  - LoRA Zoo: LoRALinear, rsLoRALinear, LoRAFALinear, DoRALinear, PiSSALinear, LoRAXSLinear
  - apply_lora options: target regex, per-spec routing, list-of-wrappers, custom wrapper
  - Initialisation invariant: all adapters are no-ops at init (output == base)
  - Merge: merged output == adapted output; no LoRAWrapper nodes remain
  - Trainable filter: only adapter arrays marked True; base weights marked False
  - Per-class invariants: DoRA magnitude, PiSSA residual, LoRAXS R=0, rsLoRA scaling
  - Regression: "wrappers" key from Model.lora() config is honoured by apply_lora
  - Model.lora() API: config structure, wrapper=, specs=
  - Integration: end-to-end training through jno.core
"""

import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import optax
import pytest

import jno
import jno.jnp_ops as jnn
from jno import LearningRateSchedule as lrs
from jno.architectures.models import nn
from jno.lora import (
    DoRALinear,
    LoRAFALinear,
    LoRALinear,
    LoRAWrapper,
    LoRAXSLinear,
    PiSSALinear,
    apply_lora,
    lora_trainable_filter,
    merge_lora,
    rsLoRALinear,
)

KEY = jax.random.PRNGKey(42)

# Small foundax MLP: 4→8→8→2  (3 linear layers)
_MLP_KW = dict(in_features=4, output_dim=2, hidden_dims=8, num_layers=2)

# Expected adapter leaf count for the 3-layer MLP with rank=2
_ADAPTER_LEAVES: dict[type[LoRAWrapper], int] = {
    LoRALinear:   6,   # 3 × (lora_A, lora_B)
    rsLoRALinear: 6,
    LoRAFALinear: 3,   # 3 × (lora_B,)
    DoRALinear:   9,   # 3 × (magnitude, lora_A, lora_B)
    PiSSALinear:  6,
    LoRAXSLinear: 3,   # 3 × (R,)
}

ZOO = list(_ADAPTER_LEAVES.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mlp(key=KEY):
    return foundax.mlp(**_MLP_KW, key=key)


def _count_wrappers(model) -> int:
    return sum(
        1
        for leaf in jax.tree_util.tree_leaves(model, is_leaf=lambda x: isinstance(x, LoRAWrapper))
        if isinstance(leaf, LoRAWrapper)
    )


def _count_adapter_leaves(model) -> int:
    return sum(1 for v in jax.tree_util.tree_leaves(lora_trainable_filter(model)) if v)


# ---------------------------------------------------------------------------
# 1. Public namespace
# ---------------------------------------------------------------------------


class TestNamespace:
    def test_zoo_classes_accessible(self):
        for cls in ZOO:
            assert getattr(jno.lora, cls.__name__) is cls

    def test_wrapper_base_accessible(self):
        assert jno.lora.LoRAWrapper is LoRAWrapper

    def test_utilities_accessible(self):
        assert jno.lora.apply_lora is apply_lora
        assert jno.lora.merge_lora is merge_lora
        assert jno.lora.lora_trainable_filter is lora_trainable_filter


# ---------------------------------------------------------------------------
# 2. apply_lora — adapter creation
# ---------------------------------------------------------------------------


class TestApplyLora:
    def test_wraps_foundax_linear(self):
        """apply_lora must wrap foundax.Linear layers (not silently no-op)."""
        adapted = apply_lora(_mlp(), rank=2, alpha=1.0, key=KEY)
        assert _count_wrappers(adapted) > 0

    def test_wraps_eqx_nn_linear(self):
        """apply_lora must recognise eqx.nn.Linear (used in ScOT / poseidon)."""

        class TinyNet(eqx.Module):
            a: eqx.nn.Linear
            b: eqx.nn.Linear

            def __call__(self, x):
                return self.b(jax.nn.relu(self.a(x)))

        net = TinyNet(a=eqx.nn.Linear(4, 8, key=KEY), b=eqx.nn.Linear(8, 2, key=KEY))
        adapted = apply_lora(net, rank=2, alpha=1.0, key=KEY)
        assert _count_wrappers(adapted) == 2, "eqx.nn.Linear layers were not wrapped"

    @pytest.mark.parametrize("cls", ZOO)
    def test_adapter_leaf_count(self, cls):
        adapted = apply_lora(_mlp(), rank=2, alpha=1.0, key=KEY, wrappers=(cls,))
        assert _count_adapter_leaves(adapted) == _ADAPTER_LEAVES[cls]

    def test_target_regex_restricts_layers(self):
        """target= regex wraps only matching layers; fewer than the default."""
        mlp = _mlp()
        n_all = _count_wrappers(apply_lora(mlp, rank=2, alpha=1.0, key=KEY))
        adapted = apply_lora(mlp, rank=2, alpha=1.0, key=KEY,
                             specs=[{"target": "0", "rank": 2, "alpha": 1.0}])
        n_filtered = _count_wrappers(adapted)
        assert 0 < n_filtered < n_all

    def test_no_double_wrap(self):
        """A second apply_lora call must not re-wrap already-adapted layers."""
        adapted = apply_lora(_mlp(), rank=2, alpha=1.0, key=KEY)
        adapted2 = apply_lora(adapted, rank=2, alpha=1.0, key=KEY)
        assert _count_wrappers(adapted) == _count_wrappers(adapted2)

    def test_per_spec_different_wrappers(self):
        """Layers with different paths can receive different adapter classes."""
        adapted = apply_lora(_mlp(), key=KEY, specs=[
            {"target": "0", "rank": 2, "alpha": 1.0, "wrapper": rsLoRALinear},
            {"target": ".*", "rank": 2, "alpha": 1.0, "wrapper": LoRALinear},
        ])
        wrapper_types = {
            type(leaf)
            for leaf in jax.tree_util.tree_leaves(adapted, is_leaf=lambda x: isinstance(x, LoRAWrapper))
            if isinstance(leaf, LoRAWrapper)
        }
        assert rsLoRALinear in wrapper_types
        assert LoRALinear in wrapper_types

    def test_list_of_wrappers_first_match_wins(self):
        """When a list is passed, the first class whose applies_to() returns True wins."""
        adapted = apply_lora(_mlp(), rank=2, alpha=1.0, key=KEY,
                             wrappers=(rsLoRALinear, LoRALinear))
        leaves = jax.tree_util.tree_leaves(adapted, is_leaf=lambda x: isinstance(x, LoRAWrapper))
        assert all(isinstance(leaf, rsLoRALinear) for leaf in leaves if isinstance(leaf, LoRAWrapper))

    def test_custom_wrapper(self):
        """A user-defined LoRAWrapper subclass plugs in and its adapter fields are counted."""
        from jno.architectures.lora import LinearLike

        class BiasAdapter(LoRAWrapper):
            adapter_fields = ("bias_delta",)
            base: eqx.Module
            bias_delta: jax.Array
            rank: int = eqx.field(static=True)
            alpha: float = eqx.field(static=True)

            @classmethod
            def applies_to(cls, leaf):
                return isinstance(leaf, LinearLike) and not isinstance(leaf, LoRAWrapper)

            def __init__(self, base, rank, alpha, *, key):
                self.base, self.rank, self.alpha = base, rank, alpha
                self.bias_delta = jnp.zeros(base.out_features)

            def __call__(self, x):
                return self.base(x) + self.bias_delta

            def merge(self):
                return self.base

        mlp = _mlp()
        adapted = apply_lora(mlp, rank=2, alpha=1.0, key=KEY, wrappers=(BiasAdapter,))
        n = _count_wrappers(adapted)
        assert n > 0
        assert _count_adapter_leaves(adapted) == n  # 1 adapter field per wrapper


# ---------------------------------------------------------------------------
# 3. Init invariant: all zoo adapters are no-ops at init
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ZOO)
def test_output_equals_base_at_init(cls):
    """Adapted model output must equal the base model at initialisation."""
    mlp = _mlp()
    adapted = apply_lora(mlp, rank=2, alpha=1.0, key=KEY, wrappers=(cls,))
    x = jnp.ones((4,))
    assert jnp.allclose(mlp(x), adapted(x), atol=1e-5), (
        f"{cls.__name__}: output differs from base at init"
    )


# ---------------------------------------------------------------------------
# 4. merge_lora correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ZOO)
class TestMergeLora:
    @staticmethod
    def _perturb(model):
        """Shift every array leaf slightly so the adapter produces a non-zero delta."""
        leaves, treedef = jax.tree_util.tree_flatten(model)
        leaves = [v + 0.01 * jnp.ones_like(v) if eqx.is_array(v) else v for v in leaves]
        return jax.tree_util.tree_unflatten(treedef, leaves)

    def test_merged_output_equals_adapted(self, cls):
        """After merge, the plain model must produce the same output as the adapted one."""
        adapted = self._perturb(apply_lora(_mlp(), rank=2, alpha=1.0, key=KEY, wrappers=(cls,)))
        merged = merge_lora(adapted)
        x = jax.random.normal(KEY, (4,))
        assert jnp.allclose(adapted(x), merged(x), atol=1e-5), (
            f"{cls.__name__}: merge_lora output differs from adapted output"
        )

    def test_no_wrapper_nodes_after_merge(self, cls):
        """merge_lora must remove every LoRAWrapper node from the tree."""
        adapted = apply_lora(_mlp(), rank=2, alpha=1.0, key=KEY, wrappers=(cls,))
        merged = merge_lora(adapted)
        assert _count_wrappers(merged) == 0, (
            f"{cls.__name__}: LoRAWrapper nodes remain after merge_lora"
        )

    def test_restores_original_linear_class(self, cls):
        """merge_lora must restore the original layer class (foundax.Linear), not jno.Linear."""
        from foundax.architectures.linear import Linear as FoundaxLinear

        adapted = apply_lora(_mlp(), rank=2, alpha=1.0, key=KEY, wrappers=(cls,))
        merged = merge_lora(adapted)
        linear_leaves = [
            leaf
            for leaf in jax.tree_util.tree_leaves(
                merged, is_leaf=lambda x: isinstance(x, eqx.Module) and hasattr(x, "weight")
            )
            if hasattr(leaf, "weight") and hasattr(leaf, "in_features")
        ]
        assert all(isinstance(leaf, FoundaxLinear) for leaf in linear_leaves), (
            f"{cls.__name__}: merge_lora did not restore foundax.Linear"
        )


# ---------------------------------------------------------------------------
# 5. lora_trainable_filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ZOO)
class TestTrainableFilter:
    def test_correct_adapter_count(self, cls):
        adapted = apply_lora(_mlp(), rank=2, alpha=1.0, key=KEY, wrappers=(cls,))
        n_true = _count_adapter_leaves(adapted)
        assert n_true == _ADAPTER_LEAVES[cls], (
            f"{cls.__name__}: expected {_ADAPTER_LEAVES[cls]} adapter leaves, got {n_true}"
        )

    def test_base_arrays_not_marked(self, cls):
        """Total True leaves == expected adapter count; no base weight bleeds through."""
        adapted = apply_lora(_mlp(), rank=2, alpha=1.0, key=KEY, wrappers=(cls,))
        n_true = _count_adapter_leaves(adapted)
        assert n_true == _ADAPTER_LEAVES[cls]


# ---------------------------------------------------------------------------
# 6. Per-class invariants
# ---------------------------------------------------------------------------


class TestDoRAInvariants:
    def test_magnitude_initialized_to_row_norms(self):
        lin = eqx.nn.Linear(4, 8, key=KEY)
        adapted = DoRALinear(lin, rank=2, alpha=1.0, key=KEY)
        expected = jnp.linalg.norm(lin.weight, axis=1)
        assert jnp.allclose(adapted.magnitude, expected, atol=1e-6)

    def test_adapter_fields(self):
        assert set(DoRALinear.adapter_fields) == {"magnitude", "lora_A", "lora_B"}


class TestPiSSAInvariants:
    def test_residual_plus_adapter_equals_original_weight(self):
        """base.weight + (alpha/rank)*B@A must exactly reconstruct the original weight."""
        lin = eqx.nn.Linear(4, 8, key=KEY)
        r, alpha = 2, 4.0  # non-unity alpha/rank to expose scaling bugs
        adapted = PiSSALinear(lin, rank=r, alpha=alpha, key=KEY)
        W_reconstructed = adapted.base.weight + (alpha / r) * (adapted.lora_B @ adapted.lora_A)
        assert jnp.allclose(W_reconstructed, lin.weight, atol=1e-5), (
            "PiSSA residual is wrong: base + adapter does not recover the original weight"
        )

    def test_adapter_fields(self):
        assert set(PiSSALinear.adapter_fields) == {"lora_A", "lora_B"}


class TestLoRAXSInvariants:
    def test_r_initialized_to_zero(self):
        lin = eqx.nn.Linear(4, 8, key=KEY)
        adapted = LoRAXSLinear(lin, rank=2, alpha=1.0, key=KEY)
        assert jnp.all(adapted.R == 0)

    def test_only_r_is_trainable(self):
        """lora_A and lora_B are frozen — only R is in adapter_fields."""
        assert LoRAXSLinear.adapter_fields == ("R",)
        assert "lora_A" not in LoRAXSLinear.adapter_fields
        assert "lora_B" not in LoRAXSLinear.adapter_fields


class TestRsLoRAInvariants:
    def test_scaling_is_alpha_over_sqrt_rank(self):
        """rsLoRA uses alpha/√rank; with the same A,B its output must differ from LoRALinear."""
        lin = eqx.nn.Linear(4, 8, key=KEY)
        x = jax.random.normal(KEY, (4,))

        std = LoRALinear(lin, rank=4, alpha=1.0, key=KEY)
        rs = rsLoRALinear(lin, rank=4, alpha=1.0, key=KEY)
        # Force a shared non-zero lora_B so the scaling difference is visible.
        delta_B = jax.random.normal(KEY, std.lora_B.shape)
        std = eqx.tree_at(lambda m: m.lora_B, std, delta_B)
        rs = eqx.tree_at(lambda m: m.lora_B, rs, delta_B)

        assert not jnp.allclose(std(x), rs(x), atol=1e-6), (
            "rsLoRALinear and LoRALinear produce identical output — scaling fix not applied"
        )


class TestLoRAFAInvariants:
    def test_only_lora_b_is_trainable(self):
        assert LoRAFALinear.adapter_fields == ("lora_B",)
        assert "lora_A" not in LoRAFALinear.adapter_fields

    def test_fewer_trainable_params_vs_standard(self):
        """LoRAFALinear must have fewer trainable adapter params than LoRALinear."""
        mlp = _mlp()

        def _adapter_param_size(model):
            filt = lora_trainable_filter(model)
            flat_arrays = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
            flat_filt = jax.tree_util.tree_leaves(filt)
            return sum(a.size for a, f in zip(flat_arrays, flat_filt) if f)

        n_std = _adapter_param_size(apply_lora(mlp, rank=2, alpha=1.0, key=KEY, wrappers=(LoRALinear,)))
        n_fa  = _adapter_param_size(apply_lora(mlp, rank=2, alpha=1.0, key=KEY, wrappers=(LoRAFALinear,)))
        assert n_fa < n_std


class TestRankClamping:
    @pytest.mark.parametrize("cls", [PiSSALinear, DoRALinear, LoRAXSLinear])
    def test_rank_clamped_to_min_dim(self, cls):
        lin = eqx.nn.Linear(4, 2, key=KEY)  # min dim = 2
        adapted = cls(lin, rank=8, alpha=1.0, key=KEY)
        assert adapted.rank == 2


# ---------------------------------------------------------------------------
# 7. Model.lora() API
# ---------------------------------------------------------------------------


class TestModelLoraAPI:
    def _net(self):
        return nn.wrap(foundax.mlp(**_MLP_KW, key=KEY))

    def test_default_config(self):
        net = self._net()
        net.lora(rank=4, alpha=1.0)
        assert net._lora_config == [
            {"target": None, "rank": 4, "alpha": 1.0, "wrappers": (LoRALinear,)}
        ]

    def test_wrapper_param_single(self):
        net = self._net()
        net.lora(rank=4, alpha=1.0, wrapper=rsLoRALinear)
        assert net._lora_config[0]["wrappers"] == (rsLoRALinear,)

    def test_wrapper_param_list(self):
        net = self._net()
        net.lora(rank=4, alpha=1.0, wrapper=[rsLoRALinear, LoRALinear])
        assert net._lora_config[0]["wrappers"] == (rsLoRALinear, LoRALinear)

    def test_specs_stores_multiple_groups(self):
        net = self._net()
        net.lora(specs=[
            {"target": "encoder", "rank": 4,  "alpha": 1.0},
            {"target": "decoder", "rank": 16, "alpha": 4.0},
        ])
        assert len(net._lora_config) == 2
        assert net._lora_config[0]["rank"] == 4
        assert net._lora_config[1]["rank"] == 16

    def test_specs_default_wrapper_is_lora_linear(self):
        net = self._net()
        net.lora(specs=[{"target": ".*", "rank": 4, "alpha": 1.0}])
        assert net._lora_config[0]["wrappers"] == (LoRALinear,)

    def test_specs_per_spec_wrapper(self):
        net = self._net()
        net.lora(specs=[{"target": ".*", "rank": 4, "alpha": 1.0, "wrapper": DoRALinear}])
        assert net._lora_config[0]["wrappers"] == (DoRALinear,)

    def test_freeze_and_lora_coexist(self):
        net = self._net()
        net.freeze()
        net.lora(rank=4, alpha=1.0)
        assert net._frozen is True
        assert net._lora_config is not None

    def test_chainable(self):
        net = self._net()
        result = net.lora(rank=4).optimizer(optax.adam, lr=lrs(1e-3))
        assert result is net


# ---------------------------------------------------------------------------
# 8. Regression: "wrappers" key in _lora_config is honoured by apply_lora
# ---------------------------------------------------------------------------


class TestRegressionWrappersKey:
    def test_custom_wrapper_via_model_lora_is_applied(self):
        """When wrapper=rsLoRALinear is passed to .lora(), the adapters are actually rsLoRALinear.

        Regression: apply_lora read "wrapper" (singular) from spec dicts but
        Model.lora() stored "wrappers" (plural), so the custom class was silently
        replaced by LoRALinear.
        """

        mlp = _mlp()
        # Simulate what core.py does: _apply_lora(model, key=key, specs=fm._lora_config)
        net = nn.wrap(mlp)
        net.lora(rank=2, alpha=1.0, wrapper=rsLoRALinear)
        adapted = apply_lora(mlp, key=KEY, specs=net._lora_config)

        wrapper_types = {
            type(leaf)
            for leaf in jax.tree_util.tree_leaves(adapted, is_leaf=lambda x: isinstance(x, LoRAWrapper))
            if isinstance(leaf, LoRAWrapper)
        }
        assert rsLoRALinear in wrapper_types, (
            "apply_lora ignored the 'wrappers' key and fell back to LoRALinear"
        )
        assert LoRALinear not in wrapper_types

    def test_pissa_output_equals_base_with_nonunity_alpha_rank(self):
        """PiSSA residual must be W - (alpha/rank)*B@A, not W - B@A.

        Regression: the original code stored W - B@A as the residual, which only
        gives identical init output when alpha == rank.
        """
        lin = eqx.nn.Linear(4, 8, key=KEY)
        x = jax.random.normal(KEY, (4,))

        for alpha in [0.5, 2.0, 8.0]:  # all != rank
            adapted = PiSSALinear(lin, rank=2, alpha=alpha, key=KEY)
            assert jnp.allclose(lin(x), adapted(x), atol=1e-5), (
                f"PiSSA (alpha={alpha}): output differs from base at init"
            )


# ---------------------------------------------------------------------------
# 9. Integration: end-to-end training through jno.core
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLoraIntegration:
    def _solve(self, net):
        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = domain.variable("interior")
        u = net(x)
        loss = (u - jnn.sin(jnn.pi * x)).mse
        stats = jno.core([loss], domain).solve(3)
        return stats.training_logs[-1]["total_loss"][-1]

    def test_standard_lora_trains(self):
        net = nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=KEY))
        net.freeze().lora(rank=4, alpha=1.0).optimizer(optax.adam, lr=lrs(1e-3))
        assert jnp.isfinite(self._solve(net))

    @pytest.mark.parametrize("cls", [rsLoRALinear, LoRAFALinear, DoRALinear, PiSSALinear, LoRAXSLinear])
    def test_zoo_class_trains(self, cls):
        net = nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=KEY))
        net.freeze().lora(rank=4, alpha=1.0, wrapper=cls).optimizer(optax.adam, lr=lrs(1e-3))
        assert jnp.isfinite(self._solve(net))

    def test_custom_wrapper_trains(self):
        """A user-defined wrapper trains through jno.core without errors."""
        from jno.architectures.lora import LinearLike

        class LearnableBias(LoRAWrapper):
            adapter_fields = ("bias_delta",)
            base: eqx.Module
            bias_delta: jax.Array
            rank: int = eqx.field(static=True)
            alpha: float = eqx.field(static=True)

            @classmethod
            def applies_to(cls, leaf):
                return isinstance(leaf, LinearLike) and not isinstance(leaf, LoRAWrapper)

            def __init__(self, base, rank, alpha, *, key):
                self.base, self.rank, self.alpha = base, rank, alpha
                self.bias_delta = jnp.zeros(base.out_features)

            def __call__(self, x):
                return self.base(x) + self.bias_delta

            def merge(self):
                return self.base

        net = nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=KEY))
        net.freeze().lora(rank=4, alpha=1.0, wrapper=LearnableBias).optimizer(
            optax.adam, lr=lrs(1e-3)
        )
        assert jnp.isfinite(self._solve(net))
