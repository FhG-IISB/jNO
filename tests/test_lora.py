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
  - Model control state combinations: mask/freeze/lora/reset chaining
  - Integration: end-to-end training through jno.core, including complex combinations
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
    VeRALinear,
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
    VeRALinear:   6,   # 3 × (b, d)
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


class TestVeRAInvariants:
    def test_b_initialized_to_zero(self):
        lin = eqx.nn.Linear(4, 8, key=KEY)
        adapted = VeRALinear(lin, rank=2, alpha=1.0, key=KEY)
        assert jnp.all(adapted.b == 0)

    def test_d_initialized_to_ones(self):
        lin = eqx.nn.Linear(4, 8, key=KEY)
        adapted = VeRALinear(lin, rank=2, alpha=1.0, key=KEY)
        assert jnp.all(adapted.d == 1)

    def test_adapter_fields(self):
        assert VeRALinear.adapter_fields == ("b", "d")

    def test_no_AB_arrays_in_pytree(self):
        """A and B must not appear as JAX arrays in the pytree."""
        lin = eqx.nn.Linear(4, 8, key=KEY)
        adapted = VeRALinear(lin, rank=2, alpha=1.0, key=KEY)
        shapes = {leaf.shape for leaf in jax.tree_util.tree_leaves(eqx.filter(adapted, eqx.is_array))}
        # A would be (rank, in_features) = (2, 4); B would be (out_features, rank) = (8, 2)
        assert (2, 4) not in shapes, "lora_A-shaped array found in pytree — A should be XLA-only"
        assert (8, 2) not in shapes, "lora_B-shaped array found in pytree — B should be XLA-only"

    def test_fewer_trainable_params_than_lora(self):
        """b+d total size must be less than lora_A+lora_B total size for same rank."""
        mlp = _mlp()

        def _adapter_param_size(model):
            filt = lora_trainable_filter(model)
            flat_arrays = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
            flat_filt = jax.tree_util.tree_leaves(filt)
            return sum(a.size for a, f in zip(flat_arrays, flat_filt) if f)

        n_lora = _adapter_param_size(apply_lora(mlp, rank=4, alpha=1.0, key=KEY, wrappers=(LoRALinear,)))
        n_vera = _adapter_param_size(apply_lora(mlp, rank=4, alpha=1.0, key=KEY, wrappers=(VeRALinear,)))
        assert n_vera < n_lora, f"VeRA ({n_vera}) should have fewer params than LoRA ({n_lora})"

    def test_same_seed_same_output(self):
        """Two VeRALinear instances with the same seed must produce identical output."""
        lin = eqx.nn.Linear(4, 8, key=KEY)
        x = jax.random.normal(KEY, (4,))

        v1 = VeRALinear(lin, rank=2, alpha=1.0, key=KEY)
        v2 = VeRALinear(lin, rank=2, alpha=1.0, key=KEY)
        # Same key → same seed → same A, B → same output given same b, d, base
        assert v1.seed == v2.seed
        assert jnp.allclose(v1(x), v2(x))

    def test_different_keys_different_seeds(self):
        """Different keys should (with overwhelming probability) produce different seeds."""
        lin = eqx.nn.Linear(4, 8, key=KEY)
        k1, k2 = jax.random.split(KEY)
        v1 = VeRALinear(lin, rank=2, alpha=1.0, key=k1)
        v2 = VeRALinear(lin, rank=2, alpha=1.0, key=k2)
        assert v1.seed != v2.seed


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

    @pytest.mark.parametrize("cls", [rsLoRALinear, LoRAFALinear, DoRALinear, PiSSALinear, LoRAXSLinear, VeRALinear])
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


# ---------------------------------------------------------------------------
# 10. Model control state combinations (structural — no jno.core required)
# ---------------------------------------------------------------------------


class TestModelControlStateCombinations:
    """Verify Model._* state fields after chaining model-control methods."""

    def _net(self):
        return nn.wrap(foundax.mlp(**_MLP_KW, key=KEY))

    def _all_true_mask(self):
        m = foundax.mlp(**_MLP_KW, key=KEY)
        return jax.tree_util.tree_map(lambda _: True, eqx.filter(m, eqx.is_array))

    def test_mask_freeze_sets_trainable_mask_not_frozen_flag(self):
        """mask(m).freeze() sets _trainable_param_mask and leaves _frozen=False."""
        net = self._net()
        net.mask(self._all_true_mask()).freeze()
        assert net._frozen is False
        assert net._trainable_param_mask is not None

    def test_global_freeze_sets_frozen_flag_clears_mask(self):
        """freeze() with no preceding mask sets _frozen=True, _trainable_param_mask=None."""
        net = self._net()
        net.freeze()
        assert net._frozen is True
        assert net._trainable_param_mask is None

    def test_mask_lora_sets_both_trainable_mask_and_config(self):
        """mask(m).lora() stores _trainable_param_mask and _lora_config simultaneously."""
        net = self._net()
        net.mask(self._all_true_mask()).lora(rank=4, alpha=1.0)
        assert net._trainable_param_mask is not None
        assert net._lora_config is not None

    def test_freeze_then_lora_leaves_no_trainable_mask(self):
        """freeze().lora() produces _frozen=True, _lora_config set, _trainable_param_mask=None.

        freeze() consumes the (absent) mask scope and leaves _mask_scope_pending=False,
        so the subsequent lora() call sees no pending mask.
        """
        net = self._net()
        net.freeze().lora(rank=4, alpha=1.0)
        assert net._frozen is True
        assert net._lora_config is not None
        assert net._trainable_param_mask is None

    def test_mask_freeze_then_lora_mask_already_consumed(self):
        """mask(m).freeze() followed by .lora() — mask scope consumed by freeze."""
        net = self._net()
        net.mask(self._all_true_mask()).freeze()   # mask scope consumed here
        net.lora(rank=4, alpha=1.0)               # no pending mask
        assert net._frozen is False               # mask.freeze(), not global
        assert net._lora_config is not None
        assert net._trainable_param_mask is None  # lora() cleared it (no pending mask)

    def test_mask_scope_consumed_after_freeze(self):
        """_mask_scope_pending is False after freeze() regardless of preceding mask()."""
        net = self._net()
        net.mask(self._all_true_mask())
        assert net._mask_scope_pending is True
        net.freeze()
        assert net._mask_scope_pending is False

    def test_reset_clears_all_controls(self):
        """reset() zeroes every training-time control field."""
        net = self._net()
        net.mask(self._all_true_mask()).freeze()
        net.lora(rank=4, alpha=1.0)
        net.optimizer(optax.adam, lr=lrs(1e-3))
        net.dtype(jnp.bfloat16)
        net.reset()
        assert net._frozen is False
        assert net._lora_config is None
        assert net._trainable_param_mask is None
        assert net._opt_fn is None
        assert net._lr is None
        assert net._dtype is None

    def test_second_lora_overrides_first(self):
        """Calling .lora() twice replaces the earlier config."""
        net = self._net()
        net.lora(rank=4, alpha=1.0)
        net.lora(rank=8, alpha=2.0)
        assert net._lora_config[0]["rank"] == 8
        assert net._lora_config[0]["alpha"] == 2.0

    def test_unfreeze_clears_frozen_and_trainable_mask(self):
        """unfreeze() clears both _frozen and _trainable_param_mask."""
        net = self._net()
        net.mask(self._all_true_mask()).freeze()
        assert net._trainable_param_mask is not None
        net.unfreeze()
        assert net._frozen is False
        assert net._trainable_param_mask is None

    def test_masked_optimizer_adds_param_group(self):
        """mask(m).optimizer(...) registers a parameter group."""
        net = self._net()
        net.mask(self._all_true_mask()).optimizer(optax.adam, lr=lrs(1e-3))
        assert len(net._param_groups) >= 1

    def test_global_optimizer_clears_param_groups(self):
        """A bare (non-masked) optimizer() call discards existing param groups."""
        net = self._net()
        net.mask(self._all_true_mask()).optimizer(optax.adam, lr=lrs(1e-3))
        assert len(net._param_groups) >= 1
        net.optimizer(optax.adamw, lr=lrs(5e-4))
        assert len(net._param_groups) == 0

    def test_freeze_lora_chainable_returns_self(self):
        """All model-control methods return self to support chaining."""
        net = self._net()
        result = net.freeze().lora(rank=4, alpha=1.0).optimizer(optax.adam, lr=lrs(1e-3))
        assert result is net

    def test_freeze_and_lora_and_dtype_all_set(self):
        """dtype, freeze, and lora config can coexist on the same model."""
        net = self._net()
        net.dtype(jnp.bfloat16).freeze().lora(rank=4, alpha=1.0)
        assert net._dtype == jnp.bfloat16
        assert net._frozen is True
        assert net._lora_config is not None


# ---------------------------------------------------------------------------
# 11. Integration: complex model-control combinations through jno.core
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegrationCombinations:
    """End-to-end solve() calls exercising complex combinations of model controls."""

    def _domain_and_x(self):
        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = domain.variable("interior")
        return domain, x

    def _net(self, key=KEY):
        return nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=key))

    def _all_true_mask(self, m):
        return jax.tree_util.tree_map(lambda _: True, eqx.filter(m, eqx.is_array))

    def _solve_single(self, net):
        domain, x = self._domain_and_x()
        loss = (net(x) - jnn.sin(jnn.pi * x)).mse
        stats = jno.core([loss], domain).solve(3)
        return stats.training_logs[-1]["total_loss"][-1]

    # ── lora without freeze: base + adapters both update ─────────────────────

    def test_lora_without_freeze_trains(self):
        """lora() alone keeps base params trainable; adapters and base both update."""
        net = self._net()
        net.lora(rank=4, alpha=1.0).optimizer(optax.adam, lr=lrs(1e-3))
        assert jnp.isfinite(self._solve_single(net))

    # ── two models: frozen feature extractor + LoRA adapter ──────────────────

    def test_frozen_feature_plus_lora_model(self):
        """Frozen model contributes to the forward pass; only LoRA adapter trains."""
        k1, k2 = jax.random.split(KEY)
        feat = self._net(key=k1)
        feat.freeze()

        adapter = self._net(key=k2)
        adapter.freeze().lora(rank=4, alpha=1.0).optimizer(optax.adam, lr=lrs(1e-3))

        domain, x = self._domain_and_x()
        loss = (feat(x) + adapter(x) - jnn.sin(jnn.pi * x)).mse
        stats = jno.core([loss], domain).solve(3)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    # ── two LoRA models with different wrappers in one solve ─────────────────

    def test_two_lora_models_different_wrappers(self):
        k1, k2 = jax.random.split(KEY)
        net1 = self._net(key=k1)
        net1.freeze().lora(rank=4, alpha=1.0, wrapper=rsLoRALinear).optimizer(
            optax.adam, lr=lrs(1e-3)
        )
        net2 = self._net(key=k2)
        net2.freeze().lora(rank=4, alpha=1.0, wrapper=LoRAFALinear).optimizer(
            optax.adam, lr=lrs(1e-3)
        )
        domain, x = self._domain_and_x()
        loss = (net1(x) + net2(x) - jnn.sin(jnn.pi * x)).mse
        stats = jno.core([loss], domain).solve(3)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    # ── partial freeze via mask (no LoRA) ─────────────────────────────────────

    def test_partial_freeze_no_lora(self):
        """mask(m).freeze() freezes selected leaves; the remaining leaves train."""
        m = foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=KEY)
        net = nn.wrap(m)
        leaves, treedef = jax.tree_util.tree_flatten(eqx.filter(m, eqx.is_array))
        # Freeze the first half, leave the second half trainable.
        half = len(leaves) // 2
        partial_mask = jax.tree_util.tree_unflatten(
            treedef, [i < half for i in range(len(leaves))]
        )
        net.mask(partial_mask).freeze()
        net.optimizer(optax.adam, lr=lrs(1e-3))
        assert jnp.isfinite(self._solve_single(net))

    # ── lora(target=...) partial coverage ────────────────────────────────────

    def test_lora_target_regex_partial_coverage(self):
        """lora(target='0') adapts only the first layer; model still converges."""
        net = self._net()
        net.freeze().lora(rank=4, alpha=1.0, target="0").optimizer(optax.adam, lr=lrs(1e-3))
        assert jnp.isfinite(self._solve_single(net))

    # ── multi-spec: different ranks per layer group ───────────────────────────

    def test_lora_multi_spec_different_ranks(self):
        net = self._net()
        net.freeze().lora(specs=[
            {"target": "0",  "rank": 2, "alpha": 1.0},
            {"target": ".*", "rank": 8, "alpha": 2.0},
        ]).optimizer(optax.adam, lr=lrs(1e-3))
        assert jnp.isfinite(self._solve_single(net))

    # ── multi-spec: different wrappers per layer group ────────────────────────

    def test_lora_multi_spec_different_wrappers(self):
        net = self._net()
        net.freeze().lora(specs=[
            {"target": "0",  "rank": 4, "alpha": 1.0, "wrapper": rsLoRALinear},
            {"target": ".*", "rank": 4, "alpha": 1.0, "wrapper": DoRALinear},
        ]).optimizer(optax.adam, lr=lrs(1e-3))
        assert jnp.isfinite(self._solve_single(net))

    # ── mask(m).lora(): adapters train, no crash ──────────────────────────────

    def test_mask_lora_no_crash_adapters_train(self):
        """mask(m).lora() stores _trainable_param_mask but core uses lora_trainable_filter;
        only adapter arrays train regardless of the mask — must not crash."""
        m = foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=KEY)
        net = nn.wrap(m)
        leaves, treedef = jax.tree_util.tree_flatten(eqx.filter(m, eqx.is_array))
        half_mask = jax.tree_util.tree_unflatten(
            treedef, [i < len(leaves) // 2 for i in range(len(leaves))]
        )
        net.mask(half_mask).lora(rank=4, alpha=1.0).optimizer(optax.adam, lr=lrs(1e-3))
        assert jnp.isfinite(self._solve_single(net))

    # ── dtype(bfloat16) + freeze + lora ─────────────────────────────────────

    def test_dtype_bfloat16_with_freeze_and_lora(self):
        net = self._net()
        net.dtype(jnp.bfloat16).freeze().lora(rank=4, alpha=1.0).optimizer(
            optax.adam, lr=lrs(1e-3)
        )
        assert jnp.isfinite(self._solve_single(net))

    # ── initialize from file + freeze + lora (transfer learning) ─────────────

    def test_initialize_then_freeze_lora(self, tmp_path):
        """Canonical transfer-learning recipe: load pretrained weights, then LoRA-finetune."""
        m = foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=KEY)
        path = str(tmp_path / "weights.eqx")
        eqx.tree_serialise_leaves(path, m)

        net = nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=KEY))
        net.initialize(path).freeze().lora(rank=4, alpha=1.0).optimizer(
            optax.adam, lr=lrs(1e-3)
        )
        assert jnp.isfinite(self._solve_single(net))

    # ── two models with separate LoRA wrappers + separate losses ─────────────

    def test_two_lora_models_separate_losses(self):
        """Two independent LoRA models with different wrappers trained jointly."""
        k1, k2 = jax.random.split(KEY)
        net1 = self._net(key=k1)
        net1.freeze().lora(rank=4, alpha=1.0, wrapper=LoRALinear).optimizer(
            optax.adam, lr=lrs(1e-3)
        )
        net2 = self._net(key=k2)
        net2.freeze().lora(rank=4, alpha=1.0, wrapper=VeRALinear).optimizer(
            optax.adamw, lr=lrs(5e-4)
        )
        domain, x = self._domain_and_x()
        loss1 = (net1(x) - jnn.sin(jnn.pi * x)).mse
        loss2 = (net2(x) - jnn.cos(jnn.pi * x)).mse
        stats = jno.core([loss1, loss2], domain).solve(3)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    # ── reset then re-configure ───────────────────────────────────────────────

    def test_reset_and_reconfigure(self):
        """Calling reset() and then re-applying controls produces valid training."""
        net = self._net()
        net.freeze().lora(rank=4, alpha=1.0).optimizer(optax.adam, lr=lrs(1e-3))
        net.reset()
        # Re-configure with a different wrapper
        net.freeze().lora(rank=8, alpha=2.0, wrapper=rsLoRALinear).optimizer(
            optax.adamw, lr=lrs(5e-4)
        )
        assert jnp.isfinite(self._solve_single(net))

    # ── three models: frozen backbone, LoRA adapter, trainable head ───────────

    def test_three_model_pipeline(self):
        """Frozen backbone → LoRA adapter → trainable head: a realistic fine-tune setup."""
        k1, k2, k3 = jax.random.split(KEY, 3)
        backbone = self._net(key=k1)
        backbone.freeze()

        adapter = self._net(key=k2)
        adapter.freeze().lora(rank=4, alpha=1.0).optimizer(optax.adam, lr=lrs(1e-3))

        head = self._net(key=k3)
        head.optimizer(optax.adam, lr=lrs(5e-3))

        domain, x = self._domain_and_x()
        u = backbone(x) + adapter(x) + head(x)
        loss = (u - jnn.sin(jnn.pi * x)).mse
        stats = jno.core([loss], domain).solve(3)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])
