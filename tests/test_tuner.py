"""Tests for jno.tuner.ArchSpace — the parameter search space builder.

Full Tuner runs are slow and depend on nevergrad; this file covers the
search-space construction API which is what most users interact with.
"""

import pytest

from jno.tuner import Arch, ArchSpace, FloatGroup, IntGroup, UniqueGroup

# ---------------------------------------------------------------------------
# ArchSpace.unique / float_range / int_range
# ---------------------------------------------------------------------------


class TestArchSpaceBuilders:
    def test_empty_space_reports_is_empty(self):
        s = ArchSpace()
        assert s.is_empty()
        assert s.groups == []

    def test_unique_adds_a_unique_group(self):
        s = ArchSpace().unique("activation", ["relu", "tanh", "gelu"])
        assert not s.is_empty()
        assert len(s.groups) == 1
        g = s.groups[0]
        assert isinstance(g, UniqueGroup)
        assert g.name == "activation"
        assert g.options == ("relu", "tanh", "gelu")

    def test_float_range_adds_a_float_group(self):
        s = ArchSpace().float_range("learning_rate", 1e-5, 1e-2, log_scale=True)
        g = s.groups[0]
        assert isinstance(g, FloatGroup)
        assert g.low == pytest.approx(1e-5)
        assert g.high == pytest.approx(1e-2)
        assert g.log_scale is True

    def test_int_range_adds_an_int_group(self):
        s = ArchSpace().int_range("hidden_dims", 16, 64)
        g = s.groups[0]
        assert isinstance(g, IntGroup)
        assert g.low == 16
        assert g.high == 64

    def test_builder_methods_chain(self):
        s = (
            ArchSpace()
            .unique("activation", ["relu", "tanh"])
            .float_range("learning_rate", 1e-4, 1e-2, log_scale=True)
            .int_range("hidden_dims", 8, 32)
        )
        assert len(s.groups) == 3


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------


class TestCategoryInference:
    def test_known_training_param_defaults_to_training_category(self):
        s = ArchSpace().unique("learning_rate", [1e-3, 1e-4])
        # "learning_rate" is in TRAINING_PARAMS
        g = s.groups[0]
        assert g.category == "training"

    def test_unknown_param_defaults_to_architecture(self):
        s = ArchSpace().unique("activation", ["relu", "tanh"])
        g = s.groups[0]
        assert g.category == "architecture"

    def test_explicit_category_override(self):
        s = ArchSpace().unique("custom_thing", ["a", "b"], category="optimizer")
        g = s.groups[0]
        assert g.category == "optimizer"


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------


class TestGrid:
    def test_grid_size_unique_only(self):
        s = ArchSpace().unique("activation", ["relu", "tanh", "gelu"])
        assert s.grid_size() == 3

    def test_grid_size_combinatorial(self):
        s = ArchSpace().unique("a", [1, 2]).unique("b", [10, 20, 30])
        assert s.grid_size() == 6

    def test_grid_returns_arch_for_each_combination(self):
        s = ArchSpace().unique("a", [1, 2]).unique("b", [10, 20])
        grid = s.grid()
        assert len(grid) == 4
        for entry in grid:
            assert isinstance(entry, Arch)

    def test_grid_includes_int_range_endpoints(self):
        s = ArchSpace().int_range("n", 2, 4)
        grid = s.grid()
        names_values = [dict(arch.choices)["n"] for arch in grid]
        assert sorted(names_values) == [2, 3, 4]

    def test_empty_space_grid_is_singleton(self):
        s = ArchSpace()
        grid = s.grid()
        assert len(grid) == 1
        assert grid[0].choices == ()


# ---------------------------------------------------------------------------
# Group accessors
# ---------------------------------------------------------------------------


class TestGroupAccessors:
    def test_get_architecture_vs_training_groups(self):
        s = (
            ArchSpace()
            .unique("activation", ["relu", "tanh"])  # architecture
            .float_range("learning_rate", 1e-4, 1e-2)  # training
        )
        arch_groups = s.get_architecture_groups()
        train_groups = s.get_training_groups()
        assert len(arch_groups) == 1 and arch_groups[0].name == "activation"
        assert len(train_groups) == 1 and train_groups[0].name == "learning_rate"

    def test_has_helpers(self):
        s = ArchSpace().unique("activation", ["relu", "tanh"])
        assert s.has_architecture_params()
        assert not s.has_training_params()
