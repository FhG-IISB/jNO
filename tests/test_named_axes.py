"""``axis=`` on a reduction accepts a coordinate ``Variable``, not just an integer.

An integer axis is a guess against the undocumented ``(B, T, N, D)`` context layout. Passing the
coordinate itself is checkable, and — crucially — order-independent, which is what proves the
resolution is derived from the Variable rather than from argument position.

Resolution happens inside the reduction's closure, at the point the concrete array exists, because a
trace expression carries no shape (``Placeholder.shape`` is itself a traced node). These tests
therefore exercise ``_resolve_axes`` against real shapes directly, plus the end-to-end path.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.jnp_ops import _axis_extent, _resolve_axes, _resolve_axis
from jno.trace import Variable
from tests.conftest import MockDomain


def _grid_domain(nx, ny, nz=None):
    """A MockDomain carrying a structured-grid extent and an (N, D) coordinate tag."""
    d = MockDomain()
    dim = 2 if nz is None else 3
    d.context["xy"] = jnp.zeros((nx * ny * (nz or 1), dim))
    d._grid_shape = (nx, ny) if nz is None else (nx, ny, nz)
    return d


def _coords(d):
    n = len(d._grid_shape)
    return tuple(Variable("xy", [i, i + 1], domain=d) for i in range(n))


class TestAxisExtent:
    def test_spatial_extent_comes_from_the_grid_via_dim0(self):
        d = _grid_domain(8, 5)
        x, y = _coords(d)
        assert _axis_extent(x) == 8
        assert _axis_extent(y) == 5

    def test_unstructured_domain_raises_with_the_fix_in_the_message(self):
        d = MockDomain(tags=["interior"])
        v = Variable("interior", [0, 1], domain=d)
        with pytest.raises(ValueError, match="axis=None"):
            _axis_extent(v)

    def test_temporal_extent_needs_a_time_dependent_domain(self):
        d = _grid_domain(4, 4)
        t = Variable("xy", [0, 1], domain=d, axis="temporal")
        with pytest.raises(ValueError, match="not time-dependent"):
            _axis_extent(t)


class TestResolveAxis:
    def test_non_square_grid_matches_directly(self):
        d = _grid_domain(8, 5)
        x, y = _coords(d)
        assert _resolve_axis((1, 8, 5, 1), x) == 1
        assert _resolve_axis((1, 8, 5, 1), y) == 2

    def test_square_grid_resolves_via_the_block_tie_break(self):
        """The flagship case — the FNO tutorial grid is square, so per-axis matching is ambiguous."""
        d = _grid_domain(128, 128)
        x, y = _coords(d)
        assert _resolve_axis((1, 128, 128, 1), x) == 1
        assert _resolve_axis((1, 128, 128, 1), y) == 2

    def test_genuinely_ambiguous_block_fails_loud_naming_both(self):
        d = _grid_domain(128, 128)
        x, _ = _coords(d)
        with pytest.raises(ValueError, match="ambiguous"):
            _resolve_axis((128, 128, 128), x)

    def test_extent_present_nowhere_raises(self):
        d = _grid_domain(128, 128)
        x, _ = _coords(d)
        with pytest.raises(ValueError, match="no axis of an array with shape"):
            _resolve_axis((1, 64, 64, 1), x)

    def test_three_dimensional_grid(self):
        d = _grid_domain(4, 6, 8)
        x, y, z = _coords(d)
        assert (_resolve_axis((4, 6, 8), x), _resolve_axis((4, 6, 8), y), _resolve_axis((4, 6, 8), z)) == (0, 1, 2)

    def test_degenerate_singleton_axis_still_resolves_when_the_block_is_unique(self):
        """A grid of extent 1 collides with every singleton axis, so per-axis matching gives three
        candidates — but the block ``(1, 5)`` occurs only at offset 1, so it still resolves. The
        tie-break earns its keep here; picking the first candidate would have given axis 0."""
        d = _grid_domain(1, 5)
        x, y = _coords(d)
        assert _resolve_axis((1, 1, 5, 1), x) == 1
        assert _resolve_axis((1, 1, 5, 1), y) == 2

    def test_repeated_block_is_ambiguous_and_raises(self):
        """When the grid block itself occurs twice there is nothing left to disambiguate with."""
        d = _grid_domain(1, 5)
        x, _ = _coords(d)
        with pytest.raises(ValueError, match="ambiguous"):
            _resolve_axis((1, 5, 1, 5), x)


class TestResolveAxes:
    def test_order_independent(self):
        """The proof that resolution is by Variable, not by argument position."""
        d = _grid_domain(8, 5)
        x, y = _coords(d)
        assert _resolve_axes((1, 8, 5, 1), (x, y)) == (1, 2)
        assert _resolve_axes((1, 8, 5, 1), (y, x)) == (2, 1)

    def test_ints_and_none_pass_through_unchanged(self):
        assert _resolve_axes((1, 8, 5, 1), None) is None
        assert _resolve_axes((1, 8, 5, 1), 0) == 0
        assert _resolve_axes((1, 8, 5, 1), -1) == -1
        assert _resolve_axes((1, 8, 5, 1), (0, 1)) == (0, 1)

    def test_mixed_variable_and_int(self):
        d = _grid_domain(8, 5)
        x, _ = _coords(d)
        assert _resolve_axes((1, 8, 5, 1), (x, 3)) == (1, 3)

    def test_duplicate_axis_raises(self):
        d = _grid_domain(8, 5)
        x, _ = _coords(d)
        with pytest.raises(ValueError, match="duplicate axes"):
            _resolve_axes((1, 8, 5, 1), (x, 1))


class TestEndToEnd:
    def _built(self, nx, ny):
        d = _grid_domain(nx, ny)
        x, y = _coords(d)
        return d, x, y

    def test_named_axes_match_the_integer_form_numerically(self):
        d, x, y = self._built(8, 5)
        arr = np.arange(1 * 8 * 5 * 1, dtype=np.float32).reshape(1, 8, 5, 1)
        v = Variable("xy", [0, 2], domain=d)
        node_named = jno.np.mean(v, axis=(x, y))
        node_int = jno.np.mean(v, axis=(1, 2))
        got = node_named.fn(jnp.asarray(arr))
        want = node_int.fn(jnp.asarray(arr))
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(got), arr.mean(axis=(1, 2)), rtol=1e-6)

    def test_the_unresolved_axis_is_what_is_stored(self):
        """`reduces_axis` keeps what the user wrote — the deferred outer-axis work reads it."""
        d, x, y = self._built(8, 5)
        node = jno.np.mean(Variable("xy", [0, 2], domain=d), axis=(x, y))
        assert node.reduces_axis == (x, y)
        assert node.reduces

    def test_norm_takes_a_named_axis_too(self):
        d, x, _ = self._built(8, 5)
        arr = jnp.asarray(np.random.default_rng(0).normal(size=(1, 8, 5, 1)).astype(np.float32))
        node = jno.np.norm(Variable("xy", [0, 2], domain=d), axis=x)
        np.testing.assert_allclose(np.asarray(node.fn(arr)), np.asarray(jnp.linalg.norm(arr, axis=1)), rtol=1e-6)

    def test_zero_size_and_singleton_extremes(self):
        d = _grid_domain(0, 5)
        x, _ = _coords(d)
        assert _resolve_axis((1, 0, 5, 1), x) == 1
