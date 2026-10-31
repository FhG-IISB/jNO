"""``jno.utils.sets.unique_values`` is ``np.unique(a)`` by sorting. Oracle: NumPy's own, on every kind of array it takes,
with the extremes: empty, one value, all equal, all distinct, NaNs, multi-dimensional. And no bare ``np.unique`` in the
package: on the keys of a sparsity pattern it is 28x slower (see the module), and one such call took most of a build."""

import ast
import pathlib

import numpy as np
import pytest

import jno
from jno.utils.sets import unique_values

_RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "a",
    [
        np.zeros(0, np.int64),
        np.array([7], np.int32),
        np.full(1000, 3, np.int64),
        np.arange(1000, dtype=np.int64)[::-1],
        _RNG.integers(-50, 50, 10_000),
        _RNG.integers(0, 2**40, 10_000, dtype=np.int64),
        _RNG.integers(0, 200, 5_000, dtype=np.uint8),
        _RNG.integers(0, 50, (40, 30), dtype=np.int32),
        _RNG.integers(0, 2, 100).astype(bool),
        _RNG.normal(size=500).round(1),
        np.array([1.0, np.nan, 2.0, np.nan, 1.0, np.inf, -np.inf, -0.0, 0.0]),
        np.array([np.nan, np.nan]),
        np.array(["b", "a", "b"]),
        np.array([1 + 1j, 1 + 1j, 2j]),
    ],
)
def test_it_is_numpy_unique(a):
    got, want = unique_values(a), np.unique(a)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_the_input_is_left_alone():
    a = np.array([3, 1, 2, 1])
    keep = a.copy()
    unique_values(a)
    np.testing.assert_array_equal(a, keep)
    one = np.array([5])
    out = unique_values(one)
    out[0] = 6
    assert one[0] == 5


def test_a_list_and_a_jax_array_are_taken():
    import jax.numpy as jnp

    np.testing.assert_array_equal(unique_values([2, 1, 2]), [1, 2])
    np.testing.assert_array_equal(unique_values(jnp.asarray([2, 1, 2])), [1, 2])


def test_no_bare_np_unique_in_the_package():
    root = pathlib.Path(jno.__file__).parent
    found = []
    for path in root.rglob("*.py"):
        if path.name == "sets.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "unique"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("np", "_np", "numpy", "onp")
                and not ({k.arg for k in node.keywords} & {"return_index", "return_inverse", "return_counts", "axis"})
            ):
                found.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not found, f"use jno.utils.sets.unique_values: {found}"
