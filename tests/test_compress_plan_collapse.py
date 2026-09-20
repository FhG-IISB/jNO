"""The host-side COO collapse must be exact, and must stay lean.

``compress_plan`` turns the raw element-block triplet list into ``(unique_indices, inverse, nse)``.
It used to spell that as ``np.unique(rows * stride + cols, return_inverse=True)`` over two int64
copies of the index columns. That is correct but allocates an ``intp`` permutation, an int64 sorted
copy AND an int64 inverse on top of the int64 key -- ~73 bytes of transient per RAW triplet to
produce a 4-byte-per-triplet answer. The raw triplet count is ``n_cells * dofs_per_cell**2`` summed
over terms, which on a 3-D operator runs to 10^8, so that scratch, not the operator, set the build's
peak memory: measured 15.9 GB for a 3-D melt pool whose assembled tangent is 98 MB.

``_collapse_coo`` computes the SAME plan with the key, its permutation and the sorted copy as the
only full-length int64 arrays. The oracle here is the spelling it replaced -- these tests pin exact
equality against it, because "uses less memory" is worthless if the pattern moves by one entry.
"""

import numpy as np
import pytest

from jno.utils.solver.fem_utils import _collapse_coo


def numpy_unique_reference(arr):
    """The pre-existing spelling, verbatim. The oracle."""
    rows = arr[:, 0].astype(np.int64)
    cols = arr[:, 1].astype(np.int64)
    stride = int(cols.max()) + 1
    uniq, inverse = np.unique(rows * stride + cols, return_inverse=True)
    idx = np.stack([uniq // stride, uniq % stride], axis=1).astype(np.int32)
    return idx, inverse.reshape(-1).astype(np.int32), int(uniq.shape[0])


def _case(name):
    """Patterns a real assembly produces, plus the degenerate ones it produces at the edges."""
    rng = np.random.default_rng(0)
    return {
        # a single-element mesh, and a pattern with nothing to collapse
        "one_triplet": np.array([[3, 4]], np.int32),
        "no_duplicates": np.stack([np.arange(5000), np.arange(5000)], 1).astype(np.int32),
        # every triplet on one entry: a one-dof block, or a coupling term with a single test function
        "all_duplicates": np.zeros((1000, 2), np.int32),
        # the ordinary case: many cells contributing to comparatively few dofs
        "heavy_duplication": rng.integers(0, 30, (200_000, 2)).astype(np.int32),
        "sparse_wide": rng.integers(0, 100_000, (200_000, 2)).astype(np.int32),
        # a rectangular block -- a coupling term whose test and trial spaces differ in size
        "rectangular": np.stack([rng.integers(0, 7, 50_000), rng.integers(0, 90_000, 50_000)], 1).astype(np.int32),
        # a degenerate axis: a scalar constraint row, or a single-dof gauge column
        "single_column": np.stack([rng.integers(0, 5000, 20_000), np.zeros(20_000)], 1).astype(np.int32),
        "single_row": np.stack([np.zeros(20_000), rng.integers(0, 5000, 20_000)], 1).astype(np.int32),
        # int64 input: what the dof maps are under x64 before the int32 cast landed upstream
        "int64_input": rng.integers(0, 1000, (50_000, 2)).astype(np.int64),
    }[name]


CASES = [
    "one_triplet",
    "no_duplicates",
    "all_duplicates",
    "heavy_duplication",
    "sparse_wide",
    "rectangular",
    "single_column",
    "single_row",
    "int64_input",
]


@pytest.mark.parametrize("name", CASES)
def test_the_collapse_is_bit_identical_to_the_spelling_it_replaced(name):
    """Exact equality, not closeness: this is an index pattern, so one differing entry is a wrong
    operator, and the inverse is consumed by ``segment_sum`` where an off-by-one silently sums the
    wrong contributions instead of raising."""
    arr = _case(name)
    idx, inv, nse = _collapse_coo(arr)
    ref_idx, ref_inv, ref_nse = numpy_unique_reference(arr)
    assert nse == ref_nse, f"{name}: nse {nse} against the reference's {ref_nse}"
    assert np.array_equal(idx, ref_idx), f"{name}: unique indices differ"
    assert np.array_equal(inv, ref_inv), f"{name}: inverse differs"


@pytest.mark.parametrize("name", CASES)
def test_the_plan_reconstructs_every_input_triplet(name):
    """The property the assembly actually relies on, stated independently of the reference: scattering
    the unique rows back through the inverse must return the input. Guards against both arrays being
    permuted consistently-but-wrongly, which equality against a reference computed the same way could
    in principle miss."""
    arr = _case(name)
    idx, inv, _ = _collapse_coo(arr)
    assert np.array_equal(idx[inv], arr[:, :2].astype(np.int32))


@pytest.mark.parametrize("name", ["heavy_duplication", "int64_input"])
def test_the_plan_is_int32_whatever_the_input_width(name):
    """The inverse is one entry per RAW triplet -- the largest array the plan holds -- so its width is
    load-bearing, and an int64 input must not widen it."""
    idx, inv, _ = _collapse_coo(_case(name))
    assert idx.dtype == np.int32, idx.dtype
    assert inv.dtype == np.int32, inv.dtype


def test_the_collapse_allocates_far_less_than_the_spelling_it_replaced():
    """The point of the change, pinned as a MECHANISM rather than as an RSS number.

    tracemalloc sees every numpy allocation here (this path is pure host numpy), so the measurement is
    not at the mercy of the allocator returning pages. Measured, flat across a 10x range in n:
    reference 73.0 B per raw triplet, ``_collapse_coo`` 29.0 -- the latter being key(8) + permutation(8)
    + sorted copy(8) + inverse(4) + flag(1). The bound below sits clear of both so a numpy version that
    shaves a temporary cannot fail it, while any regression that reintroduces a full-length int64
    intermediate (+8 B/triplet minimum) still trips it.
    """
    import tracemalloc

    n = 2_000_000
    arr = np.random.default_rng(1).integers(0, 50_000, (n, 2)).astype(np.int32)

    peaks = {}
    for label, fn in (("reference", numpy_unique_reference), ("collapse", _collapse_coo)):
        tracemalloc.start()
        result = fn(arr)
        peaks[label] = tracemalloc.get_traced_memory()[1] / n
        tracemalloc.stop()
        del result

    assert peaks["collapse"] < 45.0, (
        f"{peaks['collapse']:.1f} B per raw triplet -- a full-length int64 intermediate is back"
    )
    assert peaks["collapse"] < 0.6 * peaks["reference"], (
        f"collapse {peaks['collapse']:.1f} B/triplet against the reference's "
        f"{peaks['reference']:.1f} -- the saving has gone"
    )
