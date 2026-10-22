"""The block-wise sparsity plan is the raw-triplet plan, bit for bit, without building the raw triplets.

``compress_plan`` collapses the raw (row, col) triplets of every element block; the raw pattern of a 3-D
vector problem is ~280 triplets per DOF and materialising it was the build's host-memory peak (18 GB at
1.5M DOFs). ``compress_plan_blocks`` takes the blocks unexpanded. The oracle is ``compress_plan`` of the
expanded triplets -- an independent path (one argsort over everything) -- on patterns with duplicates
inside and across blocks, uneven batches, and plain triplet blocks mixed in.
"""

from __future__ import annotations

import numpy as np
import pytest

from jno.utils.solver import fem_utils as fu


def _blocks(seed):
    rng = np.random.default_rng(seed)
    n_dofs = 97
    vol = ("outer", rng.integers(0, n_dofs, size=(41, 6)), rng.integers(0, n_dofs, size=(41, 6)))
    surf = ("outer", rng.integers(0, n_dofs, size=(13, 3)), rng.integers(0, n_dofs, size=(13, 6)))
    flat = ("flat", rng.integers(0, n_dofs, size=29), rng.integers(0, n_dofs, size=29))
    return [vol, flat, surf]


@pytest.mark.parametrize("batch", [1, 7, 50, 1 << 20], ids=["batch-1", "batch-7", "batch-50", "one-batch"])
def test_block_plan_equals_the_raw_triplet_plan(batch, monkeypatch):
    # Break: a block expanded in the wrong (cell, row, col) order, a batch boundary dropping or repeating
    # keys, a merge losing pending uniques, or an inverse written at the wrong offset.
    monkeypatch.setattr(fu, "_PLAN_BATCH", batch)
    fu._PLAN_CACHE.clear()
    blocks = _blocks(0)
    raw = np.asarray(fu.raw_pattern_from_blocks(blocks))
    assert raw.shape[0] == sum(fu._block_len(b) for b in blocks)
    ref_idx, ref_inv, ref_nse = fu._collapse_coo(raw)
    idx, inv, nse = fu.compress_plan_blocks(blocks)
    assert nse == ref_nse < raw.shape[0]  # duplicates were collapsed
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_array_equal(inv, ref_inv)
    assert inv.dtype == np.int32 and idx.dtype == np.int32
    np.testing.assert_array_equal(idx[inv], raw)  # every raw triplet lands on its own (row, col)


def test_block_plan_is_memoized_on_content():
    fu._PLAN_CACHE.clear()
    a = fu.compress_plan_blocks(_blocks(1))
    assert fu.compress_plan_blocks(_blocks(1)) is a
    assert fu.compress_plan_blocks(_blocks(2)) is not a
