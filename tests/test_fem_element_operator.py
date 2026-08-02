"""Element-blocked (EBE) operator storage — the same operator, without the assembled sparse matrix.

A ``BCOO`` keeps one ``(row, col)`` int32 pair per nonzero, and the assemblers emit one full triplet
block **per additive weak-form term** with duplicates left unsummed. An ``ElementOperator`` keeps one
dense block per cell plus the connectivity, so the index cost falls by ``n_local`` and the per-term
duplication collapses. Memory is the point: it is the 8 GB ceiling that decides which 3-D problems fit.

These tests pin the operator against a dense oracle rather than against the BCOO, so a bug in the
scatter and a matching bug in ``assemble()`` cannot cancel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.element_operator import ElementGroup, ElementOperator


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _random_problem(n_cells=500, n_local=4, n=200, n_terms=3, seed=0):
    """``n_terms`` element-block groups sharing one connectivity — the shape a 3-D Poisson emits
    (one block per gradient term, all indexing the same cell DOFs)."""
    rng = np.random.default_rng(seed)
    conn = rng.integers(0, n, size=(n_cells, n_local)).astype(np.int32)
    blocks = [jnp.asarray(rng.standard_normal((n_cells, n_local, n_local))) for _ in range(n_terms)]
    dense = np.zeros((n, n))
    for B in blocks:
        Bn = np.asarray(B)
        for c in range(n_cells):
            for i in range(n_local):
                for j in range(n_local):
                    dense[conn[c, i], conn[c, j]] += Bn[c, i, j]
    op = ElementOperator([ElementGroup(B, conn, conn) for B in blocks], (n, n))
    return op, dense, rng


def test_matvec_transpose_and_diagonal_match_a_dense_oracle():
    """The core contract. Checked against an independently accumulated dense matrix, so an error in
    the element scatter cannot be hidden by the same error in ``assemble()``.

    The transpose matters on its own: the assembled path's ``LinearOperator.T.bcoo`` ignores the
    transpose flag, whereas here ``T`` genuinely swaps each block's row and column maps — so a
    non-symmetric adjoint solve gets the operator it asked for."""
    op, dense, rng = _random_problem()
    v = jnp.asarray(rng.standard_normal(dense.shape[0]))

    assert np.max(np.abs(np.asarray(op.mv(v)) - dense @ np.asarray(v))) < 1e-12
    assert np.max(np.abs(np.asarray(op.T.mv(v)) - dense.T @ np.asarray(v))) < 1e-12
    assert np.max(np.abs(np.asarray(op.diag()) - np.diag(dense))) < 1e-12
    assert op.shape == dense.shape and op.T.shape == dense.shape


def test_same_pattern_terms_are_summed_into_one_block():
    """Where the memory win comes from. Three additive terms over one connectivity are three
    separate triplet blocks in the assembled path (measured on a real 3-D Poisson:
    ``nnz/(cells*n_local**2)`` = 2.02, 3.02, 4.02 for one, two, three gradient terms) and **one**
    block here. Summing is exact — it is what BCOO does lazily at matvec time anyway."""
    op1, dense1, _ = _random_problem(n_terms=1)
    op3, dense3, _ = _random_problem(n_terms=3)

    assert len(op1._groups) == 1 and len(op3._groups) == 1, "same-connectivity terms must merge"
    assert op3.nbytes == op1.nbytes, "merging means term count must not change the footprint"
    # ...and merging did not corrupt the values
    v = jnp.asarray(np.random.default_rng(1).standard_normal(dense3.shape[0]))
    assert np.max(np.abs(np.asarray(op3.mv(v)) - dense3 @ np.asarray(v))) < 1e-12


def test_distinct_connectivities_are_not_merged():
    """The merge keys on the actual index arrays, not on term provenance — a multifield form whose
    fields genuinely index differently must keep its groups separate or the operator is wrong."""
    rng = np.random.default_rng(2)
    n, n_cells, n_local = 60, 40, 4
    c1 = rng.integers(0, n, size=(n_cells, n_local)).astype(np.int32)
    c2 = rng.integers(0, n, size=(n_cells, n_local)).astype(np.int32)
    B1 = jnp.asarray(rng.standard_normal((n_cells, n_local, n_local)))
    B2 = jnp.asarray(rng.standard_normal((n_cells, n_local, n_local)))
    op = ElementOperator([ElementGroup(B1, c1, c1), ElementGroup(B2, c2, c2)], (n, n))
    assert len(op._groups) == 2, "different connectivities must not be merged"

    dense = np.zeros((n, n))
    for B, cc in ((np.asarray(B1), c1), (np.asarray(B2), c2)):
        for c in range(n_cells):
            for i in range(n_local):
                for j in range(n_local):
                    dense[cc[c, i], cc[c, j]] += B[c, i, j]
    v = jnp.asarray(rng.standard_normal(n))
    assert np.max(np.abs(np.asarray(op.mv(v)) - dense @ np.asarray(v))) < 1e-12


def test_no_assembled_matrix_is_exposed_or_built_implicitly():
    """``bcoo`` is ``None`` so sparse-direct solvers and matrix-based preconditioners hit their
    existing targeted refusals. ``assemble()`` is the explicit escape hatch — never implicit,
    because silently allocating the larger matrix is precisely what this representation avoids."""
    op, dense, _ = _random_problem()
    assert op.bcoo is None
    built = op.assemble()
    assert hasattr(built, "indices"), "assemble() must return a real BCOO"
    assert np.max(np.abs(np.asarray(built.todense()) - dense)) < 1e-12


def test_element_storage_is_smaller_than_the_assembled_matrix():
    """The measurement the whole representation exists for.

    Compared against the BCOO for the SAME operator. Note this is the conservative comparison: it
    uses the already-merged matrix, whereas the real assembler emits one triplet block per term, so
    the saving on a live 3-D Poisson is larger (measured 7.1x)."""
    op, _dense, _ = _random_problem(n_terms=3)
    bcoo = op.assemble()
    assembled = int(bcoo.data.nbytes + bcoo.indices.nbytes)
    assert op.nbytes < assembled, f"EBE {op.nbytes} B should beat assembled {assembled} B"
    assert assembled / op.nbytes > 1.5, f"expected a clear margin, got {assembled / op.nbytes:.2f}x"


def test_assembled_operator_carries_no_duplicate_triplets():
    """The assembled operator must store each ``(row, col)`` once.

    Two independent sources of redundancy used to survive into it: the assemblers append one triplet
    block **per additive weak-form term**, and every interior DOF pair receives a contribution from
    each element sharing it (~20 tets for P1). BCOO summed them lazily on every ``@``, so results
    were always right — they just cost ~19x the work on each of a Krylov solve's hundreds of matvecs.

    Measured on this 3-D Poisson before the fix: 96473 stored triplets for 4999 unique pairs, 1.47
    MiB, and a matvec 5.7x slower than necessary."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    import jno

    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.25).domain()
    u, phi = d.fem_symbols()
    c = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u(cb[0], cb[1], cb[2]) - 0.0])
    A, _b = fem.operator

    idx = np.asarray(A.indices)
    n_stored, n_unique = idx.shape[0], len({tuple(t) for t in idx})
    assert n_stored == n_unique, f"{n_stored} triplets stored for {n_unique} unique pairs"


def test_summing_duplicates_leaves_the_operator_unchanged():
    """Compression must be exact — it is a storage change, not an approximation. Checked as a matvec
    against the *uncompressed* triplets rebuilt by hand, so this cannot pass by comparing a value to
    itself."""
    from jax.experimental import sparse as jsp

    from jno.utils.solver.fem_utils import sum_duplicate_triplets

    rng = np.random.default_rng(0)
    n, nnz = 60, 900
    rows = rng.integers(0, n, nnz).astype(np.int32)  # heavy duplication by construction
    cols = rng.integers(0, n, nnz).astype(np.int32)
    data = rng.standard_normal(nnz)
    raw = jsp.BCOO((jnp.asarray(data), jnp.asarray(np.stack([rows, cols], 1))), shape=(n, n))

    dense = np.zeros((n, n))
    for r, c, dv in zip(rows, cols, data):
        dense[r, c] += dv

    packed = sum_duplicate_triplets(raw)
    assert packed.nse < raw.nse, "a duplicate-heavy operator must actually shrink"
    v = jnp.asarray(rng.standard_normal(n))
    assert np.max(np.abs(np.asarray(packed @ v) - dense @ np.asarray(v))) < 1e-12
    assert np.max(np.abs(np.asarray(packed @ v) - np.asarray(raw @ v))) < 1e-12


def test_dtype_is_reported_from_the_blocks():
    """``LinearOperator`` consumers read the operator dtype (spectral bound probes, Jacobi); an
    element operator has to answer the same question."""
    op, _dense, _ = _random_problem()
    assert op.dtype == jnp.float64
    v32 = jnp.zeros(op.shape[1], jnp.float32)
    assert op.mv(v32).dtype == jnp.float64, "an f32 probe must not silently demote the operator"
