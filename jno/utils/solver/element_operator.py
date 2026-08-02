"""Element-blocked (EBE) storage for an assembled FEM operator.

The assembled ``BCOO`` and this class hold the *same numbers*; they differ in what is kept alongside
them. A BCOO stores one ``(row, col)`` int32 pair **per nonzero**, and the assemblers emit one full
triplet block **per additive weak-form term** without pre-summing duplicates
(``fem_native._make_jacobian``). An element-blocked operator stores one dense ``(n_local, n_local)``
block per cell plus the ``(n_cells, n_local)`` connectivity — so the index cost drops by ``n_local``
and the per-term duplication collapses into a single summed block.

Measured on a real ``jno.fem`` 3-D Poisson (P1 tets): ``nnz / (n_cells * n_local**2)`` is 2.02, 3.02
and 4.02 for one, two and three gradient terms — i.e. one triplet block per term plus the boundary —
against **one** element block here, for **7.1x** less operator memory.

It also removes the assembly peak: ``_make_jacobian`` materialises ``broadcast_to(...).reshape(-1)``
for rows *and* cols at full ``nnz`` before concatenating. Those two int32 arrays are never formed
here, which is what raises the largest problem that fits on a given card.

The trade is that there is no assembled matrix: ``.bcoo`` is ``None``, so sparse-direct solvers and
matrix-based preconditioners refuse (they already carry targeted errors for a matrix-free operator).
:meth:`assemble` builds the BCOO explicitly for anyone who wants one — it is never built implicitly,
because doing so would silently undo the memory saving the caller asked for.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import jax.numpy as jnp

__all__ = ["ElementGroup", "ElementOperator"]


class ElementGroup:
    """One ``(blocks, rows, cols)`` contribution: ``A[rows[c, i], cols[c, j]] += blocks[c, i, j]``.

    ``blocks`` is ``(n_cells, n_test, n_local)``; ``rows`` is ``(n_cells, n_test)`` and ``cols`` is
    ``(n_cells, n_local)``. Volume terms that share a test field share both index arrays, so their
    blocks are summed into a single group at construction — that merge is where the memory win comes
    from. Surface terms index by parent cell and stay separate.
    """

    __slots__ = ("blocks", "rows", "cols")

    def __init__(self, blocks, rows, cols):
        self.blocks = blocks
        self.rows = jnp.asarray(rows, jnp.int32)
        self.cols = jnp.asarray(cols, jnp.int32)

    @property
    def nbytes(self) -> int:
        return int(self.blocks.nbytes + self.rows.nbytes + self.cols.nbytes)

    def matvec_into(self, v):
        """This group's contribution to ``A @ v``, as a dense ``(n_dofs,)`` scatter-add source."""
        local = v[self.cols]  # (n_cells, n_local) gather
        return jnp.einsum("cij,cj->ci", self.blocks, local)  # (n_cells, n_test)

    def transpose(self) -> "ElementGroup":
        """``Aᵀ`` is the same blocks transposed with rows and cols swapped — no data movement."""
        return ElementGroup(jnp.swapaxes(self.blocks, 1, 2), self.cols, self.rows)


def _merge_groups(groups: Sequence[ElementGroup]) -> List[ElementGroup]:
    """Sum groups that scatter to identical ``(rows, cols)`` into one block.

    This is the whole point: the assemblers emit one block per additive term, all of which (for the
    volume terms of one test field) carry the *same* connectivity. Summing them is exact — addition
    is what the BCOO would do lazily at matvec time anyway — and collapses the per-term duplication.
    Keyed on the index arrays' identity and content, not on term provenance, so it stays correct for
    a multifield form where different fields genuinely index differently.
    """
    merged: List[ElementGroup] = []
    for g in groups:
        for m in merged:
            if (
                m.rows.shape == g.rows.shape
                and m.cols.shape == g.cols.shape
                and m.blocks.shape == g.blocks.shape
                and bool(jnp.all(m.rows == g.rows))
                and bool(jnp.all(m.cols == g.cols))
            ):
                m.blocks = m.blocks + g.blocks
                break
        else:
            merged.append(ElementGroup(g.blocks, g.rows, g.cols))
    return merged


class ElementOperator:
    """An assembled FEM operator kept as per-element blocks instead of a global sparse matrix.

    Implements the subset of the :class:`~jno.utils.solver.solver_api.LinearOperator` contract the
    iterative solvers use — ``mv`` / ``__matmul__`` / ``__call__``, ``shape``, ``T``, ``diag()`` —
    and reports ``bcoo = None`` so anything requiring an assembled matrix refuses rather than
    silently materialising one.
    """

    def __init__(self, groups: Sequence[ElementGroup], shape: Tuple[int, int], *, _transposed: bool = False):
        self._groups = _merge_groups(groups) if not _transposed else list(groups)
        self._shape = tuple(shape)
        self._transposed = _transposed

    # -- LinearOperator contract -------------------------------------------------------------
    @property
    def shape(self):
        m, n = self._shape
        return (n, m) if self._transposed else (m, n)

    @property
    def bcoo(self):
        """``None`` — there is no assembled matrix. See :meth:`assemble`."""
        return None

    @property
    def dtype(self):
        return self._groups[0].blocks.dtype if self._groups else jnp.result_type(float)

    @property
    def nbytes(self) -> int:
        return sum(g.nbytes for g in self._groups)

    def mv(self, v):
        v = jnp.asarray(v).reshape(-1)
        out = jnp.zeros((self.shape[0],), jnp.result_type(v.dtype, self.dtype))
        for g in self._groups:
            out = out.at[g.rows].add(g.matvec_into(v))
        return out

    __matmul__ = mv
    __call__ = mv

    @property
    def T(self) -> "ElementOperator":
        """Lazy transpose. Unlike the BCOO path this is exact — each block is transposed and its row
        and column maps swap, so a non-symmetric adjoint solve gets the operator it asked for."""
        return ElementOperator([g.transpose() for g in self._groups], self._shape, _transposed=not self._transposed)

    def diag(self):
        """Exact diagonal, ``O(nnz)`` — the entries where a block's row and column DOFs coincide.
        Cheap enough that Jacobi preconditioning composes unchanged."""
        out = jnp.zeros((self.shape[0],), self.dtype)
        for g in self._groups:
            hit = g.rows[:, :, None] == g.cols[:, None, :]  # (n_cells, n_test, n_local)
            out = out.at[g.rows].add(jnp.sum(jnp.where(hit, g.blocks, 0.0), axis=2))
        return out

    def dense(self):
        return self.assemble().todense()

    def assemble(self):
        """Build the assembled ``BCOO`` **explicitly**.

        Never called implicitly. A caller who chose this representation did so for its memory
        profile, and quietly allocating the ~7x larger matrix underneath them is the one outcome
        this class exists to avoid — so anything needing a matrix has to ask in so many words.
        """
        from jax.experimental import sparse as jsp

        data, idx = [], []
        for g in self._groups:
            r = jnp.broadcast_to(g.rows[:, :, None], g.blocks.shape).reshape(-1)
            c = jnp.broadcast_to(g.cols[:, None, :], g.blocks.shape).reshape(-1)
            data.append(g.blocks.reshape(-1))
            idx.append(jnp.stack([r.astype(jnp.int32), c.astype(jnp.int32)], axis=1))
        if not data:
            return jsp.BCOO((jnp.zeros((0,), self.dtype), jnp.zeros((0, 2), jnp.int32)), shape=self.shape)
        return jsp.BCOO((jnp.concatenate(data), jnp.concatenate(idx)), shape=self.shape)

    def __repr__(self) -> str:
        cells = sum(int(g.blocks.shape[0]) for g in self._groups)
        return (
            f"ElementOperator(shape={self.shape}, groups={len(self._groups)}, cells={cells}, {self.nbytes / 2**20:.1f} MiB)"
        )


def from_bcoo_groups(groups: Sequence[Any], shape: Tuple[int, int]) -> ElementOperator:
    """Build from raw ``(blocks, rows, cols)`` triples, as the assemblers produce them."""
    return ElementOperator([g if isinstance(g, ElementGroup) else ElementGroup(*g) for g in groups], shape)
