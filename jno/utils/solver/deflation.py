"""Coarse spaces and deflation: the correction ``Q r = Z (Z^T A Z)^{-1} Z^T r`` for a subspace ``Z``.

``Z`` is ``(n, k)`` vectors, optionally split over ``p`` parts: column ``a`` restricted to part ``i`` is its
own coarse vector, so the coarse space has ``p * k`` directions while only ``(n, k)`` and the part labels are
stored (never a dense ``n x (p k)``). ``p = 1`` is a global space (Nicolaides, SIAM J. Numer. Anal. 24(2),
1987, for the per-part constants; Tang, Nabben, Vuik & Erlangga, J. Sci. Comput. 39 (2009) 340, for how the
correction combines with another preconditioner -- ``jno.precond.coarse`` and ``>>``).

Two phases, as for :mod:`schwarz`: the symbolic one (the space resolved to numbers, the partition, the
operator's entry tables) runs once on the host; the numeric one (``E = Z^T A Z`` and its inverse) is pure JAX
from the operator's current values, so it runs inside a compiled solve, a Newton loop or a time march.

``E`` is inverted through the whitened problem ``W^T E W`` with ``W = G^{-1/2}``, ``G = Z^T Z`` (block
diagonal, one ``k x k`` block per part). A direction with ``G ~ 0`` -- a part whose vectors vanish, e.g. a
part made only of eliminated Dirichlet rows -- carries nothing and is dropped. A direction with ``G`` well
away from zero but ``E ~ 0`` is a KERNEL of ``A`` inside the space: that is a singular problem, not a slow
one, and it is refused (in double precision; in single precision the cancellation that would expose it is
below the rounding, and it is dropped like the others).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "CoarsePattern",
    "operator_graph",
    "metis_parts",
    "coarse_pattern",
    "coarse_factor",
    "coarse_apply",
    "resolve_space",
]

#: A whitened direction whose |eigenvalue| is below this fraction of the largest is a kernel of ``A``.
#: Legitimate near-null directions sit far above it (measured in tests/test_precond_coarse.py).
KERNEL_RTOL = 1e-12
#: A direction of ``G = Z^T Z`` below this fraction of its part's largest carries no vector: dropped.
NULL_RTOL = 1e-12


class CoarsePattern(NamedTuple):
    """The symbolic phase of a coarse space.

    Attributes:
        n: system size.
        p: number of parts (1: a global space).
        k: vectors per part.
        part: ``(n,)`` part of each unknown (``p`` for an unknown outside every part: an eliminated row).
        null: ``(n, k)`` the vectors, zero outside the parts.
        W: ``(p, k, k)`` the whitening ``G^{-1/2}`` per part, zero columns for the dropped directions.
        dropped: directions with no vector (``G ~ 0``).
        nnz: unique entries of ``A`` (0 when the pattern has no entry tables: a matrix-free global space).
        checksum: identifies ``A``'s pattern (``None`` without entry tables).
        crow: ``(nnz,)`` coarse block ``part[r] * p + part[c]`` of each entry (``p * p``: outside every part).
        erow: ``(nnz,)`` row of each entry.
        ecol: ``(nnz,)`` column of each entry.
    """

    n: int
    p: int
    k: int
    part: jnp.ndarray
    null: jnp.ndarray
    W: jnp.ndarray
    dropped: int
    nnz: int = 0
    checksum: Any = None
    crow: Any = None
    erow: Any = None
    ecol: Any = None


def operator_graph(A):
    """``(S, G, iso)`` of a CONCRETE operator: ``S`` its CSR (float64, duplicates summed), ``G`` the symmetric
    loop-free unit graph of ``|S| + |S|^T``, ``iso`` the unknowns with no neighbour (eliminated Dirichlet rows)."""
    from .amg import _to_scipy_csr

    S = _to_scipy_csr(A).astype(np.float64)
    S.sum_duplicates()
    n = S.shape[0]
    if S.shape != (n, n):
        raise ValueError(f"jno.precond: the operator must be square, got {S.shape}.")
    G = (abs(S) + abs(S).T).tocsr()
    G.setdiag(0)
    G.eliminate_zeros()
    G.data[:] = 1.0
    return S, G, np.diff(G.indptr) == 0


def metis_parts(G, iso, parts: int) -> np.ndarray:
    """Part of every unknown: METIS k-way on the graph without its isolated unknowns (Karypis & Kumar, SIAM J.
    Sci. Comput. 20(1), 1998), which get part ``parts`` -- out of range everywhere downstream."""
    from .schwarz import _partition

    core = np.nonzero(~iso)[0]
    parts = int(min(max(1, parts), max(1, core.size)))
    part = np.full(G.shape[0], parts, np.int64)
    part[core] = _partition(G[core][:, core], parts)
    return part


def coarse_pattern(Z, part, p: int, A=None) -> CoarsePattern:
    """Symbolic phase. ``Z`` ``(n, k)`` real, ``part`` ``(n,)`` in ``[0, p]`` (``p``: outside every part).
    With a CONCRETE assembled ``A`` the entry tables are built, so ``E`` comes from ``k^2`` streaming segment
    sums over its entries; without it only a global space (``p = 1``) can be factored, from ``k`` products."""
    Z = np.asarray(Z)
    if np.iscomplexobj(Z):
        raise ValueError("jno.precond.coarse: the space must be real (the operator may be complex).")
    Z = Z.astype(np.float64).reshape(Z.shape[0], -1)
    part = np.asarray(part, np.int64)
    n, k = Z.shape
    null = np.where((part >= p)[:, None], 0.0, Z)
    W, dropped = _whitening(null, part, int(p))
    head = (n, int(p), k, jnp.asarray(part, jnp.int32), jnp.asarray(null), jnp.asarray(W), dropped)
    if A is None:
        return CoarsePattern(*head)
    from .fsai import _checksum

    if not jax.config.jax_enable_x64:
        raise ValueError(
            "jno.precond.coarse() needs x64 for an assembled operator: its entry keys row*n + col overflow int32."
        )
    S, _G, _iso = operator_graph(A)
    coo = S.tocoo()
    keys = np.sort(np.asarray(coo.row, np.int64) * n + np.asarray(coo.col, np.int64))
    r, c = keys // n, keys % n
    out = (part[r] >= p) | (part[c] >= p)
    crow = np.where(out, p * p, part[r] * p + part[c])
    return CoarsePattern(
        *head,
        int(keys.size),
        tuple(int(v) for v in _checksum(keys, np)),
        jnp.asarray(crow, jnp.int32),
        jnp.asarray(r, jnp.int32),
        jnp.asarray(c, jnp.int32),
    )


def _whitening(null, part, p):
    """``W = G^{-1/2}`` per part, ``G = Z_i^T Z_i`` (host, once): a direction of ``G`` below ``NULL_RTOL`` of its
    part's largest carries no vector and gets a zero column; ``dropped`` counts them."""
    k = null.shape[1]
    inside = part < p
    G = np.zeros((p, k, k))
    np.add.at(G, part[inside], null[inside][:, :, None] * null[inside][:, None, :])
    g, V = np.linalg.eigh(G)
    gmax = np.max(g, axis=1, keepdims=True)
    keep = g > NULL_RTOL * np.maximum(gmax, np.finfo(float).tiny)
    W = V * np.where(keep, 1.0 / np.sqrt(np.where(keep, g, 1.0)), 0.0)[:, None, :]
    return W, int(np.sum(~keep))


def _galerkin(pat: CoarsePattern, A, mv):
    """``E = Z^T A Z`` as ``(p k, p k)``: from the entries when the pattern has tables, else ``k`` products."""
    p, k = pat.p, pat.k
    if pat.crow is not None and A is not None and hasattr(A, "indices"):
        from .fsai import FsaiPattern, _keys_and_values

        _keys, vals = _keys_and_values(FsaiPattern(pat.n, pat.nnz, pat.checksum, [], 1), A)
        N = pat.null.astype(jnp.result_type(vals.dtype, jnp.float32))
        Nr, Nc = N[pat.erow], N[pat.ecol]
        # E[(i,a), (j,b)] = sum over entries (r, c) with part(r) = i, part(c) = j of N[r,a] A_rc N[c,b]:
        # k^2 streaming segment sums, no (nnz x k^2) intermediate.
        blocks = [
            [jax.ops.segment_sum(vals * Nr[:, a] * Nc[:, b], pat.crow, num_segments=p * p) for b in range(k)]
            for a in range(k)
        ]
        E = jnp.stack([jnp.stack(row, -1) for row in blocks], -2)  # (p*p, k, k)
        return E.reshape(p, p, k, k).transpose(0, 2, 1, 3).reshape(p * k, p * k)
    if p != 1 or mv is None:
        raise TypeError(
            "jno.precond.coarse(parts=...): a partitioned coarse space is built from the ASSEMBLED operator's "
            "entries, and this solve only has a matvec. Use a global space (parts=None), which needs only "
            "products with the operator, or an assembled solve path."
        )
    N = pat.null
    AZ = jax.vmap(lambda z: mv(z.astype(N.dtype)), in_axes=1, out_axes=1)(N)
    return N.T @ AZ


def _record_kernel(count, who):
    from .solver_api import _GATE_FAILURES

    _GATE_FAILURES.append(_kernel_message(int(count), who))


def _kernel_message(count, who):
    return (
        f"{who}: the coarse space contains {count} direction(s) the operator maps to zero -- a KERNEL of the "
        "system, not a slow mode. A singular problem is fixed in the problem, not the preconditioner: add the "
        "gauge term that removes it (pin a value, a mean-value constraint, a mass/epsilon term), or drop those "
        "vectors from the space."
    )


def coarse_factor(pat: CoarsePattern, A=None, *, mv=None, scale=None, check: bool = True, who: str = "jno.precond.coarse"):
    """Numeric phase (traceable): ``Einv = W (W^T E W)^+ W^T`` ``(p k, p k)``, ``E = Z^T A Z``.

    ``scale`` is the operator's size (its largest diagonal entry): a whitened direction -- a Rayleigh quotient
    ``z^T A z / z^T z`` -- below ``KERNEL_RTOL * scale`` is a kernel. With ``check`` such a direction raises
    (concrete values) or is recorded for the solve's gate to raise (traced values) -- double precision only."""
    p, k = pat.p, pat.k
    E = _galerkin(pat, A, mv)
    rdt = jnp.finfo(E.dtype).dtype  # the real dtype of E's precision
    Wb = _block_diag(pat.W.astype(rdt))
    Et = Wb.T @ E @ Wb
    U, s, Vh = jnp.linalg.svd(Et)
    smax = jnp.max(s)
    cut = jnp.maximum(KERNEL_RTOL, 64 * jnp.finfo(rdt).eps * p * k) * smax
    sinv = jnp.where(s > cut, 1.0 / jnp.where(s > cut, s, 1.0), 0.0)
    Einv = Wb @ (Vh.conj().T * sinv[None, :]) @ U.conj().T @ Wb.T
    if check and rdt == jnp.float64:
        ref = smax if scale is None else jnp.asarray(scale, rdt)
        kernel = jnp.sum(s <= KERNEL_RTOL * ref) - pat.dropped
        if isinstance(kernel, jax.core.Tracer):
            jax.lax.cond(kernel > 0, lambda c: jax.debug.callback(_record_kernel, c, who), lambda c: None, kernel)
        elif int(kernel) > 0:
            raise ValueError(_kernel_message(int(kernel), who))
    return Einv


def _block_diag(W):
    """``(p k, p k)`` block diagonal of ``W`` ``(p, k, k)`` without a Python loop over parts."""
    p, k, _ = W.shape
    out = jnp.zeros((p, k, p, k), W.dtype)
    i = jnp.arange(p)
    out = out.at[i, :, i, :].set(W)
    return out.reshape(p * k, p * k)


def coarse_apply(pat: CoarsePattern, Einv):
    """``(fwd, t)``: ``Q r = Z Einv Z^T r`` and its transpose ``Z Einv^T Z^T r``."""
    p, k = pat.p, pat.k

    def through(M, r):
        N = pat.null.astype(jnp.finfo(r.dtype).dtype)  # the vectors in r's (real) precision
        rc = jax.ops.segment_sum(N * r[:, None], pat.part, num_segments=p).reshape(-1)  # Z^T r, (p*k,)
        xc = (M.astype(jnp.result_type(M.dtype, r.dtype)) @ rc).reshape(p, k)
        return jnp.sum(N * jnp.take(xc, pat.part, axis=0, mode="fill", fill_value=0), axis=1).astype(r.dtype)

    return (lambda r: through(Einv, r)), (lambda r: through(Einv.T, r))


# -- resolving a space value to numbers -------------------------------------------------------------------
def _is_formula_space(space) -> bool:
    return isinstance(space, (list, dict))


def _field_index(fem, field) -> int:
    """Position of ``field`` (a trial symbol or an int) in the problem's per-field layout."""
    if isinstance(field, int):
        return field
    if hasattr(fem, "block_index"):
        return fem.block_index(field)
    unknowns = list(getattr(fem, "unknowns", None) or ())
    for i, u in enumerate(unknowns):
        if u is field or (
            getattr(u, "field_key", None) is not None
            and getattr(u, "field_key", None) == getattr(field, "field_key", object())
        ):
            return i
    raise KeyError(f"jno.precond.coarse: {getattr(field, 'name', field)!r} is not an unknown of this problem.")


def _require_nodal(fem, who):
    """Formulas are evaluated at nodes: refuse a field whose DOFs are not nodal Lagrange values."""
    from ..._fem import _trial_spaces

    cons = getattr(fem, "_constraints", None)
    if cons is None:
        return
    other = _trial_spaces(cons) - {"Lagrange"}
    if other:
        raise ValueError(
            f"{who}: a space written as formulas is evaluated at the DOF nodes, which needs nodal Lagrange fields; "
            f"this problem has {sorted(other)} (DOFs that are slopes, edge moments or enrichments, not values). "
            "Give the space as an (n, k) array instead."
        )


def _eval_formula(expr, pts) -> np.ndarray:
    """A coordinate formula (or a number) at ``pts`` ``(N, dim)``, as ``(N,)``."""
    from ..._fem import _eval_value_node_at

    if isinstance(expr, (int, float, np.integer, np.floating)):
        return np.full(pts.shape[0], float(expr))
    with jax.ensure_compile_time_eval():
        v = np.asarray(_eval_value_node_at(expr, jnp.asarray(pts)), dtype=np.float64).reshape(-1)
    if v.size == 1:
        return np.full(pts.shape[0], float(v[0]))
    if v.size != pts.shape[0]:
        raise ValueError(
            f"jno.precond.coarse: a formula gave {v.size} values at {pts.shape[0]} nodes; give one value per point."
        )
    return v


def _formula_columns(fem, space, n_full, who) -> np.ndarray:
    """``(n_full, k)``: each column is one entry of the space, evaluated per field and component."""
    from ...precond import _dof_layout

    _require_nodal(fem, who)
    layout = _dof_layout(fem, n_full)
    if isinstance(space, dict):
        per_field = [(_field_index(fem, f), list(cols)) for f, cols in space.items()]
    else:
        if len(layout) != 1:
            raise ValueError(
                f"{who}: this problem has {len(layout)} fields, so a list of formulas is ambiguous. Say which field "
                "each belongs to: coarse({u: [...], p: [...]}) -- a field left out gets zeros."
            )
        per_field = [(0, list(space))]
    cols = []
    for i, entries in per_field:
        pts, idx = layout[i]
        pts = np.asarray(pts)
        ncomp = idx.shape[1]
        for entry in entries:
            comps = tuple(entry) if isinstance(entry, (tuple, list)) else (entry,) * ncomp
            if len(comps) != ncomp:
                raise ValueError(
                    f"{who}: an entry with {len(comps)} components on a field with {ncomp}; give a tuple of {ncomp} "
                    "formulas, or one formula for all of them."
                )
            col = np.zeros(n_full)
            for c, f in enumerate(comps):
                col[idx[:, c]] = _eval_formula(f, pts)
            cols.append(col)
    if not cols:
        raise ValueError(f"{who}: the space is empty.")
    return np.stack(cols, axis=1)


def _restrict(fem, Z_full, n, who) -> np.ndarray:
    """Carry full-space vectors onto the solved space (a periodic tie / slip / hanging nodes: ``P^T A P``),
    field by field with the kept-DOF gather (exact for a vector the reduced space contains)."""
    from .fem_utils import restrict_state
    from .solver_api import _field_layout

    slices, red = _field_layout(fem, n)
    if red is None:
        return Z_full
    out = np.zeros((int(n), Z_full.shape[1]))
    with jax.ensure_compile_time_eval():
        for full, sl, b in zip(fem.blocks, slices, red):
            for k in range(Z_full.shape[1]):
                out[sl, k] = np.asarray(restrict_state(b["P"], Z_full[full, k], b.get("kept"), b.get("vec", 1)))
    return out


def resolve_space(space, fem, n: int, ctx=None, *, who: str = "jno.precond.coarse") -> np.ndarray:
    """A space value as an ``(n, k)`` array on the solved space (size ``n``).

    ``space``: formulas (a list, or ``{field: [...]}``; needs the problem), an ``(n, k)`` array, an object
    with ``basis(ctx)`` (``jno.space``), or a callable ``ctx -> (n, k)``."""
    if _is_formula_space(space):
        if fem is None:
            raise ValueError(
                f"{who}: a space written as formulas is evaluated at the problem's DOFs, and this solve has no "
                "problem attached. Pass it through fem.solve(precond=...), or give an (n, k) array."
            )
        from .solver_api import _field_layout

        red = _field_layout(fem, n)[1] if getattr(fem, "blocks", None) else None
        if red is None:
            return _formula_columns(fem, space, int(n), who)
        return _restrict(fem, _formula_columns(fem, space, int(fem.blocks[-1].stop), who), n, who)
    if hasattr(space, "basis") and callable(space.basis):
        Z = space.basis(ctx if ctx is not None else _FemOnly(fem, n))
    elif callable(space) and not hasattr(space, "shape"):
        Z = space(ctx if ctx is not None else _FemOnly(fem, n))
    else:
        Z = space
    if isinstance(Z, jax.core.Tracer):
        raise TypeError(f"{who}: the space must be concrete (it is set up once, outside the solve's trace).")
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.ndim != 2 or Z.shape[0] != int(n):
        raise ValueError(
            f"{who}: the space has shape {Z.shape}, but the operator being preconditioned has {int(n)} rows (a "
            "periodic tie / slip / hanging nodes solve on the REDUCED system P^T A P). Give one row per unknown "
            "of that system, or write the space as formulas."
        )
    if not np.all(np.isfinite(Z)):
        raise ValueError(f"{who}: the space has non-finite entries.")
    return Z


class _FemOnly:
    """The context a space callable sees when resolved at setup time (``prepare``): the problem and the size."""

    def __init__(self, fem, n):
        self.fem, self.n, self.A = fem, int(n), None
