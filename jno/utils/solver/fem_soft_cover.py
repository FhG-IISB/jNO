"""Soft enrichment: a continuous per-node weight on interpolation covers (private; not exposed through ``jno.solve``).

``jno.solve.enrich`` switches a node's cover coefficients on or off with a boolean mask, applied by
pinning. A boolean carries no gradient, so the ORDER DISTRIBUTION is invisible to ``jax.grad``, and
every round re-lowers the assembly because the pins change the sparsity. This module replaces the
mask by a weight ``w_i in [0, 1]`` per node, so the solution, any functional of it, and the choice of
where to enrich all become differentiable, on one fixed sparsity pattern.

**Why a penalty and not a scaled basis.** Multiplying node ``i``'s cover functions by ``w_i`` does not
change their span, so the Galerkin solution would not depend on ``w`` at all for ``w > 0``. The weight
has to change the *problem*, not the basis. It does so through a diagonal penalty on each cover DOF
``k`` of node ``i``::

    A_w = A + diag(p),    p_k = c_k (1/g_i - 1),    c_k = A_kk,    g_i = w_min + (1 - w_min) w_i

``g = 1`` adds nothing (the full cover space); ``g -> 0`` forces the coefficient to zero (plain P1 at
that node). With ``c_k = A_kk`` a cover coefficient decoupled from the rest is exactly
``a_k(g) = g * a_k(full)``: the weight linearly interpolates the coefficient. ``w_min`` keeps ``A_w``
finite at ``w = 0``; the diagonal then grows like ``1/w_min``, which makes the system more diagonally
dominant, not less.

**The gradient at w = 0 is a hierarchical indicator.** With ``Pi(u) = 1/2 u.A u - b.u`` evaluated with
the UNPENALISED ``A``, expanding around the P1 solution gives, to first order in ``g``::

    dPi/dw_i |_{w=0} = -(1 - w_min) * sum_{k in covers(i)} r_k^2 / A_kk,    r = b - A u_P1

the diagonal hierarchical estimate of the energy each node's enrichment would buy (Bank & Smith,
*SIAM J. Numer. Anal.* **30** (1993) 921-935, Sec. 2),
here obtained by automatic differentiation rather than written by hand. Couplings between cover DOFs
enter at O(g^2). ``tests/test_fem_soft_cover.py`` checks this identity.

**No assembler change.** jNO keeps pinned DOFs as identity rows in a full-size system, so the penalty
is appended to the fully enriched operator as extra BCOO entries; ``sparse_lu_solve`` sums duplicates
with a static entry count, so one compiled program serves every ``w``.

Scope of the prototype: one ``space="cover"`` field (scalar or vector), steady linear problem, the operator
assembled with NO enrichment mask (every node carries its cover slots). Homogeneous Dirichlet data makes
``Pi`` an exact error measure: ``2 (Pi(u) - Pi_exact)`` is the squared energy-norm error of any field in
the cover space. Penalising a pinned cover DOF (Dirichlet or null-space gauge) is harmless: its row is
the identity with right-hand side 0, so it stays 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsp

from .fem_cover import cover_block, cover_count
from .linear import sparse_lu_solve

W_MIN = 1e-6


def cover_dof_index(n_nodes: int, dim: int, n_comp: int = 1) -> np.ndarray:
    """``(n_nodes, dim * n_comp)`` global DOF of every cover coefficient of ONE cover field.

    The assembler's layout (``fem_native``, the enrichment pins): node ``i``, cover ``m`` (1-based, after
    the value slot), component ``c`` sits at ``(i * (1 + dim) + m) * n_comp + c``. ``n_comp = 1`` is a
    scalar field, ``n_comp = dim`` a displacement. Columns are ordered cover-major, component-minor."""
    blk, m = cover_block(dim), cover_count(dim)
    i = np.arange(n_nodes, dtype=np.int64)[:, None, None]
    mm = (1 + np.arange(m, dtype=np.int64))[None, :, None]
    cc = np.arange(n_comp, dtype=np.int64)[None, None, :]
    return ((i * blk + mm) * n_comp + cc).reshape(n_nodes, m * n_comp).astype(np.int32)


def operator_diagonal(A, idx: np.ndarray) -> np.ndarray:
    """``A[k, k]`` at every DOF in ``idx`` (any shape), read once on the host from a BCOO operator."""
    ind = np.asarray(A.indices)
    data = np.asarray(A.data)
    on = ind[:, 0] == ind[:, 1]
    diag = np.zeros(A.shape[0], dtype=data.dtype)
    np.add.at(diag, ind[on, 0], data[on])  # duplicates of a diagonal entry are summed, as the solve does
    out = diag[np.asarray(idx)]
    if np.any(out <= 0.0):
        raise ValueError(
            "soft enrichment: a cover DOF has a non-positive diagonal, so the penalty scale c_k = A_kk is "
            "not a stiffness. Expected an SPD operator from a single scalar cover field."
        )
    return out


def gate(w, w_min: float = W_MIN):
    """The weight the penalty actually sees, ``w_min + (1 - w_min) w``: affine, so ``w = 0`` is finite
    and no clip sits in the gradient path (a clip's tie at the bound halves the derivative there)."""
    return w_min + (1.0 - w_min) * jnp.asarray(w)


def penalised_operator(A, cover_idx: np.ndarray, c: np.ndarray, w, w_min: float = W_MIN):
    """``A + diag(p)`` with ``p_k = c_k (1/g_i - 1)`` on node ``i``'s cover DOFs, as a BCOO whose entry
    count does not depend on ``w`` (the penalty entries are appended, not merged)."""
    g = gate(w, w_min)
    p = jnp.asarray(c) * (1.0 / g[:, None] - 1.0)  # (n_nodes, dim)
    flat = jnp.asarray(cover_idx).reshape(-1)
    data = jnp.concatenate([A.data, p.reshape(-1).astype(A.data.dtype)])
    ind = jnp.concatenate([A.indices, jnp.stack([flat, flat], axis=1).astype(A.indices.dtype)])
    return jsp.BCOO((data, ind), shape=A.shape)


def soft_solve(A, b, cover_idx: np.ndarray, c: np.ndarray, w, w_min: float = W_MIN):
    """Solution of the softly enriched problem; differentiable in ``w`` (and in ``A``, ``b``)."""
    return sparse_lu_solve(penalised_operator(A, cover_idx, c, w, w_min), b)


def ritz_energy(A, b, u):
    """``Pi(u) = 1/2 u.A u - b.u`` with the UNPENALISED operator: the physical energy of the field."""
    u = jnp.asarray(u).reshape(-1)
    return 0.5 * jnp.dot(u, A @ u) - jnp.dot(jnp.asarray(b).reshape(-1), u)


def hierarchical_indicator(A, b, cover_idx: np.ndarray, c: np.ndarray, u_p1, w_min: float = W_MIN):
    """The closed form the AD gradient must reach at ``w = 0``: ``-(1 - w_min) sum_k r_k^2 / A_kk`` per
    node, with ``r = b - A u_P1``. Returned with the sign of ``dPi/dw`` (non-positive)."""
    r = jnp.asarray(b).reshape(-1) - A @ jnp.asarray(u_p1).reshape(-1)
    rk = r[jnp.asarray(cover_idx)]
    return -(1.0 - w_min) * jnp.sum(rk**2 / jnp.asarray(c), axis=1)


# ---------------------------------------------------------------- decoupled switch-off (compacted pattern)


def compact_solver(
    A,
    cover_idx: np.ndarray,
    c: np.ndarray | None = None,
    w_min: float = W_MIN,
    linear_solve=None,
    full_matvec=None,
    cap: int | None = None,
):
    """``solve(w, b, data=None, params=None)``: the soft solve with every node at EXACTLY ``w = 0`` switched off for real.

    :func:`soft_solve` keeps a switched-off cover in the system behind a huge diagonal, so its couplings
    stay in the sparsity pattern and a sparse-direct factorisation pays for them as if they mattered:
    every solve costs a full-cover solve. Here a node with ``w_i == 0`` is DECOUPLED -- its cover rows
    and columns are dropped, an identity row with right-hand side 0 takes its place -- and the dropped
    entries are pointed at the out-of-bounds index BCOO uses for padding, which the CSR conversion sorts
    past the last row and the factorisation never sees. Array lengths never change, so one compiled
    program serves every mask; the WORK follows the active set.

    A node with ``0 < w_i <= 1`` is gated by the penalty exactly as in :func:`soft_solve`.

    **Gradient at a switched-off node.** Dropping the node removes its coupling from the solve, so the
    derivative is supplied by a custom JVP: it is the one-sided limit ``w_i -> 0+`` of the penalised
    model. There the node's cover coefficients move at the rate ``dx_k/dw_i = (1 - w_min) r_k / A_kk``
    with ``r = b - A x`` (the residual against the cover test functions), and the active unknowns respond
    through one solve with the compacted operator. That is the identity
    ``dPi/dw_i|_{w=0} = -(1 - w_min) sum r_k^2 / A_kk`` the tests pin, extended to any objective, so an
    optimiser can switch a node back on.

    ``data`` (the operator's stored values, default ``A.data``) and ``b`` are differentiable too, so a
    physical parameter can sit in the same trace. The penalty scale is ``c`` when given (a constant);
    with ``c=None`` it is read off the CURRENT operator's diagonal, ``c_k = A_kk(data)``, so it follows a
    coefficient that changes during a design loop (a SIMP conductivity varies by orders of magnitude
    between void and solid), and its derivative enters the tangent. At ``w in {0, 1}`` the penalty is
    zero or the node is dropped, so ``c`` does not affect the solution at all.

    ``linear_solve`` is the ``(A, b) -> x`` sparse-direct solve (default :func:`sparse_lu_solve`); a 3-D
    problem wants a factor-once backend such as ``pardiso_lu_solve``.

    **Assembling only the active set.** The one-sided gradient above reads the FULL operator twice: the
    residual of a switched-off node's covers, and the product with their rates. Those are the only places
    a switched-off cover's couplings are needed, so a caller that assembles only the active couplings
    (plus every cover DOF's diagonal, which the scale reads) passes ``full_matvec(params, v)``, the full
    operator's product evaluated another way (matrix-free, per element), and ``params`` to the solve. The
    solution depends on ``params`` only through ``data``, so ``params`` carries no tangent of its own.

    ``cap`` gathers the live entries into a fixed-length prefix before the solve, so a host-side backend
    receives (and hashes, and converts) about the active entries rather than the whole pattern. It must
    bound the live count for every mask the caller uses; a mask that overflows it returns NaN rather than
    a silently truncated operator.
    """
    linear_solve = sparse_lu_solve if linear_solve is None else linear_solve
    n = int(A.shape[0])
    n_nodes, dim = (int(s) for s in np.asarray(cover_idx).shape)
    idx = jnp.asarray(cover_idx)
    flat = idx.reshape(-1)
    ind0 = A.indices
    nse = int(ind0.shape[0])
    # positions of each cover DOF's diagonal entries in the stored arrays (duplicates are summed)
    _ind_np = np.asarray(A.indices)
    _dofs_np = np.asarray(cover_idx).reshape(-1)
    _slot = np.full(n, -1, dtype=np.int64)
    _slot[_dofs_np] = np.arange(_dofs_np.size)
    _on_diag = (_ind_np[:, 0] == _ind_np[:, 1]) & (_slot[np.minimum(_ind_np[:, 0], n - 1)] >= 0) & (_ind_np[:, 0] < n)
    _diag_pos = jnp.asarray(np.flatnonzero(_on_diag))
    _diag_seg = jnp.asarray(_slot[_ind_np[_on_diag, 0]])
    c_fixed = None if c is None else jnp.asarray(c)

    def _scale(data):
        if c_fixed is not None:
            return c_fixed
        d = jax.ops.segment_sum(data[_diag_pos], _diag_seg, num_segments=_dofs_np.size)
        return d.reshape(n_nodes, dim)

    node_of = np.full(n, -1, dtype=np.int32)
    node_of[np.asarray(cover_idx).reshape(-1)] = np.repeat(np.arange(n_nodes, dtype=np.int32), dim)
    node_of = jnp.asarray(node_of)
    is_cov = node_of >= 0
    safe_node = jnp.maximum(node_of, 0)
    pad = jnp.asarray([n, n], dtype=ind0.dtype)
    diag_ind = jnp.stack([flat, flat], axis=1).astype(ind0.dtype)
    in_bounds = ind0[:, 0] < n

    def _switched(w, data, b):
        """``(A_sw, b_sw, dof_off, overflow)``: the compacted operator, its right-hand side, the dropped DOFs."""
        off = w <= 0.0
        dof_off = is_cov & off[safe_node]
        drop = dof_off[ind0[:, 0]] | dof_off[ind0[:, 1]]
        d_kept = jnp.where(drop, jnp.zeros_like(data), data)
        i_kept = jnp.where(drop[:, None], pad[None, :], ind0)
        overflow = jnp.asarray(False)
        if cap is not None:  # the live entries first, in a fixed-length prefix; a dummy pad entry fills the rest
            live = ~drop & in_bounds
            pos = jnp.nonzero(live, size=cap, fill_value=nse)[0]
            d_kept = jnp.concatenate([d_kept, jnp.zeros(1, d_kept.dtype)])[pos]
            i_kept = jnp.concatenate([i_kept, pad[None, :]])[pos]
            overflow = jnp.sum(live) > cap
        g = gate(w, w_min)
        p = _scale(data) * (1.0 / g[:, None] - 1.0)
        p = jnp.where(off[:, None], 1.0, p).reshape(-1).astype(data.dtype)  # identity row when off
        A_sw = jsp.BCOO((jnp.concatenate([d_kept, p]), jnp.concatenate([i_kept, diag_ind])), shape=A.shape)
        return A_sw, jnp.where(dof_off, 0.0, b), dof_off, overflow

    def _guard(x, overflow):
        return jnp.where(overflow, jnp.nan, x)

    def _full(data, params, v):
        return jsp.BCOO((data, ind0), shape=A.shape) @ v if full_matvec is None else full_matvec(params, v)

    @jax.custom_jvp
    def _solve(w, data, b, params):
        A_sw, b_sw, _, overflow = _switched(w, data, b)
        return _guard(linear_solve(A_sw, b_sw), overflow)

    @_solve.defjvp
    def _solve_jvp(primals, tangents):
        w, data, b, params = primals
        dw, ddata, db, _dparams = tangents  # params acts only through data, whose tangent is ddata
        A_sw, b_sw, dof_off, overflow = _switched(w, data, b)
        x = linear_solve(A_sw, b_sw)
        off = w <= 0.0
        cj = _scale(data)
        # switched-off nodes: the one-sided rate of their cover coefficients
        r = (b - _full(data, params, x))[idx]  # (n_nodes, dim)
        s_nodes = jnp.where(off[:, None], (1.0 - w_min) * r / cj * dw[:, None], 0.0)
        s = jnp.zeros(n, dtype=x.dtype).at[flat].set(s_nodes.reshape(-1), unique_indices=True)
        # switched-on nodes: the penalty's own derivative, dP_k = -c_k (1 - w_min) / g^2 dw_i
        g = gate(w, w_min)
        dp = jnp.where(off[:, None], 0.0, -cj * (1.0 - w_min) / g[:, None] ** 2 * dw[:, None])
        if c_fixed is None:  # the scale follows the operator: dp_k also moves with dA_kk
            dp = dp + jnp.where(off[:, None], 0.0, _scale(ddata) * (1.0 / g[:, None] - 1.0))
        dPx = jnp.zeros(n, dtype=x.dtype).at[flat].set(dp.reshape(-1) * x[flat], unique_indices=True)
        dAx = jsp.BCOO((ddata, ind0), shape=A.shape) @ x
        rhs = db - dAx - dPx - _full(data, params, s)
        rhs = jnp.where(dof_off, s, rhs)  # identity rows carry the prescribed rate
        return _guard(x, overflow), linear_solve(A_sw, rhs)

    def solve(w, b, data=None, params=None):
        return _solve(
            jnp.asarray(w, dtype=A.data.dtype),
            A.data if data is None else data,
            jnp.asarray(b).reshape(-1),
            jnp.zeros(()) if params is None else params,
        )

    solve.switched = (
        lambda w, b, data=None: _switched(  # for inspection: the operator the solver sees
            jnp.asarray(w, dtype=A.data.dtype), A.data if data is None else data, jnp.asarray(b).reshape(-1)
        )[:3]
    )
    return solve


# ---------------------------------------------------------------- matrix-free product over the active set


def compact_matvec(A, cover_idx: np.ndarray, c: np.ndarray, w_min: float = W_MIN, chunk: int | None = None):
    """``(prepare, matvec)``: the switched operator of :func:`compact_solver` as a product whose WORK
    follows the active entries, for iterative solvers.

    A fixed-length sparse product reads every stored entry, zero or not, so the padded layout costs
    full-cover memory traffic per iteration however few covers are on. The one place XLA lets the
    amount of work depend on data is a loop whose trip count is a traced value. So:

    * the stored entries are sorted ONCE by row (host, at construction);
    * ``prepare(w)`` -- once per mask change, not per iteration -- stably moves the active entries to the
      front (row order kept inside the prefix) and counts them;
    * ``matvec(state, x)`` loops over ``ceil(n_active / chunk)`` fixed-size chunks of that prefix.

    Each chunk is a gather of ``x`` and a scatter-add into ``y``; the diagonal part (identity rows of the
    switched-off covers, the penalty of fractional ones) is a separate elementwise term. One compiled
    program serves every mask. ``chunk`` trades loop overhead against reading past the end of the prefix
    (at most one chunk). The default depends on the backend, because a loop trip costs a kernel launch on
    a GPU and almost nothing on a CPU: measured on a 3.5M-entry operator with 0.9M live entries, a
    32k-entry chunk was the fastest on CPU and the SLOWEST on GPU (slower than the padded product),
    where 1M-entry chunks came within 26% of an operator holding the live entries only.
    """
    if chunk is None:
        chunk = (1 << 20) if jax.default_backend() == "gpu" else (1 << 15)
    n = int(A.shape[0])
    n_nodes, dim = (int(s) for s in np.asarray(cover_idx).shape)
    ind = np.asarray(A.indices)
    data = np.asarray(A.data)
    order = np.lexsort((ind[:, 1], ind[:, 0]))  # CSR order, once
    rows_np, cols_np, data_np = ind[order, 0], ind[order, 1], data[order]
    nse = rows_np.size
    n_pad = -(-nse // chunk) * chunk
    pad = n_pad - nse
    rows0 = jnp.asarray(np.concatenate([rows_np, np.full(pad, n)]).astype(np.int32))
    cols0 = jnp.asarray(np.concatenate([cols_np, np.zeros(pad)]).astype(np.int32))
    data0 = jnp.asarray(np.concatenate([data_np, np.zeros(pad, dtype=data_np.dtype)]))
    node_of = np.full(n, -1, dtype=np.int32)
    node_of[np.asarray(cover_idx).reshape(-1)] = np.repeat(np.arange(n_nodes, dtype=np.int32), dim)
    node_of = jnp.asarray(node_of)
    is_cov, safe_node = node_of >= 0, jnp.maximum(node_of, 0)
    flat = jnp.asarray(cover_idx).reshape(-1)
    cj = jnp.asarray(c)

    @jax.jit
    def prepare(w):
        w = jnp.asarray(w, dtype=data0.dtype)
        off = w <= 0.0
        dof_off = is_cov & off[safe_node]
        live = ~(dof_off[jnp.minimum(rows0, n - 1)] | dof_off[cols0]) & (rows0 < n)
        perm = jnp.argsort(~live, stable=True)  # active first, row order kept inside the prefix
        g = gate(w, w_min)
        p = jnp.where(off[:, None], 0.0, cj * (1.0 / g[:, None] - 1.0)).reshape(-1)
        diag = jnp.zeros(n, dtype=data0.dtype).at[flat].set(p, unique_indices=True)
        return {
            "rows": rows0[perm],
            "cols": cols0[perm],
            "data": jnp.where(live[perm], data0[perm], 0.0),
            "n_chunks": (jnp.sum(live) + chunk - 1) // chunk,
            "diag": diag,
            "off": dof_off,
        }

    def matvec(state, x):
        def body(k, y):
            s = k * chunk
            r = jax.lax.dynamic_slice(state["rows"], (s,), (chunk,))
            cc = jax.lax.dynamic_slice(state["cols"], (s,), (chunk,))
            d = jax.lax.dynamic_slice(state["data"], (s,), (chunk,))
            return y.at[r].add(d * x[cc], mode="drop")

        y = jax.lax.fori_loop(0, state["n_chunks"], body, jnp.zeros_like(x))
        return jnp.where(state["off"], x, y + state["diag"] * x)

    return prepare, matvec
