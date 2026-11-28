"""Degree-robust preconditioners for matrix-free spectral elements (``jno.fem(..., matrix_free=True)``, Q_k).

Two pieces, combined by the specs ``jno.precond.fdm_schwarz`` and ``jno.precond.pmg``:

**Element fast-diagonalisation Schwarz** (Lynch, Rice & Thomas, Numer. Math. 6, 1964; Lottes & Fischer, J. Sci.
Comput. 24, 2005). Each cell's subdomain is the cell extended by one node into each neighbour, with Dirichlet
there. Its operator is replaced by the separable one ``Σ_a c_a (M̃⊗..K̃..⊗M̃) + c_0 M̃⊗..⊗M̃``, where ``K̃``,
``M̃`` are the 1-D GLL stiffness and mass of the extended reference interval (the neighbour taken as a mirror of
the cell) and ``c`` the cell's mean value and axis coefficients read off the weak form at the current state
(``fem_native._vol_cell_sem``). With ``K̃ S = M̃ S Λ``, ``Sᵀ M̃ S = I`` the inverse is ``S^{⊗d} diag(1 / (Σ_a c_a
λ_{i_a} + c_0)) S^{T⊗d}``: ``2d`` axis contractions per cell, the cost of an operator apply, and ``S``, ``Λ`` are
one ``(k + 1)²`` eigenproblem per degree. Exact on an axis-aligned cell with a constant coefficient, up to the
neighbour's coupling; the metric's off-diagonal part and a coefficient's variation inside the cell are dropped.

**Q1 coarse space.** ``P`` interpolates the mesh vertices' bilinear / trilinear functions at the Q_k nodes,
``P̂ = mask · P`` drops the Dirichlet rows, and ``A_c = P̂ᵀ A P̂`` is assembled per cell from the same per-point
coefficients (exact Galerkin for the volume terms; boundary-integral terms are left out of the coarse
operator). Solved densely up to :data:`DENSE_COARSE_MAX` unknowns, else by an AMG V-cycle whose hierarchy is
built once on the host (pyamg) and whose finest level takes the current ``A_c``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

#: A coarse operator up to this many unknowns is factored densely at every build (exact, traceable); above,
#: AMG. 4096 is 128 MiB in float64 and about 10 ms to factor on the RTX 3070.
DENSE_COARSE_MAX = 4096

_HI = jax.lax.Precision.HIGHEST


def fdm_1d(p: int):
    """``(S, lam)`` of the extended 1-D GLL problem with ``p`` nodes on ``[0, 1]`` (see the module docstring)."""
    import scipy.linalg

    from ..._spectral.bases import Legendre

    ax = Legendre(p, 0.0, 1.0)
    w = np.asarray(ax.weights, float)
    D = np.asarray(ax.diff_matrix(1), float)
    K = D.T @ (w[:, None] * D)
    Kt, Mt = K.copy(), np.diag(w)
    # the neighbour (a mirror of this cell) adds its end entries at the shared nodes; its next node is Dirichlet
    Kt[0, 0] += K[-1, -1]
    Kt[-1, -1] += K[0, 0]
    Mt[0, 0] += w[-1]
    Mt[-1, -1] += w[0]
    lam, S = scipy.linalg.eigh(Kt, Mt)
    return S, lam, np.asarray(ax.nodes, float)


@dataclass
class SemStatics:
    """Host-built structure of one Q_k field (one field per problem), shared by every build of a preconditioner."""

    n: int  # system size
    vec: int
    tdim: int
    p: int  # nodes per axis
    lex_dofs: Any  # (n_cells, p^tdim, vec) global DOFs in lexicographic node order
    S: Any  # (p, p)
    lam: Any  # (p,)
    phi_local: Any  # (n_dof, n_v): the vertex functions at the element's nodes, local order
    P: Any  # BCOO (n, n_coarse)
    Pt: Any  # BCOO (n_coarse, n)
    n_coarse: int
    coarse_rows: Any  # (nnz,) coarse pattern
    coarse_cols: Any
    coarse_inv: Any  # (n_cells * (n_v vec)^2,) -> pattern slot
    # per problem, kept across solves and specs: the frozen AMG hierarchy of the coarse operator (see `coarse_levels`)
    cache: dict = field(default_factory=dict)


def sem_statics(factors, cdofs, vec: int, cell_type: str, order: int, n: int) -> SemStatics:
    """Build :class:`SemStatics` from the field's tensor factors and its cell DOFs ``(n_cells, n_dof·vec)``."""
    import jax.experimental.sparse as jsp

    from .fem_lagrange import _lagrange_basix, basix_cell

    cell, tdim = basix_cell(cell_type)
    X = np.asarray(_lagrange_basix(cell, int(order)).points, float)  # (n_dof, tdim), local order, on [0, 1]^d
    cd = np.asarray(cdofs).astype(np.int64)
    n_cells, n_dof = cd.shape[0], X.shape[0]
    if cd.shape[1] != n_dof * vec:
        raise ValueError(f"sem_statics: {cd.shape[1]} DOFs per cell, expected {n_dof} nodes x {vec} components")
    p = int(order) + 1
    S, lam, nodes = fdm_1d(p)
    dof = np.asarray(factors.dof)  # lexicographic position -> local DOF
    lex = X[dof].reshape((p,) * tdim + (tdim,))
    grid = np.stack(np.meshgrid(*([nodes] * tdim), indexing="ij"), axis=-1)
    if not np.allclose(lex, grid, atol=1e-10):
        raise ValueError(
            "the spectral-element preconditioners need the GLL nodes of a Q_k field (k >= 3 on quadrilaterals or "
            "hexahedra); this field's nodes are not."
        )
    n_v = 2**tdim
    bits = np.array([[(j >> a) & 1 for a in range(tdim)] for j in range(n_v)])  # (n_v, tdim)
    phi = np.prod(np.where(bits[None, :, :] == 1, X[:, None, :], 1.0 - X[:, None, :]), axis=-1)  # (n_dof, n_v)
    corner_local = np.array([int(np.argmin(np.abs(X - b[None, :]).sum(1))) for b in bits.astype(float)])
    cdn = cd.reshape(n_cells, n_dof, vec)
    vkey = cdn[:, corner_local, 0]  # (n_cells, n_v): a vertex by its component-0 DOF
    uniq, vidx = np.unique(vkey.reshape(-1), return_inverse=True)
    vidx = vidx.reshape(n_cells, n_v)
    n_coarse = int(uniq.size) * vec
    gc = (vidx[:, :, None] * vec + np.arange(vec)[None, None, :]).reshape(n_cells, n_v * vec)  # (j, comp)
    # P: row (node, comp) -> col (vertex, comp), value phi; the same value from every cell that holds the node
    keep = np.abs(phi) > 1e-14
    nn, jj = np.nonzero(keep)
    rows = cdn[:, nn, :]  # (n_cells, nnz_e, vec)
    cols = vidx[:, jj][:, :, None] * vec + np.arange(vec)[None, None, :]
    vals = np.broadcast_to(phi[nn, jj][None, :, None], rows.shape)
    key = rows.reshape(-1) * n_coarse + cols.reshape(-1)
    _u, first = np.unique(key, return_index=True)
    r_, c_, v_ = rows.reshape(-1)[first], cols.reshape(-1)[first], vals.reshape(-1)[first]
    P = jsp.BCOO((jnp.asarray(v_), jnp.asarray(np.stack([r_, c_], 1).astype(np.int32))), shape=(n, n_coarse))
    Pt = jsp.BCOO((jnp.asarray(v_), jnp.asarray(np.stack([c_, r_], 1).astype(np.int32))), shape=(n_coarse, n))
    pr = np.repeat(gc, gc.shape[1], axis=1).reshape(-1)  # (cell, i, j) row-major
    pc = np.tile(gc, (1, gc.shape[1])).reshape(-1)
    pkey = pr * n_coarse + pc
    ukey, inv = np.unique(pkey, return_inverse=True)
    lex_dofs = cdn[:, dof, :]  # (n_cells, p^tdim, vec)
    return SemStatics(
        n=int(n), vec=int(vec), tdim=int(tdim), p=p, lex_dofs=jnp.asarray(lex_dofs.astype(np.int32)),
        S=jnp.asarray(S), lam=jnp.asarray(lam), phi_local=phi, P=P, Pt=Pt, n_coarse=n_coarse,
        coarse_rows=jnp.asarray((ukey // n_coarse).astype(np.int32)), coarse_cols=jnp.asarray((ukey % n_coarse).astype(np.int32)),
        coarse_inv=jnp.asarray(inv.astype(np.int32)),
    )  # fmt: skip


class SemData:
    """One build of the preconditioners' data at one state: the separable coefficients, the coarse operator and
    the Dirichlet structure (``mask`` 1 on free rows, ``pin`` the constrained rows' diagonal)."""

    def __init__(self, st: SemStatics, coeffs, Ce, mask, pin):
        self.st, self.mask, self.pin = st, jnp.asarray(mask), jnp.asarray(pin)
        c = jnp.abs(jnp.asarray(coeffs))  # (n_cells, vec, 1 + tdim)
        # an indefinite or vanishing cell still gets an invertible local problem
        top = jnp.max(c[..., 1:], axis=-1, keepdims=True)
        c = c.at[..., 1:].set(jnp.maximum(c[..., 1:], 1e-6 * top + 1e-300))
        self.coeffs = c
        data = jax.ops.segment_sum(jnp.asarray(Ce).reshape(-1), st.coarse_inv, num_segments=int(st.coarse_rows.shape[0]))
        self.coarse_data = data

    # ---- the element FDM Schwarz -----------------------------------------------------------------------------
    def schwarz(self, r):
        """``z = mask Σ_e R_eᵀ Ã_e⁻¹ R_e (mask r) + (1 - mask) r / pin``; symmetric."""
        st = self.st
        r = jnp.asarray(r)
        rm = r * self.mask.astype(r.dtype)
        n_cells = st.lex_dofs.shape[0]
        T = rm[st.lex_dofs].reshape((n_cells,) + (st.p,) * st.tdim + (st.vec,))
        S = st.S.astype(r.dtype)
        letters = "abc"[: st.tdim]
        for a in range(st.tdim):  # Sᵀ along every axis
            src = "n" + letters + "v"
            dst = src.replace(letters[a], "z")
            T = jnp.einsum(f"{letters[a]}z,{src}->{dst}", S, T, precision=_HI)
        L = self.coeffs[:, :, 0].astype(r.dtype)  # (n_cells, vec)
        L = L.reshape((n_cells,) + (1,) * st.tdim + (st.vec,))
        lam = st.lam.astype(r.dtype)
        for a in range(st.tdim):
            shape = [1] * (2 + st.tdim)
            shape[1 + a] = st.p
            ca = self.coeffs[:, :, 1 + a].astype(r.dtype).reshape((n_cells,) + (1,) * st.tdim + (st.vec,))
            L = L + ca * lam.reshape(shape)
        T = T / L
        for a in range(st.tdim):  # S along every axis
            src = "n" + letters + "v"
            dst = src.replace(letters[a], "z")
            T = jnp.einsum(f"z{letters[a]},{src}->{dst}", S, T, precision=_HI)
        z = jnp.zeros_like(r).at[st.lex_dofs.reshape(-1)].add(T.reshape(-1))
        m = self.mask.astype(r.dtype)
        return m * z + (1 - m) * r / jnp.where(self.pin != 0, self.pin, 1.0).astype(r.dtype)

    # ---- the Q1 coarse correction ------------------------------------------------------------------------------
    def coarse_matrix(self):
        import jax.experimental.sparse as jsp

        st = self.st
        idx = jnp.stack([st.coarse_rows, st.coarse_cols], axis=1)
        return jsp.BCOO((self.coarse_data, idx), shape=(st.n_coarse, st.n_coarse))

    def coarse_dense(self):
        st = self.st
        return (
            jnp.zeros((st.n_coarse, st.n_coarse), self.coarse_data.dtype)
            .at[st.coarse_rows, st.coarse_cols]
            .add(self.coarse_data)
        )


#: The coarse AMG solve: Chebyshev of this degree on the current coarse operator, preconditioned by the frozen
#: hierarchy, over the window ``[top / COARSE_WINDOW, 1.1 top]`` of the preconditioned operator.
COARSE_CHEBYSHEV_DEGREE = 4
COARSE_WINDOW = 10.0


def coarse_solver(data: SemData, levels=None):
    """``rc -> ≈ A_c⁻¹ rc``: an LU of the dense ``A_c`` (small), or, with an AMG ``levels`` hierarchy
    (:func:`amg.build_hierarchy` of the coarse operator at the reference state, frozen), Chebyshev iteration on the
    CURRENT ``A_c`` preconditioned by one frozen V-cycle, its window from power iterations on this build's own
    preconditioned operator.

    The frozen hierarchy alone is the inverse of another operator: a Newton iterate's coarse operator was ten
    times stiffer than the reference one, and the V-cycle's error propagator ``I - M A_c`` had spectral radius
    3.1e7 there (with only the finest level's window re-measured, 6.1). Chebyshev on ``M A_c`` is a fixed linear
    map, symmetric where they are, and contracts while the two operators are spectrally equivalent."""
    if levels is None:
        lu = jax.scipy.linalg.lu_factor(data.coarse_dense())
        return lambda rc: jax.scipy.linalg.lu_solve(lu, rc)
    from .amg import vcycle_apply
    from .krylov import chebyshev_apply, power_iteration_bound

    A0 = data.coarse_matrix()

    def mv(v):
        return A0 @ v

    def M(r):
        return vcycle_apply(levels, r)

    hi = 1.1 * power_iteration_bound(mv, data.st.n_coarse, dtype=data.coarse_data.dtype, iters=15, M=M)
    lo = hi / COARSE_WINDOW
    return lambda rc: chebyshev_apply(mv, rc, lmin=lo, lmax=hi, degree=COARSE_CHEBYSHEV_DEGREE, M=M)


def coarse_levels(data: SemData):
    """The frozen AMG hierarchy for ``data``'s coarse operator, or ``None`` when it is solved densely. Host-side:
    ``data`` must be concrete."""
    if data.st.n_coarse <= DENSE_COARSE_MAX:
        return None
    from .amg import build_hierarchy

    A = data.coarse_matrix()
    if isinstance(A.data, jax.core.Tracer):
        raise TypeError("coarse_levels: the coarse operator is traced; build the hierarchy from a concrete state")
    return build_hierarchy(A)


def coarse_correction(data: SemData, levels=None):
    """``r -> P̂ A_c⁻¹ P̂ᵀ r``."""
    solve = coarse_solver(data, levels)
    m = data.mask

    def apply(r):
        r = jnp.asarray(r)
        mr = m.astype(r.dtype)
        return mr * (data.st.P @ solve(data.st.Pt @ (mr * r)))

    return apply


def sem_data(jacobian, u, mask, pin, t=0.0, args=None) -> SemData:
    """:class:`SemData` of the assembler's ``jacobian`` (``fem_native``) at the state ``u``."""
    st = jacobian.sem_statics()
    out = jacobian(jnp.asarray(u), t, args, sem=(jnp.asarray(mask), jnp.asarray(st.phi_local)))
    if set(out) != {0}:
        raise ValueError(f"the spectral-element preconditioners need the volume terms of one field; found {sorted(out)}")
    coeffs, Ce = out[0]
    return SemData(st, coeffs, Ce, mask, pin)
