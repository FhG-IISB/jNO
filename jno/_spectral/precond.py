"""``jno.precond.spectral()``: the Fourier-mode block preconditioner of a spectral problem.

An operator whose coefficients do not vary along the Fourier axes -- boundary rows included, since a wall is the
same at every x -- does not couple different Fourier modes: it is block diagonal in them, one dense block per mode
over the polynomial nodes of every unknown and component (Dedalus solves channel problems this way). The blocks are
read by probing the operator with one column per polynomial node, at Fourier index 0, and transforming the response
along the Fourier axes; the preconditioner applies their (pseudo-)inverses mode by mode. It is the exact inverse
for such an operator, so GMRES converges in one or two iterations; for coefficients that vary along a Fourier axis it
is the inverse of the operator frozen at the probed position, a preconditioner.

Cost: ``B`` matvecs to build (``B`` = polynomial nodes x components x unknowns: 33 for a 64 x 33 channel, 2 for a
periodic ω/ψ box) and ``n_modes · B²`` memory; an all-polynomial problem is one block, i.e. a dense LU of the
whole operator, so ``max_block`` bounds ``B``.

Based on:
  Burns, K. J., Vasil, G. M., Oishi, J. S., Lecoanet, D. & Brown, B. P. (2020). Dedalus: A flexible framework for
  numerical simulations with spectral methods. Phys. Rev. Research 2, 023068 -- mode-decoupled sparse solves.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .bases import HIGHEST, Fourier

__all__ = ["SpectralPrecond", "materialize_for"]


class SpectralPrecond:
    """The spec ``jno.precond.spectral(...)`` returns; built for an operator by :meth:`build`."""

    name = "spectral"

    def __init__(self, max_block: int = 4096, rcond: float = 1e-12):
        self.max_block, self.rcond = int(max_block), float(rcond)

    def __repr__(self):
        return f"jno.precond.spectral(max_block={self.max_block})"

    def build(self, problem, mv):
        """``r -> M⁻¹ r`` for the linear map ``mv`` over ``problem``'s DOF vector: the Fourier-mode blocks when the
        problem has Fourier axes and the blocks fit ``max_block``, else fast diagonalisation (:func:`separable`)."""
        try:
            return self._mode_blocks(problem, mv)
        except NotImplementedError:
            return separable(problem, mv)

    def _mode_blocks(self, problem, mv):
        grids = [problem._grid_of(u) for u in problem.unknowns]
        fax = [i for i, a in enumerate(grids[0].axes) if isinstance(a, Fourier)]
        for g in grids:
            f = [i for i, a in enumerate(g.axes) if isinstance(a, Fourier)]
            if f != fax or any(g.axes[i] != grids[0].axes[i] for i in fax):
                raise NotImplementedError(
                    "jno.precond.spectral(): every unknown needs the same Fourier axes (same nodes) to decouple modes."
                )
        if not fax:
            raise NotImplementedError("jno.precond.spectral(): no Fourier axis to decouple modes along.")
        nd = grids[0].ndim
        pax = [i for i in range(nd) if i not in fax]
        fshape = tuple(grids[0].shape[i] for i in fax)
        n_modes = int(np.prod(fshape)) if fshape else 1
        # one block column per (unknown, component, polynomial node)
        cols = []
        for k, (g, o, c) in enumerate(zip(grids, problem._offsets, problem._ncomp)):
            pshape = tuple(g.shape[i] for i in pax)
            idx = np.arange(g.size).reshape(g.shape)
            sl = tuple(0 if i in fax else slice(None) for i in range(nd))
            nodes = idx[sl].reshape(-1)  # the nodes at Fourier index 0, every polynomial node
            for comp in range(c):
                cols.extend(int(o + comp * g.size + p) for p in nodes)
            del pshape
        B = len(cols)
        if B > self.max_block:
            raise NotImplementedError(
                f"jno.precond.spectral(): a mode block of {B} unknowns exceeds max_block={self.max_block} (an all-"
                "polynomial problem is one block); raise max_block or use fewer polynomial nodes."
            )
        n = problem._Ntot

        def to_modes(v):
            """``(n,)`` DOF vector -> ``(n_modes, B)`` Fourier coefficients along the Fourier axes."""
            parts = []
            for g, o, c in zip(grids, problem._offsets, problem._ncomp):
                blk = v[o : o + c * g.size].reshape((c,) + g.shape)
                blk = jnp.fft.fftn(blk, axes=tuple(1 + i for i in fax)) if fax else blk.astype(jnp.result_type(blk, 1j))
                blk = jnp.moveaxis(blk, [1 + i for i in fax], list(range(1, 1 + len(fax))))
                parts.append(blk.reshape(c, n_modes, -1).transpose(1, 0, 2).reshape(n_modes, -1))
            return jnp.concatenate(parts, axis=1)

        def from_modes(m):
            out, col = [], 0
            for g, c in zip(grids, problem._ncomp):
                pn = int(np.prod([g.shape[i] for i in pax])) if pax else 1
                blk = m[:, col : col + c * pn].reshape(n_modes, c, pn).transpose(1, 0, 2)
                col += c * pn
                blk = blk.reshape((c,) + fshape + tuple(g.shape[i] for i in pax))
                blk = jnp.moveaxis(blk, list(range(1, 1 + len(fax))), [1 + i for i in fax])
                if fax:
                    blk = jnp.fft.ifftn(blk, axes=tuple(1 + i for i in fax))
                out.append(blk.real.reshape(-1))
            return jnp.concatenate(out)

        with jax.ensure_compile_time_eval():
            E = jax.nn.one_hot(jnp.asarray(cols), n, dtype=jnp.result_type(float))
            resp = jax.lax.map(mv, E, batch_size=min(B, 64))  # (B, n): A e_j
            blocks = jax.vmap(to_modes)(resp)  # (B, n_modes, B): [col, mode, row]
            blocks = jnp.transpose(blocks, (1, 2, 0))  # (n_modes, rows, cols)
            inv = jnp.linalg.pinv(blocks, rtol=self.rcond)

        def apply(r):
            m = to_modes(jnp.asarray(r))
            return from_modes(jnp.einsum("kij,kj->ki", inv, m, precision=HIGHEST))

        return apply


def separable(problem, mv):
    """Fast diagonalisation (Lynch, Rice & Thomas 1964; Haidvogel & Zang 1979) of one scalar unknown's equation.

    The equation's constant-coefficient derivative terms along one axis each (``u.xx``, ``u.y``, ``c u``) form a
    Kronecker sum ``Σ_i I ⊗ A_i ⊗ I`` of 1-D operators. On the nodes no boundary row sits on (a tensor sub-grid)
    it is inverted by the eigendecompositions of the 1-D operators restricted to those nodes, ``O(N^(d+1))``; the
    boundary rows are taken as identity rows and their coupling into the interior is removed by one matvec of the
    true operator. That is the exact inverse for a separable constant-coefficient problem with Dirichlet rows;
    with a variable coefficient, a mixed derivative or a Neumann row it is a preconditioner.

    Raises ``NotImplementedError`` when it does not apply (several unknowns or components, no such terms, boundary
    rows that are not whole faces).
    """
    from .etd import _classify_terms

    if len(problem.unknowns) != 1 or problem._ncomp[0] != 1:
        raise NotImplementedError("fast diagonalisation: one scalar unknown only.")
    row = problem._rows[0]
    G = problem._grid_of(problem.unknowns[0])
    if row["grid"] != G or not row["replace"]:
        raise NotImplementedError("fast diagonalisation: the equation must hold on its unknown's nodes.")
    _m, lin, _rest = _classify_terms(row["eq"], problem.unknowns[0], G.ndim)
    nd = G.ndim
    ops = [np.zeros((a.n, a.n)) for a in G.axes]
    used = False
    for c, alpha in lin:
        axes = [i for i, k in enumerate(alpha) if k]
        if len(axes) > 1 or np.iscomplexobj(c):
            continue  # a mixed derivative: not separable, left to GMRES
        i = axes[0] if axes else 0
        ops[i] = ops[i] + float(c) * (np.eye(G.axes[i].n) if not axes else np.asarray(G.axes[i].diff_matrix(alpha[i])))
        used = used or bool(axes)
    if not used:
        raise NotImplementedError("fast diagonalisation: the equation has no constant-coefficient derivative term.")
    # boundary rows: whole faces of polynomial axes, which leaves a tensor sub-grid of interior nodes
    bset = np.zeros(G.size, dtype=bool)
    for _kind, _c, idx in row["bcs"]:
        bset[idx] = True
    for _g, i in row["gauges"]:
        raise NotImplementedError("fast diagonalisation: a gauge row.")
    keep = []
    for i, a in enumerate(G.axes):
        lo = hi = False
        if not a.periodic:
            lo = bool(np.all(bset[G.face(i, 0)])) if np.isclose(a.nodes[0], a.a) else False
            hi = bool(np.all(bset[G.face(i, 1)])) if np.isclose(a.nodes[-1], a.b) else False
        keep.append(np.arange(int(lo), a.n - int(hi)))
    interior = np.zeros(G.shape, dtype=bool)
    interior[np.ix_(*keep)] = True
    interior = interior.reshape(-1)
    if np.any(interior & bset) or np.any(~interior & ~bset):
        raise NotImplementedError("fast diagonalisation: the boundary rows are not whole faces.")
    eig = []
    for i, k in enumerate(keep):
        A = ops[i][np.ix_(k, k)]
        lam, V = np.linalg.eig(A)
        eig.append((lam, V, np.linalg.inv(V)))
    denom = sum(lam.reshape([lam.size if j == i else 1 for j in range(nd)]) for i, (lam, _V, _Vi) in enumerate(eig))
    small = np.abs(denom) <= 1e-12 * np.abs(denom).max()
    inv_denom = jnp.asarray(np.where(small, 0.0, 1.0 / np.where(small, 1.0, denom)))
    cplx = any(np.iscomplexobj(lam) and np.abs(lam.imag).max() > 0 for lam, _V, _Vi in eig)
    mats = [(jnp.asarray(V if cplx else V.real), jnp.asarray(Vi if cplx else Vi.real)) for _lam, V, Vi in eig]
    if not cplx:
        inv_denom = inv_denom.real
    int_shape = tuple(k.size for k in keep)
    I_idx = jnp.asarray(np.flatnonzero(interior))
    B_idx = jnp.asarray(np.flatnonzero(bset))

    def kron_solve(rhs):
        X = rhs.reshape(int_shape)
        for i, (_V, Vi) in enumerate(mats):
            X = jnp.moveaxis(jnp.tensordot(Vi, jnp.moveaxis(X, i, 0), axes=(1, 0), precision=HIGHEST), 0, i)
        X = X * inv_denom
        for i, (V, _Vi) in enumerate(mats):
            X = jnp.moveaxis(jnp.tensordot(V, jnp.moveaxis(X, i, 0), axes=(1, 0), precision=HIGHEST), 0, i)
        return (X.real if cplx else X).reshape(-1)

    def apply(r):
        r = jnp.asarray(r)
        pb = jnp.zeros_like(r).at[B_idx].set(r[B_idx])
        rhs = r[I_idx] - mv(pb)[I_idx]
        return pb.at[I_idx].set(kron_solve(rhs))

    return apply


def materialize_for(precond, problem, mv):
    if isinstance(precond, SpectralPrecond):
        return precond.build(problem, mv)
    raise NotImplementedError(
        f"jno.spectral: precond={precond!r} is not available for a spectral problem; use jno.precond.spectral()."
    )
