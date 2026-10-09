"""Reduced-order modelling for families of linear FEM problems: POD bases, geometry-mapped bases, and an
energy-controlled Galerkin solve with Krylov enrichment.

The setting: many SPD systems ``A(p) u = B(p)`` on ONE fixed mesh (a design sweep, a surrogate, an inverse loop),
where only a few scalar outputs are needed -- typically the compliance / energy matrix ``C = B^T A^-1 B`` (for a
magnetostatic problem with unit currents this is the inductance matrix, for elasticity the stiffness seen by the
loads). ``fem.solve(basis=U)`` already reduces ONE solve onto a given basis; this module covers the loop around it:

    S          = rom.unit_energy(U_snap, B_snap)          # snapshots scaled to unit energy  u / sqrt(b.u)
    V, sv      = rom.pod(S, k=128)                       # proper orthogonal decomposition (thin SVD)
    G          = rom.Galerkin(V)                         # per-basis set-up (device arrays)
    r          = G.bind(A, B).solve(tol=1e-4)            # per design: reduced solve + adaptive enrichment
    r.C, r.n_enrich, r.increment                         # B^T u_h, Krylov steps used, last relative increase

Why it is reliable (and when it is not): for SPD ``A`` the Galerkin solution is the energy-optimal one in the span,
so ``diag(C_h)`` is a LOWER bound that increases monotonically as the span grows. ``solve(tol=...)`` exploits this:
it adds Jacobi-preconditioned residual directions ``D^-1 (B - A u_h)`` (one per right-hand side) -- a POD-deflated,
CG-type iteration -- and stops when ``diag(C_h)`` rises by less than ``tol`` (relative) over ``window`` steps. A
familiar design stops after a few steps; an unfamiliar one (a parameter regime or geometry the snapshots never saw)
keeps enriching towards the full solution instead of returning a plausible wrong answer. The stopping test is on
the quantity of interest, not on the residual: a POD answer can have a large full-system residual (the
``fem.solve(basis=)`` check would reject it) while its energy is already correct to 1e-4.

Geometry changes on a fixed mesh: ``LandmarkMap`` (separable, monotone, piecewise-linear stretch through landmark
coordinates) + :func:`jno.utils.solver.fem_adapt.interpolation_matrix` give sparse ``pull``/``push`` matrices that
move snapshots onto a reference configuration and the reference modes back (``mapped_basis``), so one basis serves
geometries that were never in the snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["unit_energy", "pod", "orthonormal_union", "LandmarkMap", "map_matrix", "Galerkin", "ReducedResult"]


# ------------------------------------------------------------------------------------------------------- snapshots
def unit_energy(U, B):
    """Scale snapshot fields to unit energy: columns ``u_j / sqrt(b_j . u_j)``.

    ``U`` (n, m) solutions, ``B`` (n, m) their right-hand sides (or (n,) / (m, n) -- matched by shape). Large and
    small responses then weigh equally in the POD, which is what a relative error target on the energy needs."""
    U = np.asarray(U, np.float64)
    B = np.asarray(B, np.float64)
    if U.ndim == 1:
        return U / np.sqrt(float(B.ravel() @ U))
    if B.shape != U.shape and B.T.shape == U.shape:
        B = B.T
    e = np.einsum("nm,nm->m", B, U)
    if np.any(e <= 0):
        raise ValueError("unit_energy: b.u <= 0 for some snapshot (A not SPD, or zero right-hand side).")
    return U / np.sqrt(e)


def pod(S, k: Optional[int] = None, *, energy: Optional[float] = None, device: bool = True):
    """Proper orthogonal decomposition of a snapshot matrix ``S`` (n, m): thin SVD, float64.

    Returns ``(V, sv)``: the leading modes (n, k) (orthonormal columns, ordered by energy) and ALL singular values.
    ``k`` modes, or the fewest modes capturing ``1 - energy`` of ``sum(sv**2)`` (``energy=1e-8`` keeps 1 - 1e-8).
    ``device=True`` runs ``jax.numpy.linalg.svd`` (GPU when present), else numpy. For snapshot sets too large for a
    dense SVD use the matrix-free :func:`jno.solve.svd` (Golub-Kahan) instead."""
    if device:
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        Phi, sv, _ = jnp.linalg.svd(jnp.asarray(np.asarray(S, np.float64)), full_matrices=False)
        Phi, sv = np.asarray(Phi), np.asarray(sv)
    else:
        Phi, sv, _ = np.linalg.svd(np.asarray(S, np.float64), full_matrices=False)
    if energy is not None:
        cum = np.cumsum(sv**2) / np.sum(sv**2)
        k = int(np.searchsorted(cum, 1.0 - energy) + 1)
    k = len(sv) if k is None else min(int(k), len(sv))
    return Phi[:, :k], sv


def orthonormal_union(*bases, interleave: bool = True):
    """Orthonormal basis of the span of several bases (n, k_i): QR of their concatenation.

    ``interleave=True`` orders the columns v1_1, v2_1, v1_2, v2_2, ... before the QR, so the first 2j columns of the
    result span the first j modes of each basis (nested truncation stays meaningful)."""
    bases = [np.asarray(b, np.float64) for b in bases]
    if interleave and len({b.shape[1] for b in bases}) == 1:
        W = np.stack(bases, 2).reshape(bases[0].shape[0], -1)
    else:
        W = np.concatenate(bases, 1)
    return np.linalg.qr(W)[0]


# -------------------------------------------------------------------------------------------------- geometry maps
class LandmarkMap:
    """Separable, monotone, piecewise-linear map ``x' = f_x(x), y' = f_y(y) (, z' = f_z(z))``.

    ``src`` and ``dst`` are per-axis increasing knot sequences (same length per axis): ``src[a][i] -> dst[a][i]``,
    linear in between, constant extension outside. Used to stretch one geometry onto another through matching
    landmarks (interfaces, gap positions, winding edges) while the mesh stays fixed. ``inverse()`` swaps them."""

    def __init__(self, src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]]):
        self.src = [np.asarray(s, np.float64) for s in src]
        self.dst = [np.asarray(d, np.float64) for d in dst]
        for s, d in zip(self.src, self.dst):
            if s.shape != d.shape or np.any(np.diff(s) <= 0) or np.any(np.diff(d) <= 0):
                raise ValueError("LandmarkMap: knots must be strictly increasing and pairwise of equal length.")

    def __call__(self, points):
        P = np.asarray(points, np.float64)
        out = P.copy()
        for a, (s, d) in enumerate(zip(self.src, self.dst)):
            out[:, a] = np.interp(P[:, a], s, d)
        return out

    def inverse(self) -> "LandmarkMap":
        return LandmarkMap(self.dst, self.src)


def map_matrix(domain: Any, F, *, fill: Any = "nearest", tol: float = 1e-9, k: int = 32):
    """Sparse ``P`` with ``(P @ u)[i] = u(F(x_i))`` on the mesh nodes ``x_i``: composition of a nodal P1 field with
    a point map ``F`` (e.g. a :class:`LandmarkMap`). ``P @ U`` maps a whole block of fields.

    With ``F = M`` mapping reference -> physical, ``map_matrix(d, M)`` PULLS a physical field back to the
    reference configuration (``u o M``), and ``map_matrix(d, M.inverse())`` PUSHES a reference field forward."""
    from .utils.solver.fem_adapt import interpolation_matrix

    if isinstance(domain, (tuple, list)):
        pts = np.asarray(domain[0], np.float64)
        dim = np.asarray(domain[1]).shape[1] - 1
    else:
        dim = int(domain.dimension)
        mesh = domain.built_mesh if hasattr(domain, "built_mesh") else domain.mesh
        pts = np.asarray(mesh.points, np.float64)
    pts = pts[:, :dim]
    lo, hi = pts.min(0), pts.max(0)
    Q = np.clip(F(pts), lo, hi)
    return interpolation_matrix(domain, Q, fill=fill, tol=tol, k=k)


# ------------------------------------------------------------------------------------------------ Galerkin solve
@dataclass
class ReducedResult:
    C: np.ndarray  # (r, r) B^T u_h: the reduced compliance / energy matrix (lower bound on the diagonal)
    coeffs: Any  # (m + enrich, r) coefficients in the (enriched) basis
    n_enrich: int  # Krylov enrichment steps used (fields added per right-hand side)
    increment: float  # last relative increase of diag(C) over the stopping window (nan for fixed enrichment)
    basis: Any = None  # the enriched basis (device array), for lifting: u_h = basis @ coeffs

    def fields(self):
        """Lift to the full space: u_h (n, r)."""
        return np.asarray(self.basis @ self.coeffs)


def _to_bcoo(A, n):
    import jax.numpy as jnp
    from jax.experimental import sparse as jsp

    if isinstance(A, jsp.BCOO):
        return jsp.BCOO((jnp.asarray(A.data, jnp.float64), jnp.asarray(A.indices)), shape=(n, n))
    try:
        import scipy.sparse as sp

        if sp.issparse(A):
            return jsp.BCOO.from_scipy_sparse(sp.coo_matrix(A, dtype=np.float64))
    except ImportError:
        pass
    if hasattr(A, "todense") or hasattr(A, "shape"):
        return jsp.BCOO.fromdense(jnp.asarray(np.asarray(A, np.float64)))
    raise TypeError(f"Galerkin.bind: unsupported matrix type {type(A).__name__}")


def _diagonal(A):
    import jax.numpy as jnp

    idx, data = A.indices, A.data
    return jnp.zeros(A.shape[0], data.dtype).at[idx[:, 0]].add(jnp.where(idx[:, 0] == idx[:, 1], data, 0.0))


class Galerkin:
    """Galerkin reduction on a fixed orthonormal basis ``W`` (n, m), float64 on the default JAX device.

    ``bind(A, B)`` forms ``A W``, ``W^T A W``, ``W^T B`` once per operator (one sparse mat-mat product);
    ``solve(...)`` then costs dense m x m work plus one sparse mat-vec block per enrichment step. Bases are
    nested: ``solve(m=...)`` uses the first m columns."""

    def __init__(self, W):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        self.jnp = jnp
        self.W = jnp.asarray(np.asarray(W, np.float64))
        self.n, self.m = self.W.shape

    def bind(self, A, B):
        """``A`` (n, n) SPD: jax BCOO (as assembled by ``jno.fem``), scipy sparse or dense. ``B`` (n, r) or (r, n)."""
        jnp = self.jnp
        self.A = _to_bcoo(A, self.n)
        B = jnp.asarray(np.asarray(B, np.float64))
        self.B = B if B.shape[0] == self.n else B.T
        self.AW = self.A @ self.W
        self.G = self.W.T @ self.AW
        self.R = self.W.T @ self.B
        self.dinv = 1.0 / _diagonal(self.A)
        return self

    def solve(
        self,
        m: Optional[int] = None,
        *,
        enrich: Optional[int] = None,
        tol: Optional[float] = 1e-4,
        k_max: int = 64,
        window: int = 2,
        jit: bool = False,
    ) -> ReducedResult:
        """Reduced solve on the first ``m`` basis columns.

        ``enrich=k``: exactly k Jacobi-Krylov blocks ``D^-1 (B - A u_h), D^-1 A (...), ...`` (fixed cost).
        Otherwise (default) error-controlled: add ``D^-1 (B - A u_h)`` of the CURRENT reduced answer, orthonormalised
        against the span, re-solve, and stop when ``max_i (C_ii - C_ii[-window]) / C_ii < tol`` or after ``k_max``
        steps. ``tol=None`` with ``enrich=None`` is the plain Galerkin answer.

        ``jit=True`` runs the error-controlled loop as ONE compiled ``lax.while_loop`` on fixed-size buffers
        (``m + k_max * r`` columns, unused ones masked): no per-step dispatch or host sync, no recompilation when the
        basis grows, but every step works on the padded size. Same answer as the eager loop (default) to round-off;
        which is faster depends on the device (padding costs on CPU, dispatch costs on GPU) -- measure."""
        jnp = self.jnp
        m = m or self.m
        if jit and not enrich and tol is not None:
            C, c, k, q, W = _adaptive(
                self.A,
                self.B,
                self.dinv,
                self.W[:, :m],
                self.AW[:, :m],
                self.G[:m, :m],
                self.R[:m],
                float(tol),
                int(k_max),
                int(window),
            )
            k = int(k)
            return ReducedResult(np.asarray(C), c[: m + k * self.B.shape[1]], k, float(q), W[:, : m + k * self.B.shape[1]])
        W, AW, G, R = self.W[:, :m], self.AW[:, :m], self.G[:m, :m], self.R[:m]
        if enrich:
            Z = self.dinv[:, None] * (self.B - AW @ jnp.linalg.solve(G, R))
            blocks = []
            for _ in range(int(enrich)):
                blocks.append(Z)
                Z = self.dinv[:, None] * (self.A @ Z)
            K = jnp.linalg.qr(jnp.concatenate([W] + blocks, 1))[0][:, m:]
            AK = self.A @ K
            G = jnp.block([[G, AW.T @ K], [K.T @ AW, K.T @ AK]])
            R = jnp.concatenate([R, K.T @ self.B], 0)
            c = jnp.linalg.solve(G, R)
            return ReducedResult(np.asarray(R.T @ c), c, int(enrich), float("nan"), jnp.concatenate([W, K], 1))
        hist, k = [], 0
        while True:
            c = jnp.linalg.solve(G, R)
            C = np.asarray(R.T @ c)
            hist.append(np.diag(C).copy())
            q = float(np.max((hist[-1] - hist[-1 - window]) / hist[-1])) if len(hist) > window else np.inf
            if tol is None or q < tol or k >= k_max:
                return ReducedResult(C, c, k, q if tol is not None else float("nan"), W)
            Z = self.dinv[:, None] * (self.B - AW @ c)
            for _ in range(2):  # re-orthogonalise against the span, twice
                Z = Z - W @ (W.T @ Z)
            Z = jnp.linalg.qr(Z)[0]
            AZ = self.A @ Z
            G = jnp.block([[G, AW.T @ Z], [Z.T @ AW, Z.T @ AZ]])
            R = jnp.concatenate([R, Z.T @ self.B], 0)
            W, AW = jnp.concatenate([W, Z], 1), jnp.concatenate([AW, AZ], 1)
            k += 1


def _adaptive_impl(A, B, dinv, W0, AW0, G0, R0, tol, k_max, window):
    """Error-controlled enrichment on fixed-size buffers (see Galerkin.solve). Column block j >= 0 of the enrichment
    lives at m + j r ... m + (j+1) r; blocks not yet filled are zero in W, AW, G, R, with 1 on the diagonal of G so the
    padded system stays regular (their coefficients are exactly 0)."""
    import jax
    import jax.numpy as jnp

    n, m = W0.shape
    r = B.shape[1]
    P = m + k_max * r
    W = jnp.zeros((n, P), W0.dtype).at[:, :m].set(W0)
    AW = jnp.zeros((n, P), W0.dtype).at[:, :m].set(AW0)
    G = jnp.zeros((P, P), W0.dtype).at[:m, :m].set(G0)
    R = jnp.zeros((P, r), W0.dtype).at[:m].set(R0)
    hist = jnp.zeros((k_max + 1, r), W0.dtype)
    used0 = jnp.arange(P) < m

    def reduced(G, R, used):
        Gp = jnp.where(used[:, None] & used[None, :], G, 0.0) + jnp.diag(jnp.where(used, 0.0, 1.0))
        c = jnp.linalg.solve(Gp, R)
        return c, R.T @ c

    def cond(st):
        return ~st["done"]

    def body(st):
        k, W, AW, G, R, hist, used = st["k"], st["W"], st["AW"], st["G"], st["R"], st["hist"], st["used"]
        c, C = reduced(G, R, used)
        d = jnp.diag(C)
        hist = hist.at[k].set(d)
        prev = hist[jnp.maximum(k - window, 0)]
        q = jnp.where(k >= window, jnp.max((d - prev) / d), jnp.inf)
        done = (q < tol) | (k >= k_max)

        def enrich(args):
            W, AW, G, R, used = args
            Z = dinv[:, None] * (B - AW @ c)
            for _ in range(2):
                Z = Z - W @ (W.T @ Z)
            Z = jnp.linalg.qr(Z)[0]
            AZ = A @ Z
            s0 = m + k * r
            W = jax.lax.dynamic_update_slice(W, Z, (0, s0))
            col = AW.T @ Z  # (P, r): couplings with the existing span
            G = jax.lax.dynamic_update_slice(G, col, (0, s0))
            G = jax.lax.dynamic_update_slice(G, col.T, (s0, 0))
            G = jax.lax.dynamic_update_slice(G, Z.T @ AZ, (s0, s0))
            AW = jax.lax.dynamic_update_slice(AW, AZ, (0, s0))
            R = jax.lax.dynamic_update_slice(R, Z.T @ B, (s0, 0))
            used = used | ((jnp.arange(P) >= s0) & (jnp.arange(P) < s0 + r))
            return W, AW, G, R, used

        W, AW, G, R, used = jax.lax.cond(done, lambda a: a, enrich, (W, AW, G, R, used))
        return dict(k=jnp.where(done, k, k + 1), W=W, AW=AW, G=G, R=R, hist=hist, used=used, done=done, C=C, c=c, q=q)

    st = dict(
        k=jnp.asarray(0),
        W=W,
        AW=AW,
        G=G,
        R=R,
        hist=hist,
        used=used0,
        done=jnp.asarray(False),
        C=jnp.zeros((r, r), W0.dtype),
        c=jnp.zeros((P, r), W0.dtype),
        q=jnp.asarray(jnp.inf, W0.dtype),
    )
    st = jax.lax.while_loop(cond, body, st)
    return st["C"], st["c"], st["k"], st["q"], st["W"]


_ADAPTIVE_JIT = None


def _adaptive(A, B, dinv, W0, AW0, G0, R0, tol, k_max, window):
    global _ADAPTIVE_JIT
    if _ADAPTIVE_JIT is None:
        import jax

        _ADAPTIVE_JIT = jax.jit(_adaptive_impl, static_argnames=("k_max", "window"))
    return _ADAPTIVE_JIT(A, B, dinv, W0, AW0, G0, R0, tol, k_max=k_max, window=window)
