"""``jno.rom`` — reduced-order models for a family of linear SPD problems on one mesh.

A design sweep, a surrogate or an inverse loop solves ``A(p) u = B(p)`` for many parameters ``p`` on ONE mesh and
usually needs only a few outputs, the energy matrix ``C = B^T A^-1 B`` above all (the inductance matrix of a
magnetostatic problem with unit currents, the compliance of an elastic one). Offline, full solves of a set of
designs are compressed into a basis; online, every new design is solved on that basis::

    S = jno.rom.unit_energy(U, B)                      # snapshots u / sqrt(b . u)
    W, sv = jno.rom.pod(S, k=512)                       # proper orthogonal decomposition
    rom = jno.rom.galerkin(W, enrich=8, gate=3e-3)      # a pytree: pass it into jit / vmap as an argument
    r = rom(A, B)                                       # A (n, n) BCOO from jno.fem, B (n, r)
    r.C, r.eta, r.enriched                              # energy matrix, error indicator, enrichment used
    u = rom.lift(r, w)                                  # the field for load weights w

For an SPD ``A`` the Galerkin answer is the best one in the span measured in energy, so ``diag(C)`` is a LOWER
bound of the exact value and its error is quadratic in the field error. The indicator
``eta = max_i sqrt(r_i^T D^-1 r_i / C_ii)`` (residual ``r = B - A u_h``, ``D`` the diagonal of ``A``) estimates
the relative energy error. When it exceeds ``gate``, the span is enriched by Jacobi-preconditioned residual
directions of the current answer and re-solved, which recovers parameters the snapshots never covered. The
background is the reduced-basis literature (Quarteroni, Manzoni & Negri, *Reduced Basis Methods for Partial
Differential Equations*, Springer 2016) and the snapshot POD (Sirovich, Q. Appl. Math. 45, 1987).

Everything is JAX and runs on the default device. Precision is JAX's: enable ``jax_enable_x64``, a reduced system of
a few hundred modes is ill-conditioned in float32. For ONE solve on a given basis without error control,
``fem.solve(basis=W)`` is the shorter route; this module is the loop around many solves.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["unit_energy", "pod", "galerkin", "Galerkin", "Reduced"]


def unit_energy(U: jax.Array, B: jax.Array) -> jax.Array:
    """Scale snapshot fields to unit energy, ``u_j / sqrt(b_j . u_j)``.

    Large and small responses then weigh equally in the POD, which is what a relative error target on the energy
    needs. Traceable; the SPD check below runs only on concrete values.

    Args:
        U: ``(n, m)`` solutions (or one ``(n,)`` solution).
        B: Their right-hand sides, ``(n, m)`` or ``(m, n)``.

    Returns:
        ``(n, m)`` scaled snapshots.

    Raises:
        ValueError: ``b . u <= 0`` for a snapshot (``A`` not SPD, or a zero right-hand side).
    """
    U, B = jnp.asarray(U), jnp.asarray(B)
    if U.ndim == 1:
        return U / jnp.sqrt(B.ravel() @ U)
    if B.shape != U.shape and B.T.shape == U.shape:
        B = B.T
    e = jnp.einsum("nm,nm->m", B, U)
    if not isinstance(e, jax.core.Tracer) and bool(jnp.any(e <= 0)):
        raise ValueError("jno.rom.unit_energy: b . u <= 0 for a snapshot (A not SPD, or a zero right-hand side).")
    return U / jnp.sqrt(e)


def _svd(S):
    """Thin SVD on the default device; on the CPU device when that fails (e.g. GPU memory held elsewhere)."""
    try:
        Phi, sv, _ = jnp.linalg.svd(jnp.asarray(S), full_matrices=False)
        return Phi, sv
    except Exception as e:
        cpu = jax.devices("cpu")[0]
        warnings.warn(
            f"jno.rom.pod: the SVD on {jax.devices()[0].platform} failed ({type(e).__name__}); using the CPU.", stacklevel=3
        )
        with jax.default_device(cpu):
            Phi, sv, _ = jnp.linalg.svd(jax.device_put(jnp.asarray(S), cpu), full_matrices=False)
        return Phi, sv


def _keep(sv, energy):
    """Fewest modes whose singular values carry all but ``energy`` of the total."""
    cum = jnp.cumsum(sv**2) / jnp.sum(sv**2)
    return int(jnp.searchsorted(cum, 1.0 - energy)) + 1


def pod(
    S: jax.Array | Sequence[jax.Array],
    k: int | None = None,
    *,
    energy: float | None = None,
    block_energy: float = 1e-14,
    block_max: int = 1200,
) -> tuple[jax.Array, jax.Array]:
    """Proper orthogonal decomposition of a snapshot matrix: its leading left singular vectors.

    A LIST of snapshot blocks (one per design family, say) is compressed block by block first, each to the modes
    carrying all but ``block_energy`` of its energy (at most ``block_max``), and the POD of
    ``[Phi_1 Sigma_1, Phi_2 Sigma_2, ...]`` then gives the modes of the whole set without one SVD of all
    snapshots. The result equals the SVD of the concatenation up to the dropped energy.

    The SVD is ``jax.numpy.linalg.svd`` on the default device (the CPU if that fails), in the working precision:
    enable ``jax_enable_x64``, the trailing singular values of a POD sit far below float32 round-off. The number of
    modes depends on the data, so ``pod`` itself is not traced. For snapshot sets too large for a dense SVD,
    :func:`jno.solve.svd` gives the leading modes matrix-free.

    Args:
        S: ``(n, m)`` snapshots, or a list of ``(n, m_i)`` blocks.
        k: Modes kept (all by default).
        energy: Instead of ``k``: keep the fewest modes that capture ``1 - energy`` of ``sum(sv**2)``.
        block_energy: Energy dropped per block (list input only).
        block_max: Most modes kept per block (list input only).

    Returns:
        ``(W, sv)``: the ``(n, k)`` orthonormal modes ordered by energy, and every singular value.
    """
    if isinstance(S, (list, tuple)):
        red = []
        for Sb in S:
            Phi, sv = _svd(Sb)
            kb = min(_keep(sv, block_energy), sv.shape[0], block_max)
            red.append(Phi[:, :kb] * sv[:kb])
        S = jnp.concatenate(red, 1)
    Phi, sv = _svd(S)
    if energy is not None:
        k = _keep(sv, energy)
    k = sv.shape[0] if k is None else min(int(k), sv.shape[0])
    return Phi[:, :k], sv


class Reduced(NamedTuple):
    """The reduced answer of :class:`Galerkin` (a pytree, so it passes through ``jit`` and ``vmap``)."""

    C: jax.Array
    """``(r, r)`` energy matrix ``B^T u_h``; its diagonal is a lower bound of the exact one."""
    c: jax.Array
    """``(m + enrich * r, r)`` coefficients in the span ``[W | Z_1 ... Z_enrich]``."""
    Z: jax.Array
    """``(enrich, n, r)`` enrichment directions (zeros when the enrichment did not run)."""
    eta: jax.Array
    """Error indicator of the plain POD answer: the relative energy error is about ``eta**2``."""
    enriched: jax.Array
    """Whether the enrichment ran."""
    dC: jax.Array
    """``max_i |C_ii / C0_ii - 1|``: how much the enrichment changed the diagonal of the plain answer ``C0``."""


def _diagonal(A):
    idx, data = A.indices, A.data
    return jnp.zeros(A.shape[0], data.dtype).at[idx[:, 0]].add(jnp.where(idx[:, 0] == idx[:, 1], data, 0.0))


def _enrich(A, B, dinv, W, AW, G, R, k, f32, r0):
    """``k`` residual steps on top of ``W``, unrolled. ``W`` and ``A W`` are only read; the new directions are
    kept as separate ``(n, r)`` blocks, orthonormal among themselves (the reduced system stays SPD)."""
    lo = jnp.float32 if f32 else W.dtype
    AWl, Bl, dl = AW.astype(lo), B.astype(lo), dinv.astype(lo)
    m, r = W.shape[1], B.shape[1]
    Q, Ql, AQl = [], [], []
    c = jnp.linalg.solve(G, R)
    for j in range(k):
        if j == 0:
            Z = r0.astype(lo)
        else:
            cl = c.astype(lo)
            Z = Bl - AWl @ cl[:m]
            for i, AQi in enumerate(AQl):
                Z = Z - AQi @ cl[m + i * r : m + (i + 1) * r]
        Z = dl[:, None] * Z
        for _ in range(2):  # twice: one Gram-Schmidt pass loses orthogonality in float32
            for Qi in Ql:
                Z = Z - Qi @ (Qi.T @ Z)
        Z = jnp.linalg.qr(Z)[0]
        Z64 = Z.astype(W.dtype)
        AZ = A @ Z64
        col = jnp.concatenate([W.T @ AZ] + [Qi.T @ AZ for Qi in Q], 0)
        G = jnp.block([[G, col], [col.T, Z64.T @ AZ]])
        R = jnp.concatenate([R, Z64.T @ B], 0)
        Q.append(Z64)
        Ql.append(Z)
        AQl.append(AZ.astype(lo))
        c = jnp.linalg.solve(G, R)
    return R.T @ c, c, jnp.stack(Q)


@jax.tree_util.register_pytree_node_class
class Galerkin:
    """Galerkin projection onto a fixed orthonormal basis, with gated residual enrichment. Build it with
    :func:`galerkin`.

    A call costs one sparse product ``A W``, dense ``m x m`` algebra and, when it enriches, one sparse product
    with an ``(n, r)`` block per step. The object is a pytree whose basis is a leaf: pass it into a jitted or
    vmapped function as an ARGUMENT, so ``W`` is data rather than a constant compiled into the program.
    """

    def __init__(self, W, *, enrich: int = 0, gate: float | None = None, float32: bool = True, free=None):
        self.W = jnp.asarray(W)
        self.free = None if free is None else jnp.asarray(free, bool)
        self.enrich, self.gate, self.float32 = int(enrich), gate, bool(float32)

    def tree_flatten(self):
        return (self.W, self.free), (self.enrich, self.gate, self.float32)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.W, obj.free = children
        obj.enrich, obj.gate, obj.float32 = aux
        return obj

    @property
    def m(self) -> int:
        """Number of basis vectors."""
        return self.W.shape[1]

    def __call__(self, A: jax.Array, B: jax.Array) -> Reduced:
        """Solve one system on the basis.

        Args:
            A: ``(n, n)`` SPD operator as a ``jax.experimental.sparse.BCOO`` (what ``jno.fem`` assembles).
            B: ``(n, r)`` right-hand sides. Rows outside ``free`` are ignored.

        Returns:
            The :class:`Reduced` answer.
        """
        W = self.W
        B = jnp.asarray(B, W.dtype)
        if self.free is not None:
            B = jnp.where(self.free[:, None], B, 0.0)
        AW = A @ W
        G0, R0 = W.T @ AW, W.T @ B
        c0 = jnp.linalg.solve(G0, R0)
        C0 = R0.T @ c0
        d = _diagonal(A)
        ok = d > 0 if self.free is None else self.free & (d > 0)
        dinv = jnp.where(ok, 1.0 / jnp.where(d > 0, d, 1.0), 0.0)
        r0 = B - AW @ c0
        eta = jnp.sqrt(jnp.max(jnp.sum(dinv[:, None] * r0**2, 0) / jnp.diag(C0)))
        k, n, r = self.enrich, W.shape[0], B.shape[1]
        if not k:
            return Reduced(C0, c0, jnp.zeros((0, n, r), W.dtype), eta, jnp.asarray(False), jnp.zeros((), W.dtype))

        def enriched(_):
            return _enrich(A, B, dinv, W, AW, G0, R0, k, self.float32, r0)

        def plain(_):
            return C0, jnp.concatenate([c0, jnp.zeros((k * r, r), c0.dtype)], 0), jnp.zeros((k, n, r), c0.dtype)

        if self.gate is None:
            on = jnp.asarray(True)
            C, c, Z = enriched(None)
        else:
            on = eta > self.gate
            C, c, Z = jax.lax.cond(on, enriched, plain, None)
        return Reduced(C, c, Z, eta, on, jnp.max(jnp.abs(jnp.diag(C) / jnp.diag(C0) - 1)))

    def lift(self, res: Reduced, w: jax.Array | None = None) -> jax.Array:
        """The reduced field on the full mesh.

        Args:
            res: A :class:`Reduced` answer of this object.
            w: ``(r,)`` load weights; ``None`` lifts every right-hand side.

        Returns:
            ``u_h w`` as an ``(n,)`` array, or ``u_h`` as ``(n, r)``.
        """
        c = res.c if w is None else res.c @ w
        u = self.W @ c[: self.m]
        k = res.Z.shape[0]
        if k:
            cz = c[self.m :].reshape((k, res.Z.shape[2]) + c.shape[1:])
            u = u + jnp.einsum("knr,kr...->n...", res.Z, cz)
        return u


def galerkin(
    W: jax.Array | np.ndarray,
    *,
    enrich: int = 0,
    gate: float | None = None,
    float32: bool = True,
    free: jax.Array | np.ndarray | None = None,
) -> Galerkin:
    """Galerkin reduction of SPD systems onto the basis ``W``.

    Args:
        W: ``(n, m)`` orthonormal basis, e.g. from :func:`pod`.
        enrich: Residual-enrichment steps per call; 0 is the plain POD answer.
        gate: Enrich only when the indicator ``eta`` of the plain answer exceeds it; ``None`` always enriches.
            Inside ``vmap`` both branches run, since ``jax.lax.cond`` becomes a select there.
        float32: Build the enrichment directions in single precision. The reduced system stays in the working
            precision, so the answer is still the exact Galerkin answer on the (slightly different) span.
        free: ``(n,)`` bool mask of the unconstrained DOFs; rows of essential conditions are ignored.

    Returns:
        A :class:`Galerkin`, called as ``rom(A, B)``.
    """
    return Galerkin(W, enrich=enrich, gate=gate, float32=float32, free=free)
