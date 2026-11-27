"""Scattering matrices of the layer stack in one fixed reference medium.

Every layer state ``psi = (u, v)`` is written in the plane-wave modes of a single uniform *reference medium*,
``psi = W (a+, a-)`` with ``W = [[I, I], [Y_r, -Y_r]]`` and ``Y_r`` that medium's per-order admittance. Since the
reference is the same above and below every layer, the interfaces are implicit and layers combine directly by
the Redheffer star product (Redheffer, J. Math. Phys. 41, 1 (1962); the convention of Rumpf, PIER B 35, 241
(2011), whose free-space "gap medium" plays the same role).

* The reference permittivity is complex (``1 + 1j``), so that ``kz != 0`` for every order at every angle: no
  order of the reference is grazing and ``Y_r`` is always invertible. ``Y_r`` is block-diagonal over the orders
  (2 x 2 per order), so the change of basis costs O(n^2), never a dense product.
* A z-invariant layer is symmetric under the mirror ``z -> -z`` (``(u, v) -> (u, -v)`` maps solutions to
  solutions because ``M_L`` is block off-diagonal, and it swaps ``a+`` and ``a-``). Its S-matrix therefore has
  ``S11 = S22 = r`` and ``S12 = S21 = t``, and is stored as the pair ``(r, t)``.
* A layer of ``2^m`` identical chunks is ``m`` symmetric doublings of the chunk's ``(r, t)``: each costs four
  ``2n`` matmuls and one solve, and stays bounded however evanescent the orders are -- the stable replacement for
  ``2^m`` steps of the admittance recursion.

Block convention: ``S11`` reflects a wave arriving from above back up, ``S21`` transmits it down, ``S12``
transmits a wave arriving from below up, ``S22`` reflects it back down.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from . import layer

#: The reference medium's permittivity: lossy, so that no order of it is ever grazing.
EPS_REFERENCE = 1.0 + 1.0j


def per_order(y_full: jnp.ndarray, n: int) -> jnp.ndarray:
    """The ``(n, 2, 2)`` per-order blocks of a block-diagonal ``2n x 2n`` matrix in component-major order."""
    return jnp.stack(
        [
            jnp.stack([jnp.diagonal(y_full[:n, :n]), jnp.diagonal(y_full[:n, n:])], -1),
            jnp.stack([jnp.diagonal(y_full[n:, :n]), jnp.diagonal(y_full[n:, n:])], -1),
        ],
        -2,
    )


def inverse_2x2(y: jnp.ndarray) -> jnp.ndarray:
    """Per-order inverses of ``(n, 2, 2)`` blocks (closed form)."""
    a, b, c, d = y[:, 0, 0], y[:, 0, 1], y[:, 1, 0], y[:, 1, 1]
    det = a * d - b * c
    return jnp.stack([jnp.stack([d, -b], -1), jnp.stack([-c, a], -1)], -2) / det[:, None, None]


def right(m: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """``m @ Y`` for a block-diagonal ``Y`` given as ``(n, 2, 2)``: O(n^2)."""
    n = y.shape[0]
    mr = m.reshape(m.shape[0], 2, n)  # columns (component, order)
    return jnp.einsum("rco,ocd->rdo", mr, y).reshape(m.shape)


def left(y: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    """``Y @ m`` for a block-diagonal ``Y`` given as ``(n, 2, 2)``: O(n^2)."""
    n = y.shape[0]
    ml = m.reshape(2, n, *m.shape[1:])  # rows (component, order)
    return jnp.einsum("ocd,do...->co...", y, ml).reshape(m.shape)


def reference(transverse_wavevectors, omega, dtype) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The reference medium's per-order admittance ``Y_r`` and its inverse ``Z_r``, each ``(n, 2, 2)``."""
    kz = layer.ambient_kz(transverse_wavevectors, EPS_REFERENCE, omega, dtype=dtype)
    y = layer.ambient_admittance(transverse_wavevectors, kz, omega, dtype=dtype)
    return y, inverse_2x2(y)


def chunk_smatrix(blocks, y_r, z_r) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The symmetric S-matrix ``(r, t)`` of one chunk from its transfer blocks ``(A, B, C, D)``.

    In the reference modes the transfer is ``T' = W^-1 T W``; with ``a-`` given at the bottom and ``a+`` at the
    top, ``t = T'22^-1`` and ``r = -T'22^-1 T'21``. Only the lower block row of ``T'`` is needed:
    ``T'21 = (A + B Y - Z (C + D Y)) / 2`` and ``T'22 = (A - B Y - Z (C - D Y)) / 2``, all block-diagonal
    products (O(n^2)). A chunk's growth is bounded at build (``|q| d_c <= 1``), so ``T'22`` is well conditioned.
    """
    a, b, c, d = blocks
    if a.ndim == 3:  # per-order (n, 2, 2) blocks of a uniform layer: the same algebra, O(n)
        by, dy = b @ y_r, d @ y_r
        t21 = 0.5 * (a + by - z_r @ (c + dy))
        t22 = 0.5 * (a - by - z_r @ (c - dy))
        t = inverse_2x2(t22)
        return -t @ t21, t
    by, dy = right(b, y_r), right(d, y_r)
    t21 = 0.5 * (a + by - left(z_r, c + dy))
    t22 = 0.5 * (a - by - left(z_r, c - dy))
    two_n = a.shape[-1]
    sol = jnp.linalg.solve(t22, jnp.concatenate([t21, jnp.eye(two_n, dtype=a.dtype)], axis=-1))
    return -sol[..., :two_n], sol[..., two_n:]


def double(r: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """``S * S`` for a symmetric ``S = (r, t)``: ``t' = t (I - r^2)^-1 t``, ``r' = r + t r (I - r^2)^-1 t``.

    (``r`` commutes with ``(I - r^2)^-1``.) Four matmuls and one solve with ``2n`` right-hand sides; on
    per-order ``(n, 2, 2)`` blocks the same in O(n).
    """
    eye = jnp.eye(r.shape[-1], dtype=r.dtype)
    x = jnp.linalg.solve(eye - r @ r, t) if r.ndim == 2 else inverse_2x2(eye - r @ r) @ t
    return r + (t @ r) @ x, t @ x


def layer_smatrix(r: jnp.ndarray, t: jnp.ndarray, chunks: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """``S^chunks`` of a chunk's symmetric ``(r, t)`` by repeated doubling (``chunks`` a power of two), or by the
    binary expansion of ``chunks`` otherwise (the static count is a Python int)."""
    if chunks & (chunks - 1) == 0:
        for _ in range(chunks.bit_length() - 1):
            r, t = double(r, t)
        return r, t
    acc = None
    while chunks:
        if chunks & 1:
            acc = (r, t, t, r) if acc is None else star(acc, (r, t, t, r))
        chunks >>= 1
        if chunks:
            r, t = double(r, t)
    return acc[0], acc[1]


def star(sa, sb):
    """The Redheffer star product ``S_a * S_b`` of ``a`` above ``b``, each ``(S11, S12, S21, S22)``."""
    a11, a12, a21, a22 = sa
    b11, b12, b21, b22 = sb
    eye = jnp.eye(a11.shape[-1], dtype=a11.dtype)
    # (I - B11 A22)^-1 and (I - A22 B11)^-1 against the right-hand sides that need them
    lhs1 = jnp.linalg.solve(eye - b11 @ a22, jnp.concatenate([b11 @ a21, b12], axis=-1))
    lhs2 = jnp.linalg.solve(eye - a22 @ b11, jnp.concatenate([a21, a22 @ b12], axis=-1))
    two_n = a11.shape[-1]
    return (
        a11 + a12 @ lhs1[..., :two_n],
        a12 @ lhs1[..., two_n:],
        b21 @ lhs2[..., :two_n],
        b22 + b21 @ lhs2[..., two_n:],
    )


def ambient_reflection(y_r: jnp.ndarray, y_amb: jnp.ndarray) -> jnp.ndarray:
    """``(Y_r + Y)^-1 (Y_r - Y)`` per order, ``(n, 2, 2)``: how a semi-infinite ambient with admittance ``Y``
    reflects in the reference modes.

    Below the stack only the transmitted wave exists, ``v = Y_sub u``, so ``a- = rho a+``; above it only the
    outgoing wave, ``v = -Y_sup u``, so ``a+ = rho a-``. Both give the same formula.
    """
    return jnp.einsum("oab,obc->oac", inverse_2x2(y_r + y_amb), y_r - y_amb)


def dense(y: jnp.ndarray) -> jnp.ndarray:
    """The ``2n x 2n`` component-major matrix of per-order ``(n, 2, 2)`` blocks."""
    return jnp.block([[jnp.diag(y[:, 0, 0]), jnp.diag(y[:, 0, 1])], [jnp.diag(y[:, 1, 0]), jnp.diag(y[:, 1, 1])]])


def sweep(slabs, rho: jnp.ndarray):
    """Reflections of a terminated slab sequence, and the factors that carry a wave through it.

    ``slabs`` are symmetric ``(r, t)`` listed from the near end to the far end, where the sequence is closed by
    the reflection ``rho`` (dense ``2n x 2n``). Going from the far end back, the reflection seen in front of slab
    ``j`` is ``R_j = r + t R_{j+1} X_j`` with ``X_j = (I - r R_{j+1})^-1 t`` (Li's R-matrix recursion, here on
    S-matrices in the reference modes). A wave travelling toward the far end with amplitude ``a`` in front of slab
    ``j`` has ``X_j a`` behind it: inside the slab ``a_behind = t a + r R_{j+1} a_behind``. Every quantity stays
    bounded, however evanescent the orders are. One solve with ``2n`` right-hand sides and two matmuls per slab.

    Returns:
        ``(R, X)``: ``R[j]`` for ``j = 0..len(slabs)`` (``R[-1] = rho``) and ``X[j]`` per slab.
    """
    eye = jnp.eye(rho.shape[-1], dtype=rho.dtype)
    big_r = rho
    rs, xs = [rho], []
    for r, t in reversed(list(slabs)):
        x = jnp.linalg.solve(eye - r @ big_r, t)
        big_r = r + t @ (big_r @ x)
        rs.append(big_r)
        xs.append(x)
    return rs[::-1], xs[::-1]


def sweep_scan(r_all: jnp.ndarray, t_all: jnp.ndarray, rho: jnp.ndarray):
    """:func:`sweep` over slabs stacked as ``(L, 2n, 2n)`` arrays, as one ``lax.scan`` (compiled once, however many
    slabs). Returns the reflection in front of the first slab and the stacked ``X``."""
    eye = jnp.eye(rho.shape[-1], dtype=rho.dtype)

    def step(big_r, rt):
        r, t = rt
        x = jnp.linalg.solve(eye - r @ big_r, t)
        return r + t @ (big_r @ x), x

    return jax.lax.scan(step, rho, (r_all, t_all), reverse=True)


def carry_scan(xs: jnp.ndarray, a0: jnp.ndarray) -> jnp.ndarray:
    """The amplitude behind the last of the stacked slabs (``xs`` from :func:`sweep_scan`), as one ``lax.scan``."""
    return jax.lax.scan(lambda a, x: (x @ a, None), a0, xs)[0]


def carry(xs, a0: jnp.ndarray):
    """The amplitude in front of every slab and behind the last: ``[a0, X_0 a0, X_1 X_0 a0, ...]``."""
    out = [a0]
    for x in xs:
        out.append(x @ out[-1])
    return out


def reflect_plane_wave(big_r: jnp.ndarray, y_sup: jnp.ndarray, z_r: jnp.ndarray, u_inc: jnp.ndarray):
    """The superstrate side of a plane-wave solve, given the stack's reflection ``R`` seen from above.

    ``u = u_inc + u_ref`` and ``v = Y_sup (u_inc - u_ref)`` must satisfy ``a- = R a+``, i.e.
    ``(I - R) u = (I + R) Z_r v``: one solve for ``u_ref``, without ever inverting ``I + R``.

    Returns:
        ``(u_ref, a_plus)``: the reflected state and the downward reference amplitude at the top of the stack.
    """
    eye = jnp.eye(big_r.shape[-1], dtype=big_r.dtype)
    i_plus_r_zy = right(eye + big_r, jnp.einsum("oab,obc->oac", z_r, y_sup))
    u_ref = jnp.linalg.solve((eye - big_r) + i_plus_r_zy, (i_plus_r_zy - (eye - big_r)) @ u_inc)
    v = apply(y_sup, u_inc - u_ref)
    return u_ref, 0.5 * (u_inc + u_ref + apply(z_r, v))


def apply(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """``Y @ x`` for a block-diagonal ``Y`` given as ``(n, 2, 2)`` and a state vector ``x`` of length ``2n``."""
    return left(y, x[:, None])[:, 0]


def state(a_plus: jnp.ndarray, a_minus: jnp.ndarray, y_r: jnp.ndarray):
    """The tangential state ``(u, v) = (a+ + a-, Y_r (a+ - a-))`` of reference amplitudes."""
    return a_plus + a_minus, apply(y_r, a_plus - a_minus)
