"""The state-space layer operator ``M_L``, its eig-free transfer, and the
analytic ambient layer pieces.

Physics (Liu 2012 eqs. 15--28, the same assembly fmmax uses): with the
tangential state ``psi = (u, v)`` where ``u = (-E_y, E_x)`` and
``v = (H_x, H_y)``, a source-free, z-invariant layer satisfies

    d/dz psi = i M_L psi ,          psi(z) = expm(i M_L z) psi(0) ,

with the block-anti-diagonal form (validated three ways in the stage-0
prototype: the Maxwell curl equations with ``div E = div B = 0`` closure, the
modal picture, and the exact uniform-isotropic spectrum ``eig(M_L) = +-q``,
``q^2 = omega^2 eps - |k+G|^2``):

    M_L = [[0, B], [C, 0]] ,
    B = (omega^2 M_t - script_K) / omega ,     (Omega_K / omega)
    C = (omega^2 E_t - K) / omega ,            (K = K_E^T Z_mu^{-1} K_E)

``script_K`` uses the inverse-rule (Toeplitz-inverse) Fourier factorization, as
in fmmax. For nonmagnetic media ``M_t = I`` and ``K = K_E^T K_E``.

The transfer ``expm(i M_L h)`` is never formed as a ``4n x 4n`` exponential.
Since ``M_L^2 = diag(B C, C B)``, it is a pair of matrix functions of the
``2n x 2n`` product ``P = B C`` alone (see :func:`slice_transfer`)::

    expm(i M_L h) = [[ I - P s2,   i s B          ],
                     [ i C s,      I - C s2 B     ]],
    s  = sin(sqrt(P) h) / sqrt(P),   s2 = (1 - cos(sqrt(P) h)) / P,

both entire power series in ``P h^2`` (no square root is ever taken). They are
summed on a thin slice ``h / 2^k`` and brought back to ``h`` by ``k``
double-angle steps, ``s(2h) = 2 s (I - P s2)`` and ``s2(2h) = 2 s^2``: the
matrix-cosine analogue of scaling and squaring (Higham & Smith, Numer.
Algorithms 34, 13 (2003)), carried in the ``1 - cos`` form so that a thin layer
keeps full relative accuracy. Every product is ``2n x 2n``, 8x cheaper than a
``4n`` exponential. ``k`` is static, fixed at build from the concrete norm of
``P``, while ``h`` may be traced, so the compiled program does not depend on the
thickness and ``jax.grad`` in the thickness is native.

Uniform media never need the expm: their per-order ``M_L`` blocks are
simultaneously diagonalizable and ``expm(i M_L d)`` is the closed form
``[[c I, i s Z], [i s Y, c I]]`` with ``c = cos(q d)``, ``s = sin(q d)``,
``Z = B/q``, ``Y = C/q`` on every order. That is the *uniform-isotropic
analytic shortcut*: decoupled per-order, exact, and free of any eig.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from . import matrices

#: Bound on ``||P h0^2||_inf`` for the thin-slice series, and the series degree per real precision. The
#: truncation error is about ``SERIES_THETA^(m+1) / (2m+2)!``: 3e-17 at degree 11 (float64) and 2e-7 at
#: degree 6 (float32).
SERIES_THETA = 4.0
_SERIES_DEGREE = {8: 11, 4: 6}
#: The static doubling count is chosen for this multiple of the build thickness, so that a traced
#: thickness may grow by up to this factor without leaving the accuracy bound (one doubling, three
#: ``2n`` matmuls, per factor of two).
THICKNESS_MARGIN = 2.0


def series_degree(dtype) -> int:
    """The thin-slice series degree for a complex ``dtype`` (float32 or float64 parts)."""
    return _SERIES_DEGREE[jnp.finfo(jnp.dtype(dtype)).bits // 8]


def series_doublings(p_norm: float, thickness: float, theta: float = SERIES_THETA) -> int:
    """Smallest static ``k`` with ``p_norm * (THICKNESS_MARGIN * thickness / 2^k)^2 <= theta`` (host-side)."""
    x = float(p_norm) * (THICKNESS_MARGIN * float(thickness)) ** 2 / theta
    if not math.isfinite(x):
        raise ValueError(f"_rcwa: the layer operator has a non-finite norm ({p_norm!r}); check the permittivity.")
    return max(0, int(math.ceil(0.5 * math.log2(x)))) if x > 1.0 else 0


def _concrete(x) -> bool:
    return not isinstance(x, jax.core.Tracer)


def block2x2(a, b, c, d) -> jnp.ndarray:
    return jnp.block([[a, b], [c, d]])


def pair_matrix(x: jnp.ndarray) -> jnp.ndarray:
    """Repeats the ``n x n`` block ``x`` over the two tangential polarizations."""
    z = jnp.zeros_like(x)
    return block2x2(x, z, z, x)


# -----------------------------------------------------------------------------
# The 4n x 4n state-space operator.
# -----------------------------------------------------------------------------


def layer_bc(
    omega,
    transverse_wavevectors: jnp.ndarray,
    z_permittivity: jnp.ndarray,
    transverse_permittivity: jnp.ndarray,
    transverse_permeability: Optional[jnp.ndarray] = None,
    z_permeability: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The ``2n x 2n`` blocks ``(B, C)`` of ``M_L = [[0, B], [C, 0]]`` for a patterned layer.

    Args:
        omega: Angular frequency ``2 pi / wavelength`` (c == 1).
        transverse_wavevectors: Per-order ``(n, 2)`` wavevectors.
        z_permittivity: Fourier matrix of the zz permittivity (``n x n``).
        transverse_permittivity: The transverse permittivity matrix ``E_t``
            (from ``matrices.transverse_matrices_isotropic`` or
            ``..._anisotropic``).
        transverse_permeability: ``M_t``; ``None`` means nonmagnetic (identity).
        z_permeability: Fourier matrix of the zz permeability (``n x n``);
            ``None`` means nonmagnetic (``K = K_E^T K_E``).

    Returns:
        ``(B, C)``, each ``(2n, 2n)``, in the ``(u | v)`` block layout.
    """
    n = transverse_wavevectors.shape[-2]
    dtype = z_permittivity.dtype
    omega = jnp.asarray(omega).astype(dtype)
    two_n = 2 * n

    script_k = matrices.script_k_matrix_patterned(z_permittivity, transverse_wavevectors)
    k_mat = matrices.k_matrix_patterned(transverse_wavevectors, z_permeability_matrix=z_permeability)
    if transverse_permeability is None:
        m_t = jnp.eye(two_n, dtype=dtype)
    else:
        m_t = transverse_permeability
    return omega * m_t - script_k / omega, omega * transverse_permittivity - k_mat / omega


def assemble_M_L(
    omega,
    transverse_wavevectors,
    z_permittivity,
    transverse_permittivity,
    transverse_permeability=None,
    z_permeability=None,
) -> jnp.ndarray:
    """The full ``4n x 4n`` operator ``M_L`` (tests and diagnostics; the solve uses :func:`layer_bc`)."""
    b_mat, c_mat = layer_bc(
        omega, transverse_wavevectors, z_permittivity, transverse_permittivity, transverse_permeability, z_permeability
    )
    z = jnp.zeros_like(b_mat)
    return block2x2(z, b_mat, c_mat, z)


# -----------------------------------------------------------------------------
# The thin-slice transfer: two series in P = B C, then double-angle steps.
# -----------------------------------------------------------------------------


def _paterson_stockmeyer_pair(x: jnp.ndarray, a: Tuple[float, ...], b: Tuple[float, ...]):
    """``(sum_k a_k x^k, sum_k b_k x^k)`` with shared powers (Paterson & Stockmeyer, SIAM J. Comput. 2, 60 (1973)).

    The powers ``x^1 .. x^q`` are formed once and both polynomials are run by Horner's rule in ``x^q``, so two
    degree-``m`` series cost ``(q - 1) + 2 (ceil((m + 1) / q) - 1)`` matmuls; ``q`` minimises that. Works on
    batches ``(..., m, m)``.
    """
    m = len(a) - 1
    q = min(range(1, m + 2), key=lambda q: (q - 1) + 2 * (-(-(m + 1) // q) - 1))
    eye = jnp.broadcast_to(jnp.eye(x.shape[-1], dtype=x.dtype), x.shape)
    pw = [eye, x]
    for _ in range(q - 1):
        pw.append(pw[-1] @ x)
    y = pw[q]

    def horner(c):
        blocks = [sum(pw[i] * c[j + i] for i in range(min(q, m + 1 - j))) for j in range(0, m + 1, q)]
        r = blocks[-1]
        for blk in reversed(blocks[:-1]):
            r = r @ y + blk
        return r

    return horner(a), horner(b)


def slice_transfer(b: jnp.ndarray, c: jnp.ndarray, p: jnp.ndarray, h, doublings: int, degree: int):
    """The four ``2n x 2n`` blocks ``(A, B~, C~, D)`` of ``expm(i M_L h)``, ``M_L = [[0, b], [c, 0]]``.

    With ``P = b c`` (passed in, it is shared by every thickness of the layer)::

        A = I - P s2,   B~ = i s b,   C~ = i c s,   D = I - c s2 b,

    where ``s = sin(sqrt(P) h)/sqrt(P)`` and ``s2 = (1 - cos(sqrt(P) h))/P``. On the slice ``h0 = h / 2^k``,
    ``s = h0 sum_j (-P h0^2)^j / (2j+1)!`` and ``s2 = h0^2 sum_j (-P h0^2)^j / (2j+2)!``; then ``k`` double-angle
    steps ``s <- 2 s (I - P s2)``, ``s2 <- 2 s^2`` (both are functions of ``P`` and commute with it). Accurate to
    the series' truncation while ``||P h0^2||_inf <= SERIES_THETA``, which :func:`series_doublings` arranges
    and :func:`series_load` reports.

    Args:
        b, c: The blocks of ``M_L``, ``(..., m, m)`` (``m = 2n``, or 2 for the per-order uniform path).
        p: ``b @ c``.
        h: Thickness; may be traced.
        doublings: Static ``k``.
        degree: Static series degree (:func:`series_degree`).
    """
    h0 = h / (2.0**doublings)
    x = -p * (h0 * h0)
    fac = math.factorial
    sig, tau = _paterson_stockmeyer_pair(
        x,
        tuple(1.0 / fac(2 * j + 1) for j in range(degree + 1)),
        tuple(1.0 / fac(2 * j + 2) for j in range(degree + 1)),
    )
    s, s2 = h0 * sig, (h0 * h0) * tau
    for _ in range(doublings):
        ps2 = p @ s2
        s2 = 2.0 * (s @ s)
        s = 2.0 * (s - s @ ps2)
    eye = jnp.broadcast_to(jnp.eye(p.shape[-1], dtype=p.dtype), p.shape)
    return eye - p @ s2, 1j * (s @ b), 1j * (c @ s), eye - c @ (s2 @ b)


def series_load(p: jnp.ndarray, h, doublings: int) -> jnp.ndarray:
    """``||P h0^2||_inf / SERIES_THETA`` for the slice of :func:`slice_transfer` (traceable; <= 1 is accurate)."""
    h0 = jnp.abs(h) / (2.0**doublings)
    return jnp.max(jnp.sum(jnp.abs(p), axis=-1)) * h0 * h0 / SERIES_THETA


def matrix_inf_norm(x) -> float:
    """Host-side Inf-norm (max absolute row sum) over the last axis, maximised over any batch axes."""
    import numpy as onp

    return float(onp.max(onp.sum(onp.abs(onp.asarray(x)), axis=-1)))


def uniform_aniso_bc(omega, transverse_wavevectors, eps_comps, mu_comps, dtype=jnp.complex64):
    """Per-order ``(n, 2, 2)`` blocks ``(B_i, C_i)`` of a uniform anisotropic layer's block-diagonal ``M_L``.

    The zz Toeplitz matrices of a uniform layer are ``eps_zz I`` and ``mu_zz I``, so ``script_K`` and ``K``
    decouple per order and every order evolves by its own ``[[0, B_i], [C_i, 0]]``, with
    ``B_i = omega M_t - script_K_i / omega`` and ``C_i = omega E_t - K_i / omega``.
    """
    omega = jnp.asarray(omega, dtype=dtype)
    eps_xx, eps_xy, eps_yx, eps_yy, eps_zz = (jnp.asarray(c, dtype=dtype) for c in eps_comps)
    if mu_comps is None:
        one, zero = jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)
        mu_xx, mu_xy, mu_yx, mu_yy, mu_zz = one, zero, zero, one, one
    else:
        mu_xx, mu_xy, mu_yx, mu_yy, mu_zz = (jnp.asarray(c, dtype=dtype) for c in mu_comps)
    n = transverse_wavevectors.shape[0]
    kx = transverse_wavevectors[:, 0]
    ky = transverse_wavevectors[:, 1]

    def mat2(a, b_, c_, d):
        return jnp.stack([jnp.stack([a, b_], -1), jnp.stack([c_, d], -1)], -2)

    full = lambda v: jnp.full(n, v)  # noqa: E731
    e_t = mat2(full(eps_yy), full(-eps_yx), full(-eps_xy), full(eps_xx))
    m_t = mat2(full(mu_xx), full(mu_xy), full(mu_yx), full(mu_yy))
    kx2, ky2, kxky = kx * kx, ky * ky, kx * ky
    script_k = mat2(ky2, -kxky, -kxky, kx2) / eps_zz
    k_mat = mat2(kx2, kxky, kxky, ky2) / mu_zz
    return omega * m_t - script_k / omega, omega * e_t - k_mat / omega


# -----------------------------------------------------------------------------
# Never-silent anisotropic tensor guard (eager, concrete values only).
# -----------------------------------------------------------------------------


def validate_anisotropic_tensor(eps_comps, mu_comps, layer_index=0, uniform=False):
    """Loud guard for a physically-ill-posed anisotropic layer tensor.

    Checks -- on CONCRETE values only (callers gate on ``_concrete``) -- the
    failure mode that makes the state-space assembly singular:

    * a zero zz component anywhere (``eps_zz == 0`` or ``mu_zz == 0`` at any
      pixel / scalar): the inverse-rule z-matrix ``1/eps_zz`` (``script_K``)
      and ``1/mu_zz`` (``K``) divide per-pixel, so a single zero pixel or a
      zero uniform value blows the assembly to inf/NaN (never silent).

    For a UNIFORM (constant-tensor) layer it additionally rejects a *singular
    transverse* 2x2 tensor (``eps_xx*eps_yy - eps_xy*eps_yx == 0``): the
    per-order B/C blocks then lose the polarization structure (evanescent
    collapse, negated determinant = unphysical gain medium).

    Raises ``ValueError`` naming the layer and the offending component -- the
    house never-silent style (the seam wraps it as ``RcwaError``). Traced
    values cannot be checked (the eager pre-run validates the concrete forward
    pass).
    """
    import numpy as onp

    def guard(comps, name):
        if comps is None:
            return
        xx, xy, yx, yy, zz = [onp.asarray(c) for c in comps]
        zz_flat = zz.reshape(-1)
        zz_min = float(onp.min(onp.abs(zz_flat)))
        if not zz_min > 0.0:
            raise ValueError(
                f"_rcwa: layer {layer_index} has a zero {name}_zz component "
                f"(|{name}_zz|_min = {zz_min:.3g}) -- the inverse-rule z-matrix "
                "(script_K / K) divides by it per pixel. Fix the tensor; the "
                "assembly would silently produce inf/NaN otherwise."
            )
        if uniform:
            xx0, xy0, yx0, yy0, zz0 = (
                complex(xx.reshape(-1)[0]),
                complex(xy.reshape(-1)[0]),
                complex(yx.reshape(-1)[0]),
                complex(yy.reshape(-1)[0]),
                complex(zz.reshape(-1)[0]),
            )
            det = xx0 * yy0 - xy0 * yx0
            if abs(det) == 0.0:
                raise ValueError(
                    f"_rcwa: layer {layer_index} has a singular transverse {name} tensor "
                    f"(det = {name}_xx*{name}_yy - {name}_xy*{name}_yx = 0) -- the per-order "
                    "B/C blocks collapse; a singular tensor is never silently solved. "
                    "Fix the tensor."
                )
            # A REAL (lossless) tensor must be positive-definite: a negative
            # eigenvalue (e.g. a negative-definite diagonal) is an unphysical
            # gain medium, caught loudly instead of solving with a sign-flipped
            # spectrum. Complex (lossy) tensors are left to the energy guard.
            lossless = max(abs(xx0.imag), abs(xy0.imag), abs(yx0.imag), abs(yy0.imag), abs(zz0.imag)) <= 0
            if lossless and not (float(xx0.real) > 0 and float(det.real) > 0 and float(zz0.real) > 0):
                raise ValueError(
                    f"_rcwa: layer {layer_index} has a non-positive-definite {name} tensor "
                    f"(Re xx = {xx0.real:.3g}, det = {det.real:.3g}, zz = {zz0.real:.3g}); a "
                    "negative-definite lossless tensor is an unphysical gain medium. Fix the tensor."
                )

    guard(eps_comps, "ε")
    guard(mu_comps, "μ")


# -----------------------------------------------------------------------------
# Uniform-isotropic analytic shortcut (no expm, no eig).
# -----------------------------------------------------------------------------
def uniform_kz(
    transverse_wavevectors: jnp.ndarray,
    eps,
    omega,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """Per-order longitudinal wavenumber ``q = sqrt(eps w^2 - |k+G|^2)``.

    The branch is fixed to ``Im(q) >= 0`` (forward = +z). The transfer of a
    uniform layer is branch-independent (``cos``/``sin`` are even in `q`); the
    branch only feeds the ambient split, where it picks the decaying direction.
    """
    eps = jnp.asarray(eps, dtype=dtype)
    om = jnp.asarray(omega, dtype=dtype)
    kz2 = eps * om**2 - (transverse_wavevectors[..., 0] ** 2 + transverse_wavevectors[..., 1] ** 2)
    kz = jnp.sqrt(kz2)
    return jnp.where(jnp.imag(kz) < 0, -kz, kz)


def uniform_blocks(
    omega,
    transverse_wavevectors: jnp.ndarray,
    eps,
    thickness,
    dtype: jnp.dtype = jnp.complex64,
):
    """Per-order ``(n, 2, 2)`` blocks ``(A, B~, C~, D)`` of ``expm(i M_L d)`` for a uniform isotropic layer.

    Each order is exactly ``A = D = cos(q d) I`` and ``B~ = i sin(q d)/q B_o``, ``C~ = i sin(q d)/q C_o``, with the
    order's ``2 x 2`` blocks ``B_o``, ``C_o`` of ``M_L``: O(n), no matrix function at all.
    """
    omega = jnp.asarray(omega, dtype=dtype)
    eps = jnp.asarray(eps, dtype=dtype)
    kx = transverse_wavevectors[:, 0]
    ky = transverse_wavevectors[:, 1]
    q = uniform_kz(transverse_wavevectors, eps, omega, dtype=dtype)
    qd = q * thickness
    cos_qd = jnp.cos(qd)
    sin_q = jnp.where(q == 0, thickness, jnp.sin(qd) / jnp.where(q == 0, 1.0, q))

    def mat2(a, b_, c_, d):
        return jnp.stack([jnp.stack([a, b_], -1), jnp.stack([c_, d], -1)], -2)

    b12 = kx * ky / (eps * omega)
    b = mat2(omega - ky**2 / (eps * omega), b12, b12, omega - kx**2 / (eps * omega))
    c12 = -kx * ky / omega
    c = mat2(omega * eps - kx**2 / omega, c12, c12, omega * eps - ky**2 / omega)
    a = mat2(cos_qd, jnp.zeros_like(cos_qd), jnp.zeros_like(cos_qd), cos_qd)
    return a, 1j * sin_q[:, None, None] * b, 1j * sin_q[:, None, None] * c, a


def per_order_dense(t: jnp.ndarray) -> jnp.ndarray:
    """The ``2n x 2n`` component-major matrix of per-order ``(n, 2, 2)`` blocks."""
    return block2x2(jnp.diag(t[:, 0, 0]), jnp.diag(t[:, 0, 1]), jnp.diag(t[:, 1, 0]), jnp.diag(t[:, 1, 1]))


def uniform_layer_transfer(
    omega,
    transverse_wavevectors: jnp.ndarray,
    eps,
    thickness: float,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """The ``4n x 4n`` transfer ``expm(i M_L d)`` of a uniform isotropic layer.

    Closed form on each order (see module docstring)::

        T = [[cos(q d) I,  i sin(q d) B / q],
             [i sin(q d) C / q,  cos(q d) I]],

    where the ``B``/``C`` blocks are the per-order analogue of ``M_L``'s blocks.
    This is exact, decoupled per order, and needs no eig and no Pade.

    Args:
        omega: Angular frequency.
        transverse_wavevectors: Per-order ``(n, 2)`` wavevectors.
        eps: The uniform permittivity (scalar or a ``(1, 1)`` grid).
        thickness: The (sub-slab) chunk thickness ``d``.
        dtype: Output complex dtype.

    Returns:
        The ``4n x 4n`` transfer matrix.
    """
    a, b, c, d = uniform_blocks(omega, transverse_wavevectors, eps, thickness, dtype=dtype)
    return block2x2(per_order_dense(a), per_order_dense(b), per_order_dense(c), per_order_dense(d))


# -----------------------------------------------------------------------------
# Uniform-anisotropic analytic* fast path (no expm of the 4n block, no eig).
# -----------------------------------------------------------------------------


def uniform_anisotropic_layer_transfer(
    omega,
    transverse_wavevectors: jnp.ndarray,
    eps_comps,
    mu_comps,
    thickness: float,
    doublings: int,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """The ``4n x 4n`` transfer ``expm(i M_L d)`` of a *uniform* anisotropic layer.

    ``M_L`` is block-diagonal over the orders (:func:`uniform_aniso_bc`), so the transfer is n independent
    ``4 x 4`` transfers, each from :func:`slice_transfer` on its ``2 x 2`` blocks, assembled into the
    component-major state order. Differentiable in the tensor and the thickness, no eig.

    Args:
        omega: Angular frequency.
        transverse_wavevectors: Per-order ``(n, 2)`` wavevectors.
        eps_comps: ``(eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)`` scalars.
        mu_comps: ``(mu_xx, mu_xy, mu_yx, mu_yy, mu_zz)`` scalars, or ``None``
            for nonmagnetic (identity).
        thickness: The (sub-slab) chunk thickness; may be traced.
        doublings: Static double-angle count (from :func:`series_doublings` on the per-order blocks).
        dtype: Output complex dtype.

    Returns:
        The ``4n x 4n`` transfer matrix.
    """
    ta, tb, tc, td = uniform_aniso_blocks(omega, transverse_wavevectors, eps_comps, mu_comps, thickness, doublings, dtype)
    # (u1 over orders | u2 over orders | v1 | v2), matching the isotropic uniform transfer's block layout.
    return block2x2(per_order_dense(ta), per_order_dense(tb), per_order_dense(tc), per_order_dense(td))


def uniform_aniso_blocks(
    omega, transverse_wavevectors, eps_comps, mu_comps, thickness, doublings: int, dtype=jnp.complex64
):
    """Per-order ``(n, 2, 2)`` blocks ``(A, B~, C~, D)`` of a uniform anisotropic layer's transfer: the thin-slice
    series on each order's ``2 x 2`` blocks, O(n)."""
    b, c = uniform_aniso_bc(omega, transverse_wavevectors, eps_comps, mu_comps, dtype=dtype)
    return slice_transfer(b, c, b @ c, thickness, doublings, series_degree(dtype))


# -----------------------------------------------------------------------------
# Analytic uniform-ambient split and the plane-wave incident state.
# -----------------------------------------------------------------------------


def ambient_kz(
    transverse_wavevectors: jnp.ndarray,
    eps,
    omega,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """Per-order ``kz`` in a uniform ambient; ``Im(kz) >= 0``, forward = +z."""
    return uniform_kz(transverse_wavevectors, eps, omega, dtype=dtype)


def ambient_admittance(
    transverse_wavevectors: jnp.ndarray,
    kz: jnp.ndarray,
    omega,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """Per-order surface admittance ``Y`` (``v = Y u``), shape ``(n, 2, 2)``.

    From ``Y = C / q`` on each order: ``Y_11 = kz/w + ky^2/(w kz)`` etc. -- the
    exact uniform-medium admittance (identical to fmmax's uniform path).
    """
    kx = transverse_wavevectors[:, 0]
    ky = transverse_wavevectors[:, 1]
    w = jnp.asarray(omega, dtype=dtype)
    y11 = ky**2 / (w * kz) + kz / w
    y12 = -kx * ky / (w * kz)
    y22 = kz / w + kx**2 / (w * kz)
    return jnp.stack([jnp.stack([y11, y12], -1), jnp.stack([y12, y22], -1)], -2)


def ambient_admittance_matrix(
    transverse_wavevectors: jnp.ndarray,
    kz: jnp.ndarray,
    omega,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """The ``2n x 2n`` block-diagonal admittance matrix (component-major order)."""
    y = ambient_admittance(transverse_wavevectors, kz, omega, dtype=dtype)
    y11 = jnp.diag(y[:, 0, 0])
    y12 = jnp.diag(y[:, 0, 1])
    y22 = jnp.diag(y[:, 1, 1])
    return jnp.block([[y11, y12], [y12, y22]])


def plane_wave_incident_state(
    polarization,
    transverse_wavevectors: jnp.ndarray,
    eps_amb,
    omega,
    dtype: jnp.dtype = jnp.complex64,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The tangential state ``(u, v)`` of the incident Jones plane wave.

    Only the zeroth-order (``k_in``) order is excited: ``u = (-E_y, E_x)`` from
    the Jones vector and ``v = Y_sup u`` for the *forward* (+z) wave. This is
    exactly the state ``fmmax.sources.amplitudes_for_fields`` inverts when fed
    the same real-space field.

    Args:
        polarization: ``(px, py)`` Jones amplitudes (complex-scalar coercible).
        transverse_wavevectors: Per-order wavevectors.
        eps_amb: Superstrate permittivity.
        omega: Angular frequency.
        dtype: Output complex dtype.

    Returns:
        The ``(u, v)`` state vectors, each ``(2n,)``.
    """
    px, py = complex(polarization[0]), complex(polarization[1])
    n = transverse_wavevectors.shape[0]
    u = jnp.zeros(2 * n, dtype=dtype)
    u = u.at[0].set(-py)
    u = u.at[n].set(px)
    kz = ambient_kz(transverse_wavevectors, eps_amb, omega, dtype=dtype)
    y = ambient_admittance_matrix(transverse_wavevectors, kz, omega, dtype=dtype)
    return u, y @ u


# -----------------------------------------------------------------------------
# Readouts: signed per-order Poynting flux.
# -----------------------------------------------------------------------------


def sz_flux(u: jnp.ndarray, v: jnp.ndarray, n: int) -> jnp.ndarray:
    """Per-order signed z-Poynting ``0.5 Re(u1 conj(v1) + u2 conj(v2))``, ``(n,)``.

    jNO's exact convention: the (0,0) order of the incident state, summed over
    orders, is the incident power; a transmitted state contributes positive
    flux, a reflected state negative (hence the sign flip in ``R`` readouts).
    """
    return 0.5 * jnp.real(u[:n] * jnp.conj(v[:n]) + u[n:] * jnp.conj(v[n:]))
