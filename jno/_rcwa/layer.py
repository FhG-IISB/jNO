"""The state-space layer operator ``M_L``, its eig-free exponential, and the
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

``expm`` is a fixed-degree [13/13] Pade with a *static* scaling-squaring count
chosen eagerly at build from the concrete spectral bound (Higham's
``Theta_13 ~ 5.3719``); under trace the guard steps aside and the build-time
count is reused, so the compiled program has no data-dependent control flow and
the whole chain is plain matmul/solve -- native JVP, no eig and no custom VJP.

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

# Fixed Pade degree [m/m] and Higham's accuracy threshold for ||X||.
PADE_DEGREE = 13
THETA_13 = 5.371919
# Covers ||M_L d||_infty up to ~5.37 * 2**16 ~ 3.5e5 before the eager chunking
# guard refuses (thick evanescent layers must be chunked thinner instead).
MAX_SQUARINGS = 16


def _pade_coefficients(m: int) -> Tuple[float, ...]:
    """``b_k = (2m-k)! m! / ((2m)! k! (m-k)!)`` for k = 0..m (Pade [m/m])."""
    fac = math.factorial
    return tuple(
        float(fac(2 * m - k)) * float(fac(m)) / (float(fac(2 * m)) * float(fac(k)) * float(fac(m - k)))
        for k in range(m + 1)
    )


_PADE_B = _pade_coefficients(PADE_DEGREE)


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


def assemble_M_L(
    omega,
    transverse_wavevectors: jnp.ndarray,
    z_permittivity: jnp.ndarray,
    transverse_permittivity: jnp.ndarray,
    transverse_permeability: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """The ``4n x 4n`` operator ``M_L`` for a patterned layer.

    Args:
        omega: Angular frequency ``2 pi / wavelength`` (c == 1).
        transverse_wavevectors: Per-order ``(n, 2)`` wavevectors.
        z_permittivity: Fourier matrix of the zz permittivity (``n x n``).
        transverse_permittivity: The transverse permittivity matrix ``E_t``
            (from ``matrices.transverse_matrices_isotropic``).
        transverse_permeability: ``M_t``; ``None`` means nonmagnetic (identity).

    Returns:
        ``M_L`` of shape ``(4n, 4n)`` in the ``(u | v)`` block layout.
    """
    n = transverse_wavevectors.shape[-2]
    dtype = z_permittivity.dtype
    omega = jnp.asarray(omega).astype(dtype)
    two_n = 2 * n

    script_k = matrices.script_k_matrix_patterned(z_permittivity, transverse_wavevectors)
    k_mat = matrices.k_matrix_patterned(transverse_wavevectors)
    if transverse_permeability is None:
        m_t = jnp.eye(two_n, dtype=dtype)
    else:
        m_t = transverse_permeability
    b_mat = omega * m_t - script_k / omega
    c_mat = omega * transverse_permittivity - k_mat / omega
    z = jnp.zeros((two_n, two_n), dtype=dtype)
    return block2x2(z, b_mat, c_mat, z)


# -----------------------------------------------------------------------------
# expm: fixed-degree Pade + static scaling-squaring.
# -----------------------------------------------------------------------------


def expm_pade_scaling_squaring(x: jnp.ndarray, squarings: int) -> jnp.ndarray:
    """``exp(x)`` with Pade [13/13] on ``x / 2**squarings`` then `squarings` squarings.

    `squarings` is a Python int (static), so jit sees a static program; the
    norm of the scaled argument must be <= ``THETA_13``, guaranteed by the
    eager spectral-bound guard (``squarings_for_spectral_norm``) at build.
    """
    x_scaled = x / (2**squarings)
    n = x_scaled.shape[-1]
    b = _PADE_B
    # Numerator/denominator of the [13/13] approximant via Horner matrix powers.
    eye = jnp.eye(n, dtype=x_scaled.dtype)
    num = b[0] * eye
    den = b[0] * eye
    x_pow = x_scaled
    sign = -1.0
    for k in range(1, PADE_DEGREE + 1):
        num = num + b[k] * x_pow
        den = den + sign * b[k] * x_pow
        sign = -sign
        x_pow = x_pow @ x_scaled
    r = jnp.linalg.solve(den, num)
    for _ in range(squarings):
        r = r @ r
    return r


def squarings_for_spectral_norm(norm_x: float, max_squarings: int = MAX_SQUARINGS) -> int:
    """Smallest static squaring count with ``norm_x / 2**s <= THETA_13``."""
    if norm_x <= THETA_13:
        return 0
    s = int(math.ceil(math.log2(norm_x / THETA_13)))
    if s > max_squarings:
        raise ValueError(
            f"_rcwa: expm spectral norm {norm_x:.3g} requires {s} squarings "
            f"(cap {max_squarings}) -- chunk the layer thinner."
        )
    return s


def layer_transfer_exp(
    m_l: jnp.ndarray,
    thickness: float,
    squarings: Optional[int] = None,
    guard_norm: Optional[float] = None,
) -> jnp.ndarray:
    """The chunk transfer ``expm(i * M_L * d)`` with the static squaring guard.

    ``guard_norm`` is an eager estimate of ``||M_L * d||_infty`` used to pick
    the static squaring count when `squarings` is not given; under trace the
    count must be provided or the guard fails loudly. When an explicit
    `squarings` is given, the eager bound check still runs on concrete values
    and raises if the Pade accuracy budget would be exceeded.
    """
    x = 1j * m_l * thickness
    if squarings is None:
        if guard_norm is not None:
            squarings = squarings_for_spectral_norm(float(guard_norm))
        elif _concrete(x):
            # Same norm family as the build-time static count (inf-norm): the
            # max-abs-entry norm is weaker and can under-budget a row-concentrated
            # matrix past the Pade accuracy threshold.
            squarings = squarings_for_spectral_norm(matrix_inf_norm(x))
        else:
            raise ValueError("layer_transfer_exp needs `guard_norm` or `squarings` under trace")
    if _concrete(x):
        actual = matrix_inf_norm(x)
        if actual > THETA_13 * 2**squarings * (1 + 1e-4):
            raise ValueError(
                f"_rcwa: expm argument bound exceeded (||X||_inf={actual:.3g} > "
                f"{THETA_13 * 2**squarings:.3g}) -- re-chunk or shrink the layer."
            )
    return expm_pade_scaling_squaring(x, squarings)


def matrix_inf_norm(x) -> float:
    """Host-side Inf-norm (max absolute row sum), the static build norm."""
    import numpy as onp

    return float(onp.max(onp.sum(onp.abs(onp.asarray(x)), axis=-1)))


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
    omega = jnp.asarray(omega, dtype=dtype)
    eps = jnp.asarray(eps, dtype=dtype)
    n = transverse_wavevectors.shape[0]
    kx = transverse_wavevectors[:, 0]
    ky = transverse_wavevectors[:, 1]
    q = uniform_kz(transverse_wavevectors, eps, omega, dtype=dtype)

    # Per-order 2x2 blocks of B (u coupling) and C (v coupling).
    b11 = omega - ky**2 / (eps * omega)
    b12 = kx * ky / (eps * omega)
    b22 = omega - kx**2 / (eps * omega)
    c11 = omega * eps - kx**2 / omega
    c12 = -kx * ky / omega
    c22 = omega * eps - ky**2 / omega

    qd = q * thickness
    cos_qd = jnp.cos(qd)
    sin_qd = jnp.where(q == 0, thickness, jnp.sin(qd) / q)

    def diag(x):
        return jnp.diag(x)

    a11 = diag(cos_qd)
    a22 = diag(cos_qd)
    btilde_11 = 1j * diag(sin_qd * b11)
    btilde_12 = 1j * diag(sin_qd * b12)
    btilde_22 = 1j * diag(sin_qd * b22)
    ctilde_11 = 1j * diag(sin_qd * c11)
    ctilde_12 = 1j * diag(sin_qd * c12)
    ctilde_22 = 1j * diag(sin_qd * c22)

    zero = jnp.zeros((n, n), dtype=dtype)
    return block2x2(
        block2x2(a11, zero, zero, a22),
        block2x2(btilde_11, btilde_12, btilde_12, btilde_22),
        block2x2(ctilde_11, ctilde_12, ctilde_12, ctilde_22),
        block2x2(a11, zero, zero, a22),
    )


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
