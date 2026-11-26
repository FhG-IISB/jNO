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
    z_permeability: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """The ``4n x 4n`` operator ``M_L`` for a patterned layer.

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
        ``M_L`` of shape ``(4n, 4n)`` in the ``(u | v)`` block layout.
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


def uniform_aniso_spectral_norm(
    eps_comps,
    mu_comps,
    transverse_wavevectors: jnp.ndarray,
    omega,
    thickness: float,
    dtype: jnp.dtype = jnp.complex64,
) -> float:
    """Host-side bound on ``||expm(i M_L d)``'s argument for a *uniform* aniso layer.

    The uniform-aniso transfer is a per-order problem: ``M_L`` is block-diagonal
    over orders with ``4x4`` blocks ``[[0, B_i], [C_i, 0]]``. The Pade squaring
    count must be static, so at build we bound ``max_i ||i M_L^{(i)} d||_infty``
    from the CONCRETE prototype components -- the same stability-margin
    discipline as ``squarings_for_spectral_norm`` on patterned layers. Returns
    the norm (not the squaring count); the eager caller picks the count and the
    traced core reuses it.
    """
    import numpy as onp

    n = transverse_wavevectors.shape[0]
    kx = onp.asarray(transverse_wavevectors[:, 0])
    ky = onp.asarray(transverse_wavevectors[:, 1])

    def _scalar(c):
        """The tensor component as a complex scalar for the norm bound.

        Uniform layers carry scalar/(1,1) values; a build-PATTERNED tensor layer
        (which may collapse to a constant per-solve) contributes its max-abs
        value -- a conservative upper bound on the per-order block norm, so the
        static squaring count stays valid for any constant collapse within the
        prototype's magnitude range."""
        a = onp.asarray(c)
        if a.ndim == 0:
            return complex(a)
        return complex(a.reshape(-1)[onp.argmax(onp.abs(a.reshape(-1)))])

    ee = [_scalar(c) for c in eps_comps]
    eps_xx, eps_xy, eps_yx, eps_yy, eps_zz = ee
    if mu_comps is None:
        mu_xx = mu_yy = mu_zz = 1.0
        mu_xy = mu_yx = 0.0
    else:
        mu_xx, mu_xy, mu_yx, mu_yy, mu_zz = [_scalar(c) for c in mu_comps]
    om = complex(omega)
    cdt_eps = onp.result_type(*[onp.asarray(c).dtype for c in ee], dtype)
    x = onp.zeros((n, 4, 4), cdt_eps)
    kx2, ky2, kxky = kx * kx, ky * ky, kx * ky
    # Per-order 2x2 blocks, transposed mirror of the JAX path (only used for the norm).
    b = onp.stack(
        [
            onp.stack([om * mu_xx - ky2 / (eps_zz * om), om * mu_xy + kxky / (eps_zz * om)], -1),
            onp.stack([om * mu_yx + kxky / (eps_zz * om), om * mu_yy - kx2 / (eps_zz * om)], -1),
        ],
        -2,
    )
    c = onp.stack(
        [
            onp.stack([om * eps_yy - kx2 / (mu_zz * om), om * (-eps_yx) - kxky / (mu_zz * om)], -1),
            onp.stack([om * (-eps_xy) - kxky / (mu_zz * om), om * eps_xx - ky2 / (mu_zz * om)], -1),
        ],
        -2,
    )
    x[:, 0:2, 2:4] = b
    x[:, 2:4, 0:2] = c
    x = 1j * x * thickness
    return float(onp.max(onp.sum(onp.abs(x), axis=-1)))


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
# Uniform-anisotropic analytic* fast path (no expm of the 4n block, no eig).
# -----------------------------------------------------------------------------


def uniform_anisotropic_layer_transfer(
    omega,
    transverse_wavevectors: jnp.ndarray,
    eps_comps,
    mu_comps,
    thickness: float,
    squarings: int,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """The ``4n x 4n`` transfer ``expm(i M_L d)`` of a *uniform* anisotropic layer.

    A uniform layer (all 5/10 component grids constant) has a **block-diagonal**
    ``M_L`` over diffraction orders: the zz Toeplitz matrices are diagonal
    ``eps_zz I`` / ``mu_zz I``, ``script_K`` and ``K`` decouple per order, and
    each order's state evolves by its own ``4x4`` block

        M_L^{(i)} = [[0, B_i], [C_i, 0]],

    with ``B_i = omega M_t^{(i)} - script_K_i/omega`` and
    ``C_i = omega E_t^{(i)} - K_i/omega``. So the layer transfer is n small
    independent ``4x4`` state transfers assembled block-diagonally -- the
    uniform-anisotropic fast path (fmmax dense-eigs a ``2n`` matrix for the same
    physics). Differective in traced ε̂/μ̂, no dense eig.

    Args:
        omega: Angular frequency.
        transverse_wavevectors: Per-order ``(n, 2)`` wavevectors.
        eps_comps: ``(eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)`` scalars.
        mu_comps: ``(mu_xx, mu_xy, mu_yx, mu_yy, mu_zz)`` scalars, or ``None``
            for nonmagnetic (identity).
        thickness: The (sub-slab) chunk thickness.
        squarings: Static Pade scaling-squaring count (from the build-time
            spectral bound; the caller guards it eagerly).
        dtype: Output complex dtype.

    Returns:
        The ``4n x 4n`` transfer matrix.
    """
    omega = jnp.asarray(omega, dtype=dtype)
    eps_xx, eps_xy, eps_yx, eps_yy, eps_zz = (jnp.asarray(c, dtype=dtype) for c in eps_comps)
    if mu_comps is None:
        mu_xx = jnp.asarray(1.0, dtype=dtype)
        mu_xy = jnp.asarray(0.0, dtype=dtype)
        mu_yx = jnp.asarray(0.0, dtype=dtype)
        mu_yy = jnp.asarray(1.0, dtype=dtype)
        mu_zz = jnp.asarray(1.0, dtype=dtype)
    else:
        mu_xx, mu_xy, mu_yx, mu_yy, mu_zz = (jnp.asarray(c, dtype=dtype) for c in mu_comps)
    n = transverse_wavevectors.shape[0]
    kx = transverse_wavevectors[:, 0]
    ky = transverse_wavevectors[:, 1]

    # Per-order 2x2 blocks: E_t and M_t from the constant tensor, script_K and
    # K from the scalar zz components (the decoupling that makes this cheap).
    e_t = jnp.stack(
        [
            jnp.stack([jnp.full(n, eps_yy), jnp.full(n, -eps_yx)], -1),
            jnp.stack([jnp.full(n, -eps_xy), jnp.full(n, eps_xx)], -1),
        ],
        -2,
    )  # (n, 2, 2)
    m_t = jnp.stack(
        [
            jnp.stack([jnp.full(n, mu_xx), jnp.full(n, mu_xy)], -1),
            jnp.stack([jnp.full(n, mu_yx), jnp.full(n, mu_yy)], -1),
        ],
        -2,
    )
    kx2, ky2, kxky = kx * kx, ky * ky, kx * ky
    script_k = (
        1.0
        / eps_zz
        * jnp.stack(
            [
                jnp.stack([ky2, -kxky], -1),
                jnp.stack([-kxky, kx2], -1),
            ],
            -2,
        )
    )
    k_mat = (
        1.0
        / mu_zz
        * jnp.stack(
            [
                jnp.stack([kx2, kxky], -1),
                jnp.stack([kxky, ky2], -1),
            ],
            -2,
        )
    )

    b = omega * m_t - script_k / omega  # (n, 2, 2)
    c = omega * e_t - k_mat / omega  # (n, 2, 2)
    zeros = jnp.zeros((n, 2, 2), dtype=dtype)
    m_l = jnp.block([[zeros, b], [c, zeros]])  # (n, 4, 4) per-order blocks

    t = expm_pade_scaling_squaring(1j * m_l * thickness, squarings)  # (n, 4, 4)

    def diag_block(a, b_, c_, d):
        return block2x2(
            jnp.diag(a),
            jnp.diag(b_),
            jnp.diag(c_),
            jnp.diag(d),
        )

    # Assemble per-order 4x4 blocks into the component-major 4n state order
    # (u1 over orders | u2 over orders | v1 | v2), matching the isotropic
    # uniform transfer's block layout.
    return block2x2(
        diag_block(t[:, 0, 0], t[:, 0, 1], t[:, 1, 0], t[:, 1, 1]),
        diag_block(t[:, 0, 2], t[:, 0, 3], t[:, 1, 2], t[:, 1, 3]),
        diag_block(t[:, 2, 0], t[:, 2, 1], t[:, 3, 0], t[:, 3, 1]),
        diag_block(t[:, 2, 2], t[:, 2, 3], t[:, 3, 2], t[:, 3, 3]),
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
