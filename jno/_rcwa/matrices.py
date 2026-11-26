"""Fourier-factorization matrices for the state-space RCWA backend.

This module owns the matrix building blocks of the Liu (2012) formulation that
the layer solve consumes: the transverse permittivity operator ``E_t`` and its
inverse rule ``Z_eps^{-1}``, the ``script_K`` and ``K`` wavevector dyadics, and
``Omega_K`` (Liu eq. 26). Conventions and ordering match fmmax 1.7.1 exactly
(eq. 15--28), so the seam swap preserves parity; parity is pinned by
``tests/test_rcwa_backend_matrices.py`` with fmmax as a test-only oracle.

``E_t`` is built either with the plain Laurent/``FFT`` rule (Toeplitz blocks of
the permittivity grid) or with a vector formulation, where the Toeplitz rule is
corrected by a term involving the tangent vector field of the interfaces
(Liu eqs. 48--51). The default formulation is ``JONES_DIRECT_FOURIER`` (the
``jno.rcwa`` default): the tangent field is found in the Fourier domain by one
fixed-coefficient Newton step of an alignment loss (``vector.py`` conventions,
grad-decoupled via ``stop_gradient`` so design gradients flow only through the
permittivity itself). All eight fmmax vector schemes are implemented (``POL``,
``NORMAL``, ``JONES``, ``JONES_DIRECT`` and their ``_FOURIER`` variants), plus
the callable tangent-field hook.

Everything below is jit- and grad-clean: only JAX ops in the traced functions,
and no data-dependent control flow (the Newton trip count is a static integer).
dtype/shape contracts are asserted eagerly where the house guards run.
"""

from __future__ import annotations

import enum
import functools
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from . import basis as basis_mod
from . import fft, utils


#: How to compute the transverse permittivity matrix (fmmax-compatible names).
#: ``FFT`` uses Laurent's rule; every other member is a vector formulation, with
#: weights and post-processing matching ``fmmax.vector.VECTOR_FIELD_SCHEMES``.
class Formulation(enum.Enum):
    """Enumerates the supported Fourier-modal-method formulations.

    Each formulation specifies how the transverse permittivity matrix is
    computed. ``FFT`` uses the simple Laurent (Toeplitz) rule; the vector
    formulations additionally use a tangent vector field of the interfaces.
    """

    FFT = "fft"
    JONES_DIRECT = "jones_direct"
    JONES = "jones"
    NORMAL = "normal"
    POL = "pol"
    JONES_DIRECT_FOURIER = "jones_direct_fourier"
    JONES_FOURIER = "jones_fourier"
    NORMAL_FOURIER = "normal_fourier"
    POL_FOURIER = "pol_fourier"


#: A custom formulation hook: maps ``(permittivity, expansion, lattice_vectors)``
#: to a tangent vector field ``(tx, ty)``, exactly fmmax's ``VectorFn``.
VectorFn = Callable[
    [jnp.ndarray, basis_mod.Expansion, basis_mod.LatticeVectors],
    Tuple[jnp.ndarray, jnp.ndarray],
]

#: Vector-scheme weights, matching ``fmmax.vector`` member for member.
_FOURIER_LOSS_WEIGHT_JONES_DIRECT = 0.05
_SMOOTHNESS_LOSS_WEIGHT = 2.0
_SMOOTHNESS_LOSS_WEIGHT_JONES_DIRECT = 0.5
_FOURIER_LOSS_WEIGHT = 0.2

#: Absolute tolerance below which the tangent field counts as 1-D.
_ATOL_1D_FIELD_ANGLE = 1e-2


# -----------------------------------------------------------------------------
# Formulation dispatch.
# -----------------------------------------------------------------------------


def _scheme(formulation: Formulation) -> Tuple[bool, float, float, Optional[str]]:
    """Vector-scheme parameters for `formulation`.

    Returns ``(use_jones_direct, fourier_loss_weight, smoothness_loss_weight,
    tangent_post)``, where ``tangent_post`` is ``None`` (the tangent field is
    used as computed), ``"jones"`` (converted to the complex Jones elliptical
    field) or ``"normal"`` (element-wise normalized), matching
    ``fmmax.vector.VECTOR_FIELD_SCHEMES`` member for member.
    """
    if formulation is Formulation.JONES_DIRECT:
        return True, 0.0, _SMOOTHNESS_LOSS_WEIGHT_JONES_DIRECT, None
    if formulation is Formulation.JONES:
        return False, 0.0, _SMOOTHNESS_LOSS_WEIGHT, "jones"
    if formulation is Formulation.NORMAL:
        return False, 0.0, _SMOOTHNESS_LOSS_WEIGHT, "normal"
    if formulation is Formulation.POL:
        return False, 0.0, _SMOOTHNESS_LOSS_WEIGHT, None
    if formulation is Formulation.JONES_DIRECT_FOURIER:
        return True, _FOURIER_LOSS_WEIGHT_JONES_DIRECT, 0.0, None
    if formulation is Formulation.JONES_FOURIER:
        return False, _FOURIER_LOSS_WEIGHT, 0.0, "jones"
    if formulation is Formulation.NORMAL_FOURIER:
        return False, _FOURIER_LOSS_WEIGHT, 0.0, "normal"
    if formulation is Formulation.POL_FOURIER:
        return False, _FOURIER_LOSS_WEIGHT, 0.0, None
    raise ValueError(
        f"{formulation!r} is not a vector formulation; `FFT` has no tangent field and is handled by the caller."
    )


# -----------------------------------------------------------------------------
# Transverse permittivity matrices.
# -----------------------------------------------------------------------------


def transverse_permittivity_fft(
    permittivity: jnp.ndarray,
    expansion: basis_mod.Expansion,
) -> jnp.ndarray:
    """The transverse permittivity matrix ``E_t`` of the ``FFT`` formulation.

    Entry blocks are the Fourier-convolution (Toeplitz) matrices of the
    permittivity grid (Liu eq. 15): ``E_t = [[eps_hat, 0], [0, eps_hat]]``.

    Args:
        permittivity: Real-space permittivity, shape ``(..., nx, ny)``.
        expansion: The field expansion.

    Returns:
        ``E_t`` with shape ``(..., 2n, 2n)``.
    """
    eps_hat = fft.fourier_convolution_matrix(permittivity, expansion)
    zeros = jnp.zeros_like(eps_hat)
    return jnp.block([[eps_hat, zeros], [zeros, eps_hat]])


def transverse_permittivity_vector(
    permittivity: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    expansion: basis_mod.Expansion,
) -> jnp.ndarray:
    """``E_t`` for a vector formulation: ``eps_matrix - delta_matrix @ p_matrix``.

    This is E2 from Liu eq. 51 (eqs. 48--51, with the diagonal-block x/y swap
    correction that fmmax documents and motivates): the Toeplitz result plus a
    correction that enforces normal-field continuity through the inverse rule on
    the tangent-projected permittivity.

    Args:
        permittivity: Real-space permittivity, shape ``(..., nx, ny)``.
        tx, ty: Components of the tangent vector field (same spatial shape).
        expansion: The field expansion.

    Returns:
        ``E_t`` with shape ``(..., 2n, 2n)``.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
    eps_hat = _transform(permittivity)
    zeros = jnp.zeros_like(eps_hat)
    eps_matrix = jnp.block([[eps_hat, zeros], [zeros, eps_hat]])

    eta = 1 / permittivity
    eta_hat = _transform(eta)
    delta_hat = eps_hat - jnp.linalg.inv(eta_hat)
    delta_matrix = jnp.block([[delta_hat, zeros], [zeros, delta_hat]])

    pxx, pxy, pyx, pyy = _tangent_terms(tx, ty)
    p_matrix = jnp.block(
        [
            [_transform(pyy), _transform(pyx)],
            [_transform(pxy), _transform(pxx)],
        ]
    )
    return eps_matrix - delta_matrix @ p_matrix


def _tangent_terms(tx: jnp.ndarray, ty: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The ``(Pxx, Pxy, Pyx, Pyy)`` projections of Liu eq. 50.

    The diagonal blocks carry the fmmax x<->y swap (its docstring explains why):
    for ``tx = 0`` (1-D grating periodic in x) the tangent field has a y
    component only, and eq. 43 is recovered exactly only with the swapped order.
    """
    denom = jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
    denom_safe = jnp.where(jnp.isclose(denom, 0), 1.0, denom)
    pyy = jnp.abs(tx) ** 2 / denom_safe
    pyx = jnp.conj(tx) * ty / denom_safe
    pxy = tx * jnp.conj(ty) / denom_safe
    pxx = jnp.abs(ty) ** 2 / denom_safe
    return pxx, pxy, pyx, pyy


def transverse_matrices_isotropic(
    permittivity: jnp.ndarray,
    expansion: basis_mod.Expansion,
    formulation: Formulation | VectorFn,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
    """The Fourier matrices of a patterned isotropic layer.

    Mirrors ``fmmax.fmm._fourier_matrices_patterned_isotropic_media``: returns
    the inverse-rule z-permittivity matrix, the z-permittivity matrix, the
    transverse permittivity matrix (per `formulation`) and the tangent vector
    field (``None`` for ``FFT``).

    Args:
        permittivity: Real-space permittivity grid ``(..., nx, ny)``.
        expansion: The field expansion.
        formulation: ``Formulation.FFT``, ``Formulation.JONES_DIRECT_FOURIER``,
            or a callable tangent-field hook.
        primitive_lattice_vectors: Real-space lattice, needed only by vector
            formulations (gradient geometry).

    Returns:
        ``(inverse_z_permittivity, z_permittivity, E_t, tangent_field)``.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
    inverse_z_permittivity = _transform(1 / permittivity)
    z_permittivity = _transform(permittivity)

    if formulation is Formulation.FFT:
        transverse = transverse_permittivity_fft(permittivity, expansion)
        tangent_field = None
    else:
        tx, ty = tangent_field_for(formulation, permittivity, expansion, primitive_lattice_vectors)
        transverse = transverse_permittivity_vector(permittivity, tx, ty, expansion)
        tangent_field = (tx, ty)
    return inverse_z_permittivity, z_permittivity, transverse, tangent_field


# -----------------------------------------------------------------------------
# Anisotropic media (5-component ε̂ and 10-component ε̂+μ̂), fmmax conventions.
# -----------------------------------------------------------------------------
TensorComponents = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
TensorTuple = Tuple[TensorComponents, TensorComponents]  # (ε, μ) component tuples


def transverse_permittivity_fft_anisotropic(
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    expansion: basis_mod.Expansion,
) -> jnp.ndarray:
    """The ``E_t`` matrix of an anisotropic layer, ``FFT`` (Laurent) rule.

    Mirrors ``fmmax.fmm_matrices.transverse_permittivity_fft_anisotropic`` (Liu
    eq. 15 with the anisotropic x/y block placement)::

        E_t = [[Z(ε_yy), -Z(ε_yx)], [-Z(ε_xy), Z(ε_xx)]],

    where ``Z`` is the Fourier-convolution (Toeplitz) matrix of each component.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
    return jnp.block(
        [
            [_transform(permittivity_yy), _transform(-permittivity_yx)],
            [_transform(-permittivity_xy), _transform(permittivity_xx)],
        ]
    )


def transverse_permeability_fft_anisotropic(
    permeability_xx: jnp.ndarray,
    permeability_xy: jnp.ndarray,
    permeability_yx: jnp.ndarray,
    permeability_yy: jnp.ndarray,
    expansion: basis_mod.Expansion,
) -> jnp.ndarray:
    """The ``M_t`` matrix of an anisotropic layer, ``FFT`` (Laurent) rule.

    Mirrors ``fmmax.fmm_matrices.transverse_permeability_fft_anisotropic``::

        M_t = [[Z(μ_xx), Z(μ_xy)], [Z(μ_yx), Z(μ_yy)]].
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
    return jnp.block(
        [
            [_transform(permeability_xx), _transform(permeability_xy)],
            [_transform(permeability_yx), _transform(permeability_yy)],
        ]
    )


def transverse_permittivity_vector_anisotropic(
    permittivity_xx: jnp.ndarray,
    permittivity_xy: jnp.ndarray,
    permittivity_yx: jnp.ndarray,
    permittivity_yy: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    expansion: basis_mod.Expansion,
) -> jnp.ndarray:
    """The ``E_t`` matrix of an anisotropic layer, vector formulation.

    Port of ``fmmax.fmm_matrices.transverse_permittivity_vector_anisotropic``:
    the tensor is rotated into tangent/normal coordinates at the interface, the
    Fourier matrix is formed with Laurent's rule on the tangential/non-diagonal
    blocks and the inverse rule on the normal block, and rotated back. This is
    E2 of Liu eq. 51 generalized to a tensor (the tangential-field correction
    that makes high-contrast convergence behave).
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    (rotation_matrix, fourier_rotation_matrix, fourier_inverse_rotation_matrix) = _rotation_matrices(
        t00=ty,
        t01=tx.conj(),
        t10=-tx,
        t11=ty.conj(),
        expansion=expansion,
    )

    permittivity_tensor = jnp.block(
        [
            [
                permittivity_yy[..., jnp.newaxis, jnp.newaxis],
                -permittivity_yx[..., jnp.newaxis, jnp.newaxis],
            ],
            [
                -permittivity_xy[..., jnp.newaxis, jnp.newaxis],
                permittivity_xx[..., jnp.newaxis, jnp.newaxis],
            ],
        ]
    )
    assert rotation_matrix.shape[-2:] == permittivity_tensor.shape[-2:] == (2, 2)
    rotated_permittivity_tensor = jnp.linalg.solve(rotation_matrix, permittivity_tensor @ rotation_matrix)

    fourier_permittivity_matrix = jnp.block(
        [
            [
                _transform(rotated_permittivity_tensor[..., 0, 0]),
                _transform(rotated_permittivity_tensor[..., 0, 1]),
            ],
            [
                _transform(rotated_permittivity_tensor[..., 1, 0]),
                jnp.linalg.inv(_transform(1 / rotated_permittivity_tensor[..., 1, 1])),
            ],
        ]
    )
    return fourier_rotation_matrix @ fourier_permittivity_matrix @ fourier_inverse_rotation_matrix


def transverse_permeability_vector_anisotropic(
    permeability_xx: jnp.ndarray,
    permeability_xy: jnp.ndarray,
    permeability_yx: jnp.ndarray,
    permeability_yy: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    expansion: basis_mod.Expansion,
) -> jnp.ndarray:
    """The ``M_t`` matrix of an anisotropic layer, vector formulation.

    Port of ``fmmax.fmm_matrices.transverse_permeability_vector_anisotropic``.
    Note the rotation matrix differs from the permittivity one (there
    ``T [Et, En]^T = [-Ey, Ex]``, here ``T [Ht, Hn]^T = [Hx, Hy]``), matching
    fmmax's two distinct conventions.
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    (rotation_matrix, fourier_rotation_matrix, fourier_inverse_rotation_matrix) = _rotation_matrices(
        t00=tx,
        t01=-ty.conj(),
        t10=ty,
        t11=tx.conj(),
        expansion=expansion,
    )

    permeability_tensor = jnp.block(
        [
            [
                permeability_xx[..., jnp.newaxis, jnp.newaxis],
                permeability_xy[..., jnp.newaxis, jnp.newaxis],
            ],
            [
                permeability_yx[..., jnp.newaxis, jnp.newaxis],
                permeability_yy[..., jnp.newaxis, jnp.newaxis],
            ],
        ]
    )
    assert rotation_matrix.shape[-2:] == permeability_tensor.shape[-2:] == (2, 2)
    rotated_permeability_tensor = jnp.linalg.solve(rotation_matrix, permeability_tensor @ rotation_matrix)

    fourier_permeability_matrix = jnp.block(
        [
            [
                _transform(rotated_permeability_tensor[..., 0, 0]),
                _transform(rotated_permeability_tensor[..., 0, 1]),
            ],
            [
                _transform(rotated_permeability_tensor[..., 1, 0]),
                jnp.linalg.inv(_transform(1 / rotated_permeability_tensor[..., 1, 1])),
            ],
        ]
    )
    return fourier_rotation_matrix @ fourier_permeability_matrix @ fourier_inverse_rotation_matrix


def _rotation_matrices(
    t00: jnp.ndarray,
    t01: jnp.ndarray,
    t10: jnp.ndarray,
    t11: jnp.ndarray,
    expansion: basis_mod.Expansion,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Real-space and Fourier rotation matrices for the anisotropic vector rules.

    Port of ``fmmax.fmm_matrices._rotation_matrices``: the tangent/normal frame
    rotated into Fourier space so the rotated-tensor convolution stays consistent.
    """
    rotation_matrix = jnp.block(
        [
            [t00[..., jnp.newaxis, jnp.newaxis], t01[..., jnp.newaxis, jnp.newaxis]],
            [t10[..., jnp.newaxis, jnp.newaxis], t11[..., jnp.newaxis, jnp.newaxis]],
        ]
    )

    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)
    fourier_rotation_matrix = jnp.block(
        [
            [_transform(t00), _transform(t01)],
            [_transform(t10), _transform(t11)],
        ]
    )

    denom = jnp.abs(t00 * t11 - t10 * t01)
    denom_safe = jnp.where(jnp.isclose(denom, 0), 1.0, denom)
    fourier_inverse_rotation_matrix = jnp.block(
        [
            [_transform(t11 / denom_safe), _transform(-t01 / denom_safe)],
            [_transform(-t10 / denom_safe), _transform(t00 / denom_safe)],
        ]
    )
    return rotation_matrix, fourier_rotation_matrix, fourier_inverse_rotation_matrix


def transverse_matrices_anisotropic(
    permittivities: TensorComponents,
    permeabilities: Optional[TensorComponents],
    expansion: basis_mod.Expansion,
    formulation: Formulation | VectorFn,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
    vector_field_source: Optional[jnp.ndarray] = None,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    Optional[Tuple[jnp.ndarray, jnp.ndarray]],
]:
    """The six Fourier matrices of a patterned general-anisotropic layer.

    Mirrors ``fmmax.fmm._fourier_matrices_patterned_anisotropic_media``:
    returns ``(inverse_z_permittivity, z_permittivity, E_t,
    inverse_z_permeability, z_permeability, M_t, tangent_field)``. All Toeplitz
    transforms converge component-wise; the transverse matrices follow
    `formulation` (FFT or a vector scheme / callable hook), and the z-matrix
    inverse rules use the reciprocal of the zz component.

    Args:
        permittivities: ``(ε_xx, ε_xy, ε_yx, ε_yy, ε_zz)`` real-space grids.
        permeabilities: ``(μ_xx, μ_xy, μ_yx, μ_yy, μ_zz)``; ``None`` for
            nonmagnetic media (all-ones z and identity transverse matrices).
        expansion: The field expansion.
        formulation: ``Formulation.FFT`` / a vector scheme / a callable hook.
        primitive_lattice_vectors: Real-space lattice (vector gradients).
        vector_field_source: Density fed to the tangent-field solver for vector
            schemes (defaults to ``(ε_xx + ε_yy) / 2``, fmmax convention).
    """
    _transform = functools.partial(fft.fourier_convolution_matrix, expansion=expansion)

    (permittivity_xx, permittivity_xy, permittivity_yx, permittivity_yy, permittivity_zz) = permittivities
    inverse_z_permittivity = _transform(1 / permittivity_zz)
    z_permittivity = _transform(permittivity_zz)

    if permeabilities is None:
        inverse_z_permeability = None
        z_permeability = None
        n = expansion.num_terms
        ones = jnp.ones(permittivity_zz.shape[:-2] + (n,), dtype=jnp.result_type(permittivity_zz))
        z_permeability = utils.batch_diag(ones)
        inverse_z_permeability = utils.batch_diag(ones)
        m_t = jnp.block(
            [[z_permeability, jnp.zeros_like(z_permeability)], [jnp.zeros_like(z_permeability), z_permeability]]
        )
    else:
        (permeability_xx, permeability_xy, permeability_yx, permeability_yy, permeability_zz) = permeabilities
        inverse_z_permeability = _transform(1 / permeability_zz)
        z_permeability = _transform(permeability_zz)
        m_t = None  # decided below, per formulation

    if formulation is Formulation.FFT:
        e_t = transverse_permittivity_fft_anisotropic(
            permittivity_xx, permittivity_xy, permittivity_yx, permittivity_yy, expansion
        )
        if m_t is None:
            m_t = transverse_permeability_fft_anisotropic(
                permeability_xx, permeability_xy, permeability_yx, permeability_yy, expansion
            )
        tangent_field = None
    else:
        if vector_field_source is None:
            vector_field_source = (permittivity_xx + permittivity_yy) / 2
        tx, ty = tangent_field_for(formulation, vector_field_source, expansion, primitive_lattice_vectors)
        e_t = transverse_permittivity_vector_anisotropic(
            permittivity_xx, permittivity_xy, permittivity_yx, permittivity_yy, tx, ty, expansion
        )
        if m_t is None:
            m_t = transverse_permeability_vector_anisotropic(
                permeability_xx, permeability_xy, permeability_yx, permeability_yy, tx, ty, expansion
            )
        tangent_field = (tx, ty)
    return inverse_z_permittivity, z_permittivity, e_t, inverse_z_permeability, z_permeability, m_t, tangent_field


def tangent_field_for(
    formulation: Formulation | VectorFn,
    permittivity: jnp.ndarray,
    expansion: basis_mod.Expansion,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The tangent vector field ``(tx, ty)`` for `formulation` on `permittivity`.

    Accepts either a :class:`Formulation` (only those with an implemented
    scheme; others raise) or a callable hook of the fmmax ``VectorFn`` shape.

    Args:
        formulation: The formulation to use.
        permittivity: Real-space density driving the interfaces, ``(nx, ny)``.
        expansion, primitive_lattice_vectors: Field expansion and lattice.

    Returns:
        The tangent field components, each with the spatial shape of
        `permittivity`.
    """
    if callable(formulation):
        return formulation(permittivity, expansion, primitive_lattice_vectors)
    if not isinstance(formulation, Formulation):
        raise TypeError(f"`formulation` must be a Formulation or a callable, got {formulation!r}.")
    use_jones_direct, fourier_loss_weight, smoothness_loss_weight, tangent_post = _scheme(formulation)
    tx, ty = compute_tangent_field(
        permittivity=permittivity,
        expansion=expansion,
        primitive_lattice_vectors=primitive_lattice_vectors,
        use_jones_direct=use_jones_direct,
        fourier_loss_weight=fourier_loss_weight,
        smoothness_loss_weight=smoothness_loss_weight,
    )
    if tangent_post == "jones":
        # The Jones method converts the linear field into a complex elliptical
        # field of magnitude 1 (fmmax.vector.compute_field_jones).
        jxjy = _normalize_jones(jnp.stack([tx, ty], axis=-1))
        tx, ty = jxjy[..., 0], jxjy[..., 1]
    elif tangent_post == "normal":
        # The Normal method element-wise normalizes the linear field.
        txty = _normalize_elementwise(jnp.stack([tx, ty], axis=-1))
        tx, ty = txty[..., 0], txty[..., 1]
    return tx, ty


# -----------------------------------------------------------------------------
# k- and script-k matrices (Liu eqs. 19 and 23).
# -----------------------------------------------------------------------------


def script_k_matrix_patterned(
    z_permittivity_matrix: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
) -> jnp.ndarray:
    """``script_K = K_H^T Z_eps^{-1} K_H``, Liu eq. 19 / fmmax.

    Args:
        z_permittivity_matrix: Fourier matrix of the zz permittivity ``(2n, 2n)``
            (the inverse rule lives here).
        transverse_wavevectors: Per-order ``(n, 2)`` wavevectors.

    Returns:
        The ``2n x 2n`` script-k matrix.
    """
    dtype = jnp.promote_types(z_permittivity_matrix, transverse_wavevectors)
    kx = transverse_wavevectors[..., 0].astype(dtype)
    ky = transverse_wavevectors[..., 1].astype(dtype)
    z_inv_kx = jnp.linalg.solve(z_permittivity_matrix.astype(dtype), utils.batch_diag(kx))
    z_inv_ky = jnp.linalg.solve(z_permittivity_matrix.astype(dtype), utils.batch_diag(ky))
    return jnp.block(
        [
            [ky[..., :, jnp.newaxis] * z_inv_ky, -ky[..., :, jnp.newaxis] * z_inv_kx],
            [-kx[..., :, jnp.newaxis] * z_inv_ky, kx[..., :, jnp.newaxis] * z_inv_kx],
        ]
    )


def script_k_matrix_uniform(
    permittivity: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
) -> jnp.ndarray:
    """The per-order diagonal script_k for a uniform medium (Liu eq. 19)."""
    kx = transverse_wavevectors[..., 0]
    ky = transverse_wavevectors[..., 1]
    return jnp.block(
        [
            [
                utils.batch_diag(ky / permittivity[..., jnp.newaxis] * ky),
                utils.batch_diag(-ky / permittivity[..., jnp.newaxis] * kx),
            ],
            [
                utils.batch_diag(-kx / permittivity[..., jnp.newaxis] * ky),
                utils.batch_diag(kx / permittivity[..., jnp.newaxis] * kx),
            ],
        ]
    )


def k_matrix_patterned(
    transverse_wavevectors: jnp.ndarray,
    z_permeability_matrix: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """``K = K_E^T Z_mu^{-1} K_E``, Liu eq. 23 (fmmax, nonmagnetic default).

    Args:
        transverse_wavevectors: Per-order ``(n, 2)`` wavevectors.
        z_permeability_matrix: Fourier matrix of zz permeability; ``None`` for
            nonmagnetic media (the identity, so ``K = K_E^T K_E``).

    Returns:
        The ``2n x 2n`` k matrix.
    """
    kx = transverse_wavevectors[..., 0]
    ky = transverse_wavevectors[..., 1]
    if z_permeability_matrix is None:
        return jnp.block(
            [
                [utils.batch_diag(kx * kx), utils.batch_diag(kx * ky)],
                [utils.batch_diag(ky * kx), utils.batch_diag(ky * ky)],
            ]
        )
    dtype = jnp.promote_types(z_permeability_matrix, transverse_wavevectors)
    kx = kx.astype(dtype)
    ky = ky.astype(dtype)
    z_inv_kx = jnp.linalg.solve(z_permeability_matrix.astype(dtype), utils.batch_diag(kx))
    z_inv_ky = jnp.linalg.solve(z_permeability_matrix.astype(dtype), utils.batch_diag(ky))
    return jnp.block(
        [
            [kx[..., :, jnp.newaxis] * z_inv_kx, kx[..., :, jnp.newaxis] * z_inv_ky],
            [ky[..., :, jnp.newaxis] * z_inv_kx, ky[..., :, jnp.newaxis] * z_inv_ky],
        ]
    )


def omega_script_k_matrix(
    angular_frequency: jnp.ndarray,
    z_permittivity_matrix: jnp.ndarray,
    transverse_permeability_matrix: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
) -> jnp.ndarray:
    """``Omega_K = omega^2 M_t - script_K``, Liu eq. 26 (fmmax convention).

    Args:
        angular_frequency: Scalar ``2 pi / wavelength``.
        z_permittivity_matrix, transverse_permeability_matrix: Fourier matrices
            (``M_t`` for the transverse permeability).
        transverse_wavevectors: Per-order wavevectors.

    Returns:
        The ``2n x 2n`` omega-script-k matrix.
    """
    script_k = script_k_matrix_patterned(z_permittivity_matrix, transverse_wavevectors)
    omega_squared = angular_frequency[..., jnp.newaxis, jnp.newaxis] ** 2
    return omega_squared * transverse_permeability_matrix - script_k


# -----------------------------------------------------------------------------
# Tangent vector field generation (fmmax.vector conventions, no batch; the
# smoothness term IS implemented and used by the non-FOURIER schemes -- the
# JONES/NORMAL/POL have weight 2.0 and JONES_DIRECT 0.5, the _FOURIER four 0.0).
# -----------------------------------------------------------------------------


def compute_tangent_field(
    permittivity: jnp.ndarray,
    expansion: basis_mod.Expansion,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
    use_jones_direct: bool,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
    steps: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The tangent vector field ``(tx, ty)`` for `permittivity` (no batch).

    Port of ``fmmax.vector.compute_tangent_field`` restricted to a single
    real-space array: the Fourier coefficients of the tangent field are found by
    ``steps`` (static) Newton iterations of a quadratic alignment loss, then
    transformed back to real space. The density is grad-decoupled
    (``stop_gradient``), matching fmmax: design gradients flow through the
    permittivity itself, not through the interface geometry, so the field is a
    constant of the density for differentiation purposes.

    Args:
        permittivity: Real-space density ``(nx, ny)``.
        expansion: The field expansion.
        primitive_lattice_vectors: Real-space lattice (gradient geometry).
        use_jones_direct: Whether to seek the complex Jones field (direct
            method); otherwise the target is the plain rotated gradient.
        fourier_loss_weight: Weight of the high-frequency Fourier-penalty term
            (``0`` for the plain smoothness schemes).
        smoothness_loss_weight: Weight of the real-space smoothness-penalty term
            (``0`` for the ``_FOURIER`` schemes), fmmax convention.
        steps: Number of Newton iterations (static Python int).

    Returns:
        ``(tx, ty)``, each of shape ``(nx, ny)``.
    """
    if permittivity.ndim != 2:
        raise ValueError(f"`permittivity` must be a 2-D real-space grid, got shape {permittivity.shape}.")
    arr = jax.lax.stop_gradient(permittivity)
    grid_shape: Tuple[int, int] = arr.shape[-2:]

    # Rescale the weights so a supercell with proportionally more terms gives
    # the same field (fmmax convention).
    fourier_loss_weight /= expansion.num_terms
    smoothness_loss_weight /= expansion.num_terms

    grad = _compute_gradient(arr, primitive_lattice_vectors)
    gx = _filter_and_adjust_resolution(grad[..., 0], expansion)
    gy = _filter_and_adjust_resolution(grad[..., 1], expansion)
    grad = _normalize(jnp.stack([gx, gy], axis=-1))

    elementwise_alignment_weight = _field_magnitude(grad)

    # Provide a dummy gradient for the 1-D case, which avoids nans in the
    # Newton solve; the tangent field is then set manually.
    is_1d, grad_angle = _is_1d_field(grad)
    dummy_grad = jnp.broadcast_to(jnp.asarray([1, 1], dtype=jnp.complex64), grad.shape)
    grad = jnp.where(is_1d, dummy_grad, grad)

    # The target field the tangent field should align with: the rotated normal.
    target_field = jnp.stack([grad[..., 1], -grad[..., 0]], axis=-1)
    target_field = _normalize_elementwise(target_field)
    if use_jones_direct:
        target_field = _normalize_jones(target_field)
        initial_field = target_field
    else:
        initial_field = _normalize(jnp.stack([grad[..., 1], -grad[..., 0]], axis=-1))
    fourier_field = fft.fft(initial_field, expansion=expansion, axes=(-3, -2), centered_coordinates=False)

    flat_fourier_field = fourier_field.flatten()
    for _ in range(steps):
        _, jac, hessian = _field_loss_value_jac_and_hessian(
            flat_fourier_field=flat_fourier_field,
            expansion=expansion,
            primitive_lattice_vectors=primitive_lattice_vectors,
            target_field=target_field,
            elementwise_alignment_loss_weight=elementwise_alignment_weight,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
        )
        flat_fourier_field -= jnp.linalg.solve(hessian, jac.conj())
    fourier_field = flat_fourier_field.reshape((expansion.num_terms, 2))

    field = fft.ifft(
        fourier_field,
        expansion=expansion,
        shape=grid_shape,
        axis=-2,
        centered_coordinates=False,
    )

    # Manually set the tangent field in the 1-D case (fmmax convention).
    field_1d = jnp.stack([jnp.sin(grad_angle), jnp.cos(grad_angle)])
    field = jnp.where(is_1d, field_1d, field)
    field = _normalize(field)
    # Never-silent precision-class guard: the tangent field must share its precision
    # with the permittivity it discretizes (a mismatched pair would flow through the
    # Fourier matrices silently). A plain assert would (a) carry no message and (b)
    # vanish under `python -O`; raise an instructive ValueError instead. The comparison
    # is over static dtype objects, so this stays trace-safe.
    arr_prec = jnp.promote_types(arr.dtype, jnp.complex64)
    field_prec = jnp.promote_types(field.dtype, jnp.complex64)
    if arr_prec != field_prec:
        raise ValueError(
            f"tangent-field precision guard: permittivity promotes to {arr_prec} but the computed "
            f"tangent field promotes to {field_prec} -- they must share a precision class. "
            "Align the component-grid dtypes to the stack's dtype= (or pass dtype=jnp.complex128 "
            "under jax_enable_x64); the newer dtype can no longer be promoted consistently."
        )
    return field[..., 0], field[..., 1]


def _normalize_jones(field: jnp.ndarray) -> jnp.ndarray:
    """A Jones vector field following the "Jones" method of [2012 Liu]."""
    field = _normalize(field)
    magnitude = _field_magnitude(field)
    magnitude_near_zero = jnp.isclose(magnitude, 0.0)
    magnitude_safe = jnp.where(magnitude_near_zero, 1.0, magnitude)
    tx_norm = jnp.where(
        magnitude_near_zero,
        1 / jnp.sqrt(2),
        field[..., 0, jnp.newaxis] / magnitude_safe,
    )
    ty_norm = jnp.where(
        magnitude_near_zero,
        1 / jnp.sqrt(2),
        field[..., 1, jnp.newaxis] / magnitude_safe,
    )
    phi = jnp.pi / 8 * (1 + jnp.cos(jnp.pi * magnitude))
    theta = _angle(tx_norm + 1j * ty_norm)
    jx = jnp.exp(1j * theta) * (tx_norm * jnp.cos(phi) - ty_norm * 1j * jnp.sin(phi))
    jy = jnp.exp(1j * theta) * (ty_norm * jnp.cos(phi) + tx_norm * 1j * jnp.sin(phi))
    return jnp.concatenate([jx, jy], axis=-1)


def _normalize_elementwise(field: jnp.ndarray) -> jnp.ndarray:
    magnitude = _field_magnitude(field)
    magnitude_safe = jnp.where(jnp.isclose(magnitude, 0), 1, magnitude)
    return field / magnitude_safe


def _normalize(field: jnp.ndarray) -> jnp.ndarray:
    max_magnitude = _max_field_magnitude(field)
    max_magnitude_safe = jnp.where(jnp.isclose(max_magnitude, 0), 1, max_magnitude)
    return field / max_magnitude_safe


def _field_magnitude(field: jnp.ndarray) -> jnp.ndarray:
    magnitude_squared = jnp.sum(jnp.abs(field) ** 2, axis=-1, keepdims=True)
    is_zero = magnitude_squared == 0
    magnitude_squared_safe = jnp.where(is_zero, 1.0, magnitude_squared)
    return jnp.where(is_zero, 0.0, jnp.sqrt(magnitude_squared_safe))


def _max_field_magnitude(field: jnp.ndarray) -> jnp.ndarray:
    return jnp.amax(_field_magnitude(field), axis=(-3, -2), keepdims=True)


def _angle(x: jnp.ndarray) -> jnp.ndarray:
    is_near_zero = jnp.isclose(x, 0.0)
    x_safe = jnp.where(is_near_zero, (1.0 + 0.0j), x)
    return jnp.angle(x_safe)


def _filter_and_adjust_resolution(x: jnp.ndarray, expansion: basis_mod.Expansion) -> jnp.ndarray:
    y = fft.fft(x, expansion=expansion, centered_coordinates=False)
    min_shape = basis_mod.min_array_shape_for_expansion(expansion)
    assert x.ndim == 2
    doubled_min_shape = (
        min_shape[0] * (2 if x.shape[0] > 1 else 1),
        min_shape[1] * (2 if x.shape[1] > 1 else 1),
    )
    return fft.ifft(y, expansion=expansion, shape=doubled_min_shape, centered_coordinates=False)


def _is_1d_field(field: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Whether the field varies in one direction only (fmmax heuristic)."""
    ref_field = _field_at_max_magnitude(field)
    ref_angle = _angle(jnp.abs(ref_field[0]) + 1j * jnp.abs(ref_field[1]))
    angle = _angle(jnp.abs(field[..., 0]) + 1j * jnp.abs(field[..., 1]))
    magnitude = jnp.squeeze(_field_magnitude(field), axis=-1)
    is_1d = jnp.all(
        jnp.isclose(magnitude, 0.0)
        | jnp.isclose(angle, ref_angle - 2 * jnp.pi, atol=_ATOL_1D_FIELD_ANGLE)
        | jnp.isclose(angle, ref_angle - 1 * jnp.pi, atol=_ATOL_1D_FIELD_ANGLE)
        | jnp.isclose(angle, ref_angle, atol=_ATOL_1D_FIELD_ANGLE)
        | jnp.isclose(angle, ref_angle + 1 * jnp.pi, atol=_ATOL_1D_FIELD_ANGLE)
        | jnp.isclose(angle, ref_angle + 2 * jnp.pi, atol=_ATOL_1D_FIELD_ANGLE)
    )
    # Singleton spatial dimensions are automatically 1-D.
    assert field.ndim == 3
    is_1d = is_1d | (field.shape[0] == 1) | (field.shape[1] == 1)
    return is_1d, ref_angle


def _field_at_max_magnitude(field: jnp.ndarray) -> jnp.ndarray:
    assert field.ndim == 3
    magnitude = _field_magnitude(field)
    assert magnitude.shape[-1] == 1
    idx = jnp.argmax(magnitude.flatten())
    return field.reshape((-1, 2))[idx, :]


# -----------------------------------------------------------------------------
# Tangent-field loss: value, Jacobian, Hessian, and one static Newton step.
# -----------------------------------------------------------------------------


def _field_loss_value_jac_and_hessian(
    flat_fourier_field: jnp.ndarray,
    expansion: basis_mod.Expansion,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
    target_field: jnp.ndarray,
    elementwise_alignment_loss_weight: jnp.ndarray,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """:math:`(value, jac, hessian)` of the real alignment loss (holomorphic)."""
    assert flat_fourier_field.ndim == 1

    def fn(flat_fourier_field):
        fourier_field = jnp.reshape(flat_fourier_field, (-1, 2))
        value = _field_loss(
            fourier_field=fourier_field,
            expansion=expansion,
            primitive_lattice_vectors=primitive_lattice_vectors,
            target_field=target_field,
            elementwise_alignment_loss_weight=elementwise_alignment_loss_weight,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
        )
        return value, value

    def jac_fn(flat_fourier_field):
        jac, value = jax.jacrev(fn, has_aux=True)(flat_fourier_field)
        return jac, (value, jac)

    hessian, (value, jac) = jax.jacrev(jac_fn, has_aux=True, holomorphic=True)(flat_fourier_field)
    return value, jac, hessian


def _field_loss(
    fourier_field: jnp.ndarray,
    expansion: basis_mod.Expansion,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
    target_field: jnp.ndarray,
    elementwise_alignment_loss_weight: jnp.ndarray,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
) -> jnp.ndarray:
    """The real-space alignment loss plus the weighted Fourier/smoothness penalties."""
    shape: Tuple[int, int] = target_field.shape[-3:-1]
    field = fft.ifft(
        y=fourier_field,
        expansion=expansion,
        shape=shape,
        axis=-2,
        centered_coordinates=False,
    )
    loss = _alignment_loss(field, target_field, elementwise_alignment_loss_weight)
    if fourier_loss_weight > 0:
        loss += fourier_loss_weight * _fourier_loss(fourier_field, expansion, primitive_lattice_vectors)
    if smoothness_loss_weight > 0:
        loss += smoothness_loss_weight * _smoothness_loss(field, primitive_lattice_vectors)
    return loss


def _alignment_loss(
    field: jnp.ndarray,
    target_field: jnp.ndarray,
    elementwise_alignment_loss_weight: jnp.ndarray,
) -> jnp.ndarray:
    assert elementwise_alignment_loss_weight.ndim == field.ndim
    elementwise_loss = jnp.sum(jnp.abs(field - target_field) ** 2, axis=-1, keepdims=True)
    return jnp.mean(elementwise_alignment_loss_weight * elementwise_loss)


def _fourier_loss(
    fourier_field: jnp.ndarray,
    expansion: basis_mod.Expansion,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
) -> jnp.ndarray:
    """Penalty on high-frequency Fourier terms of the tangent field."""
    transverse_wavevectors = basis_mod.transverse_wavevectors(
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        in_plane_wavevector=jnp.zeros((2,), dtype=fourier_field.real.dtype),
    )
    basis_vectors = jnp.stack([primitive_lattice_vectors.u, primitive_lattice_vectors.v], axis=-1)
    area = jnp.abs(jnp.linalg.det(basis_vectors))
    kt = jnp.linalg.norm(transverse_wavevectors, axis=-1) * jnp.sqrt(area)
    return jnp.sum(jnp.abs(fourier_field) ** 2 * kt[..., jnp.newaxis] ** 2)


def _smoothness_loss(field: jnp.ndarray, primitive_lattice_vectors: basis_mod.LatticeVectors) -> jnp.ndarray:
    """The real-space smoothness loss of the tangent field (fmmax.vector).

    Penalizes large forward-difference gradients of the tangent vector field,
    evaluated on the real-space grid; scaled by the (sqrt-area-normalized)
    lattice so the value is unit-cell-size independent.
    """
    basis_vectors = jnp.stack([primitive_lattice_vectors.u, primitive_lattice_vectors.v], axis=-1)
    area = jnp.abs(jnp.linalg.det(basis_vectors))
    normalized_basis = basis_vectors / jnp.sqrt(area)
    grads = _field_forward_difference_gradient(field, normalized_basis)
    return jnp.mean(jnp.abs(jnp.asarray(grads)) ** 2)


def _field_forward_difference_gradient(field: jnp.ndarray, basis_vectors: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-difference gradient of each real-space component of `field`."""
    return (
        _scalar_forward_difference_gradient(field[..., 0], basis_vectors),
        _scalar_forward_difference_gradient(field[..., 1], basis_vectors),
    )


def _scalar_forward_difference_gradient(field: jnp.ndarray, basis_vectors: jnp.ndarray) -> jnp.ndarray:
    """Forward-difference gradient of a scalar field in the lattice basis."""
    axes = tuple(range(field.ndim - basis_vectors.shape[-1], field.ndim))
    partial_grad = jnp.stack([_periodic_forward_difference(field, axis=ax) for ax in axes], axis=-1)
    return _transform_gradient(partial_grad, basis_vectors)


def _periodic_forward_difference(field: jnp.ndarray, axis: int) -> jnp.ndarray:
    """``field[i+1] - field[i]`` with periodic wrap, scaled by the axis length."""
    diff = jnp.roll(field, shift=-1, axis=axis) - field
    return diff * field.shape[axis]


# -----------------------------------------------------------------------------
# Gradient of a (stop-grad) density, in lattice coordinates (fmmax.vector).
# -----------------------------------------------------------------------------


def _compute_gradient(arr: jnp.ndarray, primitive_lattice_vectors: basis_mod.LatticeVectors) -> jnp.ndarray:
    """The scaled ``(..., ny, ny, 2)`` gradient of `arr` in the lattice basis."""
    basis_vectors = jnp.stack([primitive_lattice_vectors.u, primitive_lattice_vectors.v], axis=-1)
    area = jnp.abs(jnp.linalg.det(basis_vectors))
    basis_vectors /= jnp.sqrt(area)

    pad_width = tuple([(0, 0)] * (arr.ndim - 2) + [(1, 1)] * 2)
    arr_padded = jnp.pad(arr, pad_width, mode="wrap")
    axes = tuple(range(-2, 0))
    partial_grad = jnp.gradient(arr_padded, axis=axes)
    partial_grad = [_unpad(g, pad_width) for g in partial_grad]
    partial_grad = [g * arr.shape[ax] for g, ax in zip(partial_grad, axes)]
    return _transform_gradient(jnp.stack(partial_grad, axis=-1), basis_vectors)


def _transform_gradient(partial_grad: jnp.ndarray, basis_vectors: jnp.ndarray) -> jnp.ndarray:
    metric_tensor = _metric_tensor(basis_vectors).astype(partial_grad.dtype)
    inverse_metric_tensor = jnp.linalg.inv(metric_tensor)
    return partial_grad @ inverse_metric_tensor @ basis_vectors.T


def _metric_tensor(basis_vectors: jnp.ndarray) -> jnp.ndarray:
    return basis_vectors.T @ basis_vectors


def _unpad(arr: jnp.ndarray, pad_width: Tuple[Tuple[int, int], ...]) -> jnp.ndarray:
    slices = [slice(lo, size - hi) for (lo, hi), size in zip(pad_width, arr.shape)]
    return arr[tuple(slices)]
