"""The Fourier matrices of a patterned layer: transverse permittivity and permeability, and the K dyadics.

Notation (Liu & Fan, Comput. Phys. Commun. 183, 2233 (2012), eqs. 15-28, whose state ordering the backend uses):
the tangential state is ``u = (-E_y, E_x)``, ``v = (H_x, H_y)``, and ``[[f]]`` is Laurent's Toeplitz matrix of a
grid ``f`` (``fft.fourier_convolution_matrix``). The layer operator ``M_L = [[0, B], [C, 0]]`` (``layer.py``) is
assembled from

* ``E_t``, the transverse permittivity in the ``u`` ordering, with ``D_u = E_t u``;
* ``M_t``, the transverse permeability in the ``v`` ordering (the identity for non-magnetic media);
* ``script_K = K_H^T [[eps_zz]]^-1 K_H`` and ``K = K_E^T [[mu_zz]]^-1 K_E`` with the diagonal wavevector matrices.

Factorization rules (Li, JOSA A 13, 1870 (1996)). Inside a layer every interface is vertical, so ``E_z`` and
``H_z`` are tangential to it and continuous, while ``D_z = eps_zz E_z`` and ``B_z`` jump with ``eps_zz``: that is a
product of two discontinuous functions with a continuous result, and the inverse rule applies, ``E_z = [[eps_zz]]^-1
D_z``. In the transverse plane the tangential component of ``E`` is continuous (Laurent's rule for it) and the
normal component jumps (the inverse rule, ``[[1/eps]]^-1``). With a tangent field ``t`` and the projector ``P``
onto the normal direction,

    E_t = [[eps]] - ([[eps]] - [[1/eps]]^-1) [[P]]                                                  (normal-vector)

(``FFT`` drops the correction). In the ``u = (-E_y, E_x)`` ordering the normal projector of ``t = (t_x, t_y)`` is
``P = [[|t_x|^2, t_x* t_y], [t_x t_y*, |t_y|^2]] / |t|^2`` (the rotation by 90 degrees that maps the field to ``u``
turns the normal projector in ``(x, y)`` into this matrix). It is invariant to the scale of ``t``. A tensor layer is
rotated into the frame of ``t`` and the normal, gets Laurent's rule on the three blocks that do not divide by the
normal component and the inverse rule on the one that does, and is rotated back, with the pointwise-unit frame
of ``t`` (a unitary rotation, so that the rule keeps a lossless tensor Hermitian).
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import jax.numpy as jnp

from . import basis as basis_mod
from . import fft
from .formulation import Formulation, VectorFn
from .tangent import compute_tangent_field, tangent_field_for

__all__ = [
    "Formulation",
    "VectorFn",
    "compute_tangent_field",
    "tangent_field_for",
    "transverse_permittivity_fft",
    "transverse_permittivity_vector",
    "transverse_matrices_isotropic",
    "transverse_permittivity_fft_anisotropic",
    "transverse_permeability_fft_anisotropic",
    "transverse_permittivity_vector_anisotropic",
    "transverse_permeability_vector_anisotropic",
    "transverse_matrices_anisotropic",
    "script_k_matrix_patterned",
    "k_matrix_patterned",
    "omega_script_k_matrix",
]

TensorComponents = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def _laurent(expansion: basis_mod.Expansion):
    return functools.partial(fft.fourier_convolution_matrix, expansion=expansion)


def _diag(x: jnp.ndarray) -> jnp.ndarray:
    """Batched diagonal matrix of the last axis of ``x``."""
    return x[..., :, None] * jnp.eye(x.shape[-1], dtype=x.dtype)


def _block_diag2(a: jnp.ndarray) -> jnp.ndarray:
    z = jnp.zeros_like(a)
    return jnp.block([[a, z], [z, a]])


def _abs2(x: jnp.ndarray) -> jnp.ndarray:
    """``|x|^2`` as a polynomial (``re^2 + im^2``), so that every derivative exists at 0."""
    return jnp.real(x) ** 2 + jnp.imag(x) ** 2


# -----------------------------------------------------------------------------
# Isotropic layers.
# -----------------------------------------------------------------------------


def transverse_permittivity_fft(permittivity: jnp.ndarray, expansion: basis_mod.Expansion) -> jnp.ndarray:
    """``E_t = diag([[eps]], [[eps]])``: Laurent's rule for both transverse components, ``(..., 2n, 2n)``."""
    return _block_diag2(fft.fourier_convolution_matrix(permittivity, expansion))


def normal_projector(tx: jnp.ndarray, ty: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The pointwise blocks ``(P_00, P_01, P_10, P_11)`` of the normal projector in the ``u`` ordering (module
    docstring). Where ``t = 0`` (no interface) the projector is 0, i.e. Laurent's rule."""
    norm2 = _abs2(tx) + _abs2(ty)
    safe = jnp.where(jnp.isclose(norm2, 0), 1.0, norm2)
    return _abs2(tx) / safe, jnp.conj(tx) * ty / safe, tx * jnp.conj(ty) / safe, _abs2(ty) / safe


def transverse_permittivity_vector(
    permittivity: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    expansion: basis_mod.Expansion,
) -> jnp.ndarray:
    """``E_t = [[eps]] - ([[eps]] - [[1/eps]]^-1) [[P]]`` for the tangent field ``(tx, ty)``, ``(..., 2n, 2n)``."""
    t = _laurent(expansion)
    eps = t(permittivity)
    delta = _block_diag2(eps - jnp.linalg.inv(t(1 / permittivity)))
    p00, p01, p10, p11 = normal_projector(tx, ty)
    return _block_diag2(eps) - delta @ jnp.block([[t(p00), t(p01)], [t(p10), t(p11)]])


def transverse_matrices_isotropic(
    permittivity: jnp.ndarray,
    expansion: basis_mod.Expansion,
    formulation: Formulation | VectorFn,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
    """``([[1/eps]], [[eps]], E_t, tangent_field)`` of a patterned isotropic layer (``tangent_field`` is ``None``
    for ``FFT``). ``[[eps]]`` here is the zz matrix, whose inverse enters ``script_K`` (the inverse rule for E_z)."""
    t = _laurent(expansion)
    if formulation is Formulation.FFT:
        e_t, field = transverse_permittivity_fft(permittivity, expansion), None
    else:
        field = tangent_field_for(formulation, permittivity, expansion, primitive_lattice_vectors)
        e_t = transverse_permittivity_vector(permittivity, *field, expansion)
    return t(1 / permittivity), t(permittivity), e_t, field


# -----------------------------------------------------------------------------
# Anisotropic layers: 5 permittivity components, optionally 5 permeability components.
# -----------------------------------------------------------------------------


def transverse_permittivity_fft_anisotropic(eps_xx, eps_xy, eps_yx, eps_yy, expansion) -> jnp.ndarray:
    """Laurent's ``E_t`` of a tensor: in the ``u = (-E_y, E_x)`` ordering the tensor reads
    ``[[eps_yy, -eps_yx], [-eps_xy, eps_xx]]``, each block a Toeplitz matrix."""
    t = _laurent(expansion)
    return jnp.block([[t(eps_yy), t(-eps_yx)], [t(-eps_xy), t(eps_xx)]])


def transverse_permeability_fft_anisotropic(mu_xx, mu_xy, mu_yx, mu_yy, expansion) -> jnp.ndarray:
    """Laurent's ``M_t`` of a tensor in the ``v = (H_x, H_y)`` ordering: ``[[mu_xx, mu_xy], [mu_yx, mu_yy]]``."""
    t = _laurent(expansion)
    return jnp.block([[t(mu_xx), t(mu_xy)], [t(mu_yx), t(mu_yy)]])


def _pointwise2(a, b, c, d) -> jnp.ndarray:
    """A pointwise ``2 x 2`` matrix field ``(..., nx, ny, 2, 2)`` from its four entry grids."""
    return jnp.stack([jnp.stack([a, b], -1), jnp.stack([c, d], -1)], -2)


def _rotation_matrices(t00, t01, t10, t11, expansion):
    """The frame rotation ``R`` (pointwise ``(..., 2, 2)``), its Toeplitz matrix ``[[R]]`` and the Toeplitz
    matrix of its inverse, ``[[adj(R) / |det R|]]`` (``det R = |t|^2`` is real for the frames used here)."""
    t = _laurent(expansion)
    det = jnp.abs(t00 * t11 - t10 * t01)
    det = jnp.where(jnp.isclose(det, 0), 1.0, det)
    return (
        _pointwise2(t00, t01, t10, t11),
        jnp.block([[t(t00), t(t01)], [t(t10), t(t11)]]),
        jnp.block([[t(t11 / det), t(-t01 / det)], [t(-t10 / det), t(t00 / det)]]),
    )


def _unit_field(tx, ty):
    """``t / |t|`` pointwise (``(1, 0)`` where ``t = 0``, where no interface constrains the frame).

    The tensor rule needs a *unitary* frame: then ``R^-1 = R^H`` pointwise, so ``[[R^-1]] = [[R]]^H`` exactly and
    ``[[R]] F [[R]]^H`` is Hermitian whenever the tensor is, i.e. a lossless tensor layer conserves energy. With
    the raw least-squares field (``|t|`` varies) the product lost up to 1.2 % of the power on a lossless cell
    (issue #409)."""
    m = jnp.sqrt(_abs2(tx) + _abs2(ty))
    zero = jnp.isclose(m, 0)
    m = jnp.where(zero, 1.0, m)
    return jnp.where(zero, 1.0, tx / m), jnp.where(zero, 0.0, ty / m)


def _rotated_rule(tensor, rotation, expansion) -> jnp.ndarray:
    """``[[R]] F [[R^-1]]`` where ``F`` applies Laurent's rule to ``R^-1 tensor R`` except for its ``(1, 1)`` (normal)
    block, which takes the inverse rule."""
    r, r_fourier, r_inv_fourier = rotation
    t = _laurent(expansion)
    rotated = jnp.linalg.solve(r, tensor @ r)
    f = jnp.block(
        [
            [t(rotated[..., 0, 0]), t(rotated[..., 0, 1])],
            [t(rotated[..., 1, 0]), jnp.linalg.inv(t(1 / rotated[..., 1, 1]))],
        ]
    )
    return r_fourier @ f @ r_inv_fourier


def transverse_permittivity_vector_anisotropic(eps_xx, eps_xy, eps_yx, eps_yy, tx, ty, expansion) -> jnp.ndarray:
    """Vector-rule ``E_t`` of a tensor. The frame ``R = [[t_y, t_x*], [-t_x, t_y*]]`` maps (tangential, normal)
    components to the ``u`` ordering; the normal one is the second."""
    tensor = _pointwise2(eps_yy, -eps_yx, -eps_xy, eps_xx)
    tx, ty = _unit_field(tx, ty)
    return _rotated_rule(tensor, _rotation_matrices(ty, tx.conj(), -tx, ty.conj(), expansion), expansion)


def transverse_permeability_vector_anisotropic(mu_xx, mu_xy, mu_yx, mu_yy, tx, ty, expansion) -> jnp.ndarray:
    """Vector-rule ``M_t`` of a tensor, in the ``v = (H_x, H_y)`` ordering, where the frame is
    ``R = [[t_x, -t_y*], [t_y, t_x*]]`` (tangential, normal)."""
    tensor = _pointwise2(mu_xx, mu_xy, mu_yx, mu_yy)
    tx, ty = _unit_field(tx, ty)
    return _rotated_rule(tensor, _rotation_matrices(tx, -ty.conj(), ty, tx.conj(), expansion), expansion)


def transverse_matrices_anisotropic(
    permittivities: TensorComponents,
    permeabilities: Optional[TensorComponents],
    expansion: basis_mod.Expansion,
    formulation: Formulation | VectorFn,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
    vector_field_source: Optional[jnp.ndarray] = None,
):
    """``([[1/eps_zz]], [[eps_zz]], E_t, [[1/mu_zz]], [[mu_zz]], M_t, tangent_field)`` of a patterned tensor layer.

    ``permeabilities`` ``None`` means non-magnetic (identity ``M_t`` and zz matrices). The tangent field of a vector
    formulation follows ``vector_field_source``, by default the mean transverse permittivity ``(eps_xx + eps_yy)/2``.
    """
    t = _laurent(expansion)
    exx, exy, eyx, eyy, ezz = permittivities
    if permeabilities is None:
        eye = _diag(jnp.ones(ezz.shape[:-2] + (expansion.num_terms,), dtype=jnp.result_type(ezz)))
        inv_z_mu, z_mu, m_t = eye, eye, _block_diag2(eye)
    else:
        mxx, mxy, myx, myy, mzz = permeabilities
        inv_z_mu, z_mu, m_t = t(1 / mzz), t(mzz), None
    if formulation is Formulation.FFT:
        e_t, field = transverse_permittivity_fft_anisotropic(exx, exy, eyx, eyy, expansion), None
        if m_t is None:
            m_t = transverse_permeability_fft_anisotropic(mxx, mxy, myx, myy, expansion)
    else:
        source = (exx + eyy) / 2 if vector_field_source is None else vector_field_source
        field = tangent_field_for(formulation, source, expansion, primitive_lattice_vectors)
        e_t = transverse_permittivity_vector_anisotropic(exx, exy, eyx, eyy, *field, expansion)
        if m_t is None:
            m_t = transverse_permeability_vector_anisotropic(mxx, mxy, myx, myy, *field, expansion)
    return t(1 / ezz), t(ezz), e_t, inv_z_mu, z_mu, m_t, field


# -----------------------------------------------------------------------------
# The K dyadics.
# -----------------------------------------------------------------------------


def _k_split(matrix, transverse_wavevectors):
    dtype = jnp.promote_types(matrix.dtype, transverse_wavevectors.dtype)
    kx = transverse_wavevectors[..., 0].astype(dtype)
    ky = transverse_wavevectors[..., 1].astype(dtype)
    m = matrix.astype(dtype)
    return kx, ky, jnp.linalg.solve(m, _diag(kx)), jnp.linalg.solve(m, _diag(ky))


def script_k_matrix_patterned(z_permittivity_matrix: jnp.ndarray, transverse_wavevectors: jnp.ndarray) -> jnp.ndarray:
    """``script_K = K_H^T [[eps_zz]]^-1 K_H`` with ``K_H = (K_y, -K_x)``: ``[[K_y Z K_y, -K_y Z K_x], [-K_x Z K_y,
    K_x Z K_x]]``, ``Z = [[eps_zz]]^-1`` (the inverse rule for ``E_z``)."""
    kx, ky, z_kx, z_ky = _k_split(z_permittivity_matrix, transverse_wavevectors)
    return jnp.block(
        [[ky[..., :, None] * z_ky, -ky[..., :, None] * z_kx], [-kx[..., :, None] * z_ky, kx[..., :, None] * z_kx]]
    )


def k_matrix_patterned(
    transverse_wavevectors: jnp.ndarray, z_permeability_matrix: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """``K = K_E^T [[mu_zz]]^-1 K_E`` with ``K_E = (K_x, K_y)``; for ``None`` (non-magnetic) ``[[mu_zz]] = I``."""
    if z_permeability_matrix is None:
        kx, ky = transverse_wavevectors[..., 0], transverse_wavevectors[..., 1]
        return jnp.block([[_diag(kx * kx), _diag(kx * ky)], [_diag(ky * kx), _diag(ky * ky)]])
    kx, ky, z_kx, z_ky = _k_split(z_permeability_matrix, transverse_wavevectors)
    return jnp.block(
        [[kx[..., :, None] * z_kx, kx[..., :, None] * z_ky], [ky[..., :, None] * z_kx, ky[..., :, None] * z_ky]]
    )


def omega_script_k_matrix(angular_frequency, z_permittivity_matrix, transverse_permeability_matrix, transverse_wavevectors):
    """``omega^2 M_t - script_K`` (``omega B`` of the layer operator)."""
    script_k = script_k_matrix_patterned(z_permittivity_matrix, transverse_wavevectors)
    return angular_frequency[..., None, None] ** 2 * transverse_permeability_matrix - script_k
