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
permittivity itself). The remaining vector schemes and the callable hook are
wired by the stage-2 task; this module already accepts a callable.

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
#: ``FFT`` uses Laurent's rule; every other member is a vector formulation.
#: Stage 2 wires the full suite; for now ``FFT`` and ``JONES_DIRECT_FOURIER``
#: are implemented and the rest raise a loud ``NotImplementedError``.
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

#: Vector-scheme weights, matching ``fmmax.vector`` for the default formulation.
_FOURIER_LOSS_WEIGHT_JONES_DIRECT = 0.05
_SMOOTHNESS_LOSS_WEIGHT = 2.0
_SMOOTHNESS_LOSS_WEIGHT_JONES_DIRECT = 0.5
_FOURIER_LOSS_WEIGHT = 0.2

#: Absolute tolerance below which the tangent field counts as 1-D.
_ATOL_1D_FIELD_ANGLE = 1e-2


# -----------------------------------------------------------------------------
# Formulation dispatch.
# -----------------------------------------------------------------------------


def _scheme(formulation: Formulation) -> Tuple[bool, float, float]:
    """Vector-scheme parameters ``(use_jones_direct, fourier_w, smoothness_w)``."""
    if formulation is Formulation.JONES_DIRECT_FOURIER:
        return True, _FOURIER_LOSS_WEIGHT_JONES_DIRECT, 0.0
    # The remaining schemes are stage-2 work; the enum members exist so the
    # names/formulation strings already match the seam.
    raise NotImplementedError(
        f"formulation {formulation.value!r} is not implemented yet (stage 2); "
        f"available now: 'FFT', 'JONES_DIRECT_FOURIER', or a custom callable hook."
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
    use_jones_direct, fourier_loss_weight, smoothness_loss_weight = _scheme(formulation)
    tx, ty = compute_tangent_field(
        permittivity=permittivity,
        expansion=expansion,
        primitive_lattice_vectors=primitive_lattice_vectors,
        use_jones_direct=use_jones_direct,
        fourier_loss_weight=fourier_loss_weight,
        smoothness_loss_weight=smoothness_loss_weight,
    )
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
# Tangent vector field generation (fmmax.vector conventions, no batch, no
# smoothness term -- the wired schemes use a zero smoothness weight).
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
        fourier_loss_weight: Weight of the high-frequency Fourier-penalty term.
        smoothness_loss_weight: Must be ``0.0`` (not yet wired; the stage-2
            schemes that use it will land with the rest of the suite).
        steps: Number of Newton iterations (static Python int).

    Returns:
        ``(tx, ty)``, each of shape ``(nx, ny)``.
    """
    if not float(smoothness_loss_weight) == 0.0:
        raise NotImplementedError(
            "the tangent-field smoothness loss is not wired yet (stage 2); "
            f"got smoothness_loss_weight={smoothness_loss_weight}."
        )
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
    assert jnp.promote_types(arr.dtype, jnp.complex64) == jnp.promote_types(field.dtype, jnp.complex64)
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
    del smoothness_loss_weight  # asserted zero by the caller
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
) -> jnp.ndarray:
    """The real-space alignment loss plus the weighted Fourier penalty."""
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
