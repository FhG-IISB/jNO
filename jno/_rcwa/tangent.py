"""The tangent vector field of the interfaces, for the vector Fourier factorizations.

A vector formulation needs, at every point of the cell, the direction tangent to the material interfaces (see
``matrices.transverse_permittivity_vector``). This module builds it from the permittivity grid. The scheme here
is the one fmmax introduced (Schubert & Hammond, arXiv:2308.08573): the Fourier coefficients of the field are
one Newton step of a quadratic loss that aligns the field with the rotated, band-limited gradient of the
permittivity, with an optional Fourier-domain or real-space smoothness penalty; the JONES variants turn the field
into a complex elliptical (Jones) field, NORMAL normalises it pointwise. Unlike fmmax, the field is
differentiated through (the interface moves with the design) and the degenerate inputs (no edge, a 1-D grating)
have finite derivatives.

Stage S4c of the RCWA plan replaces this construction by jNO's own, with a smooth scale in place of the maximum
(git-bug a3cd6ef: the ``amax`` makes T kink at symmetric designs).
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from . import basis as basis_mod
from . import fft
from .formulation import Formulation, VectorFn

#: Vector-scheme weights, matching ``fmmax.vector`` member for member.
_FOURIER_LOSS_WEIGHT_JONES_DIRECT = 0.05
_SMOOTHNESS_LOSS_WEIGHT = 2.0
_SMOOTHNESS_LOSS_WEIGHT_JONES_DIRECT = 0.5
_FOURIER_LOSS_WEIGHT = 0.2

#: Absolute tolerance below which the tangent field counts as 1-D.
_ATOL_1D_FIELD_ANGLE = 1e-2


def _abs2(x: jnp.ndarray) -> jnp.ndarray:
    """``|x|**2`` as ``re**2 + im**2``: the same value as ``jnp.abs(x) ** 2``, but a polynomial, so every
    derivative exists at ``x = 0``. The tangent-field Newton step differentiates its loss twice, and the
    design gradient once more; through ``abs`` that third derivative is 0 * inf = NaN wherever the field
    meets its target exactly (a 1-D grating)."""
    return jnp.real(x) ** 2 + jnp.imag(x) ** 2


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
    transformed back to real space. The loss is quadratic in the Fourier
    coefficients, so the Newton step is one linear solve and the field is a
    smooth function of the density away from the degenerate inputs: it is
    differentiated through (fmmax stops the gradient here, which drops the
    term of a moving interface; see ``tests/test_rcwa_gradients.py``). A grid
    without any edge and a 1-D grating take the safe branches below, whose
    derivatives are finite.

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
    arr = permittivity
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
    magnitude_squared = jnp.sum(_abs2(field), axis=-1, keepdims=True)
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
    elementwise_loss = jnp.sum(_abs2(field - target_field), axis=-1, keepdims=True)
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
    return jnp.sum(_abs2(fourier_field) * kt[..., jnp.newaxis] ** 2)


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
    return jnp.mean(_abs2(jnp.asarray(grads)))


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
