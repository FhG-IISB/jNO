"""The tangent vector field of the interfaces, for the vector Fourier factorizations.

A vector formulation needs a field ``t(r)`` along the material interfaces (``matrices.transverse_permittivity_vector``
uses the projector onto its normal). Following Schubert & Hammond (arXiv:2308.08573), ``t`` is the smoothest
band-limited field that aligns with the rotated permittivity gradient where the gradient is strong:

    t = argmin_c  mean_r w(r) |t_c(r) - t*(r)|^2  +  lambda_F sum_G |k_G|^2 A |c_G|^2  +  lambda_S mean_r |grad t_c|^2

over the Fourier coefficients ``c_G`` of ``t`` on the expansion's orders. ``t*`` is the unit vector along the
interface (the gradient ``g`` of the permittivity rotated by 90 degrees; for the JONES_DIRECT schemes its Jones form,
circular where ``g = 0``), the weight is ``w = |g| / s`` with a scale ``s`` of the gradient, ``A`` the cell area,
``lambda_F`` the Fourier-domain and ``lambda_S`` the real-space smoothness penalty (both per term). All three terms
are quadratic and diagonal in the two field components, so the minimiser is closed-form, two ``n x n`` Hermitian
solves (where an autodiff Hessian of the loss would be ``2n x 2n``):

    ([[w]] + diag(lambda_F |k_G|^2 A + lambda_S |d_G C|^2 / 4)) c = coefficients of (w t*),

with ``[[w]]`` the Toeplitz matrix of ``w``, ``d_G`` the symbols of the periodic forward differences on the grid and
``C`` the map from grid to lattice-Cartesian derivatives. The JONES schemes turn the solved field into a complex
elliptical field (Antos 2009; Liu & Fan 2012), NORMAL normalises it pointwise, POL uses it as it is.

The scale ``s`` is a smooth p-norm of ``|g|`` (``p = 32``), not its maximum: the maximum has a kink wherever pixels
tie for it, which a symmetric design always has, and T inherited that kink (git-bug a3cd6ef). The field is
differentiated through, and the degenerate inputs (no edge, a 1-D grating) have finite derivatives.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as onp

from . import basis as basis_mod
from . import fft
from .formulation import Formulation, VectorFn

#: Penalty weights of the schemes (Schubert & Hammond's choices), per term of the expansion.
_FOURIER_LOSS_WEIGHT_JONES_DIRECT = 0.05
_SMOOTHNESS_LOSS_WEIGHT = 2.0
_SMOOTHNESS_LOSS_WEIGHT_JONES_DIRECT = 0.5
_FOURIER_LOSS_WEIGHT = 0.2

#: Exponent of the p-norm that scales the gradient (``||g||_p >= max |g|``, smooth wherever ``g != 0``).
_SCALE_P = 32
#: Use the plain maximum instead (only to compare with the published construction; it kinks at ties).
_HARD_MAX = False

#: Angle tolerance (radians) below which the gradient counts as 1-D.
_ATOL_1D_FIELD_ANGLE = 1e-2


def _abs2(x: jnp.ndarray) -> jnp.ndarray:
    """``|x|^2`` as ``re^2 + im^2``: a polynomial, so every derivative exists at ``x = 0`` (through ``abs`` the
    design gradient of a 1-D grating's field would be ``0 * inf = NaN``)."""
    return jnp.real(x) ** 2 + jnp.imag(x) ** 2


def _scheme(formulation: Formulation) -> Tuple[bool, float, float, Optional[str]]:
    """``(jones_direct, fourier_weight, smoothness_weight, post)`` of a vector formulation; ``post`` is ``None``,
    ``"jones"`` or ``"normal"``."""
    table = {
        Formulation.JONES_DIRECT: (True, 0.0, _SMOOTHNESS_LOSS_WEIGHT_JONES_DIRECT, None),
        Formulation.JONES: (False, 0.0, _SMOOTHNESS_LOSS_WEIGHT, "jones"),
        Formulation.NORMAL: (False, 0.0, _SMOOTHNESS_LOSS_WEIGHT, "normal"),
        Formulation.POL: (False, 0.0, _SMOOTHNESS_LOSS_WEIGHT, None),
        Formulation.JONES_DIRECT_FOURIER: (True, _FOURIER_LOSS_WEIGHT_JONES_DIRECT, 0.0, None),
        Formulation.JONES_FOURIER: (False, _FOURIER_LOSS_WEIGHT, 0.0, "jones"),
        Formulation.NORMAL_FOURIER: (False, _FOURIER_LOSS_WEIGHT, 0.0, "normal"),
        Formulation.POL_FOURIER: (False, _FOURIER_LOSS_WEIGHT, 0.0, None),
    }
    if formulation not in table:
        raise ValueError(
            f"{formulation!r} is not a vector formulation; `FFT` has no tangent field and is handled by the caller."
        )
    return table[formulation]


def tangent_field_for(
    formulation: Formulation | VectorFn,
    permittivity: jnp.ndarray,
    expansion: basis_mod.Expansion,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The tangent field ``(tx, ty)`` of ``formulation`` on ``permittivity`` (``(nx, ny)``), or of a callable hook
    ``(permittivity, expansion, lattice) -> (tx, ty)``."""
    if callable(formulation):
        return formulation(permittivity, expansion, primitive_lattice_vectors)
    if not isinstance(formulation, Formulation):
        raise TypeError(f"`formulation` must be a Formulation or a callable, got {formulation!r}.")
    jones_direct, fourier_weight, smoothness_weight, post = _scheme(formulation)
    tx, ty = compute_tangent_field(
        permittivity, expansion, primitive_lattice_vectors, jones_direct, fourier_weight, smoothness_weight
    )
    field = jnp.stack([tx, ty], axis=-1)
    if post == "jones":
        field = _jones(field, _magnitude(field) / _scale(_magnitude(field)))
    elif post == "normal":
        field = _unit(field)
    return field[..., 0], field[..., 1]


def compute_tangent_field(
    permittivity: jnp.ndarray,
    expansion: basis_mod.Expansion,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
    use_jones_direct: bool,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
    steps: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The least-squares tangent field (module docstring) on the grid of ``permittivity`` (``(nx, ny)``).

    ``steps`` is accepted for compatibility and ignored: the minimiser is exact, not iterated.
    """
    del steps
    if permittivity.ndim != 2:
        raise ValueError(f"`permittivity` must be a 2-D real-space grid, got shape {permittivity.shape}.")
    grid_shape = permittivity.shape
    lat = _LatticeGeometry(primitive_lattice_vectors)

    # the permittivity gradient, band-limited to the expansion, on a grid twice as fine as the orders need
    g = lat.gradient(permittivity)
    g = jnp.stack([_band_limit(g[..., i], expansion) for i in range(2)], axis=-1)
    g = g / _scale(_magnitude(g))[..., None, None, None]
    weight = _magnitude(g)  # (Gx, Gy, 1)

    # a gradient that varies along one direction only gets its field set by hand below (the solve would be
    # degenerate); a dummy gradient keeps the solve finite there
    is_1d, angle_1d = _is_1d(g)
    g = jnp.where(is_1d, jnp.ones_like(g), g)
    target = _unit(jnp.stack([g[..., 1], -g[..., 0]], axis=-1))
    if use_jones_direct:
        target = _jones(target, _magnitude(target))

    c = _least_squares(target, weight, expansion, lat, fourier_loss_weight, smoothness_loss_weight)
    field = fft.ifft(c, expansion=expansion, shape=grid_shape, axis=-2, centered_coordinates=False)
    field = jnp.where(is_1d, jnp.stack([jnp.sin(angle_1d), jnp.cos(angle_1d)]), field)

    if jnp.promote_types(permittivity.dtype, jnp.complex64) != jnp.promote_types(field.dtype, jnp.complex64):
        raise ValueError(
            f"tangent-field precision guard: permittivity promotes to {jnp.promote_types(permittivity.dtype, jnp.complex64)} "
            f"but the computed tangent field promotes to {jnp.promote_types(field.dtype, jnp.complex64)} -- they must "
            "share a precision class. Align the component-grid dtypes to the stack's dtype= (or pass "
            "dtype=jnp.complex128 under jax_enable_x64)."
        )
    return field[..., 0], field[..., 1]


# -----------------------------------------------------------------------------
# The least-squares problem.
# -----------------------------------------------------------------------------


def _least_squares(target, weight, expansion, lat, fourier_weight, smoothness_weight):
    """The Fourier coefficients ``(n, 2)`` minimising the loss of the module docstring on the grid of ``target``
    (``(Gx, Gy, 2)``), with corner-sampled coordinates."""
    shape = target.shape[:2]
    n_grid = shape[0] * shape[1]
    coeffs = onp.asarray(expansion.basis_coefficients)
    w = weight[..., 0]
    # [[w]]_{G G'} = mean_r w(r) exp(-i (G - G') . r)
    w_hat = jnp.fft.fft2(w, norm="forward")
    diff = coeffs[:, None, :] - coeffs[None, :, :]
    a = w_hat[diff[..., 0], diff[..., 1]]
    b = jnp.fft.fft2(w[..., None] * target, axes=(0, 1), norm="forward")[coeffs[:, 0], coeffs[:, 1]]  # (n, 2)
    # Fourier penalty |k_G|^2 A, smoothness penalty |d_G C|^2 / 4 (the mean over 2 components x 2 directions)
    k = 2 * onp.pi * (coeffs @ onp.stack([lat.b1, lat.b2]))
    diag = (fourier_weight / expansion.num_terms) * onp.sum(k**2, axis=-1) * lat.area
    if smoothness_weight > 0:
        d = onp.stack(
            [
                shape[0] * (onp.exp(2j * onp.pi * coeffs[:, 0] / shape[0]) - 1),
                shape[1] * (onp.exp(2j * onp.pi * coeffs[:, 1] / shape[1]) - 1),
            ],
            axis=-1,
        )
        dc = d @ lat.grad_map
        diag = diag + (smoothness_weight / expansion.num_terms) * onp.sum(onp.abs(dc) ** 2, axis=-1) / 4
    del n_grid
    lhs = a + jnp.diag(jnp.asarray(diag, dtype=a.dtype))
    return jnp.linalg.solve(lhs, b)


class _LatticeGeometry:
    """Concrete lattice data: area, reciprocal basis, and the map from grid-axis derivatives to the lattice's
    Cartesian derivatives (with the basis scaled to unit area, as the penalties are cell-size independent)."""

    def __init__(self, lattice: basis_mod.LatticeVectors):
        basis = onp.stack([onp.asarray(lattice.u, float), onp.asarray(lattice.v, float)], axis=-1)  # columns u, v
        self.area = abs(onp.linalg.det(basis))
        unit = basis / onp.sqrt(self.area)
        self.grad_map = onp.linalg.inv(unit.T @ unit) @ unit.T  # row vector of axis derivatives @ grad_map
        # the reciprocal basis a_i . b_j = delta_ij, in numpy (lattice.reciprocal would trace inside a jit)
        self.b1, self.b2 = onp.linalg.inv(basis).T[:, 0], onp.linalg.inv(basis).T[:, 1]

    def gradient(self, arr: jnp.ndarray) -> jnp.ndarray:
        """Central-difference gradient (periodic), ``(nx, ny, 2)``, in the unit-area lattice coordinates."""
        partial = []
        for ax in (0, 1):
            d = (jnp.roll(arr, -1, axis=ax) - jnp.roll(arr, 1, axis=ax)) / 2
            partial.append(d * arr.shape[ax])
        return jnp.stack(partial, axis=-1) @ jnp.asarray(self.grad_map, dtype=jnp.result_type(arr, jnp.float32))


def _band_limit(x: jnp.ndarray, expansion: basis_mod.Expansion) -> jnp.ndarray:
    """``x`` keeping only the expansion's orders, resampled on twice the minimal grid (corner coordinates)."""
    need = basis_mod.min_array_shape_for_expansion(expansion)
    shape = (need[0] * (2 if x.shape[0] > 1 else 1), need[1] * (2 if x.shape[1] > 1 else 1))
    y = fft.fft(x, expansion=expansion, centered_coordinates=False)
    return fft.ifft(y, expansion=expansion, shape=shape, centered_coordinates=False)


# -----------------------------------------------------------------------------
# Pointwise field operations.
# -----------------------------------------------------------------------------


def _magnitude(field: jnp.ndarray) -> jnp.ndarray:
    """``|field|`` over the last (component) axis, kept as a length-1 axis; safe at 0."""
    m2 = jnp.sum(_abs2(field), axis=-1, keepdims=True)
    zero = m2 == 0
    return jnp.where(zero, 0.0, jnp.sqrt(jnp.where(zero, 1.0, m2)))


def _scale(magnitude: jnp.ndarray) -> jnp.ndarray:
    """The p-norm ``(sum |g|^p)^(1/p)`` of a magnitude field (or its maximum with ``_HARD_MAX``); 1 for a zero
    field. Evaluated relative to the (non-differentiated) maximum to avoid overflow: the value does not depend on
    that reference, so neither does its derivative."""
    m = magnitude[..., 0]
    top = jnp.max(m)
    if _HARD_MAX:
        s = top
    else:
        ref = jax.lax.stop_gradient(jnp.where(top > 0, top, 1.0))
        s = ref * jnp.sum((m / ref) ** _SCALE_P) ** (1.0 / _SCALE_P)
    return jnp.where(jnp.isclose(top, 0), 1.0, s)


def _unit(field: jnp.ndarray) -> jnp.ndarray:
    m = _magnitude(field)
    return field / jnp.where(jnp.isclose(m, 0), 1.0, m)


def _jones(field: jnp.ndarray, magnitude: jnp.ndarray) -> jnp.ndarray:
    """The Jones (elliptical) field of a linear field: the unit direction ``u`` with ellipticity
    ``phi = (pi/8)(1 + cos(pi m))`` for the relative magnitude ``m`` in ``[0, 1]`` -- linear where the field is
    strong (``m = 1``), circular where it vanishes (``m = 0``, ``u = (1, 1)/sqrt 2``)."""
    near0 = jnp.isclose(magnitude, 0.0)
    m_safe = jnp.where(near0, 1.0, _magnitude(field))
    ux = jnp.where(near0, 1 / jnp.sqrt(2), field[..., 0:1] / m_safe)
    uy = jnp.where(near0, 1 / jnp.sqrt(2), field[..., 1:2] / m_safe)
    phi = jnp.pi / 8 * (1 + jnp.cos(jnp.pi * magnitude))
    theta = _angle(ux + 1j * uy)
    jx = jnp.exp(1j * theta) * (ux * jnp.cos(phi) - 1j * uy * jnp.sin(phi))
    jy = jnp.exp(1j * theta) * (uy * jnp.cos(phi) + 1j * ux * jnp.sin(phi))
    return jnp.concatenate([jx, jy], axis=-1)


def _angle(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.angle(jnp.where(jnp.isclose(x, 0.0), 1.0 + 0.0j, x))


def _is_1d(field: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Whether the field points along one axis of directions everywhere it is non-zero, and that direction's angle
    (taken where the field is strongest)."""
    mag = _magnitude(field)[..., 0]
    ref = field.reshape(-1, 2)[jnp.argmax(mag.ravel())]
    ref_angle = _angle(jnp.abs(ref[0]) + 1j * jnp.abs(ref[1]))
    angle = _angle(jnp.abs(field[..., 0]) + 1j * jnp.abs(field[..., 1]))
    close = jnp.isclose(mag, 0.0)
    for k in (-2, -1, 0, 1, 2):
        close = close | jnp.isclose(angle, ref_angle + k * jnp.pi, atol=_ATOL_1D_FIELD_ANGLE)
    return jnp.all(close) | (field.shape[0] == 1) | (field.shape[1] == 1), ref_angle
