"""Interior field reconstruction for the state-space RCWA backend.

The readouts that only touch the two ambients (efficiency, per-order fluxes,
order readouts, the incident/reflected states) live in ``stack.py``. Interior
field reconstruction -- propagating the surface state to ``z`` offsets inside a
layer and inverse-Fourier-transforming to ``|E|^2`` volumes and z-profiles --
lives here, on top of the stack's S-matrix sweep.

The propagation is stable by construction: every film is cut at its plane
depths, and the stack's sweep runs over those cuts (``jno/_rcwa/smatrix.py``).
At a cut, the downward reference amplitude ``a+`` is carried from the top and
``a- = R a+`` with ``R`` the reflection of everything below, so growing
evanescent modes never appear as standalone growing quantities -- the same
S-matrix stabilization the plane-wave solve uses, evaluated at the requested
depth.

The algebraic ``ez``/``hz`` follow the fmmax conventions (Liu 2012 eq. 11/14,
state-space form): for a tangential state the Maxwell curl equations give, per
order,

    ez = Z_eps^{-1} * (-(kx*hy - ky*hx)/omega) ,   hz = Z_mu^{-1} * ((kx*ey - ky*ex)/omega) ,

with the inverse-rule zz Fourier matrix ``Z_eps`` (patterned layers) or the
scalar ``eps_zz`` (uniform layers and the analytic ambients). The iFFT to the
real-space grid carries the Bloch phase ``exp(i (kin·r + G·r))`` -- E(r) is
reconstructed as ``sum_G c_G exp(i (k_in + G)·r)``, the exact inverse of the
fmmax ``amplitudes_for_fields`` convention (``fmmax.sources`` divides the
real-space field by ``exp(i (kx·x + ky·y))`` to get the harmonic
coefficients).

Oblique-incidence ``|E|^2`` accuracy caveat: at NON-normal incidence the
reconstructed interior ``|E|^2`` (via :func:`volume_intensity` and the seam's
``field``/``field3d`` readouts) carries a ~5%-class deviation from the accurate
energy content, attributed to the conditioning of the state-space
reconstruction at tilted ``k_in`` (the tangential-state split into
forward/backward modal content degrades). Efficiency, per-order flux and
absorbed-power readouts are NOT affected -- they are the accurate quantities
for energy work at oblique incidence. Do not feed oblique ``field()``
``|E|^2`` into a heat-source/absorbed-volume objective without accounting for
this band; see the pinned oblique test in ``tests/test_rcwa_fields.py``.

Everything the design differentiates is traced (the per-order amplitudes and
the iFFT are native JAX), so ``|E|^2`` volumes feed coupled objectives
(absorbed power -> heat) with native gradients.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax.numpy as jnp

from . import basis as basis_mod
from . import fft


def layer_z_sampling(thicknesses: Sequence[float], density: float) -> Tuple[list, list]:
    """Per-layer z samples of the fmmax convention (inclusive linspace per layer).

    Each layer is sampled ``max(4, round(t * density))`` times on
    ``linspace(0, t, n)`` -- including both of its ends -- and the layers are
    joined cumulatively, so interface depths appear twice (once from each
    neighbouring layer; this inclusive-endpoint convention is the seam's own
    energy-balance anchor, sampled from both sides of every interface).

    Args:
        thicknesses: The full per-layer thicknesses (ambients included).
        density: Number of z samples per unit thickness.

    Returns:
        ``(znums, z)``: per-layer sample counts, and the flat global depths
        (a plain Python float list; host-side static).
    """
    znums = [max(4, int(round(float(t) * density))) for t in thicknesses]
    z, lo = [], 0.0
    for t, zn in zip(thicknesses, znums):
        for k in range(zn):
            z.append(lo + float(t) * k / (zn - 1))
        lo += float(t)
    return znums, z


def field_planes(
    thicknesses: Sequence[float],
    znums: Sequence[int],
    layer_roles: Sequence[object],
) -> Tuple[object, ...]:
    """Resolve per-plane probes to the backend's static plane specs.

    Each layer's own sample planes keep their layer's identity (unlike a raw
    global-depth lookup, this keeps the two samples sharing a z interface
    distinct -- the superstrate's end plane uses the analytic ambient field,
    the film's start plane its own).

    Args:
        thicknesses: Full per-layer thicknesses.
        znums: Per-layer sample counts (:func:`layer_z_sampling`).
        layer_roles: Per-layer role: ``"sup"``/``"sub"`` for the analytic
            ambients, else the backend film index.

    Returns:
        The static planes tuple: ``("sup", offset)`` / ``("sub", z)`` /
        ``("film", film_idx, zeta)`` -- the backend's spec format.
    """
    planes = []
    for t, zn, role in zip(thicknesses, znums, layer_roles):
        for k in range(zn):
            zl = float(t) * k / (zn - 1)
            if role == "sup":
                planes.append(("sup", zl - float(t)))
            elif role == "sub":
                planes.append(("sub", zl))
            else:
                planes.append(("film", role, zl))
    return tuple(planes)


def interior_field_amplitudes(
    stack,
    planes: Sequence[object],
    grids,
    wl,
    kin,
    pol,
    amb_sup,
    amb_sub,
    thick=None,
) -> Tuple[jnp.ndarray, ...]:
    """Per-plane Fourier field amplitudes through the stack's compiled core.

    Args:
        stack: A built and solved :class:`jno._rcwa.stack.StateStack`.
        planes: The static plane specs (:func:`field_planes`).
        grids: The FINITE-film per-solve grids (design; may be traced) -- the
            analytic ambients enter only through ``amb_sup``/``amb_sub``.
        wl: Free-space wavelength (may be traced).
        kin: Transverse incident wavevector (may be traced).
        pol: The incident Jones pair ``(px, py)``.
        amb_sup, amb_sub: The analytic ambient permittivities.
        thick: The per-film thicknesses of the solve (default: the build ones);
            they must match the thicknesses the planes were resolved with.

    Returns:
        ``(ex, ey, ez, hx, hy, hz)``: per-plane complex Fourier coefficients,
        each ``(len(planes), n)``, in plane order.
    """
    return stack.interior_field_amplitudes(grids, wl, kin, pol, amb_sup, amb_sub, tuple(planes), thick=thick)


def volume_intensity(
    ex: jnp.ndarray,
    ey: jnp.ndarray,
    ez: jnp.ndarray,
    expansion,
    kin,
    period,
    nx: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Real-space ``|E|^2`` volume from per-plane Fourier amplitudes.

    Each per-plane component is inverse-Fourier-transformed onto the ``nx``
    grid (centered coordinates, fmmax's convention) and the Bloch phase of the
    incident ``kin`` applied, then ``|E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2`` is
    stacked over planes.

    Caveat (see the module docstring): at OBLIQUE incidence the reconstructed
    ``|E|^2`` deviates by ~5% from the accurate energy content (conditioning of
    the state-space split at tilted ``k_in``). Efficiency/flux/absorbed-power
    readouts are unaffected; do not use oblique ``|E|^2`` as a heat-source
    volume without accounting for this band.

    Args:
        ex, ey, ez: Per-plane Fourier coefficients, each ``(P, n)``.
        expansion: The solve's :class:`~jno._rcwa.basis.Expansion`.
        kin: Transverse incident wavevector (Bloch phase).
        period: Lattice periods ``(Px, Py)``.
        nx: Real-space grid points per axis.

    Returns:
        ``(inten, x, y)``: ``|E|^2`` of shape ``(nx, nx, P)`` (z along the
        last axis, matching the fmmax-era seam), and the 1-D coordinate axes
        ``x``, ``y``.
    """
    x = jnp.linspace(0.0, float(period[0]), nx)
    y = jnp.linspace(0.0, float(period[1]), nx)
    phase = jnp.exp(1j * (kin[0] * x[:, jnp.newaxis] + kin[1] * y[jnp.newaxis, :]))

    def _ifft(c):
        return fft.ifft(c, expansion, (nx, nx), axis=-1) * phase

    inten = jnp.abs(_ifft(ex)) ** 2 + jnp.abs(_ifft(ey)) ** 2 + jnp.abs(_ifft(ez)) ** 2
    return jnp.transpose(inten, (1, 2, 0)), x, y


def validate_grid_resolution(nx: int, expansion, where: str) -> None:
    """Loud, actionable check that a render grid resolves the Fourier expansion.

    The expansion-aware iFFT raises too, but with a generic message; the seam
    wants a jNO-style hint naming the fix.
    """
    min_shape = basis_mod.min_array_shape_for_expansion(expansion)
    if nx < max(min_shape):
        raise ValueError(
            f"{where}: the real-space grid {nx}x{nx} is too coarse for the "
            f"{expansion.num_terms}-term expansion (needs >= {max(min_shape)} points/axis). "
            "Raise nx (field/field3d), or lower orders=."
        )
