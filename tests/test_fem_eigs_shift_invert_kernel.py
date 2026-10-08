"""Shift-invert (``eigs(sigma=...)``) next to a wide degenerate cluster: the curl-curl kernel.

A PEC cavity discretized with Nédélec (N1E) elements has, besides the cavity resonances
``π²(m² + n²)`` (unit square) / ``π²(l² + m² + n²)`` (unit cube), an exact kernel of gradients at
``λ = 0`` whose dimension grows with the mesh (~ the number of interior vertices: 102 on the square
below). With σ between 0 and the first cavity mode, the kernel's transformed eigenvalue ``θ = −1/σ``
is as large as the wanted one, and the shift-invert eigensolver -- then plain block subspace iteration
-- stalled and returned NaN (measured: 200 sweeps, residual 1.7e-2, at σ = 5). With σ = 1, where the
wanted pairs ARE kernel modes, the residual gate divided by ``max|λ| ≈ 1e-13`` and could never pass.
Near σ = π², the exact kernel copies crowded the 2π² mode out of the block.

The oracle is analytic: the ``k`` values nearest σ in the cavity spectrum, the kernel counted as a zero
of unbounded multiplicity. The discretization error at these meshes is measured, not assumed (N1E_1,
h = 0.1: 1.6e-4 on π²).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")
pytest.importorskip("basix")

import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

inner = jno.np.inner
PI2 = float(np.pi**2)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _cavity(tdim, order, h):
    """PEC unit square / cube: ``K`` = curl-curl with the tangential trace pinned, and the mass form."""
    if tdim == 2:
        d = jno.domain(box(0, 0, 1, 1), mesh_size=h)
    else:
        d = jno.shape.box(0, 0, 0, 1, 1, 1, size=h).domain()
    u, v = d.fem_symbols(value_shape=(tdim,), names=("u", "v"), space="N1E", order=order)
    c = d.variable("interior", split=True)[:tdim]
    ui, vi = u.bind(**dict(zip("xyz", c))), v.bind(**dict(zip("xyz", c)))
    b = d.variable("boundary", normals=True, split=True)
    xb, nb = b[:tdim], b[-tdim:]
    ub = u.bind(**dict(zip("xyz", xb)))
    if tdim == 2:
        nx, ny = nb
        K = jno.fem([ui.curl() * vi.curl(), ub[0] * ny - ub[1] * nx - 0.0])
    else:
        nx, ny, nz = nb
        cu, cv = ui.vector.curl(*c), vi.vector.curl(*c)
        pec = [ub[1] * nz - ub[2] * ny - 0.0, ub[2] * nx - ub[0] * nz - 0.0, ub[0] * ny - ub[1] * nx - 0.0]
        K = jno.fem([inner(cu, cv), *pec])
    return K, [inner(ui, vi)]


def _analytic_nearest(tdim, sigma, k):
    """The k cavity eigenvalues nearest sigma, the gradient kernel (λ = 0) of unbounded multiplicity."""
    modes = []
    for idx in itertools.product(range(5), repeat=tdim):
        if sum(i == 0 for i in idx) > 1:
            continue  # at most one index zero (square: (m, 0) and (0, n) are the TE_m0 / TE_0n modes)
        mult = 1 if tdim == 2 else 2 - (0 in idx)  # cube: two polarizations unless an index is zero
        modes += [PI2 * sum(i * i for i in idx)] * mult
    modes += [0.0] * (4 * k)
    modes = np.asarray(modes)
    return np.sort(modes[np.argsort(np.abs(modes - sigma), kind="stable")[:k]])


def _check(lam, exact, rtol):
    lam = np.sort(np.asarray(lam).real)
    assert np.all(np.isfinite(lam)), f"shift-invert returned NaN: {lam}"
    zero = exact == 0.0
    assert np.all(np.abs(lam[zero]) < 1e-8), (lam, exact)  # kernel modes: zero to roundoff
    np.testing.assert_allclose(lam[~zero], exact[~zero], rtol=rtol)


@pytest.mark.parametrize(
    "sigma, k",
    [
        (5.0, 3),  # kernel θ = -0.200 vs π² θ = 0.205: plain subspace iteration stalled -> NaN
        (5.0, 6),  # same, more of the kernel wanted
        (1.0, 2),  # the wanted pairs ARE kernel modes: the residual scale collapsed to roundoff
        (9.87, 6),  # just above π²: 2π² and the kernel almost equidistant; the kernel used to evict 2π²
        (30.0, 5),  # away from the kernel: unchanged behaviour
    ],
)
def test_square_cavity_shift_invert_next_to_the_kernel(sigma, k):
    K, mass = _cavity(2, 1, 0.1)
    lam, X = K.eigs(mass=mass, k=k, sigma=sigma)
    _check(lam, _analytic_nearest(2, sigma, k), rtol=2e-3)  # measured 1.6e-4 on π², 4e-4 on 4π²
    assert np.all(np.isfinite(np.asarray(X)))


def test_square_cavity_shift_invert_at_degree_two():
    """N1E_2: the kernel holds the higher-order gradients too (113 zeros at h = 0.2)."""
    K, mass = _cavity(2, 2, 0.2)
    lam, _X = K.eigs(mass=mass, k=3, sigma=5.0)
    _check(lam, _analytic_nearest(2, 5.0, 3), rtol=1e-3)


def test_cube_cavity_shift_invert_below_the_first_mode():
    """Unit cube, N1E_1: σ = 10 sits between the kernel and the triple 2π² mode; the four nearest are
    the triple and one kernel mode. h = 0.35 is coarse: the triple comes out 8% low (measured)."""
    K, mass = _cavity(3, 1, 0.35)
    lam, _X = K.eigs(mass=mass, k=4, sigma=10.0)
    _check(lam, _analytic_nearest(3, 10.0, 4), rtol=0.1)
