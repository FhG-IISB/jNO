"""`P_m`, the magnetic potential matrix -- the scalar dual of the partial inductance.

Eliminating the voxel potentials reduces the magnetic system to `(R_m + A' P_m A) I = K I_c`, so
`P_m` maps a magnetic charge on each cell to the potential it produces. Its coefficient is exactly
dual to `Lp`'s:

    Lp  = (mu0 / 4 pi) <1/r> (mom . mom)     vector, over BARS
    P_m = (1 / 4 pi mu0) <1/r>               scalar, over CELLS

The oracle is an explicit cell-by-cell double sum -- O(N^2), wrong in a different way from an FFT,
which is the point of it. Same discipline as `test_peec_fft` applies to the electric operator.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import bar_self
from jno.utils.solver.peec import bar_filaments, magnetic_potential_apply

jax.config.update("jax_enable_x64", True)

MU0 = 4e-7 * np.pi
SIG = 5.8e7


def _lattice(nx=4, ny=3, nz=2, p=0.002):
    return bar_filaments(
        jno.Shape.box(0, 0, 0, nx * p, ny * p, nz * p, size=(p, p, p)), size=(p, p, p), sigma=SIG
    )


def _dense_P(fil, quad=2):
    """The O(N^2) cell-to-cell sum the FFT replaces, written out."""
    lat = fil.lattice
    n, d = tuple(int(v) for v in lat["n"]), tuple(float(v) for v in lat["d"])
    live = np.flatnonzero(np.asarray(lat["cells"]).reshape(-1))
    idx = np.stack(np.unravel_index(live, n), axis=1) * np.array(d)
    g1, w1 = np.polynomial.legendre.leggauss(quad)
    i, j, k = (a.reshape(-1) for a in np.meshgrid(*(np.arange(quad),) * 3, indexing="ij"))
    sub = np.stack([0.5 * d[0] * g1[i], 0.5 * d[1] * g1[j], 0.5 * d[2] * g1[k]], axis=1)
    w = (w1[i] * w1[j] * w1[k]) / 8.0
    self_g = float(bar_self(np.array([d[2]]), np.array([d[0]]), np.array([d[1]]))[0])
    m = len(live)
    P = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a == b:
                P[a, b] = self_g
                continue
            dd = (idx[a] + sub[:, None, :]) - (idx[b] + sub[None, :, :])
            P[a, b] = float((w[:, None] * w[None, :] / np.sqrt((dd * dd).sum(-1))).sum())
    return P / (4.0 * np.pi * MU0)


def test_the_fft_operator_reproduces_the_dense_cell_sum():
    """To round-off, which is what makes the FFT the exact fast form and not an approximation."""
    fil = _lattice()
    P = _dense_P(fil)
    ap = magnetic_potential_apply(fil)
    rng = np.random.default_rng(0)
    q = rng.standard_normal(P.shape[0])
    assert np.allclose(np.asarray(ap(jnp.asarray(q))), P @ q, rtol=1e-10, atol=1e-14)


def test_it_is_the_dual_of_the_partial_inductance_coefficient():
    """P_m carries 1 / (4 pi mu0) where Lp carries mu0 / (4 pi) -- reciprocal, not the same.

    A sign or a factor of mu0^2 here would still produce a plausible, smoothly varying operator, so
    the coefficient is checked against the closed form rather than eyeballed.
    """
    p = 0.002
    fil = _lattice(2, 1, 1, p)
    ap = magnetic_potential_apply(fil)
    q = jnp.array([1.0, 0.0])
    got = float(np.asarray(ap(q))[1])  # the potential the far cell sees from a unit charge
    # two cells one pitch apart, volume-averaged 1/r, times 1/(4 pi mu0)
    g1, w1 = np.polynomial.legendre.leggauss(2)
    i, j, k = (a.reshape(-1) for a in np.meshgrid(*(np.arange(2),) * 3, indexing="ij"))
    sub = np.stack([0.5 * p * g1[i], 0.5 * p * g1[j], 0.5 * p * g1[k]], axis=1)
    w = (w1[i] * w1[j] * w1[k]) / 8.0
    dd = (sub[:, None, :] + np.array([p, 0, 0])) - sub[None, :, :]
    want = float((w[:, None] * w[None, :] / np.sqrt((dd * dd).sum(-1))).sum()) / (4 * np.pi * MU0)
    assert abs(got / want - 1) < 1e-12


def test_a_hole_in_the_core_is_not_charged():
    """Cells that carry no material must not appear: the operator is over occupancy, not the box."""
    p = 0.002
    solid = bar_filaments(jno.Shape.box(0, 0, 0, 6 * p, 4 * p, p, size=(p, p, p)), sigma=SIG)
    holed = bar_filaments(
        jno.Shape.box(0, 0, 0, 6 * p, 4 * p, p, size=(p, p, p)) - jno.Shape.box(2 * p, p, -p, 4 * p, 3 * p, 2 * p),
        size=(p, p, p),
        sigma=SIG,
    )
    n_solid = int(np.asarray(solid.lattice["cells"]).sum())
    n_holed = int(np.asarray(holed.lattice["cells"]).sum())
    assert n_holed < n_solid
    q = jnp.ones(n_holed)
    assert np.asarray(magnetic_potential_apply(holed)(q)).shape == (n_holed,)


def test_a_polyline_has_no_cells_and_says_so():
    """Flux divides between cells; a filament has none, so the refusal names the reason."""
    from jno.utils.solver.peec import line_filaments

    fil = line_filaments(jno.Shape.line([(0, 0, 0), (0, 0, 0.01)], r=2e-4, size=0.002))
    with pytest.raises(ValueError, match="no cells for flux to divide between"):
        magnetic_potential_apply(fil)
