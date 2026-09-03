"""The two operators the coupled solve is assembled from, each against a dense reference.

Phase 2 built the coupling GENERATOR and the potential; this is the layer above them -- the one that
lifts a bar family into the cell grid, applies the FFT, and reads the other family back out. That
lifting is where an off-by-one hides: a family along `ax` is one cell short along `ax` and the two
families are short along DIFFERENT axes, so a wrapper that reused either family's own shape would
mis-register the two meshes by one cell and still return smooth, plausible numbers.

So both operators are checked against an O(N^2) assembly that shares none of the machinery: element
positions come from `element_centres`, the kernel is summed pair by pair, and no FFT, generator or
lattice index is involved.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import bar_self, magnetic_reluctance
from jno.utils.solver.peec import (
    bar_filaments,
    element_centres,
    magnetic_coupling_apply,
    magnetic_system_apply,
)

jax.config.update("jax_enable_x64", True)

P, MU0 = 0.002, 4e-7 * np.pi
CU, MU_R = 5.8e7, 2000.0


def _meshes():
    """A conductor plate and a core plate stacked in z, on ONE grid -- both carry x and y families."""
    plate = jno.Shape.box(0, 0, 0, 3 * P, 3 * P, P, size=(P,) * 3).attach(sigma=CU).name("plate")
    core = jno.Shape.box(0, 0, 2 * P, 3 * P, 3 * P, 3 * P, size=(P,) * 3).attach(mu_r=MU_R).name("core")
    fil = bar_filaments(plate, sigma=CU, grid_shapes=[core])
    mag = bar_filaments(core, sigma=MU_R - 1.0, grid_shapes=[plate])
    assert fil.lattice["n"] == mag.lattice["n"] and fil.lattice["d"] == mag.lattice["d"]
    return fil, mag


def _cube_rule(d, quad):
    """Centred sub-points over a cell, weights summing to ONE -- an average, not a moment."""
    g1, w1 = np.polynomial.legendre.leggauss(quad)
    i, j, k = (a.reshape(-1) for a in np.meshgrid(*(np.arange(quad),) * 3, indexing="ij"))
    pts = np.stack([0.5 * d[0] * g1[i], 0.5 * d[1] * g1[j], 0.5 * d[2] * g1[k]], axis=1)
    return pts, (w1[i] * w1[j] * w1[k]) / 8.0


def _dense_coupling(fil, mag, quad=2):
    """K[m, e] straight from Ampere's law over the two element volumes -- O(N^2), no lattice."""
    d = tuple(fil.lattice["d"])
    ce, cm = np.asarray(element_centres(fil)), np.asarray(element_centres(mag))
    ax_e, ax_m = np.asarray(fil.lattice["axis"]), np.asarray(mag.lattice["axis"])
    pts, w = _cube_rule(d, quad)
    K = np.zeros((len(cm), len(ce)))
    for v in range(len(cm)):
        eb = np.eye(3)[ax_m[v]]
        for u in range(len(ce)):
            ea = np.eye(3)[ax_e[u]]
            r = (cm[v] + pts)[:, None, :] - (ce[u] + pts)[None, :, :]
            rn = np.linalg.norm(r, axis=-1)
            body = np.einsum("pqi,i->pq", np.cross(ea, r / rn[..., None]), eb) / rn**2
            K[v, u] = (w[:, None] * w[None, :] * body).sum() * d[ax_e[u]] * d[ax_m[v]] / (4.0 * np.pi)
    return K


def _dense_potential(mag, quad=2):
    """P_m over the CELLS, the same volume-averaged 1/r the operator uses, assembled densely."""
    d = tuple(mag.lattice["d"])
    nodes = np.asarray(mag.nodes)
    pts, w = _cube_rule(d, quad)
    n = len(nodes)
    P = np.zeros((n, n))
    self_g = float(bar_self(np.array([d[2]]), np.array([d[0]]), np.array([d[1]]))[0])
    for a in range(n):
        for b in range(n):
            if a == b:
                P[a, b] = self_g
                continue
            r = (nodes[a] + pts)[:, None, :] - (nodes[b] + pts)[None, :, :]
            P[a, b] = (w[:, None] * w[None, :] / np.linalg.norm(r, axis=-1)).sum()
    return P / (4.0 * np.pi * MU0)


# --------------------------------------------------------------------------- the coupling


def test_the_coupling_applies_the_ampere_block():
    """mmf(I_c) is K I_c, with K assembled pair by pair from Ampere's law."""
    fil, mag = _meshes()
    K = _dense_coupling(fil, mag)
    mmf, _flux = magnetic_coupling_apply(fil, mag)
    rng = np.random.default_rng(0)
    cur = rng.standard_normal(K.shape[1])
    got, want = np.asarray(mmf(jnp.asarray(cur))), K @ cur
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12 * np.abs(want).max())


def test_the_reverse_coupling_is_the_same_block_transposed():
    """Reciprocity is a PROPERTY of the assembled operator here, not an assumption made about it.

    If it failed, the coupled system would be non-symmetric and the energy it computes would depend
    on which way round you asked for it.
    """
    fil, mag = _meshes()
    K = _dense_coupling(fil, mag)
    _mmf, flux = magnetic_coupling_apply(fil, mag)
    rng = np.random.default_rng(1)
    im = rng.standard_normal(K.shape[0])
    got, want = np.asarray(flux(jnp.asarray(im))), K.T @ im
    assert np.allclose(got, want, rtol=1e-10, atol=1e-12 * np.abs(want).max())


def test_the_coupling_carries_a_complex_current():
    """The currents ARE complex at every frequency that matters, and the FFT embedding is real --
    so the split into parts is not an optional nicety, it is the whole path."""
    fil, mag = _meshes()
    mmf, _f = magnetic_coupling_apply(fil, mag)
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(int(np.asarray(fil.length).size)), rng.standard_normal(int(np.asarray(fil.length).size))
    got = np.asarray(mmf(jnp.asarray(a + 1j * b)))
    assert np.allclose(got, np.asarray(mmf(jnp.asarray(a))) + 1j * np.asarray(mmf(jnp.asarray(b))))


def test_two_meshes_on_different_grids_are_refused():
    """Silently applying a convolution across mismatched grids is the defect this guards.

    It is not hypothetical: built independently the two meshes came out different SIZES and offset
    from each other, and every number downstream would have been wrong with nothing to show for it.
    """
    plate = jno.Shape.box(0, 0, 0, 3 * P, 3 * P, P, size=(P,) * 3).attach(sigma=CU).name("plate")
    core = jno.Shape.box(0, 0, 2 * P, 3 * P, 3 * P, 3 * P, size=(P,) * 3).attach(mu_r=MU_R).name("core")
    with pytest.raises(ValueError, match="COMMON grid"):
        magnetic_coupling_apply(bar_filaments(plate, sigma=CU), bar_filaments(core, sigma=MU_R - 1.0))


# --------------------------------------------------------------------------- the magnetic system


def test_the_magnetic_operator_is_reluctance_plus_the_potential_of_its_divergence():
    """(R_m + A' P_m A) against the same thing assembled densely from `mag.incidence`."""
    _fil, mag = _meshes()
    chi = np.full(int(np.asarray(mag.length).size), MU_R - 1.0)
    A = mag.incidence.toarray()
    G = np.diag(np.asarray(magnetic_reluctance(mag.length, mag.area, MU_R, MU0))) + A.T @ _dense_potential(mag) @ A
    apply, _diag = magnetic_system_apply(mag, chi, MU0)
    rng = np.random.default_rng(3)
    im = rng.standard_normal(G.shape[0])
    got, want = np.asarray(apply(jnp.asarray(im))), G @ im
    assert np.allclose(got, want, rtol=1e-9, atol=1e-11 * np.abs(want).max())


def test_the_magnetisation_operator_is_positive_definite():
    """The SIGN test, and the one a wrong answer would hide behind.

    `rho_m = -div M` and the incidence reports `+I` at the cell an element leaves, so the two minus
    signs cancel and the demagnetising term ADDS. Had it come out `R_m - A' P_m A` the operator
    would still be symmetric, still solvable, and would describe a core that magnetises itself.
    """
    _fil, mag = _meshes()
    A = mag.incidence.toarray()
    lap = A.T @ _dense_potential(mag) @ A
    assert np.min(np.linalg.eigvalsh(0.5 * (lap + lap.T))) > -1e-9 * np.abs(lap).max()
    chi = np.full(A.shape[1], MU_R - 1.0)
    apply, _d = magnetic_system_apply(mag, chi, MU0)
    G = np.stack([np.asarray(apply(jnp.asarray(np.eye(A.shape[1])[k]))) for k in range(A.shape[1])], axis=1)
    assert np.min(np.linalg.eigvalsh(0.5 * (G + G.T))) > 0.0


def test_the_diagonal_is_the_operators_own():
    """The preconditioner's diagonal is EXACT, not sampled: both ends are cells of one grid, so it
    is two constants of the pitch. A diagonal that drifted from the operator would only ever show up
    as a solve that converges slowly, which is the kind of defect that never gets found."""
    _fil, mag = _meshes()
    n = int(np.asarray(mag.length).size)
    chi = np.full(n, MU_R - 1.0)
    apply, diag = magnetic_system_apply(mag, chi, MU0)
    got = np.array([np.asarray(apply(jnp.asarray(np.eye(n)[k])))[k] for k in range(n)])
    assert np.allclose(np.asarray(diag), got, rtol=1e-10)


def test_a_weaker_core_is_a_larger_reluctance():
    """chi enters as 1/chi, so halving the susceptibility doubles the reluctance -- and the
    potential term, which is geometry, does not move at all."""
    _fil, mag = _meshes()
    n = int(np.asarray(mag.length).size)
    _a1, d1 = magnetic_system_apply(mag, np.full(n, MU_R - 1.0), MU0)
    _a2, d2 = magnetic_system_apply(mag, np.full(n, 0.5 * (MU_R - 1.0)), MU0)
    rm1 = np.asarray(magnetic_reluctance(mag.length, mag.area, MU_R, MU0))
    lap = np.asarray(d1) - rm1
    assert np.allclose(np.asarray(d2) - 2.0 * rm1, lap, rtol=1e-10)
    assert np.all(lap > 0)
