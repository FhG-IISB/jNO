"""`jno.rom`: POD, Galerkin projection and gated residual enrichment.

The oracle is a dense solve. The family is a 1-D finite-element Laplacian with a coefficient that depends on two
parameters, loaded by two smooth right-hand sides; its snapshots come from `np.linalg.solve`, and every reduced
answer is compared with the exact energy matrix `C = B^T A^-1 B` of the same system.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsp

import jno.rom

N = 120
X = np.linspace(0, 1, N + 2)[1:-1]
B = np.stack([np.exp(-(((X - 0.3) / 0.1) ** 2)), np.exp(-(((X - 0.7) / 0.1) ** 2))], 1)


@pytest.fixture(autouse=True)
def _x64():
    """The reduced solves run in float64; save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _system(p):
    """SPD tridiagonal stiffness of -(k u')' = f with u = 0 at both ends, k per cell."""
    k = 1.0 + p[0] * np.sin(np.pi * np.linspace(0, 1, N + 1) * (1 + p[1])) ** 2
    return np.diag(k[:-1] + k[1:]) - np.diag(k[1:-1], 1) - np.diag(k[1:-1], -1)


def _snapshots():
    rng = np.random.default_rng(0)
    return [np.linalg.solve(_system(rng.uniform(0, 3, 2)), B) for _ in range(30)]


def _basis(k):
    return jno.rom.pod(np.concatenate([jno.rom.unit_energy(U, B) for U in _snapshots()], 1), k=k)[0]


def test_unit_energy_scales_every_snapshot_by_its_energy():
    U = np.concatenate(_snapshots(), 1)
    Bt = np.tile(B, (1, 30))
    S = jno.rom.unit_energy(U, Bt)
    np.testing.assert_allclose(np.einsum("nm,nm->m", Bt, S) ** 2, np.einsum("nm,nm->m", Bt, U), rtol=1e-12)


def test_pod_of_blocks_spans_the_pod_of_the_concatenation():
    S = np.concatenate([jno.rom.unit_energy(U, B) for U in _snapshots()], 1)
    W, sv = jno.rom.pod(S, k=10)
    np.testing.assert_allclose(W.T @ W, np.eye(10), atol=1e-12)
    assert np.all(np.diff(sv) <= 0)
    Wb, _ = jno.rom.pod([S[:, :30], S[:, 30:]], k=10)
    np.testing.assert_allclose(np.abs(Wb.T @ W), np.eye(10), atol=1e-6)


def test_galerkin_bounds_the_energy_from_below_and_enrichment_closes_the_gap():
    """A parameter outside the snapshot range: the plain POD answer is visibly off, the enriched one is not."""
    A = _system(np.array([4.5, 2.5]))
    C = B.T @ np.linalg.solve(A, B)
    As = jsp.BCOO.fromdense(jnp.asarray(A))
    W = _basis(4)
    plain = jno.rom.galerkin(W)(As, B)
    enriched = jno.rom.galerkin(W, enrich=6)(As, B)
    err = lambda r: np.max(np.abs(np.diag(r.C) / np.diag(C) - 1))

    assert np.all(np.diag(plain.C) <= np.diag(C) * (1 + 1e-12))
    assert np.all(np.diag(enriched.C) <= np.diag(C) * (1 + 1e-12))
    assert err(enriched) < 0.2 * err(plain)
    assert float(plain.eta) > 0 and not bool(plain.enriched) and bool(enriched.enriched)


def test_gate_returns_the_plain_answer_below_the_indicator_and_enriches_above():
    A = jsp.BCOO.fromdense(jnp.asarray(_system(np.array([4.5, 2.5]))))
    W = _basis(4)
    plain, enriched = jno.rom.galerkin(W)(A, B), jno.rom.galerkin(W, enrich=6)(A, B)
    high = jno.rom.galerkin(W, enrich=6, gate=10 * float(plain.eta))(A, B)
    low = jno.rom.galerkin(W, enrich=6, gate=0.1 * float(plain.eta))(A, B)
    np.testing.assert_allclose(high.C, plain.C, rtol=1e-12)
    np.testing.assert_allclose(low.C, enriched.C, rtol=1e-7)  # float32 directions: compiled vs eager round-off


def test_lift_reproduces_the_energy_matrix():
    A = jsp.BCOO.fromdense(jnp.asarray(_system(np.array([1.0, 0.5]))))
    rom = jno.rom.galerkin(_basis(6), enrich=3)
    r = rom(A, B)
    np.testing.assert_allclose(B.T @ np.asarray(rom.lift(r)), r.C, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(rom.lift(r, jnp.array([1.0, 2.0]))), np.asarray(rom.lift(r)) @ [1.0, 2.0])


def test_a_complete_basis_is_exact_under_jit_and_vmap():
    A = _system(np.array([1.0, 0.5]))
    As = jsp.BCOO.fromdense(jnp.asarray(A))
    rom = jno.rom.galerkin(np.linalg.qr(np.random.default_rng(1).standard_normal((N, N)))[0])
    C = jax.jit(lambda r, A, B: r(A, B).C)(rom, As, B)  # the Galerkin object is a pytree argument
    np.testing.assert_allclose(C, B.T @ np.linalg.solve(A, B), rtol=1e-9)
    Cs = jax.vmap(lambda b: rom(As, b).C)(jnp.stack([jnp.asarray(B), 2 * jnp.asarray(B)]))
    np.testing.assert_allclose(Cs[1], 4 * Cs[0], rtol=1e-12)
