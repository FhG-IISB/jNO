"""Small per-point contractions lower as a multiply and a sum, not as a batched GEMM.

Inside an element loop a contraction of 3x3 tensors -- the Vreman invariant ``tr((g gᵀ)²)``, the cell
metric ``J⁻ᵀJ⁻¹`` -- is batched over cells by ``vmap``, and XLA's GPU backend sends a batched
``dot_general`` to a Triton GEMM whose tiles are 16 wide: each 3x3 product padded into one. In a 3-D
residual-based VMS residual (442k DOFs, RTX 3070) the two such GEMMs were 31 of 99 ms; written as a
multiply and a reduction, the residual is 71 ms (and 412 -> 355 ms on the CPU at 55k DOFs).

Oracles: ``small_einsum`` equals ``jnp.einsum`` (values, shapes, gradients) on every form it accepts,
emits no ``dot_general`` for a small contraction and keeps it for a real matrix product; and
``jno.np.einsum`` inside a weak form assembles the same residual through either lowering.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.small_linalg import small_einsum, small_matmul


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _ops(*shapes, seed=0):
    r = np.random.default_rng(seed)
    return [jnp.asarray(r.standard_normal(s)) for s in shapes]


SMALL = [
    ("...ik,...jk,...jl,...il->...", [(4, 3, 3)] * 4),  # Vreman's tr((g gᵀ)²)
    ("ij,jk->ik", [(3, 3), (3, 3)]),
    ("...ij,...jk->...ik", [(2, 1, 3, 3), (5, 3, 3)]),  # broadcast ellipsis
    ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 2)]),
    ("i,j->ij", [(3,), (4,)]),
    ("ij->ji", [(3, 4)]),
    ("ij->", [(3, 3)]),
    ("ij,ij->i", [(3, 3), (3, 3)]),
    ("...ij,...j->...i", [(4, 3, 3), (4, 3)]),
    ("...i,...i->...", [(7, 3), (3,)]),
]


@pytest.mark.parametrize("spec, shapes", SMALL)
def test_a_small_contraction_is_einsum_without_a_dot_general(spec, shapes):
    ops = _ops(*shapes)
    want, got = jnp.einsum(spec, *ops), small_einsum(spec, *ops)
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-13, atol=1e-13 * float(jnp.abs(want).max() + 1))
    assert "dot_general" not in str(jax.make_jaxpr(lambda *o: small_einsum(spec, *o))(*ops))


@pytest.mark.parametrize(
    "spec, shapes",
    [
        ("ij,jk->ik", [(64, 64), (64, 64)]),  # a real matrix product keeps its GEMM
        ("qnij,ia,jb->qnab", [(4, 10, 3, 3), (3, 3), (3, 3)]),  # 120x3x3 per step: over the limit
        ("ii->", [(3, 3)]),  # a repeated index: jnp.einsum's own diagonal
        ("ab,b", [(2, 2), (2,)]),  # implicit output: passed through unchanged
    ],
)
def test_anything_else_is_exactly_jnp_einsum(spec, shapes):
    ops = _ops(*shapes)
    assert np.array_equal(small_einsum(spec, *ops), jnp.einsum(spec, *ops))


def test_gradients_match():
    (g,) = _ops((4, 3, 3), seed=1)
    spec = "...ik,...jk,...jl,...il->..."
    d_small = jax.grad(lambda x: small_einsum(spec, x, x, x, x).sum())(g)
    d_ref = jax.grad(lambda x: jnp.einsum(spec, x, x, x, x).sum())(g)
    assert np.allclose(d_small, d_ref, rtol=1e-13, atol=1e-12)
    (K,) = _ops((5, 3, 3), seed=2)
    assert np.allclose(small_matmul(jnp.swapaxes(K, -1, -2), K), jnp.swapaxes(K, -1, -2) @ K, atol=1e-14)


def test_jno_einsum_in_a_weak_form_assembles_what_the_gemm_lowering_does(monkeypatch):
    """A nonlinear coefficient written with ``jno.np.einsum`` (Vreman's invariant): the residual through
    the multiply-sum lowering equals the one through ``jnp.einsum``'s ``dot_general`` (limit 0)."""
    import jno.utils.solver.small_linalg as sl

    d = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=3).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"))
    x, y, z = d.variable("interior", split=True)[:3]
    X = [x, y, z]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    gu, gv = jno.np.grad(u, X), jno.np.grad(v, X)
    c = jno.np.einsum("...ik,...jk,...jl,...il->...", gu, gu, gu, gu)
    terms = [(1.0 + c) * jno.np.inner(gu, gv, n_contract=2) + jno.np.inner(ui, vi, n_contract=1)]
    plans = []
    real = sl._small_einsum_plan
    monkeypatch.setattr(sl, "_small_einsum_plan", lambda *a: plans.append(real(*a)) or plans[-1])
    fem = jno.fem(terms)
    u0 = jnp.asarray(np.random.default_rng(3).standard_normal(fem.dofs))
    small = np.asarray(fem.residual(u0))
    assert plans and all(p is not None for p in plans), "the form did not take the multiply-sum lowering"
    import jno.utils.solver.fem_utils as fu

    monkeypatch.setattr(sl, "SMALL_CONTRACTION", 0)
    # The element-kernel caches key on the form, not on this lowering switch: empty them for the rebuild.
    monkeypatch.setattr(fu, "_ELEM_MAP_CACHE", type(fu._ELEM_MAP_CACHE)())
    monkeypatch.setattr(fu, "_ELEM_MAP_CONTENT", type(fu._ELEM_MAP_CONTENT)())
    n_small = len(plans)
    gemm = np.asarray(jno.fem(terms).residual(u0))
    # A kernel cache handing back the first build's program would make this comparison vacuous.
    assert len(plans) > n_small and all(p is None for p in plans[n_small:]), "the GEMM build was not traced"
    assert np.abs(small).max() > 1.0
    assert np.allclose(small, gemm, rtol=1e-12, atol=1e-12 * np.abs(gemm).max())
