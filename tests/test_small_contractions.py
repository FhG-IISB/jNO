"""Small per-point contractions lower as a multiply and a sum, not as a batched GEMM.

Inside an element loop a contraction of 3x3 tensors -- the Vreman invariant ``tr((g gᵀ)²)``, the cell
metric ``J⁻ᵀJ⁻¹`` -- is batched over cells by ``vmap``, and XLA's GPU backend sends a batched
``dot_general`` to a Triton GEMM whose tiles are 16 wide: each 3x3 product padded into one. In a 3-D
residual-based VMS residual (442k DOFs, RTX 3070) the two such GEMMs were 31 of 99 ms; written as a
multiply and a reduction, the residual is 71 ms (and 412 -> 355 ms on the CPU at 55k DOFs).

Oracles: ``small_einsum`` equals ``jnp.einsum`` (values, shapes, gradients) on every form it accepts,
emits no ``dot_general`` for a small contraction and keeps it for a real matrix product; and
``jno.np.einsum`` inside a weak form assembles the same residual through either lowering.

``small_matmul`` is the same for a product (``jnp.matmul`` semantics: 1-D operands, broadcast batch axes),
and ``jno.np.matmul`` and the ``@`` of the trace go through it: in a 3-D St.Venant-Kirchhoff residual the
two products ``FᵀF`` and ``F·S`` as Triton GEMMs were 79% of the GPU time.
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


# ---------------------------------------------------------------------------------------------------
# small_matmul and the weak-form products
# ---------------------------------------------------------------------------------------------------


def _dot_generals(fn, *args) -> int:
    return str(jax.make_jaxpr(fn)(*args)).count("dot_general")


@pytest.mark.parametrize(
    "sa, sb",
    [
        ((3, 3), (3, 3)),
        ((3,), (3, 3)),  # vector @ matrix: the 1-D operand gets a temporary axis, removed after
        ((3, 3), (3,)),
        ((3,), (3,)),  # inner product -> scalar
        ((2, 3), (3, 4)),  # non-square
        ((5, 3, 3), (3, 3)),  # batch on one side only
        ((5, 1, 2, 3), (4, 3, 2)),  # batch axes broadcast against each other
        ((0, 3, 3), (3, 3)),  # empty batch
        ((1, 1), (1, 1)),
    ],
)
def test_small_matmul_matches_numpy_matmul(sa, sb):
    # Break: any shape rule that differs from numpy.matmul (1-D promotion, batch broadcasting).
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(sa), rng.standard_normal(sb)
    got = np.asarray(small_matmul(jnp.asarray(a), jnp.asarray(b)))
    want = np.matmul(a, b)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13)  # fp64, <= 8 terms per sum


def test_small_matmul_complex_matches_numpy():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 3, 3)) + 1j * rng.standard_normal((4, 3, 3))
    b = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
    np.testing.assert_allclose(np.asarray(small_matmul(jnp.asarray(a), jnp.asarray(b))), a @ b, rtol=1e-13, atol=1e-13)


def test_small_matmul_has_no_dot_general_for_small_and_keeps_it_for_large():
    # Break: lowering every product elementwise (a 64x64 GEMM materialised as 64^3 products) or none.
    small = jnp.ones((7, 3, 3))
    assert _dot_generals(small_matmul, small, small) == 0
    big = jnp.ones((64, 64))
    assert _dot_generals(small_matmul, big, big) == 1


def test_small_matmul_gradient_is_the_analytic_one():
    # d/dA sum(W * (A @ B)) = W @ B^T and d/dB = A^T @ W (matrix calculus, not the code under test).
    rng = np.random.default_rng(2)
    A, B, W = rng.standard_normal((5, 3, 3)), rng.standard_normal((5, 3, 2)), rng.standard_normal((5, 3, 2))
    gA, gB = jax.grad(lambda a, b: jnp.sum(jnp.asarray(W) * small_matmul(a, b)), argnums=(0, 1))(
        jnp.asarray(A), jnp.asarray(B)
    )
    np.testing.assert_allclose(np.asarray(gA), W @ np.swapaxes(B, -1, -2), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(gB), np.swapaxes(A, -1, -2) @ W, rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------------------------------
# The weak-form entry points: jno.np.matmul and `@` must lower like the einsum spelling,
# which already contracts small tensors elementwise.
# ---------------------------------------------------------------------------------------------------


def _residual_dot_generals(product):
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    x, y, z = d.variable("interior", split=True)[:3]
    xl, yl, zl = d.variable("left", split=True)[:3]
    u, phi = d.fem_symbols(value_shape=(3,))
    X = [x, y, z]
    F = jno.np.eye(3) + jno.np.grad(u, X)
    E = 0.5 * (product(jno.np.transpose(F), F) - jno.np.eye(3))  # Green-Lagrange strain
    fem = jno.fem([jno.np.double_dot(E, jno.np.grad(phi, X)), u(xl, yl, zl) - (0.0, 0.0, 0.0)])
    u0 = jnp.asarray(np.random.default_rng(3).standard_normal(fem.dofs) * 1e-2)
    return _dot_generals(fem._op.residual, u0), np.asarray(fem._op.residual(u0))


@pytest.mark.parametrize(
    "name, product",
    [
        ("jno.np.matmul", jno.np.matmul),
        ("@", lambda a, b: a @ b),
    ],
)
def test_weak_form_product_lowers_like_einsum(name, product):
    # Break: routing either back to jnp.matmul adds a batched dot_general per product.
    n_ref, r_ref = _residual_dot_generals(lambda a, b: jno.np.einsum("...ik,...kj->...ij", a, b))
    n, r = _residual_dot_generals(product)
    assert n == n_ref, f"{name}: {n} dot_general vs {n_ref} for the einsum spelling"
    np.testing.assert_allclose(r, r_ref, rtol=1e-12, atol=1e-14)  # same math, summation order only
