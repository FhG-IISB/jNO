"""Tensor ops in a weak form act on the TENSOR axes, not on the quadrature/basis axes a kernel adds.

`sym`, `antisym` and `trace` all say "over the last two axes" and implement it that way
(`jnp.swapaxes(a, -1, -2)`, `axis1=-2, axis2=-1`), and `identity` carries a leading singleton so it
broadcasts as a constant tensor. `transpose` was the one exception: it forwarded to `jnp.transpose`
with `axes=None`, which reverses EVERY axis -- so on a field of 2x2 tensors it transposed the
quadrature and local-basis axes too and died on a shape mismatch. The one in-repo caller worked around
it by passing `(0, 2, 1)` explicitly.

The oracle throughout is a velocity field whose gradient is known exactly: u = (y, 0) is linear, hence
harmonic, so a Laplace equation carrying it as Dirichlet data reproduces it EXACTLY, giving
g[i,j] = du_i/dx_j = [[0, 1], [0, 0]] -- a nilpotent tensor, which makes tr(g @ g) = 0 while
tr(g @ g^T) = 1. Those two differ precisely when the transpose is done right.

Note for anyone writing a tensor constitutive law here: the matrix product is
``einsum("...ij,...jk->...ik", A, B)`` -- the spelling ``tests/test_fem_finite_strain.py`` uses for
FᵀF. ``inner(A, B, n_contract=1)`` is NOT it: it sums the elementwise product over the last axis of
both operands, which on two 2-tensors returns a vector.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

inner_, grad, trace, sym, transpose, einsum = (
    jno.np.inner,
    jno.np.grad,
    jno.np.trace,
    jno.np.sym,
    jno.np.transpose,
    jno.np.einsum,
)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _evaluate(expr_of_u):
    """Evaluate a scalar formula of grad(u) on the exact field u = (y, 0).

    No solve: the form's residual is assembled at a PRESCRIBED dof vector, so the number depends on
    the tensor algebra alone and not on a solver, an initial guess, or `jno.lag`'s refresh point. The
    scalar block's equation is ``s*w - expr*w``; evaluated at s = 0 its residual is
    ``-expr * integral(w)``, and the P1 test functions form a partition of unity, so summing that
    block over the unit square returns ``-expr`` for a spatially constant expr. u = (y, 0) is linear,
    so grad(u) is exactly [[0, 1], [0, 0]] on every element -- nilpotent, which makes tr(g @ g) = 0
    while tr(g @ g^T) = 1.

    ``expr`` must be NONLINEAR in u: `fem.residual` is offered only on a nonlinear or transient
    problem, so a linear probe like ``trace(grad u)`` has to be squared before it can be read here.
    """
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    xi, yi = d.variable("interior", split=True)[:2]
    ax = [xi, yi]
    u, v = d.fem_symbols(value_shape=(2,), names=("u_t", "v_t"), order=1)
    s, w = d.fem_symbols(names=("s_t", "w_t"), order=1)
    si, wi = s.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            inner_(grad(u, ax), grad(v, ax), n_contract=2),
            si * wi - expr_of_u(u, ax) * wi,
        ]
    )
    pts = np.asarray(fem.points)
    uk = np.zeros(fem.dofs)
    bu = fem.blocks[fem.block_index(u)]
    uvals = np.zeros((len(pts), 2))
    uvals[:, 0] = pts[:, 1]  # u = (y, 0)
    uk[bu.start : bu.stop] = uvals.reshape(-1)
    r = np.asarray(fem.residual(uk))
    bs = fem.blocks[fem.block_index(s)]
    return -float(r[bs.start : bs.stop].sum())


def test_the_harness_reads_a_known_invariant():
    """Calibration, so a later failure is the op under test and not the harness: g:g = 1 and
    tr(g) = 0 for this field, both computed with ops that already had the right convention."""
    assert _evaluate(lambda u, ax: inner_(grad(u, ax), grad(u, ax), n_contract=2)) == pytest.approx(1.0, rel=1e-10)
    # Squared, so the form stays NONLINEAR in u -- `fem.residual` exists only on a nonlinear or
    # transient problem, and `si*wi - trace(grad u)*wi` alone would be linear.
    assert _evaluate(lambda u, ax: trace(grad(u, ax)) ** 2) == pytest.approx(0.0, abs=1e-12)


def test_transpose_swaps_the_tensor_axes_not_the_quadrature_axis():
    """g^T : g is tr(g @ g) = 0 for this nilpotent g. With an all-axes reverse this raises on shape."""
    got = _evaluate(lambda u, ax: inner_(transpose(grad(u, ax)), grad(u, ax), n_contract=2))
    assert got == pytest.approx(0.0, abs=1e-10), f"g^T:g should be tr(g@g) = 0, got {got}"


def test_transpose_agrees_with_sym_which_already_had_the_convention():
    """sym(A) = (A + A^T)/2 is implemented with `swapaxes(-1, -2)`. Writing it out with `transpose`
    must give the same thing, or the two ops disagree about what a transpose is."""
    built_in = _evaluate(lambda u, ax: inner_(sym(grad(u, ax)), sym(grad(u, ax)), n_contract=2))
    spelled = _evaluate(
        lambda u, ax: inner_(
            0.5 * (grad(u, ax) + transpose(grad(u, ax))),
            0.5 * (grad(u, ax) + transpose(grad(u, ax))),
            n_contract=2,
        )
    )
    assert built_in == pytest.approx(spelled, rel=1e-10), f"sym={built_in} vs hand-written={spelled}"


def test_an_explicit_axes_argument_is_still_honoured():
    """The one in-repo caller passes (0, 2, 1) to work around the old default; that must keep working."""
    got = _evaluate(lambda u, ax: inner_(transpose(grad(u, ax), (0, 2, 1)), grad(u, ax), n_contract=2))
    assert got == pytest.approx(0.0, abs=1e-10), f"explicit axes should also give tr(g@g) = 0, got {got}"
