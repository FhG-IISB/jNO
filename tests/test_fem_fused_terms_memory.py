"""The assembled tangent's memory does not grow with the number of additive weak-form terms.

A weak form is lowered to its additive pieces, and every piece used to be assembled as its own element
kernel with its OWN copy of the tangent's triplet pattern -- one int32 per raw triplet, so
``n_pieces x n_cells x n_test x n_local``. Pieces that share a test field and a region mask have the
identical pattern, so the copies were pure waste: on a 16^3 P1/P1 stabilised Navier-Stokes march the
momentum equation's 16 pieces held 288 MiB of identical index maps, and device memory grew at ~35 kB
per DOF (an 8 GB card ran out at 131k DOFs). The pieces are now fused per (test field, mask).

Two more sources of the same growth are pinned here as well: a march evaluating the tangent in several
places baked one copy of the pattern per evaluation, and the tangent stacked every cell's element block
before its one scatter.

Oracles:
* the assembled tangent equals ``jax.jacfwd`` of the global residual (dense, tiny mesh), an assembly-free
  reference;
* splitting one term into many identical additive pieces leaves the tangent unchanged AND leaves the
  number of mesh-sized index constants in the compiled tangent unchanged;
* a program that evaluates the tangent twice captures the index pattern once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _ns(n=3, pieces=1):
    """Stabilised P1/P1 Navier-Stokes on a cube. ``pieces`` splits the viscous term into that many equal
    additive pieces (nu/pieces each), which must not change the operator -- only how it is lowered."""
    inner, grad, trace, lap = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.laplacian
    nu, dt, e = 0.1, 0.05, 1e-9
    d = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=n).domain(time=(0.0, dt, 2))
    d.tag("wall", lambda x, y, z: (x < e) | (x > 1 - e) | (y < e) | (y > 1 - e) | (z < e) | (z > 1 - e))
    d.point_region("ppin", (0.5, 0.5, 0.5))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=1)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, zi, ti = d.variable("interior", split=True)
    xw, yw, zw = d.variable("wall", split=True)[:3]
    xp, yp, zp = d.variable("ppin", split=True)[:3]
    ci = d.variable("initial", split=True)
    X = [xi, yi, zi]
    ub, vv = u.bind(x=xi, y=yi, z=zi, t=ti), v.bind(x=xi, y=yi, z=zi, t=ti)
    pp, qq = p.bind(x=xi, y=yi, z=zi, t=ti), q.bind(x=xi, y=yi, z=zi, t=ti)
    gu, gv, gp, gq = grad(u, X), grad(v, X), grad(p, X), grad(q, X)
    adv = lambda g, w: inner(g, w, n_contract=1)  # noqa: E731
    G = d.cell_metric
    gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
    tau = jno.lag(((2.0 / dt) ** 2 + gG(ub) + 36.0 * nu**2 * inner(G, G, n_contract=2)) ** -0.5)
    r_m = adv(gu, ub) - nu * lap(u, X) + gp
    mom = inner(ub.t, vv, 1) + inner(adv(gu, ub), vv, 1) - pp * trace(gv)
    for _ in range(pieces):
        mom = mom + (nu / pieces) * inner(gu, gv, 2)
    mom = mom + tau * inner(adv(gv, ub), r_m, n_contract=1)
    con = qq * trace(gu) + tau * inner(gq, r_m, n_contract=1)
    lid = 16 * xw**2 * (1 - xw) ** 2 * 16 * yw**2 * (1 - yw) ** 2 * jno.np.where(zw > 1 - e, 1.0, 0.0)
    return jno.fem(
        [
            mom,
            con,
            p(xp, yp, zp) - 0.0,
            u(xw, yw, zw)[0] - lid,
            u(xw, yw, zw)[1] - 0.0,
            u(xw, yw, zw)[2] - 0.0,
            u(*ci)[0] - 0.0,
            u(*ci)[1] - 0.0,
            u(*ci)[2] - 0.0,
        ]
    )


def _state(fem, seed=0):
    return jnp.asarray(0.1 * np.random.default_rng(seed).standard_normal(fem.dofs))


def _mesh_sized_int_consts(fem, calls=1):
    """Sizes of the integer constants a compiled tangent captures that are at least one entry per cell
    per local DOF pair of the smallest block -- i.e. the per-raw-triplet index maps."""
    w = _state(fem)

    def f(x):
        out = 0.0
        for k in range(calls):
            out = out + fem._op.jacobian(x + k, 0.0, {}).data.sum()
        return out

    jaxpr = jax.make_jaxpr(f)(w)
    sizes = [
        int(np.size(c))
        for c in jaxpr.consts
        if hasattr(c, "dtype") and np.issubdtype(np.asarray(c).dtype, np.integer) and np.size(c) >= 16 * 16 * 27
    ]
    return sorted(sizes)


def test_the_assembled_tangent_equals_the_derivative_of_the_residual():
    fem = _ns(n=2)
    w = _state(fem)
    J = np.asarray(fem._op.jacobian(w, 0.0, {}).todense())
    J_ref = np.asarray(jax.jacfwd(lambda x: fem.residual(x, 0.0))(w))
    assert np.abs(J - J_ref).max() <= 1e-10 * max(1.0, np.abs(J_ref).max())


def test_splitting_a_term_changes_neither_the_tangent_nor_its_index_memory():
    one, many = _ns(n=3, pieces=1), _ns(n=3, pieces=8)
    w = _state(one)
    J1 = np.asarray(one._op.jacobian(w, 0.0, {}).todense())
    J8 = np.asarray(many._op.jacobian(w, 0.0, {}).todense())
    assert np.abs(J1 - J8).max() <= 1e-12 * np.abs(J1).max()
    s1 = _mesh_sized_int_consts(one)
    s8 = _mesh_sized_int_consts(many)
    # Seven more additive pieces used to add seven more copies of the velocity block's pattern.
    assert sum(s8) == sum(s1), (s1, s8)


def test_a_program_evaluating_the_tangent_twice_captures_the_pattern_once():
    fem = _ns(n=3)
    once = _mesh_sized_int_consts(fem, calls=1)
    twice = _mesh_sized_int_consts(fem, calls=2)
    assert sum(twice) == sum(once), (once, twice)


def test_splitting_a_term_leaves_the_march_unchanged():
    """The same form with the viscous term split into 8 pieces marches to the same state."""
    a = np.asarray(_ns(n=3, pieces=1).solve(time=jno.solve.bdf2()).fn())
    b = np.asarray(_ns(n=3, pieces=8).solve(time=jno.solve.bdf2()).fn())
    assert np.abs(a).max() > 1e-3  # the lid drives a non-trivial flow
    assert np.abs(a - b).max() <= 1e-10 * np.abs(a).max()
