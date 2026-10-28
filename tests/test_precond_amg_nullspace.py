"""``jno.precond.amg(nullspace=<rigid-body modes>)``: smoothed aggregation told the elasticity near-null space.

Smoothed aggregation interpolates the vectors it is given exactly on every coarse level (Vanek, Mandel &
Brezina, Computing 56 (1996) 179, Sec. 3). Given only constants -- pyamg's default -- a 3-D displacement
field's three ROTATIONS are invisible to the coarse grids, and the iteration count grows with the mesh;
given the six rigid-body modes it does not. They are written as formulas in the coordinates and evaluated at
the DOFs -- the same space value ``jno.precond.coarse`` and ``schwarz(coarse=...)`` take.

Oracle: the CG iteration count on linear 3-D elasticity, and the answer, which must not change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("pyamg")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _elasticity(size=0.12):
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    x, y, z = d.variable("interior", split=True)[:3]
    xl, yl, zl = d.variable("left", split=True)[:3]
    u, phi = d.fem_symbols(value_shape=(3,))
    X = [x, y, z]
    rigid = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-y, x, 0), (-z, 0, x), (0, -z, y)]
    eps = lambda w: jno.np.sym(jno.np.grad(w, X))  # noqa: E731
    sig = lambda e: 1.0 * jno.np.trace(e) * jno.np.identity(3) + 2 * 0.5 * e  # noqa: E731
    fem = jno.fem(
        [jno.np.inner(sig(eps(u)), eps(phi), 2) - 0.1 * phi.bind(x=x, y=y, z=z)[2], u(xl, yl, zl) - (0.0, 0.0, 0.0)]
    )
    return fem, rigid


def _iterations(fem, precond):
    u = np.asarray(fem.solve(linear=jno.solve.cg(), precond=precond))
    return fem.stats["linear_iterations"]["total"], u


def test_rigid_modes_cut_the_iterations_and_keep_the_answer():
    # Break: dropping B (pyamg's constants only: the rotations go unresolved) or a wrong mode layout.
    fem, rigid = _elasticity()
    k_plain, u_plain = _iterations(fem, jno.precond.amg())
    k_rigid, u_rigid = _iterations(fem, jno.precond.amg(nullspace=rigid))
    assert k_rigid < 0.5 * k_plain, f"{k_rigid} CG iterations with the rigid modes vs {k_plain} without"
    np.testing.assert_allclose(u_rigid, u_plain, rtol=0, atol=1e-6 * np.abs(u_plain).max())  # cg tol 1e-8


def test_rigid_modes_need_the_problem_and_say_so():
    # Break: guessing a layout from the matrix alone (a 3-component field is not visible in A).
    fem, rigid = _elasticity(size=0.4)
    with pytest.raises(ValueError, match="problem attached"):
        jno.precond.amg(nullspace=rigid).build(fem.A)
    jno.precond.amg(nullspace=rigid).build(fem.A, fem=fem)  # with it, it builds
    with pytest.raises(ValueError, match="not a space"):
        jno.precond.amg(nullspace="rigid")


def test_an_explicit_null_space_of_the_wrong_size_is_refused():
    fem, _rigid = _elasticity(size=0.4)
    with pytest.raises(ValueError, match="rows"):
        jno.precond.amg(nullspace=jnp.ones((7, 6))).build(fem.A, fem=fem)
