"""A direct solver (``lu``) in a transient march whose operator takes a runtime coefficient.

The step operator M + theta dt A(k) is constant over the march once k is given, so a sparse LU can factor
it; the step used to hand the solver only a matvec, which a direct solver cannot factor.
"""

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


def _heat(theta):
    d = jno.shape.rect(0, 0, 1, 1, size=0.2).domain(time=(0.0, 0.2, 6))
    x, y, t = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    k = jno.np.parameter(s, name="k")
    k.dtype(jnp.float64)
    grad, inner = jno.np.grad, jno.np.inner
    ut, pt = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    fem = jno.fem(
        [
            ut.t * pt + k * inner(grad(u, [x, y]), grad(phi, [x, y])) - pt,
            u(*d.variable("boundary", split=True)) - 0.0,
            u(*d.variable("initial", split=True)) - 0.0,
        ]
    )
    kv = jnp.asarray(np.random.default_rng(0).uniform(0.5, 2.0, len(d.mesh.cells_dict["triangle"])))
    return fem, kv, jno.solve.theta(theta)


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_lu_matches_the_iterative_march(theta):
    fem, kv, scheme = _heat(theta)
    direct = np.asarray(fem.solve(linear=jno.solve.lu(), time=scheme, k=kv))
    krylov = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-12), time=scheme, k=kv))
    assert np.abs(krylov).max() > 1e-3, "the source never arrived"
    np.testing.assert_allclose(direct, krylov, rtol=0, atol=1e-9)  # cg tol 1e-12 on an O(1) system
