"""jno.precond.hypre: reachable, correct on a real H(curl) block, and honest about its limits."""

import numpy as np
import pytest

pytest.importorskip("pygmsh")
pytest.importorskip("petsc4py")
import jax  # noqa: E402

import jno

inner, vec = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _curl_curl(size=0.4, beta=1.0):
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    A_, V_ = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    return (
        d,
        u,
        jno.fem(
            [
                inner(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
                + beta * inner(A_, V_)
                - inner(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), V_),
                u.vector.cross(d.variable("boundary", normals=True)),
            ]
        ),
    )


def test_hypre_ams_solves_a_real_curl_curl_block():
    """The point of the backend: a production AMS on an H(curl) operator."""
    d, u, fem = _curl_curl()
    ref = np.asarray(jno.np.asarray(fem.solve(linear=jno.solve.lu()))).reshape(-1)
    got = np.asarray(
        jno.np.asarray(
            fem.solve(linear=jno.solve.fgmres(tol=1e-9, restart=50, maxiter=500), precond=jno.precond.hypre(kind="ams"))
        )
    ).reshape(-1)
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-8 * max(np.abs(ref).max(), 1.0))


def test_complex_operator_names_real_equivalent():
    """hypre's AMS is real-only. Say so, and name the wrapper that fixes it."""
    import jax.numpy as jnp
    from jax.experimental import sparse as jsp

    from jno.precond import PrecondContext, _Hypre
    from jno.utils.solver.solver_api import LinearOperator

    idx = jnp.asarray([[0, 0], [1, 1]])
    dat = jnp.asarray([1.0 + 1j, 2.0 + 1j])
    op = LinearOperator(jsp.BCOO((dat, idx), shape=(2, 2)))
    with pytest.raises(ValueError, match="real_equivalent"):
        _Hypre("ams", {}).materialize(PrecondContext(op, None))


def test_missing_fem_for_ams_is_explained():
    """AMS needs the edge topology; a bare operator cannot supply it."""
    import jax.numpy as jnp
    from jax.experimental import sparse as jsp

    from jno.precond import PrecondContext, _Hypre
    from jno.utils.solver.solver_api import LinearOperator

    idx = jnp.asarray([[0, 0], [1, 1]])
    dat = jnp.asarray([1.0, 2.0])
    op = LinearOperator(jsp.BCOO((dat, idx), shape=(2, 2)))
    with pytest.raises(TypeError, match="edge topology"):
        _Hypre("ams", {}).materialize(PrecondContext(op, None))
