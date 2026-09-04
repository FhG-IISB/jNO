"""``jno.precond.real_equivalent``: precondition a COMPLEX system through its fused real block.

A complex symmetric ``A = K + iM`` becomes ``[[K, -M], [M, K]]``. That block is skew-dominated
(measured ``||A - A^T||/||A|| = 2.0``), so multigrid applied to it directly diverges — which is why
jNO's real-only GPU multigrid could not be used on a complex problem at all.

The classical fix is to precondition with ``K + M``: real, symmetric, and definite whenever ``K`` is
and ``M >= 0``. The inner solver then never sees a complex number, so any real spec composes and a
real-only GPU multigrid applies unchanged.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
import jax  # noqa: E402

import jno  # noqa: E402

inner_, vec_ = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _complex_eddy(size=0.34, w=1.0e4):
    """curl-curl + j*w*mass on N1E: complex symmetric, with the mass term making it non-singular."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    A_, V_ = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    return d, jno.fem(
        [
            inner_(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
            + 1j * w * inner_(A_, V_)
            - inner_(vec_(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), V_),
            u.vector.cross(d.variable("boundary", normals=True)),
        ]
    )


def _residual(fem, x):
    from jno.precond import _fem_concrete_operator

    A = _fem_concrete_operator(fem)
    Ad = np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)
    b = np.asarray(jno.np.asarray(fem.b)).reshape(-1)
    n = Ad.shape[0]
    if b.shape[0] == 2 * n:
        b = b[:n] + 1j * b[n:]
    x = np.asarray(x).reshape(-1)
    if x.shape[0] == 2 * n:
        x = x[:n] + 1j * x[n:]
    return float(np.linalg.norm(Ad @ x - b) / max(np.linalg.norm(b), 1e-300))


def test_it_solves_a_complex_system_with_a_real_inner():
    """The point: the inner spec never sees a complex number, and the complex system still solves."""
    _d, fem = _complex_eddy()
    x = fem.solve(
        linear=jno.solve.fgmres(tol=1e-9, restart=120, maxiter=600),
        precond=jno.precond.real_equivalent(jno.precond.jacobi()),
    )
    assert _residual(fem, np.asarray(jno.np.asarray(x))) < 1e-6


def test_it_is_not_complex_native():
    """It must NOT declare complex-native: it wants the fused 2n form, since handing a REAL operator
    to a real inner solver is its entire purpose. Declaring otherwise would route around it."""
    assert not getattr(jno.precond.real_equivalent(jno.precond.jacobi()), "complex_native", False)


def test_a_matrix_free_operator_is_refused():
    """It reads K and M off the assembled block, so it cannot run matrix-free — and says so."""
    import jax.numpy as jnp

    from jno.precond import PrecondContext, _RealEquivalent
    from jno.utils.solver.solver_api import LinearOperator

    class _MatFree:  # a matvec-only operator: no assembled sparse form to read K and M from
        shape = (8, 8)

        def __matmul__(self, v):
            return jnp.asarray(v)

    op = LinearOperator(_MatFree())
    with pytest.raises(ValueError, match="ASSEMBLED"):
        _RealEquivalent(jno.precond.jacobi()).materialize(PrecondContext(op, None))


def test_an_odd_sized_operator_is_refused():
    """The fused real-equivalent block is 2n by construction; an odd size means it is not one."""
    import jax.experimental.sparse as jsp
    import jax.numpy as jnp

    from jno.precond import PrecondContext, _RealEquivalent
    from jno.utils.solver.solver_api import LinearOperator

    op = LinearOperator(jsp.BCOO.fromdense(jnp.eye(7)))
    with pytest.raises(ValueError, match="EVEN"):
        _RealEquivalent(jno.precond.jacobi()).materialize(PrecondContext(op, None))


def test_the_inner_is_built_from_K_plus_M_not_from_the_complex_operator():
    """`prepare` must NOT be forwarded to the inner spec.

    `prepare` is where a spec may eagerly freeze itself against the FEM's own operator -- and that is
    the wrong operator here by construction, because this spec exists to hand the inner a real
    ``K + M``. Forwarding it made AMS freeze COMPLEX auxiliaries and then apply them to a real
    matrix: still convergent, since a preconditioner need not be exact, but preconditioning the wrong
    operator and casting each complex auxiliary solution down to the real right-hand side -- 960
    "discards the imaginary part" warnings in one solve.

    Zero such warnings is the observable, and it is also what keeps the auxiliaries REAL, which is
    what lets them run on device instead of through a host callback.
    """
    import warnings

    _d, fem = _complex_eddy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = fem.solve(
            linear=jno.solve.fgmres(tol=1e-9, restart=120, maxiter=800),
            precond=jno.precond.real_equivalent(jno.precond.ams(aux=jno.precond.amg())),
        )
        n_cast = sum("imaginary part" in str(m.message) for m in caught)
    assert n_cast == 0, f"{n_cast} complex->real casts: the inner was built from the wrong operator"
    assert _residual(fem, np.asarray(jno.np.asarray(x))) < 1e-6
