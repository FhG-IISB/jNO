"""``jno.precond.form(...)`` with the default inner factors ONCE with a direct host backend.

The default used SuperLU on the real-equivalent ``2n`` block of a complex form. It now factors the
complex ``n x n`` operator itself (MKL PARDISO where installed, else SuperLU, both complex-native) and
folds the ``2n`` vector the solve hands over around the factors.
"""

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

import jno  # noqa: E402
from jno.utils.solver import linear  # noqa: E402
from jno.utils.solver.solver_api import LinearOperator, PrecondContext  # noqa: E402

inner, vec = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _random_system(n, complex_, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.15, random_state=rng, format="csr") + 4 * sp.eye(n)
    if complex_:
        A = A + 1j * sp.random(n, n, density=0.15, random_state=rng, format="csr")
    return A.tocsr()


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("pardiso", [True, False], ids=["pardiso", "superlu"])
def test_factor_once_solves_and_transposes(complex_, pardiso, monkeypatch):
    if pardiso and not linear._pardiso_available():
        pytest.skip("pypardiso is not installed")
    if not pardiso:
        monkeypatch.setattr(linear, "_pardiso_available", lambda: False)
    A = _random_system(40, complex_)
    f = linear.FactorOnce(A)
    rng = np.random.default_rng(1)
    b = rng.standard_normal(40) + (1j * rng.standard_normal(40) if complex_ else 0)
    np.testing.assert_allclose(A @ f.solve(b), b, atol=1e-10)
    np.testing.assert_allclose(A.T @ f.solve_t(b), b, atol=1e-10)
    for _ in range(3):  # factored once, applied many times
        np.testing.assert_allclose(A @ f.solve(b), b, atol=1e-10)


def test_factor_once_takes_a_complex_vector_on_a_real_matrix():
    A = _random_system(30, False)
    f = linear.FactorOnce(A)
    b = np.random.default_rng(2).standard_normal(30) + 1j * np.random.default_rng(3).standard_normal(30)
    np.testing.assert_allclose(A @ f.solve(b), b, atol=1e-10)


def _twin(k=2.0, size=0.5):
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    nvec = d.variable("boundary", normals=True)
    face = -1j * k * inner(u.vector.cross(nvec), v.vector.cross(nvec))
    Js = vec(0.0 * x, jno.np.sin(2.0 * z), 1.0 + 0.0 * z)
    fem = jno.fem([inner(cu, cv) - k**2 * inner(ui, vi) - inner(Js, vi), face])
    return fem, [inner(cu, cv) - (1 - 0.5j) * k**2 * inner(ui, vi), face]


def test_default_form_on_a_complex_twin_applies_the_inverse_of_the_real_equivalent_block():
    """M^-1 r is the solve with the 2n real-equivalent block, and ``.T`` applies its transpose."""
    fem, twin = _twin()
    spec = jno.precond.form(twin)
    spec.prepare(fem)
    assert spec._op_cplx
    B = np.asarray(spec._op.bcoo.todense())
    ap = spec.materialize(PrecondContext(LinearOperator(spec._op.bcoo), fem))
    r = np.random.default_rng(4).standard_normal(B.shape[0])
    np.testing.assert_allclose(np.asarray(ap(jnp.asarray(r))), np.linalg.solve(B, r), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ap.T(jnp.asarray(r))), np.linalg.solve(B.T, r), rtol=1e-8, atol=1e-10)


def test_default_form_preconditions_a_complex_solve():
    fem, twin = _twin()
    x_lu = np.asarray(jnp.asarray(fem.solve(linear=jno.solve.lu(backend="host")))).reshape(-1)
    x = np.asarray(
        jnp.asarray(fem.solve(linear=jno.solve.gmres(tol=1e-9, restart=60, maxiter=200), precond=jno.precond.form(twin)))
    ).reshape(-1)
    assert np.linalg.norm(x - x_lu) / np.linalg.norm(x_lu) < 1e-6


def test_default_form_keeps_a_real_form_real():
    fem, _ = _twin()
    d = fem.domain
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
    spec = jno.precond.form([inner(ui, vi)])
    spec.prepare(fem)
    assert not spec._op_cplx
    n = int(spec._op.shape[0])
    ap = spec.materialize(PrecondContext(LinearOperator(spec._op.bcoo), fem))
    M = np.asarray(spec._op.bcoo.todense())
    r = np.random.default_rng(5).standard_normal(n)
    out = np.asarray(ap(jnp.asarray(r)))
    assert out.dtype == np.float64
    np.testing.assert_allclose(M @ out, r, atol=1e-9)
    assert isinstance(spec._op.bcoo, jsparse.BCOO)
