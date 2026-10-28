"""``jno.solve.lu(backend="cudss")`` on a complex SYMMETRIC form: the n x n complex LDL^T inside the program.

A complex form (``nu curl curl + j w sigma``) is assembled as the fused real block ``[[R, -I], [I, R]]``; the
cuDSS backend factors ``R + jI`` (LDL^T) through XLA custom calls, so ``jax.jit(jax.grad(...))`` of a solve
runs on the GPU. Oracle: the same solve on host SuperLU (the real 2n block), values and gradients.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D box meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
import jno.utils.solver.cudss_sym as cs  # noqa: E402

pytestmark = pytest.mark.skipif(not cs.available(), reason="needs a GPU with the cuDSS extension (nvcc, nvidia-cudss)")
inner = jno.np.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _form(nonsymmetric=False):
    d = jno.shape.box(0, 0, 0, 1, 1, 1).sized(0.35).domain()
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    A, V = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cA, cV = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    k = jno.np.parameter(s, name="k")
    k.dtype(jnp.float64)
    J = jno.np.vector(1.0 + 0.0 * x, y, 0.0 * x)
    terms = inner(cA, cV) + (0.5 + 2j) * k * inner(A, V) - inner(J, V)
    if nonsymmetric:
        terms = terms + 0.3j * A[0] * V[1]  # an off-diagonal tensor: not symmetric (curl A . V would be)
    fem = jno.fem([terms, u.vector.cross(d.variable("boundary", normals=True))])
    fem._probe = {"A": A, "k": k}  # for the functional test
    return fem, len(d.mesh.cells_dict["tetra"])


def test_solution_matches_host_lu():
    fem, nc = _form()
    k = jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, nc))
    a = np.asarray(fem.solve(linear=jno.solve.lu(backend="cudss"), k=k))
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"), k=k))
    assert np.iscomplexobj(a)
    np.testing.assert_allclose(a, ref, rtol=1e-9, atol=1e-11 * np.abs(ref).max())


def test_jit_gradient_matches_host_lu_and_two_solves_per_program():
    fem, nc = _form()
    k0 = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, nc))
    obs = jnp.asarray(np.random.default_rng(2).normal(size=fem.solve(linear=jno.solve.lu(backend="host"), k=k0).shape))

    def f(k, backend):
        a = fem.solve(linear=jno.solve.lu(backend=backend), k=k)
        return jnp.sum(jnp.abs(a - obs) ** 2)

    g_gpu = jax.jit(jax.grad(lambda k: f(k, "cudss")))(k0)
    g_ref = jax.grad(lambda k: f(k, "host"))(k0)
    np.testing.assert_allclose(np.asarray(g_gpu), np.asarray(g_ref), rtol=1e-7, atol=1e-10 * np.abs(g_ref).max())
    # the same form twice in one program: each solve keeps its own factors for its adjoint
    g2 = jax.jit(jax.grad(lambda k: f(k, "cudss") + f(0.7 * k, "cudss")))(k0)
    g_sep = g_ref + 0.7 * jax.grad(lambda k: f(k, "host"))(0.7 * k0)
    np.testing.assert_allclose(np.asarray(g2), np.asarray(g_sep), rtol=1e-7, atol=1e-10 * np.abs(g_sep).max())


def test_values_that_are_not_symmetric():
    """symmetric pattern, non-symmetric values: eagerly the values are checked and the solve stays on LU (the
    host answer); under jit they cannot be, and the residual gate refuses what LDL^T of the symmetric part gives"""
    fem, nc = _form(nonsymmetric=True)
    k = jnp.ones(nc)
    a = np.asarray(fem.solve(linear=jno.solve.lu(backend="cudss"), k=k))
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"), k=k))
    np.testing.assert_allclose(a, ref, rtol=1e-9, atol=1e-11 * np.abs(ref).max())
    with pytest.raises(RuntimeError, match="LDL"):
        jax.block_until_ready(jax.jit(lambda k_: fem.solve(linear=jno.solve.lu(backend="cudss"), k=k_))(k))
        from jno.utils.solver.solver_api import raise_if_gate_failed

        raise_if_gate_failed()


def test_integrate_with_the_cudss_solver_takes_the_ldl_path(monkeypatch):
    """`expr.integrate(fem, solver=lu(backend="cudss"))` fills the linear slot with the FEM as context, so the
    functional's shared solve is the in-program LDL^T one (and agrees with host LU)"""
    import jno.utils.solver.linear as L

    fem, nc = _form()
    calls = []
    orig = L.cudss_ldl_solve
    monkeypatch.setattr(L, "cudss_ldl_solve", lambda *a, **k: calls.append(1) or orig(*a, **k))
    A, k = fem._probe["A"], fem._probe["k"]
    k.initialize(jax.nn.initializers.constant(1.0))
    vals = []
    for backend in ("cudss", "host"):
        fem._functional_sol = None
        node = (k * inner(A, A)).integrate(fem, solver=jno.solve.lu(backend=backend))
        F = jno.fn(lambda i: jnp.real(i), [node], name="F")
        crux = jno.core([F], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
        vals.append(float(np.asarray(crux.eval([F])).reshape(-1)[0]))
    assert calls, "the LDL^T path was not taken"
    np.testing.assert_allclose(vals[0], vals[1], rtol=1e-9)
