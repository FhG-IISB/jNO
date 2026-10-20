"""Which real forms are factored LDL^T inside the compiled program: those whose operator is PROVABLY symmetric.

``operator_key.symmetric_operator`` says yes only when the weak form's bilinear part is unchanged with trial and
test trading places (products, sums and ``inner`` compared as commutative); Dirichlet conditions are eliminated
symmetrically on the steady linear path. Oracle: the assembled matrix itself -- every form called symmetric here
is symmetric to round-off, and every form called not symmetric is not. A GPU test checks the routing: such a form on ``lu(backend="cudss")`` takes the in-program
LDL^T (no host callback), with the values and gradient of the host LU.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

inner = jno.np.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _forms():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    u, v = d.fem_symbols(names=("u", "v"))
    U, V = u.bind(x=x, y=y), v.bind(x=x, y=y)
    g = lambda f: jno.np.grad(f, [x, y])  # noqa: E731
    rho = 1.0 + x * y
    return d, {
        "helmholtz filter": (jno.fem([0.3**2 * inner(g(U), g(V)) + U * V - rho * V]), True),
        "the same, factors in another order": (jno.fem([inner(g(V), g(U)) * 0.09 + V * U - V * rho]), True),
        "variable coefficient": (jno.fem([(1.0 + x) * inner(g(U), g(V)) + 2.0 * U * V - V]), True),
        "convection": (jno.fem([inner(g(U), g(V)) + inner(jno.np.vector(1.0 + 0 * x, 0.5 + 0 * x), g(U)) * V - V]), False),
        "Dirichlet condition": (jno.fem([inner(g(U), g(V)) - V, u(xl, yl) - 0.0]), True),
        "Dirichlet condition, convection": (
            jno.fem([inner(g(U), g(V)) + 2.0 * g(U)[..., 0] * V - V, u(xl, yl) - 1.0]),
            False,
        ),
    }


@pytest.mark.parametrize(
    "name",
    [
        "helmholtz filter",
        "the same, factors in another order",
        "variable coefficient",
        "convection",
        "Dirichlet condition",
        "Dirichlet condition, convection",
    ],
)
def test_symmetric_only_when_the_matrix_is(name):
    _d, forms = _forms()
    fem, expect = forms[name]
    assert fem._symmetric_operator() is expect
    K = np.asarray(fem.operator[0].todense())
    asym = np.abs(K - K.T).max() / np.abs(K).max()
    if expect:
        assert asym < 1e-12, f"called symmetric, but |K - K^T| / |K| = {asym:.2e}"
    else:
        assert asym > 1e-8, "a form this test calls not symmetric assembled to a symmetric matrix: the case tests nothing"


def test_a_symmetric_real_form_is_factored_in_the_program():
    """lu(backend='cudss') on the Helmholtz filter: the in-program LDL^T (cuDSS FFI), not the host callback; value and
    gradient of a functional of the solution equal those of the host LU"""
    import jno.utils.solver.cudss_sym as cs
    import jno.utils.solver.linear as L

    if not cs.available():
        pytest.skip("needs a GPU with the cuDSS extension")
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain()
    x, y = d.variable("interior", split=True)[:2]
    p, q = d.fem_symbols(names=("p", "q"))
    P, Q = p.bind(x=x, y=y), q.bind(x=x, y=y)
    g = lambda f: jno.np.grad(f, [x, y])  # noqa: E731
    k = jno.np.parameter((1,), name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.3))
    fem = jno.fem([0.2**2 * inner(g(P), g(Q)) + P * Q - k * (1.0 + x * y) * Q])
    host_calls = []
    orig = L._cudss_host_solve
    L._cudss_host_solve = lambda *a: host_calls.append(1) or orig(*a)
    try:

        def value_and_grad(linear):
            sol = p.bind(x=x, y=y).freeze(fem.solve(linear=linear))
            J = (sol * sol * (1.0 + x)).integrate(d)
            J = getattr(J, "_expr", J)  # the node under the typed view
            crux = jno.core([J], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
            import equinox as eqx

            from jno.trace_compiler import TraceCompiler

            fn = TraceCompiler.compile_traced_expression(J, crux.all_ops)
            params, static = eqx.partition(eqx.tree_inference(crux._unwrapped_models), eqx.is_inexact_array)
            f = lambda pr: jnp.sum(fn(eqx.combine(pr, static), crux.domain_data.context))  # noqa: E731
            val, grad = jax.jit(jax.value_and_grad(f))(params)
            return float(val), [np.asarray(l_) for l_ in jax.tree_util.tree_leaves(grad)]

        v_cudss, g_cudss = value_and_grad(jno.solve.lu(backend="cudss"))
        assert not host_calls, "the symmetric real form went through the host callback"
        v_host, g_host = value_and_grad(jno.solve.lu(backend="host"))
    finally:
        L._cudss_host_solve = orig
    np.testing.assert_allclose(v_cudss, v_host, rtol=1e-10)
    for a, b in zip(g_cudss, g_host):
        np.testing.assert_allclose(a, b, rtol=1e-8)
