"""One-way coupling: a field frozen to another form's solve as a coefficient of a (non-nodal) weak form.

``rho_f = r.bind(...).freeze(filter_fem.solve())`` inside ``jno.fem([...])`` makes that solve a runtime input
of the second form: jno.core evaluates the filter first, the eddy form assembles with its solution, and the
gradient flows through both solves -- the density filter of multi-material topology optimisation, Joule heating
from an electric solve, any staggered chain. Oracle: the same chain written as two explicit solves.
"""

import numpy as np
import pytest

pytest.importorskip("gmsh")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

inner = jno.np.inner
HOST = jno.solve.lu(backend="host")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _chain():
    d = jno.shape.box(0, 0, 0, 1, 1, 1).sized(0.5).domain()
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    # the filter: r^2 grad(rf).grad(q) + rf q = k q   (P1, natural BC), k a P0 design density
    rf, q = d.fem_symbols(names=("rf", "q"))
    _r0, s = d.fem_symbols(space="P0", names=("r0", "s0"))
    k = jno.np.parameter(s, name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(0.8))
    RF, Q = rf.bind(x=x, y=y, z=z), q.bind(x=x, y=y, z=z)
    grad = lambda f: jno.np.grad(f, [x, y, z])  # noqa: E731
    filt = jno.fem([0.1**2 * inner(grad(RF), grad(Q)) + RF * Q - k * Q])
    # the eddy-like form with the FILTERED density as its coefficient
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    A, V = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    curl = lambda f: f.vector.curl(x, y, z)  # noqa: E731
    J = jno.np.vector(1.0 + 0.0 * x, y, 0.0 * x)
    rho = rf.bind(x=x, y=y, z=z).freeze(filt.solve(linear=HOST))
    eddy = jno.fem([inner(curl(u), curl(v)) + (0.2 + rho * rho) * inner(A, V) - inner(J, V)])
    energy = inner(A, A)
    return d, filt, eddy, energy, k, len(d.mesh.cells_dict["tetra"])


def _two_steps(filt, eddy, energy, kv):
    """the oracle: solve the filter, hand its solution to the eddy form by name, integrate"""
    name = next(n for n in eddy.operator.runtime_parameter_exprs if n.startswith("frozen_solution_"))
    rf = filt.solve(linear=HOST, k=kv)
    a = eddy.solve(linear=HOST, **{name: rf})
    return eddy.eval(energy, a, args={name: rf})


def test_jno_core_value_equals_the_two_explicit_solves():
    d, filt, eddy, energy, k, nc = _chain()
    node = energy.integrate(eddy, solver=HOST)
    crux = jno.core([node], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    got = float(np.asarray(crux.eval([node])).reshape(-1)[0])
    want = float(_two_steps(filt, eddy, energy, jnp.full(nc, 0.8)))
    np.testing.assert_allclose(got, want, rtol=1e-10)
    # independent of the coupling path: the CONCRETE filter solution frozen through the per-vertex P1 path
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    u, v = d.fem_symbols(value_shape=(3,), names=("u2", "v2"), space="N1E")
    A, V = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    curl = lambda f: f.vector.curl(x, y, z)  # noqa: E731
    rf2, _q2 = d.fem_symbols(names=("rf2", "q2"))
    rho_c = rf2.bind(x=x, y=y, z=z).freeze(np.asarray(filt.solve(linear=HOST, k=jnp.full(nc, 0.8))))
    J = jno.np.vector(1.0 + 0.0 * x, y, 0.0 * x)
    eddy_c = jno.fem([inner(curl(u), curl(v)) + (0.2 + rho_c * rho_c) * inner(A, V) - inner(J, V)])
    a_c = eddy_c.solve(linear=HOST)
    np.testing.assert_allclose(got, float(eddy_c.eval(inner(A, A), a_c)), rtol=1e-9)


def test_the_gradient_flows_through_both_solves():
    d, filt, eddy, energy, k, nc = _chain()
    k0 = jnp.asarray(np.random.default_rng(0).uniform(0.3, 1.0, nc))
    f = lambda kv: _two_steps(filt, eddy, energy, kv)  # noqa: E731
    g = np.asarray(jax.grad(f)(k0))
    dv = jnp.asarray(np.random.default_rng(1).normal(size=nc))
    h = 1e-5
    fd = (float(f(k0 + h * dv)) - float(f(k0 - h * dv))) / (2 * h)
    np.testing.assert_allclose(g @ np.asarray(dv), fd, rtol=1e-6)
    assert np.abs(g).max() > 0


def test_an_eager_solve_without_the_coupled_value_says_how():
    d, filt, eddy, energy, k, nc = _chain()
    with pytest.raises(ValueError, match="frozen to another form"):
        eddy.operator.evaluate({})
