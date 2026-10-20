"""``fem.eval(F, u)`` / ``F.integrate(fem)`` for a test-free ``F`` on a non-nodal (N1E) form, real and complex.

Oracles are the form's own matrices: on the assembly's quadrature an energy integral is ``u^T K u`` exactly,
and the current through a source field ``E`` (an N1E field, DOFs h) of an eddy-current form,
``I = ∫ sigma (E - j w A) . E``, is ``h^T M_sigma (h - j w a)``.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D box meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_dofmap import build_dofmap  # noqa: E402

inner = jno.np.inner
W = 2.0


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _setup():
    d = jno.shape.box(0, 0, 0, 1, 1, 1).sized(0.45).domain()
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    e, _f = d.fem_symbols(value_shape=(3,), names=("e", "f"), space="N1E")
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    k = jno.np.parameter(s, name="k")
    k.dtype(jnp.float64)
    n = int(build_dofmap(np.asarray(d.mesh.cells_dict["tetra"]), "N1E", 1, n_verts=len(d.mesh.points)).n_dofs)
    h = np.random.default_rng(0).normal(size=n)
    A, V = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    E = e.bind(x=x, y=y, z=z).freeze(h)
    curl = lambda f: f.vector.curl(x, y, z)  # noqa: E731
    return d, (x, y, z), u, v, A, V, E, curl, k, h, len(d.mesh.cells_dict["tetra"])


def test_energy_integral_is_u_K_u():
    d, xyz, u, v, A, V, E, curl, k, h, nc = _setup()
    J = jno.np.vector(1.0 + 0.0 * xyz[0], xyz[1], 0.0 * xyz[0])
    fem = jno.fem([inner(curl(u), curl(v)) + inner(A, V) - inner(J, V)])
    K, _b = fem.operator
    a = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    got = float(fem.eval(inner(curl(u), curl(u)) + inner(A, A), a))
    want = float(a @ (np.asarray(K.todense()) @ a))
    np.testing.assert_allclose(got, want, rtol=1e-12)


def _eddy(d, u, v, A, V, E, curl, k):
    sigma = 3.0 * k
    fem = jno.fem([inner(curl(u), curl(v)) + 1j * W * sigma * inner(A, V) + 0.1 * inner(A, V) - sigma * inner(E, V)])
    current = sigma * inner(E - 1j * W * A, E)
    Msig = jno.fem([sigma * inner(A, V)]).operator
    return fem, current, Msig


def test_complex_current_through_a_source_field():
    d, xyz, u, v, A, V, E, curl, k, h, nc = _setup()
    fem, current, Msig = _eddy(d, u, v, A, V, E, curl, k)
    kv = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, nc))
    a = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"), k=kv))
    M = np.asarray(Msig.evaluate({"k": kv})[0].todense())
    want = h @ M @ (h - 1j * W * a)
    got = complex(fem.eval(current, a, args={"k": kv}))
    np.testing.assert_allclose(got, want, rtol=1e-11)


def test_traced_integral_and_its_gradient():
    d, xyz, u, v, A, V, E, curl, k, h, nc = _setup()
    fem, current, Msig = _eddy(d, u, v, A, V, E, curl, k)
    k.initialize(jax.nn.initializers.constant(1.0))
    node = current.integrate(fem, solver=jno.solve.lu(backend="host"))
    R = jno.fn(lambda i: jnp.real(1.0 / i), [node], name="R")
    crux = jno.core([R], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    traced = float(np.asarray(crux.eval([R])).reshape(-1)[0])

    def r_of(kv):
        a = fem.solve(linear=jno.solve.lu(backend="host"), k=kv)
        return jnp.real(1.0 / fem.eval(current, a, args={"k": kv}))

    k1 = jnp.ones(nc)
    np.testing.assert_allclose(traced, float(r_of(k1)), rtol=1e-10)
    g = np.asarray(jax.grad(r_of)(k1))
    dv = jnp.asarray(np.random.default_rng(2).normal(size=nc))
    hh = 1e-5
    fd = (float(r_of(k1 + hh * dv)) - float(r_of(k1 - hh * dv))) / (2 * hh)
    np.testing.assert_allclose(g @ np.asarray(dv), fd, rtol=1e-6)


def test_boundary_integrand_is_refused():
    d, xyz, u, v, A, V, E, curl, k, h, nc = _setup()
    fem = jno.fem([inner(curl(u), curl(v)) + inner(A, V) - inner(E, V)])
    a = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    cb = d.variable("boundary", split=True)
    xb, yb, zb = cb[0], cb[1], cb[2]
    with pytest.raises(NotImplementedError, match="boundary"):
        fem.eval(inner(u.bind(x=xb, y=yb, z=zb), u.bind(x=xb, y=yb, z=zb)), a)


def test_a_view_integrand_takes_the_fem():
    """`.integrate(fem)` on a view (here a frozen P0 field makes the integrand a ScalarView) forwards the FEM,
    like the bare expression; it used to accept keyword arguments only"""
    d, xyz, u, v, A, V, E, curl, k, h, nc = _setup()
    r, _s = d.fem_symbols(space="P0", names=("r0", "s0"))
    half = r.bind(x=xyz[0], y=xyz[1], z=xyz[2]).freeze(np.full(nc, 0.5))
    fem = jno.fem([inner(curl(u), curl(v)) + inner(A, V) - inner(E, V)])
    integrand = half * inner(A, A)
    node = integrand.integrate(fem, solver=jno.solve.lu(backend="host"))
    F = jno.fn(lambda i: i, [node], name="F")
    crux = jno.core([F], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    a = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    np.testing.assert_allclose(
        float(np.asarray(crux.eval([F])).reshape(-1)[0]), 0.5 * float(fem.eval(inner(A, A), a)), rtol=1e-10
    )
