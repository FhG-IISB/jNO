"""``expr.integrate(d)``: the integral of KNOWN fields over a domain -- frozen data and frozen SOLUTIONS of any
number of forms (``u.bind(...).freeze(fem.solve())``) -- as a differentiable node for ``jno.core``.

It lets a reciprocity matrix be written as its formula, e.g. ``C_jk = ∫ ε ∇φ_k · ∇φ_j`` with φ_k, φ_j the
solutions of two different forms. Oracles: the forms' own matrices (``φ_k^T K φ_j``), and the per-form
``expr.integrate(fem)`` route on the eddy-current admittance.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_dofmap import build_dofmap  # noqa: E402

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


def _empty():
    return jno.domain.from_array({"_": np.zeros((1, 1))})


def _eval(nodes):
    crux = jno.core([nodes[0]], domain=_empty())
    return [complex(np.asarray(v).reshape(-1)[0]) for v in crux.eval(list(nodes))]


def _two_conductors():
    """2-D Laplace, conductor k at 1 V, the other at 0 (left / right edges), a scalar parameter so each
    solve is a lazy node"""
    d = jno.shape.rect(0.0, 0.0, 2.0, 1.0, size=0.2).domain()
    phi, v = d.fem_symbols(names=("phi", "v"))
    xi, yi = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    xr, yr = d.variable("right", split=True)[:2]
    kap = jno.np.parameter((1,), name="kap")
    kap.dtype(jnp.float64)
    kap.initialize(jax.nn.initializers.constant(1.0))
    P, V = phi.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    grad = lambda f: jno.np.grad(f, [xi, yi])  # noqa: E731
    fems = [jno.fem([kap * inner(grad(P), grad(V)), phi(xl, yl) - gl, phi(xr, yr) - gr]) for gl, gr in ((1.0, 0.0), (0.0, 1.0))]
    return d, phi, (xi, yi), fems, grad


def test_cross_form_energy_is_phi_k_K_phi_j():
    d, phi, (xi, yi), fems, grad = _two_conductors()
    Phi = [phi.bind(x=xi, y=yi).freeze(f.solve(linear=HOST)) for f in fems]
    C = [[(inner(grad(Phi[k]), grad(Phi[j]))).integrate(d) for k in range(2)] for j in range(2)]
    got = np.array(_eval([C[0][0], C[0][1], C[1][0], C[1][1]])).real.reshape(2, 2)

    sol = [np.asarray(f.solve(linear=HOST, kap=jnp.ones(1))) for f in fems]
    u, w = d.fem_symbols(names=("u", "w"))
    K = np.asarray(jno.fem([inner(grad(u.bind(x=xi, y=yi)), grad(w.bind(x=xi, y=yi)))]).operator[0].todense())
    want = np.array([[sol[k] @ K @ sol[j] for k in range(2)] for j in range(2)])
    np.testing.assert_allclose(got, want, rtol=1e-9)
    assert got[0, 1] < 0 and abs(got[0, 1] - got[1, 0]) < 1e-10 * abs(got[0, 1])   # mutual: negative, symmetric


def test_gradient_with_respect_to_the_solution_values():
    """the node is differentiable in each frozen solution: d(phi_0^T K phi_1)/d(phi_0) = K phi_1"""
    d, phi, (xi, yi), fems, grad = _two_conductors()
    Phi = [phi.bind(x=xi, y=yi).freeze(f.solve(linear=HOST)) for f in fems]
    node = (inner(grad(Phi[0]), grad(Phi[1]))).integrate(d)
    sol = [jnp.asarray(f.solve(linear=HOST, kap=jnp.ones(1))) for f in fems]
    g = np.asarray(jax.grad(lambda a: node.fn(a, sol[1]))(sol[0]))
    u, w = d.fem_symbols(names=("u", "w"))
    K = np.asarray(jno.fem([inner(grad(u.bind(x=xi, y=yi)), grad(w.bind(x=xi, y=yi)))]).operator[0].todense())
    np.testing.assert_allclose(g, K @ np.asarray(sol[1]), rtol=1e-9, atol=1e-12)


def test_eddy_admittance_matches_the_per_form_integrals():
    """complex N1E solutions of two drives: Y_jk = ∫ σ (E_k - j w A_k) · E_j over the domain, with A_k the frozen
    solution of form k, equals the integral over form k"""
    d = jno.shape.box(0, 0, 0, 1, 1, 1).sized(0.5).domain()
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    e, _f = d.fem_symbols(value_shape=(3,), names=("e", "f"), space="N1E")
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    k = jno.np.parameter(s, name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    n = int(build_dofmap(np.asarray(d.mesh.cells_dict["tetra"]), "N1E", 1, n_verts=len(d.mesh.points)).n_dofs)
    H = np.random.default_rng(0).normal(size=(2, n))
    A, V = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    curl = lambda f_: f_.vector.curl(x, y, z)  # noqa: E731
    sigma, w = 3.0 * k, 2.0
    E = [e.bind(x=x, y=y, z=z).freeze(h) for h in H]
    fems = [jno.fem([inner(curl(u), curl(v)) + 1j * w * sigma * inner(A, V) + 0.1 * inner(A, V) - sigma * inner(Ek, V)]) for Ek in E]
    Ak = [u.bind(x=x, y=y, z=z).freeze(f.solve(linear=HOST)) for f in fems]
    Yd = [(sigma * inner(E[kk] - 1j * w * Ak[kk], E[j])).integrate(d) for j in range(2) for kk in range(2)]
    Yf = [(sigma * inner(E[kk] - 1j * w * A, E[j])).integrate(fems[kk], solver=HOST) for j in range(2) for kk in range(2)]
    got, want = np.array(_eval(Yd)), np.array(_eval(Yf))
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_refusals():
    d, phi, (xi, yi), fems, grad = _two_conductors()
    with pytest.raises(ValueError, match="unknown"):
        (inner(grad(phi.bind(x=xi, y=yi)), grad(phi.bind(x=xi, y=yi)))).integrate(d)
    frozen = phi.bind(x=xi, y=yi).freeze(fems[0].solve(linear=HOST))
    v = d.fem_symbols(names=("p", "q"))[1].bind(x=xi, y=yi)
    with pytest.raises(NotImplementedError, match="frozen to a solve"):
        jno.fem([inner(grad(phi.bind(x=xi, y=yi)), grad(v)) - frozen * v]).operator


def test_shared_solves_are_evaluated_once_per_evaluation(monkeypatch):
    """an expression that refers to the same integral (and so the same solves) several times assembles each form
    as often as one reference does: solve and integral nodes are combined per evaluation"""
    import jno.trace as T

    d, phi, (xi, yi), fems, grad = _two_conductors()
    Phi = [phi.bind(x=xi, y=yi).freeze(f.solve(linear=HOST)) for f in fems]
    c01 = (inner(grad(Phi[0]), grad(Phi[1]))).integrate(d)
    calls = []
    orig = T.FemLinearSystem.evaluate
    monkeypatch.setattr(T.FemLinearSystem, "evaluate", lambda self, args=None: calls.append(1) or orig(self, args))
    crux = jno.core([c01], domain=_empty())
    crux.eval([c01])
    once = len(calls)
    calls.clear()
    many = 2.0 * c01 - c01 * c01 + jno.np.stack([c01, c01])[1]
    crux.eval([many])
    assert len(calls) == once, f"{len(calls)} assemblies for one expression with 4 references, {once} for one"
