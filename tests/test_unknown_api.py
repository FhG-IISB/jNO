"""One unknown for every method: ``d.unknown()`` (and ``d.unknown.scalar / .vector / .matrix``) + ``u.test()``.

``d.unknown()`` is a valued P1 nodal field -- what ``jno.fdm`` solves for -- and, in ``jno.fem``, the trial
function: the assembler lowers it onto the symbol it was built from. Silent misclassification is the risk
(an unknown read as a known coefficient assembles a valid-looking, wrong system), so every FEM case here is
compared against the same problem written with ``fem_symbols``, and must be BIT-IDENTICAL -- not merely
close.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

J = jno.np
inner = J.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _pair(d, mode, **kw):
    """(trial, test) the old way (fem_symbols) or the new one (d.unknown + u.test())."""
    if mode == "symbols":
        return d.fem_symbols(names=("u", "v"), **kw)
    u = d.unknown(**kw)
    return u, u.test()


def _poisson(mode):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=6).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = _pair(d, mode)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - J.sin(3 * x) * vi, u(xb, yb) - 0.0]).solve(linear=jno.solve.lu())


def _elasticity(mode):
    d = jno.shape.rect(0, 0, 2, 1).structured(n=6).domain()
    d.tag("left", lambda x, y: x < 1e-9)
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    u, v = _pair(d, mode, value_shape=(2,))
    eps = lambda w: J.symgrad(w, [x, y])  # noqa: E731
    sig = lambda e: 2 * e + J.trace(e) * J.identity(2)  # noqa: E731
    vi = v.bind(x=x, y=y)
    return jno.fem([inner(sig(eps(u)), eps(v), n_contract=2) + 0.1 * vi[1], u(xl, yl) - 0.0]).solve(linear=jno.solve.lu())


def _taylor_hood(mode):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=5).domain()
    d.tag("lid", lambda x, y: y > 1 - 1e-9)
    d.tag("walls", lambda x, y: (y < 1e-9) | (x < 1e-9) | (x > 1 - 1e-9))
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("lid", split=True)[:2]
    xw, yw = d.variable("walls", split=True)[:2]
    if mode == "symbols":
        u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
        p, q = d.fem_symbols(names=("p", "q"))
    else:
        u = d.unknown.vector(2, order=2)
        p = d.unknown.scalar(name="p")
        v, q = u.test(), p.test()
    gu, gv = J.jacobian(u, [x, y]), J.jacobian(v, [x, y])
    pi, qi = p.bind(x=x, y=y), q.bind(x=x, y=y)
    fem = jno.fem(
        [
            inner(gu, gv, n_contract=2) - pi * J.trace(gv),
            -qi * J.trace(gu),
            u(xl, yl)[0] - 1.0,
            u(xl, yl)[1] - 0.0,
            u(xw, yw) - 0.0,
            p.pin(),
        ]
    )
    sol = fem.solve(linear=jno.solve.lu())
    assert fem.block_index(u) == 0 and fem.block_index(p) == 1  # an unknown resolves to its block
    return sol


def _transient(mode):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=5).domain(time=(0.0, 0.2, 11))
    x, y, t = d.variable("interior", split=True)[:3]
    xb, yb, _ = d.variable("boundary", split=True)[:3]
    ci = d.variable("initial", split=True)
    u, v = _pair(d, mode)
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    u0 = J.sin(np.pi * ci[0]) * J.sin(np.pi * ci[1])
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(*ci) - u0]).solve().fn()


def _nonlinear(mode):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=6).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = _pair(d, mode)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    return jno.fem([(1 + ui**2) * (ui.x * vi.x + ui.y * vi.y) - 5.0 * vi, u(xb, yb) - 0.0]).solve()


def _inverse(mode):
    """A trainable coefficient FIELD beside the unknown: the operator and the gradient of a loss through
    the solve. The coefficient is a plain `jno.np.parameter` and must stay a coefficient."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=5).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = _pair(d, mode)
    k = J.parameter(v, name="kfield")
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - vi, u(xb, yb) - 0.0])
    assert list(fem.operator.runtime_parameter_exprs) == ["kfield"]

    def loss(kv):
        A, b = fem.operator.evaluate({"kfield": kv})
        return jnp.sum(jnp.linalg.solve(A.todense(), b.reshape(-1)) ** 2)

    k0 = 1.0 + jnp.linspace(0.0, 1.0, 36)
    return np.concatenate([[loss(k0)], np.asarray(jax.grad(loss)(k0))])


def _periodic(mode):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=6).domain()
    d.tag("L", lambda x, y: x < 1e-9)
    d.tag("R", lambda x, y: x > 1 - 1e-9)
    d.tag("B", lambda x, y: (y < 1e-9) | (y > 1 - 1e-9))
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("L", split=True)[:2]
    xr, yr = d.variable("R", split=True)[:2]
    xb, yb = d.variable("B", split=True)[:2]
    u, v = _pair(d, mode)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    f = J.sin(2 * np.pi * x)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xl, yl) - u(xr, yr), u(xb, yb) - 0.0]).solve(
        linear=jno.solve.lu()
    )


def _complex_vector(mode):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=4).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    if mode == "symbols":
        E, v = d.fem_symbols(value_shape=(2,), names=("E", "v"), order=2, complex=True)
    else:
        E = d.unknown.vector(2, order=2, complex=True)
        v = E.test()
    Eb, vb = E.bind(x=x, y=y), v.bind(x=x, y=y)
    curl = lambda F: F.x[1] - F.y[0]  # noqa: E731
    div = lambda F: F.x[0] + F.y[1]  # noqa: E731
    weak = curl(Eb) * curl(vb) + 5 * div(Eb) * div(vb) - (4.0 + 1j) * Eb.dot(vb) - jno.complex(x, y) * vb[0]
    bcs = [E.real(xb, yb)[0] - 0.0, E.real(xb, yb)[1] - 0.0, E.imag(xb, yb)[0] - 0.0, E.imag(xb, yb)[1] - 0.0]
    return jno.fem([weak.real, *bcs]).solve(linear=jno.solve.lu())


def _edge_element(mode):
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    if mode == "symbols":
        u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    else:
        u = d.unknown.vector(3, space="N1E")
        v = u.test()
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    term = inner(u.vector.curl(x, y, z), v.vector.curl(x, y, z)) + inner(ui, vi) - vi[0]
    return jno.fem([term]).solve(linear=jno.solve.lu())


def _symmetric_stress(mode):
    d = jno.shape.rect(0.0, 0.5, 1.0, 1.0).structured(n=4).domain()
    d.tag("inflow", lambda x, y: x < 1e-9)
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("inflow", split=True)[:2]
    if mode == "symbols":
        S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"), order=2, symmetric=True)
    else:
        S = d.unknown.matrix(2, 2, symmetric=True, order=2)
        T = S.test()
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    W, D = J.array([[0.0, 0.5], [-0.5, 0.0]]), J.array([[0.0, 0.5], [0.5, 0.0]])
    R = y * Si.x - (W @ Si - Si @ W) - 2 * D
    return jno.fem([inner(R, Ti + 0.1 * y * Ti.x, n_contract=2), S(xl, yl) - 0.0]).solve(linear=jno.solve.lu())


def _full_matrix_p1(mode):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=4).domain()
    x, y = d.variable("interior", split=True)[:2]
    if mode == "symbols":
        S, T = d.fem_symbols(value_shape=(2, 2), names=("S", "T"))
    else:
        S = d.unknown.matrix(2, 2)
        T = S.test()
    Si, Ti = S.bind(x=x, y=y), T.bind(x=x, y=y)
    return jno.fem(
        [inner(Si.T, Ti, n_contract=2) + 0.1 * inner(Si @ Si, Ti, n_contract=2) - (x * Ti[0, 1] + Ti[1, 0])]
    ).solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-12, atol=1e-14))


CASES = {
    "poisson": _poisson,
    "elasticity": _elasticity,
    "taylor_hood": _taylor_hood,
    "transient": _transient,
    "nonlinear": _nonlinear,
    "inverse": _inverse,
    "periodic": _periodic,
    "complex_vector": _complex_vector,
    "edge_element": _edge_element,
    "symmetric_stress": _symmetric_stress,
    "full_matrix_p1": _full_matrix_p1,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_unknown_is_bit_identical_to_fem_symbols(case):
    """Bit-identical on the CPU. On a GPU the scatter-add reductions run in no fixed order, so the SAME
    fem_symbols build solved twice already differs in the last bits (measured: 5e-17 to 2e-44 here, 1.8e-7
    through an iterative solve before full_matrix_p1 was made direct) -- the comparison there is to round-off."""
    a = np.asarray(CASES[case]("symbols"))
    b = np.asarray(CASES[case]("unknown"))
    assert a.shape == b.shape
    if jax.default_backend() == "cpu":
        assert np.array_equal(a, b), np.max(np.abs(a - b))
    else:
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14 * max(1.0, float(np.max(np.abs(a)))))


def test_the_namespace_spells_the_shape():
    d = jno.shape.rect(0, 0, 1, 1).structured(n=3).domain()
    s, v2, m = d.unknown.scalar(), d.unknown.vector(2), d.unknown.matrix(2, 3)
    assert s.model._unknown_symbol.value_shape == ()
    assert v2.model._unknown_symbol.value_shape == (2,)
    assert m.model._unknown_symbol.value_shape == (2, 3)
    assert np.shape(v2.model.module.value) == (16, 2)  # FDM sizing, as d.unknown(value_shape=(2,))
    S = d.unknown.matrix(2, 2, symmetric=True)
    assert S.value_shape == (2, 2) and S.num_components == 3 and S.test().num_components == 3
    P2 = d.unknown.scalar(order=2)
    assert P2.order == 2 and P2.test().order == 2
    with pytest.raises(ValueError, match="square matrix"):
        d.unknown.matrix(2, 3, symmetric=True)


def test_test_is_derived_and_stable():
    d = jno.shape.rect(0, 0, 1, 1).structured(n=3).domain()
    u = d.unknown.vector(2)
    v = u.test()
    assert u.test() is v
    assert v.field_key == u.model._unknown_symbol.field_key and v.value_shape == (2,)
    a, b = d.fem_symbols(value_shape=(2,))
    assert a.test() is b  # a fem_symbols pair answers the same question the same way
    k = J.parameter(b, name="k")  # a plain coefficient field is not an unknown
    with pytest.raises(TypeError, match="domain.unknown"):
        k.test()


def test_fdm_reads_the_namespace_like_value_shape():
    def solve(make):
        d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).structured().domain()
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        U = make(d)
        Ui = U.vector.bind(x=x, y=y)
        g = J.stack([1.0 + 0 * xb, xb], axis=-1)
        return np.asarray(jno.fdm([Ui.laplacian(), U(xb, yb) - g]).solve())

    a = solve(lambda d: d.unknown(value_shape=(2,)))
    b = solve(lambda d: d.unknown.vector(2))
    assert np.array_equal(a, b)


def test_fdm_refuses_a_fem_only_unknown():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown(order=2)
    ui = u.bind(x=x, y=y)
    with pytest.raises(NotImplementedError, match="FEM-only"):
        jno.fdm([ui.xx + ui.yy - 1.0, u(xb, yb) - 0.0])
