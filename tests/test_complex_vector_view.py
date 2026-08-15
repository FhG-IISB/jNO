"""``ComplexVectorView`` — a complex *vector* field reached via ``placeholder.vector.complex``.

Layout ``[..., d, 2]`` (d vector components; last axis = ``[re, im]``). ``.real``/``.imag`` return the
real/imag parts as ``VectorView``\\s; ``.conj``/``.mul`` are componentwise complex algebra. The natural
FEM realisation is two coupled real vector fields ``(E_r, E_i)`` (validated separately via multifield).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.trace import FunctionCall
from jno.trace.views import ComplexVectorView, VectorView
from jno.trace_evaluator import TraceEvaluator


@pytest.fixture(autouse=True)
def _x64():
    """The view algebra is compared at float64 tolerances; opt into x64 per-test (the session default
    may be x64-off when co-run with test_periodic). Save/restore keeps the flag from leaking."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _const(arr):
    """A constant placeholder (0-arg FunctionCall) so the view algebra can be evaluated directly."""
    a = jnp.asarray(arr)
    return FunctionCall(lambda: a, [], "const")


def _eval(expr):
    return np.asarray(TraceEvaluator({}).evaluate(expr, {}, {}, key=jax.random.PRNGKey(0)))


def test_vector_complex_view_api_and_algebra():
    rng = np.random.RandomState(0)
    re = rng.randn(7, 3).astype(np.float64)  # 7 points, d=3 vector, real part
    im = rng.randn(7, 3).astype(np.float64)
    field = np.stack([re, im], axis=-1)  # (7, 3, 2) = complex 3-vector per point

    # API: .vector.complex turns a vector view into a complex-vector view
    E = _const(field).vector.complex
    assert isinstance(E, ComplexVectorView)
    assert isinstance(E.real, VectorView) and isinstance(E.imag, VectorView)

    np.testing.assert_allclose(_eval(E.real.expr), re, atol=1e-9)
    np.testing.assert_allclose(_eval(E.imag.expr), im, atol=1e-9)

    # .conj flips imag, keeps real
    np.testing.assert_allclose(_eval(E.conj.real.expr), re, atol=1e-9)
    np.testing.assert_allclose(_eval(E.conj.imag.expr), -im, atol=1e-9)

    # complex product E.mul(E) = (re + i·im)^2 = (re² - im²) + i·(2·re·im), componentwise
    sq = E.mul(E)
    np.testing.assert_allclose(_eval(sq.real.expr), re * re - im * im, atol=1e-9)
    np.testing.assert_allclose(_eval(sq.imag.expr), 2.0 * re * im, atol=1e-9)

    # per-component modulus
    np.testing.assert_allclose(_eval(E.abs.expr), np.sqrt(re**2 + im**2), atol=1e-9)


def test_vector_complex_mul_by_complex_scalar_broadcasts():
    """A complex *scalar* (ComplexView) multiplied into a complex vector broadcasts over the
    components — e.g. a Bloch phase factor e^{iθ} applied to the whole field."""
    rng = np.random.RandomState(1)
    re = rng.randn(5, 2).astype(np.float64)
    im = rng.randn(5, 2).astype(np.float64)
    field = np.stack([re, im], axis=-1)  # (5, 2, 2)
    cr, ci = 0.6, -0.8
    cfield = np.array([cr, ci])  # a single complex scalar [re, im]

    E = _const(field).vector.complex
    c = _const(cfield).complex  # ComplexView (scalar)
    out = E.mul(c)
    # (cr + i·ci)(re + i·im) = (cr·re − ci·im) + i(cr·im + ci·re), broadcast c over the d components
    np.testing.assert_allclose(_eval(out.real.expr), cr * re - ci * im, atol=1e-9)
    np.testing.assert_allclose(_eval(out.imag.expr), cr * im + ci * re, atol=1e-9)


def test_complex_vector_bind_is_reachable():
    """``ComplexVectorView.bind`` existed but had no wrapper registered, so it raised ``KeyError``.

    ``ComplexVectorView`` was the one view class defining ``partials``/``bind`` (views.py) while being
    absent from ``_NAMED_PARTIALS_CLS_FOR``, so every call died in the dispatch table rather than
    doing anything. Regression guard for that registration.
    """
    from jno.trace import Variable
    from jno.trace.views import NamedComplexVectorViewWithPartials

    from tests.conftest import MockDomain

    d = MockDomain()
    d.context["xy"] = jnp.zeros((10, 2))
    x = Variable("xy", [0, 1], domain=d)
    y = Variable("xy", [1, 2], domain=d)

    # (N, d, 2): 3 points, 2 vector components, [re, im]
    field = jnp.arange(12, dtype=jnp.float64).reshape(3, 2, 2)
    E = _const(field).vector.complex
    assert isinstance(E, ComplexVectorView)

    bound = E.bind(x=x, y=y)
    assert isinstance(bound, NamedComplexVectorViewWithPartials)
    assert sorted(object.__getattribute__(bound, "_coord_vars").keys()) == ["x", "y"]

    # Re-bind stays on the same class and merges (the C1 dispatch fix).
    assert type(bound.bind(x=x)) is type(bound)
