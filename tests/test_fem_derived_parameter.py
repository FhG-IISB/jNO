"""``.derives(expr)`` — a coefficient that is COMPUTED from the rest of the state each solve.

A coupled problem has coefficients that are neither constants nor unknowns. The ohmic loss driving
a thermal solve is the canonical one: it is whatever the electromagnetic solve just produced, and
re-deriving it every iteration IS the coupling.

The alternative spellings both fail for concrete reasons. Rebuilding the ``jno.fem`` each iteration
cannot be jitted, which costs 4-5x on a real coupled loop (measured: 155-217 ms eager against
39.3 ms jitted). Handing the value over by mutation would have to store a tracer to carry a
gradient, and a tracer stored on a python object belongs to the trace that made it -- it resurfaces
later as ``UnexpectedTracerError``, far from the cause.

An EXPRESSION has neither problem: it is a graph description, dispatched wherever the parameter
would have been read, once per solve at the whole-field level.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
from jno.utils.solver.linear import sparse_lu_solve
from jno.utils.solver.parametric_helpers import _is_frozen_parameter

jax.config.update("jax_enable_x64", True)

L, H = 1.0, 0.25


def problem():
    """-div(grad u) = q on a rectangle, pinned left; q is a P0 field derived from a scalar `a`."""
    d = jno.Shape.rect(0, 0, L, H, size=0.1).domain()
    _ = d.mesh
    n = int(d._cells_p1().shape[0])
    xc = np.asarray(d._points)[np.asarray(d._cells_p1())].mean(axis=1)[:, 0]

    a = jno.np.parameter((1,), name="a")
    a.dtype(jnp.float64)
    a.initialize(jax.nn.initializers.constant(1.0))
    a.optimizer(optax.sgd(0.0))  # present but inert: this test is about the derived value

    shape = lambda av: jnp.asarray(av).reshape(-1)[0] * (1.0 + jnp.asarray(xc))  # noqa: E731
    _r, s0 = d.fem_symbols(space="P0", names=("r", "s"))
    q = jno.np.parameter(s0, name="q").derives(jno.fn(shape, [a], name="qexpr"))

    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - q * vi, u(xl, yl) - 0.0])
    return d, fem, a, q, n, xc, shape


def peak(fem, av, xc):
    a, b = fem.operator.evaluate({"a": jnp.asarray([av]), "q": av * (1.0 + jnp.asarray(xc))})
    return jnp.max(sparse_lu_solve(a, jnp.asarray(b).reshape(-1)))


# --- it is the expression's value, and it is not trained -------------------------------------------


def test_the_value_comes_from_the_expression_not_the_stored_one():
    """A parameter's stored value defaults to ZEROS, and `q` is never initialised or trained.

    So if the derived expression did not fire, the source would be zero and the solution flat. This
    is the test that the mechanism is load-bearing rather than incidental.
    """
    _d, fem, _a, _q, _n, xc, _shape = problem()
    peak_node = jno.fn(lambda t: jnp.max(jnp.asarray(t).reshape(-1)), [fem.solve()], name="Tm")
    crux = jno.core([peak_node], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    got = float(np.asarray(crux.eval(peak_node)))
    assert got > 0, "the stored (zero) value was used -- the derived expression never fired"
    assert got == pytest.approx(float(peak(fem, 1.0, xc)), rel=1e-9)


def test_a_derived_parameter_is_not_a_frozen_coefficient():
    """It is untrained but its value still changes every solve, so it must stay runtime-threaded.

    Were it classed frozen it would be resolved as a known coefficient at assembly and lose the
    per-cell gather entirely.
    """
    _d, _fem, _a, q, _n, _xc, _shape = problem()
    assert getattr(q.model, "_derived_expr", None) is not None
    assert getattr(q.model, "_frozen", False) is True  # not a design variable
    assert _is_frozen_parameter(q) is False  # but NOT a baked-in known coefficient


def test_a_derived_p0_field_varies_across_the_mesh():
    """The whole point: one value per element, not one value broadcast. `shape` grows with x."""
    _d, fem, _a, _q, n, xc, shape = problem()
    vals = np.asarray(shape(jnp.asarray([1.0])))
    assert vals.shape == (n,)
    assert vals.max() / vals.min() > 1.5  # genuinely non-uniform
    # piling the same total source onto the far half must move the answer
    flat = float(peak(fem, 1.0, np.full(n, 1.0)))
    piled = float(peak(fem, 1.0, np.where(xc > L / 2, 2.0, 0.0)))
    assert abs(piled / flat - 1) > 0.05


# --- the gradient reaches what computed it ---------------------------------------------------------


def test_the_gradient_reaches_the_upstream_parameter():
    """u is linear in the source, so peak(a) = a * peak(1) exactly -- an oracle, not a re-run."""
    _d, fem, _a, _q, _n, xc, _shape = problem()
    at1 = float(peak(fem, 1.0, xc))
    g = float(jax.grad(lambda av: peak(fem, av, xc))(1.0))
    assert g == pytest.approx(at1, rel=1e-9)
    assert float(peak(fem, 2.0, xc)) == pytest.approx(2 * at1, rel=1e-9)


# --- and the reason it exists: the coupled objective jits -------------------------------------------


def test_the_derived_coupling_jits():
    _d, fem, _a, _q, _n, xc, _shape = problem()
    f = lambda av: peak(fem, av, xc)  # noqa: E731
    assert float(jax.jit(f)(1.0)) == pytest.approx(float(f(1.0)), rel=1e-12)
    assert float(jax.jit(jax.grad(f))(1.0)) == pytest.approx(float(jax.grad(f)(1.0)), rel=1e-12)


def test_the_expressions_dependencies_are_collected():
    """Regression: a model referenced ONLY inside the derived expression must still be collected.

    The graph walker reaches models through the term list. A derived parameter's dependencies hang
    off the parameter instead, so without visiting `_derived_expr` they are never registered and
    the solve dies at evaluation with "No model for Model <id>" -- pointing at a model the user can
    see in their own script.
    """
    from jno.trace_compiler import TraceCompiler

    _d, fem, a, q, _n, _xc, _shape = problem()
    found = {m.layer_id for m, _ in TraceCompiler.collect_dense_layers(fem.solve())}
    assert q.model.layer_id in found
    assert a.model.layer_id in found, "the derived expression's own dependency was not collected"
