"""A PEEC network whose material is a ``jno.np.parameter`` -- inverse design through ``jno.core``.

`jno.peec` was differentiable all along: `jax.grad` runs through a solve, and a per-cell conductivity
is exactly the SIMP density a topology optimisation moves. What was missing was the LAST STEP -- the
readouts were concrete numbers, so reaching `jno.core` meant wrapping the solve in `jno.fn` by hand
and threading the parameter yourself.

This is the same treatment `jno.rcwa` already gives its readouts (`_ParametricSol`): when the problem
carries trainable parameters, the readouts become TRACE NODES over them, so an objective is written
as `emag.solve().R` and crux threads the values. The parameters live in the problem statement, which
is the house rule everywhere else -- geometry carries the material.

The oracles are the ones that would catch it being quietly wrong: a parameter-free network must be
untouched (it is the overwhelmingly common case), a node evaluated at the initial values must equal
the concrete solve at those values (so the node is the same physics, not a re-derivation), and the
gradient must match the hand-written `jno.fn` form it replaces.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno

jax.config.update("jax_enable_x64", True)

CU, mm = 5.8e7, 1e-3


def _net(sigma):
    bar = jno.Shape.box(0, 0, 0, 16 * mm, 4 * mm, 0.8 * mm, size=(0.8 * mm,) * 3).attach(sigma=sigma).name("bar")
    d = bar.domain()
    d.tag("A", lambda x, y, z: x < 0.9 * mm)
    d.tag("B", lambda x, y, z: x > 15.1 * mm)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6)


def _ncell():
    return int(np.prod(np.asarray(_net(CU).build().fil.lattice["n"])))


def _rho(value=0.5, name="rho"):
    p = jno.np.parameter((_ncell(),), name=name)
    p.initialize(jax.nn.initializers.constant(value))
    return p


def _dummy():
    return jno.domain.from_array({"_": np.zeros((1, 1))})


def _table_for(node, name=None, value=None):
    """The evaluator table for a node's parameters, optionally substituting one value."""
    import equinox as eqx

    from jno._fem import _walk

    table = {}
    for mc in (n for n in _walk(node) if type(n).__name__ == "ModelCall"):
        mod = mc.model.module
        if name is not None and getattr(mc.model, "_parameter_name", None) == name:
            mod = eqx.tree_at(lambda m: m.value, mod, jnp.asarray(value))
        table[mc.model.layer_id] = mod
    return table


def _value_of(node, name=None, value=None):
    """Evaluate a node at an EXPLICIT parameter value.

    Not at the parameter's own `.value`: `.initialize(...)` only records an initializer for the
    training loop to apply, so a freshly built parameter still reads as zeros -- and a conductivity
    of zero is a singular network, not a small one.
    """
    from jno.trace_evaluator import TraceEvaluator

    return TraceEvaluator(_table_for(node, name, value)).evaluate(node, context={})


def test_a_parameter_free_network_is_untouched():
    """The governing constraint. Parametric readouts are for the network that asked for them; every
    existing model must still get a concrete number, not a node it has to evaluate."""
    sol = _net(CU).solve()
    assert isinstance(complex(sol.Z), complex)
    assert float(np.real(sol.R)) > 0


def test_a_parameter_makes_the_readouts_trace_nodes():
    """The feature: no `jno.fn`, no lambda, no threading by hand."""
    from jno.trace import Placeholder

    sol = _net(CU * _rho() ** 3).solve()
    for readout in (sol.R, sol.L, sol.Z, sol.joule):
        assert isinstance(readout, Placeholder), type(readout)
    assert isinstance(sol.current("A"), Placeholder)
    assert isinstance(sol.voltage("A", "B"), Placeholder)


def test_the_node_agrees_with_the_concrete_solve_at_the_same_values():
    """A node that computed something SUBTLY different would still train, and would optimise the
    wrong thing. So it is pinned against the concrete path at the initial parameter values."""
    rho0 = 0.6
    node = _net(CU * _rho(rho0) ** 3).solve().R
    got = float(np.real(np.asarray(_value_of(node, "rho", jnp.full((_ncell(),), rho0)))))
    want = float(np.real(_net(CU * rho0**3).solve().R))
    assert abs(got / want - 1) < 1e-9, (got, want)


def test_the_gradient_matches_the_hand_written_form_it_replaces():
    """`jno.fn(lambda r: ... built.solve(sigma={...}) ...)` already worked. The node has to give the
    same gradient, or it is a different objective wearing a nicer API."""
    ncell = _ncell()
    built = _net(CU).build()

    def hand(r):
        return jnp.real(built.solve(sigma={"bar": CU * r**3}).R)

    r0 = jnp.full((ncell,), 0.5)
    want = np.asarray(jax.grad(hand)(r0))

    p = _rho(0.5)
    node = _net(CU * p**3).solve().R
    got = np.asarray(_grad_of(node, "rho", r0))
    assert np.allclose(got, want, rtol=1e-6, atol=0), (got[:3], want[:3])


def _grad_of(node, name, value):
    """d(node)/d(parameter) by substituting the value into the trace and differentiating."""
    from jno.trace_evaluator import TraceEvaluator

    def f(val):
        return jnp.real(TraceEvaluator(_table_for(node, name, val)).evaluate(node, context={}))

    return jax.grad(f)(value)


def test_it_trains_through_jno_core_and_the_loss_falls():
    """End to end, which is the whole point: an objective written as `emag.solve().R` and handed to
    `jno.core`, with the parameter threaded by crux rather than by the user."""
    p = _rho(0.5)
    p.optimizer(optax.adam(5e-2))
    loss = _net(CU * p**3).solve().R * 1e3
    st = jno.core([loss], domain=_dummy()).solve(epochs=12)
    h = np.asarray(st.total_loss_history).reshape(-1)
    assert h.size >= 2 and h[-1] < h[0], (h[0], h[-1])
    assert h[-1] / h[0] < 0.95, f"resistance barely moved: {h[0]} -> {h[-1]}"


def test_dissipation_and_field_are_nodes_too():
    """The wider scope: a thermal or EMI objective needs these, and each returns a different shape --
    `dissipation()` a dict of nodes keyed by region, `field()` one node over the probe points."""
    from jno.trace import Placeholder

    sol = _net(CU * _rho() ** 3).solve()
    q = sol.dissipation()
    assert set(q) == {"bar"}
    assert isinstance(q["bar"], Placeholder)
    b = sol.field(np.array([[0.008, 0.010, 0.0]]))
    assert isinstance(b, Placeholder)

    # ...and they must EVALUATE. Asserting the type alone passes on a node that raises the moment
    # anything asks it for a number, which is most of what a readout is for.
    v = jnp.full((_ncell(),), 0.5)
    heat = np.asarray(np.real(_value_of(q["bar"], "rho", v))).reshape(-1)
    assert heat.size == 1 and np.isfinite(heat).all() and heat[0] > 0, heat
    bb = np.asarray(np.real(_value_of(b, "rho", v)))
    assert bb.shape[-1] == 3 and np.isfinite(bb).all(), bb.shape


def test_an_explicit_sigma_override_still_solves_concretely():
    """`solve(sigma=...)` names concrete values, so it is an escape hatch back to a number -- which
    is what a convergence check or a one-off evaluation wants."""
    e = _net(CU * _rho() ** 3).build()
    sol = e.solve(sigma={"bar": CU})
    assert isinstance(complex(sol.Z), complex)


def test_a_readout_evaluated_per_point_says_so_instead_of_going_singular():
    """A FEM region coefficient is evaluated PER QUADRATURE POINT, and hands each point a scalar.

    A PEEC readout is one value for the whole region, not a field, so `d.by_region({r: node})` runs
    a network solve per point -- and what it actually produced was `Factor is exactly singular` from
    inside a preconditioner callback, with nothing naming the material or the composition. The wrong
    SHAPE is the reliable signal that this has happened, and it is worth catching precisely because
    the composition that works is a one-liner away.
    """
    node = _net(CU * _rho() ** 3).solve().R
    with pytest.raises(ValueError, match="per point"):
        _value_of(node, "rho", jnp.asarray(0.5))  # a scalar where the parameter is (n_cells,)
