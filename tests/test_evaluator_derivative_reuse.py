"""Tests for work the derivative handlers should do once instead of once per node.

Covers:
  - the finite-difference paths sharing one evaluation of the target on the mesh
    across the derivative nodes of a residual (``laplacian(u) + u.d(x) + u`` used
    to trace the network once per node);
  - the ``(T, N, D)`` Laplacian/Hessian branch being linear in the point count
    rather than quadratic, and still agreeing with ``jax.hessian``;
  - the functional operators rejecting coordinates passed positionally, instead
    of failing later inside the compiler with an unrelated error.
"""

from __future__ import annotations

import foundax
import jax
import numpy as np
import optax
import pytest

import jno
from jno.trace_evaluator import TraceEvaluator

KEY = jax.random.PRNGKey(0)
FD = "finite_difference"


# ---------------------------------------------------------------------------
# finite-difference: one evaluation of the target per (target, mesh)
# ---------------------------------------------------------------------------


def _network_dispatches(build_residual, monkeypatch):
    """How many times the evaluator traces a model forward while compiling."""
    calls = {"n": 0}
    original = TraceEvaluator._eval_flax_module_call

    def counting(self, expr, ctx):
        calls["n"] += 1
        return original(self, expr, ctx)

    monkeypatch.setattr(TraceEvaluator, "_eval_flax_module_call", counting)

    dom = jno.shape.rect(0, 0, 1, 1, size=0.4).domain()
    x, y, _ = dom.variable("interior")
    net = jno.nn(foundax.mlp(in_features=2, output_dim=1, hidden_dims=4, num_layers=2, key=KEY))
    net.optimizer(optax.adam(1e-3))
    u = net(x, y)
    jno.core([build_residual(u, x, y).mse]).solve(1)
    return calls["n"]


def test_sibling_fd_derivatives_share_one_mesh_evaluation(monkeypatch):
    """Two FD partials over the same field cost the same forwards as one."""
    one = _network_dispatches(lambda u, x, y: u.d(x, scheme=FD) - 1.0, monkeypatch)
    two = _network_dispatches(lambda u, x, y: u.d(x, scheme=FD) + u.d(y, scheme=FD) - 1.0, monkeypatch)
    assert two == one


def test_fd_laplacian_and_partial_share_one_mesh_evaluation(monkeypatch):
    """A Laplacian and a gradient over the same field share the sampled field too."""
    lap_only = _network_dispatches(lambda u, x, y: jno.np.laplacian(u, [x, y], scheme=FD) - 1.0, monkeypatch)
    both = _network_dispatches(
        lambda u, x, y: jno.np.laplacian(u, [x, y], scheme=FD) + u.d(x, scheme=FD) - 1.0, monkeypatch
    )
    assert both == lap_only


def test_fd_derivatives_are_unchanged_by_the_sharing():
    """Reuse must not move the numbers: FD partials of a linear field are exact."""
    dom = jno.shape.rect(0, 0, 1, 1, size=0.2).domain()
    x, y, _ = dom.variable("interior")
    field = 3.0 * x + 5.0 * y

    dx, dy = jno.core([], domain=dom).eval([field.d(x, scheme=FD), field.d(y, scheme=FD)], domain=dom)

    assert np.allclose(np.asarray(dx), 3.0, atol=1e-4)
    assert np.allclose(np.asarray(dy), 5.0, atol=1e-4)


# ---------------------------------------------------------------------------
# the (T, N, D) branch: linear, and correct
# ---------------------------------------------------------------------------


def _windowed_laplacian(n_points, n_time=2):
    """Drive ``_eval_hessian`` with a 3-D ``(T, N, D)`` point context."""
    dom = jno.shape.rect(0, 0, 1, 1, size=0.4).domain()
    x, y, _ = dom.variable("interior")
    net = jno.nn(foundax.mlp(in_features=2, output_dim=1, hidden_dims=8, num_layers=2, key=KEY))
    lap = jno.np.laplacian(net(x, y), [x, y])
    evaluator = TraceEvaluator({net.layer_id: net.module})
    points = jax.random.uniform(jax.random.PRNGKey(1), (n_time, n_points, 2))
    fn = jax.jit(lambda p: evaluator.evaluate(lap, {x.tag: p}, {}, None))
    return fn, points, net.module


def _flops(fn, points):
    analysis = fn.lower(points).compile().cost_analysis()
    if isinstance(analysis, (list, tuple)):
        analysis = analysis[0]
    return analysis.get("flops", 0.0)


def test_windowed_laplacian_is_linear_in_the_point_count():
    """Doubling the points must double the work, not quadruple it.

    The branch used to rebuild the whole ``(T, N, D)`` array per point and
    evaluate the target over all of it to keep one scalar — O(N²).
    """
    costs = []
    for n in (8, 16, 32):
        fn, points, _ = _windowed_laplacian(n)
        costs.append(_flops(fn, points))

    for cheap, dear in zip(costs, costs[1:]):
        assert dear / cheap == pytest.approx(2.0, rel=0.15), f"{costs} is not linear growth"


def test_windowed_laplacian_matches_jax_hessian():
    fn, points, module = _windowed_laplacian(8)

    got = np.asarray(fn(points))

    hess = jax.vmap(jax.vmap(jax.hessian(lambda p: module(p[None, :])[0, 0])))(points)
    expected = np.asarray(hess[..., 0, 0] + hess[..., 1, 1])[..., None]
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# automatic differentiation: the derivatives over one point set share their passes
# ---------------------------------------------------------------------------

RE = 40.0


def _three_field_net(width=16, size=0.25):
    dom = jno.shape.rect(0, 0, 1, 1, size=size).domain()
    x, y, _ = dom.variable("interior")
    net = jno.nn(foundax.mlp(in_features=2, output_dim=3, hidden_dims=width, num_layers=2, key=KEY))
    net.optimizer(optax.adam(1e-3))
    o = net(x, y)
    return dom, x, y, net, o, (o[..., 0], o[..., 1], o[..., 2])


def _navier_stokes(u, v, p, x, y):
    """The steady incompressible residuals, summed into one expression."""
    mom_x = u * u.d(x) + v * u.d(y) + p.d(x) - u.laplacian(x, y) / RE
    mom_y = u * v.d(x) + v * v.d(y) + p.d(y) - v.laplacian(x, y) / RE
    return mom_x + mom_y + (u.d(x) + v.d(y))


def _optimal_navier_stokes(module):
    """The same residual with the fewest AD passes: one forward-over-forward pass per coordinate gives the
    first AND second derivative of all three outputs along it."""
    eye = np.eye(2, dtype=np.float32)

    def residual(z):
        f = lambda q: module(q[None, :])[0]  # noqa: E731
        (d1x, d2x), (d1y, d2y) = (jax.jvp(lambda w, e=e: jax.jvp(f, (w,), (e,))[1], (z,), (e,)) for e in eye)
        u, v, _ = f(z)
        mom_x = u * d1x[0] + v * d1y[0] + d1x[2] - (d2x[0] + d2y[0]) / RE
        mom_y = u * d1x[1] + v * d1y[1] + d1y[2] - (d2x[1] + d2y[1]) / RE
        return mom_x + mom_y + (d1x[0] + d1y[1])

    return jax.jit(jax.vmap(residual))


def test_ad_derivatives_over_one_point_set_cost_what_a_hand_written_residual_costs():
    """Eight derivative nodes over three outputs of one network: each used to re-run the network under its
    own reverse pass (3.3x the work of the hand-written step on a Navier-Stokes PINN). Sharing one
    forward-over-forward pass per coordinate brings it to the hand-written cost."""
    _, x, y, net, _, (u, v, p) = _three_field_net()
    expr = _navier_stokes(u, v, p, x, y)
    evaluator = TraceEvaluator({net.layer_id: net.module})
    points = jax.random.uniform(jax.random.PRNGKey(1), (64, 2))
    jno_flops = _flops(jax.jit(lambda q: evaluator.evaluate(expr, {x.tag: q}, {}, None)), points)
    optimal = _flops(_optimal_navier_stokes(net.module), points)

    got = np.asarray(jax.jit(lambda q: evaluator.evaluate(expr, {x.tag: q}, {}, None))(points)).reshape(-1)
    want = np.asarray(_optimal_navier_stokes(net.module)(points)).reshape(-1)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-5)
    # 1.25: room for the element-wise arithmetic of the residual, which jNO does on (N, 1) columns
    assert jno_flops <= 1.25 * optimal, f"{jno_flops:.3g} FLOPs vs {optimal:.3g} hand-written"


def test_a_node_read_several_times_is_evaluated_once(monkeypatch):
    """``u`` in every term of a residual, and the outputs of one network call, used to be traced once per
    reference -- seven forward passes of the network for two distinct calls in a Navier-Stokes loss."""
    once = _network_dispatches(lambda u, x, y: u - 1.0, monkeypatch)
    often = _network_dispatches(lambda u, x, y: u * u + 3.0 * u - jno.np.sin(u) - 1.0, monkeypatch)
    assert often == once


def test_shared_ad_derivatives_match_jax():
    """Every derivative kind the shared pass serves, against JAX on the bare network."""
    dom, x, y, net, o, (u, v, p) = _three_field_net()
    exprs = [u.d(x), v.d(y), p.d(x), o.d(x), u.laplacian(x, y), jno.np.hessian(u, [x, y]), u.d(x).d(y), v.d(y).d(y)]
    crux = jno.core([_navier_stokes(u, v, p, x, y).mse], domain=dom)
    got = [np.asarray(g)[0] for g in crux.eval(exprs)]

    pts = np.asarray(crux.domain_data.context[x.tag]).reshape(-1, 2)
    f = lambda q: net.module(q[None, :])[0]  # noqa: E731
    jac = np.asarray(jax.vmap(jax.jacrev(f))(pts))  # (N, 3, 2)
    hess = np.asarray(jax.vmap(jax.hessian(f))(pts))  # (N, 3, 2, 2)
    want = [
        jac[:, 0, 0:1],
        jac[:, 1, 1:2],
        jac[:, 2, 0:1],
        jac[:, :, 0],
        hess[:, 0, 0, 0:1] + hess[:, 0, 1, 1:2],
        hess[:, 0],
        hess[:, 0, 0, 1:2],
        hess[:, 1, 1, 1:2],
    ]
    for g, w, e in zip(got, want, exprs):
        assert g.shape == w.shape, f"{e}: {g.shape} != {w.shape}"
        assert np.allclose(g, w, rtol=1e-4, atol=1e-5), f"{e}: max error {np.abs(g - w).max():.3g}"


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------


def test_positional_coordinates_raise_a_clear_error():
    """``laplacian(u, x, y)`` puts ``y`` in the ``scheme`` slot."""
    dom = jno.shape.rect(0, 0, 1, 1, size=0.4).domain()
    x, y, _ = dom.variable("interior")
    net = jno.nn(foundax.mlp(in_features=2, output_dim=1, hidden_dims=4, num_layers=2, key=KEY))
    u = net(x, y)

    with pytest.raises(TypeError, match="scheme must be a string"):
        jno.np.laplacian(u, x, y)

    with pytest.raises(TypeError, match=r"laplacian\(u, \[x, y\]\)"):
        jno.np.hessian(u, x, y)


def test_the_list_form_and_the_method_form_still_work():
    dom = jno.shape.rect(0, 0, 1, 1, size=0.4).domain()
    x, y, _ = dom.variable("interior")
    net = jno.nn(foundax.mlp(in_features=2, output_dim=1, hidden_dims=4, num_layers=2, key=KEY))
    u = net(x, y)

    assert jno.np.laplacian(u, [x, y]).trace is True
    assert u.laplacian(x, y).trace is True


PER_NODE = "automatic_differentiation:reverse"  # an explicit submethod keeps the per-node path: the oracle


def _per_node_vs_shared(build):
    """The same derivatives through the shared pass and through the per-node path, evaluated together."""
    dom, exprs_shared, exprs_per_node, net = build()
    net.optimizer(optax.adam(1e-3))
    crux = jno.core([(e - 0.0).mse for e in exprs_shared], domain=dom)
    got = [np.asarray(v) for v in crux.eval(exprs_shared)]
    want = [np.asarray(v) for v in crux.eval(exprs_per_node)]
    return got, want


def test_time_derivatives_join_the_shared_pass_and_match_the_per_node_path():
    """``u.t`` (time as one more coordinate, one value for all points) and ``u.tt`` with the Laplacian."""

    def build():
        dom = jno.shape.rect(0, 0, 1, 1, size=0.25).domain(time=(0, 1.0, 4))
        x, y, t = dom.variable("interior")
        net = jno.nn(foundax.mlp(in_features=3, output_dim=1, hidden_dims=16, num_layers=2, key=KEY))
        u = (net(x, y, t) * x * (1 - x)).scalar.bind(x=x, y=y, t=t)
        shared = [u.t, u.tt, u.laplacian(x, y), u.d(t).d(x)]
        per_node = [
            u.d(t, scheme=PER_NODE),
            u.d(t, scheme=PER_NODE).d(t, scheme=PER_NODE),
            u.laplacian(x, y, scheme="automatic_differentiation:fwd-over-rev"),
            u.d(t, scheme=PER_NODE).d(x, scheme=PER_NODE),
        ]
        return dom, shared, per_node, net

    got, want = _per_node_vs_shared(build)
    for g, w in zip(got, want):
        assert g.shape == w.shape
        assert np.allclose(g, w, rtol=1e-4, atol=1e-5), np.abs(g - w).max()


def test_a_laplacian_of_a_laplacian_matches_the_per_node_path():
    def build():
        dom, x, y, net, o, (u, _, _) = _three_field_net()
        scheme = "automatic_differentiation:fwd-over-rev"
        return (
            dom,
            [u.laplacian(x, y).laplacian(x, y)],
            [u.laplacian(x, y, scheme=scheme).laplacian(x, y, scheme=scheme)],
            net,
        )

    (got,), (want,) = _per_node_vs_shared(build)
    assert np.allclose(got, want, rtol=1e-3, atol=1e-4), np.abs(got - want).max()


def test_an_ad_derivative_of_a_target_that_couples_points_raises():
    """``u - mean(u)`` at one point depends on every point; its per-point AD derivative used to be returned
    silently as if it did not. The coupling is measured when the problem is compiled."""
    dom, x, y, net, _, (u, _, _) = _three_field_net()
    net.optimizer(optax.adam(1e-3))
    coupled = u - jno.np.mean(u, axis=0)
    with pytest.raises(ValueError, match="OTHER points"):
        jno.core([coupled.d(x).mse], domain=dom).solve(1)


def test_a_differentiated_field_is_not_evaluated_again_for_its_value(monkeypatch):
    """``u`` in ``u * u.d(x)`` comes from the derivative pass (the target is measured pointwise), not from a
    second, batched evaluation of the network."""
    deriv = _network_dispatches(lambda u, x, y: u.d(x) - 1.0, monkeypatch)
    both = _network_dispatches(lambda u, x, y: u * u.d(x) - 1.0, monkeypatch)
    assert both == deriv


# ---------------------------------------------------------------------------
# indexing a per-point field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pick",
    [
        lambda o, u, x: o[0],
        lambda o, u, x: o[1, 0],
        lambda o, u, x: u[0],
        lambda o, u, x: u.d(x)[0],
        lambda o, u, x: (2.0 * o + x)[-1],
        lambda o, u, x: x[0],
    ],
)
def test_an_integer_index_on_a_per_point_field_raises(pick):
    """``o[0]`` used to pick collocation point 0 silently, where ``o[..., 0]`` (the first output) was meant."""
    dom, x, _, _, o, (u, _, _) = _three_field_net()
    with pytest.raises(IndexError, match=r"\[\.\.\., "):
        jno.core([(pick(o, u, x) - 1.0).mse], domain=dom)


def test_component_and_parameter_indexing_still_work():
    dom, x, _, _, o, _ = _three_field_net()
    o[..., 0], o[..., -1], o[:, 0]  # components, and an explicit slice of the point axis
    p = jno.np.parameter((3,), key=KEY)
    first = p[0] + 0.0 * x  # a parameter vector is not a point set: its [0] is its first entry
    crux = jno.core([first.mse], domain=dom)
    assert np.allclose(np.asarray(crux.eval(first)), np.asarray(crux.eval(p[..., 0] + 0.0 * x)))
