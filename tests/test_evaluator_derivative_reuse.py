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

    dom = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
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
    dom = jno.Shape.rect(0, 0, 1, 1, size=0.2).domain()
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
    dom = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
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
# argument validation
# ---------------------------------------------------------------------------


def test_positional_coordinates_raise_a_clear_error():
    """``laplacian(u, x, y)`` puts ``y`` in the ``scheme`` slot."""
    dom = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
    x, y, _ = dom.variable("interior")
    net = jno.nn(foundax.mlp(in_features=2, output_dim=1, hidden_dims=4, num_layers=2, key=KEY))
    u = net(x, y)

    with pytest.raises(TypeError, match="scheme must be a string"):
        jno.np.laplacian(u, x, y)

    with pytest.raises(TypeError, match=r"laplacian\(u, \[x, y\]\)"):
        jno.np.hessian(u, x, y)


def test_the_list_form_and_the_method_form_still_work():
    dom = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
    x, y, _ = dom.variable("interior")
    net = jno.nn(foundax.mlp(in_features=2, output_dim=1, hidden_dims=4, num_layers=2, key=KEY))
    u = net(x, y)

    assert jno.np.laplacian(u, [x, y]).trace is True
    assert u.laplacian(x, y).trace is True
