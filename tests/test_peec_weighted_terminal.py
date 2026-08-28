"""A weighted terminal: a prescribed current DISTRIBUTION over a pad, not a short across it.

Unweighted, a terminal is an equipotential node SET, and which nodes are in the set is a step
function of where the pad is. That makes a pad's POSITION useless as a design variable: sliding a
die a quarter of a millimetre changes the answer by exactly nothing, and then by 8 % when a node
crosses the boundary. Measured on a paralleled-die module: `d|I0| = +0.000 A` twice over, then
`-6.503 A`.

Weighted, the support is a frozen superset covering the travel and the weights are smooth in the
position, so the gradient exists -- the same structure-frozen, values-traced split as `sigma`.

The k-1 equipotential ties become k-1 injection-ratio rows, `w_r (A I)_i = w_i (A I)_r`, and the
port row takes the weighted-average potential. Same row count, same unknowns.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

SIG = 5.8e7
LX, LY, TZ, H = 0.040, 0.006, 0.001, 0.001


def bar():
    d = jno.Shape.box(0, 0, 0, LX, LY, TZ, size=(H, H, TZ)).attach(sigma=SIG).name("bar").domain()
    d.tag("A", lambda x, y, z: x < 1.1 * H)
    d.tag("B", lambda x, y, z: x > LX - 6 * H)  # a WIDE pad, so weighting has room to matter
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(6, None))[:3]  # noqa: E731
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=0.0).build()


# --- it reduces to the unweighted terminal where it should ------------------------------------------


def test_uniform_weights_are_a_uniform_injection_not_the_short():
    """Deliberately NOT the same as the unweighted terminal: a short is not an equal split.

    Both are legitimate models of a pad; they differ, and the weighted one must not silently claim
    to be the other.
    """
    built = bar()
    n = len(built.nodes["B"])
    short = complex(built.solve().Z)
    equal = complex(built.solve(weights={"B": jnp.full(n, 1.0 / n)}).Z)
    assert abs(equal - short) / abs(short) > 1e-6


def test_the_answer_does_not_depend_on_the_weights_scale():
    """The ratio rows are homogeneous, so only the weights' SHAPE can matter, never their size."""
    built = bar()
    n = len(built.nodes["B"])
    w = jnp.exp(-(jnp.linspace(0.0, 2.0, n) ** 2))
    a = complex(built.solve(weights={"B": w}).Z)
    for scale in (1e-6, 1.0, 1e6):
        assert complex(built.solve(weights={"B": w * scale}).Z) == pytest.approx(a, rel=1e-9)


def test_all_the_weight_on_one_node_is_that_single_node_terminal():
    """A limit worth pinning: a spike is the same as tagging that node alone."""
    built = bar()
    ids = np.asarray(built.nodes["B"])
    n = len(ids)
    spike = jnp.zeros(n).at[3].set(1.0)
    wide = jnp.full(n, 1.0 / n)
    assert abs(complex(built.solve(weights={"B": spike}).Z)) > abs(complex(built.solve(weights={"B": wide}).Z))


# --- the reason it exists: position becomes differentiable -------------------------------------------


def test_a_sliding_weight_is_smooth_and_differentiable():
    """The whole point. A Gaussian centred at `xc` slides across a FROZEN support."""
    built = bar()
    xs = np.asarray(built.fil.nodes)[np.asarray(built.nodes["B"]), 0]
    wide = 2.0 * H

    def port(xc):
        w = jnp.exp(-(((jnp.asarray(xs) - xc) / wide) ** 2))
        return jnp.real(built.solve(weights={"B": w / jnp.sum(w)}).Z)

    lo, hi = float(xs.min()), float(xs.max())
    grid = np.linspace(lo, hi, 9)
    vals = np.array([float(port(x)) for x in grid])
    assert np.all(np.isfinite(vals))
    # smooth: no step is more than 3x the median step (a node crossing a hard pad boundary is ~100x)
    steps = np.abs(np.diff(vals))
    assert steps.max() < 3.0 * np.median(steps)

    x0 = 0.5 * (lo + hi)
    g, h = float(jax.grad(port)(x0)), 1e-7
    fd = float((port(x0 + h) - port(x0 - h)) / (2 * h))
    assert g == pytest.approx(fd, rel=1e-4)
    assert abs(g) > 0  # and NOT the exactly-zero gradient a hard pad gives between node crossings


def test_the_far_end_of_a_wide_support_does_not_break_the_rows():
    """A placement weight underflows at the far end of a support wide enough to travel across.

    Anchoring the ratio rows on the FIRST node then wrote k-1 copies of the same statement and the
    block went rank deficient -- a solve that would not converge. They anchor on the largest weight.
    """
    built = bar()
    xs = np.asarray(built.fil.nodes)[np.asarray(built.nodes["B"]), 0]
    w = jnp.exp(-(((jnp.asarray(xs) - float(xs.max())) / (0.3 * H)) ** 2))  # underflows at the near end
    assert float(jnp.min(w)) < 1e-12
    assert np.isfinite(float(jnp.real(built.solve(weights={"B": w}).Z)))


# --- and it fails loudly ------------------------------------------------------------------------------


def test_the_wrong_number_of_weights_is_refused():
    built = bar()
    n = len(built.nodes["B"])
    with pytest.raises(ValueError, match=rf"{n - 1} weights for {n} nodes"):
        built.solve(weights={"B": jnp.ones(n - 1)})


def test_an_unknown_terminal_is_refused():
    built = bar()
    with pytest.raises(ValueError, match="names no terminal of this network"):
        built.solve(weights={"Q": jnp.ones(3)})
