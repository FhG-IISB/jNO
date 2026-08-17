"""Common-subexpression elimination must never merge two *distinct* nodes.

``cse`` canonicalises a trace by mapping every node to a structural key and reusing the first node
that produced a given key.  Its contract is stated in its own docstring: *"The pass is safe to run
multiple times and never changes semantics."*  That holds only for the node types ``_key`` knows
about — every other type fell through to an implicit ``return None``, so **all** of them shared one
key and collapsed into whichever unlisted node the walk happened to reach first.  Across types, too::

    jno.noise.gaussian(std=1.0) - jno.noise.gaussian(std=1000.0)   ->  identically 0
    u.integrate() - v.integrate()                                  ->  identically 0
    jno.noise.gaussian() - u.integrate()                           ->  the integral became the noise

Every one of those is a silently wrong number, not an error.  The flow-matching loss is the case that
exposed it: ``x0`` (gaussian) and ``t`` (uniform) are two draws, and merging them makes the target a
deterministic function of the network input, so training "converges" to a meaningless 1e-3.

The fix is to key an unrecognised node by its **identity**, so an unknown type is merely not shared —
never merged.  These tests pin both halves: distinct nodes stay distinct, and the sharing ``cse``
actually exists for still happens.
"""

import foundax
import jax
import optax
import pytest

import jno
from jno.trace import cse


def _nodes(expr, out=None, seen=None):
    """Type names of every jNO node reachable from *expr*, each counted once."""
    out = [] if out is None else out
    seen = set() if seen is None else seen
    if id(expr) in seen:
        return out
    seen.add(id(expr))
    out.append(type(expr).__name__)
    for attr in ("left", "right", "target", "expr", "operation"):
        child = getattr(expr, attr, None)
        if "jno" in str(type(child).__module__):
            _nodes(child, out, seen)
    for attr in ("args", "options", "variables"):
        for child in getattr(expr, attr, []) or []:
            if "jno" in str(type(child).__module__):
                _nodes(child, out, seen)
    return out


@pytest.fixture(scope="module")
def two_nets():
    """A line domain and two independently-initialised networks on it."""
    dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, *_ = dom.variable("interior")
    a = jno.nn(foundax.mlp(1, output_dim=1, key=jax.random.PRNGKey(0)))
    b = jno.nn(foundax.mlp(1, output_dim=1, key=jax.random.PRNGKey(7)))
    a.optimizer(optax.adam(1e-3))
    b.optimizer(optax.adam(1e-3))
    u, v = a(x), b(x)
    return dom, u, v, jno.core([(u - v).mse])


# ---------------------------------------------------------------------------
# Distinct nodes must survive — structural
# ---------------------------------------------------------------------------


class TestDistinctNodesSurvive:
    def test_two_noise_nodes_of_different_distributions_stay_two(self):
        expr = jno.noise.gaussian(std=1.0, ndim=2) - jno.noise.uniform(low=0.0, high=1.0)
        assert _nodes(cse(expr)).count("Noise") == 2

    def test_two_noise_nodes_of_identical_parameters_stay_two(self):
        # Structurally identical and still two *independent* random variables: each carries its own
        # ``_noise_id``, hence its own realisation. Sharing them would silently correlate them.
        expr = jno.noise.gaussian(std=1.0) - jno.noise.gaussian(std=1.0)
        assert _nodes(cse(expr)).count("Noise") == 2

    def test_two_integrals_of_different_fields_stay_two(self, two_nets):
        _, u, v, _ = two_nets
        assert _nodes(cse(u.integrate() - v.integrate())).count("Integral") == 2

    def test_nodes_of_unrelated_types_do_not_merge(self, two_nets):
        # The cross-type case: both were keyed ``None``, so an Integral could be replaced by a Noise.
        _, u, _, _ = two_nets
        names = _nodes(cse(jno.noise.gaussian(std=1.0) - u.integrate()))
        assert names.count("Noise") == 1
        assert names.count("Integral") == 1

    @pytest.mark.parametrize("count", [2, 5])
    def test_many_noise_nodes_all_survive(self, count):
        expr = jno.noise.gaussian(std=1.0)
        for i in range(1, count):
            expr = expr + jno.noise.gaussian(std=float(i))
        assert _nodes(cse(expr)).count("Noise") == count


# ---------------------------------------------------------------------------
# Distinct nodes must survive — numeric oracle
# ---------------------------------------------------------------------------


class TestCollapseChangedTheAnswer:
    def test_difference_of_two_integrals_is_not_identically_zero(self, two_nets):
        _, u, v, crux = two_nets
        i_u, i_v, difference = crux.eval([u.integrate(), v.integrate(), u.integrate() - v.integrate()])
        expected = float(i_u.ravel()[0]) - float(i_v.ravel()[0])
        assert expected != pytest.approx(0.0, abs=1e-6), "fixture nets must differ for this to test anything"
        assert float(difference.ravel()[0]) == pytest.approx(expected, rel=1e-6)

    def test_difference_of_two_noise_draws_is_not_identically_zero(self, two_nets):
        _, _, _, crux = two_nets
        # Wildly different scales: if the two nodes merge, the difference is exactly 0 rather than
        # O(1000). A key= is required — without one, noise evaluates to zeros by design.
        (difference,) = crux.eval(
            [jno.noise.gaussian(std=1.0) - jno.noise.gaussian(std=1000.0)],
            key=jax.random.PRNGKey(0),
        )
        assert float(abs(difference).max()) > 1.0


# ---------------------------------------------------------------------------
# ...while the sharing cse exists for still happens
# ---------------------------------------------------------------------------


class TestSharingStillHappens:
    def test_one_node_used_twice_stays_one_node(self):
        # The flip side of the fix: a single noise node referenced twice must keep ONE realisation,
        # or ``x_t`` and the target it is regressed against would disagree.
        g = jno.noise.gaussian(std=1.0)
        assert _nodes(cse(g * 2.0 - g)).count("Noise") == 1

    def test_identical_subtrees_are_still_shared(self, two_nets):
        _, u, v, _ = two_nets
        shared = cse((u * v) + (u * v))
        assert shared.left is shared.right
