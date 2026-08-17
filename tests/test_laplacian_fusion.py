"""Tests for the Laplacian-fusion trace pass (``jno.trace.fuse_laplacian``).

``u.xx + u.yy`` and ``u.d2(x) + u.d2(y)`` are the same operator as
``jno.np.laplacian(u, [x, y])``, but they reach the compiler as separate
per-coordinate derivative nodes.  The pass folds them into the single
``Hessian(..., trace=True)`` node so all three spellings cost the same.

Covers:
  - structural rewriting of every spelling (chained ``.d``, ``.d2``, mixed),
    in 2-D and 3-D, with unrelated terms sharing the sum;
  - the cases that must NOT fuse: a repeated coordinate, subtraction, mixed
    targets, mixed schemes, temporal derivatives, finite-difference schemes,
    and weak-form trees;
  - numerical agreement of the fused and unfused forms against a hand-computed
    ``jax.hessian`` reference;
  - the compiled cost actually dropping.
"""

from __future__ import annotations

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
from jno.trace import BinaryOp, Hessian, Jacobian, Placeholder, _child_placeholders, fuse_laplacian
from tests.conftest import MockDomain

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

KEY = jax.random.PRNGKey(0)


def _coords(dim=3):
    """A mock 3-D domain plus its x/y/z coordinate Variables and a time Variable."""
    from jno.trace import Variable

    d = MockDomain(tags=["p"], dim=dim)
    d.context["__time__"] = jnp.zeros((1, 1))
    names = [Variable("p", [i, i + 1], domain=d) for i in range(dim)]
    t = Variable("__time__", [0, 1], domain=d, axis="temporal")
    return d, names, t


def _field(d):
    """A model call over the mock domain, standing in for a network output."""
    net = jno.nn(foundax.mlp(in_features=3, output_dim=1, hidden_dims=4, num_layers=2, key=KEY))
    from jno.trace import Variable

    return net(Variable("p", [0, 3], domain=d))


def _tree(expr):
    """The Placeholder tree behind an expression.

    Semantic views (``u.scalar.bind(...)``) wrap the outside of an expression;
    ``core.compile`` only ever hands the pass the underlying Placeholder (by then
    the constraint has been through ``.mse`` / ``wrap_constraints``), so unwrap
    the same way here.
    """
    return getattr(expr, "expr", expr)


def _fuse(expr):
    return fuse_laplacian(_tree(expr))


def _as_laplacian(node):
    """Return the fused node when *node* is a single Laplacian, else ``None``."""
    if isinstance(node, Hessian) and node.trace:
        return node
    return None


def _count_derivative_nodes(expr):
    """``(n_jacobian, n_laplacian)`` reachable from *expr* — the compiled cost proxy."""
    seen, jac, lap = set(), 0, 0
    stack = [_tree(expr)]
    while stack:
        node = stack.pop()
        if not isinstance(node, Placeholder) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Hessian) and node.trace:
            lap += 1
        elif isinstance(node, Jacobian):
            jac += 1
        stack.extend(_child_placeholders(node))
    return jac, lap


# ---------------------------------------------------------------------------
# structural: what fuses
# ---------------------------------------------------------------------------


def test_chained_partials_fuse_to_one_laplacian():
    d, (x, y, _), _ = _coords()
    u = _field(d).scalar.bind(x=x, y=y)

    fused = _fuse(u.xx + u.yy)

    lap = _as_laplacian(fused)
    assert lap is not None, f"expected a single Laplacian node, got {type(fused).__name__}"
    assert [v.dim[0] for v in lap.variables] == [0, 1]
    assert lap.trace is True


def test_d2_spelling_fuses():
    d, (x, y, _), _ = _coords()
    u = _field(d)

    lap = _as_laplacian(_fuse(u.d2(x) + u.d2(y)))

    assert lap is not None
    assert [v.dim[0] for v in lap.variables] == [0, 1]


def test_mixed_spellings_fuse_together():
    """``u.xx`` (chained Jacobians) and ``u.d2(y)`` (a one-variable Hessian) are
    the same kind of term and must land in the same fused node."""
    d, (x, y, _), _ = _coords()
    u = _field(d).scalar.bind(x=x, y=y)

    lap = _as_laplacian(_fuse(u.xx + u.d2(y)))

    assert lap is not None
    assert [v.dim[0] for v in lap.variables] == [0, 1]


def test_three_dimensional_laplacian_fuses():
    d, (x, y, z), _ = _coords()
    u = _field(d).scalar.bind(x=x, y=y, z=z)

    lap = _as_laplacian(_fuse(u.xx + u.yy + u.zz))

    assert lap is not None
    assert [v.dim[0] for v in lap.variables] == [0, 1, 2]


def test_unrelated_terms_in_the_sum_survive():
    """``u.xx + f + u.yy`` keeps ``f`` — only the derivative terms collapse."""
    d, (x, y, _), _ = _coords()
    u = _field(d).scalar.bind(x=x, y=y)

    fused = _fuse(u.xx + u * 0.5 + u.yy)

    assert isinstance(fused, BinaryOp) and fused.op == "+"
    terms = [fused.left, fused.right]
    assert sum(1 for t in terms if _as_laplacian(t) is not None) == 1
    assert sum(1 for t in terms if isinstance(t, BinaryOp) and t.op == "*") == 1


def test_fusion_reaches_nested_sums():
    """A Laplacian buried under a product still fuses."""
    d, (x, y, _), _ = _coords()
    u = _field(d).scalar.bind(x=x, y=y)

    fused = _fuse(0.1 * (u.xx + u.yy))

    assert isinstance(fused, BinaryOp) and fused.op == "*"
    assert _as_laplacian(fused.left) is not None or _as_laplacian(fused.right) is not None


def test_pass_is_idempotent():
    d, (x, y, _), _ = _coords()
    u = _field(d).scalar.bind(x=x, y=y)

    once = _fuse(u.xx + u.yy)
    twice = fuse_laplacian(once)

    assert twice is once


# ---------------------------------------------------------------------------
# structural: what must NOT fuse
# ---------------------------------------------------------------------------


def test_repeated_coordinate_does_not_fuse():
    """``u.xx + u.xx`` is 2 ∂²u/∂x², not a Laplacian."""
    d, (x, y, _), _ = _coords()
    u = _field(d).scalar.bind(x=x, y=y)

    fused = _fuse(u.xx + u.xx)

    assert _as_laplacian(fused) is None
    assert isinstance(fused, BinaryOp) and fused.op == "+"


def test_subtraction_does_not_fuse():
    d, (x, y, _), _ = _coords()
    u = _field(d).scalar.bind(x=x, y=y)

    fused = _fuse(u.xx - u.yy)

    assert _as_laplacian(fused) is None


def test_different_targets_do_not_fuse():
    d, (x, y, _), _ = _coords()
    u = _field(d).scalar.bind(x=x, y=y)
    v = _field(d).scalar.bind(x=x, y=y)

    fused = _fuse(u.xx + v.yy)

    assert _as_laplacian(fused) is None


def test_mixed_ad_modes_do_not_fuse():
    """Different per-call AD modes are a deliberate user choice; leave them be."""
    d, (x, y, _), _ = _coords()
    u = _field(d)

    fused = _fuse(u.d2(x, scheme="automatic_differentiation:forward") + u.d2(y, scheme="automatic_differentiation:reverse"))

    assert _as_laplacian(fused) is None


def test_matching_ad_mode_is_carried_onto_the_fused_node():
    d, (x, y, _), _ = _coords()
    u = _field(d)

    lap = _as_laplacian(
        _fuse(u.d2(x, scheme="automatic_differentiation:forward") + u.d2(y, scheme="automatic_differentiation:forward"))
    )

    assert lap is not None
    assert lap.scheme == "automatic_differentiation:forward"


def test_finite_difference_does_not_fuse():
    """``finite_difference:cotangent`` returns the whole Laplacian for any requested
    dimension, so folding two such nodes would change the numbers."""
    d, (x, y, _), _ = _coords()
    u = _field(d)

    for scheme in ("finite_difference", "finite_difference:cotangent", "finite_difference:lsq"):
        fused = _fuse(u.d2(x, scheme=scheme) + u.d2(y, scheme=scheme))
        assert _as_laplacian(fused) is None, scheme


def test_temporal_derivative_does_not_fuse():
    """``u.tt`` evaluates through the temporal path, which indexes time rather than a
    column of the point array — it is not a Hessian diagonal entry."""
    d, (x, _, _), t = _coords()
    u = _field(d).scalar.bind(x=x, t=t)

    fused = _fuse(u.tt + u.xx)

    assert _as_laplacian(fused) is None


def test_weak_form_tree_is_left_alone():
    """FEM trees are lowered by pattern in the variational route; the pass skips them."""
    dom = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
    u, phi = dom.fem_symbols("u")
    x, y, _ = dom.variable("interior")
    weak = jno.np.inner(u.d(x), phi.d(x)) + jno.np.inner(u.d(y), phi.d(y))

    assert fuse_laplacian(_tree(weak)) is _tree(weak)


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------


def _loss_for(build_lap):
    dom = jno.Shape.rect(0, 0, 1, 1, size=0.2).domain()
    x, y, _ = dom.variable("interior")
    net = jno.nn(foundax.mlp(in_features=2, output_dim=1, hidden_dims=8, num_layers=2, key=KEY))
    net.optimizer(optax.adam(1e-9))  # effectively frozen: epoch-0 loss is the pre-update value
    u = net(x, y).scalar.bind(x=x, y=y)
    stats = jno.core([build_lap(u, x, y).mse]).solve(1)
    return float(np.asarray(stats.total_loss_history).reshape(-1)[0])


def _reference_laplacian_mse():
    dom = jno.Shape.rect(0, 0, 1, 1, size=0.2).domain()
    pts = jnp.asarray(dom.mesh_connectivity["points"])[:, :2]
    net = foundax.mlp(in_features=2, output_dim=1, hidden_dims=8, num_layers=2, key=KEY)
    H = jax.vmap(jax.hessian(lambda p: net(p[None, :])[0, 0]))(pts)
    return float(jnp.mean((H[:, 0, 0] + H[:, 1, 1]) ** 2))


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param(lambda u, x, y: jno.np.laplacian(u, [x, y]), id="laplacian"),
        pytest.param(lambda u, x, y: u.xx + u.yy, id="xx+yy"),
        pytest.param(lambda u, x, y: u.yy + u.xx, id="yy+xx"),
        pytest.param(lambda u, x, y: u.d2(x) + u.d2(y), id="d2"),
        pytest.param(lambda u, x, y: u.xx + u.d2(y), id="mixed"),
    ],
)
def test_every_spelling_matches_the_analytic_laplacian(spelling):
    assert _loss_for(spelling) == pytest.approx(_reference_laplacian_mse(), rel=1e-4)


def test_non_laplacian_sums_keep_their_own_meaning():
    """The guard cases must still compute what the user wrote, not a Laplacian."""
    dom = jno.Shape.rect(0, 0, 1, 1, size=0.2).domain()
    pts = jnp.asarray(dom.mesh_connectivity["points"])[:, :2]
    net = foundax.mlp(in_features=2, output_dim=1, hidden_dims=8, num_layers=2, key=KEY)
    H = jax.vmap(jax.hessian(lambda p: net(p[None, :])[0, 0]))(pts)

    twice_xx = float(jnp.mean((2 * H[:, 0, 0]) ** 2))
    difference = float(jnp.mean((H[:, 0, 0] - H[:, 1, 1]) ** 2))

    assert _loss_for(lambda u, x, y: u.xx + u.xx) == pytest.approx(twice_xx, rel=1e-3)
    assert _loss_for(lambda u, x, y: u.xx - u.yy) == pytest.approx(difference, rel=1e-3)


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------


def _compiled_derivative_nodes(build_lap):
    """Derivative-node counts in the constraint tree ``core`` actually compiles."""
    dom = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
    x, y, _ = dom.variable("interior")
    net = jno.nn(foundax.mlp(in_features=2, output_dim=1, hidden_dims=4, num_layers=2, key=KEY))
    net.optimizer(optax.adam(1e-3))
    u = net(x, y).scalar.bind(x=x, y=y)
    crux = jno.core([build_lap(u, x, y).mse])
    return _count_derivative_nodes(crux._constraint_exprs[0])


def test_every_spelling_compiles_to_one_laplacian_node():
    """The whole point: each spelling must reach the evaluator as one Laplacian.

    ``u.xx + u.yy`` used to compile to four chained ``Jacobian`` nodes (~1.27x the
    FLOPs of the explicit form) and ``u.d2(x) + u.d2(y)`` to two ``Hessian`` nodes
    (~1.5x); all three now lower to the single node.
    """
    explicit = _compiled_derivative_nodes(lambda u, x, y: jno.np.laplacian(u, [x, y]))
    assert explicit == (0, 1)
    assert _compiled_derivative_nodes(lambda u, x, y: u.xx + u.yy) == explicit
    assert _compiled_derivative_nodes(lambda u, x, y: u.d2(x) + u.d2(y)) == explicit
    assert _compiled_derivative_nodes(lambda u, x, y: u.xx + u.d2(y)) == explicit


def test_non_fusable_sum_keeps_its_separate_nodes():
    """The guard cases must keep every node they were written with."""
    assert _compiled_derivative_nodes(lambda u, x, y: u.xx - u.yy) == (4, 0)
    assert _compiled_derivative_nodes(lambda u, x, y: u.d2(x) - u.d2(y)) == (0, 2)


class TestSpectralFusion:
    """A spectral Laplacian folds too — and it is where the fold matters most.

    Fused, `u.xx + u.yy` is one multiply by -(kx^2 + ky^2) off ONE forward transform. Unfused it is
    a transform pair per axis. It also keeps the documented house spelling on the fast path, rather
    than making `u.laplacian(x, y)` the only efficient way to write a Laplacian.
    """

    def test_spectral_fuses(self):
        d, (x, y, _), _ = _coords()
        u = _field(d)
        fused = _fuse(u.d2(x, scheme="spectral") + u.d2(y, scheme="spectral"))
        assert _as_laplacian(fused) is not None

    def test_the_fused_node_keeps_the_spectral_scheme(self):
        d, (x, y, _), _ = _coords()
        u = _field(d)
        lap = _as_laplacian(_fuse(u.d2(x, scheme="spectral") + u.d2(y, scheme="spectral")))
        assert str(lap.scheme).startswith("spectral")

    def test_the_cosine_submethod_survives_the_fold(self):
        d, (x, y, _), _ = _coords()
        u = _field(d)
        lap = _as_laplacian(_fuse(u.d2(x, scheme="spectral:cosine") + u.d2(y, scheme="spectral:cosine")))
        assert lap is not None and lap.scheme == "spectral:cosine"

    def test_spectral_never_fuses_with_another_family(self):
        """The grouping key includes the scheme, so a mixed pair cannot fold — asserted, not assumed."""
        d, (x, y, _), _ = _coords()
        u = _field(d)
        for other in ("automatic_differentiation", "finite_difference"):
            mixed = _fuse(u.d2(x, scheme="spectral") + u.d2(y, scheme=other))
            assert _as_laplacian(mixed) is None, f"spectral must not fuse with {other}"

    def test_mismatched_spectral_submethods_do_not_fuse(self):
        d, (x, y, _), _ = _coords()
        u = _field(d)
        mixed = _fuse(u.d2(x, scheme="spectral") + u.d2(y, scheme="spectral:cosine"))
        assert _as_laplacian(mixed) is None

    def test_finite_difference_still_refuses(self):
        """Unchanged: :cotangent returns the whole Laplacian, so folding two would double it."""
        d, (x, y, _), _ = _coords()
        u = _field(d)
        for scheme in ("finite_difference", "finite_difference:cotangent"):
            assert _as_laplacian(_fuse(u.d2(x, scheme=scheme) + u.d2(y, scheme=scheme))) is None
