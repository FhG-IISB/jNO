"""``Model.flow`` — a network marched as an ODE velocity field, ``dx/dt = v(x, t, *cond)``.

Two routes, one computation: handed a concrete initial state the march runs eagerly and returns the
array; handed a symbolic one it returns a :class:`~jno.trace.Flow` node that composes into a
``jno.core([...])`` term list. The second is what makes a constraint on *generated* samples
expressible and what lets a gradient reach an optimisable latent through the whole march, so the two
agreeing exactly is the property to pin.

The oracle throughout is the constant linear velocity :math:`v(x, t) = A x`, whose flow over
:math:`t \\in [0, 1]` is exactly :math:`x(1) = e^{A} x_0` — which also pins the integrator's order.
"""

import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import optax
import pytest

import jno
from jno.trace import Flow, cse, dump_tree

# Rotation generator: expm(A) is a quarter turn, so the oracle is well conditioned and non-trivial.
A = jnp.array([[0.0, 1.0], [-1.0, 0.0]])


class LinearVelocity(eqx.Module):
    """``v(x, t) = x A^T`` — constant in t, so the exact flow is ``expm(A)``."""

    a: jnp.ndarray

    def __call__(self, x, t, *rest):
        return x @ self.a.T


class TreeVelocity(eqx.Module):
    """The same velocity applied leaf-wise to a dict-valued state."""

    a: jnp.ndarray

    def __call__(self, x, t, *rest):
        return {k: v @ self.a.T for k, v in x.items()}


class ConstantCondVelocity(eqx.Module):
    """``v(x, t, c) = c`` — the state after marching 0 → 1 from zero is exactly ``c``."""

    def __call__(self, x, t, c):
        return jnp.broadcast_to(c, x.shape)


@pytest.fixture(scope="module")
def linear_flow():
    return jno.nn(LinearVelocity(a=A)), jnp.eye(2)


# ---------------------------------------------------------------------------
# Accuracy — against the analytic flow
# ---------------------------------------------------------------------------


class TestAnalyticOracle:
    def test_matches_the_matrix_exponential(self, linear_flow):
        v, x0 = linear_flow
        got = v.flow(x0, time=(0.0, 1.0, 201))
        assert np.allclose(np.asarray(got), np.asarray(x0 @ jsl.expm(A).T), atol=1e-5)

    def test_is_second_order(self, linear_flow):
        """Heun is O(h²): halving the step must cut the error by ~4, not ~2."""
        v, x0 = linear_flow
        exact = x0 @ jsl.expm(A).T
        errs = [float(jnp.abs(v.flow(x0, time=(0.0, 1.0, n + 1)) - exact).max()) for n in (10, 20, 40)]
        for coarse, fine in zip(errs, errs[1:]):
            assert 3.0 < coarse / fine < 5.0, f"error ratio {coarse / fine:.2f} is not second order"


# ---------------------------------------------------------------------------
# The time grid
# ---------------------------------------------------------------------------


class TestTimeGrid:
    def test_array_and_the_equivalent_triple_agree(self, linear_flow):
        v, x0 = linear_flow
        assert np.allclose(
            np.asarray(v.flow(x0, time=(0.0, 1.0, 21))), np.asarray(v.flow(x0, time=jnp.linspace(0.0, 1.0, 21)))
        )

    def test_a_non_uniform_grid_is_marched_as_given(self, linear_flow):
        # Steps are the consecutive differences, so a grid dense at one end still lands on expm(A).
        v, x0 = linear_flow
        ts = 1.0 - (1.0 - jnp.linspace(0.0, 1.0, 201)) ** 2
        assert np.allclose(np.asarray(v.flow(x0, time=ts)), np.asarray(x0 @ jsl.expm(A).T), atol=1e-5)

    def test_a_decreasing_grid_inverts_the_march(self, linear_flow):
        # t: 1 -> 0 maps a sample back to the latent it came from — no flag, just the grid.
        v, x0 = linear_flow
        forward = v.flow(x0, time=(0.0, 1.0, 201))
        assert np.allclose(np.asarray(v.flow(forward, time=(1.0, 0.0, 201))), np.asarray(x0), atol=1e-5)

    def test_save_returns_the_whole_path_aligned_with_the_grid(self, linear_flow):
        v, x0 = linear_flow
        path = v.flow(x0, time=(0.0, 1.0, 21), save=True)
        assert path.shape == (21, 2, 2)
        assert np.allclose(np.asarray(path[0]), np.asarray(x0))  # the initial state is the first entry

    @pytest.mark.parametrize("bad", [(0.0, 1.0, 1), (0.0, 1.0, 0)])
    def test_a_grid_of_fewer_than_two_points_raises(self, linear_flow, bad):
        v, x0 = linear_flow
        with pytest.raises(ValueError, match=r"n counts grid POINTS"):
            v.flow(x0, time=bad)

    def test_a_scalar_time_raises(self, linear_flow):
        v, x0 = linear_flow
        with pytest.raises(ValueError, match=r"1-D sequence of at least 2 grid points"):
            v.flow(x0, time=1.0)


# ---------------------------------------------------------------------------
# Pytree state and conditioning
# ---------------------------------------------------------------------------


class TestStateAndConditioning:
    def test_a_pytree_state_marches_leafwise(self):
        v = jno.nn(TreeVelocity(a=A))
        x0 = jnp.eye(2)
        out = v.flow({"p": x0, "q": 2.0 * x0}, time=(0.0, 1.0, 201))
        exact = x0 @ jsl.expm(A).T
        assert sorted(out) == ["p", "q"]
        assert np.allclose(np.asarray(out["p"]), np.asarray(exact), atol=1e-5)
        assert np.allclose(np.asarray(out["q"]), np.asarray(2.0 * exact), atol=1e-5)

    def test_a_velocity_of_the_wrong_structure_raises(self):
        class Mismatched(eqx.Module):
            def __call__(self, x, t, *rest):
                return {"only": x["p"]}

        with pytest.raises(ValueError, match=r"same pytree structure"):
            jno.nn(Mismatched()).flow({"p": jnp.eye(2), "q": jnp.eye(2)}, time=(0.0, 1.0, 3))

    def test_conditioning_arguments_reach_the_network(self):
        # v(x, t, c) = c, marched 0 -> 1 from zero, lands exactly on c.
        v = jno.nn(ConstantCondVelocity())
        c = jnp.array([[0.25, -0.5]])
        out = v.flow(jnp.zeros((1, 2)), c, time=(0.0, 1.0, 11))
        assert np.allclose(np.asarray(out), np.asarray(c), atol=1e-6)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


class TestDifferentiable:
    def test_gradient_through_the_march_matches_the_analytic_jacobian(self, linear_flow):
        # d/dx0 sum(flow(x0)) = expm(A)^T @ 1, since the flow is exactly linear here.
        v, x0 = linear_flow
        got = jax.grad(lambda z: jnp.sum(v.flow(z, time=(0.0, 1.0, 201))))(x0)
        expected = jnp.ones((2, 2)) @ jsl.expm(A)
        assert np.allclose(np.asarray(got), np.asarray(expected), atol=1e-4)

    def test_gradient_reaches_the_velocity_weights(self):
        x0 = jnp.eye(2)

        def loss(module):
            return jnp.sum(jno.nn(module).flow(x0, time=(0.0, 1.0, 21)) ** 2)

        grad = eqx.filter_grad(loss)(LinearVelocity(a=A))
        assert np.all(np.isfinite(np.asarray(grad.a)))
        assert float(jnp.linalg.norm(grad.a)) > 0.0


# ---------------------------------------------------------------------------
# The symbolic node
# ---------------------------------------------------------------------------


def _point_cloud_setup(steps=6):
    x = np.random.default_rng(0).normal(size=(32, 2)).astype(np.float32)
    dom = jno.domain.from_array({"data": x})
    d0, d1, _ = dom.variable("data")
    net = jno.nn(foundax.mlp(3, output_dim=2, hidden_dims=16, num_layers=2, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(1e-3))
    node = net.flow(jno.noise.gaussian(std=1.0, ndim=2), time=(0.0, 1.0, steps))
    return net, node, jno.np.concatenate([d0, d1])


class TestSymbolicNode:
    def test_a_symbolic_start_defers_and_a_concrete_one_does_not(self, linear_flow):
        v, x0 = linear_flow
        assert isinstance(v.flow(jno.noise.gaussian(std=1.0, ndim=2), time=(0.0, 1.0, 3)), Flow)
        assert not isinstance(v.flow(x0, time=(0.0, 1.0, 3)), Flow)

    def test_eager_and_symbolic_marches_agree(self):
        # The same weights, the same grid, the two routes: they must be the same computation, not
        # merely close. The core must carry the model in a constraint for its weights to exist.
        net, node, target = _point_cloud_setup()
        crux = jno.core([(node - target).mse])

        z = jnp.asarray(np.random.default_rng(1).normal(size=(32, 2)).astype(np.float32))
        eager = np.asarray(net.flow(z, time=(0.0, 1.0, 6)))
        symbolic = np.asarray(crux.eval([net.flow(jno.np.function(lambda: z, [], "z0"), time=(0.0, 1.0, 6))]))
        assert np.array_equal(eager, symbolic.reshape(eager.shape))

    def test_the_march_trains_inside_a_constraint(self):
        net, node, target = _point_cloud_setup()
        crux = jno.core([(node - target).mse])
        stats = crux.solve(epochs=5)
        assert np.isfinite(stats.training_logs[-1]["total_loss"][-1])

    def test_two_distinct_flow_nodes_are_not_merged(self, linear_flow):
        # cse keys unrecognised nodes by identity; a Flow must not collapse onto another.
        v, _ = linear_flow
        a = v.flow(jno.noise.gaussian(std=1.0, ndim=2), time=(0.0, 1.0, 3))
        b = v.flow(jno.noise.gaussian(std=1.0, ndim=2), time=(0.0, 1.0, 3))
        merged = cse(a - b)
        assert merged.left is not merged.right


class TestEveryWalkerSeesIt:
    """A new node type is invisible to each tree walker until registered — assert, don't assume."""

    def test_the_walkers_all_reach_a_flow_node(self):
        net, node, target = _point_cloud_setup()
        crux = jno.core([(node - target).mse])

        # collect_dense_layers → the velocity model was discovered and initialised
        assert net.layer_id in crux.models
        # domain inference walked through the node
        assert crux.domain is not None
        # collect_tags reached the data variable inside the constraint
        assert "data" in crux._constraint_tags
        # dump_tree renders it
        assert "Flow" in dump_tree(node)


# ---------------------------------------------------------------------------
# End to end — conditional flow matching on a distribution with known structure
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_conditional_flow_matching_learns_an_eight_mode_ring():
    """Train the CFM objective, then sample with ``.flow``, and check the samples are the data.

    The 8-mode ring is chosen because it fails loudly in two independent ways: a collapsed velocity
    puts the radius somewhere other than 2, and a mode-dropping one leaves gaps around the circle.

    The loss assertion is the important one. Conditional flow matching has an **irreducible** loss —
    given ``x_t`` and ``t`` you cannot recover ``x1 - x0``, so a converged model plateaus at O(1). A
    loss near zero means the prior draw and the path time have stopped being independent, which is
    exactly what the ``cse`` node collapse used to do; it looked like excellent convergence and
    produced samples that diverged to infinity.
    """
    d, n = 2, 1024
    rng = np.random.default_rng(0)
    angle = 2 * np.pi * rng.integers(0, 8, n) / 8
    data = (np.stack([np.cos(angle), np.sin(angle)], -1) * 2.0 + 0.15 * rng.normal(size=(n, d))).astype(np.float32)

    dom = jno.domain.from_array({"data": data})
    d0, d1, _ = dom.variable("data")

    x1 = jno.np.concatenate([d0, d1])
    x0 = jno.noise.gaussian(std=1.0, ndim=d)  # prior draw   — independent of t
    t = jno.noise.uniform(low=0.0, high=1.0)  # path time

    v = jno.nn(
        foundax.mlp(d + 1, output_dim=d, hidden_dims=128, num_layers=3, activation=jax.nn.swish, key=jax.random.PRNGKey(0))
    )
    v.optimizer(optax.adam(3e-3))

    crux = jno.core([(v((1 - t) * x0 + t * x1, t) - (x1 - x0)).mse])
    stats = crux.solve(epochs=4000)

    final_loss = float(stats.training_logs[-1]["total_loss"][-1])
    assert final_loss > 1.0, f"loss {final_loss:.2e} is below the irreducible CFM floor — the draws are correlated"

    samples = np.asarray(v.flow(jax.random.normal(jax.random.PRNGKey(1), (2000, d)), time=(0.0, 1.0, 101)))
    radius = np.linalg.norm(samples, axis=-1)
    assert abs(radius.mean() - 2.0) < 0.2, f"generated radius {radius.mean():.3f} is not the data's 2.0"

    occupancy = np.histogram(
        (np.arctan2(samples[:, 1], samples[:, 0]) % (2 * np.pi)) / (2 * np.pi / 8), bins=8, range=(0, 8)
    )[0]
    assert (occupancy > 0).all(), f"modes dropped: {occupancy}"
