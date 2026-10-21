"""Mini-batching over the collocation-point axis — ``core.solve(points=...)``.

``batchsize`` draws along the **sample** axis, the one ``n * domain`` builds. A dataset attached with
``jno.domain.from_array`` has exactly one sample, so ``batchsize`` had nothing to draw from there: it
took the ``batchsize > B`` branch, sampled *with replacement*, and silently evaluated N duplicate
copies of the same full point set at N times the cost. That is now an error, and ``points=`` does the
thing those call sites meant — sub-sample the points.

The property that needs pinning hardest is **alignment**: two tags with the same point count must be
sliced identically, or an input field and its target would drift apart while every shape still
matched, which is a wrong number rather than a crash.
"""

import foundax
import jax
import numpy as np
import optax
import pytest

import jno
from jno.trace_compiler import _requires_full_point_set

N_POINTS = 64


def _residuals(crux, *, points, seed=0):
    """Un-reduced per-point residuals — the direct view of how many points a step actually saw.

    Both operands are pinned to the default device exactly as ``core``'s own resample call site does
    (``core.py``, the ``needs_residuals`` branch). ``compiled_resample_constraints_fn`` is not jitted
    with explicit ``in_shardings``, so on a GPU the mesh-sharded model params become ARG_SHARDING
    inside the function's internal vmap and cannot be reconciled with unpinned context arrays -- the
    MLP matmul raises. Every call goes through here so no test can reintroduce a placement the
    library itself never uses, which would fail on GPU while passing on CPU.
    """
    dev = jax.devices()[0]
    out = crux.compiled_resample_constraints_fn(
        jax.device_put(crux.models, dev),
        jax.device_put(crux.domain_data.context, dev),
        batchsize=None,
        key=jax.random.PRNGKey(seed),
        points=points,
    )
    return [np.asarray(o) for o in out]


def _residual_shapes(crux, *, points, seed=0):
    return [o.shape for o in _residuals(crux, points=points, seed=seed)]


@pytest.fixture(scope="module")
def point_cloud():
    """A 64-point cloud with a net over it — the `from_array` dataset shape, B = 1."""
    x = np.random.default_rng(0).normal(size=(N_POINTS, 2)).astype(np.float32)
    dom = jno.domain.from_array({"data": x})
    d0, d1, _ = dom.variable("data")
    net = jno.nn(foundax.mlp(2, output_dim=1, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(1e-3))
    return jno.core([(net(d0, d1) - 1.0).mse]), dom, net


# ---------------------------------------------------------------------------
# Sub-sampling
# ---------------------------------------------------------------------------


class TestSubsampling:
    def test_points_reduces_the_point_axis(self, point_cloud):
        crux, _, _ = point_cloud
        assert _residual_shapes(crux, points=8) == [(1, 8, 1)]

    def test_points_none_uses_every_point(self, point_cloud):
        crux, _, _ = point_cloud
        assert _residual_shapes(crux, points=None) == [(1, N_POINTS, 1)]

    def test_a_tag_smaller_than_the_request_is_used_whole(self, point_cloud):
        # Not an error: asking for more points than a tag has is a request the tag simply satisfies.
        crux, _, _ = point_cloud
        assert _residual_shapes(crux, points=N_POINTS * 2) == [(1, N_POINTS, 1)]

    def test_same_key_draws_the_same_subset(self, point_cloud):
        crux, _, _ = point_cloud
        a = _residuals(crux, points=8, seed=3)
        b = _residuals(crux, points=8, seed=3)
        assert np.allclose(a[0], b[0])

    def test_different_keys_draw_different_subsets(self, point_cloud):
        crux, _, _ = point_cloud
        a = _residuals(crux, points=8, seed=1)
        b = _residuals(crux, points=8, seed=2)
        assert not np.allclose(a[0], b[0])


# ---------------------------------------------------------------------------
# Alignment — the property that fails as a wrong number, not a crash
# ---------------------------------------------------------------------------


class TestTagsStayAligned:
    def test_two_tags_of_equal_length_are_sliced_identically(self):
        # Two tags holding the SAME coordinates. `net(a) - net(b)` is identically zero as long as
        # both tags are sliced with the same indices; an independent draw per tag makes it nonzero
        # while every shape still matches — exactly the failure that would go unnoticed.
        x = np.random.default_rng(1).normal(size=(N_POINTS, 2)).astype(np.float32)
        dom = jno.domain.from_array({"a": x, "b": x.copy()})
        a0, a1, _ = dom.variable("a")
        b0, b1, _ = dom.variable("b")
        net = jno.nn(foundax.mlp(2, output_dim=1, key=jax.random.PRNGKey(0)))
        net.optimizer(optax.adam(1e-3))
        crux = jno.core([(net(a0, a1) - net(b0, b1)).mse])

        out = _residuals(crux, points=8)
        assert np.max(np.abs(out[0])) < 1e-6


# ---------------------------------------------------------------------------
# Noise follows the sub-sampled count
# ---------------------------------------------------------------------------


class TestNoiseFollowsPoints:
    def test_noise_is_drawn_at_the_sub_sampled_count(self):
        x = np.random.default_rng(2).normal(size=(N_POINTS, 2)).astype(np.float32)
        dom = jno.domain.from_array({"data": x})
        d0, d1, _ = dom.variable("data")
        net = jno.nn(foundax.mlp(2, output_dim=1, key=jax.random.PRNGKey(0)))
        net.optimizer(optax.adam(1e-3))
        crux = jno.core([(0.0 * net(d0, d1) + jno.noise.gaussian(std=1.0)).mse])
        assert _residual_shapes(crux, points=8) == [(1, 8, 1)]

    def test_batch_members_draw_independent_noise(self):
        # Regression guard, not a fix: `jax.random.split(key, B)` already gives each stacked sample
        # its own key. Pinned because correlated draws across the batch would be invisible in the
        # loss curve and fatal for anything that needs independent samples (flow matching).
        dom = 4 * jno.domain(constructor=jno.domain.line(mesh_size=0.25))
        x, *_ = dom.variable("interior")
        net = jno.nn(foundax.mlp(1, output_dim=1, key=jax.random.PRNGKey(0)))
        net.optimizer(optax.adam(1e-3))
        crux = jno.core([(net(x) + 0.0).mse])

        val = np.asarray(crux.eval([0.0 * net(x) + jno.noise.gaussian(std=1.0)], key=jax.random.PRNGKey(0)))
        rows = val.reshape(val.shape[0], -1)
        assert rows.shape[0] == 4
        for i in range(1, 4):
            assert not np.allclose(rows[0], rows[i])


# ---------------------------------------------------------------------------
# Fail loud
# ---------------------------------------------------------------------------


class TestGuards:
    def test_batchsize_beyond_the_sample_count_raises_and_names_points(self, point_cloud):
        # Both halves matter. Raising stops the silent duplicate-evaluation path; naming `points=` is
        # what makes the error actionable, since the caller who hits this on a `from_array` dataset
        # (B = 1) wants the POINT axis and "stack more samples" would be the wrong fix.
        crux, _, _ = point_cloud
        with pytest.raises(ValueError, match=r"exceeds the number of stacked samples"):
            crux.compiled_constraints_fn(crux.models, crux.domain_data.context, batchsize=16, key=jax.random.PRNGKey(0))
        with pytest.raises(ValueError, match=r"solve\(points=16\)"):
            crux.compiled_constraints_fn(crux.models, crux.domain_data.context, batchsize=16, key=jax.random.PRNGKey(0))

    def test_points_below_one_raises(self, point_cloud):
        crux, _, _ = point_cloud
        with pytest.raises(ValueError, match=r"points must be >= 1"):
            crux.solve(epochs=1, points=0)

    def test_finite_difference_derivative_forbids_point_batching(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        net = jno.nn(foundax.mlp(1, output_dim=1, key=jax.random.PRNGKey(0)))
        net.optimizer(optax.adam(1e-3))
        crux = jno.core([net(x).scalar.bind(x=x).d(x, scheme="finite_difference").mse])
        with pytest.raises(ValueError, match=r"finite-difference derivative"):
            crux.compiled_constraints_fn(
                crux.models, crux.domain_data.context, batchsize=None, key=jax.random.PRNGKey(0), points=4
            )

    def test_guard_recognises_a_fem_assembly(self):
        # Unit-level: the assembly branch of the guard, without paying for a FEM build.
        from jno.trace import Assembly, Literal

        assert _requires_full_point_set([Assembly(Literal(1.0), 4, support="interior", region_id=0)]) == "a FEM assembly"

    def test_guard_passes_a_plain_pointwise_residual(self, point_cloud):
        crux, _, _ = point_cloud
        assert _requires_full_point_set(crux._constraint_exprs) is None


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_solve_with_points_trains(self):
        x = np.random.default_rng(4).normal(size=(256, 2)).astype(np.float32)
        dom = jno.domain.from_array({"data": x})
        d0, d1, _ = dom.variable("data")
        net = jno.nn(foundax.mlp(2, output_dim=1, key=jax.random.PRNGKey(0)))
        net.optimizer(optax.adam(1e-2))
        crux = jno.core([(net(d0, d1) - 1.0).mse])
        stats = crux.solve(epochs=20, points=32)
        assert np.isfinite(stats.training_logs[-1]["total_loss"][-1])
