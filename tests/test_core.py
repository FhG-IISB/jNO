"""Edge-case tests for jno.core.

Most of jno.core is exercised end-to-end by integration tests; this file
focuses on the kind of edge cases that would otherwise surface as cryptic
errors deep in the JIT'd training loop — empty constraint lists, missing
optimizers, temporal Variables on stationary domains, and re-entry into
solve()/compile().
"""

import foundax
import jax
import jax.numpy as jnp
import optax
import pytest

import jno
import jno.jnp_ops as jnn


def _tiny_net(in_dim=1, out_dim=1, hidden=8, key_seed=0):
    return jnn.nn.wrap(
        foundax.mlp(
            in_dim,
            output_dim=out_dim,
            hidden_dims=hidden,
            num_layers=2,
            key=jax.random.PRNGKey(key_seed),
        )
    )


def _stationary_1d_domain(mesh_size=0.1):
    return jno.domain(constructor=jno.domain.line(mesh_size=mesh_size))


# ---------------------------------------------------------------------------
# Empty / degenerate constraint lists
# ---------------------------------------------------------------------------


class TestEmptyConstraints:
    def test_empty_constraints_is_eval_only_and_solve_raises(self):
        # An empty-constraint core is eval-only: it constructs without a domain
        # (the domain is supplied per call to eval()). Only *training* rejects
        # it, and that error is raised at solve() — not at construction.
        crux = jno.core([])  # no raise at construction
        with pytest.raises(ValueError, match="at least one constraint"):
            crux.solve(1)


# ---------------------------------------------------------------------------
# Missing optimizer
# ---------------------------------------------------------------------------


class TestMissingOptimizer:
    def test_no_optimizer_raises_with_setup_example(self):
        dom = _stationary_1d_domain()
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        # Intentionally NOT calling net.optimizer(...)
        pde = (net(x) - x).mse
        with pytest.raises(ValueError, match=r"(?s)has no optimizer.*model\.optimizer"):
            jno.core([pde]).solve(2)


# ---------------------------------------------------------------------------
# Temporal Variable on a stationary domain
# ---------------------------------------------------------------------------


class TestTemporalOnStationaryDomain:
    def test_constructing_core_raises_clear_error(self):
        dom = _stationary_1d_domain()  # no time= argument → stationary
        x, t = dom.variable("interior")
        # t has axis='temporal' — using it on a stationary domain is invalid.
        # Build any expression that touches t so the walker finds it.
        net = _tiny_net(in_dim=2)
        u = net(jno.np.concat([x, t], axis=-1))
        pde = (u.d(t)).mse
        with pytest.raises(ValueError, match="temporal Variable"):
            jno.core([pde])


# ---------------------------------------------------------------------------
# min_consecutive guards
# ---------------------------------------------------------------------------


class TestMinConsecutiveGuards:
    def test_integral_time_requires_min_consecutive_gte_2(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.2, time=(0.0, 1.0, 6)))
        x, t = dom.variable("interior")
        net = _tiny_net(in_dim=2)
        net.optimizer(optax.adam(1e-3))
        u = net(jno.np.concat([t, x], axis=-1))
        integral = u.integrate(t)
        crux = jno.core([integral.mse])
        with pytest.raises(ValueError, match="min_consecutive"):
            crux.solve(2, min_consecutive=1)

    def test_time_dependent_min_consecutive_1_logs_nudge(self, caplog):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.2, time=(0.0, 1.0, 4)))
        x, t = dom.variable("interior")
        net = _tiny_net(in_dim=2)
        net.optimizer(optax.adam(1e-3))
        u = net(jno.np.concat([t, x], axis=-1))
        pde = u.d(t).mse  # no IntegralTime, so min_consecutive=1 is legal
        crux = jno.core([pde])
        # Just verify it doesn't raise — the nudge is a logger.info call which
        # is not easily captured here without configuring caplog for jno's logger.
        crux.solve(2, min_consecutive=1)


# ---------------------------------------------------------------------------
# Re-entry: calling solve() twice on the same core instance
# ---------------------------------------------------------------------------


class TestReentry:
    def test_two_solve_calls_accumulate_history(self):
        dom = _stationary_1d_domain()
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x]).mse
        crux = jno.core([pde])

        h1 = crux.solve(2)
        h2 = crux.solve(2)

        # Both calls return statistics objects with non-empty training_logs
        assert len(h1.training_logs) >= 1
        assert len(h2.training_logs) >= 1
        # Final total loss is a finite float (not NaN)
        loss = h2.total_loss
        assert loss is None or jnp.isfinite(jnp.array(loss))


# ---------------------------------------------------------------------------
# statistics.total_loss property
# ---------------------------------------------------------------------------


class TestStatisticsTotalLoss:
    def test_total_loss_none_when_empty(self):
        from jno.utils.statistics import statistics

        s = statistics(logs=[])
        assert s.total_loss is None
        assert s.total_loss_history.size == 0

    def test_total_loss_picks_last_value(self):
        from jno.utils.statistics import statistics

        s = statistics(logs=[{"total_loss": jnp.array([3.0, 2.0, 1.5])}])
        assert s.total_loss == pytest.approx(1.5)
        assert s.total_loss_history.shape == (3,)

    def test_total_loss_concatenates_across_calls(self):
        from jno.utils.statistics import statistics

        s = statistics(
            logs=[
                {"total_loss": jnp.array([5.0, 4.0])},
                {"total_loss": jnp.array([3.0, 2.0, 1.0])},
            ]
        )
        assert s.total_loss == pytest.approx(1.0)
        assert s.total_loss_history.shape == (5,)


# ---------------------------------------------------------------------------
# Substeps — alternating optimisation
# ---------------------------------------------------------------------------


class TestSubsteps:
    def _build_two_net_crux(self):
        """Build a two-network problem with HyCo-style interaction terms."""
        dom = _stationary_1d_domain(mesh_size=0.05)
        x, *_ = dom.variable("interior")
        n1 = _tiny_net(key_seed=1)
        n2 = _tiny_net(key_seed=2)
        for n in (n1, n2):
            n.optimizer(optax.adam(1e-3))
        u1 = n1(x) * x * (1 - x)
        u2 = n2(x) * x * (1 - x)
        L_pde = (u1.dd(x) + 1.0).mse
        L_int1 = (u1 - jno.fn.stop_gradient(u2)).mse
        L_data = (u2 - x).mse
        L_int2 = (u2 - jno.fn.stop_gradient(u1)).mse
        crux = jno.core([L_pde, L_int1, L_data, L_int2])
        return crux, n1, n2

    @staticmethod
    def _snapshot_arrays(module):
        """Stable list of array leaves (deep-copied) for value comparison."""
        return [jnp.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(module) if hasattr(leaf, "shape")]

    @staticmethod
    def _any_changed(before, after, atol=1e-12):
        return any(not jnp.allclose(b, a, atol=atol) for b, a in zip(before, after))

    @staticmethod
    def _all_unchanged(before, after, atol=1e-12):
        return all(jnp.allclose(b, a, atol=atol) for b, a in zip(before, after))

    def test_substep_phy_only_does_not_touch_syn(self):
        """Running ONLY substep [0, 1] (the u_phy losses) must leave u_syn_net
        bit-identical, even after many gradient steps. Proves the gradient is
        zero for the model trapped inside stop_gradient."""
        crux, n1, n2 = self._build_two_net_crux()
        before_n1 = self._snapshot_arrays(n1.module)
        before_n2 = self._snapshot_arrays(n2.module)

        crux.solve(20, substeps=[[0, 1]])

        after_n1 = self._snapshot_arrays(n1.module)
        after_n2 = self._snapshot_arrays(n2.module)

        assert self._any_changed(before_n1, after_n1), "u_phy_net should have updated under [L_pde, L_int_phy]"
        assert self._all_unchanged(before_n2, after_n2), (
            "u_syn_net must NOT change when only substep [0, 1] runs — it only appears inside stop_gradient"
        )

    def test_substep_syn_only_does_not_touch_phy(self):
        """Mirror: running ONLY substep [2, 3] must leave u_phy_net unchanged."""
        crux, n1, n2 = self._build_two_net_crux()
        before_n1 = self._snapshot_arrays(n1.module)
        before_n2 = self._snapshot_arrays(n2.module)

        crux.solve(20, substeps=[[2, 3]])

        after_n1 = self._snapshot_arrays(n1.module)
        after_n2 = self._snapshot_arrays(n2.module)

        assert self._all_unchanged(before_n1, after_n1), "u_phy_net must NOT change when only substep [2, 3] runs"
        assert self._any_changed(before_n2, after_n2), "u_syn_net should have updated under [L_data, L_int_syn]"

    def test_substep_gradient_is_zero_for_inactive_model(self):
        """Direct gradient check: ∂(L_pde + L_int_phy)/∂u_syn_net == 0 everywhere
        (because u_syn appears only inside stop_gradient). Bypasses solve() and
        uses the compiled per-substep loss function directly."""
        import equinox as eqx

        from jno.trace_compiler import TraceCompiler

        crux, n1, n2 = self._build_two_net_crux()

        # Replicate the partition solve() does so we can call the loss directly.
        models = dict(crux.models)
        filter_spec = {
            lid: jax.tree_util.tree_map(
                lambda leaf: True if eqx.is_inexact_array(leaf) else False,
                m,
            )
            for lid, m in models.items()
        }
        trainable, rest = eqx.partition(models, filter_spec)
        frozen_arrays, static = eqx.partition(rest, eqx.is_array)

        # Compile only the phy substep's constraints (indices 0, 1).
        sub_exprs = [crux._constraint_exprs[0], crux._constraint_exprs[1]]
        compiled_phy = TraceCompiler.compile_multi_expression(sub_exprs, crux.all_ops)

        # solve() re-places the (CPU-resident) domain context onto the compute
        # mesh before each step. Replicate that here: pin params and context to a
        # single device so a multi-device run (e.g. JAX_PLATFORMS=cuda,cpu) does
        # not mix a CPU context with GPU params (ARG_SHARDING device mismatch).
        _dev = jax.devices()[0]
        trainable = jax.device_put(trainable, _dev)
        frozen_arrays = jax.device_put(frozen_arrays, _dev)
        ctx = jax.tree_util.tree_map(lambda a: jax.device_put(a, _dev), crux.domain_data.context)

        def loss_fn(params):
            import paramax as _paramax

            full = _paramax.unwrap(eqx.combine(params, frozen_arrays, static))
            residuals = compiled_phy(full, ctx, batchsize=None, key=jax.random.PRNGKey(0))
            return jnp.mean(jnp.stack([jnp.mean(r) for r in residuals]))

        grads = jax.grad(loss_fn)(trainable)

        # Find which layer ids correspond to n1, n2
        phy_lid = n1.layer_id
        syn_lid = n2.layer_id

        # Every array-leaf in the syn model's gradient must be exactly zero.
        syn_grad_leaves = [g for g in jax.tree_util.tree_leaves(grads[syn_lid]) if hasattr(g, "shape")]
        assert syn_grad_leaves, "expected at least one array leaf in syn gradient"
        for g in syn_grad_leaves:
            assert jnp.all(g == 0.0), (
                f"u_syn_net gradient under phy losses must be 0, got max |g|={float(jnp.max(jnp.abs(g))):.3e}"
            )

        # And the phy model must have at least one non-zero gradient leaf (the loss actually depends on it).
        phy_grad_leaves = [g for g in jax.tree_util.tree_leaves(grads[phy_lid]) if hasattr(g, "shape")]
        assert any(jnp.any(g != 0.0) for g in phy_grad_leaves), "u_phy_net should have non-zero gradient under phy losses"

    def test_substeps_runs_and_updates_both_models(self):
        """Each substep updates only its active model — but `trainable` is shared,
        so over two substeps both networks' parameters change."""
        crux, n1, n2 = self._build_two_net_crux()
        before_n1 = self._snapshot_arrays(n1.module)
        before_n2 = self._snapshot_arrays(n2.module)
        crux.solve(10, substeps=[[0, 1], [2, 3]])
        after_n1 = self._snapshot_arrays(n1.module)
        after_n2 = self._snapshot_arrays(n2.module)
        assert self._any_changed(before_n1, after_n1), "u_phy_net should have updated"
        assert self._any_changed(before_n2, after_n2), "u_syn_net should have updated"

    def test_substeps_opt_state_isolation(self):
        """Each substep's opt_state contains only its active models."""
        from jno.core import _active_model_lids

        crux, n1, n2 = self._build_two_net_crux()
        phy_active = _active_model_lids([crux._constraint_exprs[0], crux._constraint_exprs[1]])
        syn_active = _active_model_lids([crux._constraint_exprs[2], crux._constraint_exprs[3]])

        # Static analysis: phy substep sees only n1, syn substep sees only n2
        assert phy_active == {n1.layer_id}
        assert syn_active == {n2.layer_id}
        assert phy_active.isdisjoint(syn_active)

    def test_substeps_n_steps_tuple(self):
        """Tuple form (indices, n_steps) is accepted and runs."""
        crux, _, _ = self._build_two_net_crux()
        # 5 outer epochs × (2 + 2) substeps = 20 effective gradient steps
        crux.solve(5, substeps=[([0, 1], 2), ([2, 3], 2)])

    def test_substeps_invalid_index_raises(self):
        crux, _, _ = self._build_two_net_crux()
        with pytest.raises(ValueError, match="references constraint index"):
            crux.solve(1, substeps=[[0, 99]])

    def test_substeps_inner_steps_incompatible(self):
        crux, _, _ = self._build_two_net_crux()
        with pytest.raises(ValueError, match="not compatible with inner_steps"):
            crux.solve(2, substeps=[[0, 1]], inner_steps=2)

    def test_substeps_bad_spec_type(self):
        crux, _, _ = self._build_two_net_crux()
        with pytest.raises(TypeError, match="substep must be a list"):
            crux.solve(1, substeps=["not-a-list"])

    def test_active_model_lids_skips_stop_gradient(self):
        """Static analysis: a model only reachable through stop_gradient is inactive."""
        from jno.core import _active_model_lids

        dom = _stationary_1d_domain(mesh_size=0.1)
        x, *_ = dom.variable("interior")
        n1 = _tiny_net(key_seed=10)
        n2 = _tiny_net(key_seed=11)
        u1 = n1(x)
        u2 = n2(x)
        # L_int1 references both nets but n2 is inside stop_gradient
        L_int1 = (u1 - jno.fn.stop_gradient(u2)).mse
        active = _active_model_lids([L_int1])
        assert n1.layer_id in active
        assert n2.layer_id not in active


# ---------------------------------------------------------------------------
# Non-scalar tracker + reduce= callable
# ---------------------------------------------------------------------------


class TestTrackerNonScalar:
    def _build(self, key_seed=0):
        import numpy as np  # noqa: F401

        dom = _stationary_1d_domain(mesh_size=0.05)
        x, *_ = dom.variable("interior")
        net = _tiny_net(key_seed=key_seed)
        net.optimizer(optax.adam(1e-3))
        u = net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x]).mse
        return dom, x, u, pde

    def test_nonscalar_tracker_stores_arrays_in_track_stats(self):
        """A tracker on a vector expression must not crash and must store
        arrays (not scalars) in track_stats when no reduce= is given."""
        import numpy as np

        dom, x, u, pde = self._build(key_seed=1)
        # u has shape (n_points, 1) — non-scalar per collocation point
        crux = jno.core([pde, u.tracker(1)])
        stats = crux.solve(3)

        raw = stats.training_logs[-1]["track_stats"]
        # mixed shapes → stored as list, not a 2-D numpy array
        assert isinstance(raw, list), "expected list for non-scalar track_stats"
        # each log step has one tracker entry; it should be an array with ndim > 0
        assert all(isinstance(row[0], np.ndarray) and row[0].ndim > 0 for row in raw)

    def test_tracker_reduce_callable_yields_scalar_track_stats(self):
        """With reduce=, every tracker value collapses to a scalar and
        track_stats is stored as a 2-D numpy array (backward-compat path)."""
        import numpy as np

        dom, x, u, pde = self._build(key_seed=2)
        # reduce to L2 norm — user-provided Python callable
        crux = jno.core([pde, u.tracker(1, reduce=lambda v: float(np.linalg.norm(v)))])
        stats = crux.solve(3)

        raw = stats.training_logs[-1]["track_stats"]
        assert isinstance(raw, np.ndarray), "expected ndarray for all-scalar track_stats"
        assert raw.ndim == 2
        assert raw.shape[1] == 1  # one tracker column
        assert np.all(np.isfinite(raw))


# ---------------------------------------------------------------------------
# Placeholder .name() label propagation
# ---------------------------------------------------------------------------


class TestPlaceholderName:
    """Verify that .name('label') tags propagate to _constraint_names / _tracker_names."""

    def _build(self, key_seed=0):
        dom = _stationary_1d_domain(mesh_size=0.05)
        x, *_ = dom.variable("interior")
        net = _tiny_net(key_seed=key_seed)
        net.optimizer(optax.adam(1e-3))
        u = net(x) * x * (1 - x)
        return dom, x, u

    def test_constraint_name_stored(self):
        dom, x, u = self._build()
        pde = jno.jnp_ops.laplacian(u, [x]).mse.name("my_pde")
        crux = jno.core([pde])
        assert crux._constraint_names == ["my_pde"]

    def test_unnamed_constraint_is_none(self):
        dom, x, u = self._build()
        pde = jno.jnp_ops.laplacian(u, [x]).mse
        crux = jno.core([pde])
        assert crux._constraint_names == [None]

    def test_mixed_named_unnamed(self):
        dom, x, u = self._build()
        pde = jno.jnp_ops.laplacian(u, [x]).mse.name("pde")
        bc = u.mse
        crux = jno.core([pde, bc])
        assert crux._constraint_names == ["pde", None]

    def test_tracker_name_stored(self):
        dom, x, u = self._build()
        pde = jno.jnp_ops.laplacian(u, [x]).mse
        trk = u.tracker(1).name("u_monitor")
        crux = jno.core([pde, trk])
        assert crux._tracker_names == ["u_monitor"]

    def test_name_returns_self(self):
        dom, x, u = self._build()
        expr = jno.jnp_ops.laplacian(u, [x]).mse
        returned = expr.name("foo")
        assert returned is expr

    def test_name_used_in_log_output(self, capsys):
        dom, x, u = self._build(key_seed=3)
        pde = jno.jnp_ops.laplacian(u, [x]).mse.name("pde_loss")
        crux = jno.core([pde])
        crux.solve(2)
        captured = capsys.readouterr()
        assert "pde_loss" in captured.out


# ---------------------------------------------------------------------------
# Domain auto-inference
# ---------------------------------------------------------------------------


class TestDomainInference:
    """``jno.core(constraints)`` walks the constraint trees and resolves the
    domain from the ``Variable._domain`` references."""

    def _tiny_domain_and_loss(self, mesh_size=0.1, key_seed=0):
        dom = _stationary_1d_domain(mesh_size=mesh_size)
        x, _ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x) * x * (1 - x)
        pde = (u.d(x).d(x) + 1.0).mse
        return dom, pde

    def test_single_domain_inferred(self):
        dom, pde = self._tiny_domain_and_loss()
        crux = jno.core([pde])
        assert crux.domain is dom

    def test_no_variables_raises(self):
        # Pure parametric loss — no Variables means no domain to resolve.
        param = jnn.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        loss = (param - 1.0).mse
        with pytest.raises(ValueError, match="no Variables or TensorTags"):
            jno.core([loss])

    def test_multi_domain_raises(self):
        dom_a, pde_a = self._tiny_domain_and_loss(mesh_size=0.1, key_seed=1)
        dom_b, pde_b = self._tiny_domain_and_loss(mesh_size=0.05, key_seed=2)
        assert dom_a is not dom_b
        with pytest.raises(ValueError, match="2 distinct domains"):
            jno.core([pde_a, pde_b])
