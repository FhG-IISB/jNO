"""Tests for :class:`FieldView` — FD-only partial derivatives on neural-operator
field outputs.

The "neural operator" is mocked by binding a known field as a domain Variable
and chaining ``.field.bind(...)`` on it; the FD scheme treats the bound array
as if it were the output of a Poseidon-style network.  Spatial first/second
order, multi-channel, temporal first/second order, and mixed spatiotemporal
derivatives are all checked against analytic ground truth.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.trace import (
    BinaryOp,
    FieldView,
    FieldViewWithPartials,
    Hessian,
    Jacobian,
    ScalarView,
    TemporalDerivative,
)
from jno.trace_compiler import _collect_temporal_derivative_targets
from jno.trace_evaluator import TraceEvaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid(nx_minus_1: int = 7):
    """Return (H, W) = (nx+1, ny+1) coordinate grids on [0, 1]^2 in row-major
    order — matches ``equi_distant_rect(nx=nx_minus_1)`` mesh point ordering.
    """
    n = nx_minus_1 + 1
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return xx.astype(np.float32), yy.astype(np.float32)


def _interior_slice(arr, k: int = 1):
    """Drop ``k`` cells from every spatial boundary for FD-stencil error."""
    return arr[k:-k, k:-k]


# ---------------------------------------------------------------------------
# Node construction — pure trace assertions
# ---------------------------------------------------------------------------


class TestFieldViewNodeConstruction:
    """``.field.bind().x/.xx/.t/...`` should emit the right node types."""

    @pytest.fixture
    def setup(self):
        domain = jno.domain.equi_distant_rect(nx=3, ny=3, time=(0.0, 1.0, 4))
        a = np.zeros((1, 4, 4, 4, 1), dtype=np.float32)
        a_var = domain.variable("a", a)
        x_var, y_var, t_var = domain.variable("interior")
        return a_var, x_var, y_var, t_var

    def test_field_returns_FieldView(self, setup):
        a_var, *_ = setup
        assert isinstance(a_var.field, FieldView)

    def test_bind_returns_FieldViewWithPartials(self, setup):
        a_var, x_var, y_var, t_var = setup
        fv = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        assert isinstance(fv, FieldViewWithPartials)

    def test_x_is_Jacobian_fd(self, setup):
        a_var, x_var, y_var, t_var = setup
        fv = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        node = fv.x.expr
        assert isinstance(node, Jacobian)
        assert node.scheme.startswith("finite_difference")

    def test_xx_is_Hessian_trace_fd(self, setup):
        a_var, x_var, y_var, t_var = setup
        fv = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        node = fv.xx.expr
        assert isinstance(node, Hessian)
        assert node.trace is True
        assert node.scheme.startswith("finite_difference")

    def test_xy_is_Hessian_full_fd(self, setup):
        a_var, x_var, y_var, t_var = setup
        fv = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        node = fv.xy.expr
        assert isinstance(node, Hessian)
        assert node.trace is False

    def test_t_is_TemporalDerivative(self, setup):
        a_var, x_var, y_var, t_var = setup
        fv = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        node = fv.t.expr
        assert isinstance(node, TemporalDerivative)

    def test_tt_is_nested_TemporalDerivative(self, setup):
        a_var, x_var, y_var, t_var = setup
        fv = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        node = fv.tt.expr
        assert isinstance(node, TemporalDerivative)
        assert isinstance(node.target, TemporalDerivative)

    def test_xt_temporal_outer(self, setup):
        """Left-to-right parsing: ``.xt`` → temporal-outer / spatial-inner."""
        a_var, x_var, y_var, t_var = setup
        fv = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        node = fv.xt.expr
        assert isinstance(node, TemporalDerivative)
        assert isinstance(node.target, Jacobian)

    def test_tx_spatial_outer(self, setup):
        a_var, x_var, y_var, t_var = setup
        fv = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        node = fv.tx.expr
        assert isinstance(node, Jacobian)
        assert isinstance(node.target, TemporalDerivative)


# ---------------------------------------------------------------------------
# Topological pre-computation order
# ---------------------------------------------------------------------------


class TestTemporalTargetCollection:
    """``_collect_temporal_derivative_targets`` must return innermost-first."""

    @pytest.fixture
    def fv(self):
        domain = jno.domain.equi_distant_rect(nx=3, ny=3, time=(0.0, 1.0, 4))
        a = np.zeros((1, 4, 4, 4, 1), dtype=np.float32)
        a_var = domain.variable("a", a)
        x_var, _, t_var = domain.variable("interior")
        return a_var.field.bind(x=x_var, t=t_var)

    def test_single_t_yields_one_target(self, fv):
        targets = _collect_temporal_derivative_targets(fv.t.expr)
        assert len(targets) == 1

    def test_tt_yields_two_targets_in_order(self, fv):
        """For ``.tt``, the inner ``u`` is pre-computed before the inner TD.

        Post-order traversal: ``targets[0]`` = inner TD's target (the leaf ``u``),
        ``targets[1]`` = outer TD's target (the inner TD itself).
        """
        outer_td = fv.tt.expr  # TemporalDerivative(TemporalDerivative(u, t), t)
        inner_td = outer_td.target  # TemporalDerivative(u, t)
        leaf = inner_td.target  # u

        targets = _collect_temporal_derivative_targets(outer_td)
        assert len(targets) == 2
        first_target, second_target = targets[0][0], targets[1][0]
        assert first_target is leaf
        assert second_target is inner_td


# ---------------------------------------------------------------------------
# Coordinate validation
# ---------------------------------------------------------------------------


class TestCoordinateValidation:
    def _setup(self):
        domain = jno.domain.equi_distant_rect(nx=3, ny=3)
        a = np.zeros((1, 4, 4, 1), dtype=np.float32)
        a_var = domain.variable("a", a)
        x_var, y_var, _ = domain.variable("interior")
        return a_var, x_var, y_var

    def test_matching_coords_no_warning(self):
        a_var, x_var, y_var = self._setup()
        xs, ys = _grid(nx_minus_1=3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would raise
            a_var.field.bind(x=x_var, y=y_var, x_coords=xs, y_coords=ys)

    def test_size_mismatch_emits_warning(self):
        a_var, x_var, y_var = self._setup()
        # Wrong size — mesh has 16 points, we pass 9.
        bad_xs = np.zeros((3, 3), dtype=np.float32)
        bad_ys = np.zeros((3, 3), dtype=np.float32)
        with pytest.warns(UserWarning, match="coordinate mismatch"):
            a_var.field.bind(x=x_var, y=y_var, x_coords=bad_xs, y_coords=bad_ys)

    def test_offset_coords_emit_mismatch_warning(self):
        a_var, x_var, y_var = self._setup()
        xs, ys = _grid(nx_minus_1=3)
        shifted = xs + 0.5
        with pytest.warns(UserWarning, match="coordinate mismatch"):
            a_var.field.bind(x=x_var, y=y_var, x_coords=shifted, y_coords=ys)


# ---------------------------------------------------------------------------
# End-to-end FD via jno.core(...).eval(...)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpu_only():
    """Force CPU for these tests — they don't need GPU and the smoke run shouldn't
    contend with the user's GPU memory."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _make_spatial_setup(field_values: np.ndarray):
    """Build a domain + variables for a steady-state 2-D field.

    ``field_values`` has shape ``(H, W)`` or ``(H, W, C)``.  Returns
    (domain, a_var, x_var, y_var, ctx) where ``ctx`` is the prebuilt
    evaluator context dictionary.
    """
    field = np.asarray(field_values, dtype=np.float32)
    H, W = field.shape[:2]
    if field.ndim == 2:
        field = field[..., None]
    domain = jno.domain.equi_distant_rect(nx=H - 1, ny=W - 1)
    a_var = domain.variable("a", field[None])  # add batch dim
    x_var, y_var, _ = domain.variable("interior")
    mesh_pts = jnp.asarray(domain.mesh_connectivity["points"])
    ctx = {"a": field, "interior": mesh_pts}
    return domain, a_var, x_var, y_var, ctx


def _eval_direct(expr, ctx):
    """Evaluate ``expr`` directly via :class:`TraceEvaluator` (no compiler)."""
    ev = TraceEvaluator({})
    return np.asarray(ev.evaluate(expr, ctx, {}, key=jax.random.PRNGKey(0)))


class TestSpatialFD:
    """First- and second-order spatial FD on a known polynomial field."""

    def test_xx_of_x_squared_plus_y_squared(self, cpu_only):
        """``.xx`` on a separable polynomial yields the exact second derivative."""
        H = 16
        xs, ys = _grid(nx_minus_1=H - 1)
        f = xs**2 + ys**2  # ∂²/∂x² = 2
        _, a_var, x_var, y_var, ctx = _make_spatial_setup(f)
        u = a_var.field.bind(x=x_var, y=y_var)
        val = np.squeeze(_eval_direct(u.xx.expr, ctx))
        assert val.shape == (H, H)
        np.testing.assert_allclose(_interior_slice(val, k=2), 2.0, atol=1e-3)

    def test_yy_of_x_squared_plus_y_squared(self, cpu_only):
        H = 16
        xs, ys = _grid(nx_minus_1=H - 1)
        f = xs**2 + ys**2
        _, a_var, x_var, y_var, ctx = _make_spatial_setup(f)
        u = a_var.field.bind(x=x_var, y=y_var)
        val = np.squeeze(_eval_direct(u.yy.expr, ctx))
        np.testing.assert_allclose(_interior_slice(val, k=2), 2.0, atol=1e-3)

    def test_partial_x_of_x_squared(self, cpu_only):
        H = 16
        xs, ys = _grid(nx_minus_1=H - 1)
        f = xs**2  # ∂/∂x = 2x, ∂/∂y = 0
        _, a_var, x_var, y_var, ctx = _make_spatial_setup(f)
        u = a_var.field.bind(x=x_var, y=y_var)
        val = np.squeeze(_eval_direct(u.x.expr, ctx))
        expected = 2.0 * xs
        np.testing.assert_allclose(_interior_slice(val, k=2), _interior_slice(expected, k=2), atol=1e-2)


class TestMultiChannelFD:
    """Per-channel FD on a vector-valued field."""

    def test_xx_two_channels(self, cpu_only):
        H = 16
        xs, ys = _grid(nx_minus_1=H - 1)
        # Channel 0: x² + y²  → ∂²/∂x² = 2
        # Channel 1: y² (no x) → ∂²/∂x² = 0
        f = np.stack([xs**2 + ys**2, ys**2], axis=-1)
        _, a_var, x_var, y_var, ctx = _make_spatial_setup(f)
        u = a_var.field.bind(x=x_var, y=y_var)
        val = np.squeeze(_eval_direct(u.xx.expr, ctx))
        assert val.shape == (H, H, 2)
        np.testing.assert_allclose(_interior_slice(val[..., 0], k=2), 2.0, atol=1e-3)
        np.testing.assert_allclose(_interior_slice(val[..., 1], k=2), 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Temporal FD (and mixed) — exercised via crux.eval with min_consecutive=3
# ---------------------------------------------------------------------------


def _make_spatiotemporal_setup(field_values: np.ndarray, t_vals: np.ndarray):
    """``field_values`` has shape ``(T, H, W)`` or ``(T, H, W, C)``.  Returns
    (domain, a_var, x, y, t)."""
    field = np.asarray(field_values, dtype=np.float32)
    T, H, W = field.shape[:3]
    if field.ndim == 3:
        field = field[..., None]
    t0, t1 = float(t_vals[0]), float(t_vals[-1])
    domain = jno.domain.equi_distant_rect(nx=H - 1, ny=W - 1, time=(t0, t1, T))
    field_b = field[None]  # (1, T, H, W, C)
    a_var = domain.variable("a", field_b)
    x_var, y_var, t_var = domain.variable("interior")
    return domain, a_var, x_var, y_var, t_var


class TestTemporalFD:
    """Cross-step temporal derivatives via the pre-computed cache."""

    def test_dt_of_linear_field(self, cpu_only):
        T, H, W = 5, 6, 6
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)
        # f(t, x, y) = t  → ∂/∂t = 1
        field = np.broadcast_to(ts[:, None, None], (T, H, W)).copy()
        domain, a_var, x_var, y_var, t_var = _make_spatiotemporal_setup(field, ts)
        u = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        crux = jno.core([u.t.expr.mse], domain=domain)
        val = crux.eval(u.t.expr, min_consecutive=T)
        val = np.asarray(val[0]) if isinstance(val, list) else np.asarray(val)
        # Shape: (B, T, H, W, 1) or similar; check interior step where
        # central-diff is exact for a linear function.
        val_sq = np.squeeze(val)
        np.testing.assert_allclose(val_sq, 1.0, atol=1e-4)

    def test_dt_of_t_squared(self, cpu_only):
        T, H, W = 7, 6, 6
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)
        # f(t) = t²  → ∂/∂t = 2t
        field = np.broadcast_to((ts**2)[:, None, None], (T, H, W)).copy()
        domain, a_var, x_var, y_var, t_var = _make_spatiotemporal_setup(field, ts)
        u = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        crux = jno.core([u.t.expr.mse], domain=domain)
        val = crux.eval(u.t.expr, min_consecutive=T)
        val_sq = np.squeeze(np.asarray(val[0]) if isinstance(val, list) else val)
        # Interior steps: central diff for t² gives exact 2t.
        # Edge steps use one-sided diff and are approximate.
        # Check interior step middle index 3 against 2*t[3].
        mid = T // 2
        np.testing.assert_allclose(val_sq[mid], 2.0 * ts[mid], atol=1e-4)

    def test_dtt_of_t_squared(self, cpu_only):
        """Second temporal derivative — central diff via nested TDs."""
        T, H, W = 7, 6, 6
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)
        # f(t) = t²  → ∂²/∂t² = 2
        field = np.broadcast_to((ts**2)[:, None, None], (T, H, W)).copy()
        domain, a_var, x_var, y_var, t_var = _make_spatiotemporal_setup(field, ts)
        u = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        crux = jno.core([u.tt.expr.mse], domain=domain)
        val = crux.eval(u.tt.expr, min_consecutive=T)
        val_sq = np.squeeze(np.asarray(val[0]) if isinstance(val, list) else val)
        # Far-interior step (index 3, away from both edges).
        mid = T // 2
        np.testing.assert_allclose(val_sq[mid], 2.0, atol=1e-3)

    def test_mixed_spatiotemporal(self, cpu_only):
        """``.xt`` and ``.tx`` should both produce ``d²u/(dx dt)`` on smooth fields."""
        T, H = 5, 16
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)
        xs, ys = _grid(nx_minus_1=H - 1)
        # f(t, x, y) = t * x²  → ∂²f/(∂t∂x) = ∂(t*2x)/∂t = 2x
        #                    or ∂(x²)/∂x integrated then ∂/∂t → 2x
        field = ts[:, None, None] * (xs**2)[None, :, :]
        domain, a_var, x_var, y_var, t_var = _make_spatiotemporal_setup(field, ts)
        u = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        crux = jno.core([u.xt.expr.mse], domain=domain)
        val = crux.eval(u.xt.expr, min_consecutive=T)
        val_sq = np.squeeze(np.asarray(val[0]) if isinstance(val, list) else val)
        # val_sq shape: (T, H, W) — for each interior step the spatial value
        # should be ≈ 2*x.
        mid = T // 2
        expected = 2.0 * xs
        np.testing.assert_allclose(_interior_slice(val_sq[mid], k=2), _interior_slice(expected, k=2), atol=2e-2)


class TestMissingCacheRaises:
    """Without a temporal window, ``.t`` must raise a clear error."""

    def test_direct_eval_t_raises_without_cache(self, cpu_only):
        # Directly evaluate a TemporalDerivative without populating the
        # __temporal_fd_cache__ — the evaluator must raise a clear error.
        H = W = 4
        f = np.zeros((H, W), dtype=np.float32)
        domain, a_var, x_var, y_var, ctx = _make_spatial_setup(f)
        from jno.trace import Variable

        bogus_t = Variable("__time__", [0, 1], domain=domain, axis="temporal")
        u = a_var.field.bind(x=x_var, y=y_var, t=bogus_t)
        with pytest.raises(RuntimeError, match="cache not populated|cache not"):
            _eval_direct(u.t.expr, ctx)


# ---------------------------------------------------------------------------
# Cross-term arithmetic: u.t + u.xx (heat-equation pattern)
# ---------------------------------------------------------------------------


class TestCrossTermArithmetic:
    """``u.t + u.xx`` must compile to a BinaryOp and evaluate correctly.

    This is the core use case — the heat-equation residual
    ``(u_t - κ(u_xx + u_yy)).mse`` must be constructable and evaluable with
    ``min_consecutive >= 2``.
    """

    @pytest.fixture
    def heat_setup(self, cpu_only):
        T, H = 5, 8
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)
        xs, ys = _grid(nx_minus_1=H - 1)
        # f(t, x, y) = t + x² + y²  → ∂f/∂t = 1, ∂²f/∂x² = 2, sum = 3
        field = ts[:, None, None] + xs[None] ** 2 + ys[None] ** 2
        domain, a_var, x_var, y_var, t_var = _make_spatiotemporal_setup(field, ts)
        u = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        return u, domain, T, H

    def test_t_plus_xx_is_binary_op(self, heat_setup):
        u, *_ = heat_setup
        assert isinstance((u.t + u.xx).expr, BinaryOp)

    def test_t_plus_xx_evaluates_correctly(self, heat_setup):
        u, domain, T, H = heat_setup
        crux = jno.core([(u.t + u.xx).expr.mse], domain=domain)
        val = crux.eval((u.t + u.xx).expr, min_consecutive=T)
        val_sq = np.squeeze(np.asarray(val[0]) if isinstance(val, list) else val)
        mid = T // 2
        np.testing.assert_allclose(_interior_slice(val_sq[mid], k=2), 3.0, atol=5e-2)

    def test_heat_residual_mse_is_compact(self, heat_setup):
        """``.mse`` returns a spatially-averaged (compact) tensor — not the full field."""
        u, domain, T, H = heat_setup
        crux = jno.core([(u.t - u.xx - u.yy).expr.mse], domain=domain)
        val = crux.eval((u.t - u.xx - u.yy).expr.mse, min_consecutive=T)
        result = np.asarray(val[0] if isinstance(val, list) else val)
        # Result is spatially averaged: at most (batch, T) — never field-sized
        assert result.size <= T and result.size > 0


# ---------------------------------------------------------------------------
# _rewrap type preservation through arithmetic
# ---------------------------------------------------------------------------


class TestArithmeticPreservation:
    """Arithmetic on ``FieldViewWithPartials`` stays ``FieldViewWithPartials``."""

    @pytest.fixture
    def fv(self, cpu_only):
        H = 16
        xs, ys = _grid(nx_minus_1=H - 1)
        f = xs**2 + ys**2
        _, a_var, x_var, y_var, _ = _make_spatial_setup(f)
        return a_var.field.bind(x=x_var, y=y_var), xs

    def test_sub_preserves_type(self, fv):
        u, _ = fv
        assert isinstance(u - 1.0, FieldViewWithPartials)

    def test_xx_after_constant_shift_unchanged(self, cpu_only):
        """∂²(u + c)/∂x² = ∂²u/∂x² — constant shift leaves second derivative unchanged."""
        H = 16
        xs, ys = _grid(nx_minus_1=H - 1)
        f = xs**2 + ys**2
        _, a_var, x_var, y_var, ctx = _make_spatial_setup(f)
        u = a_var.field.bind(x=x_var, y=y_var)
        val_u = np.squeeze(_eval_direct(u.xx.expr, ctx))
        val_shifted = np.squeeze(_eval_direct((u + 5.0).xx.expr, ctx))
        # FD of (u + constant) should equal FD of u; tolerate float32 rounding
        np.testing.assert_allclose(val_u, val_shifted, atol=1e-4)


# ---------------------------------------------------------------------------
# Coord binding conflict
# ---------------------------------------------------------------------------


class TestConflictingBindings:
    def test_conflicting_x_raises(self, cpu_only):
        """Two FieldViewWithPartials with the same name bound to different
        Variables raise ``ValueError`` on arithmetic."""
        H = 4
        f = np.zeros((H, H), dtype=np.float32)
        _, a1, x1, y1, _ = _make_spatial_setup(f)
        _, a2, x2, y2, _ = _make_spatial_setup(f)
        u1 = a1.field.bind(x=x1, y=y1)
        u2 = a2.field.bind(x=x2, y=y2)  # x2 is a different Variable from x1
        with pytest.raises(ValueError, match="coord binding conflict"):
            _ = u1 + u2


# ---------------------------------------------------------------------------
# Non-square domain
# ---------------------------------------------------------------------------


class TestNonSquareDomain:
    """FD must work when H ≠ W."""

    @pytest.fixture
    def nonsquare(self, cpu_only):
        H, W = 8, 12
        xs = np.linspace(0.0, 1.0, H, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, W, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        field = (yy**2)[..., None]  # (H, W, 1); ∂²/∂y² = 2
        domain = jno.domain.equi_distant_rect(nx=H - 1, ny=W - 1)
        a_var = domain.variable("a", field[None])
        x_var, y_var, _ = domain.variable("interior")
        mesh_pts = jnp.asarray(domain.mesh_connectivity["points"])
        ctx = {"a": field, "interior": mesh_pts}
        return domain, a_var, x_var, y_var, ctx, H, W

    def test_xx_non_square_shape(self, nonsquare):
        _, a_var, x_var, y_var, ctx, H, W = nonsquare
        u = a_var.field.bind(x=x_var, y=y_var)
        val = _eval_direct(u.xx.expr, ctx)
        assert val.shape == (H, W, 1)

    def test_yy_non_square_accuracy(self, nonsquare):
        _, a_var, x_var, y_var, ctx, H, W = nonsquare
        u = a_var.field.bind(x=x_var, y=y_var)
        val = np.squeeze(_eval_direct(u.yy.expr, ctx))
        np.testing.assert_allclose(_interior_slice(val, k=2), 2.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Multi-channel temporal FD
# ---------------------------------------------------------------------------


class TestMultiChannelTemporalFD:
    def test_dt_two_channels_independent(self, cpu_only):
        """`.t` on a two-channel field differentiates each channel independently."""
        T, H = 7, 6
        W = H
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)
        # Channel 0: f = t  → ∂/∂t = 1
        # Channel 1: f = 2t → ∂/∂t = 2
        ch0 = np.broadcast_to(ts[:, None, None], (T, H, W)).copy()
        ch1 = 2.0 * ch0
        field = np.stack([ch0, ch1], axis=-1)  # (T, H, W, 2)
        domain, a_var, x_var, y_var, t_var = _make_spatiotemporal_setup(field, ts)
        u = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        crux = jno.core([u.t.expr.mse], domain=domain)
        val = crux.eval(u.t.expr, min_consecutive=T)
        val_sq = np.squeeze(np.asarray(val[0]) if isinstance(val, list) else val)
        mid = T // 2
        # val_sq shape: (T, H, W, 2) after squeeze
        np.testing.assert_allclose(val_sq[mid, :, :, 0], 1.0, atol=1e-4)
        np.testing.assert_allclose(val_sq[mid, :, :, 1], 2.0, atol=1e-4)


# ---------------------------------------------------------------------------
# min_consecutive guards
# ---------------------------------------------------------------------------


class TestMinConsecutiveGuard:
    """``min_consecutive < 2`` must be caught before evaluation starts."""

    def _setup_td_domain(self):
        T, H, W = 5, 4, 4
        ts = np.linspace(0.0, 1.0, T, dtype=np.float32)
        field = np.broadcast_to(ts[:, None, None], (T, H, W)).copy()
        return _make_spatiotemporal_setup(field, ts)

    def test_solve_min_consecutive_1_raises(self, cpu_only):
        domain, a_var, x_var, y_var, t_var = self._setup_td_domain()
        u = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        crux = jno.core([u.t.expr.mse], domain=domain)
        with pytest.raises(ValueError, match="min_consecutive"):
            crux.solve(epochs=1, min_consecutive=1)

    def test_eval_min_consecutive_1_raises(self, cpu_only):
        domain, a_var, x_var, y_var, t_var = self._setup_td_domain()
        u = a_var.field.bind(x=x_var, y=y_var, t=t_var)
        crux = jno.core([u.t.expr.mse], domain=domain)
        with pytest.raises((ValueError, RuntimeError)):
            crux.eval(u.t.expr, min_consecutive=1)


# ---------------------------------------------------------------------------
# Higher-order chain node structure
# ---------------------------------------------------------------------------


class TestHigherOrderChainNodes:
    @pytest.fixture
    def fv_xt(self):
        domain = jno.domain.equi_distant_rect(nx=3, ny=3, time=(0.0, 1.0, 4))
        a = np.zeros((1, 4, 4, 4, 1), dtype=np.float32)
        a_var = domain.variable("a", a)
        x_var, y_var, t_var = domain.variable("interior")
        return a_var.field.bind(x=x_var, y=y_var, t=t_var)

    def test_xxt_structure(self, fv_xt):
        """.xxt = TemporalDerivative(Hessian(u, trace=True), t)."""
        u = fv_xt
        node = u.xxt.expr
        assert isinstance(node, TemporalDerivative)
        assert isinstance(node.target, Hessian)
        assert node.target.trace is True

    def test_txx_structure(self, fv_xt):
        """.txx = Hessian(TemporalDerivative(u, t), trace=True)."""
        u = fv_xt
        node = u.txx.expr
        assert isinstance(node, Hessian)
        assert node.trace is True
        assert isinstance(node.target, TemporalDerivative)


# ---------------------------------------------------------------------------
# Boundary condition attribute API
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    """Verify ``u.left``, ``u.right``, ``u.x.right``, ``u.x.left + α*u.left``."""

    @pytest.fixture
    def bc_setup(self, cpu_only):
        H, W = 8, 8
        xs, ys = _grid(nx_minus_1=H - 1)
        f = xs**2 + ys**2
        domain, a_var, x_var, y_var, ctx = _make_spatial_setup(f)
        u = a_var.field.bind(x=x_var, y=y_var)
        return u, domain, ctx, H, W

    def test_grid_shape(self, bc_setup):
        u, domain, ctx, H, W = bc_setup
        assert u.grid_shape == (H, W)

    def test_left_is_scalar_view_not_field_view(self, bc_setup):
        u, *_ = bc_setup
        left = u.left
        assert isinstance(left, ScalarView)
        assert not isinstance(left, FieldViewWithPartials)

    def test_dirichlet_left_shape(self, bc_setup):
        u, _, ctx, H, W = bc_setup
        val = _eval_direct(u.left.expr, ctx)
        # Left boundary: x=0 → first row → shape (1, W, 1)
        assert val.shape == (1, W, 1)

    def test_periodic_bc_evaluates_to_scalar(self, bc_setup):
        """``(u.left - u.right).mse`` produces a scalar loss."""
        u, _, ctx, *_ = bc_setup
        mse_node = (u.left - u.right).mse
        result = np.asarray(_eval_direct(mse_node, ctx))
        assert result.ndim == 0

    def test_neumann_x_right_evaluates(self, bc_setup):
        """``u.x.right.mse`` — FD gradient at right wall — evaluates to scalar."""
        u, _, ctx, *_ = bc_setup
        mse_node = u.x.right.mse
        result = np.asarray(_eval_direct(mse_node, ctx))
        assert result.ndim == 0

    def test_robin_bc_evaluates_to_scalar(self, bc_setup):
        """``(u.x.left + 0.5*u.left).mse`` Robin BC evaluates to scalar."""
        u, _, ctx, *_ = bc_setup
        robin = (u.x.left + 0.5 * u.left).mse
        result = np.asarray(_eval_direct(robin, ctx))
        assert result.ndim == 0

    def test_x_right_is_scalar_view(self, bc_setup):
        """``u.x.right`` returns ``ScalarView`` (boundary of derivative field)."""
        u, *_ = bc_setup
        assert isinstance(u.x.right, ScalarView)


# ---------------------------------------------------------------------------
# bind without temporal variable — unregistered attribute falls through
# ---------------------------------------------------------------------------


class TestSpatialOnlyBind:
    def test_unregistered_t_raises_attr_error(self, cpu_only):
        H = 4
        f = np.zeros((H, H), dtype=np.float32)
        _, a_var, x_var, y_var, _ = _make_spatial_setup(f)
        u = a_var.field.bind(x=x_var, y=y_var)  # no t
        with pytest.raises(AttributeError):
            _ = u.t  # "t" is not a registered name → falls through to expr.t → AttributeError


class TestFieldViewComposition:
    """FieldView FD partials composing into vector / matrix arithmetic.

    Assembling a gradient vector from FD partials, taking its norm, or applying
    a constant material matrix is valid — the partials are ordinary
    Placeholders.  Re-*differentiating* them is not (see
    :class:`TestFieldViewADGuard`).
    """

    def test_fd_gradient_as_vector_norm(self, cpu_only):
        H = 16
        xs, ys = _grid(nx_minus_1=H - 1)
        f = xs**2 + ys**2  # ∇f = (2x, 2y); |∇f| = 2·√(x²+y²)
        _, a_var, x_var, y_var, ctx = _make_spatial_setup(f)
        u = a_var.field.bind(x=x_var, y=y_var)
        gradvec = jno.np.vector(u.x, u.y)  # FieldViewWithPartials unwrap → FD Jacobians
        gnorm = np.squeeze(_eval_direct(gradvec.norm().expr, ctx))
        expected = 2.0 * np.sqrt(xs**2 + ys**2)
        np.testing.assert_allclose(_interior_slice(gnorm, k=2), _interior_slice(expected, k=2), atol=2e-2)

    def test_constant_matrix_times_fd_gradient(self, cpu_only):
        from jno.trace import Variable
        from tests.conftest import MockDomain

        H = 16
        xs, ys = _grid(nx_minus_1=H - 1)
        f = xs**2 + ys**2
        _, a_var, x_var, y_var, ctx = _make_spatial_setup(f)
        u = a_var.field.bind(x=x_var, y=y_var)
        gradvec = jno.np.vector(u.x, u.y)
        kd = MockDomain()
        kd.context["k"] = jnp.zeros((1, 4))
        K = Variable("k", [0, 4], domain=kd).matrix.from_flat(2)
        flux = K @ gradvec  # material flux K·∇u
        ctx2 = dict(ctx)
        ctx2["k"] = jnp.array([[1.0, 0.0, 0.0, 1.0]])  # K = I → flux = ∇u
        out = np.squeeze(_eval_direct(flux.expr, ctx2))
        assert out.shape == (H, H, 2)
        np.testing.assert_allclose(_interior_slice(out[..., 0], k=2), _interior_slice(2.0 * xs, k=2), atol=2e-2)
        np.testing.assert_allclose(_interior_slice(out[..., 1], k=2), _interior_slice(2.0 * ys, k=2), atol=2e-2)


class TestFieldViewADGuard:
    """AD differential operators over FieldView FD partials must raise.

    The field is a grid output whose coordinates are not network inputs, so
    automatic differentiation of an FD partial silently evaluates to 0 — a
    plausible-looking wrong answer.  These ops raise ``ValueError`` instead;
    use the FieldView FD API (``u.xx`` / ``u.yy`` / ``u.tt``) for higher orders.
    The guard fires at trace-construction time, so no evaluation is needed.
    """

    @pytest.fixture
    def fv(self):
        H = 8
        f = np.zeros((H, H), dtype=np.float32)
        _, a_var, x_var, y_var, _ = _make_spatial_setup(f)
        return a_var.field.bind(x=x_var, y=y_var), x_var, y_var

    def test_div_of_fd_grad_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            jno.np.vector(u.x, u.y).div(x, y)

    def test_curl_of_fd_grad_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            jno.np.vector(u.x, u.y).curl(x, y)

    def test_jacobian_of_fd_grad_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            jno.np.vector(u.x, u.y).jacobian(x, y)

    def test_second_ad_derivative_on_fd_partial_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            u.x.d(x)

    def test_ad_laplacian_on_fd_partial_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            u.x.laplacian(x, y)

    def test_fieldview_fd_higher_order_still_allowed(self, fv):
        """Contrast: the FD API itself must NOT raise."""
        u, x, y = fv
        _ = u.xx + u.yy  # FD Laplacian
        _ = u.x.x  # FD chain
        _ = u.x.d(x, scheme="finite_difference")  # explicit FD scheme

    # The functional jno.np.* API must guard the same way as the method/view API.
    def test_functional_divergence_of_fd_grad_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            jno.np.divergence([u.x, u.y], [x, y])

    def test_functional_laplacian_of_fd_partial_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            jno.np.laplacian(u.x, [x, y])

    def test_functional_grad_of_fd_partial_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            jno.np.grad(u.x, x)

    def test_functional_jacobian_of_fd_partial_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            jno.np.jacobian(u.x, [x, y])

    def test_functional_hessian_of_fd_partial_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            jno.np.hessian(u.x, [x, y])

    def test_functional_curl_2d_of_fd_partials_raises(self, fv):
        u, x, y = fv
        with pytest.raises(ValueError, match="finite-difference"):
            jno.np.curl_2d(u.x, u.y, x, y)
