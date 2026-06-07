"""Tests for AD mode switching (forward vs reverse) on derivative operators.

Covers:
  - parse_ad_scheme / parse_hessian_scheme — pure scheme-string parsing.
  - Numerical agreement of all four first-order modes (default,
    :forward, :reverse) against analytic results.
  - Numerical agreement of all four second-order compositions
    (:fwd-over-rev, :fwd-over-fwd, :rev-over-rev, :rev-over-fwd) on
    .laplacian and .hessian.
  - Dispatch routing: monkeypatched ad_fn spy proves the requested mode
    reaches trace_evaluator for first- and second-order operators.
  - Global default via jno.setup (and ad_mode.set_ad_mode) plus per-call
    suffix override.

A module-level fixture restores the global mode after each test so the
tests are order-independent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jno.trace import FunctionCall, Literal
from jno.trace_evaluator import TraceEvaluator
from jno.utils.ad_mode import (
    ad_fn,
    get_ad_mode,
    get_hessian_mode,
    parse_ad_scheme,
    parse_hessian_scheme,
    set_ad_mode,
    set_hessian_mode,
)
from jno.utils.config import apply_ad_mode_defaults
from tests.conftest import make_var

FIRST_ORDER_SCHEMES = (
    "automatic_differentiation",
    "automatic_differentiation:forward",
    "automatic_differentiation:reverse",
)
HESSIAN_SCHEMES = (
    "automatic_differentiation",
    "automatic_differentiation:fwd-over-rev",
    "automatic_differentiation:fwd-over-fwd",
    "automatic_differentiation:rev-over-rev",
    "automatic_differentiation:rev-over-fwd",
)


@pytest.fixture(autouse=True)
def _restore_ad_mode():
    """Restore module-level AD defaults after every test."""
    prev_first = get_ad_mode()
    prev_second = get_hessian_mode()
    yield
    set_ad_mode(prev_first)
    set_hessian_mode(prev_second)


def _eval(expr, context):
    ev = TraceEvaluator({})
    return ev.evaluate(expr, context, {}, key=jax.random.PRNGKey(0))


def _pts_1d(n=16):
    return jnp.linspace(0.1, 0.9, n).reshape(n, 1)


def _make_2d_vars():
    """(x, y) sharing tag 'xy' on a 2-D mock domain."""
    from tests.conftest import MockDomain

    d = MockDomain(tags=["xy"], dim=2)
    from jno.trace import Variable

    x = Variable("xy", [0, 1], domain=d)
    y = Variable("xy", [1, 2], domain=d)
    return x, y


def _pts_2d(n=6):
    xs = jnp.linspace(0.1, 0.9, n)
    ys = jnp.linspace(0.1, 0.9, n)
    gx, gy = jnp.meshgrid(xs, ys)
    return jnp.stack([gx.ravel(), gy.ravel()], axis=-1)


# ────────────────────────────────────────────────────────────────────────
# 1. Pure-parser tests
# ────────────────────────────────────────────────────────────────────────


class TestParseAdScheme:
    def test_no_suffix_uses_global_default(self):
        set_ad_mode("forward")
        assert parse_ad_scheme("automatic_differentiation") == "forward"
        set_ad_mode("reverse")
        assert parse_ad_scheme("automatic_differentiation") == "reverse"

    def test_explicit_suffix_overrides_default(self):
        set_ad_mode("forward")
        assert parse_ad_scheme("automatic_differentiation:reverse") == "reverse"

    def test_unknown_suffix_raises(self):
        with pytest.raises(ValueError):
            parse_ad_scheme("automatic_differentiation:bogus")

    def test_set_ad_mode_validates(self):
        with pytest.raises(ValueError):
            set_ad_mode("sideways")


class TestParseHessianScheme:
    def test_no_suffix_uses_global_default(self):
        set_hessian_mode("fwd-over-fwd")
        assert parse_hessian_scheme("automatic_differentiation") == ("forward", "forward")
        set_hessian_mode("rev-over-fwd")
        assert parse_hessian_scheme("automatic_differentiation") == ("reverse", "forward")

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            ("fwd-over-rev", ("forward", "reverse")),
            ("fwd-over-fwd", ("forward", "forward")),
            ("rev-over-rev", ("reverse", "reverse")),
            ("rev-over-fwd", ("reverse", "forward")),
        ],
    )
    def test_explicit_composition(self, suffix, expected):
        assert parse_hessian_scheme(f"automatic_differentiation:{suffix}") == expected

    def test_first_order_suffix_is_same_mode_for_both_layers(self):
        assert parse_hessian_scheme("automatic_differentiation:forward") == ("forward", "forward")
        assert parse_hessian_scheme("automatic_differentiation:reverse") == ("reverse", "reverse")

    def test_unknown_suffix_raises(self):
        with pytest.raises(ValueError):
            parse_hessian_scheme("automatic_differentiation:fwd-rev")


# ────────────────────────────────────────────────────────────────────────
# 2. ad_fn dispatch
# ────────────────────────────────────────────────────────────────────────


class TestAdFn:
    def test_forward_returns_jacfwd(self):
        assert ad_fn("forward") is jax.jacfwd

    def test_reverse_returns_jacrev(self):
        assert ad_fn("reverse") is jax.jacrev

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            ad_fn("middle")


# ────────────────────────────────────────────────────────────────────────
# 3. Numerical correctness across all schemes
# ────────────────────────────────────────────────────────────────────────


class TestFirstOrderNumerical:
    """d/dx(x³) = 3x² must hold for every first-order scheme."""

    @pytest.mark.parametrize("scheme", FIRST_ORDER_SCHEMES)
    def test_d_cubic(self, scheme):
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        result = jnp.ravel(_eval(u.d(x, scheme=scheme), {"x": pts}))
        expected = 3.0 * pts[:, 0] ** 2
        assert jnp.allclose(result, expected, atol=1e-4)

    @pytest.mark.parametrize("scheme", FIRST_ORDER_SCHEMES)
    def test_d_sin(self, scheme):
        x = make_var("x")
        u = FunctionCall(jnp.sin, [x], "sin")
        pts = _pts_1d()
        result = jnp.ravel(_eval(u.d(x, scheme=scheme), {"x": pts}))
        expected = jnp.cos(pts[:, 0])
        assert jnp.allclose(result, expected, atol=1e-4)

    @pytest.mark.parametrize("scheme", FIRST_ORDER_SCHEMES)
    def test_d_2d_partial(self, scheme):
        """∂/∂x(x² + y³) = 2x at (x, y)."""
        x, y = _make_2d_vars()
        u = x ** Literal(2.0) + y ** Literal(3.0)
        pts = _pts_2d()
        result = jnp.ravel(_eval(u.d(x, scheme=scheme), {"xy": pts}))
        expected = 2.0 * pts[:, 0]
        assert jnp.allclose(result, expected, atol=1e-4)


class TestSecondOrderNumerical:
    @pytest.mark.parametrize("scheme", HESSIAN_SCHEMES)
    def test_laplacian_2d_quadratic(self, scheme):
        """Δ(x² + y²) = 4."""
        x, y = _make_2d_vars()
        u = x ** Literal(2.0) + y ** Literal(2.0)
        pts = _pts_2d()
        result = jnp.ravel(_eval(u.laplacian(x, y, scheme=scheme), {"xy": pts}))
        assert jnp.allclose(result, 4.0, atol=1e-4)

    @pytest.mark.parametrize("scheme", HESSIAN_SCHEMES)
    def test_hessian_2d_quadratic(self, scheme):
        """H(x² + xy + y²) = [[2, 1], [1, 2]]."""
        x, y = _make_2d_vars()
        u = x ** Literal(2.0) + x * y + y ** Literal(2.0)
        pts = _pts_2d()
        H = _eval(u.hessian(x, y, scheme=scheme), {"xy": pts})
        expected = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        assert jnp.allclose(H, expected[None, ...], atol=1e-4)


# ────────────────────────────────────────────────────────────────────────
# 4. Dispatch verification — spy on ad_fn to prove the routed mode reaches
# the trace evaluator.
# ────────────────────────────────────────────────────────────────────────


class TestDispatchRouting:
    def test_scheme_forward_routes_to_jacfwd(self, monkeypatch):
        calls = []
        original = ad_fn

        def spy(mode):
            calls.append(mode)
            return original(mode)

        monkeypatch.setattr("jno.trace_evaluator.ad_fn", spy)
        x = make_var("x")
        u = x ** Literal(3.0)
        _eval(u.d(x, scheme="automatic_differentiation:forward"), {"x": _pts_1d()})
        assert "forward" in calls and "reverse" not in calls

    def test_scheme_reverse_routes_to_jacrev(self, monkeypatch):
        calls = []
        original = ad_fn

        def spy(mode):
            calls.append(mode)
            return original(mode)

        monkeypatch.setattr("jno.trace_evaluator.ad_fn", spy)
        x = make_var("x")
        u = x ** Literal(3.0)
        _eval(u.d(x, scheme="automatic_differentiation:reverse"), {"x": _pts_1d()})
        assert "reverse" in calls and "forward" not in calls

    def test_hessian_scheme_fwd_over_fwd_routes_to_both_jacfwd(self, monkeypatch):
        calls = []
        original = ad_fn

        def spy(mode):
            calls.append(mode)
            return original(mode)

        monkeypatch.setattr("jno.trace_evaluator.ad_fn", spy)
        x, y = _make_2d_vars()
        u = x ** Literal(2.0) + y ** Literal(2.0)
        _eval(
            u.laplacian(x, y, scheme="automatic_differentiation:fwd-over-fwd"),
            {"xy": _pts_2d()},
        )
        assert calls.count("forward") == 2 and "reverse" not in calls

    def test_hessian_scheme_rev_over_fwd_routes_mixed(self, monkeypatch):
        calls = []
        original = ad_fn

        def spy(mode):
            calls.append(mode)
            return original(mode)

        monkeypatch.setattr("jno.trace_evaluator.ad_fn", spy)
        x, y = _make_2d_vars()
        u = x * y + x ** Literal(2.0)
        _eval(
            u.laplacian(x, y, scheme="automatic_differentiation:rev-over-fwd"),
            {"xy": _pts_2d()},
        )
        assert "forward" in calls and "reverse" in calls


# ────────────────────────────────────────────────────────────────────────
# 5. Global default + per-call override
# ────────────────────────────────────────────────────────────────────────


class TestGlobalDefaultAndOverride:
    def test_set_ad_mode_changes_unscoped_scheme(self):
        """A bare 'automatic_differentiation' resolves at parse time."""
        set_ad_mode("forward")
        assert parse_ad_scheme("automatic_differentiation") == "forward"
        set_ad_mode("reverse")
        assert parse_ad_scheme("automatic_differentiation") == "reverse"

    def test_per_call_overrides_global(self):
        set_ad_mode("forward")
        # Explicit suffix beats the global default.
        assert parse_ad_scheme("automatic_differentiation:reverse") == "reverse"

    def test_numerical_default_matches_explicit_after_set(self):
        """After set_ad_mode('forward'), a bare-scheme call must produce
        the same numbers as an explicit ':forward' call."""
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()

        set_ad_mode("forward")
        r_default = jnp.ravel(_eval(u.d(x), {"x": pts}))
        r_explicit = jnp.ravel(_eval(u.d(x, scheme="automatic_differentiation:forward"), {"x": pts}))
        assert jnp.allclose(r_default, r_explicit, atol=1e-6)

    def test_apply_ad_mode_defaults_sets_modes(self):
        """jno.setup(diff_type=...) routes through apply_ad_mode_defaults."""
        set_ad_mode("reverse")
        set_hessian_mode("fwd-over-rev")
        apply_ad_mode_defaults(diff_type="forward", hessian_type="rev-over-rev")
        assert get_ad_mode() == "forward"
        assert get_hessian_mode() == "rev-over-rev"

    def test_apply_ad_mode_defaults_no_kwargs_does_not_clobber(self):
        """Omitting kwargs must not reset state already set, when TOML has no override."""
        set_ad_mode("forward")
        set_hessian_mode("fwd-over-fwd")
        from jno.utils.config import load_config

        cfg = load_config(force=True).get("jno", {})
        apply_ad_mode_defaults()
        if "diff_type" not in cfg:
            assert get_ad_mode() == "forward"
        if "hessian_type" not in cfg:
            assert get_hessian_mode() == "fwd-over-fwd"


# ────────────────────────────────────────────────────────────────────────
# 6. Cross-mode agreement on a vector-valued network output
# ────────────────────────────────────────────────────────────────────────


class TestCrossModeAgreement:
    """All four Hessian compositions must produce the same H within tolerance."""

    def test_hessian_matches_across_modes(self):
        x, y = _make_2d_vars()
        u = (x ** Literal(2.0)) * y + y ** Literal(3.0)
        pts = _pts_2d()
        results = {scheme: _eval(u.hessian(x, y, scheme=scheme), {"xy": pts}) for scheme in HESSIAN_SCHEMES}
        ref = results[HESSIAN_SCHEMES[1]]  # fwd-over-rev
        for scheme, H in results.items():
            assert jnp.allclose(H, ref, atol=1e-4), f"Hessian disagrees for scheme {scheme}"

    def test_first_order_matches_across_modes(self):
        x, y = _make_2d_vars()
        u = FunctionCall(jnp.sin, [x], "sin") * y
        pts = _pts_2d()
        results = {scheme: _eval(u.d(x, scheme=scheme), {"xy": pts}) for scheme in FIRST_ORDER_SCHEMES}
        ref = results[FIRST_ORDER_SCHEMES[1]]  # forward
        for scheme, val in results.items():
            assert jnp.allclose(val, ref, atol=1e-5), f"d disagrees for scheme {scheme}"
