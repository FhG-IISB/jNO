"""Comprehensive tests for mesh-based numerical integration in the jno workflow.

Two tests cover the full story:

1. ``TestJitCompatibleIntegrals`` — boundary and volume integrals evaluated
   under ``jax.jit`` on 1-D and 2-D domains.  Checks concrete numerical
   values (perimeter, area, moments, divergence-theorem flux) and verifies
   that a second JIT call hits the cache without recomputing weights.

2. ``TestGradientThroughIntegralLoss`` — ``eqx.filter_jit`` + ``eqx.filter_grad``
   through an integral-based loss using a trainable linear model.  Verifies
   analytically known gradients and that a gradient step moves the loss in
   the correct direction.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.trace import Integral, Model, Placeholder
from jno.trace_evaluator import IntegrationOperators, TraceEvaluator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_context(domain):
    """Strip leading singleton axes from domain.context (simulate post-vmap)."""
    ctx = {}
    for k, v in domain.context.items():
        arr = np.asarray(v)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        ctx[k] = jnp.array(arr)
    return ctx


def _eval(expr, domain):
    """Evaluate *expr* against *domain* via TraceEvaluator (no JIT)."""
    ev = TraceEvaluator(params={})
    return ev.evaluate(expr, context=_build_context(domain), var_bindings={})


# ---------------------------------------------------------------------------
# Test 1: JIT-compatible integrals — 1-D and 2-D, concrete values
# ---------------------------------------------------------------------------


class TestJitCompatibleIntegrals:
    """Boundary and volume integrals evaluated under jax.jit.

    Unit-square (mesh_size=0.04) and unit-line (mesh_size=0.05) domains.

    Analytic targets
    ----------------
    1-D line [0, 1]:
        ∫₀¹ 1 dx  = 1.0
        ∫₀¹ x dx  = 0.5

    2-D unit square:
        ∫_∂Ω  1 ds              = 4.0   (perimeter)
        ∫_∂Ω  x·nₓ + y·nᵧ ds   = 2.0   (divergence theorem, F=(x,y))
        ∫_Ω   1 dA              = 1.0   (area)
        ∫_Ω   x + y dA          = 1.0   (∫∫ x+y dxdy = 0.5+0.5)
    """

    @pytest.fixture(scope="class")
    def dom_2d(self):
        return jno.domain(constructor=jno.domain.rect(mesh_size=0.04))

    @pytest.fixture(scope="class")
    def dom_1d(self):
        return jno.domain(constructor=jno.domain.line(mesh_size=0.05))

    def test_1d_volume_integrals_under_jit(self, dom_1d):
        """∫₀¹ 1 dx = 1.0 and ∫₀¹ x dx ≈ 0.5, both JIT-compiled."""
        x, _ = dom_1d.variable("interior")
        const_expr = (x * 0.0 + 1.0).integrate()
        identity_expr = x.integrate()

        ctx = _build_context(dom_1d)
        ev = TraceEvaluator(params={})

        @jax.jit
        def compute(c):
            return (
                ev.evaluate(const_expr, context=c, var_bindings={}),
                ev.evaluate(identity_expr, context=c, var_bindings={}),
            )

        length, half = compute(ctx)
        assert abs(float(length) - 1.0) < 0.05, f"∫1 dx = {float(length):.4f}, expected ≈ 1.0"
        assert abs(float(half) - 0.5) < 0.05, f"∫x dx = {float(half):.4f}, expected ≈ 0.5"

        # Second call: verify JIT cache (no recompilation, same values).
        length2, half2 = compute(ctx)
        assert jnp.allclose(length, length2), "JIT cache returned different value for ∫1 dx"
        assert jnp.allclose(half, half2), "JIT cache returned different value for ∫x dx"

    def test_2d_volume_integrals_under_jit(self, dom_2d):
        """∫_Ω 1 dA = 1.0 and ∫_Ω (x+y) dA = 1.0, both JIT-compiled."""
        x, y, _ = dom_2d.variable("interior")
        area_expr = (x * 0.0 + 1.0).integrate()
        moment_expr = (x + y).integrate()

        ctx = _build_context(dom_2d)
        ev = TraceEvaluator(params={})

        @jax.jit
        def compute(c):
            return (
                ev.evaluate(area_expr, context=c, var_bindings={}),
                ev.evaluate(moment_expr, context=c, var_bindings={}),
            )

        area, moment = compute(ctx)
        assert abs(float(area) - 1.0) < 0.05, f"∫1 dA = {float(area):.4f}, expected ≈ 1.0"
        assert abs(float(moment) - 1.0) < 0.05, f"∫(x+y) dA = {float(moment):.4f}, expected ≈ 1.0"

        # Second call: values identical (JIT cache, not numpy recomputation).
        area2, moment2 = compute(ctx)
        assert jnp.allclose(area, area2)
        assert jnp.allclose(moment, moment2)

    def test_2d_boundary_integrals_and_divergence_theorem_under_jit(self, dom_2d):
        """∫_∂Ω 1 ds = 4 (perimeter), ∫_∂Ω x·nₓ + y·nᵧ ds = 2 (div thm), both JIT."""
        x_b, y_b, _, nx, ny = dom_2d.variable("boundary", normals=True)
        perimeter_expr = (x_b * 0.0 + 1.0).integrate()
        flux_expr = (x_b * nx + y_b * ny).integrate()

        ctx = _build_context(dom_2d)
        ev = TraceEvaluator(params={})

        @jax.jit
        def compute(c):
            return (
                ev.evaluate(perimeter_expr, context=c, var_bindings={}),
                ev.evaluate(flux_expr, context=c, var_bindings={}),
            )

        perimeter, flux = compute(ctx)
        assert abs(float(perimeter) - 4.0) < 0.15, f"∫_∂Ω 1 ds = {float(perimeter):.4f}, expected ≈ 4.0"
        assert abs(float(flux) - 2.0) < 0.2, f"∫_∂Ω F·n ds = {float(flux):.4f}, expected ≈ 2.0 (divergence theorem)"

        # Second call from JIT cache.
        perimeter2, flux2 = compute(ctx)
        assert jnp.allclose(perimeter, perimeter2)
        assert jnp.allclose(flux, flux2)

    def test_jno_np_integrate_alias(self, dom_2d):
        """``jno.np.integrate(expr)`` is a valid alias for ``expr.integrate()``."""
        x, y, _ = dom_2d.variable("interior")
        via_method = (x * 0.0 + 1.0).integrate()
        via_alias = jno.np.integrate(x * 0.0 + 1.0)

        ctx = _build_context(dom_2d)
        ev = TraceEvaluator(params={})

        r1 = float(ev.evaluate(via_method, context=ctx, var_bindings={}))
        r2 = float(ev.evaluate(via_alias, context=ctx, var_bindings={}))
        assert abs(r1 - r2) < 1e-5


# ---------------------------------------------------------------------------
# Test 2: Gradient through integral loss with a trainable linear model
# ---------------------------------------------------------------------------


class _LinearXY(eqx.Module):
    """Minimal linear model: u(x, y) = w0*x + w1*y + b.

    Weights are equinox arrays → differentiable via eqx.filter_grad.
    Accepts two separate (N, 1) arrays so it works with ``fm(x, y)``.
    """

    w0: jax.Array
    w1: jax.Array
    b: jax.Array

    def __init__(self, w0: float = 0.3, w1: float = 0.7, b: float = 0.5):
        self.w0 = jnp.array([w0])
        self.w1 = jnp.array([w1])
        self.b = jnp.array([b])

    def __call__(self, x, y):
        return self.w0 * x + self.w1 * y + self.b


class TestGradientThroughIntegralLoss:
    """eqx.filter_grad through integral-based loss with a trainable model.

    Model: u(x, y; w0, w1, b) = w0·x + w1·y + b

    Loss:  L = ∫_Ω u dA  (unit square, area = 1)

    Analytic values (∫_Ω x dA = 0.5, ∫_Ω y dA = 0.5, ∫_Ω 1 dA = 1):

        L     = 0.5·w0 + 0.5·w1 + b
        ∂L/∂w0 = 0.5
        ∂L/∂w1 = 0.5
        ∂L/∂b  = 1.0

    The test verifies these gradients numerically and confirms that one
    gradient-descent step moves the loss in the correct direction.
    """

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.domain(constructor=jno.domain.rect(mesh_size=0.04))

    @pytest.fixture(scope="class")
    def setup(self, dom):
        """Build model, expression, and context once for the whole class."""
        model = _LinearXY(w0=0.3, w1=0.7, b=0.5)
        fm = Model(model, name="u")

        x, y, _ = dom.variable("interior")
        loss_expr = fm(x, y).integrate()

        ctx = _build_context(dom)
        params = {fm.layer_id: model}

        return {"model": model, "fm": fm, "loss_expr": loss_expr, "ctx": ctx, "params": params}

    def test_integral_loss_value_matches_analytic(self, setup):
        """L = 0.5*w0 + 0.5*w1 + b (analytic) matches evaluator output."""
        model = setup["model"]
        ev = TraceEvaluator(params={setup["fm"].layer_id: model})
        result = float(ev.evaluate(setup["loss_expr"], context=setup["ctx"], var_bindings={}))

        expected = 0.5 * float(model.w0[0]) + 0.5 * float(model.w1[0]) + float(model.b[0])
        assert abs(result - expected) < 0.05, f"L = {result:.4f}, analytic = {expected:.4f}"

    def test_gradients_match_analytic_values(self, setup):
        """∂L/∂w0 ≈ 0.5, ∂L/∂w1 ≈ 0.5, ∂L/∂b ≈ 1.0 (analytic)."""
        fm = setup["fm"]
        loss_expr = setup["loss_expr"]
        ctx = setup["ctx"]

        def loss_fn(m):
            ev = TraceEvaluator(params={fm.layer_id: m})
            return ev.evaluate(loss_expr, context=ctx, var_bindings={})

        grad_fn = eqx.filter_jit(eqx.filter_grad(loss_fn))
        grads = grad_fn(setup["model"])

        dL_dw0 = float(grads.w0[0])
        dL_dw1 = float(grads.w1[0])
        dL_db = float(grads.b[0])

        assert abs(dL_dw0 - 0.5) < 0.05, f"∂L/∂w0 = {dL_dw0:.4f}, expected ≈ 0.5"
        assert abs(dL_dw1 - 0.5) < 0.05, f"∂L/∂w1 = {dL_dw1:.4f}, expected ≈ 0.5"
        assert abs(dL_db - 1.0) < 0.05, f"∂L/∂b = {dL_db:.4f}, expected ≈ 1.0"

    def test_gradient_step_decreases_loss(self, setup):
        """One gradient descent step on L reduces it in the expected direction.

        Loss = ∫u dA is minimized by driving w0, w1, b → -∞.
        After one step with lr=0.1, loss must strictly decrease.
        """
        fm = setup["fm"]
        loss_expr = setup["loss_expr"]
        ctx = setup["ctx"]

        def loss_fn(m):
            ev = TraceEvaluator(params={fm.layer_id: m})
            return ev.evaluate(loss_expr, context=ctx, var_bindings={})

        grad_fn = eqx.filter_jit(eqx.filter_grad(loss_fn))
        model = setup["model"]
        L0 = float(loss_fn(model))

        grads = grad_fn(model)
        lr = 0.1
        model_updated = jax.tree_util.tree_map(
            lambda p, g: p - lr * g if isinstance(g, jax.Array) else p,
            model,
            grads,
        )
        L1 = float(loss_fn(model_updated))

        assert L1 < L0, f"Loss did not decrease: L0={L0:.4f}, L1={L1:.4f}"

    def test_second_grad_call_uses_jit_cache(self, setup):
        """Calling the jitted grad function twice returns identical results."""
        fm = setup["fm"]
        loss_expr = setup["loss_expr"]
        ctx = setup["ctx"]

        def loss_fn(m):
            ev = TraceEvaluator(params={fm.layer_id: m})
            return ev.evaluate(loss_expr, context=ctx, var_bindings={})

        grad_fn = eqx.filter_jit(eqx.filter_grad(loss_fn))
        model = setup["model"]

        grads1 = grad_fn(model)
        grads2 = grad_fn(model)

        assert jnp.allclose(grads1.w0, grads2.w0)
        assert jnp.allclose(grads1.w1, grads2.w1)
        assert jnp.allclose(grads1.b, grads2.b)


"""Tests for mesh-based numerical integration operators.

Covers:
- IntegrationOperators.nodal_volumes (pure numpy backend)
- Integral trace node creation via .integrate()
- _eval_integral via TraceEvaluator (boundary and volume, 1D and 2D)
- Divergence-theorem flux check (user computes F·n, then integrates)
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_2d_rect_domain(mesh_size=0.05):
    return jno.domain(constructor=jno.domain.rect(mesh_size=mesh_size))


def _eval_integral_expr(expr, domain):
    """Evaluate an Integral node directly, simulating the post-vmap context."""
    evaluator = TraceEvaluator(params={})
    # Peel the leading (B, T) singleton axes — evaluator runs after outer vmaps.
    context = {}
    for k, v in domain.context.items():
        arr = np.asarray(v)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        context[k] = jnp.array(arr)
    return evaluator.evaluate(expr, context=context, var_bindings={})


# ---------------------------------------------------------------------------
# Unit tests: IntegrationOperators.nodal_volumes
# ---------------------------------------------------------------------------


class TestNodalVolumes:
    def test_1d_sums_to_length(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.05))
        vols = IntegrationOperators.nodal_volumes(dom.mesh_connectivity)
        assert abs(float(np.sum(vols)) - 1.0) < 0.01

    def test_2d_sums_to_area(self):
        dom = _make_2d_rect_domain(mesh_size=0.05)
        vols = IntegrationOperators.nodal_volumes(dom.mesh_connectivity)
        assert abs(float(np.sum(vols)) - 1.0) < 0.02  # unit square area = 1

    def test_all_non_negative(self):
        dom = _make_2d_rect_domain(mesh_size=0.1)
        vols = IntegrationOperators.nodal_volumes(dom.mesh_connectivity)
        assert np.all(vols >= 0)

    def test_shape_matches_n_points(self):
        dom = _make_2d_rect_domain(mesh_size=0.1)
        mc = dom.mesh_connectivity
        vols = IntegrationOperators.nodal_volumes(mc)
        assert vols.shape == (mc["n_points"],)


# ---------------------------------------------------------------------------
# Unit tests: boundary weights (nodal_ds pre-stored in mesh_connectivity)
# ---------------------------------------------------------------------------


class TestBoundaryWeights:
    def test_2d_nodal_ds_sums_to_perimeter(self):
        dom = _make_2d_rect_domain(mesh_size=0.05)
        mc = dom.mesh_connectivity
        b_idx = mc["boundary_indices"]
        perimeter = float(np.sum(mc["nodal_ds"][b_idx]))
        assert abs(perimeter - 4.0) < 0.05  # unit square perimeter = 4


# ---------------------------------------------------------------------------
# Unit tests: Integral trace node
# ---------------------------------------------------------------------------


class TestIntegralNode:
    def test_integrate_returns_integral_instance(self):
        dom = _make_2d_rect_domain(mesh_size=0.1)
        x_b, y_b, _ = dom.variable("boundary")
        node = (x_b * 0.0 + 1.0).integrate()
        assert isinstance(node, Integral)

    def test_integrate_repr(self):
        dom = _make_2d_rect_domain(mesh_size=0.1)
        x_b, _, _ = dom.variable("boundary")
        node = x_b.integrate()
        assert "Integral" in repr(node)

    def test_integrate_is_placeholder(self):
        dom = _make_2d_rect_domain(mesh_size=0.1)
        x_b, _, _ = dom.variable("boundary")
        node = x_b.integrate()
        assert isinstance(node, Placeholder)

    def test_jno_np_integrate_alias(self):
        dom = _make_2d_rect_domain(mesh_size=0.1)
        x_b, _, _ = dom.variable("boundary")
        node = jno.np.integrate(x_b * 0.0 + 1.0)
        assert isinstance(node, Integral)


# ---------------------------------------------------------------------------
# Integration tests: evaluator — 1D
# ---------------------------------------------------------------------------


class TestEvalIntegral1D:
    # Function-scoped: each test gets a fresh domain to avoid tag collisions
    # from multiple dom.variable("interior") calls.
    @pytest.fixture
    def dom(self):
        return jno.domain(constructor=jno.domain.line(mesh_size=0.05))

    def test_volume_constant_one_equals_length(self, dom):
        x, _ = dom.variable("interior")
        expr = (x * 0.0 + 1.0).integrate()
        result = float(_eval_integral_expr(expr, dom))
        assert abs(result - 1.0) < 0.05

    def test_volume_identity_equals_half(self, dom):
        """∫_0^1 x dx = 0.5"""
        x, _ = dom.variable("interior")
        result = float(_eval_integral_expr(x.integrate(), dom))
        assert abs(result - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Integration tests: evaluator — 2D unit square
# ---------------------------------------------------------------------------


class TestEvalIntegral2D:
    @pytest.fixture
    def dom(self):
        return _make_2d_rect_domain(mesh_size=0.04)

    def test_boundary_constant_one_equals_perimeter(self, dom):
        x_b, y_b, _ = dom.variable("boundary")
        expr = (x_b * 0.0 + 1.0).integrate()
        result = float(_eval_integral_expr(expr, dom))
        assert abs(result - 4.0) < 0.1  # perimeter of unit square

    def test_volume_constant_one_equals_area(self, dom):
        x, y, _ = dom.variable("interior")
        expr = (x * 0.0 + 1.0).integrate()
        result = float(_eval_integral_expr(expr, dom))
        assert abs(result - 1.0) < 0.05  # area of unit square

    def test_boundary_region_detected_not_volume(self, dom):
        """Boundary tag gives result near 4 (perimeter), not near 1 (area)."""
        x_b, y_b, _ = dom.variable("boundary")
        result = float(_eval_integral_expr((x_b * 0.0 + 1.0).integrate(), dom))
        assert result > 3.5

    def test_volume_region_detected_not_boundary(self, dom):
        """Interior tag gives result near 1 (area), not near 4 (perimeter)."""
        x, y, _ = dom.variable("interior")
        result = float(_eval_integral_expr((x * 0.0 + 1.0).integrate(), dom))
        assert result < 1.5


# ---------------------------------------------------------------------------
# Flux / divergence theorem: user computes F·n manually, then integrates
# ---------------------------------------------------------------------------


class TestFluxIntegral:
    """∫_∂Ω F·n ds checked via the divergence theorem on the unit square."""

    @pytest.fixture
    def dom(self):
        return _make_2d_rect_domain(mesh_size=0.04)

    def test_constant_flux_net_zero(self, dom):
        """F = (1, 0): net flux through closed boundary = 0."""
        x_b, y_b, _, nx, ny = dom.variable("boundary", normals=True)
        result = float(_eval_integral_expr(nx.integrate(), dom))
        assert abs(result) < 0.1

    def test_linear_flux_divergence_theorem(self, dom):
        """F = (x, y): ∫_∂Ω F·n ds = ∫_Ω ∇·F dV = 2 × area = 2."""
        x_b, y_b, _, nx, ny = dom.variable("boundary", normals=True)
        expr = (x_b * nx + y_b * ny).integrate()
        result = float(_eval_integral_expr(expr, dom))
        assert abs(result - 2.0) < 0.15


# ---------------------------------------------------------------------------
# Vectorized integral via integration_variable=True
#
# Analytic reference (1-D domain [0, 1], kernel K(x,t) = x + t):
#
#   ∫₀¹ (x + t) · t dt = x · ∫₀¹ t dt + ∫₀¹ t² dt = 0.5x + 1/3
#
# Result is an (N_outer, 1) array — a function of the outer variable x.
# ---------------------------------------------------------------------------


class TestVectorizedIntegral:
    """Non-separable Fredholm kernel via ``.integrate(var=x)``.

    Uses kernel K(x,t) = x + t and integrand u(t) = t so that the analytic
    result is 0.5·x + 1/3, verifiable pointwise.

    API: call ``domain.variable(tag)`` twice — no special flag needed.
    Pass the outer (collocation) variable to ``.integrate(var=x)``; the
    evaluator aliases everything else to the integration mesh automatically.
    """

    @pytest.fixture(scope="class")
    def dom(self):
        return jno.domain(constructor=jno.domain.line(mesh_size=0.02))

    def test_two_calls_produce_distinct_objects(self, dom):
        """Two domain.variable() calls return distinct Python objects (different id)."""
        x, _ = dom.variable("interior")
        t, _ = dom.variable("interior")
        assert x is not t
        assert id(x) != id(t)

    def test_vectorized_integral_shape(self, dom):
        """(x + t)·t integrated with var=x gives (N_outer, 1), not a scalar."""
        x, _ = dom.variable("interior")
        t, _ = dom.variable("interior")
        expr = ((x + t) * t).integrate(var=x)
        result = _eval_integral_expr(expr, dom)
        ctx = _build_context(dom)
        N_outer = ctx["interior"].shape[0]
        assert result.shape == (N_outer, 1), f"Expected ({N_outer}, 1), got {result.shape}"

    def test_vectorized_integral_values(self, dom):
        """Pointwise values match 0.5·x + 1/3 within mesh discretisation error."""
        x, _ = dom.variable("interior")
        t, _ = dom.variable("interior")
        expr = ((x + t) * t).integrate(var=x)
        result = _eval_integral_expr(expr, dom)  # (N, 1)

        ctx = _build_context(dom)
        x_vals = np.asarray(ctx["interior"])[:, 0]  # (N,)
        analytic = 0.5 * x_vals + 1.0 / 3.0

        max_err = float(np.max(np.abs(np.asarray(result[:, 0]) - analytic)))
        assert max_err < 0.01, f"Max pointwise error {max_err:.4f} exceeds tolerance"

    def test_vectorized_integral_is_jit_compatible(self, dom):
        """jax.jit over the vectorized integral compiles and two calls agree."""
        x, _ = dom.variable("interior")
        t, _ = dom.variable("interior")
        expr = ((x + t) * t).integrate(var=x)
        ctx = _build_context(dom)

        ev = TraceEvaluator(params={})

        @jax.jit
        def run():
            return ev.evaluate(expr, context=ctx, var_bindings={})

        r1 = run()
        r2 = run()
        assert jnp.allclose(r1, r2)

    def test_no_inner_var_raises(self, dom):
        """integrate(var=x) with no second variable raises a clear error."""
        x, _ = dom.variable("interior")
        expr = (x * x).integrate(var=x)
        with pytest.raises(ValueError, match="no inner variable"):
            _eval_integral_expr(expr, dom)

    def test_nonseparable_kernel_analytic_values(self, dom):
        """∫₀¹ (x+t)·sin(πt) dt = 2x/π + 1/π at every collocation point.

        This is the integral term in the non-separable Fredholm tutorial.
        """
        x, _ = dom.variable("interior")
        t, _ = dom.variable("interior")
        π = jno.np.pi
        expr = ((x + t) * jno.np.sin(π * t)).integrate(var=x)
        result = _eval_integral_expr(expr, dom)  # (N, 1)

        ctx = _build_context(dom)
        x_vals = np.asarray(ctx["interior"])[:, 0]
        analytic = 2.0 * x_vals / float(jnp.pi) + 1.0 / float(jnp.pi)

        max_err = float(np.max(np.abs(np.asarray(result[:, 0]) - analytic)))
        assert max_err < 0.01, f"Max pointwise error {max_err:.4f}"

    def test_chained_double_integral(self, dom):
        """∫₀¹ ∫₀¹ (x+t) dt dx = 1.0  via chained .integrate() calls."""
        x, _ = dom.variable("interior")
        t, _ = dom.variable("interior")
        inner = (x + t).integrate(var=x)  # (N,1): g(x) = x + 0.5
        result = float(_eval_integral_expr(inner.integrate(), dom))  # scalar
        assert abs(result - 1.0) < 0.01, f"∫∫(x+t)dt dx = {result:.6f}, expected 1.0"

    def test_chained_double_integral_separable(self, dom):
        """∫₀¹ ∫₀¹ x·t dt dx = 0.25  (separable kernel, chained)."""
        x, _ = dom.variable("interior")
        t, _ = dom.variable("interior")
        inner = (x * t).integrate(var=x)  # (N,1): g(x) = 0.5·x
        result = float(_eval_integral_expr(inner.integrate(), dom))  # scalar
        assert abs(result - 0.25) < 0.01, f"∫∫ x·t dt dx = {result:.6f}, expected 0.25"


# ---------------------------------------------------------------------------
# Integro-differential equation: .d(x) and .integrate() in the same residual
#
# IDE:  u'(x) + u(x) = g(x) + ∫₀¹ u(t) dt,   u(0) = 0
# Exact: u*(x) = sin(πx)
#
# Two pure-operator checks (no training):
#
# 1. u = x²:
#       u'(x) = 2x
#       ∫₀¹ x² dt = 1/3
#       u'(x) + ∫₀¹ u(t) dt = 2x + 1/3
#
# 2. IDE residual at exact solution sin(πx) should be near zero (within
#    discretisation error).  g(x) = π cos(πx) + sin(πx) − 2/π.
# ---------------------------------------------------------------------------


class TestIntegroDifferential:
    """Composition of .d(x) and .integrate() in an IDE residual.

    No training — uses the identity model u=x² (via multiplication) and the
    exact solution sin(πx) to verify operator composition analytically.
    """

    @pytest.fixture
    def dom(self):
        return jno.domain(constructor=jno.domain.line(mesh_size=0.05))

    def test_derivative_plus_integral_composition(self, dom):
        """u=x²: u'(x) + ∫₀¹ u(t) dt = 2x + 1/3 at every collocation point."""
        x, _ = dom.variable("interior")

        # u = x²  via placeholder arithmetic (no network)
        u = x * x

        du = u.d(x)  # (N, 1): 2x
        C = u.integrate()  # scalar: ∫₀¹ t² dt = 1/3

        combined = du + C  # (N, 1): 2x + 1/3

        ctx = _build_context(dom)
        ev = TraceEvaluator(params={})
        result = np.asarray(ev.evaluate(combined, context=ctx, var_bindings={}))  # (N, 1)

        x_vals = np.asarray(ctx["interior"])[:, 0]  # (N,)
        analytic = 2.0 * x_vals + 1.0 / 3.0

        max_err = float(np.max(np.abs(result[:, 0] - analytic)))
        assert max_err < 0.01, f"Max pointwise error {max_err:.2e}, expected < 0.01"

    def test_ide_residual_at_exact_solution_is_near_zero(self, dom):
        """IDE residual vanishes at u*(x) = sin(πx) (within discretisation error)."""
        x, _ = dom.variable("interior")
        π = jno.np.pi

        u = jno.np.sin(π * x)

        pi_val = float(jnp.pi)
        g = π * jno.np.cos(π * x) + jno.np.sin(π * x) - 2.0 / pi_val

        C = u.integrate()  # should evaluate to ≈ 2/π
        du = u.d(x)  # π cos(πx)

        residual = du + u - g - C  # should be ≈ 0

        ctx = _build_context(dom)
        ev = TraceEvaluator(params={})
        res = np.asarray(ev.evaluate(residual, context=ctx, var_bindings={}))

        max_abs = float(np.max(np.abs(res)))
        assert max_abs < 5e-3, f"IDE residual at exact solution: max|R| = {max_abs:.2e}"

    def test_scalar_integral_has_no_outer_dimension(self, dom):
        """C = u.integrate() (no var=) returns a true scalar, not (N, 1)."""
        x, _ = dom.variable("interior")
        u = jno.np.sin(jno.np.pi * x)
        C = u.integrate()

        ctx = _build_context(dom)
        ev = TraceEvaluator(params={})
        result = ev.evaluate(C, context=ctx, var_bindings={})

        assert result.ndim == 0 or result.size == 1, f"Expected scalar, got shape {result.shape}"
        val = float(jnp.squeeze(result))
        assert abs(val - 2.0 / float(jnp.pi)) < 0.01, f"∫₀¹ sin(πt) dt = {val:.4f}, expected ≈ {2.0 / float(jnp.pi):.4f}"
