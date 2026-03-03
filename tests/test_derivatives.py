"""Tests for the method-style derivative API on Placeholder.

Covers:
  - Structural node creation (.d, .diff, .laplacian, .hessian)
  - Numerical correctness against analytic results (AD scheme)
  - Parity between method API and jnn.grad / jnn.laplacian / jnn.hessian
  - Chained second-order derivatives  u.d(x).d(x)
  - Full Hessian matrix  u.hessian(x, y)

AD mode summary (from trace_evaluator._eval_jacobian / _eval_hessian):
  .d(x)  single-variable    -- jax.grad (reverse-mode AD), vmapped over N pts
  .d(x).d(y) nested         -- jax.grad (reverse-mode AD) applied twice
  .laplacian(*vars)          -- jax.hessian (forward-over-reverse = jacfwd ∘ jacrev), vmapped
  .hessian(*vars)            -- jax.hessian (forward-over-reverse), vmapped
  temporal .d(t)             -- jax.grad (reverse-mode) w.r.t. scalar time, vmapped
  scheme='finite_difference' -- central-difference stencils, no AD at all
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

import jno.numpy as jnn
from jno.trace import (
    Jacobian,
    Hessian,
    Literal,
    OperationDef,
    Variable,
    FunctionCall,
)
from jno.trace_evaluator import TraceEvaluator
from tests.conftest import make_var


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────


def _eval(expr, context):
    """Evaluate expr directly via TraceEvaluator with (N, D) context.

    TraceEvaluator.evaluate() expects per-point context shapes like
    (N, D) — no outer batch (B) or time (T) axes.  This is correct for
    unit tests that verify derivative values per point.
    """
    ev = TraceEvaluator({})
    return ev.evaluate(expr, context, {}, key=jax.random.PRNGKey(0))


def _pts_1d(n=20, lo=0.1, hi=0.9):
    """N points in (lo, hi) as (N, 1) array — avoids boundary singularities."""
    return jnp.linspace(lo, hi, n).reshape(n, 1)


def _pts_2d(n=8):
    """n² grid points in [0.1, 0.9]² as (n², 2) array."""
    xs = jnp.linspace(0.1, 0.9, n)
    ys = jnp.linspace(0.1, 0.9, n)
    gx, gy = jnp.meshgrid(xs, ys)
    return jnp.stack([gx.ravel(), gy.ravel()], axis=-1)


def _make_2d_vars():
    """Return (x, y) Variable pair sharing tag 'xy' on a 2-D domain.

    x reads column 0, y reads column 1 of the same (N, 2) context array.
    Both use tag 'xy' so a single context entry {'xy': pts} serves both.
    """
    from tests.conftest import MockDomain

    d = MockDomain(tags=["xy"], dim=2)  # context['xy'] shape (10, 2)
    x = Variable("xy", [0, 1], domain=d)  # dim[0]=0 → ∂/∂x₀
    y = Variable("xy", [1, 2], domain=d)  # dim[0]=1 → ∂/∂x₁
    return x, y


# ───────────────────────────────────────────────────────────────────────────────
# 1. Structural tests — verify correct graph node types are created
# ───────────────────────────────────────────────────────────────────────────────


class TestDerivativeNodes:
    def test_d_returns_jacobian(self):
        x = make_var("x")
        u = x * x
        deriv = u.d(x)
        assert isinstance(deriv, Jacobian)
        assert deriv.variables == [x]
        assert deriv.scheme == "automatic_differentiation"

    def test_diff_alias_returns_jacobian(self):
        x = make_var("x")
        u = x * x
        assert isinstance(u.diff(x), Jacobian)

    def test_d_chaining_returns_nested_jacobians(self):
        x = make_var("x")
        u = x ** Literal(3.0)
        u_xx = u.d(x).d(x)
        assert isinstance(u_xx, Jacobian)
        assert isinstance(u_xx.target, Jacobian)

    def test_laplacian_1d_returns_hessian_trace(self):
        x = make_var("x")
        u = x * x
        lap = u.laplacian(x)
        assert isinstance(lap, Hessian)
        assert lap.trace is True
        assert len(lap.variables) == 1

    def test_laplacian_multivar_returns_hessian_trace(self):
        x, y = _make_2d_vars()
        u = x * x  # expression node; no eval needed here
        lap = u.laplacian(x, y)
        assert isinstance(lap, Hessian)
        assert lap.trace is True
        assert len(lap.variables) == 2

    def test_hessian_returns_hessian_no_trace(self):
        x = make_var("x")
        u = x * x
        H = u.hessian(x)
        assert isinstance(H, Hessian)
        assert H.trace is False

    def test_d_custom_scheme_propagates(self):
        x = make_var("x")
        u = x * x
        deriv = u.d(x, scheme="finite_difference")
        assert isinstance(deriv, Jacobian)
        assert deriv.scheme == "finite_difference"

    def test_laplacian_custom_scheme_propagates(self):
        x = make_var("x")
        u = x * x
        lap = u.laplacian(x, scheme="finite_difference")
        assert isinstance(lap, Hessian)
        assert lap.scheme == "finite_difference"


# ───────────────────────────────────────────────────────────────────────────────
# 2. Parity — method API creates the same node type as jnn.grad / jnn.laplacian
# ───────────────────────────────────────────────────────────────────────────────


class TestMethodParityWithFunctionAPI:
    def test_d_same_node_type_as_jnn_grad(self):
        x = make_var("x")
        u = x ** Literal(2.0)
        via_method = u.d(x)
        via_fn = jnn.grad(u, x)
        assert type(via_method) is type(via_fn)
        assert via_method.variables == via_fn.variables
        assert via_method.scheme == via_fn.scheme

    def test_laplacian_same_node_as_jnn_laplacian(self):
        x = make_var("x")
        u = x ** Literal(2.0)
        via_method = u.laplacian(x)
        via_fn = jnn.laplacian(u, [x])
        assert type(via_method) is type(via_fn)
        assert via_method.trace == via_fn.trace
        assert via_method.variables == via_fn.variables
        assert via_method.scheme == via_fn.scheme

    def test_d_and_jnn_grad_give_same_values(self):
        """u.d(x) and jnn.grad(u, x) must produce identical numerical output."""
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        r1 = jnp.ravel(_eval(u.d(x), {"x": pts}))
        r2 = jnp.ravel(_eval(jnn.grad(u, x), {"x": pts}))
        assert jnp.allclose(r1, r2, atol=1e-5)

    def test_laplacian_and_jnn_laplacian_give_same_values(self):
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        r1 = jnp.ravel(_eval(u.laplacian(x), {"x": pts}))
        r2 = jnp.ravel(_eval(jnn.laplacian(u, [x]), {"x": pts}))
        assert jnp.allclose(r1, r2, atol=1e-5)


# ───────────────────────────────────────────────────────────────────────────────
# 3. First-order derivatives (AD = jax.grad reverse-mode, vmapped)
# ───────────────────────────────────────────────────────────────────────────────


class TestFirstDerivativeAD:
    """d/dx of known 1-D functions.

    AD mode: jax.grad (reverse-mode), vmapped over N points.
    """

    def test_d_polynomial_cubic(self):
        """d/dx(x³) = 3x²."""
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        result = jnp.ravel(_eval(u.d(x), {"x": pts}))
        expected = 3.0 * pts[:, 0] ** 2
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_d_sin(self):
        """d/dx(sin(x)) = cos(x)."""
        x = make_var("x")
        u = FunctionCall(jnp.sin, [x], "sin")
        pts = _pts_1d()
        result = jnp.ravel(_eval(u.d(x), {"x": pts}))
        expected = jnp.cos(pts[:, 0])
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_diff_alias_same_values(self):
        """u.diff(x) is an alias for u.d(x) and must give identical values."""
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        r_d = jnp.ravel(_eval(u.d(x), {"x": pts}))
        r_diff = jnp.ravel(_eval(u.diff(x), {"x": pts}))
        assert jnp.allclose(r_d, r_diff)


# ───────────────────────────────────────────────────────────────────────────────
# 4. Chained second-order derivatives via .d(x).d(x)
#    Each level uses jax.grad (reverse-mode AD), vmapped.
# ───────────────────────────────────────────────────────────────────────────────


class TestSecondDerivativeChain:
    def test_d_chain_polynomial_cubic(self):
        """d²/dx²(x³) = 6x."""
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        result = jnp.ravel(_eval(u.d(x).d(x), {"x": pts}))
        expected = 6.0 * pts[:, 0]
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_d_chain_sin(self):
        """d²/dx²(sin(x)) = -sin(x)."""
        x = make_var("x")
        u = FunctionCall(jnp.sin, [x], "sin")
        pts = _pts_1d()
        result = jnp.ravel(_eval(u.d(x).d(x), {"x": pts}))
        expected = -jnp.sin(pts[:, 0])
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_d_chain_matches_jnn_grad_chain(self):
        """u.d(x).d(x) must give the same values as jnn.grad(jnn.grad(u,x),x)."""
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        r_method = jnp.ravel(_eval(u.d(x).d(x), {"x": pts}))
        r_fn = jnp.ravel(_eval(jnn.grad(jnn.grad(u, x), x), {"x": pts}))
        assert jnp.allclose(r_method, r_fn, atol=1e-5)


# ───────────────────────────────────────────────────────────────────────────────
# 5. Laplacian (AD = jax.hessian = forward-over-reverse)
# ───────────────────────────────────────────────────────────────────────────────


class TestLaplacianAD:
    def test_laplacian_1d_polynomial(self):
        """∇²(x³) = 6x in 1-D.

        AD mode: jax.hessian (forward-over-reverse, jacfwd ∘ jacrev).
        """
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        result = jnp.ravel(_eval(u.laplacian(x), {"x": pts}))
        expected = 6.0 * pts[:, 0]
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_laplacian_2d_sum_of_squares(self):
        """∇²(x² + y²) = 2 + 2 = 4 everywhere in 2-D.

        Variables x and y share tag 'xy'; x reads col 0, y reads col 1.
        AD mode: jax.hessian (forward-over-reverse) diagonal trace.
        """
        x, y = _make_2d_vars()
        # u = x² + y² using direct Variable arithmetic
        u = x * x + y * y

        pts = _pts_2d()
        result = jnp.ravel(_eval(u.laplacian(x, y), {"xy": pts}))
        expected = jnp.full(result.shape, 4.0)
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_laplacian_method_matches_jnn_laplacian(self):
        """u.laplacian(x) must give identical values to jnn.laplacian(u, [x])."""
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        r_method = jnp.ravel(_eval(u.laplacian(x), {"x": pts}))
        r_fn = jnp.ravel(_eval(jnn.laplacian(u, [x]), {"x": pts}))
        assert jnp.allclose(r_method, r_fn, atol=1e-5)


# ───────────────────────────────────────────────────────────────────────────────
# 6. Full Hessian matrix (AD = jax.hessian = forward-over-reverse)
# ───────────────────────────────────────────────────────────────────────────────


class TestHessianAD:
    def test_hessian_1d(self):
        """Hessian of x³ in 1-D — full 1×1 matrix per point, equals 6x.

        AD mode: jax.hessian (forward-over-reverse).
        """
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        result = jnp.ravel(_eval(u.hessian(x), {"x": pts}))
        expected = 6.0 * pts[:, 0]
        assert jnp.allclose(result, expected, atol=1e-3)

    def test_hessian_is_symmetric(self):
        """H[i,j] == H[j,i] for smooth u.

        u = x²*y  →  H = [[2y, 2x], [2x, 0]]
        """
        x, y = _make_2d_vars()
        # u = x²*y using direct Variable arithmetic
        u = x * x * y

        pts = _pts_2d(n=4)
        H = _eval(u.hessian(x, y), {"xy": pts})  # shape (N, 2, 2)
        # Symmetry: H[i, 0, 1] == H[i, 1, 0]
        assert jnp.allclose(H[:, 0, 1], H[:, 1, 0], atol=1e-4), f"Hessian not symmetric: max off-diag diff = {jnp.max(jnp.abs(H[:,0,1] - H[:,1,0]))}"

    def test_hessian_2d_known_values(self):
        """u = x² + y²  →  H = [[2, 0], [0, 2]] everywhere."""
        x, y = _make_2d_vars()
        # u = x² + y² using direct Variable arithmetic
        u = x * x + y * y

        pts = _pts_2d(n=4)
        H = _eval(u.hessian(x, y), {"xy": pts})  # shape (N, 2, 2)
        N = H.shape[0]
        assert jnp.allclose(H[:, 0, 0], jnp.full(N, 2.0), atol=1e-3)
        assert jnp.allclose(H[:, 1, 1], jnp.full(N, 2.0), atol=1e-3)
        assert jnp.allclose(H[:, 0, 1], jnp.zeros(N), atol=1e-3)
        assert jnp.allclose(H[:, 1, 0], jnp.zeros(N), atol=1e-3)


# ───────────────────────────────────────────────────────────────────────────────
# 7. Integration with reduction operators (.mse, .mean)
#    When called through evaluate() the derivative gives (N,) and .mse
#    reduces over all N points to a scalar.
# ───────────────────────────────────────────────────────────────────────────────


class TestDerivativeIntegration:
    def test_d_then_mse_reduces_to_scalar(self):
        """u.d(x).mse evaluated over N points must collapse to scalar."""
        x = make_var("x")
        u = x ** Literal(2.0)
        pts = _pts_1d()
        result = _eval(u.d(x).mse, {"x": pts})
        assert result.ndim == 0 or result.shape == ()

    def test_d_chain_then_mse_reduces_to_scalar(self):
        """u.d(x).d(x).mse evaluated over N pts must collapse to scalar."""
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        result = _eval(u.d(x).d(x).mse, {"x": pts})
        assert result.ndim == 0 or result.shape == ()

    def test_laplacian_then_mse_reduces_to_scalar(self):
        """u.laplacian(x).mse over N pts must collapse to scalar."""
        x = make_var("x")
        u = x ** Literal(3.0)
        pts = _pts_1d()
        result = _eval(u.laplacian(x).mse, {"x": pts})
        assert result.ndim == 0 or result.shape == ()

    def test_mse_value_correct(self):
        """MSE of d/dx(x²) = (2x) should equal mean(4x²)."""
        x = make_var("x")
        u = x ** Literal(2.0)
        pts = _pts_1d()
        result = float(_eval(u.d(x).mse, {"x": pts}))
        expected = float(jnp.mean(4.0 * pts[:, 0] ** 2))
        assert abs(result - expected) < 1e-4


# ───────────────────────────────────────────────────────────────────────────────
# 8. Finite-difference scheme on a real mesh (L-shaped domain)
#    and temporal derivative via AD
#
#    FD uses P1-FEM stencils on the unstructured triangle mesh, so it is
#    approximate.  We use a fine mesh_size=0.05 and compare against
#    analytic values with tolerances appropriate for O(h) / O(h²) FD.
#    Boundary-adjacent stencils are less accurate, so we check the *mean*
#    error over interior-only vertices (excluding boundary_indices).
# ───────────────────────────────────────────────────────────────────────────────


def _build_l_domain():
    """Build an L-shaped domain with fine mesh and return (dom, x, y, t, pts, interior_mask)."""
    import jno

    dom = jno.domain(constructor=jno.domain.l_shape(size=1.0, mesh_size=0.05))
    (x, y, t) = dom.variable("interior")

    pts = jnp.array(dom.context["interior"][0, 0])  # (N, 2)
    bi = set(np.array(dom.mesh_connectivity["boundary_indices"]).tolist())
    interior_mask = np.array([i not in bi for i in range(pts.shape[0])])
    return dom, x, y, t, pts, interior_mask


class TestFiniteDifferenceOnLShape:
    """FD derivatives on an L-shaped unstructured triangular mesh."""

    @pytest.fixture(autouse=True, scope="class")
    def _domain(self, request):
        """Build the L-shaped domain once for all tests in this class."""
        dom, x, y, t, pts, interior_mask = _build_l_domain()
        request.cls.dom = dom
        request.cls.x = x
        request.cls.y = y
        request.cls.t = t
        request.cls.pts = pts
        request.cls.interior_mask = interior_mask

    # -- helpers --
    def _ctx(self):
        return {"interior": self.pts, "__time__": jnp.array([1.0])}

    def test_fd_gradient_dx(self):
        """∂(x² + y²)/∂x = 2x on the L-shaped mesh (FD vs analytic).

        FD scheme: P1-FEM gradient on triangles, O(h) accuracy.
        """
        u = self.x * self.x + self.y * self.y
        du_dx_fd = _eval(u.d(self.x, scheme="finite_difference"), self._ctx())
        expected = 2.0 * self.pts[:, 0:1]
        err = jnp.abs(du_dx_fd - expected)
        # Interior mean error should be small (< 0.02 for mesh_size=0.05)
        assert float(jnp.mean(err[self.interior_mask])) < 0.02

    def test_fd_gradient_dy(self):
        """∂(x² + y²)/∂y = 2y on the L-shaped mesh (FD vs analytic)."""
        u = self.x * self.x + self.y * self.y
        du_dy_fd = _eval(u.d(self.y, scheme="finite_difference"), self._ctx())
        expected = 2.0 * self.pts[:, 1:2]
        err = jnp.abs(du_dy_fd - expected)
        assert float(jnp.mean(err[self.interior_mask])) < 0.02

    def test_fd_gradient_matches_ad(self):
        """FD and AD spatial gradients should agree within O(h) tolerance."""
        u = self.x * self.x + self.y * self.y
        ctx = self._ctx()
        fd = _eval(u.d(self.x, scheme="finite_difference"), ctx)
        ad = _eval(u.d(self.x), ctx)
        err = jnp.abs(fd - ad)
        assert float(jnp.mean(err[self.interior_mask])) < 0.02

    def test_fd_laplacian(self):
        """∇²(x² + y²) = 4 everywhere (FD vs analytic).

        Second-order FD on unstructured meshes is O(h) at best; boundary
        stencils are one-sided. We check interior-mean error only.
        """
        u = self.x * self.x + self.y * self.y
        lap = _eval(u.laplacian(self.x, self.y, scheme="finite_difference"), self._ctx())
        err = jnp.abs(lap - 4.0)
        assert float(jnp.mean(err[self.interior_mask])) < 0.5

    def test_fd_d2_shorthand(self):
        """u.d2(x, scheme='finite_difference') must equal u.laplacian(x, scheme='finite_difference') for 1-D slice."""
        u = self.x * self.x  # ∂²(x²)/∂x² = 2
        ctx = self._ctx()
        d2 = _eval(u.d2(self.x, scheme="finite_difference"), ctx)
        lap = _eval(u.laplacian(self.x, scheme="finite_difference"), ctx)
        assert jnp.allclose(d2, lap, atol=1e-5)

    def test_fd_returns_correct_shape(self):
        """FD gradient must return (N, 1) matching the AD convention."""
        u = self.x * self.x
        result = _eval(u.d(self.x, scheme="finite_difference"), self._ctx())
        N = self.pts.shape[0]
        assert result.shape == (N, 1)


class TestTemporalDerivative:
    """Time derivative ∂u/∂t via AD on expressions involving a temporal variable."""

    def test_du_dt_linear_in_t(self):
        """u = x * t  ⇒  ∂u/∂t = x (reverse-mode AD w.r.t. scalar time)."""
        from tests.conftest import MockDomain

        d = MockDomain(tags=["x"], dim=1)
        d.context["__time__"] = jnp.zeros((1, 1))
        x = Variable("x", [0, 1], domain=d, axis="spatial")
        t = Variable("__time__", [0, 1], domain=d, axis="temporal")

        u = x * t
        pts = jnp.linspace(0.1, 0.9, 20).reshape(20, 1)
        t_val = jnp.array([3.0])

        result = _eval(u.d(t), {"x": pts, "__time__": t_val})
        expected = pts
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_du_dt_quadratic_in_t(self):
        """u = t²  ⇒  ∂u/∂t = 2t (reverse-mode AD)."""
        from tests.conftest import MockDomain

        d = MockDomain(tags=["x"], dim=1)
        d.context["__time__"] = jnp.zeros((1, 1))
        x = Variable("x", [0, 1], domain=d, axis="spatial")
        t = Variable("__time__", [0, 1], domain=d, axis="temporal")

        u = t * t  # t²
        pts = jnp.ones((5, 1))
        t_val = jnp.array([4.0])

        result = _eval(u.d(t), {"x": pts, "__time__": t_val})
        expected = jnp.full((5, 1), 2.0 * 4.0)  # 2t = 8
        assert jnp.allclose(result, expected, atol=1e-4)

    def test_du_dt_returns_correct_shape(self):
        """Time derivative must return (N, 1)."""
        from tests.conftest import MockDomain

        d = MockDomain(tags=["x"], dim=1)
        d.context["__time__"] = jnp.zeros((1, 1))
        x = Variable("x", [0, 1], domain=d, axis="spatial")
        t = Variable("__time__", [0, 1], domain=d, axis="temporal")

        u = x * t
        pts = jnp.ones((8, 1))
        result = _eval(u.d(t), {"x": pts, "__time__": jnp.array([1.0])})
        assert result.shape == (8, 1)
