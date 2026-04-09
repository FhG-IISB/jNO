"""Tests for the method-style derivative API on Placeholder.

Covers:
  - Structural node creation (.d, .diff, .laplacian, .hessian)
  - Numerical correctness against analytic results (AD scheme)
  - Parity between method API and jnn.grad / jnn.laplacian / jnn.hessian
  - Chained second-order derivatives  u.d(x).d(x)
  - Full Hessian matrix  u.hessian(x, y)
  - FD sub-scheme strings: "finite_difference:lsq", ":cotangent", ":uniform",
    ":inverse_distance" — routed through DifferentialOperators.parse_fd_scheme
  - Direct unit tests for DifferentialOperators class methods

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

import jno.jnp_ops as jnn
from jno.differential_operators import DifferentialOperators
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


# ───────────────────────────────────────────────────────────────────────────────
# 9. DifferentialOperators.parse_fd_scheme
# ───────────────────────────────────────────────────────────────────────────────


class TestParseFdScheme:
    """Unit tests for the scheme-string parser."""

    def test_plain_fd_defaults(self):
        main, gm, lm = DifferentialOperators.parse_fd_scheme("finite_difference")
        assert main == "finite_difference"
        assert gm == "area_weighted"
        assert lm == "gradient_of_gradient"

    def test_ad_returns_nones(self):
        main, gm, lm = DifferentialOperators.parse_fd_scheme("automatic_differentiation")
        assert main == "automatic_differentiation"
        assert gm is None
        assert lm is None

    def test_lsq_subscheme(self):
        main, gm, lm = DifferentialOperators.parse_fd_scheme("finite_difference:lsq")
        assert main == "finite_difference"
        assert gm == "least_squares"
        assert lm == "lsq_of_gradient"

    def test_least_squares_subscheme(self):
        main, gm, lm = DifferentialOperators.parse_fd_scheme("finite_difference:least_squares")
        assert gm == "least_squares"
        assert lm == "lsq_of_gradient"

    def test_cotangent_subscheme(self):
        main, gm, lm = DifferentialOperators.parse_fd_scheme("finite_difference:cotangent")
        assert main == "finite_difference"
        assert gm == "area_weighted"
        assert lm == "cotangent"

    def test_uniform_subscheme(self):
        main, gm, lm = DifferentialOperators.parse_fd_scheme("finite_difference:uniform")
        assert gm == "uniform"
        assert lm == "gradient_of_gradient"

    def test_inverse_distance_subscheme(self):
        main, gm, lm = DifferentialOperators.parse_fd_scheme("finite_difference:inverse_distance")
        assert gm == "inverse_distance"
        assert lm == "gradient_of_gradient"


# ───────────────────────────────────────────────────────────────────────────────
# 10. DifferentialOperators — direct unit tests on small analytic meshes
# ───────────────────────────────────────────────────────────────────────────────

# --------------- shared tiny meshes -----------------------------------------


def _line_mesh():
    """5 points on [0, 1], 4 line elements."""
    pts = jnp.linspace(0.0, 1.0, 5).reshape(5, 1)
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    return pts, lines


def _square_mesh():
    """Unit square split into 2 right triangles: (0,1,2) and (0,2,3)."""
    pts = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    return pts, tris


def _grid_mesh_3x3():
    """3×3 regular triangulated unit square — 9 nodes, 8 triangles."""
    xs = jnp.linspace(0.0, 1.0, 3)
    ys = jnp.linspace(0.0, 1.0, 3)
    gx, gy = jnp.meshgrid(xs, ys, indexing="xy")
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
    # node indices:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    tris = np.array(
        [
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [3, 4, 7],
            [3, 7, 6],
            [4, 5, 8],
            [4, 8, 7],
        ]
    )
    return pts, tris


class TestDifferentialOperatorsUnit:
    """Direct numerical tests of DifferentialOperators static methods.

    Uses simple analytic functions (linear, quadratic) where exact values
    are known.  All gradient methods must recover the exact gradient of a
    *linear* function (constant gradient field is within the FE polynomial
    space of each element).
    """

    # ── 1-D gradient methods ─────────────────────────────────────────────────

    @pytest.mark.parametrize("method", ["area_weighted", "uniform", "inverse_distance", "least_squares"])
    def test_1d_gradient_linear_exact(self, method):
        """All 1-D gradient methods recover slope of a linear function exactly."""
        pts, lines = _line_mesh()
        u = 3.0 * pts[:, 0] + 1.0  # u = 3x + 1  →  du/dx = 3 everywhere
        g = DifferentialOperators.compute_fd_gradient_1d_simple(u, pts, lines, method=method)
        # Interior nodes only (skip boundary nodes 0 and 4 which have one-sided stencils)
        assert jnp.allclose(g[1:4], jnp.full(3, 3.0), atol=1e-5), f"method='{method}': interior values {g[1:4]} != 3.0"

    def test_1d_gradient_quadratic_area_weighted(self):
        """Area-weighted 1-D gradient of x² → 2x (piecewise linear approx)."""
        pts, lines = _line_mesh()
        u = pts[:, 0] ** 2
        g = DifferentialOperators.compute_fd_gradient_1d_simple(u, pts, lines)
        # At interior midpoints the piecewise gradient should be close to 2x
        expected = 2.0 * pts[1:4, 0]
        assert jnp.allclose(g[1:4], expected, atol=0.15)  # O(h) accuracy, h=0.25

    # ── 2-D gradient methods ─────────────────────────────────────────────────

    @pytest.mark.parametrize("method", ["area_weighted", "uniform", "inverse_distance"])
    def test_2d_gradient_linear_exact(self, method):
        """All element-averaging gradient methods recover exact linear gradient."""
        pts, tris = _square_mesh()
        u = 2.0 * pts[:, 0] + 3.0 * pts[:, 1]  # du/dx=2, du/dy=3
        gx = DifferentialOperators.compute_fd_gradient_2d_simple(u, pts, tris, 0, method=method)
        gy = DifferentialOperators.compute_fd_gradient_2d_simple(u, pts, tris, 1, method=method)
        assert jnp.allclose(gx, jnp.full(4, 2.0), atol=1e-4), f"{method}: gx={gx}"
        assert jnp.allclose(gy, jnp.full(4, 3.0), atol=1e-4), f"{method}: gy={gy}"

    def test_2d_gradient_lsq_linear_exact(self):
        """LSQ gradient recovers exact linear gradient at interior nodes.

        Corner/boundary nodes that have only a single incident element produce
        a rank-deficient 2x2 system; accuracy is only guaranteed where the
        local patch spans at least two independent directions (interior nodes).
        """
        pts, tris = _grid_mesh_3x3()  # 9 nodes, 8 triangles; node 4 is interior
        u = 2.0 * pts[:, 0] + 3.0 * pts[:, 1]
        gx = DifferentialOperators.compute_gradient_2d_lsq(u, pts, tris, 0)
        gy = DifferentialOperators.compute_gradient_2d_lsq(u, pts, tris, 1)
        # Only check the single interior node (index 4) which has 6 incident triangles
        assert abs(float(gx[4]) - 2.0) < 1e-4, f"LSQ gx at interior = {gx[4]}"
        assert abs(float(gy[4]) - 3.0) < 1e-4, f"LSQ gy at interior = {gy[4]}"

    def test_2d_gradient_lsq_matches_area_weighted_on_linear(self):
        """LSQ and area-weighted agree on a linear function at interior nodes.

        Both methods are exact for linear functions; they should produce
        identical results at well-supported (interior) nodes.
        """
        pts, tris = _grid_mesh_3x3()
        u = 5.0 * pts[:, 0] - 2.0 * pts[:, 1]
        aw = DifferentialOperators.compute_fd_gradient_2d_simple(u, pts, tris, 0, method="area_weighted")
        lsq = DifferentialOperators.compute_gradient_2d_lsq(u, pts, tris, 0)
        # Compare only at the interior node (index 4)
        assert abs(float(aw[4]) - float(lsq[4])) < 1e-3, f"Interior node: AW={aw[4]:.6f}, LSQ={lsq[4]:.6f}"

    # ── 2-D Laplacian methods ────────────────────────────────────────────────

    def test_2d_laplacian_linear_zero(self):
        """∇²(ax + by) = 0 for all three Laplacian methods."""
        pts, tris = _grid_mesh_3x3()
        u = 2.0 * pts[:, 0] + 3.0 * pts[:, 1]
        for method in ["gradient_of_gradient", "cotangent", "lsq_of_gradient"]:
            lap = DifferentialOperators.compute_fd_laplacian_2d_simple(u, pts, tris, (0, 1), method=method)
            # Interior nodes (index 4 = centre of 3x3 grid) must be near zero
            assert abs(float(lap[4])) < 0.1, f"{method}: lap[4]={lap[4]}"

    def test_2d_cotangent_laplacian_constant_function(self):
        """∇²(c) = 0 for a constant function — cotangent Laplacian."""
        pts, tris = _grid_mesh_3x3()
        u = jnp.ones(pts.shape[0])
        lap = DifferentialOperators.compute_laplacian_2d_cotangent(u, pts, tris)
        assert jnp.allclose(lap, jnp.zeros_like(lap), atol=1e-6), f"max={jnp.max(jnp.abs(lap))}"

    def test_2d_cotangent_laplacian_linear_function(self):
        """∇²(ax + by) = 0 — cotangent Laplacian."""
        pts, tris = _grid_mesh_3x3()
        u = 2.0 * pts[:, 0] + 3.0 * pts[:, 1]
        lap = DifferentialOperators.compute_laplacian_2d_cotangent(u, pts, tris)
        # All interior nodes (index 4) must be essentially zero
        assert abs(float(lap[4])) < 1e-4, f"cotangent lap of linear at centre = {lap[4]}"

    def test_2d_cotangent_laplacian_quadratic(self):
        """∇²(x²+y²) ≈ 4 at the interior node — cotangent Laplacian.

        The normalisation uses 2*A_i (barycentric dual area).  On this
        coarse regular mesh the interior node (index 4) should be exact.
        """
        pts, tris = _grid_mesh_3x3()
        u = pts[:, 0] ** 2 + pts[:, 1] ** 2
        lap = DifferentialOperators.compute_laplacian_2d_cotangent(u, pts, tris)
        # Interior node (index 4, at (0.5, 0.5))
        assert abs(float(lap[4]) - 4.0) < 0.15, f"cotangent lap at centre = {lap[4]}"

    def test_2d_cotangent_vs_gradient_of_gradient_linear(self):
        """Cotangent and gradient-of-gradient Laplacians agree on linear functions."""
        pts, tris = _grid_mesh_3x3()
        u = 3.0 * pts[:, 0] - pts[:, 1]
        cot = DifferentialOperators.compute_laplacian_2d_cotangent(u, pts, tris)
        gog = DifferentialOperators.compute_fd_laplacian_2d_simple(u, pts, tris, (0, 1))
        # Both should give ≈ 0; at the centre node they should be close
        assert abs(float(cot[4]) - float(gog[4])) < 0.1

    # ── LSQ gradient method parameter on 2D simple function ─────────────────

    def test_2d_gradient_method_lsq_kwarg_dispatches_correctly(self):
        """compute_fd_gradient_2d_simple with method='least_squares' delegates to compute_gradient_2d_lsq."""
        pts, tris = _square_mesh()
        u = 4.0 * pts[:, 0] + pts[:, 1]
        via_kwarg = DifferentialOperators.compute_fd_gradient_2d_simple(u, pts, tris, 0, method="least_squares")
        via_direct = DifferentialOperators.compute_gradient_2d_lsq(u, pts, tris, 0)
        assert jnp.allclose(via_kwarg, via_direct, atol=1e-6)


# ───────────────────────────────────────────────────────────────────────────────
# 11. FD sub-scheme integration tests on the L-shaped mesh
#     Exercises the full pipeline: scheme string → parse_fd_scheme → operators
# ───────────────────────────────────────────────────────────────────────────────


class TestFDSubSchemesOnLShape:
    """Integration tests for 'finite_difference:<method>' scheme strings.

    Uses the same L-shaped mesh as TestFiniteDifferenceOnLShape but exercises
    the new sub-scheme syntax.  Accuracy expectations are the same: O(h)
    interior mean error for a coarse mesh.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _domain(self, request):
        dom, x, y, t, pts, interior_mask = _build_l_domain()
        request.cls.dom = dom
        request.cls.x = x
        request.cls.y = y
        request.cls.pts = pts
        request.cls.interior_mask = interior_mask

    def _ctx(self):
        return {"interior": self.pts, "__time__": jnp.array([1.0])}

    # ── scheme node structure ────────────────────────────────────────────────

    def test_lsq_scheme_stored_on_node(self):
        """u.d(x, scheme='finite_difference:lsq') stores the sub-scheme string."""
        u = self.x * self.x
        node = u.d(self.x, scheme="finite_difference:lsq")
        assert isinstance(node, Jacobian)
        assert node.scheme == "finite_difference:lsq"

    def test_cotangent_scheme_stored_on_node(self):
        u = self.x * self.x
        node = u.laplacian(self.x, self.y, scheme="finite_difference:cotangent")
        assert isinstance(node, Hessian)
        assert node.scheme == "finite_difference:cotangent"

    # ── gradient sub-schemes ────────────────────────────────────────────────

    @pytest.mark.parametrize("sub", ["uniform", "inverse_distance", "lsq"])
    def test_gradient_subscheme_accuracy(self, sub):
        """∂(x²+y²)/∂x = 2x for all gradient sub-schemes, interior mean error < 0.05."""
        u = self.x * self.x + self.y * self.y
        scheme = f"finite_difference:{sub}"
        du_dx = _eval(u.d(self.x, scheme=scheme), self._ctx())
        expected = 2.0 * self.pts[:, 0:1]
        err = float(jnp.mean(jnp.abs(du_dx - expected)[self.interior_mask]))
        assert err < 0.05, f"scheme='{scheme}': mean interior gradient error = {err:.4f}"

    @pytest.mark.parametrize("sub", ["uniform", "inverse_distance", "lsq"])
    def test_gradient_subscheme_shape(self, sub):
        """FD gradient sub-scheme must return (N, 1)."""
        u = self.x * self.x
        result = _eval(u.d(self.x, scheme=f"finite_difference:{sub}"), self._ctx())
        assert result.shape == (self.pts.shape[0], 1)

    def test_lsq_gradient_close_to_area_weighted(self):
        """LSQ and area-weighted FD gradients should be close on the L-shaped mesh."""
        u = self.x * self.x + self.y * self.y
        ctx = self._ctx()
        aw = _eval(u.d(self.x, scheme="finite_difference"), ctx)
        lsq = _eval(u.d(self.x, scheme="finite_difference:lsq"), ctx)
        mean_diff = float(jnp.mean(jnp.abs(aw - lsq)[self.interior_mask]))
        # They solve the same underlying problem; on a reasonable mesh
        # the two estimates agree to within a few percent of the gradient magnitude
        assert mean_diff < 0.1, f"AW vs LSQ mean interior diff = {mean_diff:.4f}"

    # ── Laplacian sub-schemes ────────────────────────────────────────────────

    def test_cotangent_laplacian_accuracy(self):
        """∇²(x²+y²) = 4: cotangent scheme interior mean error < 0.5."""
        u = self.x * self.x + self.y * self.y
        lap = _eval(u.laplacian(self.x, self.y, scheme="finite_difference:cotangent"), self._ctx())
        err = float(jnp.mean(jnp.abs(lap - 4.0)[self.interior_mask]))
        assert err < 0.5, f"cotangent laplacian mean error = {err:.4f}"

    def test_cotangent_laplacian_shape(self):
        """Cotangent Laplacian must return (N, 1)."""
        u = self.x * self.x + self.y * self.y
        result = _eval(u.laplacian(self.x, self.y, scheme="finite_difference:cotangent"), self._ctx())
        assert result.shape == (self.pts.shape[0], 1)

    def test_cotangent_vs_gradient_of_gradient_laplacian(self):
        """Cotangent and default FD Laplacian estimates must broadly agree."""
        u = self.x * self.x + self.y * self.y
        ctx = self._ctx()
        default_lap = _eval(u.laplacian(self.x, self.y, scheme="finite_difference"), ctx)
        cot_lap = _eval(u.laplacian(self.x, self.y, scheme="finite_difference:cotangent"), ctx)
        diff = jnp.abs(default_lap - cot_lap)
        mean_diff = float(jnp.mean(diff[self.interior_mask]))
        # Both target ≈ 4; they can differ but should be in the same ballpark
        assert mean_diff < 2.0, f"default vs cotangent: mean interior diff = {mean_diff:.4f}"

    def test_lsq_laplacian_accuracy(self):
        """∇²(x²+y²) = 4: 'finite_difference:lsq' interior mean error < 1.0."""
        u = self.x * self.x + self.y * self.y
        lap = _eval(u.laplacian(self.x, self.y, scheme="finite_difference:lsq"), self._ctx())
        err = float(jnp.mean(jnp.abs(lap - 4.0)[self.interior_mask]))
        assert err < 1.0, f"lsq laplacian mean error = {err:.4f}"


# ───────────────────────────────────────────────────────────────────────────────
# 13. FD on stacked (multi-geometry) domains
# ───────────────────────────────────────────────────────────────────────────────


class TestFDOnStackedDomains:
    """Verify FD derivatives use the correct mesh when domains are stacked via +."""

    @pytest.fixture(autouse=True, scope="class")
    def _setup(self, request):
        import jno
        from jno.trace_compiler import TraceCompiler

        # Two different geometries
        dom = 3 * jno.domain.rect(mesh_size=0.1)
        dom += 2 * jno.domain.polygon(
            [(0, 0), (1, 0), (0.5, 1)],
            mesh_size=0.1,
        )
        x, y, t = dom.variable("interior")
        request.cls.dom = dom
        request.cls.x = x
        request.cls.y = y

    def _compile_and_eval(self, expr):
        """Compile expression through the full pipeline and evaluate."""
        from jno.trace_compiler import TraceCompiler
        from jno.trace import OperationDef

        fn = TraceCompiler.compile_traced_expression(expr, [])
        ctx = dict(self.dom.context)
        return fn({}, context=ctx, key=jax.random.PRNGKey(0))

    def test_batch_domain_map_exists(self):
        """After sampling, _batch_domain_map should exist and have correct length."""
        bdm = getattr(self.dom, "_batch_domain_map", None)
        assert bdm is not None
        assert len(bdm) == 5  # 3 rect + 2 triangle
        # First 3 should be domain 0 (rect), last 2 domain 1 (triangle)
        np.testing.assert_array_equal(bdm[:3], 0)
        np.testing.assert_array_equal(bdm[3:], 1)

    def test_mesh_connectivity_stored_in_sub_domains(self):
        """Each sub-domain should have its own mesh_connectivity."""
        assert self.dom.mesh_connectivity is not None
        assert len(self.dom._sub_domains) == 1
        sd_mc = self.dom._sub_domains[0].get("mesh_connectivity")
        assert sd_mc is not None
        assert "triangles" in sd_mc
        # Primary and sub-domain meshes should be different
        assert sd_mc["n_points"] != self.dom.mesh_connectivity["n_points"]

    def test_domain_mesh_connectivities_property(self):
        """_domain_mesh_connectivities should return [primary, sd0, ...]."""
        mcs = self.dom._domain_mesh_connectivities
        assert len(mcs) == 2
        assert mcs[0] is self.dom.mesh_connectivity
        assert mcs[1] is self.dom._sub_domains[0]["mesh_connectivity"]

    def test_fd_gradient_stacked(self):
        """FD gradient of x² + y² should be correct across both geometries.

        ∂(x² + y²)/∂x = 2x for every batch, regardless of which mesh
        generated the points. The grouped-vmap should route each batch
        to its correct mesh_connectivity.
        """
        u = self.x * self.x + self.y * self.y
        du_dx = self._compile_and_eval(u.d(self.x, scheme="finite_difference"))
        # du_dx shape: (B, 1, N, 1) — B=5, T=1
        assert du_dx.shape[0] == 5

        # Extract x-coordinates from context (B, 1, N, 2) → compare
        pts = jnp.array(self.dom.context["interior"])  # (5, 1, N, 2)
        expected = 2.0 * pts[:, :, :, 0:1]  # (5, 1, N, 1)

        # Check each batch: mean absolute error should be < 0.1
        for b in range(5):
            err = float(jnp.mean(jnp.abs(du_dx[b] - expected[b])))
            assert err < 0.1, f"Batch {b}: FD gradient error = {err:.4f}"

    def test_fd_laplacian_stacked(self):
        """FD Laplacian of x² + y² should be ≈ 4 for all batches."""
        u = self.x * self.x + self.y * self.y
        lap = self._compile_and_eval(u.laplacian(self.x, self.y, scheme="finite_difference"))
        assert lap.shape[0] == 5

        for b in range(5):
            mean_lap = float(jnp.mean(lap[b]))
            # FD Laplacian is approximate, especially near boundaries
            assert abs(mean_lap - 4.0) < 2.0, f"Batch {b}: mean Laplacian = {mean_lap:.2f}, expected ≈ 4.0"


class TestFEMGuardOnStackedDomains:
    """Verify that init_fem() raises on stacked domains."""

    def test_init_fem_raises_on_stacked_domain(self):
        import jno

        dom = 2 * jno.domain.rect(mesh_size=0.3)
        dom += 1 * jno.domain.disk(mesh_size=0.3)
        with pytest.raises(ValueError, match="not supported on stacked domains"):
            dom.init_fem()
