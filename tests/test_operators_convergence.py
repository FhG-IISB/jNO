"""Mesh-refinement convergence-order tests for jNO FD operators and quadrature.

For each operator we run the same MMS problem on a sequence of nested meshes
(h, h/2, h/4) and check that the empirical convergence order — computed from
the L∞ error ratio — falls inside a documented band. The bands are loose
enough to absorb mesh-quality jitter on unstructured triangulations but tight
enough to catch a stencil order regression (e.g. O(h¹) → O(h⁰)).

This is the test that the existing one-resolution checks in
``tests/test_derivatives.py`` and ``tests/test_integration_operators.py``
cannot do: if a future stencil change silently degrades the order, those
tests still pass on a single mesh — this one fails.

**Geometry choice.** Convergence-order measurement requires a convex domain
without sharp re-entrant corners; otherwise the L∞ error saturates near the
concavity and the rate becomes a measure of mesh quality at the corner
rather than of the stencil order. We therefore use a regular hexagon
(``HEX_VERTICES``) — non-trivial (six corners, non-axis-aligned edges),
non-square, but still convex.

The companion L-shape *robustness* test only asserts that errors do not blow
up under refinement (no NaN, no exponential growth), since the L-shape's
re-entrant corner is exactly where the cotangent / gradient-of-gradient
stencils are known to misbehave.

Complementary one-resolution accuracy tests on the L-shape, square-with-
hole, and multi-region domains live in ``tests/test_operators_mms.py``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

import jno
from jno.differential_operators import DifferentialOperators
from jno.trace_evaluator import TraceEvaluator

# ────────────────────────────────────────────────────────────────────────
# Geometries
# ────────────────────────────────────────────────────────────────────────

# Regular hexagon inscribed in the unit circle — convex, non-square.
HEX_VERTICES = [(math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)) for k in range(6)]

# Concave L-shape — used only for robustness, not order measurement.
L_VERTICES = [
    (0.0, 0.0),
    (2.0, 0.0),
    (2.0, 1.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (0.0, 2.0),
]


def _hexagon(mesh_size: float):
    np.random.seed(0)
    dom = jno.domain.csg(HEX_VERTICES, name="hex")
    dom.build_mesh(mesh_size=mesh_size)
    return dom


def _lshape(mesh_size: float):
    np.random.seed(0)
    dom = jno.domain.csg(L_VERTICES, name="L")
    dom.build_mesh(mesh_size=mesh_size)
    return dom


def _interior_indices(mc) -> np.ndarray:
    all_idx = np.arange(int(mc["n_points"]))
    bnd = np.asarray(mc["boundary_indices"], dtype=np.int64)
    return np.setdiff1d(all_idx, bnd)


# ────────────────────────────────────────────────────────────────────────
# MMS field — smooth, non-aligned with hexagon edges
# ────────────────────────────────────────────────────────────────────────


def _u(x, y):
    return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)


def _u_grad(x, y):
    ux = jnp.pi * jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y)
    uy = -jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    return ux, uy


def _u_lap(x, y):
    return -2.0 * jnp.pi**2 * _u(x, y)


def _empirical_order(errors: list[float], h_values: list[float]) -> list[float]:
    """log(err[i]/err[i+1]) / log(h[i]/h[i+1]) — one rate per consecutive pair."""
    rates = []
    for i in range(len(errors) - 1):
        if errors[i] <= 0.0 or errors[i + 1] <= 0.0:
            continue
        rate = np.log(errors[i] / errors[i + 1]) / np.log(h_values[i] / h_values[i + 1])
        rates.append(float(rate))
    return rates


# ────────────────────────────────────────────────────────────────────────
# 1. Gradient convergence — area-weighted FD on the hexagon
# ────────────────────────────────────────────────────────────────────────


class TestGradientConvergence:
    """Area-weighted FD gradient on the regular hexagon.

    Observed rates on the hexagon at h ∈ [0.20, 0.10, 0.05] are ≈ 1.97–1.99 —
    super-convergent (O(h²)) because the regular hexagon's mesh is highly
    symmetric. Floor at 1.5 gives ~25% headroom and catches a regression to
    plain O(h¹) — meaning even an order-correct stencil change would surface
    if it happened to lose the super-convergence property.
    """

    H_VALUES = [0.20, 0.10, 0.05]

    def _errors(self, method: str, dim: int):
        errors = []
        for h in self.H_VALUES:
            dom = _hexagon(h)
            mc = dom.mesh_connectivity
            points = jnp.asarray(mc["points"])
            triangles = jnp.asarray(mc["triangles"])
            interior = _interior_indices(mc)

            u = _u(points[:, 0], points[:, 1])
            d = DifferentialOperators.compute_fd_gradient_2d_simple(u, points, triangles, dim=dim, method=method)
            ux_an, uy_an = _u_grad(points[:, 0], points[:, 1])
            analytic = ux_an if dim == 0 else uy_an
            err = float(jnp.max(jnp.abs(d[interior] - analytic[interior])))
            errors.append(err)
        return errors

    def test_area_weighted_gradient_x_converges(self):
        errors = self._errors(method="area_weighted", dim=0)
        rates = _empirical_order(errors, self.H_VALUES)
        assert all(r > 1.5 for r in rates), f"∂u/∂x rates {rates} (h={self.H_VALUES}, err={errors})"

    def test_area_weighted_gradient_y_converges(self):
        errors = self._errors(method="area_weighted", dim=1)
        rates = _empirical_order(errors, self.H_VALUES)
        assert all(r > 1.5 for r in rates), f"∂u/∂y rates {rates} (h={self.H_VALUES}, err={errors})"


# ────────────────────────────────────────────────────────────────────────
# 2. Laplacian convergence — gradient-of-gradient stencil on hexagon
# ────────────────────────────────────────────────────────────────────────


class TestLaplacianConvergence:
    """Gradient-of-gradient Laplacian on a convex hexagon.

    The cotangent stencil is intentionally NOT included here: it is provably
    O(h²) on Delaunay triangulations of convex domains, but its convergence
    on pygmsh-produced meshes (which are not strictly Delaunay) varies. The
    MMS suite asserts cotangent accuracy at a single h; that is sufficient
    coverage without coupling this convergence test to mesh-quality
    properties of the underlying mesher.
    """

    H_VALUES = [0.20, 0.10, 0.05]

    def _errors(self, method: str):
        """L² (RMS) error over interior nodes — the standard convergence-rate
        norm for FE-style operators on irregular meshes. L∞ is dominated by
        single worst-point mesh-quality artifacts and produces noisy rates
        on unstructured triangulations."""
        errors = []
        for h in self.H_VALUES:
            dom = _hexagon(h)
            mc = dom.mesh_connectivity
            points = jnp.asarray(mc["points"])
            triangles = jnp.asarray(mc["triangles"])
            interior = _interior_indices(mc)

            u = _u(points[:, 0], points[:, 1])
            lap = DifferentialOperators.compute_fd_laplacian_2d_simple(u, points, triangles, dims=(0, 1), method=method)
            analytic = _u_lap(points[:, 0], points[:, 1])
            diff = lap[interior] - analytic[interior]
            err = float(jnp.sqrt(jnp.mean(diff**2)))
            errors.append(err)
        return errors

    def test_gradient_of_gradient_converges(self):
        errors = self._errors(method="gradient_of_gradient")
        rates = _empirical_order(errors, self.H_VALUES)
        # Observed L² rates on the hexagon: ≈ 1.25 (h=0.2→0.1) and ≈ 0.87
        # (h=0.1→0.05). The slowdown reflects pre-asymptotic behaviour of
        # the double-FD stencil. Floor at 0.6 gives ~1.4× headroom on the
        # worse rate and catches a regression to plain O(h⁰).
        assert all(r > 0.6 for r in rates), f"Δu (gradient-of-gradient) rates {rates} (h={self.H_VALUES}, err={errors})"


# ────────────────────────────────────────────────────────────────────────
# 3. Quadrature (nodal_volumes) convergence on a smooth integrand
# ────────────────────────────────────────────────────────────────────────


def _build_context(domain):
    ctx = {}
    for k, v in domain.context.items():
        arr = np.asarray(v)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        ctx[k] = jnp.array(arr)
    return ctx


class TestBoundaryQuadratureConvergence:
    """Boundary quadrature on the regular hexagon.

    ``nodal_ds`` is a nodal assembly of the 1-D trapezoidal rule along
    boundary edges → theoretically O(h²) on smooth boundary integrands.
    Companion to ``TestQuadratureConvergence`` (volume); previously only
    volume quadrature had a convergence-rate test.

    Closed-form ``∮_∂hex x² dS = 5/2`` derived by 6-fold rotational
    symmetry:

        ∮ x² dS = (1/2) ∮ (x² + y²) dS

    (because Σ_{k=0..5} cos²(kπ/3) = Σ_{k=0..5} sin²(kπ/3) = 3 and the
    cross terms ``Σ cos·sin`` cancel). On one unit-length side from
    (1, 0) to (0.5, √3/2): ``x² + y² = 1 - t + t²`` (t ∈ [0,1]), and
    ``∫₀¹ (1-t+t²) dt = 5/6``. Six sides → ``∮(x²+y²) dS = 5``, so
    ``∮x² dS = 5/2``.

    Observed rates on the hexagon: ≈ 2.00, 2.00 — exactly O(h²).
    Floor at 1.8 gives 10% headroom.
    """

    H_VALUES = [0.20, 0.10, 0.05]
    EXACT = 5.0 / 2.0

    def _err(self, h: float) -> float:
        dom = _hexagon(h)
        x_b, _y_b, _t = dom.variable("boundary")
        expr = (x_b * x_b).integrate()
        ctx = _build_context(dom)
        ev = TraceEvaluator(params={})
        return abs(float(ev.evaluate(expr, context=ctx, var_bindings={})) - self.EXACT)

    def test_boundary_quadrature_converges(self):
        errors = [self._err(h) for h in self.H_VALUES]
        rates = _empirical_order(errors, self.H_VALUES)
        assert all(r > 1.8 for r in rates), (
            f"boundary quadrature rates {rates} (h={self.H_VALUES}, err={errors}, exact={self.EXACT:.6f}) — expected ≥ O(h²)."
        )


class TestQuadratureConvergence:
    """∫_hex (x² + y²) dV on the regular hexagon (radius 1).

    For a regular hexagon of circumradius R = 1, the second moment about
    the origin is::

        ∫_hex (x² + y²) dV = (5 √3 / 8) ≈ 1.082531754730548

    derived from the standard polar-symmetric formula ∫r² dA over the
    hexagon. The integrand is smooth → nodal_volumes quadrature should
    converge well. Floor the rate at 1.0 (catches a regression from
    quadrature order O(h²) to O(h¹)).
    """

    H_VALUES = [0.20, 0.10, 0.05]
    EXACT = 5.0 * math.sqrt(3.0) / 8.0  # ≈ 1.0825

    def _err(self, h: float) -> float:
        dom = _hexagon(h)
        x, y, _ = dom.variable("interior")
        expr = (x * x + y * y).integrate()
        ctx = _build_context(dom)
        ev = TraceEvaluator(params={})
        return abs(float(ev.evaluate(expr, context=ctx, var_bindings={})) - self.EXACT)

    def test_quadrature_converges(self):
        errors = [self._err(h) for h in self.H_VALUES]
        rates = _empirical_order(errors, self.H_VALUES)
        # Observed rates ≈ 2.000 across both refinements — exact O(h²).
        # Floor at 1.8 gives ~10% headroom and catches any regression away
        # from second-order accuracy.
        assert all(r > 1.8 for r in rates), (
            f"quadrature rates {rates} (h={self.H_VALUES}, err={errors}, exact={self.EXACT:.6f}) — expected ≥ O(h²)."
        )


# ────────────────────────────────────────────────────────────────────────
# 4. Robustness on the concave L-shape
# ────────────────────────────────────────────────────────────────────────


class TestLshapeRobustness:
    """The L-shape has a re-entrant corner at (1, 1) where the FD stencils
    can saturate. We assert only that errors do not blow up under refinement
    and that the result is finite — the precise rate is not pinned because
    near-corner mesh quality varies between gmsh runs.
    """

    H_VALUES = [0.20, 0.10, 0.05]

    def test_lshape_gradient_does_not_blow_up(self):
        errors = []
        for h in self.H_VALUES:
            dom = _lshape(h)
            mc = dom.mesh_connectivity
            points = jnp.asarray(mc["points"])
            triangles = jnp.asarray(mc["triangles"])
            interior = _interior_indices(mc)
            u = _u(points[:, 0], points[:, 1])
            d = DifferentialOperators.compute_fd_gradient_2d_simple(u, points, triangles, dim=0, method="area_weighted")
            ux_an, _ = _u_grad(points[:, 0], points[:, 1])
            err = float(jnp.max(jnp.abs(d[interior] - ux_an[interior])))
            assert np.isfinite(err), f"L-shape gradient NaN at h={h}"
            errors.append(err)
        # Error must not grow under refinement (within 10% jitter slack).
        for prev, cur in zip(errors, errors[1:]):
            assert cur < prev * 1.10, f"L-shape gradient error grew under refinement: {errors}"

    def test_lshape_quadrature_does_not_blow_up(self):
        results = []
        for h in self.H_VALUES:
            dom = _lshape(h)
            x, y, _ = dom.variable("interior")
            expr = (x + y).integrate()
            ctx = _build_context(dom)
            ev = TraceEvaluator(params={})
            r = float(ev.evaluate(expr, context=ctx, var_bindings={}))
            assert np.isfinite(r), f"L-shape quadrature NaN at h={h}"
            results.append(r)
        # ∫_L (x + y) dV = 2.5 + 2.5 = 5.0  (by the same decomposition used
        # in test_operators_mms.test_lshape_first_moment, applied to y).
        # Quadrature on a smooth integrand is O(h²); at h=0.20 the coarsest
        # mesh tolerates ~2% rel error, finer meshes much less.
        for r in results:
            assert abs(r - 5.0) < 0.10, f"L-shape ∫(x+y) dV = {r:.4f}, expected ≈ 5.0 across sequence {results}"


# ────────────────────────────────────────────────────────────────────────
# 5. Sanity guard
# ────────────────────────────────────────────────────────────────────────


def test_hexagon_build_mesh_is_deterministic():
    np.random.seed(0)
    dom1 = _hexagon(0.1)
    np.random.seed(0)
    dom2 = _hexagon(0.1)
    assert dom1.mesh_connectivity["n_points"] == dom2.mesh_connectivity["n_points"]
