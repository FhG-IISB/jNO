"""Full jNO pipeline integration tests for differential and integral operators.

Where the per-operator MMS suites in ``tests/test_operators_mms.py`` and
``tests/test_operators_mms_3d.py`` call ``TraceEvaluator.evaluate(...)``
directly with hand-crafted contexts, these tests drive each operator
through the actual user-facing path:

    dom = jno.domain(...)
    crux = jno.core([...], dom)
    (val,) = crux.eval([expr], domain=dom, min_consecutive=...)

This catches bugs that live in the compiler, the per-step vmap, the time
window handling, and the integration between operator dispatch and the
domain context — none of which the unit-style MMS exercises.

Sections:

  E1  Mixed spatial-temporal chain — ``u.d(x).d(t)`` and ``u.d(t).d(x)``
      on a time-dependent line domain. Analytic anchor on polynomial u.

  E2  3-D Green's first and second identities — driven through
      ``jno.core(...).eval()`` on a unit cube. Polynomial u, v with
      closed-form volume integrals.

  E5  2-D time-windowed Hessian — `_eval_hessian` `points.ndim == 3`
      branch with 2-D spatial. Analytic anchor on
      ``u(x, y, t) = sin(πx) cos(πy) exp(-2π² t)`` whose Laplacian
      is ``-2π² u`` at every (t, x, y) cell.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jno
import jno.jnp_ops as jnn

# ────────────────────────────────────────────────────────────────────────
# E1: Mixed spatial-temporal chain via full pipeline
# ────────────────────────────────────────────────────────────────────────


class TestMixedSpatialTemporalChain:
    """``u.d(x).d(t)`` and ``u.d(t).d(x)`` on a polynomial ``u(x, t)``.

    Real-world PDE residuals chain spatial and temporal derivatives all the
    time (e.g. the heat-equation residual ``u_t - α·Δu``). The existing
    suite tests ``.d(x).d(y)`` and ``.d(t).d(t)`` chains but never crosses
    spatial-temporal — a regression in the ``temporal_derivative_order``
    detection at ``trace_evaluator.py:990–994`` or in the per-axis context
    handling could silently break this chain.
    """

    def _build_dom(self, n_time: int = 5):
        return jno.domain(constructor=jno.domain.line(mesh_size=0.1), time=(0.0, 1.0, n_time))

    def test_uxt_mixed_partial_x_then_t(self):
        """``u(x, t) = x · t²``  →  ``∂²u/∂x∂t = 2t`` at every (x, t) cell."""
        dom = self._build_dom(n_time=5)
        x, t = dom.variable("interior")

        u = x * t * t
        d2 = u.d(x).d(t)

        (val,) = jno.core([], dom).eval([d2], domain=dom, min_consecutive=None)
        # Shape (T, N, 1); analytic value is 2t (independent of x).
        time_pts = jnp.asarray(dom._time_points)  # (T,)
        expected = (2.0 * time_pts)[:, None, None]  # broadcast to (T, 1, 1)
        assert val.shape[0] == time_pts.shape[0], f"time axis size {val.shape[0]} != {time_pts.shape[0]}"
        assert jnp.allclose(val, expected, atol=1e-5), (
            f"max abs err = {float(jnp.max(jnp.abs(val - expected))):.3e}; expected 2t at every cell"
        )

    def test_uxt_mixed_partial_t_then_x(self):
        """``u(x, t) = x · t²``  →  ``∂²u/∂t∂x = 2t``. The other order; if
        the implementation mishandles the inner Jacobian's axis the result
        will differ from the previous test."""
        dom = self._build_dom(n_time=5)
        x, t = dom.variable("interior")

        u = x * t * t
        d2 = u.d(t).d(x)

        (val,) = jno.core([], dom).eval([d2], domain=dom, min_consecutive=None)
        time_pts = jnp.asarray(dom._time_points)
        expected = (2.0 * time_pts)[:, None, None]
        assert jnp.allclose(val, expected, atol=1e-5), (
            f"max abs err = {float(jnp.max(jnp.abs(val - expected))):.3e}; expected 2t at every cell"
        )

    def test_quadratic_in_both_axes(self):
        """``u(x, t) = x² · t``  →  ``∂²u/∂x∂t = 2x``. Varies in x not t —
        catches an axis-mix-up that the constant-in-x case above would not."""
        dom = self._build_dom(n_time=5)
        x, t = dom.variable("interior")

        u = x * x * t
        d2 = u.d(x).d(t)

        (val,) = jno.core([], dom).eval([d2], domain=dom, min_consecutive=None)
        x_pts = jnp.asarray(dom.context["interior"])[0, 0, :, 0]  # (N,)
        expected = jnp.broadcast_to((2.0 * x_pts)[None, :, None], val.shape)
        assert jnp.allclose(val, expected, atol=1e-5), (
            f"max abs err = {float(jnp.max(jnp.abs(val - expected))):.3e}; expected 2x at every cell"
        )


# ────────────────────────────────────────────────────────────────────────
# E2: 3-D Green's identities via full pipeline on the unit cube
# ────────────────────────────────────────────────────────────────────────


def _build_cube_3d(mesh_size: float = 0.10):
    return jno.domain(
        constructor=jno.domain.cube(mesh_size=mesh_size),
        compute_mesh_connectivity=True,
    )


def _eval_scalar(crux, expr, dom):
    """Run crux.eval on a single expression and return the scalar value."""
    (val,) = crux.eval([expr], domain=dom)
    return float(jnp.squeeze(val))


class TestGreensFirstIdentity3D:
    """``∫_C (∇u·∇v + u·Δv) dV = ∮_∂C u (∇v·n) dS`` on the unit cube.

    Polynomial choice: ``u(x,y,z) = x² + y² + z²`` (Δu = 6) and
    ``v(x,y,z) = x·y·z`` (Δv = 0, ∇v = (yz, xz, xy)).

    Volume side: ``∫_C ∇u·∇v dV = ∫ (2x·yz + 2y·xz + 2z·xy) dV
                                 = 6 ∫_C xyz dV``.

    On the unit cube ``∫_C xyz dV = (∫₀¹ x dx)³ = 1/8``, so the volume
    integral evaluates to ``6 · 1/8 = 3/4``. The surface side must match
    to within 3-D boundary-quadrature tolerance.
    """

    def test_identity_holds_on_cube(self):
        dom = _build_cube_3d(mesh_size=0.10)
        crux = jno.core([], dom)

        # Volume side via .d() (default AD) and .integrate().
        x_v, y_v, z_v, _ = dom.variable("interior")
        u_v = x_v**2 + y_v**2 + z_v**2  # Δu = 6
        v_v = x_v * y_v * z_v  # Δv = 0
        grad_dot = u_v.d(x_v) * v_v.d(x_v) + u_v.d(y_v) * v_v.d(y_v) + u_v.d(z_v) * v_v.d(z_v)
        lhs_expr = (grad_dot + u_v * v_v.laplacian(x_v, y_v, z_v)).integrate()
        lhs = _eval_scalar(crux, lhs_expr, dom)

        # Surface side: ∮_∂C u (∇v·n) dS with ∇v = (yz, xz, xy).
        x_b, y_b, z_b, _, nx, ny, nz = dom.variable("boundary", normals=True)
        u_b = x_b**2 + y_b**2 + z_b**2
        grad_v_dot_n = y_b * z_b * nx + x_b * z_b * ny + x_b * y_b * nz
        rhs_expr = (u_b * grad_v_dot_n).integrate()
        rhs = _eval_scalar(crux, rhs_expr, dom)

        # Strong-form check on the analytic volume value (6·1/8 = 3/4).
        # 3-D volume quadrature on a smooth polynomial is O(h²); 5% gives
        # ~3× headroom on typical 1–2% rel err.
        assert lhs == pytest.approx(0.75, rel=0.05), f"Green's-1 3-D volume side = {lhs:.4f}, expected 3/4 = 0.75"

        # Identity: LHS ≈ RHS within 3-D boundary-quadrature tolerance.
        scale = max(abs(lhs), abs(rhs), 1.0)
        assert abs(lhs - rhs) / scale < 0.15, (
            f"Green's first identity 3-D: LHS={lhs:.4f}, RHS={rhs:.4f}, rel diff = {abs(lhs - rhs) / scale * 100:.2f}%"
        )


class TestGreensSecondIdentity3D:
    """``∫_C (u·Δv − v·Δu) dV = ∮_∂C (u (∇v·n) − v (∇u·n)) dS`` on the unit cube.

    Polynomial choice: same as 2-D ``TestGreensSecondIdentity`` extended to 3-D:
    ``u(x,y,z) = x² + y² + z²``, ``v(x,y,z) = x³ + y³ + z³``.

    - Δu = 6, Δv = 6x + 6y + 6z
    - ∇u = (2x, 2y, 2z), ∇v = (3x², 3y², 3z²)
    - Volume integrand: (x²+y²+z²)(6x+6y+6z) − (x³+y³+z³)·6
      = 6x³ + 6x²y + 6x²z + 6xy² + 6y³ + 6y²z + 6xz² + 6yz² + 6z³
        − 6x³ − 6y³ − 6z³
      = 6x²y + 6x²z + 6xy² + 6y²z + 6xz² + 6yz²
      = 6 ∫_C (x²y + x²z + xy² + y²z + xz² + yz²) dV

    On the unit cube, each summand factorises:
      ∫_C x²y dV = (1/3)·(1/2)·1 = 1/6, similarly each → 1/6 (6 terms).
    Volume side = 6 · 6 · 1/6 = **6** (closed-form reference).
    """

    def test_identity_holds_on_cube(self):
        dom = _build_cube_3d(mesh_size=0.10)
        crux = jno.core([], dom)

        x_v, y_v, z_v, _ = dom.variable("interior")
        u_v = x_v**2 + y_v**2 + z_v**2
        v_v = x_v**3 + y_v**3 + z_v**3
        lhs_expr = (u_v * v_v.laplacian(x_v, y_v, z_v) - v_v * u_v.laplacian(x_v, y_v, z_v)).integrate()
        lhs = _eval_scalar(crux, lhs_expr, dom)

        x_b, y_b, z_b, _, nx, ny, nz = dom.variable("boundary", normals=True)
        u_b = x_b**2 + y_b**2 + z_b**2
        v_b = x_b**3 + y_b**3 + z_b**3
        grad_u_dot_n = 2.0 * x_b * nx + 2.0 * y_b * ny + 2.0 * z_b * nz
        grad_v_dot_n = 3.0 * x_b**2 * nx + 3.0 * y_b**2 * ny + 3.0 * z_b**2 * nz
        rhs_expr = (u_b * grad_v_dot_n - v_b * grad_u_dot_n).integrate()
        rhs = _eval_scalar(crux, rhs_expr, dom)

        assert lhs == pytest.approx(6.0, rel=0.05), f"Green's-2 3-D volume side = {lhs:.4f}, expected analytic 6"
        scale = max(abs(lhs), abs(rhs), 1.0)
        assert abs(lhs - rhs) / scale < 0.15, (
            f"Green's second identity 3-D: LHS={lhs:.4f}, RHS={rhs:.4f}, rel diff = {abs(lhs - rhs) / scale * 100:.2f}%"
        )


# ────────────────────────────────────────────────────────────────────────
# E5: 2-D time-windowed Hessian via full pipeline
# ────────────────────────────────────────────────────────────────────────


class TestThirdOrderTemporalComposition:
    """``u.d(t).d(t).d(t)`` evaluates correctly by *composition*: the inner
    ``.d(t).d(t)`` is detected as ``temporal_derivative_order == 2`` and
    routed through ``jax.grad(jax.grad(...))``; the outer ``.d(t)`` then
    takes ``jax.grad`` of that. The ``NotImplementedError`` guard at
    ``trace_evaluator.py:1111`` only fires if ``temporal_derivative_order``
    somehow exceeds 2 in a single dispatch — which the order-detection
    code path at lines 990–994 cannot produce from public API.

    Pinning this confirms the composition path is healthy and catches a
    regression where someone "fixes" the order detection to set values
    above 2 (which would then hit the dead-code error).
    """

    def test_third_order_temporal_via_composition(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.2), time=(0.0, 1.0, 5))
        x, t = dom.variable("interior")

        # u(t) = t³  →  ∂³u/∂t³ = 6 (constant)
        u = t * t * t
        expr = u.d(t).d(t).d(t)
        result = jno.core([], dom).eval([expr], domain=dom, min_consecutive=None)
        val = result[0] if isinstance(result, list) else result
        assert jnp.allclose(val, 6.0, atol=1e-5), f"∂³(t³)/∂t³ should be 6, got {val}"


class TestWindowedHessian2DPipeline:
    """``_eval_hessian`` `points.ndim == 3` branch driven through
    ``jno.core(...).eval(min_consecutive=...)`` on a 2-D spatial + time
    domain.

    Track C exercised this code path in 1-D spatial only by hand-crafting
    a context. Real spatiotemporal PINNs solve on 2-D + time and rely on
    the same vmap-over-(t, point) loop in ``trace_evaluator.py:1374–1411``;
    a stencil-shape bug at the 2-D dimension jump would not be caught by
    the 1-D test.

    Analytic anchor: ``u(x, y, t) = sin(πx) cos(πy) exp(-2π² t)`` satisfies
    ``Δu = -2π² u`` at every (t, x, y) cell.
    """

    def test_windowed_laplacian_on_2d_time_window(self):
        # Coarse mesh + few time steps so the test stays fast — the goal
        # is to drive the windowed path, not measure stencil accuracy
        # (the latter is already covered by Track B's L²-rel tests).
        dom = jno.domain(constructor=jno.domain.rect(mesh_size=0.2), time=(0.0, 0.1, 3))
        x, y, t = dom.variable("interior")

        # u(x, y, t) = sin(πx) cos(πy) exp(-2π² t)
        u = jnn.sin(jnn.pi * x) * jnn.cos(jnn.pi * y) * jnn.exp(-2.0 * jnn.pi * jnn.pi * t)
        lap = u.laplacian(x, y)

        # min_consecutive=None → full time window → exercises the
        # points.ndim == 3 branch of _eval_hessian.
        (val,) = jno.core([], dom).eval([lap], domain=dom, min_consecutive=None)
        # val shape: (T, N, 1)
        assert val.ndim == 3 and val.shape[-1] == 1, f"unexpected shape {val.shape}"

        # Compare against analytic Δu = -2π² u at every (t, point) cell.
        # Reconstruct (t, x, y) values from the domain context.
        time_pts = jnp.asarray(dom._time_points)  # (T,)
        # interior context shape: (1, T, N, 2) per the time-broadcast pool.
        interior = jnp.asarray(dom.context["interior"])
        while interior.ndim > 3 and interior.shape[0] == 1:
            interior = interior[0]
        x_vals = interior[0, :, 0]  # (N,)
        y_vals = interior[0, :, 1]  # (N,)

        u_analytic = (
            jnp.sin(jnp.pi * x_vals)[None, :]
            * jnp.cos(jnp.pi * y_vals)[None, :]
            * jnp.exp(-2.0 * jnp.pi**2 * time_pts)[:, None]
        )
        expected = (-2.0 * jnp.pi**2 * u_analytic)[:, :, None]  # (T, N, 1)

        rel = float(jnp.sqrt(jnp.mean((val - expected) ** 2)) / jnp.sqrt(jnp.mean(expected**2)))
        assert rel < 0.05, f"windowed Δu L²-rel err = {rel * 100:.2f}% > 5%"
