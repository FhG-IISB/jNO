"""Tests for automatic de-dimensionalization (``jno.units.nondimensionalize``).

Phase A: per-variable characteristic scales, the magnitude walk, and the
dimensionless groups (Fourier / Péclet) it recovers.
"""

import math

import jno
from jno.trace.units import Unit
from tests.conftest import make_var


# ======================================================================
# Scale-declaration API
# ======================================================================
class TestScaleApi:
    def test_scale_sets_slot_and_chains(self):
        x = make_var("x").unit("m").scale(0.1)
        assert x._scale == 0.1
        assert isinstance(x._scale, float)
        # chains in either order relative to .unit, returns self
        y = make_var("y").scale(2.0).unit("s")
        assert y._scale == 2.0 and y._unit == Unit.parse("s")

    def test_undeclared_scale_is_unknown(self):
        a = make_var("a").unit("m")  # unit but no scale
        report = jno.units.nondimensionalize(a)
        # single-term "residual": its magnitude is unknown → π undefined
        assert report.residuals[0].terms[0].scale is None


# ======================================================================
# Magnitude-walk primitives (mirror the unit propagation tests)
# ======================================================================
class TestMagnitudeWalk:
    def _scale_of(self, node):
        from jno.trace.unit_log import _scale_of

        return _scale_of(node)

    def test_product_quotient_power(self):
        a = make_var("a").unit("m").scale(3.0)
        b = make_var("b").unit("s").scale(2.0)
        assert self._scale_of(a * b) == 6.0
        assert self._scale_of(a / b) == 1.5
        assert self._scale_of(a**2) == 9.0

    def test_jacobian_and_laplacian(self):
        x = make_var("x").unit("m").scale(4.0)
        u = make_var("u").unit("K").scale(10.0)
        assert self._scale_of(u.d(x)) == 10.0 / 4.0
        assert self._scale_of(u.d2(x)) == 10.0 / 16.0


# ======================================================================
# Discriminating analytic cases — these FAIL a net-unit implementation
# ======================================================================
class TestDimensionlessGroups:
    def test_heat_equation_fourier_number(self):
        # ∂u/∂t = α ∂²u/∂x²  →  Fourier number Fo = α₀ τ / L²
        L, tau, U, alpha0 = 0.1, 5.0, 50.0, 1e-5
        x = make_var("x").unit("m").scale(L)
        t = make_var("t").unit("s").scale(tau)
        u = make_var("u").unit("K").scale(U)
        alpha = make_var("alpha").unit("m^2/s").scale(alpha0)

        residual = u.d(t) - alpha * u.d2(x)
        report = jno.units.nondimensionalize(residual)
        terms = report.residuals[0].terms
        assert len(terms) == 2

        # term 0 = ∂u/∂t (U/τ), term 1 = α ∂²u/∂x² (α₀ U / L²)
        ratio = terms[1].pi / terms[0].pi
        assert math.isclose(ratio, alpha0 * tau / L**2, rel_tol=1e-9)
        # both terms are dimensionally identical (K/s) — the physics is in π
        assert terms[0].unit == terms[1].unit == Unit.parse("K/s")

    def test_advection_diffusion_peclet_number(self):
        # ∂u/∂t + V ∂u/∂x = α ∂²u/∂x²  →  Péclet Pe = V L / α₀
        L, tau, U, V0, alpha0 = 0.2, 3.0, 50.0, 2.0, 1e-4
        x = make_var("x").unit("m").scale(L)
        t = make_var("t").unit("s").scale(tau)
        u = make_var("u").unit("K").scale(U)
        vel = make_var("vel").unit("m/s").scale(V0)
        alpha = make_var("alpha").unit("m^2/s").scale(alpha0)

        residual = u.d(t) + vel * u.d(x) - alpha * u.d2(x)
        report = jno.units.nondimensionalize(residual)
        terms = report.residuals[0].terms
        assert len(terms) == 3

        # advection term (index 1) / diffusion term (index 2) = Péclet
        peclet = terms[1].pi / terms[2].pi
        assert math.isclose(peclet, V0 * L / alpha0, rel_tol=1e-9)

    def test_ref_term_normalisation(self):
        # the reference term's π is 1 by construction
        x = make_var("x").unit("m").scale(0.1)
        t = make_var("t").unit("s").scale(5.0)
        u = make_var("u").unit("K").scale(50.0)
        alpha = make_var("alpha").unit("m^2/s").scale(1e-5)
        report = jno.units.nondimensionalize(u.d(t) - alpha * u.d2(x))
        r = report.residuals[0]
        assert r.terms[r.ref_index].pi == 1.0


# ======================================================================
# Report file
# ======================================================================
class TestReport:
    def test_writes_report(self, tmp_path):
        x = make_var("x").unit("m").scale(0.1)
        t = make_var("t").unit("s").scale(5.0)
        u = make_var("u").unit("K").scale(50.0)
        alpha = make_var("alpha").unit("m^2/s").scale(1e-5)
        path = tmp_path / "nondim.txt"
        jno.units.nondimensionalize(u.d(t) - alpha * u.d2(x), report=str(path))
        assert path.exists()
        text = path.read_text()
        assert "π=" in text and "residual:" in text


# ======================================================================
# Phase B — transform to a solvable dimensionless problem
# ======================================================================
def _coefficient(value):
    """A realistic dimensional coefficient leaf (Constant, not a bare Variable)."""
    import jax.numpy as jnp

    from jno.trace import Constant

    return Constant("P", "c", jnp.asarray(value))


def _field(scale_value, unit="K"):
    """A realistic field leaf (non-coordinate, non-coefficient) standing in for
    the network output net(...). A bare ``make_var`` would be misread as a
    coordinate by the coefficient-excluded magnitude walk."""
    from jno.trace import TrialFunction

    return TrialFunction("u").unit(unit).scale(scale_value)


def _leading_coeff(term_node):
    """Effective scalar coefficient prepended to a transformed additive term."""
    from jno.trace import BinaryOp, Literal

    if isinstance(term_node, BinaryOp) and term_node.op == "*" and isinstance(term_node.left, Literal):
        return float(term_node.left.value)
    return 1.0


class TestRescaleTransform:
    def test_heat_transform_coefficient_is_fourier(self):
        # The transformed diffusion term's effective dimensionless coefficient,
        # combined with the physical α₀, is exactly the Fourier number ατ/L².
        from jno.trace.unit_log import _additive_terms

        L, tau, U, alpha0 = 0.1, 5.0, 50.0, 1e-5
        x = make_var("x").unit("m").scale(L)
        t = make_var("t").unit("s").scale(tau)
        u = _field(U)
        alpha = _coefficient(alpha0).unit("m^2/s").scale(alpha0)

        residual = u.d(t) - alpha * u.d2(x)
        transformed, rescaler = jno.units.rescale(residual)

        terms = _additive_terms(transformed)
        assert len(terms) == 2
        # time term is the reference → g = 1 (no Literal prepended)
        assert _leading_coeff(terms[0][1]) == 1.0
        # diffusion term: |g| = τ/L²  ⇒  |g|·α₀ = Fourier
        g_diff = abs(_leading_coeff(terms[1][1]))
        assert math.isclose(g_diff, tau / L**2, rel_tol=1e-9)
        assert math.isclose(g_diff * alpha0, alpha0 * tau / L**2, rel_tol=1e-9)
        assert isinstance(rescaler, jno.units.Rescaler)

    def test_transform_is_not_symbolically_dimensionless(self):
        # The graph keeps dimensional coords/coefficients; dimensionlessness is
        # realized numerically on the rescaled context (documents the design).
        x = make_var("x").unit("m").scale(0.1)
        t = make_var("t").unit("s").scale(5.0)
        u = _field(50.0)
        alpha = _coefficient(1e-5).unit("m^2/s").scale(1e-5)
        transformed, _ = jno.units.rescale(u.d(t) - alpha * u.d2(x))
        assert jno.units.infer(transformed) == Unit.parse("K/s")


class TestRescaler:
    def test_rescaled_context_scales_columns(self):
        import numpy as np

        from jno.trace.unit_log import Rescaler

        # one tag "xy" with x in column 0 (L=0.1) and y in column 1 (L=0.2)
        rescaler = Rescaler([("xy", (0, 1), 0.1), ("xy", (1, 2), 0.2)], field_scale=50.0)
        ctx = {"xy": np.array([[2.0, 4.0], [6.0, 8.0]])}
        out = rescaler.rescaled_context(ctx)
        assert np.allclose(out["xy"][:, 0], [20.0, 60.0])  # /0.1
        assert np.allclose(out["xy"][:, 1], [20.0, 40.0])  # /0.2
        # original untouched
        assert np.allclose(ctx["xy"], [[2.0, 4.0], [6.0, 8.0]])

    def test_rescaled_domain_does_not_mutate_original(self):
        import types

        import numpy as np

        from jno.trace.unit_log import Rescaler

        dom = types.SimpleNamespace(context={"x": np.array([[1.0], [2.0]])})
        rescaler = Rescaler([("x", (0, 1), 0.5)], field_scale=10.0)
        new = rescaler.rescaled_domain(dom)
        assert np.allclose(new.context["x"], [[2.0], [4.0]])  # /0.5
        assert np.allclose(dom.context["x"], [[1.0], [2.0]])  # original intact

    def test_to_physical_applies_field_scale(self):
        import numpy as np

        from jno.trace.unit_log import Rescaler

        rescaler = Rescaler([], field_scale=50.0)
        assert np.allclose(rescaler.to_physical(np.array([0.5, 1.0])), [25.0, 50.0])

    def test_role_classification_by_node_type(self):
        # The crux of a *solvable* transform: tell coordinates (domain Variables)
        # apart from the field (TrialFunction/Model) and coefficients (Constant)
        # by node type — they share units but get different rescaling operations.
        from jno.trace import TrialFunction
        from jno.trace.unit_log import _collect_scaled_leaves

        x = make_var("x").unit("m").scale(0.1)
        u = TrialFunction("u").unit("K").scale(50.0)  # the field
        alpha = _coefficient(1e-5).unit("m^2/s").scale(1e-5)  # a coefficient

        coords, field_scale = _collect_scaled_leaves([alpha * u.d2(x)])
        assert field_scale == 50.0  # field U picked up from the TrialFunction
        coord_tags = {tag for tag, _, _ in coords}
        assert x.tag in coord_tags  # the coordinate is rescaled
        assert "u" not in coord_tags  # the field is NOT a coordinate


def _wraps_coord(node, var, L):
    """True if *var* appears wrapped as ``L·var`` (the bare-coordinate rewrite)."""
    from jno.trace import BinaryOp, Literal

    seen = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if (
            isinstance(n, BinaryOp)
            and n.op == "*"
            and isinstance(n.left, Literal)
            and math.isclose(float(n.left.value), L, rel_tol=1e-5)
            and n.right is var
        ):
            return True
        for attr in ("left", "right", "target", "expr"):
            child = getattr(n, attr, None)
            if child is not None:
                stack.append(child)
        for attr in ("args", "variables"):
            for c in getattr(n, attr, None) or []:
                stack.append(c)
    return False


class TestBareCoordinateRewrite:
    """The crux of the general transform: coordinate-dependent analytic targets.

    These FAIL a double-count implementation (which would give g = L³/U for a
    linear source instead of L²/U), so a sin-only suite cannot catch the bug.
    """

    def test_linear_source_coefficient_is_L2_over_U(self):
        # residual:  u_xx − C·x   (Poisson with a linear source)
        # u_xx scale = U/L²; source term g = L²/U; with the ×L rewrite the
        # effective dimensionless source coefficient is C·L³/U.
        from jno.trace.unit_log import _additive_terms

        L, U, C0 = 0.2, 100.0, 3.0
        x = make_var("x").unit("m").scale(L)
        u = _field(U)
        C = _coefficient(C0).unit("K/m^3").scale(C0)

        residual = u.d2(x) - C * x
        transformed, _ = jno.units.rescale(residual)

        terms = _additive_terms(transformed)
        assert len(terms) == 2
        # source term g = L²/U  (a double-count bug yields L³/U).  Tolerance is
        # float32-grade because Literal stores the coefficient as a jax array.
        g_src = abs(_leading_coeff(terms[1][1]))
        assert math.isclose(g_src, L**2 / U, rel_tol=1e-5)
        # a double-count (g = L³/U) would be ~5× off here — well outside 1e-5
        assert not math.isclose(g_src, L**3 / U, rel_tol=1e-2)
        # and the bare coordinate x was rewritten to L·x exactly once
        assert _wraps_coord(terms[1][1], x, L)
        # effective dimensionless source coefficient = C·L³/U
        assert math.isclose(g_src * C0 * L, C0 * L**3 / U, rel_tol=1e-5)

    def test_variable_coefficient_bare_x_is_rewritten(self):
        # residual: (1 + x)·u_xx  — the bare x inside the coefficient must be
        # rewritten so it stays physical on the rescaled context.
        L, U = 0.5, 10.0
        x = make_var("x").unit("m").scale(L)
        u = _field(U)

        residual = (1.0 + x) * u.d2(x)
        transformed, _ = jno.units.rescale(residual)

        # the bare x in (1 + x) is wrapped as L·x; the x inside u.d2(x)
        # (a derivative variable) is left alone.
        assert _wraps_coord(transformed, x, L)


class TestViewUnwrap:
    def test_rescale_unwraps_a_view(self):
        # The tutorial field API wraps expressions in views (ScalarView, …);
        # rescale must unwrap (._expr) before transforming.
        L, U, alpha0 = 0.3, 20.0, 1e-4
        x = make_var("x").unit("m").scale(L)
        t = make_var("t").unit("s").scale(4.0)
        u = _field(U)
        alpha = _coefficient(alpha0).unit("m^2/s").scale(alpha0)

        residual = u.d(t) - alpha * u.d2(x)
        view = residual.scalar  # wrap in a ScalarView
        from jno.trace.views import _VIEW_TYPES

        assert isinstance(view, _VIEW_TYPES)

        transformed, rescaler = jno.units.rescale(view)
        from jno.trace import Placeholder

        assert isinstance(transformed, Placeholder)
        assert not isinstance(transformed, _VIEW_TYPES)
        assert jno.units.infer(transformed) == Unit.parse("K/s")


class TestEndToEndSolve:
    def test_rescaled_solve_recovers_physical_field(self):
        # Fit a network to a poorly-scaled physical target u*(x) = U·sin(π·x/L)
        # on x ∈ [0, L] with L, U ≠ 1 (a coordinate-EXPRESSION target — exercises
        # the bare-coordinate rewrite).  Solve the rescaled O(1) problem, then
        # check to_physical(û) recovers u* — the genuine solvable round-trip.
        # (heat_1d as-is has L=τ=U=1 and would pass even with a scale bug.)
        import foundax
        import jax
        import optax

        import jno
        import jno.jnp_ops as jnn

        L, U = 2.0, 5.0
        pi = float(jno.np.pi)

        dom = jno.domain(constructor=jno.domain.line(x_range=(0.0, L), mesh_size=L / 40))
        (x,) = dom.variable("interior")[:1]
        x = x.unit("m").scale(L)

        net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0)))
        net.optimizer(optax.adam(2e-3))

        u_field = net(x).unit("K").scale(U)  # scale attaches to the ModelCall
        # The amplitude U is a DIMENSIONAL quantity → declare it as a Constant
        # (a bare float would be a dimensionless Literal and would not be divided
        # out, leaving the network to learn U·sin instead of the O(1) sin).
        U_amp = _coefficient(U).unit("K").scale(U)
        target = U_amp * jnn.sin(pi * x / L)  # physical, coordinate-expression target
        data = u_field - target

        transformed, rescaler = jno.units.rescale([data], dom)
        rdom = rescaler.rescaled_domain(dom)

        # original domain context is untouched by the transform
        import numpy as np

        assert not np.allclose(np.asarray(rdom.context["interior"]), np.asarray(dom.context["interior"]))

        crux = jno.core([transformed[0].mse], domain=rdom)
        crux.solve(4000)

        # û on the rescaled (O(1)) domain; map back to physical units.
        u_hat = np.asarray(crux.eval([net(x)], domain=rdom)).reshape(-1)
        u_phys = rescaler.to_physical(u_hat)

        x_hat = np.asarray(rdom.context["interior"]).reshape(-1)
        u_exact = U * np.sin(pi * x_hat)  # = U·sin(π·x_phys/L)

        rel_l2 = np.linalg.norm(u_phys - u_exact) / (np.linalg.norm(u_exact) + 1e-8)
        assert rel_l2 < 0.1, f"rescaled-solve round-trip error too large: {rel_l2:.3e}"
