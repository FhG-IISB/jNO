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
        u = make_var("u").unit("K").scale(U)
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
        u = make_var("u").unit("K").scale(50.0)
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
