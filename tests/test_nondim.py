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
