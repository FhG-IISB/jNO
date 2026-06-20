"""Tests for graph-time dimensional analysis (``jno.units``).

Covers the :class:`Unit` algebra, the inference walk's propagation rules, the
dimensional-consistency warnings, and the PINN / FEM smoke paths.
"""

import jax.numpy as jnp

import jno
from jno.trace import FunctionCall, TrialFunction
from jno.trace.units import DIMENSIONLESS, Unit
from tests.conftest import make_var


# ======================================================================
# Unit algebra
# ======================================================================
class TestUnitAlgebra:
    def test_parse_round_trips(self):
        assert Unit.parse("m") == Unit({"m": 1})
        assert Unit.parse("K") == Unit({"K": 1})
        assert Unit.parse("m/s") == Unit({"m": 1, "s": -1})
        assert Unit.parse("kg/m^3") == Unit({"kg": 1, "m": -3})
        assert Unit.parse("Pa") == Unit({"kg": 1, "m": -1, "s": -2})

    def test_parse_power_syntaxes_agree(self):
        # '^', '**', and bare-digit spellings are equivalent.
        assert Unit.parse("m^2") == Unit.parse("m**2") == Unit({"m": 2})

    def test_dimensionless_spellings(self):
        for spec in ("", "1", "-", "rad"):
            assert Unit.parse(spec).is_dimensionless()
        assert DIMENSIONLESS.is_dimensionless()

    def test_mul_div_pow(self):
        assert Unit.parse("m") * Unit.parse("s^-1") == Unit.parse("m/s")
        assert Unit.parse("m") / Unit.parse("s") == Unit.parse("m/s")
        assert Unit.parse("m") ** 3 == Unit.parse("m^3")

    def test_sqrt_is_half_power(self):
        # sqrt(m^2) == m, exactly (Fraction-backed exponents).
        assert (Unit.parse("m^2") ** 0.5) == Unit.parse("m")

    def test_equality_independent_of_spelling(self):
        assert Unit.parse("m s^-1") == Unit.parse("m/s")
        assert hash(Unit.parse("m/s")) == hash(Unit.parse("m s^-1"))


# ======================================================================
# Propagation rules (queried via the non-mutating inference walk)
# ======================================================================
class TestPropagation:
    def test_jacobian(self):
        x = make_var("x").unit("m")
        u = make_var("u").unit("K")
        assert jno.units.infer(u.d(x)) == Unit.parse("K/m")

    def test_second_derivative_is_target_over_var_squared(self):
        # d2/dd build a trace=True Hessian (Laplacian); K / m^2.
        x = make_var("x").unit("m")
        u = make_var("u").unit("K")
        assert jno.units.infer(u.d2(x)) == Unit.parse("K/m^2")

    def test_full_hessian_single_var_is_second_order(self):
        # .hessian(x) is trace=False; a single variable still yields K/m^2
        # (the dimensionally-correct rule, not the plan's K/m).
        x = make_var("x").unit("m")
        u = make_var("u").unit("K")
        assert jno.units.infer(u.hessian(x)) == Unit.parse("K/m^2")

    def test_product_and_quotient(self):
        a = make_var("a").unit("m")
        b = make_var("b").unit("s")
        assert jno.units.infer(a * b) == Unit.parse("m*s")
        assert jno.units.infer(a / b) == Unit.parse("m/s")

    def test_power(self):
        a = make_var("a").unit("m")
        assert jno.units.infer(a**3) == Unit.parse("m^3")

    def test_undeclared_leaf_is_unknown(self):
        a = make_var("a")  # no unit declared
        assert jno.units.infer(a) is None
        # an op touching an unknown operand stays unknown, without warning
        b = make_var("b").unit("m")
        logger = jno.units.check([a * b])
        assert logger.entries[-1][1] is None
        assert logger.warnings == []


# ======================================================================
# Dimensional-consistency warnings
# ======================================================================
class TestWarnings:
    def test_add_mismatch_warns(self):
        u = make_var("u").unit("K")
        v = make_var("v").unit("Pa")
        logger = jno.units.check([u + v])
        assert any("mismatch" in w for w in logger.warnings)

    def test_add_match_is_silent(self):
        u = make_var("u").unit("K")
        w = make_var("w").unit("K")
        logger = jno.units.check([u + w])
        assert logger.warnings == []

    def test_exp_of_dimensioned_arg_warns(self):
        theta = make_var("theta").unit("rad")  # rad is dimensionless → silent
        logger = jno.units.check([FunctionCall(jnp.exp, [theta])])
        assert logger.warnings == []

        psi = make_var("psi").unit("K")  # genuinely dimensioned → warns
        logger = jno.units.check([FunctionCall(jnp.exp, [psi])])
        assert any("dimensionless" in w for w in logger.warnings)

    def test_exp_result_is_dimensionless(self):
        psi = make_var("psi").unit("K")
        assert jno.units.infer(FunctionCall(jnp.exp, [psi])) == DIMENSIONLESS


# ======================================================================
# Smoke tests — PINN heat equation and FEM weak form
# ======================================================================
class TestHeatEquationSmoke:
    def test_heat_equation_terms_agree(self):
        # ∂u/∂t = alpha ∂²u/∂x²  — both terms must be K/s, no warnings.
        x = make_var("x").unit("m")
        t = make_var("t").unit("s")
        u = make_var("u").unit("K")
        alpha = make_var("alpha").unit("m^2/s")

        residual = u.d(t) - alpha * u.d2(x)
        logger = jno.units.check([residual])

        assert jno.units.infer(u.d(t)) == Unit.parse("K/s")
        assert jno.units.infer(alpha * u.d2(x)) == Unit.parse("K/s")
        assert logger.warnings == []


class TestFemWeakFormSmoke:
    def test_trial_is_leaf_test_is_dimensionless(self):
        # TrialFunction carries a user-declared unit (it wraps no Variable, so
        # it is treated as a leaf); TestFunction is dimensionless by convention.
        # Imported locally so pytest doesn't try to collect ``TestFunction``.
        from jno.trace import TestFunction

        u = TrialFunction("u").unit("K")
        phi = TestFunction("phi")
        x = make_var("x").unit("m")

        assert jno.units.infer(u) == Unit.parse("K")
        assert jno.units.infer(phi) == DIMENSIONLESS
        # a flux-like weak term inherits the trial unit through the derivative
        assert jno.units.infer(u.d(x)) == Unit.parse("K/m")

    def test_check_on_integrated_weak_form(self):
        # A faithful pre-assembly weak form — the kind passed to jno.fem([...]):
        #   ∫ (∇u·∇φ − f φ) dΩ   (steady diffusion residual).
        # check() must traverse the TrialFunction/TestFunction/Jacobian/Integral
        # tree without crashing, propagate units to the root, and list the
        # trial/test symbols among its entries.
        from jno.trace import Integral, TestFunction

        u = TrialFunction("u").unit("K")
        phi = TestFunction("phi")
        x = make_var("x").unit("m")
        f = make_var("f").unit("K/m^2")  # source matched to the Laplacian term

        weak = (u.d(x) * phi.d(x) - f * phi).integrate(x)
        logger = jno.units.check([weak])

        # root is an Integral and its unit is derivable (terms agree → no warns)
        assert isinstance(weak, Integral)
        assert logger.entries[-1][1] is not None
        assert logger.warnings == []

        # the trial and test symbols were actually visited
        labels = [label for label, _, _ in logger.entries]
        assert any("TrialFunction" in lbl for lbl in labels)
        assert any("TestFunction" in lbl for lbl in labels)


class TestLogFile:
    def test_writes_log(self, tmp_path):
        x = make_var("x").unit("m")
        u = make_var("u").unit("K")
        path = tmp_path / "units.log"
        logger = jno.units.check([u.d(x)], log=str(path))
        assert path.exists()
        text = path.read_text()
        assert "K·m⁻¹" in text or "K/m" in text
        assert logger.entries  # at least the leaves + the Jacobian
