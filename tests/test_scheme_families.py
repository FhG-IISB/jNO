"""A differentiation scheme's **family** is resolved once, and an unknown one fails loud.

`scheme=` is `family` or `family:submethod`. The family picks the backend; the submethod is that
backend's own business. Before this, an unrecognised family was silently reinterpreted three
different ways, none of them an error:

* `parse_ad_scheme` discarded the family half, so any colon-less scheme returned the AD default;
* `parse_fd_scheme` fell through to the finite-difference settings, and handed an unknown
  `family:sub` to the FD kernel as `method=sub`;
* the evaluator's spatial dispatch had no `else`, so an unknown family returned `None` and surfaced
  later as `TypeError: 'NoneType' object is not subscriptable`.

Registering a family is an entry in `SCHEME_FAMILIES`; nothing else enumerates them.
"""

import jax
import jax.numpy as jnp
import pytest

import jno
from jno.differential_operators import DifferentialOperators as D
from jno.utils.ad_mode import parse_ad_scheme, parse_hessian_scheme
from jno.utils.schemes import SCHEME_FAMILIES, require_family, scheme_family
from tests.conftest import MockDomain


class TestSchemeFamily:
    @pytest.mark.parametrize(
        "scheme, family",
        [
            ("automatic_differentiation", "automatic_differentiation"),
            ("automatic_differentiation:forward", "automatic_differentiation"),
            ("finite_difference", "finite_difference"),
            ("finite_difference:cotangent", "finite_difference"),
            ("finite_difference:lsq", "finite_difference"),
        ],
    )
    def test_known_families_resolve(self, scheme, family):
        assert scheme_family(scheme) == family

    def test_unknown_family_raises_and_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            scheme_family("spectral")
        with pytest.raises(ValueError) as e:
            scheme_family("spectral:fft")
        for known in SCHEME_FAMILIES:
            assert known in str(e.value), f"the error should name {known!r}"

    def test_a_typo_raises_rather_than_being_treated_as_finite_difference(self):
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            scheme_family("finite_differance")  # codespell:ignore

    def test_non_string_raises(self):
        with pytest.raises(TypeError, match="must be a string"):
            scheme_family(object())

    def test_require_family_returns_the_submethod(self):
        assert require_family("finite_difference:cotangent", "finite_difference") == "cotangent"
        assert require_family("finite_difference", "finite_difference") == ""

    def test_require_family_rejects_the_wrong_backend(self):
        with pytest.raises(ValueError, match="routed to the wrong backend"):
            require_family("finite_difference", "automatic_differentiation")


class TestParsersNoLongerReinterpret:
    """Each parser now refuses a scheme belonging to another family."""

    def test_parse_ad_scheme_refuses_a_foreign_family(self):
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            parse_ad_scheme("spectral")  # used to return the global AD default
        with pytest.raises(ValueError, match="wrong backend"):
            parse_ad_scheme("finite_difference")

    def test_parse_hessian_scheme_refuses_a_foreign_family(self):
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            parse_hessian_scheme("spectral")

    def test_parse_fd_scheme_refuses_a_foreign_family(self):
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            D.parse_fd_scheme("spectral")  # used to return the FD defaults
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            D.parse_fd_scheme("spectral:fft")  # used to hand "fft" to the FD kernel as method=

    def test_the_valid_strings_are_unchanged(self):
        assert D.parse_fd_scheme("finite_difference") == ("finite_difference", "area_weighted", "gradient_of_gradient")
        assert D.parse_fd_scheme("finite_difference:cotangent") == ("finite_difference", "area_weighted", "cotangent")
        assert D.parse_fd_scheme("automatic_differentiation") == ("automatic_differentiation", None, None)
        assert parse_ad_scheme("automatic_differentiation:forward") == "forward"
        assert parse_hessian_scheme("automatic_differentiation:fwd-over-fwd") == ("forward", "forward")


class TestEvaluatorDispatchFailsLoud:
    """The `else` the dispatch chains never had."""

    def _u_and_x(self):
        d = MockDomain()
        d.context["xy"] = jnp.zeros((6, 2))
        from jno.trace import Variable

        x = Variable("xy", [0, 1], domain=d)
        y = Variable("xy", [1, 2], domain=d)
        return x * y, x, y

    def test_unknown_scheme_on_a_first_derivative_raises(self):
        from jno.trace_evaluator import TraceEvaluator

        u, x, _ = self._u_and_x()
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            TraceEvaluator({}).evaluate(u.d(x, scheme="spectral"), {"xy": jnp.ones((6, 2))}, {}, key=None)

    def test_unknown_scheme_on_a_second_derivative_raises(self):
        from jno.trace_evaluator import TraceEvaluator

        u, x, y = self._u_and_x()
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            TraceEvaluator({}).evaluate(u.laplacian(x, y, scheme="spectral"), {"xy": jnp.ones((6, 2))}, {}, key=None)

    def test_the_old_failure_mode_is_gone(self):
        """It used to return None and die later as a TypeError about subscripting None."""
        from jno.trace_evaluator import TraceEvaluator

        u, x, _ = self._u_and_x()
        with pytest.raises(ValueError):  # specifically NOT TypeError
            TraceEvaluator({}).evaluate(
                (u.d(x, scheme="spectral") + 1.0), {"xy": jnp.ones((6, 2))}, {}, key=jax.random.PRNGKey(0)
            )
