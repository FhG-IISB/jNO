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
            scheme_family("wavelet")
        with pytest.raises(ValueError) as e:
            scheme_family("wavelet:db4")
        for known in SCHEME_FAMILIES:
            assert known in str(e.value), f"the error should name {known!r}"

    def test_a_registered_family_resolves(self):
        """Guards the other direction: once a backend registers, its family must stop raising.
        These tests use a never-registered name precisely so they do not go stale again."""
        assert scheme_family("spectral") == "spectral"
        assert "spectral" in SCHEME_FAMILIES

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
            parse_ad_scheme("wavelet")  # used to return the global AD default
        with pytest.raises(ValueError, match="wrong backend"):
            parse_ad_scheme("finite_difference")

    def test_parse_hessian_scheme_refuses_a_foreign_family(self):
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            parse_hessian_scheme("wavelet")

    def test_parse_fd_scheme_refuses_a_foreign_family(self):
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            D.parse_fd_scheme("wavelet")  # used to return the FD defaults
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            D.parse_fd_scheme("wavelet:db4")  # used to hand "fft" to the FD kernel as method=

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
            TraceEvaluator({}).evaluate(u.d(x, scheme="wavelet"), {"xy": jnp.ones((6, 2))}, {}, key=None)

    def test_unknown_scheme_on_a_second_derivative_raises(self):
        from jno.trace_evaluator import TraceEvaluator

        u, x, y = self._u_and_x()
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            TraceEvaluator({}).evaluate(u.laplacian(x, y, scheme="wavelet"), {"xy": jnp.ones((6, 2))}, {}, key=None)

    def test_the_old_failure_mode_is_gone(self):
        """It used to return None and die later as a TypeError about subscripting None."""
        from jno.trace_evaluator import TraceEvaluator

        u, x, _ = self._u_and_x()
        with pytest.raises(ValueError):  # specifically NOT TypeError
            TraceEvaluator({}).evaluate(
                (u.d(x, scheme="wavelet") + 1.0), {"xy": jnp.ones((6, 2))}, {}, key=jax.random.PRNGKey(0)
            )


class TestRunLevelDefault:
    """`jno.setup(diff_type=...)` declares the scheme once, for the whole run.

    A BARE `"automatic_differentiation"` already meant "use what is configured" for the AD sub-mode;
    this is the same meaning one level up, at the family. So the ~40 call sites that default to that
    string need no change, and a scheme carrying a submethod stays an explicit request.
    """

    def setup_method(self):
        from jno.utils.schemes import get_default_scheme

        self._prev = get_default_scheme()

    def teardown_method(self):
        from jno.utils.schemes import set_default_scheme

        set_default_scheme(self._prev)

    def test_default_is_automatic_differentiation(self):
        from jno.utils.schemes import get_default_scheme, resolve_scheme

        assert get_default_scheme() == "automatic_differentiation"
        assert resolve_scheme("automatic_differentiation") == "automatic_differentiation"

    def test_setting_a_family_redirects_the_bare_string(self):
        from jno.utils.schemes import resolve_scheme, set_default_scheme

        set_default_scheme("spectral")
        assert resolve_scheme("automatic_differentiation") == "spectral"

    def test_an_explicit_request_is_never_overridden(self):
        from jno.utils.schemes import resolve_scheme, set_default_scheme

        set_default_scheme("spectral")
        for explicit in ("automatic_differentiation:reverse", "finite_difference", "finite_difference:cotangent"):
            assert resolve_scheme(explicit) == explicit

    def test_a_bad_default_fails_at_declaration_not_at_use(self):
        from jno.utils.schemes import set_default_scheme

        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            set_default_scheme("wavelet")

    def test_setup_accepts_a_scheme_and_an_ad_submode(self):
        import jno.utils.config as cfg
        from jno.utils.ad_mode import get_ad_mode
        from jno.utils.schemes import get_default_scheme

        cfg.apply_ad_mode_defaults(diff_type="spectral")
        assert get_default_scheme() == "spectral"

        prev_mode = get_ad_mode()
        cfg.apply_ad_mode_defaults(diff_type="forward")  # the historical meaning still works
        assert get_ad_mode() == "forward"
        assert get_default_scheme() == "spectral", "an AD sub-mode must not clobber the family default"
        from jno.utils.ad_mode import set_ad_mode

        set_ad_mode(prev_mode)


class TestAFourthFamilyIsAdditive:
    """The extension point, exercised: register a family without touching the evaluator.

    This is what the registry is FOR. If adding a backend ever needs an edit to `_eval_jacobian` or
    `_eval_hessian` again, this test is the one that should start failing.
    """

    FAMILY = "scaled_fd"

    def setup_method(self):
        import jno.trace_evaluator as te
        from jno.utils.schemes import SCHEME_FAMILIES

        # A family that differentiates stored VALUES on the mesh: scale the FD result by 10 so the
        # plumbing is verifiable without inventing new numerics.
        def _grad(mesh, scheme):
            inner = te._fd_gradient(mesh, "finite_difference")
            return lambda u_1d, axis: 10.0 * inner(u_1d, axis)

        def _lap(mesh, scheme, dims):
            inner = te._fd_laplacian(mesh, "finite_difference", dims)
            return lambda u_1d: 10.0 * inner(u_1d)

        SCHEME_FAMILIES[self.FAMILY] = "test-only: finite differences scaled by 10"
        te._MESH_FIELD_FAMILIES[self.FAMILY] = te._MeshFieldBackend(_grad, _lap, None)

    def teardown_method(self):
        import jno.trace_evaluator as te
        from jno.utils.schemes import SCHEME_FAMILIES

        SCHEME_FAMILIES.pop(self.FAMILY, None)
        te._MESH_FIELD_FAMILIES.pop(self.FAMILY, None)

    def _frozen(self):
        import numpy as np

        d = jno.Shape.rect(0, 0, 1, 1, size=1 / 8).domain(structured=True)
        x, y, _ = d.variable("interior")
        P = np.asarray(d.mesh_connectivity["points"])[:, :2]
        u, _v = d.fem_symbols()
        return u.bind(x=x, y=y).freeze(np.sin(2 * np.pi * P[:, 0])), x, y

    def test_the_new_family_is_reachable_and_used(self):
        import numpy as np

        f, x, _y = self._frozen()
        base = np.asarray(f.d(x, scheme="finite_difference").eval()).reshape(-1)
        scaled = np.asarray(f.d(x, scheme=self.FAMILY).eval()).reshape(-1)
        np.testing.assert_allclose(scaled, 10.0 * base, rtol=1e-6)

    def test_it_works_for_the_laplacian_too(self):
        import numpy as np

        f, x, y = self._frozen()
        base = np.asarray(f.d2(x, scheme="finite_difference").eval()).reshape(-1)
        scaled = np.asarray(f.d2(x, scheme=self.FAMILY).eval()).reshape(-1)
        np.testing.assert_allclose(scaled, 10.0 * base, rtol=1e-6)

    def test_and_it_is_gone_again_after_teardown(self):
        """Registration is data, so unregistering restores the previous behaviour exactly."""
        from jno.utils.schemes import scheme_family

        assert scheme_family(self.FAMILY) == self.FAMILY  # registered right now
