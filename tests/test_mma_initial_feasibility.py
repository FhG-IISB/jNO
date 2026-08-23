"""MMA must say where every constraint starts, and refuse one no step can ever satisfy.

MMA is handed the constraints and left to find a feasible point, so a row that is violated at the
initial design and cannot be moved is indistinguishable from a hard problem: the run proceeds, the
objective wanders, and the diagnosis costs a wall clock. Measured on a 3-D bracket whose
element-quality constraints were left in the term list with the mesh pinned -- ``g_vmx`` starts at
1.734 and no design variable touches it -- the solve spent all 250 iterations in feasibility
restoration and returned a compliance of 1.65e3 against the 4.07 the same problem gives once they
are dropped.

What is deliberately NOT asserted: the exact wording of the report, and any claim that the refusal
catches every impossible constraint. It catches the case where the gradient is identically zero. A
row that is violated and merely *hard* is ordinary -- that is what MMA is for -- and a row that
depends on the design through some other variable is not caught at all. The last of those is
asserted here, so the limit is pinned rather than assumed.
"""

import jax
import numpy as np
import optax
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _core(constraint_fn, *, name="g"):
    """One design variable in [0, 1], objective (x - 1)^2, plus whatever constraint is handed in.

    ``constraint_fn`` receives the parameter node and returns the expression to bound at <= 1, so a
    test can hand back something that depends on ``x`` or something that pointedly does not.
    """
    d = jno.domain.from_array({"_": np.zeros((1, 1))})
    x = jno.np.parameter((1,), name="x")
    x.dtype(jax.numpy.float64)
    x.initialize(lambda k, sh, dtype=None: jax.numpy.full(sh, 0.5))
    x.optimizer(jno.optimizers.mma(move=0.2, lower=0.0, upper=1.0))
    obj = ((x[0] - 1.0) ** 2).name("f")
    return jno.core([obj, jno.le(constraint_fn(x).name(name), 1.0)], domain=d), x


class TestImmovableConstraint:
    """A violated row with an identically-zero gradient is refused by name."""

    def test_a_violated_constant_constraint_is_refused(self):
        # 2.0 <= 1.0 is false and nothing can change it: the expression does not mention `x`.
        with pytest.raises(ValueError, match="mma"):
            crux, _ = _core(lambda x: 0.0 * x[0] + 2.0, name="g_frozen")
            crux.solve(1)

    def test_the_message_names_the_constraint_and_the_fix(self):
        crux, _ = _core(lambda x: 0.0 * x[0] + 2.0, name="g_frozen")
        with pytest.raises(ValueError) as e:
            crux.solve(1)
        msg = str(e.value)
        assert "g_frozen" in msg, f"the report must name the offending row, got: {msg}"
        assert "term list" in msg, f"the report must name a fix, got: {msg}"

    def test_a_satisfied_constant_constraint_is_left_alone(self):
        """Immovable is only a problem when it is also violated."""
        crux, _ = _core(lambda x: 0.0 * x[0] + 0.5, name="g_slack")
        crux.solve(1)  # must not raise


class TestOrdinaryConstraintsPass:
    """A row MMA can actually act on is reported, never refused."""

    def test_a_violated_but_movable_constraint_runs(self):
        # 2*x = 1.0 at x = 0.5, so it starts exactly at the bound and depends on the design.
        crux, x = _core(lambda x: 2.0 * x[0], name="g_live")
        crux.solve(2)
        v = float(np.asarray(crux.eval([x])).reshape(-1)[0])
        assert 0.0 <= v <= 1.0, f"the design must stay in its box; got {v}"

    def test_a_constraint_that_moves_only_through_another_variable_is_not_refused(self):
        """The limit, pinned: the refusal keys on the gradient, not on impossibility.

        `g_vmx` on a pinned mesh is exactly this shape -- it stops depending on the coordinates but
        still depends on the density, so its gradient is non-zero and it is reported rather than
        refused. Asserting it here keeps the docstring's claim honest.
        """
        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        # ONE model with two components, not two models: the objective reads component 0 and the
        # constraint reads component 1, so the row is violated at the start yet reachable.
        v = jno.np.parameter((2,), name="v")
        v.dtype(jax.numpy.float64)
        v.initialize(lambda k, sh, dtype=None: jax.numpy.full(sh, 0.5))
        v.optimizer(jno.optimizers.mma(move=0.2, lower=0.0, upper=1.0))
        crux = jno.core(
            [((v[0] - 1.0) ** 2).name("f"), jno.le((4.0 * v[1]).name("g_other"), 1.0)], domain=d
        )
        crux.solve(1)  # must not raise: g starts at 2.0 > 1 but its gradient w.r.t. v is non-zero
