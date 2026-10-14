"""A design step is traced with poisoned residual gates: no host callback in the compiled program (so JAX's
persistent cache can store it), and a solve that fails its residual check still stops the run.

Oracles: the compiled constraint-gradient program's text holds no host-callback custom call; a solve that cannot
converge (two CG iterations) inside an MMA step raises at that step instead of handing NaN to the optimiser; the
same expression evaluated eagerly still names the failing solve (the detailed gate is unchanged there).
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

inner = jno.np.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _design(linear):
    """minimise ∫ u² with -div(k grad u) = 1, u = 0 on the left; k a scalar design variable under MMA"""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain()
    x, y = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    u, v = d.fem_symbols(names=("u", "v"))
    U, V = u.bind(x=x, y=y), v.bind(x=x, y=y)
    g = lambda f: jno.np.grad(f, [x, y])  # noqa: E731
    k = jno.np.parameter((1,), name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(jno.optimizers.mma(move=0.2, lower=0.5, upper=2.0))
    fem = jno.fem([k * inner(g(U), g(V)) - V, u(xl, yl) - 0.0])
    sol = u.bind(x=x, y=y).freeze(fem.solve(linear=linear))
    J = (sol * sol).integrate(d)
    J = getattr(J, "_expr", J)  # the node under the typed view
    K = jno.fn(lambda a: jnp.sum(a), [k], name="k_total")
    return jno.core([J, jno.le(K, 1.5)], domain=jno.domain.from_array({"_": np.zeros((1, 1))})), J


def test_the_design_step_has_no_host_callback(monkeypatch):
    texts = {}
    orig = jax.stages.Lowered.compile

    def spy(self, *a, **k):
        t = self.as_text()
        texts[t.split("module @", 1)[1].split()[0] if "module @" in t else "?"] = t
        return orig(self, *a, **k)

    monkeypatch.setattr(jax.stages.Lowered, "compile", spy)
    crux, _J = _design(jno.solve.cg())
    crux.solve(1)
    step = [t for name, t in texts.items() if "grad_jac" in name]
    assert step, f"no constraint-gradient program was compiled: {sorted(texts)}"
    assert "callback" not in step[0], "the design step still holds a host callback (it cannot be cached)"


def test_a_solve_that_fails_its_check_stops_the_design_step():
    crux, J = _design(jno.solve.cg(maxiter=2))
    with pytest.raises(RuntimeError, match="non-finite"):
        crux.solve(2)
    from jno.utils.solver.solver_api import raise_if_gate_failed

    with pytest.raises(RuntimeError, match="cg"):
        crux.eval([J])
        raise_if_gate_failed()
