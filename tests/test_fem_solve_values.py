"""``fem.solve(param=value)`` — solve a parametric problem at a value, without rebuilding it.

A parametric problem resolves its parameters through a ``crux`` evaluation, which leaves no way to say
"solve at this value". The workaround is to rebuild `jno.fem` for each value, which re-meshes,
re-assembles and re-compiles the whole problem to change one number -- and that rebuild is the
compile-dominated cost of any continuation written as a Python loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

meshio = pytest.importorskip("meshio")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _diffusion(k_value=None):
    """``-div((1 + k u^2) grad u) = 1``; ``k`` runtime when ``k_value`` is None, constant otherwise."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=6).domain()
    u, v = d.fem_symbols(names=("u", "v"), order=1)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    k = jno.np.parameter((1,), name="k") if k_value is None else k_value
    return jno.fem([(1.0 + k * ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])


def _slip_stokes():
    """Taylor-Hood with `n.u = 0` eliminated -- a REDUCED parametric system, the configuration that
    needs the sparse-direct Newton because the matrix-free one has no saddle-point preconditioner."""
    from jno.domain.geometries import Geometries

    mesh, _, _ = Geometries.equi_distant_box(nx=3, ny=3, nz=3)(None)
    d = jno.domain(lambda g: (mesh, 3, 1 / 3), compute_mesh_connectivity=True)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    x, y, z, _ = d.variable("interior", split=True)
    eu, ev = jno.np.symgrad(u, [x, y, z]), jno.np.symgrad(v, [x, y, z])
    dd = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    pp, qq = p.bind(x=x, y=y, z=z), q.bind(x=x, y=y, z=z)
    c = d.variable("boundary", normals=True, split=True)
    ur = u.bind(x=c[0], y=c[1], z=c[2])
    k = jno.np.parameter((1,), name="visc")
    fem = jno.fem(
        [
            2.0 * (1.0 + k * dd(eu, eu)) * dd(eu, ev) - pp * jno.np.trace(ev) - 1.0 * v.bind(x=x, y=y, z=z)[0],
            -qq * jno.np.trace(eu),
            c[-3] * ur[0] + c[-2] * ur[1] + c[-1] * ur[2] - 0.0,
            p.pin(),
        ]
    )
    assert fem._periodic is not None, "this fixture is meant to be slip-reduced"
    return fem


def test_a_value_gives_the_same_answer_as_a_form_built_at_that_constant():
    got = np.asarray(_diffusion().solve(k=2.0)).reshape(-1)
    ref = np.asarray(_diffusion(k_value=2.0).solve()).reshape(-1)
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-12)


def test_the_problem_is_staged_once_across_many_values():
    """THE point, and the oracle: rebuilding also gives the right answer, slowly, so an equivalence
    check alone passes on the implementation this exists to avoid. What has to be true is that the
    solve is staged ONCE and the value arrives as a runtime argument."""
    fem = _diffusion()
    op = fem._op
    traces = {"n": 0}
    real = op.residual

    def counting(u, *a, **kw):
        if isinstance(u, jax.core.Tracer):  # a traced u means the solve is being staged afresh
            traces["n"] += 1
        return real(u, *a, **kw)

    op.residual = counting
    try:
        for v in np.linspace(0.0, 2.0, 8):
            fem.solve(k=float(v))
    finally:
        op.residual = real
    assert traces["n"] > 0, "the counter is not wired to the solve"
    assert traces["n"] < 8, f"the solve was re-staged per value ({traces['n']} traces for 8 values)"


def test_a_warm_start_composes_with_a_value():
    fem = _diffusion()
    x = None
    for v in np.linspace(0.0, 2.0, 5):
        x = np.asarray(fem.solve(k=float(v), x0=x)).reshape(-1)
    ref = np.asarray(_diffusion(k_value=2.0).solve()).reshape(-1)
    np.testing.assert_allclose(x, ref, rtol=1e-6, atol=1e-9)


def test_a_direct_newton_on_a_reduced_parametric_system():
    """Sparse-direct Newton, a slip-eliminated system, and a runtime parameter -- all three at once.
    This is the configuration a rigid-plastic contact problem actually runs in, and the one that
    `fem.solve(continuation=...)` still cannot serve (its driver hands the solver no tangent)."""
    fem = _slip_stokes()
    x = None
    for v in np.linspace(0.0, 1.0, 4):
        x = np.asarray(fem.solve(visc=float(v), x0=x, nonlinear=jno.solve.newton(direct=True, line_search=True))).reshape(
            -1
        )
    assert x.shape[0] == fem.dofs, "the solution must come back in the FULL space"
    assert np.isfinite(x).all()


def test_the_verdict_survives_the_jit():
    """The solve is jitted to avoid re-staging, and the driver's own convergence check self-disables
    under a trace -- so `fem.stats` would otherwise report nothing, or worse, an earlier solve's
    numbers. On a reduced system the judgement must be made on the REDUCED residual: the full one keeps
    the constraint's reaction and never falls."""
    fem = _slip_stokes()
    fem.solve(visc=0.5, nonlinear=jno.solve.newton(direct=True, line_search=True))
    st = fem.stats["nonlinear"]
    assert st is not None, "no verdict was recorded for a parametric solve"
    assert st["converged"] is True, st
    assert st["residual"] <= st["bound"], st


def test_a_partly_supplied_problem_is_refused_by_name():
    """Naming one of two parameters is neither a solve nor a trace node -- say which is missing rather
    than resolving the rest from somewhere else."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=5).domain()
    u, v = d.fem_symbols(names=("u", "v"), order=1)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    a = jno.np.parameter((1,), name="a")
    b = jno.np.parameter((1,), name="b")
    fem = jno.fem([(1.0 + a * ui * ui) * (ui.x * vi.x + ui.y * vi.y) - b * vi, u(xb, yb) - 0.0])
    with pytest.raises(ValueError, match=r"no value was given for \['b'\]"):
        fem.solve(a=1.0)


def test_with_no_values_the_solve_is_still_a_trace_node():
    """The `crux` path is untouched: a parametric solve with nothing supplied stays lazy."""
    from jno.trace import Placeholder

    out = _diffusion().solve()
    assert isinstance(out, Placeholder), f"expected a trace node, got {type(out).__name__}"


def test_a_parametric_solve_is_differentiable_in_its_parameter():
    """`fem.solve(k=value)` must survive `jax.grad`.

    The verdict this path records for `fem.stats` needs a CONCRETE residual, and it used to take one
    unconditionally -- so differentiating a parametric solve raised ConcretizationTypeError deep in the
    reporting, not the solving. That closes off fitting a coefficient by backprop through the solve,
    which is the entire point of having runtime parameters. Under a trace the verdict is skipped (and
    says so in `stats`), exactly as the in-driver check already does.
    """
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34).domain()
    u, v = d.fem_symbols()
    x, y, _t = d.variable("interior", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    xb, yb, _tb = d.variable("boundary", split=True)
    k = jno.np.parameter((1,), name="k")
    # NONLINEAR on purpose: the verdict this guards is recorded only on the nonlinear path (a linear
    # parametric solve returns a trace node for `crux` and never reaches it).
    fem = jno.fem([(1.0 + k + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])

    def J(kv):
        return jnp.sum(jnp.asarray(fem.solve(k=kv)) ** 2)

    g = float(jax.grad(J)(0.5))
    h = 1e-4
    fd = (float(J(0.5 + h)) - float(J(0.5 - h))) / (2.0 * h)
    assert np.isfinite(g)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0)
