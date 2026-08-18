"""``fem.solve(continuation=...)`` — march a runtime parameter, warm-starting each solve.

The engine (``run_continuation``) predates this file; what it lacked was a public builder and the
ability to run on a **reduced** system. Both are covered here.

**The oracle for the no-rebuild claim is the TRACE COUNT, not the answer.** Rebuilding the form at
every value also produces the right answer -- slowly -- so an equivalence assertion alone would pass
on the implementation this exists to avoid. What has to be true is that the operator is traced once
and the parameter arrives as a runtime argument, so an 8-step ramp costs 8 solves rather than 8
compilations. That is asserted directly by counting how often the residual is traced.
"""

from __future__ import annotations

import jax
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


def _nonlinear_diffusion(k_value=None):
    """``-div((1 + k u^2) grad u) = 1`` with u = 0 on the boundary.

    ``k`` is a runtime parameter when ``k_value`` is None, and a plain constant otherwise -- the two
    spellings assemble the same problem, which is what the equivalence test needs.
    """
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=6).domain()
    u, v = d.fem_symbols(names=("u", "v"), order=1)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    k = jno.np.parameter((1,), name="k") if k_value is None else k_value
    flux = (1.0 + k * ui * ui) * (ui.x * vi.x + ui.y * vi.y)
    return jno.fem([flux - 1.0 * vi, u(xb, yb) - 0.0])


def test_continuation_is_a_public_slot():
    """The engine was reachable only by importing the private driver; the builder is the missing half."""
    spec = jno.solve.continuation(k=np.linspace(0.0, 1.0, 3))
    assert spec.keep == "last"
    assert "k" in spec.params
    assert jno.solve.continuation(keep="all", k=[0.0, 1.0]).keep == "all"


def test_the_form_is_traced_ONCE_across_the_whole_march():
    """The point of the slot. A loop that rebuilds gives the same answer and pays N compilations."""
    fem = _nonlinear_diffusion()
    op = fem._op
    traces = {"n": 0}
    real_residual = op.residual

    def counting_residual(u, *a, **kw):
        # count TRACES, not calls: a traced `u` means the function is being staged out afresh.
        if isinstance(u, jax.core.Tracer):
            traces["n"] += 1
        return real_residual(u, *a, **kw)

    op.residual = counting_residual
    try:
        fem.solve(continuation=jno.solve.continuation(k=np.linspace(0.0, 2.0, 8)))
    finally:
        op.residual = real_residual

    assert traces["n"] > 0, "the residual was never traced — the counter is not wired to the solve"
    # The Newton body stages a handful of times (loop body, tangent, line search) but ONCE for the
    # whole march, not once per rung: 8 values must not cost 8 stagings. Before the step was jitted
    # this measured 48.
    assert traces["n"] < 8, f"the form was re-traced per step ({traces['n']} traces for 8 values)"


def test_the_marched_answer_equals_a_form_built_at_that_constant():
    """Warm-starting must not change the root it converges to."""
    marched = np.asarray(
        _nonlinear_diffusion().solve(continuation=jno.solve.continuation(k=np.linspace(0.0, 2.0, 5)))
    ).reshape(-1)
    direct = np.asarray(_nonlinear_diffusion(k_value=2.0).solve()).reshape(-1)
    np.testing.assert_allclose(marched, direct, rtol=1e-6, atol=1e-8)


def test_keep_all_returns_the_family():
    vals = np.linspace(0.0, 2.0, 4)
    fam = np.asarray(_nonlinear_diffusion().solve(continuation=jno.solve.continuation(keep="all", k=vals)))
    assert fam.shape[0] == len(vals)
    # a stiffer coefficient means a smaller peak: the family must actually vary with the parameter
    peaks = np.abs(fam).max(axis=1)
    assert peaks[0] > peaks[-1], f"the marched family does not depend on the parameter: {peaks}"


def test_a_slip_reduced_system_can_be_continued():
    """Previously refused outright: `_periodic` is how the slip elimination is carried, so a
    slip-constrained problem could not be continued at all -- which is exactly the shape of problem
    (a rigid-plastic roll contact) that needs homotopy most."""
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
            c[-3] * ur[0] + c[-2] * ur[1] + c[-1] * ur[2] - 0.0,  # slip  n.u = 0
            p.pin(),
        ]
    )
    assert fem._periodic is not None, "this problem is meant to be slip-reduced"
    out = np.asarray(
        fem.solve(
            # matrix-free Newton: the continuation driver hands the solver a residual and no
            # assembled tangent, so `newton(direct=True)` refuses here (see the xfail below).
            continuation=jno.solve.continuation(visc=np.linspace(0.0, 1.0, 3)),
        )
    ).reshape(-1)
    assert out.shape[0] == fem.dofs, "the marched solution must come back in the FULL space"
    assert np.isfinite(out).all()


def test_a_swept_and_fixed_value_for_one_parameter_is_refused():
    fem = _nonlinear_diffusion()
    with pytest.raises(ValueError, match="marched or held"):
        fem.solve(continuation=jno.solve.continuation(k=[0.0, 1.0]), k=0.5)


def test_an_unknown_parameter_name_is_refused_by_name():
    fem = _nonlinear_diffusion()
    with pytest.raises(TypeError, match="unknown runtime parameter"):
        fem.solve(continuation=jno.solve.continuation(not_a_parameter=[0.0, 1.0]))


def test_a_direct_newton_on_a_reduced_system_refuses_by_name():
    """The driver hands the solver a residual and no assembled tangent, so the sparse-direct Newton
    has nothing to factorize. It refuses clearly rather than solving something else -- but this is a
    real limitation: `fem.solve` itself supports `newton(direct=True)` on a reduced system (it
    reduces the tangent, PᵀJP), and this driver does not yet do the same."""
    from jno.domain.geometries import Geometries

    mesh, _, _ = Geometries.equi_distant_box(nx=2, ny=2, nz=2)(None)
    d = jno.domain(lambda g: (mesh, 3, 0.5), compute_mesh_connectivity=True)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=2)
    p_, q = d.fem_symbols(names=("p", "q"), order=1)
    x, y, z, _ = d.variable("interior", split=True)
    eu, ev = jno.np.symgrad(u, [x, y, z]), jno.np.symgrad(v, [x, y, z])
    dd = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    pp, qq = p_.bind(x=x, y=y, z=z), q.bind(x=x, y=y, z=z)
    c = d.variable("boundary", normals=True, split=True)
    ur = u.bind(x=c[0], y=c[1], z=c[2])
    k = jno.np.parameter((1,), name="visc")
    fem = jno.fem(
        [
            2.0 * (1.0 + k * dd(eu, eu)) * dd(eu, ev) - pp * jno.np.trace(ev),
            -qq * jno.np.trace(eu),
            c[-3] * ur[0] + c[-2] * ur[1] + c[-1] * ur[2] - 0.0,
            p_.pin(),
        ]
    )
    with pytest.raises(ValueError, match="ASSEMBLED Jacobian"):
        fem.solve(
            continuation=jno.solve.continuation(visc=[0.0, 1.0]),
            nonlinear=jno.solve.newton(direct=True),
        )


def test_a_stalled_rung_raises_naming_the_rung():
    """The guard the jit would otherwise take away.

    The per-step solve is staged once and run under ``jax.jit``, and the Newton driver's own
    stalled-solve check self-disables inside a trace (it needs a concrete residual). Without an
    eager replacement, a rung that leaves on its step cap would return its last iterate silently --
    and the march would then carry that non-root into every later rung as the warm start, so one
    quiet stall corrupts the whole family rather than one entry.
    """
    fem = _nonlinear_diffusion()
    with pytest.raises(RuntimeError, match=r"step \d+/8 at k=.*did not converge"):
        fem.solve(
            continuation=jno.solve.continuation(k=np.linspace(0.0, 2.0, 8)),
            nonlinear=jno.solve.newton(max_steps=1, rtol=1e-14, atol=1e-14),
        )


def test_stats_describe_the_last_rung_not_a_stale_entry():
    """``LAST_NEWTON_STATS`` is written by the in-driver check, which is blind under the jit -- so it
    would otherwise keep whatever a previous *eager* solve left behind, and `fem.stats` would report
    a number that has nothing to do with the march."""
    fem = _nonlinear_diffusion()
    fem.solve(continuation=jno.solve.continuation(k=np.linspace(0.0, 2.0, 5)))
    st = fem.stats["nonlinear"]
    assert st["driver"] == "continuation/newton", st
    assert st["converged"] is True
    assert st["residual"] <= st["bound"], st
