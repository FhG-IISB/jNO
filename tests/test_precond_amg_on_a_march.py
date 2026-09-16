"""A preconditioner that needs an assembled matrix must work on a NONLINEAR transient march.

A march linearises matrix-free inside ``lax.scan``, so a spec that must see a concrete matrix -- an
unbuilt ``jno.precond.amg()``, ``ilu()`` -- was handed a tracer and died with "AMG setup needs a
concrete matrix but got a traced one", six frames inside the Newton loop, on the perfectly reasonable
``fem.solve(precond=jno.precond.amg())`` over a transient problem. The step tangent at the initial
state IS concrete at compose time, so the hierarchy is built there, once, and frozen for the march.

Oracles. A preconditioner changes convergence SPEED, never the answer: every solve here is checked
against the same march with the default solver. And the hierarchy must be built ONCE for the whole
march, not per step -- otherwise "it works" would hide a per-step host setup inside the loop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("pyamg", reason="jno.precond.amg needs the optional pyamg")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _nonlinear_heat(size=0.16, nt=5):
    """``u_t = div((1 + u^2) grad u)``: a genuinely NONLINEAR transient block (mass + residual)."""
    d = jno.Shape.rect(0, 0, 1, 1, size=size).domain(time=(0.0, 0.04, nt))
    u, v = d.fem_symbols(order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.fn(lambda x, y: jnp.sin(np.pi * x) * jnp.sin(np.pi * y), [x0, y0])
    kappa = 1.0 + ui * ui
    fem = jno.fem([ui.t * vi + kappa * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(x0, y0) - ic])
    assert fem._op.is_nonlinear(), "this test needs a NONLINEAR transient block"
    return fem


def test_amg_preconditions_a_nonlinear_march():
    """The regression: this used to raise TypeError from inside the scan."""
    ref = np.asarray(_nonlinear_heat().solve().fn())
    got = np.asarray(_nonlinear_heat().solve(linear=jno.solve.fgmres(tol=1e-10), precond=jno.precond.amg()).fn())
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-8, "a preconditioner changed the answer"


def test_the_hierarchy_is_built_once_for_the_whole_march(monkeypatch):
    """Frozen, not per step: the host-side pyamg setup must not run inside the loop at all."""
    import jno.utils.solver.amg as amg_mod

    calls = []
    real = amg_mod.build_hierarchy
    monkeypatch.setattr(amg_mod, "build_hierarchy", lambda A, **kw: (calls.append(1), real(A, **kw))[1])
    _nonlinear_heat().solve(linear=jno.solve.fgmres(tol=1e-10), precond=jno.precond.amg()).fn()
    assert len(calls) == 1, f"the hierarchy was built {len(calls)} times; a march must build it once"


def test_a_traceable_spec_is_left_alone():
    """jacobi reads its diagonal off the traced operator (with an assembled tangent, which is the
    configuration it works in today), so it must NOT be frozen -- it keeps refreshing per
    linearisation, and the answer is unchanged."""
    ref = np.asarray(_nonlinear_heat().solve().fn())
    got = np.asarray(
        _nonlinear_heat()
        .solve(nonlinear=jno.solve.newton(direct=True), linear=jno.solve.fgmres(tol=1e-10), precond=jno.precond.jacobi())
        .fn()
    )
    assert np.abs(got - ref).max() < 1e-8


def test_block_children_get_concrete_sub_blocks():
    """`triangular` hands each child `ctx.sub(i)`; with the top-level materialisation concrete, an
    AMG child on every field block builds from a real sub-matrix rather than a tracer."""

    def build():
        d = jno.Shape.rect(0, 0, 1, 1, size=0.2).domain(time=(0.0, 0.04, 4))
        a, va = d.fem_symbols(order=1, names=("a", "va"))
        b, vb = d.fem_symbols(order=1, names=("b", "vb"))
        xi, yi, ti = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        x0, y0, _ = d.variable("initial", split=True)
        ai, vai = a.bind(x=xi, y=yi, t=ti), va.bind(x=xi, y=yi, t=ti)
        bi, vbi = b.bind(x=xi, y=yi, t=ti), vb.bind(x=xi, y=yi, t=ti)
        ic = jno.fn(lambda x, y: jnp.sin(np.pi * x) * jnp.sin(np.pi * y), [x0, y0])
        return (
            d,
            a,
            b,
            jno.fem(
                [
                    ai.t * vai + (1.0 + ai * ai) * (ai.x * vai.x + ai.y * vai.y) - bi * vai,
                    bi.t * vbi + (bi.x * vbi.x + bi.y * vbi.y),
                    a(xb, yb) - 0.0,
                    b(xb, yb) - 0.0,
                    a(x0, y0) - ic,
                    b(x0, y0) - 0.0,
                ]
            ),
        )

    _, _, _, fem0 = build()
    ref = np.asarray(fem0.solve().fn())
    _, a, b, fem1 = build()
    got = np.asarray(
        fem1.solve(
            linear=jno.solve.fgmres(tol=1e-10),
            precond=jno.precond.triangular((a, jno.precond.amg()), (b, jno.precond.amg())),
        ).fn()
    )
    assert np.abs(got - ref).max() < 1e-8


def test_the_frozen_wrapper_hands_back_the_applier_untouched():
    """The applier is a ``PrecondApplier(fwd, transpose)``, and the reverse-mode adjoint solve asks for
    that transpose. Freezing must therefore pass the OBJECT through, not re-wrap its forward half --
    otherwise a march would still solve while its gradient quietly lost its preconditioner."""
    from jno.utils.solver.solver_api import _FrozenMarchPrecond

    sentinel = object()
    frozen = _FrozenMarchPrecond(sentinel, "amg()")
    assert frozen(None) is sentinel
    assert frozen("a different ctx entirely") is sentinel, "the frozen applier must ignore per-solve context"
    assert "amg()" in repr(frozen), "repr feeds the driver's cache key; it must name what was frozen"


def test_amg_now_solves_at_a_size_that_used_to_amplify():
    """4751 dofs: this march stalled at a 6e-02 relative residual, because the Chebyshev bound for the
    smoother was taken from a power iteration that had stalled at 0.54*rho and the V-cycle amplified
    7.5x. With the bound verified (see test_fem_solver_amg.py) the same march solves."""
    fem = _nonlinear_heat(size=0.016, nt=11)
    got = np.asarray(fem.solve(linear=jno.solve.fgmres(tol=1e-8), precond=jno.precond.amg()).fn())
    ref = np.asarray(_nonlinear_heat(size=0.016, nt=11).solve().fn())
    assert np.abs(got - ref).max() < 1e-7, "a preconditioner changed the answer"


def test_an_amplifying_preconditioner_is_refused_at_compose_time():
    """The backstop for what a spectral bound cannot catch -- Chebyshev on a complex spectrum, say.
    A preconditioner that makes the residual worse cannot converge, and inside a march that surfaces
    only as "fgmres did not solve the system" from a debug callback, after the whole trajectory has
    been traced. One probe at compose time says it plainly instead."""

    class _Amplifier:
        traceable = False  # so the march freezes it, which is where the probe lives
        name = "amplifier()"

        def materialize(self, _ctx):
            return lambda v: -10.0 * v

    fem = _nonlinear_heat()
    with pytest.raises(ValueError, match="WORSE"):
        fem.solve(linear=jno.solve.fgmres(tol=1e-8), precond=_Amplifier()).fn()


def _nonlinear_mass_heat(size=0.16, nt=5):
    """``c(u) u_t = div(grad u)``: a STATE-DEPENDENT mass. There is no mass MATRIX at all -- the mass
    action lives in a residual whose Jacobian is assembled per state -- which is the shape a latent-heat
    (enthalpy-porosity) melt pool takes, where its heat capacity depends on temperature."""
    d = jno.Shape.rect(0, 0, 1, 1, size=size).domain(time=(0.0, 0.04, nt))
    u, v = d.fem_symbols(order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.fn(lambda x, y: jnp.sin(np.pi * x) * jnp.sin(np.pi * y), [x0, y0])
    fem = jno.fem([(1.0 + ui * ui) * ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(x0, y0) - ic])
    return fem


def test_amg_preconditions_a_march_whose_mass_depends_on_the_state():
    """The step tangent is ``J_spatial + J_mass/dt`` there, not ``M + theta*dt*J``: asking such a block
    for a mass matrix got None and refused the preconditioner by name. Found on the melt pool, whose
    heat capacity carries the latent heat."""
    fem = _nonlinear_mass_heat()
    assert fem._op.mass is None and fem._op.mass_residual is not None, "this test needs the nonlinear-mass path"
    got = np.asarray(fem.solve(linear=jno.solve.fgmres(tol=1e-10), precond=jno.precond.amg()).fn())
    ref = np.asarray(_nonlinear_mass_heat().solve().fn())
    assert np.abs(got - ref).max() < 1e-8, "a preconditioner changed the answer"


def test_a_block_of_traceable_specs_is_not_frozen():
    """`triangular`/`block_diag` assemble nothing themselves, so a tree whose LEAVES are all traceable
    must keep the per-linearisation path. Freezing it on the container's own flag then refused it at the
    probe -- a block-triangular applier need not reduce a full residual in one application the way a
    V-cycle does -- which broke a melt-pool configuration measured at 0.15 s/step."""
    fem0 = _nonlinear_heat()
    ref = np.asarray(fem0.solve().fn())
    fem1 = _nonlinear_heat()
    u = fem1.domain.fem_symbols(order=1)[0]
    got = np.asarray(
        fem1.solve(
            nonlinear=jno.solve.newton(direct=True),
            linear=jno.solve.fgmres(tol=1e-10),
            precond=jno.precond.block_diag((0, jno.precond.jacobi())),
        ).fn()
    )
    assert np.abs(got - ref).max() < 1e-8
    del u
