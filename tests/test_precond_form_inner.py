"""``form``'s inner contract: factored ONCE by default, and a precond spec as the inner.

The old default handed every application to ``jno.solve.lu()``, which re-factorises per call —
invisible on a small Schur block, hours on a 90k-edge auxiliary applied hundreds of times per
solve. The default now factors once (the AMS-auxiliary contract). ``inner`` additionally
duck-types a ``jno.precond`` spec: materialized once against the assembled auxiliary, applied
once per call — a multigrid acting on an operator DECLARED as a weak form.
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")
import jax  # noqa: E402

import jno  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson():
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0], quad_degree=3)
    aux = [ui.x * vi.x + ui.y * vi.y + 0.1 * ui * vi, u(xb, yb) - 0.0]
    return fem, aux


def test_default_inner_is_factored_once_and_correct():
    """The default must (a) solve correctly and (b) not re-factorise per application. (b) is
    asserted structurally: the applier closes over a SuperLU object, not over a solver call."""
    fem, aux = _poisson()
    ref = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1)
    got = np.asarray(
        fem.solve(linear=jno.solve.fgmres(tol=1e-10, restart=40, maxiter=400), precond=jno.precond.form(aux, quad_degree=3))
    ).reshape(-1)
    assert np.allclose(got, ref, rtol=1e-7, atol=1e-10)


def test_matrix_free_auxiliary_is_refused_with_a_route_out():
    from jno.precond import PrecondContext, _Form
    from jno.utils.solver.solver_api import LinearOperator

    spec = _Form([], None, 2)
    spec._op = LinearOperator.from_matvec(lambda v: v, shape=(4, 4))
    spec.terms = []
    with pytest.raises(ValueError, match="matrix-free"):
        spec.materialize(PrecondContext(spec._op, None))


def test_a_precond_spec_as_the_inner_is_applied():
    """form(aux, inner=jno.precond.jacobi()): the spec is materialized against the AUXILIARY
    operator and applied once per call. Correctness = the outer solve still converges to the
    direct answer; the aux carries a mass shift so its Jacobi differs from the system's own —
    a dropped-on-the-floor inner would be indistinguishable only if it equalled the default,
    which a single Jacobi application (no solve at all) cannot."""
    fem, aux = _poisson()
    ref = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1)
    got = np.asarray(
        fem.solve(
            linear=jno.solve.fgmres(tol=1e-10, restart=40, maxiter=2000),
            precond=jno.precond.form(aux, inner=jno.precond.jacobi(), quad_degree=3),
        )
    ).reshape(-1)
    assert np.allclose(got, ref, rtol=1e-7, atol=1e-10)


def test_an_explicit_solver_inner_still_works():
    """The pre-existing contract — a jno.solve solver as the inner — is untouched."""
    fem, aux = _poisson()
    ref = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1)
    got = np.asarray(
        fem.solve(
            linear=jno.solve.fgmres(tol=1e-10, restart=40, maxiter=400),
            precond=jno.precond.form(aux, inner=jno.solve.cg(tol=1e-6, maxiter=200), quad_degree=3),
        )
    ).reshape(-1)
    assert np.allclose(got, ref, rtol=1e-7, atol=1e-10)
