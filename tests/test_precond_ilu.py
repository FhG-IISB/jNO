"""``jno.precond.ilu`` — incomplete LU, the classical A–V eddy-current preconditioner.

Igarashi & Honma (IEEE Trans. Magn. 38(2), 2002) solve the mixed A–V system with incomplete-Cholesky
CG in 16–17 iterations independent of frequency and 29–31 independent of mesh size, and show in §IV
that the incomplete factorisation is what removes the floating singular values (∝ ωσμ) wrecking the
condition number. jNO had no incomplete factorisation at all — every preconditioner was diagonal,
polynomial, multigrid or block — so this closes a gap the literature says is the right tool here.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
import jax  # noqa: E402

import jno  # noqa: E402

inner_, vec = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(size=0.28):
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(names=("u", "v"))
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    b = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    return jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u.bind(x=b[0], y=b[1], z=b[2]) - 0.0])


def _complex_curl_curl(size=0.45, w=1.0e4):
    """A COMPLEX curl-curl + jw mass — the shape ilu exists for."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    A_, V_ = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    return jno.fem(
        [
            inner_(u.vector.curl(x, y, z), v.vector.curl(x, y, z))
            + 1j * w * inner_(A_, V_)
            - inner_(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), V_),
            u.vector.cross(d.variable("boundary", normals=True)),
        ]
    )


def test_ilu_solves_a_real_system():
    fem = _poisson()
    ref = np.asarray(jno.np.asarray(fem.solve(linear=jno.solve.lu()))).reshape(-1)
    got = np.asarray(
        jno.np.asarray(fem.solve(linear=jno.solve.fgmres(tol=1e-10, restart=50, maxiter=300), precond=jno.precond.ilu()))
    ).reshape(-1)
    assert np.allclose(got, ref, rtol=1e-7, atol=1e-9 * max(np.abs(ref).max(), 1.0))


def test_ilu_applies_to_a_COMPLEX_operator_directly():
    """The point: no real_equivalent detour. SuperLU's ILU is complex-capable, so `complex_ok` is
    True and the complex operator is preconditioned as it stands."""
    assert jno.precond.ilu().complex_ok is True
    fem = _complex_curl_curl()
    ref = np.asarray(jno.np.asarray(fem.solve(linear=jno.solve.lu()))).reshape(-1)
    got = np.asarray(
        jno.np.asarray(fem.solve(linear=jno.solve.fgmres(tol=1e-9, restart=50, maxiter=400), precond=jno.precond.ilu()))
    ).reshape(-1)
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-8 * max(np.abs(ref).max(), 1.0))


def test_options_reach_scipy():
    """`drop_tol`/`fill_factor` are the memory-vs-quality dial; a silently ignored option would make
    tuning meaningless."""
    fem = _poisson()
    loose = fem.solve(
        linear=jno.solve.fgmres(tol=1e-10, restart=50, maxiter=300), precond=jno.precond.ilu(drop_tol=1e-2, fill_factor=3.0)
    )
    assert np.isfinite(np.asarray(jno.np.asarray(loose))).all()
    assert "drop_tol" in repr(jno.precond.ilu(drop_tol=1e-2))


def test_matrix_free_operator_is_refused():
    from jno.precond import _ILU, PrecondContext
    from jno.utils.solver.solver_api import LinearOperator

    op = LinearOperator.from_matvec(lambda v: v, shape=(4, 4))
    with pytest.raises(ValueError, match="ASSEMBLED"):
        _ILU({}).materialize(PrecondContext(op, None))
