"""Complex assembly on the non-nodal (Nédélec / N1E) path — the Re/Im coefficient split.

The N1E basis is real, so a complex weak form ``∫curl u·curl v + i·∫u·v`` assembles as two real
systems (the ``.real``/``.imag`` legs) that ``FEM.solve()`` recombines as ``[[K,-M],[M,K]]`` into a
complex solution. This is the foundation for time-harmonic Maxwell in ``jno.fem`` (complex ε and the
``i k₀`` impedance BC). Before this, the plain real assembler SILENTLY cast the imaginary part away —
so the unwired compositions must now raise, never mislead.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

inner = jno.np.inner
_dense = lambda A: np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _n1e_cube(mesh_size=0.5):
    d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    return d, (xi, yi, zi), (ui, vi), (cu, cv)


def test_complex_coefficient_is_not_silently_dropped():
    """A complex mass coefficient on N1E must survive assembly. The real-equivalent block routes it, so
    the recovered solution is genuinely complex (a nonzero imaginary part) — not the real cast the plain
    assembler used to produce."""
    vec = jno.np.vector
    d, (xi, yi, zi), (ui, vi), (cu, cv) = _n1e_cube(0.5)
    fvec = vec(1.0 + 0.0 * xi, 0.0 * yi, 0.0 * zi)  # a real forcing on E_x

    # (K + iM) u = L  — curl-curl stiffness K, mass M, load L from the real forcing
    K = _dense(jno.fem([inner(cu, cv)]).A)
    M = _dense(jno.fem([inner(ui, vi)]).A)
    L = np.asarray(jnp.asarray(jno.fem([inner(cu, cv) - inner(fvec, vi)]).b)).reshape(-1)
    u_ref = np.linalg.solve(K + 1j * M, L)  # the complex reference

    fem = jno.fem([inner(cu, cv) + 1j * inner(ui, vi) - inner(fvec, vi)])
    assert fem.is_complex  # routed through the real-equivalent complex path
    u = np.asarray(jnp.asarray(fem.solve())).reshape(-1)

    assert np.iscomplexobj(u) and np.max(np.abs(u.imag)) > 1e-6, "imaginary part was dropped"
    np.testing.assert_allclose(u, u_ref, rtol=1e-8, atol=1e-9)


def test_imaginary_leg_is_the_mass_matrix():
    """The Im-coefficient leg of ``curl-curl + i·mass`` is exactly the N1E mass matrix, and the Re leg is
    exactly the curl-curl matrix — the split is faithful, not approximate."""
    d, _, (ui, vi), (cu, cv) = _n1e_cube(0.6)
    K = _dense(jno.fem([inner(cu, cv)]).A)
    M = _dense(jno.fem([inner(ui, vi)]).A)

    fem = jno.fem([inner(cu, cv) + 1j * inner(ui, vi)])
    op_r, op_i = fem._op  # (A_r, b_r), (A_i, b_i)
    np.testing.assert_allclose(_dense(op_r[0]), K, atol=1e-10)
    np.testing.assert_allclose(_dense(op_i[0]), M, atol=1e-10)


def test_complex_parametric_nonnodal_raises_not_silent():
    """A complex N1E *parametric/inverse* form is not wired — it must RAISE (the earlier behaviour silently
    cast the imaginary part to zero, a silently-wrong solve)."""
    d, (xi, yi, zi), (ui, vi), (cu, cv) = _n1e_cube(0.6)
    a = jno.np.parameter((), name="a").initialize(jax.nn.initializers.constant(2.0))  # a runtime parameter in ε
    with pytest.raises(NotImplementedError, match="complex non-nodal.*parametric"):
        jno.fem([inner(cu, cv) + 1j * a * inner(ui, vi)]).A
