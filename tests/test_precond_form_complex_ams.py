"""``jno.precond.form(<complex terms>, inner=ams())`` — the complex-shifted twin of a Maxwell operator.

``_Form`` had no complex-native route, so a complex form with an AMS inner was assembled and handed to AMS
as the fused real-equivalent ``2n`` block, which AMS then refused as a "MIXED system". A form whose inner
spec is complex-native now makes the solve complex-native and is assembled as the complex ``n x n``
operator ``A_r + i A_i`` (reduced with ``P^H`` on a tied system).
"""

import contextlib

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

import jno  # noqa: E402

inner, vec = jno.np.inner, jno.np.vector
_CPU = next(iter(jax.devices("cpu")), None)
_ON_CPU = jax.default_device(_CPU) if _CPU else contextlib.nullcontext()


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _solve(fem, **kw):
    with _ON_CPU:
        return np.asarray(jnp.asarray(fem.solve(**kw))).reshape(-1)


def _maxwell(k=2.0, size=0.3):
    """A driven complex N1E Maxwell cell: curl-curl - k^2 mass, a first-order absorbing face term
    ``-1j k (n x u).(n x v)`` and a source. Returns ``(fem, twin)``: the system and its complex-shifted
    twin terms (``k^2 -> (1 - 0.5j) k^2``) for the preconditioner."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    nvec = d.variable("boundary", normals=True)
    face = -1j * k * inner(u.vector.cross(nvec), v.vector.cross(nvec))
    Js = vec(0.0 * x, jno.np.sin(2.0 * z), 1.0 + 0.0 * z)
    fem = jno.fem([inner(cu, cv) - k**2 * inner(ui, vi) - inner(Js, vi), face])
    twin = [inner(cu, cv) - (1 - 0.5j) * k**2 * inner(ui, vi), face]
    return fem, twin


def test_complex_form_with_an_ams_inner_solves_the_system():
    fem, twin = _maxwell()
    x_lu = _solve(fem, linear=jno.solve.lu())
    pre = jno.precond.form(twin, inner=jno.precond.ams())
    x = _solve(fem, linear=jno.solve.fgmres(tol=1e-9, restart=80, maxiter=400), precond=pre)
    assert x.dtype == np.complex128
    assert np.linalg.norm(x - x_lu) / np.linalg.norm(x_lu) < 1e-6


def test_form_follows_its_inner_spec_for_complex_native():
    twin = _maxwell()[1]
    assert jno.precond.form(twin, inner=jno.precond.ams()).complex_native
    assert not jno.precond.form(twin, inner=jno.precond.jacobi()).complex_native
    assert not jno.precond.form(twin).complex_native


def test_ams_names_the_real_equivalent_block_when_handed_one():
    """A 2n x 2n operator of the edge space is the real-equivalent block of a complex system, not a mixed one."""
    from jno.utils.solver.solver_api import LinearOperator, PrecondContext

    fem, _ = _maxwell()
    spec = jno.precond.ams()
    spec._transfer(fem)
    n_e = int(spec._G.shape[0])
    eye = jsparse.BCOO.fromdense(jnp.eye(2 * n_e))
    with pytest.raises(ValueError, match="real-equivalent block"):
        spec._assemble_aux(PrecondContext(LinearOperator(eye), fem))


@pytest.mark.parametrize("phase", [1.0, np.exp(1j * 0.7)], ids=["periodic", "bloch"])
def test_complex_form_with_an_ams_inner_on_a_tied_system(phase):
    """The complex form is reduced with P^H like the system (a Bloch phase makes P complex)."""
    k = 2.0
    d = jno.domain(jno.shape.box(0, 0, 0, 0.6, 0.6, 1.0, size=0.3))
    e = 1e-6
    d.tag("left", lambda x, y, z: x < e)
    d.tag("right", lambda x, y, z: x > 0.6 - e)
    d.tag("front", lambda x, y, z: y < e)
    d.tag("back", lambda x, y, z: y > 0.6 - e)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    Js = vec(jno.np.sin(3.0 * z), 0.0 * x, 1.0 + 0.0 * z)

    def face(nm):
        cc = d.variable(nm, split=True)
        return u.bind(x=cc[0], y=cc[1], z=cc[2])

    ties = [face("left") - phase * face("right"), face("front") - face("back")]
    fem = jno.fem([inner(cu, cv) - k**2 * (1 + 0.3j) * inner(ui, vi) - inner(Js, vi), *ties])
    twin = [inner(cu, cv) - k**2 * (1 + 0.9j) * inner(ui, vi)]
    x_lu = _solve(fem, linear=jno.solve.lu())
    pre = jno.precond.form(twin, inner=jno.precond.ams())
    sol = _solve(fem, linear=jno.solve.fgmres(tol=1e-9, restart=80, maxiter=400), precond=pre)
    assert np.linalg.norm(sol - x_lu) / np.linalg.norm(x_lu) < 1e-6
