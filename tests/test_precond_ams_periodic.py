"""``jno.precond.ams()`` on a periodic / Bloch-tied N1E system.

A tie makes the solve run on the reduced operator ``P^H A P``. AMS used to build its discrete gradient
``G`` and the nodal interpolations ``Pi`` on the FULL edge space, so the first application failed with a
bare shape error. They are now reduced with the nodal prolongation built from the same ties
(``G_r = S_e G P_n``), the auxiliary operators are the Hermitian ``G_r^H A_r G_r``, and AMS-preconditioned
CG converges in a handful of iterations independent of the mesh.
"""

import contextlib

import jax
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


def _periodic_curlcurl(size, phase=1.0, mass=1.0):
    """curl-curl + mass N1E on a box, tied left/right (with ``phase``) and front/back, with a source."""
    d = jno.domain(jno.shape.box(0, 0, 0, 0.6, 0.6, 1.0, size=size))
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
    return jno.fem([inner(cu, cv) + mass * inner(ui, vi) - inner(Js, vi), *ties])


@pytest.mark.parametrize("size", [0.3, 0.2])
def test_ams_cg_on_a_periodic_curl_curl_converges_in_few_iterations(size):
    """The reduced discrete gradient and interpolations make AMS-CG converge in a small iteration budget
    (cg raises when it exhausts ``maxiter``), at both refinements, and reach the direct answer."""
    fem = _periodic_curlcurl(size)
    assert fem._periodic is not None and fem._periodic["n_red"] < fem._periodic["n_full"]
    x_lu = _solve(fem, linear=jno.solve.lu())
    x = _solve(fem, linear=jno.solve.cg(tol=1e-9, maxiter=18), precond=jno.precond.ams())
    assert np.linalg.norm(x - x_lu) / np.linalg.norm(x_lu) < 1e-6


def test_ams_build_on_a_periodic_system_reduces_the_transfers():
    fem = _periodic_curlcurl(0.3)
    spec = jno.precond.ams().build(fem)
    n_red = int(fem._periodic["n_red"])
    assert spec._G.shape[0] == n_red and all(P.shape[0] == n_red for P in spec._Pis)
    assert spec._G.shape[1] < fem.domain._fem_nonnodal_topology["n_verts"]  # periodic nodes are merged
    x_lu = _solve(fem, linear=jno.solve.lu())
    x = _solve(fem, linear=jno.solve.cg(tol=1e-9, maxiter=18), precond=spec)
    assert np.linalg.norm(x - x_lu) / np.linalg.norm(x_lu) < 1e-6


def test_ams_gmres_on_a_bloch_tied_curl_curl():
    """A Bloch phase makes the prolongation, the reduced gradient and the operator complex; AMS follows."""
    fem = _periodic_curlcurl(0.3, phase=np.exp(1j * 0.7))
    assert fem._periodic.get("is_bloch")
    x_lu = _solve(fem, linear=jno.solve.lu())
    x = _solve(fem, linear=jno.solve.gmres(tol=1e-9, restart=60, maxiter=200), precond=jno.precond.ams())
    assert np.linalg.norm(x - x_lu) / np.linalg.norm(x_lu) < 1e-6
