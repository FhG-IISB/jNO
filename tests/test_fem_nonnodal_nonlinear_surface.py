"""N1E tangential-trace surface terms (absorbing face, incident load) in a non-nodal form with two N1E fields.

A face term coupling two N1E fields (trial ``E_i``, test ``v_r``) used to be assembled into the first N1E
field's diagonal block, a silent wrong answer. The oracle: ``E = E_r + i E_i`` as two real N1E fields must
equal the COMPLEX linear solve of the same Maxwell form, which takes the established linear impedance path.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D box meshing")
import jax  # noqa: E402

import jno  # noqa: E402

inner, vec, cross = jno.np.inner, jno.np.vector, jno.np.cross
K0 = 2 * np.pi
EPS = (0.84 + 1.65j) ** 2


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _domain():
    Bx = jno.shape.box
    d = (
        (
            Bx(0, 0, 0, 0.6, 0.6, 0.15).name("film").attach(er=EPS.real, ei=EPS.imag, k=20.0)
            + Bx(0, 0, -0.3, 0.6, 0.6, 0.45).name("rest").attach(er=1.0, ei=0.0, k=1.0)
        )
        .sized(0.3)
        .domain()
    )
    d.tag("top", lambda x, y, z: z > 0.45 - 1e-6)
    d.tag("bottom", lambda x, y, z: z < -0.3 + 1e-6)
    return d


def _incident(d, w):
    """The x-polarised surface load on the bottom face, tested with the N1E field ``w``."""
    xb = d.variable("bottom", split=True)
    nb = d.variable("bottom", normals=True)
    g = vec(1.0 + 0 * xb[0], 0 * xb[0], 0 * xb[0])
    return inner(cross(g, nb), w.vector.cross(nb))


def _complex_linear(d):
    xi, yi, zi = d.variable("interior", split=True)[:3]
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    eps = d.er + 1j * d.ei
    nt = d.variable("top", normals=True)
    return jno.fem(
        [
            inner(u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)) - K0**2 * eps * inner(ui, vi),
            -1j * K0 * inner(u.vector.cross(nt), v.vector.cross(nt)),
            -_incident(d, v),
        ]
    )


def test_the_linear_real_pair_with_cross_field_faces_matches_the_complex_form():
    d = _domain()
    xi, yi, zi = d.variable("interior", split=True)[:3]
    er_, vr_ = d.fem_symbols(value_shape=(3,), names=("Er", "vr"), space="N1E")
    ei_, vi_ = d.fem_symbols(value_shape=(3,), names=("Ei", "vi"), space="N1E")
    Er, vr, Ei, vi = (w.bind(x=xi, y=yi, z=zi) for w in (er_, vr_, ei_, vi_))
    curl = lambda w: w.vector.curl(xi, yi, zi)  # noqa: E731
    nt = d.variable("top", normals=True)
    pair = jno.fem(
        [
            inner(curl(er_), curl(vr_)) - K0**2 * (d.er * inner(Er, vr) - d.ei * inner(Ei, vr)),
            inner(curl(ei_), curl(vi_)) - K0**2 * (d.ei * inner(Er, vi) + d.er * inner(Ei, vi)),
            K0 * inner(ei_.vector.cross(nt), vr_.vector.cross(nt)),
            -K0 * inner(er_.vector.cross(nt), vi_.vector.cross(nt)),
            -_incident(d, vr_),
        ]
    )
    sol = np.asarray(pair.solve(linear=jno.solve.lu())).reshape(-1)
    ref = np.asarray(_complex_linear(d).solve(linear=jno.solve.lu())).reshape(-1)
    n = ref.size
    np.testing.assert_allclose(sol[:n] + 1j * sol[n:], ref, rtol=0, atol=1e-9 * np.abs(ref).max())
