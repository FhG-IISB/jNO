"""jno.rcwa on a VECTOR (curl-curl / Nédélec) Maxwell constraint list — the vector front door.

The same ``jno.fem``-style list that authors a time-harmonic Maxwell problem (``inner(curl u, curl v)
− K0²·eps·inner(u, v)`` + impedance/absorbing faces + incident wave + Floquet ties) is inferred by
``jno.rcwa``: the permittivity is pulled from the vector mass term, the incident polarization/angle from
the vector source, and the solve runs on fmmax (which always solves vector Maxwell). Because it feeds the
same ε to the same solver, a scalar-Helmholtz and a vector-Maxwell list of the SAME structure agree.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jno  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

inner, vec = jno.np.inner, jno.np.vector
K0 = 2 * jnp.pi
P, LZ = 0.6, 3.2


def _tag_faces(d):
    e = 1e-6
    for nm, f in [
        ("left", lambda x, y, z: x < e),
        ("right", lambda x, y, z: x > P - e),
        ("front", lambda x, y, z: y < e),
        ("back", lambda x, y, z: y > P - e),
        ("bottom", lambda x, y, z: z < e),
        ("top", lambda x, y, z: z > LZ - e),
    ]:
        d.tag(nm, f)


def _pillar(xi, yi, zi, eps_pillar=11.0):
    return jno.fn(
        lambda x, y, z: jnp.where(((x - 0.3) ** 2 + (y - 0.3) ** 2 < 0.18**2) & (z >= 0.8) & (z < 1.15), eps_pillar, 1.0),
        [xi, yi, zi],
    )


def _scalar_constraints():
    d = jno.domain(jno.Shape.box(0, 0, 0, P, P, LZ, size=0.2))
    _tag_faces(d)
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

    def fc(n):
        c = d.variable(n, split=True)
        return u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])

    ubt, vbt = fc("bottom")
    utp, vtp = fc("top")
    ul, _ = fc("left")
    ur, _ = fc("right")
    uf, _ = fc("front")
    ubk, _ = fc("back")
    eps = _pillar(xi, yi, zi)
    return [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ul - ur,
        uf - ubk,
    ]


def _vector_constraints():
    d = jno.domain(jno.Shape.box(0, 0, 0, P, P, LZ, size=0.2))
    _tag_faces(d)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    nt, nb = d.variable("top", normals=True), d.variable("bottom", normals=True)
    cb = d.variable("bottom", split=True)
    tut, tvt = u.vector.cross(nt), v.vector.cross(nt)
    tub, tvb = u.vector.cross(nb), v.vector.cross(nb)
    eps = _pillar(xi, yi, zi)
    einc = vec(1.0 + 0.0 * cb[0], 0.0 * cb[1], 0.0 * cb[2])  # x-polarised normal-incidence plane wave

    def face(n):
        cc = d.variable(n, split=True)
        return u.bind(x=cc[0], y=cc[1], z=cc[2])

    return [
        inner(cu, cv) - K0**2 * eps * inner(ui, vi),  # vector Maxwell volume
        1j * K0 * inner(tut, tvt),  # absorbing top
        1j * K0 * inner(tub, tvb) + 2j * K0 * inner(einc, tvb),  # absorbing bottom + incident
        face("left") - face("right"),  # Floquet ties
        face("front") - face("back"),
    ]


@needs_fmmax
def test_vector_front_door_infers_and_solves():
    """A vector curl-curl Maxwell list is inferred + solved by jno.rcwa: correct period/wavelength, normal
    incidence read from the vector source (k_in≈0, not corrupted by the N1E edge signs), and T+R≈1."""
    rc = jno.rcwa(_vector_constraints(), orders=60, grid=24)
    assert rc.spec.period[0] == pytest.approx(P, abs=1e-3)
    assert rc.spec.wavelength == pytest.approx(1.0, abs=1e-3)
    assert abs(rc.spec.k_in[0]) < 1e-3 and abs(rc.spec.k_in[1]) < 1e-3  # normal incidence
    sol = rc.solve()
    T, R = float(sol.efficiency("T")), float(sol.efficiency("R"))
    assert np.isfinite(T) and np.isfinite(R)
    assert T + R == pytest.approx(1.0, abs=5e-3)  # energy conserved


@needs_fmmax
def test_vector_matches_scalar_front_door():
    """The decisive check: a vector-Maxwell list and a scalar-Helmholtz list of the SAME structure (period,
    pillar ε, wavelength) feed the same ε to the same fmmax solve, so their transmission must agree."""
    Ts = float(jno.rcwa(_scalar_constraints(), orders=100, grid=32).solve().efficiency("T"))
    Tv = float(jno.rcwa(_vector_constraints(), orders=100, grid=32).solve().efficiency("T"))
    assert Tv == pytest.approx(Ts, abs=1e-6), f"vector T {Tv} != scalar T {Ts}"
