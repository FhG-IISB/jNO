"""N1E tangential-trace surface terms (absorbing face, incident load) inside a NONLINEAR non-nodal form.

The surface mass ``c ∫ (n×u)·(n×v)`` is linear in ``u``: in a Newton solve it adds ``S u`` to the residual
and ``S`` to the tangent, and the incident load is a constant in the residual. The nonlinear path used to
refuse the combination. The case pinned here is two-way Maxwell + heat with ``E = E_r + i E_i`` as two real
N1E fields and ``T`` a P1 field, the Joule heat ``1/2 k0 Im(eps) |E|^2`` and a thermo-optic ``eps(T)``.

Oracles:
* with no thermo-optic feedback (``beta = 0``) the field part is linear, so ``E_r + i E_i`` must equal the
  COMPLEX linear solve of the same Maxwell form, which takes the established linear impedance path;
* the assembled tangent equals ``jax.jacfwd`` of the residual, surface block included, at a nonzero state.

The real pair also pins a LINEAR bug: a face term coupling two N1E fields (trial ``E_i``, test ``v_r``) used
to be assembled into the first N1E field's diagonal block, a silent wrong answer.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D box meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

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


def _two_way(d, beta):
    xi, yi, zi = d.variable("interior", split=True)[:3]
    X = [xi, yi, zi]
    er_, vr_ = d.fem_symbols(value_shape=(3,), names=("Er", "vr"), space="N1E")
    ei_, vi_ = d.fem_symbols(value_shape=(3,), names=("Ei", "vi"), space="N1E")
    T, s = d.fem_symbols(names=("T", "s"), space="Lagrange")
    Er, vr, Ei, vi = (w.bind(x=xi, y=yi, z=zi) for w in (er_, vr_, ei_, vi_))
    Tb, sb = T.bind(x=xi, y=yi, z=zi), s.bind(x=xi, y=yi, z=zi)
    curl = lambda w: w.vector.curl(xi, yi, zi)  # noqa: E731
    eps_r = d.er + beta * d.ei * Tb
    nt = d.variable("top", normals=True)
    terms = [
        # -i k0 (n×E)·(n×v) on top, split: Re gets +k0 (E_i), Im gets -k0 (E_r)
        inner(curl(er_), curl(vr_)) - K0**2 * (eps_r * inner(Er, vr) - d.ei * inner(Ei, vr)),
        inner(curl(ei_), curl(vi_)) - K0**2 * (d.ei * inner(Er, vi) + eps_r * inner(Ei, vi)),
        K0 * inner(ei_.vector.cross(nt), vr_.vector.cross(nt)),
        -K0 * inner(er_.vector.cross(nt), vi_.vector.cross(nt)),
        -_incident(d, vr_),
        d.k * inner(jno.np.grad(Tb, X), jno.np.grad(sb, X), n_contract=1)
        - 0.5 * K0 * d.ei * (inner(Er, Er) + inner(Ei, Ei)) * sb,
        T(*d.variable("bottom", split=True)[:3]) - 0.0,
    ]
    return jno.fem(terms), (er_, ei_, T)


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


def test_two_way_maxwell_heat_with_an_absorbing_face_matches_the_complex_linear_solve():
    d = _domain()
    fem, (er_, ei_, T_) = _two_way(d, beta=0.0)
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True))).reshape(-1)
    ref = np.asarray(_complex_linear(d).solve(linear=jno.solve.lu())).reshape(-1)
    blk = fem.blocks
    E = sol[blk[fem.block_index(er_)]] + 1j * sol[blk[fem.block_index(ei_)]]
    assert np.abs(ref).max() > 1e-3, "the incident load did not drive the reference field"
    np.testing.assert_allclose(E, ref, rtol=0, atol=1e-8 * np.abs(ref).max())
    T = sol[blk[fem.block_index(T_)]]
    assert T.max() > 0, "the Joule heat did not reach the temperature"


def test_the_newton_tangent_carries_the_surface_block():
    d = _domain()
    op = _two_way(d, beta=1e-2)[0].operator
    u0 = jax.random.normal(jax.random.PRNGKey(0), (op.size,), dtype=jnp.float64) * 0.1
    J = np.asarray(op.jacobian(u0).todense())
    J_ad = np.asarray(jax.jacfwd(op.residual)(u0))
    np.testing.assert_allclose(J, J_ad, rtol=0, atol=1e-9 * np.abs(J_ad).max())
