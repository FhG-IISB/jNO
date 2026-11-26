"""N1E trial-free surface term — the incident source of an absorbing face.

The weak boundary term ``inner(g, v×n)`` is the surface integral ``∫_Γ g·(v×n) dS``. Like every other
trial-free term it enters the load as ``b = -term`` (the term list sums to zero). The physical incident
source pairs the tangential traces, ``inner(cross(E_inc, n), v×n) = ∫ E_inc,t·v_t``; with the absorbing
face ``i k₀ inner(u×n, v×n)`` (``exp(+i w t)``) a plane wave ``E_inc`` enters through
``i k₀ inner(u×n, v×n) - 2 i k₀ inner(cross(E_inc, n), v×n)``.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

inner = jno.np.inner
vec = jno.np.vector
cross = jno.np.cross
_dense = lambda A: np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))  # noqa: E731
_arr = lambda x: np.asarray(jnp.asarray(x)).reshape(-1)  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _cube_with_top(mesh_size=0.4):
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain()
    d.tag("top", lambda x, y, z: z > 1.0 - 1e-6)  # a single boundary face (z = 1)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    ntop = d.variable("top", normals=True)
    ct = d.variable("top", split=True)  # top-face coords, so a source g lives on the `top` region
    tv_top = v.vector.cross(ntop)  # v×n on the top face
    return d, (xi, yi, zi), (ct[0], ct[1], ct[2]), (ui, vi), tv_top


def test_incident_load_matches_analytic_single_face():
    """Decisive geometry/sign check on one face. On the top face (n=(0,0,1)) with source g=(0,1,0) and the
    constant field u*=(1,0,0) (exact in N1E0): ∫_top g·(u*×n) = (0,1,0)·((1,0,0)×(0,0,1)) = (0,1,0)·(0,−1,0)
    = −1 over the unit-area face. The load is b = −term, the sign of every trial-free term (the volume
    projection below reads inner(u,v) − inner(u*,v) the same way), so u*·b must be +1."""
    d, (xi, yi, zi), (xt, yt, zt), (ui, vi), tv_top = _cube_with_top(0.4)
    g = vec(0.0 * xt, 1.0 + 0.0 * yt, 0.0 * zt)  # constant (0,1,0) on the top face
    u_star = vec(1.0 + 0.0 * xi, 0.0 * yi, 0.0 * zi)  # constant (1,0,0), exact in N1E0

    M = _dense(jno.fem([inner(ui, vi)]).A)
    u_dof = np.linalg.solve(M, _arr(jno.fem([inner(ui, vi) - inner(u_star, vi)]).b))  # exact projection
    b = _arr(jno.fem([inner(ui, vi), inner(g, tv_top)]).b)  # volume mass registers the field; load into b

    np.testing.assert_allclose(float(u_dof @ b), 1.0, atol=1e-9)


def test_incident_load_is_linear_in_the_source():
    """The load is linear in g: doubling the source doubles the load vector."""
    d, (xi, yi, zi), (xt, yt, zt), (ui, vi), tv_top = _cube_with_top(0.5)
    g = vec(0.3 + 0.0 * xt, 1.0 + 0.0 * yt, -0.5 + 0.0 * zt)
    b1 = _arr(jno.fem([inner(ui, vi), inner(g, tv_top)]).b)
    b2 = _arr(jno.fem([inner(ui, vi), inner(2.0 * g, tv_top)]).b)
    np.testing.assert_allclose(b2, 2.0 * b1, atol=1e-10)


def test_complex_incident_lands_in_the_imag_leg():
    """A physical ``i·k₀`` incident source is purely imaginary, so the forcing lands in the Im leg's load
    (b_i) and the Re leg's load (b_r) stays zero — composing with the complex N1E path."""
    d, (xi, yi, zi), (xt, yt, zt), (ui, vi), tv_top = _cube_with_top(0.5)
    k0 = 2.0
    g = vec(0.0 * xt, 1.0 + 0.0 * yt, 0.0 * zt)

    fem = jno.fem([inner(ui, vi) + 0j * inner(ui, vi), 2j * k0 * inner(g, tv_top)])
    assert fem.is_complex
    (_Ar, b_r), (_Ai, b_i) = fem._complex_legs  # unfused legs (``_op`` is the fused 2n block)
    b_ref = _arr(jno.fem([inner(ui, vi), inner(g, tv_top)]).b)  # real reference load
    np.testing.assert_allclose(_arr(b_r), 0.0, atol=1e-10)  # Re leg: no source
    np.testing.assert_allclose(_arr(b_i), 2.0 * k0 * b_ref, atol=1e-8)  # Im leg: 2 k0 · load


def test_combined_absorbing_plus_incident_on_one_face():
    """A single boundary entry may combine the absorbing mass and the incident load —
    ``i k₀·inner(n×u,n×v) + 2 i k₀·inner(g,n×v)`` — and the additive split routes the bilinear part into A
    and the trial-free part into b (equal to authoring them as two separate entries)."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    d.tag("top", lambda x, y, z: z > 1.0 - 1e-6)
    k0 = 2.0
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    nt = d.variable("top", normals=True)
    ct = d.variable("top", split=True)
    tu, tv = u.vector.cross(nt), v.vector.cross(nt)
    g = vec(1.0 + 0.0 * ct[0], 0.0 * ct[1], 0.0 * ct[2])

    combined = jno.fem([inner(ui, vi), 1j * k0 * inner(tu, tv) + 2j * k0 * inner(g, tv)])
    separate = jno.fem([inner(ui, vi), 1j * k0 * inner(tu, tv), 2j * k0 * inner(g, tv)])
    (Ar_c, br_c), (Ai_c, bi_c) = combined._complex_legs  # unfused legs (``_op`` is the fused 2n block)
    (Ar_s, br_s), (Ai_s, bi_s) = separate._complex_legs
    np.testing.assert_allclose(_dense(Ai_c), _dense(Ai_s), atol=1e-9)  # same surface mass into A_i
    np.testing.assert_allclose(_arr(bi_c), _arr(bi_s), atol=1e-9)  # same incident load into b_i


def test_driven_maxwell_end_to_end_solves():
    """Integration of commits 1–3: a FULL driven time-harmonic Maxwell problem — curl-curl − k₀²·mass
    volume, the i·k₀ impedance absorbing BC on the whole boundary, and a plane-wave incident source on the
    top face — assembles as a complex system and SOLVES to a finite, non-trivial complex field. (The
    impedance term provides radiation damping, so the otherwise-indefinite Helmholtz operator is well posed.)"""
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.4).domain()
    d.tag("top", lambda x, y, z: z > 1.0 - 1e-6)
    k0 = 3.0
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    ccu, ccv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    nb = d.variable("boundary", normals=True)
    nt = d.variable("top", normals=True)
    ct = d.variable("top", split=True)
    tu, tv = u.vector.cross(nb), v.vector.cross(nb)  # impedance on the whole boundary
    tv_top = v.vector.cross(nt)
    g = vec(1.0 + 0.0 * ct[0], 0.0 * ct[1], 0.0 * ct[2])  # x-polarised incident source on top

    fem = jno.fem(
        [
            inner(ccu, ccv) - k0**2 * inner(ui, vi),  # curl-curl − k0² mass
            1j * k0 * inner(tu, tv),  # impedance absorbing BC
            -2j * k0 * inner(cross(g, nt), tv_top),  # incident plane wave
        ]
    )
    assert fem.is_complex
    E = _arr(fem.solve())
    assert np.iscomplexobj(E)
    assert np.all(np.isfinite(E))
    assert np.max(np.abs(E)) > 1e-6  # a genuine, non-trivial scattered field
    assert np.max(np.abs(E.imag)) > 1e-9  # genuinely complex (radiation phase), not a real solve


def test_incident_load_follows_the_cross_order_and_reads_the_normal():
    """``inner(g, n×v) = -inner(g, v×n)``: the order of the cross is part of the term, not discarded. A
    source that reads the face normal, ``cross(g, n)``, gets the normal itself (not the face coordinates)
    and matches the same source with the constant normal written out."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    d.tag("top", lambda x, y, z: z > 1.0 - 1e-6)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c, ct = d.variable("interior", split=True), d.variable("top", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
    nt = d.variable("top", normals=True)
    g = vec(0.3 + 0.0 * ct[0], 1.0 + 0.0 * ct[1], -0.5 + 0.0 * ct[2])
    ez = vec(0.0 * ct[0], 0.0 * ct[1], 1.0 + 0.0 * ct[2])  # the top face's outward normal, written out
    tv = v.vector.cross(nt)
    b = _arr(jno.fem([inner(ui, vi), inner(g, tv)]).b)
    assert np.max(np.abs(b)) > 1e-3
    np.testing.assert_allclose(_arr(jno.fem([inner(ui, vi), inner(g, cross(nt, v))]).b), -b, atol=1e-12)
    b_n = _arr(jno.fem([inner(ui, vi), inner(cross(g, nt), tv)]).b)
    b_c = _arr(jno.fem([inner(ui, vi), inner(cross(g, ez), tv)]).b)
    assert np.max(np.abs(b_c)) > 1e-3
    np.testing.assert_allclose(b_n, b_c, atol=1e-12)


def test_surface_mass_follows_the_cross_order():
    """``inner(n×u, v×n) = -inner(u×n, v×n)``; ``inner(n×u, n×v) = inner(u×n, v×n)``."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    d.tag("top", lambda x, y, z: z > 1.0 - 1e-6)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
    nt = d.variable("top", normals=True)
    M = _dense(jno.fem([inner(ui, vi)]).A)
    S = _dense(jno.fem([inner(ui, vi), inner(u.vector.cross(nt), v.vector.cross(nt))]).A) - M
    assert np.max(np.abs(S)) > 1e-3
    S_flip = _dense(jno.fem([inner(ui, vi), inner(cross(nt, u), v.vector.cross(nt))]).A) - M
    S_both = _dense(jno.fem([inner(ui, vi), inner(cross(nt, u), cross(nt, v))]).A) - M
    np.testing.assert_allclose(S_flip, -S, atol=1e-12)
    np.testing.assert_allclose(S_both, S, atol=1e-12)


def test_incident_plane_wave_keeps_its_polarization():
    """End-to-end oracle: a homogeneous periodic slab with absorbing top and bottom faces, an x-polarised
    plane wave entering through the bottom. The field is that plane wave, ``x exp(-i k (z - z_b))``
    (``exp(+i w t)``), polarization included -- the first-order absorbing face is exact at normal incidence."""
    from jno.utils.solver.fem_dofmap import cell_jacobians
    from jno.utils.solver.fem_nonnodal import nonnodal_field_at

    P, Z, k0 = 0.5, 1.0, 2 * np.pi
    d = jno.shape.box(0, 0, 0, P, P, Z, size=0.25).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E", order=2)
    c = d.variable("interior", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
    nt, nb = d.variable("top", normals=True), d.variable("bottom", normals=True)
    cb = d.variable("bottom", split=True)
    einc = vec(1.0 + 0.0 * cb[0], 0.0 * cb[1], 0.0 * cb[2])

    def face(n):
        cc = d.variable(n, split=True)
        return u.bind(x=cc[0], y=cc[1], z=cc[2])

    tvb = v.vector.cross(nb)
    fem = jno.fem(
        [
            inner(u.vector.curl(*c[:3]), v.vector.curl(*c[:3])) - k0**2 * inner(ui, vi),
            1j * k0 * inner(u.vector.cross(nt), v.vector.cross(nt)),
            1j * k0 * inner(u.vector.cross(nb), tvb) - 2j * k0 * inner(cross(einc, nb), tvb),
            face("left") - face("right"),
            face("front") - face("back"),
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1)
    topo = d._fem_nonnodal_topology
    pts = np.asarray(d.mesh.points)[:, :3]
    rp = np.full((1, 3), 0.25)
    Eh = np.asarray(
        nonnodal_field_at(pts, topo["cells"], topo["dofmap"], jnp.asarray(sol[: topo["dofmap"].n_dofs]), ref_points=rp)
    )[:, 0]
    J, x0 = cell_jacobians(pts, topo["cells"])
    z = (x0 + np.einsum("a,cda->cd", rp[0], J))[:, 2]
    ex = np.stack([np.exp(-1j * k0 * z), 0 * z, 0 * z], 1)
    assert np.linalg.norm(Eh - ex) / np.linalg.norm(ex) < 0.05


def test_tangential_dirichlet_value_reads_the_face_normal():
    """An essential trace ``u×n = cross(g, n)`` with the face's normal Variable inside the value gets the
    facet normal at each point (it used to read the face coordinates and pin a wrong trace)."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    d.tag("top", lambda x, y, z: z > 1.0 - 1e-6)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c, ct = d.variable("interior", split=True), d.variable("top", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
    cu, cv = u.vector.curl(c[0], c[1], c[2]), v.vector.curl(c[0], c[1], c[2])
    nt = d.variable("top", normals=True)
    g = vec(1.0 + 0.0 * ct[0], 0.5 + 0.0 * ct[1], 0.0 * ct[2])
    ez = vec(0.0 * ct[0], 0.0 * ct[1], 1.0 + 0.0 * ct[2])
    vol = inner(cu, cv) + inner(ui, vi)
    with_n = _arr(jno.fem([vol, u.vector.cross(nt) - cross(g, nt)]).solve(linear=jno.solve.lu()))
    with_c = _arr(jno.fem([vol, u.vector.cross(nt) - cross(g, ez)]).solve(linear=jno.solve.lu()))
    assert np.max(np.abs(with_c)) > 1e-2
    np.testing.assert_allclose(with_n, with_c, atol=1e-10)
