"""Face operators: ``jno.np.sqrtm(L)`` of a strong-form operator on a flat periodic face.

The exact non-reflecting boundary of a periodic half-space (Givoli, J. Comput. Phys. 94, 1991, sec. 3):
for the Helmholtz equation ``du/dn = i (k^2 + d_xx)^(1/2) u`` on the face, mode by mode
``du/dn = i k_z u`` with ``k_z = sqrt(k^2 - k_x^2)`` (principal branch: an outgoing wave ``exp(+i k_z y)``,
an evanescent one decays). The oracles are the exact outgoing waves; the first-order face
``du/dn = i k u`` reflects ``(k_z - k)/(k_z + k)`` of an oblique mode and serves as the contrast.

Vector Maxwell on the same face (n = +z): ``n x curl E = -i (k^2 - curl_t curl_t) (k^2 + Delta_t)^(-1/2) E_t``,
the TE order's admittance ``k_z`` and the TM order's ``k^2 / k_z`` in one operator. Its weak form, after
integrating the surface curl by parts on the periodic face, is the term written in the tests.
"""

import jax  # noqa: E402
import numpy as np
import pytest

import jno  # noqa: E402
from jno.trace.face_operator import FaceOperator  # noqa: E402

K = 2 * np.pi  # wavelength 1


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ---- the symbol ------------------------------------------------------------------------------------


def test_symbol_of_a_tangential_laplacian():
    """``k^2 w + w.x.x + w.y.y`` has symbol ``k^2 - kx^2 - ky^2``; ``**`` multiplies the power."""
    S = jno.np.sqrtm(lambda w: K**2 * w + w.x.x + w.y.y)
    assert isinstance(S, FaceOperator) and S.power == 0.5
    kx, ky = np.array([0.0, 3.0, 9.0]), np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(S.symbol({"x": kx, "y": ky}), K**2 - kx**2 - ky**2)
    m = (S**-1).multiplier({"x": kx, "y": ky})
    sig = K**2 - kx**2 - ky**2 + 0j
    np.testing.assert_allclose(m, sig**-0.5)
    assert m[2].imag < 0 and np.isclose(sig[2] ** 0.5, 1j * np.sqrt(-sig[2].real))  # evanescent: principal branch


def test_symbol_refuses_what_has_no_fourier_symbol():
    k = {"x": np.array([1.0])}
    with pytest.raises(TypeError, match="LINEAR"):
        jno.np.sqrtm(lambda w: w + 1.0).symbol(k)
    with pytest.raises(TypeError, match="linear"):
        jno.np.sqrtm(lambda w: w * w).symbol(k)
    with pytest.raises(ValueError, match="normal"):
        jno.np.sqrtm(lambda w: w.z.z).symbol(k)  # z is not a tangential axis of this face
    with pytest.raises(ValueError, match="vanishes"):
        (jno.np.sqrtm(lambda w: 1.0 * w + w.x.x) ** -1).multiplier(k)  # 1 - 1^2 = 0: grazing order


# ---- 2-D Helmholtz, periodic in x, the exact boundary on top ---------------------------------------


def _helmholtz(P, L, h, kx, top):
    """``-Δu - k^2 u = 0`` in [0, P] x [0, L], periodic in x, ``u = cos(kx x)`` at y = 0, and the top
    face's term ``top(u_t, v_t, (xt, yt))``. Returns (fem points, solution)."""
    d = jno.shape.rect(0.0, 0.0, P, L, size=h).domain()
    e = 1e-6
    d.tag("bottom", lambda x, y: y < e)
    d.tag("top", lambda x, y: y > L - e)
    d.tag("left", lambda x, y: (x < e) & (y > e))
    d.tag("right", lambda x, y: (x > P - e) & (y > e))
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    ut, vt = u.bind(x=xt, y=yt), v.bind(x=xt, y=yt)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y - K**2 * ui * vi,
            top(ut, vt),
            u(xb, yb) - jno.np.cos(kx * xb),
            u(xl, yl) - u(xr, yr),
        ]
    )
    return np.asarray(fem.points), np.asarray(fem.solve()).reshape(-1)


def _rel_err(pts, sol, kx, kz):
    exact = np.cos(kx * pts[:, 0]) * np.exp(1j * kz * pts[:, 1])
    return float(np.linalg.norm(sol - exact) / np.linalg.norm(exact))


@pytest.mark.parametrize("P, L", [(1.5, 1.0), (0.8, 0.4)])  # an oblique propagating mode; an evanescent one
def test_exact_face_does_not_reflect(P, L):
    """The exact boundary transmits the mode: the FE error is the discretisation error alone and falls
    as h^2 (P1, L2); the first-order face, on the same meshes, reflects and its error does not fall."""
    kx = 2 * np.pi / P
    kz = np.sqrt(K**2 - kx**2 + 0j)
    S = jno.np.sqrtm(lambda w: K**2 * w + w.x.x)
    err, err1 = [], []
    for h in (0.08, 0.04):
        err.append(_rel_err(*_helmholtz(P, L, h, kx, lambda ut, vt: -1j * S(ut) * vt), kx, kz))  # du/dn = i S u
        err1.append(_rel_err(*_helmholtz(P, L, h, kx, lambda ut, vt: -1j * K * ut * vt), kx, kz))  # du/dn = i k u
    assert err[0] / err[1] > 3.0, f"exact face: errors {err} do not fall as h^2"
    assert err[1] < 0.2 * err1[1], f"first-order face should reflect: {err1} vs exact {err}"


def test_face_operator_refuses_a_real_form():
    """``S(u) * v`` alone is a complex block on a real form: write the ``1j`` that makes the form complex."""
    S = jno.np.sqrtm(lambda w: K**2 * w + w.x.x)
    with pytest.raises(NotImplementedError, match="COMPLEX"):
        _helmholtz(1.5, 0.5, 0.1, 2 * np.pi / 1.5, lambda ut, vt: S(ut) * vt)


def test_face_operator_refuses_the_test_function():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    d.tag("top", lambda x, y: y > 1 - 1e-6)
    u, v = d.fem_symbols()
    xt, yt, _ = d.variable("top", split=True)
    with pytest.raises(TypeError, match="UNKNOWN"):
        jno.np.sqrtm(lambda w: w.x.x)(v.bind(x=xt, y=yt))


# ---- 3-D Maxwell (N1E degree 2), periodic in x and y, the exact boundary on top ---------------------


def _maxwell(pol, exact_face, P=1.5, L=1.0, h=0.25, phase=None):
    """``curl curl E - k^2 E = 0`` in a vacuum cell, periodic in x and y, the tangential trace of one
    oblique order prescribed at z = 0 (TE: E along y; TM: E in the x-z plane). Returns the relative
    L2 error against that order's outgoing wave, sampled at one interior point per cell."""
    import jax.numpy as jnp

    from jno.utils.solver.fem_dofmap import cell_jacobians
    from jno.utils.solver.fem_nonnodal import nonnodal_field_at

    inner, vec, cross = jno.np.inner, jno.np.vector, jno.np.cross
    kx = 2 * np.pi / P
    kz = np.sqrt(K**2 - kx**2)
    d = jno.shape.box(0, 0, 0, P, P, L, size=h).domain()
    e = 1e-6
    d.tag("bottom", lambda x, y, z: z < e)
    d.tag("top", lambda x, y, z: z > L - e)
    d.tag("left", lambda x, y, z: x < e)
    d.tag("right", lambda x, y, z: x > P - e)
    d.tag("front", lambda x, y, z: y < e)
    d.tag("back", lambda x, y, z: y > P - e)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E", order=2)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    xt, yt, zt = d.variable("top", split=True)[:3]
    xb = d.variable("bottom", split=True)[0]
    nt, nb = d.variable("top", normals=True), d.variable("bottom", normals=True)

    def face(name):
        return u.bind(**dict(zip("xyz", d.variable(name, split=True)[:3])))

    c = jno.np.cos(kx * xb)
    Eb = vec(0 * c, c, 0 * c) if pol == "TE" else vec(kz * c, 0 * c, 0 * c)  # E_t at z = 0
    if exact_face:
        curl_t = lambda w: w.x[1] - w.y[0]  # noqa: E731  surface curl on z = const
        S = jno.np.sqrtm(lambda w: K**2 * w + w.x.x + w.y.y) ** -1  # (k^2 + Delta_t)^(-1/2)
        tE, tF = cross(nt, u.vector.cross(nt)), cross(nt, v.vector.cross(nt))  # E_t = n x (E x n)
        top = -1j * (K**2 * inner(S(tE), tF) - curl_t(S(tE)) * v.vector.curl(xt, yt, zt)[2])
    else:
        top = -1j * K * inner(u.vector.cross(nt), v.vector.cross(nt))  # n x curl E = -i k E_t
    ties = [face("left") - (face("right") if phase is None else phase * face("right")), face("front") - face("back")]
    fem = jno.fem(
        [
            inner(u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)) - K**2 * inner(ui, vi),
            top,
            u.vector.cross(nb) - cross(Eb, vec(0 * c, 0 * c, -1.0 + 0 * c)),
            *ties,
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1)
    topo = d._fem_nonnodal_topology
    dm, cells = topo["dofmap"], topo["cells"]
    pts = np.asarray(d.mesh.points)[:, :3]
    rp = np.full((1, 3), 0.25)
    Eh = np.asarray(nonnodal_field_at(pts, cells, dm, jnp.asarray(sol[: dm.n_dofs]), ref_points=rp))[:, 0]
    J, x0 = cell_jacobians(pts, cells)
    X = x0 + np.einsum("a,cda->cd", rp[0], J)
    ph, cx, sx = np.exp(1j * kz * X[:, 2]), np.cos(kx * X[:, 0]), np.sin(kx * X[:, 0])
    z0 = 0 * cx
    ex = np.stack([z0, cx, z0] if pol == "TE" else [kz * cx, z0, -1j * kx * sx], 1) * ph[:, None]
    return float(np.linalg.norm(Eh - ex) / np.linalg.norm(ex))


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_maxwell_exact_face_transmits_te_and_tm(pol):
    """One operator carries both admittances: the oblique TE and TM orders leave the cell with the FE
    error alone (measured 3.0 % / 3.2 % at h = 0.25, 1.5 % / 1.7 % at h = 0.17: second order), while
    the first-order face ``n x curl E = -i k E_t`` reflects (19 % / 23 %, not falling with h)."""
    err, err1 = _maxwell(pol, True), _maxwell(pol, False)
    assert err < 0.05, f"{pol}: exact face rel err {err:.3e}"
    assert err1 > 4 * err, f"{pol}: first-order {err1:.3e} vs exact {err:.3e}"


# ---- Bloch (oblique incidence), transient and nonlinear forms --------------------------------------


def _cell(P, L, h, **kw):
    """[0, P] x [0, L], tagged; returns (domain, u, v, {name: (x, y[, t])})."""
    d = jno.shape.rect(0.0, 0.0, P, L, size=h).domain(**kw)
    e = 1e-6
    d.tag("bottom", lambda x, y: y < e)
    d.tag("top", lambda x, y: y > L - e)
    d.tag("left", lambda x, y: (x < e) & (y > e))
    d.tag("right", lambda x, y: (x > P - e) & (y > e))
    u, v = d.fem_symbols()
    X = {n: d.variable(n, split=True) for n in ("interior", "bottom", "top", "left", "right")}
    return d, u, v, X


def _value(out):
    """``fem.solve()`` of a nonlinear or transient form is a lazy trace node; evaluate it."""
    return np.asarray(out.fn() if hasattr(out, "fn") else out)


def _bloch(h, exact_face):
    """A plane wave ``exp(i (kx x + kz y))`` at 30 degrees in a cell of period 1 (a Bloch tie
    ``u(left) = exp(-i kx P) u(right)``), driven by its normal flux at y = 0, leaving through the top."""
    P, L = 1.0, 1.0
    kx = 0.5 * K
    kz = np.sqrt(K**2 - kx**2)
    d, u, v, X = _cell(P, L, h)
    (xi, yi, _), (xb, yb, _), (xt, yt, _) = X["interior"], X["bottom"], X["top"]
    ui, vi, ut, vt = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), u.bind(x=xt, y=yt), v.bind(x=xt, y=yt)
    S = jno.np.sqrtm(lambda w: K**2 * w + w.x.x)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y - K**2 * ui * vi,
            -1j * S(ut) * vt if exact_face else -1j * K * ut * vt,
            1j * kz * jno.np.exp(1j * kx * xb) * v.bind(x=xb, y=yb),  # -g v, g = du/dn = -i kz e^(i kx x)
            u(*X["left"][:2]) - np.exp(-1j * kx * P) * u(*X["right"][:2]),
        ]
    )
    p, s = np.asarray(fem.points), _value(fem.solve()).reshape(-1)
    ex = np.exp(1j * (kx * p[:, 0] + kz * p[:, 1]))
    return float(np.linalg.norm(s - ex) / np.linalg.norm(ex))


def test_bloch_cell_exact_face_transmits_an_oblique_wave():
    """With a Bloch tie the face modes are ``exp(i (k_m + k_B) x)``: the exact face transmits the oblique
    wave with the FE error alone (8.3e-3, 2.1e-3 at h = 0.04, 0.02 measured: h^2); the first-order face
    reflects (9.7 %)."""
    err = [_bloch(h, True) for h in (0.08, 0.04)]
    err1 = _bloch(0.04, False)
    assert err[0] / err[1] > 3.0, f"Bloch exact face: {err} do not fall as h^2"
    assert err[1] < 0.2 * err1, f"first-order {err1:.3e} vs exact {err}"


def test_transient_diffusion_with_the_exterior_laplace_face():
    """``u_t = Δu`` in a periodic cell under a half-space of a quasistatic (Laplace) exterior: on the top
    face ``du/dn = -(-d_xx)^(1/2) u``, the exact exterior. The decaying mode
    ``cos(kx x) sin(mu y) exp(-(mu^2 + kx^2) t)``, ``mu cos(mu L) = -kx sin(mu L)``, is recovered (0.9 %,
    backward Euler's first-order time error: 6.2e-3 -> 3.5e-3 at half the step); an insulated top gives
    18 %. The face block is real: the symbol ``kx^2`` has a real square root."""
    from scipy.optimize import brentq

    P, L, T = 1.0, 1.0, 0.02
    kx = 2 * np.pi / P
    mu = brentq(lambda m: m * np.cos(m * L) + kx * np.sin(m * L), np.pi / 2 / L + 1e-9, np.pi / L - 1e-9)

    def run(exact_face):
        d, u, v, X = _cell(P, L, 0.08, time=(0.0, T, 81))
        xi, yi, ti = X["interior"]
        xt, yt, _ = X["top"]
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
        S = jno.np.sqrtm(lambda w: -w.x.x)
        terms = [
            ui.t * vi + ui.x * vi.x + ui.y * vi.y,
            u(*X["bottom"][:2]) - 0.0,
            u(*X["left"][:2]) - u(*X["right"][:2]),
            u(ci[0], ci[1]) - jno.np.cos(kx * ci[0]) * jno.np.sin(mu * ci[1]),
        ]
        if exact_face:
            terms.append(S(u.bind(x=xt, y=yt)) * v.bind(x=xt, y=yt))  # du/dn = -S u
        fem = jno.fem(terms)
        assert fem.is_transient
        p, s = np.asarray(fem.points), _value(fem.solve())[-1].reshape(-1)
        ex = np.cos(kx * p[:, 0]) * np.sin(mu * p[:, 1]) * np.exp(-(mu**2 + kx**2) * float(fem.t1))
        return float(np.linalg.norm(s - ex) / np.linalg.norm(ex))

    err, err0 = run(True), run(False)
    assert err < 0.02, f"exact exterior face: {err:.3e}"
    assert err0 > 8 * err, f"insulated top {err0:.3e} vs exact {err:.3e}"


def _cubic(h, exact_face, direct=False):
    """``-Δu + u^3 = u*^3`` with the harmonic ``u* = cos(kx x) exp(-kx y)``, ``u = u*`` at y = 0, and the
    exterior-Laplace face on top (where ``du/dn = -kx u*`` exactly)."""
    P, L = 1.0, 0.3
    kx = 2 * np.pi / P
    d, u, v, X = _cell(P, L, h)
    (xi, yi, _), (xb, yb, _), (xt, yt, _) = X["interior"], X["bottom"], X["top"]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    us = jno.np.cos(kx * xi) * jno.np.exp(-kx * yi)
    S = jno.np.sqrtm(lambda w: -w.x.x)
    terms = [
        ui.x * vi.x + ui.y * vi.y + ui**3 * vi - us**3 * vi,
        u(xb, yb) - jno.np.cos(kx * xb),
        u(*X["left"][:2]) - u(*X["right"][:2]),
    ]
    if exact_face:
        terms.append(S(u.bind(x=xt, y=yt)) * v.bind(x=xt, y=yt))
    fem = jno.fem(terms)
    assert not fem.is_linear
    out = fem.solve(nonlinear=jno.solve.newton(direct=True)) if direct else fem.solve()
    p, s = np.asarray(fem.points), _value(out).reshape(-1)
    ex = np.cos(kx * p[:, 0]) * np.exp(-kx * p[:, 1])
    return float(np.linalg.norm(s - ex) / np.linalg.norm(ex)), s


def test_nonlinear_form_carries_the_face_block_in_residual_and_tangent():
    """Newton sees the face block in its residual (matrix-free JFNK) and in the assembled tangent
    (``newton(direct=True)``): both reach the same solution, which converges as h^2 (9.8e-4, 2.4e-4 at
    h = 0.04, 0.02 measured); an insulated top gives 14 %."""
    (e1, s1), (e2, _s2) = _cubic(0.08, True), _cubic(0.04, True)
    e_direct, s_direct = _cubic(0.08, True, direct=True)
    e0, _ = _cubic(0.08, False)
    assert e1 / e2 > 3.0, f"nonlinear exact face: {e1:.3e}, {e2:.3e} do not fall as h^2"
    np.testing.assert_allclose(s_direct, s1, rtol=1e-5, atol=1e-8)  # both to the Newton tolerance
    assert e0 > 8 * e1, f"insulated top {e0:.3e} vs exact {e1:.3e}"


def test_face_block_couples_only_the_face_dofs():
    """The dense block lives on the face's trace DOFs. An N1E face triangle's owner tet lends all six of
    its edges, but only the three on the face have a tangential trace there; the other three must not
    enter the block (they made it several times larger than the face and widened the factorisation front)."""
    inner, cross = jno.np.inner, jno.np.cross
    P, L = 1.1, 0.5  # P = 1 would put an order at grazing incidence (refused)
    d = jno.shape.box(0, 0, 0, P, P, L, size=0.34).domain()
    e = 1e-6
    for name, f in {
        "top": lambda x, y, z: z > L - e,
        "left": lambda x, y, z: x < e,
        "right": lambda x, y, z: x > P - e,
        "front": lambda x, y, z: y < e,
        "back": lambda x, y, z: y > P - e,
    }.items():
        d.tag(name, f)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    xi, yi, zi = d.variable("interior", split=True)[:3]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    nt = d.variable("top", normals=True)
    face = lambda name: u.bind(**dict(zip("xyz", d.variable(name, split=True)[:3])))  # noqa: E731
    S = jno.np.sqrtm(lambda w: K**2 * w + w.x.x + w.y.y) ** -1
    tE, tF = cross(nt, u.vector.cross(nt)), cross(nt, v.vector.cross(nt))
    vol = inner(u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)) - K**2 * inner(ui, vi)
    ties = [face("left") - face("right"), face("front") - face("back")]

    def stored(top):  # triplets the imaginary leg stores (explicit zeros included: they still cost)
        (_, _), (Ai, _) = jno.fem([vol, top, *ties])._complex_legs
        return int(Ai.nse)

    pts, tets = np.asarray(d.mesh.points), np.asarray(d.mesh.cells_dict["tetra"])
    edges = np.unique(np.sort(tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2), 1), axis=0)
    n_face = int(np.all(np.abs(pts[edges, 2] - L) < 1e-9, axis=1).sum())  # the face's edges = its trace DOFs
    exact = stored(-1j * K**2 * inner(S(tE), tF))
    local = stored(-1j * K * inner(u.vector.cross(nt), v.vector.cross(nt)))  # the first-order face is local
    assert 0 < exact - local <= n_face**2
