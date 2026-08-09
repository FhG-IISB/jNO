"""Bloch / quasi-periodic ties: ``u(A) = e^{i k·L} u(B)``. The phase makes the prolongation ``P``
complex; on the fused ``[Re; Im]`` block state the same tie is the REAL prolongation
``B(P) = [[P_r, -P_i], [P_i, P_r]]``, and the ordinary real congruence ``B(P)ᵀ A B(P)`` equals the
Hermitian reduction ``P^H A_c P`` the Bloch space requires. A Bloch problem therefore solves through
the same fused path as every other complex problem — ``solve_fn=``, the solver slots and ``x0=`` all
apply (each used to be silently discarded by a dedicated Bloch block solve).

Plain periodic (phase == 1) must stay a real 0/1 selection (regression guard)."""

import jax
import numpy as np
import pytest

import jno

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _bloch_strip(kx, *, size=0.1, source="complex", extra_coeff=1.0):
    """``-Δu + u = f`` on the unit square: Bloch tie in x (phase ``e^{-i kx}``), Dirichlet 0 top/bottom.

    With ``f = (kx² + π² + 1)·e^{i kx x}·sin(π y)`` the exact solution is the Bloch mode
    ``u* = e^{i kx x}·sin(π y)`` (it satisfies the tie exactly: ``u*(0,y) = e^{-i kx}·u*(1,y)``).
    ``source="real"`` keeps every coefficient real (``f = Re(λ u*)``) — the *real-form* Bloch case."""
    d = jno.domain(jno.Shape.rect(0, 0, 1.0, 1.0, size=size))
    e = 1e-6
    d.tag("left", lambda x, y: (x < e) & (y > e) & (y < 1 - e))
    d.tag("right", lambda x, y: (x > 1 - e) & (y > e) & (y < 1 - e))
    d.tag("bottom", lambda x, y: y < e)
    d.tag("top", lambda x, y: y > 1 - e)
    u, phi = d.fem_symbols()
    xi, yi, *_ = d.variable("interior", split=True)
    xl, yl, *_ = d.variable("left", split=True)
    xr, yr, *_ = d.variable("right", split=True)
    xb, yb, *_ = d.variable("bottom", split=True)
    xt, yt, *_ = d.variable("top", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    lam = kx**2 + PI**2 + 1.0
    if source == "complex":
        f = lam * jno.np.exp(1j * kx * xi) * jno.np.sin(PI * yi)
    else:
        f = lam * jno.np.cos(kx * xi) * jno.np.sin(PI * yi)
    cons = [
        extra_coeff * (ui.x * vi.x + ui.y * vi.y) + u * vi - f * vi,
        u(xb, yb) - 0.0,
        u(xt, yt) - 0.0,
        u(xl, yl) - np.exp(-1j * kx) * u(xr, yr),
    ]
    return jno.fem(cons)


def test_bloch_prolongation_is_complex_plain_is_real():
    from jno.utils.solver.fem_utils import build_periodic_prolongation

    pts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    tags = {"left": np.array([0, 1]), "right": np.array([2, 3])}
    plain = build_periodic_prolongation(pts, [("left", "right")], tags)
    bloch = build_periodic_prolongation(pts, [("left", "right")], tags, phases=[np.exp(1j * 0.7)])
    assert plain["is_bloch"] is False and not np.iscomplexobj(np.asarray(plain["P"].data))
    assert bloch["is_bloch"] is True and np.iscomplexobj(np.asarray(bloch["P"].data))
    # slave rows carry the Bloch factor e^{i 0.7}
    assert np.allclose(np.abs(np.asarray(bloch["P"].data)), 1.0)


def test_bloch_steady_fuses_and_recovers_the_manufactured_mode():
    """A steady Bloch problem is an ordinary fused ``linear`` FEM: the real-equivalent 2n block plus
    the REAL prolongation ``B(P)``, recovering the exact Bloch mode ``e^{i kx x} sin(π y)``."""
    kx = PI / 2
    fem = _bloch_strip(kx)
    assert fem._mode == "linear", "a Bloch problem must fuse like every other complex problem"
    assert fem.is_complex and fem._complex_n is not None
    per2 = fem._periodic_2n
    assert per2 is not None and not per2.get("is_bloch"), "the 2n reduction must be the REAL B(P)"
    assert not np.iscomplexobj(np.asarray(per2["P"].data)), "B(P) carries real data only"
    assert per2["n_full"] == 2 * fem._complex_n

    sol = np.asarray(fem.solve())
    pts = np.asarray(fem.points)
    exact = np.exp(1j * kx * pts[:, 0]) * np.sin(PI * pts[:, 1])
    rel = np.linalg.norm(sol - exact) / np.linalg.norm(exact)
    assert rel < 5e-2, f"Bloch mode recovery rel-L2 {rel:.3e}"
    # the tie itself: u(left) = e^{-i kx} u(right), matched pointwise by y-order
    lidx, ridx = pts[:, 0] < 1e-9, pts[:, 0] > 1 - 1e-9
    ly, ry = np.argsort(pts[lidx, 1]), np.argsort(pts[ridx, 1])
    tie = np.abs(sol[lidx][ly] - np.exp(-1j * kx) * sol[ridx][ry]).max()
    assert tie < 1e-10, f"Bloch tie residual {tie:.2e}"


def test_bloch_real_form_promotes_to_the_hermitian_reduction():
    """A REAL weak form with a Bloch tie must give the same answer as its complex-classified twin.

    The real path used to reduce with the bilinear ``Pᵀ A P`` — not a Galerkin projection for a
    complex ``P`` — and was measured 8.1 rel-L2 off the Hermitian answer while satisfying the tie
    exactly, so nothing *looked* wrong. The promotion (a zero imaginary leg + the fused B(P)
    reduction) makes the two paths agree to machine precision."""
    kx = PI / 2
    fem_real = _bloch_strip(kx, source="real")  # every coefficient real
    fem_cx = _bloch_strip(kx, source="real", extra_coeff=np.exp(0j))  # complex-classified twin
    assert fem_real._mode == "linear" and fem_real.is_complex, "the real form must be promoted"
    a, b = np.asarray(fem_real.solve()), np.asarray(fem_cx.solve())
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert rel < 1e-12, f"promoted real-form Bloch disagrees with the complex path: rel {rel:.3e}"


def test_bloch_steady_honors_solve_fn_and_x0():
    """``solve_fn=`` and ``x0=`` used to be silently DISCARDED on a Bloch problem (the dedicated
    block solve ignored them and sparse-LU'd anyway). On the fused path they are honored."""
    kx = PI / 2
    fem = _bloch_strip(kx)
    ref = np.asarray(fem.solve())

    calls = []

    def counting_solver(A, b):
        calls.append(1)
        from jno.utils.solver.linear import sparse_lu_solve

        return sparse_lu_solve(A, b)

    got = np.asarray(_bloch_strip(kx).solve(solve_fn=counting_solver))
    assert len(calls) == 1, "solve_fn must be called exactly once on a Bloch problem"
    assert np.allclose(got, ref, atol=1e-10)

    # x0= (previously an explicit NotImplementedError): warm-starting from the answer reproduces it
    warm = np.asarray(_bloch_strip(kx).solve(x0=ref, linear=jno.solve.bicgstab()))
    rel = np.linalg.norm(warm - ref) / np.linalg.norm(ref)
    assert rel < 1e-6, f"x0 warm start on Bloch drifted: rel {rel:.3e}"


def test_bloch_complex_transient_marches_the_plane_wave():
    """A Bloch tie on a complex transient used to crash with a bare while_loop dtype error (the
    complex ``P`` leaked into the real 2n scan carry via ``blkdiag(P, P)``). With ``B(P)`` the march
    runs and recovers the quasi-periodic plane wave ``ψ = e^{i kx x} e^{-i kx² t}``."""
    kx = PI / 2

    def build(amp):
        d = jno.domain(jno.Shape.rect(0, 0, 1.0, 1.0, size=0.15), time=(0.0, 0.02, 21))
        e = 1e-6
        d.tag("left", lambda x, y: (x < e) & (y > e) & (y < 1 - e))
        d.tag("right", lambda x, y: (x > 1 - e) & (y > e) & (y < 1 - e))
        u, phi = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
        psi0 = amp * jno.np.exp(1j * kx * ci[0])
        return jno.fem(
            [
                ui.t * vi + 1j * (ui.x * vi.x + ui.y * vi.y),
                u(ci[0], ci[1]) - psi0,
                u(xl, yl) - np.exp(-1j * kx) * u(xr, yr),
            ]
        )

    fem = build(1.0)
    assert fem.is_complex and fem.is_transient
    traj = np.asarray(fem.solve())
    assert np.iscomplexobj(traj)
    pts = np.asarray(fem.points)
    exact = np.exp(1j * kx * pts[:, 0]) * np.exp(-1j * kx**2 * float(fem.t1))
    rel = np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact)
    assert rel < 6e-2, f"Bloch plane-wave march rel-L2 {rel:.3e}"
    norm_ratio = np.linalg.norm(traj[-1]) / np.linalg.norm(traj[0])
    assert 0.97 < norm_ratio < 1.03, f"unitary march norm ratio {norm_ratio:.4f}"

    # extreme: a zero IC stays identically zero through the B(P) reduction/prolongation
    traj0 = np.asarray(build(0.0).solve())
    assert float(np.abs(traj0).max()) < 1e-12


def test_bloch_on_a_real_march_fails_loud():
    """A Bloch phase forces a complex field; a REAL transient (heat) march cannot carry it and must
    say so at build time — it used to surface as a bare while_loop dtype crash at evaluation."""
    d = jno.domain(jno.Shape.rect(0, 0, 1.0, 1.0, size=0.2), time=(0.0, 0.1, 9))
    e = 1e-6
    d.tag("left", lambda x, y: (x < e) & (y > e) & (y < 1 - e))
    d.tag("right", lambda x, y: (x > 1 - e) & (y > e) & (y < 1 - e))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ic = jno.np.cos(2 * PI * ci[0])
    with pytest.raises(NotImplementedError, match="Bloch.*REAL transient"):
        jno.fem(
            [
                ui.t * vi + ui.x * vi.x + ui.y * vi.y,
                u(ci[0], ci[1]) - ic,
                u(xl, yl) - np.exp(-1j * 0.7) * u(xr, yr),
            ]
        )


def test_bloch_on_a_nonlinear_form_fails_loud():
    """Nonlinear + Bloch: the promotion cannot linearize the form, so it must refuse by name."""
    d = jno.domain(jno.Shape.rect(0, 0, 1.0, 1.0, size=0.2))
    e = 1e-6
    d.tag("left", lambda x, y: (x < e) & (y > e) & (y < 1 - e))
    d.tag("right", lambda x, y: (x > 1 - e) & (y > e) & (y < 1 - e))
    u, phi = d.fem_symbols()
    xi, yi, *_ = d.variable("interior", split=True)
    xl, yl, *_ = d.variable("left", split=True)
    xr, yr, *_ = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    with pytest.raises(NotImplementedError, match="NONLINEAR"):
        jno.fem(
            [
                ui.x * vi.x + ui.y * vi.y + (u * u) * vi - 1.0 * vi,
                u(xl, yl) - np.exp(-1j * 0.7) * u(xr, yr),
            ]
        )


def test_bloch_scaled_by_nonconstant_rejected():
    """A tie may be scaled only by a constant scalar; a coordinate-dependent factor is not a Bloch phase."""
    d = jno.domain(jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5))
    e = 1e-6
    d.tag("left", lambda x, y, z: x < e)
    d.tag("right", lambda x, y, z: x > 1 - e)
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    lc = d.variable("left", split=True)
    rc = d.variable("right", split=True)
    ul = u.bind(x=lc[0], y=lc[1], z=lc[2])
    ur = u.bind(x=rc[0], y=rc[1], z=rc[2])
    vi = phi.bind(x=xi, y=yi, z=zi)
    ui = u.bind(x=xi, y=yi, z=zi)
    with pytest.raises(ValueError, match="periodic tie"):
        jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - (u * vi), ul - rc[0] * ur])  # x*u(B): not constant


@pytest.mark.slow
def test_bloch_empty_cell_transmits_at_oblique():
    """The end-to-end physical check: an empty (eps=1) periodic cell must transmit fully (T=1, R=0) at
    oblique incidence for *every* Bloch phase -- validates the complex prolongation + Hermitian reduce +
    complex solve, independent of any scattering structure."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        import jax.numpy as jnp

        K0 = 2 * np.pi
        P0, Lz = 0.6, 1.6
        for deg in (0.0, 20.0, 35.0):
            kx = float(K0 * np.sin(np.deg2rad(deg)))
            kz = float(np.sqrt(K0**2 - kx**2))
            d = jno.domain(jno.Shape.box(0, 0, 0, P0, P0, Lz, size=0.12))
            e = 1e-6
            for nm, f in [
                ("left", lambda x, y, z: x < e),
                ("right", lambda x, y, z: x > P0 - e),
                ("front", lambda x, y, z: y < e),
                ("back", lambda x, y, z: y > P0 - e),
                ("bottom", lambda x, y, z: z < e),
                ("top", lambda x, y, z: z > Lz - e),
            ]:
                d.tag(nm, f)
            u, phi = d.fem_symbols()
            xi, yi, zi, _ = d.variable("interior", split=True)
            ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

            def fc(nm):
                c = d.variable(nm, split=True)
                return c, u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])

            cb, ubt, vbt = fc("bottom")
            _ct, utp, vtp = fc("top")
            _cl, ul, _ = fc("left")
            _cr, ur, _ = fc("right")
            _cf, uf, _ = fc("front")
            _ck, ubk, _ = fc("back")
            cphase = np.exp(-1j * kx * P0)
            src = jno.fn(lambda x, y: jnp.exp(1j * kx * x), [cb[0], cb[1]])
            cons = [
                ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * (u * vi),
                -(1j * kz * utp) * vtp,
                -(1j * kz * ubt - 2j * kz * src) * vbt,
                ul - cphase * ur,
                uf - ubk,
            ]
            uu = np.asarray(jno.fem(cons).solve())
            P = np.asarray(d.points)
            top = P[:, 2] > Lz - 1e-4
            bot = P[:, 2] < 1e-4
            t0 = (uu[top] * np.exp(-1j * kx * P[top, 0])).mean()
            r0 = (uu[bot] * np.exp(-1j * kx * P[bot, 0])).mean() - 1.0
            T, R = abs(t0) ** 2, abs(r0) ** 2
            assert abs(T - 1.0) < 0.03, f"theta={deg}: empty cell T={T:.3f} (expected 1)"
            assert R < 0.02, f"theta={deg}: empty cell R={R:.3f} (expected 0)"
    finally:
        jax.config.update("jax_enable_x64", prev)
