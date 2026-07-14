"""Bloch / quasi-periodic ties: ``u(A) = e^{i k·L} u(B)``. The phase makes the prolongation ``P``
complex, so the reduction is Hermitian (``P^H A P``) and the complex system is solved directly.

Plain periodic (phase == 1) must stay a real 0/1 selection (regression guard)."""

import numpy as np
import pytest

import jno


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
