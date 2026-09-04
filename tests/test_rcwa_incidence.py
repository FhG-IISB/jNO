"""Regression for the RCWA incident-mode fix (excite the true 0th order, not the max-flux eigenmode).

The engine used to pick its incident plane wave by argmax(eigenmode_poynting_flux). That is only correct for
a subwavelength (single propagating order) cell; once the period exceeds the wavelength, several ambient
orders propagate and the (0,0) order does NOT carry the max eigenmode flux, so argmax excited an OBLIQUE
order. This test uses period > λ (multiple orders) and checks the uniform slab transmits per the analytic
Airy formula at NORMAL incidence -- which the old code got wrong (it gave the oblique-angle transmission).
"""

import importlib.util
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64. The session default is x64-off (see tests/conftest.py), and this
    flag is process-wide -- save/restore keeps it from leaking to whatever module runs next."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


import jno  # noqa: E402
from jno.utils.solver.fem_adapt import _domain_from_arrays  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

K0 = 2 * jnp.pi
LZ = 3.0
EPS = 6.0  # a-Si-like slab
Z0, Z1 = 1.0, 2.0  # slab z-range (structured mesh puts nodes on 1.0 and 2.0)


def _slab_sol(period, orders=80, grid=64):
    """A uniform (laterally featureless) slab of the given transverse period, on a structured mesh."""
    P = period
    xs = np.linspace(0, P, int(round(P / 0.1)) + 1)
    zs = np.linspace(0, LZ, int(round(LZ / 0.2)) + 1)
    ny, nz = len(xs), len(zs)
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    Pt = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    vid = lambda i, j, k: (i * ny + j) * nz + k  # noqa: E731
    CUBE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    TE = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    tets = []
    for i in range(len(xs) - 1):
        for j in range(len(xs) - 1):
            for k in range(nz - 1):
                c = [vid(i + a, j + b, k + cc) for (a, b, cc) in CUBE]
                tets += [[c[t[0]], c[t[1]], c[t[2]], c[t[3]]] for t in TE]
    tets = np.asarray(tets)
    F = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    uq, cnt = np.unique(np.sort(F, 1), axis=0, return_counts=True)
    d = _domain_from_arrays(jno.Shape.box(0, 0, 0, P, P, LZ, size=0.2).domain(), Pt, tets, uq[cnt == 1], copy=True)
    e = 1e-6
    for nm, f in [
        ("bottom", lambda x, y, z: z < e),
        ("top", lambda x, y, z: z > LZ - e),
        ("left", lambda x, y, z: x < e),
        ("right", lambda x, y, z: x > P - e),
        ("front", lambda x, y, z: y < e),
        ("back", lambda x, y, z: y > P - e),
    ]:
        d.tag(nm, f)
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
    ub, _ = fc("back")
    eps = jno.fn(lambda x, y, z: jnp.where((z >= Z0) & (z < Z1), EPS, 1.0), [xi, yi, zi])
    cons = [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ul - ur,
        uf - ub,
    ]
    return jno.rcwa(cons, orders=orders, grid=grid)


def _airy_T(n, d, cos_theta=1.0):
    """Analytic slab transmission at incidence angle with cosine ``cos_theta`` (1 = normal)."""
    k0 = float(K0)
    kz = n * k0 * cos_theta  # simplistic angle handling for the discriminator (normal vs oblique)
    r = (1 - n) / (1 + n)
    ph = np.exp(2j * kz * d)
    rho = r * (1 - ph) / (1 - r**2 * ph)
    return float(1 - abs(rho) ** 2)


@needs_fmmax
def test_multi_order_slab_matches_normal_incidence_airy():
    """Period 1.6 > λ=1 -> the (±1,0),(0,±1) ambient orders propagate. The uniform slab must still transmit
    per the exact Airy formula at NORMAL incidence (matched at the slab's inferred thickness). The old
    argmax-flux incident excited an oblique order and would give the wrong-angle transmission here."""
    rc = _slab_sol(period=1.6)
    n_eff = float(np.sqrt(np.real(np.mean(np.asarray(rc.spec.layers[1][1])))))
    d_eff = float(rc.spec.layers[1][0])
    T = float(rc.solve().efficiency("T"))
    assert T == pytest.approx(_airy_T(n_eff, d_eff, cos_theta=1.0), abs=2e-3)  # normal-incidence Airy


@needs_fmmax
def test_subwavelength_slab_still_matches_airy():
    """Control: a subwavelength cell (single order) also matches normal Airy -- the fix leaves the case the
    old code already handled unchanged."""
    rc = _slab_sol(period=0.5, orders=20, grid=24)
    n_eff = float(np.sqrt(np.real(np.mean(np.asarray(rc.spec.layers[1][1])))))
    d_eff = float(rc.spec.layers[1][0])
    assert float(rc.solve().efficiency("T")) == pytest.approx(_airy_T(n_eff, d_eff), abs=2e-3)
