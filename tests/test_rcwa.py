"""Tests for jno.rcwa.

Two layers:
* the FROM-FEM inference (jno.rcwa(constraints, eps, ...)) is pure trace-walking + geometry and runs
  WITHOUT the optional fmmax backend — these assert the inferred RcwaSpec;
* the forward engine (Rcwa, and .solve()) needs fmmax and is guarded per-test."""

import importlib.util

import numpy as np
import pytest

import jno

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")

WL = 1.0
INF = np.inf


# ------------------------------------------------------------------------------------
# A small PERIODIC-supercell Helmholtz problem: an a-Si pillar slab, periodic side walls,
# absorbing top, absorbing bottom carrying a normally-incident wave. (Periodic variant of
# code/opt3d.py from the metasurface project.)
# ------------------------------------------------------------------------------------
def _build_periodic_problem(dx=0.4):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _domain_from_arrays

    K0 = 2 * np.pi
    EMAX = 6.0
    Lx = Ly = 2.4  # keep opt3d's default extent (avoids the _domain_from_arrays "no trial fields" footgun)
    Lz = 3.2
    ZL0, ZL1 = 0.8, 1.15
    Eb = 1e-6
    xs = np.linspace(0, Lx, int(round(Lx / dx)) + 1)
    ys = np.linspace(0, Ly, int(round(Ly / dx)) + 1)
    zs = np.linspace(0, Lz, int(round(Lz / dx)) + 1)
    nx, ny, nz = len(xs), len(ys), len(zs)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)

    def vid(i, j, k):
        return (i * ny + j) * nz + k

    CUBE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    TETS6 = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    tets = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                c = [vid(i + a, j + b, k + cc) for (a, b, cc) in CUBE]
                for t in TETS6:
                    tets.append([c[t[0]], c[t[1]], c[t[2]], c[t[3]]])
    tets = np.asarray(tets)
    F = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    Fs = np.sort(F, axis=1)
    uq, cnt = np.unique(Fs, axis=0, return_counts=True)
    BF = uq[cnt == 1]

    d = _domain_from_arrays(
        jno.domain.cube(x_range=(0, Lx), y_range=(0, Ly), z_range=(0, Lz), mesh_size=1.0), P, tets, BF, copy=True
    )
    d.tag("bottom", lambda x, y, z: z < Eb)
    d.tag("top", lambda x, y, z: z > Lz - Eb)
    d.tag("left", lambda x, y, z: x < Eb)
    d.tag("right", lambda x, y, z: x > Lx - Eb)
    d.tag("front", lambda x, y, z: y < Eb)
    d.tag("back", lambda x, y, z: y > Ly - Eb)

    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    bt = d.variable("bottom", split=True)
    tp = d.variable("top", split=True)
    lf = d.variable("left", split=True)
    rt = d.variable("right", split=True)
    fr = d.variable("front", split=True)
    bk = d.variable("back", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    ubt, vbt = u.bind(x=bt[0], y=bt[1], z=bt[2]), phi.bind(x=bt[0], y=bt[1], z=bt[2])
    utp, vtp = u.bind(x=tp[0], y=tp[1], z=tp[2]), phi.bind(x=tp[0], y=tp[1], z=tp[2])
    ulf = u.bind(x=lf[0], y=lf[1], z=lf[2])
    urt = u.bind(x=rt[0], y=rt[1], z=rt[2])
    ufr = u.bind(x=fr[0], y=fr[1], z=fr[2])
    ubk = u.bind(x=bk[0], y=bk[1], z=bk[2])

    lay = jno.fn(lambda x, y, z: jnp.where((z >= ZL0) & (z < ZL1), 1.0, 0.0), [xi, yi, zi])
    rho = jno.np.parameter(phi, name="rho")
    eps = 1.0 + lay * (0.5 * (1 + jno.np.tanh(rho / 2))) * (EMAX - 1.0)
    constraints = [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ulf - urt,
        ufr - ubk,
    ]
    return constraints, eps


# ------------------------------------------------------------------------------------
# Inference (no fmmax needed)
# ------------------------------------------------------------------------------------
def test_infer_spec_from_periodic_problem():
    """Everything — period, layers, permittivity AND wavelength — inferred from the constraint list."""
    constraints, _ = _build_periodic_problem()
    rc = jno.rcwa(constraints, orders=100, grid=16, nz=33)  # no eps, no wavelength
    s = rc.spec
    # periodicity + period from the Floquet ties
    assert abs(s.period[0] - 2.4) < 1e-6 and abs(s.period[1] - 2.4) < 1e-6
    assert set(s.periodic_axes) == {"x", "y"}
    # air | pillar-slab | air  -> three layers, middle ~0.35 thick, ambients semi-infinite
    assert len(s.layers) == 3, [ly[0] for ly in s.layers]
    assert s.layers[0][0] == INF and s.layers[-1][0] == INF
    assert abs(s.layers[1][0] - 0.35) < 0.12, s.layers[1][0]
    # relative permittivity recovered: patterned layer -> 3.5 at rho=0; ambients are vacuum (1.0)
    assert abs(float(np.real(s.layers[1][1]).max()) - 3.5) < 0.05
    assert abs(float(np.real(s.layers[0][1]).max()) - 1.0) < 0.02
    # wavelength inferred from the vacuum superstrate (k0 = 2*pi -> lambda = 1)
    assert abs(s.wavelength - 1.0) < 1e-3
    # normally-incident source read off the bottom face
    assert s.k_in == (0.0, 0.0)
    assert s.source_face in ("bottom", "top")


def test_explicit_wavelength_override():
    constraints, _ = _build_periodic_problem()
    rc = jno.rcwa(constraints, orders=100, wavelength=WL, grid=16, nz=33)
    assert abs(rc.spec.wavelength - WL) < 1e-9


def test_non_periodic_problem_raises():
    """A finite aperture (absorbing side walls, no ties) must be rejected, never silently periodicised."""
    constraints, _ = _build_periodic_problem()
    non_periodic = constraints[:3]  # drop the two periodic ties
    with pytest.raises(jno.RcwaError, match="periodic"):
        jno.rcwa(non_periodic, orders=100, grid=16, nz=33)


def test_bad_problem_type_raises():
    with pytest.raises(jno.RcwaError, match="constraint list"):
        jno.rcwa(42, orders=10)


# ------------------------------------------------------------------------------------
# detect_layers (pure numpy, no fmmax)
# ------------------------------------------------------------------------------------
def test_detect_layers_extruded():
    Ng, Nz = 12, 60
    z = np.linspace(0, 3.2, Nz)
    E = np.ones((Nz, Ng, Ng))
    for k, zz in enumerate(z):
        if 0.8 <= zz < 1.15:
            E[k] = 1.0
            E[k, 4:8, 4:8] = 11.0
    layers = detect_layers_ref(E, z)
    assert len(layers) == 3
    assert layers[0][0] == INF and layers[-1][0] == INF


def detect_layers_ref(E, z):
    from jno.rcwa import detect_layers

    return detect_layers(E, z)


def test_detect_layers_continuous_raises():
    from jno.rcwa import detect_layers

    Ng, Nz = 8, 40
    z = np.linspace(0, 1, Nz)
    E = np.stack([np.full((Ng, Ng), 1.0 + zz) for zz in z])  # continuous in z
    with pytest.raises(jno.RcwaError, match="continuous"):
        detect_layers(E, z)


# ------------------------------------------------------------------------------------
# Forward engine + analytic Fresnel (needs fmmax)
# ------------------------------------------------------------------------------------
def tmm_slab(nn, h, lam):
    def iface(a, b):
        r = (a - b) / (a + b)
        t = 2 * a / (a + b)
        return np.array([[1, r], [r, 1]]) / t

    beta = 2 * np.pi * nn * h / lam
    M = iface(1, nn) @ np.array([[np.exp(-1j * beta), 0], [0, np.exp(1j * beta)]]) @ iface(nn, 1)
    return abs(1 / M[0, 0]) ** 2


@needs_fmmax
@pytest.mark.parametrize("nn,h", [(1.45, 0.5), (2.0, 0.4), (3.317, 0.3)])
def test_engine_fresnel_matches_analytic(nn, h):
    from jno.rcwa import Rcwa

    # period 0.85 < wavelength 1.0 keeps all higher orders evanescent AND avoids the Rayleigh anomaly
    # at wavelength == period (grazing order, NaN); a uniform slab is period-independent otherwise.
    rc = Rcwa([(INF, 1.0), (h, nn**2), (INF, 1.0)], period=(0.85, 0.85), orders=5, wavelength=WL, assume_periodic=True)
    sol = rc.solve()
    T, R = sol.efficiency("T"), sol.efficiency("R")
    assert abs(T - tmm_slab(nn, h, WL)) < 2e-3
    assert abs(T + R - 1) < 1e-3


@needs_fmmax
def test_engine_requires_assume_periodic():
    from jno.rcwa import Rcwa

    with pytest.raises(jno.RcwaError, match="assume_periodic"):
        Rcwa([(INF, 1.0), (0.3, 11.0), (INF, 1.0)], period=(1.0, 1.0), orders=5, wavelength=WL)


@needs_fmmax
def test_efficiency_is_differentiable_in_the_design():
    """jax.grad of transmission w.r.t. the permittivity flows through the modal solve (matches FD)."""
    import jax
    import jax.numpy as jnp

    from jno.rcwa import Rcwa

    P, r, th, ng = 0.6, 0.18, 0.35, 32
    xs = (np.arange(ng) + 0.5) / ng * P
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    mask = jnp.asarray((((X - P / 2) ** 2 + (Y - P / 2) ** 2) < r**2).astype(float))
    rc = Rcwa(
        [(INF, 1.0), (th, jnp.ones((ng, ng))), (INF, 1.0)], period=(P, P), orders=40, wavelength=WL, assume_periodic=True
    )

    def T(epsval):  # transmission as a function of the pillar permittivity
        g = 1.0 + mask * (epsval - 1.0)
        return rc.solve(layers=[(INF, 1.0), (th, g), (INF, 1.0)]).efficiency("T")

    grad = float(jax.grad(T)(6.0))
    h = 1e-2
    fd = (float(T(6.0 + h)) - float(T(6.0 - h))) / (2 * h)
    assert grad == grad and abs(grad) > 1e-3, "gradient must be finite and non-zero"
    assert abs(grad - fd) < 5e-3, f"autodiff {grad} vs finite-diff {fd}"  # FD is the approximation here


@needs_fmmax
def test_parameter_anywhere_wavelength_and_eps_flow():
    """A jno.np.parameter used as the wavelength (K0) — anywhere in the graph — is a differentiable knob:
    jno.rcwa re-derives eps AND k0 from the parameterized coefficient, and jax.grad flows (matches FD)."""
    import jax
    import jax.numpy as jnp

    P0, Lz = 0.6, 3.2
    d = jno.domain(jno.Shape.box(0, 0, 0, P0, P0, Lz, size=0.2))
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
        return u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])

    ubt, vbt = fc("bottom")
    utp, vtp = fc("top")
    ul, _ = fc("left")
    ur, _ = fc("right")
    uf, _ = fc("front")
    ubk, _ = fc("back")
    K0 = jno.np.parameter((), name="k0").initialize(jax.nn.initializers.constant(2 * np.pi))  # WAVELENGTH param
    ind = jno.fn(
        lambda x, y, z: jnp.where(((x - 0.3) ** 2 + (y - 0.3) ** 2 < 0.18**2) & (z >= 0.8) & (z < 1.15), 1.0, 0.0),
        [xi, yi, zi],
    )
    eps = 1.0 + ind * (6.0 - 1.0)
    cons = [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ul - ur,
        uf - ubk,
    ]
    rc = jno.rcwa(cons, orders=40, grid=32, nz=40, params={"k0": 2 * np.pi})
    assert abs(rc.spec.wavelength - 1.0) < 1e-3

    def T(k0v):
        return rc.solve(params={"k0": k0v}).efficiency("T")

    k = 2 * np.pi
    g = float(jax.grad(T)(k))
    h = 1e-2
    fd = (float(T(k + h)) - float(T(k - h))) / (2 * h)
    assert g == g and abs(g - fd) < 5e-2, f"dT/dK0 autodiff {g} vs finite-diff {fd}"


@needs_fmmax
def test_broadband_spectrum_and_wavelength_gradient():
    """Sweeping wavelength gives a dispersion curve; T is differentiable in wavelength (matches FD)."""
    import jax
    import jax.numpy as jnp

    from jno.rcwa import Rcwa

    P, r, th, ng = 0.6, 0.18, 0.5, 32
    xs = (np.arange(ng) + 0.5) / ng * P
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    E = jnp.asarray(np.where((X - P / 2) ** 2 + (Y - P / 2) ** 2 < r**2, 6.0, 1.0))
    rc = Rcwa([(INF, 1.0), (th, E), (INF, 1.0)], period=(P, P), orders=40, wavelength=1.0, assume_periodic=True)

    def T(wl):
        return rc.solve(wavelength=wl).efficiency("T")

    spec = [float(T(wl)) for wl in (0.9, 1.0, 1.2)]
    assert max(spec) - min(spec) > 0.1, f"expected a dispersive response, got {spec}"  # not flat
    g = float(jax.grad(T)(1.0))
    h = 1e-2
    fd = (float(T(1.0 + h)) - float(T(1.0 - h))) / (2 * h)
    assert abs(g - fd) < 5e-2, f"dT/dlambda autodiff {g} vs finite-diff {fd}"


@needs_fmmax
def test_infer_and_solve_end_to_end():
    """The full path: infer from the periodic problem, then solve — energy must be conserved."""
    constraints, _ = _build_periodic_problem()
    rc = jno.rcwa(constraints, orders=100, grid=32, nz=33)
    sol = rc.solve()
    T, R = sol.efficiency("T"), sol.efficiency("R")
    assert T + R <= 1.0 + 5e-3 and T >= 0 and R >= 0
