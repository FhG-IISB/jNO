"""jno.rcwa with an in-plane PERFECTLY MATCHED LAYER (PML), inferred from the traced weak form.

A PML is a complex coordinate stretch ``S = 1 + iσ/k`` written straight into the scalar Helmholtz
volume term as anisotropic stiffness coefficients::

    (Sy/Sx)·∂ₓu∂ₓv + (Sx/Sy)·∂ᵧu∂ᵧv + (Sx·Sy)·∂_zu∂_zv − k0²·(Sx·Sy)·ε·(u·v)

The front door reads the stretch Λ = diag(Sy/Sx, Sx/Sy, Sx·Sy) off the stiffness, forms the Maxwell
uniaxial PML (ε̂ = ε·Λ, μ̂ = Λ), and routes it to fmmax's general anisotropic eigensolve. The Floquet
ties stay (fmmax is periodic); the PML frame just makes the supercell walls non-coupling, so the cell
behaves like an isolated scatterer.

Checks: (A) a trivial stretch (σ=0 ⇒ Λ=I) reduces EXACTLY to the scalar no-PML result; (B) a real
absorber laterally absorbs light diffracted by a scatterer, so T+R drops below 1 while σ=0 conserves it.
"""

import importlib.util
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the fmmax solve OOMs on a small GPU at these orders

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jno  # noqa: E402
from jno.rcwa import RcwaError  # noqa: E402
from jno.trace import FunctionCall  # noqa: E402
from jno.trace_evaluator import TraceEvaluator  # noqa: E402
from jno.utils.solver.fem_adapt import _domain_from_arrays  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

K0 = 2 * jnp.pi
Z0, Z1, LZ = 1.0, 2.0, 3.0  # slab occupies z in [1, 2]; box height 3
relu = lambda z: jno.np.maximum(z, 0.0)  # noqa: E731


def _structured_box(Lx, dx=0.1, dz=0.2):
    """A structured tet mesh of an [0,Lx]²×[0,LZ] box (z-nodes land on the layer boundaries)."""
    xs = np.linspace(0, Lx, int(round(Lx / dx)) + 1)
    zs = np.linspace(0, LZ, int(round(LZ / dz)) + 1)
    ny, nz = len(xs), len(zs)
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    vid = lambda i, j, k: (i * ny + j) * nz + k  # noqa: E731
    CUBE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    TETS6 = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    tets = []
    for i in range(len(xs) - 1):
        for j in range(len(xs) - 1):
            for k in range(nz - 1):
                c = [vid(i + a, j + b, k + cc) for (a, b, cc) in CUBE]
                tets += [[c[t[0]], c[t[1]], c[t[2]], c[t[3]]] for t in TETS6]
    tets = np.asarray(tets)
    F = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    uq, cnt = np.unique(np.sort(F, 1), axis=0, return_counts=True)
    d = _domain_from_arrays(jno.Shape.box(0, 0, 0, Lx, Lx, LZ, size=0.2).domain(), P, tets, uq[cnt == 1], copy=True)
    e = 1e-6
    for nm, f in [
        ("bottom", lambda x, y, z: z < e),
        ("top", lambda x, y, z: z > LZ - e),
        ("left", lambda x, y, z: x < e),
        ("right", lambda x, y, z: x > Lx - e),
        ("front", lambda x, y, z: y < e),
        ("back", lambda x, y, z: y > Lx - e),
    ]:
        d.tag(nm, f)
    return d


def _constraints(Lx, sigma0, eps_fn, w=None):
    """Scalar Helmholtz on the supercell with an in-plane PML frame of width ``w`` and strength ``sigma0``
    (``sigma0=0`` ⇒ no absorption). ``eps_fn(x,y,z)`` is the permittivity."""
    if w is None:
        w = 0.25 * Lx
    d = _structured_box(Lx)
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

    def face(n):
        c = d.variable(n, split=True)
        return u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])

    ubt, vbt = face("bottom")
    utp, vtp = face("top")
    ul, _ = face("left")
    ur, _ = face("right")
    uf, _ = face("front")
    ub, _ = face("back")
    sx = sigma0 * (relu(w - xi) ** 2 + relu(xi - (Lx - w)) ** 2) / w**2  # quadratic profile near x-walls
    sy = sigma0 * (relu(w - yi) ** 2 + relu(yi - (Lx - w)) ** 2) / w**2
    Sx, Sy = 1.0 + 1j * sx / K0, 1.0 + 1j * sy / K0  # complex coordinate stretch (=1 in the core)
    eps = jno.fn(eps_fn, [xi, yi, zi])
    vol = (
        (Sy / Sx) * (ui.x * vi.x)
        + (Sx / Sy) * (ui.y * vi.y)
        + (Sx * Sy) * (ui.z * vi.z)  # uniaxial 3-D stretch, Sz = 1
        - K0**2 * (Sx * Sy) * eps * (u * vi)
    )
    return [vol, -(1j * K0 * utp) * vtp, -(1j * K0 * ubt - 2j * K0) * vbt, ul - ur, uf - ub]


def _uniform(x, y, z):
    return jnp.where((z >= Z0) & (z < Z1), 6.0, 1.0)  # a-Si slab, no lateral structure


def _scatterer(x, y, z):  # a localized off-centre pillar -> diffracts light laterally
    inpill = (jnp.abs(x - 0.55) < 0.18) & (jnp.abs(y - 0.55) < 0.18)
    return jnp.where((z >= Z0) & (z < Z1) & inpill, 4.0, 1.0)


def _TR(cons, orders=80, grid=32):
    sol = jno.rcwa(cons, orders=orders, grid=grid).solve()
    return float(sol.efficiency("T")), float(sol.efficiency("R"))


@needs_fmmax
def test_zero_sigma_pml_reduces_to_scalar():
    """A PML with σ=0 (S=1 ⇒ Λ=I, μ̂=I, ε̂=ε·I) must give EXACTLY the plain scalar-ε result — the PML
    detection + general-anisotropic routing reduces correctly when the stretch is trivial."""
    Tp, Rp = _TR(_constraints(1.2, 0.0, _uniform))  # PML form, σ=0
    Ts, Rs = _TR([c for c in _scalar_slab(1.2)])  # plain scalar Helmholtz, same slab
    assert Tp == pytest.approx(Ts, abs=1e-5)
    assert Rp == pytest.approx(Rs, abs=1e-5)


def _scalar_slab(Lx):
    """The scalar no-PML reference: same box + slab, plain Laplacian stiffness."""
    d = _structured_box(Lx)
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

    def face(n):
        c = d.variable(n, split=True)
        return u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])

    ubt, vbt = face("bottom")
    utp, vtp = face("top")
    ul, _ = face("left")
    ur, _ = face("right")
    uf, _ = face("front")
    ub, _ = face("back")
    eps = jno.fn(_uniform, [xi, yi, zi])
    vol = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi)
    return [vol, -(1j * K0 * utp) * vtp, -(1j * K0 * ubt - 2j * K0) * vbt, ul - ur, uf - ub]


@needs_fmmax
def test_pml_absorbs_laterally_scattered_light():
    """The decisive PML check: for a scatterer that diffracts light into the plane, a real absorber
    (σ>0) removes the laterally-scattered power, so T+R < 1; with σ=0 the same cell conserves energy
    (T+R ≈ 1). Proves the PML actually absorbs (it is not a no-op)."""
    T0, R0 = _TR(_constraints(1.4, 0.0, _scatterer))  # no absorber -> energy conserved
    Ta, Ra = _TR(_constraints(1.4, 3.0, _scatterer))  # real absorber -> lateral loss
    assert T0 + R0 == pytest.approx(1.0, abs=1e-2)  # σ=0 supercell conserves energy
    assert Ta + Ra < 0.98  # the PML drained laterally-diffracted power


@needs_fmmax
def test_pml_scatterer_is_differentiable():
    """A design parameter on the SCATTERER inside a PML supercell flows through the solve: the PML layers are
    re-derived from the parameter (ε̂ = ε·Λ, μ̂ = Λ), rc.solve() is a trace node, and jax.grad of transmission
    matches a finite difference. This is inverse design of an isolated scatterer."""
    Lx = 1.4
    w = 0.35 * Lx
    d = _structured_box(Lx)
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

    def face(n):
        c = d.variable(n, split=True)
        return u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])

    ubt, vbt = face("bottom")
    utp, vtp = face("top")
    ul, _ = face("left")
    ur, _ = face("right")
    uf, _ = face("front")
    ub, _ = face("back")
    sx = 3.0 * (relu(w - xi) ** 2 + relu(xi - (Lx - w)) ** 2) / w**2
    sy = 3.0 * (relu(w - yi) ** 2 + relu(yi - (Lx - w)) ** 2) / w**2
    Sx, Sy = 1.0 + 1j * sx / K0, 1.0 + 1j * sy / K0
    ep = jno.np.parameter((), name="ep").initialize(jax.nn.initializers.constant(6.0))  # scatterer ε inside the PML
    ind = jno.fn(
        lambda x, y, z: jnp.where(
            (jnp.abs(x - Lx / 2) < 0.18) & (jnp.abs(y - Lx / 2) < 0.18) & (z >= 1.0) & (z < 2.0), 1.0, 0.0
        ),
        [xi, yi, zi],
    )
    eps = 1.0 + ind * (ep - 1.0)
    vol = (
        (Sy / Sx) * (ui.x * vi.x)
        + (Sx / Sy) * (ui.y * vi.y)
        + (Sx * Sy) * (ui.z * vi.z)
        - K0**2 * (Sx * Sy) * eps * (u * vi)
    )
    cons = [vol, -(1j * K0 * utp) * vtp, -(1j * K0 * ubt - 2j * K0) * vbt, ul - ur, uf - ub]
    rc = jno.rcwa(cons, orders=60, grid=30, params={"ep": 6.0})
    node = rc.solve().efficiency("T")
    assert isinstance(node, FunctionCall)

    def T(v):
        mod = eqx.tree_at(lambda m: m.value, ep.model.module, jnp.asarray(v))
        return TraceEvaluator({ep.model.layer_id: mod}).evaluate(node)

    g = float(jax.grad(T)(6.0))
    h = 1e-2
    fd = (float(T(6.0 + h)) - float(T(6.0 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=3e-2)
    assert abs(g) > 1e-3


@needs_fmmax
def test_pml_raises_on_offdiagonal_stretch():
    """A cross ∂ₓu ∂ᵧv stiffness term (a non-diagonal stretch) is rejected — only uniaxial PML is supported."""
    d = _structured_box(1.2)
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

    def face(n):
        c = d.variable(n, split=True)
        return u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])

    ubt, vbt = face("bottom")
    utp, vtp = face("top")
    ul, _ = face("left")
    ur, _ = face("right")
    uf, _ = face("front")
    ub, _ = face("back")
    eps = jno.fn(_uniform, [xi, yi, zi])
    vol = ui.x * vi.x + (1.0 + 1j) * (ui.x * vi.y) + ui.z * vi.z - K0**2 * eps * (u * vi)  # off-diagonal ∂ₓu ∂ᵧv
    cons = [vol, -(1j * K0 * utp) * vtp, -(1j * K0 * ubt - 2j * K0) * vbt, ul - ur, uf - ub]
    with pytest.raises(RcwaError, match="off-diagonal"):
        jno.rcwa(cons, orders=20, grid=16)
