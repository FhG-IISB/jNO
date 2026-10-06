"""Which polarization jno.rcwa solves, and that it is the one the constraint list describes.

The oracle is an independent 1-D lamellar-grating RCWA written here from Moharam, Grann, Pommet &
Gaylord, J. Opt. Soc. Am. A 12 (1995) 1068 -- TE (E along the lines) and TM (H along the lines, with the
Lalanne-Morris / Granet-Guizal inverse rule) -- with EXACT Fourier coefficients of the step permittivity.
It shares nothing with fmmax. It matches the Airy formula for a uniform slab and conserves energy to
1e-12; jno.rcwa samples the permittivity on a grid, so it converges to it at first order in the grid.

A scalar Helmholtz list is the TE equation for a 1-D grating: Maxwell reduces to ``Δu + k0²εu = 0``
exactly for the field component along the lines. jno.rcwa used to solve every list x-polarized, which
for lines along y is TM -- a reflectance of 0.107 where the written equation (and jno.fem) says 0.617.
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
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


import jno  # noqa: E402
from jno.rcwa import RcwaError, _scalar_polarization  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

inner, vec = jno.np.inner, jno.np.vector
K0, H = 2 * np.pi, 1.0
PERIOD, FILL, EPS_R, Z0, Z1 = 0.6, 0.5, 11.0, 0.4, 0.45  # a 0.05-thick silicon-like grating, lambda = 1
GRID, ORDERS, TOL = 256, 60, 3e-3  # jno.rcwa at grid 256 sits ~1e-3 from the exact-coefficient reference


# ---------------------------------------------------------------------------------------------------
# the independent reference
# ---------------------------------------------------------------------------------------------------
def _coef(h, a, b):
    return b + (a - b) * FILL if h == 0 else (a - b) * (1 - np.exp(-2j * np.pi * h * FILL)) / (2j * np.pi * h)


def _reference(pol, N=60):
    """Reflectance of the lamellar grating at normal incidence, lambda = 1, vacuum on both sides."""
    m = np.arange(-N, N + 1)
    M = m.size
    kx = np.diag(-m / PERIOD)
    E = np.array([[_coef(i - j, EPS_R, 1.0) for j in range(M)] for i in range(M)])
    Id, Z = np.eye(M), np.zeros((M, M))
    if pol == "TE":
        q2, W = np.linalg.eigh(kx @ kx - E)
        V_of = lambda W, q: W @ np.diag(q)  # noqa: E731
    else:
        A = np.array([[_coef(i - j, 1 / EPS_R, 1.0) for j in range(M)] for i in range(M)])
        q2, W = np.linalg.eig(np.linalg.solve(A, kx @ np.linalg.solve(E, kx) - Id))
        V_of = lambda W, q: A @ W @ np.diag(q)  # noqa: E731
    q = np.sqrt(q2.astype(complex))
    q = np.where(q.real < 0, -q, q)
    V, X = V_of(W, q), np.diag(np.exp(-K0 * q * (Z1 - Z0)))
    kz = np.sqrt((1 - np.diag(kx) ** 2).astype(complex))
    kz = np.where(kz.imag > 0, -kz, kz)
    Y = np.diag(kz)
    d0 = (m == 0).astype(complex)
    top = np.block([[-Id, W, W @ X, Z], [1j * Y, V, -V @ X, Z]])
    bot = np.block([[Z, W @ X, W, -Id], [Z, V @ X, -V, -1j * Y]])
    sol = np.linalg.solve(np.vstack([top, bot]), np.concatenate([d0, 1j * d0, np.zeros(2 * M)]))
    return float((np.abs(sol[:M]) ** 2 * np.real(kz)).sum())


R_TE, R_TM = _reference("TE"), _reference("TM")


def test_the_reference_is_a_reference():
    assert R_TE == pytest.approx(0.61684, abs=1e-4) and R_TM == pytest.approx(0.10695, abs=1e-4)


# ---------------------------------------------------------------------------------------------------
# the problems, written the way a user writes them
# ---------------------------------------------------------------------------------------------------
def _box(px, py):
    d = jno.shape.box(0, 0, 0, px, py, H, size=0.25).domain()
    e = 1e-6
    for nm, f in [
        ("bottom", lambda x, y, z: z < e),
        ("top", lambda x, y, z: z > H - e),
        ("left", lambda x, y, z: x < e),
        ("right", lambda x, y, z: x > px - e),
        ("front", lambda x, y, z: y < e),
        ("back", lambda x, y, z: y > py - e),
    ]:
        d.tag(nm, f)
    return d


def _lines(along):
    """The grating's lines run along ``along``: its permittivity varies across the other axis."""
    if along == "y":
        return lambda x, y, z: jnp.where((Z0 < z) & (z < Z1) & (x < FILL * PERIOD), 1.0, 0.0)
    return lambda x, y, z: jnp.where((Z0 < z) & (z < Z1) & (y < FILL * PERIOD), 1.0, 0.0)


def _scalar(along, pattern=None):
    px, py = (PERIOD, 0.3) if along == "y" else (0.3, PERIOD)
    d = _box(px, py)
    u, v = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)

    def on(tag):
        c = d.variable(tag, split=True)
        return u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])

    (ut, vt), (ub, vb) = on("top"), on("bottom")
    eps = 1.0 + (EPS_R - 1.0) * jno.fn(pattern or _lines(along), [xi, yi, zi])
    return [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * ut) * vt,
        -(1j * K0 * ub - 2j * K0) * vb,
        on("left")[0] - on("right")[0],
        on("front")[0] - on("back")[0],
    ]


def _vector(along, e_inc):
    px, py = (PERIOD, 0.3) if along == "y" else (0.3, PERIOD)
    d = _box(px, py)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
    cu, cv = u.vector.curl(c[0], c[1], c[2]), v.vector.curl(c[0], c[1], c[2])
    nt, nb = d.variable("top", normals=True), d.variable("bottom", normals=True)
    cb = d.variable("bottom", split=True)
    einc = vec(e_inc[0] + 0.0 * cb[0], e_inc[1] + 0.0 * cb[1], 0.0 * cb[2])
    eps = 1.0 + (EPS_R - 1.0) * jno.fn(_lines(along), [c[0], c[1], c[2]])

    def face(n):
        cc = d.variable(n, split=True)
        return u.bind(x=cc[0], y=cc[1], z=cc[2])

    return [
        inner(cu, cv) - K0**2 * eps * inner(ui, vi),
        1j * K0 * inner(u.vector.cross(nt), v.vector.cross(nt)),
        1j * K0 * inner(u.vector.cross(nb), v.vector.cross(nb)) + 2j * K0 * inner(einc, v.vector.cross(nb)),
        face("left") - face("right"),
        face("front") - face("back"),
    ]


def _R(rc):
    return float(jnp.real(rc.solve().efficiency("R")))


# ---------------------------------------------------------------------------------------------------
# a scalar list is solved in the polarization it describes
# ---------------------------------------------------------------------------------------------------
@needs_fmmax
@pytest.mark.parametrize("along", ["y", "x"])
def test_a_scalar_grating_is_solved_in_the_polarization_it_describes(along):
    """E along the lines, whichever way the lines run -- the TE answer, as jno.fem would give."""
    rc = jno.rcwa(_scalar(along), orders=ORDERS, grid=GRID)
    assert rc.spec.polarization == ((0j, 1 + 0j) if along == "y" else (1 + 0j, 0j))
    assert "TE" in rc.spec.polarization_from
    assert _R(rc) == pytest.approx(R_TE, abs=TOL)


@needs_fmmax
def test_the_other_polarization_is_one_argument_away():
    rc = jno.rcwa(_scalar("y"), orders=ORDERS, grid=GRID, polarization="x")
    assert rc.spec.polarization_from == "given"
    assert _R(rc) == pytest.approx(R_TM, abs=TOL)


@needs_fmmax
def test_circular_light_sees_the_mean_of_te_and_tm():
    """A 1-D grating in the classical mount does not mix TE and TM, so circular light reflects their mean."""
    rc = jno.rcwa(_scalar("y"), orders=ORDERS, grid=GRID, polarization=(1.0, 1j))
    assert _R(rc) == pytest.approx(0.5 * (R_TE + R_TM), abs=TOL)


@needs_fmmax
def test_a_two_dimensional_pattern_falls_back_to_x_and_says_so():
    pillar = lambda x, y, z: jnp.where((Z0 < z) & (z < Z1) & ((x - 0.3) ** 2 + (y - 0.15) ** 2 < 0.1**2), 1.0, 0.0)  # noqa: E731
    rc = jno.rcwa(_scalar("y", pattern=pillar), orders=ORDERS, grid=64)
    assert rc.spec.polarization == (1 + 0j, 0j)
    assert rc.spec.polarization_from.startswith("x by default")


def test_a_uniform_stack_lit_obliquely_is_s_polarized():
    """No pattern, incidence along (0.3, 0.4): the scalar field is E perpendicular to the plane of incidence."""
    uniform = [(np.inf, np.ones((8, 8))), (0.2, 4.0 * np.ones((8, 8))), (np.inf, np.ones((8, 8)))]
    (px, py), why = _scalar_polarization(uniform, (0.3, 0.4))
    assert (px, py) == pytest.approx((-0.8, 0.6)) and why.startswith("s:")
    assert _scalar_polarization(uniform, (0.0, 0.0))[0] == (1 + 0j, 0j)


# ---------------------------------------------------------------------------------------------------
# a vector list carries its polarization in the incident field
# ---------------------------------------------------------------------------------------------------
@needs_fmmax
@pytest.mark.parametrize(("e_inc", "expected"), [((0.0, 1.0), "TE"), ((1.0, 0.0), "TM")])
def test_a_vector_list_is_solved_in_the_polarization_of_its_incident_field(e_inc, expected):
    """It used to be solved x-polarized whatever E_inc said."""
    rc = jno.rcwa(_vector("y", e_inc), orders=ORDERS, grid=GRID)
    assert rc.spec.polarization_from == "read from the vector incident field"
    assert _R(rc) == pytest.approx(R_TE if expected == "TE" else R_TM, abs=TOL)


def test_a_polarization_that_is_not_one_raises():
    from jno.rcwa import _as_jones

    for bad in ("z", (0.0, 0.0), (1.0,), "circular"):
        with pytest.raises(RcwaError, match="polarization"):
            _as_jones(bad)
