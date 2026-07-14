"""jno.rcwa with an ANISOTROPIC permittivity tensor ε̂ — inferred from inner(ε̂ @ u, v) and solved via
fmmax's ``eigensolve_anisotropic_media`` (5 component grids ε_xx, ε_xy, ε_yx, ε_yy, ε_zz).

The same curl-curl list, but the mass term carries a 3×3 ``MatrixView`` instead of a scalar. The front
door detects the tensor, samples its 5 components per layer, and routes to the anisotropic eigensolve —
reducing exactly to the isotropic path when ε̂ = ε·I.
"""

import importlib.util
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the fmmax solve OOMs on a small GPU at these orders

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jno  # noqa: E402
from jno.trace.views import MatrixView  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

inner, vec = jno.np.inner, jno.np.vector
K0 = 2 * jnp.pi
P, LZ = 0.6, 3.2
Z0, Z1 = 0.8, 1.15  # the anisotropic slab occupies this z-range


def _common(d):
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    nt, nb = d.variable("top", normals=True), d.variable("bottom", normals=True)
    cb = d.variable("bottom", split=True)
    tut, tvt = u.vector.cross(nt), v.vector.cross(nt)
    tub, tvb = u.vector.cross(nb), v.vector.cross(nb)
    einc = vec(1.0 + 0.0 * cb[0], 0.0 * cb[1], 0.0 * cb[2])  # x-polarised normal-incidence plane wave

    def face(n):
        cc = d.variable(n, split=True)
        return u.bind(x=cc[0], y=cc[1], z=cc[2])

    bcs = [
        1j * K0 * inner(tut, tvt),
        1j * K0 * inner(tub, tvb) + 2j * K0 * inner(einc, tvb),
        face("left") - face("right"),
        face("front") - face("back"),
    ]
    return (xi, yi, zi), (ui, vi), (cu, cv), bcs


def _slab(x, y, z, val):
    return jnp.where((z >= Z0) & (z < Z1), val, 1.0)


def _aniso_constraints(exx, eyy, ezz):
    """A uniform diagonal-anisotropic slab diag(exx, eyy, ezz) embedded in vacuum."""
    d = jno.domain(jno.Shape.box(0, 0, 0, P, P, LZ, size=0.2))
    (xi, yi, zi), (ui, vi), (cu, cv), bcs = _common(d)
    ex = jno.fn(lambda x, y, z: _slab(x, y, z, exx), [xi, yi, zi])
    ey = jno.fn(lambda x, y, z: _slab(x, y, z, eyy), [xi, yi, zi])
    ez = jno.fn(lambda x, y, z: _slab(x, y, z, ezz), [xi, yi, zi])
    eps = MatrixView(vec(ex, ey, ez).expr).from_diag()
    return [inner(cu, cv) - K0**2 * inner(eps @ ui, vi), *bcs]


def _scalar_constraints(e):
    """The isotropic reference: the same slab with scalar ε."""
    d = jno.domain(jno.Shape.box(0, 0, 0, P, P, LZ, size=0.2))
    (xi, yi, zi), (ui, vi), (cu, cv), bcs = _common(d)
    eps = jno.fn(lambda x, y, z: _slab(x, y, z, e), [xi, yi, zi])
    return [inner(cu, cv) - K0**2 * eps * inner(ui, vi), *bcs]


def _T(cons, R=False):
    sol = jno.rcwa(cons, orders=60, grid=24).solve()
    return float(sol.efficiency("R" if R else "T"))


@needs_fmmax
def test_isotropic_tensor_reduces_to_scalar():
    """ε̂ = ε·I must give exactly the scalar-ε result — the anisotropic path reduces correctly (same eps fed
    to fmmax, isotropic vs anisotropic eigensolve agree for a scalar-diagonal tensor)."""
    assert _T(_aniso_constraints(6.0, 6.0, 6.0)) == pytest.approx(_T(_scalar_constraints(6.0)), abs=1e-6)


@needs_fmmax
def test_in_plane_anisotropy_x_pol_sees_exx():
    """Decisive anisotropy check: for a biaxial slab diag(εxx, εyy, εzz), an x-polarised normal-incidence
    wave sees εxx — so T matches the scalar-εxx result and differs from scalar-εyy/εzz. (ε_zz does NOT act
    at normal incidence: the field is purely transverse, no E_z to couple to.)"""
    Tbi = _T(_aniso_constraints(8.0, 4.0, 6.0))  # x-pol sees εxx = 8
    assert Tbi == pytest.approx(_T(_scalar_constraints(8.0)), abs=1e-4)
    assert abs(Tbi - _T(_scalar_constraints(6.0))) > 1e-3  # genuinely anisotropic, not the scalar answer


@needs_fmmax
def test_anisotropic_conserves_energy():
    """A lossless anisotropic slab conserves energy: T + R ≈ 1."""
    cons = _aniso_constraints(8.0, 4.0, 6.0)
    assert _T(cons) + _T(cons, R=True) == pytest.approx(1.0, abs=5e-3)
