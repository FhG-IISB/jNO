"""jno.rcwa aerial imaging — sol.aerial(): the partially-coherent litho image on top of the rigorous mask.

The RCWA solve gives the mask's diffraction spectrum; ``sol.aerial(NA, source, …)`` projects it through a
lens (NA + pupil) and sums over the illumination (Abbe) to form the wafer-plane intensity. Validated against
the imaging limits (open frame → uniform; partial coherence reduces contrast) and differentiable in the mask
design (ILT) — jax.grad of an image functional matches finite difference.
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
from jno.trace import FunctionCall  # noqa: E402
from jno.trace_evaluator import TraceEvaluator  # noqa: E402
from jno.utils.solver.fem_adapt import _domain_from_arrays  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

K0 = 2 * jnp.pi
LZ = 3.0
P = 2.8  # mask period (non-integer × wavelength -> avoids a Rayleigh anomaly)


def _sbox(dx=0.07, ny=5):
    xs = np.linspace(0, P, int(round(P / dx)) + 1)
    ys = np.linspace(0, P, ny)
    zs = np.linspace(0, LZ, int(round(LZ / 0.2)) + 1)
    nx, nyy, nz = len(xs), len(ys), len(zs)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Pt = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    vid = lambda i, j, k: (i * nyy + j) * nz + k  # noqa: E731
    CUBE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    TE = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    tets = []
    for i in range(nx - 1):
        for j in range(nyy - 1):
            for k in range(nz - 1):
                c = [vid(i + a, j + b, k + cc) for (a, b, cc) in CUBE]
                tets += [[c[t[0]], c[t[1]], c[t[2]], c[t[3]]] for t in TE]
    tets = np.asarray(tets)
    F = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    uq, cnt = np.unique(np.sort(F, 1), axis=0, return_counts=True)
    d = _domain_from_arrays(
        jno.domain.cube(x_range=(0, P), y_range=(0, P), z_range=(0, LZ), mesh_size=0.2), Pt, tets, uq[cnt == 1], copy=True
    )
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
    return d


def _cons(eps_fn, amp=None):
    d = _sbox()
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
    eps = jno.fn(eps_fn, [xi, yi, zi])
    return [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ul - ur,
        uf - ub,
    ]


def _line(x, y, z):  # a 1-D line/space grating (lines in x, uniform in y), eps=2 lines
    return jnp.where((jnp.abs(x - P / 2) < P * 0.28) & (z >= 1.0) & (z < 1.2), 2.0, 1.0)


def _open(x, y, z):  # open frame (uniform vacuum -> a clear mask)
    return 1.0 + 0.0 * z


def _contrast(img):
    prof = np.asarray(img).mean(1)  # average over y (grating uniform in y)
    return (prof.max() - prof.min()) / (prof.max() + prof.min())


@needs_fmmax
def test_open_frame_gives_uniform_image():
    """A clear (open-frame) mask images to a UNIFORM aerial image — no structure to diffract."""
    img = np.asarray(jno.rcwa(_cons(_open), orders=40, grid=48).solve().aerial(NA=0.6, source=0.5))
    assert (img.max() - img.min()) / (img.mean() + 1e-12) < 1e-3


@needs_fmmax
def test_partial_coherence_reduces_contrast():
    """Increasing the partial-coherence factor σ (a larger conventional source) reduces the image contrast —
    the defining behaviour of partially-coherent imaging."""
    sol = jno.rcwa(_cons(_line), orders=60, grid=56).solve()
    c_lo = _contrast(sol.aerial(NA=0.6, source=0.2))
    c_hi = _contrast(sol.aerial(NA=0.6, source=0.9))
    assert c_lo > c_hi > 0.0  # contrast drops with σ, but the grating is still resolved


@needs_fmmax
def test_source_shapes_run():
    """Conventional (float), annular (tuple), and a raw pupil-weight array all form a valid image."""
    sol = jno.rcwa(_cons(_line), orders=40, grid=48).solve()
    conv = sol.aerial(NA=0.6, source=0.5)
    ann = sol.aerial(NA=0.6, source=(0.4, 0.8))
    arr = sol.aerial(NA=0.6, source=np.ones((15, 15)))  # freeform pupil
    for im in (conv, ann, arr):
        im = np.asarray(im)
        assert im.shape == (128, 128) and np.all(np.isfinite(im)) and im.min() >= 0.0


@needs_fmmax
def test_aerial_is_differentiable_ilt():
    """The aerial image is differentiable in the mask design (inverse lithography): rc.solve().aerial() is a
    trace node, and jax.grad of an image functional w.r.t. a mask parameter matches a finite difference —
    the gradient flows through the imaging AND the RCWA mask solve."""
    d = _sbox()
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
    ep = jno.np.parameter((), name="ep").initialize(jax.nn.initializers.constant(2.0))
    ind = jno.fn(
        lambda x, y, z: jnp.where((jnp.abs(x - P / 2) < P * 0.28) & (z >= 1.0) & (z < 1.2), 1.0, 0.0), [xi, yi, zi]
    )
    eps = 1.0 + ind * (ep - 1.0)
    cons = [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ul - ur,
        uf - ub,
    ]
    rc = jno.rcwa(cons, orders=60, grid=56, params={"ep": 2.0})
    node = rc.solve().aerial(NA=0.6, source=0.5)
    assert isinstance(node, FunctionCall)

    def loss(v):
        mod = eqx.tree_at(lambda m: m.value, ep.model.module, jnp.asarray(v))
        return jnp.mean(TraceEvaluator({ep.model.layer_id: mod}).evaluate(node) ** 2)

    g = float(jax.grad(loss)(2.0))
    h = 1e-2
    fd = (float(loss(2.0 + h)) - float(loss(2.0 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=3e-3)
    assert abs(g) > 1e-3
