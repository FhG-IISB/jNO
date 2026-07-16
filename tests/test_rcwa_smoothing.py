"""Subpixel permittivity smoothing in jno.rcwa (``smoothing=k``).

Each RCWA pixel is supersampled ``k×k`` and the permittivity area-averaged, anti-aliasing material
boundaries. The point is inverse design: a point-sampled ε staircases as a boundary sweeps (the rasterized
fill is piecewise-constant, jumping only when the edge crosses a grid line), which makes the gradient
w.r.t. a boundary-moving parameter jumpy; area-averaging makes the fill — and the gradient — vary smoothly.
Default is off (``smoothing=1``) so existing results are unchanged; it is differentiable and opt-in.
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
from jno.rcwa import RcwaError, _sample_grid_direct  # noqa: E402
from jno.trace import FunctionCall  # noqa: E402
from jno.trace_evaluator import TraceEvaluator  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

K0 = 2 * jnp.pi
P, LZ = 1.1, 3.2


def _pillar_cons(d, hw=0.22, eps_val=1.5):
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
    eps = jno.fn(
        lambda x, y, z: jnp.where(
            (jnp.abs(x - P / 2) < hw) & (jnp.abs(y - P / 2) < hw) & (z >= 0.8) & (z < 1.15), eps_val, 1.0
        ),
        [xi, yi, zi],
    )
    return [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ul - ur,
        uf - ub,
    ]


def _dom():
    d = jno.domain(jno.Shape.box(0, 0, 0, P, P, LZ, size=0.2))
    e = 1e-6  # explicit face predicates -> multidirectional (x AND y) periodicity needs tagged faces
    d.tag("left", lambda x, y, z: x < e)
    d.tag("right", lambda x, y, z: x > P - e)
    d.tag("front", lambda x, y, z: y < e)
    d.tag("back", lambda x, y, z: y > P - e)
    d.tag("bottom", lambda x, y, z: z < e)
    d.tag("top", lambda x, y, z: z > LZ - e)
    return d


@needs_fmmax
def test_smoothing_off_is_default():
    """smoothing=1 (the default) must give EXACTLY the same result as omitting it — subpixel smoothing is
    strictly opt-in and never silently changes an existing problem."""
    cons = _pillar_cons(_dom())
    t_default = float(jno.rcwa(cons, orders=80, grid=48).solve().efficiency("T"))
    t_one = float(jno.rcwa(_pillar_cons(_dom()), orders=80, grid=48, smoothing=1).solve().efficiency("T"))
    assert t_one == pytest.approx(t_default, abs=1e-12)


def test_smoothing_reduces_rasterization_staircasing():
    """The core effect (no solve needed): as a hard pillar's half-width sweeps, the *point-sampled* rasterized
    permittivity staircases (piecewise-constant, big jumps when the edge crosses a grid line); area-averaging
    (sub>1) shrinks those jumps, so the fill — and any boundary gradient — varies smoothly."""
    xi, yi, zi, _ = _dom().variable("interior", split=True)

    def fill(hw, sub):
        eps = jno.fn(
            lambda x, y, z, hw=hw: jnp.where(
                (jnp.abs(x - P / 2) < hw) & (jnp.abs(y - P / 2) < hw) & (z >= 0.8) & (z < 1.15), 6.0, 1.0
            ),
            [xi, yi, zi],
        )
        C, _ = _sample_grid_direct(eps, 32, 48, (P, P), (0.0, LZ), {}, sub=sub)
        return float(np.real(C).sum())

    hws = np.linspace(0.20, 0.26, 25)
    jump1 = np.abs(np.diff([fill(h, 1) for h in hws])).max()
    jump4 = np.abs(np.diff([fill(h, 4) for h in hws])).max()
    assert jump4 < 0.6 * jump1  # subpixel averaging markedly reduces the worst staircase jump


@needs_fmmax
def test_smoothing_solve_conserves_energy():
    """A smoothed solve is still physical: a lossless pillar conserves energy (T + R ≈ 1)."""
    sol = jno.rcwa(_pillar_cons(_dom()), orders=80, grid=48, smoothing=2).solve()
    assert float(sol.efficiency("T")) + float(sol.efficiency("R")) == pytest.approx(1.0, abs=5e-3)


@needs_fmmax
def test_smoothing_preserves_differentiability():
    """A design parameter still flows through a smoothed solve: rc.solve() is a trace node and jax.grad of T
    w.r.t. the scatterer ε matches a finite difference (subpixel averaging is a plain mean — differentiable)."""
    d = _dom()
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
    ep = jno.np.parameter((), name="ep").initialize(jax.nn.initializers.constant(1.5))
    ind = jno.fn(
        lambda x, y, z: jnp.where(
            (jnp.abs(x - P / 2) < 0.22) & (jnp.abs(y - P / 2) < 0.22) & (z >= 0.8) & (z < 1.15), 1.0, 0.0
        ),
        [xi, yi, zi],
    )
    eps = 1.0 + ind * (ep - 1.0)
    cons = [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ul - ur,
        uf - ub,
    ]
    rc = jno.rcwa(cons, orders=80, grid=48, smoothing=2, params={"ep": 1.5})
    node = rc.solve().efficiency("T")
    assert isinstance(node, FunctionCall)

    def T(v):
        mod = eqx.tree_at(lambda m: m.value, ep.model.module, jnp.asarray(v))
        return TraceEvaluator({ep.model.layer_id: mod}).evaluate(node)

    g = float(jax.grad(T)(1.5))
    h = 1e-2
    fd = (float(T(1.5 + h)) - float(T(1.5 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=5e-2)


@needs_fmmax
def test_smoothing_invalid_raises():
    """smoothing must be a positive integer."""
    with pytest.raises(RcwaError, match="smoothing must be a positive integer"):
        jno.rcwa(_pillar_cons(_dom()), orders=40, grid=40, smoothing=0)
