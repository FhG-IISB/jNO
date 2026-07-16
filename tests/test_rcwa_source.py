"""jno.rcwa with an INTERNAL SOURCE (dipole / Gaussian emitter) inferred from a volume forcing term.

An emitter is authored exactly as in the FEM/PINN world — a forcing term in the residual: ``- src·v``
(scalar monopole) or ``- inner(J, v)`` (vector dipole). The front door detects it (trial-free, test-present
volume summand), localizes it (centroid → point vs Gaussian; which z-layer), and routes it to fmmax's
``amplitudes_for_source`` (splitting the stack at the source plane). Readouts: power radiated up / down and
the extraction fraction — the LED / Purcell / emitter observables. Differentiable in the source amplitude
and the design permittivity.
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
from jno.rcwa import Rcwa, RcwaError  # noqa: E402
from jno.trace import FunctionCall  # noqa: E402
from jno.trace_evaluator import TraceEvaluator  # noqa: E402
from jno.utils.solver.fem_adapt import _domain_from_arrays  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

K0 = 2 * jnp.pi
LZ = 3.0
inner, vec = jno.np.inner, jno.np.vector


# ----------------------------- engine-level (explicit layers, no meshing) -----------------------------
def _engine_emit(eps_sub, orient=(1, 0, 0), amp=1.0, kind="delta", fwhm=0.2):
    layers = [(float("inf"), 1.0), (1.0, 1.0), (float("inf"), eps_sub)]  # super vac / vac slab / substrate
    rc = Rcwa(layers, period=(1.0, 1.0), orders=60, wavelength=1.0, k_in=(1e-3, 1e-3), assume_periodic=True)
    src = dict(loc=[[0.5, 0.5]], layer=1, t_upper=0.5, t_lower=0.5, kind=kind, fwhm=fwhm, orient=orient, amp=amp)
    s = rc.solve(source=src)
    return float(s.power("up")), float(s.power("down")), float(s.extraction("up"))


@needs_fmmax
def test_engine_vacuum_emits_symmetrically():
    """An x-dipole at the centre of a vacuum slab radiates equally up and down (symmetric environment)."""
    up, down, extr = _engine_emit(1.0)
    assert up == pytest.approx(down, rel=1e-6)
    assert extr == pytest.approx(0.5, abs=1e-6)


@needs_fmmax
def test_engine_orientation_matters():
    """A z-oriented dipole radiates differently from an in-plane (x) dipole — orientation is a real knob."""
    up_x, _, _ = _engine_emit(1.0, orient=(1, 0, 0))
    up_z, _, _ = _engine_emit(1.0, orient=(0, 0, 1))
    assert abs(up_z - up_x) > 1e-2


@needs_fmmax
def test_engine_power_scales_with_amplitude_squared():
    """Emitted power is quadratic in the source amplitude (|amp·source|²) — the differentiable scaling."""
    up1, _, _ = _engine_emit(1.0, amp=1.0)
    up2, _, _ = _engine_emit(1.0, amp=2.0)
    assert up2 / up1 == pytest.approx(4.0, rel=1e-5)


# ----------------------------- front door (author the emitter in the weak form) -----------------------------
def _sbox(Lx, dx=0.12, dz=0.2):
    xs = np.linspace(0, Lx, int(round(Lx / dx)) + 1)
    zs = np.linspace(0, LZ, int(round(LZ / dz)) + 1)
    ny, nz = len(xs), len(zs)
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
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
    d = _domain_from_arrays(
        jno.domain.cube(x_range=(0, Lx), y_range=(0, Lx), z_range=(0, LZ), mesh_size=0.2), P, tets, uq[cnt == 1], copy=True
    )
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


def _emitter_cons(d, Lx, eps_sub=1.0, amp=None):
    """Scalar Helmholtz on the supercell with an internal Gaussian emitter in the eps=6 slab (z∈[1,2]);
    substrate eps_sub below, vacuum above. ``amp`` (an optional parameter) scales the source amplitude."""
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
    eps = jno.fn(lambda x, y, z: jnp.where((z >= 1.0) & (z < 2.0), 6.0, jnp.where(z < 1.0, eps_sub, 1.0)), [xi, yi, zi])
    A = 1.0 if amp is None else amp
    profile = jno.np.exp(-(((xi - Lx / 2) ** 2 + (yi - Lx / 2) ** 2 + (zi - 1.5) ** 2) / (2 * 0.12**2)))
    vol = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi) - A * profile * vi
    return [vol, -(1j * K0 * utp) * vtp, -(1j * K0 * ubt) * vbt, ul - ur, uf - ub]


@needs_fmmax
def test_frontdoor_emitter_readouts_finite_and_bounded():
    """Author a Gaussian emitter as a `- src·v` volume forcing: the front door infers it and returns finite
    up/down power with an extraction fraction in (0, 1)."""
    Lx = 1.4
    rc = jno.rcwa(_emitter_cons(_sbox(Lx), Lx, eps_sub=1.0), orders=50, grid=28)
    assert rc.spec.source is not None and rc.spec.source["kind"] == "gaussian"
    s = rc.solve()
    up, down, extr = float(s.power("up")), float(s.power("down")), float(s.extraction("up"))
    assert np.isfinite(up) and np.isfinite(down) and up > 0 and down > 0
    assert 0.0 < extr < 1.0
    assert float(s.power("total")) == pytest.approx(up + down, rel=1e-6)


@needs_fmmax
def test_frontdoor_high_index_substrate_biases_emission_down():
    """A higher-index substrate pulls emission downward — extraction into the top (up) drops. Classic
    substrate-emission physics, inferred from the traced problem."""
    Lx = 1.4
    extr_vac = float(jno.rcwa(_emitter_cons(_sbox(Lx), Lx, eps_sub=1.0), orders=50, grid=28).solve().extraction("up"))
    extr_hi = float(jno.rcwa(_emitter_cons(_sbox(Lx), Lx, eps_sub=9.0), orders=50, grid=28).solve().extraction("up"))
    assert extr_hi < extr_vac - 1e-2


@needs_fmmax
def test_source_amplitude_is_differentiable():
    """A trainable source amplitude (jno.np.parameter) flows through the emission solve: rc.solve().power is a
    trace node over it, and jax.grad matches a finite difference (power ∝ amp²)."""
    Lx = 1.4
    amp = jno.np.parameter((), name="amp").initialize(jax.nn.initializers.constant(1.0))
    rc = jno.rcwa(_emitter_cons(_sbox(Lx), Lx, eps_sub=2.5, amp=amp), orders=40, grid=28, params={"amp": 1.0})
    node = rc.solve().power("up")  # NO solve args -> trace node over the amplitude
    assert isinstance(node, FunctionCall)

    def P(v):
        mod = eqx.tree_at(lambda m: m.value, amp.model.module, jnp.asarray(v))
        return TraceEvaluator({amp.model.layer_id: mod}).evaluate(node)

    g = float(jax.grad(P)(1.3))
    h = 1e-3
    fd = (float(P(1.3 + h)) - float(P(1.3 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=2e-3)
    assert abs(g) > 1e-6  # the amplitude genuinely drives the power


@needs_fmmax
def test_source_width_is_differentiable():
    """The emitter's Gaussian WIDTH (a jno.np.parameter) flows through the localization + emission solve: a
    wider soft source couples less into the propagating modes, and jax.grad of the emitted power w.r.t. the
    width matches a finite difference."""
    Lx = 1.4
    d = _sbox(Lx)
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
    eps = jno.fn(lambda x, y, z: jnp.where((z >= 1.0) & (z < 2.0), 6.0, jnp.where(z < 1.0, 3.0, 1.0)), [xi, yi, zi])
    w = jno.np.parameter((), name="w").initialize(jax.nn.initializers.constant(0.14))
    src = jno.np.exp(-(((xi - Lx / 2) ** 2 + (yi - Lx / 2) ** 2 + (zi - 1.5) ** 2) / (2 * w**2)))
    vol = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi) - src * vi
    rc = jno.rcwa(
        [vol, -(1j * K0 * utp) * vtp, -(1j * K0 * ubt) * vbt, ul - ur, uf - ub], orders=40, grid=28, params={"w": 0.14}
    )
    node = rc.solve().power("up")
    assert isinstance(node, FunctionCall)

    def P(v):
        mod = eqx.tree_at(lambda m: m.value, w.model.module, jnp.asarray(v))
        return TraceEvaluator({w.model.layer_id: mod}).evaluate(node)

    g = float(jax.grad(P)(0.14))
    h = 2e-3
    fd = (float(P(0.14 + h)) - float(P(0.14 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=5e-3)
    assert abs(g) > 1e-2  # the width genuinely changes the emission


@needs_fmmax
def test_source_outside_finite_layer_raises():
    """A source that lands in a semi-infinite ambient (no permittivity contrast to define a finite slab) is
    rejected with a concrete fix."""
    Lx = 1.4
    d = _sbox(Lx)
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
    eps = jno.fn(lambda x, y, z: 1.0 + 0.0 * z, [xi, yi, zi])  # uniform vacuum -> no finite layer anywhere
    src = jno.np.exp(-(((xi - Lx / 2) ** 2 + (yi - Lx / 2) ** 2 + (zi - 1.5) ** 2) / (2 * 0.12**2)))
    vol = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi) - src * vi
    cons = [vol, -(1j * K0 * utp) * vtp, -(1j * K0 * ubt) * vbt, ul - ur, uf - ub]
    with pytest.raises(RcwaError, match="not inside a finite"):
        jno.rcwa(cons, orders=40, grid=28)
