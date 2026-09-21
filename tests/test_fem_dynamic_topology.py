"""A reconnecting march must not recompile when only the CONNECTIVITY changes.

Under ``adapt=jno.solve.remesh(alpha=...)`` the driver re-decides which nodes form elements every step
(``fem_adapt.py:5639``) and rebuilds the whole problem whenever the cell set differs -- which re-traces
the weak form and re-compiles it. Measured on a two-drop coalescence at 951 dofs: a step that rebuilt
cost **30.7 s** against **0.05 s** for one that did not, and 6 of 7 rebuilds were Delaunay edge FLIPS
that changed 2-6 cells out of 568 while leaving every array shape identical (568 cells, 64 boundary
edges, 64 boundary nodes). Compilation dominated: 60.6 s of a 62 s run, 54.4 s of it in two
compilations of the marching scan.

Nothing about a flip requires a new program. ``segment_sum`` and ``.at[].add`` take *traced* index
arrays and recompile only on a SHAPE change, and the assembler already carries dynamic vertex
positions through ``args`` (``_apply_coord_params``). Connectivity should ride the same channel.

This test pins the contract: across a march whose triangulation flips, the number of XLA compilations
must not grow with the number of flips.
"""

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture(autouse=True)
def _no_persistent_compile_cache():
    """Count COMPILATIONS, not cache lookups.

    This test asserts on ``n_long - n_short``, which cancels anything both marches pay equally. The
    PERSISTENT on-disk cache breaks that symmetry: whether either march finds its programs already
    compiled depends on what ran earlier in the session, so the difference stops being a property of
    the code under test. Observed as an order-dependent failure -- green alone and in every subset,
    red once inside a ten-file run.
    """
    prev = jax.config.jax_enable_compilation_cache
    jax.config.update("jax_enable_compilation_cache", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_compilation_cache", prev)


class _CompileCounter:
    """Counts XLA compilations. Private JAX entry point -- asserted to exist, so a JAX upgrade that
    moves it fails loudly here rather than silently making the test vacuous."""

    def __init__(self):
        from jax._src import compiler

        assert hasattr(compiler, "backend_compile_and_load"), "JAX moved backend_compile_and_load"
        self._mod, self._orig, self.n = compiler, compiler.backend_compile_and_load, 0

    def __enter__(self):
        def counted(*a, **k):
            self.n += 1
            return self._orig(*a, **k)

        self._mod.backend_compile_and_load = counted
        return self

    def __exit__(self, *exc):
        self._mod.backend_compile_and_load = self._orig


def _coalescence(nsteps):
    """Two unequal drops that touch and merge: the smallest problem that actually flips cells."""
    n = jno.np
    inner, symgrad = n.inner, n.symgrad
    ddot = lambda a, b: inner(a, b, n_contract=2)  # noqa: E731
    RHO, ETA, SIGMA, NU = 1.0, 0.01, 10.0, 0.01
    RA, RB, GAP, H, DT, C_I = 0.20, 0.15, 0.02, 0.03, 1e-4, 36.0
    xa, xb = -(RA + GAP / 2), (RB + GAP / 2)
    d = (jno.shape.disk(xa, 0.0, RA, size=H) | jno.shape.disk(xb, 0.0, RB, size=H)).domain(
        time=(0.0, nsteps * DT, nsteps + 1)
    )
    # Stated explicitly, though a geometry term now INFERS it (`_fem_impl`): the operator is assembled
    # against a RUNTIME connectivity, so a reconnection that keeps every shape hands over a new
    # triangulation instead of rebuilding. Kept explicit here so the test pins the contract it is about
    # rather than riding a default that some later change could quietly flip.
    d._fem_want_dynamic_topology = True
    nnode = len(np.asarray(d.mesh.points))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xs, ys, ts, nx, ny = d.variable("boundary", normals=True, split=True)
    x0, y0, _t0 = d.variable("initial", split=True)
    B = dict(x=xi, y=yi, t=ti)
    ub, vv, pp, qq = u.bind(**B), v.bind(**B), p.bind(**B), q.bind(**B)
    vs = v.bind(x=xs, y=ys)
    D = lambda w: symgrad(w, [xi, yi])  # noqa: E731
    ndv = lambda f, i: nx * f.x[i] + ny * f.y[i]  # noqa: E731
    div_G = lambda f: f.x[0] + f.y[1] - (nx * ndv(f, 0) + ny * ndv(f, 1))  # noqa: E731
    c0, c1 = ub[0] - xi.d(ti), ub[1] - yi.d(ti)
    conv = lambda i: c0 * ub.x[i] + c1 * ub.y[i]  # noqa: E731
    G = d.cell_metric
    gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
    tau = jno.lag(((2.0 / DT) ** 2 + gG(ub) + C_I * NU**2 * ddot(G, G)) ** -0.5)
    r0, r1 = ub.t[0] + conv(0) + pp.x / RHO, ub.t[1] + conv(1) + pp.y / RHO
    momentum = (
        RHO * (ub.t[0] * vv[0] + ub.t[1] * vv[1])
        + RHO * (conv(0) * vv[0] + conv(1) * vv[1])
        + 2.0 * ETA * ddot(D(ub), D(vv))
        - pp * (vv.x[0] + vv.y[1])
        + tau * ((c0 * vv.x[0] + c1 * vv.y[0]) * r0 + (c0 * vv.x[1] + c1 * vv.y[1]) * r1)
    )
    continuity = -qq * (ub.x[0] + ub.y[1]) - tau * (qq.x * r0 + qq.y * r1)
    uf = u.bind(x=xs, y=ys).freeze(np.zeros((nnode, 2)))
    return jno.fem(
        [
            momentum,
            continuity,
            SIGMA * div_G(vs),
            u(x0, y0)[0] - 0.0,
            u(x0, y0)[1] - 0.0,
            xs.d(ts) - uf[0],
            ys.d(ts) - uf[1],
        ]
    )


def _run(nsteps):
    fem = _coalescence(nsteps)
    with _CompileCounter() as cc:
        traj = fem.solve(
            nonlinear=jno.solve.newton(direct=True), adapt=jno.solve.remesh(alpha=1.2, every=1)
        )
    flips = sum(1 for h in (getattr(fem, "adapt_history", []) or []) if h.get("remeshed"))
    return cc.n, flips, traj


def test_a_flip_does_not_recompile_the_march():
    """Twice the steps means more flips -- but the SHAPES never change, so the compile count must not
    track them. Today it does: every flip re-enters ``run_mesh_motion`` and re-compiles the scan."""
    # H = 0.03 puts 317 nodes on the pair; measured reconnections land at steps 0, 10, 16, 17, 19, 23,
    # 25, so 8 steps sees only the MERGE and 24 sees the merge plus five flips. A coarser mesh flips at
    # all (H = 0.05 gave 1 and 1) and makes the test vacuous.
    n_short, flips_short, _ = _run(8)
    n_long, flips_long, _ = _run(24)
    assert flips_long - flips_short >= 3, (
        f"no usable flip differential ({flips_long} vs {flips_short}): the test cannot see the effect"
    )
    extra_compiles = n_long - n_short
    extra_flips = flips_long - flips_short
    # MEASURED before the fix: compilations track flips exactly 1:1 -- 2 compilations for 1 flip, 6 for
    # 5, i.e. one per flip plus one for setup. Nothing about a flip changes a shape, so after the fix the
    # extra flips must cost NO extra program. One is allowed for slack, not five.
    assert extra_compiles <= 1, (
        f"{extra_flips} extra flips caused {extra_compiles} extra compilations "
        f"({n_short} at {flips_short} flips, {n_long} at {flips_long}) -- "
        "connectivity is still baked into the program"
    )


def test_a_geometry_term_infers_dynamic_topology():
    """A moving mesh should not have to ASK for runtime connectivity.

    ``coord.d(t) - velocity`` states that the mesh moves, and a moving mesh is the only situation in
    which the connectivity can change under a fixed node set. Requiring a separate
    ``.dynamic_topology()`` is asking the caller to say the same thing twice.

    It can be the default because it is free: measured on a geometry-term march over 41 frames,
    ``relocate`` (connectivity never changes) went 1.6 s -> 1.0 s and ``remesh(alpha=1.2, every=1)``
    3.5 s -> 0.9 s. Runtime connectivity is a gather through the index array the moved vertices
    already travel on.

    An explicit ``dynamic_topology(False)`` must still win, or the inference would be a trap.
    """
    import jno

    def _mk(geometry: bool, explicit=None):
        d = jno.shape.disk(0.0, 0.0, 0.5, size=0.2).domain(time=(0.0, 0.1, 3))
        if explicit is not None:
            d.dynamic_topology(explicit)
        u, v = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
        terms = [ui.t * vi + 0.05 * (ui.x * vi.x + ui.y * vi.y), u(ci[0], ci[1]) - 1.0]
        if geometry:
            terms.append(xi.d(ti) - 0.0)
        jno.fem(terms)
        return bool(getattr(d, "_fem_want_dynamic_topology", False))

    assert _mk(geometry=True), "a geometry term must infer runtime connectivity"
    assert not _mk(geometry=False), "a static mesh must be left alone -- nothing can change its cells"
    assert not _mk(geometry=True, explicit=False), "an explicit opt-out must beat the inference"
    assert _mk(geometry=True, explicit=True)
