"""The per-node stencil of a matrix-free operator on a lattice, read off by colouring
(`jno/utils/solver/lattice.py`). Oracles: the analytic coefficients of a known stencil, and the operator's
own action on random vectors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.lattice import apply_stencil, offsets, probe


@pytest.fixture(autouse=True)
def _x64():
    """Float64 per test, restored afterwards: set at module scope it ran at import, for every module in the
    selection, and could not be undone (tests/test_x64_isolation.py)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _laplacian_matvec(shape, h):
    """-Δ on the interior with identity rows on the boundary ring, written directly as slices."""
    dim = len(shape)

    def mv(v):
        u = v.reshape((1,) + shape)
        core = (slice(None),) + tuple(slice(1, -1) for _ in range(dim))
        out = jnp.zeros_like(u)
        acc = jnp.zeros_like(u[core])
        for a in range(dim):
            plus = (slice(None),) + tuple(slice(2, None) if b == a else slice(1, -1) for b in range(dim))
            minus = (slice(None),) + tuple(slice(None, -2) if b == a else slice(1, -1) for b in range(dim))
            acc = acc + (2.0 * u[core] - u[plus] - u[minus]) / h**2
        return _with_ring(out.at[core].set(acc), u, shape)

    return mv


def _ring(shape):
    m = np.zeros(shape, bool)
    for a, n in enumerate(shape):
        sl = [slice(None)] * len(shape)
        sl[a] = 0
        m[tuple(sl)] = True
        sl[a] = n - 1
        m[tuple(sl)] = True
    return m


def _with_ring(out, u, shape):
    ring = jnp.asarray(_ring(shape))[None]
    return jnp.where(ring, u, out).reshape(-1)


def test_the_probe_reads_the_analytic_five_point_stencil():
    """-Δ_h with identity boundary rows: the probe returns exactly 4/h² on the centre, -1/h² on each
    neighbour, 1 on a boundary row, and finds the compact window."""
    shape, h = (7, 7), 0.25
    window, S = probe(_laplacian_matvec(shape, h), shape)
    assert window == (-1, 1)
    offs = offsets(*window, 2)
    centre, east = offs.index((0, 0)), offs.index((1, 0))
    assert S.shape == (1, 1) + shape + (9,)
    assert S[0, 0, 3, 3, centre] == pytest.approx(4.0 / h**2)
    assert S[0, 0, 3, 3, east] == pytest.approx(-1.0 / h**2)
    assert S[0, 0, 0, 3, centre] == pytest.approx(1.0)  # an identity boundary row
    assert np.abs(np.asarray(S[0, 0, 0, 3, :])).sum() == pytest.approx(1.0)


def _fdm_matvec(prob):
    """The problem's Jacobian as a matvec on the blocked DOF vector."""
    residual = prob._steady_residual()
    z = jnp.zeros(prob._Ntot)
    return lambda v: jax.jvp(residual, (z,), (v,))[1]


def _grid(n=8, dim=2, terms=None, scheme=None, size=None):
    shape = jno.shape.rect(0, 0, 1, 1, size=1 / n) if dim == 2 else jno.shape.box(0, 0, 0, 1, 1, 1, size=1 / n)
    d = shape.structured().domain()
    coords = d.variable("interior", split=True)
    bnd = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(**dict(zip("xyz", coords[:dim])), **({"scheme": scheme} if scheme else {}))
    return d, jno.fdm(terms(u, ui, coords, bnd))


@pytest.mark.parametrize(
    "name",
    ["poisson", "variable_coefficient", "advection", "reaction", "flux_boundary"],
)
def test_the_probe_reproduces_every_grid_operator(name):
    """Each operator's own action on a random vector is the oracle: the probed stencil must reproduce it."""
    builders = {
        "poisson": lambda u, ui, c, b: [-ui.xx - ui.yy - 1.0, u(b[0], b[1]) - 0.0],
        "variable_coefficient": lambda u, ui, c, b: [
            -((1.0 + c[0] ** 2) * ui.x).x - ((1.0 + c[1]) * ui.y).y - 1.0,
            u(b[0], b[1]) - 0.0,
        ],
        "advection": lambda u, ui, c, b: [-0.01 * (ui.xx + ui.yy) + 3.0 * ui.x - ui.y - 1.0, u(b[0], b[1]) - 0.0],
        "reaction": lambda u, ui, c, b: [-ui.xx - ui.yy + 50.0 * ui - 1.0, u(b[0], b[1]) - 0.0],
    }
    if name == "flux_boundary":

        def builder(u, ui, c, b):
            d = _flux_domain[0]
            xl, yl, _ = d.variable("left", split=True)
            n = d.variable("left", normals=True)
            ul = u.bind(x=xl, y=yl)
            return [-ui.xx - ui.yy - 1.0, ul.d(n) - 0.5, u(b[0], b[1]) - 0.0]

        _flux_domain = []
        d = jno.shape.rect(0, 0, 1, 1, size=1 / 8).structured().domain()
        _flux_domain.append(d)
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        (xr, yr, _), (xt, yt, _), (xo, yo, _) = (d.variable(r, split=True) for r in ("right", "top", "bottom"))
        nl = d.variable("left", normals=True)
        xl, yl, _ = d.variable("left", split=True)
        u = d.unknown()
        ui, ul = u.bind(x=x, y=y), u.bind(x=xl, y=yl)
        prob = jno.fdm(
            [
                -ui.xx - ui.yy - 1.0,
                ul.d(nl) - 0.5,
                u(xr, yr) - 0.0,
                u(xt, yt) - 0.0,
                u(xo, yo) - 0.0,
            ]
        )
    else:
        _, prob = _grid(terms=builders[name])
    shape = tuple(prob.domain.mesh_connectivity["grid"]["shape"])
    mv = _fdm_matvec(prob)
    window, S = probe(mv, shape)
    rng = np.random.default_rng(3)
    v = jnp.asarray(rng.standard_normal(prob._Ntot))
    got = apply_stencil(S, window, v.reshape(1, *shape)).reshape(-1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(mv(v)), rtol=1e-10, atol=1e-10)


def test_the_probe_widens_its_window_for_a_fourth_order_stencil():
    """`jno.fd(order=4)` reaches two nodes each way in the interior and further at the boundary: the window
    is found by measurement, not assumed."""
    _, prob4 = _grid(terms=lambda u, ui, c, b: [-ui.xx - ui.yy - 1.0, u(b[0], b[1]) - 0.0], scheme=jno.fd(order=4))
    shape = tuple(prob4.domain.mesh_connectivity["grid"]["shape"])
    mv = _fdm_matvec(prob4)
    window, S = probe(mv, shape)
    assert window[1] - window[0] + 1 >= 5
    rng = np.random.default_rng(5)
    v = jnp.asarray(rng.standard_normal(prob4._Ntot))
    got = apply_stencil(S, window, v.reshape(1, *shape)).reshape(-1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(mv(v)), rtol=1e-9, atol=1e-9)


def test_the_probe_handles_a_coupled_system_and_three_dimensions():
    """Two fields: the stencil carries every field pair. And the same in 3-D."""
    d = jno.shape.rect(0, 0, 1, 1, size=1 / 6).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.unknown(), d.unknown()
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    prob = jno.fdm([-ui.xx - ui.yy + 2.0 * vi - 1.0, -vi.xx - vi.yy - 0.5 * ui - 1.0, u(xb, yb) - 0.0, v(xb, yb) - 0.0])
    shape = tuple(d.mesh_connectivity["grid"]["shape"])
    mv = _fdm_matvec(prob)
    window, S = probe(mv, shape, nf=2)
    assert S.shape == (2, 2) + shape + (9,)
    rng = np.random.default_rng(7)
    vec = jnp.asarray(rng.standard_normal(prob._Ntot))
    got = apply_stencil(S, window, vec.reshape(2, *shape)).reshape(-1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(mv(vec)), rtol=1e-10, atol=1e-10)

    _, p3 = _grid(n=5, dim=3, terms=lambda u, ui, c, b: [-ui.xx - ui.yy - ui.zz - 1.0, u(b[0], b[1], b[2]) - 0.0])
    shape3 = tuple(p3.domain.mesh_connectivity["grid"]["shape"])
    mv3 = _fdm_matvec(p3)
    w3, S3 = probe(mv3, shape3)
    v3 = jnp.asarray(np.random.default_rng(9).standard_normal(p3._Ntot))
    got3 = apply_stencil(S3, w3, v3.reshape(1, *shape3)).reshape(-1)
    np.testing.assert_allclose(np.asarray(got3), np.asarray(mv3(v3)), rtol=1e-10, atol=1e-10)


def test_the_colours_run_inside_one_compiled_program():
    """The whole colour pass is one ``fori_loop``, so the operator is TRACED once (plus once for the check)
    however many colours there are. Dispatching a program per colour per level cost 12 s of multigrid setup
    at 1M nodes against 0.3 s of arithmetic."""
    shape = (7, 7)
    traces = []
    base = _laplacian_matvec(shape, 0.25)

    def counted(v):
        traces.append(1)
        return base(v)

    window, S = probe(counted, shape)
    assert window == (-1, 1)
    assert len(traces) == 2  # one fold over the 9 colours, one verification matvec


def test_a_global_operator_has_no_stencil_and_raises():
    """A spectral derivative couples every node on an axis, so no window reproduces it."""
    shape = (8, 8)

    def dense(v):
        u = v.reshape(shape)
        return (jnp.cumsum(u, axis=0) + u).reshape(-1)

    with pytest.raises(ValueError, match="no window"):
        probe(dense, shape)


def test_a_periodic_axis_wraps_whatever_its_length():
    """With a periodic axis the window wraps. Where the window's width does not divide the axis -- 8 nodes
    with a three-wide stencil -- plain ``i mod 3`` colouring would give a row two neighbours of one colour
    across the seam and fold two coefficients into one, so the nodes at the seam get colours of their own."""
    for n0 in (9, 8):  # 9 divides by 3, 8 does not
        shape, h = (n0, 7), 0.125
        base = _laplacian_matvec(shape, h)

        def wrapped(v, shape=shape, h=h, base=base):  # -Δ with the first axis periodic: add the wrap couplings
            u = v.reshape((1,) + shape)
            out = base(v).reshape((1,) + shape)
            edge = jnp.zeros_like(u).at[:, 0].set(-u[:, -1] / h**2).at[:, -1].set(-u[:, 0] / h**2)
            return (out + edge).reshape(-1)

        window, S = probe(wrapped, shape, periodic=(True, False), window=(-1, 1))
        v = jnp.asarray(np.random.default_rng(11).standard_normal(int(np.prod(shape))))
        got = apply_stencil(S, window, v.reshape(1, *shape), periodic=(True, False)).reshape(-1)
        np.testing.assert_allclose(np.asarray(got), np.asarray(wrapped(v)), rtol=1e-10, atol=1e-10)
