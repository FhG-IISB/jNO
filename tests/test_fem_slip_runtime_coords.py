"""Slip normals that follow runtime (``.trainable()``) coordinates.

A structured unit square whose TOP is tilted at solve time (y -> y (1 + a x)) through a trainable y
coordinate, with ``n·u = 0`` on that top. Oracles:

* a FRESH build of the tilted mesh (same lattice, same numbering) solves to the same answer;
* the discrete flux ``Σ u_i·N_i`` through the tilted surface -- with ``N_i`` from the reference (numpy)
  normal code on the tilted points, independent of the runtime JAX one -- is zero to round-off;
* before this, the build-time normals were used and the answer was silently different, so the test also
  checks the runtime answer is NOT the flat-normal one.
"""

from __future__ import annotations

import pytest

pytest.importorskip("meshio")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jno  # noqa: E402

N = 8
TILT = 0.35


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _lattice(tilt):
    xs = np.linspace(0.0, 1.0, N + 1)
    P = np.array([[x, y * (1.0 + tilt * x)] for y in xs for x in xs])  # node (i, j) = j*(N+1) + i
    idx = lambda i, j: j * (N + 1) + i  # noqa: E731
    tris = []
    for j in range(N):
        for i in range(N):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [[a, b, c], [a, c, d]]
    edges, tags = [], []
    for i in range(N):
        edges += [[idx(i, 0), idx(i + 1, 0)], [idx(i, N), idx(i + 1, N)]]
        tags += ["bottom", "top"]
    for j in range(N):
        edges += [[idx(0, j), idx(0, j + 1)], [idx(N, j), idx(N, j + 1)]]
        tags += ["left", "right"]
    return P, np.asarray(tris), np.asarray(edges), np.asarray(tags)


def _builder(tilt):
    import meshio

    def build(_geo):
        P, tris, edges, tags = _lattice(tilt)
        sets = {t: [np.flatnonzero(tags == t), np.empty(0, dtype=np.int64)] for t in ("bottom", "top", "left", "right")}
        sets["boundary"] = [np.arange(len(edges)), np.empty(0, dtype=np.int64)]
        sets["interior"] = [np.empty(0, dtype=np.int64), np.arange(len(tris))]
        return meshio.Mesh(P, [("line", edges), ("triangle", tris)], cell_sets=sets), 2, 1.0 / N

    return build


def _problem(tilt, trainable, order=2):
    d = jno.domain(_builder(tilt), compute_mesh_connectivity=True)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=order)
    xi, yi, _ = d.variable("interior", split=True)
    yp = yi.trainable() if trainable else None
    ct = d.variable("top", normals=True, split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    grad, inner = jno.np.grad, jno.np.inner
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    k = 1.0 + 0.5 * (ui[0] * ui[0] + ui[1] * ui[1])  # nonlinear: a solution-dependent stiffness
    f0, f1 = 1.0 + xi, 0.5 + yi  # a load with a normal component, so n·u = 0 is doing work
    weak = k * inner(grad(u, [xi, yi]), grad(v, [xi, yi]), n_contract=2) - (f0 * vi[0] + f1 * vi[1])
    ut = u(ct[0], ct[1])
    fem = jno.fem([weak, ct[-2] * ut[0] + ct[-1] * ut[1] - 0.0, u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0])
    return d, fem, yp


def _pname(expr):
    from jno.utils.solver.parametric_helpers import _collect_runtime_parameter_exprs

    named = {}
    _collect_runtime_parameter_exprs(expr, named)
    return next(iter(named))


NL = dict(nonlinear=jno.solve.newton(direct=True, rtol=1e-12, atol=1e-12), linear=jno.solve.lu(backend="host"))


@pytest.mark.parametrize("order", [1, 2])
def test_runtime_tilt_matches_a_fresh_build_and_conserves_flux(order):
    # fresh build ON the tilted mesh -- the oracle
    d1, fem1, _ = _problem(TILT, trainable=False, order=order)
    u_fresh = np.asarray(fem1.solve(**NL)).reshape(-1)

    # the flat build, tilted at solve time through the trainable y
    d0, fem0, yp = _problem(0.0, trainable=True, order=order)
    ids = np.asarray(d0._trainable_coords[0]["ids"])
    y_tilted = _lattice(TILT)[0][ids, 1]
    u_rt = np.asarray(fem0.solve(**{_pname(yp): y_tilted}, **NL)).reshape(-1)
    rel = np.linalg.norm(u_rt - u_fresh) / np.linalg.norm(u_fresh)
    assert rel < 1e-9, f"runtime-tilted solve differs from a fresh build of the tilted mesh: {rel:.2e}"

    # the flux through the TILTED top, with normals from the reference numpy code on the tilted points
    from jno._fem import _region_node_normals

    pts = np.asarray(fem1.field_points[0])
    cells = np.asarray(d1._fem_native_assembly_cells_all[0])
    Nrm = _region_node_normals(d1, pts, cells, order, "top")
    U = u_rt.reshape(-1, 2)
    flux = max(abs(float(U[i] @ n)) for i, n in Nrm.items())
    assert flux < 1e-10 * np.abs(U).max(), f"n·u on the tilted surface is {flux:.2e}, not zero"

    # and it is NOT the answer the stale (flat) normals would give
    u_flat = np.asarray(fem0.solve(**{_pname(yp): _lattice(0.0)[0][ids, 1]}, **NL)).reshape(-1)
    assert np.linalg.norm(u_flat - u_rt) / np.linalg.norm(u_rt) > 1e-2


def test_continuation_rebuilds_the_normals_per_rung():
    """The coordinates may be a continuation sequence: each rung must solve on ITS surface."""
    d0, fem0, yp = _problem(0.0, trainable=True)
    ids = np.asarray(d0._trainable_coords[0]["ids"])
    ys = np.stack([_lattice(t)[0][ids, 1] for t in (0.0, TILT / 2, TILT)])
    fam = np.asarray(fem0.solve(continuation=jno.solve.continuation(keep="all", **{_pname(yp): ys}), **NL))
    for t, got in zip((0.0, TILT / 2, TILT), fam):
        _, fem1, _ = _problem(t, trainable=False)
        ref = np.asarray(fem1.solve(**NL)).reshape(-1)
        assert np.linalg.norm(got.reshape(-1) - ref) / np.linalg.norm(ref) < 1e-9, t


def test_a_linear_problem_with_a_moving_slip_surface_is_refused():
    """Only the steady nonlinear path rebuilds P; anywhere else it must raise, not use stale normals."""
    d = jno.domain(_builder(0.0), compute_mesh_connectivity=True)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    yi.trainable()
    ct = d.variable("top", normals=True, split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    grad, inner = jno.np.grad, jno.np.inner
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    weak = inner(grad(u, [xi, yi]), grad(v, [xi, yi]), n_contract=2) - (1.0 * vi[0] + 0.5 * vi[1])
    ut = u(ct[0], ct[1])
    with pytest.raises(NotImplementedError, match="NONLINEAR"):
        jno.fem([weak, ct[-2] * ut[0] + ct[-1] * ut[1] - 0.0, u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0])


def test_the_build_time_P_cannot_be_used_silently():
    from jno.utils.solver.slip_runtime import StaleSlipP

    P = StaleSlipP((4, 3))
    assert P.shape == (4, 3)
    with pytest.raises(NotImplementedError, match="runtime"):
        _ = P.data
    with pytest.raises(NotImplementedError, match="runtime"):
        _ = P @ jnp.ones(3)


def test_the_solution_is_differentiable_in_the_slip_surface_position():
    """d(solution)/d(tilt) through the runtime normals: reverse mode against a central difference."""
    d0, fem0, yp = _problem(0.0, trainable=True, order=1)
    ids = np.asarray(d0._trainable_coords[0]["ids"])
    y0 = jnp.asarray(_lattice(0.0)[0][ids, 1])
    x0 = jnp.asarray(_lattice(0.0)[0][ids, 0])
    name = _pname(yp)

    def J(t):
        u = fem0.solve(**{name: y0 * (1.0 + t * x0)}, **NL)
        return jnp.sum(jnp.asarray(u).reshape(-1) ** 2)

    t, h = 0.2, 1e-5
    g_ad = float(jax.grad(J)(t))
    g_fd = (float(J(t + h)) - float(J(t - h))) / (2 * h)
    assert abs(g_ad - g_fd) <= 1e-5 * max(abs(g_fd), 1e-12), (g_ad, g_fd)


def test_structural_zeros_are_pruned_and_a_violated_pruning_raises():
    """With only x trainable the flat top stays flat, so the x-coupling of its normal is identically zero
    and is pruned from P. If a pruned entry is nonzero at a solve's coordinates (simulated here by marking
    a live entry as pruned) the eager check must refuse the result rather than drop the coupling."""
    d = jno.domain(_builder(0.0), compute_mesh_connectivity=True)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xp_ = xi.trainable()
    ct = d.variable("top", normals=True, split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    grad, inner = jno.np.grad, jno.np.inner
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    k = 1.0 + 0.5 * (ui[0] * ui[0] + ui[1] * ui[1])
    weak = k * inner(grad(u, [xi, yi]), grad(v, [xi, yi]), n_contract=2) - (1.0 * vi[0] + 0.5 * vi[1])
    ut = u(ct[0], ct[1])
    fem = jno.fem([weak, ct[-2] * ut[0] + ct[-1] * ut[1] - 0.0, u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0])
    plan = fem._periodic["slip_runtime"]["plan"]
    assert plan["mask"].size == 0 and plan["pruned"].size > 0  # every top coupling is structurally zero
    ids = np.asarray(d._trainable_coords[0]["ids"])
    xs = _lattice(0.0)[0][ids, 0] ** 1.2  # a nonuniform x-stretch: the top stays flat
    fem.solve(**{_pname(xp_): xs}, **NL)  # fine: pruned entries are still zero

    # now tilt: y is NOT trainable, so simulate a violated pruning by checking at coordinates the plan
    # cannot represent -- move y through the plan's own spec list
    plan["specs"].append((np.asarray(ids), 1, "__tilt__"))
    from jno.utils.solver.slip_runtime import check_pruned

    with pytest.raises(NotImplementedError, match="pruned"):
        check_pruned(fem._periodic, {_pname(xp_): xs, "__tilt__": _lattice(TILT)[0][ids, 1]})
