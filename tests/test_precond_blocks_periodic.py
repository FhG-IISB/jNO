"""Block preconditioners on a REDUCED system -- periodic ties and the exact slip elimination.

A periodic tie (or a slip condition ``n·u = 0``, or a hanging-node constraint) makes ``fem.solve`` work on
``P^T A P``, reduced block-wise per field: the reduced vector is each field's reduced DOFs concatenated.
``fem.blocks`` slice the FULL solution, and the block preconditioners used to slice the reduced operator
with them -- the last field's slice ran past the end and came back empty (``triangular((u, jacobi()),
(p, jacobi()))`` on a 3-D periodic Navier-Stokes failed with "incompatible shapes (729,), (0,)"), an
``amg`` block was built on garbage, and an auxiliary ``form`` was sized for the full space.

The oracle is a sparse-direct solve of the same system -- a preconditioner may change how fast a Krylov
method converges, never what it converges to -- and, for the channel, the exact Poiseuille profile,
which Taylor-Hood P2 reproduces to round-off. The reduced sizes are checked against a count taken
straight off the mesh (every node on the eliminated face goes), not against the reduction itself.
"""

import functools

import jax
import numpy as np
import pytest

import jno

MU, G, H, L, EPS = 1.0, 1.0, 1.0, 2.0, 1e-9


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poiseuille(y):
    return (G / (2 * MU)) * y * (H - y)


@functools.lru_cache(maxsize=None)
def _channel(periodic: bool, nonlinear: bool = False):
    """Taylor-Hood channel ``[0, L] x [0, H]`` driven by a body force ``G``; walls at ``y = 0, H``.

    ``periodic``: tied in x (the left face is eliminated onto the right). Otherwise the exact profile is
    imposed at inflow and outflow -- the non-periodic CONTROL, whose system must be unchanged. The
    pressure is pinned at an interior point (a pin on the tied face would be eliminated by the tie).
    ``nonlinear`` adds the convection term; Poiseuille is still exact (``u·∇u = 0`` for ``u = (u(y), 0)``).
    """
    inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    d = jno.shape.rect(0.0, 0.0, L, H).structured(n=8).domain()
    d.point_region("ppin", (L / 2, H / 2))
    d.tag("left", lambda x, y: x < EPS)
    d.tag("right", lambda x, y: x > L - EPS)
    # on the periodic channel the wall stops short of the eliminated (left) face: its corner nodes are
    # prolonged from the right-hand ones, which carry the wall condition
    wall = (lambda x, y: ((y < EPS) | (y > H - EPS)) & (x > EPS)) if periodic else (lambda x, y: (y < EPS) | (y > H - EPS))
    d.tag("wall", wall)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    X = list(d.variable("interior", split=True)[:2])
    gu, gv = grad(u, X), grad(v, X)
    pb, qb = p.bind(x=X[0], y=X[1]), q.bind(x=X[0], y=X[1])
    ub, vb = u.bind(x=X[0], y=X[1]), v.bind(x=X[0], y=X[1])
    div = trace
    xw, yw, _ = d.variable("wall", split=True)
    xp, yp, _ = d.variable("ppin", split=True)
    momentum = MU * inner(gu, gv, n_contract=2) - pb * div(gv) - G * vb[0]
    if nonlinear:
        momentum = momentum + inner(inner(gu, ub, n_contract=1), vb, n_contract=1)
    terms = [momentum, -qb * div(gu), u(xw, yw)[0] - 0.0, u(xw, yw)[1] - 0.0, p(xp, yp) - 0.0]
    if periodic:
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        terms += [u(xl, yl) - u(xr, yr), p(xl, yl) - p(xr, yr)]
    else:
        d.tag("ends", lambda x, y: (x < EPS) | (x > L - EPS))
        xe, ye, _ = d.variable("ends", split=True)
        terms += [u(xe, ye)[0] - _poiseuille(ye), u(xe, ye)[1] - 0.0]
    fem = jno.fem(terms)
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    return fem, u, p, pb, qb, ref


def _eliminated_counts(fem):
    """Per field, how many DOFs the x-tie removes, counted off the mesh: every node on the left face."""
    out = []
    for i, blk in enumerate(fem.blocks):
        pts = np.asarray(fem.field_points[i])
        vec = (blk.stop - blk.start) // pts.shape[0]
        out.append(vec * int(np.sum(pts[:, 0] < EPS)))
    return out


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def _arr(out):
    """A solve's result as a flat array (a periodic nonlinear solve stays a lazy node: evaluate it)."""
    return np.asarray(out.fn() if hasattr(out, "fn") else out).reshape(-1)


def _solve(fem, precond, nonlinear=None):
    kw = {"nonlinear": nonlinear} if nonlinear is not None else {}
    return _arr(fem.solve(linear=jno.solve.fgmres(tol=1e-12, restart=200, maxiter=2000), precond=precond, **kw))


def test_the_direct_oracle_is_poiseuille():
    """Anchor the oracle itself to the exact solution, so "matches lu" below means "is right"."""
    for periodic in (True, False):
        fem, *_rest, ref = _channel(periodic)
        pts = np.asarray(fem.field_points[0])
        U = ref[fem.blocks[0]].reshape(-1, 2)
        assert np.abs(U[:, 0] - _poiseuille(pts[:, 1])).max() < 1e-12
        assert np.abs(U[:, 1]).max() < 1e-12


def test_context_blocks_are_the_reduced_per_field_slices():
    """What a block preconditioner is handed: slices of the REDUCED operator, not of the full solution."""
    pytest.importorskip("pyamg")
    fem, u, p, pb, qb, ref = _channel(True)
    seen = []

    def spy(ctx):
        seen.append((ctx.A.shape[0], ctx.blocks, ctx.block_slice(u), ctx.block_slice(p)))
        # a user's own block scheme: a one-field form assembled against the WHOLE (reduced) context must
        # come back on that field's reduced space, P_p^T M_p P_p
        seen.append(ctx.assemble([pb * qb]))
        return jno.precond.triangular((u, jno.precond.amg()), (p, jno.precond.jacobi())).materialize(ctx)

    got = _solve(fem, spy)
    assert _rel(got, ref) < 1e-8
    n, blocks, su, sp = seen[0]
    full = [b.stop - b.start for b in fem.blocks]
    want = [f - e for f, e in zip(full, _eliminated_counts(fem))]
    assert [b.stop - b.start for b in blocks] == want, (blocks, want)
    assert sum(want) == n < fem.dofs  # the slices tile the reduced system exactly
    assert blocks[0].start == 0 and blocks[0].stop == blocks[1].start and blocks[-1].stop == n
    assert (su, sp) == (blocks[0], blocks[1])
    Mp = np.asarray(seen[1].dense())
    assert Mp.shape == (want[1], want[1])
    # 1ᵀ PᵀMP 1 = 1ᵀ M 1 = |Ω| (a periodic P maps constants to constants): the mass was reduced, not cut
    assert abs(Mp.sum() - L * H) < 1e-12 and np.allclose(Mp, Mp.T)
    # the public layout of the SOLUTION is untouched: fem.blocks still slice the full field
    assert [b.stop - b.start for b in fem.blocks] == full and fem.blocks[-1].stop == fem.dofs


def test_context_blocks_are_unchanged_without_a_reduction():
    """Non-periodic control: the context's slices ARE fem.blocks."""
    fem, u, p, *_rest, ref = _channel(False)
    seen = []

    def spy(ctx):
        seen.append(ctx.blocks)
        return jno.precond.triangular((u, jno.precond.jacobi()), (p, jno.precond.jacobi())).materialize(ctx)

    assert _rel(_solve(fem, spy), ref) < 1e-8
    assert seen[0] == fem.blocks


def _specs(u, p, pb, qb):
    return {
        "triangular-jacobi": lambda: jno.precond.triangular((u, jno.precond.jacobi()), (p, jno.precond.jacobi())),
        "triangular-amg": lambda: jno.precond.triangular((u, jno.precond.amg()), (p, jno.precond.amg())),
        "block_diag-amg": lambda: jno.precond.block_diag((u, jno.precond.amg()), (p, jno.precond.jacobi())),
        # an auxiliary weak form on the FULL pressure space must be reduced with the pressure's P
        "triangular-form": lambda: jno.precond.triangular((u, jno.precond.amg()), (p, jno.precond.form([pb * qb]))),
        "triangular-form-f32": lambda: jno.precond.triangular(
            (u, jno.precond.amg()), (p, jno.precond.form([pb * qb], float32=True))
        ),
        "saddle-mass": lambda: jno.precond.saddle(mass_weight=1.0 / MU),
        "saddle-cahouet-chabard": lambda: jno.precond.saddle(mass_weight=1.0 / MU, laplace_weight=1.0),
        "saddle-lsc": lambda: jno.precond.saddle(schur="lsc"),
    }


_NAMES = list(_specs(None, None, None, None))


@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "control"])
@pytest.mark.parametrize("name", _NAMES)
def test_block_preconditioner_matches_the_direct_solve(name, periodic):
    pytest.importorskip("pyamg")
    fem, u, p, pb, qb, ref = _channel(periodic)
    got = _solve(fem, _specs(u, p, pb, qb)[name]())
    assert np.all(np.isfinite(got))
    assert _rel(got, ref) < 1e-8, f"{name}: rel err {_rel(got, ref):.2e} against the direct solve"


@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "control"])
def test_block_preconditioner_on_the_reduced_newton_tangent(periodic):
    """Steady Navier-Stokes: the direct Newton hands the preconditioner the REDUCED tangent PᵀJP (the
    `_reduced` wrapper), and a solution-dependent Schur factor is refreshed from the REDUCED iterate --
    LSC and PCD prolong it before re-slicing the tangent / re-reading the velocity. (Jacobi on the
    velocity block keeps these per-linearization; an unbuilt amg() there is built once, before the
    Newton loop -- tests/test_precond_amg_steady_newton.py.)"""
    fem, u, p, pb, qb, _ref = _channel(periodic, nonlinear=True)
    newton = jno.solve.newton(direct=True, rtol=1e-11, atol=1e-12)
    ref = _arr(fem.solve(nonlinear=newton, linear=jno.solve.lu(backend="host")))
    pts = np.asarray(fem.field_points[0])
    assert np.abs(ref[fem.blocks[0]].reshape(-1, 2)[:, 0] - _poiseuille(pts[:, 1])).max() < 1e-10
    for spec in (
        jno.precond.triangular((u, jno.precond.jacobi()), (p, jno.precond.jacobi())),
        jno.precond.triangular((u, jno.precond.jacobi()), (p, jno.precond.lsc())),
        jno.precond.triangular((u, jno.precond.jacobi()), (p, jno.precond.pcd(viscosity=MU))),
    ):
        got = _solve(fem, spec, nonlinear=newton)
        assert _rel(got, ref) < 1e-8, f"{spec}: rel err {_rel(got, ref):.2e}"
    if not periodic:
        return
    # The matrix-free Newton materializes the preconditioner INSIDE its traced loop: the pressure form must
    # still be reduced to concrete data there (the default factor-once LU reads it on the host).
    spec = jno.precond.triangular(
        (u, jno.precond.inner(jno.solve.gmres(tol=1e-8, maxiter=400))), (p, jno.precond.form([pb * qb]))
    )
    got = _solve(fem, spec, nonlinear=jno.solve.newton(direct=False, rtol=1e-10, atol=1e-11))
    assert _rel(got, ref) < 1e-8, f"matrix-free Newton + form: rel err {_rel(got, ref):.2e}"


def test_block_preconditioner_on_a_periodic_transient_march():
    """The shape of the original failure: a transient march, direct Newton, fgmres + triangular blocks.
    Two coupled periodic heat fields (a real off-diagonal coupling); the march must match the direct one."""
    from jno import jnp_ops as jnn

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=8).domain(time=(0.0, 0.02, 5))
    for nm, pred in {
        "left": lambda x, y: x < 1e-6,
        "right": lambda x, y: x > 1 - 1e-6,
        "bottom": lambda x, y: y < 1e-6,
        "top": lambda x, y: y > 1 - 1e-6,
    }.items():
        d.tag(nm, pred)
    u, pu = d.fem_symbols(names=("u", "pu"))
    w, pw = d.fem_symbols(value_shape=(2,), names=("w", "pw"))
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    (xl, yl, _), (xr, yr, _) = d.variable("left", split=True), d.variable("right", split=True)
    (xb, yb, _), (xt, yt, _) = d.variable("bottom", split=True), d.variable("top", split=True)
    ub, pub = u.bind(x=xi, y=yi, t=ti), pu.bind(x=xi, y=yi, t=ti)
    wb, pwb = w.bind(x=xi, y=yi, t=ti), pw.bind(x=xi, y=yi, t=ti)
    X = [xi, yi]
    grad, inner = jno.np.grad, jno.np.inner
    mode = jnn.sin(2 * np.pi * ci[0]) * jnn.sin(2 * np.pi * ci[1])
    fem = jno.fem(
        [
            ub.t * pub + 0.1 * inner(grad(u, X), grad(pu, X), n_contract=1) + ub**3 * pub - 2.0 * wb[0] * pub,
            inner(wb.t, pwb, n_contract=1) + 0.2 * inner(grad(w, X), grad(pw, X), n_contract=2) - ub * pwb[1],
            u(xl, yl) - u(xr, yr),
            u(xb, yb) - u(xt, yt),
            w(xl, yl) - w(xr, yr),
            w(xb, yb) - w(xt, yt),
            u(ci[0], ci[1]) - mode,
            w(ci[0], ci[1])[0] - mode,
            w(ci[0], ci[1])[1] - 0.0,
        ]
    )
    newton = jno.solve.newton(direct=True, rtol=1e-11, atol=1e-12)
    ref = np.asarray(fem.solve(nonlinear=newton, linear=jno.solve.lu(backend="host")).fn())
    got = np.asarray(
        fem.solve(
            nonlinear=newton,
            linear=jno.solve.fgmres(tol=1e-12, restart=100),
            precond=jno.precond.triangular((u, jno.precond.jacobi()), (w, jno.precond.jacobi())),
        ).fn()
    )
    assert np.all(np.isfinite(got)) and np.abs(ref[-1]).max() > 1e-2  # it moved, and did not blow up
    assert _rel(got[-1], ref[-1]) < 1e-8


def test_block_preconditioner_on_a_slip_reduced_system():
    """The exact slip elimination is the same kind of reduction (the velocity block loses one DOF per
    constrained node, the pressure block none), so the same slices apply. Plug flow between two slip
    walls: u = (1, 0), p = 0 exactly."""
    pytest.importorskip("pyamg")
    inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    d = jno.shape.rect(0.0, 0.0, L, H).structured(n=8).domain()
    d.point_region("ppin", (L / 2, H / 2))
    d.tag("ends", lambda x, y: (x < EPS) | (x > L - EPS))
    d.tag("slipwall", lambda x, y: ((y < EPS) | (y > H - EPS)) & (x > EPS) & (x < L - EPS))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    X = list(d.variable("interior", split=True)[:2])
    pb, qb = p.bind(x=X[0], y=X[1]), q.bind(x=X[0], y=X[1])
    xe, ye, _ = d.variable("ends", split=True)
    xp, yp, _ = d.variable("ppin", split=True)
    c = d.variable("slipwall", normals=True, split=True)
    us = u.bind(x=c[0], y=c[1])
    fem = jno.fem(
        [
            MU * inner(grad(u, X), grad(v, X), n_contract=2) - pb * trace(grad(v, X)),
            -qb * trace(grad(u, X)),
            u(xe, ye)[0] - 1.0,
            u(xe, ye)[1] - 0.0,
            c[-2] * us[0] + c[-1] * us[1] - 0.0,  # slip  n·u = 0
            p(xp, yp) - 0.0,
        ]
    )
    assert fem._periodic is not None and fem._periodic.get("coupling") == "slip"
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)
    U = ref[fem.blocks[0]].reshape(-1, 2)
    assert np.abs(U[:, 0] - 1.0).max() < 1e-10 and np.abs(U[:, 1]).max() < 1e-10
    for spec in (
        jno.precond.triangular((u, jno.precond.amg()), (p, jno.precond.jacobi())),
        jno.precond.saddle(mass_weight=1.0 / MU),
    ):
        got = _solve(fem, spec)
        assert _rel(got, ref) < 1e-8, f"{spec}: rel err {_rel(got, ref):.2e}"


def test_a_form_on_a_single_field_periodic_system_is_reduced():
    """``jno.precond.form`` over the whole (single-field) system: assembled on the full space, it must be
    reduced with the tie's P to precondition ``PᵀAP``. The form here IS the operator, so an exact inverse
    of it converges in one outer iteration -- a wrongly reduced form would not."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=8).domain()
    d.tag("left", lambda x, y: x < 1e-6)
    d.tag("right", lambda x, y: x > 1 - 1e-6)
    d.tag("bottom", lambda x, y: y < 1e-6)
    d.tag("top", lambda x, y: y > 1 - 1e-6)
    u, v = d.fem_symbols(names=("u", "v"))
    X = list(d.variable("interior", split=True)[:2])
    ub, vb = u.bind(x=X[0], y=X[1]), v.bind(x=X[0], y=X[1])
    a = jno.np.inner(jno.np.grad(u, X), jno.np.grad(v, X), n_contract=1) + ub * vb
    f = jno.np.sin(2 * np.pi * X[0]) * jno.np.cos(2 * np.pi * X[1])
    (xl, yl, _), (xr, yr, _) = d.variable("left", split=True), d.variable("right", split=True)
    (xb, yb, _), (xt, yt, _) = d.variable("bottom", split=True), d.variable("top", split=True)
    fem = jno.fem([a - f * vb, u(xl, yl) - u(xr, yr), u(xb, yb) - u(xt, yt)])
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)
    got = np.asarray(
        fem.solve(linear=jno.solve.fgmres(tol=1e-12, restart=5, maxiter=1), precond=jno.precond.form([a]))
    ).reshape(-1)
    assert _rel(got, ref) < 1e-10


def test_a_form_of_the_wrong_size_is_refused_by_name():
    """A form that no reduction maps onto the block it preconditions is a user error: say so, with sizes,
    instead of failing deep inside the Krylov loop."""
    fem, u, p, pb, qb, _ref = _channel(True)
    # a PRESSURE-space form handed to the VELOCITY block: no reduction maps one onto the other
    wrong = jno.precond.triangular((u, jno.precond.form([pb * qb])), (p, jno.precond.jacobi()))
    with pytest.raises(ValueError, match="form auxiliary operator is .* but the operator it preconditions"):
        _solve(fem, wrong)


def test_staggered_sweeps_the_reduced_fields(monkeypatch):
    """``jno.solve.staggered`` on a periodic system is handed the REDUCED iterate, so its sweep groups must
    be the reduced per-field slices -- not ``fem.blocks``, which put the second field's indices past the end
    of the reduced vector (where a gather silently clamps them) and swept "fields" straddling the real ones.
    The answer must match a monolithic Newton of the same system."""
    import jno.utils.solver.newton_krylov as nk

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=6).domain()
    for nm, pred in {
        "left": lambda x, y: x < 1e-6,
        "right": lambda x, y: x > 1 - 1e-6,
        "bottom": lambda x, y: y < 1e-6,
        "top": lambda x, y: y > 1 - 1e-6,
    }.items():
        d.tag(nm, pred)
    u, φ = d.fem_symbols(names=("u", "phi"))
    w, ψ = d.fem_symbols(value_shape=(2,), names=("w", "psi"))
    X = list(d.variable("interior", split=True)[:2])
    grad, inner, sin, cos = jno.np.grad, jno.np.inner, jno.np.sin, jno.np.cos
    ub, φb = u.bind(x=X[0], y=X[1]), φ.bind(x=X[0], y=X[1])
    wb, ψb = w.bind(x=X[0], y=X[1]), ψ.bind(x=X[0], y=X[1])
    f, g = sin(2 * np.pi * X[0]) * cos(2 * np.pi * X[1]), cos(2 * np.pi * X[0])
    (xl, yl, _), (xr, yr, _) = d.variable("left", split=True), d.variable("right", split=True)
    (xb, yb, _), (xt, yt, _) = d.variable("bottom", split=True), d.variable("top", split=True)
    fem = jno.fem(
        [
            inner(grad(u, X), grad(φ, X), n_contract=1) + (ub + 0.1 * ub**3 - 0.5 * wb[0] - f) * φb,
            inner(grad(w, X), grad(ψ, X), n_contract=2) + inner(wb, ψb, n_contract=1) - 0.5 * ub * ψb[0] - g * ψb[1],
            u(xl, yl) - u(xr, yr),
            u(xb, yb) - u(xt, yt),
            w(xl, yl) - w(xr, yr),
            w(xb, yb) - w(xt, yt),
        ]
    )
    ref = _arr(
        fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-12, atol=1e-12), linear=jno.solve.lu(backend="host"))
    )

    seen = []
    real = nk.staggered_newton

    def spy(residual_fn, u0, blocks, **kw):
        seen.append((int(np.size(u0)), [np.asarray(b) for b in blocks]))
        return real(residual_fn, u0, blocks, **kw)

    monkeypatch.setattr(nk, "staggered_newton", spy)
    got = _arr(fem.solve(nonlinear=jno.solve.staggered([u, w], rtol=1e-12, atol=1e-12)))
    n, groups = seen[0]
    # the reduced sizes, counted off the mesh: one DOF per component per periodic image class
    kept = [
        (fem.blocks[i].stop - fem.blocks[i].start) // len(pts) * len({tuple(np.round(np.mod(q, 1.0), 9)) for q in pts})
        for i, pts in enumerate(np.asarray(fp)[:, :2] for fp in fem.field_points)
    ]
    assert n == sum(kept) < fem.dofs
    assert [len(g) for g in groups] == kept
    assert np.array_equal(np.concatenate(groups), np.arange(n))  # the groups tile the reduced vector
    assert _rel(got, ref) < 1e-8


def test_a_block_split_of_the_fused_complex_block_is_refused_by_name():
    """A complex form is solved as its fused real-equivalent ``[Re; Im]`` block, where a field's DOFs are two
    separate ranges -- no slice describes them. The layout check refuses it by name (it used to surface as an
    opaque host-callback error from the half-filled preconditioner)."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=4).domain()
    d.tag("b", lambda x, y: (x < 1e-6) | (x > 1 - 1e-6) | (y < 1e-6) | (y > 1 - 1e-6))
    u, v = d.fem_symbols(names=("u", "v"))
    w, z = d.fem_symbols(value_shape=(2,), names=("w", "z"))
    X = list(d.variable("interior", split=True)[:2])
    ub, vb, wb, zb = (s.bind(x=X[0], y=X[1]) for s in (u, v, w, z))
    grad, inner = jno.np.grad, jno.np.inner
    xb, yb, _ = d.variable("b", split=True)
    fem = jno.fem(
        [
            inner(grad(u, X), grad(v, X), n_contract=1) + 1j * ub * vb - vb - 0.3 * wb[0] * vb,
            inner(grad(w, X), grad(z, X), n_contract=2) + 1j * inner(wb, zb, n_contract=1) - 0.3 * ub * zb[0],
            u(xb, yb) - 0.0,
            w(xb, yb) - 0.0,
        ]
    )
    tri = jno.precond.triangular((u, jno.precond.jacobi()), (w, jno.precond.jacobi()))
    with pytest.raises(ValueError, match=r"fused real-equivalent \[Re; Im\] block"):
        fem.solve(linear=jno.solve.fgmres(tol=1e-10), precond=tri)
