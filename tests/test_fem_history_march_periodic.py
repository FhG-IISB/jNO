"""Eliminated constraints on the ``domain(tau=...)`` history march — periodic ties, slip, hanging nodes.

A periodic tie ``u(A) - u(B)``, an exact slip condition ``n·u = 0`` and the hanging nodes of a locally
refined mesh are not assembled: ``jno.fem`` records them as a prolongation ``u = P ũ`` on
``fem._periodic`` and every solve path has to apply it. The load-path march used to hand Newton the FULL
residual, so all three were dropped without a word — a hand-written backward-Euler heat march, periodic in
x, came back bit-identical to the same form with a natural (Neumann) condition on the tied faces.

Oracles:
* **tie, single field** — the backward-Euler step written by hand with the primary unknown's history
  ``(u - u.i(-1))/dt`` equals the tested ``u.t`` periodic transient march at ``time=theta(1.0)`` and the
  same ``dt`` (tests/test_fem_periodic_transient.py), step for step; the seam values agree exactly.
* **tie, coupled** — the same for two cross-coupled fields, each tied.
* **tie + ``.evolves``** — a running average of ``u`` over the march, read back through an L2 projection,
  against the steady periodic solve scaled by the closed-form average.
* **slip / hanging nodes** — the constraint holds to round-off at every step, measured on the solution's
  own nodal values.
* **differentiability** — ``∂/∂θ`` through the reduced march, against central finite differences.
* **a ``.bounds`` box** — ``u >= u.i(-1)`` on a tied march ratchets onto the unit-load LINEAR steady solve of
  the same tied problem, scaled by the peak load factor (every free DOF active on the way down).
* **refusals** — arc-length, the leg that does not reduce, refuses by name.

Step alignment: the march solves at every ``tau`` point, starting from the virgin buffer ``u.i(-1) = 0``,
so march step ``k`` is ``k + 1`` backward-Euler steps from zero — the ``u.t`` trajectory's frame ``k + 1``
(its frame 0 is the initial condition).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
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


DT, NS, EPS = 0.02, 5, 1e-9
PI = np.pi
grad, inner = jno.np.grad, jno.np.inner


def _square(*, corners=True, **grid):
    """Unit square, 8x8 structured. ``l``/``r`` are the tied faces (with or without the corner nodes the
    Dirichlet ``b``/``t`` faces share); ``grid`` is ``time=`` or ``tau=`` or nothing."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=8).domain(**grid)
    if corners:
        d.tag("l", lambda x, y: x < EPS)
        d.tag("r", lambda x, y: x > 1 - EPS)
    else:
        d.tag("l", lambda x, y: (x < EPS) & (y > EPS) & (y < 1 - EPS))
        d.tag("r", lambda x, y: (x > 1 - EPS) & (y > EPS) & (y < 1 - EPS))
    d.tag("b", lambda x, y: y < EPS)
    d.tag("t", lambda x, y: y > 1 - EPS)
    return d


def _grid(kind):
    """The ``u.t`` reference grid (initial condition + NS steps), or the march's NS load points."""
    return dict(time=(0.0, DT * NS, NS + 1)) if kind == "t" else dict(tau=(DT, DT * NS, NS))


def _heat(kind, *, tie=True, amp=1.0):
    """``u_t - Δu = f(x)``, u = 0 on y = 0, 1, periodic in x (or no condition there). ``kind='t'`` writes
    the time derivative as ``u.t``; ``kind='tau'`` writes the backward-Euler step by hand. ``f`` depends on
    x and is not periodic-symmetric, so the natural-Neumann answer is far from the periodic one."""
    d = _square(**_grid(kind))
    u, v = d.fem_symbols(names=("u", "v"))
    V = d.variable("interior", split=True)
    ub, vb = u.bind(x=V[0], y=V[1], t=V[2]), v.bind(x=V[0], y=V[1], t=V[2])
    on = lambda r: d.variable(r, split=True)[:2]  # noqa: E731
    f = amp * 10.0 * (1.0 + jno.np.sin(2 * PI * V[0]) + V[0])
    rate = ub.t if kind == "t" else (ub - u.i(-1)) / DT
    terms = [rate * vb + ub.x * vb.x + ub.y * vb.y - f * vb, u(*on("b")) - 0.0, u(*on("t")) - 0.0]
    if tie:
        terms.append(u(*on("l")) - u(*on("r")))
    if kind == "t":
        terms.append(u(*d.variable("initial", split=True)[:2]) - 0.0)
    return jno.fem(terms)


def _seam(fem):
    """Row-aligned DOF indices of the x = 0 and x = 1 faces (sorted by y)."""
    pts = np.asarray(fem.points)
    lo, hi = np.flatnonzero(pts[:, 0] < 1e-6), np.flatnonzero(pts[:, 0] > 1 - 1e-6)
    return lo[np.argsort(pts[lo, 1])], hi[np.argsort(pts[hi, 1])]


def _reference(fem):
    return np.asarray(fem.solve(time=jno.solve.theta(1.0)).fn())


def test_hand_written_backward_euler_with_a_tie_matches_the_u_t_march():
    march = _heat("tau")
    got = np.asarray(march.solve())
    ref = _reference(_heat("t"))
    assert got.shape == (NS, ref.shape[1]) and ref.shape[0] == NS + 1

    for k in range(NS):
        rel = np.abs(got[k] - ref[k + 1]).max() / np.abs(ref[k + 1]).max()
        assert rel < 1e-8, f"march step {k} is {rel:.2e} off the u.t periodic reference"

    # The tie is what is being tested, so it has to matter: the same march with no condition on the
    # tied faces (natural Neumann) -- which is what the march used to return -- is far away.
    neumann = np.asarray(_heat("tau", tie=False).solve())
    gap = np.abs(neumann[-1] - ref[-1]).max() / np.abs(ref[-1]).max()
    assert gap > 0.05, f"the tie barely changes this problem ({gap:.2e}); the comparison would be vacuous"


def test_the_seam_values_match_exactly_at_every_step():
    march = _heat("tau")
    got = np.asarray(march.solve())
    lo, hi = _seam(march)
    assert np.abs(got[:, lo]).max() > 0.1, "the seam never loaded; the check would be vacuous"
    assert np.array_equal(got[:, lo], got[:, hi]), (
        f"the tie is not exact on the march: seam mismatch {np.abs(got[:, lo] - got[:, hi]).max():.2e}"
    )
    # ...and the Dirichlet corners the two faces share keep their value through the reduction.
    assert np.abs(got[:, [lo[0], lo[-1], hi[0], hi[-1]]]).max() < 1e-14


def _coupled(kind):
    """Two cross-coupled heat fields, each tied in x; the tied faces exclude the Dirichlet corners."""
    d = _square(corners=False, **_grid(kind))
    u, v = d.fem_symbols(names=("u", "v"))
    w, q = d.fem_symbols(names=("w", "q"))
    V = d.variable("interior", split=True)
    B = lambda s: s.bind(x=V[0], y=V[1], t=V[2])  # noqa: E731
    ub, vb, wb, qb = B(u), B(v), B(w), B(q)
    on = lambda r: d.variable(r, split=True)[:2]  # noqa: E731
    f1 = 10.0 * (1.0 + jno.np.sin(2 * PI * V[0]) + V[0])
    f2 = 5.0 * jno.np.cos(2 * PI * V[0]) * V[1]
    ru = ub.t if kind == "t" else (ub - u.i(-1)) / DT
    rw = wb.t if kind == "t" else (wb - w.i(-1)) / DT
    terms = [
        ru * vb + ub.x * vb.x + ub.y * vb.y - 3.0 * wb * vb - f1 * vb,
        rw * qb + 0.5 * (wb.x * qb.x + wb.y * qb.y) + 3.0 * ub * qb - f2 * qb,
        u(*on("b")) - 0.0,
        u(*on("t")) - 0.0,
        w(*on("b")) - 0.0,
        w(*on("t")) - 0.0,
        u(*on("l")) - u(*on("r")),
        w(*on("l")) - w(*on("r")),
    ]
    if kind == "t":
        ci = d.variable("initial", split=True)[:2]
        terms += [u(*ci) - 0.0, w(*ci) - 0.0]
    return jno.fem(terms), (u, w)


def test_coupled_two_field_march_with_ties_matches_the_u_t_march():
    march, (u, w) = _coupled("tau")
    got = np.asarray(march.solve())
    ref_fem, _ = _coupled("t")
    ref = _reference(ref_fem)
    assert len(march.blocks) == 2 and got.shape == (NS, ref.shape[1])

    for k in range(NS):
        for field in (u, w):
            blk = march.blocks[march.block_index(field)]
            a, b = got[k, blk], ref[k + 1, blk]
            rel = np.abs(a - b).max() / np.abs(b).max()
            assert rel < 1e-8, f"coupled march step {k}, block {blk}: {rel:.2e} off the u.t reference"

    lo, hi = _seam(march)
    for field in (u, w):
        blk = got[:, march.blocks[march.block_index(field)]]
        assert np.abs(blk).max() > 0.05, "a block never responded; the check would be vacuous"
        assert np.array_equal(blk[:, lo], blk[:, hi]), "a coupled field's tie is not exact on the march"


def test_running_average_state_on_a_periodic_march():
    """``m.evolves(m.i(-1) + (u - m.i(-1))/(k+1))`` keeps the running mean of ``u`` at the quadrature
    points; a second field ``w`` reads it back by an L2 projection, ``w_k = m_{k-1}``.

    With ``-Δu = (1+τ) f`` on the grid τ = 0, 1, …, the solution is ``u_k = (1+k) u_s`` exactly, ``u_s``
    the steady periodic solve of ``-Δu = f``. So ``m_k = mean_{j<=k} (1+j) u_s = (k+2)/2 · u_s`` and
    ``w_k = (k+1)/2 · u_s`` (``w_0 = 0``, the virgin state). The projection is exact because ``m`` is the
    FE function ``u_s`` sampled at the quadrature points."""
    N = 4
    # Coupled, so the tied faces exclude the Dirichlet corners (the multi-field reduction takes no
    # prescribed-DOF exclusion and refuses a tie that would eliminate one).
    d = _square(corners=False, tau=(0.0, N - 1.0, N))
    co = d.variable("interior", split=True)
    X, tau = [co[0], co[1]], co[2]
    on = lambda r: d.variable(r, split=True)[:2]  # noqa: E731
    u, v = d.fem_symbols(names=("u", "v"))
    w, q = d.fem_symbols(names=("w", "q"))
    m, _ = d.fem_symbols(value_shape=(), names=("m", "m_"))
    f = 10.0 * (1.0 + jno.np.sin(2 * PI * co[0]) + co[0])
    fem = jno.fem(
        [
            inner(grad(u, X), grad(v, X), 1) - (1.0 + tau) * f * v,
            w * q - m.i(-1) * q,
            m.evolves(m.i(-1) + (u - m.i(-1)) / (tau + 1.0)),
            u(*on("b")) - 0.0,
            u(*on("t")) - 0.0,
            u(*on("l")) - u(*on("r")),
        ]
    )
    traj = np.asarray(fem.solve())
    assert traj.shape[0] == N

    dS = _square(corners=False)
    cs = dS.variable("interior", split=True)
    XS = [cs[0], cs[1]]
    onS = lambda r: dS.variable(r, split=True)[:2]  # noqa: E731
    uS, vS = dS.fem_symbols()
    fS = 10.0 * (1.0 + jno.np.sin(2 * PI * cs[0]) + cs[0])
    us = np.asarray(
        jno.fem(
            [inner(grad(uS, XS), grad(vS, XS), 1) - fS * vS, uS(*onS("b")) - 0.0, uS(*onS("t")) - 0.0]
            + [uS(*onS("l")) - uS(*onS("r"))]
        ).solve()
    )
    scale = np.abs(us).max()
    bu, bw = fem.blocks[fem.block_index(u)], fem.blocks[fem.block_index(w)]
    for k in range(N):
        eu = np.abs(traj[k, bu] - (1 + k) * us).max() / scale
        ew = np.abs(traj[k, bw] - (0.0 if k == 0 else (k + 1) / 2) * us).max() / scale
        assert eu < 1e-8, f"step {k}: u is {eu:.2e} off (1+k)·u_steady"
        assert ew < 1e-7, f"step {k}: the running average read back is {ew:.2e} off its closed form"


def test_slip_condition_holds_at_every_step_of_the_march():
    """Same mechanism, different ``P``: the exact slip elimination ``n·u = 0`` on a disk wall. It was
    dropped on the march exactly like a tie (the wall velocity came back at full magnitude)."""
    d = jno.shape.disk(0.0, 0.0, 1.0, size=0.35).domain(tau=(0.05, 0.15, 3))
    u, phi = d.fem_symbols(value_shape=(2,))
    co = d.variable("interior", split=True)
    X = [co[0], co[1]]
    cb = d.variable("boundary", normals=True, split=True)
    nx, ny = cb[-2], cb[-1]
    vi = phi.bind(x=co[0], y=co[1])
    ub = u(cb[0], cb[1])
    fem = jno.fem(
        [
            inner((u - u.i(-1)) / 0.05, phi, 1)
            + inner(grad(u, X), grad(phi, X), n_contract=2)
            - (1.0 * vi.component(0) + 0.5 * vi.component(1)),
            nx * ub[0] + ny * ub[1] - 0.0,
        ]
    )
    traj = np.asarray(fem.solve())
    pts = np.asarray(fem.field_points[0])
    r = np.linalg.norm(pts, axis=1)
    wall = r > 1.0 - 1e-6
    normal = pts[wall] / r[wall][:, None]
    for k in range(traj.shape[0]):
        U = traj[k].reshape(-1, 2)
        assert np.abs(U).max() > 1e-2, "the field never loaded; the check would be vacuous"
        un = np.abs((U[wall] * normal).sum(1)).max()
        assert un < 1e-12, f"step {k}: the slip wall is violated by {un:.2e}"


def test_hanging_node_constraint_holds_at_every_step_of_the_march():
    """Same mechanism again: a locally refined quad mesh's hanging nodes equal their parents' average."""
    from jno.utils.solver.fem_refine import refine_domain

    d = jno.shape.rect(0, 0, 1, 1).quad().structured(n=4).domain(tau=(0.05, 0.15, 3), compute_mesh_connectivity=False)
    p = np.asarray(d.mesh.points)[:, :2]
    cq = np.asarray(d.mesh.cells_dict["quad"])
    d = refine_domain(d, np.flatnonzero(np.linalg.norm(p[cq].mean(axis=1) - 0.5, axis=1) < 0.3))
    d.tag("bd", lambda x, y: (x < EPS) | (x > 1 - EPS) | (y < EPS) | (y > 1 - EPS))
    u, v = d.fem_symbols()
    co = d.variable("interior", split=True)
    X = [co[0], co[1]]
    fem = jno.fem(
        [
            inner(grad(u, X), grad(v, X), 1) + (u - u.i(-1)) / 0.05 * v - 1.0 * v,
            u(*d.variable("bd", split=True)[:2]) - 0.0,
        ]
    )
    traj = np.asarray(fem.solve())
    hang = d._fem_hanging_nodes
    assert hang, "the refinement produced no hanging nodes; the check would be vacuous"
    assert np.abs(traj).max() > 1e-2
    err = max(
        abs(traj[k, h] - sum(wt * traj[k, pp] for pp, wt in par)) for k in range(len(traj)) for h, par in hang.items()
    )
    assert err < 1e-13, f"a hanging node left its parents' average by {err:.2e} on the march"


def test_gradient_flows_through_the_reduced_march():
    """``∂/∂a`` of the final state through the periodic march, against central differences. The form is
    linear in the source amplitude ``a``, so the objective is quadratic and the difference is exact up to
    the solver tolerance."""
    aP = jno.np.reshape(jno.np.parameter((1,), name="a"), ())
    node = _heat("tau", amp=aP).solve()
    assert type(node).__name__ == "FunctionCall", "a parametric reduced march must stay a differentiable node"

    def objective(a):
        return jnp.sum(node.fn(jnp.reshape(a, (1,)))[-1] ** 2)

    g = float(jax.grad(objective)(0.7))
    fd = float((objective(0.7 + 1e-3) - objective(0.7 - 1e-3)) / 2e-3)
    assert np.isfinite(g) and abs(g) > 1e-3
    assert abs(g - fd) < 1e-6 * abs(fd), f"AD {g:.10e} vs FD {fd:.10e}"


def test_arclength_with_a_tie_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="arclength.*periodic tie"):
        _heat("tau").solve(tau=jno.solve.arclength())


def _ratchet(*, bounded, steady=False):
    """``-Δu = s(τ) f(x)``, ``f = 10 (1 + 0.5 sin 2πx) > 0``, u = 0 on y = 0, 1, periodic in x, with the load
    factor ``s`` rising 0 -> 1 and falling back. ``bounded`` adds ``u.bounds(u.i(-1), None)``: u may never
    decrease. ``steady`` builds the unit-load (s = 1) linear solve instead, on its own domain."""
    d = _square() if steady else _square(tau=(0.0, 1.0, 9))
    u, v = d.fem_symbols(names=("u", "v"))
    V = d.variable("interior", split=True)
    ub, vb = u.bind(x=V[0], y=V[1]), v.bind(x=V[0], y=V[1])
    on = lambda r: d.variable(r, split=True)[:2]  # noqa: E731
    f = 10.0 * (1.0 + 0.5 * jno.np.sin(2 * PI * V[0]))
    load = 1.0 if steady else 1.0 - jno.np.abs(2 * V[-1] - 1.0)  # 0 -> 1 -> 0 over tau
    terms = [ub.x * vb.x + ub.y * vb.y - load * f * vb, u(*on("b")) - 0.0, u(*on("t")) - 0.0]
    terms.append(u(*on("l")) - u(*on("r")))
    if not steady:
        s_, _ = d.fem_symbols(value_shape=(), names=("s", "sv"))
        terms[0] = terms[0] + 0.0 * s_.i(-1) * vb  # an inert state: its only job is to trigger the march
        terms.append(s_.evolves(s_.i(-1)))
    if bounded:
        terms.append(u.bounds(u.i(-1), None))
    fem = jno.fem(terms)
    return np.asarray(fem.solve()), fem


def test_bounds_with_a_tie_on_a_march_ratchet_at_the_peak():
    """A box used to be refused on a tied march. Now: the load rises and falls; with ``u >= u.i(-1)`` the
    tied field follows ``s(τ) u₁`` up -- ``u₁`` the unit-load solve of the same tied problem through the
    LINEAR steady path, an independent route -- and then holds ``u₁`` exactly while the load falls,
    because ``f > 0`` makes every free DOF want to decrease (the whole field is active). The unbounded
    control unloads to zero, so the bound is what holds it."""
    u1, _fem1 = _ratchet(bounded=False, steady=True)
    ratchet, fem = _ratchet(bounded=True)
    control, _fem = _ratchet(bounded=False)
    s = 1.0 - np.abs(2 * np.linspace(0.0, 1.0, 9) - 1.0)
    held = np.maximum.accumulate(s)  # the load factor the ratchet remembers
    scale = np.abs(u1).max()
    assert scale > 0.1
    for k in range(9):
        err = np.abs(ratchet[k] - held[k] * u1).max() / scale
        assert err < 1e-8, f"step {k}: the tied ratchet is {err:.2e} off {held[k]:.2f} u1"
    assert np.abs(control[-1]).max() / scale < 1e-8, "the unbounded control did not unload"
    lo, hi = _seam(fem)
    assert np.array_equal(ratchet[:, lo], ratchet[:, hi]), "the tie is not exact under the box"
