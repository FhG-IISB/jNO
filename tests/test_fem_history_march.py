"""The FEM history march — ``fem.solve()`` marching a ``domain(tau=...)`` load path, with nothing passed.

The load-path (pseudo-time) march is triggered by the presence of ``.i(k)`` step-history in the form
together with a ``tau=`` grid on the domain — exactly as a ``u.t`` term triggers the transient stepper.
Internal states read their past with ``state.i(-1)`` and advance with ``state.evolves(<formula>)``; the
load is written as a function of τ in the weak form.

Oracles:
* **path dependence** — a J2 elasto-plastic body loaded past yield then unloaded keeps a permanent set
  (an elastic body returns to zero); the march's stress-strain response is hysteretic.
* **correctness** — a single monotonic (proportional) increment from the virgin state reproduces the
  one-shot deformation-theory solve to solver tolerance (the return map is right).
* **differentiability** — ``∂(final state)/∂(material parameter)`` flows through the whole scan (FD-checked).
* **generality** — a BDF2 integrator reading ``u.i(-1)``, ``u.i(-2)`` (the *primary* unknown, auto-
  buffered — no ``.evolves``) marches a decay-to-source correctly: depth-2 history + the primary path.
* **fail-loud** — a ``.i(k)`` form on a non-``tau`` domain, and an internal state read at ``.i(-1)`` with
  no ``.evolves``, both error clearly.
* **multifield** — a coupled march with an *inert* second field reproduces the one-field march in its own
  block (and the second field's standalone solve in the other), and a genuinely coupled march carries an
  *irreversible* per-quadrature-point state that both fields see.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64. The session default is x64-off (see tests/conftest.py), and this
    flag is process-wide -- save/restore keeps it from leaking to whatever module runs next."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


import jno

E, NU, SY, H = 200.0, 0.3, 2.0, 20.0
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
K = LAM + 2 * MU / 3
RT = np.sqrt(1.5)


def _aliases():
    n = jno.np
    return n.sym, n.grad, n.trace, n.inner, n.sqrt, n.maximum, n.identity


def _j2_stress(u, X, ep_hist, al_hist, *, sy=SY):
    """J2 radial-return Cauchy stress as a trace formula: elastic predictor about the previous plastic
    state ``ep_hist`` (an ``ep.i(-1)``), corrected by the plastic multiplier ``dg``. Returns
    ``(sigma, dg, n_dir)`` so the evolution terms can reuse ``dg`` and the flow direction."""
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    I3 = identity(3)
    eps = lambda w: sym(grad(w, X))
    dev = lambda A: A - trace(A) / 3 * I3
    nrm = lambda A: sqrt(maximum(inner(A, A, 2), 0) + 1e-30)
    ee = eps(u) - ep_hist
    D = dev(ee)
    dd = nrm(D)
    dg = maximum(RT * 2 * MU * dd - (sy + H * al_hist), 0) / (3 * MU + H)
    n_dir = D / dd
    sig = K * trace(ee) * I3 + 2 * MU * D - 2 * MU * RT * dg * n_dir
    return sig, dg, n_dir


def _plastic_march_fem(nstep, peak, *, unload=True, size=0.5, sy=SY):
    """A clamped-base unit cube under a +z body load that ramps with τ (triangular if ``unload`` else
    monotonic 0→peak). Returns the assembled ``jno.fem`` (a pseudo-time march)."""
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain(tau=(0.0, 1.0, nstep))
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co = d.variable("interior", split=True)
    cb = d.variable("bot", split=True)
    X, tau = [co[0], co[1], co[2]], co[-1]
    u, phi = d.fem_symbols(value_shape=(3,))
    ep, _ = d.fem_symbols(value_shape=(3, 3))
    al, _ = d.fem_symbols(value_shape=())
    eps = lambda w: sym(grad(w, X))
    sig, dg, n_dir = _j2_stress(u, X, ep.i(-1), al.i(-1), sy=sy)
    zhat = jno.np.asarray([0.0, 0.0, 1.0])
    ramp = tau if not unload else (1.0 - jno.np.abs(2 * tau - 1.0))
    P = peak * ramp
    return jno.fem(
        [
            inner(sig, eps(phi), 2) - P * inner(zhat, phi, 1),
            ep.evolves(ep.i(-1) + RT * dg * n_dir),
            al.evolves(al.i(-1) + dg),
            *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)],
        ]
    )


# --------------------------------------------------------------------------------------------------
# Oracle 1 — path dependence: load past yield, unload, keep a permanent set.
# --------------------------------------------------------------------------------------------------
def test_load_unload_leaves_permanent_set():
    traj = np.asarray(_plastic_march_fem(41, peak=5.0).solve())  # nothing passed
    peak_per_step = np.abs(traj).max(axis=1)
    loaded = peak_per_step.max()
    unloaded = peak_per_step[-1]  # τ = 1, load back to zero
    assert loaded > 1e-4, "the body never deformed — load too small to yield"
    # A permanent set remains after unloading (plastic), but it is well below the loaded peak (the
    # elastic part recovered). An elastic body would return to ~0.
    assert 0.05 < unloaded / loaded < 0.95, f"expected a partial permanent set, got {unloaded / loaded:.3f}"
    # The response is a loop: displacement rises to the mid-path peak then falls on unloading.
    assert peak_per_step.argmax() not in (0, len(peak_per_step) - 1)


def test_elastic_control_returns_to_zero():
    """Same load path with NO plasticity (linear elasticity) returns to zero at τ=1 — isolating that the
    permanent set above is the plastic contribution, not a marching artefact."""
    sym, grad, trace, inner, _sqrt, _max, identity = _aliases()
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain(tau=(0.0, 1.0, 21))
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co = d.variable("interior", split=True)
    cb = d.variable("bot", split=True)
    X, tau = [co[0], co[1], co[2]], co[-1]
    u, phi = d.fem_symbols(value_shape=(3,))
    # A dummy internal state read once so the march triggers, but elastic stress ignores it.
    ep, _ = d.fem_symbols(value_shape=(3, 3))
    eps = lambda w: sym(grad(w, X))
    I3 = identity(3)
    sig = LAM * trace(eps(u)) * I3 + 2 * MU * eps(u) + 0.0 * ep.i(-1)
    zhat = jno.np.asarray([0.0, 0.0, 1.0])
    P = 5.0 * (1.0 - jno.np.abs(2 * tau - 1.0))
    traj = np.asarray(
        jno.fem(
            [
                inner(sig, eps(phi), 2) - P * inner(zhat, phi, 1),
                ep.evolves(ep.i(-1)),  # frozen state — elastic, no evolution
                *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)],
            ]
        ).solve()
    )
    peak_per_step = np.abs(traj).max(axis=1)
    assert peak_per_step.max() > 1e-4
    assert peak_per_step[-1] / peak_per_step.max() < 1e-6, "elastic body must return to zero on unload"


# --------------------------------------------------------------------------------------------------
# Oracle 2 — correctness: a monotonic proportional increment from virgin state == deformation theory.
# --------------------------------------------------------------------------------------------------
def test_first_increment_matches_deformation_theory():
    # 2-point grid [0, 1]: τ=0 gives zero load (virgin), τ=1 solves the full load with ZERO history —
    # so the flow-theory return map collapses to the one-shot deformation-theory solve, exactly.
    peak = 4.0
    march = np.asarray(_plastic_march_fem(2, peak=peak, unload=False).solve())[-1]

    # Deformation-theory reference: the same BVP with ep=0 baked (no history, plain steady solve).
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co = d.variable("interior", split=True)
    cb = d.variable("bot", split=True)
    X = [co[0], co[1], co[2]]
    u, phi = d.fem_symbols(value_shape=(3,))
    I3 = identity(3)
    eps = lambda w: sym(grad(w, X))
    dev = lambda A: A - trace(A) / 3 * I3
    nrm = lambda A: sqrt(maximum(inner(A, A, 2), 0) + 1e-30)
    ee = eps(u)
    D = dev(ee)
    dd = nrm(D)
    dg = maximum(RT * 2 * MU * dd - SY, 0) / (3 * MU + H)
    sig = K * trace(ee) * I3 + 2 * MU * D - 2 * MU * RT * dg * (D / dd)
    zhat = jno.np.asarray([0.0, 0.0, 1.0])
    ref = np.asarray(
        jno.fem(
            [inner(sig, eps(phi), 2) - peak * inner(zhat, phi, 1), *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)]]
        ).solve(nonlinear=jno.solve.newton(max_steps=60, rtol=1e-11, atol=1e-13))
    )
    assert np.allclose(march, ref, atol=1e-7, rtol=1e-6), f"max |march - deform| = {np.abs(march - ref).max():.2e}"


# --------------------------------------------------------------------------------------------------
# Oracle 3 — differentiability: ∂(final unloaded state)/∂sy flows through the whole scan.
# --------------------------------------------------------------------------------------------------
def test_gradient_flows_through_march_to_material_parameter():
    from jno.utils.solver.newton_krylov import newton_krylov

    syP = jno.np.parameter((1,), name="sy")
    fem = _plastic_march_fem(11, peak=5.0, sy=jno.np.reshape(syP, ()))
    op = fem._op
    assert op.is_parametric, "a jno.np.parameter in the form must make the march op parametric"
    # The parametric march returns a differentiable trace node (composes with crux), not an array.
    assert type(fem.solve()).__name__ == "FunctionCall"

    tau_grid = jnp.asarray(np.asarray(fem.domain._time_points))
    buffers0 = {k: jnp.zeros(s["shape"]) for k, s in op.history_specs.items()}
    u0 = jnp.zeros(int(op.size))

    def final_state_norm(sy):
        args_p = {"sy": jnp.reshape(sy, (1,))}

        def step(carry, tau_k):
            u_prev, bufs = carry
            args = {"__history__": bufs, **args_p}
            u = newton_krylov(lambda uu: op.residual(uu, args, tau_k), u_prev)
            ns = op.state_readout(u, tau_k, args)
            nb = {k: jnp.concatenate([ns[k][:, :, None, ...], bufs[k][:, :, :-1, ...]], axis=2) for k in bufs}
            return (u, nb), u

        _f, traj = jax.lax.scan(step, (u0, buffers0), tau_grid)
        return jnp.sum(traj[-1] ** 2)

    g = jax.grad(final_state_norm)(2.0)
    assert np.isfinite(g) and abs(g) > 0, "gradient w.r.t. sy vanished or is non-finite"
    fd = (final_state_norm(2.0 + 1e-4) - final_state_norm(2.0 - 1e-4)) / 2e-4
    assert np.allclose(g, fd, rtol=2e-3, atol=1e-6), f"AD grad {g:.6e} vs FD {fd:.6e}"


# --------------------------------------------------------------------------------------------------
# Oracle 4 — generality: BDF2 (primary-unknown history, depth 2, auto-buffered — no `.evolves`).
# --------------------------------------------------------------------------------------------------
def test_bdf2_primary_unknown_history_marches_to_static():
    # A BDF2-in-pseudo-time OVER-DAMPED elasticity: c·du/dt + ∇·σ(u) = load, with the rate discretized
    # BDF2 as (3u - 4u.i(-1) + u.i(-2))/(2 dt). Here u is the PRIMARY unknown (it carries the test φ), so
    # u.i(-1)/u.i(-2) are ITS history — auto-buffered from the just-solved u (NO `.evolves`), depth 2.
    # As t→∞ the rate vanishes and the march must relax to the STATIC elastic solution — a clean oracle
    # that exercises the depth-2 buffer roll and the primary-unknown path, on non-plasticity physics.
    sym, grad, trace, inner, _sqrt, _max, identity = _aliases()
    T, nstep, c = 6.0, 61, 3.0
    dt = T / (nstep - 1)
    I3 = identity(3)
    zhat = jno.np.asarray([0.0, 0.0, 1.0])

    def _sig(u, X):
        eps = lambda w: sym(grad(w, X))
        return K * trace(eps(u)) * I3 + 2 * MU * (eps(u) - trace(eps(u)) / 3 * I3)

    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain(tau=(0.0, T, nstep))
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co = d.variable("interior", split=True)
    cb = d.variable("bot", split=True)
    X = [co[0], co[1], co[2]]
    u, phi = d.fem_symbols(value_shape=(3,))
    eps = lambda w: sym(grad(w, X))
    rate = (3 * u - 4 * u.i(-1) + u.i(-2)) / (2 * dt)  # BDF2 du/dt — reads depth-2 primary history
    fem = jno.fem(
        [
            c * inner(rate, phi, 1) + inner(_sig(u, X), eps(phi), 2) - 2.0 * inner(zhat, phi, 1),
            *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)],
        ]
    )
    assert list(fem._op.history_roles.values()) == ["primary"], "u.i(-k) must be classified as primary-unknown history"
    assert [s["depth"] for s in fem._op.history_specs.values()] == [2], "u.i(-2) requires a depth-2 buffer"
    traj = np.asarray(fem.solve())  # nothing passed; u.i(-1)/u.i(-2) auto-buffered from the solved u

    # Static reference: c·du/dt → 0, the plain elastic BVP under the same load.
    dS = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain()
    dS.tag("bot", lambda x, y, z: z < 1e-6)
    co = dS.variable("interior", split=True)
    cb = dS.variable("bot", split=True)
    X = [co[0], co[1], co[2]]
    u, phi = dS.fem_symbols(value_shape=(3,))
    eps = lambda w: sym(grad(w, X))
    static = np.asarray(
        jno.fem(
            [
                inner(_sig(u, X), eps(phi), 2) - 2.0 * inner(zhat, phi, 1),
                *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)],
            ]
        ).solve()
    )
    rel = np.linalg.norm(traj[-1] - static) / np.linalg.norm(static)
    assert rel < 1e-2, f"BDF2 relaxation did not reach the static solution: rel_err={rel:.3e}"


# --------------------------------------------------------------------------------------------------
# Oracle 4b — SURFACE-quadrature-point history: a state read/evolved on a boundary term marches on the
# face quadrature points (the enabler for stick/slip friction — slip is a surface state).
# --------------------------------------------------------------------------------------------------
def test_surface_qp_history_accumulates_over_the_march():
    # A scalar state `acc` lives on the TOP face; it reads its past via `acc.i(-1)` in a boundary term and
    # advances by `acc.evolves(acc.i(-1) + 1)` — so at step k it equals k, driving a downward surface
    # traction of magnitude k. With a (nonlinear-but-never-yielding, i.e. elastic) bulk, the top deflects
    # by ∝ k: the per-step peak grows linearly, proving read + evolve + roll all work on face QPs.
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    N = 6
    d = jno.Shape.box(0, 0, 0, 3, 3, 2, size=0.9).domain(tau=(0.0, 1.0, N))
    d.tag("bot", lambda x, y, z: z < 1e-6)
    d.tag("top", lambda x, y, z: z > 2 - 1e-6)
    co = d.variable("interior", split=True)
    cb = d.variable("bot", split=True)
    tp = d.variable("top", split=True)
    xt, yt, zt = tp[0], tp[1], tp[2]
    X, I3 = [co[0], co[1], co[2]], identity(3)
    u, phi = d.fem_symbols(value_shape=(3,))
    acc, _ = d.fem_symbols(value_shape=())  # a SCALAR surface state on 'top'
    eps = lambda w: sym(grad(w, X))
    dev = lambda A: A - trace(A) / 3 * I3
    nrm = lambda A: sqrt(maximum(inner(A, A, 2), 0) + 1e-30)
    ee = eps(u)  # nonlinear op via the return map, but SY huge => never yields (elastic response)
    D = dev(ee)
    dd = nrm(D)
    dg = maximum(RT * 2 * MU * dd - 1e6, 0) / (3 * MU)
    sig = K * trace(ee) * I3 + 2 * MU * D - 2 * MU * RT * dg * (D / dd)
    phit = phi.bind(x=xt, y=yt, z=zt)
    zhat = jno.np.asarray([0.0, 0.0, 1.0])
    fem = jno.fem(
        [
            inner(sig, eps(phi), 2),
            acc.i(-1) * inner(zhat, phit, 1),  # surface traction from the accumulated state
            acc.evolves(acc.i(-1) + 1.0),  # the state advances on the face QPs
            *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)],
        ]
    )
    assert list(fem._op.surface_history_specs.values())  # 'acc' buffered on faces, not cells
    assert not fem._op.history_specs  # nothing on cell QPs
    peak = np.abs(np.asarray(fem.solve())).max(axis=1)
    assert peak[0] < 1e-9  # step 0: acc.i(-1)=0 -> no load
    ratios = peak[1:] / np.arange(1, N)
    assert np.ptp(ratios) < 1e-3, f"surface state did not accumulate linearly: peak/k = {ratios}"


# --------------------------------------------------------------------------------------------------
# Oracle 4c — stick/slip Coulomb FRICTION as a return map on the friction cone (the J2 analogue), on a
# surface slip state. Under a growing tangential drag the interface STICKS (traction rises elastically)
# then SLIPS: the traction — and hence the shear it produces — CAPS at μ·p_n instead of running away with
# the drag. That saturation is the defining friction feature and the accumulated slip is permanent. (The
# tangential displacement is read as inner(x̂, u.bind(top)); a component-indexed bound view u.bind(top)[0]
# isn't yet resolved inside a surface readout formula — a known follow-up.)
# --------------------------------------------------------------------------------------------------
def test_stick_slip_friction_caps_at_the_cone():
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    P0, MUF, k_t, D, N = 4.0, 0.3, 400.0, 0.06, 25  # normal pressure, friction coeff, tangential stiffness, drag, steps
    Lz, G = 2.0, MU  # MU (module constant) is the elastic shear modulus; MUF is the friction coefficient
    d = jno.Shape.box(0, 0, 0, 3, 3, Lz, size=0.9).domain(tau=(0.0, 1.0, N))
    d.tag("bot", lambda x, y, z: z < 1e-6)
    d.tag("top", lambda x, y, z: z > Lz - 1e-6)
    co = d.variable("interior", split=True)
    cb = d.variable("bot", split=True)
    tp = d.variable("top", split=True)
    xt, yt, zt, taut = tp[0], tp[1], tp[2], tp[-1]
    X, I3 = [co[0], co[1], co[2]], identity(3)
    u, phi = d.fem_symbols(value_shape=(3,))
    slip, _ = d.fem_symbols(value_shape=())  # scalar tangential slip — a SURFACE state
    eps = lambda w: sym(grad(w, X))
    xhat = jno.np.asarray([1.0, 0.0, 0.0])
    zhat = jno.np.asarray([0.0, 0.0, 1.0])
    phit = phi.bind(x=xt, y=yt, z=zt)
    sig = LAM * trace(eps(u)) * I3 + 2 * MU * eps(u)
    d_tau = D * taut  # obstacle x-drag: monotonic 0 -> D
    g = inner(xhat, u.bind(x=xt, y=yt, z=zt), 1) - d_tau  # tangential relative displacement
    t_tr = k_t * (g - slip.i(-1))  # elastic trial tangential traction
    mag = sqrt(t_tr * t_tr + 1e-30)
    excess = maximum(mag - MUF * P0, 0.0)  # over the friction cone (radius μ·p_n)
    t_ret = t_tr - excess * (t_tr / mag)  # returned traction, |t_ret| <= μ·p_n
    fem = jno.fem(
        [
            inner(sig, eps(phi), 2),
            P0 * inner(zhat, phit, 1),  # constant normal pressure
            t_ret * inner(xhat, phit, 1),  # stick/slip friction traction (restores toward the obstacle)
            slip.evolves(slip.i(-1) + excess / k_t * (t_tr / mag)),  # slip advances by the return
            *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)],
        ]
    )
    assert bool(fem._op.surface_history_specs)  # 'slip' is a surface state
    traj = np.asarray(fem.solve(nonlinear=jno.solve.newton(max_steps=80, rtol=1e-9, atol=1e-11, line_search=True)))
    pts = np.asarray(getattr(d, "_fem_native_dof_points", d.mesh.points))[:, :3]
    top = np.where(pts[:, 2] > Lz - 1e-6)[0]
    ux = traj.reshape(N, -1, 3)[:, top, 0].mean(1)
    cap = MUF * P0 * Lz / G  # the shear a capped friction traction produces
    # STICK -> SLIP: u_x rises then saturates near the cap instead of tracking the (much larger) drag.
    assert ux[-1] > 0.5 * cap, "the contact never reached the slip (saturated) regime"
    assert np.abs(ux).max() < 2.0 * cap, "friction traction did not cap at the cone (u_x ran away with the drag)"
    # the last few steps barely move (slipping at the cap) — the tell of saturation, not elastic tracking
    assert abs(ux[-1] - ux[-4]) < 0.15 * cap, "u_x still tracking the drag — did not saturate"


# --------------------------------------------------------------------------------------------------
# Oracle 6 — MULTIFIELD: a coupled system marching a load path. The buffers are indexed by cell, never
# by field, so a second field must change nothing about the state — this pins that. The second field is
# *inert* (its own decoupled Poisson problem), so each block has an independent, already-trusted oracle:
# the u block must equal the single-field plastic march, the w block its own standalone steady solve.
# --------------------------------------------------------------------------------------------------
def test_multifield_march_leaves_each_block_at_its_own_oracle():
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    nstep, peak, size = 6, 5.0, 0.5

    def _mesh(**kw):
        m = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain(**kw)
        m.tag("bot", lambda x, y, z: z < 1e-6)
        return m

    d = _mesh(tau=(0.0, 1.0, nstep))
    co, cb = d.variable("interior", split=True), d.variable("bot", split=True)
    X, tau = [co[0], co[1], co[2]], co[-1]
    u, phi = d.fem_symbols(value_shape=(3,))
    w, psi = d.fem_symbols()  # the INERT second field — its own Poisson, no coupling to u
    ep, _ = d.fem_symbols(value_shape=(3, 3))
    al, _ = d.fem_symbols(value_shape=())
    eps = lambda v: sym(grad(v, X))
    sig, dg, n_dir = _j2_stress(u, X, ep.i(-1), al.i(-1))
    zhat = jno.np.asarray([0.0, 0.0, 1.0])
    P = peak * (1.0 - jno.np.abs(2 * tau - 1.0))
    fem = jno.fem(
        [
            inner(sig, eps(phi), 2) - P * inner(zhat, phi, 1),
            inner(grad(w, X), grad(psi, X), 1) - 1.0 * psi,
            ep.evolves(ep.i(-1) + RT * dg * n_dir),
            al.evolves(al.i(-1) + dg),
            *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)],
            w(cb[0], cb[1], cb[2]) - 0.0,
        ]
    )
    assert len(fem.blocks) == 2, "two fem_symbols() unknowns must assemble as two blocks"
    traj = np.asarray(fem.solve())  # nothing passed
    assert traj.shape[0] == nstep
    su, sw = fem.blocks[fem.block_index(u)], fem.blocks[fem.block_index(w)]

    # Block u — the same plastic march, solved alone.
    ref_u = np.asarray(_plastic_march_fem(nstep, peak=peak, size=size).solve())
    assert traj[:, su].shape == ref_u.shape
    assert np.abs(ref_u).max() > 1e-4, "the reference march never deformed — the comparison is vacuous"
    err_u = np.abs(traj[:, su] - ref_u).max() / np.abs(ref_u).max()
    assert err_u < 1e-6, f"the coupled u block drifted from the single-field march: rel {err_u:.2e}"

    # Block w — the inert field's own standalone steady solve, identical at every load step.
    dS = _mesh()
    co, cb = dS.variable("interior", split=True), dS.variable("bot", split=True)
    XS = [co[0], co[1], co[2]]
    w2, psi2 = dS.fem_symbols()
    ref_w = np.asarray(
        jno.fem([inner(grad(w2, XS), grad(psi2, XS), 1) - 1.0 * psi2, w2(cb[0], cb[1], cb[2]) - 0.0]).solve()
    )
    assert np.abs(ref_w).max() > 1e-4
    err_w = np.abs(traj[:, sw] - ref_w[None, :]).max() / np.abs(ref_w).max()
    assert err_w < 1e-6, f"the inert block is not its own standalone solution: rel {err_w:.2e}"


# --------------------------------------------------------------------------------------------------
# Oracle 6b — MULTIFIELD + a genuinely COUPLED, IRREVERSIBLE per-quadrature-point state. This is the
# phase-field shape: a damage field ``dm`` driven by ``H``, the running maximum of the elastic energy
# density of ``u``, with ``u``'s stiffness degraded by ``(1-dm)^2`` in return.
#
# The oracle is the A/B that isolates irreversibility from everything else: the ONLY difference between
# the two runs is ``maximum(H.i(-1), psi)`` vs ``psi`` in the update. Load up then back down. With the
# running max the damage must PERSIST at zero load (a crack does not heal); without it the damage must
# follow the load back down. Same mesh, same fields, same solver — so a difference can only come from
# the buffered state surviving the march and being seen by both fields.
# --------------------------------------------------------------------------------------------------
def _coupled_damage_march(*, irreversible, nstep=7, load=1.5, ell=0.4, gc=1.0, floor=0.3):
    """``(fem, damage_trajectory)`` for one leg of the A/B. ``irreversible`` selects the running max."""
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34).domain(tau=(0.0, 1.0, nstep))
    d.tag("left", lambda x, y: x < 1e-9)
    co, cl = d.variable("interior", split=True), d.variable("left", split=True)
    X, tau = [co[0], co[1]], co[-1]
    u, phi = d.fem_symbols()
    dm, q = d.fem_symbols()
    Hs, _ = d.fem_symbols(value_shape=())
    psi_e = 0.5 * inner(grad(u, X), grad(u, X), 1)  # elastic energy density (the crack driving force)
    # Degradation, floored. The floor does two jobs and only one of them is a bound's: it keeps the
    # damage feedback from running away (unbounded, this reaches dm ~ 1e6), AND it keeps the u block
    # non-singular at dm = 1, which `dm.bounds(0, 1)` does not fix — see tests/test_fem_bounds.py.
    deg = (1.0 - dm) ** 2 + floor
    ramp = 1.0 - jno.np.abs(2 * tau - 1.0)  # triangular: 0 -> 1 -> 0
    fem = jno.fem(
        [
            deg * inner(grad(u, X), grad(phi, X), 1) - load * ramp * phi,
            (gc / ell) * dm * q + gc * ell * inner(grad(dm, X), grad(q, X), 1) - 2.0 * (1.0 - dm) * Hs.i(-1) * q,
            Hs.evolves(maximum(Hs.i(-1), psi_e) if irreversible else psi_e),
            u(*cl) - 0.0,
        ]
    )
    # Resolve the damage block by SYMBOL — the field order is first appearance in the term walk, and the
    # degradation puts `dm` ahead of `u`, so a hardcoded index reads the wrong block.
    return fem, np.asarray(fem.solve())[:, fem.blocks[fem.block_index(dm)]]


def test_coupled_damage_state_is_irreversible_only_with_the_running_max():
    irr, d_irr = _coupled_damage_march(irreversible=True)
    _rev, d_rev = _coupled_damage_march(irreversible=False)

    peak = np.abs(d_irr).max()
    assert 1e-2 < peak < 1.0, f"the damage must grow into a physical range to mean anything (got {peak:.3f})"
    assert d_irr.min() > -1e-12, "damage went negative — the regime is not the one being tested"

    # On the RISING branch the running max IS the current value, so the two legs must coincide exactly:
    # this pins that the A/B differs in nothing but the `maximum`.
    mid = d_irr.shape[0] // 2  # τ = 0.5, peak load
    assert np.abs(d_irr[mid] - d_rev[mid]).max() / peak < 1e-9, "the two updates must coincide on loading"

    # At τ=1 the load is back to zero. Irreversible: the damage is retained. Reversible: it follows the
    # load back down (not to exactly zero — the update is lagged one step, so it trails by one load level).
    assert np.abs(d_irr[-1]).max() / peak > 0.999, "the running-max state did not survive unloading"
    assert np.abs(d_rev[-1]).max() / peak < 0.35, "the control did not heal — the A/B proves nothing"

    # Monotone non-decreasing under the running max — the defining property of the irreversible state.
    assert np.all(np.diff(np.abs(d_irr).max(axis=1)) > -1e-9)

    # And the state itself is a VOLUME (cell-quadrature) buffer, seen by a form with two solved fields.
    assert len(irr.blocks) == 2 and len(irr._op.history_specs) == 1
    assert not irr._op.surface_history_specs


# --------------------------------------------------------------------------------------------------
# Oracle 6d — differentiability of the COUPLED march. A parametric coupled steady form is otherwise
# refused (the coupled *linear* assembly has no parametric route), but a history-carrying form never
# takes that route — it assembles as a residual operator that re-evaluates at the runtime args, which is
# field-agnostic. Without this the coupled march could not do the material-identification inverse the
# single-field march already does. FD-checked through the whole scan, both blocks in the objective.
# --------------------------------------------------------------------------------------------------
def test_gradient_flows_through_a_coupled_march_to_a_material_parameter():
    from jno.utils.solver.newton_krylov import newton_krylov

    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain(tau=(0.0, 1.0, 4))
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X, tau = [co[0], co[1]], co[-1]
    kP = jno.np.reshape(jno.np.parameter((1,), name="k"), ())  # the material parameter to recover
    u, phi = d.fem_symbols()
    dm, q = d.fem_symbols()
    Hs, _ = d.fem_symbols(value_shape=())
    fem = jno.fem(
        [
            inner(grad(u, X), grad(phi, X), 1) + 0.1 * u * u * phi - (1.0 + tau) * phi - kP * Hs.i(-1) * phi,
            2.5 * dm * q + 0.4 * inner(grad(dm, X), grad(q, X), 1) - Hs.i(-1) * q,
            Hs.evolves(maximum(Hs.i(-1), u)),
            u(*cb) - 0.0,
        ]
    )
    op = fem._op
    assert len(fem.blocks) == 2 and op.is_parametric, "a coupled parametric march must build as parametric"
    # The parametric march returns a differentiable trace node (composes with crux), not an array.
    assert type(fem.solve()).__name__ == "FunctionCall"

    tau_grid = jnp.asarray(np.asarray(d._time_points))
    buffers0 = {k: jnp.zeros(v["shape"]) for k, v in op.history_specs.items()}
    u0 = jnp.zeros(int(op.size))

    def final_norm(kv):
        args_p = {"k": jnp.reshape(kv, (1,))}

        def step(carry, tau_k):
            u_prev, bufs = carry
            args = {"__history__": bufs, **args_p}
            uu = newton_krylov(lambda z: op.residual(z, args, tau_k), u_prev)
            ns = op.state_readout(uu, tau_k, args)
            nb = {k: jnp.concatenate([ns[k][:, :, None, ...], bufs[k][:, :, :-1, ...]], axis=2) for k in bufs}
            return (uu, nb), uu

        _f, traj = jax.lax.scan(step, (u0, buffers0), tau_grid)
        return jnp.sum(traj[-1] ** 2)  # both blocks — the gradient must reach the coupled field too

    g = jax.grad(final_norm)(0.3)
    assert np.isfinite(g) and abs(g) > 0, "gradient w.r.t. the coupled march's parameter vanished"
    fd = (final_norm(0.3 + 1e-5) - final_norm(0.3 - 1e-5)) / 2e-5
    assert np.allclose(g, fd, rtol=1e-6), f"AD grad {g:.8e} vs FD {fd:.8e}"


# --------------------------------------------------------------------------------------------------
# Oracle 6c — a form LINEAR in every unknown but carrying step history. Every load step is a different
# linear system whose source the buffer sets, so it is still a march — but the linear assembly route
# builds a matrix/rhs pair with no history to thread, and this used to raise at BUILD time. It is on the
# phase-field critical path: the AT1 damage equation with a lagged driving force is exactly this shape.
#
# The oracle is exact superposition. With `s.evolves(s.i(-1) + 1)` the state at step k is exactly k, so
# each field solves its own linear problem at source ∝ (1+k) — and linearity makes the step-k solution
# exactly (1+k) times the step-0 one, at machine precision, in BOTH blocks (the second is driven only
# through the first, so it also pins the coupling).
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("coupled", [False, True])
def test_linear_form_with_step_history_marches_by_exact_superposition(coupled):
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    N = 5
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34).domain(tau=(0.0, 1.0, N))
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols(value_shape=())
    terms = [inner(grad(u, X), grad(phi, X), 1) - (1.0 + s.i(-1)) * phi, s.evolves(s.i(-1) + 1.0), u(*cb) - 0.0]
    if coupled:
        w, chi = d.fem_symbols()
        terms += [inner(grad(w, X), grad(chi, X), 1) - 2.0 * u * chi, w(*cb) - 0.0]
    fem = jno.fem(terms)
    assert getattr(fem._op, "state_readout", None) is not None, "a linear history form must still build a march op"
    traj = np.asarray(fem.solve())  # nothing passed
    assert traj.shape[0] == N

    for blk in fem.blocks:
        step = traj[:, blk]
        assert np.abs(step[0]).max() > 1e-4, "a block never responded — the comparison would be vacuous"
        # u_k = (1+k)·u_0 exactly: the state is k, the equations are linear, so the solution scales.
        scaled = step / (1.0 + np.arange(N))[:, None]
        rel = np.abs(scaled - scaled[0]).max() / np.abs(scaled[0]).max()
        assert rel < 1e-9, f"superposition broke: the march is not linear in the state, rel {rel:.2e}"

    # And step 0 (virgin state, s = 0) is the plain steady solve of the same BVP.
    dS = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34).domain()
    dS.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = dS.variable("interior", split=True), dS.variable("bdry", split=True)
    XS = [co[0], co[1]]
    u2, phi2 = dS.fem_symbols()
    ref = np.asarray(jno.fem([inner(grad(u2, XS), grad(phi2, XS), 1) - 1.0 * phi2, u2(*cb) - 0.0]).solve())
    got = traj[0, fem.blocks[fem.block_index(u)]]
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-8, "the virgin step is not the plain steady solve"


# --------------------------------------------------------------------------------------------------
# Oracle 5 — fail loud.
# --------------------------------------------------------------------------------------------------
def test_history_form_on_non_tau_domain_fails_loud():
    # The plastic physics reads `.i(-1)`, but on a plain (no-`tau`) domain `fem.solve()` has no load path
    # to march over — expect a clear error naming the fix (`domain(tau=...)`).
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain()  # NO tau grid
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co = d.variable("interior", split=True)
    cb = d.variable("bot", split=True)
    X = [co[0], co[1], co[2]]
    u, phi = d.fem_symbols(value_shape=(3,))
    ep, _ = d.fem_symbols(value_shape=(3, 3))
    al, _ = d.fem_symbols(value_shape=())
    eps = lambda w: sym(grad(w, X))
    sig, dg, n_dir = _j2_stress(u, X, ep.i(-1), al.i(-1))
    zhat = jno.np.asarray([0.0, 0.0, 1.0])
    fem2 = jno.fem(
        [
            inner(sig, eps(phi), 2) - 5.0 * inner(zhat, phi, 1),
            ep.evolves(ep.i(-1) + RT * dg * n_dir),
            al.evolves(al.i(-1) + dg),
            *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)],
        ]
    )
    with pytest.raises((ValueError, NotImplementedError), match="pseudo-time|tau"):
        fem2.solve()


def test_coupled_transient_with_evolves_fails_loud():
    # Lifting the single-field restriction exposed a new route: a COUPLED form reaches the multifield
    # assembler before the single-field transient check, so the `u.t` rejection has to exist there too.
    # Without it the evolution terms would be silently dropped on a coupled transient.
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain(time=(0.0, 0.1, 4))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    u, phi = d.fem_symbols(names=("u", "phi"))
    w, chi = d.fem_symbols(names=("w", "chi"))
    s, _ = d.fem_symbols(value_shape=())
    ui, pi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    wi, qi = w.bind(x=xi, y=yi, t=ti), chi.bind(x=xi, y=yi, t=ti)
    with pytest.raises(NotImplementedError, match="tau"):
        jno.fem(
            [
                ui.t * pi + (ui.x * pi.x + ui.y * pi.y) - (1.0 + s.i(-1)) * pi,
                wi.t * qi + (wi.x * qi.x + wi.y * qi.y) - ui * qi,
                s.evolves(s.i(-1) + 1.0),
                u(xb, yb) - 0.0,
                w(xb, yb) - 0.0,
                u(ci[0], ci[1]) - 0.0,
                w(ci[0], ci[1]) - 0.0,
            ]
        )


def test_internal_state_without_evolves_fails_loud():
    # Reads ep.i(-1) in the weak form but declares NO ep.evolves — the buffer would stay frozen at zero
    # (a silently wrong deformation-theory result). Must be a hard build error.
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain(tau=(0.0, 1.0, 4))
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co = d.variable("interior", split=True)
    cb = d.variable("bot", split=True)
    X = [co[0], co[1], co[2]]
    u, phi = d.fem_symbols(value_shape=(3,))
    ep, _ = d.fem_symbols(value_shape=(3, 3))
    al, _ = d.fem_symbols(value_shape=())
    eps = lambda w: sym(grad(w, X))
    sig, dg, n_dir = _j2_stress(u, X, ep.i(-1), al.i(-1))
    zhat = jno.np.asarray([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="evolves"):
        jno.fem(
            [
                inner(sig, eps(phi), 2) - 5.0 * inner(zhat, phi, 1),
                # ep.evolves(...) and al.evolves(...) DELIBERATELY omitted
                *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)],
            ]
        )


# --------------------------------------------------------------------------------------------------
# Oracle 5 — a SCALAR volume state whose evolution reads the BARE trial. This is the phase-field shape
# (`H.evolves(maximum(H.i(-1), psi(u)))`) and it is a *relationship* test, not a restatement: a scalar
# field's interpolated value and a scalar state's buffer slice must agree on rank. They did not — a
# bare scalar trial carried a spurious `(n_quad, 1)` component axis (invisible in a weak term, where it
# contracts with the test function) while the buffer slice is `(n_quad,)`, so `maximum(H.i(-1), u)`
# rank-broadcast to `(n_quad, n_quad)` and the readout wrote a buffer of the wrong rank.
# --------------------------------------------------------------------------------------------------
def test_scalar_state_reading_the_bare_trial_keeps_its_rank():
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    N = 4
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain(tau=(0.0, 1.0, N))
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X, tau = [co[0], co[1]], co[-1]
    u, phi = d.fem_symbols()
    Hs, _ = d.fem_symbols(value_shape=())  # a SCALAR volume state

    # Nonlinear (so the form takes the march route), driven up with tau; Hs tracks the running max of u.
    weak = inner(grad(u, X), grad(phi, X), 1) + 0.1 * u * u * phi - (1.0 + tau) * phi
    fem = jno.fem([weak, Hs.evolves(maximum(Hs.i(-1), u)), u(*cb) - 0.0])

    spec = fem._op.history_specs[next(iter(fem._op.history_specs))]
    n_cells, n_quad = int(spec["shape"][0]), int(spec["shape"][1])
    assert spec["value_shape"] == ()  # declared scalar...

    traj = np.asarray(fem.solve())
    assert traj.shape[0] == N

    # THE RELATIONSHIP: the readout's per-state array must match the buffer's declared rank exactly.
    out = fem._op.state_readout(
        jnp.asarray(traj[-1]), 1.0, {"__history__": {k: jnp.zeros(v["shape"]) for k, v in fem._op.history_specs.items()}}
    )
    got = np.asarray(next(iter(out.values())))
    assert got.shape == (n_cells, n_quad), f"scalar state readout is {got.shape}, buffer wants {(n_cells, n_quad)}"

    # ...and the physics: with a monotonically rising load the running max IS the current value.
    u_last = np.asarray(traj[-1])
    assert u_last.max() > 0.0
    assert np.all(np.diff(np.abs(traj).max(axis=1)) > -1e-12)  # monotone, so max_t u == u(t_end)


# --------------------------------------------------------------------------------------------------
# Oracle 5b — the same relationship for a scalar field's GRADIENT. ``value_shape == ()`` promises no
# component axis in the value (above) and equally in the gradient, but the Lagrange interpolation built
# ``(n_quad, 1, n_dims)``, so ``inner(grad(u,X), grad(u,X), 1)`` came out ``(n_quad, 1)`` — and
# ``maximum(H.i(-1), psi)`` against a ``(n_quad,)`` buffer slice rank-broadcast to ``(n_quad, n_quad)``.
# That expression IS the phase-field driving force. Checked here against an EXTERNAL oracle as well as
# the rank: on affine P1 triangles ∇u is constant per cell and follows in closed form from the vertex
# values, so the readout's energy density has an exact hand-computed reference.
# --------------------------------------------------------------------------------------------------
def test_scalar_gradient_energy_readout_matches_the_closed_form_p1_gradient():
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    N = 3
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain(tau=(0.0, 1.0, N))
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X, tau = [co[0], co[1]], co[-1]
    u, phi = d.fem_symbols()
    Hs, _ = d.fem_symbols(value_shape=())
    psi_e = 0.5 * inner(grad(u, X), grad(u, X), 1)  # the phase-field driving force
    weak = inner(grad(u, X), grad(phi, X), 1) + 0.1 * u * u * phi - (1.0 + tau) * phi
    fem = jno.fem([weak, Hs.evolves(maximum(Hs.i(-1), psi_e)), u(*cb) - 0.0])

    spec = fem._op.history_specs[next(iter(fem._op.history_specs))]
    n_cells, n_quad = int(spec["shape"][0]), int(spec["shape"][1])
    assert spec["value_shape"] == ()

    traj = np.asarray(fem.solve())
    out = fem._op.state_readout(
        jnp.asarray(traj[-1]), 1.0, {"__history__": {k: jnp.zeros(v["shape"]) for k, v in fem._op.history_specs.items()}}
    )
    got = np.asarray(next(iter(out.values())))
    assert got.shape == (n_cells, n_quad), f"gradient-energy readout is {got.shape}, buffer wants {(n_cells, n_quad)}"

    # External oracle: on an affine P1 triangle u is linear, so ∇u solves A g = Δu exactly, where A's
    # rows are the two edge vectors from vertex 0 and Δu the matching vertex-value differences.
    pts = np.asarray(d._fem_native_dof_points)[:, :2]
    cells = np.asarray(d._fem_native_assembly_cells)
    uv = np.asarray(traj[-1])[cells]  # (n_cells, 3) vertex values
    p = pts[cells]  # (n_cells, 3, 2)
    A = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=1)  # (n_cells, 2, 2) edge vectors as rows
    du = np.stack([uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]], axis=1)  # (n_cells, 2)
    g = np.linalg.solve(A, du[..., None])[..., 0]  # (n_cells, 2)
    ref = 0.5 * np.sum(g * g, axis=1)  # constant over each affine cell
    assert ref.max() > 1e-6, "the solution is flat — the gradient oracle would be vacuous"
    rel = np.abs(got - ref[:, None]).max() / ref.max()
    assert rel < 1e-10, f"gradient energy density disagrees with the closed-form P1 gradient: rel {rel:.2e}"


# --------------------------------------------------------------------------------------------------
# Oracle 6e — a τ-DEPENDENT essential value on the march: `u(top) - delta*tau`, i.e. DISPLACEMENT
# CONTROL. It is how a softening test is driven at all — under load control a specimen snaps at the
# peak and there is no branch left to follow — so it is not an optional spelling.
#
# It used to be collected and then silently DROPPED: the constraint vanished, and the solve returned
# u = 0, which is exactly what an un-loaded specimen looks like. The oracle is analytic: a clamped-base
# strip pulled at the top by delta*tau is in uniaxial extension, so every step's grip displacement is
# delta*tau_k exactly and the reaction is proportional to it.
# --------------------------------------------------------------------------------------------------
def test_tau_dependent_dirichlet_drives_the_march():
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    N, DELTA, LY = 5, 0.02, 1.0
    d = jno.Shape.rect(0.0, 0.0, 0.5, LY, size=0.15).domain(tau=(0.0, 1.0, N))
    d.tag("bot", lambda x, y: y < 1e-9)
    d.tag("top", lambda x, y: y > LY - 1e-9)
    co, cb, ct = (d.variable(r, split=True) for r in ("interior", "bot", "top"))
    X, I2 = [co[0], co[1]], identity(2)
    u, phi = d.fem_symbols(value_shape=(2,))
    s, _ = d.fem_symbols(value_shape=())
    eps = lambda w: sym(grad(w, X))
    sig = LAM * trace(eps(u)) * I2 + 2 * MU * eps(u)
    momentum = inner(sig, eps(phi), 2)
    fem = jno.fem(
        [
            momentum + 0.0 * s.i(-1) * inner(jno.np.asarray([0.0, 1.0]), phi, 1),
            s.evolves(s.i(-1)),
            u(*cb)[0] - 0.0,
            u(*cb)[1] - 0.0,
            u(*ct)[0] - 0.0,
            u(*ct)[1] - DELTA * ct[-1],  # the ramp: the TOP region's own tau
        ]
    )
    traj = np.asarray(fem.solve())
    taus = np.asarray(d._time_points)

    grip = fem.region_dofs("top", field=u, component=1)
    got = traj[:, grip]
    assert np.abs(got).max() > 1e-6, "the specimen never moved — the ramped constraint was dropped"
    want = DELTA * taus
    # The pin is a residual row `u[d] - g`, so it is satisfied to the NEWTON tolerance (atol=1e-8), not
    # exactly. Bound it well inside that rather than at round-off: 1e-12 happened to pass and was really
    # asserting where one particular iteration landed.
    assert np.abs(got - want[:, None]).max() < 1e-9, "the grip must sit on delta*tau at every step"

    # ...and the response is the uniaxial one: the reaction scales linearly with the imposed stretch.
    R = np.array([float(np.asarray(fem.eval(momentum, traj[k]))[grip].sum()) for k in range(N)])
    nz = taus > 0
    ratio = R[nz] / taus[nz]
    # Bounded by the SOLVE, not by round-off: R is read off a solution converged to newton_krylov's
    # atol/rtol of 1e-8, so linearity cannot hold tighter than that however exact the arithmetic is.
    assert np.ptp(ratio) / np.abs(ratio).max() < 1e-7, f"the reaction must be linear in the load: {ratio}"
    assert np.abs(R).max() > 1e-6


def test_tau_dependent_dirichlet_off_the_march_fails_loud():
    """A path that cannot thread the ramp must say so — dropping it returns a plausible u = 0."""
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain(tau=(0.0, 1.0, 4))
    d.tag("bot", lambda x, y: y < 1e-9)
    d.tag("top", lambda x, y: y > 1 - 1e-9)
    co, cb, ct = (d.variable(r, split=True) for r in ("interior", "bot", "top"))
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    # Nonlinear, but NO history — so this is a plain steady solve, not a march, and there is no τ to
    # evaluate the ramp at.
    with pytest.raises(NotImplementedError, match="time/τ-dependent essential value|essential value"):
        jno.fem(
            [
                inner(grad(u, X), grad(phi, X), 1) + 0.1 * u * u * phi,
                u(*cb) - 0.0,
                u(*ct) - 0.5 * ct[-1],
            ]
        )


# --------------------------------------------------------------------------------------------------
# The march runs its per-step Newton inside `lax.scan`, where the driver's own convergence check
# self-disables (it needs a concrete residual). Without a check OUTSIDE the scan a diverged step is
# carried forward as the next step's state AND history, so one silent failure contaminates the whole
# remaining path — and the trajectory comes back finite and plausible.
# --------------------------------------------------------------------------------------------------


def _yeoh_march(*, order_u, line_search, n_steps=4, load=0.4):
    """A 3-D Yeoh phase-field march. Undamped Newton on this form overshoots into an INVERTED element
    (``J = det F <= 0``, so ``J**(-2/3)`` is NaN, which is absorbing) at P2 but not at P1 — the
    higher-order element's full step produces larger gradients at its extra quadrature points."""
    n = jno.np
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    det, einsum, diff = n.det, n.einsum, n.diff
    c10, c20, c30, nu = 0.024, 1e-3, 8e-5, 0.49
    kb = 2 * (2 * c10) * (1 + nu) / (3 * (1 - 2 * nu))
    W = 4.0

    d = jno.Shape.box(0, 0, 0, W, W, 0.4, size=W / 2).domain(tau=(0.0, 1.0, n_steps))
    d.tag("bot", lambda x, y, z: y < 1e-9)
    d.tag("top", lambda x, y, z: y > W - 1e-9)
    d.tag("bk", lambda x, y, z: z < 1e-9)
    co, cb, ct, cz = (d.variable(r, split=True) for r in ("interior", "bot", "top", "bk"))
    X, I3 = [co[0], co[1], co[2]], identity(3)

    u, phi = d.fem_symbols(value_shape=(3,), order=order_u)
    dm, q = d.fem_symbols(order=1)
    hs, _ = d.fem_symbols(order=1)

    F = I3 + grad(u, X)
    J = det(F)
    i1b = J ** (-2 / 3) * trace(einsum("...ki,...kj->...ij", F, F))
    psi = c10 * (i1b - 3) + c20 * (i1b - 3) ** 2 + c30 * (i1b - 3) ** 3
    hist = maximum(psi, hs.i(-1))
    P = diff(((1 - dm) ** 2 + 0.002) * psi + 0.5 * kb * (J - 1) ** 2, F)

    fem = jno.fem(
        [
            inner(P, grad(phi, X), 2),
            2.5 * dm * q + 0.4 * inner(grad(dm, X), grad(q, X), 1) - 2.0 * (1 - dm) * hist * q,
            hs.evolves(hist),
            dm.bounds(0.0, 1.0),
            u(*cb)[0] - 0.0,
            u(*cb)[1] - 0.0,
            u(*ct)[0] - 0.0,
            u(*ct)[1] - load * ct[-1],
            u(*cz)[2] - 0.0,
        ]
    )
    alt = jno.solve.staggered([u, dm], max_sweeps=40, rtol=1e-6, atol=1e-9, line_search=line_search)
    return fem, u, alt, load


@pytest.mark.slow
def test_march_refuses_a_step_that_did_not_converge():
    """The wrong-answer case: without globalization the P2 solve diverges, and the returned trajectory
    is finite and plausible — it reported |u| = 0.70 against a grip PINNED to 0.40."""
    fem, u, alt, load = _yeoh_march(order_u=2, line_search=False)
    with pytest.raises(RuntimeError, match="load-path march did not converge"):
        fem.solve(nonlinear=alt)


@pytest.mark.slow
def test_march_converges_once_the_step_solve_is_globalized():
    """...and the same form with a line search reaches the grip exactly — so the guard above is
    reporting a real failure, not refusing a solvable problem."""
    fem, u, alt, load = _yeoh_march(order_u=2, line_search=True)
    traj = np.asarray(fem.solve(nonlinear=alt))
    got = np.abs(traj[..., fem.blocks[fem.block_index(u)]]).max()
    assert abs(got - load) < 1e-6, f"the grip is pinned to {load}, got max|u| = {got}"


@pytest.mark.slow
def test_march_guard_passes_the_case_p1_already_solved():
    """P1 solves this undamped — the guard must not turn a working march into a failure."""
    fem, u, alt, load = _yeoh_march(order_u=1, line_search=False)
    traj = np.asarray(fem.solve(nonlinear=alt))
    got = np.abs(traj[..., fem.blocks[fem.block_index(u)]]).max()
    assert abs(got - load) < 1e-6, f"the grip is pinned to {load}, got max|u| = {got}"


def test_march_guard_scores_the_min_map_not_the_bare_residual():
    """An ACTIVE box constraint leaves the bare residual non-zero by construction — that non-zero IS the
    multiplier. Scoring a converged bounded march against ``op.residual`` would read a correct answer as
    a divergence, so the guard must ask the bounds wrapper for the function it actually root-found.

    Driven hard enough that the bound is genuinely active: without ``bounds`` the state would run past 1.
    """
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    N = 4
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain(tau=(0.0, 1.0, N))
    d.tag("left", lambda x, y: x < 1e-9)
    co, cl = (d.variable(r, split=True) for r in ("interior", "left"))
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols()

    # A source ramped well past what the upper bound allows: u is pushed toward 5*tau but capped at 1.
    fem = jno.fem(
        [
            inner(grad(u, X), grad(phi, X), 1) + 10.0 * (u - 5.0 * co[-1]) * phi + 0.0 * s.i(-1) * phi,
            s.evolves(u),
            u.bounds(0.0, 1.0),
            u(*cl) - 0.0,
        ]
    )
    traj = np.asarray(fem.solve())  # must NOT raise: the cap is the answer, not a failure
    assert traj.shape[0] == N
    assert traj.max() <= 1.0 + 1e-9, "the upper bound must hold"
    assert traj.max() > 1.0 - 1e-6, "the bound must actually be ACTIVE, or this tests nothing"


def test_march_guard_uses_the_drivers_own_tolerance():
    """The net may only catch what the driver would have caught eagerly. A driver configured loosely
    must not be second-guessed — its tolerance travels with the spec and through the bounds wrapper."""
    from jno.utils.solver.solver_api import compose_nonlinear_solve_fn

    spec = jno.solve.newton(rtol=1e-3, atol=1e-4)
    assert (spec.traits["rtol"], spec.traits["atol"]) == (1e-3, 1e-4)
    composed = compose_nonlinear_solve_fn(spec, None, None, fem=None)
    assert composed.tolerances == (1e-3, 1e-4)

    stag = jno.solve.staggered([object()], rtol=1e-5, atol=1e-6)
    assert (stag.traits["rtol"], stag.traits["atol"]) == (1e-5, 1e-6)


def test_direct_driver_reaches_the_march():
    """The march runs its per-step solve through `solve_fn`, but never handed it an assembled tangent —
    so `newton(direct=True)` / `staggered(direct=True)` silently fell back to the matrix-free inner
    solve inside a `tau=` path. The answer must be unchanged, and the tangent must actually arrive."""
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    N = 4
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34).domain(tau=(0.0, 1.0, N))
    d.tag("left", lambda x, y: x < 1e-9)
    co, cl = (d.variable(r, split=True) for r in ("interior", "left"))
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols()
    fem = jno.fem(
        [
            inner(grad(u, X), grad(phi, X), 1) + 0.4 * u * u * phi - 3.0 * co[-1] * phi + 0.0 * s.i(-1) * phi,
            s.evolves(u),
            u(*cl) - 0.0,
        ]
    )
    ref = np.asarray(fem.solve())
    got = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True)))
    assert np.abs(ref).max() > 1e-3, "the march is trivial — the comparison would be vacuous"
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-7, "the direct march moved the answer"

    # ...and the tangent genuinely arrives: a driver that DEMANDS one must not be handed None.
    seen = {"jac": False}

    def spy(residual_fn, u0, *, jacobian=None):
        seen["jac"] = seen["jac"] or (jacobian is not None)
        from jno.utils.solver.newton_krylov import newton_krylov

        return newton_krylov(residual_fn, u0)

    spy.wants_jacobian = True
    from jno.utils.solver.history_march import run_history_march

    run_history_march(fem, spy)
    assert seen["jac"], "the march never threaded the assembled tangent to a wants_jacobian driver"


def test_direct_staggered_marches_a_coupled_load_path():
    """The combination this was built for: a coupled march whose displacement block is solved by a
    factorization instead of unpreconditioned Krylov — including through the `bounds` min-map."""
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    N = 4
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34).domain(tau=(0.0, 1.0, N))
    d.tag("left", lambda x, y: x < 1e-9)
    co, cl = (d.variable(r, split=True) for r in ("interior", "left"))
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    dm, q = d.fem_symbols()
    hs, _ = d.fem_symbols()
    psi = 0.5 * inner(grad(u, X), grad(u, X), 1)
    hist = maximum(psi, hs.i(-1))
    fem = jno.fem(
        [
            ((1.0 - dm) ** 2 + 1e-4) * inner(grad(u, X), grad(phi, X), 1) - 2.0 * co[-1] * phi,
            2.5 * dm * q + 0.4 * inner(grad(dm, X), grad(q, X), 1) - 2.0 * (1.0 - dm) * hist * q,
            hs.evolves(hist),
            dm.bounds(0.0, 1.0),
            u(*cl) - 0.0,
        ]
    )
    kw = dict(max_sweeps=400, rtol=1e-7, atol=1e-9)
    mf = np.asarray(fem.solve(nonlinear=jno.solve.staggered([u, dm], **kw)))
    dr = np.asarray(fem.solve(nonlinear=jno.solve.staggered([u, dm], direct=True, **kw)))
    bd = fem.block_index(dm)
    assert np.abs(mf).max() > 1e-3
    assert mf[..., fem.blocks[bd]].max() > 1e-2, "no damage — the coupling did nothing"
    assert dr[..., fem.blocks[bd]].max() <= 1.0 + 1e-9, "the bound must hold on the direct route too"
    assert np.abs(dr - mf).max() / np.abs(mf).max() < 1e-6, "direct and matrix-free marches disagree"


def test_adapt_on_a_march_fails_loud():
    """`adapt=` is dispatched before the march, so it used to return a single STEADY solve — shape
    (n_dofs,) where the caller asked for (n_steps, n_dofs) — and drop a `tau=` alongside it in silence.
    Remeshing cannot compose with the march because the per-quadrature-point state would have to be
    transferred onto each new mesh; that is wired for the transient stepper, not for `tau=`."""
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(tau=(0.0, 1.0, 4))
    d.tag("left", lambda x, y: x < 1e-9)
    co, cl = d.variable("interior", split=True), d.variable("left", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols()
    fem = jno.fem(
        [
            inner(grad(u, X), grad(phi, X), 1) + 0.4 * u * u * phi - 3.0 * co[-1] * phi + 0.0 * s.i(-1) * phi,
            s.evolves(u),
            u(*cl) - 0.0,
        ]
    )
    assert np.asarray(fem.solve()).shape[0] == 4, "the plain march must return one frame per load step"
    for kw in ({}, {"tau": jno.solve.adaptive(limit=0.5)}):
        with pytest.raises(NotImplementedError, match="LOAD-PATH march"):
            fem.solve(adapt=jno.solve.remesh(max_iters=1), **kw)
