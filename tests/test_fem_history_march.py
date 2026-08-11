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
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

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
