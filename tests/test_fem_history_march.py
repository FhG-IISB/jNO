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
