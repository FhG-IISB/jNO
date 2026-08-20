"""Load-path field — ``u.bind(...).freeze_path(frames)`` drives a ``tau=`` march with a precomputed
per-load-step nodal field (one nodal field per load step, e.g. a prior solve's field history), so the
field history *is* the load. Exercised here on a J2 plasticity load-path march driven by a prescribed
scalar field entering as an eigenstrain.

Headline oracle (method of manufactured solution): a load-path field whose frames are a known field
sampled at the nodes must reproduce, to machine precision, the SAME march written with that field as an
**analytic function of (x, τ) in the form** — for a field linear in the coordinates the P1 node→quad
interpolation is exact, so the two are identical. That single check proves the per-step slice is
delivered and interpolated correctly, all the way through the load-step scan.

Plus fail-loud guards: a load-path field on a non-march domain, and a frame-count / grid mismatch.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
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

n = jno.np
E, NU, SY, HRD, BETA = 200.0, 0.3, 2.0, 20.0, 0.5
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
K = LAM + 2 * MU / 3
RT = np.sqrt(1.5)


def _aliases():
    return n.sym, n.grad, n.trace, n.inner, n.sqrt, n.maximum, n.identity


def _j2_stress(u, X, theta, ep_hist):
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    I3 = identity(3)
    eps = lambda w: sym(grad(w, X))
    dev = lambda A: A - trace(A) / 3 * I3
    nrm = lambda A: sqrt(maximum(inner(A, A, 2), 0) + 1e-30)
    ee = eps(u) - BETA * theta * I3 - ep_hist
    D = dev(ee)
    dd = nrm(D)
    dg = maximum(RT * 2 * MU * dd - SY, 0) / (3 * MU + HRD)
    nd = D / dd
    sig = K * trace(ee) * I3 + 2 * MU * D - 2 * MU * RT * dg * nd
    return sig, dg, nd


def _march(theta_of_Xtau, nstep):
    """Build & solve a clamped-base J2 plasticity march driven by ``theta_of_Xtau(X, tau)``."""
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain(tau=(0.0, 1.0, nstep))
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co = d.variable("interior", split=True)
    cb = d.variable("bot", split=True)
    X, tau = [co[0], co[1], co[2]], co[-1]
    u, phi = d.fem_symbols(value_shape=(3,))
    ep, _ = d.fem_symbols(value_shape=(3, 3))
    eps = lambda w: sym(grad(w, X))
    sig, dg, nd = _j2_stress(u, X, theta_of_Xtau(X, tau), ep.i(-1))
    fem = jno.fem(
        [
            inner(sig, eps(phi), 2),
            ep.evolves(ep.i(-1) + RT * dg * nd),
            *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)],
        ]
    )
    return np.asarray(fem.solve()), d


def test_load_path_field_matches_analytic_in_tau():
    nstep = 11
    tau_pts = np.linspace(0.0, 1.0, nstep)

    # analytic: theta(x, tau) = tau * z  (linear in z -> P1 interpolation is exact)
    traj_analytic, d = _march(lambda X, tau: tau * X[2], nstep)

    # same field as a load-path field: frames[k] = tau_k * z_nodes (nodal samples of the analytic field)
    z_nodes = np.asarray(d.mesh.points)[:, 2]
    frames = np.stack([t * z_nodes for t in tau_pts])  # (nstep, n_nodes)

    def theta_path(X, tau):
        T = d.fem_symbols(names=("Tpath", "w"))[0]
        return T.bind(x=X[0], y=X[1], z=X[2]).freeze_path(frames)

    traj_path, _ = _march(theta_path, nstep)

    assert traj_path.shape == traj_analytic.shape
    err = np.abs(traj_path - traj_analytic).max()
    assert err < 1e-9, f"load-path field vs analytic-in-tau march differ by {err:.2e} (should be ~0)"
    # and it actually deformed (non-trivial oracle)
    assert np.abs(traj_analytic).max() > 1e-4


def test_load_path_field_on_plain_domain_fails_loud():
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain()  # NO tau grid, no history
    co = d.variable("interior", split=True)
    cb_all = d.variable("boundary", split=True)
    X = [co[0], co[1], co[2]]
    u, phi = d.fem_symbols(value_shape=(3,))
    T = d.fem_symbols(names=("Tp", "w"))[0]
    frames = np.zeros((3, np.asarray(d.mesh.points).shape[0]))
    theta = T.bind(x=X[0], y=X[1], z=X[2]).freeze_path(frames)
    eps = lambda w: sym(grad(w, X))
    weak = inner(LAM * trace(eps(u)) * identity(3) + 2 * MU * eps(u) - BETA * theta * identity(3), eps(phi), 2)
    with pytest.raises(ValueError, match="load-step march|freeze_path"):
        jno.fem([weak, u(cb_all[0], cb_all[1], cb_all[2]) - (0.0, 0.0, 0.0)])


def test_frame_count_mismatch_fails_loud():
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain(tau=(0.0, 1.0, 5))
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co = d.variable("interior", split=True)
    cb = d.variable("bot", split=True)
    X = [co[0], co[1], co[2]]
    u, phi = d.fem_symbols(value_shape=(3,))
    ep, _ = d.fem_symbols(value_shape=(3, 3))
    T = d.fem_symbols(names=("Tp", "w"))[0]
    frames = np.zeros((3, np.asarray(d.mesh.points).shape[0]))  # 3 frames != 5 load steps
    theta = T.bind(x=X[0], y=X[1], z=X[2]).freeze_path(frames)
    sym, grad, trace, inner, sqrt, maximum, identity = _aliases()
    eps = lambda w: sym(grad(w, X))
    sig, dg, nd = _j2_stress(u, X, theta, ep.i(-1))
    fem = jno.fem(
        [inner(sig, eps(phi), 2), ep.evolves(ep.i(-1) + RT * dg * nd), *[u(cb[0], cb[1], cb[2])[i] - 0.0 for i in range(3)]]
    )
    with pytest.raises(ValueError, match="frames|load step"):
        np.asarray(fem.solve())


def test_vector_load_path_field_matches_analytic_in_tau():
    """A VECTOR load-path field (``frames`` of shape ``(nstep, n_nodes, 3)``) delivers its per-step
    vec-slice through the same march. Driving the eigenstrain scalar ``theta = gx + gy + gz`` from a linear
    vector field reproduces the SAME march written with that field analytically-in-tau, to machine
    precision (P1 node->quad interpolation is exact for a linear field) -- proving all THREE components are
    delivered and interpolated correctly across the load-step scan."""
    nstep = 11
    tau_pts = np.linspace(0.0, 1.0, nstep)
    vec3, inner = jno.np.vector, jno.np.inner
    ones = vec3(1.0, 1.0, 1.0)  # theta = g . (1,1,1) = gx+gy+gz, contracted without component indexing

    def theta_analytic(X, tau):  # theta from a linear vector field g(X, tau)
        return inner(vec3(tau * X[2], 0.5 * tau * X[0], (1.0 / 3.0) * tau * X[1]), ones)

    traj_analytic, d = _march(theta_analytic, nstep)

    # the same vector field as a load-path field: frames[k] = tau_k * (z, x/2, y/3) sampled at the nodes
    nodes = np.asarray(d.mesh.points)[:, :3]
    base = np.stack([nodes[:, 2], 0.5 * nodes[:, 0], (1.0 / 3.0) * nodes[:, 1]], axis=1)  # (n_nodes, 3)
    frames = np.stack([t * base for t in tau_pts])  # (nstep, n_nodes, 3)

    def theta_path(X, tau):
        G = d.fem_symbols(value_shape=(3,), names=("Gpath", "w"))[0]
        g = G.bind(x=X[0], y=X[1], z=X[2]).freeze_path(frames)  # (nstep, n_nodes, 3) vector load-path field
        return inner(g, ones)

    traj_path, _ = _march(theta_path, nstep)

    assert traj_path.shape == traj_analytic.shape
    err = np.abs(traj_path - traj_analytic).max()
    assert err < 1e-9, f"vector load-path field vs analytic-in-tau march differ by {err:.2e} (should be ~0)"
    assert np.abs(traj_analytic).max() > 1e-4  # non-trivial deformation (a real oracle)
